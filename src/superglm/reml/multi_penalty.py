"""Finite penalty geometry on fixed, balanced component support.

Wood (2011), section 3.1 and Appendix B, gives the separation principle and
trace derivatives. This independent root implementation corrects its
preconditioner against frozen component roots and bounds the fresh actions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.reml._compensated import _dot2_selected, _dot2_value
from superglm.reml.penalty_support import (
    PenaltyNumericalError,
    _finite_double,
    _penalty_support,
    _penalty_support_from_roots,
    _PenaltySupport,
    _readonly,
)
from superglm.solvers.rank import SHARED_RANK_POLICY

_EPS = np.finfo(float).eps
_UNIT_ROUNDOFF = _EPS / 2
_SMALLEST_SUBNORMAL = np.nextafter(0.0, 1.0)
# A work threshold, not an accuracy threshold: tiny reductions reuse the
# scalar recurrence without loading Numba's native runtime on a small fit.
_DOT2_NATIVE_MIN_WORK = 256


@dataclass(frozen=True)
class _PenaltyCertificate:
    whitening_error: float
    duality_error: float
    logdet_error: float
    gradient_error: NDArray
    hessian_error: NDArray
    root_error: NDArray
    inverse_error: NDArray
    resolution_limited: bool


@dataclass(frozen=True)
class _PenaltySummaryCertificate:
    whitening_error: float
    duality_error: float
    logdet_error: float
    gradient_error: NDArray
    hessian_error: NDArray
    resolution_limited: bool


@dataclass(frozen=True)
class _PenaltySummary:
    """Admitted determinant and derivatives, without root or inverse outputs."""

    logdet_s_plus: float
    rank: int
    gradient: NDArray
    hessian: NDArray
    _support: _PenaltySupport
    _certificate: _PenaltySummaryCertificate
    _input_lambdas: NDArray
    _correction_count: int


@dataclass(frozen=True)
class _ProductEvidence:
    operands: tuple[NDArray, NDArray]
    product: NDArray
    magnitude: NDArray


@dataclass(frozen=True)
class _DualityEvidence:
    operands: tuple[NDArray, NDArray]
    product: NDArray
    error: NDArray


@dataclass(frozen=True)
class _BasisGramEvidence:
    owner: int
    basis: NDArray
    snapshot: NDArray
    arithmetic: tuple
    gram: NDArray
    error: NDArray


def _evidence_copy(value: NDArray) -> NDArray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _same_operands(operands: tuple[NDArray, NDArray], left: NDArray, right: NDArray) -> bool:
    return np.array_equal(operands[0], left) and np.array_equal(operands[1], right)


@dataclass
class SimilarityTransformResult:
    """One selected PSD penalty's determinant, root and inverse geometry."""

    logdet_s_plus: float
    S_pinv_plus: NDArray
    Q_plus: NDArray
    Q_zero: NDArray
    E_sqrt: NDArray
    rank: int
    _certificate: _PenaltyCertificate | None = None
    _component_factors: tuple[NDArray, ...] | None = None
    _support: _PenaltySupport | None = None
    _input_matrices: tuple[NDArray, ...] | None = None
    _input_lambdas: NDArray | None = None
    _correction_count: int = 0
    _gradient: NDArray | None = None
    _hessian: NDArray | None = None

    @property
    def Q_full(self) -> NDArray:
        return np.hstack([self.Q_plus, self.Q_zero])


def _gamma(count: int, unit: float = _EPS / 2) -> float:
    product = np.float64(count) * np.float64(unit)
    if product >= 1:
        raise PenaltyNumericalError("arithmetic error bound is unresolved")
    bound = product / (1 - product) / (1 - 3 * _UNIT_ROUNDOFF)
    return float(np.nextafter(float(bound), np.inf))


def _upper(value: NDArray | float) -> NDArray:
    with np.errstate(over="ignore"):
        result = np.asarray(np.maximum(value, 0), dtype=np.float64)
        result = np.nextafter(result, np.inf)
    return _finite_double(result, "arithmetic error bound")


def _norm_upper(value: NDArray) -> float:
    absolute = np.abs(np.asarray(value, dtype=np.float64))
    maximum = np.max(absolute, initial=0.0)
    if maximum == 0:
        return 0.0
    norm = maximum * np.sqrt(np.sum((absolute / maximum) ** 2, dtype=np.float64))
    return float(_upper(norm / (1 - _gamma(3 * absolute.size + 2, _UNIT_ROUNDOFF))))


def _positive_product(left: NDArray, right: NDArray) -> NDArray:
    """Upper bound for a nonnegative dot, including its own rounding.

    Native GEMM's componentwise gamma bound applies in the normal range.
    The remaining exponents use explicit gradual-underflow allowances.
    """
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    count = left.shape[-1]
    if left.ndim == right.ndim == 2 and left.shape[1] == right.shape[0]:
        maximum_left = np.max(left, initial=0.0)
        maximum_right = np.max(right, initial=0.0)
        if (
            np.isfinite(maximum_left)
            and np.isfinite(maximum_right)
            and np.min(left, initial=0.0) >= 0
            and np.min(right, initial=0.0) >= 0
        ):
            shape = (left.shape[0], right.shape[1])
            if maximum_left == 0 or maximum_right == 0:
                return np.zeros(shape)
            maximum_exponent = (
                math.frexp(float(maximum_left))[1]
                + math.frexp(float(maximum_right))[1]
                + (count - 1).bit_length()
            )
            # Every exact dot is strictly below 2**maximum_exponent.
            # Exponents avoid forming a product below the subnormal range.
            if maximum_exponent <= np.finfo(float).minexp - np.finfo(float).nmant:
                return np.full(shape, np.nextafter(0.0, 1.0))
            # Both operands are already finite, nonnegative binary64.
            # Reuse their maxima instead of rescanning/casting them.
            a_nonzero, b_nonzero = left[left > 0], right[right > 0]
            minimum_left, minimum_right = np.min(a_nonzero), np.min(b_nonzero)
            if minimum_left >= np.finfo(float).tiny and minimum_right >= np.finfo(float).tiny:
                minimum_exponent = (
                    math.frexp(float(minimum_left))[1] + math.frexp(float(minimum_right))[1]
                )
                # frexp mantissas are in [1/2, 1). These integer tests keep
                # every nonzero product normal and the positive sum finite.
                if minimum_exponent >= -1020 and maximum_exponent <= 1021:
                    return _positive_native_product(left, right)
    with np.errstate(over="ignore", invalid="ignore"):
        value = left @ right
    value = (value + (2 * count + 1) * _SMALLEST_SUBNORMAL) / (
        1 - _gamma(2 * count + 1, _UNIT_ROUNDOFF)
    )
    return _upper(value)


def _native_product(left: NDArray, right: NDArray) -> NDArray:
    """Native binary64 product; the enclosing caller supplies the error bound."""
    return np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)


def _matmul_enclosed(
    left: NDArray, right: NDArray, *, _evidence: list[_ProductEvidence] | None = None
) -> tuple[NDArray, NDArray]:
    left_value, right_value = (
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )
    magnitude = _positive_product(np.abs(left_value), np.abs(right_value))
    product_value = _native_product(left_value, right_value)
    result = _finite_double(product_value, "penalty factor product")
    # A length-k dot has at most k rounding factors per term, even though
    # it executes k products and k-1 additions. The positive magnitude bound
    # also proves that its partial sums cannot overflow. Gradual underflow
    # adds at most one minimum subnormal per executed operation.
    count = left.shape[-1]
    error = _gamma(count) * magnitude + (2 * count + 1) * _SMALLEST_SUBNORMAL
    error = _upper(error / (1 - _gamma(3)))
    if _evidence is not None:
        _evidence.append(
            _ProductEvidence(
                (_evidence_copy(left_value), _evidence_copy(right_value)),
                _evidence_copy(product_value),
                _evidence_copy(magnitude),
            )
        )
    return result, error


def _root_error_product(left: NDArray, right: NDArray) -> NDArray:
    """Reuse identical tiny error rows within the enclosed positive product."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if (
        left.ndim == right.ndim == 2
        and left.shape[0] > 1
        and left.shape[1] == right.shape[0]
        and np.all(np.isfinite(left))
        and np.all(left >= 0)
        and np.all(left <= np.finfo(float).tiny / 2)
        and np.all(left == left[:1])
    ):
        # These operands cannot pass the normal-range gate. Reusing one row
        # retains the inner dimension and every gradual-underflow allowance.
        row = _positive_product(left[:1], right)
        return np.repeat(row, left.shape[0], axis=0)
    return _positive_product(left, right)


def _basis_gram(support: _PenaltySupport, basis: NDArray) -> tuple[NDArray, NDArray]:
    """Memoize the unchanged support Gram, retaining its original enclosure."""
    if basis is not support.Q_plus:
        return _matmul_enclosed(basis.T, basis)
    arithmetic = (
        _EPS,
        _UNIT_ROUNDOFF,
        _SMALLEST_SUBNORMAL,
        _matmul_enclosed,
        _native_product,
        _positive_product,
        _gamma,
        _upper,
        _finite_double,
    )
    cached = support._basis_gram_evidence
    if (
        isinstance(cached, _BasisGramEvidence)
        and cached.owner == id(support)
        and cached.basis is basis
        and cached.arithmetic == arithmetic
        and np.array_equal(cached.snapshot, basis)
    ):
        return cached.gram, cached.error
    gram, error = _matmul_enclosed(basis.T, basis)
    evidence = _BasisGramEvidence(
        id(support),
        basis,
        _evidence_copy(basis),
        arithmetic,
        _evidence_copy(gram),
        _evidence_copy(error),
    )
    object.__setattr__(support, "_basis_gram_evidence", evidence)
    return evidence.gram, evidence.error


def _inverse_gram_enclosed(inverse_root: NDArray, eta: float) -> tuple[NDArray, NDArray]:
    """Materialize the inverse after admission using a bounded native Gram.

    Native GEMM has a componentwise gamma bound for the stored float64 J.
    The spectral metric uncertainty couples arbitrary rows of J, so its
    entrywise bound uses the outer product of upper bounds on their norms.
    """
    J = np.asarray(inverse_root, dtype=np.float64)
    rank = J.shape[1]
    nonzero = np.abs(J[J != 0])
    normal = np.all(np.isfinite(J)) and (
        not nonzero.size
        or (
            np.min(nonzero) >= np.nextafter(np.sqrt(np.finfo(float).tiny), np.inf)
            and np.max(nonzero) <= np.sqrt(np.finfo(float).max / (4 * max(rank, 1)))
        )
    )
    if not normal:
        inverse, product_rounding = _matmul_enclosed(J, J.T)
        rounding = product_rounding
        row_squares = _upper(np.diag(inverse) + np.diag(rounding))
    else:
        magnitude = _positive_native_product(np.abs(J), np.abs(J.T))
        inverse = J @ J.T
        gamma = _gamma(2 * rank + 1)
        rounding = gamma * magnitude
        # Signed partial sums may underflow despite the normal-product gate.
        rounding += (2 * rank + 1) * _SMALLEST_SUBNORMAL / (1 - gamma)
        row_squares = _upper(np.diag(magnitude))
    row_norms = _upper(np.sqrt(row_squares) / (1 - _gamma(4, _UNIT_ROUNDOFF)))
    row_products = _positive_product(row_norms[:, None], row_norms[None, :])
    error = rounding + np.float64(eta) / (1 - np.float64(eta)) * row_products
    return inverse, _upper((error + 32 * _SMALLEST_SUBNORMAL) / (1 - _gamma(32, _UNIT_ROUNDOFF)))


def _trace_bound(value: NDArray, error: NDArray) -> float:
    diagonal = np.diag(value)
    trace = np.sum(diagonal, dtype=np.float64)
    bound = np.sum(np.diag(error), dtype=np.float64)
    bound += _gamma(2 * len(diagonal) + 1, _UNIT_ROUNDOFF) * np.sum(np.abs(diagonal))
    return float(_upper((abs(trace) + bound) / (1 - _gamma(4 * len(diagonal) + 4, _UNIT_ROUNDOFF))))


def _logdet_defect_bound(product: NDArray, error: NDArray) -> float:
    """Trace-series bound, retaining cancellation-free second-order terms."""
    defect = product - np.eye(len(product))
    error = _upper((error + _gamma(1) * np.abs(defect) + _SMALLEST_SUBNORMAL) / (1 - _gamma(3)))
    radius = float(_upper(_norm_upper(defect) + _norm_upper(error)))
    if radius >= 1:
        return np.inf
    # tr(log(I+D)) = tr(D) + remainder. For k >= 2,
    # |tr(D**k)| <= ||D||_F**2 * ||D||_2**(k-2).
    return float(
        _upper((_trace_bound(defect, error) + radius**2 / (2 * (1 - radius))) / (1 - _gamma(6)))
    )


def _materialization_logdet_bound(
    left: NDArray,
    right: NDArray,
    product: NDArray,
    inverse: NDArray,
    input_bound: NDArray | None = None,
    *,
    _product_evidence: _ProductEvidence | None = None,
    _duality_evidence: list[_DualityEvidence] | None = None,
) -> float:
    """Enclose the determinant effect of an observed, signed product residual."""
    if _product_evidence is not None and _same_operands(_product_evidence.operands, left, right):
        product_value = _product_evidence.product
        magnitude = _product_evidence.magnitude
    else:
        product_value = np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)
        magnitude = _positive_product(np.abs(left), np.abs(right))
    residual = product_value - product
    uncertain = _gamma(left.shape[-1]) * magnitude
    uncertain += (2 * left.shape[-1] + 1) * _SMALLEST_SUBNORMAL
    uncertain += _gamma(1) * np.abs(residual) + _SMALLEST_SUBNORMAL
    if input_bound is not None:
        uncertain += input_bound
    uncertain = _upper(uncertain / (1 - _gamma(6)))
    action, error = _matmul_enclosed(residual, inverse)
    error = _upper(error + _positive_product(_upper(uncertain), np.abs(inverse)))
    dual, dual_error = _matmul_enclosed(product, inverse)
    if _duality_evidence is not None:
        _duality_evidence.append(
            _DualityEvidence(
                (_evidence_copy(product), _evidence_copy(inverse)),
                _evidence_copy(dual),
                _evidence_copy(dual_error),
            )
        )
    defect = dual - np.eye(len(dual))
    defect_error = _upper(
        (dual_error + _gamma(1) * np.abs(defect) + _SMALLEST_SUBNORMAL) / (1 - _gamma(3))
    )
    duality = float(_upper(_norm_upper(defect) + _norm_upper(defect_error)))
    norm = float(_upper(_norm_upper(action) + _norm_upper(error)))
    if duality >= 1 or norm >= 1 - duality:
        return np.inf
    relative = float(_upper(norm / (1 - duality) / (1 - _gamma(2))))
    if relative >= 1:
        return np.inf
    return float(
        _upper(
            (
                _trace_bound(action, error)
                + norm * duality / (1 - duality)
                + relative**2 / (2 * (1 - relative))
                + 8 * _SMALLEST_SUBNORMAL
            )
            / (1 - _gamma(8))
        )
    )


def _weights(lambdas: NDArray, count: int) -> NDArray:
    source = np.asarray(lambdas)
    if np.iscomplexobj(source):
        raise ValueError("smoothing parameters must be finite and non-negative")
    values = np.asarray(source, dtype=np.float64)
    if values.shape != (count,) or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            "smoothing parameters must be finite and non-negative with one per component"
        )
    return values


def _separation_parameter(eps_rank: float | None) -> float:
    value = _EPS ** (1 / 3) if eps_rank is None else float(eps_rank)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("eps_rank must be finite and strictly between zero and one")
    return value


def similarity_transform_logdet(
    penalty_matrices: list[NDArray],
    lambdas: NDArray,
    eps_rank: float | None = None,
) -> SimilarityTransformResult:
    """Evaluate a finite PSD sum without a positive-weight rank decision.

    eps_rank controls separation only. Exact-zero weights are inactive.
    Numerical inability to retain support or materialize a required output
    raises PenaltyNumericalError rather than reducing rank.
    """
    if not penalty_matrices:
        raise ValueError("at least one penalty matrix is required")
    values = _weights(lambdas, len(penalty_matrices))
    support = _penalty_support(penalty_matrices)
    result = _evaluate_penalty_support(support, values, _separation_parameter(eps_rank))
    result._input_matrices = tuple(_readonly(matrix) for matrix in penalty_matrices)
    return result


def _reference_root_actions(
    roots: Sequence[NDArray],
    lambdas: NDArray,
    inverse_root: NDArray,
    *,
    _refine: bool = True,
) -> tuple[tuple[NDArray, ...], tuple[NDArray, ...]]:
    """Fresh reference actions with componentwise arithmetic bounds.

    Every product and its bound use binary64, with selective scalar Dot2
    refinement when the native product cannot meet the admission budget.
    """
    values = _weights(lambdas, len(roots))
    J = np.asarray(inverse_root, dtype=np.float64)
    actions, bounds = [], []
    rows, rank = sum(len(root) for root in roots), J.shape[1]
    budget = _gamma(8 * (rows + J.shape[0] + rank + len(roots) + 1))
    dot_budget = budget / (8 * max(rank, 1) * math.sqrt(max(rows, 1)))
    for root, weight in zip(roots, values, strict=True):
        H = np.sqrt(weight) * np.asarray(root, dtype=np.float64)
        magnitude = _positive_product(np.abs(H), np.abs(J))
        product_value = _native_product(H, J)
        action = _finite_double(product_value, "reference root action")
        error = _gamma(2 * root.shape[1] + 4, _UNIT_ROUNDOFF) * magnitude
        # Scaling a root entry can underflow before its multiplication by J.
        # That absolute error must follow the action through J.
        scaling_underflow = np.where((root != 0) & (weight != 0), _SMALLEST_SUBNORMAL, 0.0)
        error += _positive_product(scaling_underflow, np.abs(J))
        error += (2 * root.shape[1] + 4) * _SMALLEST_SUBNORMAL
        selected = np.argwhere(error > dot_budget) if _refine else np.empty((0, 2), dtype=int)
        if len(selected):
            try:
                # The scalar Dot2 enclosure needs |root| @ |J|, not the
                # rounded weighted H magnitude. Build it once per component.
                # All reuse ends here: a changed root, weight or J recomputes.
                dot_magnitude = _positive_product(np.abs(root), np.abs(J))
                if len(selected) * root.shape[1] >= _DOT2_NATIVE_MIN_WORK:
                    dots, success = _dot2_selected(np.asarray(root, dtype=np.float64), J, selected)
                else:
                    # Keep shared enclosures for tiny batches without paying
                    # native-kernel startup or revalidating every scalar dot.
                    dots = np.array([_python_dot2_value(root[r], J[:, c]) for r, c in selected])
                    success = np.isfinite(dots)
                row, column = selected.T
                unit, inner = _UNIT_ROUNDOFF, root.shape[1]
                dot_error = (
                    unit * np.abs(dots)
                    + np.float64(_gamma(inner)) ** 2 * dot_magnitude[row, column]
                    + 5 * inner * _SMALLEST_SUBNORMAL
                ) / (1 - unit)
                dot_error = _upper(dot_error / (1 - _gamma(12, _UNIT_ROUNDOFF)))
                scale = np.sqrt(weight)
                scale_error = _gamma(1, _UNIT_ROUNDOFF) * abs(scale) + _SMALLEST_SUBNORMAL
                scaled = scale * dots
                refined = _finite_double(scaled, "compensated root action")
                refined_error = (
                    abs(scale) * dot_error
                    + scale_error * (np.abs(dots) + dot_error)
                    + _gamma(1, _UNIT_ROUNDOFF) * np.abs(scaled)
                    + 2 * _SMALLEST_SUBNORMAL
                )
                refined_error = _upper(refined_error / (1 - _gamma(12, _UNIT_ROUNDOFF)))
                use = success & (refined_error < error[row, column])
                action[row[use], column[use]] = refined[use]
                error[row[use], column[use]] = refined_error[use]
                selected = selected[~use]
            except PenaltyNumericalError:
                # A batch magnitude or scaled bound can exceed the range
                # even when individual selected entries remain refinable.
                pass
        for row, column in selected:
            try:
                dot, dot_error = _compensated_dot(root[row], J[:, column])
                scale = np.sqrt(weight)
                scale_error = _gamma(1, _UNIT_ROUNDOFF) * abs(scale) + _SMALLEST_SUBNORMAL
                scaled = scale * np.float64(dot)
                refined = float(_finite_double(scaled, "compensated root action"))
                refined_error = (
                    abs(scale) * dot_error
                    + scale_error * (abs(dot) + dot_error)
                    + _gamma(1, _UNIT_ROUNDOFF) * abs(scaled)
                    + 2 * _SMALLEST_SUBNORMAL
                )
                refined_error = float(_upper(refined_error / (1 - _gamma(12, _UNIT_ROUNDOFF))))
            except PenaltyNumericalError:
                continue
            if refined_error < error[row, column]:
                action[row, column] = refined
                error[row, column] = refined_error
        actions.append(action)
        bounds.append(_upper(error / (1 - _gamma(8))))
    return tuple(actions), tuple(bounds)


def _two_product_error(left: float, right: float, product: float) -> float:
    fma = getattr(math, "fma", None)
    if fma is not None:
        return fma(left, right, -product)
    # Ogita, Rump and Oishi (2005), Algorithm 3.3; Python 3.12 fallback.
    splitter = float(2**27 + 1)
    a, b = splitter * left, splitter * right
    if not math.isfinite(a) or not math.isfinite(b):
        raise PenaltyNumericalError("compensated product requires exponent scaling")
    left_high, right_high = a - (a - left), b - (b - right)
    left_low, right_low = left - left_high, right - right_high
    return left_low * right_low - (
        ((product - left_high * right_high) - left_low * right_high) - left_high * right_low
    )


def _python_dot2_value(x: NDArray, y: NDArray) -> float:
    """The value recurrence shared by scalar and tiny-batch Dot2 refinement."""
    if not len(x):
        return 0.0
    product = float(x[0]) * float(y[0])
    correction = _two_product_error(float(x[0]), float(y[0]), product)
    for a, b in zip(x[1:], y[1:], strict=True):
        a, b = float(a), float(b)
        term = a * b
        term_error = _two_product_error(a, b, term)
        updated = product + term
        recovered = updated - product
        addition_error = (product - (updated - recovered)) + (term - recovered)
        correction += addition_error + term_error
        product = updated
    return product + correction


def _compensated_dot(left: NDArray, right: NDArray) -> tuple[float, float]:
    """Dot2 with its computed-value enclosure, not a rounding heuristic.

    Ogita, Rump and Oishi (2005), Algorithms 3.1/3.3/3.5 and 5.3,
    Proposition 5.5, with gradual-underflow allowances for each operation.
    """
    if not len(left):
        return 0.0, 0.0
    x = _finite_double(left, "compensated operand")
    y = _finite_double(right, "compensated operand")
    magnitude = float(_positive_product(np.abs(x)[None, :], np.abs(y)[:, None])[0, 0])
    value, compiled = _dot2_value(x, y) if len(x) >= _DOT2_NATIVE_MIN_WORK else (0.0, False)
    if not compiled:
        value = _python_dot2_value(x, y)
    if not math.isfinite(value):
        raise PenaltyNumericalError("compensated dot is not representable")
    unit = _UNIT_ROUNDOFF
    count = len(x)
    error = (
        unit * abs(value)
        + np.float64(_gamma(count)) ** 2 * magnitude
        + 5 * count * _SMALLEST_SUBNORMAL
    ) / (1 - unit)
    return value, float(_upper(error / (1 - _gamma(12, _UNIT_ROUNDOFF))))


def _squared_norm_enclosed(value: NDArray, error: NDArray) -> tuple[float, float]:
    """A Dot2 squared norm, plus entrywise input perturbations.

    All callers pass admitted near-isometric factors or their cross products,
    so their entries and squared norms are in the float64 exponent range.
    The positive dot bounds 2*|value|*error + error**2 without squaring tiny
    bounds before their contribution has been enclosed.
    """
    flat, uncertainty = np.ravel(value), np.ravel(error)
    result, arithmetic = _compensated_dot(flat, flat)
    perturbation = _positive_product(
        _upper(2 * np.abs(flat) + uncertainty)[None, :], uncertainty[:, None]
    )[0, 0]
    return result, float(_upper(arithmetic + perturbation))


def _reference_correct_once(
    roots: Sequence[NDArray],
    lambdas: NDArray,
    E: NDArray,
    J: NDArray,
    logdet: float,
    *,
    _evidence: list | None = None,
    _refine: bool = True,
) -> tuple[NDArray, NDArray, float, tuple[NDArray, ...]]:
    actions, _ = _reference_root_actions(roots, lambdas, J, _refine=_refine)
    upper = scipy.linalg.qr(np.vstack(actions), mode="r", check_finite=False)[0][: J.shape[1]]
    if upper.shape != (J.shape[1], J.shape[1]) or np.any(np.diag(upper) == 0):
        raise PenaltyNumericalError("reference QR cannot retain the fixed support")
    product_evidence = [] if _evidence is not None else None
    new_E, root_error = _matmul_enclosed(upper, E, _evidence=product_evidence)
    new_J = _triangular_solve(upper.T, J.T).T
    _finite_double(new_J, "corrected inverse root")
    terms = [2 * math.log(abs(v)) for v in np.diag(upper)]
    new_logdet = math.fsum([logdet, *terms])
    factors = tuple(
        scipy.linalg.solve_triangular(upper.T, action.T, lower=True, check_finite=False).T
        for action in actions
    )
    if _evidence is not None:
        duality_evidence = []
        materialization = _materialization_logdet_bound(
            upper,
            E,
            new_E,
            new_J,
            _product_evidence=product_evidence[0],
            _duality_evidence=duality_evidence,
        )
        _evidence.append(
            (
                root_error,
                math.fsum(map(abs, terms)),
                _gamma(4 * len(terms) + 2) * (abs(logdet) + math.fsum(map(abs, terms))),
                materialization,
                duality_evidence[0],
            )
        )
    return new_E, new_J, new_logdet, factors


def _triangular_solve(matrix: NDArray, rhs: NDArray) -> NDArray:
    """Substitution with a scaled RHS, without a reciprocal coordinate basis."""
    A = np.asarray(matrix, dtype=np.float64)
    result = np.array(rhs, dtype=np.float64, copy=True)
    for i in range(len(A)):
        if A[i, i] == 0:
            raise PenaltyNumericalError("singular coordinate triangular factor")
        result[i] = (result[i] - A[i, :i] @ result[:i]) / A[i, i]
    if not np.all(np.isfinite(result)):
        raise PenaltyNumericalError("coordinate inverse action is not representable")
    return result


def _candidate_product(left: NDArray, right: NDArray) -> NDArray | None:
    """Choose a coordinate proposal; reference certificates remain separate."""
    for operand in (left, right):
        absolute = np.abs(operand)
        if not np.all(np.isfinite(absolute)) or np.any(
            (absolute != 0) & ((absolute < 2.0**-128) | (absolute > 2.0**128))
        ):
            return None
    # Normal binary64 operands/products and any addressable inner dimension
    # fit safely in this envelope. Accuracy here only changes the proposed C.
    return np.asarray(left, dtype=float) @ np.asarray(right, dtype=float)


def _direct_candidate(
    support: _PenaltySupport,
    values: NDArray,
    separation: float,
    *,
    _native_candidate: bool = False,
) -> tuple[NDArray, NDArray, float, float, float] | None:
    """Native full-rank QR after safe column scaling, when well conditioned."""
    rank, width = support.rank, support.Q_plus.shape[0]
    basis = np.eye(width) if rank == width else support.Q_plus
    stack = np.vstack(
        [np.sqrt(value) * root for root, value in zip(support.component_roots, values, strict=True)]
    )
    weighted = _candidate_product(stack, basis) if _native_candidate else None
    if weighted is None:
        # Packed copy: Q_plus is a strided slice and matmul rounding depends on layout.
        weighted = _native_product(stack, basis.copy(order="K"))
    maxima = np.max(np.abs(weighted), axis=0)
    if np.any(maxima == 0):
        return None
    scales = maxima * np.sqrt(np.sum((weighted / maxima) ** 2, axis=0))
    normalized = _finite_double(weighted / scales, "scaled reference roots")
    upper = scipy.linalg.qr(normalized, mode="r", check_finite=False)[0][:rank]
    if np.any(np.diag(upper) == 0):
        return None
    inverse = scipy.linalg.solve_triangular(upper, np.eye(rank), check_finite=False)
    condition = np.linalg.norm(upper, ord=np.inf) * np.linalg.norm(inverse, ord=np.inf)
    if not np.isfinite(condition) or condition * min(separation, _EPS * rank) >= 1:
        return None
    compact = upper * scales
    product_evidence = []
    E, E_error = _matmul_enclosed(compact, basis.T, _evidence=product_evidence)
    magnitude = product_evidence[0].magnitude
    E_error = _upper(E_error + _gamma(1, _UNIT_ROUNDOFF) * magnitude)
    inverse = _triangular_solve(upper[::-1, ::-1], np.eye(rank)[::-1])[::-1]
    J = basis @ (inverse / scales[:, None])
    _finite_double(J, "candidate inverse root")
    terms = [
        *(2 * float(np.log(v)) for v in scales),
        *(2 * math.log(abs(v)) for v in np.diag(upper)),
    ]
    volume_scale = rank + math.fsum(map(abs, terms))
    gram, gram_error = _basis_gram(support, basis)
    # det((C B.T)(C B.T).T) = det(C)**2 det(B.T B): one basis-volume charge.
    basis_log_error = _logdet_defect_bound(gram, gram_error)
    materialization = _materialization_logdet_bound(
        compact,
        basis.T,
        E,
        J,
        _gamma(1, _UNIT_ROUNDOFF) * magnitude,
        _product_evidence=product_evidence[0],
    )
    if not np.isfinite(basis_log_error) or not np.isfinite(materialization):
        return None
    log_error = _gamma(4 * len(terms) + 4) * volume_scale + basis_log_error + 2 * materialization
    return E, J, math.fsum(terms), log_error, volume_scale


def _separated_candidate(
    support: _PenaltySupport,
    values: NDArray,
    separation: float,
    *,
    _native_candidate: bool = False,
) -> tuple[NDArray, NDArray, float, float, float]:
    """Independent root rows precondition the unchanged reference sum."""
    direct = (
        _direct_candidate(support, values, separation, _native_candidate=True)
        if _native_candidate
        else _direct_candidate(support, values, separation)
    )
    if direct is not None:
        return direct
    if _native_candidate:
        raise PenaltyNumericalError("native proposal cannot span the fixed penalty support")
    rank = support.rank
    candidates = []
    for owner, coordinates in enumerate(support.balanced_coordinates):
        if values[owner] == 0:
            continue
        log_scale = math.log(values[owner]) / 2 + support.root_log_scales[owner]
        for row in coordinates:
            norm = float(np.linalg.norm(row))
            if norm:
                candidates.append((log_scale + math.log(norm), log_scale, row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen, logs = [], []
    rotation = np.zeros((rank, rank))
    for threshold in (
        max(separation, SHARED_RANK_POLICY.factor_rcond),
        SHARED_RANK_POLICY.factor_rcond,
    ):
        chosen, logs = [], []
        for _, log_scale, row in candidates:
            residual = row.copy()
            previous = rotation[:, : len(chosen)]
            for _ in range(2):
                residual -= previous @ (previous.T @ residual)
            norm = float(np.linalg.norm(residual))
            if norm <= threshold * float(np.linalg.norm(row)):
                continue
            rotation[:, len(chosen)] = residual / norm
            chosen.append(row)
            logs.append(log_scale)
            if len(chosen) == rank:
                break
        if len(chosen) == rank:
            break
    if len(chosen) != rank:
        raise PenaltyNumericalError("candidate cannot span the fixed penalty support")
    coordinates = np.asarray(chosen, dtype=np.float64) @ rotation
    # These are candidate zeros only. Omitted components and entries remain
    # present in every reference correction and fresh certificate.
    coordinates[np.triu_indices(rank, 1)] = 0
    weighted = np.exp(np.asarray(logs, dtype=np.float64))[:, None] * coordinates
    maxima = np.max(np.abs(weighted), axis=0)
    scales = maxima * np.sqrt(np.sum((weighted / maxima) ** 2, axis=0))
    normalized = _finite_double(weighted / scales, "scaled candidate roots")
    upper = scipy.linalg.qr(normalized, mode="r", check_finite=False)[0][:rank]
    if np.any(np.diag(upper) == 0):
        raise PenaltyNumericalError("zero candidate pivot on fixed penalty support")
    compact = (upper * scales) @ rotation.T @ support.coordinate_triangular.T
    E, E_error = _matmul_enclosed(compact, support.Q_plus.T)
    rhs = _triangular_solve(upper[::-1, ::-1], np.eye(rank)[::-1])[::-1]
    rhs = rotation @ (rhs / scales[:, None])
    solved = _triangular_solve(support.coordinate_triangular.T, rhs)
    J = support.Q_plus @ solved
    _finite_double(J, "candidate inverse root")
    terms = [
        *(2 * math.log(abs(v)) for v in np.diag(support.coordinate_triangular)),
        *(2 * float(np.log(v)) for v in scales),
        *(2 * math.log(abs(v)) for v in np.diag(upper)),
    ]
    volume_scale = rank + math.fsum(map(abs, terms))
    logdet = math.fsum(terms)
    sign, volume = np.linalg.slogdet(support.Q_plus.T @ support.Q_plus)
    if sign <= 0:
        raise PenaltyNumericalError("invalid support coordinate volume")
    logdet += volume
    product_error = _norm_upper(_positive_product(E_error, np.abs(J)))
    log_error = _gamma(4 * len(terms) + 4) * volume_scale
    log_error += 2 * rank * product_error + _gamma(4 * rank + 4) * rank
    return E, J, logdet, log_error, volume_scale


def _whitening_certificate(
    actions: Sequence[NDArray],
    action_bounds: Sequence[NDArray],
    E: NDArray,
    J: NDArray,
    *,
    _duality: float | None = None,
    _evidence: list | None = None,
) -> tuple[float, float]:
    Z, B = np.vstack(actions), np.vstack(action_bounds)
    gram, multiplication_bound = _matmul_enclosed(Z.T, Z)
    error = multiplication_bound
    error += _positive_product(np.abs(Z.T), B)
    error += _positive_product(B.T, np.abs(Z))
    error += _positive_product(B.T, B)
    error = _upper(error / (1 - _gamma(4)))
    if _evidence is not None:
        _evidence.append((gram, error))
    defect = gram - np.eye(J.shape[1])
    defect_error = _upper(
        (error + _gamma(1) * np.abs(defect) + _SMALLEST_SUBNORMAL) / (1 - _gamma(3))
    )
    eta = _norm_upper(defect) + _norm_upper(defect_error)
    if _duality is None:
        product, product_bound = _matmul_enclosed(E, J)
        duality = _norm_upper(product - np.eye(J.shape[1]))
        duality += _norm_upper(product_bound)
        _duality = float(np.nextafter(duality, np.inf))
    return float(np.nextafter(eta, np.inf)), _duality


def _cross_value(left: NDArray, right: NDArray) -> tuple[float, NDArray, NDArray]:
    product, bound = _matmul_enclosed(left, right.T)
    value, _ = _compensated_dot(product.ravel(), product.ravel())
    return value, product, bound


def _positive_native_product(left: NDArray, right: NDArray) -> NDArray:
    count = left.shape[-1]
    value = left @ right
    allowance = _gamma(2 * count + 4)
    underflow = (2 * count + 1) * np.nextafter(0.0, 1.0)
    return _upper((value + underflow) / (1 - allowance))


def _component_gram(factor: NDArray, bound: NDArray) -> tuple[NDArray, NDArray]:
    gram = factor.T @ factor
    arithmetic = _gamma(2 * len(factor) + 1) * _positive_native_product(
        np.abs(factor.T), np.abs(factor)
    )
    error = (
        arithmetic
        + _positive_native_product(np.abs(factor.T), bound)
        + _positive_native_product(bound.T, np.abs(factor))
        + _positive_native_product(bound.T, bound)
    )
    return gram, _upper(error / (1 - _gamma(6)))


def _gram_cross(
    left: tuple[NDArray, NDArray], right: tuple[NDArray, NDArray], dimension: int
) -> tuple[float, float] | None:
    """Use a cheap Gram contraction only when cancellation is resolved."""
    A, A_error = left
    B, B_error = right
    value, error = _compensated_dot(A.ravel(), B.ravel())
    error += np.sum(np.abs(A) * B_error + A_error * np.abs(B) + A_error * B_error)
    error += 3 * A.size * _SMALLEST_SUBNORMAL  # Three perturbation products per entry.
    error = float(_upper(error / (1 - _gamma(6 * A.size + 4, _UNIT_ROUNDOFF))))
    if value <= 0 or error > _gamma(8 * dimension) * value:
        return None
    return value, error


def _derivative_values(
    factors: Sequence[NDArray], bounds: Sequence[NDArray], eta: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    count = len(factors)
    row_counts = np.array([len(factor) for factor in factors], dtype=float)
    if count and 0 <= eta < 1 and row_counts.sum() == factors[0].shape[1]:
        # The caller's fresh whitening certificate establishes full row rank.
        # With N=r, det+(F.T D**2 F)=det(F F.T)*det(D)**2, so log-weight
        # derivatives are exact row counts even for nonorthogonal components.
        return row_counts, np.zeros((count, count)), np.zeros(count), np.zeros((count, count))
    gradient, gradient_error, trace_bounds, frobenius = np.zeros((4, count))
    d = float(_upper(eta / (1 - eta) / (1 - _gamma(2))))
    for i, (factor, bound) in enumerate(zip(factors, bounds, strict=True)):
        value, arithmetic = _squared_norm_enclosed(factor, bound)
        gradient[i], trace_bounds[i] = value, arithmetic
        trace_upper = float(_upper(value + arithmetic))
        # C_i = Z_i.T Z_i <= G = sum C_i and ||G-I||_F <= eta.
        # Thus ||C_i||_F**2 <= (1+eta) tr(C_i), also <= tr(C_i)**2.
        # Cauchy-Schwarz gives |tr((G^-1-I) C_i)| <= d ||C_i||_F.
        frobenius[i] = min(
            trace_upper,
            float(_upper(math.sqrt((1 + eta) * trace_upper) / (1 - _gamma(4)))),
        )
        gradient_error[i] = float(_upper((arithmetic + d * frobenius[i]) / (1 - _gamma(3))))
    hessian = np.zeros((count, count))
    hessian_error = np.zeros_like(hessian)
    grams = tuple(
        _component_gram(factor, bound) for factor, bound in zip(factors, bounds, strict=True)
    )
    # Expanding tr(G^-1 C_i G^-1 C_j) around I gives two linear
    # perturbations and one quadratic term, each bounded by Frobenius norms.
    metric = float(_upper((2 * d + d * d) / (1 - _gamma(3))))
    for i in range(count):
        for j in range(i + 1, count):
            fast = _gram_cross(
                grams[i], grams[j], factors[i].shape[1] + len(factors[i]) + len(factors[j]) + 1
            )
            if fast is not None:
                cross, arithmetic = fast
            else:
                product, product_bound = _matmul_enclosed(factors[i], factors[j].T)
                product_bound = _upper(
                    (
                        product_bound
                        + _positive_product(np.abs(factors[i]), bounds[j].T)
                        + _positive_product(bounds[i], np.abs(factors[j].T))
                        + _positive_product(bounds[i], bounds[j].T)
                    )
                    / (1 - _gamma(4))
                )
                cross, arithmetic = _squared_norm_enclosed(product, product_bound)
            error = float(
                _upper((arithmetic + metric * frobenius[i] * frobenius[j]) / (1 - _gamma(4)))
            )
            hessian[i, j] = hessian[j, i] = -cross
            hessian_error[i, j] = hessian_error[j, i] = error
    for i in range(count):
        hessian[i, i] = -math.fsum(hessian[i])
        hessian_error[i, i] = float(
            _upper(math.fsum(hessian_error[i]) + _gamma(count + 1) * hessian[i, i])
        )
    return gradient, hessian, gradient_error, hessian_error


def _evaluate_penalty_support(
    support: _PenaltySupport, lambdas: NDArray, eps_rank: float | None = None
) -> SimilarityTransformResult:
    return cast(
        SimilarityTransformResult,
        _evaluate_penalty_geometry(support, lambdas, eps_rank, summary_only=False),
    )


def _evaluate_penalty_summary(
    support: _PenaltySupport, lambdas: NDArray, eps_rank: float | None = None
) -> _PenaltySummary:
    """Admit the same geometry while omitting unused dense output matrices."""
    values = _weights(lambdas, len(support.component_roots))
    if support.rank >= 128 and np.all(values > 0):
        previous_basis = support._basis_gram_evidence
        try:
            return cast(
                _PenaltySummary,
                _evaluate_penalty_geometry(
                    support, values, eps_rank, summary_only=True, _native_candidate=True
                ),
            )
        except PenaltyNumericalError:
            # Every factor, correction and error ledger below starts again
            # from the original proposal. No failed candidate state transfers.
            object.__setattr__(support, "_basis_gram_evidence", previous_basis)
    return cast(
        _PenaltySummary,
        _evaluate_penalty_geometry(support, values, eps_rank, summary_only=True),
    )


def _evaluate_penalty_geometry(
    support: _PenaltySupport,
    lambdas: NDArray,
    eps_rank: float | None,
    *,
    summary_only: bool,
    _native_candidate: bool = False,
) -> SimilarityTransformResult | _PenaltySummary:
    values = _weights(lambdas, len(support.component_roots))
    separation = _separation_parameter(eps_rank)
    count, width = len(values), support.Q_plus.shape[0]
    if np.any(values == 0):
        active_support = _penalty_support_from_roots(
            [
                root if value > 0 else np.empty((0, width))
                for root, value in zip(support.component_roots, values, strict=True)
            ],
            resolution_limited=support.component_resolution_limited,
            input_error_bounds=[
                bound if value > 0 else np.empty((0, width))
                for bound, value in zip(support.component_root_error_bounds, values, strict=True)
            ],
        )
        support = replace(
            active_support,
            component_reconstruction_bounds=support.component_reconstruction_bounds,
            support_projection_bounds=tuple(
                _readonly(_upper(previous + current)) if value > 0 else current
                for previous, current, value in zip(
                    support.support_projection_bounds,
                    active_support.support_projection_bounds,
                    values,
                    strict=True,
                )
            ),
        )
    rank = support.rank
    if rank == 0:
        if summary_only:
            gradient, hessian = _readonly(np.zeros(count)), _readonly(np.zeros((count, count)))
            return _PenaltySummary(
                0.0,
                0,
                gradient,
                hessian,
                support,
                _PenaltySummaryCertificate(
                    0.0,
                    0.0,
                    0.0,
                    gradient,
                    hessian,
                    any(support.component_resolution_limited),
                ),
                _readonly(values),
                0,
            )
        zero = np.zeros((width, width))
        certificate = _PenaltyCertificate(
            0.0,
            0.0,
            0.0,
            _readonly(np.zeros(count)),
            _readonly(np.zeros((count, count))),
            _readonly(zero),
            _readonly(zero),
            any(support.component_resolution_limited),
        )
        return SimilarityTransformResult(
            0.0,
            zero.copy(),
            support.Q_plus,
            support.Q_zero,
            zero.copy(),
            0,
            certificate,
            tuple(_readonly(np.empty((0, 0))) for _ in values),
            support,
            None,
            _readonly(values),
            0,
        )
    E, J, logdet, factor_log_error, volume_scale = (
        _separated_candidate(support, values, separation, _native_candidate=True)
        if _native_candidate
        else _separated_candidate(support, values, separation)
    )
    rows = sum(len(root) for root in support.component_roots)
    target = _gamma(8 * (rows + width + rank + count + 1))
    correction_count = 0
    for correction_index in range(2):
        evidence = []
        E, J, logdet, cached = _reference_correct_once(
            support.component_roots,
            values,
            E,
            J,
            logdet,
            _evidence=evidence,
            _refine=correction_index > 0,
        )
        correction_count += 1
        materialization_error, added_volume, added_log_error = evidence[0][:3]
        volume_scale += added_volume
        factor_log_error += added_log_error
        # Geometry evidence belongs to this QR update. Refreshing only the
        # reference actions below must neither omit nor charge it twice.
        if len(evidence[0]) > 3:
            factor_log_error += 2 * evidence[0][3]
        retained_dual = evidence[0][4] if len(evidence[0]) > 4 else None
        if retained_dual is not None and _same_operands(retained_dual.operands, E, J):
            dual, dual_error = retained_dual.product, retained_dual.error
        else:
            dual, dual_error = _matmul_enclosed(E, J)
        duality = _norm_upper(dual - np.eye(rank))
        duality = float(np.nextafter(duality + _norm_upper(dual_error), np.inf))
        if duality >= 1:
            continue
        update_error = _norm_upper(_positive_product(materialization_error, np.abs(J))) / (
            1 - duality
        )
        if update_error >= 1:
            continue
        if len(evidence[0]) <= 3:
            factor_log_error -= 2 * rank * math.log1p(-update_error)
        dual_log_error = 2 * _logdet_defect_bound(dual, dual_error)
        accepted = False
        for refine in (False, True) if correction_index == 0 else (True,):
            action, arithmetic_bounds = _reference_root_actions(
                support.component_roots, values, J, _refine=refine
            )
            bounds = tuple(
                _upper(
                    bound
                    + _root_error_product(
                        _upper((np.sqrt(value) * error + _SMALLEST_SUBNORMAL) / (1 - _gamma(3))),
                        np.abs(J),
                    )
                )
                for bound, error, value in zip(
                    arithmetic_bounds, support.component_root_error_bounds, values, strict=True
                )
            )
            whitening_evidence = []
            eta, _ = _whitening_certificate(
                action, bounds, E, J, _duality=duality, _evidence=whitening_evidence
            )
            if eta >= 1:
                continue
            discrepancy = max(
                (_norm_upper(a - b) for a, b in zip(action, cached, strict=True)), default=0.0
            )
            cache_allowance = target * rank + math.fsum(_norm_upper(b) for b in bounds)
            if discrepancy > cache_allowance:
                if correction_index > 0:
                    raise PenaltyNumericalError(
                        "stored derivative factor disagrees with reference action accuracy certificate"
                    )
                break
            gradient, hessian, g_error, h_error = _derivative_values(action, bounds, eta)
            log_error = (
                factor_log_error + _logdet_defect_bound(*whitening_evidence[0]) + dual_log_error
            )
            if (
                np.max(g_error, initial=0) <= target * rank
                and np.max(h_error, initial=0) <= target * rank**2
                and log_error <= target * volume_scale
            ):
                accepted = True
                break
        if accepted:
            break
    else:
        raise PenaltyNumericalError("reference root geometry cannot meet the accuracy contract")
    if summary_only:
        return _PenaltySummary(
            float(logdet),
            rank,
            _readonly(gradient),
            _readonly(hessian),
            support,
            _PenaltySummaryCertificate(
                eta,
                duality,
                float(_upper(log_error)),
                _readonly(g_error),
                _readonly(h_error),
                any(support.component_resolution_limited),
            ),
            _readonly(values),
            correction_count,
        )
    try:
        inverse, inverse_error = _inverse_gram_enclosed(J, eta)
    except PenaltyNumericalError as exc:
        raise PenaltyNumericalError("required dense penalty inverse is not representable") from exc
    if np.any((np.diag(inverse) == 0) & np.any(J != 0, axis=1)):
        raise PenaltyNumericalError("required dense penalty inverse is not representable")
    # The exact Gram is symmetric. Copying one computed triangle and its
    # enclosure makes it symmetric without another rounding/underflow step.
    upper = np.triu_indices(width, k=1)
    lower = (upper[1], upper[0])
    inverse[lower] = inverse[upper]
    inverse_error[lower] = inverse_error[upper]
    root = np.zeros((width, width))
    root[:rank] = E
    root_error = np.zeros_like(root)
    # An equivalent reference root is G**(1/2) (EJ)**(-1) E. Bound its
    # distance from the returned E, including both measured defects.
    root_relative = (eta / (1 + math.sqrt(1 - eta)) + duality) / (1 - duality)
    column_norm = np.array([_norm_upper(column) for column in E.T])
    root_error[:rank] = _upper(
        root_relative * column_norm / (1 - _gamma(4 * rank + 12, _UNIT_ROUNDOFF))
    )
    certificate = _PenaltyCertificate(
        eta,
        duality,
        float(_upper(log_error)),
        _readonly(g_error),
        _readonly(h_error),
        _readonly(root_error),
        _readonly(inverse_error),
        any(support.component_resolution_limited),
    )
    return SimilarityTransformResult(
        float(logdet),
        inverse,
        support.Q_plus,
        support.Q_zero,
        root,
        rank,
        certificate,
        tuple(_readonly(item) for item in action),
        support,
        None,
        _readonly(values),
        correction_count,
        _readonly(gradient),
        _readonly(hessian),
    )


def _derivative_result(result, penalty_matrices, lambdas) -> SimilarityTransformResult:
    values = _weights(lambdas, len(penalty_matrices))
    if result._component_factors is None:
        return similarity_transform_logdet(penalty_matrices, values)
    if result._input_lambdas is None or not np.array_equal(values, result._input_lambdas):
        raise ValueError("derivative weights do not match the retained penalty geometry")
    if result._input_matrices is not None and (
        len(penalty_matrices) != len(result._input_matrices)
        or any(
            not np.array_equal(a, b)
            for a, b in zip(penalty_matrices, result._input_matrices, strict=True)
        )
    ):
        raise ValueError("derivative components do not match the retained penalty geometry")
    if result._input_matrices is None and result._support is not None:
        if any(
            not np.array_equal(a, b)
            for a, b in zip(penalty_matrices, result._support.component_roots, strict=True)
        ):
            raise ValueError("derivative roots do not match the retained penalty geometry")
    return result


def logdet_s_gradient(result, penalty_matrices: list[NDArray], lambdas: NDArray) -> NDArray:
    """Gradient in log weights from bounded reference component factors.

    Private root-entry results accept the ordered frozen roots as the second
    argument, avoiding a Gram construction solely to check provenance.
    """
    result = _derivative_result(result, penalty_matrices, lambdas)
    if result._gradient is not None:
        return result._gradient.copy()
    return np.array(
        [float(np.sum(factor**2, dtype=np.float64)) for factor in result._component_factors]
    )


def logdet_s_hessian(result, penalty_matrices: list[NDArray], lambdas: NDArray) -> NDArray:
    """Hessian in log weights; inactive rows and columns are zero."""
    result = _derivative_result(result, penalty_matrices, lambdas)
    if result._hessian is not None:
        return result._hessian.copy()
    factors = result._component_factors
    count = len(factors)
    hessian = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            cross, _, _ = _cross_value(factors[i], factors[j])
            hessian[i, j] = hessian[j, i] = -cross
    for i in range(count):
        hessian[i, i] = -math.fsum(hessian[i])
    return hessian
