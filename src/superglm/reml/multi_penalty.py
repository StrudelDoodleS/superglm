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

from superglm.reml._compensated import _dot2_value
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
_LD = np.longdouble
_U_LD = np.finfo(_LD).eps / 2
_TINY_LD = np.nextafter(_LD(0), _LD(1))


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
    wide: NDArray
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


def _gamma(count: int, unit: float | np.longdouble = _EPS / 2) -> float:
    product = _LD(count) * _LD(unit)
    if product >= 1:
        raise PenaltyNumericalError("arithmetic error bound is unresolved")
    bound = product / (1 - product) / (1 - 3 * _U_LD)
    return float(np.nextafter(float(bound), np.inf))


def _upper(value: NDArray | float) -> NDArray:
    result = _finite_double(np.maximum(value, 0), "arithmetic error bound")
    with np.errstate(over="ignore"):
        result = np.nextafter(result, np.inf)
    return _finite_double(result, "arithmetic error bound")


def _norm_upper(value: NDArray) -> float:
    absolute = np.abs(np.asarray(value, dtype=_LD))
    maximum = np.max(absolute, initial=_LD(0))
    if maximum == 0:
        return 0.0
    norm = maximum * np.sqrt(np.sum((absolute / maximum) ** 2, dtype=_LD))
    return float(_upper(norm / (1 - _gamma(3 * absolute.size + 2, _U_LD))))


def _positive_product(left: NDArray, right: NDArray) -> NDArray:
    """Upper bound for a nonnegative dot, including its own rounding.

    Outward operand casts preserve the inequality before native GEMM's
    componentwise gamma bound (Higham and Mary, 2022, sections 3--4).
    Unsupported exponents retain the wider product below.
    """
    left, right = np.asarray(left, dtype=_LD), np.asarray(right, dtype=_LD)
    count = left.shape[-1]
    if left.ndim == right.ndim == 2 and left.shape[1] == right.shape[0]:
        maximum_left = np.max(left, initial=_LD(0))
        maximum_right = np.max(right, initial=_LD(0))
        if (
            np.isfinite(maximum_left)
            and np.isfinite(maximum_right)
            and np.min(left, initial=_LD(0)) >= 0
            and np.min(right, initial=_LD(0)) >= 0
        ):
            shape = (left.shape[0], right.shape[1])
            if maximum_left == 0 or maximum_right == 0:
                return np.zeros(shape)
            maximum_exponent = (
                int(np.frexp(maximum_left)[1])
                + int(np.frexp(maximum_right)[1])
                + (count - 1).bit_length()
            )
            # Every exact dot is strictly below 2**maximum_exponent.
            # Original wide exponents avoid multiplying or casting tiny bounds.
            if maximum_exponent <= np.finfo(float).minexp - np.finfo(float).nmant:
                return np.full(shape, np.nextafter(0.0, 1.0))
        with np.errstate(over="ignore", under="ignore"):
            a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
            a = np.where(a.astype(_LD) < left, np.nextafter(a, np.inf), a)
            b = np.where(b.astype(_LD) < right, np.nextafter(b, np.inf), b)
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a >= 0) and np.all(b >= 0):
            a_nonzero, b_nonzero = a[a > 0], b[b > 0]
            if (
                a_nonzero.size
                and b_nonzero.size
                and np.min(a_nonzero) >= np.finfo(float).tiny
                and np.min(b_nonzero) >= np.finfo(float).tiny
            ):
                minimum_exponent = (
                    math.frexp(float(np.min(a_nonzero)))[1]
                    + math.frexp(float(np.min(b_nonzero)))[1]
                )
                maximum_exponent = (
                    math.frexp(float(np.max(a_nonzero)))[1]
                    + math.frexp(float(np.max(b_nonzero)))[1]
                )
                # frexp mantissas are in [1/2, 1). These integer tests keep
                # every nonzero product normal and the positive sum finite.
                if (
                    minimum_exponent >= -1020
                    and maximum_exponent + (count - 1).bit_length() <= 1021
                ):
                    return _positive_native_product(a, b)
    value = left @ right
    value = (value + (2 * count + 1) * _TINY_LD) / (1 - _gamma(2 * count + 1, _U_LD))
    return _upper(value)


def _dyadic_slices(normalized: NDArray) -> tuple[tuple[NDArray, ...], NDArray]:
    """Extract four exact, sign-preserving 21-bit dyadic slices."""
    residual = np.array(normalized, dtype=_LD, copy=True)
    slices = []
    for bits in (21, 42, 63, 84):
        scaled = residual * _LD(2.0**bits)
        integral = np.trunc(np.asarray(scaled, dtype=float))
        # |scaled| < 2**21. Its native cast can cross an integer only away
        # from zero; repair that case before using the exactly stored integer.
        integral -= np.where(np.abs(integral.astype(_LD)) > np.abs(scaled), np.sign(integral), 0.0)
        component = integral * 2.0**-bits
        slices.append(component)
        residual -= component
    return tuple(slices), residual


def _dyadic_product(left: NDArray, right: NDArray, magnitude: NDArray) -> NDArray | None:
    """Return a witness only inside the existing wide-product allowance.

    Ozaki, Ogita, Oishi and Rump (2013), Theorem 2: a dyadic grid and a
    bounded absolute partial sum make each native slice GEMM exact. Here
    21+21+ceil(log2(4*k)) <= 53, k <= 512. Up to four products on each
    shared-grid diagonal also add exactly in binary64. Sign-preserving
    truncation and a bounded residual replace that paper's extraction and
    termination procedure.
    """
    info = np.finfo(_LD)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
        or not 8 <= left.shape[1] <= 512
        or info.nmant <= np.finfo(float).nmant
        or info.minexp > 2 * np.finfo(float).minexp - 2 * info.nmant - 16
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or not np.all(np.isfinite(magnitude))
        or np.any(magnitude < 0)
    ):
        return None
    count = left.shape[1]
    ea = np.frexp(np.max(np.abs(left), axis=1, initial=_LD(0)))[1]
    eb = np.frexp(np.max(np.abs(right), axis=0, initial=_LD(0)))[1]
    with np.errstate(over="ignore", under="ignore"):
        sa = np.ldexp(np.ones((left.shape[0], 1), dtype=_LD), ea[:, None])
        sb = np.ldexp(np.ones((1, right.shape[1]), dtype=_LD), eb[None, :])
        scale = sa * sb
    if any(np.any(~np.isfinite(value)) or np.any(value < info.tiny) for value in (sa, sb, scale)):
        return None
    a, b = left / sa, right / sb
    for source, normalized, factor in ((left, a, sa), (right, b, sb)):
        if (
            not np.all(np.isfinite(normalized))
            or np.any((normalized != 0) & (np.abs(normalized) < np.finfo(float).tiny))
            or not np.array_equal(normalized * factor, source)
        ):
            return None
    slices_a, ra = _dyadic_slices(a)
    slices_b, rb = _dyadic_slices(b)
    abar = a - ra

    def upper(value: NDArray, operations: int) -> NDArray:
        bound = value / (1 - _LD(_gamma(operations, _U_LD)))
        # The format/normalization gates keep these scalar operations normal.
        return np.where(bound == 0, _LD(0), np.nextafter(bound, _LD(np.inf)))

    sum_b = upper(np.sum(np.abs(b), axis=0, dtype=_LD), count)
    sum_abar = upper(np.sum(np.abs(abar), axis=1, dtype=_LD), count)
    tail = np.max(np.abs(ra), axis=1, initial=_LD(0))[:, None] * sum_b[None, :]
    tail += sum_abar[:, None] * np.max(np.abs(rb), axis=0, initial=_LD(0))[None, :]
    tail = upper(tail, 5)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        restored_tail = tail * scale
        reconstruction = _LD(_gamma(15, _U_LD)) * magnitude.astype(_LD)
        total = upper(reconstruction + restored_tail, 4)
        allowance = _LD(_gamma(2 * count + 1, _U_LD)) * magnitude.astype(_LD)
    if (
        not np.all(np.isfinite(total))
        or not np.all(np.isfinite(allowance))
        or np.any((tail != 0) & (restored_tail < info.tiny))
        or np.any((magnitude != 0) & (reconstruction < info.tiny))
    ):
        return None
    allowance = np.nextafter(allowance, _LD(-np.inf))
    if np.any(total > allowance):
        return None
    result = None
    for diagonal in range(7):
        native = None
        for i in range(max(0, diagonal - 3), min(3, diagonal) + 1):
            a_slice, b_slice = slices_a[i], slices_b[diagonal - i]
            product = a_slice @ b_slice
            if native is None:
                native = product
            else:
                native += product
        if result is None:
            result = native.astype(_LD)
        else:
            result += native
    # Each native diagonal's absolute integer sum is below 4*k*2**42 <= 2**53.
    # Its grid is no finer than 2**-168. The six wide additions have no
    # underflow and remain covered by the existing conservative gamma15 term.
    with np.errstate(over="ignore", under="ignore"):
        restored = result * scale
    if not np.all(np.isfinite(restored)) or np.any((result != 0) & (np.abs(restored) < info.tiny)):
        return None
    return restored


def _wide_product(left: NDArray, right: NDArray, magnitude: NDArray | None = None) -> NDArray:
    left, right = np.asarray(left, dtype=_LD), np.asarray(right, dtype=_LD)
    # Preparation pays off on the measured tensor dimensions. Small products
    # retain their original dispatch, independently of the k<=512 proof gate.
    if (
        left.ndim == right.ndim == 2
        and min(*left.shape, *right.shape) >= 128
        and left.shape[1] <= 512
        and np.finfo(_LD).nmant > np.finfo(float).nmant
    ):
        if magnitude is None:
            try:
                magnitude = _positive_product(np.abs(left), np.abs(right))
            except PenaltyNumericalError:
                # Optional preparation must not refuse a finite signed wide
                # product merely because its absolute bound exceeds float64.
                return left @ right
        result = _dyadic_product(left, right, magnitude)
        if result is not None:
            return result
    return left @ right


def _matmul_enclosed(
    left: NDArray, right: NDArray, *, _evidence: list[_ProductEvidence] | None = None
) -> tuple[NDArray, NDArray]:
    left_ld, right_ld = np.asarray(left, dtype=_LD), np.asarray(right, dtype=_LD)
    magnitude = _positive_product(np.abs(left_ld), np.abs(right_ld))
    wide = _wide_product(left_ld, right_ld, magnitude)
    result = _finite_double(wide, "penalty factor product")
    error = _gamma(2 * left.shape[-1] + 1, _U_LD) * magnitude.astype(_LD)
    error += np.abs(wide - result.astype(_LD)) + (2 * left.shape[-1] + 1) * _TINY_LD
    if _evidence is not None:
        _evidence.append(
            _ProductEvidence(
                (_evidence_copy(left_ld), _evidence_copy(right_ld)),
                _evidence_copy(wide),
                _evidence_copy(magnitude),
            )
        )
    return result, _upper(error)


def _root_error_product(left: NDArray, right: NDArray) -> NDArray:
    """Reuse identical tiny error rows within the existing wide product."""
    left, right = np.asarray(left, dtype=_LD), np.asarray(right, dtype=_LD)
    if (
        left.ndim == right.ndim == 2
        and left.shape[0] > 1
        and left.shape[1] == right.shape[0]
        and np.finfo(_LD).nmant > np.finfo(float).nmant
        and np.all(np.isfinite(left))
        and np.all(left >= 0)
        and np.all(left <= np.finfo(float).tiny / 2)
        and np.all(left == left[:1])
    ):
        # Outward casts of these operands cannot pass the native normal-range
        # gate. The wide dot retains its inner dimension and every allowance.
        row = _positive_product(left[:1], right)
        return np.repeat(row, left.shape[0], axis=0)
    return _positive_product(left, right)


def _basis_gram(support: _PenaltySupport, basis: NDArray) -> tuple[NDArray, NDArray]:
    """Memoize the unchanged support Gram, retaining its original enclosure."""
    if basis is not support.Q_plus:
        return _matmul_enclosed(basis.T, basis)
    arithmetic = (
        _LD,
        _EPS,
        _U_LD,
        _TINY_LD,
        _matmul_enclosed,
        _wide_product,
        _dyadic_product,
        _dyadic_slices,
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
    """Materialize the inverse after admission, enclosing the operand cast.

    Higham and Mary (2022), sections 3--4: native GEMM has a componentwise
    gamma bound. Normal-range rounding also gives |J - B| <= alpha |B|,
    B = float64(J), alpha = u/(1-u). This bounds the cast cross terms.
    The spectral metric uncertainty couples arbitrary rows of J, so its
    entrywise bound uses the outer product of upper bounds on their norms.
    """
    J = np.asarray(inverse_root, dtype=_LD)
    rank = J.shape[1]
    with np.errstate(over="ignore", under="ignore"):
        native = np.asarray(J, dtype=np.float64)
    nonzero = np.abs(native[native != 0])
    normal = (
        np.all(np.isfinite(native))
        and not np.any((J != 0) & (native == 0))
        and (
            not nonzero.size
            or (
                np.min(nonzero) >= np.nextafter(np.sqrt(np.finfo(float).tiny), np.inf)
                and np.max(nonzero) <= np.sqrt(np.finfo(float).max / (4 * max(rank, 1)))
            )
        )
    )
    if not normal:
        inverse, wide_rounding = _matmul_enclosed(J, J.T)
        rounding = wide_rounding.astype(_LD)
        row_squares = _upper(np.diag(inverse).astype(_LD) + np.diag(rounding))
    else:
        magnitude = _positive_native_product(np.abs(native), np.abs(native.T)).astype(_LD)
        inverse = native @ native.T
        unit = _LD(_EPS) / 2
        alpha = unit / (1 - unit)
        gamma = _gamma(2 * rank + 1)
        rounding = (gamma + 2 * alpha + alpha**2) * magnitude
        # Signed partial sums may underflow despite the normal-product gate.
        rounding += (2 * rank + 1) * _LD(np.nextafter(0.0, 1.0)) / (1 - gamma)
        row_squares = _upper((1 + alpha) ** 2 * np.diag(magnitude) / (1 - _gamma(8, _U_LD)))
    row_norms = _upper(np.sqrt(row_squares.astype(_LD)) / (1 - _gamma(4, _U_LD)))
    row_products = _positive_product(row_norms[:, None], row_norms[None, :]).astype(_LD)
    error = rounding + _LD(eta) / (1 - _LD(eta)) * row_products
    return inverse, _upper((error + 32 * _TINY_LD) / (1 - _gamma(32, _U_LD)))


def _trace_bound(value: NDArray, error: NDArray) -> float:
    diagonal = np.diag(value).astype(_LD)
    trace = np.sum(diagonal, dtype=_LD)
    bound = np.sum(np.diag(error).astype(_LD), dtype=_LD)
    bound += _gamma(2 * len(diagonal) + 1, _U_LD) * np.sum(np.abs(diagonal))
    return float(_upper((abs(trace) + bound) / (1 - _gamma(4 * len(diagonal) + 4, _U_LD))))


def _logdet_defect_bound(product: NDArray, error: NDArray) -> float:
    """Trace-series bound, retaining cancellation-free second-order terms."""
    defect = product.astype(_LD) - np.eye(len(product), dtype=_LD)
    radius = _norm_upper(defect) + _norm_upper(error)
    if radius >= 1:
        return np.inf
    # tr(log(I+D)) = tr(D) + remainder. For k >= 2,
    # |tr(D**k)| <= ||D||_F**2 * ||D||_2**(k-2).
    return float(_upper(_trace_bound(defect, error) + radius**2 / (2 * (1 - radius))))


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
    """Enclose the determinant effect of an observed, signed cast residual."""
    if _product_evidence is not None and _same_operands(_product_evidence.operands, left, right):
        wide = _product_evidence.wide
        magnitude = _product_evidence.magnitude.astype(_LD)
    else:
        wide = np.asarray(left, dtype=_LD) @ np.asarray(right, dtype=_LD)
        magnitude = _positive_product(np.abs(left), np.abs(right)).astype(_LD)
    residual = wide - product.astype(_LD)
    uncertain = _gamma(2 * left.shape[-1] + 1, _U_LD) * magnitude
    if input_bound is not None:
        uncertain += input_bound
    action, error = _matmul_enclosed(residual, inverse)
    error = _upper(error.astype(_LD) + _positive_product(_upper(uncertain), np.abs(inverse)))
    dual, dual_error = _matmul_enclosed(product, inverse)
    if _duality_evidence is not None:
        _duality_evidence.append(
            _DualityEvidence(
                (_evidence_copy(product), _evidence_copy(inverse)),
                _evidence_copy(dual),
                _evidence_copy(dual_error),
            )
        )
    duality = _norm_upper(dual.astype(_LD) - np.eye(len(dual))) + _norm_upper(dual_error)
    norm = _norm_upper(action) + _norm_upper(error)
    if duality >= 1 or norm >= 1 - duality:
        return np.inf
    relative = norm / (1 - duality)
    return float(
        _upper(
            _trace_bound(action, error)
            + norm * duality / (1 - duality)
            + relative**2 / (2 * (1 - relative))
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

    Extended precision is bounded at its actual platform precision, without
    assuming that every platform provides additional mantissa bits.
    """
    values = _weights(lambdas, len(roots))
    J = np.asarray(inverse_root, dtype=_LD)
    actions, bounds = [], []
    rows, rank = sum(len(root) for root in roots), J.shape[1]
    budget = _gamma(8 * (rows + J.shape[0] + rank + len(roots) + 1))
    dot_budget = budget / (8 * max(rank, 1) * math.sqrt(max(rows, 1)))
    for root, weight in zip(roots, values, strict=True):
        H = np.sqrt(_LD(weight)) * np.asarray(root, dtype=_LD)
        magnitude = _positive_product(np.abs(H), np.abs(J)).astype(_LD)
        wide = _wide_product(H, J, magnitude)
        action = _finite_double(wide, "reference root action")
        error = _gamma(2 * root.shape[1] + 4, _U_LD) * magnitude
        # Scaling a root entry can underflow before its multiplication by J.
        # That absolute error must follow the action, even on platforms where
        # longdouble has only the exponent range of double.
        scaling_underflow = np.where((root != 0) & (weight != 0), _TINY_LD, _LD(0))
        error += _positive_product(scaling_underflow, np.abs(J))
        error += np.abs(wide - action.astype(_LD)) + (2 * root.shape[1] + 4) * _TINY_LD
        for row, column in np.argwhere(error > dot_budget) if _refine else ():
            try:
                dot, dot_error = _compensated_dot(root[row], J[:, column])
                scale = np.sqrt(_LD(weight))
                scale_error = _gamma(1, _U_LD) * abs(scale) + _TINY_LD
                scaled = scale * _LD(dot)
                refined = float(_finite_double(scaled, "compensated root action"))
                refined_error = (
                    abs(scale) * dot_error
                    + scale_error * (abs(dot) + dot_error)
                    + _gamma(1, _U_LD) * abs(scaled)
                    + abs(scaled - _LD(refined))
                    + 2 * _TINY_LD
                )
                refined_error = float(_upper(refined_error / (1 - _gamma(12, _U_LD))))
            except PenaltyNumericalError:
                continue
            if refined_error < error[row, column]:
                action[row, column] = refined
                error[row, column] = refined_error
        actions.append(action)
        bounds.append(_upper(error))
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


def _compensated_dot(left: NDArray, right: NDArray) -> tuple[float, float]:
    """Dot2 with its computed-value enclosure, not a rounding heuristic.

    Ogita, Rump and Oishi (2005), Algorithms 3.1/3.3/3.5 and 5.3,
    Proposition 5.5. Splitting the wider operand adds an explicit residual.
    """
    if not len(left):
        return 0.0, 0.0
    right = np.asarray(right, dtype=_LD)
    high = _finite_double(right, "compensated operand")
    low = _finite_double(right - high.astype(_LD), "compensated operand remainder")
    residual = right - high.astype(_LD) - low.astype(_LD)
    x, y = np.tile(left, 2), np.concatenate([high, low])
    magnitude = float(_positive_product(np.abs(x)[None, :], np.abs(y)[:, None])[0, 0])
    value, compiled = _dot2_value(x, y)
    if not compiled:
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
        value = product + correction
    if not math.isfinite(value):
        raise PenaltyNumericalError("compensated dot is not representable")
    unit = _LD(_EPS) / 2
    count = len(x)
    error = (
        unit * abs(value)
        + _LD(_gamma(count)) ** 2 * magnitude
        + 5 * count * _LD(np.nextafter(0.0, 1.0))
    ) / (1 - unit)
    error += _positive_product(np.abs(left)[None, :], np.abs(residual)[:, None])[0, 0]
    return value, float(_upper(error / (1 - _gamma(12, _U_LD))))


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
    A = np.asarray(matrix, dtype=_LD)
    result = np.array(rhs, dtype=_LD, copy=True)
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
    return np.asarray(np.asarray(left, dtype=float) @ np.asarray(right, dtype=float), dtype=_LD)


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
        [
            np.sqrt(_LD(value)) * root.astype(_LD)
            for root, value in zip(support.component_roots, values, strict=True)
        ]
    )
    weighted = _candidate_product(stack, basis) if _native_candidate else None
    if weighted is None:
        weighted = _wide_product(stack, basis.astype(_LD))
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
    compact = upper.astype(_LD) * scales
    product_evidence = []
    E, E_error = _matmul_enclosed(compact, basis.T, _evidence=product_evidence)
    magnitude = product_evidence[0].magnitude
    E_error = _upper(E_error.astype(_LD) + _gamma(1, _U_LD) * magnitude)
    inverse = _triangular_solve(upper[::-1, ::-1], np.eye(rank)[::-1])[::-1]
    J = basis.astype(_LD) @ (inverse / scales[:, None])
    _finite_double(J, "candidate inverse root")
    terms = [
        *(2 * float(np.log(v)) for v in scales),
        *(2 * math.log(abs(v)) for v in np.diag(upper)),
    ]
    volume_scale = rank + math.fsum(map(abs, terms))
    gram, gram_error = _basis_gram(support, basis)
    orthogonality = _norm_upper(gram.astype(_LD) - np.eye(rank)) + _norm_upper(gram_error)
    materialization = _materialization_logdet_bound(
        compact,
        basis.T,
        E,
        J,
        _gamma(1, _U_LD) * magnitude,
        _product_evidence=product_evidence[0],
    )
    if orthogonality >= 1 or not np.isfinite(materialization):
        return None
    log_error = (
        _gamma(4 * len(terms) + 4) * volume_scale
        - rank * math.log1p(-orthogonality)
        + 2 * materialization
    )
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
    coordinates = np.asarray(chosen, dtype=_LD) @ rotation.astype(_LD)
    # These are candidate zeros only. Omitted components and entries remain
    # present in every reference correction and fresh certificate.
    coordinates[np.triu_indices(rank, 1)] = 0
    weighted = np.exp(np.asarray(logs, dtype=_LD))[:, None] * coordinates
    maxima = np.max(np.abs(weighted), axis=0)
    scales = maxima * np.sqrt(np.sum((weighted / maxima) ** 2, axis=0))
    normalized = _finite_double(weighted / scales, "scaled candidate roots")
    upper = scipy.linalg.qr(normalized, mode="r", check_finite=False)[0][:rank]
    if np.any(np.diag(upper) == 0):
        raise PenaltyNumericalError("zero candidate pivot on fixed penalty support")
    compact = (
        (upper.astype(_LD) * scales)
        @ rotation.T.astype(_LD)
        @ support.coordinate_triangular.T.astype(_LD)
    )
    E, E_error = _matmul_enclosed(compact, support.Q_plus.T)
    rhs = _triangular_solve(upper[::-1, ::-1], np.eye(rank)[::-1])[::-1]
    rhs = rotation.astype(_LD) @ (rhs / scales[:, None])
    solved = _triangular_solve(support.coordinate_triangular.T, rhs)
    J = support.Q_plus.astype(_LD) @ solved
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
) -> tuple[float, float]:
    Z, B = np.vstack(actions), np.vstack(action_bounds)
    gram, multiplication_bound = _matmul_enclosed(Z.T, Z)
    error = multiplication_bound.astype(_LD)
    error += _positive_product(np.abs(Z.T), B)
    error += _positive_product(B.T, np.abs(Z))
    error += _positive_product(B.T, B)
    eta = _norm_upper(gram.astype(_LD) - np.eye(J.shape[1], dtype=_LD))
    eta += _norm_upper(_upper(error))
    if _duality is None:
        product, product_bound = _matmul_enclosed(E, J)
        duality = _norm_upper(product.astype(_LD) - np.eye(J.shape[1], dtype=_LD))
        duality += _norm_upper(product_bound)
        _duality = float(np.nextafter(duality, np.inf))
    return float(np.nextafter(eta, np.inf)), _duality


def _cross_value(left: NDArray, right: NDArray) -> tuple[float, NDArray, NDArray]:
    product, bound = _matmul_enclosed(left, right.T)
    return float(np.sum(product.astype(_LD) ** 2, dtype=_LD)), product, bound


def _positive_native_product(left: NDArray, right: NDArray) -> NDArray:
    count = left.shape[-1]
    value = left @ right
    allowance = _gamma(2 * count + 4)
    underflow = (2 * count + 1) * np.nextafter(0.0, 1.0)
    return _upper((value.astype(_LD) + underflow) / (1 - allowance))


def _component_gram(factor: NDArray, bound: NDArray) -> tuple[NDArray, NDArray]:
    gram = factor.T @ factor
    arithmetic = _gamma(2 * len(factor) + 1) * _positive_native_product(
        np.abs(factor.T), np.abs(factor)
    )
    error = (
        arithmetic.astype(_LD)
        + _positive_native_product(np.abs(factor.T), bound)
        + _positive_native_product(bound.T, np.abs(factor))
        + _positive_native_product(bound.T, bound)
    )
    return gram, _upper(error)


def _gram_cross(
    left: tuple[NDArray, NDArray], right: tuple[NDArray, NDArray], dimension: int
) -> tuple[float, float] | None:
    """Use a cheap Gram contraction only when cancellation is resolved."""
    A, A_error = (item.astype(_LD) for item in left)
    B, B_error = (item.astype(_LD) for item in right)
    wide = np.sum(A * B, dtype=_LD)
    value = float(wide)
    error = abs(wide - _LD(value)) + _gamma(2 * A.size + 1, _U_LD) * np.sum(np.abs(A * B))
    error += np.sum(np.abs(A) * B_error + A_error * np.abs(B) + A_error * B_error)
    error = float(_upper(error / (1 - _gamma(6 * A.size + 4, _U_LD))))
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
    gradient, gradient_error, trace_bounds = np.zeros((3, count))
    for i, (factor, bound) in enumerate(zip(factors, bounds, strict=True)):
        value = np.sum(factor.astype(_LD) ** 2, dtype=_LD)
        gradient[i] = float(value)
        arithmetic = abs(value - _LD(gradient[i])) + _gamma(2 * factor.size + 1, _U_LD) * value
        arithmetic += np.sum(2 * np.abs(factor.astype(_LD)) * bound + bound.astype(_LD) ** 2)
        trace_bounds[i] = float(_upper(arithmetic))
        gradient_error[i] = float(_upper(arithmetic + eta / (1 - eta) * (value + arithmetic)))
    hessian = np.zeros((count, count))
    hessian_error = np.zeros_like(hessian)
    grams = tuple(
        _component_gram(factor, bound) for factor, bound in zip(factors, bounds, strict=True)
    )
    metric = 2 * eta / (1 - eta) + (eta / (1 - eta)) ** 2
    for i in range(count):
        for j in range(i + 1, count):
            fast = _gram_cross(
                grams[i], grams[j], factors[i].shape[1] + len(factors[i]) + len(factors[j]) + 1
            )
            if fast is not None:
                cross, arithmetic = fast
            else:
                cross, product, product_bound = _cross_value(factors[i], factors[j])
                product_bound = _upper(
                    product_bound.astype(_LD)
                    + _positive_product(np.abs(factors[i]), bounds[j].T)
                    + _positive_product(bounds[i], np.abs(factors[j].T))
                    + _positive_product(bounds[i], bounds[j].T)
                )
                wide = np.sum(product.astype(_LD) ** 2, dtype=_LD)
                arithmetic = abs(wide - _LD(cross)) + _gamma(2 * product.size + 2, _U_LD) * wide
                arithmetic += np.sum(
                    2 * np.abs(product.astype(_LD)) * product_bound + product_bound.astype(_LD) ** 2
                )
            error = float(
                _upper(
                    arithmetic
                    + metric * (gradient[i] + trace_bounds[i]) * (gradient[j] + trace_bounds[j])
                )
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
                _readonly(_upper(previous.astype(_LD) + current.astype(_LD)))
                if value > 0
                else current
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
        duality = _norm_upper(dual.astype(_LD) - np.eye(rank, dtype=_LD))
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
                    bound.astype(_LD)
                    + _root_error_product(np.sqrt(_LD(value)) * error.astype(_LD), np.abs(J))
                )
                for bound, error, value in zip(
                    arithmetic_bounds, support.component_root_error_bounds, values, strict=True
                )
            )
            eta, _ = _whitening_certificate(action, bounds, E, J, _duality=duality)
            if eta >= 1:
                continue
            discrepancy = max(
                (_norm_upper(a - b) for a, b in zip(action, cached, strict=True)), default=0.0
            )
            cache_allowance = target * rank + math.fsum(_norm_upper(b) for b in bounds)
            if discrepancy > cache_allowance:
                if correction_index > 0:
                    raise PenaltyNumericalError(
                        "stored derivative factor disagrees with reference action"
                    )
                break
            gradient, hessian, g_error, h_error = _derivative_values(action, bounds, eta)
            log_error = factor_log_error - rank * math.log1p(-eta) + dual_log_error
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
    column_norm = np.sqrt(np.sum(E.astype(_LD) ** 2, axis=0))
    root_error[:rank] = _upper(root_relative * column_norm / (1 - _gamma(4 * rank + 12, _U_LD)))
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
        [float(np.sum(factor.astype(_LD) ** 2, dtype=_LD)) for factor in result._component_factors]
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
