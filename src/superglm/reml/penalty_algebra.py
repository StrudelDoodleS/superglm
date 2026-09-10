"""REML penalty eigenstructure and log-determinant algebra.

Pre-computes per-term penalty eigenstructure (Wood 2011 Section 3.1) and
provides log|S|₊ / ∂log|S|₊ / ∂²log|S|₊ for both single and multi-penalty
groups.

References
----------
- Wood (2011): Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  JRSS-B 73(1), 3-36.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.factor_smooth_geometry import (
    adjoint_sum_to_zero_blocks,
    expand_sum_to_zero_blocks,
    sum_to_zero_contrast,
    sum_to_zero_penalty,
)
from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.reml.result import PenaltyCache
from superglm.types import GroupSlice, PenaltyComponent


@dataclass(frozen=True)
class TensorPairLogdetSummary:
    """Static spectral ingredients for a 2-penalty discrete tensor block."""

    group_name: str
    tensor_id: int
    lambda_names: tuple[str, str]
    eigvals_left: NDArray
    eigvals_right: NDArray


@dataclass(frozen=True)
class TensorPairLogdetEvaluation:
    """Closed-form log|S|_+ summary for one tensor lambda pair."""

    group_name: str
    tensor_id: int
    lambda_names: tuple[str, str]
    logdet_s_plus: float
    rank: float
    gradient: dict[str, float]
    hessian: dict[tuple[str, str], float]
    logdet_error: float = 0.0
    gradient_error: dict[str, float] = field(default_factory=dict)
    hessian_error: dict[tuple[str, str], float] = field(default_factory=dict)
    support_rank: int | None = None


@dataclass(frozen=True)
class _PenaltyLogdetEvaluation:
    """One component representative for scalar rank and determinant derivatives."""

    rank: int
    logdet: float
    gradient: dict[str, float]
    hessian: dict[tuple[str, str], float]
    logdet_error: float
    gradient_error: dict[str, float]
    hessian_error: dict[tuple[str, str], float]


def _frozen_array(value: NDArray) -> NDArray:
    """Expose a read-only view whose owned backing data are also read-only."""
    backing = np.array(value, dtype=float, copy=True)
    backing.setflags(write=False)
    result = backing.view()
    result.setflags(write=False)
    return result


def _enclosed_bound_sum(*bounds: NDArray | float | np.longdouble) -> NDArray:
    from superglm.reml.multi_penalty import _gamma, _upper

    wide = np.sum(np.asarray(bounds, dtype=np.longdouble), axis=0)
    unit = np.finfo(np.longdouble).eps / 2
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    return _upper((wide + len(bounds) * tiny) / (1 - _gamma(len(bounds) + 2, unit)))


def _enclosed_root_gram(root: NDArray, error: NDArray) -> tuple[NDArray, NDArray]:
    from superglm.reml.multi_penalty import _matmul_enclosed, _positive_product

    gram, arithmetic = _matmul_enclosed(root.T, root)
    bound = _enclosed_bound_sum(
        arithmetic,
        _positive_product(np.abs(root).T, error),
        _positive_product(error.T, np.abs(root)),
        _positive_product(error.T, error),
    )
    return gram, bound


def _context_product(left: NDArray, right: NDArray, *, refine: bool) -> tuple[NDArray, NDArray]:
    from superglm.reml.multi_penalty import _compensated_dot, _matmul_enclosed

    value, error = _matmul_enclosed(left, right)
    if refine:
        for row, column in np.ndindex(value.shape):
            corrected, bound = _compensated_dot(left[row], right[:, column])
            if bound < error[row, column]:
                value[row, column], error[row, column] = corrected, bound
    return value, error


def _retained_coordinate_map(
    support, coordinate_map: NDArray, *, refine: bool = False
) -> tuple[NDArray, NDArray]:
    """Enclose (Q.T Q)^-1 Q.T C without amplifying off-support root residue."""
    from superglm.reml.multi_penalty import (
        _gamma,
        _norm_upper,
        _positive_product,
        _upper,
    )
    from superglm.reml.penalty_support import PenaltyNumericalError

    basis, rank = support.Q_plus, support.rank
    if rank == 0:
        empty = np.empty((0, coordinate_map.shape[1]))
        return empty, empty.copy()
    gram, gram_error = _context_product(basis.T, basis, refine=refine)
    defect_bound = _enclosed_bound_sum(np.abs(gram - np.eye(rank)), gram_error)
    eta = _norm_upper(defect_bound)
    if eta >= 1:
        raise PenaltyNumericalError("retained penalty coordinates are unresolved")
    mapped, mapped_error = _context_product(basis.T, coordinate_map, refine=refine)
    result = scipy.linalg.solve(gram, mapped, assume_a="pos", check_finite=False)
    product, product_error = _context_product(gram, result, refine=refine)
    residual = _enclosed_bound_sum(
        np.abs(product.astype(np.longdouble) - mapped.astype(np.longdouble)),
        product_error,
        mapped_error,
        _positive_product(gram_error, np.abs(result)),
    )
    # The Neumann series gives an elementwise enclosure of G^-1:
    # |G^-1| <= I + |G-I| + eta**2/(1-eta), since every tail entry is
    # bounded by the corresponding sum of spectral norms.
    unit = np.finfo(np.longdouble).eps / 2
    tail = _upper(np.longdouble(eta) ** 2 / (1 - eta) / (1 - _gamma(5, unit)))
    inverse_bound = _enclosed_bound_sum(np.eye(rank), defect_bound, np.full_like(gram, tail))
    return result, _positive_product(inverse_bound, residual)


def _ssp_component_roots(support, coordinate_map: NDArray, *, refine: bool = False):
    """One common retained-coordinate target, with optional sharper products."""
    from superglm.reml.multi_penalty import _positive_product

    full = support.rank == support.Q_plus.shape[0]
    if not full:
        retained_map, retained_error = _retained_coordinate_map(
            support, coordinate_map, refine=refine
        )
    roots, errors = [], []
    for source, source_error in zip(
        support.component_roots, support.component_root_error_bounds, strict=True
    ):
        if full:
            root, root_error = _context_product(source, coordinate_map, refine=refine)
            root_error = _enclosed_bound_sum(
                root_error, _positive_product(source_error, np.abs(coordinate_map))
            )
        else:
            coordinates, coordinate_error = _context_product(source, support.Q_plus, refine=refine)
            coordinate_error = _enclosed_bound_sum(
                coordinate_error, _positive_product(source_error, np.abs(support.Q_plus))
            )
            root, root_error = _context_product(coordinates, retained_map, refine=refine)
            root_error = _enclosed_bound_sum(
                root_error,
                _positive_product(coordinate_error, np.abs(retained_map)),
                _positive_product(np.abs(coordinates), retained_error),
                _positive_product(coordinate_error, retained_error),
            )
        roots.append(_frozen_array(root))
        errors.append(_frozen_array(root_error))
    return tuple(roots), tuple(errors)


def _near_identity_logdet(gram: NDArray, error: NDArray) -> tuple[float, float]:
    """Trace expansion with an enclosed Frobenius-norm remainder."""
    from superglm.reml.multi_penalty import _gamma, _norm_upper, _upper
    from superglm.reml.penalty_support import PenaltyNumericalError

    defect = gram.astype(np.longdouble) - np.eye(len(gram), dtype=np.longdouble)
    eta = float(_upper(_norm_upper(defect) + _norm_upper(error)))
    if eta >= 1:
        raise PenaltyNumericalError("SSP coordinate volume cannot certify injectivity")
    diagonal = np.diag(defect)
    value = math.fsum(map(float, diagonal))
    # For symmetric D with ||D||_2 <= eta < 1,
    # |log det(I+D) - tr D| <= ||D||_F**2 / (2*(1-eta)).
    wide_eta = np.longdouble(eta)
    bound = (
        np.longdouble(math.fsum(map(float, np.diag(error))))
        + wide_eta**2 / (2 * (1 - wide_eta))
        + np.longdouble(_gamma(len(gram) + 2)) * math.fsum(map(float, np.abs(diagonal)))
    )
    unit = np.finfo(np.longdouble).eps / 2
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    return value, float(_upper((bound + 8 * tiny) / (1 - _gamma(8, unit))))


def _support_coordinate_volume(
    support, coordinate_map: NDArray, *, _refine: bool = False
) -> tuple[float, float]:
    """Bound the volume ratio of one fixed active support under an SSP map.

    For B = C.T Q, the log pseudodeterminant changes by
    log det(B.T B) - log det(Q.T Q). The denominator accounts for the stored
    Q's finite orthogonality. This common-map identity preserves log-weight
    derivatives; it does not introduce independent component-root errors.
    """
    from superglm.reml.multi_penalty import (
        _compensated_dot,
        _finite_double,
        _gamma,
        _matmul_enclosed,
        _positive_product,
        _triangular_solve,
        _upper,
    )
    from superglm.reml.penalty_support import PenaltyNumericalError

    rank = support.rank
    width = coordinate_map.shape[0]
    if rank == 0 or np.array_equal(coordinate_map, np.eye(width)):
        return 0.0, 0.0
    basis = support.Q_plus
    mapped, mapped_error = _matmul_enclosed(coordinate_map.T, basis)
    if _refine:
        for row, column in np.ndindex(mapped.shape):
            value, error = _compensated_dot(coordinate_map[:, row], basis[:, column])
            if error < mapped_error[row, column]:
                mapped[row, column], mapped_error[row, column] = value, error
    _, upper = scipy.linalg.qr(mapped, mode="economic", check_finite=False)
    if upper.shape != (rank, rank) or np.any(np.diag(upper) == 0):
        raise PenaltyNumericalError("SSP coordinate map does not preserve penalty support")
    triangular = _finite_double(
        _triangular_solve(upper[::-1, ::-1], np.eye(rank)[::-1])[::-1],
        "SSP volume preconditioner",
    )
    if np.any(np.diag(triangular) == 0):
        raise PenaltyNumericalError("SSP volume preconditioner is singular")
    whitened, arithmetic = _matmul_enclosed(mapped, triangular)
    action_error = _enclosed_bound_sum(
        arithmetic, _positive_product(mapped_error, np.abs(triangular))
    )
    gram, gram_error = _enclosed_root_gram(whitened, action_error)
    numerator, numerator_error = _near_identity_logdet(gram, gram_error)
    basis_gram, basis_error = _matmul_enclosed(basis.T, basis)
    denominator, denominator_error = _near_identity_logdet(basis_gram, basis_error)
    # T is a chosen checked float64 triangular matrix. This identity needs
    # neither an exact inverse of the QR factor nor a bound on that solve:
    # det((B T).T (B T)) = det(T)**2 det(B.T B).
    terms = [-2 * math.log(abs(value)) for value in np.diag(triangular)]
    value = math.fsum([*terms, numerator, -denominator])
    operation_scale = rank + math.fsum(map(abs, [*terms, numerator, denominator]))
    error = float(
        _enclosed_bound_sum(
            numerator_error,
            denominator_error,
            _upper(np.longdouble(_gamma(4 * rank + 8)) * np.longdouble(operation_scale)),
        )
    )
    rows = sum(len(root) for root in support.component_roots)
    target = _gamma(8 * (rows + width + rank + len(support.component_roots) + 1))
    if error > target * operation_scale:
        if not _refine:
            return _support_coordinate_volume(support, coordinate_map, _refine=True)
        raise PenaltyNumericalError("SSP coordinate volume cannot meet the accuracy contract")
    return value, error


def _active_support_volume_error(support) -> float:
    """Bound the volume omitted by projecting an enclosed fixed-rank target.

    Root-action errors only enclose the target compressed to the selected
    range. Its perpendicular part contributes a weight-independent volume.
    If T spans that range and ||(H T).T (H T)-I|| <= eta < 1, then
    log det(I+L L.T) <= ||L||_F**2 <= ||T||_F**2 ||H(I-P)||_F**2/(1-eta).
    Component scaling preserves both ranges and avoids reciprocal root scales.
    """
    from superglm.reml.multi_penalty import (
        _finite_double,
        _gamma,
        _norm_upper,
        _positive_product,
        _triangular_solve,
        _upper,
    )
    from superglm.reml.penalty_support import PenaltyNumericalError

    basis, rank = support.Q_plus, support.rank
    width = basis.shape[0]
    if rank == 0 or rank == width:
        return 0.0
    unit = np.finfo(np.longdouble).eps / 2
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    roots, errors = [], []
    for root, error in zip(
        support.component_roots, support.component_root_error_bounds, strict=True
    ):
        scale = np.longdouble(np.max(np.abs(root), initial=0.0))
        if scale == 0:
            scale = np.longdouble(1)
        wide = root.astype(np.longdouble) / scale
        normalized = _finite_double(wide, "active support roots")
        bound = (
            error.astype(np.longdouble) / scale
            + _gamma(1, unit) * np.abs(wide)
            + np.abs(wide - normalized.astype(np.longdouble))
            + tiny
        ) / (1 - _gamma(6, unit))
        roots.append(normalized)
        errors.append(_finite_double(_upper(bound), "active support root errors"))
    root, root_error = np.vstack(roots), np.vstack(errors)
    coordinates, coordinate_error = _context_product(root, basis, refine=False)
    coordinate_error = _enclosed_bound_sum(
        coordinate_error, _positive_product(root_error, np.abs(basis))
    )
    _, upper = scipy.linalg.qr(coordinates, mode="economic", check_finite=False)
    if upper.shape != (rank, rank) or np.any(np.diag(upper) == 0):
        raise PenaltyNumericalError("active support volume has unresolved coordinates")
    triangular = _finite_double(
        _triangular_solve(upper[::-1, ::-1], np.eye(rank)[::-1])[::-1],
        "active support volume preconditioner",
    )
    action, action_error = _context_product(coordinates, triangular, refine=False)
    action_error = _enclosed_bound_sum(
        action_error, _positive_product(coordinate_error, np.abs(triangular))
    )
    gram, gram_error = _enclosed_root_gram(action, action_error)
    eta = _norm_upper(_enclosed_bound_sum(np.abs(gram - np.eye(rank)), gram_error))
    if eta >= 1:
        raise PenaltyNumericalError("active support volume cannot certify its projected rank")
    mapped, mapped_error = _context_product(basis, triangular, refine=False)
    map_norm = _norm_upper(_enclosed_bound_sum(np.abs(mapped), mapped_error))
    # The chosen numeric coordinates times Q.T lie exactly in range(Q).
    # Their construction need not be an exact orthogonal projection.
    projected, projection_error = _context_product(coordinates, basis.T, refine=False)
    off = _enclosed_bound_sum(
        np.abs(root.astype(np.longdouble) - projected.astype(np.longdouble)),
        projection_error,
        root_error,
    )
    off_norm = _norm_upper(off)
    volume_error = float(
        _upper(
            ((np.longdouble(map_norm) * off_norm) ** 2 / (1 - np.longdouble(eta)) + tiny)
            / (1 - _gamma(8, unit))
        )
    )
    target = _gamma(8 * (len(root) + width + rank + len(roots) + 1))
    if not np.isfinite(volume_error) or volume_error > target * rank:
        raise PenaltyNumericalError("active support volume cannot meet the accuracy contract")
    return volume_error


def _joint_near_isometry_volume_error(
    row_map: NDArray, *, rank: int, root_rows: int, components: int
) -> float:
    """Bound one common coordinate volume without changing local targets.

    For a block-diagonal fixed local penalty with orthonormal active range U,
    an injective row map A contributes log det(U.T A A.T U). If
    ||A A.T-I|| <= delta < 1, its absolute value is at most
    rank * -log(1-delta) <= rank * delta/(1-delta). This joint Gram includes
    cross-block terms; the volume is independent of positive weights on each
    fixed active set, so all log-weight derivatives are unchanged.
    """
    from superglm.reml.multi_penalty import _gamma, _matmul_enclosed, _norm_upper, _upper
    from superglm.reml.penalty_support import PenaltyNumericalError

    row_map = np.asarray(row_map, dtype=float)
    if (
        row_map.ndim != 2
        or not np.all(np.isfinite(row_map))
        or rank < 0
        or rank > row_map.shape[0]
        or row_map.shape[0] > row_map.shape[1]
    ):
        raise PenaltyNumericalError("finite coefficient map has invalid dimensions")
    if rank == 0:
        return 0.0
    gram, arithmetic = _matmul_enclosed(row_map, row_map.T)
    # Subtract in the wider type and enclose its rounding too. This avoids
    # relying on exact diagonal subtraction when the map is malformed.
    unit = np.finfo(np.longdouble).eps / 2
    tiny = np.nextafter(np.longdouble(0), np.longdouble(1))
    difference = gram.astype(np.longdouble) - np.eye(len(gram), dtype=np.longdouble)
    defect = _enclosed_bound_sum(
        np.abs(difference), arithmetic, _gamma(1, unit) * np.abs(difference) + tiny
    )
    delta = _norm_upper(defect)
    if delta >= 1:
        raise PenaltyNumericalError("finite coefficient map does not preserve penalty support")
    error = float(
        _upper(
            (np.longdouble(rank) * delta / (1 - np.longdouble(delta)) + tiny)
            / (1 - _gamma(8, unit))
        )
    )
    # Use the existing dimension gamma and its minimum rank operation scale
    # for this additional volume certificate, as in the active-support bound.
    dimension = root_rows + row_map.shape[1] + rank + components + 1
    if error > _gamma(8 * dimension) * rank:
        raise PenaltyNumericalError("finite coefficient volume cannot meet the accuracy contract")
    return error


def _component_geometry_key(component: PenaltyComponent) -> tuple:
    return (
        component.name,
        component.group_name,
        component.group_index,
        component.group_sl,
        component.penalty_kind,
        component.repeat_count,
        component.block_width,
    )


def _raw_evidence_value(value):
    """Own exact static values for the one explicit raw-context handoff."""
    if isinstance(value, np.ndarray):
        return (np.ndarray, value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, tuple):
        return (tuple, tuple(_raw_evidence_value(item) for item in value))
    if is_dataclass(value):
        return (
            type(value),
            tuple(
                (item.name, _raw_evidence_value(getattr(value, item.name)))
                for item in fields(value)
            ),
        )
    return (type(value), value)


def _raw_evidence_readonly(value) -> bool:
    if isinstance(value, np.ndarray):
        return not value.flags.writeable
    if isinstance(value, tuple):
        return all(_raw_evidence_readonly(item) for item in value)
    if is_dataclass(value):
        return all(_raw_evidence_readonly(getattr(value, item.name)) for item in fields(value))
    return True


def _raw_support_values(support) -> tuple:
    # The basis-Gram memo is derived from these inputs and has its own token.
    # It is not part of the selected mathematical support or its error ledger.
    return tuple(
        (item.name, getattr(support, item.name))
        for item in fields(support)
        if item.name != "_basis_gram_evidence"
    )


def _raw_penalty_arithmetic() -> tuple:
    from superglm.reml import multi_penalty, penalty_support
    from superglm.solvers import rank

    return (
        multi_penalty._LD,
        multi_penalty._EPS,
        multi_penalty._U_LD,
        multi_penalty._TINY_LD,
        penalty_support._LD,
        penalty_support._EPS,
        _raw_evidence_value(rank.SHARED_RANK_POLICY),
        _raw_evidence_value(multi_penalty.SHARED_RANK_POLICY),
        _raw_evidence_value(penalty_support.SHARED_RANK_POLICY),
        tuple(
            getattr(multi_penalty, name, None)
            for name in (
                "_evaluate_penalty_summary",
                "_evaluate_penalty_geometry",
                "_matmul_enclosed",
                "_reference_root_actions",
                "_direct_candidate",
                "_candidate_product",
                "_wide_product",
                "_dyadic_product",
                "_dyadic_slices",
                "_positive_product",
                "_gamma",
                "_upper",
                "_finite_double",
            )
        ),
        penalty_support._penalty_support,
        penalty_support._penalty_support_from_roots,
        penalty_support._component_root,
        penalty_support.decompose_gram,
    )


def _raw_family_inputs(grouped: Sequence[PenaltyComponent]) -> tuple:
    return tuple(
        (
            _component_geometry_key(component),
            _raw_evidence_value(component.component_type),
            _raw_evidence_value(component.lambda_policy),
            _raw_evidence_value(np.asarray(component.omega_raw)),
        )
        for component in grouped
    )


@dataclass(frozen=True)
class _RawPenaltyFamilyReceipt:
    support: object
    inputs: tuple
    support_values: tuple
    arithmetic: tuple

    @classmethod
    def capture(cls, support, grouped):
        return cls(
            support,
            _raw_family_inputs(grouped),
            _raw_evidence_value(_raw_support_values(support)),
            _raw_penalty_arithmetic(),
        )

    def matches(self, geometry, source, target) -> bool:
        values = _raw_support_values(self.support)
        return bool(
            geometry.support is self.support
            and self.arithmetic == _raw_penalty_arithmetic()
            and self.inputs == _raw_family_inputs(source) == _raw_family_inputs(target)
            and _raw_evidence_readonly(values)
            and self.support_values == _raw_evidence_value(values)
        )


def _raw_summary_values(summary) -> tuple:
    return tuple(
        (item.name, getattr(summary, item.name))
        for item in fields(summary)
        if item.name != "_support"
    )


@dataclass(frozen=True)
class _RawPenaltySummaryReceipt:
    summary: object
    support: object
    weights: tuple[float, ...]
    values: tuple
    arithmetic: tuple

    @classmethod
    def capture(cls, summary, weights):
        return cls(
            summary,
            summary._support,
            weights,
            _raw_evidence_value(_raw_summary_values(summary)),
            _raw_penalty_arithmetic(),
        )

    def matches(self, geometry) -> bool:
        values = _raw_summary_values(self.summary)
        return bool(
            geometry.last_weights == self.weights
            and geometry.last_evaluation is not None
            and geometry.last_evaluation[0] is self.summary
            and geometry.support is self.support is self.summary._support
            and all(value > 0 for value in self.weights)
            and np.array_equal(self.summary._input_lambdas, self.weights)
            and self.arithmetic == _raw_penalty_arithmetic()
            and _raw_evidence_readonly(values)
            and self.values == _raw_evidence_value(values)
        )


def _reusable_raw_geometry(source, target):
    if not source:
        return None
    indices = _group_penalties(list(source)).get(target[0].group_name, [])
    originals = [source[index] for index in indices]
    if not originals:
        return None
    geometry = _context_geometry(originals)
    if (
        geometry is None
        or geometry.coordinate_map is None
        or geometry.raw_family is None
        or not geometry.raw_family.matches(geometry, originals, target)
    ):
        return None
    return geometry


@dataclass
class _PenaltyGroupGeometry:
    """Immutable context inputs and one retained weight/volume evaluation."""

    matrices: tuple[NDArray, ...]
    keys: tuple[tuple, ...]
    repeat: int
    support: object | None = None
    coordinate_map: NDArray | None = None
    matrix_error_bounds: tuple[NDArray, ...] = ()
    ssp_roots: tuple[NDArray, ...] = ()
    ssp_root_errors: tuple[NDArray, ...] = ()
    ssp_refined: bool = False
    face_activity: tuple[bool, ...] | None = None
    face_support: object | None = None
    face_logdet_error: float = 0.0
    last_weights: tuple[float, ...] | None = None
    last_evaluation: tuple | None = None
    volume_activity: tuple[bool, ...] | None = None
    volume: tuple[float, float] | None = None
    raw_family: _RawPenaltyFamilyReceipt | None = None
    raw_summary: _RawPenaltySummaryReceipt | None = None

    def __getstate__(self) -> dict:
        # Function identities certify this fit's arithmetic. Pickle resolves
        # names in the loading process, so it cannot retain that receipt.
        state = self.__dict__.copy()
        state["raw_family"] = state["raw_summary"] = None
        return state

    def get_support(self):
        from superglm.reml.penalty_support import _penalty_support

        if self.support is None:
            self.support = _penalty_support(self.matrices)
        return self.support

    def evaluate(self, values: NDArray):
        from superglm.reml.multi_penalty import _evaluate_penalty_summary
        from superglm.reml.penalty_support import PenaltyNumericalError

        values = np.asarray(values, dtype=float)
        if (
            values.shape != (len(self.matrices),)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError("smoothing parameters must be finite and non-negative")
        weights = tuple(map(float, values))
        if weights == self.last_weights:
            return self.last_evaluation
        if self.coordinate_map is not None and np.any(values == 0):
            try:
                result = self._evaluate_face(values)
            except PenaltyNumericalError:
                if self.ssp_refined:
                    raise
                self.ssp_roots, self.ssp_root_errors = _ssp_component_roots(
                    self.get_support(), self.coordinate_map, refine=True
                )
                self.ssp_refined = True
                self.face_activity = self.face_support = None
                result = self._evaluate_face(values)
            evaluation = (result, 0.0, self.face_logdet_error)
            self.last_weights, self.last_evaluation = weights, evaluation
            self.raw_summary = None
            return evaluation
        result = _evaluate_penalty_summary(self.get_support(), values)
        volume = (0.0, 0.0)
        if self.coordinate_map is not None:
            activity = tuple(value > 0 for value in weights)
            if activity == self.volume_activity:
                volume = self.volume
            else:
                volume = _support_coordinate_volume(result._support, self.coordinate_map)
                self.volume_activity, self.volume = activity, volume
        evaluation = (result, *volume)
        # A refused candidate never replaces the last complete evaluation.
        if self.raw_family is not None:
            self.raw_summary = _RawPenaltySummaryReceipt.capture(result, weights)
        self.last_weights, self.last_evaluation = weights, evaluation
        return evaluation

    def _evaluate_face(self, values: NDArray):
        from superglm.reml.multi_penalty import _evaluate_penalty_summary
        from superglm.reml.penalty_support import PenaltyNumericalError, _penalty_support_from_roots

        activity = tuple(value > 0 for value in values)
        if activity != self.face_activity:
            raw = self.get_support()
            width = raw.Q_plus.shape[0]

            def active_arrays(arrays):
                return [
                    array if active else np.empty((0, width))
                    for array, active in zip(arrays, activity, strict=True)
                ]

            raw_roots = active_arrays(raw.component_roots)
            raw_active = _penalty_support_from_roots(
                raw_roots,
                resolution_limited=raw.component_resolution_limited,
                input_error_bounds=[np.zeros_like(root) for root in raw_roots],
            )
            roots = active_arrays(self.ssp_roots)
            selected = _penalty_support_from_roots(
                roots,
                resolution_limited=raw.component_resolution_limited,
                input_error_bounds=[np.zeros_like(root) for root in roots],
            )
            if selected.rank != raw_active.rank:
                raise PenaltyNumericalError("SSP active support cannot preserve the raw rank")
            # Enclose the original fixed SSP roots, not another projected
            # target: charge the entire selected-root displacement as well as
            # each original root's construction error.
            errors = tuple(
                _frozen_array(_enclosed_bound_sum(error, displacement))
                for error, displacement in zip(
                    active_arrays(self.ssp_root_errors),
                    selected.support_projection_bounds,
                    strict=True,
                )
            )
            selected = replace(selected, component_root_error_bounds=errors)
            volume_error = _active_support_volume_error(selected)
            self.face_support, self.face_logdet_error = selected, volume_error
            self.face_activity = activity
        # Inactive components have no root rows. Unit carrier weights retain
        # their ordered zero derivatives without invoking a second projection.
        return _evaluate_penalty_summary(self.face_support, np.where(values > 0, values, 1.0))


def _context_geometry(grouped: list[PenaltyComponent]) -> _PenaltyGroupGeometry | None:
    geometry = getattr(grouped[0], "_penalty_geometry", None)
    if geometry is None or len(grouped) != len(geometry.matrices):
        return None
    if any(
        getattr(component, "_penalty_geometry", None) is not geometry
        or component.omega_ssp is not matrix
        or matrix.flags.writeable
        or _component_geometry_key(component) != key
        for component, matrix, key in zip(grouped, geometry.matrices, geometry.keys, strict=True)
    ):
        return None
    return geometry


def _attach_context_geometry(
    grouped: list[PenaltyComponent],
    *,
    support=None,
    coordinate_map=None,
    matrix_errors=(),
    ssp_roots=(),
    ssp_errors=(),
    raw_family=None,
) -> None:
    if not grouped or any(component.omega_ssp is None for component in grouped):
        return
    for component in grouped:
        component.omega_ssp = _frozen_array(component.omega_ssp)
    first = grouped[0]
    repeat = (
        first.repeat_count - (first.penalty_kind == "sum_to_zero")
        if first.penalty_kind in {"repeated", "sum_to_zero"}
        else 1
    )
    geometry = _PenaltyGroupGeometry(
        tuple(component.omega_ssp for component in grouped),
        tuple(_component_geometry_key(component) for component in grouped),
        repeat,
        support,
        None if coordinate_map is None else _frozen_array(coordinate_map),
        tuple(_frozen_array(bound) for bound in matrix_errors),
        ssp_roots,
        ssp_errors,
        raw_family=raw_family,
    )
    if coordinate_map is not None:
        # Rank queries also rely on injectivity; certify it before exposing
        # the raw rank through this context, independently of positive weights.
        geometry.volume = _support_coordinate_volume(support, geometry.coordinate_map)
        geometry.volume_activity = tuple(True for _ in grouped)
    for component in grouped:
        component._penalty_geometry = geometry


def _rebind_penalty_context(
    source: Sequence[PenaltyComponent], copied: Sequence[PenaltyComponent]
) -> None:
    """Preserve a complete local family through predictor qualification and placement.

    Moving a coefficient block is an isometric injection. Its selected raw
    support, local SSP map and arithmetic evidence remain valid when the local
    matrices are copied exactly. Each target family gets its own mutable owner;
    only already-owned immutable geometry and volume evidence are shared.
    """
    if len(source) != len(copied):
        raise ValueError("penalty context copy requires matching component counts")
    target_ids = {id(component) for component in copied}
    if len(target_ids) != len(copied) or target_ids.intersection(map(id, source)):
        raise ValueError("penalty context copy requires distinct target components")
    target_groups = _group_penalties(list(copied))
    pending = []
    for indices in _group_penalties(list(source)).values():
        originals = [source[index] for index in indices]
        geometry = _context_geometry(originals)
        if geometry is None:
            # Manual descriptors, incomplete families and invalidated contexts
            # carry no transferable authority.
            continue
        targets = [copied[index] for index in indices]
        first = targets[0]
        if (
            isinstance(first.group_index, bool)
            or not isinstance(first.group_index, int | np.integer)
            or first.group_index < 0
            or target_groups[first.group_name] != indices
            or any(
                (item.group_name, item.group_index, item.group_sl)
                != (first.group_name, first.group_index, first.group_sl)
                for item in targets
            )
        ):
            raise ValueError("copied penalty family has an inconsistent coefficient block")
        for original, target in zip(originals, targets, strict=True):
            suffix = (
                "wiggle"
                if original.name == original.group_name
                else original.name.removeprefix(f"{original.group_name}:")
            )
            block = target.group_sl
            if (
                (
                    target.name != f"{target.group_name}#{suffix}"
                    and _component_geometry_key(target) != _component_geometry_key(original)
                )
                or block.start is None
                or block.stop is None
                or block.start < 0
                or block.stop - block.start != original.group_sl.stop - original.group_sl.start
                or block.step != original.group_sl.step
                or (target.penalty_kind, target.repeat_count, target.block_width)
                != (original.penalty_kind, original.repeat_count, original.block_width)
                or target.omega_ssp is None
                or not np.array_equal(target.omega_ssp, original.omega_ssp)
            ):
                raise ValueError("copied penalty component changed its ordered local geometry")
        for index, other in enumerate(copied):
            if index not in indices and (
                other.group_index == first.group_index
                or max(first.group_sl.start, other.group_sl.start)
                < min(first.group_sl.stop, other.group_sl.stop)
            ):
                raise ValueError("copied penalty families overlap in the target layout")
        pending.append((geometry, targets))
    # Validate every family before modifying any target descriptor.
    for geometry, targets in pending:
        for component in targets:
            component.omega_ssp = _frozen_array(component.omega_ssp)
        owner = _PenaltyGroupGeometry(
            matrices=tuple(component.omega_ssp for component in targets),
            keys=tuple(_component_geometry_key(component) for component in targets),
            repeat=geometry.repeat,
            support=geometry.support,
            coordinate_map=geometry.coordinate_map,
            matrix_error_bounds=geometry.matrix_error_bounds,
            ssp_roots=geometry.ssp_roots,
            ssp_root_errors=geometry.ssp_root_errors,
            ssp_refined=geometry.ssp_refined,
            volume_activity=geometry.volume_activity,
            volume=geometry.volume,
        )
        for component in targets:
            component._penalty_geometry = owner


def _snapshot_penalty_context(
    components: Sequence[PenaltyComponent],
) -> tuple[PenaltyComponent, ...]:
    """Own declared component arrays while preserving complete selected targets."""
    copied = tuple(
        replace(
            component,
            omega_raw=None if component.omega_raw is None else _frozen_array(component.omega_raw),
            omega_ssp=None if component.omega_ssp is None else _frozen_array(component.omega_ssp),
            eigvals_omega=(
                None if component.eigvals_omega is None else _frozen_array(component.eigvals_omega)
            ),
        )
        for component in components
    )
    _rebind_penalty_context(components, copied)
    return copied


def _penalty_component_omega_ssp(
    component: PenaltyComponent,
    group_matrix: GroupMatrix | None = None,
) -> NDArray | None:
    """Return a dense solver-space penalty only for non-identity components."""
    if component.penalty_kind == "identity":
        return None
    if component.omega_ssp is not None:
        return component.omega_ssp
    if group_matrix is None or component.omega_raw is None:
        raise ValueError(f"Dense penalty component {component.name!r} has no solver-space matrix.")
    return group_matrix.R_inv.T @ component.omega_raw @ group_matrix.R_inv


def _repeated_penalty_geometry(
    component: PenaltyComponent,
) -> tuple[int, int]:
    """Return and validate ``(repeat_count, local_width)`` metadata."""
    if component.penalty_kind != "repeated":
        raise ValueError(f"Penalty component {component.name!r} is not repeated.")
    repeat_count = int(component.repeat_count)
    block_width = component.block_width
    if repeat_count < 1 or block_width is None or int(block_width) < 1:
        raise ValueError(f"Repeated penalty component {component.name!r} has invalid geometry.")
    block_width = int(block_width)
    group_width = component.group_sl.stop - component.group_sl.start
    if repeat_count * block_width != group_width:
        raise ValueError(
            f"Repeated penalty component {component.name!r} geometry "
            f"{repeat_count} x {block_width} does not match group width {group_width}."
        )
    return repeat_count, block_width


def _sum_to_zero_penalty_geometry(
    component: PenaltyComponent,
) -> tuple[int, int]:
    """Return and validate ``(raw_level_count, local_width)`` metadata."""
    if component.penalty_kind != "sum_to_zero":
        raise ValueError(f"Penalty component {component.name!r} is not sum-to-zero.")
    n_levels = int(component.repeat_count)
    block_width = component.block_width
    if n_levels < 2 or block_width is None or int(block_width) < 1:
        raise ValueError(f"Sum-to-zero penalty component {component.name!r} has invalid geometry.")
    block_width = int(block_width)
    group_width = component.group_sl.stop - component.group_sl.start
    if (n_levels - 1) * block_width != group_width:
        raise ValueError(
            f"Sum-to-zero penalty component {component.name!r} geometry "
            f"({n_levels} - 1) x {block_width} does not match group width {group_width}."
        )
    return n_levels, block_width


def penalty_component_dense_matrix(
    component: PenaltyComponent,
    group_matrix: GroupMatrix | None = None,
) -> NDArray:
    """Materialize one component for an explicitly dense reference path."""
    width = component.group_sl.stop - component.group_sl.start
    if component.penalty_kind == "identity":
        return np.eye(width, dtype=np.float64)
    omega = np.asarray(
        _penalty_component_omega_ssp(component, group_matrix),
        dtype=np.float64,
    )
    if component.penalty_kind == "sum_to_zero":
        n_levels, block_width = _sum_to_zero_penalty_geometry(component)
        if omega.shape != (block_width, block_width):
            raise ValueError(
                f"Sum-to-zero penalty component {component.name!r} has local shape "
                f"{omega.shape}; expected {(block_width, block_width)}."
            )
        return sum_to_zero_penalty(omega, n_levels)
    if component.penalty_kind == "repeated":
        repeat_count, block_width = _repeated_penalty_geometry(component)
        if omega.shape != (block_width, block_width):
            raise ValueError(
                f"Repeated penalty component {component.name!r} has local shape "
                f"{omega.shape}; expected {(block_width, block_width)}."
            )
        return np.kron(np.eye(repeat_count, dtype=np.float64), omega)
    if omega.shape != (width, width):
        raise ValueError(
            f"Dense penalty component {component.name!r} has shape {omega.shape}; "
            f"expected {(width, width)}."
        )
    return omega


def penalty_component_quadratic(
    component: PenaltyComponent,
    beta_group: NDArray,
    group_matrix: GroupMatrix | None = None,
) -> float:
    """Return ``beta.T @ Omega @ beta`` without materializing identity penalties."""
    beta = np.asarray(beta_group, dtype=np.float64)
    if component.penalty_kind == "identity":
        return float(beta @ beta)
    omega = _penalty_component_omega_ssp(component, group_matrix)
    if component.penalty_kind == "sum_to_zero":
        n_levels, block_width = _sum_to_zero_penalty_geometry(component)
        free = beta.reshape(n_levels - 1, block_width)
        raw = expand_sum_to_zero_blocks(free)
        return float(np.einsum("ki,ij,kj->", raw, omega, raw, optimize=True))
    if component.penalty_kind == "repeated":
        repeat_count, block_width = _repeated_penalty_geometry(component)
        blocks = beta.reshape(repeat_count, block_width)
        return float(np.einsum("ki,ij,kj->", blocks, omega, blocks, optimize=True))
    return float(beta @ omega @ beta)


def penalty_component_matvec(
    component: PenaltyComponent,
    beta_group: NDArray,
    group_matrix: GroupMatrix | None = None,
) -> NDArray:
    """Return ``Omega @ beta`` using the component's compact representation."""
    beta = np.asarray(beta_group, dtype=np.float64)
    if component.penalty_kind == "identity":
        return beta.copy()
    omega = _penalty_component_omega_ssp(component, group_matrix)
    if component.penalty_kind == "sum_to_zero":
        n_levels, block_width = _sum_to_zero_penalty_geometry(component)
        free = beta.reshape(n_levels - 1, block_width)
        raw_product = expand_sum_to_zero_blocks(free) @ omega.T
        return adjoint_sum_to_zero_blocks(raw_product).ravel()
    if component.penalty_kind == "repeated":
        repeat_count, block_width = _repeated_penalty_geometry(component)
        blocks = beta.reshape(repeat_count, block_width)
        return np.asarray(blocks @ omega.T, dtype=np.float64).ravel()
    return omega @ beta


def penalty_component_trace(
    component: PenaltyComponent,
    inverse_block_or_diagonal: NDArray,
    group_matrix: GroupMatrix | None = None,
) -> float:
    """Return ``trace(H^-1_jj Omega)`` from a selected block or identity diagonal."""
    inverse = np.asarray(inverse_block_or_diagonal, dtype=np.float64)
    if component.penalty_kind == "identity":
        if inverse.ndim == 1:
            return float(np.sum(inverse))
        if inverse.ndim == 2:
            return float(np.trace(inverse))
        raise ValueError("Identity penalty trace requires an inverse diagonal or square block.")
    if component.penalty_kind == "sum_to_zero":
        n_levels, block_width = _sum_to_zero_penalty_geometry(component)
        omega = _penalty_component_omega_ssp(component, group_matrix)
        if inverse.ndim != 2 or inverse.shape != (
            (n_levels - 1) * block_width,
            (n_levels - 1) * block_width,
        ):
            raise ValueError("Sum-to-zero penalty trace requires its full selected inverse block.")
        contrast_gram = sum_to_zero_contrast(n_levels).T @ sum_to_zero_contrast(n_levels)
        blocks = inverse.reshape(
            n_levels - 1,
            block_width,
            n_levels - 1,
            block_width,
        )
        return float(
            np.einsum(
                "aibj,ba,ji->",
                blocks,
                contrast_gram,
                omega,
                optimize=True,
            )
        )
    if component.penalty_kind == "repeated":
        repeat_count, block_width = _repeated_penalty_geometry(component)
        omega = _penalty_component_omega_ssp(component, group_matrix)
        if inverse.ndim == 1:
            if inverse.shape != (repeat_count * block_width,):
                raise ValueError("Repeated penalty inverse diagonal has the wrong width.")
            off_diagonal = omega - np.diag(np.diag(omega))
            if not np.allclose(off_diagonal, 0.0, atol=1e-14):
                raise ValueError(
                    "A repeated non-diagonal penalty trace requires a selected inverse block."
                )
            return float(np.sum(inverse.reshape(repeat_count, block_width) * np.diag(omega)))
        if inverse.shape != (
            repeat_count * block_width,
            repeat_count * block_width,
        ):
            raise ValueError("Repeated penalty trace requires its full selected inverse block.")
        blocks = inverse.reshape(
            repeat_count,
            block_width,
            repeat_count,
            block_width,
        )
        return float(
            sum(np.trace(blocks[level, :, level, :] @ omega) for level in range(repeat_count))
        )
    if inverse.ndim != 2:
        raise ValueError("Dense penalty trace requires a selected inverse block.")
    omega = _penalty_component_omega_ssp(component, group_matrix)
    return float(np.trace(inverse @ omega))


def total_penalty_quadratic(
    beta: NDArray,
    lambdas: float | dict[str, float],
    penalties: list[PenaltyComponent],
    group_matrices: list[GroupMatrix],
) -> float:
    """Return the full weighted penalty quadratic from compact components."""
    total = 0.0
    for component in penalties:
        lam = lambdas[component.name] if isinstance(lambdas, dict) else lambdas
        if lam == 0:
            continue
        group_matrix = (
            group_matrices[component.group_index]
            if 0 <= component.group_index < len(group_matrices)
            else None
        )
        total += float(lam) * penalty_component_quadratic(
            component,
            np.asarray(beta)[component.group_sl],
            group_matrix,
        )
    return total


def total_penalty_matvec(
    beta: NDArray,
    lambdas: float | dict[str, float],
    penalties: list[PenaltyComponent],
    group_matrices: list[GroupMatrix],
) -> NDArray:
    """Apply the full weighted penalty from compact components."""
    values = np.asarray(beta, dtype=np.float64)
    product = np.zeros_like(values)
    for component in penalties:
        lam = lambdas[component.name] if isinstance(lambdas, dict) else lambdas
        if lam == 0:
            continue
        group_matrix = (
            group_matrices[component.group_index]
            if 0 <= component.group_index < len(group_matrices)
            else None
        )
        product[component.group_sl] += float(lam) * penalty_component_matvec(
            component,
            values[component.group_sl],
            group_matrix,
        )
    return product


def _extract_tensor_marginal_eigvals(
    omega: NDArray,
    p1: int,
    p2: int,
    *,
    tol: float = 1e-10,
) -> tuple[str | None, NDArray | None]:
    """Recover marginal eigenvalues from a tensor penalty component if possible."""
    q = p1 * p2
    if omega.shape != (q, q):
        return None, None

    norm = max(float(np.linalg.norm(omega)), 1e-300)
    omega4 = omega.reshape(p1, p2, p1, p2)

    left = 0.5 * (omega4[:, 0, :, 0] + omega4[:, 0, :, 0].T)
    right = 0.5 * (omega4[0, :, 0, :] + omega4[0, :, 0, :].T)
    left_err = float(np.linalg.norm(np.kron(left, np.eye(p2)) - omega) / norm)
    right_err = float(np.linalg.norm(np.kron(np.eye(p1), right) - omega) / norm)

    if left_err <= tol and left_err <= right_err:
        return "left", np.clip(np.linalg.eigvalsh(left), 0.0, None)
    if right_err <= tol and right_err <= left_err:
        return "right", np.clip(np.linalg.eigvalsh(right), 0.0, None)
    return None, None


def _tensor_marginal_rank_logdet(
    gm: GroupMatrix,
    omega_raw: NDArray,
    *,
    eps_thresh: float,
) -> tuple[float, float, NDArray] | None:
    """Fast spectral summary for unprojected tensor marginal penalties."""
    if not isinstance(gm, DiscretizedTensorGroupMatrix):
        return None
    if getattr(gm, "projection", None) is not None:
        return None

    r_inv = getattr(gm, "R_inv", None)
    if r_inv is None or r_inv.ndim != 2 or r_inv.shape[0] != r_inv.shape[1]:
        return None
    if not np.array_equal(r_inv, np.eye(r_inv.shape[0], dtype=r_inv.dtype)):
        return None

    p1 = int(gm.B1_unique_t.shape[1])
    p2 = int(gm.B2_unique_t.shape[1])
    side, eigvals = _extract_tensor_marginal_eigvals(omega_raw, p1, p2)
    if side is None or eigvals is None:
        return None

    eigvals = np.asarray(eigvals, dtype=np.float64)
    thresh = eps_thresh * max(float(eigvals.max()), 1e-12) if eigvals.size else 0.0
    pos = eigvals[eigvals > thresh]
    repeat = p2 if side == "left" else p1
    rank = float(pos.size * repeat)
    log_det = float(repeat * np.sum(np.log(np.maximum(pos, 1e-300)))) if pos.size else 0.0
    pos_eigvals = np.sort(np.repeat(pos, repeat))[::-1] if pos.size else np.array([])
    return rank, log_det, pos_eigvals


def _penalty_group_cache_key(index: int, group: GroupSlice, gm: GroupMatrix) -> tuple:
    """Return a cache key for penalty components tied to one fixed solver basis."""
    omega_components = tuple(
        (suffix, id(omega_j)) for suffix, omega_j in (getattr(gm, "omega_components", None) or ())
    )
    return (
        "penalty_components",
        int(index),
        group.name,
        group.start,
        group.end,
        gm,
        id(getattr(gm, "R_inv", None)),
        id(getattr(gm, "omega", None)),
        id(getattr(gm, "projection", None)),
        id(getattr(gm, "component_types", None)),
        id(getattr(gm, "lambda_policies", None)),
        omega_components,
    )


def _can_cache_penalty_group(gm: GroupMatrix) -> bool:
    """Return whether a group has a lambda-invariant solver penalty basis."""
    return (
        isinstance(gm, DiscretizedTensorGroupMatrix)
        and getattr(gm, "omega_components", None) is not None
        and getattr(gm, "projection", None) is None
    )


def _tensor_pair_summary_cache_key(
    group_name: str,
    gm: DiscretizedTensorGroupMatrix,
    penalties: list[PenaltyComponent],
    p1: int,
    p2: int,
) -> tuple:
    """Return a cache key for one tensor pair logdet summary."""
    return (
        "tensor_pair_logdet_summary",
        group_name,
        gm,
        int(gm.tensor_id),
        p1,
        p2,
        tuple((pc.name, id(pc.omega_ssp), id(pc.omega_raw)) for pc in penalties),
    )


def build_tensor_pair_logdet_summaries(
    group_matrices: list[GroupMatrix],
    penalties: list[PenaltyComponent],
    cache: dict | None = None,
) -> dict[str, TensorPairLogdetSummary]:
    """Build closed-form tensor summaries for eligible shared tensor penalty pairs."""
    summaries: dict[str, TensorPairLogdetSummary] = {}
    summary_cache = None if cache is None else cache.setdefault("tensor_pair_logdet_summaries", {})
    for group_name, indices in _group_penalties(penalties).items():
        if len(indices) != 2:
            continue
        pcs = [penalties[i] for i in indices]
        group_index = pcs[0].group_index
        if any(pc.group_index != group_index for pc in pcs[1:]):
            continue
        if any(pc.group_sl != pcs[0].group_sl for pc in pcs[1:]):
            continue

        gm = group_matrices[group_index]
        if not isinstance(gm, DiscretizedTensorGroupMatrix):
            continue
        if getattr(gm, "projection", None) is not None:
            # Once a tensor block has been projected (e.g. parent-side or
            # global side constraints), the penalty pair is no longer a simple
            # separable Kronecker pair in solver space. Fall back to the
            # generic multi-penalty algebra rather than using the closed-form
            # tensor shortcut.
            continue
        tensor_id = getattr(gm, "tensor_id", None)
        if tensor_id is None:
            continue

        p1 = int(gm.B1_unique_t.shape[1])
        p2 = int(gm.B2_unique_t.shape[1])
        cache_key = _tensor_pair_summary_cache_key(group_name, gm, pcs, p1, p2)
        if summary_cache is not None and cache_key in summary_cache:
            cached = summary_cache[cache_key]
            if cached is not None:
                summaries[group_name] = cached
            continue

        extracted: dict[str, tuple[str, NDArray]] = {}
        for pc in pcs:
            omega = pc.omega_ssp
            if omega is None:
                continue
            side, eigvals = _extract_tensor_marginal_eigvals(omega, p1, p2)
            if side is None or eigvals is None:
                extracted.clear()
                break
            extracted[side] = (pc.name, eigvals)

        if set(extracted) != {"left", "right"}:
            if summary_cache is not None:
                summary_cache[cache_key] = None
            continue

        left_name, left_eigvals = extracted["left"]
        right_name, right_eigvals = extracted["right"]
        summary = TensorPairLogdetSummary(
            group_name=group_name,
            tensor_id=int(tensor_id),
            lambda_names=(left_name, right_name),
            eigvals_left=left_eigvals,
            eigvals_right=right_eigvals,
        )
        summaries[group_name] = summary
        if summary_cache is not None:
            summary_cache[cache_key] = summary
    return summaries


def evaluate_tensor_pair_logdet_summaries(
    summaries: dict[str, TensorPairLogdetSummary],
    lambdas: dict[str, float],
) -> dict[str, TensorPairLogdetEvaluation]:
    """Evaluate cached tensor summaries for one lambda dictionary."""
    evaluations: dict[str, TensorPairLogdetEvaluation] = {}
    unit = np.finfo(float).eps / 2.0
    eps_thresh = np.finfo(float).eps ** (2 / 3)

    def marginal_logs(values: NDArray, weight: float) -> tuple[NDArray, NDArray, NDArray]:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("tensor marginal spectra must be finite and non-negative")
        logs = np.full(values.shape, -np.inf)
        errors = np.zeros(values.shape)
        positive = values > eps_thresh * float(np.max(values, initial=0.0))
        if weight > 0.0:
            # The unweighted marginal representative fixes support. No weighted
            # cutoff is allowed, including when a weighted eigenvalue overflows.
            log_weight = math.log(weight)
            log_values = np.log(values[positive])
            logs[positive] = log_weight + log_values
            errors[positive] = (
                4 * unit * (1 + abs(log_weight) + np.abs(log_values) + np.abs(logs[positive]))
            )
        return logs, errors, positive

    for group_name, summary in summaries.items():
        left_name, right_name = summary.lambda_names
        lam_left = float(lambdas.get(left_name, 1.0))
        lam_right = float(lambdas.get(right_name, 1.0))
        if any(not math.isfinite(value) or value < 0 for value in (lam_left, lam_right)):
            raise ValueError("smoothing parameters must be finite and non-negative")
        left, left_error, left_positive = marginal_logs(summary.eigvals_left, lam_left)
        right, right_error, right_positive = marginal_logs(summary.eigvals_right, lam_right)
        cell_log = np.logaddexp(left[:, None], right[None, :])
        active = np.isfinite(cell_log)
        left_log = np.broadcast_to(left[:, None], cell_log.shape)[active]
        right_log = np.broadcast_to(right[None, :], cell_log.shape)[active]
        retained = cell_log[active]
        left_bound = np.broadcast_to(left_error[:, None], cell_log.shape)[active]
        right_bound = np.broadcast_to(right_error[None, :], cell_log.shape)[active]
        cell_bound = np.maximum(left_bound, right_bound) + 4 * unit * (1 + np.abs(retained))
        left_fraction = np.exp(left_log - retained)
        right_fraction = np.exp(right_log - retained)
        # A one-sided cell contributes exactly one to its active derivative.
        left_fraction[~np.isfinite(right_log)] = 1.0
        right_fraction[~np.isfinite(left_log)] = 1.0

        def fraction_bound(logs, fractions, input_bound):
            bound = np.zeros(fractions.shape)
            finite = np.isfinite(logs)
            perturbation = (
                input_bound[finite]
                + cell_bound[finite]
                + unit * np.abs(logs[finite] - retained[finite])
            )
            bound[finite] = fractions[finite] * (np.expm1(perturbation) + 4 * unit)
            bound[finite] += np.nextafter(0.0, 1.0)
            return bound

        left_fraction_bound = fraction_bound(left_log, left_fraction, left_bound)
        right_fraction_bound = fraction_bound(right_log, right_fraction, right_bound)
        cross_terms = left_fraction * right_fraction
        cross = math.fsum(cross_terms)
        logdet = math.fsum(retained)
        grad = {left_name: math.fsum(left_fraction), right_name: math.fsum(right_fraction)}
        hess = {
            (left_name, left_name): cross,
            (right_name, right_name): cross,
            (left_name, right_name): -cross,
            (right_name, left_name): -cross,
        }
        rank = int(np.count_nonzero(active))
        # logaddexp is 1-Lipschitz in the infinity norm. Exp propagates an
        # input perturbation d as expm1(d); the product bound expands both
        # factors. The 4u elementary-function allowance is our float64 libm
        # policy. These bounds concern the selected marginal spectra only.
        inflation = 1.0 / (1.0 - (16 + 4 * rank) * unit)
        logdet_error = inflation * (math.fsum(cell_bound) + 2 * unit * abs(logdet))
        gradient_error = {
            left_name: inflation * (math.fsum(left_fraction_bound) + 2 * unit * grad[left_name]),
            right_name: inflation * (math.fsum(right_fraction_bound) + 2 * unit * grad[right_name]),
        }
        cross_error = inflation * (
            math.fsum(
                left_fraction * right_fraction_bound
                + right_fraction * left_fraction_bound
                + left_fraction_bound * right_fraction_bound
                + 2 * unit * cross_terms
            )
            + 2 * unit * cross
        )

        evaluations[group_name] = TensorPairLogdetEvaluation(
            group_name=group_name,
            tensor_id=summary.tensor_id,
            lambda_names=summary.lambda_names,
            logdet_s_plus=logdet,
            rank=rank,
            gradient=grad,
            hessian=hess,
            logdet_error=logdet_error,
            gradient_error=gradient_error,
            hessian_error={key: cross_error for key in hess},
            support_rank=int(np.count_nonzero(left_positive[:, None] | right_positive[None, :])),
        )
    return evaluations


def resolve_component_lambda(
    lambda2: float | dict[str, float],
    group_name: str,
    suffix: str,
) -> float:
    """Resolve the smoothing weight for one penalty component of a group.

    Dict lambdas are keyed by component name (``"<group>:<suffix>"``, the
    naming used by ``PenaltyComponent`` and REML-fitted lambdas). A
    group-wide key acts as the default for every component not listed by
    its full name, so ``{"a:b": 5.0, "a:b:margin_a": 2.0}`` resolves
    ``margin_a`` to 2.0 and every other component of ``a:b`` to 5.0.
    Components matched by neither key resolve to 0.0.
    """
    if not isinstance(lambda2, dict):
        return float(lambda2)
    return float(lambda2.get(f"{group_name}:{suffix}", lambda2.get(group_name, 0.0)))


def build_penalty_matrix(
    group_matrices: list[GroupMatrix],
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
    p: int,
    reml_penalties: list[PenaltyComponent] | None = None,
) -> NDArray:
    """Build the block-diagonal penalty matrix ``S`` in solver coordinates.

    This is the shared penalty-assembly contract used by REML objective and
    optimizer code.  It was lifted out of ``solvers.irls_direct`` so REML
    modules no longer need to reach through a solver-private helper.

    When ``reml_penalties`` is supplied, its components are authoritative for
    every group index they represent.  The SCOP group fallback below covers
    only SCOP groups omitted from that component list.
    """
    S = np.zeros((p, p))

    if reml_penalties is not None:
        represented_group_indices = {pc.group_index for pc in reml_penalties}
        for pc in reml_penalties:
            gm = group_matrices[pc.group_index]
            lam = lambda2[pc.name] if isinstance(lambda2, dict) else lambda2
            if lam == 0:
                continue
            if pc.penalty_kind == "identity":
                diagonal_indices = np.arange(pc.group_sl.start, pc.group_sl.stop)
                S[diagonal_indices, diagonal_indices] += lam
                continue
            if pc.penalty_kind in ("repeated", "sum_to_zero"):
                S[pc.group_sl, pc.group_sl] += lam * penalty_component_dense_matrix(pc, gm)
                continue
            omega_ssp = (
                pc.omega_ssp if pc.omega_ssp is not None else (gm.R_inv.T @ pc.omega_raw @ gm.R_inv)
            )
            S[pc.group_sl, pc.group_sl] += lam * omega_ssp

        for group_index, (gm, g) in enumerate(zip(group_matrices, groups)):
            if group_index in represented_group_indices:
                continue
            if g.scop_reparameterization is not None and g.penalized:
                lam_g = lambda2.get(g.name, 0.0) if isinstance(lambda2, dict) else lambda2
                if lam_g > 0:
                    S[g.sl, g.sl] += lam_g * g.scop_reparameterization.penalty_matrix()

        return S

    for gm, g in zip(group_matrices, groups):
        if not g.penalized:
            continue

        if isinstance(
            gm,
            SparseSSPGroupMatrix
            | SplineCategoricalGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | DiscretizedSSPGroupMatrix,
        ):
            omega_components = getattr(gm, "omega_components", None)
            if omega_components is not None:
                for suffix, omega_j in omega_components:
                    lam_j = resolve_component_lambda(lambda2, g.name, suffix)
                    if lam_j == 0:
                        continue
                    S[g.sl, g.sl] += lam_j * (gm.R_inv.T @ omega_j @ gm.R_inv)
                continue
            lam_g = lambda2.get(g.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if lam_g == 0:
                continue
            omega = gm.omega
            if omega is None:
                continue
            S[g.sl, g.sl] += lam_g * gm.R_inv.T @ omega @ gm.R_inv
        elif g.scop_reparameterization is not None:
            lam_g = lambda2.get(g.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if lam_g == 0:
                continue
            S[g.sl, g.sl] += lam_g * g.scop_reparameterization.penalty_matrix()

    return S


def build_penalty_components(
    group_matrices: list,
    reml_groups: list[tuple[int, object]],
    cache: dict | None = None,
    *,
    _reuse_raw_from: Sequence[PenaltyComponent] | None = None,
) -> list[PenaltyComponent]:
    """Build PenaltyComponent list — the single source of penalty eigenstructure.

    Wood (2011) Section 3.1: pre-compute eigenstructure of each Ω_j in
    SSP coordinates so that log|S|₊ and rank(Ω_j) are O(1) per Newton step.

    Currently produces one PenaltyComponent per REML-eligible group (single
    penalty per term). Multi-penalty terms would produce multiple components
    per group, each with its own lambda optimized by REML.

    Parameters
    ----------
    group_matrices : list of GroupMatrix
    reml_groups : list of (group_index, GroupSlice) tuples

    Returns
    -------
    list of PenaltyComponent, one per smoothing parameter.
    """
    components: list[PenaltyComponent] = []
    component_cache = None if cache is None else cache.setdefault("penalty_components", {})
    eps_thresh = np.finfo(float).eps ** (2 / 3)

    def _canonicalize_ssp_penalty(
        omega_ssp: NDArray,
        rank: float,
        *,
        eigenvalues: NDArray | None = None,
        eigenvectors: NDArray | None = None,
    ) -> NDArray:
        """Enforce the declared PSD rank after a noisy SSP congruence.

        Spline penalties are PSD by construction, but transforming them with
        ``R_inv.T @ omega @ R_inv`` can leave 1e-12-scale negative curvature
        in an exact null direction.  Rank selection has already identified
        the authoritative retained subspace, so reconstructing that subspace
        removes only numerical null-space contamination instead of weakening
        downstream PSD validation.
        """
        symmetric = 0.5 * (np.asarray(omega_ssp, dtype=float) + np.asarray(omega_ssp).T)
        width = symmetric.shape[0]
        retained = int(rank)
        if retained == width:
            return symmetric
        if retained == 0:
            return np.zeros_like(symmetric)
        if eigenvalues is None or eigenvectors is None:
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        retained_values = np.asarray(eigenvalues[-retained:], dtype=float)
        if np.any(retained_values <= 0.0):
            raise ValueError("declared penalty rank includes non-positive SSP curvature")
        retained_vectors = np.asarray(eigenvectors[:, -retained:], dtype=float)
        canonical = (retained_vectors * retained_values) @ retained_vectors.T
        return 0.5 * (canonical + canonical.T)

    def _rank_and_logdet(
        omega_raw: NDArray,
        omega_ssp: NDArray,
        *,
        force_solver_rank: bool = False,
    ) -> tuple[float, float, NDArray, NDArray]:
        """Compute basis-invariant rank from raw penalty, log|Ω|₊ from SSP.

        For square SSP congruences, rank is computed from the raw
        (basis-invariant) penalty to avoid threshold sensitivity:
        ``R_inv.T @ Ω @ R_inv`` can shift near-null eigenvalues above/below
        the threshold depending on which valid ``R_inv`` was used.

        When extra side constraints have removed coefficient directions, the
        REML rank must instead be the rank of the projected solver-space
        penalty, because the removed raw directions no longer contribute
        coefficients or degrees of freedom.

        ``log|Ω|₊`` must stay in SSP coordinates because it enters the REML
        objective as ``log|S|₊ - log|H|``, where ``log|H|`` is also in SSP
        coordinates. The ``2*log|R_inv|`` factors cancel only when both terms
        use the same basis.
        """
        # Rank from raw penalty (basis-invariant)
        raw_eigvals = np.linalg.eigvalsh(omega_raw)
        raw_thresh = eps_thresh * max(raw_eigvals.max(), 1e-12)
        raw_rank = float(np.sum(raw_eigvals > raw_thresh))

        # log|Ω|₊ and eigvals from SSP penalty (same basis as log|H|)
        ssp_eigvals, ssp_eigvectors = np.linalg.eigh(omega_ssp)
        ssp_thresh = eps_thresh * max(ssp_eigvals.max(), 1e-12)
        ssp_rank = float(np.sum(ssp_eigvals > ssp_thresh))
        rank = ssp_rank if force_solver_rank or raw_rank > omega_ssp.shape[0] else raw_rank

        # Use the effective rank to select the top eigenvalues from SSP.
        n_pos = int(rank)
        if n_pos > 0:
            sorted_ssp = np.sort(ssp_eigvals)[::-1]
            pos_eigvals = sorted_ssp[:n_pos]
            log_det = float(np.sum(np.log(np.maximum(pos_eigvals, 1e-300))))
        else:
            pos_eigvals = np.array([])
            log_det = 0.0

        canonical_ssp = _canonicalize_ssp_penalty(
            omega_ssp,
            rank,
            eigenvalues=ssp_eigvals,
            eigenvectors=ssp_eigvectors,
        )
        return rank, log_det, pos_eigvals, canonical_ssp

    for idx, g in reml_groups:
        gm = group_matrices[idx]
        can_cache_group = component_cache is not None and _can_cache_penalty_group(gm)
        cache_key = _penalty_group_cache_key(idx, g, gm) if can_cache_group else None
        if cache_key is not None and cache_key in component_cache:
            components.extend(component_cache[cache_key])
            continue

        group_components: list[PenaltyComponent] = []
        raw_support = None
        raw_coordinate_map = None
        raw_family = reused_geometry = None
        group_ssp_roots = group_ssp_errors = ()
        matrix_errors = []

        if isinstance(gm, RandomEffectGroupMatrix):
            lp_map = gm.lambda_policies or {}
            components.append(
                PenaltyComponent(
                    name=g.name,
                    group_name=g.name,
                    group_index=idx,
                    group_sl=g.sl,
                    omega_raw=None,
                    omega_ssp=None,
                    rank=float(g.size),
                    log_det_omega_plus=0.0,
                    eigvals_omega=None,
                    lambda_policy=lp_map.get(g.name) or lp_map.get("_default"),
                    penalty_kind="identity",
                )
            )
            continue

        if isinstance(gm, FactorSmoothGroupMatrix):
            lp_map = gm.lambda_policies or {}
            if gm.factor_basis == "sz":
                repeated_components = gm.repeated_penalty_components
                if len(repeated_components) != 1 or repeated_components[0][0] != "wiggle":
                    raise ValueError("SZ factor smooths require exactly one 'wiggle' component.")
                suffix, omega_j = repeated_components[0]
                local_rank, local_log_det, local_eigvals, omega_ssp_j = _rank_and_logdet(
                    omega_j,
                    omega_j,
                )
                n_levels = gm.n_levels
                full_eigvals = np.sort(
                    np.concatenate(
                        (
                            np.tile(local_eigvals, max(n_levels - 2, 0)),
                            n_levels * local_eigvals,
                        )
                    )
                )[::-1]
                group_components.append(
                    PenaltyComponent(
                        name=f"{g.name}:{suffix}",
                        group_name=g.name,
                        group_index=idx,
                        group_sl=g.sl,
                        omega_raw=omega_j,
                        omega_ssp=omega_ssp_j,
                        rank=float((n_levels - 1) * local_rank),
                        log_det_omega_plus=float(
                            (n_levels - 1) * local_log_det + local_rank * np.log(n_levels)
                        ),
                        eigvals_omega=full_eigvals,
                        component_type="wiggle",
                        lambda_policy=lp_map.get(suffix),
                        penalty_kind="sum_to_zero",
                        repeat_count=n_levels,
                        block_width=gm.block_size,
                    )
                )
                _attach_context_geometry(group_components)
                components.extend(group_components)
                continue
            for suffix, omega_j in gm.repeated_penalty_components:
                local_rank, local_log_det, local_eigvals, omega_ssp_j = _rank_and_logdet(
                    omega_j,
                    omega_j,
                )
                group_components.append(
                    PenaltyComponent(
                        name=f"{g.name}:{suffix}",
                        group_name=g.name,
                        group_index=idx,
                        group_sl=g.sl,
                        omega_raw=omega_j,
                        omega_ssp=omega_ssp_j,
                        rank=float(gm.n_levels * local_rank),
                        log_det_omega_plus=float(gm.n_levels * local_log_det),
                        eigvals_omega=np.tile(local_eigvals, gm.n_levels),
                        component_type="wiggle" if suffix == "wiggle" else "null",
                        lambda_policy=lp_map.get(suffix),
                        penalty_kind="repeated",
                        repeat_count=gm.n_levels,
                        block_width=gm.block_size,
                    )
                )
        elif getattr(gm, "omega_components", None) is not None:
            # Multi-penalty path: N components share this coefficient block.
            ct_map = getattr(gm, "component_types", None) or {}
            lp_map = getattr(gm, "lambda_policies", None) or {}
            if len(gm.omega_components) > 1 and not isinstance(gm, DiscretizedTensorGroupMatrix):
                coordinate_map = np.asarray(gm.R_inv, dtype=float)
                raw_width = gm.omega_components[0][1].shape[0]
                if coordinate_map.shape == (raw_width, raw_width):
                    from superglm.reml.penalty_support import _penalty_support

                    if not np.all(np.isfinite(coordinate_map)):
                        raise ValueError("SSP coordinate map must be finite")
                    raw_targets = [
                        PenaltyComponent(
                            name=f"{g.name}:{suffix}",
                            group_name=g.name,
                            group_index=idx,
                            group_sl=g.sl,
                            omega_raw=omega,
                            component_type=ct_map.get(suffix),
                            lambda_policy=lp_map.get(suffix),
                        )
                        for suffix, omega in gm.omega_components
                    ]
                    reused_geometry = _reusable_raw_geometry(_reuse_raw_from, raw_targets)
                    if reused_geometry is None:
                        raw_support = _penalty_support([omega for _, omega in gm.omega_components])
                        raw_family = _RawPenaltyFamilyReceipt.capture(raw_support, raw_targets)
                    else:
                        raw_support = reused_geometry.support
                        raw_family = reused_geometry.raw_family
                    raw_coordinate_map = coordinate_map
                    group_ssp_roots, group_ssp_errors = _ssp_component_roots(
                        raw_support, coordinate_map
                    )
            for component_index, (suffix, omega_j) in enumerate(gm.omega_components):
                tensor_summary = _tensor_marginal_rank_logdet(
                    gm,
                    omega_j,
                    eps_thresh=eps_thresh,
                )
                if raw_support is not None:
                    # Transport one selected raw family through the common map.
                    # Independent eigen-truncations of transformed Grams can
                    # rotate their null spaces and invent a union direction.
                    root = group_ssp_roots[component_index]
                    root_error = group_ssp_errors[component_index]
                    omega_ssp_j, matrix_error = _enclosed_root_gram(root, root_error)
                    matrix_errors.append(matrix_error)
                    singular = scipy.linalg.svdvals(root, check_finite=False)
                    rank = float(len(singular))
                    if np.any(singular <= 0):
                        from superglm.reml.penalty_support import PenaltyNumericalError

                        raise PenaltyNumericalError("SSP map lost a selected component direction")
                    log_det = math.fsum(2 * math.log(value) for value in singular)
                    pos_eigvals = singular**2
                elif tensor_summary is not None:
                    rank, log_det, pos_eigvals = tensor_summary
                    omega_ssp_j = _canonicalize_ssp_penalty(omega_j, rank)
                else:
                    omega_ssp_j = gm.R_inv.T @ omega_j @ gm.R_inv
                    force_solver_rank = (
                        isinstance(gm, DiscretizedTensorGroupMatrix)
                        and getattr(gm, "projection", None) is not None
                    )
                    rank, log_det, pos_eigvals, omega_ssp_j = _rank_and_logdet(
                        omega_j,
                        omega_ssp_j,
                        force_solver_rank=force_solver_rank,
                    )
                group_components.append(
                    PenaltyComponent(
                        name=f"{g.name}:{suffix}",
                        group_name=g.name,
                        group_index=idx,
                        group_sl=g.sl,
                        omega_raw=omega_j,
                        omega_ssp=omega_ssp_j,
                        rank=rank,
                        log_det_omega_plus=log_det,
                        eigvals_omega=pos_eigvals,
                        component_type=ct_map.get(suffix),
                        lambda_policy=lp_map.get(suffix),
                    )
                )
        else:
            # Single-penalty path.
            lp_map = getattr(gm, "lambda_policies", None) or {}
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            force_solver_rank = (
                isinstance(gm, DiscretizedTensorGroupMatrix)
                and getattr(gm, "projection", None) is not None
            )
            rank, log_det, pos_eigvals, omega_ssp = _rank_and_logdet(
                gm.omega,
                omega_ssp,
                force_solver_rank=force_solver_rank,
            )
            group_components.append(
                PenaltyComponent(
                    name=g.name,
                    group_name=g.name,
                    group_index=idx,
                    group_sl=g.sl,
                    omega_raw=gm.omega,
                    omega_ssp=omega_ssp,
                    rank=rank,
                    log_det_omega_plus=log_det,
                    eigvals_omega=pos_eigvals,
                    lambda_policy=lp_map.get(g.name) or lp_map.get("_default"),
                )
            )
        _attach_context_geometry(
            group_components,
            support=raw_support,
            coordinate_map=raw_coordinate_map,
            matrix_errors=matrix_errors,
            ssp_roots=group_ssp_roots,
            ssp_errors=group_ssp_errors,
            raw_family=raw_family,
        )
        if reused_geometry is not None:
            receipt = reused_geometry.raw_summary
            if receipt is not None and receipt.matches(reused_geometry):
                geometry = _context_geometry(group_components)
                # The raw result is map-independent. Its new owner uses only
                # the fresh map's admitted volume and starts with no face state.
                geometry.last_weights = receipt.weights
                geometry.last_evaluation = (receipt.summary, *geometry.volume)
                geometry.raw_summary = receipt
        if cache_key is not None:
            component_cache[cache_key] = tuple(group_components)
        components.extend(group_components)
    return components


def coerce_reml_penalties(
    reml_groups=None,
    reml_penalties=None,
    group_matrices=None,
    penalty_caches=None,
):
    """Coerce legacy ``reml_groups`` inputs into ``PenaltyComponent`` objects."""
    if reml_penalties is not None:
        return reml_penalties
    if reml_groups is None:
        raise ValueError("Either reml_penalties or reml_groups must be provided")

    components = []
    for idx, g in reml_groups:
        gm = group_matrices[idx] if group_matrices is not None else None
        omega_ssp = None
        rank = 0.0
        log_det = 0.0
        eigvals = None
        omega_raw = None
        if penalty_caches is not None and g.name in penalty_caches:
            cache = penalty_caches[g.name]
            omega_ssp = cache.omega_ssp
            rank = cache.rank
            log_det = cache.log_det_omega_plus
            eigvals = cache.eigvals_omega
        elif (
            gm is not None
            and hasattr(gm, "R_inv")
            and hasattr(gm, "omega")
            and gm.omega is not None
        ):
            if (
                not isinstance(gm, FactorSmoothGroupMatrix)
                and getattr(gm, "omega_components", None) is None
            ):
                # This is an internally formed singleton, with the same
                # canonical geometry used to build its compatibility cache.
                # A raw rounded congruence can contaminate its exact nullspace.
                components.extend(build_penalty_components(group_matrices, [(idx, g)]))
                continue
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
        if gm is not None and hasattr(gm, "omega"):
            omega_raw = gm.omega
        components.append(
            PenaltyComponent(
                name=g.name,
                group_name=g.name,
                group_index=idx,
                group_sl=g.sl,
                omega_raw=omega_raw,
                omega_ssp=omega_ssp,
                rank=rank,
                log_det_omega_plus=log_det,
                eigvals_omega=eigvals,
            )
        )
    return components


def build_penalty_caches(
    group_matrices: list,
    reml_groups: list[tuple[int, object]],
    cache: dict | None = None,
) -> dict[str, PenaltyCache]:
    """Build PenaltyCache dict — thin wrapper over build_penalty_components.

    Retained for backward compatibility. New code should prefer
    build_penalty_components directly.
    """
    components = build_penalty_components(group_matrices, reml_groups, cache=cache)
    return {
        c.name: PenaltyCache(
            omega_ssp=c.omega_ssp,
            log_det_omega_plus=c.log_det_omega_plus,
            rank=c.rank,
            eigvals_omega=c.eigvals_omega,
        )
        for c in components
    }


def build_penalty_context(
    group_matrices: list,
    reml_groups: list[tuple[int, object]],
    cache: dict | None = None,
    *,
    _reuse_raw_from: Sequence[PenaltyComponent] | None = None,
) -> tuple[list[PenaltyComponent], dict[str, PenaltyCache], dict[str, float]]:
    """Build penalty components, caches, and rank lookup in one pass."""
    components = build_penalty_components(
        group_matrices, reml_groups, cache=cache, _reuse_raw_from=_reuse_raw_from
    )
    caches = {
        c.name: PenaltyCache(
            omega_ssp=c.omega_ssp,
            log_det_omega_plus=c.log_det_omega_plus,
            rank=c.rank,
            eigvals_omega=c.eigvals_omega,
        )
        for c in components
    }
    ranks = {c.name: c.rank for c in components}
    return components, caches, ranks


def cached_logdet_s_plus(
    lambdas: dict[str, float],
    penalty_caches: dict[str, PenaltyCache],
) -> float:
    """Compute log|S|₊ from cached penalty eigenstructure.

    Wood (2011) Section 3.1: stable block-diagonal identity
    log|S|₊ = Σ_j (r_j · log(λ_j) + log|Ω_j|₊), avoiding repeated
    eigendecompositions of the full penalty matrix.

    NOTE: This is the single-penalty-per-block shortcut. For multi-penalty
    groups sharing a coefficient block, use ``compute_logdet_s_plus``
    which correctly computes the joint log-determinant.
    """
    penalties = []
    for name, cache in penalty_caches.items():
        identity = cache.omega_ssp is None
        if identity:
            width = int(cache.rank)
            if width < 0 or width != cache.rank or cache.log_det_omega_plus != 0.0:
                raise ValueError("identity penalty cache has invalid analytic geometry")
        else:
            width = cache.omega_ssp.shape[0]
        penalties.append(
            PenaltyComponent(
                name=name,
                group_name=name,
                group_index=len(penalties),
                group_sl=slice(0, width),
                omega_raw=None,
                omega_ssp=cache.omega_ssp,
                penalty_kind="identity" if identity else "dense",
            )
        )
    return _compute_penalty_logdet_evaluation(lambdas, penalties).logdet


def _group_penalty_matrices(grouped: list[PenaltyComponent]) -> tuple[list[NDArray], int]:
    """Use a single local block for identically repeated component geometry."""
    if any(component.group_sl != grouped[0].group_sl for component in grouped[1:]):
        raise ValueError("Components in one group must share their coefficient slice.")
    if all(component.penalty_kind == "repeated" for component in grouped):
        geometry = _repeated_penalty_geometry(grouped[0])
        if any(_repeated_penalty_geometry(component) != geometry for component in grouped[1:]):
            raise ValueError("Repeated components in one group must share geometry.")
        matrices = [np.asarray(_penalty_component_omega_ssp(component)) for component in grouped]
        if any(matrix.shape != (geometry[1], geometry[1]) for matrix in matrices):
            raise ValueError("Repeated penalty local matrix does not match its geometry.")
        return matrices, geometry[0]
    return [penalty_component_dense_matrix(component) for component in grouped], 1


def _group_penalty_rank(grouped: list[PenaltyComponent]) -> int:
    from superglm.reml.penalty_support import _penalty_support

    geometry = _context_geometry(grouped)
    if geometry is not None:
        return geometry.repeat * geometry.get_support().rank
    if len(grouped) == 1:
        component = grouped[0]
        if component.penalty_kind == "identity":
            return component.group_sl.stop - component.group_sl.start
        if component.penalty_kind == "sum_to_zero":
            levels, _ = _sum_to_zero_penalty_geometry(component)
            return (levels - 1) * _penalty_support([_penalty_component_omega_ssp(component)]).rank
    matrices, repeat = _group_penalty_matrices(grouped)
    return repeat * _penalty_support(matrices).rank


def compute_total_penalty_rank(
    penalties: list[PenaltyComponent],
    tensor_pair_evaluations: dict[str, TensorPairLogdetEvaluation] | None = None,
) -> float:
    """Rank the supplied components as active, before applying any weights."""
    total = 0.0
    for group_name, indices in _group_penalties(penalties).items():
        if tensor_pair_evaluations is not None and group_name in tensor_pair_evaluations:
            tensor = tensor_pair_evaluations[group_name]
            total += (
                tensor.support_rank
                if tensor.support_rank is not None
                else _group_penalty_rank([penalties[index] for index in indices])
            )
        else:
            total += _group_penalty_rank([penalties[index] for index in indices])
    return total


def _matrix_penalty_rank(penalty_matrix: NDArray) -> int:
    """Numerical rank fallback for an already assembled PSD penalty."""
    from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_factor

    penalty_matrix = np.asarray(penalty_matrix, dtype=np.float64)
    if penalty_matrix.ndim != 2 or penalty_matrix.shape[0] != penalty_matrix.shape[1]:
        raise ValueError("penalty_matrix must be square")
    symmetric = 0.5 * (penalty_matrix + penalty_matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    negative_tolerance = 1e-10 * scale
    if eigenvalues.size and eigenvalues[0] < -negative_tolerance:
        raise ValueError("penalty_matrix must be positive semidefinite")
    spectral_cutoff = SHARED_RANK_POLICY.gram_rcond * scale
    positive = eigenvalues > spectral_cutoff
    penalty_factor = (
        np.sqrt(eigenvalues[positive])[:, None] * eigenvectors[:, positive].T
        if np.any(positive)
        else np.empty((0, penalty_matrix.shape[0]))
    )
    return decompose_factor(penalty_factor).rank


def _structural_active_penalty_rank(
    penalties: list[PenaltyComponent],
    lambdas: dict[str, float],
) -> int:
    """Rank the active component null-space intersection without lambda scaling."""
    total = 0
    for indices in _group_penalties(penalties).values():
        active: list[PenaltyComponent] = []
        for index in indices:
            component = penalties[index]
            lam = float(lambdas.get(component.name, 1.0))
            if not np.isfinite(lam) or lam < 0.0:
                raise ValueError("smoothing parameters must be finite and non-negative")
            if lam > 0.0:
                active.append(component)
        if not active:
            continue
        total += _group_penalty_rank(active)
    return total


def compute_penalty_nullity(
    penalty_matrix: NDArray | None = None,
    *,
    hessian_rank: int,
    penalties: list[PenaltyComponent] | None = None,
    lambdas: dict[str, float] | None = None,
    coefficient_width: int | None = None,
) -> float:
    """Return Wood's ``M_p`` in the identifiable full coefficient space.

    Production REML callers should supply ``penalties`` and ``lambdas``.  The
    rank is then computed from balanced active component roots:
    every finite positive lambda is structurally active, while an exact zero
    is inactive.  This makes ``null(S)`` invariant to arbitrary positive
    smoothing-parameter ratios.

    ``penalty_matrix`` alone is a deliberately limited numerical fallback for
    independent dense oracles that do not own component metadata.  An already
    scaled matrix cannot distinguish a genuine small eigenvalue from an
    extreme lambda ratio.

    The intercept is already included in ``hessian_rank``.  For a full-rank
    augmented Hessian, ``M_p = p + 1 - rank(S)``.  The identified Hessian rank
    excludes unpenalized coefficient aliases that the data cannot identify.
    """
    if penalty_matrix is not None:
        penalty_matrix = np.asarray(penalty_matrix, dtype=np.float64)
        if penalty_matrix.ndim != 2 or penalty_matrix.shape[0] != penalty_matrix.shape[1]:
            raise ValueError("penalty_matrix must be square")
        matrix_width = penalty_matrix.shape[0]
        if coefficient_width is not None and coefficient_width != matrix_width:
            raise ValueError("coefficient_width does not match penalty_matrix")
        coefficient_width = matrix_width
    elif coefficient_width is None:
        coefficient_width = (
            max(component.group_sl.stop for component in penalties)
            if penalties
            else max(hessian_rank - 1, 0)
        )

    max_hessian_rank = coefficient_width + 1
    if hessian_rank < 0 or hessian_rank > max_hessian_rank:
        raise ValueError("hessian_rank is incompatible with the penalty dimension")

    if penalties is not None:
        if lambdas is None:
            raise ValueError("lambdas are required with penalty components")
        penalty_rank = _structural_active_penalty_rank(penalties, lambdas)
    elif penalty_matrix is not None:
        penalty_rank = _matrix_penalty_rank(penalty_matrix)
    else:
        penalty_rank = 0
    return float(max(hessian_rank - penalty_rank, 0))


def _group_penalties(penalties: list[PenaltyComponent]) -> dict[str, list[int]]:
    """Group penalty component indices by group_name."""
    groups: dict[str, list[int]] = {}
    for i, pc in enumerate(penalties):
        groups.setdefault(pc.group_name, []).append(i)
    return groups


def compute_logdet_s_plus(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
    tensor_pair_evaluations: dict[str, TensorPairLogdetEvaluation] | None = None,
) -> float:
    """Return the determinant of the common checked component representative."""
    return _compute_penalty_logdet_evaluation(lambdas, penalties, tensor_pair_evaluations).logdet


def compute_logdet_s_derivatives(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
    tensor_pair_evaluations: dict[str, TensorPairLogdetEvaluation] | None = None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Return log-lambda determinant derivatives on the checked support."""
    evaluation = _compute_penalty_logdet_evaluation(lambdas, penalties, tensor_pair_evaluations)
    return evaluation.gradient, evaluation.hessian


def _compute_penalty_logdet_evaluation(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
    tensor_pair_evaluations: dict[str, TensorPairLogdetEvaluation] | None = None,
) -> _PenaltyLogdetEvaluation:
    """Evaluate rank, determinant and derivatives from one representative per group.

    Singleton and compact repeated identities follow Wood (2011), section
    3.1. Their numerical ranks come from the shared root support, never stale
    component metadata. Bounds concern that selected representative.
    """
    from superglm.reml.multi_penalty import (
        _evaluate_penalty_support,
        logdet_s_gradient,
        logdet_s_hessian,
        similarity_transform_logdet,
    )
    from superglm.reml.penalty_support import PenaltyNumericalError, _penalty_support

    values = np.asarray([lambdas.get(component.name, 1.0) for component in penalties], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("smoothing parameters must be finite and non-negative")
    names = [component.name for component in penalties]
    if len(set(names)) != len(names):
        raise ValueError("penalty component names must be unique")
    gradient = dict.fromkeys(names, 0.0)
    gradient_error = dict.fromkeys(names, 0.0)
    hessian: dict[tuple[str, str], float] = {}
    hessian_error: dict[tuple[str, str], float] = {}
    log_terms = []
    log_errors = []
    rank = 0
    unit = np.finfo(float).eps / 2
    for group_name, indices in _group_penalties(penalties).items():
        grouped = [penalties[index] for index in indices]
        group_names = [component.name for component in grouped]
        group_values = values[indices]
        for name_i in group_names:
            for name_j in group_names:
                hessian[name_i, name_j] = hessian_error[name_i, name_j] = 0.0
        if not np.any(group_values > 0):
            continue
        if tensor_pair_evaluations is not None and group_name in tensor_pair_evaluations:
            evaluation = tensor_pair_evaluations[group_name]
            rank += int(evaluation.rank)
            log_terms.append(evaluation.logdet_s_plus)
            log_errors.append(evaluation.logdet_error)
            gradient.update(evaluation.gradient)
            hessian.update(evaluation.hessian)
            gradient_error.update(evaluation.gradient_error)
            hessian_error.update(evaluation.hessian_error)
            continue
        component = grouped[0]
        if len(grouped) == 1 and component.penalty_kind == "identity":
            group_rank = component.group_sl.stop - component.group_sl.start
            term = group_rank * math.log(float(group_values[0]))
            rank += group_rank
            log_terms.append(term)
            log_errors.append(4 * unit * abs(term))
            gradient[component.name] = float(group_rank)
            continue
        extra_volume = 0.0
        geometry = _context_geometry(grouped)
        if len(grouped) == 1 and component.penalty_kind == "sum_to_zero":
            levels, _ = _sum_to_zero_penalty_geometry(component)
            matrices = [_penalty_component_omega_ssp(component)]
            repeat = levels - 1
            extra_volume = math.log(levels)
        elif geometry is not None:
            matrices, repeat = geometry.matrices, geometry.repeat
        else:
            matrices, repeat = _group_penalty_matrices(grouped)
        if len(grouped) == 1:
            # Establish the unweighted representative once, then use the
            # analytic affine log-lambda identity. Its derivatives are exact.
            support = geometry.get_support() if geometry is not None else _penalty_support(matrices)
            log_weight = math.log(float(group_values[0]))
            try:
                result = (
                    geometry.evaluate(np.ones(1))[0]
                    if geometry is not None
                    else _evaluate_penalty_support(support, np.ones(1))
                )
            except PenaltyNumericalError as exc:
                if str(exc) != "required dense penalty inverse is not representable":
                    raise
                # The actual weighted system may have a representable inverse
                # even when a component's arbitrary units make P^-1 overflow.
                result = (
                    geometry.evaluate(group_values)[0]
                    if geometry is not None
                    else _evaluate_penalty_support(support, group_values)
                )
                log_weight = 0.0
            group_rank = repeat * result.rank
            term = math.fsum(
                [
                    repeat * result.logdet_s_plus,
                    result.rank * extra_volume,
                    group_rank * log_weight,
                ]
            )
            rank += group_rank
            log_terms.append(term)
            gradient[component.name] = float(group_rank)
            certificate = result._certificate
            if certificate is None:
                raise ValueError("penalty evaluation did not provide arithmetic evidence")
            log_errors.append(
                repeat * certificate.logdet_error
                + 8
                * unit
                * (
                    abs(repeat * result.logdet_s_plus)
                    + abs(result.rank * extra_volume)
                    + abs(group_rank * log_weight)
                )
            )
            continue
        if geometry is None:
            result = similarity_transform_logdet(matrices, group_values)
            volume, volume_error = 0.0, 0.0
            grad = logdet_s_gradient(result, matrices, group_values)
            hess = logdet_s_hessian(result, matrices, group_values)
        else:
            result, volume, volume_error = geometry.evaluate(group_values)
            grad, hess = result.gradient, result.hessian
        certificate = result._certificate
        if certificate is None:
            raise ValueError("penalty evaluation did not provide arithmetic evidence")
        rank += repeat * result.rank
        term = repeat * math.fsum([result.logdet_s_plus, volume])
        log_terms.append(term)
        log_errors.append(
            repeat * (certificate.logdet_error + volume_error)
            + 4 * unit * repeat * (abs(result.logdet_s_plus) + abs(volume))
        )
        for i, name_i in enumerate(group_names):
            gradient[name_i] = float(repeat * grad[i])
            gradient_error[name_i] = float(
                repeat * certificate.gradient_error[i] + 2 * unit * abs(gradient[name_i])
            )
            for j, name_j in enumerate(group_names):
                key = (name_i, name_j)
                hessian[key] = float(repeat * hess[i, j])
                hessian_error[key] = float(
                    repeat * certificate.hessian_error[i, j] + 2 * unit * abs(hessian[key])
                )
    logdet = math.fsum(log_terms)
    return _PenaltyLogdetEvaluation(
        rank=rank,
        logdet=logdet,
        gradient=gradient,
        hessian=hessian,
        logdet_error=math.fsum(log_errors) + 2 * unit * abs(logdet),
        gradient_error=gradient_error,
        hessian_error=hessian_error,
    )
