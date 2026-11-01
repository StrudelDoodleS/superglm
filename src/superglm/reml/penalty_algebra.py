"""REML penalty eigenstructure and log-determinant algebra.

Pre-computes per-term penalty eigenstructure (Wood 2011 Section 3.1) and
provides log|S|₊ / ∂log|S|₊ / ∂²log|S|₊ for both single and multi-penalty
groups.  GLM REML evaluates them block by block (``penalty_logdet``); the
distributional smoother's evaluator also bounds their rounding
(``_compute_penalty_logdet_evaluation`` on a ``certified`` context).

References
----------
- Wood (2011): Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  JRSS-B 73(1), 3-36.
- Wood, Pya and Saefken (2016): Smoothing parameter and model selection for
  general smooth models. JASA 111, 1548-1563, section 3.1.1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm._roundoff import PENALTY_RANK_CUT, SMALLEST_SUBNORMAL, UNIT_ROUNDOFF
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
    # Each margin's REML rank when a structural rank was declared
    # (``_lifted_positive``); evaluation keeps that many eigenvalues.
    rank_left: int | None = None
    rank_right: int | None = None


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
    support_rank: int | None = None


@dataclass(frozen=True)
class PenaltyLogdet:
    """``log|S_lambda|_+``, its rank, and its gradient and Hessian in log weights."""

    rank: int
    logdet: float
    gradient: dict[str, float]
    hessian: dict[tuple[str, str], float]


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


def _enclosed_bound_sum(*bounds: NDArray | float | np.float64) -> NDArray:
    from superglm.reml.multi_penalty import _gamma, _upper

    computed = np.sum(np.asarray(bounds, dtype=np.float64), axis=0)
    return _upper((computed + len(bounds) * SMALLEST_SUBNORMAL) / (1 - _gamma(len(bounds) + 2)))


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
    from superglm.reml.multi_penalty import _matmul_enclosed, _refine_product

    value, error = _matmul_enclosed(left, right)
    if refine:
        _refine_product(left, right, value, error)
    return value, error


def _retained_coordinate_map(
    support, coordinate_map: NDArray, *, refine: bool = False, products=None
) -> tuple[NDArray, NDArray]:
    """Enclose (Q.T Q)^-1 Q.T C without amplifying off-support root residue.

    ``products`` are the enclosed ``C.T Q`` and ``Q.T Q`` already formed for
    this map's volume (``_shared_map_products``); a refined call forms its own.
    """
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
    shared = products is not None and not refine
    if shared:
        gram, gram_error = products[2], products[3]
    else:
        gram, gram_error = _context_product(basis.T, basis, refine=refine)
    defect_bound = _enclosed_bound_sum(np.abs(gram - np.eye(rank)), gram_error)
    eta = _norm_upper(defect_bound)
    if eta >= 1:
        raise PenaltyNumericalError("retained penalty coordinates are unresolved")
    if shared:
        # The transpose of an enclosed C.T Q encloses Q.T C entrywise.
        mapped, mapped_error = products[0].T, products[1].T
    else:
        mapped, mapped_error = _context_product(basis.T, coordinate_map, refine=refine)
    result = scipy.linalg.solve(gram, mapped, assume_a="pos", check_finite=False)
    product, product_error = _context_product(gram, result, refine=refine)
    residual = _enclosed_bound_sum(
        np.abs(product - mapped),
        product_error,
        mapped_error,
        _positive_product(gram_error, np.abs(result)),
    )
    # The Neumann series gives an elementwise enclosure of G^-1:
    # |G^-1| <= I + |G-I| + eta**2/(1-eta), since every tail entry is
    # bounded by the corresponding sum of spectral norms.
    tail = _upper(np.float64(eta) ** 2 / (1 - eta) / (1 - _gamma(5)))
    inverse_bound = _enclosed_bound_sum(np.eye(rank), defect_bound, np.full_like(gram, tail))
    return result, _positive_product(inverse_bound, residual)


def _ssp_component_roots(support, coordinate_map: NDArray, *, refine: bool = False, products=None):
    """One common retained-coordinate target, with optional sharper products."""
    from superglm.reml.multi_penalty import _positive_product

    full = support.rank == support.Q_plus.shape[0]
    if not full:
        retained_map, retained_error = _retained_coordinate_map(
            support, coordinate_map, refine=refine, products=products
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

    defect = gram - np.eye(len(gram))
    error = _enclosed_bound_sum(error, _gamma(1) * np.abs(defect) + np.nextafter(0.0, 1.0))
    eta = float(_upper(_norm_upper(defect) + _norm_upper(error)))
    if eta >= 1:
        raise PenaltyNumericalError("SSP coordinate volume cannot certify injectivity")
    diagonal = np.diag(defect)
    value = math.fsum(map(float, diagonal))
    # For symmetric D with ||D||_2 <= eta < 1,
    # |log det(I+D) - tr D| <= ||D||_F**2 / (2*(1-eta)).
    eta_value = np.float64(eta)
    bound = (
        np.float64(math.fsum(map(float, np.diag(error))))
        + eta_value**2 / (2 * (1 - eta_value))
        + np.float64(_gamma(len(gram) + 2)) * math.fsum(map(float, np.abs(diagonal)))
    )
    unit = UNIT_ROUNDOFF
    tiny = SMALLEST_SUBNORMAL
    return value, float(_upper((bound + 8 * tiny) / (1 - _gamma(8, unit))))


def _shared_map_products(support, coordinate_map: NDArray) -> tuple[NDArray, ...]:
    """Enclosed ``C.T Q`` for this map and ``Q.T Q`` for the support, formed once.

    The volume and the retained coordinate map both need them. ``Q.T Q`` does
    not depend on the map, so ``_basis_gram`` keeps it on the support, which
    the raw family carries from one REML context to the next.
    """
    from superglm.reml.multi_penalty import _basis_gram, _matmul_enclosed

    basis = support.Q_plus
    return (*_matmul_enclosed(coordinate_map.T, basis), *_basis_gram(support, basis))


def _support_coordinate_volume(
    support, coordinate_map: NDArray, *, _refine: bool = False, products=None
) -> tuple[float, float]:
    """Bound the volume ratio of one fixed active support under an SSP map.

    For B = C.T Q, the log pseudodeterminant changes by
    log det(B.T B) - log det(Q.T Q). The denominator accounts for the stored
    Q's finite orthogonality. This common-map identity preserves log-weight
    derivatives; it does not introduce independent component-root errors.
    """
    from superglm.reml.multi_penalty import (
        _finite_double,
        _gamma,
        _matmul_enclosed,
        _positive_product,
        _refine_product,
        _triangular_solve,
        _upper,
    )
    from superglm.reml.penalty_support import PenaltyNumericalError

    rank = support.rank
    width = coordinate_map.shape[0]
    if rank == 0 or np.array_equal(coordinate_map, np.eye(width)):
        return 0.0, 0.0
    basis = support.Q_plus
    if products is not None and not _refine:
        mapped, mapped_error = products[0], products[1]
    else:
        mapped, mapped_error = _matmul_enclosed(coordinate_map.T, basis)
    if _refine:
        _refine_product(coordinate_map.T, basis, mapped, mapped_error)
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
    if products is not None:
        basis_gram, basis_error = products[2], products[3]
    else:
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
            _upper(np.float64(_gamma(4 * rank + 8)) * np.float64(operation_scale)),
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
    unit = UNIT_ROUNDOFF
    tiny = SMALLEST_SUBNORMAL
    roots, errors = [], []
    for root, error in zip(
        support.component_roots, support.component_root_error_bounds, strict=True
    ):
        scale = np.max(np.abs(root), initial=0.0)
        if scale == 0:
            scale = 1.0
        computed = root / scale
        normalized = _finite_double(computed, "active support roots")
        bound = (error / scale + _gamma(1, unit) * np.abs(computed) + tiny) / (1 - _gamma(6, unit))
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
        np.abs(root - projected),
        projection_error,
        root_error,
    )
    off_norm = _norm_upper(off)
    volume_error = float(
        _upper(
            ((np.float64(map_norm) * off_norm) ** 2 / (1 - np.float64(eta)) + tiny)
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
    # Enclose subtraction rounding too. This avoids
    # relying on exact diagonal subtraction when the map is malformed.
    unit = UNIT_ROUNDOFF
    tiny = SMALLEST_SUBNORMAL
    difference = gram - np.eye(len(gram))
    defect = _enclosed_bound_sum(
        np.abs(difference), arithmetic, _gamma(1, unit) * np.abs(difference) + tiny
    )
    delta = _norm_upper(defect)
    if delta >= 1:
        raise PenaltyNumericalError("finite coefficient map does not preserve penalty support")
    error = float(
        _upper((np.float64(rank) * delta / (1 - np.float64(delta)) + tiny) / (1 - _gamma(8, unit)))
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
    # False for a context built for the plain block-wise evaluator, which
    # carries no error ledgers; a model saved before that split is certified.
    certified: bool = True
    # For the next context of the same fit: copies of the raw penalties, the
    # raw support selected from them (kept when this map's agreement bars
    # refused it, since the next map rechecks them), and whether the raw route
    # was refused for reasons no map changes (selection or rank).
    raw_inputs: tuple[NDArray, ...] | None = None
    raw_support: object | None = None
    raw_refused: bool = False
    # The plain evaluator's weight-independent pieces and last evaluation.
    plain: _PlainBlock | None = None

    def __getstate__(self) -> dict:
        # The plain block is derived from the support and map; a reloaded
        # model rebuilds it on first use.  The raw inputs and selection serve
        # only the next context of the same fit.
        state = self.__dict__.copy()
        state["plain"] = None
        state["raw_inputs"] = state["raw_support"] = None
        if not self.certified and self.coordinate_map is not None and len(self.matrices) > 1:
            # Releases before the plain evaluator read a mapped multi-penalty
            # context's zero-weight faces from its enclosed SSP roots, which the
            # plain evaluator does not keep.
            from superglm.reml.penalty_support import PenaltyNumericalError

            try:
                state["ssp_roots"], state["ssp_root_errors"] = _ssp_component_roots(
                    self.get_support(), self.coordinate_map
                )
            except PenaltyNumericalError:
                pass
        return state

    def __setstate__(self, state: dict) -> None:
        # Pickle protocols below 5 and deepcopy restore these owned arrays as
        # writeable, and _context_geometry rejects a writeable matrix, so a
        # reloaded model would switch to the solver-space support. They were
        # read-only when saved and this restore is their only writer.
        self.__dict__.update(state)
        for array in (
            *self.matrices,
            self.coordinate_map,
            *self.matrix_error_bounds,
            *self.ssp_roots,
            *self.ssp_root_errors,
        ):
            if array is not None:
                array.setflags(write=False)

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
    volume=None,
    certified: bool = False,
    raw_inputs: tuple[NDArray, ...] | None = None,
    raw_support=None,
    raw_refused: bool = False,
    plain: _PlainBlock | None = None,
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
        certified=certified,
        raw_inputs=raw_inputs,
        raw_support=raw_support,
        raw_refused=raw_refused,
        plain=plain,
    )
    if coordinate_map is not None and certified:
        # Rank queries also rely on injectivity; certify it before exposing
        # the raw rank through this context, independently of positive weights.
        # A caller may pass the volume it already certified on this map.
        geometry.volume = (
            _support_coordinate_volume(support, geometry.coordinate_map)
            if volume is None
            else volume
        )
        geometry.volume_activity = tuple(True for _ in grouped)
    for component in grouped:
        component._penalty_geometry = geometry


@dataclass
class _PlainBlock:
    """One group's weight-independent pieces for the plain block-wise evaluator.

    ``rank`` is the support rank of one repeat.  A single penalty keeps
    ``constant``, its log-determinant at unit weight with the map's volume
    included.  A multi-penalty group keeps its component ``roots`` in support
    coordinates, the map's ``volume`` on that support, and, when its two
    penalties are exactly ``A (x) I`` and ``I (x) B``, the margins' retained
    eigenvalues in ``kronecker``.  ``faces`` holds each zero-weight pattern's
    active support; ``last`` the last evaluation and its weights.
    """

    rank: int
    constant: float = 0.0
    roots: tuple[NDArray, ...] = ()
    volume: float = 0.0
    kronecker: tuple[NDArray, NDArray, bool] | None = None
    faces: dict = field(default_factory=dict)
    last: tuple | None = None


def _support_roots(support) -> tuple[NDArray, ...]:
    """Component roots in the coordinates of the support basis, ``R_j Q``."""
    basis = np.ascontiguousarray(support.Q_plus)
    return tuple(np.ascontiguousarray(root) @ basis for root in support.component_roots)


def _transported_roots(support, coordinate_map: NDArray) -> tuple[NDArray, ...]:
    """Each component root carried through the map, ``R_j Q (Q'Q)^-1 Q' C``.

    The retained coordinates keep a root's residue off the support out of the
    map's image; a full-width support needs no projection.
    """
    basis = support.Q_plus
    if support.rank == basis.shape[0]:
        return tuple(root @ coordinate_map for root in support.component_roots)
    retained = scipy.linalg.solve(
        basis.T @ basis, basis.T @ coordinate_map, assume_a="pos", check_finite=False
    )
    return tuple((root @ basis) @ retained for root in support.component_roots)


def _transported_factor(support, coordinate_map: NDArray) -> tuple[NDArray, NDArray]:
    """A single penalty's transported root ``Y`` and the triangular ``U`` of ``Y' = Q U``.

    ``log pdet(C' Omega C) = log det(Y Y') = 2 sum log |U_ii|``.
    """
    root = _transported_roots(support, coordinate_map)[0]
    return root, scipy.linalg.qr(root.T, mode="r", check_finite=False)[0][: len(root)]


def _map_volume(support, coordinate_map: NDArray | None) -> float:
    from superglm.reml.penalty_blocks import map_volume
    from superglm.reml.penalty_support import PenaltyNumericalError

    if coordinate_map is None:
        return 0.0
    volume = map_volume(support.Q_plus, coordinate_map)
    if volume is None:
        raise PenaltyNumericalError("SSP coordinate map does not preserve penalty support")
    return volume


def _kronecker_margins(
    support, raw: Sequence[NDArray | None]
) -> tuple[NDArray, NDArray, bool] | None:
    """The margins' retained eigenvalues when the two raw penalties are a Kronecker pair.

    Each margin keeps as many eigenvalues as its component's support root has
    rows per repeat, so the rank decisions stay the support's; the pair is used
    only when its cell count is the support rank and ``eigvalsh`` resolves every
    retained eigenvalue; otherwise, as for a non-finite margin or one the
    eigensolver fails on, the support roots evaluate the block.

    ``eigvalsh`` moves each eigenvalue by up to ``p(n) eps ||A||_2`` (LAPACK
    Users' Guide, 3rd ed., sec. 4.7; ``_eigensolver_relative_bar``), however
    the margin is graded: relative accuracy for small eigenvalues needs a
    Jacobi-type method on the scaled matrix (Demmel and Veselic, SIAM J.
    Matrix Anal. Appl. 13(4), 1992).  The support's equilibrated roots can keep
    a direction below that floor.  Each retained eigenvalue enters ``log|S|_+``
    through its logarithm, so the closed form needs each one at least
    ``1 / sqrt(eps)`` times the floor, a relative error of at most ``sqrt(eps)``;
    otherwise the support roots evaluate the block.
    """
    from superglm.reml.penalty_blocks import kronecker_split
    from superglm.solvers.rank import SHARED_RANK_POLICY, _eigensolver_relative_bar

    if len(raw) != 2:
        return None
    first, second = raw
    if first is None or second is None:
        return None
    split = kronecker_split(first, second)
    if split is None:
        return None
    p1, p2, left, right, swapped = split
    rows = [len(root) for root in support.component_roots]
    left_rows, right_rows = (rows[1], rows[0]) if swapped else (rows[0], rows[1])
    if left_rows % p2 or right_rows % p1:
        return None
    left_rank, right_rank = left_rows // p2, right_rows // p1
    margins = []
    for matrix, rank in ((left, left_rank), (right, right_rank)):
        # Halved before the sum, which can overflow on a finite margin.
        symmetric = 0.5 * matrix + 0.5 * matrix.T
        if not np.all(np.isfinite(symmetric)):
            return None
        try:
            values = np.linalg.eigvalsh(symmetric)[::-1].copy()
        except np.linalg.LinAlgError:
            return None
        floor = _eigensolver_relative_bar(len(values)) * float(np.max(np.abs(values)))
        if rank and not values[rank - 1] * SHARED_RANK_POLICY.factor_rcond > floor:
            return None
        values[rank:] = 0.0
        margins.append(values)
    if p1 * p2 - (p1 - left_rank) * (p2 - right_rank) != support.rank:
        return None
    return margins[0], margins[1], swapped


def _plain_multi_block(
    support,
    coordinate_map: NDArray | None,
    raw: Sequence[NDArray | None] | None,
    previous=None,
) -> _PlainBlock:
    """A multi-penalty block on its support; map-independent pieces come from ``previous``."""
    if previous is not None:
        roots, kronecker = previous.roots, previous.kronecker
    else:
        roots = _support_roots(support)
        kronecker = None if raw is None else _kronecker_margins(support, raw)
    return _PlainBlock(
        support.rank,
        roots=roots,
        volume=_map_volume(support, coordinate_map),
        kronecker=kronecker,
    )


def _plain_block(geometry, grouped: list[PenaltyComponent], matrices) -> _PlainBlock:
    """The group's plain block, kept on its context when it has one."""
    from superglm.reml.penalty_blocks import unit_logdet
    from superglm.reml.penalty_support import _penalty_support

    block = None if geometry is None else getattr(geometry, "plain", None)
    if block is not None:
        return block
    if geometry is not None:
        support, coordinate_map = geometry.get_support(), geometry.coordinate_map
    else:
        support, coordinate_map = _penalty_support(matrices), None
    if len(grouped) == 1:
        if coordinate_map is None:
            constant = unit_logdet(_support_roots(support)) if support.rank else 0.0
        else:
            diagonal = np.abs(np.diag(_transported_factor(support, coordinate_map)[1]))
            constant = 2.0 * math.fsum(np.log(diagonal).tolist())
        block = _PlainBlock(support.rank, constant=constant)
    else:
        raw = (
            tuple(component.omega_raw for component in grouped)
            if coordinate_map is not None
            else None
        )
        block = _plain_multi_block(support, coordinate_map, raw)
    if geometry is not None:
        geometry.plain = block
    return block


def _plain_block_evaluate(
    block: _PlainBlock, geometry, matrices, values: NDArray
) -> tuple[int, float, NDArray, NDArray]:
    """Rank, ``log|S|_+``, gradient and Hessian of one multi-penalty block at ``values``."""
    from superglm.reml.penalty_blocks import kronecker_sum_logdet, multi_penalty_logdet
    from superglm.reml.penalty_support import (
        PenaltyNumericalError,
        _penalty_support,
        _penalty_support_from_roots,
    )

    weights = tuple(map(float, values))
    if block.last is not None and block.last[0] == weights:
        return block.last[1]
    count = len(weights)
    active = np.asarray(values) > 0
    if block.rank == 0:
        result = (0, 0.0, np.zeros(count), np.zeros((count, count)))
    elif np.all(active):
        if block.kronecker is not None:
            left, right, swapped = block.kronecker
            left_weight, right_weight = (weights[1], weights[0]) if swapped else weights
            with np.errstate(divide="ignore"):
                logdet, gradient, hessian, _ = kronecker_sum_logdet(
                    math.log(left_weight) + np.log(left), math.log(right_weight) + np.log(right)
                )
            if swapped:
                gradient, hessian = gradient[::-1], hessian[::-1, ::-1]
        else:
            logdet, gradient, hessian = multi_penalty_logdet(block.roots, np.asarray(weights))
        result = (block.rank, math.fsum([logdet, block.volume]), gradient, hessian)
    else:
        pattern = tuple(bool(item) for item in active)
        face = block.faces.get(pattern)
        if face is None:
            support = geometry.get_support() if geometry is not None else _penalty_support(matrices)

            def active_support(component_roots):
                roots = [
                    root if on else np.empty((0, root.shape[1]))
                    for root, on in zip(component_roots, pattern, strict=True)
                ]
                return _penalty_support_from_roots(
                    roots,
                    resolution_limited=support.component_resolution_limited,
                    input_error_bounds=[np.zeros_like(root) for root in roots],
                )

            selected = active_support(support.component_roots)
            coordinate_map = None if geometry is None else geometry.coordinate_map
            # The solver stores the Grams of the transported roots, and the
            # nullity ranks those; a face whose transported roots keep another
            # rank than the raw ones is refused, as the certified context does.
            if (
                coordinate_map is not None
                and active_support(_transported_roots(support, coordinate_map)).rank
                != selected.rank
            ):
                raise PenaltyNumericalError("SSP active support cannot preserve the raw rank")
            face = (
                selected.rank,
                _support_roots(selected),
                _map_volume(selected, coordinate_map) if selected.rank else 0.0,
            )
            block.faces[pattern] = face
        rank, roots, volume = face
        gradient, hessian = np.zeros(count), np.zeros((count, count))
        logdet = 0.0
        if rank:
            indices = np.flatnonzero(active)
            logdet, local_gradient, local_hessian = multi_penalty_logdet(
                [roots[index] for index in indices], np.asarray(weights)[indices]
            )
            gradient[indices] = local_gradient
            hessian[np.ix_(indices, indices)] = local_hessian
        result = (rank, math.fsum([logdet, volume]), gradient, hessian)
    block.last = (weights, result)
    return result


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
            certified=geometry.certified,
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
    rank = int(round(float(component.rank)))
    if omega is not None and 0 < rank < omega.shape[0]:
        values, vectors = _penalty_range_basis(component, omega, rank)
        with np.errstate(over="ignore", invalid="ignore"):
            projection = vectors.T @ beta
        if np.all(np.isfinite(projection)):
            return float(np.sum(values * projection * projection))
        # As in _range_product, exact powers of two keep the projection
        # from overflowing before a small eigenvalue brings it into range.
        # The quadratic has two coefficient factors, hence 2 * beta_exponent.
        # Retain a finite unscaled projection above: normalizing a large
        # null-space coefficient could underflow a small range coefficient.
        beta_exponent = _binary_exponent(beta)
        finite = np.isfinite(projection)
        terms = np.empty_like(values)
        terms[finite] = values[finite] * projection[finite] * projection[finite]
        scaled_projection = vectors[:, ~finite].T @ np.ldexp(beta, -beta_exponent)
        # A single scale for all eigenvalues can erase a small eigenvalue
        # whose overflowing projection still contributes a large finite
        # energy. Scale each affected mode independently; keep every finite
        # original projection, including small coefficients beside the
        # overflowing mode.
        mantissa, exponent = np.frexp(values[~finite])
        terms[~finite] = np.ldexp(
            mantissa * scaled_projection * scaled_projection, 2 * beta_exponent + exponent
        )
        return float(np.sum(terms))
    return float(beta @ omega @ beta)


def _penalty_range_basis(
    component: PenaltyComponent, omega: NDArray, rank: int
) -> tuple[NDArray, NDArray]:
    """The ``rank`` largest eigenpairs of a dense rank-deficient penalty, its range.

    ``beta' Omega beta`` is evaluated over the penalty's range, the rank fixed
    by the component (``log |S|_+`` sums the same positive eigenvalues): the
    rank is pre-computed once, so the penalty and its pseudo-determinant
    agree at every lambda (Wood, Pya & Safken 2016, arXiv 1511.03864, section
    3.1.1).  The dense product instead charged ``lambda`` times the rounding
    of the null space's eigenvalue (``1.6e-15`` in a reparametrised main
    spline) times the squared null-space coefficient: about one unit of the
    REML objective at lambda 1e10 beside a 222 coefficient (an ``sz`` term
    beside weightless levels), so the objective became noise wherever REML
    drove the smoothing parameter up, and REML wandered to its bound.  Owner:
    the component, for its stored solver-space penalty (``omega_ssp``) once
    frozen read-only (``_frozen_array``), keyed on that array and the rank;
    lifetime: the component's; invalidation: another ``omega_ssp`` or rank.
    A penalty formed from a group matrix's current reparametrisation, or
    still writeable, is decomposed on each call.
    """
    cached = getattr(component, "_range_basis", None)
    if cached is not None and cached[0] is omega and cached[2].shape[1] == rank:
        return cached[1], cached[2]
    values, vectors = np.linalg.eigh(0.5 * (omega + omega.T))
    values, vectors = np.ascontiguousarray(values[-rank:]), np.ascontiguousarray(vectors[:, -rank:])
    if omega is component.omega_ssp and not omega.flags.writeable:
        component._range_basis = (omega, values, vectors)  # ty: ignore[unresolved-attribute] -- per-component cache, as _penalty_geometry
    return values, vectors


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
    rank = int(round(float(component.rank)))
    if omega is not None and 0 < rank < omega.shape[0]:
        # Over the penalty's range, as ``penalty_component_quadratic`` evaluates it:
        # ``V (Lambda (V' beta))`` keeps its rounding in the range.  The dense
        # product leaks rounding into the null space, where the penalty has no
        # curvature to absorb it, and at very large lambda that leak stalls PIRLS
        # (test_sz_reml_beside_weightless_levels_converges_from_a_large_main_lambda).
        values, vectors = _penalty_range_basis(component, omega, rank)
        return _range_product(values, vectors, beta)
    return omega @ beta


def penalty_product_rounding(width: int, components: Sequence[PenaltyComponent] | None) -> int:
    """``k`` with ``|fl(S v) - S v| <= gamma_k fl(|S| |v|)`` entrywise, for a penalty product.

    The product is either one formed dense ``S`` times ``v`` (``components``
    None), or the components' products summed as ``product[sl] += lam *
    penalty_component_matvec(...)`` (``irls_direct``'s ``penalty_matvec``,
    ``total_penalty_matvec``), with the magnitude ``|S| |v|`` summed the same
    way (``penalty_component_magnitude_matvec``).  Counting roundings along
    each entry's chain (Higham 2002, sections 3.1 and 3.5):

    - one component's entry, against its own magnitude: none for the
      identity; a ``b``-term dot for a repeated block (``b <= w``); a
      ``w``-term dot for a full-rank dense penalty; over a rank-deficient
      dense penalty's range, ``V' v`` (``w`` terms), the eigenvalue scaling
      and ``V (.)`` (``r`` terms), ``w + r + 1 <= 2 w`` since ``r < w`` (the
      powers of two of ``_range_product`` are exact); for sum-to-zero, the
      expansion's ``L - 2`` additions, a ``b``-term dot and the adjoint's
      subtraction, ``L + b - 1 <= 2 w`` for ``w = (L - 1) b``, ``L >= 2``.
      So at most ``2 w``, ``w`` the component's width, and none for an
      identity;
    - one more for ``lam`` times it, and ``m - 1`` for adding the ``m``
      components that cover the coefficient: at most ``2 w + m``;
    - the magnitude is formed by the same chain on non-negative terms, so
      the computed one is at least ``(1 - gamma_k)`` times the exact one, and
      ``gamma_k / (1 - gamma_k) <= gamma_{k+1}`` (when ``k (k + 1) u <= 1``)
      puts the bound on the computed magnitude: one more.

    That is ``2 w_max + m_max + 1`` over the components, ``w_max`` the
    widest that is not an identity and ``m_max`` the most covering one
    coefficient.  A formed dense
    ``S`` takes one ``width``-term dot and the magnitude's own rounding,
    ``width + 1``; ``width + 2``, the count these products' readers always
    used for it, covers that.
    """
    if components is None:
        return int(width) + 2
    covering = np.zeros(int(width), dtype=np.int64)
    widest = 0
    for component in components:
        covering[component.group_sl] += 1
        if component.penalty_kind != "identity":
            widest = max(widest, component.group_sl.stop - component.group_sl.start)
    return 2 * widest + max(int(np.max(covering, initial=0)), 1) + 1


def _binary_exponent(values: NDArray) -> int:
    """``e`` with the largest ``|values|`` in ``[2**(e - 1), 2**e)``; 0 when there is none."""
    largest = float(np.max(np.abs(values), initial=0.0))
    return int(np.frexp(largest)[1]) if np.isfinite(largest) and largest > 0.0 else 0


def _range_product(values: NDArray, vectors: NDArray, beta: NDArray) -> NDArray:
    """``V (Lambda (V' beta))``, with ``beta`` and ``Lambda`` brought to ``[1/2, 1)`` by powers of two.

    Each scaling is exact, and so is undoing both at the end, so wherever no
    intermediate leaves the normal range the result is bit for bit the
    unscaled product.  The scaling keeps ``V' beta`` from overflowing ahead
    of an eigenvalue small enough to bring the product back into range, and
    a subnormal eigenvalue from losing digits (AGENTS.md, numerical policy).
    A result outside the range is inf, or rounds towards 0, as the true one
    does; callers check it is finite.
    """
    beta_exponent = _binary_exponent(beta)
    value_exponent = _binary_exponent(values)
    scaled = vectors @ (
        np.ldexp(values, -value_exponent) * (vectors.T @ np.ldexp(beta, -beta_exponent))
    )
    return np.ldexp(scaled, beta_exponent + value_exponent)


def penalty_component_magnitude_matvec(
    component: PenaltyComponent,
    beta_magnitude: NDArray,
    group_matrix: GroupMatrix | None = None,
) -> NDArray:
    """Return ``|E|' |Omega| |E| |beta|``, the magnitude ``penalty_component_matvec`` sums.

    ``beta_magnitude`` is ``|beta|`` and ``E`` the component's expansion (the
    identity, ``[I; -1]`` for sum-to-zero, or the repeat).  Each entry that
    routine forms rounds to within ``gamma_m`` times this for ``m`` additions
    along its chain (Higham 2002, section 3.1), at most the group width plus two.
    A rank-deficient dense penalty's product over its range sums ``|V| |Lambda|
    |V'| |beta|`` instead, along a chain of at most twice the group width.
    """
    magnitude = np.asarray(beta_magnitude, dtype=np.float64)
    if component.penalty_kind == "identity":
        return magnitude.copy()
    signed = np.asarray(_penalty_component_omega_ssp(component, group_matrix), dtype=np.float64)
    rank = int(round(float(component.rank)))
    if component.penalty_kind == "dense" and 0 < rank < signed.shape[0]:
        values, vectors = _penalty_range_basis(component, signed, rank)
        return _range_product(np.abs(values), np.abs(vectors), magnitude)
    omega = np.abs(signed)
    if component.penalty_kind == "sum_to_zero":
        n_levels, block_width = _sum_to_zero_penalty_geometry(component)
        free = magnitude.reshape(n_levels - 1, block_width)
        raw = np.concatenate((free, free.sum(axis=0, keepdims=True))) @ omega.T
        return (raw[:-1] + raw[-1:]).ravel()
    if component.penalty_kind == "repeated":
        repeat_count, block_width = _repeated_penalty_geometry(component)
        return (magnitude.reshape(repeat_count, block_width) @ omega.T).ravel()
    return omega @ magnitude


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
) -> tuple[str | None, NDArray | None, NDArray | None]:
    """Recover a tensor penalty component's marginal block and its eigenvalues if possible."""
    q = p1 * p2
    if omega.shape != (q, q):
        return None, None, None

    norm = max(float(np.linalg.norm(omega)), 1e-300)
    omega4 = omega.reshape(p1, p2, p1, p2)

    left = 0.5 * (omega4[:, 0, :, 0] + omega4[:, 0, :, 0].T)
    right = 0.5 * (omega4[0, :, 0, :] + omega4[0, :, 0, :].T)
    left_err = float(np.linalg.norm(np.kron(left, np.eye(p2)) - omega) / norm)
    right_err = float(np.linalg.norm(np.kron(np.eye(p1), right) - omega) / norm)

    if left_err <= tol and left_err <= right_err:
        return "left", np.clip(np.linalg.eigvalsh(left), 0.0, None), left
    if right_err <= tol and right_err <= left_err:
        return "right", np.clip(np.linalg.eigvalsh(right), 0.0, None), right
    return None, None, None


def _lifted_positive(
    values: NDArray, eps_thresh: float, structural_rank: int | None, matrix: NDArray | None
) -> NDArray:
    """The eigenvalues REML counts positive: above its relative cut, or the structural rank's.

    As ``_rank_and_logdet`` does, a declared structural rank replaces a smaller
    threshold count only where ``matrix`` certifiably has that many positive
    eigenvalues (``_certifies_rank_at_least``); it then keeps the
    ``structural_rank`` largest.
    """
    from superglm.reml.multi_penalty import _certifies_rank_at_least

    values = np.asarray(values, dtype=np.float64)
    positive = values > eps_thresh * max(float(np.max(values, initial=0.0)), 1e-12)
    if (
        structural_rank is not None
        and matrix is not None
        and int(np.count_nonzero(positive)) < structural_rank <= values.size
        and np.sort(values)[::-1][structural_rank - 1] > 0
        and _certifies_rank_at_least(matrix, structural_rank)
    ):
        positive = np.zeros(values.shape, dtype=bool)
        positive[np.argsort(values)[::-1][:structural_rank]] = True
    return positive


def _marginal_structural_rank(gm: GroupMatrix, suffix: str, repeat: int) -> int | None:
    """A tensor margin's own structural rank from its component's (``rank(A) * repeat``)."""
    declared = (getattr(gm, "structural_ranks", None) or {}).get(suffix)
    if declared is None or repeat <= 0 or declared % repeat:
        return None
    return declared // repeat


def _tensor_marginal_rank_logdet(
    gm: GroupMatrix,
    omega_raw: NDArray,
    *,
    eps_thresh: float,
    suffix: str | None = None,
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
    side, eigvals, block = _extract_tensor_marginal_eigvals(omega_raw, p1, p2)
    if side is None or eigvals is None:
        return None

    eigvals = np.asarray(eigvals, dtype=np.float64)
    repeat = p2 if side == "left" else p1
    structural = None if suffix is None else _marginal_structural_rank(gm, suffix, repeat)
    pos = eigvals[_lifted_positive(eigvals, eps_thresh, structural, block)]
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
    eps_thresh = PENALTY_RANK_CUT
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

        extracted: dict[str, tuple[str, NDArray, int | None]] = {}
        for pc in pcs:
            omega = pc.omega_ssp
            if omega is None:
                continue
            side, eigvals, block = _extract_tensor_marginal_eigvals(omega, p1, p2)
            if side is None or eigvals is None:
                extracted.clear()
                break
            suffix = pc.name[len(group_name) + 1 :]
            repeat = p2 if side == "left" else p1
            structural = _marginal_structural_rank(gm, suffix, repeat)
            # The decision is the summary's: evaluation keeps that many.
            decided = (
                None
                if structural is None
                else int(np.count_nonzero(_lifted_positive(eigvals, eps_thresh, structural, block)))
            )
            extracted[side] = (pc.name, eigvals, decided)

        if set(extracted) != {"left", "right"}:
            if summary_cache is not None:
                summary_cache[cache_key] = None
            continue

        left_name, left_eigvals, left_rank = extracted["left"]
        right_name, right_eigvals, right_rank = extracted["right"]
        summary = TensorPairLogdetSummary(
            group_name=group_name,
            tensor_id=int(tensor_id),
            lambda_names=(left_name, right_name),
            eigvals_left=left_eigvals,
            eigvals_right=right_eigvals,
            rank_left=left_rank,
            rank_right=right_rank,
        )
        summaries[group_name] = summary
        if summary_cache is not None:
            summary_cache[cache_key] = summary
    return summaries


def evaluate_tensor_pair_logdet_summaries(
    summaries: dict[str, TensorPairLogdetSummary],
    lambdas: dict[str, float],
) -> dict[str, TensorPairLogdetEvaluation]:
    """Evaluate cached tensor summaries for one lambda dictionary (``kronecker_sum_logdet``)."""
    from superglm.reml.penalty_blocks import kronecker_sum_logdet

    evaluations: dict[str, TensorPairLogdetEvaluation] = {}
    eps_thresh = PENALTY_RANK_CUT

    def marginal_logs(
        values: NDArray, weight: float, structural_rank: int | None
    ) -> tuple[NDArray, NDArray]:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("tensor marginal spectra must be finite and non-negative")
        logs = np.full(values.shape, -np.inf)
        positive = values > eps_thresh * float(np.max(values, initial=0.0))
        if structural_rank is not None:
            positive = np.zeros(values.shape, dtype=bool)
            positive[np.argsort(values)[::-1][:structural_rank]] = True
        if weight > 0.0:
            # The unweighted marginal representative fixes support. No weighted
            # cutoff is allowed, including when a weighted eigenvalue overflows.
            logs[positive] = math.log(weight) + np.log(values[positive])
        return logs, positive

    for group_name, summary in summaries.items():
        left_name, right_name = summary.lambda_names
        lam_left = float(lambdas.get(left_name, 1.0))
        lam_right = float(lambdas.get(right_name, 1.0))
        if any(not math.isfinite(value) or value < 0 for value in (lam_left, lam_right)):
            raise ValueError("smoothing parameters must be finite and non-negative")
        left, left_positive = marginal_logs(summary.eigvals_left, lam_left, summary.rank_left)
        right, right_positive = marginal_logs(summary.eigvals_right, lam_right, summary.rank_right)
        logdet, gradient, hessian, rank = kronecker_sum_logdet(left, right)
        cross = float(hessian[0, 0])
        evaluations[group_name] = TensorPairLogdetEvaluation(
            group_name=group_name,
            tensor_id=summary.tensor_id,
            lambda_names=summary.lambda_names,
            logdet_s_plus=logdet,
            rank=rank,
            gradient={left_name: float(gradient[0]), right_name: float(gradient[1])},
            hessian={
                (left_name, left_name): cross,
                (right_name, right_name): cross,
                (left_name, right_name): -cross,
                (right_name, left_name): -cross,
            },
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


def ssp_penalty_matrix(coordinate_map: NDArray, omega: NDArray) -> NDArray:
    """``C' Omega C``, formed as the Gram ``(B C)'(B C)`` of a transported root ``B'B = Omega``.

    ``penalty_factor`` decides the penalty's rank from the formed matrix, so
    the formation has to keep the construction's null space.  The congruence
    ``C' Omega C`` rounds within ``gamma_k |C|'|Omega||C|`` (Higham 2002,
    *Accuracy and Stability of Numerical Algorithms*, eq. 3.13), which
    cancellation in ``C`` (a basis whitened on clustered data) makes far
    larger than the result: an exact null of ``Omega`` came out as a scaled
    eigenvalue of ``1.2e-14`` against a cut of ``8.9e-16``, and a data-null
    direction counted as identified (#434; that block's componentwise bound
    was ``6.7e-11``).  The Gram of a root with ``r = rank(Omega)`` rows has
    rank at most ``r`` exactly, and it rounds within ``gamma_r |BC|'|BC|``,
    at most ``gamma_r sqrt(S_ii S_jj)`` (Higham eq. 3.5 and Cauchy-Schwarz:
    a sum of squares keeps its relative accuracy), the scaled model
    ``penalty_factor`` cuts on.  ``B`` is ``penalty_factor``'s root of the
    raw ``Omega`` (formed without a congruence), so its rank is decided on
    ``Omega``'s own Jacobi equilibration, as every penalty root is.
    """
    from superglm.solvers.centered_system import penalty_factor

    coordinate_map = np.asarray(coordinate_map, dtype=np.float64)
    transported = penalty_factor(np.asarray(omega, dtype=np.float64)) @ coordinate_map
    gram = transported.T @ transported
    return 0.5 * (gram + gram.T)


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
                pc.omega_ssp
                if pc.omega_ssp is not None
                else ssp_penalty_matrix(gm.R_inv, pc.omega_raw)
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
                    S[g.sl, g.sl] += lam_j * ssp_penalty_matrix(gm.R_inv, omega_j)
                continue
            lam_g = lambda2.get(g.name, 0.0) if isinstance(lambda2, dict) else lambda2
            if lam_g == 0:
                continue
            omega = gm.omega
            if omega is None:
                continue
            S[g.sl, g.sl] += lam_g * ssp_penalty_matrix(gm.R_inv, omega)
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
    certified: bool = False,
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
    certified : bool
        Build the error ledgers the distributional smoother's certified
        evaluator reads (``_compute_penalty_logdet_evaluation``).  Otherwise
        the context serves the plain block-wise evaluator (``penalty_logdet``).
    _reuse_raw_from : list of PenaltyComponent
        The previous context of the same fit.  A plain context takes over a
        group's selected support, and a refused raw route, whenever the
        penalties it was selected from are unchanged.

    Returns
    -------
    list of PenaltyComponent, one per smoothing parameter.
    """
    components: list[PenaltyComponent] = []
    component_cache = None if cache is None else cache.setdefault("penalty_components", {})
    eps_thresh = PENALTY_RANK_CUT
    previous = {} if certified else _plain_geometries(_reuse_raw_from)

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
        structural_rank: int | None = None,
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

        ``structural_rank``, the rank the penalty's construction gives it
        (``GroupInfo.structural_ranks``), replaces a smaller threshold rank:
        the relative cut fixes units but not the spread within a penalty, and
        an integrated derivative penalty on uneven knots spreads past it. It
        is taken only where both penalties certifiably have that many positive
        eigenvalues (``_certifies_rank_at_least``, Rump's test on the leading
        eigenvectors), since a direction binary64 cannot certify positive is
        not penalised whatever the construction says. It also caps a larger
        solver-space count, where the raw count agrees with it.
        """
        from superglm.reml.multi_penalty import _certifies_rank_at_least

        # Rank from raw penalty (basis-invariant)
        raw_eigvals = np.linalg.eigvalsh(omega_raw)
        raw_thresh = eps_thresh * max(raw_eigvals.max(), 1e-12)
        raw_rank = float(np.sum(raw_eigvals > raw_thresh))

        # log|Ω|₊ and eigvals from SSP penalty (same basis as log|H|)
        ssp_eigvals, ssp_eigvectors = np.linalg.eigh(omega_ssp)
        ssp_thresh = eps_thresh * max(ssp_eigvals.max(), 1e-12)
        ssp_rank = float(np.sum(ssp_eigvals > ssp_thresh))
        rank = ssp_rank if force_solver_rank or raw_rank > omega_ssp.shape[0] else raw_rank
        if structural_rank is not None and raw_rank <= structural_rank < rank:
            # A congruence cannot raise a rank, rank(R' Ω R) <= rank(Ω), and the
            # construction gives Ω the structural rank. But it can compress the
            # spectrum by cond(R)**2 (Ostrowski; Horn and Johnson, Theorem
            # 4.5.9), past the relative cut, so round-off on a null direction
            # can count in SSP: 56 for 55 on ARM64 for a decomposed tensor's
            # margin, whose R_inv has cond(R)**2 = 3.8e10. The raw penalty,
            # counted without the congruence, must agree.
            rank = float(structural_rank)
        if (
            structural_rank is not None
            and rank < structural_rank <= omega_ssp.shape[0]
            and np.sort(ssp_eigvals)[::-1][structural_rank - 1] > 0
            and _certifies_rank_at_least(omega_raw, structural_rank)
            and _certifies_rank_at_least(omega_ssp, structural_rank, ssp_eigvectors)
        ):
            rank = float(structural_rank)

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
        structural_ranks = getattr(gm, "structural_ranks", None) or {}
        raw_support = None
        raw_coordinate_map = None
        raw_volume = raw_inputs = raw_selected = plain_block = None
        raw_refused = False
        prior = previous.get(g.name)
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
                suffixes = tuple(suffix for suffix, _ in repeated_components)
                if suffixes not in (("wiggle",), ("wiggle", "null")):
                    raise ValueError(
                        "SZ factor smooths require a 'wiggle' component, optionally "
                        "followed by its level lines' 'null' component."
                    )
                n_levels = gm.n_levels
                for suffix, omega_j in repeated_components:
                    local_rank, local_log_det, local_eigvals, omega_ssp_j = _rank_and_logdet(
                        omega_j,
                        omega_j,
                    )
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
                            # select=True's lines are a selection penalty, as Spline's null space
                            # is, so the REML bootstraps' no-signal snap targets them.  For this
                            # group, whose name holds ':', the bootstrap's step cap limits that
                            # move to e**4 in lambda per bootstrap step.
                            component_type="wiggle" if suffix == "wiggle" else "selection",
                            lambda_policy=lp_map.get(suffix),
                            penalty_kind="sum_to_zero",
                            repeat_count=n_levels,
                            block_width=gm.block_size,
                        )
                    )
                _attach_context_geometry(group_components, certified=certified)
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
                    raw_coordinate_map = coordinate_map
                    if certified:
                        raw_support = _penalty_support([omega for _, omega in gm.omega_components])
                        group_ssp_roots, group_ssp_errors = _ssp_component_roots(
                            raw_support, coordinate_map
                        )
                    else:
                        raw_inputs = tuple(_frozen_array(omega) for _, omega in gm.omega_components)
                        reused = _reusable_raw_context(prior, raw_inputs)
                        raw_support = raw_selected = (
                            _penalty_support(list(raw_inputs))
                            if reused is None
                            else reused.raw_support
                        )
                        group_ssp_roots = _transported_roots(raw_support, coordinate_map)
                        plain_block = _plain_multi_block(
                            raw_support,
                            coordinate_map,
                            raw_inputs,
                            previous=None if reused is None else reused.plain,
                        )
            for component_index, (suffix, omega_j) in enumerate(gm.omega_components):
                tensor_summary = _tensor_marginal_rank_logdet(
                    gm,
                    omega_j,
                    eps_thresh=eps_thresh,
                    suffix=suffix,
                )
                if raw_support is not None:
                    # Transport one selected raw family through the common map.
                    # Independent eigen-truncations of transformed Grams can
                    # rotate their null spaces and invent a union direction.
                    root = group_ssp_roots[component_index]
                    if certified:
                        omega_ssp_j, matrix_error = _enclosed_root_gram(
                            root, group_ssp_errors[component_index]
                        )
                        matrix_errors.append(matrix_error)
                    else:
                        omega_ssp_j = root.T @ root
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
                        structural_rank=structural_ranks.get(suffix),
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
                structural_rank=structural_ranks.get(""),
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
            # A structural lift takes the solver-space support; see
            # _single_penalty_raw_family for why the raw route cannot admit it.
            structural = structural_ranks.get("")
            lifted = (
                structural is not None
                and rank == structural
                and np.sum(
                    (raw_values := np.linalg.eigvalsh(gm.omega))
                    > eps_thresh * max(float(raw_values.max()), 1e-12)
                )
                < structural
            )
            if force_solver_rank or lifted:
                pass
            elif certified:
                raw = _single_penalty_raw_family(gm, group_components, rank, rank_rcond=eps_thresh)
                if raw:
                    (
                        raw_support,
                        raw_coordinate_map,
                        group_ssp_roots,
                        group_ssp_errors,
                        raw_volume,
                    ) = raw
            else:
                route = _plain_single_raw_route(
                    gm, group_components, rank, prior, rank_rcond=eps_thresh
                )
                if route is not None:
                    raw_inputs, raw_selected, raw_coordinate_map, plain_block = route
                    raw_refused = raw_selected is None
                    if plain_block is not None:
                        raw_support = raw_selected
        if raw_support is None and plain_block is None and prior is not None:
            # A solver-space support follows its matrices unchanged.
            reused = _reusable_solver_context(prior, group_components)
            if reused is not None:
                raw_support, plain_block = reused.support, reused.plain
        _attach_context_geometry(
            group_components,
            support=raw_support,
            coordinate_map=raw_coordinate_map,
            matrix_errors=matrix_errors,
            ssp_roots=group_ssp_roots if certified else (),
            ssp_errors=group_ssp_errors,
            volume=raw_volume,
            certified=certified,
            raw_inputs=raw_inputs,
            raw_support=raw_selected,
            raw_refused=raw_refused,
            plain=plain_block,
        )
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
        group_matrices,
        reml_groups,
        cache=cache,
        _reuse_raw_from=_reuse_raw_from,
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
    return penalty_logdet(lambdas, penalties).logdet


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
    """Numerical rank fallback for an already assembled PSD penalty: its root's.

    The root is ``penalty_factor``'s, cut block by block on the Jacobi
    equilibration at its eigensolver resolution ``n_b eps ||A_b||_2``, as
    every factor route's penalty root is.  A cut at ``eps ||S||_2`` of the
    whole matrix sat on the rounding of an exact null once the penalty is
    formed as the Gram of its root (``ssp_penalty_matrix``, #434): the count
    then changed between ``lambda e^(+-1e-5)``.
    """
    from superglm.solvers.centered_system import penalty_factor
    from superglm.solvers.rank import decompose_factor

    penalty_matrix = np.asarray(penalty_matrix, dtype=np.float64)
    if penalty_matrix.ndim != 2 or penalty_matrix.shape[0] != penalty_matrix.shape[1]:
        raise ValueError("penalty_matrix must be square")
    symmetric = 0.5 * (penalty_matrix + penalty_matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    negative_tolerance = 1e-10 * scale
    if eigenvalues.size and eigenvalues[0] < -negative_tolerance:
        raise ValueError("penalty_matrix must be positive semidefinite")
    return decompose_factor(penalty_factor(symmetric)).rank


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
    """``log|S_lambda|_+``, block by block (``penalty_logdet``)."""
    return penalty_logdet(lambdas, penalties, tensor_pair_evaluations).logdet


def compute_logdet_s_derivatives(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
    tensor_pair_evaluations: dict[str, TensorPairLogdetEvaluation] | None = None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Gradient and Hessian of ``log|S_lambda|_+`` in log weights (``penalty_logdet``)."""
    evaluation = penalty_logdet(lambdas, penalties, tensor_pair_evaluations)
    return evaluation.gradient, evaluation.hessian


def penalty_logdet(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
    tensor_pair_evaluations: dict[str, TensorPairLogdetEvaluation] | None = None,
) -> PenaltyLogdet:
    """``log|S_lambda|_+`` and its log-weight derivatives, one coefficient block at a time.

    Wood, Pya and Saefken (2016), sec. 3.1.1: an identity or single-penalty
    block adds ``rank log(lambda)`` to its unit-weight log-determinant, with
    gradient ``rank`` and no Hessian; a multi-penalty block is evaluated at its
    weights after the Wood (2011) Appendix B transform, or in closed form when
    it is a Kronecker sum (``penalty_blocks``).  Ranks come from each group's
    selected support, as in the context that built it; a supplied tensor-pair
    evaluation stands in for its group.
    """
    values = np.asarray([lambdas.get(component.name, 1.0) for component in penalties], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("smoothing parameters must be finite and non-negative")
    names = [component.name for component in penalties]
    if len(set(names)) != len(names):
        raise ValueError("penalty component names must be unique")
    gradient = dict.fromkeys(names, 0.0)
    hessian: dict[tuple[str, str], float] = {}
    log_terms = []
    rank = 0
    for group_name, indices in _group_penalties(penalties).items():
        grouped = [penalties[index] for index in indices]
        group_names = [component.name for component in grouped]
        group_values = values[indices]
        for name_i in group_names:
            for name_j in group_names:
                hessian[name_i, name_j] = 0.0
        if not np.any(group_values > 0):
            continue
        if tensor_pair_evaluations is not None and group_name in tensor_pair_evaluations:
            evaluation = tensor_pair_evaluations[group_name]
            rank += int(evaluation.rank)
            log_terms.append(evaluation.logdet_s_plus)
            gradient.update(evaluation.gradient)
            hessian.update(evaluation.hessian)
            continue
        component = grouped[0]
        if len(grouped) == 1 and component.penalty_kind == "identity":
            group_rank = component.group_sl.stop - component.group_sl.start
            rank += group_rank
            log_terms.append(group_rank * math.log(float(group_values[0])))
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
            if component.penalty_kind == "sum_to_zero":
                # The local blocks repeat K - 1 times under (I + 1 1') kron P,
                # whose determinant adds rank(P) log K.
                extra_volume = math.log(_sum_to_zero_penalty_geometry(component)[0])
        else:
            matrices, repeat = _group_penalty_matrices(grouped)
        block = _plain_block(geometry, grouped, matrices)
        if len(grouped) == 1:
            group_rank = repeat * block.rank
            rank += group_rank
            log_terms.append(
                math.fsum(
                    [
                        repeat * block.constant,
                        block.rank * extra_volume,
                        group_rank * math.log(float(group_values[0])),
                    ]
                )
            )
            gradient[component.name] = float(group_rank)
            continue
        block_rank, block_logdet, block_gradient, block_hessian = _plain_block_evaluate(
            block, geometry, matrices, group_values
        )
        rank += repeat * block_rank
        log_terms.append(math.fsum([repeat * block_logdet, block_rank * extra_volume]))
        for i, name_i in enumerate(group_names):
            gradient[name_i] = float(repeat * block_gradient[i])
            for j, name_j in enumerate(group_names):
                hessian[name_i, name_j] = float(repeat * block_hessian[i, j])
    return PenaltyLogdet(rank=rank, logdet=math.fsum(log_terms), gradient=gradient, hessian=hessian)


def _single_penalty_raw_family(gm, grouped, declared_rank, *, rank_rcond=None):
    """Select one dense penalty on its raw basis, independently of the SSP map.

    The solver penalty is ``C.T @ Omega @ C`` for the group's SSP coordinate
    map ``C = R_inv``, which discrete REML and EFS rebuild whenever lambda
    changes. With ``U`` an orthonormal basis of range(Omega) and
    ``Omega = U L U.T``, the nonzero eigenvalues of ``C.T U L U.T C`` are those
    of ``L^1/2 U.T C C.T U L^1/2``, because ``AB`` and ``BA`` share their nonzero
    spectrum. Hence, whenever ``U.T C`` has full row rank,

        log pdet(C.T Omega C) = log pdet(Omega) + log det(U.T C C.T U),

    the congruence case of Cauchy-Binet for pseudo-determinants (Knill 2014,
    Linear Algebra Appl. 459). The rank and first term depend on Omega alone;
    ``_support_coordinate_volume`` certifies the second term and that ``U.T C``
    has full row rank for each map. The stored solver penalty is unchanged,
    and ``_solver_penalty_agreement`` certifies, for every map, a Weyl gap in
    the stored penalty at the raw rank (one-sided; see there) and adds the
    Weyl term between the log pseudo-determinants to the volume's error.

    The raw support is selected at ``rank_rcond``, the relative threshold
    that set ``declared_rank``, and only where the raw spectrum has a gap
    there (``penalty_support._gapped_rank_mask``). At the eigensolver's
    resolution floor instead, the formation round-off in a constrained
    penalty's exact null space (cr: about 0.1 of the floor on x86-64) could
    add a direction on another BLAS build and refuse the raw path there.
    A structural rank above that threshold's count (``_rank_and_logdet``)
    is not set by any threshold, and the caller does not come here: the raw
    route cannot admit it. The lifted direction's raw eigenvalue is below
    ``eps**(2/3)`` of the largest by definition, while the raw root and its
    transport carry absolute errors of order ``u ||Omega||``, so the
    agreement ratio ``x`` of ``_solver_penalty_agreement`` is of order
    ``u / eps**(2/3)``, about ``3e-6`` (lognormal(0, 1.5) bs with 20
    quantile_rows knots: ``x = 1.0e-6``), against its admission bar
    ``sqrt(eps)``, ``1.5e-8``. The solver-space support evaluates the same
    penalty in SSP coordinates, where that direction is 9e-5 of the largest.

    Returns ``None`` when the cheap map checks decline (non-float64 inputs,
    inconsistent shapes, a non-finite map, or ``np.eye``, which is either the
    raw basis or a coordinate truncation), and ``False`` when the raw support
    cannot be selected, its rank differs from the declared solver rank, or
    the volume or agreement cannot be certified.  Either way the group keeps
    its solver-space support.
    """
    from superglm.reml.penalty_support import PenaltyNumericalError, _penalty_support

    omega = getattr(gm, "omega", None)
    coordinate_map = getattr(gm, "R_inv", None)
    if omega is None or coordinate_map is None or len(grouped) != 1:
        return None
    omega, coordinate_map = np.asarray(omega), np.asarray(coordinate_map)
    stored = grouped[0].omega_ssp
    if (
        omega.dtype != np.float64
        or coordinate_map.dtype != np.float64
        or coordinate_map.shape != (omega.shape[0], stored.shape[0])
        or not np.all(np.isfinite(coordinate_map))
        or np.array_equal(coordinate_map, np.eye(*coordinate_map.shape))
    ):
        return None
    try:
        support = _penalty_support([omega], rcond=rank_rcond)
    except (PenaltyNumericalError, ValueError):
        # Its ValueErrors are the raw matrix's own admissibility checks
        # (symmetry, semidefiniteness), which the solver-space path never
        # applies to the raw penalty, so they refuse rather than fail.
        return False
    if support.rank != declared_rank:
        return False
    try:
        products = _shared_map_products(support, coordinate_map)
        volume, volume_error = _support_coordinate_volume(
            support, coordinate_map, products=products
        )
        roots, errors = _ssp_component_roots(support, coordinate_map, products=products)
        agreement = _solver_penalty_agreement(support, stored, roots[0], errors[0])
    except PenaltyNumericalError:
        return False
    volume_error = float(_enclosed_bound_sum(volume_error, agreement))
    return support, coordinate_map, roots, errors, (volume, volume_error)


def _plain_geometries(
    source: Sequence[PenaltyComponent] | None,
) -> dict[str, _PenaltyGroupGeometry]:
    """The plain contexts of a previous build, by group name."""
    if not source:
        return {}
    found = {}
    for name, indices in _group_penalties(list(source)).items():
        geometry = _context_geometry([source[index] for index in indices])
        if geometry is not None and not geometry.certified:
            found[name] = geometry
    return found


def _same_arrays(left: Sequence[NDArray], right: Sequence[NDArray]) -> bool:
    return len(left) == len(right) and all(
        a is b
        or (
            np.shape(a) == np.shape(b)
            and np.asarray(a).dtype == np.asarray(b).dtype
            and np.array_equal(a, b)
        )
        for a, b in zip(left, right, strict=True)
    )


def _reusable_raw_context(prior, raw_inputs: tuple[NDArray, ...]):
    """The previous context when it selected, or refused, a raw support from these raw penalties."""
    if prior is None or prior.raw_inputs is None or not _same_arrays(prior.raw_inputs, raw_inputs):
        return None
    if not prior.raw_refused and prior.raw_support is None:
        return None
    return prior


def _transport_rounding(support, coordinate_map: NDArray) -> NDArray:
    """A componentwise bound on the rounding of ``_transported_roots``' single root.

    A computed product ``fl(A B)`` is within ``gamma_k |A| |B|`` of ``A B`` for
    inner dimension ``k`` (Higham 2002, sec. 3.5).  The full-width root is
    ``R C``.  Otherwise it is ``K M`` with ``K = R Q`` and ``M = (Q' Q)^-1 Q' C``,
    where ``Q' Q`` is the identity to working accuracy, so ``M`` carries the
    rounding of ``Q' C``, of the Gram ``Q' Q`` it is solved against, and of the
    solve, ``gamma_w (|Q|' |C| + |Q|' |Q| |M|) + gamma_2r |M|``; the product adds
    ``gamma_r |K| |M|`` and ``K``'s own ``gamma_w |R| |Q| |M|``.
    """
    from superglm._roundoff import gamma

    source = support.component_roots[0]
    width, rank = support.Q_plus.shape
    absolute_map = np.abs(coordinate_map)
    if rank == width:
        return gamma(width) * (np.abs(source) @ absolute_map)
    basis = support.Q_plus
    retained = np.abs(
        scipy.linalg.solve(
            basis.T @ basis, basis.T @ coordinate_map, assume_a="pos", check_finite=False
        )
    )
    coordinates, absolute_basis = np.abs(source @ basis), np.abs(basis)
    retained_rounding = (
        gamma(width)
        * (absolute_basis.T @ absolute_map + (absolute_basis.T @ absolute_basis) @ retained)
        + gamma(2 * rank) * retained
    )
    return (
        gamma(rank) * (coordinates @ retained)
        + gamma(width) * ((np.abs(source) @ absolute_basis) @ retained)
        + coordinates @ retained_rounding
    )


def _reusable_solver_context(prior, grouped: list[PenaltyComponent]):
    """The previous solver-space context when its stored matrices are these ones."""
    stored = [component.omega_ssp for component in grouped if component.omega_ssp is not None]
    if (
        prior.coordinate_map is not None
        or prior.support is None
        or len(stored) != len(grouped)
        or not _same_arrays(prior.matrices, stored)
    ):
        return None
    return prior


def _frobenius(values: NDArray) -> float:
    """``||A||_F`` of ``A`` scaled by the power of two at its largest entry.

    The scaling is exact, so this is the unscaled norm wherever that neither
    underflows nor overflows, and it stays nonzero wherever the largest entry
    is: ``1e-200`` entries have squares below binary64.  A norm beyond binary64
    is infinite, which no agreement bar admits.
    """
    largest = float(np.max(np.abs(values), initial=0.0))
    if largest == 0 or not math.isfinite(largest):
        return largest
    exponent = math.frexp(largest)[1]
    try:
        return math.ldexp(float(np.linalg.norm(np.ldexp(values, -exponent))), exponent)
    except OverflowError:
        return math.inf


def _plain_single_raw_route(gm, grouped, declared_rank, prior, *, rank_rcond):
    """A single penalty's raw support and unit log-determinant under the SSP map.

    The raw support is selected once per fit, on the raw penalty, at the
    threshold that set the declared rank (``_single_penalty_raw_family``);
    each map then contributes the transported root ``Y = R P C``, and
    ``log pdet(C' Omega C) = log det(Y Y')``, from a QR factor of ``Y'``.  The
    solver keeps its stored penalty ``S``.  The raw route stands in for it when, with
    ``X = Y' Y`` for the exact ``Y`` and ``mu <= sigma_min(Y)**2``,
    ``||S - X||_F <= sqrt(eps) mu`` and ``mu - ||S - X||_F`` clears the shared
    Gram rank bar ``max(eps, p(n) eps) ||S||_F``: by Weyl's inequality ``S``
    then has the raw rank across a gap, and its log pseudo-determinant is
    within ``r sqrt(eps)`` of the raw one.  These are the agreement test's bars
    and its lower bound ``mu`` from the inverse triangular factor of ``Y'``
    (``_solver_penalty_agreement``).  ``||S - X||_F`` is bounded by
    ``||S - G||_F`` for the computed Gram ``G``, plus ``gamma_r |Y|' |Y|``
    for ``G``'s own rounding (Higham 2002, sec. 3.5) and ``2 |Y|' E + E' E``
    for the rounding ``E`` of ``Y`` (``_transport_rounding``): ``S`` is itself a
    rounded congruence, and the computed ``||S - G||_F`` alone can share its
    rounding and read near zero where ``S`` has lost the raw spectrum.

    Returns ``None`` when the map is declined before any raw work, and
    otherwise ``(inputs, support, map, block)``.  A refusal that no map can
    change (the selection, or its rank) returns no support, and later
    contexts of the fit refuse at once.  A map whose agreement bars refuse
    returns the support without a map or block, so the next map rechecks them.
    """
    from superglm._roundoff import gamma
    from superglm.reml.penalty_support import PenaltyNumericalError, _penalty_support
    from superglm.solvers.rank import SHARED_RANK_POLICY, _eigensolver_relative_bar

    omega = getattr(gm, "omega", None)
    coordinate_map = getattr(gm, "R_inv", None)
    if omega is None or coordinate_map is None or len(grouped) != 1:
        return None
    omega, coordinate_map = np.asarray(omega), np.asarray(coordinate_map)
    stored = grouped[0].omega_ssp
    if (
        omega.dtype != np.float64
        or coordinate_map.dtype != np.float64
        or coordinate_map.shape != (omega.shape[0], stored.shape[0])
        or not np.all(np.isfinite(coordinate_map))
        or np.array_equal(coordinate_map, np.eye(*coordinate_map.shape))
    ):
        return None
    inputs = (_frozen_array(omega),)
    reused = _reusable_raw_context(prior, inputs)
    if reused is not None and reused.raw_refused:
        return inputs, None, None, None
    if reused is not None:
        support = reused.raw_support
    else:
        try:
            support = _penalty_support([omega], rcond=rank_rcond)
        except (PenaltyNumericalError, ValueError):
            # The raw matrix's own admissibility checks refuse the route; the
            # solver-space path never applies them to the raw penalty.
            return inputs, None, None, None
    if support.rank == 0 or support.rank != declared_rank:
        return inputs, None, None, None
    unadmitted = (inputs, support, None, None)
    root, upper = _transported_factor(support, coordinate_map)
    diagonal = np.abs(np.diag(upper))
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0):
        return unadmitted
    # With Y' = Q U and T = U^-1, sigma_min(Y)**2 = 1 / ||T||_2**2 >= 1 / || |T|' |T| ||_inf.
    inverse = np.abs(scipy.linalg.solve_triangular(upper, np.eye(len(root)), check_finite=False))
    smallest = 1.0 / float(np.max(np.sum(inverse.T @ inverse, axis=1)))
    stored = np.asarray(stored, dtype=np.float64)
    magnitude, rounding = np.abs(root), _transport_rounding(support, coordinate_map)
    allowance = gamma(len(root)) * (magnitude.T @ magnitude) + 2.0 * (magnitude.T @ rounding)
    allowance += rounding.T @ rounding
    discrepancy = _frobenius(stored - root.T @ root) + _frobenius(allowance)
    bar = max(SHARED_RANK_POLICY.gram_rcond, _eigensolver_relative_bar(len(stored)))
    if not (
        discrepancy <= SHARED_RANK_POLICY.factor_rcond * smallest
        and smallest - discrepancy > bar * _frobenius(stored)
    ):
        return unadmitted
    block = _PlainBlock(support.rank, constant=2.0 * math.fsum(np.log(diagonal).tolist()))
    return inputs, support, coordinate_map, block


def _solver_penalty_agreement(support, stored: NDArray, root: NDArray, root_error: NDArray):
    """Bound the stored solver penalty's log pseudo-determinant by the raw target's.

    The raw path evaluates ``log pdet(X)`` for ``X = Y.T @ Y``, where
    ``Y = R P C`` carries the selected raw root ``R`` through the exact
    projector ``P`` onto range(Q_plus) and the SSP map ``C``; ``root`` encloses
    ``Y`` within ``root_error`` (``_ssp_component_roots``). PIRLS, log|H| and
    beta' S beta use the stored ``omega_ssp`` ``S`` instead, a separately
    rounded and truncated congruence.

    With ``eps >= ||S - X||_2`` and ``0 < mu <= lambda_r(X)``, Weyl's
    inequality ``max_i |lambda_i(S) - lambda_i(X)| <= ||S - X||_2`` (Weyl
    1912; Horn and Johnson, Matrix Analysis, 2nd ed., 2013, section 4.3)
    leaves ``S`` with exactly ``r`` eigenvalues of at least ``mu - eps`` and
    the rest in ``[-eps, eps]`` when ``2 eps < mu``: ``S`` has a gap at ``r``
    and, with ``x = eps / mu``,

        |log pdet_r(S) - log pdet(X)| <= r * -log(1 - x) <= r x / (1 - x).

    ``eps`` is the Frobenius norm of an elementwise enclosure of
    ``|S - G| + |G - X|``, where ``G`` is the rounded Gram of ``root``
    (Higham 2002, section 3.5). ``mu`` whitens ``Y.T``: for the triangular
    ``T`` with ``||T.T Y Y.T T - I||_2 <= eta < 1``, every nonzero eigenvalue
    of ``X``, an eigenvalue of ``Y Y.T``, is at least
    ``(1 - eta) / ||T||_2**2``, and ``||T||_2**2 <= || |T|.T |T| ||_inf``
    (compare Rump, BIT 51(2), 2011, for verified smallest singular values).

    The raw path only stands in for the solver-space support, so it is
    admitted when ``x <= sqrt(eps)``, the shared factor rank resolution, which
    keeps the second-order term ``x**2`` at or below machine epsilon, and when
    ``mu - eps`` clears the shared Gram rank bar for ``S``. Returns the bound
    above; raises ``PenaltyNumericalError`` otherwise.

    The rank claim is one-sided. The retained eigenvalues of ``S`` clear the
    rank bar, but its trailing eigenvalues are certified only to lie within
    ``eps`` of zero, not below that bar: the enclosures put ``eps`` above
    ``bar * ||S||_2`` (42 to 8e4 times on pg17 step A discrete), so the
    upper-side check would refuse every context. ``S`` is canonicalised to the
    declared rank, so its tail is rounding residue.
    """
    from superglm.reml.multi_penalty import (
        _finite_double,
        _gamma,
        _matmul_enclosed,
        _norm_upper,
        _positive_product,
        _triangular_solve,
        _upper,
    )
    from superglm.reml.penalty_support import PenaltyNumericalError
    from superglm.solvers.rank import SHARED_RANK_POLICY, _eigensolver_relative_bar

    rank, tiny = support.rank, np.nextafter(0.0, 1.0)
    if rank == 0 or root.shape[0] != rank:
        raise PenaltyNumericalError("raw target rows do not match its selected rank")
    gram, gram_error = _enclosed_root_gram(root, root_error)
    difference = np.asarray(stored, dtype=np.float64) - gram
    eps = _norm_upper(
        _enclosed_bound_sum(np.abs(difference), _gamma(1) * np.abs(difference) + tiny, gram_error)
    )
    _, upper = scipy.linalg.qr(root.T, mode="economic", check_finite=False)
    if upper.shape != (rank, rank) or np.any(np.diag(upper) == 0):
        raise PenaltyNumericalError("raw target loses a direction under the SSP map")
    triangular = _finite_double(
        _triangular_solve(upper[::-1, ::-1], np.eye(rank)[::-1])[::-1],
        "solver agreement preconditioner",
    )
    whitened, arithmetic = _matmul_enclosed(root.T, triangular)
    action_error = _enclosed_bound_sum(
        arithmetic, _positive_product(root_error.T, np.abs(triangular))
    )
    whitened_gram, whitened_error = _enclosed_root_gram(whitened, action_error)
    defect = whitened_gram - np.eye(rank)
    eta = _norm_upper(
        _enclosed_bound_sum(np.abs(defect), _gamma(1) * np.abs(defect) + tiny, whitened_error)
    )
    if eta >= 1:
        raise PenaltyNumericalError("raw target's smallest retained eigenvalue is unresolved")
    absolute = np.abs(triangular)
    norm_squared = np.max(_enclosed_bound_sum(*_positive_product(absolute.T, absolute).T))
    mu = float((1 - eta) / norm_squared * (1 - _gamma(4)))
    if not mu > 0:
        raise PenaltyNumericalError("raw target's smallest retained eigenvalue is unresolved")
    x = float(_upper(eps / mu / (1 - _gamma(1))))
    bar = max(SHARED_RANK_POLICY.gram_rcond, _eigensolver_relative_bar(len(stored)))
    cutoff = float(_upper(bar * _norm_upper(stored) / (1 - _gamma(2))))
    if x > SHARED_RANK_POLICY.factor_rcond or (mu - eps) * (1 - _gamma(1)) <= cutoff:
        raise PenaltyNumericalError("stored solver penalty does not agree with its raw target")
    return float(_upper(rank * x / (1 - x) / (1 - _gamma(4))))


def _compute_penalty_logdet_evaluation(
    lambdas: dict[str, float],
    penalties: list[PenaltyComponent],
) -> _PenaltyLogdetEvaluation:
    """Evaluate rank, determinant and derivatives, with error bounds, per group.

    The distributional smoother's evaluator: it reads the bounds. Singleton
    and compact repeated identities follow Wood (2011), section 3.1. Their
    numerical ranks come from the shared root support, never stale component
    metadata. Bounds concern that selected representative, so a context built
    for the plain evaluator, which carries no error ledgers, is not reused.
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
    unit = UNIT_ROUNDOFF
    for group_name, indices in _group_penalties(penalties).items():
        grouped = [penalties[index] for index in indices]
        group_names = [component.name for component in grouped]
        group_values = values[indices]
        for name_i in group_names:
            for name_j in group_names:
                hessian[name_i, name_j] = hessian_error[name_i, name_j] = 0.0
        if not np.any(group_values > 0):
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
        if geometry is not None and not geometry.certified:
            geometry = None
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
            # A raw-support context adds its certified SSP coordinate volume,
            # whose error includes the Weyl term to the stored solver penalty;
            # a solver-space support has neither.
            volume = volume_error = 0.0
            try:
                if geometry is not None:
                    result, volume, volume_error = geometry.evaluate(np.ones(1))
                else:
                    result = _evaluate_penalty_support(support, np.ones(1))
            except PenaltyNumericalError as exc:
                if str(exc) != "required dense penalty inverse is not representable":
                    raise
                # The actual weighted system may have a representable inverse
                # even when a component's arbitrary units make P^-1 overflow.
                if geometry is not None:
                    result, volume, volume_error = geometry.evaluate(group_values)
                else:
                    result = _evaluate_penalty_support(support, group_values)
                log_weight = 0.0
            group_rank = repeat * result.rank
            term = math.fsum(
                [
                    repeat * result.logdet_s_plus,
                    repeat * volume,
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
                repeat * (certificate.logdet_error + volume_error)
                + 8
                * unit
                * (
                    abs(repeat * result.logdet_s_plus)
                    + abs(repeat * volume)
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
            if component.penalty_kind == "sum_to_zero":
                # The local blocks repeat K - 1 times under (I + 1 1') kron P,
                # whose determinant adds rank(P) log K (the one-component case).
                extra_volume = math.log(_sum_to_zero_penalty_geometry(component)[0])
        certificate = result._certificate
        if certificate is None:
            raise ValueError("penalty evaluation did not provide arithmetic evidence")
        rank += repeat * result.rank
        term = repeat * math.fsum([result.logdet_s_plus, volume])
        error = repeat * (certificate.logdet_error + volume_error) + 4 * unit * repeat * (
            abs(result.logdet_s_plus) + abs(volume)
        )
        if extra_volume:
            term = math.fsum([term, result.rank * extra_volume])
            error += 4 * unit * (abs(term) + abs(result.rank * extra_volume))
        log_terms.append(term)
        log_errors.append(error)
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
