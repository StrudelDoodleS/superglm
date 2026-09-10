"""Unweighted PSD representatives and balanced common penalty support.

Component balancing follows Wood (2011), section 3.1 and Appendix B. The
factor-space construction and retained input evidence do not use smoothing
magnitudes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.solvers.rank import SHARED_RANK_POLICY, decompose_gram

_EPS = np.finfo(np.float64).eps
_LD = np.longdouble


class PenaltyNumericalError(np.linalg.LinAlgError):
    """Finite penalty geometry could not meet its numerical contract."""


def _readonly(value: NDArray) -> NDArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _finite_double(value: NDArray, name: str) -> NDArray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise PenaltyNumericalError(f"{name} is not representable")
    return result


@dataclass(frozen=True)
class _PenaltySupport:
    component_roots: tuple[NDArray, ...]
    balanced_coordinates: tuple[NDArray, ...]
    root_log_scales: NDArray
    coordinate_map: NDArray
    coordinate_triangular: NDArray
    Q_plus: NDArray
    Q_zero: NDArray
    component_resolution_limited: tuple[bool, ...]
    component_reconstruction_bounds: tuple[NDArray, ...]
    support_projection_bounds: tuple[NDArray, ...]
    component_root_error_bounds: tuple[NDArray, ...]
    _basis_gram_evidence: object | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def rank(self) -> int:
        return self.Q_plus.shape[1]

    def __getstate__(self) -> dict[str, object]:
        # Derived evidence binds live array/callable identities and precision.
        # Reconstructed or copied support must acquire its own evidence.
        state = self.__dict__.copy()
        state["_basis_gram_evidence"] = None
        return state


def _validated_matrix(matrix: NDArray) -> NDArray:
    source = np.asarray(matrix)
    if np.iscomplexobj(source):
        raise ValueError("penalty matrices must be real")
    values = np.asarray(source, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("penalty matrices must be square")
    if not np.all(np.isfinite(values)):
        raise ValueError("penalty matrices must be finite")
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale:
        normalized = values.astype(_LD) / _LD(scale)
        allowance = 4 * max(len(values), 1) * _EPS
        if np.max(np.abs(normalized - normalized.T), initial=0) > allowance:
            raise ValueError("penalty matrices must be symmetric")
        if np.min(np.diag(normalized), initial=0) < -allowance:
            raise ValueError("penalty matrices must be positive semidefinite")
    # Equal entries need no arithmetic: halving a subnormal before adding it
    # back can erase or change an exactly symmetric input. For unequal pairs,
    # sum first when safe; the other branch cannot overflow its partial sum.
    symmetric = values.copy()
    rows, columns = np.triu_indices(len(values), k=1)
    left, right = values[rows, columns], values[columns, rows]
    different = left != right
    safe_sum = np.maximum(np.abs(left), np.abs(right)) <= np.finfo(float).max / 2
    average = left.copy()
    small = different & safe_sum
    large = different & ~safe_sum
    average[small] = (left[small] + right[small]) * 0.5
    average[large] = left[large] * 0.5 + right[large] * 0.5
    symmetric[rows, columns] = symmetric[columns, rows] = average
    return symmetric


def _component_root(matrix: NDArray) -> tuple[NDArray, bool, NDArray]:
    """Root of the full selected PSD representative, plus Gram evidence."""
    values = _validated_matrix(matrix)
    try:
        d = decompose_gram(values, allow_indefinite=False, fallback_factor=None)
    except ValueError as exc:
        raise ValueError("penalty matrices must be positive semidefinite") from exc
    if d.rank == 0:
        root = np.empty((0, d.width))
    elif d.method == "cholesky":
        root = np.zeros((d.rank, d.width))
        root[:, d.active_columns] = d.cholesky_factor.T * d.column_scale[d.active_columns]
    elif d.method in {"gram_eigh", "pivoted_cholesky"}:
        retained = d.retained_values
        if retained is None or not np.all(np.isfinite(retained)) or np.any(retained <= 0):
            raise PenaltyNumericalError("invalid retained component spectrum")
        root = np.sqrt(retained)[:, None] * d.estimable_functional_basis.T
    else:
        raise PenaltyNumericalError("unsupported component root representation")
    root = _finite_double(root, "component root")
    scale = d.column_scale.astype(_LD)
    scale = np.where(scale > 0, scale, _LD(1))
    equilibrated = values.astype(_LD) / scale[:, None] / scale[None, :]
    scaled_root = root.astype(_LD) / scale
    represented = scaled_root.T @ scaled_root
    bound = np.abs(equilibrated - represented)
    bound += 4 * max(d.width, 1) * _EPS * (np.abs(scaled_root).T @ np.abs(scaled_root))
    return (
        _readonly(root),
        bool(d.resolution_limited),
        _readonly(_finite_double(bound, "root bound")),
    )


def _penalty_support(penalty_matrices: Sequence[NDArray]) -> _PenaltySupport:
    if not penalty_matrices:
        raise ValueError("at least one penalty matrix is required")
    extracted = tuple(_component_root(matrix) for matrix in penalty_matrices)
    if len({root.shape[1] for root, _, _ in extracted}) != 1:
        raise ValueError("penalty matrices must have a common shape")
    support = _penalty_support_from_roots(
        [root for root, _, _ in extracted],
        resolution_limited=[limited for _, limited, _ in extracted],
        input_error_bounds=[np.zeros_like(root) for root, _, _ in extracted],
    )
    return replace(
        support, component_reconstruction_bounds=tuple(bound for _, _, bound in extracted)
    )


def _penalty_support_from_roots(
    component_roots: Sequence[NDArray],
    *,
    resolution_limited: Sequence[bool],
    input_error_bounds: Sequence[NDArray],
) -> _PenaltySupport:
    """Select unweighted support from roots with componentwise root errors.

    Each input error bound has the same shape as its root. Gram reconstruction
    evidence from matrix decomposition is a separate ledger, not a root error.
    """
    if any(np.iscomplexobj(root) for root in component_roots):
        raise ValueError("component roots must be real")
    roots = tuple(np.asarray(root, dtype=np.float64) for root in component_roots)
    if not roots or any(root.ndim != 2 or not np.all(np.isfinite(root)) for root in roots):
        raise ValueError("component roots must be nonempty finite matrices")
    width = roots[0].shape[1]
    if any(root.shape[1] != width for root in roots):
        raise ValueError("component roots must have a common width")
    if len(resolution_limited) != len(roots) or len(input_error_bounds) != len(roots):
        raise ValueError("one resolution flag and error bound per component are required")
    root_errors = tuple(np.asarray(bound, dtype=np.float64) for bound in input_error_bounds)
    if any(
        bound.shape != root.shape or not np.all(np.isfinite(bound)) or np.any(bound < 0)
        for bound, root in zip(root_errors, roots, strict=True)
    ):
        raise ValueError("root error bounds must match roots and be finite and non-negative")
    scales, balanced = [], []
    for root in roots:
        maximum = np.max(np.abs(root), initial=0.0)
        scale = (
            _LD(maximum) * np.sqrt(np.sum((root.astype(_LD) / maximum) ** 2))
            if maximum > 0
            else _LD(1)
        )
        scales.append(scale)
        balanced.append(root.astype(_LD) / scale)
    stacked = np.vstack(balanced)
    column_max = np.max(np.abs(stacked), axis=0, initial=_LD(0))
    normalized = np.zeros_like(stacked)
    np.divide(stacked, column_max, out=normalized, where=column_max > 0)
    column_scale = column_max * np.sqrt(np.sum(normalized * normalized, axis=0))
    active = np.flatnonzero(column_scale > 0)
    if not len(active):
        return _PenaltySupport(
            tuple(_readonly(root) for root in roots),
            tuple(_readonly(np.empty((len(root), 0))) for root in roots),
            _readonly(np.zeros(len(roots))),
            _readonly(np.empty((width, 0))),
            _readonly(np.empty((0, 0))),
            _readonly(np.empty((width, 0))),
            _readonly(np.eye(width)),
            tuple(bool(value) for value in resolution_limited),
            tuple(_readonly(np.zeros((width, width))) for _ in roots),
            tuple(_readonly(np.zeros_like(root)) for root in roots),
            tuple(_readonly(bound) for bound in root_errors),
        )
    factor = _finite_double(stacked[:, active] / column_scale[active], "balanced support factor")
    _, singular, vh = scipy.linalg.svd(factor, full_matrices=False, check_finite=False)
    cutoff = SHARED_RANK_POLICY.factor_rcond * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    vectors = vh[:rank].T
    mapped = np.zeros((width, rank), dtype=_LD)
    mapped[active] = column_scale[active, None] * vectors
    coordinate_map = _finite_double(mapped, "support coordinate map")
    full_q, full_t = scipy.linalg.qr(coordinate_map, mode="full", check_finite=False)
    triangular = full_t[:rank]
    if np.any(np.diag(triangular) == 0):
        raise PenaltyNumericalError("singular support coordinate map")
    plus, zero = full_q[:, :rank], full_q[:, rank:]
    coordinates = tuple(
        _finite_double(item[:, active] / column_scale[active] @ vectors, "component coordinates")
        for item in balanced
    )
    selected, projection_bounds, selected_errors = [], [], []
    common_limited = rank < min(factor.shape)
    for root, error in zip(roots, root_errors, strict=True):
        if rank == width:
            projected = root.copy()
            selected_error = error
        else:
            projected = _finite_double(
                (root.astype(_LD) @ plus.astype(_LD)) @ plus.T.astype(_LD), "support projection"
            )
            # Both nonnegative products include their own rounding; a final
            # nextafter alone cannot enclose a rounded positive matrix dot.
            unit = np.finfo(_LD).eps / 2
            tiny = np.nextafter(_LD(0), _LD(1))
            intermediate = error.astype(_LD) @ np.abs(plus.astype(_LD))
            allowance = (2 * width + 4) * unit
            intermediate = (intermediate + (2 * width + 1) * tiny) / (1 - allowance)
            projected_error = intermediate @ np.abs(plus.T.astype(_LD))
            allowance = (2 * rank + 4) * unit
            selected_error = _finite_double(
                (projected_error + (2 * rank + 1) * tiny) / (1 - allowance), "projected root error"
            )
        projection_bounds.append(np.abs(projected.astype(_LD) - root.astype(_LD)))
        selected.append(_readonly(projected))
        selected_errors.append(_readonly(np.nextafter(selected_error, np.inf)))
    residual = factor - (factor @ vectors) @ vectors.T
    if np.linalg.norm(residual, ord="fro") > (
        np.sqrt(min(factor.shape)) * cutoff
        + 8 * max(factor.shape) * _EPS * np.linalg.norm(factor, ord="fro")
    ):
        raise PenaltyNumericalError("inconsistent balanced penalty support")
    return _PenaltySupport(
        tuple(selected),
        tuple(_readonly(value) for value in coordinates),
        _readonly(np.log(np.asarray(scales, dtype=_LD))),
        _readonly(coordinate_map),
        _readonly(triangular),
        _readonly(plus),
        _readonly(zero),
        tuple(bool(flag) or common_limited for flag in resolution_limited),
        tuple(_readonly(np.zeros((width, width))) for _ in roots),
        tuple(_readonly(_finite_double(bound, "projection bound")) for bound in projection_bounds),
        tuple(selected_errors),
    )
