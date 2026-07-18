"""Stable centered weighted products for grouped design matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tabmat  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ._group_matrix_kernels import (
    _disc_disc_2d_hist,
    _fused_bincount_2,
    _pattern_support_summaries,
)
from ._group_matrix_tabmat import _tabmat_vector

_MAX_PACKED_HIST_CELLS = 5_000_000
_MAX_PATTERN_SUMMARY_CELLS = 5_000_000
_MIN_MIXED_RAW_MOMENT_CELLS = 100_000
_MIN_LOW_CARDINALITY_MIXED_ROWS = 5_000


@dataclass(frozen=True)
class _CenteredSupport:
    values: NDArray
    codes: NDArray
    mean: NDArray
    mass: NDArray
    weighted_z: NDArray


@dataclass(frozen=True)
class _PatternPlan:
    unique_codes: NDArray
    row_patterns: NDArray
    sizes: NDArray
    marginal_offsets: NDArray
    pair_left: NDArray
    pair_right: NDArray
    pair_offsets: NDArray
    pair_right_sizes: NDArray
    widths: NDArray
    starts: NDArray
    tensor_group: int
    tensor_grid_row: NDArray
    tensor_grid_col: NDArray
    own_margins: tuple[tuple[int, int], ...]


class _TensorGridCache:
    __slots__ = ("w_grid",)

    def __init__(self, w_grid: NDArray) -> None:
        self.w_grid = w_grid

    def tensor_w_grid(self, _group, _W: NDArray) -> NDArray:
        return self.w_grid


def _readonly(values: NDArray, *, dtype=None) -> NDArray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _certify_raw_centering(
    *,
    raw_gram: NDArray,
    xtw: NDArray,
    raw_rhs: NDArray,
    weighted_z: NDArray,
    sum_w: float,
    sum_weighted_z: float | None = None,
) -> tuple[NDArray, NDArray, NDArray] | None:
    # Raw moments can overflow even when the anchor-centered fallback remains
    # finite (for example, a large finite location plus modest variation).
    # Keep that implementation detail independent of the caller's errstate and
    # reject non-finite intermediates before they reach rank calculations.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_x = xtw / sum_w
        centered_gram = raw_gram - np.outer(xtw, mean_x)
        centered_gram = 0.5 * (centered_gram + centered_gram.T)
        centered_diagonal = np.diag(centered_gram)
        if (
            not np.all(np.isfinite(mean_x))
            or not np.all(np.isfinite(centered_gram))
            or not np.all(np.isfinite(centered_diagonal))
            or np.any(centered_diagonal < 0.0)
        ):
            return None
        centered_scale = np.sqrt(centered_diagonal / sum_w)

    if not _raw_centering_well_scaled(mean_x, centered_scale):
        return None

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if sum_weighted_z is None:
            sum_weighted_z = float(np.sum(weighted_z, dtype=np.float64))
        centered_rhs = raw_rhs - mean_x * sum_weighted_z
    if not np.isfinite(sum_weighted_z) or not np.all(np.isfinite(centered_rhs)):
        return None
    return mean_x, centered_gram, centered_rhs


def _raw_centering_well_scaled(mean_x: NDArray, centered_scale: NDArray) -> bool:
    """Return whether raw-moment subtraction stays in its rounding envelope."""
    mean_x = np.asarray(mean_x, dtype=np.float64)
    centered_scale = np.asarray(centered_scale, dtype=np.float64)
    if not np.all(np.isfinite(mean_x)) or not np.all(np.isfinite(centered_scale)):
        return False
    # Keep intercept profiling within the ordinary rounding envelope of a
    # Gram calculation.  Allowing a larger mean than centered RMS amplifies
    # raw-moment subtraction error beyond that envelope and can erase a
    # near-collinear direction that the shared normal-equation rank policy
    # would otherwise retain.
    return bool(
        np.all((np.abs(mean_x) <= centered_scale) | ((mean_x == 0.0) & (centered_scale == 0.0)))
    )


def _try_tabmat_centering(
    *,
    tabmat_split,
    W: NDArray,
    z_centered: NDArray,
    sum_w: float,
    preflight: bool,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Use native categorical Tabmat kernels when raw centering is safe."""
    if tabmat_split is None or not any(
        isinstance(component, tabmat.CategoricalMatrix) for component in tabmat_split.matrices
    ):
        return None
    if any(
        np.dtype(component.dtype) != np.dtype(np.float64) for component in tabmat_split.matrices
    ):
        return None

    # Tabmat 4.2.1's compiled weighted kernels require a writable contiguous
    # weight buffer. In particular, strided weights can otherwise compute an
    # incorrect result without raising, while read-only weights are rejected.
    tabmat_weights = _tabmat_vector(W)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if preflight:
            # MatrixBase.standardize expects probability weights; it does not
            # normalize arbitrary working weights itself.  We use only its
            # cheap location/scale summary, never its raw centered sandwich.
            normalized_weights = tabmat_weights / sum_w
            _standardized, mean_x, centered_scale = tabmat_split.standardize(
                normalized_weights,
                center_predictors=True,
                scale_predictors=True,
            )
            if centered_scale is None or not _raw_centering_well_scaled(mean_x, centered_scale):
                return None
            xtw = np.asarray(mean_x, dtype=np.float64) * sum_w
        else:
            xtw = np.asarray(tabmat_split.transpose_matvec(tabmat_weights), dtype=np.float64)

        weighted_z = _tabmat_vector(tabmat_weights * z_centered)
        raw_gram = np.asarray(tabmat_split.sandwich(tabmat_weights), dtype=np.float64)
        raw_rhs = np.asarray(tabmat_split.transpose_matvec(weighted_z), dtype=np.float64)
    return _certify_raw_centering(
        raw_gram=raw_gram,
        xtw=xtw,
        raw_rhs=raw_rhs,
        weighted_z=weighted_z,
        sum_w=sum_w,
    )


def _try_factored_tensor_centering(
    *,
    dm,
    W: NDArray,
    weighted_z: NDArray,
    sum_w: float,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Retain factored tensor products when raw-moment centering is certified.

    The specialized block assembler is substantially cheaper for tensor
    products because it contracts the two marginal bases independently.  Raw
    moment subtraction is used only when every solver column has enough
    centered scale relative to its mean to keep cancellation below the shared
    square-root-epsilon boundary.  Ill-scaled inputs fall through to the
    anchor-centered support implementation below.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        sum_weighted_z = float(np.sum(weighted_z, dtype=np.float64))
    if not np.isfinite(sum_weighted_z):
        return None

    moments = dm.execution_plan._moments_prevalidated(
        W,
        rhs=(weighted_z,),
        include_xtw=True,
    )
    if moments.xtw is None:  # pragma: no cover - guaranteed by include_xtw
        raise RuntimeError("execution plan did not return X'W")
    return _certify_raw_centering(
        raw_gram=moments.gram,
        xtw=moments.xtw,
        raw_rhs=moments.xt_rhs[0],
        weighted_z=weighted_z,
        sum_w=sum_w,
        sum_weighted_z=sum_weighted_z,
    )


def _mixed_raw_centering_preflight(
    *,
    plan,
    W: NDArray,
    sum_w: float,
) -> NDArray | None:
    """Return augmented X'W when first-call raw centering is safe."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        augmented_mean, augmented_scale = plan.augmented_location_scale(_tabmat_vector(W) / sum_w)
    if (
        augmented_scale is None
        or not np.all(np.isfinite(augmented_mean))
        or not np.all(np.isfinite(augmented_scale))
    ):
        return None
    ordinary = plan.ordinary_augmented_indices
    ordinary_mean = augmented_mean[ordinary]
    ordinary_scale = augmented_scale[ordinary]
    if not _raw_centering_well_scaled(
        ordinary_mean,
        ordinary_scale,
    ):
        return None
    for block in plan.compressed_blocks:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mass = augmented_mean[block.augmented_indices] * sum_w
            xtw = block.support.T @ mass
            raw_diagonal = np.einsum(
                "ij,i,ij->j",
                block.support,
                mass,
                block.support,
                optimize=False,
            )
            mean = xtw / sum_w
            centered_diagonal = raw_diagonal - xtw * mean
            if (
                not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(centered_diagonal))
                or np.any(centered_diagonal < 0.0)
            ):
                return None
            scale = np.sqrt(centered_diagonal / sum_w)
        if not _raw_centering_well_scaled(mean, scale):
            return None
    augmented_xtw = augmented_mean * sum_w
    if not np.all(np.isfinite(augmented_xtw)):
        return None
    return augmented_xtw


def _try_mixed_discrete_centering(
    *,
    dm,
    W: NDArray,
    z_centered: NDArray,
    sum_w: float,
    preflight: bool = True,
) -> tuple[bool, tuple[NDArray, NDArray, NDArray] | None]:
    """Use the cached augmented bin-space plan for a certified mixed design."""
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )

    allowed_types = {DenseGroupMatrix, CategoricalGroupMatrix, DiscretizedSSPGroupMatrix}
    compressed_groups = tuple(
        group for group in dm.group_matrices if type(group) is DiscretizedSSPGroupMatrix
    )
    categorical_groups = tuple(
        group
        for group in dm.group_matrices
        if type(group) is CategoricalGroupMatrix and group.shape[1] > 0
    )
    has_ordinary = any(
        type(group) in {DenseGroupMatrix, CategoricalGroupMatrix} and group.shape[1] > 0
        for group in dm.group_matrices
    )
    if (
        not compressed_groups
        or not has_ordinary
        or len(categorical_groups) > 1
        or any(type(group) not in allowed_types for group in dm.group_matrices)
        or dm.p == 0
        or dm.n * dm.p < _MIN_MIXED_RAW_MOMENT_CELLS * len(compressed_groups)
        # Below this measured row crossover, constructing a native low-cardinality
        # block costs more than the stable dense-categorical fallback. High-cardinality
        # blocks retain their strong win even on smaller designs.
        or (
            categorical_groups
            and categorical_groups[0].n_levels <= 100
            and dm.n < _MIN_LOW_CARDINALITY_MIXED_ROWS
        )
    ):
        return False, None

    plan = dm.mixed_bin_space_centering_plan
    if plan is None:
        return False, None
    augmented_xtw = None
    if preflight:
        augmented_xtw = _mixed_raw_centering_preflight(
            plan=plan,
            W=W,
            sum_w=sum_w,
        )
        if augmented_xtw is None:
            return True, None

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        weighted_z = W * z_centered
        sum_weighted_z = float(np.sum(weighted_z, dtype=np.float64))
    if not np.isfinite(sum_weighted_z) or not np.all(np.isfinite(weighted_z)):
        return True, None

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        moments = plan.moments(W, weighted_z, augmented_xtw=augmented_xtw)
    if moments.xtw is None:  # pragma: no cover - guaranteed by include_xtw
        raise RuntimeError("execution plan did not return X'W")
    return (
        True,
        _certify_raw_centering(
            raw_gram=moments.gram,
            xtw=moments.xtw,
            raw_rhs=moments.xt_rhs[0],
            weighted_z=weighted_z,
            sum_w=sum_w,
            sum_weighted_z=sum_weighted_z,
        ),
    )


def _build_pattern_plan(dm) -> _PatternPlan | None:
    """Compress repeated combinations of discrete support codes once per fit."""
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
    )

    groups = dm.group_matrices
    tensor_groups = [
        index for index, group in enumerate(groups) if type(group) is DiscretizedTensorGroupMatrix
    ]
    if len(tensor_groups) != 1:
        return None
    tensor_group = tensor_groups[0]
    tensor = groups[tensor_group]

    group_codes: list[NDArray] = []
    sizes: list[int] = []
    for group in groups:
        if isinstance(group, CategoricalGroupMatrix):
            group_codes.append(group.codes)
            sizes.append(group.n_levels + 1)
        elif type(group) in (DiscretizedSSPGroupMatrix, DiscretizedTensorGroupMatrix):
            group_codes.append(group.bin_idx)
            sizes.append(group.n_bins)
        else:
            return None

    own_margins: list[tuple[int, int]] = []
    for index, group in enumerate(groups):
        if index == tensor_group or type(group) is not DiscretizedSSPGroupMatrix:
            continue
        same_first = group.n_bins == tensor.n_bins1 and np.array_equal(group.bin_idx, tensor.idx1)
        same_second = group.n_bins == tensor.n_bins2 and np.array_equal(group.bin_idx, tensor.idx2)
        if same_first != same_second:
            own_margins.append((index, 1 if same_first else 2))

    own_groups = {index for index, _margin in own_margins}
    pairs: list[tuple[int, int]] = []
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if tensor_group in (left, right):
                other = right if left == tensor_group else left
                if other in own_groups:
                    continue
            pairs.append((left, right))

    sizes_array = np.asarray(sizes, dtype=np.intp)
    pair_left = np.asarray([left for left, _right in pairs], dtype=np.intp)
    pair_right = np.asarray([right for _left, right in pairs], dtype=np.intp)
    pair_right_sizes = sizes_array[pair_right]
    pair_cells = sizes_array[pair_left] * pair_right_sizes
    if pair_cells.size and (
        np.any(pair_cells > _MAX_PACKED_HIST_CELLS)
        or int(np.sum(pair_cells, dtype=np.int64)) > _MAX_PATTERN_SUMMARY_CELLS
    ):
        return None

    mixed_key = np.zeros(dm.n, dtype=np.uint64)
    radix_product = 1
    max_uint64 = int(np.iinfo(np.uint64).max)
    for codes, size in zip(group_codes, sizes, strict=True):
        if size <= 0 or radix_product > max_uint64 // size:
            return None
        np.multiply(mixed_key, np.uint64(size), out=mixed_key)
        np.add(mixed_key, codes, out=mixed_key, casting="unsafe")
        radix_product *= size
    unique_keys, row_patterns = np.unique(mixed_key, return_inverse=True)
    if unique_keys.size > np.iinfo(np.int32).max:
        return None
    row_patterns = np.asarray(row_patterns, dtype=np.int32)
    remaining = unique_keys.copy()
    unique_codes = np.empty((len(unique_keys), len(sizes)), dtype=np.int32)
    for group in range(len(sizes) - 1, -1, -1):
        size = np.uint64(sizes[group])
        unique_codes[:, group] = (remaining % size).astype(np.int32)
        remaining //= size

    first_observation = np.full(tensor.n_bins, tensor.shape[0], dtype=np.intp)
    np.minimum.at(
        first_observation,
        tensor.bin_idx,
        np.arange(tensor.shape[0], dtype=np.intp),
    )
    if np.any(first_observation == tensor.shape[0]):
        return None
    tensor_grid_row = tensor.idx1[first_observation]
    tensor_grid_col = tensor.idx2[first_observation]
    if not np.array_equal(tensor_grid_row[tensor.bin_idx], tensor.idx1) or not np.array_equal(
        tensor_grid_col[tensor.bin_idx], tensor.idx2
    ):
        return None

    widths = np.asarray([group.shape[1] for group in groups], dtype=np.intp)
    marginal_offsets = np.concatenate(
        [np.zeros(1, dtype=np.intp), np.cumsum(sizes_array, dtype=np.intp)]
    )
    pair_offsets = np.concatenate(
        [np.zeros(1, dtype=np.intp), np.cumsum(pair_cells, dtype=np.intp)]
    )
    starts = np.concatenate([np.zeros(1, dtype=np.intp), np.cumsum(widths, dtype=np.intp)])
    return _PatternPlan(
        unique_codes=_readonly(np.ascontiguousarray(unique_codes), dtype=np.int32),
        row_patterns=_readonly(np.ascontiguousarray(row_patterns), dtype=np.int32),
        sizes=_readonly(sizes_array, dtype=np.intp),
        marginal_offsets=_readonly(marginal_offsets, dtype=np.intp),
        pair_left=_readonly(pair_left, dtype=np.intp),
        pair_right=_readonly(pair_right, dtype=np.intp),
        pair_offsets=_readonly(pair_offsets, dtype=np.intp),
        pair_right_sizes=_readonly(pair_right_sizes, dtype=np.intp),
        widths=_readonly(widths, dtype=np.intp),
        starts=_readonly(starts, dtype=np.intp),
        tensor_group=tensor_group,
        tensor_grid_row=_readonly(tensor_grid_row, dtype=np.intp),
        tensor_grid_col=_readonly(tensor_grid_col, dtype=np.intp),
        own_margins=tuple(own_margins),
    )


def _pattern_plan(dm) -> _PatternPlan | None:
    cached = dm._centered_pattern_plan
    if cached is False:
        return None
    if cached is None:
        cached = _build_pattern_plan(dm)
        dm._centered_pattern_plan = cached if cached is not None else False
    return cached


def _solver_supports(dm) -> tuple[NDArray, ...]:
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
    )

    cached = dm._centered_solver_supports
    if cached is not None:
        return cached
    supports: list[NDArray] = []
    for group in dm.group_matrices:
        if type(group) is DiscretizedTensorGroupMatrix:
            supports.append(group.B_unique)
        elif type(group) is DiscretizedSSPGroupMatrix:
            supports.append(np.ascontiguousarray(group.B_unique @ group.R_inv))
        elif isinstance(group, CategoricalGroupMatrix):
            values = np.zeros((group.n_levels + 1, group.n_levels), dtype=np.float64)
            values[np.arange(group.n_levels), np.arange(group.n_levels)] = 1.0
            supports.append(values)
        else:  # pragma: no cover - guarded by pattern-plan construction
            raise TypeError(type(group).__name__)
    cached = tuple(supports)
    dm._centered_solver_supports = cached
    return cached


def _try_pattern_tensor_centering(
    *,
    dm,
    W: NDArray,
    z_centered: NDArray,
    weighted_z: NDArray,
    sum_w: float,
) -> tuple[bool, tuple[NDArray, NDArray, NDArray] | None]:
    """Assemble all discrete summaries through compressed joint code patterns."""
    from ._group_matrix_algebra import _cross_gram_tensor_own_margin

    plan = _pattern_plan(dm)
    if plan is None:
        return False, None
    marginal_w, marginal_wz, joint_w = _pattern_support_summaries(
        plan.row_patterns,
        plan.unique_codes,
        W,
        weighted_z,
        plan.marginal_offsets,
        plan.pair_left,
        plan.pair_right,
        plan.pair_offsets,
        plan.pair_right_sizes,
    )
    supports = _solver_supports(dm)
    groups = dm.group_matrices
    tensor_group = plan.tensor_group
    tensor = groups[tensor_group]
    raw_gram = np.zeros((dm.p, dm.p), dtype=np.float64)
    xtw = np.zeros(dm.p, dtype=np.float64)
    raw_rhs = np.zeros(dm.p, dtype=np.float64)

    for group_index, support in enumerate(supports):
        if group_index == tensor_group:
            continue
        support_slice = slice(
            plan.marginal_offsets[group_index], plan.marginal_offsets[group_index + 1]
        )
        coefficient_slice = slice(plan.starts[group_index], plan.starts[group_index + 1])
        mass = marginal_w[support_slice]
        weighted_response = marginal_wz[support_slice]
        raw_gram[coefficient_slice, coefficient_slice] = support.T @ (mass[:, None] * support)
        xtw[coefficient_slice] = support.T @ mass
        raw_rhs[coefficient_slice] = support.T @ weighted_response

    tensor_support_slice = slice(
        plan.marginal_offsets[tensor_group], plan.marginal_offsets[tensor_group + 1]
    )
    tensor_mass = marginal_w[tensor_support_slice]
    tensor_weighted_response = marginal_wz[tensor_support_slice]
    w_grid = np.zeros((tensor.n_bins1, tensor.n_bins2), dtype=np.float64)
    wz_grid = np.zeros_like(w_grid)
    w_grid[plan.tensor_grid_row, plan.tensor_grid_col] = tensor_mass
    wz_grid[plan.tensor_grid_row, plan.tensor_grid_col] = tensor_weighted_response
    tensor_gram, tensor_xtw, tensor_rhs = tensor.gram_rmatvec_from_grids(w_grid, wz_grid)
    tensor_slice = slice(plan.starts[tensor_group], plan.starts[tensor_group + 1])
    raw_gram[tensor_slice, tensor_slice] = tensor_gram
    xtw[tensor_slice] = tensor_xtw
    raw_rhs[tensor_slice] = tensor_rhs

    for pair in range(plan.pair_left.size):
        left = int(plan.pair_left[pair])
        right = int(plan.pair_right[pair])
        histogram = joint_w[plan.pair_offsets[pair] : plan.pair_offsets[pair + 1]].reshape(
            plan.sizes[left], plan.sizes[right]
        )
        if left == tensor_group:
            cross = tensor.R_inv.T @ (tensor.B_unique.T @ histogram @ supports[right])
        elif right == tensor_group:
            cross = (supports[left].T @ histogram @ tensor.B_unique) @ tensor.R_inv
        else:
            cross = supports[left].T @ histogram @ supports[right]
        left_slice = slice(plan.starts[left], plan.starts[left + 1])
        right_slice = slice(plan.starts[right], plan.starts[right + 1])
        raw_gram[left_slice, right_slice] = cross
        raw_gram[right_slice, left_slice] = cross.T

    grid_cache = _TensorGridCache(w_grid)
    for own_group, _margin in plan.own_margins:
        cross = _cross_gram_tensor_own_margin(
            tensor,
            groups[own_group],
            W,
            grid_cache,
        )
        if cross is None:  # pragma: no cover - plan construction certified this match
            return True, None
        own_slice = slice(plan.starts[own_group], plan.starts[own_group + 1])
        raw_gram[own_slice, tensor_slice] = cross
        raw_gram[tensor_slice, own_slice] = cross.T

    return (
        True,
        _certify_raw_centering(
            raw_gram=raw_gram,
            xtw=xtw,
            raw_rhs=raw_rhs,
            weighted_z=weighted_z,
            sum_w=sum_w,
        ),
    )


def _anchor_center_support(
    *,
    values: NDArray,
    codes: NDArray,
    W: NDArray,
    Wz: NDArray,
    sum_w: float,
    transform: NDArray | None = None,
) -> _CenteredSupport:
    """Center compact support rows before any weighted cross-products."""
    mass, weighted_z = _fused_bincount_2(codes, W, Wz, len(values))
    anchor = int(np.argmax(mass))
    differences = values - values[anchor]
    mean_difference = mass @ differences / sum_w
    centered = differences - mean_difference
    mean = values[anchor] + mean_difference
    if transform is not None:
        centered = centered @ transform
        mean = mean @ transform
    return _CenteredSupport(
        values=centered,
        codes=codes,
        mean=mean,
        mass=mass,
        weighted_z=weighted_z,
    )


def packed_centered_gram_rhs(
    *,
    dm,
    W: NDArray,
    z_centered: NDArray,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Build centered products from indexed supports when every group is eligible."""
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
    )

    supports: list[_CenteredSupport] = []
    widths: list[int] = []
    eligible_types = (DiscretizedSSPGroupMatrix, DiscretizedTensorGroupMatrix)
    if any(
        type(gm) not in eligible_types and not isinstance(gm, CategoricalGroupMatrix)
        for gm in dm.group_matrices
    ):
        return None
    weighted_z = W * z_centered
    sum_w = float(np.sum(W, dtype=np.float64))
    if any(type(gm) is DiscretizedTensorGroupMatrix for gm in dm.group_matrices):
        pattern_attempted, patterned = _try_pattern_tensor_centering(
            dm=dm,
            W=W,
            z_centered=z_centered,
            weighted_z=weighted_z,
            sum_w=sum_w,
        )
        if patterned is not None:
            return patterned
        if not pattern_attempted:
            factored = _try_factored_tensor_centering(
                dm=dm,
                W=W,
                weighted_z=weighted_z,
                sum_w=sum_w,
            )
            if factored is not None:
                return factored

    for gm in dm.group_matrices:
        if type(gm) in eligible_types:
            supports.append(
                _anchor_center_support(
                    values=gm.B_unique,
                    codes=gm.bin_idx,
                    W=W,
                    Wz=weighted_z,
                    sum_w=sum_w,
                    transform=gm.R_inv,
                )
            )
        elif isinstance(gm, CategoricalGroupMatrix):
            values = np.zeros((gm.n_levels + 1, gm.n_levels), dtype=float)
            values[np.arange(gm.n_levels), np.arange(gm.n_levels)] = 1.0
            supports.append(
                _anchor_center_support(
                    values=values,
                    codes=gm.codes,
                    W=W,
                    Wz=weighted_z,
                    sum_w=sum_w,
                )
            )
        widths.append(gm.shape[1])

    if any(
        len(left.values) * len(right.values) > _MAX_PACKED_HIST_CELLS
        for i, left in enumerate(supports)
        for right in supports[i + 1 :]
    ):
        return None

    p = dm.p
    gram = np.zeros((p, p), dtype=float)
    rhs = np.zeros(p, dtype=float)
    mean_x = (
        np.concatenate([support.mean for support in supports])
        if supports
        else np.zeros(0, dtype=float)
    )
    starts = np.cumsum([0, *widths])

    for i, support_i in enumerate(supports):
        sl_i = slice(starts[i], starts[i + 1])
        gram[sl_i, sl_i] = support_i.values.T @ (support_i.mass[:, None] * support_i.values)
        rhs[sl_i] = support_i.values.T @ support_i.weighted_z

        for j in range(i + 1, len(supports)):
            support_j = supports[j]
            sl_j = slice(starts[j], starts[j + 1])
            n_j = len(support_j.values)
            joint_mass = _disc_disc_2d_hist(
                support_i.codes,
                support_j.codes,
                W,
                len(support_i.values),
                n_j,
            )
            cross = support_i.values.T @ joint_mass @ support_j.values
            gram[sl_i, sl_j] = cross
            gram[sl_j, sl_i] = cross.T

    return mean_x, gram, rhs


def _compensated_add(total: NDArray, compensation: NDArray, value: NDArray) -> None:
    corrected = value - compensation
    updated = total + corrected
    compensation[...] = (updated - total) - corrected
    total[...] = updated


def centered_gram_rhs(
    *,
    dm,
    W: NDArray,
    mean_x: NDArray,
    z_centered: NDArray,
    chunk_size: int = 8192,
) -> tuple[NDArray, NDArray]:
    """Return centered ``X'WX`` and ``X'Wz`` without raw-moment subtraction.

    Rows are materialized only in bounded chunks. Centering happens before
    multiplication, so large feature offsets cannot cancel two raw moments.
    Group-specific ``row_subset`` implementations preserve sparse/discretized
    storage and avoid materializing the full training design.
    """
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    mean_x = np.asarray(mean_x, dtype=float)
    z_centered = np.asarray(z_centered, dtype=float)
    if W.shape != (n,) or z_centered.shape != (n,):
        raise ValueError("W and z_centered must match the design row count")
    if mean_x.shape != (p,):
        raise ValueError("mean_x must match the design column count")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if p == 0:
        return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)

    gram = np.zeros((p, p), dtype=float)
    gram_compensation = np.zeros_like(gram)
    rhs = np.zeros(p, dtype=float)
    rhs_compensation = np.zeros_like(rhs)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=float)
        block -= mean_x
        W_block = W[start:stop]
        gram_block = block.T @ (W_block[:, None] * block)
        rhs_block = block.T @ (W_block * z_centered[start:stop])
        _compensated_add(gram, gram_compensation, gram_block)
        _compensated_add(rhs, rhs_compensation, rhs_block)

    gram = 0.5 * (gram + gram.T)
    return gram, rhs


def centered_rhs(
    *,
    dm,
    W: NDArray,
    mean_x: NDArray,
    z_centered: NDArray,
    chunk_size: int = 8192,
) -> NDArray:
    """Return ``(X - mean_x)' W z_centered`` without rebuilding the Gram."""
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    mean_x = np.asarray(mean_x, dtype=float)
    z_centered = np.asarray(z_centered, dtype=float)
    if W.shape != (n,) or z_centered.shape != (n,):
        raise ValueError("W and z_centered must match the design row count")
    if mean_x.shape != (p,):
        raise ValueError("mean_x must match the design column count")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if p == 0:
        return np.zeros(0, dtype=float)

    rhs = np.zeros(p, dtype=float)
    compensation = np.zeros_like(rhs)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=float)
        block -= mean_x
        rhs_block = block.T @ (W[start:stop] * z_centered[start:stop])
        _compensated_add(rhs, compensation, rhs_block)
    return rhs
