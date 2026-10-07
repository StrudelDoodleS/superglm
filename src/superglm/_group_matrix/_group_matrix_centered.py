"""Stable centered weighted products for grouped design matrices."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import tabmat  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ._group_matrix_algebra import _BlockWeightCache
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


@dataclass
class RawMomentRejection:
    """The raw moments a rung's certificate rejected during one centred build.

    Owner: one ``_raw_rung_system`` call, that is one Gram build at one weight
    vector; nothing outlives it, so no weight, basis or penalty can change
    under it.  ``source`` names the rung that formed the moments: ``"pattern"``
    and ``"factored"`` (``packed_centered_gram_rhs``) or ``"raw_moment"``
    (``try_raw_moment_centering``).  The factored rung and the raw-moment
    rung form the same moments by the same execution-plan call, so a
    factored rejection decides the raw-moment rung of the same build.
    ``supports`` maps each discretized SSP group those two rungs' build
    projected to that projection, ``B_unique @ R_inv``
    (``_BlockWeightCache.supports``), which the repair reads instead of
    projecting again; ``None`` from the pattern rung.
    """

    source: str | None = None
    raw_gram: NDArray | None = None
    xtw: NDArray | None = None
    raw_rhs: NDArray | None = None
    weighted_z: NDArray | None = None
    sum_weighted_z: float | None = None
    supports: dict | None = None

    def record(
        self, source, *, raw_gram, xtw, raw_rhs, weighted_z, sum_weighted_z=None, supports=None
    ) -> None:
        self.source = source
        self.raw_gram, self.xtw, self.raw_rhs = raw_gram, xtw, raw_rhs
        self.weighted_z, self.sum_weighted_z = weighted_z, sum_weighted_z
        self.supports = supports


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


def _certify_or_record(
    rejected: RawMomentRejection | None, source: str, supports=None, **moments
) -> tuple[NDArray, NDArray, NDArray] | None:
    """``_certify_raw_centering(**moments)``; a rejection hands the moments to ``rejected``."""
    certified = _certify_raw_centering(**moments)
    if certified is None and rejected is not None:
        moments.pop("sum_w")
        rejected.record(source, supports=supports, **moments)
    return certified


def _raw_centering_well_scaled(mean_x: NDArray, centered_scale: NDArray) -> bool:
    """Return whether raw-moment subtraction stays in its rounding envelope."""
    return bool(np.all(_raw_centering_admitted(mean_x, centered_scale)))


def _raw_centering_admitted(mean_x: NDArray, centered_scale: NDArray) -> NDArray:
    """Per column, whether raw-moment subtraction stays in its rounding envelope.

    A column is admitted when its mean and centred RMS are finite and
    ``|mean| <= RMS``, which is ``kappa^2 = 1 + mean^2 / RMS^2 <= 2`` (Chan,
    Golub & LeVeque 1983, eq. 3.3).
    """
    mean_x = np.asarray(mean_x, dtype=np.float64)
    centered_scale = np.asarray(centered_scale, dtype=np.float64)
    finite = np.isfinite(mean_x) & np.isfinite(centered_scale)
    # Keep intercept profiling within the ordinary rounding envelope of a
    # Gram calculation.  Allowing a larger mean than centered RMS amplifies
    # raw-moment subtraction error beyond that envelope and can erase a
    # near-collinear direction that the shared normal-equation rank policy
    # would otherwise retain.
    with np.errstate(invalid="ignore"):
        return finite & (
            (np.abs(mean_x) <= centered_scale) | ((mean_x == 0.0) & (centered_scale == 0.0))
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


def _try_raw_spline_tabmat_centering(
    *,
    plan,
    W: NDArray,
    z_centered: NDArray,
    sum_w: float,
    preflight: bool,
    profile: dict | None = None,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Use one raw-basis Tabmat sandwich and transform to solver coordinates."""
    started = perf_counter()
    if profile is not None:
        profile["centered_spline_tabmat_attempts"] = (
            profile.get("centered_spline_tabmat_attempts", 0) + 1
        )
    # Raw SSP bases have compact support, while arbitrary solver transforms can
    # still create unsafe locations.  The authoritative transformed-moment
    # certificate below covers both cases, so a separate Tabmat standardize
    # pass would only duplicate the weighted sparse traversal.
    _ = preflight
    tabmat_weights = _tabmat_vector(W)
    result = None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_xtw = np.asarray(
            plan.split.transpose_matvec(tabmat_weights),
            dtype=np.float64,
        )
        weighted_z = _tabmat_vector(tabmat_weights * z_centered)
        raw_gram = np.asarray(plan.split.sandwich(tabmat_weights), dtype=np.float64)
        raw_rhs = np.asarray(plan.split.transpose_matvec(weighted_z), dtype=np.float64)
        xtw = plan.transform_vector(raw_xtw)
        gram = plan.transform_gram(raw_gram)
        rhs = plan.transform_vector(raw_rhs)
        result = _certify_raw_centering(
            raw_gram=gram,
            xtw=xtw,
            raw_rhs=rhs,
            weighted_z=weighted_z,
            sum_w=sum_w,
        )
    if profile is not None:
        outcome = "accepts" if result is not None else "rejections"
        key = f"centered_spline_tabmat_{outcome}"
        profile[key] = profile.get(key, 0) + 1
        profile["centered_spline_tabmat_s"] = (
            profile.get("centered_spline_tabmat_s", 0.0) + perf_counter() - started
        )
        profile["centered_spline_tabmat_retained_bytes"] = max(
            profile.get("centered_spline_tabmat_retained_bytes", 0),
            plan.retained_bytes,
        )
    return result


def try_raw_moment_centering(
    *,
    dm,
    W: NDArray,
    weighted_z: NDArray,
    sum_w: float,
    rejected: RawMomentRejection | None = None,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Centre from raw per-block moments, for any design the plan can dispatch.

    The specialised rungs above each require a particular set of group-matrix
    types; a design outside those sets falls to a chunked pass that rebuilds the
    ``(n, p)`` design block by block.  The execution plan already computes the
    same raw moments per block pair far more cheaply, so this rung feeds those
    through the shared scaling certificate instead.

    Generalises :func:`_try_factored_tensor_centering`, which performs the same
    subtraction but only for designs containing a factored tensor product.
    Returns ``None`` whenever the certificate rejects, leaving the caller on its
    stable chunked path; ``rejected``, when given, then receives the moments.
    """
    # Same measured crossover the mixed rung uses: below this many design cells
    # the raw-moment accumulation costs more than the stable chunked pass.
    if dm.n * dm.p < _MIN_MIXED_RAW_MOMENT_CELLS:
        return None

    with np.errstate(over="ignore", invalid="ignore"):
        sum_weighted_z = float(np.sum(weighted_z, dtype=np.float64))
    if not np.isfinite(sum_weighted_z):
        return None

    # Raw moments can overflow on ill-scaled designs.  The certificate below
    # rejects non-finite intermediates, but the accumulation itself must not
    # raise under a caller that has promoted floating-point warnings to errors.
    cache = _BlockWeightCache()
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            moments = dm.execution_plan._moments_prevalidated(
                W,
                rhs=(weighted_z,),
                include_xtw=True,
                _cache=cache,
            )
    except FloatingPointError:
        return None
    if moments.xtw is None:  # pragma: no cover - guaranteed by include_xtw
        return None
    return _certify_or_record(
        rejected,
        "raw_moment",
        supports=cache.supports,
        raw_gram=moments.gram,
        xtw=moments.xtw,
        raw_rhs=moments.xt_rhs[0],
        weighted_z=weighted_z,
        sum_w=sum_w,
        sum_weighted_z=sum_weighted_z,
    )


def _try_factored_tensor_centering(
    *,
    dm,
    W: NDArray,
    weighted_z: NDArray,
    sum_w: float,
    rejected: RawMomentRejection | None = None,
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

    cache = _BlockWeightCache()
    moments = dm.execution_plan._moments_prevalidated(
        W,
        rhs=(weighted_z,),
        include_xtw=True,
        _cache=cache,
    )
    if moments.xtw is None:  # pragma: no cover - guaranteed by include_xtw
        raise RuntimeError("execution plan did not return X'W")
    return _certify_or_record(
        rejected,
        "factored",
        supports=cache.supports,
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
    rejected: RawMomentRejection | None = None,
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
        _certify_or_record(
            rejected,
            "pattern",
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
    """Center compact support rows before any weighted cross-products.

    The support and transform are read as float64 before any arithmetic, as
    every other reader of the column converts them; an integer difference
    would wrap. The mean projects the anchor and the shift apart: rounding
    their sum first can erase a shift below the anchor's spacing, which a
    projection that cancels the anchor then exposes.
    """
    values = np.asarray(values, dtype=np.float64)
    mass, weighted_z = _fused_bincount_2(codes, W, Wz, len(values))
    anchor = int(np.argmax(mass))
    differences = values - values[anchor]
    mean_difference = mass @ differences / sum_w
    centered = differences - mean_difference
    if transform is None:
        mean = values[anchor] + mean_difference
    else:
        transform = np.asarray(transform, dtype=np.float64)
        centered = centered @ transform
        mean = values[anchor] @ transform + mean_difference @ transform
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
    state=None,
    rejected: RawMomentRejection | None = None,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Build centered products from indexed supports when every group is eligible.

    ``state`` (a fit-local ``TabmatCenteringState``), when given, carries a
    rejection of the tensor raw rungs across the fit's iterations.
    ``rejected``, when given, receives the raw moments a tensor rung's
    certificate rejected (``column_local_centering`` repairs them).
    """
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
    )

    # isinstance, not `type(gm) in`: SupportCompressedSSPGroupMatrix is a
    # DiscretizedSSPGroupMatrix that adds no state (`__slots__ = ()`) and is
    # numerically identical over the same basis -- its bin_idx indexes exact
    # distinct rows rather than bins, so the packed build is *more* justified
    # for it than for its binned parent.  An exact-type test rejected it, and
    # one such group sent the whole design to the chunked dense fallback.
    # SCOP and SplineCategorical are separate hierarchies, so this widening
    # admits nothing that lacks B_unique/bin_idx/R_inv.
    eligible_types = (DiscretizedSSPGroupMatrix, DiscretizedTensorGroupMatrix)
    if any(
        not isinstance(gm, eligible_types) and not isinstance(gm, CategoricalGroupMatrix)
        for gm in dm.group_matrices
    ):
        return None
    weighted_z = W * z_centered
    sum_w = float(np.sum(W, dtype=np.float64))
    tensor_rejected = False
    if any(type(gm) is DiscretizedTensorGroupMatrix for gm in dm.group_matrices) and (
        state is None or state.tensor_raw_eligible is not False
    ):
        pattern_attempted, patterned = _try_pattern_tensor_centering(
            dm=dm,
            W=W,
            z_centered=z_centered,
            weighted_z=weighted_z,
            sum_w=sum_w,
            rejected=rejected,
        )
        if patterned is not None:
            return patterned
        if not pattern_attempted:
            factored = _try_factored_tensor_centering(
                dm=dm,
                W=W,
                weighted_z=weighted_z,
                sum_w=sum_w,
                rejected=rejected,
            )
            if factored is not None:
                return factored
        tensor_rejected = True

    anchored = _anchor_support_gram_rhs(dm=dm, W=W, weighted_z=weighted_z, sum_w=sum_w)
    if tensor_rejected and anchored is not None and state is not None:
        # A rejection latches for the fit, as the other raw rungs' do: later
        # weights could pass, but the anchor-centred route is at least as
        # accurate, and repeating rejected raw moments doubled each
        # iteration's centring cost.  It latches only when that route served
        # the build.  The route declines on support sizes alone (a basis
        # property), and the caller then falls to the chunked dense pass,
        # which costs far more than retrying the tensor rungs, so a design it
        # declines keeps retrying them: a later iterate's weights may pass
        # (ten tensor pairs on the 678k-row book: 5 chunked builds latched
        # against 2 retried).
        state.tensor_raw_eligible = False
    return anchored


def _compact_support_rows(gm) -> int | None:
    """Row count of ``_compact_support(gm)``'s support from metadata alone, else ``None``.

    The size preflight: a categorical's support is a dense ``(K+1, K)``
    identity, so its size must be refused before anything is allocated.
    """
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )

    if isinstance(gm, DiscretizedSSPGroupMatrix):
        return int(gm.B_unique.shape[0])
    if isinstance(gm, DiscretizedSCOPGroupMatrix):
        return int(gm.B_scop_unique.shape[0])
    if isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
        return int(gm.n_bins) + 1
    if isinstance(gm, CategoricalGroupMatrix):
        return int(gm.n_levels) + 1
    return None


def _compact_supports_fit(group_matrices, cells) -> bool:
    """Whether every group has a compact support of at most ``_MAX_PACKED_HIST_CELLS`` cells.

    The size preflight, read from metadata alone (``_compact_support_rows``)
    before any support is materialised: a categorical's support is a dense
    ``(K+1, K)`` identity, so a large K must be refused before it is
    allocated. ``cells(rows, width)`` is what a caller then allocates from a
    support of ``rows`` rows for a group of ``width`` columns.
    """
    for gm in group_matrices:
        rows = _compact_support_rows(gm)
        if rows is None or cells(rows, gm.shape[1]) > _MAX_PACKED_HIST_CELLS:
            return False
    return True


def _compact_support(gm) -> tuple[NDArray, NDArray, NDArray | None] | None:
    """``(values, codes, transform)`` with rows ``values[codes] @ transform``, else ``None``.

    Discretized SSP groups (tensor and support-compressed ones included) and
    discretized SCOP groups index a dense support by bin; a categorical's
    support is the identity with a zero row for its base level.  One level of
    a discretized spline-by-category term is its spline support on that
    level's rows and zero elsewhere: the support gains a zero row, which every
    row outside the level is coded to.
    """
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )

    if isinstance(gm, DiscretizedSSPGroupMatrix):
        return gm.B_unique, gm.bin_idx, gm.R_inv
    if isinstance(gm, DiscretizedSCOPGroupMatrix):
        return gm.B_scop_unique, gm.bin_idx, None
    if isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
        values = np.zeros((gm.n_bins + 1, gm.B_unique.shape[1]), dtype=np.float64)
        values[: gm.n_bins] = gm.B_unique
        codes = np.full(gm.n_rows, gm.n_bins, dtype=np.intp)
        codes[gm.row_idx] = gm.bin_idx_level
        return values, codes, gm.R_inv
    if isinstance(gm, CategoricalGroupMatrix):
        values = np.zeros((gm.n_levels + 1, gm.n_levels), dtype=np.float64)
        values[np.arange(gm.n_levels), np.arange(gm.n_levels)] = 1.0
        return values, gm.codes, None
    return None


def anchor_support_centered_gram_rhs(
    *,
    dm,
    W: NDArray,
    z_centered: NDArray,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Centre-first products from compact supports, discretized SCOP groups included.

    The stable fallback for a design every raw-moment rung rejected: each
    support row is centred once about its weighted mean, before any product,
    so no raw moment is subtracted, and no row of the design is materialised.
    ``None`` when a group has no compact support or one is oversized.
    """
    weighted_z = W * z_centered
    sum_w = float(np.sum(W, dtype=np.float64))
    return _anchor_support_gram_rhs(dm=dm, W=W, weighted_z=weighted_z, sum_w=sum_w)


def _anchor_support_gram_rhs(
    *,
    dm,
    W: NDArray,
    weighted_z: NDArray,
    sum_w: float,
) -> tuple[NDArray, NDArray, NDArray] | None:
    """Anchor-centred compact supports, their Grams and cross-Grams by joint histogram.

    One row pass (``_disc_disc_2d_hist``) per pair of groups, except within a
    tensor's derived family (``_derived_support_family``): a group whose codes
    are a function of the tensor's cell codes, as its own margins' are, takes
    every cross-Gram from the tensor's tables (``_add_family_cross``), which
    saves its passes against every other group, the tensor and each other.

    **Error.**  Every cross entry is ``sum_r W_r s(r)_j t(r)_k`` over rows,
    ``W >= 0``, of two computed centred supports ``s``, ``t``.  The per-pair
    route sums rows into a joint table and then over the two supports' rows;
    the derived route sums rows into the tensor's table and then over its
    cells and the other support's rows.  Each term meets two multiplications
    and at most ``n + n_t + n_c`` additions on either route (``n`` rows,
    ``n_t`` the other support's rows, ``n_c`` the larger of the child's rows
    and the tensor's cells), and a sum of products in any order and
    association lies within ``gamma_K sum_r W_r |s(r)_j| |t(r)_k|`` of its
    exact value with ``K`` that count plus two (Higham 2002, secs. 3.1 and
    4.2).  The routes therefore differ by at most twice that.  The supports,
    means, diagonal blocks, right-hand side and every block outside the family
    are bitwise those of the per-pair route.
    """
    # Reject an oversized support BEFORE materialising it.  A categorical
    # block's anchor support is a dense ``(K+1, K)`` identity and its Gram
    # costs O(K^3); for a crossed interaction K is ``(L1-1)*(L2-1)``, so it
    # grows multiplicatively in the parents' cardinalities.  Falling back here
    # costs the chunked path, which is what this design got before.  The same
    # bound covers every pair's joint table, ``n_i n_j <= max(n_i, n_j)^2``
    # (a centred support keeps its compact support's rows), so no decline
    # waits for a row pass.
    if not _compact_supports_fit(dm.group_matrices, lambda rows, width: rows * rows):
        return None
    compact = [_compact_support(gm) for gm in dm.group_matrices]

    supports: list[_CenteredSupport] = []
    for gm, (values, codes, transform) in zip(dm.group_matrices, compact, strict=True):
        # Supports are read as float64, as every other reader of the column
        # converts them (exact for float32 and for integers up to 2**53, which
        # round alike in all of them). Complex, object and wider dtypes decline.
        if any(
            operand is not None and (operand.dtype.kind not in "biuf" or operand.dtype.itemsize > 8)
            for operand in (values, transform)
        ):
            return None
        supports.append(
            _anchor_center_support(
                values=values,
                codes=codes,
                W=W,
                Wz=weighted_z,
                sum_w=sum_w,
                transform=transform,
            )
        )
    widths = [gm.shape[1] for gm in dm.group_matrices]

    p = dm.p
    gram = np.zeros((p, p), dtype=float)
    rhs = np.zeros(p, dtype=float)
    mean_x = (
        np.concatenate([support.mean for support in supports])
        if supports
        else np.zeros(0, dtype=float)
    )
    starts = np.cumsum([0, *widths])

    family = _derived_support_family(dm, compact)
    derived = set() if family is None else {family[0], *family[1]}
    for i, support_i in enumerate(supports):
        sl_i = slice(starts[i], starts[i + 1])
        gram[sl_i, sl_i] = support_i.values.T @ (support_i.mass[:, None] * support_i.values)
        rhs[sl_i] = support_i.values.T @ support_i.weighted_z

        for j in range(i + 1, len(supports)):
            if i in derived or j in derived:
                continue
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

    if family is not None:
        _add_family_cross(gram, supports, starts, W, *family)
    return mean_x, gram, rhs


def _derived_support_family(dm, compact) -> tuple[int, dict[int, NDArray]] | None:
    """``(tensor, {group: tensor cell -> group code})`` for the groups that are functions of a tensor's cells.

    A tensor's own margins (its marginal splines on the same bins) are the
    case that arises: every row of a tensor cell lies in one margin bin.  It
    depends on the basis only, so it is decided once per design and kept on
    it, keyed by the identity of the design's group matrices (owner: the
    design; invalidated with its groups).  A cell no row reaches maps to code
    0, harmlessly: its tensor table rows and mass are exactly zero.
    """
    from superglm.group_matrix import DiscretizedTensorGroupMatrix

    groups = dm.group_matrices
    cached = getattr(dm, "_centered_support_family", None)
    if cached is not None and cached[0] is groups:
        return cached[1]
    family = None
    for parent, matrix in enumerate(groups):
        if type(matrix) is not DiscretizedTensorGroupMatrix:
            continue
        cell_values, cells, _transform = compact[parent]
        children = {}
        for child, (_values, codes, _child_transform) in enumerate(compact):
            if child == parent or type(groups[child]) is DiscretizedTensorGroupMatrix:
                continue
            cell_map = np.zeros(cell_values.shape[0], dtype=np.intp)
            cell_map[cells] = codes
            if np.array_equal(cell_map[cells], codes):
                children[child] = cell_map
        if children:
            family = (parent, children)
            break
    dm._centered_support_family = (groups, family)
    return family


def _add_family_cross(
    gram: NDArray,
    supports: list[_CenteredSupport],
    starts: NDArray,
    W: NDArray,
    parent: int,
    children: dict[int, NDArray],
) -> None:
    """Every cross-Gram touching a tensor's derived family, one row pass per other group.

    A child's codes are ``m(c)`` for tensor cell ``c``, so its joint table
    with any group ``x`` is the tensor's summed over cells, ``H_ax = M' H_tx``
    (``M`` the cell-to-code selection), and ``S_a' H_ax S_x = E' H_tx S_x``
    with ``E = S_a[m]`` the child's centred support at each cell.  Against
    the tensor and another child the table is the tensor's cell mass on the
    map's pattern: ``S_a' H_at S_t = (E * mass_t)' S_t`` and ``(E_a *
    mass_t)' E_b``.  A tensor block against another group is evaluated in the
    per-pair route's order, so it is bitwise that route's.
    """

    def put(left: int, right: int, block: NDArray) -> None:
        rows = slice(starts[left], starts[left + 1])
        columns = slice(starts[right], starts[right + 1])
        gram[rows, columns] = block
        gram[columns, rows] = block.T

    tensor = supports[parent]
    expanded = {child: supports[child].values[cell_map] for child, cell_map in children.items()}
    for child, values in expanded.items():
        weighted = values * tensor.mass[:, None]
        put(child, parent, weighted.T @ tensor.values)
        for other, other_values in expanded.items():
            if other > child:
                put(child, other, weighted.T @ other_values)
    for index, support in enumerate(supports):
        if index == parent or index in children:
            continue
        if index < parent:
            joint = _disc_disc_2d_hist(
                support.codes, tensor.codes, W, len(support.values), len(tensor.values)
            )
            projected = support.values.T @ joint
            put(index, parent, projected @ tensor.values)
            for child, values in expanded.items():
                put(index, child, projected @ values)
        else:
            joint = _disc_disc_2d_hist(
                tensor.codes, support.codes, W, len(tensor.values), len(support.values)
            )
            put(parent, index, tensor.values.T @ joint @ support.values)
            projected = joint @ support.values
            for child, values in expanded.items():
                put(child, index, values.T @ projected)


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
    mean_lo: NDArray | None = None,
    first: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return centered ``X'WX`` and ``X'Wz`` without raw-moment subtraction.

    Rows are materialized only in bounded chunks. Centering happens before
    multiplication, so large feature offsets cannot cancel two raw moments.
    Group-specific ``row_subset`` implementations preserve sparse/discretized
    storage and avoid materializing the full training design.
    ``mean_lo``, when given, is the low half of the centre as an exact pair
    ``(mean_x, mean_lo)`` (``centered_system.weighted_mean_pair``): rows are
    centred as ``(x - mean_x) - mean_lo``, so a dense column at an offset is
    centred about its anchor exactly and then by the small remainder.
    ``first``, when given (``(p,)`` zeros), receives ``sum W (x - mean_x)``
    from the same rows, compensated across chunks: the moment Björck's
    correction of the corrected two-pass algorithm reads
    (``centered_system.two_pass_centred_gram``).
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
    first_compensation = np.zeros(p, dtype=float)
    rhs = np.zeros(p, dtype=float)
    rhs_compensation = np.zeros_like(rhs)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=float)
        block -= mean_x
        if mean_lo is not None:
            block -= mean_lo
        W_block = W[start:stop]
        gram_block = block.T @ (W_block[:, None] * block)
        rhs_block = block.T @ (W_block * z_centered[start:stop])
        _compensated_add(gram, gram_compensation, gram_block)
        _compensated_add(rhs, rhs_compensation, rhs_block)
        if first is not None:
            _compensated_add(first, first_compensation, block.T @ W_block)

    gram = 0.5 * (gram + gram.T)
    return gram, rhs


def centered_signed_grams(
    *,
    dm,
    weights: Sequence[NDArray],
    mean_x: NDArray,
    chunk_size: int = 8192,
    mean_lo: NDArray | None = None,
) -> list[NDArray]:
    """Reuse centered row chunks across signed Gram products in input order.

    ``mean_lo``, when given, is the low half of the centre as an exact pair
    ``(mean_x, mean_lo)`` (``centered_system.weighted_mean_pair``): rows are
    centred as ``(x - mean_x) - mean_lo``, so a dense column at an offset is
    centred about its anchor exactly and then by the small remainder.
    """
    n, p = dm.shape
    weights = [np.asarray(channel, dtype=float) for channel in weights]
    mean_x = np.asarray(mean_x, dtype=float)
    if any(channel.shape != (n,) for channel in weights):
        raise ValueError("weights must match the design row count")
    if mean_x.shape != (p,):
        raise ValueError("mean_x must match the design column count")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    grams = [np.zeros((p, p), dtype=float) for _ in weights]
    if not weights or p == 0:
        return grams
    compensations = [np.zeros_like(gram) for gram in grams]
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = np.asarray(dm.row_subset(np.arange(start, stop)).toarray(), dtype=float)
        block -= mean_x
        if mean_lo is not None:
            block -= mean_lo
        for weights_j, gram, compensation in zip(weights, grams, compensations, strict=True):
            contribution = block.T @ (weights_j[start:stop, None] * block)
            _compensated_add(gram, compensation, contribution)
    return [0.5 * (gram + gram.T) for gram in grams]


def _compact_centered_rmatvec(
    *,
    dm,
    rows: NDArray,
    mean_x: NDArray,
    mean_lo: NDArray | None,
    chunk_size: int = 8192,
) -> NDArray | None:
    """``(X - 1 c')' rows`` with each support row centred before its product; else ``None``.

    For a compact group the rows are ``s_b`` (support row ``b``) on the rows
    coded ``b``, so the product is ``sum_b (s_b - c) R_b`` with ``R_b`` the
    sum of ``rows`` coded ``b``: every term is the centred row the chunked
    pass forms, with the same rounding, and only the order of accumulation
    differs (``np.bincount``, as ``rmatvec`` sums by bin).  A dense group is
    centred row by row over its own columns.  ``None`` for any other group,
    or a support of more than ``_MAX_PACKED_HIST_CELLS`` entries, refused from
    metadata before any support is materialised (``_compact_supports_fit``).
    """
    from superglm.group_matrix import DenseGroupMatrix

    compact_groups = [gm for gm in dm.group_matrices if type(gm) is not DenseGroupMatrix]
    if not _compact_supports_fit(compact_groups, lambda rows, width: rows * width):
        return None
    result = np.empty(dm.p, dtype=np.float64)
    offset = 0
    for gm in dm.group_matrices:
        width = gm.shape[1]
        centre = mean_x[offset : offset + width]
        if mean_lo is not None:
            centre_lo = mean_lo[offset : offset + width]
        if type(gm) is DenseGroupMatrix:
            accumulated = np.zeros(width, dtype=np.float64)
            compensation = np.zeros(width, dtype=np.float64)
            for start in range(0, dm.n, chunk_size):
                stop = min(start + chunk_size, dm.n)
                block = np.asarray(gm.M[start:stop], dtype=np.float64) - centre
                if mean_lo is not None:
                    block -= centre_lo
                _compensated_add(accumulated, compensation, block.T @ rows[start:stop])
            result[offset : offset + width] = accumulated
            offset += width
            continue
        values, codes, transform = _compact_support(gm)
        centred = (values if transform is None else values @ transform) - centre
        if mean_lo is not None:
            centred -= centre_lo
        aggregated = np.bincount(codes, weights=rows, minlength=values.shape[0])
        result[offset : offset + width] = centred.T @ aggregated
        offset += width
    return result


def centered_rhs(
    *,
    dm,
    W: NDArray,
    mean_x: NDArray,
    z_centered: NDArray,
    chunk_size: int = 8192,
    mean_lo: NDArray | None = None,
) -> NDArray:
    """Return ``(X - mean_x)' W z_centered`` without rebuilding the Gram.

    ``mean_lo``, when given, is the low half of the centre as an exact pair
    ``(mean_x, mean_lo)`` (``centered_system.weighted_mean_pair``): rows are
    centred as ``(x - mean_x) - mean_lo``, so a dense column at an offset is
    centred about its anchor exactly and then by the small remainder.

    A design whose groups all have compact supports (or are dense) centres
    each support row once and aggregates ``W z_centered`` by support row
    (``_compact_centered_rmatvec``); any other design is centred in row chunks.
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
        return np.zeros(0, dtype=float)
    compact = _compact_centered_rmatvec(dm=dm, rows=W * z_centered, mean_x=mean_x, mean_lo=mean_lo)
    if compact is not None:
        return compact

    rhs = np.zeros(p, dtype=float)
    compensation = np.zeros_like(rhs)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = np.arange(start, stop)
        block = np.asarray(dm.row_subset(rows).toarray(), dtype=float)
        block -= mean_x
        if mean_lo is not None:
            block -= mean_lo
        rhs_block = block.T @ (W[start:stop] * z_centered[start:stop])
        _compensated_add(rhs, compensation, rhs_block)
    return rhs
