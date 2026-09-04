"""Private algebra helpers for group-matrix block operations."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_kernels import (
    _cat_cat_weighted_crosstab,
    _cat_weighted_bincount,
    _csr_weighted_bincount,
    _disc_disc_2d_hist,
    _disc_disc_2d_hist_channels,
    _fused_2d_bincount_2,
    _weighted_bincount_2d,
)

if TYPE_CHECKING:
    from ..group_matrix import (
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        GroupMatrix,
        RandomEffectGroupMatrix,
    )
else:
    GroupMatrix = Any

_MAX_DISC_DISC_HIST_CELLS = 5_000_000
_MAX_DISC_DISC_CHANNEL_HIST_CELLS = 5_000_000

# Transient ceiling for the row-expanded cross-gram fallback, matching the byte
# budgets used elsewhere in this package.  The histogram cap above bounds CELLS;
# this bounds the ROWS the fallback expands, which the cell cap cannot.
_MAX_CROSS_EXPANSION_BYTES = 64 << 20

# Ceiling on a CROSS-SHAPED aggregate: an array whose row count comes from one
# block and whose column count comes from the other.  Every gate in this
# subsystem bounds one block against its OWN width -- the support gate bounds
# n_support * p_b, the histogram caps bound n_bins_i * n_bins_j -- and none of
# them bounds a row count from one block against a width from the other.  That
# product is what the aggregate allocates, so it needs its own ceiling.
_MAX_AGGREGATE_CELLS = _MAX_CROSS_EXPANSION_BYTES // 8


def _aggregate_column_chunk(n_bins: int, n_cols: int) -> int:
    """Output columns per pass of a cross-shaped aggregate, sized in CELLS."""
    return max(1, min(int(n_cols), int(_MAX_AGGREGATE_CELLS // max(int(n_bins), 1))))


def _profile_add(profile: dict[str, Any] | None, key: str, value: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + value


def _profile_count(profile: dict[str, Any] | None, key: str, value: int = 1) -> None:
    if profile is not None:
        profile[key] = int(profile.get(key, 0)) + value


def _profile_elapsed(profile: dict[str, Any] | None, key: str, start: float) -> None:
    if profile is not None:
        _profile_add(profile, key, perf_counter() - start)


class _BlockWeightCache:
    """Per-block-assembly cache for weighted discrete summaries."""

    __slots__ = ("_hist2d", "_profile")

    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        self._hist2d: dict[tuple[int, int, int, int, int], NDArray] = {}
        self._profile = profile

    @staticmethod
    def _key(idx_a: NDArray, idx_b: NDArray, W: NDArray, n_a: int, n_b: int):
        return (id(idx_a), id(idx_b), id(W), int(n_a), int(n_b))

    def disc_disc_hist(
        self,
        idx_a: NDArray,
        idx_b: NDArray,
        W: NDArray,
        n_a: int,
        n_b: int,
    ) -> NDArray:
        key = self._key(idx_a, idx_b, W, n_a, n_b)
        cached = self._hist2d.get(key)
        if cached is not None:
            _profile_count(self._profile, "block_hist2d_reuses")
            return cached

        rev_key = self._key(idx_b, idx_a, W, n_b, n_a)
        rev_cached = self._hist2d.get(rev_key)
        if rev_cached is not None:
            hist = rev_cached.T
            self._hist2d[key] = hist
            _profile_count(self._profile, "block_hist2d_reuses")
            return hist

        t0 = perf_counter() if self._profile is not None else 0.0
        hist = _disc_disc_2d_hist(idx_a, idx_b, W, n_a, n_b)
        _profile_elapsed(self._profile, "block_hist2d_s", t0)
        _profile_count(self._profile, "block_hist2d_builds")
        self._hist2d[key] = hist
        return hist

    def tensor_w_grid(self, gm: DiscretizedTensorGroupMatrix, W: NDArray) -> NDArray:
        return self.disc_disc_hist(gm.idx1, gm.idx2, W, gm.n_bins1, gm.n_bins2)

    def tensor_w_wz_grid(
        self, gm: DiscretizedTensorGroupMatrix, W: NDArray, Wz: NDArray
    ) -> tuple[NDArray, NDArray]:
        w_key = self._key(gm.idx1, gm.idx2, W, gm.n_bins1, gm.n_bins2)
        wz_key = self._key(gm.idx1, gm.idx2, Wz, gm.n_bins1, gm.n_bins2)
        w_grid = self._hist2d.get(w_key)
        wz_grid = self._hist2d.get(wz_key)
        if w_grid is None and wz_grid is None:
            t0 = perf_counter() if self._profile is not None else 0.0
            w_grid, wz_grid = _fused_2d_bincount_2(gm.idx1, gm.idx2, W, Wz, gm.n_bins1, gm.n_bins2)
            _profile_elapsed(self._profile, "block_hist2d_s", t0)
            _profile_count(self._profile, "block_hist2d_builds")
            self._hist2d[w_key] = w_grid
            self._hist2d[wz_key] = wz_grid
            return w_grid, wz_grid

        if w_grid is None:
            w_grid = self.tensor_w_grid(gm, W)
        if wz_grid is None:
            wz_grid = self.disc_disc_hist(gm.idx1, gm.idx2, Wz, gm.n_bins1, gm.n_bins2)
        return w_grid, wz_grid


def _runtime_group_matrix_types():
    """Import group-matrix runtime classes lazily to avoid circular imports."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )

    return (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )


def _agg_by_bin_fits(gm: GroupMatrix, n_bins: int) -> bool:
    """Whether ``_agg_by_bin``'s output is small enough to materialise.

    Its result is ``(n_bins, gm.shape[1])`` -- the row count from the block
    supplying the bins, the width from the block being aggregated.  Those are
    DIFFERENT blocks at every caller, so this is the cross shape, and no gate in
    the subsystem bounds the product.  An earlier version of the invariant test
    exempted these calls on the stated ground that both dimensions came from the
    same block; that was simply wrong, and a narrow million-row support beside a
    wide sparse term is the counterexample.
    """
    return int(n_bins) * _agg_by_bin_width(gm) <= _MAX_AGGREGATE_CELLS


def _agg_by_bin_width(gm: GroupMatrix) -> int:
    """The width ``_agg_by_bin`` actually ALLOCATES at, not the width it returns.

    The SSP branches aggregate in basis space and only then apply ``R_inv``, so
    the intermediate is ``_p_b`` wide while ``shape[1]`` is the post-transform
    width -- 600 against 4 on the pairing that exposed this.  Budgeting against
    the returned width silently permits the allocation it is meant to stop.
    """
    for attribute in ("_p_b",):
        width = getattr(gm, attribute, None)
        if width is not None:
            return int(width)
    matrix = getattr(gm, "M", None)
    if matrix is not None:
        return int(matrix.shape[1])
    unique = getattr(gm, "B_unique", None)
    if unique is not None:
        return max(int(unique.shape[1]), int(gm.shape[1]))
    return int(gm.shape[1])


def _agg_by_bin(gm: GroupMatrix, bin_idx: NDArray, W: NDArray, n_bins: int) -> NDArray:
    """Aggregate W * gm's columns by bin index → (n_bins, p_g) dense array.

    Dispatches to the most efficient kernel for each GroupMatrix type:
    - SparseGroupMatrix: CSR-aware kernel (avoids toarray, O(nnz) not O(n*p))
    - SparseSSPGroupMatrix: CSR kernel in B-spline space + R_inv transform
    - DenseGroupMatrix / other: fused dense kernel (avoids W-broadcast alloc)
    """
    (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
    from ..group_matrix import FactorSmoothGroupMatrix

    if isinstance(gm, CategoricalGroupMatrix):
        return _cat_weighted_bincount(gm.codes, bin_idx, W, n_bins, gm.n_levels)
    if isinstance(gm, SparseGroupMatrix):
        return _csr_weighted_bincount(
            np.asarray(gm.M.data, dtype=np.float64),
            gm.M.indices,
            gm.M.indptr,
            gm.M.shape[1],
            bin_idx,
            W,
            n_bins,
        )
    if isinstance(gm, SparseSSPGroupMatrix):
        B_agg = _csr_weighted_bincount(
            gm._data, gm._indices, gm._indptr, gm._p_b, bin_idx, W, n_bins
        )
        return B_agg @ gm.R_inv
    if isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
        rows = gm.row_idx
        # Chunked, and it is the only branch here that needed saying so: the
        # SCOP, SSP and factor-smooth branches all decline into
        # ``_aggregate_group_matrix_columns``, which is column-at-a-time and
        # bounded, while this one expanded the level per observation row.
        # Reached when ``_cross_gram_discrete_spline_categorical`` declines at
        # its cell cap and dispatch falls through to the disc-x-non-disc branch,
        # which is newly reachable because a lossless support makes ``n_bins``
        # large on both sides.  ``B_unique @ R_inv`` is ``(n_support, p_g)``,
        # bounded by the support gate; the gather off it was not.
        return _chunked_support_bincount_2d(
            bin_idx[rows], W[rows], gm.B_unique @ gm.R_inv, gm.bin_idx_level, n_bins
        )
    if isinstance(gm, SplineCategoricalGroupMatrix):
        rows = gm.row_idx
        B_agg = _csr_weighted_bincount(
            gm._data,
            gm._indices,
            gm._indptr,
            gm._p_b,
            bin_idx[rows],
            W[rows],
            n_bins,
        )
        return B_agg @ gm.R_inv
    if isinstance(gm, DiscretizedSCOPGroupMatrix):
        n_cells = n_bins * gm.n_bins
        if n_cells <= _MAX_DISC_DISC_HIST_CELLS:
            weight_grid = _disc_disc_2d_hist(
                bin_idx,
                gm.bin_idx,
                W,
                n_bins,
                gm.n_bins,
            )
            return weight_grid @ gm.B_scop_unique
        return _aggregate_group_matrix_columns(gm, bin_idx, W, n_bins)
    if isinstance(gm, DiscretizedSSPGroupMatrix):
        n_cells = n_bins * gm.n_bins
        if n_cells <= _MAX_DISC_DISC_HIST_CELLS:
            weight_grid = _disc_disc_2d_hist(
                bin_idx,
                gm.bin_idx,
                W,
                n_bins,
                gm.n_bins,
            )
            return (weight_grid @ gm.B_unique) @ gm.R_inv
        return _aggregate_group_matrix_columns(gm, bin_idx, W, n_bins)
    if isinstance(gm, FactorSmoothGroupMatrix):
        return _aggregate_group_matrix_columns(gm, bin_idx, W, n_bins)
    X = gm.toarray()
    return _weighted_bincount_2d(bin_idx, W, X, n_bins)


def _aggregate_group_matrix_columns(
    gm: GroupMatrix,
    bin_idx: NDArray,
    W: NDArray,
    n_bins: int,
) -> NDArray:
    """Aggregate a compact matrix by bins without an observation-by-column temporary."""
    result = np.empty((n_bins, gm.shape[1]), dtype=np.float64)
    unit = np.zeros(gm.shape[1], dtype=np.float64)
    for column in range(gm.shape[1]):
        unit[column] = 1.0
        values = gm.matvec(unit)
        result[:, column] = np.bincount(
            bin_idx,
            weights=W * values,
            minlength=n_bins,
        )
        unit[column] = 0.0
    return result


def _random_effect_cross_gram(
    random_effect: RandomEffectGroupMatrix,
    other: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """Return ``X_re.T @ diag(W) @ X_other`` by direct level aggregation."""
    # Guarded like every other _agg_by_bin caller.  The output is
    # (n_levels, width-of-other) -- cross-shaped, and a high-cardinality random
    # effect beside a wide raw-basis SSP term is the case that reaches it.  An
    # earlier audit only inspected functions named _cross_gram, so this call sat
    # outside its scope entirely.
    if not _agg_by_bin_fits(other, random_effect.n_levels):
        return _cross_gram_by_columns(random_effect, other, W)
    return _agg_by_bin(
        other,
        random_effect.codes,
        W,
        random_effect.n_levels,
    )


def _cross_gram_tensor_tensor(
    gm_i: DiscretizedTensorGroupMatrix,
    gm_j: DiscretizedTensorGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray:
    """Cross-gram between two tensor groups sharing the same marginals.

    Used for decomposed tensor subgroups (bilinear × wiggly) that share
    the same B1_unique, B2_unique, idx1, idx2 but have different R_inv.
    """
    if cache is None:
        w_grid = _disc_disc_2d_hist(gm_i.idx1, gm_i.idx2, W, gm_i.n_bins1, gm_i.n_bins2)
    else:
        w_grid = cache.tensor_w_grid(gm_i, W)
    G_raw = gm_i._factored_gram_raw(w_grid)
    return gm_i.R_inv.T @ G_raw @ gm_j.R_inv


def _tensor_margin_parts(
    gm: DiscretizedTensorGroupMatrix,
    margin: int,
) -> tuple[NDArray, NDArray, int, NDArray, NDArray, int, bool]:
    if margin == 1:
        return (
            gm.B1_unique_t,
            gm.idx1,
            gm.n_bins1,
            gm.B2_unique_t,
            gm.idx2,
            gm.n_bins2,
            True,
        )
    return (
        gm.B2_unique_t,
        gm.idx2,
        gm.n_bins2,
        gm.B1_unique_t,
        gm.idx1,
        gm.n_bins1,
        False,
    )


def _same_discrete_margin(
    gm_i: DiscretizedTensorGroupMatrix,
    margin_i: int,
    gm_j: DiscretizedTensorGroupMatrix,
    margin_j: int,
) -> bool:
    _B_i, idx_i, n_i, *_ = _tensor_margin_parts(gm_i, margin_i)
    _B_j, idx_j, n_j, *_ = _tensor_margin_parts(gm_j, margin_j)
    return n_i == n_j and np.array_equal(idx_i, idx_j)


def _cross_gram_tensor_tensor_shared_margin(
    gm_i: DiscretizedTensorGroupMatrix,
    gm_j: DiscretizedTensorGroupMatrix,
    W: NDArray,
) -> NDArray | None:
    """Cross-Gram for two tensor terms that share exactly one marginal index."""
    matches = [
        (margin_i, margin_j)
        for margin_i in (1, 2)
        for margin_j in (1, 2)
        if _same_discrete_margin(gm_i, margin_i, gm_j, margin_j)
    ]
    if len(matches) != 1:
        return None

    margin_i, margin_j = matches[0]
    (
        B_shared_i,
        idx_shared,
        n_shared,
        B_other_i,
        idx_other_i,
        n_other_i,
        i_shared_first,
    ) = _tensor_margin_parts(gm_i, margin_i)
    (
        B_shared_j,
        _idx_shared_j,
        _n_shared_j,
        B_other_j,
        idx_other_j,
        n_other_j,
        j_shared_first,
    ) = _tensor_margin_parts(gm_j, margin_j)

    n_cells = n_shared * n_other_i * n_other_j
    if n_cells > _MAX_DISC_DISC_HIST_CELLS:
        return None

    flat = (idx_shared * n_other_i + idx_other_i) * n_other_j + idx_other_j
    joint = np.bincount(flat, weights=W, minlength=n_cells).reshape(
        n_shared,
        n_other_i,
        n_other_j,
    )

    K_shared_i = B_shared_i.shape[1]
    K_other_i = B_other_i.shape[1]
    K_shared_j = B_shared_j.shape[1]
    K_other_j = B_other_j.shape[1]
    raw4 = np.zeros(
        (K_shared_i, K_other_i, K_shared_j, K_other_j),
        dtype=np.float64,
    )
    for idx in range(n_shared):
        other_cross = B_other_i.T @ joint[idx] @ B_other_j
        raw4 += np.einsum(
            "p,q,rs->prqs",
            B_shared_i[idx],
            B_shared_j[idx],
            other_cross,
            optimize=True,
        )

    axes = (
        (0, 1) if i_shared_first else (1, 0),
        (2, 3) if j_shared_first else (3, 2),
    )
    raw = raw4.transpose(*axes[0], *axes[1]).reshape(
        gm_i.R_inv.shape[0],
        gm_j.R_inv.shape[0],
    )
    return gm_i.R_inv.T @ raw @ gm_j.R_inv


def _cross_gram_tensor_main(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_main: DiscretizedSSPGroupMatrix,
    W: NDArray,
) -> NDArray:
    """Blocked cross-gram between a tensor and a main-effect discretized group.

    Returns X_main.T @ diag(W) @ X_tensor in SSP space, shape (p_main, p_tensor).

    Chooses the cheaper channel orientation. If the second tensor margin is
    narrower, aggregate channels over B2 and contract with B1. If the first
    margin is narrower, aggregate channels over B1 and contract with B2.

    Column ordering: j1 * K2 + j2, matching _row_kron_dense().
    """
    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    B_main = gm_main.B_unique
    K1, K2 = B1.shape[1], B2.shape[1]
    K_main_raw = B_main.shape[1]

    n_cells_b2 = gm_main.n_bins * gm_tensor.n_bins1 * K2
    n_cells_b1 = gm_main.n_bins * gm_tensor.n_bins2 * K1
    channel_over_b2 = n_cells_b2 <= n_cells_b1
    n_cells = n_cells_b2 if channel_over_b2 else n_cells_b1
    if n_cells <= _MAX_DISC_DISC_CHANNEL_HIST_CELLS:
        if not channel_over_b2:
            H_flat = _disc_disc_2d_hist_channels(
                gm_main.bin_idx,
                gm_tensor.idx2,
                gm_tensor.idx1,
                W,
                B1,
                gm_main.n_bins,
                gm_tensor.n_bins2,
            )
            tmp = B_main.T @ H_flat.reshape(gm_main.n_bins, gm_tensor.n_bins2 * K1)
            tmp_3d = tmp.reshape(K_main_raw, gm_tensor.n_bins2, K1)

            result_raw = np.empty((K_main_raw, K1 * K2))
            for j1 in range(K1):
                result_raw[:, j1 * K2 : (j1 + 1) * K2] = tmp_3d[:, :, j1] @ B2
            return gm_main.R_inv.T @ result_raw @ gm_tensor.R_inv

        H_flat = _disc_disc_2d_hist_channels(
            gm_main.bin_idx,
            gm_tensor.idx1,
            gm_tensor.idx2,
            W,
            B2,
            gm_main.n_bins,
            gm_tensor.n_bins1,
        )
        tmp = B_main.T @ H_flat.reshape(gm_main.n_bins, gm_tensor.n_bins1 * K2)
        tmp_3d = tmp.reshape(K_main_raw, gm_tensor.n_bins1, K2)

        result_raw = np.empty((K_main_raw, K1 * K2))
        for j2 in range(K2):
            result_raw[:, j2::K2] = tmp_3d[:, :, j2] @ B1
        return gm_main.R_inv.T @ result_raw @ gm_tensor.R_inv

    result_raw = np.zeros((K_main_raw, K1 * K2))
    if not channel_over_b2:
        for j1 in range(K1):
            w_col = W * B1[gm_tensor.idx1, j1]
            H = _disc_disc_2d_hist(
                gm_main.bin_idx,
                gm_tensor.idx2,
                w_col,
                gm_main.n_bins,
                gm_tensor.n_bins2,
            )
            result_raw[:, j1 * K2 : (j1 + 1) * K2] = B_main.T @ H @ B2
        return gm_main.R_inv.T @ result_raw @ gm_tensor.R_inv

    for j2 in range(K2):
        # Weight observations by B2[idx2[obs], j2]
        w_col = W * B2[gm_tensor.idx2, j2]
        # 2D histogram: (n_bins_main, n_bins1)
        H = _disc_disc_2d_hist(
            gm_main.bin_idx,
            gm_tensor.idx1,
            w_col,
            gm_main.n_bins,
            gm_tensor.n_bins1,
        )
        # Contract: (K_main, n_bins_main) × (n_bins_main, n_bins1) × (n_bins1, K1)
        result_raw[:, j2::K2] = B_main.T @ H @ B1

    return gm_main.R_inv.T @ result_raw @ gm_tensor.R_inv


def _cross_gram_tensor_own_margin(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_main: DiscretizedSSPGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray | None:
    """Cross-Gram for tensor × one of its own discretized marginal smooths.

    mgcv bam(discrete=TRUE) stores tensor marginals as compact matrices plus
    row index arrays and forms X'WX from that packed representation via XWXd.
    For tensor × own-margin main-effect cross-blocks, reuse the tensor 2D
    W-grid instead of rescanning observations through the generic tensor-main
    channel histogram.

    Returns X_main.T @ diag(W) @ X_tensor in SSP space, or None if the main
    group is not exactly one unambiguous tensor margin.
    """
    cache_key = (id(gm_main), id(gm_main.bin_idx), gm_main.n_bins)
    margin = gm_tensor._own_margin_cache.get(cache_key, -1)
    if margin == -1:
        same_margin1 = gm_main.n_bins == gm_tensor.n_bins1 and np.array_equal(
            gm_main.bin_idx, gm_tensor.idx1
        )
        same_margin2 = gm_main.n_bins == gm_tensor.n_bins2 and np.array_equal(
            gm_main.bin_idx, gm_tensor.idx2
        )
        margin = None if same_margin1 == same_margin2 else (1 if same_margin1 else 2)
        gm_tensor._own_margin_cache[cache_key] = margin
    if margin is None:
        return None

    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    B_main = gm_main.B_unique
    K1, K2 = B1.shape[1], B2.shape[1]
    K_main_raw = B_main.shape[1]
    result_raw = np.empty((K_main_raw, K1 * K2), dtype=np.float64)
    if cache is None:
        w_grid = _disc_disc_2d_hist(
            gm_tensor.idx1,
            gm_tensor.idx2,
            W,
            gm_tensor.n_bins1,
            gm_tensor.n_bins2,
        )
    else:
        w_grid = cache.tensor_w_grid(gm_tensor, W)

    if margin == 1:
        weighted_margin2 = w_grid @ B2  # (n_bins1, K2)
        for j2 in range(K2):
            result_raw[:, j2::K2] = B_main.T @ (B1 * weighted_margin2[:, j2][:, None])
    else:
        weighted_margin1 = w_grid.T @ B1  # (n_bins2, K1)
        for j1 in range(K1):
            result_raw[:, j1 * K2 : (j1 + 1) * K2] = B_main.T @ (
                B2 * weighted_margin1[:, j1][:, None]
            )

    return gm_main.R_inv.T @ result_raw @ gm_tensor.R_inv


def _cross_gram_tensor_spline_categorical(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_spline_cat: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram X_tensor.T W X_spline_cat without materialising joint support."""
    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    K1, K2 = B1.shape[1], B2.shape[1]
    K_cat = gm_spline_cat._p_b
    result_raw = np.empty((K1 * K2, K_cat), dtype=np.float64)
    rows = gm_spline_cat.row_idx
    W_rows = W[rows]
    idx1 = gm_tensor.idx1[rows]
    idx2 = gm_tensor.idx2[rows]

    if hasattr(gm_spline_cat, "B_unique"):
        bin_cat = gm_spline_cat.bin_idx_level
        # A lossless spline_cat support is bounded by its row count, not by a
        # bin count, so the joint histogram is not automatically small; expand
        # the level's rows once and aggregate on them when it is not.
        if gm_tensor.n_bins1 * gm_spline_cat.n_bins <= _MAX_DISC_DISC_HIST_CELLS:
            for j2 in range(K2):
                w_col = W_rows * B2[idx2, j2]
                H = _disc_disc_2d_hist(
                    idx1,
                    bin_cat,
                    w_col,
                    gm_tensor.n_bins1,
                    gm_spline_cat.n_bins,
                )
                result_raw[j2::K2, :] = B1.T @ H @ gm_spline_cat.B_unique
            return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv

        # Over the cell cap.  Chunk the row expansion rather than materialising
        # the whole level: the cap bounds CELLS, and a lossless support leaves
        # the row count unbounded underneath it.  The chunk loop is outermost so
        # each row is still expanded exactly once across all K2 passes.
        n_level = int(W_rows.shape[0])
        chunk = min(
            max(n_level, 1),
            _cross_expansion_chunk_rows(K_cat, 0, _MAX_CROSS_EXPANSION_BYTES),
        )
        # Holding every channel at once lets each row be expanded exactly once,
        # but the accumulator is (K2, n_bins1, K_cat) -- itself cross-shaped, and
        # n_bins1 follows a per-feature n_bins that nothing caps.  So the fast
        # nesting is used only while that buffer fits, and otherwise the loops
        # invert: one channel at a time, one (n_bins1, K_cat) accumulator, at the
        # cost of re-expanding the rows per channel.  Bounded memory bought with
        # time, declared rather than assumed.
        if K2 * gm_tensor.n_bins1 * K_cat <= _MAX_AGGREGATE_CELLS:
            agg = np.zeros((K2, gm_tensor.n_bins1, K_cat), dtype=np.float64)
            for start in range(0, n_level, chunk):
                stop = min(start + chunk, n_level)
                block = _expand_support_rows(gm_spline_cat.B_unique, bin_cat[start:stop])
                idx1_chunk = idx1[start:stop]
                for j2 in range(K2):
                    w_col = W_rows[start:stop] * B2[idx2[start:stop], j2]
                    agg[j2] += _weighted_bincount_2d(idx1_chunk, w_col, block, gm_tensor.n_bins1)
                del block  # next expansion is evaluated before the rebind
            for j2 in range(K2):
                result_raw[j2::K2, :] = B1.T @ agg[j2]
            return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv

        # One channel at a time is still not enough: a single (n_bins1, K_cat)
        # channel is itself cross-shaped, and n_bins1 follows a configured
        # n_bins.  So the basis dimension is tiled inside the channel as well,
        # and only the columns of that tile are ever expanded.  What is left at
        # the smallest tile is (n_bins1, 1), which is the least an aggregate
        # over n_bins1 bins can occupy.
        tile = _aggregate_column_chunk(gm_tensor.n_bins1, K_cat)
        for j2 in range(K2):
            for first in range(0, K_cat, tile):
                last = min(first + tile, K_cat)
                columns = gm_spline_cat.B_unique[:, first:last]
                channel = np.zeros((gm_tensor.n_bins1, last - first), dtype=np.float64)
                for start in range(0, n_level, chunk):
                    stop = min(start + chunk, n_level)
                    block = _expand_support_rows(columns, bin_cat[start:stop])
                    w_col = W_rows[start:stop] * B2[idx2[start:stop], j2]
                    channel += _weighted_bincount_2d(
                        idx1[start:stop], w_col, block, gm_tensor.n_bins1
                    )
                    del block
                result_raw[j2::K2, first:last] = B1.T @ channel
        return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv

    for j2 in range(K2):
        w_col = W_rows * B2[idx2, j2]
        B_cat_agg = _csr_weighted_bincount(
            gm_spline_cat._data,
            gm_spline_cat._indices,
            gm_spline_cat._indptr,
            K_cat,
            idx1,
            w_col,
            gm_tensor.n_bins1,
        )
        result_raw[j2::K2, :] = B1.T @ B_cat_agg

    return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv


def _expand_support_rows(B_unique: NDArray, bin_idx: NDArray) -> NDArray:
    """Materialise a support block on its observation rows.

    Named so the chunking above it can be pinned by row count in a test rather
    than asserted about.
    """
    return B_unique[bin_idx]


def _cross_expansion_chunk_rows(p_i: int, p_j: int, max_bytes: int) -> int:
    """Rows per chunk of the expanded cross-gram, sized in BYTES.

    Two expanded blocks are live at once and the right one is scaled in place,
    so a chunk costs ``rows * (p_i + p_j) * 8``.  Pass ``p_j=0`` where only one
    block is expanded.
    """
    return max(1, int(max_bytes // max((p_i + p_j) * 8, 1)))


def _chunked_support_bincount_2d(
    bin_idx: NDArray,
    weights: NDArray,
    B_unique: NDArray,
    support_idx: NDArray,
    n_bins: int,
    max_bytes: int | None = None,
) -> NDArray:
    """``_weighted_bincount_2d`` over support-indexed rows, expanded in chunks.

    The aggregation is a sum over rows, so partitioning the rows partitions the
    sum.  Chunking matters here for the same reason it does in
    :func:`_support_support_raw_cross`: a lossless support bounds the number of
    DISTINCT rows, not the number of observation rows the level owns, so
    materialising the level in one go is unbounded in ``n``.

    Below the threshold the loop runs once and the result is bit-identical to
    the unchunked form, so ordinary fits are unaffected.
    """
    # Resolved at call time, not bound as a default: the budget is a module
    # global so that it can be lowered in a test, and a default argument would
    # freeze it at import.
    budget = _MAX_CROSS_EXPANSION_BYTES if max_bytes is None else max_bytes
    p_b = int(B_unique.shape[1])
    out = np.zeros((int(n_bins), p_b), dtype=np.float64)
    n_rows = int(np.size(support_idx))
    if n_rows == 0:
        return out
    chunk = min(n_rows, _cross_expansion_chunk_rows(p_b, 0, budget))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = _expand_support_rows(B_unique, support_idx[start:stop])
        out += _weighted_bincount_2d(bin_idx[start:stop], weights[start:stop], block, int(n_bins))
        # See _support_support_raw_cross: the next expansion is evaluated before
        # the name is rebound, so without this the ceiling is 2x the budget.
        del block
    return out


def _mixed_chunk_stop(
    indptr: NDArray, start_row: int, n_rows: int, dense_rows: int, max_nnz: int
) -> int:
    """End of a mixed-pair row chunk, budgeting the SPARSE payload as well.

    Sizing by the dense expansion alone budgets only ``rows * p_b * 8``.  The
    weighted CSR slice beside it costs ``nnz-in-range * 12`` (float64 plus a
    32-bit index), and ``nnz`` per row is a property of the OTHER block -- for a
    cardinal-CR basis the rows are structurally dense.  A narrow compressed side
    therefore permits a huge row count, and the sparse payload follows it: five
    compressed columns admit ~1.68M rows, which against a 20-column dense CSR is
    tens of millions of stored entries.  Boundaries come off ``indptr``, so the
    bound is the range's ACTUAL nonzeros rather than an average that a skewed
    row density would defeat.
    """
    stop_row = min(start_row + dense_rows, n_rows)
    limit = int(indptr[start_row]) + max_nnz
    # Largest row boundary whose cumulative nnz is still inside the budget.
    by_payload = int(np.searchsorted(indptr, limit, side="right")) - 1
    stop_row = min(stop_row, by_payload)
    # One row can never be split, so always make progress.
    return max(stop_row, start_row + 1)


def _weighted_row_chunk(csr, W_rows: NDArray, start_row: int, stop_row: int):
    """``diag(W[a:b]) @ csr[a:b]`` at ONE live buffer per stored entry.

    ``csr[a:b].multiply(w[:, None])`` costs three: the slice is a full copy
    (12 B/entry), and scipy routes the product through COO, allocating a row
    array, gathered weights and the product itself before returning -- measured
    32.2 B/entry against the ``// 12`` the chunk budget assumes, so the budget
    was optimistic by 2.7x exactly where it is load-bearing.

    Slicing ``indptr``/``indices``/``data`` directly keeps the index array a
    VIEW, and scaling into the ``repeat`` buffer rather than out of it leaves
    one float64 array live: measured 8.6 B/entry at full density and 8.9 at
    35%, which puts the budget back on the conservative side of the truth.

    Bit-exact, not merely close: the same two floats are multiplied in the same
    order as ``multiply`` performs them, so no reassociation occurs and fitted
    values are unchanged.  Scaling the DENSE side instead would be cheaper
    still, but computes ``b * (w * c)`` where this computes ``(w * b) * c``.
    """
    lo = int(csr.indptr[start_row])
    hi = int(csr.indptr[stop_row])
    row_ptr = csr.indptr[start_row : stop_row + 1]
    data = np.repeat(W_rows[start_row:stop_row], np.diff(row_ptr))
    data *= csr.data[lo:hi]
    # Built from the input's own class: this module never imports scipy, it
    # duck-types on whatever ``tocsr()`` returned.
    return csr.__class__(
        (data, csr.indices[lo:hi], row_ptr - lo),
        shape=(stop_row - start_row, csr.shape[1]),
    )


def _support_csr_raw_cross(
    B_unique: NDArray,
    support_idx: NDArray,
    B_csr,
    W_rows: NDArray,
) -> NDArray:
    """``B_support.T @ diag(W) @ B_csr`` with NO observation-level temporary.

    The mixed pairing -- one ``spline_cat`` block compressed, the other still
    CSR -- is created by the compression gate itself, so it is a regression
    rather than a pre-existing path: before, two exact blocks contracted sparse
    against sparse.  Expanding the compressed side and densifying the weighted
    CSR side made BOTH sides worse than they had been.

    Aggregating the CSR side onto the support bins first avoids the choice.
    The only dense array is ``(n_support, p_csr)``, bounded by the same support
    gate that bounds ``B_unique``; the CSR side is never densified and the
    observation rows are never materialised, so no chunking is needed.
    """
    # Row-chunked, in ONE pass over the data.  Column chunking bounded the
    # memory but made every pass re-walk all n row pointers -- measured at 88
    # passes for n_support=1e6 against a 700-column term, and the CSC rewrite
    # that removes the slicing cost still leaves that traversal, so it bought
    # 1.10x rather than fixing it.  Contracting over row chunks instead touches
    # each nonzero once, keeps the CSR side sparse throughout, and never forms
    # the (n_support, p_csr) aggregate at all -- so the cross-shaped array this
    # helper existed to bound is simply not built.
    csr = B_csr.tocsr()
    p_b = int(B_unique.shape[1])
    p_csr = int(csr.shape[1])
    support_idx = np.asarray(support_idx, dtype=np.intp)
    W_rows = np.asarray(W_rows, dtype=np.float64)
    out = np.zeros((p_b, p_csr), dtype=np.float64)
    n_rows = int(W_rows.shape[0])
    if n_rows == 0:
        return out
    dense_rows = min(n_rows, _cross_expansion_chunk_rows(p_b, 0, _MAX_CROSS_EXPANSION_BYTES))
    # float64 payload plus a 32-bit column index per stored entry.
    max_nnz = max(1, _MAX_CROSS_EXPANSION_BYTES // 12)
    start_row = 0
    while start_row < n_rows:
        stop_row = _mixed_chunk_stop(csr.indptr, start_row, n_rows, dense_rows, max_nnz)
        left = _expand_support_rows(B_unique, support_idx[start_row:stop_row])
        right = _weighted_row_chunk(csr, W_rows, start_row, stop_row)
        # sparse.T @ dense keeps the CSR side sparse; the product is (p_csr, p_b).
        out += np.asarray(right.T @ left, dtype=np.float64).T
        del left, right
        start_row = stop_row
    return out


def _support_support_raw_cross(
    B_unique_i: NDArray,
    bin_idx_i: NDArray,
    B_unique_j: NDArray,
    bin_idx_j: NDArray,
    W_rows: NDArray,
    max_bytes: int | None = None,
) -> NDArray:
    """``B_i.T @ diag(W) @ B_j`` for two support-indexed blocks over shared rows.

    The 2-D weight histogram both callers prefer costs ``n_bins_i * n_bins_j``
    cells, which is bounded only when the supports are bins.  A lossless
    support is bounded by the row count instead, so on wide supports this
    contracts over the shared rows and lets BLAS do the work.

    Chunked over rows, because the row count is exactly what a lossless support
    does NOT bound: expanding both sides in one go costs
    ``n_rows * (p_i + p_j) * 8`` bytes, which for a dominant level on a large
    book runs to hundreds of MB per call, inside solver iterations.  The cap
    that routes here bounds cells; without this it would only move the memory
    rather than bound it.  Accumulating in chunks keeps the transient at
    ``max_bytes`` regardless of row count, and the contraction is a sum over
    rows so partitioning it changes nothing but summation order.
    """
    budget = _MAX_CROSS_EXPANSION_BYTES if max_bytes is None else max_bytes
    p_i = int(B_unique_i.shape[1])
    p_j = int(B_unique_j.shape[1])
    out = np.zeros((p_i, p_j), dtype=np.float64)
    n_rows = int(W_rows.shape[0])
    if n_rows == 0:
        return out
    chunk = min(n_rows, _cross_expansion_chunk_rows(p_i, p_j, budget))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        left = _expand_support_rows(B_unique_i, bin_idx_i[start:stop])
        right = _expand_support_rows(B_unique_j, bin_idx_j[start:stop])
        # Fancy indexing already returned a fresh array, so scaling it in place
        # keeps the live count at two blocks rather than three.
        right *= W_rows[start:stop, None]
        out += left.T @ right
        # Released explicitly: the next iteration's expansion is evaluated
        # BEFORE its name is rebound, so without this the previous pair is
        # still referenced at the allocation instant and the real ceiling is
        # 1.5x the budget rather than 1x.
        del left, right
    return out


def _cross_gram_categorical_spline_categorical(
    gm_cat: GroupMatrix,
    gm_spline_cat: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram X_cat.T W X_spline_cat via one categorical aggregation."""
    if hasattr(gm_spline_cat, "B_unique"):
        rows = gm_spline_cat.row_idx
        # Chunked: this expands the level to observation rows, which no cap
        # above it bounds.  Pre-dates support compression -- the binned path
        # reaches it too -- but compression puts it on the hot path of every
        # model pairing a Categorical main effect with a spline_cat term,
        # which is every model this compression targets.
        B_agg = _chunked_support_bincount_2d(
            gm_cat.codes[rows],
            W[rows],
            gm_spline_cat.B_unique,
            gm_spline_cat.bin_idx_level,
            gm_cat.n_levels + 1,
        )
        return B_agg[: gm_cat.n_levels] @ gm_spline_cat.R_inv

    B_agg = _csr_weighted_bincount(
        gm_spline_cat._data,
        gm_spline_cat._indices,
        gm_spline_cat._indptr,
        gm_spline_cat._p_b,
        gm_cat.codes[gm_spline_cat.row_idx],
        W[gm_spline_cat.row_idx],
        gm_cat.n_levels + 1,
    )
    return B_agg[: gm_cat.n_levels] @ gm_spline_cat.R_inv


def _cross_gram_spline_categorical_spline_categorical(
    gm_i: GroupMatrix,
    gm_j: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram between compact spline-category level groups."""
    same_cat_parent = getattr(gm_i, "spline_cat_feature", None) is not None and getattr(
        gm_i, "spline_cat_feature", None
    ) == getattr(gm_j, "spline_cat_feature", None)
    if same_cat_parent and getattr(gm_i, "spline_cat_level", None) != getattr(
        gm_j, "spline_cat_level", None
    ):
        return np.zeros((gm_i.shape[1], gm_j.shape[1]))

    if same_cat_parent and np.array_equal(gm_i.row_idx, gm_j.row_idx):
        rows = gm_i.row_idx
        i_discrete = hasattr(gm_i, "B_unique")
        j_discrete = hasattr(gm_j, "B_unique")
        if i_discrete and j_discrete:
            if gm_i.n_bins * gm_j.n_bins <= _MAX_DISC_DISC_HIST_CELLS:
                H = _disc_disc_2d_hist(
                    gm_i.bin_idx_level,
                    gm_j.bin_idx_level,
                    W[rows],
                    gm_i.n_bins,
                    gm_j.n_bins,
                )
                raw = gm_i.B_unique.T @ H @ gm_j.B_unique
            else:
                raw = _support_support_raw_cross(
                    gm_i.B_unique,
                    gm_i.bin_idx_level,
                    gm_j.B_unique,
                    gm_j.bin_idx_level,
                    W[rows],
                )
            return gm_i.R_inv.T @ raw @ gm_j.R_inv
        if i_discrete:
            raw = _support_csr_raw_cross(gm_i.B_unique, gm_i.bin_idx_level, gm_j.B_level, W[rows])
            return gm_i.R_inv.T @ raw @ gm_j.R_inv
        if j_discrete:
            raw = _support_csr_raw_cross(gm_j.B_unique, gm_j.bin_idx_level, gm_i.B_level, W[rows]).T
            return gm_i.R_inv.T @ raw @ gm_j.R_inv

        raw = gm_i.B_level.T @ gm_j.B_level.multiply(W[rows][:, None])
        if hasattr(raw, "toarray"):
            raw = raw.toarray()
        return gm_i.R_inv.T @ np.asarray(raw, dtype=np.float64) @ gm_j.R_inv

    common_rows, pos_i, pos_j = np.intersect1d(
        gm_i.row_idx,
        gm_j.row_idx,
        assume_unique=True,
        return_indices=True,
    )
    if common_rows.size == 0:
        return np.zeros((gm_i.shape[1], gm_j.shape[1]))

    i_discrete = hasattr(gm_i, "B_unique")
    j_discrete = hasattr(gm_j, "B_unique")
    if i_discrete and j_discrete:
        if gm_i.n_bins * gm_j.n_bins <= _MAX_DISC_DISC_HIST_CELLS:
            H = _disc_disc_2d_hist(
                gm_i.bin_idx_level[pos_i],
                gm_j.bin_idx_level[pos_j],
                W[common_rows],
                gm_i.n_bins,
                gm_j.n_bins,
            )
            raw = gm_i.B_unique.T @ H @ gm_j.B_unique
        else:
            raw = _support_support_raw_cross(
                gm_i.B_unique,
                gm_i.bin_idx_level[pos_i],
                gm_j.B_unique,
                gm_j.bin_idx_level[pos_j],
                W[common_rows],
            )
        return gm_i.R_inv.T @ raw @ gm_j.R_inv

    if i_discrete:
        raw = _support_csr_raw_cross(
            gm_i.B_unique, gm_i.bin_idx_level[pos_i], gm_j.B_level[pos_j], W[common_rows]
        )
        return gm_i.R_inv.T @ raw @ gm_j.R_inv

    if j_discrete:
        raw = _support_csr_raw_cross(
            gm_j.B_unique, gm_j.bin_idx_level[pos_j], gm_i.B_level[pos_i], W[common_rows]
        ).T
        return gm_i.R_inv.T @ raw @ gm_j.R_inv

    B_i = gm_i.B_level[pos_i]
    B_j = gm_j.B_level[pos_j]
    raw = B_i.T @ B_j.multiply(W[common_rows][:, None])
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    return gm_i.R_inv.T @ np.asarray(raw, dtype=np.float64) @ gm_j.R_inv


def _cross_gram_discrete_spline_categorical(
    gm_disc: DiscretizedSSPGroupMatrix,
    gm_spline_cat: GroupMatrix,
    W: NDArray,
) -> NDArray | None:
    """Cross-gram X_disc.T W X_spline_cat from compressed support bins."""
    if not hasattr(gm_spline_cat, "B_unique"):
        return None
    rows = gm_spline_cat.row_idx
    n_joint = gm_disc.n_bins * gm_spline_cat.n_bins
    if n_joint > _MAX_DISC_DISC_HIST_CELLS:
        return None
    H = _disc_disc_2d_hist(
        gm_disc.bin_idx[rows],
        gm_spline_cat.bin_idx_level,
        W[rows],
        gm_disc.n_bins,
        gm_spline_cat.n_bins,
    )
    raw = gm_disc.B_unique.T @ H @ gm_spline_cat.B_unique
    return gm_disc.R_inv.T @ raw @ gm_spline_cat.R_inv


def _support_support_cross_gram(
    gm_disc: DiscretizedSSPGroupMatrix,
    gm_spline_cat: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """The same product as :func:`_cross_gram_discrete_spline_categorical`,
    for the case that one declines.

    Two lossless supports can make the joint histogram exceed the cell cap.
    Falling through from there used to reach ``_agg_by_bin``, whose output is
    ``(gm_disc.n_bins, p_spline_cat)`` -- cross-shaped, with the row count from
    one block and the width from the other, so neither support gate bounds it.
    Contracting over the shared rows instead reuses the row-chunked path, whose
    transient is the byte budget regardless of either support size.
    """
    rows = gm_spline_cat.row_idx
    raw = _support_support_raw_cross(
        gm_disc.B_unique,
        gm_disc.bin_idx[rows],
        gm_spline_cat.B_unique,
        gm_spline_cat.bin_idx_level,
        W[rows],
    )
    return gm_disc.R_inv.T @ raw @ gm_spline_cat.R_inv


def _cross_gram_by_columns(gm_i: GroupMatrix, gm_j: GroupMatrix, W: NDArray) -> NDArray:
    """Form a cross-product one generated column at a time.

    This is the bounded-memory fallback for factored support-space groups.
    It preserves the same smaller-width loop count as the generic fallback
    without materializing either effective observation-level design block.
    """
    p_i = gm_i.shape[1]
    p_j = gm_j.shape[1]
    if p_i <= p_j:
        result = np.empty((p_i, p_j), dtype=np.float64)
        unit = np.zeros(p_i, dtype=np.float64)
        for column in range(p_i):
            unit[column] = 1.0
            result[column] = gm_j.rmatvec(W * gm_i.matvec(unit))
            unit[column] = 0.0
        return result

    result = np.empty((p_i, p_j), dtype=np.float64)
    unit = np.zeros(p_j, dtype=np.float64)
    for column in range(p_j):
        unit[column] = 1.0
        result[:, column] = gm_i.rmatvec(W * gm_j.matvec(unit))
        unit[column] = 0.0
    return result


def _cross_gram(
    gm_i: GroupMatrix,
    gm_j: GroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Compute X_i.T @ diag(W) @ X_j efficiently.

    For two DiscretizedSSPGroupMatrix, uses a 2D weight histogram to avoid
    materializing either (n, p) matrix. For disc × non-disc, aggregates by
    disc bins in a single compiled pass (fused W-weighting, no toarray for
    sparse groups). Otherwise falls back to materializing the smaller group
    and using rmatvec on the larger.
    """
    (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        _SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
    from ..group_matrix import FactorSmoothGroupMatrix

    SplineCatTypes = (SplineCategoricalGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)

    if isinstance(gm_i, SplineCatTypes) and isinstance(gm_j, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_spline_categorical_spline_categorical(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_spline_cat_spline_cat_s", t0)
        return result
    # Tensor × tensor (same marginals, e.g. decomposed bilinear/wiggly)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and gm_i.tensor_id == gm_j.tensor_id
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_tensor(gm_i, gm_j, W, cache)
        _profile_elapsed(profile, "block_cross_tensor_tensor_s", t0)
        return result
    if isinstance(gm_i, DiscretizedTensorGroupMatrix) and isinstance(
        gm_j, DiscretizedTensorGroupMatrix
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_tensor_shared_margin(gm_i, gm_j, W)
        if result is not None:
            _profile_elapsed(profile, "block_cross_tensor_tensor_s", t0)
            return result

    if isinstance(gm_i, DiscretizedTensorGroupMatrix) and isinstance(gm_j, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_spline_categorical(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_tensor_spline_cat_s", t0)
        return result
    if isinstance(gm_j, DiscretizedTensorGroupMatrix) and isinstance(gm_i, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_spline_categorical(gm_j, gm_i, W).T
        _profile_elapsed(profile, "block_cross_tensor_spline_cat_s", t0)
        return result

    if isinstance(gm_i, CategoricalGroupMatrix) and isinstance(gm_j, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_categorical_spline_categorical(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_cat_spline_cat_s", t0)
        return result
    if isinstance(gm_j, CategoricalGroupMatrix) and isinstance(gm_i, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_categorical_spline_categorical(gm_j, gm_i, W).T
        _profile_elapsed(profile, "block_cross_cat_spline_cat_s", t0)
        return result

    if (
        isinstance(gm_i, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedSplineCategoricalGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_discrete_spline_categorical(gm_i, gm_j, W)
        if result is None:
            # Declined at the cell cap.  Falling through would reach the
            # disc-x-non-disc branch and _agg_by_bin, whose output is
            # cross-shaped (gm_i.n_bins, p_j) and bounded by neither support
            # gate.  Contract over rows instead: already chunked, already
            # bounded, and it is the same product.
            result = _support_support_cross_gram(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result
    if (
        isinstance(gm_j, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and isinstance(gm_i, DiscretizedSplineCategoricalGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_discrete_spline_categorical(gm_j, gm_i, W)
        if result is None:
            result = _support_support_cross_gram(gm_j, gm_i, W)
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result.T

    # Tensor × discretized main-effect (not tensor × tensor with different ids)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_j, DiscretizedTensorGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        own_margin = _cross_gram_tensor_own_margin(gm_i, gm_j, W, cache)
        if own_margin is not None:
            _profile_elapsed(profile, "block_cross_tensor_own_margin_s", t0)
            return own_margin.T
        result = _cross_gram_tensor_main(gm_i, gm_j, W).T
        _profile_elapsed(profile, "block_cross_tensor_main_s", t0)
        return result
    if (
        isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and isinstance(gm_i, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_i, DiscretizedTensorGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        own_margin = _cross_gram_tensor_own_margin(gm_j, gm_i, W, cache)
        if own_margin is not None:
            _profile_elapsed(profile, "block_cross_tensor_own_margin_s", t0)
            return own_margin
        result = _cross_gram_tensor_main(gm_j, gm_i, W)
        _profile_elapsed(profile, "block_cross_tensor_main_s", t0)
        return result

    if isinstance(gm_i, DiscretizedSCOPGroupMatrix) and isinstance(
        gm_j, DiscretizedSCOPGroupMatrix
    ):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            t0 = perf_counter() if profile is not None else 0.0
            W_2d = (
                _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
                if cache is None
                else cache.disc_disc_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            )
            result = gm_i.B_scop_unique.T @ W_2d @ gm_j.B_scop_unique
            _profile_elapsed(profile, "block_cross_disc_disc_s", t0)
            return result

    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and isinstance(gm_j, DiscretizedSCOPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            t0 = perf_counter() if profile is not None else 0.0
            W_2d = (
                _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
                if cache is None
                else cache.disc_disc_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            )
            BtWB = gm_i.B_unique.T @ W_2d @ gm_j.B_scop_unique
            result = gm_i.R_inv.T @ BtWB
            _profile_elapsed(profile, "block_cross_disc_disc_s", t0)
            return result

    if isinstance(gm_i, DiscretizedSCOPGroupMatrix) and isinstance(gm_j, DiscretizedSSPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            t0 = perf_counter() if profile is not None else 0.0
            W_2d = (
                _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
                if cache is None
                else cache.disc_disc_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            )
            BtWB = gm_i.B_scop_unique.T @ W_2d @ gm_j.B_unique
            result = BtWB @ gm_j.R_inv
            _profile_elapsed(profile, "block_cross_disc_disc_s", t0)
            return result

    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and isinstance(gm_j, DiscretizedSSPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            # Fused 2D histogram: single O(n) pass, no (n,) temp allocations.
            t0 = perf_counter() if profile is not None else 0.0
            W_2d = (
                _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
                if cache is None
                else cache.disc_disc_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            )
            BtWB = gm_i.B_unique.T @ W_2d @ gm_j.B_unique
            result = gm_i.R_inv.T @ BtWB @ gm_j.R_inv
            _profile_elapsed(profile, "block_cross_disc_disc_s", t0)
            return result

    if isinstance(gm_i, DiscretizedSCOPGroupMatrix):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_j, gm_i.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins)
        result = gm_i.B_scop_unique.T @ WX_agg
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    if isinstance(gm_j, DiscretizedSCOPGroupMatrix):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_i, gm_j.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_i, gm_j.bin_idx, W, gm_j.n_bins)
        result = (gm_j.B_scop_unique.T @ WX_agg).T
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    # Disc × non-disc: batch aggregate by disc bins, then dense matmuls.
    # Avoids per-column rmatvec loop, toarray() for sparse groups, and
    # the (n, p) W-broadcast allocation.
    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_j, DiscretizedSSPGroupMatrix
    ):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_j, gm_i.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins)
        result = gm_i.R_inv.T @ (gm_i.B_unique.T @ WX_agg)
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    if isinstance(gm_j, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_i, DiscretizedSSPGroupMatrix
    ):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_i, gm_j.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_i, gm_j.bin_idx, W, gm_j.n_bins)
        result = (gm_j.R_inv.T @ (gm_j.B_unique.T @ WX_agg)).T
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    # Cat × cat: weighted crosstab — O(n) with no dense allocation.
    if isinstance(gm_i, CategoricalGroupMatrix) and isinstance(gm_j, CategoricalGroupMatrix):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cat_cat_weighted_crosstab(gm_i.codes, gm_j.codes, W, gm_i.n_levels, gm_j.n_levels)
        _profile_elapsed(profile, "block_cross_cat_cat_s", t0)
        return result

    # Factored support-space groups must never be selected for the generic
    # observation-matrix materialization below. Generate the narrower side a
    # column at a time and retain O(n) working memory instead.
    support_space_types = (_SparseSSPGroupMatrix, FactorSmoothGroupMatrix, *SplineCatTypes)
    if isinstance(gm_i, support_space_types) or isinstance(gm_j, support_space_types):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_by_columns(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_fallback_s", t0)
        return result

    # Non-disc × non-disc: materialize smaller side, rmatvec larger side.
    t0 = perf_counter() if profile is not None else 0.0
    if gm_i.shape[1] <= gm_j.shape[1]:
        X_i = gm_i.toarray()
        WX_i = W[:, None] * X_i
        result = np.vstack([gm_j.rmatvec(WX_i[:, k]) for k in range(WX_i.shape[1])])
        _profile_elapsed(profile, "block_cross_fallback_s", t0)
        return result

    X_j = gm_j.toarray()
    WX_j = W[:, None] * X_j
    result = np.column_stack([gm_i.rmatvec(WX_j[:, k]) for k in range(WX_j.shape[1])])
    _profile_elapsed(profile, "block_cross_fallback_s", t0)
    return result


def _gram_any_sign(gm: GroupMatrix, W: NDArray) -> NDArray:
    """Compute X'diag(W)X for arbitrary-sign weights.

    SSP and Discretized groups handle any-sign W natively (they never use
    sqrt(W)).  Dense and Sparse groups use sqrt(W) internally, which fails
    for negative W, so we fall back to explicit W[:, None] * X for those.
    """
    (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
    from ..group_matrix import FactorSmoothGroupMatrix

    if isinstance(
        gm,
        SparseSSPGroupMatrix
        | SplineCategoricalGroupMatrix
        | DiscretizedSplineCategoricalGroupMatrix
        | DiscretizedSSPGroupMatrix
        | DiscretizedSCOPGroupMatrix,
    ):
        return gm.gram(W)
    if isinstance(gm, FactorSmoothGroupMatrix):
        return gm.gram(W)
    if isinstance(gm, CategoricalGroupMatrix):
        return gm.gram(W)  # bincount-based diagonal, handles any-sign W
    X = gm.toarray()
    return (W[:, None] * X).T @ X


def _execution_plan_for_blocks(gms, groups, W: NDArray, tabmat_split):
    """Build the compatibility plan and verify legacy solver-column spans."""
    from ._group_matrix_execution import MatrixExecutionPlan

    plan = MatrixExecutionPlan(
        gms,
        n=len(W),
        ordinary_tabmat=tabmat_split is not None,
        prepared_ordinary_split=tabmat_split,
    )
    plan.validate_group_spans(groups)
    return plan


def _block_xtwx(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Compatibility entry point for a weighted design Gram."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    return plan.moments(W, signed=False, profile=profile).gram


def _block_xtwx_rhs(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    Wz: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compatibility entry point for a Gram, ``X'W``, and ``X'Wz``."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    moments = plan.moments(
        W,
        rhs=(Wz,),
        include_xtw=True,
        signed=False,
        profile=profile,
    )
    if moments.xtw is None:  # pragma: no cover - guaranteed by include_xtw
        raise RuntimeError("execution plan did not return X'W")
    return moments.gram, moments.xtw, moments.xt_rhs[0]


def _block_xtwx_signed(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Compatibility entry point for an arbitrary-sign weighted Gram."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    return plan.moments(W, signed=True, profile=profile).gram
