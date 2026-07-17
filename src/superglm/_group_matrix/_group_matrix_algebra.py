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
from ._group_matrix_tabmat import _tabmat_vector

if TYPE_CHECKING:
    from ..group_matrix import (
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        GroupMatrix,
    )
else:
    GroupMatrix = Any

_MAX_DISC_DISC_HIST_CELLS = 5_000_000
_MAX_DISC_DISC_CHANNEL_HIST_CELLS = 5_000_000


def _profile_add(profile: dict[str, Any] | None, key: str, value: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + value


def _profile_count(profile: dict[str, Any] | None, key: str, value: int = 1) -> None:
    if profile is not None:
        profile[key] = int(profile.get(key, 0)) + value


def _profile_elapsed(profile: dict[str, Any] | None, key: str, start: float) -> None:
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
            return cached

        rev_key = self._key(idx_b, idx_a, W, n_b, n_a)
        rev_cached = self._hist2d.get(rev_key)
        if rev_cached is not None:
            hist = rev_cached.T
            self._hist2d[key] = hist
            return hist

        t0 = perf_counter() if self._profile is not None else 0.0
        hist = _disc_disc_2d_hist(idx_a, idx_b, W, n_a, n_b)
        _profile_elapsed(self._profile, "block_hist2d_s", t0)
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


def _agg_by_bin(gm: GroupMatrix, bin_idx: NDArray, W: NDArray, n_bins: int) -> NDArray:
    """Aggregate W * gm's columns by bin index → (n_bins, p_g) dense array.

    Dispatches to the most efficient kernel for each GroupMatrix type:
    - SparseGroupMatrix: CSR-aware kernel (avoids toarray, O(nnz) not O(n*p))
    - SparseSSPGroupMatrix: CSR kernel in B-spline space + R_inv transform
    - DenseGroupMatrix / other: fused dense kernel (avoids W-broadcast alloc)
    """
    (
        CategoricalGroupMatrix,
        _DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        _DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
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
        X_level = (gm.B_unique @ gm.R_inv)[gm.bin_idx_level]
        return _weighted_bincount_2d(bin_idx[rows], W[rows], X_level, n_bins)
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
    X = gm.toarray()
    return _weighted_bincount_2d(bin_idx, W, X, n_bins)


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


def _cross_gram_categorical_spline_categorical(
    gm_cat: GroupMatrix,
    gm_spline_cat: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram X_cat.T W X_spline_cat via one categorical aggregation."""
    if hasattr(gm_spline_cat, "B_unique"):
        rows = gm_spline_cat.row_idx
        B_agg = _weighted_bincount_2d(
            gm_cat.codes[rows],
            W[rows],
            gm_spline_cat.B_unique[gm_spline_cat.bin_idx_level],
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
            H = _disc_disc_2d_hist(
                gm_i.bin_idx_level,
                gm_j.bin_idx_level,
                W[rows],
                gm_i.n_bins,
                gm_j.n_bins,
            )
            raw = gm_i.B_unique.T @ H @ gm_j.B_unique
            return gm_i.R_inv.T @ raw @ gm_j.R_inv
        if i_discrete:
            B_i = gm_i.B_unique[gm_i.bin_idx_level]
            raw = B_i.T @ np.asarray(gm_j.B_level.multiply(W[rows][:, None]).toarray())
            return gm_i.R_inv.T @ np.asarray(raw, dtype=np.float64) @ gm_j.R_inv
        if j_discrete:
            B_j = gm_j.B_unique[gm_j.bin_idx_level]
            raw = np.asarray((gm_i.B_level.multiply(W[rows][:, None]).T @ B_j), dtype=np.float64)
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
        H = _disc_disc_2d_hist(
            gm_i.bin_idx_level[pos_i],
            gm_j.bin_idx_level[pos_j],
            W[common_rows],
            gm_i.n_bins,
            gm_j.n_bins,
        )
        raw = gm_i.B_unique.T @ H @ gm_j.B_unique
        return gm_i.R_inv.T @ raw @ gm_j.R_inv

    if i_discrete:
        B_i = gm_i.B_unique[gm_i.bin_idx_level[pos_i]]
        B_j = gm_j.B_level[pos_j]
        raw = B_i.T @ np.asarray(B_j.multiply(W[common_rows][:, None]).toarray())
        return gm_i.R_inv.T @ np.asarray(raw, dtype=np.float64) @ gm_j.R_inv

    if j_discrete:
        B_i = gm_i.B_level[pos_i]
        B_j = gm_j.B_unique[gm_j.bin_idx_level[pos_j]]
        raw = np.asarray((B_i.multiply(W[common_rows][:, None]).T @ B_j), dtype=np.float64)
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
        if result is not None:
            _profile_elapsed(profile, "block_cross_disc_other_s", t0)
            return result
    if (
        isinstance(gm_j, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and isinstance(gm_i, DiscretizedSplineCategoricalGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_discrete_spline_categorical(gm_j, gm_i, W)
        if result is not None:
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
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins)
        result = gm_i.B_scop_unique.T @ WX_agg
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    if isinstance(gm_j, DiscretizedSCOPGroupMatrix):
        t0 = perf_counter() if profile is not None else 0.0
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
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins)
        result = gm_i.R_inv.T @ (gm_i.B_unique.T @ WX_agg)
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    if isinstance(gm_j, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_i, DiscretizedSSPGroupMatrix
    ):
        t0 = perf_counter() if profile is not None else 0.0
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
    if isinstance(
        gm,
        SparseSSPGroupMatrix
        | SplineCategoricalGroupMatrix
        | DiscretizedSplineCategoricalGroupMatrix
        | DiscretizedSSPGroupMatrix
        | DiscretizedSCOPGroupMatrix,
    ):
        return gm.gram(W)
    if isinstance(gm, CategoricalGroupMatrix):
        return gm.gram(W)  # bincount-based diagonal, handles any-sign W
    X = gm.toarray()
    return (W[:, None] * X).T @ X


def _block_xtwx(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Compute X.T @ diag(W) @ X block-by-block.

    Uses gm.gram(W) for diagonal blocks (O(n_bins) for discretized groups)
    and _cross_gram for off-diagonal blocks (2D histogram for disc-disc pairs).
    Avoids materializing the full (n, p_total) matrix.
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.sandwich.
    """
    _profile_count(profile, "block_calls")
    if tabmat_split is not None:
        t0 = perf_counter() if profile is not None else 0.0
        result = np.asarray(tabmat_split.sandwich(_tabmat_vector(W)))
        _profile_elapsed(profile, "block_tabmat_s", t0)
        return result
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))
    cache = _BlockWeightCache(profile)

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        # Diagonal block
        t0 = perf_counter() if profile is not None else 0.0
        XtWX[sl_i, sl_i] = gm_i.gram(W)
        if profile is not None:
            (
                _CategoricalGroupMatrix,
                DiscretizedSCOPGroupMatrix,
                DiscretizedSplineCategoricalGroupMatrix,
                DiscretizedSSPGroupMatrix,
                DiscretizedTensorGroupMatrix,
                _SparseGroupMatrix,
                _SparseSSPGroupMatrix,
                SplineCategoricalGroupMatrix,
            ) = _runtime_group_matrix_types()
            if isinstance(gm_i, DiscretizedTensorGroupMatrix):
                _profile_elapsed(profile, "block_diag_tensor_s", t0)
            elif isinstance(
                gm_i,
                DiscretizedSSPGroupMatrix
                | DiscretizedSCOPGroupMatrix
                | DiscretizedSplineCategoricalGroupMatrix
                | SplineCategoricalGroupMatrix,
            ):
                _profile_elapsed(profile, "block_diag_discrete_ssp_s", t0)
            else:
                _profile_elapsed(profile, "block_diag_other_s", t0)

        # Cross blocks with subsequent groups
        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W, cache, profile)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX


def _block_xtwx_rhs(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    Wz: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute X'WX, X'W, and X'Wz in a single pass over the data.

    For DiscretizedSSPGroupMatrix, shares the O(n) bincount between gram and
    rmatvec operations.  Returns (XtWX, XtW1, XtWz) where XtW1 = X.T @ W
    and XtWz = X.T @ Wz.
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.
    """
    (
        _CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        _SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
    _profile_count(profile, "block_calls")
    if tabmat_split is not None:
        t0 = perf_counter() if profile is not None else 0.0
        tabmat_W = _tabmat_vector(W)
        tabmat_Wz = _tabmat_vector(Wz)
        XtWX = np.asarray(tabmat_split.sandwich(tabmat_W))
        XtW1 = np.asarray(tabmat_split.transpose_matvec(tabmat_W))
        XtWz_out = np.asarray(tabmat_split.transpose_matvec(tabmat_Wz))
        _profile_elapsed(profile, "block_tabmat_s", t0)
        return XtWX, XtW1, XtWz_out
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))
    XtW1 = np.zeros(p_total)
    XtWz_out = np.zeros(p_total)
    cache = _BlockWeightCache(profile)

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        # Diagonal block + rmatvecs via shared bincount
        if isinstance(gm_i, DiscretizedTensorGroupMatrix):
            t0 = perf_counter() if profile is not None else 0.0
            w_grid, wz_grid = cache.tensor_w_wz_grid(gm_i, W, Wz)
            gram_i, xtw_i, xtwz_i = gm_i.gram_rmatvec_from_grids(w_grid, wz_grid)
            XtWX[sl_i, sl_i] = gram_i
            XtW1[sl_i] = xtw_i
            XtWz_out[sl_i] = xtwz_i
            _profile_elapsed(profile, "block_diag_tensor_s", t0)
        elif isinstance(
            gm_i,
            DiscretizedSSPGroupMatrix
            | DiscretizedSCOPGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | SplineCategoricalGroupMatrix,
        ):
            t0 = perf_counter() if profile is not None else 0.0
            gram_i, xtw_i, xtwz_i = gm_i.gram_rmatvec(W, Wz)
            XtWX[sl_i, sl_i] = gram_i
            XtW1[sl_i] = xtw_i
            XtWz_out[sl_i] = xtwz_i
            _profile_elapsed(profile, "block_diag_discrete_ssp_s", t0)
        else:
            t0 = perf_counter() if profile is not None else 0.0
            XtWX[sl_i, sl_i] = gm_i.gram(W)
            XtW1[sl_i] = gm_i.rmatvec(W)
            XtWz_out[sl_i] = gm_i.rmatvec(Wz)
            _profile_elapsed(profile, "block_diag_other_s", t0)

        # Cross blocks with subsequent groups
        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W, cache, profile)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX, XtW1, XtWz_out


def _block_xtwx_signed(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Like _block_xtwx but safe for arbitrary-sign weights.

    Uses _gram_any_sign for diagonal blocks (avoids sqrt(W) in Dense/Sparse
    groups) and _cross_gram for off-diagonals (already sign-safe).
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.sandwich
    (which handles any-sign weights natively).
    """
    _profile_count(profile, "block_calls")
    if tabmat_split is not None:
        t0 = perf_counter() if profile is not None else 0.0
        result = np.asarray(tabmat_split.sandwich(_tabmat_vector(W)))
        _profile_elapsed(profile, "block_tabmat_s", t0)
        return result
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))
    cache = _BlockWeightCache(profile)

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        t0 = perf_counter() if profile is not None else 0.0
        XtWX[sl_i, sl_i] = _gram_any_sign(gm_i, W)
        if profile is not None:
            (
                _CategoricalGroupMatrix,
                DiscretizedSCOPGroupMatrix,
                DiscretizedSplineCategoricalGroupMatrix,
                DiscretizedSSPGroupMatrix,
                DiscretizedTensorGroupMatrix,
                _SparseGroupMatrix,
                _SparseSSPGroupMatrix,
                SplineCategoricalGroupMatrix,
            ) = _runtime_group_matrix_types()
            if isinstance(gm_i, DiscretizedTensorGroupMatrix):
                _profile_elapsed(profile, "block_diag_tensor_s", t0)
            elif isinstance(
                gm_i,
                DiscretizedSSPGroupMatrix
                | DiscretizedSCOPGroupMatrix
                | DiscretizedSplineCategoricalGroupMatrix
                | SplineCategoricalGroupMatrix,
            ):
                _profile_elapsed(profile, "block_diag_discrete_ssp_s", t0)
            else:
                _profile_elapsed(profile, "block_diag_other_s", t0)

        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W, cache, profile)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX
