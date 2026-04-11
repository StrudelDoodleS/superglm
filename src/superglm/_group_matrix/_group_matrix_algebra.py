"""Private algebra helpers for group-matrix block operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_kernels import (
    _cat_cat_weighted_crosstab,
    _cat_weighted_bincount,
    _csr_weighted_bincount,
    _disc_disc_2d_hist,
    _disc_disc_2d_hist_channels,
    _weighted_bincount_2d,
)

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


def _runtime_group_matrix_types():
    """Import group-matrix runtime classes lazily to avoid circular imports."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
    )

    return (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
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
        _DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
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
    X = gm.toarray()
    return _weighted_bincount_2d(bin_idx, W, X, n_bins)


def _cross_gram_tensor_tensor(
    gm_i: DiscretizedTensorGroupMatrix,
    gm_j: DiscretizedTensorGroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram between two tensor groups sharing the same marginals.

    Used for decomposed tensor subgroups (bilinear × wiggly) that share
    the same B1_unique, B2_unique, idx1, idx2 but have different R_inv.
    """
    w_grid = _disc_disc_2d_hist(gm_i.idx1, gm_i.idx2, W, gm_i.n_bins1, gm_i.n_bins2)
    G_raw = gm_i._factored_gram_raw(w_grid)
    return gm_i.R_inv.T @ G_raw @ gm_j.R_inv


def _cross_gram_tensor_main(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_main: DiscretizedSSPGroupMatrix,
    W: NDArray,
) -> NDArray:
    """Blocked cross-gram between a tensor and a main-effect discretized group.

    Returns X_main.T @ diag(W) @ X_tensor in SSP space, shape (p_main, p_tensor).

    Uses K2 passes through 2D histograms — O(n*K2) total observation passes
    with O(n_bins_main * n_bins1) memory per pass.  No 3D histogram.

    Column ordering: j1 * K2 + j2, matching _row_kron_dense().
    """
    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    B_main = gm_main.B_unique
    K1, K2 = B1.shape[1], B2.shape[1]
    K_main_raw = B_main.shape[1]

    n_cells = gm_main.n_bins * gm_tensor.n_bins1 * K2
    if n_cells <= _MAX_DISC_DISC_CHANNEL_HIST_CELLS:
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


def _cross_gram(gm_i: GroupMatrix, gm_j: GroupMatrix, W: NDArray) -> NDArray:
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
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        _SparseSSPGroupMatrix,
    ) = _runtime_group_matrix_types()
    # Tensor × tensor (same marginals, e.g. decomposed bilinear/wiggly)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and gm_i.tensor_id == gm_j.tensor_id
    ):
        return _cross_gram_tensor_tensor(gm_i, gm_j, W)

    # Tensor × discretized main-effect (not tensor × tensor with different ids)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_j, DiscretizedTensorGroupMatrix)
    ):
        return _cross_gram_tensor_main(gm_i, gm_j, W).T
    if (
        isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and isinstance(gm_i, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_i, DiscretizedTensorGroupMatrix)
    ):
        return _cross_gram_tensor_main(gm_j, gm_i, W)

    if isinstance(gm_i, DiscretizedSCOPGroupMatrix) and isinstance(
        gm_j, DiscretizedSCOPGroupMatrix
    ):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            W_2d = _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            return gm_i.B_scop_unique.T @ W_2d @ gm_j.B_scop_unique

    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and isinstance(gm_j, DiscretizedSCOPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            W_2d = _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            BtWB = gm_i.B_unique.T @ W_2d @ gm_j.B_scop_unique
            return gm_i.R_inv.T @ BtWB

    if isinstance(gm_i, DiscretizedSCOPGroupMatrix) and isinstance(gm_j, DiscretizedSSPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            W_2d = _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            BtWB = gm_i.B_scop_unique.T @ W_2d @ gm_j.B_unique
            return BtWB @ gm_j.R_inv

    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and isinstance(gm_j, DiscretizedSSPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            # Fused 2D histogram: single O(n) pass, no (n,) temp allocations.
            W_2d = _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            BtWB = gm_i.B_unique.T @ W_2d @ gm_j.B_unique
            return gm_i.R_inv.T @ BtWB @ gm_j.R_inv

    if isinstance(gm_i, DiscretizedSCOPGroupMatrix):
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins)
        return gm_i.B_scop_unique.T @ WX_agg

    if isinstance(gm_j, DiscretizedSCOPGroupMatrix):
        WX_agg = _agg_by_bin(gm_i, gm_j.bin_idx, W, gm_j.n_bins)
        return (gm_j.B_scop_unique.T @ WX_agg).T

    # Disc × non-disc: batch aggregate by disc bins, then dense matmuls.
    # Avoids per-column rmatvec loop, toarray() for sparse groups, and
    # the (n, p) W-broadcast allocation.
    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_j, DiscretizedSSPGroupMatrix
    ):
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins)
        return gm_i.R_inv.T @ (gm_i.B_unique.T @ WX_agg)

    if isinstance(gm_j, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_i, DiscretizedSSPGroupMatrix
    ):
        WX_agg = _agg_by_bin(gm_i, gm_j.bin_idx, W, gm_j.n_bins)
        return (gm_j.R_inv.T @ (gm_j.B_unique.T @ WX_agg)).T

    # Cat × cat: weighted crosstab — O(n) with no dense allocation.
    if isinstance(gm_i, CategoricalGroupMatrix) and isinstance(gm_j, CategoricalGroupMatrix):
        return _cat_cat_weighted_crosstab(gm_i.codes, gm_j.codes, W, gm_i.n_levels, gm_j.n_levels)

    # Non-disc × non-disc: materialize smaller side, rmatvec larger side.
    if gm_i.shape[1] <= gm_j.shape[1]:
        X_i = gm_i.toarray()
        WX_i = W[:, None] * X_i
        return np.vstack([gm_j.rmatvec(WX_i[:, k]) for k in range(WX_i.shape[1])])

    X_j = gm_j.toarray()
    WX_j = W[:, None] * X_j
    return np.column_stack([gm_i.rmatvec(WX_j[:, k]) for k in range(WX_j.shape[1])])


def _gram_any_sign(gm: GroupMatrix, W: NDArray) -> NDArray:
    """Compute X'diag(W)X for arbitrary-sign weights.

    SSP and Discretized groups handle any-sign W natively (they never use
    sqrt(W)).  Dense and Sparse groups use sqrt(W) internally, which fails
    for negative W, so we fall back to explicit W[:, None] * X for those.
    """
    (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        SparseSSPGroupMatrix,
    ) = _runtime_group_matrix_types()
    if isinstance(
        gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix | DiscretizedSCOPGroupMatrix
    ):
        return gm.gram(W)
    if isinstance(gm, CategoricalGroupMatrix):
        return gm.gram(W)  # bincount-based diagonal, handles any-sign W
    X = gm.toarray()
    return (W[:, None] * X).T @ X


def _block_xtwx(gms: list[GroupMatrix], groups: list, W: NDArray, *, tabmat_split=None) -> NDArray:
    """Compute X.T @ diag(W) @ X block-by-block.

    Uses gm.gram(W) for diagonal blocks (O(n_bins) for discretized groups)
    and _cross_gram for off-diagonal blocks (2D histogram for disc-disc pairs).
    Avoids materializing the full (n, p_total) matrix.
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.sandwich.
    """
    if tabmat_split is not None:
        return np.asarray(tabmat_split.sandwich(W))
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        # Diagonal block
        XtWX[sl_i, sl_i] = gm_i.gram(W)

        # Cross blocks with subsequent groups
        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX


def _block_xtwx_rhs(
    gms: list[GroupMatrix], groups: list, W: NDArray, Wz: NDArray, *, tabmat_split=None
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
        DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        _SparseSSPGroupMatrix,
    ) = _runtime_group_matrix_types()
    if tabmat_split is not None:
        XtWX = np.asarray(tabmat_split.sandwich(W))
        XtW1 = np.asarray(tabmat_split.transpose_matvec(W))
        XtWz_out = np.asarray(tabmat_split.transpose_matvec(Wz))
        return XtWX, XtW1, XtWz_out
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))
    XtW1 = np.zeros(p_total)
    XtWz_out = np.zeros(p_total)

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        # Diagonal block + rmatvecs via shared bincount
        if isinstance(gm_i, DiscretizedSSPGroupMatrix | DiscretizedSCOPGroupMatrix):
            gram_i, xtw_i, xtwz_i = gm_i.gram_rmatvec(W, Wz)
            XtWX[sl_i, sl_i] = gram_i
            XtW1[sl_i] = xtw_i
            XtWz_out[sl_i] = xtwz_i
        else:
            XtWX[sl_i, sl_i] = gm_i.gram(W)
            XtW1[sl_i] = gm_i.rmatvec(W)
            XtWz_out[sl_i] = gm_i.rmatvec(Wz)

        # Cross blocks with subsequent groups
        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX, XtW1, XtWz_out


def _block_xtwx_signed(
    gms: list[GroupMatrix], groups: list, W: NDArray, *, tabmat_split=None
) -> NDArray:
    """Like _block_xtwx but safe for arbitrary-sign weights.

    Uses _gram_any_sign for diagonal blocks (avoids sqrt(W) in Dense/Sparse
    groups) and _cross_gram for off-diagonals (already sign-safe).
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.sandwich
    (which handles any-sign weights natively).
    """
    if tabmat_split is not None:
        return np.asarray(tabmat_split.sandwich(W))
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        XtWX[sl_i, sl_i] = _gram_any_sign(gm_i, W)

        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX
