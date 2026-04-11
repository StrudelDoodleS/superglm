"""Private numba kernels shared by group-matrix helpers."""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True)
def _csr_weighted_gram(data, indices, indptr, W, p):
    """B.T @ diag(W) @ B exploiting CSR sparsity (symmetric accumulation)."""
    result = np.zeros((p, p))
    n = len(W)
    for i in range(n):
        w = W[i]
        start = indptr[i]
        end = indptr[i + 1]
        for a in range(start, end):
            ja = indices[a]
            va = data[a] * w
            for b in range(a, end):
                jb = indices[b]
                prod = va * data[b]
                result[ja, jb] += prod
                if a != b:
                    result[jb, ja] += prod
    return result


@njit(cache=True)
def _weighted_bincount_2d(bin_idx, W, M, n_bins):
    """Fused W-weighted multi-column bincount for dense M."""
    n = len(bin_idx)
    n_cols = M.shape[1]
    result = np.zeros((n_bins, n_cols))
    for i in range(n):
        b = bin_idx[i]
        w = W[i]
        for c in range(n_cols):
            result[b, c] += w * M[i, c]
    return result


@njit(cache=True)
def _csr_weighted_bincount(data, indices, indptr, n_cols, bin_idx, W, n_bins):
    """Fused CSR-aware W-weighted bincount."""
    n = len(bin_idx)
    result = np.zeros((n_bins, n_cols))
    for row in range(n):
        b = bin_idx[row]
        w = W[row]
        for ptr in range(indptr[row], indptr[row + 1]):
            col = indices[ptr]
            result[b, col] += w * data[ptr]
    return result


@njit(cache=True)
def _disc_disc_2d_hist(bin_idx_i, bin_idx_j, W, n_bins_i, n_bins_j):
    """Fused 2D histogram for disc-disc cross-gram."""
    n = len(W)
    result = np.zeros((n_bins_i, n_bins_j))
    for obs in range(n):
        result[bin_idx_i[obs], bin_idx_j[obs]] += W[obs]
    return result


@njit(cache=True)
def _disc_disc_2d_hist_channels(bin_idx_i, bin_idx_j, chan_idx, W, chan_vals, n_bins_i, n_bins_j):
    """Fused multi-channel 2D histogram for tensor-main cross-grams."""
    n = len(W)
    n_channels = chan_vals.shape[1]
    result = np.zeros((n_bins_i * n_bins_j, n_channels))
    for obs in range(n):
        row = bin_idx_i[obs] * n_bins_j + bin_idx_j[obs]
        w = W[obs]
        c_src = chan_idx[obs]
        for c in range(n_channels):
            result[row, c] += w * chan_vals[c_src, c]
    return result


@njit(cache=True)
def _fused_bincount_2(bin_idx, W, Wz, n_bins):
    """Fused dual bincount: aggregate W and Wz by bin in one O(n) pass."""
    n = len(bin_idx)
    W_agg = np.zeros(n_bins)
    Wz_agg = np.zeros(n_bins)
    for i in range(n):
        b = bin_idx[i]
        W_agg[b] += W[i]
        Wz_agg[b] += Wz[i]
    return W_agg, Wz_agg


@njit(cache=True)
def _fused_2d_bincount_2(idx1, idx2, W, Wz, n_bins1, n_bins2):
    """Fused dual 2D bincount for tensor gram_rmatvec."""
    n = len(idx1)
    W_grid = np.zeros((n_bins1, n_bins2))
    Wz_grid = np.zeros((n_bins1, n_bins2))
    for i in range(n):
        a = idx1[i]
        b = idx2[i]
        W_grid[a, b] += W[i]
        Wz_grid[a, b] += Wz[i]
    return W_grid, Wz_grid


@njit(cache=True)
def _cat_weighted_bincount(codes, bin_idx, W, n_bins, n_levels):
    """Scatter W into (n_bins, n_levels) by (bin_idx, codes) simultaneously."""
    result = np.zeros((n_bins, n_levels))
    for i in range(len(codes)):
        c = codes[i]
        if c < n_levels:
            result[bin_idx[i], c] += W[i]
    return result


@njit(cache=True)
def _cat_cat_weighted_crosstab(codes_i, codes_j, W, n_levels_i, n_levels_j):
    """Weighted crosstab: X_i.T @ diag(W) @ X_j for two categoricals."""
    result = np.zeros((n_levels_i, n_levels_j))
    for k in range(len(W)):
        ci = codes_i[k]
        cj = codes_j[k]
        if ci < n_levels_i and cj < n_levels_j:
            result[ci, cj] += W[k]
    return result
