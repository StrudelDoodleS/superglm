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
def _random_effect_sufficient_stats(codes, W, Wz, n_levels):
    """Aggregate local Hessian diagonals and working RHS in one observation pass."""
    level_W = np.zeros(n_levels)
    level_Wz = np.zeros(n_levels)
    for i in range(len(codes)):
        level = codes[i]
        level_W[level] += W[i]
        level_Wz[level] += Wz[i]
    return level_W, level_Wz


@njit(cache=True)
def _factor_smooth_csr_matvec(data, indices, indptr, codes, raw_coefficients):
    """Apply a level-specific raw spline coefficient block to CSR rows."""
    result = np.zeros(len(codes))
    for row in range(len(codes)):
        level = codes[row]
        value = 0.0
        for ptr in range(indptr[row], indptr[row + 1]):
            value += data[ptr] * raw_coefficients[level, indices[ptr]]
        result[row] = value
    return result


@njit(cache=True)
def _factor_smooth_support_matvec(basis, bin_idx, codes, raw_coefficients):
    """Apply level-specific coefficients through a shared discrete support basis."""
    result = np.zeros(len(codes))
    width = basis.shape[1]
    for row in range(len(codes)):
        support_row = bin_idx[row]
        level = codes[row]
        value = 0.0
        for column in range(width):
            value += basis[support_row, column] * raw_coefficients[level, column]
        result[row] = value
    return result


@njit(cache=True)
def _factor_smooth_csr_rmatvec(data, indices, indptr, codes, values, n_levels, width):
    """Aggregate an observation vector into level-by-raw-basis coordinates."""
    result = np.zeros((n_levels, width))
    for row in range(len(codes)):
        level = codes[row]
        value = values[row]
        for ptr in range(indptr[row], indptr[row + 1]):
            result[level, indices[ptr]] += data[ptr] * value
    return result


@njit(cache=True)
def _factor_smooth_support_rmatvec(basis, bin_idx, codes, values, n_levels):
    """Aggregate an observation vector through a shared discrete support basis."""
    width = basis.shape[1]
    result = np.zeros((n_levels, width))
    for row in range(len(codes)):
        support_row = bin_idx[row]
        level = codes[row]
        value = values[row]
        for column in range(width):
            result[level, column] += basis[support_row, column] * value
    return result


@njit(cache=True)
def _factor_smooth_csr_sufficient_stats(
    data,
    indices,
    indptr,
    codes,
    weights,
    rhs,
    n_levels,
    width,
):
    """Fuse exact factor-smooth local Grams and two transpose products."""
    gram = np.zeros((n_levels, width, width))
    xtw = np.zeros((n_levels, width))
    xt_rhs = np.zeros((n_levels, width))
    for row in range(len(codes)):
        level = codes[row]
        weight = weights[row]
        rhs_value = rhs[row]
        start = indptr[row]
        end = indptr[row + 1]
        for left_ptr in range(start, end):
            left = indices[left_ptr]
            left_value = data[left_ptr]
            xtw[level, left] += left_value * weight
            xt_rhs[level, left] += left_value * rhs_value
            weighted_left = left_value * weight
            for right_ptr in range(left_ptr, end):
                right = indices[right_ptr]
                product = weighted_left * data[right_ptr]
                gram[level, left, right] += product
                if left != right:
                    gram[level, right, left] += product
    return gram, xtw, xt_rhs


@njit(cache=True)
def _factor_smooth_support_cell_aggregates(
    bin_idx,
    codes,
    weights,
    rhs,
    n_levels,
    n_bins,
):
    """Aggregate changing FactorSmooth values by level/support cell."""
    cell_weights = np.zeros((n_levels, n_bins))
    cell_rhs = np.zeros((n_levels, n_bins))
    for row in range(len(codes)):
        level = codes[row]
        support = bin_idx[row]
        cell_weights[level, support] += weights[row]
        cell_rhs[level, support] += rhs[row]
    return cell_weights, cell_rhs


@njit(cache=True)
def _factor_smooth_csr_dense_cross(
    data,
    indices,
    indptr,
    codes,
    weights,
    dense_small,
    n_levels,
    width,
):
    """Aggregate exact factor-smooth by dense-small weighted cross-products."""
    small_width = dense_small.shape[1]
    result = np.zeros((n_levels, width, small_width))
    for row in range(len(codes)):
        level = codes[row]
        weight = weights[row]
        for ptr in range(indptr[row], indptr[row + 1]):
            basis_column = indices[ptr]
            weighted_basis = weight * data[ptr]
            for small_column in range(small_width):
                result[level, basis_column, small_column] += (
                    weighted_basis * dense_small[row, small_column]
                )
    return result


@njit(cache=True)
def _factor_smooth_support_dense_cross(
    basis,
    bin_idx,
    codes,
    weights,
    dense_small,
    n_levels,
):
    """Aggregate discrete factor-smooth by dense-small weighted cross-products."""
    width = basis.shape[1]
    small_width = dense_small.shape[1]
    result = np.zeros((n_levels, width, small_width))
    for row in range(len(codes)):
        level = codes[row]
        support_row = bin_idx[row]
        weight = weights[row]
        for basis_column in range(width):
            weighted_basis = weight * basis[support_row, basis_column]
            for small_column in range(small_width):
                result[level, basis_column, small_column] += (
                    weighted_basis * dense_small[row, small_column]
                )
    return result


@njit(cache=True)
def _factor_smooth_support_dense_cell_aggregates(
    bin_idx,
    codes,
    weights,
    dense_small,
    n_levels,
    n_bins,
):
    """Aggregate weighted dense values once by level/support cell."""
    small_width = dense_small.shape[1]
    cells = np.zeros((n_levels, n_bins, small_width), dtype=np.float64)
    for row in range(len(codes)):
        level = codes[row]
        support = bin_idx[row]
        weight = weights[row]
        for small_column in range(small_width):
            cells[level, support, small_column] += weight * dense_small[row, small_column]
    return cells


@njit(cache=True)
def _dense_small_weighted_moments(X, W, Wz):
    """Fuse ``X'WX``, ``X'W``, and ``X'Wz`` for a narrow dense Schur block."""
    n, width = X.shape
    gram = np.zeros((width, width))
    xtw = np.zeros(width)
    xtwz = np.zeros(width)
    for row in range(n):
        weight = W[row]
        weighted_rhs = Wz[row]
        for left in range(width):
            value = X[row, left]
            xtw[left] += weight * value
            xtwz[left] += weighted_rhs * value
            weighted_value = weight * value
            for right in range(left, width):
                product = weighted_value * X[row, right]
                gram[left, right] += product
                if left != right:
                    gram[right, left] += product
    return gram, xtw, xtwz


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
def _pattern_support_summaries(
    row_patterns,
    unique_codes,
    W,
    Wz,
    marginal_offsets,
    pair_left,
    pair_right,
    pair_offsets,
    pair_right_sizes,
):
    """Aggregate all indexed-support marginals and pairs via unique row patterns."""
    pattern_w = np.zeros(unique_codes.shape[0], dtype=np.float64)
    pattern_wz = np.zeros(unique_codes.shape[0], dtype=np.float64)
    for obs in range(row_patterns.size):
        pattern = row_patterns[obs]
        pattern_w[pattern] += W[obs]
        pattern_wz[pattern] += Wz[obs]

    marginal_w = np.zeros(marginal_offsets[-1], dtype=np.float64)
    marginal_wz = np.zeros(marginal_offsets[-1], dtype=np.float64)
    joint_w = np.zeros(pair_offsets[-1], dtype=np.float64)
    for pattern in range(unique_codes.shape[0]):
        w = pattern_w[pattern]
        wz = pattern_wz[pattern]
        for group in range(unique_codes.shape[1]):
            cell = marginal_offsets[group] + unique_codes[pattern, group]
            marginal_w[cell] += w
            marginal_wz[cell] += wz
        for pair in range(pair_left.size):
            left = pair_left[pair]
            right = pair_right[pair]
            cell = (
                pair_offsets[pair]
                + unique_codes[pattern, left] * pair_right_sizes[pair]
                + unique_codes[pattern, right]
            )
            joint_w[cell] += w
    return marginal_w, marginal_wz, joint_w


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


def _warmup_group_matrix_kernels() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    codes = np.array([0, 1], dtype=np.intp)
    csr_indices = np.array([0, 1], dtype=np.int32)
    csr_indptr = np.array([0, 1, 2], dtype=np.int32)
    matrix = np.eye(2, dtype=np.float64)
    frozen_matrix = matrix.copy()
    frozen_matrix.setflags(write=False)
    row_patterns = np.array([0, 1], dtype=np.int32)
    unique_codes = np.array([[0, 0], [1, 1]], dtype=np.int32)
    marginal_offsets = np.array([0, 2, 4], dtype=np.intp)
    pair_left = codes[:1]
    pair_right = codes[1:]
    pair_offsets = np.array([0, 4], dtype=np.intp)
    pair_right_sizes = np.array([2], dtype=np.intp)
    for array in (
        row_patterns,
        unique_codes,
        marginal_offsets,
        pair_left,
        pair_right,
        pair_offsets,
        pair_right_sizes,
    ):
        array.setflags(write=False)

    _csr_weighted_gram(values, csr_indices, csr_indptr, values, 2)
    _weighted_bincount_2d(codes, values, matrix, 2)
    _csr_weighted_bincount(values, csr_indices, csr_indptr, 2, codes, values, 2)
    _disc_disc_2d_hist(codes, codes, values, 2, 2)
    _disc_disc_2d_hist_channels(codes, codes, codes, values, matrix, 2, 2)
    _fused_bincount_2(codes, values, values, 2)
    _random_effect_sufficient_stats(codes, values, values, 2)
    _factor_smooth_csr_matvec(values, csr_indices, csr_indptr, codes, matrix)
    _factor_smooth_support_matvec(matrix, codes, codes, matrix)
    _factor_smooth_csr_rmatvec(values, csr_indices, csr_indptr, codes, values, 2, 2)
    _factor_smooth_support_rmatvec(matrix, codes, codes, values, 2)
    _factor_smooth_csr_sufficient_stats(
        values, csr_indices, csr_indptr, codes, values, values, 2, 2
    )
    _factor_smooth_support_cell_aggregates(codes, codes, values, values, 2, 2)
    _factor_smooth_csr_dense_cross(
        values, csr_indices, csr_indptr, codes, values, frozen_matrix, 2, 2
    )
    _factor_smooth_support_dense_cross(matrix, codes, codes, values, frozen_matrix, 2)
    _factor_smooth_support_dense_cell_aggregates(codes, codes, values, frozen_matrix, 2, 2)
    _dense_small_weighted_moments(frozen_matrix, values, values)
    _fused_2d_bincount_2(codes, codes, values, values, 2, 2)
    _pattern_support_summaries(
        row_patterns,
        unique_codes,
        values,
        values,
        marginal_offsets,
        pair_left,
        pair_right,
        pair_offsets,
        pair_right_sizes,
    )
    _cat_weighted_bincount(codes, codes, values, 2, 2)
    _cat_cat_weighted_crosstab(codes, codes, values, 2, 2)
