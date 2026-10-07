"""Private numba kernels shared by group-matrix helpers.

The kernels a Gram block can reach (``_POOLED_BLOCK_KERNELS``) are compiled
``nogil=True``: the block queue (``_block_queue``) runs blocks on worker
threads, and a kernel that held the GIL would run them one at a time.  So are
the ones that write a TSQR leaf's rows on its worker (``_POOLED_LEAF_KERNELS``).  Every
one is nopython, so it touches no Python object while the GIL is released;
none uses ``prange``, so it starts no numba threads of its own; and each
writes only the arrays it allocates or the scratch its caller owns, which is
never shared between workers.  Kernels no pooled block reaches (the batched
channel passes, the structured and centred-assembly kernels) keep numba's
default and hold the GIL.
"""

from __future__ import annotations

from fractions import Fraction
from math import frexp

import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True, nogil=True)
def _tensor_operand_in_reassociation_range(values):
    """Check the exponent interval without allocating absolute-value/mask arrays."""
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if value != 0.0 and not 2.0**-128 <= abs(value) <= 2.0**128:
                return False
    return True


@njit(cache=True, nogil=True)
def _float64_operand_exponent_bounds(values):
    """Scan IEEE binary64 magnitudes with integer extrema and no array scratch."""
    bits = values.view(np.uint64)
    if values.flags.f_contiguous and not values.flags.c_contiguous:
        bits = bits.T
    smallest = np.uint64(0x7FF0000000000000)
    largest = np.uint64(0)
    for row in range(bits.shape[0]):
        for col in range(bits.shape[1]):
            # Positive binary64 encodings sort by magnitude, including
            # subnormals. Exclude both signed zeros from the minimum.
            magnitude = bits[row, col] & np.uint64(0x7FFFFFFFFFFFFFFF)
            smallest = min(smallest, magnitude if magnitude else np.uint64(0x7FF0000000000000))
            largest = max(largest, magnitude)
    # Every infinity and NaN encoding sorts at or above positive infinity.
    if largest >= np.uint64(0x7FF0000000000000):
        return -1024, 1024
    if largest == 0:
        return 0, 0
    return (
        frexp(np.uint64(smallest).view(np.float64))[1] - 1,
        frexp(np.uint64(largest).view(np.float64))[1],
    )


@njit(cache=True, nogil=True)
def _operand_exponent_bounds(values):
    """Enclose nonzero magnitudes by powers of two, without array scratch."""
    if values.dtype == np.dtype(np.float64):
        return _float64_operand_exponent_bounds(values)
    smallest, largest = np.inf, 0.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = abs(values[row, col])
            if not np.isfinite(value):
                return -1024, 1024
            if value:
                smallest = min(smallest, value)
                largest = max(largest, value)
    if largest == 0:
        return 0, 0
    return frexp(smallest)[1] - 1, frexp(largest)[1]


def _ssp_gram_needs_exact(*operands) -> bool:
    """Select exceptional binary64 arithmetic before raw moments lose range.

    The Gram sandwich has five input factors and at most four reductions,
    including bin aggregation. With operands in [2**-128, 2**128] and 63-bit
    addressable reductions, non-cancelling magnitudes stay below 2**892 and
    individual products stay above the underflow range. This selects
    arithmetic, not rank. Complex and wider dtypes keep their NumPy route.
    """
    arrays = [np.asarray(operand) for operand in operands]
    if any(array.dtype.kind not in "bifu" or array.dtype.itemsize > 8 for array in arrays):
        return False
    for array in arrays:
        # Numba cannot scan float16 arrays. Promotion is exact and affects
        # only classification; the native moment path keeps its operands.
        if array.dtype == np.float16:
            array = array.astype(np.float64)
        values = array if array.ndim == 2 else array.reshape(-1, 1)
        if not _tensor_operand_in_reassociation_range(values):
            return True
    return False


def _exact_ssp_moments(basis, transform, weights, weighted_rhs=None, *, bin_indices=None):
    """Round the requested source-factor moments once on exceptional inputs.

    Finite binary64 entries are exact dyadic rationals. Accumulating B R,
    X' W X and optional X' W / X' Wz over those entries avoids an overflowing
    raw Gram, reciprocal basis or bin mass. No rank or regularization decision
    is made here. The target is the represented source product B R, before
    floating-point matrix multiplication rounds that product.

    A shared binary scale per source operand makes every inner operation an
    integer operation. Binary64's finite exponent span bounds integer widths,
    apart from logarithmic growth with reduction length; no inner product
    needs rational normalization. Only final outputs use Fraction rounding.
    Only one effective row and the requested output are retained. Sparse B
    stays sparse outside the current row; discrete weights are aggregated
    exactly before visiting their support rows. Ordinary operands never use
    this path.
    """
    weights = np.asarray(weights)
    if bin_indices is None:
        n_observations = basis.shape[0]
    else:
        bin_indices = np.asarray(bin_indices)
        if (
            bin_indices.ndim != 1
            or bin_indices.dtype.kind not in "iu"
            or np.any(bin_indices < 0)
            or np.any(bin_indices >= basis.shape[0])
        ):
            raise ValueError("SSP bins must be one-dimensional support-row indices.")
        n_observations = bin_indices.size
    if weights.shape != (n_observations,):
        raise ValueError("SSP moments require one weight per observation.")
    if weighted_rhs is not None:
        weighted_rhs = np.asarray(weighted_rhs)
        if weighted_rhs.shape != weights.shape:
            raise ValueError("SSP moments require one weighted RHS per observation.")

    def binary_scale(values):
        values = np.asarray(values, dtype=np.float64)
        _, exponents = np.frexp(values)
        # A binary64 value with frexp exponent e is an integer times 2**(e-53).
        # Including nonfinite values in this scan is harmless: conversion
        # below validates all weights/transforms, but only active basis rows.
        return int(np.min(exponents, where=values != 0, initial=1024)) - 53

    def exact(value, scale):
        try:
            numerator, denominator = float(value).as_integer_ratio()
        except (ValueError, OverflowError) as error:
            raise np.linalg.LinAlgError("SSP moments require finite source factors.") from error
        shift = 1 - denominator.bit_length() - scale
        # The chosen scale divides every represented source value exactly,
        # including when large integral inputs require a right shift.
        return numerator << shift if shift >= 0 else numerator >> -shift

    def rounded(value, scale):
        return float(value << scale) if scale >= 0 else float(Fraction(value, 1 << -scale))

    sparse = getattr(basis, "format", None) == "csr"
    basis_scale = binary_scale(basis.data if sparse else basis)
    transform_scale = binary_scale(transform)
    weight_scale = binary_scale(weights)
    rhs_scale = 0 if weighted_rhs is None else binary_scale(weighted_rhs)
    coefficients = [
        [exact(value, transform_scale) for value in row] for row in np.asarray(transform)
    ]
    width = transform.shape[1]
    gram = [[0 for _ in range(width)] for _ in range(width)]
    xtw = [0 for _ in range(width)] if weighted_rhs is not None else None
    xtrhs = [0 for _ in range(width)] if weighted_rhs is not None else None
    if bin_indices is None:
        masses = [exact(value, weight_scale) for value in weights]
        rhs_masses = (
            None if weighted_rhs is None else [exact(value, rhs_scale) for value in weighted_rhs]
        )
    else:
        masses = [0 for _ in range(basis.shape[0])]
        rhs_masses = None if weighted_rhs is None else [0 for _ in range(basis.shape[0])]
        for row, index in enumerate(bin_indices):
            masses[index] += exact(weights[row], weight_scale)
            if rhs_masses is not None:
                rhs_masses[index] += exact(weighted_rhs[row], rhs_scale)

    for row, mass in enumerate(masses):
        rhs_mass = 0 if rhs_masses is None else rhs_masses[row]
        if mass == 0 and rhs_mass == 0:
            continue
        if sparse:
            # Convert each stored term before summing duplicate columns.
            # Densifying or canonicalizing first can lose their exact sum.
            nonzero = [
                (basis.indices[index], exact(basis.data[index], basis_scale))
                for index in range(basis.indptr[row], basis.indptr[row + 1])
                if basis.data[index] != 0
            ]
        else:
            raw = np.asarray(basis[row]).ravel()
            nonzero = [
                (index, exact(value, basis_scale)) for index, value in enumerate(raw) if value != 0
            ]
        effective = [
            sum(value * coefficients[index][column] for index, value in nonzero)
            for column in range(width)
        ]
        for left, value in enumerate(effective):
            weighted = mass * value
            for right in range(left, width):
                gram[left][right] += weighted * effective[right]
            if xtw is not None and xtrhs is not None:
                xtw[left] += weighted
                xtrhs[left] += rhs_mass * value
    for left in range(width):
        for right in range(left):
            gram[left][right] = gram[right][left]
    effective_scale = basis_scale + transform_scale
    try:
        rounded_gram = np.array(
            [[rounded(value, weight_scale + 2 * effective_scale) for value in row] for row in gram]
        ).reshape(width, width)
        rounded_xtw = (
            None
            if xtw is None
            else np.array([rounded(value, weight_scale + effective_scale) for value in xtw])
        )
        rounded_rhs = (
            None
            if xtrhs is None
            else np.array([rounded(value, rhs_scale + effective_scale) for value in xtrhs])
        )
    except OverflowError as error:
        raise np.linalg.LinAlgError("Requested SSP moment is not representable.") from error
    return rounded_gram, rounded_xtw, rounded_rhs


@njit(cache=True, nogil=True)
def _indexed_row_dot(left, right, left_idx, right_idx):
    """Row dot products gathered from two support tables, without row panels."""
    result = np.empty(len(left_idx), dtype=np.float64)
    for row in range(len(left_idx)):
        value = 0.0
        for col in range(left.shape[1]):
            value += left[left_idx[row], col] * right[right_idx[row], col]
        result[row] = value
    return result


@njit(cache=True, nogil=True)
def _csr_weighted_gram(data, indices, indptr, W, p, absolute_weights=False):
    """B.T @ diag(W) @ B exploiting CSR sparsity (symmetric accumulation)."""
    result = np.zeros((p, p))
    n = len(W)
    for i in range(n):
        w = W[i]
        if absolute_weights:
            w = abs(w)
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
def _csr_weighted_gram_channels(data, indices, indptr, W, start, width, p, absolute_weights):
    """``_csr_weighted_gram`` of ``W[:, start + c]`` in ``result[:, :, c]``.

    One row pass serves ``width`` channels. Each channel's entries receive
    the same products in the same row and pair order as the single kernel.
    """
    result = np.zeros((p, p, width))
    w = np.empty(width)
    va = np.empty(width)
    for i in range(W.shape[0]):
        for c in range(width):
            w[c] = abs(W[i, start + c]) if absolute_weights else W[i, start + c]
        end = indptr[i + 1]
        for a in range(indptr[i], end):
            data_a = data[a]
            for c in range(width):
                va[c] = data_a * w[c]
            for b in range(a, end):
                data_b = data[b]
                upper = result[indices[a], indices[b]]
                if a == b:
                    for c in range(width):
                        upper[c] += va[c] * data_b
                else:
                    lower = result[indices[b], indices[a]]
                    for c in range(width):
                        prod = va[c] * data_b
                        upper[c] += prod
                        lower[c] += prod
    return result


@njit(cache=True, nogil=True)
def _csr_weighted_cross(data, indices, indptr, other_data, other_indices, other_indptr, W, p, q):
    """B.T @ diag(W) @ C for two CSR blocks over shared rows, weighting B.

    Each entry adds fl(fl(W[i] * B[i, a]) * C[i, b]) over rows in ascending
    order, as SciPy's Gustavson product of the weighted transpose does.
    """
    result = np.zeros((p, q))
    for i in range(len(W)):
        w = W[i]
        for a in range(indptr[i], indptr[i + 1]):
            ja = indices[a]
            va = w * data[a]
            for b in range(other_indptr[i], other_indptr[i + 1]):
                result[ja, other_indices[b]] += va * other_data[b]
    return result


@njit(cache=True)
def _csr_weighted_cross_channels(
    data, indices, indptr, other_data, other_indices, other_indptr, W, start, width, p, q
):
    """``_csr_weighted_cross`` of ``W[:, start + c]`` in ``result[:, :, c]``."""
    result = np.zeros((p, q, width))
    va = np.empty(width)
    for i in range(W.shape[0]):
        w = W[i, start : start + width]
        for a in range(indptr[i], indptr[i + 1]):
            data_a = data[a]
            for c in range(width):
                va[c] = w[c] * data_a
            row = result[indices[a]]
            for b in range(other_indptr[i], other_indptr[i + 1]):
                cell = row[other_indices[b]]
                data_b = other_data[b]
                for c in range(width):
                    cell[c] += va[c] * data_b
    return result


def _csr_row_chunk(csr, start: int, stop: int, *, data=None):
    """CSR rows with shared entries and owned, rebased row pointers.

    SciPy's array constructor prunes small slices by copying their backing
    arrays, even with copy=False. Attach the buffers after constructing an
    empty CSR to preserve genuine views and the source index dtype. Weighted
    callers can supply their one owned value buffer without a second CSR.
    """
    lo, hi = int(csr.indptr[start]), int(csr.indptr[stop])
    result = csr.__class__((stop - start, csr.shape[1]), dtype=csr.dtype)
    result.data = csr.data[lo:hi] if data is None else data
    result.indices = csr.indices[lo:hi]
    result.indptr = csr.indptr[start : stop + 1] - lo
    return result


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _support_weighted_bincount_2d(out, bin_idx, W, B_unique, support_idx, col_start):
    """Add ``W[i] * B_unique[support_idx[i], col_start + c]`` into ``out[bin_idx[i], c]``.

    ``_weighted_bincount_2d`` of the gathered support rows, accumulated in
    place: the same products added in the same row order, with neither the
    gathered rows nor a separate result allocated.
    """
    n_cols = out.shape[1]
    for i in range(len(bin_idx)):
        b = bin_idx[i]
        w = W[i]
        row = support_idx[i]
        for c in range(n_cols):
            out[b, c] += w * B_unique[row, col_start + c]


@njit(cache=True, nogil=True)
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
def _csr_weighted_bincount_channels(
    data, indices, indptr, n_cols, bin_idx, W, start, width, n_bins
):
    """``_csr_weighted_bincount`` of ``W[:, start + c]`` in ``result[:, :, c]``."""
    result = np.zeros((n_bins, n_cols, width))
    for row in range(W.shape[0]):
        w = W[row, start : start + width]
        cells = result[bin_idx[row]]
        for ptr in range(indptr[row], indptr[row + 1]):
            cell = cells[indices[ptr]]
            value = data[ptr]
            for c in range(width):
                cell[c] += w[c] * value
    return result


@njit(cache=True, nogil=True)
def _disc_disc_2d_hist(bin_idx_i, bin_idx_j, W, n_bins_i, n_bins_j):
    """Fused 2D histogram for disc-disc cross-gram."""
    n = len(W)
    result = np.zeros((n_bins_i, n_bins_j))
    for obs in range(n):
        result[bin_idx_i[obs], bin_idx_j[obs]] += W[obs]
    return result


@njit(cache=True)
def _weighted_hist_channels(idx_a, idx_b, W, start, width, n_a, n_b):
    """Weighted 2-D histograms of ``W[:, start + c]`` in ``result[:, :, c]``.

    One row pass serves ``width`` channels: each sums its weights over the
    rows with an index pair in ascending row order, the additions of
    ``_disc_disc_2d_hist``, ``_cat_weighted_bincount``,
    ``_cat_cat_weighted_crosstab`` and ``np.bincount``. Rows whose index
    reaches ``n_a`` or ``n_b`` are skipped, as those kernels skip categorical
    sink codes. ``idx_b=None`` compiles the one-index bincount (``n_b == 1``).
    """
    result = np.zeros((n_a, n_b, width))
    for row in range(W.shape[0]):
        a = idx_a[row]
        if idx_b is None:
            b = 0
        else:
            b = idx_b[row]
        if a < n_a and b < n_b:
            cell = result[a, b]
            w = W[row, start : start + width]
            for c in range(width):
                cell[c] += w[c]
    return result


@njit(cache=True, nogil=True)
def _disc_disc_2d_hist_channels(bin_idx_i, bin_idx_j, chan_idx, W, chan_vals, n_bins_i, n_bins_j):
    """Fused multi-channel 2D histogram: tensor-main cross-grams and the dense
    stage of the tensor-tensor channel route."""
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


@njit(inline="always")
def _add_raw_row(acc, w, bin1, bin2, offsets1, values1, offsets2, values2, k2_raw):
    """Add ``w * kron(raw1[bin1], raw2[bin2])`` to one accumulator row, band by band."""
    origin = offsets1[bin1] * k2_raw + offsets2[bin2]
    # The row's width1 x width2 raw non-zeros: the algorithm is this nested loop.
    for a in range(values1.shape[1]):
        weighted = w * values1[bin1, a]
        base = origin + a * k2_raw
        for b in range(values2.shape[1]):
            acc[base + b] += weighted * values2[bin2, b]


@njit(cache=True, nogil=True)
def _gather_cell_order(order, bin1, bin2):
    """Permute a partner's channel bins into a grid's cell order.

    Two sequential passes so that the cell loop streams every input;
    gathering through ``order`` inside the cell loop instead cost as much as
    the dense kernel it replaces (31.5 against 30.3 ms, measured), the kernel
    being latency-bound on dependent random reads.  The weights take the same
    permutation once per grid tensor per build (``_BlockWeightCache``).
    """
    n = order.shape[0]
    bin1_sorted = np.empty(n, dtype=np.intp)
    bin2_sorted = np.empty(n, dtype=np.intp)
    for t in range(n):
        bin1_sorted[t] = bin1[order[t]]
    for t in range(n):
        bin2_sorted[t] = bin2[order[t]]
    return bin1_sorted, bin2_sorted


@njit(cache=True, nogil=True)
def _cell_hist_raw_kron(ptr, bin1, bin2, w, offsets1, values1, offsets2, values2, k2_raw, out):
    """Raw-band channel histogram over a cell-CSR: ``out[c] = sum over the rows
    ``t`` of cell ``c`` of ``w[t] * kron(raw1[bin1[t]], raw2[bin2[t]])``.

    ``bin1``, ``bin2`` and ``w`` are in cell order (``_gather_cell_order``), so
    every input streams, each cell sums into one L1-resident row, and ``out``
    is written once, sequentially.  Every row of ``out`` is written -- the
    empty cells with zeros -- so the caller passes scratch without zeroing it.
    """
    width = out.shape[1]
    acc = np.zeros(width)
    # Explicit loops for the zeroing and the copy: numba's slice assignment
    # made the whole kernel 2.5x slower (17.6 against 7.1 ms, measured).
    for cell in range(ptr.shape[0] - 1):
        for column in range(width):
            acc[column] = 0.0
        for t in range(ptr[cell], ptr[cell + 1]):
            _add_raw_row(acc, w[t], bin1[t], bin2[t], offsets1, values1, offsets2, values2, k2_raw)
        for column in range(width):
            out[cell, column] = acc[column]


@njit(cache=True, nogil=True)
def _cell_csr_matches(ptr, order, idx1, idx2, n_bins1, n_bins2):
    """Validate live cell membership without retaining another index copy."""
    n = len(idx1)
    if len(idx2) != n or len(order) != n or len(ptr) != n_bins1 * n_bins2 + 1:
        return False
    if ptr[0] != 0 or ptr[-1] != n:
        return False
    for cell in range(len(ptr) - 1):
        if not 0 <= ptr[cell] <= ptr[cell + 1] <= n:
            return False
        previous = -1
        for position in range(ptr[cell], ptr[cell + 1]):
            row = order[position]
            if not previous < row < n or idx1[row] * n_bins2 + idx2[row] != cell:
                return False
            previous = row
    return True


@njit(cache=True, nogil=True)
def _cell_csr(idx1, idx2, n_bins1, n_bins2):
    """Stable counting sort of the rows by grid cell ``idx1 * n_bins2 + idx2``.

    Returns ``(ptr, order)``: ``order[ptr[c]:ptr[c + 1]]`` are the rows of cell
    ``c`` in ascending row order, so a sum over a cell runs in one fixed order
    whatever visits it.  O(n + cells); ``np.argsort(kind="stable")`` on the
    same 300,000 rows measured 24x slower.
    """
    n_cells = n_bins1 * n_bins2
    ptr = np.zeros(n_cells + 1, dtype=np.int64)
    for row in range(idx1.shape[0]):
        ptr[idx1[row] * n_bins2 + idx2[row] + 1] += 1
    for cell in range(n_cells):
        ptr[cell + 1] += ptr[cell]
    fill = ptr[:n_cells].copy()
    order = np.empty(idx1.shape[0], dtype=np.intp)
    for row in range(idx1.shape[0]):
        cell = idx1[row] * n_bins2 + idx2[row]
        order[fill[cell]] = row
        fill[cell] += 1
    return ptr, order


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _factor_smooth_csr_rmatvec(data, indices, indptr, codes, values, n_levels, width):
    """Aggregate an observation vector into level-by-raw-basis coordinates."""
    result = np.zeros((n_levels, width))
    for row in range(len(codes)):
        level = codes[row]
        value = values[row]
        for ptr in range(indptr[row], indptr[row + 1]):
            result[level, indices[ptr]] += data[ptr] * value
    return result


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _cat_weighted_bincount(codes, bin_idx, W, n_bins, n_levels):
    """Scatter W into (n_bins, n_levels) by (bin_idx, codes) simultaneously."""
    result = np.zeros((n_bins, n_levels))
    for i in range(len(codes)):
        c = codes[i]
        if c < n_levels:
            result[bin_idx[i], c] += W[i]
    return result


@njit(cache=True, nogil=True)
def _level_sums(codes, weights, length):
    """``np.bincount(codes, weights, minlength=length)``, row by row in the same order.

    Bitwise ``np.bincount``'s sums (one ``+=`` per row, in row order), without
    the GIL ``np.bincount`` holds.  ``ok`` is False where a code falls
    outside ``[0, length)`` or the lengths differ; the caller then takes
    ``np.bincount`` itself.
    """
    out = np.zeros(length)
    if weights.shape[0] != codes.shape[0]:
        return out, False
    for i in range(codes.shape[0]):
        code = codes[i]
        if code < 0 or code >= length:
            return out, False
        out[code] += weights[i]
    return out, True


@njit(cache=True, nogil=True)
def _cat_cat_weighted_crosstab(codes_i, codes_j, W, n_levels_i, n_levels_j):
    """Weighted crosstab: X_i.T @ diag(W) @ X_j for two categoricals."""
    result = np.zeros((n_levels_i, n_levels_j))
    for k in range(len(W)):
        ci = codes_i[k]
        cj = codes_j[k]
        if ci < n_levels_i and cj < n_levels_j:
            result[ci, cj] += W[k]
    return result


@njit(cache=True, nogil=True)
def _weighted_centred_rows(out, column, table, index, first, center, center_lo, sqrt_weights):
    """``out[i, column + j] = ((table[index[first + i], j] - c) - c_lo) * sqrt_weights[i]``.

    ``c`` and ``c_lo`` are ``center[column + j]`` and ``center_lo[column +
    j]``; an empty one is left out.  The operations and their order are a
    TSQR leaf's (``rank._tsqr_leaf``), so its entries are bitwise the same.
    """
    centred = center.shape[0] > 0
    split = center_lo.shape[0] > 0
    for i in range(out.shape[0]):
        row = index[first + i]
        scale = sqrt_weights[i]
        for j in range(table.shape[1]):
            value = table[row, j]
            if centred:
                value = value - center[column + j]
            if split:
                value = value - center_lo[column + j]
            out[i, column + j] = value * scale


@njit(cache=True, nogil=True)
def _weighted_centred_one_hot(out, column, codes, n_levels, first, center, center_lo, sqrt_weights):
    """:func:`_weighted_centred_rows` of the one-hot rows of ``codes`` (code ``n_levels``: none)."""
    centred = center.shape[0] > 0
    split = center_lo.shape[0] > 0
    for i in range(out.shape[0]):
        code = codes[first + i]
        scale = sqrt_weights[i]
        for j in range(n_levels):
            value = 1.0 if code == j else 0.0
            if centred:
                value = value - center[column + j]
            if split:
                value = value - center_lo[column + j]
            out[i, column + j] = value * scale


_POOLED_LEAF_KERNELS = (_weighted_centred_rows, _weighted_centred_one_hot)
"""The kernels a TSQR leaf (``rank._tsqr_leaf``) runs on its pool worker: ``nogil`` too."""

_POOLED_BLOCK_KERNELS = (
    _tensor_operand_in_reassociation_range,
    _float64_operand_exponent_bounds,
    _operand_exponent_bounds,
    _indexed_row_dot,
    _csr_weighted_gram,
    _csr_weighted_cross,
    _weighted_bincount_2d,
    _support_weighted_bincount_2d,
    _csr_weighted_bincount,
    _disc_disc_2d_hist,
    _disc_disc_2d_hist_channels,
    _gather_cell_order,
    _cell_hist_raw_kron,
    _cell_csr_matches,
    _cell_csr,
    _fused_bincount_2,
    _factor_smooth_csr_matvec,
    _factor_smooth_support_matvec,
    _factor_smooth_csr_rmatvec,
    _factor_smooth_support_rmatvec,
    _factor_smooth_csr_sufficient_stats,
    _factor_smooth_support_cell_aggregates,
    _factor_smooth_csr_dense_cross,
    _factor_smooth_support_dense_cross,
    _fused_2d_bincount_2,
    _cat_weighted_bincount,
    _cat_cat_weighted_crosstab,
    _level_sums,
)
"""The kernels a diagonal or cross Gram block can reach: exactly the ``nogil`` ones."""


def _warmup_group_matrix_kernels() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    codes = np.array([0, 1], dtype=np.intp)
    csr_indices = np.array([0, 1], dtype=np.int32)
    csr_indptr = np.array([0, 1, 2], dtype=np.int32)
    matrix = np.eye(2, dtype=np.float64)
    frozen_matrix = matrix.copy()
    frozen_matrix.setflags(write=False)
    frozen_codes = codes.copy()
    frozen_codes.setflags(write=False)
    # Maps and packed curvature columns also supply Fortran and strided
    # operands. Cover both mutabilities so their first fit need not compile,
    # and call the float64 scan directly: a caller loaded from numba's
    # on-disk cache links it without compiling its dispatcher.
    for operand in (matrix, np.asfortranarray(matrix), np.ones((3, 4))[:, ::2]):
        _tensor_operand_in_reassociation_range(operand)
        _operand_exponent_bounds(operand)
        _float64_operand_exponent_bounds(operand)
        frozen_operand = operand.view()
        frozen_operand.setflags(write=False)
        _tensor_operand_in_reassociation_range(frozen_operand)
        _operand_exponent_bounds(frozen_operand)
        _float64_operand_exponent_bounds(frozen_operand)
    for support in (matrix, frozen_matrix):
        for indices in (codes, frozen_codes):
            _indexed_row_dot(matrix, support, indices, indices)
    # A TSQR leaf writes its block whole (no response) or less its last column.
    for block in (np.empty((2, 2)), np.empty((2, 3))[:, :2]):
        for centre in (values, values[:0]):
            _weighted_centred_rows(block, 0, matrix, codes, 0, centre, values[:0], values)
            _weighted_centred_one_hot(block, 0, codes, 2, 0, centre, values[:0], values)
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
    _csr_weighted_gram(values, csr_indices, csr_indptr, values, 2, absolute_weights=True)
    _csr_weighted_cross(
        values, csr_indices, csr_indptr, values, csr_indices, csr_indptr, values, 2, 2
    )
    # Batched W-derivative moments pass an (n, m) C-order block of directions.
    for absolute in (False, True):
        _csr_weighted_gram_channels(values, csr_indices, csr_indptr, matrix, 0, 2, 2, absolute)
    _csr_weighted_cross_channels(
        values, csr_indices, csr_indptr, values, csr_indices, csr_indptr, matrix, 0, 2, 2, 2
    )
    _csr_weighted_bincount_channels(values, csr_indices, csr_indptr, 2, codes, matrix, 0, 2, 2)
    for first in (codes, frozen_codes):
        _weighted_hist_channels(first, None, matrix, 0, 2, 2, 1)
        for second in (codes, frozen_codes):
            _weighted_hist_channels(first, second, matrix, 0, 2, 2, 2)
    _weighted_bincount_2d(codes, values, matrix, 2)
    # Support crosses pass read-only C-contiguous operands and a fresh output.
    frozen_values = values.copy()
    frozen_values.setflags(write=False)
    _support_weighted_bincount_2d(
        np.zeros((2, 2)), frozen_codes, frozen_values, frozen_matrix, frozen_codes, 0
    )
    _csr_weighted_bincount(values, csr_indices, csr_indptr, 2, codes, values, 2)
    _disc_disc_2d_hist(codes, codes, values, 2, 2)
    _disc_disc_2d_hist_channels(codes, codes, codes, values, matrix, 2, 2)
    cell_ptr, cell_order = _cell_csr(codes, codes, 2, 2)
    for first in (codes, frozen_codes):
        for second in (codes, frozen_codes):
            _cell_csr_matches(cell_ptr, cell_order, first, second, 2, 2)
    bin1, bin2 = _gather_cell_order(cell_order, codes, codes)
    w = values[cell_order]
    # Two raw columns per margin at band width two: every window starts at 0.
    starts = np.zeros(2, dtype=np.intp)
    _cell_hist_raw_kron(
        cell_ptr, bin1, bin2, w, starts, matrix, starts, matrix, 2, np.empty((4, 4))
    )
    # Inlined into the cell kernel at the IR level, so the row helper acquires
    # a signature of its own only through a direct call.
    _add_raw_row(np.zeros(4), 1.0, 0, 0, starts, matrix, starts, matrix, 2)
    _fused_bincount_2(codes, values, values, 2)
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
    frozen_weights = values.copy()
    frozen_weights.setflags(write=False)
    for level_codes in (codes, frozen_codes):
        for level_weights in (values, frozen_weights):
            _level_sums(level_codes, level_weights, 3)
