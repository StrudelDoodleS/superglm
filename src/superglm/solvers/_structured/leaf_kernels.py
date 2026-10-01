"""Compiled row kernels of the nested chain's centred leaf pass (``moments._centered_leaf_pass``).

Every kernel is serial and visits rows in leaf order and columns left to
right, so each sum has one fixed order whatever the thread count.  The
arithmetic is the pass's own: a border row is formed as its matrix forms it
(a table row, a one-hot row, or a sparse row times ``R_inv`` in stored
order), less the global centre ``c``; a segment mean in the shifted form
about its reference row; the centred row, its weighted copy and the absolute
mass from it.
"""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True)
def _coded_rows(
    out,
    codes,
    tables,
    table_start,
    table_width,
    table_column,
    one_hot_width,
    one_hot_column,
    center,
):  # pragma: no cover - compiled
    """Write each row's table and one-hot blocks less the centre (``nested.LeafRows``)."""
    count = len(table_start)
    for row in range(codes.shape[0]):
        for block in range(count):
            column, width = table_column[block], table_width[block]
            base = table_start[block] + codes[row, block] * width
            source, target = tables[base : base + width], out[row, column : column + width]
            shift = center[column : column + width]
            for j in range(width):
                target[j] = source[j] - shift[j]
        for block in range(len(one_hot_width)):
            column, width = one_hot_column[block], one_hot_width[block]
            for j in range(width):
                out[row, column + j] = 0.0 - center[column + j]
            level = codes[row, count + block]
            if level < width:
                out[row, column + level] = 1.0 - center[column + level]


@njit(cache=True)
def _sparse_rows(out, column, indptr, indices, data, basis, lo, hi, center):  # pragma: no cover
    """Rows ``lo:hi`` of ``B @ basis`` from ``B``'s CSR arrays, entries in stored order, less the centre."""
    width = basis.shape[1]
    for row in range(hi - lo):
        for j in range(width):
            out[row, column + j] = 0.0
        for position in range(indptr[lo + row], indptr[lo + row + 1]):
            value = data[position]
            coefficient = indices[position]
            for j in range(width):
                out[row, column + j] += value * basis[coefficient, j]
        for j in range(width):
            out[row, column + j] -= center[column + j]


@njit(cache=True)
def _segment_means(rows, a, bounds, reference, total, means):  # pragma: no cover - compiled
    """Each segment's mean ``x_ref + sum_r a_r (x_r - x_ref) / W`` (0 where ``W == 0``) into ``means``."""
    width = rows.shape[1]
    for segment in range(len(bounds) - 1):
        for j in range(width):
            means[segment, j] = 0.0
        if total[segment] == 0.0:
            continue
        ref = reference[segment]
        for row in range(bounds[segment], bounds[segment + 1]):
            weight = a[row]
            for j in range(width):
                means[segment, j] += (rows[row, j] - rows[ref, j]) * weight
        for j in range(width):
            means[segment, j] = rows[ref, j] + means[segment, j] / total[segment]


@njit(cache=True)
def _center_rows(
    rows, weighted, split, a, error, bounds, means, absolute, sums, negative
):  # pragma: no cover - compiled
    """Centre each segment's rows on its mean in place and form the row statistics.

    Each weighted row ``x`` goes to ``split`` as ``sqrt(|a|) x``: rows of
    positive weight packed from the top, of negative weight from the bottom,
    and the two counts are returned, so the chunk's scatter is ``P'P - N'N``.
    ``weighted = a x`` when it has rows; ``absolute += e x^2`` with ``e`` the
    row's weight-error scale (``error``); ``sums[s] += sum_{r in s} a_r x_r``
    and ``negative[s] += sum_{r in s, a_r < 0} |a_r| x_r`` when those arrays
    have rows.  A row of zero weight adds exact zeros to every sum.
    """
    width = rows.shape[1]
    want_weighted = weighted.shape[0] > 0
    want_sums = sums.shape[0] > 0
    want_negative = negative.shape[0] > 0
    last = split.shape[0] - 1
    positive, negative_count = 0, 0
    for segment in range(len(bounds) - 1):
        for row in range(bounds[segment], bounds[segment + 1]):
            weight = a[row]
            scale = error[row]
            for j in range(width):
                rows[row, j] -= means[segment, j]
            for j in range(width):
                absolute[j] += scale * rows[row, j] * rows[row, j]
            if weight != 0.0:
                if weight > 0.0:
                    target = positive
                    positive += 1
                else:
                    target = last - negative_count
                    negative_count += 1
                    if want_negative:
                        for j in range(width):
                            negative[segment, j] -= weight * rows[row, j]
                root = np.sqrt(abs(weight))
                for j in range(width):
                    split[target, j] = root * rows[row, j]
            if want_weighted:
                for j in range(width):
                    weighted[row, j] = weight * rows[row, j]
            if want_sums:
                for j in range(width):
                    sums[segment, j] += weight * rows[row, j]
    return positive, negative_count


@njit(cache=True)
def _shifted_sums(rows, a, reference, out):  # pragma: no cover - compiled
    """``out += sum_r a_r (x_r - x_ref)`` over the rows, left to right, skipping zero weights."""
    width = rows.shape[1]
    for row in range(rows.shape[0]):
        weight = a[row]
        if weight == 0.0:
            continue
        for j in range(width):
            out[j] += weight * (rows[row, j] - reference[j])


@njit(cache=True)
def _first_distinct_level_rows(
    data, indices, indptr, codes, weights, n_levels, cap
):  # pragma: no cover
    """Each level's first ``cap`` distinct basis rows of positive weight, in row order.

    ``_distinct_level_rows`` on the rows as they are (no level sort, no copy of
    the basis): ``count[l]`` is the level's number of distinct rows, stopped at
    ``cap``, and ``first[l, :count[l]]`` their row indices.  A level that has
    its ``cap`` rows costs one comparison per further row.
    """
    width = max(cap, 1)
    count = np.zeros(n_levels, np.int64)
    first = np.full((n_levels, width), -1, np.int64)
    for r in range(len(codes)):
        if not weights[r] > 0.0:
            continue
        level = codes[r]
        held = count[level]
        if held >= cap:
            continue
        new = True
        for t in range(held):
            q = first[level, t]
            a0, a1, b0, b1 = indptr[r], indptr[r + 1], indptr[q], indptr[q + 1]
            if a1 - a0 != b1 - b0:
                continue
            same = True
            for u in range(a1 - a0):
                if indices[a0 + u] != indices[b0 + u] or data[a0 + u] != data[b0 + u]:
                    same = False
                    break
            if same:
                new = False
                break
        if new:
            first[level, held] = r
            count[level] = held + 1
    return count, first


@njit(cache=True)
def _first_distinct_level_bins(bins, codes, weights, n_levels, cap):  # pragma: no cover
    """``_first_distinct_level_rows`` for a discrete term, whose distinct rows are its bins."""
    width = max(cap, 1)
    count = np.zeros(n_levels, np.int64)
    first = np.full((n_levels, width), -1, np.int64)
    for r in range(len(codes)):
        if not weights[r] > 0.0:
            continue
        level = codes[r]
        held = count[level]
        if held >= cap:
            continue
        new = True
        for t in range(held):
            if bins[first[level, t]] == bins[r]:
                new = False
                break
        if new:
            first[level, held] = r
            count[level] = held + 1
    return count, first


@njit(cache=True)
def _distinct_level_rows(data, indices, indptr, starts, weights, cap):  # pragma: no cover
    """Each level's count of distinct basis rows of positive weight, stopped at ``cap``.

    ``data``/``indices``/``indptr`` are the CSR rows in level order (``starts``
    ``(K + 1,)``); two rows are the same when their stored entries are equal
    bit for bit.  A level keeps at most ``cap`` representatives, so the scan
    is ``O(n cap)``.
    """
    K = len(starts) - 1
    out = np.zeros(K, np.int64)
    reps = np.empty(max(cap, 1), np.int64)
    for level in range(K):
        count = 0
        for r in range(starts[level], starts[level + 1]):
            if not weights[r] > 0.0:
                continue
            new = True
            for t in range(count):
                q = reps[t]
                a0, a1, b0, b1 = indptr[r], indptr[r + 1], indptr[q], indptr[q + 1]
                if a1 - a0 != b1 - b0:
                    continue
                same = True
                for u in range(a1 - a0):
                    if indices[a0 + u] != indices[b0 + u] or data[a0 + u] != data[b0 + u]:
                        same = False
                        break
                if same:
                    new = False
                    break
            if new:
                reps[count] = r
                count += 1
                if count >= cap:
                    break
        out[level] = count
    return out
