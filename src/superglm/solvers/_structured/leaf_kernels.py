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
import numpy.random
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


# The pair kernels of a random-effect border block.  Within a leaf, the rows
# that take one level of the block (a pair) share their centred indicator row
# ``e_t - m``, so ``sum_r a_r (x_rj - m_j) y_r = (1 - m_j) Y_j - m_j R_j`` for any
# rows ``y``, with ``Y_t`` the pair sums of ``a y`` and ``R_j`` the sum over the
# leaf's other rows.  That is the row sum with the weights of a pair added
# first: every product keeps its factors ``1 - m`` and ``m``, so an entry stays
# within the row sum's bound ``gamma sum_r |a_r| |x_rj - m_j| |y_r|`` and is an
# exact zero wherever every weighted row's factor is.  ``R_j`` is summed pair by
# pair, never formed as the leaf total less ``Y_j``.  A sum of weights meets one
# of its two factors before the other, as a row's ``(a d_j) d_k`` does: the
# product of two means underflows where the entry itself does not.
@njit(cache=True)
def _pair_sums(own, bounds, spare, a, error, weighted, sums):  # pragma: no cover - compiled
    """Add each row of a chunk to its pair's sums: one entry a row.

    ``own[r]`` is row ``r``'s pair among the chunk's pairs, or negative for a
    row with no level, which goes to its segment's spare row
    ``spare[segment]``.  That row of ``sums`` ``(pairs and spare rows, width +
    3)`` gains ``a x`` (``weighted``'s row), then ``a``, ``error`` and
    ``max(-a, 0)``.  A row of zero weight adds its ``error`` alone.
    """
    width = weighted.shape[1]
    for segment in range(len(bounds) - 1):
        for row in range(bounds[segment], bounds[segment + 1]):
            pair = own[row]
            if pair < 0:
                pair = spare[segment]
            weight = a[row]
            sums[pair, width + 1] += error[row]
            if weight == 0.0:
                continue
            for j in range(width):
                sums[pair, j] += weighted[row, j]
            sums[pair, width] += weight
            if weight < 0.0:
                sums[pair, width + 2] -= weight


@njit(cache=True)
def _other_sums(sums, spare, count, width, rest, carry):  # pragma: no cover - compiled
    """``rest[t] = spare + sum_{s != t} sums[s]`` over the first ``count`` rows and ``width`` columns.

    The rows after ``t`` are summed from the last one back and the rows before
    it, after ``spare``, from the first one on: each entry sums the other rows'
    terms and nothing else.  ``carry`` ``(width,)`` is scratch.
    """
    for j in range(width):
        carry[j] = 0.0
    for t in range(count - 1, -1, -1):
        for j in range(width):
            rest[t, j] = carry[j]
            carry[j] += sums[t, j]
    for j in range(width):
        carry[j] = spare[j]
    for t in range(count):
        for j in range(width):
            rest[t, j] += carry[j]
            carry[j] += sums[t, j]


@njit(cache=True)
def _center_pairs(
    sums, pair_bounds, spare, column, mean, deviation, mass, out
):  # pragma: no cover - compiled
    """Centre a chunk's pair sums on their leaf means: ``(1 - m_t) sums[t] - m_t R_t``.

    Segment ``s`` holds the pairs ``pair_bounds[s]:pair_bounds[s + 1]`` and the
    spare row ``spare[s]`` of ``sums``; ``R_t`` sums its other rows
    (``_other_sums``), in scratch one segment long.  The ``a x`` columns are
    added to row ``column[t]`` of ``out``, the last three kept in
    ``deviation[t]``, and ``mass[t] = E_t (1 - m_t)^2 + R_t m_t^2`` on the
    ``error`` column.
    """
    columns = sums.shape[1]
    width = columns - 3
    longest = 0
    for segment in range(len(pair_bounds) - 1):
        longest = max(longest, pair_bounds[segment + 1] - pair_bounds[segment])
    rest, carry = np.empty((longest, columns)), np.empty(columns)
    for segment in range(len(pair_bounds) - 1):
        lo, hi = pair_bounds[segment], pair_bounds[segment + 1]
        _other_sums(sums[lo:hi], sums[spare[segment]], hi - lo, columns, rest, carry)
        for t in range(lo, hi):
            m = mean[t]
            u = 1.0 - m
            other = rest[t - lo]
            mass[t] = sums[t, width + 1] * u * u + other[width + 1] * m * m
            for j in range(width):
                out[column[t], j] += u * sums[t, j] - m * other[j]
            for j in range(3):
                deviation[t, j] = u * sums[t, width + j] - m * other[width + j]


@njit(cache=True)
def _pair_block(
    pair_bounds, spare, column, mean, weight, offset, out
):  # pragma: no cover - compiled
    """Add one indicator block's own ``sum_r a_r (x_r - m)(x_r - m)'`` to ``out``, from its pair weights.

    With ``u = 1 - m`` and ``w`` the pair sums of ``a`` (``weight``, a
    segment's spare row at ``spare[segment]``), a leaf adds ``w_j u_j^2 + R_j
    m_j^2`` at ``(j, j)`` and ``R_jk m_j m_k - (w_j u_j m_k + w_k m_j u_k)`` at
    ``(j, k)`` and ``(k, j)``: ``R_j`` and ``R_jk`` the weight of the leaf's
    rows outside pair ``j``, and outside pairs ``j`` and ``k``.  Pair ``t`` is
    row ``column[t]`` of ``out`` and column ``offset + column[t]``.
    """
    pairs = len(column)
    after = np.empty(pairs)
    for segment in range(len(pair_bounds) - 1):
        lo, hi = pair_bounds[segment], pair_bounds[segment + 1]
        running = 0.0
        for t in range(hi - 1, lo - 1, -1):
            after[t - lo] = running
            running += weight[t]
        before = weight[spare[segment]]
        for j in range(lo, hi):
            mj, wj = mean[j], weight[j]
            uj = 1.0 - mj
            out[column[j], offset + column[j]] += wj * uj * uj + (before + after[j - lo]) * mj * mj
            between = before
            for k in range(j + 1, hi):
                mk, wk = mean[k], weight[k]
                value = (between + after[k - lo]) * mj * mk - (wj * uj * mk + wk * mj * (1.0 - mk))
                out[column[j], offset + column[k]] += value
                out[column[k], offset + column[j]] += value
                between += wk
            before += wj


@njit(cache=True)
def _crossed_blocks(
    bounds, a, own_x, bounds_x, column_x, mean_x, own_y, bounds_y, column_y, mean_y, offset, out
):  # pragma: no cover - compiled
    """Add two indicator blocks' ``sum_r a_r (x_r - m)(y_r - m)'`` to ``out``, both mirrored halves.

    Per leaf, ``joint[k, t]`` sums ``a`` over the rows in pair ``t`` of ``x``
    and pair ``k`` of ``y``, one entry a row (the last index of either holds
    its rows with no level).  Centring along ``y`` gives ``sum_{r in t} a_r
    (y_rk - m_k)``, the pair sums of ``x``, and centring those along ``x`` the
    block: the row sum regrouped twice (``_other_sums``).  The three tables
    share flat scratch the size of the largest leaf's ``joint``, each shaped
    as its sweep reads it.
    """
    segments, size, side = len(bounds) - 1, 1, 1
    for segment in range(segments):
        count_x = 1 + bounds_x[segment + 1] - bounds_x[segment]
        count_y = 1 + bounds_y[segment + 1] - bounds_y[segment]
        size, side = max(size, count_x * count_y), max(side, count_x, count_y)
    joint_flat, sums_flat, rest_flat = np.empty(size), np.empty(size), np.empty(size)
    carry = np.empty(side)
    for segment in range(segments):
        x0, y0 = bounds_x[segment], bounds_y[segment]
        count_x, count_y = bounds_x[segment + 1] - x0, bounds_y[segment + 1] - y0
        joint = joint_flat[: (count_y + 1) * (count_x + 1)].reshape(count_y + 1, count_x + 1)
        for k in range(count_y + 1):
            for t in range(count_x + 1):
                joint[k, t] = 0.0
        for row in range(bounds[segment], bounds[segment + 1]):
            t, k = own_x[row], own_y[row]
            t = t - x0 if t >= 0 else count_x
            k = k - y0 if k >= 0 else count_y
            joint[k, t] += a[row]
        rest = rest_flat[: count_y * (count_x + 1)].reshape(count_y, count_x + 1)
        _other_sums(joint, joint[count_y], count_y, count_x + 1, rest, carry)
        sums = sums_flat[: (count_x + 1) * count_y].reshape(count_x + 1, count_y)
        for k in range(count_y):
            mk = mean_y[y0 + k]
            for t in range(count_x + 1):
                sums[t, k] = (1.0 - mk) * joint[k, t] - mk * rest[k, t]
        rest = rest_flat[: count_x * count_y].reshape(count_x, count_y)
        _other_sums(sums, sums[count_x], count_x, count_y, rest, carry)
        for t in range(count_x):
            mt, row_x = mean_x[x0 + t], column_x[x0 + t]
            for k in range(count_y):
                value = (1.0 - mt) * sums[t, k] - mt * rest[t, k]
                row_y = column_y[y0 + k]
                out[row_x, offset + row_y] += value
                out[row_y, offset + row_x] += value


@njit(cache=True)
def _absent_levels(absent, shift, piece, width, out):  # pragma: no cover - compiled
    """Add to ``out`` a piece's scatter on the indicator columns none of its rows takes.

    Every row's centred value on such a column ``i`` (``absent``) is
    ``shift[i] = -m_i``, so its row of ``sum_r a_r (x_r - m)(x_r - m)'`` is
    ``shift[i]`` times ``piece``, the piece's ``sum_r a_r (x_r - m)`` over all
    columns, and its column the mirror.  ``shift`` is zero off ``absent``.
    """
    for i in absent:
        for c in range(len(piece)):
            out[i, c] += shift[i] * piece[c]
    for t in range(len(shift)):
        value = piece[width + t]
        if shift[t] == 0.0 and value != 0.0:
            for i in absent:
                out[t, width + i] += value * shift[i]


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


# Entry-by-entry updates of a 2-D block for the FactorSmooth kernels: numba
# compiles array expressions and sliced assignments into costly broadcasting
# loops, and these give the same operations in the same order, bit for bit.
@njit(cache=True)
def _assign(target, source):  # pragma: no cover - compiled
    """``target[:] = source``."""
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            target[i, j] = source[i, j]


@njit(cache=True)
def _add(target, source):  # pragma: no cover - compiled
    """``target += source``."""
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            target[i, j] += source[i, j]


@njit(cache=True)
def _add_scaled(target, factor, source):  # pragma: no cover - compiled
    """``target += factor * source``."""
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            target[i, j] += factor * source[i, j]


@njit(cache=True)
def _absolute(A):  # pragma: no cover - compiled
    """``np.abs(A)`` of a 2-D block, without compiling the ufunc."""
    out = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            out[i, j] = abs(A[i, j])
    return out


def _warmup_leaf_kernels() -> None:
    """Compile the kernels a RandomEffect leaf pass calls (no rows).

    A RandomEffect large enough for the structured engine (one with a few
    hundred levels, nested or not) reaches the first four, exact or binned,
    and the pair kernels when another RandomEffect is crossed with it; the
    FactorSmooth kernels compile in ``block_leaves._warmup_factor_smooth``.
    """
    f1, i64 = np.zeros(0), np.zeros(0, dtype=np.int64)
    f2 = np.zeros((0, 0))
    one = np.zeros(1, dtype=np.int64)
    center = np.zeros(0)
    center.flags.writeable = False  # ``moments._leaf_rows`` passes it read-only
    _coded_rows(f2, np.zeros((0, 0), dtype=np.int32), f1, i64, i64, i64, i64, i64, center)
    _segment_means(f2, f1, one, i64, f1, f2)
    _center_rows(f2, f2, f2, f1, f1, one, f2, f1, f2, f2)
    _shifted_sums(f2, f1, f1, f1)
    sums = np.zeros((1, 3))
    _pair_sums(i64, one, i64, f1, f1, f2, sums)
    _center_pairs(sums, one, i64, i64, f1, np.zeros((0, 3)), f1, f2)
    _pair_block(one, i64, i64, f1, np.zeros(1), 0, f2)
    _crossed_blocks(one, f1, i64, one, i64, f1, i64, one, i64, f1, 0, f2)
    _absent_levels(i64, f1, f1, 0, f2)
