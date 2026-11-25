"""The row pass's random-effect border blocks by the row expansion it used before pair sums.

A frozen copy of the previous implementation: every row is written once for
each level its leaf meets and the blocks come from scipy sparse products. It
remains test-only, as an independent parity oracle for
``moments._indicator_pairs``: ``install`` puts it in the pass's place.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from superglm.solvers._structured import moments


def _indicator_entries(indicators, mean, width, lo, leaves, counts):
    """A chunk's centred indicator rows ``x - m`` on the levels of each row's leaf.

    Returns ``(row, column, value)``: chunk row, column among the indicator
    columns and ``x_rj - mean[l, width + column]``.
    """
    parts = [(np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0))]
    leaf, offset = np.repeat(leaves, counts), 0
    for cells in indicators:
        per_row = cells.start[leaf + 1] - cells.start[leaf]
        row = np.repeat(np.arange(len(leaf)), per_row)
        first = cells.start[leaf] - np.cumsum(per_row) + per_row
        cell = np.repeat(first, per_row) + np.arange(len(row))
        column = offset + cells.level[cell]
        value = (cells.row_cell[lo + row] == cell) - mean[leaf[row], width + column]
        parts.append((row, column, value))
        offset += len(cells.columns)
    row, column, value = (np.concatenate(values) for values in zip(*parts, strict=True))
    return row, column, value


def _segment_sums(entries, weights, owner, segments, width):
    """``sum_{r in s} weights_r v_rj`` ``(segments, width)`` over sparse row entries ``v``."""
    row, column, value = entries
    key = owner[row] * width + column
    sums = np.bincount(key, weights=weights[row] * value, minlength=segments * width)
    return sums.reshape(segments, width)


def row_expanded_pairs(indicators, mean, width, lo, leaves, bounds, a, error, weighted):
    """``moments._indicator_pairs`` from the expanded rows, and the entries it wrote.

    The scatter rows are the old chunk product's, entry for entry; the pair
    statistics are the old per-segment sums read at each pair.
    """
    counts = np.diff(bounds)
    entries = _indicator_entries(indicators, mean, width, lo, leaves, counts)
    row, column, value = entries
    count = mean.shape[1] - width
    indicator = scipy.sparse.csr_array((value, (row, column)), shape=(len(a), count))
    weighted_indicator = scipy.sparse.csr_array((a[row] * value, (row, column)), indicator.shape)
    scatter = np.hstack(
        [indicator.T @ weighted, (indicator.T @ weighted_indicator).toarray()]
    ).reshape(count, mean.shape[1])
    row_owner = np.repeat(np.arange(len(leaves)), counts)
    segments = len(leaves)
    squares = np.bincount(
        row_owner[row] * count + column, weights=error[row] * value**2, minlength=segments * count
    ).reshape(segments, count)
    sums = [
        _segment_sums(entries, weights, row_owner, segments, count)
        for weights in (a, error, np.maximum(-a, 0.0))
    ]
    owners, columns, offset = [np.empty(0, np.intp)], [np.empty(0, np.intp)], 0
    for cells in indicators:
        per_leaf = cells.start[leaves + 1] - cells.start[leaves]
        owners.append(np.repeat(np.arange(segments), per_leaf))
        columns.append(offset + cells.level[cells.start[leaves[0]] : cells.start[leaves[-1] + 1]])
        offset += len(cells.columns)
    owner, pair_column = np.concatenate(owners), np.concatenate(columns)
    centred = np.column_stack([values[owner, pair_column] for values in sums])
    return scatter, owner, pair_column, centred, squares[owner, pair_column], len(row)


def install(monkeypatch) -> list[int]:
    """Run the row pass on the expanded rows; returns the list its entry counts are appended to."""
    written: list[int] = []

    def expanded(*args):
        *statistics, entries = row_expanded_pairs(*args)
        written.append(entries)
        return tuple(statistics)

    monkeypatch.setattr(moments, "_indicator_pairs", expanded)
    return written
