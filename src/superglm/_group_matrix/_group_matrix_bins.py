"""Private discretization helpers for group-matrix construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# A bin holding more than this many times its fair share (1 / n_bins) of the
# rows is crowded: the rows inside it would all be read at one value.
_CROWDED_SHARE = 8


def discretize_column(
    x: NDArray, n_bins: int = 256, knots: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """Compress a continuous variable to at most ``n_bins`` support points.

    A column with at most ``n_bins`` distinct values keeps them exactly.
    Without ``knots``, a longer column is cut into ``n_bins`` equal-width bins
    between its minimum and maximum, and each row is represented by the
    midpoint of its bin.

    ``knots`` are the knots of the spline the column feeds; those strictly
    inside ``[min, max]`` cut the column into spans, the pieces of the curve.
    The equal-width bins are kept when they already follow those pieces and
    no bin is crowded: with ``m`` spans, every span that holds a row has a
    width share of at least ``b_min = max(1, n_bins // (2 * m))`` bins, and no
    bin holds more than ``8 / n_bins`` of the rows.

    Otherwise the bins follow the knots and the data, within the same budget.
    ``n_w = n_bins // 2`` cells come from the knots: every span that holds a
    row gets ``b_min`` equal-width cells (``b_min`` computed from ``n_w``)
    plus a share of the rest in proportion to its width (largest remainder),
    spans without rows get none, and ``n_w`` equal-width cells are used when
    more spans hold rows than there are cells. The other ``n_q = n_bins -
    n_w`` come from the rows: the lower ``j / n_q`` quantiles, ``j = 1 ..
    n_q - 1``, are observed values and become cell edges too. A row on an
    edge belongs to the cell on its right and the maximum to the last cell,
    so there are at most ``n_w + n_q - 1`` cells. Each cell is represented by
    the mean of its rows, a cell without rows by its midpoint.

    So each cell lies inside one piece of the curve, holds at most ``1 / n_q``
    of the rows unless it is a single pile of equal values, and is read where
    its rows actually are: a pile is read at its own value. The support points
    are sorted, and each lies within its own rows' range, so none crosses a
    knot away from its rows.

    Returns the support points (sorted) and each row's index into them.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    unique_vals, inverse = np.unique(x, return_inverse=True)
    if len(unique_vals) <= n_bins:
        return unique_vals.astype(np.float64), inverse.astype(np.intp)

    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return np.array([lo]), np.zeros(len(x), dtype=np.intp)
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    if knots is None:
        return bin_centers, bin_idx

    span_edges, occupied = _row_spans(unique_vals, lo, hi, knots)
    widths = np.diff(span_edges)
    b_min = _min_cells_per_span(n_bins, occupied.size)
    resolves = np.all(n_bins * widths[occupied] / (hi - lo) >= b_min)
    crowded = np.bincount(bin_idx, minlength=n_bins).max() * n_bins > _CROWDED_SHARE * x.size
    if resolves and not crowded:
        return bin_centers, bin_idx
    return _adaptive_bins(x.size, unique_vals, inverse, span_edges, occupied, n_bins)


def _min_cells_per_span(n_cells: int, n_spans: int) -> int:
    return max(1, n_cells // (2 * n_spans))


def _row_spans(
    unique_vals: NDArray, lo: float, hi: float, knots: NDArray
) -> tuple[NDArray, NDArray]:
    """The span edges the knots inside ``(lo, hi)`` make, and which spans hold rows.

    A row on a knot belongs to the span on its right; the maximum to the last.
    """
    k = np.unique(np.asarray(knots, dtype=np.float64).ravel())
    inner = k[(k > lo) & (k < hi)]
    occupied = np.zeros(inner.size + 1, dtype=bool)
    occupied[np.searchsorted(inner, unique_vals, side="right")] = True
    return np.concatenate([[lo], inner, [hi]]), occupied


def _knot_aligned_left_edges(span_edges: NDArray, occupied: NDArray, n_cells: int) -> NDArray:
    """Left edges of ``n_cells`` cells that follow the knots (``discretize_column``)."""
    widths = np.diff(span_edges)
    b_min = _min_cells_per_span(n_cells, occupied.size)
    n_occupied = int(occupied.sum())
    if b_min * n_occupied > n_cells:
        return np.linspace(span_edges[0], span_edges[-1], n_cells + 1)[:-1]

    remaining = n_cells - b_min * n_occupied
    share = remaining * widths[occupied] / widths[occupied].sum()
    extra = np.floor(share).astype(np.intp)
    leftover = remaining - int(extra.sum())
    if leftover > 0:
        # Largest remainder; ties go to the leftmost span.
        order = np.argsort(-(share - extra), kind="stable")
        extra[order[:leftover]] += 1
    spans = np.flatnonzero(occupied)
    return np.concatenate(
        [
            np.linspace(span_edges[j], span_edges[j + 1], b_min + n_extra + 1)[:-1]
            for j, n_extra in zip(spans, extra, strict=True)
        ]
    )


def _adaptive_bins(
    n_rows: int,
    unique_vals: NDArray,
    inverse: NDArray,
    span_edges: NDArray,
    occupied: NDArray,
    n_bins: int,
) -> tuple[NDArray, NDArray]:
    """Cells on the knots and the row quantiles, each read at its rows' mean."""
    n_w = n_bins // 2
    n_q = n_bins - n_w
    lo, hi = float(span_edges[0]), float(span_edges[-1])
    width_edges = _knot_aligned_left_edges(span_edges, occupied, max(n_w, 1))

    # Lower (inverted-CDF) quantiles: the smallest value with at least
    # ceil(j * n / n_q) rows at or below it, so every edge is an observed value.
    counts = np.bincount(inverse, minlength=unique_vals.size)
    j = np.arange(1, n_q, dtype=np.int64)
    rank = (j * n_rows + n_q - 1) // n_q
    quantile_edges = unique_vals[np.searchsorted(np.cumsum(counts), rank, side="left")]
    left = np.unique(np.concatenate([width_edges, quantile_edges[quantile_edges > lo]]))

    # A row on an edge belongs to the cell on its right; the maximum to the last.
    cell_of_value = np.searchsorted(left, unique_vals, side="right") - 1
    n_cells = left.size
    rows = np.bincount(cell_of_value, weights=counts, minlength=n_cells)
    held = rows > 0
    # The mean as a convex combination of the cell's values, so no partial sum
    # can exceed the largest value; then clipped into the range of the cell's
    # values, which rounding could otherwise leave.
    share_of_cell = counts / rows[cell_of_value]
    mean = np.bincount(cell_of_value, weights=unique_vals * share_of_cell, minlength=n_cells)
    cells = np.arange(n_cells)
    first = unique_vals[
        np.minimum(np.searchsorted(cell_of_value, cells, side="left"), unique_vals.size - 1)
    ]
    last = unique_vals[np.searchsorted(cell_of_value, cells, side="right") - 1]
    right = np.append(left[1:], hi)
    support = np.where(held, np.clip(mean, first, last), 0.5 * (left + right))
    return support, cell_of_value[inverse].astype(np.intp)


__all__ = ["discretize_column"]
