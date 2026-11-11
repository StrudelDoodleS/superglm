"""Private discretization helpers for group-matrix construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def discretize_column(
    x: NDArray, n_bins: int = 256, knots: NDArray | None = None
) -> tuple[NDArray, NDArray]:
    """Compress a continuous variable to at most ``n_bins`` support points.

    A column with at most ``n_bins`` distinct values keeps them exactly.
    Otherwise it is cut into ``n_bins`` equal-width bins between its minimum
    and maximum, and each row is represented by the midpoint of its bin.

    ``knots`` are the knots of the spline the column feeds; those strictly
    inside ``[min, max]`` cut the column into spans, the pieces of the curve.
    With ``m`` spans, each span that holds a row needs
    ``b_min = max(1, n_bins // (2 * m))`` bins, or the fit cannot see the
    curve inside it. When every such span's width share of the equal-width
    bins reaches ``b_min``, the equal-width bins are returned unchanged.
    Otherwise the bins follow the knots: every row-holding span gets
    ``b_min`` equal-width cells plus a share of the remaining cells in
    proportion to its width (largest remainder), spans without rows get none,
    and the total stays within ``n_bins``. A row on a knot belongs to the span
    on its right and the maximum to the last cell. When more spans hold rows
    than there are bins, no grid can resolve them all and the equal-width bins
    are kept.

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
    if knots is not None:
        aligned = _knot_aligned_bins(x, unique_vals, lo, hi, n_bins, knots)
        if aligned is not None:
            return aligned
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, bin_idx


def _knot_aligned_bins(
    x: NDArray,
    unique_vals: NDArray,
    lo: float,
    hi: float,
    n_bins: int,
    knots: NDArray,
) -> tuple[NDArray, NDArray] | None:
    """The knot-aligned grid of ``discretize_column``, or None to keep equal-width bins."""
    k = np.unique(np.asarray(knots, dtype=np.float64).ravel())
    inner = k[(k > lo) & (k < hi)]
    if inner.size == 0:
        return None
    span_edges = np.concatenate([[lo], inner, [hi]])
    widths = np.diff(span_edges)
    n_spans = widths.size
    b_min = max(1, n_bins // (2 * n_spans))
    # A row on a knot belongs to the span on its right; the maximum to the last.
    row_spans = np.minimum(np.searchsorted(inner, unique_vals, side="right"), n_spans - 1)
    occupied = np.zeros(n_spans, dtype=bool)
    occupied[row_spans] = True
    if np.all(n_bins * widths[occupied] / (hi - lo) >= b_min):
        return None
    n_occupied = int(occupied.sum())
    if b_min * n_occupied > n_bins:
        return None

    cells = np.zeros(n_spans, dtype=np.intp)
    remaining = n_bins - b_min * n_occupied
    share = remaining * widths[occupied] / widths[occupied].sum()
    extra = np.floor(share).astype(np.intp)
    leftover = remaining - int(extra.sum())
    if leftover > 0:
        # Largest remainder; ties go to the leftmost span.
        order = np.argsort(-(share - extra), kind="stable")
        extra[order[:leftover]] += 1
    cells[occupied] = b_min + extra

    left_edges = []
    right_edges = []
    for j in np.flatnonzero(occupied):
        cell_edges = np.linspace(span_edges[j], span_edges[j + 1], cells[j] + 1)
        left_edges.append(cell_edges[:-1])
        right_edges.append(cell_edges[1:])
    lefts = np.concatenate(left_edges)
    rights = np.concatenate(right_edges)
    bin_idx = np.clip(np.searchsorted(lefts, x, side="right") - 1, 0, lefts.size - 1)
    return 0.5 * (lefts + rights), bin_idx.astype(np.intp)


__all__ = ["discretize_column"]
