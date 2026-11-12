"""Private discretization helpers for group-matrix construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# A bin holding more than this many times its fair share (1 / n_bins) of the
# rows is crowded: the rows inside it would all be read at one value.
_CROWDED_SHARE = 8


def discretize_column(
    x: NDArray,
    n_bins: int = 256,
    knots: NDArray | None = None,
    weights: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Compress a continuous variable to at most ``n_bins`` support points.

    A column with at most ``n_bins`` distinct values keeps them exactly.
    Without ``knots``, a longer column is cut into ``n_bins`` equal-width bins
    between its minimum and maximum, and each row is represented by the
    midpoint of its bin.

    ``knots`` are the knots of the spline the column feeds; those strictly
    inside ``[min, max]`` cut the column into spans, the pieces of the curve.
    The equal-width bins are kept in three cases. First, when they already
    follow those pieces and no bin is crowded: with ``m`` spans, every span
    that holds a row has a width share of at least ``b_min = max(1, n_bins //
    (2 * m))`` bins, and no bin holds more than ``8 / n_bins`` of the rows.
    Second, when the budget is below four: the knot half of the grid below
    (``n_w``) would be one cell, which cannot separate the minimum from the
    maximum. Third, when at least ``n_bins`` spans hold rows: no grid within
    the budget gives each of them a cell.

    Otherwise the bins follow the knots and the data, within the same budget.
    ``n_w = n_bins // 2`` cells come from the knots: every span that holds a
    row gets ``b_min`` equal-width cells (``b_min`` computed from ``n_w``)
    plus a share of the rest in proportion to its width (largest remainder),
    and spans without rows get none. When more spans hold rows than that,
    ``n_w`` is their number and each gets exactly one cell. The other ``n_q
    = n_bins - n_w`` come from the rows: the lower ``j / n_q`` quantiles, ``j
    = 1 .. n_q - 1``, are observed values and become cell edges too. A row on
    an edge belongs to the cell on its right and the maximum to the last
    cell, so there are at most ``n_w + n_q - 1 = n_bins - 1`` cells. Each
    cell is represented by the mean of its rows, a cell without rows by its
    midpoint.

    So on this grid each cell lies inside one piece of the curve, holds at
    most ``1 / n_q`` of the rows unless it is a single pile of equal values,
    and is read where its rows actually are: a pile is read at its own value.
    The support points are sorted, and each lies within its own rows' range,
    so every row is read inside its own span. The equal-width bins promise
    none of this.

    ``weights`` are the rows' frequency weights (nonnegative; ``None`` counts
    every row once). Wherever a rule here counts rows (the crowding test, the
    quantiles, the means), a row counts by its weight, so a column with
    positive integer weights and its literally replicated twin get the same
    bins; a cell whose rows all weigh zero is read at its midpoint, clipped
    into their range. Which spans and cells hold rows depends only on the
    values present, zero-weight rows included, and the equal-width bins
    themselves never read the weights. A caller that should not let
    zero-weight rows shape the support leaves them out (the main-effect site
    of a binned fit does; the interaction sites bin every row).

    The width shares are formed on the column scaled by a power of two to
    magnitude at most one, which is exact and changes no share of a column
    whose widths are finite, so a column near the float64 limit allocates its
    cells without overflow.

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
    bin_of_value = np.clip(np.searchsorted(edges, unique_vals, side="right") - 1, 0, n_bins - 1)
    bin_idx = bin_of_value[inverse]
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    if knots is None or n_bins < 4:
        return bin_centers, bin_idx

    mass = _value_mass(inverse, unique_vals.size, weights, x.size)
    total = float(mass.sum())
    span_edges, occupied = _row_spans(unique_vals, lo, hi, knots)
    scaled = _unit_scaled(span_edges)
    widths = np.diff(scaled)
    b_min = _min_cells_per_span(n_bins, occupied.size)
    resolves = np.all(n_bins * widths[occupied] / (scaled[-1] - scaled[0]) >= b_min)
    bin_mass = np.bincount(bin_of_value, weights=mass, minlength=n_bins)
    crowded = bin_mass.max() * n_bins > _CROWDED_SHARE * total
    # With a span holding rows for every bin, no grid gives each one a cell.
    if (resolves and not crowded) or int(occupied.sum()) >= n_bins:
        return bin_centers, bin_idx
    return _adaptive_bins(mass, total, unique_vals, inverse, span_edges, occupied, n_bins)


def _value_mass(inverse: NDArray, n_values: int, weights: NDArray | None, n_rows: int) -> NDArray:
    """Each distinct value's mass: its row count, or the sum of its rows' weights.

    Formed per distinct value, so a weighted column and its replicated twin
    carry the same float for every value and every later step agrees.
    """
    if weights is None:
        return np.bincount(inverse, minlength=n_values).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (n_rows,):
        raise ValueError(f"weights must have length {n_rows}, got {w.size}")
    return np.bincount(inverse, weights=w, minlength=n_values)


def _lower_quantile_values(mass: NDArray, total: float, unique_vals: NDArray, n_q: int) -> NDArray:
    """The smallest values whose cumulative mass reaches ``j * total / n_q``, ``j = 1 .. n_q - 1``.

    Integer masses (row counts, integer frequency weights) whose total is
    below ``2**53`` are compared exactly: the float partial sums are integers
    below ``2**53`` and so exact, and ``cumulative >= j * total / n_q`` is
    ``cumulative >= ceil(j * total / n_q)``, formed in int64 as
    ``j * q + ceil(j * r / n_q)`` with ``total = q * n_q + r`` so nothing
    overflows. Replicated rows then pick the same values as their weights.
    Fractional masses compare in float64; a target within rounding of a
    partial sum may land on either neighbour, which only moves one edge.
    """
    j = np.arange(1, n_q, dtype=np.int64)
    integral = total < 2.0**53 and bool(np.all(mass == np.floor(mass)))
    if integral:
        cumulative = np.cumsum(mass.astype(np.int64))
        q, r = divmod(int(total), n_q)
        target = j * q + (j * r + n_q - 1) // n_q
    else:
        cumulative = np.cumsum(mass)
        target = j * total / n_q
    position = np.searchsorted(cumulative, target, side="left")
    return unique_vals[np.minimum(position, unique_vals.size - 1)]


def _unit_scaled(edges: NDArray) -> NDArray:
    """``edges`` times the power of two that brings their largest magnitude into ``[0.5, 1)``.

    Exact unless an edge is far enough below the largest to become subnormal,
    which moves it by at most ``2**-1075`` against a scaled range of at least
    one half. Differences of the scaled edges are then at most two, so width
    shares formed from them cannot overflow, and they are the unscaled shares
    bit for bit wherever those were finite.
    """
    _, exponent = np.frexp(np.max(np.abs(edges)))
    return np.ldexp(edges, -int(exponent))


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
    """Left edges of ``n_cells`` cells that follow the knots (``discretize_column``).

    ``n_cells`` is at least the number of spans holding rows, so each of them
    gets ``b_min >= 1`` cells.
    """
    widths = np.diff(_unit_scaled(span_edges))
    b_min = _min_cells_per_span(n_cells, occupied.size)
    n_occupied = int(occupied.sum())
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
    mass: NDArray,
    total: float,
    unique_vals: NDArray,
    inverse: NDArray,
    span_edges: NDArray,
    occupied: NDArray,
    n_bins: int,
) -> tuple[NDArray, NDArray]:
    """Cells on the knots and the row quantiles, each read at its rows' mean.

    ``mass`` is each distinct value's row mass (``_value_mass``) and ``total``
    its sum.
    """
    n_w = n_bins // 2
    n_occupied = int(occupied.sum())
    if _min_cells_per_span(n_w, occupied.size) * n_occupied > n_w:
        # One cell for each span that holds rows; the rows get the rest.
        n_w = n_occupied
    n_q = n_bins - n_w
    lo, hi = float(span_edges[0]), float(span_edges[-1])
    width_edges = _knot_aligned_left_edges(span_edges, occupied, n_w)

    # Lower (inverted-CDF) quantiles of the row mass, so every edge is an observed value.
    quantile_edges = _lower_quantile_values(mass, total, unique_vals, n_q)
    left = np.unique(np.concatenate([width_edges, quantile_edges[quantile_edges > lo]]))

    # A row on an edge belongs to the cell on its right; the maximum to the last.
    cell_of_value = np.searchsorted(left, unique_vals, side="right") - 1
    n_cells = left.size
    held = np.bincount(cell_of_value, minlength=n_cells) > 0
    cell_mass = np.bincount(cell_of_value, weights=mass, minlength=n_cells)
    # The mean as a convex combination of the cell's values, so no partial sum
    # can exceed the largest value; then clipped into the range of the cell's
    # values, which rounding could otherwise leave. A cell whose rows all have
    # zero weight is read at its midpoint, clipped the same way.
    value_cell_mass = cell_mass[cell_of_value]
    share_of_cell = np.divide(
        mass, value_cell_mass, out=np.zeros_like(mass), where=value_cell_mass > 0.0
    )
    mean = np.bincount(cell_of_value, weights=unique_vals * share_of_cell, minlength=n_cells)
    cells = np.arange(n_cells)
    first = unique_vals[
        np.minimum(np.searchsorted(cell_of_value, cells, side="left"), unique_vals.size - 1)
    ]
    last = unique_vals[np.searchsorted(cell_of_value, cells, side="right") - 1]
    right = np.append(left[1:], hi)
    # Halved before the sum, which cannot then overflow; halving is exact.
    midpoint = 0.5 * left + 0.5 * right
    read_at = np.where(cell_mass > 0.0, mean, midpoint)
    support = np.where(held, np.clip(read_at, first, last), midpoint)
    return support, cell_of_value[inverse].astype(np.intp)


__all__ = ["discretize_column"]
