"""Polynomial ranges on a spline: edge knots, pinning rows and their null space.

A range pins the spline to a polynomial of ``degree`` on ``[lo, hi]``. Its
edges become knots: repeated ``spline_degree`` times for a kink, which leaves
the curve only C0 there, or once, which keeps the spline's own continuity (a
knot of multiplicity m leaves C^(spline_degree - m): the Curry-Schoenberg
theorem; de Boor, *A Practical Guide to Splines*). The spline's other knots
inside the range are dropped, so the range is ONE polynomial piece, and
pinning that piece to degree d means its (d+1)-th derivative -- a polynomial of
degree ``spline_degree - d - 1`` -- vanishes: ``spline_degree - d`` rows at
distinct points of the piece. The rows are absorbed as ``beta = Z theta`` with
Z from the QR of C' (Wood 2017, *Generalized Additive Models*, section 1.8.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

JOINS = ("kink", "smooth")
SHAPE_NAMES = ("Flat", "Line", "Quadratic", "Cubic")


@dataclass(frozen=True)
class PolynomialRange:
    """Pin a spline to a polynomial of ``degree`` (0-3) on ``[lo, hi]``.

    ``join="kink"`` keeps the curve continuous at the edges and lets its slope
    change there; ``"smooth"`` keeps the spline's own continuity. On an
    ordered term ``lo`` and ``hi`` may be band names.
    """

    lo: float | str
    hi: float | str
    degree: int
    join: str = "kink"

    def __post_init__(self) -> None:
        if isinstance(self.degree, bool) or not isinstance(self.degree, (int, np.integer)):
            raise ValueError(f"PolynomialRange degree must be an integer, got {self.degree!r}")
        if not 0 <= int(self.degree) <= 3:
            raise ValueError(f"PolynomialRange degree must be 0-3, got {self.degree}")
        if self.join not in JOINS:
            raise ValueError(f"PolynomialRange join must be one of {JOINS}, got {self.join!r}")

    @property
    def label(self) -> str:
        return SHAPE_NAMES[int(self.degree)]


def validate_ranges(
    ranges: Sequence[PolynomialRange], degree: int, lo: float, hi: float
) -> tuple[PolynomialRange, ...]:
    """Return the ranges with float edges, sorted by ``lo``, refusing anything ill-posed."""
    numeric = (replace(r, lo=float(r.lo), hi=float(r.hi)) for r in ranges)
    ordered = tuple(sorted(numeric, key=lambda r: r.lo))
    for r in ordered:
        if not float(r.lo) < float(r.hi):
            raise ValueError(f"PolynomialRange lo must be below hi, got [{r.lo}, {r.hi}]")
        if float(r.lo) < lo or float(r.hi) > hi:
            raise ValueError(
                f"PolynomialRange [{r.lo}, {r.hi}] must lie inside the fitted range [{lo}, {hi}]"
            )
        if int(r.degree) > degree:
            raise ValueError(
                f"PolynomialRange degree {r.degree} exceeds the spline degree {degree}"
            )
    for left, right in zip(ordered[:-1], ordered[1:]):
        if float(right.lo) < float(left.hi):
            raise ValueError(
                f"PolynomialRanges [{left.lo}, {left.hi}] and [{right.lo}, {right.hi}] overlap"
            )
        if float(right.lo) == float(left.hi) and "smooth" in (left.join, right.join):
            raise ValueError("Adjacent PolynomialRanges must meet at a kink")
    return ordered


def merged_interior_knots(
    base: NDArray, ranges: Sequence[PolynomialRange], degree: int, lo: float, hi: float
) -> NDArray:
    """Base interior knots outside every range, plus each range's edge knots.

    A base knot within ``1e-9 * (hi - lo)`` of a closed range is dropped: it
    would otherwise leave a sliver interval beside the edge. An edge on ``lo``
    or ``hi`` is already a boundary knot and is not added; an edge two ranges
    share takes the larger multiplicity.
    """
    base = np.asarray(base, dtype=np.float64)
    tolerance = 1e-9 * (hi - lo)
    lows = np.array([float(r.lo) for r in ranges], dtype=np.float64)
    highs = np.array([float(r.hi) for r in ranges], dtype=np.float64)
    near = (base[:, None] >= lows - tolerance) & (base[:, None] <= highs + tolerance)
    copies = np.array([degree if r.join == "kink" else 1 for r in ranges], dtype=np.intp)
    edges = np.concatenate([lows, highs])
    interior = (edges > lo) & (edges < hi)
    unique_edges, which = np.unique(edges[interior], return_inverse=True)
    multiplicity = np.zeros(unique_edges.size, dtype=np.intp)
    np.maximum.at(multiplicity, which, np.concatenate([copies, copies])[interior])
    kept = base[~near.any(axis=1)]
    return np.sort(np.concatenate([kept, np.repeat(unique_edges, multiplicity)]))


def pinned_intervals(
    ranges: Sequence[PolynomialRange], lo: float, hi: float
) -> list[tuple[float, float]]:
    """The intervals each range pins, open past the fitted boundary at an end.

    A range touching ``lo`` or ``hi`` pins the end piece, whose extrapolation
    is what the spline takes beyond the boundary, so it pins that too.
    """
    return [(-np.inf if r.lo <= lo else r.lo, np.inf if r.hi >= hi else r.hi) for r in ranges]


def pinning_rows(knots: NDArray, degree: int, ranges: Sequence[PolynomialRange]) -> NDArray:
    """Rows C with ``C @ beta = 0`` iff each range's piece has at most its degree.

    ``knots`` must come from :func:`merged_interior_knots`, so that no knot
    lies strictly inside a range.
    """
    n_basis = len(knots) - degree - 1
    blocks = [np.zeros((0, n_basis))] + [_piece_rows(knots, degree, r) for r in ranges]
    return np.vstack(blocks)


def _piece_rows(knots: NDArray, degree: int, r: PolynomialRange) -> NDArray:
    """The (d+1)-th derivative at ``degree - d`` evenly spaced interior points."""
    n_points = degree - int(r.degree)
    fractions = np.arange(1, n_points + 1) / (n_points + 1)
    points = float(r.lo) + (float(r.hi) - float(r.lo)) * fractions
    return derivative_design(knots, degree, points, int(r.degree) + 1)


def derivative_design(knots: NDArray, degree: int, points: NDArray, order: int) -> NDArray:
    """Order-``order`` derivative of every basis function at ``points``."""
    identity = np.eye(len(knots) - degree - 1)
    return BSpline(knots, identity, degree)(points, nu=order)


def constraint_null_space(C: NDArray) -> NDArray:
    """Orthonormal Z with ``C @ Z = 0`` for a certified full-row-rank C.

    The rank is decided on rows scaled to unit length, which leaves the null
    space unchanged and makes the decision independent of the feature's units:
    derivative rows of different orders scale with different powers of the
    knot spacing. The threshold is NumPy's ``matrix_rank`` default,
    ``max(C.shape) * eps * sigma_max``. Z is the trailing block of the complete
    QR of C' (Wood 2017, section 1.8.1), taken on the rows as given: Householder
    QR is backward stable column by column (Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., ch. 19), so column scaling cannot improve
    it, and the natural-spline null space stays bit-identical to its old QR.
    """
    C = np.asarray(C, dtype=np.float64)
    if np.linalg.matrix_rank(C / np.linalg.norm(C, axis=1, keepdims=True)) < C.shape[0]:
        raise ValueError(
            "polynomial range constraints are dependent; ranges this close need to meet at a kink"
        )
    Q, _ = np.linalg.qr(C.T, mode="complete")
    return Q[:, C.shape[0] :]
