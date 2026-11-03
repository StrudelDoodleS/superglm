"""Rows for shape rules confined to a stretch [a, b] of a B-spline curve.

A rule on [a, b] is stated on the coefficients of ``f^(order)`` after ``a`` and
``b`` are inserted into the derivative spline's knot vector to full
multiplicity (knot insertion; Lyche & Mørken, *Spline Methods*, ch. 4). The
refined space contains ``f^(order)`` exactly and splits at ``a`` and ``b``, so
its coefficients on [a, b] bound ``f^(order)`` there (partition of unity) and
constrain nothing outside the stretch. Their first and last coefficients are
the values of ``f^(order)`` at ``a`` and ``b``.

Inequality rows follow the full-range builders in ``_spline_constraints``:
``A @ c >= 0`` means the rule holds, decreasing and concave rows are negated,
and each row is scaled by its largest entry. A rule on the whole range uses
those builders' rows, which exist at every degree.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from superglm.features._spline_constraints import (
    build_curvature_difference_constraints,
    build_monotone_difference_constraints,
)
from superglm.features._spline_ranges import derivative_design
from superglm.features.constraint import ConstraintRules, ConstraintSpec
from superglm.types import LinearConstraintSet

ShapePiece = tuple[str, float, float]

_SIGN = {"increasing": 1.0, "decreasing": -1.0, "convex": 1.0, "concave": -1.0}
_DIRECTION = ("increasing", "decreasing")
_CURVATURE = ("convex", "concave")


def derivative_value_row(knots: NDArray, degree: int, x: float, order: int) -> NDArray:
    """The row ``r`` with ``r @ c = f^(order)(x)``."""
    return derivative_design(knots, degree, np.array([float(x)]), order)[0]


def full_range_rows(
    knots: NDArray,
    degree: int,
    kind: str,
    *,
    domain: tuple[float, float],
    natural_ends: bool,
) -> NDArray:
    """The rows ``A @ c >= 0`` a direction or curvature rule on the whole range builds.

    Direction rules use the coefficient differences, curvature rules the
    knot-scaled curvature rows bounded at ``domain``. With ``natural_ends``
    the first and last curvature rows are dropped: the natural boundary
    conditions already hold ``f''`` at zero there as equalities, and their
    round-off projections would read as active inequalities in the QP.
    """
    if kind in _DIRECTION:
        return build_monotone_difference_constraints(len(knots) - degree - 1, kind).A
    rows = build_curvature_difference_constraints(knots, degree, kind, domain=domain).A
    return rows[1:-1] if natural_ends else rows


def _refined(
    knots: NDArray, degree: int, order: int, lo: float, hi: float
) -> tuple[NDArray, NDArray]:
    """The rows of :func:`refined_derivative_rows` and their clamped knot vector."""
    p = degree - order
    if p < 1:
        raise ValueError(
            f"a shape rule on a stretch needs a spline of degree {order + 1} or more, got "
            f"degree {degree}; raise the degree (degree=3 is a cubic)"
        )
    knots = np.asarray(knots, dtype=np.float64)
    t_r = knots[order : len(knots) - order]
    inner = t_r[(t_r > lo) & (t_r < hi)]
    t_ref = np.concatenate([np.full(p + 1, lo), inner, np.full(p + 1, hi)])
    tau = sliding_window_view(t_ref[1:-1], p).mean(axis=1)
    B_ref = BSpline.design_matrix(tau, t_ref, p).toarray()
    return np.linalg.solve(B_ref, derivative_design(knots, degree, tau, order)), t_ref


def refined_derivative_rows(
    knots: NDArray, degree: int, order: int, lo: float, hi: float
) -> NDArray:
    """Rows ``R`` with ``R @ c`` the coefficients of ``f^(order)`` on [lo, hi].

    The coefficients are those of the degree ``p = degree - order`` spline on
    the derivative knots ``knots[order:-order]`` refined by ``lo`` and ``hi``
    at multiplicity ``p + 1``, keeping the basis functions supported in
    [lo, hi]. Full multiplicity makes the refined space a direct sum of the
    pieces left of, inside and right of the stretch, so the kept coefficients
    are found by collocating on [lo, hi] alone, at the Greville abscissae of
    its clamped knot vector (Schoenberg–Whitney holds there; de Boor, *A
    Practical Guide to Splines*, ch. XIII). Collocating the whole refined space
    at once would repeat the abscissa at each split point and is singular.
    ``f^(order)`` must be continuous at ``hi``, which holds when the knot
    multiplicity there is at most ``p``, as for every simple interior knot.
    """
    return _refined(knots, degree, order, lo, hi)[0]


def _outside(a: float, b: float, holes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[a, b] less the union of ``holes``, as non-empty intervals."""
    parts = []
    for s, e in sorted(holes):
        if s > a:
            parts.append((a, min(s, b)))
        a = max(a, e)
    parts.append((a, b))
    return [(s, e) for s, e in parts if s < e]


def shape_pieces(
    rules: ConstraintRules | ConstraintSpec,
    lo: float,
    hi: float,
    *,
    feature: str | None = None,
    stacklevel: int = 1,
) -> list[ShapePiece]:
    """The direction, curvature and flat rules as numeric pieces on [lo, hi].

    Open ends become ``lo`` or ``hi``. A reversed stretch, or two that overlap
    within a rule group, is refused (``ConstraintRules.validate_stretches``). A
    stretch reaching past the range is clipped with a warning; one with no
    overlap is refused. Direction and
    curvature pieces are split around flat pieces, where a flat curve already
    satisfies them. Caps and step limits are not pieces and are skipped.
    ``feature`` names the feature in messages; ``stacklevel`` counts frames
    above this function for the clipping warning, 1 being its caller.
    """
    where = "" if feature is None else f"feature {feature!r}: "
    if isinstance(rules, ConstraintSpec):
        rules = ConstraintRules((rules,))
    rules.validate_stretches()
    pieces = []
    for rule in rules:
        if rule.kind not in _DIRECTION + _CURVATURE + ("flat",):
            continue
        if isinstance(rule.lo, str) or isinstance(rule.hi, str):
            raise ValueError(
                f"{where}{rule.describe()} ends at a band label, which only an ordered categorical "
                "feature can place; give the stretch in the feature's own values"
            )
        a = lo if rule.lo is None else float(rule.lo)
        b = hi if rule.hi is None else float(rule.hi)
        if not (a < hi and b > lo):
            raise ValueError(
                f"{where}{rule.describe()} lies outside the feature's range [{lo:g}, {hi:g}]; "
                "move the stretch inside the range or drop the rule"
            )
        if a < lo or b > hi:
            a, b = max(a, lo), min(b, hi)
            warnings.warn(
                f"{where}{rule.describe()} reaches past the feature's range [{lo:g}, {hi:g}]; "
                f"the rule applies on [{a:g}, {b:g}]",
                UserWarning,
                stacklevel=stacklevel + 1,
            )
        pieces.append((rule.kind, a, b))
    flats = [(a, b) for kind, a, b in pieces if kind == "flat"]
    split = [
        (kind, s, e) for kind, a, b in pieces if kind != "flat" for s, e in _outside(a, b, flats)
    ]
    return split + [("flat", a, b) for a, b in flats]


def _zero_regions(knots: NDArray, flats: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Where flat stretches hold ``f'`` at zero: the knot spans they meet, merged.

    Regions that touch or overlap are merged, since ``f'`` is zero on their
    union.
    """
    spans = sorted((knots[knots <= a].max(), knots[knots >= b].min()) for a, b in flats)
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def flat_zero_regions(knots: NDArray, pieces: list[ShapePiece]) -> list[tuple[float, float]]:
    """Where the flat pieces hold the curve constant: every whole knot span they touch, merged."""
    flats = [(a, b) for kind, a, b in pieces if kind == "flat"]
    return _zero_regions(np.asarray(knots, dtype=np.float64), flats)


def natural_flat_extent(
    knots: NDArray, degree: int, lo: float, hi: float, *, natural_ends: bool
) -> tuple[float, float]:
    """Where a flat zero region [lo, hi] holds a cubic with natural ends constant.

    A natural end holds ``f''`` at zero at the end of the knot vector's base
    interval. A zero region that begins at the first interior knot leaves the
    boundary span a cubic with ``f'' = 0`` at both of its ends, so ``f''`` is
    zero on it, and with ``f' = 0`` at the knot the curve is constant there
    too. The same holds at the right end.
    """
    if degree != 3 or not natural_ends:
        return lo, hi
    n_basis = len(knots) - degree - 1
    if lo == knots[degree + 1]:
        lo = knots[degree]
    if hi == knots[n_basis - 1]:
        hi = knots[n_basis]
    return lo, hi


def _held_at_zero(lo: NDArray, hi: NDArray, zero: list[tuple[float, float]]) -> NDArray:
    """Which supports [lo, hi] overlap a stretch where ``f'`` is held at zero."""
    held = np.zeros(np.shape(lo), dtype=bool)
    for s, e in zero:
        held |= (lo < e) & (hi > s)
    return held


def _flat_rows(
    knots: NDArray, degree: int, zero: list[tuple[float, float]], natural_ends: bool
) -> NDArray:
    """Rows holding ``f' = 0`` on the zero regions, less any a natural end implies.

    Each region is a union of whole knot spans, and B-splines are linearly
    independent on every span, so ``f' = 0`` there exactly when every
    first-derivative coefficient ``d_j = degree (c[j+1] - c[j]) / (t[j+degree+1]
    - t[j+1])`` whose support meets a region is zero. Those rows are a subset of
    a full-row-rank difference operator, so they are independent however the
    regions sit. A natural spline's ``f''`` at an end of its clamped base
    interval is a combination of the two derivative coefficients there, so a
    region reaching that end drops the second.
    """
    n_basis = len(knots) - degree - 1
    spans = knots[degree + 1 : degree + n_basis] - knots[1:n_basis]
    D = degree * np.diff(np.eye(n_basis), axis=0) / spans[:, None]
    held = _held_at_zero(knots[1:n_basis], knots[degree + 1 : degree + n_basis], zero)
    if natural_ends:
        held[1] &= not any(s == knots[degree] for s, _ in zero)
        held[-2] &= not any(e == knots[n_basis] for _, e in zero)
    return D[held]


def shape_rows(
    knots: NDArray,
    degree: int,
    pieces: list[ShapePiece],
    *,
    domain: tuple[float, float],
    natural_ends: bool,
) -> tuple[LinearConstraintSet, NDArray]:
    """Inequality rows ``A @ c >= 0`` and flat equality rows ``E @ c = 0``.

    A piece on the whole of ``domain`` (the feature's range) takes the rows
    :func:`full_range_rows` builds. A piece on a stretch takes the refined
    rows, which need ``degree - order >= 1``: a direction stretch needs
    degree 2 or more, a curvature stretch degree 3 or more.

    A direction piece with a curvature piece on the same stretch keeps one
    end-slope row: with ``f'`` monotone on [a, b], its sign there is its sign
    at one end (increasing & concave: ``f'(b) >= 0``; increasing & convex:
    ``f'(a) >= 0``; decreasing & convex: ``f'(b) <= 0``; decreasing & concave:
    ``f'(a) <= 0``).

    A polynomial that is constant on part of a knot span is constant on all
    of it, so a flat stretch holds ``f'`` and ``f''`` at zero on every knot
    span it meets. The flat rows ``E`` state ``f' = 0`` on those spans through
    the spline's own derivative coefficients, which keeps them independent.

    Inequality rows the equalities already hold at zero are dropped, since
    they compose to round-off and degrade the QP's active set: every refined
    coefficient whose basis function is active on a zero region, and, at a
    piece end inside a closed zero region, the first ``p`` coefficients, which
    are fixed by ``f^(order)`` and its ``p - 1`` derivatives there (continuous
    at a simple knot, so zero). With ``natural_ends``, ``f''`` is held at zero
    at the ends of the knot vector's base interval, where a natural spline
    (clamped at its range) puts its boundary conditions: a curvature piece
    drops its coefficient there, and a zero region reaching that end drops the
    flat row the boundary row implies. A zero region that starts (ends) at the
    first (last) interior knot holds the boundary span flat beside it
    (:func:`natural_flat_extent`), so the rows above are dropped for that span too.
    """
    knots = np.asarray(knots, dtype=np.float64)
    n_basis = len(knots) - degree - 1
    flats = [(a, b) for kind, a, b in pieces if kind == "flat"]
    zero = _zero_regions(knots, flats)
    held_zero = [
        natural_flat_extent(knots, degree, s, e, natural_ends=natural_ends) for s, e in zero
    ]
    ends = (knots[degree], knots[n_basis])
    curvature = {(a, b): kind for kind, a, b in pieces if kind in _CURVATURE}

    blocks = [np.zeros((0, n_basis))]
    for kind, a, b in pieces:
        if kind == "flat":
            continue
        sign = _SIGN[kind]
        if kind in _DIRECTION and (a, b) in curvature:
            end = b if sign * _SIGN[curvature[(a, b)]] < 0 else a
            if not any(s <= end <= e for s, e in held_zero):
                blocks.append(sign * derivative_value_row(knots, degree, end, 1)[None, :])
            continue
        if (a, b) == tuple(domain):
            rows = full_range_rows(knots, degree, kind, domain=domain, natural_ends=natural_ends)
            blocks.append(rows)
            continue
        order = 1 if kind in _DIRECTION else 2
        if degree - order < 1:
            raise ValueError(
                f"{kind} on the stretch [{a:g}, {b:g}] needs a spline of degree {order + 1} or "
                f"more, and this term has degree {degree}; raise the degree (degree=3 is a "
                "cubic) or give the rule on the whole range"
            )
        rows, t_ref = _refined(knots, degree, order, a, b)
        m, p = rows.shape[0], degree - order
        keep = ~_held_at_zero(t_ref[:m], t_ref[p + 1 :], held_zero)
        if any(s <= a <= e for s, e in held_zero):
            keep[:p] = False
        if any(s <= b <= e for s, e in held_zero):
            keep[m - p :] = False
        if order == 2 and natural_ends:
            keep[0] &= a != ends[0]
            keep[-1] &= b != ends[1]
        blocks.append(sign * rows[keep])
    A = np.vstack(blocks)
    A = A / np.max(np.abs(A), axis=1, keepdims=True) if A.shape[0] else A
    E = _flat_rows(knots, degree, zero, natural_ends)
    return LinearConstraintSet(A=A, b=np.zeros(A.shape[0])), E


def spline_cap_rows(
    knots: NDArray,
    degree: int,
    mean_row: NDArray,
    bounds: tuple[float, float],
    *,
    ends: tuple[float, float] | None,
) -> LinearConstraintSet:
    """Rows ``A @ c >= b`` holding the printed curve ``(B(x) - m) @ c`` in ``[log lo, log hi]``.

    ``m`` (``mean_row``) is the training basis mean whose product with ``c`` is
    the shift the printed curve is centred by. With ``ends`` (a curve monotone
    on the whole range, whose extremes are at its ends) the rows bound the
    curve at the two ends, which is exact. Otherwise they bound every
    coefficient, ``log lo <= (e_j - m) @ c <= log hi``: B-splines are a
    non-negative partition of unity, so the curve is a convex combination of
    its coefficients and the rows are sufficient, strict by a gap that shrinks
    as O(h^2 |f''|) (Lyche & Mørken, *Spline Methods*, Lemma 9.17, Thm 9.18).
    Both are invariant to adding a constant to ``c``, since ``m`` sums to one.
    Each row is scaled by its largest entry; a zero row bounds nothing and is
    dropped.
    """
    knots = np.asarray(knots, dtype=np.float64)
    n_basis = len(knots) - degree - 1
    if ends is None:
        values = np.eye(n_basis)
    else:
        values = derivative_design(knots, degree, np.asarray(ends, dtype=np.float64), 0)
    shifted = values - np.asarray(mean_row, dtype=np.float64)[None, :]
    A = np.vstack([shifted, -shifted])
    log_lo, log_hi = np.log(bounds[0]), np.log(bounds[1])
    b = np.concatenate([np.full(len(shifted), log_lo), np.full(len(shifted), -log_hi)])
    scale = np.max(np.abs(A), axis=1)
    keep = scale > 0.0
    return LinearConstraintSet(A=A[keep] / scale[keep, None], b=b[keep] / scale[keep])


__all__ = [
    "ShapePiece",
    "derivative_value_row",
    "flat_zero_regions",
    "full_range_rows",
    "natural_flat_extent",
    "refined_derivative_rows",
    "shape_pieces",
    "shape_rows",
    "spline_cap_rows",
]
