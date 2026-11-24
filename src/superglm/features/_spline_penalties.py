"""Private penalty-construction helpers for spline feature specs."""

from __future__ import annotations

from collections.abc import Sequence
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import EPS, UNIT_ROUNDOFF
from superglm.features._spline_ranges import _REML_RANK_THRESHOLD, derivative_design
from superglm.solvers.rank import SHARED_RANK_POLICY

# 1/sqrt(eps): solvers/rank.py's warning_condition for a Gram.
_CONDITION_LIMIT = float(1.0 / np.sqrt(EPS))


def build_difference_penalty(n_basis: int, order: int) -> NDArray:
    """Difference-penalty matrix from order-`m` finite differences."""
    if order >= n_basis:
        raise ValueError(
            f"Difference order {order} >= n_basis {n_basis}. Increase n_knots or reduce m."
        )
    Dm = np.diff(np.eye(n_basis), n=order, axis=0)
    return Dm.T @ Dm


def _general_difference_penalty(
    knots: NDArray, degree: int, order: int
) -> tuple[NDArray, str, NDArray]:
    """General difference penalty for B-spline coefficients on any knot vector.

    Li and Cao (2022, arXiv:2201.06808, sec. 2): each difference step divides by the span
    ``(t[j+d-k] - t[j]) / (d - k)``, ``d = degree + 1``, so ``D_m beta`` holds the m-th
    derivative's B-spline coefficients and the null space is the polynomials of degree below
    ``order`` on any knots. Scaled to the standard penalty's largest eigenvalue, it is kept while
    finite with nonzero-spectrum condition at most ``_CONDITION_LIMIT = 1/sqrt(eps)``.
    Past it the term takes ``Delta_m (I - Q Q')``, ``Q`` an orthonormal basis of
    those polynomials' coefficients: same null space, the standard scale and spread, a Gram of
    its factor. Returns ``(penalty, kind, root)``, ``kind`` ``"general"`` or ``"projected"``,
    and ``root' root`` the penalty (``difference_penalty_root``).
    """
    t = np.asarray(knots, dtype=np.float64)
    d = degree + 1
    n_basis = len(t) - d
    if not 1 <= order <= degree:
        raise ValueError(f"General difference order {order} needs 1 <= order <= degree ({degree}).")
    hbar = (t[n_basis] - t[degree]) / (n_basis - degree)
    D = np.eye(n_basis)
    # Knots 1e-160 apart overflow (hbar / span)**(2 * order); that penalty
    # takes the projected standard factor below, never inf or NaN entries.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(1, order + 1):
            j = np.arange(k, n_basis)
            span = (t[j + d - k] - t[j]) / (d - k)
            D = np.diff(D, axis=0) * (hbar / span)[:, None]
        penalty = D.T @ D
    standard = np.diff(np.eye(n_basis), n=order, axis=0)
    if np.isfinite(penalty).all():
        eigenvalues = np.linalg.eigvalsh(penalty)
        if eigenvalues[order] * _CONDITION_LIMIT >= eigenvalues[-1]:
            scale = np.linalg.eigvalsh(standard.T @ standard)[-1] / eigenvalues[-1]
            return penalty * scale, "general", D * np.sqrt(scale)
    Q, _ = np.linalg.qr(_polynomial_coefficients(t, degree, order))
    standard -= (standard @ Q) @ Q.T
    return standard.T @ standard, "projected", standard


def normalized_knots(knots: NDArray, left: int, right: int) -> NDArray:
    """Map the domain endpoints to -1 and 1 without overflowing or losing a common offset."""
    _, exponent = np.frexp(np.max(np.abs(knots)))
    scaled = np.ldexp(knots, -exponent)
    return 2.0 * ((scaled - scaled[left]) / (scaled[right] - scaled[left])) - 1.0


def _polynomial_coefficients(knots: NDArray, degree: int, order: int) -> NDArray:
    """B-spline coefficients of ``1, s, ..., s**(order - 1)``, ``s`` the domain mapped to [-1, 1].

    Marsden's identity (de Boor, *A Practical Guide to Splines*, rev. ed.
    2001, ch. IX): the coefficient of ``s**p`` on ``B_j`` is the p-th
    elementary symmetric function of ``s(t[j+1]), ..., s(t[j+degree])`` over
    ``binomial(degree, p)``.
    """
    n_basis = len(knots) - degree - 1
    s = normalized_knots(knots, degree, n_basis)
    windows = np.lib.stride_tricks.sliding_window_view(s[1 : n_basis + degree], degree)
    elementary = np.zeros((n_basis, order))
    elementary[:, 0] = 1.0
    for k in range(degree):
        elementary[:, 1:] += windows[:, k : k + 1] * elementary[:, :-1]
    return elementary / np.array([comb(degree, p) for p in range(order)])


def difference_penalty_for(spec, order: int) -> NDArray:
    """A difference-penalised spline's penalty: general once its knots are not evenly placed.

    Evenly spaced knots keep the standard penalty, whether the ``"uniform"``
    rule placed them or they were stated, as a refit from ``fitted_knots``
    states them. Other stated and quantile-placed knots take the general
    penalty, which needs ``order <= degree``; a higher order keeps the
    standard penalty.

    Records the penalty taken for ``order`` in ``spec._difference_penalty``,
    which knot placement empties: ``"standard"``, ``"general"`` or
    ``"projected"`` (the standard factor with the polynomials projected out,
    for knots too uneven for the general penalty).
    """
    if (
        spec._knot_strategy_actual != "uniform"
        and order <= spec.degree
        and not _evenly_spaced(spec)
    ):
        penalty, kind, _ = _general_difference_penalty(spec._knots, spec.degree, order)
    else:
        penalty, kind = build_difference_penalty(spec._n_basis, order), "standard"
    # A spec saved before the record existed has no attribute until it is placed again.
    if getattr(spec, "_difference_penalty", None) is None:
        spec._difference_penalty = {}
    spec._difference_penalty[order] = kind
    return penalty


def difference_null_basis(spec, order: int) -> NDArray | None:
    """Raw coefficients of the null space of the difference penalty ``difference_penalty_for`` built.

    The standard penalty annihilates the sequences ``j**p``, ``p < order``
    (Eilers and Marx 1996, sec. 5). Constants and lines have a well-resolved
    basis. For higher powers, use the difference operator's right null space
    if its numerical rank resolves the declared order. The general and
    projected penalties annihilate polynomials in the knot variable.
    """
    if getattr(spec, "_difference_penalty", {}).get(order) == "standard":
        if order > 2:
            from scipy.linalg import null_space

            root = difference_penalty_root(spec, order)
            basis = null_space(root, rcond=SHARED_RANK_POLICY.factor_rcond)
            return basis if basis.shape[1] == order else None
        return np.vander(np.arange(spec._n_basis, dtype=np.float64), order, increasing=True)
    return _polynomial_coefficients(spec._knots, spec.degree, order)


def difference_penalty_root(spec, order: int) -> NDArray:
    """The factor ``R`` with ``R' R`` the difference penalty ``difference_penalty_for`` built.

    The difference matrix itself (Eilers and Marx 1996, sec. 5; Li and Cao
    2022, sec. 2.4): ``Delta_m`` for the standard penalty, the scaled
    general ``D_m``, or ``Delta_m (I - Q Q')`` for the projected one.
    """
    if getattr(spec, "_difference_penalty", {}).get(order) == "standard":
        return np.diff(np.eye(spec._n_basis), n=order, axis=0)
    return _general_difference_penalty(spec._knots, spec.degree, order)[2]


def difference_penalty_kind(spec) -> str | None:
    """Which difference penalty a built spline took, for reports; None without one.

    One name when every penalty order took the same one, otherwise each
    order's, as ``"m=2 general, m=3 projected"``.
    """
    kinds = getattr(spec, "_difference_penalty", None) or {}
    named = [(order, kinds[order]) for order in getattr(spec, "_m_orders", ()) if order in kinds]
    if not named:
        return None
    if len({kind for _, kind in named}) == 1:
        return named[0][1]
    return ", ".join(f"m={order} {kind}" for order, kind in named)


def _evenly_spaced(spec) -> bool:
    """Whether the interior knots split ``[lo, hi]`` into equal intervals, to rounding.

    Knots bitwise equal to the uniform rule's own ``np.linspace(lo, hi, k + 2)[1:-1]``
    (as a refit from ``fitted_knots`` states) are evenly spaced; others must spread
    their gaps by no more than that construction's rounding can leave.  With ``M =
    max(|lo|, |hi|)`` each ``np.linspace`` point lies within ``7 u M`` of its place
    (four roundings), a gap within ``14 u M`` of the even gap, two gaps spread by
    ``28 u M``, and rounding each gap adds ``2 u M`` in all: ``30 u M`` to first
    order.  The tolerance is ``32 u M``.
    """
    interior = spec._knots[spec.degree + 1 : -(spec.degree + 1)]
    if np.array_equal(interior, np.linspace(spec._lo, spec._hi, interior.size + 2)[1:-1]):
        return True
    points = np.r_[spec._lo, interior, spec._hi]
    gaps = np.diff(points)
    return bool(np.ptp(gaps) <= 32 * UNIT_ROUNDOFF * np.abs(points).max())


def build_integrated_derivative_penalty(
    knots: NDArray,
    degree: int,
    order: int,
    excluded: Sequence[tuple[float, float]] = (),
    intervals: int | None = None,
) -> NDArray:
    """Integrated squared derivative penalty via Gauss-Legendre quadrature, in knot-interval units.

    The integral is an exact sum of per-knot-interval blocks (Wood 2016,
    arXiv:1605.02446, section 1), so leaving out the intervals inside an
    ``excluded`` ``(lo, hi)`` leaves the curve there unpenalised.

    It is taken over ``u = (x - t[degree]) / hbar``, ``hbar`` the mean knot interval
    (``mean_knot_interval``), i.e. ``hbar**(2 * order - 1)`` times the integral over
    ``x``.  Over ``x`` it scales with the column's units to the power ``1 - 2 *
    order``; over ``u`` it is unit-free.  On evenly spaced knots, away from clamped
    ends, it is ``Delta_m' G Delta_m``, the standard difference penalty sandwiched by
    the Gram ``G`` of the degree ``degree - order`` B-splines on unit spacing (Li and
    Cao, arXiv:2201.06808, section 2.4), so a fixed ``spline_penalty`` smooths it
    about as strongly as a P-spline's.  Forming the integral on the mapped knots
    keeps every intermediate near one whatever the units.  ``intervals``, when
    given, is the number of knot intervals ``hbar`` divides the domain into
    (``_unit_knots``).
    """
    if order > degree:
        raise ValueError(
            f"Derivative order {order} > spline degree {degree}. "
            "Integrated-derivative penalty requires order <= degree."
        )
    K = len(knots) - degree - 1
    if _collapsed(knots, degree):
        return np.zeros((K, K))
    knots, bounds = _unit_knots(knots, degree, excluded, intervals)
    return sum(_interval_blocks(knots, degree, order, bounds), np.zeros((K, K)))


def integrated_derivative_root(
    knots: NDArray,
    degree: int,
    order: int,
    excluded: Sequence[tuple[float, float]] = (),
    intervals: int | None = None,
) -> NDArray:
    """The quadrature root ``R = W^(1/2) D`` of ``build_integrated_derivative_penalty``.

    Its rows are the ``order``-th derivatives of the basis at each kept
    interval's Gauss-Legendre nodes, times the square roots of their
    weights, so ``R' R`` is the penalty (Wood 2016, arXiv:1605.02446,
    section 1, steps 5 and 6). ``R`` keeps its null space, the polynomials
    of degree below ``order``, to ``u kappa(R)``; the formed penalty keeps
    it only to ``u kappa(Omega) = u kappa(R)**2``.
    """
    K = len(knots) - degree - 1
    if _collapsed(knots, degree):
        return np.zeros((0, K))
    knots, bounds = _unit_knots(knots, degree, excluded, intervals)
    rows = [
        np.sqrt(w_q)[:, None] * Dm_q for Dm_q, w_q in _interval_nodes(knots, degree, order, bounds)
    ]
    return np.vstack(rows) if rows else np.zeros((0, K))


def mean_knot_interval(breaks: NDArray) -> float:
    """The mean width of the intervals between the distinct ``breaks``.

    The length an integrated derivative penalty is measured in
    (``build_integrated_derivative_penalty``): it scales with the column's
    units and, on evenly spaced knots, is the knot spacing. Repeated knots,
    such as a polynomial range's edges, are one break.
    """
    distinct = np.unique(breaks)
    return float((distinct[-1] - distinct[0]) / (distinct.size - 1))


def _collapsed(knots: NDArray, degree: int) -> bool:
    """Whether the domain ``[t[degree], t[n_basis]]`` has no width, as on a one-value column.

    Such a domain has no knot interval to integrate over, so its penalty is
    zero, and ``_unit_knots`` would divide its zero width by zero.
    """
    return bool(knots[degree] == knots[len(knots) - degree - 1])


def _unit_knots(
    knots: NDArray,
    degree: int,
    excluded: Sequence[tuple[float, float]],
    intervals: int | None = None,
) -> tuple[NDArray, NDArray]:
    """``knots`` and the ``excluded`` bounds mapped to ``(x - t[degree]) / hbar``.

    ``hbar`` is the mean knot interval on the domain ``[t[degree], t[n_basis]]``: its
    width over ``intervals``, or over the number of intervals between the distinct
    knots when not given.  A spline with polynomial ranges passes its base layout's
    count, so a range's edge knots do not rescale the penalty on the rest of the
    curve.  Both go through the same monotone map, so a bound equal to a knot stays
    equal to it.
    """
    t = np.asarray(knots, dtype=np.float64)
    origin = t[degree]
    if intervals is None:
        hbar = mean_knot_interval(t[degree : t.size - degree])
    else:
        hbar = float((t[t.size - degree - 1] - origin) / intervals)
    bounds = np.asarray(excluded, dtype=np.float64).reshape(-1, 2)
    return (t - origin) / hbar, (bounds - origin) / hbar


def structural_derivative_penalty(
    knots: NDArray,
    degree: int,
    order: int,
    excluded: Sequence[tuple[float, float]] = (),
    intervals: int | None = None,
) -> NDArray:
    """The same penalty with each interval's block scaled to unit norm.

    A sum of positive semidefinite blocks has the null space of every positive
    weighting of them, so this has the penalty's exact rank without the
    ``width**(1 - 2 * order)`` spread a narrow interval gives the penalty's eigenvalues.
    """
    K = len(knots) - degree - 1
    if _collapsed(knots, degree):
        return np.zeros((K, K))
    knots, bounds = _unit_knots(knots, degree, excluded, intervals)
    blocks = _interval_blocks(knots, degree, order, bounds)
    return sum((block / np.linalg.norm(block) for block in blocks), np.zeros((K, K)))


def _interval_blocks(knots, degree, order, excluded):
    """Each unpinned knot interval's block of the integrated squared ``order``-th derivative."""
    for Dm_q, w_q in _interval_nodes(knots, degree, order, excluded):
        yield Dm_q.T @ (Dm_q * w_q[:, None])


def _interval_nodes(knots, degree, order, excluded):
    """Each unpinned interval's ``order``-th basis derivatives at its Gauss-Legendre nodes, and weights."""
    unique_knots = np.unique(knots)
    starts, ends = unique_knots[:-1], unique_knots[1:]
    bounds = np.asarray(excluded, dtype=np.float64).reshape(-1, 2)
    pinned = (starts[:, None] >= bounds[:, 0]) & (ends[:, None] <= bounds[:, 1])
    kept = (ends - starts >= 1e-15) & ~pinned.any(axis=1)
    xi, wi = np.polynomial.legendre.leggauss(max(order + 1, degree))
    for a, b in zip(starts[kept], ends[kept]):
        x_q = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w_q = 0.5 * (b - a) * wi
        yield derivative_design(knots, degree, x_q, order), w_q


def structural_penalty_ranks(spec: Any, info: Any) -> dict[str, int] | None:
    """The rank of each of ``info``'s penalties, from the structural penalty in its coordinates.

    An integrated derivative penalty's null space is the polynomials of degree below
    its order wherever the knots are, so its rank is a property of the construction,
    not of its spectrum (which spreads as ``width**(1 - 2 * order)``); smoothers declare it with
    the penalty (mgcv ``smooth.construct`` ``rank`` and ``null.space.dim``; Wood,
    Pya and Saefken, JASA 111, 2016, section 3.1.1).  Each penalty in ``info`` is
    ``T' S T`` for the raw penalty ``S`` and projection ``T``, so ``T' structural T``
    has its null space; its rank is decided at REML's cut only inside a gap of
    ``certification_band``, else nothing is declared.

    Keys are the component suffixes, or ``""`` for ``penalty_matrix`` when there are
    no components; ``"null"`` (select's identity) is not declared.  Returns ``None``
    when the spec has no structural penalty.
    """
    if info.penalty_matrix is None or spec._structural_penalty_for_order(spec._m_orders[0]) is None:
        return None
    projection = info.projection

    def rank_for(orders: Sequence[int]) -> int | None:
        structural = sum(spec._structural_penalty_for_order(order) for order in orders)
        if projection is not None:
            structural = projection.T @ structural @ projection
        return _gapped_rank(np.linalg.eigvalsh(structural))

    if info.penalty_components is None:
        keyed = {"": spec._m_orders}
    else:
        keyed = {
            suffix: (int(suffix[1:]),) if suffix.startswith("d") else spec._m_orders
            for suffix, _ in info.penalty_components
            if suffix != "null"
        }
    ranks = {key: rank_for(orders) for key, orders in keyed.items()}
    return {key: rank for key, rank in ranks.items() if rank is not None} or None


def _gapped_rank(values: NDArray) -> int | None:
    top = max(float(values[-1]), 0.0)
    cut = _REML_RANK_THRESHOLD * top
    band = SHARED_RANK_POLICY.certification_band
    if top == 0.0 or np.any((values > cut / band) & (values <= cut * band)):
        return None
    return int(np.count_nonzero(values > cut))


__all__ = [
    "build_difference_penalty",
    "build_integrated_derivative_penalty",
    "mean_knot_interval",
    "structural_derivative_penalty",
    "structural_penalty_ranks",
]
