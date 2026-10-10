"""Private penalty-construction helpers for spline feature specs."""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np
from numpy.typing import NDArray

from superglm.features._spline_ranges import derivative_design

# 1/sqrt(eps): solvers/rank.py's warning_condition for a Gram.
_CONDITION_LIMIT = float(1.0 / np.sqrt(np.finfo(np.float64).eps))


def build_difference_penalty(n_basis: int, order: int) -> NDArray:
    """Difference-penalty matrix from order-`m` finite differences."""
    if order >= n_basis:
        raise ValueError(
            f"Difference order {order} >= n_basis {n_basis}. Increase n_knots or reduce m."
        )
    Dm = np.diff(np.eye(n_basis), n=order, axis=0)
    return Dm.T @ Dm


def build_general_difference_penalty(knots: NDArray, degree: int, order: int) -> NDArray:
    """General difference penalty for B-spline coefficients on any knot vector.

    Li and Cao, "General P-splines for non-uniform B-splines" (2022,
    arXiv:2201.06808, section 2): each difference step divides by the span of
    the derivative's B-spline, ``(t[j+d-k] - t[j]) / (d - k)`` with
    ``d = degree + 1`` (de Boor's derivative formula), so ``D_m beta`` holds the
    B-spline coefficients of the m-th derivative and the null space is the
    polynomials of degree below ``order`` however the knots are spaced. The
    standard difference penalty loses that on uneven knots.

    The penalty is scaled to the standard one's largest eigenvalue for the same
    basis size, so a fixed ``spline_penalty`` weighs it as it weighs the
    standard penalty, and its magnitude is never ``(hbar / span)**(2 * order)``
    (``hbar`` the mean knot interval): 1e6 on five quantile knots of
    lognormal(0, 1) data, against the standard penalty's 16, past the absolute
    tolerances the fit applies to penalties. On evenly spaced knots the scale
    factor is one to rounding.

    Row ``i`` carries the weight ``prod(hbar / span)`` of the spans it divides
    by, so knots clustered on a short stretch spread the penalty's eigenvalues
    past what float64 holds: on 8 knots within 1e-4 of a unit axis they run
    from 1e-3 to 5e16, and even the correctly rounded Gram of the exact
    penalty is indefinite by 3 in its null space, so no way of forming it
    helps, and knots 1e-160 apart on a unit axis overflow it. It is kept while
    its entries are finite and the condition of its nonzero spectrum, the
    largest eigenvalue over the ``order + 1``-th smallest, is at most
    ``_CONDITION_LIMIT = 1/sqrt(eps)``: the condition past which a Gram keeps
    under half of float64's digits, where the shared rank policy
    (``solvers/rank.py``, ``warning_condition``) starts to ask for a factor
    certificate. REML's rank cut, ``_REML_RANK_THRESHOLD`` of the largest
    eigenvalue, sits below that, and so far above the eigensolver's
    resolution of ``n * eps`` of it; the REML derivatives of a ``select=True``
    split failed their accuracy certificate at a condition of 2e10 on the
    book's ``VehAge``, within the rank cut. Past the limit the term takes the standard difference factor with
    the polynomials of degree below ``order`` projected out of it,
    ``Delta_m (I - Q Q')``, ``Q`` an orthonormal basis of their B-spline
    coefficients: the null space is still exactly the polynomials, the scale
    and spread are the standard penalty's, and the penalty is a Gram of its
    factor, never of a matrix with ``(hbar / span)**order`` entries.
    """
    return _general_difference_penalty(knots, degree, order)[0]


def _general_difference_penalty(knots: NDArray, degree: int, order: int) -> tuple[NDArray, str]:
    """``build_general_difference_penalty`` and which penalty it took: ``"general"`` or ``"projected"``.

    The limit is on the general penalty's own condition, not on its condition
    over the standard penalty's on the same basis, although the standard
    penalty's condition grows with the basis (roughly as ``n_basis**(2 * order)``)
    and so leaves less room for uneven knots as the basis grows: 6.7e7 over a
    measured 633 on 12 basis functions at ``order = 2``, over a measured 4.2e6
    on 40 at ``order = 3``. The fit's numerics see the absolute condition:
    a limit relative to the standard penalty would keep the general penalty at
    1.9e10 on freMTPL2's ``VehAge`` with 12 ``quantile_rows`` knots, the case
    whose ``select=True`` REML derivatives failed their certificate. The
    switch is therefore a step in the knots, and the spec records which side
    a term fell on (``difference_penalty_kind``).
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
            top = np.linalg.eigvalsh(standard.T @ standard)[-1]
            return penalty * (top / eigenvalues[-1]), "general"
    Q, _ = np.linalg.qr(_polynomial_coefficients(t, degree, order))
    standard -= (standard @ Q) @ Q.T
    return standard.T @ standard, "projected"


def _polynomial_coefficients(knots: NDArray, degree: int, order: int) -> NDArray:
    """B-spline coefficients of ``1, s, ..., s**(order - 1)``, ``s`` the domain mapped to [-1, 1].

    Marsden's identity (de Boor, *A Practical Guide to Splines*, rev. ed.
    2001, ch. IX): the coefficient of ``s**p`` on ``B_j`` is the p-th
    elementary symmetric function of ``s(t[j+1]), ..., s(t[j+degree])`` over
    ``binomial(degree, p)``.
    """
    n_basis = len(knots) - degree - 1
    lo, hi = knots[degree], knots[n_basis]
    s = (2.0 * knots - (lo + hi)) / (hi - lo)
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
        penalty, kind = _general_difference_penalty(spec._knots, spec.degree, order)
    else:
        penalty, kind = build_difference_penalty(spec._n_basis, order), "standard"
    # A spec saved before the record existed has no attribute until it is placed again.
    if getattr(spec, "_difference_penalty", None) is None:
        spec._difference_penalty = {}
    spec._difference_penalty[order] = kind
    return penalty


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

    Knots bitwise equal to the uniform rule's own construction,
    ``np.linspace(lo, hi, k + 2)[1:-1]`` for ``k`` interior knots, are evenly
    spaced; a refit from ``fitted_knots`` states exactly those. Other knots must
    spread their gaps by no more than that construction's rounding can leave.
    With ``M = max(|lo|, |hi|)``, each point of ``np.linspace`` is four roundings
    (the difference, the quotient by ``k + 1``, the product with ``i``, the sum
    with ``lo``): the first three act on quantities of at most ``2M``, the last
    on one of at most ``M``, so a point lies within ``7 u M`` of its place for
    any ``k``. A gap is then within ``14 u M`` of the even gap, two gaps spread
    by ``28 u M``, and rounding each gap adds ``u M`` to its deviation, ``2 u M``
    in all: ``30 u M`` to first order. The tolerance is ``32 u M``, its last two
    units covering the second order.
    """
    interior = spec._knots[spec.degree + 1 : -(spec.degree + 1)]
    if np.array_equal(interior, np.linspace(spec._lo, spec._hi, interior.size + 2)[1:-1]):
        return True
    points = np.r_[spec._lo, interior, spec._hi]
    gaps = np.diff(points)
    u = np.finfo(np.float64).eps / 2
    return bool(np.ptp(gaps) <= 32 * u * np.abs(points).max())


def build_integrated_derivative_penalty(
    knots: NDArray,
    degree: int,
    order: int,
    excluded: Sequence[tuple[float, float]] = (),
) -> NDArray:
    """Integrated squared derivative penalty via Gauss-Legendre quadrature.

    The integral is an exact sum of per-knot-interval blocks (Wood 2016,
    arXiv:1605.02446, section 1), so leaving out the intervals inside an
    ``excluded`` ``(lo, hi)`` leaves the curve there unpenalised.
    """
    if order > degree:
        raise ValueError(
            f"Derivative order {order} > spline degree {degree}. "
            "Integrated-derivative penalty requires order <= degree."
        )
    K = len(knots) - degree - 1
    return sum(_interval_blocks(knots, degree, order, excluded), np.zeros((K, K)))


def structural_derivative_penalty(
    knots: NDArray,
    degree: int,
    order: int,
    excluded: Sequence[tuple[float, float]] = (),
) -> NDArray:
    """The same penalty with each interval's block scaled to unit norm.

    A sum of positive semidefinite blocks has the null space of every positive
    weighting of them, so this has the penalty's exact rank without the
    ``width**-3`` spread a narrow interval gives the penalty's eigenvalues.
    """
    K = len(knots) - degree - 1
    blocks = _interval_blocks(knots, degree, order, excluded)
    return sum((block / np.linalg.norm(block) for block in blocks), np.zeros((K, K)))


def _interval_blocks(knots, degree, order, excluded):
    """Each unpinned knot interval's block of the integrated squared ``order``-th derivative."""
    unique_knots = np.unique(knots)
    starts, ends = unique_knots[:-1], unique_knots[1:]
    bounds = np.asarray(excluded, dtype=np.float64).reshape(-1, 2)
    pinned = (starts[:, None] >= bounds[:, 0]) & (ends[:, None] <= bounds[:, 1])
    kept = (ends - starts >= 1e-15) & ~pinned.any(axis=1)
    xi, wi = np.polynomial.legendre.leggauss(max(order + 1, degree))
    for a, b in zip(starts[kept], ends[kept]):
        x_q = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w_q = 0.5 * (b - a) * wi
        Dm_q = derivative_design(knots, degree, x_q, order)
        yield Dm_q.T @ (Dm_q * w_q[:, None])


__all__ = [
    "build_difference_penalty",
    "build_integrated_derivative_penalty",
    "structural_derivative_penalty",
]
