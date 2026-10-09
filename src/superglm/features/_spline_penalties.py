"""Private penalty-construction helpers for spline feature specs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from superglm.features._spline_ranges import derivative_design


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
    standard difference penalty loses that on uneven knots. ``D_m`` is scaled by
    ``hbar**order``, ``hbar`` the basis domain's span over its interval count,
    so the scale stays that of the standard penalty on knots of the same count.
    """
    t = np.asarray(knots, dtype=np.float64)
    d = degree + 1
    n_basis = len(t) - d
    if not 1 <= order <= degree:
        raise ValueError(f"General difference order {order} needs 1 <= order <= degree ({degree}).")
    hbar = (t[n_basis] - t[degree]) / (n_basis - degree)
    D = np.eye(n_basis)
    for k in range(1, order + 1):
        j = np.arange(k, n_basis)
        span = (t[j + d - k] - t[j]) / (d - k)
        D = np.diff(D, axis=0) * (hbar / span)[:, None]
    return D.T @ D


def difference_penalty_for(spec, order: int) -> NDArray:
    """A difference-penalised spline's penalty: general once its knots are not evenly placed.

    Knots placed by the ``"uniform"`` rule keep the standard penalty. Stated
    knots and quantile-placed ones take the general one, which needs
    ``order <= degree``; a higher order keeps the standard penalty.
    """
    if spec._knot_strategy_actual != "uniform" and order <= spec.degree:
        return build_general_difference_penalty(spec._knots, spec.degree, order)
    return build_difference_penalty(spec._n_basis, order)


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
