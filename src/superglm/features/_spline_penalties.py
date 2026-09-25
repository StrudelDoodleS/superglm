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
