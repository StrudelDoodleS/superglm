"""Private penalty-construction helpers for spline feature specs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline as BSpl


def build_difference_penalty(n_basis: int, order: int) -> NDArray:
    """Difference-penalty matrix from order-`m` finite differences."""
    if order >= n_basis:
        raise ValueError(
            f"Difference order {order} >= n_basis {n_basis}. Increase n_knots or reduce m."
        )
    Dm = np.diff(np.eye(n_basis), n=order, axis=0)
    return Dm.T @ Dm


def build_integrated_derivative_penalty(knots: NDArray, degree: int, order: int) -> NDArray:
    """Integrated squared derivative penalty via Gauss-Legendre quadrature."""
    if order > degree:
        raise ValueError(
            f"Derivative order {order} > spline degree {degree}. "
            "Integrated-derivative penalty requires order <= degree."
        )

    K = len(knots) - degree - 1
    unique_knots = np.unique(knots)
    omega = np.zeros((K, K))
    n_quad = max(order + 1, degree)

    for a, b in zip(unique_knots[:-1], unique_knots[1:]):
        if b - a < 1e-15:
            continue
        xi, wi = np.polynomial.legendre.leggauss(n_quad)
        x_q = 0.5 * (b - a) * xi + 0.5 * (a + b)
        w_q = 0.5 * (b - a) * wi

        Dm_q = np.zeros((len(x_q), K))
        for j in range(K):
            c = np.zeros(K)
            c[j] = 1.0
            spl = BSpl(knots, c, degree)
            Dm_q[:, j] = spl(x_q, nu=order)

        omega += Dm_q.T @ (Dm_q * w_q[:, None])

    return omega


__all__ = ["build_difference_penalty", "build_integrated_derivative_penalty"]
