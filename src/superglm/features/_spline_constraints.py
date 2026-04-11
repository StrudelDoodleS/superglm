"""Private constraint helpers for spline feature specs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline as BSpl

from superglm.types import LinearConstraintSet


def build_monotone_difference_constraints(
    n_basis: int, monotone: str | None
) -> LinearConstraintSet:
    """Build first-difference monotonicity constraints on raw spline coefficients."""
    D = np.diff(np.eye(n_basis), axis=0)
    if monotone == "decreasing":
        D = -D
    return LinearConstraintSet(A=D, b=np.zeros(n_basis - 1))


def build_natural_constraint_null_space(
    knots: NDArray,
    degree: int,
    *,
    lo: float,
    hi: float,
) -> NDArray:
    """Compute the null space of natural-boundary spline constraints."""
    n_basis = len(knots) - degree - 1
    C = np.zeros((2, n_basis))
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spl = BSpl(knots, c, degree)
        C[0, j] = spl(lo, nu=2)
        C[1, j] = spl(hi, nu=2)
    Q, _ = np.linalg.qr(C.T, mode="complete")
    return Q[:, 2:]


__all__ = ["build_monotone_difference_constraints", "build_natural_constraint_null_space"]
