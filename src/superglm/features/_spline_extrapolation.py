"""Private extrapolation helpers for spline bases."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.interpolate import BSpline as BSpl


def basis_value_and_slope_at(
    x0: float,
    *,
    knots: NDArray,
    degree: int,
    n_basis: int,
) -> tuple[NDArray, NDArray]:
    """Return the raw basis row and its first derivative at ``x0``."""
    basis = BSpl.design_matrix(
        np.array([x0], dtype=np.float64), knots, degree, extrapolate=False
    ).toarray()[0]
    slope = np.zeros(n_basis)
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        slope[j] = BSpl(knots, c, degree)(x0, nu=1)
    return basis, slope


def boundary_linear_rows(
    *,
    knots: NDArray,
    degree: int,
    n_basis: int,
    lo: float,
    hi: float,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute basis value/slope rows for linear continuation at the boundaries."""
    basis_lo, slope_lo = basis_value_and_slope_at(lo, knots=knots, degree=degree, n_basis=n_basis)
    basis_hi, slope_hi = basis_value_and_slope_at(hi, knots=knots, degree=degree, n_basis=n_basis)
    return basis_lo, slope_lo, basis_hi, slope_hi


def linear_tail_basis_matrix(
    x: NDArray,
    *,
    knots: NDArray,
    degree: int,
    n_basis: int,
    lo: float,
    hi: float,
    basis_lo: NDArray,
    slope_lo: NDArray,
    basis_hi: NDArray,
    slope_hi: NDArray,
) -> sp.csr_matrix:
    """Evaluate the raw basis with explicit linear continuation outside the fit range."""
    x = np.asarray(x, dtype=np.float64).ravel()
    lo_mask = x < lo
    hi_mask = x > hi
    mid_mask = ~(lo_mask | hi_mask)

    rows = np.zeros((len(x), n_basis))
    if np.any(mid_mask):
        rows[mid_mask] = BSpl.design_matrix(x[mid_mask], knots, degree, extrapolate=False).toarray()

    if np.any(lo_mask):
        rows[lo_mask] = basis_lo + (x[lo_mask, None] - lo) * slope_lo
    if np.any(hi_mask):
        rows[hi_mask] = basis_hi + (x[hi_mask, None] - hi) * slope_hi

    return sp.csr_matrix(rows)


__all__ = ["basis_value_and_slope_at", "boundary_linear_rows", "linear_tail_basis_matrix"]
