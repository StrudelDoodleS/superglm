"""Private cardinal cubic regression spline helpers."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


def _second_differences(knots: NDArray) -> NDArray:
    """The ``(K - 2, K)`` second divided differences at the knots."""
    K = len(knots)
    h = np.diff(knots)
    B_d = np.zeros((K - 2, K))
    for i in range(K - 2):
        B_d[i, i] = 1.0 / h[i]
        B_d[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        B_d[i, i + 2] = 1.0 / h[i + 1]
    return B_d


def structural_cr_penalty(knots: NDArray) -> NDArray:
    """The cardinal penalty's null space without its knot-spacing spread.

    The penalty is ``B_d' D^{-1} B_d`` with ``D`` positive definite, so its
    null space is that of ``B_d``: the lines through the knot values. Each
    row of ``B_d`` scaled to unit norm keeps that null space (a positive
    reweighting of the rank-one blocks ``b_i b_i'``) and drops the ``h**-3``
    spread, as Wood, Pya and Saefken's balanced penalty does (JASA 111, 2016,
    section 3.1.1).
    """
    B_d = _second_differences(knots)
    unit = B_d / np.linalg.norm(B_d, axis=1)[:, None]
    return unit.T @ unit


def _second_derivative_map(knots: NDArray) -> tuple[NDArray, NDArray]:
    """``B_d`` and ``D^{-1} B_d``, the map from knot values to second derivatives at the knots."""
    K = len(knots)
    h = np.diff(knots)

    B_d = _second_differences(knots)

    D = np.zeros((K - 2, K - 2))
    for i in range(K - 2):
        D[i, i] = (h[i] + h[i + 1]) / 3.0
        if i < K - 3:
            D[i, i + 1] = h[i + 1] / 6.0
            D[i + 1, i] = h[i + 1] / 6.0
    return B_d, np.linalg.solve(D, B_d)


def build_cr_penalty_matrices(knots: NDArray) -> tuple[NDArray, NDArray]:
    """Build the cardinal CR second-derivative map and penalty matrix.

    The penalty is ``integral f''**2`` over ``u = (x - knots[0]) / hbar``,
    ``hbar`` the mean knot interval, which is ``hbar**3`` times the integral
    over ``x`` and so independent of the column's units, as
    ``build_integrated_derivative_penalty`` takes it for the B-spline kinds.
    The second-derivative map stays in the units of ``x``: the basis needs it.
    """
    K = len(knots)
    _, D_inv_Bd = _second_derivative_map(knots)
    hbar = (knots[-1] - knots[0]) / (K - 1)
    B_u, D_inv_Bu = _second_derivative_map((knots - knots[0]) / hbar)
    cr_S = B_u.T @ D_inv_Bu

    cr_M = np.zeros((K, K))
    cr_M[1 : K - 1, :] = D_inv_Bd
    return cr_M, cr_S


def cardinal_boundary_slopes(
    knots: NDArray, M: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return basis value and slope at the boundary knots for linear extrapolation."""
    K = len(knots)
    h = np.diff(knots)

    basis_lo = np.zeros(K)
    basis_lo[0] = 1.0
    slope_lo = np.zeros(K)
    slope_lo[0] = -1.0 / h[0]
    slope_lo[1] = 1.0 / h[0]
    slope_lo -= (h[0] / 6.0) * M[1, :]

    basis_hi = np.zeros(K)
    basis_hi[K - 1] = 1.0
    slope_hi = np.zeros(K)
    slope_hi[K - 2] = -1.0 / h[K - 2]
    slope_hi[K - 1] = 1.0 / h[K - 2]
    slope_hi += (h[K - 2] / 6.0) * M[K - 2, :]

    return basis_lo, slope_lo, basis_hi, slope_hi


def eval_cardinal_basis(x: NDArray, knots: NDArray, M: NDArray) -> sp.csr_matrix:
    """Vectorised cardinal cubic regression spline evaluation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    K = len(knots)
    h = np.diff(knots)
    n = len(x)

    j = np.searchsorted(knots, x, side="right") - 1
    j = np.clip(j, 0, K - 2)

    hj = h[j]
    t = (x - knots[j]) / hj

    X = np.zeros((n, K))
    rows = np.arange(n)
    X[rows, j] = 1.0 - t
    X[rows, j + 1] = t

    c1 = hj**2 * ((1.0 - t) ** 3 - (1.0 - t)) / 6.0
    c2 = hj**2 * (t**3 - t) / 6.0
    X += c1[:, None] * M[j, :] + c2[:, None] * M[j + 1, :]

    return sp.csr_matrix(X)


def linear_tail_cardinal_basis(
    x: NDArray,
    *,
    lo: float,
    hi: float,
    knots: NDArray,
    M: NDArray,
) -> sp.csr_matrix:
    """Evaluate cardinal basis with natural-spline linear tails outside range."""
    x = np.asarray(x, dtype=np.float64).ravel()
    lo_mask = x < lo
    hi_mask = x > hi
    mid_mask = ~(lo_mask | hi_mask)

    K = len(knots)
    X = np.zeros((len(x), K))

    if np.any(mid_mask):
        X[mid_mask] = eval_cardinal_basis(x[mid_mask], knots, M).toarray()

    if np.any(lo_mask) or np.any(hi_mask):
        basis_lo, slope_lo, basis_hi, slope_hi = cardinal_boundary_slopes(knots, M)
        if np.any(lo_mask):
            X[lo_mask] = basis_lo + (x[lo_mask, None] - lo) * slope_lo
        if np.any(hi_mask):
            X[hi_mask] = basis_hi + (x[hi_mask, None] - hi) * slope_hi

    return sp.csr_matrix(X)


__all__ = [
    "build_cr_penalty_matrices",
    "cardinal_boundary_slopes",
    "eval_cardinal_basis",
    "linear_tail_cardinal_basis",
]
