"""Numeric-margin sufficient statistics for mixed-type screening.

A numeric covariate enters every v1 probe LINEARLY (``z * contrast`` slopes,
``z1 * z2`` products), so the pair needs no joint grid: z-weighted moments
accumulated over the other margin's cells are the complete sufficient
statistics, at any support size of the numeric side.  Channels: ``s``,
``s*z``, ``w``, ``w*z``, ``w*z**2`` (and the symmetric set for two numerics).
These moments never approximate — but they are not free of the OTHER margin:
an ``(n_g, k)`` menu makes every block here scale with ``k``, so the caller
budgets ``n_g`` and refuses a pair whose blocks it cannot afford rather than
compressing anything.  Exactness is pinned against the dense assembly in
tests/test_interaction_screening.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def numeric_pair_moments(
    codes_g: NDArray,
    n_g: int,
    menu_g: NDArray,
    z: NDArray,
    score: NDArray,
    working_weights: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Moments for probe ``menu_g[codes] * z`` with overlap ``[1 | menu | z]``.

    Returns ``(U, V, C, M, u_m)`` with ``k = menu_g.shape[1]`` probe columns
    and overlap width ``q = 1 + k + 1``.
    """
    codes_g = np.asarray(codes_g, dtype=np.intp)
    menu_g = np.asarray(menu_g, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(working_weights, dtype=np.float64)
    if not (codes_g.shape == z.shape == score.shape == w.shape):
        raise ValueError("codes, z, score, and working weights must share one row dimension")
    if codes_g.size and (int(codes_g.min()) < 0 or int(codes_g.max()) >= n_g):
        raise ValueError("codes_g fall outside [0, n_g)")

    def cell(v):
        return np.bincount(codes_g, weights=v, minlength=n_g)

    s0, s1 = cell(score), cell(score * z)
    w0, w1, w2 = cell(w), cell(w * z), cell(w * z * z)

    k = menu_g.shape[1]
    q = 1 + k + 1
    U = menu_g.T @ s1
    V = menu_g.T @ (menu_g * w2[:, None])

    M = np.empty((q, q), dtype=np.float64)
    sl = slice(1, 1 + k)
    M[0, 0] = w0.sum()
    M[0, sl] = w0 @ menu_g
    M[0, -1] = w1.sum()
    M[sl, 0] = M[0, sl]
    M[-1, 0] = M[0, -1]
    M[sl, sl] = menu_g.T @ (menu_g * w0[:, None])
    M[sl, -1] = menu_g.T @ w1
    M[-1, sl] = M[sl, -1]
    M[-1, -1] = w2.sum()

    C = np.empty((q, k), dtype=np.float64)
    C[0] = menu_g.T @ w1
    C[sl] = menu_g.T @ (menu_g * w1[:, None])
    C[-1] = menu_g.T @ w2

    u_m = np.empty(q, dtype=np.float64)
    u_m[0] = s0.sum()
    u_m[sl] = menu_g.T @ s0
    u_m[-1] = s1.sum()
    return U, V, C, M, u_m


def numeric_numeric_moments(
    z1: NDArray,
    z2: NDArray,
    score: NDArray,
    working_weights: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Moments for probe ``z1 * z2`` with overlap span ``[1 | z1 | z2]``."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    w = np.asarray(working_weights, dtype=np.float64)
    if not (z1.shape == z2.shape == score.shape == w.shape):
        raise ValueError("z1, z2, score, and working weights must share one row dimension")
    p = z1 * z2
    U = np.array([p @ score])
    V = np.array([[(p * p) @ w]])
    ones = np.ones_like(z1)
    span = (ones, z1, z2)
    M = np.array([[(a * b) @ w for b in span] for a in span])
    C = np.array([[(a * p) @ w] for a in span])
    u_m = np.array([a @ score for a in span])
    return U, V, C, M, u_m
