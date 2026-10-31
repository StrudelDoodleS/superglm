"""Reference producers of the dense screening route, kept as this suite's arbiter.

Issue #257 moved the dense screening path onto design factors: ``_pair_factor``
reduces the pair's cell tables to one triangular factor and the ladder reads
``V_eff`` off a block of it, so no Gram is formed.  What is below is the
DEFINITION that factor is graded against: ``numeric_pair_moments`` and
``numeric_numeric_moments`` accumulate the numeric margins' z-weighted moments
over the other margin's cells (channels ``s``, ``s*z``, ``w``, ``w*z``,
``w*z**2``; exact, with blocks scaling with the other margin's menu width),
``pair_score_curvature`` assembles the tensor block's score and Gram from the
cell tables, and ``penalized_score_statistic`` ranks one pair on the first rung
of the ladder.  Nothing in ``src`` calls them; an import guard keeps it so
(``test_no_production_module_imports_a_retired_gram_producer``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.screening._pair_factor import PairFactor
from superglm.screening._score_stat import ScreenedPair, penalized_score_statistic_ladder


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
    if menu_g.ndim != 2 or menu_g.shape[0] != n_g:
        raise ValueError(
            f"menu_g must be (n_g, k) with one row per gridded cell; got shape "
            f"{menu_g.shape} for n_g={n_g}"
        )
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


def pair_score_curvature(
    B_a: NDArray,
    B_b: NDArray,
    S_cell: NDArray,
    W_cell: NDArray,
) -> tuple[NDArray, NDArray]:
    """Score vector and unadjusted curvature of the pair's tensor block.

    With ``X_T`` the row-Kronecker design ``kron(B_a[i_a[r]], B_b[i_b[r]])``:

    ``U = X_T' s  = vec(B_a' S_cell B_b)``
    ``V = X_T' diag(W) X_T``, assembled from ``W_cell`` without forming rows:
    ``V[(p,q),(r,s)] = sum_i B_a[i,p] B_a[i,r] * (sum_j W[i,j] B_b[j,q] B_b[j,s])``

    Flattening is C-order, matching ``np.kron`` column ordering.

    Not on the production path since issue #257 -- see the module docstring for
    what it is for now and why it stays.
    """
    B_a = np.asarray(B_a, dtype=np.float64)
    B_b = np.asarray(B_b, dtype=np.float64)
    k_a, k_b = B_a.shape[1], B_b.shape[1]
    U = (B_a.T @ S_cell @ B_b).reshape(k_a * k_b)
    inner = np.einsum("ij,jq,js->iqs", W_cell, B_b, B_b, optimize=True)
    V = np.einsum("ip,ir,iqs->pqrs", B_a, B_a, inner, optimize=True)
    return U, V.reshape(k_a * k_b, k_a * k_b)


def penalized_score_statistic(
    pair: PairFactor,
    penalty_root: NDArray | None,
    *,
    edf0: float = 4.0,
) -> ScreenedPair:
    """Rank one candidate pair by its penalized efficient-score statistic.

    ``pair`` carries the overlap span the mains model already explains as the
    leading block of its own factor, so the profiling that used to need ``C``,
    ``M`` and the overlap's own score is a slice of a triangular matrix here
    rather than a difference of two Grams.  With ``penalty_root`` absent or
    empty the statistic reduces to the unpenalized ``U_eff' V_eff^+ U_eff`` and
    ``lambda0`` is reported as 0.

    Scoring a ladder of budgets? Use
    :func:`penalized_score_statistic_ladder`, which shares one decomposition
    across every rung instead of rebuilding it per call.
    """
    return penalized_score_statistic_ladder(pair, penalty_root, budgets=(float(edf0),))[0]
