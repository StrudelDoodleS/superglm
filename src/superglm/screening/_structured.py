"""Structured screening kernel for a spline x categorical candidate pair.

The dense path assembles the pair's ``(k, k)`` curvature with ``k = k_a * L``
and factorizes it, which is cubic in the level count and quadratic in memory.
Both are pure waste here, because with the categorical margin's treatment
menu the pair's bordered system

    K(lambda) = [[V + lambda S, C'],
                 [C,            M ]]

is EXACTLY a block-arrow matrix once its variables are grouped by level:

  * ``V`` is block-diagonal — levels have disjoint row support, so no tensor
    column of one level meets any column of another;
  * ``S = kron(S_a, I)`` is block-diagonal on the same grouping — one copy of
    the spline penalty per level, which is the varying-coefficient layout;
  * ``M``'s categorical-main block is diagonal, and ``C``'s categorical-main
    rows are level-local: level q's indicator meets level q's tensor columns
    and nothing else.

What is left coupling the levels is only the intercept and the spline main —
a border of ``1 + k_a`` columns, independent of ``L``.  So the pair is
``L`` blocks of size ``k_a + 1`` around a border of size ``1 + k_a``, and
:mod:`superglm.screening._arrow` factors that in time and memory LINEAR in
``L``.  The dense path's ``V`` alone is 2.4 TB at fifty thousand levels; the
same pair's blocks are 48 MB.

The two quantities the ladder needs both read off one such factorization:

    T(lambda)   = U_eff' A(lambda)^-1 U_eff
    edf(lambda) = rank(A) - lambda * tr(A(lambda)^-1 S)

the second because ``V_eff = A - lambda S`` identically, which trades a trace
against a DENSE ``V_eff`` for one against a block-diagonal ``S`` — and a
block-diagonal ``S`` needs only the diagonal blocks of the inverse, which the
arrow factorization already has.

This path is entered only where the dense path is refused, so every pair the
dense path can score is still scored by it, bit for bit.  The two agree to
the tolerance pinned in tests/test_structured_screening.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.screening._arrow import factor_arrow
from superglm.screening._score_stat import ScreenedPair

_EDF_TOL = 1e-6
_MAX_BISECT = 200


@dataclass(frozen=True)
class SplineCatPair:
    """Level-blocked moments of one spline x categorical pair.

    Every array is linear in the level count ``L``.  ``V``/``U``/``c``/``m``
    are per-level; ``border``/``u_border`` are the level-independent border,
    the overlap span's ``[intercept | spline main]`` corner.
    """

    V: NDArray  # (L, k_a, k_a) tensor curvature per level
    U: NDArray  # (L, k_a)      tensor score per level
    c: NDArray  # (L, k_a)      level indicator against level q's tensor columns
    m: NDArray  # (L,)          weight in each level
    S_a: NDArray  # (k_a, k_a)  the spline penalty, shared by every level
    border: NDArray  # (r, r)        border block of M, r = 1 + k_a
    u_border: NDArray  # (r,)
    u_cat: NDArray  # (L,)

    @property
    def dims(self) -> tuple[int, int]:
        return self.U.shape  # (L, k_a)


def spline_cat_moments(
    B_a: NDArray,
    S_a: NDArray,
    S_cell: NDArray,
    W_cell: NDArray,
    level_rows: NDArray,
) -> SplineCatPair:
    """Assemble the level-blocked moments straight from the pair's cell tables.

    ``level_rows[q]`` is the cell-table column the q-th contrast indicates —
    the treatment menu is one-hot with a zeroed base row, so selecting that
    column IS multiplying by the menu, and the ``(L, L-1)`` menu is never
    built.  Nor is the dense path's ``(n_a, L, L)`` curvature intermediate,
    which is diagonal in its last two axes: measured at 380,880 doubles
    carrying 690 nonzeros for a 24-level factor.
    """
    if S_a is None:
        raise ValueError("the spline margin of a spline_cat pair must carry a penalty")
    B_a = np.asarray(B_a, dtype=np.float64)
    n_a, k_a = B_a.shape
    level_rows = np.asarray(level_rows, dtype=np.intp)
    Wq = W_cell[:, level_rows]
    Sq = S_cell[:, level_rows]

    # One GEMM for every level's k_a x k_a curvature: the outer products of
    # the spline menu are level-independent, so they are formed once and
    # contracted against each level's weights.
    AA = (B_a[:, :, None] * B_a[:, None, :]).reshape(n_a, k_a * k_a)
    V = (Wq.T @ AA).reshape(-1, k_a, k_a)

    w_row = W_cell.sum(axis=1)
    s_row = S_cell.sum(axis=1)
    r = 1 + k_a
    border = np.empty((r, r), dtype=np.float64)
    border[0, 0] = W_cell.sum()
    border[0, 1:] = w_row @ B_a
    border[1:, 0] = border[0, 1:]
    border[1:, 1:] = B_a.T @ (B_a * w_row[:, None])

    u_border = np.empty(r, dtype=np.float64)
    u_border[0] = S_cell.sum()
    u_border[1:] = B_a.T @ s_row

    return SplineCatPair(
        V=V,
        U=(B_a.T @ Sq).T,
        c=(B_a.T @ Wq).T,
        m=Wq.sum(axis=0),
        S_a=np.asarray(S_a, dtype=np.float64),
        border=border,
        u_border=u_border,
        u_cat=Sq.sum(axis=0),
    )


def _overlap_arrow(p: SplineCatPair):
    """The overlap curvature ``M`` in arrow form: one scalar block per level.

    ``M``'s categorical-main block is diagonal and its border is the same
    ``[intercept | spline main]`` corner, so ``M`` is an arrow matrix with
    ``g = 1``.  Profiling therefore costs O(L) rather than the O(L^3) a dense
    ``M^-1`` would.
    """
    L, k_a = p.dims
    E = np.empty((L, 1 + k_a, 1), dtype=np.float64)
    E[:, 0, 0] = p.m
    E[:, 1:, 0] = p.c
    return factor_arrow(p.m.reshape(L, 1, 1), E, p.border)


def _pair_arrow(p: SplineCatPair, lam: float):
    """``K(lambda)`` in arrow form: one ``(k_a + 1)`` block per level.

    Level q's block holds its tensor coefficients beside its own contrast;
    the border holds the intercept and the spline main, the only two things
    every level shares.  ``C``'s spline-main rows are literally ``V``'s
    diagonal blocks — both are ``sum_i w_i A_i A_i'`` restricted to the level
    — so they are taken from the same array rather than reassembled.
    """
    L, k_a = p.dims
    g, r = k_a + 1, 1 + k_a
    G = np.empty((L, g, g), dtype=np.float64)
    G[:, :k_a, :k_a] = p.V + lam * p.S_a
    G[:, :k_a, k_a] = p.c
    G[:, k_a, :k_a] = p.c
    G[:, k_a, k_a] = p.m
    E = np.empty((L, r, g), dtype=np.float64)
    E[:, 0, :k_a] = p.c
    E[:, 0, k_a] = p.m
    E[:, 1:, :k_a] = p.V
    E[:, 1:, k_a] = p.c
    return factor_arrow(G, E, p.border)


def _profile(p: SplineCatPair) -> tuple[NDArray, int]:
    """``(U_eff, rank(M))`` — the whole lambda-independent half of the work.

    ``U_eff = U - C' M^-1 u_m``.  Column ``(p, q)`` of ``C`` is nonzero in
    exactly three places — the intercept row, the spline-main rows, and level
    q's own contrast row — so the contraction never touches a level other
    than its own.  ``M`` depends on no lambda, so this is computed once and
    every rung of the ladder reuses it.
    """
    f = _overlap_arrow(p)
    L, _ = p.dims
    w_cat, w_border = f.solve(p.u_cat.reshape(L, 1), p.u_border)
    U_eff = p.U - (p.c * (w_border[0] + w_cat.reshape(L))[:, None] + p.V @ w_border[1:])
    return U_eff, f.rank


def _evaluate(p: SplineCatPair, U_eff: NDArray, rank_m: int, lam: float) -> tuple[float, float]:
    """``(T, edf)`` at one lambda, from ONE arrow factorization.

    ``rank(V_eff + lambda S)`` is Guttman rank additivity on the bordered
    system — ``rank(K) = rank(M) + rank(A)`` — and both ranks come free from
    the arrow factorizations, which already counted the directions they kept.
    """
    L, k_a = p.dims
    f = _pair_arrow(p, lam)
    b = np.zeros((L, k_a + 1), dtype=np.float64)
    b[:, :k_a] = U_eff
    x, _ = f.solve(b, np.zeros(1 + k_a, dtype=np.float64))
    T = float(np.sum(U_eff * x[:, :k_a]))
    blocks = f.diag_blocks()[:, :k_a, :k_a]
    edf = (f.rank - rank_m) - lam * float(np.einsum("lpr,rp->", blocks, p.S_a, optimize=True))
    return T, edf


def structured_ladder(
    p: SplineCatPair,
    *,
    budgets: tuple[float, ...] = (4.0,),
) -> list[ScreenedPair]:
    """Score one spline x categorical pair at every budget, structurally.

    Mirrors :func:`penalized_score_statistic_ladder`'s contract — clamp a
    budget outside the bracket to the nearest edge and report the edf
    actually achieved — but every evaluation is an arrow factorization rather
    than a dense one.

    A wide factor never searches: ``kron(S_a, I)`` has a null space of about
    one dimension per level, so ``edf`` at maximum penalty already exceeds
    every budget on the ladder and all rungs clamp to the same edge.  The
    search below therefore runs only for factors narrow enough that the dense
    path would have taken the pair anyway.
    """
    U_eff, rank_m = _profile(p)
    # The dense path scales its bracket by tr(V_eff)/tr(S); the unprofiled
    # tr(V) is used here because profiling the trace structurally would cost
    # more than the bracket is worth.  The two differ by a few percent, ten
    # orders of magnitude outside the region where edf varies with lambda.
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    tr_V = float(np.einsum("lpp->", p.V, optimize=True))
    scale = max(tr_V, 1e-300) / max(tr_S, 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale

    stat_lo, edf_lo = _evaluate(p, U_eff, rank_m, lo)
    stat_hi, edf_hi = _evaluate(p, U_eff, rank_m, hi)

    out: list[ScreenedPair] = []
    for budget in budgets:
        edf0 = float(budget)
        if not edf_hi < edf0 < edf_lo:
            lam = lo if edf0 >= edf_lo else hi
            stat, achieved = (stat_lo, edf_lo) if lam == lo else (stat_hi, edf_hi)
            out.append(ScreenedPair(statistic=stat, edf0=achieved, lambda0=float(lam)))
            continue
        a, b = lo, hi
        lam, stat, achieved = hi, stat_hi, edf_hi
        for _ in range(_MAX_BISECT):
            if b <= a * (1.0 + 1e-12):
                break
            lam = float(np.sqrt(a * b))
            stat, achieved = _evaluate(p, U_eff, rank_m, lam)
            if abs(achieved - edf0) <= _EDF_TOL:
                break
            if achieved > edf0:
                a = lam
            else:
                b = lam
        out.append(ScreenedPair(statistic=stat, edf0=achieved, lambda0=float(lam)))
    return out
