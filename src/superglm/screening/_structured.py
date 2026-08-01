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

from superglm.screening._arrow import factor_arrow, psd_ranks
from superglm.screening._score_stat import ScreenedPair

_EDF_TOL = 1e-6
_MAX_BISECT = 200

# The most steps one rung's bisection can take.  It halves the log of a
# bracket spanning 1e20 and stops when the two ends are within 1e-12 of each
# other relatively, so it exhausts at ceil(log2(ln(1e20) / ln(1 + 1e-12)))
# steps whatever the data does -- _EDF_TOL usually stops it sooner (measured
# 26 to 28 on the pairs that search), but nothing guarantees that, and a
# ceiling is what lets a caller decide BEFORE the search whether to pay for
# it rather than after.
_MAX_STEPS_PER_RUNG = min(_MAX_BISECT, 46)


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

    A missing ``S_a`` is carried as a zero penalty rather than refused, which
    is what the dense path does with the same input: an unpenalized block has
    no bandwidth to scan, and :func:`structured_ladder` reports it at a
    single rung.  Refusing here would abort the whole sweep over one pair,
    where the contract is a NaN row.
    """
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
        S_a=np.zeros((k_a, k_a)) if S_a is None else np.asarray(S_a, dtype=np.float64),
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


def _unpenalized_blocks(p: SplineCatPair) -> NDArray:
    """The ``(L, k_a + 1, k_a + 1)`` level blocks at ``lambda = 0``."""
    L, k_a = p.dims
    g = k_a + 1
    G = np.empty((L, g, g), dtype=np.float64)
    G[:, :k_a, :k_a] = p.V
    G[:, :k_a, k_a] = p.c
    G[:, k_a, :k_a] = p.c
    G[:, k_a, k_a] = p.m
    return G


def block_ranks(p: SplineCatPair) -> NDArray:
    """The rank of every level block, counted where the two terms BALANCE.

    A block is ``P_q + lambda T`` with both terms PSD, so its null space is
    ``null(P_q) & null(T)`` for every ``lambda > 0`` and its rank is the same
    throughout.  That lets it be counted ONCE, and counted on the one member
    of the family a relative eigenvalue cut can actually resolve: each term
    scaled to unit trace, so neither is judged against the other's magnitude.

    Counting it at a bracket edge instead — which is what reading it off the
    factorization does — loses a whole degree of freedom whenever the two
    terms are far enough apart there.  Measured on a 20-level pair with one
    level weighted 1/100th of the others: the block-rank sum reads 227 at
    each edge where the balanced count and ``numpy.linalg.matrix_rank`` both
    read 228, and the pair's ``edf0`` came back 17.99995 against the dense
    path's 18.99991.  The low edge is not safe either — it under-counted by
    one on the same pair at EQUAL weights, the mirror image of the same
    failure.  Rare levels are the routine case at high cardinality, so this
    is not an edge case; see :mod:`superglm.screening._arrow`.
    """
    _, k_a = p.dims
    tiny = np.finfo(np.float64).tiny
    G = _unpenalized_blocks(p)
    G /= np.maximum(np.einsum("lpp->l", G, optimize=True), tiny)[:, None, None]
    G[:, :k_a, :k_a] += p.S_a / max(float(np.trace(p.S_a)), tiny)
    return psd_ranks(G)


def _pair_arrow(p: SplineCatPair, lam: float, ranks: NDArray | None = None):
    """``K(lambda)`` in arrow form: one ``(k_a + 1)`` block per level.

    Level q's block holds its tensor coefficients beside its own contrast;
    the border holds the intercept and the spline main, the only two things
    every level shares.  ``C``'s spline-main rows are literally ``V``'s
    diagonal blocks — both are ``sum_i w_i A_i A_i'`` restricted to the level
    — so they are taken from the same array rather than reassembled.
    """
    L, k_a = p.dims
    g, r = k_a + 1, 1 + k_a
    G = _unpenalized_blocks(p)
    G[:, :k_a, :k_a] += lam * p.S_a
    E = np.empty((L, r, g), dtype=np.float64)
    E[:, 0, :k_a] = p.c
    E[:, 0, k_a] = p.m
    E[:, 1:, :k_a] = p.V
    E[:, 1:, k_a] = p.c
    return factor_arrow(G, E, p.border, block_ranks=ranks)


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


def _evaluate(
    p: SplineCatPair,
    U_eff: NDArray,
    rank_m: int,
    lam: float,
    ranks: NDArray | None = None,
) -> tuple[float, float]:
    """``(T, edf)`` at one lambda, from ONE arrow factorization.

    ``rank(V_eff + lambda S)`` is Guttman rank additivity on the bordered
    system — ``rank(K) = rank(M) + rank(A)``.  ``ranks`` carries the level
    blocks' contribution, counted once by :func:`block_ranks` because it does
    not depend on ``lambda`` and cannot be resolved reliably at the edges
    where the ladder brackets; the border's own rank comes from this
    factorization.
    """
    L, k_a = p.dims
    f = _pair_arrow(p, lam, block_ranks(p) if ranks is None else ranks)
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
    max_evaluations: int | None = None,
) -> list[ScreenedPair] | None:
    """Score one spline x categorical pair at every budget, structurally.

    Mirrors :func:`penalized_score_statistic_ladder`'s contract — clamp a
    budget outside the bracket to the nearest edge and report the edf
    actually achieved — but every evaluation is an arrow factorization rather
    than a dense one.

    **Whether the ladder searches is not a function of the pair's dimensions,
    so the caller caps it and the decision is taken here.**  A rung whose
    budget falls inside the bracket bisects, and each step of that bisection
    is a fresh arrow factorization where the dense ladder's equivalent is
    ``O(k)`` on a prebuilt pencil.  Whether any rung does depends on ``edf``
    at maximum penalty, which is the dimension of the penalty's null space
    per level: measured at ``L - 1`` for ``ps``, ``bs`` and ``cr`` margins,
    where every rung clamps and the whole ladder is 2 evaluations — but at
    ZERO for ``ns``, whose penalty is full rank, so every rung searches and a
    400-level pair measured 106.  ``max_evaluations`` bounds the arrow
    factorizations this call may spend.  The bracket settles which rungs
    search, and the worst case for those is checked against the ceiling
    BEFORE the first bisection step, so a pair that cannot afford its search
    pays only for the bracket and returns ``None`` — the caller's cue for the
    same NaN row an unaffordable dense pair gets.  ``None`` means unbounded.
    """
    U_eff, rank_m = _profile(p)
    ranks = block_ranks(p)

    if not np.any(p.S_a):
        # No penalty to scan, exactly the predicate the dense ladder applies:
        # one rung, at the block's own achieved rank, with lambda0 = 0.  A
        # zero penalty would otherwise make the bracket below infinite and
        # every rung NaN, since inf * 0 is not a number.
        stat, rank = _evaluate(p, U_eff, rank_m, 0.0, ranks)
        return [ScreenedPair(statistic=stat, edf0=rank, lambda0=0.0) for _ in budgets]

    # The dense path scales its bracket by tr(V_eff)/tr(S); the unprofiled
    # tr(V) is used here because profiling the trace structurally would cost
    # more than the bracket is worth.  The two differ by a few percent, ten
    # orders of magnitude outside the region where edf varies with lambda.
    #
    # The 1e+-10 edges are the dense ladder's, kept identical so a pair the
    # two paths can both score gets the same lambda0.  Nothing in the arrow
    # kernel's numerics is tied to their width any more: rank is counted from
    # a balanced reference (see block_ranks) precisely so that widening them
    # cannot move an edf by a whole degree of freedom.
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    tr_V = float(np.einsum("lpp->", p.V, optimize=True))
    scale = max(tr_V, 1e-300) / max(tr_S, 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale

    stat_lo, edf_lo = _evaluate(p, U_eff, rank_m, lo, ranks)
    stat_hi, edf_hi = _evaluate(p, U_eff, rank_m, hi, ranks)

    searching = sum(1 for b in budgets if edf_hi < float(b) < edf_lo)
    if max_evaluations is not None and 2 + _MAX_STEPS_PER_RUNG * searching > max_evaluations:
        return None

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
            stat, achieved = _evaluate(p, U_eff, rank_m, lam, ranks)
            if abs(achieved - edf0) <= _EDF_TOL:
                break
            if achieved > edf0:
                a = lam
            else:
                b = lam
        out.append(ScreenedPair(statistic=stat, edf0=achieved, lambda0=float(lam)))
    return out
