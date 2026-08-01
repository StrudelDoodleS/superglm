"""Penalized efficient-score statistic for one candidate pair.

Given the pair's cell-assembled score ``U`` and curvature ``V`` (Task 1), the
overlap cross-moments ``C``/``M`` against the span the mains already fit, and
the pair's tensor penalty ``S``, the statistic is

    T = U_eff' (V_eff + lambda0 * S)^{-1} U_eff

with the efficient-score adjustments ``U_eff = U - C' M^{-1} u_m`` and
``V_eff = V - C' M^{-1} C``, and ``lambda0`` chosen so the smooth is compared
at a fixed screening complexity: ``tr((V_eff + lambda0 S)^{-1} V_eff) = edf0``.
Fixing the effective degrees of freedom across pairs makes raw ``T`` values
comparable regardless of each pair's basis size or penalty scaling — at a
COMMON budget; across different budgets compare the normalized ``z`` the
ladder scan reports, never raw ``T``.

Ranking-only: calibration is by confirmatory refit, never by this number.

**How lambda0 is found.** Both quantities the search needs are closed forms in
one simultaneous diagonalization of the pencil ``(V_eff, S)``.  Whitening by
``G = V_eff + S`` and diagonalizing ``V_eff`` in that basis gives ``B`` with
``B' V_eff B = diag(a)`` and ``B' S B = diag(1 - a)``, and then

    edf(lambda) = sum_j a_j / (a_j + lambda * (1 - a_j))
    T(lambda)   = sum_j u_j^2 / (a_j + lambda * (1 - a_j)),  u = B' U_eff

so every subsequent lambda costs O(k) rather than a fresh O(k^3) solve.  The
decomposition depends on neither ``lambda`` nor ``edf0``, so ONE of them serves
an entire ladder of budgets — which is why ``penalized_score_statistic_ladder``
exists and why callers sweeping a ladder should prefer it.

``G`` is the right thing to whiten by, rather than ``V_eff``: where ``V_eff``
is singular but ``V_eff + lambda S`` is not, those directions still contribute
to ``edf``, and whitening by ``V_eff`` alone silently drops them.  The common
null space of both contributes nothing to either sum and is discarded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.screening._arrow import psd_ranks

_EDF_TOL = 1e-6
_MAX_BISECT = 200

# Below this share of a matrix's own largest eigenvalue a direction is treated
# as ABSENT rather than merely small.  It is deliberately a round-off floor and
# not a smallness heuristic, and it is pinned to what the fallback solver here
# actually inverts with: ``numpy.linalg.pinv``'s default cut, measured at 1e-15
# relative on numpy 2.4.2 by bisecting ``diag(1, x, 0)`` -- the pinv trace flips
# between x = 1e-14 and x = 1e-15.  Matching it buys an invariant: on the
# pseudo-inverse branch a direction this module COUNTS is exactly a direction
# the solve RESOLVES.
#
# It was 1e-12, which is four orders above round-off and therefore a claim
# about smallness rather than absence.  That silently deleted identifiable
# modes: with ``V = S = diag(1, 1e-13, 0)`` and ``U = (0, sqrt(1e-13), 0)`` the
# whitening below discarded the 1e-13 direction -- which carries a genuine
# ``a = 0.5`` and ALL of ``U``'s mass -- and the ladder returned
# ``statistic 0, lambda0 1e-10`` where the direct pseudo-inverse ladder
# resolves ``lambda0 1, statistic 0.5`` at the same edf.
_RCOND = 1e-15


@dataclass(frozen=True)
class ScreenedPair:
    """Ranking output for one candidate pair."""

    statistic: float
    edf0: float
    lambda0: float


@dataclass(frozen=True)
class _Pencil:
    """Simultaneous diagonalization of ``(V_eff, S)`` with ``U_eff`` rotated in.

    ``a`` holds ``V_eff``'s eigenvalues in the ``G``-whitened basis and ``1 - a``
    holds ``S``'s, since the two sum to the identity there by construction.
    """

    a: NDArray
    u: NDArray


def _solve_psd(A: NDArray, B: NDArray) -> NDArray:
    """Solve A X = B for symmetric PSD A, falling back to a pseudo-inverse."""
    try:
        factor = scipy.linalg.cho_factor(A, check_finite=False)
        return scipy.linalg.cho_solve(factor, B, check_finite=False)
    except scipy.linalg.LinAlgError:
        return np.linalg.pinv(A, hermitian=True) @ B


def _edge(V: NDArray, S: NDArray | None, lam: float) -> tuple[float, Callable[[NDArray], NDArray]]:
    """Factor ``V + lam * S`` ONCE; return its edf and a solver against it.

    The bracket and the clamped rungs ask the same two matrices for both an
    edf and a statistic.  Answering each from its own factorization repeated
    the expensive half three times over: the edf needs ``A^-1 V``, a solve
    with k right-hand sides, which costs about 3x the factorization it rides
    on -- measured at 58% of a wide pair's total time against 17% for the
    factorizations themselves.  One factorization per edge, reused, removes
    both duplicates.

    ``S = None`` is the unpenalized block, where ``A`` IS ``V`` and the edf is
    its RANK — so it is counted, not traced.  ``tr(A^-1 V)`` reports the rank
    only when ``A^-1`` is a pseudo-inverse; where ``cho_factor`` succeeds it
    reports ``k``, and a barely positive-definite block is exactly the one
    ``cho_factor`` is entitled to accept, since such a block is mathematically
    PD.  Whether it accepts is then decided by rounding alone: on 200
    replicates of one 210-level ``numeric_cat`` layout with a single singleton
    level — a level whose numeric margin is constant is exactly collinear with
    the profiled-out span, so every replicate's block has rank 208 of 209 —
    ``cho_factor`` succeeded on 76 and the reported edf came back 209 on those
    and 208 on the other 124.  ``numpy.linalg.matrix_rank`` read 208 on all
    200.  Through the public screen the same layout reported 209 on 3 of 12
    seeds, moving ``z`` by +0.050; at 4 df of 5 the same flip moved it by
    +0.29, since ``z`` divides by ``sqrt(2 * edf0)``.
    :func:`superglm.screening._arrow.psd_ranks` is the counter, at this
    module's ``_RCOND`` rather than the arrow kernel's own: same RULE — count
    eigenvalues, respect their sign — at a cut each path can justify, the
    arrow kernel counting on a deliberately BALANCED reference where 1e-12
    means something and this path counting on a raw profiled block where the
    fallback solver's cut does.

    The cut here was 1e-12 when this counting first landed, on the argument
    that matching the module elsewhere beat matching the previous dense
    behaviour.  A later finding on the whitening path retired that argument:
    at 1e-12 an identifiable mode whose only fault is being small beside
    another direction is deleted outright, which is a worse failure than the
    one being fixed.  Pinning ``_RCOND`` to round-off instead settles both,
    and it makes the count agree with the fallback ``pinv`` solve exactly, so
    a direction counted here is one that solve resolves.  The primary fix is
    untouched by that: the modes #199 is about sit at 1e-16 relative — over
    the 20-replicate layout in the regression test the largest dropped one is
    4.6e-16 and the smallest kept 1.2e-02 — so they are dropped by any
    round-off cut, and the spectrum is bimodal enough that no measured block
    lands near either candidate.

    One difference from ``pinv`` is kept on purpose: it scores directions by
    ``|lambda|``, so it counted a curvature of -1e-11 as a degree of freedom
    and inverted it, where ``psd_ranks`` reads the sign and drops it.  On a
    block formed by subtraction small negative eigenvalues are ordinary —
    -4.042e-15 measured on the worked freMTPL2 ``cat_cat`` block — and a
    negative curvature is not a degree of freedom.

    The count is NOT free: eigenvalues cost more than the k-right-hand-side
    trace they replace, by 2.13x, 1.60x, 1.28x, 1.19x and 1.42x at k = 100,
    209, 400, 800 and 1600 (single-threaded).  Against the whole pair it is
    much less -- 1.056x end to end on the widest unpenalized pair the cubic
    budget admits, see the calibration comment in
    :mod:`superglm.model.screening_ops`.  This is a correctness price, paid on
    ONE rung of the one ladder that has no bandwidth to scan.

    Only the edf moves.  The statistic is ``U_eff' A^-1 U_eff``, and ``U_eff``
    is orthogonal to ``V_eff``'s null space by construction — a direction the
    overlap absorbs carries no profiled score — so the Cholesky and
    pseudo-inverse solves answer it the same: over the 76 affected replicates
    above, ``|T_chol - T_pinv| / |T_pinv|`` had median 3.9e-16 and max 2.9e-15.
    ``apply`` is deliberately NOT rank-limited to match.  With ``_RCOND`` at
    ``pinv``'s own cut the two now agree on the fallback branch by
    construction, so the only remaining gap is the sign case above — a
    direction ``pinv`` inverts and this does not count — which is the
    difference being sought rather than a defect.  On the Cholesky branch the
    solve makes no rank decision at all, which is exactly why the edf is
    counted separately here.
    """
    A = V if S is None else V + lam * S
    try:
        factor = scipy.linalg.cho_factor(A, check_finite=False)
        apply = partial(scipy.linalg.cho_solve, factor, check_finite=False)
    except scipy.linalg.LinAlgError:
        apply = np.linalg.pinv(A, hermitian=True).__matmul__
    if S is None:
        return float(psd_ranks(V, _RCOND)), apply
    return float(np.trace(apply(V))), apply


def _build_pencil(V: NDArray, S: NDArray, U: NDArray) -> _Pencil:
    """Diagonalize ``(V, S)`` simultaneously and rotate ``U`` into that basis.

    Prefers the generalized symmetric-definite driver, which does the whole
    reduction in one pass; falls back to explicit whitening when ``G`` is
    singular, i.e. when the two share a null space.  That common null space
    contributes to neither sum, so discarding it is exact.

    The whitening cut is relative to ``G``'s own largest eigenvalue, with
    only a floor against a matrix that is genuinely zero.  An absolute floor
    would make the statistic depend on the units the curvature is carried in:
    with an absolute 1.0, ``V = 1e-13 diag(1, 2, 0)`` against
    ``S = 1e-13 diag(2, 1, 0)`` classified every identifiable direction as
    null and returned ``statistic 0, edf0 0`` where the identically scaled
    problem at a relative cut returned ``2/3`` and ``1``.

    Relative is necessary but not sufficient: the cut must also sit at
    ROUND-OFF, because what it is entitled to discard is the common null
    space and nothing else.  At 1e-12 it discarded identifiable modes whose
    only fault was being small beside another direction -- see ``_RCOND``,
    where the case that forced this is recorded.  A mode surviving the cut
    is whitened by ``1 / sqrt(w)``, so a cut this low does amplify a direction
    it keeps; that is the right trade, because keeping a noise direction costs
    at most one spurious degree of freedom in ``edf`` while dropping a real
    one zeroes the statistic outright.
    """
    G = 0.5 * (V + S + (V + S).T)
    try:
        a, basis = scipy.linalg.eigh(V, G, check_finite=False)
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        w, Q = np.linalg.eigh(G)
        top = float(w.max()) if w.size else 0.0
        keep = w > _RCOND * max(top, np.finfo(np.float64).tiny)
        if not np.any(keep):
            return _Pencil(a=np.zeros(0), u=np.zeros(0))
        whiten = Q[:, keep] / np.sqrt(w[keep])
        Vt = whiten.T @ V @ whiten
        a, R = np.linalg.eigh(0.5 * (Vt + Vt.T))
        basis = whiten @ R
    # a and 1 - a are both variance shares in G-space, so they live in [0, 1];
    # the clip only absorbs round-off at the ends.
    a = np.clip(a, 0.0, 1.0)
    return _Pencil(a=a, u=basis.T @ U)


def _pencil_edf(p: _Pencil, lam: float) -> float:
    den = p.a + lam * (1.0 - p.a)
    ok = den > 0.0
    return float(np.sum(p.a[ok] / den[ok]))


def _pencil_stat(p: _Pencil, lam: float) -> float:
    den = p.a + lam * (1.0 - p.a)
    ok = den > 0.0
    return float(np.sum(p.u[ok] ** 2 / den[ok]))


def _lambda_for_edf(p: _Pencil, edf0: float, scale: float) -> float:
    """Smallest-error ``lambda`` hitting ``edf0``, clamped to the bracket edges.

    ``edf(lambda)`` decreases monotonically from ``rank(V_eff)`` toward the
    dimension of the penalty null space, so a target outside the bracket is
    unreachable; clamping to the nearest edge keeps the pair in the table
    rather than failing it, and the achieved value is reported so a caller can
    see the budget was not met.
    """
    lo, hi = 1e-10 * scale, 1e10 * scale
    if _pencil_edf(p, lo) <= edf0:
        return lo
    if _pencil_edf(p, hi) >= edf0:
        return hi
    lam = lo
    for _ in range(_MAX_BISECT):
        if hi <= lo * (1.0 + 1e-12):
            break  # bracket exhausted at float resolution; nearest lam wins
        lam = float(np.sqrt(lo * hi))
        achieved = _pencil_edf(p, lam)
        if abs(achieved - edf0) <= _EDF_TOL:
            break
        if achieved > edf0:
            lo = lam
        else:
            hi = lam
    return lam


def penalized_score_statistic_ladder(
    U: NDArray,
    V: NDArray,
    C: NDArray | None = None,
    M: NDArray | None = None,
    S_ti: NDArray | None = None,
    *,
    budgets: tuple[float, ...] = (4.0,),
    U_nuisance: NDArray | None = None,
) -> list[ScreenedPair]:
    """Score one pair at every budget in ``budgets``, sharing one decomposition.

    Equivalent to calling :func:`penalized_score_statistic` once per budget,
    but the pencil that makes ``edf`` and ``T`` closed forms depends on neither
    ``lambda`` nor ``edf0`` — so an entire ladder costs one decomposition
    instead of one per rung, each of which previously also paid for its own
    bisection.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    V = 0.5 * (V + V.T)

    if (C is None) != (M is None):
        raise ValueError("C and M profile the overlap together; supply both or neither")
    if C is not None:
        C = np.asarray(C, dtype=np.float64)
        MinvC = _solve_psd(np.asarray(M, dtype=np.float64), C)
        V = V - C.T @ MinvC
        V = 0.5 * (V + V.T)
        if U_nuisance is not None:
            U = U - MinvC.T @ np.asarray(U_nuisance, dtype=np.float64)

    if S_ti is None or not np.any(S_ti):
        # No penalty to scan: ONE factorization of the block answers the
        # statistic, and the achieved rank is COUNTED beside it rather than
        # read off that factorization -- see _edge on why the trace cannot
        # answer it.
        rank, apply = _edge(V, None, 0.0)
        T = float(U @ apply(U))
        return [ScreenedPair(statistic=T, edf0=rank, lambda0=0.0) for _ in budgets]

    S = np.asarray(S_ti, dtype=np.float64)
    S = 0.5 * (S + S.T)
    scale = max(float(np.trace(V)), 1e-300) / max(float(np.trace(S)), 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale

    # Bracket first, with two ordinary solves shared by every rung.  A budget
    # outside the bracket clamps to an edge and needs no search at all, and for
    # the wide-factor spline_cat that clamps at EVERY rung this is the whole
    # computation -- decomposing there would be strictly more expensive than
    # the solves it replaces.  The pencil is built only if some rung genuinely
    # has to search, and then it serves all of them.
    (edf_lo, apply_lo), (edf_hi, apply_hi) = _edge(V, S, lo), _edge(V, S, hi)
    needs_search = any(edf_hi < float(b) < edf_lo for b in budgets)
    pencil = _build_pencil(V, S, U) if needs_search else None

    out: list[ScreenedPair] = []
    edges = {lo: (edf_lo, apply_lo), hi: (edf_hi, apply_hi)}
    edge_cache: dict[float, tuple[float, float]] = {}
    for budget in budgets:
        edf0 = float(budget)
        if edf_hi < edf0 < edf_lo:
            # Genuine search: the closed forms answer it in O(k).
            lam = _lambda_for_edf(pencil, edf0, scale)
            out.append(
                ScreenedPair(
                    statistic=_pencil_stat(pencil, lam),
                    edf0=_pencil_edf(pencil, lam),
                    lambda0=float(lam),
                )
            )
            continue
        # Clamped: answer it the direct way even when a pencil exists for some
        # other rung, so a clamped rung stays bit-identical to what it always
        # reported.  Both edges were already factored to bracket the ladder,
        # so a clamped rung adds one k-vector solve and nothing else.
        lam = lo if edf0 >= edf_lo else hi
        if lam not in edge_cache:
            achieved, apply = edges[lam]
            edge_cache[lam] = (float(U @ apply(U)), achieved)
        stat, achieved = edge_cache[lam]
        out.append(ScreenedPair(statistic=stat, edf0=achieved, lambda0=float(lam)))
    return out


def penalized_score_statistic(
    U: NDArray,
    V: NDArray,
    C: NDArray | None = None,
    M: NDArray | None = None,
    S_ti: NDArray | None = None,
    *,
    edf0: float = 4.0,
    U_nuisance: NDArray | None = None,
) -> ScreenedPair:
    """Rank one candidate pair by its penalized efficient-score statistic.

    ``C`` (overlap x tensor cross-moments) and ``M`` (overlap curvature)
    profile out the span the mains model already explains; ``U_nuisance`` is
    the overlap block's own score (zero at an exactly stationary fit, the
    penalty gradient ``S_M beta`` otherwise).  With ``S_ti`` absent or zero
    the statistic reduces to the unpenalized ``U' V^{-1} U`` and ``lambda0``
    is reported as 0.

    Scoring a ladder of budgets? Use
    :func:`penalized_score_statistic_ladder`, which shares one decomposition
    across every rung instead of rebuilding it per call.
    """
    return penalized_score_statistic_ladder(
        U,
        V,
        C,
        M,
        S_ti,
        budgets=(float(edf0),),
        U_nuisance=U_nuisance,
    )[0]
