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

_EDF_TOL = 1e-6
_MAX_BISECT = 200
_RCOND = 1e-12


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
    rank_v: float


def _solve_psd(A: NDArray, B: NDArray) -> NDArray:
    """Solve A X = B for symmetric PSD A, falling back to a pseudo-inverse."""
    try:
        factor = scipy.linalg.cho_factor(A, check_finite=False)
        return scipy.linalg.cho_solve(factor, B, check_finite=False)
    except scipy.linalg.LinAlgError:
        return np.linalg.pinv(A, hermitian=True) @ B


def _edf(V: NDArray, S: NDArray, lam: float) -> float:
    return float(np.trace(_solve_psd(V + lam * S, V)))


def _edge(V: NDArray, S: NDArray, lam: float) -> tuple[float, Callable[[NDArray], NDArray]]:
    """Factor ``V + lam * S`` ONCE; return its edf and a solver against it.

    The bracket and the clamped rungs ask the same two matrices for both an
    edf and a statistic.  Answering each from its own factorization repeated
    the expensive half three times over: the edf needs ``A^-1 V``, a solve
    with k right-hand sides, which costs about 3x the factorization it rides
    on -- measured at 58% of a wide pair's total time against 17% for the
    factorizations themselves.  One factorization per edge, reused, removes
    both duplicates.
    """
    A = V + lam * S
    try:
        factor = scipy.linalg.cho_factor(A, check_finite=False)
        apply = partial(scipy.linalg.cho_solve, factor, check_finite=False)
    except scipy.linalg.LinAlgError:
        apply = np.linalg.pinv(A, hermitian=True).__matmul__
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
    problem at ``1e-12`` returned ``2/3`` and ``1``.
    """
    G = 0.5 * (V + S + (V + S).T)
    try:
        a, basis = scipy.linalg.eigh(V, G, check_finite=False)
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        w, Q = np.linalg.eigh(G)
        top = float(w.max()) if w.size else 0.0
        keep = w > _RCOND * max(top, np.finfo(np.float64).tiny)
        if not np.any(keep):
            return _Pencil(a=np.zeros(0), u=np.zeros(0), rank_v=0.0)
        whiten = Q[:, keep] / np.sqrt(w[keep])
        Vt = whiten.T @ V @ whiten
        a, R = np.linalg.eigh(0.5 * (Vt + Vt.T))
        basis = whiten @ R
    # a and 1 - a are both variance shares in G-space, so they live in [0, 1];
    # the clip only absorbs round-off at the ends.
    a = np.clip(a, 0.0, 1.0)
    return _Pencil(a=a, u=basis.T @ U, rank_v=float(np.sum(a > _RCOND)))


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
        # No penalty to scan: one solve, at the block's own dimension.
        T = float(U @ _solve_psd(V, U))
        rank = _edf(V, np.zeros_like(V), 0.0)
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
