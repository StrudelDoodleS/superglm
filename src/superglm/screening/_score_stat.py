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

**THRESHOLD TYPES.  Read this before adding a constant to this module.**

Every cut here answers one of two questions, and they have different standing.

*Type 1 -- "is this arithmetic meaningless?"*  A statement about floating
point, DERIVABLE from backward stability as a function of machine epsilon and
dimension.  It holds for every input because the bound covers all of them, so
there is no unmeasured regime waiting to break it.  :func:`_rank_floor` is of
this kind: ``max(n, 1) * eps``, LAPACK's convention and ``matrix_rank``'s own
tolerance.  So is ``_solve_floor`` in :mod:`superglm.screening._arrow`.

*Type 2 -- "is this answer small?"*  A statement about DATA.  Not derivable;
any constant is a claim about the datasets its author had seen, and there is
always a legitimate dataset on the other side of it.

**Only a Type 1 bound may justify DISCARDING a pair.**  Weak identification is
a finding about the data and belongs to ``z``, which ranks such a pair down on
its merits.  Needing a fitted constant to decide whether to discard is the
signal that a data judgement is being made in the wrong place.

That rule cost a guard.  A wholly absorbed block -- every probe column a
multiple of its level's indicator, so the true profiled rank is 0 -- would be
worth detecting, and the natural test is ``max(mu)`` over the pencil
``(V_eff, V + C' M^-1 C)``, the largest share of curvature profiling leaves
behind.  There is no Type 1 threshold for it.  Measured on EXACTLY absorbed
blocks at FIXED ``k = 24``, varying only how unevenly the levels carry weight:
``max(mu)`` ranges over 9.6e-16, 1.1e-15, 1.7e-12, 5.9e-12, 9.1e-15 and
2.4e-05 -- eleven orders at one dimension, so no power of ``k`` governs it.
``eps * cond(M)`` does not bound it usefully either, since the overlap block
is near-singular by construction here (measured ``cond(M)`` ~ 1e20 throughout,
which makes that bound vacuous).  A dimension-scaled cut therefore cannot
separate absorption from weak identification, and the fitted one that was
tried both deleted a legitimately weak block (``V = (1 + 1e-4) I``, whose
``V_eff`` is ``1e-4 I`` and full rank) and failed to fire on a genuinely
absorbed one at the same ``k``.

So absorption is NOT detected and such a pair is NOT discarded.  It is scored:
the rank comes from :func:`_rank_floor` like any other block, the statistic
comes out at round-off, and ``z`` puts the pair near the bottom where it
belongs.  Its ``edf0`` is not reproducible across seeds, which is the honest
signature of a numerically indeterminate block rather than something to hide.

There IS a Type 1 route to the same answer -- Guttman rank additivity,
``rank([[V, C'], [C, M]]) - rank(M)``, where both ranks are of PSD matrices
formed by ADDITION, so neither cancels and both are countable at
:func:`_rank_floor`.  It is not taken here on cost: the overlap of a
``numeric_cat`` pair is as wide as the probe, so the bordered system is ``2k``
and its eigendecomposition is 8x the ``k`` one -- about 1.1 s to 2.3 s at the
budget's own ceiling, against the ~1.5 s per-pair target the cubic constants
were fitted to.  It is affordable for ``cat_cat``, where the overlap is small
beside the probe.  Worth its own issue rather than this module's guesswork.
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


def _rank_floor(n: int) -> float:
    """Share of the largest eigenvalue below which a direction is ROUND-OFF.

    ``max(n, 1) * eps`` -- LAPACK's convention and exactly what
    ``numpy.linalg.matrix_rank`` uses by default.  It scales with the
    dimension because round-off accumulates with it, and that dependence is
    the whole point: no fixed constant works at both ends.

    Two failures bound it from either side, and they are three orders apart,
    so this is not a free parameter.

    ABOVE round-off it deletes real curvature.  At 1e-12, ``V = S =
    diag(1, 1e-13, 0)`` with ``U = (0, sqrt(1e-13), 0)`` had its 1e-13
    direction discarded by the whitening below -- a direction carrying a
    genuine ``a = 0.5`` and ALL of ``U``'s mass -- and the ladder returned
    ``statistic 0, lambda0 1e-10`` where the direct pseudo-inverse ladder
    resolves ``lambda0 1, statistic 0.5``.  Here that direction sits 150x
    above the floor and survives.

    AT round-off it keeps subtraction dust and reports a degree of freedom
    that does not exist.  A fixed 1e-15 is only 4.5x ``eps``, which is inside
    the dust's own distribution rather than above it: measured over 400
    replicates of a 39-wide profiled block whose true rank is 38, the
    round-off eigenvalue has median 2.2e-16 and a tail to 1.2e-15, so 2 of
    400 read rank 39.  ``39 * eps`` is 8.7e-15, above the whole measured tail
    by 7x.

    Because this IS ``matrix_rank``'s tolerance, the count it produces cannot
    exceed ``matrix_rank`` -- the contract the unpenalized rung rests on holds
    by construction rather than by luck, and only tightens where a negative
    eigenvalue makes ``psd_ranks`` stricter than a singular-value count.

    It is only meaningful where the matrix's own largest eigenvalue is a real
    scale.  On a wholly PROFILED-AWAY block that is not guaranteed -- see the
    THRESHOLD TYPES note at the top of this module for what is and is not
    decidable there.
    """
    return max(int(n), 1) * float(np.finfo(np.float64).eps)


def _psd_rank(A: NDArray) -> float:
    """Numerical rank of a symmetric PSD matrix at the round-off floor."""
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    if w.size == 0:
        return 0.0
    top = float(w[-1])
    if top <= 0.0:
        return 0.0
    return float(np.sum(w > _rank_floor(w.size) * top))


def _profiled_rank(V_eff: NDArray, joint: tuple | None) -> float:
    """Rank of the profiled block, decided where NOTHING cancels.

    ``rank(V_eff)`` is the unpenalized rung's edf, and no cut taken on
    ``V_eff`` itself is safe.  ``V_eff = V - C' M^-1 C`` is a difference, so an
    absorbed direction survives only as round-off -- and on a block the overlap
    has largely absorbed, that round-off IS the largest eigenvalue, so a
    relative cut is taken against the very noise it must reject.  Three rules
    were tried against the block itself and all three fail on the REACHABLE
    path, 5-level wholly absorbed pairs screened end to end, 20 seeds:

    * relative to ``V_eff``'s own top: nonzero rank on 37 of 40 replicates;
    * plus ``max(-lambda_min, 0)``, a certificate that ``V_eff >= 0`` is
      violated: 12 of 40.  It is silent whenever the perturbation lands
      positive-definite, which for a symmetric perturbation on an exactly null
      subspace is a coin flip at EVERY width;
    * plus an a-priori ``k * eps * tr(V)`` bound on the difference: 9 of 20.
      It is short because the dominant error is not the subtraction at all --
      ``cond(M)`` measured 5.8e+14 to 5.8e+15 on every failing seed, so the
      overlap is numerically singular and the error comes from inverting it.
      No bound written in terms of ``||V||`` can see that.

    Getting an edf too HIGH is not a neutral failure here.  ``z = (T - e) /
    sqrt(2 e)`` DECREASES in ``e``, so a partly-rejected block scores higher
    than an unrejected one: on a measured ``statistic = 145.508`` an edf of 10
    gives ``z = 30.3`` where an edf of 1 gives ``z = 102.2``.  Only rank 0 is
    safe, which is why a rule that reaches it most of the time is not a rule.

    So the rank is taken where nothing has cancelled.  Guttman rank additivity
    on the JOINT moment matrix,

        rank([[V, C'], [C, M]]) = rank(M) + rank(V - C' M^-1 C),

    holds for a PSD joint, and both ranks on the left are of matrices assembled
    by ADDITION.  Neither hides a cancellation, so each is countable at
    :func:`_rank_floor` -- the round-off floor that is derivable from eps and
    dimension -- and the difference is exact integer arithmetic.  No inverse is
    formed, no conditioning enters, and there is no threshold on a cancelled
    quantity anywhere.  Measured 0 of 20 on the same reachable path that
    defeats all three rules above, and it is right on the other corners too: a
    three-direction block whose probe is exactly nested (true rank 2, where a
    positive-only cancellation makes the certificate silent), and a weak but
    real ``V_eff = 1e-4 I`` at k = 2, 4 and 12 (true rank k).

    **Precondition: the working weights are non-negative.**  Guttman rank
    additivity as used here needs the joint ``[[V, C'], [C, M]]`` to be PSD,
    which needs ``working_weights >= 0``.  Satisfied on the reachable path --
    ``screening_ops`` builds them as ``weights * dmu_deta**2 / var_mu`` with
    ``var_mu`` floored, from weights already validated finite and
    non-negative, so it rests on Fisher scoring plus that validation.  Worth
    stating because a negative weight would not merely degrade the count: the
    joint would stop being PSD, the additivity would no longer hold, and the
    rank could come out either side of the truth.

    **The cost is real, and measured rather than filed.**  The bordered system
    is ``k + q``, and for ``numeric_cat`` the overlap is as wide as the probe,
    so the eigendecomposition is of ``2k`` where a direct count is of ``k``.
    Interleaved A/B in one process at one BLAS thread, whole pair end to end:

        k =  209   0.046 s -> 0.052 s   (1.12x)
        k = 1709   1.134 s -> 3.408 s   (3.01x)

    So it is nearly free at ordinary widths and 3x at the cubic budget's own
    ceiling.  It is paid on ONE rung of the one ladder with no bandwidth to
    scan, and only where an overlap was profiled out.

    **No pair changes whether it is screened at all.**  ``_within_cubic_budget``
    is ``k**3 <= _CUBIC_BUDGET_FACTOR * max_cells``, a DIMENSIONAL gate
    evaluated on ``k`` alone, so admission is identical before and after and
    nothing silently becomes a NaN row.  What does move is that constant's
    calibration: it was fitted against a ~1.5 s per-pair target, and the widest
    unpenalized pair it admits now takes 3.4 s.  Re-fitting it is a separate
    decision and is not taken here.
    """
    if joint is None:
        return _psd_rank(V_eff)
    V, C, M = joint
    k, q = V.shape[0], M.shape[0]
    K = np.empty((k + q, k + q), dtype=np.float64)
    K[:k, :k] = V
    K[:k, k:] = C.T
    K[k:, :k] = C
    K[k:, k:] = M
    return max(_psd_rank(K) - _psd_rank(M), 0.0)


@dataclass(frozen=True)
class ScreenedPair:
    """Ranking output for one candidate pair."""

    statistic: float
    edf0: float
    lambda0: float


@dataclass(frozen=True)
class _Pencil:
    """Simultaneous diagonalization of ``(V_eff, S)`` with ``U_eff`` rotated in.

    ``v`` and ``s`` are the two transformed diagonal terms, held SEPARATELY
    rather than one derived from the other: ``edf(lam) = sum v / (v + lam s)``
    and ``T(lam) = sum u^2 / (v + lam s)``.  Deriving ``s`` as ``1 - v`` loses
    it whenever ``v`` rounds to 1, which is exactly what happens when the
    curvature dwarfs the penalty -- see :func:`_build_pencil`.
    """

    v: NDArray
    s: NDArray
    u: NDArray


def _solve_psd(A: NDArray, B: NDArray) -> NDArray:
    """Solve A X = B for symmetric PSD A, falling back to a pseudo-inverse."""
    try:
        factor = scipy.linalg.cho_factor(A, check_finite=False)
        return scipy.linalg.cho_solve(factor, B, check_finite=False)
    except scipy.linalg.LinAlgError:
        return np.linalg.pinv(A, hermitian=True) @ B


def _edge(
    V: NDArray, S: NDArray | None, lam: float, joint: tuple | None = None
) -> tuple[float, Callable[[NDArray], NDArray]]:
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
    :func:`superglm.screening._arrow.psd_ranks` is the counter, at
    :func:`_rank_floor` rather than the arrow kernel's own constant: same RULE
    — count eigenvalues, respect their sign — at a cut each path can justify,
    the arrow kernel counting on a deliberately BALANCED reference where a
    fixed 1e-12 means something, and this path counting on a raw profiled
    block where only round-off does.

    The cut here was a fixed 1e-12 when this counting first landed, then a
    fixed 1e-15; both were wrong and in opposite directions, which is why it
    is now dimension-scaled.  ``_rank_floor`` carries the two measurements
    that bound it.  What matters for THIS rung is the lower one: at a fixed
    1e-15 the count could exceed ``matrix_rank`` on a profiled block whose
    round-off dust ran into the tail, reporting a degree of freedom that does
    not exist — the exact defect this counting exists to remove, reintroduced
    from the other side.  At ``k * eps`` the count is ``matrix_rank``'s own
    tolerance, so it cannot.

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
    ``apply`` is deliberately NOT rank-limited to match.  The count sits a
    little above ``pinv``'s own cut, so on the fallback branch a direction
    between them is inverted by the solve without being counted here — which
    is the conservative side of that gap: the edf never claims a degree of
    freedom the data does not carry, and ``U_eff`` has no mass there to
    contribute anyway (measured above at 2.9e-15 relative, worst case).  On
    the Cholesky branch the solve makes no rank decision at all, which is
    exactly why the edf is counted separately here.
    """
    A = V if S is None else V + lam * S
    try:
        factor = scipy.linalg.cho_factor(A, check_finite=False)
        apply = partial(scipy.linalg.cho_solve, factor, check_finite=False)
    except scipy.linalg.LinAlgError:
        apply = np.linalg.pinv(A, hermitian=True).__matmul__
    if S is None:
        return _profiled_rank(V, joint), apply
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
    only fault was being small beside another direction -- see
    :func:`_rank_floor`, which records the case that forced it.  ``G`` is a
    SUM of two PSD terms, so nothing cancels in forming it and its round-off
    floor really is ``eps`` times its own largest eigenvalue; that is why the
    same floor serves here and on the profiled block in :func:`_edge`, where
    the arithmetic is a difference but was measured not to cancel.

    A mode surviving the cut is whitened by ``1 / sqrt(w)``, so a floor this
    low does amplify a direction it keeps.  That is the right trade: keeping
    a noise direction costs at most one spurious degree of freedom in ``edf``,
    where dropping a real one zeroes the statistic outright.
    """
    # BALANCE before summing.  ``G = V + S`` is formed in floating point, so
    # if the two terms live on far-apart scales the smaller is simply lost:
    # with ``V = 1e20 I`` and ``S = I``, ``V + S`` rounds to ``V`` exactly,
    # every share comes back 1, and the ladder reports the full dimension as
    # its edf at every lambda -- the answer then depends on the units the
    # curvature is carried in, or on a frequency-weight scale, which is the
    # same defect a relative whitening cut was introduced to remove.
    # Scaling S to V's trace makes the sum lossless; the scaling is undone
    # exactly below, so nothing depends on the balance point itself.
    tr_v = abs(float(np.trace(V)))
    tr_s = abs(float(np.trace(S)))
    balance = tr_v / tr_s if tr_v > 0.0 and tr_s > 0.0 else 1.0
    Sb = balance * S
    G = 0.5 * (V + Sb + (V + Sb).T)
    try:
        share, basis = scipy.linalg.eigh(V, G, check_finite=False)
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        w, Q = np.linalg.eigh(G)
        top = float(w.max()) if w.size else 0.0
        keep = w > _rank_floor(w.size) * max(top, np.finfo(np.float64).tiny)
        if not np.any(keep):
            return _Pencil(v=np.zeros(0), s=np.zeros(0), u=np.zeros(0))
        whiten = Q[:, keep] / np.sqrt(w[keep])
        Vt = whiten.T @ V @ whiten
        share, R = np.linalg.eigh(0.5 * (Vt + Vt.T))
        basis = whiten @ R
    # In the balanced G-metric the two shares sum to one by construction, so
    # both are recovered directly rather than one by subtracting from the
    # other -- which is what kept the smaller term representable.
    share = np.clip(share, 0.0, 1.0)
    return _Pencil(v=share, s=(1.0 - share) / balance, u=basis.T @ U)


def _pencil_edf(p: _Pencil, lam: float) -> float:
    den = p.v + lam * p.s
    ok = den > 0.0
    return float(np.sum(p.v[ok] / den[ok]))


def _pencil_stat(p: _Pencil, lam: float) -> float:
    den = p.v + lam * p.s
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
    # The JOINT moment matrix, kept so the unpenalized rank can be counted
    # where nothing has cancelled; see _profiled_rank.  None when nothing was
    # profiled out, where V is not a difference and its own scale is already
    # the right reference.
    joint = None
    if C is not None:
        C = np.asarray(C, dtype=np.float64)
        M = np.asarray(M, dtype=np.float64)
        joint = (V, C, M)
        MinvC = _solve_psd(M, C)
        V = V - C.T @ MinvC
        V = 0.5 * (V + V.T)
        if U_nuisance is not None:
            U = U - MinvC.T @ np.asarray(U_nuisance, dtype=np.float64)

    if S_ti is None or not np.any(S_ti):
        # No penalty to scan: ONE factorization of the block answers the
        # statistic, and the achieved rank is COUNTED beside it rather than
        # read off that factorization -- see _edge on why the trace cannot
        # answer it.
        rank, apply = _edge(V, None, 0.0, joint)
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
