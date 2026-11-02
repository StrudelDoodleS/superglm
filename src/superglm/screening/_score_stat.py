"""Penalized efficient-score statistic for one candidate pair.

The pair arrives as one triangular factor of its weighted joint design
(:mod:`superglm.screening._pair_factor`) and one factor of its tensor penalty
(:func:`superglm.screening._overlap.tensor_penalty_root`). Frisch-Waugh-Lovell reads
the profiled block off the first as a slice, so

    T = U_eff' (V_eff + lambda0 S)^{-1} U_eff,   V_eff = R_eff' R_eff,   U_eff = R_eff' z_t,

never as ``V - C' M^-1 C``, never by inverting an overlap Gram. Where the overlap
is rank deficient, :func:`superglm.screening._pair_factor._profiled_factor` stacks
back the directions the overlap never spanned. ``lambda0`` fixes the smooth's
complexity: ``tr((V_eff + lambda0 S)^{-1} V_eff) = edf0``.

For ``U_eff ~ N(0, phi V_eff)`` and ``a_j = c_j^2 / (c_j^2 + lambda s_j^2)``, ``T/phi``
has the reference law ``sum_j a_j Z_j^2`` with mean ``sum(a)`` and variance
``2 sum(a**2)``; each rung carries both moments into the ranking. This is a
reference calculation, not a calibrated p-value after fitting the baseline and
selecting a rung; assess candidate refits on held-out data.

Both quantities the search needs are closed forms in one generalized SVD of
``(R_eff, rootS)`` (the QR-plus-CS construction of :func:`_pair_pencil`; Van Loan,
*SIAM J. Numer. Anal.* 13, 1976; Stewart, *Numer. Math.* 40, 1983; Bai and Demmel,
*SIAM J. Sci. Comput.* 14, 1993):

    edf(lambda) = sum_j c_j^2 / (c_j^2 + lambda s_j^2),
    T(lambda)   = sum_j u_j^2 / (c_j^2 + lambda s_j^2),

so a rung costs O(k), and one decomposition serves a whole ladder of budgets
(:func:`penalized_score_statistic_ladder`). One estimator answers every rung, so
two rungs landing on one lambda report one edf. This is the standard Tikhonov
influence-trace computation, which takes the design or a backward-stable factor,
never the Gram (Eldén, *BIT* 17, 1977; Golub, Heath and Wahba, *Technometrics* 21,
1979; Wood, *JRSS-B* 70:495-518, 2008, sec. 3.2). A Gram squares the spectrum, so
directions the answer depends on can fall below float64's resolution in it.

Scale discipline. (1) A relative threshold is meaningful only on an equilibrated
operand: scale symmetrically by the diagonal first (a congruence, so rank is
preserved). (2) Two quantities may be added, subtracted or differenced only once
they share a scale. The pencil stacks ``R_eff`` under a balanced ``rootS`` rather
than forming ``V + S``, so the smaller term cannot be lost; the balance now serves
accuracy, since each block's singular values are resolved to ``eps`` absolutely.

Threshold types. A type-1 cut ("is this arithmetic meaningless?") is derivable from
backward stability as a function of ``eps`` and dimension (:func:`_rank_floor`).
A type-2 cut ("is this answer small?") is a statement about data, and only a type-1
bound may justify discarding a pair; weak identification belongs to ``z``. A wholly
absorbed block therefore is not detected: no dimension-scaled cut separates it from
weak identification (``max(mu)`` ranges over eleven orders at fixed ``k = 24``). It
is scored, the statistic comes out at round-off, and ``z`` ranks it last. The
unpenalized rung's edf is ``rank(V_eff)``, counted against the joint design's scale
(:func:`superglm.screening._pair_factor._profiled_rank_scale`), because on an
absorbed block ``R_eff`` is entirely round-off and a relative cut against its own
top reports a nonzero edf (20 of 20 seeds); a too-high edf lowers ``z``.

Low-edge sensitivity. For a positive definite pencil, ``edf = k - lambda tr(A^-1 S)``
with ``A = V_eff + lambda S``, and for a perturbation ``E`` of the operand
``d edf = <E, G>_F`` with ``G = lambda A^-1 S A^-1 + alpha (d edf/d lambda) I``,
where ``lambda = alpha tr(V_eff)``, ``alpha = 1e-10 / tr(S)``. Over ``||E||_F = c`` the
differential is maximised at ``c ||G||_F``, so one rounding of the operand
(``c = eps ||V_eff||_F``) moves edf by about ``eps ||V_eff||_F ||G||_F``, a
first-order sensitivity rather than a bound. It spans twelve orders across designs
and is large exactly when a level's rows put the constant in the span of its own
spline columns (issue #279). It does not cover a pair whose ``V_eff`` and ``S``
share a null space, which the pencil discards. No published forward-error bound
exists for ``tr((A + lambda S)^-1 A)`` under perturbation of ``A``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.screening._factor_kernels import _factor_rank_floor
from superglm.screening._pair_factor import (
    PairFactor,
    _pair_scale,
    _profiled_factor,
    _profiled_rank_scale,
)

_EDF_TOL = 1e-6
_MAX_BISECT = 200


@dataclass(frozen=True)
class ScreenedPair:
    """One rung, including Var(T/phi) under U ~ N(0, phi V)."""

    statistic: float
    edf0: float
    lambda0: float
    reference_variance: float


@dataclass(frozen=True)
class _Pencil:
    """One factor decomposition for the pair's entire penalty ladder.

    Normally ``v = cosine**2``, ``s = sine**2 / balance`` and ``u`` is the rotated
    score; a rung evaluates ``sum(v / (v + lam*s))`` and ``sum(u**2 / (v + lam*s))``.
    Exceptional penalty units use ``square_exponents``: the three arrays then hold
    mantissas of the squared terms (including the squared score) and the evaluators
    combine exponents before rounding. ``tr_v`` is the profiled block's trace and
    ``tr_s`` the penalty factor's squared norm, with a power of two in
    ``penalty_exponent``; the balance and the search bracket share both.
    """

    v: NDArray
    s: NDArray
    u: NDArray
    tr_v: float
    tr_s: float = 0.0
    penalty_exponent: int = 0
    # Exceptional penalty units require mantissa/exponent pairs. In that
    # case v, s and u hold mantissas of the three squared terms, including
    # the squared score. Ordinary pencils keep the direct representation.
    square_exponents: tuple[NDArray, NDArray, NDArray] | None = None


def _empty_pencil(tr_v: float = 0.0, tr_s: float = 0.0, penalty_exponent: int = 0) -> _Pencil:
    return _Pencil(
        v=np.zeros(0),
        s=np.zeros(0),
        u=np.zeros(0),
        tr_v=float(tr_v),
        tr_s=tr_s,
        penalty_exponent=penalty_exponent,
    )


def _pair_pencil(pair: PairFactor, penalty_root: NDArray | None) -> _Pencil:
    """Diagonalize ``(V_eff, S)`` from the pair's factor, never from its Gram.

    The GSVD of ``(R_eff, rootS)`` by a pivoted QR of the stack and the CS decomposition of its
    orthonormal factor (*LAPACK Users' Guide*, 3rd ed., sec. 4.7). With
    ``[R_eff ; rootS] = Q Rg``, ``Q1 = Uc C W'``, ``Q2 = Us S W'``: ``V_eff = Y C^2 Y'``,
    ``S = Y S^2 Y'``, ``Y = Rg' W``, ``u = C Uc' z_t``, so every rung is a sum of filter factors
    in ``[0, 1]``, none a difference. ``rootS`` is first scaled by ``sqrt(tr(V_eff) / tr(S))``,
    as each block's SVD is accurate to ``eps`` absolutely. ``penalty_root = None`` is the
    unpenalized block: ``edf`` is the retained rank and ``T`` the squared norm of ``z_t`` on the
    resolved directions.
    """
    R_eff, z_t = _profiled_factor(pair)
    # ``tr(V_eff)`` once, off the block this pencil scores; the ladder's bracket reuses it.
    tr_v = _pair_scale(R_eff)
    k = int(pair.tensor_width)
    balance = 1.0
    tr_s = 0.0
    penalty_exponent = 0
    scaled = False
    if penalty_root is None or penalty_root.size == 0:
        root = np.zeros((0, k), dtype=np.float64)
    else:
        root = np.asarray(penalty_root, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore"):
            tr_s = float(np.sum(root**2))
        proposed = tr_v / tr_s if tr_s > 0.0 else 1.0
        scaled = (
            not np.isfinite(tr_s)
            or (tr_v > 0.0 and tr_s > 0.0 and not np.finfo(float).tiny <= proposed < np.inf)
            or (tr_s == 0.0 and bool(np.any(root)))
        )
        if scaled:
            # R_S = 2**e * root. Keep that power outside the squared terms:
            # both a finite root's norm and its balancing ratio can exceed
            # float64's range while lambda*S remains moderate.
            peak = float(np.max(np.abs(root), initial=0.0))
            if not np.isfinite(peak):
                raise ValueError("penalty_root must contain only finite values")
            exponent = int(np.frexp(peak)[1]) - 1
            with np.errstate(under="ignore"):
                root = np.ldexp(root, -exponent)
            penalty_exponent = 2 * exponent
            tr_s = float(np.sum(root**2))
        if tr_v > 0.0 and tr_s > 0.0:
            proposed = tr_v / tr_s
            if np.isfinite(proposed) and proposed > 0.0:
                balance = proposed
        root = np.sqrt(balance) * root
    if k == 0:
        return _empty_pencil(tr_v, tr_s, penalty_exponent)

    stack = np.concatenate((R_eff, root), axis=0)
    orthonormal, triangular, pivot = scipy.linalg.qr(
        stack, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0 or float(diagonal[0]) <= np.finfo(np.float64).tiny:
        return _empty_pencil(tr_v, tr_s, penalty_exponent)
    # With a penalty the cut asks what neither operand resolves: a direction the stack
    # does not resolve carries neither curvature nor penalty, so discarding it is exact,
    # and no further rank decision is taken. Its reference is the first pivot, the stack's
    # largest column norm (Businger and Golub, *Numer. Math.* 7, 1965), so
    # ``sigma_1/sqrt(k) <= |R_00| <= sigma_1``: it under-scales the floor by up to
    # sqrt(k) against a ``sigma_1`` reference and retains directions that one would drop.
    # That is deliberate (as in LAPACK ``xGELSY``; the RRQR bounds carry dimension
    # constants of the same shape: Chan, 1987; Hong and Pan, 1992): on 13 adversarial
    # pairs the two choose the same rank on 12, and on the 13th the pivot reference is
    # 6.25e-07 from an unthresholded augmented-QR oracle where ``sigma_1`` is 0.995 off.
    # The contested direction sits at 2.78e-08 of the stack's spectrum, 7.7e-16 of the
    # Gram's (issue #257).
    #
    # Without a penalty the cut is the answer (``edf`` is ``rank(V_eff)``) and
    # ``R_eff``'s own top is round-off on the blocks that matter, so
    # :func:`_profiled_rank_scale` supplies the joint design's scale.
    cut = (
        _factor_rank_floor(k) * float(diagonal[0])
        if root.size
        else _factor_rank_floor(k + int(pair.overlap_width)) * _profiled_rank_scale(pair)
    )
    rank = int(np.count_nonzero(diagonal > cut))
    if rank == 0:
        return _empty_pencil(tr_v, tr_s, penalty_exponent)

    top = orthonormal[: R_eff.shape[0], :rank]
    bottom = orthonormal[R_eff.shape[0] :, :rank]
    if scaled:
        # Explicit Householder Q can round a tiny leading-block entry to
        # zero. Recover Q1 from R_eff[:, pivot] = Q1 @ Rg on the retained
        # columns, preserving the original small curvature and row score.
        top = scipy.linalg.solve_triangular(
            triangular[:rank, :rank].T,
            R_eff[:, pivot[:rank]].T,
            lower=True,
            check_finite=False,
        ).T
    # The common basis ``W`` comes from the penalty block. ``Q'Q = I`` gives
    # ``c_j**2 + s_j**2 = 1`` for any orthonormal ``W``, so one decomposition settles both
    # terms; what differs is which clusters are ambiguous. The filter factor is a function
    # of ``s`` alone, so a rotation within a cluster of near-equal ``s`` leaves every sum
    # invariant, whereas near ``c = 1`` float64 cannot recover ``s = sqrt(1 - c**2)``.
    # Taking ``W`` from ``Q1`` put ``moderate_pair``'s high-edge statistic 3.89e-06 from the
    # stacked-QR arbiter against 5.4e-13 here (Stewart, 1983; Van Loan, 1985).
    if bottom.size:
        # ``full_matrices`` only where the penalty root is shorter than the pencil: ``Q2`` then
        # has a null space whose exact zeros belong at the end of the descending list.
        _, values, basis = np.linalg.svd(bottom, full_matrices=bottom.shape[0] < rank)
        sines = np.concatenate((values, np.zeros(rank - values.size)))[:rank]
        carried = top @ basis.T
    else:
        sines = np.zeros(rank, dtype=np.float64)
        carried = top
    if scaled:
        # Squaring a small sine or cosine here would lose information before
        # its physical exponent is restored. hypot also avoids squaring the
        # columns while measuring their norms.
        cosines = np.hypot.reduce(carried, axis=0)
        cm, ce = np.frexp(np.clip(cosines, 0.0, 1.0))
        sm, se = np.frexp(np.clip(sines, 0.0, 1.0))
        um, ue = np.frexp(carried.T @ z_t)
        bm, be = np.frexp(balance)
        return _Pencil(
            v=cm**2,
            s=sm**2 / bm,
            u=um**2,
            tr_v=tr_v,
            tr_s=tr_s,
            penalty_exponent=penalty_exponent,
            square_exponents=(2 * ce, 2 * se - be + penalty_exponent, 2 * ue),
        )
    cosines = np.linalg.norm(carried, axis=0)
    return _Pencil(
        v=np.clip(cosines**2, 0.0, 1.0),
        s=np.clip(sines**2, 0.0, 1.0) / balance,
        u=carried.T @ z_t,
        tr_v=tr_v,
        tr_s=tr_s,
    )


def _sum_scaled_positive(mantissas: NDArray, exponents: NDArray) -> float:
    """Sum before final rounding, including individually subnormal terms."""
    nonzero = mantissas > 0.0
    if not np.any(nonzero):
        return 0.0
    exponent = int(np.max(exponents[nonzero]))
    with np.errstate(under="ignore", over="ignore"):
        total = np.sum(np.ldexp(mantissas[nonzero], exponents[nonzero] - exponent))
        return float(np.ldexp(total, exponent))


def _scaled_pencil_moments(p: _Pencil, lam: float) -> tuple[float, float, float]:
    """EDF, statistic and variance without forming unrepresentable squares."""
    assert p.square_exponents is not None
    ve, se, ue = p.square_exponents
    lm, le = np.frexp(lam)
    penalty = lm * p.s
    pe = se + le
    de = np.maximum(ve, pe)
    # A zero term's exponent carries no scale and must not set the divisor.
    de = np.where(p.v == 0.0, pe, de)
    de = np.where(penalty == 0.0, ve, de)
    with np.errstate(under="ignore"):
        den = np.ldexp(p.v, ve - de) + np.ldexp(penalty, pe - de)
    filters = np.divide(p.v, den, out=np.zeros_like(p.v), where=den > 0.0)
    scores = np.divide(p.u, den, out=np.zeros_like(p.u), where=den > 0.0)
    fe = ve - de
    return (
        _sum_scaled_positive(filters, fe),
        _sum_scaled_positive(scores, ue - de),
        _sum_scaled_positive(filters**2, 2 * fe + 1),
    )


def _pencil_edf(p: _Pencil, lam: float) -> float:
    if p.square_exponents is not None:
        return _scaled_pencil_moments(p, lam)[0]
    den = p.v + lam * p.s
    ok = den > 0.0
    return float(np.sum(p.v[ok] / den[ok]))


def _pencil_stat(p: _Pencil, lam: float) -> float:
    if p.square_exponents is not None:
        return _scaled_pencil_moments(p, lam)[1]
    den = p.v + lam * p.s
    ok = den > 0.0
    return float(np.sum(p.u[ok] ** 2 / den[ok]))


def _pencil_reference_variance(p: _Pencil, lam: float) -> float:
    """Twice the sum of squared filters, for fixed Gaussian score geometry."""
    if p.square_exponents is not None:
        return _scaled_pencil_moments(p, lam)[2]
    den = p.v + lam * p.s
    ok = den > 0.0
    filters = p.v[ok] / den[ok]
    return 2.0 * float(np.sum(filters**2))


def _lambda_bracket(
    scale: float, *, denominator: float = 1.0, exponent: int = 0
) -> tuple[float, float]:
    """Keep both search edges in the positive finite float64 range.

    A change of penalty units can move an edge beyond that range while the
    target remains reachable. If the target itself is unrepresentable, the
    usual edge clamp reports the EDF achieved at a finite penalty instead.
    """
    limits = np.finfo(np.float64)
    smallest, largest = float(limits.smallest_subnormal), float(limits.max)
    if denominator != 1.0 or exponent != 0:
        numerator_m, numerator_e = np.frexp(scale)
        denominator_m, denominator_e = np.frexp(denominator)
        ratio_m = numerator_m / denominator_m
        ratio_e = int(numerator_e) - int(denominator_e) + exponent
        with np.errstate(over="ignore", under="ignore"):
            edges = np.ldexp(np.array([1e-10, 1e10]) * ratio_m, ratio_e)
        return float(np.clip(edges[0], smallest, largest)), float(
            np.clip(edges[1], smallest, largest)
        )
    return (
        min(max(1e-10 * scale, smallest), largest),
        min(max(1e10 * scale, smallest), largest),
    )


def _lambda_for_edf(p: _Pencil, edf0: float, bracket: tuple[float, float]) -> float:
    """Smallest-error ``lambda`` hitting ``edf0``, clamped to the bracket edges.

    ``edf(lambda)`` decreases monotonically from ``rank(V_eff)`` toward the
    dimension of the penalty null space, so a target outside the bracket is
    unreachable; clamping to the nearest edge keeps the pair in the table
    rather than failing it, and the achieved value is reported so a caller can
    see the budget was not met.
    """
    lo, hi = bracket
    edf_lo, edf_hi = _pencil_edf(p, lo), _pencil_edf(p, hi)
    if edf_lo <= edf0:
        return lo
    if edf_hi >= edf0:
        return hi
    for _ in range(_MAX_BISECT):
        if hi <= lo * (1.0 + 1e-12):
            break
        # The geometric mean is representable whenever both endpoints are;
        # their product can overflow or underflow after a change of units.
        lam = float(np.sqrt(lo) * np.sqrt(hi))
        if not lo < lam < hi:
            break  # Adjacent subnormal lambdas need not attain the target.
        achieved = _pencil_edf(p, lam)
        if abs(achieved - edf0) <= _EDF_TOL:
            return lam
        if achieved > edf0:
            lo, edf_lo = lam, achieved
        else:
            hi, edf_hi = lam, achieved
    # Choose by achieved EDF, with the smaller lambda breaking an exact tie.
    return lo if abs(edf_lo - edf0) <= abs(edf_hi - edf0) else hi


def penalized_score_statistic_ladder(
    pair: PairFactor,
    penalty_root: NDArray | None,
    *,
    budgets: tuple[float, ...] = (4.0,),
) -> list[ScreenedPair]:
    """Score one pair at every budget in ``budgets``, sharing one decomposition.

    ``pair`` is the pair's weighted joint design reduced to one triangular factor
    (:mod:`superglm.screening._pair_factor`); ``penalty_root`` is a factor of the tensor
    penalty (``rootS' rootS = S_ti``, from
    :func:`superglm.screening._overlap.tensor_penalty_root`) or ``None`` for an
    unpenalized block. Neither is a ``(k, k)`` Gram: a Gram's spectrum is its factor's
    squared (issue #257), and a Gram-passing caller gets a ``TypeError``. Penalties are
    rooted by one policy, :func:`superglm.screening._factor_kernels._penalty_root`, as on
    the structured path (issue #323). A budget outside the bracket clamps to the nearest
    edge and one inside bisects; both read the same pencil, so equal lambdas report equal
    ``edf`` and ``statistic`` bit-for-bit.
    """
    root = None if penalty_root is None else np.asarray(penalty_root, dtype=np.float64)
    if root is None or root.size == 0:
        # No penalty to scan: every rung is the block's achieved rank at ``lambda0 = 0``; an
        # all-zero penalty reaches here as an empty root.
        p = _pair_pencil(pair, None)
        stat, rank = _pencil_stat(p, 0.0), _pencil_edf(p, 0.0)
        return [
            ScreenedPair(statistic=stat, edf0=rank, lambda0=0.0, reference_variance=2.0 * rank)
            for _ in budgets
        ]

    # ``tr(S) = ||rootS||_F**2`` off the factor; the structured ladder uses the same
    # ``tr(V_eff) / (tr(S_a) * L)`` over the same 1e+-10 edges, so a pair both paths score
    # gets one ``lambda0``. The numerator comes off the pencil, the block it scored: a
    # re-slice of the factor was 18.06% low where the overlap is rank deficient.
    p = _pair_pencil(pair, root)
    if p.square_exponents is not None or p.penalty_exponent:
        lo, hi = _lambda_bracket(
            p.tr_v if p.tr_v > 0.0 else 1e-300,
            denominator=p.tr_s if p.tr_s > 0.0 else 1e-300,
            exponent=-p.penalty_exponent,
        )
    else:
        scale = max(p.tr_v, 1e-300) / max(p.tr_s, 1e-300)
        lo, hi = _lambda_bracket(scale)
    edf_lo, edf_hi = _pencil_edf(p, lo), _pencil_edf(p, hi)

    out: list[ScreenedPair] = []
    for budget in budgets:
        edf0 = float(budget)
        if edf_hi < edf0 < edf_lo:
            lam = _lambda_for_edf(p, edf0, (lo, hi))
        else:
            lam = lo if edf0 >= edf_lo else hi
        out.append(
            ScreenedPair(
                statistic=_pencil_stat(p, lam),
                edf0=_pencil_edf(p, lam),
                lambda0=float(lam),
                reference_variance=_pencil_reference_variance(p, lam),
            )
        )
    return out
