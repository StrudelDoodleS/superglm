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

This path is entered only where the dense path is refused, so no production
pair is ever scored twice and nothing at runtime cross-checks the two.  What
stands behind the reorganization is the parity pinned in
tests/test_structured_screening.py, and that parity is a tolerance, not
bit-for-bit equality: away from the bracket the two agree to round-off, at
the edges they agree to the tolerances measured there.

**KNOWN DIVERGENCE, NOT CLOSED.**  That parity is measured on the fixtures the
suite carries and does not generalize.  On a ``ps(8)`` pair with 20 levels, 30
rows in every level, unit weights and four levels squeezed into a narrow band
of the covariate, ``|arrow - dense|`` at the ladder's high edge reaches
11.57 df — 0.62 RELATIVE, against the 1e-5 relative the suite's parity test
pins where both paths can score.  Neither path is reliably the better one
there either: at a width of 3e-4 the arrow value sits 6.93 df above the dense
one and at 5e-4 it sits 4.89 df below it.

**THAT HIGH-EDGE DIVERGENCE IS NOT A DEFECT OF EITHER PATH, AND THE LOW-EDGE
ONE IS THIS PATH'S.**  Settled against arb ball arithmetic — a RIGOROUS
enclosure rather than a high-precision guess, radius below 1e-200 df at 800
bits, agreeing to nine decimals across three algebraically distinct forms of
the same quantity.  An 882-point sweep, 853 of them scored by both paths, over
``ps``/``cr``/``bs``/``ns`` at 3 to 8 knots, ``L`` in 6/12/20, 3/12/30 rows per
level, seven band layouts and two seeds, each path graded at its OWN lambda:

* **The high edge is not determined by the assembled moments.**  ``lambda`` is
  1e10 times the pair's scale there, and a spline penalty assembled in float64
  carries a smallest eigenvalue that is round-off of either sign, so
  ``lambda S_a`` inherits an indefiniteness of order ``lambda eps sigma_max``.
  On the geometry above that is 4.7e-06 against a ``V_eff`` whose own smallest
  eigenvalues are 1e-15: the exact ``V_eff + lambda S`` is INDEFINITE, its
  filter factors leave [0, 1], and the exact edf falls outside ``[0, k]`` on 12
  of the 853 scored points — as far out as 524.70 on a ``k = 209`` pair and
  -69.49 on a ``k = 35`` one.  A ONE-ULP symmetric perturbation of ``S_a``
  moves the exact high-edge edf by 1.30 to 145.87 df across the geometries
  measured, and by 11.11, 31.83 and 77.54 df on the very pair whose 11.57 df
  this docstring quotes — 40 draws at each of three generator seeds, and a max
  over finitely many draws is a LOWER bound on the spread.  The DERIVATIVE of
  the exact edf in one ulp of the stored moments is 7.073e+01 df on that pair:
  ``lam e (sum |A^-1 S A^-1| |V| + sum |A^-1 V A^-1| |S|)`` at ``e = 2^-52``,
  evaluated in arb because ``A^-1`` in float64 is meaningless at
  ``cond(A) = 7.2e+15``.  It is a derivative and NOT a bound on a finite step:
  that needs ``r = ||dA|| / min|eig(A)| < 1``, and here ``r = 6.12e+01``, so a
  one-ulp step can cross a singularity of ``A`` and nothing — not 70.7 df, not
  a multiple of it — bounds the exact edf at this point.  The 77.54 df draw
  already exceeds the derivative.  What the moments DO determine is measured
  instead, and is an observable rather than a scale: the directions of ``A``
  below ``lam eps |S|_2 = 4.72e-06`` carry 3.11 df of ``V_eff``'s mass at this
  high edge and 0 df at the same pair's low edge.  Three degrees of freedom of
  the answer rest on a direction whose own eigenvalue is round-off.  The
  divergence is inside that, not above it.  Where the
  high edge IS determined — no narrow band, spread 1.0e-04 df — both paths are
  right: 18.999995 and 18.999936 against a certified 18.999990.
* **The low edge IS determined** — the exact first-order sensitivity is
  5.554e-03 df on the ``ps(8)`` pair and 8.284e-05 df on the ``ns(6)`` one
  below, and random one-ulp draws reach 1.2x and 2.2x of those, so the
  derivative is very nearly attained rather than an underestimate.  It is NOT a
  certified finite-step bound, and the tests say so rather than implying
  otherwise: put a one-ulp BALL on every stored entry and arb returns a radius
  of 1.5109 df on the ``ps(8)`` low edge and refuses the ``ns(6)`` one
  outright.  Which bound is used DOES matter on one of the two, and that is
  disclosed rather than smoothed over: the ``ps(8)`` low-edge error is 0.525 df,
  which is 94x the derivative but only 0.35x the ball radius, so interval
  arithmetic does not certify it.  The ``ns(6)`` one needs no bound at all —
  this path reads 0.245 where the certified value is 30.989, which is 0.8% of
  the answer on a ``k = 35`` pair, and no noise model puts 99.2% of a degrees-
  of-freedom quantity inside round-off.  That is the case the verdict rests on.
  And there THIS path is the wrong one.  Over the
  20 sweep points where both paths clamp to one lambda and the answer is
  attainable to 1e-2 df, the arrow error has median 1.17 df and maximum
  30.74 df and lands inside the attainable band on 3 of 20; the dense path's
  clamped rung has median 6.9e-05 df, maximum 2.7e-02 df, and lands inside on
  19 of 20.  Worst measured anywhere: an ``ns(6)`` pair, 6 levels, 12 rows in
  each, two inside a 1e-1 band, where this path reads 0.245 against a certified
  30.989 — ``rank - lambda tr(A^-1 S)`` splits as 35.00 - 34.75 where the exact
  split is 35.00 - 4.01.  That is not an edf-only error: ``z`` divides by the
  edf, so the same pair ranks at ``z = 0.13`` from the dense path and
  ``z = 35.42`` from this one at a budget of 2.

**THE LOW-EDGE ERROR DOES NOT SHRINK AS THE LEVEL COUNT GROWS**, which is what
decides whether any of the above reaches the regime this kernel actually runs
in.  Every graded point above is a pair the DENSE path can also score, and this
path is entered only where that one is refused — above ``k = 1357`` for a
penalized block at the default ``max_cells``, so ``L`` past about 124 for a
``ps(8)`` margin.  Walking one family (``ps(8)``, 12 rows in every level, four
inside a 1e-3 band, seed 3) from ``L = 6`` to ``L = 80``, which is ``k = 55`` to
``k = 869`` — 64% of the way to that refusal — against the same certified
oracle at each pair's own low-edge lambda:

    L      6      12      20      40      60      80
    k     55     121     209     429     649     869
    arrow  +10.89  +3.97   +4.14   +5.16   +4.28   +4.55
    dense  -1.4e-02 +8.3e-05 -1.3e-04 -7.0e-05 -1.6e-03 +1.3e-03
    noise   1.7e-02  1.2e-03  1.4e-03  4.4e-03  5.4e-03  3.2e-03

The dense clamped rung tracks the oracle to 1.6e-03 df at every width; this
path sits at 4 to 5 df and is flat in ``L``.  The extrapolation past the
routing threshold is therefore unfavourable rather than benign — but it is
still an extrapolation, since an arb oracle is ``O(k^3)`` in ball arithmetic
and ``k = 869`` already costs 319 s.

**WHAT "THE TARGET IS ``tr(A^-1 V_eff)``" HIDES.**  The comparison above grades
this path against an arb evaluation of ``tr(A^-1 V_eff)`` with a TRUE inverse,
and the dense path it is compared to does not compute that.
:func:`superglm.screening._score_stat._edge` falls back to
``numpy.linalg.pinv``, which zeroes rather than inverts every direction below
``1e-15`` of the largest singular value — NumPy's back-compatible default for a
call that passes neither ``rcond`` nor ``rtol`` — and at the high edge that
fallback is the branch that runs: 4 of 209 directions dropped on the ``ps(8)``
pair above at a cut of 2.1261e-05, worth -8.8838e-01 df of the gap between the
dense reading and the certified oracle.  At the LOW edge, where this path's own
error is graded,
``cho_factor`` succeeds on both pairs and ``pinv`` would drop nothing, so the
low-edge verdict is unaffected — but any fix that adopts the dense path's
estimator adopts its truncation with it, and that has to be a decision rather
than an inheritance.

**EVERY LOW-EDGE MAGNITUDE ABOVE IS ONE INTERPRETER'S, AND SO IS THE SIGN.**
Same numpy 2.4.2 and scipy 1.18.0, same fixtures and seeds, three runs:

    quantity              local 3.14   CI 3.12     CI 3.14
    arrow error, ps(8)    -0.525 df    -2.780 df   +0.0825 df
    arrow error, ns(6)    -30.74 df    -0.0358 df  --

The ``ns(6)`` reading quoted above as 0.245 comes back as **30.9529** on CI's
3.12, against a certified 30.9887 — the collapse does not happen there at all.
And on CI's 3.14 the ``ps(8)`` error changes SIGN, this path reading 172.9870
against the dense path's 172.9045.  So neither "0.245 against 30.989" nor "this
path reads low" is a property of the algorithm.

WHAT IS PORTABLE, on every run observed: the two paths DISAGREE at the low edge,
by 0.0825 to 30.74 df, where a repaired arrow path would agree to the dense
path's own ~1e-4; and the DENSE path matches the certified oracle inside its
pinned tolerance every time (1.27e-04 and 9.4e-05 against 6e-2).  Those two
together are the verdict — they disagree and the dense one is accurate — and
they need neither the sign nor the size of this path's error.  The tests assert
exactly that and nothing more.

The pins are in tests/test_screening_edf_target.py, three of them
``xfail(strict=True)`` because they are the defect rather than the contract,
and each one carries the first-order noise floor of its own point beside it.

**WHAT THE REACH-AWARE COUNT IS WORTH, MEASURED WIDE RATHER THAN ON ONE
GEOMETRY.**  Against a 50-digit oracle evaluated at each path's own lambda,
over 6 levels x 12 rows at unit weight with two levels squeezed into a band of
``x`` of 16 widths from 1e-1 to 1e-6, at 3, 4, 6 and 8 knots in all five bases
this library assembles (``ps``, ``cr``, ``bs``, cardinal ``cr``, ``ns``): of
312 well-posed points the reach-aware count in :func:`block_ranks` is nearer
the oracle at the high edge on 86, farther on 12 and unchanged on 214, and the
worst error it INTRODUCES anywhere is 1.0 df.  Well posed excludes the 8 of
those 320 where this path refuses at the bracket's high edge, and 2 of the 8
refuse only HERE: at ``bs(6)`` width 1e-5 and ``bs(8)`` width 2.15e-4 the
reach-aware count is one rank lower than the one it replaces, the penalty
trace it is subtracted from is bit-identical, and the difference puts the edf
0.18 and 0.34 df BELOW zero — past round-off, so
:class:`_UnstableStructuredEDFError` declines a value that was returned before
(0.824 and 0.664 df).  Nothing moved the other way.  That refusal is the
bracket's own EDGE rather than a budget bisecting inside it, and it is
user-visible on one of the two: at ``bs(8)``/2.15e-4 a four-budget
:func:`structured_ladder` goes from a malformed single rung of 16.0 df to
``None``.  On the narrower family the residue floor governs — a two-column
``cr`` margin over 24 seeds x 4 level layouts, 96 points — it is nearer on
29, farther on 16 and unchanged on 51, and the worst error introduced is
2.0 df.  The class is narrowed and NOT closed.

**THE LOW EDGE IS ALMOST, BUT NOT ENTIRELY, UNTOUCHED.**  Over those same 320
points the arrow value at the ladder's LOW edge moved on 2, and moved AWAY
from the oracle on both — ``cr`` at 8 knots, band widths 2.15e-6 and 1e-6,
error 2.149 -> 3.149 df and 1.277 -> 2.277 df.  It is bit-identical on the
other 318, on all 96 of the two-column ``cr`` points, and on every fixture
this suite carries.

On the ``ps(8)`` pair of the first paragraph, walked across eleven band widths
from 1e-1 to 1e-4, the level ranks counted here still disagree with what
:func:`superglm.screening._arrow._psd_pinv` keeps, by up to +3 at the high
edge and on 6 of those 11 widths (9 of 11 before the reach-aware count).  The
invariant :func:`block_ranks` is written to satisfy — count what the inverse
resolves, no more — is improved, not achieved.

**THAT +3 IS ONE GEOMETRY'S NUMBER AND NOT A BOUND**, so the measured range is
stated instead.  Over 192 further pipeline configurations (``ps``/``cr``/
``bs``/``ns`` at 3 to 8 knots, ``L`` in 6 to 20, 3 to 30 rows per level, with
and without a narrow band, two seeds, two widths) the worst HIGH-edge
disagreement is also +3; on the four wide pairs of the shape this kernel
exists for it is +10, +9, +9 and +9 at the lambda each clamps to; and at the
LOW edge, on pairs carrying levels with fewer distinct covariate values than
the margin has columns, it reaches +63.

A NONZERO DISAGREEMENT IS NOT BY ITSELF AN EDF ERROR, and on the wide pairs
the disagreeing count is the accurate one.  ``block_ranks`` enters ``edf``
only through ``rank_term``, so ``edf`` counted here and ``edf`` counted from
the inverse's own keep differ by exactly that disagreement.  Against a
40-digit oracle for ``tr((V_eff + lambda S)^-1 V_eff)`` on the delivered
moments (108.996797 / 108.998928 / 107.999757 / 109.000276) the four wide
pairs come back at 108.985705 / 108.997280 / 108.005895 / 109.001261 — inside
0.012 df — while adopting the inverse's own count would put them 9 to 10 df
low.  The low-edge disagreement is a different, PRE-EXISTING defect: it is the
unconditional contrast term ``np.where(p.m > 0.0, 1, 0)`` counting a dimension
the pair does not have, ``rank(M)`` counts it too so it does not cancel, and
``origin/master`` reproduces it identically.  It is not this branch's to fix
and is tracked separately.

The ladder's edf is also not monotone in lambda on that same pair, and that
predates any of this: walking 81 lambdas across the bracket, the arrow edf
INCREASES at 14 of the 80 steps, by as much as 47.36 df, identically before
and after the reach-aware count.  A budget inside the bracket bisects rather
than clamps, so lowering ``edf_hi`` converts clamped rungs into searching
ones and a search can land on a step it cannot attain — the pair then comes
back as a NaN row instead of four scored rungs.  The defect is reached by
that change, not caused by it.

The dense path carries its own version of the same failure and it is not
touched here: on one of four wide pairs of the shape this kernel exists for
it reports 1.03 df BELOW a high-precision oracle at the high edge, where
``numpy.linalg.pinv``'s inherited default ``rcond`` drops a direction the
penalty leaves free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.screening._arrow import _RCOND, _solve_floor, factor_arrow
from superglm.screening._score_stat import ScreenedPair

# What to believe about an assembled ``S_a``'s residue in a direction its own
# eigendecomposition reports as an EXACT zero -- and only then, since a small
# nonzero residue is a measurement and this is not.  One round-off unit on the
# penalty's largest eigenvalue.  The regime, the evidence that it is real, and
# the error this carries are all in :func:`block_ranks`.
_PENALTY_RESIDUE_FLOOR = float(np.finfo(np.float64).eps)

_EDF_TOL = 1e-6
_EDF_ROUNDOFF_FACTOR = 64.0
_EDF_ABSOLUTE_DUST = np.finfo(np.float64).eps ** 2
_MAX_BISECT = 200
_TRACE_CHUNK_DOUBLES = 262_144

# The most steps one rung's bisection can take.  It halves the log of a
# bracket spanning 1e20 and stops when the two ends are within 1e-12 of each
# other relatively, so it exhausts at ceil(log2(ln(1e20) / ln(1 + 1e-12)))
# steps whatever the data does -- _EDF_TOL usually stops it sooner (measured
# 26 to 28 on the pairs that search), but nothing guarantees that, and a
# ceiling is what lets a caller decide BEFORE the search whether to pay for
# it rather than after.
_MAX_STEPS_PER_RUNG = min(_MAX_BISECT, 46)


class _UnstableStructuredEDFError(FloatingPointError):
    """The arrow rank and inverse disagree by more than numerical dust."""


def _edf_roundoff(*values: float) -> float:
    """Scale-aware dust allowance for EDF identities and ordering."""
    return max(
        _EDF_ABSOLUTE_DUST,
        _EDF_ROUNDOFF_FACTOR
        * np.finfo(np.float64).eps
        * sum(abs(float(value)) for value in values),
    )


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
    # None means the global centered-row rank lies inside its certified
    # numerical ambiguity band; the structured route must be refused.
    profiled_trace: float | None

    @property
    def dims(self) -> tuple[int, int]:
        return self.U.shape  # (L, k_a)


def _centered_level_factors(B: NDArray, W: NDArray) -> NDArray:
    """Return QR factors of each level's centered weighted spline rows.

    The raw-moment identity ``B' W B - (B' w)(B' w)' / sum(w)`` is unusable
    here: the trace this module needs can be twelve or more orders below both
    terms.  This two-pass form subtracts each level's mean from the basis
    rows first.  It then returns a square ``R_l`` whose Gram is the centered
    geometry, without forming that Gram:

        R_l' R_l = (sqrt(W_l) D_l)' (sqrt(W_l) D_l).

    Shifting every row by the first basis row before taking the mean makes the
    centering invariant to a large common offset without subtracting that
    offset twice.  Zero-mass levels contribute an exact zero matrix.
    """
    B = np.asarray(B, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n, k = B.shape
    n_levels = W.shape[1]
    if n_levels == 0:
        return np.empty((0, k, k), dtype=np.float64)
    if n == 0 or k == 0:
        return np.zeros((n_levels, k, k), dtype=np.float64)

    shifted = B - B[0]
    mass = W.sum(axis=0)
    means = np.zeros((n_levels, k), dtype=np.float64)
    np.divide(W.T @ shifted, mass[:, None], out=means, where=mass[:, None] > 0.0)
    centered = shifted[:, None, :] - means[None, :, :]
    centered *= np.sqrt(W)[:, :, None]
    raw = np.linalg.qr(np.moveaxis(centered, 1, 0), mode="r")
    if raw.shape[1] == k:
        return raw
    factors = np.zeros((n_levels, k, k), dtype=np.float64)
    factors[:, : raw.shape[1], :] = raw
    return factors


def _combine_row_factors(left: NDArray, right: NDArray) -> NDArray:
    """Compact two weighted-row factors without squaring either one."""
    return np.linalg.qr(np.concatenate((left, right), axis=0), mode="r")


def _representative_projection(
    row_factor: NDArray,
    *,
    n_rows: int,
    n_levels: int,
) -> tuple[NDArray, NDArray] | None:
    """Return a stable representative span and its structural null action.

    Column-pivoted QR of the small global weighted-row factor chooses
    representative basis columns ``active``.  In pivot order,

        Z = Z_active [I, C],       R11 C = R12.

    A coefficient action using only the active rows is therefore sufficient:
    every other representative differs only by a null-space action and has
    the same fitted rows and residual energies.  ``null_action`` is
    ``I - P`` built structurally as active rows ``[0, -C]`` and inactive rows
    ``[0, I]``.  No cancelling ``I - H^+ H`` is formed.

    The cutoff is the square root of the Hermitian pseudo-inverse policy used
    by the dense path.  Rank is refused, rather than guessed, when a pivot
    intersects its QR backward-error interval.  Each Householder reduction
    contributes an additive ``O(eps * leading_scale)`` perturbation; the
    conservative operation depth below covers one ``n_rows x k`` local QR per
    level, the sequential ``2k x k`` suffix merges, and the final ``k x k``
    pivoted QR.  Relative to a ``sqrt(k*eps)`` cutoff that uncertainty is
    ``O(sqrt(eps))`` — the scale on which row/level order can otherwise flip
    a retained direction.  Refusal lets routing fall back instead of making
    that platform-specific rank choice observable.
    """
    k = row_factor.shape[1]
    _, pivoted, permutation = scipy.linalg.qr(
        row_factor,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(pivoted))
    if diagonal.size == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty_like(row_factor),
        )

    eps = np.finfo(np.float64).eps
    leading_scale = float(diagonal[0])
    if leading_scale <= np.finfo(np.float64).tiny:
        return (
            np.empty(0, dtype=np.intp),
            np.zeros_like(row_factor),
        )
    cutoff = np.sqrt(max(k, 1) * eps) * leading_scale
    reduction_depth = (
        max(int(n_rows), 1) * max(k, 1)
        + 2 * max(int(n_levels), 1) * max(k, 1) ** 2
        + max(k, 1) ** 2
    )
    # Cap the band at half the cutoff.  ``reduction_depth`` grows with n_rows
    # and n_levels while ``cutoff`` does not, so an uncapped band overtakes it
    # at ``reduction_depth >= 2**22 * sqrt(k)`` -- ``sqrt(eps)`` is exactly
    # ``2**-26`` -- and from there ``[cutoff - unc, cutoff + unc]`` contains
    # zero, so EVERY pivot reads as ambiguous and the projection refuses a
    # geometry it resolved perfectly well at a smaller budget.  That inverts
    # the documented monotone budget behaviour: raising ``max_cells`` made a
    # scored pair go dark.
    #
    # Half the cutoff is the largest band that cannot swallow zero, so the
    # refusal keeps meaning "this pivot is genuinely too close to the cutoff to
    # call" rather than "the flop count got large".  Below the crossover the
    # min is inert and the band is bit-identical to before.
    uncertainty = min(16.0 * eps * reduction_depth, 0.5 * np.sqrt(max(k, 1) * eps))
    uncertainty *= leading_scale
    if np.any(np.abs(diagonal - cutoff) <= uncertainty):
        return None
    rank = int(np.count_nonzero(diagonal > cutoff))
    active = np.asarray(permutation[:rank], dtype=np.intp)
    inactive = np.asarray(permutation[rank:], dtype=np.intp)

    null_action = np.zeros((k, k), dtype=np.float64)
    if rank and inactive.size:
        relation = scipy.linalg.solve_triangular(
            pivoted[:rank, :rank],
            pivoted[:rank, rank:],
            check_finite=False,
        )
        null_action[np.ix_(active, inactive)] = -relation
    null_action[inactive, inactive] = 1.0
    return active, null_action


def _aligned_representative_actions(
    active: NDArray,
    local: NDArray,
    other: NDArray,
) -> tuple[NDArray, NDArray]:
    """Project local and complementary rows in one aligned QR coordinate.

    ``local`` and ``other`` are row factors for disjoint pieces of the same
    global centered design.  Stacking their representative columns and
    applying the resulting orthogonal coordinate to the aligned right-hand
    sides gives both least-squares actions directly:

        G = [local_active; other_active] = Q R
        A_q(active)  = R^-1 Q_local' local
        A_-q(active) = R^-1 Q_other' other.

    This is algebraically the same representative fit as ``H^+ H_q`` and
    ``H^+ H_-q``, but it never forms ``G'G`` or solves through ``R'R``.  The
    two right-hand sides share one triangular solve.
    """
    k = local.shape[1]
    projection = np.zeros((k, k), dtype=np.float64)
    complement = np.zeros((k, k), dtype=np.float64)
    if active.size == 0:
        return projection, complement

    aligned = np.concatenate((local[:, active], other[:, active]), axis=0)
    Q, triangular = np.linalg.qr(aligned, mode="reduced")
    right_hand_sides = np.concatenate(
        (Q[:k].T @ local, Q[k:].T @ other),
        axis=1,
    )
    actions = scipy.linalg.solve_triangular(
        triangular,
        right_hand_sides,
        check_finite=False,
    )
    projection[active] = actions[:, :k]
    complement[active] = actions[:, k:]
    return projection, complement


def _trace_chunk_width(n_rows: int, n_cols: int, n_levels: int) -> int:
    """Bound centered-row and factor temporaries by a small fixed chunk."""
    per_level = max(int(n_rows) * int(n_cols), int(n_cols) ** 2, 1)
    return max(1, min(int(n_levels), _TRACE_CHUNK_DOUBLES // per_level or 1))


def _profiled_curvature_trace(
    B: NDArray,
    W_cell: NDArray,
    level_rows: NDArray,
) -> float | None:
    """Compute ``tr(V_eff)`` as an exact sum of squared residual norms.

    Profiling the intercept and categorical main centers the spline geometry
    separately inside every level.  Write that weighted centered design as
    ``Z_l = sqrt(W_l) D_l``.  A QR factor ``R_l`` has the same action norm:
    ``||Z_l A||_F = ||R_l A||_F`` for every coefficient action ``A``.

    A column-pivoted QR of the global row factor chooses a stable
    representative span without ever forming the ill-conditioned normal
    matrix ``H = sum_l Z_l' Z_l``.  For emitted interaction block ``q``, let
    ``A_q`` be its representative projection coefficients, ``A_-q`` the
    coefficients for all other levels, and ``N`` the structural null action
    returned by :func:`_representative_projection`.  The residual action is
    assembled additively as ``N + A_-q``.  The diagonal Schur-complement trace
    is the exact nonnegative factor norm

        ||R_q (N + A_-q)||_F^2 + ||R_-q A_q||_F^2.

    Summing those nonnegative terms over emitted levels is exactly
    ``tr(V - C' M^+ C)``.  Unlike that difference, it remains representable
    when the mains absorb all but round-off of the interaction.

    ``R_-q`` is a QR compaction of a prefix and precomputed suffix of actual
    row factors.  It is never obtained by subtracting level ``q`` from a
    global Gram.  Only that suffix is a level-sized trace scratch stack; the
    centered factors are recomputed in bounded chunks on the forward pass,
    then all trace scratch is discarded before the arrow ladder.  Work and
    memory remain linear in the level count; no SVD, normal-matrix inverse, or
    dense ``V_eff`` is formed.
    """
    B = np.asarray(B, dtype=np.float64)
    W_cell = np.asarray(W_cell, dtype=np.float64)
    level_rows = np.asarray(level_rows, dtype=np.intp)
    n_rows, k_a = B.shape
    n_levels = W_cell.shape[1]
    if level_rows.size == 0 or k_a == 0:
        return 0.0

    chunk = _trace_chunk_width(n_rows, k_a, n_levels)
    suffix = np.empty((n_levels + 1, k_a, k_a), dtype=np.float64)
    suffix[-1] = 0.0
    for stop in range(n_levels, 0, -chunk):
        start = max(0, stop - chunk)
        factors = _centered_level_factors(B, W_cell[:, start:stop])
        for level in range(stop - 1, start - 1, -1):
            suffix[level] = _combine_row_factors(factors[level - start], suffix[level + 1])

    representative = _representative_projection(
        suffix[0],
        n_rows=n_rows,
        n_levels=n_levels,
    )
    if representative is None:
        return None
    active, null_action = representative
    emitted = np.zeros(n_levels, dtype=bool)
    emitted[level_rows] = True
    prefix = np.zeros((k_a, k_a), dtype=np.float64)
    trace = 0.0
    correction = 0.0

    for start in range(0, n_levels, chunk):
        stop = min(n_levels, start + chunk)
        factors = _centered_level_factors(B, W_cell[:, start:stop])
        for level in range(start, stop):
            local = factors[level - start]
            if emitted[level]:
                other = _combine_row_factors(prefix, suffix[level + 1])
                projection, complement = _aligned_representative_actions(active, local, other)
                residual = null_action + complement
                term = float(
                    np.sum(np.square(local @ residual)) + np.sum(np.square(other @ projection))
                )
                # Neumaier-style compensated accumulation keeps the final
                # scalar independent of level order down to the factor error.
                updated = trace + term
                if abs(trace) >= abs(term):
                    correction += (trace - updated) + term
                else:
                    correction += (term - updated) + trace
                trace = updated
            prefix = _combine_row_factors(prefix, local)

    return trace + correction


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
    del AA
    # V persists into the ladder, but its n_a*k_a^2 construction scratch has
    # been released before the one level-sized suffix stack below is formed.
    profiled_trace = _profiled_curvature_trace(B_a, W_cell, level_rows)

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
        profiled_trace=profiled_trace,
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


@dataclass(frozen=True)
class _RankGeometry:
    """:func:`block_ranks` with the ``lambda`` divided out of it.

    The free set is the SAME at every positive ``lambda``: both sides of
    ``reach <= _solve_floor(k_a + 1) * top`` carry one factor of it, so it
    cancels.  Everything the mask then selects is ``lambda``-free too — the
    projection ``N``, the Schur complement, and the batched eigendecomposition
    of it, which is the expensive part.  Only the threshold moves, and it moves
    by a scalar, so a rung reached by bisection can rescale rather than rebuild.

    ``unit_top`` and ``unit_residue`` are the two thresholds at ``lambda = 1``;
    at any other they are ``lam`` times these.  They are held apart from the
    eigenvalues for exactly that reason.
    """

    ranks: NDArray  # (L,)  contrast + the directions the penalty reaches
    curvature_eigs: NDArray  # (L, d) ascending, the free block's own spectrum
    dust: NDArray  # (L,)  ``_RCOND`` on the level's raw moments
    unit_top: float
    unit_residue: float
    any_free: bool


def _rank_geometry(p: SplineCatPair, all_free: bool = False) -> _RankGeometry:
    """Build the ``lambda``-free half of :func:`block_ranks` once.

    ``all_free`` is the zero-``lambda`` reading, where no direction is reached
    at all and the count is the unpenalized block's own rank.  It is a separate
    argument rather than a ``lambda`` of 0 flowing through the comparison
    because the two give different masks and only one of them can be cached.
    """
    _, k_a = p.dims
    S_a = 0.5 * (p.S_a + p.S_a.T)
    sigma, directions = np.linalg.eigh(S_a)
    unit_reach = np.abs(sigma)
    unit_top = float(unit_reach.max()) if unit_reach.size else 0.0
    free = (
        np.ones(unit_reach.shape, dtype=bool)
        if all_free
        else unit_reach <= _solve_floor(k_a + 1) * unit_top
    )
    ranks = np.where(p.m > 0.0, 1, 0) + int(np.count_nonzero(~free))
    if not free.any():
        empty = np.zeros((p.dims[0], 0), dtype=np.float64)
        return _RankGeometry(ranks, empty, empty, unit_top, 0.0, False)

    N = directions[:, free]
    mass = np.where(p.m > 0.0, p.m, 1.0)
    projected = p.c @ N
    curvature = np.einsum("ip,lij,jq->lpq", N, p.V, N, optimize=True)
    curvature -= (projected[:, :, None] * projected[:, None, :]) / mass[:, None, None]
    curvature = 0.5 * (curvature + np.swapaxes(curvature, -1, -2))
    return _RankGeometry(
        ranks,
        np.linalg.eigvalsh(curvature),
        _RCOND * (np.einsum("lpp->l", p.V) + p.m),
        unit_top,
        float(unit_reach[free].max()),
        True,
    )


def block_ranks(p: SplineCatPair, lam: float, geometry: _RankGeometry | None = None) -> NDArray:
    """Every level block's rank, counted to AGREE with the inverse at ``lam``.

    ``edf`` is ``rank(A) - lambda tr(A^-1 S)`` and the two halves have to be
    counted against each other: a direction the rank COUNTS and the inverse
    DROPS contributes ``1 - 0``, a whole degree of freedom with no penalty
    offset.  ``_solve_floor`` in :mod:`superglm.screening._arrow` states the
    inverse's side of that bargain; this is the rank's.

    The penalty splits the block.  Where ``S_a`` has a genuine eigenvalue the
    inverse normally resolves that direction and ``tr(A^-1 S)`` subtracts the
    right share back, so it is counted — as is the level's own contrast, which
    cancels against ``rank(M)``.

    NORMALLY, NOT ALWAYS, and the exception is named rather than assumed away.
    An earlier form of this paragraph claimed ``lambda sigma_j`` was above the
    inverse's own resolution EVERYWHERE the ladder brackets; it is not, and the
    counterexample is a level carrying fewer distinct covariate values than the
    margin has columns.  On ``Categorical() x Spline(kind="ps", n_knots=5)``,
    8 categorical levels — 7 level blocks after treatment coding — 3 rows in
    each, 4 of them inside a 1e-3 band of ``x``, unit weights,
    ``default_rng(3)``, at the ladder's LOW edge:
    ``lambda min|sigma_genuine|`` is ``4.14e-15`` against
    :func:`~superglm.screening._arrow._psd_pinv`'s own cut of ``6.28e-15`` to
    ``7.89e-15`` on the blocks it inverts — BELOW it on all seven levels, and
    the two counts part, 63 here against 56 there.  Populate the same margin
    properly (``ps(8)``, 20 levels, 30 rows each) and the claim holds as
    written: ``2.69e-13`` against cuts of ``8.02e-14`` to ``1.09e-13``.  The
    low-edge disagreement is not this function's to fix — it is driven by the
    unconditional contrast term below, ``rank(M)`` counts the same dimension,
    and ``origin/master`` produces it identically — but it is not covered by
    the guarantee this paragraph used to state, so the guarantee is withdrawn.

    What is left is ``null(S_a)``, and there the RANK is the whole answer,
    because at the ladder's high edge the block's largest eigenvalue is
    ``lambda sigma_max`` and a free direction sits below the relative cut of
    any inverse.  Its exact contribution is ``v / (v + lambda s)``, so it is
    counted when the level's own curvature beats the penalty still reaching
    it and dropped when it does not.  ``v`` is the block's own Schur
    complement ``V_q - c_q c_q' / m_q`` — the level's curvature after its own
    contrast has absorbed what it can, which is what the pair's rank turns
    on, and the only place a level's SHARE OF THE WEIGHT enters.

    Scaling each block by its own trace, which is how this was counted
    before, divides that share out by construction.  Measured on
    ``_vanishing_mass_pair(1e-12)``: a level holding 6.2e-14 of the weight
    (free curvature 1.7e-12) and one holding 6.2e-02 of it (free curvature
    2.1e+00) present balanced free eigenvalues of 5.51e-03 and 5.67e-03 —
    indistinguishable, with twelve orders between them in the quantity that
    decides.  Three such levels bought that pair three degrees of freedom it
    did not have.

    ``reach[free].max()`` IS A SCALAR ON PURPOSE, AND IT IS DECISIVE ONLY
    ABOVE NULLITY ONE.  Where the free subspace has more than one dimension
    the free reaches differ, and the whole subspace is still judged against
    the LARGEST of them rather than each direction against its own.  That is
    not a coarsening of a finer rule available for free — it is the matching
    half of the bargain at the top of this docstring.  ``_psd_pinv``'s cut is
    likewise ONE number per block, relative to that block's own largest
    eigenvalue, so a per-direction refinement of the rank's side refines
    against a reference the inverse does not share, and every direction it
    adds arrives as ``1 - 0``.  Measured on
    ``Categorical() x Spline(kind="ps", k=5, degree=3, m=3)`` — ``k_a = 4``,
    nullity 2 — 6 levels, 12 rows in each, two inside a 1e-3 band, over seeds
    0 to 39 at the ladder's high edge: the arrow inverse resolves 23
    directions on every one of the 40, this rule counts 23 or 24, and a
    per-direction rule (each free direction against the reach it carries,
    i.e. the generalized eigenvalues of ``(curvature_q, diag(reach_free))``
    above one) counts 24 on all 40 — above what the inverse resolves
    everywhere.  It disagrees with this rule on 15 of the 40, always by
    exactly +1.000000 df, and against a 50-digit oracle for
    ``tr((V_eff + lambda S)^-1 V_eff)`` (cross-checked at 90 digits) it is
    nearer on 7 of those 15 and farther on 8 — but the split is not luck:
    on the 5 draws where this rule is already within 0.5 df of the oracle it
    is farther on 5 of 5, turning 0.059 df of error into 1.059.  Its wins are
    a fixed +1 partially cancelling the route's OWN error, which on that
    narrow-band family runs to 8.00 df at the high edge.  Nullity above one is
    reached by no shipped default — measured on the centered ``S_a`` this
    receives, nullity is ``m - 1`` for ``ps`` (0, 1, 2, 3 at ``m`` = 1 to 4)
    and for ``bs`` (``m = 4`` refused, order > degree), but ``cr`` stops at 1
    (``m = 3`` still gives 1, ``m = 4`` refused) and ``ns`` is 0 until
    ``m = 4`` — so at the default ``m = 2`` every one of the four is nullity 1
    or 0 and this collapse is a no-op.  The choice is pinned by
    ``test_a_multi_null_penalty_is_ranked_against_one_reach_not_per_direction``,
    which had to reach for ``m = 3`` to test it at all.

    ``s`` IS THE EIGENSOLVER'S OWN ANSWER, WHATEVER ITS SIGN.  A spline
    penalty's null space is null in exact arithmetic and round-off in float,
    and how much round-off depends on the DATA as well as on the basis, so the
    geometry has to be named with the number.  On 6 levels, 12 rows in every
    level, unit weights and ``x`` uniform on [0.05, 0.95] from
    ``default_rng(3)``, scored as ``Categorical() x Spline(kind, n_knots=k)``,
    ``|sigma_min| / sigma_max`` comes out at 0.0100, 0.0391, 0.0486, 0.0512,
    0.0567, 0.0847, 0.0887, 0.1474, 0.2076, 0.2159, 0.4337 and 0.6157 times
    ``eps`` over the twelve ``ps``/``cr``/``bs`` margins at 3, 4, 6 and 8
    knots, with the sign varying between them (``ns`` is full rank at all four
    and has no free direction at all).  Read that as a tenth of a round-off
    unit to a whole one rather than as a bound: the same twelve margins drawn
    from ``default_rng(11)`` instead measure 0.0547 to 0.3340 eps, and the
    margin that is smallest on one draw is not the smallest on the other.
    Spelling the knot count ``k=``
    is a different experiment — ``ps`` and ``bs`` reject ``k=3`` and ``k=4``
    outright, needing 5 at degree 3, so only eight of those twelve margins
    exist, and ``cr(k=3)``, the two-column margin, lands in the exact-zero
    regime below rather than resolving anything.

    Round-off is therefore what the reach IS here, and ``np.abs`` is taken
    before anything else: clamping a negative residue up to zero would report
    no residue at all and substitute the floor for one the eigensolver DID
    resolve.  Measured on the ``cr(n_knots=3)`` family
    tests/test_structured_screening.py draws, replacing ``np.abs(sigma)`` with
    ``np.maximum(sigma, 0.0)`` is invisible on the seeds whose residue happens
    to come out positive and costs exactly 1.000000 df on four of the twelve
    scanned — the seeds where the sign is negative AND a level's curvature
    sits between the residue and the floor.  The same round-off decides which
    directions are free at all: ``_solve_floor``, the inverse's own cut, is
    ``(k_a + 1) eps`` of the largest reach, which is 13x the residue at the
    margin where the two come closest (``bs`` at 4 knots) and 797x it where
    they are farthest (``ps`` at 4 knots), so the split is not close.

    :data:`_PENALTY_RESIDUE_FLOOR` replaces that reach in ONE case: where the
    eigensolver returned an EXACT zero for every free direction.  ``reach`` is
    already an absolute value, so ``reach[free].max() == 0.0`` is exactly that
    case and no tolerance is involved — deliberately, because a residue merely
    SMALL is still a measurement and overriding it is what drops real degrees
    of freedom.  Measured against an exact-rational oracle (agreeing with
    80-digit mpmath to nine decimals) on a ``cr(3)`` pair, 6 levels, 12 rows
    in every one, unit weights, two levels inside a 2e-3 band of ``x``: an
    unconditional floor cost exactly 1.000000 df on three of eight seeds, one
    of them a 435x accuracy regression (0.0023 df of error becoming 1.0023),
    on directions whose exact shares were 0.946, 1.000 and 1.000.

    The exact-zero regime is REAL and is not exotic.  A two-column spline
    margin carries a rank-one penalty, and ``eigh`` returned an exact 0.0 for
    its residue on 52 of 192 ``cr(k=3) x Categorical`` fixtures scanned,
    while exact rational arithmetic on the same float matrix put that residue
    at ``0.0012`` to ``0.1773 eps`` times ``sigma_max`` — nonzero every time.
    Believing the zero costs whole degrees of freedom: of four wide pairs of
    the shape this kernel exists for (hundreds of levels, a two-column
    margin), three report an exact zero, and with the floor removed they come
    back at 114.00 / 113.01 / 114.00 against a high-precision oracle's
    109.00 / 108.00 / 109.00 — five each.  The fourth, whose residue the
    eigensolver DOES resolve at 8.7e-19 of ``sigma_max``, never reads the
    floor and is unmoved either way.

    THE FLOOR IS A BOUNDED-ERROR CONVENTION, NOT AN UNDECIDABLE CASE, and the
    band is stated rather than hidden.  ``1 eps`` over-states every residue
    measured above by 5.6x to 830x, so a direction whose curvature sits just
    under the floor is dropped although the exact share it carries is close to
    one: over those 52 fixtures the largest exact share actually discarded was
    ``0.9910``.  Inflating the floor is worse in the same currency: eleven
    directions over those same fixtures fall in the ``(1, 3] eps`` band, their
    exact shares run from ``0.9641`` to ``0.9954``, and a 3x floor discards
    all eleven.  A RANGE over a stated number of directions is what that scan
    supports; an earlier form of this sentence claimed ``0.9942`` as a lower
    bound on them, which was the artefact of a max taken over per-fixture
    MINIMA — an upper bound on the minimum, the opposite of what was wanted.
    Both sides of the constant are pinned in
    tests/test_structured_screening.py rather than left free.

    The second floor is the one this always had.  ``_RCOND`` times the block's
    own trace keeps a direction whose curvature is round-off of the level's
    raw moments out of the count at the LOW edge, where the reach is
    negligible and the Schur complement above is a cancelling difference.
    """
    lam = float(lam)
    # A zero lambda frees EVERY direction rather than the ones the spectrum
    # picks out -- the unpenalized block's own rank, which is what that rung
    # reports -- so it is asked for explicitly instead of emerging from a
    # ``0 <= 0`` comparison, and it never reads a cache built the other way.
    geometry = (
        _rank_geometry(p, all_free=lam <= 0.0) if geometry is None or lam <= 0.0 else geometry
    )
    if not geometry.any_free:
        return geometry.ranks

    # ``unit_reach`` is already an absolute value, so ``residue`` is 0.0 here
    # only where the eigensolver returned a signed zero for EVERY free
    # direction -- the one case where it has reported no residue at all rather
    # than a small one.  A comparison against a tolerance would not do: a
    # residue below the floor is still a MEASUREMENT, and substituting the
    # floor for it is what loses whole degrees of freedom.  Taking the absolute
    # value first is equally load bearing, since the residue is signed.
    top = lam * geometry.unit_top
    residue = lam * geometry.unit_residue
    flattened = residue if residue > 0.0 else _PENALTY_RESIDUE_FLOOR * top
    floor = np.maximum(flattened, geometry.dust)
    return geometry.ranks + (geometry.curvature_eigs > floor[:, None]).sum(axis=-1)


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
    geometry: _RankGeometry | None = None,
) -> tuple[float, float]:
    """``(T, edf)`` at one lambda, from ONE arrow factorization.

    ``rank(V_eff + lambda S)`` is Guttman rank additivity on the bordered
    system — ``rank(K) = rank(M) + rank(A)``.  ``ranks`` carries the level
    blocks' contribution, counted by :func:`block_ranks` AT THIS LAMBDA
    because that is what makes it agree with the inverse the same
    factorization supplies; the border's own rank comes from this
    factorization.
    """
    L, k_a = p.dims
    f = _pair_arrow(p, lam, block_ranks(p, lam, geometry) if ranks is None else ranks)
    b = np.zeros((L, k_a + 1), dtype=np.float64)
    b[:, :k_a] = U_eff
    x, _ = f.solve(b, np.zeros(1 + k_a, dtype=np.float64))
    T = float(np.sum(U_eff * x[:, :k_a]))
    blocks = f.diag_blocks()[:, :k_a, :k_a]
    rank_term = float(f.rank - rank_m)
    penalty_term = lam * float(np.einsum("lpr,rp->", blocks, p.S_a, optimize=True))
    edf = rank_term - penalty_term
    # A PSD penalty and inverse require both penalty_term >= 0 and
    # 0 <= edf <= rank_term.  Violating either side means the factorization's
    # numerical rank and inverse action disagree, and accepting it can make a
    # ladder search converge to a plausible but wrong row.  Correct only
    # round-off at an endpoint; signal anything material so the caller refuses
    # this structured route.
    roundoff = _edf_roundoff(rank_term, penalty_term)
    if (
        not np.isfinite(penalty_term)
        or not np.isfinite(edf)
        or penalty_term < -roundoff
        or edf < -roundoff
        or edf > rank_term + roundoff
    ):
        raise _UnstableStructuredEDFError(
            f"structured EDF is numerically inconsistent: {edf} "
            f"(rank={rank_term}, penalty trace={penalty_term})"
        )
    if penalty_term < 0.0 or edf > rank_term:
        edf = rank_term
    if edf < 0.0:
        edf = 0.0
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
    same NaN row an unaffordable dense pair gets.  ``max_evaluations=None``
    means unbounded.

    **A SEARCHING ladder pays for the rank count, because :func:`block_ranks`
    depends on lambda and so runs once per evaluation rather than once per
    pair.**  Measured on an ``L = 100`` ``ps(8)`` pair with four budgets forced
    to search, ``time.process_time`` with all six thread pools pinned to one,
    interleaved A/B/B/A over 8 pairs: 0.1682 s per ladder against 0.1547 s for
    the lambda-independent count that preceded it, +9.2% at the median of the
    8 paired ratios and slower in every one of them (+5.3% to +14.5%; the
    first four pairs' arm ranges are disjoint, the pooled eight overlap by
    1.4% because the machine's load fell between rounds).  A CLAMPED ladder is
    two evaluations whatever it counts, and ``ns`` returns before any of this,
    so the cost lands only where the bisection does.

    **WHICH IS A GEOMETRY THAT HAD TO BE FORCED, AND THE COUNT SAYS SO.**
    That regression is real but the budgets it needs are not the ones
    screening uses.  Arrow factorizations for a WHOLE ladder at the default
    ``(2, 4, 8, 16)``, counted by instrumenting :func:`_pair_arrow`: 2 for
    each of four wide real pairs (``L = 118``, two-column margin), and 2 for
    ``ps(8)`` at ``L = 50`` and ``L = 100``, ``cr(6)`` at ``L = 100``,
    ``bs(6)`` at ``L = 50`` and ``ps(8)`` at ``L = 20``.  Their edf at maximum
    penalty is 49.00, 99.00 and 108.01 to 109.00 respectively — far above
    every budget in use, so no rung's target falls inside the bracket and
    every one of them clamps.  The same ``ps(8)`` ``L = 100`` pair
    driven at ``(150, 250, 400, 600)``, which is the paragraph above, pays
    120.  The one margin that searches by default is ``ns``: its penalty is
    full rank, edf at maximum penalty is 0, and ``L = 100`` pays 106 — but
    ``free`` is then empty and :func:`block_ranks` returns before any of the
    work this paragraph is about.

    A numerical failure reached while bisecting one target refuses that target,
    not independent targets or already certified edge clamps.  The returned
    list may therefore contain fewer entries than ``budgets``.  If no rung
    survives, ``None`` preserves the pair-refusal signal and lets a speculative
    structured route hand the dense path back.
    """
    if p.profiled_trace is None:
        return None

    U_eff, rank_m = _profile(p)
    # Built once for the whole ladder.  Every lambda this evaluates is strictly
    # positive -- the bracket's own low edge is ``1e-10 * scale`` and the
    # zero-penalty pair returns below without reaching here -- so the cached
    # free set is the one each rung would have computed for itself.
    geometry = _rank_geometry(p)

    def evaluate(lam: float) -> tuple[float, float] | None:
        try:
            return _evaluate(p, U_eff, rank_m, lam, geometry=geometry)
        except _UnstableStructuredEDFError:
            return None

    if not np.any(p.S_a):
        # No penalty to scan, exactly the predicate the dense ladder applies:
        # one rung, at the block's own achieved rank, with lambda0 = 0.  A
        # zero penalty would otherwise make the bracket below infinite and
        # every rung NaN, since inf * 0 is not a number.
        evaluated = evaluate(0.0)
        if evaluated is None:
            return None
        stat, rank = evaluated
        return [ScreenedPair(statistic=stat, edf0=rank, lambda0=0.0) for _ in budgets]

    # Use the curvature the pencil actually turns on.  ``profiled_trace`` is
    # assembled from nonnegative centered residual energies in
    # ``spline_cat_moments``; neither a dense V_eff nor the catastrophic
    # ``tr(V) - tr(C' M^+ C)`` difference is formed.  This direct scale fixes
    # the bracket itself, so the reachable-rank contract proposed in issue
    # #204 is unnecessary: there are no speculative endpoint evaluations to
    # classify, and evaluation counts, duplicate caching and genuine
    # unreachable clamps retain their existing contract.
    #
    # The 1e+-10 edges are the dense ladder's, kept identical so a pair the
    # two paths can both score gets the same lambda0.
    tr_S = float(np.trace(p.S_a)) * p.dims[0]
    scale = max(p.profiled_trace, 1e-300) / max(tr_S, 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale

    evaluated_lo = evaluate(lo)
    evaluated_hi = evaluate(hi)
    if evaluated_lo is None or evaluated_hi is None:
        return None
    stat_lo, edf_lo = evaluated_lo
    stat_hi, edf_hi = evaluated_hi
    # More penalty cannot add effective dimensions.  The ladder itself only
    # resolves EDF to ``_EDF_TOL``, so sub-tolerance ordering noise is not
    # evidence of a different numerical branch.
    if edf_hi > edf_lo + max(_EDF_TOL, _edf_roundoff(edf_lo, edf_hi)):
        return None

    # DISTINCT search targets, not rungs: the ladder's budgets are permitted to
    # repeat, every copy of one bisects to the same lambda, and charging each
    # copy separately would let repeating a budget decide whether the pair is
    # screenable at all.  The same set drives the cache below, so a repeat
    # costs nothing rather than a second bisection.
    searchable = {float(b) for b in budgets if edf_hi < float(b) < edf_lo}
    if max_evaluations is not None and 2 + _MAX_STEPS_PER_RUNG * len(searchable) > max_evaluations:
        return None

    solved: dict[float, ScreenedPair | None] = {}
    out: list[ScreenedPair] = []
    for budget in budgets:
        edf0 = float(budget)
        if edf0 not in searchable:
            lam = lo if edf0 >= edf_lo else hi
            stat, achieved = (stat_lo, edf_lo) if lam == lo else (stat_hi, edf_hi)
            out.append(ScreenedPair(statistic=stat, edf0=achieved, lambda0=float(lam)))
            continue
        if edf0 not in solved:
            a, b = lo, hi
            edf_a, edf_b = edf_lo, edf_hi
            lam, stat, achieved = hi, stat_hi, edf_hi
            refused = False
            for _ in range(_MAX_BISECT):
                if b <= a * (1.0 + 1e-12):
                    break
                lam = float(np.sqrt(a * b))
                evaluated = evaluate(lam)
                if evaluated is None:
                    refused = True
                    break
                stat, achieved = evaluated
                # Preserve the same monotone certificate inside the shrinking
                # bracket; a finite in-range EDF can still be on a broken
                # numerical branch.  As at the endpoints, noise below the
                # ladder's own target tolerance is not a refusal.
                ordering_tolerance = max(_EDF_TOL, _edf_roundoff(edf_a, achieved, edf_b))
                if achieved > edf_a + ordering_tolerance or achieved < edf_b - ordering_tolerance:
                    refused = True
                    break
                achieved = min(max(achieved, edf_b), edf_a)
                if abs(achieved - edf0) <= _EDF_TOL:
                    break
                if achieved > edf0:
                    a = lam
                    edf_a = achieved
                else:
                    b = lam
                    edf_b = achieved
            # Width exhaustion is not convergence: a numerically discontinuous
            # EDF curve can keep the target bracketed while never attaining it.
            # Do not cache or publish a plausible nearest endpoint as the rung.
            if refused or abs(achieved - edf0) > _EDF_TOL:
                solved[edf0] = None
            else:
                solved[edf0] = ScreenedPair(
                    statistic=stat,
                    edf0=achieved,
                    lambda0=float(lam),
                )
        result = solved[edf0]
        if result is not None:
            out.append(result)
    # A failure while searching one target says nothing about already certified
    # edge clamps or independent targets.  Preserve those rungs; if none
    # survive, retain the existing pair-refusal signal so a speculative
    # structured route can hand the dense path back.
    return out or None
