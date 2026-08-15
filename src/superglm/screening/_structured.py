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
    edf(lambda) = tr(A(lambda)^-1 V_eff)

**THE SECOND IS A SUM, NOT A DIFFERENCE, AND THAT IS THE WHOLE POINT.**
Diagonalize the pencil and ``tr(A^+ V_eff)`` is ``sum_j a_j / (a_j + lambda
s_j)`` — one Tikhonov filter factor per direction, every term in ``[0, 1]``,
bounded and cancellation-free.  That is the standard form for the effective
degrees of freedom of a regularized fit: Wood, *JRSS-B* 73(1):3-36 (2011),
§3.6 and Appendix H define ``edf = tr(F)`` with ``F = (X'WX + S)^-1 X'WX`` and
evaluate it as ``||K||_F^2``; Golub, Heath & Wahba, *Technometrics*
21:215-223 (1979); Eldén, *BIT* 17:134-145 (1977) and 22:487-502 (1982) for
the standard-form transformation, and *BIT* 24:467-472 (1984) for computing
the trace in O(n) by bidiagonalization; Hansen, *Regularization Tools* manual
v4.1 (March 2008, Technical University of Denmark), §2.4 — Eq. (2.19) for the
filter-factor expansion, (2.30) for the unregularized ``null(L)`` component
and (2.63) for the trace itself, ``trace(I_m - A A^I) = m - (n - p) -
sum_i f_i``, with the general-form ``f_i = gamma_i^2 / (gamma_i^2 +
lambda^2)`` stated as prose in the same section; journal versions *Numer.
Algorithms* 6:1-35 (1994) and 46:189-194 (2007).  (Hansen, Nagy & O'Leary,
*Deblurring Images* (SIAM 2006) is cited for spectral filtering in the
STANDARD form only — its ch. 6 has no general form; ch. 7 §7.3, Eq. (7.9) is
the general one and uses no GSVD.)

**WE ARE OUTSIDE EVERY ONE OF THOSE PAPERS' STATED ASSUMPTIONS, AND THIS SAYS
SO WHERE THE BOUND IS CLAIMED RATHER THAN IN AN ASIDE.**  Wood 2011 assumes
Fisher weights, which make both ``X'WX + S`` and ``X'WX`` positive definite
and the edf well defined; Hansen assumes ``rank(L) = p`` and ``N(A) & N(L) =
{0}``.  Here ``S`` is semidefinite by construction and a probe on the starved
family found 6 of 77 directions in ``null(V_eff) & null(S)``.  A pencil with a
common null space is SINGULAR, not merely ill conditioned (Cao, *LAA*
92:187-196, 1987), and LAPACK says the same about what it can return: the
trivial eigenvalues, "those corresponding to the leading ``n - r`` columns of
``X``, which span the common null space of ``A^T A`` and ``B^T B``", are NOT
WELL DEFINED (*LAPACK Users' Guide*, 3rd ed., SIAM 1999, §2.3.5).  Fix &
Heiberger, *SIAM J. Numer. Anal.* 9:78-88 (1972), doi:10.1137/0709009, refuse
such a pencil outright.  Contribution ZERO on the common null space is the
defensible convention, it is what ``tr(A^+ V_eff)`` already implements, and it
is a CHOICE about an undefined quantity rather than an approximation to a
defined one.

**THE ONE SUBTRACTION LEFT WAS THE WHOLE DEFECT, AND ITS CAUSE WAS ONE
MATRIX.**  This used to be evaluated as ``edf = local - border``, a difference
of two nonnegative traces, because ``V_eff`` was carried as ``blockdiag(D_q) -
D' Omega D`` with ``Omega = Sigma_M^-1[1:, 1:]`` — a pseudo-inverse of a GRAM.
That carries ``kappa(Abar)^2`` where the row-space factor carries
``kappa(Abar)``, measured at 1e+12 against 1e+6 on the starved family, and
squaring the condition number is the entire story.  On an 8-level ``ps``-like
pair with 3 rows per level against an 11-column margin, at ``lambda = 1e-4``,
that form published **4.950 against a certified 6.000** — a ``-1.05`` df error
that sits INSIDE the range guard and was not refused.  Both halves are
``tr(A^-1 . PSD)`` and hence exactly nonnegative, so ``local = edf + border``
is an identity and the cancellation ratio is exactly ``border / edf``; on that
point it is 5.7.

``V_eff`` is dense, so evaluating the trace against it needs its structure
rather than the matrix — but it is the Gram of a RESIDUALIZED design, not a
difference.  With ``Zc_q`` level ``q``'s centered weighted spline rows,
``J = blockdiag(Zc_q)`` and ``P`` the projector onto the residualized overlap
span ``Abar``,

    V_eff = J' (I - P) J = G' G,      G = (I - P) J,

which is Frisch-Waugh-Lovell absorption.  With ``Abar = Q R`` thin and
``Psi_q = Q_q' Zc_q`` bounded by the data scale,

    V_eff = blockdiag(D_q) - Psi' Psi,   A(lambda) = blockdiag(N_q) - Psi' Psi.

Everything below is built from ``[R_q ; sqrt(lambda) rootS]``-style QRs and
never from a sum or a difference of Grams: Gill, Golub, Murray & Saunders,
*Math. Comp.* 28(126):505-535 (1974) §5.2 is the authority — computing a Schur
complement via a triangular solve is "numerically less satisfactory than
computing R using orthogonal matrices" — and (p. 516) downdating is stable
only when the result's smallest eigenvalue is large relative to the norm of
the removed term, which is this module's documented failure mode rather than
its exception.  The block-angular structure is standard: Golub & Plemmons,
*LAA* 34:3-28 (1980); Golub, Manneback & Toint, *SIAM J. Sci. Stat. Comput.*
7(3):799-816 (1986); Scott & Tuma, *Numer. Algorithms* 79(4):1147-1168 (2018).
Golub & Van Loan, *Matrix Computations* 4th ed. is the reference for block
Cholesky — with the border eliminated first the trailing factor IS the
Cholesky factor of the Schur complement — and for the stacked least-squares
form.  Kaufman & Rosset, *Biometrika* 101(4):771-784 (2014) stays where it
sits, for monotonicity under a DEFINITE pencil.

**A NEGATIVE RESULT, RECORDED BECAUSE IT IS ONE.**  A sweep for
``tr(A^-1 V)`` where ``V`` is an explicitly formed Schur complement found
nothing.  The literature's move is to never form it; there is no published
stabilization of the difference because the difference is not what anyone
computes.  Holding a factor of ``V_eff`` directly does not work here either
and that was verified rather than assumed: ``V_eff`` is block-diagonal minus
rank ``k_a``, hence dense, so any factor of it is dense — O(L^2) memory and
O(L^3) time, which is the cost this module exists to avoid.

**WHAT THE MOMENTS COST, AND WHY THE edf HALF NO LONGER READS THEM.**  The
``edf`` half is computed from the row-space factors ``p.R`` and the compacted
overlap rows, which come from the DESIGN.  ``V``, ``c``, ``m`` and ``border``
are read only by the statistic.  That is not a tidiness argument: forming the
pair's moments in float64 destroys the quantity before any screening rule
runs.  Measured on the starved family, three defensible exact-arithmetic
policies for ``M^+`` applied to the SAME delivered float64 moments -- invert
everything above 1e-38, drop the negative directions, cut at 1e-14 -- return
10.0, 10.0 and 11.0 on one 8-level pair and 8.0, 8.0 and 9.0 on another, where
the exact DESIGN gives 6.000000 under every one of those cuts and under 1e-30,
1e-25 and 1e-20 as well.  So the moment-formed answer is ambiguous by up to a
degree of freedom and wrong by four, and the design-formed one is not
ambiguous at all.  On a healthy pair all three policies agree to six digits
and the gap to the design is 5.4e-03.  **The factored route does not RESOLVE
that ambiguity, it avoids INCURRING it**, which is the same defect as issue
#268 and is why this route reaches 1.1e-06 median where a moment-based one
cannot.

**WHAT IT IS WORTH, MEASURED AGAINST AN EXACT-DESIGN ORACLE.**  Seven
geometries -- one healthy, three starved (3 rows per level against 9 to 12
columns), one with more columns than rows per level, one thin-level and one
with a level at 1e-8 weight -- at eight lambdas each, against mpmath at 50
digits on each pair's exact float64 design.  Of the 56 points, the old form
published 54 and refused 2; this publishes all 56.  Median error falls
**3.06e-03 -> 1.10e-06**, worst **22.01 -> 1.00 df**, and points worse than
0.1 df fall 19 -> 13.  Every one of the 13 is on a geometry where the oracle
itself is policy-dependent -- ``null(V_eff) & null(S)`` is nonempty -- and the
error is exactly ``+1.00`` df, the irreducible singular-pencil offset below.
On the two starved pairs where the pencil is NOT singular the error is at
round-off: 6.000000 against 6.000000 to 5e-15 and 3e-13 at every lambda in the
bracket, where the old form ranged from -1.05 to +22.2 df and refused twice.

**THE +1.00 df ON SINGULAR-PENCIL GEOMETRIES IS IRREDUCIBLE IN FLOAT64.**  It
is shared by this form, by the form it replaces, and by the textbook O(L^3)
Wood stacked-QR method; it is the same statement as the LAPACK sentence above.
Do not read the range guard as implying exactness.

**WHERE THIS IS WORSE, STATED RATHER THAN AVERAGED AWAY.**  On the
near-absorbed 1-column pair, whose ``V_eff`` is 1e-12 of its own level block,
the relative error is 4e-16 at ten of eleven lambdas across the bracket
against the old form's 1e-10 to 1.3e-04 -- but at the two lambdas where the
0.5 rung sits it is 2.4e-04, against 5e-13 for the old form, and that is
enough that a bisection targeting ``_EDF_TOL = 1e-6`` no longer converges and
the pair is refused where it used to publish.  The cause is located: ``H``'s
small eigenvalue is 2e-12 there and ``1/h`` multiplies an ``eps``-level
residue.  Writing the level's contribution in the shifted coordinates
``[K_q | K_q Y_q - Phi_q]``, and separately equilibrating ``W_q`` before its
PSD factorization, each fix that rung and each break three others; both were
measured and neither is adopted.

**WHAT THE RANGE GUARD IS FOR NOW.**  Every term of the sum is a squared norm,
so ``edf >= 0`` holds by construction rather than by luck, and on the suite's
own starved ``bs(10)`` pair -- where the old form left ``[0, L k_a]`` at 101
of 200 lambdas and the whole ladder returned ``None`` -- this leaves it at
NONE of them.  The upper end is still a guard: ``A^+`` rests on a deflation
decision, and a deflation that misfires inverts a direction the pencil does
not resolve.  Measured once, at 2237 against a ceiling of 77.

**AND "BY CONSTRUCTION" IS ONLY WORTH SOMETHING IF THE CONSTRUCTION IS
CERTIFIED.**  Two clips are what make it true, and each is now bounded rather
than trusted, because a clip that silences roundoff silences a breakdown just
as well and leaves a plausible number inside ``[0, ceiling]`` either way.
:func:`_psd_factor`'s ``max(w, 0)`` is held to :func:`_psd_clip_allowance` --
``eigh``'s documented ``n eps ||M||`` plus the measured closure residual over
the deflation, the second dominating three orders on the starved family --
and what it removes beyond that is carried into DEGREES OF FREEDOM by
``||F_q||_F^2`` and refused against the same dust allowance the range guard
takes.  :func:`_penalty_root`'s projection is held to ``n^2 eps`` of the
penalty's trace, which is what separates a penalty the two halves can share
from one where the statistic and the edf would be scoring different pencils.
Neither fires on data: over the five screening suites, 86 pair geometries
and 767 evaluations, there are ZERO refusals from either, and the worst
approach is 0.25 of the first allowance and 0.1875 of the second.

**WHAT IS NOT A TRADE: REPRODUCIBILITY.**  Permuting the level coordinates
leaves ``edf`` unchanged in exact arithmetic and changes only the order of
every reduction — which is what a different BLAS kernel or thread count does.
Over 40 such relabelings of ``_thin_level_pair`` the OLD form's HIGH-edge value
spreads 1.54e-04, 6.30e-04 and 3.07e-03 df across the three weights; the last
is LARGER than the 3e-3 parity bound the suite asserts there, so that assertion
was sitting on its own noise floor and would have gone red on a machine that
reduced in a different order.  This form spreads 1.44e-12, 1.29e-12 and
8.24e-13.  Its low-edge spread moves the other way, 0.0 -> 2.8e-09, which is
3600x inside the bound asserted there.

**WHAT IT IS NOT WORTH: MONOTONICITY.**  ``edf`` is decreasing in lambda for a
DEFINITE pencil (Kaufman & Rosset, *Biometrika* 101:771-784, 2014); this one is
not always definite, and evaluating the right identity does not make it so.
Walking 81 lambdas across the bracket of issue #263's own geometry — ``ps(8)``,
20 levels, 30 rows each, four squeezed into a narrow band — at band width 1e-3
the old form increases at 12 of the 80 steps with a worst increase of +6.17 df
and the new one at 4 of 80 with +0.75; at width 1e-4, 19 of 80 with +70.78
against 16 of 80 with +37.89.  On the vanishing-mass pairs both sit at 1 to 2
of 80 and every increase is +0.0000.  On a nullity-2 ``m = 3`` family the old
form increases at 30 of 76 with +7.56 and refuses 2 points; the new one at 35
of 80 with +1.36 and refuses none.  So the worst increase falls 1.9x to 8.3x
and the refusals go, but the COUNT does not improve.  **#263 is reduced and NOT
closed**, and a bisection over this curve can still land on a step it cannot
attain.  Monotonicity would follow from delivering NONNEGATIVE per-direction
shares, which this route does not.

**WHERE THE ARITHMETIC IS BEYOND EVERY FORM.**  Give a pair 8 levels with 3
rows in each against an eleven-column margin and the overlap arrow's border
Schur complement is nearly singular; its inverse then runs to 1e+10-1e+12 and
every organization of this quantity loses eleven digits.  The three traces the
old form summed measure 6.7e+10, -1.3e+11 and 6.7e+10 against an answer of 8.1.
Both forms are wrong there — the old by up to 53.08 df, the new by up to
22.01 — and what the new one buys is that its worst cases leave ``[0, L k_a]``
and are refused instead of published.  That is a defect in the FACTORIZATION's
border cut rather than in the edf form, and it is not fixed here.

**HOW FAR BEYOND: ONE ULP OF LAMBDA USED TO DECIDE IT.**  On a 7-level
``bs(10)`` pair with 3 rows per level the OLD difference form measured
``local`` and ``border`` both at 9.654e+07 at the ladder's high edge, with a
difference of ``-13.61`` — outside ``[0, 78]``, so refused — while moving
lambda by a SINGLE ulp, 594.8079378729931 against ...32, which is just the
difference between ``1e10 * a / b`` and ``1e10 * (a / b)``, returned
``edf = 0.8229`` instead.  ``edf`` was not a continuous function of lambda at
float64 resolution on that family.  **THE SUM DOES NOT DO THIS AND THE SAME
FIXTURE IS THE EVIDENCE**: the two sides of that ulp now read 0.99999632699
and 0.99999632618, agreeing to 8.1e-10, the sum leaves ``[0, L k_a]`` at NONE
of 200 log-spaced lambdas where the difference left it at 101, and the ladder
scores the pair at 1.0000 on every budget instead of handing it back.
``tests/test_structured_screening.py`` scans the whole bracket rather than
asserting at either lambda, because what is portable here is the width of the
region and not a value at one point.

**WHAT A DROPPED DIRECTION COSTS NOW.**  The pair arrow's relative cut still
drops directions, and where one carries a filter factor near 1 the STATISTIC
loses it.  ``_mixed_rank_cells`` in the suite is built to sit exactly there —
its only ``V_eff`` direction is ``1e-16`` RELATIVE to its own level block,
under ``_solve_floor``'s ``3 eps``.  **The edf half no longer pays that**, and
the reason is the whole point of working on the factor: ``N_q`` is ``T_q' T_q``
from a QR of ``[R_q ; sqrt(lambda) rootS]`` and ``R_q``'s smallest direction
sits at ``d``, not at ``d^2``, so the cut is applied to a square root.  Against
a 50-digit oracle of 1.000000 / 0.500000 / 0.000000 across the bracket this
reads 0.9999999999 / 0.5000000000 / 5.7e-10, and the ladder attains the 0.5
rung the form it replaces answered with a single rung at 0.0.

The dense path carries its own version of the same failure and it is not
touched here: on one of four wide pairs of the shape this kernel exists for
it reports 1.03 df BELOW a high-precision oracle at the high edge, where
``numpy.linalg.pinv``'s inherited default ``rcond`` drops a direction the
penalty leaves free.

**WHAT THIS SETTLES AND WHAT IT DOES NOT.**  #249, #258, #265 and #271 are all
statements about ``block_ranks`` -- an unconditional contrast term, a
disagreement with what the inverse resolves, a global-versus-per-block
reference scale, and that disagreement turning out to be two-signed.  There is
no rank here to make any of those statements about, so they go with it; #271
in particular asked whether the invariant was one-signed, and an invariant
between a count and an inverse cannot be violated in either direction once
there is no count.  #263 is REDUCED, NOT closed, with the numbers above.
#262's three geometries were re-measured on both forms: at ``bs(6)``/1e-5 the
old form refuses at the high edge and the whole ladder returns ``None``, while
this one returns 3.0003 and one rung -- that refusal is lifted; at
``cr(4)``/4.64e-5 both publish one rung, 2.5336 against 3.0038; but at
``bs(8)``/2.15e-4 the old form publishes three rungs (3.0, 8.0, 16.0) and this
one publishes one, because its lower high edge (3.0001 against 3.0000, with a
low edge of 28.24 against 35.22) leaves two budgets bisecting over the
non-monotone curve #263 describes, and they fail.  So #262 moves in both
directions and stays open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.screening._arrow import factor_arrow
from superglm.screening._score_stat import ScreenedPair

_EDF_TOL = 1e-6
_EDF_ROUNDOFF_FACTOR = 64.0
_PSD_CLIP_FACTOR = 8.0
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
    """A filter-factor sum came back outside ``[0, k]``, so it is not one."""


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
    # THE ROW-SPACE FACTORS, NOT THE MOMENTS.  ``R[q]' R[q] == D_q`` exactly
    # in the sense that both are Grams of the same centered weighted rows, but
    # ``R`` is that geometry's FACTOR: it carries kappa where ``D_q`` carries
    # kappa^2.  The ladder's edf is computed from these and from
    # ``overlap_rows`` alone, never from ``V``/``c``/``m``; see the module
    # docstring's "WHAT THE MOMENTS COST".
    R: NDArray  # (L, k_a, k_a)     centered row factor of each emitted level
    overlap_rows: NDArray  # (., 1 + k_a)  compacted rows of the unemitted levels

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


def _reduce_row_factors(base: NDArray, blocks: NDArray) -> NDArray:
    """Compact ``base`` stacked under a batch of per-level row factors.

    The reduction itself is forced: the levels' factors have to meet in one
    ``w``-column factor and no Gram may be formed on the way, so what is left
    to choose is the SHAPE OF THE OPERANDS, and stacking the whole batch into
    one ``(n * h, w)`` matrix chooses badly.  Its row count is the level
    count, which puts ``L`` inside a matrix dimension: the work stays linear,
    but the temporary is ``L``-sized and the factorization is one long
    sequential Householder chain.  ``tests/test_screening_cost_scaling.py``
    asserts the opposite property -- that ``L`` appears only as a batch count
    -- and it is right to.

    This is instead the binary reduction tree of Demmel, Grigori, Hoemmen &
    Langou, *SIAM J. Sci. Comput.* 34(1):A206-A239 (2012) (communication-
    avoiding TSQR, their §2), on the ``R`` factors alone: pair the blocks,
    factor every pair in ONE batched call, and repeat.  Every operand is
    ``(2w, w)`` however many levels there are, the depth is ``log2(n)`` calls
    instead of one, and the total work is the same ``n`` small factorizations.
    TSQR is unconditionally backward stable -- each level of the tree is an
    orthogonal transformation of a matrix whose norm it preserves -- so this
    is a reorganization of the same reduction and not a relaxation of it.

    Blocks shorter than ``w`` are zero-padded rather than factored, since
    their ``R`` is themselves; blocks taller are reduced once, batched, before
    entering the tree.
    """
    base = np.asarray(base, dtype=np.float64)
    blocks = np.asarray(blocks, dtype=np.float64)
    n, h, w = blocks.shape
    if n == 0 or w == 0:
        return base
    if h > w:
        blocks = np.linalg.qr(blocks, mode="r")
        h = w
    if h < w:
        padded = np.zeros((n, w, w), dtype=np.float64)
        padded[:, :h, :] = blocks
        blocks = padded
    while blocks.shape[0] > 1:
        pairs = blocks.shape[0] // 2
        merged = np.linalg.qr(blocks[: 2 * pairs].reshape(pairs, 2 * w, w), mode="r")
        odd = blocks[2 * pairs :]
        blocks = np.concatenate((merged, odd), axis=0) if odd.size else merged
    return _combine_row_factors(base, blocks[0])


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
    *,
    keep_factors: NDArray | None = None,
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

    ``keep_factors`` is an ``(len(level_rows), k, k)`` output array that
    receives ``R_q`` for each emitted level as the forward pass forms it.  The
    ladder needs exactly those factors and they are otherwise discarded here,
    so keeping them costs one store rather than a third chunked QR pass.
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
    emitted_index = np.full(n_levels, -1, dtype=np.intp)
    emitted_index[level_rows] = np.arange(level_rows.size, dtype=np.intp)
    prefix = np.zeros((k_a, k_a), dtype=np.float64)
    trace = 0.0
    correction = 0.0

    for start in range(0, n_levels, chunk):
        stop = min(n_levels, start + chunk)
        factors = _centered_level_factors(B, W_cell[:, start:stop])
        for level in range(start, stop):
            local = factors[level - start]
            if emitted[level]:
                if keep_factors is not None:
                    keep_factors[emitted_index[level]] = local
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


def _unemitted_overlap_rows(
    B: NDArray,
    W_cell: NDArray,
    level_rows: NDArray,
) -> NDArray:
    """Compact the overlap rows of the levels the contrast menu does not emit.

    The menu is treatment coded, so the levels it emits carry their own
    indicator and the base levels do not.  Residualizing the overlap span on
    those indicators therefore centers an emitted level's rows inside the
    level -- which is exactly :func:`_centered_level_factors`, and which sends
    its intercept column to zero -- and leaves a base level's rows
    ``[sqrt(w) | sqrt(w) B]`` untouched.  Only the COMBINED factor of the base
    rows is ever needed, so they are compacted chunk by chunk and never held.
    """
    B = np.asarray(B, dtype=np.float64)
    W_cell = np.asarray(W_cell, dtype=np.float64)
    n_rows, k_a = B.shape
    n_levels = W_cell.shape[1]
    width = 1 + k_a
    unemitted = np.ones(n_levels, dtype=bool)
    unemitted[np.asarray(level_rows, dtype=np.intp)] = False
    columns = np.flatnonzero(unemitted)
    factor = np.zeros((0, width), dtype=np.float64)
    if columns.size == 0 or n_rows == 0:
        return factor
    chunk = _trace_chunk_width(n_rows, width, columns.size)
    for start in range(0, columns.size, chunk):
        block = columns[start : start + chunk]
        root = np.sqrt(W_cell[:, block]).T[:, :, None]  # (levels, n_rows, 1)
        rows = np.concatenate((root, root * B[None, :, :]), axis=2)
        factor = _reduce_row_factors(factor, np.linalg.qr(rows, mode="r"))
    return factor


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

    # THE TRACE PASS RUNS FIRST, AND THE ORDER IS THE MEMORY GATE.
    # ``screening_ops._within_structured_budget`` admits a pair on TWO
    # coexisting level-sized stacks, which is what this phase held before the
    # row factors ``R`` were retained: the curvature ``V`` and the trace's
    # additive suffix.  Keeping ``V``'s construction where it was would make
    # that three -- ``V``, ``R`` and the suffix -- and admit the same maximum
    # pair against a budget that never counted the third.  ``V`` is not read
    # until the pair is assembled, so building it AFTER the suffix has been
    # released keeps the phase at two stacks and the gate honest, at no cost:
    # the two passes read ``W_cell`` and ``B_a`` and nothing of each other.
    R = np.zeros((level_rows.size, k_a, k_a), dtype=np.float64)
    profiled_trace = _profiled_curvature_trace(B_a, W_cell, level_rows, keep_factors=R)
    overlap_rows = _unemitted_overlap_rows(B_a, W_cell, level_rows)

    # One GEMM for every level's k_a x k_a curvature: the outer products of
    # the spline menu are level-independent, so they are formed once and
    # contracted against each level's weights.  Its n_a*k_a^2 construction
    # scratch is released before anything else level-sized is formed.
    AA = (B_a[:, :, None] * B_a[:, None, :]).reshape(n_a, k_a * k_a)
    V = (Wq.T @ AA).reshape(-1, k_a, k_a)
    del AA

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
        R=R,
        overlap_rows=overlap_rows,
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


def _penalty_root(S_a: NDArray) -> NDArray:
    """``rootS`` with ``rootS' rootS`` the NEAREST PSD matrix to ``S_a``.

    ``edf`` is a sum of filter factors ``a_j / (a_j + lambda s_j)``, and a
    NEGATIVE ``s_j`` puts that term outside ``[0, 1]``: there is no bound to
    keep and no nonnegative decomposition to have.  Assembled penalties here
    are not all inside the cone.  A difference penalty IS exactly PSD as
    stored -- exact rational LDL certifies exactly ``m`` zero pivots, and the
    ``-1e-15`` an eigensolver reports on it is the eigensolver's own backward
    error -- but the integrated-derivative penalty ``bs`` and ``cr`` margins
    carry is genuinely not, and ``fl(lambda * S_a)`` leaves the cone even for
    the exactly-PSD one.

    **A CUT AT THE MODULE'S USUAL ``k eps`` RELATIVE FLOOR IS WRONG HERE, AND
    THE REASON IS MEASURED.**  On the suite's vanishing-mass pair the
    penalty's smallest eigenvalue is ``1.374e-16`` of its largest, which
    ``lambda_hi = 1e10 * scale`` amplifies into a real penalty of
    ``1.86e-08 * tr(V_eff)`` that reaches three levels' free directions.
    Dropping it reports 19 free directions where an independent closed form
    counts 16.  The residue is data.

    **AND ITS SIGN IS NOT, SO NEITHER ``max(w, 0)`` NOR A DROP MAY DECIDE
    THREE DEGREES OF FREEDOM.**  Assembly round-off puts that eigenvalue on
    either side of zero depending on the data -- the same fixture measures
    ``+2.11e-15`` at one seed and ``-5.03e-16`` at another, and the same
    fixture at the same seed measures either sign on different machines.
    ``max(w, 0)`` is the Euclidean projection onto the PSD cone (Higham,
    *Linear Algebra Appl.* 103:103-118, 1988) and is the right thing to do to
    an eigenvalue that is RESOLVED; applied to one that is not, it turns a
    coin flip into a 3 df move in a published ``edf0``, and CI caught exactly
    that -- 19.000000 where this machine reads 16.000374.

    So an eigenvalue inside ``eigh``'s own error bar, ``n eps ||S||_2``
    (*LAPACK Users' Guide*, 3rd ed., SIAM 1999, sec. 4.7), is taken at its
    MAGNITUDE, which is the only sign-independent choice that keeps it.
    Checked against 40-digit mpmath on both signs of the residue: the
    magnitude gives 15.999993 and 16.000012, and clamping the negative case
    up to zero gives back the full ``k_b`` -- wrong by three degrees of
    freedom.  Inside the bar every PSD matrix within ``n eps ||S||_2`` of
    ``S_a`` is equally admissible, so what is chosen there cannot be settled
    by nearness to ``S_a``; it is settled by requiring the answer to be a
    function of the data rather than of the rounding.

    Outside the bar a negative eigenvalue is real, no magnitude is taken and
    the direction is dropped -- and :func:`_profile` then refuses the pair,
    because the statistic is still scoring ``S_a`` raw.
    """
    n = S_a.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    w, Q = np.linalg.eigh(0.5 * (S_a + S_a.T))
    unresolved = float(n) * np.finfo(np.float64).eps * float(np.max(np.abs(w), initial=0.0))
    lifted = np.where(w >= -unresolved, np.abs(w), 0.0)
    keep = lifted > 0.0
    return (Q[:, keep] * np.sqrt(lifted[keep])).T


@dataclass(frozen=True)
class _PairGeometry:
    """Everything the ladder needs that does NOT depend on ``lambda``.

    ``V_eff`` is dense, but it is the Gram of a RESIDUALIZED design and never
    has to be written as a difference of two Grams.  Write level ``q``'s
    centered weighted spline rows as ``Zc_q`` (``R_q' R_q = Zc_q' Zc_q``), let
    ``J = blockdiag(Zc_q)`` over emitted levels and let ``P`` project onto the
    span of the residualized overlap rows ``Abar``.  Then

        V_eff = J' (I - P) J = G' G,       G = (I - P) J,

    which is Frisch-Waugh-Lovell absorption: the tensor block residualized on
    the overlap span.  With ``Abar = Q R`` thin and ``Psi_q = Q_q' Zc_q``,

        V_eff = blockdiag(D_q) - Psi' Psi,      A(lambda) = blockdiag(N_q) - Psi' Psi

    with ``N_q = D_q + lambda S_a``.  ``Psi`` is bounded by the data scale
    because ``Q`` has orthonormal columns.

    **THAT IS THE WHOLE CHANGE.**  The form this replaces coupled the levels
    through ``Omega = Sigma_M^-1[1:, 1:]``, a pseudo-inverse of a GRAM: it
    carries ``kappa(Abar)^2`` where this carries ``kappa(Abar)``, measured at
    1e+12 against 1e+6 on the starved family.  Squaring the condition number
    was the entire mechanism behind ``edf = local - border``; nothing else
    about that difference had to be fixed once the squared inverse was gone.

    ``coupling`` is the ``(k_a, r)`` map from a level's row factor into the
    shared overlap coordinates -- ``Psi_q = (R_q coupling)' R_q`` -- and
    ``base_gram`` is a factor of ``sum_b Q_b' Q_b`` over the levels the menu
    does not emit.  Nothing here is ``L``-sized except ``p.R``, which the
    trace pass already formed.

    ``ceiling`` is ``L * k_a``: the number of Tikhonov filter factors summed,
    each in ``[0, 1]``, hence a bound on ``edf`` that takes no rank decision.
    **A TIGHTER BOUND WAS TRIED AND IS NOT SAFE.**  ``rank(V_eff)`` is the
    mathematically right ceiling and is 10 to 65 df tighter here, but every way
    of computing it in O(L) is a numerical rank, and a numerical rank that
    comes out LOW refuses true values: counted by Guttman additivity off the
    unpenalized arrow -- ``rank(K(0)) - rank(M)`` -- it reads 12 on a 7-level
    ``cr(5)`` pair with 3 rows per level whose certified ``edf`` at the
    ladder's low edge is 17.618, and 36 against a certified rank of 45 on a
    5-level ``ps(8)`` pair.  Evaluating this module's own ``edf`` at
    ``lambda = 0`` is no better: 30.014 on a pair whose low-edge ``edf`` is
    30.072, so it refuses that pair's own bracket.

    **A THIRD COUNTER WAS MEASURED AGAINST THAT SAME COUNTEREXAMPLE AND ALSO
    FAILS IT.**  The factors this route now carries give a structurally exact
    O(L) bound, ``sum_q rank(R_q) + rank(overlap_rows) - r``, which is a rank
    count on FACTORS rather than on Grams and is 4x to 12x tighter -- it equals
    the true ``rank(V_eff)`` on 6 of 7 synthetic geometries.  On the ``cr(5)``
    pair above it reads 12, exactly as Guttman additivity does, against a
    low-edge ``edf`` of 17.618: the pair's own bracket would be refused.  The
    counterexample kills the tight ceiling on the factor route too, so the
    loose bound that cannot lie is kept.
    """

    U_eff: NDArray  # (L, k_a)      the profiled score
    coupling: NDArray  # (k_a, r)   R_q -> the shared overlap coordinates
    base_gram: NDArray  # (., r)    factor of sum_b Q_b' Q_b, unemitted levels
    root_penalty: NDArray  # (., k_a)  rootS' rootS = the PSD part of S_a
    penalty_clip: float  # relative mass the PSD projection removed from S_a
    orthonormality: float  # ||sum_q Q_q' Q_q - I_r||_2, the deflation's floor
    overlap_rank: int  # r
    ceiling: float  # L * k_a


def _pair_arrow(p: SplineCatPair, lam: float, penalty: NDArray | None = None):
    """``K(lambda)`` in arrow form: one ``(k_a + 1)`` block per level.

    Level q's block holds its tensor coefficients beside its own contrast;
    the border holds the intercept and the spline main, the only two things
    every level shares.  ``C``'s spline-main rows are literally ``V``'s
    diagonal blocks — both are ``sum_i w_i A_i A_i'`` restricted to the level
    — so they are taken from the same array rather than reassembled.

    **``penalty`` IS HOW THE TWO HALVES ARE MADE TO SCORE ONE PENCIL.**  The
    ladder chooses lambda from ``edf``, which is evaluated against
    ``rootS' rootS``, and then publishes a statistic from this factorization.
    Adding ``p.S_a`` raw here would make those two different matrices whenever
    the projection did anything -- and it is at ``lambda_hi = 1e10 * scale``
    that the difference is largest, which is exactly a rung the ladder
    publishes.  Measured before this argument was threaded through: on a
    nullity-two pair the high-edge statistic reads 2.999250 against 2.821425,
    a 5.9e-02 relative gap, on a penalty perturbation of ``2.2e-14``.  The
    caller passes the same projection ``edf`` uses; ``None`` keeps ``S_a`` for
    the callers that have no geometry to hand.
    """
    L, k_a = p.dims
    g, r = k_a + 1, 1 + k_a
    G = _unpenalized_blocks(p)
    G[:, :k_a, :k_a] += lam * (p.S_a if penalty is None else penalty)
    E = np.empty((L, r, g), dtype=np.float64)
    E[:, 0, :k_a] = p.c
    E[:, 0, k_a] = p.m
    E[:, 1:, :k_a] = p.V
    E[:, 1:, k_a] = p.c
    return factor_arrow(G, E, p.border)


def _profile(p: SplineCatPair) -> _PairGeometry:
    """The whole lambda-independent half of the work.

    ``U_eff = U - C' M^-1 u_m``.  Column ``(p, q)`` of ``C`` is nonzero in
    exactly three places — the intercept row, the spline-main rows, and level
    q's own contrast row — so the contraction never touches a level other
    than its own.  ``M`` depends on no lambda, so this is computed once and
    every rung of the ladder reuses it.  That is the STATISTIC's half and it
    still reads the moments.

    The ``edf`` half reads none of them.  One orthonormal basis for the
    residualized overlap span is built by TSQR over the per-level row factors
    the trace pass already formed -- ``[0 | R_q]`` for an emitted level,
    ``p.overlap_rows`` for the rest -- and rank-revealed by an SVD of that
    single ``(1 + k_a)``-column factor, at the LAPACK / ``matrix_rank`` cut
    ``max(shape) * eps * s_max``.  The cut lands on the FACTOR's singular
    values, so it is the square root of what the same decision costs on a
    Gram; that is the whole reason this route can see directions
    ``Omega = (sum_q D_q)^+`` cannot.  Wood, *JRSS-B* 73(1):3-36 (2011)
    §3.3.1 is the same move -- rank-reveal the balanced stack and drop --
    and Gill, Golub, Murray & Saunders, *Math. Comp.* 28(126):505-535 (1974)
    §5.2 is why the Gram route is the one that has to go.

    Nothing here is ``L``-sized: the TSQR accumulates chunk by chunk into one
    ``(1 + k_a, 1 + k_a)`` factor.
    """
    f = _overlap_arrow(p)
    L, k_a = p.dims
    w_cat, w_border = f.solve(p.u_cat.reshape(L, 1), p.u_border)
    U_eff = p.U - (p.c * (w_border[0] + w_cat.reshape(L))[:, None] + p.V @ w_border[1:])

    width = 1 + k_a
    combined_base = np.asarray(p.overlap_rows, dtype=np.float64).reshape(-1, width)
    combined = combined_base
    if L:
        chunk = _trace_chunk_width(k_a, width, L)
        padded = np.zeros((min(chunk, L), k_a, width), dtype=np.float64)
        for start in range(0, L, chunk):
            block = p.R[start : start + chunk]
            padded[: block.shape[0], :, 1:] = block
            combined = _reduce_row_factors(combined, padded[: block.shape[0]])
    if combined.size:
        row_factor = np.linalg.qr(combined, mode="r")
        singular, right = np.linalg.svd(row_factor)[1:]
        cut = max(row_factor.shape) * np.finfo(np.float64).eps * float(singular[0])
        rank = int(np.count_nonzero(singular > cut))
        basis = right[:rank].T / singular[:rank]
    else:
        rank = 0
        basis = np.zeros((width, 0), dtype=np.float64)

    root_penalty = _penalty_root(p.S_a)
    trace_S = float(np.trace(p.S_a))
    kept_S = float(np.sum(np.square(root_penalty)))
    clip = abs(trace_S - kept_S) / max(abs(trace_S), np.finfo(np.float64).tiny)
    # THE TWO HALVES SCORE ONE PENCIL AND THIS IS THE SECOND LINE, NOT THE
    # FIRST.  ``_evaluate`` hands this same projection to ``_pair_arrow``, so
    # the statistic and the edf are built from ``rootS' rootS`` together; what
    # is left to check is that the PROJECTION ITSELF is roundoff, because the
    # DENSE path still assembles ``S_ti`` raw and the two routes have to stay
    # comparable.  A projection that removed something material would mean the
    # structured route was scoring a different model from the one the caller
    # specified, and silently.
    #
    # The certification is the eigensolver's own documented error bound rather
    # than a chosen tolerance.  A symmetric eigendecomposition returns
    # eigenvalues within ``p(n) eps ||S||_2`` of the true ones, with
    # ``p(n) = n`` in the LAPACK Users' Guide's bound (3rd ed., SIAM 1999,
    # sec. 4.7), so a genuinely PSD ``S_a`` cannot show a negative eigenvalue
    # below ``-n eps ||S||_2``.  There are at most ``n`` of them; each moves
    # the trace by ``|w|`` if dropped and by ``2 |w|`` if taken at its
    # magnitude, which is what :func:`_penalty_root` does inside that bar; and
    # ``tr(S) >= ||S||_2`` for a matrix in the cone.  So the relative trace
    # mass a certified-roundoff projection can move is at most ``2 n^2 eps``.
    # Measured across every geometry this suite exercises, the worst observed
    # is 0.1875 of that -- the bound is derived, and it is not tight to what was
    # seen.
    if clip > 2.0 * p.S_a.shape[0] ** 2 * np.finfo(np.float64).eps:
        raise _UnstableStructuredEDFError(
            f"the penalty's PSD projection removed {clip} of its trace, which is "
            "more than an eigensolver's backward error: the statistic and the edf "
            "would be scoring different pencils"
        )

    # ``sum over ALL levels of Q_q' Q_q`` is EXACTLY ``I_r``: that is what
    # makes ``Q`` an orthonormal basis, and every bound the ladder rests on --
    # ``Psi N^+ Psi' <= I``, hence ``H >= 0``, hence the deflation -- is that
    # identity in disguise.  Its computed residual is the floor those bounds
    # actually hold to, so it is measured here rather than assumed, in one
    # chunked pass that touches nothing the trace pass has not already formed.
    coupling = np.ascontiguousarray(basis[1:])
    base_gram = combined_base @ basis
    closure = base_gram.T @ base_gram
    if L:
        chunk = _trace_chunk_width(k_a, rank, L)
        for start in range(0, L, chunk):
            carried = p.R[start : start + chunk] @ coupling
            closure += np.einsum("lkr,lks->rs", carried, carried, optimize=True)
    defect = float(np.linalg.norm(closure - np.eye(rank), 2)) if rank else 0.0

    return _PairGeometry(
        U_eff=U_eff,
        coupling=coupling,
        base_gram=base_gram,
        root_penalty=root_penalty,
        penalty_clip=clip,
        orthonormality=defect,
        overlap_rank=rank,
        ceiling=float(L * k_a),
    )


def _block_inverse_factors(triangular: NDArray) -> NDArray:
    """Batched ``T^+`` for a stack of small triangular factors, at the LAPACK cut."""
    u, singular, vt = np.linalg.svd(triangular)
    cut = max(triangular.shape[-2:]) * np.finfo(np.float64).eps
    top = np.maximum(singular[..., :1], np.finfo(np.float64).tiny)
    keep = singular > cut * top
    inv = np.where(keep, 1.0 / np.where(keep, singular, 1.0), 0.0)
    return (np.swapaxes(vt, -1, -2) * inv[..., None, :]) @ np.swapaxes(u, -1, -2)


def _psd_clip_allowance(n: int, scale: float, closure: float) -> float:
    """Largest negative eigenvalue an exactly-PSD block can show at ``scale``.

    ``_psd_factor``'s ``max(w, 0)`` is the Euclidean projection onto the PSD
    cone (Higham, *LAA* 103:103-118, 1988) and is the right thing to do to
    roundoff -- but it does the same thing to a block that is materially
    indefinite because a deflation misfired, and there the contribution is
    still a squared norm and still lands inside ``[0, ceiling]``.  A clip
    nobody bounds turns a numerical breakdown into a plausible published
    ``edf``, so the two cases are separated here.

    Both terms are documented error bounds rather than chosen tolerances:

    * ``eigh`` returns eigenvalues within ``p(n) eps ||M||_2`` of the true
      ones, with ``p(n) = n`` in the *LAPACK Users' Guide* (3rd ed., SIAM
      1999, sec. 4.7) bound for the symmetric eigenproblem;
    * ``W_q`` reaches it through at most three chained products in ``n``
      dimensions, each contributing ``n eps`` relative (Higham, *ASNA* 2nd
      ed., Thm 3.5).

    So ``4 n eps ||M||_2``, taken at ``8 n eps`` to round the chain depth up
    rather than fit it.

    **THE SCALE IS NOT ALWAYS THE BLOCK'S OWN NORM, AND THAT IS THE WHOLE
    SUBTLETY.**  ``What_q``'s leading block is ``I + Y H^+ Y'``, so its norm
    is at least 1 and a relative test is well posed.  ``Xi`` alone is not:
    its eigenvalues are ``(1 - s)(1 + s) / s^2`` for ``H``'s singular values
    ``s``, which lie in ``[0, 1]`` EXACTLY, so a fully absorbed direction
    gives ``s = 1 + eps`` and an eigenvalue of ``-2 eps / s^2`` while the
    block's own norm is that same ``2 eps`` -- every relative bound passes
    trivially and the test says nothing.  The reference there is what bounds
    the block, ``||H^+||_2``, so ``scale`` is ``max(||M||_2, ||H^+||_2)``.

    **AND ON THIS GEOMETRY eps IS NOT THE TERM THAT DOMINATES.**  ``W_q``'s
    PSD-ness is a knife edge, not a margin.  In direction ``j`` its 2x2
    section is ``[[1 + y^2/h, -y/h], [-y/h, (1 - h)/h]]`` with determinant
    ``(1 - h - y^2)/h``, so it is PSD exactly to the extent that
    ``sum_q y_qj^2 = 1 - h_j`` -- which is ``sum_q Q_q' Q_q = I_r`` again, the
    identity everything here rests on, and whose computed residual
    ``closure`` this module already measures.  A residual of ``d`` moves that
    determinant by ``d/h``, hence an eigenvalue by ``d ||H^+||``, and on the
    starved fixture that term is ``2.1e-11`` where ``n eps ||M||`` is
    ``3.7e-14``: three orders larger, and it is the one that decides.  A
    clip below their sum is roundoff on a quantity that is only determined
    that far; above it, something other than arithmetic moved the block.
    """
    return _PSD_CLIP_FACTOR * max(int(n), 1) * float(np.finfo(np.float64).eps) * np.maximum(
        scale, 0.0
    ) + np.maximum(closure, 0.0)


def _psd_factor(M: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Batched ``F`` with ``F F' = M`` for symmetric ``M``, negatives clipped.

    Returns the factor, the size of what the clip threw away
    (``max(-w_min, 0)``, which is exactly ``||M - M_+||_2``) and the block's
    own norm, so the caller can decide whether the clip was roundoff.  See
    :func:`_psd_clip_allowance` for why it is the caller and not this function
    that owns the decision.

    Equilibrating ``M`` to a unit diagonal before the eigendecomposition was
    tried and is NOT adopted: on the near-absorbed pair it moves the
    mid-bracket rung from 2.4e-04 relative to 1.1e-16 and the surrounding
    three lambdas the other way, from 4.4e-16 to 2.2e-04, and it reds one more
    of this module's own fixtures than it greens.  Recorded so the next reader
    does not repeat it.
    """
    w, Q = np.linalg.eigh(0.5 * (M + np.swapaxes(M, -1, -2)))
    clipped = np.maximum(-np.min(w, axis=-1, initial=0.0), 0.0)
    top = np.maximum(np.max(np.abs(w), axis=-1, initial=0.0), 0.0)
    return Q * np.sqrt(np.where(w > 0.0, w, 0.0))[..., None, :], clipped, top


def _absorption_floor(n_terms: int, rank: int, defect: float) -> float:
    """Cut below which a singular value of ``H``'s factor is not resolvable.

    ``H = I_r - Psi N^+ Psi'`` is delivered here as ``R_H' R_H`` with ``R_H``
    assembled by orthogonal transformations only, so the quantity being cut is
    a SINGULAR VALUE and not an eigenvalue of a difference.  ``H``'s exact
    spectrum lies in ``[0, 1]``, so the scale is 1 and the cut is absolute.

    Two things bound how far float64 can move it, and both are measured:

    * ``defect`` is ``||sum_q Q_q' Q_q - I_r||_2``.  ``sum_q X_q' X_q +
      Gb' Gb`` is ``H`` only because that sum is ``I_r``, so the residual is
      subtracted from ``H``'s eigenvalues one for one -- and an eigenvalue
      floor of ``defect`` is a singular-value floor of ``sqrt(defect)``.
    * the TSQR that assembles ``R_H`` over ``L`` blocks is backward stable
      with ``||R_H|| <= 1``, so its singular values carry an absolute error of
      order ``L k_a eps``.

    Measured, the gap this has to land in is enormous: on the starved fixture
    at the bracket's low edge ``R_H``'s singular values are ``1.00, 0.849,
    0.664`` and then ``3.4e-13`` and below -- twelve orders in ``sigma``,
    twenty-four in ``H`` -- where the SAME quantity taken as ``I_r`` minus a
    Gram separates only ``0.44`` from ``7e-13``.  Nothing is fitted; both
    terms are dimensions or measured residuals.
    """
    eps = float(np.finfo(np.float64).eps)
    return float(np.sqrt(max(defect, 0.0))) + max(int(n_terms), int(rank), 1) * eps


def _filter_factor_sum(
    p: SplineCatPair, geometry: _PairGeometry, lam: float
) -> tuple[float, float]:
    """``tr(A(lambda)^+ V_eff)`` as a sum of squared norms, and its clip bound.

    **THIS IS THE SUM THE MODULE HEADLINE CLAIMS, AND NOW IT IS ONE.**  With
    ``E_q`` the selector for level ``q``'s block, ``M_q = [E_q, -Psi']`` and
    ``F_q = [Zc_q | Q_q]``,

        C_q = F_q' F_q,   W_q = M_q' A^+ M_q,
        edf = sum over ALL levels of tr(W_q C_q) = sum || F_q chol(W_q) ||_F^2,

    which closes because ``sum_q Q_q' Q_q = I_r``.  ``W_q`` and ``C_q`` are
    both Gram matrices, so every term is a squared norm: no difference of two
    large numbers survives into the answer.  That is Wood's ``tr(F) =
    ||K||_F^2`` (*JRSS-B* 73(1):3-36, 2011, §3.6 and App. H) carried onto a
    block-angular structure instead of a dense ``X``.

    **EVERY PIECE IS BUILT BY ORTHOGONAL TRANSFORMATIONS.**  Per emitted level
    one QR delivers all three of them at once::

        [[  R_q          ,  Phi_q ],        [[ T_q , Y_q ],
         [ sqrt(lam) rootS,   0    ]]  =  Q  [[  0  , X_q ]]

    with ``Phi_q = R_q coupling``, so ``T_q' T_q = D_q + lambda S_a`` is never
    summed, ``Y_q = T_q^-T Psi_q'`` is never solved for, and

        X_q' X_q = Gam_q - Psi_q N_q^+ Psi_q'

    is never differenced.  Summing the trailing blocks over emitted levels and
    adding the unemitted levels' ``Gb`` gives ``H = I_r - Psi N^+ Psi'``
    itself, again with no subtraction, because ``sum_q Gam_q = I_r``.  That is
    the block-angular QR of Golub & Plemmons, *LAA* 34:3-28 (1980) -- local
    eliminations feeding one border -- and it is exactly what Gill, Golub,
    Murray & Saunders, *Math. Comp.* 28(126):505-535 (1974) §5.2 prescribes
    over forming a Schur complement and factoring it: "numerically less
    satisfactory than computing R using orthogonal matrices", and (p. 516)
    downdating is stable only when the result's smallest eigenvalue is large
    relative to the norm of the removed term, which is this module's regime.

    ``A^+`` IS NOT ASSEMBLED, and it does not have to be.  With

        B = N^+ + N^+ Psi' H^+ Psi N^+

    -- every term PSD, nothing indefinite, no Woodbury middle to invert --
    ``B`` differs from ``A^+`` only on ``null(A)``, and ``range(V_eff) <=
    range(A)`` because ``A = V_eff + lambda S`` with both summands PSD.  So
    ``tr(B V_eff) = tr(A^+ V_eff)`` EXACTLY: if ``A x = A x'`` then
    ``v'(x - x') = 0`` for every ``v`` in ``range(A)``, and both are solutions
    of the same consistent system.  Where ``H`` is singular the deflation is
    then just ``H^+``, taken at :func:`_absorption_floor`.

    **THE CONTRACTION IS DONE BEFORE THE INVERSE, NOT AFTER.**  ``A^+``'s own
    blocks run to ``||T_q^+||^2``, which is 1e+17 on a starved pair at the
    bracket's low edge while the answer is 6.  Every appearance of ``T_q^+``
    here is already contracted against ``R_q`` as ``K_q = R_q T_q^+``, whose
    norm is bounded by 1 because ``R_q' R_q <= T_q' T_q``; the level's whole
    contribution is then ``|| [K_q | Phi_q] chol(What_q) ||_F^2`` with

        What_q = [[ I + Y_q H^+ Y_q' , -Y_q (H^+ + P_0) ],
                  [        .          ,       Xi        ]]

    bounded by ``||H^+||``.  Measured on the starved fixture, contracting
    afterwards instead publishes 10.996 df against a certified 6.000.

    Two chunked passes over the levels, each bounded by
    ``_TRACE_CHUNK_DOUBLES``; ``K`` and ``Y`` are what they carry between them.

    The second return value is ``sum_q ||F_q||_F^2 ||W_q - W_q+||_2``, a bound
    IN DEGREES OF FREEDOM on everything :func:`_psd_factor`'s clip removed;
    :func:`_evaluate` refuses the evaluation when it is not dust.
    """
    L, k_a = p.dims
    r = geometry.overlap_rank
    coupling = geometry.coupling
    root = np.sqrt(lam) * geometry.root_penalty
    height, width = k_a + root.shape[0], k_a + r
    chunk = _trace_chunk_width(height, width, L) if L else 1
    blocks = [(s, min(L, s + chunk)) for s in range(0, L, chunk)]

    # --- pass 1: one block-angular QR per level, and H's factor ------------
    contracted = np.empty((L, k_a, k_a), dtype=np.float64)
    cross = np.empty((L, k_a, r), dtype=np.float64)
    border = np.asarray(geometry.base_gram, dtype=np.float64)
    local = np.zeros((min(chunk, max(L, 1)), height, width), dtype=np.float64)
    for start, stop in blocks:
        rows = p.R[start:stop]
        view = local[: stop - start]
        view[:, :k_a, :k_a] = rows
        view[:, :k_a, k_a:] = rows @ coupling
        view[:, k_a:, :k_a] = root
        # Scale the leading column block to unit norm.  The QR's backward
        # error is relative to the WHOLE matrix, and ``X_q`` is bounded by 1
        # while ``R_q`` carries the data scale, so without this the trailing
        # block is delivered to ``eps ||R_q||`` rather than to ``eps``.  A
        # column scaling leaves ``Y_q`` and ``X_q`` untouched and divides
        # ``T_q`` by the same factor, which is undone below.
        scale = np.linalg.norm(view[:, :, :k_a], axis=(1, 2))
        scale = np.where(scale > 0.0, scale, 1.0)
        view[:, :, :k_a] /= scale[:, None, None]
        factored = np.linalg.qr(view, mode="r")
        contracted[start:stop] = (rows @ _block_inverse_factors(factored[:, :k_a, :k_a])) / (
            scale[:, None, None]
        )
        cross[start:stop] = factored[:, :k_a, k_a:]
        border = _reduce_row_factors(border, factored[:, k_a:, k_a:])

    # --- H's spectrum, from its factor's singular values -------------------
    if r:
        singular, right = np.linalg.svd(border)[1:]
        keep = singular > _absorption_floor(L * k_a, r, geometry.orthonormality)
        occupied = singular * singular
        free = (1.0 - singular) * (1.0 + singular)
        safe = np.where(keep, occupied, 1.0)
        # ``I + H^+ (I - H)`` is ``H^+`` PLUS the projector onto
        # ``null(H)``: dropping the second term loses a whole degree of
        # freedom per deflated direction -- measured, 22.0 against a certified
        # 6.000 on the starved fixture.
        resolved = (right.T * np.where(keep, 1.0 / safe, 0.0)) @ right
        extended = (right.T * np.where(keep, 1.0 / safe, 1.0)) @ right
        xi = np.where(keep, free / safe, free)
        coupled = (right.T * xi) @ right
        coupled = 0.5 * (coupled + coupled.T)
        deflation = float(np.max(np.where(keep, 1.0 / safe, 1.0)))
    else:
        resolved = extended = coupled = np.zeros((0, 0), dtype=np.float64)
        deflation = 1.0

    # --- pass 2: one PSD factor per level, and the squared norms -----------
    total = 0.0
    correction = 0.0
    uncertified = 0.0
    block = np.empty((min(chunk, max(L, 1)), width, width), dtype=np.float64)
    for start, stop in blocks:
        carried = p.R[start:stop] @ coupling
        Y = cross[start:stop]
        contract = contracted[start:stop]
        Yt = np.swapaxes(Y, -1, -2)
        crossed = Y @ extended
        view = block[: stop - start]
        view[:, :k_a, :k_a] = np.eye(k_a) + Y @ resolved @ Yt
        view[:, :k_a, k_a:] = -crossed
        view[:, k_a:, :k_a] = -np.swapaxes(crossed, -1, -2)
        view[:, k_a:, k_a:] = coupled
        rows = np.concatenate((contract, carried), axis=2)
        factor, negative, top = _psd_factor(view)
        term = float(np.sum(np.square(rows @ factor)))
        # What the clip removed BEYOND what roundoff explains, carried into
        # the units of the answer: for the excess ``e_q``,
        # ``|tr(F_q (W - W_+) F_q')| <= ||F_q||_F^2 e_q`` by Weyl plus the
        # trace bound, so this is a bound in DEGREES OF FREEDOM and
        # :func:`_evaluate` can hold it against the guard's own allowance.
        # The scale is the largest INTERMEDIATE the block passed through,
        # ``||H^+|| max(1, ||Y_q||)^2``, and not the block's own norm: the
        # products that build it can cancel, and a bound taken on the result
        # of a cancellation is not a bound on the cancellation.
        scale = np.maximum(top, deflation * np.square(1.0 + np.linalg.norm(Y, axis=(1, 2))))
        allowance = _psd_clip_allowance(width, scale, geometry.orthonormality * deflation)
        excess = np.maximum(negative - allowance, 0.0)
        uncertified += float(np.sum(np.sum(np.square(rows), axis=(1, 2)) * excess))
        # Neumaier-style compensated accumulation: the chunk totals are
        # nonnegative, so this only keeps the scalar independent of chunking.
        updated = total + term
        if abs(total) >= abs(term):
            correction += (total - updated) + term
        else:
            correction += (term - updated) + total
        total = updated

    # The levels the menu does not emit all carry the SAME ``W_q = Xi``, so
    # their whole contribution is one term against the compacted Gram factor
    # rather than one term per level.
    factor, negative, top = _psd_factor(coupled)
    total += float(np.sum(np.square(geometry.base_gram @ factor)))
    excess = np.maximum(
        negative
        - _psd_clip_allowance(r, np.maximum(top, deflation), geometry.orthonormality * deflation),
        0.0,
    )
    uncertified += float(np.sum(np.square(geometry.base_gram)) * float(excess))
    return total + correction, uncertified


def _evaluate(p: SplineCatPair, geometry: _PairGeometry, lam: float) -> tuple[float, float]:
    """``(T, edf)`` at one lambda.

    ``T`` still reads the pair arrow, and therefore still reads the moments.
    ``edf`` reads neither: :func:`_filter_factor_sum` works from the row-space
    factors ``p.R`` and the compacted overlap rows, which is why the two are
    two factorizations rather than one.  What that buys is stated in the
    module docstring under "WHAT THE MOMENTS COST"; what it costs is one extra
    factorization per evaluation, measured there too.  The one thing they DO
    share is the penalty: both are built from ``rootS' rootS``, so the lambda
    the ladder chooses and the statistic it publishes come from one pencil.
    """
    L, k_a = p.dims
    f = _pair_arrow(p, lam, geometry.root_penalty.T @ geometry.root_penalty)
    b = np.zeros((L, k_a + 1), dtype=np.float64)
    b[:, :k_a] = geometry.U_eff
    x, _ = f.solve(b, np.zeros(1 + k_a, dtype=np.float64))
    T = float(np.sum(geometry.U_eff * x[:, :k_a]))

    # THE TWO HALVES ARE INDEPENDENT, SO THEIR TEMPORARIES MUST NOT OVERLAP.
    # ``f`` holds the arrow's ``Ginv`` and ``Y``, both level-sized, and ``b``
    # and ``x`` are two more; the filter pass then allocates ``contracted``
    # and ``cross``.  Nothing below reads any of the four, and the structured
    # allocation gate is written for the stacks that have to COEXIST, so they
    # are released here rather than at the end of the frame.  Measured peak of
    # one evaluation at L = 2000, k_a = 13: 26.18 -> 19.46 MB, which is the
    # 6.72 MB those four arrays hold (Ginv 2000x14x14, Y 2000x14x14, b and x
    # 2000x14) to three digits.
    del f, b, x

    edf, uncertified = _filter_factor_sum(p, geometry, lam)

    # Every term of the sum is a filter factor ``a_j / (a_j + lambda s_j)``
    # with both parts nonnegative, so the sum lies in ``[0, L * k_a]`` for
    # every lambda and every pair -- a property of the identity, not a
    # tolerance.  Every term COMPUTED above is a squared norm, so the lower
    # end of that bound now holds by construction rather than by luck.
    #
    # THE UPPER END IS STILL A GUARD AND NOT AN ASSERTION.  ``A^+`` is
    # assembled from a deflation decision, and a deflation that misfires can
    # invert a direction the pencil does not resolve; measured once, at 2237
    # against a ceiling of 77 on a starved pair at the bracket's low edge.
    # That is exactly what the bound is here to refuse.  The allowance is
    # taken on the answer and the ceiling because both terms are now
    # nonnegative and there is no cancellation left to bound: pairwise
    # summation of ``n`` nonnegative terms has error at most ``log2(n) eps``
    # times their total, and ``_EDF_ROUNDOFF_FACTOR`` covers ``n`` up to
    # ``2**64``.
    roundoff = _edf_roundoff(edf, geometry.ceiling)
    # ``uncertified`` is what made every term a squared norm in the first place.
    # Being nonnegative BY CONSTRUCTION is only worth something if the
    # construction is certified: ``max(w, 0)`` silences an indefinite block
    # just as effectively as it silences roundoff, so the two are separated
    # here, on the one scale where the question is decidable -- degrees of
    # freedom, against the same allowance the range guard takes.
    if not np.isfinite(edf) or edf < -roundoff or edf > geometry.ceiling + roundoff:
        raise _UnstableStructuredEDFError(
            f"structured EDF is not a filter-factor sum: {edf} (ceiling={geometry.ceiling})"
        )
    if not uncertified <= roundoff:
        raise _UnstableStructuredEDFError(
            f"the PSD clip removed {uncertified} df beyond its backward error, on {edf} "
            f"(allowance={roundoff}, ceiling={geometry.ceiling})"
        )
    # THE DUST BAND IS TWO-SIDED, SO THE COLLAPSE IS TOO.  Everything inside
    # ``[-roundoff, roundoff]`` was just declared indistinguishable from zero;
    # returning the positive half of it as a value would be the guard
    # contradicting itself one line later, and the contradiction is not
    # cosmetic.  ``screen_interactions`` divides by ``sqrt(2 * edf0)``, so a
    # published ``edf0`` of 3e-17 -- the same measurement as -3e-17, which
    # collapses to 0.0 and skips the rung -- inflates that pair's ``z`` by
    # ~1e8 and sorts a pair that resolved nothing to the top of the screen.
    # Which side of zero a cancellation residue lands on is not something this
    # module gets to decide, so it must not be what decides a ranking.
    if abs(edf) <= roundoff:
        return T, 0.0
    return T, min(edf, geometry.ceiling)


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

    **EVERY EVALUATION NOW COSTS THE SAME, BECAUSE NOTHING LAMBDA-DEPENDENT IS
    COUNTED.**  The form this replaces ran a batched eigendecomposition per
    evaluation to count the level ranks at that lambda; this one contracts
    ``V_eff`` against the factorization the evaluation already built.  Whole
    ladders, ``time.process_time`` with all six thread pools pinned to one,
    median of five, filter factors against ``rank - lambda tr(A^-1 S)``:
    clamped ``ps(8)`` at ``L = 200`` 5.51 ms against 5.97 (0.92x), the same at
    ``L = 2000`` 50.87 against 45.52 (1.12x), clamped ``cr(3)`` at ``L = 2000``
    7.03 against 7.26 (0.97x), a ``ps(8)`` ladder driven to bisect four rungs
    345.2 against 318.5 (1.08x), and ``ns(8)``, whose full-rank penalty makes
    every rung search, 195.8 against 176.4 (1.11x).  0.92x to 1.12x, and the
    +9.2% a searching ladder used to pay for its per-lambda count is gone.

    **THAT IS THE KERNEL, WHICH IS NOT THE THING A CALLER PAYS.**  The same
    four configurations through the PUBLIC entry -- ``fit_reml`` then
    ``screen_interactions`` -- with dispatch counted at both kernels, CPU the
    median of five whole screens, memory the ``tracemalloc`` peak over one:

      ps(8) L=400     198.86 ms against 210.13   0.95x   peak 34.42 MiB
      ps(8) L=2000    294.76 against 293.00      1.01x   peak 25.42 MiB
      cr(3) L=2000    248.11 against 240.74      1.03x   peak 22.07 MiB
      ns(8) L=200     238.30 against 232.36      1.03x   peak 11.91 MiB

    Peak allocation is IDENTICAL TO THE BYTE on all four, so the per-level
    ``D`` and the contraction temporaries do not move it; the arrays they
    replace were the same size.  Dispatch is identical too -- one structured
    call, zero dense calls, zero refusals on every configuration -- so the
    production route is unchanged and these are the same pairs being scored.
    Of the published columns, ``statistic`` and ``lambda0`` are BIT-IDENTICAL
    on all four; ``edf0`` moves 1.75e-03, 1.18e-02, 4.75e-03 and 4.8e-12, and
    the ``z`` the screen ranks on moves 9.11e-05, 2.81e-04, 1.13e-04 and
    1.3e-12.  That is the user-visible size of this change on wide pairs.

    Arrow factorizations for a WHOLE ladder at the default ``(2, 4, 8, 16)``,
    counted by instrumenting :func:`_pair_arrow`: 2 for ``ps(8)`` at ``L = 50``
    and ``L = 100``, ``cr(6)`` at ``L = 100``, ``bs(6)`` at ``L = 50`` and
    ``ps(8)`` at ``L = 20``.  Their edf at maximum penalty is far above every
    budget in use, so no rung's target falls inside the bracket and every one
    of them clamps.  The one margin that searches by default is ``ns``: its
    penalty is full rank, edf at maximum penalty is 0, and ``L = 100`` pays
    106.

    A numerical failure reached while bisecting one target refuses that target,
    not independent targets or already certified edge clamps.  The returned
    list may therefore contain fewer entries than ``budgets``.  If no rung
    survives, ``None`` preserves the pair-refusal signal and lets a speculative
    structured route hand the dense path back.  A failure in the
    lambda-INDEPENDENT half refuses the pair outright, since no lambda can
    make it go away.
    """
    if p.profiled_trace is None:
        return None

    # One M factorization for the whole ladder: it carries U_eff and both
    # halves of V_eff, none of which depends on lambda.
    try:
        geometry = _profile(p)
    except _UnstableStructuredEDFError:
        return None

    def evaluate(lam: float) -> tuple[float, float] | None:
        try:
            return _evaluate(p, geometry, lam)
        except _UnstableStructuredEDFError:
            return None

    if not np.any(p.S_a):
        # No penalty to scan, exactly the predicate the dense ladder applies:
        # one rung, at the block's own achieved edf, with lambda0 = 0.  That
        # edf is `tr(V_eff^+ V_eff)`, which IS `rank(V_eff)` in exact
        # arithmetic -- but it is not counted, and on a near-rank pair it is
        # not an integer; see the module docstring, which measures how far
        # that lands from the dense path's counted rank and why the trace is
        # the better of the two.  A
        # zero penalty would otherwise make the bracket below infinite and
        # every rung NaN, since inf * 0 is not a number.
        evaluated = evaluate(0.0)
        if evaluated is None:
            return None
        stat, edf = evaluated
        return [ScreenedPair(statistic=stat, edf0=edf, lambda0=0.0) for _ in budgets]

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
