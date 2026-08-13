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
Diagonalize the pencil and ``tr(A^-1 V_eff)`` is ``sum_j a_j / (a_j + lambda
s_j)`` — one Tikhonov filter factor per direction, every term in ``[0, 1]``,
bounded and cancellation-free.  That is the standard form for the effective
degrees of freedom of a regularized fit: Golub, Heath & Wahba,
*Technometrics* 21:215-223 (1979); Elden, *BIT* 17:134-145 (1977) and 22:487-
502 (1982); Hansen, Nagy & O'Leary, *Deblurring Images* (SIAM 2006), ch. 6.

This used to be written ``rank(A) - lambda tr(A^-1 S)``, using
``V_eff = A - lambda S``, which is the same number as a DIFFERENCE of two
independently thresholded quantities — an integer rank against a trace.  A
direction the rank counted and the inverse dropped contributed ``1 - 0``: a
whole degree of freedom with no penalty offset, and no bound on the error
because the two halves were decided by different cuts.  Roughly 300 lines of
this module existed to make those two cuts agree; they are gone, and with them
issues #249, #258, #263's stated mechanism, #265 and #271, all of which are
statements about a rank that no longer exists.

``V_eff`` is dense, so evaluating the trace against it needs its structure
rather than the matrix.  Profiling the level contrasts out of each block
leaves

    V_eff = blockdiag(D_q) - D' Omega D,     D_q = V_q - c_q c_q' / m_q

with ``Omega`` the ``(k_a, k_a)`` spline-main corner of ``Sigma_M^-1``, the
border Schur complement of the OVERLAP arrow — verified against a dense
``V - C' M^+ C`` to 4.8e-14 relative.  **``V_eff`` IS NOT BLOCK DIAGONAL**:
the correction has rank exactly ``k_a`` whatever ``L`` is, but it is not
small, so anything that treats the levels as separable is missing a
first-order term.  Both halves contract through the arrow inverse's own
blocks in O(L): the block-diagonal half against ``diag_blocks()``, and the
rank-``k_a`` half against ``Y`` and ``Sinv``, since
``[A^-1]_qq' = delta_qq' Ginv_q + Y_q Sinv Y_q''`` is all the off-diagonal
structure there is.

This path is entered only where the dense path is refused, so no production
pair is ever scored twice and nothing at runtime cross-checks the two.  What
stands behind the reorganization is the parity pinned in
tests/test_structured_screening.py, and that parity is a tolerance, not
bit-for-bit equality: away from the bracket the two agree to round-off, at
the edges they agree to the tolerances measured there.

**KNOWN DIVERGENCE, NOT CLOSED.**  That parity is measured on the fixtures the
suite carries and does not generalize; on narrow-band geometries the two paths
part by whole degrees of freedom at the ladder's high edge, and neither is
reliably the better one.  This change does not close that class — it moves the
arrow side of it and the measurements below say by how much and in which
direction.  Parity is therefore no longer the top-level evidence: every claim
here is against a high-precision oracle on the delivered moments, which can
arbitrate where parity cannot.

**WHAT THE FILTER-FACTOR FORM IS WORTH, MEASURED.**  Against an mpmath oracle
on each pair's DELIVERED float moments — the pencil simultaneously diagonalized
once and every lambda read off the filter factors, evaluated at 50 and 70
digits and identical to every digit quoted below at both — over 22 pairs at the
bracket's low edge, mid-bracket and high edge.  The pairs are ``ps``/``cr``/
``bs``/``ns`` margins at 5 to 10 knots, ``L`` from 5 to 39, 3 to 40 rows per
level, unit and 1e-12 weights, narrow-band and not, plus a real
``freMTPL2freq`` pair.  Of the 57 points where both forms return a value the
new one is nearer on 32, farther on 23 and identical on 2, and the worst error
anywhere falls from 53.08 df to 22.01 df.

That summary hides the shape, so the shape is stated by edge:

* **LOW edge**, which is where the largest budgets clamp: nearer on 15 of 19.
  On a 19-level ``ps(8)`` pair 3.998 -> 0.0068 df, 588x; on 5-level ``ps``/
  ``cr`` pairs 4.235 -> 0.700, 3.100 -> 0.747, 3.114 -> 0.695, 1.092 -> 0.754.
  On the starved family below, nearer on 9 of 9, worst 44.39 -> 22.01 df.
* **HIGH edge**: nearer on 8 of 18.  On the wide two-column pairs this kernel
  exists for, 2.34e-04 -> 1.63e-05 df (14x) and 6.07e-05 -> 2.64e-05.  Farther
  on the 5-level narrow-band pairs, worst 1.000 -> 2.000 df.
* **MID-bracket**: FARTHER on 9 of 10 well-posed pairs, 1.1e-09 -> 1.1e-05 df.
  That is the price of the one subtraction this form still carries — the
  block-diagonal half of ``V_eff`` against its rank-``k_a`` correction — and on
  the WELL-POSED family it is five orders below the errors it removes.  It is
  not five orders below them everywhere; see the tail below.

On 8 of the 66 points the new value lands outside ``[0, L k_a]`` and is refused
where the old form PUBLISHED numbers wrong by 0.998 to 53.08 df; on one the old
form refused and the new one returns a value 2.23 df from the oracle.

**THE ADVERSARIAL TAIL, MEASURED SEPARATELY AND PARTLY AGAINST THIS CHANGE.**
The 22-pair sample above is a cross-section; it is not the worst case, and
three of its lines read better than the truth because of which pairs it holds.
A second, deliberately hostile sample — 12 pairs, all narrow-band ``cr(3)``,
nullity-two ``ps(5, m=3)`` and starved ``ps(8)``/``cr(5)`` geometries — was
measured at the same three bracket points against ARB BALL ARITHMETIC
(python-flint) on each pair's exact design at 320 AND 640 bits, agreeing to
every digit reported and returning balls of radius 1e-137 or smaller.  On those
36 points **the two forms split 18/18**.  What the new one buys is the tail:
the worst error anywhere falls 7.995 -> 3.108 df (2.6x), and per edge

* LOW: nearer on 5 of 12; worst 7.995 -> 2.746 df.  The starved ``ps(8)``
  8-levels-by-3-rows pair goes 7.995 -> 0.401 (19.9x) — that is the case this
  form exists for.  Against it, ``cr(3)`` at band 1e-4 goes 5.4e-04 -> 0.911
  and ``cr(5)`` starved goes 2.5e-04 -> 0.148.
* MID: nearer on 5 of 12, and **FARTHER ON ALL SEVEN NARROW-BAND PAIRS**, by
  up to 0.513 df (``cr(3)``, band 1e-4, 2.1e-08 -> 0.513) and 0.170 df (band
  1e-4 at ``L = 5``).  The "1.1e-09 -> 1.1e-05" line above is the well-posed
  family only and must not be read as the mid-bracket cost in general.
* HIGH: nearer on 8 of 12; worst 5.410 -> 3.108 df.  **The published high-edge
  value REGRESSES by up to +2.017 df** (``cr(3)``, band 2e-3, 1.7e-03 ->
  2.019) and by +0.232 df on nullity-two ``ps(5, m=3)``.  Against that it goes
  1.753 -> 0.053, 2.156 -> 0.456, 5.410 -> 3.108 and 0.708 -> 0.0026 elsewhere
  on the same sample.

The same regression is visible on the suite's own ``_thin_level_pair`` at both
edges: the high edge goes 2.15e-05 -> 3.90e-05 at unit weight and 6.11e-06 ->
1.20e-04 at 0.01 (19.6x), improving only at 0.001 (1.31e-03 -> 5.60e-04); the
LOW edge goes 6.97e-09 -> 3.96e-11 at unit weight (175x nearer) and **2.69e-09
-> 1.29e-07 at 0.001, 47.8x FARTHER**.  All six are inside the derived bounds
that fixture asserts, which is why they are disclosure and not a defect, and
all six are stated in
``test_a_thin_level_does_not_cost_the_pair_a_degree_of_freedom``.

**THE ZERO-PENALTY RUNG STOPS BEING AN INTEGER, DELIBERATELY.**  Where ``S_a``
is absent or identically zero the ladder publishes one rung at ``lambda = 0``.
The old form returned ``rank(A)`` there, so that rung was an integer by
construction; ``tr(A^-1 V_eff)`` is ``tr(V_eff^+ V_eff)``, which IS
``rank(V_eff)`` in exact arithmetic but is whatever the inverse resolves in
float64.  THE INTEGER WAS NEVER THE ACCURACY: against the dense path's own
counted rank over twelve near-rank fixtures the old integer disagreed by up to
3.000 df and this disagrees by at most 0.9965, nearer on 6 and farther on 4,
both exact on the two well-posed ones (1.7e-12 and 7.2e-11).  Restoring an
integer would restore the larger disagreement.  Rank is simply not well defined
on these geometries: on one ``cr(3)`` pair the dense path counts 9,
``numpy.linalg.matrix_rank`` on the residualized design counts 10, and the old
arrow form counted 6.  On a nullity-two pair the ``lambda = 0`` value is only
determined to 1.98e-02 under exact relabeling, so it is not pinnable tighter
than two decimals by any implementation.

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

**HOW FAR BEYOND: ONE ULP OF LAMBDA DECIDES IT.**  On a 7-level ``bs(10)``
pair with 3 rows per level, ``local`` and ``border`` both measure 9.654e+07 at
the ladder's high edge and their difference is ``-13.61``.  Move lambda by a
SINGLE ulp — 594.8079378729931 against ...32, which is just the difference
between ``1e10 * a / b`` and ``1e10 * (a / b)`` — and the same evaluation
returns ``edf = 0.8229`` instead.  ``edf`` is not a continuous function of
lambda at float64 resolution on this family, so no bound on the VALUE is
available there and none is claimed; the guard is what carries it, and
``structured_ladder`` refuses the pair on either side of that ulp.
``tests/test_structured_screening.py`` pins that refusal on this geometry.

**WHAT A DROPPED DIRECTION COSTS NOW.**  The inverse's relative cut still
drops directions, and where one carries a filter factor near 1 the answer
loses it.  ``_mixed_rank_cells`` in the suite is built to sit exactly there —
its only ``V_eff`` direction is ``1e-16`` RELATIVE to its own level block,
under ``_solve_floor``'s ``3 eps`` — and a 50-digit oracle puts ``edf`` at
1.000000 / 0.500000 / 0.000000 across the bracket where this reads
0.0 / 0.0 / 0.0.  The form it replaces got that fixture's low edge right by
counting the direction against the PENALTY's scale rather than the block's,
which is the same mismatch that cost whole degrees of freedom elsewhere.  The
published row is unchanged: ``screen_interactions`` skips a rung with
``edf0 <= 0``, so the pair takes the same NaN row the old refusal produced.

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
class _PairGeometry:
    """Everything the ladder needs that does NOT depend on ``lambda``.

    ``V_eff`` is dense, but it is a rank-``k_a`` downdate of a block-diagonal
    matrix and both pieces are level-local::

        V_eff = blockdiag(D_q) - D' Omega D,     D_q = V_q - c_q c_q' / m_q

    ``D_q`` is level q's own curvature after its own contrast has absorbed what
    it can; ``Omega`` is the spline-main corner of ``Sigma_M^-1``, the border
    Schur complement of the OVERLAP arrow, and it is what couples the levels.
    Both come out of the one ``M`` factorization :func:`_profile` already
    builds for ``U_eff``, so this costs nothing beyond the ``(L, k_a, k_a)``
    array itself -- linear in ``L``, like everything else here.

    ``ceiling`` is ``L * k_a``: the number of Tikhonov filter factors summed,
    each in ``[0, 1]``, hence an exact bound on ``edf`` that takes no rank
    decision.  **A TIGHTER BOUND WAS TRIED AND IS NOT SAFE.**  ``rank(V_eff)``
    is the mathematically right ceiling and is 10 to 65 df tighter here, but
    every way of computing it in O(L) is a numerical rank, and a numerical rank
    that comes out LOW refuses true values: counted by Guttman additivity off
    the unpenalized arrow -- ``rank(K(0)) - rank(M)`` -- it reads 12 on a
    7-level ``cr(5)`` pair with 3 rows per level whose certified ``edf`` at the
    ladder's low edge is 17.618, and 36 against a certified rank of 45 on a
    5-level ``ps(8)`` pair.  Evaluating this module's own ``edf`` at
    ``lambda = 0`` is no better: 30.014 on a pair whose low-edge ``edf`` is
    30.072, so it refuses that pair's own bracket.  The loose bound that cannot
    lie is preferred to the tight one that can.
    """

    U_eff: NDArray  # (L, k_a)      the profiled score
    D: NDArray  # (L, k_a, k_a) per-level curvature, own contrast absorbed
    Omega: NDArray  # (k_a, k_a)    what couples the levels through V_eff
    ceiling: float  # L * k_a


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
    G = _unpenalized_blocks(p)
    G[:, :k_a, :k_a] += lam * p.S_a
    E = np.empty((L, r, g), dtype=np.float64)
    E[:, 0, :k_a] = p.c
    E[:, 0, k_a] = p.m
    E[:, 1:, :k_a] = p.V
    E[:, 1:, k_a] = p.c
    return factor_arrow(G, E, p.border)


def _profile(p: SplineCatPair) -> _PairGeometry:
    """The whole lambda-independent half of the work, from ONE ``M`` factor.

    ``U_eff = U - C' M^-1 u_m``.  Column ``(p, q)`` of ``C`` is nonzero in
    exactly three places — the intercept row, the spline-main rows, and level
    q's own contrast row — so the contraction never touches a level other
    than its own.  ``M`` depends on no lambda, so this is computed once and
    every rung of the ladder reuses it.

    The same factorization delivers ``V_eff``.  Writing ``M``'s inverse in
    arrow form and contracting ``C' M^-1 C`` through it, every level's contrast
    row cancels against its own block and what survives is

        C' M^-1 C = blockdiag(c_q c_q' / m_q) + D' Omega D

    with ``Omega = Sigma_M^-1[1:, 1:]`` — the spline-main corner, because the
    intercept row of ``C - Y' C_cat`` is identically zero.  Subtracting that
    from the block-diagonal ``V`` gives the decomposition in
    :class:`_PairGeometry`.  A level holding no mass contributes an exact zero
    ``D_q``, since its ``c_q`` and ``V_q`` are already zero.
    """
    f = _overlap_arrow(p)
    L, k_a = p.dims
    w_cat, w_border = f.solve(p.u_cat.reshape(L, 1), p.u_border)
    U_eff = p.U - (p.c * (w_border[0] + w_cat.reshape(L))[:, None] + p.V @ w_border[1:])
    mass = np.where(p.m > 0.0, p.m, 1.0)
    D = p.V - (p.c[:, :, None] * p.c[:, None, :]) / mass[:, None, None]
    D += np.swapaxes(D, -1, -2)
    D *= 0.5
    return _PairGeometry(U_eff=U_eff, D=D, Omega=f.Sinv[1:, 1:], ceiling=float(L * k_a))


def _evaluate(p: SplineCatPair, geometry: _PairGeometry, lam: float) -> tuple[float, float]:
    """``(T, edf)`` at one lambda, from ONE arrow factorization.

    ``edf = tr(A^-1 V_eff)`` is the sum of the pencil's Tikhonov filter
    factors.  ``V_eff``'s block-diagonal half contracts against the inverse's
    diagonal blocks; its rank-``k_a`` half needs the OFF-diagonal blocks too,
    and ``[A^-1]_qq' = delta_qq' Ginv_q + Y_q Sinv Y_q''`` reduces the whole
    double sum over levels to one ``(k_a, r)`` accumulator ``Z``.  Nothing
    here loops over ``L`` and nothing here counts.
    """
    L, k_a = p.dims
    f = _pair_arrow(p, lam)
    b = np.zeros((L, k_a + 1), dtype=np.float64)
    b[:, :k_a] = geometry.U_eff
    x, _ = f.solve(b, np.zeros(1 + k_a, dtype=np.float64))
    T = float(np.sum(geometry.U_eff * x[:, :k_a]))

    D = geometry.D
    # ``[A^-1]_qq = Ginv_q + Y_q Sinv Y_q'`` restricted to the tensor
    # coordinates, which is all ``V_eff`` lives on.  Sliced BEFORE the product
    # rather than after: ``diag_blocks()`` would build the full ``(L, g, g)``
    # inverse block and then discard its border row and column.  Measured
    # bit-identical to that route at both bracket edges on the thin-level,
    # vanishing-mass and narrow-band pairs, so this is a temporary removed and
    # not a different number.
    Yt = f.Y[:, :k_a, :]
    diag = f.Ginv[:, :k_a, :k_a] + Yt @ f.Sinv @ np.swapaxes(Yt, -1, -2)
    local = float(np.einsum("lpq,lqp->", diag, D, optimize=True))
    Z = np.einsum("lpq,lqr->pr", D, Yt, optimize=True)
    coupled = np.einsum("lpq,lqs,lst->pt", D, f.Ginv[:, :k_a, :k_a], D, optimize=True)
    coupled += Z @ f.Sinv @ Z.T
    border = float(np.einsum("pq,qp->", geometry.Omega, coupled, optimize=True))
    edf = local - border

    # Every term of the sum is a filter factor ``a_j / (a_j + lambda s_j)``
    # with both parts nonnegative, so the sum lies in ``[0, L * k_a]`` for
    # every lambda and every pair -- a property of the identity, not a
    # tolerance.
    #
    # THAT IS A PROPERTY OF THE EXACT IDENTITY, WHICH IS WHY THIS IS A GUARD
    # AND NOT AN ASSERTION.  The two ``V_eff`` in ``tr(A^-1 V_eff)`` reach
    # float64 by different routes: the one in the numerator is built above
    # from the OVERLAP arrow (``D`` from the level moments, ``Omega`` from
    # that arrow's ``Sinv``), while the one inside ``A^-1`` is whatever the
    # PAIR arrow's own cut leaves of ``V + lambda S - C' M^-1 C``.  They are
    # the same matrix in exact arithmetic and two independent numerical
    # objects here, so the computed sum is not the filter-factor sum of any
    # ONE pencil and the bound can be left.  That mismatch is the structural
    # reason ``_mixed_rank_cells`` reads 0.0 where the oracle reads 1.0 -- the
    # pair arrow drops the direction from ``Ginv`` while ``D`` still carries
    # it -- and it is the most likely source of the mid-bracket error too.
    #
    # Outside the bound the two halves have cancelled away the answer, and
    # accepting the residue can make a ladder search converge to a plausible
    # but wrong row.  The dust allowance is taken on the two halves rather
    # than on the result, because it is their magnitude that bounds the
    # cancellation: at the ladder's low edge on a starved pair they run to 5e5
    # against an answer of 73, and on the ``bs(10)`` pair in the suite to
    # 9.65e7 against a residue of -13.6.
    roundoff = _edf_roundoff(local, border)
    if not np.isfinite(edf) or edf < -roundoff or edf > geometry.ceiling + roundoff:
        raise _UnstableStructuredEDFError(
            f"structured EDF is not a filter-factor sum: {edf} "
            f"(local={local}, border={border}, ceiling={geometry.ceiling})"
        )
    return T, min(max(edf, 0.0), geometry.ceiling)


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
    structured route hand the dense path back.
    """
    if p.profiled_trace is None:
        return None

    # One M factorization for the whole ladder: it carries U_eff and both
    # halves of V_eff, none of which depends on lambda.
    geometry = _profile(p)

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
