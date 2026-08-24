"""Row-space factor kernels shared by BOTH screening paths.

Nothing here is specific to a pair's shape.  What lives in this module is the
arithmetic that both the structured kernel of
:mod:`superglm.screening._structured` and the dense one of
:mod:`superglm.screening._pair_factor` do to a factor rather than to a Gram:
merging two weighted-row factors without squaring either, and rooting an
assembled penalty on a bar the caller can see.

It exists because the dense path needs both and cannot reach
:mod:`superglm.screening._structured`, which already imports ``ScreenedPair``
from :mod:`superglm.screening._score_stat`.  Adding the reverse edge would be
a cycle; a leaf module both can import is not.  The two routines below moved
here unchanged from :mod:`superglm.screening._structured` -- same body, same
docstring, same measurements -- so every number in them was taken on the
structured path and every test that pinned them there still pins them here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _combine_row_factors(left: NDArray, right: NDArray) -> NDArray:
    """Compact two weighted-row factors without squaring either one."""
    return np.linalg.qr(np.concatenate((left, right), axis=0), mode="r")


def _penalty_root(S_a: NDArray) -> tuple[NDArray, float, float]:
    """``rootS`` with ``rootS' rootS`` a PSD matrix WITHIN ``eigh``'s bar of ``S_a``.

    Returns the factor, the spectral-norm distance the DROP branch below moved
    ``S_a``, and the CUT that branch cut on -- so the caller decides on the
    same bar this function decided on rather than on a second one of its own.
    Issue #323 is what happens when it does not.

    **THE SUMMARY LINE USED TO SAY "NEAREST" AND THAT IS THE ONE WORD IT MAY
    NOT SAY.**  The nearest PSD matrix in ANY unitarily invariant norm is
    ``max(w, 0)`` -- Higham, *Linear Algebra Appl.* 103:103-118 (1988) for the
    Frobenius case, and Goulart, Nakatsukasa & Rontsis, "Accuracy of
    approximate projection to the semidefinite cone", arXiv:1908.01606, Lemma
    2.1, for every unitarily invariant norm.  Taking ``|w|`` instead moves
    ``2|w|`` in that direction where clipping moves ``|w|``, so this is a
    PERTURBATION BOUNDED BY ``2 n eps ||S||_2`` and deliberately not a
    projection.  The reason is below and it is a good one, but the two are not
    the same object and the docstring may not claim the cheaper word.

    ``edf`` is a sum of filter factors ``a_j / (a_j + lambda s_j)``, and a
    NEGATIVE ``s_j`` puts that term outside ``[0, 1]``: there is no bound to
    keep and no nonnegative decomposition to have.  Assembled penalties here
    are not all inside the cone.  A difference penalty IS exactly PSD as
    stored -- exact rational LDL certifies exactly ``m`` zero pivots, and the
    ``-1e-15`` an eigensolver reports on it is the eigensolver's own backward
    error -- but the integrated-derivative penalty ``bs`` and ``cr`` margins
    carry is not PSD BY CONSTRUCTION, and ``fl(lambda * S_a)`` leaves the cone
    even for the exactly-PSD one.

    **"NOT PSD BY CONSTRUCTION" IS NOT "NOT PSD AT THE SCALE THIS FUNCTION
    CUTS", AND THE DIFFERENCE MATTERS NOW THAT THE DROP REFUSES.**  An earlier
    draft said those penalties are "genuinely not" in the cone, which read
    literally is the condition the refusal below fires on -- so either every
    ``bs`` pair refuses or the sentence was overstated.  It was the sentence.
    Measured over fifteen margin shapes, ``bs``, ``cr`` and ``ps`` at 5, 8, 12,
    16 and 20 knots, taken through the same route the kernel takes ``S_a``:
    NONE is dropped, SIX carry no negative eigenvalue at all, and the worst
    margin is ``ps(12)`` at 22x INSIDE the bar.  The ``bs`` family -- the one
    the sentence was about -- runs 37x to 1339x inside, because its
    ``||S||_2`` is eight to ten orders larger, so its bar is too.  Their
    departure from the cone is real in exact arithmetic and is not resolvable
    in float64, which is the only sense in which this function may speak of it.

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
    the direction is dropped -- and the largest such magnitude is RETURNED, so
    :func:`_profile` refuses on the cut this function made rather than on a
    second one.  That is the whole of issue #323's fix and the rest of this
    docstring is why it is one cut and not two.

    **THE OLD PROMISE WAS WRONG IN BOTH HALVES.**  It said :func:`_profile`
    "then refuses the pair, because the statistic is still scoring ``S_a``
    raw".

    * **The reason is stale, and issue #298 made it more so.**  The statistic
      and the ``edf`` now come off ONE factorization -- :func:`_evaluate`
      delegates to :func:`_filter_factor_sum`, which stacks
      ``sqrt(lam) * root_penalty`` under each level's rows -- so the statistic
      scores the PROJECTED penalty by construction, and there is no separate
      moment-space arrow left for it to score ``S_a`` raw in.  (An earlier
      draft here named that arrow; it no longer exists.)  The guard's real
      reason is the one :func:`_profile` states at its own site: cross-route
      comparability, because the DENSE path still assembles ``S_ti`` raw.
    * **The refusal was not guaranteed, and the window it left has a closed
      form.**  Dropping happened at ``|c| > n eps ||S||_2``; :func:`_profile`
      raised only at ``clip = |c| / |tr S| > 2 n^2 eps``.  Two thresholds on
      two scales, so ``n eps ||S||_2 < |c| <= 2 n^2 eps |tr S|`` was dropped
      WITHOUT a refusal, and the ratio between them is

          (2 n^2 eps |tr S|) / (n eps ||S||_2)  =  2 n tr(S) / ||S||_2,

      which is ``2 n`` times the INTRINSIC DIMENSION ``tr(S) / ||S||_2`` of the
      penalty -- between ``2n`` and ``2n^2``, and never empty, because
      ``tr S >= ||S||_2`` holds for everything in the cone.  Measured, not
      reasoned: ``n = 10``, spectrum ``[1]*9 + [-1e-14]``, ``||S||_2 = 1``.
      The bar is 2.220e-15 so the eigenvalue is 4.43x to 4.48x outside it,
      depending on the kernel, and dropped -- ``rootS`` came back with 9 rows
      of 10 -- while ``clip`` was 7.89e-16 to 1.18e-15 against a threshold of
      4.441e-14, a factor of 37.5x to 56.25x short.  (Swept over seven
      ``OPENBLAS_CORETYPE`` kernels at 1 and 8 threads; a single run would not
      have been evidence for either number.  ``clip`` here is
      ``sum(rootS**2)`` against ``tr S``, which is what :func:`_profile`
      computes -- an earlier draft of this paragraph summed the EIGENVALUES
      instead and reported a range the code never sees.)  On the public ``freMTPL2freq``
      screen the same ratio measures 85.6x, 157.8x and 160.0x over ten
      structured-route factorizations, matching ``2 n tr(S)/||S||_2`` to four
      figures at ``n = 11`` and ``n = 15``.  A window two orders wide is not a
      seam, so it is closed rather than documented.

    **THE FIX ADDS NO TOLERANCE, WHICH IS WHY IT IS ALLOWED TO BE THIS CHEAP.**
    The drop branch fires exactly when this function has CERTIFIED the
    negativity is not the eigensolver's backward error.  "Was the projection
    roundoff?" is the question :func:`_profile`'s guard asks, and the drop
    branch is definitionally its NO.  So the refusal keys to the drop itself,
    at ``|c| > n eps ||S||_2`` and no other constant.  LAPACK's ``?PSTRF`` --
    Cholesky with complete pivoting for a semidefinite matrix -- is the same
    shape: ONE tolerance, defaulting to ``n u max_k A(k,k)``, decides both the
    ``RANK`` it returns and the ``INFO > 0`` it raises, and its documentation
    declines to distinguish "rank deficient" from "not positive semidefinite"
    at that tolerance because there they are one event.  What this module must
    do differently is split them by SIGN, since clause 2 forbids refusing on
    singularity: ``w = 0`` leaves ``rootS`` through ``keep`` and publishes,
    ``w < -bar`` refuses.

    **AND THE PRICE IS PAID IN THE ONE PROPERTY THIS MODULE OTHERWISE CLAIMS
    OUTRIGHT, SO IT IS STATED RATHER THAN LEFT TO BE FOUND.**  "WHAT IS NOT A
    TRADE: REPRODUCIBILITY" promises the same ``edf`` whatever the BLAS kernel
    or thread count.  A cut AT the eigensolver's own resolution cannot promise
    that about which SIDE it falls on: a penalty within a factor of about one
    of the bar can refuse on one kernel and publish on another.  What moves
    there is a ROUTE -- and, at the non-speculative entry, whether the row
    exists at all -- and not the value of a published number, which is the
    trade clause 2 makes deliberately: a coin flip over a value that no one can
    tell is a coin flip is worse than a coin flip over a route.  It is not free,
    and calling it free would be the same overstatement this docstring has had
    to withdraw twice.  **AND THE NUMBER THAT BELONGS HERE IS THE ONE WITH A
    SPREAD, NOT THE SMALLEST ONE.**  No PIPELINE-BUILT margin penalty comes
    near the cut -- fifteen ``bs`` / ``cr`` / ``ps`` shapes, the closest 22x
    inside -- but that is a spread-free property of the constructors and so
    says nothing about kernel-dependence, which is what this paragraph is
    about.  The measured quantity that HAS that property is the vanishing-mass
    fixture's reconstruction, whose distance to the cut moves 7x across
    kernels: 69.6x inside on SKYLAKEX down to **10.0x on NEHALEM**.  Still an
    order clear, and it is a reconstructed test fixture rather than anything a
    caller builds, but 10.0x-with-a-7x-spread is the honest headroom for a
    claim about which side of the cut something lands on.

    **THE TRACE WAS ALSO THE WRONG NORM, AND THAT IS A PUBLISHED RESULT
    RATHER THAN A PREFERENCE.**  ``|tr S - ||rootS||_F^2|`` is the TRACE-NORM
    distance to the PSD cone, and the projection onto that cone is
    nonexpansive in the Frobenius norm and in no other standard one -- Goulart,
    Nakatsukasa & Rontsis, arXiv:1908.01606, sec. 2.2, which gives
    counterexamples for the spectral and trace norms by name.  The per-direction
    question is a SPECTRAL one (Halmos's distance to the cone is
    ``max{|w| : w < 0}``), so it gets a spectral answer.

    **WHICH WAY THAT WINDOW MOVED ``edf`` IS CLAUSE 4's DIRECTION, NOT CLAUSE
    3's.**  Dropping a direction from ``rootS`` removes it from the PENALTY,
    not from the sum: ``T_q' T_q = D_q + lambda rootS' rootS``, so in that
    direction the filter factor becomes ``d / (d + 0) = 1``.  The direction is
    reported FREE and ``edf`` goes UP by as much as one per level -- the same
    end as "WHAT CLAUSE 4 COSTS", and the OPPOSITE of clause 3's harm, which is
    losing a degree of freedom.  (If ``D_q`` carries no mass there either,
    :func:`_block_inverse_factors` zeroes it and it contributes zero, which is
    clause 1.)  Stating the sign because getting it backwards once already cost
    this docstring a correction.

    The sharper reading is that enlarging ``null(S)`` can MANUFACTURE a common
    null space where none existed -- a penalty projection creating clause-1
    directions is the exact condition this policy is organized around, and is
    more interesting than a deflation rather than less.

    **WHAT THE REFUSAL COSTS, MEASURED BEFORE IT WAS CHOSEN.**  Across the
    twelve pair geometries this suite builds, ZERO factor a penalty with a
    negative eigenvalue outside the bar, so none of them changes.  Across the
    public ``freMTPL2freq`` screen at five ``max_cells`` settings, ten
    structured-route factorizations, again ZERO.  What DID sit in the window
    was one arm of the suite's own guard test, which injected a negative
    direction at a tenth of the TRACE bound -- 8.5x to 8.6x outside the
    eigensolver's bar on every kernel swept -- and asserted it published.  The
    suite pinned the defect; it is re-pinned the other way and a third arm now
    holds the magnitude branch open.

    **AND ON A TWO-COLUMN MARGIN THERE IS NOTHING LEFT TO TAKE THE MAGNITUDE
    OF, WHICH IS CLAUSE 4 OF THE SINGULAR-PENCIL POLICY AT ITS WORST.**
    ``_rank_one_penalty_pair``'s ``S_a`` has ``|sigma_min|`` bracketed EXACTLY,
    by rational arithmetic over the float entries, in ``[3.269e-17,
    6.538e-17]`` -- a fifth of an ulp of ``sigma_max``, so ``eigh`` hands back
    a bit-exact ``0.0`` and the residue is not merely unresolved but absent.
    The direction is then free, and the ladder's high edge publishes
    ``edf = 9.000000`` where a 40-digit oracle on the same exact design reads
    ``5.720073``.  That 3.28 df is the policy, not this routine: the suite's
    textbook stacked-QR reference reads 9.000000 on the same point.  No float64
    penalty factorization recovers it, so nothing here is waiting on a better
    cut.
    """
    n = S_a.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), 0.0, 0.0
    w, Q = np.linalg.eigh(0.5 * (S_a + S_a.T))
    unresolved = float(float(n) * np.finfo(np.float64).eps * float(np.max(np.abs(w), initial=0.0)))
    lifted = np.where(w >= -unresolved, np.abs(w), 0.0)
    keep = lifted > 0.0
    # The spectral-norm distance to the cone contributed by the CERTIFIED
    # negatives only.  A zero eigenvalue -- exact, or positive and inside the
    # bar -- leaves ``rootS`` through ``keep`` and is not counted here: that is
    # clause 1's deflation of a null direction, and clause 2 forbids refusing
    # on it.  Only ``w < -unresolved`` is the eigensolver saying the sign is
    # real.  ``unresolved`` travels out with it so the caller reports the cut
    # this function cut on rather than recomputing ``||S||_2`` by a second
    # routine that would agree only to roundoff.
    dropped = float(np.max(-w, initial=0.0, where=w < -unresolved))
    return (Q[:, keep] * np.sqrt(lifted[keep])).T, dropped, unresolved
