"""Penalized efficient-score statistic for one candidate pair.

The pair arrives as ONE triangular factor of its own weighted joint design
(:mod:`superglm.screening._pair_factor`) and one factor of its tensor penalty
(:func:`superglm.screening._overlap.tensor_penalty_root`).  Frisch-Waugh-Lovell
reads the profiled block off the first as a slice, so

    T = U_eff' (V_eff + lambda0 * S)^{-1} U_eff

with ``V_eff = R_eff' R_eff`` and ``U_eff = R_eff' z_t`` -- never as
``V - C' M^-1 C``, never by inverting an overlap Gram, and never as a
difference of two quantities of the same size.  (The slice is the whole of
``R_eff`` only while the overlap has full rank; where it does not,
:func:`superglm.screening._pair_factor._profiled_factor` stacks back the
directions the overlap never spanned, and everything below reads ITS block --
``tr(V_eff)`` included.)  ``lambda0`` is chosen so the
smooth is compared at a fixed screening complexity:
``tr((V_eff + lambda0 S)^{-1} V_eff) = edf0``.  Fixing the effective degrees of
freedom across pairs makes raw ``T`` values comparable regardless of each
pair's basis size or penalty scaling — at a COMMON budget; across different
budgets compare the normalized ``z`` the ladder scan reports, never raw ``T``.

Ranking-only: calibration is by confirmatory refit, never by this number.

**How lambda0 is found.**  Both quantities the search needs are closed forms in
one generalized singular value decomposition of the PAIR OF FACTORS
``(R_eff, rootS)`` -- the QR-plus-CS construction of :func:`_pair_pencil`.
With ``c`` and ``s`` its cosines and sines and ``u`` the rotated score,

    edf(lambda) = sum_j c_j^2 / (c_j^2 + lambda s_j^2)
    T(lambda)   = sum_j u_j^2 / (c_j^2 + lambda s_j^2)

so every subsequent lambda costs O(k) rather than a fresh O(k^3) solve.  The
decomposition depends on neither ``lambda`` nor ``edf0``, so ONE of them serves
an entire ladder of budgets — which is why ``penalized_score_statistic_ladder``
exists and why callers sweeping a ladder should prefer it.

**ONE ESTIMATOR ANSWERS EVERY RUNG, AND THAT IS ISSUE #257's CLOSURE.**  A
clamped rung used to be answered from a factorization of ``V_eff + lambda S``
and a searching rung from a diagonalization of the same pencil, so two rungs
landing on ONE lambda could publish two different degrees of freedom for it.
Measured on this suite's own fixtures, with budgets placed just inside each
bracket edge::

    _vanishing_mass_pair(1e-12)   16.000004 against 16.999987   1.000 df
    _vanishing_mass_pair(1e-10)   16.000555 against 17.000221   1.000 df
    _starved_bs_pair()             0.999999 against  3.999986   3.000 df
    _starved_bs_pair()            14.605916 against 22.799934   8.194 df

``_thin_level_pair`` at 1.0 and 0.001 did NOT reproduce it, which is why the
pin carries five geometries: the size of the disagreement is a property of how
far apart the two estimators land on a pair.  There is one estimator now and a
rung's lambda decides its answer completely, so the invariant is asserted as
EXACT equality and carries no constant --
``test_the_dense_ladder_reports_one_edf_per_lambda``.

**SCALE DISCIPLINE.  Read this before combining two matrices.**

Four defects on an earlier branch were one mistake: *two quantities combined as
though they were on one scale when they are not*.  By adding, by subtracting,
or by thresholding separately and differencing.

* the original units lesson (e9f7227): an ABSOLUTE whitening cut made the
  statistic depend on the units the curvature was carried in;
* ``G = V + S`` formed in floating point, which loses ``S`` entirely when the
  curvature dwarfs it.  Deriving ``s`` as ``1 - a`` then loses it a second
  time -- and that half was never removed, only made harmless;
* the same ``1 - a`` parameterisation in the test oracle, where it disagreed
  with two algebraically equivalent forms by 2.3e-03 at the ladder's high edge;
* two ranks thresholded independently and then DIFFERENCED, where a probe
  block in large units dominates the joint enough to drop a nuisance direction
  from one count but not the other.

The rule, and it is enforceable rather than advisory:

  1. **A relative threshold is only meaningful on an equilibrated operand.**
     Scale symmetrically by the diagonal first -- a congruence, so rank is
     preserved -- and only then ask whether a direction is small.
  2. **Two quantities may only be added, subtracted or differenced once they
     share a scale.**  Balance before combining.

**BOTH OF THE SITES THIS NOTE USED TO NAME ARE GONE, AND THE RULE IS STRONGER
FOR IT.**  It used to say "there are exactly two such sites": ``_psd_rank``,
the module's only relative-rank threshold, and the ``G`` of ``_build_pencil``,
its only sum of two independently scaled matrices.  Neither exists.

* ``G = V + S`` is not formed at all.  The pencil STACKS ``R_eff`` under
  ``rootS`` and reduces, so there is no floating-point sum for the smaller term
  to be lost in: with ``V = 1e20 I`` against ``S = I`` the stack keeps the
  penalty by construction rather than by balancing.  A balance is still
  applied, and it is now an ACCURACY measure rather than a representability
  one -- each block's singular values are resolved to ``eps`` ABSOLUTELY, so a
  block whose norm is 1e-10 of the other would be resolved to six figures
  instead of sixteen.  See :func:`_pair_pencil`.
* the ``1 - a`` subtraction is gone with it.  ``c^2 + s^2 = 1`` now holds by
  ORTHONORMALITY of one factor rather than by construction, so it is a
  derivable invariant instead of a tolerated site: measured over this suite's
  eight-fixture bank, ``max |c^2 + s^2 - 1|`` is 2.109e-14 at one thread and
  1.310e-14 at eight, against ``k eps`` of 4.641e-14 -- inside its own derived
  bar by 2.2x at worst.  Pinned in
  ``test_the_pencil_carries_its_own_orthonormality_invariant``.
* the surviving relative cuts are :func:`_factor_rank_floor`'s, and they are
  taken on FACTORS.  The rank of a profiled block is still balanced before it
  is counted, for exactly rule 1's reason, and
  ``test_screening_is_invariant_to_the_units_of_a_numeric_margin`` still
  enforces it end to end.

**THRESHOLD TYPES.  Read this before adding a constant to this module.**

Every cut here answers one of two questions, and they have different standing.

*Type 1 -- "is this arithmetic meaningless?"*  A statement about floating
point, DERIVABLE from backward stability as a function of machine epsilon and
dimension.  It holds for every input because the bound covers all of them, so
there is no unmeasured regime waiting to break it.  :func:`_rank_floor` is of
this kind: ``max(n, 1) * eps``, LAPACK's convention and ``matrix_rank``'s own
tolerance, and :func:`_factor_rank_floor` is its square root, which is the same
statement about a factor.  So is ``_solve_floor`` in
:mod:`superglm.screening._arrow`.

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
A dimension-scaled cut therefore cannot separate absorption from weak
identification, and the fitted one that was tried both deleted a legitimately
weak block (``V = (1 + 1e-4) I``, whose ``V_eff`` is ``1e-4 I`` and full rank)
and failed to fire on a genuinely absorbed one at the same ``k``.

So absorption is NOT detected and such a pair is NOT discarded.  It is scored:
the unpenalized rung counts the rank of the profiled FACTOR against the joint
design's own scale, which on an absorbed block is 0, the statistic comes out at
round-off, and ``z`` puts the pair near the bottom where it belongs.

**WHERE THAT RANK IS DECIDED, AND WHY NOT ON THE PROFILED BLOCK ITSELF.**  The
unpenalized rung's ``edf`` IS ``rank(V_eff)``, and the reference it is counted
against cannot be ``R_eff``'s own largest direction: on a block the overlap has
absorbed, ``R_eff`` is round-off in its entirety, so a relative cut there is
taken against the noise it must reject.  Measured on the reachable path --
5-level wholly absorbed ``numeric_cat`` pairs screened end to end, 20 seeds --
that reference reports a nonzero ``edf0`` on 20 of 20.  ``edf`` too HIGH is not
a neutral failure: ``z = (T - e) / sqrt(2 e)`` DECREASES in ``e``, so a
partly-rejected block scores higher than an unrejected one.

The reference is therefore the JOINT design's, where nothing has been
residualized away -- :func:`superglm.screening._pair_factor.
_profiled_rank_scale`.  That is the factor-space form of the Guttman argument
the moment route used, ``rank([[V, C'], [C, M]]) - rank(M)``, which counted
both operands against the joint's scale rather than against the difference's;
here it is one count rather than two, because the factor carries the difference
as a block instead of leaving it to be formed.  Measured 0 of 20 on the same
reachable path.  **The precondition that argument rested on -- non-negative
working weights, so the joint moment matrix is PSD -- is no longer needed:
nothing is differenced, so nothing has to be PSD for the count to mean what it
says.**

**THE ACCURACY CEILING WAS ARCHITECTURAL AND THIS MODULE NO LONGER SITS UNDER
IT.**

The ceiling was this: the module was handed MOMENTS, ``V_eff`` arrived as a
Gram, so its spectrum was the design's SQUARED, and on a pair with a starved
level the smallest directions fell under the noise floor of the operator they
were read from with no correct digits left in them.  ``cond(V_eff)`` on the
starved pair is 2.78e+20 against a representable ``1 / eps`` of 4.5e+15, where
``sqrt(cond)`` is 1.67e+10 -- comfortably inside float64.  Reading the factor
is what puts the deciding direction back in range, and it is the same move
:mod:`superglm.screening._structured` made in #285 and #322.

**WHAT IT BOUGHT, MEASURED ON THIS BRANCH AGAINST ORACLES THAT PREDATE IT.**
The moment route is reconstructed for the comparison -- ``V - C' M^-1 C`` with
the cho-or-pinv solve, the two bracket edges from a factorization's trace, and
a searching rung from the balanced congruence -- and handed the same inputs, so
what separates the columns is the estimator and not the data.

At the ladder's LOW edge, against ``_CERTIFIED_EDGES``, which is 640-bit arb
ball arithmetic on the pair's exact design::

    _thin_level_pair    moment route      factor route
    1.0                 1.3662e-10        0.0000e+00
    0.01                2.4045e-09        0.0000e+00
    0.001               3.2264e-09        2.8422e-14

Two of the three are BIT-IDENTICAL to the certified value and the third is one
ulp from it.  That edge is where the whole-degree-of-freedom defect this
module's oldest tests were written for lived.

At the HIGH edge the two routes trade, and the trade is stated rather than
averaged.  On ``_vanishing_mass_pair`` the factor route reproduces the
independent stacked-QR oracle to the printed digits where the moment route is
5.004e-04 (at 1e-10) and 7.600e-05 (at 1e-12) away from it.  On
``_thin_level_pair``, against the same arb-certified constants::

    _thin_level_pair    moment route      factor route / structured route
    1.0                 1.6929e-05        8.1497e-05
    0.01                2.9110e-05        2.7668e-04
    0.001               4.0894e-05        2.1075e-03

**That column is not a pencil error and no rearrangement here narrows it: it is
the PENALTY ROOT, and it is shared with the structured path deliberately.**
``S_a``'s smallest eigenvalue on this fixture reads 1.4476e-15 against an
``eigh`` bar of 3.7592e-14, so it is 26x INSIDE what the eigensolver resolves;
the certified value carries the exact residue and any float64 route carries
the eigensolver's reading of it instead.  At ``lambda = 1e10 * scale`` that
difference is what the table shows.  The moment route escaped it only by never
rooting the penalty at all -- it carried ``S_ti`` raw inside ``V + balance S``
-- and the price of that was a second policy: issue #323 is what happens when
two parts of this package decide the same question on different bars.  What the
shared root buys is exact: the dense and structured arms' high-edge ``edf``
now agree to 3.7e-11, 2.0e-10 and 1.4e-12 on the three weights, where they
differed by 9.8e-05, 2.5e-04 and 2.1e-03.  The suite asserts both against the
certified constant at ``abs=1e-2``, which the worst reading clears by 4.7x.

**AND THE PENCIL ITSELF IS EXACT WHERE AN ORACLE CAN SEE IT.**  On
``moderate_pair``'s high edge, whose condition number is ~1e11, against the
textbook stacked-QR arbiter: the statistic moves 1.1159e-05 -> 3.451e-13 and
``edf`` 4.3374e-06 -> 1.959e-12, four to six orders, and the per-lambda stacked
QR on the same factor agrees with the pencil to 2.0e-13 on the statistic and
1.1e-15 on ``edf``, so ONE lambda-free
decomposition costs nothing against a fresh factorization per rung.

**THE RANK CUT IS NO LONGER A CONVENTION, AND THAT IS THE MEASUREMENT THIS
SECTION EXISTS FOR.**  The moment route's disagreement with the arrow kernel
was defensible as a CONVENTION: an independent evaluation's rank cut, swept,
gave several plateaus, and nothing in the data chose between them.  Re-measured
on this branch -- ``_thin_level_pair(1e-12)`` at the ladder's high edge,
whitening ``V_eff`` at a relative cut for the Gram column and truncating
``R_eff`` at that cut's square root for the factor column, then evaluating by
stacked QR::

    rank cut          Gram column     factor column
    1e-18 .. 1e-6     17.99991        17.99991

Thirteen decades, ONE value, on both columns.  The earlier three-plateau table
is DELETED rather than carried: it was a measurement of a construction this
branch does not reproduce, and re-stating a number nobody can re-take is the
defect #324 was about.  What is asserted instead is the regime the change was
made for --
``test_the_dense_path_s_ceiling_is_its_gram_and_not_its_arithmetic`` still
COUNTS the directions ``V_eff``'s Gram cannot resolve, four on the starved pair
against one elsewhere, and that count is what says the Gram route was reading
absent information.

**REPRODUCIBILITY, WHICH IS WHAT ACTUALLY REACHED A USER.**  On the guide's
published twelve-row ``freMTPL2freq`` screen -- an 80,000-row sample under
``weight_semantics="frequency"``, the specification ``docs/guide/screening.md``
prints and ``tests/fixtures/screening_guide_fremtpl.json`` pins, ``phi``
4.821136 -- one thread against eight with all six pools pinned together::

                          worst |dz|      worst |d edf0|
    moment route          4.5597e-05      7.7570e-04
    factor route          1.1252e-13      7.0699e-13

Eight orders, and the table order is identical on both routes and the
``z > sqrt(edf0 / 2)`` gate admits the same SINGLE pair -- ``VehAge x
BonusMalus`` -- on every one of the four runs.  On this suite's own fixtures
the same comparison gives a worst ``edf`` move of 5.58e-13 across 1 and 8
threads, against the 4.05e-04 :mod:`superglm.screening._structured` recorded
for the dense arm on ``_thin_level_pair``.

**WHAT THE CHANGE MOVES IN THE PUBLISHED TABLE**, one thread, the same screen,
one route against the other: the order is identical, the gate admits the same
single pair, and the worst ``|dz|`` is 3.6886e-05 on ``VehAge x VehBrand``,
followed by 8.97e-06, 4.05e-06, 6.54e-07, 3.23e-07 and 1.74e-07.  Every ``ti``,
``cat_cat`` and ``numeric_cat`` row moves at round-off (2.46e-12 or below).  So
the change is visible only on ``spline_cat`` pairs and only in the fifth
decimal of ``z``, which is the size #257 said it would be.

**AND THE PUBLISHED TABLE'S OWN PINS WERE RE-TAKEN RATHER THAN ASSUMED**, since
a fifth-decimal move against a ``rel=1e-4, abs=1e-4`` pin is only 2.7x of the
bound and that is not a headroom anyone should infer.  Against the committed
fixture, at one thread, the shipped factor route uses 0.180 of the ``z`` bound
(5.6x inside, worst on ``VehAge x Region``), 0.191 of the ``statistic`` bound,
0.070 of the ``edf0`` bound and 0.00037 of the ``lambda0`` bound.  The moment
route the fixture was generated with uses 0.215, 0.296, 0.063 and 0.00037 of
the same four.  So the change does not consume the fixture's headroom; on ``z``
it widens it, because the moment route's own thread-to-thread spread on this
screen is larger than the distance between the two routes.

**AN EARLIER REVISION OF THIS SECTION QUOTED A DIFFERENT SCREEN, AND SAID SO
IN THE PUBLISHED TABLE'S NAME.**  It reported these tables on "100,000 rows,
the guide's own specification" and said the gate admitted FOUR pairs.  The
guide's screen is 80,000 rows AND carries ``weight_semantics="frequency"``,
which sets ``phi`` and therefore every ``z``; on it the gate admits one.  The
numbers above are the published screen's.

**THE LOW EDGE HAS ITS OWN LIMIT AND IT IS A FIRST-ORDER SENSITIVITY,
``eps ||V_eff||_F ||G||_F``.**  (Not a ceiling: finite realizations exceed it,
measured at up to 1.0617x.)  The section above is about the ladder's HIGH edge;
the low edge fails differently and the two must not be conflated.

At the low edge nothing is unresolved in the sense counted above.  Over 21
draws of the 20-level spline-by-categorical geometry, the operator
``V_eff + lambda_lo S`` has ZERO directions within ``10x eps ||A||`` on every
one, the nearest clearing that cut by at least 1879x.  The mechanism is the
derivative instead of the noise floor: ``edf = tr(A^-1 V_eff)`` with
``A = V_eff + lambda S`` is exactly ``k - lambda tr(A^-1 S)``, so for any
perturbation ``E`` of the operand::

    d edf = <E, G>_F ,      G := lambda A^-1 S A^-1 + alpha (d edf / d lambda) I

**FOR NONSINGULAR ``A`` ONLY.**  The identity above inverts ``A``, so it does
not cover a pair whose ``V_eff`` and ``S`` share a null space -- a supported
case, which the pencil answers by discarding that common null space, and on
which ``k - lambda tr(A^-1 S)`` is not evaluable.  Everything in this section is
about the low edge of a positive definite pencil.

The second term is the BRACKET's own response: ``lambda = alpha tr(V_eff)``
with ``alpha = 1e-10 / tr(S)``, so a probe with nonzero trace moves ``lambda``
too.  It is part of the gradient, not an afterthought.

Over ``||E||_F = c`` the differential is maximised at ``E = c G / ||G||_F``,
giving exactly ``c ||G||_F``.  Take ``c = eps ||V_eff||_F`` -- ONE ROUNDING of
the operand's norm -- and the low edge's first-order sensitivity is
``eps ||V_eff||_F ||G||_F``.  That maximisation is exact for the DIFFERENTIAL
and is not a bound on the finite response, since ``edf(V + E)`` is nonlinear in
``E``; measured realizations reach 1.0617 of it.

**THE RADIUS IS A STATED PROBE AND NOTHING MORE.**  It is not "smaller than the
error already committed in forming the Gram" -- that asserted a backward-error
bound nobody had derived -- and it is not established to be smaller either.
Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Ch. 3 gives
``fl(X'X) = X'X + D`` with ``|D| <= gamma_n |X|'|X|`` COMPONENTWISE, measured
here at 98.6x to 204.7x the probe radius -- but that is an UPPER bound, and an
upper bound puts no floor under the actual error.  So what follows is the
answer's SENSITIVITY to a one-rounding probe.

**THE DERIVATION IS BASIS-FREE AND SURVIVES THE MOVE; THE PROBE'S RADIUS WAS
ONE ROUNDING OF A GRAM AND IS NOW ONE ROUNDING OF A FACTOR'S GRAM.**  The
sensitivity is a property of the QUANTITY ``tr((V_eff + lambda S)^-1 V_eff)``
and of the pair the caller supplies, not of how this module reads it, so it is
unchanged -- and
``test_the_low_edge_edf_is_only_as_determined_as_the_gram_it_is_read_from``
drives the shipped ladder through the factor route and still separates its
resolved geometries from its unresolved ones.  What DID change is where the
perturbation enters: the caller now hands a factor, so a rounding of the
operand is a rounding of ``R_eff``, and the same displacement reaches ``edf``
through a spectrum that has not been squared.

**WHAT THE SENSITIVITY IS ON THIS FAMILY, PER DRAW RATHER THAN AS A
CONSTANT.**  Over the 21 draws it spans **1.6298e-15 to 4.9265e-04** -- twelve
orders, tracking the residualized design's smallest singular value.  All ten
draws whose design is rank deficient sit at ~2.6e-04; the well-conditioned ones
sit at 1e-15 to 3e-09.  So on exactly the draws that can fail it, the answer's
sensitivity is ~26x the ``abs=1e-5`` the suite asserts, which is the whole of
issue #279: which draw is used decides whether it passes.  Do not carry the
number to another fixture -- carry the derivation.

Which draws are bad is decided in the DESIGN, before this module is called: the
bad ones are those where a level's rows put the constant vector in the span of
that level's own spline columns.  ``1 / lambda`` then multiplies whatever is
left there.  The test asserts the sensitivity against ONE ULP of the answer --
``np.spacing``, the real adjacent-float distance -- with worst asserted-resolved
4.0956e-08 ulp against best asserted-unresolved 3.0293e+10 ulp, so the boundary
clears by 2.44e+07x and 3.03e+10x.  Separately the maximising perturbation
attains 0.9425 to 1.0617 of the predicted displacement against a first-order
value of 1; that ratio is REPORTED and not asserted.

**WHICH OF THOSE FIGURES ISSUE #257 COULD HAVE MOVED, AND WHICH IT COULD NOT.**
The SENSITIVITY itself -- the 1.6298e-15..4.9265e-04 span -- is
``eps ||V_eff||_F ||G||_F``, computed from the pair's own ``V_eff`` and the
operator at the low edge, so it is a property of the QUANTITY and the same
number whichever route reads it.  The two figures that involve a realized
displacement -- the ulp separation and the 0.9425..1.0617 attainment ratio --
were taken before the move and are NOT re-taken here, because what they
support is a boundary the test recomputes on every run rather than a constant
it carries.  That test drives the shipped ladder, so the boundary it asserts is
the factor route's; the numbers beside it are the moment route's record of how
wide the gap was, and are labelled as such rather than restated as current.

**A SAMPLED WIDTH WAS TRIED FIRST AND IS NOT SOUND**, which is worth recording
because it looks convincing.  Taking ``max - min`` of ``edf`` over 32 random
perturbations and requiring it to be LARGE asserts a lower bound on a sample
range, where the theory supplies only an upper bound on sensitivity.  It moves
with both of its own arbitrary constants: 0.2548 at 4 draws against 0.8653 at
128, and 0.5063 to 0.8085 over 24 perturbation seeds at one configuration.  A
deterministic norm has neither axis.

**WHERE THIS SITS IN THE LITERATURE, AND IT IS NOW THE JUSTIFICATION RATHER
THAN A DISCLOSURE.**  ``edf`` is the trace of the influence matrix of a
general-form Tikhonov problem -- equivalently the sum of filter factors over
the generalized singular values of the pair ``(X_w, L)`` with ``L' L = S``,
which is literally what :func:`_pair_pencil` computes.  Every standard
algorithm for it takes the DESIGN or a backward-stable factor of it, never the
Gram: Elden, *BIT* 17 (1977) 134-145 and *BIT* 24 (1984) 467-472; Golub, Heath
and Wahba, *Technometrics* 21 (1979) 215-223; Hutchinson and de Hoog, *Numer.
Math.* 47 (1985) 99-106; Wood, *JRSS-B* 70 (2008) 495-518 sec. 3.2, which gives
exactly the stacked-QR form ``[W X; E] = Q R`` with ``A = K K'`` and calls the
Cholesky-of-``X'W^2X`` alternative the less stable of the two for "the
exacerbation of any numerical ill-conditioning that accompanies explicit
formation of ``X'W^2X``".  The moment route was published, and published as the
route the standard method exists to avoid.

The GSVD itself is Van Loan, *SIAM J. Numer. Anal.* 13(1):76-83 (1976);
computing it through the CS decomposition is Stewart, *Numer. Math.*
40:297-306 (1983) and Van Loan, *SIAM J. Numer. Anal.* 22(3):579-592 (1985),
with the modern arrangement in Bai and Demmel, *SIAM J. Sci. Comput.*
14(6):1464-1486 (1993).  The *LAPACK Users' Guide* (3rd ed., SIAM 1999,
sec. 4.7) recommends Cholesky-plus-GSVD over the generalized symmetric definite
driver when the second matrix is ill conditioned, and an earlier revision of
this module recorded that recommendation as UNREACHABLE because ``dggsvd3``,
``dggsvp3`` and ``dtgsja`` are absent from ``scipy.linalg.lapack``.  **That was
half wrong and the half matters**: those three really are absent at 1.18.0, but
the construction needs only a pivoted QR and one SVD, both of which are there.
The DRIVER was missing, not the method.

Two halves of the mechanism are settled and one is not.

* SQUARING.  Conventional eigen/SVD drivers are accurate ABSOLUTELY, to
  ``eps sigma_1``, so directions at or below ``eps sigma_1`` carry no relative
  accuracy (Demmel, *Accurate SVDs of Structured Matrices*, LAPACK Working Note
  130, 1997; the *LAPACK Users' Guide*'s own SVD error bound).  Forming a Gram
  squares ``sigma_1``.  Huang and Jia, arXiv:1907.10392, reject the
  cross-product pencil ``(A'A, B'B)`` for the GSVD on precisely this ground --
  small generalized singular values from it "may be recovered much less
  accurately and even may have no accuracy" -- which is one algebraic step from
  the quantity summed here, and is the pencil this module used to form.
* AMPLIFICATION.  ``1 / (lambda s_j)`` is elementary calculus, above.
* THE COMPOSITION IS OURS.  A sweep of the GCV/edf and inverse-problems
  literature found NO published forward-error bound for
  ``tr((A + lambda S)^-1 A)`` under perturbation of ``A``, and no method that
  recovers it accurately from a Gram alone.  The nearest published relatives
  bound the generalized singular values themselves (Huang and Jia, sec. 2.1,
  improving Stewart and Sun, *Matrix Perturbation Theory*, 1990) or the
  Tikhonov SOLUTION, not this trace.  Stated as ours rather than cited.

AND DO NOT REACH BACK FOR THE STANDARD RESCUES ON THE MOMENTS; they were
checked and each needed what the moment route did not have.  Seminormal
equations and its corrected form, and iterative refinement on the normal
equations, all require ``A`` itself or an ``R`` that is the factor of a
backward-stable decomposition of it (Bjorck, *BIT* 7 (1967) 257-278 and *LAA*
88 (1987) 31-48) -- which is exactly what this module is now handed, so the
barrier they ran into is the one that has been removed.  The
high-relative-accuracy theory needs a rank-revealing decomposition whose
factors are themselves accurate (Demmel and Veselic, *SIMAX* 13 (1992)
1204-1245; Demmel et al., *LAA* 299 (1999) 21-80), and a Gram formed in
floating point is not one.

**WHAT IS NOT CLAIMED HERE: COST.**  The reduction that produces the factor is
``O(n_outer * k_inner * w^2)`` where the moment assembly was
``O(n_outer * k^2)``, so it costs a factor of the narrower margin's width, and
the pencil is one pivoted QR plus one SVD where the moment route ran a
generalized eigendecomposition.  No timing was taken on this branch and none is
stated: the budget gates in :mod:`superglm.model.screening_ops` are DIMENSIONAL
(``k**3 <= _CUBIC_BUDGET_FACTOR * max_cells``), so which pairs are admitted is
unchanged and no published ceiling moves, but their ~1.5 s per-pair calibration
was fitted against the moment route and re-fitting it is a separate decision
taken with a machine to itself.
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
    """Ranking output for one candidate pair."""

    statistic: float
    edf0: float
    lambda0: float


@dataclass(frozen=True)
class _Pencil:
    """The pair ``(V_eff, S)`` diagonalized once, with ``U_eff`` rotated in.

    ``v`` and ``s`` are the two transformed diagonal terms and ``u`` the
    rotated score, so a rung costs no arithmetic beyond
    ``edf(lam) = sum v / (v + lam s)`` and ``T(lam) = sum u^2 / (v + lam s)``.

    **NEITHER TERM IS DERIVED FROM THE OTHER, AND THAT IS NEW.**  They are the
    squared cosines and sines of the CS decomposition of one orthonormal
    factor, taken from two independent singular value decompositions -- so
    ``v + s == 1`` holds by orthonormality of that factor rather than being
    imposed, and there is no ``1 - v`` anywhere for a direction with ``v``
    rounding to 1 to fall through.  The form this replaces returned
    ``s = (1 - share) / balance`` and had to argue, from measurement, that the
    subtraction happened to be harmless; SCALE DISCIPLINE rule 2's one
    tolerated site is closed rather than re-argued.

    ``v`` is ``c**2`` and ``s`` is ``s**2 / balance``, the balance being the
    scaling that put the two blocks of the stack on one scale before they were
    reduced together.  Undoing it is one division on a quantity that never
    cancelled.

    ``tr_v`` is ``tr(V_eff)`` of the block this pencil was built from, carried
    out rather than recomputed by the caller.  It is the bracket's numerator
    AND this pencil's own balance, and computing it twice is exactly how the
    two came to disagree: the balance read :func:`_profiled_factor`'s block and
    the bracket re-sliced ``joint``, which is a different matrix whenever the
    overlap is rank deficient.  One field, one value, one place it is formed.
    """

    v: NDArray
    s: NDArray
    u: NDArray
    tr_v: float


def _empty_pencil(tr_v: float = 0.0) -> _Pencil:
    return _Pencil(v=np.zeros(0), s=np.zeros(0), u=np.zeros(0), tr_v=float(tr_v))


def _pair_pencil(pair: PairFactor, penalty_root: NDArray | None) -> _Pencil:
    """Diagonalize ``(V_eff, S)`` from the pair's FACTOR, never from its Gram.

    This is the generalized singular value decomposition of the pair
    ``(R_eff, rootS)``, computed the way the standard references construct it
    -- a QR of the stack followed by the CS decomposition of the resulting
    orthonormal factor.  Van Loan, *SIAM J. Numer. Anal.* 13(1):76-83 (1976)
    for the GSVD itself; Stewart, *Numer. Math.* 40:297-306 (1983) and Van
    Loan, *SIAM J. Numer. Anal.* 22(3):579-592 (1985) for computing it through
    the CSD; Bai & Demmel, *SIAM J. Sci. Comput.* 14(6):1464-1486 (1993) for
    the modern arrangement.  It is what the *LAPACK Users' Guide* (3rd ed.,
    SIAM 1999, sec. 4.7) recommends over the generalized symmetric-definite
    driver when the second matrix is ill conditioned, and what this module's
    refused-remedy 4 used to record as unreachable.  **That record was half
    wrong**: ``dggsvd3``, ``dggsvp3`` and ``dtgsja`` really are absent from
    ``scipy.linalg.lapack`` (re-verified at 1.18.0), but the construction needs
    only a pivoted QR and one SVD per block, both of which are present.  The
    DRIVER was missing, not the method.

    With ``[R_eff ; rootS] = Q Rg``, ``Q = [Q1 ; Q2]``, and ``Q1 = Uc C W'``,
    ``Q2 = Us S W'`` sharing their right singular vectors,

        V_eff = Y C^2 Y' ,   S = Y S^2 Y' ,   Y = Rg' W ,

    so with ``u = C Uc' z_t`` every rung is a sum of Tikhonov filter factors:

        edf(lam) = sum_j c_j^2 / (c_j^2 + lam s_j^2)
        T(lam)   = sum_j u_j^2 / (c_j^2 + lam s_j^2) .

    Every term of the first is in ``[0, 1]`` by construction, no term of
    either is a difference, and both are exact where ``Y`` is rank deficient:
    ``Y`` has full column rank after the truncation below, so ``Y^+ Y = I``
    and the pseudo-inverse of the operator collapses out of both traces.  ONE
    decomposition serves an entire ladder, which is what makes a rung O(k).

    **BALANCE BEFORE STACKING, FOR ACCURACY RATHER THAN FOR REPRESENTABILITY.**
    The form this replaces had to balance because it FORMED ``V + S`` and lost
    the smaller term outright when the two lived on far-apart scales -- ``V = 1e20 I`` against ``S = I`` rounds to ``V``, every share comes
    back 1, and the ladder reports the full dimension at every lambda.  Stacking
    adds nothing, so that failure is gone by construction and the balance is no
    longer load-bearing for it.  It is kept for a different and smaller reason:
    the SVD of each block is accurate to ``eps`` ABSOLUTELY, so a block whose
    norm is ``1e-10`` of the other has its singular values resolved to six
    figures rather than sixteen.  Scaling ``rootS`` by ``sqrt(tr(V_eff) /
    tr(S))`` puts both blocks at O(1) and the division that undoes it is one
    rounding on a quantity nothing cancelled.

    ``penalty_root = None`` is the unpenalized block: the stack is ``R_eff``
    alone, every ``s_j`` is exactly zero and every surviving ``c_j`` gives a
    filter factor of exactly 1, so ``edf`` comes back as the retained RANK and
    ``T`` as the squared norm of ``z_t``'s projection onto the directions the
    design actually resolves.  It is the same estimator as every other rung
    rather than a second one beside it, which is why there is no separate
    counting routine here any more.
    """
    R_eff, z_t = _profiled_factor(pair)
    # ``tr(V_eff)`` ONCE, off the block this pencil is about to score.  It
    # leaves on ``_Pencil.tr_v`` because the ladder's bracket needs the same
    # number, and taking it there from the factor's own slice is what put the
    # bracket 18.06% below the block on a rank-deficient overlap.
    tr_v = _pair_scale(R_eff)
    k = int(pair.tensor_width)
    balance = 1.0
    if penalty_root is None or penalty_root.size == 0:
        root = np.zeros((0, k), dtype=np.float64)
    else:
        root = np.asarray(penalty_root, dtype=np.float64)
        tr_s = float(np.sum(root**2))
        if tr_v > 0.0 and tr_s > 0.0:
            balance = tr_v / tr_s
        root = np.sqrt(balance) * root
    if k == 0:
        return _empty_pencil(tr_v)

    stack = np.concatenate((R_eff, root), axis=0)
    orthonormal, triangular, _pivot = scipy.linalg.qr(
        stack, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0 or float(diagonal[0]) <= np.finfo(np.float64).tiny:
        return _empty_pencil(tr_v)
    # TWO DIFFERENT QUESTIONS, AND THEY GET TWO DIFFERENT REFERENCES.
    #
    # With a penalty, the cut asks what neither operand resolves: a direction
    # the STACK does not resolve carries neither curvature nor penalty, so it
    # contributes zero to both sums and discarding it is exact.  Its reference
    # is the stack's own largest direction -- the same policy the whitening
    # branch this replaces applied to ``G``'s spectrum, at the same cut and one
    # square root earlier.  No rank decision beyond that is taken on a
    # penalized rung: a direction with little curvature and real penalty gets a
    # filter factor near zero and contributes what it should, so there is
    # nothing here for a threshold to arbitrate.
    #
    # Without one, the cut IS the answer -- ``edf`` is ``rank(V_eff)`` -- and
    # the stack is ``R_eff`` alone, whose own top is round-off on exactly the
    # blocks that matter.  :func:`_profiled_rank_scale` supplies the joint
    # design's scale instead, which is where nothing has been residualized
    # away.
    cut = (
        _factor_rank_floor(k) * float(diagonal[0])
        if root.size
        else _factor_rank_floor(k + int(pair.overlap_width)) * _profiled_rank_scale(pair)
    )
    rank = int(np.count_nonzero(diagonal > cut))
    if rank == 0:
        return _empty_pencil(tr_v)

    top = orthonormal[: R_eff.shape[0], :rank]
    bottom = orthonormal[R_eff.shape[0] :, :rank]
    # THE COMMON BASIS COMES FROM THE PENALTY BLOCK, AND WHICH BLOCK IT COMES
    # FROM IS THE WHOLE OF THE CONSTRUCTION'S ACCURACY.
    #
    # ``Q'Q = I`` gives ``c_j**2 + s_j**2 = 1`` for any orthonormal basis ``W``
    # of the ``rank``-dimensional row space, so ONE decomposition settles both
    # terms: take ``W`` from one block and read the other off ``|| . W ||``.
    # There is no cancellation either way -- what differs is WHICH clusters are
    # ambiguous, and only one of the two choices is safe.
    #
    # A singular value decomposition determines its singular vectors only up to
    # a rotation inside a cluster of equal singular values.  Here ``s`` decides
    # the filter factor outright -- ``f = (1 - s**2) / (1 - s**2 + lam s**2)``
    # is a function of ``s`` alone -- so directions with near-equal ``s`` have
    # near-equal ``f``, and a rotation among them leaves both ``sum f`` and
    # ``sum proj * f`` invariant.  A cluster in ``c`` carries no such promise:
    # near ``c = 1``, ``s = sqrt(1 - c**2)`` is exactly the quantity float64
    # cannot recover, so directions whose ``c`` are bit-identical can have
    # ``s`` an order apart and ``f`` with them.
    #
    # Measured on ``moderate_pair``'s high edge against the stacked-QR arbiter,
    # where 23 directions come back with ``c**2`` rounding to exactly 1.0 while
    # their ``s**2`` spread from 1.4e-16 to 2.1e-15: taking ``W`` from ``Q1``
    # puts the statistic 3.89e-06 out and this puts it 5.4e-13 out, with
    # ``edf`` 1.96e-12 either way.  An independent ``svd`` of each block was
    # tried first and is worse still (5.47e-06), because it does not even pair
    # the two consistently.  Stewart, *Numer. Math.* 40:297-306 (1983) and Van
    # Loan, *SIAM J. Numer. Anal.* 22(3):579-592 (1985) construct the CS
    # decomposition from one block and transform the other for the same reason.
    if bottom.size:
        # ``full_matrices`` only where the penalty root is SHORTER than the
        # pencil: there ``Q2`` has a genuine null space and its basis has to be
        # completed, and the singular values it cannot carry are the exact
        # zeros that belong at the end of a descending list.
        _, values, basis = np.linalg.svd(bottom, full_matrices=bottom.shape[0] < rank)
        sines = np.concatenate((values, np.zeros(rank - values.size)))[:rank]
        carried = top @ basis.T
    else:
        sines = np.zeros(rank, dtype=np.float64)
        carried = top
    cosines = np.linalg.norm(carried, axis=0)
    return _Pencil(
        v=np.clip(cosines**2, 0.0, 1.0),
        s=np.clip(sines**2, 0.0, 1.0) / balance,
        u=carried.T @ z_t,
        tr_v=tr_v,
    )


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
    pair: PairFactor,
    penalty_root: NDArray | None,
    *,
    budgets: tuple[float, ...] = (4.0,),
) -> list[ScreenedPair]:
    """Score one pair at every budget in ``budgets``, sharing one decomposition.

    Equivalent to calling :func:`penalized_score_statistic` once per budget,
    but the pencil that makes ``edf`` and ``T`` closed forms depends on neither
    ``lambda`` nor ``edf0`` — so an entire ladder costs one decomposition
    instead of one per rung, each of which previously also paid for its own
    bisection.

    **BOTH ARGUMENTS ARE FACTORS, AND BOTH ARE REQUIRED.**  ``pair`` is the
    pair's whole weighted joint design reduced to one triangular factor by
    :mod:`superglm.screening._pair_factor`; ``penalty_root`` is a factor of the
    tensor penalty, ``rootS' rootS = S_ti``, from
    :func:`superglm.screening._overlap.tensor_penalty_root`, or ``None`` for an
    unpenalized block.  Neither is the ``(k, k)`` Gram the moment route passed,
    and that is the whole of issue #257: a Gram's spectrum is its factor's
    SQUARED, so a direction the answer depends on can sit five orders past what
    float64 resolves in one and comfortably inside it in the other.  The
    penalty is a factor for the same reason and it is not a cosmetic
    symmetry -- rooting the ASSEMBLED ``kron(S_a, I)`` rather than the margin
    it is built from costs six orders at the ladder's high edge, measured in
    :func:`superglm.screening._overlap.tensor_penalty_root`.  A caller still
    holding the five moment matrices gets a ``TypeError`` here rather than a
    plausible answer.  ROOTING IS STILL ONE POLICY,
    :func:`superglm.screening._factor_kernels._penalty_root`'s, applied to the
    same margin penalties the structured path applies it to; issue #323 is what
    happened when there were two.

    **ONE ESTIMATOR ANSWERS EVERY RUNG.**  A budget outside the bracket clamps
    to the nearest edge and a budget inside it bisects, but both then read the
    same pencil, so two rungs landing on one ``lambda`` report one ``edf`` and
    one ``statistic`` -- bit-for-bit, since the closed forms are deterministic
    in ``lambda``.  The form this replaces answered a clamped rung from a
    factorization of ``V_eff + lambda S`` and a searching rung from a
    diagonalization of the same pencil, and the two could differ by whole
    degrees of freedom at one lambda.  Pinned in
    ``test_the_dense_ladder_reports_one_edf_per_lambda``.
    """
    root = None if penalty_root is None else np.asarray(penalty_root, dtype=np.float64)
    if root is None or root.size == 0:
        # No penalty to scan: every rung is the block's own achieved RANK, at
        # ``lambda0 = 0``.  The pencil answers it like any other rung -- with
        # an empty penalty root every filter factor is exactly 1, so the sum
        # IS the retained rank of the factor.  An all-zero penalty reaches here
        # as an EMPTY root rather than as a zero matrix, which is the same
        # predicate ``not np.any(S_ti)`` used to make and one fewer scan of a
        # ``(k, k)`` block.
        p = _pair_pencil(pair, None)
        stat, rank = _pencil_stat(p, 0.0), _pencil_edf(p, 0.0)
        return [ScreenedPair(statistic=stat, edf0=rank, lambda0=0.0) for _ in budgets]

    # ``tr(S)`` off the factor: ``||rootS||_F**2``, which is the trace of the
    # penalty it roots and never assembles it.  The structured ladder's bracket
    # is ``tr(V_eff) / (tr(S_a) * L)`` over the same 1e+-10 edges, and the two
    # are pinned equal so a pair both paths can score gets one ``lambda0``.
    #
    # THE NUMERATOR COMES OFF THE PENCIL, WHICH IS WHY THE PENCIL IS BUILT
    # FIRST.  ``tr(V_eff)`` is the mass of the block the pencil scored, and
    # this line used to re-slice the factor for it instead -- the same number
    # only while the overlap has full rank, and 18.06% low on the
    # ``OrderedCategorical`` geometry where it is not.  See
    # :func:`superglm.screening._pair_factor._pair_scale`.
    p = _pair_pencil(pair, root)
    scale = max(p.tr_v, 1e-300) / max(float(np.sum(root**2)), 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale
    edf_lo, edf_hi = _pencil_edf(p, lo), _pencil_edf(p, hi)

    out: list[ScreenedPair] = []
    for budget in budgets:
        edf0 = float(budget)
        if edf_hi < edf0 < edf_lo:
            lam = _lambda_for_edf(p, edf0, scale)
        else:
            lam = lo if edf0 >= edf_lo else hi
        out.append(
            ScreenedPair(
                statistic=_pencil_stat(p, lam),
                edf0=_pencil_edf(p, lam),
                lambda0=float(lam),
            )
        )
    return out


def penalized_score_statistic(
    pair: PairFactor,
    penalty_root: NDArray | None,
    *,
    edf0: float = 4.0,
) -> ScreenedPair:
    """Rank one candidate pair by its penalized efficient-score statistic.

    ``pair`` carries the overlap span the mains model already explains as the
    leading block of its own factor, so the profiling that used to need ``C``,
    ``M`` and the overlap's own score is a slice of a triangular matrix here
    rather than a difference of two Grams.  With ``penalty_root`` absent or
    empty the statistic reduces to the unpenalized ``U_eff' V_eff^+ U_eff`` and
    ``lambda0`` is reported as 0.

    Scoring a ladder of budgets? Use
    :func:`penalized_score_statistic_ladder`, which shares one decomposition
    across every rung instead of rebuilding it per call.
    """
    return penalized_score_statistic_ladder(pair, penalty_root, budgets=(float(edf0),))[0]
