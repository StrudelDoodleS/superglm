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

**SCALE DISCIPLINE.  Read this before combining two matrices.**

Four defects on this branch were one mistake: *two quantities combined as
though they were on one scale when they are not*.  By adding, by subtracting,
or by thresholding separately and differencing.

* the original units lesson (e9f7227): an ABSOLUTE whitening cut made the
  statistic depend on the units the curvature was carried in;
* ``G = V + S`` formed in floating point, which loses ``S`` entirely when the
  curvature dwarfs it.  Deriving ``s`` as ``1 - a`` then loses it a second
  time -- **and that half was never removed, only made harmless**; see rule 2
  below and :class:`_Pencil`, which measure it;
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
     share a scale.**  Balance before summing (:func:`_build_pencil`);
     equilibrate before counting (:func:`_psd_rank`).

     This rule once also read "carry both transformed terms rather than
     deriving one from the other (:class:`_Pencil`)".  **That clause was
     false about the code and has been removed rather than reworded.**
     :func:`_build_pencil` returns ``s = (1 - share) / balance`` and always
     has.  What makes the subtraction safe is the BALANCE, which puts the two
     terms on one scale before ``G`` is formed, so ``1 - share`` cancels only
     where the penalty share is genuinely zero -- measured in
     :class:`_Pencil` on three geometries, all such directions in
     ``null(S)``, none fabricated.  Deriving one term from the other is still
     the wrong default; it is tolerated at exactly one site, for a stated and
     measured reason.

**There are exactly two such sites**, which is what makes this a chokepoint
rather than a habit: :func:`_psd_rank` is the module's only relative-rank
threshold, and the ``G`` of :func:`_build_pencil` is its only sum of two
independently scaled matrices.  Equilibration and balancing live at those two
places rather than at their call sites, so a new caller cannot omit them.

Each has its own enforcing test, and each was verified to fail when its
site is reverted -- stated separately because they do NOT cover each other:

* :func:`_psd_rank` --
  ``test_screening_is_invariant_to_the_units_of_a_numeric_margin``.  Rescaling
  a numeric covariate is a change of units and nothing else, so the whole
  table must come back identical.  Reverting the equilibration turns a
  ``numeric_numeric`` pair from ``edf0 = 1`` into a NaN row at a scale of 1e4.
* ``G``'s BALANCING, and that alone --
  ``test_a_curvature_that_dwarfs_its_penalty_keeps_the_penalty``.  Reverting
  the balancing fails it while the units test still PASSES, which is measured
  rather than assumed: rescaling a spline's covariate rescales its penalty
  with it, so that route never reaches the ``V >> S`` regime.

  **It does NOT enforce the ``(v, s)`` parameterisation, and this entry used
  to claim it did.**  The shipped pencil derives ``s`` as ``1 - share`` and
  that test is green, so it cannot be holding the two apart; it is green
  because the balancing keeps the smaller term representable in ``G``.  A
  test for the parameterisation would have to defeat the balancing first and
  none exists.  That is the honest state: one site, one enforced property.

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

**THE ACCURACY CEILING IS ARCHITECTURAL, NOT ALGORITHMIC.  Read this before
rearranging the arithmetic below to chase a degree of freedom.**

This module is handed MOMENTS.  ``V_eff`` arrives as a Gram, so its spectrum is
the design's SQUARED, and squaring is what decides the whole question: on a
pair with a starved level the smallest directions fall under the noise floor of
the operator they are read from, and no correct digits are left in them.

**Count those directions; do not divide to find them.**  A condition number is
not measurable here -- its denominator IS the noise, so it is not a property of
the pair.  ``_thin_level_pair(1.0)``'s largest-over-smallest-positive ratio
reads 7.24e+06 at one thread and 6.11e+19 at sixteen on one box, changing
nothing else.  What IS stable, bit-for-bit across 1, 4 and 16 threads and
across Python 3.12 and 3.13, is a magnitude compared against ``eps`` times a
norm:

===================  =====================  ==========================
low weight           directions of          directions of ``A`` within
                     ``V_eff`` below        10x ``eps ||A||`` at the
                     ``k eps ||V_eff||``    ladder's high edge
===================  =====================  ==========================
1.0                  1                      0
1e-4                 1                      0
1e-12                **4**                  **1**
===================  =====================  ==========================

The starved pair carries four directions its own Gram cannot resolve and one
that survives into the high-edge operator -- exactly the one degree of freedom
the routes disagree about.  They are not disagreeing about an answer; they are
each reading a different rounding of the same absent information.

That is why the disagreement is a CONVENTION and not an error, and it is
measurable as one.  Sweeping the rank cut of an independent stacked-QR
evaluation over ``1e-18 .. 1e-6``:

======================  =============  ==========
rank cut                ``edf(hi)``    decades
======================  =============  ==========
``1e-18 .. 1e-16``      18.99995       three
``1e-15 .. 1e-13``      19.00000       three
``1e-12 .. 1e-7``       **18.27481**   **six**
``1e-6``                18.05829       --
======================  =============  ==========

**THERE ARE THREE PLATEAUS AND THEY DISAGREE BY 0.725 df.**  Two earlier
drafts of this paragraph said "there is no plateau", then "the widest window
giving one answer is under three decades".  Both were false against the table
directly above them -- 18.27481 holds to eight significant figures across six
decades, twice the width claimed.

The correct statement is stronger than either.  A single wide plateau is what
would CERTIFY a cut: it says the answer is insensitive to where the cut is put
within it.  Three of them, separated by most of a degree of freedom, say the
answer is a property of WHICH plateau the cut lands on -- and nothing in the
data chooses between them.  Any routine reporting a number for such a pair is
reporting its own threshold.

An earlier draft also contrasted this with "the arrow kernel's own nine-decade
plateau".  **That comparison was stale and is withdrawn** — it described a
rank cut the structured path no longer has.  Since it moved onto design
factors it evaluates ``edf`` as a sum of filter factors and takes no rank
decision at all, which :func:`_edge` says in its own docstring.  The contrast
that survives is sharper anyway: the other path does not need a plateau
because it does not need a cut.

**Four remedies have been measured and refused.  Do not re-derive them.**

1. *Give the pseudo-inverse fallback a stated cut.*  ``_edge`` already counts
   its unpenalized rank at :func:`_rank_floor`; what has no stated cut is the
   ``np.linalg.pinv`` branch it falls to when ``cho_factor`` refuses, which
   takes NumPy's shape-derived default ``rcond``.  Passing the arrow path's
   ``_solve_floor`` there -- the same ``max(n, 1) * eps`` expression -- gives
   **-8.17 df** against the shipped -1.21 df.  No scalar cut can work,
   because the deciding curvature (8.896e-05) sits BELOW the eigensolver's
   noise floor on the matrix it is read from (``eps ||A|| = 1.653e-04``).
2. *Answer every rung from the pencil* instead of from ``_edge``, the
   "balanced congruence" remedy: measured WORSE.  Across 1/2/4/8 threads the
   pencil's high-edge ``edf`` moves 1.0000 df on the starved pair (18.99998 at
   one thread, 17.99997 at two or more) where ``_edge`` moves 5.3e-07, and it
   is worse on three of four geometries.  The claim that this comes back
   bit-identical across thread counts does not reproduce.
3. *Force the whitening branch* (the Fix & Heiberger construction already
   below) rather than reaching it only on a hard ``LinAlgError``: still flips
   at eight threads, and 29x worse than the generalized driver on the 1e-4
   pair.
4. *The GSVD*, which is what the LAPACK Users' Guide (3rd ed., SIAM 1999,
   sec. 4.7 and its "Further Details" for the generalized symmetric definite
   eigenproblem) actually recommends here — it gives the driver's error as
   ``sqrt(n) (||B^-1||_2 ||A||_2 + cond(B) |lambda_i|) eps`` and names
   Cholesky-plus-GSVD as the tighter alternative when ``B`` is ill
   conditioned.  **SciPy exposes no GSVD**: ``dggsvd3``, ``dggsvp3`` and
   ``dtgsja`` are all absent from ``scipy.linalg.lapack`` at 1.18.0.  So the
   recommended method is unavailable, not rejected.

The remedy that WOULD work is the one :mod:`superglm.screening._structured`
took: read the design factors and never form the Gram.  A factor's spectrum is
the Gram's square root, so a direction the Gram has pushed under ``eps`` sits
at its square root in the factor — representable, with about half the digits
still there.  Measured on the structured side, that took the high-edge error
from 5.1e-06 to 7.4e-12 against a 60-digit oracle.  It is a change to what the
CALLER hands this module, not to anything in it.

**THE LOW EDGE HAS ITS OWN LIMIT AND IT IS A FIRST-ORDER SENSITIVITY,
``eps ||V_eff||_F ||G||_F``.**  (Not a ceiling: finite realizations exceed it,
measured at up to 1.0617x below.)
The section above is about the ladder's HIGH edge, and it left the impression
-- recorded in issue #279 and in the sweep that opened it -- that the low edge
was the good end.  It is not; it fails differently, and the two must not be
conflated.

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
case, which :func:`_edge` answers through ``pinv``, and on which
``k - lambda tr(A^-1 S)`` is not evaluable (at ``V = S = diag(1, 0)`` the
shipped ``edf`` is 1, and substituting a pseudo-inverse while keeping ``k = 2``
gives the wrong value).  Everything in this section is about the low edge of a
positive definite pencil; the common-null-space case is discarded before the
pencil is formed and is not what #279 is about.

The second term is the BRACKET's own response: ``lambda = alpha tr(V_eff)``
with ``alpha = 1e-10 / tr(S)``, so a probe with nonzero trace moves ``lambda``
too.  It is part of the gradient, not an afterthought -- an earlier revision
displayed only the first term here while the test used both, which is a formula
readers would have carried away different from the quantity asserted.

and over ``||E||_F = c`` that is maximised at ``E = c G / ||G||_F``, giving
exactly ``c ||G||_F``.  Take ``c = eps ||V_eff||_F`` -- ONE ROUNDING of the
operand's norm -- and the low edge's first-order sensitivity is
``eps ||V_eff||_F ||G||_F``.  **That maximisation is exact for the
DIFFERENTIAL and is not a bound on the finite response**, since ``edf(V + E)``
is nonlinear in ``E``; measured realizations reach 1.0617 of it.  ``G`` is the
TOTAL gradient, including the bracket's own response ``alpha (d edf / d lambda)
I`` from ``lambda = alpha tr(V_eff)`` -- which changes its norm by at most 1.6%
here, but only the whole gradient gives the ladder's maximising direction.

**THE RADIUS IS A STATED PROBE AND NOTHING MORE, AND TWO OVER-CLAIMS ARE
WITHDRAWN.**  It is not "smaller than the error already committed in forming
the Gram" -- that asserted a backward-error bound nobody had derived -- and it
is not established to be smaller either.  Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., Ch. 3 gives ``fl(X'X) = X'X + D`` with
``|D| <= gamma_n |X|'|X|`` COMPONENTWISE, measured here at 98.6x to 204.7x the
probe radius -- but that is an UPPER bound, and an upper bound puts no floor
under the actual error, which may be anywhere below it and is zero for exactly
representable products.  So what follows is the answer's SENSITIVITY to a
one-rounding probe.  Nothing here certifies an information floor, and the word
is not used for it.

**THAT FORM IS LOAD-BEARING AND TWO WEAKER ONES WERE REFUSED IN REVIEW.**

* ``eps / lambda``, which drops both norms, is dimensionally wrong rather than
  merely loose: rescaling the design leaves ``edf`` and its uncertainty alike,
  since ``V``, the perturbation and the bracket's ``lambda`` all move together,
  while ``eps / lambda`` moves inversely.  Measured on one geometry at design
  scales 1, 1e3 and 1e-3 and with the PENALTY scaled by 1e4, it reads
  6.47e-01, 6.38e+05, 8.47e-07 and 6.47e-05 -- twelve orders from a change of
  units alone.
* ``eps ||V_eff||_F / (lambda ||S||_2)`` fixes the units but substitutes the
  LARGEST penalty eigenvalue for the one in the deciding direction.  On a
  rotated ``S`` with eight orders of spectrum that understates the response by
  up to four orders: where the theory says 1 it reads 55.6 to 126.9 and 5924 to
  11124.  ``G`` carries ``S`` in the right place and needs no such proxy.

``G`` costs only solves against ``A``.  An earlier revision claimed ``lambda``
bounds ``cond(A)`` at about 1e10 "by construction"; **that is false for an
anisotropic penalty and is withdrawn.**  The only available bound is
``||A||_2 / (lambda lambda_min(S))``, which a rotated penalty makes vacuous
(1.6e+18) and a singular ``S`` makes unavailable outright.  Measured instead,
``cond(A)`` runs 1.0e+02 and 1.0e+06 on resolved geometries, 1.9e+11 and
2.2e+11 on isotropic unresolved ones, and 5.8e+12 and **5.4e+14** on rotated
ones -- so at the hardest the solve for ``G`` keeps about one digit in its
smallest direction.  That is disclosure rather than derivation; what makes the
quantity usable is measured stability, which is what the sweep checks.

**WHAT THE SENSITIVITY IS ON THIS FAMILY, PER DRAW RATHER THAN AS A
CONSTANT.**  Over the 21 draws it spans **1.6298e-15 to 4.9265e-04** -- twelve orders,
tracking the residualized design's smallest singular value, and NOT the tidy
2% band the ``||S||_2`` form reported before it was corrected.  All ten draws
whose design is rank deficient sit at ~2.6e-04; the well-conditioned ones sit
at 1e-15 to 3e-09.

**So on exactly the draws that can fail it, the answer's sensitivity is ~26x
the ``abs=1e-5`` the suite asserts**, and that is the whole of issue #279: the
bound is far below how far those answers move under a one-rounding probe, so
which draw is used decides whether it passes.  Seed 1's 1.2574e-05 is 0.049x
its own 2.5688e-04, and 1.26x the bound.  Nothing was mis-tuned; the number
was placed against observed errors rather than against a floor, because the
floor had not been derived.  Do not carry the number to another fixture --
carry the derivation.

The seed the fixture happens to run has a sensitivity of 2.6282e-04 like its
nine siblings, and reports an error of 1.37e-10, so it clears the suite's bound by
193x on a draw whose answer MOVES in the fourth decimal under a one-rounding
probe.  That is mobility, not uncertainty: the probe establishes no lower bound
on the Gram's actual error, so nothing here says seed 3's 1.37e-10 is anything
but accurate.  What it says is that the same geometry admits answers 2.6e-04
apart under perturbations of the size the arithmetic works in, so a tolerance
placed at 1e-5 is being decided by the draw.  A draw is not evidence about the
family here, and neither is a passing bound.

Which draws are bad is decided in the DESIGN, before this module is called: the
bad ones are those where a level's rows put the constant vector in the span of
that level's own spline columns.  Squaring sends such a direction under
``eps``; ``1 / lambda`` then multiplies whatever round-off is left there.
Pinned in
``test_the_low_edge_edf_is_only_as_determined_as_the_gram_it_is_read_from``,
which asserts the sensitivity against ONE ULP of the answer -- ``np.spacing``,
the real adjacent-float distance, not ``eps |edf|``, which is a relative scale
running 1.045x to 1.679x it.  Worst asserted-resolved 4.0956e-08 ulp against
best asserted-unresolved 3.0293e+10 ulp, so the boundary clears by 2.44e+07x
and 3.03e+10x.  Separately the maximising perturbation attains 0.9425 to 1.0617
of the predicted displacement against a first-order value of 1 -- and that it
EXCEEDS 1 is the direct evidence that this quantity is a first-order measure
rather than an upper bound on the finite response.  That ratio is REPORTED and
not asserted.  Two intervals were tried and both refused: bounding it fits a
window around observations, on a difference of two evaluations at ``cond(A)``
up to 5.4e+14, which is the sampled-width objection in another form.  So this
test carries ONE asserted boundary and no fitted constant.

**TWO EARLIER REVISIONS GOT THE BOUNDARY WRONG IN OPPOSITE DIRECTIONS.**  One
asserted 1e5 ulp while the prose said one -- a midpoint of the two observed
populations, which would have passed a resolved geometry amplified by five
orders.  The next asserted one ulp but measured it as ``eps |edf|``, which put
the ``sigma_min = 1e-3`` geometries at a 1.59x margin that the correct spacing
turns into 1.06x.  Those geometries are transitional and are now measured and
disclosed rather than asserted on either side.

**A SAMPLED WIDTH WAS TRIED FIRST AND IS NOT SOUND**, which is worth recording
because it looks convincing.  Taking ``max - min`` of ``edf`` over 32 random
perturbations and requiring it to be LARGE asserts a lower bound on a sample
range, where the theory supplies only an upper bound on sensitivity.  It moves
with both of its own arbitrary constants: 0.2548 at 4 draws against 0.8653 at
128, and 0.5063 to 0.8085 over 24 perturbation seeds at one configuration.  A
deterministic norm has neither axis.

**THE PROBE ALSO DRIVES THE LADDER RATHER THAN ``_edge`` AT A PINNED LAMBDA**,
because the bracket is ``1e-10 tr(V_eff) / tr(S)``, so a probe with nonzero
trace moves ``lambda`` too.  (An earlier revision justified that trace by
calling ``G`` PSD.  **It is not**, once the bracket term is in it: scale
invariance of the ladder gives ``<G, V_eff>_F = 0``, which no nonzero PSD
matrix can satisfy against a positive definite ``V_eff``.  The claim is
withdrawn; the term is carried because it belongs in the gradient, not because
of any sign property.)  Review is right that the omitted term can
cancel the response outright -- in ONE dimension ``edf = V / (V + 1e-10 V)`` is
constant in ``V`` while a fixed-lambda calculation reports a move.  It does not
cancel here, because the response is carried by the near-null direction at
``1 / (lambda s)`` while the bracket shifts with the TOTAL trace, which the
saturated directions dominate: measured, the ladder's displacement equals the
fixed-lambda one to **1.000 on all eight geometries**, with ``lambda`` moving 0
to 7.9e-16 relative.  The ladder is driven anyway, because one extra call
removes an argument.

**No arrangement of the arithmetic below narrows this.**  It is the same
architectural limit as the high edge, reached by a different route, and it
has the same remedy and the same tracking issue -- design factors, #257.  What
is genuinely different is that at the low edge there is nothing to COUNT, so
the high edge's probe reports a clean bill on exactly the draws that miss.

**WHERE THIS SITS IN THE LITERATURE, because half of it is textbook and half
of it is not.**  ``edf`` is the trace of the influence matrix of a general-form
Tikhonov problem -- equivalently the sum of filter factors over the generalized
singular values of the pair ``(X_w, L)`` with ``L' L = S``.  Every standard
algorithm for it takes the DESIGN or a backward-stable factor of it, never the
Gram: Elden, *BIT* 17 (1977) 134-145 and *BIT* 24 (1984) 467-472; Golub, Heath
and Wahba, *Technometrics* 21 (1979) 215-223; Hutchinson and de Hoog, *Numer.
Math.* 47 (1985) 99-106; Wood, *JRSS-B* 70 (2008) 495-518 sec. 3.2, which gives
exactly the stacked-QR form ``[W X; E] = Q R`` with ``A = K K'`` and calls the
Cholesky-of-``X'W^2X`` alternative the less stable of the two for "the
exacerbation of any numerical ill-conditioning that accompanies explicit
formation of ``X'W^2X``".  So the moment route is published, and published as
the route the standard method exists to avoid.

Two halves of the mechanism are settled and one is not.

* SQUARING.  Conventional eigen/SVD drivers are accurate ABSOLUTELY, to
  ``eps sigma_1``, so directions at or below ``eps sigma_1`` carry no relative
  accuracy (Demmel, *Accurate SVDs of Structured Matrices*, LAPACK Working Note
  130, 1997; the *LAPACK Users' Guide*'s own SVD error bound).  Forming a Gram
  squares ``sigma_1``.  Huang and Jia, arXiv:1907.10392, reject the
  cross-product pencil ``(A'A, B'B)`` for the GSVD on precisely this ground --
  small generalized singular values from it "may be recovered much less
  accurately and even may have no accuracy" -- which is one algebraic step from
  the quantity summed here.
* AMPLIFICATION.  ``1 / (lambda s_j)`` is elementary calculus, above.
* THE COMPOSITION IS OURS.  A sweep of the GCV/edf and inverse-problems
  literature found NO published forward-error bound for
  ``tr((A + lambda S)^-1 A)`` under perturbation of ``A``, and no method that
  recovers it accurately from a Gram alone.  The nearest published relatives
  bound the generalized singular values themselves (Huang and Jia, sec. 2.1,
  improving Stewart and Sun, *Matrix Perturbation Theory*, 1990) or the
  Tikhonov SOLUTION, not this trace.  Stated as ours rather than cited.

AND DO NOT REACH FOR THE STANDARD RESCUES; they were checked and each needs
what this module does not have.  Seminormal equations and its corrected form,
and iterative refinement on the normal equations, all require ``A`` itself or
an ``R`` that is the factor of a backward-stable decomposition of it (Bjorck,
*BIT* 7 (1967) 257-278 and *LAA* 88 (1987) 31-48).  The high-relative-accuracy
theory needs a rank-revealing decomposition whose factors are themselves
accurate (Demmel and Veselic, *SIMAX* 13 (1992) 1204-1245; Demmel et al.,
*LAA* 299 (1999) 21-80), and a Gram formed in floating point is not one: its
entries carry ABSOLUTE errors, which is an unbounded RELATIVE error exactly in
the direction that matters.  This is an information barrier, not an algorithmic
one.

One caution about the published tripwire, since it is the obvious thing to
reach for.  ``cond <= eps^-1/2`` recurs across that literature as the boundary
for the cross-product route, and here it is NOT conservative enough: the draw
that misses worst has ``sigma_min / sigma_max = 2.72e-07``, i.e. a design
condition number of 3.7e+06 against an ``eps^-1/2`` of 6.7e+07, and it is still
1.2574e-05 out.  ``1 / lambda`` moves the boundary, which is why the regime
test above is keyed to the width of the answer set and not to a conditioning
threshold.

What this costs in practice is small, and that is measured too rather than
assumed: on the published twelve-row freMTPL2 screen, one thread against
eight, the table order is identical, the ``z > sqrt(edf0 / 2)`` gate admits
the same single pair, and the worst ``|dz|`` is 2.93e-05.  The instability is
real, and it lives on geometries the screen ranks at the bottom anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from superglm.screening._pair_factor import (
    PairFactor,
    _pair_scale,
    _profiled_factor,
    _profiled_rank_scale,
)

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

    **THIS MODULE NO LONGER APPLIES IT DIRECTLY, AND THAT IS THE POINT OF
    ISSUE #257.**  Every cut here is now taken on a FACTOR, at
    :func:`_factor_rank_floor` -- this expression's square root, which is the
    same cut on the Gram the factor would square to.  The Gram form survives
    because it is what the sibling cut is derived FROM, and because
    ``test_the_dense_path_s_ceiling_is_its_gram_and_not_its_arithmetic``
    counts against it to establish the regime the change was made for.
    """
    return max(int(n), 1) * float(np.finfo(np.float64).eps)


def _factor_rank_floor(n: int) -> float:
    """:func:`_rank_floor`'s cut, expressed for a FACTOR rather than a Gram.

    A factor's singular values are the Gram's eigenvalues' square roots, so
    ``sigma > sqrt(n eps) * sigma_max`` and ``w > n eps * w_max`` are the same
    statement about the same direction.  Taking it here rather than after
    squaring is the whole of #257: the deciding direction on a starved pair
    sits at ``2.78e+20`` of conditioning in the Gram against ``1.67e+10`` in
    the factor, and only the second is inside what float64 carries.

    It is the same cutoff
    :func:`superglm.screening._structured._representative_projection` takes,
    whose docstring calls it "the square root of the Hermitian pseudo-inverse
    policy used by the dense path".  That sentence used to be a borrowing;
    since this module reads factors it is one policy at two sites, and the
    derivation lives here.
    """
    return float(np.sqrt(_rank_floor(n)))


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
    """

    v: NDArray
    s: NDArray
    u: NDArray


def _empty_pencil() -> _Pencil:
    return _Pencil(v=np.zeros(0), s=np.zeros(0), u=np.zeros(0))


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
    :func:`_build_pencil`, which this replaces, had to balance because it FORMED
    ``V + S`` and lost the smaller term outright when the two lived on far-apart
    scales -- ``V = 1e20 I`` against ``S = I`` rounds to ``V``, every share comes
    back 1, and the ladder reports the full dimension at every lambda.  Stacking
    adds nothing, so that failure is gone by construction and the balance is no
    longer load-bearing for it.  It is kept for a different and smaller reason:
    the SVD of each block is accurate to ``eps`` ABSOLUTELY, so a block whose
    norm is ``1e-10`` of the other has its singular values resolved to six
    figures rather than sixteen.  Scaling ``rootS`` by ``sqrt(tr(V_eff) /
    tr(S))`` puts both blocks at O(1) and the division that undoes it is one
    rounding on a quantity nothing cancelled.

    ``penalty_root = None`` is the unpenalized block: the stack is ``R_eff``
    alone,
    every ``s_j`` is exactly zero and every surviving ``c_j`` gives a filter
    factor of exactly 1, so ``edf`` comes back as the retained RANK and ``T``
    as ``|| Uc' z_t ||^2`` -- the projection of the profiled score onto the
    directions the design actually resolves.  It is the same estimator as
    every other rung rather than a second one beside it.
    """
    R_eff, z_t = _profiled_factor(pair)
    k = int(pair.tensor_width)
    balance = 1.0
    if penalty_root is None or penalty_root.size == 0:
        root = np.zeros((0, k), dtype=np.float64)
    else:
        root = np.asarray(penalty_root, dtype=np.float64)
        tr_v = _pair_scale(pair)
        tr_s = float(np.sum(root**2))
        if tr_v > 0.0 and tr_s > 0.0:
            balance = tr_v / tr_s
        root = np.sqrt(balance) * root
    if k == 0:
        return _empty_pencil()

    stack = np.concatenate((R_eff, root), axis=0)
    orthonormal, triangular, _pivot = scipy.linalg.qr(
        stack, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0 or float(diagonal[0]) <= np.finfo(np.float64).tiny:
        return _empty_pencil()
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
        return _empty_pencil()

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
    scale = max(_pair_scale(pair), 1e-300) / max(float(np.sum(root**2)), 1e-300)
    lo, hi = 1e-10 * scale, 1e10 * scale

    p = _pair_pencil(pair, root)
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
