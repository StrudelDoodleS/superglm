# PSST reference variance: derivation and measurements

PSST now standardizes each ladder rung by its Gaussian-reference variance,
`2 * sum(a**2)`, where `a` contains the shrinkage eigenvalues. Previously it
used `2 * sum(a)`. The two agree on an identified unpenalized block, but
shrinkage makes the old denominator too large. This changes the ranking score;
it does not turn PSST into a calibrated test of a fitted GAM.

The correction covers the dense and structured spline-by-category paths.
The public table keeps its nine columns. Fitting, candidate construction,
dispersion estimation and the EDF budgets keep their existing contracts.

## The reference law

Work on the retained candidate space after projecting out the pair's
intercept and main-effect columns. Let `U` be its working score, `V` its
positive-semidefinite Fisher curvature and `S` its positive-semidefinite
penalty. Fix `lambda >= 0`, assume `A` is positive definite on the retained
space, and write

\[
A=V+\lambda S,\qquad T=U^\top A^{-1}U.
\]

The inverse is restricted to the identified space. A common null direction
of `V` and `S` contributes nothing. Relative to the nuisance-only profiled
optimum, completing the square in

\[
q(b)=U^\top b-\tfrac12b^\top A b
\]

gives `max q = T/2`. This is an exact working-quadratic identity. A complete
refit changes that quadratic and need not achieve the same gain.

Now assume the geometry is fixed and
\(U\sim N(0,\phi V)\), with known dispersion `phi > 0`. Set
\(F=V^{1/2}A^{-1}V^{1/2}\). An orthogonal diagonalization of `F` gives

\[
T/\phi\overset d=\sum_j a_j Z_j^2,\qquad
E(T/\phi)=\sum_j a_j,\qquad
\operatorname{Var}(T/\phi)=2\sum_j a_j^2,
\]

because the independent standard normals have `E[Z²] = 1` and
`Var(Z²) = 2`. Since `A >= V`, every eigenvalue satisfies `0 <= a <= 1`.
Thus `sum(a²) <= sum(a)`, with strict inequality if any identified direction
is partially shrunk. Matching EDF fixes the reference mean, but not the
reference variance.

For example, filters `(0.5, 0.5, 0.5, 0.5)` and `(1, 0.5, 0.25, 0.25)`
both have EDF 2. Their reference variances are respectively 2 and 2.75;
the previous denominator treated both variances as 4. These are exact
fixtures in the regression suite.

The new score is

\[
z_\lambda=
\frac{T_\lambda/\phi-\mathrm{edf}_\lambda}
     {\sqrt{2\sum_j a_j(\lambda)^2}}.
\]

PSST still reports the largest score over the chosen EDF budgets. Those
rungs are dependent. Standardizing each rung does not standardize their
maximum, equalize their tails, or establish the null distribution after
fitting the baseline and estimating weights, dispersion and smoothing.
In particular, normal simulation from the fixed candidate matrix alone
does not supply exact p-values for that entire procedure.

## Dense calculation

The existing generalized factor decomposition gives nonnegative arrays `v`
and `s` such that the filters are `v / (v + lambda*s)`. The correction sums
their squares at the achieved lambda. It reuses the decomposition already
needed for the statistic and EDF, with linear additional work in its width.
Zero-denominator common null directions contribute zero.

The lambda search also uses `sqrt(lo) * sqrt(hi)` for its geometric midpoint.
For positive finite representable endpoints the geometric mean lies between
them. Forming `lo*hi` first can overflow or underflow even when that midpoint
is representable. Extreme penalty-unit regressions reproduced this failure:
the old search returned zero or infinite lambda and missed the requested EDF.

Review exposed a separate failure in the endpoints themselves. Both ladders
now clip the bracket to positive finite float64 values. The dense pencil
also retains its unbalanced factors when their trace ratio cannot be
represented as a positive finite number. A required lambda beyond float64's
range cannot be recovered by better midpoint arithmetic; the ladder reports
the EDF achieved at the finite edge instead of claiming target attainment.

## Structured calculation

Let `G` be the profiled design in its orthogonally compressed row space and
`B` the symmetric retained inverse used by the existing structured statistic
and EDF. Put `V = G.T G` and `F_B = V^(1/2) B V^(1/2)`.
Its row-space smoother is `C = G B G.T`. Cyclicity of trace gives

\[
\operatorname{tr}(F_B^2)=\operatorname{tr}(C^2)=\|C\|_F^2.
\]

When `G (B-A⁺) G.T = 0`, this is the same reference variance derived above.
In floating point the calculation follows
the existing retained-space policy; the consistency checks below do not
establish equivalence to an untruncated original pencil.

The calculation therefore needs squared smoother blocks, including the
off-diagonal blocks between levels. Squaring the EDF or adding only the
diagonal blocks would give the wrong variance.

The existing per-level QR supplies `K_q = R_q T_q⁺`, the cross factor `Y_q`,
and overlap rows `Phi_q`. Put `E_q = K_q Y_q` and `Z_q = [E_q, Phi_q]`.
Using the same resolved, extended and coupled border matrices as the EDF
calculation, the off-diagonal block is

\[
C_{qt}=Z_q M Z_t^\top,\qquad
M=\begin{pmatrix}
\mathrm{resolved}&-\mathrm{extended}\\
-\mathrm{extended}&\mathrm{coupled}
\end{pmatrix}\quad(q\ne t).
\]

Existing PSD factors give each diagonal block as `D_q D_q.T`. The compacted
base block participates in both sums; it cannot be omitted merely because
it has no emitted coefficients. Hence

\[
\|C\|_F^2=
\sum_q\|D_qD_q^\top\|_F^2+
2\sum_{q<t}\|Z_q M Z_t^\top\|_F^2.
\]

In a retained border direction with squared singular value `h`, the raw
two-by-two metric is

\[
\begin{pmatrix}1/h&-1/h\\-1/h&(1-h)/h\end{pmatrix}.
\]

For `h < 1/2`, transform that direction's leaf columns to
`[(E-Phi)/sqrt(h), Phi]`; its metric becomes `diag(1, -1)`. For larger
retained directions the raw metric has norm below 4. An absorbed direction
keeps the projector terms `[[0, -1], [-1, 1-h]]`. Deleting those terms is
incorrect even when the reciprocal is zero. This transformation bounds the
metric; subtraction in the transformed leaf still amplifies error by
`1/sqrt(h)`, which enters the consistency allowance.

At `h=1/2`, the raw metric's eigenvalues are `(3 ± sqrt(17))/2`, giving
norm 3.561553, still strictly below 4.

A streaming binary QR tree evaluates the cross-level sum. At a merge of
two disjoint leaf sets, their QR factors `R_left` and `R_right` contribute
`2 * ||R_left M R_right.T||²`. Orthogonal transformations preserve this
norm. Every distinct pair of leaves first meets at exactly one merge, so
each off-diagonal block is counted once. The merged rows are then compressed
by QR. All accumulated terms are nonnegative; no large traces are subtracted.

For `L` levels, spline width `k_a` and overlap rank `r`, the local contractions
and tree add at most `O(L*(k_a+r)³)` work: linear in the number of levels at
fixed widths. The tree retains `O(k_a*r + r² log L)` state, including an
uncompressed leaf. Existing chunk buffers bound the other temporaries. No
full cross-level smoother is assembled. Variance is evaluated only at
distinct emitted lambdas, not at every bisection step.
Each final evaluation requires another factor pass and is included in the
caller's evaluation budget. Duplicate budgets share it.

## Numerical scope and regression evidence

For any upper bound `d` on the retained dimension, the reference eigenvalues
imply `EDF²/d <= tr(F²) <= EDF`. The implementation uses the ambient ceiling
`d = L*k_a`, not a separately computed retained rank. It checks these
bounds with an allowance based on its existing PSD assembly model, QR depth,
dimensions, machine epsilon and the transformed leaf's error amplification.
For a smoother perturbation of Frobenius norm at most `delta`, the squared
norm can change by at most `2*||C||_F*delta + delta²`.

These are consistency checks under that assembly model. They are not a
complete forward-error certificate for every operation of the original
factorization or a proof of all rank decisions. Existing rank-gap and
PSD-assembly refusal checks still apply. An invalid final variance refuses
that rung while preserving independent surviving rungs.

Regressions cover the exact equal-EDF examples, unpenalized ranks, zero rank,
penalty-unit rescaling, an actual public smooth-interaction score, and
structured variances against independently assembled observation designs.
The independent oracles factor the augmented design, not the implementation's
block formulas. Tolerances include dimensions, epsilon and conditioning.
Separate tests count actual factor passes and check refusal behavior.

With zero base-level mass and zero penalty, the independent design has four
identified directions, so the smoother is a rank-four projector and its
reference variance is 8. Deleting the absorbed projector terms publishes
5.3333, which the regression rejects. The original `2*EDF` normalization
also failed the penalized fixtures before the correction. An independent
review checked 36 further weighted structured cases against augmented-factor
oracles; the largest norm-scaled variance difference was `3.14e-14`.

## What changed in measured rankings

The 80,000-row freMTPL2 guide example was regenerated, including its two
complete candidate refits. Its top score changes from 1.8455 to 2.3235.
The twelve-pair order is unchanged. The two training deviance gains remain
43.0370 and 73.0481; those gains do not establish held-out benefit.

The null battery was rerun on both revisions: four families, 40 seeds,
8,000 rows per dataset, 160 complete fits and 3,520 candidate pairs per arm.
Neither arm failed a fit or returned a non-finite score. The maximum `ti`
score rises from 7.31 to 9.48 and the maximum `spline_cat` score from 5.53
to 5.66. Unpenalized scores are unchanged. These are measured maxima, not
calibrated rejection thresholds.

A separate small Gaussian experiment used six features and eleven eligible
pairs, with three planted interactions: smooth-by-smooth, numeric-by-category
and category-by-category. The fixed design used seeds 10, 20 and 30 at each
of strengths 0.05, 0.10 and 0.20, with 8,000 training and 8,000 independent
held-out observations per dataset. Every candidate was refitted on both
revisions; all fits converged.

| Signal strength | Planted pairs in top three, across three datasets | Old / corrected |
|---|---|---|
| 0.05 | out of nine | 4 / 4 |
| 0.10 | out of nine | 9 / 9 |
| 0.20 | out of nine | 9 / 9 |

Both revisions selected the same top-three set in all nine datasets, while
the complete order changed in seven. Candidate held-out gains were identical
between revisions. The weak-signal cases include false selections and
negative held-out gains. This small experiment establishes neither a power
improvement nor a universal ranking advantage.

Nor does stronger shrinkage alone guarantee more power. If all `d` identified
directions have the same filter `a > 0`, the factor cancels from the corrected
score: `(a*Q - a*d) / sqrt(2*d*a²) = (Q-d)/sqrt(2*d)`. Equal filters are the
condition; an identity ridge penalty need not give equal filters when Fisher
curvature is anisotropic. Under the unpenalized Gaussian reference, fixing
the dimension and profiled noncentrality also fixes the entire score law,
regardless of how the signal is distributed across whitened directions.

The correction is justified by the reference law. A stronger detector needs
a separate comparison: detection at matched false-positive rates, and
held-out gains for a fixed candidate-refit budget. Correlated main effects,
weak interactions and wider candidate menus are useful next stress cases.
The current evidence supports retaining PSST as a shortlist tool while
building that evaluation, rather than claiming that larger scores solved
interaction selection.

## Complete-fit and screening cost

`benchmarks/screening_reference_variance.py` was run in separate baseline and
corrected processes, sequentially after the other numerical experiments.
Each process warmed compilation, then repeated three complete fits and
screens on 200,000 training rows, with a separate 200,000-row prediction set.
The baseline was `7bbb415c`, whose source tree matches master at `c0ed3a62`.
Python was 3.13.14, NumPy 2.5.2 and SciPy 1.18.0; BLAS and OpenMP used one
thread and Numba used two.

| Case | Fit median, old / corrected | Screen median, old / corrected | Process peak RSS, old / corrected |
|---|---|---|---|
| Six features, eleven dense pairs | 3.720 / 3.417 s | 0.249 / 0.196 s | 502.125 / 501.969 MiB |
| One smooth, 160-level factor | 0.284 / 0.356 s | 0.150 / 0.195 s | 496.566 / 496.762 MiB |

The dense screen ranges overlap: 0.212–0.267 seconds before and 0.181–0.279
after. These three repetitions support no speedup claim. The structured
screen pays for the additional variance pass and tree; its median increases
by about 45 milliseconds in this case. Fit routines were unchanged, and
their timings also vary between runs.

The recorded fit EDF, deviance, dispersion, held-out loss and prediction
summaries are identical across revisions. Every repeated model gives the
same predictions. Dispatch instrumentation observed eleven dense ladder
calls in each mixed screen and one structured call in each wide-factor
screen, with no refusals. Peak RSS includes imports, both datasets, warmup
and repeated models; it is not an isolated screening allocation measurement.
The observed process peaks differ by less than 0.2 MiB.

The reproducible records are
`benchmarks/screening_reference_variance_receipt.json` (performance and all
nine ranking experiments) and
`benchmarks/screening_reference_variance_null_receipt.json` (the paired null
battery). Each performance and ranking run records its source hash. Later
source edits qualify docstrings and bind each tree sibling to a local
variable for static type narrowing; numerical operations and counts are
unchanged. The script records candidate refits
with `--refit`; run its identical file with each revision's environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
  uv run --no-sync python /path/to/screening_reference_variance.py \
  --case structured --rows 200000 --repeats 3 --output /tmp/receipt.json
```

## Review regression: an affordable binned pair was refused

[Codex identified a routing gap](https://github.com/StrudelDoodleS/superglm/pull/389#discussion_r3997115819)
in the extra-pass accounting. At the default `max_cells=5_000_000`, a
width-11 spline with 5,094 support points and 200 factor levels fits the
allocation gates but has work allowance for only two factor passes. The
penalized ladder needs at least three. The caller admitted the exact support,
received a refusal and returned `NaN` without reaching its binning fallback.

The first routing fix, `08a19c3e`, checked whether two passes could suffice
after the allocation gates, using the built marginal penalty. A later review
found that this special case did not cover every work-budget refusal. The
current caller uses a distinct budget signal and resumes binning. A cheap
minimum-pass check still runs before allocating cell tables; the ladder
determines additional search and variance costs. Numerical certification
failures retain their refusal contract.

The public regression failed against `c2be0f6` with a non-finite score before
the routing fix. It now passes, alongside controls that keep the exact
support when the allowance is raised to four or the penalty is zero.
`test_variance_pass_budget_reaches_the_spline_binning_fallback` runs a complete
Gaussian fit and the real structured kernel in all three cases.

`benchmarks/screening_reference_variance_review_receipt.json` records the
original comparison against `08a19c3e`: three complete fits and screens per
revision after warmup, using the same
5,094-row dataset and seed 389:

| Measurement | Before routing fix | After routing fix |
|---|---|---|
| Structured support | 5,094 | 256 |
| Work allowance | 2 passes | 680 passes |
| Pair outcome | Refused, `NaN` | `z = 1.128655`, `approx=True` |
| Fit median | 0.098 s | 0.118 s |
| Screen median | 0.248 s | 0.103 s |
| Process peak RSS | 460.715 MiB | 400.496 MiB |

The recorded fit EDF, deviance, dispersion and prediction summaries are
identical. Each screen used one structured ladder call; all three baseline
calls refused and all three corrected calls returned a score. These timings
compare a refusal with a completed binned calculation, so they establish no
general speedup. RSS includes imports, warmup and complete models. Reproduce
the receipt with the benchmark command above, replacing the case with
`--case variance_budget --rows 5094 --seed 389`.

## Further review checks

[Claude's review](https://github.com/StrudelDoodleS/superglm/pull/389#issuecomment-5648671666)
derived the dense reference moments, structured off-diagonal metric,
base-level compaction and tree identity independently. Its environment had
no shell, so the following are local executed checks of its findings.

- Dense roots scaled by `2**-500`, `2**-537` and `2**-700` failed before the
  endpoint fix. The first has a representable target lambda; the last two
  require a finite-edge clamp. The structured penalty at `2**-1000` also
  failed, with SVD nonconvergence after its endpoint overflowed. The extended
  tests now pass, checking attained EDF or the independently computed
  near-identity smoother as appropriate.
- The observation-space oracle now includes ten levels and forced
  two-level chunks. Replacing QR compression with row truncation leaves all
  six original four-level cases passing but fails all twelve larger cases.
  Those cases consume compressed factors in later merges, covering a path
  the original oracle did not reach.
- The benchmark's `sqrt(edf0/2)` Cp score threshold is now explicitly
  restricted to unpenalized Gaussian rows. For penalized rows, the same
  algebraic score rule can be applied through `statistic > 2*edf0` without
  adding a variance column. It is not a guarantee about a penalized refit.

The fixed-seed null smoke bounds remain unchanged. The review identified no
failing null fixture, and these checks do not draw fresh seeds on each run.
The guide continues to distinguish their bounds from calibrated thresholds.

### Width-45 variance cost

Codex's follow-up asked whether a variance pass costs several ordinary
passes because the tree's width reaches `2*r`. The requested geometry was
measured with 200,000 rows, 101 support points, spline width 45, 34 factor
levels and overlap rank 46. At `max_cells=1_000_000`, it receives an allowance
of three passes and uses the structured route on exact support.

Three sequential complete-fit measurements per revision give fit medians
of 0.367 seconds before the variance correction and 0.358 after, screening
medians of 0.212 and 0.254 seconds, and process peak RSS of 513.8 and
511.9 MiB. Recorded fit outputs are identical; both arms score the pair.

A separate profiler run covers two complete screens. It records six calls
to `_filter_factor_sum`, two final variance evaluations, and 66 tree merges.
Per screen, merges take 5.55 milliseconds and the final variance evaluation
42.49 milliseconds; the ordinary evaluations average 37.04 milliseconds.
The recorded ordinary block QR is already 90 by 91, while the extra tree
QRs have up to 184 rows and 92 columns. Comparing tree width 92 with the
budget's dimensional proxy 46 does not compare two actual factorizations.

The allowance continues to count factor passes under a dimensional estimate.
It certifies neither floating-operation count nor elapsed time; the docstring
now states both limits. The measured extra cost is retained explicitly rather
than using this single geometry to recalibrate admissions. The paired raw
receipts, profile counts and observed QR shapes are in
`benchmarks/screening_reference_variance_review_receipt.json`. Reproduce the
timings with `--case structured_wide --rows 200000 --repeats 3`; run the same
command separately under `python -m cProfile` for attribution.

## Follow-up review: routing progress and extreme penalty units

[Codex reproduced an infinite routing loop](https://github.com/StrudelDoodleS/superglm/pull/389#discussion_r3997759709)
at 1,000 spline support points, width five, six factor levels and
`max_cells=6250`. The dense path tried the structured path before binning.
Its two-pass special case restored the unchanged dense state, which tried
the same handoff again. A regression limits repeated visits to an unchanged
routing state so this defect fails without hanging the test process.

The ladder now distinguishes work-budget refusals from numerical refusals.
A budget refusal resumes binning, and a failed speculative handoff latches
so the caller cannot repeat it. This also addresses
[Claude's two-edge concern](https://github.com/StrudelDoodleS/superglm/pull/389#issuecomment-5648872796).
On the 5,094-point, width-11, 200-level fixture, `max_cells=5_005_000` affords
three passes. Budgets `(2, 3000)` clamp to different edges and need four.
The previous commit returned `NaN`; the new path bins and scores the pair.
Both public failures were reproduced against `9817a76c`, with the three
existing routing controls passing.

[Codex also found a finite-root overflow](https://github.com/StrudelDoodleS/superglm/pull/389#discussion_r3997759711).
For an identity design and penalty root `2**538 * I`, squaring the root
overflows. Yet at the smallest positive float64 lambda, the physical penalty
is `4*I`. The correct filters are all `1/5`, giving EDF `0.8`, reference
variance `0.32`, and statistic `6` for row score `(1, 2, 3, 4)`.

When the penalty norm or balancing ratio exceeds the ordinary representation,
the dense path now writes `rootS = 2**e * scaled_rootS`. It carries that
exponent through balancing and the lambda bracket. Each squared cosine,
sine and rotated score keeps its own mantissa and exponent. Thus a small
direction survives until multiplication by the physical lambda. The
denominator contains only nonnegative aligned terms. Moment terms are summed
before final float64 rounding, since several individually unrepresentable
terms can have a representable total.

The exceptional path also recovers the curvature block of Q by solving
against the retained triangular factor. Forming Q explicitly can round a
tiny entry to zero even when the balanced stack is well conditioned. The
ordinary path retains its existing arithmetic and rank policy. This change
does not certify arbitrary ill-conditioned factors or recover information
already lost in the input or decomposition.
The recovered Q1 block has a conditioning-dependent solve error and need
not share the explicit Q's orthogonality error. Root entries outside float64's
range after rescaling can round to zero. These range regressions exercise
the internal factor-scoring functions; they do not establish an extreme-value
failure reached through `screen_interactions`.

Independent diagonal regressions cover the uniform example, a penalty null
direction, `diag(2**538, 1)` at lambda one, and a tiny curvature and score
direction whose correct moments are `(1.5, 1.5, 2.5)`. Further cases preserve
subnormal moment totals and recover a bracket whose intermediate trace
quotient underflows. All eight added range cases fail on the previous code.

At extreme scales, a requested EDF can lie between adjacent representable
lambdas. For the uniform example, EDF is `0.8` at the smallest lambda and
`4/9` at twice that lambda. Targets `0.7` and `0.6` therefore select different
endpoints. The dense search chooses the closest achieved EDF when its
midpoint cannot advance, breaking an exact tie with the smaller lambda.
The structured path's separate numerical certification policy is unchanged.

Five small factor-reconstruction probes include a rotated full-rank case.
Their retained triangular condition numbers range from 1 to 2.072. With
`d = max(stack.shape)`, the measured orthogonality defect divided by
`d*eps*cond(R11)` is at most 0.154; relative reconstruction residuals divided
by `d*eps` are at most 0.127. The largest absolute moment error against the
analytic or positive-definite oracle is `6.67e-16`. These scaled residuals
are diagnostics, not proved error bounds. Reproduce them with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/screening_extreme_penalty.py \
  --output /tmp/closure.json
```

### Cost after the follow-up fixes

The same complete-fit benchmark compared `9817a76c` with the follow-up
source in separate processes, with three fits and screens after warmup.
The first two cases use 200,000 rows; the budget cases use 5,094 rows and
seed 389. The thread limits and package versions are those reported above.

| Case | Fit median, before / after | Screen median, before / after | Process peak RSS, before / after |
|---|---|---|---|
| Eleven dense pairs | 3.402 / 3.234 s | 0.218 / 0.220 s | 517.469 / 502.285 MiB |
| Width 45, 34 levels | 0.429 / 0.406 s | 0.288 / 0.258 s | 512.113 / 513.719 MiB |
| One clamped edge, two affordable passes | 0.107 / 0.124 s | 0.090 / 0.106 s | 410.945 / 400.188 MiB |
| Two clamped edges, three affordable passes | 0.111 / 0.107 s | 0.278 / 0.356 s | 442.020 / 442.203 MiB |

Recorded fit EDF, deviance, dispersion, held-out loss and prediction
summaries are identical in all four comparisons. Screening tables and
dispatch are identical in the first three. The last case changes from a
refusal to a finite binned score, `z=1.422649`. It pays for an exact budget
refusal and then a binned structured call. The cheap minimum-pass check
keeps the third case at one binned call. These repetitions support no
general speed or memory improvement claim.

`benchmarks/screening_reference_variance_round3_receipt.json` contains the
paired runs, source hashes and factor-reconstruction results. The benchmark
case names are `mixed`, `structured_wide`, `variance_budget` and
`variance_budget_two_edges`. Local verification passed 423 screening and
design-factor tests with real data required and no skips. After restoring
the cheap preflight check, all 165 structured and variance tests passed again.

## Final routing check: retry after support changes

Both [Codex](https://github.com/StrudelDoodleS/superglm/pull/389#discussion_r3997877878)
and [Claude](https://github.com/StrudelDoodleS/superglm/pull/389#discussion_r3997883761)
found that the speculative refusal flag survived binning. If the dense
intermediate still did not fit, it blocked an affordable structured retry.
The budget flag now clears when a support is binned. Numerical certification
failures keep their existing refusal policy for the pair. Each reset follows
a monotone bin flag change, so it cannot reintroduce the loop on unchanged
support.

Two complete-fit regressions fail against `9af47afe` and pass with the fix:

- A width-six natural spline, 71 levels and 1,557 support points at
  `max_cells=221206` has 120 structured passes before binning and 393 after.
  Its four searchable targets become affordable on 256 points, while the
  dense intermediate remains too large.
- A width-five P-spline, 20 levels and 300 support points at `max_cells=6800`
  has two passes before binning and twelve after. The cheap preflight refuses
  the exact support; the binned structured route can score it.

A third case injects a geometry certification failure on the exact support.
It remains refused after binning. Clearing a shared budget/numerical flag
makes this control fail by publishing a score, which is why the two flags
are separate. All 124 structured, routing and cost tests pass after the fix.

Three complete fits and screens per revision on the natural-spline fixture
preserve the recorded fit summaries. The previous screen returns `NaN` in
0.029 seconds; the corrected screen returns `z=2.272069` in 0.251 seconds,
including the exact budget refusal and the successful binned search. Fit
medians are 0.177 and 0.173 seconds. Process peak RSS is 407.621 and
393.965 MiB; these process measurements establish no memory improvement.
The raw runs and source hashes are in
`benchmarks/screening_reference_variance_routing_receipt.json`. Reproduce
with `--case structured_search_retry --rows 1557 --seed 391 --repeats 3`.
