# Practical smoothing convergence follow-through

This follows the C3/C1 implementation at `1a8952a4`. The earlier insurance
comparison reported a strict smoothing stop; it did not establish that the
book could not be fitted usefully. Original raw receipts remain unchanged.

## Corrected diagnosis

The freMTPL2 NB2 fixture has 610,212 training policies and 67,801 held out,
raw claim counts, log exposure offsets, and 92 coefficients. The earlier
`efs+newton` run disabled practical stopping. Its final lambdas ranged from
about 0.157 to 18.457, so this was not an infinite-penalty or cap problem.

Repeating that path with practical stopping enabled reproduces the same
`objective_rejected` result. An accepted Newton step improves negative LAML
by 2.3612. Fresh LAML differentiation then becomes unavailable. The old
recovery tries an opposite half-step, rejects its worse objective, and stops.
It also labels derivatives from the preceding source fit as terminal evidence
at the newer fit. Those old derivative fields must not be treated as a fresh
stationarity assessment of the accepted endpoint.

The unmodified plain-EFS practical route already succeeds:

| Route | Smoothing stop | Outer iterations | Coefficient fits | Negative LAML |
| --- | --- | ---: | ---: | ---: |
| Strict EFS plus Newton | `objective_rejected` | 8 | 9 | 125368.4676792604 |
| Practical EFS plus Newton | `objective_rejected` | 8 | 9 | 125368.4676792604 |
| Practical EFS | `practical_plateau` | 24 | 25 | 125370.4367427982 |
| Interim EFS plus Newton recovery (`8b66828d`) | `practical_plateau` | 11 | 12 | 125368.3549527070 |
| Final EFS plus Newton (`5f994c8f`) | `stationary` | 9 | 10 | 125367.9491525044 |

These are different smoothing solutions, not representation-equivalence tests.
The EFS practical endpoint is not claimed to minimize exact LAML. Relative L2
held-out mean-prediction difference from the earlier strict point is 0.128%,
while coefficient covariance and some dispersion parameters differ materially.
For NB2, a large change in theta near the Poisson limit need not imply a large
change in the response distribution. The observed relative L2 differences are
1.283% for predictive variance and 0.343% for probability of any claim; maximum
pointwise differences remain larger. These comparisons describe the endpoints,
not a universal tolerance or guarantee.

## Recovery behavior

Commit `8b66828d` retains the accepted coefficient fit when fresh Newton
gradient evaluation is unavailable. It resumes the existing bounded EFS route,
disables further Newton handoffs for that solve, and clears unavailable
terminal derivative metadata. Existing full-profile objective acceptance
continues to control progress. Hessian-only failures still use the existing
BFGS fallback. A released beyond-cap state is not silently clipped into the
ordinary EFS box; it retains an explicit `derivative_unavailable` result when
fresh derivative evaluation fails. This differs from `gradient_unresolved`,
which carries evaluated gradient evidence whose certificate cannot resolve
stationarity. The integration regression exercises a real bracket release,
then injects fresh derivative failure: the accepted model and released penalty
survive without stale derivative fields or an exception, including serialization.

Real Gaussian fault-injection regressions demonstrate the old missing EFS
continuation and the new retained-state behavior, bounded budgets, actual EFS
progress, and truthful derivative provenance. The full endgame suite passes
25 tests, including Hessian-only fallback coverage. The independent scoped
review found no blocking issue.

The frozen recovery implementation completes the identical NB2 fixture in 11
outer iterations. The [receipt](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_pragmatic_convergence_receipt.json)
and [comparison script](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_pragmatic_compare.py) record raw-file
hashes, source provenance, work counts, dispatch, memory and held-out comparisons.
Timing is unmeasured because numerical work may overlap Headroom/Kompress and
tests; these NB2 numerical runs support no speed claim.

Held-out means differ from the old rejected Newton endpoint by at most
`4.04e-9`. Its smoothing penalties change, however, and conditional coefficient
covariance changes by 12.4% in relative Frobenius norm. Compared with the plain-EFS practical
endpoint, the recovered covariance differs by 36.98% (using plain EFS as the
reference). Both covariance matrices
accurately invert their recorded terminal penalized curvature; this does not
prove the coefficient modes are equally accurate.

The recovered coefficient solve stops on `objective_and_step` after 28
backtracks. Its remaining local Newton decrement squared is about 0.1476, so
retained-score stationarity and stable inference are not established by the
practical outer stop. The targeted fixed-lambda check found that the full Newton correction genuinely
improves the coefficient objective by about 0.07448, agreeing with the local
quadratic prediction. Fresh scores agree with independent directional
differences. The kernel rejects the step because one zero-count policy crosses
its numerical `mu/theta >= 2**-26` guard. This is a binary64 evaluation limit,
not a fitted-model constraint or evidence that the entire book is Poisson.
Commit `5f994c8f` extends the low-mean ratio floor to `2**-52` using the
existing checked finite-NB2 arithmetic. The reciprocal-ratio bound remains
`theta/mu >= 2**-26`; absolute/effective exponent, weight, overflow and
derivative-representation checks are retained. Ratios below the new finite
range are still refused, and an exact Poisson active face remains unsupported.

## Finite-NB2 range validation

The [finite-NB2 receipt](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_finite_nb2_receipt.json) and
[comparison script](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_finite_nb2_compare.py) retain the
controlled before/after numerical evidence and its provenance limitations.

The controlled before/after check uses the same saved coefficients, penalties,
frame and response. The formerly rejected full Newton trial now evaluates,
improving the package's penalized objective by `0.0744831634074`; an independent
exact-count likelihood recurrence gives `0.0744831633638`. Fresh geometry
reproduces the saved score. This fixes an evaluation-domain obstruction,
rather than altering the likelihood or its derivatives.

The asymmetric floor has a bounded exponent rationale. At the retained
absolute and weight limits, the leading zero-count natural theta-Hessian term
scales as `(weight/theta) * (mean/theta)**2`. A ratio floor of `2**-52`
keeps the worst exponent near `-1004`, above the binary64 normal floor
`-1022`. [NumPy documents the binary64 epsilon and exponent conventions](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html).
Removing the floor entirely can underflow both the natural and comparison
channels, so doing so would defeat the existing retention check.

Focused regressions cover the newly supported band against high-precision
exact-count native/log score and Hessian oracles, prior/frequency-weight
corners, the actual low-exposure policy, high-order curvature differences, and
continued refusal outside the bounded range. All 168 focused kernel/family/EFS
checks pass. Independent mathematical review confirmed the asymmetric domain
and nonzero extreme Hessian behavior.

The high-order tests combine the existing Richardson truncation indicator with
an explicit floating-point stencil-roundoff allowance. As in the original C3
assessment, the indicator alone is not a total derivative-error enclosure.
This change does not introduce a new global certification system.

The final whole-book fit on frozen source
`5f994c8f6ac0501606594e2f36bfc0cd24050ec1` reaches `stationary` in nine
outer iterations, with ten recorded coefficient fits and 34 inner iterations.
Its [final NB2 receipt](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_pragmatic_nb2_final_receipt.json)
and [comparison script](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_pragmatic_nb2_final_compare.py)
retain the full numerical comparison and source/configuration checks.

The terminal coefficient solve stops on `objective_and_score`, with relative
score `1.43e-8` below `inner_tol=1e-7` and zero backtracks. Its local Newton
decrement squared is `1.68e-7`, corresponding to an estimated remaining
quadratic gain of about `8.4e-8`. The earlier stalled-mode concern is resolved.

The smoothing gradient is freshly evaluated at the retained fit: its largest
component is `2.65e-4`, or `2.11e-9` after objective normalization, below the
configured `reml_tol=1e-6`. No cap remains unresolved. This satisfies the
existing numerical stationarity contract. The separate approximate EFS residual
is not the Newton endgame's stationarity authority.

Conditional covariance is positive definite and has normalized inverse backward
error `2.38e-17`. This is a corrected fit and uncertainty result: its covariance
differs by 62.93% from the old rejected Newton point, 47.75% from interim EFS
recovery, and 29.66% from plain practical EFS, in relative Frobenius norm using
each older result as reference. Relative L2 changes from interim recovery are
0.1025% for held-out means, 1.160% for predictive variances, and 0.2664% for
any-claim probabilities; the maximum probability difference is 0.006373.
Tail theta values and coefficient uncertainty still require model validation.
These different fitted solutions are not representation equivalence.

Final validation and the separate performance comparison are complete; see the
recorded checks and current-source measurements below.

## Practical large-penalty reasoning

Increasing a smoothing penalty can remove negligible wiggles while leaving a
useful, stable finite fit. A useful practical stop does not require the numerical
lambda values themselves to settle. It must still distinguish fresh outward
smoothing pressure from an over-smoothed fit that calls for a smaller penalty.

For one Gaussian ridge coefficient with unit information and score t,
beta = t/(1 + lambda), conditional variance = 1/(1 + lambda), and negative LAML
up to a constant is

`0.5 * [log1p(1/lambda) - t*t/(1 + lambda)]`.

At lambda = 1e6, t = 0.5 genuinely favors infinite smoothing, whereas t = 2
has its finite optimum at lambda = 1/3. Both have tiny outward fit changes.
Fresh EFS pressure distinguishes them: the respective multiplicative updates
are approximately 4 and 0.25. This supplies a numerical-regression invariant
without depending on floating-point signs near zero.

[mgcv fitting controls](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.control.html)
document practical EFS stopping, a log-penalty cap and bounded optimizer
fallbacks. [Wood and Fasiolo](https://arxiv.org/html/1606.04802) motivate bounded
updates and objective step control. This work uses those principles without
importing implementation code.

## Implemented practical outward policy

The additional `practical_plateau` route checks the configured number of
accepted EFS updates (default three). Every exempt smoothing coordinate must
have fresh outward pressure, positive accepted movement throughout, and a
cumulative increase of at least one in log lambda (a factor of `e`). Both each
update and the entire window must satisfy the objective and all-natural-
parameter tolerances. Other coordinates retain their existing step-trend or
small-residual checks. Inward lower-bound pressure and oscillations do not
establish this evidence. Duplicate states or proposed movement beyond the cap
cannot substitute for actual accepted movement; a positive final step reaching
the cap can count toward the required accepted span.

The result retains finite penalties and raw upper-pressure evidence.
`matched_certified` stays false; the policy does not claim an exact infinite
face, a LAML minimum, or an error bound for inference. Strict stopping is
unchanged. Existing endpoint handling remains available for tiny cap excursions.

Saved artifacts carry an optional immutable map of signed terminal EFS steps,
covering every terminal coordinate in order, including zeros for fixed
penalties. Runtime and replay apply the same rule. A regression demonstrates
that removing a stationary lower-bound coordinate's pressure cannot bypass the
veto. Legacy artifacts without this field remain supported.

Analytic ridge regressions separate genuine infinite smoothing from an
over-smoothed finite optimum despite tiny fitted changes. A real Gaussian fit
checks finite practical predictions and covariance against the strict null-face
solution. Tests also cover moving scale despite stable mean, cumulative drift,
insufficient individual spans, configured one/two/three-update windows, and
fixed zero/cap penalties through serialization.

## GPD negative controls

The [GPD receipt](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_pragmatic_gpd_receipt.json) and
[comparison script](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_pragmatic_gpd_compare.py) preserve the
configurations, input/artifact hashes, source provenance and numerical checks.

The historical synthetic tail fixture has 1,401 excesses and 98 coefficients.
It already enabled practical stopping. With an explicit objective plateau
tolerance of `2e-6` and the existing parameter tolerance `1e-3`, both
implementations retain the same cap fit and stop at `lambda_cap_unresolved`.
The last accepted window passes the objective and outward-coordinate checks,
but its scaled scale/shape changes are 0.005945 and 0.004386. True relative
changes reach 0.595% and 3.157%, respectively. This is remaining fitted-model
movement, rather than lambda growth alone.

Expanding the finite cap to `1e12` while keeping the default tolerances lets
the original capped coordinate become numerically quiet. Both versions then
stop at an ordinary EFS objective rejection after iteration 16. The preceding
window still contains appreciable parameter changes and reversals in other
smoothing coordinates following an accelerated step. One quiet final iteration
is insufficient to establish sustained insensitivity. These are negative
controls, not timing measurements; the existing strict Newton controls remain
the supported solution for this fixture. Relative to that strict solution,
conditional coefficient covariance differs by 33.45% in Frobenius norm at the
original cap and 32.14% at the wider cap. Shape-link conditional standard errors
differ by about 10.59% and 10.46% in relative L2 norm. These are different fitted
solutions with material uncertainty differences, not representation equivalence.

## What the earlier memory limits mean

There are still arrays with values for every observation, including previous
solver states. For one million observations and two distributional predictors,
one retained pair of eta/theta arrays uses about 32 MB; ten such snapshots use
about 320 MB before other allocations.

The coefficient-space limit is different: some tables relate every fitted
coefficient to every other coefficient. One float64 square table costs 8 MB
at 1,000 coefficients and 800 MB at 10,000 coefficients. Several such tables
and their factorizations may be needed. These are scaling considerations,
not reasons to label a practically settled fit unusable.

## Final complete-fit performance comparison

The [independent performance receipt](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_pragmatic_performance_receipt.json)
compares the released v0.31.0 source `8962c452` with final production source
`5f994c8f`. The workload is Gaussian log-severity with 449,000 training rows:
20 repetitions of 22,450 real freMTPL2 severity policies, plus the original
2,494 held-out policies and 92 coefficients. These are replicated rows, not
449,000 distinct claims. The 610,212-policy NB2 refit above is separate numerical
evidence; its elapsed time was not measured in isolation.

| Measurement | Released baseline | Final candidate |
| --- | ---: | ---: |
| Median complete public-fit time | 25.5616 s | 20.9602 s |
| Range across three fresh processes | 24.5150–26.3158 s | 19.4036–20.9965 s |
| Median process high-water RSS | 1,490.48 MiB | 1,007.39 MiB |
| RSS range | 1,486.84–1,496.83 MiB | 1,005.90–1,009.85 MiB |
| Actual observed-assembly dispatch | `distributional-dense-v1` | `distributional-chunked-v1` |
| Observed-assembly chunk rows | dense | 8,738 |

This is an observed **18.0% reduction in median fit time and 32.4% reduction in
median process high-water RSS**. RSS at fit return and at worker completion
was identical in every run; it includes imports and data preparation and does
not isolate solver allocations. Candidate observed assembly selects 8,738 rows
from its 8 MiB budget and 960 estimated bytes per row. This differs from the
4,096-row limit used for derivative/posterior design replay.

All six workers ran serially in BCCBBC order with numerical thread limits set
to one. Agent numerical work and tests had ended before the timed window.
Headroom/Kompress v2 remained active: endpoint CPU samples recorded 0.016–0.062
average cores for it and 0.130–0.209 cores for all external persistent processes.
The first preflight screen refused 0.165 external cores against a 0.1-core
threshold before any timed worker started. A second window explicitly used a
0.25-core screen and recorded 0.179 cores at preflight. These are qualified
local observations with measured background activity, not quiet-machine
approval. The neutral generated summary retains its pending review label;
the independent receipt supplies this narrower assessment.

Raw worker `perf_counter` measurements, JSON/NPZ outputs, source and input
hashes, and exact worker PIDs are authoritative. Headroom tool-response clocks
and compressed text are not timing inputs. Fresh processes retained existing
compiled/disk caches; there was no explicit warmup or cache flush. Endpoint
process snapshots cannot detect every short-lived task, contention peak or
memory-bandwidth/cache effect. Three observations per arm do not establish a
general speed guarantee.

The baseline uses `discrete=False` and the candidate `discrete=True`; other
model, solver and data controls match. This comparison therefore measures a
version and representation change, rather than isolating one optimization.
Arrays repeat bitwise within each arm. Across arms, the maximum held-out
natural-parameter difference is `6.55e-13` and the relative Frobenius covariance
difference is `2.97e-12`; the receipt also records objectives, EDF, likelihoods,
smoothing work and backend dispatch. This observed agreement does not establish
general absence of discretization error. The earlier
[completion evidence](2026-09-c3-c1-completion-evidence.md) separately records
exact compiled-design regressions and continuous-grid sensitivity at 64, 256
and 1,024 bins.

The exact executed controller and summary are now tracked in
[benchmarks/c3_practical](https://github.com/StrudelDoodleS/superglm/blob/4c5783e4/benchmarks/c3_practical/README.md), with hashes,
required source/data layout and commands for a fresh window or historical
replay. Replaying the tracked summary against the retained raw window reproduced
the existing summary exactly without changing any raw artifact.

## Final validation and remaining limits

Final production source `5f994c8f` passed **11,546 tests with 174 skips** using
`SUPERGLM_REQUIRE_DATA=1`, the local freMTPL2 files and mpmath numerical oracles.
Dataset absence therefore could not silently pass the real-data suites.
The final source also passed Ruff, formatting, lock consistency, dependency
checks and `run_test.py`. The latter's output matches the previous run,
including its existing U-shape diagnostic. No additional production changes
followed those checks.

Raw validation logs remain under
`.benchmark-artifacts/c3-practical/validation/`. The full pytest log SHA256 is
`e3e009ff7b6105b56e69839533970aeab84fada6acfdaf0fb3208a5ad7ee9d82`;
the `run_test.py` log SHA256 is
`be730b9662683cba47a4646874847861d3792b16b9c175d761ff427f484f04ac`.
Independent reviews covered Newton recovery, outward stopping mathematics,
finite-NB2 high-precision derivatives, final stationarity/covariance and the
performance receipt. The original uncommitted strategy files in
`.worktrees/roadmap-dossier` remain byte-identical.

Practical completion certifies the recorded stability rule, not an exact
infinite-penalty face or a global minimum. The final NB2 example meets the
configured stationarity contract; an exact Poisson active face remains
unsupported. The coupled GPD negative controls still require the existing
strict Newton settings. Observation histories and dense coefficient-space
matrices retain the memory limits described above. Cross-predictor penalties
remain deferred by the user's scope decision. These limitations are explicit
follow-on choices; they do not invalidate a practically settled fit.
