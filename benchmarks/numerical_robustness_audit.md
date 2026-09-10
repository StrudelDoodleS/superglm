# Numerical robustness audit

This audit follows the Gamma convergence repair. It examines shared numerical
decomposition authority and the thresholds used by callers. It does not claim
that all numerical failure modes have been covered.

The [final family-arithmetic addition](family_arithmetic.md) records the scalar
working-weight and Pearson fixes, four LSS variance fixes, current verification
and new complete-fit comparisons. The earlier checkpoints below retain their
own source digests and measurements.

## Tensor checkpoint before the family-arithmetic addition

The final candidate restores the complete tensor fit to 12.10698 seconds versus
11.27126 seconds for the resumed baseline, measured with five alternating pairs.
It retains the strict accuracy policy. The remaining difference is 0.83572
seconds, or 7.41%; median process peak RSS is 446.76 versus 404.67 MiB.
All ten fits converge in twelve smoothing iterations. Maximum prediction
difference is `4.14e-11`, and relative objective difference is `4.37e-10`.

The source digest is
`c7ee44778b9bf8cd424b48a21dbca6ba6ffeb8c88ef982ec0f743d86668c7aaf`.
The final profile matches the complete output payload of all five timed
candidate fits. Source, drivers, helper files and input data also match.
The [profile report](solver_repair_profile_comparison.md#final-tensor-candidate)
explains the changes and the remaining cost. Final receipts are preserved under
`.superpowers/sdd/2026-09-10-numerical-resume/artifacts/complete-fit-tensor-v4-final/`
and `artifacts/profile-tensor-v4-final/`.

The fixes reuse authenticated raw penalty geometry, avoid unused QR outputs,
move eligible strict products to exact float64 slice products, and use batched
native operations for ordinary tensor Gram and cross-Gram calculations. A
native initial-factor proposal is accepted only after the unchanged strict
certificates pass; numerical refusal retries the original arithmetic. No
larger-error experimental policy was adopted.

Final combined verification has 14,106 passes and 87 recorded skips, including
all 88 required real-data checks. The four-shard run first returned 14,101
passes and three failures. One old dispatch assertion still expected the
uncompressed SSP fallback; the updated test positively requires the batched
route. All 89 support-compression and cross-dispatch tests then passed. Two
Tweedie failures came from concurrent mutation tests sharing a Numba disk cache;
all 113 unchanged Tweedie tests passed with a private cache. Both 16-worker
controls also passed separately. The original failure receipts are retained.
Production source did not change during or after the full run; its only later
test change is the dispatch expectation. Ruff, formatting, lock consistency,
dependency consistency, `git diff --check` and `run_test.py` pass.

The untimed dispatch execution matches all five timed candidate payloads. It
records 153 admitted strict dyadic products, 2,448 completed float64 slice
products, 46 wide fallbacks, twelve native factor proposals, 94 dense tensor
Grams and 282 batched SSP cross-Grams with no legacy cross-column calls.
Actual SciPy sparse kernels are recorded separately. Native pool sampling
reports one thread and no observer errors. This observes executed operators
and their source-bound dispatch conditions; it does not intercept BLAS symbols.
See `artifacts/tensor-v4-untimed-final.json` and
`artifacts/verification-tensor-v4/closure.json` beside the timing receipts.

The tests and timings below retain their original source attribution. In
particular, the older 14-case timing table is not a fresh timing claim for this
final source. The tensor benchmark uses an explicit REML tolerance of `1e-6`.

## Earlier verification checkpoints

The reproduced rank, scalar stopping and multi-penalty defects below now have
independently reviewed repairs. A later complete tensor fit exposed a separate
support change during terminal SSP reconstruction; its repair also passed
review and the complete fit. The earlier verification result and the completed
14-case performance checkpoint are below. The subsequent SCOP reuse repair
has separate passing verification and measured before/after results. The observations record
the failures and intermediate review findings that motivated these changes.

The frozen numerical checkpoint, before the later SCOP cache change, passed
the complete non-browser suite:
13,941 tests passed, with no failures and all 88 required real-data checks
included. Two worker-limit skips passed separately with 16 workers, bringing
combined verification to 13,943 passes and 87 optional/inapplicable skips.
Source and tests remained unchanged during that run. The package source digest is
`3292dc721a89eeae0d84b77b73e78271969e3b394b42256ececaee2b3c8f6c8a`.
The subsequent benchmark-only observer correction passed all 39 directly
affected tests and independent review; every production file still matches
the full-suite snapshot. A cold Gamma execution also passes with Numba
initially uninitialized and an empty compilation cache.
The five-pair performance comparison for that production snapshot is complete.
It predates the new SCOP caller cache; its results must not be attributed to
that later source change.

The SCOP cache candidate is `6cd33c35`. Its only production delta from that
checkpoint is `src/superglm/reml/scop_efs.py`, with one new regression file.
All 564 affected SCOP/shape/monotone tests pass on Python 3.13, and the 35 new
tests also pass on actual Python 3.12. Four mutations detect discarded caches
and incorrectly reused targets. This is targeted verification of the new
delta; the 13,943-pass full-suite result above belongs to the earlier source.
Independent review accepts the SCOP change after 34 controls on each Python
version and analytic baseline, range and dispatch replays.

## Broader audited paths

The expanded audit reproduced failures beyond the original convergence and
rank callers. The following repairs have focused regressions and independent
acceptance, including the final SZ public-rank admission correction.
Final integration and complete-fit performance remain separate requirements.

| Path | Reproduced failure and correction |
| --- | --- |
| Rank and structured factors | Pairwise averaging could erase small entries beside unrelated huge entries; scaled diagonal checks could discard negative curvature; structured determinants and solves could lose finite results. The repairs preserve the represented operator and its rank/error authority. |
| Raw SSP Gram and screening kernels | Reassociation or premature sparse aggregation could destroy a finite operator, and averaging could overflow. Extreme-input regressions use exact represented operators and check actual dispatch separately. |
| Constrained QP and SCOP | Unit-dependent checks changed feasibility/convergence or admitted uphill steps. A homogeneous full-rank active face now produces its exact zero representative while retaining the original KKT checks. Near-rank cases retain honest score-based refusal. |
| Sum-to-zero factors | Internal congruence scaling could bypass the public model's numerical-rank decision. A compact original-operator residual bound now certifies admission; unresolved small systems use shared Gram authority, and unresolved wide systems refuse before dense allocation. |
| Scalar families and LSS kernels | Intermediate overflow, underflow and cancellation lost representable densities, deviances or weighted derivatives. A finite generalized-gamma shape also disagreed with its derivative channels. Range and high-precision probability-law controls cover the corrected formulas. |
| LSS assembly and chunking | Curvature averaging and three-factor derivative products lost finite entries; chunking swallowed a derivative-order contract error. Finite entries are preserved and the hard contract error propagates. |
| Posterior calculations | Error trust failed to cover the full Hessian, and unit-dependent cutoffs dropped positive modes. The repair checks the full enclosure and preserves representable inverse and sampling actions. |
| Penalty consumers | Copying and endpoint extraction discarded retained target/error evidence; assessments could substitute a different target or lose an earlier objective's uncertainty. Explicit ownership and original-error receipts now preserve that evidence. |

The audit also records boundaries without claiming a demonstrated repaired
public-fit failure: extreme Hessian cross-traces outside ordinary REML bounds,
duplicated generic Gamma expressions, and a source-only negative-binomial
profiling formula. The result-lifetime inventory identifies further possible
reuse; it does not establish a speed benefit without an executed comparison.
This work does not certify every numerical branch or every possible input.

## Executed comparisons with other libraries

The original Gamma book fits succeed in mgcv, including its unbounded
log-dispersion parameterization; see [the matched model comparison](lss_convergence_repair.md#mgcv-comparison).
Those fits establish that our original failure needed repair. Separate
adversarial arithmetic experiments also found limits in the other libraries:

| Executed library and case | Observed result |
| --- | --- |
| mgcv 1.9-3, roots `(1,0)`, `(1,1e-14)`, `(0,1)` weighted by `(1e30,1e30,1)` | `gam.reparam` reports log determinant `69.770699970381315`; the represented-root oracle gives `73.7025256031056369`, a determinant ratio of about 51. The first and second log-weight derivatives also disagree. Root orientation, support preprocessing and all six component orders were checked. |
| mgcv 1.9-3, three coordinate roots weighted by `(1e300,1,1e-300)` | Some Hessian entries are NaN although the independent reference Hessian is finite. Ordinary diagonal and alias controls pass. |
| mgcv 1.9-3, matched public SZ design and fixed penalty | Both `gam` and `magic` agree with the repaired rank: 69/70 at final-level weight `1e-20`, and 70/70 at `2^-20`. Ordinary-weight predictions agree within `4.38e-15`; the almost weightless level differs by as much as `2.355`, so rank agreement is not representative agreement. |
| pyGAM 0.12.0, duplicate linear columns rescaled by `1e-6` with no requested penalty | The fixed numerical ridge changes the requested least-squares optimum: maximum prediction difference is about `0.49574`. The returned fit agrees closely with the independently evaluated ridge problem. |
| pyGAM 0.12.0, near-alias linear columns | The numerical ridge suppresses a resolved response direction; maximum prediction difference from the requested optimum is about `0.99805`, despite zero terminal coefficient movement. |
| pyGAM 0.12.0, uniformly positive weights `1e-18` | Its working-row cutoff removes every row and fitting raises `OptimizationError`; the intended unpenalized least-squares problem remains well defined. |
| pyGAM 0.12.0, large spline penalties | At lambda `1e8` and `1e12`, retries add about `0.009 I`; at `1e16` and `1e20`, penalty factorization refuses before incorporating the full-rank data design. |

Eleven public pyGAM fits and five matching SuperGLM linear fits were executed.
The five SuperGLM prediction errors were at most `6.67e-16` on those controls.
pyGAM's fixed-lambda coefficient fit **passes** the ratio-51 geometry control;
it has no matching LAML determinant-derivative API in the inspected paths.
These experiments distinguish changing a requested problem, inaccurate
arithmetic and honest refusal. They do not establish a general library ranking
or identical failures in full distributional models. The methods were derived
independently; no library implementation was copied or translated.

The SZ comparison executes eight fixed objectives through each mgcv interface:
the original 480-by-70 fixture, its moderate-weight control, and six two-level
controls spanning three common curvature units. Binary round trips verify
the effective design, penalty, response, weights and offset. The tiny-weight
case has weighted prediction RMS difference `4.82e-11`; both implementations
return the same rounded original objective. These are stable-observable
comparisons, not certified forward accuracy in the discarded direction.
The full report and exact receipts are preserved in
`.superpowers/sdd/2026-09-10-numerical-resume/sz-mgcv-comparison.md` and
`sz-mgcv-receipts/` beside it. Matched constrained QP and joint SCOP external
executions remain explicit coverage gaps. The earlier pyGAM cases do not
cover those constrained methods or this SZ fixture.

## Complete-fit performance checkpoint

The following medians come from five alternating baseline/candidate pairs per
case: 140 complete fits in fresh processes, with one native thread and no
solver or native-pool observers inside the measured fit. First-use work stays
inside the fit clock; this is not a claim that every disk compilation cache
was empty. Separate untimed executions check actual dispatch against the
saved timed outputs. Team numerical work was paused for the timing batch.

The resumed baseline is `167873e1`; the repaired checkpoint is `3292dc72`.
The baseline already contains the original Gamma convergence repair. These
numbers therefore compare the later numerical work with that resumed state,
not with unmodified master or the original failing implementation.

| Complete fit | Baseline seconds | Repaired seconds | Ratio | Baseline / repaired peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: |
| Scalar tensor | 11.8574 | 26.9870 | 2.276 | 404.6 / 438.9 |
| Gaussian shared penalties | 0.1203 | 0.2037 | 1.694 | 334.0 / 335.9 |
| Gamma shared penalties, discrete | 0.4800 | 0.6144 | 1.280 | 375.9 / 377.5 |
| LSS tensor shared penalties, discrete | 0.8684 | 1.6554 | 1.906 | 394.9 / 396.2 |
| Scalar shared penalties | 0.2097 | 0.2693 | 1.284 | 374.8 / 376.4 |
| Scalar full rank | 1.5960 | 1.6014 | 1.003 | 694.4 / 694.9 |
| Scalar rank deficient | 0.2678 | 0.2692 | 1.005 | 406.6 / 407.1 |
| Gamma book, dense | 4.6315 | 4.9472 | 1.068 | 520.9 / 522.6 |
| Gamma book, chunked | 6.1163 | 6.1873 | 1.012 | 507.4 / 508.4 |
| Inactive constrained QP | 0.0382 | 0.0521 | 1.361 | 334.7 / 336.3 |
| Binding constrained QP | 0.0428 | 0.0597 | 1.394 | 334.8 / 336.1 |
| Single SCOP | 0.0347 | 0.0851 | 2.452 | 334.1 / 335.6 |
| Joint SCOP, discrete | 0.3400 | 1.3410 | 3.944 | 375.9 / 377.7 |
| Sum-to-zero | 0.4144 | 0.5142 | 1.241 | 382.2 / 383.8 |

Rank fixtures, both Gamma book fits, inactive QP and both SCOP fixtures retain
bitwise-identical coefficient/prediction and objective outputs across this
comparison. The tensor prediction difference is at most `3.86e-11`; its
relative objective difference is `4.54e-10`. Shared-penalty coefficient
differences are at most `1.01e-10`. These are observed comparisons, not forward
accuracy guarantees for near-rank coefficients.

The joint SCOP fixture reaches its 40-iteration smoothing limit on both sides.
Its coefficient fits converge, with the same 42 coefficient fits and 140
Newton steps, but smoothing convergence remains false. Finishing the benchmark
does not turn that unresolved smoothing result into success.

Eight cases exceed both 10% slowdown and three baseline median absolute
deviations. Both sides of all eight were profiled; each complete profile
output matches all five corresponding timed outputs. Full timings, dispersion,
memory, outputs and source identities are in
`.superpowers/sdd/2026-09-10-numerical-resume/artifacts/complete-fit-final-v3/`.

### Measured static reuse

A separate five-pair ablation changes only the accepted support-basis memo
and repeated input-error row reuse. The tensor median falls from 28.0872 to
26.8276 seconds, a 4.49% reduction. Median process peak RSS changes from 436.6
to 438.4 MiB; the retained basis arrays occupy 1,206,016 bytes for this support.
All ten complete numerical output payloads are identical. The receipt directory
is `artifacts/static-reuse-five-pairs/` beside the checkpoint above.

This removes demonstrated repetition without changing the rank or error
criteria. It does not explain away the remaining tensor regression.

### SCOP caller reuse repair

The SCOP slowdown was repeated construction of fixed latent penalty geometry.
Retaining the authenticated target per fit removes that repetition and uses
the existing determinant/derivative summary API without constructing an unused
dense inverse. Coefficient-dependent work remains current.

A new five-pair comparison isolates this single production-file change:
`3292dc72` before the cache, `6cd33c35` after it. The single SCOP median falls
from 0.07564 to 0.03810 seconds; joint discrete SCOP falls from 1.21276 to
0.33774 seconds, a 72.2% reduction. Peak RSS is essentially unchanged:
335.57/335.71 MiB and 377.53/377.38 MiB respectively. All ten complete outputs
per case are identical. This is a separate batch from the 14-case table;
its before values are freshly measured, not taken from that older batch.

Complete-fit profiles confirm the proposed reduction: joint SCOP changes from
250 support builds and 166 full evaluations to two support builds and two
summaries. Single SCOP changes from 11 support builds and seven full evaluations
to one support build and one summary. Newton steps, coefficient fits and the
joint fixture's existing smoothing-iteration limit stay the same. The
[profile comparison](solver_repair_profile_comparison.md#repeated-scop-penalty-construction)
records timings, dispersion, exact output checks and preserved receipts.

## Why the earlier tensor fit was slower

The final source now has a fresh [paired call-stack comparison](solver_repair_profile_comparison.md).
Its five-pair tensor medians are 11.857 versus 26.987 seconds. Profiling both
frozen sources shows nearly identical Gram, moment and coefficient-fit work;
penalty evaluation grows from 1.065 to 13.294 diagnostic seconds. Complete
profile outputs equal every corresponding timed fit. This final comparison
supersedes the earlier tensor snapshots below. The complete 14-case timings
and separate static-reuse comparison are reported above.

### Earlier measurements and reuse decisions

A quiet complete-fit preflight compared the resumed baseline with source digest
`905ad599bd6ab363caffd9dcb01047ff5258d10bc1a6c399853d6a1ee5da5b08`.
The scalar tensor model has 2,000 rows, 255 coefficients and a 225-column tensor
penalty with retained rank 224. One baseline/candidate pair took
10.756/30.532 seconds, with process peak RSS 405.2/448.1 MiB. Both converged;
the maximum prediction difference was `3.86e-11`. This single preflight
identified a slowdown; final acceptance requires five alternating pairs.

Separate diagnostic profiles explain this case's added work:

| Work | Baseline | Repaired candidate |
| --- | ---: | ---: |
| Smoothing iterations | 12 | 12 |
| Coefficient solves | 14 | 14 |
| Centered-system builds | 46 | 46 |
| Moment evaluations | 94 | 94 |
| Group Gram calculations | 282 | 282 |
| Penalty evaluations | 47 old evaluations | 17 certified summaries from 141 requests |

The candidate's summaries account for 17.20 seconds in the profile. Extended
precision matrix products dominate: the enclosed signed-product helper has
7.74 seconds of self time. Native QR accounts for 0.077 seconds, and this fit
calls neither compensated Dot2 nor dense inverse materialization. These
profile timings locate work; instrumentation makes them unsuitable as a
complete-fit speed comparison. Nested cumulative times are not additive.

The result cache is already avoiding repeated evaluations. The immediate
implementation defect is narrower: the same enclosed matrix product and its
absolute-product bound are recomputed during materialization checking, and
the duality product is then computed again by the evaluator. Reusing their
existing values and evidence within that evaluation has passed independent
review at core `e7366f59`. Accuracy targets, rank decisions and the maximum
number of corrections stay unchanged. The result-lifetime audit records
further discarded results and unused materialization along the fitting path.

Three alternating interim pairs after that reuse change gave baseline times
`10.9045, 11.0389, 10.8066` seconds and candidate times
`27.5402, 27.7087, 28.2547` seconds. Their medians are **10.9045 versus
27.7087 seconds (2.541x)**; median absolute deviations are 0.0979 and 0.1685
seconds. Median peak RSS is 405.3 versus 437.0 MiB. Each fit ran in a fresh
process with one native thread, alternating order, frozen sources, matching
input hashes and the same driver. Team numerical jobs were paused, but an
unrelated single-thread CPU workload remained on the 16-logical-CPU host.
These are interim paired measurements, not the final fully quiet acceptance
run. Receipt metadata records that qualification.

The candidate digest is
`536c24f65ccca6b6b7e9762d2e9257085a645c7f0e9088b1295c1bd8ce33a15c`.
All three pairs have the same numerical differences: maximum prediction
difference `3.86e-11`, deviance difference `5.80e-10` and EDF difference
`8.03e-10`; both sides converge with coefficient rank 255. The candidate's
outputs also exactly match a separate plain diagnostic profile. No final
claim about the newer broad-audit repairs follows from this older snapshot.

The fresh profile still has 141 requests, 17 summaries, 12 outer iterations,
14 coefficient solves and 94 moment evaluations. Summary evaluation accounts
for 15.13 diagnostic seconds; `_matmul_enclosed` has 7.20 seconds of self time,
reference root actions 2.61, and direct candidate construction 1.90. Context
construction takes 2.92 cumulative seconds. These overlapping profile values
identify the remaining work; they are not additive fit-time measurements.
The subsequent reuse repair has passed independent review. It memoizes the
unchanged support-basis Gram and its error bound, with owner, basis, precision
and callable invalidation. It also evaluates exactly repeated tiny input-error
rows once and copies the bounded result. For three captured width-225 support
evaluations, six error products shrink from 210 rows to one row each:
63,504,000 multiplication terms become 302,400. Two repeated basis Grams and
their magnitude products are also removed. Executed before/after controls
preserve complete outputs and certificate bounds exactly. The basis memo
retains 1,206,016 array bytes for this support; complete-fit time and peak RSS
for this newer source remain to be measured.

Preflight receipts are in
`/tmp/superglm-numerical-resume-9106rchs/preflight-final-tensor/`; profiles,
caller counts and the before/after work comparison are in
`/tmp/superglm-numerical-resume-9106rchs/profile-final-tensor-consumer/`.
This diagnosis applies to the scalar tensor case. The earlier contended
measurements below cannot establish runtime or explain a different fit's cost.

The interim repeated receipts and comparisons are in
`/tmp/superglm-numerical-resume-9106rchs/reuse-tensor-three-pairs/`;
the new plain profile is in `profile-reuse-tensor/` beside that directory.

## What went wrong

The reproduced failures have different causes. They must not all be treated
as an overly strict REML stopping rule.

- Weak fixed starts let flexible Gamma scale predictors nearly isolate an
  individual claim. In the reproduced weak-start case the next mode can lie
  beyond representable dispersion. The automatic information-scaled start
  avoids that path on the tested uncapped book. Explicit weak starts remain
  available and must report their unresolved result honestly.
- Small coefficient or objective movement could report convergence with a
  large unresolved score. A retry could also retain an earlier success flag.
  Fresh score or observed Newton-decrement evidence now governs that result.
- Exact-face replay required bitwise coefficient equality. A harmless
  projection change of about `7e-18` could invalidate an otherwise consistent
  endpoint. The repair admits bounded projection roundoff while preserving
  the stationarity and provenance checks.
- Several numerical cutoffs depended on arbitrary units. The scalar block
  eigenvalue floor can erase positive curvature, the radial root tolerance
  applies to a dimensional quantity, and the final global score scale can
  hide an unfinished feature update behind the intercept. The LSS bracket
  had a separate absolute derivative shortcut.
- Shared decomposition arithmetic could invent bad conditioning, lose
  rank, overflow a finite inverse, or bypass required factor verification
  after zero padding. These are arithmetic and authority errors.
- Multi-penalty roots, determinants and derivatives could retain different
  directions. A three-component recursion also has a dimension error. These
  are algebraic inconsistencies, not reasonable conservative stopping.

The first three causes explain the original reproduced Gamma behavior.
The other probes establish wider defects but do not establish that every
ordinary fit encounters them or that every past estimate is wrong.

The base is `66141d2873afc03287ca924f4592e33482281a0e`. The original four probes
ran before any changes to `src/superglm/solvers/rank.py`, using float64,
NumPy 2.5.2, SciPy 1.18.0 and one OpenBLAS native thread. The six existing
rank/authority suites passed 217 tests while these cases remained uncovered.

## Confirmed decomposition defects

| Case | Unfixed behavior | Required contract |
| --- | --- | --- |
| Append a structural zero column to a near-collinear Gram | Factor verification changes from required to unnecessary | Identical active geometry must have identical verification requirements |
| Scale an identity factor by `1e-170` or `1e160` | Column norms become zero or infinity, and the factor is reported as rank zero | Finite, condition-one factors must retain their rank and factor-space predictions |
| Invert `diag(1e-308, 1)` | Solve is finite but inverse symmetrization produces infinity | A representable inverse must agree with inverse action and reconstruct the identity |
| Gram input `[[0, 1], [1, 0]]` | Zero diagonals cause nonzero rows to be dropped, returning authoritative rank zero | The PSD route must refuse the resolved indefinite matrix |

The first defect is a logic error in the factor-verification predicate. The
extreme-scale cases are arithmetic failures, not evidence of bad conditioning.
The indefinite example is invalid input for a PSD decomposition and should
be refused. These findings do not establish that ordinary fits encounter
every one of these paths.

## Reproduction

Run in the development environment with one native thread:

```python
import numpy as np
from superglm.solvers.rank import (
    decompose_factor,
    decompose_gram,
    decompose_gram_if_authoritative,
    needs_factor_certification,
)

factor = np.array([[1.0, 1.0], [0.0, 1e-7]])
gram = factor.T @ factor
for matrix in (gram, np.pad(gram, ((0, 1), (0, 1)))):
    result = decompose_gram(matrix)
    print(result.rank, needs_factor_certification(result),
          decompose_gram_if_authoritative(matrix) is not None)

for scale in (1.0, 1e-170, 1e160):
    factor = scale * np.eye(2)
    result = decompose_factor(factor, retain_factor_solve=True)
    beta = result.solve_factor_rhs(factor @ np.array([1.0, -2.0]))
    print(scale, result.rank, factor @ beta / scale, result.log_pdet)

for result in (
    decompose_gram(np.diag([1e-308, 1.0])),
    decompose_factor(np.diag([1e-154, 1.0])),
):
    print(result.solve(np.array([1.0, 0.0])), result.pseudo_inverse())

print(decompose_gram_if_authoritative(np.array([[0.0, 1.0], [1.0, 0.0]])))
```

## Numerical references

LAPACK documents [safe scaling for Euclidean norms](https://netlib.org/lapack/explore-html/d1/d2a/group__nrm2_gab5393665c8f0e7d5de9bd1dd2ff0d9d0.html).
Its [DPOCON documentation](https://www.netlib.org/lapack/explore-html/d3/dfd/group__pocon_gaadb1e19663b71521d30b5b5bbe3091f8.html)
describes a condition estimate, which must not be confused with a rigorous
error enclosure. [DGESVX](https://www.netlib.org/lapack/explore-html/d5/dbe/group__gesvx_ga82173a93234afc15d70b64233b3e5bc8.html)
separates conditioning, backward error and forward-error estimates.

## Caller thresholds

The scalar radial block solver has a confirmed fitting defect. Its cutoff
`eps * width * max(max(abs(eigenvalues)), 1)` discards valid small positive
curvature. An unpenalized Gaussian fit with `X = [[-c], [c]]` and `y = [-c, c]`
returns coefficient one at `c = 1`, but zero at `c = 1e-8`; both report
convergence. Common scaling of feature and response preserves the true
coefficient optimum. The defect is in `solvers/pirls.py`.

The LSS beyond-cap bracket also has a unit-dependent root shortcut.
`bracket_beyond_cap(a, -a, lambda u: a * (1-u), log_span=2)` returns the true
root one at `a = 1`, but returns the far endpoint two at `a = 1e-11`.
The absolute `1e-10` derivative threshold causes this. The caller evaluates
`phi = -lambda * dF/dlog(lambda)`. Changing penalty units and inversely changing
lambda preserves the statistical objective and root in log-ratio coordinates,
but rescales phi. This demonstrates a bracket-policy defect; it does not by
itself establish a particular complete fit's false stationarity.

Other findings remain policy or coverage concerns. The distributional penalty
symmetry tolerance has an absolute floor and therefore unit-dependent input
acceptance. Symmetrizing a supplied matrix preserves its quadratic form, so
this is not a demonstrated model change. Dense/chunked initialization authority,
the constrained KKT gate and SCOP absolute objective checks need targeted
evidence before changing behavior.

## Candidate repair review

The first four decomposition corrections pass 225 focused tests. The new eight
tests yield seven failures and one pass against the unmodified rank module.
Those are executed red/green results, not a claim of complete robustness.

Astra review required an internal rank-policy version update because safe
norms change the retained columns on an identical deficient input. This is
now internal policy version four; the package version remains untouched.

Review also found an existing signed-decomposition defect. The matrix
`diag(1, [[0, 1], [1, 0]])` has rank three and condition number one. The signed
route returned rank two and lost the first coordinate. The initial regression
mistakenly expected rank two. The corrected signed scaling and regression now
retain all three directions.

The scaling step causes that defect. It produces column scales
`[1, sqrt(eps), sqrt(eps)]` and equilibrated eigenvalues
`[-1/eps, 1, 1/eps]`. The resulting cutoff is three, so the unit eigenvalue is
discarded. The original matrix was well conditioned. Signed scaling must not
introduce this loss.

## Multi-penalty support

An independent audit found an inconsistent support decision in
`reml/multi_penalty.py`. For `P1 = diag(1, 0)`, `P2 = diag(0, 1)` and weights
`[1e12, 3]`, the returned penalty root retains both directions, but the reported
rank is one. The log determinant is `log(1e12)` instead of `log(3e12)`, and its
derivatives with respect to the log weights are `[1, 0]` instead of `[1, 1]`.
The same calculation is correct at weights `[1, 3]`.

The final global eigenvalue cutoff loses a resolved penalty direction after
the recursive transformation. Scalar and LSS smoothing paths both call this
code. This probe establishes a mathematical inconsistency, not its prevalence
in real fits. It does not explain the original Gamma example, which used
single-penalty groups.

A three-component coordinate example with weights `[1e24, 1e12, 1]` also
raises a matrix-dimension error inside the recursive calculation. The repair
must cover more than the two-component example.

## Verification before the numerical repair

The latest affected LSS suite passed 2,172 tests, including all three guarded
real-data refits. Two opt-in performance tests skipped and one slow test was
deselected. The scalar core, composite optimizer and structured solver suites
passed 226 tests. These results include the saved beyond-cap bracket candidate,
but do not cover all of the newly identified defects.

Independent scoped review approved removal of the bracket's absolute
derivative shortcut. Direct old/new probes reproduce the wrong-root and false
root-acceptance failures. Exact-zero endpoints still work. A separate existing
limitation remains: `max_evaluations` is passed to the root finder as an
iteration limit, so it is not a strict limit on callback calls.

Complete-fit checks compared the current candidate with saved pre-rank-change
receipts. The deficient scalar baseline imported the unmodified source tree.
The other baselines already included the original LSS convergence repair.

| Fit | Before / candidate peak MiB | Numerical and dispatch check |
| --- | --- | --- |
| Scalar, 6,000 training rows, 1,626 coefficients | 694.9 / 694.5 | Recorded outputs unchanged; Cholesky, full rank |
| Scalar, 400 training rows, 347 coefficients | 420.8 / 408.7 | Recorded outputs unchanged; QR/SVD, rank 344, three representative zeros |
| Gamma LSS, 21,124 claims, all nine features in mean and scale | 519.0 / 520.6 | Identical saved prediction CSV; dense backend, rank 146, no Fisher fallback |

Wall time is unmeasured: these single runs used one native thread under
concurrent machine load. Peak memory is the process high-water mark, not solver
scratch allocation. The scalar comparison covers the rounded
numerical fields recorded by its existing script, not equality of every
coefficient. Receipts and LSS predictions are in
`/tmp/superglm-rank-robustness.gRMtsc/`. Fresh Ruff and format checks passed,
and `git diff --check` found no whitespace errors.

The user approved the numerical blueprint and requested Astra max for
implementation and independent review. That work corrected the signed case,
internal policy version, scalar block normalization and consistent
multi-penalty calculations. All candidate edits remain uncommitted.

## Rank review and scalar design follow-up

The version-four rank candidate passed 243 focused tests and a separate
56-test integration run covering rank, initialization, coefficient stopping
and all three guarded uncapped Gamma refits. Independent review still found
two finite-output failures, so those green results did not close the task.

For `A = [[1e308, 1e-308], [1e-308, 1e-308]]` and RHS `[0, 1]`, row-first
sequential scaling erases one off-diagonal entry before symmetrization. The
candidate solve has componentwise backward error `1/3`. Dividing first by the
smaller coordinate scale preserves that coupling and the smallest-subnormal
determinant regression in the reviewer's independent probes.

For `F = 1e-309 * I` and response `F @ [1, -2]`, a reciprocal basis overflows
although the requested coefficient answer is finite. The scoped repair also
covers a deficient factor with finite solve but invalid stored bases. Safe
scaling must preserve both the solve and the subspaces used by later checks.
Both fixes subsequently passed implementation and independent review.

The scalar design uses a homogeneous per-block proximal residual and an
explicit arithmetic allowance. A unit-floor denominator was rejected in the
design stage because it accepted mapped coefficients `[1, 0.6]` instead of
`[0.6, 0.8]` after response and threshold scaling. A purely relative check was
also rejected because it exhausted iterations on a well-conditioned
zero-signal problem whose remaining score was assembly roundoff. The approved
bounded implementation plan covers both cases and the custom proximal path.
