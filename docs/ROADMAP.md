# Roadmap

Last strategic review: **2026-09-09**. Starting baseline: `origin/master` at
`8962c452` (published v0.31.0); C3/C1 implementation through `1a8952a4`, with convergence follow-through at
`5f994c8f`.

This is **directional project state, not an implementation specification**,
delivery commitment, or authorization to start a capability. Scope implementation
separately. If code, tests, benchmarks, or user evidence contradict this roadmap,
report the discrepancy and propose an update; do not change the implementation
to satisfy an outdated assumption. Explicitly scoped user work takes precedence.

The [research dossier](research/2026-09-superglm-feature-roadmap-dossier.md)
contains the candidate details and literature (C1–C14 below). Its recommendations,
estimates, and original living-document header are prior analysis; this file
records the current ordering. Update the review date, revision, and evidence when
a capability lands or a gate changes; retire completed work from the queue.

## Current position

The priority is useful, trustworthy distributional modelling on CPU, building on
the scalar engine's existing performance and interpretability.

- **Scalar foundations already exist:** exact row-support compression, discrete
  REML, structured random effects and FS/SZ smooths, shape constraints, and
  [interaction screening](guide/screening.md). The [historical benchmark record]
  includes a 678,013-row fit at a 0.805 s median; its timing certification is
  explicitly disabled pending a reproducible current machine profile.
- **LSS is substantial already:** nine built-in families, coupled observed
  curvature, fused family kernels, reused dense designs, cheap rejection of
  backtracks, and joint inference. [PR #376] records a 100,000-row, 192-coefficient
  Tweedie fit at 9.712 s, seven outer iterations, dense dispatch, and about
  1.08 GiB whole-process peak RSS. This is an earlier single-pair receipt ending
  at `practical_plateau`, not a current-head certified benchmark or an assembly
  timing. Scalar and LSS timings describe different workloads.
- **Existing smoothing and inference:** [smoothing](models/distributional.md#how-smoothing-parameters-are-chosen)
  has safeguarded EFS, optional Newton/BFGS, stationarity evidence, and exact
  infinity-face decisions. [Inference](models/distributional-inference.md)
  already includes callable posterior bounds, supported tail functionals,
  smoothing-uncertainty correction, proper scores, and portfolio simulation.
- **Production grouped LSS is implemented:** public `discrete=True` supports
  observed-curvature families, including Tweedie and NB2, with actual backend
  telemetry. Smoothing and posterior derivative replay use bounded design
  blocks; categorical spline chunks reuse row lookups. Existing [grouped
  assembly][Grouped assembly] and [rectangular cross-products] remain the
  underlying architecture.
- **Real gaps remain:** LSS shape constraints warn and fit unconstrained;
  cross-predictor penalties are unsupported. Dense coefficient factors, family
  derivative arrays and retained coefficient histories still limit size. Some
  smoothing fits can still lack strict certification. The tested large real-book
  NB2 configuration now reaches configured stationarity after a finite-NB2
  numerical-range fix.

## Implemented foundation: C3 + C1

The user explicitly selected C3 and C1 on 2026-09-08, overriding the previous
C5-before-C1 ordering. The [implementation plan](superpowers/plans/2026-09-08-c3-c1-completion.md),
[design](superpowers/specs/2026-09-08-c3-c1-completion-design.md) and
[completion evidence](research/2026-09-c3-c1-completion-evidence.md) record the
bounded scope and validation. Original uncommitted strategy inputs remain
unchanged in `.worktrees/roadmap-dossier`.

**C3 follow-through:** the original correlated Tweedie and GPD EFS failures
reproduce on v0.31.0. Existing strict EFS plus Newton controls reach endpoints
passing the current stationarity checks from three smoothing starts on both
baseline and implementation. [Source-bound stress receipts](research/2026-09-c3-stress-evidence.md)
retain the failures, terminal checks, independent reference results and measured
prediction/conditional-SE sensitivity. No new optimizer, shape penalty, relaxed
tolerance or acceptance rule was needed. These checks establish the existing
first-order numerical contract, not a rigorous enclosure of every derivative
error or a local/global minimum guarantee. Unresolved reference probes remain
visible.

The [pragmatic convergence follow-through](research/2026-09-pragmatic-convergence.md)
adds practical stopping across substantial outward lambda movement, retained-fit
EFS recovery when Newton derivatives are unavailable, and a bounded finite-NB2
low-mean range extension. The last change removes a numerical guard that blocked
an improving coefficient step on one policy. The 610,212-policy NB2 book now
reaches configured coefficient and LAML stationarity in nine outer iterations.
Its corrected uncertainty differs materially from the earlier stopped fits.
This does not add an exact Poisson face or a global-minimum guarantee.

**C1 production route:** public observed chunking, bounded derivative replay and
cached categorical row extraction are implemented and regression tested.
Complete smoothing fits cover signed geometry, weight semantics, interactions,
prediction/covariance and serialization. The insurance evidence shows a memory
benefit on 449,000 synthetic rows formed by repeating 22,450 real severity
policies; the smaller real book does not save RSS. The initial large real-book
NB2 comparison stopped without strict certification. Its Newton derivative
recovery and finite numerical range have since been corrected, and the same
workload reaches configured stationarity;
see the [follow-through evidence](research/2026-09-pragmatic-convergence.md).
The historical RSS comparison remains separate from the new convergence result.

Exact compiled-design checks and continuous-grid sensitivity are separate
evidence. Continuous Gaussian grids of 64, 256 and 1,024 bins reduce held-out
prediction differences in this fixture; they do not establish a universal
approximation rate. Whole-process RSS, actual dispatch, complete-fit work and
the timing audit are recorded in the completion report. This closes the chosen
C1 implementation foundation without claiming constant memory, universal
speedups or 10⁷–10⁸-row capability. The subsequently requested discrete
performance gate below remains open within C1.

The final-source [performance receipt](../benchmarks/c3_pragmatic_performance_receipt.json)
records three serial runs per arm on the replicated 449,000-row severity
workload: median complete-fit time 25.56 → 20.96 s (18.0% lower), and process
high-water RSS 1,490.48 → 1,007.39 MiB (32.4% lower). Headroom remained active;
the independently reviewed result is a qualified local observation, with
background CPU and screening changes recorded. The comparison changes both
source version and `discrete=False` to `True`; observed numerical agreement
does not replace separate discretization-error evidence. Final production
validation passed 11,546 tests with 174 skips and mandatory real-data availability.

## Next

**Immediate execution gate: discrete performance.** Before starting another
capability, investigate and improve discrete complete-fit execution. The
[implementation plan](research/2026-09-discrete-performance-plan.md) covers
cost-aware histogram dispatch and avoidable chunk preparation, with signed
stored-design equivalence, bounded memory and complete-fit evidence required.
Compact storage alone does not demonstrate faster fitting. The capabilities
below remain the subsequent priority order until this gate is resolved.

The fixed-layout size sweep at `ed84669a` through one million rows preserves a
memory/time tradeoff: chunked discrete execution uses substantially less memory,
while the same stored discrete basis runs faster through the existing dense
backend. Controlled BLAS threading improves that dense route but provides no
clear chunked speedup. Profiles identified repeated row preparation and rendering;
the resulting bounded range and renderer changes reduce complete-fit medians by
21% for ordinary chunks and 29% for explicit panels on the 262,144-row fixture.
The constructor-inclusive raw-basis tabmat pilot is slower than those panels,
so the existing signed matrix-product architecture is retained.

Automatic panels now admit a narrow mixed ordinary layout supported by those
measurements, with a separate 64 MiB workspace allowance and existing numerical
refusal. Automatic dense selection remains deferred. Full non-browser validation
passes 12,350 tests, including all 84 required real-data checks, with 109 other
skips. Final actual-default validation completes 28 timed public fits and three
separate dispatch witnesses. Discrete medians improve by 58.5% for fragmented
Gaussian, 49.4% for support-32, and 76.6% for Gamma severity against the frozen
post-C3 source; severity exact/discrete timing ranges overlap.

The earlier mixed-layout comparison found discrete fitting taking 44.5% longer
than exact while using 499.5 MiB less fit high-water RSS. The subsequent global
moment implementation addresses that gap, with further latency work still
required before another roadmap capability;
see the [performance report](research/2026-09-discrete-performance-report.md).
Scalar SuperGLM's cached-weight discrete REML optimizer remains a separate
approximation contract.

The implemented execution change computes on marginal supports at the chosen
resolution: aggregate changing row scores and signed curvature weights before
contracting the support bases. Earlier profiles confirmed repeated row
expansion in mixed panels. A three-condition geometry-batch ablation confirmed
lost support-contraction amortization, but using one whole-book batch still does
not beat the current panels in those diagnostics. Remaining mixed pair work
includes repeated weighted-column scans and spline-by-category expansion.
A bounded global moment prototype now processes the small ordinary block
together while preserving observation-level likelihood semantics and the
distributional optimizer. Independent signed/masked rectangular oracles pass.
Six controlled complete fits yield medians of 10.821 s for current discrete,
7.781 s for the prototype and 7.639 s for exact; prototype/exact timing ranges
overlap, while the prototype uses 496.79 MiB less median fit-end RSS. The stored
discrete basis is identical and holdout differences are below 9e-16. This
supports production integration with numerical guards, fallback and explicit
size-selection evidence. That implementation now passes independent review and
12,449 distinct latest tests across the broad run and documented followups,
including all 84 required real-data cases. Fifteen actual-default timed fits and
five separate witnesses now validate production `5c1ce17e`: the 262k discrete
median falls 12.036 to 7.884 s, and the million-row/P102 sample falls 39.439 to
29.321 s versus 31.956 s exact. The latter uses 1,869.37 MiB fit-end highwater
versus 3,893.99 MiB exact. Same-discrete outputs agree at roundoff scale;
discretization error remains separately measured. Global moments execute without
refusal on the two large mixed cases; smaller controls retain existing routes.

The user judged the 8% one-thread time advantage over exact insufficient and
explicitly selected further latency reduction on 2026-09-09. Single-fit wall
time is the objective; using more CPU is acceptable when it saves time. The C1
gate stays active. An eight-fit screen on 262k/P102 and 1m/P182 shows that four
BLAS threads substantially help dense execution but offer little discrete gain.
At the wider million-row shape, exact/discrete one-thread samples are
75.795/46.956 s, and four-thread samples 46.844/44.732 s. These are single
observations at two coupled N/P shapes, not a full crossover study.

The production profile and scalar policy control are complete. Scalar exact
forced1/forced4/auto samples take 5.118/5.395/5.120 s; discrete auto takes
0.648 s. A subsequent source audit corrects the optimizer interpretation:
the discrete arm enters the direct-REML wrapper and delegates to cached-W;
the witness missed the imported inner-function alias. Auto correctly selects
one thread for that 79-coefficient fixture, but its exact/discrete speed ratio
does not isolate representation cost or establish a universal scalar policy.
LSS geometry's diagnostic 21.485 s divides mainly into 12.693 s accumulating
global moments and 8.591 s producing chunks; only 1.619 s of chunk production
is family evaluation. These instrumented intervals are not fit-time estimates.

The four-condition full-pass comparison is complete with separate witnesses.
Current / 65,536-row geometry / full geometry / all full-row passes take
27.104 / 24.794 / 29.413 / 26.978 s, with fit highwaters
1,869.01 / 1,894.90 / 2,460.33 / 2,461.75 MiB. Stored designs match,
iterations remain unchanged and output differences are at rounding scale.
These are single samples; full-row execution still constructs copied state.
No default changed. The user now emphasizes reductions in row-dependent
complexity, so further batching/threading work is held while auditing scalar
cached-W reuse and exact/approximate joint-row aggregation. Distinguish marginal
binning from fewer likelihood records; the intended approximation tradeoff is
being clarified. The shared BLAS
controller currently sees only a 1,500-coefficient threshold, with no row-count,
backend or timing input; `-1` disables intervention rather than selecting an
optimal count. It does not parallelize native moment loops. Use live
`SuperLSS.diagnose()` alongside kernel profiles, captured after fit clocks/RSS.
Any new threading/accumulator decision must preserve coupled signed curvature,
numerical certification and bounded ownership. Initial admission remains a
scope limit, without universal speed parity or a novelty claim. Wood's 2020
review described the multiple-predictor large-data extension as not yet usable,
not mathematically infeasible; references and complexity are in the report.

After the active C1 performance gate, the candidate priority order is:

**1. Shape-constrained LSS (C5).** Close the explicit gap between scalar pricing
constraints and distributional fits, starting with demanded monotone effects.
Reuse scalar experience; settle the [SCOP acceptance/termination question]
before transplanting its rules. **Gate:** demonstrated modelling need, joint
likelihood/constraint correctness, and defensible inference at active boundaries.
Constrained predictor shape must be distinguished from shape of a derived risk
quantity. Use the relevant C3 convergence evidence above; constrained boundaries still
need their own acceptance and inference contracts.

**2. Extend functional inference for actuarial decisions (C12).** Build on
`posterior_bounds` and family-owned functionals for loss layers, risk contrasts,
and feature effects; add derivative-based uncertainty only where it is useful
and justified. **Gate:** a concrete quantity not adequately served today, checked
moment existence, atom/boundary behaviour, and uncertainty calibration. Existing
joint covariance is the dependency; neither C1 nor a general AD engine is needed.

## Later

| Capability | Rationale, dependency, and promotion gate |
| --- | --- |
| Shared smoothness or effects (C4) | Useful parameter pooling. Separate shared bases, shared lambda, and shared coefficients: equal lambda alone does not require off-diagonal penalties. Scope a real model and prove identifiability/joint penalty traces. Existing `select=True` shrinkage is not missing. Does not require C1. |
| Distributional structure discovery (C9) | Extend the existing PSST/refit workflow when held-out residual structure shows predictive value. Require affordable candidate fits and separate selection from confirmatory inference; benchmark against distributional boosting. The booster scripts are experiments, not a public LSS builder. |
| Conformal prediction wrapper (C10) | Complement effect inference when likelihood prediction coverage is inadequate. Require a credible calibration split/exchangeability assumption and segment evaluation; promise only the coverage the chosen method supports. Independent of solver scaling. |
| Deterministic portfolio aggregation (C11) | Complement existing simulation when tail precision or simulation cost blocks a use case. Specify conditional independence/dependence and parameter uncertainty first; validate discretization and tail error against reference laws. FFT, Panjer, and saddlepoint have different applicability; heavy tails may have no usable CGF. |
| Joint body–tail model (C13) | Go beyond the documented fixed-threshold splice recipe only when threshold uncertainty or tail fit matters in held-out decisions. Requires normalized body/tail likelihoods and identifiable threshold treatment; shape regularization is a modelling choice, not a cap repair. |

## Research

- **Sparse Laplace and small family-derivative tools (C2):** investigate separately.
  Structured scalar credibility is already available. Measure factor fill and
  family-development cost before choosing either sparse factorization or AD;
  neither guarantees the other pays. No core replacement without workload evidence.
- **Transformation models and non-crossing quantiles (C6/C7):** first demonstrate
  persistent family misspecification. Monotonicity and a response-dependent design
  need their own contracts; the current observation contract supports complete
  observations only. Censoring/truncation needs an extension. Inverting a fitted
  CDF does not reproduce calibrated quantile-loss regression automatically.
- **Dependence-aware validation/NCV:** assess when ordinary held-out validation
  misrepresents the intended book. The dossier calls NCV already road-mapped, but
  the repository has no public NCV implementation; it is not an assumed prerequisite.

## Deferred

- **Copula/multivariate LSS (C8):** require stable marginal fits and evidence that
  residual dependence changes decisions. Begin bivariately if promoted; flexible
  high-dimensional dependence remains research, not a consequence of adding AD.
- **Adaptive density GAMs (C14):** bounded/light-tailed feasibility studies only
  after named families fail. Per-row normalization and dense response-basis
  curvature need an affordable algorithm and an explicit tail model before a build.

## Killed

Standalone distributional PCA, Fisher–Rao manifold GAMs, and a full INLA/SPDE port:
no sufficiently scoped actuarial payoff with tractable, identifiable inference.
IDR as a core estimator: retain it as a possible comparator, not an effect model.
A GPU-first rewrite or dense high-rank adaptive density as the default scaling
strategy: no measured case over the CPU/structured approach. These are rejected
project directions, not claims of mathematical impossibility or bans on testing
another predictor count. Reopen only with a concrete use case and validating evidence.

## What changes the order

- New demonstrated wrong-answer or uncertainty failures take precedence over
  expansion. Reopen C3 when a scoped unsupported endpoint or reproduced
  convergence failure blocks an intended model. The tested NB2 stop prompted
  EFS recovery, pragmatic stopping and a bounded finite-NB2 range extension;
  it now reaches configured stationarity.
- C1 remains active through the discrete execution gate above. After that gate,
  reopen scaling work when an intended book exceeds its memory or latency
  budget. Profile derivative evaluation, accumulation, retained history and
  coefficient factors before selecting another execution change. Bin
  sensitivity or covariance disagreement blocks a representation change even
  if it improves speed.
- Promote C12/C11 for a blocked layer/capital decision, C10 for a demonstrated
  coverage gap, and C9 for repeatable held-out predictive gains. Compare
  log/tail scores, calibration and total cost on multiple books; novelty claims
  do not establish predictive value. Revalidate the dossier's dated competitor
  claims.
- Follow the [cost and timing policy](development/cost-and-timing.md): record
  exact revision/data/configuration, stopping status, work/allocation and
  dispatch. Use raw worker clocks and artifacts, with an activity audit during
  each timing run; tool completion clocks are not fit timings. Current work
  follows the user's correction that Headroom and Kompress are uninstalled.
  Historical scalar timings and one dense LSS receipt
  cannot justify 10⁷–10⁸-row or sub-second LSS promises.

## Dossier corrections to retain

Discretization reduces basis-product work but retains row likelihood and weight
aggregation passes; it does not remove all dependence on row count or guarantee
a sparse coefficient Hessian. C1 and C2 are not universal prerequisites.

The current GPD link keeps `0 < shape < 1`: expected shortfall exists throughout
that mathematical domain, while variance needs `shape < 0.5`. The dossier's
half-shape TVaR refusal and blanket second-moment requirement are incorrect.
A bounded loss layer can have finite expectation even when the response mean
diverges; finite CRPS can also coexist with infinite variance. Retain quantity-specific
existence and numerical-resolution checks rather than the dossier's generic rules.

[historical benchmark record]: https://github.com/StrudelDoodleS/superglm/blob/21007082/benchmarks/results/local_perf_baselines.json
[PR #376]: https://github.com/StrudelDoodleS/superglm/pull/376
[Grouped assembly]: https://github.com/StrudelDoodleS/superglm/blob/21007082/src/superglm/distributional/solver/assembly.py
[rectangular cross-products]: https://github.com/StrudelDoodleS/superglm/blob/21007082/src/superglm/_group_matrix/_cross_matrix_execution.py
[tests against materializing discrete slopes]: https://github.com/StrudelDoodleS/superglm/blob/21007082/tests/test_distributional_grouped_assembly.py
[SCOP acceptance/termination question]: https://github.com/StrudelDoodleS/superglm/issues/366
