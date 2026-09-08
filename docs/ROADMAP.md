# Roadmap

Last strategic review: **2026-09-08**. Starting baseline: `origin/master` at
`8962c452` (published v0.31.0); C3/C1 implementation through `9277baef`.

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
  smoothing fits, including the tested large real-book NB2 configuration, remain
  uncertified.

## Completed scoped work: C3 + C1

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

**C1 production route:** public observed chunking, bounded derivative replay and
cached categorical row extraction are implemented and regression tested.
Complete smoothing fits cover signed geometry, weight semantics, interactions,
prediction/covariance and serialization. The insurance evidence shows a memory
benefit on 449,000 synthetic rows formed by repeating 22,450 real severity
policies; the smaller real book does not save RSS. The large real-book NB2 comparison
reproduces an uncertified smoothing stop in both versions, so its lower RSS is
not a claim of reliable converged fitting.

Exact compiled-design checks and continuous-grid sensitivity are separate
evidence. Continuous Gaussian grids of 64, 256 and 1,024 bins reduce held-out
prediction differences in this fixture; they do not establish a universal
approximation rate. Whole-process RSS, actual dispatch, complete-fit work and
the timing audit are recorded in the completion report. This closes the chosen
C1 scope without claiming constant memory, universal speedups or 10⁷–10⁸-row
capability.

## Next

In current priority order:

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
  expansion. Reopen C3 for a scoped unsupported endpoint or a reproduced
  convergence failure that blocks an intended model; the observed NB2 stop is
  an explicit candidate, not a hidden passing result.
- Reopen C1 when an intended book exceeds its memory or latency budget. Profile
  family derivatives, retained history and coefficient factors before adding
  another assembler. Bin sensitivity or covariance disagreement blocks a
  representation change even if it improves speed.
- Promote C12/C11 for a blocked layer/capital decision, C10 for a demonstrated
  coverage gap, and C9 for repeatable held-out predictive gains. Compare
  log/tail scores, calibration and total cost on multiple books; novelty claims
  do not establish predictive value. Revalidate the dossier's dated competitor
  claims.
- Follow the [cost and timing policy](development/cost-and-timing.md): record
  exact revision/data/configuration, stopping status, work/allocation and
  dispatch. Headroom/Kompress passthrough makes transformed tool output and
  proxy clocks unsuitable evidence; use raw worker artifacts and audit activity
  during a timing run. Historical scalar timings and one dense LSS receipt
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
