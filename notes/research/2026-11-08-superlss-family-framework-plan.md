# SuperLSS Declarative Severity Families: First-stage Implementation Plan

> **Archived planning snapshot.** P1 has since been implemented and locally validated. See the [implementation validation](../../benchmarks/declared_scalar_families_validation.md) for delivered scope and remaining acceptance items. Original proposal language and unchecked tasks below retain their historical planning status.

> **For agentic workers:** after owner approval, use `superpowers:executing-plans` to implement this plan task by task. Checkboxes describe future work, not work performed in the research session. This document does not request delegation or authorize implementation, commits, publication or dependency changes.

**Goal:** make a new scalar-response SuperLSS severity family mostly a law declaration plus explicit modelling hooks, proven by useful Lomax and log-logistic families.

**Architecture:** extend the existing production symbolic generator and compiled kernels; add an internal immutable scalar-family specification and adapter using the existing `DistributionalFamily` contract. Keep named public classes, stable per-family numerical escape hatches and optional structural capabilities. Do not create another solver or wrap an arbitrary SciPy distribution and claim it is ready for regression.

**Tech stack:** Python 3.12+, NumPy, SciPy **>=1.15**, existing Numba runtime; existing SymPy/mpmath development dependencies for generation and independent oracles. Portable float64 production numerics.

**Spec and research rationale:** [SuperLSS family roadmap](2026-11-08-superlss-family-roadmap.md), particularly sections 3–5. It owns literature, mathematical identities, modelling alternatives and conditional later stages.

**Source baseline:** `origin/master` **`6643ed9c94497eba0efb8869b945a17c5d380caf`**, fetched and checked on 2026-11-08. The checkout used to write this plan is older; every existing-code assertion was checked at the remote pin. File names below refer to that revision or are explicitly marked new. Before implementing, compare the approved baseline with then-current master and reconcile relevant changes. Do not implement against the old research checkout by accident.

## Global constraints

- SuperLSS only. Scalar SuperGLM catalogue expansion, Dirichlet production fitting, a new smoothing algorithm, negative-xi GPD, parameter-dependent response support and a public arbitrary-law factory are outside P1.
- Retain `scipy>=1.15`. No runtime dependency on newer `make_distribution` custom-object support, SymPy, mpmath, JAX, PyTorch, `tools` or research modules.
- Generated derivatives and stable primitives already ship. Extend `tools/generate_severity_families.py`, `kernels/_declared/` and existing window composition; do not revive a competing prototype generator.
- Natural evaluation accepts exactly integer orders 0, 1, 2, excluding booleans; higher directions are separate protocols. Raw signed Hessians and canonical packed order are unchanged.
- Retain the current observation API: exact rows, left/right censoring and per-row truncation. General interval-censored observations with independent lower/upper observation endpoints are outside P1; finite interval-probability numerics do not imply that public capability.
- Unit prior weights and exact integer frequency replication are the first pilots' supported weight contracts. Non-unit prior weights are explicitly unsupported. Use the existing resolver and row provenance.
- Mathematical domain and numerical evaluability are distinct. Preserve all positive Lomax shape values; do not force finite means by clipping. Preserve legitimate infinite moments and explain finite-result overflow separately.
- Existing convergence and derivative-authority disclosure remains authoritative. Do not tighten every fit into mandatory certification, relax criteria, or mask failures to deliver a family.
- Public APIs are exported through `src/superglm/__init__.py` and established distributional namespaces. Preserve unrelated edits. Use `apply_patch` for repository edits.
- A future feature PR should declare `release:minor` with its family/API rationale; a docs-only PR would declare `release:none`. Neither changes version files. This research delivery creates no PR or commit.

## Review focus

These five risks have explicit tests in the tasks below:

1. **Coordinates and capabilities:** natural jets misread as predictor jets, or optional stub methods advertising unavailable Fisher/ES/higher derivatives (Tasks 1, 3, 5).
2. **Finite but wrong tail channels:** particularly the known extreme fourth log-CDF failure, narrow windows and the Lomax zero endpoint (Tasks 1, 3).
3. **Likelihood accounting:** frequency multiplier applied twice, dropped carrier, or omitted truncation denominator (Tasks 2, 3).
4. **Moment boundaries:** finite bounded layers at infinite mean, misleading posterior ES summaries and finite-variance requirements in numeric scoring (Tasks 4, 6).
5. **Identity and replay:** wrong response/order/observation/configuration reused after slicing or serialization (Tasks 2, 5, 6).

## 1. First-stage design decisions proposed for approval

P1 includes two named families, `LomaxLSS()` and `LogLogisticLSS()`. Both have ordered natural parameters `("scale", "shape")`, positive supports and log links. Roles describe a monetary scale and a power-tail exponent. Lomax accepts finite `y>=0`; log-logistic accepts exact `y>0`. Neither fits a location. Both accept windows with endpoint zero when the observation contract allows them.

Their mathematical shapes are unrestricted positive values. `predict()` is the conditional mean, including positive infinity when it does not exist, following current `model._readonly_default_prediction`. `predict_parameters`, CDF and interior quantiles remain useful without finite moments. Named posterior quantities already refuse non-finite summaries; retain that policy. Log-logistic gets no expected-shortfall or Fisher protocol until independently justified. Lomax has simple analytic finite-domain ES and complete-row Fisher information, specified in Task 4.

After their observation tests pass, both pilots declare core `max_derivative_order=2`, `cdf=True`, `quantile=True`, `random=False`, `response_mean=True`, `censored_response=True` and `continuous_curvature=True`. `expected_information` is true only for Lomax after Task 4. Higher directional methods are separate capabilities. `response_mean=True` advertises a defined evaluation policy including divergent-moment results; it is not a promise of a finite mean on every row. Until a capability's task passes, it is absent or declared false.

The internal framework is deliberately smaller than a public plugin system. Specs are immutable constants referenced by stable law keys; public objects are ordinary top-level dataclasses. No mutable process registry, dynamic class synthesis, pickled closures, runtime symbolic expressions or global user registration. Existing families are controls; they need not migrate in P1.

Illustrative **proposed** public use (these classes do not exist at the analysis pin):

```python
from superglm import LomaxLSS, SuperLSS, bind_predictor, s

family = LomaxLSS()
model = SuperLSS(
    family,
    bind_predictor(family, "scale", s("age", k=8)),
    bind_predictor(family, "shape"),  # constant tail first
)
```

Existing `ScalePredictor`/`ShapePredictor` helpers can supply `family.scale(...)` and `family.shape(...)` without inventing new predictor plumbing. A later mean parametrisation or moment-constrained variant needs a separate specification. The owner can change these choices at review; the tasks assume this design if approved.

## 2. File and responsibility map

| Files | Planned responsibility |
| --- | --- |
| Existing `tools/generate_severity_families.py`; new generated `src/superglm/distributional/kernels/_declared/lomax.py` and `log_logistic.py` | Add the two law declarations, preserve stable symbolic primitives, regenerate deterministic output and its existing provenance |
| New `src/superglm/distributional/kernels/_declared_adapter.py` | Small numerical result type and mechanical generated-density invocation; no distributional contracts |
| New `src/superglm/distributional/kernels/lomax.py`, `log_logistic.py` | Per-law domain handling, zero/limit paths, stable windows and prediction/moment numerics; delegate ordinary jets to generated modules |
| New `src/superglm/distributional/families/_declared.py` | Frozen runtime specification, immutable plan, generic core adapter and optional capability mixins |
| New `src/superglm/distributional/families/lomax.py`, `log_logistic.py` | Named public class, static specification, family-specific initialization, capability composition and audited reuse registration |
| Existing `families/_base.py`, `families/_response_bounds.py`, `kernels/_declared_bounds.py`, `kernels/_response_tails.py` | Reuse link-direction and observation/window helpers. Modify only if a demonstrated shared gap requires it; no replacement engine |
| Existing `src/superglm/__init__.py`, `distributional/__init__.py`, `distributional/families/__init__.py` | Export named public families using current lazy/import patterns |
| New `tests/test_declared_scalar_adapter.py`, `test_lomax_lss_family.py`, `test_lomax_lss_kernel.py`, `test_log_logistic_lss_family.py`, `test_log_logistic_lss_kernel.py`, `test_declared_scalar_vertical_slice.py`, `_declared_scalar_oracles.py` | Focused adapter, numerical, public-fit and independent-oracle coverage |
| Existing `tests/_severity_fixtures.py`, `test_declared_severity_families.py`, `test_distributional_family_kernel_architecture.py` | Reuse generator tests and retain adversarial cases; extend exact import-edge allowlists without weakening architecture rules |
| New `benchmarks/declared_scalar_families.py`; later source-bound receipt beside it | Complete-fit comparison, work/dispatch evidence and held-out synthetic tail/layer validation |
| Existing `docs/development/internals/distributional-family-development.md`, `docs/api/distributional/families.md`; new `docs/how-to/fit-a-lomax-distributional-model.md` | Developer extension recipe, capability/domain documentation and a modest applied example |

All non-document paths in this table are future work. The research session creates only this plan and the linked research note.

## 3. Proposed internal interfaces

These are design proposals, not verified existing import APIs. Existing types retain their definitions in `distributional/family.py`. `Array` below means a float64 NumPy array; `Mask` a boolean array; `Jet` is the existing packed tuple `(J0, ..., Jp)`, with widths `comb(K+d-1,d)` at order `d`.

### Numerical boundary

In new `kernels/_declared_adapter.py`, define a frozen `ScalarJetResult` with `jets: Jet`, `carrier: Array` and `valid: Mask`. Jets contain optimizing log-density derivatives; the carrier is parameter-independent, separately weighted, and included once in reported likelihood. A generated module's coordinate label must be checked rather than inferred from its name.

Each per-law numerical module exposes:

```python
density_jets(y: Array, theta: Array, multiplier: Array, *, order: int) -> ScalarJetResult
window_jets(lower: Array, upper: Array, theta: Array, *, order: int) -> tuple[Jet, Mask]
cdf(y: Array, theta: Array) -> Array
quantile(p: Array, theta: Array) -> Array
mean(theta: Array) -> Array
variance(theta: Array) -> Array
positive_moment_exists(theta: Array, r: float) -> Mask
```

`theta` has shape `(n,2)` in **natural** coordinates. `density_jets` supports orders zero through four; only the requested arrays are allocated/evaluated. `window_jets` is **unit mass**, has no density carrier, and also supports only numerically admitted orders through four. The family adapter supplies frequency multipliers once when composing windows. Ordinary generated calls use existing `law_jets`; use existing guarded dispatch or an equivalent documented numerical admission check when a fast path has a smaller safe envelope. Never claim an array fallback exists until one has been implemented and tested.

Both modules expose `positive_moment_exists` internally to avoid repeating domain conditions; it is not a new public engine protocol. The Lomax module additionally exposes `expected_information(theta: Array, multiplier: Array) -> Array` in packed natural information order and `expected_shortfall(p: Array, theta: Array) -> Array`. Log-logistic does not add those names in P1.

Unrepresentable ordinary rows use the established validity/error conventions. A known inaccurate higher channel needs an explicit numerical-domain signal, even when finite. Translate that signal at the family boundary into the existing unsupported-analytic-direction route; do not invalidate an otherwise sound order-two fit just because its analytic fourth derivative is unavailable.

### Family boundary

In new `families/_declared.py`, define frozen `DeclaredScalarSpec` with:

- `law_key: str`, `schema_version: int`, and a stable `kernel_module: str` naming the numerical wrapper, not a live module object;
- `parameters: tuple[ParameterSpec, ...]` and `response_lower: float`, `response_lower_inclusive: bool`;
- `capabilities: FamilyCapabilities` reflecting core order two, observation support and continuous curvature;
- `default_prediction_name: str | None` and explicit supported-operation metadata validated against the concrete class's protocols.

For P1 use law keys `"lomax/scale-tail/v1"` and `"log-logistic/scale-tail/v1"`, schema version `1`, natural parameter order `("scale", "shape")`, and default prediction name `"conditional_mean"`. Initialization remains a top-level family function with signature `initialize_parameters(y: Array, plan: DeclaredScalarPlan) -> InitialParameterState`; do not hide `scipy.stats.fit` in a generic iteration path.

`DeclaredScalarPlan` owns resolved weights, frozen exact responses and carriers, observation metadata, row-law semantics, law/configuration identity and their digests. Its `take(indices)` preserves the same positional/weight invariants as existing plans. Use `immutable`, `array_digest`, `typed_plan` and observation helpers rather than reproducing their validation rules. `plan_identifier` must distinguish changed data, weight semantics, row order, bounds, law version and family configuration. Generated artifact digests validate the build; they are not a substitute for semantic version/configuration identity.

`DeclaredScalarFamily` supplies `parameters`, `bind_likelihood`, `validate_likelihood_plan`, `evaluate_natural` and `to_config`. Named classes supply `initialize` by calling their initialization function. `to_config` identifies type, law key, schema and parametrisation with a nonempty immutable-in-effect snapshot, consistent with null fitting. Optional distribution-function, prediction, window and direction methods belong to explicit mixins used by the concrete classes. Omit unsupported mixins; returning `NotImplementedError` from every optional stub is not structural absence.

Add no family-name switch in solvers, inference or the facade. Reuse registration is per exact concrete family/plan type and requires audited prepared fields, link types and deterministic replay. Do not inherit eligibility from a prototype or claim arbitrary custom-link reuse.

## 4. Tasks and acceptance criteria

### Task 1: production Lomax declaration and numerical boundary

**Files:** generator; new generated Lomax module; new numerical bridge and `kernels/lomax.py`; new kernel/oracle tests; existing generator fixtures/tests and architecture test.

**Consumes:** existing `Declaration`, `law_jets`, packed-jet convention, stable primitives. **Produces:** `ScalarJetResult` and `lomax.density_jets`; frozen declaration metadata checks. Other numerical functions arrive in Tasks 3–4.

- [ ] Add `test_lomax_law_and_all_mixed_jets` against independently written high-precision `log(alpha)-log(scale)-(alpha+1)*log1p(y/scale)`, differentiated in the oracle, and SciPy body values with `loc=0`. Test all mixed entries through order four and both natural/predictor-coordinate transformations; an oracle derived only from the same generated expression is insufficient.
- [ ] Add `test_lomax_zero_limit` and `test_lomax_lazy_orders`. At zero, density is `alpha/scale` and natural log-density jets follow `log(alpha)-log(scale)`; every valid order-two point must have the same optimizing value/carrier at order zero. Instrument unused high-order arrays/calls so an eager implementation fails.
- [ ] Run these focused tests before implementation and record that the missing path fails. Add a mutation that swaps an off-diagonal packed entry or treats natural jets as log-coordinate jets; the corresponding oracle test must fail.
- [ ] Add a stable positive-response Lomax declaration to the existing generator, simplifying away the fixture's cancelling `-log(y)+log(y/scale)` terms. Supply the explicit zero-limit numerical path. Reuse stable ratio/log-tail primitives; if a new primitive is needed, derive and test its domain/error model before using it.
- [ ] Implement the numerical result/invocation boundary and regenerate checked-in output. `PARAMETERS`, `COORDINATES`, `LAWS`, `MAX_ORDER` and runtime spec order must agree. Include supported-domain validation and no symbolic imports at runtime.
- [ ] Run `uv run pytest tests/test_lomax_lss_kernel.py tests/test_declared_severity_families.py tests/test_distributional_family_kernel_architecture.py -q` and `uv run python tools/generate_severity_families.py --check`. Expected: new tests pass, regeneration is clean, and only explicitly retained pre-existing xfails remain.

**Numerical grid:** scales `1e-6, 1, 1e6`; alpha `0.25, 0.75, 1, 1.25, 2, 5, 25`; zero plus body probabilities `1e-3, 0.1, 0.5, 0.9, 0.999`; representable left/right tail points at `1e-12, 1e-30, 1e-100`. Add `nextafter` neighbours of moment boundaries and independent response/scale range-stress cells. Nonrepresentable points are labelled range cases, not silently deleted from a passing grid.

**Error gate:** retain the existing expression-composed oracle check `abs(error) <= 2*u*A + 1e-300` and its non-log-CDF quality check `A <= 1024*S` on their established grid (`u=2**-53`). These are existing tests, not a blanket certificate for new zero/limit branches. Derive absolute/scaled bounds for those branches, including cancellation and conditioning; record their derivation and oracle measurement with the test. Roots use absolute bounds, not relative error. The known fourth-log-CDF failure is handled in Task 3, not waived by this gate.

### Task 2: immutable adapter, weights and complete-row Lomax fits

**Files:** new `families/_declared.py`, `families/lomax.py`, adapter and family tests. Reuse `family.py`, `weights.py`, `families/_base.py`; change shared contracts only if a separately demonstrated gap requires it.

**Consumes:** Task 1 numerical jets and proposed spec. **Produces:** core `DeclaredScalarFamily`, `DeclaredScalarPlan`, and `LomaxLSS` with complete-fit configuration and initialization. Public exports may wait for Task 5.

- [ ] Add `test_declared_plan_is_bound_to_law_rows_weights_and_observation`: mutate original caller arrays after binding, swap responses, permute rows, vary weights or law key, and use a plan from another configuration. Mutating callers must not change accepted state; mismatched reuse must fail with a specific contract explanation. Valid `take`/chunk operations must agree with unsliced evaluation.
- [ ] Add `test_lomax_frequency_equals_literal_replication` for optimizing values, carrier, all packed derivative orders and an intercept fit. Include integer weights `1,2,5`, zero-weight dropping through the resolver, fractional-frequency refusal, unit-prior equality and non-unit-prior refusal. Mutating the implementation to multiply derivatives twice must fail.
- [ ] Add `test_declared_derivative_order_contract`, including `False`, `True`, negative values, floats and values above two; assert missing score/Hessian is `None`, not a zero array. Add a carrier-bearing test fixture so an all-zero Lomax carrier cannot conceal generic accounting errors.
- [ ] Run focused tests and capture failures. Implement the adapter via existing validation and plan helpers. Preserve full reported likelihood as optimizing value plus carrier exactly once. Reuse the current configuration snapshot and null-fit path.
- [ ] Implement deterministic frequency-aware initialization: start alpha at `2`; derive scale from a weighted positive response/finite endpoint median divided by `sqrt(2)-1`. If no positive information exists, use unit scale as a disclosed starting convention, not a fitted constraint. A fully uninformative/all-zero sample must not be presented as an identified finite optimum. Bounds are a source of starting values only, never imputed exact outcomes in the likelihood.
- [ ] Run `uv run pytest tests/test_declared_scalar_adapter.py tests/test_lomax_lss_family.py tests/test_distributional_family_contract.py tests/test_distributional_likelihood_weights.py tests/test_distributional_extensibility.py -q`. Expected: core fits and invariant tests pass; unsupported weights and unavailable optional methods fail as documented.

**Acceptance:** no solver/facade branch keyed on `LomaxLSS`; no mutable runtime registry; no SciPy fitting during iteration. Initialization is finite and inside declared support even for highly skewed or censored inputs. Statistical nonidentification must remain distinguishable from malformed data.

### Task 3: observation windows and honest higher derivatives

**Files:** Lomax numerical wrapper, shared adapter/mixins, new kernel/family tests, existing response-bound/declared-family tests. Reuse `_declared_bounds.py`, `_response_tails.py`, `families/_response_bounds.py` and `_base.py`; amend only the narrowly necessary helper if evidence requires it.

**Consumes:** core plan/jet boundary. **Produces:** `window_jets`, `ResponseWindowFamily`, censored/truncated binding, and first/second predictor-curvature directional protocols with correct fallback behaviour.

The existing `censored=` flags are `left`, `right` and `none` (or their supported boolean forms). Arbitrary finite intervals below are numerical probability queries, not a new interval-censored row kind.

- [ ] Add `test_lomax_mixed_observation_likelihood` for exact, left- and right-censored rows, left/right/two-sided truncation and combined censoring/truncation using the current observation contract. Compare each contribution and derivatives with an independent high-precision normalized likelihood. A mutation omitting the truncation term, retaining the exact density on a censored row, or duplicating its frequency multiplier must fail.
- [ ] Add `test_lomax_window_extremes`: whole support gives log probability zero and zero parameter derivatives; empty probability regions are handled explicitly; finite far-right/narrow windows remain accurate when both ordinary CDFs round to one; intervals adjacent to zero and tail probabilities down to `1e-100` get oracle checks. A naive `log(cdf(hi)-cdf(lo))` replacement must fail.
- [ ] Add `test_lomax_deep_left_fourth_is_not_false_analytic_evidence`, using the currently strict-xfailed case. Demonstrate failure against the pinned generated implementation or an explicit zero-channel mutation. The production path must either return an accurate derivative or signal that its analytic route is unavailable; it must never label the wrong finite zero as exact.
- [ ] Run the focused tests before implementing. Implement unit-mass windows through the existing tail-difference/narrow-window helpers. Supply a derived rate bound for log-response density variation; do not reuse Gamma's rate formula for Lomax. Compose observed/truncation adjustments with `bounded_channels`, `censored_carrier` and `bounded_higher_derivatives` as appropriate.
- [ ] Compose natural third/fourth jets with links using existing `natural_curvature_direction` and `natural_curvature_second_direction`. Check stacked directions and equality with independent directional differentiation of the full weighted, bounded likelihood. Unsupported links/regions take the established fallback route; supported closed forms and numerically bounded paths retain distinct authority.
- [ ] Run `uv run pytest tests/test_lomax_lss_kernel.py tests/test_lomax_lss_family.py tests/test_lss_response_bounds.py tests/test_distributional_bounded_derivatives.py tests/test_distributional_analytic_directions.py tests/test_distributional_fourth_derivatives.py -q` and the generator check. Expected: new tests pass, no false analytic claim, unchanged existing cases.

**Admission gate:** do not advertise unrestricted analytic fourth log-CDF evaluation while the known failure remains. Fixing or narrowly routing that case is sufficient; this task does not require every optional derivative to be certifiable everywhere. If a useful bound/fallback cannot be established, keep the affected capability out of P1 and explicitly review the reduced scope before public delivery. Censoring/truncation at orders zero through two must still have its own acceptance evidence.

### Task 4: moments, prediction and optional Lomax Fisher/ES

**Files:** Lomax numerical module/class and focused tests; reuse current prediction/posterior facade. Do not create a generic tail quadrature API.

**Consumes:** admitted family law. **Produces:** CDF, quantile, mean, variance, moment-domain checks, complete-row Fisher information and expected shortfall. The linked research note specifies the identities and domain rationale.

- [ ] Add tests for CDF support/monotonicity and interior quantile inversion; mean at alpha below/equal/above one; variance below/equal/above two; ES below/equal/above one. Verify `predict()` preserves legitimate `+inf`, while invalid probabilities, NaN and finite-result numerical overflow receive distinct explanations. Test predictions in changed monetary units.
- [ ] Add `test_lomax_fisher_matches_integrated_score_outer_product` using independent high-precision integration. For `(scale=s, shape=a)`, packed unit-row information is `(a/(s*s*(a+2)), -1/(s*(a+1)), 1/(a*a))`; frequency multiplies it once. Its determinant is `1/(s*s*a*(a+2)*(a+1)**2)>0`. Do not substitute this matrix for the raw observed Hessian.
- [ ] Test that current `expected_information_provider` does not offer complete-row Fisher information to a censored/truncated plan. Test that selecting Fisher where unavailable remains an explicit unsupported operation.
- [ ] Add `test_infinite_mean_does_not_preclude_bounded_layer`: a test-local stable survival-integral callback gives a value in `[0,l]`, is continuous through alpha one, and works with existing callable `posterior_bounds`. Compare to independent integration; replace its near-one calculation with direct power subtraction to demonstrate the adversarial test's sensitivity. No new public layer method is required in P1.
- [ ] Add `test_nonfinite_tail_posterior_is_disclosed`: named posterior ES must not average only finite draws when a draw has alpha at/below one. Also preserve the current numeric-CRPS refusal when its required finite-variance/tail-bound evidence is unavailable; mathematical existence alone does not establish the implementation's tail bound.
- [ ] Run the failing tests, implement stable numerical functions and explicit capability mixins, then run `uv run pytest tests/test_lomax_lss_family.py tests/test_lomax_lss_kernel.py tests/test_lss_expected_shortfall.py tests/test_lss_posterior_numerics.py tests/test_lss_scores.py tests/test_distributional_support_predictions.py -q`. Expected: all relevant tests pass without changing existing non-finite summary policy.

**Acceptance:** no shape clipping for convenience; no certified ES synthesized merely from a quantile function. Expected shortfall is finite only where its population moment exists. Bounded-layer posterior summaries remain conditional on the chosen family/threshold and coefficient-uncertainty approximation.

### Task 5: second-family proof and public extension recipe

**Files:** new log-logistic declaration/generated/numerical/family modules and tests; public export modules; architecture tests; developer and API documentation.

**Consumes:** shared adapter/spec and optional mixins. **Produces:** public `LomaxLSS`, `LogLogisticLSS`, and a reproducible extension recipe that does not require modifying generic solver code.

- [ ] Add independent Fisk/log-logistic law/derivative tests using `loc=0`; use the same scale grid, shape `0.6,1,1.5,2,4,12`, body/tail probabilities and observation cases as above. Test exact-zero refusal without silently moving the response; windows can still begin at zero.
- [ ] Add `test_log_logistic_reuses_adapter_without_false_optional_capabilities`. Assert its core/default/CDF/quantile/window/directional protocols are present only when implemented, while Fisher and ES are absent. Its median is scale; mean and variance have the correct moment boundaries, with independent-oracle checks near and far from them.
- [ ] Add a deliberately differently named three-parameter test specification, using a small scalar-response law, to prove the adapter does not assume two parameters or names `scale/shape`. Reuse the existing four-parameter extensibility coverage as an engine control. Do not turn this test into a third public family.
- [ ] Run tests to establish missing functionality, then add the second declaration, numerical hooks and class. Initialize using a weighted positive median for scale and shape `2`, with the same explicit endpoint-start policy as Lomax. Do not modify the solver, smoothing loop or predictor binding for this family.
- [ ] Add public imports and fresh-process pickle/joblib round trips, including complete and bounded plans, predictions, likelihood identity and protocol capabilities. Unsupported schema versions must fail clearly; no mutable registry setup may be needed after loading.
- [ ] Document the declaration-to-generated-kernel-to-adapter workflow, exact weights, optional capabilities, coordinate labels, generation check, numerical envelope and source attribution. Distinguish the new public families from the still-test-only Weibull/Burr declarations. Explain full-domain means and the overlap with GPD.
- [ ] Run `uv run pytest tests/test_log_logistic_lss_kernel.py tests/test_log_logistic_lss_family.py tests/test_declared_scalar_adapter.py tests/test_distributional_family_kernel_architecture.py tests/test_distributional_public_face.py -q`, the generator check, and a runtime import test with SymPy/mpmath imports blocked. Expected: named families work without generator/development dependencies.

**Acceptance:** adding the second law changes the declaration, its small numerical/modelling hooks, exports, tests and docs; it does not copy plan/weight/bounds/serialization mechanics. If major edits to the generic adapter are required, review the spec boundary before calling this a reusable framework.

### Task 6: complete-fit, tail evaluation and performance evidence

**Files:** new vertical-slice tests and benchmark script/receipt; family guide and example. Existing benchmark/diagnostic helpers may be reused. No performance number is promised by this plan.

**Consumes:** both public families. **Produces:** source-bound engineering evidence, a reproducible applied recipe and a clear list of remaining modelling gates.

- [ ] Add intercept-only comparisons with independent SciPy optimization (`loc=0`) and a direct high-precision likelihood, then a small fixed-penalty regression comparison with independently assembled objective/gradient. Compare both solutions using the **same** evaluation function and stable sum. Use well-conditioned cases for parameter comparisons; use likelihood, quantile and layer agreement for weak-tail/near-boundary cases. Derive tolerances from evaluation error, curvature/conditioning and solver score residual, not a blanket coefficient tolerance.
- [ ] Add GPD/Lomax overlap tests for alpha greater than one: pointwise density/tails/windows and correctly transformed derivatives. Compare intercept MLEs, with covariance transformed where estimable. Do not assert equivalence of rich penalized GAM fits with different links.
- [ ] Add complete `fit` and `fit_reml` tests with modest scale and shape predictors, dense and supported discrete/chunk paths, offsets, integer frequency weights, retained/compact rows and save/load. Exercise default outer behaviour, explicit EFS and `efs+newton` from `initial_lambda` values `0.1,1,10`. Compare objective, target predictions, EDF/rank and covariance; keep stopping reasons and derivative authority visible. A same-looking prediction does not excuse a false certification flag.
- [ ] Build deterministic synthetic validation scenarios: correctly specified Lomax/log-logistic; variable scale with constant versus modest variable tail; misspecified body-tail severity; and known censoring/truncation. Training and held-out draws use independent saved seeds. Fix thresholds/layers before evaluating held-out draws, include moment domains on both sides of alpha one/two, and record the true bounded-layer oracle. No automatic mean-based start should be required in infinite-mean cases.
- [ ] Report observed-law log score, censored tail log score, exceedance calibration/counts, quantile loss and bounded-layer mean calibration/error. Censored real observations cannot validate latent payments that were not observed. Declare any real dataset's observation semantics, licence and split before making an applied claim; no dataset or real-book superiority is assumed here.
- [ ] Run focused vertical slices with `uv run pytest tests/test_declared_scalar_vertical_slice.py -q`. For development use small cases; final performance runs use the script below after correctness passes. Record failures as failures, not omitted timing cells.
- [ ] Implement `benchmarks/declared_scalar_families.py` with `--output PATH` and case selection, emitting one compact receipt. Complete-fit cases should include intercept-only `n=2000`, mixed scale/shape smooths at `n=20000` and `n=100000`, and bounded-response cases. These are proposed benchmark sizes, not statements about available data or achieved runtime.
- [ ] Compare the shared adapter with a benchmark-only direct adapter over the **same kernels and same law/links/penalties** to measure abstraction cost. This control is not an independent numerical oracle. Also run current production GPD on the equivalent intercept-only finite-mean problem and existing Gamma/log-normal controls to detect shared regressions; richer differently parametrised GPD fits are modelling comparisons, not equal-work performance baselines.
- [ ] In fresh processes record cold start separately from warmed complete fits; hold threads, machine, environment, data and seeds fixed. Include wall-time distribution, process peak RSS and retained payload/cache memory, fit iterations/retries, objective/log likelihood, parameter/prediction/SE/EDF/rank outputs, requested versus actual assembly backend and generated-kernel dispatch evidence. Use `SuperLSS.diagnose()` and narrow instrumentation; profiling is separate from reported timings. Baseline/candidate timing is same-session or accompanied by a same-session drift control. Do not reuse the October prototype timings.
- [ ] Run future repository-required checks at the actual implementation head: the pinned master uses `uv run python scripts/run_test_suite.py`, Ruff checks/format, lock/environment checks and `run_test.py`. Run the focused numerical regressions on native Windows, macOS and Linux ARM64 where portability is claimed. Real-data suites require both freMTPL2 files and `SUPERGLM_REQUIRE_DATA=1`; do not label skips as parity evidence.

**Performance acceptance:** no claimed speedup without complete-fit evidence and unchanged target mathematics/dispatch accounting. Every material time/RSS regression needs attribution and an explicit review decision; the owner has not supplied a numeric performance budget. Correctness must pass before deciding whether an observed overhead is acceptable. A slower prototype cannot be described as production-ready simply because its kernel microbenchmark is fast.

**Completion receipt:** state baseline/candidate SHAs, command, environment, full case definitions, timing controls, numerical/dispatch comparisons, meaningful warnings and native coverage. Explain any lost capability or unresolved uncertainty. P1 establishes family/framework correctness and engineering cost; P2 establishes usefulness on the selected real severity problem.

## 5. Conditional later stages: boundaries, not implementation tickets

The following remain research or scoping decisions. They are not tasks silently added to P1:

| Candidate | First bounded deliverable | Gate before production |
| --- | --- | --- |
| Burr XII | Compare standard `(scale,c,k)` with proposed `(scale,body_shape,tail_exponent)` on tail/layer targets | Identifiable additional shape, stable windows/higher derivatives and useful held-out behaviour |
| Fixed-threshold splice (#570) | One law assembled from already fitted body/exceedance/tail pieces, with explicit normalization and threshold provenance | Known-component likelihood contract; cross-threshold censoring needs joint treatment rather than an assumed factorization |
| Threshold/layer workflow | Common validation targets, threshold-sensitivity plots and bounded-layer posterior recipes | Reliable target data and uncertainty accounting; not automatic optimal-threshold selection |
| Beta LSS | Complete interior unit-interval mean/precision family and `K=2` oracle | Explicit zero/one policy; incomplete-beta tail derivatives if bounds are added |
| Dirichlet D1 | Fixed `K=3` positive vector-response feasibility, independent log-alpha predictors, beta reduction and response-shape impact map | Existing `model.py`, `fit_state.py`, `null_model.py`, checks/residuals/CV/serialization all audited; no fake scalar PIT or quantile |
| Dirichlet D2 | Owner-selected continuous shares, category counts, or reserving specification | Zero/total/missingness policy and suitable data; mean/precision coupling/identifiability if chosen |
| Dagum/GB2 | Small modelling and derivative study against simpler controls | Marginal payoff; for GB2 stable incomplete-beta parameter derivatives and four-predictor identifiability |
| Public declarative factory | Review the two-family extension experience and propose a versioned user API | Capability validation, packaging, save/load and maintenance costs demonstrated |
| AD/numerical backend | Optional research comparison on a specific difficult law | Float64 tail accuracy, higher derivatives, complete-fit RSS/time, platform/dependency burden and honest evidence labels |

Read the [research note's Dirichlet section](2026-11-08-superlss-family-roadmap.md#6-dirichlet-an-explicit-feasibility-project) before scoping D1. Core parameter-count generality is already present; the response shape, coupled transformations and application law are the work to investigate.

## 6. Approval boundary and self-review

This is a concrete proposal to review, not a request for more design questions before delivering the plan. The owner has not selected an implementation or a dataset. A later implementation request should confirm or amend the named-family scope, full-domain scale/tail parametrisation and application target; then recheck the analysis pin against current master.

Self-review for this document covered the existing production generator, supported dependency floor, protocol shapes and optionality, both weighting meanings, observation normalization, known strict xfails, full versus finite-moment domains, current infinite-mean prediction behaviour, scalar/vector distinction, source-pin integrity and test ownership. The unresolved numerical and empirical gates remain explicit. No checkboxes above were executed during this documentation-only task.

**Artifact handling:** the prepared checkout's existing `.gitignore` ignores `docs/superpowers/`. This file is saved at the requested path but is not staged or committed; any later collection of changes must include it explicitly. The research task does not change ignore rules.
