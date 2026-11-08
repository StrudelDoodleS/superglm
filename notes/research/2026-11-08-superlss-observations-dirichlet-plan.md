# SuperLSS Observations and Dirichlet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The parent has selected fresh Astra max stages; each child executes its assigned stage itself and starts no subagents. This design-stage author starts no agents. The user has already authorized the scope; this handoff adds no repetitive implementation approval gate.

**Goal:** finish censoring/truncation across all eleven scalar SuperLSS families and deliver production `DirichletLSS` for complete strictly positive continuous compositions.

**Architecture:** reuse the family-bound likelihood/weight contracts and generic coupled solvers. A small shared foundation supplies atom-aware endpoints, positive-sum jets and explicit vector response metadata. Family stages implement numerical laws; parent integration owns cross-family coverage, public documentation, full validation and review.

**Tech Stack:** Python 3.12–3.14 (development 3.13), NumPy, SciPy >=1.15, Numba; portable IEEE float64. SymPy/mpmath only in existing development/generation/independent-oracle tooling.

**Spec:** [Implementation design](2026-11-08-superlss-observations-dirichlet-design.md). Read it together with this plan; formulas, endpoint conventions and numerical admission there are part of these tasks.

**Base:** `1b63f7c59e3f351fcb194f0eb25ac4e1106f2ca7`. Existing validation at `6e5639a115e0def2f8b1d213fed3b96eed8eb738` is historical baseline evidence, not acceptance of this expansion. The design checkout is `/home/max/.t3/worktrees/superglm/t3-2328ce62`, branch `t3/superlss-dirichlet-censoring-design`; implementation checkouts are prepared by the parent/harness.

## Global Constraints

- All four remaining scalar families and production Dirichlet are authorized deliverables. Optional bounded analytic third/fourth directions may be refused as designed; core order-two support may not be replaced by a CDF-only capability flag.
- Retain `truncation=(lower,upper)` and left/right censor flags; no general interval-censoring API. Frequency multiplies the complete normalized likelihood once. Prior weights retain each family's distinct row law.
- Portable float64; no production `longdouble`, mpmath, symbolic or arbitrary-precision layer. High precision belongs in independent test oracles. No new dependency, version pin, lock or release edits.
- Preserve scalar APIs, rank/inference policy, numerical restrictions and convergence/derivative-authority disclosures. No family-name switches in solvers, fake scalar Dirichlet distribution functions or silent zero repair.
- Each child is **not alone in the repository**: edit only assigned files, preserve others' changes and report interface needs to the parent. Use `apply_patch`. No checkout creation/switch/rebase inside a child's work.
- Fresh stage names carry model and effort: `observations_foundation_astra_max`, `two_piece_observations_astra_max`, `nb_tweedie_observations_astra_max`, `dirichlet_vector_astra_max`. Model `gpt-6-astra`, effort `max`, explicitly selected by the user. Parent consolidates. No child runs full suites or waits more than about four minutes; hand off exact remaining commands/results instead of timer polling.
- A future feature PR declares exactly `release:minor` with the new observation/Dirichlet rationale. This plan neither creates a PR nor authorizes publication.

## Review Focus

Each risk below has owning tests in Tasks 1–7:

1. A zero atom/count at an open lower endpoint is included by continuous support clipping, or a noninteger censor limit is rounded as an exact response (Tasks 1, 3, 4, 7).
2. A plausible finite conditional score/Hessian is wrong because an absolute tail cutoff or numerator/denominator cancellation is ignored; a fast complete-row kernel bypasses bounds (Tasks 2–4, 7).
3. A `(n,K)` response becomes `n*K` observations, or CV/serialization loses component labels, zero-weight provenance or bound identity (Tasks 1, 5–7).
4. Large-concentration Dirichlet cancellation creates false convergence, wrong rank/SE, or a fabricated finite optimum for constant shares (Tasks 5–7).
5. Two-piece moving joins/mean transforms or scalar-only diagnostics advertise unavailable derivatives/operations (Tasks 2, 6, 7).

## File ownership and dependency graph

All paths below are repository-relative. `D/` abbreviates `src/superglm/distributional/` **only in this ownership table**.

| Owner | Exclusive production responsibility | Tests/artifacts |
| --- | --- | --- |
| Foundation, Task 1 | `D/response.py` (new), `D/family.py`, `D/observation.py`, `D/families/_base.py`, `D/families/_response_bounds.py`, `D/kernels/_response_tails.py` | New `tests/test_lss_observation_foundation.py`, `tests/test_distributional_response_spec.py`; existing focused shared tests |
| Two-piece, Task 2 | `D/families/two_piece.py`, `D/kernels/two_piece.py`, new `D/kernels/_two_piece_windows.py`; existing generated kernels are read-only dependencies | New `tests/_two_piece_observation_oracle.py`, `tests/test_two_piece_response_bounds.py` |
| NB/Tweedie, Tasks 3–4 | `D/families/negative_binomial.py`, `D/families/tweedie.py`, existing matching kernel modules; new `D/kernels/_incomplete_beta.py`, `_negative_binomial_windows.py`, `_gamma_windows.py`, `_tweedie_windows.py` | New `tests/_count_mixture_window_oracle.py`, `tests/test_negative_binomial_response_bounds.py`, `tests/test_tweedie_response_bounds.py`, `tests/test_incomplete_beta_engine.py` |
| Vector/Dirichlet, Tasks 5–6 | New `D/families/dirichlet.py`, `D/kernels/dirichlet.py`; `D/model.py`, `D/api.py`, `D/fit_state.py`, `D/null_model.py`, `D/solver/solver.py`, `D/solver/chunks.py`, `D/smoothing/endpoint_direction.py`, `D/checks/scores.py`, `D/checks/compare.py`, `D/cross_validation.py`, `D/posterior.py`, `D/separation.py`, `src/superglm/model_selection.py`; guarded changes in `D/residuals.py`, `D/surfaces.py`, `D/checks/calibration.py`, `src/superglm/editor/lss/view.py`, `src/superglm/editor/session.py` | New `tests/_dirichlet_oracle.py`, `tests/test_dirichlet_lss_kernel.py`, `test_dirichlet_lss_family.py`, `test_dirichlet_lss_vertical_slice.py`, `test_distributional_vector_response.py` |
| Parent, Tasks 7–8, after all children | Public exports; seven existing bounded adapters/declared adapter for audited endpoint-validation repairs; shared interfaces only after children stop; docs, `src/superglm/__init__.py` warmup/architecture allowlists, `tools/generate_severity_families.py` and any regenerated files, CI and benchmark harness | New all-eleven audit, lifecycle/operation tests and complete-fit harness/receipt; final full checks and independent review |

`Task 1 -> {Task 2, Tasks 3–4, Task 5 -> Task 9 -> Task 6} -> Task 7 -> Task 8`.

Task9 was added by the user's subsequent request: declare LSS offsets as a
predictor keyword, including a list of multiple offsets. It follows the
Dirichlet core stage so it can update every family helper without competing
API/model writers, and precedes the lifecycle stage that consumes it.

After Task 1 is integrated, the three family stages have disjoint ownership and may run in parallel. Tasks 3 and 4 share one owner and execute sequentially; Tasks 5 and 6 likewise. Parents supply Task 1's commit and Interfaces blocks to fresh stages. Do not concurrently edit a shared helper because an algorithm needs a change: agree the interface and have the owner/parent make it at a synchronization boundary. The parent keeps working on independent review/benchmark preparation while family stages run.

## Task 1: shared observation and response contracts

**Files:** ownership table's foundation files; read existing `tests/test_lss_response_bounds.py`, `tests/test_response_window_jets.py`, `tests/test_distributional_family_contract.py`, `tests/test_distributional_take_map.py`.

**Interfaces:**

- In new `response.py`, frozen `ResponseSpec(shape: tuple[int,...]=(), component_names: tuple[str,...]=())`; `response_spec(family)->ResponseSpec` defaults to scalar; `unvalidated_response(y, *, family, n_rows, input_index=None)->ndarray` validates original shape/labels without value coercion; `validated_response_array(y, *, family, n_rows)->ndarray[float64]` validates retained row-first shape/finiteness. Scalar broadcasting for score APIs remains an explicit caller option, not a solver default.
- In `family.py`, structural `ResponseSpecFamily.response_spec` and `VectorDefaultPredictionFamily.default_prediction_columns` protocols. Dirichlet implements these; scalar adapters need no new mandatory methods.
- `ResponseBounds.on_support(support_lower, *, atom_at_lower=False)`; same keyword in `bind_observation` and atomic-aware probability-value dispatch. `response_row_count` becomes row-first; `validated_float_response` remains scalar-only.
- In `_response_tails.py`, `log_mixture_jets(log_terms, term_score=None, term_hessian=None, *, order:int)->Jet` for small batches and a streaming equivalent `CoreLogMixtureAccumulator(k_parameters, order)` with `add(jet:Jet)` / `finish()->Jet`. Both use the same centered-moment mathematics; do not build a second derivative-packing convention.
- Extend `bounded_channels` with optional `normalized_rows`, `normalized` and `carrier` keywords. `normalized(rows,order)->TailChannels` returns **unit-mass complete normalized log contributions**. Exclude override rows from ordinary window subtraction, set their optimizing values to `multiplier*value`, multiply derivative channels once, and set validity from the requested channels. Require supplied `carrier` to be zero on override rows; do not subtract/re-add a large carrier. Default calls remain unchanged. The four new bounded adapters use zero carriers throughout the bounded plan, with stable full-density exact channels; complete plans keep their old representation.

- [ ] Write foundation regressions: continuous versus atomic `on_support(0)`; exact `y==lo` refused; left-at-zero atom valid; positive-only zero window refused; original-position errors after dropping invalid zero rows; bounds digest/take invariants; constant and two-component mixtures with analytic covariance term; normalized override with frequency `3`, and refusal of a nonzero override carrier. Prove an override can succeed when separate large numerator/denominator channels cannot be resolved.
- [ ] Write response-spec regressions: default scalar refuses `(n,2)`; opted-in toy vector accepts `(n,2)`; labels reorder by declared components; missing/duplicate columns and mismatched pandas row indices fail; zero-weight value coercion is deferred; no `n*K` counting. This task need not fit a real vector family yet.
- [ ] Run the new tests on the unchanged implementation and record the expected failures. Implement the interfaces above with backward-compatible defaults. Reject impossible bounds before continuous support clipping can erase their provenance. Supply scalar exact/threshold validation utilities for parent adoption across the seven existing families.
- [ ] Run `uv run pytest tests/test_lss_observation_foundation.py tests/test_distributional_response_spec.py tests/test_response_window_jets.py tests/test_distributional_family_contract.py tests/test_distributional_take_map.py -q`. Expected: all pass; existing scalar contracts unchanged except the explicitly tested open-endpoint correction. If the command exceeds the short-stage budget, hand its remainder to the parent.
- [ ] Commit only owned files with `refactor: define shared observation and vector response contracts`; provide signatures and commit to all consuming stages. No public family capability is flipped here.

## Task 2: two-piece bounded likelihood and complete fits

**Files:** two-piece ownership in table. Existing tests to retain: `tests/test_two_piece_lss_family.py`, `tests/test_two_piece_lss_kernel.py`, `tests/test_two_piece_lss_vertical_slice.py`.

**Interfaces:** consume Task 1 and existing normal primitives. New `_two_piece_windows.py` provides `two_piece_window_jets(lower, upper, theta, *, order, response_transform, parametrisation)->Jet`, where `response_transform` is `identity` or `log`, and `parametrisation` is the existing `location`/`mean` convention. Public families implement the existing `log_window_probability(lower,upper,theta,weights=None)` signature. Keep complete-plan identity compatible; bounded identities include bound digest and version, and `take()` takes observations too.

- [ ] Add independent normalized likelihood fixtures for real-line and log response, log-response location and mean parametrisations, both skew signs, same-side/crossing/join-endpoint windows, exact/left/right contributions and lower/upper/two-sided combinations. Check value, natural score/Hessian and linked predictor derivatives; the independent oracle uses piecewise normal CDF/integration with high precision, not production jets.
- [ ] Demonstrate failures before implementation: binder refuses bounds; a frozen clipped split-point mutation yields wrong join/cross derivatives; a missing mean-loading cross derivative changes the Hessian; density carrier survives censoring. Add one-ULP log windows and remote same-side tails; reference tolerances follow standardized width and scale.
- [ ] Implement explicit half-tail formulas and analytic transform derivatives from design section 4. Use normal window machinery only where its fixed-endpoint semantics apply. Skip density evaluation on censored rows with out-of-density-support thresholds. Add bounded plan identity, slice, initializer and zero-carrier/full-value accounting, together with working evaluation and `censored_response=True`. Keep complete carriers unchanged.
- [ ] Gate every optional bounded direction/certified direction and Fisher method with `refuse_bounded`; preserve complete-row join certificates and `continuous_curvature=False`. Test a forced optional refusal reaches existing evidenced fallback without silently returning complete-density derivatives. Verify censored all-one-side data discloses a nonfinite optimum/failed convergence rather than successful precise estimates.
- [ ] Add small intercept fits against an independent normalized optimizer; fixed smooth fits with a common extracted design and penalty; one smoothing fit for each public family and both log-normal parametrisations. Compare likelihood, score, predictions and covariance projections at stationary endpoints. Test integer-frequency replication and explicit nonunit-prior refusal.
- [ ] Run `uv run pytest tests/test_two_piece_response_bounds.py tests/test_two_piece_lss_family.py tests/test_two_piece_lss_kernel.py -q`. Run the small new vertical fixtures separately if needed. Expected: no mutation survives; complete-row outputs/optional policies remain compatible.
- [ ] Commit owned files with `feat: fit censored and truncated two-piece responses`; hand off exact short checks, numerical refusals and fit/backend witnesses. No shared helper/docs/export edits by this stage.

## Task 3: negative-binomial window jets and observation binding

**Files:** NB subset of the NB/Tweedie owner; new beta engine and count-window oracle/tests. Preserve `_count_lattice.py` and current complete-count policy unless a demonstrated endpoint helper needs parent-owned shared work.

**Interfaces:** `_incomplete_beta.py: log_beta_tail_logit_jets(a,b,logit_x,*,upper:bool,order:int)->Jet`, partials in `(a,b,logit_x)`, orders 0–2 only; `_negative_binomial_windows.py: count_window_endpoints(lower,upper,prior_weights)->(L,U,empty)`, and `negative_binomial_window_jets(lower,upper,theta,prior_weights,*,order)->Jet` in natural `(mean,theta)`. `prior_weights` is ones for frequency. Infinite bounds remain sentinels, never cast to integer. Keep original thresholds in identity even if two thresholds map to the same count set.

- [ ] Add exact count-window tests for `(-inf,0]`, `(0,1]`, `(0.1,0.9]`, `(1.2,4.8]`, both integer and noninteger censor limits, threshold neighbors, one-point windows, negative support limits and huge finite thresholds. For prior weights use `w=3`, `c=float(1/3)` and `nextafter` on both sides, plus nonuniform weights. Test finite-boundary range refusal near `2**53` and indistinguishable prior-rate neighbors; an infinite upper tail must not be truncated to that representation limit. Assert exact-row mapping is never applied to censored thresholds.
- [ ] Add independent mpmath finite sums/beta tails and numerical derivative oracles for the **whole normalized** likelihood. Include small/large mean-to-size ratios, deep upper/lower tails, very narrow count windows, the existing low-mean range, and near-Poisson cases. Pin a conditional case that fails if density-only Poisson classification drops nonnegligible window size information.
- [ ] Run failing tests against the base and keep the compact failure list. Implement the beta fraction with derivative-aware convergence and stable prefactor/full PMF seed, count recurrence, geometric derivative-tail bounds and range/work refusal described in design section 5. Test integer beta shapes where the value fraction terminates but a shape derivative does not. Record the actual overlap envelope; no all-domain certificate claim from successive-iterate agreement.
- [ ] Extend NB plan with observation, bound digest, exact-row mask/counts and zero carriers for the full bounded plan; preserve old complete identifiers/carriers and legacy loading. Use the stable full PMF in bounded exact channels and a finite-law entry point which cannot independently select a density-only Poisson surrogate. Bounded `take` rebuilds consistent child identity. Exact conditional singleton and observed-equals-truncation channels must be algebraically zero. A huge-count two/three-point conditional fixture must retain its small log likelihood instead of losing it to factorial-carrier cancellation. Preserve frequency/prior effective `M,R` mapping and initialize from exact rows or admissible representatives only for starts.
- [ ] Add `log_window_probability`, `censored_response=True`, bounded optional-direction/Fisher refusals. Keep CDF/quantile/random flags false. Distinguish data-impossible empty sets from a trial whose beta/recurrence evaluation fails. A forced work-cap or fraction failure must backtrack/disclose, never yield a clipped probability or a successful fake fit.
- [ ] Add intercept and small fixed/smoothing fits under both weighting meanings with independent reference objective and stationarity checks. Verify frequency replication, aliases/rank, exact/scaled-count complete-row parity, chunk slicing and null fit.
- [ ] Run `uv run pytest tests/test_incomplete_beta_engine.py tests/test_negative_binomial_response_bounds.py tests/test_negative_binomial_lss_family.py tests/test_negative_binomial_low_mean_ratio.py -q`. Expected: all core/reference cases pass and forced refusals are visible. Hand any long derivative-oracle grid to the parent.
- [ ] Commit `feat: fit censored and truncated negative-binomial responses`, then proceed to Task 4 in this stage while context is fresh.

## Task 4: Tweedie compound-mixture windows and bounded fitting

**Files:** Tweedie subset of same owner; new `_gamma_windows.py`, `_tweedie_windows.py`; existing `_incomplete_gamma.py` is a dependency, not a replacement target. If its tested envelope blocks an ordinary required case, isolate and report the minimal primitive change to the parent with oracle evidence.

**Interfaces:** `_gamma_windows.py: gamma_shape_scale_window_jets(lower,upper,shape,log_scale,*,order)->Jet` in `(shape,log_scale)`; `_tweedie_windows.py: tweedie_window_jets(lower,upper,theta,prior_weights,*,order)->Jet` in natural `(mean,dispersion,power)`. Internal result diagnostics include evaluated terms, gamma iterations, route and refusal reason for untimed instrumentation; do not retain term tables on the fitted model.

- [ ] Test gamma-engine call semantics independently: scaled plain partials versus factorial-divided jets; fixed-argument versus fixed-ratio composition; narrow positive and zero-touching windows. Compare to independent high precision, including the existing switch line and small-shape route. This guards wrong argument scaling masquerading as a valid Tweedie Hessian.
- [ ] Add mixtures with zero rows and positive rows: left censor at zero, right censor at zero, `(-inf,0]`, `(0,hi]`, `(negative,0]`, `(0,0]` public probability query, upper/lower/two-sided truncation, and combinations under frequency/prior semantics. Include mixed finite per-row limits and zero-weight malformed bounds. A mutation using `.on_support(0)` or adding frequency twice must fail.
- [ ] Add a rare event with probability well below `1e-12` for which the old CDF truncation is inadequate; derivative-tail remainder checks must force more terms. Test power near configured walls, small/large rates, narrow windows and forced max-work/nonconvergence. The oracle sums a longer high-precision mixture or integrates an independently defined density for tiny fixtures only.
- [ ] Implement design section 6 using the existing compound parameters and gamma engine. Accumulate unit-mass window jets and the Poisson-tail moment bounds; scale bounds outward in actual float64 operations. Agree on an explicit aggregate work budget after short work-count measurements, retaining the existing 20,000 upper-index ceiling initially. Publish the numerical support envelope and refusal conditions; no arbitrary additional power/shape wall.
- [ ] Extend Tweedie plan/identity/take and add public window evaluation. Preserve complete identifiers where needed for older serialized fits. Build bounded starts from exact data or finite admissible representatives, never feed impossible censor placeholders into the density initializer. Route bounded `predictor_rows` to `None`; guard every complete-row curvature hook. Test both forced fast dispatch and natural fallback to prove the fitted objective actually includes bounds. Keep bounded-atom residual refusal explicit and joint log scoring functional.
- [ ] Add small independent intercept fits and fixed/smoothing fits with prior weights and frequency replication. Include exact zeros, both censor sides and combined truncation in each benchmark family. Check complete-row CDF/quantile/atom and variance helpers remain unchanged.
- [ ] Run `uv run pytest tests/test_tweedie_response_bounds.py tests/test_tweedie_lss_distribution_functions.py tests/test_tweedie_lss_family.py tests/test_tweedie_lss_kernel.py -q`. Expected: references and failure-policy tests pass; the fixture requiring extra mixture terms cannot pass a mutation of the stopping rule back to absolute CDF mass.
- [ ] Commit `feat: fit censored and truncated Tweedie responses`; provide Tasks 3–4 receipts, exact work/dispatch counts, unresolved limits and parent-owned long commands, then end.

## Task 5: vector response plumbing and analytic Dirichlet core

**Files:** new `src/superglm/distributional/families/dirichlet.py`, `kernels/dirichlet.py`; vector owner's model/row-state/null/solver/endpoint and separation files; new kernel/family/vector tests. Public exports wait for the parent; import the concrete module in stage tests.

**Interfaces:**

- `DirichletLSS(components: tuple[str,...])`, immutable config; `.alpha(component,*terms,intercept=True,link=None)->BoundPredictor`; `.components`, `.component_parameters`, `.response_spec`, `.default_prediction_columns`.
- Parameter names `alpha_0...alpha_(K-1)`; generic `bind_predictor` and offset mapping unchanged. `default_prediction(theta)->(n,K)` and `response_covariance(theta)->(n,K,K)`; no `VarianceFamily` method.
- Frozen `DirichletLikelihoodPlan` contains `weights`, ordered component/config/policy identity, supplied canonical-order response bytes on retained rows, prepared normalized log shares, `observation=COMPLETE_OBSERVATION`, and `take(indices)`. `plan_identifier` hashes raw retained input/configuration/shape/weight/observation identity, not logs; dropped-row contents remain irrelevant while positions/provenance are retained. `validate_likelihood_plan` checks exact shape/order/response identity and returns an immutable canonical response.
- `evaluate_dirichlet_rows(response,theta,multiplier,*,derivative_order)->NaturalLikelihoodEvaluation` or the repository's equivalent primitive result plus adapter. Values `(n,)`, scores `(n,K)`, Hessian `(n,K*(K+1)//2)`, finite placeholders/validity policy as existing. Orders 0/1/2 share one target. Carrier zero.
- Family-local first/second predictor-curvature methods use the exact existing protocol signatures and link derivatives through order four, with `O(n*K**2)` directional storage. No dense fourth tensor or contracted/full tensor confusion.

- [ ] Write K=2 tests against `scipy.stats.beta.logpdf(y[:,0],alpha[:,0],alpha[:,1])` and an independent high-precision oracle. For K=3 test exact score/Hessian/polygamma orders, all mixed Hessian channels, first/second curvature directions, linked scores, Fisher information and literal integer replication. Use `stats.dirichlet.logpdf(y.T,alpha)` only for constant alpha; loop explicitly for row-varying alpha.
- [ ] Write response/identity tests for labeled/array inputs, permitted column permutation, duplicate/missing labels, pandas row-index mismatch, `K=2,3,5`, materially invalid simplex sums, roundoff-only normalization, nonunit-prior refusal, zero-weight invalid values and original input positions. Changing a component label/order or one retained response entry must invalidate the plan. A subset/permuted `take` preserves row provenance; duplicate row selections retain the shared weight-plan refusal.
- [ ] Demonstrate preimplementation failures. Implement only structural response-aware guards in solver/null/row-state paths; scalar adapters remain strict. Audit `endpoint_direction`, factor-smooth empty-boundary paths and `_curvature_disclosures`' original-row reconstruction shape. Do not register Dirichlet in `_likelihood_cache` or modify its certified 1-D authority to get a speedup. Prepared log-share storage follows the response-only owner/lifetime/invalidation policy in design section 7.
- [ ] Implement stable core, Fisher and directional formulas from design section 8, including compensated normalized ratios, gnorm/psires routes and scaled linked-channel representation checks. Test small alphas, mixtures of tiny/large alphas, concentration `1e2,1e6,1e10,1e12` where representable, responses within a few ulps of fitted mean, and positive shares down to `1e-250`. Test extreme cases beyond admission for clear core/optional refusal, including finite natural channels whose underflow corrupts a representable linked result. These grid values are test points, not shape walls or blanket domain promises.
- [ ] Mutate away the cross Hessian, use raw gammaln/digamma subtraction in the high-concentration fixture, flatten the response, count weights per component, and drop normalized-row compensation; each has a failing test. Compare stable mean/covariance projections and normalized likelihood, not alpha recovery in ill-conditioned cases.
- [ ] Implement finite starts, complete null fit and retained/compact row state. Constant-share data must not be reported as a certified finite-concentration MLE; preserve accepted fit and diagnostics. Test explicit valid initial state and invalid initial range.
- [ ] Run `uv run pytest tests/test_dirichlet_lss_kernel.py tests/test_dirichlet_lss_family.py tests/test_distributional_vector_response.py tests/test_distributional_family_contract.py tests/test_distributional_null_model.py -q`. Expected: vector fixed fits and null/shape contracts pass, no scalar validation regression. Commit `feat: add analytic Dirichlet likelihood and vector fitting`.

## Task 9: predictor-declared offsets (additional authorized scope)

**Files:** `distributional/binding.py`, `predictor.py`, family predictor helpers
including Dirichlet, a focused offset-resolution module if justified, and
minimal model/API/null/rebuild/scoring/CV/inference integration; new
`tests/test_distributional_predictor_offsets.py` and relevant existing tests.
Run after Task5, before Task6. Implementer remains Astra max; parent consolidates
and reviews. No parallel writers own these files during this stage.

- [ ] Add `family.mu(..., offset=...)` and the equivalent keyword on every
  family helper and generic binder. Support a link-scale numeric vector or
  column source, and a list of vectors/columns summed with coefficient one.
  Specify unambiguous arraylike/list parsing. Preserve immutable declaration
  ownership and template-copy/clone/serialization behavior. No implicit logs,
  fitted offset coefficients, expression language or exposure-law changes.
- [ ] Resolve source shapes on original input rows and values after zero-weight
  row removal; preserve input positions. Resolve column sources on the current
  prediction frame. Numeric training vectors need new offset data at prediction;
  never silently reuse them. Keep legacy fit/predict mappings compatible.
  Reject duplicate fit specifications and define new-data replacement clearly,
  including mixed sources. Internal resolved offsets must not apply twice.
- [ ] Test fixed/REML fits, null fits, retained/compact state, dense/chunked
  prediction, scalar and Dirichlet component ordering, scoring/CV, posterior
  inference, replay/refit and pickle/joblib. Compare with equivalent legacy
  arrays on synthetic data for likelihood, predictions, covariance/EDF, work
  and dispatched backend. Test multiple offsets, new-row lengths, invalid and
  missing sources, source ownership, zero-weight rows and template-copy mutation.
- [ ] Run focused regressions and short complete-fit controls, record RED/GREEN
  evidence and ownership/lifetime of any stored arrays, then hand broad timing
  and compatibility checks to Task8. No version/release/dependency edits.

## Task 6: Dirichlet prediction, inference, scoring, CV and lifecycle

**Files:** remaining vector-owner API/scoring/posterior/CV files, `src/superglm/model_selection.py`, explicit scalar-operation guards; tests `test_dirichlet_lss_vertical_slice.py`, `test_distributional_vector_response.py`. Inspect `inference.py`, `prediction_design.py`, `serialization.py`, `likelihood_identity.py`, `rebuild.py` before editing; if already shape-neutral, prove reuse by tests and leave them unchanged.

**Interfaces:** public vector `predict` DataFrame with component columns/X index; existing `predict_parameters` DataFrame with alpha parameter names. NLL remains a scalar per row. CV adapters expose `prediction_shape` (scalar default `()`, Dirichlet `(K,)`) and response canonicalization before `model_selection` discards labels. `_held_out_predictions` and OOF allocation use `(n,*prediction_shape)` and count unpriced **rows**, not NaN cells. Named scalar posterior quantities remain scalar; vector `mean` directs the caller to a scalar share callable.

- [ ] Add small deterministic K=3 intercept and linear/spline fits against an independent SciPy optimizer using an independently coded normalized objective, score and dense reference design/penalty. At least one reference uses a non-SuperLSS optimizer and natural-alpha constant fitting. Require stationary endpoints; label nonidentified solutions rather than comparing meaningless coefficients.
- [ ] Fit ordinary/fixed-penalty and `fit_reml` models, including different per-component smooth specifications, permuted components **with permuted predictors/penalties**, frequency replication, one constant alpha, offsets, compact/retained state, dense and chunked/discrete routes. Inspect actual backend; requested `discrete=True` alone is not evidence. Check null fit, EDF/rank, SE and joint covariance against independent Hessian projections.
- [ ] Implement DataFrame predictions and label preservation. Test alpha parameter order, component mean sums/positivity, underflow-to-zero range refusal and response covariance PSD/all-ones null direction. Posterior parameter draws preserve cross-component dependence; component/share-sum/log-ratio/concentration callables work. Compare mean-share uncertainty with a delta-method projection at a well-conditioned endpoint. Test corrected covariance succeeds with analytic authority or refuses explicitly when a forced optional direction fails.
- [ ] Enable joint log scoring on vector rows and default CV NLL only. Preserve input positions/weight mass after zero-row dropping and defer outer-CV value coercion accordingly. Test CV labeled inputs, 2-D OOF, zero-weight unpriced-row fallback and custom scalar scores. Users call `scores(...,which=("log",))`; preserve the scalar default and refuse requested CRPS rather than silently removing it. Enable paired log comparison for the same component set, canonicalizing labeled responses for each fit; positional comparisons require matching component order. Explicit CRPS/Gini/actual-expected/calibration/Murphy requests fail early with a named unsupported operation.
- [ ] Add table-driven refusals for scalar CDF/quantile/PIT/residuals/plots, threshold/tail scores, distribution surfaces, scalar default posterior mean, quantile-based posterior prediction/portfolio simulation and vector editor entry. They must fail before any flattening or silent component selection; family has no fake structural scalar protocols.
- [ ] Round-trip pickle and joblib, clone/refit, hand edits, retained/compact and rebound inference; alter input labels/response/bounds/weights to ensure identity checks detect a mismatch. Preserve scalar saved-fixture compatibility. Do not hash derived logs to force same-platform byte equality.
- [ ] Run `uv run pytest tests/test_dirichlet_lss_vertical_slice.py tests/test_distributional_vector_response.py tests/test_distributional_inference.py tests/test_distributional_fit_state.py -q`; keep new small lifecycle/refusal tests in this pass. Hand broad slow posterior/CV tests to the parent if needed. Commit `feat: expose Dirichlet prediction inference and validation`; provide both task commits/results and end.

## Task 7: parent integration, all-eleven regression and public documentation

**Files:** public `src/superglm/__init__.py`, `distributional/__init__.py`, `distributional/families/__init__.py`; seven already bounded adapters and `_declared.py` for audited validation changes; `tests/test_all_scalar_observations.py`, `tests/_scalar_observation_oracle.py` (new); existing bound tests and architecture/warmup census; `docs/api/distributional/families.md`, `docs/how-to/fit-a-distributional-model.md`, `docs/how-to/check-a-distributional-fit.md`, `docs/development/internals/distributional-family-development.md`; new `docs/how-to/fit-a-dirichlet-distributional-model.md`. Update the current roadmap only after implementation evidence exists, without rewriting archived receipts.

**Interfaces:** export `DirichletLSS` through existing public namespaces; no new public factory. All eleven scalar `censored_response` flags now denote complete censored **and** truncated binding. Update stale error text listing five supporting families. Complete prior APIs and old serialized models remain compatible.

- [ ] Integrate the shared foundation and three family stages, then inspect the consolidated diff. Resolve interface changes explicitly; run focused relevant tests once after each actual integration correction.
- [ ] Build the all-eleven matrix with the seven controls `GaussianLS`, `GammaLS`, `LogNormalLS`, `GeneralizedGammaLSS`, `GeneralizedParetoLSS`, `LomaxLSS`, `LogLogisticLSS` and the four additions. Test both lognormal and two-piece-lognormal parametrisations. Exercise exact, each censor side, each truncation side, two-sided and combined observations at natural/predictor order two; order-zero target equality; frequency replication and weights `1` equivalence. Nonunit prior cases succeed only for Gaussian, Gamma, NB and Tweedie; others explicitly refuse.
- [ ] Audit exact-versus-threshold validation across the seven existing families. Adopt Task 1 utilities where needed so supported boundary censor thresholds are not fed into exact-density validation. Fix tested defects under this parent ownership; preserve existing numerical restrictions, including log-logistic conditional-derivative guards. Test an interior ordinary row in every capability cell; finite-range refusals are a separate matrix, never counted as core support passes.
- [ ] Run mutation/unfixed demonstrations for endpoint inequality, atom clipping, factorial/log-response carrier, double mass, missing denominator, stale `take`/plan identity and fast-path bypass. Use isolated temporary source copies or monkeypatches, never reset someone else's working tree. Record which test fails and why.
- [ ] Document supported/disallowed operations from design section 9, numeric range versus mathematical support, prior-rate count boundary convention, small simplex normalization, covariance distinctions and convergence disclosures. The Dirichlet guide uses known continuous positive shares; no pseudocount/unknown-total/reserving suggestion.
- [ ] Run `uv run pytest tests/test_all_scalar_observations.py tests/test_lss_response_bounds.py tests/test_gamma_ls_response_bounds.py tests/test_generalized_gamma_response_bounds.py tests/test_log_logistic_window_stability.py tests/test_distributional_public_face.py tests/test_distributional_family_kernel_architecture.py -q`. Expected: all support cells actually execute; no new xfail hides approved delivery. Run architecture/generation checks for any new generated/import path.
- [ ] Commit integration/docs/tests. Reconcile plan checkboxes with executed evidence and retain unexecuted acceptance items explicitly.

## Task 8: parent complete-fit cost, full checks and independent review

**Files:** new `benchmarks/superlss_observations_dirichlet.py`, compact tracked `benchmarks/superlss_observations_dirichlet_validation.md` and source-bound receipt; use ignored `.benchmark-artifacts/` for large arrays/logs. Reuse `benchmarks/books.py` and `declared_scalar_families.py` measurement patterns without importing a runtime benchmark dependency.

**Harness interface:** `--output PATH --baseline-source PATH --case NAME` (repeatable), `--rows N`, `--repeats N`, `--discrete`, `--compact`, and `--profile`. Default cases include complete-row controls for all four additions, bounded controls for the seven existing families, each new bounded family with exact/mixed/combined windows and prior/frequency variants, and K=2/K=3/K=5 Dirichlet intercept/fixed/smooth fits. Small stage witnesses use `--rows 1000 --repeats 1`; final synthetic sizes are 20,000 and 100,000 when executable. Keep heavy Tweedie stress cells with failure/work-limit results rather than deleting them.

- [ ] Save deterministic data/spec/seeds and thread/machine/environment fingerprints. Baseline source is the approved `1b63...` implementation state, in a parent-prepared separate checkout. Do not compare new bounded fits to a complete-row fit of censored values as if their objectives matched. For new functionality report absolute fit cost plus an independent small-reference fit; for shared-path overhead use same-law baseline controls.
- [ ] Record cold and warm complete-fit timing separately in fresh processes; same-session baseline/candidate controls for any wall-time claim. Record peak RSS and retained unique buffer/cache bytes, including prepared log shares and window data; Python/JIT exclusions are explicit. Record numerical outputs, covariance/SE/EDF/rank, convergence/warnings, phase/iteration/retry/null-fit work and actual assembly/kernel dispatch. Run profiling/instrumentation separately and confirm it reproduces timed outputs.
- [ ] Test chunked/discrete exact-repeat equivalence separately from continuous-grid approximation sensitivity. Compare full fits at multiple starts where identification allows; report nonconvergence and refused optional covariance rather than filtering the receipt. Establish no universal speed/portability claim from these local runs.
- [ ] Parent runs the final long commands below once on the integrated head. No child waits for these. Focused tests and short work counts are not substitutes for the required complete-fit and full-run evidence.

```bash
uv sync --python 3.13 --extra dev
uv run python scripts/fetch_fremtpl.py --dest data/
SUPERGLM_REQUIRE_DATA=1 uv run python scripts/run_test_suite.py -n 4
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv lock --check
uv pip check
uv run python run_test.py
uv run python tools/generate_severity_families.py --check
```

After the parent prepares `.worktrees/superlss-observations-base` at the exact base (do not recreate an existing checkout), the new harness commands are:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python benchmarks/superlss_observations_dirichlet.py --baseline-source .worktrees/superlss-observations-base/src --rows 20000 --repeats 3 --output .benchmark-artifacts/superlss-observations-20k.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python benchmarks/superlss_observations_dirichlet.py --baseline-source .worktrees/superlss-observations-base/src --rows 100000 --repeats 3 --discrete --compact --output .benchmark-artifacts/superlss-observations-100k-compact.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python benchmarks/superlss_observations_dirichlet.py --case tweedie-mixed --rows 20000 --profile --output .benchmark-artifacts/superlss-observations-profile.json
```

These harness commands are specifications for Task 8's new file; it does not exist at the design base. Use its case list to isolate long cells rather than making children wait. A quick regression pass during development is `uv run python scripts/run_test_suite.py -m "not slow and not browser and not docs" -n 4`; it does not replace the final full run. The fetch may reuse verified pinned raw data when remote data is unavailable; record it and require all mandatory real-data tests to execute.

- [ ] Run SciPy 1.15 focused-family/observation/vector tests in a parent-managed isolated existing or CI environment. Run supported Python 3.12/3.13/3.14 checks and native Windows, macOS and Linux ARM64 numerical regressions. No version/lock edit is needed to test the supported floor. Record exact coverage; unrun native jobs are open acceptance, not portable success.
- [ ] Request the parent's fresh independent review of the integrated source and numerical receipt, using the parent reviewer model; the user's later instruction reserves Astra max for implementation only. Give the reviewer the original scope, source SHAs and mutation failures. Fix small findings directly, rerun only affected checks, and obtain fresh focused re-review where required; never spend a large idle agent context on a full suite.
- [ ] Final handoff reports source heads for every measured figure, supported operations/numerical range, remaining refusals and convergence failures. Future PR body states `release:minor` once, with rationale; no feature-stage version bump/tag/publication.

## Plan self-review and design-stage verification

Coverage checked against design sections 1–11: shared endpoint/normalization foundation (Task 1), both two-piece parametrisations (2), NB and prior count encoding (3), Tweedie atom/series derivatives (4), vector/Dirichlet kernel (5), public inference/CV/lifecycle/refusals (6), all-eleven regression/exports/docs (7), complete-fit/full/native/review gates (8). Shared-file writes are sequenced. No unresolved user decision blocks implementation. Numerical admission and cost budgets still need measured implementation evidence; they are assigned work, not reasons to reduce the authorized scope.

This plan's checkboxes are intentionally unexecuted in the design stage. That stage ran source inspection, primary-source reading and the small algebra probe recorded in the design, then checked document paths, Markdown links, diff scope and whitespace before committing only the two new documents. It ran no production tests or baseline suites.
