# Tweedie Profile Correctness and Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `SuperGLM.estimate_p()` default to a correct, state-safe, lazy-CI exact MLE profile with bounded Brent over `p`, an analytic log-dispersion score, and explicit convergence/boundary/fallback diagnostics.

**Architecture:** Keep the public API in `model/api.py` and `model/profile_ops.py`, while concentrating Tweedie density, inner-dispersion profiling, candidate records, and outer search in `profiling/tweedie.py`. A prepared fixed-`(y, mu, p, weights)` density evaluator computes the exact objective and analytic score in one pass. Both ordinary-fit and REML contexts run on scratch clones and cache immutable complete candidate records; public finalization refits the caller once and synchronizes every canonical result object.

**Tech Stack:** Python 3.10+, NumPy, SciPy (`wright_bessel`, `brentq`, bounded `minimize_scalar`), pandas, pytest, Ruff, mypy, uv.

---

## Task 1: Enforce the Tweedie input and prior-weight contract

**Files:**
- Modify: `src/superglm/_utils.py`
- Modify: `src/superglm/dm_builder.py`
- Modify: `src/superglm/profiling/tweedie.py`
- Test: `tests/test_tweedie_profile.py`

- [ ] **Step 1: Add failing low-level validation tests**

Add parametrized tests covering `tweedie_logpdf()` and `estimate_phi()` with one weight equal to `0`, `-1`, `nan`, and `inf`. Add shape/length tests for two-dimensional and mismatched weights. Assert the stable message contains `weights must be finite and strictly positive`.

Also add tests that reject negative `y`, non-positive/non-finite `mu`, non-positive/non-finite `phi`, invalid `p`, and non-matching `y`/`mu` shapes rather than silently returning zero-filled likelihood entries.

- [ ] **Step 2: Add failing model-level mutation tests**

For both `fit_mode="fit"` and `fit_mode="reml"`, pass a valid Tweedie dataset with one invalid `sample_weight`. Snapshot the caller's family, fitted result (when present), distribution, and prediction before the call. Assert `estimate_p()` raises before feature auto-detection or profile mutation and the snapshot is unchanged.

Add one ordinary `SuperGLM(..., family=Tweedie(...)).fit(...)` regression showing that the Tweedie family rejects zero prior weights. Add a non-Tweedie regression showing this task does not silently broaden the compatibility change to every GLM family.

- [ ] **Step 3: Run the tests and confirm RED**

Run:

```bash
rtk uv run pytest tests/test_tweedie_profile.py -q -k "invalid and (weight or input)"
```

Expected: failures show unchecked division, unchecked shapes, or caller mutation.

- [ ] **Step 4: Implement one shared strict prior-weight validator**

Add a private helper in `_utils.py` that:

- converts to `float64` only after checking the supplied array is one-dimensional;
- checks exact length against `n`;
- checks every value is finite and strictly positive;
- returns the validated array without normalizing or treating it as replication frequency.

Use it in the Tweedie branch of `dm_builder.build_design_matrix()` after distribution resolution, while retaining existing behavior for other families. Use the same helper at the start of `estimate_tweedie_p()`, before any model mutation, and in exported low-level density/dispersion functions.

Validate `y`, `mu`, `phi`, and `p` in the density preparation path. Preserve the EDM convention `phi_eff = phi / weight`.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run:

```bash
rtk uv run pytest tests/test_tweedie_profile.py -q -k "invalid and (weight or input)"
rtk uv run pytest tests/test_core.py tests/test_gaussian.py tests/test_tweedie_profile.py -q -k "weight or logpdf or estimate_phi"
```

- [ ] **Step 6: Self-review and commit**

Verify validation occurs before `model._auto_detect_features()` and that no generic frequency-weight documentation was changed in this task.

```bash
rtk git diff --check
rtk git add src/superglm/_utils.py src/superglm/dm_builder.py src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py
rtk git commit -m "Validate Tweedie prior weights and density inputs"
```

## Task 2: Prepare exact density terms and implement the analytic log-phi score

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Test: `tests/test_tweedie_profile.py`

- [ ] **Step 1: Add the all-invalid Wright-Bessel regression**

Monkeypatch `wright_bessel` to return all `nan` values for a positive-response batch whose terms select the Wright branch. Assert `_tweedie_logpdf_impl()` fills every term from `_saddlepoint()` and reports `n_saddlepoint == n_positive`. This must catch the current branch where assignment is nested under `if np.any(valid)` and the output incorrectly remains zero.

- [ ] **Step 2: Add centered-finite-difference score tests**

Create a parametrized `TestTweedieLogPhiScore` covering:

- all zero observations;
- exact positive observations;
- mixed zero/positive observations;
- unequal strictly positive prior weights;
- forced saddlepoint evaluation (`t_arg_limit=0`);
- weighted forced-saddlepoint evaluation;
- `p=1.05` and `p=1.95` edge-near cases.

For `u = log(phi)` and centered step `h=1e-5`, compare the analytic mean-NLL score with

```python
(mean_nll(exp(u + h)) - mean_nll(exp(u - h))) / (2 * h)
```

using tight absolute/relative tolerances appropriate to each branch. Assert constant weight `w` exactly matches an unweighted evaluation at `phi / w` for both log-density and score.

- [ ] **Step 3: Run the tests and confirm RED**

```bash
rtk uv run pytest tests/test_tweedie_profile.py -q -k "all_invalid_wright or log_phi_score"
```

- [ ] **Step 4: Introduce prepared density/evaluation records**

Inside `profiling/tweedie.py`, add focused frozen private dataclasses for:

- fixed arrays/masks/canonical terms for one `(y, mu, p, weights, t_arg_limit)` profile;
- one density evaluation containing `logpdf`, optional per-observation mean-NLL log-phi score, and `_TweedieLogpdfDiagnostics`.

Prepare zero-rate numerators, positive masks, `log(y)`, canonical `C`, saddlepoint deviance, `log(weight)`, and the phi-independent part of `log(t)` once. Compare `log(t)` with `log(t_arg_limit)` before exponentiation; do not alter branch selection by clipping `log(t)`.

Keep the public `tweedie_logpdf()` return type unchanged and make `_tweedie_logpdf_impl()` a compatibility wrapper over the prepared evaluator.

- [ ] **Step 5: Implement the exact branch-aligned score**

For `phi_i = phi / w_i`, implement the per-observation derivative of mean NLL with respect to `u=log(phi)`:

```text
y == 0:
    -mu**(2-p) / ((2-p) * phi_i)

exact y > 0:
    R/(p-1) + C/phi_i
    C = y * mu**(1-p)/(1-p) - mu**(2-p)/(2-p)
    R = t * W(a, a, t) / W(a, 0, t)
    a = (2-p)/(p-1)

saddlepoint y > 0:
    1/2 - deviance/(2*phi_i)
```

Use the stable Wright recurrence

```text
W(a, 0, t) = a * t * W(a, a + 1, t)
R = W(a, a, t) / (a * W(a, a + 1, t))
```

when its terms are finite and positive. The density value and score must use the same exact/saddlepoint mask. If the exact density is valid but only its derivative is invalid, mark the score invalid so the optimizer can use the exact value-only fallback; never substitute a saddlepoint score for an exact likelihood value.

Move fallback assignment outside the `np.any(valid)` branch so the all-invalid batch is populated.

- [ ] **Step 6: Run score/fallback tests and the existing density suite**

```bash
rtk uv run pytest tests/test_tweedie_profile.py -q -k "logpdf or saddlepoint or wright or log_phi_score"
```

- [ ] **Step 7: Self-review and commit**

Check score sign conventions against mean NLL, verify the weighted likelihood is not multiplied by `weights`, and ensure the SciPy minimum remains unchanged.

```bash
rtk git diff --check
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py
rtk git commit -m "Add analytic Tweedie log-dispersion score"
```

## Task 3: Replace nested value-only phi profiling with a safeguarded score solver

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Test: `tests/test_tweedie_profile.py`

- [ ] **Step 1: Add optimizer-equivalence and boundary tests**

Add tests that compare the new MLE `(phi, nll)` against an independent derivative-free bounded `minimize_scalar` reference on:

- regular exact data;
- zero-heavy data;
- unequal prior weights;
- forced saddlepoint data.

Assert agreement in both `log(phi)` and exact mean NLL. Assert the analytic score is near zero for an interior optimum. For all-zero data, assert the legitimate upper-bound optimum is returned and diagnosed rather than called an interior convergence.

- [ ] **Step 2: Add failure/fallback and evaluation-count tests**

Force a derivative-only failure and assert the value-only fallback still optimizes the unchanged exact likelihood. Force a non-finite objective and assert it cannot be reported as converged. Exercise a mocked unsuccessful bounded fallback and assert status propagates.

Instrument density evaluations and assert the score path performs one combined value/score density pass per score call. Against the current independent bounded-reference optimizer, assert fewer density passes on a deterministic representative exact profile without weakening objective/parameter tolerances; avoid wall-clock assertions.

- [ ] **Step 3: Run the tests and confirm RED**

```bash
rtk uv run pytest tests/test_tweedie_profile.py -q -k "profile_phi and (reference or boundary or fallback or evaluation)"
```

- [ ] **Step 4: Add an immutable detailed phi result while preserving compatibility**

Add a frozen `_PhiProfileResult` carrying at least:

- `phi`, `nll`, `converged`;
- total density evaluations and value-only fallback evaluations;
- optimizer/root method;
- score at the chosen candidate when available;
- lower/upper boundary status;
- whether value-only fallback was used;
- final density diagnostics and objective-finite status.

Implement a new detailed private function used by profile contexts. Preserve `_profile_phi(...) -> tuple[float, float]` as a thin compatibility wrapper because tests and downstream private users already unpack it.

- [ ] **Step 5: Implement warm-started bounded score-root profiling**

Use the previous candidate's MLE when supplied; otherwise use a finite positive Pearson/mean-deviance start. Work in `u=log(phi)` with hard bounds `[log(1e-12), log(1e12)]`.

Cache each prepared evaluator call by `u`. Starting around the warm/data start, expand a finite bracket toward the hard bounds. When finite endpoint scores have opposite signs, solve with `scipy.optimize.brentq`. Validate the root's exact NLL, score tolerance, objective improvement, and bounds.

When no trustworthy sign-changing bracket exists, the score is invalid, a branch switch creates an unusable score, or root validation fails, run bounded value-only `minimize_scalar` on the exact same NLL. Compare the chosen candidate with the finite start and finite hard endpoints. Reject all-non-finite objectives. Treat a true bound optimum as finite but boundary-diagnosed.

Use a practical score tolerance around `1e-6`; retain a safeguarded bounded optimizer tolerance precise enough for MLE equivalence, rather than the current loose `xatol=1e-3` in log-phi.

- [ ] **Step 6: Run the complete inner-phi test slice**

```bash
rtk uv run pytest tests/test_tweedie_profile.py -q -k "EstimatePhi or profile_phi or log_phi_score"
```

- [ ] **Step 7: Self-review and commit**

Confirm every optimizer exit is checked, exact objective values are never replaced by approximate ones merely because a derivative failed, and boundary success is distinguishable from interior convergence.

```bash
rtk git diff --check
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py
rtk git commit -m "Accelerate exact Tweedie phi profiling"
```

## Task 4: Make candidate fitting exact, immutable, offset-aware, and isolated

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Test: `tests/test_tweedie_profile.py`
- Test: `tests/test_constrained_fit_profile.py`

- [ ] **Step 1: Add winning-record regressions for fit and REML contexts**

Force candidate order `1.2, 1.5, 1.8`, with `1.5` winning and `1.8` evaluated last. For both contexts, assert final `phi`, NLL, EDF, convergence, and density diagnostics exactly match the winning candidate. Assert finalization does not perform an extra fit and `n_evaluations == len(search_trace)`.

This test must expose both the mutable `last_mu`/`last_edf` bug and the eager evaluation of the current `dict.get(key, self.evaluate(...))` default.

- [ ] **Step 2: Add solver-dispatch and spline-EDF regressions**

Spy a one-point fit profile and assert PIRLS receives `lambda2`, `max_iter_outer`, `tol`, `active_set`, and `convergence`. Add the equivalent direct-solver assertions for `lambda2`, `max_iter`, `tol`, `direct_solve`, and `convergence`.

Add constrained and SCOP dispatch tests matching ordinary `fit()`'s direct-solver decision. With a flexible penalized spline, nonzero selection penalty, and strong `lambda2`, compare profile trace EDF and Pearson phi to an independent ordinary fixed-`p` fit. They must agree within numerical tolerance.

- [ ] **Step 3: Add REML-offset and isolation regressions**

Use a nonconstant offset and a one-point REML grid. Independently fit an equivalent scratch model with `fit_reml(..., offset=offset)`, compute `predict(X, offset=offset)`, and assert profile phi/NLL agree and differ from the offset-free reference.

For low-level `estimate_tweedie_p()` on a pre-fitted caller, snapshot family, distribution, result and solver-result identities/values, design/groups, penalty, REML attributes, fit statistics/caches, and predictions. Assert both ordinary and REML search leave the caller unchanged, including after an explicit later CI probe.

- [ ] **Step 4: Run the tests and confirm RED**

```bash
rtk uv run pytest tests/test_tweedie_profile.py tests/test_constrained_fit_profile.py -q -k "winning_record or forwards or edf or offset or isolated"
```

- [ ] **Step 5: Cache complete immutable profile evaluations**

Add a frozen `_ProfileEvaluation` containing at least:

- exact candidate `p`, `phi`, and mean NLL;
- a copied read-only `mu` array and EDF;
- fit iterations/convergence and optional fit trace;
- the complete `_PhiProfileResult` and density diagnostics.

Replace each `dict[float, float]` NLL cache with `dict[float, _ProfileEvaluation]`. `evaluate()` may still return `record.nll` for SciPy, but trace construction and `finalize()` must read the exact cached record. Delete mutable `last_p_eval`, `last_mu`, and `last_edf`; never rerun `_profile_phi()` during finalization.

Append, without renaming existing trace fields: `edf`, `phi_converged`, `phi_n_evaluations`, `phi_boundary`, `phi_optimizer`, `objective_finite`, and saddlepoint counts/fraction.

- [ ] **Step 6: Mirror ordinary fit dispatch and controls**

Capture the scratch model's `_tol`, `_max_iter`, `_convergence`, `_active_set`, `_direct_solve`, `lambda2`, groups, and penalty. Use the ordinary-fit dispatch rule: constraints or SCOP imply direct IRLS; otherwise direct IRLS is also used for zero/no-target selection penalty. Pass all corresponding controls to `fit_irls_direct()` or `fit_pirls()`.

Preserve the ordinary-fit guards for unsupported mixed constraints/selection-penalty combinations rather than silently choosing a different solver.

- [ ] **Step 7: Run both profile paths on isolated clones**

After any necessary shorthand feature auto-detection, create a scratch clone for the ordinary profile path as REML already does. Pass `lambda2=model.lambda2` explicitly to `_clone_without_features(set(), lambda2=...)` so prior REML lambdas cannot silently seed a profile that differs from the public final refit. Build the design, calibrate penalties, warm-start coefficients, and retain the CI objective only on the scratch clone.

For REML, reconstruct the candidate mean from the scratch model's offset-aware retained `_fit_mu` or `predict(X, offset=self.offset)`, never `predict(X)` without its offset. Apply the same complete-record cache.

- [ ] **Step 8: Run context and state tests**

```bash
rtk uv run pytest tests/test_tweedie_profile.py tests/test_constrained_fit_profile.py -q -k "winning_record or forwards or edf or offset or isolated or fit_mode"
```

- [ ] **Step 9: Self-review and commit**

Check ndarray read-only flags, cache-key behavior, exact record identity at finalization, and that the caller—not merely its family attribute—is unchanged by low-level profiling.

```bash
rtk git diff --check
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py tests/test_constrained_fit_profile.py
rtk git commit -m "Make Tweedie profile evaluations state safe"
```

## Task 5: Make outer search globally honest and surface nested diagnostics

**Files:**
- Modify: `src/superglm/profiling/tweedie.py`
- Test: `tests/test_tweedie_profile.py`
- Test: `tests/test_profile_ci.py`

- [ ] **Step 1: Add endpoint and convergence regressions**

With a fake monotone objective, assert bounded Brent explicitly evaluates both `p_bounds`, selects the lower or upper endpoint when it wins, and reports which boundary. Add a case where SciPy returns `success=True` but the winning fit or inner phi did not converge; the public result must report `converged=False`.

Add all-non-finite candidate and invalid-final-record tests that raise descriptive errors rather than returning a nominally converged profile. Cover grid and grid-refine selection so a coarse/grid endpoint cannot be discarded in favor of a worse local refinement.

- [ ] **Step 2: Add evaluation-count and CI-diagnostic regressions**

Assert `n_evaluations` remains the immutable search/finalization count and equals the returned search trace length. Stub `profile_ci_p()` so the first explicit CI call probes two new values: `n_total_evaluations` must grow, `n_evaluations` must not, and a repeated same-alpha call must grow neither.

Add tests distinguishing a CI truncated at the configured range from a CI objective that is non-finite or failed; do not catch every `ValueError` and silently turn it into a range endpoint.

- [ ] **Step 3: Run the tests and confirm RED**

```bash
rtk uv run pytest tests/test_tweedie_profile.py tests/test_profile_ci.py -q -k "endpoint or boundary or converged or evaluation_count or truncated or nonfinite"
```

- [ ] **Step 4: Probe and compare bounded-search endpoints**

For bounded Brent, evaluate both `p_bounds` endpoints in addition to the optimizer candidate and finalize the best finite complete record. Apply equivalent best-seen selection to grid-refine and profile-opt initialization so a better already-evaluated candidate is never discarded.

Add outer boundary status and aggregate convergence to `TweedieProfileResult`. Aggregate success requires finite winning objective, outer search success, winning fit convergence, and winning inner-phi convergence. Boundary optima remain usable estimates but must be clearly diagnosed and warned.

- [ ] **Step 5: Expose nested result and trace diagnostics**

Extend `TweedieProfileResult` without removing existing fields. Include outer boundary, winning fit/inner convergence, inner evaluation/fallback/boundary status, and final saddlepoint information. Build warnings for outer/inner boundaries, optimizer failures, and approximation-sensitive saddlepoint fractions.

Keep `n_evaluations` as the compatibility search count. Store a private count callback into the scratch context and expose `n_total_evaluations` plus a derived post-search/CI count so later objective work is visible without mutating the original trace snapshot.

- [ ] **Step 6: Tighten CI objective handling**

Keep cached CI behavior, but explicitly reject non-finite objective values and propagate fitting/inner-profile failures. Represent or warn about genuine range truncation separately from numerical invalidity. Do not claim an LR endpoint was found when only a configured search bound was returned.

- [ ] **Step 7: Run search and CI tests**

```bash
rtk uv run pytest tests/test_tweedie_profile.py tests/test_profile_ci.py -q -k "search or grid or brent or boundary or converged or evaluation or ci"
```

- [ ] **Step 8: Self-review and commit**

Verify endpoint probes are cached, no method finalizes a worse record than one it already evaluated, and nested failures cannot be overwritten by outer optimizer success.

```bash
rtk git diff --check
rtk git add src/superglm/profiling/tweedie.py tests/test_tweedie_profile.py tests/test_profile_ci.py
rtk git commit -m "Harden Tweedie profile search diagnostics"
```

## Task 6: Switch the public contract and synchronize final fitted state

**Files:**
- Modify: `src/superglm/model/api.py`
- Modify: `src/superglm/model/profile_ops.py`
- Modify: `src/superglm/model/report_ops.py`
- Modify: `src/superglm/inference/metrics.py`
- Modify: `src/superglm/inference/summary.py`
- Modify: `src/superglm/editor/widget.py`
- Modify: `src/superglm/editor/app/summary.js`
- Modify: `src/superglm/profiling/tweedie.py`
- Modify: `docs/guide/families.md`
- Test: `tests/test_tweedie_profile.py`
- Test: `tests/test_weighted_forwarding.py`
- Test: `tests/test_editor.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_rating_table_export.py`

- [ ] **Step 1: Add default and lazy-CI contract tests**

Assert the signatures/default behavior of `SuperGLM.estimate_p()`, `model.profile_ops.estimate_p()`, and exported `estimate_tweedie_p()` all use `phi_method="mle"` and `method="brent"`.

Assert immediately after public estimation:

```python
result.phi_method == "mle"
result._ci_cache == {}
result.n_total_evaluations == result.n_evaluations
```

Make `model.summary()`, metrics summaries, editor payloads, and export paths use a stub result whose `ci()` raises if invoked. They must display the point estimate with a missing CI until `_ci_cache[alpha]` exists. After populating that cache, a fresh summary must include it; include cached-CI state in the summary cache key so an earlier no-CI summary cannot remain stale.

- [ ] **Step 2: Add Pearson inference/plot tests**

Construct a Pearson `TweedieProfileResult` directly. Assert `ci()` raises a clear error containing `exact MLE` and `bootstrap/sandwich`. Assert `profile_plot()` remains usable but omits LR cutoff/shading, labels the line `Estimate`, and uses neutral approximate-profile wording rather than `MLE`. For an MLE result with no cached interval, assert plotting does not call `ci()`; it adds LR cutoff/shading only after the caller explicitly populates the matching-alpha cache with `result.ci()`.

Assert reports say `Profile MLE (Brent)` for exact MLE and `Approximate profile (Brent; Pearson plug-in)` for Pearson.

- [ ] **Step 3: Add canonical final-state tests**

Parameterize public `estimate_p()` over ordinary/REML final refit and `retain_fit_state=True/False`. Assert:

- `model.family.p == model._distribution.p == result.p_hat`;
- public `_result.phi`, `_solver_result.phi`, and REML `pirls_result.phi` (when present) all equal `result.phi_hat`;
- `_tweedie_profile_result is result` after final refit;
- offset-aware final log likelihood equals a direct density calculation at final predictions and profiled phi;
- predictions, covariance, metrics, and summary still work after optional fit-state release.

- [ ] **Step 4: Run the tests and confirm RED**

```bash
rtk uv run pytest tests/test_tweedie_profile.py tests/test_weighted_forwarding.py tests/test_editor.py tests/test_metrics.py tests/test_rating_table_export.py -q -k "default or lazy or pearson or final_state or profile_ci"
```

- [ ] **Step 5: Make exact MLE the public default and CI explicit**

Change all three defaults to MLE and document Pearson as an explicit approximate plug-in. Remove the unconditional `result.ci(alpha=0.05)` block and its automatic `profile_ci` progress phase from `model/profile_ops.py`.

Make `TweedieProfileResult.ci()` reject non-MLE results before consulting or populating cache. For MLE, retain explicit cached CI. Change profile plotting so it only renders LR CI/cutoff labels from an already-cached matching-alpha interval; an uncached MLE plot and every Pearson plot use a neutral criterion curve and do not trigger inference.

Reports, metrics, editor payloads, and exports must only read an already-cached interval; they must never call `ci()` on the user's behalf. Update summary rendering and cache keys for an absent/present interval. Remove the editor's promise of an automatic `profile_ci` phase.

- [ ] **Step 6: Synchronize the final public model atomically**

Refactor the public final-refit path so it temporarily retains the arrays needed to apply profiled phi and recompute statistics even when the model was configured with `retain_fit_state=False`; restore the setting in `finally`.

Use `dataclasses.replace`, not in-place mutation, to apply `result.phi_hat` consistently to public `_result`, `_solver_result`, and `_reml_result.pirls_result`. Recompute offset-aware fit statistics at the final family/predictions, refresh retained fit means/null means, invalidate phi-dependent covariance/inference, metric, runtime-canonical, and summary caches, then invoke the existing fit-state release helper if retention was originally disabled. Preserve `_tweedie_profile_result` after every refit/cache operation.

- [ ] **Step 7: Update user-facing documentation and migration guidance**

In `docs/guide/families.md`:

- fix the nonexistent `p_range` example argument to `p_bounds`;
- describe MLE/Brent as defaults and Pearson as an opt-in speed/approximation tradeoff;
- show `result.ci()` as a separate expensive explicit call available only for MLE;
- state `Var(Y_i) = phi * mu_i**p / w_i`, weights are EDM prior weights rather than replication counts, and zero-weight rows must be removed;
- qualify REML as a plug-in likelihood over REML-selected smooths, not joint mgcv-style REML;
- describe exact Wright-Bessel evaluation with diagnosed saddlepoint fallback.

Update API docstrings consistently. Do not rewrite committed notebook outputs in this task; explicitly pin/label approximate examples later if a notebook test requires it.

- [ ] **Step 8: Run public/reporting tests**

```bash
rtk uv run pytest tests/test_tweedie_profile.py tests/test_weighted_forwarding.py tests/test_editor.py tests/test_metrics.py tests/test_rating_table_export.py -q -k "default or lazy or pearson or final_state or profile or summary"
```

- [ ] **Step 9: Self-review and commit**

Check default consistency across layers, ensure no implicit caller invokes `ci()`, and verify state release occurs only after all phi-dependent caches are rebuilt.

```bash
rtk git diff --check
rtk git add src/superglm/model/api.py src/superglm/model/profile_ops.py src/superglm/model/report_ops.py src/superglm/inference/metrics.py src/superglm/inference/summary.py src/superglm/editor/widget.py src/superglm/editor/app/summary.js src/superglm/profiling/tweedie.py docs/guide/families.md tests/test_tweedie_profile.py tests/test_weighted_forwarding.py tests/test_editor.py tests/test_metrics.py tests/test_rating_table_export.py
rtk git commit -m "Make exact Tweedie profiling the safe default"
```

## Task 7: Validate statistical recovery, compatibility, and measured performance

**Files:**
- Modify: `tests/test_tweedie_profile.py`
- Create: `tests/test_tweedie_profile_performance.py`
- Modify if needed: `docs/guide/families.md`

- [ ] **Step 1: Add focused statistical regressions**

Using deterministic seeds and tolerances justified by simulation variability, cover exact-MLE `p`/`phi` recovery for:

- unweighted compound Poisson-Gamma data;
- unequal EDM prior weights generated through `phi / w`;
- a zero-heavy dataset;
- a flexible penalized spline where the corrected profile EDF matches the final fit.

Keep representative correctness cases in the normal suite; mark only genuinely expensive larger simulations with `@pytest.mark.slow`. Existing search-mechanics tests that do not test phi MLE should explicitly request `phi_method="pearson"` to avoid multiplying CI-matrix runtime while retaining focused MLE coverage.

- [ ] **Step 2: Add a deterministic density-evaluation performance regression**

In `tests/test_tweedie_profile_performance.py`, compare the production inner MLE to a local derivative-free bounded reference on the same prepared exact objective. Assert:

- equal optimum NLL to a tight tolerance;
- compatible `log(phi)`;
- strictly fewer exact density passes for the production analytic path on the representative fixture;
- no hidden saddlepoint substitution or looser optimizer tolerance.

Do not assert elapsed wall time in CI. Record elapsed time only in the explicit benchmark command below.

- [ ] **Step 3: Run focused statistical/performance tests**

```bash
rtk uv run pytest tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py -q -k "recovery or performance or flexible_spline or prior_weight"
```

- [ ] **Step 4: Run format, lint, and touched-module type checks**

```bash
rtk uv run ruff format src/superglm/_utils.py src/superglm/dm_builder.py src/superglm/profiling/tweedie.py src/superglm/model/api.py src/superglm/model/profile_ops.py src/superglm/model/report_ops.py src/superglm/inference/metrics.py src/superglm/inference/summary.py tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py
rtk uv run ruff check src/ tests/
rtk uv run mypy src/superglm/profiling/tweedie.py src/superglm/model/profile_ops.py src/superglm/dm_builder.py
```

- [ ] **Step 5: Run all non-slow and full tests**

```bash
rtk uv run pytest -m "not slow" -q
rtk uv run pytest tests/ -q
```

- [ ] **Step 6: Run an explicit before/after performance comparison**

Use the committed deterministic fixture and instrumented evaluator to report search evaluations, inner density passes, saddlepoint fraction, phi fallback count, `p_hat`, `phi_hat`, NLL, and elapsed time for production analytic MLE versus the derivative-free reference. Run each case more than once and report medians; correctness comparisons precede timing comparisons.

```bash
rtk uv run pytest tests/test_tweedie_profile_performance.py -q -s
```

- [ ] **Step 7: Review the complete branch diff and commit final tests/docs**

```bash
rtk git diff --check origin/master...HEAD
rtk git status --short
rtk git add tests/test_tweedie_profile.py tests/test_tweedie_profile_performance.py docs/guide/families.md
rtk git commit -m "Verify Tweedie profile recovery and performance"
```

- [ ] **Step 8: Request final independent review**

Dispatch a fresh reviewer over `origin/master...HEAD`. Require explicit checks of the score algebra, exact-vs-saddlepoint branch alignment, solver dispatch parity, immutable winning-record finalization, caller isolation, canonical public state, lazy CI semantics, strict prior weights, optimizer/boundary propagation, and benchmark methodology. Resolve every important finding, rerun the affected focused tests, then repeat Ruff, non-slow tests, and the full suite before declaring completion.
