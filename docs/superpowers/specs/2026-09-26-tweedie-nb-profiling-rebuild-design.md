# Tweedie and NB2 profiling rebuild — design

Date: 2026-09-26 · Branch: `claude/tweedie-profiling-refactor-7be675` (from `origin/master` 9e0fe6f9) · Impact: `release:minor` (public API shrinks)

## 1. Intent

**Stated:** a complete tidy-up of Tweedie (mainly) and NB2 profiling and fitting — "a thorough clobbering". The public API may shrink.

**Standing constraints this design holds to:**

- Internal source lines are justified only by complete-fit time or by deleting complexity. Tests may stay verbose.
- No regression in complete-fit wall time or memory.
- Estimates (p̂, φ̂, CI endpoints, log-likelihood, θ̂) stay within tolerances derived from float64 error analysis, and every existing mgcv oracle passes unchanged.
- Production numerics are portable float64. Higher precision appears only in test oracles.
- Code shape: no guards that cannot fire, no nested Python loops outside compiled kernels, flat control flow, one responsibility per helper.
- One PR.

**Success criteria:**

1. The cluster below (§2) shrinks from about 10,200 source lines to **3,000 or fewer**, counted the same way before and after, with source and test lines reported separately.
2. The GLM paths use one compiled Tweedie series implementation. There are two only if the gated LSS stage (§8, stage 6) fails its gate.
3. The production Tweedie density calls none of `wright_bessel` or `ive`/`i1e`/`i0e`. It uses the saddlepoint only for rows past the series work bound. *Amended 2026-09-27 (Max):* there the peak index exceeds about 3.4e9·(a+1), and the saddlepoint's O(1/j_max) error is below 1e-10. Such rows appear only in degenerate near-noiseless fits (φ≈1e-26 in the suite). Real data needs at most a few thousand terms.
4. Every mgcv oracle test passes with unchanged tolerances:
   - `test_tweedie_reml_exact_scale.py`
   - `test_nb_theta_estimation_correctness.py`
   - the LSS oracle suites
5. Characterisation fixtures captured from the current code (§7) agree within the derived tolerances.
6. Complete-fit benchmarks (§9): no case slower than 1.05× baseline wall time, and peak RSS no higher.
7. The full suite, ruff, `uv lock --check` and the strict off-mode docs build all pass.
8. **p is recovered** (Max, 2026-09-26: "the ability to recover p"). On constant-φ compound Poisson–gamma simulations, `estimate_p` under both fit modes lands within 3 profile standard errors of the true p. The fixed grid is true p ∈ {1.2, 1.5, 1.8}, three seeds each, n = 20,000, with a spline and a factor in the mean. The SE comes from the profile curvature. This is a permanent test. Constant φ is required: a constant-φ model absorbs mean-correlated dispersion into p, and that is a property of the model, not a defect of the estimator.

   *Amended 2026-09-27 (Max):* the search runs under the fit mode the caller picks, with no automatic regime switch. `search_fit_mode` stays an explicit opt-in. REML-searched recovery at true p = 1.8 is a strict expected failure. REML cannot certify its penalized mode above p≈1.69 on these simulations, identically on master. That is a REML certification limit near p→2, recorded as a follow-up, not a profiler defect. Every other cell of the grid must pass.
9. **Fast** (Max: "fast"). `estimate_p` wall time drops below its baseline on every §9 case. Stage 0 attributes baseline time three ways: candidate fits, φ/density, and search bookkeeping. The rebuild removes the second and third buckets down to the one compiled φ solve per candidate. The PR reports the before/after breakdown.

## 2. Current state (measured 2026-09-26)

| File | Lines | Role |
|---|---|---|
| `profiling/tweedie.py` | 6,638 | Density, CPG simulation, φ MLE, five p searches, result objects, CI |
| `_tweedie_profile_kernel.py` | 662 | Two compiled Dunn–Smyth series: joint (p, log φ) statistics and per-row moments |
| `reml/scale.py` (Tweedie part) | 619 | Saturated-likelihood REML scale: p=1.5 `i1e` path, series Newton, bounded fallback plus polish |
| `model/profile_ops.py` | 790 | `estimate_p` / `estimate_theta` publication |
| `profiling/nb.py` | 1,066 | θ ML, CI, `NBProfileResult` (142 lines of that is a plot) |
| `profiling/_reporting.py` | 165 | Summary and report labels |
| `profiling/harness.py` | 281 | CPU/memory sampling harness; no caller in src |
| NB parts of `model/fit_ops.py` | ~180 | Auto-θ and the REML fixed-point alternation |

**Findings that drive the design.** "Inventory" is the 2026-09-26 audit (coverage over 4,279 tests); "sweep" is the literature sweep.

- **Four density routes.** The density evaluator switches per row between four routes: series-first, Wright (`wright_bessel`), p=1.5 `ive`, and saddlepoint. As φ moves, rows switch routes, so the φ profile is discontinuous. That discontinuity is what the 657 lines of branch-transition machinery exist to manage.
- **The series is exact; Wright is not.**
  - Against 50-digit mpmath on 237 cases (p from 1.01 to 1.99, φ from 1e-3 to 100, y from 1e-3 to 1e3, peak index up to 3×10⁵), the compiled series' worst error in log W, relative to max(1, |log W|), is 1.8e-15.
  - `wright_bessel` cannot be evaluated on 73 of those cases: its argument overflows float64 at p=1.01. Where it can be evaluated, its worst error is 1.9e-12.
  - At p=1.5 the Wright route is off by up to 3.8e-11 relative (inventory).
- **The saddlepoint biases p toward the Gaussian region** (Lunde, Kleppe & Skaug, arXiv:1811.05678). Dunn & Smyth (2005) also found it poor for data with exact zeros.
- **Search methods.**
  - `joint_ml` runs only when the model is unpenalized; the default spline penalty routes every spline model to Brent.
  - The editor never sends `auto`.
  - `grid`, `grid_refine` and `profile_opt` are standard only as diagnostics, or are superseded by Brent (sweep).
- **Dead or near-dead paths.**
  - Tests are the only callers of `_saddlepoint`, `_profile_phi` and kernel `exact_profile_statistics`.
  - `method="integrated"` passes validation and then raises `NotImplementedError`.
  - REML scale's density-evaluator fallback is run by 0 of 4,279 tests; its bounded φ search by 1.
- **NB log-likelihood drift.** NB2 has two log-likelihoods. `_nb2_nll` under frequency weights uses naive gammaln differences and drifts from `NegativeBinomial.log_likelihood` by up to 5e-8 inside the default θ bounds.

## 3. Methods adopted (literature gate)

| Quantity | Method | Source | What it assumes |
|---|---|---|---|
| Tweedie log W, 1<p<2 | Dunn–Smyth series, one per row. Start at the closed-form peak index j_max = y^(2−p) / ((2−p)φ) (within one of the argmax), sum outward to log W_max − 37, combine with log-sum-exp. | Dunn & Smyth (2005), *Stat. Comput.* 15:267 | All terms positive (1<p<2). Cost is about 17·√((p−1)·j_max) terms per row; accuracy stays at machine level. |
| d log W / d log φ and d² log W / d(log φ)² | −E[N\|y]/(p−1) and Var[N\|y]/(p−1)², moments of the same normalised terms | Wood, Pya & Säfken (2016) supplementary App. J; Dunn & Smyth (2005) | Same as above |
| φ given (p, μ) | Newton in log φ, safeguarded by bisection inside a sign-change bracket (Numerical Recipes `rtsafe`), started at the saddlepoint root | Dunn & Smyth (2005) (quasi-Newton on analytic derivatives) | p and μ held fixed. No concavity or uniqueness result is known (sweep: two alphaXiv searches plus a web search), so the bracket safeguard stays. |
| p | Bounded Brent on the profile, refitting μ at each candidate p with warm starts | Dunn & Smyth (2005); `tweedie.profile` documentation | β is orthogonal to (φ, p). The lower bound on p avoids the spurious maximum as p→1 on rounded data (Dunn & Smyth 2005). |
| CI for p and θ | Likelihood-ratio interval {2[ℓ(ĥ) − ℓ(h)] ≤ χ²₁,₁₋α}, one bracketed root per side, censored at a search bound | Venzon & Moolgavkar (1988); Fischer & Lewis, arXiv:2004.00231 | Interior optimum |
| NB θ | Score root in log θ, bracketed (MASS `theta.ml` style), alternating with the mean fit; θ → upper bound reported as the Poisson boundary | Lawless (1987), *Can. J. Stat.* 15:209; MASS `glm.nb` documentation | β asymptotically independent of θ. Boundary asymptotics apply at 1/θ = 0. |

**Not adopted, with reason:**

- **Fourier inversion** (Dunn & Smyth 2008) is needed only for p>2 or CDFs.
- **The saddlepoint** biases p.
- **`wright_bessel`** is less accurate than the series and cannot be evaluated near p=1.
- **The p=1.5 Bessel closed form** stays as a test oracle only, unless stage 0 shows it buys complete-fit time (§8).
- **Grid, grid-refine and generic optimizers** add nothing over Brent for a smooth one-dimensional profile.
- **Joint (p, φ) Newton** is valid only for unpenalized ML models, which the defaults almost never produce.

**Recorded follow-ups, not in this PR:**

- Estimating p and θ inside the LAML outer loop, as in Wood, Pya & Säfken (2016) and mgcv `tw()`/`nb()`. This is the next fit-time win, but it is new REML machinery.
- Any quantification of fixed-μ versus refitted profiles; the sweep found none.

## 4. The one identity everything shares

For an EDM, log f(y; μ, φ/w) = log f(y; y, φ/w) − w·d(y, μ)/(2φ) exactly. At μ = y a zero row contributes 0. Hence, for the positive rows only:

    Q(φ) = D / (2φ) − ℓ_sat(φ) − (M / 2)·log(2πφ)

- ℓ_sat(φ) is the saturated log-likelihood, summed over positive rows.
- ℓ_sat depends only on (y, w, p). It is prepared once per power and reused across every fit at that power.

**One φ solver minimises Q in log φ for all three callers:**

| Caller | D | M |
|---|---|---|
| p-profile, ML mean | weighted deviance | 0 |
| p-profile, REML mean | weighted deviance | 0 |
| REML scale criterion | penalized deviance Dp | penalty nullity Mp |

**Consequences:**

- The p objective keeps its current definition, −ℓ(y; μ̂(p), φ̂_ML(p)), in both fit modes.
- Each Newton step is one compiled pass over the positive rows, returning ℓ_sat, T = d(−ℓ_sat)/d log φ, and dT/d log φ.

## 5. Architecture

| Module | Responsibility | Depends on | Target lines |
|---|---|---|---|
| `src/superglm/_tweedie_series.py` (replaces `_tweedie_profile_kernel.py`) | Compiled per-row series. `series_moments(log_t, a) -> (ok, log_W, mean_j, var_j)`. Rows are independent: no global term budget and no batch coupling. It uses the per-call lgamma table (the source of the #419 speed). A row past the term cap, or whose mode is past 2^52, returns `ok=False`. | numba, numpy | ≤ 180 |
| `src/superglm/_tweedie.py` (new, private) | `TweedieRows` (prepared, phi-invariant, positive-row state for one (y, w, p)); `tweedie_logpdf`; `unit_deviance` (moved unchanged from `_tweedie_positive_unit_deviance`); `solve_log_phi(rows, D, M)`, the §4 solver; `generate_tweedie_cpg` (bit-identical draws, validation trimmed to the boundary) | `_tweedie_series`, numpy | ≤ 550 |
| `src/superglm/profiling/_scalar.py` (new) | One-parameter profile machinery shared by p and θ: a bounded Brent that records every evaluation, `likelihood_ratio_interval(objective, h_hat, nll_hat, bounds, alpha)` with per-side censoring, and one `profile_plot` | scipy, numpy, plotting (lazy) | ≤ 220 |
| `src/superglm/profiling/tweedie.py` | Tweedie p search: refit at p with warm start, φ̂ from `solve_log_phi`, record; `TweedieProfileResult` | `_tweedie`, `_scalar`, model fit entry points | ≤ 600 |
| `src/superglm/profiling/nb.py` | θ ML score root and alternation helper; `NBProfileResult`; NB log-likelihood from the family (`NegativeBinomial.log_likelihood`) only | `_scalar`, `distributions` | ≤ 320 |
| `src/superglm/reml/scale.py` (Tweedie part) | `prepare_tweedie_reml_scale_data` builds `TweedieRows`; `profile_tweedie_reml_scale` calls `solve_log_phi` and returns `ProfiledScaleTerm` (the d(1/φ)/d(Dp) contract is unchanged) | `_tweedie` | ≤ 130 |
| `src/superglm/model/profile_ops.py` | `estimate_p` / `estimate_theta` publication: search → final refit at the estimate → re-profile φ on the published mean → install | profiling, fit_ops | ≤ 450 |
| NB parts of `model/fit_ops.py` | Auto-θ and the REML fixed-point alternation, both through `profiling/nb.py` | profiling/nb | ≤ 110 |
| `profiling/_reporting.py` | Summary labels for the slimmed results | — | ≤ 80 |
| `profiling/harness.py` | **Moves** to `benchmarks/_harness.py` (its only users are two benchmarks and one test) | — | 0 in src |

**Layering change.** `distributions.py` and `reml/scale.py` import `_tweedie`, not `profiling.tweedie`. Today both import upward into `profiling/`, and that import goes away.

**Implementer latitude.** The line targets are budgets, not goals. The implementer may merge `_scalar.py` into its callers, or split a module, if that reads better and the cluster total still meets criterion 1.

## 6. Public API after the change

**`SuperGLM.estimate_p(X, y, sample_weight=None, offset=None, *, fit_mode="fit", p_bounds=(1.05, 1.95), xatol=1e-3, ci_alpha=None, max_reml_iter=None, progress_callback=None)`**

- Removed:
  - `method` and `phi_method`;
  - the `**kwargs` passthrough (`n_grid`, `grid`, `n_grid_coarse`, `optimizer`, `trace_callback`, `trace_iterations`, `verbose`, `maxiter`);
  - `search_fit_mode`, unless stage 0 measures a complete-fit gain of 1.5× or more on the REML benchmark, in which case it stays as is.
- `progress_callback` stays (the editor uses it).

**`SuperGLM.estimate_theta(X, y, sample_weight=None, offset=None, *, fit_mode="fit", theta_bounds=(1e-8, 1e8), ci_alpha=None)`**

- Explicit keywords replace `**kwargs`.
- `contract_already_checked` goes.

**Result objects**

- **`TweedieProfileResult`**
  - Fields: `p_hat`, `phi_hat`, `nll`, `converged`, `fit_mode`, `evaluations` (a DataFrame of p, nll, phi), `warnings`.
  - Methods: `ci(alpha=0.05) -> (lower, upper)` and `profile_plot()`.
  - A side that does not cross the threshold inside the bounds returns that bound, and the censoring is recorded in `warnings` and in the summary's `tweedie_p_ci_status`.
- **`NBProfileResult`**: `theta_hat`, `nll`, `converged`, `evaluations`, `warnings`, `ci()`, `profile_plot()`.

**Root exports**

- Kept:
  - `TweedieProfileResult`, `NBProfileResult`
  - `NBThetaBoundWarning`, `PublicationModeError`
  - `tweedie_logpdf(y, mu, phi, p, weights=None)` (loses `t_arg_limit`)
  - `generate_tweedie_cpg`
  - `TweedieLSS`, `NegativeBinomialLS`
- Removed:
  - `estimate_tweedie_p`, `estimate_nb_theta`, `estimate_phi`: the model methods are the single entry point;
  - `TweedieProfileCIDetails`, `TweedieProfileCIEndpoint`, `TweedieProfileCIEvaluation`, `TweedieProfileCIDensityProvenance`;
  - `profile_ci_p` and `profile_ci_theta` from `superglm.profiling`.

**Summary fields**

- Kept: `tweedie_p`, `tweedie_p_ci`, `tweedie_p_ci_status`, `tweedie_phi`, `tweedie_profile_nll`, `nb_theta`, `nb_theta_ci`, `nb_profile_nll`.
- Removed: `tweedie_p_method` and `nb_theta_method`, since there is a single method.

**Editor**

- The profile method picker and the φ-method picker go (`index.html`, `summary.js`).
- `server._profile_options` accepts only the surviving keywords.
- The editor calls `estimate_p` with defaults and `p_bounds`.

**Docs**

- Update `docs/api/families-and-links.md`, `docs/api/internals.md`, `docs/api/model/fit.md` and `docs/explanation/families-and-weights.md` (the method/φ-method/search_fit_mode prose).
- Update the notebooks `tweedie_profile_estimation.ipynb` and `editor_demo.ipynb`, which currently pass `method="brent"`.
- Update `test_tweedie_profile_docs.py` to match.

## 7. Correctness evidence

**Characterisation fixtures (stage 0, before any code changes).** A script under `benchmarks/` runs the current `origin/master` code and writes `tests/fixtures/tweedie_nb_characterisation.json`. The script and the master SHA go in the file's provenance header. It records:

- `tweedie_logpdf` on a (y, μ, φ, p) grid spanning p ∈ [1.01, 1.99] and φ ∈ [1e-3, 1e2], plus zero rows and weights.
- `estimate_p` under `fit` and `reml` on:
  - the three mgcv `re_*.csv` fixtures;
  - two synthetic CPG books: one about 90% zeros, one about 96% positive (the editor-demo shape);
  - one unpenalized design.
  
  For each it records p̂, φ̂, nll at p̂, and the 95% CI endpoints.
- REML-scale φ at fixed p on the same fixtures.
- `estimate_theta` on `nb_clamp005.csv` and `nb_worst.csv`, plus a Poisson-limit case: θ̂, nll and CI.

A private validation dataset held outside the repository is run through the same script into the scratchpad only. Its results are compared, never committed, and never named in committed text.

**Tolerances, derived rather than fitted:**

- **Series vs mpmath (new permanent oracle test, 50 digits).**
  - Bound: |Δ log W| ≤ c·ε·(|j_max·log t| + |lgamma(j_max+1)| + |lgamma(a·j_max)|), where c = 16 covers the per-term lgamma error, the log-sum-exp and the summation of the windowed terms. The bound is written out in the test next to the assertion.
  - The p=1.5 closed form √x·I₁(2√x) serves as a second oracle at larger modes.
- **New vs characterisation, `tweedie_logpdf`.** The old routes differ from exact by at most 3.8e-11 relative on Wright rows. Rows the old code sent to the saddlepoint are compared against mpmath instead of the fixture. The bound is |Δ| ≤ 4e-11·max(1, |log f|), with the 4e-11 taken from the stage 0 measurement of old-route error against mpmath, not from what passes.
- **New vs characterisation, fits.**
  - nll: bounded by the summed per-row logpdf bound.
  - φ̂: bounded by the score perturbation divided by the Q'' curvature, both recorded by the fixture script.
  - p̂: within the Brent resolution `xatol`, because a comparison flip at 1e-12 can move Brent's path by at most its final bracket. Cases that flip are listed, not hidden.
  - CI endpoints: within the root-finder tolerance plus the nll bound divided by the profile slope.
- **mgcv oracles.** Existing tests with existing tolerances, unchanged.

**Tests that pin removed internals** (the inventory classification) are deleted when the behaviour is gone:

- search methods;
- density provenance;
- branch masks;
- CI record classes.

Otherwise they are rewritten against the public maths. A new regression test must fail against the unfixed code (AGENTS.md). For each rewritten behavioural test, the implementer records the mutation that reddens it.

After the suite changes, the `.test_durations` file is regenerated so the coverage-contract check stays green.

## 8. Delivery stages

Each stage leaves the focused tests green. The full suite runs once, at stage 5.

| Stage | Work | Gate |
|---|---|---|
| 0. Baseline | Characterisation fixtures and baseline performance receipts (§9) on unmodified master code. Four measurements: (a) how often the series refuses a row (term cap or mode) across the fixtures, the private dataset and the suite's Tweedie tests, per p and φ; (b) `search_fit_mode` gain on the REML benchmark; (c) p=1.5 `i1e` vs series inside a complete `fit_reml`; (d) `joint_ml` vs Brent on the unpenalized design | Receipts written. The decisions they drive are recorded in the plan. |
| 1. Kernel and density | `_tweedie_series.py`, `_tweedie.py`, mpmath oracle test. Switch `distributions.py` and `reml/scale.py` onto them; delete the bounded fallback and polish (if stage 0 (a) shows no refusals, the fallback has nothing to catch; if it does, the refusal becomes an error that names p, φ and the row count), and the REML Bessel path unless stage 0 (c) keeps it. | mgcv REML oracles and logpdf fixtures pass |
| 2. Scalar layer and Tweedie p | `_scalar.py`, the new `profiling/tweedie.py` search and result, `profile_ops.estimate_p`, editor and summary | Fit fixtures and CI fixtures pass |
| 3. NB | `profiling/nb.py` on `_scalar`, the family log-likelihood, and the fit_ops alternation | mgcv θ oracles and NB fixtures pass |
| 4. Delete and document | Remove old code and exports; move the harness; update docs and notebooks; delete or rewrite internal-pinning tests; regenerate `.test_durations` | ruff, strict off-mode docs build |
| 5. Evidence | §9 benchmarks after the change; full suite; line counts before and after | Success criteria 1–7 |
| 6. LSS (gated) | Port `distributional/kernels/_tweedie_numba.py` onto `_tweedie_series.py`, adding the p-derivative channels (digamma/trigamma per term) that the LSS Hessian needs | The LSS oracle suites pass unchanged, TweedieLSS fit wall time is at most 1.05× baseline, and LSS kernel lines fall. On failure the stage is dropped from this PR and recorded as a follow-up. |

## 9. Performance evidence

These are complete fits on representative data, before and after, with all thread pools pinned. They are interleaved runs on a quiet machine. Profiling runs are kept separate from the timing runs that get reported.

| Case | Harness |
|---|---|
| `estimate_p`, fit and reml, ~90% zeros | `benchmarks/tweedie_profile_end_to_end.py` |
| `estimate_p`, reml, ~96% positive (editor shape) | same harness, second fixture |
| `fit_reml` Tweedie at fixed p (the #419 case) | `benchmarks/profile_tweedie_reml_fit.py` |
| REML search cost | `benchmarks/tweedie_reml_search_cost.py` |
| NB auto-θ, `fit_reml` | New stage 0 script: the `nb_worst.csv` design replicated to 200k rows, with a spline and a factor |
| Private validation dataset: `estimate_p` reml | Scratch only |

Each case reports: wall time, peak RSS, output deltas against §7, candidate-fit count, φ-solver passes per fit, and the backend dispatched. Any regression is recorded with its cause in the PR body.

## 10. Error handling

A refusal appears only for a state that occurs and is tested:

- ~~A series row past the term cap or the safe mode raises `FloatingPointError` naming p, φ and the row count.~~ *Amended 2026-09-27 (Max, criterion 3):* such a row takes the saddlepoint, so it is not a refusal state (Task 7b).
- No interior φ optimum (2·N_pos ≤ (p−1)·M) raises `ValueError`, unchanged.
- p̂ or a CI side at a search bound: a warning recorded in `result.warnings`. This is a real statistical state (Dunn & Smyth's spurious maximum as p→1).
- A REML candidate power whose mode cannot be certified is skipped as infeasible, as today. The search routes around it, and the result records it.
- NB θ at the upper bound triggers `NBThetaBoundWarning` (the Poisson boundary), unchanged.

Input validation happens once, at `estimate_p` / `estimate_theta` / `tweedie_logpdf`, and nowhere below them.

## 11. Out of scope

- p and θ inside the REML outer loop (the Wood 2016 extended family).
- Fourier inversion and p>2.
- Stage 6, if it fails its gate.
- Any change to the Tweedie unit deviance numerics. That function moves unchanged.
