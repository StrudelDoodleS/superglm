# Subsystem report: families-profiling

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ origin/master (f082e9b).
Files: `src/superglm/families.py` (67 L), `src/superglm/links.py` (466 L), `src/superglm/distributions.py` (418 L), `src/superglm/profiling/tweedie.py` (6011 L), `src/superglm/profiling/nb.py` (599 L). Supporting modules read for context: `src/superglm/_tweedie_series.py` (207 L), `src/superglm/_tweedie_profile_kernel.py` (491 L), `src/superglm/profiling/__init__.py`, `profiling/_reporting.py`, `profiling/harness.py`.

All paths below are relative to `src/superglm/` unless prefixed.

---

## 1. MODULE MAP

### 1.1 `families.py` — convenience constructors (67 lines)

Pure sugar: `poisson()` (families.py:26), `gaussian()` (:31), `gamma()` (:36), `binomial()` (:41), `nb2(theta)` (:46), `tweedie(p)` (:58). Each returns the corresponding `distributions.py` class instance. `nb2("auto")` is the trigger for automatic theta profiling in `fit()` (checked at model/fit_ops.py:650). No logic, no state. Callers: user code; re-exported from `superglm/__init__.py:26`.

### 1.2 `links.py` — link functions (466 lines)

- `Link` protocol (links.py:18–47), `@runtime_checkable`, requires `link/inverse/deriv/deriv_inverse`; `deriv2_inverse`/`deriv3_inverse` documented as optional, detected by `hasattr`.
- Concrete links, all stateless except two parametric ones:
  - `LogLink` (:50), `IdentityLink` (:73), `LogitLink` (:96), `ProbitLink` (:132), `CloglogLink` (:173), `CauchitLink` (:207), `InverseLink` (:236), `InverseSquaredLink` (:262), `SqrtLink` (:288), `PowerLink(power)` (:314), `NegativeBinomialLink(theta)` (:357). **All eleven implement `deriv2_inverse` and `deriv3_inverse`** (Wood 2011 App. D formulas), so the "optional" fallback path is only ever taken for user-supplied custom links.
- `stabilize_eta(eta, link)` (:404–429): per-link-class eta clipping used by every solver/prediction path (grep: solvers/pirls.py, solvers/irls_direct.py via working rows, _reporting_state.py:127, editor/apply.py:475, diagnostics/*, inference/_term_covariance.py:44, profiling/tweedie.py:3783, profiling/nb.py:489).
- `resolve_link(link, family)` (:445): string → class map `_LINK_SHORTCUTS` (:432). Called from dm_builder.py:701 during design-matrix construction.

Callers of the second/third derivatives (the REML W(ρ) correction): reml/w_derivatives.py:72–125 (hasattr-guarded, returns `None` fallback) and reml/observed_geometry.py:263–291,463–474 (**raises `NotImplementedError`** when absent — see Suspects §5.5).

### 1.3 `distributions.py` — EDM families (418 lines)

- `Distribution` protocol (distributions.py:22–55): `scale_known`, `default_link`, `variance`, `deviance_unit`, `log_likelihood`; optional `variance_derivative`, `variance_second_derivative` (REML W(ρ) correction — same consumer pair as the link derivatives).
- Families: `Poisson` (:58), `Gaussian` (:94), `Gamma` (:129), `NegativeBinomial(theta|"auto")` (:164), `Binomial` (:228), `Tweedie(p)` (:268). All six implement both variance derivatives.
- `Tweedie.deviance_unit` (:303–307) and `Tweedie.log_likelihood` (:309–314) **delegate via deferred import to `profiling/tweedie.py`** (`_tweedie_positive_unit_deviance`, `tweedie_logpdf`) — a layering inversion (Suspects §5.2).
- Module helpers: `DISTRIBUTION_SHORTCUTS` (:317), `resolve_distribution` (:325), `validate_response` (:355, includes a custom-hook dispatch at :385–388), `initial_mean` (:391), `clip_mu` (:408). Numeric guard constants `_POSITIVE_MU_MIN/MAX`, `_VARIANCE_FLOOR` (:16–19) are imported by solvers (solvers/working_rows.py:11, solvers/pirls.py:18, solvers/irls_direct.py:40, inference/_term_covariance.py:11, _reporting_state.py:11) — a private constant used as a cross-module contract.
- Consumers of `variance`/`deriv_inverse` in the PIRLS working-weight formula: solvers/working_rows.py:53–55, solvers/pirls.py:500–502, 823–825, 1282–1283; deviance via solvers/irls_state.py:94.

### 1.4 `profiling/tweedie.py` — Tweedie p/phi profile likelihood (6011 lines)

**What it does:** estimates the Tweedie power parameter p by profile likelihood. Outer loop over p; at each candidate p it (a) fits the full penalised GLM (via `fit_pirls`/`fit_irls_direct`, or `model.fit_reml()` on the REML path), (b) profiles out the dispersion φ at fixed (μ, p) — either Pearson plug-in or exact MLE over log φ — and returns mean NLL. It also owns: the exact Tweedie log-density (Wright-Bessel / vectorized series / p=1.5 Bessel / saddlepoint hybrid), compound Poisson–Gamma simulation, profile-likelihood CIs for p, and result plotting.

Section map (with imports at :40–56 pulling in `model.fit_state`, `penalties.base`, `solvers.irls_direct`, `solvers.pirls`, and `_tweedie_profile_kernel` → **numba at module import**):

| Lines | Section | Key symbols |
|---|---|---|
| 60–313 | CPG simulation | `generate_tweedie_cpg` (:251) + ~200 lines of validators `_normalize_cpg_*` (:75–133), `_draw_cpg_counts` (:167), `_draw_cpg_positive_values` (:195) |
| 320–435 | logpdf state/validation | `_TweedieLogpdfDiagnostics` (:320), `_PreparedTweedieDensity` (:335), `_TweedieDensityEvaluation` (:358), `_validate_tweedie_inputs` (:376) |
| 437–512 | unit deviance | `_tweedie_positive_unit_deviance` (:437) — 4-branch cancellation-safe deviance (series near μ, log1p/expm1 regular, log-scale for delta≈−1, factored form for y≫μ) |
| 515–985 | density evaluation | `_prepare_tweedie_density` (:515), `_evaluate_tweedie_density` (:605) — branch dispatch: series-first (:672), Wright-Bessel exact (:679), p=1.5 scaled `ive` Bessel with asymptotic tail (:714), series retry (:779), saddlepoint fallback (:784); optional analytic d/d(log φ) score (:800–853). Public `tweedie_logpdf` (:896), `_tweedie_logpdf_pair` (:937, used by model/fit_ops.py:294–306 for fit stats), dead `_saddlepoint` (:981) |
| 988–1039 | Pearson dispersion | `_tweedie_pearson_contributions` (:993), public `estimate_phi` (:1004) |
| 1042–2798 | **φ profile (MLE)** | constants (:1042–1061); `_PhiProfileResult` (:1064), `_PhiBranchMask` (:1097), `_PhiProfilePoint` (:1134), `_PhiEvaluationCache` (:1192); analytic-score search `_search_phi_score_candidates` (:1596) with brentq root (:1739); **branch-switch certification machinery**: `_positive_saddlepoint_mask_at_u_values` (:1375), `_locate_first_realized_phi_branch_transition` (:1427), `_better_phi_branch_edge_probes` (:1517), `_calibrate_wright_log_t_ceiling` (:1889), `_verified_phi_branch_transitions_at_thresholds` (:2009, chunked over all positive obs), `_run_phi_bounded_fallback` (:2138, global grid scan over log φ ∈ [−27.6, 27.6]); `_finalize_phi_mle_result` (:2308); compiled fast path `_profile_phi_exact_newton` (:2489, calls the numba kernel); dispatcher `_profile_phi_detailed` (:2688), thin `_profile_phi` (:2780) |
| 2801–3680 | result types & plots | `_TRACE_COLUMNS` (:2805), density classification `_classify_density_diagnostics` (:2900), CI dataclasses (:2951–3016), `TweedieProfileResult` (:3018) incl. pickle/legacy compat (`__post_init__`/`__setstate__`/deprecated `.cache`, :3109–3169), `ci()` (:3218), `ci_details()` (:3277), `trace_plot` (:3297), `profile_plot` (:3365) |
| 3682–4290 | profile contexts | `_ProfileEvaluation` (:3577), `_ProfileContext` (fit path, :3687; `_fit_at_power` :3740 dispatches to `fit_irls_direct`/`fit_pirls`; `evaluate` :3832; `evaluate_exact_phi` :3862), `_clone_profile_model` (:3945), `_build_profile_context` (:3989, replicates fit() guards), `_ProfileContextREML` (:4103, calls `model.fit_reml` per p at :4151), `_build_profile_context_reml` (:4238) |
| 4290–5130 | search methods | `_finalize_profile_record` (:4349), `_search_brent` (:4534), `_search_grid` (:4569), `_search_grid_refine` (:4597), `_search_profile_opt` (:4655, logit-transformed p), joint fast path: `_joint_ml_fallback_to_brent` (:4738), `_predict_joint_phi` (:4777), `_validate_joint_profile_record` (:4799), `_search_joint_ml` (:4852, safeguarded Newton on p with fused exact derivatives) |
| 5137–5316 | public entry | `estimate_tweedie_p` (:5137); `method="auto"` → joint_ml when `fit_mode="fit"` and `phi_method="mle"`, else Brent (:5294–5296); REML + joint_ml silently degrades to Brent (:5304–5314); `"integrated"` reserved/NotImplemented (:5261) |
| 5319–6011 | profile CI for p | `_profile_ci_p_detailed` (:5653, outward scan + brentq on LR cutoff), `_aggregate_ci_density_provenance` (:5482), public `profile_ci_p` (:5985) |

Callers: `superglm/__init__.py:84–93` (public re-exports; **imported at package import time**), model/profile_ops.py:39,71 (`SuperGLM.estimate_p` orchestration), model/fit_ops.py:294 (`_tweedie_logpdf_pair`, `_tweedie_pearson_contributions` for fit statistics), distributions.py:305,311 (Tweedie deviance/loglik), stats/model_tests.py:125, plotting/diagnostics.py:157 (`generate_tweedie_cpg` for randomized quantile residuals).

### 1.5 `profiling/nb.py` — NB2 theta estimation (599 lines)

- `NBProfileResult` (nb.py:42–270): result dataclass with publication-locking `__setattr__` (:60), immutability plumbing (`_published_with_data` :65, `_detached_public_copy` :91, `__deepcopy__` :108, `__getstate__/__setstate__` :129–144), `ci()` (:146), `profile_plot` (:162).
- `_theta_ml` (:273–323): Newton on the closed-form digamma/trigamma NB2 profile score (Lawless 1987), O(n) per step, ≤10 steps.
- `_nb2_nll` (:332): weighted mean NB2 NLL (duplicates `NegativeBinomial.log_likelihood` algebra, distributions.py:215–225).
- `estimate_nb_theta` (:344–538): alternating GLM-fit/profile-score-update scheme (Venables & Ripley 2002, ch. 7.4) — build design once (**mutating the caller's model**, :409–419), then per outer iteration: warm-started GLM fit (`fit_irls_direct` :460 with optional REML penalty context :438–447, or `fit_pirls` :475) → `_theta_ml` update; converge on |Δθ| < xatol (~3–5 iterations).
- `profile_ci_theta` (:541–599): fixed-μ LRT inversion via brentq, O(n) per evaluation.

Callers: model/profile_ops.py:292,316 (`SuperGLM.estimate_theta`), model/fit_ops.py:647–653 (`_maybe_estimate_nb_theta`, auto-run during `fit()`/`fit_reml()` when `theta="auto"`, call sites :802, :1155), `superglm/__init__.py:83`.

### 1.6 Supporting modules (not in scope but load-bearing)

- `_tweedie_series.py`: `tweedie_log_series` (:92) — vectorized batched compound-Poisson normalizer with a global term budget (default 1e6 terms, batches of 262144); returns log mass, E[J] (for the analytic φ score), and per-row exactness flags.
- `_tweedie_profile_kernel.py`: numba `@njit` kernel `_exact_profile_statistics_kernel` (:166–411) — per-row series sweep producing mean NLL plus full gradient/Hessian in (p, log φ) (`ExactProfileStatistics` :23); own scalar digamma/trigamma implementations (:39–102). Used only by `_profile_phi_exact_newton` and `_search_joint_ml`.
- `profiling/harness.py` / `profiling/_reporting.py`: the former is **benchmark telemetry** (SystemSample etc.), unrelated to statistical profiling, imported only by `benchmarks/profile_superbooster_interactions.py`; the latter is read-only summary/report helpers (`cached_tweedie_profile_ci` :20) used by inference/metrics.py:37 and editor/widget.py:45.

---

## 2. DATA FLOW

Notation: n = rows, p_cols = built design columns, m = smooth terms, q = penalties. This subsystem is almost entirely O(n)-vector algebra; the p_cols/m/q-dimensional work happens inside the solvers/REML it dispatches to.

### 2.1 Family/link plumbing (per PIRLS iteration, in solvers)

`eta (n,) → stabilize_eta → link.inverse → mu (n,) → clip_mu → family.variance (n,) → W = w · (dμ/dη)²/V (n,)` (solvers/working_rows.py:53–55). Each link/distribution method allocates 1–3 fresh (n,) temporaries. REML W(ρ) correction additionally evaluates `deriv2_inverse`, `deriv3_inverse`, `variance_derivative`, `variance_second_derivative` — four more (n,) arrays per REML iteration (reml/w_derivatives.py:75–125).

### 2.2 `estimate_tweedie_p` fit path

1. `_snapshot_profile_inputs` (:3979) deep-copies X and copies y/w/offset → +O(n·raw cols).
2. `_clone_profile_model` (:3945) clones model; `_build_profile_context` (:3989) builds the design matrix **once** (n × p_cols, the dominant allocation) under a temporary `Tweedie(p=1.5)` family (:4026–4035).
3. Search method calls `ctx.evaluate(p)` (:3832). Per distinct p:
   - `_fit_at_power` (:3740): one full warm-started GLM fit → `result.beta` (p_cols,), then `eta = dm.matvec(beta)+…` and `mu (n,)`.
   - `_profile_phi_detailed` (:2688): `_prepare_tweedie_density` (:515) materialises ~10 read-only arrays — 6 of length n (`y, mu, weights, zero_mask, positive_mask, zero_rate_numerator, log_weight`) plus 5 of length n₊ (positive rows). Each φ evaluation (`_PhiEvaluationCache.evaluate` :1203 → `_evaluate_tweedie_density` :605) allocates ~8–12 temporaries of length n₊ and possibly runs the vectorized series (≤1e6 terms) and/or `wright_bessel` over n₊ rows; each cached `_PhiProfilePoint` retains a packed branch mask of n₊/8 bytes.
   - `_store_evaluation` (:3790) **copies μ (n,) into the immutable record** (:3804) and advances warm starts.
4. `finalize` → `_finalize_profile_record` (:4349) materialises the trace DataFrame and builds `TweedieProfileResult` whose `_objective=ctx.evaluate` (:4435) **retains the whole context (design matrix, y, w, all cached μ vectors) for the life of the result** so that `ci()`/`profile_plot()` can lazily fit at new p values.

### 2.3 joint_ml fast path

`_search_joint_ml` (:4852): per candidate p, one GLM fit + `_profile_phi_exact_newton` (:2489) which calls the numba kernel (O(total series terms), O(1) memory, fused NLL/gradient/Hessian in (p, log φ)); Newton in log φ (≤12 iters × ≤12 halvings), then a Newton/secant step in p with cross-curvature φ prediction (`_predict_joint_phi` :4777). Winner is re-validated through the authoritative vectorized density (`_validate_joint_profile_record` :4799). Any failure certificate downgrades the entire search: `_joint_ml_fallback_to_brent` (:4738) **re-profiles every cached record's φ through the defensive path** before Brent reuses the cache.

### 2.4 REML path

`_ProfileContextREML.evaluate` (:4139): sets `model.family = Tweedie(p)` and calls `model.fit_reml()` — a full nested REML optimisation (outer p × REML ρ iterations × PIRLS) **with no solver warm start across p**, then the same φ profile on the fitted μ. This wraps the entire REML machinery in a scalar optimizer; the φ profile is an ML profile layered on a REML fit (documented as a profile objective, not pure ML).

### 2.5 φ MLE profile detail

Seeds (warm/Pearson/mean-deviance, :1330) → analytic-score bracketing with branch-signature guards (:1596–1714) → brentq on the score (:1739) → probe validation (:1767–1790) → optional branch-edge probes (:1517). Any doubt (including **always when p < 1.02**, :2763) triggers `_run_phi_bounded_fallback` (:2138): full-range bounded minimize + ~56-point grid over log φ + up to 64 analytic edge probes ×2 + 128 numeric bisection probes + 256 root-branch probes + **chunked verification of every positive observation's branch edge** (:2009–2048, 2 `wright_bessel` scalar-vector calls per 65536-row chunk) + up to 8 basin refinements each running `minimize_scalar` (≤200 iterations).

### 2.6 NB path

`estimate_nb_theta`: design built once on the **caller's** model; alternation loop holds `warm_beta (p_cols,)`, `mu (n,)`; `_theta_ml` is O(n) per Newton step. Result copies y/μ/w into bytes-backed immutable buffers (nb.py:326–329) for later `ci()`/plots — +3·O(n) retained per result.

---

## 3. STATE OBJECTS

| Object (location) | Fields / lifecycle | Overlap notes |
|---|---|---|
| `_PreparedTweedieDensity` (tweedie.py:335) | 11 read-only arrays (6×n, 5×n₊) + scalars; created fresh per `_profile_phi_detailed`, `_profile_phi_exact_newton`, `_tweedie_logpdf_pair`, `_validate_joint_profile_record` call | Re-prepared repeatedly for the same (y, μ, p) within one outer evaluation (Newton fast path + fallback + validation each prepare their own) |
| `_TweedieDensityEvaluation` (:358) | logpdf (n,), optional score (n,), saddle mask (n₊,), diagnostics | One per φ evaluation; immediately reduced to scalars by the cache |
| `_TweedieLogpdfDiagnostics` (:320) | n_positive/n_saddlepoint/n_series | Re-derived into `_DensitySummary` (:2852) at every reporting site |
| `_PhiBranchMask` (:1097) / `_PhiProfilePoint` (:1134) / `_PhiCandidate` (:1160) | packed branch bits + scalars per evaluated log φ; live in `_PhiEvaluationCache.points` dict for one φ profile | |
| `_PhiEvaluationCache` (:1192) | prepared density + point dict + 4 eval counters; per-profile lifetime | |
| `_PhiScoreSearchResult` (:1168), `_PhiBoundedResult` (:1178), `_ExactPhiNewtonOutcome` (:1087) | intermediate result carriers | |
| `_PhiProfileResult` (:1064) | 19 fields of φ diagnostics | Flattened nearly 1:1 into `TweedieProfileResult.phi_*` fields (:3093–3103) **and** into trace rows (`_materialize_profile_trace_row` :3623) — three parallel representations |
| `ExactProfileStatistics` (_tweedie_profile_kernel.py:23) | status + NLL + 5 derivatives + counts | Cached per p in `_ProfileContext._exact_statistics_cache` (:3721) |
| `_ProfileEvaluation` (tweedie.py:3577) | step, p, **μ copy (n,)**, edf, fit flags, fit trace, `_PhiProfileResult` | Insertion-ordered cache = search trace; retained via result closures |
| `_ProfileContext` (:3687) / `_ProfileContextREML` (:4103) | design matrix / model + warm state + evaluation cache | Two near-parallel implementations of evaluate/finalize/trace (§5.9); retained by `TweedieProfileResult._objective` |
| `TweedieProfileResult` (:3018) | ~35 public fields + private CI caches, `_objective`, `_evaluation_count/record` callbacks; legacy pickle shims (:3109–3186) | Aggregates record + φ result + density summary; long-lived, keeps context alive |
| `TweedieProfileCI{Evaluation,Endpoint,Details,DensityProvenance}` (:2951–3016), `_CIDensityAggregate` (:5456) | immutable CI evidence | Density provenance duplicates `_DensitySummary` content per point |
| `_DensitySummary` (:2852) | validated density classification | recomputed at :3117, :3627, :4367, :5930 |
| `NBProfileResult` (nb.py:42) | θ̂, nll, cache map, immutable y/μ/w copies, publication lock | Much lighter than Tweedie counterpart; no trace DataFrame |

---

## 4. COMPLEXITY TABLE

All dense NumPy unless stated. n₊ = positive-response rows; E = number of outer p evaluations (Brent typically 10–25, grid = n_grid); F = φ evaluations per profile (score path ~8–20; fallback ~60–500+).

| Routine | Time | Memory | Notes |
|---|---|---|---|
| link/dist methods (links.py, distributions.py) | O(n) | 1–3 (n,) temporaries each | `IdentityLink.link/inverse` copy (:77,:80); logit paths compute `expit` twice per weight build |
| `_tweedie_positive_unit_deviance` (tweedie.py:437) | O(n) | ~10 (n,) temporaries | 4 masked branches; 8-term series loop on near rows |
| `_prepare_tweedie_density` (:515) | O(n) | 6×(n,) + 5×(n₊,) **copies** (read-only) | called ≥1× per outer p evaluation, plus again for validation paths |
| `_evaluate_tweedie_density` (:605) | O(n₊) `wright_bessel` + optional series O(≤1e6 terms) | ~8–12 (n₊,) temporaries per call | series-first pre-pass for all rows when n₊≥32 and p≠1.5 (:672–677) — inside every φ evaluation |
| `tweedie_log_series` (_tweedie_series.py:92) | O(total terms ≤ max_total_terms) | O(batch)=262144 floats | deterministic budget selection O(n₊ log n₊) lexsorts |
| `_exact_profile_statistics_kernel` (_tweedie_profile_kernel.py:166) | O(Σ per-row series width), numba scalar loop | O(1) | work-limited (1e5/row, 1e6 total); returns full (p, log φ) Hessian in one sweep |
| `_profile_phi_detailed` mle (:2688) | F × [O(n₊) + series] | cache: F points × (n₊/8 B mask) | fallback adds ~56-grid + ≤64 edge×2 + ≤128+256 probes + ≤8 `minimize_scalar` runs |
| `_verified_phi_branch_transitions_at_thresholds` (:2009) | O(n₊) scalar-threshold Wright evals ×2 | O(chunk)=65536 | full-population pass inside the fallback safeguard |
| `_profile_phi_exact_newton` (:2489) | ≤12×12 kernel calls | O(1) beyond prepared density | plus one full vectorized validation pass when `validate=True` |
| `ctx.evaluate` fit path (:3832) | 1 GLM fit: O(iter·(n·p_cols² + p_cols³)) dense (delegated) + φ profile | +1 μ copy (n,) retained per distinct p | warm-started; cache keyed by exact float p — brentq probes at 1e-10 spacing never reuse |
| `ctx.evaluate` REML path (:4139) | 1 **full fit_reml** (REML iters × PIRLS × O(n·p_cols²+p_cols³) + q-dim work) + φ profile | +1 μ copy (n,) per p | no cross-p warm start |
| `estimate_tweedie_p` (:5137) | E × above | context + E μ-copies retained by result | joint_ml typically E≈5–10 with O(1)-memory inner profiles |
| `_joint_ml_fallback_to_brent` (:4738) | re-runs defensive φ profile for every cached p | — | fallback doubles inner-profile work already paid |
| `ci()` / `_profile_ci_p_detailed` (:3218/:5653) | up to ~16 scan points + 2 brentq roots (≤2 tolerance passes), each a **full GLM fit + φ profile** | evidence dicts | can exceed the original search cost |
| `estimate_nb_theta` (nb.py:344) | ≤maxiter(30, typ. 3–5) GLM fits + O(10n) Newton | 3×(n,) immutable copies in result | warm-started |
| `profile_ci_theta` (nb.py:541) | O(n) per brentq iteration, fixed μ | O(n) temporaries | cheap by design |

---

## 5. SUSPECTS

1. **numba (and pandas) imported at package import time via the Tweedie module.** `superglm/__init__.py:84` imports `profiling/tweedie.py`, whose top-level imports include `_tweedie_profile_kernel` (tweedie.py:41–48), which does `from numba import njit` (_tweedie_profile_kernel.py:9), plus `import pandas` (tweedie.py:35). Every `import superglm` pays numba's import cost even for pure-Poisson users. Verify: time `import superglm` with/without numba cached; check whether the kernel import can be deferred to `_profile_phi_exact_newton`.

2. **Layering inversion: `distributions.Tweedie` depends on the 6000-line orchestration module.** distributions.py:305,311 deferred-import `_tweedie_positive_unit_deviance`/`tweedie_logpdf` from profiling/tweedie.py, which itself imports `model.fit_state`, `penalties.base`, and both solvers (tweedie.py:53–56). Computing a Tweedie deviance inside PIRLS thus (first time) imports the model/solver stack from below. The density/deviance code (≈lines 320–1039) is self-contained and separable from the search orchestration. Verify: import graph; whether the density block has any real dependence on the model imports.

3. **φ branch-certification machinery is ~1750 lines (tweedie.py:1042–2798) and its fallback is O(n)-heavy inside the outer p loop.** The apparatus exists to certify that a brentq root of the analytic score is not an artifact of the exact↔saddlepoint density branch switching (Wright-Bessel validity edges). Any fallback reason — including **unconditionally whenever p < `_PHI_GLOBAL_COMPARE_POWER` = 1.02** (:1061, :2763–2769) — triggers `_run_phi_bounded_fallback` (:2138): ~56-point log-φ grid, ≤64 analytic edge probes ×2, ≤128+256 bounded probes, per-positive-observation chunked edge verification (:2009–2048), ≤8 bounded refinements ×200 iterations. Each probe is a full O(n₊) density pass. This can run once per outer p evaluation. Verify: profile `estimate_tweedie_p` on data with p̂ near 1.05 and count `n_fallback_evaluations` in the trace.

4. **Series-first heuristic runs the full vectorized series for all rows on every φ evaluation** when n₊ ≥ 32 and p ≠ 1.5 (:672–677), before trying the cheap Wright-Bessel path — up to 1e6 series terms per φ point inside brentq. Verify: benchmark `_evaluate_tweedie_density` with/without the pre-pass on mid-range t where `wright_bessel` alone suffices.

5. **Doc–code mismatch on optional derivative fallback.** links.py:26–31 and distributions.py:30–33 promise "if absent, the correction is skipped and REML falls back". reml/w_derivatives.py:72–75 honors that (returns None), but reml/observed_geometry.py:263–266, 283–288 **raises `NotImplementedError`** for the same missing attributes. Custom link/distribution objects would hit different behavior depending on code path. Verify: which fit configurations reach observed_geometry with custom families.

6. **`stabilize_eta` has no branch for `SqrtLink` or `CauchitLink`** (links.py:404–429): both fall into the custom-link catch-all `clip(eta, -20, 20)`. For sqrt link this caps μ = η² at 400 (silent prediction ceiling for Poisson counts with mean > 400); for cauchit it caps μ to ≈[0.016, 0.984], defeating the heavy-tailed link's purpose. Verify: fit Poisson(sqrt) with mean ≈ 1000 and inspect fitted μ.

7. **Dead code:** `_saddlepoint` (tweedie.py:981–985) has no callers anywhere in src/ (grep confirms; the live saddlepoint math is inlined in `_prepare/_evaluate_tweedie_density`). Also `profiling/harness.py` is benchmark telemetry living inside the statistical profiling package, imported only by `benchmarks/`; and `_reporting.py:93–102` `_search_method_label` maps keys `"lbfgsb"`/`"powell"` that can never occur (actual `result.method` values are `brent/grid/grid_refine/profile_opt/joint_ml`; `joint_ml` renders as "Joint Ml").

8. **Memory retention by results.** Every `_ProfileEvaluation` stores an owned μ copy (n,) (tweedie.py:3804, :4191) — E×n floats for the search; `TweedieProfileResult._objective = ctx.evaluate` (:4435) then keeps the entire context (design matrix n×p_cols, y, w, all μ copies, exact-statistics caches) alive as long as the result (which `SuperGLM.estimate_p` attaches to the model). At n = 10⁷ and 15 evaluations this is ≈1.2 GB of μ copies plus the design. Verify: RSS after `estimate_p` returns vs. after `del result`.

9. **Duplicated orchestration and algebra:**
   - `_ProfileContext` vs `_ProfileContextREML` (tweedie.py:3687/:4103): parallel evaluate/finalize/trace implementations.
   - `_build_profile_context` (:4014–4073) replicates `fit()`'s guard/dispatch logic (lambda_policy check, monotone/SCOP exclusions, `use_direct` selection) — drift risk against model/base+fit_ops.
   - Exact-density winner validation appears twice nearly verbatim: `_profile_phi_exact_newton` (:2612–2650) vs `_validate_joint_profile_record` (:4811–4849).
   - Three implementations of the compound-Poisson series: vectorized (`_tweedie_series.py`), numba with derivatives (`_tweedie_profile_kernel.py`), and the Wright/p=1.5-Bessel branches in `_evaluate_tweedie_density`, with different work budgets (1e6 vs `_FIT_STATS_SERIES_MAX_TOTAL_TERMS`=4096 at :431/:960).
   - `_nb2_nll` (nb.py:332) duplicates `NegativeBinomial.log_likelihood` (distributions.py:215–225); Pearson φ duplicated at tweedie.py:1004 and :1311.

10. **Isolation asymmetry: `estimate_nb_theta` mutates the caller's model.** nb.py:409–419 runs `model._auto_detect_features`, temporarily reassigns `model.family`, and builds `model._dm`/`model._groups` on the passed model, whereas the Tweedie path deep-snapshots inputs and clones (`_clone_profile_model`, tweedie.py:3945–3976; `_snapshot_profile_inputs` :3979). Public `profiling.estimate_nb_theta` therefore has side effects the Tweedie twin was specifically engineered to avoid. (Via `SuperGLM.fit`, a workspace model absorbs this; direct calls do not.) Also nb.py:526: if `maxiter ≤ 0`, `mu` is unbound (NameError). Verify: call `estimate_nb_theta(model, …)` directly and inspect `model._specs/_dm` afterwards.

11. **Exact-float cache keys defeat reuse across optimizer probes.** `_ProfileContext._evaluation_cache` is keyed by `float(p)` exactly (:3837); `minimize_scalar`/brentq evaluate p values differing by ~1e-10 during convergence checks, each costing a full GLM fit. Similarly `_PhiEvaluationCache` keys on exact `u` (:1211). Verify: count distinct p in `search_trace` vs. nominal Brent iterations.

12. **Defensive validation volume.** ~150 lines validate numpy's own sampler outputs in `generate_tweedie_cpg` (:167–248, e.g. re-checking dtype/shape/finiteness of `rng.poisson` results); nearly every private function re-validates already-validated arrays (`_validate_tweedie_inputs` called from `tweedie_logpdf`, `estimate_phi`, `_prepare_tweedie_density`, …). This is a major contributor to the 6000-line size along with the branch-certification machinery (§3), embedded plotting (:3297–3529), CI provenance bookkeeping (:5455–5650), and legacy pickle compat (:3109–3186).

13. **REML-path p search cost.** Each of the E ≈ 10–25 Brent evaluations runs a complete `fit_reml` from scratch (:4151) — no ρ or β warm starts across p — followed by an MLE φ profile. For models where a single `fit_reml` is minutes, `estimate_p(fit_mode="fit_reml")` is E× that. Verify: wall-time trace column in `search_trace` on a mid-size REML model.

---

## Bottom line

`families.py`/`links.py`/`distributions.py` are a clean, small abstraction (protocols + hasattr-optional REML derivatives) whose only structural wart is the Tweedie family's upward dependency on the profiling module. `profiling/tweedie.py` is large not because the statistics demand it but because it fuses five concerns — density evaluation, dispersion certification, model-fit orchestration (twice), search strategies (five), and reporting/CI/plot/back-compat — and wraps each in exhaustive defensive validation and branch-audit machinery. Its genuinely expensive behaviors are the per-p full GLM/REML fits (inherent), the fallback φ scan (avoidable in the common case), and the import/memory retention side effects (incidental).
