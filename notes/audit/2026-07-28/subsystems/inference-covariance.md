# Architecture report: inference-covariance subsystem

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ `f082e9b` (origin/master).
Scope: `src/superglm/inference/*.py`, `src/superglm/stats/*.py`. All paths below relative to `src/superglm/` unless absolute.

Notation: `n` = rows, `p` = total built columns, `p_a` = active columns, `p_g` = columns in one group, `m` = number of smooth terms/groups, `q` = number of penalty components, `G` = grid points (default 200), `K` = categorical levels, `k` = factor-smooth basis dim.

---

## 1. MODULE MAP

### inference/covariance.py (707 lines) — penalised (X'WX+S)^-1 utilities + lazy covariance accessors
Docstring (lines 1–6) says it was "Extracted from metrics.py to break the inference <-> metrics circular dependency". It is now **also imported by the solver/REML layer**, i.e. it sits below the solver stack in practice.

| Symbol | Lines | Role |
|---|---|---|
| `_selector_indices` | 30–48 | normalize NumPy-style selectors for accessor `__getitem__` |
| `StructuredSlopeCovarianceAccessor` | 51–115 | lazy phi-scalable view over `HessianFactor` inverse (slopes only); `selected_block`/`selected_diagonal`/`solve`/`trace`/`quadratic_form`/`__getitem__`/`__array__` |
| `StructuredCovarianceAccessor` | 118–277 | augmented (p+1)×(p+1) view over profiled Schur/sum-to-zero factors, with `intercept_shift` mapping solver→public intercept; refuses huge dominant-block materialisation (`_check_block_request`, 172–185) |
| `covariance_selected_block` / `covariance_selected_diagonal` | 280–293 | duck-typed dispatch dense-ndarray vs accessor |
| `covariance_factor_smooth_raw_level_block` | 296–342 | raw-level SZ covariance from public K-1 coordinates (fast path via `ProfiledSumToZeroBlockFactor.raw_level_inverse_block`, line 327; dense fallback with tiled contrast, 336–342) |
| `covariance_quadratic_form` | 345–350 | duck-typed c'Vc |
| `covariance_slope_view` | 353–357 | phi-scaled slope view; **dense path copies the whole (p_a×p_a) slice** (line 357) |
| `_second_diff_penalty` | 360–363 | D2'D2 fallback penalty |
| `_active_penalty_matrix` | 366–459 | build S in compact active coordinates; three code paths: `S_override` slice (382–391), `reml_penalties` walk (394–435), legacy per-group-matrix walk (437–459) |
| `_penalised_xtwx_inv` | 462–603 | dense augmented-QR path: `X_a = hstack(gm.toarray())` (534), sqrt(S) rows via per-group `eigh` (582–585), `decompose_factor(A).pseudo_inverse()` twice (589, 601) |
| `_penalised_xtwx_inv_gram` | 606–707 | gram path: ephemeral `MatrixExecutionPlan(active_gms)` + `moments` (675–676), `decompose_gram(M).pseudo_inverse()` (689), augmented (p_a+1) system (700–705) |

Callers (grep-verified):
- `_penalised_xtwx_inv_gram`: `reml/runner.py:98` (import) and `reml/runner.py:221` — **inside the REML fixed-point outer loop** (BCD path, `use_direct=False`); also `inference/_term_covariance.py:35` (via `compute_coef_covariance`, itself dead — see Suspects).
- `_penalised_xtwx_inv`: **no production callers**. Only `tests/test_reml.py:17,244,248,277` (oracle for the gram path) and `tests/test_import_compat.py:207`.
- Accessors constructed only in `model/state_ops.py:313` (`_structured_covariance_state`) and `covariance.py:150` (self).
- `_active_penalty_matrix`: `inference/metrics.py:23` (import; used at metrics.py:755, 804) and `model/state_ops.py:141/156` (`_legacy_active_state`).

### inference/metrics.py (1387 lines) — `ModelMetrics` post-fit diagnostics
| Symbol | Lines | Role |
|---|---|---|
| `_active_feature_columns` | 56–70 | map feature groups → local column indices |
| `_selected_group_state` | 73–110 | selected columns + compact GroupSlices; cross-checks `rank_info.selected_columns` (104–109) |
| `_grouped_active_design` | 113–123 | restrict fitted `DesignMatrix` to active groups |
| `_profiled_augmented_covariance` | 126–147 | assemble (p_a+1)² augmented inverse from centered gram + XtW1 |
| `_certified_data_rank` / `_certified_coefficient_rank` / `_certified_profile_rank` | 150–207 | gram decomposition + streamed n-row factor re-certification when rank is ambiguous |
| `_coefficient_estimability` | 210–223 | map active estimability to full beta coordinates |
| `_requires_wood_inference` | 226–245 | gate for computing R factor / EDF |
| `ModelMetrics.__init__` | 265–356 | 90-line guard cascade deciding `_uses_fit_rows`/`_uses_fit_design`/`_fit_geometry_matches`/`_uses_compact_fit_inference` |
| `_build_S_from_penalties` | 358–371 | **dead** (no callers anywhere) |
| `_fit_working_weights` | 373–376 | delegates to `state_ops._solver_space_working_weights` |
| `_working_eta_mu` | 387–402 | eta/mu for evaluation rows |
| scalar props (aic/bic/aicc/ebic/pearson/null model Newton) | 404–523 | null-mu Newton loop 449–461 |
| `residuals` / `_quantile_residuals` | 527–679 | family-specific quantile residuals; Tweedie CPG series loop 663–669 |
| `_active_info` | 684–837 | **the central 150-line branch cascade** returning `(X_a, W, XtWX_inv, XtWX_inv_aug, active_groups)`; 4 branches: (a) scop/compact-fit-inference → reuse `model._fit_inference_info` (716–745); (b) evaluation rows ≠ fit design → `EvaluationDesign` + certified grams (747–788); (c) fitted rank state reuse (790–795); (d) grouped legacy recompute (797–837) |
| `_active_design_moments` | 839–843 | **dead** (no callers) |
| `_active_centered_data_gram` | 845–849 | cached centered gram |
| `_active_R_factor` | 851–876 | square R with R'R = centered gram; reuses fit `R_a` when weights match |
| `_influence_edf` | 878–906 | edf = diag(F), edf1 = 2edf − diag(F²), F = (X'WX+S)⁻¹·G |
| `_hat_diag` | 913–927 | leverage via chunked quadratic-form diagonal |
| `cooks_distance` / std residuals | 934–959 | |
| `coefficient_se` / `coefficient_se_raw` | 975–1053 | per-group SE loops (near-identical twins) |
| `intercept_se` / `intercept_se_raw` | 1055–1076 | |
| `_feature_se_impl` / `feature_se` | 1078–1197 | curve/level SE by spec type; own copy of the M@Cov@M' recipe (1170–1178) |
| `_build_coef_rows` | 1208–1251 | wires `build_coef_rows` with precomputed R/EDF |
| `summary` | 1253–1387 | assembles `ModelSummary` (incl. `build_basis_detail`, `build_summary_level_display`) |

Callers: constructed in `model/explain_ops.py:69` (`model.metrics()`, with fit-cache), `plotting/diagnostics.py:480` (**inside the QQ simulation envelope loop**, one instance per simulated response), exported at package root `__init__.py:58`.

### inference/_metrics_design.py (380 lines) — memory-bounded design algebra
| Symbol | Lines | Role |
|---|---|---|
| `MappedColumnFactor` | 24–114 | rectangular factor stored on a narrow column subset (structured R_a); `__array__`/`.T` densify (101–114) |
| `_bounded_chunk_rows` | 117–125 | 16 MiB / 8192-row chunk budget |
| `_exact_runtime_design_block` | 142–203 | re-runs frozen prediction-plan transforms per chunk |
| `EvaluationDesign` | 206–285 | lazy exact design on evaluation rows; `weighted_moments` (261–285) streams raw + anchored grams |
| `centered_gram_from_moments` | 291–301 | longdouble intercept profiling |
| `iter_dense_chunks` / `quadratic_form_diagonal` / `weighted_moments` | 304–371 | chunked dispatch over DesignMatrix / EvaluationDesign / ndarray; DesignMatrix branch delegates to `solvers.centered_system.build_centered_system` (362–371) |
| `factor_from_gram` | 374–380 | eigh-based square factor |

Callers: metrics.py (13–20), coef_tables.py:10, model/state_ops.py:12 (`MappedColumnFactor`).

### inference/coef_tables.py (1019 lines) — summary coefficient rows + basis detail
| Symbol | Lines | Role |
|---|---|---|
| `build_coef_rows` | 22–932 | 900-line monolith: per-group SEs (120–142), ordered-reference intercept transform (146–173), lazy R/edf/group-edf closures (193–237), then one branch per spec type: RandomEffect/FactorSmooth (295–315), OrderedCategorical spline/step (318–473), `_SplineBase` (475–562), Categorical (564–587), SplineCategorical (589–637), Polynomial (639–656), PolynomialCategorical (658–698), CategoricalInteraction (700–716), NumericCategorical (718–734), Numeric/PolynomialInteraction (736–793), TensorInteraction (795–847), Numeric (849–868), fallback (870–886); quasi-separation heuristics (896–931) |
| `build_basis_detail` | 935–1019 | per-basis-coefficient rows for active 1-D splines |

Callers: `metrics.py:21` + `metrics._build_coef_rows` (1225), `model/report_ops.py:261` (the fast `model.summary()` path — a second, parallel orchestration of the same builder).

### inference/summary.py (1079 lines) — presentation
`_CoefRow` (18–63), `_BasisDetailRow` (66–77), `_compute_coef_stats` (80–91, z/p/CI from SE), `ModelSummary` (168–1067) with ~400-line ASCII renderer (246–641) and ~420-line HTML renderer (648–1067) that duplicate each other's layout logic branch-for-branch. Called by metrics.summary and report_ops.summary.

### inference/summary_levels.py (235 lines) — grouped/expanded level presentation
`LevelGroupLegend` (17–22), `SummaryLevelDisplay` (26–35), `validate_level_display` (38–45), `build_summary_level_display` (48–235): rewrites `_CoefRow` lists to expand grouped categorical levels, synthesizes reference rows, computes insertion points by name-matching rows against `"{term}[{level}]"` string patterns (103–111, 195–223). Purely presentational but does O(rows × features × levels) string matching.

### inference/term.py (53 lines) — compat shell
Pure re-export facade ("Wave 1 extraction", lines 1–5) over `_term_covariance`, `_term_helpers`, `_term_model_ops`, `_term_ops`, `_term_types`.

### inference/_term_types.py (186 lines) — result dataclasses
`SplineMetadata` (23–34), `SmoothCurve` (37–52), `TermInference` (55–149, with `to_dataframe`), `InteractionInference` (152–176), `_safe_exp`/`_MAX_LOG_REL` (15–20).

### inference/_term_covariance.py (258 lines) — covariance/SE for term inference
| Symbol | Lines | Role |
|---|---|---|
| `compute_coef_covariance` | 23–54 | **dead in production** (only re-exported and imported in `tests/test_import_compat.py:178`); recomputes W from scratch (39–48) then calls `_penalised_xtwx_inv_gram`; superseded by `model/state_ops.coef_covariance` (state_ops.py:321) |
| `_active_subgroup_columns` | 57–71 | duplicate of `metrics._active_feature_columns` (metrics.py:56) and inline copy in `_term_helpers._spline_se` (119–125) |
| `feature_se_from_cov` | 74–174 | SE per spec type from a covariance; RandomEffect branch uses `covariance_selected_diagonal` (138–141); spline branch delegates to `_spline_se` |
| `simultaneous_bands` | 177–251 | calls `covariance_fn()` (201), rebuilds curve SEs, Cholesky of Cov_g + 1e-12·I (230), n_sim×p_g normal draws (231), max-t simulation (235–236) |

Callers: `_term_ops.term_inference`, `_term_model_ops.relativities`, `coef_tables.build_coef_rows` (82, 334, 245), `metrics._feature_se_impl` (1106), `model/explain_ops.py:8/134` (public `model._feature_se_from_cov`, `model.simultaneous_bands`).

### inference/_term_helpers.py (358 lines) — shared term helpers
`_recenter_term` (29–81), `_spline_se` (87–136, the canonical M@Cov@M' curve-SE), `_build_spline_metadata` (139–154), `_expand_grouped_term` (157–258, PCHIP re-interpolation for grouped ordered levels), `_compute_term_edf` (261–273), `_resolve_term_lambda` (276–292), `_resolve_group_lambda` (295–318, geometric mean over `"group:suffix"` multi-penalty keys), `spline_group_enrichment` (321–345).

### inference/_term_ops.py (534 lines) — `term_inference` entry point
`term_inference` (44–529): dispatch per spec type (RandomEffect 158–195, OrderedCategorical 198–328, Spline 331–399, Categorical 402–444, Polynomial 447–486, Numeric 489–526). Covariance obtained lazily via `covariance_fn()` (147). Simultaneous bands re-derive the critical value by median back-solve from the returned CI columns (369–372). Called from `model/explain_ops.py:188` (`model.term_inference`), which is used by plotting (`model/plot_ops.py:141,295`), editor (`editor/session.py:139,147,1289`, `editor/payloads.py:170`), export (`export/rating_tables.py:159,175`), interactions plotting (`plotting/interactions.py:950`).

### inference/_term_interactions.py (79 lines)
`_interaction_inference` (17–76): reconstruct-only (no SEs) interaction results.

### inference/_term_model_ops.py (369 lines) — model-level operations
`relativities` (21–201, plot-ready DataFrames, per-feature `feature_se_from_cov` when `with_se`), `_requires_reml_term_names` (204–209), `drop1` (212–309, refits full model per feature), `refit_unpenalised` (312–362; note `lam2 = ...` Ellipsis sentinel at 349–353). Called via `model/explain_ops.py` / `model/api.py:823–868`.

### inference/_ordered_reference.py (85 lines)
`ordered_reference_intercept` (27–50) and `ordered_reference_beta_contrast` (53–82): affine shift of intercept/covariance for ordered-spline base levels. Callers: `coef_tables.py:147,154`, `model/report_ops.py:356`, `export/rating_tables.py:631`.

### inference/random_effects.py (464 lines)
`RandomEffectResult` (32–57), `vectorized_conditional_unpooled_effect` (59–151, exact Poisson/log aggregate branch 92–110, generic Newton loop 128–148), `_resolve_random_effect` (154–166), `_lambda_for_group` (169–183), `_support_from_retained_design` (186–208), `_stored_support` (211–222), `_reporting_rows` (225–288), `random_effect_result` (291–457, posterior SEs from `_fit_inference_info["XtWX_inv_aug"]` diagonal at 380–396, credibility = info/(info+lambda) at 398–401). Caller: `model/api.py:658` (`model.random_effects`).

### inference/factor_smooths.py (506 lines)
`FactorSmoothResult` (32–58), `_resolve_factor_smooth` (61–82), `_factor_penalties` (85–110), `_stored_support` (147–161), `factor_smooth_result` (181–503): fs branch computes per-level local credibility via K solves of k×k systems (314–323) and per-level curve covariances via `covariance_selected_block` (356–359); sz branch stacks K raw-level inverse blocks (401–413), computes level EDF from penalty traces (415–425) and reconciles against the coefficient-EDF sum (427–435). Caller: `model/api.py:679` (`model.factor_smooth`).

### stats/davies.py (183 lines)
`psum_chisq` (25–130): Imhof (1961) CF inversion via `scipy.integrate.quad` with a **pure-Python per-eigenvalue loop inside the integrand** (94–115). `satterthwaite` (136–183) moment-matching fallback. Callers: wood_pvalue only (+ root re-export).

### stats/wood_pvalue.py (250 lines)
Wood (2013) test: `_pivoted_test_space` (42–48, pivoted QR of X_j then r V r'), `_effective_rank` (56–72), `_identity_rank_maps` (75–87), `_fractional_rank_maps` (90–131), `_fractional_tail` (139–159), `_fallback_tail` (162–173), `_mixture_pvalue` (176–199, **unused by `wood_test_smooth`** — see Suspects), `wood_test_smooth` (202–250). Callers: `coef_tables.py:367,502,598,807` (4 call sites) + root export.

### stats/model_tests.py (473 lines)
Zero-inflation index (169–231), van den Broek score test (234–298), Cameron–Trivedi dispersion test (301–377), Vuong test (380–473) with per-family per-observation log-likelihood helpers (89–163). All O(n) vectorised; standalone (no covariance dependency).

### inference/__init__.py (75), stats/__init__.py (31)
Re-export surfaces; `inference/__init__` still exports the dead `_penalised_xtwx_inv` and `compute_coef_covariance` (lines 8, 31).

---

## 2. DATA FLOW

**Fit-time state produced elsewhere, consumed here.** The solver stores on the model: `_solver_result` (PIRLSResult with optional `rank_info` and `scop_inference`), `_linear_system_state` (StructuredLinearSystemState for `direct_solve='structured'`), `_dm` (DesignMatrix, may be released with `retain_fit_state=False`), `_fit_weights/_fit_offset/_fit_X_ref/...`, `_reml_lambdas`, `_reml_penalties`.

**Canonical covariance pipeline (post-fit).** Three cached properties on `SuperGLM` (model/api.py:630–642) delegate to `model/state_ops.py`:
- `coef_covariance` (state_ops:321) → `(Cov_slopes, active_groups)`; phi-scaled; branch on scop → structured accessor → rank_info → legacy.
- `fit_active_info` (state_ops:350) → `(X_a, W, XtWX_inv, XtWX_inv_aug, active_groups)`.
- `fit_inference_info` (state_ops:411) → dict `{W, XtWX_inv, XtWX_inv_aug (p_a+1)², active_groups, R_a (p_a×p_a or MappedColumnFactor), edf (p_a,), edf1 (p_a,), group_edf_map, coefficient_estimable (p,)}`.

Every branch recomputes working weights `W = w·(dμ/dη)²/V` at n cost via `_solver_space_working_weights` (state_ops:33–46), which itself does a full `dm.matvec(beta)` (O(n·p) equivalent through group kernels).

**Consumers.**
1. `model.summary()` → `model/report_ops.py:255–284` reads `_fit_inference_info` and calls `build_coef_rows` directly (X_a passed as an *empty array*, report_ops.py:265).
2. `model.metrics(X,y,...)` → `ModelMetrics`. `_active_info` (metrics.py:684) either reuses `_fit_inference_info` (same geometry) or *rebuilds everything* on evaluation rows: `EvaluationDesign` streams exact design chunks (≤8192 rows × p_a) through `weighted_moments` producing raw gram (p_a²), XtW1 (p_a), centered gram (p_a²); three `decompose_gram` + possible streamed re-certification passes over all n rows; then `_profiled_augmented_covariance` (p_a+1)². Downstream cached properties: `_hat_diag` (chunked X(X'WX+S)⁻¹X' diagonal, O(n·p_a²)), `_influence_edf` (F = V·G, p_a³), `_active_R_factor` (eigh p_a³), residual/IC scalars (O(n)).
3. `model.term_inference(name)` → `_term_ops.term_inference` with `covariance_fn = lambda: model._coef_covariance` (explain_ops.py:194). Reconstructs the term curve from beta, then `feature_se_from_cov`/`_spline_se`: build M (G×p_g) via `spec.transform`, slice Cov_g (p_g×p_g) out of Cov_active, curve SE = rowsum((M·Cov_g)∘M). Simultaneous bands add a Cholesky (p_g³) + (n_sim×p_g) sampling + (n_sim×G) matmul.
4. `model.summary()` smooth p-values: `build_coef_rows` pulls `R_a`, `edf1`, augmented block V_b_j per smooth group and calls `wood_test_smooth`: pivoted QR of `X_j = R_a[:, sl]` (p_a×p_g → p_g² work per term), eigh of p_g×p_g test covariance, quadratic forms, Imhof integration (scipy quad, ~sub-ms per term).
5. `model.random_effects(name)` / `model.factor_smooth(name)`: read `support_totals` (per-level counts/weights/information from structured solver state or recomputed from retained design), take diagonal/blocks of `_fit_inference_info["XtWX_inv_aug"]`, assemble pandas tables (K rows) and curve frames (K·G rows).

**Matrices materialised (worst case, dense legacy/evaluation path):** X_a chunks (8192×p_a), raw gram + centered gram + coefficient inverse + profile inverse + augmented inverse ≈ 5·p_a² float64, R_a (p_a²), F (p_a² transient), per-term M (G×p_g), Cov_g (p_g²). Structured path replaces all p² dense objects with factor-backed accessors; only requested blocks materialise.

**REML loop (out-of-subsystem consumer):** `reml/runner.py:221` calls `_penalised_xtwx_inv_gram(beta, W, ..., S_override=S_rro)` once per REML outer iteration on the BCD path — full p×p `XtWX` via a fresh `MatrixExecutionPlan`, plus p³ `pseudo_inverse`, per iteration.

---

## 3. STATE OBJECTS

| Object | File:lines | Fields | Lifecycle | Overlap |
|---|---|---|---|---|
| `StructuredSlopeCovarianceAccessor` | covariance.py:51 | factor, scale, shape | created per `coef_covariance`/`fit_*_info` call chain; immutable, `scaled()` clones | thin wrapper duplicating `HessianFactor` API with scale |
| `StructuredCovarianceAccessor` | covariance.py:118 | factor, augmented_factor, intercept_shift (p,), scale | same | overlaps `_profiled_augmented_covariance` / `_rank_augmented_covariance` in role (augmented covariance), different representation |
| `MappedColumnFactor` | _metrics_design.py:24 | local_factor (r×s), column_indices, width | stored in `_fit_inference_info["R_a"]` (structured branch) | R_a also exists as dense p_a×p_a on other branches — two representations of one concept |
| `EvaluationDesign` | _metrics_design.py:206 | model ref, EagerFrame, selected_columns | per-ModelMetrics evaluation | third design representation next to `DesignMatrix` and raw ndarray (`MetricsDesign` union, line 288) |
| `ModelMetrics` | metrics.py:248 | ~12 guard booleans, y/weights/offset/mu copies, ~15 cached_properties, plus ad-hoc `self.__dict__[...]` cache stuffing (`_coefficient_estimable` written at 728, 735, 739, 765, 794, 815; `_active_centered_data_gram` at 763, 813) | per metrics() call; cached on model for fit rows | duplicates `_fit_inference_info` content when geometry matches; the `__dict__` side-channel bypasses cached_property semantics |
| `TermInference` / `SmoothCurve` / `SplineMetadata` / `InteractionInference` | _term_types.py:23–176 | frozen result values | returned to user | none |
| `_CoefRow` | summary.py:18 | 30+ optional fields spanning 6 row kinds (parametric, spline, structured, reference, level, interaction) | built by coef_tables, mutated post-hoc for estimable/quasi_separated flags (coef_tables.py:892–931) | a discriminated union flattened into one mutable dataclass |
| `_BasisDetailRow` | summary.py:66 | 8 floats | | |
| `SummaryLevelDisplay` / `LevelGroupLegend` | summary_levels.py:16–35 | rewritten row tuples | per summary() | second row list parallel to `_coef_rows` inside `ModelSummary` |
| `ModelSummary` | summary.py:168 | data dict, info dict, coef_rows, display_rows, basis_detail | returned to user | `_data` dict duplicates scalars already in `_info` |
| `RandomEffectResult` | random_effects.py:32 | scalars + DataFrame + diagnostics dict | returned | shares boundary-constant/collapse logic with `FactorSmoothResult` |
| `FactorSmoothResult` | factor_smooths.py:32 | scalars, per-component dicts, table + curves DataFrames | returned | duplicated `_LAMBDA_LOWER/UPPER_BOUND` constants (random_effects.py:28–29 vs factor_smooths.py:28–29) and duplicated `_stored_support`/`_support_from_retained_design` helpers (random_effects.py:186–222 vs factor_smooths.py:118–161) |
| result dataclasses in stats/model_tests.py:23–61 | 4 small frozen results | returned | none |

Cross-cutting overlap: `_fit_inference_info` (dict), `fit_active_info` (tuple), `coef_covariance` (tuple), and `ModelMetrics._active_info` (tuple) are four containers for essentially the same inference state, with three different intercept conventions (slopes-only, augmented-index-shifted, accessor).

---

## 4. COMPLEXITY TABLE

| Routine | Time | Memory | Notes |
|---|---|---|---|
| `_penalised_xtwx_inv` (covariance.py:462) | O(n·p_a²) QR ×2 (plain + augmented) + m·p_g³ eigh | dense (n+p_a)×p_a stacked A, twice | dense `gm.toarray()` per group (507); production-dead |
| `_penalised_xtwx_inv_gram` (covariance.py:606) | O(n·Σp_g²) gram + O(p_a³) pinv ×2 | p_a² ×4 (XtWX, S, M, M_aug, 2 inverses) | called **per REML outer iteration** (runner.py:221); fresh `MatrixExecutionPlan` each call (675) |
| `_active_penalty_matrix` (366) | O(q·p_g³) (R_inv'ΩR_inv per component) | p_a² | rebuilt each call; no caching between summary/metrics/REML uses |
| `StructuredCovarianceAccessor.selected_block` (187) | s solves against augmented factor | (p+1)×s contrasts (163) | contrast build is a dense Python loop over s columns (164–169) |
| `covariance_factor_smooth_raw_level_block` fallback (336–342) | O(((K−1)k)²·k) | ((K−1)k)² public block | final-level dense fallback materialises the whole SZ block |
| `ModelMetrics._active_info` branch (b)/(d) (747–837) | O(n·p_a²/chunk streaming) for moments; up to 3 additional full-data streamed factor passes on rank ambiguity (`_certified_*`, 150–207); 3× O(p_a³) decompositions | ~5 p_a² arrays | re-runs feature transforms per chunk on evaluation rows (`_exact_runtime_design_block`) |
| `weighted_moments` (EvaluationDesign 261 / ndarray 339) | O(n·p_a²) | 2 p_a² accumulators + anchored copies | anchored-gram accumulation duplicated verbatim in both branches |
| `_hat_diag` (913) | O(n·p_a²) chunked | n vector | fine |
| `_influence_edf` (878) | O(p_a³) (F product + `diagonal_of_square`) | p_a² F | duplicated in coef_tables:214–225 and state_ops:545–548 |
| `_active_R_factor` (851) | O(p_a³) eigh | p_a² | third eigh of the same centered gram in some paths |
| `_quantile_residuals` Tweedie branch (633–671) | **O(n_pos · k_max)** with k_max ≈ max(λ_i)+6√λ (663) | n per k | k_max driven by the row with the largest λ; heavy-exposure rows inflate cost for all rows |
| `build_coef_rows` (22) | per smooth group: QR p_a×p_g + eigh p_g³ (wood) + `_curve_se_range` G·p_g² ; per categorical group: bincounts O(n) (110–115) | M (G×p_g), V_b_j (p_g²) | `_curve_se_range` recomputes `covariance_slope_view` per group (243), which on the dense path copies the full (p_a×p_a) slice per spline term (covariance.py:357) |
| `build_summary_level_display` (48) | O(rows·features·levels) string matching | rows copies | presentation only |
| `term_inference` spline + SE (331–399) | G·p_g² + reconstruct | M, Cov_g | |
| `simultaneous_bands` (177) | p_g³ chol + n_sim·p_g·G matmul (n_sim=10k default) | n_sim×G f_sim | one extra `covariance_fn()` call (201); Cov_g block extraction duplicated with the caller's SE path |
| `drop1` (212) | (#features) × full `fit()` refit | full models | inherent to design |
| `random_effect_result` (291) | O(n) bincounts + K diag lookups; generic-family Newton O(iter·n) | K-row table | |
| `factor_smooth_result` fs (309) | K × k³ solves + K × (G·k²) curves | K×k×k stacks | fine |
| `factor_smooth_result` sz (394) | K × SZ block extraction; final level costs a ((K−1)k)² dense block on non-structured covariances | K×k×k raw blocks | |
| `wood_test_smooth` (202) | QR(p_a×p_g) + eigh(p_g) + quad integration | p_g² | integrand is pure Python over eigenvalues (davies.py:100–105); fine at p_g ≤ 20, called once per smooth per summary |
| `psum_chisq` (25) | O(quad_evals · r) Python | — | |
| model_tests functions | O(n) | n | |

---

## 5. SUSPECTS

1. **Dead production code kept as exports.** `_penalised_xtwx_inv` (covariance.py:462–603) has zero production callers — only tests use it as an oracle (tests/test_reml.py:244+, test_import_compat.py:207) — yet it is exported from `inference/__init__.py:8,43`. Its docstring (473) claims "Shared by `model._coef_covariance` and `ModelMetrics._active_info`", which is false on current master (both use state_ops / certified-gram paths). Likewise `compute_coef_covariance` (_term_covariance.py:23–54) is only reachable via import-compat tests; `ModelMetrics._build_S_from_penalties` (metrics.py:358–371) and `ModelMetrics._active_design_moments` (metrics.py:839–849) have no callers at all. Verify with a repo-wide grep + coverage run; these are ~200 lines of stale algebra whose doc comments actively mislead.

2. **Inference module algebra inside the REML optimizer loop.** `reml/runner.py:221` calls `inference.covariance._penalised_xtwx_inv_gram` once per REML fixed-point iteration (BCD path): fresh `MatrixExecutionPlan` (covariance.py:675), full p×p gram, full `decompose_gram(...).pseudo_inverse()` (689), plus `build_penalty_matrix` called up to three times per iteration (runner.py:141/213/230). This is post-fit inference code doing solver-loop work — O(n·Σp_g² + p³) per outer iteration — and inverts the intended layering (covariance.py's own header says it exists to break an inference↔metrics cycle, lines 1–6, not to serve the solver). Verify: profile a BCD REML fit with many smooth terms; check whether XtWX can be cached/updated instead of rebuilt (the direct path already caches `centered_XtWX`, runner.py:130).

3. **Four parallel builders of the same augmented covariance.** (a) `_penalised_xtwx_inv_gram` M_aug (covariance.py:700–705); (b) `metrics._profiled_augmented_covariance` (metrics.py:126–147); (c) `state_ops._rank_augmented_covariance` (state_ops.py:280–293); (d) `state_ops._legacy_active_state` inline assembly (state_ops.py:225–233). (b)–(d) are byte-similar bordered-inverse formulas; (a) inverts the (p_a+1) system directly instead of profiling. Same for the "identify active groups + re-index GroupSlice" loop, which appears in covariance.py:495–525, covariance.py:637–664, metrics.py:73–110, and state_ops.py:92–123 — four near-identical copies. Verify equivalence and consolidate later; risk today is drift (e.g. the direct-inverse variant (a) has different null-space behaviour from the profiled ones under rank deficiency).

4. **`ModelMetrics._active_info` duplicates `state_ops.fit_inference_info` orchestration.** Both implement the same four-way branch cascade (scop → structured → rank_info → legacy): metrics.py:684–837 vs state_ops.py:411–579, and metrics branch (a) then re-reads `model._fit_inference_info` anyway (metrics.py:717–719). EDF (F = V·G, diag, diag-of-square) is implemented three times: metrics.py:900–906, coef_tables.py:218–225, state_ops.py:545–548 (+ `_rank_edf1` state_ops.py:256–279). Any fix to EDF conventions must land in 3–4 places. Verify: unit-diff `edf/edf1` from `model.summary()` vs `model.metrics().summary()` on all solver backends.

5. **Two competing `summary()` pipelines.** `model.summary()` (model/report_ops.py:255–284+) and `model.metrics().summary()` (metrics.py:1253–1387) each independently orchestrate `build_coef_rows` + SE dicts + `ModelSummary`. report_ops passes `X_a=np.empty((0,0))` (report_ops.py:265) and relies entirely on precomputed values; metrics recomputes SE dicts through its own `coefficient_se`/`coefficient_se_raw` twins (metrics.py:975–1053) which repeat the identical per-group loop of coef_tables.py:120–142 and report_ops.py:286–309 — **four** copies of "augmented-diagonal → sqrt → NaN-mask" code. Also check whether report_ops passes `basis_detail` (metrics path does at 1360–1371); if not, HTML disclosure content differs between the two public summaries. Verify by string-diffing both summaries on one model.

6. **Wood-test call recipe duplicated four times inside `build_coef_rows`.** coef_tables.py:367–392 (ordered spline), 502–515 (`_SplineBase`), 598–609 (SplineCategorical), 807–818 (TensorInteraction) each repeat: slice augmented block → `_get_R_factor` → `_get_influence_edf` → `edf1_j` → `X_j = R_a[:, sl]` → `res_df` → try/except `wood_test_smooth`. Two of them catch bare `Exception` (513, 609, 818) vs `np.linalg.LinAlgError` (391, 497), silently converting any bug into "no p-value". The parametric group-Wald `b'V⁻¹b` recipe is likewise triplicated (491–497, 669–673, 762–766). Verify: introduce a deliberate error in wood_test_smooth and observe which summary rows silently show NaN.

7. **`_mixture_pvalue` in wood_pvalue.py:176–199 is orphaned.** `wood_test_smooth` never calls it (integer-rank case jumps straight to `p_val = 2.0` sentinel → `_fallback_tail`, lines 244–248). That means for integer `edf1` the Davies/Imhof machinery is bypassed entirely and a plain chi²/F fallback is always used, while the docstring (lines 8–14) advertises "evaluate the tail under a weighted chi-square mixture (known scale)". Either `_mixture_pvalue` is dead code or the integer-rank branch forgot to use it — either way, doc–code mismatch with statistical consequences. Verify against mgcv's `testStat` behaviour for integer-rank smooths.

8. **Wasted covariance block extraction in `feature_se_from_cov`.** _term_covariance.py:143 computes `Cov_g = Cov_active[np.ix_(indices, indices)]` before the spline branch (145–154), which ignores it and re-extracts inside `_spline_se` (line 113). On a `StructuredSlopeCovarianceAccessor` each extraction is a factored solve of width |indices| — done twice per spline term. Similarly `simultaneous_bands` re-extracts the same block and re-computes the same curve SEs already produced by the caller (_term_covariance.py:210–218 vs _term_ops.py:342–353), and `term_inference` then reverse-engineers the simulated critical value from the CI columns by median back-solve (_term_ops.py:369–372) because `simultaneous_bands` returns only a DataFrame. Verify with a call-count spy on `selected_block`.

9. **`covariance_slope_view` dense path copies the whole matrix per term.** covariance.py:357 does `float(scale) * np.asarray(covariance)[1:, 1:]` — a fresh (p_a×p_a) allocation. `build_coef_rows._curve_se_range` (coef_tables.py:239–247) calls it once per active spline group, so a model with m splines allocates m full covariance copies during a single summary. Same pattern at coef_tables.py:333 and metrics.py:1122. Verify with a memory profiler on summary() for p_a ≈ 2000, m ≈ 20.

10. **Three copies of the feature→local-column mapping helper.** `_active_subgroup_columns` (_term_covariance.py:57–71), `_active_feature_columns` (metrics.py:56–70), and the inline loop in `_spline_se` (_term_helpers.py:119–125) implement the same mapping with the same subtle assumption (`feature_groups[0].start` / `groups[0].start` as base offset). A select=True spline whose *first* subgroup is inactive exercises the assumption differently in each copy. Verify with a select=True model where the linear subgroup is zeroed.

11. **`ModelMetrics.__init__` guard cascade + `__dict__` side-channel caching.** Four overlapping booleans (`_uses_fit_rows`, `_uses_fit_design`, `_fit_geometry_matches`, `_uses_compact_fit_inference`, metrics.py:284–343) gate later branches, and `_active_info` writes `_coefficient_estimable`/`_active_centered_data_gram` into `self.__dict__` from six different sites (728, 735, 739–744, 763–765, 794, 813–820) that `_current_coefficient_estimable` (963–973) then reads back with a `_ = self._active_info` side-effect trigger (966). This is control-flow-as-state; correctness depends on evaluation order of cached properties. Verify: call `metrics.coefficient_se` before/after `metrics.summary()` on a non-fit-geometry evaluation and diff the NaN masks.

12. **Tweedie quantile-residual truncation scales with the max-λ row.** metrics.py:663–669: `k_max` is computed from the max weight-adjusted Poisson rate over all rows and the k-loop over Poisson-Gamma components evaluates `gamma.cdf` on *all* positive rows for each k. With heterogeneous exposure (λ up to 1e3+), this is O(n·1000+) gamma CDF evaluations for a plot residual. Verify by timing `residuals("quantile")` on a Tweedie fit with wide exposure range.

13. **`ModelMetrics` instantiated per simulation inside the QQ envelope loop.** plotting/diagnostics.py:474–486 builds a fresh `ModelMetrics` per simulated response (default sim count × constructor guard machinery incl. O(n) array comparisons and offset allocation at metrics.py:305–343). Only `_quantile_residuals` is needed; the guard cascade, fit-data matching, and `_null_mu` machinery are dead weight per iteration. Verify with cProfile on `plot_diagnostics` QQ panel.

14. **Duplicated support/boundary machinery between random_effects.py and factor_smooths.py.** `_LAMBDA_LOWER/UPPER_BOUND` constants (random_effects.py:28–29, factor_smooths.py:28–29), `_support_from_retained_design` (186–208 vs 118–144), `_stored_support` (211–222 vs 147–161), collapse warnings (312–316 vs 217–222), variance-component math (307 vs 202–205). These are the same protocol specialised twice; drift risk when a third structured term type arrives. Verify by diffing the two files.

15. **`refit_unpenalised` Ellipsis sentinel.** _term_model_ops.py:349–353 sets `lam2 = ...` (the `Ellipsis` object) as a "keep smoothing" marker passed into `model._clone_without_features(..., lambda2=lam2)`. Untyped sentinel crossing a public-ish model-cloning API; if the clone path ever validates `lambda2` numerically this silently breaks. Verify `_clone_without_features` handling and whether a `None` sentinel is the intended convention elsewhere.

16. **`weighted_moments` anchored-gram code duplicated verbatim.** _metrics_design.py:269–285 (EvaluationDesign) vs 345–360 (ndarray branch): identical anchor/shift/accumulate logic maintained twice in one module; the DesignMatrix branch instead delegates to `solvers.centered_system.build_centered_system` (362–371) — three numerically *different* centering schemes (anchored longdouble vs solver-centered) for the same quantity. Verify centered-gram agreement between branches at float64 tolerance on ill-conditioned columns.

17. **Presentation duplication in `ModelSummary`.** ASCII (summary.py:246–641) and HTML (648–1067) renderers re-implement every row-kind branch, EDF-breakdown computation (254–267 vs 681–689), header tables, profile rows, QS footnotes. ~800 lines where each display change must be made twice. Low risk, high maintenance cost.

18. **Two conditional/frequentist conventions live only in comments.** SEs are consistently documented as Bayesian/conditional (`coefficient_se` note, metrics.py:986–989: "conditional-on-the-selected-model ... same convention as glmnet / mgcv"); there is no Ve (frequentist sandwich) or selection-corrected covariance anywhere, and no flag distinguishing Vb vs Ve in APIs — mgcv users may expect `unconditional=TRUE`-style options. Not a bug; an architecture gap to note for the target-architecture phase.
