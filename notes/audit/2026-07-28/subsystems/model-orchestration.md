# Subsystem report: model-orchestration

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ origin/master (f082e9b).
Scope: `src/superglm/model/*.py` (23 modules, ~11.5k LOC). All paths below are relative to `src/superglm/` unless absolute.

Notation: n = rows, p = total built design columns, p_a = active columns, m = number of smooth terms, q = number of penalty components (REML lambdas), G = number of coefficient groups (G >= m), k_g = columns of group g, C = number of raw input columns, L = n_lambda path length, I = IRLS/PIRLS iterations, R = REML outer iterations.

---

## 1. MODULE MAP

### 1.1 `model/api.py` (1734 lines) — public facade
Single class `SuperGLM` (api.py:52). Almost every method is a one-line delegation; the file holds docstrings and parameter plumbing only.

- `__init__` (api.py:60-204) -> `base.init_model` (base.py:630).
- `clone_unfitted` (api.py:216-271) -> `self._config.materialize(SuperGLM)` (fit_state.py:177); subclass path reconstructs from `ModelConfig.constructor_kwargs` (fit_state.py:140) with signature introspection.
- Property setters `family/link/penalty/lambda2` (api.py:283-346) mutate `_family_config`/etc. AND rebuild `self._config` via `ModelConfig.with_value`, bumping `_config_revision`.
- `fit` (api.py:405-484) -> `fit_ops.fit` (fit_ops.py:718). Resolves tol/max_iter/convergence kwargs against constructor defaults, re-emits the experimental-convergence warning (api.py:462-472, duplicating base.py:682-692).
- `fit_path` (api.py:486-510) -> `fit_ops.fit_path` (fit_ops.py:890).
- `fit_reml` (api.py:512-609) -> `fit_ops.fit_reml` (fit_ops.py:1038).
- `result` property (api.py:613-621); `_solver_pirls_result` (api.py:623-627).
- 4 `cached_property` inference caches (api.py:629-643): `_coef_covariance`, `_fit_active_info`, `_fit_inference_info`, `_group_edf` -> `state_ops.*`.
- Reporting/inference delegations: `diagnostics`/`summary`/`reconstruct_feature`/`knot_summary` -> `report_ops`; `metrics`/`drop1`/`relativities`/`term_inference`/`term_importance`/... -> `explain_ops`; `plot`/`plot_data` -> `plot_ops`; `estimate_p`/`estimate_theta` -> `profile_ops`; `monotonize`/`apply_shape_postfit` -> `monotone_ops`/`shape_ops`; `training_telemetry`/`reml_diagnostics` -> `telemetry_ops`; `design_summary` -> `design_summary.build_design_summary`.
- **REML adapter block** (api.py:1527-1734): `_compute_dW_deta`, `_reml_w_correction`, `_reml_laml_objective`, `_reml_direct_gradient`, `_reml_direct_hessian`, `_optimize_direct_reml`, `_optimize_discrete_reml_cached_w`, `_optimize_efs_reml`, `_run_reml_once` — each delegates to `fit_ops.model_*` which are themselves re-exports of `reml_ops.model_*`. Comment says "used by reml_optimizer" but **no production code calls any of these methods** (grep: only tests `test_reml_fd.py`, `test_hessian_ift.py`, `test_scop_efs.py` use `_reml_w_correction`/`_reml_laml_objective`; `_run_reml_once` and `_optimize_discrete_reml_cached_w` have zero callers anywhere including tests, except `tests/test_import_compat.py:147,151` which imports the underlying free functions).

### 1.2 `model/base.py` (1030 lines) — constructor + prediction engine + design-matrix bridge
- `init_model` (base.py:630-787): body of `__init__`; installs ~45 `_`-prefixed instance slots (base.py:695-726), resolves penalty shorthand (`resolve_penalty`, base.py:514-548), classifies tuple vs explicit interactions (base.py:732-771), validates namespaces (`validate_term_name_namespace`, `validate_factor_smooth_configuration` base.py:584-627), captures `ModelConfig` (base.py:787).
- Prediction stack: `_prediction_plan` (base.py:252-262, lazy) / `freeze_prediction_plan` (base.py:265-272) -> `_build_prediction_plan` (base.py:203-235) -> `_compile_fast_prediction_state` (base.py:165-200) which reads `model._fit_X_ref` to compile per-feature discretizer metadata (`_fit_discretizer_metadata` base.py:73). Scoring: `_predict_eta` (base.py:389-444) walks plan terms, adds intercept, offset, `stabilize_eta`; exact scorers `_score_prediction_term_exact` (base.py:360) and fast-discrete scorers `_score_feature_fast_discrete` (base.py:275) / `_score_interaction_fast_discrete` (base.py:289, einsum over observed support pairs). Public wrappers `predict_eta_exact/predict_exact/predict_eta_fast_discrete/predict_fast_discrete` (base.py:447-511). Called from api.py:1281-1357, `runtime_canonicalize._live_public_runtime_state`, sklearn wrappers, diagnostics.
- Design bridge: `model_build_design_matrix` (base.py:919-955) -> `dm_builder.build_design_matrix`; sets `_dm`, `_groups`, `_distribution`, `_link`, clears `_pending_interactions`. `rebuild_dm_with_lambdas` (base.py:1009-1019) -> `dm_builder.rebuild_design_matrix_with_lambdas` (used by discrete REML rebuilds, reml_ops.py:186-189, 274-276, 324-326, reml_finalize.py:366).
- Selection-penalty policy: `compute_lambda_max` (base.py:958-971, O(n·p) rmatvec + per-group norms), `resolve_selection_penalty_for_fit` (base.py:974), `validate/resolve_selection_penalty_for_reml` (base.py:987-1001) — the enforcement point of the "fit_reml has no selection penalty" contract.
- `clone_without_features` (base.py:790-875): rebuilds a fresh SuperGLM with subset features; consumes `fitted_penalty`/`fitted_lambda2`; used by drop1/refit_unpenalised.
- **Dead code**: module-level `predict` (base.py:1022-1031) duplicates `predict_exact` delegation; no callers found in src or tests.

### 1.3 `model/fit_ops.py` (1434 lines) — orchestration core for fit / fit_path / fit_reml
- `PathResult` (fit_ops.py:98-168): frozen dataclass; `__post_init__` re-materialises every array via `_immutable_path_array` (fit_ops.py:90-95) — `tobytes()` + `frombuffer` = **two copies of each (L, p) array** for immutability.
- Guards: `_reject_random_effect_selection_fit` (fit_ops.py:171), `_reject_structured_fit_constraints` (fit_ops.py:195).
- Statistics: `_compute_null_mu` (fit_ops.py:240-275, 25-iteration Newton for offset-aware null intercept, O(25n)); `_compute_fit_stats` (fit_ops.py:278-326, Tweedie special-cased).
- Cache lifecycle: `_clear_profile_results` (374), `_clear_fit_inference_caches` (380-402, pops 4 cached_properties + 16 slots), `_clear_reml_state` (405-410), `_store_fit_arrays` (413-418, copies weights/offset), `_prime_fit_caches` (544-607: recomputes eta/mu/null_mu, captures `FitDataGuard`/`FitGeometryGuard`, stores X/y/w/offset refs), `_maybe_release_fit_state` (610-644: distills `_fit_inference_info` into `_group_edf` + `_coef_covariance` then nulls `_dm` and row-scale buffers when `retain_fit_state=False`).
- Solver policy: `_solve_coefficients` (fit_ops.py:659-715): constraints/SCOP/lambda1==0 -> `fit_irls_direct`; else `fit_pirls` (PIRLS->BCD).
- `fit` (718-759): validate -> `FitWorkspace.start` -> `_fit_in_workspace` (762-887) -> `capture_fit_state` -> `_install_fit_state`.
- `fit_path` (890-934) -> `_fit_path_in_workspace` (937-1035) -> `path_ops.run_lambda_path`; publishes final lambda's fit as model state; `canonicalize_intercept_path` (1018-1023) maps solver intercept path to public.
- `fit_reml` (1038-1093) -> `_fit_reml_in_workspace` (1116-1434). Sub-steps inside the workspace body:
  1. `resolve_selection_penalty_for_reml` (1146).
  2. `_maybe_estimate_nb_theta` (1155; auto-theta profile *before* design build).
  3. `model_build_design_matrix` (1167) — timed into `_profile`.
  4. `collect_reml_groups` (1173, reml_setup.py:19); `constraint_engine_flags` (1174).
  5. `_make_reml_debug_recorder` (1175, fit_ops.py:421-466).
  6. No-eligible-groups fallback (1188-1236): inline replica of `_fit_in_workspace`'s post-solve block.
  7. `build_penalty_context` (1241-1245, reml/penalty_algebra: per-penalty eigenstructure, computed once).
  8. `initialize_component_lambdas` + `inject_fixed_scop_lambdas` (1249-1250).
  9. QP passthrough staging: `strip_qp_constraints` (1263, reml_setup.py:154).
  10. Route: `run_fixed_monotone_reml` (1274) | `run_scop_efs_reml` (1315) | `optimize_reml_best` (1364, reml_execute.py:259) then `finalize_reml_fit` (1387, reml_finalize.py:317).
  11. `_prime_fit_caches` (1408), `_canonicalize_fitted_model` (1418, fit_ops.py:524-541 -> runtime_canonicalize), `_maybe_release_fit_state` (1427).
  12. `finally: restore_qp_constraints` (1431-1434).
- `_record_reml_terminal_best_effort` (1096-1113): post-install trace emit, exception-swallowing by design.
- Re-export block (fit_ops.py:43-53, 73-87): imports all 9 `model_*` REML adapters from `reml_ops` and re-exports them via `__all__` purely so `api.py` can call `fit_ops.model_*`.

### 1.4 `model/fit_state.py` (705 lines) — config templates + atomic publication
- `FrozenMapping` (13-33): pickle-safe read-only mapping for `FitState.projections`.
- Accessors: `configured_family/link/penalty/lambda2` (36-53) return `_*_config` slots; `fitted_lambda2` (56-64) and `fitted_penalty` (67-73) prefer `_fit_state.resolved_*` then `_reml_lambdas`/`_resolved_penalty` then config.
- `ModelConfig` (76-242): frozen constructor intent. `capture` (101-134) deepcopies family/link/penalty/spec templates; `with_value` (136); `constructor_kwargs` (140-175); `materialize` (177-242) builds a bare model dict of ~50 slots — **this dict must stay in sync with `init_model` by hand** (base.py:695-726 vs fit_state.py:180-240).
- `ModelConfigPublication` (245-277): constructor-state identities re-installed at publication.
- `FitState` (280-291): frozen authoritative fit identity (revision, selection_penalty, distribution, projections FrozenMapping, retained flag, resolved penalty/lambda2).
- `FitCandidate` (294-299): prepared replacement `__dict__`.
- `FittedStateRevision` (302-381): second workspace type for **post-fit** coefficient revisions (shape repair, editor). `start` (312-365) shallow-copies the model dict and makes mutable copies of `_result`/`_solver_result` (via `_mutable_copy`), re-aliases `_reml_result.pirls_result`; `commit` (367-381) -> `_capture_model_state` -> `_install_fit_state`.
- `invalidate_revised_coefficient_mode` (384-418): nulls state_id/log_det_H/objective etc. after arbitrary coefficient revision.
- `_FIT_PROJECTION_NAMES` (421-459): 38 slots snapshotted into `FitState.projections`.
- `_validate_workspace_result` (462-524): pre-publication invariants — beta finite, `beta.shape == (dm.p,)`, `sum(group.size) == p`, **`np.array_equal(result.beta, solver_result.beta)`** (479), scalar equality between public and solver results, canonical intercept relation `result.intercept == solver.intercept + intercept_shift` within 1e-13 (507-517).
- `_freeze_candidate_arrays` (532-563): sets `writeable=False` on result betas, covariance, inference dict arrays, `_fit_weights/_fit_offset/_fit_mu/_fit_null_mu`.
- `_publish_workspace_extension_state` (566-614): second alias-preserving deepcopy of subclass state at publication (mirror of `fit_workspace._copy_subclass_state`).
- `_capture_model_state` (617-675) / `capture_fit_state` (678-700, also drops the raw-spline Tabmat plan cache via `release_raw_spline_tabmat_plan`) / `_install_fit_state` (703-705: **`model.__dict__ = candidate.prepared_model_dict`** — the single allocation-free commit point).

### 1.5 `model/fit_workspace.py` (101 lines) — attempt isolation
- `FitWorkspace.start` (79-101): `config.materialize(type(public_model))` -> fresh work model with zero prior fitted buffers; copies `_ATTEMPT_RUNTIME_OPTIONS` (`_max_analytical_per_w`, `_select_snap`, `_suppress_reporting_support`, fit_workspace.py:8-12) and subclass state.
- `_copy_subclass_state` (16-66): on first fit, diff of `public_model.__dict__` vs materialized dict identifies subclass constructor attrs; names memoised under `_fit_workspace_subclass_state_names`; alias-preserving deepcopy.
- Callers: `fit_ops.fit/fit_path/fit_reml` (fit_ops.py:737, 913, 1065), `profile_ops.estimate_p/estimate_theta` (profile_ops.py:66, 91, 311, 334).

### 1.6 `model/state_ops.py` (585 lines) — post-fit covariance/EDF computation
- `_solver_space_working_weights` (33-46): W = w·(dmu/deta)²/V on training rows; O(np) matvec + O(n).
- `_public_intercept_shift` (49-67) / `_public_augmented_covariance` (70-89): map solver-space augmented covariance into public coordinates using the canonical column means.
- `_grouped_active_state` (92-122): re-slices `_dm.group_matrices` into an active-only `DesignMatrix`; `_rank_active_state` (125-136) same in rank-info order.
- `_legacy_active_state` (139-238): full rebuild of centered gram + penalty curvature + rank decompositions + pseudo-inverses for old fits lacking `rank_info` — the fallback covariance engine.
- Three public entry points, each with **four parallel branches** (scop_inference / StructuredLinearSystemState / rank_info / legacy):
  - `coef_covariance` (321-347) — cached as `SuperGLM._coef_covariance`.
  - `fit_active_info` (350-389) — cached; consumed by `inference/metrics.py:722`.
  - `fit_inference_info` (411-578) — cached; returns dict W, XtWX_inv, XtWX_inv_aug, active_groups, R_a, edf, edf1, group_edf_map, coefficient_estimable; docstring (417-421) documents O(n + p³) instead of O(n·p²).
- `group_edf` (581-585) -> `fit_inference_info["group_edf_map"]`.

### 1.7 `model/reml_setup.py` (173 lines) — pre-REML classification
- `collect_reml_groups` (19-48): REML-eligible groups = penalized RandomEffect/FactorSmooth matrices, or SSP-like matrices with `omega is not None`. Called from fit_ops.py:1173 and `profiling/nb.py:439`.
- `initialize_component_lambdas` (51-72): seeds lambda dict per penalty component (fixed vs estimated split).
- SCOP lambda policy helpers (75-136): `scop_fixed_lambda_value`, `inject_fixed_scop_lambdas`, `promote_estimated_scop_lambdas` (used by reml_execute.py:199).
- `constraint_engine_flags` (139-151); `strip_qp_constraints` (154-163) / `restore_qp_constraints` (166-173) mutate `GroupSlice.monotone_engine/constraints` in place.

### 1.8 `model/reml_ops.py` (333 lines) — model-to-free-function adapters
Every function unpacks `model._dm/._distribution/._link/._groups/...` and forwards to `superglm.reml.*`:
- `model_compute_dW_deta` (13) -> `reml.w_derivatives.compute_dW_deta`.
- `model_reml_w_correction` (18) -> `reml.w_derivatives.reml_w_correction`.
- `model_reml_laml_objective` (46) -> `reml.objective.reml_laml_objective` (injects `_reml_result.scop_states`, `_reml_penalties`).
- `model_reml_direct_gradient` (72) / `model_reml_direct_hessian` (97) -> `reml.gradient.*`.
- `model_optimize_direct_reml` (135-190) -> `reml.direct.optimize_direct_reml`; **on discrete, rebuilds `model._dm` afterwards** (186-189).
- `model_optimize_discrete_reml_cached_w` (193-232) -> `reml.discrete.optimize_discrete_reml_cached_w` — **no production caller** (real discrete dispatch happens *inside* `optimize_direct_reml`, reml/direct.py:108).
- `model_optimize_efs_reml` (235-283) -> `reml.efs.optimize_efs_reml`; reassigns `model._dm`.
- `model_run_reml_once` (286-333) -> `reml.runner.run_reml_once` (runner.py:68); reassigns `model._dm` — **no production caller**.

### 1.9 `model/reml_execute.py` (372 lines) — REML route execution
- Trace helpers `_trace_rows_enabled` (17), `_lambda_max_delta` (22), `_record_non_scop_reml_trace` (37), `record_reml_terminal` (56-87, only path emitting the authoritative terminal event, called post-install from fit_ops.py:1092).
- `run_fixed_monotone_reml` (90-174): fixed-lambda constrained path; SCOP -> `reml.scop_efs.fit_fixed_scop_reml`, else `fit_irls_direct`; then inline eta/mu/fit-stats block (153-174).
- `run_scop_efs_reml` (177-256): `reml.scop_efs.optimize_scop_efs_reml` + inline eta/mu/fit-stats block (234-255).
- `optimize_reml_best` (259-372): four-way dispatch (estimated_names x use_direct). When `estimated_names` is empty it still calls the full optimizer with `max_reml_iter=1, reml_tol=1.0` as a fixed-lambda hack (287-328). Receives `model_optimize_direct_reml`/`model_optimize_efs_reml` **as function parameters** (fit_ops.py:1383-1384) even though they are plain imports one module away.

### 1.10 `model/reml_finalize.py` (688 lines) — terminal refit and publication of REML fits
- `finalize_reml_fit` (317-688), the heaviest orchestration routine:
  - Installs `best.pirls_result`, `_reml_lambdas`, `_reml_penalties`, `_reml_result` (339-346).
  - Direct path: `rebuild_dm_with_lambdas` (366) + second `build_penalty_context` (367), `_map_beta_between_bases` (370), then a **full terminal `fit_irls_direct` refit** (378-398) with `return_xtwx=True` (tolerance tightened to 1e-10 for observed-curvature terminals, 377).
  - `maybe_qp_passthrough_refit` (275-314): third constrained refit for QP passthrough.
  - `_build_reml_reporting_support_state` (162-194): compact reporting distillation when `retain_fit_state=False` (or forced by a structured terminal).
  - `_build_structured_linear_system_state` (62-146): distills profiled Schur/sum-to-zero factors into `StructuredLinearSystemState`; `_structured_information_by_group` (149-159).
  - Terminal LAML re-evaluations: QP branch (460-498), observed-curvature branch (499-584, incl. `build_observed_reml_geometry` + `observed_penalized_mode_score` with a hard RuntimeError gate at 552-557), Fisher branch (586-627). Each rebuilds S via `build_penalty_matrix` when unstructured.
  - `compute_profiled_phi` (204-272): Gaussian/Gamma Wood-Eq(4) profile, generic reduced profile otherwise; phi injected via `replace(final_pirls, phi=phi_fixed)` (656).
  - `update_reml_r_inv` (662 -> reml_state.py:22); inline eta/mu/fit-stats block (669-678); `restore_qp_group_state` (687, duplicate of reml_setup.restore_qp_constraints).

### 1.11 `model/reml_state.py` (79 lines) — spec reparametrisation sync
- `update_reml_r_inv` (22-79): after REML convergence, pushes each SSP group's `R_inv` back onto feature/interaction specs (`set_reparametrisation`) so predict/reconstruct operate in the fitted basis. Inner lookups use `next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)` inside nested loops — O(G²) name scans.

### 1.12 `model/runtime_canonicalize.py` (528 lines) — solver->public canonicalization
- `canonicalize_fitted_model` (456-502): called at the end of every fit path (fit_ops.py:883, 1016, and via `_canonicalize_fitted_model` for REML). Steps: `_compile_runtime_terms` (289-341) computes per-term training column means (`_group_column_means` 132 via `rmatvec(ones)`, or exact means via `_runtime_training_feature_column_means` 170-202: `np.unique` over n rows + chunked basis transform with Kahan summation), mutates specs in place (`_apply_r_inv_centering` 240, `_apply_scop_centering` 246), accumulates the public intercept shift; `_build_public_result` (444-453) creates the public `PIRLSResult` with shifted intercept and **copied but numerically identical beta**; when `validate=True`, `_compute_public_parity_diagnostics` (399-430) does one solver-space `dm.matvec` pass plus `_live_public_runtime_state` (344-396) which re-scores every term on all training rows through the public scoring contract — two extra full-data passes.
- `_IdentityCoefficientMap` (30-88): lazy identity "solver_to_public" map — only ever identity because `_validate_workspace_result` requires beta equality.
- `canonicalize_intercept_path` (505-528): O(L·p) path intercept mapping.

### 1.13 `model/path_ops.py` (113 lines)
- `validate_lambda_path_controls` (13-36), `resolve_lambda_sequence` (39-53, geomspace), `run_lambda_path` (56-113): warm-started PIRLS loop over L lambdas; allocates `coef_path` (L, p); mutates `configured_penalty(model).lambda1` in place per step (path_ops.py:81) — safe only because the penalty is workspace-owned.

### 1.14 `model/profile_ops.py` (465 lines) — Tweedie p / NB theta profile orchestration
- `estimate_p` (17-159): validate -> profile `FitWorkspace` (`estimate_tweedie_p` runs candidate fits inside it) -> discard -> second `FitWorkspace` with `config_overrides={family, retain_fit_state=True}` -> `_fit_reml_in_workspace` or `_fit_in_workspace` -> `_synchronize_tweedie_profile_refit` (217-280: recompute eta/mu/null_mu/fit-stats at profiled phi, swap phi into public+solver+reml results) -> optional CI -> `capture_fit_state` with a `ModelConfigPublication` carrying the new family (144-155) -> `_install_fit_state`.
- `estimate_theta` (283-403): same shape for NB.
- `_installed_tweedie_profile_copy` (171-199): surgical copy sharing `_objective/_evaluation_*` runtime but owning CI caches.
- `_resolve_profile_fit_mode` (406-419): "inherit" reads `_last_fit_meta["method"]`.

### 1.15 `model/shape_ops.py` (766 lines) — post-fit shape repair (transactional)
- `apply_shape_postfit` (638-766): guards (`require_unchanged_fit_data` 658, `FitDataGuard.matches` 661), no-op fast path when constraints already feasible (673), `FittedStateRevision.start` (694), per-term isotonic/curvature repair (`_repairer` 36) with heavy pre-publication certification `_validate_repair_for_publication` (399-512: constraint certificate, centering check, candidate-vs-zero-fallback objective comparison, each requiring `_profile_repaired_intercept` 209-329 — an up-to-50-iteration Newton with 21-step halving over n rows per evaluation), `_replace_result_beta`/`_synchronize_repaired_intercept_state` (60-113), then `invalidate_revised_coefficient_mode`, `_refresh_fit_statistics` (editor.apply), `_refresh_repaired_geometry` (540-584: possibly full `fit_inference_info` rebuild), `_refresh_repaired_scale_and_statistics` (587-635), `revision.commit()`.
- `_build_smooth_penalty_terms` (128-176) / `_smooth_penalty_value` (179-206): compact beta'Sbeta evaluation, REML-penalty-aware.

### 1.16 Remaining modules
- `model/input_validation.py` (105 lines): `validate_fit_input` (58-105) — frame coercion, complex/datetime rejection per column, y/weight/offset vector checks, Tweedie strict prior weights, `validate_response`. Called only by `fit_ops._validate_entrypoint_input` (fit_ops.py:359) and thus by all four entry points.
- `model/fit_data_guard.py` (187 lines): `FitDataGuard` (102-174, blake2b column digest + full y copy) and `FitGeometryGuard` (32-98, constant-size digests incl. weights/offset); `require_unchanged_fit_data` (177). Consumers: `_prime_fit_caches`, `explain_ops.metrics`, shape_ops, editor.
- `model/telemetry_ops.py` (364 lines): `training_telemetry` (16), `reml_diagnostics` (40), `feature_schema` (74), `edf_by_term` (118), `metrics_for_logging` (135); defensive `getattr` everywhere; `_json_ready` (322) round-trips through `json.dumps` on every call.
- `model/report_ops.py` (567 lines): `summary` (66-343) — computes AIC/BIC/AICc/EBIC, consumes `_fit_inference_info`, `build_coef_rows`, per-group SEs from `covariance_selected_diagonal` (303), memoised in `_summary_cache` keyed on (alpha, detail, level_display, tweedie identity) (84-93); `diagnostics` (23-63); `_build_editor_stale_coef_rows` (351-512); `reconstruct_feature` (529-546); `knot_summary` (549-567); `feature_groups` (346-348) — linear scan helper used widely.
- `model/explain_ops.py` (258 lines): thin delegation to `superglm.inference.*` and `superglm.diagnostics.*`; `metrics` (33-83) implements the identity+guard-based metrics cache; editor-stale warnings (118-120, 184-187).
- `model/plot_ops.py` (316 lines): `plot` (22-206) and `plot_data` (209-317) share ~70 lines of identical term-classification logic (67-111 vs 239-277).
- `model/monotone_ops.py` (29 lines): compatibility aliases -> `shape_ops.apply_shape_postfit`.
- `model/design_summary.py` (189 lines): `build_design_summary` (140-186) — read-only DataFrame over `design.execution_plan`; type->metadata table (42-97).
- `model/reml_debug.py` (392 lines): `REMLDebugRecorder` (62-97, JSONL side-channel gated by `SUPERGLM_DEBUG` level), loaders/summarisers/plotters (100-376) — offline tooling, only `REMLDebugRecorder` is imported by fit_ops (fit_ops.py:436).
- `model/__init__.py` (8 lines): exports `SuperGLM`, `PathResult`; rewrites `SuperGLM.__module__`.

### 1.17 The fit_reml call chain, explicitly

Live path (direct/exact and discrete):
```
SuperGLM.fit_reml                        api.py:512
 -> fit_ops.fit_reml                     fit_ops.py:1038   (validate, FitWorkspace.start, publish)
  -> _fit_reml_in_workspace              fit_ops.py:1116   (design build, group/penalty setup, routing)
   -> reml_execute.optimize_reml_best    reml_execute.py:259  (4-way dispatch)
    -> reml_ops.model_optimize_direct_reml  reml_ops.py:135  (unpack model attrs)
     -> reml.direct.optimize_direct_reml    reml/direct.py:56   (Newton outer loop; discrete
                                             branches internally to reml/discrete.py:164)
   -> reml_finalize.finalize_reml_fit    reml_finalize.py:317 (terminal refit + LAML + phi)
  -> fit_state.capture_fit_state         fit_state.py:678
  -> fit_state._install_fit_state        fit_state.py:703  (dict swap)
 -> _record_reml_terminal_best_effort    fit_ops.py:1096
```
That is 5 orchestration layers between the public method and the actual optimizer, plus 2 more inside `reml/` for the discrete path. The EFS (lambda1>0) route swaps layer 5 for `reml_ops.model_optimize_efs_reml` -> `reml.efs.optimize_efs_reml`.

`reml/runner.run_reml_once` is NOT on any live path. Its only inbound chain is `SuperGLM._run_reml_once` (api.py:1706) -> `fit_ops.model_run_reml_once` (re-export, fit_ops.py:52) -> `reml_ops.model_run_reml_once` (reml_ops.py:286) -> `reml.runner.run_reml_once` (reml/runner.py:68), and nothing calls `SuperGLM._run_reml_once` (grep over src + tests: only `tests/test_import_compat.py:151` imports the free function).

---

## 2. DATA FLOW

### 2.1 Ordinary fit()
1. Caller arrays enter `fit_ops.fit` (fit_ops.py:718). Original references kept as `X_ref/y_ref/...` for cache identity.
2. `validate_fit_input` (input_validation.py:58) materialises `y (n,) float64`, `sample_weight (n,)` (ones if None), `offset (n,)|None`; column scans are O(n) per required column.
3. `FitWorkspace.start` (fit_workspace.py:79) creates a fresh model from `ModelConfig` (deepcopy of all spec templates — no row data).
4. `model_build_design_matrix` (base.py:919) -> `dm_builder.build_design_matrix`: constructs per-group matrices (sparse spline bases ~n·(degree+1) nnz per spline; categorical code vectors (n,); discretized unique-basis matrices (bins, k) + codes (n,)). `model._dm` (n x p logical), `model._groups` (G GroupSlices) installed. This is the dominant construction allocation.
5. `_store_fit_arrays` (fit_ops.py:413) copies weights/offset -> `_fit_weights (n,)`, `_fit_offset (n,)`.
6. `_solve_coefficients` (659): PIRLS/IRLS materialises per-iteration W (n,), working response z (n,), gram X'WX (p,p) dense (gram path) or QR of weighted design (n,p) (qr path). Returns `PIRLSResult` (beta (p,), intercept, phi, edf, rank_info, ...).
7. Post-solve: eta (n,) = `dm.matvec(beta)` + intercept + offset (fit_ops.py:857-860), mu (n,), null_mu (n,) via 25-step Newton (863), `_compute_fit_stats` scalars (864).
8. `_prime_fit_caches` (875 -> 544): **recomputes eta/mu (564-568) and null_mu (569-575) from scratch**, stores `_fit_mu`, `_fit_null_mu`, X/y refs, `FitDataGuard` (full y copy (n,) + blake2b digest over guard columns) and `FitGeometryGuard` (digests of y/w/offset, offset zeros allocated if absent, fit_ops.py:598-603).
9. `canonicalize_fitted_model` (runtime_canonicalize.py:456): per spline-backed term, column means (k_g,) via `rmatvec(ones (n,))` or exact unique-value means (`np.unique` on (n,)); mutates spec `_R_inv` in place; builds public `PIRLSResult` with intercept += Σ means·beta_g; validation does 1 solver matvec pass + 1 full public re-scoring pass over all n rows (399-430, 344-396); freezes the prediction plan.
10. `capture_fit_state` -> `_capture_model_state` (fit_state.py:617): validates, freezes arrays, snapshots 38 projection slots into `FitState`, shallow-transfers the workspace `__dict__`, restores caller config identities.
11. `_install_fit_state` (703): one dict swap; the public model now owns the workspace buffers (no row-scale copies at publication).
12. `_maybe_release_fit_state` (610): if `retain_fit_state=False`, evaluates `_fit_inference_info` (O(p³)) first, keeps (p_a,p_a) covariance + group EDF map, then nulls `_dm` and all (n,)-scale buffers.

### 2.2 fit_path()
Same as above through step 5, then `run_lambda_path` (path_ops.py:56): L warm-started PIRLS solves; `coef_path (L,p)`, `intercept_path (L,)`, deviance/edf/n_iter/converged (L,). Only the last solution becomes model state; `PathResult.__post_init__` re-copies each array twice (fit_ops.py:90-95). `canonicalize_intercept_path` adds `coef_path[:, sl] @ means` per applied term (O(L·p)).

### 2.3 fit_reml()
Steps 1-5 as fit, then:
- `collect_reml_groups` -> list of (index, GroupSlice) length q' (penalized structured/SSP groups).
- `build_penalty_context` (reml/penalty_algebra, fit_ops.py:1241): per penalty component eigendecomposition of omega (k_j x k_j) -> `penalty_caches`, `penalty_ranks`; computed once and threaded through the optimizer.
- `lambdas: dict[str,float]` (q entries), `estimated_names: set[str]`.
- `optimize_direct_reml` (outside subsystem): per REML iteration runs inner PIRLS to convergence, forms `XtWX_S_inv (p,p)`, gradient (q,), Hessian (q,q), W-correction terms; on discrete uses cached-W fREML kernels. Returns `REMLResult best` with `pirls_result`, `lambdas`, `lambda_history`, `objective`.
- `model_optimize_direct_reml` post-step (reml_ops.py:186-189) rebuilds `model._dm` at final lambdas when discrete (fresh discretized SSP blocks; O(Σ bins·k²) per group).
- `finalize_reml_fit` (reml_finalize.py:317): for direct paths, rebuilds `_dm` **again** at final lambdas (366; the discrete case therefore rebuilds twice in a row), `_map_beta_between_bases` maps beta across reparametrisations (per-group k_g² multiplies), terminal `fit_irls_direct` refit to convergence returning `final_xtwx (p,p)` and factor; QP passthrough triggers a third refit; observed-curvature terminals additionally build observed geometry (extra O(np²)/O(p³)-class work outside this subsystem) and re-evaluate LAML; `compute_profiled_phi` computes beta'Sbeta via compact components; `update_reml_r_inv` pushes (k_g x k_g) `R_inv` blocks into specs; final eta/mu/fit-stats pass (669-678).
- Back in `_fit_reml_in_workspace`: `_prime_fit_caches` (recompute eta/mu/null_mu again), `_canonicalize_fitted_model` (skips full-row validation when n > 100_000 or fast_candidate, fit_ops.py:71, 501-521), publish.
- State installed on model afterwards: `_reml_lambdas` (q floats), `_reml_penalties` (q PenaltyComponents holding (k_j,k_j) omegas), `_reml_result` (REMLResult incl. lambda/objective history), `_reml_profile` (timings dict), optionally `_linear_system_state` (compact structured factors) and `_reporting_support_state`.

### 2.4 Post-fit consumers
- `state_ops.fit_inference_info`: W (n,) via one matvec, centered data gram (p_a,p_a) built from group kernels (no n x p materialisation; `_rank_centered_data_gram` 241-253), eigh (O(p_a³)) for R_a, pseudo-inverses (O(p_a³)). Cached until next fit/repair.
- `report_ops.summary` consumes the info dict; cached per (alpha, detail, level_display, tweedie-identity).
- `explain_ops.metrics` reuses `_fit_mu/_fit_null_mu/_fit_stats` only when caller passes the identical objects AND digests match (33-83).
- Prediction: `_prediction_plan` -> per-term `spec.score/transform` on new rows; fast-discrete path bins rows against fit-time support metadata (base.py:91-112).

---

## 3. STATE OBJECTS

| Object | Where | Fields | Lifecycle | Overlap notes |
|---|---|---|---|---|
| `ModelConfig` | fit_state.py:76 | 20 frozen constructor-intent fields incl. deepcopied spec/interaction templates | Captured at `init_model` end and on every property set / `_add_interaction`; source of truth for every `FitWorkspace` | Redundant with the loose `_*_config` + `_specs/_splines/_n_knots/...` slots that `init_model` also writes (base.py:661-693); `materialize` (177) must mirror `init_model`'s slot list by hand |
| `ModelConfigPublication` | fit_state.py:245 | config, revision, penalty/link/family/lambda2 identities | Captured at publication to re-install caller-owned config over the workspace dict | Pure identity bundle |
| `FitState` | fit_state.py:280 | revision, selection_penalty, distribution, `projections` (FrozenMapping of 38 slots), retained, repair_revision, resolved_penalty, resolved_lambda2 | Frozen at publication; read by `fitted_penalty/fitted_lambda2`, `clone_without_features`, `selection_penalty_` | Its `projections` alias the exact same objects stored as model attributes — a parallel read view, not a copy |
| `FitCandidate` | fit_state.py:294 | state + prepared `__dict__` | Transient between capture and install | — |
| `FitWorkspace` | fit_workspace.py:69 | work model, mode, validated_inputs, previous_revision | One per fit attempt; discarded on failure (strong exception safety) | Overlaps in purpose with `FittedStateRevision` (both are transactional workspaces with different cloning strategies) |
| `FittedStateRevision` | fit_state.py:302 | target_model, shallow-copied work model, revision, repair_revision | Post-fit coefficient revisions (shape_ops, editor) | Shares heavy buffers with target by design; must hand-deepcopy nested mutables (shape_ops.py:697-701) |
| `PIRLSResult` (`_result` public vs `_solver_result` solver-space) | solvers/pirls | beta (p,), intercept, phi, edf, rank_info, iteration_log, scop_inference, ... | Both installed per fit; enforced identical beta, intercept differs by canonical shift | Two objects whose only sanctioned difference is a scalar; kept coherent by `_validate_workspace_result` and by triple-update loops in shape_ops (61-68, 567-583, 618-624) |
| `_runtime_canonical_state` (dict) | runtime_canonicalize.py:495 | terms {group metadata, column_means (k_g,), shifts}, diagnostics, intercept_shift, solver_to_public (identity map), solver_to_public_complete | Built at every canonicalization; consumed by `state_ops._public_intercept_shift`, `shape_ops`, `canonicalize_intercept_path` | Column means duplicated per term here and re-derivable from `_dm`; the identity map machinery is currently vestigial (beta equality enforced) |
| `_fit_state trio` of REML slots: `_reml_lambdas`, `_reml_penalties`, `_reml_result`, `_reml_profile` | set in reml_execute/reml_finalize | q lambdas, q PenaltyComponents, REMLResult (incl. aliased pirls_result), timing dict | Cleared by `_clear_reml_state` (fit_ops.py:405) on ordinary fits; not part of `materialize`'s base dict (hasattr-guarded in `_FIT_PROJECTION_NAMES`) | `_reml_result.pirls_result` must alias `_solver_result` — enforced at fit_state.py:519, re-synced manually at reml_finalize.py:658, profile_ops.py:266 |
| `FitDataGuard` | fit_data_guard.py:102 | frame digest, full y snapshot (n,), columns | Captured in `_prime_fit_caches` only when retained; None otherwise | Overlaps `FitGeometryGuard`; two guards with different equality semantics (identity-cache vs geometry) both digesting the same frame |
| `FitGeometryGuard` | fit_data_guard.py:32 | digests of X/y/w/offset, n_rows | Always captured (even with offset=None → zeros allocated) | see above |
| `StructuredLinearSystemState` | solvers/structured, built at reml_finalize.py:62-146 | coefficient/profiled/augmented factors, system, operators, support totals | Only for structured direct terminals; consumed by all three state_ops entry points | Third covariance representation next to rank_info and scop_inference |
| `_fit_inference_info` dict | state_ops.py:411 | W (n,), XtWX_inv (p_a,p_a), XtWX_inv_aug (p_a+1)², R_a, edf, edf1, group_edf_map, coefficient_estimable | cached_property; invalidated by 4 different hand-maintained lists (fit_ops.py:380, shape_ops.py:46, profile_ops.py:271, editor/apply.py:333) | Overlaps `_coef_covariance` and `_fit_active_info` (all three recompute W and re-branch the same 4-way dispatch) |
| `_prediction_plan` / `_fast_prediction_state` | base.py:203/165 | per-term spec refs, beta index arrays, discretizer metadata (deepcopied) | Frozen at canonicalization; rebuilt lazily if popped | Spec objects aliased with `_specs`; fast metadata deepcopied twice (state + plan) |
| `_fit_metrics_cache` / `_summary_cache` | explain_ops.py:80 / report_ops.py:84 | ModelMetrics object; dict keyed summaries | Identity+digest guarded; cleared in ≥5 places | — |
| `REMLDebugRecorder` / `REMLDebugRun` | reml_debug.py:62/25 | trace level, run id, JSONL sinks | Only when `SUPERGLM_DEBUG` level > 0 | Legacy `append_jsonl` rows explicitly non-authoritative vs canonical `TraceRun` events — two trace schemas live side by side (reml_debug.py:86-97) |
| `_CompactPenaltyTerms` | shape_ops.py:116 | lambdas, penalty components, group matrices | Transient during repair | Duplicates `total_penalty_quadratic` inputs |
| `CanonicalizationDiagnostics` | runtime_canonicalize.py:19 | eta/mu deltas, shifts, term means | Stored as dict inside `_runtime_canonical_state` | — |
| `ValidatedFitInput` | input_validation.py:17 | frame + 3 arrays | Transient | — |

---

## 4. COMPLEXITY TABLE

| Routine | Time | Memory (extra) | Notes |
|---|---|---|---|
| `validate_fit_input` (input_validation.py:58) | O(n·C_req) | O(n) per coerced vector | column dtype scans; object-dtype triggers `infer_dtype` O(n) |
| `ModelConfig.capture/materialize` (fit_state.py:101/177) | O(spec sizes) deepcopy | full spec templates duplicated per fit attempt | knot arrays etc. copied twice per fit (capture at init + materialize per attempt) |
| `model_build_design_matrix` (base.py:919) | O(n·Σk_g) construction | dm ~ O(Σ nnz_g); discretized groups O(n + bins·k) | dominant allocation; dense per-group only for Dense/Polynomial |
| `compute_lambda_max` (base.py:958) | O(nnz(dm)) rmatvec + O(p) | O(p) | once per fit/fit_path |
| `_solve_coefficients` -> `fit_irls_direct` gram path | O(I·(nnz + p²·assembly + p³ solve)) | W,z (n,), gram (p,p) | qr path O(I·n·p²) documented at api.py:142-144 |
| `_solve_coefficients` -> `fit_pirls` (BCD) | O(I_outer·I_bcd·Σ n·k_g) | per-group workspaces | lambda1>0 only |
| `_compute_null_mu` (fit_ops.py:240) | O(25n) with offset; O(n) without | O(n) | executed **twice per fit** (see suspects) |
| `_compute_fit_stats` (fit_ops.py:278) | O(n); Tweedie logpdf pair heavier | O(n) | Tweedie path uses series evaluation |
| `_prime_fit_caches` (fit_ops.py:544) | O(nnz) matvec + O(25n) + O(n·C_guard) digests | `_fit_mu`,`_fit_null_mu` (n,) + y snapshot (n,) + zeros offset (n,) | duplicates caller-computed eta/mu/null_mu |
| `canonicalize_fitted_model` validate=True (runtime_canonicalize.py:456) | O(nnz) matvec + O(Σ_m n·k) public re-score + per-term O(n log n) unique | contributions dict m×(n,) | skipped for n>100k in REML auto mode only; ordinary `fit()` always validates (fit_ops.py:883 passes no flag) |
| `_runtime_training_feature_column_means` (runtime_canonicalize.py:170) | O(n log n) unique + O(u·k) transform per materializable term | (u,k) chunks of 8192 | u = unique values |
| `capture_fit_state` + `_install_fit_state` (fit_state.py:678/703) | O(#attrs) shallow + validation O(p) | none (ownership transfer) | deliberate zero-copy publication |
| `run_lambda_path` (path_ops.py:56) | L × PIRLS cost | coef_path (L,p) | plus `PathResult` immutability re-copy ×2 (fit_ops.py:90-95) and `canonicalize_intercept_path` O(L·p) |
| `collect_reml_groups` / `initialize_component_lambdas` (reml_setup.py:19/51) | O(G) / O(q) | O(q) | trivial |
| `build_penalty_context` (called fit_ops.py:1241, again reml_finalize.py:343/367) | O(Σ k_j³) eigen per component | caches (k_j,k_j) | recomputed up to 3× per REML fit (initial + non-direct finalize + direct rebuild) |
| `optimize_direct_reml` (via reml_ops.py:135) | R × (inner PIRLS + O(p³) inverse + O(q·p²) gradient/Hessian + W-correction) | XtWX_S_inv (p,p), per-penalty caches | outside subsystem but driven from here |
| `rebuild_dm_with_lambdas` (base.py:1009) | O(Σ bins·k² + reparam) per rebuild | new group matrices | called twice back-to-back on discrete path (reml_ops.py:186 then reml_finalize.py:366) |
| `finalize_reml_fit` terminal refit (reml_finalize.py:378) | full extra IRLS to convergence (tol 1e-10 for observed) + optional QP refit + LAML O(p³) + observed geometry (heavy) | final_xtwx (p,p), S (p,p) unless structured | 1-3 extra full solves after the optimizer already converged |
| `update_reml_r_inv` (reml_state.py:22) | O(G²) name scans + O(Σ k_g²) hstacks | R_inv copies per spec | quadratic in group count |
| `state_ops._solver_space_working_weights` (33) | O(nnz) + O(n) | (n,) | recomputed independently by each of the three cached entry points on first access |
| `state_ops.fit_inference_info` (411) | O(nnz + p_a²) gram + O(p_a³) eigh/pinv (×2-3 decompositions in legacy path) | several (p_a,p_a) | legacy branch does 3 rank decompositions + 2 pseudo-inverses (199-229) |
| `state_ops._legacy_active_state` (139) | O(p_a³)×3 + factor certifications O(n·p_a²) worst case | grams (p_a,p_a) | only for fits without rank_info |
| `report_ops.summary` (66) | O(G·k) rows + `covariance_selected_diagonal` O(Σ k_g²) | coef rows | cached |
| `shape_ops.apply_shape_postfit` (638) | per repaired term: isotonic O(grid) + certification with ~2×(50×21 worst-case) Newton evaluations O(n) each + `_shape_term_eta_delta` O(nnz) ×2 + possible full `fit_inference_info` O(p³) | eta copies (n,) | intercept profiling is the n-scale hotspot |
| `profile_ops.estimate_p/theta` (17/283) | full profile search (many fits) + one complete final fit | two sequential workspaces | final refit rebuilds design from scratch (expected) |
| `telemetry_ops.training_telemetry` (16) | O(size of payload) + `json.dumps` validation pass | payload dict | double serialization (validate + caller's own dump) |
| `FitDataGuard.capture` (fit_data_guard.py:110) | O(n·C_guard) blake2b + O(n) y copy | y snapshot (n,) | per fit when retained |

---

## 5. SUSPECTS

Ordered by expected value to the audit; each with files+lines, why suspicious, and what to verify.

### S1. Dead REML adapter stack: `run_reml_once` chain and `model_optimize_discrete_reml_cached_w`
- **Where**: api.py:1648-1676 (`_optimize_discrete_reml_cached_w`), api.py:1706-1734 (`_run_reml_once`); fit_ops.py:46,52,80,86 (re-exports); reml_ops.py:193-232, 286-333; the entire `reml/runner.py` module (its only importers are `reml/__init__.py:50` and reml_ops.py:9).
- **Why**: grep over src+tests shows no caller of `SuperGLM._run_reml_once` or `SuperGLM._optimize_discrete_reml_cached_w` anywhere; the discrete cached-W optimizer is dispatched internally by `optimize_direct_reml` (reml/direct.py:108). The api comment "REML adapter methods (used by reml_optimizer)" (api.py:1527) is stale — `reml_execute.optimize_reml_best` receives the two live optimizers as function parameters from fit_ops (fit_ops.py:1383-1384), not through the model methods. Roughly 3 of the 9 adapter methods have live callers only in tests.
- **Verify**: run the test suite after deleting `_run_reml_once`/`_optimize_discrete_reml_cached_w` and `reml/runner.py`; check external/notebook usage; confirm `test_import_compat.py` is the only guard keeping `reml.runner` alive.

### S2. Triple-layer adapter indirection (api -> fit_ops re-export -> reml_ops -> reml/*)
- **Where**: fit_ops.py:43-53 + 73-87 (pure re-export of 9 functions); api.py:1527-1734 (method wrappers over the re-exports); reml_ops.py entire file.
- **Why**: three module layers carry identical signatures with zero logic; `optimize_reml_best` additionally takes the optimizers as parameters (reml_execute.py:279-280) although they are importable one hop away — dependency-injection with a single production binding. This is the main contributor to the 5-layer fit_reml chain (§1.17).
- **Verify**: confirm no subclass overrides these `SuperGLM._reml_*` hooks (that would be the only reason to keep model-level methods); check tests that monkeypatch `fit_ops.model_optimize_direct_reml`.

### S3. Duplicated post-solve finalization block (eta/mu/null-mu/fit-stats) — 6 near-identical copies, each followed by a full recomputation in `_prime_fit_caches`
- **Where**: fit_ops.py:857-873 (`_fit_in_workspace`), fit_ops.py:991-1006 (`_fit_path_in_workspace`), fit_ops.py:1203-1218 (REML no-groups fallback), reml_execute.py:153-161 (`run_fixed_monotone_reml`), reml_execute.py:234-248 (`run_scop_efs_reml`), reml_finalize.py:669-678 (`finalize_reml_fit`); then `_prime_fit_caches` (fit_ops.py:564-575) recomputes eta = `dm.matvec(beta)`, mu, and re-runs the 25-iteration `_compute_null_mu` Newton.
- **Why**: (a) copy-paste maintenance hazard — the six blocks differ only in which result object they read; (b) measurable waste: every fit does 2× full-design matvec + 2× null-model Newton (each O(25n) with offset). For large n REML fits (the library's headline use case) this is two avoidable O(n)-scale passes stacked on top of the canonicalization passes (S4).
- **Verify**: profile a 1-5M-row Poisson `fit_reml` and measure `_prime_fit_caches` + duplicate eta/mu time vs total; confirm `_compute_null_mu` results are bit-identical between the two computations (they consume the same inputs).

### S4. Post-fit full-data validation passes in canonicalization — ordinary `fit()` always pays them
- **Where**: runtime_canonicalize.py:399-430 (`_compute_public_parity_diagnostics`: solver matvec over n rows) and 344-396 (`_live_public_runtime_state`: re-scores every term through the public `spec.score/transform` contract over all n training rows); the `validate` flag exists (456) and is auto-skipped for large REML fits (fit_ops.py:501-521, threshold 100_000 rows), but `_fit_in_workspace` (fit_ops.py:883) and `_fit_path_in_workspace` (1016) call `canonicalize_fitted_model(model)` with default `validate=True` — **no size gate on the ordinary fit path**.
- **Why**: for large-n `fit()` (no REML), the parity diagnostic adds 2 extra full-data passes (one of them through the slower per-term public scoring path, including sparse basis re-construction per spline term) purely for a max-abs-delta diagnostic. Also `_runtime_training_feature_column_means` adds O(n log n) `np.unique` per materializable spline term.
- **Verify**: time `canonicalize_fitted_model` on a wide spline model at n=10⁶ via `fit()`; check whether any consumer reads `diagnostics["max_abs_eta_delta"]` outside tests; confirm skipping validation for large ordinary fits would not break `_validate_workspace_result` (it doesn't read the diagnostics, only `intercept_shift`).

### S5. Discrete REML rebuilds the design matrix twice back-to-back
- **Where**: reml_ops.py:186-189 (`model_optimize_direct_reml`: `if model._discrete: model._dm = rebuild_dm_with_lambdas(...)`) followed immediately by reml_finalize.py:364-367 (`if use_direct: model._dm = rebuild_dm_with_lambdas(...)` for **all** direct fits) inside the same `_fit_reml_in_workspace` invocation. Also `build_penalty_context` is recomputed at reml_finalize.py:343 (non-direct) and 367 (direct) after the initial computation at fit_ops.py:1241.
- **Why**: the reml_ops rebuild result is discarded — finalize rebuilds from the same lambdas again. Each rebuild reconstructs reparametrised group matrices (Cholesky/eigen per group, O(Σ k³ + bins·k²)); penalty context recomputation repeats O(Σ k³) eigendecompositions. Wasted work scales with m and k, and this sits directly on the headline `discrete=True` large-n path.
- **Verify**: check whether `rebuild_dm_with_lambdas` is deterministic in lambdas (it appears to be); instrument both call sites on a discrete fit and confirm the first rebuild's output is never read before being overwritten (only `finalize_reml_fit` touches `model._dm` after `optimize_reml_best` returns via `best`).

### S6. `state_ops` four-way covariance dispatch triplicated across three entry points
- **Where**: state_ops.py:321-347 (`coef_covariance`), 350-389 (`fit_active_info`), 411-578 (`fit_inference_info`) — each re-implements the scop_inference / StructuredLinearSystemState / rank_info / legacy branch ladder; `_legacy_active_state` (139-238) is a fourth covariance engine with its own rank certification cascade; each entry point independently recomputes `_solver_space_working_weights` (O(nnz)+O(n)).
- **Why**: unclear ownership — the same algebra ((X'WX+S)⁻¹, augmented intercept covariance, EDF) exists in 4 representations x 3 consumers; any new solver state (e.g., another structured factor) needs edits in 9+ places. `fit_active_info` appears to have exactly one consumer (`inference/metrics.py:722`) yet is a separately cached near-duplicate of `fit_inference_info`.
- **Verify**: enumerate which branch combinations are reachable per fit route (e.g., can scop_inference and rank_info be simultaneously None on a modern fit? — `_legacy_active_state` may be dead for freshly fitted models); check whether `coef_covariance` could be derived from `fit_inference_info` outputs without the separate branch ladder.

### S7. QP constraint save/restore duplicated and doubly executed
- **Where**: reml_setup.py:154-173 (`strip_qp_constraints`/`restore_qp_constraints`) vs reml_finalize.py:197-201 (`restore_qp_group_state` — identical body); restore runs inside `finalize_reml_fit` (687), again in `maybe_qp_passthrough_refit` (294), and again in the `finally` of `_fit_reml_in_workspace` (fit_ops.py:1431-1434).
- **Why**: idempotent but triple-owned; the in-place mutation of shared `GroupSlice` objects (monotone_engine/constraints) inside a "private workspace" is the one place the workspace mutates state that `_dm`/groups aliases could observe mid-fit. The two-stage QP passthrough is explicitly a heuristic (fit_ops.py:1252-1256) — fine — but its state management is scattered across three modules.
- **Verify**: confirm `model._groups` in the workspace is never aliased by the public model (it is workspace-built, so safe); consolidate to one restore owner and check tests.

### S8. Fixed-lambda REML routed through the full optimizer with `max_reml_iter=1, reml_tol=1.0`
- **Where**: reml_execute.py:285-328 (`optimize_reml_best` when `estimated_names` empty).
- **Why**: encoding "just fit at these lambdas" as a degenerate optimizer call performs one full gradient/Hessian evaluation (O(p³) + W-correction) that is thrown away, and couples the fixed-lambda semantics to optimizer implementation details (does 1 iteration with tol 1.0 really guarantee zero lambda movement? damped-Newton could still take one step if the loop body updates before checking).
- **Verify**: read `optimize_direct_reml`'s loop structure to confirm iteration-1 cannot move lambdas; measure the discarded derivative cost for wide models; compare against a plain `fit_irls_direct` at fixed lambdas + terminal LAML.

### S9. Cache-invalidation lists hand-maintained in ≥5 places
- **Where**: fit_ops.py:380-402 (`_clear_fit_inference_caches`, 20 slots), shape_ops.py:46-57 (`_invalidate_repair_caches`, 7 entries), profile_ops.py:271-280 (`_synchronize_tweedie_profile_refit`, 7 entries), editor/apply.py:333 (external), fit_state.py:180-240 (`materialize` baseline dict) and base.py:695-726 (`init_model` slot list — a sixth copy of the attribute inventory).
- **Why**: the model's ~50-slot attribute inventory is replicated by hand in three constructors/materializers and three invalidators; `_FIT_PROJECTION_NAMES` (fit_state.py:421) is a fourth partial copy. A newly added cache that misses one list becomes a stale-state bug (the class of bug the whole workspace architecture exists to prevent).
- **Verify**: diff `init_model` slots vs `materialize` dict vs `_FIT_PROJECTION_NAMES` for drift (e.g., `_fit_geometry_guard` is in materialize + projections; `_shape_repairs` is created only lazily in shape_ops.py:700 and is in none of the baseline dicts — confirm intentional).

### S10. Two parallel transactional-workspace mechanisms plus two subclass-state deepcopy engines
- **Where**: `FitWorkspace`/`_copy_subclass_state` (fit_workspace.py:16-101) vs `FittedStateRevision` (fit_state.py:302-381) with `_publish_workspace_extension_state` (fit_state.py:566-614).
- **Why**: both implement "clone -> mutate privately -> validate -> dict-swap" with different cloning strategies and two near-identical alias-preserving deepcopy routines for subclass state (fit_workspace.py:44-66 vs fit_state.py:578-614). The duplication is subtle enough that behavioural drift between the fit path and the repair path is plausible (e.g., `_freeze_candidate_arrays` runs `auxiliary=True` for fresh fits but `freeze_auxiliary_arrays` is a flag on revisions).
- **Verify**: whether `FittedStateRevision.start(increment=True)` bumping both revision and repair_revision matches consumers' expectations; whether the two deepcopy engines produce identical alias graphs for a subclass with self-references.

### S11. Vestigial solver-to-public machinery: beta equality is enforced, so the "coefficient map" is always identity
- **Where**: fit_state.py:479 (`np.array_equal(beta, solver_beta)` hard requirement), runtime_canonicalize.py:30-88 (`_IdentityCoefficientMap`), 433-441 (`_solver_to_public_state`), 444-453 (`_build_public_result` copies beta unchanged).
- **Why**: the architecture carries a general solver->public *linear map* abstraction (with `__matmul__`, `rmatvec`, array materialisation) whose only possible value is identity-or-None under the publication validator. All real canonicalization is a scalar intercept shift + spec-side `_R_inv` centering. Either the map is future-proofing (then the validator contradicts it) or it is dead complexity.
- **Verify**: git history intent; consumers of `solver_to_public` (grep) — if only `runtime_canonical_state` readers check `solver_to_public_complete` as a boolean, the map object itself may be removable.

### S12. `update_reml_r_inv` O(G²) scans and string-parsing of group names
- **Where**: reml_state.py:32, 58, 67, 76 (`next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)` inside loops), 70 (`fg.name.split("[")[1].rstrip("]")` to recover a factor level from a display name).
- **Why**: quadratic in group count (matters for factor smooths with many levels, where G scales with levels), and level identity recovered by parsing the human-readable group name — fragile against any naming change; the level naming contract is implicit.
- **Verify**: whether GroupSlice carries a structured level field that could replace the parse; measure G for a realistic FactorSmooth (e.g., 500-level portfolio) and time this function.

### S13. `PathResult` immutability copies and path memory
- **Where**: fit_ops.py:90-95 (`_immutable_path_array`: `tobytes()` + `frombuffer` — 2 transient copies), 110-125 (applied to all 7 arrays incl. coef_path (L,p)).
- **Why**: for L=50, p=5000 that is ~2MB×3 live at once per array — modest but pure overhead for an immutability guarantee that `setflags(write=False)` mostly provides; the comment (93-95) explains the choice, so this is a deliberate trade — flag as measured cost only.
- **Verify**: confirm no user relies on `np.asarray(path.coef_path)` being writable-copyable cheaply.

### S14. Doc-code mismatches / stale comments
- api.py:152-156: "Larger values (e.g. ``1e-6``) converge faster" — 1e-6 **is** the default; the example presumably predates a default change.
- api.py:1527: "REML adapter methods (used by reml_optimizer)" — the optimizers no longer call back into the model (see S1/S2).
- api.py:462: the experimental-convergence warning tests the raw `convergence` parameter, not `resolved_convergence`; combined with the constructor warning (base.py:682-692) a user gets the warning at construction but *not* when relying on the constructor value per-fit — inconsistent but arguably intentional (warn once).
- fit_ops.py:1136 `durable_retain_fit_state` parameter of `_fit_reml_in_workspace` is only supplied by profile_ops (105-118, 348-361); the public `fit_reml` never passes it — undocumented coupling between profile publication and REML reporting-state retention.
- state_ops.py:417-421 docstring claims O(n + p³) summary; verify the `needs_factor_certification` fallback (204-219) which builds `grouped_weighted_factor` — that path is O(n·p_a²)-class and contradicts the docstring in the worst case.
- base.py:1022-1031: dead module-level `predict`.

### S15. Guard/digest overhead on every retained fit
- **Where**: fit_ops.py:589-604 (`FitDataGuard.capture` full y copy + column digest; `FitGeometryGuard.capture` digests y/w/offset and allocates an (n,) zeros offset when none was supplied), fit_data_guard.py:22-28 (blake2b over each vector), frame digest over all guard columns.
- **Why**: at n=10⁷ with many columns this is a non-trivial serial hashing pass executed inside every fit, purely to defend against caller mutation of retained inputs. Reasonable policy, but it belongs in any accounting of "why is fit() slower than the solver".
- **Verify**: profile share of `frame.digest` in total fit time at large n; consider whether `FitDataGuard` (identity-cache guard) is redundant with `FitGeometryGuard` (superset digests) — both are captured together (fit_ops.py:589-604) and both digest X.

### S16. `plot_ops.plot` vs `plot_ops.plot_data` duplicated term-classification (~70 lines)
- **Where**: plot_ops.py:66-111 vs 239-277 — byte-for-byte identical mode/ambiguity/unknown-term logic.
- **Why**: straightforward copy-paste divergence risk (error messages already differ only in function name).
- **Verify**: trivially extractable; check no behavioural differences hide in the two copies.

### S17. Protected-semantics observance (no violations found, for the record)
- fit vs fit_reml separation is enforced structurally (`resolve_selection_penalty_for_reml` base.py:997-1001 rejects selection intent; `_reject_random_effect_selection_fit` fit_ops.py:171 keeps structured terms out of fit/fit_path; `lambda_policy` rejected in fit at fit_ops.py:789-795).
- `discrete=True` shares the same `optimize_direct_reml` entry with an internal branch (reml/direct.py:108) and the same finalize path, which is the right shape to prevent silent drift — but the double-rebuild in S5 and the `curvature_source` defaulting (`"fisher" if model._discrete else classify_reml_curvature(...)`, reml_finalize.py:357-362) are the two places where discrete/exact behaviour intentionally forks inside this subsystem; any change there needs parity tests.
- `sample_weight` semantics: input_validation.py:93-98 gives Tweedie strict prior weights, others plain nonnegative weights — matches CLAUDE.md.
