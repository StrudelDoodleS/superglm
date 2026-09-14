# Subsystem report: reml-core

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ origin/master (f082e9b).
All paths below are relative to `src/superglm/` unless absolute. Notation: **n** = rows, **p** = total built columns,
**p_g** = width of one group block, **m** = smooth terms, **q** = number of penalty components (= number of lambdas;
q >= m because multi-penalty groups, e.g. tensors, contribute 2+ components per term).

Total subsystem size: 10,466 lines across 16 files. The four largest files (scop_efs 1851, discrete 1369,
penalty_algebra 1303, observed_geometry 1054) account for 53% of the subsystem.

---

## 1. MODULE MAP

### 1.1 `reml/__init__.py` (56 lines)
Pure re-export surface (`reml/__init__.py:24-56`). Exports optimizers, algebra, and `REMLResult`/`PenaltyCache`.
Docstring at `reml/__init__.py:21` says "import siblings directly, not through this __init__" yet the file re-exports
everything anyway; `superglm/__init__.py:95` imports `REMLResult` through it.

### 1.2 `reml/runner.py` (413 lines) — legacy Fellner–Schall fixed-point outer loop
- `_center_cached_direct_gram` (`runner.py:44-65`): validates the cached centered Gram for cheap iterations.
- `run_reml_once` (`runner.py:68-413`): a *dual-mode* FP loop. `use_direct=True` runs `fit_irls_direct` per iteration
  and re-inverts a cached centered Gram on "cheap" iterations (`runner.py:202-211` via
  `solvers.irls_direct._invert_xtwx_plus_penalty`); `use_direct=False` runs `fit_pirls` (BCD) and inverts via
  `inference.covariance._penalised_xtwx_inv_gram` (`runner.py:221-223`). FP lambda update `runner.py:249-274`
  (`lam = r_j / (inv_phi*quad + trace)`), Anderson(1) acceleration `runner.py:276-292`, DM rebuild on large steps
  `runner.py:333-343`, final refit `runner.py:354-389`.
- **Callers**: only `model/reml_ops.py:306` (`model_run_reml_once`) -> `model/api.py:1721` (`_run_reml_once`).
  Grep of `src/` shows **no production caller invokes `_run_reml_once`/`model_run_reml_once`** — the only reference
  outside the wrapper chain is `tests/test_import_compat.py`. This whole file is a compat shim (see Suspects S1).

### 1.3 `reml/direct.py` (981 lines) — exact-path damped Newton (Wood 2011)
- `optimize_direct_reml` (`direct.py:56-981`), single monolithic function:
  - `discrete=True` delegation to `optimize_discrete_reml_cached_w` at `direct.py:105-133` (protected-API boundary:
    the cached-W/fREML branch is explicitly marked as the BAM-style approximation boundary, `direct.py:106-107`).
  - Curvature classification `direct.py:142-149` (`classify_reml_curvature` -> "observed" for non-canonical pairs);
    structured backend decision `direct.py:150-158` (`resolve_structured_backend`).
  - Bootstrap: one `fit_irls_direct` at conservative lambdas (`direct.py:243-263`), then a hand-rolled FP step to
    initialise `rho` (`direct.py:322-354`) with a `select_snap` special case for `component_type == "selection"`
    (`direct.py:338-344`) — i.e. mgcv-style `select=True` double-penalty null-space components.
  - Newton loop `direct.py:364-944`: per outer iter — full PIRLS (`direct.py:387-410`), optional
    `build_observed_reml_geometry` + `observed_penalized_mode_score` (`direct.py:424-469`), objective
    (`direct.py:472-491`), gradient (`direct.py:571-579`), `reml_w_correction` (`direct.py:584-599`, skipped when
    `discrete` — dead condition here, see S8), Hessian (`direct.py:662-678`), active-set freeze (`direct.py:684-704`),
    modified-Newton eigendecomposition of the (q x q) Hessian (`direct.py:714-720`), Armijo step-halving line search
    where **every trial re-runs full PIRLS** (`direct.py:740-930`) plus observed geometry per trial
    (`direct.py:802-819`).
  - `latch_runtime_backend` closure (`direct.py:204-224`) pins the run to Gram after a structured-backend fallback.
- **Callers**: `model/reml_ops.py:158` (`model_optimize_direct_reml`) <- `model/reml_execute.py:287,331`
  (`optimize_reml_best`) <- `fit_reml()` orchestration.
- **Calls**: `solvers.irls_direct.fit_irls_direct`, `reml.objective`, `reml.gradient`, `reml.w_derivatives`,
  `reml.observed_geometry`, `reml.penalty_algebra`, `reml.scale`, `reml.convergence`, `solvers.structured`,
  `solvers.hessian_factor.as_hessian_factor`.

### 1.4 `reml/discrete.py` (1369 lines) — POI fREML (mgcv bam-style), the `discrete=True` path
- `_solve_cached_profiled_system` (`discrete.py:61-129`): one lambda trial in the intercept-profiled centered geometry.
  Equilibrated Cholesky first (`discrete.py:86-119`) with rcond certification against `SHARED_RANK_POLICY`, fallback
  `decompose_gram` (`discrete.py:122-126`). Returns beta, intercept, `log_det_H = log(sum W) + log|H_c|_+`, rank.
- `_shared_tensor_group_names` / `_shared_tensor_penalty_pairs` (`discrete.py:132-161`): detect 2-penalty discrete
  tensor blocks.
- `optimize_discrete_reml_cached_w` (`discrete.py:164-1369`): Performance-Oriented Iteration —
  - Bootstrap: DM rebuild at boot lambdas (`discrete.py:328-334`), `build_penalty_context` (`discrete.py:337-342`),
    full PIRLS (`discrete.py:356-380`), FP rho init with per-component diagnostics (`discrete.py:467-519`).
  - POI loop (`discrete.py:529-1187`): **one** PIRLS step (`max_iter=1`, `discrete.py:551-578`) refreshing W and the
    cached centered system (`cache_out`: `centered_XtWX`, `centered_rhs`, `mean_x`, `mean_z`, `sum_W`,
    `structured_system`, `discrete.py:592-602`); objective with tensor-pair closed forms (`discrete.py:606-628`);
    convergence decision **before** constructing the next Newton step (`discrete.py:719-751`); gradient/Hessian
    (`discrete.py:709-769`); tensor u/v reparametrised 2x2 pair Newton solve with asymmetric trust caps
    (`discrete.py:812-869`); line search: for tensor-surrogate mode a pure quadratic model (`discrete.py:901-915`)
    with one deferred full evaluation (`discrete.py:1014-1101`), otherwise cached O(p^3) re-solve + O(n) eta/deviance
    + full objective per trial (`discrete.py:917-1011`); **DM rebuild + penalty-context rebuild + beta remap + tensor
    summary rebuild every outer iteration** (`discrete.py:1152-1187`).
  - Final full IRLS refit at converged lambdas (`discrete.py:1189-1304`); the refit is authoritative
    (`discrete.py:1296-1301`), never the POI surrogates.
  - Ignored legacy kwargs `max_analytical_per_w`, `select_snap` are still accepted (`discrete.py:184-186`) — but
    `select_snap` **is** read at `discrete.py:504` (see S9: comment says ignored, code uses it).
- **Callers**: only `direct.py:108` (delegation) and `model/reml_ops.py:213` (`model_optimize_discrete_reml_cached_w`,
  which itself has no production caller past `model/api.py:1663` — a second compat wrapper).

### 1.5 `reml/efs.py` (426 lines) — generalized Fellner–Schall for the group-lasso (lambda1>0) BCD path
- `optimize_efs_reml` (`efs.py:40-426`): bootstrap PIRLS + one FP step (`efs.py:95-154`), DM rebuild
  (`efs.py:156-164`), main loop (`efs.py:187-381`): full `fit_pirls` or cheap re-inversion of cached
  `X'WX + S` via `_safe_decompose_H` (`efs.py:221-225`), FP update with +/-5 log-step clamp (`efs.py:236-269`),
  Anderson(1) (`efs.py:271-288`), **stale-basis uphill-step guard** that evaluates the objective twice at stale
  geometry (`efs.py:290-339`; the comment at `efs.py:290-299` honestly documents it as a heuristic), tiering
  (`efs.py:364-379`), final refit (`efs.py:383-404`).
- **Callers**: `model/reml_ops.py:257` (`model_optimize_efs_reml`) <- `model/reml_execute.py:310,354`.
- Note `efs.py:93`: the docstring cites "optimize_direct_reml lines 417-467", which no longer matches
  `direct.py` (bootstrap is now at `direct.py:226-354`) — stale line-number reference.

### 1.6 `reml/gradient.py` (370 lines) — d/drho and d^2/drho^2 of the LAML criterion
- `_penalty_block_trace` (`gradient.py:32-42`): declared helper for `tr(H^-1 dS_i H^-1 dS_j)` — **no callers anywhere**
  (grep: only its definition). Dead code (S6).
- `reml_direct_gradient` (`gradient.py:49-95`): `grad_i = 0.5*(lam_i*(inv_phi*quad_i + tr(H^-1 Omega_i)) - r_i)` with
  `r_i` from `compute_logdet_s_derivatives`. All traces go through the `HessianFactor` protocol
  (`solvers/hessian_factor.py:86`), so dense and structured backends share one code path.
- `reml_direct_hessian` (`gradient.py:98-370`): two trace strategies — a compact factor-trace path
  (`use_compact_trace`, `gradient.py:146`, taken whenever there is no dense `dH_extra`) and a dense path that
  materialises `F_i = H^{-1} dH_i` as full p x p per component (`gradient.py:172-183`), then pairwise
  `sum(F_i * F_j.T)` (`gradient.py:303`). Diagonal `g_i + 0.5 r_i` (`gradient.py:307-312`), shared-block
  log|S|+ curvature (`gradient.py:314-323`), implicit-beta term `-inv_phi * S_beta' H^-1 S_beta`
  (`gradient.py:325-331`), estimated-scale outer-product corrections (`gradient.py:333-364`), second-order W
  cross-term `dH2_cross` (`gradient.py:366-368`).
- **Callers**: `direct.py:571,662`, `discrete.py:709,754`; also exposed via `model/reml_ops.py:84,115` wrappers
  (compat-only, no production caller).

### 1.7 `reml/objective.py` (329 lines) — the single LAML criterion
- `REMLObjectiveEvaluation` (`objective.py:38-46`): value + profiled scale + penalty quad + nullity + penalized
  deviance (lets optimizers reuse phi and M_p).
- `reml_laml_objective` (`objective.py:49-326`): V(rho) = nll + 0.5*(beta'S beta + log|H| - log|S|_+), or the
  phi-profiled variant. Highly polymorphic inputs: it accepts precomputed `XtWX`, `XtW1/sum_W`, precomputed
  `log_det_H` + `hessian_rank`, `S_override`, `reml_penalties`, `scop_states`, `tensor_pair_evaluations`,
  precomputed Gaussian/Gamma scale data — 6 alternative geometry-supply modes with a fallback lattice:
  - Recompute W and moments from scratch if nothing supplied (`objective.py:113-122`, one full O(np^2) pass).
  - `log|S|_+`: components path -> `compute_logdet_s_plus` (`objective.py:186-191`); cache path ->
    `cached_logdet_s_plus` (`objective.py:192-193`); dense eigvalsh fallback (`objective.py:195-200`).
  - `log|H|`: precomputed (`objective.py:209-210`); SCOP joint assembly + `decompose_gram` (`objective.py:211-226`);
    centered Schur complement + `decompose_gram` (`objective.py:232-237`); historical slope-Gram fallback
    (`objective.py:238-245`).
  - Estimated scale: `compute_penalty_nullity` with a 4-level `hessian_rank` resolution chain
    (`objective.py:253-277`), Gaussian/Gamma closed-form profilers (`objective.py:279-296`), reduced criterion for
    Tweedie/other (`objective.py:297-301`).
  - Poisson shortcut `nll = deviance/2` (`objective.py:311-313`), otherwise a **fresh eta/mu data pass** for the
    log-likelihood (`objective.py:315-318`).
- **Callers**: every optimizer (`runner.py:398`, `direct.py:472,850`, `discrete.py:609,982,1075,1276`,
  `efs.py:301,315,412`, `scop_efs.py:646`), plus `model/reml_finalize.py:478,563,606` for terminal re-evaluation.

### 1.8 `reml/observed_geometry.py` (1054 lines) — observed-information LAML geometry
- `_deriv4_inverse` (`observed_geometry.py:112-168`) / `_variance_third_derivative`
  (`observed_geometry.py:171-190`): exact 4th link / 3rd variance derivatives via a builtin type-switch.
- `_compute_observed_row_bundle` (`observed_geometry.py:203-330`): observed rows w_i and their first/second eta
  derivatives; hand-optimised log-link closed forms for Gamma/Poisson/NB/Tweedie
  (`observed_geometry.py:219-261`), generic formula otherwise.
- `classify_reml_curvature` (`observed_geometry.py:431-449`) + `_builtin_fisher_equals_observed`
  (`observed_geometry.py:384-405`): fisher vs observed decision; custom families must declare a
  `reml_curvature` protocol. `classify_scop_reml_curvature` (`observed_geometry.py:497-522`) is a near-verbatim
  duplicate for SCOP with a different hook name (S5).
- `validate_observed_derivative_capability` (`observed_geometry.py:452-494`).
- `observed_penalized_mode_score` (`observed_geometry.py:692-797`): KKT residual of the penalized score with a
  careful pre-cancellation normalisation (`observed_geometry.py:777-787`). One O(np) pass.
- `_stable_signed_mean` (`observed_geometry.py:800-817`): compensated chunked mean for signed weights —
  materialises the design in 8192-row dense chunks via `row_subset(...).toarray()`.
- `build_observed_reml_geometry` (`observed_geometry.py:820-1054`): recompute eta/mu (O(np)), observed rows,
  then either a structured operator path (`observed_geometry.py:898-993`: `build_structured_system` +
  `build_augmented_structured_factor`, indefiniteness check via `eigvalsh` of the Schur core at
  `observed_geometry.py:950`) or dense: `build_centered_system` (O(np^2)) + `decompose_gram` (O(p^3)) +
  optional factor recertification (`observed_geometry.py:1028-1037`). Returns `ObservedREMLGeometry`.
- **Callers**: `direct.py:424,802` (candidate + every observed line-search trial), `model/reml_finalize.py:514`
  (terminal objective), `w_derivatives.py` (consumes the geometry), `scop_geometry.py:20`
  (`compute_scop_observed_information_weights`).

### 1.9 `reml/w_derivatives.py` (581 lines) — W(rho) correction
- `compute_dW_deta` (`w_derivatives.py:50-78`), `compute_d2W_deta2` (`w_derivatives.py:81-109`, analytic
  `:112-142`, FD fallback `:145-173`).
- `reml_w_correction` (`w_derivatives.py:176-574`): for each component j: `dbeta_j = -H^{-1} S_j beta` (IFT,
  `w_derivatives.py:437`), `deta_j = X_c dbeta_j` (`:442`), `a_j = dW/deta * deta_j` (`:445`),
  `C_j = X_c' diag(a_j) X_c` via `centered_signed_gram` (`:454`, an O(np^2) execution-plan moment call **per
  component**), gradient term `0.5 tr(H^-1 C_j) + 0.5 d(sum W)/sum W` (`:461-468`), and `dH_extra[j] = C_j` for the
  Hessian. `w_correction_order=2` adds the full q^2/2 second-order block (`w_derivatives.py:485-570`): per (i,j)
  pair one `centered_rmatvec` (O(np)), one factor solve (O(p^2)), one `centered_matvec` (O(np)) and one more
  `centered_signed_gram` (O(np^2)). The cost comment at `w_derivatives.py:481-484` acknowledges m^2/2 gram ops.
- Structured variants use `CenteredBlockOperator`/`SumBlockOperator`/`LowRankSymmetricOperator` instead of dense
  Grams (`w_derivatives.py:366-408`, `:536-553`).
- **Callers**: `direct.py:585` (exact path only), `model/reml_ops.py:30` (compat wrapper).

### 1.10 `reml/penalty_algebra.py` (1303 lines) — penalty eigenstructure & log|S|_+
- Compact-representation kernels: `penalty_component_dense_matrix` (`penalty_algebra.py:119-152`),
  `penalty_component_quadratic` (`:155-174`), `penalty_component_matvec` (`:177-196`),
  `penalty_component_trace` (`:199-265`), `total_penalty_quadratic` (`:268-290`), `total_penalty_matvec`
  (`:293-316`) — all dispatch on `penalty_kind` in {identity, sum_to_zero, repeated, dense}.
- Tensor closed forms: `_extract_tensor_marginal_eigvals` (`:319-343`, Kronecker-structure detection with a
  norm test), `_tensor_marginal_rank_logdet` (`:346-377`), `build_tensor_pair_logdet_summaries` (`:429-498`,
  cached by identity keys `_tensor_pair_summary_cache_key` `:410-426`), `evaluate_tensor_pair_logdet_summaries`
  (`:501-554`, O(p1*p2) per lambda evaluation, giving exact log|lam1 A(x)I + lam2 I(x)B|_+, its gradient and
  Hessian).
- `build_penalty_matrix` (`:557-631`): dense p x p S assembly; component-authoritative path plus a *second* legacy
  path when `reml_penalties is None` (`:605-631`) that only understands SSP-like group matrices and SCOP
  reparameterizations (S7).
- `build_penalty_components` (`:634-908`): the single source of penalty eigenstructure. Per group: 1-2 `eigh` of
  size p_g (raw rank + SSP logdet, `_rank_and_logdet` `:693-743`), PSD re-canonicalisation
  (`_canonicalize_ssp_penalty` `:661-691`), special-cased RandomEffect (identity kind, `:755-772`), FactorSmooth
  sz/repeated (`:774-838`), multi-penalty `omega_components` (`:839-877`), single-penalty (`:878-904`). Caching:
  only groups passing `_can_cache_penalty_group` (`:401-407`; unprojected discretized tensors) are cached in the
  optimizer-supplied dict.
- `coerce_reml_penalties` (`:911-959`): legacy `reml_groups` -> `PenaltyComponent` shim, called defensively at the
  top of nearly every function in the subsystem.
- `build_penalty_caches` (`:962-981`): back-compat wrapper — **zero callers** outside `__init__` re-export (S6).
- `build_penalty_context` (`:984-1001`): returns (components, caches, ranks) — three views of the same data.
- `cached_logdet_s_plus` (`:1004-1023`), `compute_total_penalty_rank` (`:1026-1061`),
  `_matrix_penalty_rank` (`:1064-1087`), `_structural_active_penalty_rank` (`:1090-1143`),
  `compute_penalty_nullity` (`:1146-1198`, Wood's M_p in the identified space),
  `compute_logdet_s_plus` (`:1209-1246`), `compute_logdet_s_derivatives` (`:1249-1303`).
- **Callers**: everything in the subsystem; externally `solvers/irls_direct.py:323`, `solvers/pirls.py:654`,
  `inference/metrics.py:363`, `model/reml_finalize.py:27-31`, `model/fit_ops.py:1239`, `profiling/nb.py:440`.

### 1.11 `reml/multi_penalty.py` (356 lines) — Wood (2011) Appendix B similarity transform
- `SimilarityTransformResult` (`multi_penalty.py:24-48`).
- `similarity_transform_logdet` (`:51-283`): recursive dominant/subdominant separation; explicitly a prototype —
  the comment at `:194-198` admits it forms `S_transformed` and eigendecomposes it instead of exploiting the
  recursion. It also always builds `S_pinv_plus`, `Q_plus`, `Q_zero`, and `E_sqrt` (an extra Cholesky/eigh,
  `:251-274`) even though grep shows **no consumer anywhere** of `E_sqrt`/`Q_plus`/`Q_zero`/`Q_full` (only
  `S_pinv_plus` is used, by the two derivative helpers in this same file) (S4).
- `logdet_s_gradient` (`:286-315`), `logdet_s_hessian` (`:318-356`): O(M q_g^2) / O(M^2 q_g^2) Frobenius traces.
- **Callers**: only `penalty_algebra.compute_logdet_s_plus`/`compute_logdet_s_derivatives`
  (`penalty_algebra.py:1221,1263`) for shared-block groups not covered by the tensor closed form.

### 1.12 `reml/convergence.py` (68 lines)
`project_reml_gradient` (`convergence.py:21-42`), `evaluate_reml_candidate` (`:45-68`) — the compound
score+objective criterion shared by `direct.py:627` and `discrete.py:729`. Cleanest module in the subsystem.

### 1.13 `reml/scale.py` (311 lines)
`ProfiledScaleTerm` (`scale.py:15-22`), `GammaScaleProfileData` (`:25-42`), `prepare_gamma_reml_scale_data`
(`:45-71`), `profile_gaussian_reml_scale` (`:74-113`, closed form), `profile_gamma_reml_scale` (`:220-302`,
Brent root-find on the shape score with asymptotic digamma expansions `:116-217`). O(n) once per fit for the
prep, O(1) per evaluation. Called from `objective.py:282-296`, `direct.py:291-306`, `discrete.py:430-444`.

### 1.14 `reml/result.py` (94 lines)
- `PenaltyCache` (`result.py:26-37`): 4-field subset of `PenaltyComponent`.
- `REMLResult` (`result.py:40-69`): 20 fields, 12 of which are SCOP-only diagnostics.
- `_map_beta_between_bases` (`result.py:72-94`): remaps beta across R_inv changes via `lstsq` per SSP group.

### 1.15 `reml/scop_geometry.py` (904 lines) — SCOP latent-coordinate geometry
- `SCOPJointGeometry` (`scop_geometry.py:55-67`), `SCOPInferenceInfo` (`:69-84`), `SCOPModeScore` (`:86-93`).
- `_joint_jacobian_diag` (`:108-127`): diag(exp(beta_eff)) for all SCOP slices.
- `scop_penalized_mode_score` (`:130-251`): latent KKT residual with Higham roundoff-bound zero classification
  (`:225-241`).
- `build_scop_postfit_inference` (`:355-500`): Pya–Wood covariance (expected Hessian) + EDF (Newton-block
  influence, `:471-487`), factor-space rank certification lattice (`_factor_certifier` `:276-312`,
  `_decompose_with_factor_certification` `:315-330`, `_decompose_on_certified_range` `:333-352`).
- `build_cached_scop_joint_geometry` (`:503-651`): Fisher cross-blocks + retained per-group Newton blocks,
  fallback to pure Fisher on indefiniteness (`:627-634`).
- `assemble_observed_scop_hessian` (`:698-766`), `build_observed_scop_joint_geometry` (`:769-904`): exact
  observed latent Hessian, with Fisher fallback and O(np^2) centered-system builds.
- **Callers**: only `scop_efs.py` and `solvers/irls_direct` (via `install_scop_postfit_inference`).

### 1.16 `reml/scop_efs.py` (1851 lines) — SCOP-aware EFS optimizer
- Module-level tuning constants with mixed units (`scop_efs.py:58-68`, self-documented).
- `_SCOPREMLFitContext` (`:71-90`), `_SCOPREMLMode` (`:92-116`).
- Multi-SCOP discrete cleanup machinery: `_multi_scop_discrete_cleanup_enabled/_names` (`:118-144`),
  `_update_multi_scop_discrete_stability_counts` (`:195-223`), `_freeze_multi_scop_discrete_lambdas`
  (`:226-245`), `_multi_scop_discrete_plateau_converged` (`:248-266`).
- `build_scop_penalty_components` (`:269-306`), `_merge_scop_penalty_components` (`:309-320`),
  `compute_scop_aware_penalty_quad` (`:323-374`, subtract-then-re-add beta-space quad correction),
  `assemble_joint_hessian` (`:377-484`, gamma->beta_eff cross-block scaling + intercept Schur).
- `_evaluate_scop_reml_mode` (`:536-685`): one coherent mode -> geometry -> objective.
- `_fit_scop_reml_mode` (`:748-964`): fit + evaluate + Newton-correction certification with up to 2 tighter-tol
  recursive retries (`:903-924`).
- `_finalize_scop_reml_mode` (`:1038-1094`) + `_hydrate_scop_terminal_rank_info` (`:978-1035`): terminal-only
  rank/EDF hydration (3 `decompose_gram`/factor certifications).
- `_backtrack_scop_efs_candidate` (`:1097-1172`): damped + reflected log-lambda trials, each a full inner fit.
- `scop_efs_lambda_update` (`:1185-1226`): explicitly deprecated (`:1195-1197`), retained only for
  `tests/test_scop_efs.py` (S6).
- `fit_fixed_scop_reml` (`:1229-1303`): fixed-lambda single-mode entry, called from `model/reml_execute.py:112`.
- `_joint_efs_lambda_step` (`:1306-1411`): rEDF/pSp log-scale EFS with adaptive per-name alpha and
  suppression detection.
- `optimize_scop_efs_reml` (`:1414-1851`): bootstrap mode + EFS loop + backtracking + managed cleanup +
  plateau/strict convergence + terminal finalisation. Called from `model/reml_execute.py:209`.

### 1.17 Orchestration (outside subsystem, for context)
`model/reml_execute.py:259-372` (`optimize_reml_best`) selects: `use_direct` -> `optimize_direct_reml`
(which internally forwards `model._discrete` to the POI path), else -> `optimize_efs_reml` (lambda1>0 / BCD).
SCOP terms route earlier through `reml_execute.py:109-209` to `fit_fixed_scop_reml` / `optimize_scop_efs_reml`.
`model/reml_finalize.py` re-evaluates the terminal objective and observed geometry after the optimizer returns.

---

## 2. DATA FLOW

### 2.1 Entry state
All optimizers receive: `dm` (DesignMatrix; group_matrices list, `dm.p = p`, `dm.n = n`), `y`, `sample_weight`,
`offset_arr` (all length n), `reml_groups: list[(group_index, GroupSlice)]`, `lambdas: dict[str,float]` (q entries),
`penalty_ranks: dict[str,float]`, optional prebuilt `reml_penalties: list[PenaltyComponent]` (q entries) and
`penalty_caches: dict[str, PenaltyCache]`, `estimated_names` policy set.

### 2.2 Exact direct path (`fit_reml`, `discrete=False`)
1. `coerce_reml_penalties` (`direct.py:98`) — pass-through when components prebuilt.
2. Bootstrap `fit_irls_direct` returns `(PIRLSResult, XtWX_S_inv (p x p or HessianFactor), XtWX (p x p))`
   (`direct.py:243`). FP init of `rho` (length q).
3. Per Newton iteration: dense `S_cand` (p x p) built unless structured (`direct.py:374-384`); full PIRLS with warm
   start (`direct.py:387`); if observed curvature: a **fresh full data pass** builds eta, mu, observed rows (n),
   centered Gram (p x p, O(np^2)) and its decomposition (`observed_geometry.py:878-1054`), replacing the Fisher
   `XtWX_S_inv` for all derivative work (`direct.py:445`); objective consumes `XtWX`+`log_det_H`; gradient consumes
   `beta` (p), the inverse factor, and per-component `Omega` (p_g x p_g); `reml_w_correction` materialises per
   component: `dbeta_j` (p), `deta_j` (n), `C_j` (p x p dense or compact operator) — so q dense p x p matrices live
   simultaneously in `dH_extra`; Hessian is (q x q); line search re-enters PIRLS + observed geometry per trial.
4. Best (lambdas, PIRLSResult) tracked; result = `REMLResult` (`direct.py:971-981`). Terminal re-evaluation happens
   later in `model/reml_finalize.py:478-616`.

### 2.3 Discrete/fREML path (`discrete=True`)
1. Bootstrap: `rebuild_design_matrix_with_lambdas` (new group matrices with lambda-dependent `R_inv`), fresh
   `build_penalty_context` (eigendecompositions), tensor-pair summaries (marginal eigenvalue vectors, O(p1)+O(p2)
   floats per tensor).
2. POI loop: one PIRLS step; the cached centered system {`centered_XtWX` (p x p), `centered_rhs` (p), `mean_x` (p),
   `mean_z`, `sum_W`} (or a structured `*StructuredSystem`) is the invariant against which all lambda trials are
   solved with **no data pass** — `_solve_cached_profiled_system` is one p x p Cholesky per trial. Each accepted or
   evaluated trial still pays O(n): eta = `dm.matvec(beta)` + deviance (`discrete.py:965-967`).
3. End of every outer iteration: full DM rebuild + `_map_beta_between_bases` (lstsq per SSP group) +
   `build_penalty_context` (eigh per non-cacheable group) + tensor summaries rebuild (`discrete.py:1152-1187`).
4. Final: one more rebuild + full PIRLS refit + objective; POI surrogate results are discarded
   (`discrete.py:1296-1301`).

### 2.4 EFS path (lambda1 > 0)
`fit_pirls` (BCD over group blocks) -> W (n) -> `cached_xtwx = dm.execution_plan.moments(W).gram` (p x p, O(np^2))
-> `H = cached_xtwx + S`, `H_inv` dense (p x p) -> per-component quad/trace FP update -> Anderson(1) -> stale-basis
objective guard (2 objective evaluations at stale geometry) -> tier decision (rebuild DM + full PIRLS vs cheap
re-inversion O(p^3)).

### 2.5 SCOP path
`_fit_scop_reml_mode` -> `fit_irls_direct(return_scop_state=True)` -> `scop_states` (mutable dicts holding
`beta_eff`, `gamma_eff`, `S_scop`, `H_scop_penalized` per group) -> joint latent Hessian (p x p) via
`assemble_joint_hessian`/`build_*_scop_joint_geometry` -> dense pseudo-inverse -> EFS rEDF/pSp lambda step ->
`_backtrack_scop_efs_candidate` (full inner fit per trial) -> terminal hydration (3 rank certifications + Pya–Wood
EDF influence matrix (p+1 x p+1)).

### 2.6 Materialised arrays (worst-case dense, per outer iteration)
| Array | Shape | Where |
|---|---|---|
| `S` (penalty) | p x p | `build_penalty_matrix`, rebuilt per candidate + per line-search trial (`direct.py:374,751`; `discrete.py:537,888`) |
| `XtWX`, `XtWX_S_inv` | p x p each | every `fit_irls_direct(return_xtwx=True)` |
| observed centered gram + hessian + inverse | 3 x (p x p) | `observed_geometry.py:995-1053` |
| `dH_extra` (W-correction) | q x (p x p) | `w_derivatives.py:470` |
| `full_HdHj` (dense Hessian branch) | q x (p x p) | `gradient.py:160,172-183` |
| POI cache (`centered_XtWX`) | p x p | `discrete.py:592` |
| SCOP joint Hessian + inverse + influence | 3 x (p x p) | `scop_geometry.py` |
| eta/mu/W/row bundles | O(n) each (up to 5 live) | all paths |

---

## 3. STATE OBJECTS

| Object | File:lines | Fields | Lifecycle | Overlap notes |
|---|---|---|---|---|
| `PenaltyComponent` (external, `types.py`) | referenced throughout | name, group_name/index/sl, omega_raw, omega_ssp, rank, log_det_omega_plus, eigvals_omega, component_type, lambda_policy, penalty_kind, repeat_count, block_width | built by `build_penalty_components`, rebuilt on every DM rebuild | authoritative penalty eigenstructure |
| `PenaltyCache` | `result.py:26-37` | omega_ssp, log_det_omega_plus, rank, eigvals_omega | built by `build_penalty_context` alongside components | **strict 4-field subset of PenaltyComponent**; `penalty_caches` dict + `penalty_ranks` dict + components = three parallel views of the same data, all threaded through optimizer signatures |
| `REMLResult` | `result.py:40-69` | 20 fields | returned by all 6 optimizers | 12 fields are SCOP-only; non-SCOP paths leave them None — result type doubles as a diagnostics grab-bag |
| `REMLObjectiveEvaluation` | `objective.py:38-46` | value, profiled_scale, penalty_quad, penalty_nullity, penalized_deviance | per objective call | both `direct.py:497-504` and `discrete.py:634-639` carry an isinstance fallback for "scalar-returning stubs" (test seam leaking into production control flow) |
| `REMLCandidateConvergence` | `convergence.py:12-19` | 4 fields | per candidate | clean |
| `TensorPairLogdetSummary` / `Evaluation` | `penalty_algebra.py:41-63` | marginal eigvals / logdet+grad+hess dicts | summary: per DM rebuild (cached); evaluation: per lambda vector | discrete-path only; the exact same shared-block need on the direct path is served by `similarity_transform_logdet` instead (duplicated capability, different algorithms) |
| `SimilarityTransformResult` | `multi_penalty.py:24-48` | logdet, S_pinv_plus, Q_plus, Q_zero, E_sqrt, rank | per shared-block logdet call, discarded | Q_plus/Q_zero/E_sqrt have zero consumers |
| `ObservedREMLGeometry` | `observed_geometry.py:655-671` | eta, mu, weights, 2 weight-derivative rows, sum_w, mean_x, centered gram+hessian, inverse, log_det_H, rank | per candidate/trial in observed mode, read-only frozen | atomic determinant/inverse/centering payload; deliberately supersedes the Fisher inverse (`w_derivatives.py:262-265`) |
| `ObservedModeScore` / `SCOPModeScore` | `observed_geometry.py:673-680` / `scop_geometry.py:86-93` | intercept, slopes, max_abs, relative_max | per trial | two structurally identical KKT-residual types with different normalisation logic |
| `SCOPJointGeometry` | `scop_geometry.py:55-67` | centered hessian, inverse, intercept cross, sum_w, logdet, rank, curvature_source, mean | per SCOP mode | SCOP analogue of ObservedREMLGeometry (parallel hierarchy) |
| `SCOPInferenceInfo` | `scop_geometry.py:69-84` | coefficient/augmented inverses, EDFs | terminal only | |
| `_SCOPREMLFitContext` / `_SCOPREMLMode` | `scop_efs.py:71-90` / `:92-116` | frozen contexts / one mode + its full geometry | per optimizer run / per candidate | `_SCOPREMLMode` is the pattern the non-SCOP paths lack: candidate state travels as one coherent object instead of 8 loose locals |
| `scop_states` | untyped `dict[int, dict]` | string keys: S_scop, beta_eff, gamma_eff, group_sl, group_name, H_scop_penalized, bin_idx, last_step_norm, last_fisher_fallback, penalty_rank/log_det/eigvals (memoised in-place at `scop_efs.py:147-181`) | produced by the inner solver, mutated by REML | stringly-typed shared mutable state crossing the solver/REML boundary |
| `ProfiledScaleTerm` / `GammaScaleProfileData` | `scale.py:15-42` | phi, inverse_phi, criterion, derivative / 2 sufficient stats | per evaluation / per fit | clean |
| `TabmatCenteringState` (external) | `direct.py:161` | centering plan cache | one per direct run | |

---

## 4. COMPLEXITY TABLE

All dense unless noted. "per iter" = per outer REML iteration. K = PIRLS iterations per solve, L = line-search
trials, w = width of a shared multi-penalty block (w = p_g).

| Routine | Time | Memory | Notes |
|---|---|---|---|
| `fit_irls_direct` full solve (`direct.py:387`) | O(K(np^2 + p^3)) | O(p^2 + n) | per candidate **and** per line-search trial on the exact path -> O((1+L)K np^2) per iter |
| `build_observed_reml_geometry` dense (`observed_geometry.py:995-1053`) | O(np^2 + p^3) | 3 p^2 | per candidate + per observed line-search trial (`direct.py:802`); trials skip the inverse (`compute_inverse=False`) but still pay the Gram + decomposition |
| `observed_penalized_mode_score` (`observed_geometry.py:692`) | O(np) | O(n+p) | per candidate + trial |
| `reml_laml_objective` with everything precomputed | O(p_g^2 sum + q) | O(p^2) if S built | Poisson known-scale short-circuit is O(1) beyond penalty quad |
| `reml_laml_objective` cold (no XtWX/log_det_H) | O(np^2 + p^3) | O(p^2) | hit by EFS uphill guard (2x per iter, `efs.py:301-328`) and legacy callers |
| non-Poisson known-scale nll inside objective (`objective.py:315-318`) | O(np) extra data pass | O(n) | recomputes eta/mu even when the caller just ran PIRLS at this beta |
| `build_penalty_matrix` (`penalty_algebra.py:557`) | O(sum p_g^2), zeros O(p^2) | new p x p per call | called 1 + L times per direct iter, 1 + L per discrete iter, ~3x per runner iter |
| `build_penalty_components` (`penalty_algebra.py:634`) | 2 eigh per group: O(sum p_g^3) | O(sum p_g^2) | **inside the discrete outer loop** (rebuilt each iteration, `discrete.py:1173`) and each EFS full tier (`efs.py:372`); only unprojected discrete tensors are cache-eligible (`penalty_algebra.py:401-407`) |
| `similarity_transform_logdet` (`multi_penalty.py:51`) | ~5 eigh + 1 chol of w x w: O(w^3) | O(M w^2) | per shared-block group per objective/gradient/Hessian evaluation on paths without tensor summaries (direct/EFS/SCOP); direct-path line search re-runs it per trial |
| `evaluate_tensor_pair_logdet_summaries` (`penalty_algebra.py:501`) | O(p1 p2) | O(p1 p2) | closed-form replacement, discrete path only |
| `reml_direct_gradient` (`gradient.py:49`) | O(sum p_g^2) traces via factor | O(q) | cheap |
| `reml_direct_hessian` compact path (`gradient.py:195-301`) | O(q^2) cross-traces, each O(p_g^3) worst | O(p_g^2) cached blocks | same-slice memoisation `:200-232` |
| `reml_direct_hessian` dense path (`gradient.py:172-183,303`) | O(q p^2 p_g) build + O(q^2 p^2) traces | **O(q p^2)** (`full_HdHj`) | taken whenever dense `dH_extra` present, i.e. exact path with W-correction on dense backends |
| `reml_w_correction` order 1 (`w_derivatives.py:427-477`) | per component: O(p^2) solve + O(np) matvecs + **O(np^2) signed gram** -> O(q np^2) per iter | q dense p x p in `dH_extra` | this is the dominant exact-path cost after PIRLS for moderate q |
| `reml_w_correction` order 2 (`w_derivatives.py:485-570`) | O(q^2 (np^2 + p^2)) | + (q x q) | opt-in |
| `_solve_cached_profiled_system` (`discrete.py:61`) | O(p^3) chol (+ pocon) | O(p^2) equilibrated copy | per line-search full trial; no data pass |
| discrete trial deviance (`discrete.py:965-967`) | O(np) | O(n) | per full trial |
| `rebuild_design_matrix_with_lambdas` + `build_penalty_context` + `_map_beta_between_bases` (`discrete.py:1152-1187`) | DM rebuild (basis-dependent, up to O(np_g) per spline) + eigh per group + lstsq O(p_g^3) per SSP group | new group matrices | **every** discrete outer iteration, even when lambda moved little |
| `_penalised_xtwx_inv_gram` (runner BCD, `runner.py:221`) | O(np^2 + p^3) | O(p^2) | per iter |
| EFS cheap iteration (`efs.py:221-225`) | O(p^3) | O(p^2) | no data pass |
| EFS uphill guard (`efs.py:301-328`) | 2 objective evals; log|S|_+ dominates | — | both at stale geometry |
| SCOP `_fit_scop_reml_mode` (`scop_efs.py:748`) | inner fit O(K np^2) + joint geometry O(np^2 + p^3) + mode score O(np) + Newton-relative O(p^2) | several p x p | per candidate + per backtrack trial (up to 8 + 4 reflected, `scop_efs.py:67-68`) |
| `build_scop_postfit_inference` (`scop_geometry.py:355`) | 2-3 decompositions O(p^3) + influence O(p^3) (`diagonal_of_square`) | 3-4 p x p | terminal only |
| `_hydrate_scop_terminal_rank_info` (`scop_efs.py:978`) | 3 x decompose_gram O(p^3) (+ lazy factor passes O(np)) | p x p each | terminal only |
| `_stable_signed_mean` (`observed_geometry.py:800`) | O(np) with dense 8192 x p chunks | O(8192 p) | rare (signed observed rows) |

---

## 5. SUSPECTS

**S1 — `runner.run_reml_once` is a production-dead third optimizer (413 lines).**
`runner.py:68-413` reimplements the Fellner–Schall FP loop *including* Anderson(1) acceleration, the cheap-iteration
tiering, phi estimation, and DM rebuild — nearly all of which also exists in `efs.py:187-381`. Its only reachable
call chain is `model/reml_ops.py:306` -> `model/api.py:1721` (`_run_reml_once`), and grep shows no call to
`_run_reml_once` anywhere in `src/`; the only repo reference is `tests/test_import_compat.py`. Verify: confirm no
dynamic dispatch (`getattr`) reaches `_run_reml_once`; if so this is ~400 lines of duplicated FP algebra kept alive
purely for an import-compat test. Same status for the wrapper `model_optimize_discrete_reml_cached_w`
(`model/reml_ops.py:193-235` / `api.py:1663`) — the production route to the discrete optimizer is exclusively the
`optimize_direct_reml` delegation at `direct.py:105-133`.

**S2 — Bootstrap + FP-init + select-snap block duplicated verbatim between direct and discrete.**
`direct.py:226-354` and `discrete.py:322-519` contain the same ~90-line sequence (conservative interaction boot
lambdas, boot fit, boot phi, fixed-lambda restoration dict, per-component FP rho init, select-snap, interaction
log-step cap of 4.0), with the discrete copy adding diagnostics. The `latch_runtime_backend` closure is likewise
duplicated (`direct.py:204-224` vs `discrete.py:297-320`), as is the whole "objective evaluation -> phi resolution"
block (`direct.py:493-537` vs `discrete.py:630-673`) and the modified-Newton eigen-flip step
(`direct.py:706-729` vs `discrete.py:785-810`). Any fix to one (e.g. the select-snap threshold) must be manually
mirrored. Verify by diffing the blocks; behavioural drift risk is the concern, not just line count.

**S3 — Shared-block log|S|_+ computed by two unrelated algorithms depending on path.**
The discrete path uses the exact O(p1 p2) tensor closed form (`penalty_algebra.py:429-554`), threaded as
`tensor_pair_evaluations` through objective/gradient/Hessian. The exact direct path never builds tensor summaries
(no `build_tensor_pair_logdet_summaries` call in `direct.py`), so the same shared tensor block falls into
`similarity_transform_logdet` (`multi_penalty.py:51`) — the "prototype" Appendix-B transform — **once per objective,
once per gradient, once per Hessian, and once per line-search trial objective** per iteration (call sites
`objective.py:187`, `gradient.py:75`, `gradient.py:154` via `penalty_algebra.py:1244,1294`). Besides the redundant
O(w^3) eigh chains, this is a subtle numerical-consistency risk: exact vs discrete paths can disagree on log|S|_+
for identical lambdas by the two algorithms' different rank thresholds (`_EPS**(2/3)` final cut at
`multi_penalty.py:207` vs the same threshold applied to a different spectrum in
`evaluate_tensor_pair_logdet_summaries` `penalty_algebra.py:507-516`). Verify with a 2-penalty tensor model fitted
both ways.

**S4 — `similarity_transform_logdet` computes never-consumed outputs on every call.**
`multi_penalty.py:215-274` builds `Q_plus`, `Q_zero`, `S_pinv_plus`, and `E_sqrt` (an extra eigendecomposition-sorted
split plus a preconditioned Cholesky). Grep across `src/` and `tests/` finds zero consumers of `E_sqrt`, `Q_plus`,
`Q_zero`, `Q_full`; only `S_pinv_plus` is used, and only by `logdet_s_gradient`/`logdet_s_hessian` in the same file.
The objective-only call path (`compute_logdet_s_plus` -> `penalty_algebra.py:1244`) needs just the scalar logdet but
pays for the full result. Also `multi_penalty.py:194-198` self-documents the recursion as not actually being used
for the determinant ("prototype"). Verify: check test suite doesn't import these fields before concluding.

**S5 — Parallel near-duplicate curvature-classification and mode-score stacks for ordinary vs SCOP.**
`classify_reml_curvature` (`observed_geometry.py:431-449`) and `classify_scop_reml_curvature`
(`observed_geometry.py:497-522`) are the same function modulo the protocol hook name;
`observed_penalized_mode_score` (`observed_geometry.py:692-797`) and `scop_penalized_mode_score`
(`scop_geometry.py:130-251`) compute the same KKT residual with divergent normalisation/roundoff policies
(pre-cancellation bound vs Higham gamma_k bounds); `ObservedREMLGeometry` vs `SCOPJointGeometry` and
`reml_w_correction`'s centered machinery vs `scop_geometry`'s centered machinery form two parallel geometry
hierarchies. Responsibility boundary between `observed_geometry.py` and `scop_geometry.py` is by-family rather than
by-function. Verify which normalisation policy is the intended one — they will accept/reject the same mode
differently near boundaries.

**S6 — Dead/compat code inside the subsystem.**
(a) `gradient._penalty_block_trace` (`gradient.py:32-42`): zero callers.
(b) `penalty_algebra.build_penalty_caches` (`penalty_algebra.py:962-981`): zero callers beyond the `__init__`
re-export; its docstring says "retained for backward compatibility".
(c) `scop_efs.scop_efs_lambda_update` (`scop_efs.py:1185-1226`): explicitly deprecated, only `tests/test_scop_efs.py`
uses it.
(d) `coerce_reml_penalties` (`penalty_algebra.py:911-959`) is called defensively at the top of 6+ functions even
though all production optimizers now receive prebuilt `reml_penalties` — the legacy `reml_groups` branch builds
components *without* rank/logdet (rank=0.0), which silently degrades gradient `r_j` to `penalty_ranks` fallback
(`gradient.py:92-93`); verify whether any production caller still hits the legacy branch.
(e) `discrete.py:184-186` accepts "legacy kwargs accepted but ignored" — see S9.
(f) `objective`/`isinstance(REMLObjectiveEvaluation)` fallbacks for "scalar-returning stubs" (`direct.py:502-504`,
`discrete.py:638-639`) keep a test seam in production control flow.

**S7 — `build_penalty_matrix` has two disagreeing assembly semantics.**
Component path (`penalty_algebra.py:576-603`) **adds** (`+=`) per-component blocks and covers identity /
repeated / sum_to_zero / dense kinds plus a SCOP fallback for unrepresented groups. Legacy path
(`penalty_algebra.py:605-631`, taken when `reml_penalties is None`) **assigns** (`=`), covers only SSP-like group
matrices and SCOP reparameterizations, and silently ignores RandomEffect/FactorSmooth/multi-penalty groups. Callers
still on the legacy path: `solvers/pirls.py:656` (`build_penalty_matrix(gms, groups, lambda2, p)` with no
components). If the BCD path is ever run with random-effect or factor-smooth penalised groups, S would silently drop
those penalties. Verify what group types can reach `fit_pirls` with `lambda2` dict and no components.

**S8 — `direct.py` retains a `discrete` conditional after the discrete early-return.**
`direct.py:584-601`: `if not discrete: w_corr = reml_w_correction(...) else: w_corr = None` — but `discrete=True`
already returned at `direct.py:105-133`, so the else branch is unreachable. Harmless, but it misleads readers into
thinking the exact Newton loop can run in discrete mode (doc-code mismatch with the delegation contract).

**S9 — `discrete.py` claims `select_snap` is ignored but uses it.**
Signature comment `discrete.py:184-186` says "Legacy kwargs accepted but ignored (removed in POI rewrite)" for
`max_analytical_per_w, select_snap`, yet `select_snap` gates the null-space snap at `discrete.py:503-509`.
`max_analytical_per_w` is genuinely unused. Since `select=True` double-penalty behaviour is protected API, the
comment (not the behaviour) is the bug — but it invites someone to delete a live parameter. Verify with a
`select=True, discrete=True` fit.

**S10 — O(np^2)-class work inside the exact-path optimizer loop.**
Per exact Newton iteration with observed curvature: (a) full PIRLS O(K np^2); (b) `build_observed_reml_geometry`
O(np^2 + p^3) — a *second* full-data Gram at the same beta PIRLS just converged, because the observed rows differ
from Fisher rows; (c) `reml_w_correction` performs **one O(np^2) centered signed Gram per penalty component**
(`w_derivatives.py:454`), i.e. O(q np^2) per iteration, and with `w_correction_order=2` O(q^2 np^2); (d) each
line-search trial repeats (a)+(b). For insurance-scale n with several smooths this makes gradient/Hessian
preparation cost a multiple of the fit itself. The discrete path avoids all of this by design; the exact path has no
row-subsampling or reuse of the PIRLS-final weighted moments for the W-correction Grams (the moments differ only by
`dW/deta * deta_j` reweighting, which is inherently a new Gram — but the q Grams could be one blocked pass). Verify
against the profile counters `reml_w_correction_s` / `reml_observed_geometry_s` (`direct.py:958-959`) on a large fit.

**S11 — Discrete path rebuilds design + penalty eigenstructure every outer iteration unconditionally.**
`discrete.py:1152-1187` and again `:1195-1228` (final): `rebuild_design_matrix_with_lambdas` +
`build_penalty_context` (per-group eigh) + `_map_beta_between_bases` (per-group lstsq) + tensor summaries run even
when the accepted lambda step was tiny or the line search was rejected (rho unchanged -> `current_lambdas` equal to
the previous rebuild's lambdas). The EFS path has exactly the threshold this needs (`cheap_threshold=0.01`,
`efs.py:185,365`); the POI loop has none. `_penalty_group_cache_key` caches only unprojected discretized tensors
(`penalty_algebra.py:401-407`), so ordinary spline groups pay 2 eigh per group per iteration. Verify with the
profile counters `reml_rebuild_dm_s` / `reml_penalty_context_s` (`discrete.py:1347-1349`).

**S12 — `_solve_cached_profiled_system` duplicates `decompose_gram`'s stabilised solve.**
`discrete.py:79-126` hand-rolls equilibration + Cholesky + pocon + probe-residual certification and then falls back
to `decompose_gram`, which itself implements the shared rank policy. Two rank/solve policies for the same matrix in
one function; the fast path's acceptance thresholds (`probe_residual <= 1e-6`, `certification_band * gram_rcond`)
are local constants rather than `SHARED_RANK_POLICY` values (only partially shared at `discrete.py:100`). Verify
whether `decompose_gram` already has an equivalently cheap full-rank fast path; if so the local one is redundant.

**S13 — `REMLObjectiveEvaluation`-vs-scalar and phi-resolution lattice repeated in four places.**
The block "resolve phi/inverse_phi/penalty_nullity from evaluation or recompute from S + hessian_rank" appears in
`direct.py:493-537`, `discrete.py:630-673`, plus bootstrap variants `direct.py:269-312` and `discrete.py:407-451`,
and again in `runner.py:225-247`. Each copy has its own fallback ordering; e.g. runner uses
`compute_total_penalty_rank` when not direct while the others use `compute_penalty_nullity` with hessian rank —
meaning the BCD FP update in runner and EFS uses M_p = total penalty rank while the direct paths use the
identified-space nullity. That is a real (if small) criterion difference between paths for estimated-scale
families. Verify: `efs.py:229-233` (`compute_total_penalty_rank`) vs `objective.py:261-277`.

**S14 — Runner cheap-iteration validator ignores its own inputs.**
`_center_cached_direct_gram` (`runner.py:44-65`) is named/documented as producing the centered Gram but merely
validates and returns `centered_XtWX` unchanged; `mean_x`/`sum_w` are loaded and range-checked but unused for any
computation. Combined with S1 (dead path) this is vestigial. Low priority, but confirms runner.py has drifted.

**S15 — Untyped mutable `scop_states` dicts cross the solver/REML boundary and are memoised in place.**
`_get_scop_penalty_metadata` (`scop_efs.py:147-181`) writes `penalty_rank`/`penalty_log_det_omega_plus`/
`penalty_eigvals_omega` back into the solver-owned dict; `assemble_joint_hessian` (`scop_efs.py:377-484`) and three
geometry builders all read half a dozen string keys with per-call revalidation. Shape errors surface as runtime
ValueErrors deep in geometry code rather than at the boundary. This is the least-typed state in the subsystem while
carrying the most intricate math (latent-vs-mapped coordinates).

**S16 — Objective recomputes an O(np) data pass for non-Poisson known-scale families even when the caller has one.**
`objective.py:315-318`: when `mu is None` (always true when `XtWX` was supplied by the optimizer), Binomial/known
Gamma/etc. pay a fresh `dm.matvec(beta)` + inverse-link per objective evaluation — including every line-search trial
on both direct (`direct.py:850`) and discrete (`discrete.py:982`) paths. PIRLS just computed this eta at the same
beta. Poisson alone has the deviance shortcut (`objective.py:311-313`). Verify magnitude on a Binomial fit profile.

**S17 — Stale doc references.**
`efs.py:93` cites "optimize_direct_reml lines 417-467" (now ~`direct.py:226-354`). `runner.py` module docstring
(`runner.py:1-17`) claims to contain "gradient, Hessian, W(rho) correction, Laplace objective, and the three
optimizer kernels" — those were long since extracted to sibling modules; runner contains only the FP kernel.
`reml/__init__.py:21` instructs "import siblings directly, not through this __init__" while exporting everything.

---

## 6. Cross-cutting summary of decomposition traffic (where eigen/Cholesky happen)

Per outer iteration, worst-case dense exact path with observed curvature and shared-block penalties:
- PIRLS: K Cholesky/decompose_gram of (p x p) (inside `fit_irls_direct`).
- Observed geometry: 1 `decompose_gram` (p x p) per candidate + 1 per trial (`observed_geometry.py:1022`).
- Objective: `similarity_transform_logdet` ~5 eigh (w x w) per shared block (`multi_penalty.py:90,147,167,206,223,261`).
- Gradient + Hessian: same transform again via `compute_logdet_s_derivatives` (`gradient.py:75,154`); (q x q) eigh
  for the modified Newton step (`direct.py:714`).
- W-correction: q factor solves (backsubstitution against the retained decomposition).
Discrete path replaces per-trial PIRLS with one (p x p) Cholesky (`discrete.py:89`) but adds per-iteration
`build_penalty_context` eigh (2 per group) and, at the end, a final full PIRLS + context rebuild.
What is cached between lambda steps: warm beta/intercept (all paths), the POI centered system (discrete, within one
outer iteration only), `cached_xtwx` (EFS, until a full tier), `penalty_context_cache` (discrete: tensor components
and pair summaries only), `TabmatCenteringState` (direct observed path), Anderson(1) state (runner/EFS), adaptive
alpha + prev_dlsp (SCOP). Not cached anywhere: non-tensor group eigenstructure across DM rebuilds, S assembly, and
eta at the current beta across PIRLS -> objective boundaries.
