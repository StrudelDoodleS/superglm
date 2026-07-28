# Subsystem report: penalties-constraints

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` at origin/master (f082e9b).
Scope: `src/superglm/penalties/*.py`, `src/superglm/constraints.py`, `src/superglm/features/_spline_select.py`, `src/superglm/features/_spline_multi_penalty.py`, plus the direct call/consumer surface (PIRLS penalty dispatch, SSP reparametrisation in `dm_builder.py`, shape-repair orchestration in `model/shape_ops.py`, fit-time constraint machinery in `solvers/constrained_qp.py` / `solvers/scop.py`).

Conventions: n = rows, p = total built solver columns, p_g = columns of group g, m = number of smooth terms / groups, q = number of penalty components (smoothing parameters), K = raw basis dimension of one spline, C = number of shape-constraint probe points, G = grid size for post-fit repair (default 500).

---

## 1. MODULE MAP

### 1.1 `src/superglm/penalties/base.py` (154 lines)

Responsibility: the `Penalty` and `Flavor` protocols plus small policy helpers shared by the model layer and solvers.

| Symbol | Lines | Role |
|---|---|---|
| `normalize_penalty_features` | 13-30 | normalises `features=` filter (str/iterable → frozenset) |
| `penalty_targets_group` | 33-40 | whether the lambda1 penalty applies to a `GroupSlice` (checks `group.penalized`, then `feature_name`/`name` membership) |
| `validate_penalty_features` | 43-58 | post-DM-build check that every filter name exists |
| `penalty_has_targets` | 61-63 | any-group version |
| `penalty_can_zero_groups` | 66-81 | selection semantics: Ridge never zeroes, GroupElasticNet zeroes iff `alpha>0`, unknown penalties default to "can zero" |
| `Penalty` protocol | 84-127 | `lambda1`, `flavor`, `features`, `prox`, `prox_group`, `eval` |
| `Flavor` protocol | 129-154 | `adjust_weights(groups, beta_init, group_matrices)` |

Called from:
- `solvers/pirls.py:29` (`Penalty`, `penalty_can_zero_groups`, `penalty_targets_group`) — the solver hot path.
- `model/base.py:34-43` (resolution + validation: `resolve_penalty` base.py:514-548, `normalize_selection_penalty` 551-567, `compute_lambda_max` 958-971, `resolve_selection_penalty_for_fit` 974-984, `validate_selection_penalty_for_reml` 987-1001, `model_has_lambda1_targets` 1004-1006, `validate_penalty_features` call at model/base.py:953).
- `solvers/rank.py:885` (`selected_group_name_set` legacy fallback), `profiling/nb.py:37` and `profiling/tweedie.py:54` (`penalty_has_targets`), `sklearn.py:47` / `model/api.py:16` (typing).

Note the inverted dependency: `base.py:74-75` imports the concrete `GroupElasticNet` and `Ridge` classes inside `penalty_can_zero_groups`, so the "base" module knows about its leaves (lazy import avoids a cycle, but the type-switch policy lives in the wrong layer).

### 1.2 `src/superglm/penalties/flavors.py` (68 lines)

`Adaptive` (17-68): the only Flavor. `adjust_weights` (41-68) computes per-group weight `sqrt(p_g)/(norm_g+eps)^expon`; with `group_matrices` it uses RMS fitted-value norms `||X_g beta_g||/sqrt(n)` (scale-invariant under SSP). Shallow-copies each `GroupSlice` (`copy.copy`, 65) and mutates `weight`.
Called only from `solvers/pirls.py:1469-1472` (two-stage fit) — nowhere else in the library.

### 1.3 `src/superglm/penalties/ridge.py` (57), `group_lasso.py` (63), `group_elastic_net.py` (82), `sparse_group_lasso.py` (76)

All four share an identical shape: constructor storing `lambda1` (float | "auto" | None), optional `flavor`, `features`; `prox_group` (per-block closed form); `prox` (loop over groups, copies beta); `eval`.

- `Ridge.prox_group` ridge.py:37-41: `bg / (1 + step*lambda1)`; no flavor support (ridge.py:34).
- `GroupLasso.prox_group` group_lasso.py:40-48: block soft-threshold at `step*lambda1*group.weight`.
- `GroupElasticNet.prox_group` group_elastic_net.py:52-65: soft-threshold then ridge denominator (correct composite prox; the comment at 56-59 documents why the order matters).
- `SparseGroupLasso.prox_group` sparse_group_lasso.py:44-59: elementwise L1 soft-threshold then group L2 threshold (exact SGL prox by decomposition).

Note the alpha conventions are opposite: GroupElasticNet `alpha=1` → pure group lasso (group_elastic_net.py:26), SparseGroupLasso `alpha=1` → pure L1 (sparse_group_lasso.py:27). Documented, but easy to misuse.

Registered in `model/base.py` `_PENALTY_SHORTCUTS` (imports at 40-43) and re-exported from `superglm/__init__.py:77-81` and `penalties/__init__.py`.

### 1.4 `src/superglm/features/_spline_select.py` (164 lines) — select=True machinery

| Symbol | Lines | Role |
|---|---|---|
| `eigendecompose_select` | 14-45 | `eigh(omega_c)`; requires exactly 2 null eigenvalues (`< 1e-10` absolute, line 25-31); removes the constant direction from the null space (36-41), keeping a 1-D `U_null` plus `U_range`/`omega_range`; maps back through constraint projection Z (43-44) |
| `resolve_lambda_policies` | 48-71 | `LambdaPolicy | dict → per-component dict`, validated against `penalty_components` suffixes |
| `build_select_group_info` | 74-125 | assembles the double-penalty `GroupInfo`: components `[("null", I on null coord), ("wiggle", omega_range)]` or per-order `d{order}` components via `U_combined_c.T @ omega_c_j @ U_combined_c` (110); `penalty_matrix = sum(components)` (113); `component_types={"null": "selection"}` (100); `projection=U_combined` |
| `build_select` | 128-156 | exact-path driver: builds penalty (max order if multi-m), applies constraints, computes `_interaction_projection`, eigendecomposes, delegates to `build_select_group_info` |

Callers: `features/spline.py:312-327` (`_SplineBase._eigendecompose_select`, `_resolve_lambda_policies`, `_build_select`), `features/_spline_build.py:76` (exact build), `features/_spline_build.py:152-155` (discrete `build_knots_and_penalty`), `dm_builder.py:872` (discrete lambda-policy resolution). `features/factor_smooth.py:214` has its own separate `_resolve_lambda_policies`.

### 1.5 `src/superglm/features/_spline_multi_penalty.py` (31 lines)

`build_multi_m_components` (11-28): for each order in `m_orders`, rebuilds the raw penalty, re-applies constraints, re-applies identifiability, yielding `("d{order}", omega_c)` components. Called from `features/spline.py:254-265`, used by `_spline_build.build_group_info:101` and `build_knots_and_penalty:162`.

### 1.6 `src/superglm/constraints.py` (659 lines) — post-fit shape repair + certificates

| Symbol | Lines | Role |
|---|---|---|
| `MonotoneRepairResult` | 16-33 | dataclass; per-feature repair record (grid, before/after curves, repaired full beta, violations) |
| `ShapeConstraintCertificate` | 36-48 | frozen dataclass; span-wise minimum signed derivative + scaled slack |
| `_shape_order_and_sign` | 51-56 | kind → (derivative order, sign) |
| `_raw_shape_coefficients` | 59-64 | solver beta → raw B-spline coefficients via `spec._R_inv`; hard-errors on SCOP coordinates (62) |
| `_shape_polynomial` | 67-86 | exact PPoly/CubicSpline of the fitted curve |
| `_shape_breakpoints`/`_uses_span_jump_constraints`/`_shape_probe_points` | 89-110 | probe-point selection, incl. degree-0/1 jump special cases |
| `shape_derivative_matrix` | 113-138 | (C, p_g) exact derivative rows of the fitted basis (`raw @ R_inv`) |
| `_shape_span_jump_matrix` | 141-177 | one-sided derivative jumps for degree-0/1 bases (per-point Python loop) |
| `_shape_constraint_rows` | 180-191 | dispatch of the above |
| `_certificate_candidates` | 194-230 | all within-span extrema: polynomial stationary points + breakpoints + one-sided limits |
| `_normalized_nonzero_shape_rows` | 233-269 | two-stage row normalization avoiding underflow; local (per-row) zero classification |
| `shape_constraint_certificate` | 272-303 | span-wise certificate for a fitted beta |
| `_violating_shape_constraint_points` | 306-320 | candidates with scaled slack < -tol |
| `shape_constraint_is_roundoff_feasible` / `_shape_roundoff_tolerance` | 323-340 | roundoff-scale feasibility test |
| `MonotoneRepairer` | 343-430 | projection of the fitted curve onto the monotone cone in fitted-basis coordinates |
| `monotonicity_violation` / `curvature_violation` | 433-451 | grid-difference violation metrics (test-only consumers) |
| `_project_shape_in_fitted_basis` | 454-583 | weighted LS projection with SLSQP + cutting-plane refinement of constraint points; feasible-zero objective certificate (576-581) |
| `CurvatureRepairer` | 586-652 | convex/concave analogue of MonotoneRepairer (repair() is a near-clone of MonotoneRepairer.repair) |
| `derivative_grid_matrix` | 655-659 | raises NotImplementedError ("reserved for future constrained IRLS") |

Callers: `superglm/__init__.py:27` (public export of `MonotoneRepairer`, `MonotoneRepairResult`); `model/shape_ops.py:37-42` (`_repairer`), `:434-436` (`shape_constraint_certificate`), `:533-537` (`shape_constraint_is_roundoff_feasible`). `model/monotone_ops.py` is a thin compat wrapper (`monotonize` → `apply_shape_postfit`); `model/api.py:1361-1418` exposes `monotonize`, `apply_shape_postfit`, and a further compat alias.

### 1.7 Consumer surface traced (context, not owned by this subsystem)

- **PIRLS penalty dispatch** `solvers/pirls.py`: type-switch on the penalty at 837-862 — GroupLasso/GroupElasticNet get exact per-block eigensystem solves (`_radial_block_eigensystems` 414-440, `_solve_radial_block` 443-476 with brentq secular equation); Ridge gets exact factored solves (`_ridge_block_factors` 398-411); everything else (SparseGroupLasso, custom penalties) falls back to a proximal-gradient step through `prox_group` (936-938). Zero-group KKT probe via `prox_group` at 950-963; convergence-time KKT check `_composite_kkt_violation` 479-538; inference curvature `_add_selection_local_curvature` 541-592 (again a `type(penalty)` switch limited to the three built-ins); Breheny-Huang group-lasso EDF fallback 1334-1346; two-stage flavor fit `fit_pirls` 1443-1497.
- **Solver selection policy** `model/fit_ops.py:660-715`: constraints/SCOP or lambda1∈{None,0}/no-targets → `fit_irls_direct`; otherwise `fit_pirls`. Guards: monotone+selection_penalty rejected (820-831), SCOP+QP mix rejected (833-836). REML path: `validate/resolve_selection_penalty_for_reml` (model/base.py:987-1001) enforces the protected "fit_reml has no selection penalty" contract; QP constraints are stripped for passthrough REML and restored (`model/reml_setup.py:154-172`, used at fit_ops.py:1263/1434).
- **SSP reparametrisation** `dm_builder.py`: `compute_R_inv` 88-110 (`R = chol(B'WB/Σw + λΩ + 1e-8 I)ᵀ`, `R_inv = inv(R)`), `compute_projected_R_inv` 113-135 (same in the constraint/identifiability-projected subspace); applied per group in `_process_info` 400-651; constraints composed into solver coordinates at 641-649 by mutating the `GroupInfo` in place; `rebuild_design_matrix_with_lambdas` 1202+ recomputes R_inv per lambda change (used by `model/base.py:1009-1019 rebuild_dm_with_lambdas` from REML).
- **Fit-time constraints**: difference-matrix builders `features/_spline_constraints.py:12-27`; QP engine `solvers/constrained_qp.py` (primal active set, `solve_constrained_qp` 57-199) driven from `solvers/irls_direct.py:831-843` (block constraints assembled into model-wide A) and 1578-1633 (per-IRLS-iteration QP with warm-started active set); SCOP engine `solvers/scop.py` (`SCOPReparameterization` 62-177 raw exp-map, `SCOPSolverReparam` 214-288 solver-space wrapper, QP initialisers at 143-177/264-288); GroupInfo/GroupSlice carry `constraints`, `monotone_engine`, `scop_reparameterization` (types.py:142-151, 270-272).
- **REML select handling**: `component_types={"null": "selection"}` flows GroupInfo → GroupMatrix (`dm_builder.py:498/540/573/595`) → `PenaltyComponent.component_type` (`reml/penalty_algebra.py:874`) → the select-snap heuristic in `reml/direct.py:335-343` and `reml/discrete.py:501-509` (degenerate null-space lambda snapped to the upper bound).

## 2. DATA FLOW

### 2.1 Selection penalties (fit() path)

1. User passes `penalty=` string/object + `selection_penalty=` → `resolve_penalty` (model/base.py:514-548) deep-copies user objects and normalises `lambda1` (None | "auto" | float ≥ 0). sklearn wrappers pre-pick `penalty="group_lasso"` when only `selection_penalty` is set (sklearn.py:301-316).
2. After the DM is built, `validate_penalty_features` runs against the final `GroupSlice` list (model/base.py:953). `resolve_selection_penalty_for_fit` (974-984) resolves "auto" → `0.1 * compute_lambda_max(...)`; `compute_lambda_max` (958-971) does one `rmatvec` of the null-model residual, O(np) time, O(p) memory.
3. `fit_pirls` receives `groups: list[GroupSlice]` and the penalty object. Per outer IRLS iteration: block Hessians `X_g' W X_g (+ S_gg)` (O(Σ n·p_g²) dense, less for discretized group matrices), then the penalty-type-specific factor/eigendecomposition (O(Σ p_g³)). Inner BCD sweeps update the shared residual r (n-vector) and, when a smoothing S exists, the running `S_beta` p-vector. All prox math is per-group on (p_g,) slices — no full-p penalty algebra except `penalty.eval` in the merit function (pirls.py:729).
4. Flavors: stage-1 fit → `Adaptive.adjust_weights` builds a *new* groups list (flavors.py:57-68) → stage-2 warm-started fit (pirls.py:1469-1495). The model keeps the original `model._groups`; the adjusted weights live only inside the second solver call.

### 2.2 Smoothing penalties and SSP

1. Each spline spec's `build()` (`_spline_build.build_group_info` 54-129) materialises: raw basis B (n×K sparse CSR), penalty Ω (K×K dense), constraint projection Z (K×K'), identifiability projection (K'×K'-1). The resulting `GroupInfo.penalty_matrix` is in post-projection coordinates; `projection` holds raw→solver map for select/discrete paths.
2. `dm_builder._process_info` computes R_inv (p_g×p_g dense) from `B'WB/Σw + λΩ`, then materialises the solver design: `DenseGroupMatrix(B @ R_inv)` (n×p_g dense copy) or keeps B sparse/discretized with R_inv applied on the fly (`SparseSSPGroupMatrix`, `DiscretizedSSPGroupMatrix`). Ω (and per-component Ω_j, expanded to `P Ω_j P'` when a projection exists — dm_builder.py:493-495, 536-539) are attached to the GroupMatrix for REML.
3. In solver coordinates the smoothing quadratic is assembled once per PIRLS call as a dense p×p `S` (`build_penalty_matrix`, reml/penalty_algebra.py:557+, invoked from pirls.py:651-656) — but only when some group actually carries lambda2>0 (`_has_structural_smoothing_penalty` pirls.py:58).
4. On each REML lambda update, `rebuild_design_matrix_with_lambdas` (dm_builder.py:1202+) recomputes R_inv (and hence the SSP basis) for every group whose lambda changed, including a full recomputation of the weighted Gram `B'WB` inside `compute_R_inv` / `compute_projected_R_inv`.

### 2.3 select=True double penalty

Exact path: `build_group_info:75-76` → `build_select` → penalty Ω_c eigendecomposed (K'×K' `eigh`); null space (2-dim: constant + linear for m≤2) reduced to 1 column by removing the constant direction; `GroupInfo(columns=B(raw), projection=U_combined (K×(1+range)), penalty_components=[null(1×1 identity block), wiggle/d{order}], component_types={"null": "selection"})`. `_process_info` then computes projected R_inv inside `span(U_combined)`; REML sees two `PenaltyComponent`s per select term and optimises both lambdas, with the "selection"-typed one eligible for the snap heuristic (reml/direct.py:335-343). Identifiability projection is *skipped* for select terms (`absorbs_intercept` = `not select`, spline.py:240-252) — the constant direction was already removed in the eigendecomposition.

Discrete path: `build_knots_and_penalty:147-156` does the eigendecomposition on the spec, then `dm_builder.py:791-841` re-derives the null/wiggle components inline (see SUSPECT S1) with `columns=None` and `B_unique`/`bin_idx` support geometry.

`selection_penalty` (lambda1) and `select=True` never meet in the same code path: lambda1 lives on the Penalty object and acts through prox thresholds in PIRLS; select=True lives in the penalty-component list and acts through REML-optimised quadratic penalties. fit_reml rejects lambda1>0 (model/base.py:987-994); PIRLS treats the select "null" component as just another block of S.

### 2.4 Post-fit shape repair

`apply_shape_postfit` (shape_ops.py:638-766): gather pending postfit-constrained spline terms (515-528) → skip everything if all are roundoff-feasible (531-537, 673-676) → refuse when SCOP terms exist (677-681) → open a `FittedStateRevision` transaction. Per term: histogram grid weights (25-33, G=500 bins of the training column), `MonotoneRepairer/CurvatureRepairer.repair` (constraints.py:351-430/594-652): reconstruct curve on grid, certificate; if infeasible, `_project_shape_in_fitted_basis` builds the (G×p_g) fitted basis, (p_g×p_g) normal matrix, and runs SLSQP with cutting-plane refinement over analytically-located violating derivative extrema. Result validated against a feasible zero-term fallback on deviance and full penalized merit (`_validate_repair_for_publication` 399-512, using `_smooth_penalty_value` for beta'Sbeta without materialising S, 179-206, and `penalty.eval` for the selection part, 368) with a Newton-profiled intercept (209-329). Published beta/intercept synchronised across public/solver results (60-113), caches invalidated, EDF/phi/statistics refreshed (540-635).

### 2.5 Fit-time shape constraints

QP engine: raw first/second-difference cone on coefficients (`_spline_constraints.py:12-27`) → composed with the identifiability projection at build (`_spline_build.py:107-112`, discrete: dm_builder.py:742-752) → composed with R_inv_local into solver coordinates (dm_builder.py:641-649) → assembled into a model-wide (ΣC_g × p) A matrix and solved per IRLS iteration by the dense active-set QP (irls_direct.py:831-843, 1578-1633). SCOP engine: exp-reparametrised coefficients (scop.py), solver sees a nonlinear per-group map via `GroupSlice.scop_reparameterization`; select/lambda1/postfit interplay is guarded (fit_ops.py:820-836; _spline_build.py:63-68 rejects fit-time constraints + select; constraints.py:62 rejects certificates on SCOP coordinates).

## 3. STATE OBJECTS

| Object | Where | Fields / lifecycle | Overlap notes |
|---|---|---|---|
| `Penalty` instances (Ridge/GroupLasso/GEN/SGL) | penalties/*.py | `lambda1`, `alpha`, `flavor`, `features`. Constructed by user or `resolve_penalty`; **mutated in place** by `resolve_selection_penalty_for_fit`/`_for_reml` (model/base.py:983, 1000) which overwrite `lambda1` ("auto"→float). The model deep-copies user-supplied objects (base.py:546) so the mutation is on an owned copy | Fitted lambda recovered later via `fitted_penalty(model)` + `_selection_penalty_fitted` (model/fit_state.py:521-524, 627-670) — two records of the same fact |
| `GroupSlice` | types.py:259-284 | name/start/end/weight/penalized/feature_name/subgroup_type/constraints/monotone_engine/scop_reparameterization. Created in dm_builder feature loop (895-909); lives on `model._groups` for the model lifetime | Carries *both* column bookkeeping and constraint-engine state; `strip_qp_constraints` mutates it temporarily during REML (reml_setup.py:154-172). Weight duplicated/diverges from flavor-adjusted copies used inside PIRLS stage 2 |
| `GroupInfo` | types.py:115-255 | 30+ optional fields: columns, penalty_matrix, penalty_components, component_types, lambda_policies, projection, constraints, monotone_engine, raw_to_solver_map, scop_reparameterization, spline-cat compact fields, factor-smooth compact fields. Built by feature specs (or inline by dm_builder for discrete paths), consumed and partially **mutated** by `_process_info` (constraints/raw_to_solver_map recomposed in place, dm_builder.py:644-649; lambda_policies/penalty_components injected at 869-877) | It is simultaneously the feature→builder contract *and* a scratch container the builder rewrites; the comment `subgroup_name: "linear" or "spline"` (types.py:131) is stale — only "bilinear"/"wiggly" are produced today (features/interaction.py:1080, 1088) |
| `LambdaPolicy` | types.py:59-90 | frozen; mode estimate/fixed(+value). Resolved per component by `_spline_select.resolve_lambda_policies` (and a parallel implementation in factor_smooth.py:214) | duplicate resolution logic (see S8) |
| `PenaltyComponent` | types.py:288-315 | REML-side view of one (Ω, λ): omega_raw/omega_ssp, rank, log-det, eigvals, component_type, lambda_policy, penalty_kind, repeat metadata. Built in reml/penalty_algebra.py from GroupMatrix attributes | third representation of the same penalty (GroupInfo.penalty_components → GroupMatrix.omega_components → PenaltyComponent); component_types dict is stringly-typed glue across all three |
| `LinearConstraintSet` | types.py:24-55 | A, b; `compose(P)` returns new set. Attached to GroupInfo and GroupSlice | composed at three different sites (build, discrete build, _process_info) with in-place field replacement |
| `MonotoneRepairResult` | constraints.py:16-33 | mutable; `feature_name` filled by caller (shape_ops.py:729), `repaired_log_effect`/`max_violation_after` overwritten again in `_validate_repair_for_publication` (shape_ops.py:509-511). Stored in both `model._shape_repairs` and, for monotone kinds, duplicated into `model._monotone_repairs` (shape_ops.py:746-748) | double bookkeeping for compat |
| `ShapeConstraintCertificate` | constraints.py:36-48 | frozen, ephemeral | — |
| `SCOPReparameterization` / `SCOPSolverReparam` | scop.py:62-177 / 214-299 | raw vs solver-space exp maps; solver reparam wraps the raw one; spec also caches `_scop_Sigma`, `_scop_null_dim`, `_scop_col_means` for predict (dm_builder.py:771-773) | same geometry stored on spec, GroupInfo, and GroupSlice |
| `QPResult` | constrained_qp.py:29-35 | beta, active_set (warm start carried across IRLS iterations, irls_direct.py:1626-1630) | — |

## 4. COMPLEXITY TABLE

| Routine | Time | Memory | Notes |
|---|---|---|---|
| `prox_group` (all penalties) | O(p_g) | O(p_g) | closed form; `prox` full-vector variant copies beta O(p) but is test-only |
| `penalty.eval` | O(p) | O(1) | called once per state evaluation (pirls.py:729), incl. every line-search trial |
| `Adaptive.adjust_weights` | O(Σ n·p_g) = O(np) with group_matrices, else O(p) | O(n) transient | one matvec per group; runs once |
| `compute_lambda_max` (model/base.py:958) | O(np) | O(p) | one rmatvec |
| `compute_R_inv` / `compute_projected_R_inv` (dm_builder.py:88-135) | Gram O(n·p_g²) dense / O(nnz·p_g) sparse; chol+`inv` O(p_g³) | O(p_g²) | explicit `np.linalg.inv(R)` instead of triangular solves; dense `B @ R_inv` copy (n×p_g) for dense groups (dm_builder.py:531) |
| `rebuild_design_matrix_with_lambdas` (dm_builder.py:1202+) | per changed group: full Gram recompute + O(p_g³) | new GroupMatrix per group | **inside the REML optimiser loop**; B and W unchanged across iterations yet `B'WB` is recomputed every time (1224/1227/1244/1247/...) |
| PIRLS `_build_group_hessians` + penalty factors (pirls.py:830-862) | O(Σ n·p_g²) + O(Σ p_g³) per outer iteration | O(Σ p_g²) | eigh per block per outer iteration for GroupLasso/GEN; eigvalsh for generic penalties |
| BCD inner sweep (pirls.py:899-965) | per group O(n·p_g) matvecs + O(p_g²) solve; smoothing update `S_beta += S[:, g.sl] @ d` is **O(p·p_g)** | dense S is O(p²) | S is block-diagonal but stored dense; the column-slice update wastes a factor p/p_g per group update (see S5) |
| `_composite_kkt_violation` (pirls.py:479-538) | O(np + Σ p_g²) | O(n) | only at convergence check; recomputes Hessians if L_groups missing |
| `build_penalty_matrix` (penalty_algebra.py:557) | O(Σ p_g²) fills | O(p²) dense | once per PIRLS call |
| `eigendecompose_select` (_spline_select.py:14-45) | O(K'³) | O(K'²) | build-time, per select term |
| `build_select_group_info` multi-m (_spline_select.py:107-111) | per order: `lstsq(Z, U)` O(K·K'²) + projections O(K'³) | O(K'²) per component | duplicated verbatim in dm_builder discrete path |
| `build_multi_m_components` (_spline_multi_penalty.py:11-28) | per order: penalty build O(K²)–O(K³) + constraints (NaturalSpline null space: 2K BSpline evals + QR O(K³)) + identifiability (np.unique O(n log n) + basis at support O(u·K)) | O(K²) per order | identifiability projection recomputed per order though it is order-independent (see S6) |
| `shape_constraint_certificate` (constraints.py:272-303) | candidates O(K) roots + rows O(C·K²) (basis eval) | O(C·p_g) | post-fit only |
| `_project_shape_in_fitted_basis` (constraints.py:454-583) | per refinement: SLSQP on p_g vars, C constraints, up to max(8, 2p_g) refinements, maxiter 500 | O(G·p_g + C·p_g) | post-fit only; G=500 |
| `apply_shape_postfit` validation (shape_ops.py:399-512) | per term: 2 profiled intercepts (Newton over n rows, ≤50×21 evals worst case) + 2 objective evals O(n) + eta deltas O(n·p_g) | O(n) | runs per repaired term; `current_eta` re-profiled cumulatively |
| `solve_constrained_qp` (constrained_qp.py:57-199) | per active-set iteration: dense KKT solve O((p+a)³); blocking-constraint scan O(m_c·p) with Python loop (181-191) | O((p+a)²) | **called every IRLS iteration** (irls_direct.py:1621) on the full model-p system, warm-started |
| `SCOP jacobian` (scop.py:104-111, 248-250) | O(q²) per call (raw, with Python column loop) / O(q_eff²) diag | O(q²) | inside Newton inner loops (scop_newton.py) |
| `GroupInfo.__post_init__` component check (types.py:202-216) | O(q_g·p_g²) allclose | O(p_g²) | every GroupInfo construction |

## 5. SUSPECTS

**S1. Discrete select=True path duplicates `build_select_group_info` line-for-line.**
`dm_builder.py:791-841` re-implements `features/_spline_select.py:86-125` (null block, wiggle embedding, per-order `lstsq(Z, U)` projections, `component_types={"null": "selection"}`) inline for the discretized path instead of calling the shared helper. This is exactly the kind of split that can make `discrete=True` drift from the exact path (a protected semantic). Differences today: the inline version never consults `spec._lambda_policy` inside the component build (patched afterwards by the generic 869-877 block), and any future fix to `_spline_select` must be applied twice. Verify: diff outputs of both paths for a multi-m select spline (components, penalty_matrix, component_types) and consider whether `build_select_group_info(B=None, ...)` could serve the discrete path.

**S2. `compute_lambda_max` scaling vs the solver's actual threshold — "auto" calibration doc mismatch.**
`model/base.py:958-971` computes `max_g ||X'(w(y-mu0))||_g / w_g / n` (divides by n), and `resolve_selection_penalty_for_fit` (978) uses `0.1 *` that as the fitted lambda1. But the PIRLS objective is the *unnormalised* `0.5·D(beta) + penalty.eval(beta)` (pirls.py:646-647) and the prox threshold is `step·lambda1·w_g` with step from the unnormalised block Hessian — so the true smallest all-zeroing lambda is `max_g ||grad_g||/w_g` *without* `/n`. The docstring ("Smallest lambda1 at which all groups are zeroed") therefore appears wrong by a factor of n, making `selection_penalty="auto"` effectively `0.1·lambda_max/n` — much weaker than "10% of lambda_max" advertised in group_lasso.py:24 and sklearn docs. The only test (tests/test_select.py:404-419) checks positivity, not the zeroing property. Verify numerically: fit with `lambda1 = compute_lambda_max(...)` and check whether all groups are actually zero (I expect not, for n ≫ 1). If /n is intentional (mean-deviance convention elsewhere), the docstrings are wrong instead.

**S3. `Penalty.prox` (full-vector) is production-dead; protocol docstring stale.**
The solver only ever calls `prox_group` (pirls.py:521, 938, 958). `prox` is called exclusively from tests (tests/test_core.py:357, tests/test_penalties.py passim). Yet `base.py:88` says "The solver calls prox() in the inner loop", and every penalty implements both. Custom user penalties must implement two operators when one is used. Verify: grep confirms no non-test caller of `.prox(`.

**S4. Penalty type-switches scattered across the solver break the protocol abstraction.**
`type(penalty) in (GroupLasso, GroupElasticNet)` / `type(penalty) is Ridge` dispatch at pirls.py:837-862, exact-curvature inference limited to the same trio at pirls.py:541-604 (`type(penalty) not in (...) → return`), and `penalty_can_zero_groups` (base.py:66-81) special-casing Ridge/GEN by isinstance. SparseGroupLasso — a first-class exported penalty — silently gets the slower generic prox-gradient inner step and the "historical protocol fallback" inference curvature. Not a bug, but three independent penalty-classification points that must be kept in sync when a penalty is added. Verify: fit SGL and confirm which branches it takes; check no behavioural asymmetry beyond speed (e.g. `_selection_local_curvature_depends_on_beta` pirls.py:595-604 returns False for SGL, so its EDF/covariance treatment differs from GroupLasso's).

**S5. Dense p×p S in PIRLS with column-slice updates in the BCD inner loop.**
`build_penalty_matrix` materialises block-diagonal S as dense p×p (penalty_algebra.py:574) and PIRLS updates `S_beta += S[:, g.sl] @ d` (pirls.py:946) — O(p·p_g) per group update where O(p_g²) suffices (off-block rows are structurally zero). For wide models (many smooth terms, large p) this is O(p²) memory and an O(p/p_g) constant-factor waste in the hot inner loop, plus `S @ beta` refreshes at 869/887. Verify with a profile at p ≈ 2-5k; a per-group block list would remove both.

**S6. Multi-m components recompute constraints and identifiability per order.**
`_spline_multi_penalty.build_multi_m_components` (11-28) calls `apply_constraints` + `apply_identifiability` per order; `apply_identifiability_for_spec` (_spline_identifiability.py:75-88) recomputes `np.unique(x)` (O(n log n)) and the support basis per order, though the projection depends only on x and the constraint projection, not the order. For NaturalSpline each `_apply_constraints` call rebuilds the boundary null space (2K BSpline evaluations + complete QR, _spline_constraints.py:30-47). Additionally `build_group_info` already ran the same constraint+identifiability pass at 95-96 before calling the multi-m builder — so with m orders the whole chain runs m+1 times. Build-time only, but O((m+1)·n log n) on large n is measurable. Verify by timing build() with m=(1,2,3) at n=1e6.

**S7. In-place mutation of `GroupInfo` during DM build.**
`_process_info` rewrites `info.constraints`/`info.raw_to_solver_map` in place (dm_builder.py:644-649), and the feature loop injects `lambda_policies`/synthetic `penalty_components` into infos (869-877). If a `GroupInfo` were ever reused across two builds (specs cache them? currently they don't) constraints would be composed twice — the correctness currently rests on build() returning fresh objects every time. Also note the injected synthetic component `("wiggle", penalty_matrix)` at 876 sets `component_types={"wiggle": "wiggle"}` while `_spline_build.py:126-127` uses `{"wiggle": "difference"}` for the same situation on the exact path — a stringly-typed drift between the two paths (consumers: `reml/penalty_algebra.py:874` propagates it into PenaltyComponent.component_type; only "selection" is semantically consumed today, so this is latent). Verify: confirm no consumer distinguishes "wiggle" vs "difference".

**S8. Two parallel `resolve_lambda_policies` implementations.**
`features/_spline_select.py:48-71` (splines; validates against penalty_components, defaults `{"wiggle"}`) and `features/factor_smooth.py:214` (factor smooths). Same contract, separately maintained validation/error text. Verify divergence on dict inputs with unknown keys.

**S9. `eigendecompose_select` uses an absolute eigenvalue cutoff and hard-requires exactly 2 null values.**
`_spline_select.py:24-31`: `eigvals < 1e-10` on an unscaled penalty matrix; a rescaled x (penalties scale like Δx^{-2m+1}-ish) or large K can push true null-space eigenvalues above / small positive ones below the cutoff, producing the opaque "select=True requires exactly 2 null eigenvalues" error, and the check rejects m=1 P-splines (1 null value) by construction even though `_select_compatible` (spline.py:103-111) only checks `max(m) <= 2`. Verify: `Spline(select=True, m=1)` and a covariate scaled by 1e6 — does build fail?

**S10. Post-fit repair machinery: near-duplicate `repair()` bodies and test-only helpers.**
`MonotoneRepairer.repair` (constraints.py:351-430) and `CurvatureRepairer.repair` (594-652) are ~60 identical lines differing only in the kind token; `monotonicity_violation`/`curvature_violation` (433-451) and `derivative_grid_matrix` (655-659, NotImplementedError placeholder since the QP/SCOP engines shipped) have no production callers (tests only: tests/test_monotone.py, tests/test_shape_postfit.py). `MonotoneRepairResult.kind` property (31-33) exists purely as a compat alias for `direction`, and shape_ops keeps parallel `_shape_repairs` and `_monotone_repairs` dicts (shape_ops.py:746-748). Compat shims worth an inventory pass.

**S11. Stale metadata comments on GroupInfo/GroupSlice subgroups.**
types.py:131 (`subgroup_name: "linear" or "spline"`) and types.py:269 (`subgroup_type: "linear", "spline", or None`) no longer match producers: only `"bilinear"`/`"wiggly"` (features/interaction.py:1080/1088) and the synthetic `"ordered_spline"` label injected directly in inference tables (inference/coef_tables.py:412) exist. Meanwhile inference/summary/export still branch on `subgroup_type == "linear"` (inference/coef_tables.py:476, inference/summary.py:449/814, export/summary.py:271, inference/metrics.py:239) — those branches look unreachable for current builders. Verify by grepping for any remaining producer of `"linear"`/`"spline"` subgroup names; if none, several inference branches are dead.

**S12. `rebuild_design_matrix_with_lambdas` refactorises inside the REML loop.**
dm_builder.py:1202+ recomputes the weighted Gram `B'WB` from scratch inside `compute_R_inv`/`compute_projected_R_inv` on every lambda update, per group, although B and the sample weights are fixed for the whole fit — only λΩ changes. Caching G per group (it is p_g×p_g) would reduce each rebuild from O(nnz·p_g + p_g³) to O(p_g³). Also `compute_R_inv` uses explicit `np.linalg.inv(R)` (dm_builder.py:110) rather than `solve_triangular`, and the 1e-8 ridge (108) is absolute rather than scale-relative. Verify how often the REML runners call `rebuild_dm_with_lambdas` (model/base.py:1009) per fit.

**S13. Flavor-adjusted weights are invisible to post-fit consumers.**
`fit_pirls` stage 2 fits with `adjusted_groups` (pirls.py:1470-1479) but `model._groups` keeps the original `sqrt(p_g)` weights (dm_builder.py:901). Anything recomputing selection curvature or Breheny-Huang EDF after the fit from `model._groups` + `fitted_penalty` (e.g. `_add_selection_local_curvature` via inference/covariance, `_refresh_repaired_geometry` in shape_ops.py:540+) will use the wrong `group.weight` for adaptive fits. Verify: fit with `GroupLasso(flavor=Adaptive())`, then call `summary()`/`apply_shape_postfit()` and compare EDF against the in-fit rank_info.

**S14. `_project_feasible` in the QP solver is not an exact projection and can fail silently.**
constrained_qp.py:38-54 runs at most 100 single-constraint projections and `break`s on feasibility, but if 100 iterations are exhausted the returned point may still be infeasible and the active-set loop starts from it without a check; the multiplier drop test uses `(A_eq A_eq')^{-1}` which is singular for linearly dependent active constraints (155-157 falls back to lstsq, fine, but combined with the Python blocking-constraint loop at 181-191 this whole solver is O(m_c·p) Python-level per iteration). It runs every IRLS iteration for QP-constrained fits (irls_direct.py:1621). Verify with a constraint set of a few hundred rows (fine-knot monotone spline + interactions) — Python-loop cost and the infeasible-start possibility.

**S15. `penalty_can_zero_groups` default for unknown penalties.**
base.py:73-81: any third-party penalty with lambda1>0 is presumed able to zero groups ("historical sparse default"). Rank/selection logic (pirls.py:1334-1346 EDF, rank.py:885-895 selected-name fallback) then treats zero-norm groups from, say, a custom pure-L2 penalty as *deselected*, changing EDF and summaries. Documented as intentional, but it is a semantic trap the protocol cannot express; verify whether any doc tells custom-penalty authors about it.

---

### Cross-checks performed
- All grep-based call-site traces executed against the audit worktree (imports of `superglm.penalties.*`, `superglm.constraints`, `_spline_select`, `_spline_multi_penalty`, `monotone_engine`, `solve_constrained_qp`, `subgroup_name/subgroup_type`, `component_type`, `selection_penalty`, `lambda_max`).
- Protected semantics respected in analysis: fit vs fit_reml separation (model/base.py:987-1001), select=True vs selection_penalty separation (fit_ops.py:820-831 also rejects lambda1>0 with fit-time monotone), discrete vs exact select path (S1 is precisely a drift-risk observation, not a proposal to merge).
