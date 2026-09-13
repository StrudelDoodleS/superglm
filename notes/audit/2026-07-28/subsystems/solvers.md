# Architecture report: `solvers` subsystem

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ origin/master (f082e9b).
All paths below are relative to `src/superglm/solvers/` unless prefixed.
Conventions: n = rows, p = total built columns, m = number of smooth terms (groups), q = number of penalties / penalty components, K = dominant-block level count, k = per-level block size, q_s = "dense-small" column count in the structured backend, p_g = columns of group g.

---

## 1. MODULE MAP

### 1.1 Top-level solver entry points

**`pirls.py` (1497 lines) — proximal-Newton BCD PIRLS (the "selection" solver).**
- `fit_pirls` (pirls.py:1401) — public entry. Wraps a dense `X` into a `DesignMatrix` if needed (`_wrap_dense_X`, pirls.py:1394), runs `_fit_pirls_inner`, and if `penalty.flavor is not None` (adaptive penalties) runs a **two-stage** fit: uniform-weight fit, `flavor.adjust_weights`, warm-started refit (pirls.py:1443-1495).
- `_fit_pirls_inner` (pirls.py:607) — the outer PIRLS loop + inner BCD loop + line search + terminal rank/EDF/dispersion assembly. Details in §2.1.
- Block algebra helpers: `_build_group_hessians` (342), `_compute_group_hessians` (363), `_factor_psd_block`/`_solve_factored_block` (383/391), `_ridge_block_factors` (398), `_radial_block_eigensystems` (414), `_solve_radial_block` (443 — exact trust-region-style radial solve via `brentq` for the group-lasso prox with full block Hessian), `_composite_kkt_violation` (479), `_add_selection_local_curvature` (541), `_selection_local_curvature_depends_on_beta` (595).
- Result publication machinery: `PIRLSResult` (175) with `_publish`/`_mutable_copy`/`__deepcopy__`/pickle protocol (204-264), `_immutable_array_copy` (267), `_freeze_result_arrays` (273), `_FrozenResultMapping` (102), `IterationDiagnostics` (125).
- Callers of `fit_pirls`: `model/path_ops.py:82`, `model/fit_ops.py:700`, `profiling/nb.py:475`, `profiling/tweedie.py:3764`, `reml/runner.py:176,374`, `reml/efs.py:96,192,389`.

**`irls_direct.py` (2601 lines) — direct penalised IRLS (no BCD; the REML/smoothing solver).**
- `fit_irls_direct` (382) — thin retry wrapper: runs `_fit_irls_direct_once`, and on `SumToZeroIdentifiabilityError` with `direct_solve="auto"` retries once with `direct_solve="gram"` (460-470).
- `_fit_irls_direct_once` (473) — one monolithic function (~2100 lines) containing: backend resolution (structured vs gram vs qr), penalty assembly, constrained-QP branch, SCOP branch, the IRLS loop, line search, curvature-rescue controller, terminal geometry/rank/EDF/dispersion, cache export, and SCOP post-fit inference. Details in §2.2.
- Helpers: `_stable_penalized_deviance_delta` (116 — polarization-identity merit delta using `math.fsum`), `_has_constant_irls_weights` (256), `_working_sums` (271), `_robust_solve` (283, **dead in src**, only `tests/test_robust_solve.py`), `_solve_profiled_intercept_from_h_inv` (299, **dead — no callers anywhere**), `_build_penalty_matrix` (315, compat wrapper around `reml.penalty_algebra.build_penalty_matrix`), `_sqrt_penalty_augmented` (334), `_invert_xtwx_plus_penalty` (347, used only by `reml/runner.py:205`), `_safe_decompose_H` (375, "compatibility wrapper", used by `reml/efs.py:135,225`).
- SCOP trial-state types: `_SCOPGroupSpec` (154), `_SCOPGroupState` (166), `_SCOPTrialState` (178), `_CenteredFactorCertification` (186), `_evaluate_scop_trial` (196).
- Callers of `fit_irls_direct`: `model/fit_ops.py`, `model/reml_execute.py`, `model/reml_finalize.py`, `profiling/nb.py`, `profiling/tweedie.py`, `reml/direct.py`, `reml/discrete.py`, `reml/runner.py`, `reml/scop_efs.py`.

**`irls_state.py` (195 lines) — shared immutable IRLS snapshots and step selection.**
- `SolverState` (17) frozen dataclass; `_IRLSState = SolverState` (37) is an explicit "migration alias".
- `_evaluate_irls_state` (63) — evaluates beta/intercept → eta (via `dm.matvec`), stabilized eta, mu, deviance; freezes all arrays.
- `_irls_trial_is_unsafe` (141) — rejects non-finite states or merit increases beyond 64·eps roundoff; supports an injectable `merit_delta`.
- `_select_irls_trial` (174) — fixed-endpoint step halving: alpha = 1, 1/2, ..., 2^-20; returns `_IRLSStepDecision` (40) or atomic rejection. Used by **both** solvers (pirls.py:1009, irls_direct.py:1727 & 1806).

### 1.2 Shared linear-algebra infrastructure

**`centered_system.py` (347 lines) — intercept-profiled weighted systems.**
- `CenteredSystem` (33) frozen dataclass: `sum_w, mean_x (p,), mean_z, data_gram (p,p), rhs (p,), penalty (p,p), hessian (p,p)`; `raw_weighted_moments` (44) reconstitutes raw `X'WX, X'W1, X'Wz, sum_wz`.
- `build_centered_system` (184) — the per-iteration Gram builder. Tries, in order: `packed_centered_gram_rhs` (fully discrete), mixed discrete tabmat centering, raw-spline tabmat centering, tabmat split centering, then the stable fallback `centered_gram_rhs`. Ends with a PSD guard: try `np.linalg.cholesky(hessian)`; on failure eigendecompose and clip negative eigenvalues (283-291).
- `build_anchor_centered_system` (303) — anchored variant for predictors whose location exceeds their scale (used only on the SCOP-reduced system, irls_direct.py:1392).
- `refresh_centered_rhs` (147) — reuses an invariant Gram, recomputes only the RHS (constant-weight cache path).
- `TabmatCenteringState` (53) — fit-local mutable accept/reject memo for tabmat acceleration.
- Streaming factor certification: `iter_grouped_design_chunks` (61, bounded 16 MiB chunks), `grouped_weighted_factor(_rhs)` (71/83), `penalty_factor` (101, eigh-based sqrt of S), `grouped_augmented_factor(_rhs)` (110/123).
- External callers: `inference/_metrics_design.py:362`, `model/state_ops.py:143-248`, `reml/observed_geometry.py:51` — this module is shared fitting/inference infrastructure, not solver-private.

**`rank.py` (895 lines) — the versioned rank policy.**
- `RankPolicy` / `SHARED_RANK_POLICY` (14/25): factor rcond = sqrt(eps), gram rcond = eps, certification band 32, warning condition 1/sqrt(eps).
- `decompose_gram` (453): equilibrate (diagonal scaling, `_equilibrate_gram` 352) → fast Cholesky with `pocon` condition estimate + probe-solve residual check (486-544) → else `eigh` with cutoff, optional second Cholesky attempt (574-610), pivoted-Cholesky "representative column" selection at rank deficiency (640-689, greedy loop) → `gram_eigh` fallback. Produces `RankDecomposition` (134) carrying solve/pseudo-inverse/log_pdet/null bases.
- `decompose_factor` (725): factor-space rule via SVD of the column-equilibrated factor; optional retained factor-RHS solve (`solve_factor_rhs`, 179); same greedy representative-column path (786-851).
- `decompose_symmetric` (710) = `decompose_gram(allow_indefinite=True)`.
- `needs_factor_certification` (99): decides when a Gram decomposition near the resolution boundary must be re-certified from a streamed observation factor.
- `streamed_weighted_factor(_rhs)` (43/64): chunked QR accumulation, factor never exceeds (p+1, p) rows retained.
- `_retained_log_pdet` (428) + `_scaled_subspace_logdet` (403): high-relative-accuracy pseudo-logdet via LAPACK `dgejsv` and Jacobi complementary-minor identity.
- `RankInfo` (297): fitted-subspace metadata (selected columns, three decompositions: data / augmented / coefficient, feature EDF, group EDF).
- `selected_group_name_set` (875): legacy selection fallback.
- External callers: `reml/objective.py`, `reml/discrete.py`, `reml/penalty_algebra.py`, `reml/observed_geometry.py`, `reml/scop_efs.py`, `reml/scop_geometry.py`, `inference/covariance.py`, `inference/metrics.py`, `model/state_ops.py` — rank policy is the shared numerical backbone.

**`hessian_factor.py` (267 lines) — `HessianFactor` protocol (86) + `DenseHessianFactor` (136) + `as_hessian_factor` (256).** Defines the operation set REML/inference need from a penalized Hessian (solve, logdet, selected inverse blocks/diagonals, penalty traces, operator cross-traces). Dense adapter wraps a historical dense inverse. Consumed by `reml/w_derivatives.py`, `reml/gradient.py`, `reml/direct.py:331`, `reml/discrete.py:477`, `inference/covariance.py:20`. `_component_indices`/`_component_omega`/`_expanded_component_omega` (17/71/27) expand compact `PenaltyComponent`s (identity / repeated / sum_to_zero / dense).

**`working_rows.py` (125 lines) — working-response geometry.**
- `CoefficientWorkingRows` (16), `supports_observed_newton` (25, exactly Gamma×LogLink by concrete type), `coefficient_working_rows` (65): Fisher rows by default; guarded exact observed-Newton rows for Gamma/log (`w·y/mu` weights); Gaussian/identity short-circuits to bit-exact `(sample_weight, y)` (46-52) to keep constant-geometry caches valid. Any invalid observed row rejects the whole observed model (fit-wide fallback flag consumed at irls_direct.py:1304-1318).

**`dispersion.py` (35 lines)** — `dispersion_likelihood_size` (11) and `pearson_residual_degrees_of_freedom` (22): frequency-weight vs Tweedie EDM-prior-weight df contract. Used by both solvers' phi computation (pirls.py:1373, irls_direct.py:2472,2586).

**`constrained_qp.py` (199 lines) — primal active-set QP for shape constraints (monotone splines with `monotone_engine != "scop"`).**
- `solve_constrained_qp` (57): min ½β'Hβ − g'β s.t. Aβ ≥ b. Unconstrained solve first; Dykstra-like feasibility projection `_project_feasible` (38); per-iteration dense KKT solve (126-141); multiplier check via `(A_eq A_eq')⁻¹` (155); blocking-constraint ratio test in a Python loop (181-191). `QPResult` (29) carries the active set for warm starting.
- Called from: irls_direct.py:1621 (with `active_set_init=prev_active_set` warm start across IRLS iterations, reverted on step rejection at 1830), and SCOP `qp_initialize` (scop.py:171,286).

### 1.3 SCOP (shape-constrained P-splines)

**`scop.py` (299 lines) — SCOP reparameterisation maps** (Pya & Wood 2015). `SCOPReparameterization` (62, raw space: null basis + exp-transformed shape block; monotone Σ = lower-triangular ones, convex/concave second-order Σ), `SCOPSolverReparam` (214, solver space, `forward = exp(clip(β))`, diagonal Jacobian, difference penalty `D'D`), builders (180/291), `qp_initialize` (143/264) delegating to `solve_constrained_qp`.

**`scop_newton.py` (1091 lines) — damped full-Newton on the SCOP latent objective.**
- `scop_newton_step` (228): single group. Gradient/Hessian exploit the diagonal Jacobian: `H_gn = (j j') ∘ B'WB + λS`; full-Newton adds `diag(grad_data)` (333); PD failure → Fisher (Gauss-Newton) with escalating ridges 1e-8, 1e-4, then plain gradient step (339-348); step halving vs `_safe_trial_objective` (356-364).
- `scop_joint_newton_step` (773): all SCOP groups in one joint Hessian with cross-group blocks `_compute_cross_gram` (733 — discretized×discretized uses `_disc_disc_2d_hist` 2-D weight histogram; mixed cases scatter the discretized side to observation level, materialising an (n, q_eff) block, 764-770). Joint line search uses gamma-space quadratic caches `_JointObjectiveCache` (83) + `_joint_objective_from_gammas` (645), so each halving trial is O(Σq_i²) instead of O(n).
- Private prototype switches `_SCOPPrototypeConfig` (37) select direct vs MINRES (`_solve_step_minres` 448, block-Jacobi preconditioner 402); default is direct.
- **Dead**: `_joint_objective_from_bin_etas` (616-642) — see Suspect S1.

**`scop_exact_support.py` (31 lines)** — `build_exact_scop_support` (26): collapse duplicate rows of a dense SCOP basis via `np.unique(axis=0)` into `ExactSCOPSupport` (9) with bincount-aggregated weighted products; used for a *single* non-discretized SCOP group (irls_direct.py:913).

### 1.4 Structured backend (large-K random effects / factor smooths)

**`structured.py` (253 lines)** — pure re-export facade ("Compatibility facade for structured solver internals", structured.py:1) flattening `_structured/*` into one namespace; `__all__` lists 100+ names including ~60 underscore-private helpers (structured.py:136-253).

**`_structured/selection.py` (647 lines) — backend eligibility & cost policy.**
- `resolve_structured_backend` (390): called once per direct fit (irls_direct.py:615). Selects a dominant group (`select_structured_group` 194: at most one FactorSmooth, else largest RandomEffect; SCOP/constraints disqualify), applies measured cost crossovers (`_structured_auto_is_beneficial` 47: p ≥ 32 and structured/dense cubic-cost ratio ≤ 0.75; block variant 73; SZ variant 108), validates `S_override` block structure via `_structured_override_incompatibility` (overrides.py:74), and rejects intercept-aliased zero-penalty random effects (496-533) and singular factor-smooth local blocks (552-637, with weight-digest memoisation `_factor_smooth_singular_local_level` 295 caching on the GroupMatrix by blake2b hash of W).

**`_structured/layout.py` (364 lines)** — `ScalarStructuredLayout` (21) / `BlockStructuredLayout` (46): frozen coefficient partitions (small vs structured indices), plus either a fused dense small matrix (small width ≤ `_MAX_FUSED_DENSE_SMALL_WIDTH` = 32, layout.py:71) or a `MatrixExecutionPlan` for the small block. Cached on the DesignMatrix (`dm._scalar_structured_layout_cache`, layout.py:181/276) keyed by group geometry — reused across REML candidate fits. `structured_design_matvec/rmatvec` (309/338).

**`_structured/moments.py` (436 lines)** — per-iteration sufficient statistics. `build_structured_system` (402) dispatches to `build_scalar_structured_system` (87: small Gram A (q_s,q_s), cross C (K,q_s) via `_random_effect_cross_gram`, level diagonal d (K,) via bincount sufficient stats) or `build_block_structured_system` (203: block D (K,k,k), C (K,k,q_s); SZ basis gets raw all-level moments + adjoint-transformed public moments, 352-379). Outputs `ScalarStructuredSystem` (39) / `BlockStructuredSystem` (54) / `SumToZeroBlockStructuredSystem` (69). Never materialises the K×K (or Kk×Kk) block.

**`_structured/operators.py` (1019 lines)** — compact symmetric operators: `SymmetricBlockOperator` (18: A, C, d), `BlockSymmetricOperator` (72: A, C (K,k,q_s), D (K,k,k)), `SumToZeroBlockOperator` (148), `CenteredBlockOperator` (231), `LowRankSymmetricOperator` (292), `SumBlockOperator` (320); plus diagonal-plus-low-rank (`_DiagonalLowRank` 359) and block-diagonal-plus-low-rank (`_BlockDiagonalLowRank` 547) calculi with trace/diag/product kernels (`_trace_symmetric_dlr` 454, `_multiply_symmetric_bdlr_coalesced` 830, `_general_bdlr_square_diagonal` 918, ...) used by factors for EDF/REML traces in O(Kq_s² + q_s³) instead of O(p³). `materialize_compact_operator` (986) is the dense escape hatch (used by `DenseHessianFactor`).

**`_structured/factors.py` (1396 lines)** — Schur factorizations implementing the `HessianFactor` protocol:
- `ScalarSchurFactor` (72): eliminate the diagonal d block; dense-small Schur complement Q = A − C'D⁻¹C, Cholesky with probe-residual check and absolute cancellation floor (146-172), SVD fallback with coupled-null-space rejection `_reject_coupled_schur_null_space` (43); logdet = Σlog d + logdet Q (205); `solve` (227), selected inverse blocks/diagonals (280/314), penalty/operator traces via DLR algebra (403-512). `max_structured_inverse_block=256` guard (289) refuses dense K×K inverse blocks.
- `BlockSchurFactor` (515): same with per-level k×k Cholesky/inverse (`np.linalg.cholesky(self.D)` batched, 586) and residual checks (592-603).
- `ProfiledBlockSchurFactor` (981) / `ProfiledScalarSchurFactor` (1165): intercept-profiled views (shift indices by 1, logdet − log sum_w).
- Constructed via `_structured/assembly.py` and consumed by irls_direct terminal geometry (irls_direct.py:2310-2358) and REML.

**`_structured/assembly.py` (908 lines)** — penalty assembly + cached λ-solves:
- `build_penalized_scalar/block/sum_to_zero_operator` (100/225/395): add λ·Ω per `PenaltyComponent` (or legacy per-group omega, or validated `S_override`) directly onto compact A/d/D — no p×p S is formed.
- `build_augmented_scalar/block/sum_to_zero_factor` (596/643/693): prepend the unpenalised intercept row/col into the small block and factor → `(ScalarSchurFactor|BlockSchurFactor|SumToZeroBlockFactor, rhs (p+1,))`.
- `solve_cached_*` (763-908): λ-only re-solves against **cached** working moments — the fREML "performance iteration" fast path (called from `reml/discrete.py:926,1023`). Returns `Cached*StructuredSolution` (44/56/68) with beta, intercept, profiled factor, logdet, rank.

**`sum_to_zero.py` (966 lines) — constrained SZ factor.**
- `SumToZeroBlockFactor` (304): factors the all-level (K blocks) system under the exact Σ_levels β_level = 0 constraint via a bordered KKT system: per-level PSD pseudo-inverses/null bases `_decompose_local_psd_batch` (57, batched eigh with dimension-scaled threshold), border = [[Q, E, −R'], [E', 0, N'], [−R, N, −M]] (403-413), constraint-covariance equilibration `_constraint_equilibration` (138), LDL border factor with residual-checked SVD fallback `_SymmetricBorderFactor` (158), inertia check → `SumToZeroIdentifiabilityError` (428-433, the trigger for fit_irls_direct's gram retry). Solve scatters through per-level pinv + border solve (490-539); BDLR inverse representation `_inverse_bdlr` (607) for traces; `raw_level_inverse_block` (597) for per-level inference.
- `ProfiledSumToZeroBlockFactor` (769): intercept-profiled wrapper.

**`_structured/geometry.py` (1516 lines)** — estimability/null-space certification for structured factors (bounded centered estimability, Ritz-based spectral bounds for SZ public geometry, lifted null row norms, etc.). Post-fit reporting infrastructure, called from factors (`_coefficient_estimable_from_null_basis`, factors.py:12) and `_reporting_state`; not on the per-iteration hot path.

**`_structured/state.py` (68 lines)** — `StructuredLinearSystemState` (37): the authoritative retained-fit bundle (coefficient/profiled/augmented factors + system + penalized operator + centered data operator + support totals), constructed in `model/reml_finalize.py:137`.

**`__init__.py` (6 lines)** — exports `fit_pirls`, `fit_irls_direct`, `PIRLSResult`, `IterationDiagnostics`, `QPResult`, `solve_constrained_qp`.

---

## 2. DATA FLOW

### 2.1 PIRLS (BCD) path — `_fit_pirls_inner`

1. **Setup**: beta (p,) zeros or warm start; intercept from `initial_mean`; S (p,p) built once from `reml.penalty_algebra.build_penalty_matrix` (pirls.py:654) only if a structural smoothing penalty exists (`_has_structural_smoothing_penalty`, pirls.py:58) or `S_override` given; else S = None (branch-free selection-only fast path).
2. **Committed state**: `_evaluate_irls_state` materialises eta/eta_unclipped/mu (each (n,)) and deviance; penalized deviance = D + β'Sβ + 2·penalty.eval (pirls.py:720-731).
3. **Outer iteration** (pirls.py:809): from the committed snapshot compute W = w·(dμ/dη)²/V (n,), z (n,); build **per-group Hessians** `X_g'WX_g + S_gg` via `gm.gram(W)` (pirls.py:831) — this is the per-iteration data pass, O(Σ n·p_g²) dense or bin-level for discretized groups. Then penalty-specific block preparation: GroupLasso/GroupElasticNet → full eigendecomposition of every block (`_radial_block_eigensystems`, pirls.py:839); Ridge → Cholesky factors per block (859); generic → largest eigenvalue for Lipschitz step (855).
4. **Inner BCD** (pirls.py:881, ≤ max_iter_inner=5 cycles): residual r = z − Xβ − intercept − offset (n,) updated incrementally; closed-form intercept step (892); per group: gradient `−X_g'(W r) + (Sβ)_g` (907), then exact Ridge solve, exact radial solve (`_solve_radial_block` with brentq root-finding on the secular equation), or prox-gradient step `penalty.prox_group` (911-938); rank-1 residual and `S_beta` updates (941-946). Residual refreshed from scratch every 5 cycles (883). **Active set** (opt-in flag): a zeroed group is skipped in later cycles iff it is a fixed point of its own prox (`zero_probe`, 950-965) — one extra rmatvec per zeroed group.
5. **Line search**: full proposal evaluated (976); `_select_irls_trial` halves along the fixed committed→proposal segment interpolating beta/intercept/eta simultaneously (986-1014); atomic rejection restores the committed snapshot.
6. **Convergence**: deviance-relative or coefficient criterion, then a mandatory **KKT check** `_composite_kkt_violation` (1048): one extra full row pass (rmatvec) + per-group prox probes, only when the stagnation test first passes.
7. **Terminal assembly** (1234-1391): group selection from final nonzero blocks; a *selected* sub-DesignMatrix is built; `build_centered_system` at final W; three `RankDecomposition`s (data / augmented / coefficient), each escalated to a streamed factor certification when `needs_factor_certification` (1292-1319); EDF = diag(H_aug⁺ · G_data) if inference curvature present else Breheny–Huang group-lasso allocation (1323-1346); Pearson phi (1372-1374); `RankInfo` + `PIRLSResult`.

Arrays materialised per outer iteration: W, z, r (n,) each; per-group Hessians Σp_g² floats; trial states hold beta (p,) + 3×(n,) arrays per cached alpha.

### 2.2 Direct IRLS path — `_fit_irls_direct_once`

1. **Backend resolution** (irls_direct.py:615-647): `resolve_structured_backend` picks structured/gram; `direct_solve="qr"` forces the dense QR path (disabled when constraints/SCOP present, 1024); constraints force the raw-Gram QP branch; SCOP forces the reduced-system branch.
2. **Penalty**: dense S (p,p) via `_build_penalty_matrix` unless structured (penalty then goes directly into compact operators; `penalty_matvec` closure applies components without expanding identity random-effect blocks, 671-690).
3. **Per iteration** (1281): `coefficient_working_rows` → W, z (n,). Then one of:
   - **Gram (default)**: `get_centered_system` (1082) → `build_centered_system` = one data pass O(n·p²) dense / O(n_bins·K²) discretized / tabmat-accelerated; for Gaussian-identity and Gamma-log the whole `CenteredSystem` is cached and only the RHS refreshed (`refresh_centered_rhs`, 1091-1097). Solve: `decompose_gram(centered.hessian)` O(p³); near the resolution boundary escalate once per geometry to a streamed factor+RHS QR certificate (`certify_centered_factor`, 1113-1172, cached by CenteredSystem identity); intercept recovered as mean_z − mean_x·β (1573).
   - **Structured**: `build_structured_system` (one data pass, compact moments) + `build_penalized_structured_operator` + `build_augmented_structured_factor` → Schur solve; never forms p×p (1511-1548).
   - **QR**: stack `sqrtW·(X_full − mean_x)` over `sqrt(S)` and SVD-decompose the (n+p, p) factor each iteration (1336-1352).
   - **Constrained QP**: raw augmented (p+1,p+1) moments via the execution plan, intercept profiled analytically, `solve_constrained_qp` with warm-started active set (1578-1633).
   - **SCOP**: subtract SCOP eta from z, solve the *reduced* non-SCOP system in centered (or anchor-centered) coordinates (1375-1432), then `scop_joint_newton_step` on the latent blocks (1446-1461); trial state = `_SCOPTrialState` interpolated in latent coordinates (1692-1725).
4. **Line search**: same `_select_irls_trial`; non-SCOP path passes the `_stable_penalized_deviance_delta` polarization merit (1811-1815).
5. **Curvature controller** (1886-2102): a Gamma/log step rejection activates observed-Newton rows for subsequent proposals ("curvature_rescue"); a rejection *under* observed Newton falls back to Fisher permanently ("curvature_fallback"); both clear the constant-weight caches.
6. **Terminal geometry** (2183-2475): recompute W/z at the retained model (skipped when `_return_working_system=True` — the fREML performance-iteration mode that deliberately reuses the last working system); rebuild the centered/structured system; `cache_out` exports raw + centered moments so the cached-W fREML optimizer can re-solve with new S in O(p³) with **zero** data passes (2269-2295, consumed by `reml/discrete.py` via `solve_cached_structured`); REML geometry: profiled slope inverse `XtWX_S_inv` (dense pseudo-inverse or Profiled*SchurFactor implementing `HessianFactor`), `log_det_H = log(sum_w) + log|H_c|` (2374); EDF via diag(H⁺G); optional full `RankInfo` (dense only — structured returns `rank_info=None` by design, 2359); Pearson phi; SCOP post-fit inference installed exactly once (2527-2590).
7. Returns `(PIRLSResult, XtWX_S_inv[, XtWX][, scop_state])`.

### 2.3 State hand-offs to REML

- `cache_out` dict: XtWX/centered_XtWX or structured system+operators, XtWz, XtW1, sums, means (irls_direct.py:2269-2295).
- `return_xtwx=True` gives the raw slope Gram/operator for cheap W-fixed REML iterations.
- `scop_state_init` / returned `scop_converged` dict warm-starts SCOP latent state across EFS outer iterations (irls_direct.py:874-945, 2504-2523).
- `beta_init`/`intercept_init`/`_deviance_init` warm-start coefficients between λ trials.

---

## 3. STATE OBJECTS

| Object | File:line | Fields / role | Lifecycle | Overlap notes |
|---|---|---|---|---|
| `SolverState` (`_IRLSState`) | irls_state.py:17 | beta, intercept, eta_unclipped, eta, mu, deviance, penalized_deviance, trace ids | per evaluation, frozen | canonical; alias `_IRLSState` retained (irls_state.py:37) |
| `_IRLSStepDecision` | irls_state.py:40 | alpha, halvings, rejected | per outer iteration | — |
| `IterationDiagnostics` | pirls.py:125 | 40+ fields incl. duplicated `w_min`/`raw_w_min` etc. | opt-in log | both solvers fill it; `cond_estimate`/`used_svd_fallback` only meaningful for direct |
| `PIRLSResult` | pirls.py:175 | coefficients + geometry + publication lock | fit result; `_publish` freezes | carries SCOP geometry slots used only by direct solver (pirls.py:201-202) |
| `RankPolicy`/`RankDecomposition`/`RankInfo` | rank.py:14/134/297 | rank policy, factorization, fitted-subspace metadata | per decomposition / per fit | `RankInfo.augmented` duplicates what `HessianFactor` provides on the structured path (which sets rank_info=None) |
| `CenteredSystem` | centered_system.py:33 | sum_w, mean_x, mean_z, data_gram, rhs, penalty, hessian — all frozen | per iteration (or cached whole fit for constant-W) | instance identity doubles as RHS-generation id for factor certificates (irls_direct.py:1131-1141) |
| `TabmatCenteringState` | centered_system.py:53 | eligible / raw_spline_eligible tri-state | fit-local mutable | — |
| `_CenteredFactorCertification` | irls_direct.py:186 | system, factor, decomposition, transformed rhs | cached per geometry | — |
| `CoefficientWorkingRows` | working_rows.py:16 | W, z, curvature source, fallback reason | per iteration | — |
| `QPResult` | constrained_qp.py:29 | beta, active set | per QP solve; active set threaded across IRLS iterations | — |
| `_SCOPGroupSpec` / `_SCOPGroupState` / `_SCOPTrialState` | irls_direct.py:154/166/178 | static spec vs dynamic latent state vs joint trial | per fit / per trial | **duplicates** the mutable `_scop_state: dict[int, dict]` (irls_direct.py:874); explicit sync code copies between them (1748-1760, 2504-2523) |
| `SCOPNewtonResult`, `_JointObjectiveCache`, `_SCOPPrototypeConfig` | scop_newton.py:50/83/37 | step result, objective caches, global mutable prototype config | per step / module-global | prototype config is process-global mutable state |
| `ExactSCOPSupport` | scop_exact_support.py:9 | unique rows + row map | per fit | — |
| `ScalarStructuredLayout` / `BlockStructuredLayout` | layout.py:21/46 | index partitions, fused small matrix or execution plan | cached on DesignMatrix across REML trials | two near-identical dataclasses (only `structured_indices` shape differs) |
| `ScalarStructuredSystem` / `BlockStructuredSystem` / `SumToZeroBlockStructuredSystem` | moments.py:39/54/69 | compact operator + xtw/xtwz/sums | per iteration; final one cached in cache_out | — |
| `SymmetricBlockOperator` family | operators.py:18/72/148/231/292/320 | compact A/C/d(D) blocks | per penalty assembly | — |
| `_DiagonalLowRank` / `_BlockDiagonalLowRank` (+General variants) | operators.py:359/547 | inverse representations for trace algebra | cached on factors | — |
| `ScalarSchurFactor` / `BlockSchurFactor` / `SumToZeroBlockFactor` + `Profiled*` | factors.py:72/515/981/1165; sum_to_zero.py:304/769 | Schur/bordered factorizations implementing `HessianFactor` | per solve; terminal ones retained | ScalarSchurFactor and BlockSchurFactor share ~200 lines of near-identical Q-factor logic (factors.py:137-209 vs 615-690) |
| `_LocalPSD`, `_SymmetricBorderFactor` | sum_to_zero.py:33/158 | per-level pinv/null; bordered LDL | inside SZ factor | — |
| `StructuredLinearSystemState` | _structured/state.py:37 | authoritative retained factors bundle | built at finalize (model/reml_finalize.py:137) | — |
| `Cached*StructuredSolution` | assembly.py:44/56/68 | λ-only re-solve results | per REML trial | — |
| `StructuredGroupSelection` / `StructuredBackendDecision` | selection.py:24/33 | backend choice + fallback reason | once per fit | — |
| `HessianFactor` protocol / `DenseHessianFactor` | hessian_factor.py:86/136 | REML/inference operation contract | adapter | — |
| Untyped side channels | irls_direct.py:396-397 | `profile: dict`, `cache_out: dict` | per fit | stringly-keyed contracts consumed by reml/discrete.py |

---

## 4. COMPLEXITY TABLE

Per-iteration unless noted. Dense counts assume no discretization; discretized groups replace n with n_bins for Gram work but keep O(n) bincounts.

| Routine | Time | Memory | Notes |
|---|---|---|---|
| `_evaluate_irls_state` (irls_state.py:63) | O(n·p) matvec + O(n) deviance | 3×(n,) new frozen arrays | called ≥2× per outer iteration (proposal + committed), + per halving trial |
| pirls `_build_group_hessians` (pirls.py:342/831) | O(Σ n·p_g²) = O(n·p·max p_g) | Σp_g² | **rebuilt every outer iteration**, even for constant-W families (no cache analogous to irls_direct's) |
| pirls `_radial_block_eigensystems` (pirls.py:414) | O(Σ p_g³) eigh | Σp_g² | every outer iteration for GroupLasso/GEN |
| pirls inner BCD cycle (pirls.py:881-965) | O(Σ n·p_g) = O(n·p) per cycle (matvec/rmatvec per group) + prox | (n,) residual | ≤5 cycles/outer; radial solve adds brentq iterations O(p_g) each |
| pirls `_composite_kkt_violation` (pirls.py:479) | O(n·p) | (n,) | only when stagnation test passes |
| pirls terminal ranks (pirls.py:1286-1319) | 3× `decompose_gram` O(p³); certification adds streamed QR O(n·p²) | p² each; chunked (≤8192, p) dense rows | once per fit |
| `build_centered_system` (centered_system.py:184) | O(n·p²) dense; O(n·p + n_bins·k²) discretized/tabmat; + trial Cholesky O(p³) + possible eigh O(p³) PSD projection | p² gram + p² hessian | per IRLS iteration (direct gram path) unless constant-W cached |
| `refresh_centered_rhs` (centered_system.py:147) | O(n·p) | (p,) | constant-W path |
| `decompose_gram` (rank.py:453) | Cholesky O(p³); eigh fallback O(p³); **representative-pivot loop worst case O(p⁴)** (rank.py:645-656: eigvalsh of growing principal minors inside a p-loop) | p² | per iteration on direct gram path |
| `decompose_factor` (rank.py:725) | SVD O(r·p²) for (r,p) factor; QR path factor is (n+p, p) → **O(n·p²) per iteration** under `direct_solve="qr"` | factor copy | representative loop also O(p⁴) worst case (rank.py:789-798) |
| `streamed_weighted_factor(_rhs)` (rank.py:43/64) | O(n·p²) total, chunked QR | ≤ (chunk+p, p) | rare certification only |
| irls_direct gram solve (1549-1577) | O(p³) | p² | |
| structured moments `build_structured_system` (moments.py:87/203) | O(n) bincounts + O(n·q_s²) small gram + O(n·q_s) cross (dense) or cell-level for discrete | Kq_s (scalar) / K·k·q_s (block) | per iteration |
| `ScalarSchurFactor.__init__` (factors.py:77) | O(K·q_s + K·q_s²) elimination + O(q_s³) Cholesky/SVD | K·q_s (F) + q_s² | per iteration |
| `BlockSchurFactor.__init__` (factors.py:520) | O(K·k³) batched local Chol/inv + O(K·k·q_s²) + O(q_s³) | K·k² + K·k·q_s + q_s² | per iteration |
| `SumToZeroBlockFactor.__init__` (sum_to_zero.py:309) | O(K·k³) batched eigh + O(K·k·q_s²) border assembly + O((q_s+null+k)³) LDL | border² + K·k·q_s | per iteration; `_border_inverse` adds O(border³) once when traces requested |
| `solve_cached_structured` (assembly.py:873) | O(K·k³ + q_s³) — **no data pass** | compact | fREML λ trials |
| `solve_constrained_qp` (constrained_qp.py:57) | per active-set iteration: dense KKT solve O((p+a)³) rebuilt from scratch + blocking scan O(m·p) with `i in active` list scan O(m·a) | (p+a)² | warm-started; a = active count |
| `scop_newton_step` (scop_newton.py:228) | O(n·q_eff) or O(n + n_bins·q_eff²) + O(q_eff³) | q_eff² | per IRLS iteration |
| `scop_joint_newton_step` (scop_newton.py:773) | + cross grams: disc×disc O(n + nb_i·nb_j) 2-D hist; **mixed dense×disc scatters (n, q_eff)** (733-770); halving trials O(Σq_i²) via gamma cache | Σq_i² joint H | per IRLS iteration; final refresh runs one extra joint step at fit end (irls_direct.py:2131-2155) |
| `_fit_irls_direct_once` terminal geometry (2303-2451) | dense: O(p³) pseudo-inverse + O(p³) coefficient/data ranks; structured: O(K·k³ + q_s³) | p² (dense) | once per fit; skipped under `_compute_reml_geometry=False` |
| `PIRLSResult._publish` / `_freeze_result_arrays` (pirls.py:215/273) | O(total result bytes), `tobytes` **double copy** per array (pirls.py:267-270) | duplicate of all result arrays incl. O(p²) RankInfo matrices | at publication boundary |
| QR path setup (irls_direct.py:1036) | `np.hstack([gm.toarray()...])` O(n·p) materialisation; per iteration `A_data` + vstack copies O(n·p) | (n,p) retained whole fit + (n+p, p) per iteration | direct_solve="qr" only |

---

## 5. SUSPECTS

**S1. Dead + broken: `_joint_objective_from_bin_etas` (scop_newton.py:616-642).** References `cache.wz_aggs`, `cache.w_aggs`, `cache.w_2ds` — fields that do not exist on `_JointObjectiveCache` (scop_newton.py:83-89: only `half_zwz`, `btwz`, `diag_btwb`, `cross_btwb`). Zero call sites in src. Would raise `AttributeError` if ever called. Verify: grep confirms only the definition; remove or fix in a later phase.

**S2. Dead helpers in irls_direct.** `_solve_profiled_intercept_from_h_inv` (irls_direct.py:299) has no callers anywhere. `_robust_solve` (283) is called only by `tests/test_robust_solve.py`. `_sqrt_penalty_augmented` (334) builds a (p+1,p+1) matrix whose intercept row/col is always zero and is only ever consumed as `_L_aug[1:, 1:]` (irls_direct.py:1344) — the augmentation is vestigial. Verify: confirm no dynamic access, then classify as declutter candidates.

**S3. Two IRLS orchestrations with heavily duplicated scaffolding.** The split pirls (proximal/L1) vs irls_direct (smooth/REML) is architecturally intentional (docstring irls_direct.py:13-16), but the *scaffolding* is duplicated nearly verbatim: `evaluate_state`/trace-emission closures (pirls.py:689-792 vs irls_direct.py:743-828), fixed-endpoint trial closures (pirls.py:986-1007 vs irls_direct.py:1782-1804), step-decision/termination-reason/state-commit blocks (pirls.py:1028-1107 vs irls_direct.py:1841-1944), IterationDiagnostics filling (pirls.py:1126-1180 vs irls_direct.py:1959-2008), final W/z recomputation + Pearson phi (pirls.py:1282-1374 vs irls_direct.py:2186-2190, 2469-2475). Divergence risk is real: e.g. only irls_direct uses the `_stable_penalized_deviance_delta` merit (1811); pirls line search compares raw penalized deviances (irls_state.py:171) and could reject terminal steps in ill-conditioned smooth bases for the reason documented at irls_direct.py:121-127. Verify: whether pirls fits with large S ever hit spurious `step_rejected`.

**S4. `_fit_irls_direct_once` is a ~2100-line function** (irls_direct.py:473-2601) with ≥5 interleaved backends (gram/qr/structured/constrained/SCOP), 13 leading-underscore keyword switches, and mode-interaction guards enforced by runtime ValueErrors (854-861). Responsibility boundaries (fit vs REML-geometry export vs SCOP inference installation) live in one scope. This is the subsystem's main comprehension/maintenance hotspot. Verify: cross-mode invariants are only covered by the two `raise ValueError` guards.

**S5. Sequential SCOP path retained "for parity comparison".** `_scop_joint=True` is the only production value; the `else` branches (irls_direct.py:1462-1496 and 2156-2181) duplicate the Gauss–Seidel update and are never exercised from src (`_scop_joint=False` appears nowhere outside tests). Dual maintenance of the same algebra.

**S6. Duplicated SCOP dynamic state.** The mutable `_scop_state: dict[int, dict]` (irls_direct.py:874, stringly-keyed: "beta_scop", "gamma_eff", "H_scop_penalized", "penalty_rank", ...) coexists with the frozen `_SCOPGroupSpec`/`_SCOPGroupState` dataclasses; explicit copy-synchronisation runs after every accepted step (1748-1760) and again for the EFS export (2504-2523). Two representations of the same latent state with hand-written sync is an invariant-drift risk. Verify: whether any key written by scop_efs (`penalty_eigvals_omega` etc., 936-944) can go stale relative to the dataclass path.

**S7. Rank-deficiency representative-selection loops are O(p⁴)/O(p_active⁴) worst case.** `decompose_gram` (rank.py:645-656) and `decompose_factor` (rank.py:789-798) run `eigvalsh` on growing principal minors inside a Python loop over candidate columns to pick reproducible alias representatives. For a nearly-rank-deficient wide model this is a quartic blowup executed **per IRLS iteration** on the gram path (via `decompose_gram(centered.hessian)`, irls_direct.py:1554). Verify: measure with p≈200 and one exact alias.

**S8. Repeated p×p factorizations of the same matrix per iteration.** On the dense gram path a single iteration can factor the (p,p) Hessian up to three times: PSD-guard `np.linalg.cholesky` inside `build_centered_system` (centered_system.py:284, result discarded), then `decompose_gram`'s Cholesky + `pocon` + probe solve (rank.py:487-513), plus an eigh if the guard fails. p is small (~50-80) so this is cheap today, but it is pure duplication on the hottest loop. Verify with profile counters.

**S9. PIRLS rebuilds block Hessians + eigensystems every outer iteration with no constant-weight reuse.** `_build_group_hessians` + `_radial_block_eigensystems` (pirls.py:831-862) rerun O(n·Σp_g²) + O(Σp_g³) per outer iteration even for Gaussian/identity where W never changes — the direct solver has `_has_constant_irls_weights` caching (irls_direct.py:256, 1054) but pirls has none. Verify: profile a Gaussian group-lasso path fit; Gram time should dominate.

**S10. Layering inversion: solvers import from `reml`.** pirls imports `reml.penalty_algebra.build_penalty_matrix` (pirls.py:654), irls_direct imports `reml.penalty_algebra` (323, 684), `reml.observed_geometry.classify_scop_reml_curvature` (851), and `reml.scop_geometry` (2533) — while `reml/*` imports solvers back (`reml/efs.py:35` imports `_safe_decompose_H`; `reml/runner.py:39` imports `_invert_xtwx_plus_penalty`). Private solver helpers used cross-package are de-facto public API. Verify import cycles are only broken by function-local imports.

**S11. Compat shims / migration residue.** (a) `structured.py` is a 253-line facade re-exporting ~60 underscore-private names in `__all__` (structured.py:136-253) — private helpers made importable package-wide. (b) `_IRLSState = SolverState` alias "retained while direct IRLS and downstream private callers move" (irls_state.py:35-37) — the migration never completed; both files use `_IRLSState` throughout. (c) `_build_penalty_matrix`, `_safe_decompose_H`, `_robust_solve` all self-describe as "backward-compatible/compatibility wrappers" (irls_direct.py:322, 287, 377).

**S12. Doc–code mismatches.** (a) `_fit_irls_direct_once` docstring: "tol : Deviance convergence tolerance (default 1e-6)" but the signature default is `1e-8` (irls_direct.py:485 vs 544). (b) Module docstring "p is ~50-80" and "the 33-iteration aliasing from shared B matrices... vanishes" (irls_direct.py:8, 15-16) are stale historical narrative, not current contracts. (c) `IterationDiagnostics` carries `raw_w_min/raw_w_max/raw_w_ratio` always set identical to `w_min/w_max/w_ratio` in both solvers (pirls.py:1141-1155; irls_direct.py:1966-1982) — vestigial duplicate fields. (d) irls_direct `_t_eta` timing accumulator is initialised and reported into `profile["irls_eta_s"]` but never incremented (irls_direct.py:1188, 2198) — always 0.0, a misleading metric.

**S13. Potential NameError at max_iter=0.** Both `_fit_pirls_inner` (uses `retained`, `dev`, `outer`, `termination_reason` after the loop, pirls.py:1218, 1376-1391) and `_fit_irls_direct_once` (uses `it`, `retained`, `dev`, `step_rejected`, `termination_reason` after the loop, irls_direct.py:2118-2131, 2483-2499) reference loop-body locals unconditionally after `for ... in range(max_iter)`. `max_iter=0` (or `max_iter_outer=0`) raises `NameError` instead of a clean validation error. Verify: no callers currently pass 0, but it is unvalidated public surface.

**S14. `direct_solve="qr"` is an O(n·p²)-per-iteration landmine with full (n,p) materialisation.** `_X_full = np.hstack([gm.toarray()...])` (irls_direct.py:1036) plus a fresh (n+p+1, p) stack + full SVD via `decompose_factor(..., retain_factor_solve=True)` **every iteration** (1343-1348). The discretized-group warning exists (1030-1035) but the path also defeats the constant-weight cache and streamed chunking. Fine as an explicit escape hatch; worth confirming it is documented as such at the model layer (`model/base.py:670` accepts it silently).

**S15. Publication freezing double-copies all result arrays.** `_immutable_array_copy` copies via `tobytes` then `frombuffer` (pirls.py:267-270) — two full copies per array — and `_freeze_result_arrays` walks the entire result graph including O(p²) RankInfo decompositions at every `_publish`/deepcopy/unpickle (pirls.py:273-339). For large p this doubles peak result memory transiently. Verify: how often `_mutable_copy` is invoked by fitted-state revisions in model/state ops.

**S16. Constrained-QP robustness gaps.** `solve_constrained_qp` rebuilds the dense KKT from scratch each active-set change with no factor updates (constrained_qp.py:126-141), computes multipliers through the potentially ill-conditioned `A_eq A_eq'` normal equations (155-157), and `_project_feasible` caps at 100 sweeps without reporting failure (45-53). Blocking-constraint search uses `if i in active` — O(m·a) list membership per iteration (182-183). Acceptable at current m (a handful of monotone constraints) but the only unguarded dense `np.linalg.solve(H, g)` calls in the subsystem (97, 100, 117) bypass the shared rank policy entirely.

**S17. `_has_constant_irls_weights` (irls_direct.py:256) and `working_rows._fisher_rows` Gaussian/identity special-case (working_rows.py:46-52) encode the same family/link knowledge in two places** with exact-type checks; both must stay in sync for the constant-geometry cache to remain bit-valid (the comment at working_rows.py:46-50 documents the dependency in one direction only).

**S18. SCOP terminal Hessian refresh runs a full extra joint Newton step** (irls_direct.py:2131-2155) solely to obtain `H_scop_penalized` at the retained model — including its internal line search — rather than a Hessian-only evaluation. Cheap at typical q_eff but conceptually a solve used as a derivative probe; the returned `beta_new` is discarded.
