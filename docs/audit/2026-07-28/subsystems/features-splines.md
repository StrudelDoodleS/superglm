# Architecture audit — subsystem: features-splines

Target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ origin/master (f082e9b).
All paths below are relative to `src/superglm/` unless absolute.

Notation used throughout: **n** = rows, **u** = #unique values of a covariate (u ≈ n for continuous data), **b** = n_bins (discrete path, default 256), **K** = per-smooth raw basis dimension (≈ user `k`), **p** = total built columns, **L** = categorical levels, **m** = number of smooth terms, **q** = number of penalty components, **d** = spline degree.

---

## 1. MODULE MAP

### 1.1 `features/spline.py` (1015 lines) — public spline classes; pure delegation façade
Nearly every method is a one-line delegate into a `_spline_*` helper module; the class hierarchy holds capability flags and per-kind overrides only.

| Symbol | Lines | Role |
|---|---|---|
| `_weighted_quantile_knots` | 36–38 | "Compatibility wrapper" for `_spline_knots.weighted_quantile_knots`. No callers in src or tests (grep: only self-reference). Dead compat shim. |
| `_SplineBase` | 41–393 | Base spec. Capability metadata `_penalty_semantics/_max_penalty_order/_multi_m_supported/_select_supported/_tensor_supported` (96–101); `__init__` delegates to `_spline_config.initialize_spec` (127–160); `build()` → `_spline_build.build_group_info` (329–333); `build_knots_and_penalty()` → `_spline_build.build_knots_and_penalty` (335–354, discrete path); `transform/score/reconstruct` → `_spline_runtime` (356–368); `tensor_marginal_ingredients()` → `_spline_build.tensor_marginal_info` (370–393); identifiability (274–291), select eigendecompose (312–319), natural null space (293–310). |
| `_BSplineBase` | 396–418 | Open (padded) knot vector for PSpline/BSplineSmooth via `_spline_subclass_ops.assemble_open_knot_vector`. |
| `PSpline` | 421–519 | Difference penalty (`_build_penalty_for_order` 515–516). SCOP reparam hook for fit-time shape constraints (503–513). |
| `BSplineSmooth` | 522–623 | Integrated-derivative penalty (609–613); QP shape constraints (615–620). `_penalty_semantics="integrated_derivative"`. |
| `NaturalSpline` | 626–696 | Difference penalty + natural f''=0 constraints via Z projection (`_apply_constraints` 690–696); `_select_supported=False` (644). Only class using the base default clamped knot assembly (197–203). |
| `CubicRegressionSpline` | 699–811 | Integrated-f'' penalty + mandatory natural constraints (806–811); exact clamped knots (771–778); `_max_penalty_order=3`. |
| `CardinalCRSpline` | 814–951 | mgcv `bs="cr"` cardinal basis; own `_place_knots` (895–897) and matrices `_cr_knots/_cr_M/_cr_S` (891–893); "experimental… not yet the default for kind='cr'" (827–828). |
| `n_knots_from_k` / `Spline` factory | 959–1004 | Delegate to `_spline_factory`. |

**Callers of the classes** (grep): `dm_builder.py` (build path, lines 730–735, 857), `model/*`, `inference/*`, `editor/*`, `export/*`, `plotting/*`, `sklearn.py`, `diagnostics/*`, `features/ordered_categorical.py`, `features/interaction.py`, `features/factor_smooth.py`.

### 1.2 `_spline_config.py` (178) — init & validation
- `initialize_spec` (69–134): normalises constraint token, sets legacy mirrors `spec.monotone`/`spec.monotone_mode` (98–100, explicit compat), m-order tuple (103–107), explicit knots/boundary validation (109–133).
- `validate_m_orders` (14–27, phase-1 static), `validate_m_orders_build` (30–38, phase-2 after knot placement), `validate_select` (41–52).
- `_initialize_runtime_state` (157–178): zeroes all mutable build-time state (see §3).

### 1.3 `_spline_knots.py` (57) — knot placement
- `weighted_quantile_knots` (9–22): O(n log n) unique+cumsum.
- `resolve_interior_knots` (25–54): explicit / quantile / quantile_rows / quantile_tempered / uniform; silent fallback to uniform when quantiles collapse (50–51). Called from `_spline_runtime.place_knots:113` and `_spline_cardinal_spec.place_knots:26`.

### 1.4 `_spline_runtime.py` (192) — evaluation
- `prepare_eval_points` (18–35): clip / extend / error policies.
- `basis_matrix` (38–41): `scipy BSpline.design_matrix` → sparse, O(n·d).
- `raw_basis_matrix` (44–48): **densifies** via `.toarray()` → dense (n, K).
- `boundary_linear_rows` (64–82) + `linear_tail_basis_matrix` (85–99): linear-tail extrapolation caching.
- `place_knots` (102–123) and `assemble_clamped_knot_vector` (126–136, note the **1e-6·range pad**).
- `transform` (139–149): dense basis, then SCOP-sigma or `@ R_inv`.
- `score` (152–174): repeated-support fast path (unique+transform on support, thresholds at 13–15) else 8192-row chunked transform.
- `reconstruct` (177–192): 200-point grid.

### 1.5 `_spline_penalties.py` (52)
- `build_difference_penalty` (10–17): `np.diff(np.eye(K))`, O(K²)–O(K³).
- `build_integrated_derivative_penalty` (20–49): Gauss–Legendre per knot span; **inner Python loop constructs one `BSpline` object per (span, basis-fn)** → ~K² object constructions per call; `leggauss` recomputed inside the span loop (36).

### 1.6 `_spline_constraints.py` (54)
- Monotone/curvature difference constraint sets (12–27) for the QP engine.
- `build_natural_constraint_null_space` (30–47): 2×K constraint of f'' at boundaries, again one `BSpline` per basis fn (40–45), QR complete → Z (K, K−2).

### 1.7 `_spline_identifiability.py` (96) — sum-to-zero constraint (the k → k−1 contract)
- `build_identifiability_projection_for_spec` (55–72): `np.unique(x)` (O(n log n)), evaluates basis **dense at all u support points** (`.toarray()`, line 63), then `build_identifiability_projection` (11–30): counts@basis column-sum, complete QR of a (K,1) vector, drop first column → (K, K−1) projection.
- `apply_identifiability` (33–52): ω ← zᵀωz. Skipped when `absorbs_intercept` is False (select=True; spline.py:241–252).

### 1.8 `_spline_select.py` (164) — select=True (mgcv double penalty)
- `eigendecompose_select` (14–45): `eigh(ω_c)`, requires exactly 2 null eigenvalues (24–30), rotates constant out of null space → 1-D linear null block + range block.
- `build_select_group_info` (74–125): assembles combined `[U_null|U_range]` projection, `("null", ω_null)` + `("wiggle", ω_wiggle)` (or per-order `d{m}`) components; `lstsq(Z, U)` back-projections (92–93).
- `build_select` (128–156): entry from `build_group_info`; also sets `_interaction_projection` (137).
- `resolve_lambda_policies` (48–71): maps `LambdaPolicy`/dict to component names.

### 1.9 `_spline_multi_penalty.py` (31)
- `build_multi_m_components` (11–28): per order — rebuild penalty, re-apply constraints, **re-apply identifiability (which re-evaluates the u×K dense support basis each time)**.

### 1.10 `_spline_build.py` (245) — the build orchestrator
- Shape-constraint mode predicates with legacy `monotone` fallbacks (13–51).
- `build_group_info` (54–129): place knots → validate → sparse basis (73) → select branch (75–76) / SCOP branch (79–93) / normal: `_apply_constraints` → `_apply_identifiability` (95–96) → multi-m components (100–102) → QP constraints (107–112) → `GroupInfo` (114–124) → lambda-policy synthesis (125–128).
- `build_knots_and_penalty` (132–167): discrete-path variant "without building the full basis"; select eigendecompose (147–156); identifiability (158) — see suspect S1.
- `tensor_marginal_info` (170–245): raw dense basis at eval points (205), constraints, **inline re-implementation of the identifiability QR** (213–226), returns `TensorMarginalInfo`.

### 1.11 `_spline_factory.py` (165)
- `n_knots_from_k` (17–39): the **k contract**: ps/bs `n_knots=k−d−1` (open vector → n_basis=k), cr `n_knots=k−d+1` (clamped → n_basis=k+2, −2 natural constraints → k), cr_cardinal `n_knots=k−2` (K=k). Identifiability then removes 1 → **built columns = k−1** in every kind. Contract verified consistent.
- `Spline` (42–162): kind dispatch with three near-identical constructor call branches (106–162); rejects `constraint` for ns (88–92), rejects `k`+`n_knots` (83–86).

### 1.12 `_spline_subclass_ops.py` (77)
- `assemble_open_knot_vector` (13–27): 0.001·range padding + regular extension.
- `assemble_clamped_knot_vector` (30–39): exact-boundary clamped (CRS) — same name as `_spline_runtime.py:126` but **no pad** (see S9).
- `build_scop_reparameterization` (42–67): SCAM-style Σ transform, centering, stores `_scop_Sigma/_scop_null_dim/_scop_col_means` on spec.
- `build_shape_constraints_raw` (70–77).

### 1.13 `_spline_extrapolation.py` (75)
- `basis_value_and_slope_at` (11–27): per-basis-fn `BSpline` loop for slopes.
- `linear_tail_basis_matrix` (44–72): dense (n, K) rows then csr.

### 1.14 `_spline_cardinal.py` (120) + `_spline_cardinal_spec.py` (100)
- `build_cr_penalty_matrices` (cardinal.py 10–33): tridiagonal D, `solve(D, B_d)`, S = B_dᵀD⁻¹B_d — O(K³) once.
- `eval_cardinal_basis` (60–82): vectorised searchsorted evaluation, **builds dense (n, K) X** then wraps `sp.csr_matrix(X)` (82) — density is 100% by construction (each row mixes M rows), so the csr wrapper only adds overhead.
- `_spline_cardinal_spec.place_knots` (14–40) **duplicates** `_spline_runtime.place_knots` boundary/reset logic (17–24 vs runtime 104–111).

### 1.15 `interaction.py` (1290) — 7 interaction types
| Class | Lines | Design-matrix product |
|---|---|---|
| `SplineCategorical` | 32–267 | Per non-base level: one GroupInfo with `columns=None`, shared sparse raw basis `spline_cat_basis` + level mask; penalty projected through parent `_interaction_projection` (89–99). `build_discrete` (119–177) uses `_discretize_column` support + `B_unique`. `set_reparametrisation` splits a stacked R_inv per level (179–195). |
| `PolynomialCategorical` | 273–390 | Legendre basis × level indicator, dense; `_scale/_basis` re-implemented (294–301). |
| `NumericCategorical` | 396–485 | (L−1) masked-slope columns, dense. |
| `CategoricalInteraction` | 491–611 | (L1−1)(L2−1) sparse indicator columns via coo assembly (537–559). |
| `NumericInteraction` | 617–669 | single product column. |
| `PolynomialInteraction` | 675–768 | d1·d2 Legendre cross products; third copy of `_scale/_basis` (697–706). |
| `TensorInteraction` | 821–1290 | ti()-style tensor: `_marginal_from_spec` (873–976, incl. mgcv-contract check 887–900, CR→CardinalCR reroute 938–960, n_knots-override cloning 962–975, `inspect.signature` legacy-subclass shim 920–936); `build` (1111–1132): row-Kronecker `T = _row_kron(B1,B2)` + `kron(S1,I)+kron(I,S2)`; `build_discrete` (1134–1180): integer pair-encoding on observed support, `_row_kron_dense`; `_build_group_infos` (1049–1109): optional decompose into bilinear-null + wiggly subgroups via eigh; chunked/support-aware `score` (1196–1260). |
| helpers | `_row_kron` 774–791 (Python loop over k1·k2 sparse column multiplies), `_row_kron_dense` 794–796 (einsum), `_normalize_tensor_penalty` 799–810 (unit leading eigenvalue, "matching mgcv"). |

### 1.16 `factor_smooth.py` (512) — added with random effects in #165
- `_combine_qr_r` (24–35): streamed tall-skinny QR merge, chunk = 65 536 rows (20).
- `_natural_parameterization_from_r` (38–92): R⁻¹ via triangular solve, whiten penalty, `eigh(driver="evr")` for deterministic null rotation (60–67), scale penalized block by `sqrt(n·rank/Σ1/λ)` (78–82) and null block by `sqrt(n)`; emits ("wiggle", diag) + one `null_i` component per null dim (84–92).
- `FactorSmooth` (95–509): validation-heavy `__init__` (106–184); `_streaming_safe` (222–236) gates TSQR vs dense QR (`fs` with m>2, or heterogeneous per-null lambda policies → dense compat); `_initialize_marginal_spline` (238–265) builds an owned `Spline(kind="ps", k, penalty="none")` and **re-assembles knots on an expanded boundary** (256–263, deliberately diverging from ordinary P-spline knots for backwards compat, per comment 248–251); `_build_marginal` (267–328); `build` (358–374) exact (retains sparse basis), `build_discrete` (376–402) support basis + bin_idx; GroupInfo carries compact geometry (`factor_smooth_*` fields, 330–356); prediction/score/transform/reconstruct (404–509) incl. sz sum-to-zero expansion via `factor_smooth_geometry.expand_sum_to_zero_blocks` (476, 493–495).
- Consumed by `dm_builder._process_info` → `FactorSmoothGroupMatrix` (dm_builder.py:601–617), `model/base.py:403,586,805`, `inference/factor_smooths.py`, `export/rating_tables.py:61`.

### 1.17 `random_effect.py` (110)
- `RandomEffect.build` (39–61): `pd.factorize` all levels (no base drop), GroupInfo with `cat_codes` + `structured_kind="random_effect"`, `n_cols=L`. Score/transform via code lookup (79–97). Consumed as `RandomEffectGroupMatrix` (dm_builder.py:618–623).

### 1.18 `categorical.py` (193), `ordered_categorical.py` (563), `numeric.py` (48), `polynomial.py` (90), `grouping.py` (184), `constraint.py` (28)
- `Categorical.build` (84–148): factorize O(n), base by `bincount` exposure (118–122), remapped `cat_codes` with base = −1 → `CategoricalGroupMatrix`. `transform` (150–161) dense one-hot; `score` (163–179).
- `_validate_categorical_levels` (17–46): **pure-Python `any(...)` per-element NaN scan** (31) — see S12.
- `OrderedCategorical` (49–563): spline mode delegates to internal `Spline` clone (`_init_spline` 298–333, with n_knots clamping to L−1); deprecated step mode `_build_step` (394–444, one-hot + projected full-rank D1ᵀD1 penalty); large deprecation-warning matrix (180–221); legacy shortcut attributes mirrored back from the spline object (286–290).
- `Polynomial` (19–90): Legendre, degree columns.
- `grouping.py`: frozen `LevelGrouping` dataclass (17–37) + `collapse_levels` (40–184), pure metadata.
- `constraint.py`: frozen `ConstraintSpec` + `Constraint.fit/postfit.*` token namespace.

### 1.19 Caller side (for orientation; owned by another subsystem)
`dm_builder.build_design_matrix` (667–1093): exact path calls `spec.build` (857); discrete path calls `spec.build_knots_and_penalty` + `spec._raw_basis_matrix(bin_centers)` (730–735); **re-implements the whole select=True GroupInfo assembly inline** (791–841); interactions dispatched by `should_discretize_*` (947–986); `_process_info` (400–651) converts GroupInfo → GroupMatrix flavours and computes R_inv. `rebuild_design_matrix_with_lambdas` (1202+) only recomputes R_inv inside the REML loop — feature `build()` runs once per fit.

---

## 2. DATA FLOW

### 2.1 Exact 1-D smooth (fit)
```
x (n,) ─ build_group_info (_spline_build.py:54)
  ├─ place_knots: interior knots (O(n log n) for quantile strategies), knot vector (K+d+1,)
  ├─ B = BSpline.design_matrix(x) sparse csr (n, K), d+1 nnz/row     [_spline_runtime.py:38]
  ├─ ω raw (K, K) dense                                              [_spline_penalties.py]
  ├─ constraints: Z (K, K−2) for ns/cr; ω ← ZᵀωZ                     [spline.py:690,806]
  ├─ identifiability: unique(x) → support (u,), DENSE basis (u, K)   [_spline_identifiability.py:62–63]
  │    projection (K, K−1) (composed with Z when present); ω ← zᵀωz
  └─ GroupInfo(columns=B sparse (n,K), n_cols=k−1, penalty (k−1,k−1), projection (K,k−1))
dm_builder._process_info: R_inv (K, k−1) = projection @ SSP factor → SparseSSPGroupMatrix
  keeps B sparse and folds projection+SSP into R_inv (never materialises B@R_inv).
```
Per fitted model: m sparse (n, K_i) bases + m dense (K_i, k_i−1) R_invs; p = Σ(k_i−1) + categorical cols + interactions.

### 2.2 Discrete (fREML) 1-D smooth
```
dm_builder.py:730–736:
  ω, n_cols, projection = spec.build_knots_and_penalty(x)   # no full-x basis... except S1
  bin_centers, bin_idx = _discretize_column(x, b)           # (b,), (n,) intp
  B_unique = spec._raw_basis_matrix(bin_centers)            # DENSE (b, K)
  exposure_agg = bincount(bin_idx, sample_weight)           # (b,)
→ GroupInfo(columns=None) → DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
```
The only O(n) objects retained are `bin_idx` and `exposure_agg`. **But** `build_knots_and_penalty` internally evaluates a dense (u, K) support basis for the identifiability constraint (S1), so the exact-vs-discrete column geometry matches by construction (same ω, same projection) — good for the "no silent drift" contract, costly for memory.

### 2.3 select=True
Exact: `build_select` (`_spline_select.py:128`) → GroupInfo(columns = raw B sparse (n,K), projection = [U_null(K,1)|U_range(K,K_c)] so n_cols = k−1, penalty_components = null + wiggle/d{m}).
Discrete: `build_knots_and_penalty` populates `_U_null/_U_range/_omega_range` on the spec (spline.py:312–319), then **dm_builder re-assembles the same components inline** (791–841).

### 2.4 Tensor (`ti`) interaction
Exact: per margin `tensor_marginal_info` → dense centered basis (n, K_eff), penalty (K_eff, K_eff), projection (K_raw, K_eff); `T = _row_kron(B1,B2)` "(sparse)" (n, p1·p2) where p_i = K_i,eff; ω = kron(S1,I)+kron(I,S2) dense (p1p2, p1p2). Because the centered marginals are dense after projection, T is a ~100%-dense CSR (S4).
Discrete: two `_discretize_column` calls, marginals evaluated on supports (b_i, K_eff), joint observed pairs encoded as int64 codes (1159–1164), `B_joint` dense (n_pairs, p1p2), factored parts kept in `DiscreteTensorBuildResult` for `DiscretizedTensorGroupMatrix`. `_maybe_apply_tensor_side_constraints` (dm_builder.py:1158–1199) may further project the block by an SVD null space of cross-Grams against overlapping tensor terms.

### 2.5 SplineCategorical
Shares one raw basis across levels: exact keeps B sparse (n, K) once + per-level boolean mask (n,), penalty projected to (k−1, k−1); columns are never materialised per level (GroupInfo.columns=None; comment interaction.py:92–95). Discrete: `B_unique` (b, K) + bin_idx shared per level. Note the dense round-trip at build: `sp.csr_matrix(spline_spec._raw_basis_matrix(x))` (interaction.py:87, 226) — dense (n, K) intermediate then re-sparsified (S11).

### 2.6 FactorSmooth
`x, group (n,)` → codes (n,) intp; marginal PSpline basis QR'd (streamed R (k,k) or dense (n,k)); `natural_map` (k,k); GroupInfo carries codes + either exact sparse basis (n,k) or `basis_unique` (b,k)+bin_idx; penalties stay (k,k) `repeated_penalty_components`, never expanded to (L·k)² — expansion happens virtually in `FactorSmoothGroupMatrix`. n_cols = L·k (fs) or (L−1)·k (sz).

### 2.7 Predict path
Every spec has `transform` (materialise columns, dense) and `score` (direct contribution, chunked / support-compressed for splines and tensors: `_spline_runtime.score:152`, `interaction.score:1196`). `R_inv` set post-build via `set_reparametrisation` (dm_builder.py:913–915, 1024–1076) is the bridge from solver coordinates back to raw basis space.

---

## 3. STATE OBJECTS

| Object | Where | Fields / lifecycle | Overlap notes |
|---|---|---|---|
| Spline spec mutable state | set by `_spline_config._initialize_runtime_state` (157–178) | `_knots, _n_basis, _lo, _hi, _knot_strategy_actual, _R_inv, _interaction_projection, _basis_lo/_hi/_d1_lo/_d1_hi, _U_null, _U_range, _omega_range, _penalty_components, _lambda_policy`; plus `_explicit_knots/_explicit_boundary` (109–132); subclass extras `_Z` (spline.py:677, 769), `_cr_knots/_cr_M/_cr_S` (891–893), SCOP `_scop_Sigma/_scop_null_dim/_scop_col_means` (set in `_spline_subclass_ops.py:61–63` **and** dm_builder.py:771–773). Lifecycle: init → mutated by `_place_knots`/`build*` → mutated again by dm_builder (`set_reparametrisation`, SCOP fields) → read by transform/score/reconstruct forever after. | Feature specs are simultaneously config objects, fitted-state holders, and prediction engines. `_penalty_components` on the spec (build_knots_and_penalty:162) duplicates `GroupInfo.penalty_components`; `_R_inv` duplicates GroupMatrix.R_inv. |
| `GroupInfo` | types.py:116–256 | 30 fields: core (columns, n_cols, penalty_matrix, reparametrize, penalized, projection), select (subgroup_name, penalty_components, component_types, lambda_policies), constraints (constraints, monotone_engine, raw_to_solver_map, scop_reparameterization), spline-cat family (6 fields, 154–161), factor-smooth family (10 fields, 163–176), cat_codes. `__post_init__` validates shapes and that components sum to penalty_matrix (O(q·p_g²) allclose, 202–216). Lifecycle: transient — consumed by `_process_info`, fields mutated in place there (constraints composition, 644–649). | A tagged-union in disguise: at most one of {columns, cat_codes, spline_cat_*, factor_smooth_*} families is active per instance; invariants enforced only for factor_smooth (217–255). |
| `GroupSlice` | types.py:259–284 | name/start/end/weight/penalized/constraints/monotone_engine/scop — long-lived solver bookkeeping. | constraints/monotone_engine/scop duplicated from GroupInfo. |
| `PenaltyComponent` | types.py:288–315 | REML-side per-lambda record; built downstream from GroupInfo.penalty_components. | Third representation of the same penalty data (spec `_penalty_components` → GroupInfo.penalty_components → PenaltyComponent). |
| `TensorMarginalInfo` | types.py:319–337 | basis, penalty, knots, lo/hi, projection, K_eff, degree, `raw_basis_eval` **callable closure over the spec** (336), normalize_penalty. Long-lived: stored as `TensorInteraction._marginal1/_marginal2` and used at predict time — keeps the (possibly cloned) marginal spec alive via the closure. | Duplicates spec geometry (knots/lo/hi/degree) that the closure's owner already holds. |
| `DiscreteTensorBuildResult` | types.py:342–356 | infos + B_joint + factored parts; transient into `_process_info`. | B_joint carried "for fallback compatibility" alongside the factored form it duplicates (346–348). |
| `LinearConstraintSet` | types.py:24–55 | A, b; composed through projection then R_inv_local (dm_builder.py:644–645). | — |
| `LambdaPolicy` | types.py:60–90 | frozen estimate/fixed. | — |
| `ConstraintSpec` / `Constraint` | constraint.py:7–25 | frozen token. Legacy mirrors `spec.monotone/monotone_mode` still written (config 98–100) and still read as fallback (build 13–20, dm_builder 717–720). | Dual representation of one setting. |
| `LevelGrouping` | grouping.py:17–37 | frozen mapping; read by Categorical/OrderedCategorical. | — |
| `FactorSmooth` fitted state | factor_smooth.py:179–184, 325–327 | `_levels, _level_to_code, _spline` (owned PSpline), `_natural_map`, `_base_penalty_components`, `_marginal_build_backend`. | `_natural_map` also shipped inside GroupInfo.factor_smooth_transform. |
| `OrderedCategorical` state | ordered_categorical.py:226–290 | `kind/select/penalty/degree/n_knots` shortcut mirrors **plus** owned `_spline` object holding the same truths (286–290 re-syncs); `_level_to_value`, `_ordered_levels`, `_base_level`, `_R_inv` (step mode only). | Two sources of truth kept in sync manually. |

---

## 4. COMPLEXITY TABLE

| Routine | Time | Memory | Notes |
|---|---|---|---|
| `resolve_interior_knots` (quantile*) `_spline_knots.py:25` | O(n log n) | O(u) | per smooth, once. |
| `BSpline.design_matrix` via `basis_matrix` `_spline_runtime.py:38` | O(n·d) | sparse (d+1)·n nnz | good. |
| `raw_basis_matrix` `_spline_runtime.py:44–48` | O(n·d) + densify | **dense n·K** | every caller pays K/(d+1) memory blowup; callers: dm_builder:735 (b rows, fine), SplineCategorical (n rows, S11), tensor marginals (n rows, S4), FactorSmooth compat path. |
| identifiability projection `_spline_identifiability.py:55–72` | O(n log n) unique + O(u·d) eval + O(u·K) counts@basis | **dense u·K** | u≈n continuous. Runs in exact build, discrete build (S1), select build, and **once per m-order** in multi-m (S2b). |
| `build_difference_penalty` `_spline_penalties.py:10` | O(K³) (dense diff of eye + matmul) | O(K²) | K ≤ ~50, negligible. |
| `build_integrated_derivative_penalty` `_spline_penalties.py:20` | O(K spans × K fns × quad) Python-level, ~K² BSpline constructions | O(K²) | per order; re-run per m-order and again for select eigen basis. |
| `build_natural_constraint_null_space` `_spline_constraints.py:30` | O(K) BSpline constructions + O(K²) QR | O(K²) | called on every `_apply_constraints` — up to 3–4× per build for multi-m/select paths since `_apply_constraints` is re-invoked per component (`_spline_multi_penalty.py:25`, `_spline_select.py:109,135`, `_spline_build.py:145,151`). |
| `eigendecompose_select` `_spline_select.py:14` | O(K³) | O(K²) | may run twice on discrete select (build_knots_and_penalty:147–154 then again if build later called). |
| `build_group_info` total (exact) | O(n log n + n·d + u·K + K³) | sparse basis + dense u·K | u·K dense is the dominant transient for continuous x. |
| `build_knots_and_penalty` (discrete) | claimed O(b); actual O(n log n + u·d + u·K) | **dense u·K transient** | S1. |
| `_row_kron` `interaction.py:774` | O(p1·p2) Python iterations, each O(n) sparse multiply; total O(n·p1·p2) w/ large constants | CSR ≈ **dense n·p1·p2** (marginals are dense after centering) | exact tensor path only. n=1e6, p1=p2=9 → ~0.65 GB data + ~0.65 GB indices. |
| `tensor_marginal_info` (exact) | O(n·d) + O(n·K·K_eff) projection matmul | 2–3 dense (n, K) copies (lines 205, 210–212, 227) | per margin. |
| `TensorInteraction.build_discrete` | O(n) pair encoding + O(n log n) unique + O(n_pairs·K1K2) | B_joint dense (n_pairs, K1K2), n_pairs ≤ b² | good design. |
| `TensorInteraction.score` support path `interaction.py:1214–1241` | O(n log n) + batched einsum | bounded by `_MAX_TENSOR_SCORE_SUPPORT_CELLS` | good. |
| `FactorSmooth._build_marginal` streamed | O(n·k²) TSQR | O(chunk·k) = 65 536·k | fs m≤2. |
| `FactorSmooth._build_marginal` dense compat (297–304) | O(n·k²) | **dense n·k** | fs with m>2 or heterogeneous null policies. |
| `_natural_parameterization_from_r` | O(k³) + `matrix_rank` SVD ×2 (53, 309) | O(k²) | once. |
| `Categorical.build` | O(n) factorize + bincount | codes n·8B | good. |
| `Categorical.transform` (150–161) | O(n·L) dense | dense n·(L−1) | predict-time only. |
| `Categorical.score` (163–179) | O(n) + **O(L²)** `.index()` loop (177–178) | O(n) | minor unless L large. |
| `_validate_categorical_levels` (categorical.py:17–46) | **O(n) pure-Python element loop** (31) + O(n log n) unique | — | on every categorical/ordered/interaction build+transform+score call. |
| `OrderedCategorical._choose_base` (343–362) | **O(n·L)** per-level mask sum (349–351) | O(n) temporaries per level | contrast Categorical's O(n) bincount. |
| `CategoricalInteraction.build` (512–561) | O(n·L1·L2) mask loop (541–547) | sparse n nnz | pair factorisation would be O(n). |
| `GroupInfo.__post_init__` component check (types.py:202–216) | O(q·p_g²) | O(p_g²) | build-time only (rebuild path reuses GroupMatrix, not GroupInfo). |
| `rebuild_design_matrix_with_lambdas` (dm_builder.py:1202) | per REML iter: Σ O(p_g³) Cholesky-ish per changed group; **no feature rebuilds** | — | feature build is out of the optimizer loop — confirmed. |

---

## 5. SUSPECTS

**S1 — Discrete path materialises a dense (u×K) basis it promised to avoid.**
`spline.py:335–341` documents `build_knots_and_penalty` as "without building the full basis… avoid the O(n) basis construction". But `_spline_build.build_knots_and_penalty:158` → `apply_identifiability_for_spec` → `build_identifiability_projection_for_spec` (`_spline_identifiability.py:62–63`) does `np.unique(x)` then `spec._basis_matrix(support).toarray()` — dense (u, K). For continuous covariates u ≈ n, so the fREML path still pays an O(n·K) dense transient (plus O(n log n) sort) per smooth term at build. The constraint direction is just the column-sum of the basis over rows; it could be computed from the sparse basis or from `B_unique`+bin counts (as `tensor_marginal_info:213–217` already does with support+counts). Verify: memory profile of `fit_reml(discrete=True)` build phase with n=1e7; confirm the sum-over-rows is what the projection uses (it is: counts@basis, line 24–25).

**S2 — Select=True GroupInfo assembly is fully duplicated between `_spline_select.build_select_group_info` (74–125) and `dm_builder.build_design_matrix` (791–841).**
Same algebra line for line: n_null=1, `lstsq(Z, U_null/U_range)` back-projection, ω_null/ω_wiggle block assembly, per-order components, `component_types={"null": "selection"}`. Any change to select semantics (a protected API) must now be made twice; drift here would be a silent exact-vs-discrete divergence — exactly what the `discrete=True` contract forbids. Verify: diff the two code paths' outputs on a ps/cr select smooth, exact vs discrete.

**S2b — Multi-m rebuilds the identifiability projection per order.**
`_spline_multi_penalty.build_multi_m_components:23–27` calls `apply_identifiability` per order, and each call re-runs `build_identifiability_projection_for_spec` — i.e. re-`unique`s x and re-densifies the (u, K) support basis for every m-order, though the projection is identical across orders (same basis, same constraints). Same for `_apply_constraints` re-running the K-loop `build_natural_constraint_null_space` per component. Verify: cProfile a `Spline(kind="bs", m=(1,2,3))` build.

**S3 — Third copy of the identifiability algebra inside `tensor_marginal_info`.**
`_spline_build.py:213–226` re-implements centered-direction + complete-QR-drop-first-column inline instead of calling `_spline_identifiability.build_identifiability_projection` (11–30). The two differ subtly: tensor version normalises the direction first (222–223) and supports weighted counts; core version does not normalise. Numerically equivalent, but the duplication invites divergence in the k−1 contract. Verify: unit-compare projections from both paths on identical inputs.

**S4 — Exact tensor path builds a ~100%-dense matrix in sparse clothing.**
`TensorInteraction._centered_marginal_basis` (978–982) computes `B @ info.projection` — dense (projection mixes all K columns) — then wraps in csr. `_row_kron` (774–791) then loops over p1·p2 column pairs of these dense-as-sparse matrices, producing T (n, p1·p2) CSR with ≈ full density: 12 bytes/entry vs 8 for dense, plus a Python loop of p1·p2 sparse multiplies and an hstack. Same pattern in `eval_cardinal_basis` (`_spline_cardinal.py:82`) and `linear_tail_basis_matrix` (`_spline_extrapolation.py:72`). Verify: nnz/size ratio of T on a real fit; time `_row_kron` vs a dense einsum at n=1e6, p1=p2=9.

**S5 — Tensor marginal penalty normalisation is asymmetric across kinds.**
`_normalize_tensor_penalty` docstring (interaction.py:799–806) says mgcv rescales marginal penalties for tensors. Here `normalize_penalty=True` is set only on the CR→CardinalCR reroute (959); `TensorMarginalInfo.normalize_penalty` defaults to False (`_spline_build.py:244`, types.py:337). A `ps ⊗ cr` tensor therefore normalises one margin and not the other, putting margins on incomparable penalty scales — the exact problem the helper cites mgcv for. Verify: compare REML lambdas / EDF against mgcv `ti(x1, x2, bs=c("ps","cr"))`.

**S6 — CR parents are silently rerouted (with changed knot strategy) for tensor marginals.**
`_marginal_from_spec` (interaction.py:938–960): a `CubicRegressionSpline` parent becomes a `CardinalCRSpline` marginal, and if the parent used `knot_strategy="uniform"` without explicit knots it is switched to `"quantile"` (939–941). So the tensor marginal does not inherit the parent smooth's geometry, contradicting both `tensor_marginal_ingredients`' docstring ("Reuses the parent's knot vector… so that tensor marginals inherit the parent spline geometry", spline.py:370–387) and the CardinalCRSpline docstring's "not yet the default for kind='cr'" (827–828). Verify: fit `s(x1, kind="cr") + ti(x1,x2)` and compare marginal knots to the parent's.

**S7 — Legacy/compat shims that carry real complexity.**
(a) `spline.py:36–38` `_weighted_quantile_knots` — zero callers in src and tests; dead.
(b) `spec.monotone/monotone_mode` mirrors written in `_spline_config.py:98–100`, read as fallback in `_spline_build.py:13–20` and dm_builder.py:717–720 — dual representation of `constraint_kind/mode`.
(c) `_marginal_from_spec`'s `inspect.signature`-based dispatch for "custom spline subclasses overriding the old one-argument method" (interaction.py:920–936) — runtime reflection in the build hot path to support an undocumented extension contract.
(d) The mgcv-contract rejection is triplicated: `tensor_marginal_info` (`_spline_build.py:184–195`), `_marginal_from_spec` (interaction.py:887–900), and implicitly `validate_select`. Verify each shim's reachability with tests.

**S8 — CardinalCRSpline docstring documents parameters that don't exist.**
spline.py:844–848 documents `monotone : str or None` and `monotone_mode : str` ("'fit' (not yet implemented)"); the actual `__init__` (859–890) takes `constraint=` and no monotone params. Doc-code mismatch on an experimental-but-shipping class.

**S9 — Two functions named `assemble_clamped_knot_vector` with different semantics.**
`_spline_runtime.py:126–136` pads boundaries by `1e-6·range`; `_spline_subclass_ops.py:30–39` uses exact boundaries. The padded one is reachable only through `_SplineBase._assemble_knot_vector` default (spline.py:197–203), whose only concrete user is `NaturalSpline`. Likewise `_spline_cardinal_spec.place_knots` (14–40) duplicates `_spline_runtime.place_knots` (102–123) boundary/reset logic. Name collision + duplication around a numerically delicate spot (boundary placement interacts with the natural f''=0 constraints). Verify which knot assembly each kind actually gets, and whether NaturalSpline's pad is intentional given CRS explicitly removed it (771–777).

**S10 — OrderedCategorical duplicates Categorical's responsibilities with worse asymptotics and drift-prone mirrored state.**
`_choose_base` (ordered_categorical.py:343–362) is an O(n·L) re-implementation of Categorical's O(n) bincount base selection (categorical.py:117–122); the class also mirrors `kind/select/penalty/degree/n_knots` from the owned Spline object (226–231, 286–290) — two sources of truth reconciled manually; a ~40-line deprecation-warning combinatorics block (180–221) is pure legacy-API accommodation. The deprecated `step` mode (394–444) carries its own penalty algebra and R_inv handling that nothing else shares.

**S11 — `_raw_basis_matrix` densify-then-resparsify round trips.**
`_spline_runtime.raw_basis_matrix:48` returns `.toarray()`. `SplineCategorical.build:87` and `score:226` immediately wrap it back into `sp.csr_matrix`, paying a dense (n, K) intermediate plus conversion for a matrix that `BSpline.design_matrix` had already produced in CSR. For an n=1e7, K=13 smooth that's a ~1 GB avoidable transient per spline-by-factor term. Verify with memory profiler on SplineCategorical build.

**S12 — Pure-Python O(n) NaN scan in `_validate_categorical_levels`.**
categorical.py:31: `any(v is None or (isinstance(v, float) and np.isnan(v)) for v in x)` iterates every element in Python; it runs on every build/transform/score of Categorical, OrderedCategorical, and all four categorical-flavoured interactions (interaction.py:200, 222, 340, 354, 448, 461, 566, 581 — twice per call for two-cat interactions). At n=1e7 this is seconds of interpreter time per call, in both fit and predict paths. `pd.isna(x).any()` is the vectorised equivalent already used at `_grouping_labels` (categorical.py:54). Verify with cProfile on a categorical-heavy predict.

**S13 — FactorSmooth marginal deliberately diverges from ordinary P-spline knots, gated by a backend flag with behavioural consequences.**
`_initialize_marginal_spline` (factor_smooth.py:246–263) re-assembles knots on a ±0.1%-expanded boundary "for backwards compatibility" — the fitted fs marginal is not the same basis a standalone `Spline(kind="ps", k)` would produce on the same data. `_streaming_safe` (222–236) silently switches between `streamed_tsqr` and `dense_qr_compat` backends based on m and lambda-policy homogeneity; the comment says QR sign/null rotations may not "preserve the declared penalty geometry" — i.e. results are backend-dependent in the unsafe cases and the dense path is O(n·k) memory. Verify: fs with m=3 at large n (memory), and cross-backend coefficient equivalence for m=2.

**S14 — GroupInfo is an untyped union with 30 fields and mutation-in-place downstream.**
types.py:116–176: four mutually exclusive field families (columns / cat_codes / spline_cat_* / factor_smooth_*) share one dataclass; `_process_info` mutates `info.constraints` and `info.raw_to_solver_map` in place (dm_builder.py:644–649), and dm_builder also back-fills `lambda_policies`/`penalty_components` post-hoc for the discrete path (869–877) — replicating what `build_group_info:125–128` does for the exact path (another exact/discrete duplication pair). Not a bug, but the single biggest unclear-responsibility surface in the subsystem.

**S15 — Repeated small-scale refactorisation inside per-component loops.**
`_apply_constraints` (natural null space: K `BSpline` constructions + QR) is recomputed once per m-order component and once more for the select eigen penalty (`_spline_multi_penalty.py:25`, `_spline_select.py:109, 135`, `_spline_build.py:145–151`), and `build_integrated_derivative_penalty` re-runs `leggauss` per knot span (`_spline_penalties.py:36`). All O(K²)–O(K³) so cheap in isolation, but multiplied across m terms × q components × (exact + discrete duplicate paths) it is measurable interpreter overhead at build; more importantly it signals missing "compute constraints once per spec" structure.

---

## 6. Contract verification notes (protected semantics)

- **k / k−1 contract:** verified consistent across ps/bs (open vector, `_spline_factory.py:24–25`), cr (clamped +2, −2 natural, `:31`), cr_cardinal (K=k, `:27–29`); identifiability removes exactly 1 column in all kinds (`_spline_identifiability.py:28–30`). FactorSmooth intentionally keeps k columns per level (no identifiability; fully penalised), matching mgcv `bs="fs"`.
- **select=True vs selection_penalty:** select handled entirely here via null/range decomposition with a `component_types={"null": "selection"}` marker; no group-lasso coupling found inside features/ — separation respected.
- **discrete=True no-drift:** geometry (ω, projection, n_cols) is shared between exact and discrete paths by construction (`build_knots_and_penalty` reuses the same helpers), **except** the select assembly which is duplicated code (S2) — that duplication is the main drift risk.
- **sample_weight:** feature builds accept it; splines explicitly discard it (`del sample_weight`, `_spline_build.py:60, 141`) so the identifiability constraint is unweighted row-mean-zero (documented at spline.py:241–247). Categorical base selection does use it (exposure, categorical.py:117–122). Consistent with docs.
