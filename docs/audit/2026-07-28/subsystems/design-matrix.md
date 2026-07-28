# Subsystem audit: design-matrix (GroupMatrix abstraction)

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ origin/master (f082e9b).
All paths below are relative to `src/superglm/` unless prefixed otherwise. Notation: n = rows, p = total built solver columns, p_g = one group's width, G = number of groups (~ m smooth terms + parametric terms), K = raw basis width of one smooth (mgcv k; built columns k-1), b = n_bins per discretized smooth (default 256), q = number of penalties.

---

## 1. MODULE MAP

### `group_matrix.py` (450 lines) — public facade + `DesignMatrix` container
- Re-exports all 11 concrete GroupMatrix classes from `_group_matrix/_group_matrix_core.py` and `_group_matrix/_group_matrix_discretized.py`, rebranding `__module__` (group_matrix.py:55–65). `GroupMatrix` is a union type alias (group_matrix.py:73–85).
- **Compat wrapper layer** (group_matrix.py:68–188): `_discretize_column` (:68), `_agg_by_bin` (:91), `_cross_gram_tensor_tensor` (:96), `_disc_disc_2d_hist` (:101), `_cross_gram_tensor_main` (:111), `_cross_gram` (:116), `_gram_any_sign` (:126), `_block_xtwx` (:131), `_block_xtwx_rhs` (:149), `_block_xtwx_signed` (:172) — thin delegations into `_group_matrix_algebra`. Production callers: `_discretize_column` (dm_builder.py:734, features/interaction.py:150,1143–1144, features/factor_smooth.py:393, model/base.py:75), `_disc_disc_2d_hist` (solvers/scop_newton.py:32,758), `_cross_gram` (dm_builder.py:1183, solvers/_structured/moments.py:11). **`_block_xtwx`, `_block_xtwx_rhs`, `_block_xtwx_signed`, `_agg_by_bin`, `_gram_any_sign`, `_cross_gram_tensor_tensor`, `_cross_gram_tensor_main` have no production callers** — only tests (test_matrix_execution_plan.py, test_theory_invariants.py, ...).
- `_LazyTabmatSplit` (:190–204): shared lazy tabmat SplitMatrix holder (no back-ref to DesignMatrix, pickle-safe). `_LazyRawSplineTabmatPlan` (:207–229): releasable raw-spline CSC acceleration holder with build timing.
- `DesignMatrix` (:232–450): tuple of GroupMatrix + shape validation (:235–251); pickle shims (`__getstate__`/`__setstate__` :266–313, incl. legacy `_tabmat_split`/`_tabmat_built` migration); lazy caches: tabmat split (:315–341), raw-spline plan (:343–376), `execution_plan` (:378–393, builds `MatrixExecutionPlan`), `mixed_bin_space_centering_plan` (:395–412); `matvec`/`rmatvec` (:414–438, per-group loop with a retained-tabmat-vector fast path), `toarray` (:440–442, full (n,p) dense hstack), `row_subset` (:444–450).
- Called from: `dm_builder.build_design_matrix` (dm_builder.py:1084), `rebuild_design_matrix_with_lambdas` (dm_builder.py:1356), and ~30 consumer modules (solvers/pirls.py, solvers/irls_direct.py, solvers/centered_system.py, reml/*, inference/*, model/*).

### `_group_matrix/_group_matrix_core.py` (804 lines) — exact (non-discretized) backends
- `DenseGroupMatrix` (:31–56): wraps (n, p_g) ndarray; `gram` via sqrt(W) scaling (:48–50) — cannot handle signed W.
- `SparseGroupMatrix` (:59–83): CSR wrapper; `gram` via `.multiply(sqrtW)` (:74–77) → also sqrt(W)-based.
- `CategoricalGroupMatrix` (:86–133): stores int codes only, base level remapped −1→n_levels sink bin (:99–100); matvec = fancy index (:105–111), rmatvec/gram = bincount (:113–120). No matrix stored at all.
- `RandomEffectGroupMatrix` (:136–161): all-level categorical subclass (no dropped base), carries `lambda_policies`.
- `FactorSmoothGroupMatrix` (:164–630): compact factor-by-spline; dual storage — exact CSR marginal basis `B` + per-observation `codes`, or discrete `B_unique` + `bin_idx` (:255–285). fs vs sz coefficient bases (:239–240, sum-to-zero contrast via `factor_smooth_geometry`). Rich fused API: `factor_smooth_sufficient_stats` (:358–400, per-level Gram/XtW/Xtrhs), `factor_smooth_discrete_cell_moments` (:402–438), `factor_smooth_dense_cross_gram` (:464–501), `factor_smooth_discrete_dense_cell_cross_gram` (:503–532), `factor_smooth_discrete_shared_bin_cross_gram` (:534–577), `gram_rmatvec` (:579–589). Consumed by solvers/_structured/moments.py:197–336.
- `SparseSSPGroupMatrix` (:633–690): factored sparse B (CSR) + dense `R_inv` (K, p_g); effective matrix B@R_inv never formed; `gram` via numba `_csr_weighted_gram` then R_inv sandwich (:676–679). Carries penalty metadata slots set externally: `omega`, `projection`, `omega_components`, `component_types`, `lambda_policies` (:662–666).
- `SplineCategoricalGroupMatrix` (:693–804): one spline-by-category level; stores full B **and** row-sliced `B_level = B[row_idx]` (:732, a copy) plus its CSR internals; `gram_rmatvec` fused (:768–781). Note `row_subset` (:789–804) uses `np.isin` + re-slice.

### `_group_matrix/_group_matrix_discretized.py` (429 lines) — discrete=True backends
- `DiscretizedSSPGroupMatrix` (:15–88): dense `B_unique` (b, K) at bin centers + `bin_idx` (n,) intp + `R_inv` (K, p_g). All ops aggregate weights by bin first: matvec = gather (:49–52), rmatvec = bincount (:54–57), `gram` = bincount + (K,K) dense gram + R_inv sandwich (:59–63), `gram_rmatvec` fuses W and Wz bincounts in one numba pass (:65–77). Same external penalty metadata slots (:43–47).
- `DiscretizedSCOPGroupMatrix` (:91–139): like SSP but stores pre-centered SCOP design at bin centers; no R_inv (columns already solver-space).
- `DiscretizedSplineCategoricalGroupMatrix` (:142–292): spline support grid restricted to one category level; `bin_idx_level` indexed by `row_idx`; `row_subset` does an argsort/searchsorted intersection (:258–292).
- `DiscretizedTensorGroupMatrix` (:295–429): subclass of DiscretizedSSP; stores factored marginals `B1_unique_t` (b1, K1), `B2_unique_t` (b2, K2), `idx1`, `idx2`, **plus the materialized Kronecker `B_joint` as inherited `B_unique` (n_pairs, K1·K2) "for fallback compatibility"** (:304–307). `_factored_gram_raw` (:341–364) computes the raw tensor Gram via batch BLAS in O(b1·b2·K2² + b1·K1²·K2²); `gram_rmatvec_from_grids` (:371–380) accepts precomputed 2D weight grids; matvec has a cost-model switch between direct-pair evaluation and factored observation path (:389–407). `_own_margin_cache` dict keyed by `(id(gm_main), id(bin_idx), n_bins)` (:317, populated at algebra.py:496–506).

### `_group_matrix/_group_matrix_kernels.py` (391 lines) — numba `@njit(cache=True)` kernels
17 kernels, all O(n) or O(nnz) single-pass: `_csr_weighted_gram` (:10), `_weighted_bincount_2d` (:31), `_csr_weighted_bincount` (:45), `_disc_disc_2d_hist` (:59), `_disc_disc_2d_hist_channels` (:69), `_fused_bincount_2` (:84), `_random_effect_sufficient_stats` (:97), 8 factor-smooth kernels (:109–289), `_dense_small_weighted_moments` (:293), `_fused_2d_bincount_2` (:316), `_pattern_support_summaries` (:330, the fREML pattern-compression workhorse), `_cat_weighted_bincount` (:372), `_cat_cat_weighted_crosstab` (:383).

### `_group_matrix/_group_matrix_bins.py` (28 lines)
`discretize_column` (:9–25): np.unique first — exact support if ≤ n_bins distinct values, else equal-width bins on [min, max] with bin centers as representative values. O(n log n) per feature (unique sort).

### `_group_matrix/_group_matrix_algebra.py` (1095 lines) — cross-product dispatch
- Profiling helpers (:35–47); `_BlockWeightCache` (:50–111): per-assembly-call cache of 2D weight histograms keyed by `id()` tuples (:59–61), including transposed-hit reuse (:76–81) and fused W/Wz tensor grids (:92–111).
- `_runtime_group_matrix_types` (:114–136): lazy import shim to break the circular import with `group_matrix.py` — called at the top of every dispatch function (`_agg_by_bin`, `_cross_gram`, `_gram_any_sign`).
- `_agg_by_bin` (:139–219): aggregate W·X_g by an external bin index → (n_bins, p_g); per-type dispatch (categorical bincount, CSR kernel, disc×disc 2D hist under a 5M-cell cap, dense kernel fallback, and `_aggregate_group_matrix_columns` (:222–240) unit-vector matvec loop for factored types).
- Specialized cross-grams: `_random_effect_cross_gram` (:243), tensor×tensor same-marginals (:257–273), tensor×tensor shared-one-margin (:312–384, python loop over shared bins with einsum), tensor×main generic channel histograms (:387–476, orientation cost model), tensor×own-margin reusing the tensor W-grid (:479–538, mirrors mgcv XWXd packed marginals; cited in docstring :487–491), tensor×spline-cat (:541–584), cat×spline-cat (:587–612), spline-cat×spline-cat with same-parent zero shortcut and row-intersection general case (:615–696), disc×spline-cat (:699–719), `_cross_gram_by_columns` bounded-memory unit-vector fallback (:722–746).
- `_cross_gram` (:749–996): the master ~250-line isinstance dispatch ladder; final generic fallback materializes the *smaller* group dense `(n, p_g)` plus a W-broadcast copy, then per-column rmatvec on the other (:983–996).
- `_gram_any_sign` (:999–1032): signed-W diagonal Gram dispatch (discretized/SSP/factor types handle signed W natively; dense/sparse fall back to explicit `W[:,None]*X` since their `gram()` uses sqrt(W)).
- Compat entry points `_block_xtwx`, `_block_xtwx_rhs`, `_block_xtwx_signed` (:1049–1096) each construct a **fresh `MatrixExecutionPlan` per call** via `_execution_plan_for_blocks` (:1035–1046). No production callers (see module map for group_matrix.py).

### `_group_matrix/_group_matrix_execution.py` (438 lines) — `MatrixExecutionPlan`
- `GroupSpan` (:24–34, frozen dataclass: index/start/end), `WeightedMoments` (:37–43: gram, xtw, xt_rhs tuple), `OrdinaryPartitionDecision` (:46–51).
- `MatrixExecutionPlan` (:54–438): immutable group layout (setattr guard :63–68); partitions groups into "ordinary" (dense/sparse/categorical → one tabmat SplitMatrix sandwich) vs "compressed" (everything else → specialized kernels). Automatic ordinary partition (`_ordinary_partition_decision` :148–188) is deliberately narrow: all groups ordinary + exactly 1 categorical with >100 levels + dense width ≥3 + no sparse + n ≥ 50_000 (`_MIN_AUTO_TABMAT_MOMENT_ROWS` :21); each rejection records a stable reason string.
- `moments` (:256–273) public/validated; `_moments_prevalidated` (:275–299) trusted fit-internal (skips finite/negativity scans); `_moments_impl` (:301–438): optional full-tabmat shortcut, otherwise zero-init (p,p) gram + per-group diagonal (fused `gram_rmatvec` when a rhs/xtw fusion vector exists and the group supports it, incl. tensor grid cache :397–411) + O(G²) upper-triangle `_cross_gram` blocks (:430–436). `_compressed_signed_gram` (:231–254) is a near-duplicate of the cross-block loop for the signed/no-rhs case.
- Production callers: `dm.execution_plan.moments` in reml/objective.py:119, reml/efs.py:120,219, reml/w_derivatives.py:396, inference/covariance.py:676, model/design_summary.py:156; `_moments_prevalidated` in solvers/irls_direct.py:1587, solvers/_structured/moments.py:137,293, and `_try_factored_tensor_centering` (centered.py:250).

### `_group_matrix/_group_matrix_tabmat.py` (302 lines) — tabmat integration
- `RawSplineTabmatPlan` (:24–84): one CSC SparseMatrix over hstacked raw spline bases B (never forming B@R_inv), with slice/transform metadata; `transform_vector` (:35), `transform_gram` (:50, block-wise R_invᵀ·raw·R_inv), `retained_bytes` estimator (:65–84).
- Gate `_is_raw_spline_tabmat_centering_candidate` (:87–106): ≥2 groups, all exactly `SparseSSPGroupMatrix`, nnz ≤ 4/row, density ≤ 1/3, n ≥ 8_000, n·raw_width ≤ 4M cells. `_defer_raw_spline_tabmat_plan` (:109–124) — **defined but never called anywhere** (grep: no callers in src or tests).
- `_tabmat_vector` (:167–172): defensive copy to writable C-contiguous float64 (tabmat 4.2.1 kernels silently miscompute on strided input — comment at centered.py:141–144).
- `_native_categorical_matrix` (:175–186): remaps SuperGLM's sink-bin encoding to tabmat `drop_first=True`.
- `_is_tabmat_centering_candidate` (:189–213): no compressed/support-space groups + ≥1 non-categorical + ≥1 categorical with >100 levels.
- `_is_retained_tabmat_vector_candidate` (:216–239): gate for retaining the split for `DesignMatrix.matvec/rmatvec` (n ≥ 10_000, ≥ 8 single-column dense groups, all categoricals >100 levels, ratio caps).
- `_build_tabmat_split` (:242–302): returns None if any support-space group present (:257–269) or all-small-categoricals (:271–274); otherwise per-type tabmat blocks. **The `SparseSSPGroupMatrix` branch (:296–297, `toarray()` densify) is unreachable** — guarded out at :257–269. The final generic `else: toarray()` (:300–301) is likewise unreachable for all current types.

### `_group_matrix/_group_matrix_bin_space.py` (374 lines) — augmented mixed-design tabmat plan
- `CompressedBinSpaceBlock` (:33–40), `MixedBinSpaceCenteringPlan` (:43–152): one tabmat SplitMatrix over an *augmented* column space where each DiscretizedSSP contributes b one-hot bin-indicator columns (as tabmat CategoricalMatrix) and dense groups one shared slab; `moments` (:84–152) does one sandwich in augmented space (Σb + dense + cat columns) and maps to solver space by `supportᵀ · raw · support` per block (support = B_unique@R_inv, :341).
- `build_mixed_bin_space_centering_plan` (:155–374): eligibility (only Dense/Categorical/DiscretizedSSP, ≤1 categorical, :183–195) plus seven 64 MiB byte-budget gates with hand-derived retained/construction/transient estimates (:197–288) — ~90 lines of memory-model arithmetic.

### `_group_matrix/_group_matrix_centered.py` (990 lines) — centered-moment strategies
- `_certify_raw_centering` (:68–104): the shared numerical certificate — accepts raw-moment subtraction only when |mean| ≤ centered RMS per column (`_raw_centering_well_scaled` :107–120).
- Strategy implementations: `_try_tabmat_centering` (:123–172, native categorical tabmat), `_try_raw_spline_tabmat_centering` (:175–226), `_try_factored_tensor_centering` (:229–264, delegates to `execution_plan._moments_prevalidated`), `_mixed_raw_centering_preflight` + `_try_mixed_discrete_centering` (:267–397, uses the bin-space plan; row-count gates `_MIN_MIXED_RAW_MOMENT_CELLS`=100k cells, low-cardinality n ≥ 5_000).
- Pattern-compression path for fREML tensor designs: `_PatternPlan` (:34–49), `_build_pattern_plan` (:400–517) — requires **exactly one** `DiscretizedTensorGroupMatrix` (:410–414) and only Categorical/DiscretizedSSP/Tensor groups; builds a uint64 radix key per row, `np.unique` to unique code patterns, detects tensor own-margins, precomputes pair histogram layout under 5M-cell caps. Cached on `dm._centered_pattern_plan` with a False sentinel (:520–527). `_solver_supports` (:530–554) caches per-group (support-rows × p_g) matrices incl. B_unique@R_inv. `_try_pattern_tensor_centering` (:557–658): one `_pattern_support_summaries` numba pass over n + all marginal/pair contractions on supports.
- `packed_centered_gram_rhs` (:689–795): all-discrete/categorical designs; anchor-centers each support (`_anchor_center_support` :661–686) then per-pair 2D histograms (O(n) each).
- Chunked fallbacks (used when no strategy certifies): `stable_centered_gram_rhs` (:805–873, two passes of `row_subset(chunk).toarray()` with Kahan-compensated accumulation), `stable_centered_matvec` (:876–908), `centered_gram_rhs` (:911–956), `centered_rhs` (:959–990). All materialize (chunk, p) dense blocks.
- Consumers: solvers/centered_system.py:12–18 (the strategy cascade at :211–276), reml/w_derivatives.py:389, reml/scop_geometry.py:198, reml/observed_geometry.py, reml/discrete.py.

### `dm_builder.py` (1358 lines) — construction & lambda-rebuild orchestration
- Name-collision validators (:49–73). `compute_R_inv` (:88–110): SSP transform R⁻¹ from chol(B'WB/n + λ₂Ω + 1e-8·I) (Wood 2011 §3.1/§5); `compute_projected_R_inv` (:113–135) same within a constraint-projected subspace. Both use explicit `np.linalg.inv` (:110, :135).
- Discretization policy: `should_discretize` (:138–148, SSP splines only; per-spec override then model flag), tensor/spline-cat/factor-smooth variants (:151–181), `resolve_discrete_n_bins` (:184–203, spec.n_bins > per-feature dict > 256).
- `auto_detect_features` (:211–259) via `EagerFrame.column_kind`; `_spec_kind` (:262–280); interaction factory table `_INTERACTION_FACTORIES` (:337–345) + `add_interaction` (:348–391) with orientation swap.
- `_process_info` (:400–651): the central GroupInfo → GroupMatrix constructor. Four top-level branches: spline-cat (:428–500), projection (:502–542), reparametrize (:544–575), plain (:577–639); each independently wires R_inv/omega/omega_components/component_types/lambda_policies onto the group matrix. Composes constraints and raw_to_solver_map with R_inv_local (:641–649). Returns (gm, R_inv, n_cols).
- `BuildResult` (:654–664). `build_design_matrix` (:667–1093): per-feature loop — discrete path builds knots/penalty, discretizes column, evaluates `spec._raw_basis_matrix(bin_centers)`, bincounts exposure (:729–736); SCOP-discrete branch (:754–789) builds bin-level centered SCOP design; select=True branch (:791–841) assembles null+range double-penalty components; lambda-policy backfill shim (:869–877 with explanatory comment, synthesizes `[("wiggle", omega)]`). Interaction loop (:927–1080) with per-kind discrete builds and `_maybe_apply_tensor_side_constraints` (:1158–1199) which computes cross-grams against every prior overlapping interaction (dm_builder.py:1177–1185) and null-space-projects the tensor (`_null_space_projection` :1127, SVD).
- `rebuild_design_matrix_with_lambdas` (:1202–1358): per-group re-derivation of R_inv for changed λ; five near-identical class-specific branches (SparseSSP :1216, SplineCategorical :1235, DiscretizedSplineCategorical :1257, DiscretizedTensor :1298 — skipped entirely when it has omega_components and no projection, mirroring mgcv bam(discrete=TRUE) fixed packed design, comment :1302–1309 — DiscretizedSSP :1335). Builds a fresh `DesignMatrix` and copies **only** `_centered_pattern_plan` forward (:1356–1358).
- Callers: model/base.py:933 (build) and model/base.py:1013, reml/discrete.py:328,1154,1197 (rebuild — the :1154 call is inside the fREML outer ρ-iteration loop).

### `_frame.py` (251 lines) — dataframe boundary
- `EagerFrame` (:27–186): pandas/Polars adapter (narwhals for Polars only); caches extracted column ndarrays in `_arrays` dict keyed by column name (:113–129, "at most once" extraction; retained for the frame's lifetime); `column_kind` classification (:71–99, incl. Polars Decimal→categorical parity comment), `take_rows` (:131), `select_native` (:140), `digest` (:152–185, blake2b row-hash for retained-fit-data verification). `as_eager_frame` (:188–222) rejects LazyFrame with a targeted message.

### `validation.py` (789 lines) — **not design-matrix code**
Actuarial model-validation charts: `lift_chart` (:217), `double_lift_chart` (:305), `lorenz_curve` (:477), `loss_ratio_chart` (:695), tie-collapsed Gini via pair concordance (:155–201). Shares nothing with the GroupMatrix subsystem except the name; assigned to this reader by the audit file list — see Suspect S12.

---

## 2. DATA FLOW

**Build (once per fit call)** — model/base.py:933 → `build_design_matrix` (dm_builder.py:667):
1. `EagerFrame.column_array(name)` extracts each feature column once and caches it (`_frame.py:113–129`) — one (n,) float/object array per feature, retained on the frame view.
2. Per feature: either `spec.build(x, sample_weight)` → `GroupInfo` with materialized `columns` ((n, p_g) dense or CSR), or the discrete path (dm_builder.py:729–855): `discretize_column` (O(n log n) unique; (n,) intp `bin_idx` + (≤b,) centers), raw basis at bin centers `B_unique` (b, K), exposure bincount (b,).
3. `_process_info` (dm_builder.py:400) computes `R_inv` — for discrete groups from the (b, K) support with binned exposure weights (O(b·K² + K³)), for exact groups from B with full sample_weight (O(nnz·K + K³)) — and constructs the concrete GroupMatrix. Materializations per group type:
   - Dense numeric: (n, p_g) ndarray (already extracted). SSP dense: `columns @ R_inv` materialized (dm_builder.py:531,570).
   - SparseSSP: CSR B (n, K) + R_inv (K, k−1); B@R_inv never formed.
   - Categorical/RandomEffect: (n,) intp codes only.
   - DiscretizedSSP: (b, K) + (n,) intp + (K, k−1).
   - DiscretizedTensor: (b1,K1)+(b2,K2)+(n,) idx1+(n,) idx2 **and** (n_pairs, K1·K2) B_joint + (n,) pair_idx (n_pairs ≤ b1·b2 observed support).
   - FactorSmooth: shared CSR (n, K) or (b, K) support + (n,) codes + (n,) bin_idx.
4. Interactions loop appends more groups; tensor side-constraint projection may shrink a tensor's width via SVD null-space (dm_builder.py:1158–1199), calling `_cross_gram` against prior interaction groups over full n.
5. `DesignMatrix(group_matrices, n, p)` — p = Σ p_g; each spline contributes k−1 built columns (SSP identifiability), consistent with the protected `k` contract.

**Fit-time weighted moments (per PIRLS/BCD iteration)** — two parallel consumers:
- *Uncentered*: `dm.execution_plan.moments(W, rhs=(Wz,))` (execution.py:256) → hybrid: one tabmat sandwich for the ordinary partition (only under the narrow auto gate or forced policy), per-group fused `gram_rmatvec` diagonals, O(G²/2) `_cross_gram` off-diagonal blocks. Output `WeightedMoments`: (p,p) gram + (p,) vectors. Discrete×discrete cross blocks each run one O(n) fused 2D histogram (bounded by `_BlockWeightCache` reuse when W/Wz identity-match).
- *Centered* (solvers/centered_system.py:184–298, the numerically-armored path used by direct PIRLS/REML): cascade `packed_centered_gram_rhs` (all-discrete; internally pattern-tensor → factored-tensor → per-pair anchored histograms) → `_try_mixed_discrete_centering` (augmented bin-space tabmat sandwich) → `_try_raw_spline_tabmat_centering` (raw CSC sandwich + R_inv transform) → `_try_tabmat_centering` (native categorical) → chunked `centered_gram_rhs` fallback ((chunk, p) dense blocks, two-pass, Kahan compensation). Each strategy self-certifies via `_certify_raw_centering`; rejection latches per-fit via `TabmatCenteringState`.
- Vector ops: `dm.matvec/rmatvec` per-group loops (group_matrix.py:414–438); eta updates in PIRLS.

**REML λ-update loop (fREML/discrete)** — reml/discrete.py:1154: `rebuild_design_matrix_with_lambdas` re-derives R_inv per λ-owning group (per-group O(n) exposure bincount for discretized groups, O(nnz) weighted CSR gram for exact SSP groups), constructs a **new** DesignMatrix (fresh lazy tabmat holders, fresh execution plan, fresh mixed-bin-space plan, fresh `_centered_solver_supports`), carrying only `_centered_pattern_plan` (dm_builder.py:1357). Discrete tensors with penalty components skip rebuild entirely (S(λ) moves instead; dm_builder.py:1302–1309).

**Prediction/inference**: `spec.transform` + stored R_inv on specs (`set_reparametrisation`, dm_builder.py:913–915,1024–1076); `dm.row_subset` for subsampled diagnostics; `toarray()` only for small-model oracles and the chunked fallbacks.

---

## 3. STATE OBJECTS

| Object | File:line | Fields (essence) | Lifecycle | Overlap notes |
|---|---|---|---|---|
| `DesignMatrix` | group_matrix.py:232 | group tuple, n, p + **8 mutable cache slots**: `_tabmat_holder`, `_raw_spline_tabmat_holder`, `_tabmat_centering_candidate`, `_tabmat_vector_candidate`, `_execution_plan`, `_mixed_bin_space_centering_plan(+_attempted)`, `_centered_pattern_plan`, `_centered_solver_supports`, `_scalar_structured_layout_cache` | per fit; recreated on every λ-rebuild; pickle drops all caches except tabmat holder | cache slots are written by 3 different modules (group_matrix.py, centered.py:520–554, solvers/_structured) — DesignMatrix is a cache bag for other subsystems |
| `_LazyTabmatSplit` / `_LazyRawSplineTabmatPlan` | group_matrix.py:190/207 | group refs + built flag + split/plan | lazy, per-DesignMatrix; raw-spline plan releasable post-fit (model/fit_state.py:687) | two holders with the same build-once pattern, different release semantics |
| `MatrixExecutionPlan` | execution.py:54 | immutable spans + ordinary partition + masks + lazy ordinary split | lazily built per DesignMatrix; discarded on rebuild/pickle | duplicates DesignMatrix's tabmat split via `ordinary_split_factory` indirection (group_matrix.py:386); `GroupSpan` duplicates `GroupSlice` (types.py:260) start/end bookkeeping |
| `GroupSpan` / `WeightedMoments` / `OrdinaryPartitionDecision` | execution.py:24/37/46 | frozen coordinate/result records | per plan / per call | `validate_group_spans` (execution.py:219) exists purely to reconcile GroupSpan with solver GroupSlice |
| `RawSplineTabmatPlan` | tabmat.py:24 | CSC split + slices + R_inv transforms | lazy, releasable | third independent tabmat representation of the same design |
| `MixedBinSpaceCenteringPlan` / `CompressedBinSpaceBlock` | bin_space.py:43/33 | augmented split, dense slab copy (n×dense_cols), supports (b×p_g), index maps, 3 byte estimates | lazy, once per DesignMatrix (not carried across rebuild) | fourth tabmat representation; dense slab duplicates DenseGroupMatrix.M |
| `_PatternPlan` / `_CenteredSupport` / `_TensorGridCache` | centered.py:34/25/52 | unique code patterns, radix layout, pair offsets / anchored support moments / one-grid duck-typed cache | `_PatternPlan` per fit (carried across rebuild via `dm._centered_pattern_plan`, False sentinel); others per call | `_TensorGridCache` duck-types `_BlockWeightCache.tensor_w_grid` (centered.py:52–59) |
| `_BlockWeightCache` | algebra.py:50 | id()-keyed 2D-hist dict + profile | per moments-assembly call | |
| `GroupInfo` | types.py:116 | 30+ optional fields spanning 6 term families (spline-cat, factor-smooth, SCOP, multi-penalty, constraints…) | transient build-time carrier, mutated by `_process_info` (constraints/raw_to_solver_map composed in place, dm_builder.py:644–649) | the union-of-all-features record; penalty metadata (omega, components, policies) is then copied onto GroupMatrix instances, giving penalty state two homes |
| `GroupSlice` | types.py:260 | name/start/end/weight/penalized/constraints… | long-lived solver metadata | start/end duplicated by GroupSpan; penalty λ addressing by name string (`_resolve_group_lambda`, dm_builder.py:1101) |
| `BuildResult` | dm_builder.py:654 | dm, groups, distribution, link, y, w, offset | transient | |
| `EagerFrame` | _frame.py:27 | native frame + `_arrays` column cache + schema cache | per model-data operation | column cache retains one ndarray per touched column |
| Penalty metadata on GroupMatrix (`omega`, `omega_components`, `component_types`, `lambda_policies`, `projection`) | core.py:662–666, discretized.py:43–47, etc. | set externally post-construction | copied field-by-field in ≥7 places (core.py:684–690, discretized.py:82–88,283–291,425–428; dm_builder.py:1229–1233,1249–1255,1290–1296,1330–1333,1349–1352) | q penalties live as per-group component lists here *and* in the REML penalty context |

---

## 4. COMPLEXITY TABLE

| Routine | Time | Memory (extra) | Notes |
|---|---|---|---|
| `discretize_column` (bins.py:9) | O(n log n) | O(n) | np.unique sort per feature, per build |
| `compute_R_inv` exact (dm_builder.py:88) | O(nnz·K + K³) | O(K²) | per SSP group per build **and per REML rebuild iteration** (sparse `multiply` makes an nnz-sized copy, :105) |
| `compute_R_inv` discrete | O(b·K² + K³) | O(K²) | support-space; cheap |
| `DenseGroupMatrix.gram` (core.py:48) | O(n·p_g²) | O(n·p_g) copy | `M * sqrt(W)[:,None]` full copy per call |
| `SparseGroupMatrix.gram` (core.py:74) | O(nnz·p_g) | O(nnz) copy | `.multiply` copy per call; `(Mw.T@Mw).todense()` |
| `SparseSSP.gram` (core.py:676) | O(n·d² + K²·p_g) | O(K²) | d = nnz/row (≈degree+1); numba, no copies |
| `CategoricalGroupMatrix.gram/rmatvec` (core.py:113) | O(n) | O(L) | bincount |
| `DiscretizedSSP.gram_rmatvec` (disc.py:65) | O(n + b·K² + K²·p_g) | O(b) | one fused numba pass over n |
| `DiscretizedTensor._factored_gram_raw` (disc.py:341) | O(b1·b2·K2² + b1·K1²·K2²) | O(b1·b2·K2) temp `WB2` + O(b1·K1²) `B1_outer` | e.g. 256²·10 ≈ 5 MB temp; per gram call |
| `DiscretizedTensor.matvec` (disc.py:389) | min(n_pairs·p_g, b1·p_g + n·K2) + O(n) | O(n·K2) worst | cost-model switch at :399–404 |
| `_disc_disc_2d_hist` (kernels.py:59) | O(n) | O(b_i·b_j) | per disc×disc cross block, per moments call |
| `_cross_gram_tensor_main` (algebra.py:387) | O(n·K_small + b_m·b_t·K + K-contractions) | O(b_m·b_t·K_small) channel hist | capped at 5M cells; fallback loops K columns × O(n) hists (:448–476) |
| `_cross_gram_tensor_tensor_shared_margin` (algebra.py:312) | O(n + b_sh·(b_i·b_j + K⁴)) | O(b_sh·b_i·b_j) | python loop over shared bins with einsum per bin (:366–374) |
| `_cross_gram` generic fallback (algebra.py:983) | O(n·p_i·(cost of rmatvec)) | **O(n·p_small)·2** | `toarray()` + `W[:,None]*X` copies inside solver loop |
| `_cross_gram_by_columns` (algebra.py:722) | O(p_small·(matvec+rmatvec)) = O(p_small·n) | O(n) | bounded-memory factored fallback |
| `_aggregate_group_matrix_columns` (algebra.py:222) | O(p_g·n) | O(n) | unit-vector loop; hit when disc×other exceeds hist cap |
| `MatrixExecutionPlan.moments` (execution.py:301) | O(G²·n) worst (all-discrete pairwise hists) + O(p²) init | O(p²) gram + hist grids | **per PIRLS iteration**; `np.zeros((p,p))` each call |
| `_pattern_support_summaries` (kernels.py:330) | O(n·G + U·(G + pairs)) | O(Σ marginals + Σ pair cells) ≤ 5M | U = unique patterns; single pass replaces G²/2 hists |
| `_build_pattern_plan` (centered.py:400) | O(n·G + n log n) | O(n) uint64 + O(U·G) | once per fit (carried across rebuilds) |
| `packed_centered_gram_rhs` (centered.py:689) | O(G·n) bincounts + O(G²·n) pair hists | O(p²) | per PIRLS iteration when pattern/factored paths decline |
| `MixedBinSpace.moments` (bin_space.py:84) | O(tabmat sandwich over augmented cols) ≈ O(n·(dense+1 per cat block)) + O(Σ b_i·b_j) block transforms | O(A²), A = augmented cols; all gated ≤ 64 MiB | plan build: O(n) per compressed group (CategoricalMatrix) + dense slab copy O(n·dense_cols) |
| `build_mixed_bin_space_centering_plan` (bin_space.py:155) | O(n·(dense_cols + G_disc)) | dense slab n×dense_cols copy + n×G_disc int32 codes | **rebuilt per REML outer iteration** (not carried by rebuild) |
| `_try_raw_spline_tabmat_centering` (centered.py:175) | O(nnz) sandwich + O(Σ K_i·K_j·p) transform | plan retains CSC+CSR ≈ 2·nnz | per iteration once eligible |
| `stable_centered_gram_rhs` (centered.py:805) | O(n·p²) flops, **2 full data passes**, n/8192 `row_subset().toarray()` materializations | O(chunk·p) + 2·O(p²) compensation buffers | last-resort; also `stable_centered_matvec` O(n·p) per call |
| `centered_gram_rhs` (centered.py:911) | O(n·p²) | O(chunk·p) + O(p²)·2 | fallback per iteration when no strategy certifies |
| `rebuild_design_matrix_with_lambdas` (dm_builder.py:1202) | Σ_groups [O(n) bincount or O(nnz) gram + O(K³)] | new DesignMatrix + new R_invs | **per fREML outer iteration** (reml/discrete.py:1154) |
| `_maybe_apply_tensor_side_constraints` (dm_builder.py:1158) | O(#prior interactions · cross_gram) + SVD O((Σp)·p_t²) | cross blocks | build-time only |
| `DesignMatrix.toarray` (group_matrix.py:440) | O(n·p) | **O(n·p) dense** | small-model oracles + chunked fallbacks only |
| `FactorSmooth.factor_smooth_sufficient_stats` exact (core.py:374) | O(n·d²) numba + O(L·K²·k) einsum | O(L·K²) raw grams | L = levels |
| `_csr_weighted_gram` (kernels.py:10) | O(n·d²) | O(K²) | |

---

## 5. SUSPECTS

**S1 — Dead compat wrapper surface in `group_matrix.py`.** `_block_xtwx`, `_block_xtwx_rhs`, `_block_xtwx_signed` (group_matrix.py:131–188), `_agg_by_bin` (:91), `_gram_any_sign` (:126), `_cross_gram_tensor_tensor` (:96), `_cross_gram_tensor_main` (:111) have zero production callers (grep across src/superglm; only tests import them). The `_block_xtwx*` trio also constructs a fresh `MatrixExecutionPlan` per call (`_execution_plan_for_blocks`, algebra.py:1035–1046), so any future caller would silently pay plan construction per Gram. Verify: run test-suite import graph; confirm nothing external (docs/notebooks) uses them.

**S2 — Duplicated, partially unused constants.** `_MAX_DISC_DISC_HIST_CELLS` / `_MAX_DISC_DISC_CHANNEL_HIST_CELLS` are defined in group_matrix.py:87–88 (never referenced in that file) and again in algebra.py:31–32 (the live copies); `_MAX_PACKED_HIST_CELLS` in centered.py:19 repeats the same 5M value with different semantics. Divergence risk if one is tuned.

**S3 — Unreachable branches in `_build_tabmat_split`.** tabmat.py:296–297 (`SparseSSPGroupMatrix → toarray()` densify) and the generic `else: toarray()` at :300–301 are dead: the guard at :257–269 returns None for every support-space type, and all remaining types have explicit branches. A future edit relaxing the guard would silently densify B@R_inv at O(n·p_g). Verify with coverage.

**S4 — `_defer_raw_spline_tabmat_plan` is dead code.** tabmat.py:109–124 defined, never called anywhere in src or tests. Its policy ("cold CSC construction loses to one stable data pass") is documented but not wired; either the policy was superseded by `TabmatCenteringState.raw_spline_eligible` latching in solvers/centered_system.py:230–250 or a call site was lost. Verify against git history.

**S5 — Per-REML-iteration DesignMatrix rebuild discards caches.** `rebuild_design_matrix_with_lambdas` (dm_builder.py:1202–1358) runs inside the fREML ρ loop (reml/discrete.py:1147–1186) and constructs a new DesignMatrix carrying only `_centered_pattern_plan` (:1356–1358). Consequences per outer iteration: (a) `mixed_bin_space_centering_plan` rebuilt from scratch — O(n) tabmat CategoricalMatrix per compressed group + n×dense dense-slab copy (bin_space.py:299–359); (b) `_centered_solver_supports` recomputed (legitimately λ-dependent via R_inv, but the categorical identity blocks are not); (c) fresh `MatrixExecutionPlan` and tabmat holders; (d) per-group `np.bincount(bin_idx, weights=sample_weight)` (dm_builder.py:1264,1312,1340) recomputed although `sample_weight` and `bin_idx` are fit-constants — the same exposure aggregate could be computed once per fit. For exact (non-discrete) SSP groups the rebuild recomputes the O(nnz) weighted CSR gram in `compute_R_inv` (dm_builder.py:105) every iteration. Verify with the existing `_t_rebuild_dm` timer in reml/discrete.py.

**S6 — Quintuplicated rebuild logic and triplicated metadata wiring.** dm_builder.py:1216–1353 repeats the same "resolve λ → recompute R_inv (projected or not) → reconstruct group → copy 5–7 metadata fields" block five times with class-specific weights; `_process_info` (dm_builder.py:428–639) wires omega/omega_components/component_types/lambda_policies in four separate branches. The metadata copy also appears in every `row_subset` (core.py:684–690; disc.py:82–88, 283–291, 425–428) — and those `row_subset` copies *omit* `lambda_policies` for SparseSSP/DiscretizedSSP/DiscretizedTensor (core.py:684–690, disc.py:82–88, 425–429) while SplineCategorical variants copy it (core.py:796–803, disc.py:285–291). If any consumer of a row-subset design reads `lambda_policies`, it silently sees None. Verify whether subsetted designs ever reach penalty construction (reml bootstrap path, reml/discrete.py:328).

**S7 — `moments` is O(G²·n) per PIRLS iteration for discrete designs without exactly one tensor.** The pattern-compression single-pass path (`_build_pattern_plan`, centered.py:410–414) requires `len(tensor_groups) == 1`; designs with zero or ≥2 tensor terms fall back to per-pair O(n) 2D histograms — in `packed_centered_gram_rhs` (centered.py:775–793) and in `_moments_impl` cross blocks (execution.py:430–436). With G ≈ 20 discrete smooths that is ~190 O(n) passes per iteration. mgcv's XWXd handles arbitrary term counts with one pass. Verify with a profile on a 10-smooth, no-tensor discrete fit (`block_hist2d_s` profile key).

**S8 — `DiscretizedTensorGroupMatrix` retains the materialized Kronecker `B_joint`.** disc.py:304–307 admits it is kept "for fallback compatibility in any code path that doesn't know about the factored representation"; it is (n_pairs, K1·K2) — up to 65 536×(K1·K2) floats per tensor — and is also duplicated per rebuild-skip path and consumed by `matvec`'s direct-pair branch (:405) and the pattern plan cross (:625–627). The B_joint could be derived on demand from the factored margins. Verify which consumers genuinely need it (grep `B_unique` uses where the object is a tensor).

**S9 — Generic `_cross_gram` fallback allocates O(n·p_g) twice inside the solver loop.** algebra.py:983–996: `toarray()` then `W[:, None] * X_i` for dense×dense / dense×sparse pairs, plus a python per-column rmatvec loop with `np.vstack`. For wide dense blocks (polynomials, non-SSP splines) this is repeated every moments call. Also `DenseGroupMatrix.gram`/`SparseGroupMatrix.gram` (core.py:48–50, 74–77) make full sqrt(W)-scaled copies per call. Verify by profiling `block_cross_fallback_s`.

**S10 — Centering-strategy cascade responsibility is split and duplicated.** Six strategies + certification live in `_group_matrix_centered.py` (990 lines) but the dispatch order, latching (`TabmatCenteringState`), and preflight economics live in solvers/centered_system.py:211–276; `_try_factored_tensor_centering` reaches back into `execution_plan._moments_prevalidated` (centered.py:250), and `_try_mixed_discrete_centering` re-implements the eligibility test that `build_mixed_bin_space_centering_plan` already performs (centered.py:333–361 vs bin_space.py:176–195 — two near-identical type-gate blocks that can drift). The DesignMatrix carries one cache slot per strategy. This is the single largest source of "which path executed and why" opacity in the subsystem. Verify: enumerate the strategy-selection matrix in tests (test_mixed_bin_space_centering.py) and check both gates agree on every layout.

**S11 — `_own_margin_cache` grows across REML iterations.** algebra.py:496–506 caches per-`(id(gm_main), id(bin_idx), n_bins)` on the *tensor* group matrix. Discrete tensors with penalty components are reused across λ rebuilds (dm_builder.py:1302–1309) while main-effect DiscretizedSSP groups are recreated each iteration with fresh ids → one new dict entry per main group per outer iteration, unbounded for long ρ optimizations. Correctness holds only because `bin_idx` arrays are shared across rebuilds (dm_builder.py:1348 passes `gm.bin_idx` through); an id()-reuse collision on `gm_main` paired with a *different* bin_idx object of the same id would return a stale margin — improbable but id()-keying on transient objects is fragile. Verify: len of `_own_margin_cache` after a 50-iteration fREML fit.

**S12 — File-organization mismatch: `validation.py` is actuarial charting, not design-matrix validation.** validation.py:1–7 docstring: lift/double-lift/Lorenz/Gini toolkit. Nothing imports it from the design-matrix subsystem. Any audit or refactor plan keyed on filenames will mis-scope it. (Included here because the audit brief listed it.)

**S13 — Threshold constants are scattered magic numbers with "measured" justifications.** 100-level categorical crossover appears in four places (execution.py:176, tabmat.py:212, tabmat.py:271–274, tabmat.py:291, bin_space/centered gates at centered.py:358); byte budgets of 64 MiB ×7 in bin_space.py:15–21; row thresholds 8k/10k/50k/5k/100k across tabmat.py:12–21, execution.py:21, centered.py:19–23. No shared policy module; tuning one crossover requires editing several files consistently. Verify: none is configurable at model level.

**S14 — `_resolve_group_lambda` third return value unused; `has_comp` bound and discarded at every call site** (dm_builder.py:1220, 1239, 1263, 1311, 1339). Minor dead code, signals interface drift.

**S15 — `compute_R_inv` uses explicit `np.linalg.inv` of a triangular factor** (dm_builder.py:110, :135). `scipy.linalg.solve_triangular`/`dtrtri` would be cheaper and slightly more accurate; K is small so impact is minor, but it runs per group per REML iteration (via S5).

**S16 — `SplineCategoricalGroupMatrix` stores both `B` and the row-sliced copy `B_level`** (core.py:717–736): duplicated storage O(nnz_level) and the full-B copy is only needed for `row_subset`/`rebuild`; also `toarray` densifies. The discretized variant avoids this. Verify retained-memory on a many-level spline-by-categorical fit.

**S17 — Chunked fallback paths reconstruct GroupMatrix objects per chunk.** `centered_gram_rhs`/`stable_centered_gram_rhs` (centered.py:844–872, 944–953) call `dm.row_subset(rows).toarray()` per 8192-row chunk: for FactorSmooth this re-runs full constructor validation per chunk (core.py:606–630), for DiscretizedSplineCategorical an argsort/searchsorted intersection per chunk (disc.py:258–276), n/8192 times, twice (mean pass + gram pass in `stable_`). The anchor row is additionally fetched via a 1-row DesignMatrix (centered.py:841–846). Acceptable as a last resort but silently expensive when certification keeps failing (ill-scaled columns ⇒ every iteration takes the O(n·p²) two-pass road with per-chunk object churn). Verify: count fallback activations via the missing profile key — note this path records no profile counter, unlike every other strategy.

**S18 — `DesignMatrix.matvec` tabmat fast path depends on side effects.** group_matrix.py:416–418 uses the retained split only if `holder.split is not None`, i.e. only after something else (moments/tabmat_split property) built it; the `_is_retained_tabmat_vector_candidate` gate (tabmat.py:216–239) never triggers a build itself. Whether the fast path engages therefore depends on call ordering elsewhere — behaviorally benign but non-deterministic performance. Verify which solver paths touch `tabmat_split` before the first matvec.

**S19 — `__setstate__` legacy-pickle shim.** group_matrix.py:280–313 migrates pre-shared-holder pickles (`_tabmat_split`/`_tabmat_built` keys) and tolerates missing keys via setdefault. Compat surface that will accrete; no version tag on the pickle format. Verify how old the supported pickle vintage must be.

**S20 — `EagerFrame._arrays` column cache retention.** _frame.py:113–129 caches every extracted column ndarray for the adapter's lifetime; if the EagerFrame is retained on the model (fit-data verification via `digest`), the extracted float64 copies of all feature columns are retained alongside the native frame — effectively 2× column memory. Verify whether model/base.py keeps the EagerFrame or only the digest.

---

### Cross-checks performed
- Protected semantics intact in this subsystem: built spline columns = k−1 via R_inv width (core.py:661, disc.py:42); discrete path shares `spec.build_knots_and_penalty` with exact path (dm_builder.py:730) so bases match; `discrete=True` gating is per-spec-then-model (dm_builder.py:138–148); select=True double-penalty components built at dm_builder.py:791–841 distinct from group-lasso machinery.
- mgcv parallels: discretized marginal storage + XWXd-style packed cross products explicitly cited (algebra.py:487–491, dm_builder.py:1305–1309).
