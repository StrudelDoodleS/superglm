# Architecture report — subsystem: tests-benchmarks

Audit target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` at origin/master `f082e9b`.
All paths below are relative to that worktree unless absolute. This subsystem does not itself implement
solver algebra; it *measures* and *pins* it. The report therefore emphasises (a) what benchmark
evidence exists, (b) what each harness actually measures, (c) which architectural properties the test
suite freezes, and (d) where coverage of solver/REML internals is thin.

---

## 0. HEADLINE BENCHMARK EVIDENCE (tracked results)

### 0.1 The flagship 30-rep MTPL2 comparison — superglm currently LOSES to the reference implementation

Tracked results (`git ls-files` confirms only these four files under `benchmarks/results/` are
committed; the rest of `benchmarks/results/` is gitignored via `.gitignore:34`):

| file | tool / method | n | median | mean | std | min–max | deviance | edf |
|---|---|---|---|---|---|---|---|---|
| `benchmarks/results/the reference implementation_30rep.json` | the reference implementation `bam(fREML, discrete=TRUE, weights)` (the reference implementation 1.9.3, R 4.5.2) | 678,013 | **1.567 s** | 1.561 s | 0.054 | 1.454–1.651 | 212023.256 | 44.198 |
| `benchmarks/results/superglm_30rep.json` | superglm `fit_reml(discrete=True)` "cached-W" | 678,013 | **2.102 s** | 2.103 s | 0.038 | 2.034–2.172 | 212055.369 | 43.342 |

- **Ratio superglm/the reference implementation = 1.34x** (computed exactly as `timing_30rep_compare.py:52` does:
  `sg["median_s"]/the reference implementation["median_s"]`). The project's stated goal ("match the reference implementation accuracy, beat it on
  speed") is *not currently met* on its own flagship scenario; superglm is ~34% slower, single-threaded,
  same machine.
- Accuracy parity at this scale: deviance differs by 32.1 (rel 1.5e-4), EDF differs by 0.86. Both are
  inside the loose test tolerances (see §4.3) but the EDF gap is a real fREML-vs-fREML discrepancy.
- Both harnesses measure a *fresh model fit per rep* with data prep outside the timer
  (`benchmarks/timing_30rep_superglm.py:60-69`, `benchmarks/timing_30rep_the reference implementation.R:42-56`), 1 BLAS thread,
  first rep discarded as warmup (`timing_30rep_superglm.py:82-83`). Methodologically clean and
  genuinely comparable: rate response + exposure weights on both sides, matched basis sizes
  (superglm `CubicRegressionSpline(n_knots=18/13/13)` ↔ the reference implementation `k=20/15/15 bs="cr"`; the mapping is
  documented at `benchmarks/reml_benchmark_harness.py:207-211`).

### 0.2 `benchmarks/results/multi_scop_discrete_convergence.csv`

Produced by `benchmarks/benchmark_multi_scop_discrete_convergence.py` (A/B of a "managed cleanup"
toggle for multi-SCOP discrete REML, both execution orders per repeat):

| dataset | n | baseline | optimized | speedup |
|---|---|---|---|---|
| synthetic | 25,000 | 0.547 s | 0.535 s | 1.023x |
| freMTPL2 | 678,013 | 3.037 s | 3.028 s | 1.003x |

Notable: on synthetic, `lambda_max_abs_diff = 184.67` (VehAge λ 15494.7 vs 15679.4) is recorded as a
*passing* run because predictions agree to 5.7e-5 — evidence that λ on a flat plateau is not a stable
quantity, which matters for any test that compares λ values directly (cf. `tests/test_discretize_fit.py:276`
asserting 10% relative λ agreement).

### 0.3 `benchmarks/results/fit_state_transaction_baseline.json` (718 KB)

Baseline for the wall-time suite of `benchmarks/benchmark_fit_state_trace.py`. 6 cases × 10 repeats,
each in a fresh subprocess. Median wall times: `dense_fit` 0.0035 s, `categorical_fit` 0.0684 s,
`spline_fit` 0.0077 s, `exact_reml` 0.0231 s, `discrete_reml` 0.0140 s, `compact_reml` 0.0161 s.
Metadata records `git_commit: 42b48af…`, **`git_dirty: true`** — the committed baseline was produced
from a dirty tree, which undercuts the harness's own strict comparison-context validation
(`benchmark_fit_state_trace.py:834-880` rejects mismatched numpy/pandas/CPU/threads but cannot
retroactively certify this baseline's code state). File size is dominated by full
`prediction_values`/`beta_values` vectors frozen per case in the `cases` summaries
(`benchmark_fit_state_trace.py:465-491`), while per-sample copies are deliberately stripped
(`_strip_fidelity_vectors`, lines 883-890).

---

## 1. MODULE MAP

### 1.1 Benchmarks — cross-tool comparison harnesses (the the reference implementation evidence chain)

| file | responsibility | key functions | callers / calls |
|---|---|---|---|
| `benchmarks/reml_benchmark_harness.py` (317) | superglm side of REML benchmark: synthetic Poisson/Gamma n=800 (exact+discrete, 3 reps) + MTPL2 678k (exact+discrete, 1 rep). Exports CSVs for R. | `run_one` :85-138 (times `fit_reml`, harvests `model._reml_result/_reml_lambdas/_reml_profile`), `run_synthetic_benchmarks` :144, `run_mtpl2_benchmarks` :187, phase-profile printer :263-298 | writes `results/superglm_results.json`, `bench_synthetic_*.csv`, `bench_mtpl2.csv`; consumed by `reml_benchmark_compare.py` and `reml_benchmark_harness.R` |
| `benchmarks/reml_benchmark_harness.R` (210) | the reference implementation side: `gam(REML)` on synthetics, `gam(REML)` on 200k MTPL2 subsample, `bam(fREML,discrete)` count+offset, `bam(fREML,discrete,weights)` (:173-191, the apples-to-apples config). | `extract_results` :33-57 | reads CSVs from Python harness; writes `results/the reference implementation_results.json` |
| `benchmarks/reml_benchmark_compare.py` (202) | joins the two JSONs, prints tables, λ comparison with explicit "not directly comparable" caveat :89-97, comparability notes :161-167 | `match_and_compare` :61, `print_lambda_comparison` :89 | reads `{superglm,the reference implementation}_results.json` (both gitignored → compare inputs are ephemeral) |
| `benchmarks/timing_30rep_superglm.py` (124) | 30-rep MTPL2 discrete REML timing (§0.1). Model spec :49-54, timing loop :60-78 | writes `results/superglm_30rep.json` (tracked) |
| `benchmarks/timing_30rep_the reference implementation.R` (106) | the reference implementation 30-rep counterpart, `bam` formula :44-55 | reads `results/bench_mtpl2.csv` (must run Python harness first :30-32); writes `results/the reference implementation_30rep.json` (tracked) |
| `benchmarks/timing_30rep_compare.py` (61) | prints both + ratio :51-57 | reads the two tracked 30rep JSONs |
| `benchmarks/benchmark_tensor_ti_freq.py` (1063) | discrete MTPL2 tensor-interaction stress matrix: baseline, +1/2/3 tensors, +spline-by-cat, mixed; docstring says its purpose is "a tracked reproduction of the current discrete=True fit_reml tensor-interaction failure mode" (:11-15). Exports train/test CSVs for R comparator. | `build_superglm_cases`, `build_fairness_cases`, `build_case_deltas`, `FitControls`, `thread_control_metadata` (imported by `tests/test_tensor_ti_benchmark_matrix.py:7-17`) | uses `multiprocessing` to isolate fits |
| `benchmarks/benchmark_tensor_ti_the reference implementation.R` (316) | the reference implementation oracle for the same 7-case matrix, `bam(discrete=TRUE, fREML)`, "behavior oracle without copying the reference implementation source" :12-15 | reads the exported split; writes `tensor_ti_the reference implementation.json` |

### 1.2 Benchmarks — regression-gate harnesses (subprocess-isolated, schema-validated)

| file | responsibility | key items | consumers |
|---|---|---|---|
| `benchmarks/benchmark_fit_state_trace.py` (1003) | Authoritative wall-time + RSS suite over 6 canonical fit paths (`CASES` :176-183: dense/categorical/spline `fit`, exact/discrete/compact `fit_reml`). One fresh subprocess per sample (`_run_worker_subprocess` :778-807); authoritative timing on an uninstrumented fit; a *second* fresh model in the same worker gets tracemalloc + tabmat kernel-call counting (`_run_worker_case` :617-735, `_count_tabmat_kernel_calls` :206-228). Numerical-fidelity comparator `compare_runs` :231-323 (rtol 1e-10); context validator :834-880; suite-quality gate rejecting unconverged fits :513-524. | `PreparedCase` dataclass :68-78; measurement contract declares phase timings are NOT regression gates :928-935 | baseline `results/fit_state_transaction_baseline.json`; unit-tested by `tests/test_fit_state_trace_benchmark.py` (34 tests) |
| `benchmarks/benchmark_dataframe_boundary.py` (758) | Same subprocess pattern for dataframe-boundary overhead across pandas/polars backends; 6 scenarios (`SCENARIOS` :254-261) incl. `predict_exact`, `predict_fast_discrete` (calls private `model._predict_fast_discrete` :285). Comparator `_compare` :634-714 enforces wall-time thresholds **3% predict / 5% fit** (:649), traced-peak and RSS growth ≤ max(1 MiB, 5%) (:670-679), plus kernel-call/matrix-structure/numerical equality. | `PreparedScenario` :63-74, `_kernel_counts` :289-331 patches `tabmat.SplitMatrix`, `MatrixExecutionPlan._moments_impl`, and 4 compressed group-matrix types (imported from private `superglm._group_matrix._group_matrix_discretized` :33-39) | `tests/test_dataframe_boundary_benchmark.py` |
| `benchmarks/benchmark_multi_scop_discrete_convergence.py` (528) | §0.2 A/B harness; counterbalanced execution orders; monkey-patches `superglm.reml.scop_efs` internals via `unittest.mock.patch` (:19, :26) to toggle the cleanup gate | `SummaryRow`, `_aggregate_lambda_metrics`, `_execution_orders_for_repeat`, `_prediction_metrics` | `tests/test_multi_scop_discrete_walltime.py:8-13` imports them |

### 1.3 Benchmarks — profiling & exploratory scripts

| file | responsibility |
|---|---|
| `benchmarks/profile_structured_credibility.py` (835) | cProfile/tracemalloc/system-telemetry harness for random-effect + factor-smooth structured fits; results narrated in `benchmarks/structured_credibility_profile_summary.md` (tracked, dated 2026-07-24, base rev 86b2a1c): e.g. K=30 structured-vs-Gram 70.6x; K=300,k=10 stress case 35.16 s with ~15.0 s penalty cross traces + ~13.9 s tabmat cross-Gram aggregation named as "the next cProfile target"; the 817.8 MB eager p×p identity in runtime canonicalization fixed to 24.2 MB. Smoke-tested by `tests/test_structured_credibility_benchmark.py` (loads harness via importlib :14-21). |
| `benchmarks/profile_superbooster_interactions.py` (477) | cProfile wrapper around `superbooster_interaction_challenger.py` cases; uses `superglm.profiling.harness` utilities (SystemSampler etc.) — those utilities are unit-tested in `tests/test_profiling_harness.py`. |
| `benchmarks/_constrained_fit_profile.py` (303) + `benchmarks/profile_constrained_fit_paths.py` (302) | shared SCOP/QP profiling scenarios (`ProfileScenario` :24-33) + driver; tested by `tests/test_constrained_fit_profile.py:7`. |
| `benchmarks/profile_factor_smooth_construction.py` (69) | one-shot cProfile of `FactorSmooth` marginal construction. |
| `benchmarks/multi_scop_scaling.py` (226), `benchmarks/scop_discrete_limit.py` (350), `benchmarks/scop_lambda_sensitivity.py` (563), `benchmarks/benchmark_shape_constraints.py` (199), `benchmarks/benchmark_scop_exact_support.py` (52), `benchmarks/debug_fit_reml_convergence.py` (159) | SCOP/shape-constraint sweeps (scaling, λ-grid sensitivity, identifiability plots, REML-debug trajectory plots via `superglm.model.reml_debug`). |
| `benchmarks/scop_numerical_experiments.py` (307) | drives **private prototype solver modes** via `configure_scop_prototype`/`reset_scop_prototype` from `superglm.solvers.scop_newton` (:25, :100) — MINRES/truncation experiments; contains a machine-absolute path `MTPL2_PATH = Path("/home/mhick/…/scratch/r_experiments/mtpl2_prepared.csv")` (:33). |
| `benchmarks/benchmark_freq_gini.py` (178), `benchmark_personal_lines_serving.py` (268) | holdout Gini for exact+discrete REML; serving-latency simulation (predict-only vs request-path, thread-pool batches). |
| `benchmarks/superbooster_interaction_challenger.py` (442), `superbooster_shap_compression_spike.py` (365), `superbooster_visual_report.py` (732) | GBM-hybrid spikes (XGBoost on backbone eta via `base_margin`, SHAP leaf-signature compression, Plotly report). Reach into `superglm.model.base` internals (`superbooster_shap_compression_spike.py:33`). |

### 1.4 Tests — shared infrastructure

| file | responsibility |
|---|---|
| `tests/conftest.py` (28) | only two hooks: `--run-browser` gate for Playwright editor tests (:6-16) and autouse matplotlib-figure cleanup (:19-27). No shared numerical fixtures — every test file builds its own data. |
| `tests/_datasets.py` (48) | cached freMTPL2 parquet loader, search order `$SUPERGLM_DATA_DIR` → `~/.cache/superglm` → `<repo>/data` (:20-27); returns None → real-data tests skip on CI. |
| `tests/_fit_state_oracles.py` (79) | `ModelBehaviorSnapshot` (:16-27) + `assert_model_behavior_unchanged` (:63-79): strong-exception-guarantee oracle asserting identity (`is`) of 16 private model projections (`_result`, `_dm`, `_fit_state`, `_summary_cache`, … :31-48) after a failed refit. |
| `tests/_wood_reml_oracles.py` (232) | independent dense REML/LAML oracles from Wood (2011, 2016); docstring mandates they "must not call SuperGLM's REML, PIRLS, determinant, or rank helpers" (:3-6). `DenseWoodState` :29-44. Used by `tests/test_wood_reml_oracles.py` (429 lines). |
| `pyproject.toml:119-124` | `testpaths=["tests"]`, one custom marker `slow` (14 files use it). No timeout, no benchmark plugin. |

### 1.5 Tests — the four files named in the brief

**`tests/test_reml.py` (1904)** — the central REML contract file. Classes and what they pin:
- `TestOmegaStored` :62-109 — SSP group matrices store ω (PSD; CRS ω ≠ second-difference penalty :106-109).
- `TestPenaltyComponents` :115-191 — `build_penalty_components` ≡ `build_penalty_caches` (atol 1e-14); SSP congruence round-off must not make a rank-1 prior indefinite (:146-169).
- `TestPenalisedXtwxInvOmega` :197-331 — covariance path: dict-vs-scalar λ equivalence, `_penalised_xtwx_inv_gram` ≡ QR path (atol 1e-8 :285-286), and a *fusion* contract: covariance must use `gram_rmatvec`, never separate `gram`/`rmatvec` compressed passes (monkeypatch pytest.fail :301-310).
- `TestMgcvStyleSmoothTestInput` :378-456 — summary p-value uses weighted-QR factor (R'R = X'WX, atol 1e-8 :444-448).
- `TestREMLMultistart` :504-587 — λ2_init ∈ {default, 0.1, 100} → objective spread < 1e-2, log-λ spread < 0.5.
- `TestREMLConvergence` :593-632; `TestREMLSelectionPenaltyRejected` :638-682 (fit_reml requires selection_penalty=0 — protected-semantics boundary); `TestREMLFallbacks` :685-765 (no-REML-groups falls back to ordinary solver *dispatch*, monkeypatch on `fit_pirls` :725-728).
- `TestREMLSelectTrue` :771-840 — select=True double penalty: `x1:null`/`x1:wiggle` λs exist and log|S|+ decomposes per component (atol 1e-12 :834).
- `TestREMLBackwardCompat` :846-1039 — custom link/family *without* declared REML curvature is rejected (`NotImplementedError` "explicit ordinary REML curvature" :887, :936) and leaves `_fit_state is None`; declared Fisher pairs get the W(ρ) correction (:940-1039).
- `TestREMLDiscreteRobustness` :1112-1227 — **discrete-vs-exact parity tolerances**: deviance rel < 1e-3, |ΔEDF| < 0.5 (Poisson :1175-1180); Gamma cached-W: rel < 5e-3, |ΔEDF| < 1.0 (:1222-1227); λ2_init=1e5 must still converge (:1141-1144).
- `TestMultiPenaltyPostFitInference` :1233-1323 — post-fit covariance frozen at fitted multi-penalty S (RankInfo freeze; rtol 1e-12 :1294).
- `TestREMLObjectiveFastPath` :1326-1444 — Poisson cached-XtWX objective must not call `dm.matvec` (:1333-1337); overflowed working weights / non-finite dW rejections.
- `TestDiscreteCachedSolve` :1447-1716 — discrete internals: line-search surrogate evals ≥ full profiled solves (:1483-1484); `_solve_cached_profiled_system` ≡ augmented (p+1) system incl. log|H| (rtol 1e-10/1e-12 :1556-1559); `_penalty_block_trace` compact ≡ materialised (1e-12 :1517); tensor-pair closed-form log-det summaries ≡ generic objective/gradient/Hessian (1e-10 :1714-1716).
- `TestStaleREMLClearing` :1719-1784 — `fit()`/`fit_path()` after `fit_reml()` must null `_reml_lambdas/_reml_penalties/_reml_result` (protects fit/fit_reml separation).
- `TestMultiPenaltyTensorREML` :1790-1904 — ti() margins get separate λs; component ωs PSD; single-spline path unchanged.

**`tests/test_discretize_fit.py` (1499)** — discrete=True architecture contract:
- Accuracy: coefficients rel < 0.10, deviance rel < 0.005, prediction max-rel < 0.05 at n=2000/256 bins (:70-82); REML deviance rel < 0.01 (:238); fREML λ per-group rel < 0.10 (:276) — see §0.2 caveat about λ plateau fragility.
- Structure: spline groups become `DiscretizedSSPGroupMatrix` with `n_bins` (:84-100); per-feature discrete overrides model-level (:102-120); categoricals stay exact (:137); low-unique-support columns use exact support (predictions match exact rtol 1e-8, :543-573); default n_bins=256 (:521).
- Discrete REML uses PIRLS not the direct solver (:281); rejects nonpositive n_bins (:331).
- Prediction parity: `_predict_fast_discrete` ≡ exact canonical predict for main effects/tensor terms incl. shifted holdout (:605-682); tensor metadata frozen at fit time (:683); exact spline/tensor prediction dedupes repeated support values (monkeypatch counters :577, :840, :881).
- Tensor discrete: retains only support-sized marginal bases (:971); decomposed tensors share `tensor_id` (:1005); rebuild preserves tensor type and freezes unprojected basis (:1028-1099); penalty-context and tensor-pair-summary caches are reused (:1314, :1356); public PIRLS controls forwarded (:1261); `interaction_mode`/`runtime_validation` kwargs validated (:1127-1259).

**`tests/test_theory_invariants.py` (1957)** — statistical + backend-algebra invariants:
- `TestSolverTheoryInvariants` :40-181 — Poisson score equations at convergence; **integer frequency weights ≡ row replication** (:70, the `sample_weight` exposure semantics oracle); group-lasso KKT conditions (:113); row-order invariance (:150).
- `TestBackendLinearAlgebraInvariants` :183-1645 — ~45 tests asserting every compressed/blocked Gram product (`_block_xtwx`, `_cross_gram`, tensor/spline-categorical/categorical crosses) matches a dense oracle, AND negative-dispatch contracts (e.g. cross-gram must not call `tensor.rmatvec` :763, must not materialise tensor rows :844, own-margin detection cached :930, Hessian same-slice fast path :1194).
- `TestPredictionTimeContracts` :1647-1857 — unseen/NaN categorical raise; spline extrapolation is flat clamp (:1747); constant predictors contribute zero.
- `TestREMLInteraction` :1860-1957 — spline×categorical REML converges, λ>0, interaction deviance < main-effects.

**`tests/test_rank_policy.py` (1896)** — 60 tests on the shared centered numerical-rank policy
(`SHARED_RANK_POLICY`, `decompose_gram/factor`, `build_centered_system`): rank boundary shared by
gram and factor rules (:143, :1395); centered system avoids raw-moment cancellation (:153); tabmat
centering must not materialise categorical/discrete/tensor rows (:200, :839, :910); factor
certification governs equal-rank truncation (:565); log-pdet stability under extreme column scaling
(:1310-1373); alias/estimability suppression (:1744-1811); EDF1 spectral branch ≡ direct influence
(:1712); "legacy inference ≡ profiled rank state" (:1869).

### 1.6 Tests — other perf/parity clusters (skimmed)

- `tests/test_reml_the reference implementation_parity.py` (227): hardcoded the reference implementation references (R 4.5.2 / the reference implementation 1.9-3) at :32-45;
  deviance rel < 1% (:72, :149), EDF ±2.0 (:94), λ within 10x order-of-magnitude only (:114, :203),
  Gamma scale rel < 20% (:183), REML iter ≤ 15 (:134). Skips unless fixture CSVs generated by
  `scratch/r_experiments/reml_parity_reference.R` exist (:52-55).
- `tests/test_realdata_parity.py` (503): freMTPL2 parity — exact REML deviance rel < 0.1% vs the reference implementation
  (:350), EDF ±2.0 (:359), Pearson scale rel < 10% (:376); **discrete-vs-exact on real data**:
  deviance rel < 0.5% (:419), EDF ±2.0 (:428); NB2 theta near-Poisson check (:241). Skips without data.
- `tests/test_cached_w_validation.py`: docstring (:1-11) — the trust gate for the cached-W discrete
  fREML optimizer: exact-vs-discrete agreement, restart robustness, W-refresh sensitivity, large-n smoke.
- `tests/test_tweedie_profile_performance.py` (960): "correctness-first and has no wall-clock
  assertion" (:3-4); regression currency is **evaluation counts** (density passes, series budget
  ≤ 4096 :959-960; vectorisation checks :900, :929) — the suite's only count-based perf gates.
- `tests/test_multi_scop_discrete_walltime.py` (295): despite the name, tests *harness plumbing*
  (order counterbalancing, λ-metric aggregation, CSV schema, cleanup-freeze bookkeeping on mocked
  `scop_efs` :112-268) — no wall-time assertion.
- `tests/test_fit_state_trace_benchmark.py` (550): 34 tests locking the wall-time harness itself —
  required case set (:82), tabmat dispatch expectations per fixture (:115-183), determinism (:185),
  drift rejection (:229-276), scaled workers converge (:404), env forced single-thread (:486).
- `tests/test_tensor_ti_benchmark_matrix.py` (271): locks the 7-case tensor benchmark matrix and
  fairness/attribution controls of `benchmark_tensor_ti_freq.py`.
- `tests/test_structured_credibility_benchmark.py` (131) and `tests/test_dataframe_boundary_benchmark.py`
  (100), `tests/test_profiling_harness.py` (100): smoke/contract tests for the remaining harnesses.
- Oracle families: `test_wood_reml_oracles.py` (independent Wood-paper oracles),
  `test_statsmodels_coef_consistency.py`, `test_factor_smooth*_the reference implementation_parity.py`,
  `test_random_effect_the reference implementation_parity.py` — four independent external reference systems (the reference implementation-R numbers,
  Wood formulas, statsmodels, dense NumPy oracles).

Total: 3,758 test functions across ~180 files (108,794 lines in `tests/`).

---

## 2. DATA FLOW

### 2.1 Cross-tool the reference implementation comparison chain
1. `reml_benchmark_harness.py` generates synthetic frames (n=800, 2 numeric cols) and/or loads MTPL2
   parquet (n=678,013), applies the canonical prep (clip ClaimNb≤4, Exposure≥0.01, DrivAge 18–90,
   VehAge 0–20, BonusMalus 50–150, log1p Density; :66-79) and materialises `y_freq = ClaimNb/Exposure`
   (length n) and exposure weights.
2. It exports `bench_mtpl2.csv` (n×7, ~678k rows, only if absent :199-205) and synthetic CSVs so both
   tools consume byte-identical data.
3. Timed region = `model.fit_reml(...)` only (`run_one` :107-109). Fit-side matrices (X of shape
   n×p with p ≈ 1+19+14+14+5 ≈ 53 built columns for the MTPL2 spec, per-penalty ω of shape
   (p_g×p_g), q = 4 penalties) are materialised *inside* the library, not the harness.
4. Rep-0 harvest: `_reml_result` (n_reml_iter, converged), `result.deviance/effective_df/phi`,
   `_reml_lambdas` (q floats), optional `_reml_profile` (13 phase timings + 4 counters :274-298 — all
   keys verified to exist in `src/`, so the phase list is live, not stale).
5. R side reads the CSVs, fits gam/bam, `extract_results` harvests deviance/EDF/sp; both sides write
   JSON; `reml_benchmark_compare.py` joins on name substrings (:64-77).

### 2.2 Regression-gate suites
`benchmark_fit_state_trace.py`: parent process → per-sample `subprocess.run(sys.executable, __file__,
--worker …)` with forced `PYTHONHASHSEED=0`, all 4 thread env vars = 1 (:770-775) → worker builds case
(n = 1.2k–6k rows scaled by `--case-scale`), times one uninstrumented fit, then builds a *second*
identical model for tracemalloc + tabmat kernel counts → JSON record (validated, :326-426) → parent
aggregates medians, freezes prediction/beta vectors, gate-checks convergence, optionally diffs against
a baseline file and returns exit 2 on numerical drift (:992-999). Arrays in flight per worker:
X (n×cols), prediction (n), beta (p) — all O(n) / O(p); the harness itself adds no super-linear work.

`benchmark_dataframe_boundary.py`: same pattern; additional flow is backend duality (pandas vs polars
frames through `_frame` :86-93) and predict-path scenarios where a fitted model is prepared and warmed
*before* the timed `predict` on n=60k rows (:192-251).

### 2.3 Test-side data flow
Tests overwhelmingly generate independent synthetic frames per test function (rng seeds inline);
real-data tests stream through `tests/_datasets.py`. There is no shared fitted-model cache across
test files, so expensive REML fits (e.g. `test_reml.py:552-562` fits the same 1800-row model three
times) are re-run per test session — a test-suite wall-time cost, not a library cost.

---

## 3. STATE OBJECTS

| object | file:line | fields | lifecycle | overlap |
|---|---|---|---|---|
| `PreparedCase` | `benchmarks/benchmark_fit_state_trace.py:68-78` | model, X, y, sample_weight, offset, fit_method, fit_kwargs | built per worker invocation; consumed once | near-duplicate of `PreparedScenario` |
| `PreparedScenario` | `benchmarks/benchmark_dataframe_boundary.py:63-74` | model, operation, X, y, seed, sample_weight, offset, kwargs | same | near-duplicate of `PreparedCase` (adds `operation`/`seed`; drops `fit_method` naming) — two harnesses, one concept |
| `ProfileScenario` | `benchmarks/_constrained_fit_profile.py:24-33` | name, engine, n, k, n_constrained, repeated_support, discrete, use_fremtpl | shared between two SCOP profilers | scenario-descriptor role also served by `CaseConfig` in `profile_structured_credibility.py` and per-script dataclasses (`Row` :16-19 in `benchmark_scop_exact_support.py`, `RunRow` in `scop_discrete_limit.py`, `MultiSCOPScenario` in `multi_scop_scaling.py`, `SensitivityScenario` in `scop_lambda_sensitivity.py`, `Scenario` in `debug_fit_reml_convergence.py`) — six bespoke scenario dataclasses with overlapping fields |
| `SummaryRow` | `benchmarks/benchmark_multi_scop_discrete_convergence.py` (imported at `tests/test_multi_scop_discrete_walltime.py:8`) | 26 columns incl. gate/freeze counters, λ diffs | one per dataset; CSV row | none |
| `ModelBehaviorSnapshot` | `tests/_fit_state_oracles.py:16-27` | model `__dict__` identity, `_fit_state`, `_fit_revision`, predictions, beta, intercept, deviance, summary, 16 named projections | captured before an injected fit failure; asserted after | this is the test-side mirror of the model's fit-state transaction design; its projection list (:31-48) is a hardcoded copy of the model's private attribute inventory — drifts whenever `SuperGLM` grows a cached attribute |
| `DenseWoodState` | `tests/_wood_reml_oracles.py:29-44` | beta, intercept, deviance, penalty_quad, slope_xtwx, full_hessian, logdet_s_plus, penalty_nullity | per-oracle-evaluation | intentionally overlaps `PIRLSResult`+penalty caches — by design, as an independent re-derivation |
| `_PhiFixture`/`_CountedProfile`/`_BoundedReference`/`_EndToEndProfileCase` | `tests/test_tweedie_profile_performance.py:31-60` | Tweedie fixture + counted-pass records | per-test | none |

Baseline JSON payloads themselves act as persisted state objects with explicit `schema_version = 1`
(`benchmark_fit_state_trace.py:44`, `benchmark_dataframe_boundary.py:48`) and metadata blocks
(git commit/dirty, CPU model, thread env: :566-589 and :411-444 respectively) — two parallel,
slightly divergent metadata schemas.

---

## 4. COMPLEXITY TABLE (harness-side; the measured library work is other subsystems' scope)

| routine | time | memory | notes |
|---|---|---|---|
| `timing_30rep_superglm.py` main loop :60-78 | 30 × T_fit(n=678k, p≈53, q=4) ≈ 65 s | one DataFrame (n×12) + model per rep; models kept only via loop variable (GC'd) | fresh model per rep is correct methodology; deviances/edfs lists O(reps) |
| `reml_benchmark_harness.py:199-205` CSV export | O(n) write of 678k×7 CSV (~40 MB) | O(n) | one-time; R side re-parses it every run (`read.csv`, no cache) |
| `benchmark_fit_state_trace.py` suite (default warmups=2, repeats=10) | (2+10)×6 = 72 subprocess spawns, each running the fit **twice** (timing + diagnostic) → 144 fits | per-worker O(n·cols); parent holds all 60 sample dicts incl. n-length prediction vectors until `_strip_fidelity_vectors` :883-890 | subprocess-per-sample is the dominant fixed cost for these sub-100 ms cases (interpreter+import ≈ several × the measured fit); acceptable for isolation but makes the suite minutes-long for milliseconds of signal |
| `_summarize_samples` :433-510 | O(cases × repeats × n) for `stable_vector` allclose over prediction vectors :465-476 | freezes one n-length vector per case into JSON | this is why the 718 KB baseline is 718 KB |
| `benchmark_dataframe_boundary.py` `_kernel_counts` :289-331 | O(1) per patched call | patches 3 classes + 4 compressed types | monkeypatching production classes in-process; restored in finally |
| `compare_runs` :231-323 / `_numerically_equal` :615-631 | O(p + n) per case | — | rtol 1e-10 pointwise on beta and predictions — a *bit-stability* gate, far tighter than any statistical tolerance |
| `test_theory_invariants.py` dense oracles (e.g. :184-221, :1410-1645) | each builds dense X (n×p, n ≈ 200–2500) and computes X'WX at O(np²) | O(np) | fine at test scale; these are the tests that would catch dense-fallback regressions in the compressed kernels |
| `test_reml.py` `TestREMLMultistart` :550-562 | 3 full REML fits (n=1800, 8 features, ≤50 outer iters) | — | one of the slowest single tests; not marked `slow` |
| Test suite overall | 3,758 tests / 108k lines; 14 files carry `slow` markers | — | no pytest timeout, no wall-time regression gate anywhere in `tests/` (only count-based Tweedie budgets :959-960) |

---

## 5. SUSPECTS

### S1. The flagship benchmark contradicts the project goal — superglm 1.34x slower tha reference
- `benchmarks/results/superglm_30rep.json:71` (median 2.102 s) vs `benchmarks/results/the reference implementation_30rep.json:13`
  (median 1.567 s), same n=678,013, single thread, same machine, matched model.
- Why suspicious: this is the repo's own tracked apples-to-apples evidence, and the current dev branch
  in the main worktree is literally named `perf-the reference implementation-fit-performance`. Any architecture decision should
  treat this 0.54 s gap as the primary optimisation target; the harness's phase profile
  (`reml_benchmark_harness.py:274-298`: dm_build / irls working / gram / eigh solve / W-correction /
  Hessian+Newton / line search) already tells you where to look.
- Verify: re-run both 30-rep scripts at f082e9b; pull `_reml_profile` phase breakdown for the MTPL2
  discrete case and rank phases.

### S2. Real-scale discrete parity gap in EDF is untested
- Tracked evidence: EDF 43.342 vs the reference implementation 44.198 (Δ0.86) and deviance Δ32.1 at n=678k. The only
  discrete-vs-exact EDF tolerances are |Δ| < 0.5 at n=800 (`tests/test_reml.py:1180`), < 1.0 for Gamma
  (:1227), and ±2.0 on real data (`tests/test_realdata_parity.py:428`). Nothing pins the *the reference implementation-vs-
  superglm-discrete* EDF at scale; the 30rep JSONs record it but no test reads them.
- Why suspicious: "discrete=True must not silently drift from exact REML" is protected; a 0.86 EDF gap
  vs the reference implementation's fREML could be legitimate basis-difference or could be drift — currently unfalsifiable.
- Verify: run exact REML on an MTPL2 subsample and triangulate exact-vs-discrete-vs-the reference implementation EDF.

### S3. Parity tolerances are scattered and mutually inconsistent
- Discrete-vs-exact deviance: 1e-3 (`test_reml.py:1175`), 5e-3 Gamma (:1222), 5e-3 fit-level
  (`test_discretize_fit.py:76`), 1e-2 REML (`test_discretize_fit.py:238`), 5e-3 real data
  (`test_realdata_parity.py:419`). λ agreement: 10% (`test_discretize_fit.py:276`) — despite the
  repo's own benchmark CSV showing λ_max_abs_diff=184.67 on a plateau being harmless
  (`benchmarks/results/multi_scop_discrete_convergence.csv`, synthetic row).
- Why suspicious: there is no single documented parity contract; a refactor could pass one file's gate
  while violating another's, and the λ-based assertions are fragile by the project's own evidence.
- Verify: enumerate all discrete/exact/the reference implementation tolerance assertions (grep done above) and check which have
  ever been near-threshold in CI history.

### S4. Two near-identical ~1000-line measurement harnesses
- `benchmarks/benchmark_fit_state_trace.py` and `benchmarks/benchmark_dataframe_boundary.py` duplicate:
  subprocess worker protocol (:745-807 vs :482-523), `_worker_environment` (:770-775 vs :500-505),
  `_rss_peak_bytes` (:592-600 vs :334-338), `_cpu_model` (:555-563 vs :400-408), metadata blocks
  (:566-589 vs :411-444), JSON writers, order-alternation logic (:186-189 vs :589-597), and numerical
  comparators (`compare_runs` :231-323 vs `_numerically_equal` :615-631) with *different* strictness
  and different schemas.
- Why suspicious: divergent duplication of the measurement contract itself; a fix to one comparator
  (e.g. NaN handling) won't reach the other. Each also has its own 100–550-line test file.
- Verify: diff the two `_compare`/`compare_runs` semantics; check whether any consumer depends on the
  schema differences.

### S5. MTPL2 preparation copy-pasted across 11 benchmark scripts
- The clip/log1p prep block appears in `timing_30rep_superglm.py:26-37`, `reml_benchmark_harness.py:66-79`,
  `benchmark_freq_gini.py:34-40`, `benchmark_personal_lines_serving.py`, `benchmark_tensor_ti_freq.py`,
  `benchmark_multi_scop_discrete_convergence.py`, `_constrained_fit_profile.py`, `scop_lambda_sensitivity.py`,
  and all three superbooster scripts (grep count: 11 files).
- Why suspicious: any prep drift silently invalidates cross-script comparability with the R-side CSVs
  (which are exported once and reused, `reml_benchmark_harness.py:199-205`). Also duplicated
  `.worktrees` parent-path fallback boilerplate in ≥6 scripts (e.g. `benchmark_freq_gini.py:28-29`).
- Verify: hash-compare the prep outputs across scripts; check `bench_mtpl2.csv` freshness logic (it is
  never regenerated once present).

### S6. Committed baseline was recorded from a dirty tree
- `benchmarks/results/fit_state_transaction_baseline.json` metadata: `git_commit: 42b48af…`,
  `git_dirty: true`. The harness itself refuses cross-environment comparisons
  (`benchmark_fit_state_trace.py:834-880`) but has no gate on baseline dirtiness at record time
  (`_git_dirty` :541-552 is recorded, not enforced).
- Why suspicious: the whole point of the file is to be an authoritative wall-time/fidelity anchor.
- Verify: whether CI or a skill actually consumes this baseline via `--compare`, and regenerate clean.

### S7. Machine-absolute path and private prototype hooks in a committed benchmark
- `benchmarks/scop_numerical_experiments.py:33` hardcodes
  `/home/mhick/python_projects/superglm/scratch/r_experiments/mtpl2_prepared.csv`; :25 imports
  `configure_scop_prototype`/`reset_scop_prototype` from `superglm.solvers.scop_newton` — production
  solver module carrying experiment-only prototype configuration (MINRES / cross-block truncation).
- Why suspicious: dead-weight experimental surface inside `solvers/scop_newton.py` (1091 lines) that
  only a non-portable benchmark exercises; classic stale-path candidate.
- Verify: grep prototype-mode reachability from `fit`/`fit_reml`; confirm no default-path behaviour
  depends on prototype state.

### S8. Benchmarks are load-bearing test dependencies
- `tests/test_multi_scop_discrete_walltime.py:8`, `tests/test_fit_state_trace_benchmark.py:7-27`,
  `tests/test_tensor_ti_benchmark_matrix.py:3-17`, `tests/test_dataframe_boundary_benchmark.py:6-7`,
  `tests/test_constrained_fit_profile.py:7`, `tests/test_multi_scop_scaling.py:5` import `benchmarks.*`
  as a package; `tests/test_structured_credibility_benchmark.py:14-21` loads a harness by file path.
- Why suspicious: `benchmarks/` is effectively a second source tree (10,651 lines) with test-enforced
  API, but it sits outside `src/`, outside packaging, and mixes tracked regression gates with
  throwaway spikes (superbooster, plotly reports) in one namespace. Responsibility boundary is unclear.
- Verify: which benchmark modules are imported by tests (above) vs never referenced — the latter set
  (`superbooster_*`, `scop_numerical_experiments`, `debug_fit_reml_convergence`) are exploratory and
  could rot silently.

### S9. Harnesses and tests reach deep into private model attributes
- `reml_benchmark_harness.py:114-127` (`_reml_result`, `_reml_lambdas`, `_reml_profile`),
  `benchmark_fit_state_trace.py:648-651` (`_reml_profile`, `_fit_stats`, `_dm._tabmat_split`),
  `benchmark_dataframe_boundary.py:244/285` (`_predict_fast_discrete`), `tests/_fit_state_oracles.py:31-48`
  (16 private projections asserted by identity).
- Why suspicious: any model-state refactor (a likely audit outcome given six `_reml_*`/`_fit_*`
  attribute families) breaks the measurement layer and the strong-exception oracle simultaneously;
  the private-attribute inventory in `_fit_state_oracles.py` is a manually maintained copy.
- Verify: cross-reference with the model-subsystem reader's state-object inventory.

### S10. No wall-time regression gate runs in the test suite
- Grep confirms zero timing assertions in `tests/` (only Tweedie evaluation-count budgets,
  `test_tweedie_profile_performance.py:959-960`, and the deliberately gate-free phase timings,
  `benchmark_fit_state_trace.py:931` `phase_timings_are_regression_gates: False`). Wall-time
  enforcement exists only in manually-run comparators (`benchmark_dataframe_boundary.py:649` 3%/5%;
  `benchmark_fit_state_trace.py` exit-2 on *numerical* drift only).
- Why suspicious: the 1.34x the reference implementation gap (S1) could regress further without any automated signal; the
  performance story relies on humans re-running harnesses.
- Verify: CI config / skills for any benchmark invocation (none found under `tests/`).

### S11. Thinly covered solver/REML internals (import-count survey)
- `src/superglm/solvers/dispersion.py` — **0** test-file imports; `src/superglm/reml/convergence.py`
  (1 file, `tests/test_reml_convergence.py`), `src/superglm/reml/efs.py` (1 — and
  `tests/test_reml_efs.py:1-6` says the file now tests only the *rejection* contract of the old EFS
  path), `src/superglm/solvers/working_rows.py` (1), `src/superglm/reml/runner.py` (2 — called
  "legacy fixed-point REML runner" by `tests/test_reml_runner.py:1` yet still live via
  `src/superglm/model/reml_ops.py:9`). By contrast `penalty_algebra` (24), `irls_direct` (23),
  `pirls` (16), `rank` (12) are heavily pinned.
- Why suspicious: the biggest modules with the thinnest direct coverage are exactly the orchestration
  layers (`runner.py` 413 lines, `efs.py` 426 lines, `scop_efs.py` 1851 lines with 6 importing files
  mostly via mocks); "legacy" naming in tests vs live import in `reml_ops` is a doc-code mismatch.
- Verify: coverage run scoped to `superglm.reml.runner`/`efs`/`solvers.dispersion` to distinguish
  indirect (integration) coverage from none.

### S12. `reml_benchmark_compare.py` operates on untracked inputs
- It reads `results/{superglm,the reference implementation}_results.json` (:17-25), which are gitignored; only the 30rep JSONs
  and the multi-SCOP CSV are tracked. The richer multi-configuration comparison (synthetic exact vs
  discrete, MTPL2 exact) therefore has no committed evidence — the tracked story is discrete-only.
- Verify: regenerate `superglm_results.json` to capture the exact-path MTPL2 timing (the reference implementation's
  `gam(REML)` full-data fit is documented as infeasible — the R harness subsamples to 200k, :124-131).

---

## 6. WHAT THE SUITE PINS WELL (for the later redesign phase to respect)

1. Bit-stability of fits under refactor (rtol 1e-10 pointwise on beta/predictions across repeats and
   against baselines — `benchmark_fit_state_trace.py:231-323`, `benchmark_dataframe_boundary.py:615-631`).
2. Dispatch contracts, not just values: dozens of monkeypatch-fail tests assert *which kernel* runs
   (no dense materialisation, no separate gram+rmatvec passes, cached detections) — any orchestration
   rewrite must preserve call-shape, not only numbers.
3. Protected semantics have direct tests: exposure-weight ≡ row-replication (`test_theory_invariants.py:70`),
   fit/fit_reml separation + stale-REML clearing (`test_reml.py:1719-1784`), select=True vs
   selection_penalty (`test_reml.py:638-682, 771-840`), k↔built-columns mapping documented in parity
   headers (`test_realdata_parity.py:33-38`, `benchmarks/reml_benchmark_harness.py:207-211`).
4. Independent oracles exist at three levels (the reference implementation numbers, Wood-formula dense oracles, dense NumPy
   gram oracles), so solver algebra can be rewritten with strong external anchors.
