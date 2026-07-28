# SuperGLM Architecture Audit

**Target:** `StrudelDoodleS/superglm` @ `origin/master` = `f082e9b` ("Add structured credibility effects and factor smooths, #165")
**Date:** 2026-07-28 · **Method:** 3-stage multi-agent audit (9 subsystem readers + 2 cProfile baseline profilers → 7 specialist auditors → 19 adversarial verifiers), all claims verified against code at this commit; measured claims via cProfile (project standard). Evidence artifacts: subsystem maps, auditor reports, raw `.prof`/`.pstats` dumps and probe scripts under `~/.claude/jobs/e3eef6ba/tmp/audit/`.
**Scope guard:** analysis only — no source changes. All recommendations respect the protected semantics in `CLAUDE.md` (fit vs fit_reml distinct; discrete=True must not drift from exact; select= vs selection_penalty distinct; `k`/k−1 contract; sample_weight=exposure; sklearn wrapper contracts).

---

## A. Current architecture map

### A.1 Layer stack (as-built)

```
User API      SuperGLM (model/api.py, 1734 ln facade) · sklearn wrappers · editor/plotting/export
Orchestration model/fit_ops.py → fit_workspace (transactional attempt) → reml_execute → reml_finalize
              (5 layers between fit_reml() and the first line of optimizer math)
Smoothing     reml/direct.py   exact damped-Newton on ρ=log λ  (Wood 2011 geometry + W(ρ) correction)
  optimizers  reml/discrete.py POI/fREML cached-W Newton       (BAM-style Fisher boundary, no W-correction)
              reml/scop_efs.py SCOP-EFS fixed point            (shape-constrained terms)
              reml_execute.py:90 fixed-λ monotone path
              [DEAD] reml/runner.py (413 ln FP optimizer) + reml/efs.py (426 ln EFS) — runtime-proven unreachable
Coefficients  solvers/irls_direct.py (2601 ln; _fit_irls_direct_once monolith 2129 ln, 5 backends, 13 modes)
              solvers/pirls.py (BCD + group-lasso prox, SSP-reparametrised)
              solvers/constrained_qp.py (shape-constraint active set) · scop_newton.py (SCOP joint Newton)
              solvers/rank.py SHARED_RANK_POLICY (pivoted-Cholesky certification + fallbacks)
Penalty       reml/penalty_algebra.py + multi_penalty.py: S(λ) assembly (two semantics), log|S|₊ (two algorithms),
  algebra     ranks/nullity (two M_p definitions), eigencontexts
Design matrix dm_builder.py → DesignMatrix → GroupMatrix classes (_group_matrix/*): Dense / SparseSSP /
              DiscretizedSSP / DiscretizedTensor / Categorical / structured; MatrixExecutionPlan kernel dispatch;
              centering strategy ladder (solvers/centered_system.py, 6 rungs, self-certifying)
Features      features/*: Spline (13 _spline_* modules), Categorical, interactions/tensors, FactorSmooth,
              RandomEffect; SSP reparametrisation (λ-dependent basis whitening)
Inference     inference/* + stats/*: 4 covariance representations, EDF, Wood p-values, term inference
Frame layer   _frame.EagerFrame (pandas/polars boundary; never enters solvers — as documented)
```

### A.2 Definitive fit_reml dispatch (runtime-verified)

```
SuperGLM.fit_reml → fit_ops.fit_reml → _fit_reml_in_workspace
  ├ selection penalty forbidden (forced λ1:=0)          ⇒ use_direct is INVARIANTLY True
  ├ no penalised groups            → plain-fit fallback
  ├ monotone ∧ all λ fixed         → run_fixed_monotone_reml (SCOP fixed / irls_direct fixed)
  ├ any unfixed SCOP term          → scop_efs.optimize_scop_efs_reml
  └ else → optimize_reml_best → reml.direct.optimize_direct_reml
       ├ discrete=False → exact damped-Newton loop (direct.py:364-944)
       └ discrete=True  → delegation (direct.py:105-133) → discrete.optimize_discrete_reml_cached_w
  then finalize_reml_fit (terminal refit(s) + terminal LAML + φ + atomic publication)
```

Consequences: `reml/efs.py`'s `optimize_efs_reml` is called 0 times on any argument combination (spy-verified);
`reml/runner.py`'s fixed-point optimizer is reachable only through a 3-layer adapter that nothing calls.
The docstring in runner.py ("the three optimizer kernels") describes an architecture that no longer exists.

### A.3 Exact vs discrete: what actually differs (verified)

| Aspect | exact (`direct.py`) | discrete (`discrete.py`) |
|---|---|---|
| Newton geometry | Wood 2011 grad/Hessian on ρ, + W(ρ) implicit-diff correction | same, minus W-correction (cached-W Fisher boundary) |
| Inner solve per λ-step | full PIRLS to convergence, **incl. per line-search trial** | one PIRLS step refreshes cached centered system; per-trial O(p³) re-solve, no data pass |
| Gram formation | rebuilt from data every IRLS iteration | cached centered XᵀWX, POI updates |
| log\|S\|₊ | similarity-transform recursion (multi_penalty.py) | tensor closed form (penalty_algebra.py:429-554) — **agree to ≤3e-6** (probed over 16 decades of λ) |
| Criterion parity | — | **verified: no drift** (deviance/EDF/λ parity within test tolerances; measured n-scaling: EDF 41.93 vs 41.82 at n=1M) |

### A.4 The Gram-formation strategy ladder (heart of large-n behaviour)

`build_centered_system` (centered_system.py:211-275) tries, in order: packed bin-space Gram (all groups
Discretized/Tensor/Categorical — any number of categoricals) → mixed bin-space plan (Dense allowed, but bails at
≥2 categoricals, `_group_matrix_centered.py:349`) → tabmat plans (refuse discretized designs / raw-spline+cat
mixes) → **chunked-dense fallback** `centered_gram_rhs` (O(n·p²), per-8192-row `row_subset→toarray→hstack`
marshalling ≈ half its runtime). Verified trigger (adversarial re-measure): a Disc/Tensor/Cat-only design runs
packed; adding **any plain `Numeric()` (Dense) group** to a ≥2-categorical model fails every rung, so the
bread-and-butter frequency model (splines + 2 cats + numerics) pays the fallback on **both** exact and
discrete paths at every n — see RFC-1.

---

## B. Complexity map (measured + code-verified)

Notation: n rows, p built columns, m smooth terms, q penalty components (q ≥ m; multi-penalty smooths add more),
G groups, b bins/group, L categorical levels. cProfile on Poisson/log frequency models with exposure weights.

### B.1 By n (p=64 fixed: 4 ps-splines k=10 + cats 8/20 + 2 numerics)

| n | fit() | fit_reml exact | fit_reml discrete | discrete/exact |
|---:|---:|---:|---:|---:|
| 5k | — | 2.2 s | 0.39 s | 5.7× |
| 20k | — | 5.5 s | 1.8 s | 3.0× |
| 80k | 1.5 s | 8.9 s | 3.4 s | 2.6× |
| 320k | — | 21.5 s | 6.9 s | 3.1× |
| 1M | — | 53.7 s | 17.6 s | 3.05× |

- Per-Gram-pass cost is O(n) in both paths (slope 0.7–1.0); wall-time sublinearity is falling iteration counts.
- Exact at 1M: 82% of wall = 59 chunked Gram passes (43.8 s; ≈half marshalling, not FLOPs); 39% = W-correction
  signed Grams (one per spline group per Newton iter). Factorisation at p=64: negligible.
- Discrete at 1M: 68% = 14 chunked Gram passes. Its advertised O(b·k²) bincount Gram **never fired** (ladder
  cliff, RFC-1); the 3× speedup is entirely fewer passes (cached W) + no W-correction.
- fit() at 80k: 1/3 solver, **2/3 fixed overhead**, of which parity-validation passes ≈32% of wall (RFC-5).
- Memory at 1M: modest (chunked); no O(n·p) dense materialisation on the discrete path except tensor B_joint
  retention (up to ~42 MB/tensor, fallback-only consumer).

### B.2 By p and q (n=20k fixed)

| config | p | q | wall | dominant cost |
|---|---:|---:|---:|---|
| 5 splines | 45 | 5 | 4.8 s | W-correction 51% |
| 15 splines | 135 | 15 | 59.9 s | W-correction 58% |
| 25 splines | 225 | 25 | 172 s | **W-correction 85%** (147 s: 325 signed Grams, each built column-at-a-time — 1.8M sparse matvecs) |
| 40 splines | 360 | 40 | >520 s (timeout) | — |
| 4 splines + 2×25-level cats | 84 | 4 | 2.9 s | PIRLS Grams |
| 4 splines + 2×100 | 234 | 4 | 8.6 s | PIRLS Grams |
| 4 splines + 2×400 | 834 | 4 | 54 s | Gram 27 s + dense Cholesky 16 s (p³ emerging) |
| group-lasso fit(), 40 splines | 360 | — | 1.7 s | Gram/Hessian formation |
| group-lasso fit(), 80 splines | 720 | — | 3.3 s | ~linear in p |

**Headline:** exact-REML cost ≈ O(outer-iters · q · n · p²) with a 10–50× Python/scipy constant on the
W-correction track. **q, not p, is the dominant driver**: p≈230 costs 8.6 s at q=4 but 172 s at q=25.
With q fixed, growth is p^1.1→1.4 trending to p³ (dense factorisation). The BCD/group-lasso path is ~linear
in p at this n. Memory: O(q·p²) dense derivative matrices on the exact path (RFC-7) caps exact REML near
p≈2–5k; peak RSS 726 MB at p=834 incl. ~300 MB interpreter baseline.

### B.3 By m and q — where each path's outer loop spends

- Exact: per outer iter = full PIRLS (i IRLS iters × O(n·p²) Gram) + q signed Grams O(q·n·p²) + q×q Newton
  algebra (cheap until p~10³) + line-search trials (each a **full PIRLS**, RFC-12).
- Discrete: per outer iter = 1 POI step (cached re-solve O(p³)) + per-trial O(p³); but DM + penalty
  eigenstructure + tensor summaries rebuilt unconditionally each iter (RFC-9); bootstrap + terminal refits
  ≈2× the λ-loop cost on the flagship (verified; RFC-11).
- SCOP-EFS: EFS fixed point, each trial a full inner fit (≤8+4 trials).

### B.4 Benchmark-evidence correction (verified twice, independently)

The repo's tracked flagship comparison (`benchmarks/results/superglm_30rep.json`: median 2.102 s vs mgcv 1.567 s,
"1.34× slower") is **stale and unrunnable**: `timing_30rep_superglm.py` uses two removed ctor/fit kwargs
(`lambda1=`, `exposure=`) and TypeErrors at HEAD. Re-measured at f082e9b on the same freMTPL2 n=678k task
(deviance parity 212055.39 reproduced): **master now beats mgcv, ~1.36 s vs 1.57 s** (independent verifier
certifies ≥1.25× improvement over the tracked superglm numbers even under heavy box load). The public speed
narrative is currently backed by broken, dirty-tree-provenance artifacts — see RFC-15.

---

## C. Confirmed decluttering findings

All items below survived adversarial verification (call-graph claims re-grepped across src+tests+benchmarks;
"dead" means zero production callers at f082e9b). LOC ≈ lines directly removable or consolidated.

**C-1 · Dead REML optimizer stack (~950 LOC) — high.** `reml/runner.py` (413) + `reml/efs.py` (426) +
adapter chain `api._run_reml_once`/`_optimize_efs_reml`/`_optimize_discrete_reml_cached_w` → `reml_ops.py`
adapters + orphaned `inference/covariance._penalised_xtwx_inv*` (~200) — runtime-proven unreachable (spied
across 12 fit_reml configurations incl. discrete/select/SCOP/tensor/monotone). **Caveat from verification:**
not test-only churn — `superglm.reml.run_reml_once` / `optimize_efs_reml` / `inference.compute_coef_covariance`
are public re-exports and `docs/guide/optimization.md` §13.3 documents `optimize_efs_reml`, so deletion needs a
deprecation cycle + doc update; `tests/test_reml_runner.py` imports a private helper directly. The M_p
rank-vs-nullity inconsistency lives only in this dead code (live paths all use Wood-correct nullity), and dies
with it — except the `gradient.py:353` fallback, which should pass `hessian_rank=1+p` like `objective.py:271`.

**C-2 · Dead/broken code scatter (~290 LOC across 14 verified units) — medium.** Includes
`scop_newton._joint_objective_from_bin_etas` (would raise AttributeError if ever called), `group_matrix.py`
compat trio building a fresh execution plan per call, `_defer_raw_spline_tabmat_plan` (never called), two
unreachable tabmat branches whose relaxation would silently densify B@R_inv, `spline._weighted_quantile_knots`,
`tweedie._saddlepoint`, dead irls_direct helpers (:283/:299/:334), `penalty_algebra.build_penalty_caches`,
`gradient._penalty_block_trace`.

**C-3 · Post-solve finalization copied 6×, then recomputed — medium.** The eta/mu/null-mu/fit-stats block exists
at fit_ops.py:857-882/991-1015/1203-1227, reml_execute.py:153-174/234-255, reml_finalize.py:669-678 and is then
**fully recomputed** by `_prime_fit_caches` (2× design matvec + 2× 25-iter null-model Newton per fit).

**C-4 · direct/discrete duplicated candidate blocks (~350 mirrored LOC) — medium.** ~7 near-verbatim blocks
(bootstrap FP-init, select-snap, phi resolution, modified-Newton step, latch) with identical magic constants —
and **one substantive divergence already exists** (phi/nullity fallback: direct falls back to
`hessian_rank=1+p`, discrete does not). The `select_snap` "ignored" comment on a live protected-semantics
parameter is an active foot-gun.

**C-5 · select=True twin assembly — medium (protected-contract adjacent).** `dm_builder.py:791-841` duplicates
`_spline_select.py:86-125` line-for-line; runtime probe shows outputs **byte-identical today** (max|Δ|=0.0), and
the `component_types` label fork (`'wiggle'` vs `'difference'`) is latent (sole semantic consumer tests
`=='select_null'`). Consolidate now, while equal; a select-parity test alone cannot detect the label fork — add
a non-select lambda-policy term to the validation.

**C-6 · Inference/covariance 4-way sprawl — medium.** Four covariance representations dispatched by triplicated
branch ladders (state_ops, metrics, coef_tables); bordered-inverse formula 3-4×; EDF 3×; per-group SE loop 4×;
Wood-test orchestration 4× with two copies swallowing bare `Exception` into NaN p-values; the integer-rank
branch makes `_mixture_pvalue` dead (doc-code mismatch with the module's claimed Davies/Imhof evaluation).

**C-7 · Model-state registry mirrors — medium.** ~50-slot attribute inventory hand-mirrored in 6 places; two
parallel transactional workspaces (FitWorkspace, FittedStateRevision); two overlapping input-mutation guards
(FitDataGuard + FitGeometryGuard) both digesting X, plus a full y snapshot per retained fit.

**C-8 · SCOP dual state + parity residue — medium.** Frozen `_SCOPGroupSpec/State` mirrored by a stringly-keyed
`_scop_state` dict with hand sync at 3 sites; production-dead `_scop_joint=False` Gauss–Seidel branch;
prototype MINRES/truncation hooks (`configure_scop_prototype`) in the production module, exercised only by one
benchmark script; terminal Hessian refreshed via a full **discarded** Newton step.

**C-9 · dm_builder rebuild wiring — medium.** Resolve-lambda/recompute-R_inv/copy-fields block quintuplicated;
penalty metadata copied field-by-field in 7+ places; several `row_subset` copies silently drop
`lambda_policies` (real None-hazard for subsetted-design consumers, e.g. discrete bootstrap).

**C-10 · Compat facades & aliases — low (aesthetic).** `solvers/structured.py` 253-line facade re-exporting ~60
underscore-private names; `inference/term.py` shell; triple `monotonize` alias chain; half-finished `_IRLSState`
migration alias.

**C-11 · Misplaced modules — low.** `validation.py` (789 ln) is actuarial lift/Gini charting, unrelated to its
name-implied role; `profiling/harness.py` is benchmark telemetry inside the statistical package;
`benchmarks/` doubles as a load-bearing second source tree (7 test files import `benchmarks.*`).

**C-12 · Stale docs that misdescribe live behaviour — low.** runner.py docstring (describes a deleted
architecture); efs.py cites moved line numbers; irls_direct tol docstring (1e-6 vs actual 1e-8);
CardinalCRSpline documents nonexistent params; `IterationDiagnostics.raw_w_*` always duplicate `w_*`;
`_t_eta` metric never incremented.

**C-13 · Tweedie profile module (5,993 ln) — medium (reader-verified, not adversarially probed).** Layering
inversion (`distributions.Tweedie` → profiling orchestration → model+solvers), import-time numba/pandas cost
for all users, ~1,750 ln of phi branch-certification with O(n)-heavy fallback scans inside the optimizer loop,
duplicated fit-guard/dispatch/winner-validation blocks, exact-float cache keys defeating optimizer-probe reuse.
Separable: density math (~700 ln) vs orchestration vs certification.

**C-14 · Benchmark/test hygiene — medium.** Flagship harness broken at HEAD (B.4); committed baseline recorded
from a dirty tree; two ~1,000-ln harnesses duplicating the measurement contract with divergent comparators;
MTPL2 prep copy-pasted across 11 scripts; **no automated wall-time regression gate anywhere** — the speed goal
has zero CI signal.

---

## D. Target solver/REML architecture

Grounded in confirmed findings only; every component maps existing code. (Verified verdicts: ADOPT for 1-6;
REJECT for stochastic log-det, iterative preconditioners, fused single-pass kernel, full IRLS merge — see G.)

```
superglm/
  matrixops/                       ← _group_matrix/*, group_matrix.py
    operator.py   DesignMatrixOperator: weighted_moments(W; rhs, signed=, centered=, cols=)
                  → Weighted/CenteredMoments; matvec/rmatvec; whitened_pass(L) → (leverage, q grams);
                  owns the strategy cascade + certification + per-fit DENSE-COLUMN ANCHOR SHIFT
                  (verification amendment: certificate alone rejects ill-located numerics)
    policy.py     single home for every crossover threshold / byte budget / hist cap
                  (today: 100-level crossover in 4+ places, 7×64MiB budgets, 5 row thresholds)
    kernels/      numba kernels + group-matrix classes; bin-space plan generalised to ≥2 cats
                  (per-pair accumulation — the fused one-sweep variant is measured-slower, see G-1)
  linalg/
    rank.py       SHARED_RANK_POLICY unchanged in semantics; O(p⁴) representative loop replaced by
                  order-respecting incremental factorization (earliest-column convention preserved;
                  policy version bump + set-validity assertions, NOT identical-pivots assertions)
    factor.py     HessianFactor protocol + FactorizationDenseHessianFactor retaining L (today only
                  the explicit inverse is kept); structured Schur factors; pure-H QP solves routed
                  here (the indefinite KKT saddle system stays on its own path — verified)
  penalty/
    system.py     PenalizedSystem: components + eigenstructure computed once per fit; S(λ) as block
                  list (kills dense-S BCD updates); ONE log|S|₊ algorithm + derivatives (closed form;
                  the similarity-transform recursion retired — measured agreement 1.9e-6, same ranks
                  over 49-cell λ-grid to 1e16); ONE M_p=nullity definition; legacy `=`-assembly path
                  deleted (it silently drops component-named penalties reachable TODAY via the public
                  lambda2 setter — B1 verification)
  glm/
    curvature.py  CurvatureProvider: working rows, dW/dη, d²W/dη², observed bundles, Fisher/observed
                  classification, constant-W flag (today duplicated across working_rows /
                  w_derivatives / observed_geometry / scop_geometry, two mode-score policies)
    solver.py     PIRLS chassis split (not merged): direct | BCD-prox | QP | SCOP inner steps;
                  returns typed CoefficientSolveResult{result, factor: HessianFactor,
                  working: WorkingSystemCache, log_det_H, rank} — replaces the stringly cache_out
  reml/
    derivatives.py DerivativeEngine: gradient/Hessian with BATCHED WHITENED W-CORRECTION —
                  one chunk pass computing leverage + all q signed grams as BLAS dgemms
                  (identities verified to 1e-16; same O(T·q·n·p²) flops at ~10× better constant;
                  gradient-only mode O(R(n·p² + q·n)) via leverage identity)
    optimizers/   direct (exact Newton) | poi (discrete cached-W) | scop_efs | fixed-monotone —
                  thin drivers over (CoefficientSolver, PenalizedSystem, DerivativeEngine, Operator);
                  runner.py + efs.py deleted after deprecation cycle
  model/          public surface unchanged; adapter stack collapsed (api → fit_ops → optimizer);
                  ONE terminal refit owned by finalize; validation size-gate extended to fit()
```

Data flow, exact path, per outer iteration (target): operator supplies centered moments → solver returns
factorization-backed factor + typed working cache → DerivativeEngine does ONE whitened pass (leverage + q BLAS
grams) → PenalizedSystem supplies log|S|₊ + blocks from cached eigenstructure → line search re-solves against
cached moments (POI-style) with an exact Armijo re-check at acceptance (verified safe; pure-surrogate trials are
not).

---

## E. Ranked RFC backlog

Ranking = verified impact ÷ risk. Sev/conf are post-verification. "Validation" = the gate that proves the win.

| # | RFC | Sev/conf | Verified evidence | Expected effect | Key risk | Validation |
|---|---|---|---|---|---|---|
| 1 | **Dense-anchored raw-moments centering rung** (+ lift ≥2-cat mixed gate) | crit/confirmed | 10.7× kernel measured; agreement 1.7e-15; cliff hits standard frequency shapes on BOTH paths; anchor amendment required (certificate rejects VehPower-like columns) | discrete 1M: 17.6→~6-8 s; exact 1M: 53.7→~25-30 s | cancellation on ill-scaled columns (anchored accumulation measured ~1e-15 safe); certificate + chunked fallback retained per call | ladder-dispatch counter test incl. a Numeric + 2 cats fixture; discrete-vs-exact parity suite; n-scaling re-profile |
| 2 | **Batched whitened W-correction** | crit/confirmed | 39-85% of exact fits; 1.8M column matvecs → q dgemms; identities verified 1e-16; ~10× constant confirmed | a_m25: 172→~25-35 s; exact 1M: −15-20 s | needs factor-L retention (RFC-7); validate by fixed-ρ derivative equality + converged objective/prediction — NOT λ-trajectory equality (reordering noise ~1e-12) | test_reml_fd.py FD gradients; Wood-oracle tests; a_m15/a_m25 re-profile |
| 3 | **λ_max calibration fix** (drop /n **and** add family null-score factor (dμ/dη)/V for non-canonical links) | high/confirmed | boundary reproduced digit-for-digit at exactly n×; auto ⇒ λ=true/(10n); fit_path grid shares the helper; GroupElasticNet α factor included | selection_penalty="auto" and fit_path become meaningful; unblocks screening (RFC-17) | changes fitted models for auto/fit_path users → **bugfix release + changelog**, no deprecation cycle (verified recommendation); docstrings become true for canonical links | new KKT boundary test (24/24 zero at 1.01×λ_max, 23/24 at 0.99×); Gamma/Tweedie log-link calibration test |
| 4 | **Rank-policy incremental factorization** | high/confirmed | measured 203.7 s vs 0.31 s (660×) for ONE deficient decompose_gram at p=400 (default threading); reachable per IRLS iteration | rank-deficient wide fits usable | earliest-representative convention must be preserved; dpstrf unsound for it; policy version bump; set-validity (not identical-pivot) assertions | test_rank_policy.py + new deficient-p benchmark |
| 5 | **fit() validation size gate** | high/confirmed | 2 unconditional full-data parity passes; ~32% of a 1M-row fit(); gate exists but only wired to fit_reml (4 sites) | fit() at 1M ~1.5× faster | parity check loses coverage above gate — same tradeoff already accepted for fit_reml | existing canonicalization tests + n-scaling fit() point |
| 6 | **Dead-optimizer resolution** (runner.py, efs.py, adapters, orphaned inference algebra) — ⚠ **benchmark before deleting** | high/confirmed dead; **deletion no longer recommended unconditionally** | 12-config runtime spy proof; ~950 LOC. **But both dead optimizers are Fellner–Schall-family** — the exact family that avoids the W(ρ) correction which is 39-85% of exact-path runtime — and `runner.py`'s variant has Anderson(1) acceleration, which is the standard remedy for FS's linear convergence. `scop_efs.py` proves the family works in-tree. | either −950 LOC, **or** a cheap optimizer that beats exact Newton on smooth-heavy models | deleting a cheap optimizer whose per-iteration cost omits the dominant term would be a real error; deprecation warnings must be held until the benchmark resolves, or a public API gets un-deprecated | **RFC-6a spike (T1):** fix the M_p rank→nullity bug, wire EFS behind a private flag, benchmark vs exact Newton on a_m5/a_m15/a_m25 (q=5/15/25) and b_L800. Then delete the loser. Adapters and orphaned inference algebra can be deleted regardless. |
| 7 | **Factorization-backed HessianFactor** (retain L; route discrete cached-solve + pure-H QP solves through shared policy) | high/confirmed | dense backend keeps only explicit inverse; per-candidate p×p pseudo-inverse materialisation at irls_direct.py:2373; discrete fork's weaker certificate | enables RFC-2; ~halves exact-REML peak memory ((2q+3)p² → ~q·p²+p²); O(q·p²) floor remains (pairwise traces) | KKT saddle system is indefinite — keep it out (verified); certificate semantics unified upward not weakened | test_scop_exact_support / cached-W validation suites |
| 8 | **PenalizedSystem consolidation** (single assembly, single log|S|₊, single M_p; delete legacy `=` path) | high/confirmed | legacy path reachable TODAY via public lambda2 setter with component-named dicts ⇒ **silent total penalty drop** (probe: ‖S_block‖=0 vs 20.8); two log|S|₊ algorithms agree 1.9e-6 | latent-bug closure + one algebra home; small exact-path speedup (~5 eigh per evaluation retired) | assembly change must be bit-safe on reachable configs (measured 1.2e-14 scalar-λ agreement) | 2-penalty tensor fitted both paths; lambda2-dict regression test |
| 9 | **Discrete build/support reuse** | high/confirmed | full-column np.unique + dense basis at ~n uniques inside discrete build; raw-vs-binned constraint inconsistency real | discrete build cost ~O(n) once, not per structure | constraint must use UNWEIGHTED bincount (exposure-weighted variant would silently change centering semantics — verified) | discretize-fit parity + build-time benchmark |
| 10 | **DesignMatrixOperator + policy module** | high/confirmed | two Gram stacks with duplicated eligibility gates; thresholds scattered (4+ homes for the 100-level crossover) | single dispatch home; prerequisite hygiene for RFC-1/2 | pure refactor; certification latching must stay caller-visible | kernel-dispatch counter tests |
| 11 | **Terminal/bootstrap dedup** | med/confirmed | duplicate finalize refit at identical λ (verified); 3 DM rebuilds (one IS read — old_gms); bootstrap 23% of flagship PIRLS block; recoverable ~7-9% of flagship wall (not 23%) | flagship −7-9%; bootstrap A/B (max_iter cap) validated safe | atomic terminal-publication contract; one rebuild is consumed — dedupe by threading, not deletion | fREML parity suite + fixed 30-rep flagship |
| 12 | **Exact-path cached line-search trials** | med/adjusted | trials are 17% of b_L800 optimizer wall (92% claim corrected — candidate fits dominate and stay); 8 trials measured | b_L800-class −~15% | exact Armijo re-check at acceptance mandatory; fall back to exact trials before declaring line_search failure | b_L800 re-profile + convergence suite |
| 13 | **Behavioral small-fix batch**: max_iter=0 UnboundLocalError → ValueError; port polarization merit_delta to pirls (extended with group-lasso delta term); QP pure-H solves through rank policy + surface QPResult.converged; observed-fallback silent-drop → warning | med/confirmed | all reproduced through public API; merit divergence shown to misread a −1.5e-7 improvement as +1.7e-5 | correctness hygiene | none material | targeted unit tests per item |
| 14 | **Sparse-factor + selected-inverse (Takahashi) backend** — generalises the one-block structured Schur | high/adjusted | single-dominant-block gate verified; realistic credibility models are multi-block; the shipped #165 factor protocol (`selected_inverse_diagonal`, `trace_inverse_penalty`, `logdet`) is already the right seam, hand-rolled for one block | wide-categorical/credibility fits escape dense p³ **and** exact REML survives at large p (§H.4); subsumes RFC-7's dense-inverse removal | fill-reducing ordering quality determines everything; must bound DLR trace-machinery rank growth; keep the dense path as reference oracle | dense-vs-sparse agreement on models small enough for both (the #165 design already mandates this); b_L800-class + credibility benchmark |
| 15 | **Benchmark repair + CI perf gate** | med/confirmed | flagship harness TypeErrors at HEAD; dirty-tree baseline; no wall-time gate anywhere; master actually beats mgcv now (B.4) | speed narrative becomes enforceable; regressions visible | keep R/mgcv side version-pinned | fixed 30-rep harness in CI (reduced reps) with threshold vs tracked baseline |
| 16 | **CurvatureProvider + typed solver contract** | med/confirmed | four geometry homes, two divergent mode-score normalisation policies; stringly cache_out consumed via bare string keys | one geometry seam; enables backend evolution | mode-score reconciliation may flip accept/reject near boundaries — needs tolerance study | observed-geometry + SCOP suites |
| 17 | **Sequential strong rules on fit_path** (after RFC-3) | low/confirmed | per-group Hessians+eigensystems rebuilt for ALL groups every outer iteration; KKT verifier exists (`_composite_kkt_violation`) | path fits on wide models | screening economics modest at insurance p; sequenced last | path-KKT exactness test |

Also fix opportunistically (low, no RFC needed): eta/nll threading into LAML objective (0.23% now, matters for
Binomial/NB at scale); B_joint retention; on-pass fused deviance/link kernels; row_subset lambda_policies;
component_types label unification.

---

## F. Implementation tranches

*(Revised after §H. The first draft of this section was written before the parallelism and dtype findings and
before §H's cost model; three corrections are marked ⚠ and carried below.)*

### F.0 The real dependency graph

Tranches are a **review and release order, not a dependency chain**. There are only four hard edges; everything
else is prioritisation, and independent items can run concurrently if more than one person is on this.

```
RFC-15 (working benchmark + CI gate) ──→ every performance claim in T2/T3/T4   [soft but load-bearing]
RFC-7  (factor retains L)            ──→ RFC-2 (batched W-correction)          [hard]
RFC-3  (λmax calibration)            ──→ RFC-17 (strong-rule screening)        [hard]
RFC-6  (deprecation warnings ship)   ──→ RFC-6 (deletion, one release later)   [calendar]
RFC-1  (bin-space actually fires)    ──→ dtype/threading/tiling multipliers     [hard: they multiply
                                                                                 against a dead path]
```

Everything in T1 is mutually independent. RFC-1 does **not** require the operator facade (RFC-10): it is an
additive rung following the existing `_try_factored_tensor_centering` pattern, with latching and the chunked
fallback preserved.

### Tranche 1 — Credibility & correctness (small diffs, immediate)

1. ⚠ **RFC-15 first, not fourth.** Repair the flagship harness (two removed kwargs), regenerate baselines from
   a clean tree, record commit hashes, wire a reduced-rep wall-time gate into CI. Half a day, and it is the
   instrument every later gate reads. It also publishes the fact that master already beats mgcv (~1.36 s vs
   1.57 s), which today is claimed nowhere and contradicted by the tracked artifacts.
2. RFC-3 λmax/auto calibration (+ non-canonical-link family factor) with KKT boundary tests. Minor-version
   bugfix with a prominent changelog entry: `selection_penalty="auto"` currently resolves ~n× too weak, so
   affected users have been fitting near-unpenalised selection. Silently continuing is worse than changing
   results.
3. RFC-8 (bug half only): guard/delete the legacy `=` assembly path so component-named `lambda2` dicts stop
   silently dropping penalties. Full PenalizedSystem deferred to T4.
4. RFC-13 behavioural batch: `max_iter=0` → ValueError; port the polarization merit delta to pirls (extended
   with the group-lasso term); route pure-H QP solves through the rank policy and surface `QPResult.converged`;
   warn instead of silently dropping the W(ρ) term for links lacking `deriv2_inverse`.
5. `gradient.py:353` M_p fallback fix (`hessian_rank=1+p`) + C-12 stale-doc sweep + delete the dead *adapters*
   and orphaned inference algebra (uncontroversial). ⚠ **Hold the deprecation warnings on `optimize_efs_reml`
   and `run_reml_once`** pending item 6.
6. ⚠ **RFC-6a spike — EFS vs exact Newton benchmark.** Fix the M_p rank→nullity bug in `efs.py`, wire it behind
   a private flag, and measure against exact Newton on a_m5/a_m15/a_m25 (q=5/15/25) and b_L800. Rationale:
   EFS omits both the REML Hessian and the W(ρ) correction — 85% of the a_m25 fit — at the cost of linear
   rather than quadratic convergence, and `runner.py` already implements the Anderson-accelerated variant that
   addresses precisely that weakness. Cheap to run, and it gates two things: whether ~950 LOC gets deleted,
   and how much RFC-2 actually needs to deliver.
   **Gate:** suites green; new calibration/KKT tests; flagship benchmark in CI with clean baselines; a decision
   record on EFS-vs-Newton with numbers.

### Tranche 2 — Large-n speed (the flagship path)

6. ⚠ **RFC-1 first, facade second.** Dense-anchored raw-moments rung + lift the ≥2-categorical mixed gate. The
   single biggest measured win (6-10× on the dominant kernel for standard frequency-model shapes); every day it
   is unlanded, users pay it. Doing a dispatch refactor first is regression risk with no user-visible benefit.
7. ⚠ **New: narrow dtypes + deterministic threading** (from §H.2 — not in the first draft). Bin codes
   uint8/uint16 instead of `np.intp`; float32 streaming inputs with float64 accumulators; `prange`/`nogil` on
   the accumulation kernels, which today have **zero** parallelism anywhere in the numerical core. Together
   ~8× bandwidth and ~4-8× cores. **Design constraint:** threading must be bit-deterministic — fixed row-block
   partition independent of thread count, per-block private accumulators, reduction in fixed block order.
   Run-to-run reproducibility is a model-risk requirement for pricing work and the repo already has rtol-1e-10
   fidelity comparators; non-deterministic reduction would break both.
8. RFC-10 policy module + operator facade — now *cleanup informed by* the new rung (six strategies, one
   dispatch home, one threshold home) rather than speculative preparation. Dispatch counters as tests.
9. Chunk-buffer marshalling fix in the retained fallback (preallocated dense chunk, no hstack/CSR slices) —
   ~1.5× for shapes the certificate still rejects.
10. RFC-5 fit() validation gate; RFC-11 terminal/bootstrap dedup; RFC-9 discrete build/support reuse.
    **Gate:** n-scaling re-profile — discrete 1M ≤ ~8 s (from 17.6), exact 1M ≤ ~30 s (from 53.7), fit() 1M
    overhead ≤ 5%; discrete-vs-exact parity suite unchanged; bit-identical results across thread counts;
    flagship CI margin over mgcv widens.

### Tranche 3 — Exact-REML performance

11. RFC-7 factorization-backed HessianFactor (retain L; unify the discrete cached-solve certificate) +
    RFC-4 rank-policy incremental factorization (removes the measured 660× rank-deficient cliff).
12. RFC-2 batched whitened W-correction — **requires 11**. The large-p headline: a_m25-class 172 s → ~25-35 s.
13. RFC-12 cached line-search trials with exact Armijo re-check at acceptance.
    **Gate:** a_m25 ≤ ~35 s; a_m40 completes (currently times out); b_L800 ≤ ~35 s (from 54); exact-REML peak
    memory ~halved at p≈800; FD-gradient and Wood-oracle suites unchanged.

### Tranche 4 — Consolidation & large-p structure

⚠ *(Split out: the first draft's T3 packed nine RFCs into one "tranche" — a quarter of work, not a reviewable
unit.)*

14. RFC-8 full PenalizedSystem: one assembly, one log|S|₊ algorithm, one M_p, block-list S for BCD.
15. RFC-14 structured multi-block backend (nested Schur; exact REML preserved via sparse factor log-det) —
    the highest-value genuine large-p item for this niche.
16. RFC-16 CurvatureProvider + typed CoefficientSolveResult/WorkingSystemCache contract.
17. RFC-6 final resolution after the RFC-6a benchmark and deprecation window: delete the losing optimizer
    (~950 + 290 LOC), or promote EFS to a supported path and delete only the other. RFC-17 screening if
    `fit_path` usage warrants it.

### Not yet placed — pending the research sweep

Two candidates could restructure T3/T4 if the literature supports them, so they are deliberately unplaced:
**GLAM/array algebra** for tensor terms (computes tensor-product `X'WX` from marginal bases without ever
forming the row-Kronecker design — would delete the retained `B_joint` rather than trim it, *if* it extends
from gridded to bin-space-discretized scattered data), and **SAP/SOP** P-spline mixed-model reparameterisation
(diagonalises the penalty structure; if it makes smoothing-parameter cost sublinear in q it outranks RFC-2).
Also under evaluation: **widening tabmat eligibility** — tabmat is already a dependency, its `sandwich_product`
is OpenMP-threaded and hand-optimised for exactly the categorical `X'WX` problem, and no tabmat frame appears
in any profile. Reaching already-parallel code may be cheaper and lower-risk than new `prange` kernels for the
exact/categorical path, though it cannot cover the discretized-spline representation, which still needs
first-party kernels.

### Parallel design track (no code until the note lands)

- **Subsample-λ contract** (§H.3): potentially ~3×, larger than any single kernel item, but it changes what is
  promised about λ̂ reproducibility. Needs a design note first.
- **Pair-accumulation tiling** (§H.2): only pays at G ≳ 10; sequence after T2 with a benchmark at that shape.
- **Streaming/out-of-core contract** (§H.5): blocked on RFC-10; revisit after T2.

---

## G. Attractive ideas that should NOT be implemented

1. **Fused single-pass all-pairs discrete Gram kernel** ("one sweep over an (n,G) code matrix"). Measured
   3-8.5× **slower** than the existing per-pair histogram passes (105 scattered updates/row across a 55 MB
   working set vs one L2-resident 512 KB histogram per pass). Per-pair bin-space accumulation is the correct
   target; mgcv's XWXd is itself per-term-pair, not one all-pairs sweep.
2. **Stochastic trace/log-det (Hutchinson/SLQ)**. Wrong for the niche: exact insurance-scale p (10²-10⁴) is
   served by dense/structured exact factorizations; SLQ's error×iteration budget only pays at p≳10⁵ and
   would put a Monte-Carlo tolerance inside a criterion the project promises matches mgcv.
3. **Iterative solvers + block/low-rank preconditioners as a solve backend**. Same niche argument; the
   structured Schur backend already covers the genuinely-large-p credibility case exactly. Quarantine the
   default-off MINRES prototype hooks out of the production SCOP module instead (C-8).
4. **Full IRLS chassis unification** (merging pirls + irls_direct). The two behavioral payloads (merit_delta,
   max_iter=0) are separable small fixes (RFC-13); the rest is taste at real migration risk. Keep two solvers,
   one typed contract (RFC-16).
5. **λ-trajectory equality as a refactor gate** (e.g. asserting λ agreement at 1e-8 across path changes).
   Summation reordering legitimately perturbs λ trajectories (~1e-12 gradient noise, and the repo's own
   tracked CSV shows λ plateaus differing 184.7 with identical predictions). Gate on fixed-ρ derivative
   equality + converged objective/deviance/EDF/predictions instead.
6. **Dropping the W(ρ) correction from the exact path** (Fisher-only everywhere) to buy speed. It IS the exact
   path's honesty — the discrete path already exists for the Fisher boundary; removing it silently changes
   exact REML gradients for non-canonical geometry. Speed comes from RFC-2, not from deleting the term.
7. **Exposure-weighted support constraints in the discrete rebuild** (tempting while implementing RFC-9): the
   identifiability constraint must stay on unweighted counts or centering semantics silently change.
8. **Out-of-core/streaming backend now.** The DesignMatrix contract genuinely blocks it (C-verified), but no
   current user pain justifies the churn before the operator facade (RFC-10) exists; revisit after T2.
9. **Pattern-compression generalisation beyond one tensor** — U≈n unique patterns on smooth-heavy designs
   defeats it; subsumed by the per-pair accumulation decision (G-1).

---

## H. Scaling ceiling — what "truly big" would cost

Addendum answering: how far can the discrete path actually be pushed, and where does distribution become
necessary rather than decorative?

### H.1 The structural reason the discrete path is scalable

Once covariates are discretized, **n enters the fit through exactly one operation**: the weighted accumulation
of bin-space sufficient statistics. For a term pair (i,j) the block is `B_i' H_ij B_j` where
`H_ij = Σ_r W_r · e_{bin_i[r]} e_{bin_j[r]}'` is a b_i×b_j weighted cross-tabulation. Everything downstream —
penalty algebra, log|S|₊, the p×p solve, the REML Newton step, covariance — touches only the accumulated
statistics and is **completely n-independent**.

Three consequences follow, and they are what make "BigData" tractable rather than aspirational:

1. **The per-iteration cost is memory-bandwidth, not FLOPs.** The accumulation streams index arrays and W; the
   arithmetic is one multiply-add per row per pair. Optimising it is a bandwidth exercise (dtype width, pass
   count, cache tiling), not a numerical-methods exercise.
2. **Accumulation is a sum over rows ⇒ exactly parallel and exactly reducible.** Row partitions compute partial
   `H_ij` and sum them. Nothing is approximated. The reduce payload is `Σ b_i b_j` (or p² if reducing the Gram
   directly) — **independent of n**.
3. **The representation is a 30–100× memory compression.** At n=10⁹, G=20 terms: bin codes (uint8) + response +
   weight + offset + working vectors ≈ **48 GB**, versus 1.44 TB for a dense n×p design. A billion-row GAM is a
   single-fat-node problem, not a cluster problem.

### H.2 Cost model and the ladder of multipliers

Per IRLS iteration, bytes streamed per row (G=20 terms, b≤256 bins):

⚠ *Corrected by the research sweep (§I): the "saturated multi-core" row below was optimistic. Accumulation is
memory-bandwidth bound — one core already saturates ~27 GB/s and threading it measured only **1.3×, flat in
thread count**. The billion-row figure is therefore **~55 s, not 39 s**. Threads belong on the compute-bound
block axis (measured 6-10×), not the observation axis.*

| Configuration | bytes/row/iter | n=10⁹, 14 passes |
|---|---:|---:|
| Today's bin-space path *if it fired* (int64 bins, float64 W, G²/2 per-pair passes, 1 core @ ~12 GB/s) | 4,720 | **92 min** |
| Today's actual path (chunked-dense fallback, RFC-1 unfixed) | — | hours (extrapolates ≥3× worse) |
| uint8/uint16 bins + float32 W (float64 accumulators) | 1,140 | 22 min |
| \+ cache-tiled pair accumulation (4-term tiles ⇒ 15 sweeps, histograms L2-resident) | 140 | 2.7 min |
| \+ saturated multi-core bandwidth (~50 GB/s) | 140 | **39 s** |

Compounding factor ≈ **34× from representation and tiling, plus ~4-8× from threading ≈ 100-200× overall**,
before any statistical shortcut. Three currently-unclaimed levers make up that ladder:

- **Zero parallelism exists in the numerical core** (verified: no `prange`, `parallel=True`, `nogil`,
  thread-pool or `threadpoolctl` anywhere in `src/superglm/` outside the editor's web server). All ~40 numba
  kernels are single-threaded; the only multi-core work is inside BLAS `dgemm`. On bandwidth-bound accumulation
  this is a free 4–8×.
- **Bin indices are `np.intp` (int64) and everything is float64** (`_group_matrix_discretized.py:40`,
  `_group_matrix_bins.py:17`; zero `float32` in the numerical core). Bins with b≤256 need 1 byte; the streamed
  weight vector can be float32 with float64 accumulators. 8× on the hottest arrays.
- **Pair accumulation is untiled.** Per-pair passes are correct at small G (and measurably better than a fused
  all-pairs sweep — see G-1) but scale as G² full data passes. Tiling over term-blocks makes the histogram
  working set L2-resident while cutting streaming volume ~6× at G=20. This is the standard blocked-GEMM
  transformation applied to histogram accumulation, and it reconciles the two failed extremes.

### H.3 The statistical lever (likely the largest single win)

Of the ~14 full-data passes a discrete fit performs, most serve the **smoothing-parameter search**, not the
final coefficient fit. λ is a low-dimensional nuisance parameter whose selection error at n=10⁷ is already
negligible for prediction. Selecting λ on a subsample and running only the terminal PIRLS on all rows
(mgcv gestures at this with `bam(samfrac=)`) collapses 14 full passes to ~3-5, a further **~3×** that composes
with everything in H.2. Combined ceiling: **a billion-row, 20-term GAM in ~15-20 s on one machine.**
This deserves its own design note — the contract question is what to promise about λ̂ reproducibility.

### H.4 Large p is a different problem (discretisation does not help it)

Bin-space accumulation reduces the *n* factor only. For large p the ladder is:

- **p ≤ ~5·10³** — today's dense algebra is correct; fix the constants (RFC-2/4/7).
- **p ~10³-10⁵ dominated by high-cardinality factors** (the realistic insurance case: many rating levels,
  credibility/random effects). A single categorical's own `XᵀWX` block is **diagonal** — each row touches one
  level — so eliminating it by Schur complement is O(L), not O(L³). Nested Schur elimination plus a
  fill-reducing sparse Cholesky (CHOLMOD/AMD, as `lme4` and `mgcv::gamm` use) keeps `log|XᵀWX+S|` exact.

  **The piece that completes it: Takahashi's equations (selected inverse).** A sparse factor alone is not
  enough for REML, which also needs `tr(H⁻¹S_j)` per penalty, `diag(H⁻¹)` for EDF and standard errors, and
  `tr(H⁻¹S_iH⁻¹S_j)` for the Hessian. Takahashi, Fagan & Chin (1973) give a backward recursion that produces
  **exactly** the entries of `H⁻¹` on the sparsity pattern of `L+Lᵀ` — including the whole diagonal — at
  roughly factorisation cost, without ever forming `H⁻¹`. Because each `S_j` is supported inside one term's
  block, that pattern covers every trace REML needs. This is the standard route in sparse-GMRF and
  animal-breeding REML (Rue & Held 2005; Misztal & Pérez-Enciso 1993) and is what INLA uses.

  Consequence: **exact REML at large p is achievable**, resolving the tension flagged below — provided the
  structure is sparse-factorisable. It also makes §G's rejection of stochastic trace/log-det *stronger*, not
  weaker: Hutchinson/SLQ buys an approximation to something Takahashi delivers exactly at comparable cost.

  SuperGLM already ships a hand-rolled **one-block special case** of this: the structured credibility backend
  (`docs/superpowers/specs/2026-07-24-credibility-structured-effects-design.md`, shipped in #165) specifies a
  factor protocol with `selected_inverse_block`, `selected_inverse_diagonal`, `trace_inverse_penalty` and
  `logdet`, eliminating one term via `Hinv_aa = Q⁻¹`, `Hinv_bb = D⁻¹ + FQ⁻¹Fᵀ`, `Hinv_ba = −FQ⁻¹`, and it
  explicitly notes that with several structured terms only the largest is eliminated ("this is exact; it merely
  limits the speedup"). Generalising that hand-rolled elimination into sparse-Cholesky + Takahashi is the
  natural upgrade: it handles arbitrarily many blocks automatically and subsumes RFC-7 and RFC-14 into one
  architecture. The existing factor protocol is already the right seam — it was designed for this.

  For the residual hard case (two or more high-cardinality factors with a dense cross-block), the econometrics
  high-dimensional-fixed-effects literature (Guimarães & Portugal 2010; Gaure 2013 `lfe`; Correia `reghdfe`)
  solves the coefficient problem by alternating projections in O(n) per iteration — though note it does *not*
  give the REML traces, so it complements rather than replaces the selected-inverse route.
- **p ≳ 10⁵ with dense coupling** (e.g. many continuous×continuous tensor interactions) — no structural escape;
  iterative solves plus stochastic log-det/trace become necessary, **and exact REML is lost**. Note this is
  precisely the machinery rejected in §G for the current niche: the rejection is niche-specific, not permanent.
  The honest statement is that **exact REML and dense-coupled p ≳ 10⁵ are in tension**, and mgcv does not go
  there either.

### H.5 If distribution is genuinely required

Only past single-node RAM (~10¹⁰ rows at this footprint). Two options beat a bespoke Kubernetes cluster:

- **Push the accumulation to the data.** The bin-space statistic is literally a `GROUP BY`:
  `SELECT bin_i, bin_j, SUM(w) ... GROUP BY 1,2`. Warehouse-resident data (BigQuery/Snowflake/Spark/DuckDB) can
  produce the sufficient statistics without moving rows; IRLS then needs only a small coefficient table pushed
  down each iteration to recompute η. Per-iteration network payload is O(Σ b_i b_j), n-independent.
- **Out-of-core on one node.** Chunked parquet streaming with the same accumulation kernels. This is blocked
  today by the eager-materialisation `DesignMatrix`/`EagerFrame` contract (§C, `large-n:streaming-contract`),
  which is why RFC-10's operator facade is the prerequisite for any of it.

If a true cluster is ever needed, the model is plain data-parallel IRLS: each worker owns a row slab, computes
partial `XᵀWX`/`XᵀWz`, all-reduce (O(p²) doubles), driver solves and broadcasts β (O(p)). Exact, textbook, and
the interesting engineering is entirely in the loader, not the statistics.

### H.6 Ordering, and honest limits

Sequence: RFC-1 (nothing else matters until bin-space actually fires) → narrow dtypes + threading → pair tiling
→ subsample-λ design note → streaming contract → structured multi-block for large p.

Limits worth stating plainly: pass count cannot go below ~3-5 without approximation; bin resolution trades
accuracy for the n-independent b² term; discretisation contributes **nothing** to large p; and dense-coupled
very large p costs exact REML. The single-machine ceiling (~10¹⁰ rows) exceeds any plausible motor/household
book by orders of magnitude, so a cluster is a warehouse-integration decision, not a performance one.

---

## I. Research-sweep addendum — deltas to sections D–G

Five literature/benchmark tracks were run after the audit. Full detail:
`~/.claude/jobs/e3eef6ba/tmp/audit/findings/research-CONSOLIDATED.md` and the five `research-*.md` reports.
Items below **supersede** the corresponding RFC entries above.

**RFC-2 is superseded.** The W(ρ) gradient correction reduces exactly to a leverage diagonal:
`tr(H⁻¹X'diag(a_j)X) = Σₖ a_j[k]·hₖ` with `h = diag(XH⁻¹X')` computed once. That is
**O(q·n·p²) → O(np² + qn) — flat in q**, not merely a better BLAS constant. Verified 1.8e-15; measured 6.2× at
q=25 and 9.8× at q=40 with cost flat across q. Requires also removing the W-correction from the Hessian, which
is **parity-safe by construction** because `direct.py:334-346` tests convergence on the gradient. Combined:
**172 s → ~32-40 s**. Chunk `M = XL^{-T}` over rows (1.2 GB at MTPL2 n=678k).

**RFC-6 resolution: AI-REML, not EFS.** AI-REML approximates only the *Hessian* — the gradient stays exact, so
the fixed point is the **exact REML optimum**. EFS/SOP make the PQL simplification and change the estimator.
`gⱼ = Sⱼβ̂`, `hⱼ = H⁻¹gⱼ` (**q solves total**), `AIᵢⱼ = gᵢ'hⱼ`. AIREMLF90 converges in 5-15 rounds vs 50-300.
Two prerequisites, both in already-shipped code: superglm's EFS implements the provably shorter-stepping
"accelerated EM" update (`λ*=r/(b+g)` instead of Wood–Fasiolo's `φ(r−λg)/b` — measured 49→25 iterations), and
`efs.py:251`/`runner.py:233` hardcode `tr(S_λ⁻Sⱼ)=rⱼ/λⱼ`, **valid only for non-overlapping penalties and
therefore silently wrong under `select=True` and tensor `ti()` terms.**

**RFC-7's root cause is sharper than stated.** `_safe_decompose_H` (`solvers/irls_direct.py:235-350`) does
`cho_solve((L,True), np.eye(p))` — an explicit dense p×p inverse — on essentially every PIRLS/Newton/line-search
iteration, and `inference/covariance.py` builds a second one by another route. The design comment at
`irls_direct.py:8` still reads *"p is ~50-80… making the p×p solve trivially fast."*

**RFC-14 gains a verified architecture and a gate.** sparse Cholesky → Takahashi selected inversion (first-order
traces + `diag(H⁻¹)`) → AI (curvature) is the ASReml/BLUPF90/WOMBAT standard; Takahashi and AI are
**complements** (the first-order trace does not cancel under AI). Evaluate **Smith (1995) reverse-mode Cholesky
AD** as a cheaper gradient route — **LMMsolver (Boer 2023) reports mgcv 600 s → 1 s and 38 min → 30 s**.
scikit-sparse has no selected inversion (PR unmerged) — **port Davis's `sparseinv`, ~200 lines, BSD-3**, which
also resolves the GPL concern. **Gate on a measured `nnz(L_H)`**: MSSM found sparse Cholesky *slower* than
dense when H wasn't genuinely sparse.

**New RFC — row-tensor Gram blocks in bin space.** `(A′WA) = G(Ã₁)′W̄G(Ã₂)` with W̄ the 2-D histogram already
built and `G(Ãⱼ)` weight-independent marginals precomputed once. **77-352× on tensor blocks** (verified 5e-15),
Gram stage 11.7× / 39× threaded; off-diagonal tensor×main blocks free from the same W̄. Deletes the
`(n,p₁p₂)` row-Kronecker loop (`interaction.py:774-791`) and `B_joint`. Wins iff `m₁m₂ ≲ n`; cap tensor-marginal
bins ~500-1000. **Grid-vs-scattered settled: GLAM proper needs a lattice and joint gridding is explicitly
rejected by Li & Wood — but the row-tensor identity is grid-free.** Size against real tensor usage; the
profiles contained no tensor terms.

**Threading is deterministic only if designed so.** Measured: numba's automatic `prange` reduction is **not**
bit-reproducible across thread counts (~30 machine epsilon); fixed-chunk privatization with a compile-time
constant chunk count **is**. Prefer partition-free parallelism (block/group axis: 6-10×, bit-identical by
construction). Forbid `fastmath=True` in the numerical core. **A thread-count-invariance CI test must land
before any threading work.**

**Free wins.** Stop converging λ to 1e-7 — risk is flat to a 12× change in λ (MISE moves 20-35%); stop on
EDF < ~0.01/term instead (2-5 passes). Li & Wood's loop order is column-major-tuned and they say to reverse it
for row-major (**NumPy is C-order**); and **BLAS quality alone was 10×** in their timings — audit which BLAS
this path hits.

**Subsample-λ, corrected.** λ̂ is **not** scale-free in n (measured `d log λ̂/d log n = 0.43`), and mgcv's
`samfrac` carries only coefficients, never `sp`, and is skipped under `discrete=TRUE`. The defensible version is
**warm-starting λ from a rescaled subsample fit and converging on full data** — start-independent fixed point,
zero contract change, ~2-2.5×. Frozen-λ, if ever shipped, must be refused under `select=True`.

**Also rejected on evidence** (adding to §G): GAP safe screening (measured improvement factor 0.98-1.00);
coresets for Poisson-log (Ω(n) lower bound — a theorem); sketching/RandNLA (forming the sketch costs a full data
pass); leverage-score sampling (superseded); HDFE alternating projections as a source of REML traces (Kline et
al. needed 8 h on 32 cores for what selected inversion gives directly); AD for the W(ρ) correction (the analytic
form already *is* the implicit-function-theorem solution); Rügamer factorization-machine tensors (rank-F
approximation breaks parity by construction).

---

## J. perf/cheap-interactions — landed work, re-validation, superseding decisions (2026-07-28)

Written after the branch review + continuation session on the **16-core workstation** (the new benchmark
reference box; the 5950X numbers above no longer bind). All numbers below are same-machine, same-data
(freMTPL2 n=100k, `benchmarks/benchmark_tensor_cost.py`), measured at the stated commits. Items below
**supersede** the corresponding entries in §E/§F/§I.

### J.1 Landed on the branch (verified: math review + full suite + re-measurement)

| work item | verdict | measured effect (this box) |
|---|---|---|
| **Support-compressed SSP groups** (lossless row dedup; calibrated gate ×6.0) | exact; algebra = weight aggregation over bit-identical rows; external formalization arXiv:2511.12732 Thm 3.2 | the dominant win in tensor exact 51.8→11.9 s pre-RFC-12a; gate re-measured here: 9/9 crossover signs hold, implied BLAS advantage 6.8–30 vs conservative 6.0 |
| **RFC-1 raw-moment centering rung** | sound; certificate bounds cancellation ratio ≤2; rejection latches per fit, success re-certifies per W. ⚠ **anchor shift NOT implemented** — ill-located Dense numerics (VehPower-shape) still latch to chunked | part of the 51.8→11.9 s composite |
| **RFC-3 λ_max calibration** | correct; score `w·(dμ/dη)(y−μ)/V` matches solver KKT threshold exactly; α division for elastic net; behaviorally pinned 24/24 @1.01×, 23/24 @0.99× | `selection_penalty="auto"`/`fit_path` now meaningful (bugfix release + changelog) |
| **Line-search trial carry-forward** | sound; not bit-identical (different warm start, same fixed point), honestly documented; call-site parameter parity verified | 7 of 8 candidate fits eliminated on the flagship |
| **Discrete tensor cross-constraint removal** | correct root-cause fix: null(C) ⊇ the block's own null space, so the projection either no-ops or retains garbage; ti() marginal centering keeps shared-marginal tensors jointly identifiable; matches exact path & mgcv | discrete shared-marginal models fit instead of raising; deviance parity 2.6e-5 |
| **RFC-6a spike: AI-REML** | **rejection CONFIRMED** — REML Hessian is 0.04–0.49% of wall; 6 outer Newton iters ≤ AIREMLF90's own 5–15; the research headline compared AI vs EM-REML, and AI-REML keeps the exact gradient so it never touched the W-correction cost either | RFC-6 deletion path unblocked (deprecation cycle for `run_reml_once`/`optimize_efs_reml` re-exports, then delete ~950+290 LOC) |
| **NEW · RFC-12a: in-loop fits skip rank metadata** (commit 096c171) | in-loop candidates/trials never consumed fit statistics; the three O(p) quantities the gradient/objective read (mean_x, sum_w, data-gram column scales) now travel on a `REMLGeometrySummary`; terminal refit unchanged → published stats unchanged; trace/debug runs keep full stats | tensor exact **11.90→5.99 s**, base exact 2.92→2.25 s; streamed-QR certifications 11→2 per fit; trajectory identical (8 iters, 7 reuses) |
| **NEW · hashed row grouping** (commit bb9ed48) | detection was the last big cost (byte-keyed lexicographic sort, 1.34 s of a 5.7 s fit); now a verified 64-bit mix: 8-byte sort + bitwise verification, collision → byte-keyed fallback; deterministic across machines; NaN/−0.0 semantics preserved | base exact **2.25→1.12 s**, tensor exact **5.99→4.70 s** |

**Cumulative, master f082e9b → branch HEAD, same box:** base exact 8.59→**1.26 s (~6.8×)**, tensor exact
51.77→**4.46 s (~11.6×)** (branch numbers are medians, see below; the master baselines are single runs).
The suite is green throughout (4723 passed incl. mgcv parity, Wood oracles, FD gradients, freMTPL2 real-data
parity).

⚠ **Correction (same day, median re-measurement):** an earlier draft of this section (and commit bb9ed48's
message) claimed the plain-spline exact fit had overtaken discrete at n=100k (0.89×). That was single-run
noise — both paths vary ±15% run-to-run on this box. Median-of-reps standings at branch HEAD, plus a
first full-data measurement:

| n | model | exact | discrete | exact/discrete |
|---:|---|---:|---:|---:|
| 100k | base (median of 5) | 1.26 s | 1.08 s | 1.17× |
| 100k | +tensor (median of 3) | 4.46 s | 3.19 s | 1.40× |
| 678k (full) | base (single) | 2.75 s | 1.19 s | 2.3× |
| 678k (full) | +tensor (single) | 14.41 s | 6.96 s | 2.07× |

The honest headline: the exact/discrete gap collapsed from 6.0× (base) and 17.6× (tensor) on master to
**1.2×/1.4× at 100k and ~2.1–2.3× at full data** — exactness at mid-scale is now nearly free, and
`discrete=True` keeps its clear value at portfolio scale. Note the discrete path received no optimization on
this branch and still carries known waste (RFC-9 rebuilds, RFC-11 bootstrap/terminal duplication), so its lead
at scale is understated. This also re-ranks nothing in §J.4: the remaining exact-path costs at 678k are the
line-search trial PIRLS passes (RFC-12b) and per-iteration O(n) work the discrete cached-W path avoids by
design.

### J.2 Superseding corrections to the backlog

- **RFC-2 (batched/leverage W-correction) is DEMOTED, not superseded-by-implementation:** compression already
  made each signed Gram cheap — W-correction is 0.36 s of the 4.7 s tensor fit here. It matters only for designs
  whose covariates don't compress (truly continuous). Do not schedule ahead of RFC-12b.
- **RFC-12 splits:** 12a landed (above). **12b — cached-factorization trial evaluation with exact Armijo
  re-check at acceptance — is the next big exact-path item** (line search still 1.5 s of the 4.7 s tensor fit;
  trials still run full PIRLS). Template: Wood's NCV machinery (arXiv:2404.16490) — rank-1 Givens/hyperbolic
  up/downdates of the retained Cholesky, Woodbury fallback; mgcv-grade production precedent. Prerequisite is a
  retained factor on the result (RFC-7's factor protocol seam).
- **RFC-18 (detection cost) resolved** by the hashed grouping — the `plan_row_support` covariate-plumbing
  variant is no longer worth its GroupInfo surface change; keep `plan_row_support` as the seam for callers that
  already own a grouping.
- **RFC-14/§H.4 caution (research):** for **crossed** high-cardinality factors the sparse Cholesky factor is
  provably dense (arXiv:2411.04729, random-multipartite fill; arXiv:2505.11674 measures the same on MovieLens —
  the *second* crossed factor is what goes dense). Takahashi selected inversion pays for nested /
  one-dominant-factor / banded-spatiotemporal structure only. Exact REML traces for 2+ crossed factors remain
  open (published answers are stochastic — rejected here); CG covers the *solves* (dimension-free iterations).
- **§G additions (rejected, new evidence):** safe screening for smooth-group lasso re-confirmed dead (only
  Group-OWL/SLOPE progress exists); TabPFN-distilled interaction selection (TabDistill 2604.13332) is capped at
  ~50k rows — irrelevant at this niche's n. Working-set + KKT-recheck remains the philosophically matching
  pattern (Hessian Screening Rule, arXiv:2104.13026).
- **Row-tensor G-operator:** audit §I claimed the identity verified 5e-15; the branch's implementation attempt
  failed sign-gauge recovery. The identity is textbook (it's XWXd's tensor trick) — the failure is
  SSP-reparametrisation-specific. Moot while compression handles tensor Grams; revisit only if a
  large-support tensor (gate-declined) shows up in profiles.

### J.3 Interaction discovery — promoted to first-class objective

Fitting interactions is now cheap; *finding* them is the unclaimed differentiator and the point of the branch
name. Plan written: `docs/superpowers/plans/2026-07-28-interaction-screening.md` (commit 517fe41). One-line
summary: rank every candidate `ti(a,b)` by a **penalized efficient-score statistic** assembled from per-pair
2-D weighted cell moments — the same sufficient statistics the fit already computes (score vector = RFC-3's
family-factor score; histograms = the disc×disc cross-gram path; codes = compression's exact `bin_idx`). One
O(n) bincount pass per pair, no refits; confirmatory `fit_reml` refit of the top-k is the gate. Published
foundations: BOLT-SSI (1902.03525, sure screening + quantified binning loss ≤1.21) and sprinter-GLM
(2401.08159, frozen-offset score ranking ≡ conditional-covariance ranking); **neither covers penalized smooth
groups — publishable gap.**

### J.4 Updated ranked backlog (fit_reml GAM path first)

1. **Interaction screening plan, Tasks 1–5** (J.3) — the value-prop item; all infrastructure exists.
2. **RFC-12b** cached-factor line-search trials (NCV up/downdates; exact re-check at acceptance) — the last
   structural exact-path cost at ~1.5 s/4.7 s.
3. **RFC-15 CI perf gate** — five landed wins now protected by nothing; wire `benchmark_tensor_cost.py`
   (reduced n) + the 30-rep flagship into CI with thresholds against this box's baselines.
4. **RFC-1 anchor shift** for Dense numerics — completes centering for splines+cats+numerics books.
5. **RFC-6** deprecation warnings now, dead-optimizer deletion next release (~950+290 LOC), plus the
   uncontroversial adapter/orphaned-algebra deletions and `gradient.py:353` M_p fallback fix.
6. **RFC-8 bug-half** (lambda2 `=`-assembly silent penalty drop) and **RFC-13** behavioural batch — small,
   correctness-first.
7. Candidates, unmeasured: Demmler–Reinsch λ-bracket init (arXiv:2205.15157) to replace fixed log-λ clips and
   possibly shave 1–2 outer iterations; subsample-λ warm start (measured d log λ̂/d log n = 0.43 — also a
   publishable gap, nothing in the literature); deterministic reduction-tree design for future threading
   (arXiv:2607.18758).

### J.5 Stress battery (benchmarks/benchmark_support_stress.py, n=100k, this box)

Run after a challenge that freMTPL2's default columns are too low-cardinality to trust. Cardinalities used:
DrivAge 82 · VehAge 55 · BonusMalus 98 · VehPower 12 · **Density 1,568** · joint DrivAge:BonusMalus 2,635 ·
**joint DrivAge:Density 35,005** · joint VehAge:BonusMalus 1,363.

| case | result |
|---|---|
| **A · exactness** — identical tensor fit, detection on vs off | max\|Δβ\| 2.5e-10 (rel 1.2e-10) · deviance agrees to 12 significant digits (Δrel 3e-14) · EDF equal to 8 dp · max rel \|Δμ̂\| 5.3e-9 — the same magnitude as a BLAS/thread-count change. Wall: 4.38 s compressed vs 83.1 s uncompressed. |
| **B · high-cardinality integer** — s(Density) + ti(DrivAge, Density) | Everything compresses, including the 1,568-support spline and the **35,005-support tensor** (22.7 MB unique-row buffer, under the 64 MB cap; the dense-row flop model accepts ratio 0.35). Exact 7.80 s vs discrete 3.71 s (2.1×). Gate decisions all correct per `design_summary()`. |
| **C · continuous covariate (jittered Density)** — the honest worst case | Gate correctly declines Density and the tensor (support = n); the four integer splines still compress. Exact fit **53.5 s** vs discrete ~3.7 s — **the uncompressed-tensor regime remains the big exact-path gap** (it is master's old cost structure, minus RFC-12a). Detection is not the problem: with detection fully disabled the fit is 71.6 s, i.e. compressing the remaining groups buys more than detection costs even when the big blocks decline. |
| **D · two tensors sharing a marginal** | Both compress (2,635 and 1,363 supports); cross-gram cells 3.6 M stay under the 5 M histogram cap; fits in 16.1 s. Multi-tensor fits scale super-linearly in wall (4.4 → 16.1 s for +1 tensor) — geometry-side certification of two holey penalized blocks is the suspected driver; probe alongside RFC-12b. |

**Named limitation (now documented, was implicit):** the branch's exact-path wins are conditional on covariates
taking repeated values (integer/low-cardinality rating factors — the insurance norm). A truly continuous
covariate inside a `ti()` keeps the old SparseSSP cost structure. Mitigations, in preference order: RFC-2's
leverage-diagonal W-correction is **re-promoted for exactly this regime** (it was demoted only for compressible
designs); the chunked-fallback marshalling fix; the row-tensor G-operator revisit; and `discrete=True`, which
exists precisely for this trade and is 14× faster here with its usual disclosed binning.

---

## Appendix: evidence trail

- Subsystem maps (9): `~/.claude/jobs/e3eef6ba/tmp/audit/reports/*.md`
- Auditor reports (7): `~/.claude/jobs/e3eef6ba/tmp/audit/findings/*.md`; structured findings:
  `findings/all_findings.json` (73 findings); verification verdicts: `findings/verdicts.json` (19 verifiers,
  16 clusters; 3 critical clusters dual-lens).
- Profiles: `~/.claude/jobs/e3eef6ba/tmp/audit/profiles/` (`.prof`/`.pstats.txt` per config + two analysis
  reports); probe scripts: `findings/probes*/`.
- Worktree under audit: `.worktrees/audit-master` @ f082e9b (read-only; venv built for measurement).

