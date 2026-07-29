# RFC-12b: Cached-Factor Line-Search Trials — Design Note + Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the full PIRLS solve per REML line-search trial with a
cached-factorization surrogate evaluation (retained Cholesky, rank-r
up/downdates, Woodbury fallback), keeping an exact Armijo re-check at
acceptance.

**Architecture:** The exact-Newton REML loop (`reml/direct.py`) keeps the
current accepted iterate's centered-system Cholesky factor (already computed
inside `fit_irls_direct` as a `RankDecomposition`, today discarded after its
pseudo-inverse is extracted). Each line-search trial then evaluates a
POI-style surrogate — frozen working weights, one factor update for
`S(λ_trial) − S(λ_current)`, one solve, one exact deviance pass — instead of
a full PIRLS fit. Acceptance always re-runs the exact fit and re-checks
Armijo on the exact objective.

**Tech Stack:** numpy/scipy dense linear algebra; the existing
`RankDecomposition` (`solvers/rank.py`), `PenaltyComponent` eigenstructure
(`reml/penalty_algebra.py`), `PIRLSResult` (`solvers/pirls.py`).

**Template:** Wood's NCV machinery (arXiv:2404.16490): rank-1
Givens/hyperbolic up/downdates of a retained Cholesky with Woodbury
fallback — mgcv-grade production precedent. Audit references: 2026-07-28
architecture audit §J.2 (RFC-12 split), §E items 7/12, §J.4 item 2.

## Global Constraints

- **Converged-trial carry-forward invariant** (the `if trial_result.converged`
  gate under `if trial_accepted` in `optimize_direct_reml`'s step-halving
  loop, regression-tested by `tests/test_reml_trial_carry_forward.py`):
  only a CONVERGED exact fit may become `_carry_forward`. Surrogate trial
  states must NEVER enter carry-forward — they are not PIRLS-stationary.
- **Exact values only in the retained optimizer state.** Carry-forward is
  not the only trial-derived state that escapes the line search:
  `best_obj`/`best_pirls`/`best_lambdas` (which become
  `REMLResult.objective` and the returned fit) and
  `lambda_history`/`objective_history` (which feed the convergence
  criterion) are updated inside `if trial_accepted` and are NOT gated on
  `converged`. They stay correct today only because acceptance always
  evaluates the exact objective. Under the surrogate design this becomes
  load-bearing: those variables must record the exact-objective value of
  the exact acceptance fit; a surrogate value must never reach them.
- **Exact Armijo re-check at acceptance is mandatory** (audit §E item 12:
  "verified safe; pure-surrogate trials are not"). On exact-recheck
  failure, fall back to exact trials for the remaining step-halvings
  before declaring line_search failure.
- **Surrogate exhaustion must also fall back to exact trials.** The
  re-check above covers only optimistic surrogates. If the surrogate is
  PESSIMISTIC and rejects all step-halvings, the loop would otherwise hit
  the `had_feasible_trial` branch and set
  `termination_reason = "line_search_failed"` — terminating the WHOLE
  optimizer on surrogate evidence with no exact trial ever run (that
  branch's comment, "the fully converged exact objective rejected every
  feasible trial", would be false). On surrogate exhaustion, re-run the
  step-halving loop with exact trials before `line_search_failed` may be
  declared.
- **`log_det_H` measure contract** (the contract comment on
  `PIRLSResult.log_det_H` in `solvers/pirls.py`): at full rank
  log|H_aug|; under rank truncation the identified-coordinate measure
  `log(sum(W)) + log|H_c|_+`. Surrogate logdets must reproduce the same
  measure or the LAML objective silently shifts.
- **NaN-poisoning stat-skip contract** (RFC-12a): in-loop fits skip
  statistics; published stats come from the terminal refit only.
- Benchmark canon: single-thread BLAS both sides, medians not single runs.
- Protected semantics (CLAUDE.md): fit vs fit_reml distinct; sample_weight
  = exposure; discrete=True never drifts from exact.

## Where the cost is (audit-measured baselines)

- Line search runs full PIRLS per trial: ~1.5 s of the 4.7 s exact tensor
  fit at n=100k (§J.2); trials are 17% of b_L800 optimizer wall; 8 trials
  measured per accepted step in the profiled runs (§E item 12).
- Carry-forward already eliminates 7 of 8 *candidate* fits (§F); trials
  are the remaining structural exact-path cost.

## Design

### Scope

The surrogate applies to the DENSE exact-Newton path only
(`optimize_direct_reml`, `use_structured=False`,
`use_observed_geometry=False`). Exclusions, all falling back to today's
exact trials:

- `use_observed_geometry=True`: observed-mode trials are gated on
  per-trial convergence and mode-score residuals — a frozen-W surrogate
  cannot produce those gates.
- `use_structured=True`: the factor is a structured Schur complement, not
  a dense Cholesky; updating it is a different (later) design.
- `discrete=True`: delegated before this loop (`direct.py:105-133`).
- **Non-converged base fit**: the surrogate is anchored by
  `surrogate(λ_current) = exact(λ_current)`, which holds only when the
  frozen-W solve reproduces `β_current` — i.e. when the base fit is
  PIRLS-stationary. The candidate fit at the top of the outer loop is NOT
  required to be converged (only carry-forward gates on `converged`), so
  surrogate eligibility must additionally require
  `base_result.converged` — from a non-stationary base the Armijo
  comparison mixes two different measures, which is also the most likely
  trigger of the pessimistic-exhaustion case above.

### Frozen state at the accepted iterate

After the candidate fit at the top of the outer iteration (or the carried-
forward accepted trial), retain:

- `R` — the centered slope-system Cholesky from `decompose_gram(
  centered_final.hessian)` (a `RankDecomposition`; Task 1 stops discarding
  it),
- `XtWz_c`, `mean_x`, `sum_w`, `column_scale` — the centered moments
  (already summarized in `REMLGeometrySummary`; the RHS needs retaining),
- the deviance function's inputs (X, y, weights, offset) — already in
  scope in the loop.

`H_c(λ) = XtWX_c + S_c(λ)` in centered slope coordinates, where
`S_c(λ) = Σ_j λ_j Ω_j` and every `Ω_j` already carries eigenstructure on
its `PenaltyComponent` (`eigvals_omega`, `omega_ssp`).

### Surrogate trial evaluation

For trial lambdas `λ_t = exp(ρ + s·Δ)`:

1. `ΔS = Σ_j (λ_t,j − λ_c,j) Ω_j`. Factor each component as
   `Ω_j = U_j U_jᵀ` (rank r_j, precomputed once per fit). CAVEAT: not
   every `PenaltyComponent` carries a full coefficient-space
   eigendecomposition — identity components have `omega_ssp` and
   `eigvals_omega` as `None`, and repeated / sum-to-zero factor-smooth
   components store only a local `block_width × block_width` matrix while
   `group_sl` spans all levels; all of these can occur on the dense path
   (`direct_solve="gram"`). Build `U_j` per `penalty_kind` via the
   existing expansion helpers (`penalty_component_dense_matrix` /
   the sum-to-zero and repeated expanders), computed once per fit. An
   identity component over a wide block has r_j = block width — for such
   components the update is a diagonal shift whose rank defeats
   incremental updating, which the Givens-vs-Woodbury crossover (below)
   must treat as a refactorize-or-exact-trial case, not a rank-1 loop.
   Additionally, the retained factor decomposes the EQUILIBRATED system
   `D⁻¹ H_c D⁻¹` (D = the decomposition's own `column_scale`, restricted
   to `active_columns`) — every update vector must therefore be applied
   as `D⁻¹ U_j` in that basis, not as raw solver-space columns.
2. Update a COPY of the retained factor (the base `cholesky_factor` is
   frozen read-only by the rank policy, and every step-halving trial must
   update from the SAME base `R` — an in-place update would raise on the
   read-only buffer and, worked around, would cumulatively corrupt the
   base across trials): for `Δλ_j > 0`, r_j Givens rank-1 updates
   (`cholupdate`); for `Δλ_j < 0`, hyperbolic rotations (downdate).
   Downdates can lose positive definiteness numerically — on failure (or
   when `Σ r_j` exceeds a crossover, ~p/4 by flop count, to be measured),
   use the Woodbury fallback instead:
   `(H + U C Uᵀ)⁻¹ rhs` via the retained `R` and the small
   `(Σr_j × Σr_j)` capacitance system; `log|H + U C Uᵀ| = log|H| +
   log|C⁻¹ + Uᵀ H⁻¹ U| + log|C|` (matrix determinant lemma).
3. Surrogate coefficients: `β̃ = H_c(λ_t)⁻¹ XtWz_c` (one triangular solve
   pair), intercept via the centered-system profile.
4. Surrogate objective: exact deviance pass at `β̃` (O(n), the POI trick),
   exact `log|S(λ_t)|₊` (already closed-form/cached per component),
   `log|H(λ_t)|` from the updated factor diagonal plus `log(sum_w)`,
   penalty quadratic at `β̃`. Same identified-coordinate measure as the
   contract above.
5. Armijo test on the surrogate objective. Rejected trial cost:
   O(p²·Σr_j + n) instead of a full PIRLS fit.

### Acceptance

On surrogate Armijo acceptance, run today's full exact trial fit
(`fit_irls_direct` warm-started at `β̃`) and re-evaluate the exact
objective. Accept only if the EXACT value passes Armijo. If it fails,
discard the surrogate's remaining trust and continue the step-halving loop
with exact trials (today's code path unchanged). Symmetrically, if the
surrogate REJECTS every step-halving, re-run the loop with exact trials
before `line_search_failed` may be declared (see Global Constraints — a
pessimistic surrogate must not terminate the optimizer). The accepted
exact fit — and only it, and only when `converged` — feeds
`_carry_forward`, preserving the invariant verbatim, and only its exact
objective value may reach `best_obj`/`best_pirls`/`objective_history`.

Expected fit counts per outer iteration: 1 candidate (usually carried
forward) + 1 exact acceptance fit; all rejected trials become surrogates.

### Rank-deficiency and certification

`decompose_gram` may return `pivoted_cholesky` / `gram_eigh` / `qr_svd`
methods. The up/downdate path requires a clean `cholesky` factor in the
retained-column basis — and `method == "cholesky"` alone is NOT a
sufficient gate: the cholesky return path can itself carry a truncated
active set (`rank_truncated = len(active_columns) < width` inside the
cholesky branch of `decompose_gram`). Freezing a truncated active set
across trials freezes the identified subspace, while the exact objective
at `λ_t` would use `λ_t`'s own subspace — added penalty mass can
re-identify a dropped column and a downdate can lose one, so the
surrogate and exact `log_det_H` would be measured in different coordinate
systems (the silent LAML shift the measure contract warns about; the
Woodbury fallback returns a plausible number rather than an error, so it
cannot detect this).

Eligibility gate: `method == "cholesky" and not rank_truncated` (full
active width) and `needs_factor_certification` did not fire. After every
factor update, check conditioning against the same
`policy.gram_rcond * ‖·‖₁` band `decompose_gram` applies, and bail to an
exact trial on failure. Rank-deficient fits are exactly where frozen-W
extrapolation is least trustworthy.

Note on equilibration: freezing the decomposition's `column_scale` is
harmless — `log_pdet` already carries the `2·Σ log(scale[active])` term
and `|D A D| = |D|²|A|` is exact bookkeeping under a fixed `D`. It is the
ACTIVE SET, not the scale, that cannot be frozen. Beware also that
`REMLGeometrySummary.column_scale` is `sqrt(diag(data_gram))` — a
DIFFERENT matrix's diagonal than the decomposition's
`sqrt(diag(data_gram + S))`; the surrogate's equilibrated basis must use
the retained decomposition's own `column_scale`, never the summary's.

### Open questions (resolve in the implementation sessions)

- Whether `scipy.linalg.cholesky_update` (added 1.14) is available at the
  project floor (`scipy>=1.10`) — otherwise implement the Givens/hyperbolic
  kernels directly (small, well-documented algorithms) or always use
  Woodbury (still O(p²·r) via triangular solves; measure both).
- The Givens-vs-Woodbury crossover `Σr_j` threshold — measure on a_m15 /
  b_L800-class fits.
- Whether the surrogate should also serve the outer-iteration candidate
  fit warm start (it already produces `β̃`; the candidate fit currently
  warm-starts from the accepted trial's β — likely equivalent, verify).

### Validation plan (later sessions)

- Fixed-ρ equality: surrogate objective vs exact objective on a λ-grid
  around a converged fit — agreement tolerance set by the frozen-W error,
  NOT bit-equality (frozen W is an approximation; only ordering and
  Armijo decisions must be robust).
- Factor-update algebra: updated `R` vs fresh `decompose_gram` at trial λ
  — 1e-10 relative on `log|H|` and solves, across up, down, and mixed
  steps, including a downdate-instability case that must trigger Woodbury.
- Convergence parity: full suite + REML oracle tests; final
  objective/prediction parity (λ-trajectory equality NOT required —
  reordering noise, per audit RFC-2 validation note).
- `tests/test_reml_trial_carry_forward.py` must pass unmodified.
- Perf: b_L800-class re-profile targeting −15% optimizer wall; flagship
  and tensor-cost CI gates stay green.

---

## Task 1 (SHIPPED in this PR): retain the slope-system decomposition on the result

The seam RFC-12b needs, shipped ahead of the algorithm: `fit_irls_direct`
already computes `reml_slope_rank = decompose_gram(centered_final.hessian)`
(in the dense `_compute_reml_geometry` branch of `_fit_irls_direct_once`)
and keeps only its pseudo-inverse. Retain the `RankDecomposition` itself
on `PIRLSResult` behind an opt-in internal kwarg (default off: zero
behavior and memory change for every existing caller).

**Files:**
- Modify: `src/superglm/solvers/pirls.py` (PIRLSResult field)
- Modify: `src/superglm/solvers/irls_direct.py` (kwarg + population)
- Test: `tests/test_reml_factor_retention.py`

**Interfaces:**
- Produces: `PIRLSResult.reml_slope_decomposition: RankDecomposition | None`
  (default `None`); `fit_irls_direct(..., _retain_reml_decomposition=False)`
  (underscore-prefixed: internal-only, matching `_compute_reml_geometry`
  et al.). Later 12b sessions consume `.cholesky_factor`, `.solve()`,
  `.log_pdet`, `.method`, `.rank_truncated`, and the decomposition's OWN
  `column_scale` (see the equilibration note above —
  `REMLGeometrySummary.column_scale` is a different matrix's diagonal),
  together with `REMLGeometrySummary` (`mean_x`, `sum_w`).

- [x] **Step 1: Write the failing tests**: retention is `None` by default
  and without REML geometry; opt-in retention matches an independent
  oracle (`slogdet`/`solve` of the centered Hessian assembled from
  `S_override`) and the published geometry; the certified rank-truncated
  path retains the certified decomposition with the measure identity
  intact; the structured path retains `None` by design.
- [x] **Step 2: Run and watch them fail** (unknown kwarg / missing field).
- [x] **Step 3: Add the field and kwarg, populate under
  `_compute_reml_geometry and _retain_reml_decomposition`.**
- [x] **Step 4: Tests pass; full suite green.**
- [x] **Step 5: Commit.**

## Task 2+ (LATER SESSIONS, in order)

2. Retain the centered RHS (`XtWz_c`) alongside the decomposition (extend
   `REMLGeometrySummary` or the retention kwarg's payload) — needed for
   `β̃` solves. DECIDE FIRST where retained state lives when the flag is
   actually on: `PIRLSResult` retention means `_publish` /
   `_freeze_result_arrays` clones the decomposition (an extra O(p²) copy
   of `cholesky_factor` per publication, plus `copy.deepcopy` at every
   `_mutable_copy`) — the exact per-trial cost RFC-12b exists to remove.
   The `cache_out` idiom in `fit_irls_direct` already retains dense
   per-fit state for the cached-W optimizer WITHOUT entering the
   published result graph; the line-search consumer may want that home
   instead, keeping `PIRLSResult` retention for terminal/diagnostic use.
3. Factor-update kernel: `update_cholesky(R, U, sign)` + Woodbury fallback
   + determinant-lemma logdet, TDD'd against fresh decompositions
   (validation tolerances above).
4. Surrogate trial evaluation function (frozen moments + kernel + exact
   deviance pass) with fixed-ρ equality tests.
5. Wire into the `direct.py` step-halving loop behind a private flag with
   the exact-acceptance re-check and exact-trial fallback; parity suite;
   flip default after b_L800/a_m15 measurement.
