# Interaction Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank every candidate `ti(a, b)` pair of a fitted main-effects `fit_reml` model by a penalized score statistic computed from bin-space sufficient statistics — no per-pair refits — so interaction *discovery* becomes as cheap as interaction *fitting* just became.

**Why now:** `perf/cheap-interactions` made a fitted tensor cost ~1.35× the discrete path (was 17.6× on master, this machine). The remaining differentiator for the fit_reml GAM story is telling the user *which* interactions to fit. Everything the statistic needs is already in the codebase: the exact null score vector is the same quantity `compute_lambda_max` builds (RFC-3), the per-pair weighted 2-D histograms are the same aggregation `_cross_gram`'s disc×disc path runs, and lossless support compression already stores exact `row_index` codes per group.

**Positioning vs published work (research sweep 2026-07-28):** BOLT-SSI (arXiv:1902.03525) screens all pairs by likelihood-ratio increments over 3-way contingency tables and proves sure screening with a quantified discretization loss (relative efficiency ≤ 1.21 worst-case Gaussian); sprinter-GLM (arXiv:2401.08159) proves frozen-main-effects 1-D score ranking is equivalent to conditional-covariance ranking. Neither handles *penalized smooth groups* (a `ti()` block with its wiggliness penalty) — that is the gap this plan fills, and it is publishable territory.

---

## Statistic

Let the fitted mains model give `eta_hat`, `mu_hat`, and the exact null score vector

```
s_r = w_r * (dmu/deta)_r * (y_r - mu_hat_r) / V(mu_hat_r)        # identical to compute_lambda_max's score
W_r = w_r * (dmu/deta)_r^2 / V(mu_hat_r)                          # converged working weights
```

For a candidate pair (a, b) with marginal bases `B_a (n_a × k_a)`, `B_b (n_b × k_b)` evaluated on the distinct values (support codes `i_a[r]`, `i_b[r]`):

1. **Cell aggregation (one O(n) pass per pair):** `S_cell[i,j] = Σ_{r in cell} s_r` and `W_cell[i,j] = Σ_{r in cell} W_r` via one fused bincount over the joint index `i_a * n_b + i_b`.
2. **Score:** `U = vec(B_a' S_cell B_b)` — exact, no row-space work.
3. **Curvature:** `V_full = (B_a ⊗ B_b)' diag(W_cell) (B_a ⊗ B_b)` assembled from `W_cell` by the existing disc×disc sandwich; profile out the pair's own overlap span (intercept + s(a) + s(b) columns, cross-moments from the *same* histograms): `V = V_full − C' M⁻¹ C`. Other terms' co-adjustment is deliberately omitted (their effect enters through `mu_hat`; the omitted variance adjustment is the same approximation BOLT-SSI and sprinter make, and the confirmatory refit is the gate).
4. **Penalized statistic:** with the pair's tensor penalty `S_ti` (existing construction), pick `lambda_0` such that `tr((V + lambda_0 S_ti)⁻¹ V) = edf_0` (bisection on a scalar, default `edf_0 = 4`), then

```
T = U' (V + lambda_0 S_ti)⁻¹ U
```

ranked across pairs (optionally normalized per edf_0; edf_0 fixed across pairs makes raw T comparable).

**What T is:** a penalized efficient-score statistic at the null of the fitted mains — large when the working residuals carry structure over the (a,b) cells that a smooth-of-fixed-complexity surface can absorb. Selection is by rank; calibration is by refit, not by the screening p-value.

**Exactness contract:** for supports taken from lossless compression the aggregation is exact (it is the same sufficient-statistic algebra as the fit). Continuous covariates without a stored support get screening-only marginal bins (default 64 quantile bins) — that approximation lives in screening ONLY and never enters a fit. `discrete=True` semantics are untouched.

**Cost:** one fused bincount pass + O(k² cells) dense algebra per pair. 10 rating factors → 45 pairs → ~45 O(n) passes ≈ well under a second at n=100k; the confirmatory refit of the top-k is the expensive step and is user-triggered.

---

## Non-goals

- No automatic model mutation. Screening returns a ranking; the user (or an explicit convenience kwarg) adds terms.
- No hierarchy assumption beyond "mains are fitted first" (pairs whose mains are absent are out of scope v1).
- No 3-way interactions in v1 (the same algebra extends; ranking cost becomes O(p³) passes).
- No changes to protected semantics: `select=` vs `selection_penalty`, `k`/k−1 contract, `sample_weight`=exposure, exact-vs-discrete separation all untouched.

---

### Task 1: Null score + per-pair cell moments, exactness-pinned

**Files:** create `src/superglm/screening/_pair_moments.py`, `tests/test_interaction_screening.py`

- [ ] Test: build a small Poisson fit, one candidate pair with integer covariates; assemble `U` and `V_full` densely from the materialized `(n, k_a·k_b)` row-Kronecker design and compare to the cell-aggregated assembly — agreement `rtol 1e-12` (this is the exactness pin; do not loosen).
- [ ] Test: signed `s` (score vectors are signed by construction).
- [ ] Implement: `pair_cell_moments(codes_a, codes_b, n_a, n_b, s, W) -> (S_cell, W_cell)` (fused bincount) and `pair_score_curvature(B_a, B_b, S_cell, W_cell) -> (U, V_full)`.
- [ ] Reuse the score-vector construction shared with `compute_lambda_max` (extract the family-factor score into a helper both call; behavior of `compute_lambda_max` must stay bit-identical — its KKT boundary tests pin it).

### Task 2: Overlap profiling + penalized statistic

**Files:** `src/superglm/screening/_score_stat.py`, tests

- [ ] Test: with `lambda_0 -> inf` the statistic goes to the unpenalized-in-null-space limit; with `S_ti = 0` it reduces to `U'V⁻¹U`; EDF solver hits `edf_0` within 1e-6.
- [ ] Test: profiling out the pair's own margins removes score mass explainable by `s(a) + s(b)` alone (construct a case where the "interaction" is purely additive → T near null level).
- [ ] Implement `penalized_score_statistic(U, V, C, M, S_ti, edf_0) -> ScreenedPair(T, edf_0, lambda_0)` with the scalar bisection for `lambda_0`.

### Task 3: `screen_interactions` public API

**Files:** `src/superglm/model/screening_ops.py`, `model/api.py` facade hook, tests

- [ ] Test (oracle): simulate Poisson with a known `ti(x1, x2)` signal among ≥8 noise features across 5 seeds — the true pair ranks first every time.
- [ ] Test (null): no interaction anywhere — the max statistic across pairs stays within generous null bounds across seeds (rank stability, not strict calibration).
- [ ] Test (API): requires a fitted `fit_reml` model; unfitted → clear error; `candidates=` restricts pairs; result is a small frame/dataclass sorted by T with `(pair, T, edf_0, lambda_0, n_cells)`.
- [ ] Implement: derive codes from fitted group matrices — support-compressed groups expose exact `bin_idx`; Categorical exposes codes; Dense/continuous falls to Task 4's binning. Marginal bases evaluated on support values (reuse the fitted spline builders).

### Task 4: Screening-only binning fallback for continuous covariates

**Files:** `screening/_binning.py`, tests

- [ ] Test: a continuous covariate (no repeated values) screens via 64-bin quantile codes; the oracle test passes with continuous x1/x2; a doc-visible flag on the result marks pairs that used approximate binning.
- [ ] Implement weighted quantile binning (screening-only; document loudly it never enters fits).

### Task 5: Real-data validation + measurement

- [ ] freMTPL2 n=100k, mains = 4 splines + Area: screen all pairs; assert wall < 1 s for the sweep and record the ranking in the plan (domain expectation: DrivAge:BonusMalus and DrivAge:VehAge near top).
- [ ] Confirmatory loop: refit top-2 as `ti()` terms, record deviance/EDF gain vs mains-only — the end-to-end user story this branch exists for.
- [ ] Record: screening cost table (pairs × n), and the gap between screening rank and refit deviance-gain rank (the honesty metric for the approximation).

---

## Follow-ups deliberately out of scope

- Fusing the 45 bincount passes into one blocked pass (matters only at n ≳ 10⁷).
- Permutation/GCV calibration of a screening threshold (v1 ranks; the refit decides).
- 3-way sweeps; `by=`-style varying-coefficient candidates; sz/fs factor-smooth pairs (different support geometry, own plan).
