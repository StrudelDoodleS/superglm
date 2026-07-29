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

---

## Task 3 design amendment (recorded 2026-07-28, pre-implementation)

Investigation of `features/interaction.py` collapses Tasks 3a/3b:

1. **Do not extract bases or codes from fitted group matrices.** Profiling is
   span-invariant (`V - C'M^-1 C` is the Schur complement onto the span's
   orthogonal complement), so screen in the RAW centered-marginal coordinates
   that `_prepare_centered_marginals(x1, x2, parent_specs)` already returns:
   `(B1, B2, S1, S2)` — exactly the basis/penalty a real `ti()` would be
   given. `S_ti = kron(S1, I) + kron(I, S2)` (interaction.py:1124-1125),
   column order = `_row_kron`'s C-order, which pair_score_curvature matches
   (verified by the Task 1 review). Codes are screening-owned:
   `np.unique(x, return_inverse=True)` per margin — the Categorical sink-bin
   trap never arises.
2. **Every overlap quantity comes from the two cell tables.** With menus
   `A = B1[support]`, `B = B2[support]` and the Task 1 cells `W_cell`,
   `S_cell`: `M` blocks are `sum(W_cell)`, `A' diag(rowsum W) A`,
   `A' W_cell B`, ...; `u_m` is `[sum(S_cell), A' rowsum(S_cell),
   B' colsum(S_cell)]`; `C` blocks are `vec(A' W_cell B)` and
   `einsum('ij,ic,ip,jq->cpq', W_cell, A, A, B)` (+ mirror). No extra data
   passes beyond Task 1's single fused bincount.
3. **U_nuisance:** pass the overlap block's actual score `u_m` as above. At a
   freshly fitted mains model it is near zero for the fitted spline spans and
   exactly `sum(s)`-driven for the intercept; passing the measured value
   rather than assuming zero keeps the additive-case null exact.
4. **s and W at the fitted state:** `working_score(y, mu_hat, eta_hat, w, ...)`
   and `W = w * dmu_deta^2 / max(V(mu_hat), floor)` from the fitted model's
   predict path — the same quantities the solver's own IRLS forms.
5. Review cadence agreed with Max: no standalone Task 2 review; one Opus 5
   max review over Tasks 2+3 together after 3c lands, attention on the
   U_nuisance choice, edf-clamp ranking at bracket edges, and coordinate
   consistency with interaction.py.

---

## Naming and positioning (decided 2026-07-28)

**The method is named PSST — Penalized Smooth Score Test** (Max's pick over
SPECS/TRACE). Formal phrase in docs: "penalized smooth score screening". The
API stays `screen_interactions()`; PSST appears in docstrings, the result
object, and any paper. Novelty posture, honestly stated: the primitive
(score-testing a candidate smooth at a fitted null) is classical — Lin 1997,
Zhang & Lin 2003, the RLRT line, Wood 2013 — and the claimed contribution is
the combination: all-pairs screening at one fused histogram per pair via the
cell algebra, the fixed-EDF lambda_0 calibration that makes raw statistics
rank across pairs, and the exact-support exposure-weighted GAM setting. A
targeted prior-art sweep (non-arXiv test literature, glinternet/hierNet/xyz,
mboost, anything citing Wood 2013 for screening, actuarial venues and
Emblem/Radar-era proprietary heuristics) is REQUIRED before any public
novelty claim.

**Linear interaction-search literature — positioning, not machinery:**
glinternet/hierNet are selection-by-fitting (all pair blocks in one penalized
fit) — the expensive endgame PSST screening avoids; superglm's own
selection_penalty path already offers that shape as an alternative
confirmatory stage and both are paper baselines. xyz (Thanei et al., JMLR
2018) solves subquadratic pair SEARCH for huge p — irrelevant at rating-factor
p (45-1200 pairs brute-forces in well under a second) but the backstop idea if
screening ever faces hundreds of engineered features. Hierarchy stance stated
for v1: mains-first (screen only pairs whose margins are fitted), i.e. weak
hierarchy by construction.

---

## Task 5 first measurement (2026-07-28, reference box, benchmarks/benchmark_psst.py)

freMTPL2 n=100k, mains = 5 ps-splines (incl. Density, 1,568 distinct) + Area:

- mains fit_reml 1.24 s (dev 31912.49); **PSST sweep of all 10 pairs 2.0 s**;
  confirmatory refit of top-2 as ti() 5.1 s, deviance gain 116.5.
- Ranking: **DrivAge:BonusMalus first (T=27.7)** — the domain-expected pair —
  then VehAge:VehPower 19.2, VehAge:BonusMalus 18.7, DrivAge:VehPower 18.0;
  Density pairs (129-154k cells) screen mid-pack without drama.
- The plan's "< 1 s sweep" target missed at 2.0 s: ~10 pairs x 2 full-length
  marginal-basis builds each. Obvious v2: build each feature's centered
  marginal ONCE (5 builds, not 20) and reuse across pairs — expected to
  recover most of the gap. Not blocking: 2 s to rank vs 5 s to confirm one
  candidate is already the right economics.

---

## edf0 sensitivity, measured (2026-07-28, scratchpad edf0_sweep)

freMTPL2 100k, rank stability across edf0: **3-8 is a plateau** — same top
pair (DrivAge:BonusMalus), z-separation 7.3 -> 8.4 -> 8.9 -> 8.9 with
diminishing gains; edf0=2 is too blunt (a 2-knob probe cannot represent the
curved age x BM corner, DrivAge:VehPower overtakes); by edf0=12 the question
changes (VehAge:VehPower's higher-frequency structure overtakes, T growing
19 -> 56 across budgets vs 28 -> 54 — itself a finding worth a refit).

Synthetic power confirms the bandwidth theory exactly: a smooth z1*z2 signal
is best at the SMALLEST budget (z-sep 45.8 at edf0=2, monotone down to 19.2
at 12); a sin(2.2 z) x sin(2.2 z) signal is invisible at edf0<=4 (rank 9/10)
and only emerges by 12 (rank 2). edf0 is a probe bandwidth: it must be at
least the true shape's complexity, and every knob beyond that costs noise.

**Decision:** default stays 4 — the cheap end of the measured real-data
plateau, matched to the low-complexity prior for pricing interactions.
Recommended protocol in docs: screen at edf0=4 and once more at 8-12; stable
ranks are trustworthy, disagreements flag high-frequency candidates for a
refit rather than a bigger screen.

**Ladder default (2026-07-29):** edf0 is now a scan — default (2,4,8,16),
ranked by z = (T-edf0)/sqrt(2*edf0) at each pair's best rung; per-pair cost
is unchanged (cells/menus/profiling once, one small solve per rung). The
wiggly synthetic goes rank 9/10 -> 2/10 (all seeds, winning at 16). On
freMTPL2 the scan surfaces VehAge:VehPower as the top signal (z=9.2 at
edf0=16, T=68 nearly unpenalized) ahead of DrivAge:BonusMalus (z=8.9 at 8),
and the winning-rung column doubles as a shape diagnostic: Density pairs win
at rung 2 (tilt-level evidence only). Sweep still 2.0 s; suite 4754.

---

## Why the ladder is a grid, not an optimizer (2026-07-29, scratchpad edf0_curve)

Question raised: the ladder {2,4,8,16} is a set — is there a continuous
function underneath, and is picking edf0 a convex/Brent-able problem?

There IS a function: z(h) = (T(h) - h)/sqrt(2h) over the continuum of
achievable budgets h; the ladder is a 4-point sample and max-over-rungs
approximates sup_h z(h). In the penalty eigenbasis (S q = d V~ q after
profiling), with shrinkage h_i(lambda) = 1/(1 + lambda d_i):
T = sum u_i^2 h_i, edf = sum h_i, so z = sum (u_i^2 - 1) h_i / sqrt(2 sum
h_i) — a standardized soft-windowed sum of noisy signal-minus-noise
increments swept across the eigenvalue axis. That object is generically
MULTIMODAL: a bump appears wherever a run of eigendirections carries
signal, so any pair with energy in two spectral regions (tilt + high
frequency) has two peaks.

Measured (fine geometric grid, 26 rungs, h in [1.3, 36]):
- 6 of 10 freMTPL2 pairs are multimodal; VehAge:BonusMalus has 3 peaks.
- VehAge:VehPower is the counterexample to any local optimizer: local max
  z=3.6 at h=1.3, true peak z=9.8 at h~28. Brent from a low start reports
  the sweep's #1 pair as mediocre.
- Even unimodal cases often peak at the BOUNDARY (strong smooth signals:
  z climbs as h -> h_min; smooth synthetic sup=57 at h=1.3), so "optimize
  h per pair" mostly returns an edge, not a meaningful bandwidth.
- Eigen-location confirms the mechanism: smooth synthetic has 96% of
  excess score energy in the 5 smoothest directions (unimodal); wiggly has
  27%, energies at ranks {3,4,8,21,24}/25 (two bumps).
- Ladder vs sup over the grid: top-5 ranking IDENTICAL; max z left on the
  table 1.29 (all large gaps are left-boundary climbers already ranked
  high, plus VehAge:VehPower's peak at 28 vs rung 16, gap 0.55).

Where the clean structure lives: the INNER problem — lambda0 from a budget
via tr((V+lambda S)^-1 V) = edf0 — is smooth and monotone (each h_i is),
which is exactly why the existing bisection is sound. T(lambda) and
edf(lambda) are individually convex in lambda; convexity dies in the
standardized ratio, by construction, because u_i^2 are noisy.

Lineage (adds to the prior-art sweep list): max over a budget ladder of
standardized score sums is the adaptive-testing family — Fan 1996 adaptive
Neyman (hard-truncation h_i in {0,1} version of our z), Eubank & Hart 1992
order selection, Dumbgen & Spokoiny 2001 multiscale testing. Canonical
practice there: geometric grids (corr(z(h), z(h')) depends on h'/h, so
dyadic spacing samples at ~equal correlation steps; sup over ALL h needs
Darling-Erdos loglog corrections). The ladder is the standard
discretization of this object, not an arbitrary set. Novelty posture
unchanged: the scan is known machinery; the penalized-smooth-score +
cell-collapse core is where the new work is.

Recorded refinements, none blocking (ranking measured stable without
them): (a) extend ladder to {1.5,2,4,8,16,32} — two extra small solves per
pair, covers both boundary regimes; (b) exact-variance standardization
z = (T - sum h_i)/sqrt(2 sum h_i^2) — var(T) = 2 tr((AV)^2) <= 2 edf0, so
current z is conservative where shrinkage is diffuse; (c) rung covariance
is closed-form, cov(T_r, T_s) = 2 tr(A_r V A_s V), enabling a calibrated
max-z p-value per pair if screening ever needs more than a ranking.

---

## Fork prototypes: exact-variance z, extended ladder, calibrated max-z (2026-07-29)

Two full-context fork agents prototyped the recorded refinements in the
session scratchpad (forkA_*, forkB_* files; repo untouched during the
experiments). Facts both forks agree on:

- Exact-variance standardization z = (T - tr(AV~))/sqrt(2 tr((AV~)^2)) is
  algebraically verified (0.0 trace mismatch over 1140 rung solves; the
  shipped sd overstatement is one-sided, multiplier 1.006-1.347, peaking
  at rungs 4-8). Its ONLY ranking effect anywhere: swaps the near-tied
  freMTPL2 leaders (DrivAge:BonusMalus over VehAge:VehPower — rung-8
  shrinkage is more diffuse, sum h_i^2 ~ 5.2 vs edf 8, so conservative z
  understates it more). Costs one rank on one wiggly seed via a literal
  tie-flip (T 27.0375 vs 27.0358). Widens null right tail ~14% (correct
  scaling, not a defect); z<10 gate keeps >=2.1x headroom.
- Extended ladder (1.5,2,4,8,16,32): closes both boundary regimes from
  the curve analysis (VehAge:VehPower gap-to-sup 0.55 -> 0.07, wins at
  rung 32; left-boundary climbers recover ~2/3 of their gaps at rung
  1.5), promotes wiggly seed 0 to rank 1, degrades nothing, +~1% sweep
  cost. Under the pure null the winning rung loads on the extremes
  (~58% at {1.5,32}) — shape diagnostic is only meaningful at material z.
- Measured adjacent-rung correlations 0.85-0.93 (exact, from
  cov(T_r,T_s) = 2 tr(A_r V~ A_s V~)): the dyadic grid sits in the sweet
  band — correlated enough that no peak hides between rungs, distinct
  enough that rungs aren't duplicates. This is the shipped-form answer to
  "how do we know the grid is fine enough" — computable per pair/dataset.
- Calibrated max-z p-value: plain-Gaussian route is materially
  miscalibrated (null KS p=0.006; overstates far-tail nlp by >20 decades)
  — never ship it. Moment-matched route (Satterthwaite scaled-chi2
  marginals + Gaussian copula with exact R) validates against 40k-draw
  simulation: MAE 0.011, tail MAE 0.0016/max 0.0031, null uniformity KS
  p=0.276. Bonferroni far-tail bracket <=0.15 decades on leaders.
  Cost ~10-30 ms/pair beyond rung solves (+~1s on the 2s sweep).
  IMPLEMENTATION TRAP: scipy Genz MVN-CDF at default tolerance is ~100x
  slower on the near-singular rung R; must pass abseps~5e-4,
  maxpts~50k*dim.
- Null scope limit: calibration validated on n=2500 Poisson synthetics
  (10 seeds x 10 pairs); re-run the null rig on structurally new data
  shapes (sparse counts, heavy cells) before trusting p's there.

Fork verdicts diverged on one point: A rejects exact-variance z as the
DEFAULT ranking (no measured power gain anywhere, breaks the doc-simple
formula, tie-flip risk); B endorses it as "more correct" and needing
sign-off. Synthesis: exact variance is calibration machinery, not
ranking machinery — the headline z stays conservative and doc-simple;
the exact sd lives inside the opt-in calibration layer where its
correctness actually matters (it is step 1 of the mm route).

Proposed landing (pending decision): (1) extended ladder as default —
as the RULE "octaves filling the achievable [null-dim, rank] range",
not hardcoded constants; (2) opt-in calibrate= flag adding mm-route
-log10 p columns, Gaussian route excluded; (3) posture guard: screening
calibration conditional on the working model, not confirmatory
inference — ranking-only stance and the confirmatory-refit gate stand.
Whatever lands folds into the pending Tasks 2+3 Opus 5 max review.

---

## Review 3 (Opus 5 max, Tasks 2+3) and fixes (2026-07-29)

Verified sound by independent reconstruction (not merely read): coordinate
consistency end-to-end at 3e-15 with asymmetric margins/penalties (the
dense pins DO catch transpositions); span invariance of the profiled
statistic to 1e-11; pinv-fallback consistency (single MinvC reused);
achieved edf is what feeds z, clamped rungs report honest achieved values;
sqrt(2*edf0) conservative as claimed; working score/weights match the
fitting path to 2.8e-16 across Poisson/Gamma/Binomial; NaN-row path and
Task-1 interface consumption correct.

Five CONFIRMED findings, all fixed same day:

1. OFFSET (correctness): screening dropped model._fit_offset — an
   offset-fitted model was linearized at the wrong mean and leftover main
   effects screened as interaction signal (silent, plausible-looking).
   Fix: offset= parameter defaulting to the fit offset (shape_ops
   precedent). Regression test pins default==explicit and the offset null.
2. DISPERSION (statistical validity): E[T] = phi*edf0, so z assumed
   phi=1; a sigma=3 Gaussian pure null hit z=25 with the winning rung
   pinned at the endpoint (null mean of z is (phi-1)*sqrt(edf0/2), an
   increasing function of the budget — the scan's argmax was
   deterministic off phi=1). Fix: T is reported on the T/phi_hat scale,
   phi_hat = Pearson dispersion of the mains fit with (n - edf_mains)
   denominator, floored at tiny. Unit-dispersion families unchanged.
3. best=None crash: NaN y / zero weights / empty edf0 killed the sweep
   with AttributeError. Fix: explicit input validation up front + NaN row
   if every rung degenerates.
4. edf0 validation: 0-d arrays now work; empty/NaN/<=0 budgets raise
   (a <=0 budget used to clamp to the null dimension and WIN the scan).
5. candidates/select validation: malformed pairs, unknown or duplicate
   features raise with the screenable-feature list; select=True parents
   raise one clear error up front instead of NotImplementedError per pair
   three modules down. Docstrings: statistic column is winning-rung and
   not row-comparable (rank by z); clamped lambda0 is a bracket edge.

MEASURED CONSEQUENCE — the honest freMTPL2 table: phi_hat = 2.480 on the
100k Poisson mains (edf 29.2), so all previously recorded freMTPL2 z
values carried ~2.5x dispersion inflation plus a rightward rung drift.
Post-fix: DrivAge:BonusMalus z=2.53 @ rung 4 (was 8.9 @ 8),
VehAge:VehPower z=2.14 @ rung 8 (was 9.2 @ 16), DrivAge:VehPower 1.94 @ 2,
VehAge:BonusMalus 1.91 @ 2; Density pairs sink to z <= 0.78. Same top-4
membership — the domain-expected pair is back on top; the confirmatory
refit gain is unchanged at 116.49 deviance for the same top-2. Ranking
robust, magnitudes now honest, and the "wants to wiggle at rung 16" story
about VehAge:VehPower was partly a dispersion artifact (it now wins at 8).

SUPERSESSION: freMTPL2-specific z magnitudes, winning rungs, and z(h)
curve geometry in ALL earlier entries (Task 5, ladder default, grid-vs-
optimizer, fork prototypes) predate the dispersion fix — treat as shape
evidence only. Unit-dispersion synthetic results (oracle, wiggly rank
recovery, fork null calibration) stand as recorded. The extended-ladder
ADOPT verdict and the fork freMTPL2 numbers must be RE-MEASURED post-fix
before any landing decision; the multimodality mechanism and the wiggly
Brent counterexample are unaffected.

Suite: 4759 passed / 152 skipped (5 new regression tests).

---

## Ladder re-measured on the fixed statistic: extension verdict REVERSED (2026-07-29)

Re-ran the full curve/ladder analysis post-5b3d8cf (T/phi_hat scale,
phi_hat=2.480 on freMTPL2; scratchpad edf0_curve_fixed.py).

Honest z(h) geometry: peaks shift LEFT across the board (DrivAge:BonusMalus
argmax 4.9, VehAge:VehPower 9.5 — was 6.4 / 27.6); NO pair peaks beyond
~10, so every pre-fix rung-32 win was dispersion artifact. Real-data
multimodality drops to 2/10 pairs (was 6/10) but the Brent counterexample
stands: VehAge:VehPower keeps a false local optimum at h=1.3 (z=0.98) vs
its true peak z=2.19 at h=9.5, and the wiggly synthetic keeps 2 peaks
(argmax 11.9). Honest leaders are near-tied: sup_z 2.60 / 2.57 / 2.54 /
2.19 — the pre-fix separation (9.2 vs 8.9 vs 6.3) was manufactured by
phi inflation.

The decisive test — confirmatory refit of each ladder's top-2:
  4-rung top-2 (DaBM + VaVP): deviance gain 116.49
  6-rung top-2 (DaBM + DaVP): deviance gain  79.55
  union top-3:                              140.56
Rung 1.5 promotes tilt-dominant pairs (DrivAge:VehPower, z 1.94 -> 2.35)
above VehAge:VehPower, but the refit shows VaVP carries ~61 marginal
deviance vs DaVP's ~24: z at tiny edf measures evidence DENSITY (per df),
the refit harvests TOTAL evidence. Detection-optimal is not
payoff-optimal, and the screening product exists to feed refits.

Ladder facts post-fix: oracle rank 1 on 5/5 seeds under both ladders;
wiggly ladder4 rank 2/2/2 vs ladder6 1/2/2 (marginal); null (5 seeds x 10
pairs) best-z p95 1.46 (4-rung) vs 1.79 (6-rung), max 1.81 vs 2.30 — both
far under the z<10 gate and tighter than pre-fix (studentization by
phi_hat shrinks null tails).

DECISION (reverses the fork-A ADOPT): keep the default ladder (2,4,8,16).
Rung 32's real-data case was artifact; rung 1.5 buys detection of
low-payoff tilts at the cost of worse default top-2 refit picks. The tilt
information is already surfaced honestly by the winning-rung column
(rung-2 win = tilt-level evidence). Users screening genuinely rough
domains can pass edf0=(2,4,8,16,32) per call — the API accepts any
budgets. Doc protocol: with near-tied leaders, refit top-3, not top-2
(union gain 140.56).

Held items status: extended-ladder-as-default is now REJECTED on
evidence; the calibrate= layer question stays open (its simulation-
validated machinery is unaffected by the fix, but its freMTPL2 table
must be regenerated on the fixed statistic if pursued).

---

## Generalization evidence: synthetic gauntlet + three new real books (2026-07-29)

Question answered: "what happens if I point this at other datasets?"
All runs through the PUBLIC screen_interactions, default ladder, post-fix
statistic (scratchpad gauntlet.py, real_books.py; data fetch documented in
real_books.py — CASdatasets beMTPL97/pg15training via pyreadr, freMTPL2sev
via OpenML joined to the local freq table on IDpol).

Synthetic gauntlet (10 configs x null/planted x 3 seeds, 9s wall):
- NULLS HONEST EVERYWHERE: max best-z by config 1.73 (Poisson) to 4.29
  (heavy lognormal exposure), incl. phi=9 Gaussian (2.07), phi=100
  Gaussian (2.07 — pre-fix this was z~300), Gamma phi=0.5 (3.11),
  Bernoulli (2.45), rho=0.85 copula-correlated pair (2.96), 300x220 grid
  at n=20k (1.99), n=800 (2.05), mu-dependent mixed-Poisson (3.56).
- phi_hat TRACKS TRUTH in every family: 1.01/1.44/9.0/100.0/0.50/1.00...
- POWER: planted smooth pair ranks 1/10 in ALL configs (z 11 at n=800 to
  359 on the 300x220 grid). Two honest weak classes measured:
  corner-localized hinge interaction ranks 6/10 worst-seed (z~0.4-0.9 —
  little data in the active corner; a refit would struggle equally), and
  rho=0.85 correlated pairs demote to rank 4 (ridge support leaves the
  off-ridge tensor signal thin — same identifiability the ti() refit
  faces). Neither is a validity failure; both are power limits, and both
  null cleanly.

Three real books, end to end (mains+screen+top-2 confirmatory refit):
- beMTPL97 (Belgian MTPL, 163k, 6 splines incl raw long/lat): the screen
  surfaces long:lat FIRST at z=14.5 (wins at rung 16, lambda0=0.67) —
  independently rediscovering the published 2-D spatial-smooth modeling
  choice for this exact book, never having been told the columns are
  coordinates. bm:agec second at z=10.8. Top-2 refit: deviance gain
  203.7. Screen 3.5s for 15 pairs at 163k. phi_hat=1.16.
- freMTPL2sev (French severity, 26.4k claims, Gamma, cap q99.5=34,377,
  phi_hat=3.55): weak-to-moderate structure, DrivAge:BonusMalus top at
  z=1.63; top-2 refit gain 93.6 raw (~26 phi-scaled) — modest, matching
  the modest z.
- pg15training (Pricing Game TPPD, 100k, phi_hat=1.03): little
  interaction structure, top z=3.6 (Poldur:Value); top-2 refit gain 13.6
  — small gain matching small z.
Across books, z magnitude and refit payoff now COHERE (z 14.5/10.8 ->
gain 204; z 1.6 -> ~26 scaled; z 3.6 -> 14): the dispersion fix makes z
meaningful across datasets, which is the property that transfers.
Zero crashes, zero NaN skips (largest grid 545k cells), screens 1.3-4s.

Pre-flight protocol for a new book (docs material): (1) check the
printed phi_hat is sane for the family; (2) read z against the measured
null envelope (best-z < ~4.5 across every config tested; the z<10 gate
has >2x headroom); (3) winning rung 2 = tilt-level evidence; (4) always
confirmatory-refit top-3 — the refit, not the screen, is the gate.
Known gaps unchanged: continuous high-cardinality pairs above the cell
budget NaN-skip (Task 4 quantile fallback), corner-localized power.

---

## Task 4 shipped: quantile-bin fallback for continuous pairs (2026-07-29)

Pairs whose raw unique-value grid exceeds max_cells no longer NaN-skip:
any margin above screen_bins (default 256) unique values is compressed to
empirical-quantile bins with the basis evaluated at within-bin means, and
the row is flagged approx=True (new result column). The exact path is
untouched — pairs within budget never bin, flagged approx=False. NaN rows
remain only for pairs still over budget after binning (n_cells reports
the attempted grid) or degenerate statistics.

Measured contract (pinned in tests at 10%, measured 3.5%): on a SIGNAL
pair the binned z tracks the exact z (59.1 vs 57.1 on a 2500x2500-unique
continuous pair, same winning rung; the 6.25M-cell exact run takes 0.1s
when the budget is raised, so the comparison is direct). On NULL pairs no
tight agreement is promised or wanted: at clamped unpenalized rungs,
2500-point support picks up high-frequency noise curvature that 256
bin-means smooth away (null z 1.59 exact vs 0.60 binned) — the binned
null is the better-behaved one. Docs updated (guide + docstrings).
Suite 4762 / 152.

---

## PR #171 review response (2026-07-29)

Two reviewers (claude, codex) produced 12 distinct findings; disposition:

FIXED (commits f857a2f, 3670884, d7322da, 2a319e9 + docs):
- sample_weight now inherits from the fit like offset (BLOCKING - the
  offset bug's twin; regression test mirrors the offset pin).
- y/sample_weight row-alignment and covariate finiteness validated.
- Per-feature support+marginal caches: P basis builds instead of
  P*(P-1); verified identical to 1 ULP; freMTPL2 sweep 1.9s -> 1.3s.
- Cell budget now also bounds the (n_a*k_b^2) curvature intermediates;
  quantile binning escalates largest-margin-first.
- Bisection stops when the bracket exhausts float resolution.
- BLAS wide-design release is now owned by the declaring fit's scope:
  re-arms for remaining narrow fits, exception-safe entry.
- Support compression gates run before any densify (no more full-basis
  allocation on the no-repeats decline) and the accept path densifies
  the support exactly once.

REASONED PUSHBACK (codex P1, dispersion denominator): using
pearson_residual_degrees_of_freedom (frequency-weight sum(w)-edf) is
internally consistent but measurably miscalibrated under the protected
sample_weight=exposure contract: known-phi exposure nulls read 1.55 at
truth 1.0, freMTPL2 phi inflates 2.48 -> 4.7, and the top-2 the refit
gate prefers (116.49 deviance) is demoted for a 79.55 set. n-edf kept;
divergence documented at the computation site. NOTE the underlying
semantic tension (CLAUDE.md exposure contract vs dispersion.py
frequency contract) is real and pre-existing - flagged for a future
decision at the library level.

DOCUMENTED (codex P1, discrete=True mains): screening probes the
exact-basis tensor; discrete refits bin marginal supports - same
support-discretization gap as the quantile fallback (~3.5% on signal
pairs). Native discrete-support screening recorded as follow-up.

DEFERRED (minor): is_lossless_support has no production consumer
(wire-or-drop follow-up); _solve_psd check_finite note stands as
fail-safe given upstream finiteness validation.

Release declaration: release:minor, 0.16.0 (new public feature:
screen_interactions + screen_bins/approx surface). Suite 4771 / 152.

---

## PR #171 review rounds 3-5 (2026-07-29)

Both reviewers verified every prior fix; iterative narrowing continued.
Fixed on-branch (18dc392, b976ed0, and this commit): released-fit-state
models refuse silent unit-weight/no-offset substitution (surviving
_fit_used_weights flag); SGL lambda_max via the composite KKT boundary
(solver-pinned at alpha 0/0.35/1); byte-bounded support-scan chunks;
screening eta taken from _predict_eta_exact (link(predict()) flipped
score signs for non-injective links - sqrt - on negative-eta rows);
inherited fit arrays verified against the FitDataGuard so reordered
frames raise instead of silently mispairing rows; already-fitted tensor
pairs excluded from the sweep (iterative screen->refit->screen works);
marginal width estimate biased LOW (over-estimates were terminal:
binned/skipped pairs inside the true budget, contradicting the approx
contract); approx semantics documented on every user surface including
the discrete=True rule; cache equivalence pinned (full sweep == per-pair
candidates screens).

Recorded follow-ups (claude round 3): support-only marginal evaluation
via tensor_marginal_info(support=, counts=) - the largest remaining
sweep transient, needs ULP-tolerance test updates; is_lossless_support
wire-or-drop; mu finiteness on the NaN path. Dispersion thread: claude
re-affirmed n-edf on the exposure-unit-invariance argument; thread
stays open by design with phi= as the frequency-weight exit.

---

## PR #171 review rounds 6-7 and merge readiness (2026-07-29)

Fixed (8487be5, aae1523): editor-desync of _fit_used_weights (non-unit
inheritance derived from the stored array at read time); approx flags
discrete pairs only when binning is lossy (cardinality > resolved
n_bins; lossless binning returns the exact support); width estimate =
n_knots (degree-0 closure); support-built menus (no (n,K) transient —
the recorded top perf follow-up, landed once claude verified the
support path is the lossless-discretization machinery); offset shape,
family-domain y, and max_cells validation; zero-weight rows excluded
from the Pearson residual d.f.; unweighted models screen holdouts
without arguments again.

Open follow-ups (recorded, not blocking): BLAS cap release before wide
build-phase reparameterizations (needs pre-build width discovery;
codex thread left open with rationale); is_lossless_support
wire-or-drop; mu finiteness on the NaN path. Dispersion thread open by
agreement. Suite 4788 / 152.

---

## Terminal pre-merge review (Opus 5 max) and merge (2026-07-29)

Verdict: MERGE. Full suite re-run independently (4788/152 exact);
escalation-loop invariant verified (menus assigned only after gates run
on authoritative dims; termination <= 6 iterations); budget guard
matches einsum_path's true intermediates; all cross-file fixes complete
on every call path; release contract clean (0.16.0 everywhere).

Doc-truth fixes landed pre-merge (approx-vs-budget independence, the
curvature-block NaN cause, intermediate budget in binning trigger,
positive-weight-row dispersion denominator). Recorded follow-ups from
the terminal review: scope-less allow_wide_design release (reachable
via estimate_nb_theta; perf-only, needs concurrent fits); coverage
gaps (curvature-block gate line, width-estimate ps cases via a bare
except, max_cells/offset validation raises, degenerate-statistic row,
largest-margin ordering with unequal margins); menu allocation
unbounded by a factor k_a/n_b (design note); no CHANGELOG file for the
compute_lambda_max /n breaking-change note (added to PR body instead).
