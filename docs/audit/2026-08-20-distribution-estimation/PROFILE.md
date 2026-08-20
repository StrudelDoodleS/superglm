# Tweedie `fit_reml` performance profile — v0.29.0

Call-stack analysis first, optimisation second. All measurements on
`244c2f87` (v0.29.0) unless an arm is named; single machine, all six thread
pools pinned to 1 (`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS`), every timed fit
alone under `flock /tmp/superglm-bench.lock`, CPU time reported beside wall
(CPU/wall = 0.997–0.999 in every timed cell, so the pins held). Import guard
asserts `superglm.__file__` resolves into the measured worktree.
Refs #339.

Instrumentation lives in `benchmarks/profile_tweedie_reml_fit.py`: counters
patched over the consumer namespaces for `reml_laml_objective`,
`profile_tweedie_reml_scale`, `prepare_tweedie_reml_scale_data`,
`minimize_scalar`/`brentq` inside the scale module, the saturated value and
analytic-score entry points, and `_evaluate_tweedie_density` (rows, calls,
time, split by value/score context). Same script runs a v0.28.0 tree, where
the scale probes self-disable and criterion counting still works. Workloads:

- **freMTPL2** (public, cached parquet): y = ClaimNb/Exposure, weight =
  Exposure, s(DrivAge, cr k=20) + s(VehAge, cr 15) + s(BonusMalus, cr 15) +
  Area, Tweedie(1.5), 94.9% zeros — banded covariates, ~5% positive rows.
- **synthetic** (CPG Tweedie(1.5), ~84% zeros): two cr-20 splines on
  continuous uniforms + 8-level categorical; "weak" truths put REML in the
  heavy-smoothing decades (λ̂ 23 / 18.9k) the original A/B's 400k case lived
  in; a "strong" variant (λ̂ < 1) is kept in the harness.
- **random-effect**: RandomEffect over 250 levels, n = 30k, ~85% zeros.

Profilers: `cProfile`/`pstats` for counts and cumulative time; `py-spy`
(150 Hz, 2,625 samples on the 400k fit) for sampling wall attribution that
survives native calls.

## 1. Where the time goes

### freMTPL2 ladder (instrumented, exact path)

| n | wall s | CPU s | REML iters | criterion evals (`laml`) | scale-profile calls | scale s | scale % of fit |
|---|---|---|---|---|---|---|---|
| 50k | 0.74 | 0.73 | 6 | 12 | 13 | 0.102 | 13.8% |
| 100k | 1.01 | 1.00 | 5 | 10 | 11 | 0.163 | 16.2% |
| 200k | 1.87 | 1.86 | 6 | 12 | 13 | 0.336 | 18.0% |
| 400k | 3.80 | 3.80 | 6 | 12 | 13 | 0.695 | 18.3% |
| 678k (full) | 6.39 | 6.37 | 6 | 12 | 13 | 1.149 | 18.0% |

Answers to the three seeded questions:

1. **`profile_tweedie_reml_scale` calls per fit: criterion evaluations + 1**
   (the discrete/direct bootstrap adds one; three finalize terminal
   evaluations re-hit the φ-cache and cost ~0). 11–17 calls per fit across
   every workload measured; the count is flat in n.
2. **Evaluations per solve.** Every solve: exactly 15–16 bounded
   `minimize_scalar` criterion evaluations (xatol 1e-9, no bracket restarts
   observed anywhere), + 1 criterion evaluation at the polished optimum,
   + 2 score evaluations bracketing the polish, + 4.2–4.6 brentq score
   evaluations, + 2 score evaluations for curvature. The per-fit φ-cache
   absorbs ~55% of value and ~57% of score evaluations (the bounded ladder's
   early probes are bracket-fixed, brentq re-evaluates its bracket ends, and
   finalize repeats whole solves), leaving ~7.7 fresh value + ~3.5 fresh
   score density passes per call.
3. **Fraction of fit: 14–21%,** plateauing ≈18% on freMTPL2 (5% positive
   rows), ≈19–21% on the synthetic (15–18% positive), ≈30% on the small
   RandomEffect fit where the per-call cost is a fixed floor against a
   sub-second fit. Essentially 100% of subsystem time is
   `_evaluate_tweedie_density` over positive rows (value passes ~56%, score
   passes ~44%); `prepare_tweedie_reml_scale_data` is 2 calls and ~1ms/fit —
   nothing left to hoist.

### Full-fit decomposition

cProfile, freMTPL2 full (6.28s total): IRLS state evaluations 1.62s (26%) —
of which `_tweedie_positive_unit_deviance` 1.35s over 61 deviance
evaluations of all 678k rows; scale profiler 1.13s (18%); centered-Gram
linear algebra ~1.0s (16%, `packed_centered_gram_rhs` — the banded packed
rung); design build ~1.0s (16%, one-shot); W-correction + finalize ~0.9s.

cProfile + py-spy, synthetic 400k (12–14s total): **Gram/centering 48.8%**
(`build_centered_system` → raw-moment rung → `_cross_gram_by_columns` →
3,064 sparse `csc_matvec` leaves, 31.4% of all samples on scipy's
`_matmul_vector` line); scale profiler 16–19%; deviance evaluations 6.7%;
design build 6.5%. Continuous covariates cannot take the packed banded rung
freMTPL2 gets, and the design (2 × `SparseSSPGroupMatrix` +
`CategoricalGroupMatrix`) is rejected by the all-spline tabmat gate
(`_is_raw_spline_tabmat_centering_candidate`) and by the mixed-discrete gate
(needs a >100-level categorical, no splines), so it lands on the per-column
moment rung — see §3.3 for why that is measured to be the right call anyway.

Inside the density evaluation, py-spy puts 11.7% of the *whole fit* on the
`wright_bessel(a, a+1, t)` value pass and 2.2% on the `wright_bessel(a, a, t)`
score pass — the special function is ~75–85% of subsystem time; the numpy
glue around it is thin.

random-effect 30k: the v0.28→v0.29 regression is the scale profiler and
nothing else — cProfile arms agree on every other bucket to ±0.02s while
0.29.0 adds 0.28s of density passes to a 0.55s fit.

## 2. The open A/B question, settled with counters

Same harness, same workloads, v0.28.0 tree vs v0.29.0 tree:

| case | criterion evals 0.28.0 → 0.29.0 | wall 0.28.0 → 0.29.0 | scale s (0.29.0) | λ̂ 0.28.0 → 0.29.0 |
|---|---|---|---|---|
| synthetic 400k | 13 → 12 | 11.37 → 12.19s (+7%) | 2.51 | x1 4.2→23.2, x2 5.9→18,896 |
| synthetic 100k | 16 → 17 | 3.43 → 4.00s (+17%) | 0.69 | x1 0.1→16,789, x2 1,192→11,382 |
| random-effect 30k | 16 → 16 | 0.37 → 0.92s | 0.28 | grp 19.3→53.0 |

**Outer evaluation counts are unchanged by the criterion change (±1 on
identical workloads).** What changed is per-evaluation cost: v0.28.0's
Gaussian-shaped scale term was closed-form (measured 1.3ms of criterion time
per *fit*), v0.29.0's exact profile costs 50–210ms per evaluation depending
on positive-row count. The original measurement's 400k *speed-up* is a
landing-point effect, not an evaluation-count effect: with the exact
criterion the fit lands on much heavier smoothing, and at 400k the
non-scale work per evaluation (inner P-IRLS on a better-conditioned,
heavier-penalised system) ran ~1.7s cheaper at equal evaluation count in
this reproduction — where that saving exceeds the scale-profile cost, 0.29.0
nets faster; where it doesn't (small n, RandomEffect), 0.29.0 pays the full
subsystem cost. Both arms converge everywhere measured; the λ̂ divergence is
the criterion fix itself (the issue-339 verification in this directory shows
the new landing points are the defensible ones).

## 3. Optimisation: what landed, what did not

### 3.1 Landed — zero-row shortcut in the Tweedie unit deviance

`_tweedie_positive_unit_deviance` routed y = 0 rows (95% of freMTPL2, and
the bulk of any zero-inflated fit, ~61 full-length evaluations per fit)
through its δ = −1 recovery branch: `log(0)`, two `exp`s and branch masks
per zero row, to produce what is exactly `2·μ^(2−p)/(2−p)`. The shortcut
computes that closed form on the zero rows and runs the careful machinery on
positive rows only. **Bitwise identical** by construction and by test
(`np.array_equal` on outputs across p ∈ {1.1, 1.5, 1.83}, zero fractions
{0, 0.5, 0.95, 1.0}, with y/μ ratio extremes down to 1e−300 and up to
1e300); isolated cost 25.5 → 7.6ms per evaluation at 95% zeros.

Interleaved A/B (5 reps per arm, one fit per lock hold, medians with
min–max spread; equivalence = λ̂, φ̂, edf, deviance, REML objective and
probe-prediction digests JSON-identical across arms and reps — they were,
in every cell):

| case | base wall s | optimised wall s | speedup | sign-stable across reps |
|---|---|---|---|---|
| freMTPL2 full | 6.039 [5.888–6.682] | 5.008 [4.924–5.082] | **1.21×** | yes |
| synthetic 400k | 13.574 [13.282–13.803] | 12.919 [12.630–13.217] | **1.05×** | yes |
| random-effect 30k | 0.666 [0.657–0.850] | 0.591 [0.583–0.621] | **1.13×** | yes |

CPU/wall 0.997–0.999 in all cells. (Absolute levels drift a few percent
between sessions — the earlier ladder measured 6.39s where this A/B's base
arm measured 6.04s — which is exactly why the arms are interleaved; the
contrast is the measurement.)

### 3.2 Tried and rejected — scale-profiler evaluation economy

Measured on captured real solve sequences (the exact (Dp, Mp) ladders from
the freMTPL2-full and synthetic-400k fits, replayed against isolated
profiler variants, 3 reps each):

- **Warm-started score-root solve** (skip the bounded minimizer after the
  first call; bracket the analytic-score root around the previous optimum;
  same brentq/xtol): returns the identical optimum (max |Δlog φ| ≤ 9e−16
  across both sequences) but saves only ~5% of subsystem time (1.17→1.03s
  freMTPL2 sequence; 2.16→2.09s synthetic). The evaluations it removes are
  the bounded ladder's — bracket-fixed, hence φ-cache-aligned and cheap;
  the ones it keeps are fresh score passes, the expensive kind. Cross-
  populating the value cache from score evaluations adds nothing visible.
  Rejected: complexity and a new fallback path for ~1% of fit.
- **Loosening `xatol` (1e-9 → 1e-6 → 1e-4)**: identical optima (≤ 4e−16;
  the brentq polish pins the root regardless) and *slower* — 1.01→1.03→1.11s
  freMTPL2, 2.18→2.3s synthetic. A looser ladder stops earlier but leaves
  the optimum farther from the fixed ±6.4e−8 polish bracket, forcing
  bracket expansions paid in fresh score passes, and degrades the
  cross-call cache alignment. The seeded hypothesis that 1e-9 is wastefully
  tight is wrong in this architecture: 1e-9 is accidentally optimal.
- The per-call anatomy (16-eval ladder + 9-eval polish/curvature) is
  therefore already within ~5% of what any evaluation-scheduling change can
  reach; the real floor is the `wright_bessel` pass itself (§4).

### 3.3 Tried and rejected — mixed spline+categorical tabmat rung

The synthetic design falls between two accelerated centering rungs (§1), so
extending the raw-spline tabmat plan with a native `CategoricalMatrix`
block looked like the 49%-bucket fix. Measured on the exact design shape
(400k × [cr-20 sparse, cr-20 sparse, 8-level categorical], steady-state,
single thread): tabmat `SplitMatrix.sandwich` = **134ms** per weighted Gram
(matches the reference Gram to 3.6e−12), scipy sparse X'WX = 208ms,
per-column operator emulation = 222ms — while the in-tree raw-moment rung
the fit actually uses costs **~88ms** per build (5.96s / 68 builds in the
real fit). The "slow" rung is already the fastest of the four routes
measured for continuous splines; the tabmat extension would have *regressed*
it. Rejected on measurement. (freMTPL2 is 10× cheaper here only because its
banded covariates admit the packed level-compressed rung — that advantage
is the design's, not the code path's.)

## 4. What I would do next

1. **p = 1.5 Bessel-primary density.** At p = 1.5 (default and by far the
   most common power), Wright's Φ(1, 2; t) = I₁(2√t)/√t and Φ(1, 1; t) =
   I₀(2√t): the evaluator's existing `ive`-based p15 branch — today only an
   overflow fallback — could serve all rows, replacing the `wright_bessel`
   passes that are 14% of the 400k fit (75–85% of subsystem time) at an
   expected several-fold density speedup. Mathematically identical,
   numerically ~1e−14 different per row, so it moves fitted λ̂/φ̂ at the
   1e−10 level — it needs the frozen-fixture re-derivation pass that this
   branch is not the place to make.
2. **Banded/compressed Gram kernel for continuous splines** (native
   banded-×-banded cross-Gram; the 88ms/build raw-moment rung is ~3–9× off
   an nnz-touch floor at 400k×47), or simply document `discrete=True` as
   the intended path for continuous covariates at scale (measured 4.2×
   elsewhere at nil loss on banded-enough data).
3. **Design build (~16% one-shot) and finalize/W-correction (~15%)** at full
   size, if single-fit latency (not throughput) ever matters — neither was
   examined beyond attribution here.

## 5. Suite

Full suite on the optimised tree (`PYTHONPATH=src … -m pytest`, serial —
this venv has no xdist): **7890 passed, 155 skipped, 0 failed** in 10:56,
exit 0. The zero-row shortcut is bitwise-identical by test, so no fixture
moved.
