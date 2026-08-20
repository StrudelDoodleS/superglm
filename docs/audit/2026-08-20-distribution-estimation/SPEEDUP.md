# Recovering the v0.29.0 Tweedie REML regression — measurements

Follows `PROFILE.md`. Branch `perf/distribution-estimation-profiling`, base
`99af8f60` (v0.29.0 plus the landed zero-row deviance shortcut). Every timed
fit alone under `flock /tmp/superglm-bench.lock`, one fit per lock hold, all
six thread pools pinned to 1 (`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/BLIS`), arms
interleaved A,B,A,B,…, medians with min–max spread, CPU reported beside wall
(CPU/wall 0.996–0.999 in every cell below, so the pins held). Each arm is its
own git worktree with an `--expect-tree` import assertion, because this venv is
a shared editable install pointing at a third checkout. Refs #339.

## 0. Headline

The workload is a **synthetic reproduction** of the reported burn-cost shape:
67,000 rows, Tweedie(1.5) log link, sample weights and an offset, five
ordered-categorical banded axes (13/15/17/20/24 levels, `Spline(kind="cr")`)
plus four plain categoricals (3/8/12/23 levels), 82.5 % zeros, 11,708 positive
rows, gamma-distributed positives. Levels, effects, exposure and response are
all drawn from a fixed seed; no real data is read. It is in
`benchmarks/profile_tweedie_reml_fit.py` as `--dataset burn-cost`.

**5-fold cross-validation — the reported workload's shape** (3 reps/arm):

| arm | wall s | CPU s | vs v0.28.0 |
|---|---|---|---|
| v0.28.0 | 6.664 [6.613–7.302] | 6.651 | — |
| v0.29.0 + landed shortcut | 10.218 [10.105–10.913] | 10.202 | **+53.3 %** |
| this branch | 7.233 [7.041–7.233] | 7.219 | **+8.5 %** |

**1.413× on the CV, 84.0 % of the regression recovered**, sign-stable in every
rep (1.413/1.435/1.509). The reported real workload showed +49.9 %; this
synthetic reproduces +53.3 %, which is the evidence that the shape transfers.

**Single 67k fit** (6 reps/arm):

| arm | wall s | CPU s | vs v0.28.0 |
|---|---|---|---|
| v0.28.0 | 1.491 [1.460–1.543] | 1.488 | — |
| v0.29.0 + landed shortcut | 2.558 [2.538–2.581] | 2.555 | **+71.5 %** |
| this branch | 1.786 [1.720–1.833] | 1.784 | **+19.8 %** |

**1.432×**, sign-stable in all 6 reps (1.398–1.476).

**The exact scale profiler went from 30.5 % of the fit to 4.4 %.** What remains
of the gap to v0.28.0 is *not* the profiler — see §4.

## 1. Where the fit's time actually goes on this shape

Wall-clock timers (not cProfile, which inflates Python-heavy code and
understates numpy), one instrumented fit per tree:

| bucket | v0.28.0 | baseline v0.29.0 | this branch |
|---|---|---|---|
| `build_centered_system` (packed Gram) | 0.284 s (55 builds) | — | 0.378 s (79 builds) |
| `reml_w_correction` | 0.202 s (5 calls) | — | 0.335 s (8 calls) |
| design build (`dm_build_s`, one-shot) | 0.337 s | — | 0.368 s |
| `canonicalize_fitted_model` (one-shot) | 0.154 s | — | 0.156 s |
| **Tweedie scale profiler** | **0** | **0.815 s (30.5 %)** | **0.080 s (4.4 %)** |
| Tweedie unit deviance | 0.139 s (60 calls) | 0.142 s (168 calls) | 0.077 s (168 calls) |
| `_validate_categorical_levels` | 0.058 s | — | 0.058 s |
| `reml_optimizer_s` (outer loop total) | 0.778 s (5 iters) | — | 1.136 s (8 iters) |
| instrumented fit total | 1.427 s | 2.67 s | 1.821 s |

(Instrumented fits carry their own timer overhead and run a few percent slower
than the untimed A/B cells in §0; the decomposition is the point, not the level.)

Inside the profiler, one fit makes **17 `profile_tweedie_reml_scale` calls =
criterion evaluations + 1**, each fresh solve costing 17 density passes, with
7 of the 17 calls exact `(Dp, Mp)` repeats costing zero (`_carry_forward` hands
the accepted line-search trial to the next candidate). That confirms
PROFILE.md §1 on this shape and refutes the hypothesis that a banded/OC
specification runs 40–70 profile calls: it runs 17.

Regime census of `wright_bessel` arguments over a *real* fit (not a uniform
ladder): 90.47 % land in scipy's 18-term `x<=1` Taylor branch, 6.16 % in the
20-term `x<=2` branch, 2.79 % asymptotic/quadrature, 0.59 % above the
representability limit. This matches the analyst census (87 %) closely.

## 2. What landed

### 2.1 Stop rebuilding the phi cache the search already filled — `2549a6ed`

Two bitwise-identical savings, one commit.

**(a) finalize's terminal evaluations discarded the per-fit cache.** All three
`reml_laml_objective` calls in `model/reml_finalize.py` omitted
`tweedie_scale_data=`, so `reml/objective.py` constructed a fresh
`TweedieScaleProfileData` with empty memo dicts and re-solved a `(Dp, Mp)` the
search had already solved. `REMLResult` now carries the optimizer's prepared
state (a non-compared, non-repr field) and finalize hands it back.

Instrumented: `prepare_tweedie_reml_scale_data` objects per fit 2 → 1, fresh
density passes 177 → 167. It does not go to zero because the terminal PIRLS
refit recomputes `penalized_deviance` and lands one ulp away
(`5196478.9714364875` → `5196478.971436487`), so the polish re-solves; the
shared cache still absorbs the bracket-fixed ladder probes.

**(b) the score pass computed the saturated value and threw it away.**
`compute_score=True` does not touch `logpdf`, and `brentq` returns its last
evaluated point, so the criterion evaluation at the polished optimum lands on
exactly one of the score keys. Cross-filling the value cache from the score
pass makes it a hit. 167 → 158 fresh passes.

This contradicts PROFILE.md §3.2's "cross-populating the value cache from score
evaluations adds nothing visible" — that was measured inside the *warm-start*
variant, where the score points and the value point had been decoupled. In the
shipped architecture they coincide by construction.

A/B, 6 reps/arm: **2.568 s [2.504–3.137] → 2.432 s [2.398–2.493], 1.056×**,
sign-stable. Equivalence: λ̂, φ̂, edf, deviance, REML objective, REML iteration
count and prediction digest **JSON-identical** across arms and reps.

### 2.2 Closed-form saturated Tweedie(1.5) scale kernel — `8bc06703`

At p = 1.5 the Wright parameter `a = (2-p)/(p-1)` is exactly 1, and **DLMF
10.46.2** — `I_ν(z) = (z/2)^ν φ(1, ν+1; z²/4)`, fetched and quoted rather than
recalled — collapses Wright's function to a modified Bessel function:

    Φ(1,2;t) = I₁(2√t)/√t        Φ(1,1;t) = I₀(2√t)

This is the established special case, not a new derivation. Dunn & Smyth
(2005), *Statist. Comput.* 15(4):267–280, name p = 1.5 as Siegel's (1979)
noncentral χ² with zero degrees of freedom; their Fourier-inversion companion
(2008, *Statist. Comput.* 18:73–86, Table 2) uses the Bessel form as its
*reference truth* at p = 1.5. `wright_bessel` is the ecosystem default
(statsmodels' exact Tweedie `loglike_obs` calls it too) and it leaves the
p = 1.5 reduction on the table.

The REML scale profiler is the **saturated** case, which is simpler still:
`prepare_tweedie_reml_scale_data` passes `y_positive` as both `y` and `mu`, so
the unit deviance is identically zero and

    l_sat(φ) = C₀ − n₊·log φ + Σᵢ log i1e(Kᵢ/φ)
    T(φ)     = Σᵢ zᵢ (i0e(zᵢ) − i1e(zᵢ)) / i1e(zᵢ),     z = K/φ

with `K = 4w√y` and `C₀ = Σ(log 2 + log w − log(y)/2)` both fit-invariant.

Scoped to `TweedieScaleProfileData`. `_evaluate_tweedie_density` is untouched,
so the `n_saddlepoint`/`n_series` diagnostics, the branch-edge safeguard in the
φ profiler (`tweedie.py:1560`, which gates on `n_saddlepoint == 0`),
`tweedie_logpdf` and the p search all behave exactly as before, and every power
other than 1.5 keeps the current path bit-for-bit. The large-argument score
rule is copied verbatim from the density evaluator, so the change introduces no
score behaviour of its own; any non-finite intermediate falls through to the
general route, which keeps its `FloatingPointError` contract.

**Isolated**, n₊ = 11,708 (medians of 6, warm):

| | current | closed form | ratio |
|---|---|---|---|
| value pass | 2.685 ms | 0.407 ms | **6.6×** |
| score pass | 5.435 ms | 0.790 ms | **6.9×** |
| bare special function | `wright_bessel(1,2,t)` 2.415 ms | `i1e(z)` 0.387 ms | **6.2×** |

The glue is thin, not 23 %: the current value pass is 2.685 ms against 2.415 ms
of bare `wright_bessel`, so ~90 % of the subsystem is the special function.
Also measured: `ive(1,x)` 1.031 ms vs `i1e(x)` 0.387 ms — `ive` is the `dd->d`
AMOS entry point, `i1e` the `d->d` Cephes Chebyshev, **2.7× apart**, and they
agree only 25.0 % of the time bitwise (max relative difference 2.08e-15;
`ive(0,·)` vs `i0e(·)` 9.7e-16, 24.8 % bitwise).

**Accuracy, against mpmath at 45 digits.** Both routes agree with the exact
Wright series to ≤ 5.6e-16 relative in `log Φ(1,2;t)` over `t ∈ [6e-6, 6e4]`,
and the E[J] score ratio to ≤ 1.3e-15, with the Bessel form the better of the
two wherever they differ (e.g. 1.4e-16 vs 1.3e-15 at t = 148). scipy's own
header documents its asymptotic regime at "~1e-11 … down to ~1e-8 or 1e-7", and
`wright_bessel(1,2,t)` overflows to `+inf` above t ≈ 1.26e5, where the current
code pays a full evaluation, has it rejected, and then pays the `ive` fallback
as well.

**Equivalence, end to end.** `l_sat` agrees with the Wright route to ≤ 3.8e-13
relative over `log φ ∈ [-12, 12]` (49 points), `T` to ≤ 1.1e-9 at the far-field
extremes. Solved optima, three shapes × seven `(Dp, Mp)` pairs:

| shape | φ̂ rel | criterion rel | d(1/φ)/dDp rel |
|---|---|---|---|
| burn-cost 67k | 0.0 / 0.0 / 0.0 | ≤1.3e-16 | 0.0 … 1.7e-13 |
| freMTPL2-like 30k unit weights | 2.8e-16 / 4.8e-16 | 0.0 | 0.0 … 5.0e-14 |
| small heavy-tailed 4k | 0.0 / 0.0 | ≤1.6e-16 | 0.0 |

φ̂ is *identical* in five of seven solves. The largest residual, 1.7e-13 on the
derivative, is the documented 1e-3 central difference amplifying evaluation
noise 500× (`scale.py`'s own comment predicts exactly this); its only consumer
is the REML Newton Hessian.

Whole fit: **λ̂ ≤ 2.1e-14, φ̂ 1.7e-15, edf 2.6e-16, deviance 1.8e-16, REML
objective 1.8e-14, predictions ≤ 1.8e-15, REML iteration count unchanged.**
On the 5-fold CV: out-of-fold predictions ≤ 8.9e-16, per-fold λ̂ ≤ 2.1e-13,
φ̂ ≤ 2.0e-15, edf ≤ 2.6e-16, and per-fold **out-of-sample deviance identical to
12 decimal places in all five folds**.

**The frozen-fixture re-derivation cost that PROFILE.md §4.1 flagged as
blocking is measured at zero.** Full suite: 7890 passed, 155 skipped, 0 failed
— the same counts as the baseline. Not one fixture moved.

A/B, 6 reps/arm: **2.624 s [2.547–2.677] → 1.923 s [1.852–1.980], 1.364×**,
sign-stable in every rep.

### 2.3 Split the unit deviance on positions, not on a boolean mask — `de6eb621`

`_tweedie_positive_unit_deviance` partitioned zero from positive rows with a
boolean mask and then indexed with it four times — two gathers, two scatters,
plus two `np.any` scans and an invert, all over n. Each of those re-walks the
mask. Taking `np.flatnonzero` once and indexing with integer positions removes
the re-walks and folds both `np.any` scans into the `flatnonzero` that has to
run anyway.

The prior decomposition is the reason this was worth doing: **67.6 % of the
call was the partition** (0.931 ms of 1.377 ms at n = 67,000), against 0.32 ms
of actual arithmetic — much larger than the ~30 % predicted.

Isolated: 1.47 ms → 0.56 ms, and 2.5–2.7× on *every* branch (realistic μ,
δ → +∞, δ → −1, and the near-arm μ = y).

**Bitwise identical, tested rather than argued.** A blake2b digest of the
output over 105 cells — zero fraction ∈ {0, .25, .5, .83, .95, .999, 1} ×
p ∈ {1.05, 1.1, 1.5, 1.83, 1.99} × scale ∈ {1e-8, 1, 1e6} — plus hand-built
extremes spanning y/μ from 1e-320 to 1e300 matches the baseline tree **exactly**,
NaN pattern included.

A/B, 6 reps/arm: **2.001 s [1.881–2.548] → 1.843 s [1.738–2.075], 1.085×**,
sign-stable; whole-fit digest bitwise identical.

In-fit bucket: 0.142 s → 0.077 s (7.9 % → 4.2 % of the fit).

## 3. What did NOT pay — measured and declined

| candidate | measurement | verdict |
|---|---|---|
| Skip `wright_bessel` on rows that provably overflow (t ≥ 1.2566e5) | **0.59 %** of row-evaluations in a real fit, not the 3.9 % an artificial uniform log-φ ladder suggests. Worth ≲0.5 % of the subsystem | Declined; subsumed by §2.2 at p = 1.5 anyway |
| Single-vector bincount when `gram_rmatvec(W, W)` is aliased | **200 of 200** `gram_rmatvec` calls per fit ARE aliased, but the whole bucket is **0.85 % of the fit**; `_fused_bincount_2` 57.9 µs vs a single bincount 34.2 µs ⇒ projected saving **0.25 % of the fit** | Declined on measurement |
| `centered_signed_gram` → `_moments_prevalidated` | 40 calls × one `np.all(np.isfinite(W))` over 67k ≈ 3 ms (0.15 %), and it converts a loud `ValueError` on a pathological `dW/dη·dη/dρ` into a silent NaN Gram | Declined: the risk exceeds the gain |
| Newton root-find replacing the bounded ladder + brentq polish | Post-§2.2 the whole subsystem is **0.080 s = 4.4 %** of the fit, of which ~95 % is the bare `i1e` Chebyshev. Cutting ~10.4 passes/solve to ~7 saves ≈ 30 % of 4.4 % = **1.3 % of the fit** for a large change that moves the profile curvature | Declined: effort and Hessian risk exceed 1.3 % |
| Closed-form `φ₀ = Dp/(n₊ − M_p)` bracket seed | Same denominator: ≲5 ladder probes/solve out of a 4.4 % subsystem ⇒ ≲1.5 % | Declined |
| `ive` → `i1e`/`i0e` in the density evaluator's p15 fallback | Correct and 2.7× on that call, but after §2.2 **no measured path on this workload reaches it** (the fallback fires on 0.59 % of rows the REML profiler no longer evaluates); it would move results at 2e-15 for zero measured gain | Declined here; still the right change if the p-search or `tweedie_logpdf` is ever measured to matter |
| Route **all** p = 1.5 rows in `_evaluate_tweedie_density` to the Bessel branch | Strictly worse than §2.2 for the REML path (keeps ~45 array passes and ~12 scratch allocations), and it drives `diagnostics.n_saddlepoint` to 0 — which is a **gate**, not a report: `tweedie.py:1560` skips the branch-edge safeguard entirely when every root candidate has `n_saddlepoint == 0` | Declined: higher risk, no measured consumer |
| Batch the W-correction's per-λ signed Grams; fuse `packed_centered_gram_rhs`'s bincounts and histograms | The two biggest remaining buckets (0.335 s / 18.4 % and 0.378 s / 20.8 %), and both are **v0.28.0 costs too** — per build the optimised arm is 4.6 ms against v0.28.0's 5.2 ms; the extra total is purely the extra outer iterations. Sizing the analyst's model at this shape: ~9 % of the fit if the fusion delivers, needing a new numba multi-channel kernel, design-level caching of the code matrix and pair offsets, and a cells cap | Declined for this pass, sized for the next: the scatter target grows from ≤4.6 KB (L1-resident) to ~78 KB (L2), and that residency is unmeasured |
| `_validate_missing_only`'s per-element genexpr (938,014 calls, ~0.058 s) | Real, but `categorical.py`'s own comment argues at length that the narrow per-element test *is* the validation boundary and that short-circuiting on `pd.isna` widens the accepted surface | Declined: a documented correctness decision, not mine to reverse in a perf branch |
| `_compute_public_parity_diagnostics` re-scoring every term through the public path (0.156 s, 8.6 %) | Deliberate public/solver parity guard under `runtime_validation="auto"` | Declined: correctness guard, out of scope |

## 4. The residual +8.5 % is the landing point, not the profiler

v0.28.0 converges in **5** outer REML iterations on this shape; v0.29.0
converges in **8**. That is the corrected criterion landing on a different,
much heavier-smoothing solution (λ̂ 43/32/165/54/11 → 160/182/507/261/31303,
edf 63.9 → 55.7 per fold), which PROFILE.md §2 and the issue-339 verification
already established as the fix doing its job.

Decomposing the residual (instrumented single fits, v0.28.0 1.427 s vs this
branch 1.821 s):

- one-shot work (design build, canonicalize, cache priming) is essentially
  unchanged: 0.551 s → 0.592 s;
- `reml_optimizer_s` 0.778 s → 1.136 s, **+0.358 s**, of which only
  **0.080 s (22 %) is the scale profiler**; the other 0.278 s is 3 extra outer
  iterations, 24 extra centered-Gram builds and 3 extra W-corrections.

**Per outer REML iteration this branch is now cheaper than v0.28.0**:
1.136/8 = 0.142 s against 0.778/5 = 0.156 s, scale profiler included. The exact
criterion's per-evaluation cost is no longer the story; its iteration count is,
and that is not a cost defect.

## 5. Projection onto the reported workload

Stated as a projection, with its assumption named. The reported fold spent
2.08 s of 6.25 s (33.3 %) in the scale subsystem. Here the subsystem went from
0.815 s to 0.080 s, a factor of **0.098**. If it shrinks by the same factor
there (it should: the factor is set by the special function and the pass count,
both of which are per-positive-row and independent of the rest of the design):

    fold      6.25 s → 6.25 − 2.08 + 2.08 × 0.098 = 4.37 s
    5 folds  31.25 s → 21.9 s,  against v0.28.0's 20.85 s
    residual regression  +49.9 %  →  ≈ +5 %

The measured CV on the reproduction lands at +8.5 % residual from +53.3 %, so
+5 % to +9 % is the honest band.

## 6. Suite

Full suite on this branch, serial (this venv has no xdist), all pools pinned:

    7890 passed, 155 skipped, 0 failed — exit 0

Identical counts to the PROFILE.md baseline. No frozen fixture moved.

## 7. One correctness finding, reported not landed

`solvers/irls_direct.py:1073` writes `beta[groups[gi].sl] = gamma_eff` during
SCOP QP initialisation, *after* `beta_init` is copied and *before* the committed
state at `:1258` consumes `deviance=_deviance_init`. Any caller passing
`_deviance_init` for a fit with a SCOP group therefore gets a committed state
whose `deviance` belongs to a different `beta` than its `mu` — a wrong
line-search baseline at iteration 1. `_has_scop` is already resolved at `:893`,
so the guard is one line: `deviance=None if _has_scop else _deviance_init`.

It is **latent, not live**: the only caller passing `_deviance_init` is
`reml/discrete.py:600`, and SCOP fits route through `run_scop_efs_reml` /
`fit_fixed_scop_reml` rather than the cached-W discrete path. It is a
contract-level hazard, so it belongs in a correctness change with its own test
rather than in a performance branch.

## 8. Reproducing

    # one fit, three arms, interleaved
    flock /tmp/superglm-bench.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      BLIS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
      benchmarks/profile_tweedie_reml_fit.py --dataset burn-cost --rows 67000 \
      --expect-tree perf-profiling

Drop `--no-probes` to get the counters (profile calls, minimize/brentq nfev,
cache hit rates, density passes split by value/score context); add
`--cprofile out.pstats` for the call-stack dump.
