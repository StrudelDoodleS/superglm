# Research track 5: screening, trace estimators, kernel engineering — findings

Includes verified micro-benchmarks run on this machine (numba 0.64.0, 32 cores) + source inspection of
`adelie` (Yang & Hastie's group-lasso solver, current speed leader).

## ⚠ Correction to audit §H.2: threading is NOT a 4-8× multiplier on the accumulation kernel

**Measured:** weighted histogram, n=40M, 64 bins — serial 30.5 ms; fixed-chunk parallel 22-24 ms at 4/8/16
threads (**1.28-1.37×, flat in thread count**, any chunk count 32→1024). The kernel moves 640 MB doing 1 flop
per 16 bytes; **one core already saturates most memory bandwidth (~27 GB/s measured on this box)**.

Consequence for the §H.2 billion-row estimate: at 140 bytes/row/iter × 1e9 × 14 passes ≈ 1.96 TB, at ~28 GB/s
single-thread → ~70 s, at ~35 GB/s threaded → **~55 s, not 39 s**. Order of magnitude survives; the "saturated
multi-core" row was optimistic. (Bandwidth is machine-dependent — a many-channel server box would do better —
but the *structural* point stands: threads don't fix a bandwidth-bound kernel.)

**Where threads DO pay: the compute-bound group axis.** Measured on a superglm-shaped problem (40 groups ×
9 cols, n=200k, weighted Gram): 697 ms → 130 ms (2 thr) → 74 ms (4 thr) = **6-9×**, and **bit-identical at
every thread count** because each group's Gram is accumulated entirely by one thread. Note degradation past
4 threads with only 40 groups — adelie's `n_threads <= n_groups` guard is necessary, not decorative.

**Bytes, not threads, is the accumulation lever — measured:**
| idx dtype | time | throughput | speedup |
|---|---|---|---|
| int64 | 24.2 ms | 26.5 GB/s | 1.00× |
| int32 | 17.1 ms | 28.0 GB/s | **1.41×** |
| int16 | 14.2 ms | 28.3 GB/s | **1.70×** |

## ⚠ Determinism: numba's automatic `prange` reduction is NOT bit-reproducible (measured)

```
scalar prange sum, n=4e6:   threads=2 → rel diff 5.09e-15   (bit-identical: False)
                            threads=4 → 1.88e-16            (False)
                            threads=8 → 6.22e-15            (False)
weighted histogram, numba automatic array reduction: False at 2/4/8
weighted histogram, FIXED-CHUNK privatization (64 chunks): True at 1/2/4/8
```
Mechanism: numba privatizes per *thread* and merges in thread order, so the summation order is a function of
`NUMBA_NUM_THREADS`. For a pricing library this is a latent defect — same model, 4-core CI runner vs 32-core
prod box, differs in the last bits; after ~20 PIRLS iterations through a nonlinear link that can surface at
~1e-10 in a premium.

**Recommended design (priority order):**
1. **Prefer partition-free parallelism** — parallelise over an axis where each thread owns a whole independent
   output (groups, λ-path points, bootstrap replicates). No reduction ⇒ bit-identical by construction.
2. **When a reduction is unavoidable, fixed-chunk privatization** with `NCHUNKS` a **compile-time constant,
   never `get_num_threads()`** — partition is a pure function of `n` and the constant. Pad rows to a cache
   line. Merge with `buf.sum(axis=0)` in fixed chunk order. Keep `set_parallel_chunksize(0)` (static).
3. **Forbid `fastmath=True` in the numerical core** — licenses reassociation, breaks reproducibility even
   single-threaded. Pin/assert the threading layer (`numba.threading_layer()`); tbb/omp/workqueue schedule
   differently.
4. **CI test: fit at `NUMBA_NUM_THREADS ∈ {1,2,4,8}`, assert `np.array_equal` on coefficients.** ~15 lines.
   **Must land BEFORE any threading work, not after.** Arguably worth more than any single speedup.

Atomics: reject (non-deterministic *and* contention-bound at few bins). Sort-based: only wins at huge bin
counts. Demmel & Nguyen reproducible summation / ReproBLAS is the heavy hammer if distributed fitting ever
needs partition-independent reproducibility — not needed for shared memory.

## Verdict: group screening — real but rank it #4, and use the STRONG rule, not GAP safe

**Economics favour superglm**: the test costs O(n·p_g) (one gradient block) while skipping O(n·p_g²) Hessian +
O(p_g³) eigensystem — the test is ~p_g× cheaper than what it saves, much better leverage than plain lasso.

**But:** achievable screen fraction is regime-limited. Feser & Evangelou (ICML 2025, arXiv:2405.17094) report
79-90% screened / 5-20× — on **synthetic data with 80% pure-noise groups**. An insurance GAM with curated
rating factors is not 80% noise; at the CV-optimal λ expect 40-90% active. Honest expectation: **85-95%
screened near λ_max, 30-60% at ship-λ, ~0% at the path bottom; ~2-4× over a full 100-point path**, mostly from
the top half where you weren't spending time anyway. **~1.1-2.5× at the λ you actually ship.**

**DO NOT implement GAP safe** (Ndiaye, Fercoq, Gramfort & Salmon, JMLR 18(128), 2017). Feser & Evangelou
measured GAP safe and the strong rule reaching near-identical screened sets (input proportion 0.204 vs 0.209)
but GAP safe's **improvement factor 0.98-1.00 — no speedup at all**: computing the safe region (dual norm over
all groups + duality gap + per-group ‖X_g‖₂) costs as much as it saves. Their words: *"the cost of calculating
safe regions appears to nullify any gain in dimensionality reduction."* adelie evaluated SAFE/strong/EDPP and
shipped **only the strong rule**.

**Sequential strong rule** (Tibshirani et al., JRSS-B 74(2):245-266, 2012; group form = adelie eq. 18):
discard g if `‖X_gᵀ∇ℓ(η̃)‖₂ < α ω_g (2λ − λ̃)`. Heuristic ⇒ needs a post-convergence KKT check
(`‖X_gᵀ∇ℓ(η)‖₂ ≤ λαω_g` for g ∉ S), costing one gradient pass O(n·p) — amortises over the inner solve.

**superglm-specific caveats (important):**
- **The test must be evaluated in the SSP basis** the group penalty acts in (`T_g^{-ᵀ}X_gᵀ∇ℓ`), not the raw
  block. Getting this wrong silently invalidates the rule.
- **The smoothing penalty is differentiable ⇒ belongs in the smooth part**: gradient becomes
  `∇ℓ(η) + 2λ_g S_g β_g`, and λ_max must be recomputed accordingly. This is group-*elastic-net* structure
  (adelie's `α`), not pure group lasso.
- **`select=True` double-penalty is NOT screenable this way** — null-space penalty, not a non-smooth group
  norm, no KKT discard test. Keep the mechanisms separate (as CLAUDE.md already requires).

## Verdict: stochastic trace/log-det — no crossover in this niche; modern estimators don't rescue it

- vs dense exact: crossover ~p ≈ 2-4×10⁴. One Cholesky is p³/3, log-det then free, each `tr(V S_r)` is
  `‖L⁻¹P D_r‖_F²` at p²·rank(S_r). Total ≈1.3p³ → at p=10⁴ that's **~2 s multicore**. At p=10⁵: 4 h *and*
  80 GB dense — dies on memory before flops.
- vs sparse exact: moves to p ≳ 10⁵-10⁶, and for banded/tensor GAM Hessians with fill-reducing ordering may
  never arrive. **MSSM (Krause, Borst & van Rij 2025, arXiv:2506.13132)** fits *tens of thousands* of
  coefficients with sparse pivoted Cholesky + exactly the `B_r = L⁻¹P D_r`, `tr = Σ B_r²` trick and **zero**
  stochastic estimation.
- **Killer argument:** you need `tr(V S_r)` with `rank(S_r) = k_r ≪ p`. **`V S_r` is rank-k_r — its trace is
  exact in k_r solves.** Stochastic estimators exist to avoid d probes on a d×d matrix; the exact answer
  already costs k_r ≪ p. Hutch++ would be **strictly worse**.
- **Second killer:** every matvec with `V = H⁻¹` requires a solve. Hutch++/XTrace optimise *matvec count*; the
  cost driver is *the factorisation that makes a matvec possible*. Wrong axis.
- **Third:** trace noise enters the REML gradient (1e-3 relative error makes a Newton step meaningless) and is
  not reproducible run-to-run.

For the record if p ≳ 10⁵ ever arrives: Hutch++ (Meyer/Musco/Musco/Woodruff SOSA 2021, arXiv:2010.09649,
O(1/ε) vs O(1/ε²) matvecs); **XTrace/XNysTrace (Epperly, Tropp & Webber, SIMAX 45(1):1-23, 2024,
arXiv:2301.07825** — exchangeability principle, 240×/2400× more accurate than Hutch++ at m=40 on decaying
spectra, plus a free a-posteriori error estimate); SLQ (Ubaru, Chen & Saad, SIMAX 38(4), 2017);
**Cortinovis & Toni arXiv:2601.05778 (Jan 2026)** preconditioned one-sample SLQ — state of the art for
log-det specifically, the right tool for the narrow case of `log|XᵀWX+S|` alone at p ≳ 10⁵ with bad fill.
Note: XTrace's own authors say all variance-reduced estimators degenerate to Hutchinson on flat spectra, and
penalised GAM Hessians are not strongly low-rank.

## Verdict: no eigensystem update trick exists — and the state of the art rebuilds too

Inspected adelie source. `solver_glm_naive.hpp`: *"in GLM fitting, the three screen_* inputs are modified at
every IRLS loop"* and at each proximal-Newton iteration it *"repopulate[s] every entry using the new weights"*.
**The fastest published group-lasso GLM solver fully rebuilds `X_gᵀWX_g` and its eigendecomposition every IRLS
iteration.** Why: `W = diag(w)` changing in all entries is a rank-n perturbation, not rank-1, so
Bunch–Nielsen–Sorensen / Gu–Eisenstat secular-equation updating does not apply.

**So the superglm fix is RESTRICT + PARALLELISE, not update:** rebuild only for g ∈ screen set; iterate mostly
over the active subset; `omp_parallel_for` over the per-group `cov → rankUpdate → SelfAdjointEigenSolver`.

Other adelie findings worth stealing:
- **Newton-ABS block update** (§3.1-3.2): the group block subproblem's secular equation solved by Newton with
  adaptive bisection, **quadratic convergence**. They note isotropic-majorisation approaches (Meier et al.,
  Beck–Teboulle) *"converge very slowly either when X_gᵀX_g is near singular or the dimension of the block
  increases"* — exactly the spline-block regime. Worth swapping if superglm uses ISTA/FISTA inner loops.
- **No line search in the proximal quasi-Newton outer loop**: *"we do not perform the line search since it
  will significantly impact the overall runtime and it does not seem necessary in practice thanks to the
  warm-starts... usually only requiring 1 to 5 iterations."*
- Breheny & Huang (Stat. Comput. 25(2):173-187, 2015, `grpreg`) SVD-orthonormalisation per group → closed-form
  soft-thresholding forever after. **Rule out deliberately**: changes penalty semantics (penalises each group's
  contribution to the linear predictor, not coefficients) and doesn't survive changing W either.
- **Rank deficiency: use LAPACK `dpstrf`** (Cholesky with complete pivoting + rank detection, O(p³)), exposed
  as `scipy.linalg.lapack.dpstrf`. Kills the measured O(p⁴) / 660×-at-p=400 loop. MSSM §3.1 discusses the
  sparsity-vs-stability pivoting trade-off and falls back to sparsity-preserving QR of `[X; E_λᵀ]` with
  Heath's method when the condition estimate looks bad.
- **Li & Wood (2019), Stat. Comput. 30:19-25** — marginal discrete crossproducts, **30× reduction in
  crossproduct time** on their Black Smoke model. Composes with int32: it makes the bin-index pass dominant,
  which is exactly what narrow dtypes accelerate.

## Suggested sequencing from this track

1. Cache + restrict per-group Hessian/eigensystem to screen/active set; active-set on by default with the
   two-level (screen ⊃ active) discipline. **This is the measured defect, not missing screening.**
2. **Thread-count-invariance CI test — lands BEFORE any threading work.**
3. `prange` over the **group** axis with a `min(n_threads, n_groups)` guard. 6-9×, deterministic.
4. int32 (or int16) bin indices. 1.41-1.70×, trivial, exact.
5. Sequential strong rule + KKT check (SSP basis; smoothing penalty in the smooth part).
6. `dpstrf` for rank deficiency.
