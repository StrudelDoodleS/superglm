# Research sweep — consolidated findings (5 tracks)

Source reports: `research-array-methods.md` (track 1), `research-smoothing-params.md` (2),
`research-sparse-factor.md` (3), `research-subsampling.md` (4), `research-screening-kernels.md` (5).
All measured claims below were benchmarked or numerically verified by the reporting track.

---

## 1. Cross-track conflicts, resolved

| Conflict | Resolution |
|---|---|
| **int32 bin indices**: track 5 measured **1.41×**; track 1 measured **slower** | Track 5 benchmarked a **hand-written numba kernel**; track 1 benchmarked **`np.bincount`, which upcasts internally**. Both correct. **Narrow dtypes pay ONLY inside first-party kernels — worthless as a NumPy-level dtype change.** Sequencing: fold dtype narrowing into kernel work, never ship it standalone. |
| **Threading multiplier**: audit §H.2 assumed 4-8× on accumulation; track 5 measured **1.3× and flat** | Accumulation is **memory-bandwidth bound** (one core already saturates ~27 GB/s). But the **compute-bound block/group axis gives 6-10×** (track 5: 6-9× on per-group Hessians; track 1: 9.9× on Gram blocks). **Thread the block axis, not the observation axis.** Audit §H.2's billion-row estimate corrects from ~39 s to **~55 s**. |
| **EFS vs AI-REML**: track 2 championed EFS/SOP; track 3 says AI-REML is strictly better | **AI-REML primary.** AI approximates only the *Hessian* — the gradient stays exact, so the fixed point is the **exact REML optimum**. EFS/SOP make the PQL simplification and change the estimator. AI preserves the mgcv-parity claim by construction; EFS does not. |
| **Does anything fix the G² pass problem?** | **No — confirmed by both tracks 1 and 5, and by Li & Wood themselves.** The literature keeps one accumulation per term-pair. The win is in **not needing most pairs** (row-tensor identity), not in fusing them. |

---

## 2. Ranked findings — the ones that change the plan

### ⭐⭐ A. Leverage-diagonal W-correction gradient — flat in q, exact (track 2)
`tr(H⁻¹X'diag(a_j)X) = Σₖ a_j[k]·hₖ`, `h = diag(XH⁻¹X')` computed once. **O(q·n·p²) → O(np² + qn).**
Verified 1.8e-15. Measured: q=25 **6.2×**, q=40 **9.8×**, **cost flat in q**.
Requires also removing the W-correction from the Hessian — which is **parity-safe by construction** because
`direct.py:334-346` tests convergence on the *gradient*. Combined: **172 s → ~32-40 s**; q=40 from >520 s
timeout to well under a minute. **Supersedes RFC-2**, which only claimed a better constant.
Memory caveat: `M = XL^{-T}` is 1.2 GB at MTPL2 n=678k — **chunk over rows**.

### ⭐⭐ B. Row-tensor Gram blocks in bin space — 77-352× on tensor blocks (track 1)
`(A′WA) = G(Ã₁)′W̄G(Ã₂)` where W̄ is the 2-D weighted histogram already built and `G(Ãⱼ)` are weight-independent
small marginals precomputed once per fit. Verified ≤5e-15. **Never touches an n×p array.** Off-diagonal
tensor×main-effect blocks reuse the same W̄ **free**. Deletes the `(n,p₁p₂)` full-density row-Kronecker loop
(`interaction.py:774-791`) and the retained `B_joint` outright.
End-to-end Gram stage: **2523 ms → 216 ms (11.7×), → 65 ms threaded (39×)**.
Crossover: wins iff **m₁m₂ ≲ n**; cap tensor-marginal bins at ~500-1000 so W̄ stays L3-resident.
**Grid-vs-scattered SETTLED:** GLAM *proper* needs a lattice — do not pursue; joint gridding is explicitly
rejected by Li & Wood (discretisation error exceeds statistical error). **But the row-tensor identity is
grid-free** — marginal binning makes W small a different way.
*Novelty caveat:* a combination of Currie et al. + Li & Wood not published as such; mgcv does not appear to do
it. Opportunity, and a reason to validate carefully.

### ⭐⭐ C. AI-REML — exact gradient, approximate Hessian (track 3)
`gⱼ = Sⱼβ̂`, `hⱼ = H⁻¹gⱼ` (**q solves total**), `AIᵢⱼ = gᵢ'hⱼ`. Drop `+½tr(H⁻¹SᵢH⁻¹Sⱼ)`.
AIREMLF90: **5-15 rounds vs REMLF90's 50-300**. **No published instance applied to GAM smoothing parameters —
a genuine gap.** Non-PD updates handled by blending with an EM step. Composes with A (both attack the same
Hessian cost from different sides) and with the animal-breeding architecture below.

### ⭐ D. The dense-inverse root cause is sharper than RFC-7 said (track 3)
`_safe_decompose_H` (`solvers/irls_direct.py:235-350`) does `cho_solve((L,True), np.eye(p))` — an **explicit
dense p×p inverse on essentially every PIRLS/Newton/line-search iteration** — and `inference/covariance.py`
builds a **second** one by a different route. The stale design comment at `irls_direct.py:8` reads
*"p is ~50-80… making the p×p solve trivially fast."* **That assumption is the actual root of the large-p
ceiling.**

### ⭐ E. Block-axis threading — 6-10×, bit-deterministic by construction (tracks 1 & 5, independent)
Each thread owns a **disjoint output block ⇒ no reduction ⇒ bitwise reproducible**. Li & Wood §4 recommend
exactly this; process blocks in decreasing cost; guard `n_threads ≤ n_blocks` (measured degradation past
4 threads at 40 groups).
**⚠ MEASURED: numba's automatic `prange` reduction is NOT bit-reproducible across thread counts**
(~30× machine epsilon). Fixed-chunk privatization with `NCHUNKS` a **compile-time constant** IS. Forbid
`fastmath=True` in the numerical core. **A thread-count-invariance CI test must land BEFORE any threading
work.**

### ⭐ F. Two one-line-ish fixes in already-shipped EFS (track 2)
1. **Wrong update.** superglm computes `λ* = r/(b+g)` — the *"accelerated EM"* update Wood & Fasiolo prove
   (their Thm 2) takes a **strictly shorter step** than their `λ* = φ(r−λg)/b`. Same fixed point ⇒ estimates
   unaffected. Measured **49→25 and 68→42 iterations (~1.6-2×)**.
2. **⚠ Correctness bug.** `efs.py:251` / `runner.py:233` hardcode `tr(S_λ⁻Sⱼ) = rⱼ/λⱼ` — **valid only for
   non-overlapping penalties**, so silently wrong under **`select=True` double penalties and tensor `ti()`
   terms**. Protected-semantics territory. **Mandatory prerequisite to any EFS/AI resurrection.**
Also: EFS is already shipped and merely unwired (reachable only on λ₁>0 and SCOP paths) — **wiring is a far
smaller change than the audit assumed.**

### ⭐ G. λ̂ is not scale-free in n; and mgcv's `samfrac` doesn't do what the audit claimed (track 4)
Measured **d log λ̂ / d log n = 0.43**. Raw transplant of λ̂ from a 20% subsample is still **1.03 SE** off;
with the `(n/m)^α̂` rescale, 5% gets inside **0.28 SE**. **`samfrac` carries only `coefficients`, never `sp`,
runs at loose tolerance, and is skipped entirely when `discrete=TRUE`.**
**Best version (zero contract change):** warm-start λ from a rescaled subsample fit, then converge on full
data. Because Fellner–Schall is a fixed point, **the limit is start-independent** ⇒ final λ is a genuine
full-data REML estimate. **~2-2.5×.** Theory: Sun, Zhong & Ma (2021, Biometrika 108(1):149-166) — ASP-A
estimates the exponent empirically.
**Frozen-λ (opt-in only) must be refused under `select=True`** — λ selection *is* term selection under the
double penalty, so a term zeroed on 5% of the book is a model-structure decision made on a subsample.
**Also: use design-preserving stratified sampling for the λ subsample; OSMAC-style residual-driven optimal
subsampling deliberately distorts the design distribution the REML criterion integrates over.**

### ⭐ H. Convergence criterion is statistically meaningless (track 4) — free win
Converging λ to 1e-7 when risk is flat to a **12× change in λ** (MISE moves only 20-35%) is wasted work. Stop
on **EDF change < ~0.01/term** or linear-predictor movement below a fraction of its SE. No estimator change.
**Likely 2-5 passes saved on its own.**

### I. Sparse Cholesky + Takahashi + AI = the verified production architecture (track 3)
Meyer verbatim: the AI-REML gradient *"require[s] selected elements of C⁻¹… feasible due to the sparse matrix
inversion method of Takahashi et al. (1973)."* ASReml/BLUPF90/WOMBAT/DMU all do this. **Takahashi and AI are
complements — the first-order trace does not cancel under AI.**
**Smith (1995) reverse-mode Cholesky AD** may be cheaper for the gradient: all q traces at O(1) factorizations,
~2× one likelihood evaluation. **LMMsolver (Boer 2023) uses it: mgcv 600 s → 1 s, 38 min → 30 s** — the
closest published prior art. Recommended split: **Smith AD for gradient traces, Takahashi for `diag(H⁻¹)`**.
**Practicalities:** scikit-sparse has **no** selected inversion (PR open, unmerged) — **port Davis's
`sparseinv`, ~200 lines, BSD-3**, which also resolves the GPL concern. **Gate the whole sparse investment on a
measured `nnz(L_H)` fill-in test** — MSSM found sparse Cholesky *slower than dense* when H wasn't genuinely
sparse, and Newton by-products are *"inherently dense."*
**lme4's derivative-free escape is closed to superglm**: lme4 profiles β,σ out and needs only `log|L_θ|`
because it **never reports penalised EDF or per-coefficient SE**. superglm does. Symbolic phase is
θ-independent ⇒ **symbolic analysis once, numeric refactorization per iteration.**

### J. Li & Wood (2020) Algs 0-3 + two free wins (track 1)
Alg 0 = the bypassed bincount Gram; Alg 1 = what superglm does; **Alg 2/3 = the missing fallback for
n < m_A·m_B**. Reported **30× on X′WX** (7024 s → 230 s). Two free items: their loop order is tuned for
**column-major** and they say to reverse it for row-major (**NumPy is C-order — check the loop nests**); and
**BLAS quality alone was 10×** (reference 126.3 min vs OpenBLAS 12.3 min) — **audit which BLAS this path hits.**

### K. Interaction discovery falls out of B (my derivation, unverified — check before building)
FAST (Lou, Caruana, Gehrke & Hooker, KDD 2013; InterpretML's EBM) ranks candidate pairs from **2-D residual
histograms** — the same object as W̄. The score for a candidate tensor is `U = Ã₁′R̄Ã₂` with
`R̄ₚq = Σ wᵢuᵢ` (residual-weighted 2-D histogram), and the efficient information is built from the row-tensor
Gram + the free cross blocks + the already-available `(X′WX)⁻¹`. So **the exact penalized score statistic for
every candidate pair costs one extra accumulator per histogram cell — no extra data pass, no n×p array.**
At the measured 1.27 ms/histogram, **all 190 pairs of a 20-term model screen in ~0.25 s at n=1e6.**
Beats FAST's piecewise-constant RSS heuristic (correct variance function and exposure weights) at the same
cost. **Would replace `benchmarks/superbooster_interaction_challenger.py`**, which currently uses an XGBoost
proxy on a 30k subsample outside the library.
**Caveats:** post-selection inference (screening invalidates naive p-values — needs held-out validation);
score test is *local*; must penalise-adjust or high-k tensors win spuriously; bin resolution caps detectability.
**Strategic: mgcv has no automated pairwise interaction search — this is superglm exceeding mgcv on
capability, enabled by the speed work.**

---

## 3. Rejected, with evidence

| Idea | Verdict |
|---|---|
| **Fused single-pass all-pairs Gram kernel** | **Independently reproduced 3-8.5× SLOWER** (tracks 1 & 5). Panel-blocking (the textbook fix) never beats per-pair, degrading monotonically (0.19-0.70×). Bandwidth-bound; fusion blows the histogram working set past L2/L3. |
| **GLAM proper / joint gridding** | Requires a complete lattice. Li & Wood explicitly reject joint discretisation: *"the grid then has to become coarser… errors from discretisation rapidly exceed the statistical error."* Only the grid-free row-tensor *identity* transfers. |
| **GAP safe screening** | Measured **improvement factor 0.98-1.00 — no speedup**. Near-identical screened set to the strong rule, but computing the safe region costs what it saves. adelie evaluated SAFE/strong/EDPP and shipped **only strong**. |
| **Stochastic trace/log-det (Hutchinson, Hutch++, XTrace, SLQ)** | Wrong axis at this scale: `tr(V S_r)` with `rank(S_r)=k_r ≪ p` is **already exact in k_r solves**; every "matvec" needs the very factorisation that makes the trace free; and noise poisons the REML gradient and breaks reproducibility. Crossover vs dense ≈ p 2-4×10⁴; vs sparse may never arrive. |
| **AD / implicit differentiation for the W(ρ) correction** | superglm's analytic form **IS already the IFT solution** (`w_derivatives.py:253-254` = Lorraine et al. Thm 1). The ML cost story doesn't transfer: superglm's objective is a **log-determinant** whose gradient is a **trace per j**, not one vector solve. Useful only as an implementation aid for painful higher derivatives. |
| **Coresets for Poisson-log** | **Theorem, not a difficulty**: Lie & Munteanu (arXiv:2410.22872) prove an **Ω(n) lower bound**; *"subsampling for the log-link is not possible with multiplicative (1±ε) error guarantees"*, extending to arbitrary data reduction up to log(n). |
| **Sketching / RandNLA** | The win is avoiding O(mn²) dense QR, which superglm already avoids. **Forming the sketch costs a full data pass — the scarce resource.** |
| **Leverage-score sampling** | Ma, Mahoney & Yu: neither leverage nor uniform dominates; superseded by OSMAC's shrinkage. Loses to mV/mVc for Poisson in Ai et al.'s own benchmarks. |
| **HDFE alternating projections for REML traces** | **Near-proof**: Kline, Saggio & Sølvsten needed **8 hours on 32 cores** doing n separate PCG solves for what selected inversion gives directly. `lfe` gives FE SEs only by bootstrapping; `reghdfe` not at all; Somaini–Wolak cannot pass two dimensions. Complements, never replaces, the selected-inverse route. |
| **Rügamer (2024) factorization-machine tensors** | Rank-F approximation ⇒ **breaks mgcv parity by construction**. A different estimator, not a faster one. |
| **Histogram privatization/replication for speed** | Measured best case 1.13×, mostly 0.07-0.67×. (Still adopt fixed-chunk privatization for *determinism*, not speed.) |
| **`w_correction_order=2` at large q** | O(q²np²) — 325 Grams/iteration at q=25. Hard-gate it. |
| **Array structure to speed up BCD** | glamlasso authors: *"it is not obvious how to exploit the array structure to reduce the computational complexity."* Array tricks help Gram formation only. |
| **SpGEMM/GPU sparse kernels** | Optimise irregular symbolic structure with unknown output patterns; superglm's accumulation has dense, known-shape, cache-resident output. Li & Wood's own sparse-hash Alg 4 is gated behind p>15 for this reason. |

---

## 4. Revised sequencing (deltas to the audit's tranches)

**Tranche 1 (correctness & credibility)** — add:
- EFS Wood–Fasiolo update fix (one line, ~2× iterations) **and** the overlapping-penalty correctness bug
  (mandatory before any EFS/AI resurrection).
- **RFC-6a spike now benchmarks AI-REML, not just EFS** — AI keeps the gradient exact, so it is the
  parity-preserving option.
- Convergence criterion → EDF/linear-predictor (free, 2-5 passes).
- **Thread-count-invariance CI test** (before any threading work).

**Tranche 2 (large-n)** — RFC-1 still first; then:
- **Row-tensor Gram** becomes a headline item (sized by actual tensor usage — unmeasured in the profiles).
- Block-axis threading (deterministic) **replaces** "add prange to accumulation kernels".
- Narrow dtypes fold **into** kernel work, never standalone.
- Li & Wood Alg 2/3 fallback + row-major loop-order fix + BLAS audit.

**Tranche 3 (exact-REML performance)** — substantially revised:
- **Leverage-diagonal gradient + drop W-correction from Hessian replaces RFC-2** (flat in q, not a better
  constant).
- **AI-REML** added.
- RFC-7 re-aimed at the sharper root cause: kill `cho_solve(L, np.eye(p))` in `_safe_decompose_H` and the
  second inverse in `covariance.py`; fix the stale `p is ~50-80` comment.

**Tranche 4 (large-p structure)** — gate on a measured `nnz(L_H)` fill-in test, then:
sparse Cholesky (symbolic once, numeric per iteration) + Smith-1995 AD for gradient traces + Takahashi
(`sparseinv`, BSD-3) for `diag(H⁻¹)`.

**New parallel design track:** interaction discovery via score statistics on the same histograms (K).
