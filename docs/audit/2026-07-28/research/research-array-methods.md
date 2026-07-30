# Research track 1: array/Kronecker methods & large-data discrete GAMs — findings

**Headline: the highest-value idea in the whole research sweep, and it was on nobody's list.** The
Currie–Durbán–Eilers *row-tensor identity* transplanted into Li & Wood's *marginal* bin space. Derived,
verified exact (rel. err ≤5e-15 vs dense ground truth), benchmarked: **77-352× on tensor-product Gram blocks**,
**11.7× on the whole Gram stage** of a realistic insurance model, **39× with threading**.

## The grid-vs-scattered question — SETTLED, and it cuts both ways

The audit's hypothesis was "bin-space discretization manufactures the grid GLAM needs." That is **half right,
and the wrong half would have burned a quarter.**

1. **GLAM proper requires a complete lattice — do NOT pursue it.** Currie, Durbán & Eilers (2006, JRSS-B
   68:259-280) require X = X_d ⊗ … ⊗ X₁ with y = vec(Y) and n = ∏ nⱼ. The speedup comes precisely from the
   marginal basis being only nⱼ rows tall.
2. **Joint binning onto a grid — the classical remedy for scattered data (implemented in `glamlasso`) — is
   explicitly REJECTED by Li & Wood (2020):** *"The obvious approach is to discretize the covariates jointly
   onto a grid … but to maintain computational efficiency the grid then has to become coarser and coarser as
   the number of covariates increases, and the errors from discretisation rapidly exceed the statistical
   error."* Their contribution rests on the opposite: *"the covariate discretization is **marginal**. That is
   we do not rely on discretizing covariates jointly."* superglm's bin-index + per-term unique basis IS this
   marginal structure. A joint grid over 6 insurance rating dimensions is statistically indefensible.
3. **But the row-tensor IDENTITY is grid-free — and that is the transferable part.** The G-operator
   G(M) = (M ⊗ 1′) ∘ (1′ ⊗ M) is an algebraic identity about row-Kronecker structure; it never needed the
   lattice. The lattice only made W small. **Marginal binning makes W small a different way: it becomes the
   m₁×m₂ weighted 2-D histogram superglm already computes.**

## Idea 1 — Row-tensor Gram blocks in bin space ⭐ DO THIS FIRST

For tensor term A = A₁ ⊙ A₂ with marginal discretization Aⱼ(i,·) = Ãⱼ(kⱼ(i),·):

```
(A′WA)[(a,b),(c,d)] = [ G(Ã₁)′ W̄ G(Ã₂) ][(a,c),(b,d)]      W̄ₚq = Σ_{k₁(i)=p, k₂(i)=q} wᵢ
```

W̄ is the 2-D weighted histogram already being built. G(Ãⱼ) are m×p² row-tensors of the *small* marginals —
**weight-independent, so precomputed once per fit**. Everything after the histogram is one BLAS-3 sandwich
plus reshape/transpose.

**Why it wins: gathers, not flops.** Measured at n=1e6: 2-D weighted histogram **1.27 ms**; `Ã[k]` gather
n×9 **36.9 ms**; the BLAS-3 core **0.11 ms**. Per-pair block schemes must materialise n×p gathers to build
modulated weights; **the row-tensor scheme never touches an n×p array** — only n-length index and weight
vectors.

**Measured (all verified exact, rel. err ≤5e-15), n=1e6:**
| tensor block | naive row-Kron | per-pair hist | **row-tensor** | vs per-pair |
|---|---|---|---|---|
| te(6,6), m=100² | 316 ms | 87 ms | **1.1 ms** | **77×** |
| te(10,10), m=200² | 904 ms | 280 ms | **1.6 ms** | **180×** |
| te(20,20), m=500² | 4212 ms | 1447 ms | **4.1 ms** | **352×** |
| te(10,10), m=1000² | 889 ms | 287 ms | **8.5 ms** | **34×** |

**Off-diagonal blocks come free** — te(x₁,x₂)×s(x₁) reuses the *same* W̄ (verified 2.4e-15) via
(Ã₁ ⊙ B̃)′W̄Ã₂, where the row-Kronecker is between two tiny m₁-row marginals. One histogram per covariate pair
serves the te diagonal block, both te×main-effect blocks, and main×main.
**The centered-marginal density problem evaporates** — density of an m×p matrix is irrelevant. The audit's
`(n, p₁p₂)` full-density "sparse" matrix and its Python loop (`features/interaction.py:774-791`) are **deleted
outright**, and the retained `B_joint` with it. 3-way tensors work via one 3-D histogram + einsum (3.3e-15,
13.5× at m=60³).

**End-to-end, realistic model** (6 splines m=200 + 6 rating factors m=25 + 3 two-way interactions, n=1e6):
```
current  : 172 ms hists + 2352 ms row-Kron blocks = 2523 ms
proposed : 172 ms hists +   44 ms row-tensor      =  216 ms   → 11.7×
+threaded:  21 ms hists +   44 ms                 =   65 ms   → 39×
```

**Crossover rule (= exactly Li & Wood's stated condition for their Alg 1): row-tensor wins iff m₁m₂ ≲ n.**
- **W̄ must stay cache-resident**: histogram cost flat at 1.1-1.7 ms while m₁m₂·8B ≤ 8 MB (L3), then jumps to
  7.6 ms at 32 MB. **Cap tensor-marginal bin counts at ~500-1000**, else fall back to Alg 2/3.
- **p⁴ intermediate**: core is p₁²×p₂². Fine to k≈20 (1 MB); at k=30 it is 6.5 MB and BLAS starts to bite.
  the reference implementation-typical k=5-20 is the sweet spot.
- 3-way with m=200 marginals → 64 MB histogram: falls off the cliff, needs the fallback.

**Novelty caveat (stated honestly):** this is a *combination* of two published results (Currie's G-operator +
Li & Wood's marginal bin space), not published as such. Li & Wood §3 uses a **different** decomposition —
blocks A′D(Ã_{·i})WD(B̃_{·j})B, i.e. p₁² separate accumulations each re-scanning n. **the reference implementation does not appear to
do this.** Treat as opportunity, and as a reason to validate carefully. Also: the benchmark baseline was a
dense NumPy row-Kronecker; superglm's Python-loop sparse version is likely *slower*, so the figures are
probably conservative — but re-measure against the real code.

## Idea 2 — Li & Wood (2020) Algorithms 0-4 as the general fallback ⭐

**Li, Z. & Wood, S.N. (2019/2020), "Faster model matrix crossproducts for large GLMs with discretized
covariates", Statistics and Computing 30:19-25** — open access; THE paper for this bottleneck. Structures match
superglm exactly: A(i,·) = Σ_s Ã(k_As(i),·), Ã is m_A×p_A.
- **Alg 0** (Lang et al. 2014): A=B, 1-D bincount then Ã′W̄Ã. O(p_A²m_A)+O(n). ← *superglm's bypassed fast
  bincount Gram.*
- **Alg 1** (weight accumulation): m_A×m_B 2-D histogram then Ã′W̄B̃. ← *what superglm already does.* Use only
  when n ≥ m_A m_B.
- **Alg 2/3** (right/left accumulation): for n < m_A m_B, skip W̄ entirely, accumulate C (m_A×p_B) or D
  (m_B×p_A) directly. ← **superglm's MISSING large-m fallback.**
- **Alg 4**: hash-table sparse W̄; poor locality, they gate it behind p_A or p_B > 15.

**Does it fix the G² pass problem? NO — and neither does anything else in the literature.** They keep one
accumulation per term-block pair. Their gains come from (a) per-block cost-model dispatch and (b) getting the
small products into **level-3 BLAS** — they note Wood et al. (2017) *"are entirely vector based, and are
therefore unable to make good use of optimized level 3 BLAS routines."*
**Reported: 30× on X′WX (7024 s → 230 s single-threaded, 9.45M rows); full fit >1 h → <5 min.**

**Two free, high-leverage details:**
- Their loop ordering is tuned for **column-major**; they say *"The order should probably be reversed for row
  major order."* NumPy is C-order — **check superglm's loop nests.**
- **BLAS quality dominates: reference BLAS 126.3 min vs OpenBLAS 12.3 min, same threads — 10× from linking a
  tuned BLAS alone.** Verify superglm isn't hitting a reference BLAS anywhere in this path.

## Idea 3 — Block-level thread parallelism ⭐ (independently corroborates track 5)

Li & Wood §4: *"it is very easy to parallelize the matrix cross product by computing different blocks in
different threads, using openMP."* **No reduction — each thread owns a disjoint output block ⇒ bitwise
deterministic.** Finer (tensor sub-block) granularity load-balances better; process blocks in **decreasing
computational cost**.

Measured (66 covariate pairs, n=1e6, numba `prange` over pairs): 1 thr 212 ms → 4 thr 62 ms (3.4×) →
8 thr 38 ms (5.5×) → **16 thr 21 ms (9.9×)** → 32 thr 24 ms (bandwidth-saturated).
Practical note: **warm up the JIT/thread pool before timing** — first parallel measurement read 0.4× purely
from JIT warmup.

## Negative results — confirmations that should stop work

- **Fused all-pairs kernels: independently reproduced dead.** Same 3-8.5× slowdown. Panel-blocking c
  covariates at a time (the textbook fix) **never beats per-pair** and degrades monotonically as c shrinks
  (0.19-0.70× across c=2..8). Bandwidth-bound on streaming k and w; fusion multiplies the histogram working
  set beyond L2/L3. **Per-pair with a cache-resident W̄ is already near-optimal for the pairs it must compute —
  the win is in NOT NEEDING most of them (Idea 1), not in fusing them.**
- **Histogram privatization/replication: dead here.** R=2,4,8,16 replicas: best case 1.13× (m=50²), everything
  else 0.07-0.67×. Histograms already large enough that update-conflict stalls aren't binding.
- **⚠ int64→int32 via NumPy: NO win — `np.bincount` upcasts internally** (int32 measured *slower*: 1.38 vs
  1.10 ms). **CONFLICT with track 5, which measured 1.41× — resolution: track 5 benchmarked a hand-written
  numba kernel, this track benchmarked `np.bincount`. Both correct in context. Synthesis: narrow dtypes pay
  ONLY in hand-written numba/C kernels, and are worthless as a NumPy-level dtype change.** Sequencing
  implication: narrow the dtypes *as part of* writing first-party kernels, not as a standalone PR.
- **Rügamer (2024), "Scalable Higher-Order Tensor Product Spline Models", AISTATS, arXiv:2402.01090** —
  factorization-machine approximation collapsing O(p^D M^D) → O(pMFD), memory O(npM) regardless of D. Real and
  elegant, **but it is a rank-F approximation ⇒ breaks reference parity by construction.** A different estimator,
  not a faster one. Rule out as a drop-in; note for a future "many weak interactions" mode.
- **glamlasso (Lund, Vincent & Hansen, arXiv:1510.03298) warning:** they could **not** make coordinate descent
  exploit array structure (*"it is not obvious how to exploit the array structure to reduce the computational
  complexity"*). **Don't expect the array trick to speed up the BCD inner loop — only the Gram formation
  feeding it.**
- **SpGEMM/GPU literature (MAGNUS arXiv:2501.07056; SMASH arXiv:2105.14156): does not transfer.** Those
  optimise irregular sparse *symbolic* structure with unknown output patterns; superglm's accumulation has a
  dense, known-shape, cache-resident output and a regular index stream. Li & Wood already tried the sparse-hash
  route (Alg 4) and gate it behind p>15 for exactly this reason.
- **Wood, Goude & Shaw (2015, JRSS-C 64:139-155)** — `bam`'s original iterative QR updating on sub-blocks,
  never holding X. Orthogonal to this bottleneck (superglm forms X′WX, doesn't factorise X). Low priority.

## Recommended sequence from this track

1. **Row-tensor bin-space Gram for all tensor blocks.** Biggest win, self-contained, kills the row-Kronecker
   path and `B_joint` entirely. Precompute G(Ãⱼ) once per fit. ~50× on tensor blocks, ~12× on the Gram stage.
2. **Thread over blocks**, disjoint outputs, decreasing-cost order. ~10×, deterministic. Compounds to ~39×.
3. **Cost-model dispatch across Li & Wood Algs 0-3** for blocks where m_A m_B > n; plus the row-major
   loop-order fix and a BLAS audit.
4. Cap tensor-marginal bin counts so W̄ stays ≤ L3; fall back to Alg 2/3 above that.

Sanity checks before committing: benchmarks used uniform independent bin indices (real correlated insurance
data gives sparser, *more* cache-friendly histograms — should favour the method further); re-measure the
baseline against superglm's actual row-Kronecker code rather than the NumPy stand-in.
