# p-scaling baseline profile — superglm audit

Target: `/home/mhick/python_projects/superglm/.worktrees/audit-master` @ origin/master (f082e9b).
Fixed n = 20,000 simulated Poisson-frequency rows (log link, exposure ~ U(0.2, 1) passed as
`sample_weight`, y = counts/exposure, signal on 4 smooth terms + mild categorical effects,
`numpy.random.default_rng(42)`). Splines are `Spline(kind="ps", k=10)` → 9 built columns each
(k−1 identifiability contract). All timings measured **under cProfile** (this project's standard);
a_m25 executes ~122M Python-level calls, so absolute walls carry roughly 10–20% profiler
inflation on the sparse-matvec-heavy configs, but hotspot ranking and relative scaling are robust.
Raw artifacts: `<config>.prof`, `<config>.pstats.txt`, `<config>.json` in this directory.
Driver: `run_pscaling.py` (one config per process; `ru_maxrss` is per-process peak).

## 1. Wall-time table

| config | path | m splines | cat levels | p (built) | q (penalties) | REML outer iters* | wall (s) | peak RSS (MB) |
|--------|------|-----------|-----------|-----------|----------------|-------------------|----------|----------------|
| a_m5   | fit_reml exact | 5  | —       | 45  | 5  | 5  | 4.75  | 336 |
| a_m15  | fit_reml exact | 15 | —       | 135 | 15 | 12 | 59.86 | 377 |
| a_m25  | fit_reml exact | 25 | —       | 225 | 25 | 13 | 172.10 | 439 |
| a_m40  | fit_reml exact | 40 | —       | 360 | 40 | —  | **SKIPPED: >520 s timeout** | — |
| a_m80  | fit_reml exact | 80 | —       | 720 | 80 | —  | **SKIPPED (projected ≥50 min from slope)** | — |
| b_L50  | fit_reml exact | 4  | 2×25    | 84  | 4  | 5  | 2.86  | 344 |
| b_L200 | fit_reml exact | 4  | 2×100   | 234 | 4  | 5  | 8.63  | 391 |
| b_L800 | fit_reml exact | 4  | 2×400   | 834 | 4  | 20 | 54.08 | 726 |
| c_m40  | fit(group_lasso, λ1=auto→0.00122) | 40 | — | 360 | — | — | 1.71 | 471 |
| c_m80  | fit(group_lasso, λ1=auto→0.00111) | 80 | — | 720 | — | — | 3.27 | 653 |

\* outer iterations inferred from `reml_w_correction` call counts in the profiles
(`model.reml_diagnostics()` returned no `n_iter` key). All fits converged. Interpreter+libs
baseline RSS is ~300 MB, so incremental model memory is modest (≤ ~430 MB at p=834).

An extra intermediate point (a_m25) was added because a_m40 blew the 8-minute cap, to get a
third scaling datum for part (a).

## 2. Empirical scaling exponents (log-log slope between consecutive sizes)

**(a) many smooths — q grows with p (q = m, p = 9m):**

| transition | p ratio | time ratio | slope in p |
|------------|---------|-----------|------------|
| m5 → m15 (45→135)   | 3.00× | 12.59× | **2.31** |
| m15 → m25 (135→225) | 1.67× | 2.87×  | **2.07** |
| m25 → m40 (225→360) | 1.60× | ≥3.02× (timeout floor) | **≥2.35** |

**(b) high-cardinality categoricals — q fixed at 4:**

| transition | p ratio | time ratio | slope in p |
|------------|---------|-----------|------------|
| L50 → L200 (84→234)  | 2.79× | 3.02× | **1.08** |
| L200 → L800 (234→834)| 3.56× | 6.27× | **1.44** |

**(c) group-lasso BCD path:**

| transition | p ratio | time ratio | slope in p |
|------------|---------|-----------|------------|
| m40 → m80 (360→720) | 2.00× | 1.91× | **0.94** |

**Answer to "is exact REML p², p³, does q drive an extra factor?":** at n=20k it is
empirically ~p² **times q**, i.e. the penalty count is the dominant extra factor. Compare
a_m25 (p=225, q=25, 172 s) with b_L200 (p=234, q=4, 8.6 s): *same p, 20× slower with 6× more
penalties*. Track (b) shows that with q fixed the cost is only ~p^1.1–1.4 in this regime
(Gram formation on sparse one-hot blocks is cheap; the p^1.44 tail is the emerging dense
O(p³) Cholesky/decompose share, which was 16.2 s of 54 s at p=834 and will dominate for
larger p). Track (a)'s superquadratic slope (2.1–2.3, rising) is the product of the
per-penalty O(q · n · p²) W-correction loop with a Python/scipy constant ~10–50× worse than
BLAS (see hotspots) plus growing outer-iteration counts. b_L800's jump to 20 outer iterations
(vs 5 at L50/L200) also shows iteration count is not size-stable.

## 3. Hotspots at largest p, with call-stack narrative

### (a) exact REML, many smooths — a_m25 (p=225, q=25, 172 s total)

**85% of the entire fit (147.2 s) is `reml_w_correction`** — the Wood (2011) W(rho)
implicit-differentiation correction — and inside it essentially all time is per-penalty
signed Gram formation:

```
reml/direct.py:56 optimize_direct_reml                       170.97s
└─ reml/w_derivatives.py:176 reml_w_correction   13 calls    147.23s   (once per REML outer iter)
   └─ per-penalty loop (w_derivatives.py:427): for each of q=25 penalties,
      a_j = dW/deta * X_c dbeta_j, then C_j = X_c' diag(a_j) X_c via
      └─ w_derivatives.py:366 centered_signed_gram  325 calls (=13×25)   145.44s
         └─ _group_matrix/_group_matrix_execution.py:301 _moments_impl   145.33s
            └─ _group_matrix_algebra.py:749 _cross_gram   97,500 calls   140.54s  (=325 × 300 group pairs, 25·24/2)
               └─ _group_matrix_algebra.py:722 _cross_gram_by_columns    139.65s
                  columns one at a time via unit vectors:
                  ├─ scipy csc_matvec   895,600 calls   49.19s tottime
                  ├─ scipy csr_matvec   887,850 calls   35.82s tottime
                  ├─ sparse .T constructed per call (csr transpose)      28.67s
                  └─ ~907k fresh sparse-matrix __init__ / check_format / prune  ~38s
```

I.e. `X_c' diag(a_j) X_c` is rebuilt **q times per outer iteration** for the full p×p Gram,
and because SSP spline group matrices take the "bounded-memory factored fallback", each of
the ~300 group-pair blocks is formed **one column at a time** with length-20k sparse
matvecs, allocating a fresh transposed sparse matrix per matvec. ~1.79M single-vector sparse
products; each 9×9 block costs ~18 matvecs + 18 sparse-object constructions. A dense
`(w[:,None]*Xc).T @ Xc` at p=225 is ~2e9 flops (≈0.05–0.1 s in BLAS) versus the observed
0.45 s per Gram — and the a_j-weighted Grams for all q penalties could share one densified
X_c (or at minimum cache the CSC/CSR forms instead of re-transposing 900k times).

Secondary: `solvers/irls_direct.py:473 _fit_irls_direct_once` (27 PIRLS runs, 23.8 s), of
which `centered_system.py:184 build_centered_system` → `_group_matrix_centered.py:911
centered_gram_rhs` 9.7 s and `solvers/rank.py:453 decompose_gram` 10.6 s (127 calls,
includes 3.4 s scipy Cholesky). Eigendecompositions and log-determinants are *not* material
at this size — Gram formation is.

Same structure at a_m15: `reml_w_correction` 34.6 s (58%) vs PIRLS 24.7 s; at a_m5 it is
already 51%. The w-correction share grows ~linearly with q, which is exactly the extra q
factor in the observed p^2.3 slope.

### (b) exact REML, high-cardinality categoricals — b_L800 (p=834, q=4, 54 s total)

With q small the profile flips to the PIRLS inner loop:

```
_fit_irls_direct_once           30 calls   49.72s  (92%)
├─ build_centered_system → _group_matrix_centered.py:911 centered_gram_rhs
│     72 calls, 27.18s (16.8s tottime)          ← X'WX Gram formation, once per IRLS iter
├─ solvers/rank.py:453 decompose_gram  132 calls, 16.21s
│  ├─ scipy cholesky                     5.68s   ← dense p³ factor + log|H| source
│  ├─ rank.py:352 _equilibrate_gram      1.84s
│  └─ rank.py:208 pseudo_inverse   60 calls, 2.69s
├─ group_matrix.py:440 toarray → np.hstack   216 calls, 8.8s + 6.1s  ← block densification p=834
└─ reml_w_correction   20 calls, only 3.26s  (q=4)
```

So on the categorical track the costs are (i) per-IRLS-iteration Gram (`centered_gram_rhs`,
re-formed 72 times — no weight-update reuse), (ii) dense Cholesky-based `decompose_gram` at
p=834 (this is the p³ term that produced the 1.44 slope), and (iii) repeated
`toarray`/`hstack` densification of the sparse one-hot blocks (~15 s combined). Peak RSS
726 MB is consistent with dense p×p plus densified n×p scratch.

### (c) group lasso / BCD — c_m80 (p=720, 3.27 s total)

```
fit_ops.py:718 fit → solvers/pirls.py:1401 fit_pirls → pirls.py:607 _fit_pirls_inner  2.00s
├─ centered_gram_rhs (1 call)          0.57s     ← one Gram per outer IRLS iteration
├─ decompose_gram (3 calls)            0.33s
├─ pirls.py:342 _build_group_hessians  ~0.2s     ← per-group blocks for BCD
└─ BCD cycles + prox_group (pirls.py:879/938): negligible at 4 outer iters
fixed overhead outside solver: design build 0.3s, scipy B-spline basis eval 0.4s,
runtime canonicalization + numba compile ~0.5s
```

BCD spends its time in Gram/Hessian *formation*, not in the coordinate cycles; growth is
~linear in p at this n (slope 0.94) because n·p Gram work dominates and iteration counts
stayed at 4. The fit() path is ~100× faster than exact REML at the same p — the gap is
entirely the REML outer loop's repeated Grams and W-correction.

**Sparsity caveat (flag for correctness auditors):** the requested "partial sparsity" point
could not be constructed. With `selection_penalty="auto"` the resolved λ1 was ~10% of
lambda_max (0.0012), and sweeps at λ1 ∈ {0.005, 0.02, 0.05, 0.1, 0.5, 2, 10} on the m=40
config zeroed **zero of 40 groups** in every case (checked both canonical beta and
solver-space beta; `diagnostics()` reports 40/40 active with edf≈8 and O(1) group norms even
for the 36 pure-noise splines). λ1=10 is ~800× the implied lambda_max, at which group lasso
should zero everything. Either lambda-max calibration, the prox threshold scaling, or the
active-flag/final-refit semantics deserves a correctness look (scan scripts:
`sparsity_scan*.py`). Consequently the (c) profiles reflect the no-zero-group regime.

## 4. Summary observations

1. **Exact REML cost ≈ O(iters · q · n · p²) with a very large constant**, realized as
   `reml_w_correction → centered_signed_gram → _cross_gram_by_columns` doing ~1.8M
   single-column sparse matvecs (a_m25). This, not eigen/log-det work, is why 40 smooths at
   n=20k exceeds 8 minutes. Densifying X_c once per outer iteration (n·p = 20k×360 ≈ 58 MB)
   or batching the q signed Grams over one cached design would plausibly cut track (a) by
   5–20×.
2. **q (number of penalties), not p, is the primary driver** on the smooth-heavy track:
   p≈230 costs 8.6 s with q=4 but 172 s with q=25.
3. With q fixed, exact REML is ~p^1.1→1.4 at n=20k, trending to p³ via dense
   `decompose_gram`/Cholesky (16 s of 54 s at p=834); `centered_gram_rhs` re-forms X'WX
   every IRLS iteration (72×) with no cross-iteration structure reuse, and sparse blocks
   are repeatedly densified via `toarray`+`hstack`.
4. The BCD/group-lasso path scales ~linearly in p here and is dominated by Gram/Hessian
   formation, not by proximal cycling.
5. Memory is not a constraint at this scale (≤726 MB incl. ~300 MB interpreter baseline).
6. `Skipped:` a_m40 (timed out at 520 s cap), a_m80 (not attempted; slope projection ≥50 min).
