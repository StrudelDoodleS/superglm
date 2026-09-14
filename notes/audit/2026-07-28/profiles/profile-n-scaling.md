# n-scaling baseline profile — superglm @ f082e9b (audit-master)

Profiling agent, cProfile + pstats. All runs single-process on the audit venv
(numpy 2.4.2, Python 3.14, tabmat 4.2.1 installed). Wall times are measured
**under cProfile**, so absolute numbers carry ~10–30% profiler overhead,
concentrated in call-heavy Python loops; relative comparisons and scaling are
self-consistent. Raw `.prof` files and per-run cumtime/tottime top-40 dumps sit
next to this report (`nscale_<config>_<n>.prof` / `.pstats.txt`,
`nscale_results.jsonl`).

## Model and data

Fixed moderate insurance-frequency model, seeded (`default_rng(42)`):

- 4 × `Spline(kind="ps", k=10)` → 9 built columns each (k−1 contract holds)
- 2 × `Categorical` (8 levels → 7 cols; 20 levels → 19 cols)
- 2 × `Numeric`
- total p = 64, Poisson / log link, per-exposure frequency response,
  exposure ∈ U(0.05, 1) passed as `sample_weight=`.

All fits converged; no run was skipped (largest run = 54 s, far under the
8-minute cap). Discrete-vs-exact accuracy parity is good: at n=1M deviance
306410.8 vs 306411.5, EDF 41.82 vs 41.93.

## Wall-time table

| config | n | wall (s) | PIRLS fits | gram passes¹ | EDF |
|---|---:|---:|---:|---:|---:|
| (a) fit_reml exact | 5,000 | 2.19 | 25 | 124 | 33.0 |
| (a) fit_reml exact | 20,000 | 5.45 | 23 | 109 | 34.8 |
| (a) fit_reml exact | 80,000 | 8.87 | 17 | 77 | 38.4 |
| (a) fit_reml exact | 320,000 | 21.46 | 11 | 50 | 43.5 |
| (a) fit_reml exact | 1,000,000² | 53.67 | 13 | 59 | 41.9 |
| (b) fit_reml discrete | 5,000 | 0.39 | 15 | 23 | 33.0 |
| (b) fit_reml discrete | 20,000 | 1.81 | 14 | 20 | 34.8 |
| (b) fit_reml discrete | 80,000 | 3.40 | 11 | 16 | 38.4 |
| (b) fit_reml discrete | 320,000 | 6.92 | 9 | 14 | 43.4 |
| (b) fit_reml discrete | 1,000,000 | 17.59 | 9 | 14 | 41.8 |
| (c) plain fit(), default penalty | 80,000 | 1.50 | 1 | 5 | 59.4 |

¹ calls to `centered_gram_rhs` (`_group_matrix/_group_matrix_centered.py:911`) —
one full O(n·p²) pass over the data each.
² bonus point beyond the requested grid (budget allowed); improves slope
estimates and gives both paths a common largest n.

Discrete/exact speedup: 5.7× (5k), 3.0× (20k), 2.6× (80k), 3.1× (320k),
3.05× (1M).

## Empirical scaling exponents (log-log slope between consecutive sizes)

| transition | exact | discrete |
|---|---:|---:|
| 5k → 20k | 0.66 | 1.12 |
| 20k → 80k | 0.35 | 0.46 |
| 80k → 320k | 0.64 | 0.51 |
| 320k → 1M | 0.80 | 0.82 |
| **least-squares slope (all 5 pts)** | **0.58** | **0.67** |

Wall-time exponents are misleadingly sublinear because **iteration counts fall
as n grows** (smoother REML objective: exact does 124 gram passes at 5k but
only 59 at 1M). Normalising per gram pass gives the true per-unit-work scaling:

| n | exact s/pass | discrete s/pass |
|---:|---:|---:|
| 5k | 0.011 | 0.006 |
| 20k | 0.030 | 0.034 |
| 80k | 0.082 | 0.106 |
| 320k | 0.345 | 0.343 |
| 1M | 0.743 | 0.848 |

Per-pass slopes are ≈0.7–1.0 for both paths → **both paths' inner linear
algebra is O(n)**, and — the central observation — **the discrete path's
per-pass gram cost is essentially identical to (at 1M slightly worse than) the
exact path's**. At the top end both wall times converge to slope ≈0.8, i.e.
asymptotically linear with a shrinking fixed-overhead share.

## Hotspots at the largest completed n

### Exact path, n = 1M (53.7 s total)

| cum s | % | where (file:line) |
|---:|---:|---|
| 48.6 | 91% | `reml/direct.py:56 optimize_direct_reml` — lambda outer loop |
| 43.8 | 82% | `_group_matrix/_group_matrix_centered.py:911 centered_gram_rhs` (59 calls × 0.74 s) — X'WX formation |
| 29.1 | 54% | `solvers/irls_direct.py:382 fit_irls_direct` (13 PIRLS fits, 35 iterations) |
| 25.9 | 48% | └ `solvers/centered_system.py:184 build_centered_system` (35 calls) |
| 21.0 | 39% | `reml/w_derivatives.py:176 reml_w_correction` (6 outer Newton iters) |
| 19.1 | 36% | └ `reml/w_derivatives.py:366 centered_signed_gram` (24 calls = 4 spline groups × 6 iters) → same `centered_gram_rhs` |
| 2.1 | 4% | design/basis build (`dm_builder.py:667`; scipy dierckx `data_matrix` 0.5 s) |
| 2.1 | 4% | `evaluate_state` (mu/eta/deviance per IRLS iteration) |
| 2.1 | 4% | `reml_finalize.py:317 finalize_reml_fit` |
| ~0 | — | factorisation: p = 64, cholesky/eigh never reach top-40 |

Inside the 43.8 s of `centered_gram_rhs`, only ~20 s is the actual centered
BLAS product `block.T @ (W[:,None]*block)` plus Kahan-compensated
accumulation. The rest is **chunk marshalling**: per 8192-row chunk it calls
`dm.row_subset(rows).toarray()`, which fancy-indexes each group's CSR matrix
(`scipy _major_index_fancy` + `csr_matvecs`: 7.7 s over 29,028 calls) and then
`np.hstack`s 8 dense group blocks (`group_matrix.py:440 toarray` → `hstack`
9.8 s over 7,257 calls). So ~45–55% of X'WX time is Python/scipy data
movement, not FLOPs.

**Call-stack narrative (exact):** `optimize_direct_reml` runs ~6 Newton steps
on log-lambda. Each step (i) re-runs PIRLS (`fit_irls_direct`), and every IRLS
iteration rebuilds the full centered Gram from scratch —
`build_centered_system → centered_gram_rhs`, a full O(n·p²) chunked-dense pass
(35 passes); (ii) computes the Wood-2011 W(rho) implicit-differentiation
correction — `reml_w_correction → centered_signed_gram`, which is **one more
full-data O(n·p²) gram pass per spline group per outer iteration** (4 × 6 = 24
passes, 39% of total runtime). The gradient/Hessian algebra itself
(`XtWX_S_inv` manipulations at p=64) is negligible; the outer loop's cost is
entirely these repeated data-Gram passes.

### Discrete path, n = 1M (17.6 s total)

| cum s | % | where (file:line) |
|---:|---:|---|
| 12.9 | 73% | `reml/discrete.py:164 optimize_discrete_reml_cached_w` |
| 13.9 | 79% | `fit_irls_direct` (9 PIRLS fits, 21 iterations, 14 centered systems) |
| 11.9 | 68% | `centered_gram_rhs` (14 calls × 0.85 s) — **same chunked-dense fallback as exact** |
| 2.7 | 15% | └ `_group_matrix_discretized.py:79 toarray` — gather `(B_unique @ R_inv)[bin_idx]` per chunk |
| 2.3 | 13% | └ `np.hstack` chunk assembly |
| 1.8 | 10% | design build (incl. `discretize_column` 0.7 s, spline identifiability 1.4 s) |
| 2.7 | 15% | finalize + `canonicalize_fitted_model` |
| 0.9 | 5% | `evaluate_state` |
| ~0 | — | factorisation, W-correction (absent by construction) |

**Call-stack narrative (discrete):** the cached-W optimizer holds working
weights fixed across lambda updates, so it needs only 14 Gram passes total
(vs 59) and **no per-lambda `centered_signed_gram` corrections at all**. That —
not bin-space arithmetic — is the entire ~3× speedup.

### Plain fit(), n = 80k (1.50 s)

Single IRLS fit, 5 iterations: gram formation 0.42 s, post-fit
canonicalisation + public-runtime parity validation 0.57 s, design build
0.24 s. At this scale, fixed Python overhead (validation, scoring,
canonicalisation) is ~2/3 of wall time. fit() at 80k is ~6× cheaper than
fit_reml exact and ~2.3× cheaper than fit_reml discrete — the entire REML
premium is the repeated Gram passes.

## Key architectural finding: the discrete fast Gram is silently bypassed

`DiscretizedSSPGroupMatrix` implements a true fREML-style
O(n + n_bins·p_g²) Gram (`gram_rmatvec` via fused `bincount`,
`_group_matrix_discretized.py:65`), and `centered_system.py:184` has a ladder
of fast paths (`packed_centered_gram_rhs`, `_try_mixed_discrete_centering`,
tabmat, raw-spline tabmat). **None of them fired for this model**, in either
path, at any n. Verified on the fitted design (see `check_fastpath.py`):

- `packed_centered_gram_rhs` (`_group_matrix_centered.py:689`) requires *every*
  group to be Discretized/Tensor/Categorical — the two `Numeric` features are
  `DenseGroupMatrix`, so it returns None.
- `_try_mixed_discrete_centering` (`_group_matrix_centered.py:318`) bails when
  `len(categorical_groups) > 1` — this model has two categoricals (Region 8,
  VehBrand 20), so `mixed_bin_space_centering_plan` is None. Two-plus
  categoricals is the *normal* case for insurance frequency models.
- No tabmat frames appear in any profile despite tabmat 4.2.1 being installed.

Consequently every X'WX in **both** the exact and discrete paths is formed by
the stable chunked-dense fallback `centered_gram_rhs` at O(n·p²) with heavy
per-chunk `row_subset`/`toarray`/`hstack` marshalling. `discrete=True` today
buys (i) fewer outer/inner iterations via cached W, (ii) no W-correction
passes, (iii) cheaper basis storage — but its headline O(bins) Gram
accumulation is dead code for a bread-and-butter 2-categorical + numeric
model. Per-pass Gram cost at 1M is *higher* for discrete (0.85 s) than exact
(0.74 s) because the discretized gather-`toarray` per chunk is pure overhead.

## Observations / improvement levers (ranked by measured impact)

1. **Lift the ≤1-categorical restriction in `_try_mixed_discrete_centering`**
   (or fold multiple categoricals into the augmented bin-space plan).
   At 1M discrete this would replace 11.9 s of chunked-dense Gram with
   bincount-style accumulation — the profile suggests the discrete path could
   approach basis-build + finalize cost (~4–5 s).
2. **W-correction Gram passes dominate the exact outer loop** (21 s / 39% at
   1M: 4 full-data signed Grams per Newton iteration). Batching the 4
   per-group signed Grams into one pass over the data (they share the same
   row blocks) would cut that ~4×; reusing chunks between the IRLS Gram and
   the correction Gram would help further.
3. **Chunk marshalling is ~half of Gram time in the exact path** (row_subset
   fancy-index 7.7 s + toarray/hstack 16.2 s vs ~20 s BLAS at 1M). A
   preallocated dense chunk buffer filled group-by-group (no hstack, no
   intermediate CSR slices) is a straightforward ~1.5× on the dominant kernel.
4. Iteration counts, not per-pass cost, drive the apparent sublinear wall-time
   scaling; any warm-start improvement across outer lambda steps directly
   multiplies through both paths (exact does 13 PIRLS fits at 1M).
5. Fixed overhead (canonicalisation + parity validation + design build) is
   ~10–25% below n≈100k; it is amortised by 1M and is not a scaling concern.

## Skipped runs

None. All requested configurations completed; n=1M was added for the exact
path as a bonus data point.
