# FactorSmooth Discrete Moment Optimization Design

**Date:** 2026-07-25
**Status:** Approved experimental design

## Summary

Prototype a faster large-row-count moment path for structured
`FactorSmooth(..., basis="fs"|"sz")` fits with `discrete=True`. The experiment
first removes repeated observation scans by aggregating changing PIRLS
quantities into compact `(factor level, spline support bin)` cells. Only after
that serial algorithm is measured will a bounded, thresholded parallel row
reduction be retained.

This is a benchmark-gated experiment. An optimization that does not improve
the measured workload, changes model semantics, materially regresses smaller
fits, or requires an observation-by-coefficient expansion is reverted rather
than shipped.

No public API, LSS code, or LSS semantics change.

## Profile Evidence

The reference workload is:

- Poisson REML;
- 1,000,000 rows;
- 300 factor levels;
- SZ factor smooth with `k=10`;
- one explicit global spline and four ordinary numeric columns;
- `discrete=True` and `direct_solve="structured"`.

Two clean fits took 7.69 and 7.65 seconds and converged in five REML
iterations. A whole-fit cProfile took 7.92 seconds, close enough to the clean
wall time to represent the actual call stack:

```text
fit_reml                                      7.92 s
└─ optimize_discrete_reml_cached_w            5.88 s
   └─ fit_irls_direct                         5.50 s
      └─ build_block_structured_system        3.91 s
         ├─ factor_smooth_dense_cross_gram    2.42 s
         └─ factor_smooth_sufficient_stats    0.78 s
```

The dense cross routine ran 182 times: thirteen small-side columns across
fourteen structured-system builds. Each call traversed all one million
observations. Tabmat's dense sandwich cost only 0.13 seconds and the structured
linear solves about 0.2 seconds, so neither is the first optimization target.

## Goals

1. Replace repeated discrete FactorSmooth observation scans with compact
   cell-level sufficient statistics.
2. Benefit both FS and SZ without changing their coefficient or penalty
   geometry.
3. Preserve the compact `codes + support basis` representation.
4. Retain deterministic results under a fixed chunking policy, allowing only
   strict-tolerance floating-point differences from the current summation
   order.
5. Use parallel reduction only when measured work is large enough to amortize
   scheduling and reduction overhead.
6. Re-profile the complete fit and retain only changes supported by evidence.
7. Use existing high-performance dependencies, including Numba and Tabmat,
   wherever profiling demonstrates that they are the right execution layer.

## Non-goals

- Changing the public `FactorSmooth`, `Spline`, or solver API.
- Replacing Tabmat for ordinary dense, sparse, or categorical blocks.
- Materializing an `n x Kk`, `Kk x Kk`, or general `n x q` compatibility
  design.
- Optimizing multiple simultaneous dominant FactorSmooth terms.
- Adding distributed, GPU, or stochastic REML fitting.
- Adding Cython or native Rust, C, or C++ extensions.
- Changing exact-mode algebra in the first experiment.
- Changing selection penalties or LSS fitting.

## Considered Approaches

### A. Cell aggregation followed by bounded parallelism

For each observation, define:

```text
cell = level_code * n_support_bins + spline_bin
```

Each structured-system build aggregates the changing working quantities by
cell in one row pass. Local spline Grams, transpose products, and supported
small-side crosses are then formed from the compact cell arrays.

This is the recommended approach because it changes the dominant basis work
from observation scale to occupied-cell scale before introducing threading.

### B. Parallelize the current per-column scans

Independent column scans can run concurrently after releasing the GIL.
However, this retains thirteen reads of the codes, bin indices, weights, and
basis support for the reference workload. It is likely to become
memory-bandwidth-bound and preserves the algorithmic waste identified by the
profile. It remains a fallback experiment only if cell aggregation cannot
support an important small-matrix type.

### C. Materialize the small-side design

Building an `n x q` dense matrix would allow one batched cross kernel, but the
reference workload would add roughly 104 MiB for thirteen columns. The cost
grows directly with rows and can reach gigabytes in intended production
settings. This approach is rejected.

## Serial Cell Algebra

Let `B[b, :]` be the shared `k`-column FactorSmooth basis at support bin `b`.
For level `g`, aggregate:

```text
W_cell[g, b]   = sum_i W[i]
R_cell[g, b]   = sum_i rhs[i]
```

over rows whose `(level, bin)` equals `(g, b)`. Then:

```text
D[g]     = sum_b W_cell[g, b] B[b]' B[b]
X'W[g]   = sum_b W_cell[g, b] B[b]
X'rhs[g] = sum_b R_cell[g, b] B[b]
```

These are the same raw level moments produced by the current discrete kernel.
The existing natural-map contraction and FS/SZ factor geometry remain
unchanged.

For a dense small block `Z` with width `q`, aggregate:

```text
Z_cell[g, b, :] = sum_i W[i] Z[i, :]
```

and form:

```text
C[g] = sum_b B[b]' Z_cell[g, b, :]
```

All dense columns in one group are accumulated together. The implementation
must not call the dominant cross routine once per column.

For a discretized global spline that shares the same observation bin mapping,
`W_cell` is reused directly:

```text
C_global[g] = sum_b W_cell[g, b] B_factor[b]' B_global[b, :]
```

If support mappings differ, a bounded joint-bin aggregation may be used when
its cell count is below an explicit memory threshold. Unsupported or oversized
small groups use the existing compact fallback; correctness never depends on
the optimized dispatch.

The initial prototype may support only the profiled common cases:

- dense ordinary blocks;
- a matching discretized global spline;
- no small block.

Other matrix types retain existing behavior until separately measured.

## Parallel Reduction

Parallelism is a second, independently benchmarked layer over the row
aggregation only.

- Divide observations into a fixed number of contiguous chunks.
- Each chunk writes private cell arrays, avoiding shared updates and races.
- Reduce private arrays in ascending chunk order.
- Cap the prototype at eight chunks.
- Dispatch serially below a work threshold derived from `n`, `k`, small width,
  and occupied-cell count.
- Do not alter global BLAS or Numba thread settings.

Fixed chunk boundaries and fixed-order reduction make a run deterministic for
the same geometry. The changed summation order may produce tiny differences;
existing solver and prediction parity tolerances remain authoritative.

Thread-local storage is bounded by:

```text
chunks * K * bins * (2 + supported_dense_width)
```

The implementation must check this estimate before selecting the parallel
path. If it exceeds the configured internal budget, it uses the serial cell
path.

## Integration Boundary

The grouped-matrix layer owns raw discrete cell aggregation because it owns
factor codes, support bins, and the shared basis. It exposes compact raw
moments to `build_block_structured_system`.

The structured solver continues to own:

- ordinary-small moment assembly through Tabmat and existing execution plans;
- natural-map contraction;
- FS block-Schur or SZ constrained geometry;
- penalty, trace, log-determinant, and covariance operations.

The builder chooses optimized crosses per supported small group and falls back
per group. It does not add factor-basis-specific logic to the raw kernels:
both FS and SZ consume the same `K` raw level moments.

## Testing

Implementation follows red-green-refactor cycles.

1. Raw serial cell moments match current discrete moments for randomized
   weights, signed weights, RHS values, empty levels, and repeated bins.
2. Batched dense crosses match dense reference algebra for widths 1, 4, and
   13.
3. Matching global-spline crosses match the existing column fallback.
4. FS and SZ structured systems match dense materialization.
5. Parallel and serial cell paths agree within strict tolerances and are
   deterministic across repeated runs.
6. Tests forbid dominant design materialization and a general `n x q`
   compatibility allocation.
7. A call-count regression test proves the profiled common case no longer
   performs one dominant row scan per small-side column.
8. Existing exact/discrete FS, SZ, REML, inference, allocation, and mgcv parity
   tests remain green.

## Benchmark and Keep/Revert Gates

Use clean, fixed-seed repetitions with cProfile and tracemalloc disabled for
wall timing. Profile a separate complete fit after choosing the winner.

Required cases:

| Rows | Groups | `k` | Purpose |
|---:|---:|---:|---|
| 20,000 | 300 | 10 | Detect small/medium regression |
| 100,000 | 300 | 10 | Locate dispatch crossover |
| 250,000 | 300 | 10 | Locate dispatch crossover |
| 1,000,000 | 300 | 10 | Primary large-row workload |

Keep the serial cell path only if:

- the million-row median improves materially, with 20% as the target;
- the 20,000-row case regresses by no more than 5%;
- prediction, objective, lambda, and EDF parity pass;
- peak incremental memory remains compact and does not include `n x q`.

Keep the parallel layer only if it adds a repeatable improvement over the
serial cell path at a measured crossover and does not regress cases below that
crossover. Otherwise ship the serial cell improvement alone.

The final whole-fit cProfile must show that the repeated
`factor_smooth_dense_cross_gram` stack has collapsed. If another stack becomes
dominant, further work requires a new evidence-backed scope rather than an
unplanned refactor.

## Failure and Fallback Behavior

Optimization eligibility is internal. Any unsupported geometry, excessive
cell allocation estimate, unavailable compiled parallel runtime, or failed
numerical certification uses the existing serial compact path. It must not
silently select dense Gram solving, change fitted semantics, or expose a new
user configuration requirement.
