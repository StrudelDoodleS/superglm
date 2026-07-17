# Tabmat 4.2.1 integration notes

This note is the shared API and review brief for SuperGLM's Tabmat work. It records public,
supported behavior only; optimizations must not depend on Tabmat private implementation details.

## Verified release and compatibility

- Latest stable release checked on 2026-07-17: Tabmat 4.2.1, released 2026-02-04.
- The project requires Python 3.10 or newer. PyPI publishes CPython 3.14 and 3.14t wheels.
- The 3.14t extension is not declared free-thread safe: importing it emits a warning and enables
  the GIL. Treat 3.14t as compatibility-only until Tabmat explicitly supports free threading.
- Tabmat 4.2.2 is currently unreleased. Its changelog says fast-math is disabled to avoid invalid
  edge-case results, so it must be re-evaluated when released rather than pre-emptively vendored.
- SuperGLM requires and locks Tabmat 4.2.1. The former `>=4.0` floor was not honest because the
  existing pinned-category constructor uses the `categories=` API introduced after 4.0.
- Tabmat's public matrix and standardized-matrix tests passed 5,164/5,164 on CPython 3.14.4 and
  5,164/5,164 on CPython 3.14.2t; the latter ran with the expected GIL auto-enable warning.

Primary references:

- [Official API](https://tabmat.readthedocs.io/en/latest/api.html)
- [Official benchmarks](https://tabmat.readthedocs.io/en/stable/benchmarks.html)
- [Official changelog](https://tabmat.readthedocs.io/en/latest/changelog.html)
- [PyPI release and wheels](https://pypi.org/project/tabmat/)

Context7 was queried on 2026-07-17 for Tabmat and glum; it did not return either library. Agents
should therefore use the official links above and the installed 4.2.1 public API rather than
accepting an unrelated Context7 match.

## Public operations relevant to fitting

- `MatrixBase.sandwich(d, rows=None, cols=None)` computes
  `X.T @ diag(d) @ X` and can restrict rows or columns without making an explicit subset matrix.
- `transpose_matvec(v, rows=None, cols=None, out=None)` computes `X.T @ v`. `out=` is additive and
  remains full-width when `cols=` is supplied; it must be zeroed before reuse. Measurements showed
  no useful hot-path win from adding a reusable output workspace.
- `matvec(v, cols=None, out=None)` supports active-column products without converting the matrix.
- `CategoricalMatrix` stores one-hot categorical columns through one category code per row. Its
  sandwich diagonal and transpose products avoid materializing the indicator matrix.
- `SplitMatrix` keeps dense, sparse, and categorical components in their natural representations
  while exposing one matrix API.
- `MatrixBase.standardize(weights, center_predictors, scale_predictors)` returns a
  `StandardizedMatrix` plus weighted means and optional scales. `StandardizedMatrix` retains the
  underlying sparse/categorical representation and applies shifts/scales algebraically. In 4.2.1,
  callers must normalize `weights` to sum to one; the API does not do so itself.
- `from_df` selects dense, sparse, and categorical components using `sparse_threshold`,
  `cat_threshold`, and `object_as_cat`. Since 4.1.4 it avoids unnecessary dense-array copies while
  still guaranteeing contiguous dense components.

## SuperGLM dispatch policy

1. Keep SuperGLM's compact, deterministic support/histogram kernels first when every group is
   eligible; categorical-only and discretized designs are already strong here.
2. Lazily build a `SplitMatrix` only for a route that can consume it. Numeric, low-cardinality
   categorical, and categorical-only centered fits must not retain an unused dense duplicate.
3. For a mixed design with a native `CategoricalMatrix`, use one normalized `standardize` call as
   a cheap location/scale preflight, then compute raw Gram and RHS with `sandwich` and
   `transpose_matvec`.
4. Pass raw moments through SuperGLM's existing certification as well. A rejection permanently
   locks that inner fit to stable chunks; accepted changing-weight iterations remain certified.
5. Before compiled weighted calls, provide float64, C-contiguous, writable weight buffers. Tabmat
   4.2.1 can silently miscompute with strided weights and rejects read-only sandwich weights;
   read-only predictor storage itself is fine.
6. Retain stable, bounded, explicitly centered chunks as the fallback for ill-scaled inputs or
   unsupported components.
7. Do not route small all-dense systems through Tabmat merely because a split exists; the frozen
   fixture shows that this is slightly slower than the existing dense path.
8. Never infer real Tabmat use from construction alone. Regression tests and benchmarks must count
   `sandwich` and `transpose_matvec` calls on the timed fit.

`discrete=True` remains a hybrid path. BAM-style `B_unique`, bin indices, tensor grids, and
coefficient transforms stay compressed and use specialized aggregation kernels. A future partial
plan can put eligible observation-level dense/sparse/categorical blocks in Tabmat and assemble
their cross-moments with discrete blocks in coefficient space. Materializing a discrete basis only
to pass it to Tabmat lost to the existing aggregated kernel (1.629 ms versus 0.958 ms on the tested
50,000-row mixed fixture), so wholesale conversion is not the target architecture.

## Current measured opportunity

On the frozen 6,000-row mixed numeric/high-cardinality-categorical fixture, the current centered
fallback materializes a 6,000 by 160 design. A controlled Tabmat substitution produced:

- matrix system: 11.496 ms to 0.504 ms (22.79 times faster);
- matrix-stage traced peak: 16.08 MB to 0.923 MB (94.3% lower);
- full fit: 70.964 ms to 17.286 ms (4.11 times faster);
- full-fit traced peak: 18.13 MB to 4.39 MB (75.8% lower).

Maximum coefficient and prediction differences were `3.21e-14` and `1.71e-14`; deviance and
iteration count were unchanged.

## Follow-up experiments

- Do not use `StandardizedMatrix.sandwich` to replace stable centering: large-offset experiments
  produced severe Gram error. Its mean/scale summary is a preflight only.
- Use `sandwich(cols=...)` for genuinely selected systems; it reduced work and memory substantially.
  Column-restricted `transpose_matvec` was slower in the tested cases. `rows=` is promising only
  when the active row fraction is genuinely small.
- Benchmark native low-cardinality categorical blocks behind a size crossover. They won strongly
  at 60,000 rows but lost on the small case, so unconditional conversion is not justified.
- Compare direct group construction with `from_df` for copy count and build time, but preserve
  SuperGLM's feature transformations, identifiability projections, coefficient ordering, and
  prediction metadata.
- Re-run determinism tests with default and fixed thread counts. Performance changes must preserve
  the existing numerical envelope and all convergence decisions. Fixed single-thread benchmark
  results are not guarantees under default OpenMP settings; small problems can be oversubscribed.
