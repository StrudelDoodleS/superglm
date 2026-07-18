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
3. For an eligible mixed ordinary/discretized design, use one normalized `standardize` call on
   the first attempted iteration as a cheap full-augmented location/scale preflight. Reuse its
   call-local indicator means for compressed-support screening and `X'W`.
4. On every accepted mixed iteration, compute the raw Gram with `sandwich` and the RHS with one
   `transpose_matvec`. Later iterations derive one-hot `X'W` from the sandwich diagonal and dense
   `X'W` from the bounded dense slab, without repeating standardization.
5. Pass raw moments through SuperGLM's existing certification as well. A rejection permanently
   locks that inner fit to stable chunks; accepted changing-weight iterations remain certified.
6. Before compiled weighted calls, provide float64, C-contiguous, writable weight buffers. Tabmat
   4.2.1 can silently miscompute with strided weights and rejects read-only sandwich weights;
   read-only predictor storage itself is fine.
7. Retain stable, bounded, explicitly centered chunks as the fallback for ill-scaled inputs or
   unsupported components.
8. Do not route small all-dense systems through Tabmat merely because a split exists; the frozen
   fixture shows that this is slightly slower than the existing dense path.
9. Never infer real Tabmat use from construction alone. Regression tests and benchmarks must count
   `sandwich` and `transpose_matvec` calls on the timed fit.
10. Mixed native categoricals with at most 100 retained levels require at least 5,000 rows. The
    measured 2,000-row setup cell lost, while the 5,000-row cell won; high-cardinality blocks keep
    the general scaled-work gate because their stable dense fallback is already expensive.
11. Reject a mixed plan before construction when its retained arrays, constructor peak, or
    per-moment live temporaries exceed the 64 MiB aggregate bounds. Category metadata and temporary
    code copies are part of those estimates, not assumed to fit in incidental index slack.

For repeated raw weighted moments, automatic Tabmat dispatch is narrower than centered-system
dispatch. The currently certified layout is a wholly ordinary design with exactly one native
categorical block wider than 100 levels, no sparse block, at least three dense columns in total,
and at least 50,000 rows. Explicit internal forcing remains available for controlled experiments.
The narrow rule matters: one dense column, two separately stored dense columns, sparse mixtures,
and multiple categorical blocks all produced repeatable counterexamples.

`discrete=True` remains a hybrid path. BAM-style `B_unique`, bin indices, tensor grids, and
coefficient transforms stay compressed. Eligible mixed ordinary/discretized centered systems now
  have a cached public-Tabmat bin-space plan: each spline contributes native bin indicators to one
  `SplitMatrix`, and only the bounded bin-space moments are transformed through cached supports.
  Packed all-discrete and unsupported tensor/sparse layouts retain their specialized paths. Frozen
  controller benchmarks accepted the route after measuring 4.1% faster CPU for a 10k mixed fit,
  11.6% for a 10k four-spline fit, and 76.7% for a 60k high-cardinality fit; the latter reduced the
  cold RSS delta from 65.2 to 24.4 MiB. Wholesale row-level materialization remains outside the
  architecture.

## Current measured opportunity

On the frozen 6,000-row mixed numeric/high-cardinality-categorical fixture, the current centered
fallback materializes a 6,000 by 160 design. A controlled Tabmat substitution produced:

- matrix system: 11.496 ms to 0.504 ms (22.79 times faster);
- matrix-stage traced peak: 16.08 MB to 0.923 MB (94.3% lower);
- full fit: 70.964 ms to 17.286 ms (4.11 times faster);
- full-fit traced peak: 18.13 MB to 4.39 MB (75.8% lower).

Maximum coefficient and prediction differences were `3.21e-14` and `1.71e-14`; deviance and
iteration count were unchanged.

The unified weighted-moment execution plan was also measured against the former specialized loops
with one thread, CPU affinity, counterbalanced ordering, and five rounds per cell. Specialized-loop
comparisons were bit-exact; Tabmat's different summation order stayed within `1.9e-10`. For three
dense columns plus a 120/160-level categorical block, Tabmat reduced raw-moment
time at 50,000 rows by 35% for Gram-plus-RHS, 42% for Gram alone, and 39% for signed Gram. At
100,000 rows the reductions were 42%, 47%, and 44%, respectively. The non-Tabmat compressed
execution path stayed within 1.3% of the former loops in every prevalidated RHS and signed-Gram
cell from 2,000 by 19 through 60,000 by 179. Full-vector validation remains on untrusted and
derived REML inputs; closely timed end-to-end runs showed no numerical drift and no non-categorical
case outside the 3% routine-fit noise gate, while the categorical fit retained its large time and
memory reduction.

## Follow-up experiments

- Do not use `StandardizedMatrix.sandwich` to replace stable centering: large-offset experiments
  produced severe Gram error. Its mean/scale summary is a preflight only.
- Use `sandwich(cols=...)` for genuinely selected systems; it reduced work and memory substantially.
  Column-restricted `transpose_matvec` was slower in the tested cases. `rows=` is promising only
  when the active row fraction is genuinely small.
- Re-measure the 5,000-row native low-cardinality crossover when Tabmat's categorical kernels or
  constructor ownership change; the active rule is deliberately evidence-bound.
- Compare direct group construction with `from_df` for copy count and build time, but preserve
  SuperGLM's feature transformations, identifiability projections, coefficient ordering, and
  prediction metadata.
- Re-run determinism tests with default and fixed thread counts. Performance changes must preserve
  the existing numerical envelope and all convergence decisions. Fixed single-thread benchmark
  results are not guarantees under default OpenMP settings; small problems can be oversubscribed.
