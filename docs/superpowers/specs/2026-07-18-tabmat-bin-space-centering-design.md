# Tabmat Bin-Space Centering Design

## Goal

Accelerate centered IRLS systems for mixed ordinary and discretized-spline designs without
materializing spline rows, paying Numba's first-use cost, or weakening the existing raw-moment
safety certificate.

## Evidence and selected approach

Three approaches were measured:

1. **Cached Tabmat bin-space plan (selected).** Represent every eligible discretized spline's
   observation-to-bin map as a `tabmat.CategoricalMatrix`, combine those compact blocks with the
   ordinary Tabmat blocks, and obtain all raw moments from one `SplitMatrix.sandwich` plus two
   `transpose_matvec` calls. Transform only bin-space blocks through `B_unique @ R_inv`.
2. **Replace Numba scatter kernels with NumPy.** This removes roughly 100--220 ms and 45--59 MiB
   of first-use overhead, but measured hot regressions reach 13--21% for complete centered systems
   and are much larger for wide dense aggregation.
3. **Raise the mixed-route size floor.** This avoids the worst small cold regressions but leaves
   the fixed initialization and memory cost in larger first fits and gives up verified hot wins.

The selected prototype used only public Tabmat operations, matched dense raw moments within the
existing floating-point envelope, and reduced a 10,000-row one-spline moment call from about
0.93 ms to 0.42 ms while avoiding Numba initialization.

## Architecture

Add one focused private plan class under `src/superglm/_group_matrix/`. The immutable plan owns:

- one lazily cached `tabmat.SplitMatrix` whose logical columns follow solver group order;
- ordinary augmented-column and solver-column mappings;
- one immutable descriptor per compressed spline containing its augmented bin slice, solver
  slice, and support transform `B_unique @ R_inv`.

Each `DiscretizedSSPGroupMatrix` is represented in Tabmat by an integer-code
`CategoricalMatrix` with all bins retained. Ordinary categorical predictors are also represented
natively at every eligible cardinality, including the dropped reference level, so this route
never creates a dense one-hot duplicate. Tabmat 4.2.1 may coalesce separately stored dense numeric
components into a bounded ordinary-only dense block; it must never include expanded spline or
one-hot categorical columns.

Tensor, sparse, SCOP, spline-categorical, multiple ordinary categorical, and otherwise unsupported
layouts remain structurally unattempted. `DiscretizedTensorGroupMatrix` is excluded by exact type,
not subclass checks.

## Data flow

For each changed IRLS weight vector:

1. Check the existing scaled work floor and supported topology.
2. Use virtual Tabmat standardization for ordinary location/scale summaries and bounded support
   summaries for each compressed spline. Discard the virtual standardized wrapper.
3. Compute the augmented raw Gram with `SplitMatrix.sandwich(W)` and augmented `X'W`/`X'Wz`
   with two `transpose_matvec` calls.
4. Copy the whole ordinary-ordinary block once. For each spline, transform its diagonal,
   ordinary cross block, vectors, and spline-spline cross blocks through its support transform.
5. Pass the solver-space raw moments through `_certify_raw_centering` unchanged.
6. On any numerical rejection, lock the fit-local centering state to the stable bounded fallback.

No accepted mixed call may invoke SuperGLM's Numba aggregation kernels, call a group `toarray`, or
construct an observation-by-full-design array.

## Cache lifecycle and errors

`DesignMatrix` caches the plan for reuse across IRLS iterations. Pickling deliberately clears the
cache, matching the existing execution-plan and Tabmat cache lifecycle. Unsupported topology is
not a numerical rejection and therefore must not mutate the fit-local tri-state. Invalid or unsafe
moments return the existing numerical rejection result and permanently select stable chunks for
that inner fit.

## Correctness and performance acceptance

- Dense-reference parity for ordinary plus one/multiple splines, native low/high-cardinality
  categoricals, exact aliases, and weight vectors containing zeros.
- Tensor and unsupported layouts remain unattempted.
- Unsafe large-offset layouts fall back and remain locked out.
- Accepted calls execute one Tabmat sandwich and two transpose products, with zero calls to the
  named Numba scatter kernels and zero row materializations.
- Cache identity is stable within a fit and reset by pickle.
- Focused tests, rank-policy tests, execution-plan tests, direct IRLS tests, Ruff, and diff checks
  pass.
- Cold/warm/RSS benchmarks cover 10k/60k canonical layouts, fragmented dense blocks, multiple
  splines, low/high-cardinality categoricals, and the frozen origin baseline. No accepted measured
  topology may regress full-fit time by more than 3%; transient and retained memory are reported.
