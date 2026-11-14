# Compiled Kernels

This page is for contributors. It walks through superglm's compiled layer:
what each family of Numba kernels computes, which solver step calls it, and
how the on-disk cache and `superglm.warmup()` work. The user-facing summary is
in [How To Think About REML and the Solvers](../../explanation/solvers-and-internals.md),
section 17.6.

The kernels add no mathematics of their own. Each one computes a quantity the
solver pages already define: a weighted Gram block, a working right-hand
side, a family's log-likelihood and its derivatives, a compensated sum. They
exist because those quantities are loops over $n$ rows, and a Python loop or a
chain of NumPy temporaries over a book of a million rows is too slow or too
large.

## How the layer is built

Every kernel is a module-level function decorated with `numba.njit`, in
nopython mode, with `cache=True` so Numba saves the compiled code to its
on-disk cache. A few conventions hold across the package:

- **Parallelism stays in Python.** No kernel uses `prange`. Where work runs in
  parallel, a Python thread pool calls serial kernels compiled with
  `nogil=True`. The main pool is the block queue
  (`src/superglm/_group_matrix/_block_queue.py`), which forms the blocks of
  $\mathbf{X}^\top\mathbf{W}\mathbf{X}$ on worker threads, largest first, as
  Li and Wood (*Statistics and Computing* 30:19, 2020, section 4) describe.
  SuperLSS row passes and the data rank's TSQR leaves run the same way.
- **Fixed summation order.** A kernel visits rows and columns in one fixed
  order and writes only arrays it owns or scratch its caller gives it, never
  shared between workers. A result therefore does not depend on the thread
  count. Several structured-solver kernels multiply small matrices with their
  own loops instead of calling BLAS for the same reason.
- **One signature per kernel.** Callers pass operands in one form, usually a
  read-only C-contiguous float64 array (the group-matrix module's `_native`
  helper), so each kernel compiles once rather than once per array layout.
  `compile_closed` in `src/superglm/_numba_compile.py` compiles a kernel's
  signature and then closes the dispatcher to others; a call with another
  layout is converted, not compiled again.
- **Strict IEEE where it matters.** Kernels that build exact or compensated
  sums are compiled with `fastmath=False`, since reassociation would destroy
  the error-free transformations they rely on. Kernels that should return
  `inf` or `nan` rather than raise use `error_model="numpy"`.
- **Failures are flagged, not raised.** A row a kernel cannot represent in
  float64 comes back marked (a status code or a non-finite entry), and the
  Python caller replays it on a slower route or refuses it with a message.

## Kernel families

| Family | Where | Computes | Called from |
|---|---|---|---|
| Group-matrix kernels | `src/superglm/_group_matrix/_group_matrix_kernels.py` | weighted bincounts and 2-D histograms, CSR Grams and cross-Grams, factor-smooth products, centred row blocks | every IRLS step's weighted moments, and the design's `matvec` / `rmatvec` |
| Lookup-table passes | `src/superglm/_group_matrix/_lookup_tables.py` | Grams, leverages and $\mathbf{X}^\top\mathbf{v}$ of a design made only of table lookups and dense columns | the centred system, REML's weight-derivative pass, final REML bookkeeping, SuperLSS binned rows |
| Design-build kernels | `_group_matrix_support.py`, `src/superglm/features/interaction.py`, `src/superglm/features/_basis_kernels.py` | lossless row-support detection, row-wise Kronecker products for tensors, dense spline rows for large predictions | design construction and `predict()` |
| Structured solver | `src/superglm/solvers/_structured/` | block-eliminated factors for random-effect chains and `fs` / `sz` factor smooths | the structured step of direct IRLS, and REML's cached solves on such models |
| Exact and compensated sums | `src/superglm/solvers/_exact_sums.py`, `src/superglm/reml/_compensated.py` | correctly rounded weighted sums; compensated dot products and quadratic forms | the penalized-mode score and intercept, multi-penalty log-determinants, SuperLSS penalty quadratics |
| REML leverage rows | `src/superglm/reml/_leverage_rows.py` | solved rows $\mathbf{H}^{-1}\mathbf{x}_r$ and leverages $h_r$ | the $W(\boldsymbol{\rho})$ correction's leverage route |
| GLM family kernels | `src/superglm/_tweedie_series.py`, `src/superglm/_nb2_deviance.py` | the Tweedie density series; the NB2 unit deviance | Tweedie log-likelihood, dispersion and `estimate_p`; NB2 deviance |
| SuperLSS family kernels | `src/superglm/distributional/kernels/` | per-row log-likelihood, score, Hessian and higher derivatives of each distributional family; window probabilities for truncated and censored rows | every SuperLSS iteration's row evaluation |
| SuperLSS assembly | `src/superglm/distributional/solver/` (`_joint_tables.py`, `_ordinary_packing.py`, `derivatives.py`) | joint Gram tables over shared index columns, packed row panels, the chain rule from natural to predictor derivatives | SuperLSS Newton and REML steps |
| Plotting | `src/superglm/plotting/diagnostics.py` | LOWESS smoothing | diagnostic plots only |

### Group-matrix kernels

Each group of columns in a design has a compact form: a categorical is a code
per row, a discretised spline is a bin code per row plus a small basis
evaluated at the bins, a sparse spline is a CSR basis, a tensor is two such
codes. The kernels compute weighted products in those compact forms without
building the $n \times p$ design or a temporary like `W[:, None] * M`.

- **One-dimensional aggregation.** `_fused_bincount_2` sums $W$ and $Wz$ by bin
  in one pass, the right-hand side of a discretised group.
  `_weighted_bincount_2d`, `_csr_weighted_bincount` and their row, level and
  support variants aggregate a dense or CSR block by bin with weights.
  `_level_sums` and `_cat_weighted_bincount` do the same for category codes.
- **Gram and cross-Gram blocks.** `_csr_weighted_gram` and
  `_csr_weighted_cross` form $\mathbf{B}^\top\mathbf{W}\mathbf{B}$ and
  $\mathbf{B}^\top\mathbf{W}\mathbf{C}$ for CSR blocks over the same rows.
  `_disc_disc_2d_hist` builds the weighted 2-D histogram of two discretised
  groups' codes, $T_{ab} = \sum_{r: k_1(r) = a,\, k_2(r) = b} W_r$, from which
  the cross-Gram is $\mathbf{A}_1^\top \mathbf{T} \mathbf{A}_2$ in coefficient
  space (Li and Wood 2020, Algorithm 1). `_cat_cat_weighted_crosstab` is the
  same for two categoricals. The `_channels` variants form several weight
  vectors at once, which REML's derivative Grams and SuperLSS's curvature
  channels need.
- **Tensors.** `_cell_csr` sorts rows by grid cell with a stable counting
  sort, and `_cell_hist_raw_kron` forms a tensor's Gram from the raw marginal
  bands cell by cell, so the Kronecker rows are never stored. The block queue
  can split a large tensor block into runs of cells that sum into disjoint
  rows of the histogram, so the split block is bitwise the whole one.
- **Factor smooths.** The `_factor_smooth_*` kernels apply and transpose a
  spline-by-factor term level by level, and fuse its per-level local Grams
  with its two transpose products in one pass.
- **Dense Schur blocks.** `_dense_small_weighted_moments` fuses
  $\mathbf{X}^\top\mathbf{W}\mathbf{X}$, $\mathbf{X}^\top\mathbf{W}$ and
  $\mathbf{X}^\top\mathbf{W}\mathbf{z}$ for a narrow dense block.
- **Centred rows and projections.** `_weighted_centred_rows` and
  `_weighted_centred_one_hot` write a block of centred, weight-scaled rows
  for a TSQR leaf (the QR solve option and the data rank).
  `_gather_table_rows` and `_scatter_table_rows` project rows onto the data
  rank's few grey directions without forming the rows in full.
- **Range scans.** `_float64_operand_exponent_bounds` and its relatives bound
  an operand's binary exponents in one pass, without scratch arrays, so a
  caller can decide whether reassociating a tensor product stays inside the
  normal float64 range.

`MatrixExecutionPlan.moments`, in
`src/superglm/_group_matrix/_group_matrix_execution.py`, is the orchestrator.
Compressed groups (discretised and support-compressed splines) stay in their
compact form and use the kernels above. Only a design made entirely of
ordinary groups (dense, sparse, categorical) can be handed to tabmat, and by
default only in a narrow case: one categorical with more than 100 levels, at
least three dense columns, no sparse group and at least 50,000 rows (see
[tabmat integration notes](tabmat-integration-notes.md)). The plan returns $\mathbf{X}^\top\mathbf{W}\mathbf{X}$, $\mathbf{X}^\top\mathbf{W}$
and $\mathbf{X}^\top\mathbf{W}\mathbf{z}$ block by block. Direct IRLS calls
it once per iteration; the fREML line search does not call it at all, since
it reuses the cached summaries.

### Lookup-table passes

When every group of a design is a lookup (row $r$ of the group is row
$k_t(r)$ of a small table $\mathbf{A}_t$) or a dense column, each row pass
reduces to sweeps over the index columns:

- **Gram.** Per pair of terms, the weighted 2-D histogram of their index
  columns (`_hist_sweep`), then $\mathbf{A}_t^\top \mathbf{T}_{tu} \mathbf{A}_u$
  in coefficient space (Li and Wood 2020, Algorithm 1). A pair whose
  histogram would be too large accumulates
  $\mathbf{M}[k_t] \mathrel{+}= W_r \mathbf{A}_u[k_u(r)]$ over the nonzero
  window of the row instead (`_acc_sweep`; their Algorithms 2 and 3).
- **Leverages.** $h_r = \mathbf{x}_r^\top\mathbf{V}\mathbf{x}_r$ gathered per
  row from per-pair tables $\mathbf{A}_t\mathbf{V}_{tu}\mathbf{A}_u^\top$
  (`_gather_leverages`), the scattered-row form of the GLAM identity for
  $\operatorname{diag}(\mathbf{X}\mathbf{S}\mathbf{X}^\top)$ (Currie, Durban
  and Eilers, *JRSS B* 68:259, 2006, equation 2.7).
- **Transpose products** $\mathbf{X}_c^\top\mathbf{v}$: one 1-D histogram per
  term (`_scatter_margins`, with Neumaier-compensated dense sums).

The plan (which pairs histogram and which accumulate) depends on the design
alone. It is held on the design for one fit and released when the fit ends.

### Design-build kernels

- `_csr_row_hashes` and `_csr_rows_match` in
  `src/superglm/_group_matrix/_group_matrix_support.py` find a spline group's
  distinct rows bit for bit, so a group with few distinct rows can be stored
  as a support table plus a code per row with no loss.
- `_row_kron_counts` and `_row_kron_fill` in
  `src/superglm/features/interaction.py` build the row-wise Kronecker product
  of two CSR bases, storing nonzero products only.
- `_bspline_rows` and `_cardinal_rows` in
  `src/superglm/features/_basis_kernels.py` evaluate dense B-spline and
  cardinal cubic rows, releasing the GIL, for a prediction of 8,192 rows or
  more. A smaller prediction uses NumPy and compiles nothing, so a quoting
  service needs no warmup.

### Structured solver

`direct_solve="auto"` sends large random-effect and factor-smooth terms to
block elimination instead of a dense Gram (section 6.1 of the explanation
page). Its kernels are in `src/superglm/solvers/_structured/`:

- **`balance_tree.py`, factor smooths with `basis="sz"`.** A sum-to-zero
  factor smooth is a least-squares problem with dense equality constraints.
  The factorization uses the null-space method (Nocedal and Wright 2006,
  section 16.2) in a balance (Haar) basis of the levels (Egozcue and
  Pawlowsky-Glahn 2005) and eliminates up the tree by stacked Householder QR
  (Paige and Saunders 1977; TSQR, Demmel et al. 2008). The kernels are the
  Householder reflections (`_reflect`, `_reflect_signed`), the leaf and merge
  passes (`_leaf_exports`, `_fisher_sweep`, `_signed_sweep`), the top-down
  solve (`_down_sweep`) and the trace and quadratic-form pieces REML needs.
- **`block_leaves.py`, factor smooths with `basis="fs"`.** Each level is a
  leaf. One fixed-order pass over the rows in level order reduces each level's
  rows by Householder QR (`_house_qr`, `_stacked_qr`, `_merge_level`), so no
  stack of per-level $p \times p$ Grams is formed. Rows with signed weights,
  from the observed information, use the hyperbolic QR of Chandrasekaran,
  Gu and Sayed as stated by Bojanczyk, Higham and Patel (*SIAM J. Matrix
  Anal. Appl.* 24, 2003). Per smoothing-parameter trial, one small QR per
  level with the penalty's square root inside (`_fisher_trials`,
  `_signed_trials`) gives the leaf factors without subtracting data-sized
  moments.
- **`leaf_kernels.py`, nested random-effect chains.** The row kernels of the
  chain's centred leaf pass: forming centred rows from tables, one-hot codes
  or sparse rows (`_coded_rows`, `_sparse_rows`), segment means about a
  reference row (`_segment_means`, `_center_rows`), and each level's count of
  distinct basis rows of positive weight, stopped at a cap
  (`_distinct_level_rows`), or its first such rows
  (`_first_distinct_level_rows`), which the structured layout and the
  separation scan read.

These run once per IRLS iteration on such a model, and again per line-search
trial in REML's cached solves.

### Exact and compensated sums

Some sums decide something a user sees, such as whether the penalized score
is zero to working accuracy, and cancel badly in plain float64.

- **`src/superglm/solvers/_exact_sums.py`** forms
  $\sum_r w_r (y_r - o_r - \alpha - t_r)$ and similar weighted sums exactly:
  each product is split by TwoProduct, each addition by TwoSum, and the
  partials are kept as a non-overlapping expansion (Shewchuk's grow-expansion,
  as `math.fsum` does) and rounded once at the end. A sum that would overflow
  is scaled by a power of two first. The penalized-mode score in
  `src/superglm/solvers/mode_score.py` uses these for the centred mean and
  the intercept's remainder, which the REML stop rule and the observed-REML
  certificate read.
- **`src/superglm/reml/_compensated.py`** evaluates Dot2 (Ogita, Rump and
  Oishi, *SIAM J. Sci. Comput.* 26(6), 2005), a dot product as accurate as if
  computed in twice the working precision and then rounded, and quadratic
  forms $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ built from it. Multi-penalty
  log-determinants (`src/superglm/reml/multi_penalty.py`) and the SuperLSS
  solver's penalty quadratics use it. A product outside the normal range
  returns to the Python caller's fallback.

### REML leverage rows

The $W(\boldsymbol{\rho})$ correction's gradient term is
$\tfrac12\sum_r d_{jr} h_r$, with $h_r$ row $r$'s leverage (Wood 2011,
section 3.5.1). `_solved_rows_and_leverages` in
`src/superglm/reml/_leverage_rows.py` forms each centred row's solved row
$\mathbf{H}^{-1}\mathbf{x}_r$ from per-group tables and the dense columns,
and its leverage, in one pass, in pieces so the $n \times p$ solved rows are
never all held at once. Direct REML takes this route with four or more
penalties once the derivative-Gram route would cost more (see section 13.1 of
the explanation page).

### GLM family kernels

- **`src/superglm/_tweedie_series.py`** evaluates the Dunn and Smyth (2005)
  series for the Tweedie density with $1 < p < 2$. Every term is positive, so
  the kernel starts at the largest term and sums outward until the terms fall
  far enough below it, returning $\log W$ and the first two moments of the
  series index per row. `src/superglm/_tweedie.py` uses it for the
  log-likelihood, the dispersion profile and `estimate_p`'s power search.
- **`src/superglm/_nb2_deviance.py`** forms the negative binomial (NB2) unit
  deviance in two vectorised passes around NumPy's SIMD logarithm. Each row
  takes one of three formulas: a closed form at $y = 0$, a series near
  $y = \mu$ where the direct formula would cancel, and a rearranged
  logarithm elsewhere, with a scalar repair pass for rows whose quotient is
  not a finite normal number.

### SuperLSS family kernels

Each distributional family in `src/superglm/distributional/kernels/` has a
row kernel that returns, for every row, the log-likelihood and its
derivatives in the family's natural coordinates: score and Hessian on every
Newton step, third derivatives for the REML outer Hessian, and fourth
derivatives where the REML criterion needs them at a smoothing-parameter
endpoint. Examples:

- `_tweedie_numba.py`: the Tweedie LSS point kernel, including derivatives in
  the power $p$, on the Dunn and Smyth series.
- `_negative_binomial_numba.py`: NB2 rows, with a series near the Poisson
  limit where the size parameter is large, and `_negative_binomial_crps.py`
  for the CRPS at zero by quadrature.
- `_generalized_gamma_numba.py`, `_gg_log_tail.py`, `_gg_uniform_tail.py`: the
  Prentice (1974) generalized gamma and its log tails, with series near its
  log-normal limit.
- `_incomplete_gamma.py`, `_incomplete_beta.py`, `_pareto_tail.py`: log tail
  probabilities with their partial derivatives, for truncated and censored
  rows (`_response_tails.py` packages them).
- `_dirichlet_numba.py`: the Dirichlet density and its directional
  derivatives for compositional responses.
- `_declared/`: kernels generated by `tools/generate_severity_families.py`
  from each severity law's declaration (density, log survival, log CDF);
  their generated modules say not to edit them by hand. See
  [distributional family development](distributional-family-development.md).
- `_severity_primitives.py`: shared scalar pieces (polygamma functions,
  $\log\Phi$ and its derivatives, stable `softplus` and `log1mexp`).

In `src/superglm/distributional/solver/`, `_joint_tables.py` forms the joint
Gram of a location-scale model: its predictors share covariates, so many
blocks need the same pair of index columns with different weights, and one
row pass forms all of them (Li and Wood 2020, Algorithm 1, over several
channels at once). `derivatives.py` applies the chain rule from natural to
predictor derivatives row by row, and `_ordinary_packing.py` packs a
predictor's intercept, dense and small categorical groups into one row panel.

## What a first fit compiles

A kernel compiles the first time a process calls it with a new signature,
unless Numba finds it in the cache. A first fit compiles only the kernels its
route reaches: a Poisson fit with discretised splines never compiles the
Dirichlet kernels. That is why a first fit in an empty environment takes a
few seconds longer than the second, and why the extra time depends on the
family and the terms.

`src/superglm/_numba_compile.py` also listens for the end of a superglm
compile. Numba's type inference keeps reference cycles that pin the frames on
the stack while it compiled, so the next `fs` leaf build runs one full garbage
collection after a compile, freeing those frames' arrays. A warm cache never
compiles and never collects.

## The on-disk cache

Numba saves each compiled kernel to a cache directory: `NUMBA_CACHE_DIR` when
it is set, otherwise a `__pycache__` next to the source file, otherwise the
user's cache directory (Numba documentation, *Notes on caching* and
*Environment variables*). An entry is keyed to the source file and to the
CPU model, so a cache built on one node type compiles again on another.

A kernel declared `cache=True` stops its module's import if no cache location
can be written, which is every process on a read-only file system whose only
writable place is the temporary directory (AWS Lambda, a read-only container
root). `src/superglm/_numba_cache.py` runs when superglm is imported and, in
that one case, points Numba at a new private directory in the temporary
directory (mode 0o700, removed when the process exits). Numba loads its cache
with `pickle`, so a directory another user could write into is never used. A
location the user chose (`NUMBA_CACHE_DIR`, `NUMBA_CACHE_LOCATOR_CLASSES`,
`numba.config.CACHE_DIR` set in code, or a `.numba_config.yaml`) is never
changed.

For job clusters, keep the cache on persistent storage and see
[Starting fast on ephemeral clusters](../../how-to/start-fast-on-ephemeral-clusters.md).

## `superglm.warmup()`

`warmup()` compiles the kernels of every family and route ahead of the first
fit. Its stages are listed in `_STAGES` in `src/superglm/_warmup.py`: each
names a module's own warm-up function, which calls its kernels on tiny inputs
with the operand forms the fits pass.

Numba compiles under one lock per process, so threads cannot share the work;
processes can. `warmup()` therefore:

1. groups the stages into units and runs each unit in a fresh interpreter, at
   most `n_jobs` at a time (`SUPERGLM_N_JOBS`, the physical cores by default,
   and at most one per 600 MB of `SUPERGLM_MAX_MEMORY`), longest unit first,
   each worker saving its kernels to the cache;
2. then runs every stage in this process in the original order, which loads
   those kernels from the cache and compiles whatever a worker did not leave.

One unit per process keeps the set of compiled signatures the same on every
run: within one process, Numba serves a call from an already loaded overload
whose types it can convert to, which would hide a missing signature. With one
worker, or an interpreter that cannot start workers, every stage runs in this
process alone.

Numba's cache index is a read-modify-write of one file per function. Two
processes saving different signatures of one kernel at once can both claim
the same data file, and Numba publishes the index before the data file it
names. While workers run, every save and load therefore takes a lock file in
the cache directory, which other warmups writing the same cache (cluster
nodes sharing `NUMBA_CACHE_DIR`) take as well, and a save writes its data file
before the index that names it. Where the lock cannot be taken, a save is
skipped and the kernel compiles again later. This save routine is adapted
from Numba's own and carries Numba's BSD licence notice in the source.

A few kernels are left to compile on first use, because few fits reach them:
the third and fourth derivatives a SuperLSS fit needs only at a
smoothing-parameter endpoint, a fit's diagnostics, and some rarely run
fallbacks.

## Adding a kernel

- Decorate a module-level function with `njit(cache=True)`. Add `nogil=True`
  if the block queue or another thread pool will run it, and keep its loops
  serial and in a fixed order.
- Pass it operands in one form, so it compiles one signature, and close it
  with `compile_closed` if callers hold several layouts.
- Call it from its module's warm-up function with the operand forms the fits
  pass, or add it with a reason to `_FIRST_USE` in
  `tests/test_import_compat.py`.
  `test_public_warmup_compiles_every_kernel_but_the_first_use_allowlist`
  finds every dispatcher in the package (`tests/_numba_dispatchers.py`) and
  fails on a kernel that is in neither place, or on a listed one that warmup
  now compiles.
- A new unit needs an entry in `_UNIT_SECONDS`, which sets only the order
  workers start units in.
- If the kernel changes how long a cold compile takes, check the counts that
  `tests/test_compile_budget.py` pins.
