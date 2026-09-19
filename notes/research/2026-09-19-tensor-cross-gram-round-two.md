# The tensor cross-Gram in the raw basis, and its cell-parallel accumulation

Two commits cut the constant in front of the quadratic block count again; a
third was measured, failed its pre-registered rule and is reverted by the
review-fix commit that follows this note on the branch. On the ten-pair
300,000-row synthetic design, the default-mode `fit_reml` to convergence at
four threads fell from `48.53 s` to `23.64 s` (2.05x) and its Gram time from
`35.21 s` to `13.53 s` (2.60x) on the tree that still carried the parallel
kernel, reaching the same 13 lambda states, the same 11 outer iterations,
the same termination reason and an objective of `79587.18132998943` against
`79587.181329993`, 4.5e-14 relative; the settled tree without that kernel is
the S1 row of the attribution table below (Gram 2.25x per state at four
threads), and the Bench phase re-measures the whole fit on it. Per
production block at one thread, the stored-row gather's `48.90 ms` becomes
`17.30 ms` in the raw band (2.83x best of five; 2.17-2.27x on interleaved
medians, see the S1 rule). The zero-pair and one-pair fits are bit-identical
to the round-one tree in every recorded quantity. S1 (the raw basis) carries
the win: at ten pairs and four threads S4 alone is 1.10x and S1 takes the
cumulative figure to 2.25x; S2's parallel kernel added 1.12x at a pinned
pool and lost at the default one. The block count is still `M(M-1)/2`;
again, only its constant moved.

This note records the measurements for `32e0dc8e` (S4, the cached row
structure), `d885f5b8` (S1, the raw B-spline basis) and `90defad0` (S2, the
cell-partitioned parallel accumulation) on `perf/interaction-cost-round-two`,
whose base `e130313d` is v0.34.0 plus round one's discrete tensor step fix
and channel cross-Gram. S2's measurements stay as the record of its rule's
failure; the kernel, its dispatch and its tests are gone from the branch.

## Characterisation and literature

The object is unchanged from round one. A tensor-by-tensor cross block is
`S_i^T diag(W) P_j B_j`, where `S_i` and `P_j` are one-hot row-partition
matrices — row to grid cell of tensor `i`, row to support row of tensor `j`
— fixed for the whole fit, with only `W` changing per iteration. That is a
weighted sparse accumulation with a dense payload: a scatter-add of `P_j`
doubles into one of `n1 * n2` accumulator rows, memory-bound rather than
arithmetic-bound.

Round one's per-pass profile of the ten-pair fit on `e130313d` located the
cost precisely: of `28.18 s` wall over seven states, `X.cross_gram_total`
was `22.47 s` (80%), `X.tensor_tensor_channel` `20.36 s` across 630 calls,
and the dense channel kernel at the production width,
`K.hist_channels[P=81]`, `17.40 s` across 630 calls at `27.6 ms` each. A
bandwidth probe on the same machine put a sequential copy at `35.91 GB/s`
and a random gather of 300,000 rows of 81 doubles at `3.56 GB/s`, with
stage 1 in natural row order achieving `14.00 GB/s` and `1.75 Gflop/s`. The
pass is bound by dependent random reads, not by flops.

Four sources bear on the remedy, and each assumes something this code must
be checked against.

**Li and Wood (2020), *Stat. Comput.* 30:19-25,
doi:10.1007/s11222-019-09864-2.** Algorithms 2 and 3: accumulate the compact
rows of one term by the discretised index of the other, then contract. This
is the route round one adopted. It assumes the compact row is what you
accumulate. That assumption is what S1 relaxes — the compact row is itself a
fixed linear image of a narrower one.

**Currie, Durban and Eilers (2006), *JRSS-B* 68:259-280,
doi:10.1111/j.1467-9868.2006.00543.x**, generalized linear array models.
Assumes the data sit on a full grid, so that the row sum factors into
marginal sums. That holds for the single-tensor Gram (already factored) and
for the shared-margin three-way case (the existing helper, capped at
5,000,000 cells). Two tensors on four distinct margins put the joint grid at
`256^4`, so the assumption fails and the row pass is cheaper than the array;
this remains not a lever.

**Propagation blocking (Gu, Beamer et al., arXiv:2002.11302) and MAGNUS
(Wolfson-Pou et al., arXiv:2501.07056).** Both generate locality for SpGEMM
and sparse accumulation by partitioning the accumulation targets into
cache-sized bins before accumulating. Both assume the reordering is worth
its own pass — that the accumulation dominates the permutation. Here the
degenerate one-bin-per-cell form is what S1's cell order and S2's chunking
implement; the assumption was checked directly, and the permutation
prologue costs `2.3-2.9 ms` against a kernel of `6.4-6.7 ms` serial, so it
pays, but only just, and only because the cell-CSR itself is cached across
the outer loop.

**Histogram-based boosting (EMA-FS, arXiv:2606.26337)** reports 65-70% of
its time in exactly this kernel, and its remedy — histogram subtraction of a
sibling from a parent — assumes a *tree* of nested cell sets. Our cells are
fixed, not nested, so that remedy has no analogue here. Recorded as searched
and rejected, not as unexamined.

Two facts specific to this code decide the design, and both were verified in
the source rather than assumed. First, the centred marginal basis is
`B_raw @ P` (`features/interaction.py`, `_centered_marginal_basis`), a fixed
linear projection of the raw B-spline basis; a cubic raw row has 4 non-zeros
of 10, so the joint raw row has 16 of 100 where the centred joint row is a
dense 81 of 81. Because the accumulation is linear and `P` is constant,
`kron(P_1, P_2)` commutes with the row sum and can be applied after the grid
contraction. Second, tensor group matrices are preserved across REML lambda
states (`dm_builder.py`, the multi-penalty tensor group is not rebuilt with
lambda), so a cell-CSR cached on a tensor is valid for the whole outer loop
and the counting sort is paid once per fit rather than once per block.

## Design as built

**S4, `32e0dc8e`.** `DiscretizedTensorGroupMatrix.cell_csr()` sorts the rows
by grid cell once per instance with a stable counting sort, `O(n + cells)`
and 24x faster than `np.argsort(kind="stable")` at 300,000 rows, and keeps
the result. No invalidation is needed because `idx1` and `idx2` are never
written after construction and every row-changing operation — `row_subset`,
the per-lambda rebuild — constructs a new instance. The same commit moves
the shared-margin probe's `O(1)` checks ahead of its `O(n)` index
comparison; at 256 bins every pairing is over the three-way cell cap, so
each pair had been paying four full row comparisons per Gram build for a
route it could never take (`0.375 s` of a ten-pair fit, from the profile
above).

**S1, `d885f5b8`.** Stage 1 accumulates the channel tensor's row as its raw
B-spline band — 16 products of 100 for cubic margins — in the grid tensor's
cached cell order, and applies `kron(P_1, P_2)` after the grid contraction.
`_gather_cell_order` streams the two channel bins into cell order in two
sequential passes and the build cache permutes `W` once per grid tensor per
build (the review moved it out of the per-block prologue, where it had been
repeated for every partner of a grid: 35 of 45 passes redundant at ten
pairs), so the cell loop reads everything sequentially; gathering through
the permutation inside the loop instead measured as slow as the dense kernel
it replaces, the kernel being latency-bound. The stage
is chosen by budget: the raw scratch is wider (`n1 * n2 * k1_raw * k2_raw`,
52.4 MB in production) than the stored width, and over the aggregate budget
the route falls back to the dense stage at the stored width, never to the
row route, so the band never costs a block the channel route it had. A
cardinal `cr` pair, whose functions are non-zero everywhere, has no narrower
band and keeps the dense stage.

**Row passes per outer state.** Before this round, one random-access pass
per tensor-by-tensor block: 45 per ten-pair Gram build. After it, per block
two sequential permutation passes and one pass over the cell-CSR (135 per
build), plus one permutation of `W` per grid tensor per build (9 at ten
pairs; the last tensor is never a grid side) and one counting sort per grid
tensor per fit. The count rose while the time per block fell 2.3-3.5x
because every pass now streams. A margin without a band, or a block over
the raw budget, keeps the dense stage's one random pass.

**S2, `90defad0`, reverted.** `_cell_hist_raw_kron_parallel` ran the cell
loop under `prange` over eight contiguous cell chunks per thread, with the
stage dispatching on `get_num_threads()` — the serial kernel at one thread,
the twin otherwise. Bit-identical for any thread or chunk count by
construction (one writer per accumulator row, per-cell sums in the stable
cell order). It failed the speed half of its rule (below) and, on that
dispatch, made the block slower than the serial kernel at the default
unpinned pool, so the registered consequence applies: not adopted, deleted
rather than left behind a flag.

## Measurements

Machine state: the E4 QP spike's completion marker
(`tasks/w4i1qszil.output`, 42,748 bytes) was already present at the first
poll, so nothing was waited out; `uptime` then reported a one-minute load
average of 0.19, below the 0.4 bar, and no other agent ran during any timed
run. Working tree clean at `90defad0`. All four thread pools
(`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`,
`MKL_NUM_THREADS`) are pinned before NumPy is imported, to the value each
column names. `time.process_time` is recorded next to `time.perf_counter`
throughout; the unchanged round-one scripts are wrapped by a `runpy` runner
that reports both around the whole script rather than edited to add a
counter.

The design is the round-one one: 300,000 rows, six features uniform on
`[-1, 1]`, Poisson counts with a log-uniform exposure offset, fitted as
`SuperGLM(family="poisson", discrete=True, n_bins=256, features={c: Spline(kind="ps", k=10)}, interactions=pairs)`,
generator seed 20260919.

**Two before columns.** The recorded round-one logs are one of them. But
re-running `e130313d` in this session, from a `git archive` export imported
by `PYTHONPATH` with `superglm.__file__` printed in each log, put the
ten-pair four-thread Gram at `3.318 s` per state against the `2.740 s` the
round-one log recorded — 17% apart on the same tree and the same machine.
Every ratio below is therefore quoted against the same-session control,
with the recorded-log ratio given beside it where it differs materially.
The same-session control also settles a smaller question: `e130313d` ran at
`cpu/wall` 1.00, 1.01 and 1.05 at one, four and sixteen threads, so its
"four-thread" arm was the same serial computation as its one-thread arm and
the gap the round-one logs showed between them was run-to-run spread.

### Whole fit, default mode, `max_reml_iter=30`, four threads

The zero- to six-pair rows are best of two and their before column is the
recorded round-one log. The ten-pair row is a single run of each arm,
measured in this session against the `e130313d` export.

| Pairs | Wall s before | Wall s after | Wall ratio | CPU s before | CPU s after | States before/after | Objective after |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 0 | 1.66 | 1.78 | 0.93x | 1.66 | 1.78 | 10 / 10 | 80910.58463841362 |
| 1 | 2.52 | 2.62 | 0.96x | 2.52 | 2.62 | 12 / 12 | 80283.91344140828 |
| 3 | 5.89 | 5.00 | 1.18x | 5.88 | 7.35 | 13 / 13 | 79347.31763306631 |
| 6 | 16.91 | 9.75 | 1.73x | 16.86 | 20.29 | 13 / 13 | — |
| 10 | 48.53 | 23.64 | 2.05x | 48.42 | 58.08 | 13 / 13 | 79587.18132998943 |

The zero- to six-pair script records lambda states, not outer iterations, so
only states are tabulated there; the ten-pair arm recorded both and matched
on each (13 states, 11 outer iterations, `score_objective_tolerance`), as did
the separately-run zero-, one- and three-pair agreement arms (8, 10 and 11
outer iterations on both trees). The three- and six-pair objectives come
from those agreement and trajectory runs, not from this table's script; the
six-pair objective was not captured on either tree and is not reported.

Gram time inside the ten-pair fit: `35.21 s` before, `13.53 s` after, 2.60x.
The three- and six-pair ratios are conservative: their before column comes
from the same recorded session whose control re-ran 17% slower.

**Wall fell and CPU rose.** The ten-pair fit spends `58.08 s` of CPU for
`23.64 s` of wall where the base spent `48.42 s` for `48.53 s`. The wall win
is 2.05x; the CPU cost is 1.20x worse. On a machine running one fit that is
the trade the parallel kernel and the newly-threaded BLAS were asked for; on
a machine running several it is not, and the pool should be pinned down
accordingly. With S2 reverted the route has no parallel region of its own,
and what remains of the CPU rise is BLAS's in stages 2 and 3.

### Per state, `interaction_mode="fast_candidate"`, five-iteration cap

Seven lambda states at every non-zero pair count, so Gram seconds per state
is the controlled comparison. Before is the same-session `e130313d` control.

| Pairs | Blocks | Gram s/state 1t before | 1t after | 1t ratio | 4t before | 4t after | 4t ratio | 16t before | 16t after | 16t ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.030 | 0.053 | 0.57x | 0.055 | 0.050 | 1.10x | 0.034 | 0.033 | 1.03x |
| 1 | 0 | 0.077 | 0.063 | 1.22x | 0.066 | 0.054 | 1.22x | 0.081 | 0.060 | 1.35x |
| 2 | 1 | 0.149 | 0.108 | 1.38x | 0.148 | 0.100 | 1.48x | 0.179 | 0.152 | 1.18x |
| 3 | 3 | 0.316 | 0.224 | 1.41x | 0.270 | 0.160 | 1.69x | 0.359 | 0.168 | 2.14x |
| 4 | 6 | 0.666 | 0.270 | 2.47x | 0.512 | 0.267 | 1.92x | 0.537 | 0.247 | 2.17x |
| 6 | 15 | 1.259 | 0.587 | 2.14x | 1.114 | 0.512 | 2.18x | 1.169 | 0.471 | 2.48x |
| 10 | 45 | 3.481 | 1.633 | 2.13x | 3.318 | 1.316 | 2.52x | 3.244 | 1.034 | 3.14x |

Against the recorded round-one logs instead, the ten-pair ratios are 2.01x
at one thread and 2.08x at four. The zero- and one-pair rows form no
tensor-by-tensor block and their spread (0.57x to 1.35x on Gram times of
0.03-0.08 s) is measurement noise on quantities too small to read.

Per-commit attribution, four threads, ten pairs, all measured in this
session:

| Tree | Gram s/state | Wall s/state | Gram vs base |
| --- | ---: | ---: | ---: |
| `e130313d` base | 3.318 | 4.854 | 1.00x |
| `32e0dc8e` (+S4) | 3.004 | 4.642 | 1.10x |
| `d885f5b8` (+S1) | 1.472 | 2.558 | 2.25x |
| `90defad0` (+S2) | 1.316 | 2.388 | 2.52x |

### Per block

Direct `_cross_gram` calls, weights fixed, best of five with the three arms
interleaved round-robin. The dense arm is produced by hiding `raw_channels`
on both operands, so the route is the same and only the stage differs; the
raw-serial and raw-parallel arms differ only in the pool size the stage
read (the raw-parallel column is the deleted twin, kept as the record of the
rule it failed). Production is the 81x81 block at 256 bins per margin; the
`k=20` block is measured at 64 bins because at 256 bins it needs 23.7M
aggregate cells against the 8,388,608 budget and the route declines outright.

| Fixture | Threads | Dense ms | Raw serial ms | Raw parallel ms | Dense→raw par | Serial→par |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| production 81x81 | 1 | 48.90 | 17.30 | 17.00 | 2.88x | 1.02x |
| production 81x81 | 4 | 34.48 | 13.67 | 9.79 | 3.52x | 1.40x |
| production 81x81 | 16 | 48.39 | 13.47 | 13.27 | 3.65x | 1.01x |
| k=20, 361x361 | 1 | 65.60 | 16.91 | 15.44 | 4.25x | 1.10x |
| k=20, 361x361 | 4 | 60.16 | 11.10 | 7.90 | 7.62x | 1.40x |
| k=20, 361x361 | 16 | 60.99 | 9.81 | 8.90 | 6.85x | 1.10x |

CPU over wall on the production block: dense 1.00 / 4.91 / 15.65 and raw
parallel 1.00 / 6.51 / 18.90 at one, four and sixteen threads. The dense
kernel contains no parallel region at all, so its 4.91 and 15.65 are the
thread pools spinning after their regions, not work.

Stage 1 in isolation, same fixtures and protocol, separating the permutation
prologue from the kernel:

| Fixture | Threads | Dense gather ms | Prologue ms | Kernel serial ms | Kernel parallel ms | Kernel ser→par |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| production | 1 | 28.83 | 2.30 | 6.36 | 8.53 | 0.75x |
| production | 4 | 27.11 | 2.47 | 6.72 | 2.82 | 2.38x |
| production | 16 | 31.98 | 2.42 | 6.53 | 1.39 | 4.69x |
| k=20 | 1 | 59.52 | 2.38 | 5.35 | 5.23 | 1.02x |
| k=20 | 4 | 72.70 | 2.92 | 5.81 | 1.64 | 3.54x |
| k=20 | 16 | 57.89 | 2.29 | 5.50 | 0.78 | 7.02x |

At one thread the parallel kernel is 25% slower than the serial one on the
production block, which is why the stage dispatches on the pool rather than
always taking the twin. The whole stage (prologue plus kernel) beats the
dense gather by 3.33x serial and 5.12x at four threads on the production
block, 7.70x and 15.95x on the `k=20` one.

### Thread scaling

Ten pairs, `fast_candidate`, the settled tree, with the whole-script
`cpu/wall` beside the per-state figures:

| Threads | Wall s/state | Gram s/state | Script wall s | Script CPU s | CPU/wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.753 | 1.633 | 46.60 | 46.55 | 1.00 |
| 4 | 2.388 | 1.316 | 42.86 | 82.68 | 1.93 |
| 16 | 2.012 | 1.034 | 40.09 | 205.10 | 5.12 |

The same script on `e130313d` scaled not at all: 1.00, 1.01 and 1.05
`cpu/wall`, and `3.481`, `3.318`, `3.244` Gram seconds per state.

Sixteen threads buys 6.5% of wall over four for 2.48x the CPU, and at the
block level it is actively worse — `13.27 ms` against `9.79 ms` on the
production block — because the stage-1 kernel's gain (`1.39 ms` against
`2.82 ms`) is more than spent by the prologue and the GEMMs under a
sixteen-thread BLAS pool. **Four threads is the measured optimum on this
machine for this design.** That is a property of this machine, not a default
the library should set.

The review measured the numba x BLAS grid the plan promised and this phase
did not (production block, interleaved medians of 11, `process_time` beside
`perf_counter`); the raw-serial column is the settled tree:

| numba / BLAS threads | Dense ms | Raw serial ms | Raw parallel (deleted) ms | CPU/wall, parallel arm |
| ---: | ---: | ---: | ---: | ---: |
| 1 / 1 | 34.7 | 15.3 | 15.9 | 1.00 |
| 4 / 1 | 38.5 | 16.1 | 14.0 | 2.71 |
| 4 / 4 | 43.3 | 14.5 | 11.6 | 5.62 |
| 8 / 4 | 41.5 | 14.6 | 9.1 | 8.05 |
| 16 / 16 (the default, unpinned) | 50.6 | 12.8 | 16.2 | 15.6 |

Raising BLAS from one to four threads at numba four moved the raw block
`14.0 -> 11.6 ms`, as much as the parallel kernel itself; at the default
pool the twin lost to the serial kernel (0.79x), which is why it could not
stay behind its dispatch. The serial raw stage beats the dense one at every
pool (2.27x, 2.99x and 3.95x at 1/1, 4/4 and 16/16). **The recommended pin
is four on every pool.**

### Memory

Peak RSS of the ten-pair 300,000-row discrete fit at four threads, measured
by the review: `1156 MiB` on the branch against `1126 MiB` on the
`e130313d` export, +30 MiB where the design estimated about +45 MB. Nine of
the ten tensor groups carry a cell-CSR after the fit (eight bytes a row plus
the cell pointer, 2.9 MiB each at 300,000 rows; the tenth is never a grid
side); the cell-CSR is not pickled. The per-block transients — two
`n`-vectors of permuted bins, 4.8 MB, about 216 MB of allocation per
ten-pair build — could come from the build cache like the accumulator does;
that is a follow-up.

### Agreement

**Bit-identity where no cross block forms.** The zero-pair and one-pair
fits match `e130313d` in every recorded quantity to the last digit: states,
converged, termination reason, outer iterations, objective
(`80910.58463841362` and `80283.91344140828`), deviance
(`161683.19276839928` and `160425.969601634`) and all six and eight
smoothing parameters. The one-pair arm forms a tensor group but no
tensor-by-tensor block, so neither S1 nor S2 is reachable in it.

**Fits that do form cross blocks reach the same optimum, not the same
coordinates.** Three-pair trajectory: `79347.31763305388` before,
`79347.31763306631` after, 1.57e-13 relative, with the same 13 states and 11
outer iterations. Ten-pair default mode: `79587.181329993` before,
`79587.18132998943` after, 4.5e-14 relative, same 13 states, 11 iterations
and termination reason.

**The two-pair fixture of the REML-level shared-optimum tests**, measured
rather than merely asserted. Raw stage against dense stage, both taking the
channel route on all 21 blocks of the fit (`raw` 21 against 0), same
termination and same 12 outer iterations:

| Quantity | Raw vs dense stage | Channel route vs displaced row route |
| --- | ---: | ---: |
| objective, relative | 1.552e-12 | 1.813e-12 |
| deviance, relative | 1.790e-16 | 4.833e-15 |
| phi, relative | 0.0 | 0.0 |
| effective df, relative | 2.557e-13 | 4.379e-12 |
| predictions, max relative | 4.566e-12 | 1.008e-11 |
| beta, max absolute | 3.456e-11 | 4.084e-11 |
| lambdas, worst relative | 8.874e-08 | 1.607e-07 |

The raw stage's spread is smaller than the displaced row route's on every
row, so the representation change perturbs the fit less than the route
change round one already accepted.

**Oracle bound per block.** Against the dense float64 product
`X_i.T @ (W * X_j)` formed from the materialised designs, with the
repository's bound `32 * eps * max(n, n1 * n2) * ||abs(X_i).T @ abs(W X_j)||_inf`
plus `1e-12` relative Frobenius:

| Fixture | Arm | inf-norm error | Bound | Relative Frobenius |
| --- | --- | ---: | ---: | ---: |
| production | dense stage | 9.925e-14 | 4.072e-05 | 1.280e-15 |
| production | raw serial | 1.483e-13 | 4.072e-05 | 1.863e-15 |
| production | raw parallel | 1.483e-13 | 4.072e-05 | 1.863e-15 |
| production | displaced row route | 8.730e-14 | 4.072e-05 | 1.172e-15 |
| k=20 | raw parallel | 1.436e-13 | 1.725e-05 | 1.376e-15 |
| k=20 | displaced row route | 1.118e-13 | 1.725e-05 | 1.156e-15 |

Every arm is nine orders inside the bound, and the raw stage's error is
1.7x the dense stage's — the price of one extra representation change, not a
change of regime.

**Bit-identity across thread counts.** The stage-1 histogram is bit-identical
at 1, 4 and 16 threads on both fixtures, `max_abs_diff` exactly 0.0, which is
the claim the parallel kernel actually makes. The production block as a whole
is bit-identical too. The `k=20` block is not: `1.776e-15` maximum absolute,
`2.151e-16` relative Frobenius between one thread and four or sixteen.
Tracing it stage by stage on fixed inputs put it entirely in stage 3's
`R_inv.T @ raw @ chan_map` contraction — stage 1's histogram and stage 2's
two marginal GEMMs are bit-identical, and so is the `chan_map` product
itself — that is, in OpenBLAS's blocking of a `361 x 400 x 361` product,
which is large enough to thread where the production block's `81 x 100 x 81`
is not. The pre-existing displaced row route moves by `4.996e-16` on that
same block across the same pools, so this is BLAS's thread-sensitivity and
not the new kernel's.

**Focused suites.** `tests/test_discrete_tensor_execution.py`,
`tests/test_discretize_fit.py` and `tests/test_interactions.py`: 283 passed,
one unrelated deprecation warning, 21.76 s, at `90defad0`; 280 after the
review-fix commit (the five S2 tests removed; the once-per-grid weight
permutation and the pickle round trip added). The full suite was not re-run
in this phase.

## Decision rules and outcomes

The rules were pre-registered in the round-one plan draft before any of this
was built.

**S1 — "adopt if the per-block time falls by at least 2x on the ten-pair
design and the oracle bound holds; measured interleaved, one thread and
four." PASSES, by 10-40% over the bar depending on protocol.** Per block at
one thread, 48.90 ms to 17.30 ms is 2.83x on the production block (best of
five) and 65.60 ms to 16.91 ms is 3.88x on the `k=20` one; at four threads,
2.52x and 5.42x. The review's interleaved medians of 11 give 2.27x at one
thread (34.7 to 15.3 ms) and 2.99x at four on four, and the ten-pair build
measured 2.17-2.20x per state — all clear of 2x, inside this VM's +/-20%
session band, so the range and not one number is the finding. Stage 1 alone
is 3.33x serial. The oracle bound holds with nine orders of margin on every
arm and fixture.

**S2 — "adopt if the per-block time at four threads is at least 4x below
serial and the result is bit-identical at 1, 4 and 16 threads." The
exactness half passes; the 4x half FAILS at four threads.** The parallel
kernel is bit-identical at 1, 4 and 16 threads on the stage-1 histogram of
both fixtures and on the production block end to end, and the one
non-identity found is BLAS's, outside the kernel and shared with a route
that predates this work. But the block at four threads is only 1.40x below
the raw serial block, not 4x, on both fixtures; even the kernel in isolation
is 2.38x at four threads, reaching the rule's 4x only at sixteen (4.69x, and
7.02x on `k=20`). The rule was written against a prototype measured at 12.3x
per block at four threads — on a *stage 1 that was still 81 doubles wide*.
S1 shrank the only parallel region from 59% of the block (28.83 ms of 48.90
at one thread) to 37% (6.36 ms of 17.30), and Amdahl's law bounds the block
at 1.6x however many threads the kernel gets. The rule is not met as
written, and the review added the arm this phase did not run: at the default
unpinned pool (16/16) the twin makes the block slower than the serial
kernel, `12.8 -> 16.2 ms`, while at pinned pools it is 1.15-1.61x. The
registered consequence — not adopted, deleted rather than left behind a
flag — is applied by the review-fix commit: the kernel, its dispatch, the
`get_num_threads` read and the thread-count profile key are gone and the
stage is serial again. The rule itself stands unamended; a parallel stage 1
needs a rule written against the 37% of the block it can now touch and the
numba-beside-BLAS interaction measured above.

**S4 — "no regression." PASSES.** At ten pairs and four threads, Gram
seconds per state fall from 3.318 to 3.004 and wall from 4.854 to 4.642; no
pair count is slower outside the sub-0.1 s noise band; the zero- and
one-pair fits are bit-identical.

## Limits

**The quadratic term is untouched, again.** The fit still forms `M(M-1)/2`
tensor-by-tensor cross blocks — 45 of them at ten pairs. Round one moved
the constant about 4x and this round moves it about 2.5x more; the exponent
is the same, and a design with enough pairs is still dominated by block
count. S7 (matrix-free conjugate gradients) remains the only lever that
touches it, and remains research.

**Non-Gram time barely moved.** At ten pairs the settled tree spends 2.388 s
per state of which 1.316 s is Gram, so 1.07 s per state is elsewhere,
against 1.44 s before. S5's profile named the candidates and one of them is
plainly wasteful: `discretize_column` is called 52 times for 6 distinct
columns in a ten-pair fit, `1.084 s` of the `1.233 s` total being repeats of
work already done. That is a follow-up, not fixed here.

**Where the cache can go stale.** `cell_csr()` is memoised per instance with
no invalidation hook, and its correctness rests on `idx1` and `idx2` never
being written after construction. That holds today — `row_subset` and the
per-lambda rebuild both construct new instances — and a test pins that
`row_subset` does not inherit the cache. It is not enforced by the type: an
in-place mutation of either index array anywhere would silently return a
permutation for the old rows. If those arrays ever become writable, this
needs a guard rather than a comment. The cache is dropped on pickle and
rebuilt on first use, and a tensor design pickled before this branch loads
with no band and takes the dense stage: the review found the bare slots
raising on refit of a retained model, fixed with `__setstate__` defaults
and a test.

**Thread pools.** Everything above is pinned; nothing here is a default.
Four threads is the optimum for this design on this sixteen-core machine and
sixteen is worse at the block level, but the shape of that curve depends on
the grid size, the payload width and the BLAS build, so it must be measured
per machine. With S2 reverted the library neither reads nor sets a numba
pool on this route; BLAS's pool still decides stages 2 and 3, and a caller
who leaves it unpinned gets whatever OpenBLAS's pthreads negotiate — at
sixteen threads that was 205 s of CPU for 40 s of wall on a script that
needs 47 s of CPU serially.

**The raw-basis representation change, stated exactly.** Stage 1 no longer
multiplies the stored centred joint row. It forms `(w * raw1) * raw2` on the
raw B-spline values, sums each cell in its stable row order, and applies
`kron(P_1, P_2)` after the grid contraction. For `ps` margins `raw @ P`
reproduces the stored centred margin bitwise, so the only change is
summation order and the point at which a linear map is applied; for
`Z`-projected `ns` and legacy `cr` margins it is a round-off change as well.
The contract is therefore **not** bit-identity with the dense stage — it is
the repository's oracle bound, met at `1.9e-15` relative Frobenius on the
production block, and the fit-level agreement tabulated above. Anything
downstream that pinned a discrete-tensor number to more than about 12
significant figures will move.

**Fixture caveats.** The `k=20` block is measured at 64 bins per margin, not
256, because at 256 the route declines outright on the aggregate cell budget
— which is itself a limit worth naming: wide margins at fine grids fall back
to the row route, and only the decline counter says so. And the
17%-apart before columns mean single timing numbers here should be read to
roughly that band; the arithmetic quantities — objectives, deviances,
lambdas, oracle errors — are deterministic and are quoted in full.

## Reproduction

Commits: S4 `32e0dc8e`, S1 `d885f5b8`, S2 `90defad0` (reverted by the
review-fix commit), on `perf/interaction-cost-round-two` over `e130313d`
(v0.34.0 plus round one).
The before columns were produced against `e130313d`, `32e0dc8e` and
`d885f5b8` exported with `git archive <rev> src` and imported by
`PYTHONPATH`, with `superglm.__file__` printed in each log so the arm is
unambiguous.

Scripts and logs (session scratch, not committed), under
`/tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/interaction-cost/`:

- `scaling-plan-draft.md` — the pre-registered levers S1-S7 and their
  decision rules.
- `round2/wait_e4.py` — the quiet-machine poll for the E4 spike's marker.
- `round2/profile/` — the S5 per-pass profile of the ten-pair and three-pair
  fits on `e130313d` (`fast10_passes.log`, `default3_passes.log`), the
  bandwidth roofline (`stage1_bandwidth.log`), the `discretize_column`
  redundancy count (`build_redundancy.log`) and the allocation-churn probe.
- `round2/bench/diagnose_scaling.py`, `diagnose_scaling_serial.py` and
  `diagnose_scaling_16t.py` — the round-one scaling script, copied unchanged
  for the four- and one-thread arms and with only the pool value changed for
  sixteen; run through `round2/bench/run_with_cpu.py`, which reports wall and
  CPU around an unmodified script. Logs `diagnose_scaling_round2_{1,4,16}t.log`
  and the controls `diagnose_scaling_e130313d_{1,4,16}t.log`,
  `diagnose_scaling_s4_4t.log`, `diagnose_scaling_s1_4t.log`.
- `round2/bench/time_pairs.py` (copied unchanged) and
  `round2/bench/time_pairs10.py` — the default-mode arms; logs
  `time_pairs_round2_4t.log`, `time_pairs10_{before,after}.log` and their
  JSON.
- `round2/bench/trajectory_3pairs.py` (copied unchanged) — the three-pair
  trajectory, `trajectory_3pairs_round2.log`.
- `round2/bench/perblock_round2.py` and `stage1_round2.py` — the per-block
  and stage-1 arms with their oracle checks; logs `perblock_{1,4,16}t.log`,
  `stage1_{1,4,16}t.log`.
- `round2/bench/check_bit_identity.py`, `stage3_round2.py`,
  `check_stage_identity.py`, `check_stage3_identity.py`,
  `dense_thread_control.py`, `chanmap_control.py`, `check_chanmap.py`,
  `check_control.py` — the cross-thread identity check and the stage-by-stage
  trace that located its one failure in BLAS.
- `round2/bench/agreement_arms.py` (copied unchanged),
  `compare_agreement.py` and `two_pair_optimum.py` — the zero- and one-pair
  bit-identity check against `e130313d` and the two-pair shared-optimum
  quantities.
- `round2/focused_tests.log` — the three focused suites.

Round one's prototypes remain at `stage1_cellcsr.py`, `stage1_variants.py`,
`projected_gram.py` and `oracle_small.py`, and its settled-tree logs under
`bench/`.

No timing assertion enters the test suite. The suites assert on profile
counters, cell budgets, route labels, oracle bounds and bit-identity across
numba pools.
