# Discrete tensor REML steps and the cross-Gram channel route

Two independent defects made many-interaction discrete fits both wrong and
slow. The discrete REML engine's shared-tensor branch clipped its
modified-Newton step one coordinate at a time, which rotated the step into an
ascent direction the line search could never accept: on the 300,000-row
three-pair synthetic design the directional derivative went from `-32.71`
after the plain trust-region clip to `+12.71` after the per-pair clip, every
step length was rejected, no true objective was evaluated, and the loop ran
to `max_reml_iter` publishing `79370.28242054139`. Imposing the same trust
region by damping instead of clipping converges the same fit in 11 outer
iterations at `79347.31763305388`, 22.96 REML units lower. Separately, a
cross-Gram block between two discretised tensor terms used to expand
observation-row panels; staging it through the existing channel histogram
made the block 4.5x faster at four threads and 5.5x serial. Together, on the
recorded before/after table, the default-mode three-pair fit falls from
25.10 s to 5.89 s and the six-pair fit from 105.60 s to 16.91 s, while the
additive (zero-pair) fit is bit-identical in objective and in all six
smoothing parameters. The block count is still quadratic in the number of
pairs; only its constant moved.

This note records the measurements for E0 commit `0107b201` and kernel
commit `03772633` on `perf/interaction-fit-cost`, whose base is the released
v0.34.0 (`99ca0edd`).

## Root cause

### The pair-cap ascent step

`optimize_discrete_reml_cached_w` turned on a tensor surrogate line search
whenever the family had known scale and at least one penalty group spanned a
`DiscretizedTensorGroupMatrix` with two components, that is, for
numeric-by-numeric tensor interactions. On that branch the eigen-floored
modified-Newton step `delta = -(V diag(lam)^-1 V^T) g` was first clipped
element-wise to `|delta_k| <= base_cap`, and then each shared tensor pair was
re-solved in sum/difference coordinates `u = (d_i + d_j) / 2`,
`v = (d_i - d_j) / 2` with `u` and `v` clipped against *independent* bounds
`cap_u` and `cap_v`. Nothing re-checked the composed step.

Clipping two coordinates by different factors rotates the direction. On the
`x2:x3` pair of the three-pair design the pair solve wanted
`(u, v) = (9.3886, -17.2936)`; the bounds truncated it to `(5.0, -1.0)`, a
factor of 1.9 on `u` against 17 on `v`. Because `d_i = u + v`, the step on
`x2:x3:margin_x2` changed sign, from `-7.905` to `+4.0`, against a gradient
of `+6.80199` on that coordinate. The whole step's directional derivative by
stage was: Newton `-65.0039`, after the element-wise clip `-32.7131`, after
the pair clip `+12.7084`. The quadratic surrogate then evaluated
`12.708 s + 53.38 s^2`, positive for every `s > 0`, so all five permitted
halvings rejected, no candidate was formed, the post-loop true-objective
branch never ran, and `rho` was restored unchanged. The next iteration
recomputed a virtually identical gradient and Hessian from a fresh and
expensive PIRLS Gram and rebuilt the same ascent step. The state repeated,
digit for digit, for 27 iterations.

The stall was shown, not modelled. A reconstruction of the engine's step from
its own recorded gradient, Hessian and freeze mask agrees with the engine's
`tensor_uv` record at every iteration, and the recorded surrogate verdicts at
iteration 8 were `(1.0, +66.087)`, `(0.5, +19.699)`, `(0.25, +6.513)`,
`(0.125, +2.423)`, `(0.0625, +1.0028)` — every predicted change positive.
Freezing was not the cause: 0 of 12 directions were frozen at every stalled
iteration, so the stop-gradient mask was the identity and the `6.80` gradient
sat on an estimated direction at `rho = 4.36`, far from either bound. Nor
was it a legitimate stationary point: patching the shared-tensor pair list to
empty, which removes only the pair override and leaves the surrogate search
and every other path identical, reached `79359.49018882592`, 10.79 REML units
lower. That counterfactual arm also ended at `max_reml_iter`, so 10.79 was a
lower bound on the error of the published fit rather than its size; the
repaired engine's `79347.31763305388` puts the realised gap at 22.96.

A second, independent floor lived in the same branch: `local_max_halving`
was 5 on the surrogate path against 25 on the generic one, so the backtrack
stopped at `s >= 1/32`. The counterfactual arm died at its own iteration 5
with `g . d = -0.08503` (a genuine descent direction) and `d' H d = +5.694`,
whose model minimiser sits at `s* = 0.0299`, below that floor.

Nothing counted consecutive iterations that accepted nothing, so the loop
never named the condition: `termination_reason` kept its initial value
`max_reml_iter`. The exact engine already had the right machinery —
`classify_dead_feasible_exit` in `src/superglm/reml/convergence.py` — and
`src/superglm/reml/direct.py` was already its only caller.

Cost attribution at three pairs: the profile reported 25.0 s total of which
`irls_gram_s` was 19.59 s over 30 iterations. Iterations 3 to 30 accepted
nothing, so roughly 18 s of Gram work and 22 s of wall time bought no
movement in lambda at all.

### The row-expanded tensor cross-Gram

A cross-Gram block between two `DiscretizedTensorGroupMatrix` instances with
different tensor ids reached the generic discretised-by-discretised branch,
where the joint support product `n_pairs_i * n_pairs_j` (4.2e9 in the
300,000-row, 256-bin case) exceeds the histogram cell cap and forces the
row-expanding route: two `(chunk, 81)` observation panels and a
`2 n P_i P_j`-flop GEMM per block. At the production default of 256 bins the
compact shared-margin helper also declines — a one-shared-margin pair needs
`256^3 = 16,777,216` cells against a 5,000,000 cap — so every tensor-by-tensor
pair took that route. Those blocks were 71% of a Gram assembly at three pairs
and 94% at six.

The premise that a cubic B-spline row is sparse is false for this
representation and would have produced the wrong kernel: the stored marginal
bases measure 9 nonzeros per row out of 9 columns and the materialised joint
basis 81 of 81, because constraint absorption and the SSP reparametrisation
are baked into the stored bases. The exploitable structure is Kronecker
factorisation, not sparsity.

## Fix

### Damping, not clipping

A line-search method requires a descent direction (Nocedal and Wright,
*Numerical Optimization*, 2nd ed., Springer 2006, ch. 3). The standard way to
respect a trust region while keeping one is to damp the whole step rather
than clip its coordinates: with the eigen-floored `H_pd = V diag(lam) V^T`
already formed,

    delta(mu) = -(H_pd + mu I)^-1 g = -V diag(1 / (lam + mu)) V^T g

satisfies `g . delta(mu) = -sum_k (v_k' g)^2 / (lam_k + mu) < 0` for every
`mu >= 0`, and shrinks the near-singular directions first. This is the
Levenberg-Marquardt form of the trust-region step (Nocedal and Wright ch. 4);
More and Sorensen, "Computing a trust region step", *SIAM J. Sci. Stat.
Comput.* 4 (1983) 553-572, [doi:10.1137/0904038](https://doi.org/10.1137/0904038),
give the exact ellipsoidal subproblem and the safeguarded root-finding for
`mu`. The other standard route for a *box*-shaped region is projected Newton
(Bertsekas, "Projected Newton methods for optimization problems with simple
constraints", *SIAM J. Control Optim.* 20 (1982) 221-246), and its analysis is
exactly what says that independently clipping the coordinates of a Newton
step is not a projection of the step and loses descent.

The region itself is unchanged: `|delta_k| <= base_cap` on every coordinate,
intersected for each active shared tensor pair with `|u| <= cap_u`,
`|v| <= cap_v`, including the post-stall widening of those radii. `mu = 0` is
tried first, so a step already inside the region is the plain modified-Newton
step; otherwise the smallest feasible `mu` is bracketed geometrically from
the analytic bound `|g|_2 / r` (with `r` the smallest radius, since every
constraint functional is bounded by `|delta(mu)|_2 <= |g|_2 / mu`) and
bisected on `log mu`. Each trial costs `O(q^2)` on the `q`-dimensional active
subspace. Descent is asserted rather than silently repaired, and `mu`, the
binding constraint names, the raw and used `u`/`v`, and `g . d` before and
after damping are recorded in `reml_outer_step_stats`.

The five-halving floor is gone: the backtrack schedule visits the model
minimiser `s* = -quad_grad / quad_curv`, the penalty build is deferred until a
length the surrogate admits, and a rejected true trial backtracks the way the
generic path already did.

### Exit semantics

A dead tensor line search now exits through the exact engine's
`classify_dead_feasible_exit`: `converged_at_precision` with `converged=True`
only when every active gradient is under `max(1e-7, reml_tol) * (1 + |objective|)`
*and* a true objective was evaluated and rejected, and `line_search_failed`
with `converged=False` otherwise. `src/superglm/model/api.py` was extended to
enumerate the two new discrete exits, which is a public contract change.

Two things differ from the exact engine, and the gate decided both. The
exact engine breaks on the *first* dead search; here the break waits for
`candidate_mode_stationary` (the candidate's own PIRLS converged) and never
fires on the first outer iteration. The reason is measured: on the real
Credit additive binomial arm every one of the three dead searches was
followed immediately by an accepted step, with the candidate PIRLS
unconverged at each, because one working-model update per outer iteration
moves the gradient at unchanged `rho`. Breaking there would publish a
different model, not the same model sooner; the A/B recorded a maximum
absolute log-lambda difference of 19.81 and a maximum prediction difference
of 0.564 between breaking at the first dead search and running the budget.
On the synthetic stall the flag turned true at the second dead search and the
state then repeated for 27 iterations. Second, the exit is confined to the
shared-tensor path; the generic path keeps iterating and its numbers are
untouched.

Gate measurements on v0.34.0, `max_reml_iter=20` (30 for the synthetic arm):

| Arm | Outer iters | Dead searches | Dead-search iterations | Accepted a step after a dead search | Candidate PIRLS settled at first dead search | Active gradient / bar at first dead search |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `uci_breast_cancer_additive` | 10 | 0 | — | — | — | — |
| `uci_breast_cancer_8interactions` | 20 | 0 | — | — | — | — |
| `uci_credit_default_additive` | 20 | 3 | 4, 10, 16 | yes (14 later iterations) | no | 2184 |
| `ames_housing_additive` | 12 | 0 | — | — | — | — |
| `ames_housing_8interactions` | 20 | 0 | — | — | — | — |
| `uci_bike_sharing_additive` | 4 | 0 | — | — | — | — |
| `uci_bike_sharing_6interactions` | 11 | 0 | — | — | — | — |
| `synthetic_3pairs` | 30 | 28 | 3 to 30 | no | yes | 856 |

Both dead-search arms classify as `line_search_failed` under either the
strict or the lenient reading of the predicate, so neither acquires
`converged=True`. The Credit arm's active gradient at its first dead search
is 2184 times its bar and 25 true trials had been evaluated and rejected;
the synthetic arm's is 856 times its bar with zero true trials evaluated.

### The channel route

`_cross_gram_tensor_tensor_channels` uses the Kronecker structure of one
side. Row `r` of tensor `i` is `B1_i[idx1_r] (x) B2_i[idx2_r]`, so
`X_i' diag(W) X_j` factors through `i`'s cell index:

    H[i1 * n2 + i2, cd] = sum_{r in cell} W_r * B_joint_j[bin_idx_r, cd]
    raw[a * K2 + b, cd] = sum_{i1, i2} B1_i[i1, a] B2_i[i2, b] H[i1 * n2 + i2, cd]

followed by the `R_inv` sandwich every other tensor helper ends with. Stage 1
is the existing `_disc_disc_2d_hist_channels` njit kernel — one serial
`O(n)` pass with no observation panel — and stage 2 is the two-GEMM
contraction `_cross_gram_tensor_main` already uses, so no new compiled code,
no parallel kernel and no cached permutation were added. Dispatch sits in
`_cross_gram`'s tensor-by-tensor block, after the shared-margin helper
declines and before any other branch, so the compact helper keeps priority
wherever it still fires and the new route picks up both fully distinct pairs
and the shared-margin pairs the helper declines.

Orientation is decided by `cells_i = n1_i n2_i P_j` against
`cells_j = n1_j n2_j P_i`, smaller side as the grid, with a tie keeping the
left operand. The production case ties exactly at 5,308,416, and the two
orientations differ at about one ulp, so the tie-break is part of the
numerical contract and is tested. The route declines — returning `None` and
falling through to the displaced route — on equal tensor ids, non-float64
margins, joint basis or weights, a minimum cell count above
`_MAX_AGGREGATE_CELLS` (8,388,608 cells, 64 MiB), or operands outside
`_tensor_operand_in_reassociation_range`. Gating on
`_MAX_DISC_DISC_CHANNEL_HIST_CELLS` (5,000,000) would have declined the
production block, which needs 5,308,416 cells, and the change would have
appeared to do nothing. The range guard is a new and stricter policy than the
displaced route's, which applies it only below the histogram cell cap; a
decline lands on exactly that route, and is counted so a fit that backs off
to the quadratic path says so.

Peak transient is the histogram, 40.5 MiB on the production block, below the
two 32 MiB row panels the displaced route holds live. In matched units the
arithmetic falls from `n P_i P_j = 1.97e9` multiply-adds to about `7.5e7`,
roughly 26x; the realised gain is 4.5x to 5.5x per block because stage 1
gathers and accumulates `P_j` doubles per row and is memory-bound, so the
arithmetic ratio should not be quoted as a speed claim.

**Exactness.** This is not a bit-identical refactor and should not be
described as one. The route replaces the grid side's stored joint rows by the
product of its two marginal rows — a one-ulp representation change, since the
joint basis was formed as exactly that product at build time — and it changes
the summation order. The contract is the repository's own oracle bound,
`32 eps max(n, n1 n2) ||abs(X_i)' abs(W X_j)||_inf` together with 1e-12
relative Frobenius, not bit identity. Measured agreement is 1.0e-15 relative
Frobenius per production block, 2.45e-19 on the assembled Gram, and 3.6e-16
to 4.6e-15 across fifteen small-design shape and weight combinations. Through
a REML fit the 1e-15 perturbation moves flat smoothing parameters by up to
2.5e-7 relative (coefficients 2.9e-9, predictions 1.4e-11) while the
objective moves by 2.6e-13; re-chunking the dense route's own row sum moves
the same quantities by 9.1e-8, 7.1e-9, 5.7e-12 and 7.3e-13, that is by the
same amounts, which is the control that says the movement is round-off and
not a change of answer.

## Measurements

Machine state: `uptime` load average 0.06 at the start, no other agent on the
machine during any timed run, working tree clean at `03772633`. All four
thread pools (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`,
`MKL_NUM_THREADS`) are pinned before NumPy is imported, to 4 except where a
serial column is named. `time.process_time` is recorded next to
`time.perf_counter`. The before column is the recorded v0.34.0 run on the
same machine and the same design; it was not re-timed.

The design is 300,000 rows, six features uniform on `[-1, 1]`, Poisson counts
with a log-uniform exposure offset and three planted pairs, fitted as
`SuperGLM(family="poisson", discrete=True, n_bins=256, features={c: Spline(kind="ps", k=10)}, interactions=pairs)`,
generator seed 20260919.

### Default mode, `max_reml_iter=30`, four threads

Wall and CPU are best of two runs. CPU over wall is 1.00 in every cell of
both columns, so no arm is fanning out across the pinned pools.

| Pairs | States before | States after | Wall s before | Wall s after | Wall ratio | CPU s before | CPU s after | Termination before | Termination after | Objective before | Objective after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 0 | 10 | 10 | 1.79 | 1.66 | 1.08x | 1.79 | 1.66 | `score_objective_tolerance` | `score_objective_tolerance` | 80910.58463841362 | 80910.58463841362 |
| 1 | 12 | 12 | 2.35 | 2.52 | 0.93x | 2.35 | 2.52 | `score_objective_tolerance` | `score_objective_tolerance` | 80283.89921532459 | 80283.91344140828 |
| 3 | 32 | 13 | 25.10 | 5.89 | 4.26x | 25.05 | 5.88 | `max_reml_iter` | `score_objective_tolerance` | 79370.28242054139 | 79347.31763305388 |
| 6 | 32 | 13 | 105.60 | 16.91 | 6.25x | 105.48 | 16.86 | not recorded | `score_objective_tolerance` | not recorded | 79351.91860066733 |

The six-pair before run recorded only states, wall and CPU; its 32 states
match the three-pair `max_reml_iter` signature exactly, but the termination
string and objective were not captured and are not reported here. The
six-pair after termination and objective come from a separate single run of
the same arm, whose own wall was 19.76 s against the 16.91 s best-of-two
above.

### `interaction_mode="fast_candidate"`, five-iteration cap

Seven lambda states at every non-zero pair count, so per-state Gram time is
the controlled comparison. The before column is four threads.

| Pairs | Cross blocks | States | Wall s before (4t) | Wall s after (4t) | Wall s after (1t) | Gram s/state before | Gram s/state after (4t) | Gram s/state after (1t) | Gram/state ratio (4t) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 10 | 1.84 | 1.96 | 1.93 | 0.026 | 0.028 | 0.026 | 0.93x |
| 1 | 0 | 7 | 2.06 | 2.10 | 2.13 | 0.050 | 0.053 | 0.051 | 0.94x |
| 2 | 1 | 7 | 4.94 | 3.29 | 3.35 | 0.343 | 0.138 | 0.135 | 2.49x |
| 3 | 3 | 7 | 10.17 | 4.54 | 4.40 | 0.912 | 0.273 | 0.264 | 3.34x |
| 4 | 6 | 7 | 18.11 | 6.32 | 6.75 | 1.774 | 0.444 | 0.476 | 4.00x |
| 6 | 15 | 7 | 41.04 | 11.90 | 14.17 | 4.275 | 0.982 | 1.270 | 4.35x |
| 10 | 45 | 7 | 105.17 | 29.26 | 34.09 | 11.257 | 2.740 | 3.286 | 4.11x |

The serial column is within 1.16x of the four-thread column at ten pairs and
faster than it at two, three and four pairs, which is the expected shape for
a route whose long reduction has left BLAS: the histogram pass is a serial
njit loop and the remaining GEMMs are small. The zero- and one-pair rows form
no tensor-by-tensor block at all and move by 0.94x and 0.98x, within the
run-to-run spread.

### Per block and per Gram assembly

Direct calls on the three-pair design, weights fixed, the dense arm produced
by forcing the new helper to decline, best of five per block and best of
three per assembly.

| Measurement | Threads | Channel | Dense | Ratio |
| --- | ---: | ---: | ---: | ---: |
| One 81x81 tensor block, wall | 4 | 27.8-28.4 ms | 126-128 ms | 4.53x over three blocks |
| One 81x81 tensor block, CPU | 4 | 111-124 ms | 497-511 ms | — |
| One 81x81 tensor block, wall | 1 | 29.3-31.4 ms | 163-173 ms | 5.54x over three blocks |
| `_block_xtwx` assembly, wall / CPU | 4 | 0.129 s / 0.513 s | 0.431 s / 1.725 s | 3.33x |
| `_block_xtwx` assembly, wall / CPU | 1 | 0.136 s / 0.136 s | 0.561 s / 0.560 s | 4.13x |

Route counters on the assembly: `block_cross_tensor_tensor_channel_calls` 9
and `block_cross_disc_disc_rows_calls` 0 on the channel arm against 0 and 9
on the dense arm, with `block_cross_tensor_tensor_channel_declines` 9 there.
The four-thread CPU-over-wall ratio is about 4 on both arms, so the residual
parallelism is the surrounding BLAS and not the new pass.

### Real interaction arms

Datasets from the benchmark manifest, 20-iteration budget, four threads. The
before column is a fresh v0.34.0 run on this machine rather than the
instrumented gate run, so both columns are the plain engine's own published
values.

| Arm | Reason before | Reason after | Iters before | Iters after | Objective before | Objective after | Wall s before | Wall s after |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `uci_bike_sharing_additive` | `score_objective_tolerance` | `score_objective_tolerance` | 4 | 4 | 129212.4178081844 | 129212.4178081844 | 0.91 | 0.95 |
| `uci_bike_sharing_6interactions` | `score_objective_tolerance` | `score_objective_tolerance` | 11 | 7 | 126627.40676299471 | 126627.40676251397 | 2.49 | 1.92 |
| `uci_breast_cancer_additive` | `score_objective_tolerance` | `score_objective_tolerance` | 10 | 10 | -201.0189416557526 | -201.0189416557526 | 1.89 | 1.92 |
| `uci_breast_cancer_8interactions` | `max_reml_iter` | `max_reml_iter` | 20 | 20 | -308.4712313449336 | -308.47122123763785 | 3.75 | 3.73 |
| `ames_housing_additive` | `score_objective_tolerance` | `score_objective_tolerance` | 12 | 12 | 20006.905882323892 | 20006.905882323892 | 4.14 | 3.67 |
| `ames_housing_8interactions` | `max_reml_iter` | `max_reml_iter` | 20 | 20 | 19795.30186640287 | 19795.30186640287 | 7.23 | 7.40 |
| `uci_credit_default_additive` | `max_reml_iter` | `max_reml_iter` | 20 | 20 | 7879.92185645742 | 7879.92185645742 | 3.08 | 2.59 |

The three additive arms and the Credit arm are bit-identical in objective and
in deviance, which is the direct check that the generic path is untouched:
Credit's deviance is `15477.918224365874` on both sides, and its 101
line-search objective evaluations are the same 101. The Bike interaction arm
reaches the same optimum in 7 outer iterations instead of 11 (objective
agreeing to 3.8e-12 relative). The two arms that still exhaust the budget
move at 3.3e-8 (Breast) and not at all (Ames) relative. None of these arms
has a dead line search after the fix, and none shows a dead search followed
by an accepted step.

Route evidence inside the real arms, from a wrapper that counts calls without
changing arithmetic: Ames 8-interactions consults the channel helper 391
times and accepts 391, Breast 8-interactions 459 of 459, Bike
6-interactions 54 of 54, and the additive arms never consult it. Ames being
route-switched and yet bit-identical is an observation, not a guarantee; the
contract remains the oracle bound above.

### Agreement

The zero-pair fit is bit-identical: objective `80910.58463841362`, deviance
`161683.19276839928`, and all six smoothing parameters equal to the last
digit (`x0` 14.488801046277478, `x1` 107.98981286002561, `x2`
2264250.5794695546, `x3` 95.33900444402836, `x4` 3622253.6293491144, `x5`
1754701.2010714328). That arm builds no tensor group at all, so neither
changed code path is reachable.

The one-pair fit agrees to 1.77e-7 relative on the objective
(`80283.89921532459` before, `80283.91344140828` after), inside the 1e-6 bar,
with the same termination reason and the same 10 outer iterations on both
sides. It forms one tensor group and therefore no tensor-by-tensor cross
block, so the difference is entirely the damping change: `mu` is positive and
a radius binds at iterations 1 to 5. The after point is 0.0142 REML units
*above* the before point. The two trajectories settle at different places
along a flat ridge — `x4` moves from 4.37e6 to 2.55e6 and the pair margins
from 156094 and 192677 to 80431 and 91096 — and at those magnitudes the
objective is insensitive. The 1e-6 bar is met; the sign of the difference is
recorded here because it is not what a strictly-better story would predict.

The focused suites on the changed surfaces pass on the settled tree:
`tests/test_discrete_tensor_step.py`, `tests/test_discrete_tensor_execution.py`,
`tests/test_cross_matrix_reassociation_range.py`,
`tests/test_cross_matrix_histogram_dispatch.py`, `tests/test_discretize_fit.py`,
`tests/test_reml_newton_fixes.py` and `tests/test_theory_invariants.py`, 284
tests, no failures. The full suite was not run in this phase.

## Limits

**The quadratic term is unchanged.** The fit still forms `M(M-1)/2`
tensor-by-tensor cross blocks; the channel route removed the large constant
in front of that count, not the count. The measured Gram-per-state ratio
therefore plateaus near the per-block ratio (4.35x at six pairs, 4.11x at
ten) rather than growing, and a design with enough pairs will still be
dominated by block count. Cells scale as `n1 n2 K1_j K2_j`, so wider margins
decline back to the row route silently apart from the decline counter: `k=20`
margins at 256 bins need 2.37e7 cells against the 8,388,608 budget.

**Timing spread.** The six-pair default arm measured 16.91 s as best of two
and 19.76 s as a single run, a 17% spread, and the four-thread and serial
`fast_candidate` columns cross over at small pair counts. Treat single
numbers as indicative to roughly that band. The arithmetic quantities —
gradients, steps, objectives, lambdas — are deterministic and are quoted at
full precision.

**Two arms still exhaust the budget.** Breast 8-interactions and Ames
8-interactions still terminate at `max_reml_iter` after the fix, so the
damping change is not a universal cure for non-convergence on this corpus; it
removes one specific mechanism. Breast runs the shared-tensor path and shows
no dead search, so its budget exhaustion is ordinary slow progress rather
than the defect described here.

**The one-pair objective moved the wrong way.** Within the 1e-6 bar, but it
moved, and the mechanism (a flat ridge, not a better or worse optimiser) is
inferred from the lambda magnitudes rather than proved.

**Exactness claims elsewhere need re-deriving.** A sweep of the committed
research artifacts for pinned discrete-tensor numbers found that these
records can be moved at the 1e-15-to-1e-8 level by the channel route, and
the affected ones must be re-derived rather than re-toleranced:

- `notes/research/2026-09-13-many-interaction-probe.md`, which states that
  "all sixteen final/equal-P case pairs reproduced coefficients, train/test
  predictions, non-timing telemetry and retained owner payload exactly". That
  sentence is invalidated as written for any case with two or more
  discretised tensor terms.
- `notes/research/2026-09-13-many-interaction-measurements.json` (116
  `discrete=True` runs, 13,797 pinned numbers, 25 multi-pair fields).
- `notes/research/2026-09-13-cheap-interaction-measurements.json` (48
  `discrete=True` runs, 7,245 pinned numbers, 20 multi-pair fields).
- `notes/research/2026-09-14-broad-interaction-measurements.json` (98
  `discrete=True` runs, 1,544 pinned numbers).
- `notes/research/2026-09-13-real-interaction-trials-measurements.json` (7
  `discrete=True` runs), for the interaction arms only; the fresh
  before/after table above supersedes it for those three datasets, and shows
  Ames unmoved, Bike at 3.8e-12 and Breast at 3.3e-8.

Not affected, because they pin single-tensor fits that form no
tensor-by-tensor block: `2026-09-13-tensor-support-handoff-measurements.json`
and its initial variant (one `Latitude:Longitude` pair), and
`2026-09-14-targeted-interaction-measurements.json` (one tensor mention).
That distinction was read off the sweep inventory, not re-measured file by
file, and should be confirmed before any of those artifacts is re-stamped.

**Not swept.** Only the discrete engine was examined for the
composed-step-without-descent-check pattern. The SCOP EFS path already has a
`line_search_stalled` reason and the distributional and LSS smoothing paths
were not checked. Whether the shared-margin helper's own 5,000,000-cell cap
should be raised or retired, now that the channel route picks up its
declines, was not costed.

## Reproduction

Commits: E0 `0107b201462f0ce82dea39197c4b930bed35a960`, kernel
`0377263366253fc99b86e9c6646e7e947d078557`, both on
`perf/interaction-fit-cost` over v0.34.0 (`99ca0edd`). The before column was
produced against the v0.34.0 source exported with
`git archive 99ca0edd src` and imported by `PYTHONPATH`, with
`superglm.__file__` printed in each log so the arm is unambiguous.

Reproduction scripts (session scratch, not committed), under
`/tmp/claude-1000/-home-max-projects-superglm/1736a637-5efc-4cdd-a979-ce25d1497b76/scratchpad/`:

- `api-examples/time_pairs.py`, `api-examples/diagnose_scaling.py` and
  `api-examples/diagnose_pairs.py` — the recorded v0.34.0 timings, rerun
  unchanged from `interaction-cost/bench/` for the after column, with
  `interaction-cost/bench/diagnose_scaling_serial.py` the one-thread copy.
- `interaction-cost/instrument_stall.py` and its `*_3pairs`, `*_1pair`,
  `*_6pairs*` logs — the step reconstruction and the stall record.
- `interaction-cost/probe_descent_fallback.py` — the counterfactual that
  removes only the pair override.
- `interaction-cost/gate/gate_real.py`, `gate_instrument.py`, `gate_ab.py`,
  `gate_report.py` — the real-arm gate and its A/B on breaking at the first
  dead search.
- `interaction-cost/confirm_route.py`, `dispatch_probe.py`,
  `prototype_kernel.py`, `stage1_variants.py`, `stage1_cellcsr.py`,
  `oracle_small.py`, `projected_gram.py` — route confirmation, the kernel
  prototype and the oracle agreement.
- `interaction-cost/bench_channel_route.py`, `probe_fit_equivalence2.py`,
  `screening_route_check.py`, `sweep_notes.py` — the per-block and
  per-assembly bench on this tree, the fit-level equivalence with the
  re-chunking control, the screening-path route check and the artifact sweep.
- `interaction-cost/bench/trajectory_3pairs.py`, `arm6_after.py`,
  `agreement_arms.py`, `real_arms.py`, `route_probe_real.py` — the trajectory
  capture, the six-pair after arm, the zero- and one-pair agreement arms and
  the real-arm before/after and route counts measured for this note.

No timing assertion enters the test suite. The suites assert on profile
counters, cell budgets, route labels, oracle bounds and `tracemalloc` peaks.
