# Many-interaction complete-fit probe

Date: 2026-09-13. Production source: `d7f231e3` on
`research/adaptive-interactions`. This is a research baseline, with no new
production solver or claimed speedup.

## Question and protocol

The preceding housing measurement has one tensor term at two widths. It cannot
establish how many-interaction GAMs scale, or a mathematical limit. This probe
varies term count separately from per-term width and records automatic smoothing
as well as a fixed smoothing fit. The companion
[cost analysis](2026-09-13-many-interaction-scaling-analysis.md) distinguishes
implementation costs, conditional alternatives and genuine model constraints.

The executable is
[benchmark_many_interactions.py](../../benchmarks/benchmark_many_interactions.py).
Every case runs in a new owned worker, with a 90-second whole-worker deadline
and OpenBLAS, OMP, MKL and Numba thread counts set to one. Workers run serially.
Those four controls describe the measured runs. The current runner also sets
Accelerate and BLIS limits, converts macOS RSS bytes to MiB, and refuses an
observed multithreaded pool. The [pre-merge validation record](
2026-09-14-interaction-review-validation.md#pre-merge-review-and-portable-timing-evidence)
documents these corrections without changing the measurements.
Imports and deterministic input generation precede the fit clock. The clock
covers the entire public `fit` or `fit_reml` call, including construction and
finalization. The fit-end process high-water is sampled before telemetry,
prediction, retained-storage inspection or export. It includes runtime and
inputs, so it is not incremental model memory. Retained storage uses the
previously validated NumPy/byte owner accounting helper, with its exclusions.

All cases use eight independent uniform features, Gaussian responses, cubic
regression splines, 64 discretization bins, zero selection penalty and initial
or fixed spline penalty 0.1. The response law includes the same main effects and
all 28 pair effects in every case. Training has 2,048 rows and the independently
generated test set has 1,024 rows. Seeds, exact input hashes and predictions are
recorded. Test MSE describes this synthetic law; it does not establish general
accuracy or equivalence between models with different interactions.

The primary ladder fixes `k=6`, giving five coefficients per marginal and 25
per tensor. It adds `M = 1, 2, 4, 8, 16, 28` edges from a deterministic ordering
of the complete eight-feature graph. Every consecutive round contains four
disjoint pairs. Thus total coefficient count, excluding the intercept, is
`P = 40 + 25 M`; the nominal penalty component count is `q = 8 + 2 M`.
Fixed smoothing estimates no smoothing parameters. The REML arm uses the
public defaults, including the 20-iteration limit and unchanged tolerances.
An iteration-limited fit remains in the receipt and is excluded from claims
about time to a converged fit.

Each final case has two repetitions, with the count order reversed for the
second pass. This is a small descriptive probe on a shared host, not enough
replication for confidence intervals or a fitted scaling exponent. No owned
tests, Lean compiler, other fit workers or profiled fits run concurrently with
the uninstrumented timings. Other users' activity is not controlled.

An equal-total-width comparison uses four interactions at `k=9` and eighteen
at `k=5`. Both have `P=320`, but different allocations: 64 main plus 256 tensor
coefficients versus 32 main plus 288 tensor coefficients. Their nominal penalty
counts are 16 and 44. This is an equal-P control, not an equal-function-space
comparison: marginal resolution, tensor resolution, graph and smoothing
dimension all differ. It can show why total P alone is insufficient; it cannot
attribute a difference to one of those factors by itself.

## Instrumentation revisions

The initial 24-run pilot is retained separately. It already recorded actual
group classes and REML dispatch, but omitted the result's resolved direct
backend field. Its `fitted_smoothing_parameter_count` also counted a configured
lambda entry in fixed mode. The final runner explicitly records
`model.result.direct_backend` and reports zero fitted smoothing parameters in
fixed mode. The entire primary ladder was rerun with that instrumentation;
these metadata corrections do not change the fitted model or solver. Per-run
script hashes distinguish the versions. Final tables use the second ladder,
and the receipt preserves the first instead of silently rewriting it.

## Results

The [measurement archive](2026-09-13-many-interaction-measurements.json) contains
58 worker receipts: 24 pilot runs, 24 final ladder runs, eight equal-P runs and
two separate profiles. All workers returned normally within their deadlines.
The one-interaction REML case reached `max_reml_iter` in both repetitions of
both ladders. Every other fitted case converged. The final and equal-P cases all
recorded the `gram` backend, eight `DiscretizedSSPGroupMatrix` main effects and
the requested number of `DiscretizedTensorGroupMatrix` interactions.

The table reports final medians over two independent worker processes. Memory
columns below are for REML; fixed-fit memory is in the receipt. One MiB is
1,048,576 bytes. `P` excludes the intercept.

| M | P | Estimated smoothing parameters | Fixed fit (s) | REML fit (s) | REML outer iterations | REML peak RSS (MiB) | REML retained payload (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65 | 10 | 0.243 | iteration limit: 1.543 | 20, incomplete | 390.90 | 0.97 |
| 2 | 90 | 12 | 0.257 | 0.626 | 6 | 387.46 | 1.40 |
| 4 | 140 | 16 | 0.243 | 0.763 | 7 | 389.97 | 2.61 |
| 8 | 240 | 24 | 0.317 | 1.531 | 11 | 398.91 | 5.36 |
| 16 | 440 | 40 | 0.490 | 3.582 | 13 | 429.13 | 12.22 |
| 28 | 740 | 64 | 0.950 | 4.953 | 6 | 500.72 | 25.95 |

The two final M=28 REML fits took 4.910 and 4.997 seconds. The corresponding
M=2 fits took 0.580 and 0.672 seconds. Increasing M changes the statistical
model as well as dimensions. Iteration counts are not monotone, and fixed-fit
times at the small end overlap run variation. Fitting a power law through these
medians would hide setup overhead, geometry, stopping behavior and backend
crossovers. These runs establish a reproducible baseline, not an empirical
asymptotic exponent.

At equal `P=320`, the four-term `k=9` model took 0.302 s with fixed smoothing
and 1.010 s with REML (six outer iterations; 16 smoothing parameters). The
eighteen-term `k=5` model took 0.394 s and 2.761 s (eleven outer iterations;
44 parameters). REML peak RSS was 411.56 versus 408.95 MiB, and retained payload
was 8.69 versus 7.98 MiB. Time and memory need not move together. Total P alone
does not predict complete-fit cost, and this comparison does not isolate why.

All sixteen final/equal-P case pairs reproduced coefficients, train/test
predictions, non-timing telemetry and retained owner payload exactly. Every
retained-storage traversal reported zero unresolved buffers. Exact local
replay is a repeatability check, not a portable forward-error tolerance or a
certificate comparing different models. Test MSE for the final M=28 REML fit
was 0.11046; the synthetic noise variance is 0.09, and omitted interaction
signals explain much of the loss in smaller models. No generalization claim
beyond this fixture follows.

## The bottleneck changes with interaction shape

The separate M=28 REML profile took 8.094 seconds. Profiling overhead is large
on this small, call-heavy problem; these times are excluded from the timing
table and cannot be subtracted from its medians to predict a speedup.

| Profile entry | Calls | Cumulative seconds |
| --- | ---: | ---: |
| `build_centered_system` | 9 | 5.247 |
| `_cross_gram` | 5,670 | 4.997 |
| `_cross_gram_tensor_tensor_shared_margin` | 3,402 | 4.242 |
| `einsum_path` | 96,768 | 2.343 |
| `_decompose_gram` | 143 | 0.667 |
| `_penalty_support` | 92 | 0.284 |

Rows in this table overlap and must not be added. The shared-margin entry is
a dispatcher; its invocation count is not the number of admitted three-margin
histograms. With 36 total groups, each assembly requests all
`36 * 35 / 2 = 630` distinct cross-group blocks, including
`28 * 27 / 2 = 378` tensor pairs. Nine assemblies explain the 5,670 and 3,402
dispatch counts. The fixed-fit profile builds one centered system and calls
the two dispatchers 630 and 378 times respectively. Its profile took 1.704 s,
of which the centered build accounts for 0.770 s.

For a simple interaction graph, the number of pairs sharing a feature is
`J = sum_v choose(degree(v), 2)`. Here every feature has degree seven, so
`J = 8 * 21 = 168`. The shared-margin route invokes contraction planning once
per shared-feature bin. Nine builds with 64 bins therefore give
`9 * 168 * 64 = 96,768` plans, exactly the recorded `einsum_path` count. This
is a source/graph operation count, not an estimated timing exponent. It also
shows why interaction count alone does not describe the dispatch cost.

The earlier single-wide housing profile spent 60.49 of 74.32 seconds on support
construction. Here support takes only 0.284 seconds and repeated pairwise
assembly dominates. Neither profile establishes the large-P asymptote. Together
they show why we need a width-distribution and interaction-count cost model,
and why solving one bottleneck cannot establish a mathematical limit.

## Next experiments and numerical obligations

The [structured support analysis](2026-09-13-structured-tensor-support-analysis.md)
examines rank-only work and marginal tensor certificates for wide terms. The
[many-interaction analysis](2026-09-13-many-interaction-scaling-analysis.md)
examines repeated assembly, aggregate histogram cache storage, the coupled
operator, smoothing dimension and nullspace growth. These are distinct paths
to cheaper fits; improvements must compose in a complete fit.

For the many-term regime, first determine which raw moments or contraction
plans survive an exact basis/weight/coordinate check across smoothing updates.
The caller audit traces eight solver invocations to the optimizer (bootstrap,
six one-step candidates and its final full refit), and one to finalization.
Each invocation owns a fresh constant-weight centered-system cache. Reuse works
inside an invocation but does not survive the next call. Gaussian/identity
weights are constant, while ordinary SSP coordinate maps change with lambda;
unprojected multi-penalty tensor groups remain unchanged. This makes unchanged
tensor-by-tensor blocks or their raw moments a concrete first reuse candidate.
The completed global transformed matrix cannot simply be carried unchanged.
Charge retained caches against peak and retained memory before adding one.
Then compare a coupled matrix-free action with the current compiled dense
action, under derived residual/error tolerances and measured preconditioning
costs. The [Lean proof](2026-09-13-interaction-operator-proof.md) establishes the
exact row-accumulation identity; it supplies no iteration or rounding guarantee.

The original C16/C21 adaptive representation experiment remains necessary to
reduce coefficients per term. C15/C18 become a measured parallel requirement
for many coupled terms. Extend the ladder by varying rows, width distribution,
interaction graph and predictor correlation independently, then approach the
1k/5k/20k coefficient gates under explicit memory budgets. Report stopping
failures and requested uncertainty costs. General C9 search remains downstream
of affordable validated candidate fits.

## Verification and limits

The collector verified equal source/input hashes across all 58 cases and
exact repeated numerical arrays, non-timing telemetry and retained storage
for all sixteen final/equal-P pairs. The receipt preserves all run metadata,
warnings, full result telemetry, numerical artifact hashes and profile tables.
The local immutable worker files and compressed prediction arrays are under
`.benchmark-artifacts/many-interactions/`; the runner regenerates them. No
profile timing is pooled into the uninstrumented statistics.

Focused Ruff checks and format checks passed for the new benchmark runner.
The production package is unchanged during this probe; no new solver behavior
is claimed. The full production test suite was not repeated for a research
runner and documentation change. This run does not measure row scaling,
correlated designs, many wide terms, adaptive accuracy, stochastic REML error,
or a matrix-free implementation. Those remain open experimental/theoretical
questions, not evidence of an exhausted frontier.
