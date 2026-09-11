# Compact LSS smoothing history

On the million-row C1 fixture, retained historical row buffers fall from
256 to 128 MiB and median peak process RSS falls from 1702.69 to 1569.94 MiB.
Saved numerical outputs are bitwise identical. Median fit time increases from
13.806 to 14.197 seconds across three pairs; this is a memory improvement,
with no speedup claimed.

LSS smoothing now releases obsolete historical `eta` and `theta` arrays during
optimization. The terminal fit and recent plateau checks retain their rows;
every historical entry retains coefficients, diagnostics and endpoint evidence.
Set `retain_history_rows=True` on `SuperLSS.fit_reml` or
`DistributionalEFSConfig` when a test or debugging session needs every fit's rows.
The separate `retain_rows` default and scalar fit-state retention are unchanged.

## Complete-fit results

The [receipt](compact_lss_history_receipt.json) records every fit, source and
output hashes, phase counts, memory measurements and separate profile evidence.

| Million-row discrete fit | Baseline, seconds | Compact history, seconds |
| --- | ---: | ---: |
| Pair 1 | 14.880 | 14.197 |
| Pair 2 | 13.806 | 14.169 |
| Pair 3 | 13.795 | 14.492 |
| Median | 13.806 | 14.197 |

The median increase is 0.391 seconds, or 2.83%; the signs of the paired changes
differ. Median curvature/gradient assembly increases from 7.493 to 8.025 seconds,
while predictor compilation decreases from 2.847 to 2.691 seconds. Phase counts
are unchanged. Three pairs on this host do not identify a repeatable source of
the small timing difference, and the phase medians should not be added together.
The roadmap's 12.5-second target remains open.

All eight timed fits and both separate profiles exactly match their respective
baseline's coefficients, covariance, training/holdout parameters, lambdas,
terminal score and curvature. Numerical history, convergence, EDF, rank,
iterations and backend identifiers also match. The million-row fits all use
16 inner and seven smoothing iterations, finish at `practical_plateau`, and
retain eight coefficient-space history entries. Four entries retain rows after
compaction. The execution backend is `distributional-chunked-v1` throughout.

The dense 65,536-row control takes 2.236 versus 2.174 seconds in one pair, with
identical outputs and `distributional-dense-v1` throughout. Historical row
buffers fall from 20 to 8 MiB. Peak process RSS is essentially unchanged at
612.20 versus 611.71 MiB. Fit-end resident RSS increases from 553.74 to
604.09 MiB in this pair, so reduced owned row buffers do not establish a dense
process-memory saving. RSS includes other allocations and allocator retention;
the experiment does not separate those contributions.

## Profile and reuse review

Separate million-row profiles show the same nine fixed-penalty solver calls
(including the null fit), 18 chunked geometry assemblies, 16 fused trial
evaluations and 306 likelihood-chunk iterator calls. Source certification and
prepared likelihood work retain their existing paths. There are no additional
refits or geometry assemblies in the candidate.

The eight compaction calls take 0.00145 seconds inclusive in the candidate
profile. They preserve live cache sources and do not reconstruct row arrays.
The expensive work remains predictor compilation, likelihood rows and chunked
geometry; compaction is not responsible for an extra solve or factorization.
Profiled wall times are kept separate from the uninstrumented comparisons.

## Lifetime and numerical evidence

Compaction replaces entries in the history list without mutating live solver,
retry or endpoint results. A dense reuse registry also held strong references
to obsolete fits. It now uses the existing chunked registry's weak-reference
pattern, so authenticated live results remain reusable without keeping dead
sources alive. No new factor cache or changed invalidation rule is introduced.

Compact entries record their original two-dimensional shape. Publication checks
each historical shape against the real terminal arrays. Consumers that need
rows explicitly require them. Legacy full-array artifacts derive their shape
when loaded; migration refuses a preexisting terminal-identity mismatch.

The focused regressions require exact compact/full agreement for coefficients,
predictions, covariance, smoothing parameters, objective history, endpoint
evidence and diagnosis, for EFS and Newton. Weak-reference probes check actual
array lifetimes across iterations and reuse after compaction. Those probes
failed against the original strong registry even after list entries compacted.
Malformed or rehashed historical shapes are refused. Tests that reconstruct an
earlier terminal fit explicitly retain full history.

All 441 affected regression cases pass on the exact candidate source, including
serialization, EFS, practical stopping, endpoint evidence, reuse, Newton and the
Tweedie replay. The independent Astra xhigh review approved the final change
after correcting an identity-dependent test. Ruff, formatting and the strict
documentation build pass. Type checking adds no diagnostics to the existing
backlog (891 on both isolated source snapshots).

## Measurement protocol

The baseline is exact commit `a6e2ca79eb5a69abd4da9c829ceea09a24103b9f`.
The candidate overlays only the twelve changed distributional production files.
Both exclude the unrelated local coefficient-QR experiment. Source, input,
harness and output hashes are recorded in the accompanying receipt.

The runner is [compact_lss_history.py](compact_lss_history.py), using the existing
fragmented Gaussian location/scale C1 fixture: 1,048,576 training observations,
102 coefficients, four spline knots, 256 bins and a 2,000-row holdout. It uses
EFS, practical convergence and the same explicit initial lambda of 0.1 on both
versions. Three baseline/candidate pairs run sequentially in fresh processes
with one BLAS thread and 16 Numba workers. Standard public warmup, data creation
and prediction are outside the fit clock. Profiling uses separate fits.

Python is 3.13.14, NumPy 2.5.2 and SciPy 1.18.0. Dependencies are unchanged.
Every run passes the existing external-CPU activity screen, with observed
external CPU between 0.116 and 0.248 cores. Endpoint process accounting is a
lower bound, not proof of an idle host throughout a fit.

A dense 65,536-row control exercises the dense reuse registry and omits optional
final row diagnostics. The million-row fits keep the default final diagnostics.
Both routes compare complete-fit time, peak process RSS, retained solver row
buffers, numerical outputs, iteration counts and actual execution backends.

Peak process RSS includes setup and warmup. The solver-row census counts arrays
reachable through smoothing dataclasses and containers, including endpoint
contexts; it is not a complete Python heap census. The training frame and
prepared likelihood inputs still contain full rows. This change does not
establish bounded whole-fit memory or the roadmap's larger-N capacity milestones.

To reproduce one million-row fit, point `--source` at a checkout containing
`src/superglm` and keep output outside pinned benchmark inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=16 \
  uv run --no-sync python benchmarks/compact_lss_history.py \
  --source /path/to/checkout --out /tmp/compact-history-fit.json
```

Add `--profile` for a separate instrumented fit. Use `--n 65536 --no-discrete
--retain-rows 0` for the dense control, or `--history full` to retain historical
arrays on the candidate. Raw JSON, NPZ and profile artifacts remain locally under
`.superpowers/sdd/2026-09-11-compact-lss-history/`; their hashes and the numerical
comparison results are retained in the tracked receipt.
