# Tensor support handoff: complete-fit results

Date: 2026-09-13. Final implementation:
`306f12e089234bf05213d4e2b10f3678f162d5e6`.

The unchanged rich Gaussian interaction fits faster after carrying its selected
penalty support from the optimizer into finalization. The larger housing case
uses 44.8% less complete-fit time at the median of two observations per version.
Numerical outputs and retained model payload match the baseline. Peak RSS is
2.4% higher in the smaller case; the larger-case ranges overlap. This is an
accepted time-to-fit improvement with a measured temporary-memory tradeoff.
It does not establish improved asymptotic scaling.

The [measurement receipt](2026-09-13-tensor-support-handoff-measurements.json)
contains exact observations, source/input/artifact hashes, runtime, numerical
telemetry, process snapshots, profile callers and validation evidence. The
[implementation report](2026-09-13-tensor-support-handoff-report.md) describes
the admission, ownership, mutation and serialization contracts.

## Complete-fit comparison

Both cases use 12,384 training rows from the frozen California housing split,
eight ordinary spline groups and one geographic tensor. The case names describe
row-frequency knot placement and marginal knot counts, not observation counts.
All measurements use one native worker thread and a fresh owned process.

| Case | Tensor / total coefficients excluding intercept | Baseline fit, seconds | Final fit, seconds | Median time reduction |
| --- | ---: | ---: | ---: | ---: |
| rows20 | 361 / 513 | 9.03–9.73 (median 9.38) | 6.15–6.22 (median 6.19) | 34.0% |
| rows30 | 841 / 1,013 | 129.22–133.23 (median 131.23) | 71.16–73.59 (median 72.37) | 44.8% |

| Case | Baseline fit-end peak RSS, MiB | Final fit-end peak RSS, MiB | Retained model payload, both versions |
| --- | ---: | ---: | ---: |
| rows20 | 543.57–543.69 | 556.23–556.67 | 40,694,067 bytes |
| rows30 | 901.18–916.63 | 904.31–904.40 | 170,201,663 bytes |

Fit-end RSS is the process high-water sampled immediately after the fit clock.
It includes the runtime and input data, and precedes inspection, prediction and
export. Retained payload counts distinct owning NumPy and byte buffers through
views. It excludes Python object headers, allocator slack and hidden extension
state. No unresolved external buffers were found. It is not a full heap census.

Every stored train, validation and test prediction/response array matched the
baseline by exact array comparison. MSE and all non-timing telemetry values
also matched, including lambdas and their history, EDF, dispersion, objective,
intercept, convergence reason and iteration counts. There were no fit warnings.
Both versions dispatched to the Gram direct backend with eight
DiscretizedSSPGroupMatrix groups and one DiscretizedTensorGroupMatrix.
The smoothing fits retained 9 outer iterations for rows20 and 10 for rows30.
Exact replay on this environment is evidence, not a portable tolerance policy.

These are two observations per implementation and case on a shared host, not a
confidence interval or a general speedup guarantee. Final baseline repetitions
2 and 3 bracket the two final candidate observations for each case. Repetition
2 comes from the preceding round, before the cache-scope amendment. Independent
rows20 warmups are excluded; no separate rows30 warmup was run.

No overlapping fit or test workers were launched by this session. An external
Sphinx build overlapped the final small-case observations, their last baseline,
and the start of the first final large-case observation. Process/load snapshots
record this; they are not a continuous host monitor. The earlier baseline
rows30 observation was 109.49 seconds, showing substantial variation across the
session. It remains archived and is not silently substituted into this final
bracketed comparison. The original split is a performance workload, not a new
held-out model-selection experiment.

## Where the work disappeared

Separate complete-fit profiles confirm the producer/consumer explanation:

| Profile event | Original source | Final source |
| --- | ---: | ---: |
| Total selected-support builds | 98 | 97 |
| Component-root builds | 100 | 98 |
| Two-component tensor supports | 2 | 1 |
| Singleton supports | 96 | 96 |
| Penalty-nullity calls | 22 | 22 |

Only one- and two-component families occur in this workload. Consequently,
root-build count minus support-build count identifies the tensor count; the
source trace and separate public-fit regression support this interpretation.
The 98 original support builds were never 98 large tensor decompositions.

In the original profile, the terminal nullity path accounts for 51.737
cumulative seconds. In the final profile it accounts for 0.00321 seconds;
the first bootstrap support build remains. Finalization accounts for 3.855
cumulative seconds in the final profile. Fixed-handoff admission across all
107 calls accounts for 0.0155 seconds.

These entries overlap. Their clocks come from separate instrumented runs on a
variable shared host and are not the speedup estimator. The final profile's
74.32-second fit is excluded from the timing table. External pytest/Python work
was present during that diagnostic run; its numerical telemetry still exactly
matches the final timed implementation.

## Ownership and memory amendment

The discrete optimizer now returns its populated penalty family through the
existing result carrier. Finalization admits unchanged fixed-coordinate tensor
support using exact construction inputs and selected-support/error evidence.
It creates fresh mutable owners, starts fresh weighted/face state, and releases
the obsolete optimizer owner. The final owner retains the selected support and
error ledger without retaining duplicate handoff authorization snapshots.
The numerical rank policy and tolerances are unchanged.

The [first implementation measurements](2026-09-13-tensor-support-handoff-initial-measurements.json),
on `90f519f3`, exposed unnecessary entry-family snapshots. Its median fit-end
RSS was 561.60 MiB for rows20 and 937.06 MiB for rows30, with exactly the same
retained payload as baseline. These observations remain archived.

The amendment captures fixed provenance only during fresh construction inside
an explicit component cache. All three discrete-optimizer construction sites
use its existing per-fit cache; the unused entry family does not. Cache hits
and later support evaluation cannot grant authority retrospectively.

For float64 tensor width p and component spectral ranks r1 and r2, removing the
unused entry input record saves exactly

    8 * (5*p*p + r1 + r2) bytes

of live payload: two raw matrices, two solver matrices, the coordinate map and
the two spectral vectors. That is 5,218,312 bytes for rows20 and 28,304,232 bytes
for rows30. It excludes Python overhead and predicts live bytes, not an exact
RSS reduction. The optimizer's necessary input/support snapshots and temporary
comparison bytes still have a cost. The smaller-case RSS overhead remains.

The dominant tensor-support contribution changes from two builds to one build
plus validation. Dense factorizations and products retain their cubic operation
counts in the conventional fixed-precision model; dense storage and these input
snapshots remain quadratic in p. Iteration, trace and covariance costs still
belong in the complete-fit accounting. See the
[cost investigation](2026-09-13-adaptive-interaction-costs.md).

## Verification and remaining scope

- The final change passes 139 focused tests. Analytic rank, determinant,
  derivative, reconstruction, active-face, public prediction and dispersion
  checks accompany independent dispatch and lifetime regressions.
- Six deliberate mutations of receipt, input, arithmetic, summary or face-rank
  checks were each caught. The uncached entry regression failed before the
  amendment and passed afterward.
- The broad non-slow run on the first implementation reported 14,323 passed,
  560 skipped and 144 slow tests deselected. Its sole failure came from the
  parent's NUMBA_NUM_THREADS=1 setting while a test requests two workers.
  The unchanged baseline reproduced that same ValueError. All 121 tests in
  the three worker-budget modules passed with 16 workers permitted, including
  the affected case and previously skipped parallel cases.
- All three freMTPL2-dependent suites ran with SUPERGLM_REQUIRE_DATA=1 and
  verified local frequency/severity datasets. Optional-dependency and browser
  skips are not represented as executed tests.
- Ruff, formatting, lock, dependency and diff checks passed. The ordinary
  end-to-end example converged in 19 iterations with identical printed output
  on baseline and the initial implementation, including its existing non-asserting
  U-shape CHECK.
- Independent Astra/max reviews found no blocking findings in the initial
  implementation and the exact committed cache-scope amendment.

The broad run predates the final guard amendment; the focused suite and source
review cover that narrow delta. Detailed commands and red evidence are in the
implementation report and structured receipt.

This completes the first existing-tensor performance step supporting C21.
C16 adaptive refinement, C15 matrix-free fitting and C18 hierarchy/recycling
remain ahead; C9 search still depends on affordable validated candidate fits.
The existing Lean archive proves its stated exact quadratic identities. Neither
this Python handoff nor a complete adaptive floating-point fitter is claimed
to be Lean-verified.

To repeat a case from the corresponding checkout, use the unchanged runner:

~~~sh
uv run --no-sync python benchmarks/benchmark_housing_tensor.py \
  --case rows30 --data /path/to/california_housing.parquet \
  --output .benchmark-artifacts/tensor-reuse/new-unique-run
~~~

Use a separate invocation with --profile for caller/callee analysis. Retain the
runner's source/runtime/input fingerprints and distinguish fit-end RSS from
the later process peak. Source baseline and final SHAs are pinned in the receipt.
