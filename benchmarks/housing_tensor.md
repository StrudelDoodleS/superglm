# California housing geographic tensor benchmark

This benchmark fixes a model whose richer geographic surface improved test
MSE from 0.254498 to 0.237449, but whose observed fit time increased from
8.77 to 117.41 seconds. It measures the cost of using that extra capacity.

The two-dimensional interaction grows from 19 x 19 = 361 to 29 x 29 = 841
columns when marginal k increases from 20 to 30. Cubic work on that block
would grow by about 12.6 times. The observed 13.4-fold slowdown is therefore
not sufficient evidence of a performance bug. It is a workload for examining
the actual computations, opportunities for reuse, and alternatives to dense
coefficient algebra.

## Run one fit

The default runs the selected k=30 model in a fresh process with one native
thread and a 180-second process deadline:

```bash
uv run --no-sync python benchmarks/benchmark_housing_tensor.py
```

The runner uses scikit-learn's California housing cache and downloads the
data if necessary. To use the original local parquet instead:

```bash
uv run --no-sync python benchmarks/benchmark_housing_tensor.py \
  --case rows30 \
  --data .benchmark-artifacts/psst-detection-study/real-probe-20260913/california_housing.parquet
```

Run the smaller comparison separately:

```bash
uv run --no-sync python benchmarks/benchmark_housing_tensor.py --case rows20
```

Each invocation runs exactly one fit. There is no implicit warmup or tuning
sweep. Output goes into a new timestamped directory beneath
`.benchmark-artifacts/housing-tensor/`. `--output` chooses a different new
directory; an existing directory is refused. `--timeout` sets the worker's
deadline, including imports and data loading. Timeout and worker errors
produce a nonzero exit and a `run.json` record. The parent kills and reaps its
worker when the deadline expires.

## Profile separately

```bash
uv run --no-sync python benchmarks/benchmark_housing_tensor.py --case rows30 --profile
```

This records `fit.prof` and `profile.txt`, including cumulative costs,
callers and callees for the complete `fit_reml` call. Profiled results are
explicitly marked and should not be compared with unprofiled fit clocks.
No profiling run is part of the historical reference.

## Frozen specification

Use all 20,640 rows of the California housing dataset, in its original order.
The target is `MedHouseVal` in units of $100,000. The numeric data fingerprint
includes the eight raw predictors and the response, so parquet metadata does
not affect identity. Row reorderings and response changes are refused before
the split.

Apply `log1p` to `AveRooms`, `AveBedrms`, `Population` and `AveOccup`. The other
predictors are `MedInc`, `HouseAge`, `Latitude` and `Longitude`. Use NumPy's
`default_rng(202609131).permutation(20640)`, allocating the first 12,384 rows
to training, the next 4,128 to validation and the remaining 4,128 to test.

All models use Gaussian identity-link SuperGLM, eight natural cubic main
effects, SSP penalties, zero selection penalty, 256-bin discretization, and
only the latitude-longitude tensor interaction. Fit once with `fit_reml`
and unit weights. The six non-geographic main-effect specifications always
use k=20 and `knot_strategy="quantile"`. Geographic specifications are:

| Case | Geographic k | Knot strategy | Model coefficients excluding intercept |
| --- | ---: | --- | ---: |
| `base20` | 20 | `quantile` | 513 |
| `rows20` | 20 | `quantile_rows` | 513 |
| `support30` | 30 | `quantile` | 1,013 |
| `rows30` | 30 | `quantile_rows` | 1,013 |

`quantile` uses unique coordinate values. `quantile_rows` weights coordinate
values by their frequency among training rows. Both geographic main effects
and the interaction inherit the selected specification. All coefficients
and smoothing parameters are jointly refitted.

## Reference and outputs

`housing_tensor_reference.json` records data and split fingerprints, the
source revision and source hash, historical fit diagnostics, dimensions,
losses, EDF and timing observations. `housing_tensor_predictions.npz` contains
the four models' validation/test predictions and the corresponding responses.
The runner verifies the archive hash before fitting.

The reference preserves package versions recorded by the preceding real-data
probe. The historical basis sweep did not capture its own Python, SciPy, BLAS
or hardware details; those fields remain explicitly unknown. New runs record
their actual runtime. The runner also refuses an imported SuperGLM package
outside this checkout, so its source hash describes the code being executed.

Each successful invocation saves:

- `result.json`: complete-fit seconds, process peak RSS, package versions,
  thread-pool configuration, source/script hashes, actual matrix classes,
  solver dispatch, consumed reference-file hashes and full training telemetry.
- `predictions.npz`: training, validation and test responses and predictions.
- Prediction differences from the committed reference, including maximum
  absolute difference, RMS difference, exact equality and MSE difference.
- `worker.log` and `run.json`: the owned process's output and completion state.

The fit clock excludes imports, data loading, prediction, telemetry export
and profile-file writing. Peak RSS covers the worker process through
prediction export, including its runtime and data. It is not the model's
retained memory. A successful run requires convergence, finite predictions,
the expected basis width, and matching data/split identity. Prediction
differences are measured, not silently accepted as numerically equivalent.
For a solver change, justify comparison tolerances from its numerical
contracts and certify stable observables. Exact equality is a useful replay
check in an unchanged local environment, not a portable BLAS requirement.

## Historical observations

| Case | Validation MSE | Test MSE | Fit seconds | Peak process RSS, MiB | REML iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base20` | 0.289667 | 0.291782 | 8.67 | 543 | 11 |
| `rows20` | 0.247772 | 0.254498 | 8.77 | 543 | 9 |
| `support30` | 0.272402 | 0.270184 | 117.83 | 901 | 11 |
| `rows30` | 0.228505 | 0.237449 | 117.41 | 916 | 10 |

These observations came from three concurrent fresh processes, each with one
native thread. They are not an isolated performance baseline, a latency
promise, or a CI timing threshold. Calibrate timing with sequential isolated
runs before evaluating an optimization. The runner does not add a hosted-CI
speed gate.

The historical sweep completed four fits. Six other controls were incomplete.
A Python 3.13 timeout-cleanup error extended that sweep from its 180-second
deadline to about four minutes. The committed runner uses a separate owned
subprocess with tested kill-and-reap cleanup and does not repeat that sweep.

The specification was chosen after inspecting this random split. This is
a fixed numerical/performance workload, not a fresh predictive benchmark or
a test of geographic extrapolation. No new public data inputs or interaction
pairs were added to obtain the selected model's accuracy.

## Optimization review

Compare complete unprofiled fits, peak and retained memory, stable numerical
outputs, iterations and actual dispatch against an isolated baseline. Profile
separately to trace expensive results from construction to use, including
penalty geometry, factorizations, smoothing derivatives and finalization.
Establish when any reused value becomes invalid. Preserve numerical
certification and the fitted model while measuring the effect of each change.

The existing timers show about 61 of the selected fit's 117 seconds inside
the timed optimizer region. The remainder lies outside that region and needs
attribution. Nested timers must not be added as if they were disjoint phases.
