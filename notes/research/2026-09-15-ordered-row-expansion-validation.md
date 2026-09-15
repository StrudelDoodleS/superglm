# Ordered row expansion: complete-fit validation

The CSR change was measured on production code at `0d84e6ca53b114929c13f8212b58eb1a6adee6bc`.
The baseline substitutes the LIL helper from master `ebd30f9a1640fde1275fa5e67ae855288d8985a7`
into the same checkout. Each measured call runs the entire `SuperGLM.fit_reml`,
including design construction, coefficient fitting and smoothing selection.

## Reproduce

The [benchmark](../../benchmarks/benchmark_ordered_row_expansion.py) generates
all inputs and asserts equality of fitted outputs and actual backend dispatch.
The [raw result](../../benchmarks/results/ordered_row_expansion_20260915.json)
contains every repetition, profile, memory measurement and environment record.

```bash
uv sync --python 3.13 --extra dev --extra plotting
uv run --no-sync python benchmarks/benchmark_ordered_row_expansion.py \
  --rows 100000 --repeats 3 --seed 64123 --output /tmp/ordered-row-expansion.json
```

The fixture has 100,000 generated observations and 20 fitted coefficients.
It uses Tweedie power 1.5, two ordered categorical CR terms with five uniform
knots, twelve regular levels and an additional special level, and one six-level
categorical term. This exercises repeated sparse basis rows and special-row
expansion in a complete model with many observations and modest basis width.

## Timing and process memory

Each variant was warmed before three paired fits in alternating order. Timing
runs have no profiler, optimizer spy or memory sampler. cProfile and dispatch
spies run separately. No thread environment variables were overridden; inside
fits, both OpenBLAS pools used one thread and OpenMP used sixteen.

| Discrete | DM build, LIL / CSR | Complete fit, LIL / CSR | Peak RSS, LIL / CSR | Peak RSS increase, LIL / CSR |
| --- | --- | --- | --- | --- |
| `True` | 0.686 / 0.231 s | 2.939 / 2.407 s | 472.8 / 463.3 MiB | 70.9 / 60.3 MiB |
| `False` | 0.629 / 0.239 s | 2.486 / 2.066 s | 464.5 / 454.5 MiB | 61.3 / 52.1 MiB |

Timing columns are medians. Memory uses a fresh subprocess per variant and
discrete setting. After a warm fit, garbage collection and `malloc_trim`, RSS
is sampled every 2 ms during another full fit. The baseline is measured with
the input data retained. Retained RSS increases after that fit were 50.1 / 45.5
MiB for discrete and 44.4 / 34.4 MiB for non-discrete. These are single sampled
memory runs, so allocator variation and missed short peaks limit precision.

## Numerical outputs and actual dispatch

All paired fits have bit-identical coefficients, intercept, predictions,
smoothing parameters and deviance. Every fit and REML optimization converged.
Both implementations take four outer iterations and use the Gram solver.

The actual group storage is two `SupportCompressedSSPGroupMatrix` blocks and
three `CategoricalGroupMatrix` blocks in both variants and both modes. Separate
spies record one cached-W optimizer call for each discrete fit; non-discrete
fits make five exact-solver calls and no cached-W call. The raw result records
these observations rather than inferring dispatch from the requested mode.

## Cost attribution and reuse

cProfile records two row-expansion calls per fit. Their cumulative time falls
from 0.498 to 0.004 s in the discrete profile and from 0.417 to 0.004 s in the
non-discrete profile. The two LIL sparse-row assignment calls disappear. These
instrumented timings attribute the cost; the table uses separate ordinary fits.

`OrderedCategorical._build_spline` produces the compact basis and passes it to
`_expand_rows` before the design builder consumes the expanded `GroupInfo`.
CSR expansion reuses the compact copy's data and column-index arrays, inserting
empty rows through its row pointers. The original basis is copied before
canonicalization, so its ownership and contents remain unchanged. Reuse lasts
for the returned matrix's lifetime; no result is cached across builds whose
basis values or ordered-row masks could differ.

The allocation regression in `tests/test_ordered_row_expansion.py` fails on the
LIL implementation. Separate numerical tests cover row placement, sparse
formats, zero rows, empty dimensions, noncanonical input and metadata retention.
