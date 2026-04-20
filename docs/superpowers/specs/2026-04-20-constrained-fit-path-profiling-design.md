# Constrained Fit Path Profiling Design

## Goal

Build a profiling and scaling study for constrained fit paths so we can see,
with evidence, where exact-path time and memory go before making the next
optimization change.

The study should answer:

- where exact `SCOP` and exact `QP` time is actually spent
- how performance scales with row count, basis size, and number of constrained
  features
- when `discrete=True` changes the scaling story
- whether the bottlenecks differ between synthetic and real `freMTPL`-style workloads

## Scope

This study covers fit-path profiling only. It does not change solver behavior.

Included:

- single-feature exact `SCOP`
- single-feature exact `QP`
- multiple constrained features in one model
- exact vs `discrete=True`
- synthetic scaling sweeps
- `freMTPL` real-data sanity checks
- call-stack profiling
- memory / allocation profiling

Excluded:

- tensor / interaction terms
- mixed-shape API work
- public API changes
- optimization changes beyond instrumentation and benchmark harnesses

## Questions To Answer

### Exact-vs-Discrete

- How much faster is `discrete=True` than exact for each constrained engine?
- Is the gap dominated by per-iteration cost or by more iterations?

### SCOP-vs-QP

- Does exact `SCOP` spend most of its time in state setup, weighted products,
  Newton algebra, or line search?
- Does exact `QP` spend most of its time in model-level Gram assembly,
  constraint assembly, or the QP solve itself?

### Scaling Axes

How do runtime and memory scale with:

- `n` rows, up to about `500k`
- constrained-feature count
- basis size / `k`
- engine (`SCOP` vs `QP`)
- exact vs `discrete=True`

### Real-Data Check

- Do the same hotspots appear on `freMTPL` as on synthetic stress cases?
- Are the scaling relationships directionally consistent on real data?

## Datasets

### Synthetic

Synthetic datasets should be used for controlled sweeps because they let us
vary one axis at a time.

Required synthetic scenarios:

1. single constrained `SCOP` feature
2. single constrained `QP` feature
3. multiple constrained features in one model

For each scenario, vary:

- `n`
- `k`
- exact vs `discrete=True`

The synthetic generator should be able to produce:

- repeated-support covariates
- mostly-unique covariates

so we can see whether exact support compression changes the scaling story.

### Real Data

Use `freMTPL` for sanity checks.

The real-data runs are not the main scaling sweep. They are there to answer:

- do the same hotspots matter on actual workloads?
- are the optimization priorities still the same on real data?

## Profiling Methods

Use both:

### 1. Timing

Collect:

- end-to-end fit time
- `n_reml_iter`
- `n_pirls_iter`

for each benchmark configuration

### 2. Call-Stack Profiling

Use a Python profiler such as:

- `cProfile`
- `pyinstrument`

to identify the dominant call-stack hotspots in constrained fits.

The profiling output should make it obvious whether the bottleneck is in:

- design / setup
- weighted-product formation
- constrained solver steps
- objective / line search
- outer REML orchestration

### 3. Memory / Allocation Profiling

Use a lightweight memory/allocation profiler such as:

- `tracemalloc`
- process RSS sampling

to capture:

- peak memory
- allocation-heavy stages

This is important because some exact-path slowdowns may be driven more by
temporary array churn than by floating-point arithmetic alone.

## Benchmark Matrix

### Primary Synthetic Sweeps

#### Sweep A: row count

Vary:

- `n = 10k, 50k, 100k, 250k, 500k`

for:

- single-feature `SCOP`
- single-feature `QP`
- multiple constrained features

and for each:

- exact
- `discrete=True`

#### Sweep B: basis size

Vary:

- `k = 10, 20, 40`

for the same model families.

#### Sweep C: constrained-feature count

Vary:

- `1, 2, 4, 8` constrained features

for:

- all-`SCOP`
- all-`QP`
- optionally mixed-engine models if the current implementation supports them cleanly

### Real-Data Checks

Use `freMTPL` to compare:

- single constrained `SCOP`
- single constrained `QP`
- multiple constrained features when feasible

with exact vs `discrete=True`.

## Outputs

The study should produce:

- CSV or parquet summary tables
- a compact human-readable summary in stdout
- at least one plot or table for:
  - runtime vs `n`
  - runtime vs constrained-feature count
  - runtime vs `k`
- at least one call-stack profile artifact per major scenario
- at least one memory/allocation artifact per major scenario

## Success Criteria

This phase is successful if it gives us a clear next target for optimization.

Specifically, we should be able to say:

- exact `SCOP` spends most of its time in `X`
- exact `QP` spends most of its time in `Y`
- multi-feature constrained models are dominated by `Z`

If the output does not isolate the main hotspots, the study is incomplete.

## What A Good Result Looks Like

A good result is not “lots of numbers.”

A good result is a small set of defensible conclusions such as:

- exact single-feature `SCOP` is dominated by row-level weighted products
- exact `QP` is dominated by model-level Gram assembly
- multi-feature scaling is dominated by cross-block products rather than per-feature setup
- support compression changes the slope on repeated-support data but not unique-support data

Those are the findings that unlock the next engineering move.

## Risks

### Risk: profiling overhead obscures the real picture

Mitigation:

- use low-overhead timing for the main sweeps
- reserve deeper call-stack profiling for representative scenarios

### Risk: too many axes at once

Mitigation:

- synthetic sweeps vary one primary axis at a time
- real-data runs are sanity checks, not the whole study

### Risk: outputs are noisy but not actionable

Mitigation:

- require explicit hotspot summaries
- keep the matrix small enough that we can interpret it

## Acceptance Criteria

1. The study covers both single-feature and multi-feature constrained models.
2. The study covers both `SCOP` and `QP`.
3. The study covers exact vs `discrete=True`.
4. The study includes synthetic scaling sweeps up to about `500k` rows.
5. The study includes `freMTPL` sanity checks.
6. The study captures both call-stack and memory/allocation views.
7. The output is sufficient to choose the next optimization target with confidence.

## Recommended Next Step

Write the implementation plan for the profiling study itself:

- benchmark harness
- profiler wrappers
- result aggregation
- output summaries and artifacts
