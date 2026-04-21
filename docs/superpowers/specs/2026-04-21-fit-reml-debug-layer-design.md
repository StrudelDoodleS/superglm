# Fit REML Debug Layer Design

## Goal

Add a private internal debug layer that makes constrained `fit_reml()`
behavior observable enough to diagnose convergence problems, especially
for multi-`SCOP` discrete fits.

The first target is not speed. It is understanding:

- why some constrained REML paths keep iterating
- whether lambdas are genuinely improving the fit or merely creeping
- where the iteration budget is actually being spent

## Scope

This design introduces:

- a private internal debug/tracing layer
- debug levels with increasing verbosity
- machine-readable trace outputs
- human-readable summaries
- basic trajectory plots for diagnosis

The first implementation focus is:

- `fit_reml()` broadly
- deep instrumentation for `SCOP` REML paths first
- especially single-feature and multi-feature `SCOP` discrete runs

This design does **not** add:

- a public user-facing API
- a new modeling feature
- solver behavior changes
- optimization changes in the first pass

## Why This Exists

The current profiling study showed two things:

1. exact constrained fit paths still have performance headroom
2. multi-`SCOP` discrete convergence is a real issue

In particular, `fremtpl_multi_scop_discrete` hit the configured REML
iteration cap, which suggests a convergence-dynamics problem rather than
simply a raw compute problem.

We need better visibility into:

- lambda update trajectories
- objective changes
- PIRLS stability
- SCOP Newton behavior
- whether late REML iterations materially change the fitted model at all

## Chosen Direction

Build one reusable internal debug layer with levels:

- `DEBUG=0`: off
- `DEBUG=1`: basic summaries
- `DEBUG=2`: full per-iteration and per-step traces

This should be internal/dev-facing only.

The same debug layer should be usable everywhere eventually, but the first
instrumented path is `fit_reml()`, with the most detail on `SCOP`.

## Public vs Private

This is **not** a new public API.

It should live as a private internal debug facility:

- developer-accessible
- controlled by an internal debug setting
- suitable for repo-local benchmarks and diagnostics

The user explicitly wants something like a global/internal debug knob rather
than a public interface, so this should stay out of the supported user API.

## Debug Levels

### `DEBUG=0`

- no extra tracing
- no files
- current behavior

### `DEBUG=1`

Basic summaries only.

Expected output:

- run metadata
- aggregate timing
- convergence reason
- final lambda values
- final objective/deviance/EDF summary
- selected warnings such as hitting iteration caps or heavy clipping

### `DEBUG=2`

Full traces.

Expected output:

- per-REML iteration summary
- per-PIRLS iteration summary
- per-SCOP step summary
- machine-readable trace files
- basic trajectory plots

## Debug Layer Shape

Use a dedicated private tracing module rather than ad hoc `print()` calls.

Conceptually:

- a global/internal debug level source
- one trace recorder object per run
- path-specific event writers

Recommended internal operations:

- `record_run_start(...)`
- `record_reml_iter(...)`
- `record_pirls_iter(...)`
- `record_solver_step(...)`
- `record_run_end(...)`

These should be internal names only. Final naming can change during
implementation.

## What To Instrument First

The first instrumented path is `fit_reml()`.

Within that, priority is:

1. single-feature `SCOP` discrete
2. multi-feature `SCOP` discrete
3. single-feature `SCOP` exact as control
4. one `QP` control run for comparison

This gives us:

- a clean control
- the problematic multi-`SCOP` discrete case
- enough comparison to see whether the issue is truly `SCOP`-specific

## Trace Payload

### Run Start Metadata

At the beginning of a run, record:

- model family
- exact vs `discrete=True`
- constrained engine mix
- constrained feature names
- basis sizes
- row count
- number of constrained features
- maximum configured REML iterations

### REML Iteration Trace

At `DEBUG=2`, record per REML iteration:

- iteration index
- lambdas before and after
- delta-lambda magnitude
- objective before and after
- objective delta
- accepted/clipped status
- convergence check values

### PIRLS Iteration Trace

Record:

- iteration index
- deviance
- step norm
- conditioning/fallback information
- any decomposition fallback mode

### SCOP Step Trace

Record:

- feature/group identifier
- step norm
- objective improvement
- Fisher fallback used or not
- line-search halvings
- rejected non-finite trial counts

### Final Summary

Record:

- converged or not
- hit cap or not
- total runtime
- final lambdas
- final objective/deviance/EDF summary

## Output Formats

Use both machine-readable and human-readable outputs.

### Machine-readable

At minimum:

- CSV, JSONL, or similar trace file per run

This should be easy to inspect with pandas and easy to plot later.

### Human-readable

At minimum:

- a compact textual summary log

This should be enough to read quickly without opening the raw trace files.

### Plots

At minimum:

- lambda trajectories
- objective trajectories

Potentially also:

- EDF or prediction-drift trajectories

## First Diagnostic Questions

The first pass should explicitly answer:

1. Are lambdas still moving materially late in the run?
2. Is the objective still improving materially late in the run?
3. Are predictions or constrained term curves changing materially late in the run?
4. Are we burning iterations on noisy/weak updates rather than real fit improvement?

This matters because the suspected failure mode is:

- lambdas creeping
- but fitted performance barely changing

If that is true, then the problem is more about update sensitivity or
stopping logic than about raw solver failure.

## What A Good Diagnostic Result Looks Like

A good result is not a pile of logs.

A good result is a clear statement like:

- multi-`SCOP` discrete lambdas keep moving, but objective delta is nearly flat after iteration `k`
- or
- one specific constrained feature oscillates while the others stabilize
- or
- PIRLS settles quickly, but outer REML keeps taking tiny accepted steps

That is what will justify the next engineering change.

## Risks

### Risk: Too much trace noise

Mitigation:

- use levels
- keep `DEBUG=1` compact
- reserve detailed logs for `DEBUG=2`

### Risk: Instrumentation overhead distorts timing

Mitigation:

- use the debug layer for diagnosis, not benchmark headlines
- keep separate timing-only benchmark runs for performance claims

### Risk: SCOP-only assumptions leak into “general” debug layer

Mitigation:

- keep the trace recorder generic
- add path-specific event payloads where needed
- instrument `fit_reml()` first, not only `SCOP`

## Acceptance Criteria

1. The debug layer is private/internal only.
2. It supports at least `DEBUG=0/1/2`.
3. `fit_reml()` is instrumented first.
4. `SCOP` REML paths emit full traces at `DEBUG=2`.
5. The outputs include machine-readable traces and basic plots.
6. The resulting traces are sufficient to diagnose the multi-`SCOP` discrete convergence issue.

## Recommended Next Step

Write the implementation plan for the debug layer and convergence study first,
focused on:

- the internal debug plumbing
- `fit_reml()` instrumentation
- single- and multi-`SCOP` discrete study runs
- comparison against one or two control paths
