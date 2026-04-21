# Multi-SCOP Discrete Convergence Cleanup Design

## Goal

Reduce unnecessary REML outer iterations for `multi-SCOP` models on the
`discrete=True` path without materially changing fitted curves, predictions, or
final lambda decisions.

This is a convergence and performance cleanup, not a solver rewrite.

## Why This Exists

The recent profiling and sensitivity work established:

- `single-SCOP` can often bypass lambda refinement entirely
- `multi-SCOP discrete` cannot be bypassed wholesale because lambda can still
  matter for fitted constrained curves
- full-data runtime on `freMTPL2` is acceptable, but some REML outer iterations
  are still spent on lambdas that are drifting weakly or pinned at the floor
- practical fit stability can arrive before strict lambda convergence

That means the remaining useful work is not “make SCOP mathematically
different.” It is:

- stop outer iteration when the fit has genuinely plateaued
- stop repeatedly updating inactive lambdas

## Scope

This design applies only to:

- `fit_reml()`
- `SCOP`
- `discrete=True`
- models with more than one constrained `SCOP` term

It does **not** change:

- single-`SCOP` behavior
- exact multi-`SCOP`
- QP-constrained terms
- public user-facing tolerance parameters

## Non-Goals

This pass does not:

- add a new public convergence knob
- introduce a new global Newton solver
- change SCOP parameterization or penalty algebra
- alter single-`SCOP` bypass policy work
- change the exact-path optimization strategy

## Problem Statement

The current `multi-SCOP discrete` path can converge in a practical sense while
still failing a strict lambda-movement test:

- the REML objective becomes nearly flat
- PIRLS iterations remain low
- one or more constrained lambdas pin near the floor
- another constrained lambda may continue moving slowly

In this regime, strict log-lambda convergence can overstate the remaining work.
The algorithm is spending iterations proving a result that is already
practically stable.

## Approaches Considered

### 1. Plateau-Only Stop

Add a stronger private plateau rule for `multi-SCOP discrete` and stop once the
objective is flat and lambda updates are small.

Pros:

- smallest code change
- low risk

Cons:

- still updates obviously inactive lambdas until the plateau logic fires
- likely leaves some iteration savings on the table

### 2. Plateau Stop Plus Active-Lambda Freezing

Temporarily freeze lambdas that are floor-pinned or effectively stable, and
apply plateau convergence to the remaining active set.

Pros:

- matches the observed behavior from the benchmarks
- targets the actual source of wasted outer iterations
- should improve both stability and runtime without changing the fitted curve

Cons:

- adds some state to the outer loop
- requires careful verification to avoid freezing too aggressively

### 3. Relaxed Internal Tolerance Only

Special-case `reml_tol` internally for `multi-SCOP discrete`.

Pros:

- easy to implement

Cons:

- too blunt
- weak mathematical justification
- more likely to hide real convergence issues

## Chosen Direction

Use **Approach 2**:

- private plateau convergence specialized for `multi-SCOP discrete`
- private active-lambda freezing for floor-pinned or effectively settled
  lambdas
- no new public API

This is the most sensible tradeoff between correctness, stability, and runtime.

## High-Level Design

### Activation Condition

The new logic activates only when all of the following hold:

- `model._discrete` is true
- at least two constrained `SCOP` terms are present
- the `SCOP` REML/EFS path is being used

All other cases keep the current behavior.

### Active vs Frozen Lambda Sets

Track two internal sets during the outer loop:

- `active_names`
- `frozen_names`

Only `active_names` participate in lambda updates.

### Freeze Criteria

A lambda becomes eligible for freezing when either:

- it is at or very near the floor and has remained there stably, or
- its accepted log-scale change has remained below a small internal threshold
  for several accepted outer iterations

The freeze rule should be conservative. The aim is to freeze only lambdas that
are already behaving as inactive directions.

### Plateau Convergence

For `multi-SCOP discrete`, declare plateau convergence when:

- the REML objective has flattened
- active lambda movement is small
- frozen lambdas have stayed stable for enough iterations to be trusted

Strict convergence using `reml_tol` remains valid and unchanged. Plateau
convergence is an additional internal early-stop rule for this narrow path.

### Finalization

Frozen lambdas remain part of the final reported lambda dictionary. This is a
convergence optimization, not a post-processing rewrite.

The fitted coefficients, predictions, and stored `model._reml_lambdas` should
retain the same structure as before.

## Implementation Sketch

The primary implementation site is the SCOP EFS outer loop in
`src/superglm/reml/scop_efs.py`.

Expected changes:

- narrow the special behavior to `multi-SCOP discrete`
- refine the existing `active_names` / `frozen_names` loop state
- make freezing criteria explicit and conservative
- base plateau checks on active lambdas rather than all lambdas equally
- ensure floor-pinned terms do not keep the outer loop alive by themselves

This should remain an internal implementation detail.

## Verification Strategy

### Regression Coverage

Add tests that cover:

- `multi-SCOP discrete` still converges
- final fitted curves remain effectively unchanged relative to the current path
- frozen lambdas remain present in the final lambda map
- single-`SCOP` and exact-path behavior remain unchanged

### Benchmark Checks

Run targeted benchmarks on:

- synthetic `multi-SCOP discrete`
- `freMTPL2` `multi-SCOP discrete`

Record at least:

- runtime
- `n_reml_iter`
- `n_pirls_iter`
- final lambdas
- curve similarity against the baseline implementation

### Acceptance Threshold

The change is successful if it:

- reduces or at least does not increase `n_reml_iter` for the target path
- keeps the fitted constrained curves effectively unchanged
- does not regress full-data runtime
- does not alter non-target solver paths

## Risks

### Risk: Freeze Too Early

Freezing an actually active lambda could change the fitted curve.

Mitigation:

- require repeated stability before freezing
- limit the change to `multi-SCOP discrete`
- verify against baseline curve similarity

### Risk: Plateau Rule Stops Too Early

Plateau detection could hide remaining meaningful lambda movement.

Mitigation:

- require both objective flatness and active-set stability
- keep strict convergence logic intact
- verify on both synthetic and `freMTPL2`

### Risk: Added State Makes the Loop Harder To Reason About

Mitigation:

- keep the state minimal
- document the active/frozen transition logic clearly in code comments
- use existing debug tracing where useful

## Acceptance Criteria

1. The new behavior applies only to `multi-SCOP discrete`.
2. No new public tolerance or solver parameter is added.
3. The target path stops based on practical plateau behavior rather than only
   strict lambda creep.
4. Inactive or floor-pinned lambdas stop driving extra outer iterations.
5. Fitted curves and predictions remain effectively unchanged.
6. Full-data `freMTPL2` runtime is at least as good as the current branch.

## Recommended Next Step

Write the implementation plan for:

- the exact freeze criteria
- the exact plateau rule
- regression tests
- benchmark verification on synthetic and `freMTPL2`
