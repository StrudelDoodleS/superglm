# PR105 Multi-SCOP Discrete Wall-Time Design

## Goal

Refocus PR105 on one question only:

**Can we reduce wall time for `multi-SCOP` `discrete=True` fits without
materially changing the fitted model?**

The branch should no longer treat “fewer REML iterations” as the primary goal
in isolation. Iteration counts still matter, but only as diagnostics for the
wall-time problem.

## Why This Exists

The current PR105 branch has already established several useful facts:

- `single-SCOP` and `multi-SCOP` behave differently
- `multi-SCOP discrete` is the practical path that matters
- the current cleanup gate can activate without producing a meaningful wall-time
  win
- benchmark snapshots can show identical fitted predictions/lambdas while still
  failing to improve runtime

That means the next step is not more generic theory or more public API.

It is:

- understand why the current cleanup path is not reducing wall time
- change only the part that actually matters

## Scope

This design is only for PR105 on the current branch.

It covers:

- `multi-SCOP`
- `discrete=True`
- wall-time reduction
- benchmark-driven diagnosis
- small internal solver changes if the measurements justify them

It does **not** cover:

- canonical runtime basis work
- export / SQL / rating table work
- public API changes
- broad solver rewrites

## Non-Goals

This pass is not trying to:

- make `_dm` canonical
- change the public coefficient contract
- change `predict()` semantics
- redesign exact-path prediction
- add a new public tolerance knob

Those belong to a separate future track, not PR105.

## Current Evidence

The branch already has enough evidence to justify a narrower target.

### Existing Benchmark Snapshot

From the current `multi_scop_discrete_convergence.csv` style work:

- `freMTPL2` optimized and baseline runs can produce identical predictions and
  identical lambdas
- but still show no meaningful wall-time improvement

### Existing Convergence Evidence

From the current debug/scaling studies:

- some lambdas pin to the floor
- the cleanup gate can activate
- but this activation is not automatically buying less total work

That means the likely problem is one of:

- the wrong lambdas are being managed
- freeze timing is too weak
- the expensive work is outside the portion of the loop we are changing
- the bookkeeping cost offsets any saved iterations

## Approaches Considered

### 1. Benchmark-Driven Diagnosis First

Treat the current cleanup path as an incomplete hypothesis. Instrument and
measure where time is still going, then make the smallest change that improves
wall time on the real target case.

Pros:

- highest confidence
- least likely to make the branch messier
- keeps PR105 narrowly justified

Cons:

- one more diagnostic round before changing behavior

### 2. More Aggressive Plateau / Freezing Immediately

Push the current cleanup logic harder and hope the runtime follows.

Pros:

- faster to try

Cons:

- easier to destabilize the fit
- easier to confuse iteration count improvements with real wall-time gains

### 3. Stop Here

Accept that the current cleanup branch is not buying enough and leave PR105 as
evidence-only.

Pros:

- lowest engineering risk

Cons:

- no practical payoff

## Chosen Direction

Use **Approach 1**.

PR105 should become a small, benchmark-driven wall-time investigation with one
clear output:

- either a justified internal change that improves `multi-SCOP discrete`
  wall time
- or a clear explanation that the current cleanup path is not the right lever

## Design

### Primary Target Metric

The main metric is:

- wall time on `multi-SCOP discrete`

Everything else is secondary:

- `n_reml_iter`
- aggregate inner PIRLS work
- gate activation
- freeze counts / freeze timing

These are diagnostic signals, not success criteria by themselves.

### Correctness Guardrails

Any attempted wall-time improvement must preserve:

- predictions within a tight tolerance
- lambdas within a tight tolerance
- stable convergence

If runtime improves but fitted behavior moves materially, the change fails.

### Datasets

Use both:

- a synthetic `multi-SCOP discrete` case
- full `freMTPL2`

The synthetic case is useful for controlled debugging.
The `freMTPL2` case is the real arbiter.

### Measurement Direction

The next pass should explicitly measure:

- whether the cleanup gate was consulted
- whether it returned `True`
- whether any names actually froze
- how many iterations had frozen names
- whether aggregate inner PIRLS work changed
- whether wall time changed in repeated runs

The current branch already measures some of this. The next pass should complete
the chain from “gate fired” to “work actually reduced”.

### Likely Failure Modes To Test

The design should be able to distinguish these cases:

1. Gate activates, but nothing freezes
2. Names freeze, but only cheap work is removed
3. Names freeze too late to matter
4. Names freeze correctly, but bookkeeping overhead cancels the gain
5. The expensive part of the discrete path is not in the outer-loop logic we
   are touching

That diagnostic resolution is the point of this pass.

## Acceptance Criteria

PR105 succeeds if it can show all of the following:

1. `multi-SCOP discrete` wall time improves on repeated runs, especially on
   `freMTPL2`
2. predictions remain within a tight tolerance
3. lambdas remain within a tight tolerance
4. no public API changes are introduced
5. no canonical-runtime work is mixed into the branch

If those conditions cannot be met, the branch should say so clearly rather than
keeping a complexity increase with no measured payoff.

## Risks

### Risk: Chasing Iterations Instead Of Runtime

Mitigation:

- treat iterations only as diagnostics
- keep wall time as the main outcome

### Risk: Overfitting To Synthetic Results

Mitigation:

- require `freMTPL2` confirmation before calling the work successful

### Risk: Expanding Scope Back Into Canonical Runtime Design

Mitigation:

- explicitly forbid public contract work in this PR
- keep the branch focused on internal wall-time behavior only

## Recommended Next Step

Write the implementation plan for a narrow PR105 wall-time pass:

- repeated benchmark harness refinement
- freeze / activation instrumentation
- one targeted internal improvement only if the measurements justify it
