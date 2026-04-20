# Constrained Fit Path Optimization Design

## Goal

Speed up the exact constrained fit paths without changing the public API or
changing fitted results beyond tiny numerical tolerance.

The optimization target is the fit path only:

- first: single constrained `SCOP` feature
- second: single constrained `QP` feature
- third: multiple constrained features in one model

This work should reduce obvious repeated work, avoid wasteful Python-level
loops, and preserve the current exact path as the behavioral reference.

## Scope

This optimization project covers:

- exact fit-path acceleration only
- no public API changes
- no tensor or interaction work
- no mixed-shape API work
- no solver-semantic changes to what is being optimized

The rollout is staged:

1. single-feature `SCOP`
2. single-feature `QP`
3. multiple constrained features in one model

Validation should use:

- synthetic stress cases
- real-data checks such as `freMTPL`

## Current Problem

The exact constrained fit paths are paying too much row-level work.

For `SCOP` in particular, the current exact path performs:

- row-level `eta` assembly
- row-level residual work
- row-level weighted design products
- repeated scatter/gather through full `n`

The discrete path is much faster because it compresses that work to bin-level
aggregates. The exact path keeps more fidelity, but it is clearly leaving
performance on the table.

The same general risk exists for exact `QP` constrained fitting:

- repeated row-level weighted products
- repeated coefficient-space constraint assembly work
- multi-group orchestration paying more Python overhead than needed

## Success Criteria

This project is not about hitting a single headline speedup number.

The actual gate is:

- obvious repeated work removed
- avoidable Python-level overhead reduced
- exact path still stable and consistent
- no public API change
- no visible predictive/fitted-shape degradation beyond tiny tolerance

The engineering standard is:

- current exact results remain the reference
- optimization is internal and automatic
- fallback to the current exact path remains available when the fast path is not appropriate

## Chosen Strategy

Use exact support compression where possible, and current exact row-level
fitting otherwise.

This means:

- no new approximation mode
- no new user-facing flag
- no forced discretization

Instead, the fit path should decide internally:

- when exact support-level aggregation is legal and beneficial
- when it is not

If it is not, we keep the current exact row-level path.

## Why Support Compression

For one constrained 1-D feature, much of the exact path depends only on:

- the unique support values of the constrained covariate
- exact row weights aggregated onto that support
- exact weighted residual aggregates on that support

That means we can often avoid paying full `n`-row algebra while still
representing the same objective exactly.

This is especially attractive for insurance-style features that often have
many repeated values.

## Phase 1: Single-Feature SCOP

### Objective

Speed up the exact fit path for one constrained `SCOP` feature while keeping
the current exact path as fallback.

### Targeted Waste

1. Repeated row-level weighted algebra

- repeated `eta_scop` assembly over all rows
- repeated full-row `B^T W B`
- repeated row-level residual weighting

2. Repeated transform bookkeeping

- recomputing or re-scattering structures that are stable after build

3. Python coordination overhead

- generic multi-group loops used for the one-constrained-feature case

### Planned Direction

Build an internal exact support-level representation for a single constrained
`SCOP` feature.

It should contain:

- unique support values
- row-to-support mapping
- support-level raw basis / SCOP design
- exact aggregated weights and weighted responses

Then the fit path should use support-level weighted products where that is
mathematically exact, while still reconstructing the full-row contribution
only when actually needed.

### Exactness Contract

The optimized path must preserve the same objective, up to tiny floating-point
differences.

That means:

- same knots
- same basis
- same weights
- same penalty
- same Newton / PIRLS semantics
- just reorganized algebra

### Fallback

If support compression is not useful or not safe, fall back to the current
exact row-level `SCOP` path automatically.

No new public flag.

## Phase 2: Single-Feature QP

### Objective

Apply the same philosophy to one constrained `QP` feature.

### Targeted Waste

- repeated row-level weighted products
- repeated coefficient-space constraint composition work where stable pieces
  can be cached
- general coordination overhead for a single constrained group

### Planned Direction

Use exact support-level weighted aggregation where possible, with the same
fallback rule:

- fast exact path when legal and beneficial
- current exact row-level path otherwise

The details will differ from `SCOP`, but the design principle is the same.

## Phase 3: Multiple Constrained Features

### Objective

Scale the same ideas to models with multiple constrained features.

### Main Concern

The problem is no longer just per-feature cost. The hard part becomes:

- cross-block weighted products
- repeated residual updates
- coordination overhead across constrained groups

### Planned Direction

Extend the single-feature machinery carefully:

- reuse support-level representations per constrained feature
- build exact cross-feature products from those support-level structures where possible
- keep a clean fallback to current row-level exact algebra where needed

This phase should only begin after single-feature `SCOP` and `QP` are both
validated.

## What Not To Do

Do not:

- add a public optimization flag
- silently approximate beyond current exact semantics
- invent a broad generic abstraction before the first phase proves out
- mix tensor/interactions into this work
- optimize by changing model behavior

## Data And Evaluation

Use two evaluation layers throughout:

### Synthetic

Synthetic datasets make it easier to isolate:

- repeated-support wins
- unique-support fallback behavior
- multi-feature scaling trends

### Real Data

Use real-data checks such as `freMTPL` to confirm:

- the optimization matters on actual workloads
- fitted shape and predictive performance remain stable

## Gates

### Gate 1: Single-Feature SCOP

Must show:

- exactness preserved to tolerance
- no obvious instability
- repeated work measurably reduced
- useful speedup on at least one support-compressible case
- no regression on existing monotone tests

### Gate 2: Single-Feature QP

Must show:

- same exactness discipline
- no regression on existing QP constrained tests
- useful speedup on at least one support-compressible case

### Gate 3: Multiple Constrained Features

Must show:

- exactness preserved
- no regression on single-feature paths
- improved scaling versus the current exact path

## Risks

### Risk: Hidden approximation

If support aggregation is done incorrectly, the optimized path may no longer
match the exact objective.

Mitigation:

- treat the current exact path as reference
- compare coefficients, predictions, and shape diagnostics directly
- fall back when legality is unclear

### Risk: Over-generalizing too early

If we try to build one giant constrained-fit abstraction immediately, we may
increase complexity and reduce clarity.

Mitigation:

- single-feature `SCOP` first
- single-feature `QP` second
- multi-feature third

### Risk: Speedups only on “nice” repeated-support data

Some datasets may have mostly unique values.

Mitigation:

- fallback to the current exact path
- also reduce Python overhead and repeated work in the fallback path where possible

### Risk: Multi-feature cross-block complexity dominates

Even if per-feature work is faster, multi-feature models may still spend most
of their time in cross-block algebra.

Mitigation:

- do not promise multi-feature speedups until after the single-feature gates pass
- design the multi-feature phase around cross-block products explicitly

## Acceptance Criteria

1. The optimization remains internal with no public API change.
2. The current exact fit remains the behavioral reference.
3. A single constrained `SCOP` feature gets a faster exact path when possible.
4. A single constrained `QP` feature gets a faster exact path when possible.
5. Multiple constrained features in one model become the third rollout phase, not the first.
6. Synthetic and `freMTPL`-style checks are part of validation.
7. Existing constrained tests remain green.

## Recommended Next Step

Write the implementation plan for Phase 1 only:

- single-feature exact `SCOP` fit-path acceleration
- support-compressed exact path plus clean fallback
- explicit correctness/performance gates
