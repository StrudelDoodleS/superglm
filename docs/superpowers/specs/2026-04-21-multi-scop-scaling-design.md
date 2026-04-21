# Multi-SCOP Scaling Study Design

## Goal

Understand how `SCOP` behaves as the number of constrained features grows,
especially in the `discrete=True` path that matters most in practice.

The study should answer:

- how runtime scales with the number of constrained `SCOP` features
- how memory scales
- how convergence behavior changes
- whether many lambdas are genuinely active or whether only a few matter while
  the rest pin or drift weakly

## Scope

This is a study, not a solver change.

It covers:

- multi-`SCOP` scaling
- multi-`SCOP` convergence behavior
- multi-`SCOP` lambda activity

The primary matrix is:

- `discrete=True`
- constrained-feature count: `1, 2, 4, 8, 16`

The secondary matrix is:

- exact path
- spot checks at `1, 2, 4`
- optionally `8` if still tractable

Single-`SCOP` is included only as a control/baseline, not the main focus.

## Why This Exists

The previous studies showed:

- exact constrained fit paths are slower, as expected
- single-`SCOP` lambda refinement often barely matters
- multi-`SCOP` discrete is where meaningful lambda sensitivity shows up

That means the next important question is not “is `SCOP` slow?”

It is:

**How bad does multi-`SCOP` get, and why?**

## Core Questions

### Scaling

- Does runtime grow roughly linearly, quadratically, or worse with constrained-feature count?
- Does memory blow up before runtime does?

### Convergence

- Do REML iterations increase with constrained-feature count?
- Do PIRLS iterations increase?
- Does plateau convergence dominate strict convergence as models get larger?

### Lambda Activity

- How many lambdas remain materially active?
- How many pin to the floor?
- Does one term dominate while the rest become nearly irrelevant?

## Study Design

### Primary Matrix: Discrete

Run:

- `n_constrained = 1, 2, 4, 8, 16`
- `discrete=True`

for a controlled synthetic family of multi-`SCOP` models.

This is the core study.

### Secondary Matrix: Exact

Run:

- `n_constrained = 1, 2, 4`

and optionally:

- `8`

if the runtime is still manageable.

This is not the main scaling law. It is there to show whether exact falls off
much faster than discrete and whether the convergence pattern changes.

## Data

Use synthetic data for the scaling matrix.

The synthetic generator should:

- add multiple constrained features
- allow repeated-support and mostly-unique support variants
- keep the signal structure comparable across feature counts

Real-data checks can be added later if needed, but the first study should keep
the matrix controlled and interpretable.

## Metrics To Record

For each run:

- runtime
- peak memory
- `n_reml_iter`
- `n_pirls_iter`
- strict vs plateau convergence
- final lambdas
- number of lambdas pinned at floor
- number of lambdas above a simple activity threshold

The study should also derive:

- active-lambda fraction
- floor-pinned fraction

## Outputs

At minimum:

- summary CSV
- scaling plots:
  - runtime vs constrained-feature count
  - memory vs constrained-feature count
  - REML iterations vs constrained-feature count
- lambda-activity plots:
  - count pinned at floor
  - count materially active

Optional but useful:

- one or two representative lambda-trajectory plots for larger multi-`SCOP` runs

## What A Good Result Looks Like

A good result should let us say something specific like:

- runtime is mostly driven by constrained-feature count, not `n_reml_iter`
- or
- convergence degrades because more lambdas remain active
- or
- most lambdas pin to the floor quickly, so term-freezing should help

That is the level of answer we want.

## Risks

### Risk: Too many scenarios

Mitigation:

- full matrix only on `discrete=True`
- exact is spot-check only

### Risk: Synthetic generator hides real behavior

Mitigation:

- keep synthetic structure close to the current constrained benchmark family
- add real-data checks later only if the synthetic results are ambiguous

## Acceptance Criteria

1. The study centers on multi-`SCOP`, not single-`SCOP`.
2. The full `1/2/4/8/16` matrix runs on `discrete=True`.
3. Exact is spot-checked rather than fully swept.
4. Runtime, memory, convergence, and lambda-activity metrics are all recorded.
5. The outputs are sufficient to choose the next multi-`SCOP` engineering move.

## Recommended Next Step

Write the implementation plan for the study:

- synthetic multi-`SCOP` benchmark harness
- exact/discrete matrix
- lambda-activity summary metrics
- plots and summary outputs
