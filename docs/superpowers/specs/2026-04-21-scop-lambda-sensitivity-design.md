# SCOP Lambda Sensitivity Experiment Design

## Goal

Determine whether `SCOP` lambdas materially change fitted constrained curves
and predictions in the cases we care about, especially:

- single-`SCOP`
- multi-`SCOP`
- exact
- `discrete=True`
- `freMTPL`

The key question is:

Do we really need to keep estimating these lambdas aggressively, or are we
iterating on directions that barely affect the fitted model?

## Scope

This is a diagnostic experiment, not a permanent feature.

It includes:

- a temporary internal `SCOP` lambda bypass mode for experiments
- curve and prediction comparisons
- similarity metrics
- exact vs discrete comparisons
- single- and multi-`SCOP` scenarios
- a broad log-scale lambda sensitivity sweep

It does **not** add:

- a public API
- a user-facing solver option
- permanent modeling behavior changes

## Core Question

The convergence traces suggest a possibility:

- lambdas may keep moving
- but the constrained fit may already be effectively stable

If true, that would mean:

- strict lambda convergence may not be the right stopping target
- some lambda refinement may be weakly identified
- some terms may be pinned or nearly pinned by the shape constraint already

## Chosen Direction

Compare three classes of fits:

1. integrated `SCOP` REML result
2. passthrough-style lambda estimate
3. fixed-lambda sensitivity sweep

This should be done for both exact and discrete paths, and for both single-
and multi-`SCOP` scenarios.

## Why This Experiment

A convergence trace alone is not enough.

It can show:

- lambda creep
- objective plateau
- fallback counts

But it does not prove whether the fitted model actually changes in a way that
matters.

To prove that, we need direct model comparisons.

## Comparison Modes

### 1. Integrated SCOP REML

Use the current `SCOP` REML/EFS path as the baseline.

This is the reference fit.

### 2. Passthrough-Style Lambdas

Temporarily use a QP-style passthrough idea for `SCOP`:

- estimate lambdas from an unconstrained REML fit
- refit the constrained `SCOP` model at those lambdas

This answers:

- if we skip integrated `SCOP` lambda refinement, do the constrained curves
  materially change?

### 3. Fixed-Lambda Sensitivity Grid

Take a baseline lambda and run a broad log-scale sweep such as:

- `lambda * {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}`

This answers:

- how sensitive are the constrained curves and predictions to lambda at all?

## What To Compare

### Primary Comparison

Constrained term curves on the linear predictor scale.

This is the most important comparison because the actual question is whether
lambda materially changes the constrained smooth itself.

For each constrained feature:

- curve overlay
- pointwise difference curve
- `R²` similarity
- max absolute difference
- RMSE / weighted RMSE on the evaluation grid

### Secondary Comparison

Full model predictions.

This is useful because a term can move slightly while the full model still
barely changes, or vice versa.

For predictions:

- prediction-vs-prediction scatter
- prediction `R²`
- max absolute difference
- RMSE

### Optional Holdout Comparison

If a train/test split is used, also compare:

- holdout metric drift

for example:

- deviance
- Gini or similar if relevant to the benchmark setup

## Scenario Matrix

The first experiment matrix should include:

1. single-`SCOP` exact
2. single-`SCOP` discrete
3. multi-`SCOP` exact
4. multi-`SCOP` discrete

All on the same underlying benchmark family so the results are comparable.

## Data

Use `freMTPL` for the main experiment.

This is the right place because:

- it already exposed the practical concern
- it is a real workload
- we care about the real-data constrained curves, not only synthetic behavior

Synthetic data can remain useful later, but this experiment should be anchored
on the actual problematic use case.

## Internal Experiment Mechanism

This should be implemented as private/internal experiment code only.

Possible internal mechanisms:

- a private benchmark-only switch
- a benchmark-local helper that builds the unconstrained lambda estimate and
  then refits the constrained `SCOP` model
- a benchmark-local helper that refits fixed-lambda constrained models over
  the sensitivity grid

The exact mechanism can vary, but the key rule is:

- no public API change

## Expected Outputs

The experiment should produce:

- one summary table for all scenarios
- one summary table for all lambda-grid runs
- per-feature curve plots
- difference plots
- prediction-comparison plots
- similarity metrics in CSV or parquet

The outputs should make it visually obvious whether lambda matters.

## What A Strong Result Looks Like

### If Lambdas Barely Matter

We would expect to see:

- high curve `R²`
- high prediction `R²`
- tiny pointwise differences
- small or negligible holdout drift
- even across a fairly broad lambda range

That would support:

- earlier stopping
- freezing lambdas
- de-emphasizing strict lambda convergence

### If Lambdas Matter

We would expect to see:

- visible curve movement
- meaningful prediction drift
- EDF changes that track lambda changes
- maybe holdout metric changes as lambda changes

That would support:

- keeping integrated lambda estimation
- but still maybe improving the convergence logic or inner-step robustness

## Risks

### Risk: Comparison is confounded by intercept / other terms

Mitigation:

- make constrained term curves the primary metric
- predictions are secondary but still valuable

### Risk: One baseline lambda is arbitrary

Mitigation:

- use a broad log-scale grid around the baseline
- include both integrated and passthrough baselines

### Risk: Temporary bypass becomes a permanent accidental feature

Mitigation:

- keep it benchmark-local or private/internal only
- no public API surface

## Acceptance Criteria

1. The experiment compares integrated `SCOP` REML, passthrough-style lambdas, and a fixed-lambda sensitivity grid.
2. It includes both exact and discrete paths.
3. It includes both single- and multi-`SCOP` scenarios.
4. It compares constrained term curves on the link scale.
5. It also compares full model predictions.
6. It produces enough visual and numerical output to answer whether lambda materially matters.

## Recommended Next Step

Write the implementation plan for the experiment itself:

- internal benchmark-only lambda bypass
- fixed-lambda sweep harness
- curve/prediction comparison metrics
- plots and summary outputs
