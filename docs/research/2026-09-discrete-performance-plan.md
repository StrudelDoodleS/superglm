# Discrete execution performance plan

Status: implementation in progress. Baseline: `0a15736e88a317088bfd01e56933d45c58e4ac9a`
(production source unchanged since `5f994c8f6ac0501606594e2f36bfc0cd24050ec1`).

The immediate priority is efficient discrete execution before additional roadmap
capabilities. Preserve the working grouped, signed, coupled distributional
solver and its bounded row processing. A compact representation alone does not
establish a performance improvement: complete fits must supply the evidence.

## Stage 1: establish the work and numerical contracts

The code audit found that ordinary discrete pair dispatch limits histogram
allocation to five million cells but ignores the current row count and the
cost of contracting the bin table. An existing bounded row-panel contraction
can evaluate the same stored support representation. The audit also found
that chunked likelihood, predictor-change and terminal prediction passes
construct execution plans that they discard.

Use independent synthetic fixtures to demonstrate unnecessary dispatch and
plan construction before changing either path. Separate work-count assertions
from numerical tests. Numerical oracles evaluate the stored design, including
signed weights, rectangular blocks, centering and coefficient maps. Agreement
with an exact feature basis is a different question when bins approximate
continuous covariates.

## Stage 2: implement bounded execution improvements

1. Choose histogram or stored-support row contraction using row count,
   support dimensions and contraction work, preserving the allocation ceiling
   and favorable histogram reuse.
2. Separate predictor-value evaluation from geometry-plan construction.
   Keep chunk temporaries bounded; do not retain all materialized row designs.
3. Inspect residual complete-fit profiles before adding further structural
   reuse. Weighted reductions must be recomputed for each changing curvature
   channel. Endpoint numerical certificates retain their existing provenance
   requirements.
4. Following the first profile, replace avoidable tensor Gram/prediction
   temporaries with equivalent bounded contractions, and certify strictly
   increasing row selections without sorting them again. Retain existing
   validation and provenance for general selections.

## Stage 3: demonstrate complete-fit behavior

Compare the starting source and candidate on reproducible public synthetic
and insurance fixtures. Record complete-fit wall time from the worker's own
clock, process peak RSS, fitted numerical outputs, convergence diagnostics and
actual backend dispatch. Compare discrete before/after separately from the
discrete/exact representation comparison. Include cases favorable to support
compression and cases where histogram setup dominates.

Use fresh serial subprocesses with numerical thread counts fixed to one.
Retain raw receipts and disclose compilation/cache policy, order, repetitions
and background activity. Headroom/Kompress tool timings are not fit timings.
Pause other numerical work during timing windows and distinguish noisy
measurements from supported improvements.

## Stage 4: verification and review

Run focused mathematical, dispatch, bounded-memory and complete-fit regression
tests, followed by the repository checks appropriate to the final changes.
Review public artifacts for independent reproducibility and source provenance.
Update this plan and the roadmap with measured outcomes and remaining limits.
No new capability is promoted merely because a microbenchmark improves.

## Evidence and decisions

- The baseline focused distributional assembly, grouped assembly, chunking and
  discrete fit suites passed before implementation.
- Keep the versioned automatic chunk-size policy unchanged initially; remove
  avoidable work before changing its memory/performance tradeoff.
- Do not rebuild exact feature bases to make a discrete benchmark faster.
- No release version change or publication is part of this task.
- The first three-repeat public Gaussian comparison (20,000 rows, four knots,
  256 bins) rejected the initial arithmetic-only dispatch estimate: median
  discrete fit time increased from 1.912 s to 2.087 s with unchanged iteration
  counts. Raw receipts remain under
  `.benchmark-artifacts/discrete-performance/timing-gaussian/`. Background CPU
  was audited; these are local measurements with Headroom active.
- Kernel diagnostics explain the rejection: gathering and weighting row
  panels costs much more than the initial arithmetic count represented.
  Dispatch tests must use independently demonstrated favorable cases, rather
  than treating fewer arithmetic operations as proof of faster execution.
- A separate complete-fit profile identifies factored tensor Gram products,
  tensor prediction and repeated chunk preparation as remaining work. Review
  these before claiming that the discrete execution gate has been resolved.
- The revised histogram estimate includes setup and gather overhead and keeps
  histograms near the estimated crossover. Kernel diagnostics support this
  revision, but its acceptance still depends on complete-fit measurements.
- Defer a shared chunk-index slicing context: profile evidence favors simpler
  tensor and row-selection improvements first. Do not persistently cache
  marginal outer products while the underlying support tables remain mutable.
