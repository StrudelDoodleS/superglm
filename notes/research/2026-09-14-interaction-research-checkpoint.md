# Interaction research checkpoint for review

This archive preserves the work leading to the next cheap-interaction pilot.
It is stacked on the separate tensor-support performance change. The research
PR adds no production source or public model API changes.

## Start here

- [Current research plan and roadmap mapping](2026-09-14-cheap-interaction-research-plan.md)
- [First compressibility pilot implementation plan](2026-09-14-interaction-compressibility-pilot-plan.md)
- [Broad real-data results and complete search costs](2026-09-14-broad-interaction-trials.md)
- [Saved interaction plots](2026-09-14-broad-interaction-plots.md)
- [Representation assumptions and cost derivations](2026-09-13-cheap-interaction-representations.md)
- [Gaussian error analysis](2026-09-13-adaptive-gaussian-error-bounds.md)
- [Lean tutorial and exact proof scope](lean-gaussian-certificate/README.md)

## What is delivered

The corpus manifests, adapters and bounded trial runners provide development
workloads, explicit feature/target/split contracts, and failed-attempt records.
The broad experiment selects useful interaction groups on ten of twelve real
sources, with a separate CASP decoy control. It fits no more than four pairs
per arm. Of 100 fits, 94 converge, four reach the REML limit and two fail a
data-Gram check. All attempted costs and refusals remain visible.

The selected few-pair fit ratios do not establish the cost of discovering
them. The complete broad procedure's workers total 764.94 seconds. Tests
previously inspected during this work are development evidence. New model
decisions need fresh confirmation for a confirmatory performance claim.

The Lean project checks finite-real quadratic and coupled-operator identities.
It does not prove the full error upper bound, floating-point implementation,
REML optimum, covariance accuracy or runtime. The plan names the additional
contracts. Compact direct fitting and local refinement remain planned work.

## Review scope

Review executable code and tests under `benchmarks/`, the audit and plot
scripts, and the mathematical/empirical claim boundaries. In particular,
check train-only selection, source/model identity, complete-attempt accounting,
pair-group attribution and fixed-target versus new-model distinctions.

Machine-generated measurement JSON and figure files are marked generated in
`.gitattributes` so GitHub can collapse their large diffs. They remain tracked;
the audit scripts and human-readable reports explain their contents. Raw
downloaded data and fitted pickle caches are ignored local artifacts, not
committed data. Their manifests and available source/artifact hashes remain
part of the reproducibility record.

PR preparation exposed a missing-total path in the broad runner: when no arm
was eligible for test evaluation, final worker-time totals were absent even
though proposal and fit receipts recorded the spent time. The runner now
initializes those totals from search costs before evaluation. Regression cases
cover proposal timeout, no converged fit and exhaustion of the global budget;
all three fail against the original runner. Every archived case reached
evaluation, so the recorded experiment totals are unchanged. Its original
runner and source hash remain preserved in commit `639f499e`; the updated
runner has a new source identity and cannot stand in for the frozen run.

The [PR review follow-up](2026-09-14-interaction-review-validation.md) records
the remaining audit, timeout, plotting and receipt corrections, their regression
checks, and a replay from a different source checkout. The archived scores,
costs, hashes and surface arrays are unchanged.

One archival JSON file had a second final newline removed during PR packaging:
`2026-09-13-tensor-support-handoff-initial-measurements.json`. Its parsed values
are unchanged. The original bytes remain in commit `639f499e`, with SHA-256
`3e5373d10d50df9fec3db276fd5f4f78fd982718ed906e51556b5ba24ccf7dfd`.

The documentation rebuild in PR #392 relocates the roadmap and previous
research files to `notes/`. Preserve this archive and its links when those
branches are integrated. The current checkpoint follows the user's requested
`docs/research` location; no relocation is performed here.
