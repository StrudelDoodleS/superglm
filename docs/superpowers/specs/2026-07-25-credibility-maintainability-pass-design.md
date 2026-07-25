# Credibility Solver Maintainability Pass Design

**Status:** Approved for planning on 2026-07-25

## Purpose

Prepare the `RandomEffect` and `FactorSmooth` (`fs` and `sz`) implementation
for pull-request review with a bounded, behavior-preserving simplification
pass. Keep a more ambitious solver redesign as a separate architecture
exploration so it cannot destabilize the shipping branch.

## Goals

- Remove production code, imports, branches, and comments that are proven
  obsolete.
- Reduce duplicated validation and REML terminal-candidate logic where a
  small pure helper makes the control flow easier to verify.
- Improve internal names, docstrings, and invariant comments at the compact
  FactorSmooth/structured-solver boundary.
- Preserve every public API, fitted result, termination decision, fallback,
  numerical threshold, and serialization contract.
- Preserve the current structured and discrete performance envelope.
- Document a staged architecture that could make the wider GAM engine simpler
  and materially faster in future work.

## Non-goals

- No new model terms, bases, families, or user-facing options.
- No changes to penalty mathematics, REML objectives, convergence tolerances,
  line-search rules, tensor step caps, or backend selection thresholds.
- No rewrite or package-level split of the structured solver in the shipping
  pass.
- No new parallel runtime, GPU backend, native extension, or dependency.
- No changes to LSS files, behavior, or semantics.
- No opportunistic cleanup outside the credibility and REML paths changed on
  this branch.

## Shipping Pass

### 1. Characterize the frozen behavior

Before production edits, retain machine-readable results for representative
exact and discrete RandomEffect, FS, SZ, and tensor-interaction fits. Record:

- predictions and coefficient checksums;
- REML objective, lambdas, effective degrees of freedom, and deviance;
- REML/PIRLS iteration counts and termination reason;
- resolved direct backend and fallback reason;
- clean wall time and sampled memory for the million-row FS/SZ cases.

These records are comparison evidence, not new golden constants in the public
test suite.

### 2. Remove proven-dead implementation

Delete a private implementation only when repository-wide reference searches
and focused tests prove it has no caller. The known first candidate is
`_factor_smooth_support_sufficient_stats`, superseded by compact cell
aggregation. Do not remove fallback kernels that still serve exact or
mismatched-bin geometries.

### 3. Clarify compact FactorSmooth geometry

Centralize repeated validation of discrete support (`B_unique` and `bin_idx`)
behind one private helper on `FactorSmoothGroupMatrix`. Keep the three
operations conceptually separate:

- aggregate row values into `(level, support-bin)` cells;
- contract dense-small crosses through those cells;
- reuse cell weights for another discretized spline with the identical support
  map.

Names and docstrings must state whether an array uses raw all-level geometry or
public coefficient geometry. SZ continues to retain raw `K`-level moments and
apply its sum-to-zero adjoint only at the solver/public boundary.

### 4. Simplify structured assembly locally

Reduce nesting in `build_block_structured_system` only where a small helper can
express one decision without hiding data flow. The preferred extraction is the
selection of an optimized discrete cross versus the existing compact fallback.
Do not extract a helper that merely trades visible branches for a long,
stateful argument list.

Tabmat remains responsible for the heterogeneous ordinary partition and the
dense-small side. FactorSmooth kernels remain responsible for the dominant
compact grouped-spline moments.

### 5. Consolidate pure REML candidate decisions

Exact and discrete REML share two mathematical decisions:

- project a score at fixed or active lambda bounds;
- determine whether an already evaluated candidate satisfies the compound
  score/objective criterion.

These may move into a small internal module if characterization tests first
pin their boundary behavior. The helper returns diagnostics or a decision; it
must not mutate lambdas, optimizer state, histories, or fit results. The
existing optimizer loops retain ownership of timing, line search, tensor
specialization, and terminal-result construction.

Termination reasons remain the existing strings:

- `score_objective_tolerance`
- `active_set_stationary`
- `fixed_lambdas`
- `line_search_failed`
- `max_reml_iter`

### 6. Improve comments and naming

Comments should explain invariants or numerical reasons, not narrate syntax.
Local names should distinguish:

- raw versus public SZ coordinates;
- row weights versus cell weights;
- evaluated candidate lambdas versus proposed lambdas;
- accepted state versus best observed state.

No internal rename is worthwhile if it causes broad churn without reducing
ambiguity.

## Error and Fallback Contracts

The pass preserves current errors for invalid dimensions, missing discrete
support, unsupported geometry, and forced structured-solver ineligibility.
Automatic backend fallback remains observable through the existing fallback
reason. A rejected REML trial continues to retain the last evaluated
candidate; cleanup must never reinstall an unevaluated fallback step.

## Verification

Refactoring uses the existing characterization suite as its safety net. Any
new pure helper receives focused tests before callers move to it.

Required gates:

1. Focused RandomEffect, FactorSmooth, structured solver, sum-to-zero, and
   REML Newton tests.
2. Complete repository test suite.
3. Ruff and `git diff --check`.
4. Before/after numerical comparison for exact/discrete RE, FS, SZ, and the
   tensor convergence regression.
5. Repeated million-row FS/SZ comparison with identical seed and controls.

Acceptance requires identical termination reasons and iteration counts.
Objectives, lambdas, EDF, deviance, coefficients, and predictions must remain
within the existing numerical parity tolerances. Pooled benchmark means must
not regress by more than 5%; a larger movement triggers profiling rather than
acceptance.

## Future Architecture Exploration

This section is documentation only. It does not authorize production changes
in the shipping pass.

### Current pressure points

- `solvers/structured.py` combines backend selection, layouts, sufficient
  statistics, operator construction, factorization, covariance traces, and
  retained inference state.
- Exact and discrete REML optimizers duplicate outer-loop decisions while
  differing in cache ownership, tensor handling, and evaluation cost.
- One dominant structured block is fast, but two simultaneously large
  credibility blocks make the Schur "small" side large.
- Row-scale work is already compact for discrete FactorSmooths, but repeated
  family/link and objective passes still limit million-row scaling.

### Candidate module boundaries

Keep `superglm.solvers.structured` as a compatibility facade while moving
coherent responsibilities behind it:

- `structured_selection`: eligibility and cost-based backend decisions;
- `structured_layout`: immutable coefficient partitions and execution plans;
- `structured_moments`: row/cell sufficient-statistic assembly;
- `structured_operators`: scalar, block, and sum-to-zero operators;
- `structured_factors`: Schur and constrained factorizations/solves;
- `structured_inference`: retained state, trace, and covariance operations.

For REML, explore a common outer driver parameterized by explicit backend
operations: evaluate candidate, build curvature, propose step, evaluate trial,
and finalize. Tensor-pair logic remains an optional strategy rather than
generic-loop conditionals.

### Ultra-fast GAM avenues

Evaluate these independently with correctness and crossover gates:

1. Persist immutable discretized design/support plans across refits and model
   comparisons.
2. Fuse family/link working-response, deviance, and compact moment reductions
   when doing so reduces full-row passes.
3. Reuse factorizations and sufficient statistics across lambda-only trial
   evaluations wherever the mathematics permits.
4. Use matrix-free or structured trace calculations for large penalty blocks
   rather than materialized covariance work.
5. Generalize from one dominant block to a graph of independent scalar/block
   effects with a narrow shared border.
6. Parallelize only coarse, independent block/cell contractions after proving
   that scheduling and memory-bandwidth overhead beat the serial/BLAS path.
7. Compare Newton REML with stable Fellner-Schall/EFS-style updates by model
   geometry; do not assume one optimizer is best for every term mix.

The likely ceiling is set by unavoidable row passes, not coefficient count,
once the design is discretized and structured. The practical target is
therefore fewer passes and more reusable compact state, not merely more
threads.

### Staged migration

1. Add characterization and allocation/performance gates.
2. Extract modules without changing symbols or algorithms.
3. Introduce a common REML-driver protocol behind exact/discrete adapters.
4. Prototype multi-dominant block solving against dense references.
5. Prototype fused or parallel reductions only for measured bottlenecks.

Each stage must be independently revertible and must beat its declared
correctness, memory, and wall-time gates before the next stage begins.

## Completion Criteria

The shipping pass is complete when the implementation is smaller or locally
simpler, all behavior and performance gates pass, the architecture exploration
remains documentation-only, and the branch is ready for final review without
new feature scope.
