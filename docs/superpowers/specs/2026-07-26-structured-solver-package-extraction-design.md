# Structured Solver Package Extraction Design

**Status:** Approved for planning on 2026-07-26

## Purpose

Make the structured credibility solver independently navigable before merging
PR #165. Replace the 5,920-line `superglm.solvers.structured` implementation
module with focused internal modules while retaining
`superglm.solvers.structured` as the stable import facade.

This is a source-organization refactor. It does not authorize changes to
solver mathematics, array layouts, backend selection, REML behavior, public
APIs, or performance policy.

## Goals

- Give selection, layouts, sufficient statistics, compact operators,
  estimability geometry, factorization, assembly, and retained state distinct
  owners.
- Preserve every existing import from `superglm.solvers.structured`, including
  private imports currently used by the repository test suite.
- Remove `sum_to_zero.py`'s dependency on private symbols owned by the facade;
  it will import shared low-rank primitives from their real internal owner.
- Keep module dependencies acyclic apart from existing lazy imports used to
  avoid the structured/SZ factor type cycle.
- Preserve numerical results, termination decisions, allocations, and the
  measured structured-solver performance envelope.

## Non-goals

- No algorithm consolidation or redesign.
- No changes to scalar, block, or sum-to-zero equations or thresholds.
- No common exact/discrete REML driver and no REML phase extraction.
- No public API additions, removals, or deprecations.
- No changes to Tabmat integration, kernels, parallelism, dependencies, or
  supported Python versions.
- No changes to LSS files, behavior, or semantics.
- No opportunistic cleanup outside the moved structured-solver code.

## Package Structure

```text
superglm/solvers/
├── structured.py                 # compatibility facade and re-exports
├── sum_to_zero.py                # constrained SZ factorization
└── _structured/
    ├── __init__.py               # internal-package marker only
    ├── selection.py              # eligibility, costing, fallback decisions
    ├── layout.py                 # immutable coefficient partitions/plans
    ├── moments.py                # scalar/block/SZ sufficient statistics
    ├── operators.py              # compact operators and low-rank algebra
    ├── geometry.py               # estimability and null-space geometry
    ├── factors.py                # scalar/block Schur factor classes
    ├── assembly.py               # penalties, augmented factors, cached solves
    └── state.py                  # support and retained-fit state dataclasses
```

The split follows current responsibility boundaries rather than fixed line
ranges. A definition moves exactly once, with its body unchanged except for
imports required by its new owner.

## Ownership and Dependencies

### `selection.py`

Owns `StructuredGroupSelection`, `StructuredBackendDecision`, structured-cost
heuristics, group selection, singular-local-level detection, and
`resolve_structured_backend`.

It depends on group-matrix types and constraints, but not on any other
`_structured` module.

### `operators.py`

Owns scalar, block, SZ, centered, low-rank, and sum operators; compact operator
type aliases; scalar and block diagonal-low-rank representations; trace,
product, diagonal, and materialization primitives.

This is the common low-level algebra layer. `sum_to_zero.py` imports the
required private BDLR primitives directly from this module.

### `geometry.py`

Owns coefficient-estimability, null-space, Ritz, and sum-to-zero public
geometry helpers. It depends on `operators.py` and the shared rank policy.

The large SZ spectral-certification implementation remains intact. It is moved
without combining it with `sum_to_zero.py` or changing any fallback threshold.

### `factors.py`

Owns `ScalarSchurFactor`, `BlockSchurFactor`,
`ProfiledScalarSchurFactor`, and `ProfiledBlockSchurFactor`. It depends on
`operators.py` and the minimal null-space helper from `geometry.py`.

The constrained SZ factors remain in `sum_to_zero.py`.

### `layout.py`

Owns scalar/block layout dataclasses, layout builders and caches, structured
design `matvec`/`rmatvec`, and their input validation. It depends on
group-matrix execution plans but not on factorization.

### `moments.py`

Owns scalar, block, and raw-all-level SZ system dataclasses plus construction
of their unpenalized sufficient statistics. It depends on `layout.py`,
`operators.py`, Tabmat-backed group matrices, and the existing compact kernels.

### `assembly.py`

Owns penalized operator construction, augmented factor construction, cached
lambda-only solution dataclasses, and cached solve dispatch. It depends on
`moments.py`, `operators.py`, and `factors.py`. SZ factor imports stay local to
the functions that instantiate them, preserving the existing cycle break.

### `state.py`

Owns `StructuredLevelSupport`, `FactorSmoothLevelSupport`, and
`StructuredLinearSystemState`. Runtime dependencies use the ordinary
structured types; constrained SZ factor annotations remain guarded by
`TYPE_CHECKING`.

### Dependency direction

```text
selection

layout ─────────────────> moments ──> assembly
operators ──────────────> moments
operators ──> geometry ──> factors ──> assembly
moments/operators/factors ──────────> state
operators ──────────────────────────> sum_to_zero.py
all internal owners ────────────────> structured.py facade
```

The diagram denotes conceptual ownership, not a requirement that unrelated
roots import one another. `structured.py` may import every owner solely to
re-export the legacy namespace.

## Compatibility Contract

All imports currently used by production code and tests continue to work:

```python
from superglm.solvers.structured import BlockSchurFactor
from superglm.solvers.structured import resolve_structured_backend
from superglm.solvers.structured import _operator_bdlr
```

The facade uses explicit imports rather than wildcard imports and declares
`__all__` for the supported repository-wide namespace. The classes and
functions retain their original names, signatures, docstrings, and behavior.
Pickle compatibility is not promised for private module paths, but the
existing public fitted-model serialization suite must remain green.

Monkeypatching a re-export is not part of the compatibility contract. Tests
that intercept an implementation detail will patch its owning module:

- estimability fallback patches target
  `superglm.solvers._structured.geometry`;
- allocation guards for moment construction target
  `superglm.solvers._structured.moments`.

No forwarding setter or dynamic facade shim will be added.

## Mechanical-Move Rules

- Move complete definitions without editing numerical expressions.
- Preserve constant values, tolerances, exception types/messages, array
  dtypes, writeability flags, and copy behavior.
- Preserve local imports used to break runtime cycles.
- Do not merge scalar and block implementations.
- Do not merge ordinary block and SZ implementations.
- Do not rename externally imported private symbols during this pass.
- Do not change eager versus lazy materialization.
- Keep `structured.py` free of numerical implementation after extraction.

## Testing Strategy

### Architecture test

Add a focused test that initially fails against the monolith and then proves:

- `structured.py` re-exports representative symbols from every internal owner;
- facade imports return the identical objects owned by those modules;
- `sum_to_zero.py` no longer imports shared primitives from the facade;
- `structured.py` contains no solver implementation definitions.

### Focused behavioral gates

Run the existing suites covering:

- structured selection, layouts, allocations, and IRLS;
- scalar and block Schur factors;
- RandomEffect exact/discrete fits and inference;
- FactorSmooth FS/SZ systems, parity, REML, and inference;
- sum-to-zero factorization and spectral estimability;
- serialization and retained fit state.

### Repository gates

Run Ruff, format checking, `git diff --check`, the complete non-slow suite, and
the complete repository suite required by the PR. Existing release metadata
and packaging checks remain mandatory because this branch prepares 0.15.0.

### Numerical and performance gates

Reuse the existing characterization and million-row benchmarks with identical
seeds and controls. Acceptance requires:

- identical backend decisions, termination reasons, and iteration counts;
- existing numerical parity tolerances for objectives, lambdas, EDF,
  coefficients, predictions, and covariance quantities;
- no forbidden full-width allocation in compact paths;
- no more than 5% pooled benchmark regression, with profiling required before
  accepting a larger movement.

## Commit and Review Strategy

Keep extraction commits responsibility-scoped so failures can be bisected:

1. architecture test and package skeleton;
2. operators and geometry;
3. factors and SZ import cleanup;
4. selection, layout, and moments;
5. assembly, retained state, and final facade;
6. test patch-target updates and documentation.

Each commit must leave the focused structured suite passing. After the full
verification gate, push the branch, request a fresh Codex review, resolve
actionable threads, and repeat until the review and CI are clean.

## Completion Criteria

The extraction is complete when:

- `structured.py` is a compatibility facade with no numerical implementation;
- no internal module imports structured primitives from the facade;
- all old import paths used by the repository resolve to identical objects;
- focused, full, allocation, serialization, numerical, performance, release,
  and CI gates pass;
- Codex reports no major issue and all actionable review threads are resolved;
- PR #165 remains a clean draft at version 0.15.0;
- master and LSS remain untouched.
