# Unified Matrix Execution Plan

> **For agentic workers:** Use test-driven development and subagent review for every production
> routing change. This plan deliberately keeps compressed BAM-style algebra rather than
> materializing discretized rows into Tabmat.

**Goal:** Replace separate weighted-moment assembly paths with one immutable,
backend-neutral execution plan. Tabmat owns eligible ordinary observation-level blocks; existing
support/bin/grid kernels own compressed blocks; callers see one ordered solver-coordinate result.

**Performance contract:** Adopt a route only when fixed-thread A/B measurements show a repeatable
fit-time or memory win. Preserve convergence decisions and the frozen numerical envelope. Every
raw-centered result passes the one existing centering certificate and has a stable centered
fallback.

## Architecture

- `MatrixExecutionPlan` stores immutable group spans and an optional lazy
  `OrdinaryTabmatPartition` with local-to-global column maps.
- `MomentWorkspace` is call-local and owns writable contiguous vectors, discrete histograms,
  tensor grids, and weight-dependent aggregates. No weighted data is cached on the plan or fitted
  model.
- `WeightedMoments` returns Gram, optional `X'W`, and any requested transpose products in global
  solver order.
- Ordinary × ordinary products use one Tabmat sandwich and one transpose call per requested vector
  when the cost policy approves.
- Compressed diagonals and compressed crosses continue to use `gram_rmatvec`, tensor grids,
  `_agg_by_bin`, and `_cross_gram`.
- Packed all-compressed centering stays first. Hybrid raw moments are second. Stable chunk-centered
  assembly is the numerical fallback.

## Task 1: Add the plan without production routing

**Files:**

- Create `src/superglm/_group_matrix/_group_matrix_execution.py`.
- Modify `src/superglm/group_matrix.py`.
- Create `tests/test_matrix_execution_plan.py`.

- [ ] Write dense-reference parity tests for Dense + Sparse + high-cardinality Categorical +
  DiscretizedSSP + DiscretizedTensor designs.
- [ ] Cover positive and signed weights, `X'W`, multiple RHS vectors, global column ordering,
  base-category rows, empty bins, and zero weights.
- [ ] Require exactly one Tabmat sandwich and the expected transpose calls.
- [ ] Monkeypatch compressed `toarray()` methods to fail.
- [ ] Cover strided and read-only vector buffers.
- [ ] Implement immutable group spans, lazy ordinary partition construction, scatter/gather, and
  call-local workspace.
- [ ] Run the new tests plus theory/rank tests before any caller migration.

## Task 2: Make the plan the one weighted-moment assembler

**Files:**

- Modify `src/superglm/_group_matrix/_group_matrix_algebra.py`.
- Modify `src/superglm/group_matrix.py`.
- Modify algebra and REML tests.

- [ ] Make `_block_xtwx`, `_block_xtwx_rhs`, and `_block_xtwx_signed` compatibility wrappers over
  `MatrixExecutionPlan` rather than three separate assembly loops.
- [ ] Reuse `DesignMatrix.execution_plan` in hot callers; allow an ephemeral plan only for legacy
  list-based internal calls.
- [ ] Preserve profiling counters and tensor/discrete cache sharing.
- [ ] Verify all algebra, tensor, SCOP, EFS, and REML derivative tests.

## Task 3: Route certified mixed discrete centering

**Files:**

- Modify `src/superglm/_group_matrix/_group_matrix_centered.py`.
- Modify `src/superglm/solvers/centered_system.py` and `irls_direct.py`.
- Add mixed-discrete correctness and benchmark tests.

- [ ] Add a cheap per-block location/scale preflight that rejects unsafe raw subtraction before a
  full Gram is built.
- [ ] Require at least one compressed block and a conservative measured size gate. Initial evidence
  suggests `n * p >= 100_000`; remeasure before freezing it.
- [ ] Assemble accepted hybrid moments through the execution plan and run the authoritative raw
  centering certificate.
- [ ] Lock a rejected route to stable chunks for the rest of the fit; certify every accepted
  changed-weight iteration.
- [ ] Verify no row materialization on the accepted route and exact fallback behavior for large
  offsets, overflow, rank deficiency, zero weights, and aliased columns.

## Task 4: Extend only measured Tabmat crossovers

- [ ] Benchmark categorical-only routing separately; current valid evidence supports a conservative
  experiment around 300+ coefficients, not a blanket categorical rule.
- [ ] Benchmark low-cardinality categorical routing by both row count and width.
- [ ] Benchmark plan `matvec` and `rmatvec` for large mixed designs; keep existing group methods for
  small systems and BCD group updates.
- [ ] Use `sandwich(cols=...)` only for genuinely selected systems. Do not add `out=` workspace
  complexity or column-restricted transpose calls without new evidence.

## Task 5: Verification and handoff

- [ ] Run full tests, Ruff, mypy, and the fixed-thread frozen benchmark suite.
- [ ] Run ordinary CPython 3.10–3.14 compatibility; treat 3.14t as GIL-enabled experimental
  compatibility.
- [ ] Record fit time, Python/native peak memory, numerical deltas, iteration/convergence decisions,
  and actual Tabmat kernel counts.
- [ ] Update `docs/tabmat-integration-notes.md` with only valid, reproducible measurements.
