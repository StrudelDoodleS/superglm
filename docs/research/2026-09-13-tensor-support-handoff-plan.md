# Tensor support handoff implementation plan

> For agentic workers: execute this bounded plan inline with the executing-plans and test-driven-development skills. The parent owns review and complete-fit measurements. Do not spawn agents or commit changes.

**Goal:** Build the unchanged discrete tensor family's selected support once per fit and reuse it through finalization without changing its numerical contract.

**Architecture:** Return the final optimizer-owned penalty family through `REMLResult.reml_penalties`. Add an explicit `_reuse_fixed_from` builder argument for complete unprojected discrete tensor families. Bind target provenance at component construction and capture support evidence only during the existing first support construction; the accepted target shares immutable arrays and support but owns its mutable evaluation state.

**Tech stack:** Python 3.13 development environment, NumPy, SciPy, pytest, Ruff.

**Spec:** [Research design](2026-09-13-adaptive-interactions-research-design.md), especially "First performance investigation", and [cost audit](2026-09-13-adaptive-interaction-costs.md).

## Global constraints

- Target Python 3.12+ and the existing Ruff configuration.
- Keep benchmark workers sequential, separate from other heavy work.
- Preserve mathematical names where they make the numerical implementation clearer.
- Do not change the rank policy, numerical tolerances, error ledger, solver dispatch, penalty definition or public API.
- Do not transfer the positive union rank as an active-face rank. Transfer support only; weighted summaries and zero-face state start empty.
- Reuse applies only to a complete ordered fixed-coordinate tensor family within this fit. Preserve the existing raw-coordinate route and its tensor exclusion.
- Bind source provenance at component construction. Capturing only at optimizer return would authenticate an intervening mutation.
- Drop fit-local provenance and receipts during pickle and deepcopy.
- Preserve unrelated work and leave all changes uncommitted.

## Files and interfaces

- Modify `src/superglm/reml/penalty_algebra.py` for lazy receipt creation and `_reuse_fixed_from` admission. Reuse `_raw_evidence_value`, `_raw_support_values`, `_raw_penalty_arithmetic`, and `_RawPenaltyFamilyReceipt`; do not introduce a separate support-certificate implementation.
- Modify `src/superglm/reml/discrete.py` to populate the existing result field from the final `penalties` rebuild.
- Modify `src/superglm/model/fit_ops.py` to prefer `best.reml_penalties` when it is not `None`.
- Modify `src/superglm/model/reml_finalize.py` to supply `_reuse_fixed_from` and release the old result-owned family after the terminal context is accepted.
- Create `tests/test_penalty_fixed_tensor_reuse.py` for small real tensor builder and public-fit regressions.
- Create `docs/research/2026-09-13-tensor-support-handoff-report.md` for evidence and remaining measurement gates.

The builder additions are:

```python
def build_penalty_components(
    group_matrices, reml_groups, cache=None, *,
    _reuse_raw_from=None, _reuse_fixed_from=None,
): ...

def build_penalty_context(
    group_matrices, reml_groups, cache=None, *,
    _reuse_raw_from=None, _reuse_fixed_from=None,
): ...
```

## Task 1: Demonstrate the duplicate producer and stale consumer

- [x] Add a public Gaussian fit with 260 observations, two `Spline(n_knots=5)` mains, a discrete tensor interaction and three REML iterations. Install the real `_penalty_support` counting wrapper before construction. Assert that calls with two matrices at the tensor width number one. The current implementation must fail with two.

```python
def counted(matrices):
    if len(matrices) == 2:
        tensor_widths.append(matrices[0].shape[0])
    return original(matrices)

assert tensor_widths == [tensor_group.size]
```

- [x] Capture the real optimizer family and terminal context through pass-through wrappers in a separate ownership test. Require distinct entry and optimizer families, forwarding of the populated optimizer family, a new terminal owner, and release of the optimizer owner after the fit. Keep these dispatch/lifetime assertions separate from numerical comparisons.
- [x] Run `uv run pytest tests/test_penalty_fixed_tensor_reuse.py -q`. Record the expected failing assertions before production edits.

## Task 2: Admit exact fixed tensor evidence lazily

- [x] Build a real small `DiscretizedTensorGroupMatrix` with two diagonal Kronecker penalties, a shared null direction and identity `R_inv`. Use this fixture for direct builder tests so the selected rank and positive spectrum are known independently.
- [x] Add failing tests for support sharing, cold initial construction, fresh mutable target owners and empty weighted/face state. An unused entry family must never call `get_support`.
- [x] At construction bind the existing `_penalty_group_cache_key` plus exact array/layout evidence for raw matrices, solver matrices, the coordinate map and tensor basis identity, and all component metadata/placement. Reuse the existing exact byte evidence representation; retain one exact receipt on the source through admission, then discard transfer eligibility on the target. Do not retain digest-only authority or make an extra target copy of immutable solver matrices.
- [x] Extend `_RawPenaltyFamilyReceipt` with a shared support-evidence matcher and allow a caller-supplied exact input record. Bind the arithmetic token at component construction; `get_support` may finish the receipt only when the original solver matrix evidence and arithmetic still match. Admission separately compares the current source and target against the full construction record. Check support immutability and all selected-support/error fields through `_raw_support_values`, excluding the independent `_basis_gram_evidence` memo.

```python
if self.support is None:
    inputs = self.fixed_inputs
    admitted = inputs is not None and inputs.matches_support_inputs(self)
    self.support = _penalty_support(self.matrices)
    if admitted and inputs.matches_support_inputs(self):
        self.fixed_family = _RawPenaltyFamilyReceipt.capture(
            self.support, inputs=inputs, arithmetic=inputs.arithmetic, strict_arrays=True
        )
```

- [x] Before ordinary component construction, admit a source only when `_can_cache_penalty_group` holds, the full ordered source owner is valid, the original fixed cache key and exact target evidence match, and the stored receipt still authenticates the source solver arrays and support. Otherwise use the existing fresh builder.
- [x] Construct admitted descriptors with `dataclasses.replace`; share the authenticated immutable solver arrays and support. Construct a new `_PenaltyGroupGeometry` with no `last_evaluation`, `last_weights`, `face_support` or volume state.
- [x] Clear lazy provenance and completed receipts in `__getstate__`, covering both pickle and deepcopy. A restored owner cannot reacquire authority from its deserialized support.
- [x] Add mutation/refusal controls for raw/solver arrays, dtype/layout, component order and policies, placement, map and basis, support/error fields, arithmetic helpers/rank policy, writable evidence, incomplete families and pre-support mutation. A rejected candidate must leave the accepted owner's last evaluation intact.
- [x] Run the builder tests. Demonstrate that bypassing a receipt check makes the corresponding adversarial regression fail; record the mutation check and restore the implementation.

## Task 3: Carry and release the correct family

- [x] Populate `reml_penalties=penalties` in `optimize_discrete_reml`'s final result.
- [x] Forward the producer explicitly in `fit_ops.py`:

```python
reml_penalties=(
    best.reml_penalties if best.reml_penalties is not None else reml_penalties
)
```

- [x] Pass the selected family to both `_reuse_raw_from` and `_reuse_fixed_from` in the final context build. The two builders maintain their separate admissibility rules.
- [x] After successful terminal context construction, replace an existing result-owned family with the new terminal family so the fitted model does not retain optimizer descriptors or weighted evaluations. Do not modify the accepted source owner on failed admission or construction.
- [x] Run the public count and ownership regressions. Require a single tensor support construction, independent final owner and no stale optimizer owner retained through `model._reml_result`.

## Task 4: Verify numerical invariants and bounded scope

- [x] On the analytic tensor fixture assert union rank 3, active-face rank 2 and all-zero rank 0; check reconstruction using recorded bounds. For positive weights `(2, 3)`, compare determinant and derivatives with the independent spectrum `(2, 3, 5)` and its analytic derivatives using certificate bounds plus dimension/epsilon roundoff allowances.
- [x] Compare fresh and reused evaluations for changed positive lambda values and zero faces. Do not assert coefficient-forward accuracy on near-rank fixtures.
- [x] Compare a public fit with the handoff enabled and disabled using selected rank, predictions and fit outputs with dimension/epsilon/conditioning-based tolerances. The complete housing comparisons belong to the parent's serial worker.
- [x] Run focused regressions: `uv run pytest tests/test_penalty_fixed_tensor_reuse.py tests/test_penalty_raw_context_reuse.py tests/test_penalty_algebra_support.py tests/test_distributional_penalty_context.py -q` plus the named frozen tensor, component cache and pair-summary tests from `tests/test_discretize_fit.py`.
- [x] Run Ruff on changed Python files and inspect the final diff. Do not run the full suite or expensive housing fits without parent coordination.
- [x] Write the implementation report with red/green and mutation evidence, ownership/lifetime boundaries, exact retained receipt costs, test commands and unresolved measurement gates. Send the changed-file list and report to the parent. Do not claim a complete-fit speedup until the matched measurements pass.

## Self-review

The plan addresses both producer and consumer ownership. The lazy build rule avoids a third support construction. The exact receipt validates selected support/error evidence and original provenance without changing numerical policy. Source and target own different mutable geometry, and the result releases obsolete owners after terminal construction. Existing zero-face and raw-map contracts remain in place. Dense exact evidence adds fit-phase bytes, but the terminal owner retains zero fixed-handoff authorization bytes. Complete-fit timing and peak/retained-memory acceptance remain measured gates owned by the parent.

## Execution adjustments

The original Poisson count fixture built support once because fixed dispersion did not invoke the terminal nullity consumer. Gaussian reproduces the profiled duplicate with two width-64 support builds and is the effective regression.

The initial plan shared the exact receipt with the target. The final implementation consumes that authority at the handoff. Both successful targets and fresh fallbacks start without eligibility for another transfer, so fitted models retain support/error arrays but no duplicate fixed-handoff snapshots.

Layout capture and backing-owner checks are explicit fixed-path options on the existing evidence helpers. The raw-coordinate route keeps its prior defaults. Tensor marginal object identities and column widths bind basis identity; observation rows, bin arrays and weights are outside this penalty-support record.

Execution evidence, the exact test command and remaining measurement gates are recorded in [the implementation report](2026-09-13-tensor-support-handoff-report.md). No implementation commit was created by this worker.
