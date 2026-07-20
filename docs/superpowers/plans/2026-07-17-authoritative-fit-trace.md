# Authoritative Fit Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace synthetic, mixed-state, and replay-based diagnostics with a globally ordered trace of actual evaluations, decisions, commits, convergence checks, and authoritative terminal states.

**Architecture:** A dependency-free `TraceRun` allocates run-local integer state/evaluation IDs and emits immutable events to null, memory, or JSONL sinks. Solvers emit at the computation site; REML optimizers and finalizers link their actual evaluated states; the transactional workspace owns the run and emits private/public terminal or failed events before the one public fit-state commit.

**Tech Stack:** Python dataclasses/protocols, NumPy, JSONL, pytest, existing IRLS/PIRLS/REML solvers, the shared `benchmark_fit_state_trace.py` performance harness.

---

## File map

- Create `src/superglm/_fit_trace.py`: event schema, ID allocation, sinks, and invariant validation.
- Create `tests/test_fit_trace.py`: schema, ordering, null-sink, persistence, and legacy-loading tests.
- Create `tests/test_reml_trace_lineage.py`: exact/discrete/EFS/SCOP lineage and convergence-state oracles.
- Create `tests/test_fit_trace_terminal.py`: private/public terminal and failure/commit integration.
- Create `tests/test_trace_fidelity_matrix.py`: cross-path invariant and no-extra-work matrix.
- Modify `src/superglm/solvers/irls_state.py`: truthful trial-attempt decisions and state IDs.
- Modify `src/superglm/solvers/pirls.py`: evaluated/committed state events and committed-only iteration log.
- Modify `src/superglm/solvers/irls_direct.py`: direct/SCOP proposal lineage and null-sink integration.
- Modify `src/superglm/solvers/scop_newton.py`: local proposal attempts without false authority.
- Modify `src/superglm/reml/direct.py`, `discrete.py`, `efs.py`, and `scop_efs.py`: actual optimizer evaluation and convergence events.
- Modify `src/superglm/model/reml_execute.py`: delete replay/synthetic tracing.
- Modify `src/superglm/model/reml_finalize.py`: trace actual final and QP refits.
- Modify `src/superglm/model/runtime_canonicalize.py`: distinct solver/public terminal states.
- Modify `src/superglm/model/reml_debug.py`: persistence adapters, merged ordered view, terminal-driven summaries.
- Modify `src/superglm/model/telemetry_ops.py` and `api.py`: terminal IDs, coordinate spaces, and convergence decomposition.

### Task 1: Add dependency-free trace primitives

**Files:**
- Create: `src/superglm/_fit_trace.py`
- Create: `tests/test_fit_trace.py`

- [ ] **Step 1: Write failing schema and sink tests**

```python
def test_trace_run_assigns_one_sequence_across_channels():
    sink = MemoryTraceSink()
    run = TraceRun("run-1", sink=sink, clock=lambda: 12.5)
    run.emit("evaluation", channel="pirls", state_id=run.next_state_id(), deviance=2.0)
    run.emit("evaluation", channel="reml", state_id=run.next_state_id(), objective=4.0)
    assert [event.sequence for event in sink.events] == [1, 2]


def test_null_sink_never_materializes_payload():
    called = False

    def payload():
        nonlocal called
        called = True
        return {"beta": [1.0]}

    TraceRun("run-1", sink=NullTraceSink()).emit_lazy("evaluation", payload)
    assert not called


def test_numerical_event_requires_state_identity():
    run = TraceRun("run-1", sink=MemoryTraceSink())
    with pytest.raises(ValueError, match="state_id"):
        run.emit("evaluation", channel="pirls", deviance=1.0)
```

- [ ] **Step 2: Run to verify the module is missing**

Run: `rtk uv run pytest tests/test_fit_trace.py -q`

Expected: FAIL importing `superglm._fit_trace`.

- [ ] **Step 3: Implement the schema and sinks**

```python
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TraceEvent:
    schema_version: int
    run_id: str
    sequence: int
    timestamp: float
    event_kind: str
    channel: str
    purpose: str
    authoritative: bool
    payload: Mapping[str, object]


class NullTraceSink:
    enabled = False

    def append(self, event: TraceEvent) -> None:
        return None


@dataclass
class MemoryTraceSink:
    events: list[TraceEvent] = field(default_factory=list)
    enabled: bool = True

    def append(self, event: TraceEvent) -> None:
        self.events.append(event)


class TraceRun:
    def __init__(self, run_id, sink=None, clock=time.time):
        self.run_id = run_id
        self.sink = sink or NullTraceSink()
        self.clock = clock
        self._sequence = self._state_id = self._evaluation_id = self._basis_id = 0
```

Implement monotonic allocators, eager/lazy emission, known event kinds, required state identity for numerical payloads, and immutable mappings. Use integer IDs; never hash coefficient arrays.

- [ ] **Step 4: Run primitive tests**

Run: `rtk uv run pytest tests/test_fit_trace.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/_fit_trace.py tests/test_fit_trace.py
rtk git commit -m "Add authoritative fit trace primitives"
```

### Task 2: Persist one ordered run while loading legacy files

**Files:**
- Modify: `src/superglm/model/reml_debug.py:10-121`
- Modify: `tests/test_fit_trace.py`
- Modify: `tests/test_fit_reml_debug.py`

- [ ] **Step 1: Write persistence compatibility tests**

```python
def test_jsonl_sink_preserves_run_sequence_across_suffixes(tmp_path):
    sink = JSONLTraceSink(tmp_path, "run-1")
    run = TraceRun("run-1", sink=sink, clock=lambda: 0.0)
    run.emit("evaluation", channel="pirls", state_id=run.next_state_id(), deviance=1.0)
    run.emit("evaluation", channel="reml", state_id=run.next_state_id(), objective=2.0)
    loaded = load_reml_debug_run(tmp_path, "run-1")
    assert [event["sequence"] for event in loaded.events] == [1, 2]


def test_loader_accepts_legacy_rows_without_schema_version(tmp_path):
    (tmp_path / "old_run.json").write_text('{"method":"fit_reml"}')
    (tmp_path / "old_reml.jsonl").write_text('{"iteration":1,"objective_after":3.0}\n')
    loaded = load_reml_debug_run(tmp_path, "old")
    assert loaded.reml_rows[0]["iteration"] == 1
```

- [ ] **Step 2: Run to verify there is no merged view**

Run: `rtk uv run pytest tests/test_fit_trace.py tests/test_fit_reml_debug.py -q`

Expected: FAIL because current suffix writers have no run-wide sequence or canonical events view.

- [ ] **Step 3: Add a JSONL sink and adapter**

Keep `REMLDebugRecorder.append_jsonl()` temporarily; adapt its payload to a legacy non-authoritative event. Add `events` to `REMLDebugRun`, merge canonical rows by `sequence`, and preserve `reml_rows`, `pirls_rows`, and `scop_rows`. Never infer cross-file order for legacy rows.

- [ ] **Step 4: Run debug loader/writer tests**

Run: `rtk uv run pytest tests/test_fit_trace.py tests/test_fit_reml_debug.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/reml_debug.py tests/test_fit_trace.py tests/test_fit_reml_debug.py
rtk git commit -m "Order REML debug events across trace files"
```

### Task 3: Report all IRLS trial attempts truthfully

**Files:**
- Modify: `src/superglm/solvers/irls_state.py:28-34,121-141`
- Modify: `tests/test_irls_state.py:69-171`

- [ ] **Step 1: Add attempt-count assertions**

```python
def test_total_rejection_reports_every_attempt():
    decision = _select_irls_trial(
        committed=_synthetic_state(2.0),
        proposal=_synthetic_state(10.0),
        evaluate_state=lambda alpha: _synthetic_state(10.0),
        max_halving=5,
    )
    assert decision == _IRLSStepDecision(
        alpha=0.0, step_halvings=0, step_rejected=True, trials_attempted=6
    )
```

Add full-acceptance `1` and half-step `2` cases.

- [ ] **Step 2: Run to verify the field is absent**

Run: `rtk uv run pytest tests/test_irls_state.py -q`

Expected: FAIL constructing/comparing `_IRLSStepDecision`.

- [ ] **Step 3: Extend the decision value**

Append `trials_attempted: int = 1` to the dataclass. Return `1` for full acceptance, `depth + 1` for a half-step, and `max_halving + 1` for rejection. Preserve rejection's `step_halvings == 0`.

- [ ] **Step 4: Run IRLS-state tests**

Run: `rtk uv run pytest tests/test_irls_state.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/solvers/irls_state.py tests/test_irls_state.py
rtk git commit -m "Report IRLS trial attempts accurately"
```

### Task 4: Trace PIRLS evaluated and committed states

**Files:**
- Modify: `src/superglm/solvers/pirls.py:55-104,135-414,584-687`
- Modify: `src/superglm/model/api.py:464-515`
- Modify: `tests/test_irls_state.py:183-293`

- [ ] **Step 1: Add lineage tests for full, half, and rejected steps**

```python
def assert_decision_lineage(events, *, rejected):
    decision = next(event for event in events if event.event_kind == "step_decision")
    commits = [event for event in events if event.event_kind == "state_commit"]
    committed = commits[-1].payload["state_id"]
    if rejected:
        assert decision.payload["accepted_alpha"] == 0.0
        assert decision.payload["committed_state_id"] == decision.payload["base_state_id"]
        assert not decision.payload["fit_converged"]
    assert committed == decision.payload["committed_state_id"]
```

Use the controlled Gaussian fixtures already in `tests/test_irls_state.py`; also assert null-sink results are byte-for-byte equal to the no-trace result.

- [ ] **Step 2: Run to verify PIRLS emits no canonical events**

Run: `rtk uv run pytest tests/test_irls_state.py -k 'pirls and trace' -q`

Expected: FAIL.

- [ ] **Step 3: Thread an optional `trace_run` through PIRLS**

Promote the immutable `_IRLSState` contract to the design's `SolverState` (retain `_IRLSState` as a migration alias) by adding trailing defaulted `state_id`, `evaluation_id`, `state_space`, `basis_id`, lambdas, dispersion, convergence value, and termination reason. Allocate/evaluate IDs exactly where `_evaluate_irls_state()` runs. Emit `evaluation`, then one `step_decision`, then `state_commit` only for the retained evaluated state. Separate working-state W/rank payload from retained-state eta/mu/deviance payload. Append only committed states to `iteration_log`; add trailing defaulted lineage fields to `IterationDiagnostics` and `PIRLSResult` to preserve existing constructors.

- [ ] **Step 4: Run PIRLS and API diagnostics tests**

Run: `rtk uv run pytest tests/test_irls_state.py tests/test_api.py -k 'iteration or pirls or trace' -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/solvers/pirls.py src/superglm/model/api.py tests/test_irls_state.py
rtk git commit -m "Trace committed PIRLS states"
```

### Task 5: Trace direct IRLS and enclosing SCOP decisions

**Files:**
- Modify: `src/superglm/solvers/irls_direct.py:262-292,644-685,799-850,952-1220`
- Modify: `tests/test_irls_direct.py:44-52,258-329`
- Modify: `tests/test_irls_state.py:295-373`

- [ ] **Step 1: Add direct-lineage tests**

```python
@pytest.mark.parametrize("outcome", ["full", "half", "reject"])
def test_direct_decision_commits_the_evaluated_state(outcome):
    sink = MemoryTraceSink()
    result = fit_controlled_direct(outcome, trace_run=TraceRun("direct", sink=sink))
    decision = [e for e in sink.events if e.event_kind == "step_decision"][-1]
    commit = [e for e in sink.events if e.event_kind == "state_commit"][-1]
    assert decision.payload["committed_state_id"] == commit.payload["state_id"]
    if outcome == "reject":
        assert not result.converged
        assert decision.payload["trials_attempted"] == 6
```

- [ ] **Step 2: Run to verify direct IRLS lacks lineage**

Run: `rtk uv run pytest tests/test_irls_direct.py tests/test_irls_state.py -k 'direct and trace' -q`

Expected: FAIL.

- [ ] **Step 3: Instrument actual direct evaluations**

Use the same event contract as PIRLS. Link each SCOP local proposal to the enclosing direct proposal ID, but do not call it authoritative until the enclosing IRLS state commits. Preserve level-two extrema payloads and add a compatibility adapter for `_LevelTwoRecorder` while migration proceeds.

- [ ] **Step 4: Run direct/SCOP-state tests**

Run: `rtk uv run pytest tests/test_irls_direct.py tests/test_irls_state.py tests/test_scop_irls_state.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/solvers/irls_direct.py tests/test_irls_direct.py tests/test_irls_state.py tests/test_scop_irls_state.py
rtk git commit -m "Trace direct IRLS state decisions"
```

### Task 6: Remove replay tracing and demote the synthetic compatibility row

**Files:**
- Modify: `src/superglm/model/fit_ops.py:235-276,436-595,699-1000`
- Modify: `src/superglm/model/reml_execute.py:35-95,228-375`
- Modify: `src/superglm/model/reml_ops.py:120-264`
- Modify: `tests/test_fit_reml_debug.py`

- [ ] **Step 1: Prove diagnostics currently add solver work**

```python
@pytest.mark.parametrize("discrete", [False, True])
def test_trace_level_two_does_not_add_solver_calls(discrete, monkeypatch, tmp_path):
    counts = []
    outputs = []
    for level in (0, 2):
        count = CallCounter.wrap(monkeypatch, reml_execute, "fit_irls_direct")
        model = _make_unconstrained_model(discrete=discrete)
        with debug_level(level, tmp_path):
            model.fit_reml(*_make_demo_data(), max_reml_iter=3)
        counts.append(count.value)
        outputs.append((model.result.beta.copy(), model.result.deviance))
    assert counts[0] == counts[1]
    np.testing.assert_allclose(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == pytest.approx(outputs[1][1])
```

- [ ] **Step 2: Run to expose the discarded replay**

Run: `rtk uv run pytest tests/test_fit_reml_debug.py -k 'solver_calls or replay' -q`

Expected: FAIL because level two runs `_record_non_scop_pirls_trace()`.

- [ ] **Step 3: Delete replay work and make the interim summary non-authoritative**

Remove `_record_non_scop_pirls_trace()` and its call sites. Adapt `_record_non_scop_reml_trace()` to an explicitly legacy, synthetic, non-authoritative summary so existing debug artifacts remain usable while Tasks 7–11 add real optimizer events. Thread `workspace.trace_run` through actual solver/optimizer calls. `record_diagnostics` may select payload, never branching or evaluations.

- [ ] **Step 4: Run debug and call-count tests**

Run: `rtk uv run pytest tests/test_fit_reml_debug.py tests/test_irls_direct.py tests/test_irls_state.py -q`

Expected: PASS; no event has `trace_replay=true`, and the interim summary has `authoritative=false`.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/fit_ops.py src/superglm/model/reml_execute.py src/superglm/model/reml_ops.py tests/test_fit_reml_debug.py
rtk git commit -m "Remove diagnostic replay fits"
```

### Task 7: Trace exact REML evaluations and convergence identity

**Files:**
- Modify: `src/superglm/reml/direct.py:118-136,138-340,353-520,536-544`
- Modify: `src/superglm/reml/result.py:39-67`
- Create: `tests/test_reml_trace_lineage.py`

- [ ] **Step 1: Add exact REML state-oracle tests**

```python
def test_exact_best_and_convergence_states_are_evaluated(exact_trace_result):
    result, events = exact_trace_result
    evaluations = {
        e.payload["state_id"] for e in events if e.event_kind == "evaluation"
    }
    assert result.best_state_id in evaluations
    assert result.outer_converged_state_id is None or result.outer_converged_state_id in evaluations
    terminal = [e for e in events if e.event_kind == "state_commit"][-1]
    assert terminal.payload["state_id"] == result.best_state_id
```

Also force line-search fallback and assert an unevaluated steepest-descent rho is not committed.

- [ ] **Step 2: Run to verify exact optimizer emits nothing**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py -k exact -q`

Expected: FAIL.

- [ ] **Step 3: Emit at actual exact evaluation sites**

Trace bootstrap, outer candidate, line-search trial, decision, and best-state commit where objectives are computed. Store `best_state_id` separately from `outer_converged_state_id`; a search vector is not a state until its coefficient fit and criterion are evaluated. Append defaulted result fields to avoid breaking private constructors.

- [ ] **Step 4: Run exact REML tests**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py tests/test_reml.py -k exact -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/reml/direct.py src/superglm/reml/result.py tests/test_reml_trace_lineage.py
rtk git commit -m "Trace exact REML evaluations"
```

### Task 8: Trace discrete REML and fix post-update convergence claims

**Files:**
- Modify: `src/superglm/reml/discrete.py:231-250,254-493,620-841,885-1027`
- Modify: `tests/test_reml_trace_lineage.py`

- [ ] **Step 1: Add discrete convergence-state tests**

```python
def test_discrete_terminal_lambdas_match_checked_state(discrete_trace_result):
    result, events = discrete_trace_result
    checked = next(
        e for e in reversed(events) if e.payload.get("convergence_criterion") is not None
    )
    terminal = next(e for e in reversed(events) if e.event_kind == "state_commit")
    if result.converged:
        assert checked.payload["state_id"] == terminal.payload["state_id"]
        assert checked.payload["lambdas"] == terminal.payload["lambdas"]
```

Assert analytical `n_iter=0` trial objects do not claim `fit_converged=True`, and recompute the final production objective from the final full refit.

- [ ] **Step 2: Run to reproduce identity mismatch**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py -k discrete -q`

Expected: FAIL.

- [ ] **Step 3: Trace POI, analytical trial, decision, and final refit**

Analytical trials are proposals with `fit_converged=False`. If convergence is checked at the current rho, do not retain a subsequent rho update unless the criterion is reevaluated there. Emit the actual final refit/objective and return the same state identity.

- [ ] **Step 4: Run discrete REML tests**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py tests/test_reml.py -k discrete -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/reml/discrete.py tests/test_reml_trace_lineage.py
rtk git commit -m "Tie discrete REML convergence to evaluated state"
```

### Task 9: Trace private EFS without changing public routing

**Files:**
- Modify: `src/superglm/reml/efs.py:93-168,184-420`
- Modify: `src/superglm/model/reml_ops.py:216-264`
- Modify: `tests/test_reml_trace_lineage.py`

- [ ] **Step 1: Add EFS bootstrap/main/final tests**

```python
def test_efs_terminal_is_actual_final_pirls(efs_trace_result):
    result, events = efs_trace_result
    phases = [e.payload.get("phase") for e in events if e.event_kind == "evaluation"]
    assert {"bootstrap", "outer", "final"} <= set(phases)
    terminal = next(e for e in reversed(events) if e.event_kind == "state_commit")
    np.testing.assert_allclose(terminal.payload["beta"], result.pirls_result.beta)
```

- [ ] **Step 2: Run to verify the private path is untraced**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py -k efs -q`

Expected: FAIL.

- [ ] **Step 3: Emit real EFS events**

Instrument `optimize_efs_reml()` directly. A lambda commit occurs only after a coefficient evaluation; stale-basis guards remain non-authoritative. Do not relax the public `fit_reml()` selection-penalty guard merely to expose EFS.

- [ ] **Step 4: Run EFS tests**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py tests/test_reml_efs.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/reml/efs.py src/superglm/model/reml_ops.py tests/test_reml_trace_lineage.py
rtk git commit -m "Trace EFS coefficient states"
```

### Task 10: Link SCOP local proposals to enclosing IRLS commits

**Files:**
- Modify: `src/superglm/solvers/scop_newton.py:50-119,348-388,1024-1085`
- Modify: `src/superglm/solvers/irls_direct.py:799-850,952-1037`
- Modify: `src/superglm/reml/scop_efs.py:589-612,698-925,940-995`
- Modify: `tests/test_scop_irls_state.py:133-280`
- Modify: `tests/test_scop_efs.py:1272-1319`

- [ ] **Step 1: Add rejected and half-stepped SCOP linkage tests**

```python
def test_rejected_scop_proposal_is_not_authoritative(forced_scop_rejection):
    result, events = forced_scop_rejection
    proposal = next(e for e in events if e.payload.get("solver") == "scop_newton")
    decision = next(e for e in events if e.event_kind == "step_decision")
    assert not proposal.authoritative
    assert decision.payload["accepted_alpha"] == 0.0
    assert proposal.payload["proposal_state_id"] != decision.payload["committed_state_id"]
```

Also assert local alpha/attempt counts and label outer `objective_after` as a fixed-coefficient surrogate.

- [ ] **Step 2: Run to expose unlinked SCOP rows**

Run: `rtk uv run pytest tests/test_scop_irls_state.py tests/test_scop_efs.py -k trace -q`

Expected: FAIL.

- [ ] **Step 3: Thread proposal IDs through SCOP Newton**

SCOP Newton emits non-authoritative proposal evaluations. Direct IRLS owns the acceptance/half-step/rejection and commit. SCOP EFS emits the true final refit objective; the fixed-coefficient proposed-lambda surrogate is explicitly labelled and never terminal.

- [ ] **Step 4: Run SCOP suites**

Run: `rtk uv run pytest tests/test_scop_irls_state.py tests/test_scop_efs.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/solvers/scop_newton.py src/superglm/solvers/irls_direct.py src/superglm/reml/scop_efs.py tests/test_scop_irls_state.py tests/test_scop_efs.py
rtk git commit -m "Link SCOP proposals to retained IRLS states"
```

### Task 11: Trace final, fixed-monotone, and QP constrained states

**Files:**
- Modify: `src/superglm/model/reml_finalize.py:45-195`
- Modify: `src/superglm/model/reml_execute.py:98-225`
- Modify: `src/superglm/model/fit_ops.py:849-1000`
- Modify: `tests/test_reml_trace_lineage.py`
- Modify: `tests/test_monotone_fit.py:297-313,436-542`

- [ ] **Step 1: Add finalization identity tests**

```python
def test_qp_terminal_belongs_to_constrained_refit(qp_passthrough_trace):
    model, events = qp_passthrough_trace
    unconstrained = next(e for e in events if e.purpose == "unconstrained_reml")
    terminal = next(e for e in reversed(events) if e.event_kind == "state_commit")
    assert not unconstrained.authoritative
    assert terminal.purpose == "constrained_final_fit"
    np.testing.assert_allclose(terminal.payload["beta"], model._solver_result.beta)
```

Add fixed QP/SCOP and final-refit convergence-decomposition cases.

- [ ] **Step 2: Run to verify final refits are absent**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py tests/test_monotone_fit.py -k 'terminal or final or qp' -q`

Expected: FAIL.

- [ ] **Step 3: Trace every authoritative final coefficient fit**

Emit actual fixed-lambda fit, unconstrained REML state, mapped-basis direct refit, constrained QP refit, phi/objective evaluation, and final state commit with distinct purposes. Keep `outer_converged`, `inner_converged`, and `final_refit_converged` separate. Objective, coefficients, lambdas, and phi on a terminal REML result must share one state ID.

Now that every non-SCOP optimizer path emits actual evaluations, delete `_record_non_scop_reml_trace()` and its call sites. No synthetic production event remains.

- [ ] **Step 4: Run REML/monotone tests**

Run: `rtk uv run pytest tests/test_reml_trace_lineage.py tests/test_monotone_fit.py tests/test_shape_reml.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/reml_finalize.py src/superglm/model/reml_execute.py src/superglm/model/fit_ops.py tests/test_reml_trace_lineage.py tests/test_monotone_fit.py
rtk git commit -m "Trace authoritative REML final fits"
```

### Task 12: Emit private/public terminal and failed-run events

**Files:**
- Modify: `src/superglm/model/runtime_canonicalize.py:370-427`
- Modify: `src/superglm/model/fit_state.py`
- Modify: `src/superglm/model/fit_ops.py`
- Create: `tests/test_fit_trace_terminal.py`

- [ ] **Step 1: Write terminal/transaction tests**

```python
def test_private_and_public_terminals_match_installed_results(canonical_fit_trace):
    model, events = canonical_fit_trace
    private = next(e for e in events if e.purpose == "private_terminal")
    public = next(e for e in events if e.purpose == "public_terminal")
    np.testing.assert_allclose(private.payload["beta"], model._solver_result.beta)
    np.testing.assert_allclose(public.payload["beta"], model.result.beta)
    assert private.payload["state_space"] == "solver"
    assert public.payload["state_space"] == "public_canonical"


def test_failed_run_cannot_replace_previous_terminal(fitted_model, failure_injector):
    prior_revision = fitted_model._fit_revision
    with pytest.raises(InjectedFitFailure):
        failure_injector.refit(fitted_model)
    assert fitted_model._fit_revision == prior_revision
```

- [ ] **Step 2: Run to verify lifecycle integration is absent**

Run: `rtk uv run pytest tests/test_fit_trace_terminal.py -q`

Expected: FAIL.

- [ ] **Step 3: Emit terminal/failure events at workspace boundaries**

Emit private terminal after authoritative solver/final refit, public terminal after canonicalization and before compact row release, and `run_failed` from the workspace exception boundary with the last committed workspace state and exception class. Only a successful candidate may transfer the run/terminal IDs into `FitState`; tracing must not implement rollback.

- [ ] **Step 4: Run terminal and transaction tests**

Run: `rtk uv run pytest tests/test_fit_trace_terminal.py tests/test_fit_transactions.py tests/test_fit_state_retention.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/runtime_canonicalize.py src/superglm/model/fit_state.py src/superglm/model/fit_ops.py tests/test_fit_trace_terminal.py
rtk git commit -m "Tie fit traces to state transactions"
```

### Task 13: Derive summaries and telemetry from terminal events

**Files:**
- Modify: `src/superglm/model/reml_debug.py:136-188`
- Modify: `src/superglm/model/telemetry_ops.py:40-69,194-205`
- Modify: `src/superglm/model/api.py:464-532`
- Modify: `tests/test_training_telemetry.py`
- Modify: `tests/test_fit_reml_debug.py`

- [ ] **Step 1: Add terminal-driven summary tests**

```python
def test_summary_ignores_later_nonterminal_row(debug_run_with_tail):
    summary = summarize_reml_debug_run(debug_run_with_tail)
    terminal = next(
        event for event in debug_run_with_tail.events if event["event_kind"] == "terminal"
    )
    assert summary["terminal_state_id"] == terminal["payload"]["state_id"]
    assert summary["final_lambdas_json"] == json.dumps(
        terminal["payload"]["lambdas"], sort_keys=True
    )
```

Add fixed-lambda monotone telemetry, coordinate-space columns, and legacy fallback cases.

- [ ] **Step 2: Run to expose last-row semantics**

Run: `rtk uv run pytest tests/test_training_telemetry.py tests/test_fit_reml_debug.py -q`

Expected: FAIL because current summary uses the last REML/PIRLS rows.

- [ ] **Step 3: Project terminal truth without removing old keys**

Canonical traces use the authoritative terminal event. Preserve existing DataFrame columns and telemetry keys; add run/state/basis IDs, state space, and inner/outer/final/overall convergence. Legacy artifacts without terminal events retain a clearly labelled fallback path.

- [ ] **Step 4: Run telemetry/debug tests**

Run: `rtk uv run pytest tests/test_training_telemetry.py tests/test_fit_reml_debug.py tests/test_api.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/superglm/model/reml_debug.py src/superglm/model/telemetry_ops.py src/superglm/model/api.py tests/test_training_telemetry.py tests/test_fit_reml_debug.py
rtk git commit -m "Summarize authoritative terminal states"
```

### Task 14: Enforce the full trace-fidelity and overhead matrix

**Files:**
- Create: `tests/test_trace_fidelity_matrix.py`
- Modify: `benchmarks/benchmark_fit_state_trace.py`
- Modify: `tests/test_fit_state_trace_benchmark.py`

- [ ] **Step 1: Add cross-path invariants**

```python
@pytest.mark.parametrize(
    "path",
    ["exact", "discrete", "private_efs", "scop", "fixed_qp", "fixed_scop", "qp_passthrough"],
)
def test_every_commit_and_terminal_has_evaluated_lineage(trace_case, path):
    model, events = trace_case(path)
    evaluated = {
        e.payload["state_id"] for e in events if e.event_kind == "evaluation"
    }
    for event in events:
        if event.event_kind == "state_commit":
            assert event.payload["state_id"] in evaluated
    terminal = next(e for e in reversed(events) if e.event_kind == "terminal")
    assert terminal.payload["state_id"] in evaluated
    assert not any(e.purpose == "diagnostic_replay" for e in events)
```

Add level-zero/level-two exact equality for solver, objective, deviance, derivative, and matrix-evaluation call counts.

- [ ] **Step 2: Run the full focused trace suite**

```bash
rtk uv run pytest tests/test_fit_trace.py tests/test_irls_state.py tests/test_irls_direct.py -q
rtk uv run pytest tests/test_fit_reml_debug.py tests/test_reml_trace_lineage.py -q
rtk uv run pytest tests/test_scop_irls_state.py tests/test_scop_efs.py -m "not slow" -q
rtk uv run pytest tests/test_trace_fidelity_matrix.py -q
```

Expected: PASS.

- [ ] **Step 3: Measure null-sink overhead and evaluation invariance**

Run:

```bash
rtk proxy env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --python 3.14 python benchmarks/benchmark_fit_state_trace.py --suite trace-overhead --warmups 3 --repeats 20 --output /tmp/superglm-trace-overhead.json
```

Expected: null-sink median overhead at most 1%; trace level zero and two have identical numerical results and evaluation counts. File-sink I/O cost is reported but not part of the null-sink gate.

- [ ] **Step 4: Run full static and regression verification**

```bash
rtk uv run pytest tests/ -m "not slow" -q
rtk uv run pytest tests/ -q
rtk uv run ruff check src/ tests/ benchmarks/benchmark_fit_state_trace.py
rtk uv run mypy src/
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/test_trace_fidelity_matrix.py benchmarks/benchmark_fit_state_trace.py tests/test_fit_state_trace_benchmark.py
rtk git commit -m "Verify fit trace fidelity and overhead"
```

## Completion review

Search for legacy trace writes and discarded diagnostic evaluations:

```bash
rtk proxy rg -n 'append_jsonl|trace_replay|_record_non_scop|debug_recorder' src/superglm
```

Every remaining adapter call must be explicitly non-authoritative or a compatibility reader. Verify strict global `sequence`, evaluation-backed commits, terminal/result equality, and level-independent call counts for all seven path families before claiming completion.
