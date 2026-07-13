# Editor Refit And Evidence Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make structural refits commit chart and summary as one painted transition, while semantic revisions, bounded scalar caching, one edited-model materialization, and off-lock evidence work prevent stale UI updates and redundant row-scale scoring.

**Architecture:** `EditorSession` owns the semantic model revision and edit epoch. `EditorWidget` returns post-refit transition envelopes under its mutation lock, while a new evaluation cache and a one-worker evidence coordinator operate on immutable model/dataset snapshots after releasing that lock. The native browser app atomically commits transition envelopes, removes the global overlay after a paint boundary, and refreshes only visible evidence with revision and request-sequence guards.

**Tech Stack:** Python 3.10+, NumPy, FastAPI/uvicorn, pytest, native ES modules, Node's built-in test runner, Python Playwright, hand-built SVG.

---

## Scope and dependency order

Implement these tasks in order. Do not remove the blocking structural overlay until Task 8 has proved that row-scale evidence work runs outside `EditorWidget._lock`.

`docs/superpowers/plans/2026-07-11-editor-foundation-state.md` is a completed prerequisite. Its
contracts are authoritative:

- `EditorSession.model_revision`, `edit_epoch`, and materialization slots already exist; Task 1 is
  checked below for traceability and must not be repeated;
- extend the existing `createEditorStore()`/`createEditorActions()` modules instead of creating a
  second store, action controller, package manifest, or test harness;
- place Node tests in `tests/editor_frontend/` and browser tests in the existing foundation file or
  the shared `tests/editor/` fixtures introduced by the workspace phase;
- retain the `check:frontend` npm command, strict `checkJs`, Playwright dev dependency, browser
  marker, and single Chromium CI job;
- retain `#appAlert` and the foundation recovery state; evidence errors update only their panel and
  never replace mutation recovery.

Where a standalone code sample below says to create `store.js`, `actions.js`, `package.json`, or a
browser harness, interpret it as a focused extension of the prerequisite file named above. Do not
reintroduce its legacy standalone filename or state shape.

```text
semantic revision
  -> structural transition envelope
  -> browser-compatible atomic envelope commit (overlay still blocking)
  -> bounded edited-model materialization
  -> scalar EvaluationCache and fit-artifact reuse
  -> reports sharing the scalar cache
  -> one-worker EvidenceCoordinator and immutable snapshots
  -> off-lock /metrics and /report routes
  -> background browser freshness, stale rejection, and overlay release
  -> real-browser ordering/request-count coverage
```

## File responsibility map

- Modify `src/superglm/editor/session.py`: own `model_revision`, `edit_epoch`, prediction-changing invalidation, and the one-current-revision materialized model.
- Modify `src/superglm/editor/apply.py`: clone mutable model state without copying retained evaluation frames or the fitted design matrix.
- Create `src/superglm/editor/evaluation_cache.py`: immutable keys/snapshots, fit-artifact reuse, scalar-only reference/current cache, and report-ready metric pairs.
- Create `src/superglm/editor/evidence.py`: one running work item, one latest pending work item, same-key de-duplication, and superseded outcomes.
- Modify `src/superglm/editor/metrics.py`: separate scalar metric calculation from live-strip payload assembly and add a pure fit-artifact path.
- Modify `src/superglm/editor/reports.py`: assemble reports from cached split metric dictionaries instead of rescoring models.
- Modify `src/superglm/editor/summaries.py` and `src/superglm/editor/persistence.py`: use the current edit epoch's shared materialized model.
- Modify `src/superglm/editor/widget.py`: expose revisions in state, build structural envelopes under the mutation lock, capture immutable evidence snapshots, wait for evidence outside the lock, and close the worker.
- Modify `src/superglm/editor/server.py`: preserve `/metrics` and `/report`, accept/echo revision and request sequence, and return structural envelopes from `/collapse_levels`, `/ungroup_levels`, and `/uncollapse_levels`.
- Modify `src/superglm/editor/app/state/store.js`: add authoritative summary commit and evidence
  freshness transitions to the existing state.
- Modify `src/superglm/editor/app/state/actions.js`: add atomic transitions, paint boundary,
  evidence debounce, sequence allocation, and stale-response rejection to the existing controller.
- Modify `src/superglm/editor/app/main.js`, `summary.js`, `metrics.js`, and `reports.js`: consume the store and stop sequencing secondary evidence on the primary mutation path.
- Create `tests/test_editor_evaluation_cache.py` and `tests/test_editor_evidence.py`: focused pure Python cache/concurrency coverage.
- Modify `tests/test_editor.py`: session revision, transition-envelope, route, report-reuse, and materialization integration coverage.
- Modify `tests/editor_frontend/store.test.js` and `tests/editor_frontend/actions.test.js`: pure
  transition/freshness tests through `node:test`.
- Modify `tests/test_editor_browser.py` or add `tests/editor/test_editor_refit_browser.py`: real
  FastAPI/Chromium ordering, request-count, overlay, and stale-response coverage.
- Modify `docs/editor_frontend.md`: document revisions, transition envelopes, cache lifetime, and evidence worker behavior.

### Task 1: Add semantic model revisions and edit epochs

**Prerequisite status:** Satisfied by Phase 1. The checked steps document the revision/cache-slot
contract used by later tasks and must not be executed or committed again.

**Files:**
- Modify: `src/superglm/editor/session.py:48-73,231-276,467-518,634-695,754-896,1207-1230`
- Modify: `src/superglm/editor/widget.py:90-108`
- Test: `tests/test_editor.py`

- [x] **Step 1: Write failing revision-rule tests**

Add these tests beside `test_widget_state_includes_edit_history`:

```python
def test_editor_model_revision_changes_only_when_predictions_can_change(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    assert session.model_revision == 0

    session.select_indices("region", [1])
    assert session.model_revision == 0

    session.reorder_levels("region", [1], target_index=2)
    assert session.model_revision == 0

    session.shift("region", np.log(1.05))
    assert session.model_revision == 1

    session.undo()
    assert session.model_revision == 2

    session.redo()
    assert session.model_revision == 3

    session.reset("region")
    assert session.model_revision == 4

    session.reset("region")
    assert session.model_revision == 4


def test_structural_replacement_advances_one_revision(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["B", "C"])

    session.replace_with_collapsed_levels("region", method="fit")

    assert session.model_revision == 1
    assert session.edit_epoch == 1


def test_widget_state_exposes_model_revision(editor_model):
    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    try:
        assert widget._state()["model_revision"] == 0
        session.select_indices("x_spline", [20, 21])
        session.shift("x_spline", np.log(1.05))
        assert widget._state()["model_revision"] == 1
    finally:
        widget.close()
```

- [x] **Step 2: Run the focused tests and confirm the missing contract**

Run:

```bash
rtk pytest tests/test_editor.py -k "model_revision or structural_replacement_advances" -q
```

Expected: failures report that `EditorSession` has no `model_revision` or `edit_epoch`.

- [x] **Step 3: Implement the revision owner in `EditorSession`**

Initialize the counters and cache slots in `EditorSession.__init__`:

```python
self._model_revision = 0
self._edit_epoch = 0
self._materialized_edit_model = None
self._materialized_edit_epoch: int | None = None
```

Add these members before `to_model`:

```python
@property
def model_revision(self) -> int:
    """Semantic revision for predictions and fit evidence."""
    return self._model_revision


@property
def edit_epoch(self) -> int:
    """Monotonic invalidation token for a materialized manual-edit model."""
    return self._edit_epoch


def _advance_model_revision(self) -> None:
    self._model_revision += 1
    self._edit_epoch += 1
    self._materialized_edit_model = None
    self._materialized_edit_epoch = None


@staticmethod
def _values_changed(before: NDArray, after: NDArray) -> bool:
    return not np.array_equal(
        np.asarray(before, dtype=np.float64),
        np.asarray(after, dtype=np.float64),
    )
```

In `_commit`, compute `changed = self._values_changed(before, after)` before assignment and call `_advance_model_revision()` after clearing `redo_stack` only when `changed` is true. In `reset`, copy the selected values before assignment and advance only when they differ from the restored values. In `undo` and `redo`, advance only after a record is found and its `before` and `after` differ. Call `_advance_model_revision()` once at the end of `replace_in_force_model`; collapse, ungroup, uncollapse, and distribution re-profiling already funnel through that method, so they must not increment separately.

Expose the value from `EditorWidget._state`:

```python
return {
    "model_revision": self.session.model_revision,
    "selected_term": self.selected_term,
    "terms": self.terms,
    "selection": {
        name: self.session.selection(name).astype(int).tolist()
        for name in self.session.terms
    },
    "can_uncollapse_levels": self.session.can_uncollapse_levels(),
    "last_collapse": (
        None
        if not self.session.can_uncollapse_levels()
        or self._collapsed_refit_info is None
        else dict(self._collapsed_refit_info)
    ),
    "history": history_payload(self.session),
}
```

- [x] **Step 4: Run revision and existing edit/history tests**

Run:

```bash
rtk pytest tests/test_editor.py -k "model_revision or structural_replacement_advances or shift_set_interpolate or reset_restores or history" -q
```

Expected: all selected tests pass; selection and display reordering retain the revision, while coefficient and structural changes advance it once.

- [x] **Step 5: Commit the revision contract**

```bash
rtk git add src/superglm/editor/session.py src/superglm/editor/widget.py tests/test_editor.py
rtk git commit -m "Add semantic editor model revisions"
```

### Task 2: Return structural transition envelopes under one widget lock

**Files:**
- Modify: `src/superglm/editor/widget.py:90-121,457-554,667-687`
- Modify: `src/superglm/editor/server.py:212-241`
- Test: `tests/test_editor.py:2966-3090`

- [ ] **Step 1: Change the HTTP tests to require one consistent envelope**

Replace the root-summary assertions in the collapse and uncollapse HTTP tests with:

```python
payload = _post_json(
    f"{widget.url}/collapse_levels",
    {"term": "region", "method": "fit"},
)

assert set(payload) == {"state", "summary", "timing"}
assert payload["summary"]["available"] is True
assert payload["summary"]["source"] == "in_force"
assert payload["state"]["model_revision"] == session.model_revision == 1
assert payload["state"]["terms"]["region"]["y"][1] == pytest.approx(
    payload["state"]["terms"]["region"]["y"][2]
)
assert payload["timing"]["fit_ms"] >= 0.0
assert payload["timing"]["summary_ms"] >= 0.0
assert payload["timing"]["state_ms"] >= 0.0
```

For uncollapse, assert `payload["state"]["can_uncollapse_levels"] is False`, `payload["state"]["last_collapse"] is None`, and `payload["summary"]["source"] == "in_force"` without a follow-up `/state` request.

Update the timing test clock to seven timestamps and assert:

```python
clock = iter([10.00, 10.01, 10.06, 10.07, 10.09, 10.10, 10.13])
assert payload["timing"]["fit_ms"] == pytest.approx(50.0)
assert payload["timing"]["summary_ms"] == pytest.approx(20.0)
assert payload["timing"]["state_ms"] == pytest.approx(30.0)
assert payload["timing"]["server_total_ms"] == pytest.approx(130.0)
```

- [ ] **Step 2: Run the envelope tests and verify the old summary-only response fails**

```bash
rtk pytest tests/test_editor.py -k "http_collapse_levels_refit or http_uncollapse_levels or reports_refit_timing" -q
```

Expected: assertions fail because the current routes return summary fields at the response root and omit `state_ms`.

- [ ] **Step 3: Add a non-serializing term-selection helper**

Replace nested composite calls to `_set_term` with:

```python
def _select_term(self, term: str) -> None:
    if term not in self.session.terms:
        raise KeyError(f"Unknown editable term: {term!r}")
    self.selected_term = term


def _set_term(self, term: str) -> dict[str, Any]:
    with self._lock:
        self._select_term(term)
        return self._state()
```

Use `_select_term(term)` inside `_select`, `_operate`, `_drag`, `_control`, `_set_control_count`, `_collapse_levels`, `_ungroup_levels`, and `_reorder_levels`. This removes the discarded pre-mutation `_state()` build while preserving each public route response.

- [ ] **Step 4: Add one transition finalizer and use it from all three structural methods**

Add this locked helper to `EditorWidget`:

```python
def _structural_transition(
    self,
    operation: str,
    *,
    operation_start: float,
    fit_start: float,
    fit_end: float,
) -> dict[str, Any]:
    summary_start = time.perf_counter()
    summary = summary_payload(self, "in_force")
    summary_end = time.perf_counter()
    state_start = time.perf_counter()
    state = self._state()
    state_end = time.perf_counter()
    return {
        "state": state,
        "summary": summary,
        "timing": {
            "operation": operation,
            "fit_ms": _elapsed_ms(fit_start, fit_end),
            "summary_ms": _elapsed_ms(summary_start, summary_end),
            "state_ms": _elapsed_ms(state_start, state_end),
            "server_total_ms": _elapsed_ms(operation_start, state_end),
        },
    }
```

At the end of `_collapse_levels`, return `_structural_transition("collapse_levels", ...)`; at the end of `_ungroup_levels`, return `_structural_transition("ungroup_levels", ...)`; at the end of `_uncollapse_levels`, return `_structural_transition("uncollapse_levels", ...)`. Delete `_timing_payload` after its last caller is removed. Keep each fit, selection restoration, collapse-history update, summary build, and state build inside the existing `with self._lock` block.

- [ ] **Step 5: Run the structural route tests**

```bash
rtk pytest tests/test_editor.py -k "collapse_levels or ungroup or uncollapse or refit_timing" -q
```

Expected: the transition envelope and timing tests pass, including state/summary revision equality.

- [ ] **Step 6: Commit the Python transition contract**

```bash
rtk git add src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor.py
rtk git commit -m "Return structural refit transition envelopes"
```

### Task 2B: Confirm Only Structural Actions That Discard Manual History

**Files:**
- Create: `src/superglm/editor/app/views/structural_confirm.js`
- Modify: `src/superglm/editor/app/index.html`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/styles/dialogs.css`
- Modify: `src/superglm/editor/app/styles.css`
- Create: `tests/editor_frontend/structural_confirm.test.js`
- Create: `tests/editor/test_editor_refit_browser.py`

- [ ] **Step 1: Write failing pure impact-copy tests**

Test `structuralImpact(snapshot, operation)` with zero and two active history records. The zero case
returns `{requiresConfirmation:false}`. The nonzero case returns the exact history count, selected
term, selected category labels (never truncated display labels), and operation title. Include
collapse, ungroup, and restore-previous-collapse cases.

- [ ] **Step 2: Run the focused test and verify the helper is absent**

Run:

```bash
rtk node --test tests/editor_frontend/structural_confirm.test.js
```

Expected: FAIL because `views/structural_confirm.js` does not exist.

- [ ] **Step 3: Implement pure impact derivation and an accessible dialog controller**

Export `structuralImpact(snapshot, operation)` and `bindStructuralConfirm(dialog)`. Use
`snapshot.selected_term`, `snapshot.selection[term]`, `snapshot.terms[term].levels`, and
`snapshot.history.active.length`. Return full selected labels joined for display and this message:

```text
Collapse levels B, C in region? This refit clears 2 manual edit history entries.
```

The binder returns `confirm(impact): Promise<boolean>`, opens one native `<dialog>`, resolves true
only from Continue, resolves false from Cancel/Escape, and restores focus to the launching control.
It must never open when `requiresConfirmation` is false.

- [ ] **Step 4: Add one shared dialog**

Add to `index.html`:

```html
<dialog id="structuralConfirmDialog" class="structural-confirm"
  aria-labelledby="structuralConfirmTitle" aria-describedby="structuralConfirmMessage">
  <form method="dialog">
    <h2 id="structuralConfirmTitle">Confirm structural refit</h2>
    <p id="structuralConfirmMessage"></p>
    <div class="dialog-actions">
      <button value="cancel" type="submit">Cancel</button>
      <button value="confirm" class="primary" type="submit">Continue and refit</button>
    </div>
  </form>
</dialog>
```

Move existing dialog rules from `styles.css` into `styles/dialogs.css`, load that stylesheet after
`panels.css`, and add the nested asset to the package test.

- [ ] **Step 5: Gate the action before mutation state begins**

Before calling `actions.executeStructuralMutation`, derive impact from
`selectSnapshot(store.getState())`. Await the controller only when required. Cancel returns
`{ok:false, skipped:true}` without setting busy state, changing selection, or making a request.
Continue runs the normal descriptor unchanged. The dialog never creates its own model/history state.

- [ ] **Step 6: Add browser cases and commit**

Add one case with empty history that observes no dialog and one request, and one case with two manual
edits that observes the exact term/levels/count, cancels with zero requests, reopens, continues, and
observes one request.

Run:

```bash
rtk npm run check:frontend
rtk pytest tests/editor/test_editor_refit_browser.py -m browser --run-browser -k confirmation -q
```

Expected: confirmation appears only when the refit will discard active manual history.

```bash
rtk git add src/superglm/editor/app/views/structural_confirm.js src/superglm/editor/app/index.html src/superglm/editor/app/main.js src/superglm/editor/app/styles/dialogs.css src/superglm/editor/app/styles.css tests/editor_frontend/structural_confirm.test.js tests/editor/test_editor_refit_browser.py
rtk git commit -m "Confirm destructive editor refits"
```

### Task 3: Consume structural envelopes without a recovery `/state` request

**Superseded standalone draft:** The checked steps in this task predate the foundation store/action
plan. Do not execute them; execute Task 3A immediately below instead. They remain only as rationale
for the transition ordering and source extraction.

**Files:**
- Modify: `src/superglm/editor/app/summary.js:134-217`
- Modify: `src/superglm/editor/app/main.js:353-445,837-865`
- Modify: `tests/test_editor.py:4391-4413`
- Test: `tests/editor_frontend/editor_transition.test.js`
- Create: `package.json`

- [x] **Step 1: Add the native-JavaScript test command and a failing transition test**

Create `package.json`:

```json
{
  "name": "superglm-editor-dev",
  "private": true,
  "type": "module",
  "scripts": {
    "test:frontend": "node --test tests/editor_frontend/*.test.js"
  }
}
```

Create `tests/editor_frontend/editor_transition.test.js`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { applyStructuralTransition } from "../../src/superglm/editor/app/state/actions.js";

test("structural transition commits state and summary before evidence", async () => {
  const calls = [];
  const envelope = {
    state: { model_revision: 7, terms: {}, selection: {}, history: {} },
    summary: { available: true, source: "in_force" },
    timing: { fit_ms: 10, state_ms: 2, summary_ms: 1 }
  };
  await applyStructuralTransition(envelope, {
    commitPrimary(value) { calls.push(["commit", value]); },
    waitForPaint() { calls.push(["paint"]); return Promise.resolve(); },
    endBlocking() { calls.push(["unblock"]); },
    startEvidence(revision) { calls.push(["evidence", revision]); }
  });

  assert.deepEqual(calls.map(([name]) => name), ["commit", "paint", "unblock", "evidence"]);
  assert.equal(calls[0][1].state.model_revision, 7);
  assert.equal(calls[0][1].summary.source, "in_force");
});
```

- [x] **Step 2: Run the frontend test and confirm the missing module**

```bash
rtk npm run test:frontend
```

Expected: failure reports that `src/superglm/editor/app/state/actions.js` does not exist.

- [x] **Step 3: Add the transition action helper**

Create `src/superglm/editor/app/state/actions.js`:

```javascript
export async function applyStructuralTransition(envelope, effects) {
  if (!envelope || !envelope.state || !envelope.summary) {
    throw new Error("Structural refit response is missing state or summary.");
  }
  effects.commitPrimary({
    state: envelope.state,
    summary: envelope.summary,
    timing: envelope.timing || {}
  });
  await effects.waitForPaint();
  effects.endBlocking();
  effects.startEvidence(envelope.state.model_revision);
}


export function waitForPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}
```

- [x] **Step 4: Make the summary module return data and export rendering**

Export `renderSummary`, and make each structural request function stop calling it:

```javascript
async function requestStructuralRefit(nodes, path, body, button, status) {
  const { summaryStatus, summaryFrame } = nodes;
  summaryStatus.textContent = status;
  summaryFrame.setAttribute("aria-busy", "true");
  if (button) button.disabled = true;
  try {
    return await requestJSON(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
    if (button) button.disabled = false;
  }
}

export function runCollapseRefit(nodes, termName) {
  return requestStructuralRefit(
    nodes,
    "/collapse_levels",
    { term: termName, method: "auto" },
    nodes.collapseLevels,
    "Refitting collapsed levels..."
  );
}

export function runUngroupRefit(nodes, termName) {
  return requestStructuralRefit(
    nodes,
    "/ungroup_levels",
    { term: termName, method: "auto" },
    nodes.ungroupLevels,
    "Refitting ungrouped levels..."
  );
}

export function runUncollapseRefit(nodes) {
  return requestStructuralRefit(
    nodes,
    "/uncollapse_levels",
    {},
    nodes.uncollapseLevels,
    "Restoring previous collapsed-level model..."
  );
}

export function renderSummary(payload, nodes) {
  // Retain the existing renderSummary body unchanged.
}
```

When applying this replacement, copy the current `renderSummary` body exactly; only its `export` changes.

- [x] **Step 5: Commit envelope state and summary synchronously in `main.js`**

Import `applyStructuralTransition`, `waitForPaint`, and exported `renderSummary`. Replace `runStructuralRefit` with:

```javascript
async function runStructuralRefit(label, action) {
  const operationStart = performance.now();
  setAppBusy(true, label, "Starting...");
  let overlayEnded = false;
  try {
    const requestStart = performance.now();
    const envelope = await action();
    const requestEnd = performance.now();
    await applyStructuralTransition(envelope, {
      commitPrimary(primary) {
        state = primary.state;
        render();
        renderSummary(primary.summary, summaryNodes());
      },
      waitForPaint,
      endBlocking() {
        setAppBusy(false);
        overlayEnded = true;
      },
      async startEvidence() {
        // Keep the old blocking refresh semantics until Task 8 wires the off-lock worker.
        await refreshMetricsView();
        await refreshActiveReport();
      }
    });
    const completed = performance.now();
    const timing = debugTiming(
      envelope,
      operationStart,
      requestStart,
      requestEnd,
      completed
    );
    showTimingStatus(envelope.summary, timing);
    return envelope;
  } finally {
    if (!overlayEnded) setAppBusy(false);
  }
}
```

For this intermediate commit, call `setAppBusy(false)` only after the existing awaited evidence calls complete: move `endBlocking()` after them inside `startEvidence`. Task 8 will use the ordering in the code block after off-lock execution is verified. Delete `state = await requestJSON("/state")` from the structural path.

- [x] **Step 6: Update the source characterization and run frontend/Python tests**

Update `test_editor_structural_refits_show_busy_overlay_and_timing_debug` to assert that `main.js` imports `applyStructuralTransition`, accesses `envelope.state`, accesses `envelope.summary`, and does not contain a `/state` request inside `runStructuralRefit`.

Run:

```bash
rtk npm run test:frontend
rtk pytest tests/test_editor.py -k "structural_refits_show_busy or collapse_levels or uncollapse" -q
```

Expected: all selected tests pass and the structural path has one HTTP mutation request.

- [x] **Step 7: Commit frontend envelope compatibility**

```bash
rtk git add package.json tests/editor_frontend/editor_transition.test.js src/superglm/editor/app/state/actions.js src/superglm/editor/app/summary.js src/superglm/editor/app/main.js tests/test_editor.py
rtk git commit -m "Commit structural refit envelopes atomically"
```

### Task 3A: Extend the Foundation Store for Atomic Structural Envelopes

**Files:**
- Modify: `src/superglm/editor/app/state/store.js`
- Modify: `src/superglm/editor/app/state/actions.js`
- Modify: `src/superglm/editor/app/summary.js`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `tests/editor_frontend/store.test.js`
- Modify: `tests/editor_frontend/actions.test.js`
- Modify: `tests/test_editor.py`

- [ ] **Step 1: Write failing store/action ordering tests**

Extend the foundation tests with one envelope fixture and this required intermediate ordering. The
overlay intentionally remains blocking through evidence until Task 8 proves scoring is off-lock:

```javascript
const envelope = {
  state: snapshot(7),
  summary: { available: true, source: "in_force" },
  timing: { fit_ms: 10, state_ms: 2, summary_ms: 1 }
};

test("structural envelope commits primary state once without GET state", async () => {
  const calls = [];
  const store = createEditorStore(createInitialEditorState(snapshot(6)));
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => { calls.push("post"); return envelope; },
      getState: async () => { throw new Error("GET state must not run"); }
    },
    waitForPaint: async () => { calls.push("paint"); }
  });
  store.subscribe(
    state => state.remote.snapshot?.model_revision,
    () => calls.push("commit")
  );
  const result = await actions.executeStructuralMutation({
    name: "collapse levels",
    path: "/collapse_levels",
    payload: { term: "region", method: "auto" },
    waitForSecondary: async () => { calls.push("evidence"); }
  });
  calls.push(store.getState().request.mutation.status);
  assert.equal(result.ok, true);
  assert.equal(store.getState().remote.summary.source, "in_force");
  assert.deepEqual(calls, ["post", "commit", "paint", "evidence", "idle"]);
});
```

- [ ] **Step 2: Run the focused frontend tests and verify failure**

Run:

```bash
rtk node --test tests/editor_frontend/store.test.js tests/editor_frontend/actions.test.js
```

Expected: FAIL because `remote.summary` and `executeStructuralMutation()` do not exist.

- [ ] **Step 3: Extend the store with one atomic transition reducer**

Initialize `remote` as `{snapshot, summary: null}` and add:

```javascript
/**
 * @param {EditorState} state
 * @param {{state: EditorSnapshot, summary: unknown, timing?: Record<string, number>}} envelope
 */
export function commitStructuralTransition(state, envelope) {
  const committed = commitRemote(state, envelope.state);
  return {
    ...committed,
    remote: { snapshot: envelope.state, summary: envelope.summary },
    request: {
      ...committed.request,
      mutation: { ...committed.request.mutation, status: "running", error: null }
    }
  };
}
```

Update `commitRemote()` to preserve `state.remote.summary`; ordinary mutations must not erase the
last confirmed summary while its refresh is pending.

- [ ] **Step 4: Extend the existing action controller**

Add an injected `waitForPaint` option to `createEditorActions`, defaulting to the double-frame helper
below. Factor the Phase 1 mutation-failure branch into a private `recoverMutation(error, descriptor,
confirmed)` and call it from both ordinary and structural mutations.

```javascript
export function nextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

async function executeStructuralMutation({
  name, path, payload, waitForSecondary = async () => {}
}) {
  if (store.getState().request.mutation.status === "running") {
    return { ok: false, skipped: true, error: new Error("A mutation is already running.") };
  }
  const confirmed = store.getState().remote;
  const descriptor = { name, path, payload };
  store.update((state) => ({
    ...state,
    request: {
      ...state.request,
      mutation: { status: "running", operation: name, error: null }
    }
  }));
  try {
    const envelope = await client.postJSON(path, payload);
    if (!envelope?.state || !envelope?.summary) {
      throw new Error("Structural refit response is missing state or summary.");
    }
    store.update((state) => commitStructuralTransition(state, envelope));
    await waitForPaintImpl();
    await waitForSecondary(envelope.state.model_revision);
    store.update((state) => ({
      ...state,
      request: {
        ...state.request,
        mutation: { status: "idle", operation: null, error: null }
      }
    }));
    return { ok: true, envelope };
  } catch (error) {
    return recoverMutation(error, descriptor, confirmed);
  }
}
```

Export `executeStructuralMutation` on the returned controller object. In the factory, bind
`const waitForPaintImpl = options.waitForPaint || nextPaint`; type the envelope/descriptor in
`api/contracts.js` rather than using `any`.

- [ ] **Step 5: Separate structural requests from summary rendering**

Export the existing `renderSummary(payload, nodes)` function. Delete network calls from
`runCollapseRefit`, `runUngroupRefit`, and `runUncollapseRefit`; replace them with these pure
descriptors:

```javascript
export const collapseTransition = (term) => ({
  name: "collapse levels",
  path: "/collapse_levels",
  payload: { term, method: "auto" }
});
export const ungroupTransition = (term) => ({
  name: "ungroup levels",
  path: "/ungroup_levels",
  payload: { term, method: "auto" }
});
export const uncollapseTransition = () => ({
  name: "restore collapsed levels",
  path: "/uncollapse_levels",
  payload: {}
});
```

- [ ] **Step 6: Commit state and summary through store subscriptions**

Subscribe chart/term/history rendering to `remote.snapshot` and summary rendering to
`remote.summary`. Replace `runStructuralRefit(label, action)` with:

```javascript
async function runStructuralRefit(descriptor) {
  const result = await actions.executeStructuralMutation({
    ...descriptor,
    waitForSecondary: async () => {
      await refreshMetricsView();
      await refreshActiveReport();
    }
  });
  if (result.ok) showTimingStatus(result.envelope.summary, result.envelope.timing);
  return result;
}
```

Call it with `collapseTransition(selectedTerm())`, `ungroupTransition(selectedTerm())`, or
`uncollapseTransition()`. Render the busy overlay solely from `request.mutation.status`; remove the
structural follow-up `requestJSON("/state")` and direct `state = ...` assignment. Task 9 changes
`waitForSecondary` to non-blocking only after Task 8's lock test passes.

- [ ] **Step 7: Run transition tests and commit**

Run:

```bash
rtk npm run check:frontend
rtk pytest tests/test_editor.py -k "structural_refits_show_busy or collapse_levels or uncollapse" -q
```

Expected: all selected tests pass, structural actions issue one mutation request, state/summary share
one revision, and no recovery `/state` request occurs on success.

```bash
rtk git add src/superglm/editor/app/state/store.js src/superglm/editor/app/state/actions.js src/superglm/editor/app/summary.js src/superglm/editor/app/main.js tests/editor_frontend/store.test.js tests/editor_frontend/actions.test.js tests/test_editor.py
rtk git commit -m "Commit structural refit envelopes atomically"
```

### Task 4: Retain one row-safe materialized edited model per edit epoch

**Files:**
- Modify: `src/superglm/editor/apply.py:28-59,241-350`
- Modify: `src/superglm/editor/session.py:489-518`
- Modify: `src/superglm/editor/summaries.py:84-95`
- Modify: `src/superglm/editor/persistence.py:19-44`
- Test: `tests/test_editor.py`

- [ ] **Step 1: Add failing materialization and memory-policy tests**

```python
def test_materialized_edit_model_is_reused_until_edit_epoch_changes(editor_model, monkeypatch):
    import superglm.editor.session as session_module

    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["B"])
    session.shift("region", np.log(1.05))
    calls = 0
    original = session_module.apply_edits_to_model_copy_with_data

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(session_module, "apply_edits_to_model_copy_with_data", counted)
    first = session.materialized_model()
    second = session.materialized_model()
    assert first is second
    assert calls == 1

    session.shift("region", np.log(1.01))
    third = session.materialized_model()
    assert third is not first
    assert calls == 2


def test_editor_model_copy_shares_row_scale_fit_inputs_and_design(editor_model):
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_levels("region", ["B"])
    session.shift("region", np.log(1.05))

    edited = session.materialized_model()

    assert edited is not session.model
    assert edited.result is not session.model.result
    assert edited._dm is session.model._dm
    assert edited._fit_X_ref is session.model._fit_X_ref
    assert edited._fit_y_ref is session.model._fit_y_ref
    assert edited._fit_sample_weight_ref is session.model._fit_sample_weight_ref
    assert edited._fit_offset_ref is session.model._fit_offset_ref
```

- [ ] **Step 2: Run the tests and verify fresh copies are still built**

```bash
rtk pytest tests/test_editor.py -k "materialized_edit_model or shares_row_scale" -q
```

Expected: failures report that `materialized_model` is missing.

- [ ] **Step 3: Add a row-safe fitted-model copy helper**

In `apply.py`, replace the raw `copy.deepcopy(model)` call with:

```python
_SHARED_ROW_SCALE_ATTRS = (
    "_dm",
    "_fit_X_ref",
    "_fit_y_ref",
    "_fit_sample_weight_ref",
    "_fit_offset_ref",
    "_fit_weights",
    "_fit_offset",
    "_fit_mu",
    "_fit_null_mu",
    "_fit_metrics_cache",
    "_fit_stats",
    "_prediction_plan",
    "_runtime_canonical_state",
    "_fast_prediction_state",
)


def _copy_model_for_editor_edits(model):
    memo: dict[int, object] = {}
    for name in _SHARED_ROW_SCALE_ATTRS:
        value = getattr(model, name, None)
        if value is not None:
            memo[id(value)] = value
    edited_model = copy.deepcopy(model, memo)
    for name in _SHARED_ROW_SCALE_ATTRS:
        if hasattr(model, name):
            setattr(edited_model, name, getattr(model, name))
    return edited_model
```

Use `edited_model = _copy_model_for_editor_edits(model)` in `apply_edits_to_model_copy_with_data`. Keep `_result`, `_solver_result`, feature specs, interaction specs, penalties, and coefficient arrays private to the copy. `_invalidate_model_caches` already drops prediction, summary, and fitted-value caches before refreshed statistics are installed.

- [ ] **Step 4: Add the internal materialization API without changing `to_model` semantics**

Move the apply import to module scope in `session.py`, then add:

```python
def materialized_model(self):
    """Return the private in-force model for the current edit epoch."""
    if not self.edited_terms():
        return self.model
    if (
        self._materialized_edit_model is not None
        and self._materialized_edit_epoch == self._edit_epoch
    ):
        return self._materialized_edit_model

    from superglm.editor.evaluation import default_metrics_dataset

    dataset = default_metrics_dataset(self)
    kwargs: dict[str, Any] = {}
    if dataset is not None:
        kwargs = {
            "X": dataset.X,
            "y": dataset.y,
            "sample_weight": dataset.sample_weight,
            "offset": dataset.offset,
        }
    model = apply_edits_to_model_copy_with_data(self.model, self.terms, **kwargs)
    self._materialized_edit_model = model
    self._materialized_edit_epoch = self._edit_epoch
    return model
```

Keep `to_model()` returning a fresh copy on every call because it is a public API with existing copy-isolation tests. Replace `_in_force_summary_model`'s edited branch with `return session.materialized_model()`. Make `edited_model_for_export` return `session.materialized_model()`; `joblib.dump` only reads the private materialization. Reports and metrics switch in Tasks 5 and 6.

- [ ] **Step 5: Run copy, summary, persistence, and materialization tests**

```bash
rtk pytest tests/test_editor.py -k "to_model or materialized or in_force_summary or save_model or download_model" -q
```

Expected: public `to_model` remains isolated, materialization is reused once per epoch, and row-scale inputs/design are shared by identity.

- [ ] **Step 6: Commit materialized-model reuse**

```bash
rtk git add src/superglm/editor/apply.py src/superglm/editor/session.py src/superglm/editor/summaries.py src/superglm/editor/persistence.py tests/test_editor.py
rtk git commit -m "Reuse one editor model per edit epoch"
```

### Task 5: Add the scalar-only EvaluationCache and pure fit-artifact reuse

**Files:**
- Create: `src/superglm/editor/evaluation_cache.py`
- Modify: `src/superglm/editor/metrics.py:23-112`
- Test: `tests/test_editor_evaluation_cache.py`

- [ ] **Step 1: Write failing scalar-cache and fit-artifact tests**

Create `tests/test_editor_evaluation_cache.py`:

```python
from __future__ import annotations

import numpy as np

from superglm.editor.evaluation import EvaluationDataset
from superglm.editor.evaluation_cache import EvaluationCache, EvaluationKey
from superglm.editor.metrics import compute_dataset_metrics


def test_evaluation_cache_deduplicates_original_and_bounds_current_revisions():
    cache = EvaluationCache()
    original = EvaluationKey("original", 0, 0, "validation", ("gaussian", "identity", 1.0))
    current_1 = EvaluationKey("current", 1, 0, "validation", ("gaussian", "identity", 1.0))
    current_2 = EvaluationKey("current", 2, 0, "validation", ("gaussian", "identity", 1.0))
    values = {"deviance": 1.0, "aic": 2.0}

    cache.put(original, values)
    cache.advance_current_revision(1)
    cache.put(current_1, values)
    cache.advance_current_revision(2)
    cache.put(current_2, {"deviance": 3.0, "aic": 4.0})

    assert cache.get(original) == values
    assert cache.get(current_1) is None
    assert cache.get(current_2)["deviance"] == 3.0
    assert cache.persistent_values_are_scalar() is True


def test_matching_fit_dataset_uses_fit_artifacts_without_predict(editor_model, editor_frame, monkeypatch):
    X, y = editor_frame
    dataset = EvaluationDataset("train", "Train", X, y, source="supplied")
    editor_model._fit_X_ref = X
    editor_model._fit_y_ref = y
    monkeypatch.setattr(
        editor_model,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("predict called")),
    )

    metrics = compute_dataset_metrics(editor_model, dataset)

    assert metrics["deviance"] == editor_model.result.deviance
    assert metrics["log_likelihood"] == editor_model._fit_stats.log_likelihood
```

- [ ] **Step 2: Run the focused tests and confirm the module is missing**

```bash
rtk pytest tests/test_editor_evaluation_cache.py -q
```

Expected: collection fails because `superglm.editor.evaluation_cache` does not exist.

- [ ] **Step 3: Implement immutable keys and the bounded scalar cache**

Create `src/superglm/editor/evaluation_cache.py`:

```python
"""Bounded scalar evaluation cache for one editor widget."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class EvaluationKey:
    role: Literal["original", "current"]
    model_revision: int
    dataset_epoch: int
    split: str
    metric_signature: tuple[Any, ...]


def model_metric_signature(model) -> tuple[Any, ...]:
    family = model._distribution
    link = model._link
    return (
        type(family).__module__,
        type(family).__qualname__,
        getattr(family, "p", None),
        getattr(family, "theta", None),
        type(link).__module__,
        type(link).__qualname__,
        float(model.result.phi),
        float(model.result.effective_df),
    )


class EvaluationCache:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._original: dict[EvaluationKey, dict[str, float]] = {}
        self._current: dict[EvaluationKey, dict[str, float]] = {}
        self._current_revision: int | None = None

    def advance_current_revision(self, revision: int) -> None:
        with self._lock:
            if self._current_revision != int(revision):
                self._current.clear()
                self._current_revision = int(revision)

    def get(self, key: EvaluationKey) -> dict[str, float] | None:
        with self._lock:
            payload = (self._original if key.role == "original" else self._current).get(key)
            return None if payload is None else dict(payload)

    def put(self, key: EvaluationKey, values: dict[str, float]) -> bool:
        payload = {str(name): float(value) for name, value in values.items()}
        with self._lock:
            if key.role == "current" and key.model_revision != self._current_revision:
                return False
            target = self._original if key.role == "original" else self._current
            target[key] = payload
            return True

    def persistent_values_are_scalar(self) -> bool:
        with self._lock:
            return all(
                isinstance(value, float) and np.ndim(value) == 0
                for mapping in (self._original, self._current)
                for payload in mapping.values()
                for value in payload.values()
            )
```

- [ ] **Step 4: Add a pure fit-artifact path before exact prediction**

In `metrics.py`, add:

```python
def _same_fit_dataset(model, dataset: EvaluationDataset) -> bool:
    fit_weight_ref = getattr(model, "_fit_sample_weight_ref", None)
    fit_weights = getattr(model, "_fit_weights", None)
    fit_offset_ref = getattr(model, "_fit_offset_ref", None)
    fit_offset = getattr(model, "_fit_offset", None)
    weights_match = (
        dataset.sample_weight is fit_weight_ref or dataset.sample_weight is fit_weights
    )
    offset_matches = dataset.offset is fit_offset_ref or dataset.offset is fit_offset
    return (
        dataset.X is getattr(model, "_fit_X_ref", None)
        and dataset.y is getattr(model, "_fit_y_ref", None)
        and weights_match
        and offset_matches
    )


def _fit_artifact_metrics(model, dataset: EvaluationDataset) -> dict[str, float] | None:
    fit_stats = getattr(model, "_fit_stats", None)
    if fit_stats is None or not _same_fit_dataset(model, dataset):
        return None
    edf = float(model.result.effective_df)
    n = dataset.n_obs
    log_likelihood = float(fit_stats.log_likelihood)
    aic = float(-2.0 * log_likelihood + 2.0 * edf)
    bic = float(-2.0 * log_likelihood + np.log(max(n, 1)) * edf)
    denom = n - edf - 1.0
    return {
        "deviance": float(model.result.deviance),
        "aic": aic,
        "aicc": float(aic + 2.0 * edf * (edf + 1.0) / denom) if denom > 0 else float("inf"),
        "bic": bic,
        "log_likelihood": log_likelihood,
        "explained_deviance": float(fit_stats.explained_deviance),
        "pearson_chi2": float(fit_stats.pearson_chi2),
        "effective_df": edf,
    }
```

Make `compute_dataset_metrics` return `_fit_artifact_metrics(model, dataset)` when non-`None`; otherwise retain the existing exact float64 scoring path. This reads fitted artifacts without calling `model.metrics()`, so background evaluation does not mutate the fitted model's internal cache.

- [ ] **Step 5: Run cache, metrics, offset, and column-vector tests**

```bash
rtk pytest tests/test_editor_evaluation_cache.py tests/test_editor.py -k "metrics or dataset_metrics" -q
```

Expected: fit-identity tests do not call `predict`, non-fit validation/test data still score exactly, and offset/null-deviance coverage passes.

- [ ] **Step 6: Commit scalar caching primitives**

```bash
rtk git add src/superglm/editor/evaluation_cache.py src/superglm/editor/metrics.py tests/test_editor_evaluation_cache.py
rtk git commit -m "Add bounded editor evaluation cache"
```

### Task 6: Make metrics and reports share cached split dictionaries

**Files:**
- Modify: `src/superglm/editor/metrics.py:23-61`
- Modify: `src/superglm/editor/reports.py:22-86`
- Modify: `src/superglm/editor/widget.py:39-59,231-245`
- Test: `tests/test_editor.py:3203-3265,3445-3487`

- [ ] **Step 1: Add a failing call-count test for live metrics followed by report**

```python
def test_live_metrics_and_report_share_validation_scalars(editor_model, editor_frame, monkeypatch):
    import superglm.editor.metrics as metrics_module

    X, y = editor_frame
    session = EditorSession.from_model(
        editor_model,
        terms=["x_spline"],
        train_data=(X.iloc[:300], y[:300], None),
        validation_data=(X.iloc[300:380], y[300:380], None),
        test_data=(X.iloc[380:], y[380:], None),
    )
    widget = session.widget()
    calls: list[tuple[int, str]] = []
    original = metrics_module.compute_dataset_metrics

    def counted(model, dataset):
        calls.append((id(model.result), dataset.name))
        return original(model, dataset)

    monkeypatch.setattr(metrics_module, "compute_dataset_metrics", counted)
    try:
        widget._metrics("deviance", "in_force")
        widget._report("validation")
    finally:
        widget.close()

    validation_calls = [call for call in calls if call[1] == "validation"]
    assert len(validation_calls) == 2
```

The two expected calls are one immutable original model and one current model; report reuse must add zero validation calls.

- [ ] **Step 2: Run the test and verify report rescoring**

```bash
rtk pytest tests/test_editor.py -k "share_validation_scalars" -q
```

Expected: four validation calls are recorded because metrics and report each score both models.

- [ ] **Step 3: Separate payload assembly from metric calculation**

In `metrics.py`, add:

```python
def metric_comparison_payload(
    metric: str,
    dataset: EvaluationDataset,
    original_metrics: dict[str, float],
    edited_metrics: dict[str, float],
    *,
    model_revision: int,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    selected_metric = metric if metric in METRIC_LABELS else "deviance"
    original = original_metrics[selected_metric]
    edited = edited_metrics[selected_metric]
    return {
        "available": True,
        "model_revision": int(model_revision),
        "request_sequence": request_sequence,
        "metric": selected_metric,
        "label": METRIC_LABELS[selected_metric],
        "dataset": dataset.name,
        "dataset_label": dataset.label,
        "n_obs": dataset.n_obs,
        "original": original,
        "edited": edited,
        "delta": edited - original,
        "metrics": {"original": original_metrics, "edited": edited_metrics},
    }
```

Retain `metrics_payload` as a compatibility wrapper for direct callers, but make `EditorWidget` use the new assembler with cached values.

- [ ] **Step 4: Add one cache lookup helper to `EditorWidget`**

Initialize `self._evaluation_cache = EvaluationCache()` before starting the server. Add:

```python
def _cached_dataset_metrics(self, role: str, model, dataset) -> dict[str, float]:
    revision = 0 if role == "original" else self.session.model_revision
    self._evaluation_cache.advance_current_revision(self.session.model_revision)
    key = EvaluationKey(
        role=role,
        model_revision=revision,
        dataset_epoch=0,
        split=dataset.name,
        metric_signature=model_metric_signature(model),
    )
    cached = self._evaluation_cache.get(key)
    if cached is not None:
        return cached
    values = compute_dataset_metrics(model, dataset)
    self._evaluation_cache.put(key, values)
    return values
```

Update `_metrics` under the existing lock to resolve `named_metrics_dataset`, use `reference_model`, use `session.materialized_model()` for current, call `_cached_dataset_metrics` twice, and assemble `metric_comparison_payload` with the current revision.

- [ ] **Step 5: Refactor report assembly to accept cached pairs**

In `reports.py`, replace `_split_metrics(session)` with:

```python
def split_metrics_payload(
    datasets,
    metric_pairs: dict[str, tuple[dict[str, float], dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        original, edited = metric_pairs[dataset.name]
        rows.append(
            {
                "name": dataset.name,
                "label": dataset.label,
                "n_obs": dataset.n_obs,
                "source": dataset.source,
                "metrics": {
                    "original": {metric: original[metric] for metric in _REPORT_METRICS},
                    "edited": {metric: edited[metric] for metric in _REPORT_METRICS},
                    "delta": {
                        metric: edited[metric] - original[metric]
                        for metric in _REPORT_METRICS
                    },
                },
            }
        )
    return rows
```

Change `validation_report_payload` and `final_fit_report_payload` to accept a prepared `splits` list. In `EditorWidget._report`, resolve the datasets and materialized model once, build `metric_pairs` through `_cached_dataset_metrics`, call `split_metrics_payload`, and then call the appropriate report assembler. Include `model_revision` in both report payloads.

- [ ] **Step 6: Run report reuse and existing HTTP report tests**

```bash
rtk pytest tests/test_editor.py -k "share_validation_scalars or reports_are_display_only or report_sanitizes or accepts_plain_evaluation" -q
```

Expected: the validation pair is scored twice total across live metrics plus report, and all report shapes remain unchanged apart from the added revision.

- [ ] **Step 7: Commit shared metric/report scalars**

```bash
rtk git add src/superglm/editor/metrics.py src/superglm/editor/reports.py src/superglm/editor/widget.py tests/test_editor.py
rtk git commit -m "Share editor metrics across evidence views"
```

### Task 7: Add a one-worker EvidenceCoordinator with latest-pending coalescing

**Files:**
- Create: `src/superglm/editor/evidence.py`
- Create: `tests/test_editor_evidence.py`

- [ ] **Step 1: Write deterministic concurrency tests**

Create `tests/test_editor_evidence.py`:

```python
from __future__ import annotations

import threading

from superglm.editor.evidence import EvidenceCoordinator, EvidenceKey


def test_same_evidence_key_shares_one_computation():
    coordinator = EvidenceCoordinator("dedup")
    release = threading.Event()
    started = threading.Event()
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=5)
        return {"value": 1}

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), compute)
        assert started.wait(timeout=5)
        second = coordinator.submit(EvidenceKey(1, "metrics", "validation"), compute)
        assert first is second
        release.set()
        assert first.result(timeout=5).payload == {"value": 1}
        assert calls == 1
    finally:
        release.set()
        coordinator.close()


def test_latest_pending_revision_replaces_intermediate_work():
    coordinator = EvidenceCoordinator("coalesce")
    release = threading.Event()
    started = threading.Event()
    calls: list[int] = []

    def compute(revision):
        def run():
            calls.append(revision)
            if revision == 1:
                started.set()
                release.wait(timeout=5)
            return {"revision": revision}
        return run

    try:
        first = coordinator.submit(EvidenceKey(1, "metrics", "validation"), compute(1))
        assert started.wait(timeout=5)
        second = coordinator.submit(EvidenceKey(2, "metrics", "validation"), compute(2))
        third = coordinator.submit(EvidenceKey(3, "metrics", "validation"), compute(3))
        assert second.result(timeout=5).status == "superseded"
        release.set()
        assert first.result(timeout=5).status == "complete"
        assert third.result(timeout=5).payload == {"revision": 3}
        assert calls == [1, 3]
        assert coordinator.max_active == 1
    finally:
        release.set()
        coordinator.close()
```

- [ ] **Step 2: Run the tests and confirm the coordinator module is absent**

```bash
rtk pytest tests/test_editor_evidence.py -q
```

Expected: collection fails because `superglm.editor.evidence` does not exist.

- [ ] **Step 3: Implement one running and one latest pending work item**

Create `src/superglm/editor/evidence.py`:

```python
"""Bounded background execution for editor evidence cache misses."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class EvidenceKey:
    model_revision: int
    kind: str
    discriminator: str


@dataclass(frozen=True)
class EvidenceOutcome:
    status: Literal["complete", "superseded"]
    key: EvidenceKey
    payload: dict[str, Any] | None = None


@dataclass
class _WorkItem:
    key: EvidenceKey
    compute: Callable[[], dict[str, Any]]
    future: Future[EvidenceOutcome]


class EvidenceCoordinator:
    def __init__(self, name: str) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"superglm-{name}")
        self._running: _WorkItem | None = None
        self._pending: _WorkItem | None = None
        self._closed = False
        self._active = 0
        self.max_active = 0

    def submit(
        self,
        key: EvidenceKey,
        compute: Callable[[], dict[str, Any]],
    ) -> Future[EvidenceOutcome]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Evidence coordinator is closed.")
            if self._running is not None and self._running.key == key:
                return self._running.future
            if self._pending is not None and self._pending.key == key:
                return self._pending.future
            item = _WorkItem(key, compute, Future())
            if self._running is None:
                self._start_locked(item)
            else:
                if self._pending is not None and not self._pending.future.done():
                    self._pending.future.set_result(
                        EvidenceOutcome("superseded", self._pending.key)
                    )
                self._pending = item
            return item.future

    def _start_locked(self, item: _WorkItem) -> None:
        self._running = item
        worker = self._executor.submit(self._execute, item)
        worker.add_done_callback(lambda completed: self._finish(item, completed))

    def _execute(self, item: _WorkItem) -> EvidenceOutcome:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            return EvidenceOutcome("complete", item.key, item.compute())
        finally:
            with self._lock:
                self._active -= 1

    def _finish(self, item: _WorkItem, worker: Future[EvidenceOutcome]) -> None:
        with self._lock:
            try:
                outcome = worker.result()
            except BaseException as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            else:
                if not item.future.done():
                    item.future.set_result(outcome)
            if self._running is item:
                self._running = None
            pending = self._pending
            self._pending = None
            if pending is not None and not self._closed:
                self._start_locked(pending)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._pending is not None and not self._pending.future.done():
                self._pending.future.set_result(
                    EvidenceOutcome("superseded", self._pending.key)
                )
            self._pending = None
        self._executor.shutdown(wait=False, cancel_futures=True)
```

- [ ] **Step 4: Run the deterministic concurrency tests**

```bash
rtk pytest tests/test_editor_evidence.py -q
```

Expected: same-key work runs once, revision 2 is superseded, revisions 1 and 3 run, and maximum concurrency remains one.

- [ ] **Step 5: Commit the bounded coordinator**

```bash
rtk git add src/superglm/editor/evidence.py tests/test_editor_evidence.py
rtk git commit -m "Add bounded editor evidence worker"
```

### Task 8: Capture immutable evidence snapshots and release the widget mutation lock

**Superseded lock-boundary draft:** The checked steps below correctly describe request metadata but
still call `session.materialized_model()` while holding `EditorWidget._lock`; that method performs
row-scale prediction. Do not execute them. Task 8A replaces the whole slice and is the only approved
lock-boundary implementation.

**Files:**
- Modify: `src/superglm/editor/evaluation_cache.py`
- Modify: `src/superglm/editor/widget.py:39-67,82-89,231-245`
- Modify: `src/superglm/editor/server.py:119-134`
- Test: `tests/test_editor_evidence.py`
- Test: `tests/test_editor.py`

- [x] **Step 1: Add a failing lock-release integration test**

Add to `tests/test_editor_evidence.py`:

```python
def test_metric_cache_miss_does_not_hold_widget_mutation_lock(editor_model, monkeypatch):
    import superglm.editor.widget as widget_module

    session = EditorSession.from_model(editor_model, terms=["x_spline"])
    widget = session.widget()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = widget_module.compute_dataset_metrics

    def blocked(model, dataset):
        started.set()
        release.wait(timeout=5)
        return original(model, dataset)

    monkeypatch.setattr(widget_module, "compute_dataset_metrics", blocked)
    worker = threading.Thread(
        target=lambda: (widget._metrics("deviance", "in_force"), finished.set()),
        daemon=True,
    )
    try:
        worker.start()
        assert started.wait(timeout=5)
        before = time.perf_counter()
        widget._select("x_spline", [20, 21])
        elapsed = time.perf_counter() - before
        assert elapsed < 0.1
        assert finished.is_set() is False
    finally:
        release.set()
        worker.join(timeout=5)
        widget.close()
```

Import `time`, `EditorSession`, and the existing fixtures at the top of the test file.

- [x] **Step 2: Run the test and verify the lock is held during scoring**

```bash
rtk pytest tests/test_editor_evidence.py -k "does_not_hold_widget_mutation_lock" -q
```

Expected: `_select` blocks until `release` because `_metrics` currently holds `widget._lock` around scoring.

- [x] **Step 3: Add immutable captured requests**

Add to `evaluation_cache.py`:

```python
from dataclasses import dataclass

from superglm.editor.evaluation import EvaluationDataset


@dataclass(frozen=True)
class ModelSnapshot:
    role: Literal["original", "current"]
    model_revision: int
    edit_epoch: int
    base_model: Any
    materialized_model: Any | None


@dataclass(frozen=True)
class MetricsRequest:
    model_revision: int
    request_sequence: int | None
    metric: str
    dataset: EvaluationDataset
    original: ModelSnapshot
    current: ModelSnapshot


@dataclass(frozen=True)
class ReportRequest:
    model_revision: int
    request_sequence: int | None
    report: str
    datasets: tuple[EvaluationDataset, ...]
    original: ModelSnapshot
    current: ModelSnapshot
    cv_report: Any
    summary: dict[str, Any] | None
```

Capture existing model objects and evaluation objects by reference. They are immutable snapshots for editor purposes: manual edits mutate `EditableTerm` arrays, structural changes replace `session.model`, and evaluation replacement has no public API. Never retain `EditorSession` or `EditorWidget` inside these dataclasses.

- [x] **Step 4: Wire the coordinator while keeping cache hits synchronous**

Initialize in `EditorWidget.__init__`:

```python
self._evaluation_cache = EvaluationCache()
self._evidence = EvidenceCoordinator(f"editor-evidence-{id(self):x}")
```

Add `_capture_model_snapshots`, `_capture_metrics_request`, and `_capture_report_request`; each runs under `self._lock`, calls `self._evaluation_cache.advance_current_revision`, resolves the current materialized model once, and returns only frozen snapshots. Add `_compute_metrics_request` and `_compute_report_request`; these consume snapshots, call `_cached_dataset_metrics`, and never access `self.session`, `self.selected_term`, or mutable widget fields.

Refactor `_metrics` to this shape:

```python
def _metrics(
    self,
    metric: str,
    source: str | None = None,
    *,
    dataset: str | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    with self._lock:
        current_revision = self.session.model_revision
        if model_revision is not None and int(model_revision) != current_revision:
            return {
                "status": "superseded",
                "model_revision": current_revision,
                "request_sequence": request_sequence,
            }
        request = self._capture_metrics_request(
            metric,
            source,
            dataset,
            request_sequence,
        )
        cached = self._cached_metrics_request(request)
        if cached is not None:
            return cached

    key = EvidenceKey(request.model_revision, "metrics", request.dataset.name)
    outcome = self._evidence.submit(
        key,
        lambda: self._compute_metrics_request(request),
    ).result()
    if outcome.status == "superseded":
        return {
            "status": "superseded",
            "model_revision": request.model_revision,
            "request_sequence": request.request_sequence,
        }
    return outcome.payload
```

Give `_report` the same capture/cache-hit/submit/wait shape, with discriminator equal to the report name. `EvidenceCoordinator` serializes cache misses; `_cached_dataset_metrics` is thread-safe through `EvaluationCache`.

In `close`, call `self._evidence.close()` before `self._server.close()`.

- [x] **Step 5: Accept and echo browser revision metadata on existing routes**

Update `/metrics` to pass `dataset`, `model_revision`, and `request_sequence`; update `/report` to pass `model_revision` and `request_sequence`. Use `None` when the key is absent, preserving compatibility with current direct callers.

```python
lambda: widget._metrics(
    str(payload.get("metric", "deviance")),
    None if "source" not in payload else str(payload["source"]),
    dataset=None if "dataset" not in payload else str(payload["dataset"]),
    model_revision=None if "model_revision" not in payload else int(payload["model_revision"]),
    request_sequence=None if "request_sequence" not in payload else int(payload["request_sequence"]),
)
```

- [x] **Step 6: Run lock, coordinator, route, and cache tests**

```bash
rtk pytest tests/test_editor_evidence.py tests/test_editor_evaluation_cache.py tests/test_editor.py -k "metrics or report or evidence or revision" -q
```

Expected: a blocked cache miss does not block `_select`; one worker remains active; route payloads echo revision/sequence; existing metric/report payloads still serialize.

- [x] **Step 7: Commit off-lock evidence execution**

```bash
rtk git add src/superglm/editor/evaluation_cache.py src/superglm/editor/widget.py src/superglm/editor/server.py tests/test_editor_evidence.py tests/test_editor.py
rtk git commit -m "Run editor evidence outside mutation lock"
```

### Task 8A: Materialize and Score Entirely Outside the Widget Lock

**Files:**
- Modify: `src/superglm/editor/session.py`
- Modify: `src/superglm/editor/apply.py`
- Modify: `src/superglm/editor/evaluation_cache.py`
- Modify: `src/superglm/editor/widget.py`
- Modify: `src/superglm/editor/summaries.py`
- Modify: `src/superglm/editor/reports.py`
- Modify: `src/superglm/editor/persistence.py`
- Modify: `src/superglm/editor/server.py`
- Modify: `tests/test_editor_evidence.py`
- Modify: `tests/test_editor.py`

- [ ] **Step 1: Write failing tests for both expensive phases**

Add one test that blocks `apply_edits_to_model_copy_with_data` while a manual-edit metrics request is
materializing, and a second that blocks `compute_dataset_metrics` after materialization. In both
cases, start `_metrics()` on a background route thread, wait for the blocking event, call
`widget._select("x_spline", [20, 21])`, and assert it completes in under 100 ms while the evidence
thread remains unfinished. Add identity assertions proving captured datasets are the original
`EvaluationDataset.X/y/sample_weight/offset` objects.

The materialization case must patch the symbol used by `materialize_edit_request`, not
`session.materialized_model`, so it catches accidental row-scale work inside the lock.

- [ ] **Step 2: Run the lock tests and verify materialization still blocks**

Run:

```bash
rtk pytest tests/test_editor_evidence.py -k "materialization_does_not_hold or scoring_does_not_hold" -q
```

Expected: the materialization test exceeds 100 ms because Task 4 currently builds the edited model
under `widget._lock`.

- [ ] **Step 3: Capture only plot-scale edit data under the lock**

Add this immutable request in `evaluation_cache.py`:

```python
@dataclass(frozen=True)
class EditMaterializationRequest:
    model_revision: int
    edit_epoch: int
    base_model: Any
    terms: dict[str, EditableTerm]
    dataset: EvaluationDataset | None
```

Add these `EditorSession` methods. `EditableTerm.copy()` copies only plot grids, coefficient effects,
labels, and metadata; it never copies an n-row DataFrame or design matrix:

```python
def capture_materialization_request(self) -> EditMaterializationRequest | None:
    if not self.edited_terms():
        return None
    return EditMaterializationRequest(
        model_revision=self.model_revision,
        edit_epoch=self.edit_epoch,
        base_model=self.model,
        terms={name: term.copy() for name, term in self.terms.items()},
        dataset=default_metrics_dataset(self),
    )

def cached_materialized_model(self, edit_epoch: int):
    if self._materialized_edit_epoch != int(edit_epoch):
        return None
    return self._materialized_edit_model

def publish_materialized_model(self, request, model) -> bool:
    if (
        request.model_revision != self.model_revision
        or request.edit_epoch != self.edit_epoch
        or request.base_model is not self.model
    ):
        return False
    self._materialized_edit_model = model
    self._materialized_edit_epoch = request.edit_epoch
    return True
```

Move actual construction into a pure module-level function that never receives a session/widget:

```python
def materialize_edit_request(request: EditMaterializationRequest):
    kwargs: dict[str, Any] = {}
    if request.dataset is not None:
        kwargs = {
            "X": request.dataset.X,
            "y": request.dataset.y,
            "sample_weight": request.dataset.sample_weight,
            "offset": request.dataset.offset,
        }
    return apply_edits_to_model_copy_with_data(
        request.base_model,
        request.terms,
        **kwargs,
    )
```

Keep public `to_model()` fresh-copy semantics. Remove the Task 4 `materialized_model()` method so no
caller can accidentally perform materialization while it happens to own a lock.

- [ ] **Step 4: Coordinate and publish one current materialization**

Add this widget helper. Its first and final sections own `_lock`; the worker section does not:

```python
def _current_model_for_evidence(self):
    with self._lock:
        if not self.session.edited_terms():
            return self.session.model, self.session.model_revision
        epoch = self.session.edit_epoch
        cached = self.session.cached_materialized_model(epoch)
        if cached is not None:
            return cached, self.session.model_revision
        request = self.session.capture_materialization_request()
        assert request is not None

    key = EvidenceKey(request.model_revision, "materialize", "current")
    outcome = self._evidence.submit(
        key,
        lambda: {"model": materialize_edit_request(request)},
    ).result()
    if outcome.status == "superseded" or outcome.payload is None:
        return None, request.model_revision
    model = outcome.payload["model"]

    with self._lock:
        if not self.session.publish_materialized_model(request, model):
            return None, request.model_revision
        return model, request.model_revision
```

If this returns `None`, the route returns a `status: "superseded"` payload; the browser's latest-
revision refresh obtains the new model. Same-key callers share the materialization future.

- [ ] **Step 5: Capture metrics/report requests only after off-lock materialization**

For `_metrics` and `_report`, call `_current_model_for_evidence()` before the short capture lock.
Then reacquire `_lock`, verify that the returned revision still equals `session.model_revision`,
capture original/current model references plus dataset references and cache keys, and release the lock
before submitting the scalar calculation. Cache hits may be returned from inside the short lock.

The worker functions receive frozen request dataclasses and must not access `self.session`,
`self.selected_term`, widget state, or mutable term arrays. Original/current prediction arrays are
locals, reduced to scalar dictionaries, and released before the future resolves.

- [ ] **Step 6: Route summary and export through the same materialization**

Change `summary_payload(widget, source, *, model_override=None)` so manual in-force summaries use the
explicit private model supplied by the widget. Change report assemblers to accept that same model.
Change `edited_model_for_export(session, *, model_override=None)` and `save_model(...,
model_override=None)` to serialize the override when present.

Refactor `_summary`, `_save_model`, and `_download_model` to obtain the current model through
`_current_model_for_evidence()` without holding `_lock`, then reacquire only to verify the revision
and capture names/paths. Perform summary construction, joblib serialization, and filesystem I/O
outside `_lock`. Original/refit summary sources that do not require a manual materialization continue
to capture their model reference under the short lock.

- [ ] **Step 7: Close the worker and pass revision metadata through FastAPI**

Initialize one `EvaluationCache` and one `EvidenceCoordinator` before the server starts. Close the
coordinator before closing the server. Pass optional `dataset`, `model_revision`, and
`request_sequence` into `/metrics`; pass `model_revision` and `request_sequence` into `/report` and
`/summary`. Echo both fields in successful, failed, and superseded evidence payloads.

- [ ] **Step 8: Run the complete lock/cache/materialization gate and commit**

Run:

```bash
rtk pytest tests/test_editor_evidence.py tests/test_editor_evaluation_cache.py tests/test_editor.py -k "materializ or metrics or report or summary or save or evidence or revision" -q
```

Expected: both blocked materialization and blocked scoring leave `_select` responsive; same-key work
deduplicates; one latest pending revision survives; one private materialized model is reused for
summary/metrics/report/export; no captured request owns copied evaluation frames.

```bash
rtk git add src/superglm/editor/session.py src/superglm/editor/apply.py src/superglm/editor/evaluation_cache.py src/superglm/editor/widget.py src/superglm/editor/summaries.py src/superglm/editor/reports.py src/superglm/editor/persistence.py src/superglm/editor/server.py tests/test_editor_evidence.py tests/test_editor.py
rtk git commit -m "Run editor materialization and evidence off lock"
```

### Task 9: Add browser evidence freshness, debounce, and stale-response rejection

**Superseded standalone-store draft:** The checked steps below would create a second incompatible
store. Do not execute them. Task 9A extends the Phase 1 state/actions and Phase 2 panel markup.

**Files:**
- Create: `src/superglm/editor/app/state/store.js`
- Modify: `src/superglm/editor/app/state/actions.js`
- Modify: `src/superglm/editor/app/main.js:116-140,353-385,454-507,763-783,837-870`
- Modify: `src/superglm/editor/app/metrics.js:14-40`
- Modify: `src/superglm/editor/app/reports.js:13-28`
- Modify: `src/superglm/editor/app/index.html:205-257`
- Modify: `src/superglm/editor/app/styles.css`
- Test: `tests/editor_frontend/editor_store.test.js`

- [x] **Step 1: Write pure failing store and evidence-controller tests**

Create `tests/editor_frontend/editor_store.test.js`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { createEditorStore } from "../../src/superglm/editor/app/state/store.js";
import { createEvidenceController } from "../../src/superglm/editor/app/state/actions.js";

test("transition commits remote state and summary in one notification", () => {
  const store = createEditorStore();
  const seen = [];
  store.subscribe((next) => seen.push(next));
  store.commitTransition({
    state: { model_revision: 4, terms: {}, selection: {}, history: {} },
    summary: { available: true, source: "in_force" },
    timing: {}
  });
  assert.equal(seen.length, 1);
  assert.equal(seen[0].remote.modelRevision, 4);
  assert.equal(seen[0].remote.summary.source, "in_force");
});

test("late evidence and older same-revision sequence are ignored", () => {
  const store = createEditorStore();
  store.commitTransition({
    state: { model_revision: 5, terms: {}, selection: {}, history: {} },
    summary: { available: true },
    timing: {}
  });
  store.beginEvidence("metrics", 5, 2);
  assert.equal(store.completeEvidence("metrics", { model_revision: 4, request_sequence: 3 }), false);
  assert.equal(store.completeEvidence("metrics", { model_revision: 5, request_sequence: 1 }), false);
  assert.equal(store.completeEvidence("metrics", { model_revision: 5, request_sequence: 2 }), true);
});

test("ordinary evidence debounce runs only the latest revision", async () => {
  const scheduled = [];
  const timers = [];
  const controller = createEvidenceController({
    delayMs: 150,
    setTimer(callback) { timers.push(callback); return timers.length; },
    clearTimer() {},
    run(kind, revision, sequence) { scheduled.push([kind, revision, sequence]); }
  });
  controller.schedule("metrics", 1);
  controller.schedule("metrics", 2);
  timers.at(-1)();
  assert.deepEqual(scheduled, [["metrics", 2, 2]]);
});
```

- [x] **Step 2: Run pure frontend tests and confirm store APIs are missing**

```bash
rtk npm run test:frontend
```

Expected: failures report that `state/store.js` and `createEvidenceController` are missing.

- [x] **Step 3: Implement the remote/evidence store**

Create `src/superglm/editor/app/state/store.js`:

```javascript
const emptyEvidence = () => ({
  status: "idle",
  revision: -1,
  sequence: 0,
  payload: null,
  error: ""
});

export function createEditorStore() {
  let state = {
    remote: { snapshot: null, modelRevision: -1, summary: null, timing: {} },
    request: { metrics: emptyEvidence(), report: emptyEvidence() }
  };
  const listeners = new Set();
  const publish = (next) => {
    const previous = state;
    state = next;
    for (const listener of listeners) listener(state, previous);
  };
  return {
    getState() { return state; },
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    commitTransition(envelope) {
      publish({
        ...state,
        remote: {
          snapshot: envelope.state,
          modelRevision: Number(envelope.state.model_revision),
          summary: envelope.summary,
          timing: envelope.timing || {}
        }
      });
    },
    beginEvidence(kind, revision, sequence) {
      const previous = state.request[kind];
      publish({
        ...state,
        request: {
          ...state.request,
          [kind]: { ...previous, status: "updating", revision, sequence, error: "" }
        }
      });
    },
    completeEvidence(kind, payload) {
      const current = state.request[kind];
      if (Number(payload.model_revision) !== state.remote.modelRevision) return false;
      if (Number(payload.request_sequence) !== current.sequence) return false;
      publish({
        ...state,
        request: {
          ...state.request,
          [kind]: { ...current, status: "fresh", payload, error: "" }
        }
      });
      return true;
    },
    failEvidence(kind, revision, sequence, error) {
      const current = state.request[kind];
      if (revision !== state.remote.modelRevision || sequence !== current.sequence) return false;
      publish({
        ...state,
        request: {
          ...state.request,
          [kind]: { ...current, status: "stale", error: String(error) }
        }
      });
      return true;
    }
  };
}
```

- [x] **Step 4: Implement the evidence scheduler**

Append to `state/actions.js`:

```javascript
export function createEvidenceController({
  delayMs = 150,
  setTimer = window.setTimeout.bind(window),
  clearTimer = window.clearTimeout.bind(window),
  run
}) {
  let timer = null;
  let sequence = 0;
  return {
    schedule(kind, revision, { immediate = false } = {}) {
      sequence += 1;
      const requestSequence = sequence;
      if (timer !== null) clearTimer(timer);
      const invoke = () => {
        timer = null;
        run(kind, revision, requestSequence);
      };
      if (immediate) invoke();
      else timer = setTimer(invoke, delayMs);
      return requestSequence;
    },
    cancel() {
      if (timer !== null) clearTimer(timer);
      timer = null;
    }
  };
}
```

- [x] **Step 5: Make metrics and reports retain confirmed values while updating**

Change `refreshMetrics` and `refreshReport` into request functions that accept `modelRevision` and `requestSequence`, include them in JSON, and return payloads without mutating DOM. Export `renderMetricGrid` and `renderReport`. Add small exported freshness renderers which set `aria-busy`, an `Updating...` label, or a persistent stale message without clearing the previous payload.

Use these exact request bodies:

```javascript
JSON.stringify({
  metric: "deviance",
  source: "in_force",
  model_revision: modelRevision,
  request_sequence: requestSequence
})
```

and:

```javascript
JSON.stringify({ report, model_revision: modelRevision, request_sequence: requestSequence })
```

Add `#metricFreshness` and `#reportFreshness` polite live regions to `index.html`. Style `.evidence-updating` and `.evidence-stale` without replacing the existing grid/report contents.

- [x] **Step 6: Wire only the visible evidence surface and release the overlay after paint**

Make the store authoritative for the remote snapshot. `render()` reads `store.getState().remote.snapshot`; `loadState()` commits the initial `/state` response with its separately fetched initial summary. A store subscription renders the chart and summary only when `remote` changes, metrics only when `request.metrics` changes, and report only when `request.report` changes.

Create one `EvidenceController` in `main.js`. Its `run` callback:

1. calls `store.beginEvidence`;
2. requests metrics when `activeView === "editor"`, otherwise requests the active report;
3. calls `store.completeEvidence` only for a `status !== "superseded"` payload;
4. calls `store.failEvidence` on a current revision/sequence error.

Ordinary `postJSONWithRefresh` schedules evidence with the default 150ms debounce after committing its returned state. Structural `runStructuralRefit` calls `applyStructuralTransition` using:

```javascript
await applyStructuralTransition(envelope, {
  commitPrimary(primary) { store.commitTransition(primary); },
  waitForPaint,
  endBlocking() { setAppBusy(false); overlayEnded = true; },
  startEvidence(revision) {
    const kind = activeView === "editor" ? "metrics" : "report";
    evidenceController.schedule(kind, revision, { immediate: true });
  }
});
```

Do not await evidence from the mutation path. Metrics completion must not call the application-wide `render()`; it renders only the metric strip. Report completion renders only the report.

- [x] **Step 7: Run frontend tests and focused Python source/HTTP tests**

```bash
rtk npm run test:frontend
rtk pytest tests/test_editor.py -k "structural_refits_show_busy or metrics or report or serves_editor_app_assets" -q
```

Expected: pure stale/debounce tests pass, source assets are served, and Python evidence payloads include matching revision/sequence values.

- [x] **Step 8: Commit background evidence freshness**

```bash
rtk git add src/superglm/editor/app/state/store.js src/superglm/editor/app/state/actions.js src/superglm/editor/app/main.js src/superglm/editor/app/metrics.js src/superglm/editor/app/reports.js src/superglm/editor/app/index.html src/superglm/editor/app/styles.css tests/editor_frontend/editor_store.test.js
rtk git commit -m "Refresh editor evidence in the background"
```

### Task 9A: Add Freshness and Debounce to the Existing Action Controller

**Files:**
- Modify: `src/superglm/editor/app/api/contracts.js`
- Modify: `src/superglm/editor/app/state/store.js`
- Modify: `src/superglm/editor/app/state/actions.js`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/metrics.js`
- Modify: `src/superglm/editor/app/reports.js`
- Modify: `src/superglm/editor/app/index.html`
- Modify: `src/superglm/editor/app/styles/panels.css`
- Modify: `tests/editor_frontend/actions.test.js`
- Modify: `tests/test_editor.py`

- [ ] **Step 1: Write failing freshness/debounce tests against the foundation state shape**

Extend `actions.test.js` to prove: beginning a refresh retains the prior payload with status
`updating`; a matching result becomes `current`; failure becomes `stale` and retains the payload;
an older revision or same-revision sequence is ignored; three scheduled edit revisions run only the
last revision after the injected timer fires. Assert the exact status vocabulary from
`EvidenceStatus`: `idle`, `updating`, `current`, `stale`, `error`.

- [ ] **Step 2: Run the action tests and verify freshness transitions are missing**

Run:

```bash
rtk node --test tests/editor_frontend/actions.test.js
```

Expected: FAIL because the existing evidence method has no debounce and does not retain confirmed
payloads through Updating/Stale.

- [ ] **Step 3: Add pure evidence reducers without changing store ownership**

Export these reducers from `store.js` and use them only through `store.update()`:

```javascript
export function beginEvidence(state, panel, revision, sequence, retry) {
  const previous = state.request.evidence[panel];
  return replaceEvidence(state, panel, {
    ...previous,
    status: "updating",
    revision,
    sequence,
    error: null,
    retry
  });
}

export function completeEvidence(state, panel, revision, sequence, payload) {
  const current = state.request.evidence[panel];
  if (revision !== state.remote.snapshot?.model_revision || sequence !== current.sequence) {
    return state;
  }
  return replaceEvidence(state, panel, {
    ...current,
    status: "current",
    payload,
    error: null,
    retry: null
  });
}

export function failEvidence(state, panel, revision, sequence, error) {
  const current = state.request.evidence[panel];
  if (revision !== state.remote.snapshot?.model_revision || sequence !== current.sequence) {
    return state;
  }
  return replaceEvidence(state, panel, {
    ...current,
    status: current.payload === null ? "error" : "stale",
    error: String(error)
  });
}

function replaceEvidence(state, panel, evidence) {
  return {
    ...state,
    request: {
      ...state.request,
      evidence: { ...state.request.evidence, [panel]: evidence }
    }
  };
}
```

- [ ] **Step 4: Add one per-panel debouncer inside `createEditorActions`**

Inject `setTimer`/`clearTimer` (defaulting to `window.setTimeout`/`clearTimeout`) and add
`EVIDENCE_DEBOUNCE_MS = 150`. Keep a timer map and expose:

```javascript
function schedulePanelEvidence(panel, path, payload, { immediate = false } = {}) {
  const previous = evidenceTimers.get(panel);
  if (previous !== undefined) clearTimerImpl(previous);
  const invoke = () => {
    evidenceTimers.delete(panel);
    void refreshEvidence(panel, path, payload);
  };
  if (immediate) invoke();
  else evidenceTimers.set(panel, setTimerImpl(invoke, EVIDENCE_DEBOUNCE_MS));
}
```

Expose `schedulePanelEvidence` on the controller. Refactor `refreshEvidence` to allocate the next
sequence, capture the current revision, merge
`model_revision` and `request_sequence` into the request body, call `beginEvidence`, and apply
`completeEvidence` only when the response echoes both values and does not say `status:
"superseded"`. On current failure call `failEvidence`. `retryEvidence(panel)` replays the stored
`{path,payload}` immediately.

- [ ] **Step 5: Switch structural completion to paint-then-unblock-then-evidence**

Now that Task 8A proves materialization and scoring are off-lock, change
`executeStructuralMutation()` to this final order after `commitStructuralTransition`:

```javascript
await waitForPaintImpl();
store.update((state) => ({
  ...state,
  request: {
    ...state.request,
    mutation: { status: "idle", operation: null, error: null }
  }
}));
scheduleVisibleEvidence(envelope.state.model_revision, {
  immediate: true,
  summaryCommitted: true
});
```

`scheduleVisibleEvidence` is the callback injected into `createEditorActions`. In `main.js`, an
ordinary edit schedules metrics plus summary when the Summary inspector pane is visible, or the
active report when a report view is visible. A structural envelope already contains its summary, so
its immediate callback schedules metrics or the active report only. All requests delegate to
`actions.schedulePanelEvidence(...)`; the Python coordinator serializes cache misses. Remove
`waitForSecondary` from the structural public
descriptor. Ordinary prediction-changing mutations call
`scheduleVisibleEvidence(newRevision)` with the 150 ms debounce; selection/term/zoom/display-only
changes have an unchanged revision and schedule nothing.

- [ ] **Step 6: Render only the panel whose evidence state changed**

Keep chart/term/history subscriptions keyed to `remote.snapshot`. Summary rendering selects the
current `request.evidence.summary.payload` when present and matching, otherwise `remote.summary` from
the last structural transition. Metric strip is keyed to `request.evidence.metrics`, and reports are
keyed to `request.evidence.report`. Evidence completion must not call the application-wide
`render()` or `drawChart()`.

`renderSummary`, `renderMetricGrid`, and `renderReport` retain the last payload while status is
`updating`, `stale`, or `error`. Set `data-freshness` and `aria-busy`; render a Retry button only for
stale/error. Reuse `#summaryStatus` and add polite `#metricFreshness`/`#reportFreshness` live regions,
but keep mutation failures in the assertive `#appAlert`.

- [ ] **Step 7: Run frontend/Python evidence tests and commit**

Run:

```bash
rtk npm run check:frontend
rtk pytest tests/test_editor.py -k "metrics or report or structural_refits_show_busy or serves_editor_app_assets" -q
```

Expected: stale/debounce tests pass, revision/sequence fields round-trip, the overlay ends before an
intentionally delayed evidence response, and panel completion does not redraw the chart.

```bash
rtk git add src/superglm/editor/app/api/contracts.js src/superglm/editor/app/state/store.js src/superglm/editor/app/state/actions.js src/superglm/editor/app/main.js src/superglm/editor/app/metrics.js src/superglm/editor/app/reports.js src/superglm/editor/app/index.html src/superglm/editor/app/styles/panels.css tests/editor_frontend/actions.test.js tests/test_editor.py
rtk git commit -m "Refresh editor evidence in the background"
```

### Task 10: Add real-browser transition and stale-evidence regression tests

**Superseded standalone-harness draft:** The checked steps below assume a pytest `Page` plugin fixture
that this repository does not install and duplicate the Phase 1 Playwright dependency. Do not execute
them. Use Task 10A with the shared Phase 2 browser fixture.

**Files:**
- Modify: `pyproject.toml:23-32`
- Create: `tests/test_editor_browser.py`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/index.html`

- [x] **Step 1: Add Playwright to the development-only Python dependencies**

Add `"playwright>=1.40"` to `[project.optional-dependencies].dev`. Install dependencies and Chromium:

```bash
rtk uv sync --extra dev
rtk uv run playwright install chromium
```

Expected: the Python Playwright package and Chromium install successfully; production/editor extras remain unchanged.

- [x] **Step 2: Add browser-observable revision markers**

Whenever the remote store commits, set:

```javascript
svg.dataset.modelRevision = String(remote.modelRevision);
summaryFrame.dataset.modelRevision = String(remote.modelRevision);
```

These markers expose semantic state for regression tests and diagnostics without changing displayed content.

- [x] **Step 3: Write a browser test for atomic commit, overlay release, and request count**

Create `tests/test_editor_browser.py` with the existing `editor_model` and `_post_json` helpers moved to `tests/conftest.py` if pytest cannot resolve them across modules. Add:

```python
from __future__ import annotations

import threading

from playwright.sync_api import Page

from superglm.editor import EditorSession


def test_structural_refit_commits_once_before_delayed_metrics(
    page: Page,
    editor_model,
    monkeypatch,
):
    session = EditorSession.from_model(editor_model, terms=["region"])
    session.select_indices("region", [1, 2])
    widget = session.widget()
    metrics_started = threading.Event()
    release_metrics = threading.Event()
    original_metrics = widget._metrics

    def delayed_metrics(*args, **kwargs):
        metrics_started.set()
        release_metrics.wait(timeout=5)
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(widget, "_metrics", delayed_metrics)
    requests: list[tuple[str, str]] = []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    try:
        page.goto(widget.app_url)
        page.locator("#collapseLevels").click()
        assert metrics_started.wait(timeout=5)

        page.wait_for_function("document.querySelector('#appBusyOverlay').hidden === true")
        chart_revision = page.locator("#chart").get_attribute("data-model-revision")
        summary_revision = page.locator("#summaryFrame").get_attribute("data-model-revision")
        assert chart_revision == summary_revision == str(session.model_revision)
        assert page.locator("#metricGrid").get_attribute("aria-busy") == "true"

        structural_requests = [
            (method, url)
            for method, url in requests
            if url.endswith("/collapse_levels") or url.endswith("/state")
        ]
        assert sum(url.endswith("/collapse_levels") for _, url in structural_requests) == 1
        assert sum(url.endswith("/state") for _, url in structural_requests) == 1
    finally:
        release_metrics.set()
        widget.close()
```

The single `/state` request is the initial page load. There must be no second `/state` after collapse.

- [x] **Step 4: Add a browser stale-response test**

Use `page.route("**/metrics", handler)` to hold the first response body, commit a second edit/revision, allow the second request to render, then fulfill the first response with its older `model_revision` and `request_sequence`. Assert `#metricGrid` retains the second response and `#metricFreshness` does not revert to stale revision text. Keep the test payloads under one kilobyte and include complete `metrics.original` and `metrics.edited` dictionaries so the real renderer runs.

- [x] **Step 5: Run browser tests**

```bash
rtk pytest tests/test_editor_browser.py -q
```

Expected: chart and summary expose the same revision, overlay ends while metrics remain busy, only the initial `/state` occurs, and an older metric response cannot overwrite a newer one.

- [x] **Step 6: Commit real-browser performance regressions**

```bash
rtk git add pyproject.toml tests/conftest.py tests/test_editor_browser.py src/superglm/editor/app/main.js src/superglm/editor/app/index.html
rtk git commit -m "Test atomic editor refit recovery"
```

### Task 10A: Verify Atomic Refit and Stale Evidence in the Shared Browser Harness

**Files:**
- Modify: `tests/editor/test_editor_refit_browser.py`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/index.html`

- [ ] **Step 1: Expose semantic revision markers for observation**

In the existing snapshot and summary subscriptions, set:

```javascript
chart.dataset.modelRevision = String(snapshot.model_revision);
summaryFrame.dataset.modelRevision = String(snapshot.model_revision);
```

These are diagnostics only; the store remains authoritative.

- [ ] **Step 2: Add an atomic-commit/request-count browser test**

Using `open_editor_page(selected_term="territory")`, select two categorical points, install a
`page.route("**/metrics", handler)` that captures but does not immediately continue the post-refit
metrics route, then click Collapse. After the structural response, assert:

```python
page.wait_for_function("document.querySelector('#appBusyOverlay').hidden === true")
chart_revision = page.locator("#chart").get_attribute("data-model-revision")
summary_revision = page.locator("#summaryFrame").get_attribute("data-model-revision")
assert chart_revision == summary_revision == str(session.model_revision)
assert page.locator("#metricGrid").get_attribute("data-freshness") == "updating"
assert sum(url.endswith("/collapse_levels") for url in requests) == 1
assert sum(url.endswith("/state") for url in requests) == 0
```

Register the request listener after the fixture's initial navigation, so zero `/state` means no
post-refit recovery fetch. Finally abort or fulfill the held metrics route in `finally`.

- [ ] **Step 3: Add an out-of-order same-panel browser test**

Intercept two `/metrics` calls. Commit edit revision 1 and hold its request; commit revision 2 while
the first remains pending and hold its request. Build complete sub-kilobyte metric payloads from each
request's posted `model_revision`/`request_sequence`. Fulfill revision 2 first and assert its edited
value appears; fulfill revision 1 second and assert the DOM value and `data-freshness="current"`
remain on revision 2.

Use Playwright route events rather than fixed sleeps: wait on `page.wait_for_function()` predicates
backed by captured-route count and DOM freshness/revision attributes.

- [ ] **Step 4: Run browser ordering tests and commit**

Run:

```bash
rtk pytest tests/editor/test_editor_refit_browser.py -m browser --run-browser -q
```

Expected: chart/summary commit on one revision, the overlay ends while metrics are held, no successful
structural `/state` request occurs, and the stale response cannot overwrite newer evidence.

```bash
rtk git add tests/editor/test_editor_refit_browser.py src/superglm/editor/app/main.js src/superglm/editor/app/index.html
rtk git commit -m "Test atomic editor refit recovery"
```

### Task 11: Document, diagnose, and verify the complete performance slice

**Files:**
- Modify: `docs/editor_frontend.md`
- Modify: `src/superglm/editor/server.py:374-383`
- Modify: `src/superglm/editor/app/main.js:409-452`
- Test: `tests/test_editor.py`

- [ ] **Step 1: Add a failing diagnostics-shape test**

Extend the structural timing HTTP test:

```python
assert set(payload["timing"]) >= {
    "operation",
    "fit_ms",
    "summary_ms",
    "state_ms",
    "server_total_ms",
}
```

Add an HTTP response assertion that `Server-Timing` contains `json;dur=`. This header measures serialization without attempting to mutate a body after it has been encoded.

- [ ] **Step 2: Add JSON and client phase diagnostics**

Measure `jsonable` plus `json.dumps` inside `_json_response` and return:

```python
headers = {
    **_no_store_headers(),
    "Server-Timing": f"json;dur={json_ms:.3f}",
}
```

In `main.js`, retain separate values for request wait, store/DOM commit, the double-animation-frame paint boundary, and evidence completion. Do not fold metrics/report time back into `client_recovery_ms`; display it as panel-specific evidence timing.

- [ ] **Step 3: Document the implemented contracts**

Add these sections to `docs/editor_frontend.md`:

- `model_revision` increments for coefficient/reset/undo/redo/control/structural/reprofile changes and not for selection, term switching, zoom, control-count, or display reorder;
- structural routes return `{state, summary, timing}` from one locked post-refit snapshot;
- original scalar metrics live for the widget session, current scalars only for the active revision, row predictions are transient, and supplied evaluation data is an immutable session snapshot;
- one cache miss runs outside the mutation lock, one latest pending request is retained, and superseded browser responses are ignored;
- the browser commits chart/summary first, crosses a paint boundary, then refreshes only the visible evidence surface.

- [ ] **Step 4: Run formatting, focused tests, and the complete editor suite**

```bash
rtk ruff check src/superglm/editor tests/test_editor.py tests/test_editor_evaluation_cache.py tests/test_editor_evidence.py tests/test_editor_browser.py
rtk npm run test:frontend
rtk pytest tests/test_editor_evaluation_cache.py tests/test_editor_evidence.py tests/test_editor_browser.py -q
rtk pytest tests/test_editor.py -q
```

Expected: all checks and tests pass. No test uses a wall-clock performance ceiling other than a deterministic lock-release assertion guarded by synchronization events.

- [ ] **Step 5: Run the full Python suite**

```bash
rtk pytest tests/ -q
```

Expected: the full repository suite passes.

- [ ] **Step 6: Inspect the final diff for forbidden cache growth and request sequencing**

```bash
rtk git diff --check
rtk git diff --stat
rtk rg -n "predict|_fit_mu|DataFrame|ndarray" src/superglm/editor/evaluation_cache.py
rtk rg -n "requestJSON\(\"/state\"" src/superglm/editor/app/main.js
```

Expected: cache values are scalar dictionaries, the cache owns no DataFrame/prediction arrays, and the only `/state` request is initial load or explicit failure recovery.

- [ ] **Step 7: Commit diagnostics and documentation**

```bash
rtk git add docs/editor_frontend.md src/superglm/editor/server.py src/superglm/editor/app/main.js tests/test_editor.py
rtk git commit -m "Document editor transition and evidence flow"
```

## Final acceptance checklist

- [ ] Structural collapse, ungroup, and uncollapse return state and summary from one post-refit lock scope.
- [ ] The structural browser path performs no recovery `/state` request.
- [ ] Chart and summary commit in one store notification and expose the same revision.
- [ ] The global overlay ends at the paint boundary, before delayed metrics/report completion.
- [ ] Evidence failure retains the last confirmed payload and marks only that panel stale.
- [ ] Revision plus sequence checks reject old-model and same-model out-of-order responses.
- [ ] Original scalar metrics persist by split; current scalar metrics retain only the active revision.
- [ ] Training metrics reuse fitted artifacts without calling `predict`.
- [ ] Metrics and reports share identical split/model metric dictionaries.
- [ ] One materialized manual-edit model is reused per edit epoch and does not duplicate evaluation frames or `_dm`.
- [ ] One row-scale miss runs per widget, same keys share a future, and pending intermediate revisions are superseded.
- [ ] A blocked evidence computation does not hold `EditorWidget._lock` or delay selection/mutation.
- [ ] Persistent cache entries contain no prediction arrays, dense matrices, raw DataFrames, or historical revision artifacts.
- [ ] Python, native-JavaScript, and real-browser tests pass.
