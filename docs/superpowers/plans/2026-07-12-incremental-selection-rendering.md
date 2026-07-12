# Incremental Selection Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every editor selection gesture update without reconstructing or flashing the chart.

**Architecture:** Add a dedicated provisional-selection lane to the browser store and a
selection-specific action commit that accepts the complete authoritative snapshot without advancing
`chartEpoch`. Patch selected point styling, bounds, menu placement, and context text on the existing
SVG; retain the full renderer for real model/geometry changes.

**Tech Stack:** Native ES modules, immutable browser store, SVG DOM, Node test runner, Playwright,
Python/FastAPI editor backend.

---

### Task 1: Selection state and action semantics

**Files:**
- Modify: `src/superglm/editor/app/api/contracts.js`
- Modify: `src/superglm/editor/app/state/store.js`
- Modify: `src/superglm/editor/app/state/selectors.js`
- Modify: `src/superglm/editor/app/state/actions.js`
- Test: `tests/editor_frontend/store.test.js`
- Test: `tests/editor_frontend/actions.test.js`

- [ ] **Step 1: Write failing store tests for provisional selection**

Add tests proving that initial state has `selectionPreview: null`, a provisional selection overrides
`selectCurrentSelection`, clearing it restores the committed Python selection, and a
selection-specific remote commit stores the complete response snapshot while leaving `chartEpoch`
unchanged when model revision and active term are unchanged.

```js
const initial = createInitialEditorState(snapshot(3));
const previewed = setSelectionPreview(initial, "age", [4, 2, 2]);
assert.deepEqual(selectCurrentSelection(previewed), [2, 4]);

const response = snapshot(3);
response.selection.age = [2, 4];
const committed = commitSelectionRemote(previewed, response);
assert.equal(committed.view.selectionPreview, null);
assert.strictEqual(committed.remote.snapshot, response);
assert.equal(committed.remote.chartEpoch, initial.remote.chartEpoch);
assert.deepEqual(selectCurrentSelection(committed), [2, 4]);
```

- [ ] **Step 2: Run the store tests and verify RED**

Run:

```bash
rtk proxy node --test tests/editor_frontend/store.test.js
```

Expected: failure because `setSelectionPreview` and `commitSelectionRemote` are not exported and the
view contract has no selection-preview lane.

- [ ] **Step 3: Implement the store lane and selection commit**

Add this view-state contract:

```js
/** @property {{term:string, indices:number[]}|null} selectionPreview */
```

Initialize it to `null`. Add normalization/equality helpers, `setSelectionPreview`,
`clearSelectionPreview`, and `commitSelectionRemote`. The selection commit must:

```js
if (
  current &&
  current.model_revision === snapshot.model_revision &&
  current.selected_term === snapshot.selected_term
) {
  return {
    ...state,
    remote: {
      ...state.remote,
      snapshot
    },
    view: { ...state.view, selectionPreview: null }
  };
}
return commitRemote(state, snapshot);
```

Update `selectCurrentSelection` to return the normalized provisional indices when its term is active.
All full commits must clear both curve preview and selection preview.

- [ ] **Step 4: Write failing action tests**

Cover these action-controller requirements:

```js
await actions.executeSelectionMutation({ term: "age", indices: [0] });
assert.equal(postCalls, 0); // unchanged selection

const pending = actions.executeSelectionMutation({ term: "age", indices: [2, 1] });
assert.deepEqual(selectCurrentSelection(store.getState()), [1, 2]); // before response
response.resolve(authoritativeSnapshot);
await pending;
assert.equal(store.getState().view.selectionPreview, null);
```

Add a failure case where `postJSON` rejects, `getState` returns the prior selection, and the
provisional selection is cleared while the recovered snapshot becomes authoritative without
advancing `chartEpoch`. Add a different-revision recovery case that falls back to an ordinary full
commit.

- [ ] **Step 5: Run the action tests and verify RED**

Run:

```bash
rtk proxy node --test tests/editor_frontend/actions.test.js
```

Expected: failure because `executeSelectionMutation` does not exist.

- [ ] **Step 6: Implement `executeSelectionMutation`**

The method must normalize indices, skip semantic no-ops, atomically install provisional selection
plus running mutation state, post `/select`, commit with `commitSelectionRemote`, and use the same
selection-specific commit during recovery. Retry dispatch must route `name === "select"` back to
this method. Do not schedule evidence because selection does not change model revision.

- [ ] **Step 7: Run frontend state tests and commit**

Run:

```bash
rtk proxy node --test tests/editor_frontend/store.test.js tests/editor_frontend/actions.test.js
rtk npm run typecheck:frontend
```

Expected: all pass.

Commit:

```bash
rtk git add src/superglm/editor/app/api/contracts.js src/superglm/editor/app/state/store.js \
  src/superglm/editor/app/state/selectors.js src/superglm/editor/app/state/actions.js \
  tests/editor_frontend/store.test.js tests/editor_frontend/actions.test.js
rtk git commit -m "Add dedicated editor selection state"
```

### Task 2: Incremental SVG selection patching

**Files:**
- Modify: `src/superglm/editor/app/chart.js`
- Modify: `src/superglm/editor/app/main.js`
- Test: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Write failing real-browser identity tests**

Add a no-op empty-box test that records the edited path, first data point, x-axis title, and term
option nodes. Observe requests and chart mutations, perform an empty box drag, wait two animation
frames, and assert:

```python
assert select_requests == 0
assert page.evaluate("before => before === document.querySelector('#chart path.edited')", before_path)
assert page.evaluate("before => before === document.querySelector('#chart circle.point[data-index]')", before_point)
assert session.selection("curve").tolist() == []
```

Add a changed-selection test with a delayed backend response. It must assert selected styling and
status change before releasing the response, then assert path, existing point, axis, and term-option
identity remain unchanged after Python confirmation.

- [ ] **Step 2: Run the two browser tests and verify RED**

Run:

```bash
rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py \
  --run-browser -k "selection and (noop or incremental)" -q
```

Expected: no-op test observes `/select`; both tests observe path/point replacement.

- [ ] **Step 3: Implement `updateChartSelection`**

Export a focused chart function that uses the existing `svg._scale` plus stored resolved-view
metadata. It must:

```js
for (const point of svg.querySelectorAll("circle.point[data-index]")) {
  const selected = displaySelected.has(Number(point.dataset.index));
  point.classList.toggle("selected", selected);
  point.setAttribute("r", selected ? "4.6" : "3.4");
}
```

Update or remove only `.selection-bounds` and `.selection-bounds-halo`, preserve existing point and
path nodes, and call `positionSelectionMenu` with the recalculated bounds. Preserve collapsed display
mapping through `displaySelection`.

- [ ] **Step 4: Subscribe selection independently in `main.js`**

Subscribe the full renderer to `remote.chartEpoch`. Ordinary/full commits advance the epoch, while a
same-revision, same-term `commitSelectionRemote` stores the complete snapshot without advancing it;
selection confirmation therefore does not call `render()`.

Add a separate semantic selection subscription. Its listener calls `updateChartSelection`,
`updateCollapseAction`, and `renderContextBar`; it must not rebuild term options or unrelated panels.

- [ ] **Step 5: Run browser identity tests and commit**

Run the Step 2 command again. Expected: both pass with zero chart-node replacements.

Commit:

```bash
rtk git add src/superglm/editor/app/chart.js src/superglm/editor/app/main.js \
  tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Patch editor selections without redrawing"
```

### Task 3: Interaction integration, rollback, and compatibility

**Files:**
- Modify: `src/superglm/editor/app/interactions.js`
- Modify: `tests/editor/test_editor_workspace_browser.py`
- Verify: `tests/test_editor_browser.py`
- Verify: `tests/editor/test_editor_refit_browser.py`

- [ ] **Step 1: Add failing interaction and failure-recovery tests**

Route both modifier-click and box-selection gestures through `executeSelectionMutation`. Add a
delayed-failure browser test that observes immediate provisional styling, releases a 500 response,
waits for recovery, and asserts the prior selection is restored while path/point identities remain
stable and the existing recovery alert appears.

Add a collapsed-categorical selection case verifying displayed groups map to exact source indices
without a full redraw.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py \
  --run-browser -k "selection and (failure or collapsed)" -q
```

Expected: failure until interactions use the dedicated action.

- [ ] **Step 3: Route selection gestures through the dedicated action**

In `interactions.js`, replace both generic `executeStateMutation({name:'select', ...})` calls with:

```js
await context.actions.executeSelectionMutation({
  term: context.selectedTerm(),
  indices
});
```

Keep brush creation/removal local. Do not create a term-payload preview for selection.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
rtk npm run check:frontend
rtk proxy uv run --extra dev pytest tests/test_editor_browser.py tests/editor/ --run-browser -q
rtk proxy uv run pytest tests/test_editor.py tests/test_editor_evaluation_cache.py \
  tests/test_editor_evidence.py -q
rtk ruff check src/ tests/
rtk test uv run ruff format --check src/ tests/
rtk git diff --check
```

Expected: all pass. Browser instrumentation must show no path/point replacement for no-op,
successful, or recovered selection-only changes.

- [ ] **Step 5: Commit the completed fix**

```bash
rtk git add src/superglm/editor/app/interactions.js \
  tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Eliminate editor selection flicker"
```
