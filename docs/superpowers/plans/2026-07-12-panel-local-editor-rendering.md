# Panel-local Editor Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove app-wide flashes and unrelated DOM rewrites from ordinary editor and summary interactions while preserving accessible feedback for long structural refits.

**Architecture:** Keep the existing store and vanilla-JS renderer, but give each panel a semantic selector and one DOM ownership boundary. Add backend snapshot and chart generations for cheap freshness decisions, keep selection changes incremental inside stable SVG layers, and mark only structural refits as globally blocking.

**Tech Stack:** Python 3.10+, FastAPI widget server, ES modules with `// @ts-check`, native SVG/DOM, Node test runner, pytest, Playwright Chromium.

---

## File map

- `src/superglm/editor/widget.py`: publish monotonic state/chart generations.
- `src/superglm/editor/app/api/contracts.js`: document generation and blocking fields.
- `src/superglm/editor/app/state/store.js`: reject stale snapshots and gate incremental selection commits.
- `src/superglm/editor/app/state/actions.js`: classify structural mutations as blocking.
- `src/superglm/editor/app/chart.js`: stable point/legend layers and selected-point ordering.
- `src/superglm/editor/app/styles.css`: responsive wrapped selection palette.
- `src/superglm/editor/app/main.js`: panel-local selector subscriptions and blocking-only overlay.
- `src/superglm/editor/app/summary.js`: preserve table DOM for unchanged markup.
- `tests/editor_frontend/*.test.js`: reducer, action, selector, and summary identity tests.
- `tests/editor/test_editor_workspace_browser.py`: real-browser paint order, narrow palette, focus, overlay, and DOM-isolation tests.
- `tests/editor/test_editor_refit_browser.py`: structural overlay regression.
- `tests/test_editor.py`: backend generation semantics.

### Task 1: Order complete snapshots without comparing large payloads

**Files:**
- Modify: `src/superglm/editor/widget.py`
- Modify: `src/superglm/editor/app/api/contracts.js`
- Modify: `src/superglm/editor/app/state/store.js`
- Modify: `tests/test_editor.py`
- Modify: `tests/editor_frontend/store.test.js`
- Modify: `tests/editor_frontend/actions.test.js`

- [ ] **Step 1: Add RED backend generation tests**

Assert that successive `_state()` snapshots have increasing `state_generation`, selection
does not change `chart_generation`, and control-count, edit, reorder, and structural chart
changes do. Also assert the response still carries the unchanged semantic
`model_revision` for display-only changes.

```python
first = widget._state()
selected = widget._select(term, [0])
assert selected["state_generation"] > first["state_generation"]
assert selected["chart_generation"] == first["chart_generation"]

resized = widget._set_control_count(term, 6)
assert resized["chart_generation"] > selected["chart_generation"]
```

- [ ] **Step 2: Run the focused backend tests and confirm missing-key failures**

Run: `rtk proxy uv run pytest tests/test_editor.py -k 'generation' -q`

Expected: FAIL because snapshots do not yet publish either generation.

- [ ] **Step 3: Add RED frontend freshness tests**

Cover both generation dimensions in the pure reducer and action controller:

```javascript
const changedGeometry = {
  ...snapshot,
  state_generation: 4,
  chart_generation: 2,
  model_revision: snapshot.model_revision
};
assert.equal(commitSelectionRemote(state, changedGeometry).remote.chartEpoch, 1);

const stale = { ...snapshot, state_generation: 2, chart_generation: 1 };
assert.equal(commitSelectionRemote(newerState, stale).remote.snapshot.state_generation, 3);
```

Hold a `/select` response, commit a newer initialization snapshot, release the response,
and assert the newer snapshot remains authoritative and the preview clears.

- [ ] **Step 4: Run the focused frontend tests and confirm stale/geometry failures**

Run: `rtk npm run test:frontend -- --test-name-pattern='generation|stale selection|chart geometry'`

Expected: FAIL because `commitSelectionRemote` currently checks only model revision and term.

- [ ] **Step 5: Implement widget generations**

Initialize counters in `EditorWidget.__init__`, increment `state_generation` while producing
each complete `_state()`, and increment `chart_generation` once for every widget mutation
that changes chart geometry or display structure. Do not increment chart generation for
`_select` or `_set_term`.

```python
self._state_generation = 0
self._chart_generation = 0

def _advance_chart_generation(self) -> None:
    self._chart_generation += 1

def _state(self) -> dict[str, Any]:
    with self._lock:
        self._state_generation += 1
        ...
        return {
            "state_generation": self._state_generation,
            "chart_generation": self._chart_generation,
            ...,
        }
```

- [ ] **Step 6: Implement monotonic frontend commits**

Treat generation fields as optional during migration. If both current and incoming
`state_generation` values exist and incoming is smaller, retain current remote state while
clearing previews. Take the selection-only lane only when selected term and chart identity
match; use `chart_generation` when present and the previous model-revision rule otherwise.

```javascript
if (isOlderSnapshot(current, snapshot)) return clearRemotePreviews(state);
if (sameSelectedTerm(current, snapshot) && sameChartGeneration(current, snapshot)) {
  return commitSelectionSnapshotWithoutChartEpoch(state, snapshot);
}
return commitRemote(state, snapshot);
```

- [ ] **Step 7: Run backend/frontend regression tests**

Run:

```bash
rtk proxy uv run pytest tests/test_editor.py -q
rtk npm run check:frontend
```

Expected: PASS.

- [ ] **Step 8: Commit the freshness change**

```bash
rtk git add src/superglm/editor/widget.py src/superglm/editor/app/api/contracts.js \
  src/superglm/editor/app/state/store.js tests/test_editor.py \
  tests/editor_frontend/store.test.js tests/editor_frontend/actions.test.js
rtk git commit -m "Order editor snapshot rendering"
```

### Task 2: Preserve selection paint order and palette usability

**Files:**
- Modify: `src/superglm/editor/app/chart.js`
- Modify: `src/superglm/editor/app/styles.css`
- Modify: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Add RED SVG ordering tests**

After incrementally selecting an early base marker, assert every unselected data point
precedes every selected data point in document order. Select a non-base supplemental point
and assert it precedes the legend. Retain the existing identity assertions.

```python
order = page.locator("#chart").evaluate("""svg => {
  const nodes = [...svg.querySelectorAll('circle.point[data-index], .legend-layer')];
  return nodes.map(node => node.classList.contains('legend-layer')
    ? 'legend' : node.classList.contains('selected') ? 'selected' : 'unselected');
}""")
assert max(i for i, kind in enumerate(order) if kind == "unselected") \
    < min(i for i, kind in enumerate(order) if kind == "selected")
assert max(i for i, kind in enumerate(order) if kind == "selected") < order.index("legend")
```

- [ ] **Step 2: Add a RED 360-pixel palette test**

Select two categorical levels at a `360x560` viewport. Assert `scrollWidth <= clientWidth`,
all visible direct buttons fit inside the chart horizontally, retain 40 by 50 pixel targets,
and hit-test to themselves at their center.

- [ ] **Step 3: Run focused browser tests and confirm ordering/overflow failures**

Run: `rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py --run-browser -k 'paint_order or narrow_palette' -q`

Expected: FAIL because incremental points keep stale sibling order and the palette does not wrap.

- [ ] **Step 4: Add stable point and legend layers**

Create `.point-layer` before `.legend-layer` during full drawing. Append all data points to
the point layer. During an incremental update, preserve node identity but append selected
nodes to the end of the point layer; insert supplemental points there as well. Update bounds
insertion to use the point layer as its direct SVG anchor.

```javascript
for (const point of selectedPoints) pointLayer.appendChild(point);
const legendLayer = el("g", { class: "legend-layer" });
svg.appendChild(legendLayer);
legend(legendLayer, width - 160, 26, ...);
```

- [ ] **Step 5: Wrap the palette at constrained widths**

Change `.selection-menu` to `flex-wrap: wrap` while preserving existing target dimensions,
submenus, pointer events, and collision measurement.

- [ ] **Step 6: Run focused and complete browser tests**

Run: `rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py --run-browser -q`

Expected: PASS.

- [ ] **Step 7: Commit the visual integrity change**

```bash
rtk git add src/superglm/editor/app/chart.js src/superglm/editor/app/styles.css \
  tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Keep incremental selections visually stable"
```

### Task 3: Remove the ordinary-operation overlay flash

**Files:**
- Modify: `src/superglm/editor/app/api/contracts.js`
- Modify: `src/superglm/editor/app/state/store.js`
- Modify: `src/superglm/editor/app/state/actions.js`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `tests/editor_frontend/actions.test.js`
- Modify: `tests/editor_frontend/store.test.js`
- Modify: `tests/editor/test_editor_workspace_browser.py`
- Modify: `tests/editor/test_editor_refit_browser.py`

- [ ] **Step 1: Add RED mutation-presentation tests**

Hold selection, `/term`, `/control`, `/drag`, and ordinary `/op` requests. For each, assert
the global overlay stays hidden, editor regions are not inert, and focus never enters the
busy announcement. Hold one structural refit and assert the overlay remains visible and
focus restoration still works.

- [ ] **Step 2: Run focused tests and confirm the ordinary overlay is painted**

Run: `rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py tests/editor/test_editor_refit_browser.py --run-browser -k 'ordinary_busy or structural_busy' -q`

Expected: FAIL because every running mutation currently calls `setAppBusy(true, ...)`.

- [ ] **Step 3: Add explicit blocking state**

Extend mutation state with `blocking`. Selection and ordinary state mutations set it to
`false`; `executeStructuralMutation` sets it to `true`; idle/error states reset it to false.
Render the global overlay only for `status === "running" && blocking`.

```javascript
mutation: { status: "running", operation: name, error: null, blocking: true }

function renderMutationBusy(mutation) {
  setAppBusy(
    mutation.status === "running" && mutation.blocking,
    mutation.operation || "Working...",
    "Starting..."
  );
}
```

- [ ] **Step 4: Run frontend and browser mutation tests**

Run:

```bash
rtk npm run check:frontend
rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py \
  tests/editor/test_editor_refit_browser.py --run-browser -q
```

Expected: PASS.

- [ ] **Step 5: Commit the busy-state change**

```bash
rtk git add src/superglm/editor/app/api/contracts.js src/superglm/editor/app/state/store.js \
  src/superglm/editor/app/state/actions.js src/superglm/editor/app/main.js \
  tests/editor_frontend tests/editor/test_editor_workspace_browser.py \
  tests/editor/test_editor_refit_browser.py
rtk git commit -m "Stop flashing the editor for ordinary actions"
```

### Task 4: Isolate chart, shell, history, evidence, and summary rendering

**Files:**
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/summary.js`
- Modify: `tests/editor_frontend/summary.test.js`
- Modify: `tests/editor/test_editor_workspace_browser.py`

- [ ] **Step 1: Add RED DOM-ownership tests**

Remember chart, summary rows, metrics cards, report contents, history entries, term options,
and shell navigation nodes. Trigger a chart edit and assert only chart/history/context-owned
nodes may change. Trigger a summary status-only transition and assert table identities stay
connected and unchanged. Commit a new summary payload and assert only summary-owned content
changes. Observe every rAF and assert `path.edited` is always present.

- [ ] **Step 2: Run focused browser tests and confirm broad render fan-out**

Run: `rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py --run-browser -k 'render_boundary or summary_identity or populated_frame' -q`

Expected: FAIL because chart-epoch and broad view subscriptions call monolithic `render()`.

- [ ] **Step 3: Split the monolithic renderer**

Extract named renderers and semantic selectors for app view/report visibility, term picker,
chart workspace, history/app bar, and snapshot metadata. Keep metrics, summary, report,
selection, recovery, mutation, and previews on their existing dedicated subscriptions.

```javascript
store.subscribe(selectChartRenderState, renderChartWorkspace, sameChartRenderState);
store.subscribe(selectTermPickerState, renderTermPicker, sameTermPickerState);
store.subscribe(selectHistoryRenderState, renderHistoryAndAppBar, sameHistoryRenderState);
store.subscribe((state) => state.view.activeView, renderActiveView);
```

Contribution-build animation calls `renderChartWorkspace()` directly. No chart renderer may
call metrics, summary, report, or history renderers.

- [ ] **Step 4: Preserve unchanged summary markup**

Route all summary HTML writes through a helper that compares proposed markup with current
markup and calls `replaceChildren`/`innerHTML` only when content differs. Status/freshness
updates remain separate.

```javascript
function updateSummaryMarkup(frame, markup) {
  if (frame.innerHTML === markup) return false;
  frame.innerHTML = markup;
  return true;
}
```

- [ ] **Step 5: Run focused render-boundary tests**

Run:

```bash
rtk npm run check:frontend
rtk proxy uv run --extra dev pytest tests/editor/test_editor_workspace_browser.py --run-browser -q
```

Expected: PASS, with no unrelated node replacement and no empty rendered frame.

- [ ] **Step 6: Commit panel-local rendering**

```bash
rtk git add src/superglm/editor/app/main.js src/superglm/editor/app/summary.js \
  tests/editor_frontend/summary.test.js tests/editor/test_editor_workspace_browser.py
rtk git commit -m "Render editor panels independently"
```

### Task 5: Complete verification and independent review

**Files:**
- Verify only unless a test exposes a defect.

- [ ] **Step 1: Run complete frontend and browser suites**

```bash
rtk npm run check:frontend
rtk proxy uv run --extra dev pytest tests/test_editor_browser.py tests/editor/ --run-browser -q
```

Expected: PASS.

- [ ] **Step 2: Run focused backend compatibility**

```bash
rtk proxy uv run pytest tests/test_editor.py tests/test_editor_evaluation_cache.py \
  tests/test_editor_evidence.py -q
```

Expected: PASS.

- [ ] **Step 3: Run static, documentation, and packaging checks**

```bash
rtk ruff check src/ tests/
rtk test uv run ruff format --check src/ tests/
rtk git diff --check
rtk test uv run --group docs --extra plotting mkdocs build --strict
rtk test uv build
```

Expected: PASS. The wheel must contain editor assets and contain neither
`spline_diagnostics` nor `.superglm-debug`.

- [ ] **Step 4: Run independent spec and quality reviews**

Require no Critical or Important gaps against the panel-local design, with particular
attention to stale response ordering, focus/inert semantics, selector equality, DOM
ownership, selection paint order, and narrow notebook layouts.

- [ ] **Step 5: Commit any review-proven corrections and re-run affected checks**

Use one focused corrective commit per proven issue. Finish only with a clean worktree and
fresh green results.
