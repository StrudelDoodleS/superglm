# Editor Integration, Accessibility, and Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the modernized editor as a coherent, recoverable, keyboard-usable notebook tool and publish enough user/developer documentation for an analyst and a Python-fluent maintainer to work confidently.

**Architecture:** This phase consumes the store/action controller, workspace, plot geometry, transition envelope, and evidence cache delivered by the first three plans. It adds browser-level acceptance coverage and small focused view helpers rather than another state layer, then documents the exact shipped behavior and wires a dedicated Chromium CI job.

**Tech Stack:** Native ES modules, semantic HTML/ARIA, CSS media queries, Python pytest with Playwright, MkDocs Material, FastAPI editor test server, GitHub Actions, and the existing `uv`/npm development tooling.

---

## Prerequisites

Complete these plans first:

1. `docs/superpowers/plans/2026-07-11-editor-foundation-state.md`
2. `docs/superpowers/plans/2026-07-11-editor-workspace-axis.md`
3. `docs/superpowers/plans/2026-07-11-editor-refit-evidence.md`

The selectors used below are part of the approved cross-phase contract:

```text
#appAlert, #appAlertMessage, #appAlertRetry, #appAlertDismiss
#appContent, #appBusyOverlay
.app-tab, .tool-button, .inspector-tab
#toolRail, #summaryPanel, #helpPane
#metricGrid[data-freshness]
#chart .edited, #chart [data-axis-tick]
```

### Task 1: Lock Mutation Recovery and Duplicate-Submission Behavior in the Browser

**Files:**
- Modify: `tests/test_editor_browser.py`
- Create: `src/superglm/editor/app/views/alerts.js`
- Modify: `src/superglm/editor/app/index.html`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/styles/panels.css`
- Test: `tests/editor_frontend/alerts.test.js`

- [ ] **Step 1: Write failing pure alert-view tests**

Add a fake-node test proving that errors remain visible, Retry invokes the stored retry callback once,
Dismiss clears the alert, and rendering a new error replaces the old callback:

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { createAlertView } from "../../src/superglm/editor/app/views/alerts.js";

test("alert retry is persistent, single-shot, and replaceable", async () => {
  const nodes = fakeAlertNodes();
  const alert = createAlertView(nodes);
  const calls = [];
  let releaseFirst;
  const firstPending = new Promise((resolve) => { releaseFirst = resolve; });
  alert.show("Edit failed", async () => {
    calls.push("first");
    await firstPending;
  });
  assert.equal(nodes.root.hidden, false);
  const firstClick = nodes.retry.click();
  const duplicateClick = nodes.retry.click();
  assert.deepEqual(calls, ["first"]);
  releaseFirst();
  await Promise.all([firstClick, duplicateClick]);
  alert.show("A newer failure", async () => calls.push("second"));
  await nodes.retry.click();
  assert.deepEqual(calls, ["first", "second"]);
  nodes.dismiss.click();
  assert.equal(nodes.root.hidden, true);
});

function fakeButton() {
  const listeners = new Map();
  return {
    hidden: false,
    disabled: false,
    addEventListener(type, listener) { listeners.set(type, listener); },
    async click() {
      const listener = listeners.get("click");
      if (listener) await listener();
    }
  };
}

function fakeAlertNodes() {
  return {
    root: { hidden: true },
    message: { textContent: "" },
    retry: fakeButton(),
    dismiss: fakeButton()
  };
}
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run:

```bash
rtk node --test tests/editor_frontend/alerts.test.js
```

Expected: FAIL because `views/alerts.js` does not exist.

- [ ] **Step 3: Implement the alert view and markup**

Create `alerts.js` with this public interface:

```javascript
// @ts-check
export function createAlertView({ root, message, retry, dismiss }) {
  let retryAction = null;
  let retrying = false;
  retry.addEventListener("click", async () => {
    if (!retryAction || retrying) return;
    retrying = true;
    retry.disabled = true;
    try { await retryAction(); } finally {
      retrying = false;
      retry.disabled = false;
    }
  });
  dismiss.addEventListener("click", () => hide());
  function show(text, onRetry = null) {
    message.textContent = text;
    retryAction = onRetry;
    retry.hidden = onRetry === null;
    root.hidden = false;
  }
  function hide() {
    root.hidden = true;
    retryAction = null;
  }
  return { show, hide };
}
```

Add an assertive alert immediately inside `.app-shell`, wire it to mutation errors from the action
controller, and make Retry replay only the failed action after authoritative rollback/refetch has
completed:

```html
<div id="appAlert" class="app-alert" role="alert" hidden>
  <span id="appAlertMessage"></span>
  <button id="appAlertRetry" type="button">Retry</button>
  <button id="appAlertDismiss" type="button">Dismiss</button>
</div>
```

- [ ] **Step 4: Add and run browser recovery tests**

In `tests/test_editor_browser.py`, intercept the first `**/op` request after Select all, return
`{"error":"forced edit failure"}` with HTTP 500, and assert:

```python
before = page.locator("#chart .edited").get_attribute("d")
page.locator('[data-op="shift_up"]').click()
expect(page.locator("#appAlertMessage")).to_contain_text("forced edit failure")
expect(page.locator("#chart .edited")).to_have_attribute("d", before)
page.locator("#appAlertRetry").click()
expect(page.locator("#appAlert")).to_be_hidden()
expect(page.locator("#chart .edited")).not_to_have_attribute("d", before)
```

Add a second intercepted test that delays one `/op` response, double-clicks the action, and asserts
the route counter is exactly one while `request.mutation.status` is running.

Run:

```bash
rtk test uv run pytest tests/test_editor_browser.py -m browser --run-browser -k "recovery or duplicate" -q
```

Expected: PASS.

- [ ] **Step 5: Commit the recovery slice**

```bash
rtk git add src/superglm/editor/app/views/alerts.js src/superglm/editor/app/index.html src/superglm/editor/app/main.js src/superglm/editor/app/styles/panels.css tests/editor_frontend/alerts.test.js tests/test_editor_browser.py
rtk git commit -m "Add persistent editor action recovery"
```

### Task 2: Complete Keyboard Tab and Tool Semantics

**Files:**
- Create: `src/superglm/editor/app/views/tabs.js`
- Modify: `src/superglm/editor/app/index.html`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/views/tool_rail.js`
- Test: `tests/editor_frontend/tabs.test.js`
- Test: `tests/test_editor_browser.py`

- [ ] **Step 1: Write failing keyboard-navigation unit tests**

Test the pure index helper for wrapping Home/End/Arrow navigation:

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { nextTabIndex } from "../../src/superglm/editor/app/views/tabs.js";

test("tab keyboard navigation wraps and supports Home and End", () => {
  assert.equal(nextTabIndex(0, 3, "ArrowRight"), 1);
  assert.equal(nextTabIndex(2, 3, "ArrowRight"), 0);
  assert.equal(nextTabIndex(0, 3, "ArrowLeft"), 2);
  assert.equal(nextTabIndex(1, 3, "Home"), 0);
  assert.equal(nextTabIndex(1, 3, "End"), 2);
  assert.equal(nextTabIndex(1, 3, "Enter"), 1);
});
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
rtk node --test tests/editor_frontend/tabs.test.js
```

Expected: FAIL because `views/tabs.js` does not exist.

- [ ] **Step 3: Implement roving tab focus**

Export `nextTabIndex(current, count, key)` and `bindTablist({tabs, activate})`. The binder must set
the active tab to `tabIndex=0`, all others to `-1`, focus the destination on Arrow/Home/End, call
`activate(tab.dataset.view || tab.dataset.pane)` on click/Enter/Space, and keep `aria-selected` in
sync. Use it for the application tabs and inspector tabs.

Give every tab an id and matching `aria-controls`; give each controlled region `role="tabpanel"`
and `aria-labelledby`. Give the tool rail `role="radiogroup"` and mode buttons `role="radio"` with
`aria-checked`, while Help remains a normal button with `aria-expanded` and `aria-controls`.

- [ ] **Step 4: Add browser keyboard assertions and run them**

Add a test that focuses the Editor tab, presses ArrowRight, and observes the Validation panel; then
focuses the Select tool, presses ArrowDown, and observes Move become checked; finally opens Help and
presses Escape to return focus to the Help button.

Run:

```bash
rtk test uv run pytest tests/test_editor_browser.py -m browser --run-browser -k keyboard -q
rtk node --test tests/editor_frontend/tabs.test.js
```

Expected: both commands pass.

- [ ] **Step 5: Commit the keyboard slice**

```bash
rtk git add src/superglm/editor/app/views/tabs.js src/superglm/editor/app/views/tool_rail.js src/superglm/editor/app/index.html src/superglm/editor/app/main.js tests/editor_frontend/tabs.test.js tests/test_editor_browser.py
rtk git commit -m "Complete editor keyboard navigation"
```

### Task 3: Finish Busy, Freshness, Focus, and Reduced-Motion Behavior

**Files:**
- Modify: `src/superglm/editor/app/index.html`
- Modify: `src/superglm/editor/app/main.js`
- Modify: `src/superglm/editor/app/styles/shell.css`
- Modify: `src/superglm/editor/app/styles/panels.css`
- Modify: `src/superglm/editor/app/metrics.js`
- Modify: `src/superglm/editor/app/reports.js`
- Test: `tests/test_editor_browser.py`

- [ ] **Step 1: Add failing browser assertions**

Cover these observable requirements in one focused test module section:

```python
expect(page.locator("#appBusyOverlay")).to_have_attribute("role", "status")
expect(page.locator("#appContent")).to_have_attribute("inert", "")
expect(page.locator("#metricGrid")).to_have_attribute("data-freshness", "updating")
expect(page.locator("#metricGrid")).to_contain_text(previous_metric_text)
expect(page.locator("#metricGrid")).to_have_attribute("data-freshness", "stale")
```

Delay a metrics response, then fail it. Assert the old values remain during both Updating and Stale,
and that the SVG path does not change when the delayed metrics response finishes.

- [ ] **Step 2: Run the focused browser tests and verify failure**

Run:

```bash
rtk test uv run pytest tests/test_editor_browser.py -m browser --run-browser -k "busy or freshness or reduced_motion" -q
```

Expected: FAIL on missing `inert`, freshness attributes, or reduced-motion behavior.

- [ ] **Step 3: Implement the semantics**

Wrap tabs/context/workspace in `#appContent`. While a blocking mutation runs, set
`appContent.inert = true`, show `#appBusyOverlay role="status" aria-live="polite"`, and restore the
previously focused control after commit/failure. Evidence updates never set `inert`.

Render evidence state with exact values:

```javascript
metricGrid.dataset.freshness = panel.status === "error" ? "stale" : panel.status;
metricGrid.setAttribute("aria-busy", panel.status === "updating" ? "true" : "false");
```

Add:

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; }
  .busy-spinner { animation: none; }
  .tool-popover, .sidepanel { transition: none; }
}
```

Keep `user-select: none` only on `.chart-shell`; set summary, history, reports, alerts, and Help to
`user-select: text`.

- [ ] **Step 4: Run browser and frontend verification**

Run:

```bash
rtk test uv run pytest tests/test_editor_browser.py -m browser --run-browser -k "busy or freshness or reduced_motion" -q
rtk npm run test:frontend
rtk npm run typecheck:frontend
```

Expected: all commands pass.

- [ ] **Step 5: Commit the interaction-state slice**

```bash
rtk git add src/superglm/editor/app/index.html src/superglm/editor/app/main.js src/superglm/editor/app/styles/shell.css src/superglm/editor/app/styles/panels.css src/superglm/editor/app/metrics.js src/superglm/editor/app/reports.js tests/test_editor_browser.py
rtk git commit -m "Polish editor busy and evidence states"
```

### Task 4: Publish the Analyst Editor Guide and Align In-App Help

**Files:**
- Create: `docs/guide/editor.md`
- Modify: `src/superglm/editor/app/views/help_drawer.js`
- Modify: `tests/test_editor.py`

- [ ] **Step 1: Write a failing documentation-content test**

Add a test that reads `docs/guide/editor.md` and requires these exact headings and operation terms:

```python
required = [
    "# Editing a Fitted Model",
    "## Open the Editor",
    "## Select, Move, Zoom, and Handles",
    "## Curve Selection Operations",
    "Straighten selection",
    "## Group and Ungroup Categorical Levels",
    "## Undo, Redo, and Recovery",
    "## Evidence Freshness",
    "## Save and Export",
    "## Keyboard Shortcuts",
]
assert all(text in guide for text in required)
```

- [ ] **Step 2: Run the documentation test and verify failure**

Run:

```bash
rtk test uv run pytest tests/test_editor.py -k analyst_editor_guide -q
```

Expected: FAIL because `docs/guide/editor.md` does not exist.

- [ ] **Step 3: Write the complete analyst guide**

Use this shipped-behavior text as the body of `docs/guide/editor.md` (add the screenshot in Task 6):

````markdown
# Editing a Fitted Model

The SuperGLM editor is a compact analyst workspace for reviewing and adjusting fitted one-dimensional
effects. Python owns the fitted model, edit history, summaries, and evidence; the browser provides a
fast visual preview and sends completed actions back to Python.

## Open the Editor

Create an editor session from a fitted model and display its widget in Jupyter or VS Code:

```python
from superglm.editor import EditorSession

session = EditorSession.from_model(
    model,
    terms=["age", "territory"],
    validation_data=(X_validation, y_validation, validation_weight),
)
session.widget()
```

The standard iframe is 1180 by 720 pixels. At narrower notebook widths the inspector becomes a
drawer; in a short window the workspace scrolls instead of shrinking the plot into an unusable strip.

## Select, Move, Zoom, and Handles

- **Select** brushes points or levels. Shift-click toggles a point or level; Select all selects the
  complete active term.
- **Move** drags selected relativities. The curve previews immediately and Python confirms the edit
  when the pointer is released.
- **Zoom** drags a box. The mouse wheel zooms around the pointer and Home restores the fitted extent.
- **Handles** edits a spline through fixed-x control handles. Basis contributions and the Build
  animation are available from Advanced when the fitted term exposes them.

The active mode is always visible in the left tool rail. Hover briefly over an icon, or focus it with
the keyboard, to see its name and shortcut.

## Curve Selection Operations

The floating palette acts on the current selection:

- Increase or decrease moves the selected relativities by five percent.
- Smooth reduces local variation while respecting adjacent unselected values.
- **Straighten selection** interpolates the selected relativities between their first and last points.
- Increasing and Decreasing apply anchored monotonic constraints.
- Level left, Average, and Level right flatten the selection to the named reference value.
- Snap highest and Snap lowest flatten to the selected extreme.

The icons remain compact so the plot stays large. Their hover/focus explanations describe the action
before it is run.

## Group and Ungroup Categorical Levels

Expanded and Collapsed display modes change only how an existing fitted grouping is drawn. They do
not rename levels or change the fitted model.

Collapse selected levels, Ungroup selected levels, and Restore previous collapse are structural
actions: SuperGLM refits the model and clears incompatible manual edit history. A confirmation is
shown only when history would actually be discarded; it names the term or levels and the number of
history entries. The refit overlay reports elapsed time. When the fit returns, the plot and summary
change together; metrics may remain marked Updating briefly.

Long category names may appear shortened with an end ellipsis on the x-axis. This is display-only.
Hover or focus the tick (or inspect the point tooltip) for the complete value; selection, grouping,
history, exports, and saved models retain the exact original string.

## Undo, Redo, and Recovery

Undo and Redo are visible in the application bar and expose their keyboard shortcuts. They operate on
confirmed Python history, not on an uncommitted pointer preview.

If an edit request fails, the browser restores the last confirmed Python state and shows a persistent
message. Choose Retry to repeat that action after recovery or Dismiss to keep the restored state. A
failed metric or report refresh does not undo a valid model edit; only that evidence panel becomes
Stale and offers Retry.

## Evidence Freshness

The metric strip and reports distinguish three useful states:

- **Current** describes the displayed model revision.
- **Updating** retains the last confirmed values while Python computes the new revision.
- **Stale** retains those values after a failed refresh and must not be read as current evidence.

Late responses for an older model revision are ignored. Validation and test rows remain in Python;
the browser receives aggregate metrics rather than a copy of the evaluation data.

## Inspector and Help

Summary shows the current in-force model, History shows confirmed edits, Advanced contains infrequent
curve/debug controls, and Help lists modes, gestures, operations, and shortcuts. On a narrow screen
these panes share a dismissible drawer. Escape closes a popover or drawer and returns focus to its
launcher.

## Save and Export

Save writes an edited model artifact to a chosen directory. Download writes the same artifact through
the browser when direct filesystem selection is unsuitable. The exported model uses Python's current
authoritative edit revision; display-only zoom, mode, grouping view, and truncated tick text are not
written into the model.

## Keyboard Shortcuts

- Use Tab to reach application tabs, the tool rail, context controls, the SVG action palette, and the
  inspector.
- Use arrow keys inside tab lists and the mode rail; Home and End jump to the first and last item.
- Use Enter or Space to activate a focused control.
- Use Escape to close the current popover, Help drawer, inspector drawer, or dialog.
- Use the displayed Undo/Redo shortcut for confirmed curve edits.

Pointer editing remains the primary high-density curve workflow. Full per-point keyboard editing and
an alternate editable data table are planned separately.
````

Use the same analyst-facing names in `help_drawer.js`. Its Straighten entry must read:

```text
Straighten selection
Interpolate the selected relativities between their first and last points.
```

- [ ] **Step 4: Run the focused tests and docs build**

Run:

```bash
rtk test uv run pytest tests/test_editor.py -k "analyst_editor_guide or help" -q
rtk test uv run mkdocs build --strict
```

Expected: both commands pass.

- [ ] **Step 5: Commit the analyst documentation**

```bash
rtk git add docs/guide/editor.md src/superglm/editor/app/views/help_drawer.js tests/test_editor.py
rtk git commit -m "Document the analyst editor workflow"
```

### Task 5: Rewrite the Frontend Guide for Python Developers

**Files:**
- Modify: `docs/editor_frontend.md`
- Modify: `tests/test_editor.py`

- [ ] **Step 1: Write a failing developer-guide structure test**

Require the guide to contain:

```python
required = [
    "## Mental Model for Python Developers",
    "## Authoritative Python State",
    "## Browser Store, Actions, and Selectors",
    "## JSON Requests and Semantic Revisions",
    "## DOM Events and Focus",
    "## CSS Grid, Flexbox, and Breakpoints",
    "## SVG Coordinates and Plot Geometry",
    "## Add a Tool Mode",
    "## Add a Curve Operation",
    "## Add an Inspector Panel",
    "## Add a FastAPI Route",
    "## Add a Metric",
    "## Add a Browser Regression Test",
    "rtk npm run test:frontend",
    "rtk npm run typecheck:frontend",
]
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
rtk test uv run pytest tests/test_editor.py -k frontend_guide_for_python_developers -q
```

Expected: FAIL because the current short guide lacks the required architecture and recipes.

- [ ] **Step 3: Rewrite the guide against shipped files**

Replace `docs/editor_frontend.md` with the following complete guide, adjusting only a filename if an
earlier approved phase used the equivalent responsibility-map name:

````markdown
# SuperGLM Editor Frontend for Python Developers

The editor is a same-origin browser application served by the Python kernel. It deliberately uses
native JavaScript modules, HTML, CSS, and imperative SVG: there is no frontend framework, bundler, or
production Node runtime. Python owns the model and all confirmed edits.

## Mental Model for Python Developers

Treat a browser module like a small Python module. Exported functions are its public API, objects are
usually plain dictionaries, and DOM nodes are mutable view objects. A click listener is a callback;
`await postJSON(...)` is the browser equivalent of awaiting an HTTP client in Python.

The important difference is that responses can finish out of order. Every evidence request carries
the model revision it describes, and the browser discards a response when a newer revision has
already committed.

## Authoritative Python State

`EditorSession` in `src/superglm/editor/session.py` owns terms, selection, history, the in-force model,
evaluation splits, semantic `model_revision`, and `edit_epoch`. `EditorWidget` in `widget.py` owns the
per-iframe lock, transition assembly, materialized edited model, evidence cache/coordinator, and local
server lifetime. `payloads.py` converts session objects to JSON-safe dictionaries.

Selection, active term, level display order, zoom, and inspector state do not change predictions.
Coefficient operations, undo/redo/reset, control-handle edits, structural refits, and distribution
profiling do. Tests in `tests/test_editor_state.py` lock those revision rules.

## Browser Store, Actions, and Selectors

`app/state/store.js` contains the reducer and subscription mechanism. Its durable top-level fields are:

```text
remote   latest Python-confirmed snapshot
view     mode, zoom, group display, inspector, and Help preferences
request  blocking mutation plus per-panel evidence freshness
```

`selectors.js` derives the active term, selection, enabled operations, and freshness labels. A DOM
class or select value is never a second state store. `actions.js` is the one mutation/evidence
controller: it snapshots confirmed state, allows a local preview, posts the action, commits a valid
response, rolls back/refetches on failure, and rejects stale evidence.

Pointer drag/brush/pan details remain inside `interactions.js`; they are transient and update too
frequently to belong in the durable store.

## JSON Requests and Semantic Revisions

`app/api/client.js` adds the per-widget token and throws a typed request error for non-2xx JSON.
`app/api/contracts.js` defines JSDoc shapes shared by `checkJs`. FastAPI routes live in `server.py`.

Ordinary mutations return authoritative state. Collapse, ungroup, and restore return:

```json
{
  "state": {"model_revision": 12, "terms": {}, "selection": {}, "history": {}},
  "summary": {"available": true, "compact": {}},
  "timing": {"fit_ms": 780.0, "state_ms": 10.5, "summary_ms": 54.3}
}
```

The store commits `state` and `summary` together. Metrics and reports carry `model_revision`; a
per-panel request sequence also prevents same-revision requests applying out of order.

## DOM Events and Focus

`index.html` contains stable semantic regions and ids. Focus/click handlers live in small modules
under `app/views/`; `main.js` only constructs dependencies and connects them. Application and
inspector tabs use roving tab focus. Tool modes expose radio semantics. Popovers open after a 350 ms
pointer delay, disappear immediately on pointer leave, open immediately on focus, and close with
Escape.

Blocking model mutations set `#appContent.inert`. Background evidence refreshes do not block controls.
Persistent mutation errors are rendered by `views/alerts.js` with Retry and Dismiss.

## CSS Grid, Flexbox, and Breakpoints

`styles/tokens.css` defines every colour, spacing, radius, and shadow token. `shell.css` owns the
application/context bars and workspace grid; `chart.css`, `panels.css`, and `dialogs.css` own their
named regions. `styles.css` imports them and remains the packaged entry point.

The normal workspace is a tool rail, flexible chart, and inspector. Below 1000 px the inspector is a
fixed drawer and the chart retains the flexible column. Short windows scroll the document instead of
reducing SVG height. Use `minmax(0, 1fr)` on flexible grid columns so long content cannot force the
chart outside its container.

## SVG Coordinates and Plot Geometry

`chart.js` performs imperative SVG rendering. `chart/geometry.js` contains pure layout functions.
`sx(value)` maps data x into pixels and `sy(value)` maps data y into the inverted SVG y-axis. The
renderer stores these scales for `interactions.js`; Python never receives pixel coordinates.

Categorical ticks are measured using their real SVG font. Geometry chooses orientation/density,
shortens display text with an end ellipsis, reserves a bottom gutter, and places the x title below the
tick bounds. Full labels remain in tick accessibility text, tick/point popovers, Python payloads, and
saved models.

## Evaluation Work

`evaluation_cache.py` stores scalar metric dictionaries by model token/edit epoch/dataset epoch.
`evidence.py` deduplicates identical work and permits one row-scale computation plus one coalesced
latest pending request per widget. Scoring runs outside the widget mutation lock against an immutable
snapshot. Validation/test prediction arrays are reduced to scalars and released; raw DataFrames and
string columns are referenced, not copied.

## Run Frontend Checks

```bash
rtk npm ci
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk test uv run pytest tests/test_editor_browser.py -m browser --run-browser -q
```

Node tests cover pure store/geometry/timing behavior. Python Playwright tests cover the packaged app,
real FastAPI routes, SVG/font layout, focus, responsive viewports, and request ordering.

## Add a Tool Mode

1. Add the button with `data-mode="inspect"`, an accessible name, and SVG icon in `index.html`.
2. Add `inspect` to the `EditorMode` typedef in `api/contracts.js`.
3. Add its analyst-facing popover copy in `views/tool_rail.js`.
4. Handle its pointer behavior in `interactions.js` without adding durable gesture fields to the store.
5. Add a mode-state case to `tests/editor_frontend/store.test.js` and a focus/pressed-state case to
   `tests/test_editor_browser.py`.

Run `rtk npm run test:frontend`, `rtk npm run typecheck:frontend`, and the focused browser test.

## Add a Curve Operation

1. Add a `data-op` button to the existing SVG-adjacent selection palette in `index.html`.
2. Add its label/description to the popover map; keep the Python operation id internal.
3. Add the operation branch to `EditorWidget._operate()` and implement the model mutation in
   `EditorSession` through `_commit()` so revision/history behavior stays centralized.
4. Test the numerical edit and revision in `tests/test_editor_state.py`.
5. Test the icon's accessible name and posted operation in `tests/test_editor_browser.py`.

## Add an Inspector Panel

1. Add a tab and matching `role="tabpanel"` region to `index.html` with `aria-controls` and
   `aria-labelledby`.
2. Add the pane name to `view.inspectorPane` in `state/store.js` and its selector.
3. Render the pane from a focused module in `app/views/` and bind it through `main.js`.
4. Add responsive styles to `styles/panels.css`; do not create a fourth permanent workspace column.
5. Add Node reducer and narrow-drawer browser cases.

## Add a FastAPI Route

1. Add a guarded route in `server.py`; parse untrusted JSON fields explicitly before calling a widget
   method.
2. Put authoritative mutation and locking in `EditorWidget`, not the route closure.
3. Return JSON-safe dictionaries through `_guarded_json`; never return row-scale evaluation data.
4. Add route/auth/error coverage to the focused Python editor test module.
5. Call it through `api/client.js` and the action controller, not directly from a view module.

## Add a Metric

1. Add the scalar property name and analyst label to `METRIC_LABELS` in `metrics.py`.
2. Populate it in the single scalar conversion path used by `EvaluationCache`.
3. Add it to the report subset only when it belongs in both live/report evidence.
4. Test original/current values, cache reuse, and JSON finiteness in
   `tests/test_editor_evaluation_cache.py`.
5. Add the display key to `app/metrics.js` and `app/reports.js`; do not re-score in the browser.

## Add a Browser Regression Test

Use the fixtures in `tests/test_editor_browser.py`, navigate to the tokenized widget URL, and assert
visible behavior through roles/labels/data attributes. Intercept a route to control delay/failure;
avoid fixed sleeps. Run:

```bash
rtk test uv run pytest tests/test_editor_browser.py -m browser --run-browser -k descriptive_name -q
```

When the behavior is pure, put the faster test in `tests/editor_frontend/` and keep only one browser
integration case.
````

- [ ] **Step 4: Run docs and focused tests**

Run:

```bash
rtk test uv run pytest tests/test_editor.py -k frontend_guide_for_python_developers -q
rtk test uv run mkdocs build --strict
```

Expected: both commands pass.

- [ ] **Step 5: Commit the developer documentation**

```bash
rtk git add docs/editor_frontend.md tests/test_editor.py
rtk git commit -m "Expand the editor frontend developer guide"
```

### Task 6: Add Documentation Navigation and a Deterministic Screenshot

**Files:**
- Create: `scripts/capture_editor_screenshot.py`
- Create: `docs/images/editor-workspace.png`
- Modify: `mkdocs.yml`
- Modify: `docs/guide/editor.md`

- [ ] **Step 1: Add the editor pages to MkDocs navigation**

Insert these entries without renaming existing pages:

```yaml
  - User Guide:
    - Recommended Workflows: guide/workflows.md
    - Editing a Fitted Model: guide/editor.md
  - Development:
    - Editor Frontend for Python Developers: editor_frontend.md
```

Add `![SuperGLM editor workspace](../images/editor-workspace.png)` after the analyst guide's opening
paragraph.

- [ ] **Step 2: Create the deterministic screenshot script**

Create this complete script:

```python
"""Capture the deterministic editor screenshot used by the analyst guide."""

from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from superglm import Categorical, Spline, SuperGLM
from superglm.editor import EditorSession


def main() -> None:
    rng = np.random.default_rng(20260711)
    n = 240
    age = rng.uniform(18.0, 85.0, n)
    territory = rng.choice([f"T{i:02d}" for i in range(1, 11)], n)
    territory_number = np.asarray([int(value[1:]) for value in territory], dtype=float)
    y = 0.35 + 0.12 * np.sin(age / 9.0) + 0.018 * territory_number
    y += rng.normal(0.0, 0.035, n)
    X = pd.DataFrame({"age": age, "territory": territory})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=0.1,
        features={
            "age": Spline(n_knots=8),
            "territory": Categorical(base="first"),
        },
    )
    model.fit(X, y)
    session = EditorSession.from_model(model, terms=["age", "territory"])
    widget = session.widget()
    output = Path(__file__).resolve().parents[1] / "docs/images/editor-workspace.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1180, "height": 720}, device_scale_factor=1)
            try:
                page.goto(widget.app_url, wait_until="networkidle")
                page.locator("#metricGrid[data-freshness='current']").wait_for()
                page.locator(".app-shell").screenshot(path=str(output))
            finally:
                page.close()
                browser.close()
    finally:
        widget.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate the screenshot and build documentation**

Run:

```bash
rtk test uv run python scripts/capture_editor_screenshot.py
rtk test uv run mkdocs build --strict
```

Expected: the PNG exists, is non-empty, shows the complete app shell at 1180x720, and the strict docs
build passes.

- [ ] **Step 4: Commit navigation and screenshot**

```bash
rtk git add scripts/capture_editor_screenshot.py docs/images/editor-workspace.png docs/guide/editor.md mkdocs.yml
rtk git commit -m "Publish editor documentation and screenshot"
```

### Task 7: Add the Dedicated Chromium CI Gate

**Prerequisite status:** Satisfied by the foundation plan and expanded by the workspace plan. The
checked steps below document the intended one-job boundary; do not add a second dependency group or
browser job. Phase 4 only verifies the existing job runs `tests/test_editor_browser.py` and
`tests/editor/`, then updates its path filters for documentation/screenshot tooling if required.

**Files:**
- Modify: `.github/workflows/dev-ci.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [x] **Step 1: Add the browser optional dependency and marker**

Add:

```toml
[project.optional-dependencies]
browser = [
    "playwright>=1.52",
]
```

Keep `browser: real Chromium editor integration tests` in the pytest marker list. Regenerate the lock
with:

```bash
rtk uv lock
```

- [x] **Step 2: Add one browser job, not one per Python matrix entry**

Add a `browser-editor` job to `dev-ci.yml` using Python 3.13. It must run:

```yaml
      - name: Install dependencies
        run: uv sync --extra dev --extra browser
      - name: Install Chromium
        run: uv run playwright install --with-deps chromium
      - name: Browser editor tests
        run: uv run pytest tests/test_editor_browser.py -m browser --run-browser -q
      - name: Frontend checks
        run: npm ci && npm run check:frontend
```

Mirror this single job in `ci.yml`; do not install browsers in the five-version Python unit-test
matrix.

- [x] **Step 3: Validate workflow syntax and all focused checks locally**

Run:

```bash
rtk test uv run pytest tests/test_editor_browser.py -m browser --run-browser -q
rtk npm ci
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk test uv run mkdocs build --strict
rtk ruff check src/superglm/editor tests/test_editor.py tests/test_editor_browser.py scripts/capture_editor_screenshot.py
```

Expected: all commands pass and the browser test file has no non-browser tests silently skipped.

- [x] **Step 4: Commit the CI gate**

```bash
rtk git add .github/workflows/dev-ci.yml .github/workflows/ci.yml pyproject.toml uv.lock
rtk git commit -m "Run editor browser acceptance tests in CI"
```

### Task 8: Run the Phase Acceptance Gate

**Files:**
- Verify: all files changed by this plan

- [ ] **Step 1: Run focused integration verification**

```bash
rtk test uv run pytest tests/test_editor.py tests/test_editor_browser.py tests/editor/ --run-browser -q
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk test uv run mkdocs build --strict
rtk ruff check src/superglm/editor tests/test_editor.py tests/test_editor_browser.py scripts/capture_editor_screenshot.py
```

Expected: every command exits zero.

- [ ] **Step 2: Inspect tracked scope and whitespace**

```bash
rtk git diff --check
rtk git status --short
```

Expected: no generated `site/`, `node_modules/`, Playwright reports, or unrelated notebook files are
tracked. The pre-existing `docs/notebooks/spline diagnostics/` path remains untouched.
