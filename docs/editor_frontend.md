# SuperGLM Editor Frontend for Python Developers

The editor is a same-origin browser application served by the Python kernel. It deliberately uses
native JavaScript modules, HTML, CSS, and imperative SVG: there is no frontend framework, bundler,
or production Node runtime. Python owns the model and every confirmed edit.

This guide is written for maintainers who are comfortable with Python but newer to browser
development.

## Mental Model for Python Developers

Treat a JavaScript module like a small Python module:

- exported functions are its public API;
- plain objects are usually dictionary-like records;
- an event listener is a callback;
- `await client.postJSON(...)` is the browser equivalent of awaiting an HTTP client;
- DOM nodes are mutable view objects, not application state.

The important browser-specific complication is concurrency. Requests can finish out of order. A
slow response for model revision 8 must not overwrite revision 9, and two requests for revision 9
must still be applied in request order. Evidence requests therefore carry both `model_revision` and
`request_sequence`.

The frontend has no hidden build step. Edit a source file under `src/superglm/editor/app/`, run the
native-JavaScript checks, then reload or recreate the widget so the local server serves the changed
asset.

## Authoritative Python State

`EditorSession` in `src/superglm/editor/session.py` owns:

- the in-force fitted model and immutable original reference model;
- editable term arrays and categorical metadata;
- selections and undo/redo history;
- evaluation splits;
- semantic `model_revision` and manual-edit `edit_epoch`.

`EditorWidget` in `src/superglm/editor/widget.py` owns the per-widget mutation lock, transition
assembly, materialized edited-model publication, scalar evaluation cache, evidence coordinator, and
local server lifetime. `payloads.py` converts confirmed session state into detached JSON-safe
dictionaries.

Selection, active term, display order, zoom, mode, and inspector state do not change predictions.
Coefficient operations, control-handle edits, undo/redo/reset, structural refits, and distribution
profiling do. Keep those revision rules centralized in Python and covered in `tests/test_editor.py`.

## Browser Store, Actions, and Selectors

`app/state/store.js` contains the immutable state helpers and synchronous subscription mechanism.
Its durable top-level fields are:

```text
remote   latest Python-confirmed snapshot plus structural summary
view     active term/view/mode, zoom, grouping display, inspector, and preview
request  blocking mutation, per-panel evidence freshness, and recovery state
```

`selectors.js` derives the active term, selection, renderable preview, and model revision.
`actions.js` is the single mutation/evidence controller. It snapshots request payloads, posts
actions, validates structural envelopes, commits confirmed state, reconciles uncertain failures,
debounces evidence per panel, and rejects stale responses.

Do not add a second state store in DOM classes, select values, or module globals. A DOM attribute may
expose state for accessibility or testing, but the store or Python session remains authoritative.

Pointer drag, brush, and pan details stay in `interactions.js`. They update too frequently to belong
in durable state; only the finished operation is sent to Python.

## JSON Requests and Semantic Revisions

`app/api/client.js` adds the per-widget token and throws a typed error for non-success JSON.
`app/api/contracts.js` defines JSDoc types checked by TypeScript's `checkJs` mode. FastAPI routes live
in `server.py` and call guarded methods on `EditorWidget`.

Ordinary mutations return an authoritative state snapshot. Collapse, ungroup, and restore return one
atomic envelope built from a single post-refit lock scope:

```json
{
  "state": {"model_revision": 12, "terms": {}, "selection": {}, "history": {}},
  "summary": {"available": true, "compact": {}},
  "timing": {
    "operation": "collapse_levels",
    "fit_ms": 780.0,
    "summary_ms": 54.3,
    "state_ms": 10.5,
    "server_total_ms": 851.2
  }
}
```

The store commits `state` and `summary` in one update. The browser crosses a two-animation-frame
paint boundary, releases the blocking overlay, and then starts visible evidence without awaiting it.
There is no successful post-refit `/state` fetch.

Every JSON response also exposes `Server-Timing: json;dur=...`, which measures JSON-safe conversion
and serialization separately from the route's model work.

Metrics, summaries, and reports echo the requested revision and sequence. A response is accepted
only when both still match the panel's current request. Superseded work and late responses never
redraw the chart.

## DOM Events and Focus

`index.html` contains stable semantic regions and IDs. Small modules under `app/views/` own focused
view behavior; `main.js` constructs their dependencies and connects store subscriptions.

Application tabs, inspector tabs, and the tool rail use keyboard focus deliberately. Icon popovers
open after a short pointer delay, disappear immediately on pointer leave, open immediately on focus,
and close with Escape. Native dialogs restore focus to their launcher. Global Undo/Redo shortcuts
pause while a dialog is open.

Blocking model mutations make the workspace regions inert. Background metrics, summary, and report
refreshes never make the editor inert. They retain their last confirmed payload, set `aria-busy`,
and expose Current, Updating, Stale, or Error through a polite live region. Mutation failures remain
in the assertive application alert.

## CSS Grid, Flexbox, and Breakpoints

The styles are plain CSS loaded directly by `index.html`:

- `styles/tokens.css` owns colours, spacing, radii, and shadows;
- `styles/shell.css` owns the application bars and workspace shell;
- `styles/chart.css` owns SVG/chart styling;
- `styles/panels.css` owns inspector, evidence, Help, and report regions;
- `styles/dialogs.css` owns native dialogs and profiling UI;
- `styles.css` contains the remaining shared and legacy component rules.

The normal workspace is a tool rail, flexible chart, and inspector. Below 1048 px the inspector is a
fixed drawer, while the chart keeps the flexible column. Short windows scroll instead of reducing
the SVG to an unusable height. Use `minmax(0, 1fr)` for flexible grid columns so long content cannot
force the plot beyond its container.

Do not introduce Tailwind merely to rename these rules. A framework becomes worthwhile only if the
application grows many repeated screens/components or needs a broader frontend team. For this
single, Python-served analytical workspace, native modules keep the runtime, packaging, and mental
model smaller.

## SVG Coordinates and Plot Geometry

`chart.js` performs imperative SVG rendering. Pure layout calculations live in
`chart/geometry.js`. `sx(value)` maps a data x value into pixels; `sy(value)` maps a data y value into
the inverted SVG y-axis. The renderer supplies scales to `interactions.js`; Python never receives
pixel coordinates.

Categorical ticks are measured using their real SVG font. Geometry chooses tick density and
orientation, shortens display text with a Unicode end ellipsis, reserves the required bottom gutter,
and places the x-axis title below the tick bounds. Full labels remain in accessibility text,
tick/point popovers, Python payloads, history, and saved models. Display truncation never mutates a
category string.

## Evaluation Work and Memory

`evaluation_cache.py` retains scalar metric dictionaries. Original-model scalars persist for the
widget session; current-model scalars are cleared whenever the current revision advances. Cache
values contain floats only—never DataFrames, dense design matrices, prediction arrays, or category
strings.

`evidence.py` runs at most one cache miss and retains at most one latest pending request. Identical
keys share a future; an intermediate pending revision is marked superseded. Materialization and
scoring run outside the widget mutation lock against a captured request. Evaluation frames,
weights, offsets, and the fitted design matrix are shared by identity rather than copied. Temporary
prediction arrays are reduced to scalars and released.

The training split can reuse exact fitted artifacts when its X, y, weights, and offset are the same
objects used for fitting. Validation and test splits use exact scoring and then share their cached
scalar dictionaries between the metric strip and reports.

## Run Frontend Checks

From the repository root:

```bash
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk pytest tests/editor/test_editor_refit_browser.py -m browser --run-browser -q
```

`rtk npm run check:frontend` runs the first two together. Node tests cover pure store, action,
geometry, popover, and accessibility behavior. Python Playwright tests cover the packaged app, real
FastAPI routes, responsive viewports, focus, SVG layout, and request ordering.

## Add a Tool Mode

1. Add a button with `data-mode="inspect"`, an accessible name, and its existing-style SVG icon in
   `index.html`.
2. Add `inspect` to the `EditorMode` typedef in `api/contracts.js`.
3. Add analyst-facing popover and Help copy in the relevant `app/views/` module.
4. Handle its pointer behavior in `interactions.js`; do not add durable gesture fields to the store.
5. Add a mode-state test under `tests/editor_frontend/` and a focus/pressed-state browser case.

Run the frontend check and the focused browser test before committing.

## Add a Curve Operation

1. Add a `data-op` button to the existing SVG-adjacent selection palette in `index.html`.
2. Keep the compact SVG, and add its full action name and explanation to the popover/Help content.
3. Add the operation branch to `EditorWidget._operate()`.
4. Implement the numerical mutation through the session commit path so history and revision changes
   stay centralized.
5. Test the numerical result and revision in Python, then test the accessible name and posted
   operation in the browser.

## Add an Inspector Panel

1. Add a tab and matching `role="tabpanel"` region to `index.html`, connected with `aria-controls`
   and `aria-labelledby`.
2. Add the pane name to the `inspectorPane` contract and initial view state.
3. Render and bind it from a focused module under `app/views/`.
4. Add responsive rules to `styles/panels.css`; do not create a fourth permanent workspace column.
5. Add a pure state test and a narrow-drawer browser case.

## Add a FastAPI Route

1. Add a token-guarded route in `server.py` and parse untrusted JSON fields explicitly.
2. Put authoritative mutation and locking in `EditorWidget`, not the route closure.
3. Return detached JSON-safe dictionaries; never return row-scale evaluation data.
4. Add route/auth/error coverage to `tests/test_editor.py`.
5. Call it through `api/client.js` and the action controller, not directly from a view module.

For evidence routes, echo `model_revision` and `request_sequence` on success, error, and superseded
outcomes.

## Add a Metric

1. Add the scalar property name and analyst label to `METRIC_LABELS` in `metrics.py`.
2. Populate it in the one scalar calculation path used by `EvaluationCache`.
3. Add it to the report subset only when it belongs in both live and report evidence.
4. Test original/current values, fit-artifact correctness, cache reuse, and JSON finiteness in
   `tests/test_editor_evaluation_cache.py`.
5. Add its display key to `app/metrics.js` and, if appropriate, `app/reports.js`. The browser must
   never score it.

## Add a Browser Regression Test

Use the shared fixtures under `tests/editor/`, navigate to the tokenized widget URL, and assert
visible behavior through roles, labels, and semantic data attributes. Intercept a route to control
delay or failure; coordinate through route events and DOM predicates instead of fixed sleeps.

```bash
rtk pytest tests/editor/test_editor_refit_browser.py \
  -m browser --run-browser -k descriptive_name -q
```

When behavior is pure, put the faster test in `tests/editor_frontend/` and retain only one browser
integration case.
