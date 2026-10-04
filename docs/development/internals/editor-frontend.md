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
- the structural changes waiting for Refit (`pending`) and the notes on timeline entries
  (`step_notes`);
- evaluation splits, and the cross-validation result with the rows its folds index;
- semantic `model_revision` and manual-edit `edit_epoch`.

`EditorWidget` in `src/superglm/editor/widget.py` owns the per-widget mutation lock, transition
assembly, materialized edited-model publication, scalar evaluation cache, evidence coordinator, and
local server lifetime. `payloads.py` converts confirmed session state into detached JSON-safe
dictionaries.

`session.py` keeps its public methods short and delegates to focused modules of module-level
functions that take the session as their first argument:

- `staging.py`: waiting changes, the one Refit that applies them, the order Undo and Redo take,
  notes, and the history records an exported model carries;
- `carry.py`: re-applies edited curves onto another fitted model, for Refit's carry-over and for
  Run CV and Final fit;
- `unseen.py`: the **New levels →** choice (`set_unseen`), an edit that refits nothing;
- `cv.py` and `jobs.py`: the Cross-validation tab's payload, and Run CV and Final fit as
  cancellable background jobs;
- `rating_preview.py`: one term's block of the rating table, from the Excel export's own payload
  builder;
- `persistence.py`: saved models with their `_editor_history`, structure files from
  `export_structure`, and saved sessions.

Structure files (`superglm.structure`) are library code. They rebuild terms through
`superglm.features.rebuild`, the same rebuild helpers `collapse.py` and `shapes.py` use, and import
nothing from the editor.

Selection, active term, display order, zoom, mode, and inspector state do not change predictions.
Coefficient operations, control-handle edits, undo/redo/reset, structural refits, and distribution
profiling do. A **New levels →** choice does too, because it changes predictions on data holding
new levels. Staging a change, and undoing or redoing a waiting change, do not: the model has not
changed. Keep those revision rules centralized in Python and covered in `tests/test_editor.py`.

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
`timing.js` keeps structural request, DOM commit, paint, and per-panel evidence durations separate.

Do not add a second state store in DOM classes, select values, or module globals. A DOM attribute may
expose state for accessibility or testing, but the store or Python session remains authoritative.

Pointer drag, brush, and pan details stay in `interactions.js`. They update too frequently to belong
in durable state; only the finished operation is sent to Python.

## JSON Requests and Semantic Revisions

`app/api/client.js` adds the per-widget token and throws a typed error for non-success JSON.
`app/api/contracts.js` defines JSDoc types checked by TypeScript's `checkJs` mode. FastAPI routes live
in `server.py` and call guarded methods on `EditorWidget`.

Ordinary mutations return an authoritative state snapshot. Every structural operation (collapse,
ungroup, set reference, transform, shape and revert) returns one atomic envelope built from a
single post-refit lock scope:

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

A staged change (`/stage`) returns the same envelope without fitting. Its revision is unchanged, so
it runs without the blocking overlay and re-requests no evidence. A refused request (HTTP 400) shows
Python's fixed sentence in the alert.

Every JSON response also exposes `Server-Timing: json;dur=...`, which measures JSON-safe conversion
and serialization separately from the route's model work.

Settings › Request timings shows browser request wait, synchronous store/DOM commit, and the
two-frame paint boundary as separate timings. Metrics, summary, and report completion are recorded
independently as panel evidence timings; they are not folded back into the blocking refit duration.

Metrics, summaries, and reports echo the requested revision and sequence. A response is accepted
only when both still match the panel's current request. Superseded work and late responses never
redraw the chart.

## Waiting Changes, Refit and Notes

Collapse, Ungroup, Set reference and the shapes are staged, not fitted. Each one is a
`PendingStep` in `session.pending` holding its labels-only `params` and the draft spec it leaves;
a term's draft is the last waiting step's spec for it, else the in-force fitted spec, so staged
changes on one term compose. One Refit fits every draft in a single clone, records one
`StructuralStep` whose saved state holds the waiting list (Undo brings the changes back as
waiting), and re-applies hand edits on the terms it did not restructure as one more step.

| Route | Body | Returns |
|---|---|---|
| `POST /stage` | `{operation, term, params, keep_reference, level_display}` | the structural envelope, without a fit |
| `POST /refit_pending` | `{level_display}` | the structural envelope of the one fit |
| `POST /note` | `{id, note}` | `{ok: true, state}` |
| `POST /set_unseen` | `{term, unseen}` | the state snapshot |

`params` by operation:

- `collapse`: `{levels: [label, ...], group_label: str | null}`;
- `ungroup`: `{levels: [label, ...]}`;
- `set_reference`: `{level: label}`;
- `shape`: `{lo, hi, degree, join}`, with `join` either `"tangent"` or `"kink"`.

`keep_reference` is a JSON boolean, true when absent; any other value is refused with the fixed
sentence "keep_reference must be true or false.". With Settings' "Refit after every structural
change" on, the browser posts to the operation's own route (`/collapse_levels`,
`/ungroup_levels`, `/set_reference`, `/shape_range`) instead, which stages the change and refits
it as one step that one Undo takes back.

The state snapshot carries the waiting changes in three places:

- top-level `pending`: `[{id, operation, term, label, params, note, time}]`, oldest first;
- per term, `pending`: `{groups, ranges, reference}`. `groups` is the draft's whole grouping once a
  waiting collapse or ungroup touches the term, else null; `ranges` lists the shaped ranges the
  draft adds or changes; `reference` is the level or group the draft pins, else null;
- each top-level `timeline` entry: `id` (seven hex digits), `time` (seconds since the epoch), `note`
  and `status`, one of `"applied"`, `"waiting"` or `"edit"`. A waiting or applied change is a
  `"pending"` entry.

Notes live in `session.step_notes`, keyed by step id, so they survive Undo and Redo. An exported
Python model carries the timeline up to now as `_editor_history`, one dict per entry with its id,
ISO 8601 time, operation, term, message, note, status and `predictor` (None until the SuperLSS
editor names one). The Excel workbook does not carry it.

## Settings and the Theme Switch

`app/views/settings.js` keeps the Settings pane's choices in one `localStorage` key,
`superglm.editor.settings`, as one JSON object. It exports `DEFAULT_SETTINGS`:

```js
{refitEveryChange: false, keepReference: true, followBrowserTheme: true,
 groupsDefault: "expanded", buildDurationMs: 10000, showTimings: false}
```

and `loadSettings()`, `saveSettings(patch)` and `onSettingsChange(listener)`, which returns an
unsubscribe function. Every read and write is wrapped in `try`/`catch`. A stored value is
normalised field by field, so a partial or hand-edited entry still loads. Where storage is
blocked, the choices live in memory until the page closes. A setting that affects the backend
travels with each request, as `keep_reference` does; Python stores no browser setting.

`app/views/theme.js` drives the DAY / NIGHT switch, `#themeSwitch` (`role="switch"`). The theme key
`superglm.editor.theme` decides: a stored `"light"` or `"dark"` is an explicit choice, and no
stored value means follow the browser. The pre-paint script in `index.html` reads that key alone
and writes `<html data-theme>` before first paint. `followBrowserTheme` mirrors the key, and the
module keeps the two equal. The flip's keyframes play only on a click; `tokens.css` turns every
animation and transition off under `prefers-reduced-motion: reduce`.

The older keys `superglm.editor.featureList` and `superglm.editor.shapeJoin` are separate and
unchanged.

## Background Jobs and the Cross-validation Tab

Run CV and Final fit take one fit per fold, or one fit on every row, so they run as background
jobs (`jobs.py`), polled by id:

| Route | Body | Returns |
|---|---|---|
| `POST /job_start` | `{kind}`, with `kind` either `"cv"` or `"final_fit"` | the job's status |
| `POST /job_status` | `{job_id, wait}` | `{job_id, kind, status, progress, result, error, cancel_requested, started_at, finished_at}` |
| `POST /job_cancel` | `{job_id}` | `{job_id, status, cancel_requested}` |

`status` is `"running"`, `"done"`, `"failed"` or `"cancelled"`. A job never holds the widget lock
while it works: its work runs on state captured when it starts, and its publish step takes the
lock and keeps the result only if the model revision is still the one it started from. A cancel
is a flag the work checks between folds, so a cancelled job publishes nothing. A finished job
evicts the finished jobs of its kind before it, so the runner keeps the last of each kind.

The tab is the fourth app view, `cv`. It reuses `#reportPanel`: `report_payload` has a `"cv"` kind
beside `"validation"` and `"final"`, and `app/views/cv_tab.js` renders it and starts, polls and
cancels the two jobs. `cv.py` checks the supplied result against the rows (`cv_data`, else train
data of the same row count, and the result's data fingerprint when it records one) and builds the
payload; `carry.py` puts the hand edits back on every fold and on the Final fit.

## Rating-Table Preview

`POST /rating_table` with `{term}` returns `{term, available, reason, columns, rows, note}`: the
term's block of the payload `export.rating_tables.build_rating_table_payload` builds for the Excel
workbook, on the same training split and materialised model. The payload is built once per model
revision, outside the lock like the export, and reused while the revision stands. A refusal is
one of the fixed sentences in `rating_preview.py`, never builder text.
`app/views/rating_table.js` draws it in `#ratingTableFrame`, behind the Chart / Table switch
`#termViewToggle`.

## Chart and Inspector Modules

- `chart/pending_overlay.js` draws waiting groups, ungroups and ranges over the last refit's
  curve.
- `chart/ordered_spline.js` reads an ordered spline's `spline_view`: its spline on a fine grid of
  the level axis, the level dots on it and the special levels as dots of their own.
- `chart/anchor_marks.js` marks the selection anchor that a Shift-click spans from.
- `views/summary_view.js` decides which summary rows a search and the All / Edited / Waiting filter
  keep and which term sections are open as Summary follows the chart; `summary.js` turns the
  result into markup and reapplies it after every render.

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
- `styles/dark.css` holds the warm dark palette, keyed on `<html data-theme="dark">`;
- `styles/cv.css` owns the Cross-validation tab;
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

Every split, the training split included, is scored through the model's predictions, so the
training split matches `SuperGLM.metrics` on the same data. The fit's own statistics belong to its
fitting design, which on a discrete fit is the binned one. When the training split holds the
objects used for fitting, only the weight-contract check is skipped, because the fit already ran
it. The splits share their cached scalar dictionaries between the metric strip and reports.

## Run Frontend Checks

From the repository root:

```bash
rtk npm run test:frontend
rtk npm run typecheck:frontend
rtk pytest tests/editor/test_editor_refit_browser.py -m browser --run-browser -q
```

Run the two browser suites as separate commands, never together in one `-n` run: run in one
process, `tests/editor` first leaves state behind that fails tests in the other file.

```bash
rtk pytest tests/test_editor_browser.py -m browser --run-browser -q
rtk pytest tests/editor -m browser --run-browser -q
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

## Add a Structural Operation

A structural operation replaces one term's spec, refits, and is one step on the same undo timeline
as the manual edits. All of them share one path, so a new one only supplies its spec builder and
its wiring:

1. Write the spec builder next to `collapse.py` and `shapes.py`. It builds a fresh replacement
   spec (never a mutated fitted copy) and returns `(spec, metadata)`, where `metadata["label"]` is
   the short text the Undo and Redo popovers show. Raise `EditorValueError` or `EditorTypeError`
   with fixed text for every refusal.
2. Add an `EditorSession.replace_with_...` method that passes the builder to `_refit_replacing` and
   the refit to `_push_structure`. It records one `StructuralStep` holding the editor state before
   the step (model, terms, edit history, level orders and selection) and puts the refit in force.
3. Add an `EditorWidget._...` method that calls `_structural_step` with the operation name and the
   session call. It takes the lock and returns the envelope.
4. Add a token-guarded route in `server.py` that parses the payload explicitly.
5. Add a `stage…` descriptor next to `stageReference` in `summary.js` and run it through
   `runStructuralChange` from `main.js`. The change is staged on `/stage` and waits, drawn on the
   chart, until Refit (`/refit_pending`, through `runStructuralRefit`) applies every waiting change
   in one fit. Give it a case in `refitAtOnceTransition` too: with Settings' "Refit after every
   structural change" on, the change goes to the operation's own route, which stages it and refits
   at once as one step that one Undo takes back.
6. Test the refit, the one pushed step and the refusals in `tests/test_editor_structure.py`, and add
   one browser case to `tests/editor/test_editor_structure_browser.py`.

The timeline is shared, so `EditorSession.undo` and `redo`, Revert and the `undo_redo` snapshot
field need no change. Undo takes the latest edit while there is one since the last step, and
otherwise swaps the step's stored state back in without refitting.

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
