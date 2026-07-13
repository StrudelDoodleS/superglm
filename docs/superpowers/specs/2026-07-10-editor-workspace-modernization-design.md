# Editor Workspace Modernization Design

**Status:** Approved in design discussion; pending written-spec review

**Date:** 2026-07-10

**Baseline:** `feature/editor-refit-timing-debugging` at `0790d38`

## Summary

Modernize the SuperGLM editor as a dense analyst workstation while retaining its existing
Python/FastAPI boundary, native browser modules, and hand-built SVG editing engine. The work adds an
analyst-oriented tool rail, an inspector/help surface, visible recovery actions, responsive notebook
layout, adaptive categorical labels, coherent post-refit updates, bounded evaluation caching, and
real browser tests.

The editor will not migrate to React, Lit, another framework, a metaframework, or Tailwind. The
frontend will instead be split into focused native modules with gradual JavaScript typing, explicit
state ownership, executable tests, and documentation written for a Python-fluent maintainer learning
web development.

## Context and Evidence

The current browser app has a sound product boundary: Python owns the authoritative model and edit
state; the browser renders JSON and posts completed actions. Its editing, validation, and final-fit
views are useful, the chart distinguishes original/previous/current states, and structural refits
already expose elapsed progress.

The implementation has nevertheless outgrown its original "tiny app" description. The browser
frontend is about 5,000 lines, concentrated in `styles.css` (1,161 lines), `main.js` (873),
`chart.js` (805), `summary.js` (796), and `interactions.js` (707). There is no executable DOM or
browser coverage; most frontend tests assert that source strings exist.

The audit found four immediate classes of problem:

1. **Layout and discoverability:** the fixed-height, overflow-hidden page has no responsive
   breakpoint. At narrow widths the fixed 340px inspector can reduce the chart to almost nothing.
   Undo/redo and several gestures exist only as hidden keyboard or pointer conventions.
2. **Plot-label regression:** categorical labels rotate once there are more than eight levels, but
   the SVG retains a fixed 72-unit bottom margin and a fixed x-axis-title position. Real labels
   overlap the title or extend outside the root SVG viewport.
3. **Split post-refit commit:** a structural refit returns and paints a summary, then the browser
   separately fetches `/state`, redraws the SVG, refreshes metrics, and refreshes the active report.
   The stagger is explicit in the control flow.
4. **Repeated full-data scoring:** editor metrics bypass fitted prediction/statistic caches, score
   both original and current models, and repeat the same work in reports. A representative 150k-row
   discrete run measured a 10.5ms state rebuild but a 563ms metrics refresh and a 555ms final report.

A notebook-shaped 150k-row Tweedie profile made the duplication clearer:

- default validation metrics on 22.5k rows: about 287ms;
- validation report over train/validation/test: about 1.97s;
- metrics JSON conversion and serialization: about 0.03ms;
- cache-aware core `model.metrics()` on matching fit data: about 0.2ms on first construction and
  effectively immediate thereafter.

The bottleneck is therefore repeated Python scoring and request sequencing, not vanilla JavaScript
or SVG rendering.

## Goals

- Optimize the editor for experienced pricing and modelling analysts.
- Keep domain terminology and a dense single-workspace presentation.
- Make every interaction mode and curve operation discoverable without replacing compact icons with
  permanent text.
- Preserve the current SVG editing behavior and Python-owned model state.
- Make chart and summary changes appear together immediately after a refit completes.
- Prevent metrics and reports from blocking the essential editor update.
- Remove redundant scoring while keeping cache memory bounded as data scales with rows and features.
- Make the frontend understandable to a Python developer through small modules, types, tests, and a
  task-oriented developer guide.
- Fix categorical tick/title collision and handle arbitrarily long display labels safely.

## Non-Goals

- No framework, metaframework, Tailwind, SSR, routing, or separate JavaScript server runtime.
- No rewrite of the SVG curve renderer or its selection/editing operations.
- No browser-side implementation of SuperGLM scoring, feature transforms, links, or likelihoods.
- No dense evaluation design-matrix cache.
- No persistent cache for every edit-history revision.
- No new nominal/ordered categorical chart encoding in this pass.
- No complete keyboard-editable chart table in this pass; core control semantics and focus behavior
  will improve, while full keyboard point editing remains roadmap work.
- No automatic named checkpoint before every structural refit in this pass.
- No redesign of session persistence or hosted/remote-kernel topology.

## Audience and Interaction Principles

The primary user is an analyst who understands GLMs and wants speed, density, and direct
manipulation. The editor will not behave like a beginner wizard. It will, however, stop relying on
undiscoverable gestures.

The interaction principles are:

- technical model language remains visible;
- controls map to analyst tasks rather than internal implementation terms;
- frequent actions remain one click or one shortcut away;
- consequences are explained only when an action is destructive or expensive;
- help is contextual and does not displace the chart unnecessarily;
- errors never leave the browser preview silently divergent from Python;
- stale evidence is labelled rather than displayed as current.

## Architecture

### Authoritative Boundary

`EditorSession` and `EditorWidget` remain authoritative. Python owns model copies, term values,
selection, history, refits, saved artifacts, summaries, metrics, and reports. The browser never
becomes an alternative numerical engine.

The stable request boundary remains:

```text
EditorSession
  -> JSON state/transition payload
  -> browser store
  -> focused view renderers
  -> user action
  -> one action controller
  -> FastAPI route
  -> EditorSession/EditorWidget mutation
```

### Browser Store

The browser will have one small explicit store with three durable sections:

```text
remote
  authoritative session snapshot and model revision
view
  active term/view/mode, CI, per-term zoom/group mode, inspector/help state
request
  idle/running/error state, active operation, panel freshness, recovery message
```

Pointer gesture state remains isolated in the chart interaction controller because it is transient
and performance-sensitive. It includes brush, point drag, control drag, pan, box zoom, and level
reorder previews.

DOM controls render from the store. A select element, active CSS class, or dialog attribute must not
become a competing source of truth. Pure selectors derive values such as the active term, current
selection, group display mode, enabled actions, and panel freshness.

### Model Revisions

Python responses will expose a semantic `model_revision` that changes only when predictions or fit
evidence can change. Selection, term switching, zooming, inspector changes, and display-only level
reordering do not increment it. Coefficient edits, reset, undo/redo of edits, control-handle changes,
structural refits, and distribution re-profiling do.

The browser captures the revision when starting metrics or report work and discards a response if a
newer revision has committed before it arrives. A request sequence number additionally prevents two
same-revision UI requests from applying out of order.

### Action Controller

Every mutation goes through one action controller:

1. capture the last confirmed remote snapshot;
2. set either a blocking mutation state or a non-blocking evidence-refresh state;
3. allow an immediate local chart preview where required;
4. call Python;
5. on success, atomically commit the returned authoritative snapshot;
6. on mutation failure, restore or refetch authoritative state and show a persistent recovery
   message;
7. refresh secondary evidence without blocking the primary editor commit.

The controller distinguishes mutation failure from secondary evidence failure. If a metric refresh
fails after a valid edit, the edit remains committed and the metric panel is marked stale with Retry.

### Module Boundaries

The target structure keeps native ES modules and makes `main.js` a small composition root:

```text
app/
  index.html
  styles/
    tokens.css
    shell.css
    chart.css
    panels.css
    dialogs.css
  api/
    client.js
    contracts.js
  state/
    store.js
    actions.js
    selectors.js
  views/
    app_bar.js
    context_bar.js
    tool_rail.js
    inspector.js
    popover.js
    help_drawer.js
    metrics.js
    summary.js
    history.js
    reports.js
    save_dialog.js
  chart/
    geometry.js
    render.js
    interactions.js
  main.js
```

This is a responsibility map, not a requirement to split every file before useful behavior lands.
Files will be extracted incrementally when the relevant behavior is under test. The chart remains an
imperative island; a component framework will not own individual SVG nodes.

### Gradual Typing and Tooling

Browser source remains JavaScript. `// @ts-check` and JSDoc typedefs will describe `EditorState`,
`TermPayload`, `ChartScale`, transition responses, request bodies, and evidence payloads. TypeScript
checking is development-only and produces no bundled runtime artifact. Production wheels continue
to serve the readable source modules through `importlib.resources`.

The repository gains a small development-only `package.json` and lockfile with no runtime
dependencies or bundler. TypeScript supplies `checkJs`; pure JavaScript modules use Node's built-in
test runner. Browser behavior remains in Python pytest/Playwright so the Python-fluent maintainer can
read and extend the highest-value integration tests. The developer guide documents `npm ci`, the
single frontend check command, and the focused pytest command explicitly.

## Workspace Layout

### Application Bar

The first row contains the Editor, Validation, and Final Fit views on the left. Undo, Redo, and Save
are visible on the right. Undo/redo have disabled states and shortcut hints.

### Context Bar

The second row contains the term selector, term kind, EDF, and context-dependent controls such as
group display, reference CI, and inspector visibility. Model/debug implementation details do not
occupy the default bar.

### Tool Rail

A narrow vertical rail owns interaction mode:

- Select;
- Move;
- Zoom;
- Handles;
- Help at the bottom.

Select is the quiet default. Existing wheel zoom and modifier/pan gestures remain available. Handles
mode retains basis contributions and construction animation, while animation duration and detailed
timings move under Advanced.

Tool buttons use proper pressed/radio semantics, accessible names, focus states, and shortcuts. Every
icon has a hover/focus popover. `TOOLTIP_SHOW_DELAY_MS` defaults to 350ms (the intended tuning range
is 250-400ms), pointer leave closes immediately, keyboard focus opens immediately, and Escape closes
an open popover or drawer.

### SVG Selection Palette

The existing floating selection palette and all curve operations remain SVG-adjacent compact icons.
Only their explanations change. For example, the existing `linearise` operation is presented on
hover/focus as:

```text
Straighten selection
Interpolate the selected relativities between their first and last points.
```

Internal operation names may appear in advanced details, not as the primary analyst-facing label.

### Inspector and Help

One right-hand slot hosts Summary, History, Advanced, and Help. Help does not create an additional
permanent column. It documents modes, gestures, shortcuts, selection operations, refits, history,
and saving.

At normal notebook width the inspector is visible beside the chart. Below approximately 1000px it
becomes a dismissible drawer so the chart retains usable width. Escape and an explicit close button
dismiss it. Short windows allow document/panel scrolling rather than clipping the chart and metric
strip against a fixed viewport.

### Metric Strip

The metric strip remains below the chart. During a background refresh it retains the last confirmed
values, visibly marked `Updating...`, rather than jumping to empty placeholders. A failed refresh is
marked stale and offers Retry.

## Adaptive Categorical Axis Layout

### Root Cause

The current fixed viewBox is 940x520 with margins `{left: 76, right: 76, top: 48, bottom: 72}`.
Rotated categorical labels are anchored 30 units below the plot and rotated -45 degrees, but the
x-axis title remains at `height - 20`. The projected width of a label therefore consumes unreserved
vertical space and can cross the title or SVG boundary.

### Layout Algorithm

Categorical axis layout becomes measurement-driven:

1. resolve expanded/collapsed display data before computing geometry;
2. generate candidate ticks from visible categories;
3. measure labels using the actual SVG font in a temporary measurement layer;
4. choose a horizontal or angled treatment based on measured fit;
5. calculate a per-label pixel budget from available width;
6. shorten only displayed text to the longest value that fits, using a Unicode end ellipsis;
7. if labels are still too dense, retain first, last, and an evenly spaced subset;
8. project the final rotated text bounds into the bottom-gutter calculation;
9. place the x-axis title below the final tick bounds;
10. verify that ticks and title remain inside the root SVG viewport.

The pure geometry helper accepts measured widths so density, truncation, and gutter calculations can
be unit tested independently of the DOM. A browser test verifies the final font/layout integration.

### String Integrity

Truncation is display-only. A category such as
`MyReallyLongCategoryNameThatWouldNeverFit` may display as `MyReallyLongCategoryNa…`, but its model
value, payload value, selection identity, history, grouping, export, and saved artifact remain exact.

The complete value appears in point tooltips and in a tick hover/focus popover. Accessible names use
the complete value. No fitted category string is mutated or serialized with an ellipsis.

## Structural Refit Commit and Responsiveness

### Transition Response

Collapse, ungroup, and uncollapse responses will return one transition envelope:

```json
{
  "state": {"model_revision": 12, "terms": {}, "selection": {}, "history": {}},
  "summary": {"available": true, "compact": {}},
  "timing": {"fit_ms": 780.0, "state_ms": 10.5, "summary_ms": 54.3}
}
```

The state and summary are derived from the same post-refit model under the widget lock. The browser
does not paint the summary before receiving state and does not perform a follow-up `/state` request.

### Atomic Primary Commit

After the refit response:

1. commit state and summary to the store together;
2. update the SVG and summary under the busy overlay;
3. wait for the next animation frame so the browser confirms a paint opportunity;
4. remove the global overlay;
5. start only the necessary visible evidence refreshes in the background.

Chart and summary should therefore appear as one state change within one or two frames after Python
returns. Metrics and reports may settle later without delaying interaction with the new model view.

### Background Evidence Execution

Async browser requests alone are insufficient if Python performs row-scale scoring while holding the
widget mutation lock. On a cache miss, an evidence route therefore captures an immutable scoring
snapshot and semantic cache key under the lock, releases the lock, and computes in a bounded worker.
Cache hits remain synchronous and immediate.

Each widget permits at most one row-scale evidence cache miss in flight. Requests for newer model
revisions coalesce to the latest requested revision rather than creating a queue of intermediate
manual-edit scores. Completed old-revision results may populate their semantic cache entry, but the
browser discards them and the widget does not retain old current-revision entries. This bounds CPU,
transient arrays, and memory while keeping model mutations responsive.

Ordinary edit evidence refresh is briefly debounced so a drag sequence does not score every pointer
commit. Structural refits schedule evidence only after the atomic primary commit. The worker never
mutates `EditorSession`, a fitted model, or evaluation data.

### Selective Rendering

Store subscriptions/render functions are scoped to the state they consume. Completing metrics does
not rebuild the term selector or SVG. Contribution animation redraws only the chart rather than the
whole application render path.

### Performance Diagnostics

Development diagnostics measure distinct phases:

- fit;
- summary construction;
- state construction;
- JSON serialization;
- client request;
- store commit and SVG construction;
- next-animation-frame paint;
- metrics per split/source;
- report aggregation.

Absolute performance thresholds are recorded for investigation but not used as brittle CI gates.
Tests assert request counts, update ordering, and absence of redundant scoring.

## Evaluation Cache and Memory Policy

### Current Redundancy

Editor metrics currently call `session.to_model()`, then exact-score the original and selected models.
Reports score original and selected models again for every split. Manual edits can trigger an edited
prediction during model materialization and then score that copy again for metrics and summary.

The core model already stores `_fit_mu`, `_fit_null_mu`, `_fit_stats`, and a cache-aware metrics path.
Editor evaluation will use or factor through these artifacts rather than maintaining a duplicate
uncached implementation.

### Shared Scalar Cache

A Python `EvaluationCache` supplies a complete metric dictionary to the live strip and reports. Its
semantic key contains:

- reference/current model token;
- manual edit epoch;
- evaluation dataset epoch and split;
- family/link/dispersion values that affect metrics.

The immutable original scalar metrics remain valid for the widget session. Current metrics retain
only the active model/edit revision. A new edit evicts the prior current revision; history does not
retain row-scale predictions or metric arrays.

Identical in-flight cache keys share one computation so a metric strip and newly opened report cannot
start duplicate scoring. Cache insertion is atomic, but metric computation occurs outside the widget
mutation lock against the immutable snapshot described above.

When an evaluation dataset is identical by object identity to retained fit data, evaluation reuses
the fitted model's predictions, null predictions/statistics, and fit-stat cache. Explicit validation
or test predictions are computed once, reduced to scalar metrics, and released. Live metrics and
reports then share the cached scalar dictionary.

The evaluation datasets are treated as immutable snapshots for the editor session. A future API that
replaces evaluation data must bump the dataset epoch. External in-place mutation is unsupported and
will be documented; hashing all rows on every lookup would defeat the cache.

### Edited Model Materialization

At most one materialized manual-edit model is retained for the current edit epoch and shared by
summary, metrics, reports, and export. The previous materialization is released on the next
coefficient edit/reset/undo/redo or structural model replacement.

The materialization must not duplicate evaluation DataFrames or a retained dense design matrix. Its
mutable coefficient/result/spec state is private; immutable row-scale inputs are shared by reference
or excluded where they are unnecessary. Structural refits continue to use the session's fitted
in-force model directly when there are no manual coefficient edits.

### Persistent Memory Rules

The default cache stores scalar results, not prediction vectors:

| Artifact | Policy | Persistent scaling |
|---|---|---|
| Original/current metric dictionaries | Cache | O(number of splits) |
| Existing fitted train predictions/statistics | Reuse | No new allocation |
| Explicit validation/test predictions | Aggregate then release | Transient O(n) |
| Raw DataFrames and string columns | Reference, never duplicate | No cache copy |
| Categorical codes | Optional, byte-bounded | At most configured budget |
| Dense design matrices | Never cache | Avoid O(n x p) |
| Historical prediction revisions | Never cache | No history growth |
| Browser payload | Aggregates and plot grids | Independent of evaluation n |

Only one row-scale cache miss may execute per widget, so transient prediction memory does not
multiply with rapid edits or simultaneous panel requests.

Metric calculation remains float64. The design does not downcast numerical values to save memory.

### Strings and Optional Code Cache

Raw categorical strings can dominate input memory, especially with pandas `object` columns. The
cache retains the supplied DataFrame by reference and never copies row-level strings into cache or
browser state. Unique labels remain exact.

If post-cache profiling shows first-time validation/test scoring is still too slow, a later bounded
cache may retain compact signed categorical codes (`int8`, `int16`, or `int32`, including the `-1`
missing sentinel) plus one unique-label table. The cache has a strict byte budget and LRU eviction.
It is not part of the default scalar-cache requirement because caching every categorical column
would scale with `n x p_categorical`.

### No Browser Scoring

Browser scoring would require transferring evaluation rows and implementing grouped categoricals,
ordered effects, splines, interactions, offsets, links, distributions, and exact Tweedie likelihood
twice. It would expand an aggregate metrics response of under a kilobyte into megabytes of row data
and create a numerical parity burden. Python remains the only scoring engine.

## Safety and Recovery

- Ordinary edits preserve immediate local preview.
- Mutation success replaces the browser snapshot with Python-confirmed state.
- Mutation failure rolls back to the last confirmed snapshot or refetches state before allowing the
  next edit.
- Errors appear in a persistent alert with Retry and Dismiss, not only in a title attribute.
- Secondary panels retain last-confirmed values with an explicit stale/updating marker.
- The same action cannot be submitted twice while pending.
- Structural refits retain the elapsed-time overlay.
- Collapse/ungroup asks for confirmation only when it will discard non-empty manual edit history.
- The confirmation names the affected term/levels and exact number of history entries cleared.

## Accessibility and Visual Polish

- Top and side tabs follow keyboard tab semantics and expose `aria-controls`/selection state.
- Mode buttons and toggles expose pressed/selected state, not color alone.
- Progress/status regions are polite live regions; mutation failures use an assertive alert region.
- Busy mode makes underlying controls inert, not merely pointer-disabled.
- Popovers open on focus and are dismissed with Escape.
- Reduced-motion preferences disable nonessential animation.
- Summary/report content remains text-selectable; selection suppression is limited to the chart
  interaction surface.
- Undefined CSS tokens such as `--orange` and `--text` are replaced by a complete documented token
  set.
- Exposure outlines and focus indicators meet usable contrast.

Full keyboard editing of every SVG point/handle and an alternate editable data table remain future
work, but this pass must not make the existing keyboard/focus experience worse.

## Testing Strategy

Implementation follows test-first development. Production behavior is not changed before a focused
test demonstrates the missing behavior or regression.

### Pure Frontend Tests

Pure tests cover:

- store transitions and selectors;
- model-revision and stale-response rejection;
- action failure/rollback state;
- tick density and first/last retention;
- pixel-budget truncation and ellipsis behavior;
- rotated-label/gutter geometry;
- popover hover delay, immediate pointer dismissal, focus display, and Escape;
- evaluation-panel freshness state;
- evidence debounce, latest-revision coalescing, and in-flight de-duplication.

### Browser Tests

Python pytest/Playwright tests run the real FastAPI editor at representative viewports. They cover:

- 1180x720 notebook layout;
- narrow-width inspector/help drawer behavior;
- short-window scrolling without chart/metric overlap;
- tool-rail mode selection and accessible state;
- visible Undo/Redo and disabled behavior;
- floating curve-operation popovers;
- realistic `T01`-`T10` categorical labels;
- long labels that truncate visually while retaining full tooltip/accessibility text;
- no tick/title or tick/viewport intersection;
- failed optimistic edits restoring authoritative state;
- structural state and summary committing together;
- global overlay ending before intentionally delayed metrics;
- stale delayed metric/report responses being ignored;
- no redundant `/state` request after a structural refit.

Timing tests assert ordering and request counts rather than a fixed wall-clock limit.

### Python Tests

Focused Python tests cover:

- transition-envelope state/summary consistency;
- semantic model revision rules;
- evaluation-cache hits and invalidation;
- fit-artifact reuse;
- original/current cache separation;
- report reuse of metric dictionaries;
- one edited-model materialization per edit epoch;
- no cache invalidation for selection, term switching, zoom, or display-only reorder;
- memory policy: scalar cache entries do not retain prediction arrays or copied evaluation frames;
- row-scale evidence work runs outside the mutation lock with one in-flight cache miss per widget.

## Documentation

### Analyst Guide

A navigable editor guide documents installation, opening the editor, modes, gestures, shortcuts,
selection operations, grouped levels, structural refits, history, evidence freshness, Help, and save
or export behavior. It uses current screenshots and portable notebook setup instructions.

### Frontend for Python Developers

A developer guide teaches this app through familiar concepts:

- Python state versus browser state;
- JSON requests and responses;
- DOM elements and event listeners;
- CSS Grid/Flexbox and responsive breakpoints;
- SVG coordinates and scale functions;
- store/action/selector responsibilities;
- model revision and asynchronous stale-result protection;
- how to run type checks and browser tests.

It includes short recipes for adding a tool-rail mode, a curve operation, a FastAPI route, an
inspector panel, a metric, and a browser regression test. The documentation maps each visible region
to its source file so a maintainer does not need framework knowledge to find code.

## Acceptance Criteria

The implementation is accepted when all of the following are true:

- the editor ships and runs without a frontend framework, Tailwind, or a production compilation
  step;
- Python remains authoritative and the browser has one explicit store/action path;
- the approved tool-rail, context bar, inspector/help slot, Undo/Redo, and popover behavior are live;
- normal notebook layout remains dense and narrow/short layouts remain usable without plot collapse
  or overlap;
- the existing SVG edit operations behave as before;
- categorical ticks use display-only adaptive truncation and never overlap the axis title or leave
  the SVG viewport in tested layouts;
- structural refits return state and summary together and do not make a follow-up `/state` request;
- chart and summary commit before metrics/report refresh and appear in one coherent paint;
- stale asynchronous evidence cannot overwrite a newer model revision;
- training evidence reuses fit artifacts, live metrics and reports share scalar caches, and repeated
  views do not rescore unchanged model/split pairs;
- a slow evidence cache miss does not hold the widget mutation lock, duplicate the same cache-key
  computation, or queue every intermediate edit revision;
- persistent evaluation cache entries do not retain dense matrices, copied row data, or historical
  prediction vectors;
- failures restore authoritative state or mark only the affected evidence panel stale;
- focused Python, pure frontend, and real-browser tests pass;
- analyst and Python-developer documentation is published in the project navigation.

## Risks and Mitigations

- **Large refactor regression:** extract modules incrementally behind characterization/browser tests;
  do not combine unrelated renderer changes.
- **Async races:** tag evidence with model revision and request sequence; discard late results.
- **Stale cache from external data mutation:** treat evaluation inputs as immutable snapshots and
  provide epoch-based invalidation for future replacement APIs.
- **Browser font differences:** separate pure geometry from measured widths and verify final bounds in
  Chromium plus one additional browser engine where CI cost permits.
- **Cache memory growth:** retain scalars by default, one current edit materialization, and no history
  predictions; make optional code caches byte-bounded.
- **UI density loss:** preserve analyst terminology and metrics while moving only developer/debug
  controls into Advanced.
- **Hidden CPU work after overlay removal:** label evidence as updating, compute only visible panels,
  run one cache miss per widget outside the mutation lock, coalesce to the latest revision, and ignore
  stale results rather than blocking the primary model view.

## Future Triggers

Reconsider a component framework only after this cleanup if new work repeatedly requires coordinated
component creation/destruction across several independent screens, or if state/lifecycle defects
remain common despite the explicit store and action model. If that happens, preserve the SVG editor
as one imperative island and evaluate React or Lit with Vite. A metaframework remains inappropriate
unless the editor becomes a separately hosted, routed, authenticated product.

Future performance work may add bounded categorical-code or feature-transform caches after profiling
the scalar-cache implementation. Those optimizations require separate memory/performance evidence
and are not assumed by this design.
