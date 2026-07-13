# Panel-local editor rendering and no-flash mutations

## Context

The editor currently routes every mutation through one global busy treatment and many
authoritative snapshot changes through one broad `render()` function. That makes small,
local actions look like application reloads even when the actual SVG work is fast.

Chromium traces identified the visible shared flash:

- selection displayed the full-shell blurred busy overlay for about 42 ms (two frames);
- handle release, term change, and an ordinary edit displayed it for about 18–23 ms
  (one frame);
- every display moved focus into the busy announcement and restored or redirected it;
- no animation frame or mutation-observer callback observed an empty chart;
- the synchronous SVG work occupied about 5–8 ms per geometry-changing action.

The selection-only incremental renderer already avoids rebuilding chart geometry, but
three edge cases remain: selected point stacking can become incorrect, supplemental
points can be painted above the legend, and the action palette overflows narrow notebook
viewports. Complete snapshots also need stronger ordering than `model_revision`, because
that revision intentionally excludes some display-only geometry changes.

## Goals

1. Ordinary editor actions never obscure or repaint the whole application shell.
2. A state change updates only the panel or controls that own the changed data.
3. Selection remains immediate and incremental, including success and rollback.
4. Geometry-changing chart updates may redraw the chart, but cannot redraw metrics,
   reports, history, summary, navigation, or unrelated controls.
5. Summary evidence changes may update the summary panel, but cannot redraw the editor,
   chart, or other evidence panels.
6. Status-only summary changes do not replace already-rendered table markup.
7. Long structural refits retain accessible blocking feedback.
8. The implementation remains framework-free vanilla JavaScript.

## Non-goals

- A virtual DOM, frontend framework, Tailwind migration, Canvas, or WebGL renderer.
- Incrementally diffing every SVG path after a genuine geometry edit.
- Hiding real model-refit latency or making asynchronous evidence appear synchronous.
- Changing statistical, fitting, or metric semantics.

## Rendering ownership

Each renderer owns one DOM boundary and subscribes only to its required state.

| Boundary | Inputs | Permitted DOM writes |
| --- | --- | --- |
| App view | active editor/report view | editor/report visibility only |
| Term picker | term catalogue, active term | term `<select>` only |
| Chart | chart generation, active term, chart view controls | chart SVG and chart-local palette |
| Selection | active term selection and impact | point classes/radii, selection bounds, palette position, context selection copy |
| Tool/context controls | mode, active term capabilities, local display state | tool rail and context controls only |
| History/app bar | authoritative history | history panel and undo/redo state only |
| Metrics | metrics evidence state | metrics panel only |
| Summary | summary evidence state and payload | summary status plus summary panel only |
| Report | active report and report evidence | report panel only |
| Global busy state | blocking structural mutation only | inert state, progress overlay, focus restoration |

The existing monolithic `render()` fan-out is replaced by named renderers and selector
subscriptions. Equality functions compare the smallest semantic input for each boundary,
so a new snapshot object alone is not a reason to rewrite unrelated DOM.

## Mutation behaviour

### Ordinary mutations

Selection, term switching, point/handle edits, control-count changes, categorical order,
and ordinary curve operations retain the mutation concurrency guard but are non-blocking
visually. They do not show the global overlay, make the editor inert, or move focus.
The current chart or optimistic preview remains visible until the authoritative response
arrives. The response then notifies only renderers whose semantic inputs changed.

If an ordinary request is slow, existing content stays usable as visual context while
duplicate mutations remain rejected by state. Panel-local status text may indicate work;
the full-shell overlay is not used.

### Structural mutations

Collapse/refit, ungroup/refit, and restore/refit remain explicitly blocking. Their existing
accessible overlay, inert regions, elapsed time, and focus restoration remain in place
because these operations genuinely take long enough to require modal progress feedback.
When the response commits, chart, history, summary, and evidence panels update through
their separate subscriptions rather than one application-wide render pass.

### Summary and evidence

Evidence scheduling remains asynchronous. An evidence status transition updates status,
freshness, retry, and `aria-busy` attributes without replacing the confirmed content.
Only a new semantic summary payload replaces the summary panel contents. Metrics and
reports follow the same panel isolation rule.

## Snapshot ordering and chart generation

`model_revision` tracks prediction/evidence semantics, not every render change. For
example, control-count and categorical display-order changes can alter SVG geometry at the
same model revision. Multiple tabs can also receive responses out of network order.

The Python widget therefore publishes two monotonic values:

- `state_generation`: advances for every complete state snapshot and orders responses;
- `chart_generation`: advances whenever chart-owned geometry or display structure changes,
  but not for selection-only changes.

The frontend never commits a snapshot older than its current `state_generation`.
Selection responses use the incremental path only when active term and
`chart_generation` still match; otherwise they use a full chart commit. Optional-field
fallbacks keep unit fixtures and older serialized snapshots readable during the migration.
No large arrays are duplicated or deeply compared to make this decision.

## Selection paint order and narrow layouts

Data points live in a stable chart-local point layer below a stable legend layer.
Incremental selection moves existing selected point nodes to the end of the point layer,
restoring the full renderer's `unselected < selected < legend` paint order without
recreating the nodes. Supplemental selected points are inserted into that same layer and
remain subject to the existing marker cap.

The action palette wraps at narrow widths. Its measured border box and interactive
contents remain within the chart shell, its 40 by 50 pixel direct targets remain intact,
and every visible button remains pointer- and keyboard-operable. Collision placement uses
the resulting wrapped dimensions.

## Error handling

- A failed selection keeps the provisional visual until recovery resolves, then patches
  selection state without rebuilding chart geometry when generations match.
- A stale response clears its own preview/busy state but cannot rewind newer state.
- A recovery snapshot with a different chart generation takes the normal full-chart path.
- Evidence errors retain confirmed panel contents and update only local status/retry UI.
- Structural failures dismiss the blocking overlay and restore focus as before.

## Acceptance criteria

Browser tests will prove:

1. selection, handle release, term change, and ordinary edit operations keep the global
   overlay hidden, keep editor regions non-inert, and never focus the busy announcement;
2. a held structural refit still displays the overlay and restores launcher focus;
3. no action presents an animation frame without an edited chart path;
4. selection success, no-op, and failure preserve chart geometry node identities;
5. selected data points paint above unselected data points and below the legend;
6. supplemental points remain below the legend and marker counts remain bounded;
7. every visible palette control fits and hit-tests correctly at a 360-pixel viewport;
8. same-model-revision chart changes force a chart render, while stale complete snapshots
   cannot rewind state;
9. chart changes do not mutate summary, metrics, report, or unrelated shell DOM;
10. summary status-only updates preserve summary table identities, and new summary payloads
    mutate only the summary boundary.

Frontend unit tests will cover selector equality, mutation blocking classification,
generation ordering, chart-generation fallback, and summary payload identity. Focused
Python tests will cover widget generation semantics. The complete frontend, browser,
editor backend, lint, strict documentation, and wheel checks remain final release gates.

## Maintainability

This design removes implicit fan-out instead of introducing a framework. The store remains
the single state source, each renderer has one explicit input contract, and Python supplies
two integer generations instead of requiring JavaScript to hash or compare large payloads.
The resulting code can be followed panel by panel by a Python-oriented contributor without
learning a frontend framework or virtual-DOM lifecycle.
