# Incremental Selection Rendering Design

## Problem

Selection-only gestures currently travel through the same full-state rendering path as model
changes. A no-op box selection with an already-empty selection still posts `indices=[]`, commits a
new snapshot object, calls the application-wide renderer, and clears and reconstructs the chart.

Chromium instrumentation confirms that one no-op drag:

- leaves the Python selection and model revision unchanged;
- sends one unnecessary `/select` request;
- removes and recreates every chart descendant (roughly 250 SVG nodes);
- replaces the edited path and every point between adjacent animation frames; and
- also rebuilds unrelated controls such as the term selector.

The resulting flash is deterministic and must not be treated as unavoidable browser latency.

## Goals

- Selection feedback appears immediately at pointer release.
- Selection-only work never reconstructs chart geometry, axes, paths, labels, or points.
- A semantically unchanged selection performs no request and no rendering work.
- Python remains authoritative for committed selection state.
- Failed selection synchronization restores the authoritative selection without redrawing the
  chart.
- Existing curve movement, control-handle previews, collapsed categorical mappings, recovery, and
  accessibility behavior remain intact.

## Non-Goals

- Replacing the complete SVG renderer with a general virtual DOM or keyed diff engine.
- Making model edits optimistic.
- Avoiding full redraws when model geometry, the active term, zoom, confidence intervals, grouping,
  or a structural refit genuinely changes the chart.

## Design

### Dedicated selection lane

Selection receives a dedicated browser-state lane rather than using the term-payload preview used
by curve movement. The lane contains the active term and a normalized, sorted list of source
indices. Selectors expose the provisional list while synchronization is pending and otherwise expose
the committed Python selection.

At pointer release, the interaction layer normalizes the computed source indices and compares them
with the currently displayed selection. Equal lists end the gesture immediately: the brush is
removed, no store update occurs, and `/select` is not called.

For a changed list, the browser commits a provisional selection first, producing immediate visual
feedback, and then sends the selection to Python. This provisional value is view state, not a model
revision and not evidence that Python accepted the operation.

### Selection-only remote commit

The action controller uses a selection-specific commit/recovery path. A successful `/select`
response updates the authoritative snapshot and clears the provisional selection, but marks the
commit as selection-only so the application does not invoke the full renderer. Other ordinary and
structural mutations continue through their existing commit paths.

If the response reports different model geometry, model revision, or active-term state than the
request began with, the optimization is abandoned and the normal full render runs. Selection-only
rendering must never conceal a genuine authoritative model change.

### Incremental SVG patch

The chart module exposes a focused selection updater operating on the existing SVG nodes and scale
metadata. It:

- toggles selected classes on existing point nodes;
- recreates only the small selection-bounds layer;
- repositions or hides the floating selection-operation menu; and
- leaves paths, axes, labels, exposure marks, point identities, and interaction bindings untouched.

The application layer separately refreshes selection-dependent context text and collapse/ungroup
button availability. It does not rebuild the term selector or unrelated panels.

### Failure and recovery

If `/select` fails, the action controller reconciles with Python as it does today, clears the
provisional selection, and incrementally patches the recovered authoritative selection. The existing
error/retry message remains visible. A full render is permitted only if recovery reveals an actual
term or model change.

Repeated selection submissions remain subject to the existing one-mutation-at-a-time rule.

## Accessibility

Selection counts and exposure text continue to update through the existing context/status nodes.
The selection-operation menu retains its focus and popover behavior. Incremental updates must not
replace the focused SVG point or toolbar controls, which also prevents the current focus loss caused
by full DOM replacement.

## Testing

Pure frontend tests cover selection normalization, semantic equality, provisional state, successful
commit, failed rollback, and the full-render fallback when authoritative geometry changed.

Real-browser regression tests assert:

1. An empty no-op box drag sends no `/select` request, preserves path and point identity, and leaves
   only the temporary brush mutation.
2. A changed selection appears before a delayed `/select` response, synchronizes to Python, and
   preserves path, point, axis, and term-option identities before and after confirmation.
3. A failed delayed selection rolls back styling and bounds without replacing chart geometry.
4. Expanded and collapsed categorical selections map display points to the correct source indices.
5. Existing move/control previews, selection operations, browser recovery, and accessibility suites
   remain green.

## Acceptance Criteria

- No visible selection flicker in Chromium for no-op, successful, or failed selection gestures.
- Zero edited-path or point replacements for selection-only changes.
- Zero `/select` requests for unchanged selections.
- Immediate changed-selection feedback while Python synchronization is pending.
- Python and browser selections agree after success or recovery.
- Full redraw remains available and correct for genuine geometry changes.
