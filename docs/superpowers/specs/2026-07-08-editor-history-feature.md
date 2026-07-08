# Editor Previous Curve And History Feature

## Status

Draft for review.

## Context

The editor currently shows the reference/original model as a grey dashed curve and the current
editable state as a blue curve. Manual edits are already stored as reversible `EditRecord` objects
inside `EditorSession.history`, with `before` and `after` link-scale values. Undo and redo are
stack-based and linear.

This feature keeps that linear history model. It adds a more Gam Changer-like visual comparison:
original model, previous edit, current edit, plus a compact edit-history tab beside the model
summary.

## Goals

- Show three curve states when available:
  - grey dashed reference/original model
  - orange previous edit state
  - blue current edit state
- Add an `Edit history` tab in the right panel next to `Model summary`.
- Show git-style short hashes for edits so history entries are easy to discuss and debug.
- Keep the history model linear for now.
- Preserve the current Python-owned state model: the browser remains a view/controller over JSON
  payloads.

## Non-Goals

- No branchable history tree in this version.
- No clicking old history entries to restore or checkout a state.
- No persistent history schema redesign unless existing session persistence already requires it.
- No model-DAG comparison workflow.

## Previous Edit Curve

For each term payload, Python will compute an optional `previous_y` array.

The value is derived from the latest committed `EditRecord` for that term:

1. Start from the term's current `edited_log_effect`.
2. Restore `record.before` at `record.indices`.
3. Convert the resulting link-scale values to relativity scale with `exp`.

If the active term has no matching history record, `previous_y` is omitted or set to `null`.

For grouped categorical collapsed display, `previous_y` is projected through the same grouped-display
logic as `y` and `original_y`. This keeps the orange line aligned with the visible collapsed buckets.

The chart legend will use:

- `original` or `original projection` for the grey dashed curve
- `previous edit` for the orange curve, only when present
- `current edit` for the blue curve

## Edit History Tab

The right-side panel will get local tabs:

- `Model summary`
- `Edit history`

`Model summary` remains the default.

The edit-history tab will show a compact linear list, newest first. Each item will include:

- short hash, for example `a13f9c2`
- term name
- operation name
- count of affected points
- compact operation parameters when useful

The current head is the newest item in `session.history`. When `session.redo_stack` is non-empty,
the tab shows those entries in a muted separate section labelled `Redo stack`, because they are
available but not part of the active edit chain.

## Hashing

Hashes should be deterministic for a given linear edit chain. Use a small helper that hashes:

- previous hash
- term name
- operation
- indices
- rounded `before` and `after` values
- JSON-safe params

The hash is display/debug metadata only. It is not a persistence or compatibility guarantee.

## Data Flow

The Python session remains authoritative:

```text
EditorSession.history
  -> history payload helper
  -> /state or /history JSON
  -> browser right-panel history renderer
```

The previous-curve flow is:

```text
EditorSession.history latest matching record
  -> payloads.py computes previous_y
  -> group_display.py optionally collapses previous_y
  -> chart.js renders orange line
```

## Frontend Boundaries

- `chart.js` owns drawing the orange line and legend entry.
- `main.js` owns tab switching and refresh orchestration.
- A small history renderer can live in a new `history.js` module to avoid bloating `main.js`.
- CSS additions should stay local to the right-panel tabs and history list.

## Testing

Add Python tests for:

- `previous_y` is absent with no edits.
- `previous_y` equals the term state immediately before the latest edit for that term.
- `previous_y` ignores edits to other terms.
- collapsed grouped display projects `previous_y` consistently.
- history payload hashes are stable and short.
- undo/redo updates active history and redo payloads.

Add static frontend tests where existing patterns allow:

- chart code references `previous_y` and the `previous edit` legend label.
- right panel contains model-summary/edit-history tabs.

## Risks

- A true tree should not be implied by the first UI. Use wording like `Edit history`, not `History
  tree`, until branch support exists.
- Hashes should not be presented as stable serialized IDs.
- Orange previous line can be visually noisy for tiny edits, so it should be hidden when unavailable
  and styled thinner than the current blue line.
