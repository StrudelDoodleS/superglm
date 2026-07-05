# SuperGLM Editor Frontend

The editor frontend is a small same-origin browser app served by the Python
kernel. Python owns all authoritative model and edit state. The browser renders
JSON payloads, previews pointer interactions, and posts completed actions back
to Python.

## Request Boundary

`EditorWidget` starts a local FastAPI server on `127.0.0.1` and renders an iframe
URL with a per-widget token. Frontend requests send that token as
`X-SuperGLM-Editor-Token`; API routes reject requests without it.

The main data flow is:

```text
EditorSession
  -> session_payload()
  -> GET /state
  -> chart.js render
  -> user action
  -> POST route
  -> widget method
  -> EditorSession method
  -> fresh JSON redraw
```

## Files

- `api.js`: Fetch helpers. Adds the editor token header to JSON requests.
- `main.js`: App lifecycle and toolbar wiring. Owns current UI mode, loaded
  state, selected view, and refresh orchestration.
- `chart.js`: Pure SVG rendering from Python payloads. It returns scale metadata
  used by interactions.
- `interactions.js`: Pointer and keyboard state machine for selection, dragging,
  pan/zoom, handle edits, level reorder previews, and undo/redo shortcuts.
- `summary.js`: Summary panel, fixed-offset refit, collapse/ungroup refit, and
  distribution re-profile dialogs.
- `metrics.js`: Metric strip rendering.
- `reports.js`: Validation and final fit report tabs.
- `format.js`: Shared number formatting.

## Invariants

- Python stores link-scale effects.
- Browser displays inverse-link relativities.
- Browser previews can be temporary; committed changes must go through Python.
- Manual coefficient edits are not refitted unless explicitly converted to an
  offset workflow.
- Structural refits replace the in-force model copy and clear edit history.
- Display-only level order is saved separately from model coefficients.
