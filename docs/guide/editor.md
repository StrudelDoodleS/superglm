# Editing a Fitted Model

The SuperGLM editor is a compact analyst workspace for reviewing and adjusting fitted
one-dimensional effects. Python owns the fitted model, edit history, summaries, and evidence; the
browser provides a fast visual preview and sends completed actions back to Python.

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
drawer; in a short window the workspace scrolls instead of shrinking the plot into an unusable
strip.

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
- **Straighten selection** interpolates the selected relativities between their first and last
  points.
- Increasing and Decreasing apply anchored monotonic constraints.
- Level left, Average, and Level right flatten the selection to the named reference value.
- Snap highest and Snap lowest flatten to the selected extreme.

The icons remain compact so the plot stays large. Their delayed hover and immediate focus
explanations describe the action before it is run.

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

Undo and Redo are visible in the application bar and expose their keyboard shortcuts. They operate
on confirmed Python history, not on an uncommitted pointer preview.

If an edit request fails, the browser restores the last confirmed Python state and shows a
persistent message. Choose Retry to repeat that action after recovery or Dismiss to keep the
restored state. A failed metric, summary, or report refresh does not undo a valid model edit; only
that evidence panel becomes Stale and offers Retry.

## Evidence Freshness

The metric strip, summary, and reports distinguish these states:

- **Current** describes the displayed model revision.
- **Updating** retains the last confirmed values while Python computes the new revision.
- **Stale** retains those values after a failed refresh and must not be read as current evidence.
- **Error** means that panel has no confirmed value to retain.

Late responses for an older model revision or request sequence are ignored. Validation and test rows
remain in Python; the browser receives aggregate metrics rather than a copy of the evaluation data.

## Inspector and Help

Summary shows the current in-force model, History shows confirmed edits, Advanced contains
infrequent curve and diagnostic controls, and Help lists modes, gestures, operations, and shortcuts.
On a narrow screen these panes share a dismissible drawer. Escape closes a popover or drawer and
returns focus to its launcher.

## Save and Export

Save writes an edited model artifact to a chosen directory. Download writes the same artifact
through the browser when direct filesystem selection is unsuitable. The exported model uses
Python's current authoritative edit revision; display-only zoom, mode, grouping view, and truncated
tick text are not written into the model.

## Keyboard Shortcuts

- Use Tab to reach application tabs, the tool rail, context controls, the SVG action palette, and
  the inspector.
- Use arrow keys inside tab lists and the mode rail; Home and End jump to the first and last item.
- Use Enter or Space to activate a focused control.
- Use Escape to close the current popover, Help drawer, inspector drawer, or dialog.
- Use the displayed Undo/Redo shortcut for confirmed curve edits.

Pointer editing remains the primary high-density curve workflow. Full per-point keyboard editing
and an alternate editable data table are separate future accessibility enhancements.
