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

## Find a Feature

The feature list on the left names every term in the editor, grouped as the chart groups them,
with each term's kind and effective degrees of freedom. The current term is highlighted.

- Type in the search box to filter the list. Enter opens the first match and Escape clears the
  search.
- The arrow keys step between features; Enter or a click opens one.
- The toggle at the top collapses the list to a thin strip that names the current term, so the
  plot keeps its width. The editor remembers your choice.
- In a window narrower than about 1280 pixels, the list starts collapsed until you open it.

## Select, Move, Zoom, and Handles

- **Select** brushes points or levels. Shift-click toggles a point or level; Select all selects the
  complete active term.
- **Move** drags selected relativities. The curve previews immediately and Python confirms the edit
  when the pointer is released.
- **Zoom** drags a box. The mouse wheel zooms around the pointer and Home restores the fitted extent.
- **Handles** edits a spline through fixed-x control handles. Basis contributions and the Build
  animation are available from Advanced when the fitted term exposes them.

The active mode is always visible in the mode switch at the left of the chart's toolbar. Hover
briefly over an icon, or focus it with the keyboard, to see its name and shortcut.

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

Collapse selected levels and Ungroup selected levels are structural actions: SuperGLM refits the
model straight away, and Undo brings back the model and the manual edits from before the refit
(see Undo, Redo and Revert). The refit overlay reports elapsed time. When the fit returns, the plot
and summary change together; metrics may remain marked Updating briefly.

Long category names may appear shortened with an end ellipsis on the x-axis. This is display-only.
Hover or focus the tick (or inspect the point tooltip) for the complete value; selection, grouping,
history, exports, and saved models retain the exact original string.

## Set a Reference Level

The reference level is the one whose relativity is 1.00. A chip beside the term's kind and EDF
names it and says how it was chosen: most exposed, first, or pinned.

To pin a different level, select exactly that one level and choose **Set reference and refit** in
the selection palette. SuperGLM refits the model with that level as the reference.

- Predictions stay the same unless the model has a selection penalty.
- If the term rates unseen levels at the reference, their rate moves with it.
- In the Collapsed display, selecting a whole group pins the group.
- Special levels of an ordered term cannot be the reference.
- A term used by an interaction cannot change its reference.

## Shape a Range

A shape pins part of a spline term to a simple polynomial while the rest of the term stays the
fitted smooth. It works on numeric spline terms and on ordered terms with a spline basis.

1. In Select mode, select a continuous run of points, or of bands on an ordered term.
2. Choose one of the four shape icons in the selection palette: **Flat**, **Line**, **Quadratic**
   or **Cubic**. The toggle beside them sets how the curve meets the range at each edge:
   **Tangent** (the default) leaves the shape along its slope, **Corner** lets the slope change
   there. The choice is remembered.

SuperGLM refits straight away. The range is drawn as a light band labelled with its shape; hover
over it to see the shape and its edges.

- On a numeric term the range runs from the first to the last selected point, widened to a round
  value at three significant figures of the fitted range, and never past its ends.
- When no plotted point lies between the selection and a shaped range beside it, the new range
  starts or ends exactly on that range's edge, so the two meet.
- On an ordered term the range covers whole bands and is drawn from the first band to the last.
  Two ranges can share an edge band.
- The curve stays continuous at each edge of the range, but its slope may change there.
- Flat fits a level to the range. To hold the curve at its value at a range's edge instead of
  fitting a level, use **Level from left** or **Level from right**. That is an edit, not a refit.
- To fit one polynomial over the whole axis, choose Select all and then a shape.
- A term can hold several ranges, each added as its own step.
- A new range may not overlap one already shaped. Undo the old one, or choose a range outside it.
- Choosing a new shape on exactly the same range replaces the old shape.
- A Line needs at least two distinct values in the range, a Quadratic three and a Cubic four. On
  an ordered term each band is one value, and so is a collapsed group inside the range. When the
  selection holds too few, the icon is disabled and says so on hover.
- A binned fit (`discrete=True`) sees only the centres of its occupied bins, so those are the
  values it counts. Values that share a bin count once, so four distinct values may not be enough
  for a Cubic.
- The curve outside the ranges needs values too. If a range would leave too few between it and
  the end of the axis, or the next shaped range, SuperGLM refuses it and says so. Widen the range
  to reach the end or that range.
- A range cannot start or end inside a collapsed group. Ungroup the bands at its ends first.
- A range cannot take in a special level of an ordered term.
- A band at the edge of a shaped range cannot be collapsed into a group.
- A P-spline or natural spline term is refitted as a B-spline with the same knots and a derivative
  penalty, so the penalty can leave the shaped range alone. A natural spline's curve is then no
  longer held straight at the ends.

The icons are hidden on an unordered categorical term. On any other term that cannot take a shape
they are disabled, with the reason on hover:

- The term is not a spline, such as a linear term or an ordered term without a spline basis.
- The term is a cardinal cubic regression spline.
- The term has a shape constraint, such as increasing or convex.
- The term uses `select=True`.
- The term is used by an interaction.

The summary, the Python `summary()` and the workbook note that shaped ranges were chosen in the
editor from this data. Tests are conditional on them, so judge them on validation deviance.

## Undo, Redo and Revert

**Undo** and **Redo**, in the application bar, walk one history: every manual edit and every
structural step (collapse, ungroup, set reference, shape and revert) in the order you made them,
whichever term is on screen. Each popover names what it would undo or redo, for example
*Undo: Line 30–45 in age* or *Undo: shift age*. They act on confirmed Python history, not on an
uncommitted pointer preview.

- Undoing a step puts back the model, the curves, the manual edits, the level order and the
  selection from before it. Redo puts the step back. Neither refits.
- A new edit or step clears whatever was waiting to be redone.
- A distribution re-profile starts the history afresh; Undo does not reach past it.

**Revert to original model**, next to Undo and Redo, goes back to the model the editor was opened
with, which also undoes a distribution re-profile. It is one more step, so Undo brings back
everything it cleared.

## Refresh from Python

If you change the session in the notebook, for example collapse levels from Python, the browser
does not see it straight away. Choose **Refresh from Python** in the application bar to re-read the
session and redraw. Nothing refits.

## Recovery

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

Summary shows the current in-force model. History lists every manual edit and structural step of
the session in the order you made them, each step marked as one; a line marks the current
position, so Undo takes the entry just above it, and the muted entries below it are what Redo would
put back. Undoing a step moves it and the edits after it below the line; nothing is cleared. A
distribution re-profile starts the list afresh, as it does for Undo. Advanced contains infrequent
curve and diagnostic controls, and Help lists modes, gestures, operations, and shortcuts.
On a narrow screen these panes share a dismissible drawer. Escape closes a popover or drawer and
returns focus to its launcher.

## Export

Choose **Export > Python model** to download a fitted `.joblib` artifact or write it to a kernel
path. SuperGLM loads the serialized artifact back before releasing it and, when evaluation rows are
available, compares predictions on a bounded sample. The export always uses Python's current
authoritative edit revision; display-only zoom, mode, grouping view, and truncated tick text are
not written into the model.

Choose **Export > Excel rating workbook** for deployment-oriented rating tables. This export needs
explicit training data or fit data retained by the model. Its Model Summary sheet contains typed
`ModelOverview` and `TermInference` tables, so numbers remain numeric Excel cells rather than one
large formatted text block.

## Keyboard Shortcuts

- Use Tab to reach application tabs, the feature list, the chart toolbar, the SVG action
  palette, and the inspector.
- Use arrow keys inside tab lists, the mode switch and the join toggle; Home and End jump to the
  first and last item.
- Use Enter or Space to activate a focused control.
- Use Escape to close the current popover, Help drawer, inspector drawer, or dialog.
- Use Ctrl/Cmd+Z to undo and Ctrl/Cmd+Shift+Z or Ctrl+Y to redo an edit or a structural step.

Pointer editing remains the primary high-density curve workflow. Full per-point keyboard editing
and an alternate editable data table are separate future accessibility enhancements.
