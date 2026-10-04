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
    train_data=(X_train, y_train, w_train),
    validation_data=(X_val, y_val, w_val),
)
session.widget()
```

`superglm.editor.edit(model, ...)` takes the same arguments and returns the same kind of session.

Each data argument is a tuple `(X, y)`, `(X, y, sample_weight)` or `(X, y, sample_weight, offset)`:

- **`train_data`** is the data the model was fit on. Refit, the rating table and Final fit use it.
  Without it they use the fit data the model kept, if it kept any.
- **`validation_data`** is scored on the metric strip and in the Validation report. Final fit also
  trains on these rows.
- **`test_data`** is scored in the Validation report. Final fit never trains on it.
- **`cv`** is a `cross_validate` result for the Cross-validation tab, and **`cv_data`** holds the
  rows its folds were drawn on. See [Cross-validation](#cross-validation).
- **`terms`** limits the editor to the named features. Leave it out to edit every feature.

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

- **Select** picks points or levels:
  - Click a point to select it.
  - Shift-click another point to select every point between the two, by position along the axis,
    whatever their heights.
  - Ctrl-click (Cmd-click on a Mac) adds or removes one point, on any term.
  - The point clicked last, with or without Ctrl or Cmd, is where the next Shift-click starts.
  - A click on the curve between two points selects the nearest point. A click on empty space
    changes nothing.
  - Drag a box to select the points inside it. **Select all** selects the whole term.
- **Move** drags selected relativities. The curve previews immediately and Python confirms the edit
  when the pointer is released.
- **Zoom** drags a box. The mouse wheel zooms around the pointer in every mode, Shift-drag or a
  middle-button drag pans, and Home restores the fitted extent.
- **Handles** edits a spline through fixed-x control handles. In Handles mode, when the fitted
  term exposes its basis, **Contrib** shows each basis function's contribution and **Build**
  animates them into the spline. Settings sets how long Build takes.

The active mode is always visible in the mode switch at the left of the chart's toolbar. Hover
briefly over an icon, or focus it with the keyboard, to see its name and shortcut.

## Ordered Categorical Splines

An ordered categorical fitted with a spline basis, such as
`OrderedCategorical(order=..., basis=Spline(...))`, is drawn as its spline, with a dot on each
level. Special levels (`specials=`) are drawn as separate dots.

- **Handles** works as it does on a numeric spline. Each handle is one spline coefficient, and
  moving it sets every level to the spline the handles draw, so a handle can sit off the curve.
- Special levels do not move with the handles.
- **Contrib** and **Build** show the spline's basis between the levels.
- Handles is off while some of the term's levels are collapsed into a group.
- Handles is off while the term holds a shaped range.

When Handles is off, hover over it to see the reason.

## Curve Selection Operations

The floating palette acts on the current selection:

- Increase or decrease moves the selected relativities by five percent.
- Smooth reduces local variation while respecting adjacent unselected values.
- **Straighten selection** interpolates the selected relativities between their first and last
  points.
- Increasing and Decreasing replace the selection with the closest monotone curve,
  weighted by exposure. It is not tied to the points either side, so it can
  leave a step at an edge.
- Level left, Average, and Level right flatten the selection to the named reference value.
- Snap highest and Snap lowest flatten to the selected extreme.

The icons remain compact so the plot stays large. Their delayed hover and immediate focus
explanations describe the action before it is run.

## Waiting Changes and Refit

Collapse, Ungroup, Set reference and the four shapes (Flat, Line, Quadratic and Cubic) change the
structure of a term, so the model must be refit before they take effect. They do not refit one by
one: each change waits, and **Refit** in the application bar applies every waiting change in one
fit.

While changes wait:

- the chart draws them over the last refit's curve: a waiting group is named on a dashed bracket
  under the axis, with its members' bars dashed in the group's colour, and a waiting range is a
  dashed box tagged "waiting for refit";
- the curve, the metrics and the summary still describe the last refit;
- the feature list marks each term that has a waiting change with an amber dot;
- the Refit button shows how many changes wait;
- the status line under the chart starts with the count, for example "2 changes waiting for
  refit".

Choose **Refit**, or press R, to apply them. The refit is one step in the history:

- Hand edits on terms whose structure did not change are put back on the refitted model. Putting
  them back is one more history entry, "Hand edits carried over: DrivAge, VehAge", and Undo of that
  entry returns those terms to the refitted curves.
- Hand edits on a term that a waiting change restructures are dropped. Undo brings them back.
- If the fit is refused, the model and the waiting changes stay as they were, and the message
  says to undo the last waiting change and try again.

To refit after every change instead, turn on **Refit after every structural change** in Settings.
Each change then refits straight away, and one Undo takes it back. The heading over these icons in
the selection palette reads **Structure** while changes wait for Refit, and **Refit** while each
one refits at once.

From Python, `session.stage_structural(...)` makes a change wait and `session.refit_pending()`
applies every waiting change; choose **Refresh from Python** to see them in the browser.

## Group and Ungroup Categorical Levels

Expanded and Collapsed display modes change only how an existing fitted grouping is drawn. They do
not rename levels or change the fitted model. **Groups shown as**, in Settings, chooses the mode a
term opens in.

Select levels and choose **Collapse** to combine them into one group, named after its members, for
example `B10+B11`. Select grouped levels and choose **Ungroup** to separate them again. Both wait
for Refit; see [Waiting Changes and Refit](#waiting-changes-and-refit).

With **Keep the reference level when collapsing** on in Settings, which is the default, the
reference stays where it is:

- Collapsing other levels leaves the reference level as it is.
- Collapsing the reference level into a group makes that group the reference.
- Ungrouping some levels out of the reference group leaves the reference with the levels that stay
  grouped.

The reference chip then reads "kept". With the setting off, a collapse or ungroup lets the term
choose its reference again by its own rule, most exposed or first, so the reference can move to
another level.

Long category names may appear shortened with an end ellipsis on the x-axis. This is display-only.
Hover or focus the tick (or inspect the point tooltip) for the complete value; selection, grouping,
history, exports, and saved models retain the exact original string.

## Set a Reference Level

The reference level is the one whose relativity is 1.00. A chip beside the term's kind and EDF
names it and says how it was chosen: most exposed, first, pinned, or kept through a collapse or
ungroup.

To pin a different level, select exactly that one level and choose **Set reference** in the
selection palette. It waits for Refit, and until then the chip shows the waiting choice, for
example "reference B3 · waiting".

- Predictions stay the same unless the model has a selection penalty.
- If the term rates unseen levels at the reference, their rate moves with it.
- In the Collapsed display, selecting a whole group pins the group.
- Special levels of an ordered term cannot be the reference.
- A term used by an interaction cannot change its reference.

## Where New Levels Go

When a model predicts, it can meet a level its fit never saw, such as a brand added after the fit.
On a categorical term, **New levels →** in the bar above the chart chooses what prediction does
with such a level:

- **Refuse** stops the prediction with an error that names the level. This is the default.
- **Reference** rates it at the reference level, relativity 1.00, and warns, naming the level.
- A group, listed by its name, gives it that group's relativity, and warns, naming the level.

The choice changes predictions only. Nothing refits, the curves stay as they are, and Undo takes
the choice back. History lists it as, for example, "New levels → B10+B11". The Python model export
and the Structure (JSON) export keep it.

- The control is offered on unordered categorical terms only.
- Only groups of two or more levels are listed.
- It is disabled while a change to the same term waits for Refit. Refit or undo that change first.
- It is disabled on a term used by an interaction.
- A collapse or ungroup that would remove the group new levels go to is refused. Choose where new
  levels go first, then make the change.

**A good choice is a group of thin levels.** Pool the term's thin levels into one group and send
new levels there. A new level has no rows of its own, and the pooled group's relativity is
estimated from all the thin levels' rows together. scikit-learn's `OneHotEncoder` follows the same
rule: with `handle_unknown="infrequent_if_exist"`, an unknown category maps to the infrequent
category, which pools the categories rarer than `min_frequency`, if it exists
([scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)).
In the editor, select the thin levels, choose **Collapse** and **Refit**, then choose the new group
under **New levels →**. In Python you can name the group yourself, such as "Other"; see
[Unseen levels at predict time](../how-to/specify-features.md#unseen-levels-at-predict-time).

A `RandomEffect` term needs no choice: it already gives a level it never saw the population
average.

## Shape a Range

A shape pins part of a spline term to a simple polynomial while the rest of the term stays the
fitted smooth. It works on numeric spline terms and on ordered terms with a spline basis.

1. In Select mode, select points, or bands on an ordered term. The range runs from the first
   selected point to the last, so a Shift-click span or points picked with Ctrl-click both work.
2. Choose one of the four shape icons in the selection palette: **Flat**, **Line**, **Quadratic**
   or **Cubic**. The toggle beside them sets how the curve meets the range at each edge:
   **Tangent** (the default) leaves the shape along its slope, **Corner** lets the slope change
   there. The choice is remembered.

The shape waits for Refit, drawn as a dashed box tagged with its shape and "waiting for refit".
After the refit the range is drawn as a light band labelled with its shape; hover over it to see
the shape and its edges.

- On a numeric term the range runs from the first to the last selected point, widened to a round
  value at three significant figures of the fitted range, and never past its ends.
- When no plotted point lies between the selection and a shaped range beside it, the new range
  starts or ends exactly on that range's edge, so the two meet.
- On an ordered term the range covers whole bands and is drawn from the first band to the last.
  Two ranges can share an edge band.
- The curve stays continuous at each edge of the range. With Tangent it also keeps its slope
  there; with Corner the slope may change.
- Flat fits a level to the range. To hold the curve at its value at a range's edge instead of
  fitting a level, use **Level from left** or **Level from right**. That is an edit, not a refit.
- To fit one polynomial over the whole axis, choose Select all and then a shape.
- A term can hold several ranges, each added as its own change.
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

## Rating-Table Preview

The **Chart / Table** switch above the chart shows the current term's block of the Excel rating
table in place of its curve. It is built by the same code as **Export > Excel rating workbook**, on
the same data, so it shows what the workbook will hold: the same rows, number formats and note.

- It follows every edit and every Refit.
- Like the curve, it shows the last refit: changes waiting for Refit are not in it.
- It needs the training data: `train_data`, or fit data the model kept. Without it the table says
  so.
- Interaction terms are not shown.
- When the export refuses the model, the table shows the reason instead.

## Undo, Redo and Revert

**Undo** and **Redo**, in the application bar, walk one history: every manual edit, every waiting
change and every applied step (Refit, revert and a **New levels →** choice) in the order you made
them, whichever term is on screen. Each popover names what it would undo or redo, for example
*Undo: Line 30–45 in age* or *Undo: shift age*. They act on confirmed Python history, not on an
uncommitted pointer preview.

- Undo takes back a waiting change without refitting, and Redo puts it back.
- Undo after a Refit brings its changes back as waiting. Redo applies the Refit again without
  fitting.
- Undoing a step puts back the model, the curves, the manual edits, the level order and the
  selection from before it. Redo puts the step back. Neither refits.
- A new edit or change clears whatever was waiting to be redone.
- A distribution re-profile starts the history afresh; Undo does not reach past it.
- A re-profile waits until nothing waits for Refit. Refit or undo the waiting changes first.

**Revert to original model**, next to Undo and Redo, goes back to the model the editor was opened
with. It also undoes a distribution re-profile and clears the waiting changes. It is one more step,
so Undo brings back everything it cleared.

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

The inspector beside the chart has four tabs: Summary, History, Settings and Help. On a narrow
screen they share a dismissible drawer. Escape closes a popover or drawer and returns focus to its
launcher.

### Summary

Summary shows the current model: its family, link and method, four tiles for deviance, AIC, BIC
and total EDF, and a section for each term.

- The search box at the top keeps the terms and levels whose names contain the text, ignoring
  case, highlights the matches and counts what it found. Escape clears it.
- The full summary below the table is not searched.
- **All**, **Edited** and **Waiting** show every term, the terms with hand edits, or the terms with
  changes waiting for Refit.
- Summary follows the chart: the chart's term opens and scrolls into view, and the others fold to
  one line with their kind, EDF and any waiting changes. Open any of them until the chart shows
  another term. A search opens every term it finds.
- A spline's folded line also shows the p-value of its whole-term test. A categorical term has no
  whole-term test, so its line shows none.

### History and notes

History lists the session newest first, in three parts: **Waiting for refit**, **Applied** and
**Undone**. Each entry has a short id, the time it was made and a message written for it, such as
"collapse B10 + B11 in VehBrand" or "Refit · 2 changes".

- **Undo takes this** marks the entry that Undo takes next.
- The muted entries under **Undone** are what Redo would put back, top one first.
- The pencil on an entry adds a note saying why. Enter, or leaving the field, saves it; Escape
  keeps the old one.
- A note stays with its entry through Undo and Redo.
- Notes are saved with the exported Python model; see [Export](#export).
- A distribution re-profile starts the list afresh, as it does for Undo.

### Settings

Settings, behind the gear icon, holds preferences kept in this browser:

- **Refit after every structural change**, off by default: each change refits at once instead of
  waiting for Refit.
- **Keep the reference level when collapsing**, on by default; see
  [Group and Ungroup Categorical Levels](#group-and-ungroup-categorical-levels).
- **Follow the browser's light or dark setting**, on until you flip the theme switch.
- **Groups shown as** Expanded or Collapsed when a term opens.
- **Build animation**: how long Build takes.
- **Request timings**: how long the last refit and each panel took.

Where the browser blocks storage, as some private windows do, the choices last until the page
closes.

### Help

Help lists the modes, gestures, operations and shortcuts, in the words the icons' hover text uses.

## Theme

The **DAY / NIGHT** switch in the application bar sets the light or dark theme. Until you flip it,
the editor follows the browser's light or dark setting, which inside a notebook is not always the
notebook's own theme.

- A flipped switch keeps its theme through a reload of the page.
- **Follow the browser's light or dark setting**, in Settings, hands the theme back to the browser.
- When the operating system is set to reduce motion, the switch changes without its animation.

## Export

Choose **Export** in the application bar, then one of four formats:

- **Python model** downloads a fitted `.joblib` artifact or writes it to a kernel path. SuperGLM
  loads the serialized artifact back before releasing it and, when evaluation rows are available,
  compares predictions on a bounded sample.
- **Excel rating workbook** writes deployment-oriented rating tables. This export needs
  `train_data` or fit data retained by the model. Its Model Summary sheet contains typed
  `ModelOverview` and `TermInference` tables, so numbers remain numeric Excel cells rather than one
  large formatted text block.
- **Final fit model** is the latest Final fit on all rows, from the Cross-validation tab. It is
  offered while the model has not changed since that fit.
- **Structure (JSON)** writes each term's groupings, reference, shaped ranges and where new levels
  go, without coefficients or hand edits. See [Structure Files](#structure-files).

The Python model and the workbook include your hand edits. Every export except Final fit model is
built from the last Refit:

- Changes still waiting for Refit are not included, and the dialog says how many.
- Display-only zoom, mode, grouping view, and truncated tick text are not written into the model.
- The Python model carries the history, notes included, as `model._editor_history`: one record per
  entry, with its id, time, operation, term, message, note and status.

## Structure Files

A structure file records the structural decisions made on a model, without its coefficients, so
they can be built into another model and fitted on new data, such as next year's. For each term it
holds:

- the groupings of its levels;
- the reference level;
- the shaped ranges, with their degree and join;
- where new levels go.

Choose **Export > Structure (JSON)**, or call `session.export_structure("structure.json")` from
Python. The file is JSON with sorted keys, so two versions compare cleanly in a diff.

Build it into a model and fit:

```python
from superglm import read_structure

structure = read_structure("structure.json")
next_model = structure.apply(model, X=X_next)
next_model.fit(X_next, y_next, sample_weight=w_next)
```

`apply` returns an unfitted copy of `model` with those decisions in its features. The model you
pass is left unchanged, and features the file does not name are copied as they are. The copy's
penalties are the ones `model` was declared with, even when `model` is fitted: a
`selection_penalty="auto"` is calibrated again on the new data, and smoothing that `fit_reml`
estimated is not carried over, so fit the copy with `fit_reml` to estimate it again.

- Pass `X=`, the data you will fit on. A level of a grouped term that the file does not list then
  goes where the file sends new levels, with one warning per term.
- Without `X=`, a level of a grouped term that the file does not list is refused when the model is
  fitted.
- With `X=`, a spline whose boundary comes from the data is widened to hold a shaped range that
  reaches past the new data's values, with one warning.
- A range the spline cannot take on the new data is refused, naming the feature and the range: by
  `apply` when you pass `X=`, otherwise by the fit.
- A P-spline or natural spline with a shaped range is rebuilt as a B-spline, as in the editor.
- A feature the model does not have is refused, by name.
- A feature that is another kind of term in the model is refused, by name.

Each refusal from `apply` is a `superglm.StructureError` whose message names the feature.

## Cross-validation

Pass a `cross_validate` result as `cv=` to fill the **Cross-validation** tab:

```python
from sklearn.model_selection import KFold
from superglm import cross_validate
from superglm.editor import EditorSession

cv = cross_validate(
    model_template, X_train, y_train, cv=KFold(5, shuffle=True, random_state=7),
    sample_weight=w_train, scoring=("deviance", "gini", "nll"), return_estimators=True,
)
session = EditorSession.from_model(
    model, train_data=(X_train, y_train, w_train),
    validation_data=(X_val, y_val, w_val), cv=cv,
)
session.widget()
```

- **`return_estimators=True`** keeps each fold's model, which the tab needs to draw every term's
  relativities by fold. Without it the tab shows the fold scores and says how to get the curves.
- **`cv_data=(X, y, sample_weight)`** gives the rows the folds were drawn on. Without it the editor
  uses `train_data` when its row count matches the folds.

The tab shows:

- a header naming the folds, the splitter, the rows and where the result came from;
- **Performance across folds**: each score's mean, with one dot per fold;
- **Folds**: each fold's train and test rows, scores, EDF, fit time and whether it converged;
- **Relativities across folds**: every term, least stable first, with a search box. *spread* is
  the mean distance of a fold's curve from the fold average, and *min r* the lowest correlation of
  a fold with it.

Each relativity chart draws the levels in the model's order, with every curve re-centred on its
exposure-weighted mean. Hover over or focus a fold in the chart's legend to pick it out.

Two buttons run jobs in the background, each with a **Cancel** button while it runs:

- **Run CV on current model** refits the folds the result came from, with your hand edits put
  back as you set them, and it waits while changes are waiting for Refit. Its scores are shown
  beside the supplied ones, and its fold curves replace the supplied curves. A hand-edited term is
  marked **held**: it is the same curve on every fold.
- **Final fit on all rows** uses the train and validation rows and leaves the test rows out. It
  puts your hand edits back too. Its result appears on the **Final Fit** tab, and Export offers it
  as **Final fit model**.

Run CV scores the folds with whichever of deviance, Gini and NLL the supplied result scored, or
with all three when it scored none of them. Both jobs choose the penalties as the opened model
declares them: a `selection_penalty="auto"`, or smoothing that `fit_reml` estimates, is chosen
again on each fold's rows and on the Final fit's rows, after a Refit too. A job whose model changed
while it ran is not kept; run it again.

Run CV is disabled, with the reason on hover, when it cannot use the rows:

- `cv_data` has a different row count from the data the folds were drawn on.
- `train_data` stands in for `cv_data` and has a different row count.
- The features, response, weights or offsets differ from the data the folds were drawn on. Pass
  the same rows, in the same order.

A result from an older superglm version records no fingerprint of its data, so only its row count
is checked, and the tab says so. The older `cv_report=` argument of `EditorSession.from_model`
still works: its report is shown on the Validation tab.

## Keyboard Shortcuts

- Use Tab to reach application tabs, the feature list, the chart toolbar, the SVG action
  palette, and the inspector.
- Use arrow keys inside tab lists, the mode switch and the join toggle; Home and End jump to the
  first and last item.
- Use Enter or Space to activate a focused control.
- Use Escape to close the current popover, Help drawer, inspector drawer, or dialog.
- Use Ctrl/Cmd+Z to undo and Ctrl/Cmd+Shift+Z or Ctrl+Y to redo an edit, a waiting change or a
  step.
- Press R to refit the waiting changes.

Pointer editing remains the primary high-density curve workflow. Full per-point keyboard editing
and an alternate editable data table are separate future accessibility enhancements.
