// @ts-check

/**
 * @typedef {object} HelpEntry
 * @property {string} title
 * @property {string} body
 * @property {string} [shortcut]
 */

/** @type {Readonly<Record<string, Readonly<HelpEntry>>>} */
export const TOOL_HELP = Object.freeze({
  select: Object.freeze({
    title: "Select",
    body:
      "Click a point or drag a box to select curve values. Shift-click selects every point from the last one clicked; Ctrl/Cmd-click adds or removes one.",
    shortcut: "V",
  }),
  move: Object.freeze({
    title: "Move",
    body: "Drag a selected point or selection to change relativity.",
    shortcut: "M",
  }),
  zoom: Object.freeze({
    title: "Zoom",
    body: "Drag a box to zoom. The mouse wheel zooms in every mode.",
    shortcut: "Z",
  }),
  handles: Object.freeze({
    title: "Handles",
    body: "Edit spline control handles and inspect basis contributions.",
    shortcut: "H",
  }),
  knots: Object.freeze({
    title: "Knots",
    body:
      "Drag a knot along the axis to move it, click the axis to add one, or drag one below the axis to remove it. Each change waits for Refit.",
    shortcut: "K",
  }),
  help: Object.freeze({
    title: "Help",
    body: "Open modes, gestures, shortcuts, curve operations, refits, and exporting.",
    shortcut: "?",
  }),
});

// A structural change waits for Refit unless Settings says to refit after
// every one.
const WAITS = "It waits for Refit in the top bar, or refits at once when Settings says so.";
// Settings › Keep the reference level when collapsing, on by default.
const KEEPS_REFERENCE = "With Keep the reference level on in Settings, the reference stays where it is";

/** @type {Readonly<Record<string, Readonly<HelpEntry>>>} */
export const OPERATION_HELP = Object.freeze({
  shift_up: Object.freeze({
    title: "Increase selection",
    body: "Increase selected relativities by 5%.",
  }),
  shift_down: Object.freeze({
    title: "Decrease selection",
    body: "Decrease selected relativities by 5%.",
  }),
  smooth: Object.freeze({
    title: "Smooth selection",
    body: "Reduce local variation across the selected relativities.",
  }),
  linearise: Object.freeze({
    title: "Straighten selection",
    body: "Interpolate the selected relativities between their first and last points.",
  }),
  increasing: Object.freeze({
    title: "Make increasing",
    body: "The closest non-decreasing curve to the selection, weighted by exposure. It is not tied "
      + "to the points either side, so it can leave a step at an edge.",
  }),
  decreasing: Object.freeze({
    title: "Make decreasing",
    body: "The closest non-increasing curve to the selection, weighted by exposure. It is not tied "
      + "to the points either side, so it can leave a step at an edge.",
  }),
  level_left: Object.freeze({
    title: "Level from left",
    body: "Set selected relativities to the leftmost selected value.",
  }),
  average: Object.freeze({
    title: "Average selection",
    body:
      "Set selected relativities to their exposure-weighted mean (or their unweighted mean when exposure is unavailable).",
  }),
  level_right: Object.freeze({
    title: "Level from right",
    body: "Set selected relativities to the rightmost selected value.",
  }),
  snap_highest: Object.freeze({
    title: "Snap to highest",
    body: "Set selected relativities to the highest selected value.",
  }),
  snap_lowest: Object.freeze({
    title: "Snap to lowest",
    body: "Set selected relativities to the lowest selected value.",
  }),
  collapse_levels: Object.freeze({
    title: "Collapse",
    body: `Combine the selected levels into one group. ${WAITS} ${KEEPS_REFERENCE}, and a group that takes it in becomes the reference.`,
  }),
  ungroup_levels: Object.freeze({
    title: "Ungroup",
    body: `Separate the selected grouped levels. ${WAITS} ${KEEPS_REFERENCE}, and a reference group follows the levels that stay in it.`,
  }),
  make_special: Object.freeze({
    title: "Make special",
    body: `Take the selected levels of an ordered term off its curve, each with an estimate of its own, as specials= declares. ${WAITS} The reference stays on the curve. Export a structure to make the same levels special in code.`,
  }),
  return_to_curve: Object.freeze({
    title: "Back on the curve",
    body: `Put the selected special levels back on the curve, in their place. ${WAITS} A level declared special in the term's code has no place on the curve; change the declaration instead.`,
  }),
  set_reference: Object.freeze({
    title: "Set reference",
    body:
      `Pin the selected level as the reference (relativity 1.00). ${WAITS} Predictions stay the same unless a selection penalty is on. Unseen levels rated at the reference move with it.`,
  }),
  shape_flat: Object.freeze({
    title: "Flat",
    body: `Make the selected range flat; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
  shape_line: Object.freeze({
    title: "Line",
    body:
      `Make the selected range a straight line; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
  shape_quadratic: Object.freeze({
    title: "Quadratic",
    body:
      `Make the selected range a quadratic; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
  shape_cubic: Object.freeze({
    title: "Cubic",
    body: `Make the selected range a cubic; the rest of the curve stays smooth. ${WAITS} Undo takes it back.`,
  }),
});

/** @type {Readonly<Record<string, Readonly<HelpEntry>>>} */
export const STRUCTURE_HELP = Object.freeze({
  refit_pending: Object.freeze({
    title: "Refit",
    body: "Apply every waiting change in one fit. Hand edits on terms whose structure did not change are kept. Undo brings the changes back as waiting.",
    shortcut: "R",
  }),
  revert_to_original: Object.freeze({
    title: "Revert to original model",
    body: "Go back to the model the editor was opened with. Undo brings back everything it cleared.",
  }),
  refresh_from_python: Object.freeze({
    title: "Refresh from Python",
    body: "Re-read the Python session after changing it in the notebook.",
  }),
  new_levels: Object.freeze({
    title: "New levels →",
    body: "Where levels the fit never saw go when the model predicts. Refuse stops the "
      + "prediction with an error. Reference rates them at the reference level, relativity "
      + "1.00. A group gives them that group's relativity, with a warning naming them. "
      + "Nothing refits, and Undo takes the choice back. The Structure (JSON) export keeps it.",
  }),
});

/** @type {Readonly<Record<string, Readonly<HelpEntry>>>} */
export const CONTROL_HELP = Object.freeze({
  level: Object.freeze({
    title: "Level selected values",
    body: "Choose a reference value or average for the selected relativities.",
  }),
  snap: Object.freeze({
    title: "Snap selected values",
    body: "Set selected relativities to their highest or lowest selected value.",
  }),
});

/**
 * @typedef {object} HelpSection
 * @property {string} title
 * @property {readonly string[]} [keys]
 * @property {readonly string[]} [items]
 */

/** @type {readonly Readonly<HelpSection>[]} */
export const HELP_SECTIONS = Object.freeze([
  Object.freeze({
    title: "Modes",
    keys: Object.freeze(["select", "move", "zoom", "handles", "knots"]),
  }),
  Object.freeze({
    title: "Selecting points",
    items: Object.freeze([
      "Click a point to select it. Shift-click another to select every point between the two by position along the axis, whatever their heights.",
      "Ctrl/Cmd-click adds or removes one point, on any term. The point clicked last, with or without Ctrl/Cmd, is where the next Shift-click starts.",
      "A click on the curve between points selects the nearest point; a click on empty space changes nothing.",
      "Drag a box to select the points inside it. Shift-drag pans instead.",
    ]),
  }),
  Object.freeze({
    title: "Selection operations",
    keys: Object.freeze(Object.keys(OPERATION_HELP)),
  }),
  Object.freeze({
    title: "Shaped ranges",
    items: Object.freeze([
      "Select points or bands on a spline term, then choose Flat, Line, Quadratic or Cubic. The range runs from the first selected point to the last, gaps included, and is pinned to the shape fitted to the data across it at the next Refit; the rest of the term stays the fitted smooth.",
      "At each edge the curve leaves the shape along its slope (Tangent). The toggle beside the shape icons chooses Corner instead, where the slope may change at the edge. Select all, then a shape, for one polynomial over the whole axis.",
      "Flat fits a level. To hold the curve at its value at a range's edge instead of fitting a level, use Level from left or Level from right. That is an edit, not a refit.",
      "A new range may not overlap one already shaped, and one selected right beside it meets it. The same range with a new shape replaces it. Undo takes back the latest shape.",
      "A P-spline or natural spline term becomes a B-spline with a derivative penalty so its penalty can skip the shaped range; a natural spline's ends are then no longer held straight.",
    ]),
  }),
  Object.freeze({
    title: "Knots",
    items: Object.freeze([
      "Small ticks under the x-axis mark a spline term's knots in every mode. The Knots tool (K) turns them into diamonds on the axis that you can move.",
      "Drag a knot along the axis to move it. It may pass its neighbours; one dropped too close to another settles on the nearest free spot.",
      "Click the axis where the dashed diamond shows to add a knot there. Drag a knot below the axis, onto Drop here to remove the knot, to remove it. A spline keeps at least one knot.",
      "Click a knot to select it. The arrow keys nudge it one step, ten with Shift; Delete removes it and Escape lets it go.",
      "The count and Placed by above the chart place every knot again by a rule: even spacing, quantiles of the values, quantiles of the rows, or tempered quantiles, which weight each value by its row count to the power alpha. Knots you placed by hand are dropped; Undo brings them back. Reset knots goes back to the knots declared in code.",
      "Each change waits for Refit, or refits at once when Settings says so. The knots it places are amber, and the knots it moves or removes stay as grey ghosts until Refit. Hand edits on the term are dropped by the refit, as for every structural change; Undo brings them back.",
      "On a P-spline (kind=\"ps\"), moved knots switch the term to the general P-spline penalty, which measures wiggliness on unevenly spaced knots, so the smoothing stays sound.",
      "An ordered term takes at most one knot fewer than the levels on its curve, and its knots can sit between levels. The tool is off where a term's knots cannot change; hovering it says why.",
    ]),
  }),
  Object.freeze({
    title: "Ordered splines",
    items: Object.freeze([
      "An ordered categorical fitted with a spline is drawn as its spline, with a dot on each level. Special levels are separate dots.",
      "Handles edit it like a numeric spline: each handle is one spline coefficient, and moving it sets every level to the spline the handles draw, so a handle can sit off the curve. Special levels do not move. Contrib and Build show the spline's basis.",
      "Handles are off while levels are grouped or a band is shaped; the Handles tool says which.",
    ]),
  }),
  Object.freeze({
    title: "Rating table",
    items: Object.freeze([
      "The Chart / Table switch above the chart shows the current term's block of the Excel rating table instead of its curve, with the workbook's number formats and note. It is built by the same code as the export, on the training data, and follows every edit.",
      "Interactions are not shown. A model the export refuses shows the reason instead of a table.",
    ]),
  }),
  Object.freeze({
    title: "Features",
    items: Object.freeze([
      "Type in the search box to filter the feature list; Enter opens the first match and Escape clears the search.",
      "Arrow keys step between features; Enter or a click opens one. The toggle collapses the list to a strip.",
    ]),
  }),
  Object.freeze({
    title: "Summary",
    items: Object.freeze([
      "The search box at the top of Summary keeps the terms and levels whose names contain the text, ignoring case, and counts what it found. Escape clears it. The full summary below the table is not searched.",
      "All, Edited and Waiting show every term, the terms with hand edits, or the terms with changes waiting for refit.",
      "Summary follows the chart: the chart's term opens and scrolls into view, and the others fold to one line with their kind, EDF and any changes waiting for refit. A spline's line also shows the p-value of its whole-term test; a categorical term has no whole-term test, so its line shows none. Open any term from its line until the chart shows another; a search opens every term it finds.",
    ]),
  }),
  Object.freeze({
    title: "Model structure",
    keys: Object.freeze(Object.keys(STRUCTURE_HELP)),
  }),
  Object.freeze({
    title: "Navigation",
    items: Object.freeze([
      "Mouse wheel: zoom",
      "Shift-drag or middle-drag: pan",
      "Home: reset zoom",
    ]),
  }),
  Object.freeze({
    title: "Undo, Redo and Revert",
    items: Object.freeze([
      "Undo and Redo walk one history of edits and structural steps, in the order made, whichever term is shown. Their popovers name what they would undo or redo.",
      "Undoing a step brings back the model, curves, edits and selection from before it, without refitting.",
      "Undo takes back a waiting change without refitting. Undo after a Refit brings its changes back as waiting.",
      "Ctrl/Cmd+Z: undo",
      "Ctrl/Cmd+Shift+Z or Ctrl+Y: redo",
      "Revert to original model is one more step: Undo brings back everything it cleared.",
      "History, in the inspector, lists the session newest first: the changes waiting for refit, then the applied ones. Undo takes the entry marked Undo takes this; the muted entries under Undone are what Redo would put back.",
      "Each step has a short id and the time it was made. The pencil adds a note saying why; notes are saved with the exported Python model.",
    ]),
  }),
  Object.freeze({
    title: "Theme",
    items: Object.freeze([
      "The DAY / NIGHT switch in the application bar sets the theme. Until you flip it, the editor follows the browser's light or dark setting, which inside a notebook is not always the notebook's own.",
      "A flipped switch keeps its theme through a reload of the page. Follow the browser's light or dark setting, in Settings, hands the theme back to the browser.",
    ]),
  }),
  Object.freeze({
    title: "Settings",
    items: Object.freeze([
      "Settings, in the inspector, holds preferences kept in this browser: refit after every structural change, keep the reference level when collapsing, follow the browser's light or dark setting, how groups show when a term opens, the Build animation's length, and request timings.",
      "Where the browser blocks storage, as some private windows do, the choices last until the page closes.",
    ]),
  }),
  Object.freeze({
    title: "Cross-validation",
    items: Object.freeze([
      "Pass a cross_validate() result to edit(model, cv=result). The tab shows each fold's scores and, when the result kept its fold models (return_estimators=True), every term's relativities by fold, least stable first.",
      "Each fold keeps its colour and its place within a level, fold 1 leftmost. Hover or focus a fold in a chart's legend to pick it out; a fold with no value at a level, one it never saw, leaves a gap there.",
      "Run CV on current model refits the current structure on the same folds and puts your hand edits back on each fold before scoring. It waits while changes wait for Refit, and needs the rows the folds were drawn on: cv_data=, or train data with the same row count.",
      "Final fit on all rows refits the current structure on train and validation rows and puts your hand edits back; the test split stays held out. Export then offers it as Final fit model.",
      "Both run in the background with a Cancel button. A job whose model changed while it ran is not kept.",
    ]),
  }),
  Object.freeze({
    title: "Exporting",
    items: Object.freeze([
      "Python model exports are round-trip validated and prediction-checked when evaluation rows are available.",
      "Excel rating workbooks require training or retained fit data and include structured summary tables.",
      "Final fit model is the latest Final fit on all rows, offered while the model has not changed since.",
      "Structure (JSON) holds each term's groupings, reference, shaped ranges and where new levels go, without coefficients or hand edits. In Python, superglm.read_structure(path).apply(model) builds them into another model, ready to fit on new data.",
    ]),
  }),
]);

/**
 * Return the shared help entry described by a popover trigger.
 *
 * @param {Element} element
 * @returns {Readonly<HelpEntry> | null}
 */
export function helpForElement(element) {
  const isHtml = element instanceof HTMLElement;
  const isSvg = typeof SVGElement !== "undefined" && element instanceof SVGElement;
  if (!isHtml && !isSvg) return null;

  const tool = element.dataset.tool;
  if (tool && TOOL_HELP[tool] && !element.dataset.popoverBody) return TOOL_HELP[tool];

  // A disabled icon's own reason outranks its operation help, as for tools.
  const operation = element.dataset.helpOperation || element.dataset.op;
  if (operation && OPERATION_HELP[operation] && !element.dataset.popoverBody) {
    return OPERATION_HELP[operation];
  }

  const control = element.dataset.helpControl;
  if (control && CONTROL_HELP[control]) return CONTROL_HELP[control];

  const title = element.dataset.popoverTitle;
  if (!title) return null;
  return Object.freeze({ title, body: element.dataset.popoverBody || "" });
}
