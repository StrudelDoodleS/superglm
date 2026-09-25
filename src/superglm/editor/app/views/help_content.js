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
    body: "Click points or drag a box to select curve values.",
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
  help: Object.freeze({
    title: "Help",
    body: "Open modes, gestures, shortcuts, curve operations, refits, and exporting.",
    shortcut: "?",
  }),
});

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
    title: "Collapse and refit",
    body: "Combine the selected categorical levels and refit the model.",
  }),
  ungroup_levels: Object.freeze({
    title: "Ungroup and refit",
    body: "Separate the selected grouped levels and refit the model.",
  }),
  set_reference: Object.freeze({
    title: "Set reference and refit",
    body:
      "Pin the selected level as the reference (relativity 1.00) and refit. Predictions stay the same unless a selection penalty is on. Unseen levels rated at the reference move with it.",
  }),
  shape_flat: Object.freeze({
    title: "Flat and refit",
    body: "Make the selected range flat and refit. The rest of the curve stays smooth. Undo takes it back.",
  }),
  shape_line: Object.freeze({
    title: "Line and refit",
    body:
      "Make the selected range a straight line and refit. The rest of the curve stays smooth. Undo takes it back.",
  }),
  shape_quadratic: Object.freeze({
    title: "Quadratic and refit",
    body:
      "Make the selected range a quadratic and refit. The rest of the curve stays smooth. Undo takes it back.",
  }),
  shape_cubic: Object.freeze({
    title: "Cubic and refit",
    body: "Make the selected range a cubic and refit. The rest of the curve stays smooth. Undo takes it back.",
  }),
});

/** @type {Readonly<Record<string, Readonly<HelpEntry>>>} */
export const STRUCTURE_HELP = Object.freeze({
  revert_to_original: Object.freeze({
    title: "Revert to original model",
    body: "Go back to the model the editor was opened with. Undo brings back everything it cleared.",
  }),
  refresh_from_python: Object.freeze({
    title: "Refresh from Python",
    body: "Re-read the Python session after changing it in the notebook.",
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
    keys: Object.freeze(["select", "move", "zoom", "handles"]),
  }),
  Object.freeze({
    title: "Selection operations",
    keys: Object.freeze(Object.keys(OPERATION_HELP)),
  }),
  Object.freeze({
    title: "Shaped ranges",
    items: Object.freeze([
      "Select a run of points or bands on a spline term, then choose Flat, Line, Quadratic or Cubic. That range is pinned to the shape; the rest of the term stays the fitted smooth.",
      "At each edge the curve leaves the shape along its slope (Tangent). The toggle beside the shape icons chooses Corner instead, where the slope may change at the edge. Select all, then a shape, for one polynomial over the whole axis.",
      "Flat fits a level. To hold the curve at its value at a range's edge instead of fitting a level, use Level from left or Level from right. That is an edit, not a refit.",
      "A new range may not overlap one already shaped, and one selected right beside it meets it. The same range with a new shape replaces it. Undo takes back the latest shape.",
      "A P-spline or natural spline term becomes a B-spline with a derivative penalty so its penalty can skip the shaped range; a natural spline's ends are then no longer held straight.",
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
      "Ctrl/Cmd+Z: undo",
      "Ctrl/Cmd+Shift+Z or Ctrl+Y: redo",
      "Revert to original model is one more step: Undo brings back everything it cleared.",
      "History, in the inspector, lists every edit and step of the session in order. Undo takes the entry above the current-position line; the muted entries below it are what Redo would put back.",
    ]),
  }),
  Object.freeze({
    title: "Exporting",
    items: Object.freeze([
      "Python model exports are round-trip validated and prediction-checked when evaluation rows are available.",
      "Excel rating workbooks require training or retained fit data and include structured summary tables.",
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
