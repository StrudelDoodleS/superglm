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
  breaks: Object.freeze({
    title: "Breaks",
    body:
      "Click the plot to add a break, drag a break to move it, and click × to remove it. Choose a form, then Transform and refit.",
    shortcut: "B",
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
    body: "Constrain selected relativities to a non-decreasing sequence.",
  }),
  decreasing: Object.freeze({
    title: "Make decreasing",
    body: "Constrain selected relativities to a non-increasing sequence.",
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
      "Pin the selected level as the reference (relativity 1.00) and refit. Predictions stay the same unless a selection penalty is on.",
  }),
});

/** @type {Readonly<Record<string, Readonly<HelpEntry>>>} */
export const STRUCTURE_HELP = Object.freeze({
  transform_term: Object.freeze({
    title: "Transform and refit",
    body:
      "Replace this term with the form and breaks shown, then refit the model. Restore undoes it.",
  }),
  restore_structure: Object.freeze({
    title: "Restore previous structure",
    body: "Undo the latest collapse, ungroup, transform or reference change.",
  }),
  revert_to_original: Object.freeze({
    title: "Revert to original model",
    body: "Go back to the model the editor was opened with.",
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
    keys: Object.freeze(["select", "move", "zoom", "handles", "breaks"]),
  }),
  Object.freeze({
    title: "Selection operations",
    keys: Object.freeze(Object.keys(OPERATION_HELP)),
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
    title: "History",
    items: Object.freeze([
      "Ctrl/Cmd+Z: undo",
      "Ctrl/Cmd+Shift+Z or Ctrl+Y: redo",
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

  const operation = element.dataset.helpOperation || element.dataset.op;
  if (operation && OPERATION_HELP[operation]) return OPERATION_HELP[operation];

  const control = element.dataset.helpControl;
  if (control && CONTROL_HELP[control]) return CONTROL_HELP[control];

  const title = element.dataset.popoverTitle;
  if (!title) return null;
  return Object.freeze({ title, body: element.dataset.popoverBody || "" });
}
