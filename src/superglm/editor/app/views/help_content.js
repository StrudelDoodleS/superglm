// @ts-check

import { UNSMOOTHED_HELP } from "../unsmoothed.js";

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
    body: "Drag a box or use the mouse wheel to zoom. In other tools, hold Ctrl or Cmd with the wheel, or pinch.",
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
    body: "Hold the highest value so far until the curve passes it, so the selection never falls. "
      + "It starts from the first selected point, so it can leave a step at an edge.",
  }),
  decreasing: Object.freeze({
    title: "Make decreasing",
    body: "Hold the lowest value so far until the curve drops below it, so the selection never rises. "
      + "It starts from the first selected point, so it can leave a step at an edge.",
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
      "A new range paints over the ranges it overlaps: one it covers goes, one it overlaps at an end keeps the part outside it, and one it falls inside is split in two. Each part keeps its shape. One selected right beside a range meets it. Undo takes back the latest shape.",
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
      "While a knot is dragged, or a knot change waits, the spline's basis functions show as bumps in a band along the bottom of the plot. The ones the change reshapes are in colour, the rest faint. Each knot is shared by a few neighbouring functions, so moving one reshapes only the curve near it.",
      "The count and Placed by above the chart place every knot again by a rule: even spacing, quantiles of the values, quantiles of the rows, or tempered quantiles, which weight each value by its row count to the power alpha. Knots you placed by hand are dropped; Undo brings them back. Reset knots goes back to the knots declared in code.",
      "Kind above the chart rebuilds the spline as another kind and keeps its knots. A P-spline penalises the differences between neighbouring coefficients; a B-spline penalises the curve's bending itself; a cubic regression spline and a natural spline are cubic and held straight at each end of the data. A P-spline or B-spline takes the degree declared in code. A cardinal cubic regression spline declared in code is named there, but cannot be chosen.",
      "Shrink adds a second penalty, on the term's straight-line part, so the fit can shrink the term towards a straight line and, where the data do not support it, out of the model. It is off where the term cannot take it, such as a natural spline or a term with shaped ranges; hovering it says why. A Kind or Shrink change waits for Refit like a knot change, and Undo takes it back.",
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
    title: "Unsmoothed line",
    items: Object.freeze([
      `The Unsmoothed icon above the chart, on a spline term: ${UNSMOOTHED_HELP}`,
      "It stays on as you move between terms. Each term's line is fitted the first time it shows; the icon is busy meanwhile, and the editor stays free. Hand edits keep the line; a refit fits it again.",
      "On an ordered term, Free levels draws each level fitted free, joined by a line: one button for the diamonds and the line. A level with no free value leaves a gap in the line.",
      "A spline whose rows cannot pin down every basis function without smoothing, as across a gap in the data, gets no line: the icon stays on, and its hover text says why.",
      "The line widens the chart's range by at most the curve's own range on each side; past that it runs off the plot.",
    ]),
  }),
  Object.freeze({
    title: "Rating table",
    items: Object.freeze([
      "The Chart / Table switch above the chart shows the current term's block of the Excel rating table instead of its curve, with the workbook's number formats and note. It is built by the same code as the export, on the training data, and follows every edit.",
      "Interactions are not shown. A model the export refuses shows the reason instead of a table.",
      "In a SuperLSS session the table is the term's block for its own parameter, and the note says what its factors multiply: the Mean, the CV, σ above its floor (log-normal) or overdispersion (1/θ). A parameter the export leaves out shows its reason; the other parameters keep their tables.",
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
    title: "SuperLSS models",
    items: Object.freeze([
      "A SuperLSS model has more than one curve per feature. The feature list groups them by the parameter they describe: the mean (blue), and the CV or overdispersion (orange). The CV is the coefficient of variation, standard deviation ÷ mean; overdispersion is 1/θ, where variance = mean + mean²/θ. A feature that acts in both appears in both groups, with a small curve of its effect on each row.",
      "Every curve is a relativity: the factor the feature multiplies its parameter by, read as in any other model, and 1.00 is no effect. A mean curve is the relativity of the mean; a CV curve (gamma) is the relativity of the CV; a log-normal scale curve is the relativity of σ above its floor; an overdispersion curve (negative binomial) is the relativity of the overdispersion 1/θ, so it rises where overdispersion rises; a Tweedie dispersion curve is the relativity of the dispersion. Moving a point sets the relativity to the value you drag to. No curve is read at a reference row, so editing one curve never moves another, and the Cross-validation tab draws these same values.",
      "On a CV or overdispersion curve, Increase and Decrease move the selected points by about 5% of the value shown, and Make increasing, Make decreasing and the snaps act on the curve as drawn: on an overdispersion curve, Increase raises overdispersion.",
      "Refit keeps the hand edits made in this session. A model exported with hand edits and opened again holds those edits in its coefficients, and Refit estimates those terms again from the data.",
      "History marks each step with the parameter it acted in. Undo, Redo, Revert and Refit cover both parameters in one history.",
      "The Validation tab scores each split by log score, CRPS and actual over expected. Pass a cross_validate() result to edit(model, cv=result) to score the same model across folds on the Cross-validation tab, where every mean, CV and overdispersion curve has its relativities by fold, listed under its parameter's heading with the mean curves first. Each chart's heading names its parameter and what the curve is a relativity of, for example 'Area · CV · relativity of the CV'; higher means more variation on every CV and overdispersion curve. The in-force curve is the one the Editor tab draws, value for value. The Validation tab shows a change (Δ) only once you have edited the model, and leaves out the CRPS for a family with none. Final fit runs from that tab as well.",
      "Shaped ranges are off for SuperLSS terms in the editor; a shape rule declared on the model with constraint= is enforced when it is fitted. Also off are special levels, the Unsmoothed line, free levels and new-level choices.",
      "The coefficient summary and the Structure export are off too. Their hover text says why. Export an edited SuperLSS model as joblib, which keeps its history.",
    ]),
  }),
  Object.freeze({
    title: "Distribution panel",
    items: Object.freeze([
      "Under a SuperLSS chart, the distribution panel draws the distribution of the response at each value or level of the feature, with every other feature fixed at the reference row its caption names: a numeric feature at its median, any other at its most frequent level, and every offset at zero. A categorical feature shows its 12 levels with the most exposure, and the caption says when there are more.",
      "A count model draws no distribution in this panel yet; the panel says so in place of the chart.",
      "The line marks an amount or a percentile. Drag it, use the slider, or type in its box: an amount such as 25k, €25,000 or 1.5m, or a percentile such as 99.5th or 1 in 200. Each row then reads the chance of a value above the amount, or the amount at the percentile.",
      "A comma is always a thousands separator, so 1,5m reads as 15,000,000. Write a decimal with a point, as in 1.5m.",
      "When the line cannot be drawn where you asked, a note under it says why. An amount off the amount axis is drawn at the axis edge and the readouts keep your amount; a gamma CV above 20 is drawn at 20, the largest the panel computes.",
      "Under the ridges, a density fan shows the same distributions across the feature: the shading is how likely each amount is within each value or level, blue below the line and orange beyond it, with a line for the mean and one for the median.",
      "A chance of one half or more is always written as a percentage, even with 1 in N chosen, since 1 in 1.0 says nothing. Where the room for a heading beside a plot is too narrow, the heading is left out.",
      "A Tweedie row has a chance of exactly zero, written under each row's readout. The ridges and the shading draw only the amounts above zero, and a percentile inside that chance reads 0.",
      "While you drag a curve, the panel previews the new distributions in the browser; when you let go, it redraws from the edited model.",
    ]),
  }),
  Object.freeze({
    title: "Explain tab",
    items: Object.freeze([
      "A SuperLSS session has an Explain tab. Each row at the top is a risk: a value for every feature, starting at the reference risk, which has every number at its training median and every level at its most frequent. The plus button adds a risk, a training row drawn at random, up to eight; each keeps its letter and colour until it is removed.",
      "Click a risk's letter, or work in its row, to select it. On the selected row, the icons take the values of another training row drawn at random, put the risk back at the reference risk, add a copy of it to change one value and compare, or remove it. At eight risks the plus button and the copy icon are off.",
      "A number outside the range of the training rows is accepted, and its box turns orange: the model has no data there and extrapolates. The box's hover text gives the range.",
      "The upper plot shows how likely each amount is, as a density over the amount axis, and the lower one the chance of an amount above each value, down to 1 in 10,000. The selected risk is filled and marked at its median, a dot, and its mean, a diamond.",
      "The table has a row per risk: its mean, median, 90th, 99th and 99.5th percentiles, the chance above the line, and the coefficient of variation (CV). A line at one of those percentiles adds no second column; written as 1 in N, the line's column is headed 1 in N. Two risks are followed by the ratio of the second to the first, so two risks with close means and different CVs show where they part: in the tail.",
      "The walk under the table explains the selected risk. It starts at risk A, or at the reference risk when risk A is selected, and has a row per feature that differs: the multiplier that feature alone puts on the mean and on the CV. The start times every row is the end. Where the model has interactions, what the features add only together is a row of its own. A risk that differs from its start in nothing gets one line saying so in place of the rows.",
      "The tab fits one screen with a few risks. With more, the page scrolls, and the plots keep a height that stays readable.",
      "After an edit, a single risk is read beside the original model's distribution for it, drawn in grey, and a row of the walk that the edit moved says what it was.",
      "The line is the distribution panel's: a press or a drag on the plots moves it, as its box does, and the box takes a number alone. Every offset is at zero.",
      "A Tweedie model has a chance of no claim. The table gives it for each risk. The upper plot is the density of the amount given a claim, so every curve has an area of one and its dot and diamond are the median and mean claim; the lower plot is the chance for the whole policy, so it starts at the chance of a claim. Its walk covers the mean and the dispersion; the power is one constant for the model.",
      "A model of counts (a negative binomial) draws its distribution as a bar per number of claims for each risk, for one unit of exposure, on a scale that keeps the rare counts in view, and its table reads the mean, the chance of no claim and of one, two and three or more, and the overdispersion 1/θ. For a log-normal, the walk's σ column is the standard deviation of the log amount, which its multipliers act on above its floor; the table above gives the CV.",
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
      "Ctrl or Cmd with the mouse wheel, or a pinch: zoom",
      "Mouse wheel alone: scrolls when there is more below the chart, otherwise zooms",
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
    title: "Diagnose",
    items: Object.freeze([
      "The Diagnose tab says whether each fit converged, whether its numbers can be used, and what to try if not. It reads what the fits recorded, so opening it refits nothing. It lists All rows, the model you are editing fitted on every training row, and, when folds are available, each fold, and opens on the first fit that did not converge.",
      "Each fit's row leads with its verdict: converged; converged with a looser stop, where the numbers are usable but the smoothing was not checked to the strictest standard; did not converge, where the fit's numbers are those of the last step it reached; or no smoothing search, because you set the smoothing. The tab also says which smoothing search made the fit, in a column when fits differ and above the chart when they share it. For a SuperLSS fit it reads 'Quick smoothing updates only', 'Quick smoothing updates, then Newton steps to an optimum' or 'A few quick smoothing updates, then Newton steps'; the quick updates are Fellner–Schall steps. When several converged fits share one explanation, the line above the table gives it once.",
      "Search path plots the REML score each fit gained in each smoothing round. A line that settles ends at the bottom; a crossed end is a search that stopped before it converged. What the fit reports lists the findings that bear on your numbers first, and folds those about how the search went. Smoothing by term puts every fit's EDF (effective degrees of freedom, how flexible a term is) side by side, so a fold that left a term straight when the others curve it stands out.",
      "A slow or unusual fold is often not a fault in the fit. A term's best smoothing can be effectively infinite, and one fold can differ because of a few large rows it held out, which Rows that cost the most lists. For All rows it lists the training rows the fit predicted worst; the fit saw those, so a fold's held-out rows are the fairer test. Compare the folds' scores before reading much into one curve.",
    ]),
  }),
  Object.freeze({
    title: "Cross-validation",
    items: Object.freeze([
      "Pass a cross_validate() result to edit(model, cv=result). The tab shows each fold's scores and, when the result kept its fold models (return_estimators=True), every term's relativities by fold, least stable first.",
      "The fit curve (and the edited curve, once you edit) is drawn exactly as the Editor tab draws the term, so a term reads the same on both tabs. Each fold's curve is moved by a constant factor so its exposure-weighted mean log matches the fit's: folds compare on shape, and a fold's gap and correlation do not depend on that level.",
      "Each fold keeps its colour and its place within a level, fold 1 leftmost. Hover or focus a fold in a chart's legend to pick it out; a fold with no value at a level, one it never saw, leaves a gap there.",
      "Run CV on current model refits the current structure on the same folds and puts your hand edits back on each fold before scoring. It waits while changes wait for Refit, and needs the rows the folds were drawn on: cv_data=, or train data with the same row count.",
      "Final fit on all rows refits the current structure on train and validation rows and puts your hand edits back; a test split, if you supplied one, stays held out. The Final Fit tab then says whether the refit converged. Export then offers it as Final fit model.",
      "Both run in the background with a Cancel button, so you can keep working. A pill beside the tabs shows how far a job has got on every tab, and how it ended; click it to open the job's tab. While Run CV runs, each fold's scores land on the cards and in the table as it finishes. A job whose model changed while it ran is not kept.",
    ]),
  }),
  Object.freeze({
    title: "Exporting",
    items: Object.freeze([
      "Python model exports are round-trip validated and prediction-checked when evaluation rows are available.",
      "Excel rating workbooks require training or retained fit data and include structured summary tables. A SuperLSS workbook has one sheet per parameter that has a table, then a summary; if none has one, the export says why.",
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
