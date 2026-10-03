// @ts-check
// Structural changes waiting for Refit, drawn over the last refit's curve,
// which stays as it is until Refit. A waiting range is a dashed box with a
// "waiting for refit" tag. A waiting group shows its members' exposure bars
// dashed in its group colour (chart.js colours them from pendingGroupMarks),
// rings their points, and is named on a dashed bracket under the axis labels,
// in a row the categorical axis keeps for it. A waiting ungroup marks the
// levels that leave a fitted group the same way, in that group's colour and
// without rings: the fitted group's markers still sit on its points.

import { shapeRangeDescription, shapeRangeExtent } from "../shapes.js";
import { labelWidth } from "./shape_overlay.js";
import { el, text } from "./svg.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../shapes.js').DisplayAxis} DisplayAxis */
/** @typedef {(slot:number, alpha?:number)=>string} GroupColor */
/**
 * One waiting group on the displayed axis.
 * @typedef {object} PendingGroupMark
 * @property {string} label the group's label
 * @property {string[]} members its levels, in axis order
 * @property {number[]} display the displayed points it takes in, ascending
 * @property {number} slot its colour in the level-group palette, after the fitted groups'
 */
/**
 * The levels a waiting ungroup takes out of one fitted group, to stand alone.
 * @typedef {object} PendingUngroupMark
 * @property {string} leaves the fitted group's label
 * @property {string[]} members the levels that leave it, in axis order
 * @property {number[]} display the displayed points they sit on, ascending
 * @property {number} slot the fitted group's colour in the level-group palette
 */

/** The row a waiting group's bracket takes under the axis labels, in px. */
export const WAITING_BRACKET_ROW = 20;
const BRACKET_DROP = 6;
const BRACKET_TICK = 4;
const BRACKET_LABEL_GAP = 11;
const BRACKET_PAD = 0.35;
const RING_RADIUS = 6.5;
const TAG_INSET = 6;
const TAG_PADDING = 8;
const TAG_HEIGHT = 20;
const TAG_BASELINE = 20;

/**
 * The waiting groups to draw: each group the waiting changes make that brings
 * together levels the fitted term does not already group, with its members'
 * displayed points and a palette slot after the fitted groups' slots. What is
 * left of a fitted group after a partial ungroup is not one: its members are
 * grouped already. Members are matched to the term's levels as strings, so
 * levels that look like numbers match.
 * @param {TermPayload} term @param {DisplayAxis} view
 * @returns {PendingGroupMark[]}
 */
export function pendingGroupMarks(term, view) {
  const groups = term.pending?.groups;
  if (!groups || !Array.isArray(term.levels)) return [];
  const levels = term.levels.map(String);
  const fitted = fittedGroupSources(term);
  /** @type {Omit<PendingGroupMark, "slot">[]} */
  const marks = [];
  for (const [label, members] of Object.entries(groups)) {
    const sources = [...new Set(members.map((member) => levels.indexOf(String(member))))]
      .filter((index) => index >= 0)
      .sort((left, right) => left - right);
    if (sources.length < 2) continue;
    if (fitted.some((group) => sources.every((index) => group.includes(index)))) continue;
    const display = displayedPoints(view, sources);
    if (!display.length) continue;
    marks.push({ label, members: sources.map((index) => levels[index]), display });
  }
  marks.sort((left, right) => left.display[0] - right.display[0]);
  const first = term.level_groups?.length ?? 0;
  return marks.map((mark, index) => ({ ...mark, slot: first + index }));
}

/**
 * The waiting ungroups to draw: for each fitted group, the members that stand
 * alone once the waiting changes apply, with their displayed points and the
 * group's own palette slot. A member that moves into another waiting group is
 * drawn with that group instead. Nothing is regrouped while ``pending.groups``
 * is null; an empty mapping means every fitted group breaks up.
 * @param {TermPayload} term @param {DisplayAxis} view
 * @returns {PendingUngroupMark[]}
 */
export function pendingUngroupMarks(term, view) {
  const groups = term.pending?.groups;
  if (!groups || !Array.isArray(term.levels)) return [];
  const levels = term.levels.map(String);
  const grouped = new Set(Object.values(groups).flat().map(String));
  /** @type {PendingUngroupMark[]} */
  const marks = [];
  fittedGroupSources(term).forEach((sources, slot) => {
    const leaving = sources.filter((index) => !grouped.has(levels[index]));
    if (sources.length < 2 || !leaving.length) return;
    const display = displayedPoints(view, leaving);
    if (!display.length) return;
    const leaves = String(term.level_groups?.[slot]?.label ?? "");
    marks.push({ leaves, members: leaving.map((index) => levels[index]), display, slot });
  });
  return marks.sort((left, right) => left.display[0] - right.display[0]);
}

/**
 * Each fitted group's levels as source indices, ascending, in the order the
 * chart gives the groups their colours.
 * @param {TermPayload} term @returns {number[][]}
 */
function fittedGroupSources(term) {
  const count = Array.isArray(term.levels) ? term.levels.length : 0;
  return (term.level_groups ?? []).map((group) => [...new Set(group.indices.map(Number))]
    .filter((index) => index >= 0 && index < count)
    .sort((left, right) => left - right));
}

/**
 * The displayed points that show any of these source levels, ascending.
 * @param {DisplayAxis} view @param {readonly number[]} sources @returns {number[]}
 */
function displayedPoints(view, sources) {
  /** @type {number[]} */
  const display = [];
  view.displayToSourceIndices.forEach((indices, position) => {
    if (indices.some((index) => sources.includes(index))) display.push(position);
  });
  return display;
}

/** @param {{members:readonly string[], leaves?:string}} mark */
export function waitingBracketText(mark) {
  const many = mark.members.length > 3;
  if (mark.leaves !== undefined) {
    const named = many ? `${mark.members.length} levels` : mark.members.join(", ");
    return `${named} ungrouped · waiting`;
  }
  const named = many ? `${mark.members.length} levels` : mark.members.join(" + ");
  return `${named} · waiting`;
}

/** @param {{members:readonly string[], leaves?:string}} mark */
function waitingBracketPopover(mark) {
  const members = mark.members.join(", ");
  if (mark.leaves === undefined) return `${members} become one group at the next Refit.`;
  const verb = mark.members.length === 1 ? "leaves" : "leave";
  return `${members} ${verb} the group ${mark.leaves} at the next Refit.`;
}

/**
 * Each waiting range as a dashed box over the plot, tagged at its top.
 * @param {SVGElement} svg
 * @param {{term:TermPayload, view:DisplayAxis, sx:(v:number)=>number,
 *   margin:{left:number, top:number}, innerW:number, innerH:number}} options
 */
export function drawPendingRanges(svg, { term, view, sx, margin, innerW, innerH }) {
  const ranges = term.pending?.ranges ?? [];
  if (!ranges.length) return;
  const left = margin.left;
  const right = margin.left + innerW;
  const layer = el("g", { class: "pending-layer" });
  // In the document before the tags are made, so each label can be measured.
  svg.appendChild(layer);
  for (const range of ranges) {
    const extent = shapeRangeExtent(term, view, range);
    if (!extent) continue;
    const x0 = Math.max(left, sx(extent[0]));
    const x1 = Math.min(right, sx(extent[1]));
    if (x1 <= x0) continue;
    const name = `${range.label} · waiting for refit`;
    const band = el("g", {
      class: "pending-range",
      "data-popover-title": name,
      "data-popover-body": `${shapeRangeDescription(range)} It applies at the next Refit.`
    });
    band.appendChild(el("rect", {
      class: "pending-range-box", x: x0, y: margin.top, width: x1 - x0, height: innerH
    }));
    layer.appendChild(band);
    const label = text(
      band, x0 + TAG_INSET + TAG_PADDING, margin.top + TAG_BASELINE, name,
      "pending-range-label", "start"
    );
    band.insertBefore(el("rect", {
      class: "pending-range-tag",
      x: x0 + TAG_INSET,
      y: margin.top + TAG_INSET,
      width: labelWidth(label) + TAG_PADDING * 2,
      height: TAG_HEIGHT,
      rx: TAG_HEIGHT / 2,
      ry: TAG_HEIGHT / 2
    }), label);
  }
}

/**
 * A ring in its group's colour around each point a waiting group takes in.
 * @param {SVGElement} svg @param {readonly PendingGroupMark[]} marks
 * @param {{view:{x:number[], y:number[]}, sx:(v:number)=>number, sy:(v:number)=>number,
 *   color:GroupColor}} options
 */
export function drawPendingGroupRings(svg, marks, { view, sx, sy, color }) {
  for (const mark of marks) {
    for (const position of mark.display) {
      svg.appendChild(el("circle", {
        class: "pending-group-ring",
        cx: sx(view.x[position]),
        cy: sy(view.y[position]),
        r: RING_RADIUS,
        style: `stroke: ${color(mark.slot, 0.95)}`
      }));
    }
  }
}

/**
 * A dashed bracket under the axis labels for each waiting group or ungroup,
 * named for its members. ``top`` is the bottom of the tick labels; ``left``
 * and ``right`` bound the plot, so a zoom clips the bracket and drops a group
 * zoomed out of view.
 * @param {SVGElement} svg
 * @param {readonly (PendingGroupMark|PendingUngroupMark)[]} marks
 * @param {{view:{x:number[]}, sx:(v:number)=>number, top:number, left:number, right:number,
 *   color:GroupColor}} options
 */
export function drawPendingGroupBrackets(svg, marks, { view, sx, top, left, right, color }) {
  const step = view.x.length > 1 ? Math.abs(sx(view.x[1]) - sx(view.x[0])) : right - left;
  const pad = BRACKET_PAD * step;
  const lineY = top + BRACKET_DROP;
  for (const mark of marks) {
    const first = view.x[mark.display[0]];
    const last = view.x[mark.display[mark.display.length - 1]];
    const x0 = Math.max(left, sx(first) - pad);
    const x1 = Math.min(right, sx(last) + pad);
    if (x1 <= x0) continue;
    const bracket = el("g", {
      class: "leaves" in mark ? "pending-group-bracket ungroup" : "pending-group-bracket",
      "data-popover-title": "Waiting for refit",
      "data-popover-body": waitingBracketPopover(mark)
    });
    bracket.appendChild(el("path", {
      class: "pending-group-bracket-line",
      d: `M ${x0.toFixed(2)} ${lineY - BRACKET_TICK} V ${lineY} H ${x1.toFixed(2)} V ${lineY - BRACKET_TICK}`,
      style: `stroke: ${color(mark.slot, 0.95)}`
    }));
    const label = text(
      bracket, (x0 + x1) / 2, lineY + BRACKET_LABEL_GAP, waitingBracketText(mark),
      "pending-group-label", "middle"
    );
    label.setAttribute("style", `fill: ${color(mark.slot, 1)}`);
    svg.appendChild(bracket);
  }
}
