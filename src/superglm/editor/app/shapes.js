// @ts-check
// Pure shaped-range logic for the selection palette and the chart overlay:
// no DOM, no store. Python validates every range again when it is sent.

import { fmt } from "./format.js";

/** @typedef {import('./api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('./api/contracts.js').ShapedRange} ShapedRange */
/** @typedef {import('./api/contracts.js').ShapeSupport} ShapeSupport */
/** @typedef {{x:number[], displayToSourceIndices:number[][]}} DisplayAxis */

export const SHAPE_NAMES = Object.freeze(["Flat", "Line", "Quadratic", "Cubic"]);
const SHAPE_DESCRIPTIONS = Object.freeze([
  "a flat level",
  "a straight line",
  "a quadratic",
  "a cubic"
]);
export const NOT_CONTIGUOUS = "Select a continuous run of points.";
export const TOO_FEW_POINTS = "Select at least two points to shape a range.";
export const GROUPED_EDGE =
  "A range must start and end on single bands. Ungroup the bands at its ends first.";
export const SPECIAL_LEVEL =
  "A range covers bands only. Leave special levels out of the selection.";

/**
 * The selection as a run of source indices: null unless it is non-empty and
 * consecutive, which is contiguous in the displayed order on the numeric and
 * ordered axes that can take a shape.
 * @param {Set<number>} selectedIndices @returns {[number, number]|null}
 */
function selectedRun(selectedIndices) {
  if (!selectedIndices.size) return null;
  const lo = Math.min(...selectedIndices);
  const hi = Math.max(...selectedIndices);
  return hi - lo + 1 === selectedIndices.size ? [lo, hi] : null;
}

/**
 * The range a contiguous selection names: its x extent on a numeric term,
 * its first and last band labels on an ordered one. A selected collapsed
 * group is selected by all of its source bands, so it contributes them all.
 * @param {TermPayload} term @param {Set<number>} selectedIndices
 * @returns {{lo:number|string, hi:number|string}|null}
 */
export function shapeRangeForSelection(term, selectedIndices) {
  const run = selectedRun(selectedIndices);
  if (!run) return null;
  if (term.levels) return { lo: term.levels[run[0]], hi: term.levels[run[1]] };
  return { lo: meetingEdge(term, run[0], -1), hi: meetingEdge(term, run[1], 1) };
}

/**
 * A numeric run's edge: its end point, or a shaped range's facing edge when
 * no drawn point lies between the two, so back-to-back selections meet
 * instead of leaving the free sliver that snapping each outward would open.
 * Past the first or last point ``next`` is undefined and matches nothing.
 * @param {TermPayload} term @param {number} index @param {-1|1} direction
 */
function meetingEdge(term, index, direction) {
  const x = term.x[index];
  const next = term.x[index + direction];
  const facing = term.shape.ranges.map((range) => Number(direction < 0 ? range.hi : range.lo));
  return facing.find((edge) => (edge - x) * direction > 0 && (next - edge) * direction >= 0) ?? x;
}

/**
 * Whether the palette shows the shape icons for this selection, and why one
 * of ``degree`` is disabled. Categorical terms hide them; a term that cannot
 * take a shape shows them disabled with the backend's reason.
 * @param {TermPayload} term @param {Set<number>} selectedIndices @param {number} [degree]
 * @returns {{visible:boolean, enabled:boolean, reason:string|null}}
 */
export function shapeButtonState(term, selectedIndices, degree = 0) {
  if ((term.term_type || term.kind) === "categorical") {
    return { visible: false, enabled: false, reason: null };
  }
  const reason = disabledReason(term, selectedIndices, degree);
  return { visible: true, enabled: reason === null, reason };
}

/** @param {TermPayload} term @param {Set<number>} selectedIndices @param {number} degree */
function disabledReason(term, selectedIndices, degree) {
  if (!term.shape.available) return term.shape.reason ?? "Shapes are not available here.";
  const run = selectedRun(selectedIndices);
  if (!run) return NOT_CONTIGUOUS;
  if (run[0] === run[1]) return TOO_FEW_POINTS;
  if (!term.levels) return tooFewValues(term.shape.support, run, degree);
  const labels = term.levels.slice(run[0], run[1] + 1);
  if (labels.some((label) => term.shape.specials.includes(label))) return SPECIAL_LEVEL;
  if (groupedEdge(term, run)) return GROUPED_EDGE;
  const needed = degree + 1;
  return bandCount(term, run) < needed
    ? `Select at least ${needed} bands for a ${SHAPE_NAMES[degree]}.`
    : null;
}

/**
 * The distinct values a run of bands holds: one per band, and one for a
 * collapsed group inside it, whose bands share one place on the axis.
 * @param {TermPayload} term @param {[number, number]} run
 */
function bandCount(term, [lo, hi]) {
  const inside = (term.level_groups ?? [])
    .filter(({ indices }) => lo < indices[0] && indices[0] < hi);
  return inside.reduce((count, { indices }) => count - indices.length + 1, hi - lo + 1);
}

/**
 * A numeric run holds the values the refit sees between its snapped edges,
 * which Python counts per grid point: ``through[hi] - below[lo]``. Without
 * counts (no retained data) Python still refuses a short range when sent.
 * @param {ShapeSupport|null|undefined} support @param {[number, number]} run @param {number} degree
 */
function tooFewValues(support, [lo, hi], degree) {
  const needed = degree + 1;
  if (!support || support.through[hi] - support.below[lo] >= needed) return null;
  return `Select at least ${needed} distinct values for a ${SHAPE_NAMES[degree]}.`;
}

/**
 * Python refuses a range edge inside a collapsed group (the group would
 * absorb a stated kink), so the icons say so before anything is sent.
 * @param {TermPayload} term @param {[number, number]} run
 */
function groupedEdge(term, [lo, hi]) {
  const groups = term.level_groups ?? [];
  return groups.some((group) => group.indices.includes(lo) || group.indices.includes(hi));
}

/** @param {number|string} edge */
function edgeText(edge) {
  return typeof edge === "number" ? fmt(edge) : edge;
}

/** @param {ShapedRange} range */
export function shapeRangeDescription(range) {
  const { lo, hi } = range;
  return `Pinned to ${SHAPE_DESCRIPTIONS[range.degree]} from ${edgeText(lo)} to ${edgeText(hi)}.`;
}

/**
 * The data-x extent a range pins on the displayed axis: between its edge
 * values on a numeric term, and between its edge bands' positions on an
 * ordered one, so ranges sharing an edge band meet there. Null when an edge
 * band is not on the displayed axis.
 * @param {TermPayload} term @param {DisplayAxis} view @param {ShapedRange} range
 * @returns {[number, number]|null}
 */
export function shapeRangeExtent(term, view, range) {
  if (!term.levels) return [Number(range.lo), Number(range.hi)];
  const lo = bandDisplayX(term.levels, view, range.lo);
  const hi = bandDisplayX(term.levels, view, range.hi);
  return lo === null || hi === null ? null : [lo, hi];
}

/**
 * A band's x on the displayed axis, through the collapsed display's source
 * mapping when the term is drawn collapsed.
 * @param {string[]} levels @param {DisplayAxis} view @param {number|string} label
 */
function bandDisplayX(levels, view, label) {
  const source = levels.indexOf(String(label));
  const display = view.displayToSourceIndices.findIndex((indices) => indices.includes(source));
  return display < 0 ? null : view.x[display];
}
