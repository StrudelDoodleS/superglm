// @ts-check
// The B-spline basis a knot change reshapes, drawn while a knot is dragged in
// Knots mode and while a knot change waits: the term's basis functions as
// unit-height bumps in a band along the bottom of the plot, beneath the curve
// and inside its scale. The functions the change reaches take the basis
// palette; the rest stay faint, for the tiling they sit in.

import {
  basisCount,
  basisCurves,
  changedFunctions,
  functionsHoldingKnot,
  knotVector
} from "./knot_basis.js";
import { knotIndex, knotPx } from "./knot_marks.js";
import { el } from "./svg.js";

/** @typedef {import('./knot_marks.js').KnotFrame} KnotFrame */
/** @typedef {import('./knot_marks.js').KnotUi} KnotUi */
/**
 * @typedef {object} BasisOverlay
 * @property {number[]} grid chart x along the plot
 * @property {number[][]} curves each basis function on the grid
 * @property {boolean[]} reached which functions the change reshapes
 */

const PX_PER_SAMPLE = 3;
// Room above the tallest bump inside the band's backing.
const BAND_HEADROOM = 6;
const BASIS_PALETTE_SIZE = 12;

/**
 * The knots a change in progress leaves, and the basis functions it reaches:
 * a drag reaches the ``degree + 2`` functions holding the dragged knot; a
 * removal, a change being staged or one waiting, every function the knots in
 * force do not have. Null while nothing changes.
 * @param {KnotFrame} frame @param {Readonly<KnotUi>} ui
 * @returns {{positions:number[], reached:number[]}|null}
 */
export function basisChange(frame, ui) {
  const basis = frame.basis;
  if (!basis) return null;
  /** @param {number[]} positions @param {readonly number[]} before */
  const changed = (positions, before) => {
    const next = knotVector(basis, positions);
    const previous = knotVector(basis, before);
    const tolerance = 1e-9 * (previous[previous.length - 1] - previous[0]);
    return { positions, reached: changedFunctions(next, previous, basis.degree, tolerance) };
  };
  const drag = ui.drag;
  const dragged = drag && drag.moved ? knotIndex(frame, drag.from) : null;
  if (drag && dragged !== null) {
    const others = frame.positions.filter((_, index) => index !== dragged);
    if (drag.remove) return changed(others, frame.positions);
    const positions = [...others, drag.x].sort((a, b) => a - b);
    const count = basisCount(knotVector(basis, positions), basis.degree);
    return {
      positions,
      reached: functionsHoldingKnot(positions.indexOf(drag.x), basis.degree, count)
    };
  }
  if (ui.pending) return changed([...ui.pending], frame.inForce);
  if (frame.waiting) return changed(frame.positions, frame.inForce);
  return null;
}

/**
 * The overlay for the frame and gesture, without a DOM: the basis of the
 * changed knots on a grid across the drawn part of the boundary. Null outside
 * Knots mode, for a term with no basis to draw, or while nothing changes.
 * @param {KnotFrame} frame @param {Readonly<KnotUi>} ui @returns {BasisOverlay|null}
 */
export function basisOverlay(frame, ui) {
  if (!frame.editing || !frame.basis) return null;
  const change = basisChange(frame, ui);
  if (!change) return null;
  const from = Math.max(frame.basis.boundary[0], frame.xMin);
  const to = Math.min(frame.basis.boundary[1], frame.xMax);
  if (!(to > from)) return null;
  const width = knotPx(frame, to) - knotPx(frame, from);
  const samples = Math.max(24, Math.min(400, Math.round(width / PX_PER_SAMPLE)));
  const grid = Array.from({ length: samples + 1 }, (_, g) => from + ((to - from) * g) / samples);
  const curves = basisCurves(frame.basis, change.positions, grid);
  const reached = new Set(change.reached);
  return { grid, curves, reached: curves.map((_, index) => reached.has(index)) };
}

/** The band's height: a fifth of the plot, kept between 32 and 60 px. @param {KnotFrame} frame */
export function basisBandHeight(frame) {
  return Math.round(Math.min(60, Math.max(32, (frame.axisY - frame.top) * 0.2)));
}

/**
 * One function's bump: up from the axis where it starts, along its values,
 * down where it ends. Left open, so a fill closes it along the axis without a
 * stroke there. Null where it is zero across the grid.
 * @param {KnotFrame} frame @param {readonly number[]} grid @param {readonly number[]} values
 * @param {number} height @returns {string|null}
 */
function bumpPath(frame, grid, values, height) {
  const first = values.findIndex((value) => value > 0);
  if (first < 0) return null;
  let last = values.length - 1;
  while (values[last] <= 0) last -= 1;
  const from = Math.max(0, first - 1);
  const to = Math.min(values.length - 1, last + 1);
  const at = (/** @type {number} */ g) =>
    `${knotPx(frame, grid[g]).toFixed(1)} ${(frame.axisY - values[g] * height).toFixed(1)}`;
  const points = [];
  for (let g = from; g <= to; g++) points.push(at(g));
  return `M ${points.join(" L ")}`;
}

/**
 * Draw the overlay into the chart's basis layer, which chart.js puts beneath
 * the curve, replacing what it held; the gestures redraw it as a knot moves.
 * @param {SVGSVGElement|SVGElement} svg @param {KnotFrame|null} frame @param {Readonly<KnotUi>} ui
 */
export function drawKnotBasis(svg, frame, ui) {
  const layer = svg.querySelector(":scope > .knot-basis-layer");
  if (!layer) return;
  layer.replaceChildren();
  const overlay = frame ? basisOverlay(frame, ui) : null;
  if (!overlay || !frame) return;
  const height = basisBandHeight(frame);
  // A translucent backing mutes the exposure beneath, so the bumps read on either theme.
  layer.appendChild(el("rect", {
    class: "knot-basis-band",
    x: frame.left,
    y: frame.axisY - height - BAND_HEADROOM,
    width: frame.right - frame.left,
    height: height + BAND_HEADROOM
  }));
  // The faint ones first, so the reached ones sit on top.
  const order = overlay.curves.map((_, index) => index)
    .sort((left, right) => Number(overlay.reached[left]) - Number(overlay.reached[right]));
  for (const index of order) {
    const d = bumpPath(frame, overlay.grid, overlay.curves[index], height);
    if (d === null) continue;
    const reached = overlay.reached[index];
    layer.appendChild(el("path", {
      class: reached ? "knot-basis is-reached" : "knot-basis",
      d,
      "data-basis-index": index,
      ...(reached ? { style: `--knot-basis: var(--basis-${index % BASIS_PALETTE_SIZE})` } : {})
    }));
  }
}
