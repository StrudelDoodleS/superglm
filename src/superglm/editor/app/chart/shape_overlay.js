// @ts-check
// The shaped-range overlay: one light band per pinned range, named by its
// shape, drawn beneath the curve so the pinned stretch reads as part of the plot.

import { shapeRangeDescription, shapeRangeExtent } from "../shapes.js";
import { el, text } from "./svg.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../shapes.js').DisplayAxis} DisplayAxis */

const LABEL_INSET = 6;
const LABEL_BASELINE = 14;

/**
 * @param {SVGElement} svg
 * @param {{term:TermPayload, view:DisplayAxis, sx:(v:number)=>number,
 *   margin:{left:number, top:number}, innerW:number, innerH:number}} options
 */
export function drawShapeOverlay(svg, { term, view, sx, margin, innerW, innerH }) {
  if (!term.shape.ranges.length) return;
  const left = margin.left;
  const right = margin.left + innerW;
  const layer = el("g", { class: "shape-layer" });
  for (const range of term.shape.ranges) {
    const extent = shapeRangeExtent(term, view, range);
    if (!extent) continue;
    // Clamp to the plot so a zoom keeps the label in view; a range zoomed
    // entirely out of view has nothing to show.
    const x0 = Math.max(left, sx(extent[0]));
    const x1 = Math.min(right, sx(extent[1]));
    if (x1 <= x0) continue;
    const band = el("g", {
      class: "shape-range",
      "data-shape-degree": range.degree,
      "data-popover-title": range.label,
      "data-popover-body": shapeRangeDescription(range)
    });
    band.appendChild(el("rect", { x: x0, y: margin.top, width: x1 - x0, height: innerH }));
    text(band, x0 + LABEL_INSET, margin.top + LABEL_BASELINE, range.label, "shape-range-label", "start");
    layer.appendChild(band);
  }
  svg.appendChild(layer);
}
