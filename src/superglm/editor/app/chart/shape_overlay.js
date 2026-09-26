// @ts-check
// The shaped-range overlay: one light band per pinned range, named by its
// shape, drawn beneath the curve so the pinned stretch reads as part of the plot.

import { shapeRangeDescription, shapeRangeExtent } from "../shapes.js";
import { el, text } from "./svg.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../shapes.js').DisplayAxis} DisplayAxis */

const LABEL_INSET = 10;
const LABEL_BASELINE = 18;
const TAG_HEIGHT = 18;
const TAG_PADDING = 6;

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
  // In the document before the bands are made, so each label can be measured.
  svg.appendChild(layer);
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
    const label = text(
      band, x0 + LABEL_INSET, margin.top + LABEL_BASELINE, range.label, "shape-range-label", "start"
    );
    layer.appendChild(band);
    // The label sits on a small tag at the band's top; the band's own rect
    // stays first so the range's extent is the first rect the band holds.
    band.insertBefore(el("rect", {
      class: "shape-range-tag",
      x: x0 + LABEL_INSET - TAG_PADDING,
      y: margin.top + LABEL_BASELINE - TAG_HEIGHT + 4,
      width: labelWidth(label) + TAG_PADDING * 2,
      height: TAG_HEIGHT,
      rx: 4,
      ry: 4
    }), label);
  }
}

/** @param {SVGElement} label */
function labelWidth(label) {
  const measure = /** @type {{getComputedTextLength?:()=>number}} */ (label).getComputedTextLength;
  const measured = typeof measure === "function" ? measure.call(label) : 0;
  return measured > 0 ? measured : (label.textContent || "").length * 6.5;
}
