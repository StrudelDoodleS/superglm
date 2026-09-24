// @ts-check
// The Breaks-mode overlay: segment shading, one dashed line per break with its
// label and remove handle, and a degree chip per segment on a band axis.

import { breakX, degreeName } from "../breaks.js";
import { el, text } from "./svg.js";

/** @typedef {import('../api/contracts.js').BreakDraft} BreakDraft */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

const HIT_WIDTH = 16;
const PILL_HEIGHT = 16;
const CHAR_WIDTH = 6.4;

/**
 * @param {SVGElement} svg
 * @param {{term:TermPayload, draft:BreakDraft, sx:(v:number)=>number,
 *   margin:{left:number, top:number}, innerW:number, innerH:number}} options
 */
export function drawBreakOverlay(svg, { term, draft, sx, margin, innerW, innerH }) {
  // A polynomial uses no breaks; the draft keeps them for a switch back.
  if (draft.form === "polynomial") return;
  const left = margin.left;
  const right = margin.left + innerW;
  const layer = el("g", { class: "break-layer" });
  svg.appendChild(layer);
  const xs = draft.breaks.map((value) => sx(breakX(term, value)));
  const edges = [left, ...xs.map((x) => Math.min(Math.max(x, left), right)), right];
  drawSegments(layer, edges, margin.top, innerH);
  xs.forEach((x, index) => {
    // A break zoomed out of view has no line to drag.
    if (left <= x && x <= right) drawBreak(layer, draft, index, x, margin.top, innerH);
  });
  if (draft.form === "piecewise" && term.transform?.axis) {
    drawDegreeChips(layer, draft, edges, margin.top);
  }
}

/** @param {SVGElement} layer @param {number[]} edges @param {number} top @param {number} innerH */
function drawSegments(layer, edges, top, innerH) {
  for (let i = 1; i + 1 < edges.length; i += 2) {
    layer.appendChild(el("rect", {
      class: "break-segment",
      x: edges[i],
      y: top,
      width: edges[i + 1] - edges[i],
      height: innerH
    }));
  }
}

/**
 * @param {SVGElement} layer @param {BreakDraft} draft @param {number} index
 * @param {number} x @param {number} top @param {number} innerH
 */
function drawBreak(layer, draft, index, x, top, innerH) {
  const label = String(draft.breaks[index]);
  layer.appendChild(el("line", { class: "break-line", x1: x, y1: top, x2: x, y2: top + innerH }));
  layer.appendChild(el("rect", {
    class: "break-hit",
    x: x - HIT_WIDTH / 2,
    y: top,
    width: HIT_WIDTH,
    height: innerH,
    "data-break-index": index
  }));
  const width = label.length * CHAR_WIDTH + 30;
  const pillTop = top - PILL_HEIGHT - 5;
  const pill = el("g", {
    class: "break-label",
    tabindex: 0,
    role: "slider",
    "aria-label": `Break at ${label}`,
    "aria-valuetext": label,
    "data-break-index": index
  });
  pill.appendChild(el("rect", {
    x: x - width / 2, y: pillTop, width, height: PILL_HEIGHT, rx: PILL_HEIGHT / 2, ry: PILL_HEIGHT / 2
  }));
  text(pill, x - width / 2 + 8, pillTop + 11.5, label, "break-label-text", "start");
  pill.appendChild(removeHandle(label, index, x + width / 2 - 9, pillTop + PILL_HEIGHT / 2));
  layer.appendChild(pill);
}

/** @param {string} label @param {number} index @param {number} cx @param {number} cy */
function removeHandle(label, index, cx, cy) {
  const handle = el("g", {
    class: "break-remove",
    role: "button",
    "aria-label": `Remove break at ${label}`,
    "data-break-remove": index
  });
  handle.appendChild(el("circle", { cx, cy, r: 6 }));
  handle.appendChild(el("path", { d: `M${cx - 2.4} ${cy - 2.4}l4.8 4.8m0-4.8l-4.8 4.8` }));
  return handle;
}

/** @param {SVGElement} layer @param {BreakDraft} draft @param {number[]} edges @param {number} top */
function drawDegreeChips(layer, draft, edges, top) {
  draft.degrees.forEach((degree, segment) => {
    const name = degreeName(degree);
    const width = name.length * CHAR_WIDTH + 14;
    const x = edges[segment] + 6;
    const chip = el("g", {
      class: "degree-chip",
      tabindex: 0,
      role: "button",
      "aria-label": `Segment ${segment + 1}: ${name}. Change degree`,
      "data-segment": segment
    });
    chip.appendChild(el("rect", {
      x, y: top + 8, width, height: PILL_HEIGHT, rx: PILL_HEIGHT / 2, ry: PILL_HEIGHT / 2
    }));
    text(chip, x + width / 2, top + 19.5, name, "degree-chip-text", "middle");
    layer.appendChild(chip);
  });
}
