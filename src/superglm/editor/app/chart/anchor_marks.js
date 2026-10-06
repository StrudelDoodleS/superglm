// @ts-check
// The selection anchor, shown on the chart. The point a click anchored gets a
// ring and a small "click · x" tag; once a Shift-click has spanned from it, the
// span's other end gets one too, tagged "Shift-click · x". x reads as the axis
// reads it: in its tick format on a numeric axis, as the level on a
// categorical one. The tags step aside while the pointer drags.

import { fmt } from "../format.js";
import { labelWidth } from "./shape_overlay.js";
import { el, text } from "./svg.js";

/** @typedef {import('../api/contracts.js').EditorViewState['selectionAnchor']} SelectionAnchor */
/** @typedef {import('../api/contracts.js').EditorViewState['selectionSpan']} SelectionSpan */
/**
 * The axis as drawn: each display point's x, the levels on a categorical
 * axis, and the source indices each display point shows.
 * @typedef {{x:number[], levels?:string[]|null, displayToSourceIndices:number[][]}} AnchorAxis
 */
/**
 * @typedef {object} AnchorMark
 * @property {"anchor"|"end"} role the anchor, or the far end of a Shift-click span
 * @property {number} display the display point it marks
 * @property {string} value that point's x as the axis reads it
 * @property {string} tag what its tag says
 */
/**
 * The drawn nodes of one mark. They are made once per drawing and moved or
 * hidden when the selection changes, so a selection never adds or removes
 * chart nodes.
 * @typedef {{ring:SVGElement, tag:SVGElement, box:SVGElement, label:SVGElement}} MarkNodes
 */
/** @typedef {{anchor:MarkNodes, end:MarkNodes}} AnchorMarkNodes */
/**
 * @typedef {object} AnchorScale
 * @property {(value:number)=>number} sx
 * @property {(value:number)=>number} sy
 * @property {number[]} x
 * @property {number[]} y
 * @property {number} xMin
 * @property {number} xMax
 * @property {{left:number, top:number}} margin
 * @property {number} innerW
 * @property {number} innerH
 */

const GESTURES = Object.freeze({ anchor: "click", end: "Shift-click" });
const RING_RADIUS = 7;
const TAG_HEIGHT = 16;
const TAG_PADDING = 6;
// The tag hangs below its point, starting a little to its left.
const TAG_DROP = 12;
const TAG_LEAD = 12;
const TAG_BASELINE = 11.5;

/**
 * The display point that shows a source index, or null when none does.
 * @param {AnchorAxis} axis @param {number} source
 */
function displayPointOf(axis, source) {
  const display = axis.displayToSourceIndices.findIndex(
    (sources) => Array.isArray(sources) && sources.map(Number).includes(source)
  );
  return display >= 0 ? display : null;
}

/** @param {AnchorAxis} axis @param {number} display */
function axisValue(axis, display) {
  return Array.isArray(axis.levels) && axis.levels[display] !== undefined
    ? String(axis.levels[display])
    : fmt(Number(axis.x[display]));
}

/**
 * Whether the selection is still the span a Shift-click made: the anchor it
 * spanned from has not moved, it reached past the anchor, and the selection is
 * what it selected.
 * @param {string} term @param {SelectionAnchor} anchor @param {SelectionSpan} span
 * @param {ReadonlySet<number>} selection
 */
function isShiftClickSpan(term, anchor, span, selection) {
  if (!anchor || !span || anchor.term !== term || span.term !== term) return false;
  if (anchor.index !== span.from || span.from === span.to) return false;
  return span.indices.length === selection.size &&
    span.indices.every((index) => selection.has(index));
}

/**
 * The points to mark on `term`'s axis: the anchor, then the far end of the
 * Shift-click span when the selection is that span.
 * @param {AnchorAxis} axis @param {string} term
 * @param {SelectionAnchor} anchor @param {SelectionSpan} span
 * @param {ReadonlySet<number>} selection source indices
 * @returns {AnchorMark[]}
 */
export function anchorMarks(axis, term, anchor, span, selection) {
  if (!anchor || anchor.term !== term) return [];
  const from = displayPointOf(axis, anchor.index);
  if (from === null) return [];
  /** @type {AnchorMark[]} */
  const marks = [mark("anchor", axis, from)];
  const to = span && isShiftClickSpan(term, anchor, span, selection)
    ? displayPointOf(axis, span.to)
    : null;
  if (to !== null && to !== from) marks.push(mark("end", axis, to));
  return marks;
}

/**
 * @param {"anchor"|"end"} role @param {AnchorAxis} axis @param {number} display
 * @returns {AnchorMark}
 */
function mark(role, axis, display) {
  const value = axisValue(axis, display);
  return { role, display, value, tag: `${GESTURES[role]} · ${value}` };
}

/**
 * The Shift-click span's two ends in axis order, as the status line names
 * them, or null when the selection is no such span.
 * @param {AnchorAxis} axis @param {string} term
 * @param {SelectionAnchor} anchor @param {SelectionSpan} span
 * @param {ReadonlySet<number>} selection
 * @returns {{lo:string, hi:string}|null}
 */
export function spanRange(axis, term, anchor, span, selection) {
  const marks = anchorMarks(axis, term, anchor, span, selection);
  if (marks.length < 2) return null;
  const [lo, hi] = marks.sort(
    (left, right) => Number(axis.x[left.display]) - Number(axis.x[right.display])
  );
  return { lo: lo.value, hi: hi.value };
}

/**
 * Make the marks' nodes, hidden: rings beneath the points, so each point sits
 * in its ring, and tags above them.
 * @param {SVGElement} svg @param {SVGElement} pointLayer
 * @returns {AnchorMarkNodes}
 */
export function createAnchorMarks(svg, pointLayer) {
  const rings = el("g", { class: "anchor-rings" });
  const tags = el("g", { class: "anchor-tags" });
  svg.insertBefore(rings, pointLayer);
  svg.insertBefore(tags, pointLayer.nextSibling);
  /** @param {"anchor"|"end"} role @returns {MarkNodes} */
  const nodes = (role) => {
    const ring = el("circle", {
      class: "anchor-ring",
      "data-role": role,
      r: RING_RADIUS,
      display: "none",
      "clip-path": "url(#plotInteractionClip)"
    });
    const tag = el("g", { class: "anchor-tag", "data-role": role, display: "none" });
    const box = el("rect", {
      class: "anchor-tag-box", height: TAG_HEIGHT, rx: TAG_HEIGHT / 2, ry: TAG_HEIGHT / 2
    });
    tag.appendChild(box);
    const label = text(tag, 0, 0, "", "anchor-tag-label", "start");
    rings.appendChild(ring);
    tags.appendChild(tag);
    return { ring, tag, box, label };
  };
  return { anchor: nodes("anchor"), end: nodes("end") };
}

/**
 * Move each mark's ring and tag to its point, or hide them: a mark with no
 * point, or with its point zoomed out of view, shows nothing. A tag stays
 * inside the plot, and goes above its point where it would leave the plot's
 * foot or sit on the other tag.
 * @param {AnchorMarkNodes} nodes @param {AnchorMark[]} marks @param {AnchorScale} scale
 */
export function placeAnchorMarks(nodes, marks, scale) {
  /** @type {{left:number, right:number, top:number}|null} */
  let placed = null;
  for (const role of /** @type {const} */ (["anchor", "end"])) {
    const { ring, tag, box, label } = nodes[role];
    const found = marks.find((candidate) => candidate.role === role);
    const x = found ? Number(scale.x[found.display]) : Number.NaN;
    if (!found || !(x >= scale.xMin && x <= scale.xMax)) {
      ring.setAttribute("display", "none");
      tag.setAttribute("display", "none");
      continue;
    }
    const cx = scale.sx(x);
    const cy = scale.sy(Number(scale.y[found.display]));
    ring.setAttribute("cx", String(cx));
    ring.setAttribute("cy", String(cy));
    ring.removeAttribute("display");
    tag.removeAttribute("display");
    label.textContent = found.tag;
    const width = labelWidth(label) + TAG_PADDING * 2;
    const plotRight = scale.margin.left + scale.innerW;
    const left = Math.max(scale.margin.left, Math.min(plotRight - width, cx - TAG_LEAD));
    const below = cy + TAG_DROP;
    const above = cy - TAG_DROP - TAG_HEIGHT;
    /** @type {boolean} */
    const blocked = placed !== null &&
      left < placed.right && placed.left < left + width &&
      Math.abs(placed.top - below) < TAG_HEIGHT;
    /** @type {number} */
    const top = below + TAG_HEIGHT > scale.margin.top + scale.innerH || blocked ? above : below;
    box.setAttribute("x", String(left));
    box.setAttribute("y", String(top));
    box.setAttribute("width", String(width));
    label.setAttribute("x", String(left + TAG_PADDING));
    label.setAttribute("y", String(top + TAG_BASELINE));
    placed = { left, right: left + width, top };
  }
}

/**
 * Mark `root` while the pointer drags on `svg`: pressed and moved past `slop`
 * px either way, until it is released. A drag of any kind counts, a box, a
 * pan, or a point or handle being moved. The mark sits outside the chart,
 * whose drawing a drag otherwise leaves alone.
 * @param {{addEventListener:(name:string, listener:(event:PointerEvent)=>void)=>void}} svg
 * @param {{dataset:DOMStringMap}} root
 * @param {number} slop
 */
export function bindDragWatch(svg, root, slop) {
  /** @type {{x:number, y:number}|null} */
  let press = null;
  const release = () => {
    press = null;
    delete root.dataset.dragging;
  };
  svg.addEventListener("pointerdown", (event) => {
    press = { x: event.clientX, y: event.clientY };
  });
  svg.addEventListener("pointermove", (event) => {
    if (!press || root.dataset.dragging === "true") return;
    if (Math.abs(event.clientX - press.x) > slop || Math.abs(event.clientY - press.y) > slop) {
      root.dataset.dragging = "true";
    }
  });
  svg.addEventListener("pointerup", release);
  svg.addEventListener("pointercancel", release);
  svg.addEventListener("lostpointercapture", release);
}
