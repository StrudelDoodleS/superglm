// @ts-check
// A spline term's knots on the chart. In every mode small ticks under the
// x-axis mark the knots shown. In Knots mode they are diamond handles on the
// axis with faint dashed guides up through the plot, over a light band where
// a click adds one; while one is dragged a zone below the axis removes it, and
// a dark tag over the selected or dragged knot says where it is. A knot change
// waiting for Refit draws the knots it places amber, and leaves the in-force
// knots it moves or removes as dashed grey ghosts, a removed one crossed out.

import { AT_LEAST_ONE, knotAxis, knotTagText, shownKnots } from "../knots.js";
import { labelWidth } from "./shape_overlay.js";
import { el, line, text } from "./svg.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../knots.js').KnotAxis} KnotAxis */
/**
 * What the knot layer and the gestures need from one drawn chart; chart.js
 * keeps it on the svg as ``_knotFrame``.
 * @typedef {object} KnotFrame
 * @property {KnotAxis} axis
 * @property {number[]} positions the knots shown, ascending
 * @property {boolean[]} placed which of them a waiting change places
 * @property {{x:number, removed:boolean}[]} ghosts in-force knots a waiting change moves or removes
 * @property {boolean} editing Knots mode is on
 * @property {number} xMin the x range drawn
 * @property {number} xMax
 * @property {number} left the plot's edges, in px
 * @property {number} right
 * @property {number} top
 * @property {number} axisY the x-axis line
 * @property {number} bottom the drawing's lower edge
 */
/**
 * A gesture in progress, which the gestures module owns: the selected knot by
 * position, a drag from the knot at ``from`` to ``x``, and the grid point a
 * click on the band would add a knot at.
 * @typedef {object} KnotUi
 * @property {number|null} selected
 * @property {{from:number, x:number, moved:boolean, remove:boolean}|null} drag
 * @property {number|null} hover
 */
/**
 * @typedef {object} KnotLayout
 * @property {{px:number, x:number, placed:boolean}[]} ticks
 * @property {{px:number, removed:boolean}[]} ghosts
 * @property {{px:number, placed:boolean, selected:boolean}[]} guides
 * @property {{index:number, x:number, px:number, cy:number, placed:boolean, selected:boolean,
 *   removing:boolean}[]} handles
 * @property {boolean} band
 * @property {number|null} adding where the dashed ghost of a new knot goes, in px
 * @property {{top:number, height:number, label:string, labelX:number}|null} removeZone the
 *   zone, its label in the half the knot being removed is not in
 * @property {{px:number, text:string}|null} tag
 */

/** @type {Readonly<KnotUi>} */
export const NO_KNOT_GESTURE = Object.freeze({ selected: null, drag: null, hover: null });

const HANDLE_RADIUS = 7;
const GHOST_RADIUS = 6;
const HIT_HALF_WIDTH = 11;
const HIT_HALF_HEIGHT = 13;
const BAND_HALF = 9;
const BAND_HIT_HALF = 14;
// Longer than the axis's own 5px ticks, so a knot on a tick still reads as one.
const TICK_TOP = 2;
const TICK_BOTTOM = 9;
const CROSS_TOP = 10;
const CROSS_HALF = 4;
const ZONE_OFFSET = 28;
const ZONE_HEIGHT = 22;
const TAG_TOP = 34;
const TAG_HEIGHT = 18;
const TAG_BASELINE = 21;
const TAG_PADDING = 6;
export const REMOVE_LABEL = "Drop here to remove the knot";

/**
 * The knot frame for ``term`` drawn on ``plot``, or null when it has no
 * knots to show.
 * @param {TermPayload} term
 * @param {{xMin:number, xMax:number, left:number, right:number, top:number, axisY:number,
 *   bottom:number}} plot
 * @param {boolean} editing
 * @returns {KnotFrame|null}
 */
export function knotFrame(term, plot, editing) {
  const axis = knotAxis(term);
  const shown = shownKnots(term);
  if (!axis || !shown) return null;
  const inForce = [...(term.knots?.positions ?? [])].sort((left, right) => left - right);
  const { placed, ghosts } = shown.waiting
    ? waitingMarks(inForce, shown.positions, sameKnotTolerance(axis))
    : { placed: shown.positions.map(() => false), ghosts: [] };
  return { axis, positions: shown.positions, placed, ghosts, editing, ...plot };
}

/** Two positions this close are one knot: far below the grid. @param {KnotAxis} axis */
function sameKnotTolerance(axis) {
  return axis.step * 1e-6;
}

/**
 * What a waiting change does to the knots in force: which of its knots it
 * places (no knot in force sits there), and which knots in force it leaves,
 * each marked removed or moved. Knots carry no identity, so the knots it
 * places are paired with the ones it leaves, in order and as close as they
 * go; a knot left over is removed.
 * @param {readonly number[]} inForce ascending @param {readonly number[]} draft ascending
 * @param {number} tolerance
 * @returns {{placed:boolean[], ghosts:{x:number, removed:boolean}[]}}
 */
export function waitingMarks(inForce, draft, tolerance) {
  /** @param {number} a @param {number} b */
  const same = (a, b) => Math.abs(a - b) <= tolerance;
  const placed = draft.map((x) => !inForce.some((y) => same(x, y)));
  const leaving = inForce.filter((y) => !draft.some((x) => same(x, y)));
  const arriving = draft.filter((_, index) => placed[index]);
  const removed = unpairedLeaving(leaving, arriving);
  return { placed, ghosts: leaving.map((x, index) => ({ x, removed: removed.has(index) })) };
}

/**
 * The leaving knots, by index, that no arriving knot pairs with: an
 * order-preserving pairing of every arriving knot with a leaving one that
 * least moves them in total, by dynamic programming over the two lists.
 * @param {readonly number[]} leaving @param {readonly number[]} arriving
 * @returns {Set<number>}
 */
function unpairedLeaving(leaving, arriving) {
  const m = leaving.length;
  const n = arriving.length;
  if (m <= n) return new Set();
  // cost[i][j]: the least total distance pairing the first j arriving knots
  // with j of the first i leaving ones.
  const cost = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(Infinity));
  for (let i = 0; i <= m; i++) cost[i][0] = 0;
  /** @param {number} i @param {number} j */
  const paired = (i, j) => cost[i - 1][j - 1] + Math.abs(leaving[i - 1] - arriving[j - 1]);
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= Math.min(i, n); j++) cost[i][j] = Math.min(cost[i - 1][j], paired(i, j));
  }
  const removed = new Set();
  for (let i = m, j = n; i >= 1; i--) {
    if (j > 0 && cost[i][j] === paired(i, j)) j -= 1;
    else removed.add(i - 1);
  }
  return removed;
}

/** A chart x in px. @param {KnotFrame} frame @param {number} x */
export function knotPx(frame, x) {
  const span = Math.max(frame.xMax - frame.xMin, 1e-12);
  return frame.left + ((x - frame.xMin) / span) * (frame.right - frame.left);
}

/** The chart x at a px. @param {KnotFrame} frame @param {number} px */
export function knotX(frame, px) {
  const width = Math.max(frame.right - frame.left, 1e-12);
  return frame.xMin + ((px - frame.left) / width) * (frame.xMax - frame.xMin);
}

/** @param {KnotFrame} frame @param {number} px */
function inView(frame, px) {
  return px >= frame.left - 0.5 && px <= frame.right + 0.5;
}

/**
 * The shown knot at ``x``, by index, or null.
 * @param {KnotFrame} frame @param {number|null} x
 */
export function knotIndex(frame, x) {
  if (x === null) return null;
  const tolerance = sameKnotTolerance(frame.axis);
  const index = frame.positions.findIndex((position) => Math.abs(position - x) <= tolerance);
  return index < 0 ? null : index;
}

/**
 * The knot handle under an svg point, by index: the nearest within reach.
 * @param {KnotFrame} frame @param {{x:number, y:number}} point
 * @returns {number|null}
 */
export function knotAt(frame, point) {
  if (Math.abs(point.y - frame.axisY) > HIT_HALF_HEIGHT) return null;
  /** @type {number|null} */
  let found = null;
  let reach = HIT_HALF_WIDTH;
  frame.positions.forEach((x, index) => {
    const px = knotPx(frame, x);
    const distance = Math.abs(px - point.x);
    if (inView(frame, px) && distance <= reach) {
      found = index;
      reach = distance;
    }
  });
  return found;
}

/** Whether an svg point is on the band along the axis. @param {KnotFrame} frame @param {{x:number, y:number}} point */
export function onKnotBand(frame, point) {
  return point.x >= frame.left && point.x <= frame.right &&
    Math.abs(point.y - frame.axisY) <= BAND_HIT_HALF;
}

/** @param {KnotFrame} frame */
function zoneTop(frame) {
  return Math.min(frame.axisY + ZONE_OFFSET, frame.bottom - ZONE_HEIGHT - 2);
}

/** Whether a dragged knot at this svg point is dropped to be removed. @param {KnotFrame} frame @param {{y:number}} point */
export function inRemoveZone(frame, point) {
  return point.y >= zoneTop(frame) - 2;
}

/**
 * Where every knot mark goes, without a DOM.
 * @param {KnotFrame} frame @param {Readonly<KnotUi>} ui @returns {KnotLayout}
 */
export function knotLayout(frame, ui) {
  const drag = frame.editing ? ui.drag : null;
  const dragged = drag ? knotIndex(frame, drag.from) : null;
  const selected = frame.editing ? knotIndex(frame, ui.selected) : null;
  const marks = frame.positions
    .map((position, index) => {
      const x = drag && index === dragged ? drag.x : position;
      const moving = Boolean(drag && index === dragged && drag.moved);
      return { index, x, px: knotPx(frame, x), placed: frame.placed[index] || moving };
    })
    .filter((mark) => inView(frame, mark.px));
  const ghosts = frame.ghosts
    .map((ghost) => ({ px: knotPx(frame, ghost.x), removed: ghost.removed }))
    .filter((ghost) => inView(frame, ghost.px));
  if (!frame.editing) {
    return {
      ticks: marks.map(({ px, x, placed }) => ({ px, x, placed })),
      ghosts, guides: [], handles: [], band: false, adding: null, removeZone: null, tag: null
    };
  }
  const middle = (frame.left + frame.right) / 2;
  const draggedPx = drag ? knotPx(frame, drag.x) : middle;
  const removeZone = drag
    ? {
        top: zoneTop(frame),
        height: ZONE_HEIGHT,
        label: frame.positions.length > 1 ? REMOVE_LABEL : AT_LEAST_ONE,
        // The knot dropping into the zone never covers its label.
        labelX: !drag.remove ? middle
          : draggedPx < middle ? (middle + frame.right) / 2 : (frame.left + middle) / 2
      }
    : null;
  const handles = marks.map((mark) => {
    const removing = Boolean(drag?.remove && mark.index === dragged);
    return {
      ...mark,
      cy: removing && removeZone ? removeZone.top + removeZone.height / 2 : frame.axisY,
      selected: mark.index === (dragged ?? selected),
      removing
    };
  });
  const tagged = handles.find((handle) => handle.selected);
  const full = frame.axis.maxCount !== null && frame.positions.length >= frame.axis.maxCount;
  const adding = drag || full || ui.hover === null ? null : knotPx(frame, ui.hover);
  return {
    ticks: [],
    ghosts,
    guides: handles
      .filter((handle) => !handle.removing)
      .map(({ px, placed, selected: isSelected }) => ({ px, placed, selected: isSelected })),
    handles,
    band: true,
    adding: adding !== null && inView(frame, adding) ? adding : null,
    removeZone,
    tag: tagged ? { px: tagged.px, text: knotTagText(tagged.x, frame.axis) } : null
  };
}

/** @param {number} cx @param {number} cy @param {number} r */
function diamond(cx, cy, r) {
  const at = (/** @type {number} */ v) => v.toFixed(2);
  return `M ${at(cx)} ${at(cy - r)} L ${at(cx + r)} ${at(cy)} L ${at(cx)} ${at(cy + r)} `
    + `L ${at(cx - r)} ${at(cy)} Z`;
}

/** @param {...(string|false)} names */
function classes(...names) {
  return names.filter(Boolean).join(" ");
}

/**
 * Draw the knot layer, replacing the one drawn before; null takes it away.
 * The gestures call this as a knot is dragged, without redrawing the chart.
 * @param {SVGSVGElement|SVGElement} svg @param {KnotFrame|null} frame @param {Readonly<KnotUi>} ui
 */
export function drawKnotLayer(svg, frame, ui) {
  const old = svg.querySelector(":scope > .knot-layer");
  if (!frame) {
    old?.remove();
    return;
  }
  const layout = knotLayout(frame, ui);
  const layer = el("g", { class: classes("knot-layer", frame.editing && "is-editing") });
  if (old) old.replaceWith(layer);
  else svg.insertBefore(layer, svg.querySelector(":scope > .legend-layer"));
  const { left, right, axisY, top } = frame;
  if (layout.band) {
    layer.appendChild(el("rect", {
      class: "knot-band", x: left, y: axisY - BAND_HALF, width: right - left, height: 2 * BAND_HALF,
      rx: 3, ry: 3
    }));
    layer.appendChild(el("rect", {
      class: "knot-band-hit", x: left, y: axisY - BAND_HIT_HALF, width: right - left,
      height: 2 * BAND_HIT_HALF
    }));
  }
  for (const guide of layout.guides) {
    line(layer, guide.px, top, guide.px, axisY,
      classes("knot-guide", guide.placed && "is-placed", guide.selected && "is-selected"));
  }
  for (const tick of layout.ticks) {
    const mark = line(layer, tick.px, axisY + TICK_TOP, tick.px, axisY + TICK_BOTTOM,
      classes("knot-tick", tick.placed && "is-placed"));
    mark.setAttribute("data-knot-x", String(tick.x));
  }
  for (const ghost of layout.ghosts) {
    layer.appendChild(el("path", {
      class: classes("knot-ghost", ghost.removed && "is-removed"),
      d: diamond(ghost.px, axisY, GHOST_RADIUS)
    }));
    if (ghost.removed) {
      const y = axisY + CROSS_TOP;
      const d = `M ${ghost.px - CROSS_HALF} ${y} l ${2 * CROSS_HALF} ${2 * CROSS_HALF} `
        + `M ${ghost.px + CROSS_HALF} ${y} l ${-2 * CROSS_HALF} ${2 * CROSS_HALF}`;
      layer.appendChild(el("path", { class: "knot-ghost-cross", d }));
    }
  }
  if (layout.removeZone) {
    const zone = layout.removeZone;
    layer.appendChild(el("rect", {
      class: "knot-remove-zone", x: left, y: zone.top, width: right - left, height: zone.height,
      rx: 4, ry: 4
    }));
    text(layer, zone.labelX, zone.top + zone.height / 2 + 4, zone.label, "knot-remove-label", "middle");
  }
  for (const handle of layout.handles) {
    layer.appendChild(el("path", {
      class: classes(
        "knot-handle",
        handle.placed && "is-placed",
        handle.selected && "is-selected",
        handle.removing && "is-removing"
      ),
      d: diamond(handle.px, handle.cy, HANDLE_RADIUS),
      "data-knot-x": String(handle.x)
    }));
    layer.appendChild(el("rect", {
      class: "knot-hit",
      x: handle.px - HIT_HALF_WIDTH,
      y: handle.cy - HIT_HALF_HEIGHT,
      width: 2 * HIT_HALF_WIDTH,
      height: 2 * HIT_HALF_HEIGHT,
      "data-knot-index": handle.index
    }));
  }
  if (layout.adding !== null) {
    layer.appendChild(el("path", { class: "knot-add", d: diamond(layout.adding, axisY, GHOST_RADIUS) }));
  }
  if (layout.tag) drawTag(layer, layout.tag, axisY);
}

/**
 * The dark tag over a knot; in the document before it is measured.
 * @param {SVGElement} layer @param {{px:number, text:string}} tag @param {number} axisY
 */
function drawTag(layer, tag, axisY) {
  const group = el("g", { class: "knot-tag" });
  layer.appendChild(group);
  const label = text(group, tag.px, axisY - TAG_BASELINE, tag.text, "knot-tag-label", "middle");
  const width = labelWidth(label) + 2 * TAG_PADDING;
  group.insertBefore(el("rect", {
    class: "knot-tag-box", x: tag.px - width / 2, y: axisY - TAG_TOP, width, height: TAG_HEIGHT,
    rx: 3, ry: 3
  }), label);
}
