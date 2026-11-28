// @ts-check
/**
 * Zoom and pan on a check's plot, as moves of its domain: pure functions from the scene's zoom
 * frame and a gesture in the scene's px to the domain to draw next, or null for the whole plot.
 * The scene is drawn again at that domain, so its axes, ticks and band are redrawn at the new
 * range; nothing is scaled. A domain stays inside the scene's whole one, and an axis is never
 * narrower than a 200th of it, as on the Editor chart.
 */

import { defineMachine } from "../state/machine.js";

/** @typedef {import('./check_plots.js').Domain} Domain */
/** @typedef {import('./check_plots.js').Plot} Plot */
/** @typedef {import('./check_plots.js').ZoomFrame} ZoomFrame */
/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {{x:number, y:number}} Point a position in the scene's px */
/** @typedef {'x'|'y'} Axis */

/** The zoom of one wheel tick, in and out, as the Editor chart's. */
export const ZOOM_IN = 0.82;
export const ZOOM_OUT = 1.22;
// The narrowest an axis may be, as a share of its whole range.
const NARROWEST = 1 / 200;
// A box narrower than this many px along an axis it zooms is a click, not a box.
const BOX_MIN_PX = 8;
// Two ranges this close, as a share of the whole, are the same.
const SAME = 1e-9;

/** @param {ZoomFrame} zoom @returns {Axis[]} the axes the scene zooms */
const axesOf = (zoom) => (zoom.base.x ? ["x", "y"] : ["y"]);

/** @param {[number, number]} range */
const widthOf = (range) => range[1] - range[0];

/**
 * The px `plot` spans along `axis`, low then high: its domain's ends where the axis zooms, else
 * its rect's. Quantile calibration keeps its zero row out of the y span.
 * @param {Plot} plot @param {Axis} axis @returns {[number, number]}
 */
function pxSpan(plot, axis) {
  const ends = plot[axis] ?? (axis === "x" ? [plot.rect.left, plot.rect.right] : [plot.rect.bottom, plot.rect.top]);
  return [Math.min(...ends), Math.max(...ends)];
}

/**
 * `range` moved, not cut, to lie inside `whole`, and no wider than it; a range that is not one
 * is the whole.
 * @param {[number, number]} range @param {[number, number]} whole @returns {[number, number]}
 */
function fitAxis(range, whole) {
  const [a, b] = range;
  if (!(Number.isFinite(a) && Number.isFinite(b) && b > a)) return [whole[0], whole[1]];
  const width = Math.min(b - a, widthOf(whole));
  const start = Math.min(Math.max(a, whole[0]), whole[1] - width);
  return [start, start + width];
}

/**
 * The domain a scene draws: `domain` fitted inside `base`, or `base` when there is none.
 * @param {Domain|null|undefined} domain @param {Domain} base @returns {Domain}
 */
export function fitDomain(domain, base) {
  if (!domain) return base;
  return {
    x: base.x && domain.x ? fitAxis(domain.x, base.x) : base.x,
    y: domain.y ? fitAxis(domain.y, base.y) : base.y,
  };
}

/**
 * Whether `domain` is the whole of `base` on every axis it zooms.
 * @param {Domain} domain @param {Domain} base @returns {boolean}
 */
export function isWhole(domain, base) {
  /** @param {[number, number]|null} range @param {[number, number]|null} whole */
  const same = (range, whole) => !whole || !range
    || (Math.abs(range[0] - whole[0]) <= SAME * widthOf(whole) && Math.abs(range[1] - whole[1]) <= SAME * widthOf(whole));
  return same(domain.x, base.x) && same(domain.y, base.y);
}

/** @param {Domain} next @param {ZoomFrame} zoom @returns {Domain|null} */
const settle = (next, zoom) => (isWhole(next, zoom.base) ? null : next);

/**
 * The plot `point` lies in, or the nearest one.
 * @param {ZoomFrame} zoom @param {Point} point @returns {Plot}
 */
export function plotAt(zoom, point) {
  /** @param {Plot} plot */
  const away = ({ rect }) => Math.hypot(Math.max(rect.left - point.x, 0, point.x - rect.right),
    Math.max(rect.top - point.y, 0, point.y - rect.bottom));
  return zoom.plots.reduce((best, plot) => (away(plot) < away(best) ? plot : best));
}

/**
 * The value at `px` along `axis` of `plot`, drawn at `domain`.
 * @param {Plot} plot @param {Domain} domain @param {Axis} axis @param {number} px @returns {number}
 */
function valueAt(plot, domain, axis, px) {
  const [p0, p1] = /** @type {[number, number]} */ (plot[axis]);
  const [d0, d1] = /** @type {[number, number]} */ (domain[axis]);
  return d0 + ((px - p0) * (d1 - d0)) / (p1 - p0);
}

/**
 * The domain after zooming by `factor` about `point`, which keeps its value; below 1 zooms in.
 * Every axis takes the same factor, so a Q-Q drawn with one range on both axes keeps it. A zoom that would make
 * an axis narrower than a 200th of its whole stops there.
 * @param {ZoomFrame} zoom @param {Point} point @param {number} factor @returns {Domain|null}
 */
export function zoomAbout(zoom, point, factor) {
  const { base, domain } = zoom;
  const axes = axesOf(zoom);
  const tightest = Math.max(...axes.map((axis) => (NARROWEST * widthOf(/** @type {[number, number]} */ (base[axis])))
    / widthOf(/** @type {[number, number]} */ (domain[axis]))));
  if (factor < tightest) {
    if (tightest >= 1) return settle(domain, zoom);
    factor = tightest;
  }
  const plot = plotAt(zoom, point);
  /** @type {Domain} */
  const next = { ...domain };
  for (const axis of axes) {
    const range = /** @type {[number, number]} */ (domain[axis]);
    const whole = /** @type {[number, number]} */ (base[axis]);
    const at = valueAt(plot, domain, axis, point[axis]);
    const width = Math.min(widthOf(range) * factor, widthOf(whole));
    const start = at - ((at - range[0]) * width) / widthOf(range);
    next[axis] = fitAxis([start, start + width], whole);
  }
  return settle(next, zoom);
}

/**
 * The domain after zooming by `factor` about the middle of the first plot (the keys' zoom).
 * @param {ZoomFrame} zoom @param {number} factor @returns {Domain|null}
 */
export function stepZoom(zoom, factor) {
  const [x, y] = /** @type {Axis[]} */ (["x", "y"]).map((axis) => {
    const [lo, hi] = pxSpan(zoom.plots[0], axis);
    return (lo + hi) / 2;
  });
  return zoomAbout(zoom, { x, y }, factor);
}

/**
 * The domain a box from `a` to `b` zooms to, the box cut to the plot `a` lies in: each axis the
 * scene zooms takes exactly the box's range along it, so a Q-Q box wider than it is tall gives x
 * a wider range than y. A box narrower than 8 px along an axis the scene zooms leaves the domain
 * as it is.
 * @param {ZoomFrame} zoom @param {Point} a @param {Point} b @returns {Domain|null}
 */
export function zoomToBox(zoom, a, b) {
  const { base, domain } = zoom;
  const plot = plotAt(zoom, a);
  const axes = axesOf(zoom);
  /** @param {Axis} axis @param {number} px */
  const inPlot = (axis, px) => {
    const [lo, hi] = pxSpan(plot, axis);
    return Math.min(hi, Math.max(lo, px));
  };
  if (axes.some((axis) => Math.abs(inPlot(axis, b[axis]) - inPlot(axis, a[axis])) < BOX_MIN_PX)) return settle(domain, zoom);
  /** @type {Domain} */
  const next = { ...domain };
  for (const axis of axes) {
    const whole = /** @type {[number, number]} */ (base[axis]);
    const ends = [a[axis], b[axis]].map((px) => valueAt(plot, domain, axis, inPlot(axis, px)));
    const middle = (ends[0] + ends[1]) / 2;
    const width = Math.max(Math.abs(ends[1] - ends[0]), NARROWEST * widthOf(whole));
    next[axis] = fitAxis([middle - width / 2, middle + width / 2], whole);
  }
  return settle(next, zoom);
}

/**
 * The domain after a drag from `from` to `to` on a scene whose zoom frame was `zoom` when the
 * drag began: the values under the pointer move with it, as far as the whole plot allows.
 * @param {ZoomFrame} zoom @param {Point} from @param {Point} to @returns {Domain|null}
 */
export function panBy(zoom, from, to) {
  const { base, domain } = zoom;
  const plot = plotAt(zoom, from);
  /** @type {Domain} */
  const next = { ...domain };
  for (const axis of axesOf(zoom)) {
    const range = /** @type {[number, number]} */ (domain[axis]);
    const shift = valueAt(plot, domain, axis, to[axis]) - valueAt(plot, domain, axis, from[axis]);
    next[axis] = fitAxis([range[0] - shift, range[1] - shift], /** @type {[number, number]} */ (base[axis]));
  }
  return settle(next, zoom);
}

/**
 * A check's viewport: the whole plot, or zoomed to `domain` on the rows `split` names
 * (`splitOf`). A zoom holds on those rows alone: a check whose evidence reads other rows is
 * drawn whole.
 * @typedef {{state: 'whole'} | {state: 'zoomed', split: string, domain: Domain}} Viewport
 */
/**
 * What moves a viewport: a zoom to `domain` on the rows `split`, the whole plot again (`home`,
 * which a zoom that settles whole is too), or the rows under its check replaced by `split`.
 * @typedef {{type: 'zoom', split: string, domain: Domain} | {type: 'home'}
 *   | {type: 'replace', split: string}} ViewportEvent
 */

/** A check's viewport's lifecycle; a check the page's map does not hold is whole. */
export const CHECK_VIEWPORT = defineMachine({
  name: "check viewport",
  initial: "whole",
  moves: {
    whole: { zoom: "zoomed" },
    zoomed: { zoom: "zoomed", home: "whole", replace: "whole" },
  },
});

/** @type {Readonly<Viewport>} */
const WHOLE = Object.freeze({ state: "whole" });

/**
 * The checks' viewports after `event` on `report`'s; the same map where the table takes no such
 * move, or the move changes nothing. Rows replaced by the very rows a zoom was made on replace
 * nothing.
 * @param {Readonly<Partial<Record<CheckReport, Viewport>>>} map @param {CheckReport} report
 * @param {ViewportEvent} event @returns {Readonly<Partial<Record<CheckReport, Viewport>>>}
 */
export function moveViewport(map, report, event) {
  const viewport = map[report] ?? WHOLE;
  if (CHECK_VIEWPORT.next(viewport.state, event.type) === null) return map;
  const zoomedOn = viewport.state === "zoomed" ? viewport.split : null;
  if (event.type === "replace" && event.split === zoomedOn) return map;
  if (event.type !== "zoom") {
    const next = { ...map };
    delete next[report];
    return next;
  }
  if (viewport.state === "zoomed" && zoomedOn === event.split
    && JSON.stringify(viewport.domain) === JSON.stringify(event.domain)) return map;
  return { ...map, [report]: { state: "zoomed", split: event.split, domain: event.domain } };
}

/**
 * The domain a check is drawn at: its viewport's zoom when made on `split`, the rows its evidence
 * reads, else null, the whole plot.
 * @param {Viewport|undefined} viewport @param {string} split @returns {Domain|null}
 */
export function viewportDomain(viewport, split) {
  return viewport?.state === "zoomed" && viewport.split === split ? viewport.domain : null;
}

/**
 * The rows a check's payload reads, as one key: its split's name and the identity of its rows,
 * so other rows under the same name and size are another key; empty where it reads none.
 * @param {{split?: {name: string, rows_id?: number}|null}|null} payload @returns {string}
 */
export function splitOf(payload) {
  const split = payload?.split;
  return split ? `${split.name}#${split.rows_id ?? ""}` : "";
}
