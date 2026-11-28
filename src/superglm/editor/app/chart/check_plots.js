// @ts-check
/**
 * The four checks' plots as scenes: pure functions from one check payload to the paths, labels
 * and hover targets of an SVG. Each role is one path, so the stylesheet alone colours a scene
 * (`check-{role}`), and a card's thumbnail and the large view are drawn by the same code. What
 * lies outside its band takes the `outside` role only while Highlight outside the band is on;
 * off, it is drawn as the rest of the edit is, and the band shows where it lies. A large view
 * draws at a domain, the part of each axis on screen: zooming draws the scene again at a narrower
 * one, so its axes, ticks and band are drawn at the new range, and what lies outside it is
 * neither drawn nor read out.
 */

import { escapeHTML, fmt, fmtCount, fmtPercent } from "../format.js";
import { fitDomain } from "./check_zoom.js";
import { niceTicks } from "./fan.js";

/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').CheckPayload} CheckPayload */
/** @typedef {import('../api/contracts.js').CheckBody} CheckBody */
/**
 * @typedef {'band'|'reference'|'edited'|'original'|'original-interval'|'outside'|'interval'|'weight'|'grid'} Role
 */
/** @typedef {{role: Role, d: string}} Layer one path per role; points are arcs inside the path */
/**
 * A hover target in px, its readout, and the readout's accessible text when the readout shortens
 * a name.
 * @typedef {{x:number, y:number, text:string, aria?:string}} Mark
 */
/**
 * A tick label, or with `axis` an axis or panel title.
 * @typedef {{x:number, y:number, text:string, anchor:'start'|'middle'|'end', axis?:boolean}} Label
 */
/** @typedef {{left:number, top:number, right:number, bottom:number}} Rect */
/**
 * The part of each axis a scene draws, in the axis's own values: the Q-Q's residuals, the PIT's
 * probability and count over expected, the quantile calibration's quantile index and log10
 * share, and A/E's ratio. `x` is null where the scene does not zoom across: A/E's groups.
 * @typedef {{x:[number, number]|null, y:[number, number]}} Domain
 */
/**
 * One plot of a scene, and the px its domain's ends sit at along each axis, low end first.
 * @typedef {{rect:Rect, x:[number, number]|null, y:[number, number]}} Plot
 */
/**
 * What a zoom reads from a scene: its whole domain, the one drawn, its plots (one per A/E panel,
 * all at the one domain), and whether its two axes keep one range (the Q-Q's, so y = x stays the
 * diagonal).
 * @typedef {{base:Domain, domain:Domain, plots:Plot[], square:boolean}} ZoomFrame
 */
/**
 * @typedef {{width:number, height:number, layers:Layer[], labels:Label[], marks:Mark[],
 *   zoom?:ZoomFrame|null}} Scene
 */
/** @typedef {{width:number, height:number, thumb?:boolean}} Box */
/**
 * What a scene draws beside the edit: the fitted model when `compare`, and what lies outside its
 * band in its own role when `highlight`; and at `domain`, when given, the part of each axis drawn.
 * @typedef {{compare:boolean, highlight:boolean, domain?:Domain|null}} SceneOptions
 */
/** @typedef {Record<Role, string[]>} Parts */

/**
 * @typedef {{theoretical:number[], observed:number[], band_lower:number[], band_upper:number[],
 *   index:number[], n:number}} QQStats
 */
/** @typedef {{counts:number[], edges:number[], expected:number, band_lower:number, band_upper:number}} PITStats */
/**
 * @typedef {{covariate:string, by:string, levels:string[]|null, ratio:number[], lower:number[],
 *   upper:number[], n:number[], weight:number[]}} AETable
 */
/** @typedef {{tables:AETable[]}} AEStats */
/**
 * @typedef {{quantiles:{p:number[], expected:number[], realised_exceedance:number[], n:number[]},
 *   band_lower:number[], band_upper:number[]}} CalibrationStats
 */

/** The order the layers are painted in: the ground first, the points over everything. */
const ROLES = /** @type {const} */ (
  ["grid", "weight", "band", "reference", "original-interval", "original", "interval", "edited", "outside"]
);

const THUMB_PAD = 2;
// Room for the tick labels and the axis titles around a large view's plot.
const MARGIN = { left: 52, right: 16, top: 32, bottom: 46 };
// A/E's panels: the gap between them, and the room for each one's title and tick labels.
const PANEL_GAP = { col: 24, row: 12 };
const PANEL_MARGIN = { left: 44, top: 20, bottom: 24 };
// The least height of an A/E panel's plot, and the room below the last row of panels.
const PANEL_PLOT_MIN = 120;
const AE_FOOT = 8;
// A Q-Q thumbnail's points, shared between the models when compared, a third of each model's
// share kept for the points outside its band.
const THUMB_QQ_POINTS = 60;
const THUMB_QQ_OUTSIDE_SHARE = 1 / 3;
// The share of a value axis left beyond the outermost drawn value; a Q-Q's beyond its points.
const PAD_SHARE = 0.04;
const QQ_PAD_SHARE = 0.05;
// The share of its bin a PIT bar fills, and of its group's slot a weight bar fills.
const BAR_SHARE = 0.84;
const WEIGHT_SHARE = 0.6;
// The part of a plot's height the tallest weight bar reaches.
const WEIGHT_HEIGHT = 1 / 5;
// How far the fitted model's point sits left of the edited one's in a group's slot, at most.
const COMPARE_SHIFT = 4;
// The px kept between quantile calibration's zero row and its least positive share.
const ZERO_ROW = { thumb: 8, view: 24 };

/** @param {number} v a coordinate, written to 0.01 px */
const n = (v) => String(Math.round(v * 100) / 100);

/** @param {Box} box */
const radius = (box) => (box.thumb ? 1.3 : 3);

/** @param {number} x @param {number} y @param {number} r a point, as two arcs */
const dot = (x, y, r) => `M ${n(x)} ${n(y)} m ${-r} 0 a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0`;

/** @param {number} left @param {number} right @param {number} base @param {number} top */
const bar = (left, right, base, top) => `M ${n(left)} ${n(base)} V ${n(top)} H ${n(right)} V ${n(base)} Z`;

/** @param {number} left @param {number} right @param {number} y */
const hline = (left, right, y) => `M ${n(left)} ${n(y)} H ${n(right)}`;

/** @param {number} x @param {number} from @param {number} to */
const vline = (x, from, to) => `M ${n(x)} ${n(from)} V ${n(to)}`;

/** @param {Array<[number, number]>} points */
const polyline = (points) => points.map(([x, y], i) => `${i ? "L" : "M"} ${n(x)} ${n(y)}`).join(" ");

/**
 * The linear map sending `d0` to `r0` and `d1` to `r1`.
 * @param {number} d0 @param {number} d1 @param {number} r0 @param {number} r1
 * @returns {(v: number) => number}
 */
function linear(d0, d1, r0, r1) {
  const k = (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

/**
 * The range of the finite `values`, widened by `share` of itself on each side.
 * @param {Iterable<number>} values @param {number} [share] @returns {[number, number]}
 */
function span(values, share = PAD_SHARE) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!(hi >= lo)) return [0, 1];
  if (hi === lo) return [lo - 0.5, hi + 0.5];
  const room = (hi - lo) * share;
  return [lo - room, hi + room];
}

/** @param {Box} box @returns {Rect} */
function plotRect(box) {
  const m = box.thumb ? { left: THUMB_PAD, right: THUMB_PAD, top: THUMB_PAD, bottom: THUMB_PAD } : MARGIN;
  return { left: m.left, top: m.top, right: Math.max(m.left + 1, box.width - m.right), bottom: Math.max(m.top + 1, box.height - m.bottom) };
}

/** @returns {Parts} */
function emptyParts() {
  return {
    grid: [], weight: [], band: [], reference: [], "original-interval": [], original: [], interval: [], edited: [], outside: []
  };
}

/**
 * @param {Box} box @param {Parts} parts @param {Label[]} labels @param {Mark[]} marks
 * @param {ZoomFrame|null} zoom @returns {Scene}
 */
function finish(box, parts, labels, marks, zoom) {
  const layers = ROLES.filter((role) => parts[role].length > 0).map((role) => ({ role, d: parts[role].join(" ") }));
  return { width: box.width, height: box.height, layers, labels: box.thumb ? [] : labels, marks, zoom };
}

/** @param {Rect} rect @param {number} x @param {number} y whether px (x, y) lies in `rect` */
const within = (rect, x, y) => x >= rect.left - 1e-6 && x <= rect.right + 1e-6 && y >= rect.top - 1e-6 && y <= rect.bottom + 1e-6;

/** @param {number} lo @param {number} hi @returns {(v: number) => boolean} whether a value is in [lo, hi] */
const between = (lo, hi) => (v) => v >= lo - 1e-9 * (hi - lo) && v <= hi + 1e-9 * (hi - lo);

/**
 * The part of the segment from `a` to `b` inside `rect`, or null (Liang and Barsky, "A new
 * concept and method for line clipping", ACM Trans. Graph. 3(1), 1984).
 * @param {[number, number]} a @param {[number, number]} b @param {Rect} rect
 * @returns {[[number, number], [number, number]]|null}
 */
function clipSegment(a, b, rect) {
  const [dx, dy] = [b[0] - a[0], b[1] - a[1]];
  let [t0, t1] = [0, 1];
  for (const [p, q] of [[-dx, a[0] - rect.left], [dx, rect.right - a[0]], [-dy, a[1] - rect.top], [dy, rect.bottom - a[1]]]) {
    if (p === 0) {
      if (q < 0) return null;
      continue;
    }
    const r = q / p;
    if (p < 0) t0 = Math.max(t0, r);
    else t1 = Math.min(t1, r);
    if (t0 > t1) return null;
  }
  return [[a[0] + t0 * dx, a[1] + t0 * dy], [a[0] + t1 * dx, a[1] + t1 * dy]];
}

/**
 * The parts of a polyline inside `rect`, as one path; empty when none is.
 * @param {Array<[number, number]>} points @param {Rect} rect @returns {string}
 */
function clipPolyline(points, rect) {
  /** @type {string[]} */
  const out = [];
  /** @type {[number, number]|null} */
  let end = null;
  for (let i = 1; i < points.length; i++) {
    const piece = clipSegment(points[i - 1], points[i], rect);
    if (!piece) continue;
    const [from, to] = piece;
    if (!end || end[0] !== from[0] || end[1] !== from[1]) out.push(`M ${n(from[0])} ${n(from[1])}`);
    out.push(`L ${n(to[0])} ${n(to[1])}`);
    end = to;
  }
  return out.join(" ");
}

/**
 * A bar cut to `rect`, or null when none of it is in it; a bar with no height at the foot stays.
 * @param {number} left @param {number} right @param {number} base @param {number} top @param {Rect} rect
 * @returns {{left:number, right:number, base:number, top:number}|null}
 */
function clipBar(left, right, base, top, rect) {
  const cut = { left: Math.max(left, rect.left), right: Math.min(right, rect.right), base: Math.min(base, rect.bottom), top: Math.max(top, rect.top) };
  return cut.right < cut.left || cut.top > cut.base ? null : cut;
}

/**
 * Tick texts for `values`: as `fmt` writes them, unless that writes two alike, when each keeps
 * the decimals the step between them needs.
 * @param {number[]} values @param {(v: number) => string} [write] @returns {string[]}
 */
function tickTexts(values, write = fmt) {
  const texts = values.map(write);
  if (new Set(texts).size === texts.length || values.length < 2) return texts;
  const step = Math.min(...values.slice(1).map((v, i) => Math.abs(v - values[i])));
  const digits = Math.min(12, Math.max(0, Math.ceil(-Math.log10(step) + 1e-9)));
  return values.map((v) => v.toFixed(digits));
}

/** @param {Rect} rect @param {Domain} base @param {Domain} domain @param {boolean} square @returns {ZoomFrame} */
const zoomFrame = (rect, base, domain, square) => ({
  base, domain, square, plots: [{ rect, x: [rect.left, rect.right], y: [rect.bottom, rect.top] }]
});

/**
 * A large view's grid lines, tick labels and axis titles around `rect`.
 * @param {Parts} parts @param {Label[]} labels @param {Rect} rect
 * @param {Array<[number, string]>} xs each x tick's px and text
 * @param {Array<[number, string]>} ys each y tick's px and text
 * @param {{x:string, y:string, xGrid:boolean}} titles
 */
function frame(parts, labels, rect, xs, ys, titles) {
  for (const [y, text] of ys) {
    parts.grid.push(hline(rect.left, rect.right, y));
    labels.push({ x: rect.left - 6, y: y + 4, text, anchor: "end" });
  }
  for (const [x, text] of xs) {
    if (titles.xGrid) parts.grid.push(vline(x, rect.top, rect.bottom));
    labels.push({ x, y: rect.bottom + 16, text, anchor: "middle" });
  }
  labels.push({ x: (rect.left + rect.right) / 2, y: rect.bottom + 36, text: titles.x, anchor: "middle", axis: true });
  labels.push({ x: 8, y: rect.top - 14, text: titles.y, anchor: "start", axis: true });
}

/**
 * `count` of `items`, evenly spaced and keeping both ends; all of them when there are no more.
 * @param {number[]} items @param {number} count @returns {number[]}
 */
function spaced(items, count) {
  if (count >= items.length) return items;
  if (count <= 0) return [];
  if (count === 1) return [items[0]];
  return Array.from({ length: count }, (_, j) => items[Math.round((j * (items.length - 1)) / (count - 1))]);
}

/**
 * The part of one side's half-plane, `inside(p) >= 0`, a closed polygon covers.
 * @param {Array<[number, number]>} points @param {(p: [number, number]) => number} inside
 * @returns {Array<[number, number]>}
 */
function clipSide(points, inside) {
  /** @type {Array<[number, number]>} */
  const kept = [];
  for (const [j, b] of points.entries()) {
    const a = points[(j + points.length - 1) % points.length];
    const [fa, fb] = [inside(a), inside(b)];
    if ((fa >= 0) !== (fb >= 0)) {
      const t = fa / (fa - fb);
      kept.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    if (fb >= 0) kept.push(b);
  }
  return kept;
}

/**
 * A closed polygon clipped to `rect`, one side at a time (Sutherland and Hodgman, "Reentrant
 * polygon clipping", Comm. ACM 17(1), 1974): exact for any polygon against a convex window.
 * @param {Array<[number, number]>} points @param {Rect} rect @returns {Array<[number, number]>}
 */
function clipPolygon(points, rect) {
  /** @type {Array<(p: [number, number]) => number>} */
  const sides = [(p) => p[0] - rect.left, (p) => rect.right - p[0], (p) => p[1] - rect.top, (p) => rect.bottom - p[1]];
  return sides.reduce(clipSide, points);
}

/** @param {QQStats} s @param {number} i */
const qqOutside = (s, i) => s.observed[i] < s.band_lower[i] || s.observed[i] > s.band_upper[i];

/**
 * The positions of a Q-Q body's points a scene draws: all of them, or for a thumbnail up to a
 * third of `budget` outside the band and evenly spaced others up to `budget` in all.
 * @param {QQStats} s @param {Box} box @param {number} budget @returns {number[]}
 */
function qqShown(s, box, budget) {
  const all = s.index.map((_, i) => i);
  if (!box.thumb) return all;
  const outside = spaced(all.filter((i) => qqOutside(s, i)), Math.round(budget * THUMB_QQ_OUTSIDE_SHARE));
  const inside = spaced(all.filter((i) => !qqOutside(s, i)), budget - outside.length);
  return [...outside, ...inside].sort((a, b) => a - b);
}

/**
 * What a scene function reads from its options: whether to colour what left the band, and the
 * domain asked for, which it fits inside its whole one.
 * @typedef {{highlight:boolean, domain:Domain|null}} Draw
 */

/**
 * @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box
 * @param {Draw} draw @returns {Scene}
 */
function qqScene(editedBody, originalBody, box, { highlight, domain: asked }) {
  const edited = /** @type {QQStats} */ (/** @type {unknown} */ (editedBody));
  const original = originalBody && /** @type {QQStats} */ (/** @type {unknown} */ (originalBody));
  const budget = original ? THUMB_QQ_POINTS / 2 : THUMB_QQ_POINTS;
  const shown = qqShown(edited, box, budget);
  const shownOriginal = original ? qqShown(original, box, budget) : [];
  /** @type {number[]} */
  const values = [];
  for (const i of shown) values.push(edited.theoretical[i], edited.observed[i]);
  if (original) for (const i of shownOriginal) values.push(original.theoretical[i], original.observed[i]);
  // One range for both axes, so y = x is the plot's diagonal; the points fill it, and the
  // band's flared ends are clipped to it. A zoom keeps the two ranges alike.
  const [lo, hi] = span(values, QQ_PAD_SHARE);
  const rect = plotRect(box);
  /** @type {Domain} */
  const base = { x: [lo, hi], y: [lo, hi] };
  const domain = fitDomain(asked, base);
  const [x0, x1] = /** @type {[number, number]} */ (domain.x);
  const sx = linear(x0, x1, rect.left, rect.right);
  const sy = linear(domain.y[0], domain.y[1], rect.bottom, rect.top);
  const r = radius(box);
  const parts = emptyParts();
  /** @type {Mark[]} */
  const marks = [];
  /** @type {(bound: number[]) => (i: number) => [number, number]} */
  const along = (bound) => (i) => [sx(edited.theoretical[i]), sy(bound[i])];
  const band = clipPolygon([...shown.map(along(edited.band_upper)), ...[...shown].reverse().map(along(edited.band_lower))], rect);
  if (band.length > 0) parts.band.push(`${polyline(band)} Z`);
  const diagonal = clipPolyline([[sx(lo), sy(lo)], [sx(hi), sy(hi)]], rect);
  if (diagonal) parts.reference.push(diagonal);
  for (const i of shownOriginal) {
    const [x, y] = [sx(original.theoretical[i]), sy(original.observed[i])];
    if (within(rect, x, y)) parts.original.push(dot(x, y, r));
  }
  for (const i of shown) {
    const x = sx(edited.theoretical[i]);
    const y = sy(edited.observed[i]);
    if (!within(rect, x, y)) continue;
    parts[highlight && qqOutside(edited, i) ? "outside" : "edited"].push(dot(x, y, r));
    marks.push({
      x,
      y,
      text: `Rank ${fmtCount(edited.index[i] + 1)} of ${fmtCount(edited.n)} · expected ${fmt(edited.theoretical[i])}`
        + ` · observed ${fmt(edited.observed[i])} · band ${fmt(edited.band_lower[i])} to ${fmt(edited.band_upper[i])}`
    });
  }
  /** @type {Label[]} */
  const labels = [];
  if (!box.thumb) {
    const xs = niceTicks(x0, x1, 6);
    const ys = niceTicks(domain.y[0], domain.y[1], 6);
    const [xText, yText] = [tickTexts(xs), tickTexts(ys)];
    frame(parts, labels, rect, xs.map((v, i) => [sx(v), xText[i]]), ys.map((v, i) => [sy(v), yText[i]]),
      { x: "Expected residual", y: "Observed residual", xGrid: true });
  }
  return finish(box, parts, labels, marks, zoomFrame(rect, base, domain, true));
}

/**
 * @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box
 * @param {Draw} draw @returns {Scene}
 */
function pitScene(editedBody, originalBody, box, { highlight, domain: asked }) {
  const edited = /** @type {PITStats} */ (/** @type {unknown} */ (editedBody));
  const original = originalBody && /** @type {PITStats} */ (/** @type {unknown} */ (originalBody));
  const share = (/** @type {PITStats} */ s, /** @type {number} */ i) => s.counts[i] / s.expected;
  const lower = edited.band_lower / edited.expected;
  const upper = edited.band_upper / edited.expected;
  const heights = [...edited.counts.keys()].map((i) => share(edited, i));
  if (original) heights.push(...original.counts.map((_, i) => share(original, i)));
  const [, top] = span([0, 1, upper, ...heights]);
  const rect = plotRect(box);
  const { edges } = edited;
  /** @type {Domain} */
  const base = { x: [edges[0], edges[edges.length - 1]], y: [0, top] };
  const domain = fitDomain(asked, base);
  const [x0, x1] = /** @type {[number, number]} */ (domain.x);
  const sx = linear(x0, x1, rect.left, rect.right);
  const sy = linear(domain.y[0], domain.y[1], rect.bottom, rect.top);
  const parts = emptyParts();
  /** @type {Mark[]} */
  const marks = [];
  const [bandTop, bandFoot] = [Math.max(rect.top, sy(upper)), Math.min(rect.bottom, sy(lower))];
  if (bandTop <= bandFoot) parts.band.push(`M ${n(rect.left)} ${n(bandTop)} H ${n(rect.right)} V ${n(bandFoot)} H ${n(rect.left)} Z`);
  if (between(...domain.y)(1)) parts.reference.push(hline(rect.left, rect.right, sy(1)));
  /** @param {Role} role @param {number} left @param {number} right @param {number} height */
  const draw = (role, left, right, height) => {
    const cut = clipBar(left, right, sy(0), sy(height), rect);
    if (cut) parts[role].push(bar(cut.left, cut.right, cut.base, cut.top));
    return cut;
  };
  for (const [i, count] of edited.counts.entries()) {
    const left = sx(edges[i]);
    const right = sx(edges[i + 1]);
    const gap = ((right - left) * (1 - BAR_SHARE)) / 2;
    let start = left + gap;
    // The fitted model's bar takes the left half of its bin's.
    if (original) {
      const middle = (left + right) / 2;
      draw("original", start, middle, share(original, i));
      start = middle;
    }
    const outside = count < edited.band_lower || count > edited.band_upper;
    const cut = draw(highlight && outside ? "outside" : "edited", start, right - gap, share(edited, i));
    // A bar is read out at the middle of the part of its bin in view, at its top or the plot's.
    const [inLeft, inRight] = [Math.max(left, rect.left), Math.min(right, rect.right)];
    if (!cut || inRight < inLeft) continue;
    marks.push({
      x: (inLeft + inRight) / 2,
      y: cut.top,
      text: `${fmt(edges[i])} to ${fmt(edges[i + 1])} · ${fmtCount(count)} values`
        + ` · band ${fmtCount(edited.band_lower)} to ${fmtCount(edited.band_upper)}`
    });
  }
  /** @type {Label[]} */
  const labels = [];
  if (!box.thumb) {
    const xs = niceTicks(x0, x1, 4);
    const ys = niceTicks(domain.y[0], domain.y[1], 4);
    const [xText, yText] = [tickTexts(xs), tickTexts(ys)];
    frame(parts, labels, rect, xs.map((v, i) => [sx(v), xText[i]]), ys.map((v, i) => [sy(v), yText[i]]),
      { x: "Cumulative probability", y: "Count over expected", xGrid: false });
  }
  return finish(box, parts, labels, marks, zoomFrame(rect, base, domain, false));
}

/** @param {AETable} table @param {number} i */
const aeGroup = (table, i) => (table.levels ? String(table.levels[i]) : `band ${i + 1}`);

// A readout names a group by at most this many characters (code points, so none is split), the
// last of them an ellipsis, so it is a few lines at most and fits the smallest frame; the panel's
// label keeps the whole name.
const READOUT_NAME = 40;

/** @param {string} name @returns {string} */
const readoutName = (name) => {
  const chars = Array.from(name);
  return chars.length > READOUT_NAME ? `${chars.slice(0, READOUT_NAME - 1).join("")}…` : name;
};

/** @param {AETable} table @param {number} i */
const aeDrawn = (table, i) => [table.ratio[i], table.lower[i], table.upper[i]].every(Number.isFinite);

/**
 * One A/E panel: the reference at 1, the weight bars along its foot, and each group's ratio
 * with its interval; the fitted model's beside it when compared. A ratio outside `domain` is not
 * drawn, and an interval is cut to the panel.
 * @param {Parts} parts @param {Label[]} labels @param {Mark[]} marks @param {Rect} rect
 * @param {[number, number]} domain @param {AETable} table @param {AETable|null} original @param {Box} box
 * @param {boolean} highlight
 */
function aePanel(parts, labels, marks, rect, domain, table, original, box, highlight) {
  const sy = linear(domain[0], domain[1], rect.bottom, rect.top);
  const inDomain = between(...domain);
  /** @param {number} x @param {number} lower @param {number} upper @returns {string|null} an interval cut to the panel */
  const interval = (x, lower, upper) => {
    const [foot, head] = [sy(lower), sy(upper)];
    return foot < rect.top || head > rect.bottom ? null : vline(x, Math.min(foot, rect.bottom), Math.max(head, rect.top));
  };
  const k = table.ratio.length;
  const slot = (rect.right - rect.left) / k;
  // Each model bins its own predicted Mean, so each series sits at its own groups' ordinals.
  const at = (/** @type {AETable} */ t, /** @type {number} */ i) => rect.left + ((i + 0.5) * (rect.right - rect.left)) / t.ratio.length;
  const shift = (/** @type {AETable} */ t) => (original ? Math.min(COMPARE_SHIFT, ((rect.right - rect.left) / t.ratio.length) * 0.15) : 0);
  const heaviest = Math.max(0, ...table.weight.filter(Number.isFinite));
  const r = radius(box);
  if (inDomain(1)) parts.reference.push(hline(rect.left, rect.right, sy(1)));
  if (original) {
    for (const i of original.ratio.keys()) {
      if (!aeDrawn(original, i)) continue;
      const x = at(original, i) - shift(original);
      const line = interval(x, original.lower[i], original.upper[i]);
      if (line) parts["original-interval"].push(line);
      if (inDomain(original.ratio[i])) parts.original.push(dot(x, sy(original.ratio[i]), r));
    }
  }
  for (let i = 0; i < k; i++) {
    const centre = at(table, i);
    if (heaviest > 0 && table.weight[i] > 0) {
      const h = (rect.bottom - rect.top) * WEIGHT_HEIGHT * (table.weight[i] / heaviest);
      parts.weight.push(bar(centre - (slot * WEIGHT_SHARE) / 2, centre + (slot * WEIGHT_SHARE) / 2, rect.bottom, rect.bottom - h));
    }
    if (!aeDrawn(table, i)) continue;
    const x = centre + shift(table);
    const y = sy(table.ratio[i]);
    const outside = table.lower[i] > 1 || table.upper[i] < 1;
    const line = interval(x, table.lower[i], table.upper[i]);
    if (line) parts.interval.push(line);
    if (!inDomain(table.ratio[i])) continue;
    parts[highlight && outside ? "outside" : "edited"].push(dot(x, y, r));
    const group = aeGroup(table, i);
    const values = ` · ${fmt(table.ratio[i])} · interval ${fmt(table.lower[i])} to ${fmt(table.upper[i])}`
      + ` · ${fmtCount(table.n[i])} rows`;
    const shown = readoutName(group);
    const mark = { x, y, text: `${table.covariate} ${shown}${values}` };
    marks.push(shown === group ? mark : { ...mark, aria: `${table.covariate} ${group}${values}` });
  }
  if (box.thumb) return;
  const ticks = niceTicks(domain[0], domain[1], 4);
  for (const [i, text] of tickTexts(ticks).entries()) {
    parts.grid.push(hline(rect.left, rect.right, sy(ticks[i])));
    labels.push({ x: rect.left - 6, y: sy(ticks[i]) + 4, text, anchor: "end" });
  }
  const names = Array.from({ length: k }, (_, i) => (table.levels ? String(table.levels[i]) : String(i + 1)));
  // Every `every`-th group is named, so the names do not run into each other.
  const every = Math.max(1, Math.ceil((Math.max(...names.map((s) => s.length)) * 6.5 + 8) / slot));
  for (const [i, name] of names.entries()) {
    if (i % every === 0) labels.push({ x: rect.left + (i + 0.5) * slot, y: rect.bottom + 16, text: name, anchor: "middle" });
  }
  const title = table.by === "band" ? table.covariate.charAt(0).toUpperCase() + table.covariate.slice(1) : table.covariate;
  labels.push({ x: rect.left, y: rect.top - 8, text: title, anchor: "start", axis: true });
}

/**
 * @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box
 * @param {Draw} draw @returns {Scene}
 */
function aeScene(editedBody, originalBody, box, { highlight, domain: asked }) {
  const edited = /** @type {AEStats} */ (/** @type {unknown} */ (editedBody));
  const original = originalBody && /** @type {AEStats} */ (/** @type {unknown} */ (originalBody));
  // A thumbnail draws the predicted-Mean bands alone.
  const tables = box.thumb ? [edited.tables.find((t) => t.by === "band") ?? edited.tables[0]] : edited.tables;
  const pairs = tables.map((table) => ({ table, original: original?.tables[edited.tables.indexOf(table)] ?? null }));
  /** @type {number[]} */
  const values = [1];
  for (const t of pairs.flatMap((pair) => (pair.original ? [pair.table, pair.original] : [pair.table]))) {
    for (const i of t.ratio.keys()) if (aeDrawn(t, i)) values.push(t.ratio[i], t.lower[i], t.upper[i]);
  }
  // One scale for every panel, so their ratios read alike; a zoom moves it in every panel at once.
  /** @type {Domain} */
  const base = { x: null, y: span(values) };
  const domain = fitDomain(asked, base);
  const cols = box.thumb ? 1 : Math.min(2, tables.length);
  const rows = Math.ceil(tables.length / cols);
  const outer = box.thumb
    ? plotRect(box)
    : { left: 8, top: MARGIN.top, right: box.width - MARGIN.right, bottom: box.height - AE_FOOT };
  const cellW = (outer.right - outer.left - PANEL_GAP.col * (cols - 1)) / cols;
  const fill = (outer.bottom - outer.top - PANEL_GAP.row * (rows - 1)) / rows;
  // A large view's panels keep a readable plot; when the box cannot hold them, the view grows.
  const cellH = box.thumb ? fill : Math.max(fill, PANEL_PLOT_MIN + PANEL_MARGIN.top + PANEL_MARGIN.bottom);
  const height = box.thumb ? box.height : Math.max(box.height, outer.top + rows * cellH + PANEL_GAP.row * (rows - 1) + AE_FOOT);
  const parts = emptyParts();
  /** @type {Label[]} */
  const labels = [];
  /** @type {Mark[]} */
  const marks = [];
  /** @type {Plot[]} */
  const plots = [];
  for (const [k, { table, original: fitted }] of pairs.entries()) {
    const left = outer.left + (k % cols) * (cellW + PANEL_GAP.col);
    const top = outer.top + Math.floor(k / cols) * (cellH + PANEL_GAP.row);
    const rect = box.thumb
      ? outer
      : { left: left + PANEL_MARGIN.left, top: top + PANEL_MARGIN.top, right: left + cellW, bottom: top + cellH - PANEL_MARGIN.bottom };
    aePanel(parts, labels, marks, rect, domain.y, table, fitted, box, highlight);
    plots.push({ rect, x: null, y: [rect.bottom, rect.top] });
  }
  labels.push({ x: 8, y: MARGIN.top - 14, text: "Actual over expected", anchor: "start", axis: true });
  marks.sort((a, b) => a.x - b.x);
  return finish({ ...box, height }, parts, labels, marks, { base, domain, plots, square: false });
}

/**
 * The ticks of a log10 share axis from `lo` to `hi`: its decades when it spans two, else round
 * shares inside it.
 * @param {number} lo @param {number} hi @returns {number[]} log10 shares
 */
function logTicks(lo, hi) {
  /** @type {number[]} */
  const decades = [];
  for (let j = Math.ceil(lo); j <= Math.floor(hi); j++) decades.push(j);
  if (decades.length >= 2) return decades;
  return niceTicks(10 ** lo, 10 ** hi, 4).filter((v) => v > 0).map(Math.log10).filter(between(lo, hi));
}

/**
 * @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box
 * @param {Draw} draw @returns {Scene}
 */
function calibrationScene(editedBody, originalBody, box, { highlight, domain: asked }) {
  const edited = /** @type {CalibrationStats} */ (/** @type {unknown} */ (editedBody));
  const original = originalBody && /** @type {CalibrationStats} */ (/** @type {unknown} */ (originalBody));
  const q = edited.quantiles;
  const shares = [...edited.band_lower, ...edited.band_upper, ...q.realised_exceedance, ...q.expected];
  if (original) shares.push(...original.quantiles.realised_exceedance);
  const [lo, hi] = span(shares.filter((v) => v > 0).map(Math.log10));
  const rect = plotRect(box);
  // x is the quantile's index, each in a slot of its own; y the log10 share.
  /** @type {Domain} */
  const base = { x: [-0.5, q.p.length - 0.5], y: [lo, hi] };
  const domain = fitDomain(asked, base);
  const [x0, x1] = /** @type {[number, number]} */ (domain.x);
  // A share of zero has no log: it gets its own row at the foot, below the least positive share,
  // and keeps it at any zoom.
  const zero = shares.includes(0);
  const shareFoot = rect.bottom - (zero ? ZERO_ROW[box.thumb ? "thumb" : "view"] : 0);
  const sy = linear(domain.y[0], domain.y[1], shareFoot, rect.top);
  const y = (/** @type {number} */ v) => (v === 0 ? rect.bottom : sy(Math.log10(v)));
  const xAt = linear(x0, x1, rect.left, rect.right);
  const r = radius(box);
  /** @param {number} x @param {number} v whether a share's point at x is in view */
  const inView = (x, v) => x >= rect.left - 1e-6 && x <= rect.right + 1e-6
    && (v === 0 || (y(v) >= rect.top - 1e-6 && y(v) <= shareFoot + 1e-6));
  const parts = emptyParts();
  /** @type {Mark[]} */
  const marks = [];
  const indices = [...q.p.keys()];
  const band = clipPolygon([
    ...indices.map((i) => /** @type {[number, number]} */ ([xAt(i), y(edited.band_upper[i])])),
    ...[...indices].reverse().map((i) => /** @type {[number, number]} */ ([xAt(i), y(edited.band_lower[i])]))
  ], rect);
  if (band.length > 0) parts.band.push(`${polyline(band)} Z`);
  const reference = clipPolyline(indices.map((i) => [xAt(i), y(q.expected[i])]), rect);
  if (reference) parts.reference.push(reference);
  // Both models' points sit at their quantile; the style alone tells them apart.
  if (original) {
    for (const i of indices) {
      const v = original.quantiles.realised_exceedance[i];
      if (inView(xAt(i), v)) parts.original.push(dot(xAt(i), y(v), r));
    }
  }
  for (const i of indices) {
    const realised = q.realised_exceedance[i];
    const outside = realised < edited.band_lower[i] || realised > edited.band_upper[i];
    const x = xAt(i);
    if (!inView(x, realised)) continue;
    parts[highlight && outside ? "outside" : "edited"].push(dot(x, y(realised), r));
    marks.push({
      x,
      y: y(realised),
      text: `Quantile ${fmtPercent(q.p[i])} · exceeded by ${fmtPercent(realised)}`
        + ` · band ${fmtPercent(edited.band_lower[i])} to ${fmtPercent(edited.band_upper[i])}`
    });
  }
  /** @type {Label[]} */
  const labels = [];
  if (!box.thumb) {
    const ticks = logTicks(domain.y[0], domain.y[1]);
    /** @type {Array<[number, string]>} */
    const rows = tickTexts(ticks.map((t) => 10 ** t), fmtPercent).map((text, i) => [sy(ticks[i]), text]);
    if (zero) rows.push([rect.bottom, "0"]);
    const columns = indices.filter((i) => xAt(i) >= rect.left - 1e-6 && xAt(i) <= rect.right + 1e-6);
    frame(parts, labels, rect, columns.map((i) => [xAt(i), fmtPercent(q.p[i])]), rows,
      { x: "Predicted quantile", y: "Share of rows above", xGrid: false });
  }
  return finish(box, parts, labels, marks, zoomFrame(rect, base, domain, false));
}

/** @type {Record<CheckReport, (edited: CheckBody, original: CheckBody|null, box: Box, draw: Draw) => Scene>} */
const SCENES = { check_qq: qqScene, check_pit: pitScene, check_ae: aeScene, check_calibration: calibrationScene };

/**
 * A check's plot as a scene in `box`: the edited model, with `compare` the fitted model when the
 * payload carries it, and with `highlight` what lies outside its band in the `outside` role, at
 * `domain` when one is given and else whole. An unavailable check is an empty scene with no zoom.
 * A thumbnail has no labels.
 * @param {CheckReport} report @param {CheckPayload} payload @param {Box} box
 * @param {SceneOptions} options
 * @returns {Scene}
 */
export function checkScene(report, payload, box, options) {
  const { edited, original } = payload.models;
  if (!payload.available || !edited.available) return finish(box, emptyParts(), [], [], null);
  return SCENES[report](edited, options.compare && original?.available ? original : null, box,
    { highlight: options.highlight, domain: options.domain ?? null });
}

/**
 * A scene as SVG markup: a path per layer, its labels, and an empty, hidden readout group the
 * view writes into on hover.
 * @param {Scene} scene @returns {string}
 */
export function sceneMarkup(scene) {
  const paths = scene.layers.map((layer) => `<path class="check-${layer.role}" d="${layer.d}"></path>`).join("");
  const texts = scene.labels.map((label) => `<text class="${label.axis ? "check-title" : "check-tick"}" x="${n(label.x)}"`
    + ` y="${n(label.y)}" text-anchor="${label.anchor}">${escapeHTML(label.text)}</text>`).join("");
  const w = n(scene.width);
  const h = n(scene.height);
  return `<svg class="check-plot" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">${paths}${texts}`
    + `<g class="check-readout" hidden></g></svg>`;
}

/**
 * The index of the mark nearest `x`, by a binary search of `marks` sorted by x; with `y`, the
 * nearest in the plane, swept out from there. -1 when there are no marks.
 * @param {Mark[]} marks sorted by x @param {number} x px @param {number} [y] px
 * @returns {number}
 */
export function nearestMark(marks, x, y) {
  if (marks.length === 0) return -1;
  let lo = 0;
  let hi = marks.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (marks[mid].x < x) lo = mid + 1;
    else hi = mid;
  }
  if (y === undefined) return lo > 0 && x - marks[lo - 1].x <= marks[lo].x - x ? lo - 1 : lo;
  let best = lo;
  let bestSq = (marks[lo].x - x) ** 2 + (marks[lo].y - y) ** 2;
  // Marks further than the best so far along x alone cannot be nearer in the plane.
  for (const step of [-1, 1]) {
    for (let i = lo + step; i >= 0 && i < marks.length && (marks[i].x - x) ** 2 < bestSq; i += step) {
      const sq = (marks[i].x - x) ** 2 + (marks[i].y - y) ** 2;
      if (sq < bestSq) [best, bestSq] = [i, sq];
    }
  }
  return best;
}
