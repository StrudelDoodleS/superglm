// @ts-check
/**
 * The four checks' plots as scenes: pure functions from one check payload to the paths, labels
 * and hover targets of an SVG. Each role is one path, so the stylesheet alone colours a scene
 * (`check-{role}`), and a card's thumbnail and the large view are drawn by the same code.
 */

import { escapeHTML, fmt, fmtCount, fmtPercent } from "../format.js";
import { niceTicks } from "./fan.js";

/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').CheckPayload} CheckPayload */
/** @typedef {import('../api/contracts.js').CheckBody} CheckBody */
/** @typedef {'band'|'reference'|'edited'|'original'|'outside'|'interval'|'weight'|'grid'} Role */
/** @typedef {{role: Role, d: string}} Layer one path per role; points are arcs inside the path */
/** @typedef {{x:number, y:number, text:string}} Mark a hover target in px and its readout */
/**
 * A tick label, or with `axis` an axis or panel title.
 * @typedef {{x:number, y:number, text:string, anchor:'start'|'middle'|'end', axis?:boolean}} Label
 */
/** @typedef {{width:number, height:number, layers:Layer[], labels:Label[], marks:Mark[]}} Scene */
/** @typedef {{width:number, height:number, thumb?:boolean}} Box */
/** @typedef {{left:number, top:number, right:number, bottom:number}} Rect */
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
const ROLES = /** @type {const} */ (["grid", "weight", "band", "reference", "original", "interval", "edited", "outside"]);

const THUMB_PAD = 2;
// Room for the tick labels and the axis titles around a large view's plot.
const MARGIN = { left: 52, right: 16, top: 32, bottom: 46 };
// A/E's panels: the gap between them, and the room for each one's title and tick labels.
const PANEL_GAP = { col: 24, row: 12 };
const PANEL_MARGIN = { left: 44, top: 20, bottom: 24 };
const THUMB_QQ_POINTS = 60;
const THUMB_QQ_OUTSIDE = 20;
// The share of a value axis left beyond the outermost drawn value.
const PAD_SHARE = 0.04;
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
 * The range of the finite `values`, widened by `PAD_SHARE` of itself on each side.
 * @param {Iterable<number>} values @returns {[number, number]}
 */
function span(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (!(hi >= lo)) return [0, 1];
  if (hi === lo) return [lo - 0.5, hi + 0.5];
  const room = (hi - lo) * PAD_SHARE;
  return [lo - room, hi + room];
}

/** @param {Box} box @returns {Rect} */
function plotRect(box) {
  const m = box.thumb ? { left: THUMB_PAD, right: THUMB_PAD, top: THUMB_PAD, bottom: THUMB_PAD } : MARGIN;
  return { left: m.left, top: m.top, right: Math.max(m.left + 1, box.width - m.right), bottom: Math.max(m.top + 1, box.height - m.bottom) };
}

/** @returns {Parts} */
function emptyParts() {
  return { grid: [], weight: [], band: [], reference: [], original: [], interval: [], edited: [], outside: [] };
}

/** @param {Box} box @param {Parts} parts @param {Label[]} labels @param {Mark[]} marks @returns {Scene} */
function finish(box, parts, labels, marks) {
  const layers = ROLES.filter((role) => parts[role].length > 0).map((role) => ({ role, d: parts[role].join(" ") }));
  return { width: box.width, height: box.height, layers, labels: box.thumb ? [] : labels, marks };
}

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

/** @param {QQStats} s @param {number} i */
const qqOutside = (s, i) => s.observed[i] < s.band_lower[i] || s.observed[i] > s.band_upper[i];

/**
 * The positions of a Q-Q body's points a scene draws: all of them, or for a thumbnail up to
 * `THUMB_QQ_OUTSIDE` outside the band and evenly spaced others up to `THUMB_QQ_POINTS` in all.
 * @param {QQStats} s @param {Box} box @returns {number[]}
 */
function qqShown(s, box) {
  const all = s.index.map((_, i) => i);
  if (!box.thumb) return all;
  const outside = spaced(all.filter((i) => qqOutside(s, i)), THUMB_QQ_OUTSIDE);
  const inside = spaced(all.filter((i) => !qqOutside(s, i)), THUMB_QQ_POINTS - outside.length);
  return [...outside, ...inside].sort((a, b) => a - b);
}

/** @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box @returns {Scene} */
function qqScene(editedBody, originalBody, box) {
  const edited = /** @type {QQStats} */ (/** @type {unknown} */ (editedBody));
  const original = originalBody && /** @type {QQStats} */ (/** @type {unknown} */ (originalBody));
  const shown = qqShown(edited, box);
  const shownOriginal = original ? qqShown(original, box) : [];
  /** @type {number[]} */
  const values = [];
  for (const i of shown) values.push(edited.theoretical[i], edited.observed[i], edited.band_lower[i], edited.band_upper[i]);
  if (original) for (const i of shownOriginal) values.push(original.theoretical[i], original.observed[i]);
  // One range for both axes, so y = x is the plot's diagonal.
  const [lo, hi] = span(values);
  const rect = plotRect(box);
  const sx = linear(lo, hi, rect.left, rect.right);
  const sy = linear(lo, hi, rect.bottom, rect.top);
  const r = radius(box);
  const parts = emptyParts();
  /** @type {Mark[]} */
  const marks = [];
  /** @type {(bound: number[]) => (i: number) => [number, number]} */
  const along = (bound) => (i) => [sx(edited.theoretical[i]), sy(bound[i])];
  parts.band.push(`${polyline([...shown.map(along(edited.band_upper)), ...[...shown].reverse().map(along(edited.band_lower))])} Z`);
  parts.reference.push(polyline([[sx(lo), sy(lo)], [sx(hi), sy(hi)]]));
  if (original) for (const i of shownOriginal) parts.original.push(dot(sx(original.theoretical[i]), sy(original.observed[i]), r));
  for (const i of shown) {
    const x = sx(edited.theoretical[i]);
    const y = sy(edited.observed[i]);
    parts[qqOutside(edited, i) ? "outside" : "edited"].push(dot(x, y, r));
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
    const ticks = niceTicks(lo, hi, 6);
    frame(parts, labels, rect, ticks.map((v) => [sx(v), fmt(v)]), ticks.map((v) => [sy(v), fmt(v)]),
      { x: "Expected residual", y: "Observed residual", xGrid: true });
  }
  return finish(box, parts, labels, marks);
}

/** @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box @returns {Scene} */
function pitScene(editedBody, originalBody, box) {
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
  const sx = linear(edges[0], edges[edges.length - 1], rect.left, rect.right);
  const sy = linear(0, top, rect.bottom, rect.top);
  const parts = emptyParts();
  /** @type {Mark[]} */
  const marks = [];
  parts.band.push(`M ${n(rect.left)} ${n(sy(upper))} H ${n(rect.right)} V ${n(sy(lower))} H ${n(rect.left)} Z`);
  parts.reference.push(hline(rect.left, rect.right, sy(1)));
  for (const [i, count] of edited.counts.entries()) {
    const left = sx(edges[i]);
    const right = sx(edges[i + 1]);
    const gap = ((right - left) * (1 - BAR_SHARE)) / 2;
    let start = left + gap;
    // The fitted model's bar takes the left half of its bin's.
    if (original) {
      const middle = (left + right) / 2;
      parts.original.push(bar(start, middle, sy(0), sy(share(original, i))));
      start = middle;
    }
    const y = sy(share(edited, i));
    const outside = count < edited.band_lower || count > edited.band_upper;
    parts[outside ? "outside" : "edited"].push(bar(start, right - gap, sy(0), y));
    marks.push({
      x: (left + right) / 2,
      y,
      text: `${fmt(edges[i])} to ${fmt(edges[i + 1])} · ${fmtCount(count)} values`
        + ` · band ${fmtCount(edited.band_lower)} to ${fmtCount(edited.band_upper)}`
    });
  }
  /** @type {Label[]} */
  const labels = [];
  if (!box.thumb) {
    frame(parts, labels, rect,
      niceTicks(edges[0], edges[edges.length - 1], 4).map((v) => [sx(v), fmt(v)]),
      niceTicks(0, top, 4).map((v) => [sy(v), fmt(v)]),
      { x: "Cumulative probability", y: "Count over expected", xGrid: false });
  }
  return finish(box, parts, labels, marks);
}

/** @param {AETable} table @param {number} i */
const aeGroup = (table, i) => (table.levels ? String(table.levels[i]) : `band ${i + 1}`);

/** @param {AETable} table @param {number} i */
const aeDrawn = (table, i) => [table.ratio[i], table.lower[i], table.upper[i]].every(Number.isFinite);

/**
 * One A/E panel: the reference at 1, the weight bars along its foot, and each group's ratio
 * with its interval; the fitted model's beside it when compared.
 * @param {Parts} parts @param {Label[]} labels @param {Mark[]} marks @param {Rect} rect
 * @param {[number, number]} domain @param {AETable} table @param {AETable|null} original @param {Box} box
 */
function aePanel(parts, labels, marks, rect, domain, table, original, box) {
  const sy = linear(domain[0], domain[1], rect.bottom, rect.top);
  const k = table.ratio.length;
  const slot = (rect.right - rect.left) / k;
  const shift = original ? Math.min(COMPARE_SHIFT, slot * 0.15) : 0;
  const heaviest = Math.max(0, ...table.weight.filter(Number.isFinite));
  const r = radius(box);
  parts.reference.push(hline(rect.left, rect.right, sy(1)));
  for (let i = 0; i < k; i++) {
    const centre = rect.left + (i + 0.5) * slot;
    if (heaviest > 0 && table.weight[i] > 0) {
      const h = (rect.bottom - rect.top) * WEIGHT_HEIGHT * (table.weight[i] / heaviest);
      parts.weight.push(bar(centre - (slot * WEIGHT_SHARE) / 2, centre + (slot * WEIGHT_SHARE) / 2, rect.bottom, rect.bottom - h));
    }
    if (original && aeDrawn(original, i)) {
      parts.original.push(vline(centre - shift, sy(original.lower[i]), sy(original.upper[i])),
        dot(centre - shift, sy(original.ratio[i]), r));
    }
    if (!aeDrawn(table, i)) continue;
    const x = centre + shift;
    const y = sy(table.ratio[i]);
    const outside = table.lower[i] > 1 || table.upper[i] < 1;
    parts.interval.push(vline(x, sy(table.lower[i]), sy(table.upper[i])));
    parts[outside ? "outside" : "edited"].push(dot(x, y, r));
    marks.push({
      x,
      y,
      text: `${table.covariate} ${aeGroup(table, i)} · ${fmt(table.ratio[i])}`
        + ` · interval ${fmt(table.lower[i])} to ${fmt(table.upper[i])} · ${fmtCount(table.n[i])} rows`
    });
  }
  if (box.thumb) return;
  for (const v of niceTicks(domain[0], domain[1], 4)) {
    parts.grid.push(hline(rect.left, rect.right, sy(v)));
    labels.push({ x: rect.left - 6, y: sy(v) + 4, text: fmt(v), anchor: "end" });
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

/** @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box @returns {Scene} */
function aeScene(editedBody, originalBody, box) {
  const edited = /** @type {AEStats} */ (/** @type {unknown} */ (editedBody));
  const original = originalBody && /** @type {AEStats} */ (/** @type {unknown} */ (originalBody));
  // A thumbnail draws the predicted-Mean bands alone.
  const tables = box.thumb ? [edited.tables.find((t) => t.by === "band") ?? edited.tables[0]] : edited.tables;
  const pairs = tables.map((table) => ({ table, original: original?.tables[edited.tables.indexOf(table)] ?? null }));
  /** @type {number[]} */
  const values = [1];
  for (const { table, original: fitted } of pairs) {
    for (const t of fitted ? [table, fitted] : [table]) values.push(...t.ratio, ...t.lower, ...t.upper);
  }
  // One scale for every panel, so their ratios read alike.
  const domain = span(values);
  const cols = box.thumb ? 1 : Math.min(2, tables.length);
  const rows = Math.ceil(tables.length / cols);
  const outer = box.thumb
    ? plotRect(box)
    : { left: 8, top: MARGIN.top, right: box.width - MARGIN.right, bottom: box.height - 8 };
  const cellW = (outer.right - outer.left - PANEL_GAP.col * (cols - 1)) / cols;
  const cellH = (outer.bottom - outer.top - PANEL_GAP.row * (rows - 1)) / rows;
  const parts = emptyParts();
  /** @type {Label[]} */
  const labels = [];
  /** @type {Mark[]} */
  const marks = [];
  for (const [k, { table, original: fitted }] of pairs.entries()) {
    const left = outer.left + (k % cols) * (cellW + PANEL_GAP.col);
    const top = outer.top + Math.floor(k / cols) * (cellH + PANEL_GAP.row);
    const rect = box.thumb
      ? outer
      : { left: left + PANEL_MARGIN.left, top: top + PANEL_MARGIN.top, right: left + cellW, bottom: Math.max(top + PANEL_MARGIN.top + 1, top + cellH - PANEL_MARGIN.bottom) };
    aePanel(parts, labels, marks, rect, domain, table, fitted, box);
  }
  labels.push({ x: 8, y: MARGIN.top - 14, text: "Actual over expected", anchor: "start", axis: true });
  marks.sort((a, b) => a.x - b.x);
  return finish(box, parts, labels, marks);
}

/** @param {CheckBody} editedBody @param {CheckBody|null} originalBody @param {Box} box @returns {Scene} */
function calibrationScene(editedBody, originalBody, box) {
  const edited = /** @type {CalibrationStats} */ (/** @type {unknown} */ (editedBody));
  const original = originalBody && /** @type {CalibrationStats} */ (/** @type {unknown} */ (originalBody));
  const q = edited.quantiles;
  const shares = [...edited.band_lower, ...edited.band_upper, ...q.realised_exceedance, ...q.expected];
  if (original) shares.push(...original.quantiles.realised_exceedance);
  const [lo, hi] = span(shares.filter((v) => v > 0).map(Math.log10));
  const rect = plotRect(box);
  // A share of zero has no log: it gets its own row at the foot, below the least positive share.
  const zero = shares.includes(0);
  const sy = linear(lo, hi, rect.bottom - (zero ? ZERO_ROW[box.thumb ? "thumb" : "view"] : 0), rect.top);
  const y = (/** @type {number} */ v) => (v === 0 ? rect.bottom : sy(Math.log10(v)));
  const slot = (rect.right - rect.left) / q.p.length;
  const xAt = (/** @type {number} */ i) => rect.left + (i + 0.5) * slot;
  const r = radius(box);
  const parts = emptyParts();
  /** @type {Mark[]} */
  const marks = [];
  const indices = [...q.p.keys()];
  parts.band.push(`${polyline([
    ...indices.map((i) => /** @type {[number, number]} */ ([xAt(i), y(edited.band_upper[i])])),
    ...[...indices].reverse().map((i) => /** @type {[number, number]} */ ([xAt(i), y(edited.band_lower[i])]))
  ])} Z`);
  parts.reference.push(polyline(indices.map((i) => [xAt(i), y(q.expected[i])])));
  // Both models' points sit at their quantile; the style alone tells them apart.
  if (original) for (const i of indices) parts.original.push(dot(xAt(i), y(original.quantiles.realised_exceedance[i]), r));
  for (const i of indices) {
    const realised = q.realised_exceedance[i];
    const outside = realised < edited.band_lower[i] || realised > edited.band_upper[i];
    const x = xAt(i);
    parts[outside ? "outside" : "edited"].push(dot(x, y(realised), r));
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
    /** @type {Array<[number, string]>} */
    const decades = [];
    for (let j = Math.ceil(lo); j <= Math.floor(hi); j++) decades.push([sy(j), fmtPercent(10 ** j)]);
    if (zero) decades.push([rect.bottom, "0"]);
    frame(parts, labels, rect, indices.map((i) => [xAt(i), fmtPercent(q.p[i])]), decades,
      { x: "Predicted quantile", y: "Share of rows above", xGrid: false });
  }
  return finish(box, parts, labels, marks);
}

/** @type {Record<CheckReport, (edited: CheckBody, original: CheckBody|null, box: Box) => Scene>} */
const SCENES = { check_qq: qqScene, check_pit: pitScene, check_ae: aeScene, check_calibration: calibrationScene };

/**
 * A check's plot as a scene in `box`: the edited model, and with `compare` the fitted model when
 * the payload carries it. An unavailable check is an empty scene. A thumbnail has no labels.
 * @param {CheckReport} report @param {CheckPayload} payload @param {Box} box
 * @param {{compare:boolean}} options
 * @returns {Scene}
 */
export function checkScene(report, payload, box, options) {
  const { edited, original } = payload.models;
  if (!payload.available || !edited.available) return finish(box, emptyParts(), [], []);
  return SCENES[report](edited, options.compare && original?.available ? original : null, box);
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
