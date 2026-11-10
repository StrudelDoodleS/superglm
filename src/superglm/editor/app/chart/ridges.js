// @ts-check
/**
 * Geometry of the distribution panel's ridges: one density per row on a shared log10
 * amount axis, each scaled to its own peak, and the fixed log-scale dot column beside them. Every
 * function here is pure; distribution_panel.js turns the numbers into SVG.
 */

/** @typedef {import('../distribution/families.js').Law} Law */
/** @typedef {{min_log10: number, max_log10: number}} AmountAxis */
/** @typedef {[number, number]} RidgePoint  [log10 amount, y] */

/**
 * @typedef {object} RidgeLayout
 * @property {number} width
 * @property {number} left  the plot's left edge, after the row labels
 * @property {number} right  the plot's right edge
 * @property {number} top
 * @property {number} row  the vertical step between ridges
 * @property {number} height  a ridge's peak height; taller than `row`, so ridges overlap
 * @property {(i: number) => number} baseOf  the baseline of ridge `i`
 * @property {number} axisY  the amount axis' label line
 * @property {number} total  the SVG's height
 * @property {number} dotLeft  the dot column's left end
 * @property {number} dotRight  its right end
 * @property {number} readoutRight  the right edge of the readouts
 */

const TOP = 38;
const ROW = 36;
const RIDGE_HEIGHT = 58;
const BASE_OFFSET = 16;
const AXIS_GAP = 18;
const BELOW_AXIS = 32;
// The right-hand columns: the dots, then the readouts right-aligned to the edge.
const DOT_WIDTH = 64;
const DOTS_FROM_RIGHT = 132;
const READOUT_INSET = 8;
// Room for the amount axis's last label and the dot column's first, "1 in 10k" centred on its left end.
const PLOT_TO_DOTS = 48;
export const RIDGE_SAMPLES = 180;

/**
 * @param {number} width  the SVG's width in pixels
 * @param {number} rows
 * @param {number} [labelWidth]  room for the row labels left of the plot
 * @returns {RidgeLayout}
 */
export function ridgeLayout(width, rows, labelWidth = 46) {
  const dotLeft = width - DOTS_FROM_RIGHT;
  /** @param {number} i */
  const baseOf = (i) => TOP + (i + 1) * ROW + BASE_OFFSET;
  const axisY = baseOf(Math.max(rows, 1) - 1) + AXIS_GAP;
  return {
    width,
    left: labelWidth,
    right: dotLeft - PLOT_TO_DOTS,
    top: TOP,
    row: ROW,
    height: RIDGE_HEIGHT,
    baseOf,
    axisY,
    total: axisY + BELOW_AXIS,
    dotLeft,
    dotRight: dotLeft + DOT_WIDTH,
    readoutRight: width - READOUT_INSET,
  };
}

/**
 * The x of a log10 amount on the plot, and its inverse.
 * @param {AmountAxis} axis @param {RidgeLayout} layout
 */
export function amountScale(axis, layout) {
  const span = axis.max_log10 - axis.min_log10;
  const px = layout.right - layout.left;
  return {
    /** @param {number} l */
    x: (l) => layout.left + ((l - axis.min_log10) / span) * px,
    /** @param {number} x */
    l: (x) => axis.min_log10 + ((x - layout.left) / px) * span,
  };
}

/**
 * One ridge, sampled across the axis. Heights are the density of log10 Y relative to its largest
 * sample, so every ridge peaks at the layout's height.
 * @param {Law} law @param {AmountAxis} axis
 * @param {RidgeLayout} layout @param {number} index the ridge's row
 * @param {number} [samples]
 * @returns {{points: RidgePoint[], yAt: (l: number) => number}}
 */
export function ridgeShape(law, axis, layout, index, samples = RIDGE_SAMPLES) {
  const { min_log10: lo, max_log10: hi } = axis;
  const ls = Array.from({ length: samples + 1 }, (_, j) => lo + ((hi - lo) * j) / samples);
  const densities = ls.map((l) => law.log10Density(l));
  const peak = Math.max(...densities);
  const base = layout.baseOf(index);
  /** @param {number} l */
  const yAt = (l) => base - layout.height * 10 ** (law.log10Density(l) - peak);
  /** @type {RidgePoint[]} */
  const points = ls.map((l, j) => [l, base - layout.height * 10 ** (densities[j] - peak)]);
  return { points, yAt };
}

/**
 * The x of a positive value on the dot column's fixed log10 scale, clamped to the column, so a
 * value never saturates a bar and a tiny one still shows.
 * @param {number} value @param {[number, number]} domain log10 ends of the column
 * @param {number} x0 @param {number} x1
 */
export function dotPosition(value, domain, x0, x1) {
  if (!(value > 0)) return x0;
  const f = (Math.log10(value) - domain[0]) / (domain[1] - domain[0]);
  return x0 + Math.min(1, Math.max(0, f)) * (x1 - x0);
}

/**
 * The whole decades on the axis to label, every `stride`-th counted down from the top one, with
 * the stride the smallest that keeps neighbouring labels `minGap` pixels apart.
 * @param {AmountAxis} axis @param {number} plotWidth @param {number} minGap
 * @returns {number[]}
 */
export function decadeTicks(axis, plotWidth, minGap) {
  const first = Math.ceil(axis.min_log10);
  const last = Math.floor(axis.max_log10);
  const perDecade = plotWidth / (axis.max_log10 - axis.min_log10);
  const stride = Math.max(1, Math.ceil(minGap / perDecade));
  const ticks = [];
  for (let d = first; d <= last; d += 1) if ((last - d) % stride === 0) ticks.push(d);
  return ticks;
}

const SUPERSCRIPT = /** @type {Record<string, string>} */ ({
  "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
  5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹",
});

/**
 * A decade 10^e as an axis label: 1k, 10m and 1bn above a thousand, decimals down to 0.001, and a
 * power of ten below that.
 * @param {number} e @param {string | null} unit
 */
export function decadeLabel(e, unit) {
  const u = unit ?? "";
  if (e >= 9) return `${u}${10 ** (e - 9)}bn`;
  if (e >= 6) return `${u}${10 ** (e - 6)}m`;
  if (e >= 3) return `${u}${10 ** (e - 3)}k`;
  if (e >= -3) return `${u}${(10 ** e).toFixed(Math.max(0, -e))}`;
  return `${u}10${[...String(e)].map((c) => SUPERSCRIPT[c]).join("")}`;
}
