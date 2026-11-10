// @ts-check
/**
 * The Explain tab for a model of counts (a negative binomial): each risk's distribution over the
 * number of claims, for one unit of exposure. `risk_panel.js` sets the laws side by side in its
 * table and calls `drawCounts` where an amount model has its two plots.
 */

import { nbPmf, nbSf } from "../distribution/math.js";
import { formatChance } from "../distribution/line.js";
import { fmtSignificant } from "../format.js";
import { el } from "../chart/svg.js";

/** @typedef {import('../api/contracts.js').RiskResponse} RiskResponse */
/** @typedef {{mean: number, theta: number}} CountLaw  variance = mean + mean^2 / theta */
/**
 * @typedef {object} CountCurve
 * @property {import('../api/contracts.js').RiskId} id
 * @property {string} name
 * @property {CountLaw} inForce
 * @property {CountLaw | null} original  the fitted model's law, where the edits moved this risk
 */

/** The chance axis's labels at each power of ten, as a percentage and as 1 in N. */
export const CHANCE_TICKS = /** @type {const} */ ([
  [0, "100%", "1 in 1"], [-1, "10%", "1 in 10"], [-2, "1%", "1 in 100"],
  [-3, "0.1%", "1 in 1,000"], [-4, "0.01%", "1 in 10k"],
]);
// The plot shows at least this many counts, and no more than the larger.
const MIN_REACH = 3;
const MAX_REACH = 12;

/**
 * @param {RiskResponse} reply @param {readonly number[]} eta
 * @returns {CountLaw}
 */
export function countLaw(reply, eta) {
  const at = (/** @type {(p: RiskResponse["parameters"][number]) => boolean} */ is) =>
    Math.exp(eta[reply.parameters.findIndex(is)]);
  return { mean: at((p) => p.role === "average"), theta: at((p) => p.name === "theta") };
}

/**
 * The table's columns for a law of counts.
 * @param {RiskResponse} reply @param {"percent" | "one_in"} style how a chance is written
 * @returns {Array<{label: string, title: string, read: (law: CountLaw) => number, show: (value: number) => string}>}
 */
export function countColumns(reply, style) {
  /** @param {number} p */
  const chance = (p) => formatChance(p, style);
  /** @param {number} v */
  const figure = (v) => fmtSignificant(v, 3);
  const dispersion = reply.parameters.find((p) => p.name === "theta")?.label ?? "Overdispersion";
  return [
    { label: "Mean", title: "The expected number of claims for one unit of exposure.", read: (law) => law.mean, show: figure },
    { label: "No claim", title: "The chance of no claim.", read: (law) => nbPmf(0, law.mean, law.theta), show: chance },
    ...[1, 2, 3].map((k) => ({
      label: `${k} or more`,
      title: `The chance of ${k} or more claims.`,
      read: (/** @type {CountLaw} */ law) => nbSf(k, law.mean, law.theta),
      show: chance,
    })),
    {
      label: dispersion,
      title: "Overdispersion, 1/θ: the variance is the mean plus the mean squared times it.",
      read: (law) => 1 / law.theta,
      show: figure,
    },
  ];
}

/**
 * The largest count the plot shows: the last one some law gives a chance above the floor.
 * @param {CountLaw[]} laws @param {number} floor log10 of the smallest chance drawn
 */
export function countReach(laws, floor) {
  let reach = MIN_REACH;
  while (reach < MAX_REACH && laws.some((law) => nbPmf(reach + 1, law.mean, law.theta) >= 10 ** floor)) reach += 1;
  return reach;
}

/**
 * Draws each risk's distribution: a bar per number of claims, the risks side by side, on a log
 * scale down to `floor` so the rare counts show. The selected risk's bars are the strongest and
 * carry their chance; where the session's edits moved it, a grey rule marks the original model's.
 * @param {SVGSVGElement} svg @param {CountCurve[]} curves
 * @param {{style: "percent" | "one_in", width: number, height: number, floor: number, selected: string}} options
 */
export function drawCounts(svg, curves, { style, width, height, floor, selected }) {
  const left = 58;
  const right = Math.max(left + 120, width - 18);
  const top = 30;
  const bottom = Math.max(top + 120, height - 44);
  const reach = countReach(curves.flatMap((curve) => (curve.original ? [curve.inForce, curve.original] : [curve.inForce])), floor);
  const band = (right - left) / (reach + 1);
  const bar = Math.min(46, (band * 0.84) / curves.length);
  /** @param {number} k @param {number} i the risk's place among those drawn */
  const x = (k, i) => left + band * (k + 0.5) + (i - curves.length / 2) * bar;
  /** @param {number} p */
  const y = (p) => top + (Math.max(Math.log10(p), floor) / floor) * (bottom - top);
  /** @param {string} tag @param {Record<string, string | number>} attrs @param {string} [content] @param {Element} [parent] */
  const add = (tag, attrs, content, parent = svg) => {
    const node = el(tag, attrs);
    if (content !== undefined) node.textContent = content;
    parent.appendChild(node);
    return node;
  };
  svg.replaceChildren();
  add("text", { x: 2, y: 16, class: "dist-title" }, "How likely each number of claims is");
  for (const [exponent, percent, oneIn] of CHANCE_TICKS) {
    if (exponent < floor) continue;
    const at = y(10 ** exponent);
    add("line", { x1: left, y1: at, x2: right, y2: at, class: exponent === floor ? "risk-axis" : "risk-grid" });
    add("text", { x: left - 6, y: at + 3.5, class: "dist-muted", "text-anchor": "end" }, style === "one_in" ? oneIn : percent);
  }
  for (let k = 0; k <= reach; k += 1) {
    add("text", { x: left + band * (k + 0.5), y: bottom + 16, class: "dist-muted", "text-anchor": "middle" }, String(k));
  }
  add("text", { x: right, y: bottom + 32, class: "dist-muted", "text-anchor": "end" }, "claims, for one unit of exposure");

  curves.forEach((curve, i) => {
    const chosen = curve.id === selected;
    for (let k = 0; k <= reach; k += 1) {
      const p = nbPmf(k, curve.inForce.mean, curve.inForce.theta);
      if (Math.log10(p) < floor) continue;
      const [x0, x1, at] = [x(k, i) + 1, x(k, i) + bar - 1, y(p)];
      // A bar stands on the axis and is rounded at its data end.
      const r = Math.min(4, (x1 - x0) / 2, bottom - at);
      const shape = add("path", {
        d: `M${x0.toFixed(1)},${bottom} V${(at + r).toFixed(1)} Q${x0.toFixed(1)},${at.toFixed(1)} ${(x0 + r).toFixed(1)},${at.toFixed(1)} ` +
          `H${(x1 - r).toFixed(1)} Q${x1.toFixed(1)},${at.toFixed(1)} ${x1.toFixed(1)},${(at + r).toFixed(1)} V${bottom} Z`,
        class: `risk-count-bar${chosen ? " is-selected" : ""}`, "data-risk": curve.id,
      });
      add("title", {}, `${curve.name}: ${formatChance(p, style)} chance of ${k} claim${k === 1 ? "" : "s"}`, shape);
      if (!chosen) continue;
      add("text", { x: (x0 + x1) / 2, y: at - 5, class: "dist-readout", "text-anchor": "middle" }, formatChance(p, style));
      if (curve.original) {
        const was = y(nbPmf(k, curve.original.mean, curve.original.theta));
        add("line", { x1: x0 - 3, y1: was, x2: x1 + 3, y2: was, class: "risk-line risk-line-original" });
      }
    }
  });
  svg.setAttribute("viewBox", `0 0 ${width} ${bottom + 44}`);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(bottom + 44));
  svg.classList.remove("is-amount");
  svg.setAttribute(
    "aria-label",
    `How likely each number of claims is for ${curves.map((curve) => curve.name).join(", ")}: the table beside it holds the figures`
  );
}
