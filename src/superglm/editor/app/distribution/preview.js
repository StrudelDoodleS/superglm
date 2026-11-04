// @ts-check
/**
 * The distribution panel's predictors while a curve is dragged, computed in the browser.
 *
 * A term adds its effect to one parameter's linear predictor, so moving the term's curve from
 * `committed` to `dragged` moves that parameter's predictor at a feature value by the change in
 * the term's effect there, and leaves every other parameter as it is. The effect is read back from
 * each curve's displayed values with the term's own display transform. Between the points of a
 * numeric term's grid the change is linear in x, and beyond its ends it is the end value; a level
 * term's change is read at the same level, and a level the term does not list does not move.
 */

import { fromDisplay } from "../chart/display.js";

/** @typedef {import('../api/contracts.js').DistributionPanelResponse} DistributionPanel */
/** @typedef {import('../api/contracts.js').DistributionEta} DistributionEta */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {Pick<TermPayload, 'x' | 'y' | 'levels' | 'display'>} Curve */
/** @typedef {{x: number | null, level: string | null}} PanelPoint */

/**
 * The change in the term's effect at each of its points, from `committed` to `dragged`.
 * @param {Curve} committed @param {Curve} dragged
 * @returns {number[]}
 */
function effectChange(committed, dragged) {
  return dragged.y.map(
    (value, j) => fromDisplay(dragged.display, value) - fromDisplay(committed.display, committed.y[j])
  );
}

/**
 * `ys` at `x`, linear between the points of the ascending grid `xs` and the end value beyond it.
 * @param {number[]} xs @param {number[]} ys @param {number} x
 * @returns {number}
 */
function interpolate(xs, ys, x) {
  const last = xs.length - 1;
  if (!(x > xs[0])) return ys[0];
  if (x >= xs[last]) return ys[last];
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

/**
 * The panel's edited predictors with the dragged term's change added to its parameter.
 * @param {DistributionPanel} panel
 * @param {string} key the dragged term
 * @param {Curve} committed the term as the server last confirmed it
 * @param {Curve} dragged the term as the drag shows it
 * @returns {DistributionEta}
 */
export function previewEta(panel, key, committed, dragged) {
  const edited = panel.eta.edited;
  const index = panel.parameters.findIndex((parameter) => panel.term_keys[parameter.name] === key);
  if (index < 0) return edited;
  const change = effectChange(committed, dragged);
  const levels = dragged.levels;
  /** @param {PanelPoint} point */
  const changeAt = (point) => {
    if (levels) {
      const j = point.level === null ? -1 : levels.indexOf(point.level);
      return j < 0 ? 0 : change[j];
    }
    return point.x === null ? 0 : interpolate(dragged.x, change, point.x);
  };
  /** @param {number[][]} etas @param {PanelPoint[]} points */
  const shifted = (etas, points) =>
    etas.map((eta, i) => {
      const next = eta.slice();
      next[index] += changeAt(points[i]);
      return next;
    });
  return {
    rows: shifted(edited.rows, panel.rows),
    columns: shifted(edited.columns, panel.columns),
  };
}
