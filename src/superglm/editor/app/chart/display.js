// @ts-check

import { fmt } from "../format.js";

/** @typedef {import('../api/contracts.js').DisplaySpec} DisplaySpec */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

// Curves are drawn and edited in display units; the server converts a posted
// display value back with the same maps. With no spec the unit is the
// relativity exp(effect), which is what a SuperGLM term shows.
const FLOOR = 1e-12;

/**
 * The display value of an effect on the linear predictor.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} effect
 * @returns {number}
 */
export function toDisplay(spec, effect) {
  if (!spec) return Math.exp(effect);
  const c = spec.reference_eta;
  switch (spec.kind) {
    case "lognormal_cv": {
      const s = spec.floor + Math.exp(c + effect);
      return Math.sqrt(Math.expm1(s * s));
    }
    case "gamma_cv":
      return Math.exp(c + effect);
    case "inverse_theta":
      return Math.exp(-(c + effect));
    default:
      return Math.exp(effect);
  }
}

/**
 * The effect a display value stands for. Values outside the transform's range
 * clamp to the nearest effect it can reach, so the result is always finite.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} value
 * @returns {number}
 */
export function fromDisplay(spec, value) {
  if (!spec) return Math.log(Math.max(value, FLOOR));
  const c = spec.reference_eta;
  switch (spec.kind) {
    case "lognormal_cv": {
      const s = Math.sqrt(Math.log1p(value * value));
      return Math.log(Math.max(s - spec.floor, FLOOR)) - c;
    }
    case "gamma_cv":
      return Math.log(Math.max(value, FLOOR)) - c;
    case "inverse_theta":
      return -Math.log(Math.max(value, FLOOR)) - c;
    default:
      return Math.log(Math.max(value, FLOOR));
  }
}

/**
 * The display value nearest `value` that an effect can produce.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} value
 * @returns {number}
 */
export function reachableDisplay(spec, value) {
  if (!spec || spec.kind === "relativity") return Math.max(value, FLOOR);
  return toDisplay(spec, fromDisplay(spec, value));
}

/**
 * `base` moved by `weight * deltaEffect` on the effect scale.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} base display value
 * @param {number} weight
 * @param {number} deltaEffect
 * @returns {number}
 */
export function shiftDisplay(spec, base, weight, deltaEffect) {
  if (!spec || spec.kind === "relativity") return base * Math.exp(weight * deltaEffect);
  return toDisplay(spec, fromDisplay(spec, base) + weight * deltaEffect);
}

/**
 * A value as the term's axis and tooltip show it.
 * @param {Pick<TermPayload, 'y_format'>|Record<string, never>} term
 * @param {number} value
 * @returns {string}
 */
export function formatDisplay(term, value) {
  const text = fmt(value);
  return text && term.y_format === "times" ? `${text}×` : text;
}
