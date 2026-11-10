// @ts-check

/** @typedef {import('../api/contracts.js').DisplaySpec} DisplaySpec */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

// Every curve is drawn and edited as a relativity, exp(effect); the server converts a
// posted value back with the same maps. An overdispersion (1/theta) term is the
// relativity of 1/theta, exp(-effect), which falls as the recorded effect rises.
// With no spec the unit is exp(effect), which is what a SuperGLM term shows.
const FLOOR = 1e-12;

/**
 * The way up the shown curve runs against the effect: -1 for a 1/theta term.
 * @param {DisplaySpec|undefined|null} spec
 * @returns {number}
 */
function direction(spec) {
  return spec && spec.kind === "inverse_theta" ? -1 : 1;
}

/**
 * The display value of an effect on the linear predictor.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} effect
 * @returns {number}
 */
export function toDisplay(spec, effect) {
  return Math.exp(direction(spec) * effect);
}

/**
 * The effect a display value stands for. A value at or below zero clamps to the
 * smallest positive one, so the result is always finite.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} value
 * @returns {number}
 */
export function fromDisplay(spec, value) {
  return direction(spec) * Math.log(Math.max(value, FLOOR));
}

/**
 * The display value nearest `value` that an effect can produce.
 * @param {DisplaySpec|undefined|null} _spec
 * @param {number} value
 * @returns {number}
 */
export function reachableDisplay(_spec, value) {
  return Math.max(value, FLOOR);
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
  return base * Math.exp(direction(spec) * weight * deltaEffect);
}

/**
 * Whether the term's curve is drawn in relativities. The Contrib and Build
 * overlays exponentiate basis terms, so only such a term has them.
 * @param {TermPayload|null|undefined} term
 * @returns {boolean}
 */
export function showsRelativity(term) {
  return !term || !term.display || term.display.kind === "relativity";
}

/**
 * Whether the Contrib and Build overlays can be drawn.
 * @param {TermPayload|null|undefined} term
 * @returns {boolean}
 */
export function canShowContributions(term) {
  const controls = term && term.controls;
  return Boolean(
    showsRelativity(term) &&
    controls &&
    Array.isArray(controls.basis) &&
    controls.basis.length > 0 &&
    Array.isArray(controls.build_basis) &&
    controls.build_basis.length > 0
  );
}
