// @ts-check

/** @typedef {import('../api/contracts.js').DisplaySpec} DisplaySpec */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

// A curve on a log-type link is drawn and edited as a relativity, exp(effect); the
// server converts a posted value back with the same maps. An overdispersion (1/theta)
// term is the relativity of 1/theta, exp(-effect), which falls as the recorded effect
// rises. An additive term (identity or any other non-log link) shows the effect itself,
// so its maps are the identity and it has no floor. With no spec the unit is
// exp(effect), which is what a log or logit SuperGLM term shows.
const FLOOR = 1e-12;

/**
 * Whether the curve shows the effect itself rather than a relativity.
 * @param {DisplaySpec|undefined|null} spec
 * @returns {boolean}
 */
export function isAdditive(spec) {
  return Boolean(spec && spec.kind === "additive");
}

/**
 * The display value at which a term has no effect: 0 on an additive curve, 1 on
 * every relativity.
 * @param {DisplaySpec|undefined|null} spec
 * @returns {number}
 */
export function noEffect(spec) {
  return isAdditive(spec) ? 0 : 1;
}

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
  if (isAdditive(spec)) return effect;
  return Math.exp(direction(spec) * effect);
}

/**
 * The effect a display value stands for. On a relativity a value at or below zero
 * clamps to the smallest positive one, so the result is always finite.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} value
 * @returns {number}
 */
export function fromDisplay(spec, value) {
  if (isAdditive(spec)) return value;
  return direction(spec) * Math.log(Math.max(value, FLOOR));
}

/**
 * The display value nearest `value` that an effect can produce: any value on an
 * additive curve, a positive one on a relativity.
 * @param {DisplaySpec|undefined|null} spec
 * @param {number} value
 * @returns {number}
 */
export function reachableDisplay(spec, value) {
  if (isAdditive(spec)) return value;
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
  if (isAdditive(spec)) return base + weight * deltaEffect;
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
