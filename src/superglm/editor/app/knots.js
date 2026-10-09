// @ts-check

/**
 * A spline term's knots in the browser: the axis they sit on, where a drag, a
 * click or an arrow key puts one, and the one change each gesture asks Python
 * for. Pure helpers, so the chart, the gestures and the toolbar read the
 * payload the same way. Python checks every change again when it is sent.
 */

import { fmt } from "./format.js";

/** @typedef {import('./api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('./api/contracts.js').KnotParams} KnotParams */
/** @typedef {import('./api/contracts.js').KnotRule} KnotRule */
/** @typedef {import('./api/contracts.js').KnotStrategy} KnotStrategy */

/**
 * The axis a term's knots sit on, in chart x.
 * @typedef {object} KnotAxis
 * @property {number} lo knots lie strictly between ``lo`` and ``hi``
 * @property {number} hi
 * @property {number} gap the least distance between two knots, and from ``lo`` and ``hi``
 * @property {number} step the grid a knot snaps to
 * @property {number} places the grid's decimal places
 * @property {number|null} maxCount
 * @property {Map<number, string>|null} levels an ordered term's level on the
 *   curve at each whole position; null on a numeric term
 */
/**
 * The knots a term shows: the waiting draft's while a knot change waits,
 * else those in force.
 * @typedef {object} ShownKnots
 * @property {number[]} positions ascending
 * @property {number} count
 * @property {KnotStrategy} strategy
 * @property {number|null} alpha
 * @property {boolean} waiting
 */
/**
 * What a finished gesture asks for: one change, and the knot to keep
 * selected once it is staged; or a sentence for the status line.
 * @typedef {{params:KnotParams, select:number|null}|{refusal:string}} KnotOutcome
 */

/** @type {readonly (readonly [KnotRule, string])[]} */
export const KNOT_RULES = Object.freeze([
  /** @type {const} */ (["uniform", "Even spacing"]),
  /** @type {const} */ (["quantile", "Quantiles of values"]),
  /** @type {const} */ (["quantile_rows", "Quantiles of rows"]),
  /** @type {const} */ (["quantile_tempered", "Tempered quantiles"]),
]);
/** @type {Readonly<Record<KnotRule, string>>} */
const RULE_TEXT = Object.freeze({
  uniform: "even spacing",
  quantile: "quantiles of values",
  quantile_rows: "quantiles of rows",
  quantile_tempered: "tempered quantiles",
});
/** superglm's ``knot_alpha`` default. */
export const DEFAULT_ALPHA = 0.2;
export const AT_LEAST_ONE = "A spline needs at least one knot.";
export const SHOWN_GROUPED =
  "Knots sit on the expanded level axis. Show the groups expanded to see them.";
// A distance on the grid may fall short of the least gap by this fraction of
// it, the round-off of differencing two grid points; Python allows the same.
const SLACK = 1e-9;

/** @param {unknown} value @returns {value is number} */
function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * The axis ``term``'s knots sit on, or null when it has none to adjust. A
 * numeric term snaps to three significant figures of its span, the grid a
 * shaped range's edges snap to; an ordered term to a tenth of a level.
 * @param {TermPayload} term @returns {KnotAxis|null}
 */
export function knotAxis(term) {
  const knots = term.knots;
  if (!knots || !Array.isArray(knots.positions)) return null;
  const { lo, hi } = knots;
  if (!isFiniteNumber(lo) || !isFiniteNumber(hi) || !(hi > lo)) return null;
  const ordered = Array.isArray(term.levels);
  const exponent = ordered ? -1 : Math.floor(Math.log10(hi - lo)) - 2;
  return {
    lo,
    hi,
    gap: isFiniteNumber(knots.min_gap) && knots.min_gap > 0 ? knots.min_gap : 0,
    step: 10 ** exponent,
    places: Math.max(0, -exponent),
    maxCount: isFiniteNumber(knots.max_count) ? knots.max_count : null,
    levels: ordered ? curveLevels(term) : null,
  };
}

/**
 * An ordered term's levels on its curve by display position: smooth level
 * ``i`` sits at ``i``; special levels have no place on the curve.
 * @param {TermPayload} term @returns {Map<number, string>}
 */
function curveLevels(term) {
  const specials = new Set(term.shape?.specials ?? []);
  /** @type {Map<number, string>} */
  const levels = new Map();
  (term.levels ?? []).forEach((label, index) => {
    const x = term.x[index];
    if (Number.isInteger(x) && !specials.has(String(label))) levels.set(x, String(label));
  });
  return levels;
}

/**
 * ``x`` on the axis's grid: the nearest grid point, or the next one up
 * (``direction`` 1) or down (-1). Rounding to the grid's places drops the
 * binary residue of ``k * step``.
 * @param {number} x @param {KnotAxis} axis @param {-1|0|1} [direction]
 */
export function snapKnot(x, axis, direction = 0) {
  const ratio = x / axis.step;
  const k = direction > 0
    ? Math.ceil(ratio - SLACK)
    : direction < 0 ? Math.floor(ratio + SLACK) : Math.round(ratio);
  return Number((k * axis.step).toFixed(axis.places));
}

/**
 * Whether a knot at ``x`` lies inside the axis and keeps its distance from
 * its ends and from every one of ``others``.
 * @param {number} x @param {readonly number[]} others @param {KnotAxis} axis
 */
export function knotFits(x, others, axis) {
  const tight = axis.gap * (1 - SLACK);
  if (!(x > axis.lo && x < axis.hi)) return false;
  if (x - axis.lo < tight || axis.hi - x < tight) return false;
  return others.every((other) => {
    const distance = Math.abs(other - x);
    return distance > 0 && distance >= tight;
  });
}

/**
 * Where a dragged knot goes: on the grid, and no nearer either end than the
 * least gap. It may pass its neighbours on the way.
 * @param {number} x @param {KnotAxis} axis
 */
export function clampKnot(x, axis) {
  let low = snapKnot(axis.lo + axis.gap, axis, 1);
  if (!(low > axis.lo)) low = snapKnot(low + axis.step, axis);
  let high = snapKnot(axis.hi - axis.gap, axis, -1);
  if (!(high < axis.hi)) high = snapKnot(high - axis.step, axis);
  return Math.min(high, Math.max(low, snapKnot(x, axis)));
}

/**
 * The grid points next to the knots and ends that a knot at ``x`` might crowd:
 * where the nearest free spot to any point lies.
 * @param {readonly number[]} others @param {KnotAxis} axis
 */
function spotsBeside(others, axis) {
  return [
    ...others.flatMap((other) => [
      snapKnot(other - axis.gap, axis, -1),
      snapKnot(other + axis.gap, axis, 1),
    ]),
    clampKnot(axis.lo, axis),
    clampKnot(axis.hi, axis),
  ].filter((spot) => knotFits(spot, others, axis));
}

/** @param {number[]} spots @param {number} x */
function nearest(spots, x) {
  return spots.reduce((best, spot) => (Math.abs(spot - x) < Math.abs(best - x) ? spot : best));
}

/**
 * Where a knot dropped at ``x`` settles: at ``x`` when it keeps its distance
 * from the others, else at the nearest grid point that does; null when no
 * point does, and the knot stays where it was.
 * @param {readonly number[]} others the other knots @param {number} x on the grid
 * @param {KnotAxis} axis @returns {number|null}
 */
export function freeSpot(others, x, axis) {
  if (knotFits(x, others, axis)) return x;
  const spots = spotsBeside(others, axis);
  return spots.length ? nearest(spots, x) : null;
}

/**
 * Where an arrow key moves knot ``index``: ``steps`` grid steps toward
 * ``direction``, or, where that crowds a neighbour, to the nearest free spot
 * beyond it, so a knot can pass its neighbours from the keyboard too. Null
 * when it cannot move that way.
 * @param {readonly number[]} positions @param {number} index @param {-1|1} direction
 * @param {number} steps @param {KnotAxis} axis @returns {number|null}
 */
export function nudgeKnot(positions, index, direction, steps, axis) {
  const x = positions[index];
  const others = positions.filter((_, i) => i !== index);
  const target = clampKnot(x + direction * steps * axis.step, axis);
  if ((target - x) * direction > 0 && knotFits(target, others, axis)) return target;
  const beyond = spotsBeside(others, axis).filter((spot) => (spot - x) * direction > 0);
  return beyond.length ? nearest(beyond, target) : null;
}

/** @param {Iterable<number>} positions @returns {KnotParams} */
function byHand(positions) {
  return { positions: [...positions].sort((left, right) => left - right) };
}

/**
 * Why an ordered term takes no more knots, as Python says it: ``max_count``
 * is one fewer than the levels on its curve.
 * @param {number} max
 */
export function tooManyKnots(max) {
  return `This term has ${max + 1} levels on its curve, so it takes at most ${max} `
    + `${max === 1 ? "knot" : "knots"}.`;
}

/**
 * Releasing a dragged knot: below the axis it is removed, never the last
 * one; dropped elsewhere it moves to where it settles. A press that did not
 * move, or a drop where it started or with nowhere free, asks for nothing.
 * @param {readonly number[]} positions the knots shown, ascending
 * @param {number} index the dragged knot
 * @param {{x:number, moved:boolean, remove:boolean}} drag
 * @param {KnotAxis} axis @returns {KnotOutcome|null}
 */
export function dropOutcome(positions, index, drag, axis) {
  if (drag.remove) return removeOutcome(positions, index);
  if (!drag.moved) return null;
  const others = positions.filter((_, i) => i !== index);
  const spot = freeSpot(others, drag.x, axis);
  if (spot === null || spot === positions[index]) return null;
  return { params: byHand([...others, spot]), select: spot };
}

/**
 * Removing knot ``index``, which leaves at least one.
 * @param {readonly number[]} positions @param {number} index @returns {KnotOutcome}
 */
export function removeOutcome(positions, index) {
  if (positions.length <= 1) return { refusal: AT_LEAST_ONE };
  return { params: byHand(positions.filter((_, i) => i !== index)), select: null };
}

/**
 * Moving knot ``index`` to ``x``, as an arrow key does.
 * @param {readonly number[]} positions @param {number} index @param {number} x
 * @returns {KnotOutcome}
 */
export function moveOutcome(positions, index, x) {
  return { params: byHand(positions.map((value, i) => (i === index ? x : value))), select: x };
}

/**
 * Where a click on the axis band adds a knot: the grid point under the
 * pointer, when a knot fits there; null where it would crowd one.
 * @param {readonly number[]} positions @param {number} x @param {KnotAxis} axis
 * @returns {number|null}
 */
export function addSpot(positions, x, axis) {
  const spot = snapKnot(x, axis);
  return knotFits(spot, positions, axis) ? spot : null;
}

/**
 * A click on the axis band: a knot at ``spot``, refused with a sentence when
 * the term has as many as it takes; nothing where none fits.
 * @param {readonly number[]} positions @param {number|null} spot from addSpot
 * @param {KnotAxis} axis @returns {KnotOutcome|null}
 */
export function addOutcome(positions, spot, axis) {
  if (axis.maxCount !== null && positions.length >= axis.maxCount) {
    return { refusal: tooManyKnots(axis.maxCount) };
  }
  return spot === null ? null : { params: byHand([...positions, spot]), select: spot };
}

/**
 * The knots ``term`` shows, or null when it has none to adjust.
 * @param {TermPayload} term @returns {ShownKnots|null}
 */
export function shownKnots(term) {
  const waiting = term.pending?.knots;
  if (waiting && Array.isArray(waiting.positions)) {
    return {
      positions: [...waiting.positions].sort((left, right) => left - right),
      count: waiting.positions.length,
      strategy: waiting.strategy,
      alpha: waiting.alpha ?? null,
      waiting: true,
    };
  }
  const knots = term.knots;
  if (!knots || !Array.isArray(knots.positions)) return null;
  return {
    positions: [...knots.positions].sort((left, right) => left - right),
    count: knots.positions.length,
    strategy: knots.strategy ?? "explicit",
    alpha: knots.alpha ?? null,
    waiting: false,
  };
}

/**
 * The rule a new count re-places the knots by: the shown one, else the one
 * in force, else even spacing, as superglm's own default.
 * @param {TermPayload} term @returns {{strategy:KnotRule, alpha:number}}
 */
export function stepRule(term) {
  const shown = shownKnots(term);
  for (const strategy of [shown?.strategy, term.knots?.strategy]) {
    if (strategy && strategy !== "explicit") {
      return { strategy, alpha: shown?.alpha ?? term.knots?.alpha ?? DEFAULT_ALPHA };
    }
  }
  return { strategy: "uniform", alpha: shown?.alpha ?? DEFAULT_ALPHA };
}

/**
 * ``count`` knots placed by ``strategy``; alpha goes with tempered quantiles only.
 * @param {number} count @param {KnotRule} strategy @param {number} alpha
 * @returns {KnotParams}
 */
export function ruleParams(count, strategy, alpha) {
  return strategy === "quantile_tempered" ? { count, strategy, alpha } : { count, strategy };
}

/**
 * The count stepper's two buttons: whether each can act, and its popover
 * text, the reason when it cannot.
 * @param {ShownKnots} shown @param {KnotAxis} axis @param {KnotRule} rule what re-places them
 */
export function stepperState(shown, axis, rule) {
  /** @param {number} count */
  const replaced = (count) =>
    `${count} ${count === 1 ? "knot" : "knots"}, all re-placed by ${RULE_TEXT[rule]}.`;
  const fewer = shown.count <= 1
    ? { enabled: false, body: AT_LEAST_ONE }
    : { enabled: true, body: replaced(shown.count - 1) };
  const more = axis.maxCount !== null && shown.count >= axis.maxCount
    ? { enabled: false, body: tooManyKnots(axis.maxCount) }
    : { enabled: true, body: replaced(shown.count + 1) };
  return { fewer, more };
}

/**
 * The context chip: how many knots the term shows and how they are placed,
 * and whether a knot change waits. Null for a term without knots.
 * @param {TermPayload} term @returns {{text:string, waiting:boolean}|null}
 */
export function knotChip(term) {
  const shown = shownKnots(term);
  if (!shown) return null;
  const placed = shown.strategy !== "explicit"
    ? RULE_TEXT[shown.strategy]
    : shown.waiting || term.knots?.from_editor ? "placed by hand" : "listed in code";
  const count = `${shown.count} ${shown.count === 1 ? "knot" : "knots"}`;
  return { text: `${count} · ${placed}`, waiting: shown.waiting };
}

/**
 * The tag over a dragged or selected knot: the value on a numeric axis, as
 * its tick labels print it; on an ordered one, the level it sits at or the
 * two it lies between.
 * @param {number} x @param {KnotAxis} axis
 */
export function knotTagText(x, axis) {
  if (!axis.levels) return fmt(x);
  const whole = Math.round(x);
  if (Math.abs(x - whole) < SLACK) return `at ${axis.levels.get(whole) ?? fmt(x)}`;
  const left = axis.levels.get(Math.floor(x));
  const right = axis.levels.get(Math.ceil(x));
  return left === undefined || right === undefined ? fmt(x) : `${left} to ${right}`;
}

/**
 * Whether the Knots tool works on ``term`` as drawn, and why not. Drawn with
 * its groups collapsed, an ordered term's axis is not the one its knots sit on.
 * @param {TermPayload|null|undefined} term @param {boolean} collapsed
 * @returns {{available:boolean, reason:string|null}}
 */
export function knotToolState(term, collapsed) {
  const knots = term?.knots;
  if (!knots) return { available: false, reason: null };
  if (!knots.available) return { available: false, reason: knots.reason ?? null };
  if (!knotAxis(/** @type {TermPayload} */ (term))) return { available: false, reason: null };
  if (collapsed) return { available: false, reason: SHOWN_GROUPED };
  return { available: true, reason: null };
}
