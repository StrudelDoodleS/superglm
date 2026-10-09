// @ts-check

/**
 * Special levels of an ordered term in the browser: which of Make special and
 * Back on the curve the selection can take, which levels a waiting change
 * takes off the curve or puts back, and where the free-level comparison's
 * marks go. Pure helpers, so main.js and the chart read the payload the same
 * way.
 */

/** @typedef {import('./api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('./api/contracts.js').FreeLevels} FreeLevels */

export const DECLARED_SPECIAL =
  "Declared special in the term's code, so it has no place on the curve. "
  + "Change the declaration to put it there.";
export const REFERENCE_STAYS =
  "The reference stays on the curve. Set another reference first.";

/**
 * @typedef {object} SpecialAction
 * @property {boolean} visible
 * @property {boolean} enabled
 * @property {string|null} reason why it is disabled, for its popover
 */

/**
 * The two actions for a selection of ``labels``. Make special shows when every
 * selected level is on the curve, Back on the curve when every one is special;
 * each is disabled, with its reason, where the term cannot take it.
 * @param {TermPayload} term @param {string[]} labels displayed source levels
 * @returns {{make: SpecialAction, back: SpecialAction}}
 */
export function specialActions(term, labels) {
  const hidden = { visible: false, enabled: false, reason: null };
  const type = term.term_type || term.kind || "";
  if (type !== "ordered categorical" || !labels.length || !term.shape) {
    return { make: hidden, back: hidden };
  }
  const specials = new Set(term.shape.specials || []);
  const returnable = new Set(term.shape.returnable || []);
  const allSpecial = labels.every((label) => specials.has(label));
  const noneSpecial = labels.every((label) => !specials.has(label));
  const reference = term.reference?.level ?? null;
  const holdsReference = reference !== null && labels.includes(String(reference));
  return {
    make: noneSpecial
      ? { visible: true, enabled: !holdsReference, reason: holdsReference ? REFERENCE_STAYS : null }
      : hidden,
    back: allSpecial
      ? (labels.every((label) => returnable.has(label))
        ? { visible: true, enabled: true, reason: null }
        : { visible: true, enabled: false, reason: DECLARED_SPECIAL })
      : hidden
  };
}

/**
 * The levels the term's waiting changes take off the curve, and those they put
 * back: its special levels once they apply against those in force.
 * @param {TermPayload} term
 * @returns {{freed: string[], returned: string[]}}
 */
export function waitingSpecials(term) {
  const waiting = term.pending?.specials;
  if (!Array.isArray(waiting)) return { freed: [], returned: [] };
  const now = new Set(term.shape?.specials || []);
  const next = new Set(waiting);
  return {
    freed: waiting.filter((label) => !now.has(label)),
    returned: [...now].filter((label) => !next.has(label))
  };
}

/**
 * @typedef {object} FreeLevelMark
 * @property {string} level
 * @property {number} x the displayed point's x
 * @property {number} y the free estimate, a relativity
 * @property {number} lower
 * @property {number} upper
 * @property {boolean} flagged its interval misses the curve
 */

/**
 * Where the comparison's free estimates go on the displayed axis: one mark per
 * displayed point a compared level stands at. A collapsed group shows its
 * members as one point, which shares their one free estimate.
 * @param {FreeLevels|null} free
 * @param {{x:number[], levels?:string[]|null, displayIsCollapsed?:boolean,
 *   displaySourceLevels?:string[][]}} view
 * @returns {FreeLevelMark[]}
 */
export function freeLevelMarks(free, view) {
  if (!free || !Array.isArray(view.levels)) return [];
  const flagged = new Set(free.flagged);
  /** @type {Map<string, number>} */
  const pointOf = new Map();
  view.levels.forEach((label, index) => {
    const members = view.displayIsCollapsed && view.displaySourceLevels?.[index]
      ? view.displaySourceLevels[index]
      : [label];
    for (const member of members) pointOf.set(String(member), index);
  });
  const seen = new Set();
  /** @type {FreeLevelMark[]} */
  const marks = [];
  free.levels.forEach((level, k) => {
    const index = pointOf.get(level);
    if (index === undefined || seen.has(index)) return;
    seen.add(index);
    marks.push({
      level,
      x: view.x[index],
      y: free.y[k],
      lower: free.lower[k],
      upper: free.upper[k],
      flagged: flagged.has(level)
    });
  });
  return marks;
}

/**
 * Whether ``free`` is the comparison for the term and model in view.
 * @param {FreeLevels|null} free @param {string} term @param {number|undefined} revision
 */
export function freeLevelsShown(free, term, revision) {
  return Boolean(free && free.term === term && free.model_revision === revision);
}
