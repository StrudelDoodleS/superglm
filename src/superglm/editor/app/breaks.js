// @ts-check
// Pure break-draft logic for the Breaks tool: no DOM, no store. A draft never
// changes predictions; Python validates the payload again when it is sent.

/** @typedef {import('./api/contracts.js').BreakDraft} BreakDraft */
/** @typedef {import('./api/contracts.js').BreakForm} BreakForm */
/** @typedef {import('./api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('./api/contracts.js').TransformTermRequest} TransformTermRequest */
/** @typedef {string|number} BreakValue */

export const MAX_SEGMENT_DEGREE = 3;
export const MAX_POLYNOMIAL_DEGREE = 5;
// The library's own Polynomial() default.
const DEFAULT_POLYNOMIAL_DEGREE = 3;
const DEGREE_NAMES = Object.freeze(["flat", "linear", "quadratic", "cubic"]);
/** @type {Readonly<Record<BreakForm, string>>} */
const FORM_HINTS = Object.freeze({
  piecewise: "Click the plot to add a break.",
  spline: "Knots sit at the breaks.",
  polynomial: "One polynomial across the whole axis."
});

/** @param {TermPayload|null|undefined} term @returns {term is TermPayload} */
export function isTransformable(term) {
  return Boolean(term?.transform);
}

/** @param {TermPayload} term @returns {BreakDraft} */
export function initialDraft(term) {
  const piecewise = term.transform?.piecewise;
  return {
    form: "piecewise",
    breaks: piecewise ? piecewise.breaks.slice() : [],
    degrees: piecewise ? piecewise.degrees.slice() : [1],
    degree: DEFAULT_POLYNOMIAL_DEGREE
  };
}

/** @param {TermPayload} term @returns {string[]|null} */
function bandAxis(term) {
  return term.transform?.axis ?? null;
}

/**
 * The axis position of a break: its band index, or the value itself.
 * @param {TermPayload} term @param {BreakValue} value @returns {number}
 */
function position(term, value) {
  const axis = bandAxis(term);
  return axis ? axis.indexOf(String(value)) : Number(value);
}

/** @param {TermPayload} term @param {BreakValue} value @returns {number} */
export function breakX(term, value) {
  const axis = bandAxis(term);
  if (!axis) return Number(value);
  const levels = term.levels ?? [];
  return term.x[levels.indexOf(String(value))];
}

/**
 * The break a pointer position stands for: the nearest interior band, or the
 * value on the numeric grid, strictly inside the fitted range.
 * @param {TermPayload} term @param {number} dataX @returns {BreakValue|null}
 */
export function snapBreak(term, dataX) {
  const axis = bandAxis(term);
  if (!axis) return snapValue(term, dataX);
  if (axis.length < 3) return null;
  const distances = axis.map((band) => Math.abs(breakX(term, band) - dataX));
  const nearest = distances.indexOf(Math.min(...distances));
  return axis[Math.min(Math.max(nearest, 1), axis.length - 2)];
}

/**
 * The numeric grid's exponent: three significant figures of the fitted span,
 * not of the value, so an offset axis such as years keeps every position.
 * @param {TermPayload} term
 */
function gridExponent(term) {
  return Math.floor(Math.log10(Math.max(...term.x) - Math.min(...term.x))) - 2;
}

/** @param {TermPayload} term @param {number} dataX @returns {number|null} */
function snapValue(term, dataX) {
  const lo = Math.min(...term.x);
  const hi = Math.max(...term.x);
  const exponent = gridExponent(term);
  const step = 10 ** exponent;
  // toFixed drops a step multiple's binary residue (31 * 0.1 is 3.1000000000000005).
  const value = Number((Math.round(dataX / step) * step).toFixed(Math.max(0, -exponent)));
  return lo < value && value < hi ? value : null;
}

/**
 * One band or one grid step away, or null off the interior.
 * @param {TermPayload} term @param {BreakValue} value @param {-1|1} direction
 * @returns {BreakValue|null}
 */
export function stepBreak(term, value, direction) {
  const axis = bandAxis(term);
  if (!axis) return snapValue(term, Number(value) + direction * 10 ** gridExponent(term));
  const index = axis.indexOf(String(value)) + direction;
  return 1 <= index && index <= axis.length - 2 ? axis[index] : null;
}

/**
 * @param {TermPayload} term @param {BreakDraft} draft @param {BreakValue|null} value
 * @returns {BreakDraft}
 */
export function addBreak(term, draft, value) {
  if (value === null) return draft;
  const positions = draft.breaks.map((b) => position(term, b));
  const at = position(term, value);
  if (positions.includes(at)) return draft;
  const index = positions.filter((p) => p < at).length;
  const degree = draft.degrees[index];
  // A flat segment split in two would be two flats in a row: the right half
  // becomes linear.
  const right = degree === 0 ? 1 : degree;
  return withinDegreeCaps(term, {
    ...draft,
    breaks: [...draft.breaks.slice(0, index), value, ...draft.breaks.slice(index)],
    degrees: [...draft.degrees.slice(0, index), degree, right, ...draft.degrees.slice(index + 1)]
  });
}

/**
 * @param {TermPayload} term @param {BreakDraft} draft @param {number} index
 * @param {BreakValue|null} value
 * @returns {BreakDraft}
 */
export function moveBreak(term, draft, index, value) {
  if (value === null) return draft;
  const at = position(term, value);
  const left = index > 0 ? position(term, draft.breaks[index - 1]) : -Infinity;
  const right = index < draft.breaks.length - 1
    ? position(term, draft.breaks[index + 1])
    : Infinity;
  if (!(left < at && at < right) || at === position(term, draft.breaks[index])) return draft;
  const breaks = draft.breaks.slice();
  breaks[index] = value;
  return withinDegreeCaps(term, { ...draft, breaks });
}

/**
 * A narrower segment keeps no more degree than its span holds: the library
 * refuses the draft otherwise.
 * @param {TermPayload} term @param {BreakDraft} draft @returns {BreakDraft}
 */
function withinDegreeCaps(term, draft) {
  const degrees = draft.degrees.map(
    (degree, segment) => Math.min(degree, maxSegmentDegree(term, draft, segment))
  );
  return { ...draft, degrees };
}

/** @param {TermPayload} term @param {BreakDraft} draft @param {number} index */
export function removeBreak(term, draft, index) {
  const merged = Math.max(draft.degrees[index], draft.degrees[index + 1]);
  return {
    ...draft,
    breaks: draft.breaks.filter((_, i) => i !== index),
    degrees: [...draft.degrees.slice(0, index), merged, ...draft.degrees.slice(index + 2)]
  };
}

/**
 * min(3, span) on a band axis, where span is the gap between the segment's
 * break positions; 1 on a numeric axis, where segments are straight lines.
 * @param {TermPayload} term @param {BreakDraft} draft @param {number} segment
 */
export function maxSegmentDegree(term, draft, segment) {
  const axis = bandAxis(term);
  if (!axis) return 1;
  const edges = [0, ...draft.breaks.map((b) => position(term, b)), axis.length - 1];
  return Math.min(MAX_SEGMENT_DEGREE, edges[segment + 1] - edges[segment]);
}

/** @param {TermPayload} term @param {BreakDraft} draft @param {number} segment */
export function cycleDegree(term, draft, segment) {
  const next = draft.degrees[segment] + 1;
  const degrees = draft.degrees.slice();
  degrees[segment] = next > maxSegmentDegree(term, draft, segment) ? 0 : next;
  return { ...draft, degrees };
}

/** @param {BreakDraft} draft @param {number} degree @returns {BreakDraft} */
export function setPolynomialDegree(draft, degree) {
  return { ...draft, degree: Math.min(MAX_POLYNOMIAL_DEGREE, Math.max(1, degree)) };
}

/** @param {TermPayload} term @param {BreakDraft} draft @returns {string|null} */
export function draftProblem(term, draft) {
  if (draft.form === "polynomial") return null;
  if (!draft.breaks.length) return "Add at least one break.";
  if (draft.form === "spline") return null;
  if (draft.degrees.every((d) => d === 0)) {
    return "Give at least one segment a degree: all-flat is a constant.";
  }
  if (draft.degrees.some((d, i) => i > 0 && d === 0 && draft.degrees[i - 1] === 0)) {
    return "Two flat segments in a row make one plateau: remove the break between them.";
  }
  return null;
}

/** @param {BreakDraft} draft */
export function formHint(draft) {
  return FORM_HINTS[draft.form];
}

/** @param {number} degree */
export function degreeName(degree) {
  return DEGREE_NAMES[degree];
}

/**
 * The /transform_term request: degrees only for a piecewise draft on bands.
 * @param {string} termName @param {TermPayload} term @param {BreakDraft} draft
 * @returns {TransformTermRequest}
 */
export function transformPayload(termName, term, draft) {
  if (draft.form === "polynomial") {
    return { term: termName, form: "polynomial", breaks: [], degree: draft.degree };
  }
  const payload = { term: termName, form: draft.form, breaks: draft.breaks.slice() };
  if (draft.form === "piecewise" && bandAxis(term)) {
    return { ...payload, degrees: draft.degrees.slice() };
  }
  return payload;
}
