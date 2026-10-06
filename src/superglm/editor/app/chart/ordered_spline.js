// @ts-check

/**
 * How an ordered categorical with a spline basis is drawn: its spline on a
 * fine grid of the level axis with the level dots on it, and its special
 * levels as dots of their own. Pure helpers, so the chart and the drag preview
 * read the payload's ``spline_view`` the same way.
 */

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

/**
 * The spline lines to draw for ``term``, or null to join its points as usual.
 * ``y`` is null while the edited levels are off the spline (a level edit the
 * basis cannot follow, or a level being dragged); ``originalY`` is null once
 * a structural step has replaced the fit the chart compares against.
 *
 * @param {TermPayload} term
 * @returns {{x:number[], y:number[]|null, originalY:number[]|null, levelIndices:number[]}|null}
 */
export function splineCurves(term) {
  const view = term.spline_view;
  if (!view || !view.available || !view.x || !view.level_indices) return null;
  return {
    x: view.x,
    y: view.fits_levels && view.y ? view.y : null,
    originalY: view.original_y ?? null,
    levelIndices: view.level_indices,
  };
}

/**
 * The polyline through the smooth levels only: special levels are not joined.
 *
 * @param {number[]} x
 * @param {number[]} values
 * @param {number[]} levelIndices
 * @returns {{x:number[], y:number[]}}
 */
export function levelPolyline(x, values, levelIndices) {
  return {
    x: levelIndices.map((index) => x[index]),
    y: levelIndices.map((index) => values[index]),
  };
}

/**
 * The x values the rows of ``controls.build_basis`` are sampled at: the
 * drawing grid for an ordered spline, the term's own x otherwise.
 *
 * @param {TermPayload} term
 * @returns {number[]}
 */
export function contributionX(term) {
  const controls = /** @type {{grid_x?:unknown}|null} */ (term.controls);
  return controls && Array.isArray(controls.grid_x) ? /** @type {number[]} */ (controls.grid_x) : term.x;
}

/**
 * The drawn spline after one coefficient moves by ``deltaLog``: each grid
 * value scales by ``exp(row * deltaLog)``, the same rule the drag preview
 * applies to the level dots.
 *
 * @param {number[]} baseCurve relativities on the grid before the drag
 * @param {number[]} gridRow the moved basis function on the grid
 * @param {number} deltaLog the coefficient's change on the log scale
 * @returns {number[]}
 */
export function shiftedCurve(baseCurve, gridRow, deltaLog) {
  return baseCurve.map((value, index) =>
    Math.max(1e-12, value * Math.exp((Number(gridRow[index]) || 0) * deltaLog))
  );
}
