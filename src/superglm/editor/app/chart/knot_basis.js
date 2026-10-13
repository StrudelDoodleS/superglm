// @ts-check
// A spline's B-spline basis in the browser, rebuilt from the construction the
// payload describes (term.knots.basis), so the chart can draw the functions a
// knot change reaches while a knot is dragged or a change waits. Evaluation is
// the Cox–de Boor recursion as Piegl and Tiller lay it out (The NURBS Book,
// 2nd ed., Springer, 1997: FindSpan, Algorithm A2.1, p. 68; BasisFuns,
// Algorithm A2.2, p. 70). Pure, so it tests without a DOM.

/** @typedef {import('../api/contracts.js').KnotBasis} KnotBasis */

/**
 * Chart x on the spline's own axis: the identity on a numeric term, linear
 * between an ordered term's level values, smooth level ``i`` at chart ``i``.
 * @param {number[]|null} levelValues @returns {(x:number)=>number}
 */
export function axisMap(levelValues) {
  if (!levelValues || levelValues.length < 2) return (x) => x;
  const last = levelValues.length - 1;
  return (x) => {
    const i = Math.min(Math.max(Math.floor(x), 0), last - 1);
    return levelValues[i] + (x - i) * (levelValues[i + 1] - levelValues[i]);
  };
}

/**
 * The full knot vector on the spline's axis, as superglm assembles it from the
 * interior knots: "open" widens the boundary by 0.001 of its range and carries
 * the knots on past it at the first and last spacing; "clamped" repeats each
 * end of the boundary ``degree + 1`` times.
 * @param {KnotBasis} basis @param {readonly number[]} positions interior knots, chart x
 * @returns {number[]}
 */
export function knotVector(basis, positions) {
  const toAxis = axisMap(basis.level_values);
  const p = basis.degree;
  const lo = toAxis(basis.boundary[0]);
  const hi = toAxis(basis.boundary[1]);
  const interior = [...positions].sort((a, b) => a - b).map(toAxis);
  if (basis.ends === "clamped") {
    return [...Array(p + 1).fill(lo), ...interior, ...Array(p + 1).fill(hi)];
  }
  const pad = 0.001 * (hi - lo);
  const inner = [lo - pad, ...interior, hi + pad];
  const below = inner[1] - inner[0];
  const above = inner[inner.length - 1] - inner[inner.length - 2];
  return [
    ...Array.from({ length: p }, (_, k) => inner[0] - below * (p - k)),
    ...inner,
    ...Array.from({ length: p }, (_, k) => inner[inner.length - 1] + above * (k + 1)),
  ];
}

/** How many basis functions a knot vector carries. @param {readonly number[]} knots @param {number} degree */
export function basisCount(knots, degree) {
  return knots.length - degree - 1;
}

/**
 * The non-zero basis functions at ``u``: the index of the first and the
 * ``degree + 1`` values from it. ``u`` outside the basis's span takes the end
 * span, where the values extrapolate the end pieces.
 * @param {readonly number[]} knots @param {number} degree @param {number} u
 * @returns {{first:number, values:number[]}}
 */
export function basisAt(knots, degree, u) {
  const span = findSpan(knots, degree, u);
  const values = new Array(degree + 1).fill(0);
  const left = new Array(degree + 1).fill(0);
  const right = new Array(degree + 1).fill(0);
  values[0] = 1;
  for (let j = 1; j <= degree; j++) {
    left[j] = u - knots[span + 1 - j];
    right[j] = knots[span + j] - u;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const temp = values[r] / (right[r + 1] + left[j - r]);
      values[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    values[j] = saved;
  }
  return { first: span - degree, values };
}

/**
 * The knot span holding ``u``: the ``i`` with ``knots[i] <= u < knots[i+1]``
 * among the spans the basis covers, the last one taking its right end.
 * @param {readonly number[]} knots @param {number} degree @param {number} u
 */
function findSpan(knots, degree, u) {
  const n = basisCount(knots, degree) - 1;
  if (u >= knots[n + 1]) return n;
  if (u <= knots[degree]) return degree;
  let low = degree;
  let high = n + 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (u < knots[middle]) high = middle;
    else low = middle;
  }
  return low;
}

/**
 * The basis functions moving one interior knot reshapes: the ``degree + 2``
 * whose support holds it. Function ``i`` is built on ``knots[i..i+degree+1]``,
 * and interior knot ``k`` sits at ``knots[k + degree + 1]`` in either
 * construction.
 * @param {number} interiorIndex the knot's place among the interior knots, ascending
 * @param {number} degree @param {number} count how many functions the basis has
 * @returns {number[]}
 */
export function functionsHoldingKnot(interiorIndex, degree, count) {
  const at = interiorIndex + degree + 1;
  const holding = [];
  for (let i = Math.max(0, at - degree - 1); i <= Math.min(at, count - 1); i++) holding.push(i);
  return holding;
}

/**
 * The basis functions of ``next`` that ``previous`` does not have: a function
 * is its knots ``i..i+degree+1``, so one whose knots match a function of
 * ``previous`` within ``tolerance`` is unchanged, wherever it sits.
 * @param {readonly number[]} next @param {readonly number[]} previous
 * @param {number} degree @param {number} tolerance @returns {number[]}
 */
export function changedFunctions(next, previous, degree, tolerance) {
  /** @param {readonly number[]} knots @param {number} i */
  const local = (knots, i) => knots.slice(i, i + degree + 2);
  const before = Array.from({ length: basisCount(previous, degree) }, (_, i) => local(previous, i));
  const changed = [];
  for (let i = 0; i < basisCount(next, degree); i++) {
    const own = local(next, i);
    const kept = before.some((knots) => knots.every((t, k) => Math.abs(t - own[k]) <= tolerance));
    if (!kept) changed.push(i);
  }
  return changed;
}

/**
 * Every basis function on a grid of chart x: ``curves[i][g]`` is function
 * ``i`` at ``grid[g]``.
 * @param {KnotBasis} basis @param {readonly number[]} positions interior knots, chart x
 * @param {readonly number[]} grid chart x @returns {number[][]}
 */
export function basisCurves(basis, positions, grid) {
  const knots = knotVector(basis, positions);
  const toAxis = axisMap(basis.level_values);
  const curves = Array.from({ length: basisCount(knots, basis.degree) }, () =>
    new Array(grid.length).fill(0));
  grid.forEach((x, g) => {
    const { first, values } = basisAt(knots, basis.degree, toAxis(x));
    values.forEach((value, r) => { curves[first + r][g] = value; });
  });
  return curves;
}
