// @ts-check
// A term's pointwise intervals and the points the fit does not determine.
//
// The payload's `non_estimable` runs parallel to `y`: the kind of direction
// the fit does not determine at each point, "" where it does. Such a point has
// no interval (`null` ends, `_ci_payload`), so the band is drawn over the runs
// of points between them, and the point says why in its hover text.

/** @typedef {ReadonlyArray<number|null>} Ends */

/** Each kind's reason, short enough for the point tooltip (`diagnostics.estimability`'s). */
const REASONS = /** @type {Record<string, string>} */ ({
  alias: "aliased: held by the constraint",
  separated: "separated: no finite estimate",
  unresolved: "not resolved at float64",
  unfitted: "identified, not fitted"
});

// Half the width of the bar a lone point's interval is drawn as, in pixels.
const BAR_HALF = 1.5;

/**
 * Whether point `i` has an interval to draw: both ends are numbers.
 * @param {Ends} lower
 * @param {Ends} upper
 * @param {number} i
 */
export function hasInterval(lower, upper, i) {
  return Number.isFinite(lower[i]) && Number.isFinite(upper[i]);
}

/**
 * The runs of consecutive points with an interval, each as its indices.
 * @param {Ends} lower
 * @param {Ends} upper
 * @returns {number[][]}
 */
export function intervalRuns(lower, upper) {
  /** @type {number[][]} */
  const runs = [];
  /** @type {number[]|null} */
  let run = null;
  for (let i = 0; i < lower.length; i++) {
    if (!hasInterval(lower, upper, i)) {
      run = null;
      continue;
    }
    if (!run) {
      run = [];
      runs.push(run);
    }
    run.push(i);
  }
  return runs;
}

/**
 * The band's path: one closed area per run, a lone point's a bar `2 * BAR_HALF` px wide.
 * Empty when no point has an interval.
 * @param {ReadonlyArray<number>} x
 * @param {Ends} lower
 * @param {Ends} upper
 * @param {(value: number) => number} sx
 * @param {(value: number) => number} sy
 */
export function bandPath(x, lower, upper, sx, sy) {
  const at = (/** @type {number} */ value) => value.toFixed(2);
  return intervalRuns(lower, upper).map((run) => {
    if (run.length === 1) {
      const [i] = run;
      const px = sx(x[i]);
      const top = at(sy(Number(upper[i])));
      const bottom = at(sy(Number(lower[i])));
      const left = at(px - BAR_HALF);
      const right = at(px + BAR_HALF);
      return `M ${left} ${top} L ${right} ${top} L ${right} ${bottom} L ${left} ${bottom} Z`;
    }
    const top = run.map((i, k) => `${k === 0 ? "M" : "L"} ${at(sx(x[i]))} ${at(sy(Number(upper[i])))}`);
    const bottom = run.slice().reverse().map((i) => `L ${at(sx(x[i]))} ${at(sy(Number(lower[i])))}`);
    return `${top.join(" ")} ${bottom.join(" ")} Z`;
  }).join(" ");
}

/**
 * The interval ends the y range takes: the numbers among them.
 * @param {Ends} lower
 * @param {Ends} upper
 * @returns {number[]}
 */
export function intervalValues(lower, upper) {
  return /** @type {number[]} */ ([...lower, ...upper].filter((value) => Number.isFinite(value)));
}

/**
 * Point `i`'s kind of direction the fit does not determine, "" where it determines it.
 * @param {{non_estimable?: ReadonlyArray<string>|null}} view
 * @param {number} i
 */
export function nonEstimableKind(view, i) {
  const kinds = view.non_estimable;
  return Array.isArray(kinds) && typeof kinds[i] === "string" ? kinds[i] : "";
}

/**
 * The hover lines of a point the fit does not determine, none for one it determines.
 * @param {string} kind
 * @returns {string[]}
 */
export function nonEstimableLines(kind) {
  if (!kind) return [];
  return ["Not determined: no interval", REASONS[kind] ?? kind];
}

/**
 * A collapsed display's kinds: a group takes the first kind among its levels.
 * @param {ReadonlyArray<string>|null|undefined} kinds
 * @param {ReadonlyArray<ReadonlyArray<number>>} sourceIndices
 * @returns {string[]|null}
 */
export function collapsedKinds(kinds, sourceIndices) {
  if (!Array.isArray(kinds)) return null;
  return sourceIndices.map((indices) => {
    const index = indices.find((source) => typeof kinds[source] === "string" && kinds[source] !== "");
    return index === undefined ? "" : kinds[index];
  });
}
