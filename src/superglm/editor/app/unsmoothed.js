// @ts-check

/**
 * The Unsmoothed line in the browser: whether the toolbar toggle offers itself
 * for a term and what it shows, the line's entry for the fit in force, and
 * where the line goes on the displayed axis. Pure helpers, so main.js and the
 * chart read the payload the same way.
 */

/** @typedef {import('./api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('./api/contracts.js').UnsmoothedLine} UnsmoothedLine */
/** @typedef {import('./api/contracts.js').UnsmoothedEntry} UnsmoothedEntry */

/** The toggle's hover text and Help entry: what the line is. */
export const UNSMOOTHED_HELP =
  "Draw the spline fitted again with its smoothing switched off, over the curve. It keeps its "
  + "knots, and every other term stays as fitted. One fit per term, kept until the model changes.";
export const UNSMOOTHED_BUSY = "Fitting the model with this term's smoothing switched off.";
// The line may stretch the y range by the curve's own range on each side, so
// to three times it, and by at least the 0.1 a flat curve's chart is tall.
const MIN_REACH = 0.1;

/**
 * The term's line, refusal or pending fit for the fit in force; null when there is none.
 * @param {Record<string, UnsmoothedEntry>|null|undefined} entries
 * @param {string} term @param {number|undefined} fitToken
 * @returns {UnsmoothedEntry|null}
 */
export function unsmoothedEntry(entries, term, fitToken) {
  const entry = entries?.[term];
  return entry && entry.fit_token === fitToken ? entry : null;
}

/**
 * ``entries`` with ``entry`` for ``term``, unless the term already holds one
 * for a later fit: a slow answer for a fit since replaced does not overwrite it.
 * @param {Record<string, UnsmoothedEntry>} entries
 * @param {string} term @param {UnsmoothedEntry} entry
 * @returns {Record<string, UnsmoothedEntry>}
 */
export function withUnsmoothed(entries, term, entry) {
  const current = entries[term];
  if (current && current.fit_token > entry.fit_token) return entries;
  return { ...entries, [term]: entry };
}

/**
 * What the toggle shows for ``term``: hidden on a term without smoothing to
 * switch off; pressed while the choice is on; busy while its fit runs;
 * disabled, with the sentence why, where the fit in force refused it.
 * @param {boolean} show the choice, kept for the session
 * @param {TermPayload|null|undefined} term
 * @param {UnsmoothedEntry|null} entry
 * @returns {{hidden:boolean, pressed:boolean, busy:boolean, disabled:boolean, body:string}}
 */
export function unsmoothedToggle(show, term, entry) {
  const refused = entry?.status === "refused";
  const failed = entry?.status === "failed";
  const running = entry?.status === "running";
  const note = entry?.status === "ready" ? entry.line?.note : null;
  return {
    hidden: !term?.unsmoothed,
    pressed: show,
    busy: show && running,
    disabled: show && refused,
    body: (show && (refused || failed) && entry?.reason)
      || (show && running ? UNSMOOTHED_BUSY : null)
      || (show && note ? `${UNSMOOTHED_HELP} ${note}` : UNSMOOTHED_HELP)
  };
}

/**
 * The spline's line on its own grid, null where a value is not finite.
 * @param {UnsmoothedLine|null} line
 * @returns {{x:number[], y:Array<number|null>}|null}
 */
export function unsmoothedSeries(line) {
  if (!line?.x) return null;
  const finite = (/** @type {number|null} */ value) =>
    (typeof value === "number" && Number.isFinite(value) ? value : null);
  return { x: line.x.slice(), y: line.y.map(finite) };
}

/**
 * The runs of a line between its gaps; a run of one point is a lone level.
 * @param {{x:number[], y:Array<number|null>}} series
 * @returns {Array<{x:number[], y:number[]}>}
 */
export function unsmoothedRuns(series) {
  /** @type {Array<{x:number[], y:number[]}>} */
  const runs = [];
  /** @type {{x:number[], y:number[]}|null} */
  let run = null;
  series.y.forEach((value, i) => {
    if (value === null) {
      run = null;
      return;
    }
    if (!run) {
      run = { x: [], y: [] };
      runs.push(run);
    }
    run.x.push(series.x[i]);
    run.y.push(value);
  });
  return runs;
}

/**
 * The y range with the line in it, as the chart takes any other series, but
 * stretched past the curve's own range ``[low, high]`` by at most that range
 * again on each side: a wild unsmoothed spline cannot flatten the curve, and
 * runs off the plot instead.
 * @param {number} low @param {number} high @param {Array<number|null>} values
 * @returns {[number, number]}
 */
export function unsmoothedRange(low, high, values) {
  const finite = /** @type {number[]} */ (values.filter((value) => typeof value === "number" && Number.isFinite(value)));
  if (!finite.length) return [low, high];
  const reach = Math.max(high - low, MIN_REACH);
  return [
    Math.max(Math.min(low, ...finite), low - reach),
    Math.min(Math.max(high, ...finite), high + reach)
  ];
}
