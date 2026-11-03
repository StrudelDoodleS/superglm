// @ts-check

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {{x:number, y:number, w:number, h:number}} Bar */
/** @typedef {{kind:"line", d:string}|{kind:"bars", bars:Bar[]}} Sparkline */

// Room kept inside the box so a stroke at the extreme is not clipped.
const PAD = 1.5;
// The share of a bar's slot left empty beside it.
const GAP = 0.2;

/**
 * A feature's effect as a mini curve: a line for a curve over a numeric
 * axis, bars for categorical levels, measured from the term's baseline
 * (the display value of no effect, 1 for a relativity). An ordered
 * categorical reads as a trend, so it is a line.
 *
 * @param {Pick<TermPayload, 'x'|'y'|'levels'|'term_type'> & Partial<TermPayload>} term
 * @param {{width:number, height:number}} size
 * @returns {Sparkline}
 */
export function sparkline(term, size) {
  const bars = Array.isArray(term.levels) && term.levels.length > 0
    && term.term_type !== "ordered categorical";
  return bars ? barSparkline(term, size) : lineSparkline(term, size);
}

/** @param {Pick<TermPayload, 'x'|'y'>} term @param {{width:number, height:number}} size @returns {Sparkline} */
function lineSparkline(term, { width, height }) {
  const points = term.x
    .map((x, i) => [x, term.y[i]])
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!points.length) return { kind: "line", d: "" };
  const across = scale(points.map(([x]) => x), PAD, width - PAD);
  const up = scale(points.map(([, y]) => y), height - PAD, PAD);
  const d = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"} ${round(across(x))} ${round(up(y))}`)
    .join(" ");
  return { kind: "line", d };
}

/** @param {Pick<TermPayload, 'y'> & Partial<TermPayload>} term @param {{width:number, height:number}} size @returns {Sparkline} */
function barSparkline(term, { width, height }) {
  const baseline = term.baseline ?? 1;
  const values = term.y.map((value) => (Number.isFinite(value) ? value : baseline));
  const up = scale([...values, baseline], height - PAD, PAD);
  const zero = up(baseline);
  const slot = width / values.length;
  const barWidth = slot * (1 - GAP);
  return {
    kind: "bars",
    bars: values.map((value, i) => {
      const top = Math.min(up(value), zero);
      return {
        x: round(i * slot + (slot - barWidth) / 2),
        y: round(top),
        w: round(barWidth),
        h: round(Math.abs(up(value) - zero)),
      };
    }),
  };
}

/**
 * A map from the values' range onto [from, to]; a single value lands mid-way.
 * @param {number[]} values @param {number} from @param {number} to
 * @returns {(value:number)=>number}
 */
function scale(values, from, to) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  if (!(hi > lo)) return () => (from + to) / 2;
  return (value) => from + ((value - lo) / (hi - lo)) * (to - from);
}

/** @param {number} value */
function round(value) {
  return Math.round(value * 100) / 100;
}
