// @ts-check

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {{x:number, y:number, w:number, h:number, reference:boolean}} Bar */
/** @typedef {{x:number, y:number}} LineMark */
/** @typedef {{kind:"line", d:string, reference?:LineMark}|{kind:"bars", bars:Bar[]}} Sparkline */

// Room kept inside the box so a stroke at the extreme is not clipped.
const PAD = 1.5;
// The share of a bar's slot left empty beside it.
const GAP = 0.2;
// The least height a level is drawn at, so a level at or near no effect still shows.
const MIN_BAR = 1.5;
// The least height of the reference level's mark, which is at no effect by definition.
const REFERENCE_BAR = 3;

/**
 * A feature's effect as a mini curve: a line for a curve over a numeric
 * axis, bars for categorical levels, measured from the term's baseline
 * (the display value of no effect, 1 for a relativity). A level too close
 * to it for a bar is a short mark centred on it, and the reference level is
 * flagged. An ordered categorical reads as a trend, so it is a line with a
 * mark at its reference level.
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

/** @param {Pick<TermPayload, 'x'|'y'> & Partial<TermPayload>} term @param {{width:number, height:number}} size @returns {Sparkline} */
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
  const [first] = referenceLevels(term);
  if (!(Number.isFinite(term.x[first]) && Number.isFinite(term.y[first]))) return { kind: "line", d };
  return { kind: "line", d, reference: { x: round(across(term.x[first])), y: round(up(term.y[first])) } };
}

/** @param {Pick<TermPayload, 'y'> & Partial<TermPayload>} term @param {{width:number, height:number}} size @returns {Sparkline} */
function barSparkline(term, { width, height }) {
  const baseline = term.baseline ?? 1;
  const values = term.y.map((value) => (Number.isFinite(value) ? value : baseline));
  const up = scale([...values, baseline], height - PAD, PAD);
  const zero = up(baseline);
  const slot = width / values.length;
  const barWidth = slot * (1 - GAP);
  const references = referenceLevels(term);
  return {
    kind: "bars",
    bars: values.map((value, i) => {
      const reference = references.has(i);
      const least = reference ? REFERENCE_BAR : MIN_BAR;
      const measured = Math.abs(up(value) - zero);
      return {
        x: round(i * slot + (slot - barWidth) / 2),
        y: round(measured < least ? zero - least / 2 : Math.min(up(value), zero)),
        w: round(barWidth),
        h: round(Math.max(measured, least)),
        reference,
      };
    }),
  };
}

/**
 * The positions of the term's reference level: a level, or a group's members.
 * @param {Partial<TermPayload>} term
 * @returns {Set<number>}
 */
function referenceLevels(term) {
  const level = term.reference?.level;
  if (level === undefined) return new Set();
  const group = term.level_groups?.find((entry) => entry.label === level);
  return new Set(group ? group.indices : [term.levels?.indexOf(level) ?? -1]);
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
