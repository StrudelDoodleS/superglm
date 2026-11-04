// @ts-check
/**
 * The distribution panel's line: reading what is typed into its text box, writing the readouts,
 * and pinning the line to the amount axis. All three are pure functions of their arguments.
 *
 * The line is either an amount or a percentile. A typed percentile or amount is recognised in
 * either mode; only a bare number follows the mode the panel is in.
 *
 * Grammar, after lower-casing and removing whitespace and commas, then the unit as a leading or
 * trailing token, then one currency symbol (`€`, `£`, `$`) at each end:
 *  - `1inN`          the percentile p = 1 - 1/N, written (N - 1) / N so that it is rounded once;
 *  - `v` + `st|nd|rd|th|%|pct`   the percentile p = v / 100;
 *  - `v` + `k|m`     the amount v * 1e3 or v * 1e6;
 *  - `v`             an amount if a currency symbol or the unit was removed, otherwise per mode.
 * The unit is matched before the symbols so that a unit made of letters and a symbol (`US$`, `R$`)
 * is found, and neither is removed from the middle of the text, so `2€5` is not 25.
 * An amount must be positive and finite, and p must lie strictly between 0 and 1.
 */

/** @typedef {"amount" | "percentile"} LineMode */
/** @typedef {{ok: true, mode: LineMode, value: number} | {ok: false, hint: string}} LineParse */
/** @typedef {{min_log10: number, max_log10: number}} AmountAxis */

const HINT = "Type an amount such as 25k or a percentile such as 99.5th or 1 in 200.";
const ONE_IN_N_HINT = "Type N as a number above 1, such as 200 or 10k.";
const NUMBER = String.raw`(\d*\.?\d+)`;
const ONE_IN = new RegExp(`^1in${NUMBER}$`);
const PERCENTILE = new RegExp(`^${NUMBER}(st|nd|rd|th|%|pct)$`);
const AMOUNT = new RegExp(`^${NUMBER}([km]?)$`);
const EDGE_CURRENCY_SYMBOLS = /^[€£$]|[€£$]$/g;
/** @type {Record<string, number>} */
const AMOUNT_SCALE = { "": 1, k: 1e3, m: 1e6 };

// A chance below this is written as "less than" the smallest value shown.
const CHANCE_MIN = 1e-6;
const CHANCE_FLOOR_TEXT = "<0.0001%";
const ONE_IN_FLOOR_N = ">1,000,000";

const GROUPED = new Intl.NumberFormat("en-GB");
const THREE_SIGNIFICANT = new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 3 });

/** @returns {LineParse} */
const refused = () => ({ ok: false, hint: HINT });

/**
 * @param {number} p
 * @returns {LineParse}
 */
const asPercentile = (p) =>
  p > 0 && p < 1 ? { ok: true, mode: "percentile", value: p } : refused();

/**
 * @param {number} v
 * @returns {LineParse}
 */
const asAmount = (v) =>
  v > 0 && Number.isFinite(v) ? { ok: true, mode: "amount", value: v } : refused();

/**
 * Removes the unit when it opens or closes the text, and nowhere else.
 * @param {string} text
 * @param {string} unit  lower-case, without whitespace or commas; empty for no unit
 * @returns {string}
 */
function withoutEdgeUnit(text, unit) {
  if (!unit) return text;
  if (text.startsWith(unit)) return text.slice(unit.length);
  return text.endsWith(unit) ? text.slice(0, -unit.length) : text;
}

/**
 * Reads the line's text box.
 * @param {string} raw
 * @param {LineMode} mode  what a bare number means
 * @param {string | null} unit  the panel's amount unit, read like a currency symbol
 * @returns {LineParse}
 */
export function parseLine(raw, mode, unit) {
  const typed = raw.toLowerCase().replace(/[\s,]/g, "");
  const bareUnit = unit ? unit.toLowerCase().replace(/[\s,]/g, "") : "";
  const text = withoutEdgeUnit(typed, bareUnit).replace(EDGE_CURRENCY_SYMBOLS, "");
  const markedAsAmount = text.length < typed.length;

  const oneIn = ONE_IN.exec(text);
  if (oneIn) {
    const n = Number(oneIn[1]);
    return asPercentile((n - 1) / n);
  }
  const percentile = PERCENTILE.exec(text);
  if (percentile) return asPercentile(Number(percentile[1]) / 100);

  const amount = AMOUNT.exec(text);
  if (!amount) return refused();
  const [, digits, suffix] = amount;
  if (suffix || markedAsAmount || mode === "amount") {
    return asAmount(Number(digits) * AMOUNT_SCALE[suffix]);
  }
  return asPercentile(Number(digits) / 100);
}

/**
 * Reads the box beside a fixed "1 in": only N, as a number that may end in k or m. A pasted
 * "1 in 200" reads as 200.
 * @param {string} raw
 * @returns {LineParse}
 */
export function parseOneInN(raw) {
  const text = raw.toLowerCase().replace(/[\s,]/g, "").replace(/^1in/, "");
  const number = AMOUNT.exec(text);
  if (!number) return { ok: false, hint: ONE_IN_N_HINT };
  const n = Number(number[1]) * AMOUNT_SCALE[number[2]];
  const parsed = asPercentile((n - 1) / n);
  return parsed.ok ? parsed : { ok: false, hint: ONE_IN_N_HINT };
}

/**
 * Two significant figures, grouped in thousands, with one decimal place below 10.
 * @param {number} n
 * @returns {string}
 */
function niceN(n) {
  if (n < 10) return n.toFixed(1);
  const step = 10 ** Math.max(0, Math.floor(Math.log10(n)) - 1);
  return GROUPED.format(Math.round(n / step) * step);
}

/**
 * The N of a chance written "1 in N".
 * @param {number} p
 * @returns {string}
 */
export function formatOneInN(p) {
  return p < CHANCE_MIN ? ONE_IN_FLOOR_N : niceN(1 / p);
}

/**
 * A chance as a percentage or as "1 in N". Small chances keep their significant figures instead of
 * rounding to zero.
 * @param {number} p
 * @param {"percent" | "one_in"} style
 * @returns {string}
 */
export function formatChance(p, style) {
  if (style === "one_in") return `1 in ${formatOneInN(p)}`;
  if (p >= 0.01) return `${(100 * p).toFixed(1)}%`;
  return p >= CHANCE_MIN ? `${(100 * p).toPrecision(2)}%` : CHANCE_FLOOR_TEXT;
}

/**
 * An amount to three significant figures, grouped in thousands, behind the unit with no space.
 * @param {number} v
 * @param {string | null} unit
 * @returns {string}
 */
export function formatAmount(v, unit) {
  return `${unit ?? ""}${THREE_SIGNIFICANT.format(v)}`;
}

const ORDINAL_SUFFIX = ["th", "st", "nd", "rd"];

// toFixed accepts at most this many decimals.
const MAX_DECIMALS = 100;

/** @param {string} text */
const isOpenPercent = (text) => Number(text) > 0 && Number(text) < 100;

/**
 * A percentile as an ordinal: whole numbers up to the 99th, one decimal above it. Where that would
 * print 0 or 100 for a p strictly between 0 and 1 (below the 0.5th percentile, from the 99.95th),
 * it adds decimals until the printed percentile lies strictly between 0 and 100, so the text is
 * never `0th` or `100th` and `parseLine` reads it back as the printed digits divided by 100. The
 * widening reaches every p in float64 from 1e-100 up to the largest double below 1.
 * @param {number} p
 * @returns {string}
 */
export function ordinal(p) {
  const percent = p * 100;
  let decimals = p > 0.99 ? 1 : 0;
  let text = percent.toFixed(decimals);
  while (percent > 0 && percent < 100 && !isOpenPercent(text) && decimals < MAX_DECIMALS) {
    decimals += 1;
    text = percent.toFixed(decimals);
  }
  const digits = text.replace(/\.0$/, "");
  const v = Number(digits);
  const n = Math.round(v);
  const teen = n % 100 >= 11 && n % 100 <= 13;
  const suffix = v !== n || teen ? "th" : (ORDINAL_SUFFIX[n % 10] ?? "th");
  return `${digits}${suffix}`;
}

/**
 * Clamps a log10 amount onto the axis; `pinned` is true when the clamp moved it.
 * @param {number} log10v
 * @param {AmountAxis} axis
 * @returns {{log10: number, pinned: boolean}}
 */
export function pinToAxis(log10v, axis) {
  const log10 = Math.min(Math.max(log10v, axis.min_log10), axis.max_log10);
  return { log10, pinned: log10 !== log10v };
}
