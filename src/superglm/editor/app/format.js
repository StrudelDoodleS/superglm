// @ts-check

/** @param {number} value */
export function fmt(value) {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  let digits = 2;
  if (abs < 0.001 && abs > 0) digits = 6;
  else if (abs < 0.01 && abs > 0) digits = 5;
  else if (abs < 0.1) digits = 4;
  else if (abs < 1) digits = 3;
  else if (abs < 10) digits = 2;
  else digits = 1;
  return plainFormatter(digits).format(value);
}

/**
 * A value on a curve that adds an amount, whose units can lie far below 1: as `fmt`, but a
 * value below 0.001 keeps three significant figures (-1.00e-7) rather than rounding to zero.
 * @param {number} value
 */
export function fmtAdded(value) {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  return abs > 0 && abs < 0.001 ? value.toPrecision(3) : fmt(value);
}

/**
 * An exposure or a count: thousands separators and no decimals once it is a
 * hundred or more (126,037, not 126036.5), `fmt` below that.
 * @param {number} value
 */
export function fmtCount(value) {
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value) < 100) return fmt(value);
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

/** @type {Map<number, Intl.NumberFormat>} */
const PLAIN_FORMATTERS = new Map();

// Building a formatter costs far more than using one, and axes format on every frame of a drag.
/** @param {number} digits */
function plainFormatter(digits) {
  let formatter = PLAIN_FORMATTERS.get(digits);
  if (!formatter) {
    formatter = new Intl.NumberFormat("en-US", {
      useGrouping: false,
      maximumFractionDigits: digits,
      minimumFractionDigits: 0
    });
    PLAIN_FORMATTERS.set(digits, formatter);
  }
  return formatter;
}

/**
 * `value` to `figures` significant figures, trailing zeros kept, so figures
 * listed together read alike: 11.0 beside 11.2, 9.00 beside 3.20. A whole part
 * longer than `figures` is kept whole.
 * @param {number} value @param {number} [figures]
 */
export function fmtSignificant(value, figures = 3) {
  if (!Number.isFinite(value)) return "";
  /** @param {number} magnitude */
  const places = (magnitude) => Math.min(100, Math.max(0, figures - 1 - magnitude));
  const magnitude = value === 0 ? 0 : Math.floor(Math.log10(Math.abs(value)));
  const text = value.toFixed(places(magnitude));
  // Rounding can carry into the next power of ten, as 9.996 does to 10.00.
  return Math.abs(Number(text)) >= 10 ** (magnitude + 1)
    ? value.toFixed(places(magnitude + 1))
    : text;
}

/**
 * An EDF as the feature list, the context bar and the inspector's folded
 * lines all print it: three significant figures, so 10.0, 5.00 and 11.3.
 * @param {number} value
 */
export function fmtEdf(value) {
  return `EDF ${fmtSignificant(value)}`;
}

/** @param {number} value */
export function fmtSigned(value) {
  const formatted = fmt(value);
  if (!formatted || Math.abs(value) < 1e-15) return "0";
  return value > 0 ? `+${formatted}` : formatted;
}

/** @param {number} value */
export function fmtPercent(value) {
  return `${fmt(100 * value)}%`;
}

/** @param {unknown} value */
export function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
