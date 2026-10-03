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
  return value.toLocaleString("en-US", {
    useGrouping: false,
    maximumFractionDigits: digits,
    minimumFractionDigits: 0
  });
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
