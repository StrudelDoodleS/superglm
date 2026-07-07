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

export function fmtSigned(value) {
  const formatted = fmt(value);
  if (!formatted || Math.abs(value) < 1e-15) return "0";
  return value > 0 ? `+${formatted}` : formatted;
}

export function fmtPercent(value) {
  return `${fmt(100 * value)}%`;
}

export function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
