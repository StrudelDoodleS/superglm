import { fmt, fmtSigned } from "./format.js";

const metricKeys = [
  "deviance",
  "aic",
  "bic",
  "log_likelihood",
  "explained_deviance",
  "pearson_chi2",
  "effective_df"
];

// Which way is better for each metric; EDF is a size, neither.
const higherIsBetter = new Set(["log_likelihood", "explained_deviance"]);
const lowerIsBetter = new Set(["deviance", "aic", "aicc", "bic", "pearson_chi2"]);

/**
 * Whether a change in `metric` of `delta` is better, worse, or neutral.
 * @param {string} metric @param {number} delta
 * @returns {"better"|"worse"|"neutral"}
 */
export function metricDirection(metric, delta) {
  if (!Number.isFinite(delta) || Math.abs(delta) < 1e-15) return "neutral";
  if (higherIsBetter.has(metric)) return delta > 0 ? "better" : "worse";
  if (lowerIsBetter.has(metric)) return delta < 0 ? "better" : "worse";
  return "neutral";
}

function metricLabel(metric, metricSelect) {
  const option = Array.from(metricSelect.options).find((opt) => opt.value === metric);
  return option ? option.textContent : metric;
}

function arrow(delta) {
  if (!Number.isFinite(delta) || Math.abs(delta) < 1e-15) return "";
  const cls = delta > 0 ? "metric-arrow" : "metric-arrow down";
  return `<svg class="${cls}" viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1.5 9.2 8.5H.8z"></path></svg>`;
}

export function renderMetricGrid(payload, { metricGrid, metricSelect }) {
  if (payload === null) {
    metricGrid.innerHTML = metricKeys.map((metric) => `
      <div class="metric-item" data-direction="neutral">
        <div class="metric-item-name">${metricLabel(metric, metricSelect)}</div>
        <div class="metric-pending">Pending recompute</div>
      </div>
    `).join("");
    return;
  }
  if (!payload.available) {
    metricGrid.textContent = payload.error || "Metric unavailable.";
    return;
  }
  const datasetLabel = payload.dataset_label || "Original";
  const caption = `<div class="metric-caption">${datasetLabel} data, change against the original model</div>`;
  metricGrid.innerHTML = caption + metricKeys.map((metric) => {
    const original = payload.metrics.original[metric];
    const edited = payload.metrics.edited[metric];
    const delta = edited - original;
    const label = metricLabel(metric, metricSelect);
    return `
      <div class="metric-item" data-direction="${metricDirection(metric, delta)}">
        <div class="metric-item-name" title="${label}">${label}</div>
        <div class="metric-item-delta">${arrow(delta)}<span>${fmtSigned(delta)}</span></div>
        <div class="metric-item-values"><span class="metric-item-value">${fmt(edited)}</span><span class="metric-item-origin">was ${fmt(original)}</span></div>
      </div>
    `;
  }).join("");
}
