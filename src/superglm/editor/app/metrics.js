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

function metricLabel(metric, metricSelect) {
  const option = Array.from(metricSelect.options).find((opt) => opt.value === metric);
  return option ? option.textContent : metric;
}

export function renderMetricGrid(payload, { metricGrid, metricSelect }) {
  if (payload === null) {
    metricGrid.innerHTML = metricKeys.map((metric) => `
      <div class="metric-item">
        <div class="metric-item-name">${metricLabel(metric, metricSelect)}</div>
        <div>Pending recompute</div>
      </div>
    `).join("");
    return;
  }
  if (!payload.available) {
    metricGrid.textContent = payload.error || "Metric unavailable.";
    return;
  }
  const datasetLabel = (payload.dataset_label || "Original").toLowerCase();
  metricGrid.innerHTML = metricKeys.map((metric) => {
    const original = payload.metrics.original[metric];
    const edited = payload.metrics.edited[metric];
    const delta = edited - original;
    return `
      <div class="metric-item">
        <div class="metric-item-name">${metricLabel(metric, metricSelect)}</div>
        <div class="metric-item-value">${fmt(edited)}</div>
        <div class="metric-item-delta">${datasetLabel} orig ${fmt(original)} · Δ ${fmtSigned(delta)}</div>
      </div>
    `;
  }).join('<div class="metric-divider" aria-hidden="true"></div>');
}
