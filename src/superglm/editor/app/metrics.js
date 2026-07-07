import { requestJSON } from "./api.js";
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

let metricPayload = null;

export async function refreshMetrics({ metricGrid, metricSelect }) {
  if (metricPayload === null && !metricGrid.querySelector(".metric-item")) {
    renderMetricGrid(null, { metricGrid, metricSelect });
  }
  metricGrid.setAttribute("aria-busy", "true");
  try {
    metricPayload = await requestJSON("/metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        metric: "deviance",
        source: "in_force"
      })
    });
    renderMetricGrid(metricPayload, { metricGrid, metricSelect });
  } catch (error) {
    if (metricPayload === null) {
      metricGrid.textContent = error.message;
    } else {
      metricGrid.title = error.message;
    }
  } finally {
    metricGrid.setAttribute("aria-busy", "false");
  }
}

function metricLabel(metric, metricSelect) {
  const option = Array.from(metricSelect.options).find((opt) => opt.value === metric);
  return option ? option.textContent : metric;
}

function renderMetricGrid(payload, { metricGrid, metricSelect }) {
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
