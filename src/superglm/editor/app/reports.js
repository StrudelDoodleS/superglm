import { requestJSON } from "./api.js";
import { escapeHTML, fmt, fmtSigned } from "./format.js";

const reportMetricKeys = [
  "deviance",
  "aic",
  "bic",
  "log_likelihood",
  "explained_deviance",
  "effective_df"
];

export async function refreshReport({ report, reportTitle, reportStatus, reportFrame }) {
  reportStatus.textContent = "Loading report...";
  reportFrame.setAttribute("aria-busy", "true");
  try {
    const payload = await requestJSON("/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ report })
    });
    renderReport(payload, { reportTitle, reportStatus, reportFrame });
  } catch (error) {
    reportStatus.textContent = error.message;
  } finally {
    reportFrame.setAttribute("aria-busy", "false");
  }
}

function renderReport(payload, { reportTitle, reportStatus, reportFrame }) {
  reportTitle.textContent = payload.title || "Report";
  reportStatus.textContent = payload.note || "";
  if (!payload.available) {
    reportFrame.innerHTML = `<div class="summary-empty">${escapeHTML(payload.note || "Report unavailable.")}</div>`;
    return;
  }
  const splitSection = renderSplitSection(payload);
  const cvSection = payload.report === "validation" ? renderCVSection(payload.cv_report) : "";
  const summarySection = payload.report === "final" ? renderFinalSummary(payload.summary) : "";
  reportFrame.innerHTML = `${splitSection}${cvSection}${summarySection}`;
}

function renderSplitSection(payload) {
  const labels = payload.metric_labels || {};
  return `
    <section class="report-section">
      <h3>Split Metrics</h3>
      <table class="report-table" aria-label="Split metrics">
        <thead>
          <tr>
            <th>Split</th>
            ${reportMetricKeys.map((metric) => `<th>${escapeHTML(labels[metric] || metric)}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${(payload.splits || []).map((split) => renderSplitRow(split)).join("")}
        </tbody>
      </table>
    </section>
  `;
}

function renderSplitRow(split) {
  const metrics = split.metrics || {};
  const edited = metrics.edited || {};
  const delta = metrics.delta || {};
  return `
    <tr>
      <td>
        <strong>${escapeHTML(split.label || split.name || "")}</strong>
        <span class="report-delta">${escapeHTML(String(split.n_obs || 0))} rows</span>
      </td>
      ${reportMetricKeys.map((metric) => renderMetricCell(edited[metric], delta[metric])).join("")}
    </tr>
  `;
}

function renderMetricCell(value, delta) {
  return `
    <td>
      ${escapeHTML(fmt(value))}
      <span class="report-delta">Δ ${escapeHTML(fmtSigned(delta))}</span>
    </td>
  `;
}

function renderCVSection(cvReport) {
  if (cvReport === null || cvReport === undefined) {
    return `
      <section class="report-section cv-report">
        <h3>CV Report</h3>
        <div class="report-note">No CV report supplied.</div>
      </section>
    `;
  }
  return `
    <section class="report-section cv-report">
      <h3>CV Report</h3>
      ${renderCVObject(cvReport)}
    </section>
  `;
}

function renderCVObject(value) {
  if (Array.isArray(value)) return renderGenericTable(value, "CV rows");
  if (value && typeof value === "object") {
    const tableSections = [
      ["summary", "CV Summary"],
      ["split_loss", "Split Loss"],
      ["rows", "Fold Loss"]
    ];
    const tableKeys = new Set(tableSections.map(([key]) => key));
    const metadata = Object.fromEntries(
      Object.entries(value).filter(([key]) => !tableKeys.has(key))
    );
    const header = Object.keys(metadata).length ? renderKeyValueList(metadata) : "";
    const tables = tableSections
      .filter(([key]) => Array.isArray(value[key]))
      .map(([key, title]) => renderNamedTable(title, value[key]))
      .join("");
    if (tables) return `${header}${tables}`;
    return renderKeyValueList(value);
  }
  return `<pre>${escapeHTML(String(value))}</pre>`;
}

function renderNamedTable(title, rows) {
  return `
    <div class="report-subsection">
      <h4>${escapeHTML(title)}</h4>
      ${renderGenericTable(rows, title)}
    </div>
  `;
}

function renderGenericTable(rows, label) {
  if (!rows.length) return '<div class="report-note">No rows supplied.</div>';
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row || {}))));
  return `
    <table class="report-table" aria-label="${escapeHTML(label)}">
      <thead><tr>${columns.map((column) => `<th>${escapeHTML(column)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${columns.map((column) => `<td>${escapeHTML(formatValue(row[column]))}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderKeyValueList(value) {
  return `<pre>${escapeHTML(JSON.stringify(value, null, 2))}</pre>`;
}

function renderFinalSummary(summary) {
  const compact = summary && summary.compact ? summary.compact : null;
  if (!compact) return "";
  const model = compact.model || {};
  return `
    <section class="report-section">
      <h3>Final Model Summary</h3>
      <table class="report-table" aria-label="Final model summary">
        <tbody>
          <tr><th>Family</th><td>${escapeHTML(formatValue(model.family))}</td></tr>
          <tr><th>Link</th><td>${escapeHTML(formatValue(model.link))}</td></tr>
          <tr><th>Method</th><td>${escapeHTML(formatValue(model.method))}</td></tr>
          <tr><th>Total EDF</th><td>${escapeHTML(formatValue(model.effective_df))}</td></tr>
          <tr><th>Deviance</th><td>${escapeHTML(formatValue(model.deviance))}</td></tr>
          <tr><th>AIC</th><td>${escapeHTML(formatValue(model.aic))}</td></tr>
          <tr><th>BIC</th><td>${escapeHTML(formatValue(model.bic))}</td></tr>
        </tbody>
      </table>
    </section>
  `;
}

function formatValue(value) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") return fmt(value);
  return String(value);
}
