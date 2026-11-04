// @ts-nocheck
import { escapeHTML, fmt, fmtSigned } from "./format.js";
import { metricDirection } from "./metrics.js";
import { cvSourceLine } from "./views/cv_tab.js";

const reportMetricKeys = [
  "deviance",
  "aic",
  "bic",
  "log_likelihood",
  "explained_deviance",
  "effective_df"
];

export function renderReport(payload, { reportTitle, reportStatus, reportFrame }, cvTab = null) {
  reportTitle.textContent = payload.title || "Report";
  if (payload.report === "cv" && cvTab) {
    reportStatus.textContent = cvSourceLine(payload);
    cvTab.render(payload);
    return;
  }
  reportStatus.textContent = payload.note || "";
  if (!payload.available) {
    reportFrame.innerHTML = `<div class="summary-empty">${escapeHTML(payload.note || "Report unavailable.")}</div>`;
    return;
  }
  const splitSection = renderSplitSection(payload);
  const cvSection = payload.report === "validation" ? renderCVSection(payload.cv_report) : "";
  const summarySection = payload.report === "final" ? renderFinalSummary(payload.summary) : "";
  const finalFitSection = payload.report === "final" ? renderFinalFit(payload.final_fit, payload.metric_labels) : "";
  reportFrame.innerHTML = `${splitSection}${cvSection}${summarySection}${finalFitSection}`;
}

// A SuperGLM report shows the fixed metric columns; a model whose report
// names other metrics (a SuperLSS scores by log score and CRPS) shows those.
function metricColumns(labels) {
  const named = Object.keys(labels);
  return named.length && !named.some((metric) => reportMetricKeys.includes(metric))
    ? named
    : reportMetricKeys;
}

function renderSplitSection(payload) {
  const labels = payload.metric_labels || {};
  const columns = metricColumns(labels);
  return `
    <section class="report-section">
      <h3>Split Metrics</h3>
      <table class="report-table" aria-label="Split metrics">
        <thead>
          <tr>
            <th>Split</th>
            ${columns.map((metric) => `<th>${escapeHTML(labels[metric] || metric)}</th>`).join("")}
          </tr>
        </thead>
        <tbody>
          ${(payload.splits || []).map((split) => renderSplitRow(split, columns)).join("")}
        </tbody>
      </table>
    </section>
  `;
}

// A split the model cannot score carries its reason instead of metrics.
function renderSplitRow(split, columns) {
  const metrics = split.metrics || {};
  const edited = metrics.edited || {};
  const delta = metrics.delta || {};
  const cells = split.available === false
    ? `<td class="report-note" colspan="${columns.length}">${escapeHTML(split.error || "")}</td>`
    : columns.map((metric) => renderMetricCell(metric, edited[metric], delta[metric])).join("");
  return `
    <tr>
      <td>
        <strong>${escapeHTML(split.label || split.name || "")}</strong>
        <span class="report-delta">${escapeHTML(String(split.n_obs || 0))} rows</span>
      </td>
      ${cells}
    </tr>
  `;
}

function renderMetricCell(metric, value, delta) {
  const direction = metricDirection(metric, Number(delta));
  // A score the model cannot compute (CRPS of a count model) arrives as null.
  const shownDelta = Number.isFinite(delta) ? fmtSigned(delta) : "--";
  return `
    <td>
      ${escapeHTML(Number.isFinite(value) ? fmt(value) : "--")}
      <span class="report-delta" data-direction="${direction}">Δ ${escapeHTML(shownDelta)}</span>
    </td>
  `;
}

function renderCVSection(cvReport) {
  if (cvReport === null || cvReport === undefined) return "";
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

function waitingNotIncluded(count) {
  return `${count} waiting ${count === 1 ? "change is" : "changes are"} not included.`;
}

function renderFinalFit(finalFit, labels = {}) {
  if (!finalFit) return "";
  if (!finalFit.available) {
    return `
      <section class="report-section final-fit">
        <h3>Final Fit</h3>
        <div class="report-note">${escapeHTML(finalFit.note || "")}</div>
      </section>
    `;
  }
  const model = finalFit.summary?.model || {};
  const notes = [
    `Refitted on ${Number(finalFit.n_rows).toLocaleString("en-US")} ${finalFit.splits.join(" and ")} rows; the test split stays held out.`,
    finalFit.carried.length ? `Hand edits put back: ${finalFit.carried.join(", ")}.` : "",
    finalFit.pending ? waitingNotIncluded(finalFit.pending) : "",
    finalFit.stale ? "The model has changed since; run Final fit again before exporting." : ""
  ].filter(Boolean);
  return `
    <section class="report-section final-fit">
      <h3>Final Fit</h3>
      ${notes.map((note) => `<div class="report-note">${escapeHTML(note)}</div>`).join("")}
      <table class="report-table" aria-label="Final fit">
        <tbody>
          <tr><th>Rows</th><td>${escapeHTML(formatValue(finalFit.n_rows))}</td></tr>
          ${finalFit.summary ? summaryRows(model) : ""}
          ${testScoreRows(finalFit.test_scores, labels)}
        </tbody>
      </table>
    </section>
  `;
}

// The Final fit's scores on the test split it held out, skipping a score the model has no way to compute.
function testScoreRows(scores, labels) {
  return Object.entries(scores || {})
    .filter(([, value]) => Number.isFinite(value))
    .map(([metric, value]) => `
          <tr><th>Test ${escapeHTML(labels[metric] || metric)}</th><td>${escapeHTML(fmt(value))}</td></tr>`)
    .join("");
}

function summaryRows(model) {
  return `
          <tr><th>Method</th><td>${escapeHTML(formatValue(model.method))}</td></tr>
          <tr><th>Total EDF</th><td>${escapeHTML(formatValue(model.effective_df))}</td></tr>
          <tr><th>Deviance</th><td>${escapeHTML(formatValue(model.deviance))}</td></tr>
          <tr><th>AIC</th><td>${escapeHTML(formatValue(model.aic))}</td></tr>
          <tr><th>BIC</th><td>${escapeHTML(formatValue(model.bic))}</td></tr>`;
}

function formatValue(value) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") return fmt(value);
  return String(value);
}
