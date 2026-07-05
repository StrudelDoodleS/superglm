import { requestJSON } from "./api.js";
import { escapeHTML, fmt } from "./format.js";

export async function refreshSummary(nodes) {
  const { summarySource, summaryStatus, summaryFrame } = nodes;
  const hasSummary = summaryFrame.innerHTML.trim().length > 0;
  summaryStatus.textContent = hasSummary ? "Updating summary..." : "Loading summary...";
  summaryFrame.setAttribute("aria-busy", "true");
  try {
    const payload = await requestJSON("/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: summarySource.value })
    });
    renderSummary(payload, nodes);
  } catch (error) {
    summaryStatus.textContent = error.message;
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
  }
}

export async function runOffsetRefit(nodes, refreshMetrics) {
  const { summarySource, summaryStatus, summaryFrame, refitOffset } = nodes;
  summaryStatus.textContent = "Refitting fixed offsets...";
  summaryFrame.setAttribute("aria-busy", "true");
  refitOffset.disabled = true;
  try {
    const payload = await requestJSON("/refit_offset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method: "auto" })
    });
    summarySource.value = "refit";
    renderSummary(payload, nodes);
    await refreshMetrics();
    return payload;
  } catch (error) {
    summaryStatus.textContent = error.message;
    return null;
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
    refitOffset.disabled = false;
  }
}

export async function runCollapseRefit(nodes, termName, refreshMetrics) {
  const { summarySource, summaryStatus, summaryFrame, collapseLevels } = nodes;
  summaryStatus.textContent = "Refitting collapsed levels...";
  summaryFrame.setAttribute("aria-busy", "true");
  if (collapseLevels) collapseLevels.disabled = true;
  try {
    const payload = await requestJSON("/collapse_levels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term: termName, method: "auto" })
    });
    summarySource.value = "selected";
    renderSummary(payload, nodes);
    await refreshMetrics();
    return payload;
  } catch (error) {
    summaryStatus.textContent = error.message;
    return null;
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
    if (collapseLevels) collapseLevels.disabled = false;
  }
}

export async function runUngroupRefit(nodes, termName, refreshMetrics) {
  const { summarySource, summaryStatus, summaryFrame, ungroupLevels } = nodes;
  summaryStatus.textContent = "Refitting ungrouped levels...";
  summaryFrame.setAttribute("aria-busy", "true");
  if (ungroupLevels) ungroupLevels.disabled = true;
  try {
    const payload = await requestJSON("/ungroup_levels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term: termName, method: "auto" })
    });
    summarySource.value = "selected";
    renderSummary(payload, nodes);
    await refreshMetrics();
    return payload;
  } catch (error) {
    summaryStatus.textContent = error.message;
    return null;
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
    if (ungroupLevels) ungroupLevels.disabled = false;
  }
}

export async function runUncollapseRefit(nodes, refreshMetrics) {
  const { summarySource, summaryStatus, summaryFrame, uncollapseLevels } = nodes;
  summaryStatus.textContent = "Restoring previous collapsed-level model...";
  summaryFrame.setAttribute("aria-busy", "true");
  if (uncollapseLevels) uncollapseLevels.disabled = true;
  try {
    const payload = await requestJSON("/uncollapse_levels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    summarySource.value = "selected";
    renderSummary(payload, nodes);
    await refreshMetrics();
    return payload;
  } catch (error) {
    summaryStatus.textContent = error.message;
    return null;
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
    if (uncollapseLevels) uncollapseLevels.disabled = false;
  }
}

function renderSummary(payload, { summaryStatus, summaryNote, summaryFrame }) {
  if (!payload.available) {
    summaryStatus.textContent = payload.label || "Summary";
    summaryNote.textContent = "";
    summaryFrame.innerHTML = `<div class="summary-empty">${escapeHTML(payload.error || "Summary unavailable.")}</div>`;
    return;
  }
  summaryStatus.textContent = payload.label || "Summary";
  summaryNote.textContent = payload.note || "";
  // Prefer the typed compact payload for the immediate panel. The raw HTML is
  // still included inside the disclosure for full notebook-style detail.
  summaryFrame.innerHTML = payload.compact ? renderCompactSummary(payload) : payload.html || "";
}

function renderCompactSummary(payload) {
  const compact = payload.compact || {};
  const model = compact.model || {};
  const rows = Array.isArray(compact.rows) ? compact.rows : [];
  const facts = [
    ["Family", model.family],
    ["Link", model.link],
    ["Method", model.method],
    ["Total EDF", model.effective_df],
    ["Deviance", model.deviance],
    ["AIC", model.aic],
    ["BIC", model.bic],
    ["Log lik", model.log_likelihood]
  ];
  return `
    <div class="compact-summary">
      <div class="summary-facts">
        ${facts.map(([label, value]) => renderSummaryFact(label, value)).join("")}
      </div>
      <table class="summary-table" aria-label="Compact coefficient summary">
        <thead>
          <tr>
            <th>Term</th>
            <th>EDF</th>
            <th>Estimate</th>
            <th>SE</th>
            <th>p</th>
            <th>Sig</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(renderSummaryRow).join("")}
        </tbody>
      </table>
      <details class="raw-summary">
        <summary>Full summary</summary>
        <div class="raw-summary-body">${payload.html || ""}</div>
      </details>
    </div>
  `;
}

function renderSummaryFact(label, value) {
  return `
    <div class="summary-fact">
      <span>${escapeHTML(label)}</span>
      <strong>${escapeHTML(formatSummaryValue(value))}</strong>
    </div>
  `;
}

function renderSummaryRow(row) {
  // SE cell color is data-driven from Python's significance class. The browser
  // never infers significance from display text.
  const sigClass = safeSigClass(row.sig_class);
  return `
    <tr class="summary-row ${sigClass}">
      <td class="summary-term">
        <span>${escapeHTML(row.name || "")}</span>
        ${row.kind === "spline" ? '<em>spline</em>' : ""}
      </td>
      ${renderNumberCell(row.edf)}
      ${renderNumberCell(row.coef)}
      <td class="se-cell ${sigClass}" title="${escapeHTML(sigTitle(row))}">
        ${escapeHTML(formatSE(row))}
      </td>
      <td>${escapeHTML(formatP(row.p_value))}</td>
      <td class="sig-code">${escapeHTML(row.sig_code || "")}</td>
    </tr>
  `;
}

function renderNumberCell(value) {
  return `<td class="summary-number" title="${escapeHTML(formatFullNumber(value))}">${escapeHTML(formatSummaryValue(value))}</td>`;
}

function formatSummaryValue(value) {
  const number = payloadNumber(value);
  if (number !== null) return formatCompactNumber(number);
  if (value === null || value === undefined || value === "") return "--";
  return String(value);
}

function formatSE(row) {
  const se = payloadNumber(row.se);
  if (se !== null) return formatCompactNumber(se);
  if (row.se_label) return row.se_label;
  return "--";
}

function formatCompactNumber(value) {
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001) return value.toPrecision(2);
  if (abs < 0.01) return value.toPrecision(2);
  if (abs < 0.1) return value.toPrecision(3);
  if (abs < 1) return value.toPrecision(3);
  if (abs < 10) return value.toPrecision(3);
  return fmt(value);
}

function formatFullNumber(value) {
  const number = payloadNumber(value);
  if (number === null) return "";
  return String(number);
}

function formatP(value) {
  const number = payloadNumber(value);
  if (number === null) return "--";
  if (number < 0.001) return "<0.001";
  return fmt(number);
}

function payloadNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function safeSigClass(value) {
  const allowed = new Set([
    "sig-strong",
    "sig-medium",
    "sig-standard",
    "sig-weak",
    "sig-none",
    "sig-unknown",
    "sig-qs"
  ]);
  return allowed.has(value) ? value : "sig-unknown";
}

function sigTitle(row) {
  if (row.quasi_separated) return "Quasi-separated level";
  if (payloadNumber(row.p_value) === null) return "Inference unavailable for this row";
  return `Colored by ${row.stat_label || "p"} p=${formatP(row.p_value)}`;
}
