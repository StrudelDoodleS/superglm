import { requestJSON } from "./api.js";
import { escapeHTML, fmt, fmtSignificant } from "./format.js";
import { SHAPE_NAMES } from "./shapes.js";
import {
  DEFAULT_SUMMARY_VIEW,
  highlightMatches,
  highlightRowName,
  summaryCountText,
  summaryViewModel
} from "./views/summary_view.js";

/** @typedef {import('./api/contracts.js').EmptyStructuralRequest} EmptyStructuralRequest */
/** @typedef {import('./api/contracts.js').SetReferenceRequest} SetReferenceRequest */
/** @typedef {import('./api/contracts.js').ShapeRangeRequest} ShapeRangeRequest */
/** @typedef {import('./api/contracts.js').StageRequest} StageRequest */

const PROFILE_ESTIMATE_LABELS = { p: "p_hat", theta: "theta_hat" };
const summaryMarkupByFrame = new WeakMap();
// The payload each frame shows, so a change of view can redraw it unfetched.
const summaryPayloadByFrame = new WeakMap();

export async function refreshSummary(nodes, { request = requestJSON } = {}) {
  const { summarySource, summaryStatus, summaryFrame } = nodes;
  const hasSummary = summaryFrame.innerHTML.trim().length > 0;
  summaryStatus.textContent = hasSummary ? "Updating summary..." : "Loading summary...";
  summaryFrame.setAttribute("aria-busy", "true");
  try {
    const payload = await request("/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: summarySource.value,
        level_display: requestedLevelDisplay(nodes)
      })
    });
    renderSummary(payload, nodes);
  } catch (error) {
    summaryStatus.textContent = error.message;
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
  }
}

export async function runOffsetRefit(
  nodes,
  refreshMetrics,
  { request = requestJSON } = {}
) {
  const { summarySource, summaryStatus, summaryFrame, refitOffset } = nodes;
  const levelDisplay = requestedLevelDisplay(nodes);
  summaryStatus.textContent = "Refitting fixed offsets...";
  summaryFrame.setAttribute("aria-busy", "true");
  refitOffset.disabled = true;
  try {
    const payload = await request("/refit_offset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        method: "auto",
        level_display: levelDisplay
      })
    });
    const payloadLevelDisplay = responseLevelDisplay(payload, levelDisplay);
    if (payloadLevelDisplay === requestedLevelDisplay(nodes)) {
      summarySource.value = "refit";
      renderSummary(payload, nodes);
    }
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

export async function runDistributionProfile(
  nodes,
  parameter,
  acceptProfile = async () => {},
  { request = requestJSON, pause = sleep } = {}
) {
  const { summaryStatus, summaryFrame, reprofileTweedie, reprofileNb2, profileRun } = nodes;
  const button = parameter === "tweedie_p" ? reprofileTweedie : reprofileNb2;
  const levelDisplay = requestedLevelDisplay(nodes);
  openProfileDialog(nodes);
  summaryStatus.textContent = parameter === "tweedie_p"
    ? "Re-profiling Tweedie p..."
    : "Re-estimating NB2 theta...";
  summaryFrame.setAttribute("aria-busy", "true");
  if (button) button.disabled = true;
  if (profileRun) profileRun.disabled = true;
  renderProfileTrace({
    status: "running",
    parameter,
    trace: [],
    options: profileOptionsPayload(nodes)
  }, nodes);
  try {
    const started = await request("/profile_distribution/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parameter,
        level_display: levelDisplay,
        ...profileOptionsPayload(nodes)
      })
    });
    let status = started;
    renderProfileTrace(status, nodes);
    while (status.status === "running") {
      await pause(250);
      status = await request(`/profile_distribution/status/${encodeURIComponent(started.job_id)}`);
      renderProfileTrace(status, nodes);
    }
    if (status.status === "error") {
      throw new Error(status.error || "Profile search failed.");
    }
    const payload = status.result || {};
    renderProfileTrace(status, nodes);
    if (responseLevelDisplay(payload, levelDisplay) === requestedLevelDisplay(nodes)) {
      renderSummary(payload, nodes);
    }
    await acceptProfile(payload);
    return payload;
  } catch (error) {
    summaryStatus.textContent = error.message;
    return null;
  } finally {
    summaryFrame.setAttribute("aria-busy", "false");
    if (button) button.disabled = false;
    if (profileRun) profileRun.disabled = false;
  }
}

export function showDistributionProfileDialog(nodes, parameter) {
  const label = parameter === "nb2_theta" ? "NB2 theta" : "Tweedie p";
  if (nodes.profileDialogTitle) nodes.profileDialogTitle.textContent = `Profile ${label}`;
  if (nodes.profileDialogDescription) {
    nodes.profileDialogDescription.textContent = "Candidate parameter fits and loss trace.";
  }
  if (nodes.profileRun) nodes.profileRun.textContent = `Run ${label}`;
  if (nodes.profileDialog) nodes.profileDialog.dataset.parameter = parameter;
  openProfileDialog(nodes);
}

function openProfileDialog(nodes) {
  const dialog = nodes.profileDialog;
  if (!dialog || dialog.open) return;
  if (typeof dialog.showModal === "function") {
    dialog.showModal();
  } else {
    dialog.setAttribute("open", "");
  }
}

function profileOptionsPayload(nodes) {
  const xatol = Number(nodes.profileTolerance ? nodes.profileTolerance.value : 0.001);
  return {
    xatol: Number.isFinite(xatol) && xatol > 0 ? xatol : 0.001
  };
}

function requestedLevelDisplay(nodes) {
  return nodes.summaryLevelDisplay === "grouped" ? "grouped" : "expanded";
}

function responseLevelDisplay(payload, fallback) {
  return payload.level_display === "expanded" || payload.level_display === "grouped"
    ? payload.level_display
    : fallback;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/**
 * One structural change, staged: Python builds it and keeps it waiting,
 * drawn on the chart, until Refit applies every waiting change in one fit.
 * ``name`` is what the busy overlay and an alert call it.
 * @param {StageRequest['operation']} operation @param {string} term
 * @param {Record<string, unknown>} params @param {string} name
 * @returns {{name:string, path:string, payload:StageRequest}}
 */
function stageTransition(operation, term, params, name) {
  return { name, path: "/stage", payload: { operation, term, params } };
}

/** @param {string} term @param {readonly string[]} levels the selected levels, by label */
export function stageCollapse(term, levels) {
  return stageTransition("collapse", term, { levels: [...levels] }, "collapse levels");
}

/** @param {string} term @param {readonly string[]} levels the selected levels, by label */
export function stageUngroup(term, levels) {
  return stageTransition("ungroup", term, { levels: [...levels] }, "ungroup levels");
}

/** @param {string} term @param {string} level a displayed level, which may be a group label */
export function stageReference(term, level) {
  return stageTransition("set_reference", term, { level }, "set reference");
}

/**
 * Named for its shape. ``join`` is how the range meets the free curve:
 * "tangent" (the default) or "kink" (Corner).
 * @param {string} term @param {number|string} lo @param {number|string} hi @param {number} degree
 * @param {"tangent"|"kink"} [join]
 */
export function stageShapeRange(term, lo, hi, degree, join = "tangent") {
  return stageTransition(
    "shape", term, { lo, hi, degree, join }, `make a ${SHAPE_NAMES[degree]} range`
  );
}

/**
 * Refit: every waiting change in one fit, and one step on the timeline.
 * @param {number} count how many changes wait, for the busy overlay
 * @returns {{name:string, path:string, payload:EmptyStructuralRequest}}
 */
export function refitPendingTransition(count) {
  return {
    name: `refit ${count} waiting ${count === 1 ? "change" : "changes"}`,
    path: "/refit_pending",
    payload: {}
  };
}

/**
 * The same change refitted at once, through its operation's own route, as
 * Settings' "Refit after every structural change" asks. Python stages it and
 * refits every waiting change in one fit: one step, which one Undo takes
 * back, refused with the operation's own sentences. Collapse and ungroup act
 * on the selection Python holds, the one their levels were read from.
 * @param {{name:string, payload:StageRequest}} staged a descriptor from stageCollapse,
 *   stageUngroup, stageReference or stageShapeRange
 * @returns {{name:string, path:string,
 *   payload:{term:string, method:string}|SetReferenceRequest|ShapeRangeRequest}}
 */
export function refitAtOnceTransition({ name, payload: { operation, term, params } }) {
  const method = "auto";
  switch (operation) {
    case "collapse":
      return { name, path: "/collapse_levels", payload: { term, method } };
    case "ungroup":
      return { name, path: "/ungroup_levels", payload: { term, method } };
    case "set_reference":
      return { name, path: "/set_reference", payload: { term, level: params.level, method } };
    default: {
      const { lo, hi, degree, join } = params;
      return { name, path: "/shape_range", payload: { term, lo, hi, degree, join, method } };
    }
  }
}

/** @returns {{name:string, path:string, payload:EmptyStructuralRequest}} */
export function revertTransition() {
  return {
    name: "revert to original model",
    path: "/revert_to_original",
    payload: {}
  };
}

export function renderSummary(payload, nodes) {
  const { summaryStatus, summaryNote, summaryFrame } = nodes;
  summaryPayloadByFrame.set(summaryFrame, payload);
  updateDistributionProfileActions(payload, nodes);
  if (!payload.available) {
    summaryStatus.textContent = payload.label || "Summary";
    summaryNote.textContent = "";
    updateSummaryMarkup(
      summaryFrame,
      `<div class="summary-empty">${escapeHTML(payload.error || "Summary unavailable.")}</div>`
    );
    renderSearchCount(nodes, "");
    renderSummaryHeader(payload, nodes, "");
    return;
  }
  summaryStatus.textContent = payload.label || "Summary";
  summaryNote.textContent = payload.note || "";
  // Prefer the typed compact payload for the immediate panel. The raw HTML is
  // still included inside the disclosure for full notebook-style detail. The
  // inspector's search is part of the markup, so every render reapplies it.
  const view = summaryViewOf(nodes);
  const viewModel = payload.compact ? compactViewModel(payload.compact, view) : null;
  const written = updateSummaryMarkup(
    summaryFrame,
    viewModel ? renderCompactSummary(payload, viewModel, view.query) : payload.html || ""
  );
  renderSearchCount(nodes, viewModel ? summaryCountText(viewModel, view.query) : "");
  renderSummaryHeader(payload, nodes, view.query);
  if (written && !view.query.trim()) scrollToCurrentSection(summaryFrame);
}

/**
 * Redraw the summary on show for the inspector's current view: search,
 * filter and open sections. With the compact table in the DOM only its body is
 * rewritten, so an open "Full summary" keeps its frame; otherwise the last
 * payload is rendered again. `follow` scrolls the chart's term into view.
 */
export function applySummaryView(nodes, { follow = false } = {}) {
  const { summaryFrame } = nodes;
  const payload = summaryPayloadByFrame.get(summaryFrame);
  const shown = summaryMarkupByFrame.get(summaryFrame);
  // Only a frame still holding its last render is redrawn: one emptied while
  // the other level display loads waits for that payload.
  if (!payload || !shown || shown.firstElementChild !== summaryFrame.firstElementChild) return;
  const body = payload.available && payload.compact && typeof summaryFrame.querySelector === "function"
    ? summaryFrame.querySelector(".summary-table tbody")
    : null;
  if (!body) {
    renderSummary(payload, nodes);
    return;
  }
  const view = summaryViewOf(nodes);
  const viewModel = compactViewModel(payload.compact, view);
  body.innerHTML = renderSummaryBody(
    compactRows(payload.compact),
    viewModel,
    payload.compact.has_level_groups === true,
    view.query
  );
  renderSearchCount(nodes, summaryCountText(viewModel, view.query));
  renderSummaryHeader(payload, nodes, view.query);
  // The frame now holds what a full render for this view writes, so a later
  // render of the same payload and view leaves the DOM alone.
  summaryMarkupByFrame.set(summaryFrame, {
    markup: renderCompactSummary(payload, viewModel, view.query),
    firstElementChild: summaryFrame.firstElementChild
  });
  if (follow) scrollToCurrentSection(summaryFrame);
}

// Follow the chart: bring its term's line into view in the summary frame.
function scrollToCurrentSection(summaryFrame) {
  if (typeof summaryFrame.querySelector !== "function") return;
  const line = summaryFrame.querySelector('tr.summary-section[data-current="true"]:not([hidden])');
  if (line) line.scrollIntoView({ block: "nearest" });
}

// Family, link and method as chips and four figures as tiles, beside Refit
// offsets. The header steps aside while a search narrows the table.
function renderSummaryHeader(payload, nodes, query) {
  const { summaryHeader, summaryModelChips, summaryTiles } = nodes;
  const model = payload.available && payload.compact ? payload.compact.model || {} : null;
  if (summaryModelChips) updateSummaryMarkup(summaryModelChips, model ? renderModelChips(model) : "");
  if (summaryTiles) updateSummaryMarkup(summaryTiles, model ? renderModelTiles(model) : "");
  if (summaryHeader) summaryHeader.hidden = query.trim() !== "";
}

function renderModelChips(model) {
  const link = model.link ? `${model.link} link` : "";
  return [model.family, link, model.method]
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map((value) => `<span class="summary-chip">${escapeHTML(value)}</span>`)
    .join("");
}

function renderModelTiles(model) {
  return [
    ["Deviance", model.deviance],
    ["AIC", model.aic],
    ["BIC", model.bic],
    ["Total EDF", model.effective_df]
  ].map(([label, value]) => `<div class="summary-tile"><span>${escapeHTML(label)}</span><strong title="${escapeHTML(formatFullNumber(value))}">${escapeHTML(formatSummaryValue(value))}</strong></div>`).join("");
}

function summaryViewOf(nodes) {
  return typeof nodes.summaryView === "function" ? nodes.summaryView() : DEFAULT_SUMMARY_VIEW;
}

function compactRows(compact) {
  return Array.isArray(compact.rows) ? compact.rows : [];
}

function compactViewModel(compact, view) {
  return summaryViewModel(compactRows(compact), view);
}

function renderSearchCount(nodes, text) {
  if (nodes.summarySearchCount) nodes.summarySearchCount.textContent = text;
}

// Whether the markup was written; unchanged markup leaves the DOM alone.
function updateSummaryMarkup(summaryFrame, markup) {
  const cached = summaryMarkupByFrame.get(summaryFrame);
  if (
    cached?.markup === markup &&
    cached.firstElementChild === summaryFrame.firstElementChild
  ) return false;
  summaryFrame.innerHTML = markup;
  summaryMarkupByFrame.set(summaryFrame, {
    markup,
    firstElementChild: summaryFrame.firstElementChild
  });
  return true;
}

export function updateDistributionProfileActions(payload, nodes) {
  const { reprofileTweedie, reprofileNb2 } = nodes;
  const family = String(payload && payload.compact && payload.compact.model
    ? payload.compact.model.family || ""
    : "");
  const canProfileTweedie = payload.available && family === "Tweedie";
  const canProfileNb2 = payload.available && family === "Neg. Binomial";
  if (reprofileTweedie) {
    reprofileTweedie.hidden = !canProfileTweedie;
  }
  if (reprofileNb2) {
    reprofileNb2.hidden = !canProfileNb2;
  }
  if (nodes.profileOptions) nodes.profileOptions.hidden = !(canProfileTweedie || canProfileNb2);
}

function renderProfileTrace(job, nodes) {
  const {
    profileProgress,
    profileTraceStatus,
    profileTraceLegend,
    profileTracePlot,
    profileTraceTable
  } = nodes;
  if (!profileProgress || !profileTracePlot || !profileTraceTable) return;
  const trace = Array.isArray(job.trace) ? job.trace : [];
  const estimate = profileEstimate(job);
  profileProgress.hidden = false;
  profileProgress.classList.toggle("profile-running", job.status === "running");
  profileProgress.classList.toggle("profile-finalizing", job.status === "running" && isPostSearchPhase(job.phase));
  const label = job.parameter === "nb2_theta" ? "theta" : "p";
  if (profileTraceStatus) {
    profileTraceStatus.textContent = profileStatusLabel(job, trace.length);
  }
  if (profileTraceLegend) {
    profileTraceLegend.innerHTML = profileTraceLegendHTML(trace, estimate, label);
  }
  profileTracePlot.innerHTML = profileTraceSVG(trace, estimate);
  profileTraceTable.innerHTML = profileTraceRows(trace, label);
}

function profileStatusLabel(job, traceCount) {
  const estimate = profileEstimate(job);
  const estimateText = estimate ? profileEstimateShortText(estimate) : "";
  if (job.status === "complete") {
    return estimateText ? `done · ${estimateText} · ${traceCount} evals` : `done · ${traceCount} evals`;
  }
  if (job.status === "error") return "error";
  if (job.phase === "best_found") {
    return estimateText
      ? `best ${estimateText} · ${traceCount} evals`
      : `best parameter found · ${traceCount} evals`;
  }
  if (job.phase === "final_refit") {
    return estimateText
      ? `final refit · ${estimateText}`
      : `final refit · ${traceCount} evals`;
  }
  if (job.phase === "finalizing") return `updating summary · ${traceCount} evals`;
  if (traceCount > 0) return `profiling · ${traceCount} evals`;
  return "starting";
}

function isPostSearchPhase(phase) {
  return ["best_found", "final_refit", "finalizing"].includes(phase);
}

function profileEstimate(job) {
  if (job && job.profile_estimate) return job.profile_estimate;
  if (job && job.result && job.result.profile_estimate) return job.result.profile_estimate;
  return null;
}

function profileEstimateShortText(estimate) {
  if (!estimate) return "";
  const label = estimate.label || PROFILE_ESTIMATE_LABELS[estimate.parameter] || estimate.parameter || "estimate";
  const value = formatProfileNumber(estimate.value);
  return value ? `${label} ${value}` : String(label);
}

function profileTraceSVG(trace, estimate) {
  const objective = profileObjectiveRows(trace, estimate);
  if (objective.length) return profileObjectiveSVG(objective, estimate);
  if (!trace.length) {
    return '<text x="160" y="64" text-anchor="middle" class="profile-trace-label" font-size="12">waiting for first evaluation</text>';
  }
  return '<text x="160" y="64" text-anchor="middle" class="profile-trace-label" font-size="12">no profile loss values yet</text>';
}

function profileObjectiveRows(trace, estimate) {
  const parameter = estimate && estimate.parameter === "theta" ? "theta" : "p";
  return trace
    .map((row, index) => ({
      index,
      parameter,
      value: Number(row[parameter]),
      nll: outerProfileObjective(row),
      source: row.source || ""
    }))
    .filter((row) => Number.isFinite(row.value) && Number.isFinite(row.nll))
    .sort((a, b) => a.value - b.value || a.index - b.index);
}

function profileObjectiveSVG(values, estimate) {
  const margin = { left: 28, right: 10, top: 12, bottom: 22 };
  const width = 320 - margin.left - margin.right;
  const height = 120 - margin.top - margin.bottom;
  const xMin = Math.min(...values.map((row) => row.value));
  const xMaxRaw = Math.max(...values.map((row) => row.value));
  const yMin = Math.min(...values.map((row) => row.nll));
  const yMax = Math.max(...values.map((row) => row.nll));
  const xPad = Math.max((xMaxRaw - xMin) * 0.06, Math.abs(xMaxRaw) * 0.0005, 1e-6);
  const yPad = Math.max((yMax - yMin) * 0.08, Math.abs(yMax) * 0.002, 1e-9);
  const left = xMin - xPad;
  const right = xMaxRaw + xPad;
  const low = yMin - yPad;
  const high = yMax + yPad;
  const x = (value) => margin.left + width * ((value - left) / (right - left || 1));
  const y = (nll) => margin.top + height * (1 - (nll - low) / (high - low || 1));
  const axisY = margin.top + height;
  const points = values.map((row) => `${x(row.value).toFixed(2)},${y(row.nll).toFixed(2)}`);
  const best = values.reduce((acc, row, i) => (row.nll < acc.row.nll ? { row, i } : acc), {
    row: values[0],
    i: 0
  });
  const bestLabel = estimate ? profileEstimateShortText(estimate) : "best";
  const bestXNumber = x(best.row.value);
  const bestYNumber = y(best.row.nll);
  const bestX = bestXNumber.toFixed(2);
  const bestY = bestYNumber.toFixed(2);
  const bestLabelX = Math.min(Math.max(bestXNumber + 8, margin.left + 4), margin.left + width - 78);
  const bestLabelY = Math.max(bestYNumber - 32, margin.top + 10);
  const xLabel = values[0].parameter === "theta" ? "theta" : "p";
  return `
    <line class="profile-trace-grid" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + height}"></line>
    <line class="profile-trace-grid" x1="${margin.left}" y1="${margin.top + height}" x2="${margin.left + width}" y2="${margin.top + height}"></line>
    <text x="${margin.left}" y="10" class="profile-trace-label" font-size="10">profile NLL</text>
    <polyline class="profile-trace-line" points="${points.join(" ")}"></polyline>
    ${values.map((row) => `<circle class="profile-trace-dot" cx="${x(row.value).toFixed(2)}" cy="${y(row.nll).toFixed(2)}" r="3"></circle>`).join("")}
    <line class="profile-trace-best-line" x1="${bestX}" y1="${margin.top}" x2="${bestX}" y2="${axisY}"></line>
    <circle class="profile-trace-best" cx="${bestX}" cy="${bestY}" r="4"></circle>
    <text class="profile-trace-best-label" x="${bestLabelX.toFixed(2)}" y="${bestLabelY.toFixed(2)}">${escapeHTML(bestLabel)}</text>
    <text x="${margin.left}" y="112" class="profile-trace-label" font-size="10">${escapeHTML(xLabel)}</text>
    <text x="310" y="112" text-anchor="end" class="profile-trace-label" font-size="10">profile loss</text>
  `;
}

function profileTraceLegendHTML(trace, estimate, label) {
  const estimateBlock = estimate
    ? `<div class="profile-estimate">
        <strong>${escapeHTML(profileEstimateShortText(estimate))}</strong>
        <span>${escapeHTML(profileEstimateCIText(estimate))}</span>
        <em>outer profile objective minimum</em>
      </div>`
    : "";
  const rows = trace
    .map((row, index) => ({
      index,
      value: Number(row[label]),
      loss: Number(row.nll)
    }))
    .filter((row) => Number.isFinite(row.value) && Number.isFinite(row.loss))
    .slice(-10);
  if (!rows.length) return `${estimateBlock}<div class="profile-legend-empty">waiting</div>`;
  const best = rows.reduce((acc, row) => (row.loss < acc.loss ? row : acc), rows[0]);
  return `${estimateBlock}${rows.map((row, i) => {
    const isBest = Math.abs(row.value - best.value) < 1e-9;
    return profileLegendItem({
      color: profileCurveColor(i),
      label: `${label} ${formatProfileNumber(row.value)}`,
      detail: formatProfileNumber(row.loss),
      isBest
    });
  }).join("")}`;
}

function profileEstimateCIText(estimate) {
  const low = formatProfileNumber(estimate.ci_low);
  const high = formatProfileNumber(estimate.ci_high);
  if (!low || !high) {
    return estimate.ci_status === "not computed" ? "CI not computed" : "CI pending";
  }
  return `CI [${low}, ${high}]${profileCensoredSuffix(estimate.ci_status)}`;
}

// A censored side is where the search stopped, not a likelihood-ratio crossing;
// a caution says the interval is not about the estimate as it stands.
const MARKED_CI_STATUSES = new Set(["censored", "caution", "censored with caution"]);

function profileCensoredSuffix(status) {
  return MARKED_CI_STATUSES.has(status) ? ` ${status}` : "";
}

function profileLegendItem({ color, label, detail, isBest }) {
  return `
    <div class="profile-legend-item${isBest ? " profile-legend-best" : ""}">
      <span class="profile-legend-swatch" style="background:${escapeHTML(color)}"></span>
      <strong>${escapeHTML(label)}</strong>
      <em>${escapeHTML(detail || "")}</em>
    </div>
  `;
}

function outerProfileObjective(row) {
  return Number(row.nll);
}

// The trace palette lives in tokens.css as --trace-0 to --trace-9, one
// value per theme; the SVG names its colour and the stylesheet supplies it.
function profileCurveColor(index) {
  return `var(--trace-${index % 10})`;
}

function profileTraceRows(trace, label) {
  const rows = trace.slice(-4).reverse();
  if (!rows.length) return '<div class="profile-trace-row"><span></span><span>waiting</span><span></span><span></span></div>';
  return rows.map((row) => {
    const param = row[label] !== undefined ? `${label} ${formatProfileNumber(row[label])}` : "";
    return `
      <div class="profile-trace-row">
        <span>${escapeHTML(String(row.step ?? ""))}</span>
        <strong>${escapeHTML(param)}</strong>
        <span>${escapeHTML(formatProfileNumber(row.nll))}</span>
        <span>${escapeHTML(row.source || "")}</span>
      </div>
    `;
  }).join("");
}

function formatProfileNumber(value) {
  if (value === null || value === undefined || value === "") return "";
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (Math.abs(number) >= 100) return fmt(number);
  if (Math.abs(number) >= 1) return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  return number.toPrecision(4);
}

function renderCompactSummary(payload, viewModel, query) {
  const compact = payload.compact || {};
  const model = compact.model || {};
  const rows = compactRows(compact);
  const hasLevelGroups = compact.has_level_groups === true;
  // Family, link, method and the four headline figures are the header's
  // (renderSummaryHeader); a profiled distribution parameter stays here.
  const facts = [];
  if (model.tweedie_p !== null && model.tweedie_p !== undefined) {
    facts.push(["Tweedie p", model.tweedie_p]);
    const ci = Array.isArray(model.tweedie_p_ci) ? model.tweedie_p_ci : null;
    const ciText = ci && ci.length >= 2
      ? `[${formatProfileNumber(ci[0])}, ${formatProfileNumber(ci[1])}]${profileCensoredSuffix(model.tweedie_p_ci_status)}`
      : model.tweedie_p_ci_status || "not computed";
    facts.push(["Tweedie p CI", ciText]);
  }
  if (model.nb_theta !== null && model.nb_theta !== undefined) facts.push(["NB2 theta", model.nb_theta]);
  return `
    <div class="compact-summary">
      ${facts.length ? `<div class="summary-facts">
        ${facts.map(([label, value]) => renderSummaryFact(label, value)).join("")}
      </div>` : ""}
      <table class="summary-table${hasLevelGroups ? " has-level-groups" : ""}" aria-label="Compact coefficient summary">
        <thead>
          <tr>
            <th class="summary-term">Term</th>
            ${hasLevelGroups ? '<th class="summary-level-group">Level group</th>' : ""}
            <th class="summary-edf">EDF</th>
            <th class="summary-estimate">Estimate</th>
            <th class="summary-se">SE</th>
            <th class="summary-p">p</th>
            <th class="sig-code">Sig</th>
            <th class="advisory-code" title="Low credibility or outsized standard error">LC</th>
          </tr>
        </thead>
        <tbody>
          ${renderSummaryBody(rows, viewModel, hasLevelGroups, query)}
        </tbody>
      </table>
      ${renderLevelGroupLegends(compact)}
      <details class="raw-summary">
        <summary>Full summary</summary>
        <div class="raw-summary-body">${renderRawSummaryFrame(payload.html)}</div>
      </details>
    </div>
  `;
}

function renderRawSummaryFrame(html) {
  if (!html) {
    return '<div class="summary-empty">Full summary unavailable.</div>';
  }
  return `
    <iframe
      class="raw-summary-frame"
      title="Full model summary"
      sandbox=""
      referrerpolicy="no-referrer"
      srcdoc="${escapeHTML(html)}"
    ></iframe>
  `;
}

// One header row per term, then its rows. Rows and headers outside the
// search stay in the markup, hidden, each tagged with its term.
function renderSummaryBody(rows, viewModel, hasLevelGroups, query) {
  const columnCount = hasLevelGroups ? 8 : 7;
  const empty = viewModel.empty
    ? `<tr class="summary-empty-row"><td colspan="${columnCount}">No terms match.</td></tr>`
    : "";
  return empty + viewModel.sections.map((section) => {
    const groupRow = section.header ? renderSectionHeader(section, columnCount, query) : "";
    const sectionRows = section.rows.map((entry) => renderSummaryRow(
      rows[entry.index],
      hasLevelGroups,
      section.term,
      query,
      entry.hidden
    ));
    return groupRow + sectionRows.join("");
  }).join("");
}

// A term's line: what it is and how it fits, folded or open. Its button opens
// or closes the rows under it.
function renderSectionHeader(section, columnCount, query) {
  const term = escapeHTML(section.term);
  const kind = section.kind
    ? `<span class="summary-section-kind">${escapeHTML(section.kind)}</span>`
    : "";
  const waiting = section.waiting > 0
    ? `<span class="summary-waiting">${section.waiting} waiting</span>`
    : "";
  const edf = section.edf === null
    ? ""
    : `<span class="summary-section-edf">EDF ${escapeHTML(fmtSignificant(section.edf))}</span>`;
  return `<tr class="summary-group-row summary-section" data-term="${term}" data-current="${section.current}"${section.hidden ? " hidden" : ""}><td colspan="${columnCount}"><button type="button" class="summary-section-toggle" data-summary-section="${term}" aria-expanded="${section.open}"><svg class="summary-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m6 4 4 4-4 4"></path></svg><span class="summary-section-name">${highlightMatches(section.label, query)}</span>${kind}${waiting}<span class="summary-section-fill"></span>${edf}${renderPChip(section.chip)}</button></td></tr>`;
}

// The p-value of a term's whole-term test on its line. A categorical has
// none, so its line carries no chip.
function renderPChip(chip) {
  if (!chip) return "";
  const text = `${formatP(chip.p)}${chip.sigCode ? ` ${chip.sigCode}` : ""}`;
  return `<span class="summary-p-chip ${safeSigClass(chip.sigClass)}" title="p-value of the whole-term test">${escapeHTML(text)}</span>`;
}

function renderSummaryFact(label, value) {
  return `
    <div class="summary-fact">
      <span>${escapeHTML(label)}</span>
      <strong>${escapeHTML(formatSummaryValue(value))}</strong>
    </div>
  `;
}

// Row kinds that name a whole-term group test rather than a coefficient. The
// label is rendered from the kind itself so a new group-row type only has to be
// added here, instead of silently rendering as a spline or as nothing at all.
const GROUP_ROW_KINDS = new Set(["spline", "piecewise"]);

function renderSummaryRow(row, hasLevelGroups, term, query, hidden) {
  // SE cell color is data-driven from Python's significance class. The browser
  // never infers significance from display text.
  const sigClass = safeSigClass(row.sig_class);
  const levelGroupCell = hasLevelGroups
    ? `<td class="summary-level-group">${highlightMatches(String(row.level_group || ""), query)}</td>`
    : "";
  return `
    <tr class="summary-row ${sigClass}" data-term="${escapeHTML(term)}"${hidden ? " hidden" : ""}>
      <td class="summary-term">
        <span>${highlightRowName(String(row.name || ""), term, query)}</span>
        ${GROUP_ROW_KINDS.has(row.kind) ? `<em>${escapeHTML(row.kind)}</em>` : ""}
      </td>
      ${levelGroupCell}
      ${renderNumberCell(row.edf, "summary-edf")}
      ${renderNumberCell(row.coef, "summary-estimate")}
      <td class="summary-se se-cell ${sigClass}" title="${escapeHTML(sigTitle(row))}">
        ${escapeHTML(formatSE(row))}
      </td>
      <td class="summary-p">${escapeHTML(formatP(row.p_value))}</td>
      <td class="sig-code">${escapeHTML(row.sig_code || "")}</td>
      <td class="advisory-code" title="${escapeHTML(advisoryTitle(row))}">${escapeHTML(row.advisory_code || "")}</td>
    </tr>
  `;
}

function renderNumberCell(value, columnClass) {
  return `<td class="summary-number ${columnClass}" title="${escapeHTML(formatFullNumber(value))}">${escapeHTML(formatSummaryValue(value))}</td>`;
}

function renderLevelGroupLegends(compact) {
  if (compact.level_display !== "grouped") return "";
  const groups = Array.isArray(compact.level_groups) ? compact.level_groups : [];
  if (!groups.length) return "";
  const byFeature = new Map();
  for (const group of groups) {
    const feature = String(group.feature || "");
    if (!byFeature.has(feature)) byFeature.set(feature, []);
    byFeature.get(feature).push(group);
  }
  return `
    <section class="summary-level-groups" aria-label="Level group membership">
      ${[...byFeature.entries()].map(([feature, featureGroups]) => `
        <div class="summary-level-groups-feature">
          <strong>Level groups (${escapeHTML(feature)}):</strong>
          ${featureGroups.map((group) => {
            const members = Array.isArray(group.members) ? group.members : [];
            return `
              <div class="summary-level-group-members">
                <span class="summary-level-group-id">${escapeHTML(group.group_id || "")}</span>
                <span aria-hidden="true"> = </span>
                <span>${members.map((member) => escapeHTML(String(member))).join(", ")}</span>
              </div>
            `;
          }).join("")}
        </div>
      `).join("")}
    </section>
  `;
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
    "sig-reference"
  ]);
  return allowed.has(value) ? value : "sig-unknown";
}

function sigTitle(row) {
  if (payloadNumber(row.p_value) === null) return "Inference unavailable for this row";
  return `Colored by ${row.stat_label || "p"} p=${formatP(row.p_value)}`;
}

// The advisory is a separate column from Sig because it is a separate finding:
// the rule behind it reads a level's volume, or a standard error's size, and
// never the p-value, so it cannot stand in for one (issue #239).
//
// Keyed off the marker the cell renders, so a row cannot carry a marker with no
// title or a title with no marker. The TEXT comes from the trigger that fired,
// because the tooltip is a row-level statement and the two triggers license
// different ones: a thin level is short of experience, while an outsized
// standard error can be a predictor on a much smaller scale with the whole
// sample behind it. Same split the exported Warning cell makes.
const ADVISORY_TITLES = {
  thin_level: "Low credibility: this level carries little experience",
  outsized_se:
    "Outsized standard error: wide next to this model's typical one — check the predictor's units",
  // Neutral, so an unrecognised kind does not fall into either row-level claim.
  unknown: "Flagged — see the LC note below the table"
};

function advisoryTitle(row) {
  if (!row.advisory_code) return "";
  return ADVISORY_TITLES[row.advisory_kind] || ADVISORY_TITLES.unknown;
}
