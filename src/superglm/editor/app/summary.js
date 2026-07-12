import { requestJSON } from "./api.js";
import { escapeHTML, fmt } from "./format.js";

const PROFILE_ESTIMATE_LABELS = { p: "p_hat", theta: "theta_hat" };
const summaryMarkupByFrame = new WeakMap();

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

export async function runDistributionProfile(
  nodes,
  parameter,
  acceptProfile = async () => {},
  { request = requestJSON, pause = sleep } = {}
) {
  const { summaryStatus, summaryFrame, reprofileTweedie, reprofileNb2, profileRun } = nodes;
  const button = parameter === "tweedie_p" ? reprofileTweedie : reprofileNb2;
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
    options: profileOptionsPayload(nodes, parameter)
  }, nodes);
  try {
    const started = await request("/profile_distribution/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parameter, ...profileOptionsPayload(nodes, parameter) })
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
    renderSummary(payload, nodes);
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

function profileOptionsPayload(nodes, parameter) {
  const xatol = Number(nodes.profileTolerance ? nodes.profileTolerance.value : 0.001);
  const payload = {
    xatol: Number.isFinite(xatol) && xatol > 0 ? xatol : 0.001
  };
  if (parameter === "tweedie_p" && nodes.profilePhiMethod) {
    payload.method = nodes.profileMethod ? nodes.profileMethod.value : "brent";
    payload.phi_method = nodes.profilePhiMethod.value;
    payload.trace_iterations = true;
  }
  return payload;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export function collapseTransition(term) {
  return {
    name: "collapse levels",
    path: "/collapse_levels",
    payload: { term, method: "auto" }
  };
}

export function ungroupTransition(term) {
  return {
    name: "ungroup levels",
    path: "/ungroup_levels",
    payload: { term, method: "auto" }
  };
}

export function uncollapseTransition() {
  return {
    name: "restore collapsed levels",
    path: "/uncollapse_levels",
    payload: {}
  };
}

export function renderSummary(payload, nodes) {
  const { summaryStatus, summaryNote, summaryFrame } = nodes;
  updateDistributionProfileActions(payload, nodes);
  if (!payload.available) {
    summaryStatus.textContent = payload.label || "Summary";
    summaryNote.textContent = "";
    updateSummaryMarkup(
      summaryFrame,
      `<div class="summary-empty">${escapeHTML(payload.error || "Summary unavailable.")}</div>`
    );
    return;
  }
  summaryStatus.textContent = payload.label || "Summary";
  summaryNote.textContent = payload.note || "";
  // Prefer the typed compact payload for the immediate panel. The raw HTML is
  // still included inside the disclosure for full notebook-style detail.
  updateSummaryMarkup(
    summaryFrame,
    payload.compact ? renderCompactSummary(payload) : payload.html || ""
  );
}

function updateSummaryMarkup(summaryFrame, markup) {
  if (summaryMarkupByFrame.get(summaryFrame) === markup) return;
  summaryFrame.innerHTML = markup;
  summaryMarkupByFrame.set(summaryFrame, markup);
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
  if (nodes.profileMethodWrap) nodes.profileMethodWrap.hidden = !canProfileTweedie;
  if (nodes.profilePhiWrap) nodes.profilePhiWrap.hidden = !canProfileTweedie;
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
  const status = profileStatusLabel(job, trace.length);
  const kind = fitTraceKind(trace);
  if (profileTraceStatus) {
    profileTraceStatus.textContent = kind ? `${status} · ${kind}` : status;
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
  if (job.phase === "profile_ci") {
    return estimateText
      ? `profile CI · ${estimateText}`
      : `profile CI · ${traceCount} evals`;
  }
  if (job.phase === "finalizing") return `updating summary · ${traceCount} evals`;
  if (traceCount > 0) return `profiling · ${traceCount} evals`;
  return "starting";
}

function isPostSearchPhase(phase) {
  return ["best_found", "final_refit", "profile_ci", "finalizing"].includes(phase);
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

function fitTraceKind(trace) {
  const row = trace.find((item) => Array.isArray(item.fit_trace) && item.fit_trace.length);
  return row && row.fit_trace_kind ? String(row.fit_trace_kind) : "";
}

function profileTraceSVG(trace, estimate) {
  const objective = profileObjectiveRows(trace, estimate);
  if (objective.length) return profileObjectiveSVG(objective, estimate);
  if (!trace.length) {
    return '<text x="160" y="64" text-anchor="middle" fill="#57606a" font-size="12">waiting for first evaluation</text>';
  }
  return '<text x="160" y="64" text-anchor="middle" fill="#57606a" font-size="12">no profile loss values yet</text>';
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
    <text x="${margin.left}" y="10" fill="#57606a" font-size="10">profile NLL</text>
    <polyline class="profile-trace-line" points="${points.join(" ")}"></polyline>
    ${values.map((row) => `<circle class="profile-trace-dot" cx="${x(row.value).toFixed(2)}" cy="${y(row.nll).toFixed(2)}" r="3"></circle>`).join("")}
    <line class="profile-trace-best-line" x1="${bestX}" y1="${margin.top}" x2="${bestX}" y2="${axisY}"></line>
    <circle class="profile-trace-best" cx="${bestX}" cy="${bestY}" r="4"></circle>
    <text class="profile-trace-best-label" x="${bestLabelX.toFixed(2)}" y="${bestLabelY.toFixed(2)}">${escapeHTML(bestLabel)}</text>
    <text x="${margin.left}" y="112" fill="#57606a" font-size="10">${escapeHTML(xLabel)}</text>
    <text x="310" y="112" text-anchor="end" fill="#57606a" font-size="10">profile loss</text>
  `;
}

function profileLearningCurvesSVG(curves, estimate) {
  const visible = curves.slice(-10);
  const allPoints = visible.flatMap((curve) => curve.points);
  const margin = { left: 48, right: 10, top: 12, bottom: 28 };
  const width = 320 - margin.left - margin.right;
  const height = 120 - margin.top - margin.bottom;
  const xMax = Math.max(1, ...allPoints.map((point) => point.iteration));
  const yMin = Math.min(...allPoints.map((point) => point.loss));
  const yMax = Math.max(...allPoints.map((point) => point.loss));
  const yPad = Math.max((yMax - yMin) * 0.08, Math.abs(yMax) * 0.002, 1e-9);
  const low = yMin - yPad;
  const high = yMax + yPad;
  const x = (iteration) => margin.left + width * (iteration / xMax);
  const y = (loss) => margin.top + height * (1 - (loss - low) / (high - low || 1));
  const yTicks = [high, (high + low) / 2, low];
  const xTicks = profileFitIterTicks(xMax);
  const axisY = margin.top + height;
  return `
    ${yTicks.map((tick) => `
      <line class="profile-trace-grid" x1="${margin.left}" y1="${y(tick).toFixed(2)}" x2="${margin.left + width}" y2="${y(tick).toFixed(2)}"></line>
      <text x="${margin.left - 5}" y="${(y(tick) + 3).toFixed(2)}" text-anchor="end" fill="#57606a" font-size="9">${escapeHTML(formatProfileNumber(tick))}</text>
    `).join("")}
    <line class="profile-trace-grid" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${axisY}"></line>
    <line class="profile-trace-grid" x1="${margin.left}" y1="${axisY}" x2="${margin.left + width}" y2="${axisY}"></line>
    ${xTicks.map((tick) => `
      <line class="profile-trace-tick" x1="${x(tick).toFixed(2)}" y1="${axisY}" x2="${x(tick).toFixed(2)}" y2="${axisY + 4}"></line>
      <text x="${x(tick).toFixed(2)}" y="${axisY + 14}" text-anchor="middle" fill="#57606a" font-size="9">${tick}</text>
    `).join("")}
    ${visible.map((curve, i) => profileLearningCurvePath(curve, i, x, y, profileCurveIsBest(curve, visible, estimate))).join("")}
    <text x="${margin.left}" y="114" fill="#57606a" font-size="10">fit iter</text>
    <text x="310" y="114" text-anchor="end" fill="#57606a" font-size="10">loss</text>
  `;
}

function profileFitIterTicks(xMax) {
  const max = Math.max(0, Math.ceil(xMax));
  const step = Math.max(1, Math.ceil(max / 4));
  const ticks = [];
  for (let tick = 0; tick <= max; tick += step) {
    ticks.push(tick);
  }
  if (!ticks.includes(max)) ticks.push(max);
  return ticks;
}

function profileLearningCurvePath(curve, index, x, y, isBest) {
  const points = curve.points
    .map((point) => `${x(point.iteration).toFixed(2)},${y(point.loss).toFixed(2)}`)
    .join(" ");
  const color = profileCurveColor(index);
  const opacity = 0.45 + 0.5 * ((index + 1) / 10);
  const curveClass = isBest ? "profile-learning-curve profile-learning-best" : "profile-learning-curve";
  const markers = curve.points.map((point, pointIndex) => {
    const isLast = pointIndex === curve.points.length - 1;
    const classes = isLast
      ? `profile-learning-point profile-learning-end${isBest ? " profile-learning-best-point" : ""}`
      : "profile-learning-point";
    const radius = isLast && isBest ? 3.8 : (isLast ? 2.8 : 2.2);
    return `<circle class="${classes}" cx="${x(point.iteration).toFixed(2)}" cy="${y(point.loss).toFixed(2)}" r="${radius}" fill="${color}"></circle>`;
  }).join("");
  return `
    <polyline class="${curveClass}" points="${points}" stroke="${color}" opacity="${isBest ? "1" : opacity.toFixed(2)}"></polyline>
    ${markers}
  `;
}

function profileTraceLegendHTML(trace, estimate, label) {
  const fitCurves = trace
    .filter((row) => Array.isArray(row.fit_trace) && row.fit_trace.length)
    .map((row, index) => ({
      index,
      p: Number(row.p),
      theta: Number(row.theta),
      profileLoss: outerProfileObjective(row),
      finalFitLoss: profileFinalFitLoss(row),
      row
    }))
    .filter((curve) => Number.isFinite(curve.profileLoss));
  const estimateBlock = estimate
    ? `<div class="profile-estimate">
        <strong>${escapeHTML(profileEstimateShortText(estimate))}</strong>
        <span>${escapeHTML(profileEstimateCIText(estimate))}</span>
        <em>outer profile objective minimum</em>
      </div>`
    : "";
  if (fitCurves.length) {
    return `${estimateBlock}${profileLearningCurveLegend(fitCurves, estimate)}`;
  }
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
  if (!low || !high) return "CI pending";
  return `CI [${low}, ${high}]`;
}

function profileLearningCurveLegend(curves, estimate) {
  return curves.map((curve, i) => {
    const labelValue = Number.isFinite(curve.p) ? curve.p : curve.theta;
    const label = Number.isFinite(labelValue) ? formatProfileNumber(labelValue) : String(curve.index + 1);
    const title = Number.isFinite(curve.finalFitLoss)
      ? `inner fit trace final ${formatProfileNumber(curve.finalFitLoss)}`
      : "inner fit trace";
    return profileLegendItem({
      color: profileCurveColor(i),
      label: `p ${label}`,
      detail: `profile loss ${formatProfileNumber(curve.profileLoss)}`,
      title,
      isBest: profileCurveIsBest(curve, curves, estimate)
    });
  }).join("");
}

function profileLegendItem({ color, label, detail, title = "", isBest }) {
  return `
    <div class="profile-legend-item${isBest ? " profile-legend-best" : ""}" title="${escapeHTML(title)}">
      <span class="profile-legend-swatch" style="background:${escapeHTML(color)}"></span>
      <strong>${escapeHTML(label)}</strong>
      <em>${escapeHTML(detail || "")}</em>
    </div>
  `;
}

function profileCurveIsBest(curve, curves, estimate) {
  const target = Number(estimate && estimate.value);
  const curveValue = Number.isFinite(curve.p) ? curve.p : curve.theta;
  if (Number.isFinite(target) && Number.isFinite(curveValue)) {
    return Math.abs(curveValue - target) < 5e-4;
  }
  const withLoss = curves.filter((item) => Number.isFinite(profileCurveFinalLoss(item)));
  if (!withLoss.length) return false;
  const best = withLoss.reduce(
    (acc, item) => (profileCurveFinalLoss(item) < profileCurveFinalLoss(acc) ? item : acc),
    withLoss[0]
  );
  return curve === best;
}

function profileFinalFitLoss(row) {
  if (!Array.isArray(row.fit_trace) || !row.fit_trace.length) return NaN;
  return Number(row.fit_trace.at(-1).loss);
}

function profileCurveFinalLoss(curve) {
  if (Number.isFinite(curve.profileLoss)) return curve.profileLoss;
  if (curve.points && curve.points.length) return Number(curve.points.at(-1).loss);
  return NaN;
}

function outerProfileObjective(row) {
  return Number(row.nll);
}

function profileCurveColor(index) {
  const colors = ["#0969da", "#2da44e", "#bf3989", "#d97706", "#8250df", "#1f6feb", "#cf222e", "#0a7f8f", "#6f42c1", "#57606a"];
  return colors[index % colors.length];
}

function profileTraceRows(trace, label) {
  const fitRows = trace
    .filter((row) => Array.isArray(row.fit_trace) && row.fit_trace.length)
    .slice(-4)
    .reverse();
  if (fitRows.length) return profileFitTraceRows(fitRows, label);

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

function profileFitTraceRows(rows, label) {
  const rangeLabel = "inner fit trace start -> final";
  return rows.map((row) => {
    const param = row[label] !== undefined ? `${label} ${formatProfileNumber(row[label])}` : "";
    const losses = row.fit_trace
      .map((point) => Number(point.loss))
      .filter((loss) => Number.isFinite(loss));
    const start = losses[0];
    const final = losses.at(-1);
    const range = losses.length
      ? `${formatProfileNumber(start)} -> ${formatProfileNumber(final)}`
      : "";
    const profileLoss = formatProfileNumber(outerProfileObjective(row));
    const kind = shortFitTraceKind(row.fit_trace_kind);
    return `
      <div class="profile-trace-row profile-fit-trace-row">
        <span>${escapeHTML(String(row.step ?? ""))}</span>
        <strong>${escapeHTML(param)}</strong>
        <span class="profile-fit-loss" title="${escapeHTML(`${rangeLabel}: ${range}`)}">profile loss ${escapeHTML(profileLoss)}</span>
        <span>${escapeHTML(kind)}</span>
      </div>
    `;
  }).join("");
}

function shortFitTraceKind(kind) {
  if (kind === "REML objective") return "REML obj";
  if (kind === "weighted deviance") return "deviance";
  return kind || "";
}

function formatProfileNumber(value) {
  if (value === null || value === undefined || value === "") return "";
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (Math.abs(number) >= 100) return fmt(number);
  if (Math.abs(number) >= 1) return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  return number.toPrecision(4);
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
  if (model.tweedie_p !== null && model.tweedie_p !== undefined) facts.push(["Tweedie p", model.tweedie_p]);
  if (model.nb_theta !== null && model.nb_theta !== undefined) facts.push(["NB2 theta", model.nb_theta]);
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
          ${renderSummaryRows(rows)}
        </tbody>
      </table>
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

function renderSummaryRows(rows) {
  let previousGroup = "";
  return rows.map((row) => {
    const group = summaryRowGroup(row);
    const showGroup = group && group !== previousGroup && group !== "Intercept";
    previousGroup = group || previousGroup;
    const groupRow = showGroup
      ? `<tr class="summary-group-row"><td colspan="6">${escapeHTML(group)}</td></tr>`
      : "";
    return `${groupRow}${renderSummaryRow(row)}`;
  }).join("");
}

function summaryRowGroup(row) {
  const group = row && row.group ? String(row.group) : "";
  if (group) return group;
  const name = row && row.name ? String(row.name) : "";
  const bracket = name.indexOf("[");
  return bracket > 0 ? name.slice(0, bracket) : name;
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
