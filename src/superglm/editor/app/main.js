import { requestJSON, postJSON } from "./api.js";
import { drawChart, groupedTerms } from "./chart.js";
import { fmt, fmtPercent } from "./format.js";
import { refreshMetrics } from "./metrics.js";
import { refreshReport } from "./reports.js";
import {
  refreshSummary,
  runDistributionProfile,
  runCollapseRefit,
  runOffsetRefit,
  runUncollapseRefit,
  runUngroupRefit
} from "./summary.js";
import { bindInteractions } from "./interactions.js";

const appTabs = Array.from(document.querySelectorAll(".app-tab"));
const editorView = document.getElementById("editorView");
const reportPanel = document.getElementById("reportPanel");
const reportTitle = document.getElementById("reportTitle");
const reportStatus = document.getElementById("reportStatus");
const reportFrame = document.getElementById("reportFrame");
const svg = document.getElementById("chart");
const selectionMenu = document.getElementById("selectionMenu");
const termSelect = document.getElementById("term");
const modeSelect = document.getElementById("mode");
const handleCountWrap = document.getElementById("handleCountWrap");
const handleCount = document.getElementById("handleCount");
const handleCountValue = document.getElementById("handleCountValue");
const basisToggle = document.getElementById("basisToggle");
const contribPlay = document.getElementById("contribPlay");
const buildDurationWrap = document.getElementById("buildDurationWrap");
const buildDuration = document.getElementById("buildDuration");
const buildDurationValue = document.getElementById("buildDurationValue");
const resetZoom = document.getElementById("resetZoom");
const ciToggle = document.getElementById("ciToggle");
const resetOrder = document.getElementById("resetOrder");
const collapseLevels = document.getElementById("collapseLevels");
const ungroupLevels = document.getElementById("ungroupLevels");
const uncollapseLevels = document.getElementById("uncollapseLevels");
const metricSelect = document.getElementById("metricSelect");
const metricGrid = document.getElementById("metricGrid");
const summarySource = document.getElementById("summarySource");
const refitOffset = document.getElementById("refitOffset");
const reprofileTweedie = document.getElementById("reprofileTweedie");
const reprofileNb2 = document.getElementById("reprofileNb2");
const summaryStatus = document.getElementById("summaryStatus");
const summaryNote = document.getElementById("summaryNote");
const summaryFrame = document.getElementById("summaryFrame");
const statusNode = document.getElementById("status");

let state = null;
let showCi = false;
let showContrib = false;
let graphMode = "select";
let buildProgress = null;
let buildFrame = null;
let renderedTerm = "";
let activeView = "editor";
const zoomState = {};

const chartContext = {
  svg,
  selectionMenu,
  modeSelect,
  zoomState,
  selectedTerm,
  visualMode: () => graphMode,
  showCi: () => showCi,
  showContrib: () => showContrib,
  buildProgress: () => buildProgress
};

async function loadState() {
  state = await requestJSON("/state");
  render();
}

async function postJSONWithRefresh(path, payload, options = {}) {
  state = await postJSON(path, payload);
  render();
  if (options.refreshMetrics) await refreshMetricsView();
  if (options.refreshSummary) await refreshSummaryView();
  await refreshActiveReport();
}

function selectedTerm() {
  if (!state) return "";
  return state.selected_term || Object.keys(state.terms)[0] || "";
}

function currentTerm() {
  return state ? state.terms[selectedTerm()] : null;
}

function currentSelection() {
  return state ? new Set(state.selection[selectedTerm()] || []) : new Set();
}

function summaryNodes() {
  return {
    summarySource,
    refitOffset,
    reprofileTweedie,
    reprofileNb2,
    collapseLevels,
    ungroupLevels,
    uncollapseLevels,
    summaryStatus,
    summaryNote,
    summaryFrame
  };
}

async function refreshMetricsView() {
  await refreshMetrics({ metricGrid, metricSelect });
}

async function refreshSummaryView() {
  await refreshSummary(summaryNodes());
}

async function refreshActiveReport() {
  if (activeView === "editor") return;
  await refreshReport({ report: activeView, reportTitle, reportStatus, reportFrame });
}

async function showView(view) {
  activeView = view === "final" ? "final" : view === "validation" ? "validation" : "editor";
  editorView.hidden = activeView !== "editor";
  reportPanel.hidden = activeView === "editor";
  for (const tab of appTabs) {
    const active = tab.dataset.view === activeView;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  await refreshActiveReport();
}

function render() {
  if (!state) return;
  const terms = state.terms || {};
  const selected = selectedTerm();
  termSelect.innerHTML = "";
  for (const [group, names] of groupedTerms(terms)) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group;
    for (const name of names) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      option.selected = name === selected;
      optgroup.appendChild(option);
    }
    termSelect.appendChild(optgroup);
  }
  const term = currentTerm();
  if (!term) return;
  if (selected !== renderedTerm) {
    renderedTerm = selected;
    applyTermDefaults(term);
  }
  const selection = currentSelection();
  const impact = term.impact || {};
  const rel = fmt(impact.weighted_mean_relativity || 1);
  const selectedShare = fmtPercent(impact.selected_weight_share || 0);
  const edf = term.effective_df === null || term.effective_df === undefined
    ? ""
    : ` · EDF ${fmt(term.effective_df)}`;
  statusNode.textContent = `${selected} · ${term.term_type || term.kind}${edf} · ${selection.size} of ${term.n_points} selected · average edit relativity ${rel}x · selected exposure ${selectedShare}`;
  updateHandleCount(term);
  updateCollapseAction(term, selection);
  updateResetOrderAction(term);
  drawChart(term, selection, chartContext);
}

function updateCollapseAction(term, selection) {
  const type = term.term_type || term.kind || "";
  const isLevelTerm = type === "categorical" || type === "ordered categorical";
  if (collapseLevels) {
    collapseLevels.hidden = !(isLevelTerm && selection.size >= 2);
  }
  if (ungroupLevels) {
    ungroupLevels.hidden = !(isLevelTerm && selectionTouchesCollapsedGroup(term, selection));
  }
  if (uncollapseLevels) {
    uncollapseLevels.hidden = !(
      isLevelTerm &&
      state.can_uncollapse_levels &&
      state.last_collapse &&
      selectionTouchesCollapsedGroup(term, selection)
    );
  }
}

function selectionTouchesCollapsedGroup(term, selection) {
  const groups = Array.isArray(term.level_groups) ? term.level_groups : [];
  if (!groups.length || !selection.size) return false;
  for (const group of groups) {
    const indices = Array.isArray(group.indices) ? group.indices : [];
    if (indices.some((index) => selection.has(Number(index)))) return true;
  }
  return false;
}

function updateResetOrderAction(term) {
  if (!resetOrder) return;
  const type = term.term_type || term.kind || "";
  const isLevelTerm = type === "categorical" || type === "ordered categorical";
  resetOrder.hidden = !(isLevelTerm && term.level_order_changed);
}

function updateHandleCount(term) {
  const controls = term.controls;
  const active = graphMode === "handles" && controls && controls.count;
  handleCountWrap.hidden = !active;
  const canShowContrib = active && Array.isArray(controls.basis) && controls.basis.length > 0;
  basisToggle.hidden = !canShowContrib;
  contribPlay.hidden = !canShowContrib;
  buildDurationWrap.hidden = !canShowContrib;
  contribPlay.disabled = buildFrame !== null;
  updateBuildDurationLabel();
  basisToggle.style.background = showContrib && canShowContrib ? "#dbeafe" : "#f6f8fa";
  if (!canShowContrib) {
    showContrib = false;
    stopContributionBuild();
  }
  if (!active) return;
  const min = Math.max(3, Number(controls.min_count || 3));
  const max = Math.max(min, Number(controls.max_count || controls.count || min));
  const value = Math.min(max, Math.max(min, Number(controls.count || min)));
  handleCount.min = String(min);
  handleCount.max = String(max);
  handleCount.value = String(value);
  handleCountValue.textContent = String(value);
}

function applyTermDefaults(term) {
  stopContributionBuild();
  if (canShowContributions(term)) {
    modeSelect.value = "handles";
    graphMode = "handles";
    showContrib = true;
  } else if (modeSelect.value === "handles" || graphMode === "handles") {
    modeSelect.value = "select";
    graphMode = "select";
    showContrib = false;
  }
}

function canShowContributions(term) {
  const controls = term && term.controls;
  return Boolean(
    controls &&
    Array.isArray(controls.basis) &&
    controls.basis.length > 0 &&
    Array.isArray(controls.build_basis) &&
    controls.build_basis.length > 0
  );
}

function buildDurationMs() {
  return Math.max(500, Number(buildDuration.value) || 10000);
}

function updateBuildDurationLabel() {
  const seconds = buildDurationMs() / 1000;
  buildDurationValue.textContent = Number.isInteger(seconds)
    ? `${seconds}s`
    : `${seconds.toFixed(1)}s`;
}

function startContributionBuild() {
  const term = currentTerm();
  if (!canShowContributions(term)) return;
  stopContributionBuild();
  modeSelect.value = "handles";
  graphMode = "handles";
  showContrib = true;
  runContributionBuild(0);
}

function runContributionBuild(fromProgress) {
  const initialProgress = Math.max(0, Math.min(1, Number(fromProgress) || 0));
  const duration = Math.max(1, buildDurationMs() * (1 - initialProgress));
  const started = performance.now();
  const step = (now) => {
    const elapsed = Math.max((now - started) / duration, 0);
    const progress = Math.min(initialProgress + elapsed * (1 - initialProgress), 1);
    buildProgress = progress;
    if (progress >= 1) buildFrame = null;
    render();
    if (progress < 1) {
      buildFrame = requestAnimationFrame(step);
    }
  };
  buildProgress = initialProgress;
  buildFrame = requestAnimationFrame(step);
  render();
}

function advanceContributionBuild() {
  const term = currentTerm();
  if (buildFrame === null || !canShowContributions(term)) return false;
  const controls = term.controls || {};
  const basis = Array.isArray(controls.build_basis) && controls.build_basis.length
    ? controls.build_basis
    : controls.basis;
  const count = Array.isArray(basis) ? basis.length : 0;
  if (!count) return false;
  const current = Math.max(0, Math.min(1, Number(buildProgress) || 0));
  const next = Math.min(1, (Math.floor(current * count) + 1) / count);
  cancelAnimationFrame(buildFrame);
  buildFrame = null;
  buildProgress = next;
  if (next < 1) {
    runContributionBuild(next);
  } else {
    render();
  }
  return true;
}

function stopContributionBuild() {
  if (buildFrame !== null) {
    cancelAnimationFrame(buildFrame);
    buildFrame = null;
  }
  buildProgress = null;
}

svg.addEventListener(
  "pointerdown",
  (event) => {
    if (event.button !== 0) return;
    if (!advanceContributionBuild()) return;
    event.preventDefault();
    event.stopImmediatePropagation();
  },
  true
);

const interactions = bindInteractions({
  svg,
  modeSelect,
  zoomState,
  selectedTerm,
  currentTerm,
  currentSelection,
  getState: () => state,
  hasState: () => state !== null,
  render,
  drawChart: (term, selection) => drawChart(term, selection, chartContext),
  postJSON: postJSONWithRefresh
});

for (const tab of appTabs) {
  tab.addEventListener("click", () => {
    showView(tab.dataset.view || "editor");
  });
}

termSelect.addEventListener("change", async () => {
  await postJSONWithRefresh("/term", { term: termSelect.value });
});

modeSelect.addEventListener("change", () => {
  stopContributionBuild();
  if (modeSelect.value !== "zoom") graphMode = modeSelect.value;
  if (graphMode === "handles" && canShowContributions(currentTerm())) {
    showContrib = true;
  }
  render();
});

basisToggle.addEventListener("click", () => {
  stopContributionBuild();
  showContrib = !showContrib;
  render();
});

contribPlay.addEventListener("click", startContributionBuild);

buildDuration.addEventListener("input", updateBuildDurationLabel);
buildDuration.addEventListener("change", updateBuildDurationLabel);

handleCount.addEventListener("input", () => {
  handleCountValue.textContent = handleCount.value;
});

handleCount.addEventListener("change", async () => {
  await postJSONWithRefresh("/control_count", {
    term: selectedTerm(),
    count: Number(handleCount.value)
  });
});

for (const button of document.querySelectorAll("button[data-op]")) {
  button.addEventListener("click", async () => {
    const operation = button.dataset.op;
    if (operation !== "select_all") stopContributionBuild();
    const displayOnly = isDisplayOnlyOperation(operation);
    await postJSONWithRefresh(
      "/op",
      { operation },
      {
        refreshMetrics: operation !== "select_all" && !displayOnly,
        refreshSummary: operation !== "select_all" && !displayOnly
      }
    );
  });
}

function isDisplayOnlyOperation(operation) {
  return operation === "reorder_levels" || operation === "reset_order";
}

ciToggle.addEventListener("click", () => {
  showCi = !showCi;
  ciToggle.style.background = showCi ? "#dbeafe" : "#f6f8fa";
  render();
});

resetZoom.addEventListener("click", interactions.resetZoomView);
summarySource.addEventListener("change", refreshSummaryView);
refitOffset.addEventListener("click", async () => {
  await runOffsetRefit(summaryNodes(), refreshMetricsView);
  await refreshActiveReport();
});
if (reprofileTweedie) {
  reprofileTweedie.addEventListener("click", async () => {
    stopContributionBuild();
    summarySource.value = "selected";
    await runDistributionProfile(summaryNodes(), "tweedie_p", refreshMetricsView);
    state = await requestJSON("/state");
    render();
    await refreshActiveReport();
  });
}
if (reprofileNb2) {
  reprofileNb2.addEventListener("click", async () => {
    stopContributionBuild();
    summarySource.value = "selected";
    await runDistributionProfile(summaryNodes(), "nb2_theta", refreshMetricsView);
    state = await requestJSON("/state");
    render();
    await refreshActiveReport();
  });
}
if (collapseLevels) {
  collapseLevels.addEventListener("click", async () => {
    stopContributionBuild();
    summarySource.value = "selected";
    await runCollapseRefit(summaryNodes(), selectedTerm(), refreshMetricsView);
    state = await requestJSON("/state");
    render();
    await refreshMetricsView();
    await refreshActiveReport();
  });
}
if (ungroupLevels) {
  ungroupLevels.addEventListener("click", async () => {
    stopContributionBuild();
    summarySource.value = "selected";
    await runUngroupRefit(summaryNodes(), selectedTerm(), refreshMetricsView);
    state = await requestJSON("/state");
    render();
    await refreshMetricsView();
    await refreshActiveReport();
  });
}
if (uncollapseLevels) {
  uncollapseLevels.addEventListener("click", async () => {
    stopContributionBuild();
    await runUncollapseRefit(summaryNodes(), refreshMetricsView);
    state = await requestJSON("/state");
    render();
    await refreshMetricsView();
    await refreshActiveReport();
  });
}

loadState().then(async () => {
  await refreshMetricsView();
  await refreshSummaryView();
}).catch((error) => {
  statusNode.textContent = error.message;
  statusNode.style.color = "#b42318";
});
