import { editorClient } from "./api/client.js";
import { drawChart, groupedTerms } from "./chart.js";
import { fmt, fmtPercent } from "./format.js";
import { renderHistory } from "./history.js";
import { renderMetricGrid } from "./metrics.js";
import { renderReport } from "./reports.js";
import { createEditorActions } from "./state/actions.js";
import {
  selectActiveTermName,
  selectCurrentSelection,
  selectGroupDisplayMode,
  selectRenderableTerm
} from "./state/selectors.js";
import {
  createEditorStore,
  createInitialEditorState,
  patchView as patchViewState,
  setPreviewTerm as setPreviewTermState
} from "./state/store.js";
import {
  refreshSummary,
  runDistributionProfile,
  showDistributionProfileDialog,
  runCollapseRefit,
  runOffsetRefit,
  runUncollapseRefit,
  runUngroupRefit
} from "./summary.js";
import { bindInteractions } from "./interactions.js";

const appTabs = Array.from(document.querySelectorAll(".app-tab"));
const appShell = document.querySelector(".app-shell");
const appBusyOverlay = document.getElementById("appBusyOverlay");
const appBusyTitle = document.getElementById("appBusyTitle");
const appBusyDetail = document.getElementById("appBusyDetail");
const appAlert = document.getElementById("appAlert");
const appAlertMessage = document.getElementById("appAlertMessage");
const appAlertRetry = document.getElementById("appAlertRetry");
const appAlertDismiss = document.getElementById("appAlertDismiss");
const editorView = document.getElementById("editorView");
const reportPanel = document.getElementById("reportPanel");
const reportTitle = document.getElementById("reportTitle");
const reportStatus = document.getElementById("reportStatus");
const reportFrame = document.getElementById("reportFrame");
const svg = document.getElementById("chart");
const selectionMenu = document.getElementById("selectionMenu");
const termSelect = document.getElementById("term");
const modeSelect = document.getElementById("mode");
const groupDisplayWrap = document.getElementById("groupDisplayWrap");
const groupDisplayMode = document.getElementById("groupDisplayMode");
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
const saveModel = document.getElementById("saveModel");
const saveDialog = document.getElementById("saveDialog");
const saveDialogClose = document.getElementById("saveDialogClose");
const saveDirectory = document.getElementById("saveDirectory");
const saveOpenDirectory = document.getElementById("saveOpenDirectory");
const saveFilename = document.getElementById("saveFilename");
const saveConfirm = document.getElementById("saveConfirm");
const saveDownload = document.getElementById("saveDownload");
const saveStatus = document.getElementById("saveStatus");
const collapseLevels = document.getElementById("collapseLevels");
const ungroupLevels = document.getElementById("ungroupLevels");
const uncollapseLevels = document.getElementById("uncollapseLevels");
const metricSelect = document.getElementById("metricSelect");
const metricGrid = document.getElementById("metricGrid");
const summarySource = document.getElementById("summarySource");
const refitOffset = document.getElementById("refitOffset");
const reprofileTweedie = document.getElementById("reprofileTweedie");
const reprofileNb2 = document.getElementById("reprofileNb2");
const profileDialog = document.getElementById("profileDialog");
const profileDialogTitle = document.getElementById("profileDialogTitle");
const profileDialogDescription = document.getElementById("profileDialogDescription");
const profileDialogClose = document.getElementById("profileDialogClose");
const profileOptions = document.getElementById("profileOptions");
const profileMethodWrap = document.getElementById("profileMethodWrap");
const profileMethod = document.getElementById("profileMethod");
const profilePhiWrap = document.getElementById("profilePhiWrap");
const profilePhiMethod = document.getElementById("profilePhiMethod");
const profileTolerance = document.getElementById("profileTolerance");
const profileRun = document.getElementById("profileRun");
const profileProgress = document.getElementById("profileProgress");
const profileTraceStatus = document.getElementById("profileTraceStatus");
const profileTraceLegend = document.getElementById("profileTraceLegend");
const profileTracePlot = document.getElementById("profileTracePlot");
const profileTraceTable = document.getElementById("profileTraceTable");
const summaryStatus = document.getElementById("summaryStatus");
const summaryNote = document.getElementById("summaryNote");
const summaryFrame = document.getElementById("summaryFrame");
const summaryTab = document.getElementById("summaryTab");
const historyTab = document.getElementById("historyTab");
const summaryPane = document.getElementById("summaryPane");
const historyPane = document.getElementById("historyPane");
const historyFrame = document.getElementById("historyFrame");
const statusNode = document.getElementById("status");

let buildProgress = null;
let buildFrame = null;
let renderedTerm = "";
let appBusyTimer = null;
let appBusyStarted = 0;
let retainedRecovery = null;

const store = createEditorStore(createInitialEditorState());
const actions = createEditorActions({
  store,
  client: editorClient,
  scheduleEvidence: () => {
    resetSummarySourceAfterInvalidatingEdit();
    void refreshMetricsView();
    void refreshSummaryView();
    void refreshActiveReport();
  }
});

const chartContext = {
  svg,
  selectionMenu,
  modeSelect,
  get zoomState() {
    return store.getState().view.zoomByTerm;
  },
  selectedTerm,
  visualMode,
  showCi: () => store.getState().view.showCi,
  showContrib: () => store.getState().view.showContrib,
  buildProgress: () => buildProgress,
  groupDisplayMode: () => activeGroupDisplayMode()
};

async function loadState() {
  await actions.initialize();
}

async function executeStateMutation(path, payload) {
  return actions.executeStateMutation({
    name: mutationName(path, payload),
    path,
    payload
  });
}

function mutationName(path, payload) {
  if (path === "/op" && typeof payload.operation === "string") return payload.operation;
  return path.replace(/^\//, "") || "editor mutation";
}

function resetSummarySourceAfterInvalidatingEdit() {
  if (summarySource.value === "refit") {
    summarySource.value = "selected";
  }
}

function selectedTerm() {
  return selectActiveTermName(store.getState());
}

function currentTerm() {
  return selectRenderableTerm(store.getState());
}

function currentSelection() {
  return new Set(selectCurrentSelection(store.getState()));
}

function interactionMode() {
  return store.getState().view.mode;
}

function setInteractionPreview(term, payload) {
  store.update((state) => setPreviewTermState(state, term, payload));
}

function setZoom(term, range) {
  store.update((state) => patchViewState(state, {
    zoomByTerm: { ...state.view.zoomByTerm, [term]: range }
  }));
}

function clearZoom(term) {
  store.update((state) => {
    if (!(term in state.view.zoomByTerm)) return state;
    const zoomByTerm = { ...state.view.zoomByTerm };
    delete zoomByTerm[term];
    return patchViewState(state, { zoomByTerm });
  });
}

function activeGroupDisplayMode() {
  return selectGroupDisplayMode(store.getState());
}

function visualMode() {
  const view = store.getState().view;
  return view.mode === "zoom" && view.showContrib ? "handles" : view.mode;
}

function summaryNodes() {
  return {
    summarySource,
    refitOffset,
    reprofileTweedie,
    reprofileNb2,
    profileDialog,
    profileDialogTitle,
    profileDialogDescription,
    profileRun,
    profileOptions,
    profileMethodWrap,
    profileMethod,
    profilePhiWrap,
    profilePhiMethod,
    profileTolerance,
    profileProgress,
    profileTraceStatus,
    profileTraceLegend,
    profileTracePlot,
    profileTraceTable,
    collapseLevels,
    ungroupLevels,
    uncollapseLevels,
    summaryStatus,
    summaryNote,
    summaryFrame
  };
}

if (profileDialogClose && profileDialog) {
  profileDialogClose.addEventListener("click", () => {
    if (typeof profileDialog.close === "function") {
      profileDialog.close();
    } else {
      profileDialog.removeAttribute("open");
    }
  });
}

if (saveDialogClose && saveDialog) {
  saveDialogClose.addEventListener("click", () => {
    if (typeof saveDialog.close === "function") {
      saveDialog.close();
    } else {
      saveDialog.removeAttribute("open");
    }
  });
}

async function runProfileFromDialog() {
  if (!profileDialog) return;
  const parameter = profileDialog.dataset.parameter || "tweedie_p";
  stopContributionBuild();
  summarySource.value = "selected";
  const payload = await runDistributionProfile(summaryNodes(), parameter, refreshMetricsView);
  if (payload) {
    await actions.initialize();
    await refreshActiveReport();
  }
}

async function saveEditedModel() {
  if (!saveConfirm) return;
  saveConfirm.disabled = true;
  if (saveStatus) saveStatus.textContent = "Saving...";
  try {
    const payload = await editorClient.requestJSON("/save_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        directory: saveDirectory ? saveDirectory.value : ".",
        filename: saveFilename ? saveFilename.value : "superglm_edited_model.joblib"
      })
    });
    if (saveStatus) saveStatus.textContent = `Saved ${payload.path}`;
  } catch (error) {
    if (saveStatus) saveStatus.textContent = error.message;
  } finally {
    saveConfirm.disabled = false;
  }
}

async function downloadEditedModel() {
  if (!saveDownload) return;
  saveDownload.disabled = true;
  if (saveStatus) saveStatus.textContent = "Preparing download...";
  const requestedName = saveFilename ? saveFilename.value : "superglm_edited_model.joblib";
  try {
    const response = await editorClient.requestBlob(
      `/download_model?filename=${encodeURIComponent(requestedName || "superglm_edited_model.joblib")}`
    );
    const blob = await response.blob();
    const filename =
      filenameFromDisposition(response.headers.get("content-disposition")) ||
      requestedName ||
      "superglm_edited_model.joblib";
    const message = await saveBlobToFile(blob, filename);
    if (saveStatus) saveStatus.textContent = message;
  } catch (error) {
    if (saveStatus) saveStatus.textContent = error.message;
  } finally {
    saveDownload.disabled = false;
  }
}

function filenameFromDisposition(disposition) {
  if (!disposition) return "";
  const match = disposition.match(/filename="([^"]+)"/i);
  return match ? match[1] : "";
}

async function saveBlobToFile(blob, filename) {
  if (typeof window.showSaveFilePicker === "function" && window.isSecureContext) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [
          {
            description: "Joblib model",
            accept: { "application/octet-stream": [".joblib"] }
          }
        ]
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return `Saved ${filename}`;
    } catch (error) {
      if (error && error.name === "AbortError") return "Download cancelled.";
    }
  }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = "noopener";
  anchor.style.display = "none";
  document.body.append(anchor);
  anchor.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    anchor.remove();
  }, 0);
  return `Downloaded ${filename}`;
}

async function openSaveDialog() {
  if (saveStatus) saveStatus.textContent = "";
  await initializeSaveDirectory();
  if (saveDialog && typeof saveDialog.showModal === "function") {
    saveDialog.showModal();
  } else if (saveDialog) {
    saveDialog.setAttribute("open", "");
  }
}

async function initializeSaveDirectory() {
  if (!saveDirectory || saveDirectory.value) return;
  try {
    const payload = await editorClient.requestJSON("/save_directory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: saveDirectory ? saveDirectory.value : "" })
    });
    if (payload && payload.path) saveDirectory.value = payload.path;
  } catch (error) {
    if (saveStatus) saveStatus.textContent = formatSaveRouteError(error);
  }
}

async function openDirectoryInFileManager() {
  if (!saveOpenDirectory) return;
  saveOpenDirectory.disabled = true;
  if (saveStatus) saveStatus.textContent = "Opening folder...";
  try {
    const payload = await editorClient.requestJSON("/open_directory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: saveDirectory ? saveDirectory.value : "." })
    });
    if (saveStatus) saveStatus.textContent = `Opened ${payload.path}`;
  } catch (error) {
    if (saveStatus) saveStatus.textContent = formatSaveRouteError(error);
  } finally {
    saveOpenDirectory.disabled = false;
  }
}

function formatSaveRouteError(error) {
  const message = error && error.message ? error.message : String(error);
  if (message === "not found") {
    return "Save controls are newer than this running editor server. Rerun session.widget() or restart the kernel.";
  }
  return message;
}

async function refreshMetricsView() {
  await actions.refreshEvidence("metrics", "/metrics", {
    metric: "deviance",
    source: "in_force"
  });
}

async function refreshSummaryView() {
  await refreshSummary(summaryNodes());
}

async function refreshActiveReport() {
  const activeView = store.getState().view.activeView;
  if (activeView === "editor") return;
  await actions.refreshEvidence("report", "/report", { report: activeView });
}

async function runStructuralRefit(label, action) {
  const operationStart = performance.now();
  setAppBusy(true, label, "Starting...");
  try {
    const requestStart = performance.now();
    const payload = await action();
    const requestEnd = performance.now();
    if (!payload) return null;
    await actions.initialize();
    await refreshMetricsView();
    await refreshActiveReport();
    const completed = performance.now();
    const timing = debugTiming(payload, operationStart, requestStart, requestEnd, completed);
    showTimingStatus(payload, timing);
    return payload;
  } finally {
    setAppBusy(false);
  }
}

function setAppBusy(active, title = "Working...", detail = "") {
  if (!appShell || !appBusyOverlay) return;
  if (appBusyTimer !== null) {
    clearInterval(appBusyTimer);
    appBusyTimer = null;
  }
  appShell.classList.toggle("is-busy", active);
  appShell.setAttribute("aria-busy", active ? "true" : "false");
  appBusyOverlay.hidden = !active;
  if (!active) return;
  appBusyStarted = performance.now();
  const update = () => {
    const elapsed = performance.now() - appBusyStarted;
    if (appBusyTitle) appBusyTitle.textContent = title;
    if (appBusyDetail) {
      appBusyDetail.textContent = `${detail || "Refitting model"} · ${formatMilliseconds(elapsed)} elapsed`;
    }
  };
  update();
  appBusyTimer = window.setInterval(update, 250);
}

function debugTiming(payload, operationStart, requestStart, requestEnd, completed) {
  const server = payload && payload.timing ? payload.timing : {};
  return {
    ...server,
    client_request_ms: Math.max(0, requestEnd - requestStart),
    client_recovery_ms: Math.max(0, completed - requestEnd),
    client_total_ms: Math.max(0, completed - operationStart)
  };
}

function showTimingStatus(payload, timing) {
  if (!timing) return;
  if (summaryStatus) {
    summaryStatus.textContent = `Refit completed in ${formatMilliseconds(timing.client_total_ms)}`;
  }
  if (summaryNote) {
    const details = formatTimingDetails(timing);
    summaryNote.textContent = payload.note ? `${payload.note} · ${details}` : details;
  }
}

function formatTimingDetails(timing) {
  const parts = [];
  if (Number.isFinite(Number(timing.server_total_ms))) {
    parts.push(`server ${formatMilliseconds(timing.server_total_ms)}`);
  }
  if (Number.isFinite(Number(timing.fit_ms))) {
    parts.push(`fit ${formatMilliseconds(timing.fit_ms)}`);
  }
  if (Number.isFinite(Number(timing.summary_ms))) {
    parts.push(`summary ${formatMilliseconds(timing.summary_ms)}`);
  }
  if (Number.isFinite(Number(timing.client_recovery_ms))) {
    parts.push(`browser recovery ${formatMilliseconds(timing.client_recovery_ms)}`);
  }
  return parts.length ? `Timing: ${parts.join(", ")}` : "";
}

function formatMilliseconds(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (number < 1000) return `${Math.round(number)} ms`;
  return `${(number / 1000).toFixed(2)} s`;
}

async function showView(view) {
  const activeView = view === "final" ? "final" : view === "validation" ? "validation" : "editor";
  actions.patchView({ activeView });
  await refreshActiveReport();
}

function renderAppView(activeView) {
  editorView.hidden = activeView !== "editor";
  reportPanel.hidden = activeView === "editor";
  for (const tab of appTabs) {
    const active = tab.dataset.view === activeView;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
}

function render() {
  const editorState = store.getState();
  const snapshot = editorState.remote.snapshot;
  if (!snapshot) return;
  const view = editorState.view;
  renderAppView(view.activeView);
  showSidepanelPane(view.inspectorPane);
  renderMetricsEvidence(editorState.request.evidence.metrics);
  renderReportEvidence(editorState.request.evidence.report, view.activeView);
  renderHistory(snapshot.history, historyFrame);
  modeSelect.value = view.mode;
  ciToggle.style.background = view.showCi ? "#dbeafe" : "#f6f8fa";
  const terms = snapshot.terms || {};
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
    if (applyTermDefaults(term)) return;
  }
  const selection = currentSelection();
  const impact = term.impact || {};
  const rel = fmt(impact.weighted_mean_relativity || 1);
  const selectedShare = fmtPercent(impact.selected_weight_share || 0);
  const edf = term.effective_df === null || term.effective_df === undefined
    ? ""
    : ` · EDF ${fmt(term.effective_df)}`;
  const collapsedOriginalNote =
    activeGroupDisplayMode() === "collapsed" && term.group_display && term.group_display.available
      ? " · original line is grouped by exposure-weighted averaging"
      : "";
  statusNode.style.color = "";
  statusNode.textContent = `${selected} · ${term.term_type || term.kind}${edf} · ${selection.size} of ${term.n_points} selected · average edit relativity ${rel}x · selected exposure ${selectedShare}${collapsedOriginalNote}`;
  if (updateHandleCount(term)) return;
  updateGroupDisplayControl(term);
  updateCollapseAction(term, selection);
  updateResetOrderAction(term);
  drawChart(term, selection, chartContext);
}

function renderRecovery({ recovery, mutationStatus }) {
  if (!appAlert || !appAlertMessage || !appAlertRetry || !appAlertDismiss) return;
  if (recovery) retainedRecovery = recovery;
  const retrying = mutationStatus === "running" && retainedRecovery !== null;
  if (!recovery && !retrying) {
    retainedRecovery = null;
    appAlert.hidden = true;
    appAlertMessage.textContent = "";
    appAlertRetry.disabled = false;
    appAlertDismiss.disabled = false;
    return;
  }
  const visibleRecovery = recovery || retainedRecovery;
  appAlertMessage.textContent = visibleRecovery ? visibleRecovery.message : "Editor request failed.";
  appAlertRetry.disabled = retrying;
  appAlertDismiss.disabled = retrying;
  appAlert.hidden = false;
}

function renderMetricsEvidence(evidence) {
  const busy = evidence.status === "updating";
  metricGrid.setAttribute("aria-busy", busy ? "true" : "false");
  metricGrid.title = "";
  renderMetricGrid(evidence.payload, { metricGrid, metricSelect });
  if (evidence.status !== "error") return;
  if (evidence.payload === null) {
    metricGrid.textContent = evidence.error || "Metric unavailable.";
  } else {
    metricGrid.title = evidence.error || "Metric unavailable.";
  }
}

function renderReportEvidence(evidence, activeView) {
  const busy = evidence.status === "updating";
  reportFrame.setAttribute("aria-busy", busy ? "true" : "false");
  if (activeView === "editor") return;
  const payloadMatchesView = evidence.payload !== null && evidence.payload.report === activeView;
  if (payloadMatchesView) {
    renderReport(evidence.payload, { reportTitle, reportStatus, reportFrame });
  } else {
    reportTitle.textContent = activeView === "final" ? "Final Fit Report" : "Validation Report";
    reportFrame.innerHTML = "";
  }
  if (evidence.status === "error") {
    reportStatus.textContent = evidence.error || "Report unavailable.";
  } else if (!payloadMatchesView || busy || evidence.status === "idle") {
    reportStatus.textContent = "Loading report...";
  }
}

function showSidepanelPane(view) {
  const showHistory = view === "history";
  if (summaryPane) summaryPane.hidden = showHistory;
  if (historyPane) historyPane.hidden = !showHistory;
  if (summaryTab) {
    summaryTab.classList.toggle("active", !showHistory);
    summaryTab.setAttribute("aria-selected", showHistory ? "false" : "true");
  }
  if (historyTab) {
    historyTab.classList.toggle("active", showHistory);
    historyTab.setAttribute("aria-selected", showHistory ? "true" : "false");
  }
}

function updateGroupDisplayControl(term) {
  if (!groupDisplayWrap || !groupDisplayMode) return;
  const available = Boolean(term && term.group_display && term.group_display.available);
  groupDisplayMode.disabled = !available;
  if (!available) {
    groupDisplayMode.value = "expanded";
    return;
  }
  groupDisplayMode.value = activeGroupDisplayMode();
}

function updateCollapseAction(term, selection) {
  const snapshot = store.getState().remote.snapshot;
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
      snapshot &&
      snapshot.can_uncollapse_levels &&
      snapshot.last_collapse &&
      snapshot.last_collapse.term === selectedTerm() &&
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
  resetOrder.hidden = !(type === "categorical" && term.level_order_changed);
}

function updateHandleCount(term) {
  const view = store.getState().view;
  const controls = term.controls;
  const active = visualMode() === "handles" && controls && controls.count;
  handleCountWrap.hidden = !active;
  const canShowContrib = active && Array.isArray(controls.basis) && controls.basis.length > 0;
  basisToggle.hidden = !canShowContrib;
  contribPlay.hidden = !canShowContrib;
  buildDurationWrap.hidden = !canShowContrib;
  contribPlay.disabled = buildFrame !== null;
  updateBuildDurationLabel();
  basisToggle.style.background = view.showContrib && canShowContrib ? "#dbeafe" : "#f6f8fa";
  if (!canShowContrib) {
    stopContributionBuild();
    if (view.showContrib) {
      actions.patchView({ showContrib: false });
      return true;
    }
  }
  if (!active) return false;
  const min = Math.max(3, Number(controls.min_count || 3));
  const max = Math.max(min, Number(controls.max_count || controls.count || min));
  const value = Math.min(max, Math.max(min, Number(controls.count || min)));
  handleCount.min = String(min);
  handleCount.max = String(max);
  handleCount.value = String(value);
  handleCountValue.textContent = String(value);
  return false;
}

function applyTermDefaults(term) {
  stopContributionBuild();
  const view = store.getState().view;
  const patch = {};
  if (
    term.group_display &&
    term.group_display.available &&
    !view.groupModeByTerm[selectedTerm()]
  ) {
    patch.groupModeByTerm = {
      ...view.groupModeByTerm,
      [selectedTerm()]: term.group_display.default_mode || "expanded"
    };
  }
  if (canShowContributions(term)) {
    if (view.mode !== "handles") patch.mode = "handles";
    if (!view.showContrib) patch.showContrib = true;
  } else {
    if (view.mode === "handles") patch.mode = "select";
    if (view.showContrib) patch.showContrib = false;
  }
  if (!Object.keys(patch).length) return false;
  actions.patchView(patch);
  return true;
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
  buildProgress = 0;
  actions.patchView({ mode: "handles", showContrib: true });
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
  mode: interactionMode,
  selectedTerm,
  currentTerm,
  currentSelection,
  setPreviewTerm: setInteractionPreview,
  setZoom,
  clearZoom,
  actions,
  drawChart: (term, selection) => drawChart(term, selection, chartContext),
});

for (const tab of appTabs) {
  tab.addEventListener("click", () => {
    showView(tab.dataset.view || "editor");
  });
}

termSelect.addEventListener("change", async () => {
  const term = termSelect.value;
  const result = await executeStateMutation("/term", { term });
  if (result.ok) {
    actions.patchView({ activeTerm: term });
    return;
  }
  const snapshot = store.getState().remote.snapshot;
  const authoritativeTerm = snapshot?.selected_term;
  if (authoritativeTerm && snapshot.terms[authoritativeTerm]) {
    actions.patchView({ activeTerm: authoritativeTerm });
  }
});

modeSelect.addEventListener("change", () => {
  stopContributionBuild();
  const mode = modeSelect.value;
  const view = store.getState().view;
  actions.patchView({
    mode,
    showContrib: mode === "zoom"
      ? view.showContrib
      : mode === "handles" && canShowContributions(currentTerm())
  });
});

if (groupDisplayMode) {
  groupDisplayMode.addEventListener("change", () => {
    const view = store.getState().view;
    const term = selectedTerm();
    const zoomByTerm = { ...view.zoomByTerm };
    delete zoomByTerm[term];
    actions.patchView({
      groupModeByTerm: { ...view.groupModeByTerm, [term]: groupDisplayMode.value },
      zoomByTerm
    });
  });
}

basisToggle.addEventListener("click", () => {
  stopContributionBuild();
  actions.patchView({ showContrib: !store.getState().view.showContrib });
});

contribPlay.addEventListener("click", startContributionBuild);

buildDuration.addEventListener("input", updateBuildDurationLabel);
buildDuration.addEventListener("change", updateBuildDurationLabel);

handleCount.addEventListener("input", () => {
  handleCountValue.textContent = handleCount.value;
});

handleCount.addEventListener("change", async () => {
  await executeStateMutation("/control_count", {
    term: selectedTerm(),
    count: Number(handleCount.value)
  });
});

for (const button of document.querySelectorAll("button[data-op]")) {
  button.addEventListener("click", async () => {
    const operation = button.dataset.op;
    if (operation !== "select_all") stopContributionBuild();
    await executeStateMutation("/op", { operation });
  });
}

ciToggle.addEventListener("click", () => {
  actions.patchView({ showCi: !store.getState().view.showCi });
});

resetZoom.addEventListener("click", interactions.resetZoomView);
if (appAlertRetry) {
  appAlertRetry.addEventListener("click", () => {
    void actions.retryMutation();
  });
}
if (appAlertDismiss) {
  appAlertDismiss.addEventListener("click", () => actions.dismissRecovery());
}
if (saveModel) {
  saveModel.addEventListener("click", openSaveDialog);
}
if (saveConfirm) {
  saveConfirm.addEventListener("click", saveEditedModel);
}
if (saveDownload) {
  saveDownload.addEventListener("click", downloadEditedModel);
}
if (saveOpenDirectory) {
  saveOpenDirectory.addEventListener("click", openDirectoryInFileManager);
}
if (summaryTab) {
  summaryTab.addEventListener("click", () => actions.patchView({ inspectorPane: "summary" }));
}
if (historyTab) {
  historyTab.addEventListener("click", () => actions.patchView({ inspectorPane: "history" }));
}
summarySource.addEventListener("change", refreshSummaryView);
refitOffset.addEventListener("click", async () => {
  const payload = await runOffsetRefit(summaryNodes(), refreshMetricsView);
  if (payload) {
    await actions.initialize();
    await refreshActiveReport();
  }
});
if (reprofileTweedie) {
  reprofileTweedie.addEventListener("click", () => {
    stopContributionBuild();
    summarySource.value = "selected";
    showDistributionProfileDialog(summaryNodes(), "tweedie_p");
  });
}
if (reprofileNb2) {
  reprofileNb2.addEventListener("click", () => {
    stopContributionBuild();
    summarySource.value = "selected";
    showDistributionProfileDialog(summaryNodes(), "nb2_theta");
  });
}
if (profileRun) {
  profileRun.addEventListener("click", runProfileFromDialog);
}
if (collapseLevels) {
  collapseLevels.addEventListener("click", async () => {
    stopContributionBuild();
    summarySource.value = "selected";
    await runStructuralRefit(
      "Refitting collapsed levels",
      () => runCollapseRefit(summaryNodes(), selectedTerm())
    );
  });
}
if (ungroupLevels) {
  ungroupLevels.addEventListener("click", async () => {
    stopContributionBuild();
    summarySource.value = "selected";
    await runStructuralRefit(
      "Refitting ungrouped levels",
      () => runUngroupRefit(summaryNodes(), selectedTerm())
    );
  });
}
if (uncollapseLevels) {
  uncollapseLevels.addEventListener("click", async () => {
    stopContributionBuild();
    await runStructuralRefit(
      "Restoring previous collapsed-level model",
      () => runUncollapseRefit(summaryNodes())
    );
  });
}

store.subscribe((state) => state, () => render());
store.subscribe(
  (state) => ({
    recovery: state.request.recovery,
    mutationStatus: state.request.mutation.status
  }),
  renderRecovery,
  (next, previous) => (
    next.recovery === previous.recovery && next.mutationStatus === previous.mutationStatus
  )
);

loadState().then(async () => {
  await refreshMetricsView();
  await refreshSummaryView();
}).catch((error) => {
  statusNode.textContent = error.message;
  statusNode.style.color = "#b42318";
});
