import { editorClient } from "./api/client.js";
import { drawChart, groupedTerms, updateChartSelection } from "./chart.js";
import { renderHistory } from "./history.js";
import { renderMetricGrid } from "./metrics.js";
import { renderReport } from "./reports.js";
import { createEditorActions } from "./state/actions.js";
import {
  selectActiveTermName,
  selectCurrentSelection,
  selectGroupDisplayMode,
  selectRenderableTerm,
  selectSnapshot
} from "./state/selectors.js";
import {
  createEditorStore,
  createInitialEditorState,
  patchView as patchViewState,
  setPreviewTerm as setPreviewTermState
} from "./state/store.js";
import {
  clientTransitionTiming,
  createEvidenceTimingTracker
} from "./state/timing.js";
import {
  collapseTransition,
  renderSummary,
  runDistributionProfile,
  showDistributionProfileDialog,
  runOffsetRefit,
  uncollapseTransition,
  ungroupTransition
} from "./summary.js";
import { bindInteractions } from "./interactions.js";
import { bindAppBar, renderAppBar } from "./views/app_bar.js";
import { renderContextBar } from "./views/context_bar.js";
import { renderHelpDrawer } from "./views/help_drawer.js";
import { bindInspector, renderInspector } from "./views/inspector.js";
import { bindPopovers } from "./views/popover.js";
import { bindStructuralConfirm, structuralImpact } from "./views/structural_confirm.js";
import { bindToolRail, renderToolRail } from "./views/tool_rail.js";

const appBar = document.getElementById("appBar");
const undoAction = document.getElementById("undoAction");
const redoAction = document.getElementById("redoAction");
const appShell = document.querySelector(".app-shell");
const appBusyOverlay = document.getElementById("appBusyOverlay");
const appBusyAnnouncement = document.getElementById("appBusyAnnouncement");
const appBusyTitle = document.getElementById("appBusyTitle");
const appBusyMessage = document.getElementById("appBusyMessage");
const appBusyDetail = document.getElementById("appBusyDetail");
const appAlert = document.getElementById("appAlert");
const appAlertMessage = document.getElementById("appAlertMessage");
const appAlertRetry = document.getElementById("appAlertRetry");
const appAlertDismiss = document.getElementById("appAlertDismiss");
const contextBar = document.querySelector(".context-bar");
const editorView = document.getElementById("editorView");
const reportPanel = document.getElementById("reportPanel");
const reportTitle = document.getElementById("reportTitle");
const reportStatus = document.getElementById("reportStatus");
const reportFreshness = document.getElementById("reportFreshness");
const reportRetry = document.getElementById("reportRetry");
const reportFrame = document.getElementById("reportFrame");
const svg = document.getElementById("chart");
const selectionMenu = document.getElementById("selectionMenu");
const termSelect = document.getElementById("term");
const termKind = document.getElementById("termKind");
const termEdf = document.getElementById("termEdf");
const inspectorToggle = document.getElementById("inspectorToggle");
const inspectorNode = document.getElementById("inspector");
const inspectorClose = document.getElementById("inspectorClose");
const inspectorScrim = document.getElementById("inspectorScrim");
const helpPane = document.getElementById("helpPane");
const toolRail = document.getElementById("toolRail");
const groupDisplayWrap = document.getElementById("groupDisplayWrap");
const groupDisplayMode = document.getElementById("groupDisplayMode");
const handleCountWrap = document.getElementById("handleCountWrap");
const handleCount = document.getElementById("handleCount");
const handleCountValue = document.getElementById("handleCountValue");
const basisToggle = document.getElementById("basisToggle");
const contribPlay = document.getElementById("contribPlay");
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
const structuralConfirmDialog = document.getElementById("structuralConfirmDialog");
const metricSelect = document.getElementById("metricSelect");
const metricGrid = document.getElementById("metricGrid");
const metricFreshness = document.getElementById("metricFreshness");
const metricRetry = document.getElementById("metricRetry");
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
const summaryRetry = document.getElementById("summaryRetry");
const summaryNote = document.getElementById("summaryNote");
const summaryFrame = document.getElementById("summaryFrame");
const advancedTiming = document.getElementById("advancedTiming");
const historyFrame = document.getElementById("historyFrame");
const statusNode = document.getElementById("status");
const uiPopover = document.getElementById("uiPopover");
if (!uiPopover) throw new Error("Editor popover element is missing");
bindPopovers({ root: document, popover: uiPopover });
if (!(structuralConfirmDialog instanceof HTMLDialogElement)) {
  throw new Error("Structural confirmation dialog is missing");
}
const structuralConfirm = bindStructuralConfirm(structuralConfirmDialog);

let buildProgress = null;
let buildFrame = null;
let renderedTerm = "";
let appBusyTimer = null;
let appBusyStarted = 0;
let appBusyActive = false;
let appBusyOpener = null;
let retryInProgress = false;
let retryRecovery = null;
let latestTransitionTiming = null;
let latestTimingNote = "";

const store = createEditorStore(createInitialEditorState());
const actions = createEditorActions({
  store,
  client: editorClient,
  scheduleVisibleEvidence
});
const evidenceTiming = createEvidenceTimingTracker({
  onComplete: () => renderAdvancedTiming()
});

const undo = () => actions.executeStateMutation({
  name: "undo",
  path: "/op",
  payload: { operation: "undo" }
});
const redo = () => actions.executeStateMutation({
  name: "redo",
  path: "/op",
  payload: { operation: "redo" }
});

bindAppBar({
  root: appBar,
  undoButton: undoAction,
  redoButton: redoAction,
  onView: showView,
  onUndo: undo,
  onRedo: redo
});

const chartContext = {
  svg,
  selectionMenu,
  zoomState: () => store.getState().view.zoomByTerm,
  selectedTerm,
  visualMode,
  showCi: () => store.getState().view.showCi,
  showContrib: () => store.getState().view.showContrib,
  buildProgress: () => buildProgress,
  groupDisplayMode: () => activeGroupDisplayMode()
};

let openHelp = () => inspectorToggle.click();

const narrowQuery = window.matchMedia("(max-width: 1047px)");
renderHelpDrawer(helpPane);
const inspector = bindInspector({
  root: inspectorNode,
  toggle: inspectorToggle,
  closeButton: inspectorClose,
  scrim: inspectorScrim,
  onPanelChange: (panel) => {
    actions.patchView({ inspectorPane: panel });
    const snapshot = store.getState().remote.snapshot;
    if (panel === "history" && snapshot) renderHistory(snapshot.history, historyFrame);
  },
  onOpenChange: (open) => actions.patchView({ inspectorOpen: open }),
  isOpen: () => store.getState().view.inspectorOpen,
  isNarrow: () => narrowQuery.matches,
});
openHelp = () => inspector.open("help");

function renderInspectorView() {
  const view = store.getState().view;
  renderInspector({
    root: inspectorNode,
    toggle: inspectorToggle,
    scrim: inspectorScrim,
    panel: view.inspectorPane,
    open: view.inspectorOpen,
    narrow: narrowQuery.matches,
  });
}

/** @param {MediaQueryList|MediaQueryListEvent} [event] */
function syncViewport(event = narrowQuery) {
  const focusIsInsideClosingInspector =
    event.matches && inspectorNode.contains(document.activeElement);
  if (focusIsInsideClosingInspector) {
    inspector.close({ restoreFocus: false });
    inspectorToggle.focus();
  } else {
    actions.patchView({ inspectorOpen: !event.matches });
  }
  renderInspectorView();
}

store.subscribe(
  (state) => ({ pane: state.view.inspectorPane, open: state.view.inspectorOpen }),
  renderInspectorView,
  (left, right) => left.pane === right.pane && left.open === right.open,
);
narrowQuery.addEventListener("change", syncViewport);
syncViewport();

bindToolRail({
  root: toolRail,
  onMode: (mode) => {
    const view = store.getState().view;
    const showContrib = mode === "zoom"
      ? view.showContrib
      : mode === "handles" && canShowContributions(currentTerm());
    if (view.mode === mode && view.showContrib === showContrib) return;
    stopContributionBuild();
    actions.patchView({ mode, showContrib });
  },
  onHelp: () => openHelp()
});

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

function setInteractionPreview(term, payload, selection) {
  store.update((state) => setPreviewTermState(state, term, payload, selection));
}

function clearInteractionPreview() {
  store.update((state) => {
    if (state.view.preview === null) return state;
    return patchViewState(state, { preview: null });
  });
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
  await actions.refreshEvidence("summary", "/summary", {
    source: summarySource.value
  });
}

async function refreshActiveReport() {
  const activeView = store.getState().view.activeView;
  if (activeView === "editor") return;
  await actions.refreshEvidence("report", "/report", { report: activeView });
}

function scheduleVisibleEvidence(revision, { immediate = false, summaryCommitted = false } = {}) {
  const state = store.getState();
  if (state.remote.snapshot?.model_revision !== revision) return;
  resetSummarySourceAfterInvalidatingEdit();
  if (state.view.activeView !== "editor") {
    actions.schedulePanelEvidence(
      "report",
      "/report",
      { report: state.view.activeView },
      { immediate }
    );
    return;
  }
  actions.schedulePanelEvidence(
    "metrics",
    "/metrics",
    { metric: "deviance", source: "in_force" },
    { immediate }
  );
  if (!summaryCommitted && state.view.inspectorOpen && state.view.inspectorPane === "summary") {
    actions.schedulePanelEvidence(
      "summary",
      "/summary",
      { source: summarySource.value },
      { immediate }
    );
  }
}

async function runStructuralRefit(descriptor) {
  while (true) {
    const state = store.getState();
    if (appBusyActive || state.request.mutation.status !== "idle") {
      return { ok: false, skipped: true };
    }
    const snapshot = selectSnapshot(store.getState());
    if (!snapshot) return { ok: false, skipped: true };
    const impact = structuralImpact(snapshot, descriptor);
    if (impact.requiresConfirmation && !(await structuralConfirm.confirm(impact))) {
      return { ok: false, skipped: true };
    }

    const confirmedState = store.getState();
    if (appBusyActive || confirmedState.request.mutation.status !== "idle") {
      return { ok: false, skipped: true };
    }
    if (selectSnapshot(confirmedState) === snapshot) break;
  }

  stopContributionBuild();
  if (descriptor.name !== "restore collapsed levels") {
    summarySource.value = "selected";
  }
  const operationStart = performance.now();
  const requestStart = performance.now();
  const milestones = {
    operationStart,
    requestStart,
    requestEnd: requestStart,
    commitEnd: requestStart,
    paintEnd: requestStart
  };
  const result = await actions.executeStructuralMutation({
    ...descriptor,
    onRequestSettled: () => {
      milestones.requestEnd = performance.now();
    },
    onPrimaryCommitted: () => {
      milestones.commitEnd = performance.now();
    },
    onPaintSettled: () => {
      milestones.paintEnd = performance.now();
    }
  });
  if (!result.ok) return null;
  const envelope = result.envelope;
  const timing = clientTransitionTiming(envelope, milestones);
  showTimingStatus(envelope.summary, timing);
  return envelope;
}

function setAppBusy(active, title = "Working...", detail = "") {
  if (!appShell || !appBusyOverlay) return;
  const starting = active && !appBusyActive;
  const stopping = !active && appBusyActive;
  if (starting) {
    const focused = document.activeElement;
    appBusyOpener = focused instanceof Element &&
      focused !== document.body &&
      typeof focused.focus === "function"
      ? focused
      : null;
  }
  if (appBusyTimer !== null) {
    clearInterval(appBusyTimer);
    appBusyTimer = null;
  }
  for (const region of [appBar, contextBar, editorView, reportPanel]) {
    if (region) region.toggleAttribute("inert", active);
  }
  appBusyActive = active;
  appShell.classList.toggle("is-busy", active);
  appShell.setAttribute("aria-busy", String(active));
  appBusyOverlay.hidden = !active;
  if (!active) {
    const opener = appBusyOpener;
    appBusyOpener = null;
    if (stopping) restoreFocusAfterBusy(opener);
    return;
  }
  const message = detail || "Refitting model";
  if (starting) {
    if (appBusyTitle) appBusyTitle.textContent = title;
    if (appBusyMessage) appBusyMessage.textContent = message;
    if (appBusyAnnouncement) appBusyAnnouncement.focus({ preventScroll: true });
  }
  appBusyStarted = performance.now();
  const update = () => {
    const elapsed = performance.now() - appBusyStarted;
    if (appBusyDetail) {
      appBusyDetail.textContent = `${formatMilliseconds(elapsed)} elapsed`;
    }
  };
  update();
  appBusyTimer = window.setInterval(update, 250);
}

function restoreFocusAfterBusy(opener) {
  for (const candidate of [opener, termSelect, inspectorToggle]) {
    if (!candidate || !candidate.isConnected || typeof candidate.focus !== "function") continue;
    candidate.focus({ preventScroll: true });
    if (document.activeElement === candidate) return;
  }
}

if (new URLSearchParams(window.location.search).get("test") === "1") {
  window.__superglmTest = Object.freeze({ setAppBusy });
}

function showTimingStatus(payload, timing) {
  if (!timing) return;
  latestTransitionTiming = timing;
  latestTimingNote = payload.note || "";
  if (summaryStatus) {
    summaryStatus.textContent = `Refit completed in ${formatMilliseconds(timing.client_total_ms)}`;
  }
  renderAdvancedTiming();
  if (summaryNote) summaryNote.textContent = payload.note || "";
}

function renderAdvancedTiming() {
  if (!advancedTiming) return;
  const sections = [];
  if (latestTransitionTiming) sections.push(formatTimingDetails(latestTransitionTiming));
  const evidenceDetails = formatEvidenceTimingDetails(evidenceTiming.durations());
  if (evidenceDetails) sections.push(evidenceDetails);
  const details = sections.filter(Boolean).join(" · ");
  advancedTiming.textContent = latestTimingNote && details
    ? `${latestTimingNote} · ${details}`
    : latestTimingNote || details;
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
  if (Number.isFinite(Number(timing.client_request_ms))) {
    parts.push(`request ${formatMilliseconds(timing.client_request_ms)}`);
  }
  if (Number.isFinite(Number(timing.client_commit_ms))) {
    parts.push(`DOM commit ${formatMilliseconds(timing.client_commit_ms)}`);
  }
  if (Number.isFinite(Number(timing.client_paint_ms))) {
    parts.push(`paint ${formatMilliseconds(timing.client_paint_ms)}`);
  }
  return parts.length ? `Timing: ${parts.join(", ")}` : "";
}

function formatEvidenceTimingDetails(durations) {
  const parts = Object.entries(durations).map(
    ([panel, duration]) => `${panel} ${formatMilliseconds(duration)}`
  );
  return parts.length ? `Evidence: ${parts.join(", ")}` : "";
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
}

function render() {
  const editorState = store.getState();
  const snapshot = editorState.remote.snapshot;
  if (!snapshot) return;
  const view = editorState.view;
  renderAppView(view.activeView);
  renderMetricsEvidence(editorState.request.evidence.metrics);
  renderReportEvidence(editorState.request.evidence.report, view.activeView);
  renderHistory(snapshot.history, historyFrame);
  ciToggle.style.background = view.showCi ? "#dbeafe" : "#f6f8fa";
  ciToggle.setAttribute("aria-pressed", String(view.showCi));
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
    stopContributionBuild();
  }
  if (applyTermDefaults(term)) return;
  const selection = view.preview && view.preview.term === selected
    ? new Set(view.preview.selection)
    : currentSelection();
  statusNode.style.color = "";
  if (updateHandleCount(term)) return;
  renderToolRail(toolRail, {
    mode: view.mode,
    handlesAvailable: Boolean(term.controls)
  });
  updateGroupDisplayControl(term);
  updateCollapseAction(term, selection);
  updateResetOrderAction(term);
  drawChart(term, selection, chartContext);
  const collapsedOriginalNote = selectionContextNote(term);
  renderAppBar({
    root: appBar,
    activeView: view.activeView,
    undoButton: undoAction,
    redoButton: redoAction,
    canUndo: Boolean(snapshot.history.active.length),
    canRedo: Boolean(snapshot.history.redo.length)
  });
  renderContextBar(
    { kindNode: termKind, edfNode: termEdf, statusNode },
    { name: selected, term, selectionSize: selection.size, note: collapsedOriginalNote }
  );
}

function selectionContextNote(term) {
  return activeGroupDisplayMode() === "collapsed" &&
    term.group_display &&
    term.group_display.available
    ? "original line is grouped by exposure-weighted averaging"
    : "";
}

function renderSelectionState({ termName, indices }) {
  if (termName !== selectedTerm()) return;
  const term = currentTerm();
  if (!term) return;
  const selection = new Set(indices);
  updateChartSelection(term, selection, chartContext);
  updateCollapseAction(term, selection);
  renderContextBar(
    { kindNode: termKind, edfNode: termEdf, statusNode },
    {
      name: termName,
      term,
      selectionSize: selection.size,
      note: selectionContextNote(term)
    }
  );
}

function selectSelectionState(state) {
  return {
    termName: selectActiveTermName(state),
    indices: selectCurrentSelection(state)
  };
}

function sameSelectionState(next, previous) {
  if (next.termName !== previous.termName || next.indices.length !== previous.indices.length) {
    return false;
  }
  return next.indices.every((value, index) => value === previous.indices[index]);
}

function selectChartBearingState(state) {
  const snapshot = selectSnapshot(state);
  if (!snapshot) return null;
  return {
    snapshot,
    modelRevision: snapshot.model_revision,
    selectedTerm: snapshot.selected_term,
    terms: snapshot.terms,
    history: snapshot.history,
    canUncollapseLevels: snapshot.can_uncollapse_levels,
    lastCollapse: snapshot.last_collapse
  };
}

function sameChartBearingState(next, previous) {
  if (next === null || previous === null) return next === previous;
  return next.modelRevision === previous.modelRevision &&
    next.selectedTerm === previous.selectedTerm &&
    next.terms === previous.terms &&
    next.history === previous.history &&
    next.canUncollapseLevels === previous.canUncollapseLevels &&
    next.lastCollapse === previous.lastCollapse;
}

function sameViewOutsidePreview(next, previous) {
  return next.activeTerm === previous.activeTerm &&
    next.activeView === previous.activeView &&
    next.mode === previous.mode &&
    next.showCi === previous.showCi &&
    next.showContrib === previous.showContrib &&
    next.zoomByTerm === previous.zoomByTerm &&
    next.groupModeByTerm === previous.groupModeByTerm;
}

function renderMutationBusy(mutation) {
  const active = mutation.status === "running";
  setAppBusy(active, mutation.operation || "Working...", "Starting...");
}

function renderInteractionPreview(preview) {
  if (!preview || preview.term !== selectedTerm()) return;
  drawChart(preview.payload, new Set(preview.selection), chartContext);
}

function renderInteractionState(current, previous) {
  if (current.preview) {
    renderInteractionPreview(current.preview);
    return;
  }
  // A confirmed remote commit is rendered by the remote subscription. When
  // only the private preview is cleared (cancel or failed request), repaint
  // from the unchanged authoritative snapshot.
  if (!previous.preview || current.snapshot !== previous.snapshot) return;
  const term = currentTerm();
  if (term) drawChart(term, currentSelection(), chartContext);
}

function sameInteractionState(next, previous) {
  return next.preview === previous.preview && next.snapshot === previous.snapshot;
}

function renderRecovery(recovery) {
  if (!appAlert || !appAlertMessage || !appAlertRetry || !appAlertDismiss) return;
  const visibleRecovery = recovery || (retryInProgress ? retryRecovery : null);
  if (!visibleRecovery) {
    appAlert.hidden = true;
    appAlertMessage.textContent = "";
    appAlertRetry.hidden = false;
    appAlertRetry.disabled = false;
    appAlertDismiss.disabled = false;
    return;
  }
  appAlertMessage.textContent = visibleRecovery.message;
  appAlertRetry.hidden = !visibleRecovery.retry;
  appAlertRetry.disabled = retryInProgress || !visibleRecovery.retry;
  appAlertDismiss.disabled = retryInProgress;
  appAlert.hidden = false;
}

async function retryFailedMutation() {
  if (retryInProgress) return;
  const recovery = store.getState().request.recovery;
  if (!recovery || !recovery.retry) return;
  retryRecovery = recovery;
  retryInProgress = true;
  renderRecovery(null);
  try {
    await actions.retryMutation();
  } finally {
    retryInProgress = false;
    retryRecovery = null;
    renderRecovery(store.getState().request.recovery);
  }
}

function renderMetricsEvidence(evidence) {
  const busy = evidence.status === "updating";
  metricGrid.setAttribute("aria-busy", busy ? "true" : "false");
  metricGrid.dataset.freshness = evidence.status;
  renderMetricGrid(evidence.payload, { metricGrid, metricSelect });
  if ((evidence.status === "error" || evidence.status === "stale") && evidence.payload === null) {
    metricGrid.textContent = evidence.error || "Metric unavailable.";
  }
  renderEvidenceFreshness(evidence, {
    statusNode: metricFreshness,
    retryButton: metricRetry,
    loading: "Loading metrics...",
    updating: "Updating metrics...",
    stale: "Metrics may be stale.",
    error: "Metrics unavailable."
  });
}

function renderReportEvidence(evidence, activeView) {
  const busy = evidence.status === "updating";
  reportFrame.setAttribute("aria-busy", busy ? "true" : "false");
  reportFrame.dataset.freshness = evidence.status;
  if (activeView === "editor") return;
  const payloadMatchesView = evidence.payload !== null && evidence.payload.report === activeView;
  if (payloadMatchesView) {
    renderReport(evidence.payload, { reportTitle, reportStatus, reportFrame });
  } else {
    reportTitle.textContent = activeView === "final" ? "Final Fit Report" : "Validation Report";
    reportFrame.innerHTML = "";
  }
  if (!payloadMatchesView && evidence.status !== "error" && evidence.status !== "stale") {
    reportStatus.textContent = "Loading report...";
  }
  renderEvidenceFreshness(evidence, {
    statusNode: reportFreshness,
    retryButton: reportRetry,
    loading: "Loading report...",
    updating: "Updating report...",
    stale: "Report may be stale.",
    error: "Report unavailable."
  });
}

function renderSummaryEvidence(evidence) {
  const state = store.getState();
  const revision = state.remote.snapshot?.model_revision;
  const evidenceMatchesRevision = evidence.revision === revision;
  const payload = evidenceMatchesRevision && evidence.payload !== null
    ? evidence.payload
    : state.remote.summary;
  const effectiveEvidence = evidenceMatchesRevision
    ? evidence
    : { ...evidence, status: "current", error: null };
  summaryFrame.setAttribute(
    "aria-busy",
    effectiveEvidence.status === "updating" ? "true" : "false"
  );
  summaryFrame.dataset.freshness = effectiveEvidence.status;
  if (payload !== null) {
    renderSummary(payload, summaryNodes());
  } else if (effectiveEvidence.status === "error" || effectiveEvidence.status === "stale") {
    summaryStatus.textContent = effectiveEvidence.error || "Summary unavailable.";
  }
  const summaryLabel = summaryStatus.textContent || "Summary";
  renderEvidenceFreshness(effectiveEvidence, {
    statusNode: summaryStatus,
    retryButton: summaryRetry,
    loading: "Loading summary...",
    updating: `${summaryLabel} · Updating...`,
    stale: `${summaryLabel} · Summary may be stale.`,
    error: "Summary unavailable."
  });
}

function renderEvidenceFreshness(evidence, options) {
  const { statusNode, retryButton, loading, updating, stale, error } = options;
  const retryable = evidence.status === "stale" || evidence.status === "error";
  if (retryButton) retryButton.hidden = !retryable;
  if (!statusNode) return;
  statusNode.dataset.freshness = evidence.status;
  if (evidence.status === "idle" && evidence.payload === null) {
    statusNode.textContent = loading;
  } else if (evidence.status === "updating") {
    statusNode.textContent = updating;
  } else if (evidence.status === "stale") {
    statusNode.textContent = evidence.error || stale;
  } else if (evidence.status === "error") {
    statusNode.textContent = evidence.error || error;
  } else if (statusNode !== summaryStatus) {
    statusNode.textContent = "";
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
  if (!term.controls) {
    if (view.mode === "handles") patch.mode = "select";
    if (view.showContrib) patch.showContrib = false;
  } else if (!canShowContributions(term) && view.showContrib) {
    patch.showContrib = false;
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
  clearPreviewTerm: clearInteractionPreview,
  setZoom,
  clearZoom,
  actions,
});

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
    void retryFailedMutation();
  });
}
if (appAlertDismiss) {
  appAlertDismiss.addEventListener("click", () => actions.dismissRecovery());
}
if (metricRetry) {
  metricRetry.addEventListener("click", () => { void actions.retryEvidence("metrics"); });
}
if (summaryRetry) {
  summaryRetry.addEventListener("click", () => { void actions.retryEvidence("summary"); });
}
if (reportRetry) {
  reportRetry.addEventListener("click", () => { void actions.retryEvidence("report"); });
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
    await runStructuralRefit(collapseTransition(selectedTerm()));
  });
}
if (ungroupLevels) {
  ungroupLevels.addEventListener("click", async () => {
    await runStructuralRefit(ungroupTransition(selectedTerm()));
  });
}
if (uncollapseLevels) {
  uncollapseLevels.addEventListener("click", async () => {
    await runStructuralRefit(uncollapseTransition());
  });
}

store.subscribe(selectChartBearingState, (chartState) => {
  if (chartState) {
    svg.dataset.modelRevision = String(chartState.modelRevision);
    summaryFrame.dataset.modelRevision = String(chartState.modelRevision);
  }
  render();
}, sameChartBearingState);
store.subscribe(
  (state) => state.remote.summary,
  (summary) => {
    if (summary) {
      summarySource.value = "selected";
    }
    renderSummaryEvidence(store.getState().request.evidence.summary);
  }
);
store.subscribe(
  (state) => state.request.evidence.summary,
  (evidence, previous) => {
    renderSummaryEvidence(evidence);
    evidenceTiming.observe("summary", evidence, previous);
  }
);
store.subscribe((state) => state.view, () => render(), sameViewOutsidePreview);
store.subscribe(
  (state) => ({ preview: state.view.preview, snapshot: state.remote.snapshot }),
  renderInteractionState,
  sameInteractionState,
);
store.subscribe(selectSelectionState, renderSelectionState, sameSelectionState);
store.subscribe((state) => state.request.recovery, renderRecovery);
store.subscribe((state) => state.request.mutation, renderMutationBusy);
store.subscribe(
  (state) => state.request.evidence.metrics,
  (evidence, previous) => {
    renderMetricsEvidence(evidence);
    evidenceTiming.observe("metrics", evidence, previous);
  }
);
store.subscribe(
  (state) => state.request.evidence.report,
  (evidence, previous) => {
    renderReportEvidence(evidence, store.getState().view.activeView);
    evidenceTiming.observe("report", evidence, previous);
  }
);

loadState().then(async () => {
  await refreshMetricsView();
  await refreshSummaryView();
}).catch((error) => {
  statusNode.textContent = error.message;
  statusNode.style.color = "#b42318";
});
