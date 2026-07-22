import { editorClient } from "./api/client.js";
import { drawChart, groupedTerms, updateChartSelection } from "./chart.js";
import { renderHistory } from "./history.js";
import { renderMetricGrid } from "./metrics.js";
import { renderReport } from "./reports.js";
import { createEditorActions } from "./state/actions.js";
import {
  selectActiveTermName,
  selectCurrentSelection,
  selectEvidenceNeedsRefresh,
  selectGroupDisplayMode,
  selectRenderableTerm,
  selectSnapshot,
  selectSummaryLevelDisplay,
  selectVisibleEvidencePanels
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
import { bindExportDialog } from "./views/export_dialog.js";
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
const exportAction = document.getElementById("exportAction");
const exportDialog = document.getElementById("exportDialog");
const exportDialogClose = document.getElementById("exportDialogClose");
const exportDirectory = document.getElementById("exportDirectory");
const exportOpenDirectory = document.getElementById("exportOpenDirectory");
const exportFilename = document.getElementById("exportFilename");
const exportSave = document.getElementById("exportSave");
const exportDownload = document.getElementById("exportDownload");
const exportStatus = document.getElementById("exportStatus");
const exportFormatInputs = [...document.querySelectorAll('input[name="exportFormat"]')];
const collapseLevels = document.getElementById("collapseLevels");
const ungroupLevels = document.getElementById("ungroupLevels");
const uncollapseLevels = document.getElementById("uncollapseLevels");
const structuralConfirmDialog = document.getElementById("structuralConfirmDialog");
const metricSelect = document.getElementById("metricSelect");
const metricGrid = document.getElementById("metricGrid");
const metricFreshness = document.getElementById("metricFreshness");
const metricRetry = document.getElementById("metricRetry");
const summarySource = document.getElementById("summarySource");
const summaryLevelDisplayInputs = /** @type {HTMLInputElement[]} */ (
  [...document.querySelectorAll('input[name="summaryLevelDisplay"]')]
);
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
    scheduleVisibleEvidenceCatchUp();
  },
  onOpenChange: (open) => {
    actions.patchView({ inspectorOpen: open });
    if (open) scheduleVisibleEvidenceCatchUp();
  },
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
    const open = !event.matches;
    actions.patchView({ inspectorOpen: open });
    if (open) scheduleVisibleEvidenceCatchUp();
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
    get summaryLevelDisplay() {
      return selectSummaryLevelDisplay(store.getState());
    },
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

async function runProfileFromDialog() {
  if (!profileDialog) return;
  const parameter = profileDialog.dataset.parameter || "tweedie_p";
  stopContributionBuild();
  summarySource.value = "selected";
  await runDistributionProfile(summaryNodes(), parameter, async () => {
    const snapshot = await actions.initialize();
    scheduleVisibleEvidence(snapshot.model_revision, { immediate: true });
  });
}

async function saveBlobToFile(blob, filename, fileType) {
  if (typeof window.showSaveFilePicker === "function" && window.isSecureContext) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [{ description: fileType.description, accept: fileType.accept }]
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return `Saved ${filename}`;
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") return null;
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

if (
  !(exportAction instanceof HTMLElement) ||
  !(exportDialog instanceof HTMLDialogElement) ||
  !(exportDialogClose instanceof HTMLElement) ||
  !(exportDirectory instanceof HTMLInputElement) ||
  !(exportFilename instanceof HTMLInputElement) ||
  !(exportSave instanceof HTMLButtonElement) ||
  !(exportDownload instanceof HTMLButtonElement) ||
  !(exportStatus instanceof HTMLElement) ||
  !exportFormatInputs.every((input) => input instanceof HTMLInputElement)
) {
  throw new Error("Editor export dialog is incomplete");
}
bindExportDialog({
  client: editorClient,
  nodes: {
    action: exportAction,
    dialog: exportDialog,
    close: exportDialogClose,
    formatInputs: exportFormatInputs,
    filename: exportFilename,
    directory: exportDirectory,
    download: exportDownload,
    saveToKernel: exportSave,
    openDirectory: exportOpenDirectory instanceof HTMLButtonElement
      ? exportOpenDirectory
      : null,
    status: exportStatus
  },
  saveBlobToFile
});

async function refreshMetricsView() {
  await actions.refreshEvidence("metrics", "/metrics", {
    metric: "deviance",
    source: "in_force"
  });
}

function summaryRequestPayload() {
  return {
    source: summarySource.value,
    level_display: selectSummaryLevelDisplay(store.getState())
  };
}

async function refreshSummaryView() {
  await actions.refreshEvidence("summary", "/summary", summaryRequestPayload());
}

async function refreshActiveReport() {
  const activeView = store.getState().view.activeView;
  if (activeView === "editor") return;
  await actions.refreshEvidence("report", "/report", { report: activeView });
}

function scheduleVisibleEvidence(
  revision,
  { immediate = false, summaryCommitted = false, onlyStale = false } = {}
) {
  const state = store.getState();
  if (state.remote.snapshot?.model_revision !== revision) return;
  if (!onlyStale) resetSummarySourceAfterInvalidatingEdit();
  for (const panel of selectVisibleEvidencePanels(state, { summaryCommitted })) {
    if (onlyStale && !selectEvidenceNeedsRefresh(state, panel)) continue;
    if (panel === "report") {
      actions.schedulePanelEvidence(
        panel,
        "/report",
        { report: state.view.activeView },
        { immediate }
      );
    } else if (panel === "metrics") {
      actions.schedulePanelEvidence(
        panel,
        "/metrics",
        { metric: "deviance", source: "in_force" },
        { immediate }
      );
    } else {
      actions.schedulePanelEvidence(
        panel,
        "/summary",
        summaryRequestPayload(),
        { immediate }
      );
    }
  }
}

function scheduleVisibleEvidenceCatchUp() {
  const revision = store.getState().remote.snapshot?.model_revision;
  if (revision === undefined) return;
  scheduleVisibleEvidence(revision, { immediate: true, onlyStale: true });
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
    payload: {
      ...descriptor.payload,
      level_display: selectSummaryLevelDisplay(store.getState())
    },
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
  if (activeView === "editor") {
    scheduleVisibleEvidenceCatchUp();
    return;
  }
  await refreshActiveReport();
}

function renderAppView(activeView) {
  editorView.hidden = activeView !== "editor";
  reportPanel.hidden = activeView === "editor";
}

function renderChartWorkspace() {
  const editorState = store.getState();
  const snapshot = editorState.remote.snapshot;
  if (!snapshot) return;
  const view = editorState.view;
  ciToggle.style.background = view.showCi ? "#dbeafe" : "#f6f8fa";
  ciToggle.setAttribute("aria-pressed", String(view.showCi));
  const selected = selectedTerm();
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
  renderContextBar(
    { kindNode: termKind, edfNode: termEdf, statusNode },
    { name: selected, term, selectionSize: selection.size, note: collapsedOriginalNote }
  );
}

function renderChartOnly() {
  const state = store.getState();
  const term = currentTerm();
  if (!state.remote.snapshot || !term) return;
  const selection = state.view.preview && state.view.preview.term === selectedTerm()
    ? new Set(state.view.preview.selection)
    : currentSelection();
  drawChart(term, selection, chartContext);
}

function termCatalogueKey(terms) {
  return groupedTerms(terms).map(
    ([group, names]) => `${group}\u0000${names.join("\u0000")}`
  ).join("\u0001");
}

function selectTermPickerRenderState(state) {
  const snapshot = selectSnapshot(state);
  return {
    ready: snapshot !== null,
    catalogueKey: snapshot ? termCatalogueKey(snapshot.terms || {}) : "",
    activeTerm: selectActiveTermName(state)
  };
}

function sameTermPickerRenderState(next, previous) {
  return next.ready === previous.ready &&
    next.catalogueKey === previous.catalogueKey &&
    next.activeTerm === previous.activeTerm;
}

function renderTermPickerState(next, previous) {
  const snapshot = selectSnapshot(store.getState());
  if (!snapshot) return;
  if (!previous.ready || next.catalogueKey !== previous.catalogueKey) {
    termSelect.innerHTML = "";
    for (const [group, names] of groupedTerms(snapshot.terms || {})) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = group;
      for (const name of names) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        optgroup.appendChild(option);
      }
      termSelect.appendChild(optgroup);
    }
  }
  if (termSelect.value !== next.activeTerm) termSelect.value = next.activeTerm;
}

function selectChartRenderState(state) {
  const activeTerm = selectActiveTermName(state);
  const view = state.view;
  return {
    ready: state.remote.snapshot !== null,
    chartEpoch: state.remote.chartEpoch,
    activeTerm,
    mode: view.mode,
    showCi: view.showCi,
    showContrib: view.showContrib,
    zoom: view.zoomByTerm[activeTerm] || null,
    groupMode: Object.prototype.hasOwnProperty.call(view.groupModeByTerm, activeTerm)
      ? view.groupModeByTerm[activeTerm]
      : null
  };
}

function sameChartRenderState(next, previous) {
  return next.ready === previous.ready &&
    next.chartEpoch === previous.chartEpoch &&
    next.activeTerm === previous.activeTerm &&
    next.mode === previous.mode &&
    next.showCi === previous.showCi &&
    next.showContrib === previous.showContrib &&
    next.zoom === previous.zoom &&
    next.groupMode === previous.groupMode;
}

function selectHistoryRenderState(state) {
  const history = state.remote.snapshot?.history || null;
  return { history, key: history ? JSON.stringify(history) : "" };
}

function sameHistoryRenderState(next, previous) {
  return next.key === previous.key;
}

function renderHistoryState({ history }) {
  if (history) renderHistory(history, historyFrame);
}

function selectAppBarRenderState(state) {
  const snapshot = state.remote.snapshot;
  const selectedTerm = snapshot?.selected_term;
  return {
    ready: snapshot !== null,
    activeView: state.view.activeView,
    canUndo: Boolean(
      selectedTerm && snapshot?.history.active.some((record) => record.term === selectedTerm)
    ),
    canRedo: Boolean(
      selectedTerm && snapshot?.history.redo.some((record) => record.term === selectedTerm)
    )
  };
}

function sameAppBarRenderState(next, previous) {
  return next.ready === previous.ready &&
    next.activeView === previous.activeView &&
    next.canUndo === previous.canUndo &&
    next.canRedo === previous.canRedo;
}

function renderAppBarState(state) {
  if (!state.ready) return;
  renderAppBar({
    root: appBar,
    activeView: state.activeView,
    undoButton: undoAction,
    redoButton: redoAction,
    canUndo: state.canUndo,
    canRedo: state.canRedo
  });
}

function selectActiveViewRenderState(state) {
  return { ready: state.remote.snapshot !== null, activeView: state.view.activeView };
}

function sameActiveViewRenderState(next, previous) {
  return next.ready === previous.ready && next.activeView === previous.activeView;
}

function renderActiveViewState({ ready, activeView }) {
  if (!ready) return;
  renderAppView(activeView);
  renderReportEvidence(store.getState().request.evidence.report, activeView);
}

function renderSnapshotRevision(revision) {
  if (revision === null) return;
  svg.dataset.modelRevision = String(revision);
  summaryFrame.dataset.modelRevision = String(revision);
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
  const termName = selectActiveTermName(state);
  const impact = selectSnapshot(state)?.terms[termName]?.impact || {};
  return {
    termName,
    indices: selectCurrentSelection(state),
    weightedMeanRelativity: impact.weighted_mean_relativity,
    selectedWeightShare: impact.selected_weight_share
  };
}

function sameSelectionState(next, previous) {
  if (
    next.termName !== previous.termName ||
    next.weightedMeanRelativity !== previous.weightedMeanRelativity ||
    next.selectedWeightShare !== previous.selectedWeightShare ||
    next.indices.length !== previous.indices.length
  ) {
    return false;
  }
  return next.indices.every((value, index) => value === previous.indices[index]);
}

function renderMutationBusy(mutation) {
  const active = mutation.status === "running" && mutation.blocking === true;
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

function renderSummaryEvidence(evidence, previous = null) {
  const state = store.getState();
  const revision = state.remote.snapshot?.model_revision;
  const evidenceMatchesRevision = evidence.revision === revision;
  const retainedPayloadIsFromPriorRevision = evidence.status === "updating" &&
    previous !== null && previous.revision !== evidence.revision;
  const payloadMatchesLevelDisplay = evidence.payload === null ||
    (evidence.payload.level_display || "expanded") === selectSummaryLevelDisplay(state);
  const payload = evidenceMatchesRevision &&
    !retainedPayloadIsFromPriorRevision &&
    payloadMatchesLevelDisplay
    ? evidence.payload
    : null;
  const retainedPayloadIsFromOtherLevelDisplay = evidenceMatchesRevision &&
    !retainedPayloadIsFromPriorRevision &&
    evidence.status === "updating" &&
    evidence.payload !== null &&
    !payloadMatchesLevelDisplay;
  const effectiveEvidence = !evidenceMatchesRevision
    ? { ...evidence, status: "stale", error: null }
    : retainedPayloadIsFromOtherLevelDisplay
      ? { ...evidence, status: "updating", error: null }
      : evidence;
  summaryFrame.setAttribute(
    "aria-busy",
    effectiveEvidence.status === "updating" ? "true" : "false"
  );
  summaryFrame.dataset.freshness = effectiveEvidence.status;
  if (payload !== null) {
    renderSummary(payload, summaryNodes());
  } else if (retainedPayloadIsFromOtherLevelDisplay) {
    summaryFrame.innerHTML = "";
    summaryStatus.textContent = "Updating summary...";
  } else if (
    summaryFrame.innerHTML.trim().length === 0 &&
    (effectiveEvidence.status === "error" || effectiveEvidence.status === "stale")
  ) {
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
    if (progress >= 1) {
      buildFrame = null;
      contribPlay.disabled = false;
    }
    renderChartOnly();
    if (progress < 1) {
      buildFrame = requestAnimationFrame(step);
    }
  };
  buildProgress = initialProgress;
  buildFrame = requestAnimationFrame(step);
  contribPlay.disabled = true;
  renderChartOnly();
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
    contribPlay.disabled = false;
    renderChartOnly();
  }
  return true;
}

function stopContributionBuild() {
  if (buildFrame !== null) {
    cancelAnimationFrame(buildFrame);
    buildFrame = null;
  }
  buildProgress = null;
  contribPlay.disabled = false;
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
    termSelect.value = authoritativeTerm;
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

for (const input of summaryLevelDisplayInputs) {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    actions.patchView({ summaryLevelDisplay: input.value });
    void refreshSummaryView();
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
    if (operation === "select_all") {
      const term = currentTerm();
      if (!term) return;
      await actions.executeSelectionMutation({
        term: selectedTerm(),
        indices: term.x.map((_, index) => index)
      });
      return;
    }
    stopContributionBuild();
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

store.subscribe(selectChartRenderState, () => renderChartWorkspace(), sameChartRenderState);
store.subscribe(
  selectTermPickerRenderState,
  renderTermPickerState,
  sameTermPickerRenderState
);
store.subscribe(selectHistoryRenderState, renderHistoryState, sameHistoryRenderState);
store.subscribe(selectAppBarRenderState, renderAppBarState, sameAppBarRenderState);
store.subscribe(
  selectActiveViewRenderState,
  renderActiveViewState,
  sameActiveViewRenderState
);
store.subscribe(
  (state) => state.remote.snapshot?.model_revision ?? null,
  renderSnapshotRevision
);
store.subscribe(
  (state) => state.remote.summary,
  (summary) => {
    if (summary) {
      summarySource.value = "selected";
    }
    renderSummaryEvidence(store.getState().request.evidence.summary);
  }
);
store.subscribe(selectSummaryLevelDisplay, (levelDisplay) => {
  for (const input of summaryLevelDisplayInputs) {
    input.checked = input.value === levelDisplay;
  }
  renderSummaryEvidence(store.getState().request.evidence.summary);
});
store.subscribe(
  (state) => state.request.evidence.summary,
  (evidence, previous) => {
    renderSummaryEvidence(evidence, previous);
    evidenceTiming.observe("summary", evidence, previous);
  }
);
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

for (const input of summaryLevelDisplayInputs) {
  input.checked = input.value === selectSummaryLevelDisplay(store.getState());
}

loadState().then(async () => {
  await refreshMetricsView();
  await refreshSummaryView();
}).catch((error) => {
  statusNode.textContent = error.message;
  statusNode.style.color = "#b42318";
});
