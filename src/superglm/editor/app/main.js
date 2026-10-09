import { editorClient } from "./api/client.js";
import {
  bindPointLens,
  drawChart,
  groupedTerms,
  selectionSpanRange,
  updateChartSelection
} from "./chart.js";
import { bindDragWatch } from "./chart/anchor_marks.js";
import { chartSize } from "./chart/geometry.js";
import { bindHistory, renderHistory } from "./history.js";
import { renderMetricGrid } from "./metrics.js";
import { renderReport } from "./reports.js";
import { shapeButtonState, shapeRangeForSelection } from "./shapes.js";
import { createEditorActions } from "./state/actions.js";
import {
  selectActiveTermName,
  selectCurrentSelection,
  selectEvidenceNeedsRefresh,
  selectGroupDisplayMode,
  selectModelRevision,
  selectPendingSteps,
  selectRenderableTerm,
  selectSnapshot,
  selectSummaryLevelDisplay,
  selectVisibleEvidencePanels,
  selectWaitingTerms
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
  applySummaryView,
  refitAtOnceTransition,
  refitPendingTransition,
  renderSummary,
  runDistributionProfile,
  showDistributionProfileDialog,
  runOffsetRefit,
  revertTransition,
  stageCollapse,
  stageOnCurve,
  stageReference,
  stageShapeRange,
  stageSpecial,
  stageUngroup
} from "./summary.js";
import { freeLevelsShown, specialActions } from "./specials.js";
import { CLICK_SLOP, bindInteractions } from "./interactions.js";
import { bindAppBar, renderAppBar, revertAvailable } from "./views/app_bar.js";
import {
  placeTermViewToggle,
  renderContextBar,
  renderNewLevelsControl
} from "./views/context_bar.js";
import { createCVTab } from "./views/cv_tab.js";
import { bindExportDialog } from "./views/export_dialog.js";
import {
  bindFeatureList,
  readFeatureListOpen,
  renderFeatureList,
  storeFeatureListOpen
} from "./views/feature_list.js";
import { renderHelpDrawer } from "./views/help_drawer.js";
import { bindInspector, renderInspector } from "./views/inspector.js";
import {
  bindJoinToggle,
  effectiveShapeJoin,
  readShapeJoin,
  renderJoinToggle,
  storeShapeJoin
} from "./views/join_toggle.js";
import { bindPopovers } from "./views/popover.js";
import {
  bindSettingsPane,
  loadSettings,
  onSettingsChange,
  renderSettingsPane,
  saveSettings
} from "./views/settings.js";
import {
  bindSummaryFilter,
  bindSummarySearch,
  bindSummarySections,
  renderSummaryFilter,
  waitingCounts
} from "./views/summary_view.js";
import { mountThemeSwitch } from "./views/theme.js";
import {
  RATING_TABLE_FAILED,
  RATING_TABLE_LOADING,
  bindTermViewToggle,
  ratingTableMessage,
  ratingTableModel,
  renderRatingTable,
  renderTermViewToggle
} from "./views/rating_table.js";
import { bindToolRail, renderToolRail } from "./views/tool_rail.js";

const appBar = document.getElementById("appBar");
const undoAction = document.getElementById("undoAction");
const redoAction = document.getElementById("redoAction");
const revertAction = document.getElementById("revertAction");
const refreshAction = document.getElementById("refreshAction");
const refitPendingAction = document.getElementById("refitPendingAction");
const refitPendingCount = document.getElementById("refitPendingCount");
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
const plotColumn = document.querySelector(".plot-column");
const termViewToggle = document.getElementById("termViewToggle");
const ratingTableFrame = document.getElementById("ratingTableFrame");
const featureListNodes = Object.freeze({
  root: document.getElementById("featureList"),
  search: document.getElementById("featureSearch"),
  rows: document.getElementById("featureRows"),
  toggle: document.getElementById("featureListToggle"),
  strip: document.getElementById("featureListStrip")
});
let featureQuery = "";
// Open by default only where the open list still leaves the chart 600px beside
// the inspector: 1086px plus the 192px the open list takes over its strip.
let featureListOpen = readFeatureListOpen(window.matchMedia("(min-width: 1278px)").matches);
const termNameNode = document.getElementById("termName");
const termKind = document.getElementById("termKind");
const termEdf = document.getElementById("termEdf");
const termReference = document.getElementById("termReference");
const helpAction = document.getElementById("helpAction");
const inspectorToggle = document.getElementById("inspectorToggle");
const inspectorNode = document.getElementById("inspector");
const inspectorClose = document.getElementById("inspectorClose");
const inspectorScrim = document.getElementById("inspectorScrim");
const helpPane = document.getElementById("helpPane");
const toolRail = document.getElementById("toolRail");
const groupDisplayWrap = document.getElementById("groupDisplayWrap");
const groupDisplayMode = document.getElementById("groupDisplayMode");
const newLevelsWrap = document.getElementById("newLevelsWrap");
const newLevelsMode = document.getElementById("newLevelsMode");
const handleCountWrap = document.getElementById("handleCountWrap");
const handleCount = document.getElementById("handleCount");
const handleCountValue = document.getElementById("handleCountValue");
const basisToggle = document.getElementById("basisToggle");
const contribPlay = document.getElementById("contribPlay");
const contribTools = document.getElementById("contribTools");
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
const exportPendingNote = document.getElementById("exportPendingNote");
const collapseLevels = document.getElementById("collapseLevels");
const ungroupLevels = document.getElementById("ungroupLevels");
const setReference = document.getElementById("setReference");
const makeSpecial = document.getElementById("makeSpecial");
const returnToCurve = document.getElementById("returnToCurve");
const freeLevelsToggle = document.getElementById("freeLevelsToggle");
const shapeButtons = [...document.querySelectorAll("button[data-shape-degree]")];
const shapeJoin = document.getElementById("shapeJoin");
const shapeJoinSeparator = document.getElementById("shapeJoinSeparator");
const selectionRefitBreak = document.getElementById("selectionRefitBreak");
const selectionRefitLabel = document.getElementById("selectionRefitLabel");
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
const summarySearch = document.getElementById("summarySearch");
const summarySearchCount = document.getElementById("summarySearchCount");
const summaryFilterNode = document.getElementById("summaryFilter");
const summaryHeader = document.getElementById("summaryHeader");
const summaryModelChips = document.getElementById("summaryModelChips");
const summaryTiles = document.getElementById("summaryTiles");
let summaryQuery = "";
let summaryFilter = "all";
// Sections the analyst opened or closed; cleared when the chart's term or the
// search changes, so the summary goes back to following the chart.
const summaryToggled = new Map();
const settingsTiming = document.getElementById("settingsTiming");
const settingsNodes = Object.freeze({
  root: document.getElementById("settingsPane"),
  buildDuration: document.getElementById("buildDuration"),
  buildDurationValue: document.getElementById("buildDurationValue"),
  timing: settingsTiming
});
const historyFrame = document.getElementById("historyFrame");
const statusNode = document.getElementById("status");
const uiPopover = document.getElementById("uiPopover");
if (!uiPopover) throw new Error("Editor popover element is missing");
bindPopovers({ root: document, popover: uiPopover });

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
  onComplete: () => renderTimingReadout()
});

// The Cross-validation tab draws into the report frame, which the other
// reports share, so it draws a job's progress only while it is the open view.
// A finished Run CV or Final fit changes the tab, and a Final fit also what
// Export offers; a mutation running when it publishes may hold a snapshot from
// before it, so the refresh waits for that mutation to settle.
const cvTab = createCVTab({
  frame: reportFrame,
  client: editorClient,
  onJobSettled: async (kind) => {
    if (kind === "final_fit") await actions.refreshFromPythonWhenIdle();
    await refreshActiveReport();
  },
  isShown: () => store.getState().view.activeView === "cv"
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
  revertButton: revertAction,
  refreshButton: refreshAction,
  refitButton: refitPendingAction,
  onView: showView,
  onUndo: undo,
  onRedo: redo,
  onRevert: () => runStructuralRefit(revertTransition()),
  onRefresh: refreshFromPython,
  onRefit: refitPending
});
// The DAY / NIGHT switch keeps "Follow the browser" equal to the theme key.
mountThemeSwitch({
  button: document.getElementById("themeSwitch"),
  root: document.documentElement,
  media: window.matchMedia("(prefers-color-scheme: dark)"),
  settings: { load: loadSettings, save: saveSettings, subscribe: onSettingsChange }
});

// Settings keep their choices in this browser (views/settings.js).
function renderSettingsView() {
  renderSettingsPane(settingsNodes, { settings: loadSettings() });
  // The selection menu names its structural row by what its icons do.
  selectionRefitLabel.textContent = loadSettings().refitEveryChange ? "Refit" : "Structure";
}

bindSettingsPane(settingsNodes, {
  onToggle: (key) => saveSettings({ [key]: !loadSettings()[key] }),
  onGroupsDefault: (groupsDefault) => saveSettings({ groupsDefault }),
  onBuildDuration: (buildDurationMs) => saveSettings({ buildDurationMs })
});
onSettingsChange(renderSettingsView);
renderSettingsView();

async function refreshFromPython() {
  const result = await actions.refreshFromPython();
  if (result.ok) {
    statusNode.textContent = `Synced with Python · revision ${result.snapshot.model_revision}`;
  } else if (!result.skipped) {
    statusNode.textContent = result.error.message;
    statusNode.classList.add("is-error");
  }
}

const chartContext = {
  svg,
  selectionMenu,
  zoomState: () => store.getState().view.zoomByTerm,
  selectedTerm,
  visualMode,
  showCi: () => store.getState().view.showCi,
  freeLevels: () => shownFreeLevels(),
  showContrib: () => store.getState().view.showContrib,
  buildProgress: () => buildProgress,
  groupDisplayMode: () => activeGroupDisplayMode(),
  selectionAnchor: () => store.getState().view.selectionAnchor,
  selectionSpan: () => store.getState().view.selectionSpan
};

let openHelp = () => inspectorToggle.click();

// The inspector sits beside the chart only where a 600px chart, the tool rail,
// the collapsed feature strip and the inspector all fit.
const narrowQuery = window.matchMedia("(max-width: 1085px)");
renderHelpDrawer(helpPane);
const inspector = bindInspector({
  root: inspectorNode,
  toggle: inspectorToggle,
  closeButton: inspectorClose,
  scrim: inspectorScrim,
  onPanelChange: (panel) => {
    actions.patchView({ inspectorPane: panel });
    const snapshot = store.getState().remote.snapshot;
    if (panel === "history" && snapshot) renderHistory(snapshot.timeline, historyFrame);
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

// A History note is saved through the action controller like an edit, so a
// failed save gets the same alert and Retry.
bindHistory(historyFrame, {
  onNote: (id, note) => executeStateMutation("/note", { id, note })
});

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

// The chart is drawn at its own pixel size, so a panel that changes size is
// redrawn inside the observer callback: that runs once per frame, after
// layout and before paint, so no frame shows the old drawing scaled. The
// redraw never changes the chart's layout, so it cannot loop. A hidden chart
// keeps its drawing until it is shown again.
new ResizeObserver(redrawChartToFit).observe(svg);

// The toolbar's rows change with its width and its fonts, not with where the
// Chart / Table switch stands, which changes only its height; so a width that
// has not changed asks for nothing, and placing the switch cannot loop.
let contextBarWidth = 0;
new ResizeObserver(([entry]) => {
  if (entry.contentRect.width === contextBarWidth) return;
  contextBarWidth = entry.contentRect.width;
  placeTermViewToggle(contextBar, termViewToggle, contribTools);
}).observe(contextBar);
document.fonts?.ready.then(() => placeTermViewToggle(contextBar, termViewToggle, contribTools));

function redrawChartToFit() {
  const drawn = svg.viewBox.baseVal;
  const box = chartSize(svg.clientWidth, svg.clientHeight);
  if (!svg.clientWidth || (box.width === drawn.width && box.height === drawn.height)) return;
  const preview = store.getState().view.preview;
  if (preview && preview.term === selectedTerm()) renderInteractionPreview(preview);
  else renderChartOnly();
}

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
// Help sits in the app bar, outside the tool rail's own click handling.
helpAction.addEventListener("click", () => openHelp());

// The join a shaped range gets at its edges, kept across pages when storage
// allows; a term that cannot take it shows, and gets, the join it can.
let shapeJoinChoice = readShapeJoin();
function renderShapeJoin(term) {
  const joins = term?.shape?.joins;
  renderJoinToggle(
    shapeJoin, effectiveShapeJoin(shapeJoinChoice, joins), joins, term?.shape?.join_reason ?? null
  );
}
bindJoinToggle(shapeJoin, {
  onChange: (join) => {
    shapeJoinChoice = join;
    storeShapeJoin(join);
    renderShapeJoin(currentTerm());
  }
});
renderJoinToggle(shapeJoin, shapeJoinChoice);

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

function selectionAnchor() {
  return store.getState().view.selectionAnchor;
}

function setSelectionAnchor(anchor) {
  actions.patchView({ selectionAnchor: anchor });
}

function setSelectionSpan(span) {
  actions.patchView({ selectionSpan: span });
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
    profileTolerance,
    profileProgress,
    profileTraceStatus,
    profileTraceLegend,
    profileTracePlot,
    profileTraceTable,
    collapseLevels,
    ungroupLevels,
    summaryStatus,
    summaryNote,
    summaryFrame,
    summarySearchCount,
    summaryHeader,
    summaryModelChips,
    summaryTiles,
    summaryView
  };
}

// What the inspector shows of the summary; summary.js reapplies it on every
// render, so a refit or a new payload keeps the search.
function summaryView() {
  const state = store.getState();
  const snapshot = state.remote.snapshot;
  const terms = snapshot ? snapshot.terms : {};
  const names = Object.keys(terms);
  return {
    query: summaryQuery,
    termNames: names,
    filter: summaryFilter,
    currentTerm: selectActiveTermName(state),
    toggled: summaryToggled,
    edited: names.filter((name) => terms[name].edited === true),
    waiting: waitingCounts(snapshot?.pending ?? []),
    kinds: Object.fromEntries(
      names.map((name) => [name, terms[name].term_type || terms[name].kind || ""])
    )
  };
}

// What the summary reads from the state besides its payload: which terms
// carry hand edits and which wait for a refit.
function selectSummaryMarks(state) {
  const snapshot = state.remote.snapshot;
  if (!snapshot) return "";
  const edited = Object.keys(snapshot.terms).filter((name) => snapshot.terms[name].edited === true);
  return JSON.stringify([edited, waitingCounts(snapshot.pending ?? [])]);
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
// Inside a notebook cell the kernel runs on another machine, so its file
// manager would open there: the dialog saves to a kernel path and stops.
const notebookHost = globalThis.superglmEditorHost?.kind === "notebook";
if (notebookHost && exportOpenDirectory instanceof HTMLElement) exportOpenDirectory.hidden = true;
// One editor can show in several notebook outputs. When another output
// changes the session, this one re-reads it, once for a burst of changes.
if (notebookHost) {
  let refreshing = false;
  let again = false;
  globalThis.superglmEditorHost.onRemoteChange = async () => {
    if (refreshing) {
      again = true;
      return;
    }
    refreshing = true;
    try {
      do {
        again = false;
        await actions.refreshFromPythonWhenIdle();
      } while (again);
    } finally {
      refreshing = false;
    }
  };
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
    openDirectory: !notebookHost && exportOpenDirectory instanceof HTMLButtonElement
      ? exportOpenDirectory
      : null,
    status: exportStatus,
    pendingNote: exportPendingNote instanceof HTMLElement ? exportPendingNote : null
  },
  saveBlobToFile,
  pendingCount: () => selectPendingSteps(store.getState()).length,
  finalFitAvailable: () => {
    const finalFit = store.getState().remote.snapshot?.final_fit;
    return Boolean(finalFit?.available && !finalFit.stale);
  }
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

// A structural change waits: Python builds it and keeps it, drawn on the
// chart, until Refit applies every waiting change in one fit. Nothing is
// fitted, so nothing blocks the page. With "Refit after every structural
// change" on in Settings, the change goes to its operation's own route
// instead, which stages it and refits at once: one step, which one Undo
// takes back.
async function runStructuralChange(descriptor) {
  const { keepReference, refitEveryChange } = loadSettings();
  if (refitEveryChange) {
    const atOnce = refitAtOnceTransition(descriptor);
    return runStructuralRefit({
      ...atOnce,
      payload: { ...atOnce.payload, keep_reference: keepReference }
    });
  }
  if (appBusyActive || store.getState().request.mutation.status !== "idle") return null;
  stopContributionBuild();
  const result = await actions.executeStructuralMutation({
    ...descriptor,
    blocking: false,
    payload: {
      ...descriptor.payload,
      keep_reference: keepReference,
      level_display: selectSummaryLevelDisplay(store.getState())
    }
  });
  return result.ok ? result.envelope : null;
}

// The Refit button and its R shortcut come here.
async function refitPending() {
  const count = selectPendingSteps(store.getState()).length;
  if (count === 0) return null;
  return runStructuralRefit(refitPendingTransition(count));
}

// A structural step loses nothing: Undo puts back the state before it, edits
// included, so it runs without asking.
async function runStructuralRefit(descriptor) {
  if (appBusyActive || store.getState().request.mutation.status !== "idle") {
    return { ok: false, skipped: true };
  }
  stopContributionBuild();
  summarySource.value = "selected";
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
  for (const candidate of [opener, featureListNodes.search, inspectorToggle]) {
    if (!candidate || !candidate.isConnected || typeof candidate.focus !== "function") continue;
    candidate.focus({ preventScroll: true });
    if (document.activeElement === candidate) return;
  }
}

if (new URLSearchParams(window.location.search).get("test") === "1") {
  window.__superglmTest = Object.freeze({
    setAppBusy,
    // A selection posts without the busy overlay; tests wait on this before
    // the next click, which a running mutation would skip.
    mutationStatus: () => store.getState().request.mutation.status
  });
}

function showTimingStatus(payload, timing) {
  if (!timing) return;
  latestTransitionTiming = timing;
  latestTimingNote = payload.note || "";
  if (summaryStatus) {
    summaryStatus.textContent = `Refit completed in ${formatMilliseconds(timing.client_total_ms)}`;
  }
  renderTimingReadout();
  if (summaryNote) summaryNote.textContent = payload.note || "";
}

// Settings › Request timings: the last refit's and each panel's durations.
function renderTimingReadout() {
  if (!settingsTiming) return;
  const sections = [];
  if (latestTransitionTiming) sections.push(formatTimingDetails(latestTransitionTiming));
  const evidenceDetails = formatEvidenceTimingDetails(evidenceTiming.durations());
  if (evidenceDetails) sections.push(evidenceDetails);
  const details = sections.filter(Boolean).join(" · ");
  settingsTiming.textContent = latestTimingNote && details
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
  const activeView = ["validation", "cv", "final"].includes(view) ? view : "editor";
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
  // The Cross-validation tab lays the panel out its own way (cv.css).
  reportPanel.dataset.report = activeView;
}

function renderChartWorkspace() {
  const editorState = store.getState();
  const snapshot = editorState.remote.snapshot;
  if (!snapshot) return;
  const view = editorState.view;
  ciToggle.setAttribute("aria-pressed", String(view.showCi));
  renderFreeLevelsToggle(snapshot);
  const selected = selectedTerm();
  const term = currentTerm();
  if (!term) return;
  if (selected !== renderedTerm) {
    renderedTerm = selected;
    stopContributionBuild();
  }
  if (applyTermDefaults(term)) return;
  const tableView = renderTermView(view.termView);
  const selection = view.preview && view.preview.term === selected
    ? new Set(view.preview.selection)
    : currentSelection();
  statusNode.classList.remove("is-error");
  if (updateHandleCount(term)) return;
  renderToolRail(toolRail, {
    mode: view.mode,
    handlesAvailable: Boolean(term.controls),
    handlesReason: term.spline_view?.reason ?? null
  });
  updateGroupDisplayControl(term);
  updateNewLevelsControl(term);
  updateCollapseAction(term, selection);
  updateShapeActions(term, selection);
  updateResetOrderAction(term);
  if (!tableView) drawChart(term, selection, chartContext);
  const collapsedOriginalNote = selectionContextNote(term);
  renderContextBar(
    {
      nameNode: termNameNode,
      kindNode: termKind,
      edfNode: termEdf,
      referenceNode: termReference,
      statusNode
    },
    {
      name: selected,
      term,
      selectionSize: selection.size,
      note: collapsedOriginalNote,
      pendingCount: selectPendingSteps(editorState).length,
      range: selectionSpanRange(term, selection, chartContext)
    }
  );
  placeTermViewToggle(contextBar, termViewToggle, contribTools);
}

// Table puts the term's rating-table block where the chart was; the chart
// keeps its mode, zoom and selection for when Chart comes back.
function renderTermView(termView) {
  const tableView = termView === "table";
  renderTermViewToggle(termViewToggle, termView);
  plotColumn.classList.toggle("is-table-view", tableView);
  // An SVG element has no `hidden` property; the attribute is what CSS hides.
  svg.toggleAttribute("hidden", tableView);
  ratingTableFrame.hidden = !tableView;
  if (tableView) stopContributionBuild();
  return tableView;
}

let ratingTableSequence = 0;

function selectRatingTableRequest(state) {
  return {
    table: state.view.termView === "table",
    term: selectActiveTermName(state),
    revision: selectModelRevision(state)
  };
}

function sameRatingTableRequest(next, previous) {
  return next.table === previous.table &&
    next.term === previous.term &&
    next.revision === previous.revision;
}

// One request per term and model revision while Table is shown; a reply
// that a newer request has overtaken is dropped.
async function refreshRatingTable({ table, term, revision }) {
  if (!table || !term || revision < 0) return;
  const sequence = ++ratingTableSequence;
  renderRatingTable(ratingTableFrame, ratingTableMessage(term, RATING_TABLE_LOADING));
  let model;
  try {
    model = ratingTableModel(await editorClient.ratingTable(term));
  } catch {
    model = ratingTableMessage(term, RATING_TABLE_FAILED);
  }
  if (sequence === ratingTableSequence) renderRatingTable(ratingTableFrame, model);
}

function renderChartOnly() {
  const state = store.getState();
  const term = currentTerm();
  if (!state.remote.snapshot || !term || state.view.termView === "table") return;
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

// The revision stands in for every row's EDF, which only a refit changes; a
// stage leaves the revision, so the waiting terms are keyed on their own.
function selectFeatureListRenderState(state) {
  const snapshot = selectSnapshot(state);
  return {
    ready: snapshot !== null,
    catalogueKey: snapshot ? termCatalogueKey(snapshot.terms || {}) : "",
    revision: selectModelRevision(state),
    activeTerm: selectActiveTermName(state),
    waiting: selectWaitingTerms(state).join("\u0000")
  };
}

function sameFeatureListRenderState(next, previous) {
  return next.ready === previous.ready &&
    next.catalogueKey === previous.catalogueKey &&
    next.revision === previous.revision &&
    next.activeTerm === previous.activeTerm &&
    next.waiting === previous.waiting;
}

function renderFeatureListState() {
  const state = store.getState();
  const terms = selectSnapshot(state)?.terms ?? {};
  renderFeatureList(featureListNodes, {
    groups: groupedTerms(terms),
    terms,
    activeTerm: selectActiveTermName(state),
    query: featureQuery,
    open: featureListOpen,
    waiting: new Set(selectWaitingTerms(state))
  });
}

function selectChartRenderState(state) {
  const activeTerm = selectActiveTermName(state);
  const view = state.view;
  return {
    ready: state.remote.snapshot !== null,
    chartEpoch: state.remote.chartEpoch,
    activeTerm,
    mode: view.mode,
    termView: view.termView,
    showCi: view.showCi,
    showContrib: view.showContrib,
    freeLevels: view.freeLevels,
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
    next.termView === previous.termView &&
    next.showCi === previous.showCi &&
    next.showContrib === previous.showContrib &&
    next.freeLevels === previous.freeLevels &&
    next.zoom === previous.zoom &&
    next.groupMode === previous.groupMode;
}

function selectHistoryRenderState(state) {
  const timeline = state.remote.snapshot?.timeline || null;
  return { timeline, key: timeline ? JSON.stringify(timeline) : "" };
}

function sameHistoryRenderState(next, previous) {
  return next.key === previous.key;
}

function renderHistoryState({ timeline }) {
  if (timeline) renderHistory(timeline, historyFrame);
}

function selectAppBarRenderState(state) {
  const snapshot = state.remote.snapshot;
  return {
    ready: snapshot !== null,
    activeView: state.view.activeView,
    undoLabel: snapshot?.undo_redo.undo ?? null,
    redoLabel: snapshot?.undo_redo.redo ?? null,
    canRevert: Boolean(snapshot && revertAvailable(snapshot)),
    busy: state.request.mutation.status === "running",
    pendingCount: selectPendingSteps(state).length
  };
}

function sameAppBarRenderState(next, previous) {
  return next.ready === previous.ready &&
    next.activeView === previous.activeView &&
    next.undoLabel === previous.undoLabel &&
    next.redoLabel === previous.redoLabel &&
    next.canRevert === previous.canRevert &&
    next.busy === previous.busy &&
    next.pendingCount === previous.pendingCount;
}

function renderAppBarState(state) {
  if (!state.ready) return;
  renderAppBar({
    root: appBar,
    activeView: state.activeView,
    undoButton: undoAction,
    redoButton: redoAction,
    revertButton: revertAction,
    refreshButton: refreshAction,
    undoLabel: state.undoLabel,
    redoLabel: state.redoLabel,
    canRevert: state.canRevert,
    busy: state.busy,
    refitButton: refitPendingAction,
    refitCount: refitPendingCount,
    pendingCount: state.pendingCount
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
  updateShapeActions(term, selection);
  renderContextBar(
    {
      nameNode: termNameNode,
      kindNode: termKind,
      edfNode: termEdf,
      referenceNode: termReference,
      statusNode
    },
    {
      name: termName,
      term,
      selectionSize: selection.size,
      note: selectionContextNote(term),
      pendingCount: selectPendingSteps(store.getState()).length,
      range: selectionSpanRange(term, selection, chartContext)
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
    selectedWeightShare: impact.selected_weight_share,
    // The anchor's marks and a Shift-click's range follow them as well.
    anchor: state.view.selectionAnchor,
    span: state.view.selectionSpan
  };
}

function sameSelectionState(next, previous) {
  if (
    next.termName !== previous.termName ||
    next.weightedMeanRelativity !== previous.weightedMeanRelativity ||
    next.selectedWeightShare !== previous.selectedWeightShare ||
    next.anchor !== previous.anchor ||
    next.span !== previous.span ||
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

const REPORT_TITLES = Object.freeze({
  validation: "Validation Report",
  cv: "Cross-validation",
  final: "Final Fit Report"
});

function renderReportEvidence(evidence, activeView) {
  const busy = evidence.status === "updating";
  reportFrame.setAttribute("aria-busy", busy ? "true" : "false");
  reportFrame.dataset.freshness = evidence.status;
  if (activeView === "editor") return;
  const payloadMatchesView = evidence.payload !== null && evidence.payload.report === activeView;
  if (payloadMatchesView) {
    renderReport(evidence.payload, { reportTitle, reportStatus, reportFrame }, cvTab);
  } else {
    reportTitle.textContent = REPORT_TITLES[activeView] || REPORT_TITLES.validation;
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

function updateNewLevelsControl(term) {
  if (!newLevelsWrap || !(newLevelsMode instanceof HTMLSelectElement) || !term) return;
  renderNewLevelsControl({ wrap: newLevelsWrap, select: newLevelsMode }, term);
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
  if (setReference) {
    const label = isLevelTerm ? selectedLevelLabel(term, selection) : null;
    setReference.hidden = label === null || label === term.reference?.level;
  }
  const special = specialActions(term, isLevelTerm ? selectedLevels(term, selection) : []);
  renderSpecialAction(makeSpecial, special.make);
  renderSpecialAction(returnToCurve, special.back);
}

// Make special and Back on the curve show their state; a disabled one says
// why in its popover.
function renderSpecialAction(button, state) {
  if (!button) return;
  button.hidden = !state.visible;
  button.setAttribute("aria-disabled", String(!state.enabled));
  renderShapeReason(button, state.reason);
}

// The free-level comparison in view: the last one fitted, while its term and
// model revision are the ones shown.
function shownFreeLevels() {
  const state = store.getState();
  const free = state.view.freeLevels;
  return freeLevelsShown(free, selectedTerm(), state.remote.snapshot?.fit_token) ? free : null;
}

// Free levels refits the model with the term's levels free, as Refit does a
// structural change, and draws them until the term or the model changes.
async function toggleFreeLevels() {
  if (shownFreeLevels()) {
    actions.patchView({ freeLevels: null });
    return;
  }
  if (appBusyActive || store.getState().request.mutation.status !== "idle") return;
  const term = selectedTerm();
  stopContributionBuild();
  setAppBusy(true, "Fitting free levels", `Refitting the model with ${term}'s levels free`);
  try {
    const free = await editorClient.freeLevels(term);
    actions.patchView({ freeLevels: free });
    if (free.notice) actions.showNotice(free.notice);
  } catch (error) {
    actions.showNotice(error instanceof Error ? error.message : String(error));
  } finally {
    setAppBusy(false);
  }
}

function renderFreeLevelsToggle(snapshot) {
  if (!freeLevelsToggle) return;
  const term = snapshot.terms?.[selectedTerm()];
  freeLevelsToggle.hidden = (term?.term_type || term?.kind) !== "ordered categorical";
  freeLevelsToggle.setAttribute("aria-pressed", String(shownFreeLevels() !== null));
}

// The four shape icons share one state per selection, except that the bands
// of an ordered term bound the degree they can carry. The join toggle shows
// with them, and the palette's refit row only when something is on it.
function updateShapeActions(term, selection) {
  let shapesVisible = false;
  for (const button of shapeButtons) {
    const degree = Number(button.dataset.shapeDegree);
    const { visible, enabled, reason } = shapeButtonState(term, selection, degree);
    button.hidden = !visible;
    button.setAttribute("aria-disabled", String(!enabled));
    renderShapeReason(button, reason);
    shapesVisible = shapesVisible || visible;
  }
  shapeJoin.hidden = !shapesVisible;
  renderShapeJoin(term);
  shapeJoinSeparator.hidden = !shapesVisible;
  const refitVisible = shapesVisible ||
    [collapseLevels, ungroupLevels, setReference, makeSpecial, returnToCurve]
      .some((button) => button && !button.hidden);
  selectionRefitBreak.hidden = !refitVisible;
  selectionRefitLabel.hidden = !refitVisible;
}

// A disabled shape icon says why; its own popover text outranks the operation help.
function renderShapeReason(button, reason) {
  if (reason === null) {
    delete button.dataset.popoverTitle;
    delete button.dataset.popoverBody;
    return;
  }
  button.dataset.popoverTitle = button.getAttribute("aria-label");
  button.dataset.popoverBody = reason;
}

// The selected source levels by label, in axis order: what a staged change names.
function selectedLevels(term, selection) {
  const levels = Array.isArray(term.levels) ? term.levels : [];
  return [...selection]
    .sort((left, right) => left - right)
    .filter((index) => index >= 0 && index < levels.length)
    .map((index) => String(levels[index]));
}

// One displayed level: a single source level, or one whole collapsed group.
function selectedLevelLabel(term, selection) {
  if (selection.size === 1) {
    const [index] = selection;
    return term.levels?.[index] ?? null;
  }
  const groups = Array.isArray(term.level_groups) ? term.level_groups : [];
  const group = groups.find((candidate) =>
    candidate.indices.length === selection.size &&
    candidate.indices.every((index) => selection.has(Number(index)))
  );
  return group ? group.label : null;
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
  contribTools.hidden = !canShowContrib;
  contribPlay.disabled = buildFrame !== null;
  basisToggle.setAttribute("aria-pressed", String(Boolean(view.showContrib && canShowContrib)));
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
  // A grouped term opens as Settings' "Groups shown as" says.
  if (
    term.group_display &&
    term.group_display.available &&
    !view.groupModeByTerm[selectedTerm()]
  ) {
    patch.groupModeByTerm = {
      ...view.groupModeByTerm,
      [selectedTerm()]: loadSettings().groupsDefault
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
  return loadSettings().buildDurationMs;
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
  selectionAnchor,
  setSelectionAnchor,
  setSelectionSpan,
  setPreviewTerm: setInteractionPreview,
  clearPreviewTerm: clearInteractionPreview,
  setZoom,
  clearZoom,
  actions,
});
bindPointLens(svg);
// The anchor's tags step aside while the pointer drags on the chart.
bindDragWatch(svg, document.querySelector(".chart-shell"), CLICK_SLOP);

async function selectFeature(term) {
  if (term === selectedTerm()) return;
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
}

bindFeatureList(featureListNodes, {
  onSelect: selectFeature,
  onQuery: (query) => {
    featureQuery = query;
    renderFeatureListState();
  },
  onToggle: () => {
    featureListOpen = !featureListOpen;
    storeFeatureListOpen(featureListOpen);
    renderFeatureListState();
  }
});
renderFeatureListState();

bindSummarySearch(summarySearch, (query) => {
  summaryQuery = query;
  summaryToggled.clear();
  applySummaryView(summaryNodes());
});
bindSummaryFilter(summaryFilterNode, (filter) => {
  summaryFilter = filter;
  renderSummaryFilter(summaryFilterNode, filter);
  applySummaryView(summaryNodes());
});
bindSummarySections(summaryFrame, (term, open) => {
  summaryToggled.set(term, !open);
  applySummaryView(summaryNodes());
});

// New levels → is a session operation on the in-force model: no refit, and
// one entry on the one Undo history.
if (newLevelsMode instanceof HTMLSelectElement) {
  newLevelsMode.addEventListener("change", async () => {
    await executeStateMutation("/set_unseen", {
      term: selectedTerm(),
      unseen: newLevelsMode.value
    });
    // A refused choice leaves the policy in force, which the select shows again.
    updateNewLevelsControl(currentTerm());
  });
}

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
    actions.schedulePanelEvidence(
      "summary",
      "/summary",
      summaryRequestPayload(),
      { immediate: true }
    );
  });
}

basisToggle.addEventListener("click", () => {
  stopContributionBuild();
  actions.patchView({ showContrib: !store.getState().view.showContrib });
});

contribPlay.addEventListener("click", startContributionBuild);

bindTermViewToggle(termViewToggle, {
  onChange: (termView) => actions.patchView({ termView })
});

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
    // Each operation names the term this page shows: Python's selected term
    // is shared, and another page of the same session may have moved it.
    const term = selectedTerm();
    await executeStateMutation("/op", term ? { operation, term } : { operation });
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
    const term = currentTerm();
    if (!term) return;
    await runStructuralChange(stageCollapse(selectedTerm(), selectedLevels(term, currentSelection())));
  });
}
if (ungroupLevels) {
  ungroupLevels.addEventListener("click", async () => {
    const term = currentTerm();
    if (!term) return;
    await runStructuralChange(stageUngroup(selectedTerm(), selectedLevels(term, currentSelection())));
  });
}
if (setReference) {
  setReference.addEventListener("click", async () => {
    const term = currentTerm();
    const label = term ? selectedLevelLabel(term, currentSelection()) : null;
    if (label === null) return;
    await runStructuralChange(stageReference(selectedTerm(), label));
  });
}
for (const [button, stage] of [[makeSpecial, stageSpecial], [returnToCurve, stageOnCurve]]) {
  if (!button) continue;
  button.addEventListener("click", async () => {
    const term = currentTerm();
    if (!term || button.getAttribute("aria-disabled") === "true") return;
    await runStructuralChange(stage(selectedTerm(), selectedLevels(term, currentSelection())));
  });
}
if (freeLevelsToggle) freeLevelsToggle.addEventListener("click", toggleFreeLevels);
for (const button of shapeButtons) {
  button.addEventListener("click", async () => {
    const term = currentTerm();
    const range = term && shapeRangeForSelection(term, currentSelection());
    if (!range || button.getAttribute("aria-disabled") === "true") return;
    const degree = Number(button.dataset.shapeDegree);
    await runStructuralChange(
      stageShapeRange(
        selectedTerm(), range.lo, range.hi, degree,
        effectiveShapeJoin(shapeJoinChoice, term.shape.joins)
      )
    );
  });
}


store.subscribe(selectChartRenderState, () => renderChartWorkspace(), sameChartRenderState);
store.subscribe(selectRatingTableRequest, refreshRatingTable, sameRatingTableRequest);
store.subscribe(
  selectFeatureListRenderState,
  renderFeatureListState,
  sameFeatureListRenderState
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
store.subscribe(selectActiveTermName, () => {
  summaryToggled.clear();
  applySummaryView(summaryNodes(), { follow: true });
});
store.subscribe(selectSummaryMarks, () => applySummaryView(summaryNodes()));
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
  statusNode.classList.add("is-error");
});
