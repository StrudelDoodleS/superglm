// @ts-check
import { isReportName } from "../api/contracts.js";

/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */
/** @typedef {import('../api/contracts.js').ReportName} ReportName */

/** @type {number[]} */
const EMPTY_SELECTION = [];
Object.freeze(EMPTY_SELECTION);

/** @param {EditorState} state */
export const selectSnapshot = (state) => state.remote.snapshot;

/** @param {EditorState} state */
export const selectModelRevision = (state) => state.remote.snapshot?.model_revision ?? -1;

/** @param {EditorState} state */
export function selectActiveTermName(state) {
  const snapshot = selectSnapshot(state);
  if (!snapshot) return "";
  if (snapshot.terms[state.view.activeTerm]) return state.view.activeTerm;
  if (snapshot.terms[snapshot.selected_term]) return snapshot.selected_term;
  return Object.keys(snapshot.terms)[0] || "";
}

/** @param {EditorState} state */
export function selectCurrentTerm(state) {
  const snapshot = selectSnapshot(state);
  return snapshot?.terms[selectActiveTermName(state)] ?? null;
}

/** @param {EditorState} state */
export function selectCurrentSelection(state) {
  const active = selectActiveTermName(state);
  const snapshot = selectSnapshot(state);
  return snapshot?.selection[active] ?? EMPTY_SELECTION;
}

/** @param {EditorState} state */
export function selectRenderableTerm(state) {
  const active = selectActiveTermName(state);
  return state.view.preview?.term === active
    ? state.view.preview.payload
    : selectCurrentTerm(state);
}

/** @type {readonly import('../api/contracts.js').PendingStep[]} */
const NO_PENDING = Object.freeze([]);

/**
 * The structural changes waiting for Refit, oldest first.
 * @param {EditorState} state
 * @returns {readonly import('../api/contracts.js').PendingStep[]}
 */
export function selectPendingSteps(state) {
  return selectSnapshot(state)?.pending ?? NO_PENDING;
}

/**
 * Each term with a change waiting for Refit, once, in the order first staged.
 * @param {EditorState} state
 * @returns {string[]}
 */
export function selectWaitingTerms(state) {
  return [...new Set(selectPendingSteps(state).map((step) => step.term))];
}

/** @param {EditorState} state */
export function selectGroupDisplayMode(state) {
  const active = selectActiveTermName(state);
  const term = selectCurrentTerm(state);
  return state.view.groupModeByTerm[active]
    || term?.group_display?.default_mode
    || "expanded";
}

/** @param {EditorState} state */
export const selectSummaryLevelDisplay = (state) => state.view.summaryLevelDisplay;

/** @param {EditorState} state */
export const selectMutation = (state) => state.request.mutation;

/** @param {EvidencePanel} panel @returns {(state: EditorState) => EvidenceState} */
export const selectEvidence = (panel) => (state) => state.request.evidence[panel];

/** @param {EditorState} state @param {EvidencePanel} panel */
export function selectEvidenceNeedsRefresh(state, panel) {
  const revision = state.remote.snapshot?.model_revision;
  if (revision === undefined) return false;
  const evidence = state.request.evidence[panel];
  return evidence.revision !== revision || evidence.status === "stale";
}

/**
 * Whether report `panel`, once shown, must be asked of Python: the page holds no answer
 * for this model, or the one it holds lapsed or failed. A failure is asked again here,
 * where `selectEvidenceNeedsRefresh` leaves it for Retry.
 *
 * @param {EditorState} state @param {ReportName} panel
 */
export function selectReportNeedsAsk(state, panel) {
  const revision = state.remote.snapshot?.model_revision;
  if (revision === undefined) return false;
  const { revision: held, status } = state.request.evidence[panel];
  return held !== revision || status === "idle" || status === "stale" || status === "error";
}

/**
 * Return only evidence panels currently presented to the user.
 *
 * @param {EditorState} state
 * @param {{summaryCommitted?:boolean}} [options]
 * @returns {EvidencePanel[]}
 */
export function selectVisibleEvidencePanels(state, { summaryCommitted = false } = {}) {
  const view = state.view.activeView;
  // A report shows its own evidence; the Explain tab asks for its own risks and shows none.
  if (view !== "editor") return isReportName(view) ? [view] : [];
  // A panel whose tool the session does not offer is never requested.
  const off = state.remote.snapshot?.unavailable ?? {};
  const panels = /** @type {EvidencePanel[]} */ (off.metrics ? [] : ["metrics"]);
  if (
    !summaryCommitted &&
    !off.summary &&
    state.view.inspectorOpen &&
    state.view.inspectorPane === "summary"
  ) {
    panels.push("summary");
  }
  return panels;
}
