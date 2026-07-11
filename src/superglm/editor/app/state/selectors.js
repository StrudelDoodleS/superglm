// @ts-check
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */

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
  if (state.view.selectionPreview?.term === active) {
    return state.view.selectionPreview.indices;
  }
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

/** @param {EditorState} state */
export function selectGroupDisplayMode(state) {
  const active = selectActiveTermName(state);
  const term = selectCurrentTerm(state);
  return state.view.groupModeByTerm[active]
    || term?.group_display?.default_mode
    || "expanded";
}

/** @param {EditorState} state */
export const selectMutation = (state) => state.request.mutation;

/** @param {EvidencePanel} panel @returns {(state: EditorState) => EvidenceState} */
export const selectEvidence = (panel) => (state) => state.request.evidence[panel];
