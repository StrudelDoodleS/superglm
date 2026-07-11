// @ts-check

/** @typedef {'editor'|'validation'|'final'} AppView */
/** @typedef {'select'|'move'|'zoom'|'handles'} EditorMode */
/** @typedef {'idle'|'running'|'error'} MutationStatus */
/** @typedef {'idle'|'updating'|'current'|'stale'|'error'} EvidenceStatus */
/** @typedef {'metrics'|'summary'|'report'} EvidencePanel */
/**
 * @typedef {Object} EditorHistory
 * @property {Array<Record<string, unknown>>} active
 * @property {Array<Record<string, unknown>>} redo
 */
/**
 * @typedef {Object} GroupDisplayPayload
 * @property {boolean} available
 * @property {string} default_mode
 * @property {Record<string, unknown>|null} collapsed
 */
/**
 * @typedef {Object} TermPayload
 * @property {string} kind
 * @property {string} term_type
 * @property {number[]} x
 * @property {number[]} y
 * @property {number[]} original_y
 * @property {number[]|null} previous_y
 * @property {string[]|null} levels
 * @property {number} n_points
 * @property {Record<string, unknown>|null} controls
 * @property {GroupDisplayPayload|null} group_display
 * @property {Record<string, unknown>} impact
 */
/**
 * @typedef {Object} EditorSnapshot
 * @property {number} model_revision
 * @property {string} selected_term
 * @property {Record<string, TermPayload>} terms
 * @property {Record<string, number[]>} selection
 * @property {boolean} can_uncollapse_levels
 * @property {Record<string, unknown>|null} last_collapse
 * @property {EditorHistory} history
 */
/**
 * @typedef {Object} EvidenceState
 * @property {EvidenceStatus} status
 * @property {number|null} revision
 * @property {number} sequence
 * @property {unknown} payload
 * @property {string|null} error
 * @property {{path:string, payload:Record<string, unknown>}|null} retry
 */
/**
 * @typedef {Object} EditorViewState
 * @property {string} activeTerm
 * @property {AppView} activeView
 * @property {EditorMode} mode
 * @property {boolean} showCi
 * @property {boolean} showContrib
 * @property {Record<string, unknown>} zoomByTerm
 * @property {Record<string, string>} groupModeByTerm
 * @property {'summary'|'history'|'advanced'|'help'} inspectorPane
 * @property {boolean} inspectorOpen
 * @property {{term:string, payload:TermPayload, selection:number[]}|null} preview
 */
/**
 * @typedef {Object} MutationRequestState
 * @property {MutationStatus} status
 * @property {string|null} operation
 * @property {string|null} error
 */
/**
 * @typedef {Object} RecoveryRequestState
 * @property {string} message
 * @property {{name:string, path:string, payload:Record<string, unknown>}|null} retry
 */
/**
 * @typedef {Object} EditorRequestState
 * @property {MutationRequestState} mutation
 * @property {Record<EvidencePanel, EvidenceState>} evidence
 * @property {RecoveryRequestState|null} recovery
 * @property {number} nextSequence
 */
/**
 * @typedef {Object} EditorState
 * @property {{snapshot:EditorSnapshot|null}} remote
 * @property {EditorViewState} view
 * @property {EditorRequestState} request
 */
/** @typedef {{panel:EvidencePanel, revision:number, sequence:number}} EvidenceToken */
/** @typedef {{ok:true, snapshot:EditorSnapshot}|{ok:false, skipped?:boolean, error:Error}} ActionResult */

export const EVIDENCE_PANELS = Object.freeze(
  /** @type {const} */ (["metrics", "summary", "report"])
);

/** @returns {EvidenceState} */
export function createEmptyEvidenceState() {
  return {
    status: "idle",
    revision: null,
    sequence: 0,
    payload: null,
    error: null,
    retry: null
  };
}
