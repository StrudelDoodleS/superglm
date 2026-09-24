// @ts-check

/** @typedef {'editor'|'validation'|'final'} AppView */
/** @typedef {'select'|'move'|'zoom'|'handles'|'breaks'} EditorMode */
/** @typedef {'idle'|'running'|'error'} MutationStatus */
/** @typedef {'idle'|'updating'|'current'|'stale'|'error'} EvidenceStatus */
/** @typedef {'metrics'|'summary'|'report'} EvidencePanel */
/** @typedef {'expanded'|'grouped'} SummaryLevelDisplay */
/**
 * @typedef {Object} SummaryPayload
 * @property {boolean} available
 * @property {SummaryLevelDisplay} [level_display]
 * @property {string} [source]
 * @property {string} [label]
 * @property {string} [note]
 * @property {string} [error]
 * @property {string} [html]
 * @property {Record<string, unknown>} [compact]
 */
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
 * @typedef {Object} ImpactPayload
 * @property {number} [weighted_mean_relativity]
 * @property {number} [selected_weight_share]
 */
/**
 * @typedef {Object} TermReference
 * @property {string} level
 * @property {'most_exposed'|'first'|'pinned'} policy
 */
/** @typedef {'piecewise'|'spline'|'polynomial'} BreakForm */
/**
 * A per-term Breaks draft. Breaks are band labels on an ordered axis and x
 * values on a numeric axis, exactly as /transform_term takes them.
 * @typedef {Object} BreakDraft
 * @property {BreakForm} form
 * @property {Array<string|number>} breaks
 * @property {number[]} degrees
 * @property {number} degree
 */
/**
 * @typedef {Object} TermTransform
 * @property {string[]|null} axis
 * @property {{breaks:Array<string|number>, degrees:number[]}|null} piecewise
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
 * @property {ImpactPayload} impact
 * @property {number|null} [effective_df]
 * @property {TermReference|null} [reference]
 * @property {TermTransform|null} [transform]
 */
/**
 * @typedef {Object} StructureHistory
 * @property {number} depth
 * @property {{operation:string, term:string|null, label:string}|null} last
 */
/**
 * @typedef {Object} EditorSnapshot
 * @property {number} model_revision
 * @property {number} [state_generation]
 * @property {number} [chart_generation]
 * @property {string} selected_term
 * @property {Record<string, TermPayload>} terms
 * @property {Record<string, number[]>} selection
 * @property {StructureHistory} structure_history
 * @property {EditorHistory} history
 */
/**
 * @typedef {Object} StructuralTransitionTiming
 * @property {string} operation
 * @property {number} fit_ms
 * @property {number} summary_ms
 * @property {number} state_ms
 * @property {number} server_total_ms
 */
/**
 * @typedef {Object} StructuralTransitionEnvelope
 * @property {EditorSnapshot} state
 * @property {SummaryPayload} summary
 * @property {StructuralTransitionTiming} timing
 */
/**
 * @typedef {Object} MutationDescriptor
 * @property {string} name
 * @property {string} path
 * @property {Record<string, unknown>} payload
 */
/**
 * @typedef {MutationDescriptor & {
 *   onRequestSettled?:()=>void|Promise<void>,
 *   onPrimaryCommitted?:()=>void|Promise<void>,
 *   onPaintSettled?:()=>void|Promise<void>
 * }} StructuralMutationDescriptor
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
 * @property {SummaryLevelDisplay} summaryLevelDisplay
 * @property {Record<string, unknown>} zoomByTerm
 * @property {Record<string, string>} groupModeByTerm
 * @property {Record<string, BreakDraft>} breakDraftByTerm
 * @property {'summary'|'history'|'advanced'|'help'} inspectorPane
 * @property {boolean} inspectorOpen
 * @property {{term:string, payload:TermPayload, selection:number[]}|null} preview
 * @property {{term:string, indices:number[]}|null} selectionPreview
 */
/**
 * @typedef {Object} MutationRequestState
 * @property {MutationStatus} status
 * @property {string|null} operation
 * @property {string|null} error
 * @property {boolean} [blocking]
 */
/**
 * @typedef {Object} RecoveryRequestState
 * @property {string} message
 * @property {MutationDescriptor|null} retry
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
 * @property {{snapshot:EditorSnapshot|null, summary:SummaryPayload|null, chartEpoch:number}} remote
 * @property {EditorViewState} view
 * @property {EditorRequestState} request
 */
/** @typedef {{panel:EvidencePanel, revision:number, sequence:number}} EvidenceToken */
/** @typedef {{ok:true, snapshot:EditorSnapshot}|{ok:false, skipped?:boolean, error:Error}} ActionResult */
/**
 * @typedef {{ok:true, envelope:StructuralTransitionEnvelope}|
 * {ok:false, skipped?:boolean, error:Error}} StructuralActionResult
 */

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
