// @ts-check

/** @typedef {'editor'|'validation'|'final'} AppView */
/** @typedef {'select'|'move'|'zoom'|'handles'} EditorMode */
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
 * One action on the session's timeline, oldest first. The "marker" entry is
 * the current position: Undo takes the entry before it, and the `redo`
 * entries after it are what Redo would put back, in that order.
 * @typedef {Object} TimelineEntry
 * @property {"edit"|"structural"|"marker"} kind
 * @property {string} [operation]
 * @property {string|null} [term]
 * @property {string} [label]
 * @property {number} [n_points]
 * @property {Record<string, unknown>} [params]
 * @property {string} [hash]
 * @property {boolean} [redo]
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
/**
 * A range of a term pinned to a polynomial. Edges are x values on a numeric
 * term and band labels on an ordered one; degree 0-3 is Flat, Line,
 * Quadratic or Cubic, which ``label`` names. ``join`` is how the range meets
 * the free curve at its edges: "tangent", or "kink" (Corner).
 * @typedef {Object} ShapedRange
 * @property {number|string} lo
 * @property {number|string} hi
 * @property {number} degree
 * @property {string} label
 * @property {"tangent"|"kink"} [join]
 */
/**
 * How many values a numeric term's refit sees, per grid point: ``below[k]``
 * under the lower edge a selection starting at k snaps to, ``through[k]`` up
 * to the upper edge one ending at k snaps to.
 * @typedef {Object} ShapeSupport
 * @property {number[]} below
 * @property {number[]} through
 */
/**
 * The palette's shape state for a term: the ranges in force, the hover
 * reason when the term cannot take one, a numeric term's support counts and
 * an ordered term's special levels, which no range can cover.
 * @typedef {Object} TermShape
 * @property {boolean} available
 * @property {string|null} reason
 * @property {ShapedRange[]} ranges
 * @property {ShapeSupport|null} support
 * @property {string[]} specials
 * @property {("tangent"|"kink")[]} [joins] the joins the term can take
 * @property {string|null} [join_reason] why a join is missing from ``joins``
 */
/**
 * The /shape_range request: the selection's edges as shapeRangeForSelection
 * names them, the degree of the icon chosen, and the join toggle's choice.
 * @typedef {Object} ShapeRangeRequest
 * @property {string} term
 * @property {number|string} lo
 * @property {number|string} hi
 * @property {number} degree
 * @property {"tangent"|"kink"} join
 * @property {string} method
 */
/**
 * The /set_reference request: a displayed level, which may be a group label.
 * @typedef {Object} SetReferenceRequest
 * @property {string} term
 * @property {string} level
 * @property {string} method
 */
/**
 * The /revert_to_original request carries no fields.
 * @typedef {Record<string, never>} EmptyStructuralRequest
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
 * @property {Array<{label:string, indices:number[]}>} [level_groups]
 * @property {TermShape} shape
 */
/**
 * What Undo and Redo would take next, edits and structural steps alike; null
 * when there is nothing.
 * @typedef {Object} UndoRedo
 * @property {string|null} undo
 * @property {string|null} redo
 */
/**
 * @typedef {Object} EditorSnapshot
 * @property {number} model_revision
 * @property {number} [state_generation]
 * @property {number} [chart_generation]
 * @property {string} selected_term
 * @property {Record<string, TermPayload>} terms
 * @property {Record<string, number[]>} selection
 * @property {UndoRedo} undo_redo
 * @property {TimelineEntry[]} timeline
 * @property {boolean} in_force_is_original
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
