// @ts-check

/** @typedef {'editor'|'validation'|'cv'|'final'} AppView */
/** @typedef {'select'|'move'|'zoom'|'handles'} EditorMode */
/** @typedef {'chart'|'table'} TermView */
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
 * entries after it are what Redo would put back, in that order. A structural
 * change is a "pending" entry, with status "waiting" until a Refit applies it
 * and "applied" after. Read the status, not the kind.
 * @typedef {Object} TimelineEntry
 * @property {"edit"|"structural"|"pending"|"marker"} kind
 * @property {string} [operation]
 * @property {string|null} [term]
 * @property {string} [label]
 * @property {number} [n_points]
 * @property {Record<string, unknown>} [params]
 * @property {string} [hash]
 * @property {boolean} [redo]
 * @property {string} [id] the step's short id: 7 hex digits
 * @property {number} [time] when the step was made, in Unix seconds
 * @property {string|null} [note] the analyst's note, saved with the exported model
 * @property {"applied"|"waiting"|"edit"} [status]
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
 * @property {'most_exposed'|'first'|'pinned'|'kept'} policy
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
 * It, /shape_range, /collapse_levels and /ungroup_levels refit at once:
 * Settings' "Refit after every structural change" sends a change this way.
 * @typedef {Object} SetReferenceRequest
 * @property {string} term
 * @property {string} level
 * @property {string} method
 */
/** @typedef {"collapse"|"ungroup"|"set_reference"|"shape"} StagedOperation */
/**
 * The /stage request: one structural change, with its parameters by label as
 * the session stores them. Collapse and ungroup take ``levels``. Set
 * reference takes a displayed ``level``, which may be a group label. A shape
 * takes ``lo``, ``hi``, ``degree`` and ``join``. main.js adds the Settings
 * that shape the change: ``keep_reference``, and ``level_display`` for the
 * summary.
 * @typedef {Object} StageRequest
 * @property {StagedOperation} operation
 * @property {string} term
 * @property {Record<string, unknown>} params
 * @property {boolean} [keep_reference]
 * @property {string} [level_display]
 */
/**
 * A structural change waiting for Refit. The snapshot lists them oldest first.
 * @typedef {Object} PendingStep
 * @property {string} id
 * @property {StagedOperation} operation
 * @property {string} term
 * @property {string} label
 * @property {Record<string, unknown>} params
 * @property {string|null} note
 * @property {number} time Unix seconds
 */
/**
 * What a term's waiting changes make of it at the next Refit: its groups by
 * label with their member levels, its shaped ranges, and its reference level.
 * @typedef {Object} TermPending
 * @property {Record<string, string[]>|null} groups
 * @property {ShapedRange[]} ranges
 * @property {string|null} reference
 */
/**
 * The /revert_to_original request carries no fields.
 * @typedef {Record<string, never>} EmptyStructuralRequest
 */
/**
 * An ordered categorical with a spline basis, drawn as its spline: the grid
 * ``x`` (level ``i`` at ``i``), the current and fitted curves as relativities,
 * and the display indices of the smooth levels. ``available`` is false, with
 * a fixed ``reason``, while the term's handles are off; ``fits_levels`` is
 * false while the edited levels are off the spline. Null for every other term.
 * @typedef {Object} SplineView
 * @property {boolean} available
 * @property {string|null} reason
 * @property {number[]|null} x
 * @property {number[]|null} y
 * @property {number[]|null} original_y
 * @property {number[]|null} level_indices
 * @property {boolean} fits_levels
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
 * @property {TermPending|null} [pending]
 * @property {boolean} [edited] whether the term carries hand edits
 *   (Python's `EditorSession.edited_terms()`)
 * @property {SplineView|null} [spline_view]
 * @property {TermUnseen|null} [unseen] the New levels choice; null except on a plain categorical
 */
/**
 * Where a plain categorical's levels unseen at fit go: the in-force
 * ``policy`` ("error", "base" or a group label), the control's ``choices``
 * (Refuse, Reference, then one per group) and the fixed ``reason`` the choice
 * cannot be made now, else null.
 * @typedef {Object} TermUnseen
 * @property {string} policy
 * @property {Array<{value:string, label:string}>} choices
 * @property {string|null} reason
 */
/**
 * The /set_unseen request: the term and the policy chosen.
 * @typedef {Object} SetUnseenRequest
 * @property {string} term
 * @property {string} unseen
 */
/**
 * The /rating_table response: the term's main-effect block of the Excel
 * export, with the number format and the note the workbook gives it, and
 * ``available`` false with a fixed ``reason`` when there is no table.
 * @typedef {Object} RatingTableResponse
 * @property {string} term
 * @property {boolean} available
 * @property {string|null} reason
 * @property {string[]} columns
 * @property {Array<Array<string|number|boolean|null>>} rows
 * @property {Array<string|null>} formats
 * @property {string|null} note
 * @property {number} model_revision
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
 * @property {PendingStep[]} [pending]
 * @property {{available:boolean, stale:boolean}} [final_fit] whether Export can offer the Final fit model
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
 * ``blocking`` is false for a change that fits nothing, such as a stage: no
 * busy overlay, and nothing goes inert.
 * @typedef {MutationDescriptor & {
 *   blocking?:boolean,
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
 * @property {TermView} termView
 * @property {boolean} showCi
 * @property {boolean} showContrib
 * @property {SummaryLevelDisplay} summaryLevelDisplay
 * @property {Record<string, unknown>} zoomByTerm
 * @property {Record<string, string>} groupModeByTerm
 * @property {'summary'|'history'|'settings'|'help'} inspectorPane
 * @property {boolean} inspectorOpen
 * @property {{term:string, payload:TermPayload, selection:number[]}|null} preview
 * @property {{term:string, indices:number[]}|null} selectionPreview
 * @property {{term:string, index:number}|null} selectionAnchor the point the next
 *   Shift-click spans from: a source index of `term`, set by a click or a Ctrl/Cmd-click
 * @property {{term:string, from:number, to:number, indices:number[]}|null} selectionSpan
 *   the last Shift-click's span, by source index: from the anchor it spanned from to
 *   the point clicked, and every index it selected. It describes the selection only
 *   while the anchor is still `from` and the selection is still `indices`.
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
/** @typedef {'cv'|'final_fit'} JobKind */
/**
 * A background job's status (/job_start, /job_status). ``progress`` entries
 * carry a ``phase`` ("fold", "curves", "fitting", "carrying") and its details.
 * @typedef {Object} JobStatus
 * @property {string} job_id
 * @property {JobKind} kind
 * @property {'running'|'done'|'failed'|'cancelled'} status
 * @property {Array<Record<string, unknown>>} progress
 * @property {Record<string, unknown>|null} result
 * @property {string|null} [error]
 * @property {boolean} [cancel_requested]
 */
/**
 * @typedef {Object} CVFoldRow
 * @property {number} fold
 * @property {number} n_train
 * @property {number} n_test
 * @property {number|null} fit_time_s
 * @property {boolean} converged
 * @property {number|null} effective_df
 * @property {Record<string, number|null>} scores
 */
/**
 * One cross-validation run: the supplied result, or Run CV on the current model.
 * @typedef {Object} CVResultPayload
 * @property {string} label
 * @property {'supplied'|'run'} origin
 * @property {number|null} model_revision
 * @property {boolean} stale
 * @property {CVFoldRow[]} folds
 * @property {Record<string, number|null>} mean
 * @property {Record<string, number|null>} std
 * @property {Record<string, number>} pooled
 */
/**
 * One fold's curve for a term: ``fold`` is its 0-based number, which picks
 * its colour and its place within a level; ``values`` is null where the
 * fold has no value, a level it never saw.
 * @typedef {{fold:number, label:string, values:Array<number|null>}} CVFoldCurve
 */
/**
 * One term's relativities by fold, each curve re-centred on its
 * exposure-weighted mean log; levels in the model's order.
 * @typedef {Object} CVTermItem
 * @property {string} name
 * @property {'levels'|'continuous'} kind
 * @property {number[]|null} x
 * @property {string[]|null} levels
 * @property {number[]} weights
 * @property {CVFoldCurve[]} folds
 * @property {number[]} fit
 * @property {number[]|null} edited
 * @property {boolean} [held] a hand edit Run CV put back on every fold; it has no spread
 * @property {number|null} spread
 * @property {number|null} min_correlation
 */
/**
 * The /report payload for the Cross-validation tab.
 * @typedef {Object} CVReportPayload
 * @property {'cv'} report
 * @property {string} title
 * @property {string} note
 * @property {number} model_revision
 * @property {{supplied:boolean, n_folds:number, splitter:string|null, n_rows:number|null}} header
 * @property {number} pending
 * @property {{available:boolean, reason:string|null, note:string|null}} run_cv
 * @property {{available:boolean, reason:string|null, note:string|null, done:boolean, stale:boolean, n_rows:number|null}} final_fit
 * @property {Array<{name:string, label:string, lower_is_better:boolean}>} metrics
 * @property {CVResultPayload[]} results
 * @property {{available:boolean, origin:string|null, stale:boolean, note:string|null, terms:CVTermItem[]}} relativities
 * @property {Record<JobKind, JobStatus|null>} jobs
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
