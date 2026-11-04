// @ts-check

/** @typedef {'editor'|'validation'|'cv'|'final'} AppView */
/** @typedef {'select'|'move'|'zoom'|'handles'|'knots'} EditorMode */
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
 * @property {string|null} [predictor] SuperLSS only: the family parameter the step acted in
 * @property {string} [predictor_label] SuperLSS only: the history chip, "Average" or "Spread"
 * @property {string} [term_label] SuperLSS only: the bare feature name, without its parameter prefix
 * @property {boolean} [display_decreasing] SuperLSS only: the shown curve falls as the recorded effect rises
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
 * @property {number} [total_weight]
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
 * @property {boolean} available whether a new range can be drawn on the term
 *   as its waiting changes leave it
 * @property {string|null} reason
 * @property {ShapedRange[]} ranges
 * @property {ShapeSupport|null} support
 * @property {string[]} specials
 * @property {string[]} [returnable] the special levels that can go back on the
 *   curve: those a structural step took off it, not those the code declares
 * @property {("tangent"|"kink")[]} [joins] the joins the term can take
 * @property {string|null} [join_reason] why a join is missing from ``joins``
 */
/** @typedef {"collapse"|"ungroup"|"set_reference"|"shape"|"special"|"on_curve"|"knots"|"basis"} StagedOperation */
/**
 * How a spline's knots are placed: a rule, or "explicit" positions (set in
 * code, or by hand in the editor).
 * @typedef {"uniform"|"quantile"|"quantile_rows"|"quantile_tempered"|"explicit"} KnotStrategy
 */
/** @typedef {Exclude<KnotStrategy, "explicit">} KnotRule */
/**
 * A spline term's knots, in chart x: a numeric spline's own values, an
 * ordered term's display positions (smooth level ``i`` at ``i``). The fields
 * after ``reason`` are null, and the flags false, where the term has no knots
 * to adjust.
 * @typedef {Object} TermKnots
 * @property {boolean} available whether the Knots tool works on the term
 * @property {string|null} reason why it does not, for the tool's popover
 * @property {number[]|null} positions the interior knots in force, ascending
 * @property {number|null} count
 * @property {KnotStrategy|null} strategy
 * @property {number|null} alpha the tempered quantiles' power
 * @property {boolean} from_editor the knots in force were set in the editor
 * @property {number|null} lo knots lie strictly between ``lo`` and ``hi``
 * @property {number|null} hi
 * @property {number|null} min_gap an ordered term's least distance between two
 *   knots, and between a knot and ``lo`` or ``hi``: a tenth of a level, also
 *   its grid. Null on a numeric term, where a knot a change places keeps two
 *   significant figures of the space between the knots or ends beside it
 *   (``knotGrid``); the knots it keeps stay as close as they are
 * @property {number|null} max_count the most knots the term takes; null for no limit
 * @property {boolean} resettable the knots in force or waiting differ from the
 *   original model's
 * @property {string|null} even_only why the term takes evenly spaced knots
 *   only (a natural spline, or a P-spline whose penalty order exceeds its
 *   degree), as Python refuses uneven ones; null where any spacing goes
 * @property {KnotBasis|null} [basis] how the browser rebuilds the term's
 *   B-spline basis to draw it; null for a cardinal spline, which has none
 * @property {KnotBasis|null} [waiting_basis] how it rebuilds the basis the
 *   term's waiting changes put in force, such as another kind's, while that is
 *   built differently from the one in force; null otherwise
 * @property {string|null} [difference_penalty] a P-spline's smoothing penalty
 *   in force: "standard" on evenly spaced knots, "general" (Li and Cao's) on
 *   uneven ones, "projected" (the standard one with the polynomials of degree
 *   below m left unpenalised) on knots too uneven for the general one; each
 *   order's, as "m=2 general, m=3 projected", when they differ; null for
 *   other kinds
 * @property {BasisKind|null} [kind] the spline's kind in force
 * @property {boolean|null} [select] whether shrinkage (``select=True``) is on
 * @property {BasisKind[]} [kinds] the kinds the term can be switched to
 * @property {boolean} [select_available] whether Shrink can be turned on or
 *   off on the term as its waiting changes leave it
 * @property {string|null} [select_reason] why it cannot, for its popover
 */
/**
 * A spline's kind: P-spline, B-spline, cubic regression, natural, or the
 * cardinal cubic regression spline, which code can declare and the editor
 * shows but does not offer.
 * @typedef {"ps"|"bs"|"cr"|"ns"|"cr_cardinal"} BasisKind
 */
/**
 * The basis a term's waiting changes put in force, while it differs from the
 * one in force.
 * @typedef {Object} PendingBasis
 * @property {BasisKind} kind
 * @property {boolean} select
 */
/**
 * One basis change, as /stage takes it: another kind, or Shrink
 * turned on or off.
 * @typedef {{kind:Exclude<BasisKind, "cr_cardinal">}|{select:boolean}} BasisParams
 */
/**
 * A spline basis's construction: its ``degree``; its ``ends``, "open" for a
 * P-spline or B-spline (the boundary widened by 0.001 of its range, the knots
 * carried on past it at the end spacings) or "clamped" for a cubic regression
 * or natural spline (each end repeated ``degree + 1`` times); the fitted
 * ``boundary`` in chart x; and an ordered term's ``level_values``, its smooth
 * levels on the spline's own axis, through which chart x maps to it (null on
 * a numeric term).
 * @typedef {Object} KnotBasis
 * @property {number} degree
 * @property {"open"|"clamped"} ends
 * @property {[number, number]} boundary
 * @property {number[]|null} level_values
 */
/**
 * The knots a waiting change leaves on a term, in chart x, ascending.
 * @typedef {Object} PendingKnots
 * @property {number[]} positions
 * @property {number} count
 * @property {KnotStrategy} strategy
 * @property {number|null} alpha
 * @property {boolean} from_editor the knots it leaves were set in the editor, as
 *   `TermKnots.from_editor` reads them after a Refit
 */
/**
 * One knot change, as /stage takes it: a count placed by a rule
 * (``alpha`` with tempered quantiles only), positions placed by hand in chart
 * x, or back to the knots the original model declares.
 * @typedef {{count:number, strategy:KnotRule, alpha?:number}
 *   |{positions:number[]}
 *   |{reset:true}} KnotParams
 */
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
 * @property {boolean} [refit] refit every waiting change at once
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
 * @property {string|null} [predictor] SuperLSS only: the family parameter the step acts in
 * @property {string} [predictor_label] SuperLSS only: "Average" or "Spread"
 * @property {string} [term_label] SuperLSS only: the bare feature name
 */
/**
 * What a term's waiting changes make of it at the next Refit: its groups by
 * label with their member levels, its shaped ranges, and its reference level.
 * @typedef {Object} TermPending
 * @property {Record<string, string[]>|null} groups
 * @property {ShapedRange[]} ranges
 * @property {string|null} reference
 * @property {string[]|null} [specials] the term's special levels once its
 *   waiting changes apply, when one of them takes levels off the curve or back
 * @property {PendingKnots|null} [knots] the term's knots once its waiting
 *   changes apply, while a knot change waits
 * @property {PendingBasis|null} [basis] the term's kind and shrinkage once its
 *   waiting changes apply, while they differ from the ones in force
 */
/**
 * An ordered term's levels fitted free beside its curve (/free_levels): for
 * each level on the curve, its free estimate and interval as relativities on
 * the chart's scale, and the levels whose interval misses the curve.
 * @typedef {Object} FreeLevels
 * @property {string} term
 * @property {string[]} levels
 * @property {number[]} y
 * @property {number[]} lower
 * @property {number[]} upper
 * @property {string[]} flagged
 * @property {string[]} gaps the levels on the curve with no free value, where the line
 *   joining the free estimates breaks
 * @property {number} confidence the chance no interval misses a curve every level lies on
 * @property {number} z
 * @property {boolean} shrunk whether a selection penalty still shrinks the free levels
 * @property {number} fit_token the fit in force it was compared with; a hand edit keeps it
 * @property {string|null} notice what the comparison leaves out or could not measure, and why
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
 * How a SuperLSS term's effect on the linear predictor is shown and edited:
 * `reference_eta` is the parameter's link value at the reference point and
 * `floor` the scale floor of a log-normal spread.
 * @typedef {Object} DisplaySpec
 * @property {'relativity'|'lognormal_cv'|'gamma_cv'|'inverse_theta'} kind
 * @property {number} reference_eta
 * @property {number} floor
 */
/**
 * @typedef {Object} TermPayload
 * @property {number[]} [weights] each point's weight; the selected share's numerator
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
 * @property {TermKnots|null} [knots] the term's knots and the Knots tool's state
 * @property {boolean} [unsmoothed] whether the term has smoothing the Unsmoothed
 *   toggle can switch off: a spline
 * @property {DisplaySpec} [display] SuperLSS terms only: the unit of `y`; absent means relativity
 * @property {number} [baseline] SuperLSS terms only: the display value where the term's effect is zero
 * @property {string} [y_label]
 * @property {'relativity'|'times'|'plain'} [y_format]
 * @property {string} [x_label]
 * @property {string} [title]
 * @property {string} [parameter]
 * @property {string} [parameter_label]
 * @property {string} [parameter_role]
 * @property {string} [term_label]
 * @property {Record<string, string>} [unavailable] SuperLSS terms only: the reason for each term tool that is off
 */
/**
 * A spline term fitted with its smoothing switched off (/unsmoothed). Its line
 * is on the term's own ``x``, with relativities under the same centring as the
 * curve. ``note`` holds what to show when the refit stopped before it
 * converged or a selection penalty shrinks the line, and null otherwise;
 * ``fit_token`` is the fit in force it was fitted beside.
 * @typedef {Object} UnsmoothedLine
 * @property {string} term
 * @property {number[]} x
 * @property {number[]} y
 * @property {string|null} note
 * @property {number} fit_token
 */
/**
 * One term's Unsmoothed line in the browser, for one fit in force: its fit
 * running, the line, the sentence Python refused it with, or the failure of
 * the request itself, which turning the toggle on again retries.
 * @typedef {Object} UnsmoothedEntry
 * @property {number} fit_token
 * @property {'running'|'ready'|'refused'|'failed'} status
 * @property {UnsmoothedLine|null} line
 * @property {string|null} reason
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
 * A parameter's link as the browser inverts it: exp(eta), or floor + exp(eta).
 * @typedef {{kind:"log"} | {kind:"floored_log", floor:number}} DistributionLink
 */
/**
 * @typedef {Object} DistributionParameter
 * @property {string} name the model's parameter name
 * @property {string} label
 * @property {"average"|"spread"} role
 * @property {DistributionLink} link
 */
/**
 * Link-scale predictors, one row per panel row or fan column, in `parameters` order.
 * @typedef {{rows:number[][], columns:number[][]}} DistributionEta
 */
/**
 * The /distribution response (SuperLSS sessions only): one feature's rows and
 * fan columns with every other feature at the reference row, each with the
 * fitted and the edited model's predictors. `kind` "empty" draws only
 * `empty_reason`; its `amount_axis` is null. A reply overtaken by a newer
 * model revision is `{status:"superseded", model_revision}` instead.
 * @typedef {Object} DistributionPanelResponse
 * @property {string} feature
 * @property {"lognormal_mean"|"gamma"|"negative_binomial"} family
 * @property {"numeric"|"levels"|"empty"} kind
 * @property {string|null} empty_reason
 * @property {Array<{label:string, x:number|null, level:string|null}>} rows
 * @property {Array<{x:number|null, level:string|null}>} columns
 * @property {DistributionParameter[]} parameters
 * @property {Record<string, string>} term_keys each parameter's term for this feature
 * @property {{fitted:DistributionEta, edited:DistributionEta}} eta
 * @property {{min_log10:number, max_log10:number, clipped_below:boolean}|null} amount_axis
 * @property {string|null} amount_unit
 * @property {Array<{column:string, value:unknown}>} reference the other features' values
 * @property {string|null} cap_note
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
 * @property {Record<string, string>} [term_versions] each term's version, by which the
 *   API client completes a reply that leaves out the terms it already holds
 * @property {string} selected_term
 * @property {Record<string, TermPayload>} terms
 * @property {Record<string, number[]>} selection
 * @property {UndoRedo} undo_redo
 * @property {TimelineEntry[]} timeline
 * @property {boolean} in_force_is_original
 * @property {PendingStep[]} [pending]
 * @property {{available:boolean, stale:boolean}} [final_fit] whether Export can offer the Final fit model
 * @property {JobStatus[]} [jobs_running] jobs still running when this state was read
 * @property {number} [fit_token] changes only when the fitted model in force is replaced
 * @property {Record<string, string>} [unavailable] SuperLSS sessions only: the reason for each session tool that is off
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
 * @property {FreeLevels|null} freeLevels the last free-level comparison, drawn
 *   while it is shown and its term and the fit in force are the ones in view
 * @property {boolean} showFreeLevels Free levels is on; turned off, the comparison
 *   is kept, so turning it on again for the same term and fit draws it at once
 * @property {boolean} showUnsmoothed the Unsmoothed toggle, off until turned on
 * @property {Record<string, UnsmoothedEntry>} unsmoothed each term's latest line
 *   or refusal, drawn while its fit is the one in force
 * @property {SummaryLevelDisplay} summaryLevelDisplay
 * @property {Record<string, unknown>} zoomByTerm
 * @property {Record<string, string>} groupModeByTerm
 * @property {'summary'|'history'|'settings'|'help'} inspectorPane
 * @property {boolean} inspectorOpen
 * @property {{term:string, payload:TermPayload, selection:number[]}|null} preview
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
 * carry a ``phase`` ("fold", "fold_done", "curves", "fitting", "carrying") and its details.
 * @typedef {Object} JobStatus
 * @property {string} job_id
 * @property {JobKind} kind
 * @property {'running'|'done'|'failed'|'cancelled'} status
 * @property {Array<Record<string, unknown>>} progress
 * @property {Record<string, unknown>|null} result
 * @property {string|null} [error]
 * @property {boolean} [cancel_requested]
 * @property {number} [started_at]
 * @property {number} [elapsed_s]
 */
/**
 * @typedef {Object} CVFoldRow
 * @property {number} fold
 * @property {number} n_train
 * @property {number} n_test
 * @property {number|null} fit_time_s
 * @property {boolean} converged
 * @property {number|null} effective_df
 * @property {number|null} [n_reml_iter] rounds of smoothing search, under fit_reml
 * @property {Record<string, number|null>} scores
 */
/**
 * One cross-validation run: the supplied result, or Run CV on the current model.
 * @typedef {Object} CVResultPayload
 * @property {string} label
 * @property {'supplied'|'run'} origin
 * @property {number|null} model_revision
 * @property {boolean} stale
 * @property {boolean} [held_out] every row is held out by exactly one fold
 * @property {boolean} [live] a run still going: its folds so far
 * @property {number} [n_folds] how many folds a live run has in all
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
 * @property {string} [label] how the tab names the term; its name when absent
 * @property {string} [short] its name within its group's list
 * @property {string|null} [group] the heading it is listed under; a SuperGLM term has none
 * @property {'average'|'spread'|null} [role]
 * @property {'levels'|'continuous'} kind
 * @property {number[]|null} x
 * @property {string[]|null} levels
 * @property {number[]} weights
 * @property {CVFoldCurve[]} folds
 * @property {number[]} fit
 * @property {number[]|null} edited
 * @property {string} [value_label] what the curve is a relativity of
 * @property {boolean} [held] a hand edit Run CV put back on every fold; it has no fold gap
 * @property {number|null} fold_gap
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
 * @property {Array<{name:string, label:string, short:string, lower_is_better:boolean|null}>} metrics a null direction means closer to 1 is better
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
