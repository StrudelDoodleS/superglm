"""Structural changes that wait for one Refit, and the time order of the editor history.

Each function takes the :class:`~superglm.editor.session.EditorSession` first;
the session keeps one-line methods that delegate here (spec D1, D11). A
waiting change is built on its term's draft, the last waiting change's spec,
so changes to one term compose, and nothing is fitted while it waits. Undo
and Redo take edits, waiting changes and applied steps back in time order.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from superglm.editor._types import (
    EditableTerm,
    EditRecord,
    PendingStep,
    SessionState,
    StructuralStep,
)
from superglm.editor.carry import carried_curve
from superglm.editor.collapse import (
    collapsed_feature_spec,
    reference_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.errors import (
    EditorClientError,
    EditorKeyError,
    EditorTypeError,
    EditorValueError,
)
from superglm.editor.shapes import shaped_feature_spec
from superglm.editor.unseen import require_group_kept
from superglm.features._spline_ranges import (
    ConstantRangesError,
    NarrowGapError,
    RangeError,
    UndeterminedRangeError,
    UndeterminedStretchError,
)
from superglm.features.ordered_categorical import GroupNamedAsSpecialError

if TYPE_CHECKING:
    from superglm.editor.session import EditorSession

_SHAPE_REFUSED = (
    "That range cannot be shaped. Choose a range with more distinct values, or a lower degree."
)
_CONSTANT_REFUSED = (
    "A Flat range over the whole axis leaves the term one constant, which the intercept "
    "already carries. Choose a Line, or leave part of the axis free."
)
_STRETCH_REFUSED = (
    "That range leaves too few values beside it to fit the rest of the curve. "
    "Widen it to the end of the axis or to the next shaped range."
)

_NARROW_REFUSED = (
    "That range's edge falls too close to the end of the axis, another range or a knot "
    "to fit stably. Move the edge, or let it meet the next range."
)
_COLLAPSE_IN_RANGE_REFUSED = (
    "Collapsing those bands leaves a shaped range too few bands for its shape. "
    "Collapse bands outside it, or undo the range first."
)
_COLLAPSE_STRETCH_REFUSED = (
    "Collapsing those bands leaves too few bands beside a shaped range to fit the rest "
    "of the curve. Collapse fewer bands, or undo the range first."
)
# Most specific first: every range refusal is a RangeError.
_SHAPE_SENTENCES = (
    (UndeterminedStretchError, _STRETCH_REFUSED),
    (ConstantRangesError, _CONSTANT_REFUSED),
    (NarrowGapError, _NARROW_REFUSED),
    (RangeError, _SHAPE_REFUSED),
)
_COLLAPSE_NAMED_AS_SPECIAL = (
    "That group name is how a free level of this term is spelled, so the free level "
    "would claim the group's rows. Give the group another name."
)
_COLLAPSE_SENTENCES = (
    (UndeterminedRangeError, _COLLAPSE_IN_RANGE_REFUSED),
    (UndeterminedStretchError, _COLLAPSE_STRETCH_REFUSED),
    (GroupNamedAsSpecialError, _COLLAPSE_NAMED_AS_SPECIAL),
)
_STAGED_SENTENCES = {"collapse": _COLLAPSE_SENTENCES, "shape": _SHAPE_SENTENCES}
# A waiting change's operation, and the operation its step carries when a
# legacy call refits it at once (``replace_with_*``).
_REFIT_AT_ONCE = {
    "collapse": "collapse_levels",
    "ungroup": "ungroup_levels",
    "set_reference": "set_reference",
    "shape": "shape_range",
}
_UNKNOWN_ENTRY = "Unknown history entry."
_NOTE_LIMIT = 2000
_REFIT_REFUSED = "The refit was refused. Undo the last waiting change and try again."
_NOTHING_WAITING = "No changes are waiting for a refit."


def _refit_label(count: int) -> str:
    return f"Refit · {count} change{'' if count == 1 else 's'}"


def _range_refusal(exc: BaseException, sentences) -> str | None:
    """The sentence for the first refusal ``sentences`` names on ``exc``'s cause chain, or None.

    The fit re-raises a term's build refusal to name the term, so the
    library's own error can sit one or more causes down. Anything else, a
    solver failure say, is not one of these refusals and keeps its own error.
    """
    return next(filter(None, (_sentence_for(cause, sentences) for cause in _causes(exc))), None)


def _sentence_for(exc: BaseException, sentences) -> str | None:
    return next((sentence for kind, sentence in sentences if isinstance(exc, kind)), None)


def _causes(exc: BaseException | None):
    while exc is not None:
        yield exc
        exc = exc.__cause__


def draft_spec(session: EditorSession, term: str):
    """``term``'s spec as waiting changes leave it: the last one's draft, else the fitted spec."""
    session._require_term(term)
    waiting = _waiting_draft(session, term)
    return session.model._specs[term] if waiting is None else waiting


def stage_structural(
    session: EditorSession,
    operation: str,
    term: str,
    params: dict[str, Any],
    *,
    keep_reference: bool = True,
    X=None,
) -> PendingStep:
    """Stage one structural change to wait for a Refit.

    ``operation`` is ``"collapse"`` (``levels``, optional ``group_label``),
    ``"ungroup"`` (``levels``), ``"set_reference"`` (``level``) or
    ``"shape"`` (``lo``, ``hi``, ``degree``, optional ``join``); levels are
    display labels. A change its builder refuses is refused now, with
    today's sentence. Nothing is fitted: the model, the curves and the model
    revision stay as they are. ``X`` is the frame the refit will read
    (default: the session's refit data).
    """
    editable = session._require_term(term)
    if operation not in _REFIT_AT_ONCE:
        raise EditorValueError(f"Unknown structural change: {operation!r}")
    if not isinstance(params, dict):
        raise EditorValueError("params must be an object.")
    X_ref = session._resolve_refit_data(None, None, None, None)[0] if X is None else X
    try:
        replacement, metadata = _draft_for(
            session, operation, editable, params, keep_reference=keep_reference, X=X_ref
        )
    except EditorClientError:
        raise
    except ValueError as exc:
        sentence = _range_refusal(exc, _STAGED_SENTENCES.get(operation, ()))
        if sentence is None:
            raise
        raise EditorValueError(sentence) from exc
    require_group_kept(term, replacement)
    step = PendingStep(
        operation=operation,
        term=term,
        label=str(metadata["label"]),
        params=_label_params(operation, metadata),
        draft_spec=replacement,
        history_position=len(session.history),
        metadata=dict(metadata),
    )
    session.pending.append(step)
    session._end_redo()
    return step


def refit_pending(
    session: EditorSession, *, method: str = "auto", **refit_kwargs: Any
) -> StructuralStep:
    """Apply every waiting change in one fit, as one structural step (spec D1).

    Undo of the step brings the changes back as waiting; Redo puts the refit
    back without fitting. Hand edits on terms no change restructured are
    carried over (spec D2) as one more entry. A fit-time refusal leaves the
    waiting changes and the model as they were and raises one fixed
    sentence; Python callers keep the library's error as its cause.
    ``refit_kwargs`` are ``X``, ``y``, ``sample_weight``, ``offset``,
    ``lambda1``, ``lambda2`` and fit keywords.
    """
    try:
        return _apply_pending(session, method=method, **refit_kwargs)
    except EditorClientError:
        raise
    except ValueError as exc:
        raise EditorValueError(_REFIT_REFUSED) from exc


def stage_and_refit(
    session: EditorSession,
    operation: str,
    term: str,
    params: dict[str, Any],
    *,
    keep_reference: bool = True,
    **refit_kwargs: Any,
):
    """Stage one change and refit every waiting change at once: the legacy ``replace_with_*`` calls.

    The step keeps the state from before the change was staged, so one Undo
    takes the whole call back and any earlier waiting changes wait again. A
    refusal leaves nothing staged; a library range refusal reads as the
    operation's own fixed sentence, as it always has.
    """
    before = session._capture_state()
    future = (list(session.redo_stack), list(session.pending_redo), list(session.structure_redo))
    change = None
    try:
        change = stage_structural(
            session, operation, term, params, keep_reference=keep_reference, X=refit_kwargs.get("X")
        )
        _apply_pending(session, before=before, alone=change, **refit_kwargs)
    except BaseException as exc:
        if change is not None and session.pending and session.pending[-1] is change:
            session.pending.pop()
            session.redo_stack, session.pending_redo, session.structure_redo = future
        refused = isinstance(exc, ValueError) and not isinstance(exc, EditorClientError)
        sentence = _range_refusal(exc, _STAGED_SENTENCES.get(operation, ())) if refused else None
        if sentence is None:
            raise
        raise EditorValueError(sentence) from exc
    return session.model


def put_in_force(
    session: EditorSession,
    model,
    *,
    restructured: set[str],
    at_once: bool = False,
    **step: Any,
) -> StructuralStep:
    """Push ``model`` as one structural step, then carry over edits on the terms it left alone.

    ``step`` goes to the session's ``_push_structure``: ``operation``,
    ``term``, ``label`` and optionally ``state``, ``changes`` and ``step_id``.
    A change refitted at once (``at_once``, the legacy ``replace_with_*``
    calls) is one step that one Undo takes back, so its carry-over is part of
    that step; a Refit's carry-over is an entry of its own.
    """
    previous = session.terms
    held = [name for name in session.edited_terms() if name not in restructured]
    session._push_structure(model, **step)
    pushed = session.structure_history[-1]
    _carry_edits(session, {name: previous[name] for name in held}, own_entry=not at_once)
    return pushed


def selected_labels(session: EditorSession, term: str) -> list[str]:
    """The selected levels' labels in display order: how a waiting change names them."""
    editable = session._require_term(term)
    idx = np.unique(session._require_selection(term))
    if editable.levels is None:
        raise EditorTypeError(f"Term {term!r} does not expose categorical levels.")
    return [str(editable.levels[int(index)]) for index in idx]


def undo_target(session: EditorSession) -> EditRecord | PendingStep | StructuralStep | None:
    """What a plain undo takes next: the latest edit or waiting change, else a step."""
    if session.pending and session.pending[-1].history_position >= len(session.history):
        return session.pending[-1]
    if session.history:
        return session.history[-1]
    return session.structure_history[-1] if session.structure_history else None


def redo_target(session: EditorSession) -> EditRecord | PendingStep | StructuralStep | None:
    """What a plain redo puts back next, in the reverse of the order Undo took."""
    if session.pending_redo and (
        not session.redo_stack or session.pending_redo[-1].history_position <= len(session.history)
    ):
        return session.pending_redo[-1]
    if session.redo_stack:
        return session.redo_stack[-1]
    return session.structure_redo[-1] if session.structure_redo else None


def timeline_items(
    session: EditorSession,
) -> tuple[list[tuple[Any, str]], list[tuple[Any, str]]]:
    """Every action, oldest first, split at the current position, each with its status.

    Done: for each applied step, the edits made before it and the waiting
    changes it applied, as they happened, then the step; then the live
    edits and the changes still waiting. Undone: what Redo would put back,
    in the order it would. A change refitted at once on its own shares its
    step's id and is listed once, as the step. Statuses are ``"edit"``,
    ``"waiting"`` and ``"applied"``.
    """
    done: list[tuple[Any, str]] = []
    for step in session.structure_history:
        applied = [change for change in step.changes if change.step_id != step.step_id]
        done += _with_status(_in_time_order(step.state.history, applied), "applied")
        done.append((step, "applied"))
    done += _with_status(_in_time_order(session.history, session.pending), "waiting")
    undone = _with_status(
        _redo_order(len(session.history), session.redo_stack, session.pending_redo), "waiting"
    )
    for step in reversed(session.structure_redo):
        undone.append((step, "applied"))
        state = step.state
        undone += _with_status(
            _redo_order(len(state.history), state.redo_stack, state.pending_redo), "waiting"
        )
    return done, undone


def set_step_note(session: EditorSession, step_id: str, note: str | None) -> None:
    """Write ``note`` on the timeline entry ``step_id``; an empty note removes it.

    Notes survive undo and redo, and travel with an exported model
    (:func:`editor_history_records`).
    """
    done, undone = timeline_items(session)
    if step_id not in {item.step_id for item, _ in (*done, *undone)}:
        raise EditorKeyError(_UNKNOWN_ENTRY)
    text = "" if note is None else str(note).strip()
    if len(text) > _NOTE_LIMIT:
        raise EditorValueError(f"A note can be at most {_NOTE_LIMIT} characters.")
    if text:
        session.step_notes[step_id] = text
    else:
        session.step_notes.pop(step_id, None)


def editor_history_records(session: EditorSession) -> list[dict[str, Any]]:
    """The timeline up to now, oldest first, as an exported model's ``_editor_history``.

    One dict per edit, waiting change and applied step: id, time (ISO 8601,
    UTC), operation, term, message, note, status and predictor (None until
    the SuperLSS editor names one). What Redo would put back is not history.
    """
    done, _ = timeline_items(session)
    return [
        {
            "id": item.step_id,
            "time": datetime.fromtimestamp(item.created_at, tz=UTC).isoformat(timespec="seconds"),
            "operation": item.operation,
            "term": item.term,
            "message": item.label,
            "note": session.step_notes.get(item.step_id),
            "status": status,
            "predictor": getattr(item, "predictor", None),
        }
        for item, status in done
    ]


def _apply_pending(
    session: EditorSession,
    *,
    before: SessionState | None = None,
    alone: PendingStep | None = None,
    **refit_kwargs: Any,
) -> StructuralStep:
    """Fit every waiting change in one refit and put it in force as one structural step.

    ``before`` and ``alone`` come from a change refitted at once: the step
    keeps the state from before that change was staged and, when it is the
    only change, is that change, under its id, operation and label.
    Nothing changes unless the fit succeeds.
    """
    if not session.pending:
        raise EditorValueError(_NOTHING_WAITING)
    changes = tuple(session.pending)
    # A later change on a term was built on the earlier ones' draft, so the
    # last one holds them all.
    drafts = {change.term: change.draft_spec for change in changes}
    refit_model, method_used = session._refit_with_drafts(drafts, **refit_kwargs)
    if alone is not None and len(changes) == 1 and changes[0] is alone:
        identity = {
            "operation": _REFIT_AT_ONCE[alone.operation],
            "label": alone.label,
            "step_id": alone.step_id,
        }
        refit_model._editor_step = {**alone.metadata, "method": method_used}
    else:
        label = _refit_label(len(changes))
        identity = {"operation": "refit_pending", "label": label, "step_id": None}
        refit_model._editor_step = {
            "format": "superglm.editor.refit.v1",
            "label": label,
            "changes": [dict(change.metadata) for change in changes],
            "method": method_used,
            "message": "The waiting structural changes were applied and the full model was refit.",
        }
    term = next(iter(drafts)) if len(drafts) == 1 else None
    return put_in_force(
        session,
        refit_model,
        restructured=set(drafts),
        at_once=before is not None,
        state=before,
        term=term,
        changes=changes,
        **identity,
    )


def _carry_edits(
    session: EditorSession, edited: dict[str, EditableTerm], *, own_entry: bool = True
) -> None:
    """Re-apply hand-edited curves over a refit, as one entry Undo takes back (spec D2).

    ``edited`` holds the edited terms the refit did not restructure. Each
    keeps its rows and ``n_points``, so its grid and labels match and its
    curve goes back exactly (``carried_curve``); a term whose grid moved
    anyway is left at the refit. Undo of the entry returns the carried
    terms to the refitted curves; Undo of the refit then puts back every
    edit, those on restructured terms included.

    Without ``own_entry`` the curves join the step just pushed: its state
    is the one from before it, so the live terms are its own fresh ones,
    and Undo of that step takes the carry-over back with the change.
    """
    carried = {}
    for name, term in edited.items():
        curve = carried_curve(term, session.terms[name])
        if curve is not None:
            carried[name] = curve
    if not carried:
        return
    if not own_entry:
        for name, curve in carried.items():
            session.terms[name].edited_log_effect = curve
        return
    refitted = session._capture_state()
    # A fresh dict of copies: the entry's state keeps the refitted terms.
    session.terms = {name: term.copy() for name, term in session.terms.items()}
    for name, curve in carried.items():
        session.terms[name].edited_log_effect = curve
    label = f"Hand edits carried over: {', '.join(carried)}"
    session.structure_history.append(StructuralStep(refitted, "carry_edits", None, label))
    session._advance_model_revision()


def _waiting_draft(session: EditorSession, term: str):
    """The last waiting change's draft for ``term``, or None when none waits."""
    return next((step.draft_spec for step in reversed(session.pending) if step.term == term), None)


def _draft_for(
    session: EditorSession, operation: str, editable: EditableTerm, params, *, keep_reference, X
):
    """``operation``'s builder on the term's draft: the replacement spec and its metadata."""
    draft = _waiting_draft(session, editable.name)
    if operation == "collapse":
        return collapsed_feature_spec(
            session.model,
            editable,
            _level_indices(editable, _param(params, "levels")),
            X=X,
            group_label=params.get("group_label"),
            draft_spec=draft,
            keep_reference=keep_reference,
        )
    if operation == "ungroup":
        return ungrouped_feature_spec(
            session.model,
            editable,
            _level_indices(editable, _param(params, "levels")),
            X=X,
            draft_spec=draft,
            keep_reference=keep_reference,
        )
    if operation == "set_reference":
        level = str(_param(params, "level"))
        return reference_feature_spec(session.model, editable, level, X=X, draft_spec=draft)
    return shaped_feature_spec(
        session.model,
        editable.name,
        lo=_param(params, "lo"),
        hi=_param(params, "hi"),
        degree=_param(params, "degree"),
        join=params.get("join", "tangent"),
        X=X,
        draft_spec=draft,
    )


def _level_indices(editable: EditableTerm, labels) -> NDArray[np.intp]:
    """Display indices of ``labels``, which a waiting change names its levels by."""
    if editable.levels is None:
        raise EditorTypeError(f"Term {editable.name!r} does not expose categorical levels.")
    if not isinstance(labels, list | tuple):
        raise EditorValueError("levels must be a list of level labels.")
    position = {level: index for index, level in enumerate(editable.levels)}
    missing = [label for label in labels if str(label) not in position]
    if missing:
        raise EditorKeyError(f"Unknown level(s) for term {editable.name!r}: {missing}")
    return np.array([position[str(label)] for label in labels], dtype=np.intp)


def _param(params: dict[str, Any], name: str) -> Any:
    if name not in params:
        raise EditorValueError(f"Missing required field: {name}.")
    return params[name]


def _label_params(operation: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """A waiting change's parameters by label, as its builder resolved them."""
    if operation == "collapse":
        return {"levels": list(metadata["levels"]), "group_label": metadata["group_label"]}
    if operation == "ungroup":
        return {"levels": list(metadata["levels"])}
    if operation == "set_reference":
        return {"level": metadata["level"]}
    return {name: metadata[name] for name in ("lo", "hi", "degree", "join")}


def _in_time_order(records, waiting) -> list:
    """Edits and waiting changes as they happened: one staged after k edits follows the k-th."""
    queue = list(waiting)
    ordered: list = []
    for index, record in enumerate(records):
        while queue and queue[0].history_position <= index:
            ordered.append(queue.pop(0))
        ordered.append(record)
    return [*ordered, *queue]


def _redo_order(n_live: int, records, waiting) -> list:
    """What Redo would put back, in order: ``redo_target``'s rule run forward."""
    records, waiting, ordered = list(records), list(waiting), []
    while records or waiting:
        if waiting and (not records or waiting[-1].history_position <= n_live):
            ordered.append(waiting.pop())
        else:
            ordered.append(records.pop())
            n_live += 1
    return ordered


def _with_status(items, status: str) -> list[tuple[Any, str]]:
    """Each item with its timeline status: an edit is an ``"edit"``, a change ``status``."""
    return [(item, "edit" if isinstance(item, EditRecord) else status) for item in items]
