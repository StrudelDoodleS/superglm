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

from superglm.editor._types import EditableTerm, EditRecord, PendingStep, StructuralStep
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
from superglm.features._spline_ranges import (
    ConstantRangesError,
    NarrowGapError,
    RangeError,
    UndeterminedRangeError,
    UndeterminedStretchError,
)

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
_COLLAPSE_SENTENCES = (
    (UndeterminedRangeError, _COLLAPSE_IN_RANGE_REFUSED),
    (UndeterminedStretchError, _COLLAPSE_STRETCH_REFUSED),
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


def _range_refusal(exc: BaseException, sentences) -> str | None:
    """The sentence for the first range refusal on ``exc``'s cause chain, or None.

    The fit re-raises a term's build refusal to name the term, so the
    library's own error can sit one or more causes down. Anything else, a
    solver failure say, is not a range refusal and keeps its own error.
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
    # A new action ends the future of whatever was undone.
    session.redo_stack.clear()
    session.pending_redo.clear()
    session.structure_redo.clear()
    return step


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
