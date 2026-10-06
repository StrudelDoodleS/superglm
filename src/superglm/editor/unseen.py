"""Where a categorical's new levels go: the editor's "New levels →" control (spec addendum S6).

A plain categorical's ``unseen`` policy says what prediction does with a level
the fit never saw: ``"error"`` refuses it (Refuse), ``"base"`` rates it at the
reference (Reference), and a group label gives it that group's effect. The
choice changes no fitted value, so it is made on the in-force fitted spec with
no refit, as one entry on the session's one Undo history. The opened model is
never changed: the in-force model becomes a copy carrying the new policy.

The choice changes predictions on data holding new levels, validation and
test rows included, so it moves the model revision like any edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from superglm.editor._types import EditRecord
from superglm.editor.apply import _copy_model_for_editor_edits
from superglm.editor.collapse import _require_not_interaction_parent
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.features.categorical import Categorical

if TYPE_CHECKING:
    from superglm.editor.session import EditorSession

OPERATION = "set_unseen"
# The library's named policies, by the names the control gives them.
_CHOICE_NAMES = {"error": "Refuse", "base": "Reference"}

_NOT_CATEGORICAL = "Only a categorical term has a New levels choice; {term!r} is not one."
_NOT_A_GROUP = (
    "{choice!r} is not a group of {term!r}. Choose Refuse, Reference or one of its groups."
)
_WAITING = "Refit or undo the waiting changes to {term!r} before choosing where its new levels go."
_GROUP_REMOVED = (
    "New levels of {term!r} go to the group {group!r}, which that change would remove. "
    "Choose where new levels go first, then make the change."
)


@dataclass
class UnseenChoice(EditRecord):
    """One New levels choice on the edit history, with the in-force model on each side.

    It moves no curve value, so its indices are empty. ``params`` holds the
    policy chosen (``"unseen"``) and the one it replaced (``"previous"``).
    """

    model_before: Any = None
    model_after: Any = None

    @property
    def label(self) -> str:
        """The History message: ``"New levels → Other"``."""
        return f"New levels → {choice_name(self.params['unseen'])}"


def choice_name(policy: str) -> str:
    """The control's name for ``policy``: Refuse, Reference, or the group's label."""
    return _CHOICE_NAMES.get(policy, policy)


def unseen_payload(session: EditorSession, term: str) -> dict[str, Any] | None:
    """The term's New levels control, or None where it is not offered.

    ``policy`` is the in-force policy and ``choices`` the control's entries,
    Refuse and Reference first, then one per group. ``reason`` is the fixed
    sentence saying why the choice cannot be made now, else None.
    """
    spec = session.model._specs.get(term)
    if not isinstance(spec, Categorical):
        return None
    values = _choices(session, term, spec)
    if spec.unseen not in values:
        values.append(spec.unseen)
    return {
        "policy": spec.unseen,
        "choices": [{"value": value, "label": choice_name(value)} for value in values],
        "reason": _unavailable_reason(session, term),
    }


def set_unseen(session: EditorSession, term: str, policy: Any) -> UnseenChoice | None:
    """Send ``term``'s new levels to ``policy`` in the in-force model, as one history entry.

    ``policy`` is ``"error"``, ``"base"`` or the label of one of the term's
    groups. Nothing is fitted. Choosing the policy in force changes nothing
    and returns None.
    """
    session._require_term(term)
    spec = session.model._specs[term]
    if not isinstance(spec, Categorical):
        raise EditorTypeError(_NOT_CATEGORICAL.format(term=term))
    reason = _unavailable_reason(session, term)
    if reason is not None:
        raise EditorValueError(reason)
    if not isinstance(policy, str) or policy not in _choices(session, term, spec):
        raise EditorValueError(_NOT_A_GROUP.format(choice=policy, term=term))
    if policy == spec.unseen:
        return None
    before = session.model
    record = UnseenChoice(
        term=term,
        operation=OPERATION,
        indices=np.empty(0, dtype=np.intp),
        before=np.empty(0, dtype=np.float64),
        after=np.empty(0, dtype=np.float64),
        params={"unseen": policy, "previous": spec.unseen},
        model_before=before,
        model_after=model_with_unseen(before, term, policy),
    )
    session.history.append(record)
    session._end_redo()
    session.model = record.model_after
    session._advance_model_revision()
    return record


def step_across(session: EditorSession, record: UnseenChoice, *, undo: bool) -> None:
    """Put the in-force model from the other side of ``record`` back: before it on Undo.

    Undo and Redo walk the history in time order, so the in-force model is
    the record's own and its other side goes back as it was, the opened model
    included. An undo limited to one term can take a choice out of that
    order; the in-force model then keeps the later choices and only this
    term's policy changes.
    """
    here, there = (
        (record.model_after, record.model_before)
        if undo
        else (record.model_before, record.model_after)
    )
    policy = record.params["previous" if undo else "unseen"]
    session.model = (
        there if session.model is here else model_with_unseen(session.model, record.term, policy)
    )
    session._advance_model_revision()


def model_with_unseen(model, term: str, policy: str):
    """A copy of fitted ``model`` whose ``term`` sends new levels to ``policy``.

    The copy's declaration says so too, so a refit of it (Run CV, Final fit)
    keeps the choice. ``model`` is not changed. The fit is the same, so the
    copy shares its row-length outputs rather than holding another copy for
    each choice; the prediction plan, which holds the specs, is its own.
    """
    copied = _copy_model_for_editor_edits(model, share_fit_outputs=True)
    spec = copied._specs[term]
    spec.unseen = policy
    declared = dict(getattr(getattr(copied, "_config", None), "feature_templates", ())).get(term)
    if isinstance(declared, Categorical):
        declared.unseen = policy
    try:
        # The fit's own check, which the copy never meets.
        spec._require_unseen_group()
    except ValueError as exc:
        raise EditorValueError(_NOT_A_GROUP.format(choice=policy, term=term)) from exc
    return copied


def require_group_kept(term: str, draft) -> None:
    """Refuse a waiting change whose draft no longer has the group new levels go to.

    A collapse can merge that group into another and an ungroup can take it
    apart; the refit would then refuse the policy, so the change is refused
    now, by name.
    """
    if not isinstance(draft, Categorical) or draft.unseen in _CHOICE_NAMES:
        return
    grouping = draft._grouping
    if grouping is None or draft.unseen not in grouping.grouped_levels:
        raise EditorValueError(_GROUP_REMOVED.format(term=term, group=draft.unseen))


def _choices(session: EditorSession, term: str, spec: Categorical) -> list[str]:
    """Refuse, Reference and ``spec``'s groups, then a one-level group the term sends new levels to.

    A one-level group of its own name reads as a plain level, so it is not
    offered as a group; but where the opened model or the in-force one sends
    new levels to it, choosing it again must stay possible.
    """
    values = [*_CHOICE_NAMES, *_groups(spec)]
    opened = session.reference_model._specs.get(term)
    for policy in (getattr(opened, "unseen", None), spec.unseen):
        if isinstance(policy, str) and policy not in values and _receives(spec, policy):
            values.append(policy)
    return values


def _receives(spec: Categorical, label: str) -> bool:
    """Whether ``label`` is one of ``spec``'s fitted groups, so new levels can go to it."""
    grouping = spec._grouping
    return grouping is not None and label in grouping.grouped_levels and label in spec._levels


def _groups(spec: Categorical) -> list[str]:
    """The labels of ``spec``'s groups that new levels can go to, in fitted order.

    A group merges or renames levels and has a place in the fitted levels. A
    group named "error" or "base" would read as that policy, so it is not
    offered.
    """
    grouping = spec._grouping
    if grouping is None:
        return []
    return [
        label
        for label in grouping.grouped_levels
        if isinstance(label, str)
        and label not in _CHOICE_NAMES
        and label in spec._levels
        and [str(member) for member in grouping.group_to_originals[label]] != [label]
    ]


def _unavailable_reason(session: EditorSession, term: str) -> str | None:
    """Why the choice cannot be made now, as one fixed sentence; None when it can."""
    try:
        # An interaction keeps its own copy of the parent's policy.
        _require_not_interaction_parent(session.model, term, operation="choose where new levels go")
    except EditorValueError as exc:
        return exc.public_message
    if any(step.term == term for step in session.pending):
        # A waiting change's draft carries the policy it was built with.
        return _WAITING.format(term=term)
    return None
