"""Special levels of an ordered term: taken off its curve, or put back on it.

A special level is fitted off the term's curve with a free estimate of its
own, as ``OrderedCategorical(specials=[...])`` declares. The editor's "Make
special" takes levels off the curve and "Back on the curve" puts them back,
each a structural change that waits for Refit like a collapse. A level taken
off keeps its place on the axis (:data:`~superglm.features.rebuild.FREED_LEVELS_ATTRIBUTE`),
so it can go back; a level the declaration makes special has no place, and
only the declaration can put it on the curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from superglm._frame import as_eager_frame
from superglm.editor._types import EditableTerm
from superglm.editor.collapse import (
    _mark_kept,
    _reference_to_keep,
    _require_not_interaction_parent,
    _stated_break_bands,
)
from superglm.editor.errors import EditorTypeError, EditorValueError
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.piecewise import Piecewise
from superglm.features.rebuild import (
    freed_levels,
    full_level_order,
    rebuilt_ordered_spec,
    special_labels,
)

_NOT_ORDERED = (
    "Make special is for ordered terms: every level of {term!r} already has an estimate of its own."
)
_ALREADY = "{level!r} is already a special level of {term!r}."
_NOT_SPECIAL = "{level!r} is on the curve of {term!r} already."
_DECLARED = (
    "{level!r} is declared special in {term!r}, so it has no place on the curve; "
    "declare it in the term's order to put it there."
)
_GROUPED = "{level!r} is in group {group!r} of {term!r}; ungroup it first."
_REFERENCE = (
    "{level!r} is the reference of {term!r}, which must stay on the curve; "
    "set another reference first."
)
_BREAK = (
    "{term!r} has a break, knot or shaped-range edge at {level!r}, so it can't leave the curve; "
    "move or remove it first."
)
_POSITIONAL_BREAKS = (
    "{term!r} states its breaks by position, which a level leaving the curve would move; "
    "state them by band name to make a level special."
)
_TOO_FEW = "{term!r} needs at least two levels on its curve; make fewer levels special."
_NO_ROWS = (
    "{level!r} has no rows in the data the refit reads, so it has nothing to estimate "
    "a free value from."
)
_INSIDE_GROUP = (
    "{level!r} would go back between members of group {group!r} of {term!r}; ungroup it first."
)
_NO_LEVELS = "Select the levels to {action}."


def special_feature_spec(
    model,
    term: EditableTerm,
    labels: list[str],
    *,
    special: bool,
    X,
    draft_spec=None,
) -> tuple[Any, dict[str, Any]]:
    """A fresh spec for ``term`` with ``labels`` made special, or put back on the curve.

    ``labels`` are displayed level labels. The reference the in-force fit
    resolved is kept. ``draft_spec`` is the term's spec as waiting changes
    leave it (None: the fitted spec), so changes to one term compose. A
    request the term cannot take is refused in a fixed sentence.
    """
    fitted = model._specs[term.name]
    spec = fitted if draft_spec is None else draft_spec
    action = "make special" if special else "put back on the curve"
    if not isinstance(spec, OrderedCategorical):
        raise EditorTypeError(_NOT_ORDERED.format(term=term.name))
    _require_not_interaction_parent(model, term.name, operation=f"{action} levels")
    chosen = list(dict.fromkeys(str(label) for label in labels))
    if not chosen:
        raise EditorValueError(_NO_LEVELS.format(action=action))
    grouping = getattr(spec, "_grouping", None)
    for label in chosen:
        _require_alone(spec, grouping, term, label)
    declared, kept, level = _reference_to_keep(fitted, spec, term)
    frame = as_eager_frame(X)
    frame.require_columns((term.name,))
    column = frame.column_array(term.name)
    if special:
        present = {str(value) for value in pd.unique(np.asarray(column, dtype=object))}
        _require_free_to_leave(spec, term, chosen, declared, present)
    else:
        _require_free_to_return(spec, grouping, term, chosen)
    changes = {"freed": tuple(chosen)} if special else {"returned": tuple(chosen)}
    replacement = rebuilt_ordered_spec(
        spec,
        grouping=grouping,
        base=declared,
        data=column,
        level=level,
        **changes,
    )
    _mark_kept(replacement, declared, kept, grouping, level=level)
    joined = " + ".join(chosen)
    label = (
        f"make {joined} special in {term.name}"
        if special
        else f"put {joined} back on the curve in {term.name}"
    )
    message = (
        f"{joined} of {term.name} {'were' if len(chosen) > 1 else 'was'} "
        f"{'made special' if special else 'put back on the curve'} and the full model was refit."
    )
    metadata = {
        "format": "superglm.editor.special_levels.v1",
        "term": term.name,
        "levels": chosen,
        "special": special,
        "label": label,
        "message": message,
    }
    return replacement, metadata


def returnable_levels(spec) -> list[str]:
    """The special levels of ``spec`` that can go back on its curve, as the editor shows them."""
    if not isinstance(spec, OrderedCategorical):
        return []
    freed = freed_levels(spec)
    return [str(level) for level in spec._special_display if str(level) in freed]


def _require_alone(spec, grouping, term: EditableTerm, label: str) -> None:
    """Refuse a label that is not one of the term's levels, or stands in a group."""
    if grouping is not None and len(grouping.group_to_originals.get(label, ())) > 1:
        raise EditorValueError(_GROUPED.format(level=label, group=label, term=term.name))
    if term.levels is None or label not in term.levels:
        raise EditorValueError(f"{label!r} is not a level of term {term.name!r}.")
    group = None if grouping is None else grouping.original_to_group.get(label)
    if group is not None and len(grouping.group_to_originals.get(group, ())) > 1:
        raise EditorValueError(_GROUPED.format(level=label, group=group, term=term.name))


def _require_free_to_leave(
    spec, term: EditableTerm, chosen: list[str], reference, present: set[str]
) -> None:
    """Refuse levels that cannot leave the curve. ``present`` holds the levels the refit's rows hold."""
    specials = special_labels(spec)
    breaks = set(_stated_break_bands(spec))
    basis = getattr(spec, "_spline_obj", None)
    if isinstance(basis, Piecewise) and isinstance(basis.breaks, list):
        if any(not isinstance(entry, str) for entry in basis.breaks):
            raise EditorValueError(_POSITIONAL_BREAKS.format(term=term.name))
    for label in chosen:
        if label in specials:
            raise EditorValueError(_ALREADY.format(level=label, term=term.name))
        if label == str(reference):
            raise EditorValueError(_REFERENCE.format(level=label, term=term.name))
        if label in breaks:
            raise EditorValueError(_BREAK.format(level=label, term=term.name))
        if label not in present:
            raise EditorValueError(_NO_ROWS.format(level=label, term=term.name))
    on_curve = [
        level for level in full_level_order(spec) if level not in specials and level not in chosen
    ]
    grouping = getattr(spec, "_grouping", None)
    bands = {
        level if grouping is None else str(grouping.original_to_group.get(level, level))
        for level in on_curve
    }
    if len(bands) < 2:
        raise EditorValueError(_TOO_FEW.format(term=term.name))


def _require_free_to_return(spec, grouping, term: EditableTerm, chosen: list[str]) -> None:
    specials = special_labels(spec)
    freed = freed_levels(spec)
    for label in chosen:
        if label not in specials:
            raise EditorValueError(_NOT_SPECIAL.format(level=label, term=term.name))
        if label not in freed:
            raise EditorValueError(_DECLARED.format(level=label, term=term.name))
    if grouping is None:
        return
    order = full_level_order(spec)
    on_curve = [level for level in order if level not in specials or level in chosen]
    group_of = {
        level: str(grouping.original_to_group.get(level, level))
        for level in on_curve
        if level not in chosen
    }
    for label in chosen:
        at = on_curve.index(label)
        before = next((group_of[x] for x in reversed(on_curve[:at]) if x in group_of), None)
        after = next((group_of[x] for x in on_curve[at + 1 :] if x in group_of), None)
        if before is not None and before == after and len(grouping.group_to_originals[before]) > 1:
            raise EditorValueError(_INSIDE_GROUP.format(level=label, group=before, term=term.name))
