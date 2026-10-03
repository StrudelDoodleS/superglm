"""Categorical level-collapse refits for the editor."""

from __future__ import annotations

import copy
import re
import warnings
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd

from superglm._frame import as_eager_frame
from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorIndexError, EditorTypeError, EditorValueError
from superglm.features.categorical import Categorical
from superglm.features.grouping import LevelGrouping, collapse_levels
from superglm.features.ordered_categorical import (
    _CLAMP_WARNING_PREFIX,
    OrderedCategorical,
)
from superglm.features.piecewise import Piecewise

_SYMBOLIC_BASE_POLICIES = {"first", "most_exposed"}
# Marks a spec whose reference a collapse or ungroup held in place. The state
# payload reads it to label the reference "kept".
KEPT_REFERENCE_ATTRIBUTE = "_editor_kept_reference"
# The original levels a kept reference stands for. A group that took the
# reference in becomes the reference, yet the level it took in is what was
# kept: when the group later breaks up into a tie, that level settles it.
_KEPT_LEVELS_ATTRIBUTE = "_editor_kept_reference_levels"


def collapsed_feature_spec(
    model,
    term: EditableTerm,
    selected_indices: np.ndarray,
    *,
    X,
    group_label: str | None = None,
    draft_spec=None,
    keep_reference: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that collapses selected levels.

    ``keep_reference`` holds the reference the in-force fit resolved, or the
    new group when it takes that level in. ``False`` hands the declared base
    to the refit, where a symbolic policy (``most_exposed``, ``first``)
    resolves again and can move the reference.

    ``draft_spec`` is the term's spec as waiting changes leave it (None: the
    fitted spec); the collapse is built on it, so changes to one term compose.
    """
    if term.levels is None:
        raise EditorTypeError(f"Term {term.name!r} does not expose categorical levels.")
    if selected_indices.size < 2:
        raise EditorValueError(f"Select at least two levels to collapse term {term.name!r}.")

    fitted = model._specs[term.name]
    spec = fitted if draft_spec is None else draft_spec
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Collapse levels is only available for categorical terms, got {term.name!r}."
        )
    _require_not_interaction_parent(model, term.name, operation="collapse levels")
    frame = as_eager_frame(X)
    frame.require_columns((term.name,))
    values = frame.column_array(term.name)

    idx = np.unique(np.asarray(selected_indices, dtype=np.intp))
    if idx.min() < 0 or idx.max() >= len(term.levels):
        raise EditorIndexError(f"Selection indices out of range for term {term.name!r}.")
    selected_levels = [str(term.levels[i]) for i in idx]

    existing = getattr(spec, "_grouping", None)
    if isinstance(spec, OrderedCategorical):
        selected_originals = _selected_original_members(
            selected_levels,
            existing,
            _displayed_members(term, existing),
        )
        _require_no_special_members(spec, term.name, selected_originals)
        _require_no_break_members(spec, term.name, selected_originals)
        if not _members_are_contiguous(
            selected_originals,
            _original_level_order(spec, term, existing),
        ):
            raise EditorValueError(
                f"Ordered categorical collapse for {term.name!r} must be contiguous "
                "in fitted order."
            )

    existing_labels = [str(level) for level in term.levels]
    if existing is not None:
        existing_labels.extend(str(level) for level in existing.grouped_levels)
    label = _unique_group_label(
        _default_group_label(selected_levels),
        existing_levels=existing_labels,
        selected_levels=selected_levels,
    )
    if group_label is not None:
        label = _unique_group_label(
            str(group_label),
            existing_levels=existing_labels,
            selected_levels=selected_levels,
        )
    grouping = _collapse_grouping(
        spec,
        term,
        values,
        selected_levels=selected_levels,
        group_label=label,
    )
    declared, kept = _reference_to_keep(fitted, spec, term) if keep_reference else (spec.base, [])
    base = _collapsed_base(declared, kept, selected_levels, label, existing, grouping)

    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(spec, grouping=grouping, base=base, data=values)
    else:
        replacement = _rebuilt_categorical(spec, fitted, base=base, grouping=grouping, data=values)
    if keep_reference:
        _mark_kept(replacement, base, kept, grouping)

    metadata = {
        "format": "superglm.editor.level_collapse.v1",
        "term": term.name,
        "group_label": label,
        "levels": selected_levels,
        "label": f"collapse {' + '.join(selected_levels)} in {term.name}",
        "message": "Selected categorical levels were collapsed and the full model was refit.",
    }
    return replacement, metadata


def ungrouped_feature_spec(
    model,
    term: EditableTerm,
    selected_indices: np.ndarray,
    *,
    X,
    draft_spec=None,
    keep_reference: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that removes selected levels from groups.

    ``keep_reference`` holds the in-force reference: a reference group that
    loses members follows the members that stay. ``False`` hands the declared
    base on, as ``collapsed_feature_spec`` does.

    ``draft_spec`` is the term's spec as waiting changes leave it (None: the
    fitted spec); the ungroup is built on it, so changes to one term compose.
    """
    if term.levels is None:
        raise EditorTypeError(f"Term {term.name!r} does not expose categorical levels.")

    fitted = model._specs[term.name]
    spec = fitted if draft_spec is None else draft_spec
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Ungroup levels is only available for categorical terms, got {term.name!r}."
        )
    _require_not_interaction_parent(model, term.name, operation="ungroup levels")
    frame = as_eager_frame(X)
    frame.require_columns((term.name,))
    values = frame.column_array(term.name)
    existing = getattr(spec, "_grouping", None)
    if existing is None:
        raise EditorValueError(f"Term {term.name!r} does not have collapsed levels.")

    idx = np.unique(np.asarray(selected_indices, dtype=np.intp))
    if idx.size == 0:
        raise EditorValueError(f"Select at least one grouped level to ungroup term {term.name!r}.")
    if idx.min() < 0 or idx.max() >= len(term.levels):
        raise EditorIndexError(f"Selection indices out of range for term {term.name!r}.")
    selected_levels = [str(term.levels[i]) for i in idx]
    grouping = _ungroup_grouping(
        spec,
        term,
        values,
        selected_levels=selected_levels,
    )
    replacement_grouping = None if _is_identity_grouping(grouping) else grouping

    if keep_reference:
        declared, kept = _reference_to_keep(fitted, spec, term)
        base = _kept_base_after_ungroup(declared, kept, selected_levels, existing, grouping)
    else:
        base = _valid_base_after_ungroup(spec.base, selected_levels, grouping)
    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(
            spec, grouping=replacement_grouping, base=base, data=values
        )
    else:
        # Without a grouping the fit reads native values (3, not "3").
        replacement = _rebuilt_categorical(
            spec, fitted, base=base, grouping=replacement_grouping, data=values
        )
    if keep_reference:
        _mark_kept(replacement, base, kept, replacement_grouping)

    metadata = {
        "format": "superglm.editor.level_ungroup.v1",
        "term": term.name,
        "levels": selected_levels,
        "label": ungroup_label(term.name, selected_levels),
        "message": "Selected categorical levels were removed from collapsed groups and the full model was refit.",
    }
    return replacement, metadata


def ungroup_label(term_name: str, levels: list[str]) -> str:
    return f"ungroup {', '.join(levels)} in {term_name}"


def reference_feature_spec(
    model, term: EditableTerm, level: str, *, X, draft_spec=None
) -> tuple[Any, dict[str, Any]]:
    """Return a fresh replacement spec whose reference is the displayed ``level``.

    ``draft_spec`` is the term's spec as waiting changes leave it (None: the
    fitted spec), so a reference can name a group a waiting collapse made.
    """
    fitted = model._specs[term.name]
    spec = fitted if draft_spec is None else draft_spec
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise EditorTypeError(
            f"Set reference is only available for categorical terms, got {term.name!r}."
        )
    _require_not_interaction_parent(model, term.name, operation="set the reference level")
    grouping = getattr(spec, "_grouping", None)
    label = _fitted_level_label(spec, grouping, term, level)
    frame = as_eager_frame(X)
    frame.require_columns((term.name,))
    values = frame.column_array(term.name)
    if isinstance(spec, OrderedCategorical):
        replacement = rebuilt_ordered_spec(spec, grouping=grouping, base=label, data=values)
    else:
        # Fitted levels keep their native type (an integer level stays 3, not "3").
        replacement = _rebuilt_categorical(spec, fitted, base=label, grouping=grouping, data=values)
    metadata = {
        "format": "superglm.editor.reference_level.v1",
        "term": term.name,
        "level": label,
        "label": f"set reference of {term.name} to {label}",
        "message": (
            f"The reference level of {term.name} was set to {label} and the full model was refit."
        ),
    }
    return replacement, metadata


def _fitted_level_label(spec, grouping, term: EditableTerm, level: str) -> str:
    """The fitted level that carries the reference for a displayed ``level``."""
    if isinstance(spec, OrderedCategorical) and level in special_labels(spec):
        raise EditorValueError(
            f"A special level can't be the reference of {term.name!r}; choose an ordered level."
        )
    # A group in the Collapsed display, or a selection of all its members, is
    # sent by its own label, which the fitted spec already carries as a level.
    if grouping is not None and level in grouping.group_to_originals:
        return level
    if level not in term.levels:
        raise EditorValueError(f"{level!r} is not a level of term {term.name!r}.")
    return level if grouping is None else str(grouping.original_to_group.get(level, level))


def clone_with_replaced_features(model, replacements: dict[str, Any], *, lambda1=..., lambda2=...):
    """Clone a model and replace feature specs before fitting.

    Each replacement is deep-copied in, so fitting the clone never touches the
    caller's spec: a waiting step's draft stays unfitted, and a refit that is
    undone and run again fits a fresh copy of the same draft.
    """
    new_model = model._clone_without_features(set(), lambda1=lambda1, lambda2=lambda2)
    for term, replacement in replacements.items():
        new_model._specs[term] = copy.deepcopy(replacement)
    new_model._config = new_model._config.with_value(
        feature_templates=tuple((name, new_model._specs[name]) for name in new_model._feature_order)
    )
    new_model._config_revision += 1
    return new_model


def clone_with_replaced_feature(model, term: str, replacement, *, lambda1=..., lambda2=...):
    """Clone a model and replace one feature spec before fitting."""
    return clone_with_replaced_features(
        model, {term: replacement}, lambda1=lambda1, lambda2=lambda2
    )


def interaction_users(model, term: str) -> list[str]:
    """The interactions that use ``term`` as a parent."""
    return [
        str(name)
        for name, spec in getattr(model, "_interaction_specs", {}).items()
        if term in getattr(spec, "parent_names", ())
    ]


def _require_not_interaction_parent(model, term: str, *, operation: str) -> None:
    interactions = interaction_users(model, term)
    if interactions:
        joined = ", ".join(interactions)
        raise EditorValueError(
            f"Cannot {operation} for term {term!r} because it is used by interaction(s): "
            f"{joined}. Refit a model without those interactions first."
        )


def _collapse_grouping(
    spec,
    term: EditableTerm,
    data,
    *,
    selected_levels: list[str],
    group_label: str,
) -> LevelGrouping:
    existing = getattr(spec, "_grouping", None)
    displayed_members = _displayed_members(term, existing)
    selected_set = set(selected_levels)
    selected_originals = _selected_original_members(selected_levels, existing, displayed_members)
    selected_original_set = set(selected_originals)
    groups: dict[str, list[str]] = {group_label: selected_originals}
    if existing is not None:
        for label in existing.grouped_levels:
            members = [str(member) for member in existing.group_to_originals.get(label, [])]
            if len(members) < 2:
                continue
            remaining = [member for member in members if member not in selected_original_set]
            if len(remaining) < 2:
                continue
            if remaining == members:
                groups[str(label)] = members
            else:
                groups[
                    _unique_group_label(
                        _default_group_label(remaining),
                        existing_levels=_existing_labels_for_grouping(term, existing),
                        selected_levels=remaining,
                    )
                ] = remaining
    else:
        for level, members in displayed_members.items():
            if level in selected_set:
                continue
            if len(members) > 1 or members[0] != level:
                groups[level] = list(members)

    return collapse_levels(
        data,
        groups=groups,
        order=_original_level_order(spec, term, existing),
    )


def _selected_original_members(
    selected_levels: list[str],
    grouping,
    displayed_members: dict[str, list[str]],
) -> list[str]:
    if grouping is None:
        candidates = [
            member for level in selected_levels for member in displayed_members.get(level, [level])
        ]
    else:
        originals = {str(level) for level in grouping.all_original_levels}
        candidates = []
        for level in selected_levels:
            if level in originals:
                candidates.append(level)
            else:
                candidates.extend(
                    str(member) for member in grouping.group_to_originals.get(level, [level])
                )
    return list(dict.fromkeys(candidates))


def _existing_labels_for_grouping(term: EditableTerm, grouping) -> list[str]:
    labels = [str(level) for level in term.levels or []]
    if grouping is not None:
        labels.extend(str(level) for level in grouping.grouped_levels)
    return labels


def _ungroup_grouping(
    spec,
    term: EditableTerm,
    data,
    *,
    selected_levels: list[str],
) -> LevelGrouping:
    existing = getattr(spec, "_grouping", None)
    selected = set(selected_levels)
    groups: dict[str, list[str]] = {}
    existing_labels = [str(level) for level in term.levels]
    existing_labels.extend(str(level) for level in existing.grouped_levels)
    original_order = _original_level_order(spec, term, existing)
    for label in existing.grouped_levels:
        members = [str(member) for member in existing.group_to_originals.get(label, [])]
        if len(members) < 2:
            continue
        remaining = [member for member in members if member not in selected]
        if len(remaining) < 2:
            continue
        if remaining == members:
            groups[str(label)] = members
        else:
            new_label = _unique_group_label(
                _default_group_label(remaining),
                existing_levels=existing_labels,
                selected_levels=remaining,
            )
            if isinstance(spec, OrderedCategorical) and not _members_are_contiguous(
                remaining, original_order
            ):
                raise EditorValueError(
                    "Ungrouping selected levels would leave a non-contiguous ordered group."
                )
            groups[new_label] = remaining

    if not any(
        len(existing.group_to_originals.get(existing.original_to_group.get(level, level), [])) > 1
        for level in selected
    ):
        raise EditorValueError("Selected levels are not part of a collapsed group.")

    return collapse_levels(
        data,
        groups=groups,
        order=original_order,
    )


def _displayed_members(term: EditableTerm, grouping) -> dict[str, list[str]]:
    if grouping is None:
        return {str(level): [str(level)] for level in term.levels or []}
    return {
        str(level): [
            str(member)
            for member in grouping.group_to_originals.get(
                grouping.original_to_group.get(str(level), str(level)),
                [level],
            )
        ]
        for level in term.levels or []
    }


def _original_level_order(spec, term: EditableTerm, grouping) -> list[str]:
    if grouping is not None:
        return [str(level) for level in grouping.all_original_levels]
    if isinstance(spec, OrderedCategorical):
        original_values = getattr(spec, "_original_level_to_value", None)
        if original_values is not None:
            return [str(level) for level in original_values]
        return [str(level) for level in getattr(spec, "_ordered_levels", term.levels or [])]
    return [str(level) for level in term.levels or []]


def rebuilt_ordered_spec(
    spec: OrderedCategorical,
    *,
    grouping: LevelGrouping | None,
    base: Any,
    data,
    basis=None,
) -> OrderedCategorical:
    """A fresh, unfitted OrderedCategorical like ``spec`` with this grouping and base.

    ``basis`` replaces the inner basis (a shaped range). By default the pristine
    declared basis is cloned. A fitted spec is never mutated: its resolved base
    is sticky and would silently survive a changed ``base``.
    """
    values, native_base = _ordered_original_values(spec, grouping, data, base)
    # Clone the RAW declarations, not the string-coerced ``_specials``. A special
    # declared as ``9`` on a float column matches through its raw label -- the
    # string view renders 9.0 as "9.0", which never equals "9" -- so rebuilding
    # from the coerced form silently drops that fallback and the special's
    # indicator comes back all-zero on a refit.
    specials = list(spec._special_raw) or list(spec._specials)
    source = _pristine_basis(spec) if basis is None else basis
    # Collapsing levels shrinks the level count, so the pristine spline's
    # ``n_knots`` routinely exceeds the new ``n_levels - 1`` and construction
    # clamps it. That clamp is the caller's own basis being re-fitted to the
    # levels the caller just asked to merge, not a configuration mistake, and
    # the user-facing construction already warned if the original declaration
    # over-specified. Do not repeat it from an internal editor clone.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=re.escape(_CLAMP_WARNING_PREFIX),
            category=UserWarning,
        )
        return OrderedCategorical(
            values=values,
            basis=source,
            base=native_base,
            grouping=grouping,
            specials=specials or None,
        )


def _pristine_basis(spec: OrderedCategorical):
    # Clone from the pristine caller-supplied spline, not the clamped inner
    # copy: the new spec re-clamps against ITS OWN level count, which can
    # exceed the current one when a grouping is being undone. `_spline_obj` is
    # always set on a spec this version constructed; the fallback covers a
    # pre-0.24 pickle, whose `_basis_spline` read refuses a step-mode spec
    # loudly instead of silently cloning it onto the default P-spline.
    #
    # Read it with `getattr`, not `spec._spline_obj`: an attribute-less read
    # only reaches the fallback when the key EXISTS and is None, so a pickle
    # old enough to predate the attribute raised a bare AttributeError naming
    # `_spline_obj` -- still loud, but without the migration sentence, and
    # never reaching `_basis_spline` where that sentence lives.
    spline_obj = getattr(spec, "_spline_obj", None)
    if spline_obj is not None:
        source = copy.deepcopy(spline_obj)
    else:
        # A shortcut-era pickle has no pristine declaration, only the inner
        # spline -- and its `n_knots` was already clamped to the level count it
        # was BUILT against. Cloning that alone keeps the reduced basis, which
        # is wrong in exactly the direction ungrouping goes: back to MORE
        # levels. The removed shortcut path rebuilt from the then-plain
        # `n_knots` attribute, i.e. the count the caller REQUESTED, and that
        # entry survives in the pickled __dict__ because the class property
        # only shadows it. Recover it and let __init__ re-clamp against the new
        # level count, so the clamp is still enforced -- just against the right
        # number of levels.
        source = copy.deepcopy(spec._basis_spline)
        requested = spec.__dict__.get("n_knots")
        if isinstance(requested, int | np.integer) and int(requested) > source.n_knots:
            source.n_knots = int(requested)
    return source


def _ordered_original_values(
    spec: OrderedCategorical,
    grouping: LevelGrouping | None,
    data,
    base,
) -> tuple[dict[Any, float], Any]:
    original_values = getattr(spec, "_original_level_to_value", None)
    if original_values is not None:
        values = {str(k): float(v) for k, v in original_values.items()}
    else:
        values = {str(k): float(v) for k, v in spec._level_to_value.items()}
    if grouping is not None:
        return values, base

    native_by_label: dict[str, Any] = {}
    for raw in np.asarray(data, dtype=object).ravel():
        native_by_label.setdefault(str(raw), raw)
    native_values = {native_by_label.get(label, label): value for label, value in values.items()}
    native_base = base if base in _SYMBOLIC_BASE_POLICIES else native_by_label.get(str(base), base)
    return native_values, native_base


def _in_force_reference(spec) -> Any:
    """The reference ``spec``'s fit resolved, in its native type (3, not "3").

    Before a fit there is none, and the declared base stands in.
    """
    level = getattr(spec, "_base_level", "")
    return spec.base if level == "" else level


def _collapsed_base(
    base: Any,
    kept: list[str],
    selected_levels: list[str],
    group_label: str,
    existing_grouping: LevelGrouping | None,
    grouping: LevelGrouping,
) -> str:
    """The level holding ``base`` once the collapse into ``grouping`` is made.

    A reference group the collapse splits follows the new level holding most
    of its members; a tie goes to the level holding a ``kept`` one, the
    original levels a kept reference stands for (``_reference_to_keep``).
    """
    base = str(base)
    if base in _SYMBOLIC_BASE_POLICIES:
        return base

    valid = {str(level) for level in grouping.grouped_levels}
    if base in valid:
        return base
    if base in set(selected_levels) and group_label in valid:
        return group_label

    base_originals = _base_original_members(base, existing_grouping)
    if not base_originals:
        return group_label if group_label in valid else base

    mapped = [str(grouping.original_to_group.get(member, member)) for member in base_originals]
    candidates = [candidate for candidate in mapped if candidate in valid]
    if not candidates:
        return group_label if group_label in valid else base

    holding = _holding(kept, grouping)
    counts = {candidate: candidates.count(candidate) for candidate in dict.fromkeys(candidates)}
    return max(counts, key=lambda candidate: (counts[candidate], candidate in holding))


def _base_original_members(base: str, grouping: LevelGrouping | None) -> list[str]:
    if grouping is None:
        return [base]
    if base in grouping.group_to_originals:
        return [str(member) for member in grouping.group_to_originals[base]]
    if base in grouping.all_original_levels:
        return [base]
    return []


def _valid_base_after_ungroup(
    base: str, selected_levels: list[str], grouping: LevelGrouping
) -> str:
    base = str(base)
    if base in _SYMBOLIC_BASE_POLICIES:
        return base
    valid = set(grouping.grouped_levels) | set(grouping.all_original_levels)
    if base in valid:
        return base
    return selected_levels[0] if selected_levels else "most_exposed"


def _kept_base_after_ungroup(
    base: Any,
    kept: list[str],
    selected_levels: list[str],
    existing: LevelGrouping,
    grouping: LevelGrouping,
) -> str:
    """The level holding the in-force reference once ``selected_levels`` leave their groups.

    A reference group that loses members follows the new level holding most of
    them (the collapse rule in ``_collapsed_base``); a tie goes to a level that
    was not pulled out, then to the level holding a ``kept`` one, so a group
    ungrouped whole gives back the level it took in. A reference the old
    grouping does not know keeps the declared-base rule.
    """
    base = str(base)
    if base in _SYMBOLIC_BASE_POLICIES or base in grouping.grouped_levels:
        return base
    members = _base_original_members(base, existing)
    if not members:
        return _valid_base_after_ungroup(base, selected_levels, grouping)
    mapped = [str(grouping.original_to_group.get(member, member)) for member in members]
    pulled = set(selected_levels)
    holding = _holding(kept, grouping)
    counts = {label: mapped.count(label) for label in dict.fromkeys(mapped)}
    return max(counts, key=lambda label: (counts[label], label not in pulled, label in holding))


def _reference_to_keep(fitted, spec, term: EditableTerm) -> tuple[Any, list[str]]:
    """The reference a keep-reference step keeps, named in ``spec``'s own levels.

    ``spec`` is the term's draft, or ``fitted`` when nothing waits. A draft that
    already names a level or group (a reference an earlier waiting step kept
    or set) keeps it. Otherwise the fitted reference is kept while the draft
    still has that level or group; a draft regrouped by a step staged with
    keep-reference off keeps its own policy.

    Also returns the original levels the reference stands for: those an
    earlier keep-reference step recorded, else all of its members.
    """
    in_force = _in_force_reference(fitted)
    if spec is fitted:
        return in_force, _kept_levels(fitted, in_force)
    if str(spec.base) not in _SYMBOLIC_BASE_POLICIES:
        return spec.base, _kept_levels(spec, spec.base)
    grouping = getattr(spec, "_grouping", None)
    names = term.levels if grouping is None else grouping.grouped_levels
    held = {str(name) for name in names or []}
    if str(in_force) in held:
        return in_force, _kept_levels(fitted, in_force)
    return spec.base, []


def _kept_levels(spec, base) -> list[str]:
    """The original levels ``spec``'s reference ``base`` stands for.

    A keep-reference step records them (``_mark_kept``); a record that no
    longer lies inside ``base`` is stale, and every member stands for it.
    """
    members = _base_original_members(str(base), getattr(spec, "_grouping", None))
    recorded = [level for level in getattr(spec, _KEPT_LEVELS_ATTRIBUTE, ()) if level in members]
    return recorded or members


def _mark_kept(replacement, base, kept: list[str], grouping: LevelGrouping | None) -> None:
    """Mark ``replacement``'s reference as kept and record the levels it stands for."""
    setattr(replacement, KEPT_REFERENCE_ATTRIBUTE, True)
    members = set(_base_original_members(str(base), grouping))
    setattr(replacement, _KEPT_LEVELS_ATTRIBUTE, tuple(level for level in kept if level in members))


def _holding(kept: list[str], grouping: LevelGrouping) -> set[str]:
    """The levels of ``grouping`` that hold a ``kept`` original level."""
    return {str(grouping.original_to_group.get(level, level)) for level in kept}


def _native_levels(spec: Categorical, fitted: Categorical, data) -> dict[str, Any]:
    """Each level's native value by its text: declared, then fitted, then seen in ``data``.

    A draft has no fitted ``_levels``, so the in-force spec and the column stand
    in for it. A grouped spec's fitted levels are group labels, not raw ones.
    """
    fitted_levels = fitted._levels if getattr(fitted, "_grouping", None) is None else []
    observed = pd.unique(np.asarray(data).ravel()).tolist()
    native: dict[str, Any] = {}
    for level in chain(spec._declared_levels or [], fitted_levels, observed):
        native.setdefault(str(level), level)
    return native


def _rebuilt_categorical(
    spec: Categorical,
    fitted: Categorical,
    *,
    base,
    grouping,
    data,
    unseen: str | None = None,
    levels: list | None = None,
) -> Categorical:
    """A fresh Categorical like ``spec`` with this grouping and base.

    It keeps ``levels=`` and ``unseen=``, which a collapse or ungroup used to
    drop; ``unseen`` replaces the policy and ``levels`` the declared universe.
    Grouped, the design speaks the grouping's text labels; ungrouped, the base
    goes back to its native value, so an integer level stays 3, not "3".
    """
    if grouping is None and str(base) not in _SYMBOLIC_BASE_POLICIES:
        base = _native_levels(spec, fitted, data).get(str(base), base)
    return Categorical(
        base=base,
        grouping=grouping,
        levels=spec._declared_levels if levels is None else levels,
        unseen=spec.unseen if unseen is None else unseen,
    )


def _is_identity_grouping(grouping: LevelGrouping) -> bool:
    return not any(
        len([str(member) for member in grouping.group_to_originals.get(label, [])]) > 1
        for label in grouping.grouped_levels
    )


def _require_contiguous(indices: np.ndarray, term_name: str) -> None:
    if indices.size and np.any(np.diff(np.sort(indices)) != 1):
        raise EditorValueError(
            f"Ordered categorical collapse for {term_name!r} must be contiguous."
        )


def _members_are_contiguous(members: list[str], order: list[str]) -> bool:
    if len(members) < 2:
        return True
    positions = sorted(order.index(member) for member in members)
    return bool(np.all(np.diff(positions) == 1))


def _require_no_special_members(
    spec: OrderedCategorical, term_name: str, members: list[str]
) -> None:
    """Refuse a collapse selection that contains a free (special) level."""
    specials = special_labels(spec)
    if not specials:
        return
    selected = [member for member in members if member in specials]
    if not selected:
        return
    joined = ", ".join(repr(member) for member in selected)
    raise EditorValueError(
        f"Ordered categorical collapse for {term_name!r} cannot include free level(s) "
        f"{joined}: specials are fitted outside the smooth and cannot be grouped."
    )


def _require_no_break_members(spec: OrderedCategorical, term_name: str, members: list[str]) -> None:
    """Refuse, in words, a group that takes in a stated break band.

    The library refuses a grouping that absorbs or straddles a stated break,
    and an ordered group is contiguous, so straddling one means taking it in.
    """
    absorbed = [band for band in _stated_break_bands(spec) if band in members]
    if absorbed:
        raise EditorValueError(
            f"{term_name!r} has a break or a shaped-range edge at {absorbed[0]!r}, so that "
            "band can't be collapsed; move or remove it first."
        )


def _stated_break_bands(spec: OrderedCategorical) -> list[str]:
    """The bands where ``spec`` states a Piecewise break, a named knot or a range edge."""
    basis = getattr(spec, "_spline_obj", None)
    if isinstance(basis, Piecewise):
        # Int-mode breaks are placed from the data: nothing is stated.
        stated = basis.breaks if isinstance(basis.breaks, list) else []
    else:
        # A numeric knot states a coordinate, not a band; the library guards names only.
        named = getattr(basis, "_named_knots", None) or []
        ranges = getattr(basis, "polynomial_ranges", ())
        edges = chain.from_iterable((r.lo, r.hi) for r in ranges)
        stated = [entry for entry in chain(named, edges) if isinstance(entry, str)]
    declared = [str(level) for level in spec._declared_smooth_levels]
    return [entry if isinstance(entry, str) else declared[int(entry)] for entry in stated]


def special_labels(spec: OrderedCategorical) -> set[str]:
    """The free (special) levels of ``spec``, in every spelling the editor displays."""
    # Both namespaces: displayed levels arrive in the DISPLAY spelling, so
    # matching only the str-coerced `_specials` leaves a guard INERT on a float
    # domain ("9" vs "9.0") -- and a guard that fails open here silently smooths
    # a level that `specials=` still reports as free.
    return {str(level) for level in spec._specials} | {
        str(level) for level in spec._special_display
    }


def _default_group_label(selected_levels: list[str]) -> str:
    return "+".join(str(level) for level in selected_levels)


def _unique_group_label(
    candidate: str,
    *,
    existing_levels: list[str],
    selected_levels: list[str],
) -> str:
    blocked = set(existing_levels) - set(selected_levels)
    if candidate not in blocked:
        return candidate
    i = 2
    while f"{candidate} ({i})" in blocked:
        i += 1
    return f"{candidate} ({i})"
