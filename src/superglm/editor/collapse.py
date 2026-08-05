"""Categorical level-collapse refits for the editor."""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np

from superglm._frame import as_eager_frame
from superglm.editor._types import EditableTerm
from superglm.features.categorical import Categorical
from superglm.features.grouping import LevelGrouping, collapse_levels
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.spline import Spline

_SYMBOLIC_BASE_POLICIES = {"first", "most_exposed"}


def collapsed_feature_spec(
    model,
    term: EditableTerm,
    selected_indices: np.ndarray,
    *,
    X,
    group_label: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that collapses selected levels."""
    if term.levels is None:
        raise TypeError(f"Term {term.name!r} does not expose categorical levels.")
    if selected_indices.size < 2:
        raise ValueError(f"Select at least two levels to collapse term {term.name!r}.")

    spec = model._specs[term.name]
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise TypeError(
            f"Collapse levels is only available for categorical terms, got {term.name!r}."
        )
    _require_not_interaction_parent(model, term.name, operation="collapse levels")
    frame = as_eager_frame(X)
    frame.require_columns((term.name,))
    values = frame.column_array(term.name)

    idx = np.unique(np.asarray(selected_indices, dtype=np.intp))
    if idx.min() < 0 or idx.max() >= len(term.levels):
        raise IndexError(f"Selection indices out of range for term {term.name!r}.")
    selected_levels = [str(term.levels[i]) for i in idx]

    existing = getattr(spec, "_grouping", None)
    if isinstance(spec, OrderedCategorical):
        selected_originals = _selected_original_members(
            selected_levels,
            existing,
            _displayed_members(term, existing),
        )
        _require_no_special_members(spec, term.name, selected_originals)
        if not _members_are_contiguous(
            selected_originals,
            _original_level_order(spec, term, existing),
        ):
            raise ValueError(
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
    base = _collapsed_base(spec.base, selected_levels, label, existing, grouping)

    if isinstance(spec, OrderedCategorical):
        replacement = _ordered_spec_with_grouping(
            spec,
            grouping,
            selected_levels,
            base,
            values,
        )
    else:
        replacement = Categorical(
            base=base,
            grouping=grouping,
        )

    metadata = {
        "format": "superglm.editor.level_collapse.v1",
        "term": term.name,
        "group_label": label,
        "levels": selected_levels,
        "message": "Selected categorical levels were collapsed and the full model was refit.",
    }
    return replacement, metadata


def ungrouped_feature_spec(
    model,
    term: EditableTerm,
    selected_indices: np.ndarray,
    *,
    X,
) -> tuple[Any, dict[str, Any]]:
    """Return a replacement feature spec that removes selected levels from groups."""
    if term.levels is None:
        raise TypeError(f"Term {term.name!r} does not expose categorical levels.")

    spec = model._specs[term.name]
    if not isinstance(spec, Categorical | OrderedCategorical):
        raise TypeError(
            f"Ungroup levels is only available for categorical terms, got {term.name!r}."
        )
    _require_not_interaction_parent(model, term.name, operation="ungroup levels")
    frame = as_eager_frame(X)
    frame.require_columns((term.name,))
    values = frame.column_array(term.name)
    existing = getattr(spec, "_grouping", None)
    if existing is None:
        raise ValueError(f"Term {term.name!r} does not have collapsed levels.")

    idx = np.unique(np.asarray(selected_indices, dtype=np.intp))
    if idx.size == 0:
        raise ValueError(f"Select at least one grouped level to ungroup term {term.name!r}.")
    if idx.min() < 0 or idx.max() >= len(term.levels):
        raise IndexError(f"Selection indices out of range for term {term.name!r}.")
    selected_levels = [str(term.levels[i]) for i in idx]
    grouping = _ungroup_grouping(
        spec,
        term,
        values,
        selected_levels=selected_levels,
    )
    replacement_grouping = None if _is_identity_grouping(grouping) else grouping

    base = _valid_base_after_ungroup(spec.base, selected_levels, grouping)
    if isinstance(spec, OrderedCategorical):
        replacement = _ordered_spec_with_grouping(
            spec,
            replacement_grouping,
            selected_levels,
            base,
            values,
        )
    else:
        replacement = Categorical(base=base, grouping=replacement_grouping)

    metadata = {
        "format": "superglm.editor.level_ungroup.v1",
        "term": term.name,
        "levels": selected_levels,
        "message": "Selected categorical levels were removed from collapsed groups and the full model was refit.",
    }
    return replacement, metadata


def clone_with_replaced_feature(model, term: str, replacement, *, lambda1=..., lambda2=...):
    """Clone a model and replace one feature spec before fitting."""
    new_model = model._clone_without_features(set(), lambda1=lambda1, lambda2=lambda2)
    new_model._specs[term] = replacement
    new_model._config = new_model._config.with_value(
        feature_templates=tuple((name, new_model._specs[name]) for name in new_model._feature_order)
    )
    new_model._config_revision += 1
    return new_model


def _require_not_interaction_parent(model, term: str, *, operation: str) -> None:
    interactions: list[str] = []
    for name, spec in getattr(model, "_interaction_specs", {}).items():
        parent_names = getattr(spec, "parent_names", ())
        if term in parent_names:
            interactions.append(str(name))
    if interactions:
        joined = ", ".join(interactions)
        raise ValueError(
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
                raise ValueError(
                    "Ungrouping selected levels would leave a non-contiguous ordered group."
                )
            groups[new_label] = remaining

    if not any(
        len(existing.group_to_originals.get(existing.original_to_group.get(level, level), [])) > 1
        for level in selected
    ):
        raise ValueError("Selected levels are not part of a collapsed group.")

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


def _ordered_spec_with_grouping(
    spec: OrderedCategorical,
    grouping: LevelGrouping | None,
    selected_levels: list[str],
    base: str,
    data,
) -> OrderedCategorical:
    values, native_base = _ordered_original_values(spec, grouping, data, base)
    specials = list(spec._specials)
    if spec.basis == "spline":
        basis = (
            copy.deepcopy(spec._spline_obj)
            if spec._spline_obj is not None
            else Spline(
                kind=spec.kind,
                n_knots=spec.n_knots,
                degree=spec.degree,
                select=spec.select,
                penalty=spec.penalty,
            )
        )
        return OrderedCategorical(
            values=values,
            basis=basis,
            base=native_base,
            grouping=grouping,
            specials=specials or None,
        )

    # The user-facing constructor already warned when this legacy step spec was
    # created. Do not repeat that warning from an internal editor compatibility clone.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"OrderedCategorical step smoothing .* is deprecated",
            category=FutureWarning,
        )
        return OrderedCategorical(
            values=values,
            basis="step",
            base=native_base,
            grouping=grouping,
        )


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


def _collapsed_base(
    base: str,
    selected_levels: list[str],
    group_label: str,
    existing_grouping: LevelGrouping | None,
    grouping: LevelGrouping,
) -> str:
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

    counts = {candidate: candidates.count(candidate) for candidate in dict.fromkeys(candidates)}
    return max(counts, key=counts.__getitem__)


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


def _is_identity_grouping(grouping: LevelGrouping) -> bool:
    return not any(
        len([str(member) for member in grouping.group_to_originals.get(label, [])]) > 1
        for label in grouping.grouped_levels
    )


def _require_contiguous(indices: np.ndarray, term_name: str) -> None:
    if indices.size and np.any(np.diff(np.sort(indices)) != 1):
        raise ValueError(f"Ordered categorical collapse for {term_name!r} must be contiguous.")


def _members_are_contiguous(members: list[str], order: list[str]) -> bool:
    if len(members) < 2:
        return True
    positions = sorted(order.index(member) for member in members)
    return bool(np.all(np.diff(positions) == 1))


def _require_no_special_members(
    spec: OrderedCategorical, term_name: str, members: list[str]
) -> None:
    """Refuse a collapse selection that contains a free (special) level."""
    specials = {str(level) for level in spec._specials}
    if not specials:
        return
    selected = [member for member in members if member in specials]
    if not selected:
        return
    joined = ", ".join(repr(member) for member in selected)
    raise ValueError(
        f"Ordered categorical collapse for {term_name!r} cannot include free level(s) "
        f"{joined}: specials are fitted outside the smooth and cannot be grouped."
    )


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
