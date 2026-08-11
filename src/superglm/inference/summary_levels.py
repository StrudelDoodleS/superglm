"""Summary-only presentation helpers for categorical levels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from superglm.inference.summary import _CoefRow
from superglm.types import GroupSlice

LevelDisplay = Literal["expanded", "grouped"]
_VALID_LEVEL_DISPLAYS = frozenset({"expanded", "grouped"})


@dataclass(frozen=True)
class LevelGroupLegend:
    """Exact original members represented by one fitted categorical group."""

    feature: str
    group_id: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class SummaryLevelDisplay:
    """Rows and legends for one requested summary presentation."""

    level_display: LevelDisplay
    rows: tuple[_CoefRow, ...]
    level_groups: tuple[LevelGroupLegend, ...]

    @property
    def has_level_groups(self) -> bool:
        return bool(self.level_groups)


def validate_level_display(value: object) -> LevelDisplay:
    """Validate and narrow a categorical summary display mode."""
    if not isinstance(value, str) or value not in _VALID_LEVEL_DISPLAYS:
        raise ValueError(
            f"level_display={value!r} is not valid. "
            f"Expected one of {sorted(_VALID_LEVEL_DISPLAYS)}."
        )
    return cast(LevelDisplay, value)


def build_level_universes(
    specs: Mapping[str, Any],
    interaction_specs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-term level-universe provenance for the summary payload (spec §3.8).

    The audit trail a rating-governance reader needs about a categorical-family
    term: the bound universe, where it came from (``declared`` / ``dtype`` /
    ``full-frame`` / ``inferred``), the reference level and whether an empty
    declared base forced a swap, and any level pinned to base for want of
    training rows.

    Keys mirror ``reconstruct()`` exactly, because both are read as one record
    of the same fitted state and a second spelling is how they drift apart. A
    term with no universe at all -- every numeric spec -- is absent rather than
    present-and-empty, so ``level_universes`` reads as the list of terms that
    have levels.

    ``interaction_specs`` is the model's interaction namespace, read on the same
    terms as the main one. ``FactorSmooth`` is the reason it exists: it carries
    a bound group universe and a ``_level_source``, and it lives ONLY there, so
    scanning ``_specs`` alone drops precisely the term whose universe is least
    obvious from the coefficient table. The two namespaces share one key space
    (``validate_term_name_namespace`` refuses a name in both), so a flat payload
    keyed by term name cannot collide.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.random_effect import RandomEffect

    universes: dict[str, dict[str, Any]] = {}
    scanned = {**dict(specs), **dict(interaction_specs or {})}
    for feature, spec in scanned.items():
        if not isinstance(spec, Categorical | OrderedCategorical | RandomEffect | FactorSmooth):
            continue
        levels = (
            spec._ordered_levels if isinstance(spec, OrderedCategorical) else list(spec._levels)
        )
        if not levels:
            continue
        # OrderedCategorical declares through `order=`/`values=` and pins only
        # specials; RandomEffect and FactorSmooth pool empty levels through
        # their penalty and so never pin. Read both pin lists rather than
        # branching on the type: a term carries at most one of them.
        pinned = list(getattr(spec, "_pinned_levels", ())) + list(
            getattr(spec, "_pinned_specials", ())
        )
        # `""` is the pre-build sentinel `Categorical.__init__` writes, and an
        # absent attribute is a term with no base at all (RandomEffect,
        # FactorSmooth). Those two are the only "no base" cases, so they are the
        # only ones tested for: a falsey test instead would erase the perfectly
        # real base levels `0` and `False` on an integer or boolean universe and
        # report the audit trail as having no reference level.
        base_level = getattr(spec, "_base_level", None)
        if isinstance(base_level, str) and base_level == "":
            base_level = None
        universes[feature] = {
            "levels": list(levels),
            "level_source": getattr(spec, "_level_source", "declared"),
            "base_level": base_level,
            "base_fallback": getattr(spec, "_base_fallback", None),
            "pinned_levels": pinned,
        }
    return universes


def _with_piecewise_reference_rows(
    rows: list[_CoefRow],
    *,
    specs: Mapping[str, Any],
    groups: Sequence[GroupSlice],
) -> list[_CoefRow]:
    """Give a ``Piecewise`` term the base row a ``Categorical`` already displays.

    Every coefficient in a piecewise block is a contrast against the base knot,
    and without a row naming that knot the summary is the only surface that
    never says which one it is: the workbook prints all ``J + 2`` knots and the
    editor shows a handle at each, so the three surfaces the design promises to
    keep identical would differ by exactly one row.  With ``breaks=int`` the
    knot vector is not printed either, so the base cannot even be inferred by
    elimination.

    Display-only, and built here rather than in ``coef_tables`` for the same
    reason the categorical reference row is: it carries no coefficient, so it
    must not reach ``_coef_rows``, which is what the export payload, the edf
    buckets and every consumer that counts fitted parameters read.
    """
    from superglm.features.piecewise import Piecewise

    out = list(rows)
    for feature, spec in specs.items():
        if not isinstance(spec, Piecewise) or spec._knots.size == 0:
            continue
        feature_groups = [group for group in groups if group.feature_name == feature]
        if not feature_groups:
            continue
        term_prefix = feature_groups[0].name
        knot_name = {
            index: f"{term_prefix}[{float(spec._knots[index]):.10g}]"
            for index in range(spec._knots.size)
        }
        base_index = int(spec._base_index)
        if any(row.name == knot_name[base_index] and row.group == term_prefix for row in out):
            continue
        # `_non_base_indices` is ascending and the coefficient rows are emitted in
        # that order, so the k-th matched position is the k-th non-base knot and
        # the base row belongs at position `base_index` among them -- which puts
        # it in knot order however far in or out the base sits.
        non_base_names = {knot_name[int(index)] for index in spec._non_base_indices}
        positions = [
            i
            for i, row in enumerate(out)
            if row.group == term_prefix and row.name in non_base_names
        ]
        if not positions:
            continue
        insert_at = positions[base_index] if base_index < len(positions) else positions[-1] + 1
        # Mirror the whole-term row's active state rather than hardcode True:
        # when selection drops the group, coef_tables emits the term row with
        # active=False, and a display row still claiming active would be the
        # one surface disagreeing about whether the term survived.
        term_row = next(
            (row for row in out if row.name == term_prefix and row.is_spline),
            None,
        )
        out.insert(
            insert_at,
            _CoefRow(
                name=knot_name[base_index],
                group=term_prefix,
                coef=0.0,
                is_reference=True,
                active=bool(term_row.active) if term_row is not None else True,
            ),
        )
    return out


def build_summary_level_display(
    coef_rows: Sequence[_CoefRow],
    *,
    specs: Mapping[str, Any],
    groups: Sequence[GroupSlice],
    level_display: object = "expanded",
) -> SummaryLevelDisplay:
    """Build summary-only rows without mutating canonical coefficient rows."""
    mode = validate_level_display(level_display)
    rows = _with_piecewise_reference_rows(list(coef_rows), specs=specs, groups=groups)
    legends: list[LevelGroupLegend] = []

    from superglm.features.categorical import Categorical
    from superglm.features.ordered_categorical import OrderedCategorical

    for feature, spec in specs.items():
        if not isinstance(spec, Categorical | OrderedCategorical):
            continue

        feature_groups = [group for group in groups if group.feature_name == feature]
        term_prefix = (
            feature_groups[0].name if isinstance(spec, Categorical) and feature_groups else feature
        )
        fitted_levels = [
            str(level)
            for level in (spec._levels if isinstance(spec, Categorical) else spec._ordered_levels)
        ]
        grouping = getattr(spec, "_grouping", None)
        if grouping is None:
            original_levels = fitted_levels
            original_to_group = {level: level for level in original_levels}
            presentation_fitted_levels = fitted_levels
        else:
            original_levels = [str(level) for level in grouping.all_original_levels]
            original_to_group = {
                str(original): str(fitted)
                for original, fitted in grouping.original_to_group.items()
            }
            presentation_fitted_levels = [str(level) for level in grouping.grouped_levels]

        members_by_fitted: dict[str, list[str]] = {}
        for original in original_levels:
            fitted = original_to_group[original]
            members_by_fitted.setdefault(fitted, []).append(original)
        group_ids = {
            fitted: f"G{index}"
            for index, (fitted, members) in enumerate(
                (
                    (fitted, members)
                    for fitted, members in members_by_fitted.items()
                    if len(members) > 1
                ),
                start=1,
            )
        }
        expected_names = {fitted: f"{term_prefix}[{fitted}]" for fitted in fitted_levels}
        row_by_fitted: dict[str, _CoefRow] = {}
        matched_indices: list[int] = []
        for index, row in enumerate(rows):
            for fitted, expected_name in expected_names.items():
                if row.name == expected_name and row.group == term_prefix:
                    row_by_fitted[fitted] = row
                    matched_indices.append(index)
                    break

        base_level = str(spec._base_level)
        # A pinned level is bound and known but has no coefficient, so no
        # canonical row names it. Dropping it here would leave the summary the
        # only surface that never mentions the pin, which is the one thing a
        # reader has to see: `reconstruct()` reports it at relativity 1.0 and
        # the term still predicts it as base.
        pinned_levels = {str(level) for level in getattr(spec, "_pinned_levels", ())}
        reference_only = bool(presentation_fitted_levels) and all(
            fitted == base_level for fitted in presentation_fitted_levels
        )
        # Preserve unknown canonical layouts, but a reference-only feature has
        # no non-reference coefficient to match and must be synthesized here.
        if not matched_indices and not reference_only:
            continue

        legends.extend(
            LevelGroupLegend(feature, group_ids[fitted], tuple(members))
            for fitted, members in members_by_fitted.items()
            if fitted in group_ids
        )
        display_rows: list[_CoefRow] = []
        edf_emitted: set[int] = set()
        diagnostics_emitted: set[int] = set()
        level_items = (
            [(original, original_to_group[original]) for original in original_levels]
            if mode == "expanded"
            else [
                (members_by_fitted[fitted][0], fitted)
                for fitted in presentation_fitted_levels
                if fitted in members_by_fitted
            ]
        )
        for original, fitted in level_items:
            source = row_by_fitted.get(fitted)
            member_count = len(members_by_fitted[fitted])
            row_name = (
                term_prefix
                if mode == "grouped" and member_count > 1
                else f"{term_prefix}[{original}]"
            )
            if fitted == base_level:
                if source is None:
                    source = _CoefRow(
                        name=f"{term_prefix}[{fitted}]",
                        group=term_prefix,
                        coef=0.0,
                        active=True,
                    )
                display_row = replace(
                    source,
                    name=row_name,
                    level_group=group_ids.get(fitted, ""),
                    is_reference=True,
                    active=True,
                    coef=0.0,
                    se=None,
                    z=None,
                    p=None,
                    ci_low=None,
                    ci_high=None,
                    edf=None,
                )
            elif source is not None:
                display_row = replace(
                    source,
                    name=row_name,
                    level_group=group_ids.get(fitted, ""),
                    is_reference=False,
                    edf=source.edf if id(source) not in edf_emitted else None,
                )
                if source.edf is not None:
                    edf_emitted.add(id(source))
            elif fitted in pinned_levels:
                display_row = _CoefRow(
                    name=row_name,
                    group=term_prefix,
                    coef=0.0,
                    active=True,
                    is_reference=True,
                    level_group=group_ids.get(fitted, ""),
                    level_fit="pinned",
                )
            else:
                continue
            if source is not None and (
                source.level_n_obs is not None or source.level_exposure_share is not None
            ):
                if id(source) in diagnostics_emitted:
                    display_row = replace(
                        display_row,
                        level_n_obs=None,
                        level_exposure_share=None,
                    )
                else:
                    diagnostics_emitted.add(id(source))
            display_rows.append(display_row)

        if matched_indices:
            insert_at = min(matched_indices)
        else:
            feature_indices = [index for index, row in enumerate(rows) if row.group == term_prefix]
            if feature_indices:
                insert_at = max(feature_indices) + 1
            else:
                group_positions = [
                    index for index, group in enumerate(groups) if group.feature_name == feature
                ]
                later_group_names = (
                    {
                        name
                        for group in groups[max(group_positions) + 1 :]
                        for name in (group.name, group.feature_name)
                    }
                    if group_positions
                    else set()
                )
                insert_at = next(
                    (
                        index
                        for index, row in enumerate(rows)
                        if row.group in later_group_names
                        or row.name in later_group_names
                        or any(
                            row.name.startswith(f"{group_name}[")
                            for group_name in later_group_names
                        )
                    ),
                    len(rows),
                )
        matched = set(matched_indices)
        rows = [
            *rows[:insert_at],
            *display_rows,
            *(
                row
                for index, row in enumerate(rows[insert_at:], start=insert_at)
                if index not in matched
            ),
        ]

    return SummaryLevelDisplay(mode, tuple(rows), tuple(legends))
