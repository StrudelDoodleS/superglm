"""Summary-only presentation of collapsed categorical levels."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.features import Categorical, OrderedCategorical, Spline
from superglm.features.grouping import collapse_levels
from superglm.inference.summary import _CoefRow
from superglm.types import GroupSlice


def _categorical_case():
    levels = ["A", "B", "C", "D", "E", "F"]
    grouping = collapse_levels(
        levels,
        groups={"B+C fitted label": ["B", "C"], "D+E fitted label": ["D", "E"]},
        order=levels,
    )
    spec = Categorical(base="A", grouping=grouping)
    spec.build(np.asarray(levels))
    rows = [
        _CoefRow(name="Intercept", coef=0.5),
        _CoefRow(
            name="territory[B+C fitted label]",
            group="territory",
            coef=0.2,
            se=0.05,
            z=4.0,
            p=0.001,
            ci_low=0.1,
            ci_high=0.3,
            edf=1.0,
        ),
        _CoefRow(
            name="territory[D+E fitted label]",
            group="territory",
            coef=-0.1,
            se=0.04,
            z=-2.5,
            p=0.012,
            ci_low=-0.18,
            ci_high=-0.02,
        ),
        _CoefRow(name="territory[F]", group="territory", coef=0.05, se=0.03),
    ]
    return spec, rows, [GroupSlice("territory", 0, 3)]


@pytest.mark.parametrize("value", ["", "expand", "ungrouped", "GROUPED", None])
def test_validate_level_display_rejects_unknown_values(value):
    from superglm.inference.summary_levels import validate_level_display

    with pytest.raises(ValueError, match=r"expanded.*grouped"):
        validate_level_display(value)


def test_expanded_display_uses_original_levels_and_deterministic_groups():
    from superglm.inference.summary_levels import build_summary_level_display

    spec, rows, groups = _categorical_case()
    display = build_summary_level_display(
        rows,
        specs={"territory": spec},
        groups=groups,
        level_display="expanded",
    )

    territory = [row for row in display.rows if row.group == "territory"]
    assert [row.name for row in territory] == [
        "territory[A]",
        "territory[B]",
        "territory[C]",
        "territory[D]",
        "territory[E]",
        "territory[F]",
    ]
    assert [row.level_group for row in territory] == ["", "G1", "G1", "G2", "G2", ""]
    assert [row.edf for row in territory].count(1.0) == 1
    assert territory[0].is_reference is True
    assert (territory[0].coef, territory[0].se, territory[0].p) == (0.0, None, None)
    assert territory[1].coef == territory[2].coef == 0.2
    assert rows[1].name == "territory[B+C fitted label]"


def test_grouped_display_uses_one_row_per_fitted_group_and_legends():
    from superglm.inference.summary_levels import build_summary_level_display

    spec, rows, groups = _categorical_case()
    display = build_summary_level_display(
        rows,
        specs={"territory": spec},
        groups=groups,
        level_display="grouped",
    )

    territory = [row for row in display.rows if row.group == "territory"]
    assert [(row.name, row.level_group) for row in territory] == [
        ("territory[A]", ""),
        ("territory", "G1"),
        ("territory", "G2"),
        ("territory[F]", ""),
    ]
    assert [(item.feature, item.group_id, item.members) for item in display.level_groups] == [
        ("territory", "G1", ("B", "C")),
        ("territory", "G2", ("D", "E")),
    ]
    assert territory[0].is_reference is True
    assert territory[1].coef == 0.2


def test_identity_categorical_adds_reference_without_group_metadata():
    from superglm.inference.summary_levels import build_summary_level_display

    spec = Categorical(base="A")
    spec.build(np.asarray(["A", "B", "C"]))
    canonical = [
        _CoefRow(name="region[B]", group="region", coef=0.2, se=0.1, edf=1.0),
        _CoefRow(name="region[C]", group="region", coef=-0.1, se=0.1),
    ]

    display = build_summary_level_display(
        canonical,
        specs={"region": spec},
        groups=[GroupSlice("region", 0, 2)],
    )

    assert [row.name for row in display.rows] == ["region[A]", "region[B]", "region[C]"]
    assert display.rows[0].is_reference is True
    assert display.has_level_groups is False


def test_reference_group_expands_to_reference_members_and_compacts_to_one_row():
    from superglm.inference.summary_levels import build_summary_level_display

    levels = ["A", "B", "C", "D"]
    grouping = collapse_levels(
        levels,
        groups={"base fitted label": ["A", "B"]},
        order=levels,
    )
    spec = Categorical(base="base fitted label", grouping=grouping)
    spec.build(np.asarray(levels))
    canonical = [
        _CoefRow(name="region[C]", group="region", coef=0.2, se=0.1, edf=1.0),
        _CoefRow(name="region[D]", group="region", coef=-0.1, se=0.1),
    ]
    groups = [GroupSlice("region", 0, 2)]

    expanded = build_summary_level_display(
        canonical,
        specs={"region": spec},
        groups=groups,
        level_display="expanded",
    )
    grouped = build_summary_level_display(
        canonical,
        specs={"region": spec},
        groups=groups,
        level_display="grouped",
    )

    assert [(row.name, row.level_group, row.is_reference) for row in expanded.rows[:2]] == [
        ("region[A]", "G1", True),
        ("region[B]", "G1", True),
    ]
    assert (grouped.rows[0].name, grouped.rows[0].level_group) == ("region", "G1")
    assert grouped.rows[0].is_reference is True


def test_ordered_spline_whole_feature_row_stays_before_expanded_reference():
    from superglm.inference.summary_levels import build_summary_level_display

    levels = ["A", "B", "C", "D"]
    grouping = collapse_levels(
        levels,
        groups={"B+C fitted label": ["B", "C"]},
        order=levels,
    )
    spec = OrderedCategorical(
        order=levels,
        basis=Spline(kind="ps", k=5),
        base="A",
        grouping=grouping,
    )
    spec.build(np.asarray(levels))
    canonical = [
        _CoefRow(name="band", group="band", is_spline=True, active=True, edf=1.5),
        _CoefRow(name="band[A]", group="band", coef=0.0, se=0.0),
        _CoefRow(name="band[B+C fitted label]", group="band", coef=0.2, se=0.1),
        _CoefRow(name="band[D]", group="band", coef=0.3, se=0.1),
    ]

    display = build_summary_level_display(
        canonical,
        specs={"band": spec},
        groups=[GroupSlice("band", 0, 3)],
        level_display="expanded",
    )

    assert [row.name for row in display.rows] == [
        "band",
        "band[A]",
        "band[B]",
        "band[C]",
        "band[D]",
    ]
    assert display.rows[1].is_reference is True
    assert display.rows[1].active is True
    assert [row.edf for row in display.rows] == [1.5, None, None, None, None]


def test_ordered_step_groups_expand_in_original_order():
    from superglm.inference.summary_levels import build_summary_level_display

    levels = ["low", "medium", "high", "very high"]
    grouping = collapse_levels(
        levels,
        groups={"upper fitted label": ["high", "very high"]},
        order=levels,
    )
    with pytest.warns(FutureWarning, match="step smoothing"):
        spec = OrderedCategorical(
            order=levels,
            basis="step",
            base="low",
            grouping=grouping,
        )
    spec.build(np.asarray(levels))
    canonical = [
        _CoefRow(name="risk[medium]", group="risk", coef=0.1, se=0.03, edf=1.5),
        _CoefRow(name="risk[upper fitted label]", group="risk", coef=0.4, se=0.08),
    ]

    display = build_summary_level_display(
        canonical,
        specs={"risk": spec},
        groups=[GroupSlice("risk", 0, 2)],
        level_display="expanded",
    )

    assert [row.name for row in display.rows] == [
        "risk[low]",
        "risk[medium]",
        "risk[high]",
        "risk[very high]",
    ]
    assert [row.level_group for row in display.rows] == ["", "", "G1", "G1"]


def test_group_ids_restart_per_feature_and_unmatched_rows_survive():
    from superglm.inference.summary_levels import build_summary_level_display

    territory_spec, territory_rows, territory_groups = _categorical_case()
    zone_levels = ["K", "L", "M"]
    zone_grouping = collapse_levels(
        zone_levels,
        groups={"L+M fitted label": ["L", "M"]},
        order=zone_levels,
    )
    zone_spec = Categorical(base="K", grouping=zone_grouping)
    zone_spec.build(np.asarray(zone_levels))
    unmatched = _CoefRow(name="territory[legacy-unmatched]", group="territory", coef=9.0)
    rows = [
        *territory_rows,
        unmatched,
        _CoefRow(name="zone[L+M fitted label]", group="zone", coef=0.4, se=0.1),
    ]

    display = build_summary_level_display(
        rows,
        specs={"territory": territory_spec, "zone": zone_spec},
        groups=[*territory_groups, GroupSlice("zone", 3, 4)],
        level_display="expanded",
    )

    assert [(item.feature, item.group_id) for item in display.level_groups] == [
        ("territory", "G1"),
        ("territory", "G2"),
        ("zone", "G1"),
    ]
    assert any(row is unmatched for row in display.rows)
