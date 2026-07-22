"""Summary-only presentation of collapsed categorical levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features import Categorical, OrderedCategorical, Spline
from superglm.features.grouping import collapse_levels
from superglm.inference.summary import ModelSummary, _CoefRow
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


def _model_info() -> dict[str, object]:
    return {
        "family": "Poisson",
        "link": "Log",
        "penalty": "None",
        "method": "ML",
        "n_obs": 100,
        "effective_df": 4.0,
        "phi": 1.0,
        "pearson_chi2": 98.0,
        "deviance": 95.0,
        "log_likelihood": -50.0,
        "aic": 108.0,
        "aicc": 108.5,
        "bic": 118.0,
        "ebic": 118.0,
        "converged": True,
        "n_iter": 4,
    }


def _rendered_summary(level_display: str) -> ModelSummary:
    from superglm.inference.summary_levels import build_summary_level_display

    spec, rows, groups = _categorical_case()
    presentation = build_summary_level_display(
        rows,
        specs={"territory": spec},
        groups=groups,
        level_display=level_display,
    )
    return ModelSummary({}, _model_info(), rows, level_presentation=presentation)


@pytest.fixture(scope="module")
def grouped_model_data():
    rng = np.random.default_rng(20260722)
    territory = np.tile(np.asarray(["A", "B", "C", "D"]), 40)
    X = pd.DataFrame({"territory": territory})
    means = {"A": 1.0, "B": 1.4, "C": 1.4, "D": 0.8}
    y = rng.poisson(np.asarray([means[level] for level in territory])).astype(float)
    weights = np.linspace(0.8, 1.2, len(X))
    grouping = collapse_levels(
        territory,
        groups={"B+C fitted label": ["B", "C"]},
        order=["A", "B", "C", "D"],
    )
    model = SuperGLM(
        features={"territory": Categorical(base="A", grouping=grouping)},
    )
    model.fit(X, y, sample_weight=weights)
    return model, X, y, weights


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


def test_numeric_identity_categorical_preserves_its_reference_level():
    from superglm.inference.summary_levels import build_summary_level_display

    spec = Categorical(base="first")
    spec.build(np.asarray([10, 20, 30]))
    canonical = [
        _CoefRow(name="region[20]", group="region", coef=0.2, se=0.1, edf=1.0),
        _CoefRow(name="region[30]", group="region", coef=-0.1, se=0.1),
    ]

    display = build_summary_level_display(
        canonical,
        specs={"region": spec},
        groups=[GroupSlice("region", 0, 2)],
    )

    assert [row.name for row in display.rows] == ["region[10]", "region[20]", "region[30]"]
    assert display.rows[0].is_reference is True
    assert (display.rows[0].coef, display.rows[0].se, display.rows[0].p) == (0.0, None, None)


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


def test_reference_only_ordered_group_still_expands_and_groups():
    from superglm.inference.summary_levels import build_summary_level_display

    levels = ["A", "B", "C"]
    grouping = collapse_levels(
        levels,
        groups={"all fitted label": levels},
        order=levels,
    )
    with pytest.warns(UserWarning, match="clamped to 0"):
        spec = OrderedCategorical(
            order=levels,
            base="all fitted label",
            grouping=grouping,
        )
    info = spec.build(np.asarray(levels))
    canonical = [
        _CoefRow(name="band", group="band", is_spline=True, active=True, edf=1.0),
        _CoefRow(name="band[all fitted label]", group="band", coef=0.0),
    ]
    groups = [GroupSlice("band", 0, info.n_cols)]

    expanded = build_summary_level_display(
        canonical,
        specs={"band": spec},
        groups=groups,
        level_display="expanded",
    )
    grouped = build_summary_level_display(
        canonical,
        specs={"band": spec},
        groups=groups,
        level_display="grouped",
    )

    assert [row.name for row in expanded.rows] == ["band", "band[A]", "band[B]", "band[C]"]
    assert all(row.is_reference for row in expanded.rows[1:])
    assert [row.level_group for row in expanded.rows[1:]] == ["G1", "G1", "G1"]
    assert [(row.name, row.level_group, row.is_reference) for row in grouped.rows] == [
        ("band", "", False),
        ("band", "G1", True),
    ]
    assert [(legend.group_id, legend.members) for legend in grouped.level_groups] == [
        ("G1", ("A", "B", "C")),
    ]


@pytest.mark.parametrize("include_feature_row", [False, True])
def test_reference_only_group_synthesizes_rows_without_a_canonical_level_row(
    include_feature_row,
):
    from superglm.inference.summary_levels import build_summary_level_display

    levels = ["A", "B", "C"]
    grouping = collapse_levels(
        levels,
        groups={"all fitted label": levels},
        order=levels,
    )
    with pytest.warns(UserWarning, match="clamped to 0"):
        spec = OrderedCategorical(
            order=levels,
            base="all fitted label",
            grouping=grouping,
        )
    info = spec.build(np.asarray(levels))
    canonical = [_CoefRow(name="Intercept", coef=0.5)]
    if include_feature_row:
        canonical.append(_CoefRow(name="band", group="band", is_spline=True, active=True, edf=1.0))
    canonical.append(_CoefRow(name="region[B]", group="region", coef=0.2))
    groups = [
        GroupSlice("band", 0, info.n_cols),
        GroupSlice("region", info.n_cols, info.n_cols + 1),
    ]

    expanded = build_summary_level_display(
        canonical,
        specs={"band": spec},
        groups=groups,
        level_display="expanded",
    )
    grouped = build_summary_level_display(
        canonical,
        specs={"band": spec},
        groups=groups,
        level_display="grouped",
    )

    assert [row.name for row in expanded.rows] == [
        "Intercept",
        *(["band"] if include_feature_row else []),
        "band[A]",
        "band[B]",
        "band[C]",
        "region[B]",
    ]
    first_level = 2 if include_feature_row else 1
    assert all(row.is_reference for row in expanded.rows[first_level : first_level + 3])
    assert [(row.name, row.level_group, row.is_reference) for row in grouped.rows] == [
        ("Intercept", "", False),
        *(([("band", "", False)]) if include_feature_row else []),
        ("band", "G1", True),
        ("region[B]", "", False),
    ]
    assert [(legend.group_id, legend.members) for legend in grouped.level_groups] == [
        ("G1", ("A", "B", "C")),
    ]


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


def test_model_summary_retains_canonical_rows_and_accepts_presentation():
    summary = _rendered_summary("expanded")

    assert summary._coef_rows[1].name == "territory[B+C fitted label]"
    assert [row.name for row in summary._display_rows[1:]] == [
        "territory[A]",
        "territory[B]",
        "territory[C]",
        "territory[D]",
        "territory[E]",
        "territory[F]",
    ]
    assert summary._level_display == "expanded"


def test_ascii_summary_renders_expanded_rows_group_column_and_reference():
    text = str(_rendered_summary("expanded"))

    assert "Level group" in text
    assert "territory[A]" in text
    assert "territory[B]" in text
    assert "territory[C]" in text
    assert "territory[B+C fitted label]" not in text
    assert "G1" in text and "G2" in text
    reference_line = next(line for line in text.splitlines() if "territory[A]" in line)
    assert "0.0000" in reference_line
    assert "ref" in reference_line


def test_ascii_grouped_summary_uses_short_rows_and_membership_legend():
    text = str(_rendered_summary("grouped"))

    assert "territory[B+C fitted label]" not in text
    assert "Level groups (territory):" in text
    assert "G1 = B, C" in text
    assert "G2 = D, E" in text
    assert max(len(line) for line in text.splitlines()) <= 100


def test_html_summary_renders_expanded_group_column_and_reference():
    html = _rendered_summary("expanded")._repr_html_()

    assert "Level group" in html
    assert "territory[A]" in html
    assert "territory[B]" in html
    assert "territory[C]" in html
    assert "territory[B+C fitted label]" not in html
    reference_row = next(row for row in html.split("</tr>") if "territory[A]" in row)
    assert "0.0000" in reference_row
    assert "ref" in reference_row


def test_html_grouped_summary_escapes_exact_members_in_wrapped_legend():
    from superglm.inference.summary_levels import build_summary_level_display

    levels = ["A", "<script>alert(1)</script>", "B & C"]
    grouping = collapse_levels(
        levels,
        groups={"malicious fitted label": levels[1:]},
        order=levels,
    )
    spec = Categorical(base="A", grouping=grouping)
    spec.build(np.asarray(levels))
    canonical = [
        _CoefRow(
            name="region[malicious fitted label]",
            group="region",
            coef=0.2,
            se=0.1,
        )
    ]
    presentation = build_summary_level_display(
        canonical,
        specs={"region": spec},
        groups=[GroupSlice("region", 0, 1)],
        level_display="grouped",
    )
    summary = ModelSummary(
        {},
        _model_info(),
        canonical,
        level_presentation=presentation,
    )

    html = summary._repr_html_()
    assert "Level groups (region):" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "B &amp; C" in html


def test_ungrouped_summary_omits_level_group_column():
    from superglm.inference.summary_levels import build_summary_level_display

    spec = Categorical(base="A")
    spec.build(np.asarray(["A", "B"]))
    canonical = [_CoefRow(name="region[B]", group="region", coef=0.2, se=0.1)]
    presentation = build_summary_level_display(
        canonical,
        specs={"region": spec},
        groups=[GroupSlice("region", 0, 1)],
        level_display="expanded",
    )
    summary = ModelSummary({}, _model_info(), canonical, level_presentation=presentation)

    assert "Level group" not in str(summary)
    assert "Level group" not in summary._repr_html_()


def test_model_summary_defaults_to_expanded_original_levels(grouped_model_data):
    model, _, _, _ = grouped_model_data

    summary = model.summary()

    territory = [row for row in summary._display_rows if row.group == "territory"]
    assert [row.name for row in territory] == [
        "territory[A]",
        "territory[B]",
        "territory[C]",
        "territory[D]",
    ]
    assert [row.level_group for row in territory] == ["", "G1", "G1", ""]


def test_model_summary_caches_expanded_and_grouped_modes_separately(grouped_model_data):
    model, _, _, _ = grouped_model_data

    expanded = model.summary()
    grouped = model.summary(level_display="grouped")

    assert expanded._level_display == "expanded"
    assert grouped._level_display == "grouped"
    assert model.summary() is expanded
    assert model.summary(level_display="grouped") is grouped
    assert expanded is not grouped
    territory = [row for row in grouped._display_rows if row.group == "territory"]
    assert [(row.name, row.level_group) for row in territory] == [
        ("territory[A]", ""),
        ("territory", "G1"),
        ("territory[D]", ""),
    ]


def test_model_summary_validates_level_display_before_cache(grouped_model_data):
    model, _, _, _ = grouped_model_data
    model.summary(level_display="grouped")

    with pytest.raises(ValueError, match=r"expanded.*grouped"):
        model.summary(level_display="ungrouped")


def test_model_and_metrics_summary_share_grouped_display(grouped_model_data):
    model, X, y, weights = grouped_model_data

    model_rows = model.summary(level_display="grouped")._display_rows
    metric_rows = (
        model.metrics(X, y, sample_weight=weights).summary(level_display="grouped")._display_rows
    )

    assert [(row.name, row.level_group) for row in metric_rows] == [
        (row.name, row.level_group) for row in model_rows
    ]


@pytest.mark.parametrize("wrapper_name", ["SuperGLMRegressor", "SuperGLMClassifier"])
def test_sklearn_summary_forwards_detail_and_level_display(grouped_model_data, wrapper_name):
    from superglm import sklearn as sklearn_module

    model, _, _, _ = grouped_model_data
    wrapper_class = getattr(sklearn_module, wrapper_name)
    wrapper = wrapper_class()
    wrapper._model = model
    wrapper.n_features_in_ = 1

    summary = wrapper.summary(detail="full", level_display="grouped")

    assert summary._detail == "full"
    assert summary._level_display == "grouped"


@pytest.mark.parametrize("level_display", ["expanded", "grouped"])
def test_editor_stale_inference_stays_suppressed_after_level_adaptation(
    grouped_model_data,
    level_display,
):
    model, _, _, _ = grouped_model_data
    prior_stale = getattr(model, "_editor_inference_stale", False)
    prior_edits = getattr(model, "_editor_edits", None)
    prior_cache = model._summary_cache
    model._editor_inference_stale = True
    model._editor_edits = {"terms": ["territory"]}
    model._summary_cache = {}
    try:
        summary = model.summary(level_display=level_display)
    finally:
        model._editor_inference_stale = prior_stale
        model._editor_edits = prior_edits
        model._summary_cache = prior_cache

    rows = [row for row in summary._display_rows if row.group == "territory"]
    assert rows
    assert all(row.se is None for row in rows)
    assert all(row.p is None for row in rows)
    assert all(row.ci_low is None and row.ci_high is None for row in rows)
