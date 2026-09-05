"""Summary-only presentation of collapsed categorical levels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, SuperGLM
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


def test_expanded_display_reports_group_level_qs_diagnostics_once():
    from superglm.inference.summary_levels import build_summary_level_display

    spec, rows, groups = _categorical_case()
    rows[1].quasi_separated = True
    rows[1].advisory_trigger = "thin_level"
    rows[1].level_n_obs = 12
    rows[1].level_exposure_share = 0.004
    presentation = build_summary_level_display(
        rows,
        specs={"territory": spec},
        groups=groups,
        level_display="expanded",
    )
    summary = ModelSummary({}, _model_info(), rows, level_presentation=presentation)
    grouped_members = [row for row in presentation.rows if row.level_group == "G1"]

    assert [row.quasi_separated for row in grouped_members] == [True, True]
    assert [row.level_n_obs for row in grouped_members] == [12, None]
    assert [row.level_exposure_share for row in grouped_members] == [0.004, None]
    # The diagnostics are dropped from the later member so the footnote does
    # not double-count, but the TRIGGER survives on both -- a renderer that
    # re-derived it from ``level_n_obs`` here would call the second member an
    # outsized standard error, on a categorical level with no units whose
    # standard error was never tested (issue #239).
    assert [row.advisory_trigger for row in grouped_members] == ["thin_level", "thin_level"]

    from superglm.editor.summaries import _compact_summary_row

    assert [_compact_summary_row(row)["advisory_kind"] for row in grouped_members] == [
        "thin_level",
        "thin_level",
    ]
    text = str(summary)
    html = summary._repr_html_()
    assert text.count("12 obs (0.40% exposure)") == 1
    assert "territory G1: 12 obs (0.40% exposure)" in text
    assert html.count("12 obs (0.40% exposure)") == 1
    assert "territory G1: " in html


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


@pytest.mark.parametrize(
    ("include_feature_row", "later_row_name", "later_row_group"),
    [
        (False, "region[B]", "region"),
        (False, "region[P1]", "region P(3)"),
        (True, "region[B]", "region"),
    ],
)
def test_reference_only_group_synthesizes_rows_without_a_canonical_level_row(
    include_feature_row,
    later_row_name,
    later_row_group,
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
    canonical.append(_CoefRow(name=later_row_name, group=later_row_group, coef=0.2))
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
        later_row_name,
    ]
    first_level = 2 if include_feature_row else 1
    assert all(row.is_reference for row in expanded.rows[first_level : first_level + 3])
    assert [(row.name, row.level_group, row.is_reference) for row in grouped.rows] == [
        ("Intercept", "", False),
        *(([("band", "", False)]) if include_feature_row else []),
        ("band", "G1", True),
        (later_row_name, "", False),
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


def test_ascii_summary_separates_columns_for_large_fitted_coefficients():
    rng = np.random.default_rng(226)
    x = np.linspace(-1.0, 1.0, 300)
    X = pd.DataFrame({"x": x})
    y = 250_000.0 + 3_000.0 * x + rng.normal(0.0, 500.0, len(x))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(X, y)

    text = str(model.summary())
    header = next(line for line in text.splitlines() if "coef" in line and "std err" in line)
    intercept = next(line for line in text.splitlines() if "Intercept" in line)
    header_fields = header.strip("║ ").replace("std err", "std_err").split()
    intercept_fields = intercept.strip("║ ").split()

    assert abs(float(intercept_fields[1])) >= 1e5
    assert len(intercept_fields) == len(header_fields)
    assert all(np.isfinite(float(value)) for value in intercept_fields[1:7])
    box_lines = [line for line in text.splitlines() if line.startswith(("╔", "║", "╠", "╟", "╚"))]
    assert len({len(line) for line in box_lines}) == 1


def test_ascii_summary_large_coefficient_ci_brackets_the_printed_estimate():
    rng = np.random.default_rng(226)
    x = np.linspace(-1.0, 1.0, 3000)
    X = pd.DataFrame({"x": x})
    y = 250_000.0 + 3_000.0 * x + rng.normal(0.0, 500.0, len(x))
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(X, y)

    summary = model.summary()
    fitted = next(row for row in summary._coef_rows if row.name == "Intercept")
    assert abs(fitted.coef) >= 1e5
    true_width = fitted.ci_high - fitted.ci_low
    assert true_width > 0.0

    intercept = next(line for line in str(summary).splitlines() if "Intercept" in line)
    fields = intercept.strip("║ ").split()
    printed_coef = float(fields[1])
    printed_low = float(fields[5])
    printed_high = float(fields[6])

    assert printed_high > printed_low
    assert printed_low <= printed_coef <= printed_high
    assert printed_high - printed_low == pytest.approx(true_width, rel=1e-3)


@pytest.mark.parametrize("magnitude", [10232.9551, 249965.223])
def test_ascii_summary_number_precision_does_not_depend_on_sign(magnitude: float):
    from superglm.inference.summary import _format_ascii_number

    positive = _format_ascii_number(magnitude, decimals=4, width=10)
    negative = _format_ascii_number(-magnitude, decimals=4, width=10)

    assert len(positive) <= 10
    assert len(negative) <= 10
    assert float(positive) == pytest.approx(magnitude, rel=1e-7)
    assert float(negative) == pytest.approx(-magnitude, rel=1e-7)


def test_ascii_summary_uses_bounded_scientific_notation_for_extreme_nonzero_values():
    summary = ModelSummary(
        {},
        _model_info(),
        [
            _CoefRow(
                name="Extreme",
                coef=-1e100,
                se=1e-14,
                z=-1e114,
                p=1e-300,
                ci_low=-1e100,
                ci_high=1e100,
            )
        ],
    )

    text = str(summary)
    row = next(line for line in text.splitlines() if "Extreme" in line)
    fields = row.strip("║ ").split()

    assert len(fields) == 9
    assert float(fields[1]) == pytest.approx(-1e100)
    assert float(fields[2]) == pytest.approx(1e-14)
    assert float(fields[3]) == pytest.approx(-1e114)
    assert float(fields[4]) == pytest.approx(1e-300)
    assert float(fields[5]) == pytest.approx(-1e100)
    assert float(fields[6]) == pytest.approx(1e100)
    assert all("e" in field.lower() for field in fields[1:7])
    box_lines = [line for line in text.splitlines() if line.startswith(("╔", "║", "╠", "╟", "╚"))]
    assert len({len(line) for line in box_lines}) == 1


@pytest.mark.parametrize(
    "value",
    [
        np.finfo(np.float64).max,
        np.nextafter(np.finfo(np.float64).max, 0.0),
        -np.finfo(np.float64).max,
        -np.nextafter(np.finfo(np.float64).max, 0.0),
    ],
)
def test_ascii_summary_float_boundary_tokens_parse_as_finite(value: float):
    summary = ModelSummary(
        {},
        _model_info(),
        [
            _CoefRow(
                name="Boundary",
                coef=value,
                se=abs(value),
                z=value,
                p=abs(value),
                ci_low=-abs(value),
                ci_high=abs(value),
            )
        ],
    )

    row = next(line for line in str(summary).splitlines() if "Boundary" in line)
    fields = row.strip("║ ").split()

    assert len(fields) == 9
    assert all(np.isfinite(float(token)) for token in fields[1:7])


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


# ── Bound level universes (spec 2026-08-11 §3.8) ──────────────────


def _pinned_level_model(base: str = "A"):
    rng = np.random.default_rng(20260811)
    territory = np.tile(np.asarray(["A", "B", "C"]), 40)
    X = pd.DataFrame({"territory": territory})
    means = {"A": 1.0, "B": 1.4, "C": 0.8}
    y = rng.poisson(np.asarray([means[level] for level in territory])).astype(float)
    model = SuperGLM(
        features={"territory": Categorical(base=base, levels=["A", "B", "C", "D"])},
    )
    with pytest.warns(UserWarning, match=r"pinned to base|falling back") as caught:
        model.fit(X, y)
    assert any("pinned to base" in str(w.message) for w in caught)
    return model


@pytest.fixture(scope="module")
def pinned_level_model():
    return _pinned_level_model()


def test_summary_gives_a_pinned_level_its_own_row_marked_pinned(pinned_level_model):
    # A declared level with no training rows has no coefficient, so nothing in
    # the canonical rows names it and the level display used to drop it
    # silently -- the one surface that must not, because the pin is exactly
    # what the reader has to see.
    rows = [row for row in pinned_level_model.summary()._display_rows if row.group == "territory"]

    assert [row.name for row in rows] == [
        "territory[A]",
        "territory[B]",
        "territory[C]",
        "territory[D]",
    ]
    pinned = rows[-1]
    assert pinned.level_fit == "pinned"
    assert pinned.coef == 0.0
    assert pinned.is_reference
    assert pinned.se is None and pinned.p is None
    # The pin marks the pinned level alone; the base row is a reference for a
    # different reason and the fitted levels are not pinned at all.
    assert [row.level_fit for row in rows[:-1]] == [None, None, None]


def test_ascii_summary_prints_the_pin_in_the_fit_column(pinned_level_model):
    text = str(pinned_level_model.summary())

    assert "Fit" in text
    pinned_line = next(line for line in text.splitlines() if "territory[D]" in line)
    assert "pinned" in pinned_line
    assert "0.0000" in pinned_line


def test_summary_records_the_bound_universe_and_its_source(pinned_level_model):
    universes = pinned_level_model.summary()["level_universes"]

    assert universes["territory"] == {
        "levels": ["A", "B", "C", "D"],
        "level_source": "declared",
        "base_level": "A",
        "base_fallback": None,
        "pinned_levels": ["D"],
    }


def test_summary_records_a_base_that_fell_back_to_an_observed_level():
    model = _pinned_level_model(base="D")

    record = model.summary()["level_universes"]["territory"]

    assert record["base_fallback"] == ("D", "A")
    assert record["base_level"] == "A"
    assert record["pinned_levels"] == ["D"]


def test_model_and_metrics_summaries_record_the_same_universes(pinned_level_model):
    model = pinned_level_model
    X = pd.DataFrame({"territory": np.tile(np.asarray(["A", "B", "C"]), 40)})
    y = np.asarray(model.predict(X), dtype=float)

    metrics_summary = model.metrics(X, y).summary()

    assert metrics_summary["level_universes"] == model.summary()["level_universes"]


def test_summary_reports_an_inferred_universe_without_pins(grouped_model_data):
    model, _, _, _ = grouped_model_data

    record = model.summary()["level_universes"]["territory"]

    assert record["level_source"] == "inferred"
    assert record["pinned_levels"] == []
    assert record["base_level"] == "A"


def test_summary_marks_a_pinned_ordered_special_as_pinned():
    # The Fit column already separates "smooth" from "free"; a special that had
    # no rows is neither -- it is a declared level carrying no contribution,
    # and reading "free" there claims a fitted coefficient that does not exist.
    order = ["1", "2", "3", "4", "MISSING"]
    band = np.tile(np.asarray(order[:4]), 20)
    X = pd.DataFrame({"band": band})
    rng = np.random.default_rng(20260811)
    y = rng.poisson(1.0, size=len(X)).astype(float)
    model = SuperGLM(
        features={
            "band": OrderedCategorical(
                order=order,
                specials=["MISSING"],
                basis=Spline(kind="ps", k=5),
            )
        },
    )
    with pytest.warns(UserWarning, match="pinned to zero contribution"):
        model.fit(X, y)

    rows = [row for row in model.summary()._display_rows if row.name.startswith("band[")]
    assert [row.name for row in rows] == [f"band[{level}]" for level in order]
    assert [row.level_fit for row in rows] == ["smooth"] * 4 + ["pinned"]
    assert model.summary()["level_universes"]["band"]["pinned_levels"] == ["MISSING"]


def test_summary_reports_an_integer_base_level_of_zero():
    # The audit payload used to read the base as `getattr(...) or None`, which
    # is the one falsey test that erases a real answer: `0` and `False` are
    # perfectly ordinary base levels on an integer or boolean universe, and the
    # governance record then claimed the term had no reference level at all --
    # while `reconstruct()`, reading the same attribute directly, said `0`.
    rng = np.random.default_rng(20260811)
    band = np.tile(np.asarray([0, 1, 2]), 40)
    X = pd.DataFrame({"band": band})
    y = rng.poisson(1.0, size=len(X)).astype(float)
    model = SuperGLM(features={"band": Categorical(base="first")})
    model.fit(X, y)
    assert model._specs["band"]._base_level == 0, "precondition: the base IS the falsey level"

    record = model.summary()["level_universes"]["band"]

    assert record["base_level"] == 0
    assert record["base_level"] is not None
    assert model.reconstruct_feature("band")["base_level"] == record["base_level"]


def test_an_unbuilt_spec_still_reports_no_base_level():
    # The empty string is the pre-build sentinel `__init__` writes, and it must
    # keep reading as "no base" -- the fix narrows the falsey test, it does not
    # remove it.
    from superglm.inference.summary_levels import build_level_universes

    spec = Categorical(base="first")
    spec._levels = ["a", "b"]  # a universe with no build behind it

    record = build_level_universes({"g": spec})["g"]

    assert record["base_level"] is None


def _factor_smooth_model():
    """A REML fit whose only level-bearing term lives in the interactions."""
    from superglm import FactorSmooth, LambdaPolicy

    x_level = np.linspace(-1.0, 1.0, 40)
    observed = np.array(["a", "b", "c"], dtype=object)
    x = np.tile(x_level, len(observed))
    group = np.repeat(observed, len(x_level))
    y = 1.1 + np.concatenate(
        [
            0.8 * np.sin(2.2 * x_level),
            -0.4 * np.cos(1.7 * x_level),
            0.3 * x_level,
        ]
    )
    X = pd.DataFrame({"x": x, "group": group})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        direct_solve="gram",
        interactions=[
            FactorSmooth(
                "x",
                group="group",
                k=6,
                levels=["a", "b", "c", "ghost"],
                lambda_policy=LambdaPolicy.fixed(1.4),
            )
        ],
    )
    model.fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")
    return model, X, y


def test_summary_records_a_factor_smooth_group_universe():
    # FactorSmooth carries a bound group universe and a level source, but it
    # lives ONLY in the interaction namespace -- so a payload built from
    # `model._specs` alone dropped precisely the term whose universe is least
    # visible in the coefficient table, and the governance record showed nothing
    # at all for a model whose only level-bearing term was this one.
    model, _, _ = _factor_smooth_model()

    universes = model.summary()["level_universes"]

    assert "x:group:fs" in universes
    assert universes["x:group:fs"] == {
        "levels": ["a", "b", "c", "ghost"],
        "level_source": "declared",
        # A penalized term has no reference level and pools empty levels
        # through its penalty, so it never pins.
        "base_level": None,
        "base_fallback": None,
        "pinned_levels": [],
    }


def test_model_and_metrics_summaries_agree_on_a_factor_smooth_universe():
    # The two payloads are read as one surface; adding a namespace to one of
    # them and not the other is exactly how they drift apart.
    model, X, y = _factor_smooth_model()

    assert model.metrics(X, y).summary()["level_universes"] == model.summary()["level_universes"]
