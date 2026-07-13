import warnings
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries

from superglm import (
    Categorical,
    Numeric,
    OrderedCategorical,
    Polynomial,
    Spline,
    SuperGLM,
    export_rating_tables,
)
from superglm.export.excel import write_rating_table_workbook
from superglm.export.rating_tables import RatingTablePayload, build_rating_table_payload
from superglm.export.summary import (
    SummaryExportPayload,
    SummaryOverviewRow,
    SummaryTermRow,
    build_summary_export_payload,
)


def _fit_export_model():
    rng = np.random.default_rng(123)
    n = 300
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "region": rng.choice(["A", "B", "C"], n),
            "score": rng.normal(0.0, 1.0, n),
        }
    )
    eta = -1.0 + 0.15 * np.sin(X["age"].to_numpy() / 8.0) + 0.2 * (X["region"] == "B")
    eta = eta + 0.05 * X["score"].to_numpy()
    y = rng.poisson(np.exp(eta)).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "age": Spline(n_knots=8),
            "region": Categorical(base="first"),
            "score": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=w)
    return model, X, y, w


def _fit_offset_export_model(distinct_terms: int = 2):
    rng = np.random.default_rng(987)
    n = 240
    if distinct_terms == 2:
        term = np.resize(np.array([12.0, 36.0]), n)
    else:
        term = np.linspace(12.0, 48.0, n)
    X = pd.DataFrame({"region": rng.choice(["A", "B"], n)})
    offset = np.log(term / 12.0)
    exposure = rng.uniform(0.5, 2.0, n)
    eta = -1.4 + 0.25 * (X["region"].to_numpy() == "B") + offset
    y = rng.poisson(np.exp(eta)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"region": Categorical(base="first")},
    )
    model.fit(X, y, sample_weight=exposure, offset=offset)
    return model, X, y, exposure


def _fit_term_offset_export_model(distinct_terms: int = 2, *, retain_fit_state: bool = True):
    rng = np.random.default_rng(988)
    n = 240
    if distinct_terms == 2:
        term = np.resize(np.array([12.0, 36.0]), n)
    else:
        term = np.linspace(12.0, 48.0, n)
    X = pd.DataFrame({"region": rng.choice(["A", "B"], n)})
    offset = np.log(term / 12.0)
    exposure = rng.uniform(0.5, 2.0, n)
    eta = -1.35 + 0.2 * (X["region"].to_numpy() == "B") + offset
    y = rng.poisson(np.exp(eta)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        retain_fit_state=retain_fit_state,
        features={"region": Categorical(base="first")},
    )
    model.fit(X, y, sample_weight=exposure, offset=offset)
    return model, X, y, exposure, term, offset


def _overview_values(payload: SummaryExportPayload) -> dict[tuple[str, str], object]:
    return {(row.section, row.metric): row.value for row in payload.overview}


def _workbook_payload(summary: SummaryExportPayload) -> RatingTablePayload:
    return RatingTablePayload(
        base_relativity=1.0,
        selected_n_bins=20,
        main_effects=[],
        interactions=[],
        discretization_impact=pd.DataFrame({"n_bins": [20]}),
        summary=summary,
    )


def _write_workbook(payload: RatingTablePayload, target) -> None:
    write_rating_table_workbook(
        payload,
        target,
        sheet_name="Rating Tables",
        summary_sheet_name="Model Summary",
        impact_sheet_name="Discretization Impact",
    )


def _table_records(ws, table_name: str) -> list[dict[str, object]]:
    min_col, min_row, max_col, max_row = range_boundaries(ws.tables[table_name].ref)
    headers = [ws.cell(row=min_row, column=column).value for column in range(min_col, max_col + 1)]
    return [
        {
            header: ws.cell(row=row, column=column).value
            for header, column in zip(headers, range(min_col, max_col + 1), strict=True)
        }
        for row in range(min_row + 1, max_row + 1)
    ]


def _fit_ordered_export_model():
    rng = np.random.default_rng(20260712)
    levels = [f"L{i}" for i in range(7)]
    codes = np.tile(np.arange(len(levels)), 80)
    rng.shuffle(codes)
    X = pd.DataFrame({"band": np.asarray(levels, dtype=object)[codes]})
    x_numeric = codes / (len(levels) - 1)
    weights = rng.uniform(0.6, 1.8, len(codes))
    eta = -0.8 + 0.9 * x_numeric + 0.15 * np.sin(2.0 * np.pi * x_numeric)
    y = rng.poisson(np.exp(eta) * weights).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "band": OrderedCategorical(
                order=levels,
                base="first",
                basis=Spline(kind="ps", k=7),
            )
        },
    )
    model.fit(X, y, sample_weight=weights)
    return model, levels


def _fit_selected_away_export_model():
    n = 160
    X = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, n),
            "cat": np.resize(np.array(["A", "B", "C", "D"], dtype=object), n),
        }
    )
    y = np.ones(n)
    model = SuperGLM(
        family="poisson",
        selection_penalty=10.0,
        features={"x": Numeric(), "cat": Categorical(base="first")},
    )
    model.fit(X, y)
    return model


def _fit_polynomial_categorical_export_model():
    rng = np.random.default_rng(20260713)
    n = 240
    x = rng.uniform(-1.0, 1.0, n)
    cat = rng.choice(["A", "B", "C"], n)
    y = rng.poisson(np.exp(0.2 + 0.3 * x + 0.15 * (cat == "B"))).astype(float)
    X = pd.DataFrame({"x": x, "cat": cat})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Polynomial(degree=2), "cat": Categorical(base="first")},
        interactions=[("x", "cat")],
    )
    model.fit(X, y)
    return model


def _fit_large_polynomial_interaction_export_model():
    rng = np.random.default_rng(20260714)
    n = 240
    a = rng.uniform(-1.0, 1.0, n)
    b = rng.uniform(-1.0, 1.0, n)
    y = rng.poisson(np.exp(0.2 + 0.25 * a - 0.15 * b + 0.1 * a * b)).astype(float)
    X = pd.DataFrame({"a": a, "b": b})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"a": Polynomial(degree=2), "b": Polynomial(degree=3)},
        interactions=[("a", "b")],
    )
    model.fit(X, y)
    return model


def test_summary_export_preserves_typed_overview_and_intercept_inference():
    model, _, _, _ = _fit_export_model()

    payload = build_summary_export_payload(model)
    overview = _overview_values(payload)
    intercept = next(row for row in payload.terms if row.term == "Intercept")

    assert overview[("Model", "Method")] == "MLE"
    assert isinstance(overview[("Fit", "Observations")], int)
    assert isinstance(overview[("Fit", "Effective DF")], float)
    assert isinstance(overview[("Fit", "Converged")], bool)
    assert isinstance(overview[("Fit", "Iterations")], int)
    assert isinstance(overview[("Information Criteria", "AIC")], float)
    assert intercept.kind == "coefficient"
    assert isinstance(intercept.estimate, float)
    assert isinstance(intercept.p_value, float)
    assert intercept.statistic_type == "z"
    assert intercept.active is True


def test_summary_export_maps_spline_to_one_global_wood_test():
    model, _, _, _ = _fit_export_model()

    payload = build_summary_export_payload(model)
    age_rows = [row for row in payload.terms if row.group == "age"]
    smooth = next(row for row in age_rows if row.kind == "smooth")

    assert len([row for row in age_rows if row.kind == "smooth"]) == 1
    assert smooth.estimate is None
    assert smooth.std_error is None
    assert smooth.ci_lower is None
    assert smooth.ci_upper is None
    assert isinstance(smooth.statistic, float)
    assert smooth.statistic_type == "chi2"
    assert isinstance(smooth.p_value, float)
    assert isinstance(smooth.edf, float)
    assert isinstance(smooth.smoothing_lambda, float)
    assert smooth.active is True
    assert "Wood (2013)" in "\n".join(payload.notes)


def test_summary_export_keeps_ordered_level_estimates_but_only_one_global_p_value():
    model, levels = _fit_ordered_export_model()

    payload = build_summary_export_payload(model)
    rows = [row for row in payload.terms if row.group == "band"]
    smooth_rows = [row for row in rows if row.kind == "smooth"]
    level_rows = [row for row in rows if row.kind == "level"]

    assert len(smooth_rows) == 1
    assert isinstance(smooth_rows[0].p_value, float)
    assert len(level_rows) == len(levels)
    assert [row.term for row in level_rows] == [f"band[{level}]" for level in levels]
    for row in level_rows:
        assert isinstance(row.estimate, float)
        assert isinstance(row.std_error, float)
        assert isinstance(row.ci_lower, float)
        assert isinstance(row.ci_upper, float)
        assert row.statistic is None
        assert row.statistic_type == ""
        assert row.p_value is None
        assert row.significance == ""


def test_summary_export_keeps_distribution_profile_values_typed():
    model, _, _, _ = _fit_export_model()
    model._nb_profile_result = SimpleNamespace(
        theta_hat=np.float64(2.75),
        nll=10.0,
        ci=lambda alpha: (np.float64(2.0), np.float64(3.5)),
    )
    model._tweedie_profile_result = SimpleNamespace(
        p_hat=np.float64(1.55),
        phi_hat=np.float64(0.8),
        method="exact",
        phi_method="pearson",
        nll=11.0,
        ci=lambda alpha: (np.float64(1.4), np.float64(1.7)),
    )
    model._summary_cache = None

    overview = _overview_values(build_summary_export_payload(model))

    assert overview[("Distribution Profile", "NB2 Theta")] == 2.75
    assert overview[("Distribution Profile", "NB2 Theta CI Lower")] == 2.0
    assert overview[("Distribution Profile", "NB2 Theta CI Upper")] == 3.5
    assert overview[("Distribution Profile", "NB2 Theta Method")] == "Profile (exact)"
    assert overview[("Distribution Profile", "Tweedie p")] == 1.55
    assert overview[("Distribution Profile", "Tweedie p CI Lower")] == 1.4
    assert overview[("Distribution Profile", "Tweedie p CI Upper")] == 1.7
    assert overview[("Distribution Profile", "Tweedie phi")] == 0.8
    assert overview[("Distribution Profile", "Tweedie p Method")] == (
        "Profile (exact, phi=pearson)"
    )


def test_summary_export_preserves_stale_and_fixed_offset_editor_caveats():
    model, _, _, _ = _fit_export_model()
    model._editor_inference_stale = True
    model._editor_edits = {"terms": ["score"]}
    model._editor_offset = {"terms": ["region"]}
    model._summary_cache = None

    notes = "\n".join(build_summary_export_payload(model).notes)

    assert "suppressed" in notes
    assert "Edited terms: score" in notes
    assert "conditional on those fixed offsets" in notes
    assert "Offset terms: region" in notes


def test_summary_export_marks_selected_away_parametric_and_level_rows_inactive():
    model = _fit_selected_away_export_model()
    assert tuple(model.result.rank_info.selected_group_names) == ()

    payload = build_summary_export_payload(model)

    intercept = next(row for row in payload.terms if row.term == "Intercept")
    selected_away = [row for row in payload.terms if row.term != "Intercept"]

    assert intercept.active is True
    assert {row.kind for row in selected_away} == {"coefficient", "level"}
    assert all(row.estimate == 0.0 for row in selected_away)
    assert all(row.active is False for row in selected_away)


def test_summary_export_does_not_make_nonfinite_selected_away_row_active():
    model = _fit_selected_away_export_model()
    assert tuple(model.result.rank_info.selected_group_names) == ()
    model.result.beta[0] = np.nan

    row = next(row for row in build_summary_export_payload(model).terms if row.term == "x")

    assert row.estimate is None
    assert row.active is False


def test_summary_export_labels_polynomial_categorical_tests_as_group_wald():
    payload = build_summary_export_payload(_fit_polynomial_categorical_export_model())

    rows = [row for row in payload.terms if row.group == "x:cat"]
    notes = "\n".join(payload.notes)

    assert len(rows) == 2
    assert all(row.kind == "group" for row in rows)
    assert all(row.estimate is None and row.std_error is None for row in rows)
    assert all(row.statistic_type == "chi2" for row in rows)
    assert all(isinstance(row.statistic, float) for row in rows)
    assert all(isinstance(row.p_value, float) for row in rows)
    assert "Group chi-square p-values are Wald approximations" in notes
    assert "Wood (2013)" not in notes


def test_summary_export_labels_large_polynomial_interaction_as_group_wald():
    payload = build_summary_export_payload(_fit_large_polynomial_interaction_export_model())

    row = next(row for row in payload.terms if row.term == "a:b")
    notes = "\n".join(payload.notes)

    assert row.kind == "group"
    assert row.estimate is None
    assert row.std_error is None
    assert row.statistic_type == "chi2"
    assert isinstance(row.statistic, float)
    assert isinstance(row.p_value, float)
    assert "Group chi-square p-values are Wald approximations" in notes
    assert "Wood (2013)" not in notes


def test_public_export_api_exists(tmp_path):
    model, X, y, w = _fit_export_model()
    output = tmp_path / "rating_tables.xlsx"

    payload = model.rating_table_payload(X, y, sample_weight=w)
    result_path = export_rating_tables(model, output, X, y, sample_weight=w)
    method_path = model.export_rating_tables(
        tmp_path / "rating_tables_method.xlsx", X, y, sample_weight=w
    )

    assert payload.base_relativity > 0.0
    assert result_path == output
    assert output.exists()
    assert method_path.exists()


def test_default_selected_bins_are_150():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    age_block = next(block for block in payload.main_effects if block.name == "age")

    assert payload.selected_n_bins == 150
    assert isinstance(payload.summary, SummaryExportPayload)
    assert not hasattr(payload, "summary_lines")
    assert len(age_block.table) <= 150
    assert {"age", "Relativity", "Weight"} <= set(age_block.table.columns)


def test_default_impact_sweep_bins():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    assert sorted(payload.discretization_impact["n_bins"].unique().tolist()) == [
        20,
        50,
        100,
        200,
        250,
    ]
    assert set(payload.discretization_impact["feature"]) == {"age"}


def test_categorical_and_numeric_blocks_are_exported():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    names = [block.name for block in payload.main_effects]
    assert names == ["age", "region", "score"]

    region = next(block for block in payload.main_effects if block.name == "region")
    score = next(block for block in payload.main_effects if block.name == "score")

    assert set(region.table["region"]) == {"A", "B", "C"}
    assert np.isclose(region.table["Weight"].sum(), w.sum())
    assert score.table["score"].tolist() == ["per_unit"]


def test_integer_categorical_block_weights_do_not_warn_on_pandas_integer_keys():
    rng = np.random.default_rng(124)
    n = 120
    X = pd.DataFrame({"region": rng.choice([1, 2, 3], n)})
    eta = -1.0 + 0.2 * (X["region"].to_numpy() == 2)
    y = rng.poisson(np.exp(eta)).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"region": Categorical(base="first")},
    )
    model.fit(X, y, sample_weight=w)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Series.__getitem__ treating keys as positions is deprecated",
            category=FutureWarning,
        )
        payload = build_rating_table_payload(model, X, y, sample_weight=w)

    region = next(block for block in payload.main_effects if block.name == "region")
    assert np.isclose(region.table["Weight"].sum(), w.sum())


def test_fit_offset_exports_exact_multiplier_block_when_support_is_small():
    model, X, y, w = _fit_offset_export_model(distinct_terms=2)

    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    offset_block = next(
        block for block in payload.main_effects if block.name == "Offset Multiplier"
    )
    assert offset_block.kind == "offset"
    assert offset_block.table.columns.tolist() == ["Offset Multiplier", "Relativity", "Weight"]
    assert sorted(offset_block.table["Offset Multiplier"].tolist()) == [1.0, 3.0]
    np.testing.assert_allclose(
        offset_block.table.sort_values("Offset Multiplier")["Relativity"].to_numpy(),
        [1.0, 3.0],
    )
    assert np.isclose(offset_block.table["Weight"].sum(), w.sum())


def test_fit_offset_exports_binned_multiplier_block_when_support_is_large():
    model, X, y, w = _fit_offset_export_model(distinct_terms=40)

    payload = build_rating_table_payload(model, X, y, sample_weight=w, n_bins=5)

    offset_block = next(
        block for block in payload.main_effects if block.name == "Offset Multiplier"
    )
    assert offset_block.kind == "offset"
    assert len(offset_block.table) == 5
    assert offset_block.table["Offset Multiplier"].str.startswith("[").all()
    assert np.isclose(offset_block.table["Weight"].sum(), w.sum())
    assert offset_block.table["Relativity"].between(1.0, 4.0).all()


def test_fit_offset_source_exports_raw_term_lookup():
    model, X, y, w, term, offset = _fit_term_offset_export_model()

    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=w,
        offset=offset,
        offset_source=term,
        offset_name="Term",
    )

    offset_block = next(block for block in payload.main_effects if block.name == "Term")
    assert offset_block.kind == "offset"
    assert offset_block.table.columns.tolist() == ["Term", "Relativity", "Weight"]
    table = offset_block.table.sort_values("Term")
    assert table["Term"].tolist() == [12.0, 36.0]
    np.testing.assert_allclose(table["Relativity"].to_numpy(), [1.0, 3.0])
    assert np.isclose(table["Weight"].sum(), w.sum())


def test_fit_offset_source_resolves_string_and_series_names():
    model, X, y, w, term, offset = _fit_term_offset_export_model()
    X_with_term = X.assign(term_months=term)

    from_column = build_rating_table_payload(
        model,
        X_with_term,
        y,
        sample_weight=w,
        offset=offset,
        offset_source="term_months",
    )
    from_series = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=w,
        offset=offset,
        offset_source=pd.Series(term, name="Policy Term"),
    )

    assert any(block.name == "term_months" for block in from_column.main_effects)
    assert any(block.name == "Policy Term" for block in from_series.main_effects)


def test_fit_offset_source_rejects_reserved_and_blank_names():
    model, X, y, w, term, offset = _fit_term_offset_export_model()

    for bad_name in ["", "   ", "Relativity", "Weight"]:
        with pytest.raises(ValueError, match="offset_name"):
            build_rating_table_payload(
                model,
                X,
                y,
                sample_weight=w,
                offset=offset,
                offset_source=term,
                offset_name=bad_name,
            )


def test_fit_offset_source_requires_name_for_unnamed_array():
    model, X, y, w, term, offset = _fit_term_offset_export_model()

    with pytest.raises(ValueError, match="offset_name is required"):
        build_rating_table_payload(
            model,
            X,
            y,
            sample_weight=w,
            offset=offset,
            offset_source=term,
        )


def test_fit_offset_source_rejects_inconsistent_mapping():
    model, X, y, w, term, offset = _fit_term_offset_export_model()
    bad_offset = offset.copy()
    bad_offset[term == 12.0] = 0.0
    bad_offset[np.flatnonzero(term == 12.0)[0]] = np.log(1.2)

    with pytest.raises(ValueError, match="maps to multiple offset multipliers"):
        build_rating_table_payload(
            model,
            X,
            y,
            sample_weight=w,
            offset=bad_offset,
            offset_source=term,
            offset_name="Term",
        )


def test_fit_offset_source_rejects_high_cardinality_source():
    model, X, y, w, term, offset = _fit_term_offset_export_model(distinct_terms=40)

    with pytest.raises(ValueError, match="exceeding offset_max_exact_levels=20"):
        build_rating_table_payload(
            model,
            X,
            y,
            sample_weight=w,
            offset=offset,
            offset_source=term,
            offset_name="Term",
        )


def test_fit_offset_source_rejects_missing_source_values():
    model, X, y, w, term, offset = _fit_term_offset_export_model()
    source = term.copy()
    source[0] = np.nan

    with pytest.raises(ValueError, match="offset_source cannot contain missing values"):
        build_rating_table_payload(
            model,
            X,
            y,
            sample_weight=w,
            offset=offset,
            offset_source=source,
            offset_name="Term",
        )


def test_fit_offset_source_allows_non_bijective_mapping():
    model, X, y, w, _term, offset = _fit_term_offset_export_model()
    source = np.resize(np.array([-1.0, 1.0]), len(X))
    shared_offset = np.zeros(len(X), dtype=np.float64)

    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=w,
        offset=shared_offset,
        offset_source=source,
        offset_name="Signed Source",
    )

    offset_block = next(block for block in payload.main_effects if block.name == "Signed Source")
    table = offset_block.table.sort_values("Signed Source")
    assert table["Signed Source"].tolist() == [-1.0, 1.0]
    np.testing.assert_allclose(table["Relativity"].to_numpy(), [1.0, 1.0])


def test_fit_offset_source_ignores_unused_categorical_levels():
    model, X, y, w, term, offset = _fit_term_offset_export_model()
    source = pd.Series(
        pd.Categorical(term, categories=[12.0, 36.0, 60.0]),
        name="Term",
    )

    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=w,
        offset=offset,
        offset_source=source,
    )

    offset_block = next(block for block in payload.main_effects if block.name == "Term")
    table = offset_block.table.sort_values("Term")
    assert table["Term"].tolist() == [12.0, 36.0]
    np.testing.assert_allclose(table["Relativity"].to_numpy(), [1.0, 3.0])


def test_fit_offset_source_reordered_frame_requires_explicit_offset():
    model, X, y, w, term, _offset = _fit_term_offset_export_model()
    order = np.arange(len(X))[::-1]

    with pytest.raises(ValueError, match="Pass offset="):
        build_rating_table_payload(
            model,
            X.iloc[order].reset_index(drop=True),
            y[order],
            sample_weight=w[order],
            offset_source=term[order],
            offset_name="Term",
        )


def test_fit_offset_source_reordered_frame_uses_aligned_offset():
    model, X, y, w, term, offset = _fit_term_offset_export_model()
    order = np.arange(len(X))[::-1]

    payload = build_rating_table_payload(
        model,
        X.iloc[order].reset_index(drop=True),
        y[order],
        sample_weight=w[order],
        offset=offset[order],
        offset_source=term[order],
        offset_name="Term",
    )

    offset_block = next(block for block in payload.main_effects if block.name == "Term")
    table = offset_block.table.sort_values("Term")
    assert table["Term"].tolist() == [12.0, 36.0]
    np.testing.assert_allclose(table["Relativity"].to_numpy(), [1.0, 3.0])


def test_fit_offset_export_rejects_offset_for_model_fit_without_offset():
    model, X, y, w = _fit_export_model()

    with pytest.raises(ValueError, match="requires a model fitted with an offset"):
        build_rating_table_payload(
            model,
            X,
            y,
            sample_weight=w,
            offset=np.zeros(len(X)),
            offset_source=np.ones(len(X)),
            offset_name="Term",
        )


def test_fit_offset_export_with_released_fit_state_requires_explicit_offset():
    model, X, y, w, term, _offset = _fit_term_offset_export_model(retain_fit_state=False)

    with pytest.raises(ValueError, match="Pass offset="):
        build_rating_table_payload(
            model,
            X,
            y,
            sample_weight=w,
            offset_source=term,
            offset_name="Term",
        )


def test_fit_offset_export_rejects_non_log_link_model():
    rng = np.random.default_rng(989)
    n = 80
    X = pd.DataFrame({"score": rng.normal(size=n)})
    offset = np.resize(np.array([0.0, 1.0]), n)
    y = 1.0 + 0.5 * X["score"].to_numpy() + offset
    model = SuperGLM(
        family="gaussian",
        link="identity",
        selection_penalty=0.0,
        features={"score": Numeric()},
    )
    model.fit(X, y, offset=offset)

    with pytest.raises(ValueError, match="log-link models"):
        build_rating_table_payload(
            model,
            X,
            y,
            offset=offset,
            offset_source=np.exp(offset),
            offset_name="Offset Source",
        )


def test_rating_table_payload_passes_offset_to_discretization_impact(monkeypatch):
    model, X, y, w = _fit_export_model()
    offset = np.full(len(X), np.log(2.0), dtype=np.float64)
    model.fit(X, y, sample_weight=w, offset=offset)
    seen_offsets: list[np.ndarray | None] = []

    def fake_discretization_impact(_X, _y, sample_weight=None, **kwargs):
        seen_offsets.append(kwargs.get("offset"))
        table = pd.DataFrame(
            {
                "bin_from": [18.0],
                "bin_to": [80.0],
                "relativity": [1.0],
                "log_relativity": [0.0],
                "n_obs": [len(_X)],
                "sample_weight": [float(np.sum(sample_weight))],
            }
        )
        return type(
            "FakeDiscretizationResult",
            (),
            {
                "tables": {"age": table},
                "metrics": {
                    "deviance_original": 1.0,
                    "deviance_discretized": 1.0,
                    "deviance_change": 0.0,
                    "deviance_change_pct": 0.0,
                    "mean_abs_prediction_change_pct": 0.0,
                    "max_abs_prediction_change_pct": 0.0,
                    "prediction_correlation": 1.0,
                },
            },
        )()

    monkeypatch.setattr(model, "discretization_impact", fake_discretization_impact)

    build_rating_table_payload(model, X, y, sample_weight=w, offset=offset, n_bins=1)

    assert len(seen_offsets) == 6
    for seen in seen_offsets:
        np.testing.assert_allclose(seen, offset)


def test_excel_workbook_layout(tmp_path):
    model, X, y, w = _fit_export_model()
    output = tmp_path / "tables.xlsx"
    expected_summary = build_summary_export_payload(model)

    model.export_rating_tables(output, X, y, sample_weight=w, n_bins=20)

    wb = load_workbook(output, data_only=True)
    assert wb.sheetnames == ["Rating Tables", "Discretization Impact", "Model Summary"]
    ws = wb["Rating Tables"]
    assert ws["A2"].value == "Base"
    assert isinstance(ws["C2"].value, float)
    assert ws["A5"].value == "age"
    assert ws["A7"].value == "age"
    assert ws["B7"].value == "Relativity"
    assert ws["C7"].value == "Weight"
    assert ws["D5"].value == "region"
    assert ws["D7"].value == "region"
    assert ws["G5"].value == "score"
    assert ws["G7"].value == "score"

    impact_ws = wb["Discretization Impact"]
    headers = [impact_ws.cell(row=1, column=i).value for i in range(1, 11)]
    assert headers[:3] == ["n_bins", "feature", "actual_bins"]

    summary_ws = wb["Model Summary"]
    assert summary_ws["A1"].value == "Model Summary"
    assert summary_ws["A1"].font.bold
    assert summary_ws["A1"].font.sz == pytest.approx(14.0)
    assert summary_ws["A3"].value == "Fit and model overview"
    assert [summary_ws.cell(row=4, column=column).value for column in range(1, 4)] == [
        "Section",
        "Metric",
        "Value",
    ]
    assert set(summary_ws.tables) == {"ModelOverview", "TermInference"}
    assert summary_ws.freeze_panes == "A5"

    overview_table = summary_ws.tables["ModelOverview"]
    overview_bounds = range_boundaries(overview_table.ref)
    term_table = summary_ws.tables["TermInference"]
    term_min_col, term_min_row, term_max_col, term_max_row = range_boundaries(term_table.ref)
    assert term_min_row == overview_bounds[3] + 3
    term_headers = [
        summary_ws.cell(row=term_min_row, column=column).value
        for column in range(term_min_col, term_max_col + 1)
    ]
    assert term_headers == [
        "Term",
        "Group",
        "Kind",
        "Estimate",
        "Std Error",
        "Statistic",
        "Statistic Type",
        "P Value",
        "CI Lower",
        "CI Upper",
        "EDF",
        "Lambda",
        "Active",
        "Significance",
        "Warning",
    ]

    overview = {row["Metric"]: row["Value"] for row in _table_records(summary_ws, "ModelOverview")}
    assert isinstance(overview["Observations"], int)
    assert not isinstance(overview["Observations"], bool)
    assert isinstance(overview["AIC"], float)
    assert isinstance(overview["Converged"], bool)

    terms = _table_records(summary_ws, "TermInference")
    assert [(row["Term"], row["Kind"]) for row in terms] == [
        (row.term, row.kind) for row in expected_summary.terms
    ]
    assert {row["Kind"] for row in terms} == {"coefficient", "level", "smooth"}
    assert summary_ws.cell(row=term_max_row + 3, column=1).value == "Notes"
    assert summary_ws.cell(row=term_max_row + 3, column=1).font.bold
    assert [
        summary_ws.cell(row=term_max_row + 4 + index, column=1).value
        for index in range(len(expected_summary.notes))
    ] == list(expected_summary.notes)
    assert all(
        "SuperGLM Results" not in str(cell.value) for row in summary_ws.iter_rows() for cell in row
    )
    assert summary_ws.column_dimensions["A"].width < 60


def test_ordered_spline_workbook_keeps_only_global_inference(tmp_path):
    model, levels = _fit_ordered_export_model()
    summary = build_summary_export_payload(model)
    output = tmp_path / "ordered_summary.xlsx"

    _write_workbook(_workbook_payload(summary), output)

    summary_ws = load_workbook(output, data_only=True)["Model Summary"]
    rows = [row for row in _table_records(summary_ws, "TermInference") if row["Group"] == "band"]
    smooth_rows = [row for row in rows if row["Kind"] == "smooth"]
    level_rows = [row for row in rows if row["Kind"] == "level"]

    assert len(smooth_rows) == 1
    assert isinstance(smooth_rows[0]["P Value"], float)
    assert len(level_rows) == len(levels)
    for row in level_rows:
        assert row["Estimate"] is not None
        assert row["Std Error"] is not None
        assert row["CI Lower"] is not None
        assert row["CI Upper"] is not None
        assert row["Statistic"] is None
        assert row["P Value"] is None
        assert row["Significance"] in (None, "")


def test_excel_workbook_writes_to_binary_stream():
    summary = SummaryExportPayload(
        overview=(SummaryOverviewRow("Fit", "Observations", 12),),
        terms=(
            SummaryTermRow(
                term="Intercept",
                group="",
                kind="coefficient",
                estimate=0.25,
                std_error=0.1,
                statistic=2.5,
                statistic_type="z",
                p_value=0.012,
                ci_lower=0.05,
                ci_upper=0.45,
                edf=None,
                smoothing_lambda=None,
                active=True,
                significance="*",
                warning="",
            ),
        ),
        notes=("Typed workbook stream.",),
    )
    target = BytesIO()

    _write_workbook(_workbook_payload(summary), target)

    assert not target.closed
    target.seek(0)
    wb = load_workbook(target, data_only=True)
    assert wb.sheetnames == ["Rating Tables", "Discretization Impact", "Model Summary"]
    assert set(wb["Model Summary"].tables) == {"ModelOverview", "TermInference"}


def test_excel_workbook_empty_terms_has_valid_blank_table_row():
    summary = SummaryExportPayload(
        overview=(SummaryOverviewRow("Fit", "Observations", 0),),
        terms=(),
        notes=(),
    )
    target = BytesIO()

    _write_workbook(_workbook_payload(summary), target)

    target.seek(0)
    summary_ws = load_workbook(target, data_only=True)["Model Summary"]
    table = summary_ws.tables["TermInference"]
    min_col, min_row, max_col, max_row = range_boundaries(table.ref)
    assert max_row == min_row + 1
    assert [
        summary_ws.cell(row=max_row, column=column).value for column in range(min_col, max_col + 1)
    ] == [None] * 15


def test_excel_workbook_includes_fit_offset_multiplier(tmp_path):
    model, X, y, w = _fit_offset_export_model(distinct_terms=2)
    output = tmp_path / "tables_with_offset.xlsx"

    model.export_rating_tables(output, X, y, sample_weight=w)

    ws = load_workbook(output, data_only=True)["Rating Tables"]
    assert ws["D5"].value == "Offset Multiplier"
    assert ws["D7"].value == "Offset Multiplier"
    assert ws["E7"].value == "Relativity"
    assert sorted([ws["D8"].value, ws["D9"].value]) == [1.0, 3.0]


def test_excel_workbook_includes_source_aware_fit_offset(tmp_path):
    model, X, y, w, term, offset = _fit_term_offset_export_model(distinct_terms=2)
    output = tmp_path / "tables_with_source_offset.xlsx"

    model.export_rating_tables(
        output,
        X,
        y,
        sample_weight=w,
        offset=offset,
        offset_source=term,
        offset_name="Term",
    )

    ws = load_workbook(output, data_only=True)["Rating Tables"]
    assert ws["D5"].value == "Term"
    assert ws["D7"].value == "Term"
    assert ws["E7"].value == "Relativity"
    assert sorted([ws["D8"].value, ws["D9"].value]) == [12.0, 36.0]
    rows = sorted(
        [(ws["D8"].value, ws["E8"].value), (ws["D9"].value, ws["E9"].value)],
        key=lambda row: row[0],
    )
    assert [row[0] for row in rows] == [12.0, 36.0]
    np.testing.assert_allclose([row[1] for row in rows], [1.0, 3.0])


def test_interactions_start_two_blank_rows_below_main_effects(tmp_path):
    rng = np.random.default_rng(321)
    n = 400
    X = pd.DataFrame(
        {
            "region": rng.choice(["A", "B", "C"], n),
            "type": rng.choice(["X", "Y"], n),
        }
    )
    y = rng.poisson(1.0 + 0.2 * (X["region"] == "B")).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"region": Categorical(base="first"), "type": Categorical(base="first")},
        interactions=[("region", "type")],
    )
    model.fit(X, y)
    output = tmp_path / "interaction.xlsx"

    model.export_rating_tables(output, X, y)

    ws = load_workbook(output, data_only=True)["Rating Tables"]
    main_last_row = (
        8
        + max(len(block.table) for block in build_rating_table_payload(model, X, y).main_effects)
        - 1
    )
    interaction_title_row = main_last_row + 3
    assert ws.cell(row=interaction_title_row - 1, column=1).value is None
    assert ws.cell(row=interaction_title_row, column=1).value == "region:type"


def test_continuous_interaction_export_uses_selected_bins():
    rng = np.random.default_rng(456)
    n = 500
    X = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "vehicle_age": rng.uniform(0, 20, n),
        }
    )
    eta = (
        -1.0
        + 0.08 * np.sin(X["age"].to_numpy() / 9.0)
        + 0.05 * np.cos(X["vehicle_age"].to_numpy() / 4.0)
        + 0.03 * (X["age"].to_numpy() - 45.0) * (X["vehicle_age"].to_numpy() - 8.0) / 100.0
    )
    y = rng.poisson(np.exp(eta)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "age": Spline(n_knots=7),
            "vehicle_age": Spline(n_knots=6),
        },
        interactions=[("age", "vehicle_age")],
    )
    model.fit(X, y)

    payload = build_rating_table_payload(model, X, y, n_bins=12)

    interaction = next(block for block in payload.interactions if block.name == "age:vehicle_age")
    assert interaction.table.shape == (12, 13)
    assert interaction.table.columns[0] == "age"
    assert np.isfinite(interaction.table.iloc[:, 1:].to_numpy(dtype=float)).all()


def test_export_rejects_unsupported_format(tmp_path):
    model, X, y, w = _fit_export_model()
    with pytest.raises(ValueError, match="Unsupported rating table export format"):
        model.export_rating_tables(tmp_path / "tables.csv", X, y, sample_weight=w)


def test_export_validates_lengths(tmp_path):
    model, X, y, w = _fit_export_model()
    with pytest.raises(ValueError, match="same length"):
        model.export_rating_tables(tmp_path / "tables.xlsx", X.iloc[:-1], y, sample_weight=w)
