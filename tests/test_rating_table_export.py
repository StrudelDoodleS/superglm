import warnings

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from superglm import Categorical, Numeric, Spline, SuperGLM, export_rating_tables
from superglm.export.rating_tables import build_rating_table_payload


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


def _score_payload_row(payload, row: dict, *, sql_exposure: float) -> float:
    score = float(payload.base_relativity) * float(sql_exposure)
    for block in payload.main_effects:
        key_col = block.table.columns[0]
        if key_col not in row:
            continue
        matches = block.table[block.table[key_col] == row[key_col]]
        if matches.empty:
            raise AssertionError(f"No exported row for {key_col}={row[key_col]!r}")
        score *= float(matches["Relativity"].iloc[0])
    return score


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


def test_exported_factor_offset_scoring_rule_uses_unit_sql_exposure():
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

    assert payload.offset_scoring_rule == "EXPORTED_FACTOR"
    assert any(block.name == "Term" for block in payload.main_effects)

    expected = model.predict(X.iloc[:6].reset_index(drop=True), offset=offset[:6])
    scored = np.array(
        [
            _score_payload_row(
                payload,
                {"region": X.iloc[i]["region"], "Term": term[i]},
                sql_exposure=1.0,
            )
            for i in range(6)
        ]
    )
    np.testing.assert_allclose(scored, expected, rtol=1e-10, atol=1e-12)


def test_already_applied_sql_exposure_rule_omits_offset_block_and_uses_exp_offset():
    model, X, y, w, _term, offset = _fit_term_offset_export_model()

    payload = build_rating_table_payload(
        model,
        X,
        y,
        sample_weight=w,
        offset=offset,
        offset_scoring_rule="ALREADY_APPLIED_SQL_EXPOSURE",
    )

    assert payload.offset_scoring_rule == "ALREADY_APPLIED_SQL_EXPOSURE"
    assert not any(block.kind == "offset" for block in payload.main_effects)

    expected = model.predict(X.iloc[:6].reset_index(drop=True), offset=offset[:6])
    scored = np.array(
        [
            _score_payload_row(
                payload,
                {"region": X.iloc[i]["region"]},
                sql_exposure=float(np.exp(offset[i])),
            )
            for i in range(6)
        ]
    )
    np.testing.assert_allclose(scored, expected, rtol=1e-10, atol=1e-12)


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
    assert ws["A3"].value == "SQL Offset Scoring Rule"
    assert ws["C3"].value == "EXPORTED_FACTOR"
    assert ws["A4"].value == "SQL Exposure Input"
    assert "@exposure = 1.0" in ws["C4"].value
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
