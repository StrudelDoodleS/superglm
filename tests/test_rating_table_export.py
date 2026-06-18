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
