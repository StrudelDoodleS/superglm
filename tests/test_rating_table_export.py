import numpy as np
import pandas as pd
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


def test_public_export_api_exists(tmp_path):
    model, X, y, w = _fit_export_model()
    output = tmp_path / "rating_tables.xlsx"

    result_path = export_rating_tables(model, output, X, y, sample_weight=w)
    method_path = model.export_rating_tables(
        tmp_path / "rating_tables_method.xlsx", X, y, sample_weight=w
    )

    assert result_path == output
    assert output.exists()
    assert method_path.exists()


def test_default_selected_bins_are_150():
    model, X, y, w = _fit_export_model()
    payload = build_rating_table_payload(model, X, y, sample_weight=w)

    age_block = next(block for block in payload.main_effects if block.name == "age")

    assert payload.selected_n_bins == 150
    assert len(age_block.table) <= 150
    assert {"Level", "Relativity", "Weight"} <= set(age_block.table.columns)


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

    assert set(region.table["Level"]) == {"A", "B", "C"}
    assert np.isclose(region.table["Weight"].sum(), w.sum())
    assert score.table["Level"].tolist() == ["per_unit"]


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
    assert ws["A7"].value == "Level"
    assert ws["B7"].value == "Relativity"
    assert ws["C7"].value == "Weight"
    assert ws["D5"].value == "region"
    assert ws["G5"].value == "score"

    impact_ws = wb["Discretization Impact"]
    headers = [impact_ws.cell(row=1, column=i).value for i in range(1, 11)]
    assert headers[:3] == ["n_bins", "feature", "actual_bins"]


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
