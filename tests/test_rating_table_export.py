import numpy as np
import pandas as pd

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
