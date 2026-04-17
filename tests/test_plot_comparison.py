"""Tests for Plotly term comparison utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM


@pytest.fixture
def comparison_data():
    rng = np.random.default_rng(42)
    n = 300
    veh_age = rng.uniform(0, 20, n)
    bonus_levels = ["50-60", "60-70", "70-80", "80-100", "100+"]
    bonus = rng.choice(bonus_levels, n, p=[0.2, 0.25, 0.25, 0.2, 0.1])
    sample_weight = rng.uniform(0.4, 1.0, n)
    bonus_effect = {"50-60": 0.0, "60-70": 0.08, "70-80": 0.14, "80-100": 0.2, "100+": 0.3}
    eta = -1.8 + 0.02 * np.sin(veh_age / 3.0) + np.array([bonus_effect[b] for b in bonus])
    y = rng.poisson(np.exp(eta) * sample_weight).astype(float)
    X = pd.DataFrame({"VehAge": veh_age, "BonusBand": bonus})
    return X, y, sample_weight


@pytest.fixture
def fitted_comparison_models(comparison_data):
    X, y, sample_weight = comparison_data
    ordered = SuperGLM(
        features={
            "VehAge": Spline(n_knots=8),
            "BonusBand": OrderedCategorical(order=["50-60", "60-70", "70-80", "80-100", "100+"]),
        }
    )
    ordered.fit(X, y, sample_weight=sample_weight)

    categorical = SuperGLM(
        features={
            "VehAge": Spline(n_knots=8),
            "BonusBand": Categorical(base="first"),
        }
    )
    categorical.fit(X, y, sample_weight=sample_weight)
    return X, sample_weight, {"ordered": ordered, "categorical": categorical}


def test_resolve_comparable_terms_allows_categorical_and_ordered_categorical(
    fitted_comparison_models,
):
    from superglm.plotting.comparison import _resolve_comparable_terms

    _, _, models = fitted_comparison_models
    terms, skipped = _resolve_comparable_terms(models, terms=None)

    assert "VehAge" in terms
    assert "BonusBand" in terms
    assert skipped == {}


def test_build_term_comparison_data_uses_shared_domains(fitted_comparison_models):
    from superglm.plotting.comparison import _build_term_comparison_data

    X, sample_weight, models = fitted_comparison_models
    payload = _build_term_comparison_data(
        models=models,
        terms=["VehAge", "BonusBand"],
        X=X,
        sample_weight=sample_weight,
        n_points=41,
    )

    veh_age = payload["terms"][0]
    bonus = payload["terms"][1]
    assert veh_age["family"] == "continuous"
    assert len(veh_age["domain"]["x"]) == 41
    assert set(veh_age["series"]) == {"ordered", "categorical"}
    assert bonus["family"] == "level"
    assert list(bonus["domain"]["levels"]) == ["50-60", "60-70", "70-80", "80-100", "100+"]


def test_build_term_comparison_data_can_store_per_label_support(fitted_comparison_models):
    from superglm.plotting.comparison import _build_term_comparison_data

    X, sample_weight, models = fitted_comparison_models
    support_by_label = {
        "ordered": {"X": X.iloc[:120], "sample_weight": sample_weight[:120]},
        "categorical": {"X": X.iloc[120:], "sample_weight": sample_weight[120:]},
    }
    payload = _build_term_comparison_data(
        models=models,
        terms=["VehAge", "BonusBand"],
        X=X,
        sample_weight=sample_weight,
        n_points=41,
        support_by_label=support_by_label,
    )

    veh_age_support = payload["terms"][0]["support"]
    bonus_support = payload["terms"][1]["support"]
    assert veh_age_support["mode"] == "by_label"
    assert set(veh_age_support["series"]) == {"ordered", "categorical"}
    assert len(veh_age_support["series"]["ordered"]["x"]) == 41
    assert bonus_support["mode"] == "by_label"
    assert set(bonus_support["series"]) == {"ordered", "categorical"}
    assert len(bonus_support["series"]["ordered"]["density"]) == 5


def test_plot_term_comparison_builds_dropdown_and_scale_toggle(fitted_comparison_models):
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    from superglm import plot_term_comparison

    X, sample_weight, models = fitted_comparison_models
    fig = plot_term_comparison(
        models=models,
        X=X,
        sample_weight=sample_weight,
        terms=None,
        engine="plotly",
        n_points=31,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.updatemenus) == 2
    assert fig.layout.updatemenus[0].type == "buttons"
    assert fig.layout.updatemenus[1].type == "dropdown"


def test_plot_term_comparison_uses_one_trace_per_label(fitted_comparison_models):
    pytest.importorskip("plotly")
    from superglm import plot_term_comparison

    X, sample_weight, models = fitted_comparison_models
    fig = plot_term_comparison(
        models=models,
        X=X,
        sample_weight=sample_weight,
        terms=["VehAge"],
        engine="plotly",
        n_points=31,
    )

    top_traces = [t for t in fig.data if getattr(t, "xaxis", None) == "x"]
    labels = {t.name for t in top_traces if t.name in models}
    assert labels == {"ordered", "categorical"}


def test_build_term_comparison_data_uses_full_grid_support_for_continuous(
    fitted_comparison_models,
):
    from superglm.plotting.comparison import _build_term_comparison_data

    X, sample_weight, models = fitted_comparison_models
    payload = _build_term_comparison_data(
        models=models,
        terms=["VehAge"],
        X=X,
        sample_weight=sample_weight,
        n_points=61,
    )

    support = payload["terms"][0]["support"]
    assert len(support["x"]) == 61
    assert len(support["density"]) == 61


def test_build_term_comparison_data_keeps_raw_level_exposure(fitted_comparison_models):
    from superglm.plotting.comparison import _build_term_comparison_data

    X, sample_weight, models = fitted_comparison_models
    payload = _build_term_comparison_data(
        models=models,
        terms=["BonusBand"],
        X=X,
        sample_weight=sample_weight,
        n_points=31,
    )

    support = payload["terms"][0]["support"]
    grouped = (
        pd.DataFrame({"level": X["BonusBand"].astype(str), "sample_weight": sample_weight})
        .groupby("level", sort=False)["sample_weight"]
        .sum()
    )
    expected = [float(grouped[level]) for level in support["levels"]]
    np.testing.assert_allclose(support["density"], expected)


def test_plot_term_comparison_support_trace_matches_main_effect_defaults(
    fitted_comparison_models,
):
    pytest.importorskip("plotly")
    from superglm import plot_term_comparison
    from superglm.plotting.main_effects_plotly import _resolve_plotly_style

    X, sample_weight, models = fitted_comparison_models
    fig = plot_term_comparison(
        models=models,
        X=X,
        sample_weight=sample_weight,
        terms=["VehAge"],
        engine="plotly",
        n_points=61,
    )

    support_trace = next(t for t in fig.data if t.name == "Exposure density")
    style = _resolve_plotly_style(None)
    assert len(support_trace.x) == 61
    assert support_trace.line.color == style["density_edge_color"]
