"""Native-dataframe contracts for inference, diagnostics, and shape repair."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, Numeric, PSpline, Spline, SuperGLM
from superglm.debug_weights import compare_irls_weights, inspect_worst_observations
from superglm.stats.model_tests import dispersion_test, score_test_zi

pl = pytest.importorskip("polars")


def _to_polars(frame: pd.DataFrame):
    return pl.DataFrame({name: frame[name].to_numpy() for name in frame.columns})


@pytest.fixture
def diagnostic_case():
    rng = np.random.default_rng(20260720)
    n = 240
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.normal(size=n)
    category = rng.choice(["A", "B", "C"], size=n)
    eta = 0.15 + 0.25 * np.sin(1.7 * x) - 0.18 * z + 0.2 * (category == "B")
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    weights = rng.uniform(0.7, 1.4, n)
    X = pd.DataFrame({"x": x, "z": z, "category": category})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={
            "x": Spline(n_knots=6),
            "z": Numeric(),
            "category": Categorical(base="first"),
        },
        interactions=[("z", "category")],
    ).fit(X, y, sample_weight=weights)
    return model, X, _to_polars(X), y, weights


def test_polars_evaluation_metrics_are_chunked_without_whole_frame_conversion(
    diagnostic_case, monkeypatch
):
    from superglm._frame import EagerFrame
    from superglm.inference import _metrics_design

    model, X, _, y, weights = diagnostic_case
    X_eval = X.copy()
    X_eval["x"] += 0.05
    X_eval_pl = _to_polars(X_eval)

    monkeypatch.setattr(_metrics_design, "_MAX_DESIGN_CHUNK_ROWS", 17)
    pandas_metrics = model.metrics(X_eval, y, sample_weight=weights)
    pandas_leverage = pandas_metrics.leverage
    pandas_info = pandas_metrics._active_info

    original_block = _metrics_design._exact_runtime_design_block
    seen_chunks: list[int] = []

    def observed_block(model_arg, frame, selected_columns):
        assert isinstance(frame, EagerFrame)
        assert frame.backend == "polars"
        assert isinstance(frame.native, pl.DataFrame)
        seen_chunks.append(len(frame))
        return original_block(model_arg, frame, selected_columns)

    def unexpected_whole_frame_conversion(*_args, **_kwargs):
        pytest.fail("diagnostic evaluation converted the whole Polars frame")

    monkeypatch.setattr(_metrics_design, "_exact_runtime_design_block", observed_block)
    monkeypatch.setattr(pl.DataFrame, "to_numpy", unexpected_whole_frame_conversion)

    polars_metrics = model.metrics(X_eval_pl, y, sample_weight=weights)
    polars_leverage = polars_metrics.leverage
    polars_info = polars_metrics._active_info

    assert len(seen_chunks) > 1
    assert max(seen_chunks) <= 17
    np.testing.assert_allclose(polars_leverage, pandas_leverage, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(polars_info[2], pandas_info[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(polars_info[3], pandas_info[3], rtol=0.0, atol=0.0)


def test_polars_term_diagnostics_match_pandas(diagnostic_case):
    model, X, X_pl, y, weights = diagnostic_case

    pandas_importance = model.term_importance(X, sample_weight=weights)
    polars_importance = model.term_importance(X_pl, sample_weight=weights)
    pd.testing.assert_frame_equal(polars_importance, pandas_importance)

    pandas_holdout = model.term_drop_diagnostics(
        X,
        y,
        sample_weight=weights,
        mode="holdout",
        X_val=X,
        y_val=y,
    )
    polars_holdout = model.term_drop_diagnostics(
        X_pl,
        y,
        sample_weight=weights,
        mode="holdout",
        X_val=X_pl,
        y_val=y,
    )
    pd.testing.assert_frame_equal(polars_holdout, pandas_holdout)

    pandas_redundancy = model.spline_redundancy(X, sample_weight=weights)["x"]
    polars_redundancy = model.spline_redundancy(X_pl, sample_weight=weights)["x"]
    assert polars_redundancy.feature_name == pandas_redundancy.feature_name
    for field in (
        "knot_locations",
        "knot_spacing",
        "support_mass",
        "adjacent_basis_corr",
        "coef_energy_penalized",
        "small_singular_values",
    ):
        np.testing.assert_array_equal(
            getattr(polars_redundancy, field),
            getattr(pandas_redundancy, field),
        )
    assert polars_redundancy.effective_rank == pandas_redundancy.effective_rank


def test_polars_refit_diagnostics_match_pandas(diagnostic_case):
    model, X, X_pl, y, weights = diagnostic_case

    pandas_drop = model.drop1(X, y, sample_weight=weights)
    polars_drop = model.drop1(X_pl, y, sample_weight=weights)
    pd.testing.assert_frame_equal(polars_drop, pandas_drop)

    pandas_term_drop = model.term_drop_diagnostics(
        X,
        y,
        sample_weight=weights,
        mode="refit",
    )
    polars_term_drop = model.term_drop_diagnostics(
        X_pl,
        y,
        sample_weight=weights,
        mode="refit",
    )
    pd.testing.assert_frame_equal(polars_term_drop, pandas_term_drop)

    pandas_refit = model.refit_unpenalised(X, y, sample_weight=weights)
    polars_refit = model.refit_unpenalised(X_pl, y, sample_weight=weights)
    np.testing.assert_allclose(
        polars_refit.predict(X_pl),
        pandas_refit.predict(X),
        rtol=0.0,
        atol=0.0,
    )


def test_polars_discretization_impact_matches_pandas(diagnostic_case):
    model, X, X_pl, y, weights = diagnostic_case

    pandas_result = model.discretization_impact(X, y, sample_weight=weights, n_bins=12)
    polars_result = model.discretization_impact(X_pl, y, sample_weight=weights, n_bins=12)

    np.testing.assert_array_equal(polars_result.predictions, pandas_result.predictions)
    np.testing.assert_array_equal(
        polars_result.original_predictions,
        pandas_result.original_predictions,
    )
    assert polars_result.metrics == pandas_result.metrics
    assert polars_result.tables.keys() == pandas_result.tables.keys()
    for name in pandas_result.tables:
        pd.testing.assert_frame_equal(polars_result.tables[name], pandas_result.tables[name])


def test_polars_model_tests_match_pandas(diagnostic_case):
    model, X, X_pl, y, _ = diagnostic_case

    assert asdict(score_test_zi(model, X_pl, y)) == asdict(score_test_zi(model, X, y))
    assert asdict(dispersion_test(model, X_pl, y)) == asdict(dispersion_test(model, X, y))


def test_polars_plot_diagnostics_matches_pandas(diagnostic_case):
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    model, X, X_pl, y, weights = diagnostic_case
    pandas_figure = model.plot_diagnostics(X.copy(), y, sample_weight=weights, n_sim=3, seed=17)
    polars_figure = model.plot_diagnostics(X_pl, y, sample_weight=weights, n_sim=3, seed=17)

    pandas_axes = pandas_figure.get_axes()
    polars_axes = polars_figure.get_axes()
    assert len(polars_axes) == len(pandas_axes) == 4
    for pandas_axis, polars_axis in zip(pandas_axes, polars_axes, strict=True):
        assert polars_axis.get_title() == pandas_axis.get_title()
        assert len(polars_axis.lines) == len(pandas_axis.lines)
        assert len(polars_axis.collections) == len(pandas_axis.collections)
        assert len(polars_axis.patches) == len(pandas_axis.patches)
        for pandas_line, polars_line in zip(pandas_axis.lines, polars_axis.lines, strict=True):
            np.testing.assert_array_equal(polars_line.get_xdata(), pandas_line.get_xdata())
            np.testing.assert_array_equal(polars_line.get_ydata(), pandas_line.get_ydata())
        for pandas_patch, polars_patch in zip(
            pandas_axis.patches,
            polars_axis.patches,
            strict=True,
        ):
            assert polars_patch.get_x() == pandas_patch.get_x()
            assert polars_patch.get_y() == pandas_patch.get_y()
            assert polars_patch.get_width() == pandas_patch.get_width()
            assert polars_patch.get_height() == pandas_patch.get_height()

    plt.close(pandas_figure)
    plt.close(polars_figure)


@pytest.fixture
def debug_weight_case():
    x = np.linspace(-1.0, 1.0, 40)
    z = np.cos(x)
    X = pd.DataFrame({"x": x, "z": z})
    X_pl = _to_polars(X)
    y = 1.5 + 0.4 * x - 0.2 * z
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    ).fit(X, y, record_diagnostics=True)
    return model, X, X_pl, y


def test_polars_worst_observation_report_matches_pandas(debug_weight_case):
    model, X, X_pl, y = debug_weight_case

    pandas_worst = inspect_worst_observations(model, X, y)
    polars_worst = inspect_worst_observations(model, X_pl, y)
    pd.testing.assert_frame_equal(polars_worst, pandas_worst)


def test_polars_irls_weight_comparison_matches_pandas(debug_weight_case):
    pytest.importorskip("statsmodels")
    model, X, X_pl, y = debug_weight_case

    pandas_comparison = compare_irls_weights(model, X, y, max_iter=1)
    polars_comparison = compare_irls_weights(model, X_pl, y, max_iter=1)
    pd.testing.assert_frame_equal(polars_comparison, pandas_comparison)


def _shape_case():
    rng = np.random.default_rng(20260720)
    x = np.linspace(0.0, 1.0, 180)
    y = -((x - 0.4) ** 2) + 0.03 * rng.normal(size=x.size)
    X = pd.DataFrame({"x": x})
    return X, _to_polars(X), y


def _shape_model():
    return SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=9, constraint=Constraint.postfit.convex)},
    )


def test_polars_shape_repair_matches_pandas():
    X, X_pl, y = _shape_case()
    pandas_model = _shape_model().fit(X, y)
    polars_model = _shape_model().fit(X_pl, y)

    pandas_model.apply_shape_postfit(X)
    polars_model.apply_shape_postfit(X_pl)

    np.testing.assert_allclose(
        polars_model.result.beta, pandas_model.result.beta, rtol=0.0, atol=0.0
    )
    assert polars_model.result.intercept == pandas_model.result.intercept
    np.testing.assert_allclose(
        polars_model.predict(X_pl),
        pandas_model.predict(X),
        rtol=0.0,
        atol=0.0,
    )
    assert polars_model._shape_repairs["x"].max_violation_after == (
        pandas_model._shape_repairs["x"].max_violation_after
    )


def test_failed_polars_shape_repair_rolls_back(monkeypatch):
    X, X_pl, y = _shape_case()
    model = _shape_model().fit(X_pl, y)
    original_dict = model.__dict__
    original_result = model.result
    original_beta = model.result.beta.copy()
    original_intercept = model.result.intercept
    original_revision = model._fit_revision
    original_predictions = model.predict(X_pl)
    original_metrics = model.metrics(X_pl, y)
    model.summary()
    original_summary_cache = model._summary_cache
    original_covariance = model._coef_covariance[0].copy()
    original_nb_profile = model._nb_profile_result
    original_tweedie_profile = model._tweedie_profile_result

    class FailingRepairer:
        def repair(self, *_args, **_kwargs):
            raise RuntimeError("injected Polars repair failure")

    from superglm.model import shape_ops

    monkeypatch.setattr(shape_ops, "_repairer", lambda _kind: FailingRepairer())

    with pytest.raises(RuntimeError, match="injected Polars repair failure"):
        model.apply_shape_postfit(X_pl)

    assert model.__dict__ is original_dict
    assert model.result is original_result
    assert model.result.intercept == original_intercept
    assert model._fit_revision == original_revision
    assert model._fit_metrics_cache is original_metrics
    assert model._summary_cache is original_summary_cache
    assert model._nb_profile_result is original_nb_profile
    assert model._tweedie_profile_result is original_tweedie_profile
    np.testing.assert_array_equal(model.result.beta, original_beta)
    np.testing.assert_array_equal(model._coef_covariance[0], original_covariance)
    np.testing.assert_array_equal(model.predict(X_pl), original_predictions)
