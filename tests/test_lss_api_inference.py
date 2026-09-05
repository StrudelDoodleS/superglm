"""Contract tests for the ``SuperLSS`` inference façade.

Every method under test is a delegation: what is asserted here is that it
reaches the right builder, on the right rows, with the weight in the right
slot, and returns that builder's payload.  Whether the statistics inside are
right is the business of the builders' own test files.
"""

from __future__ import annotations

import inspect
import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import special

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from superglm import Categorical, Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.checks.binned import BinnedCheck, BinnedCheck2D, binned_check
from superglm.distributional.checks.calibration import (
    ActualExpected,
    CalibrationPayload,
    actual_expected_check,
)
from superglm.distributional.checks.compare import Comparison, compare_models
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.family import PriorWeightedVarianceFamily, VarianceFamily
from superglm.distributional.posterior import PosteriorDraws, posterior_bounds, posterior_predictive
from superglm.distributional.residuals import ResidualSet, compute_residuals
from superglm.distributional.surfaces import DensityFan, Portfolio, RiskCurves, Spread, portfolio
from superglm.distributional.terms import ParameterTermEffect, TermTest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def case():
    """A seeded Gaussian location-scale fit with a numeric and a level covariate."""
    rng = np.random.default_rng(20260903)
    n = 800
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    frame = pd.DataFrame({"x": x, "g": g})
    location = 0.6 * np.sin(2.4 * x) + np.where(g == "a", 0.3, np.where(g == "b", -0.2, 0.0))
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    y = location + scale * rng.standard_normal(n)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=6), "g": Categorical()}),
            Predictor("scale", {"x": Spline("cr", k=5)}),
        ],
    ).fit_reml(frame, y)
    return model, frame, y


@pytest.fixture(scope="module")
def flat_model(case):
    """A comparison candidate with a constant scale."""
    _, frame, y = case
    return SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=6)}),
            Predictor("scale", {}),
        ],
    ).fit_reml(frame, y)


def _weighted_frame(rng, n):
    x = rng.uniform(-1.0, 1.0, n)
    return pd.DataFrame({"x": x}), x, rng.uniform(0.2, 1.0, n)


@pytest.fixture(scope="module")
def weighted_fits():
    """One prior-weighted fit per family that knows its closed-form variance."""
    rng = np.random.default_rng(4242)
    fits: dict[str, tuple] = {}

    frame, x, weights = _weighted_frame(rng, 240)
    mean = np.exp(0.5 + 0.4 * np.sin(2.0 * x))
    cv = np.exp(-0.6 + 0.2 * x)
    y = rng.gamma(weights / cv**2, mean * cv**2 / weights)
    fits["gamma"] = (
        SuperLSS(
            family=GammaLS(),
            predictors=[Predictor("mean", {"x": Spline("cr", k=5)}), Predictor("scale", {})],
        ).fit_reml(frame, y, weights),
        frame,
        y,
        weights,
    )

    location = 0.4 * np.sin(2.0 * x)
    sigma = np.exp(-0.5 + 0.3 * x)
    y = location + sigma / np.sqrt(weights) * rng.standard_normal(len(x))
    fits["gaussian"] = (
        SuperLSS(
            family=GaussianLS(),
            predictors=[Predictor("location", {"x": Spline("cr", k=5)}), Predictor("scale", {})],
        ).fit_reml(frame, y, weights),
        frame,
        y,
        weights,
    )

    power = 1.6
    mu = np.exp(0.3 + 0.4 * np.sin(2.0 * x))
    phi = 0.8
    rate = weights * mu ** (2.0 - power) / (phi * (2.0 - power))
    shape = (2.0 - power) / (power - 1.0)
    scale = phi * (power - 1.0) * mu ** (power - 1.0) / weights
    counts = rng.poisson(rate)
    y = np.where(counts > 0, rng.gamma(shape * np.maximum(counts, 1), scale), 0.0)
    fits["tweedie"] = (
        SuperLSS(
            family=TweedieLSS(),
            predictors=[
                Predictor("mean", {"x": Spline("cr", k=5)}),
                Predictor("dispersion", {}),
                Predictor("power", {}),
            ],
        ).fit_reml(frame, y, weights),
        frame,
        y,
        weights,
    )
    return fits


# --------------------------------------------------------------------------- #
# Residuals and checks
# --------------------------------------------------------------------------- #


def test_an_unfitted_model_refuses_every_inference_method():
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[Predictor("location", {}), Predictor("scale", {})],
    )
    frame = pd.DataFrame({"x": [0.1, 0.2, 0.3]})
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_parameters(frame)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.residuals(frame, np.zeros(3))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.summary()


def test_residuals_return_the_builder_array(case):
    model, frame, y = case
    fitted = model._require_fitted()
    expected = compute_residuals(fitted, frame, y, seed=7)
    assert np.array_equal(model.residuals(frame, y, seed=7), expected.quantile)
    assert np.array_equal(model.residuals(frame, y, kind="pit", seed=7), expected.pit)
    payload = model.residual_set(frame, y, seed=7)
    assert isinstance(payload, ResidualSet)
    assert np.array_equal(payload.pit, expected.pit)


def test_check_accepts_a_column_name_or_an_array(case):
    model, frame, y = case
    named = model.check(frame, y, "x")
    assert isinstance(named, BinnedCheck)
    assert named.covariate == "x"
    given = model.check(frame, y, frame["x"].to_numpy(), name="x")
    assert np.array_equal(named.mean, given.mean)
    fitted = model._require_fitted()
    expected = binned_check(compute_residuals(fitted, frame, y), frame["x"].to_numpy(), name="x")
    assert np.array_equal(named.mean, expected.mean)


def test_check_2d_returns_the_two_dimensional_payload(case):
    model, frame, y = case
    payload = model.check_2d(frame, y, "x", frame["x"].to_numpy() ** 2, names=("x", "x squared"))
    assert isinstance(payload, BinnedCheck2D)
    assert int(payload.count.sum()) == len(frame)


def test_a_zero_weight_row_leaves_the_covariate_too(case):
    """A zero weight drops a row from the residuals, so the covariate is cut with it."""
    model, frame, y = case
    weights = np.ones(len(frame))
    weights[3] = 0.0
    checked = model.check(frame, y, "x", sample_weight=weights, n_bins=5, n_boot=8)
    assert int(checked.n.sum()) == len(frame) - 1
    pair = model.check_2d(
        frame, y, "x", frame["x"].to_numpy() ** 2, sample_weight=weights, n_bins=(4, 4)
    )
    assert int(pair.count.sum()) == len(frame) - 1
    figure = model.plot_diagnostics(frame, y, sample_weight=weights, n_sim=4)
    assert len(figure.axes) == 6
    worm = model.plot_data("worm", X=frame, y=y, covariate="x", sample_weight=weights)
    assert sum(panel["n"] for panel in worm["panels"]) == len(frame) - 1


def test_a_covariate_must_name_a_column_or_match_the_rows(case):
    model, frame, y = case
    with pytest.raises(ValueError, match="one value per row"):
        model.check(frame, y, np.zeros(3))
    with pytest.raises(ValueError, match="not a column of X"):
        model.check(frame, y, "nope")


def test_actual_expected_returns_the_ratio_table(case):
    model, frame, y = case
    table = model.actual_expected(frame, y, "x", n_bins=8)
    assert isinstance(table, ActualExpected)
    assert table.covariate == "x"
    assert len(table.ratio) == 8


def test_calibration_returns_the_four_tables(case):
    model, frame, y = case
    payload = model.calibration(frame, y, levels=(0.5, 0.9), quantile_grid=(0.1, 0.9))
    assert isinstance(payload, CalibrationPayload)
    assert set(payload.coverage["level"]) == {0.5, 0.9}


def test_scores_returns_one_column_per_score(case):
    model, frame, y = case
    table = model.scores(frame, y, which=("log", "crps"))
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["log", "crps"]
    assert len(table) == len(frame)


def test_gamma_prior_crps_uses_cv_over_sqrt_weight_through_the_public_facade(weighted_fits):
    model, frame, y, weights = weighted_fits["gamma"]
    rows = frame.head(12)
    response = y[:12]
    law = weights[:12]
    theta = np.asarray(model.predict_parameters(rows), dtype=np.float64)
    mean = theta[:, 0]
    weighted_cv = theta[:, 1] / np.sqrt(law)
    shape = 1.0 / weighted_cv**2
    scale = mean * weighted_cv**2
    expected = (
        response * (2.0 * special.gammainc(shape, response / scale) - 1.0)
        - shape * scale * (2.0 * special.gammainc(shape + 1.0, response / scale) - 1.0)
        - scale / special.beta(0.5, shape)
    )

    closed = model.scores(
        rows,
        response,
        which=("crps",),
        sample_weight=law,
        method="closed",
    )["crps"].to_numpy()
    numeric = model.scores(
        rows,
        response,
        which=("crps",),
        sample_weight=law,
        method="numeric",
        n_nodes=64,
    )["crps"].to_numpy()

    assert np.allclose(closed, expected, rtol=1.0e-13, atol=1.0e-13)
    assert np.allclose(numeric, expected, rtol=1.0e-6, atol=1.0e-10)

    # A Gamma-specific cv / w mutation must not satisfy the independent law.
    mutated_cv = theta[:, 1] / law
    mutated_shape = 1.0 / mutated_cv**2
    mutated_scale = mean * mutated_cv**2
    mutated = (
        response * (2.0 * special.gammainc(mutated_shape, response / mutated_scale) - 1.0)
        - mutated_shape
        * mutated_scale
        * (2.0 * special.gammainc(mutated_shape + 1.0, response / mutated_scale) - 1.0)
        - mutated_scale / special.beta(0.5, mutated_shape)
    )
    assert not np.allclose(closed, mutated, rtol=1.0e-6, atol=1.0e-10)


def test_compare_scores_two_candidates(case, flat_model):
    model, frame, y = case
    payload = model.compare(flat_model, frame, y, which="log")
    assert isinstance(payload, Comparison)
    assert payload.overall["n"] == len(frame)


def test_compare_explicitly_delegates_sample_weight(case, flat_model):
    """Dropping the named façade argument would hide this public contract."""
    model, frame, y = case
    weights = np.ones(len(frame))
    weights[17] = 0.0

    parameter = inspect.signature(SuperLSS.compare).parameters["sample_weight"]
    assert parameter.default is None
    actual = model.compare(flat_model, frame, y, which="log", sample_weight=weights)
    expected = compare_models(
        model._require_fitted(),
        flat_model._require_fitted(),
        frame,
        y,
        which="log",
        sample_weight=weights,
    )

    assert actual.overall == expected.overall
    assert actual.overall["n"] == len(frame) - 1


# --------------------------------------------------------------------------- #
# Term inference and summary
# --------------------------------------------------------------------------- #


def test_term_inference_uses_the_retained_training_frame(case):
    model, frame, _ = case
    effect = model.term_inference("scale", "x", n_points=40, n_sim=200)
    assert isinstance(effect, ParameterTermEffect)
    assert len(effect.effect) == 40
    assert effect.multiplier is not None
    outcome = model.term_test("location", "x")
    assert isinstance(outcome, TermTest)
    assert outcome.p_value < 1.0e-3


def test_summary_lists_every_intercept_and_term(case):
    model, frame, _ = case
    table = model.summary()
    assert list(table.columns) == [
        "parameter",
        "term",
        "edf",
        "lambda",
        "statistic",
        "rank",
        "p_value",
        "estimate",
        "se",
        "note",
    ]
    assert set(table["parameter"]) == {"location", "scale"}
    assert set(table["note"]) == {""}
    assert {"x", "g"} <= set(table["term"])


def test_a_restored_model_needs_the_training_frame(case):
    model, frame, _ = case
    restored = SuperLSS.from_bytes(model.to_bytes())
    with pytest.raises(RuntimeError, match="X_train"):
        restored.summary()
    with pytest.raises(RuntimeError, match="X_train"):
        restored.term_inference("location", "x")
    table = restored.summary(X_train=frame)
    assert table.equals(model.summary())


# --------------------------------------------------------------------------- #
# Posterior primitive and surfaces
# --------------------------------------------------------------------------- #


def test_posterior_methods_return_the_primitive_payloads(case):
    model, frame, _ = case
    draws = model.posterior_draws(64, seed=3)
    assert isinstance(draws, PosteriorDraws)
    assert draws.n_draws == 64
    bounds = model.posterior_bounds(frame.head(20), ("quantile", 0.9), n_draws=64, seed=3)
    assert list(bounds.columns) == ["estimate", "mean", "sd", "lower", "upper"]
    simulated = model.posterior_predictive(frame.head(20), 16, seed=3)
    assert simulated.shape == (16, 20)


def test_risk_curves_and_density_fan_sweep_a_covariate(case):
    model, frame, _ = case
    curves = model.risk_curves({"g": "a"}, "x", n_points=12, n_draws=64, quantiles=(0.5, 0.9))
    assert isinstance(curves, RiskCurves)
    assert curves.values.shape == (2, 12)
    fan = model.density_fan({"g": "a"}, "x", n_points=6, n_y=40)
    assert isinstance(fan, DensityFan)
    assert fan.density.shape == (6, 40)


def test_spread_and_portfolio_summarise_a_book(case):
    model, frame, y = case
    spread = model.parameter_spread(frame, threshold=float(np.quantile(y, 0.9)), n_bins=5)
    assert isinstance(spread, Spread)
    assert len(spread.identically_priced) == 5
    book = model.portfolio(frame, n_draws=32, by="g", seed=5)
    assert isinstance(book, Portfolio)
    assert len(book.by_segment) == 3


# --------------------------------------------------------------------------- #
# Figures and payload dictionaries
# --------------------------------------------------------------------------- #


def test_plot_diagnostics_draws_the_six_panels(case):
    model, frame, y = case
    figure = model.plot_diagnostics(frame, y, n_sim=8)
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 6


def test_plot_returns_one_figure_per_parameter(case):
    model, frame, _ = case
    single = model.plot(parameter="scale", n_sim=64)
    assert isinstance(single, Figure)
    every = model.plot(n_sim=64)
    assert set(every) == {"location", "scale"}
    assert all(isinstance(figure, Figure) for figure in every.values())
    named = model.plot(parameter="location", terms="x", n_sim=64)
    assert isinstance(named, Figure)


def test_plot_refuses_a_parameter_with_no_plottable_term(case, flat_model):
    _, frame, y = case
    with pytest.raises(ValueError, match="one-dimensional effect grid"):
        flat_model.plot(parameter="scale", n_sim=64)
    only_location = flat_model.plot(n_sim=64)
    assert set(only_location) == {"location"}
    flat = SuperLSS(
        family=GaussianLS(),
        predictors=[Predictor("location", {}), Predictor("scale", {})],
    ).fit_reml(frame, y)
    with pytest.raises(ValueError, match="no parameter has a term"):
        flat.plot()
    with pytest.raises(ValueError, match="among the terms asked for"):
        flat_model.plot(parameter="location", terms="nope", n_sim=64)


def _plot_data_arguments(frame, y, other):
    return {
        "qq": {"X": frame, "y": y, "n_sim": 8},
        "worm": {"X": frame, "y": y, "covariate": "x"},
        "pit": {"X": frame, "y": y},
        "binned": {"X": frame, "y": y, "covariate": "x", "n_boot": 16},
        "actual_expected": {"X": frame, "y": y, "covariate": "x", "n_bins": 6},
        "calibration": {"X": frame, "y": y, "levels": (0.9,), "quantile_grid": (0.5,)},
        "scores": {"X": frame, "y": y, "which": ("log",)},
        "comparison": {"other": other, "X": frame, "y": y},
        "term": {"parameter": "location", "term": "x", "n_points": 20, "n_sim": 64},
        "risk_curves": {
            "reference": {"g": "a"},
            "covariate": "x",
            "n_points": 6,
            "n_draws": 32,
            "quantiles": (0.9,),
        },
        "density_fan": {"reference": {"g": "a"}, "covariate": "x", "n_points": 4, "n_y": 20},
        "spread": {"X": frame, "threshold": float(np.quantile(y, 0.9)), "n_bins": 4},
        "portfolio": {"X": frame, "n_draws": 16},
    }


def test_plot_data_is_json_clean_for_every_kind(case, flat_model):
    model, frame, y = case
    arguments = _plot_data_arguments(frame, y, flat_model)
    for kind, keywords in arguments.items():
        payload = model.plot_data(kind, **keywords)
        assert json.loads(json.dumps(payload)) == payload, kind
    whole = model.plot_data("worm", X=frame, y=y)
    assert len(whole["panels"]) == 1 and whole["covariate"] is None
    records = model.plot_data("scores", X=frame, y=y, which=("log",))
    assert isinstance(records, list)
    assert len(records) == len(frame)


def test_plot_data_names_the_kinds_it_knows(case):
    model, frame, y = case
    with pytest.raises(ValueError, match="kind must be one of"):
        model.plot_data("nonsense", X=frame, y=y)


def test_the_plotting_package_exports_both_engines():
    """matplotlib names import with the base install; plotly names need the extra."""
    import superglm.plotting as plotting
    from superglm.plotting.distributional import plot_qq

    assert plotting.plot_qq is plot_qq
    assert "plot_diagnostics_figure" in plotting.__all__
    assert "plotly_diagnostics_figure" in dir(plotting)
    assert "plotly_diagnostics_figure" not in plotting.__all__
    with pytest.raises(AttributeError):
        plotting.plotly_nonsense


def test_plotly_engine_returns_a_plotly_figure(case):
    plotly = pytest.importorskip("plotly")
    import superglm.plotting as plotting

    assert plotting.plotly_diagnostics_figure.__name__ == "plotly_diagnostics_figure"
    model, frame, y = case
    figure = model.plot_diagnostics(frame, y, engine="plotly", n_sim=8)
    assert isinstance(figure, plotly.graph_objects.Figure)
    grid = model.plot(parameter="scale", engine="plotly", n_sim=64)
    assert isinstance(grid, plotly.graph_objects.Figure)


def test_an_unknown_engine_is_refused(case):
    model, frame, y = case
    with pytest.raises(ValueError, match="engine must be"):
        model.plot_diagnostics(frame, y, engine="ggplot")


# --------------------------------------------------------------------------- #
# Weights: which slot a sample weight reaches
# --------------------------------------------------------------------------- #


def test_a_prior_weight_reaches_the_row_law(weighted_fits):
    model, frame, _, weights = weighted_fits["gamma"]
    fitted = model._require_fitted()
    rows = frame.head(30)
    law = weights[:30]
    bounds = model.posterior_bounds(rows, ("quantile", 0.9), n_draws=32, seed=11, sample_weight=law)
    expected = posterior_bounds(fitted, rows, ("quantile", 0.9), n_draws=32, seed=11, weights=law)
    assert np.allclose(bounds["estimate"], expected["estimate"])
    assert not np.allclose(
        bounds["estimate"],
        posterior_bounds(fitted, rows, ("quantile", 0.9), n_draws=32, seed=11)["estimate"],
    )
    book = model.portfolio(frame, n_draws=16, seed=11, sample_weight=weights)
    assert np.isclose(
        book.total_mean,
        portfolio(fitted, frame, n_draws=16, seed=11, weights=weights).total_mean,
    )


def test_a_prior_weight_also_weighs_the_spread_table(weighted_fits):
    model, frame, y, weights = weighted_fits["gamma"]
    spread = model.parameter_spread(
        frame, threshold=float(np.quantile(y, 0.9)), n_bins=4, sample_weight=weights
    )
    unweighted = model.parameter_spread(frame, threshold=float(np.quantile(y, 0.9)), n_bins=4)
    assert not np.allclose(spread.identically_priced["mean"], unweighted.identically_priced["mean"])


def test_a_frequency_weight_is_not_a_row_law(case):
    _, frame, y = case
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[Predictor("location", {"x": Spline("cr", k=5)}), Predictor("scale", {})],
        weight_semantics="frequency",
    ).fit_reml(frame, y)
    weights = np.full(len(frame), 2.0)
    with pytest.raises(ValueError, match="frequency"):
        model.posterior_predictive(frame.head(10), 4, sample_weight=weights[:10])
    with pytest.raises(ValueError, match="frequency"):
        model.portfolio(frame, n_draws=8, sample_weight=weights)


def test_an_aggregation_weight_reaches_the_table(weighted_fits):
    model, frame, y, weights = weighted_fits["gamma"]
    fitted = model._require_fitted()
    table = model.actual_expected(frame, y, "x", n_bins=5, sample_weight=weights)
    expected = actual_expected_check(
        fitted, frame, y, frame["x"].to_numpy(), name="x", n_bins=5, sample_weight=weights
    )
    assert np.allclose(table.expected, expected.expected)
    assert np.allclose(table.weight, expected.weight)


# --------------------------------------------------------------------------- #
# The family variance protocols
# --------------------------------------------------------------------------- #


def test_the_families_declare_the_variance_protocols():
    for family in (GaussianLS(), GammaLS(), TweedieLSS(), LogNormalLS()):
        assert isinstance(family, VarianceFamily), type(family).__name__
    for family in (GaussianLS(), GammaLS(), TweedieLSS()):
        assert isinstance(family, PriorWeightedVarianceFamily), type(family).__name__
    # LogNormalLS refuses non-unit prior weights outright, so it has no weighted
    # law to report a variance from.
    assert not isinstance(LogNormalLS(), PriorWeightedVarianceFamily)


def test_the_log_normal_variance_is_its_second_moment():
    """Both parametrisations report ``E[Y]^2 (exp(sigma^2) - 1)``."""
    theta = np.array([[2.0, 0.4], [5.0, 0.9]])
    mean_form = LogNormalLS().variance(theta)
    assert np.allclose(mean_form, theta[:, 0] ** 2 * (np.exp(theta[:, 1] ** 2) - 1.0))
    location = np.column_stack([np.log(theta[:, 0]) - 0.5 * theta[:, 1] ** 2, theta[:, 1]])
    assert np.allclose(LogNormalLS(parametrisation="location").variance(location), mean_form)


@pytest.mark.parametrize("name", ["gaussian", "gamma", "tweedie"])
def test_the_prior_weighted_variance_scales_the_unit_one(weighted_fits, name):
    model, frame, _, weights = weighted_fits[name]
    fitted = model._require_fitted()
    family = fitted.family
    theta = np.asarray(fitted.predict_parameters(frame), dtype=np.float64)
    assert np.allclose(
        family.variance_prior_weighted(theta, weights),
        np.asarray(family.variance(theta)) / weights,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("name", "n_rows", "n_draws"),
    [("gaussian", 24, 4000), ("gamma", 24, 4000), ("tweedie", 20, 1200)],
)
def test_the_prior_weighted_variance_matches_the_predictive_moment(
    weighted_fits, name, n_rows, n_draws
):
    model, frame, _, weights = weighted_fits[name]
    fitted = model._require_fitted()
    rows = frame.head(n_rows)
    law = weights[:n_rows]
    theta = np.asarray(fitted.predict_parameters(rows), dtype=np.float64)
    closed_form = np.asarray(fitted.family.variance_prior_weighted(theta, law), dtype=np.float64)
    simulated = posterior_predictive(
        fitted, rows, n_draws, parameter_uncertainty=False, seed=17, weights=law
    )
    ratio = np.var(simulated, axis=0, ddof=1) / closed_form
    error = float(np.std(ratio, ddof=1)) / np.sqrt(len(ratio))
    assert abs(float(ratio.mean()) - 1.0) < 4.0 * error


def test_actual_expected_reads_the_prior_weighted_variance(weighted_fits):
    model, frame, y, weights = weighted_fits["tweedie"]
    fitted = model._require_fitted()
    weighted = actual_expected_check(
        fitted, frame, y, frame["x"].to_numpy(), name="x", n_bins=5, sample_weight=weights
    )
    assert weighted.variance_law == "family_prior_weighted"
    assert np.all(np.isfinite(weighted.ratio_se))
    unit = actual_expected_check(fitted, frame, y, frame["x"].to_numpy(), name="x", n_bins=5)
    assert unit.variance_law == "family"
