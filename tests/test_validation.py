"""Tests for superglm.validation — actuarial validation toolkit (T6-T12)."""

from __future__ import annotations

import importlib.util

import matplotlib
import numpy as np
import pandas as pd
import pytest

import superglm.validation as validation_module

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from superglm.validation import (
    DoubleLiftChartResult,
    LiftChartResult,
    LorenzCurveResult,
    LossRatioChartResult,
    double_lift_chart,
    lift_chart,
    lorenz_curve,
    loss_ratio_chart,
)

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None
EXTENDED_RANGE_AVAILABLE = (
    np.finfo(np.longdouble).max > np.finfo(np.float64).max
    and np.finfo(np.longdouble).tiny < np.finfo(np.float64).tiny
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


# ── T6: lift_chart basic ─────────────────────────────────────────


class TestLiftChartBasic:
    """T6: Basic lift_chart test."""

    def test_returns_lift_chart_result(self):
        y_obs = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.1, 4.2, 4.8], dtype=float)
        result = lift_chart(y_obs, y_pred, n_bins=5)
        assert isinstance(result, LiftChartResult)

    def test_bins_columns(self):
        y_obs = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.1, 4.2, 4.8], dtype=float)
        result = lift_chart(y_obs, y_pred, n_bins=5)
        expected_cols = {"bin", "exposure_share", "observed", "predicted", "obs_pred_ratio"}
        assert expected_cols == set(result.bins.columns)

    def test_bins_count(self):
        y_obs = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.1, 4.2, 4.8], dtype=float)
        result = lift_chart(y_obs, y_pred, n_bins=5)
        assert len(result.bins) <= 5

    def test_well_calibrated_ratio(self):
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.exponential(2.0, n)
        y_pred = y + rng.normal(0, 0.1, n)  # almost perfect
        result = lift_chart(y, y_pred, n_bins=10)
        # A/E ratios should be close to 1.0
        ratios = result.bins["obs_pred_ratio"].values
        assert np.all(np.abs(ratios - 1.0) < 0.5)

    def test_figure_returned(self):
        result = lift_chart([1, 2, 3], [1, 2, 3], n_bins=3)
        assert result.figure is not None


# ── T7: lift_chart with sample_weight and exposure ────────────────


class TestLiftChartWeighted:
    """T7: Weighted lift_chart tests."""

    def test_exposure_weighted_bins(self):
        rng = np.random.default_rng(42)
        n = 500
        y_obs = rng.poisson(3.0, n).astype(float)
        y_pred = np.full(n, 3.0) + rng.normal(0, 0.5, n)
        exposure = rng.uniform(0.5, 2.0, n)
        result = lift_chart(y_obs, y_pred, exposure=exposure, n_bins=5)
        # Exposure shares should sum to ~1
        assert abs(result.bins["exposure_share"].sum() - 1.0) < 1e-6

    def test_sample_weight_affects_result(self):
        y_obs = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.1, 4.2, 4.8], dtype=float)
        w1 = np.ones(5)
        w2 = np.array([10, 1, 1, 1, 1], dtype=float)
        r1 = lift_chart(y_obs, y_pred, sample_weight=w1, n_bins=3)
        r2 = lift_chart(y_obs, y_pred, sample_weight=w2, n_bins=3)
        # Different weights should generally produce different results
        # At minimum, the weighted means should differ
        assert not np.allclose(r1.bins["observed"].values, r2.bins["observed"].values)

    def test_consistency_weighted_mean(self):
        """sum(bins.observed * bins.exposure_share) ≈ overall observed mean."""
        rng = np.random.default_rng(42)
        n = 500
        y_obs = rng.poisson(5.0, n).astype(float)
        y_pred = y_obs + rng.normal(0, 1, n)
        exposure = rng.uniform(0.5, 2.0, n)
        w = rng.uniform(0.5, 1.5, n)
        result = lift_chart(y_obs, y_pred, sample_weight=w, exposure=exposure, n_bins=10)
        # Weighted overall mean
        we = w * exposure
        overall_mean = np.sum(we * y_obs) / we.sum()
        reconstructed = (result.bins["observed"] * result.bins["exposure_share"]).sum()
        assert abs(reconstructed - overall_mean) < 0.5


# ── T8: double_lift_chart ────────────────────────────────────────


class TestDoubleLiftChart:
    """T8: Double lift chart — CAS RPM 2016 methodology."""

    def test_identical_models(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.poisson(3.0, n).astype(float)
        pred = np.abs(y + rng.normal(0, 0.5, n)) + 0.01
        result = double_lift_chart(y, pred, pred, n_bins=5)
        assert isinstance(result, DoubleLiftChartResult)
        # When model == current, their indices should be identical
        np.testing.assert_allclose(
            result.bins["model_index"].values,
            result.bins["current_index"].values,
            rtol=1e-10,
        )

    def test_exposure_shares_sum_to_one(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.poisson(5.0, n).astype(float)
        pred_m = np.abs(y + rng.normal(0, 0.2, n)) + 0.01
        pred_c = np.abs(y + rng.normal(0, 2.0, n)) + 0.01
        result = double_lift_chart(y, pred_m, pred_c, n_bins=10)
        np.testing.assert_allclose(result.bins["exposure_share"].sum(), 1.0, atol=1e-10)

    def test_required_columns(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.poisson(3.0, n).astype(float)
        pred = np.abs(y + rng.normal(0, 0.5, n)) + 0.01
        result = double_lift_chart(y, pred, pred, n_bins=5)
        required = {
            "bin",
            "n_rows",
            "exposure_sum",
            "exposure_share",
            "target_sum",
            "actual_avg",
            "model_avg",
            "current_avg",
            "actual_index",
            "model_index",
            "current_index",
            "sort_score_min",
            "sort_score_max",
        }
        assert required.issubset(set(result.bins.columns))

    def test_overall_average_reconstruction(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.poisson(5.0, n).astype(float)
        exp = rng.uniform(0.5, 2.0, n)
        pred_m = np.abs(y + rng.normal(0, 0.2, n)) + 0.01
        pred_c = np.abs(y + rng.normal(0, 1.0, n)) + 0.01
        result = double_lift_chart(y, pred_m, pred_c, exposure=exp, n_bins=10)
        df = result.bins
        # Reconstruct overall actual from bin summaries
        reconstructed = (df["actual_avg"] * df["exposure_sum"]).sum() / df["exposure_sum"].sum()
        direct = np.sum(exp * y) / np.sum(exp)
        np.testing.assert_allclose(reconstructed, direct, rtol=1e-6)


# ── T9: lorenz_curve and Gini ────────────────────────────────────


class TestLorenzCurveGini:
    """T9: Lorenz curve and Gini coefficient tests."""

    def test_perfect_model(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.exponential(2.0, n)
        result = lorenz_curve(y, y)  # perfect prediction
        assert isinstance(result, LorenzCurveResult)
        assert abs(result.gini_ratio - 1.0) < 0.05

    def test_random_model(self):
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.exponential(2.0, n)
        y_pred = np.full(n, y.mean())  # constant prediction
        result = lorenz_curve(y, y_pred)
        # Model Gini should be near 0 (random ordering)
        assert abs(result.gini_model) < 0.1
        # Gini ratio should be near 0
        assert abs(result.gini_ratio) < 0.1

    def test_constant_predictions_give_exact_zero_gini(self):
        """Constant predictions should produce no ranking signal."""
        y = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        y_pred = np.ones_like(y)
        exposure = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
        result = lorenz_curve(y, y_pred, exposure=exposure)
        assert result.gini_model == pytest.approx(0.0, abs=1e-12)
        assert result.gini_ratio == pytest.approx(0.0, abs=1e-12)

    def test_near_constant_target_has_stable_gini_ratio(self):
        y = np.array([1.0, 1.0, np.nextafter(1.0, 2.0)])
        exposure = np.array([0.1, 0.2, 10.1])

        constant = lorenz_curve(y, np.ones(3), exposure=exposure)
        perfect = lorenz_curve(y, y, exposure=exposure)
        reverse = lorenz_curve(y, -y, exposure=exposure)

        assert constant.gini_ratio == 0.0
        assert perfect.gini_ratio == pytest.approx(1.0)
        assert reverse.gini_ratio == pytest.approx(-1.0)
        assert constant.gini_model == 0.0
        for result in (perfect, reverse):
            assert result.gini_perfect > 0.0
            assert result.gini_ratio == pytest.approx(result.gini_model / result.gini_perfect)

    def test_tied_predictions_are_permutation_invariant(self):
        """Rows with identical scores should not depend on input order."""
        y = np.array([10.0, 1.0, 8.0, 2.0, 6.0, 3.0])
        y_pred = np.array([0.2, 0.2, 0.5, 0.5, 0.9, 0.9])
        exposure = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 3.0])

        result_a = lorenz_curve(y, y_pred, exposure=exposure)
        perm = np.array([1, 0, 3, 2, 5, 4])
        result_b = lorenz_curve(y[perm], y_pred[perm], exposure=exposure[perm])

        assert result_a.gini_model == pytest.approx(result_b.gini_model, abs=1e-12)
        assert result_a.gini_ratio == pytest.approx(result_b.gini_ratio, abs=1e-12)
        np.testing.assert_allclose(
            result_a.curve["cum_loss_share_model"].values,
            result_b.curve["cum_loss_share_model"].values,
            atol=1e-12,
        )

    def test_gini_bounds(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.exponential(2.0, n)
        y_pred = y + rng.normal(0, 1, n)
        result = lorenz_curve(y, y_pred)
        assert result.gini_model >= -0.01  # allow small numerical noise
        assert result.gini_perfect >= result.gini_model - 0.01
        assert 0.0 <= result.gini_ratio <= 1.01

    def test_lorenz_monotonic(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.exponential(2.0, n)
        y_pred = y + rng.normal(0, 1, n)
        result = lorenz_curve(y, y_pred)
        cum_loss = result.curve["cum_loss_share_model"].values
        # Should be monotonically non-decreasing
        assert np.all(np.diff(cum_loss) >= -1e-10)

    def test_lorenz_nonuniform_exposure_diagonal(self):
        """The random ordering diagonal must equal cum_exposure_share, even
        when exposure is non-uniform (the core insurance use case)."""
        rng = np.random.default_rng(42)
        n = 500
        y = rng.exponential(2.0, n)
        y_pred = y + rng.normal(0, 1, n)
        exposure = rng.uniform(0.5, 5.0, n)  # highly non-uniform
        result = lorenz_curve(y, y_pred, exposure=exposure)
        curve = result.curve
        np.testing.assert_allclose(
            curve["cum_loss_share_ordered"].values,
            curve["cum_exposure_share"].values,
            atol=1e-12,
        )

    def test_lorenz_nonuniform_exposure_gini_bounds(self):
        """Gini bounds should still hold with non-uniform exposure."""
        rng = np.random.default_rng(42)
        n = 500
        y = rng.exponential(2.0, n)
        y_pred = y + rng.normal(0, 1, n)
        exposure = rng.uniform(0.5, 5.0, n)
        result = lorenz_curve(y, y_pred, exposure=exposure)
        assert result.gini_model >= -0.01
        assert result.gini_perfect >= result.gini_model - 0.01
        assert 0.0 <= result.gini_ratio <= 1.01

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_plotly_engine_returns_plotly_figure(self):
        import plotly.graph_objects as go

        rng = np.random.default_rng(42)
        n = 300
        y = rng.exponential(2.0, n)
        y_pred = y + rng.normal(0, 0.5, n)
        result = lorenz_curve(y, y_pred, engine="plotly")
        assert isinstance(result.figure, go.Figure)
        assert [trace.name for trace in result.figure.data] == ["Random", "Model", "Perfect"]

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_plotly_engine_rejects_matplotlib_ax(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="engine='matplotlib'"):
            lorenz_curve([1, 2, 3], [1, 2, 3], engine="plotly", ax=ax)

    def test_lorenz_endpoints(self):
        rng = np.random.default_rng(42)
        n = 500
        y = rng.exponential(2.0, n)
        y_pred = y + rng.normal(0, 1, n)
        result = lorenz_curve(y, y_pred)
        curve = result.curve
        # Starts at (0, 0)
        assert abs(curve["cum_exposure_share"].iloc[0]) < 1e-10
        assert abs(curve["cum_loss_share_model"].iloc[0]) < 1e-10
        # Ends at (1, 1)
        assert abs(curve["cum_exposure_share"].iloc[-1] - 1.0) < 1e-10
        assert abs(curve["cum_loss_share_model"].iloc[-1] - 1.0) < 1e-10


# ── T10: loss_ratio_chart ────────────────────────────────────────


class TestLossRatioChart:
    """T10: Loss ratio chart tests."""

    def test_with_feature_values(self):
        rng = np.random.default_rng(42)
        n = 500
        feature = rng.uniform(0, 10, n)
        y_obs = rng.poisson(3.0, n).astype(float)
        y_pred = np.full(n, 3.0) + rng.normal(0, 0.5, n)
        result = loss_ratio_chart(
            y_obs, y_pred, feature_values=feature, feature_name="age", n_bins=5
        )
        assert isinstance(result, LossRatioChartResult)
        assert len(result.bins) <= 5

    def test_without_feature(self):
        rng = np.random.default_rng(42)
        n = 200
        y_obs = rng.poisson(3.0, n).astype(float)
        y_pred = np.full(n, 3.0) + rng.normal(0, 0.5, n)
        result = loss_ratio_chart(y_obs, y_pred, n_bins=5)
        assert isinstance(result, LossRatioChartResult)
        assert "observed" in result.bins.columns
        assert "predicted" in result.bins.columns


# ── T11: ax parameter ────────────────────────────────────────────


class TestAxParameter:
    """T11: Test the ax parameter behavior."""

    def test_preexisting_ax_returns_no_figure(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = lift_chart([1, 2, 3], [1, 2, 3], n_bins=3, ax=ax)
        assert result.figure is None

    def test_no_ax_returns_figure(self):
        result = lift_chart([1, 2, 3], [1, 2, 3], n_bins=3)
        assert result.figure is not None

    def test_lorenz_ax_parameter(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = lorenz_curve([1, 2, 3], [1, 2, 3], ax=ax)
        assert result.figure is None

    def test_double_lift_ax_parameter(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = double_lift_chart([1, 2, 3], [1, 2, 3], [1, 2, 3], n_bins=3, ax=ax)
        assert result.figure is None

    def test_loss_ratio_ax_parameter(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = loss_ratio_chart([1, 2, 3], [1, 2, 3], n_bins=3, ax=ax)
        assert result.figure is None


# ── T12: Edge cases ──────────────────────────────────────────────


class TestEdgeCases:
    """T12: Edge cases for validation functions."""

    def test_all_zero_y_obs(self):
        y_obs = np.zeros(100)
        y_pred = np.ones(100)
        result = lorenz_curve(y_obs, y_pred)
        assert result.gini_model == 0.0

    def test_single_observation(self):
        result = lift_chart([1.0], [1.0], n_bins=1)
        assert isinstance(result, LiftChartResult)

    def test_negative_predictions(self):
        rng = np.random.default_rng(42)
        y_obs = rng.exponential(2.0, 100)
        y_pred = rng.normal(0, 1, 100)  # some negative
        result = lift_chart(y_obs, y_pred, n_bins=5)
        assert isinstance(result, LiftChartResult)


@pytest.mark.parametrize(
    ("chart", "prediction_name"),
    [
        (lambda y, pred: lift_chart(y, pred), "y_pred"),
        (lambda y, pred: double_lift_chart(y, pred, np.ones_like(y)), "y_pred_model"),
        (lambda y, pred: lorenz_curve(y, pred), "y_pred"),
        (lambda y, pred: loss_ratio_chart(y, pred), "y_pred"),
    ],
)
def test_public_charts_reject_prediction_length_mismatches(chart, prediction_name):
    y_obs = np.arange(1.0, 11.0)

    with pytest.raises(
        ValueError,
        match=rf"{prediction_name} must have length 10, got 3",
    ):
        chart(y_obs, np.ones(3))


@pytest.mark.parametrize(
    "chart",
    [
        lambda: lift_chart([], []),
        lambda: double_lift_chart([], [], []),
        lambda: lorenz_curve([], []),
        lambda: loss_ratio_chart([], []),
    ],
)
def test_public_charts_reject_empty_inputs(chart):
    with pytest.raises(ValueError, match="y_obs must be non-empty"):
        chart()


@pytest.mark.parametrize(
    "empty",
    [np.empty((0, 1)), pd.DataFrame(index=[])],
)
def test_public_charts_reject_every_empty_observation_container_as_empty(empty):
    with pytest.raises(ValueError, match="y_obs must be non-empty"):
        lift_chart(empty, [])


@pytest.mark.parametrize(
    ("chart", "prediction_name"),
    [
        (lambda: lift_chart([1.0], None), "y_pred"),
        (lambda: lorenz_curve([1.0], None), "y_pred"),
        (lambda: loss_ratio_chart([1.0], None), "y_pred"),
        (lambda: double_lift_chart([1.0], None, [1.0]), "y_pred_model"),
        (lambda: double_lift_chart([1.0], [1.0], None), "y_pred_current"),
    ],
)
def test_public_charts_reject_none_for_required_prediction_vectors(chart, prediction_name):
    with pytest.raises(ValueError, match=rf"{prediction_name} must be one-dimensional"):
        chart()


@pytest.mark.parametrize(
    "chart",
    [
        lambda w: lift_chart([1.0, 2.0], [1.0, 2.0], sample_weight=w),
        lambda w: double_lift_chart(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            sample_weight=w,
        ),
        lambda w: lorenz_curve([1.0, 2.0], [1.0, 2.0], sample_weight=w),
        lambda w: loss_ratio_chart([1.0, 2.0], [1.0, 2.0], sample_weight=w),
    ],
)
def test_public_charts_reject_all_zero_weights(chart):
    with pytest.raises(ValueError, match="sample_weight must not be all zero"):
        chart(np.zeros(2))


@pytest.mark.parametrize(
    "chart",
    [
        lambda n_bins: lift_chart([1.0, 2.0], [1.0, 2.0], n_bins=n_bins),
        lambda n_bins: double_lift_chart(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            n_bins=n_bins,
        ),
        lambda n_bins: loss_ratio_chart([1.0, 2.0], [1.0, 2.0], n_bins=n_bins),
    ],
)
@pytest.mark.parametrize(
    "n_bins",
    [0, -1, 1.5, np.float64(2.0), True, False, None, "2", np.array(2)],
)
def test_binned_charts_require_positive_integer_n_bins(chart, n_bins):
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        chart(n_bins)


@pytest.mark.parametrize(
    ("chart", "field_name"),
    [
        (
            lambda bad: lift_chart([1.0, 2.0], [1.0, 2.0], sample_weight=bad),
            "sample_weight",
        ),
        (
            lambda bad: lift_chart([1.0, 2.0], [1.0, 2.0], exposure=bad),
            "exposure",
        ),
        (
            lambda bad: double_lift_chart(
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                sample_weight=bad,
            ),
            "sample_weight",
        ),
        (
            lambda bad: double_lift_chart(
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
                exposure=bad,
            ),
            "exposure",
        ),
        (
            lambda bad: lorenz_curve([1.0, 2.0], [1.0, 2.0], sample_weight=bad),
            "sample_weight",
        ),
        (
            lambda bad: lorenz_curve([1.0, 2.0], [1.0, 2.0], exposure=bad),
            "exposure",
        ),
        (
            lambda bad: loss_ratio_chart(
                [1.0, 2.0],
                [1.0, 2.0],
                sample_weight=bad,
            ),
            "sample_weight",
        ),
        (
            lambda bad: loss_ratio_chart([1.0, 2.0], [1.0, 2.0], exposure=bad),
            "exposure",
        ),
        (
            lambda bad: loss_ratio_chart(
                [1.0, 2.0],
                [1.0, 2.0],
                feature_values=bad,
            ),
            "feature_values",
        ),
        (
            lambda bad: double_lift_chart(
                [1.0, 2.0],
                [1.0, 2.0],
                bad,
            ),
            "y_pred_current",
        ),
    ],
)
def test_every_public_chart_vector_has_an_explicit_length_boundary(chart, field_name):
    with pytest.raises(ValueError, match=rf"{field_name} must have length 2, got 1"):
        chart([1.0])


@pytest.mark.parametrize(
    ("chart", "field_name"),
    [
        (lambda bad: lift_chart(bad, [1.0, 2.0]), "y_obs"),
        (lambda bad: lift_chart([1.0, 2.0], bad), "y_pred"),
        (
            lambda bad: double_lift_chart(bad, [1.0, 2.0], [1.0, 2.0]),
            "y_obs",
        ),
        (
            lambda bad: double_lift_chart([1.0, 2.0], bad, [1.0, 2.0]),
            "y_pred_model",
        ),
        (
            lambda bad: double_lift_chart([1.0, 2.0], [1.0, 2.0], bad),
            "y_pred_current",
        ),
        (lambda bad: lorenz_curve(bad, [1.0, 2.0]), "y_obs"),
        (lambda bad: lorenz_curve([1.0, 2.0], bad), "y_pred"),
        (lambda bad: loss_ratio_chart(bad, [1.0, 2.0]), "y_obs"),
        (lambda bad: loss_ratio_chart([1.0, 2.0], bad), "y_pred"),
        (
            lambda bad: loss_ratio_chart(
                [1.0, 2.0],
                [1.0, 2.0],
                feature_values=bad,
            ),
            "feature_values",
        ),
    ],
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_every_public_chart_numeric_vector_rejects_nonfinite_values(
    chart,
    field_name,
    bad_value,
):
    with pytest.raises(ValueError, match=rf"{field_name} must contain only finite values"):
        chart([1.0, bad_value])


@pytest.mark.parametrize(
    "chart",
    [
        lambda **kwargs: lift_chart([1.0, 2.0], [1.0, 2.0], **kwargs),
        lambda **kwargs: double_lift_chart(
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
            **kwargs,
        ),
        lambda **kwargs: lorenz_curve([1.0, 2.0], [1.0, 2.0], **kwargs),
        lambda **kwargs: loss_ratio_chart([1.0, 2.0], [1.0, 2.0], **kwargs),
    ],
)
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_weight": [1.0, -1.0]}, "sample_weight must be nonnegative"),
        (
            {"sample_weight": [1.0, np.nan]},
            "sample_weight must contain only finite values",
        ),
        ({"exposure": [1.0, -1.0]}, "exposure must be nonnegative"),
        ({"exposure": [1.0, np.inf]}, "exposure must contain only finite values"),
        (
            {"exposure": [0.0, 0.0]},
            "sample_weight \\* exposure must not be all zero",
        ),
        (
            {"sample_weight": [0.0, 1.0], "exposure": [1.0, 0.0]},
            "sample_weight \\* exposure must not be all zero",
        ),
        (
            {
                "sample_weight": [np.finfo(float).max, np.finfo(float).max],
                "exposure": [1.0, 1.0],
            },
            "sample_weight \\* exposure must have a finite total",
        ),
        (
            {"sample_weight": [np.finfo(float).max, np.finfo(float).max]},
            "sample_weight must have a finite total",
        ),
    ],
)
def test_public_chart_weight_boundaries_are_shared(chart, kwargs, message):
    with pytest.raises(ValueError, match=message):
        chart(**kwargs)


@pytest.mark.parametrize(
    "chart",
    [
        lambda w: lift_chart([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sample_weight=w),
        lambda w: double_lift_chart(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            sample_weight=w,
        ),
        lambda w: lorenz_curve([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sample_weight=w),
        lambda w: loss_ratio_chart(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            sample_weight=w,
        ),
    ],
)
def test_public_charts_reject_exact_weight_total_overflow(chart):
    maximum = np.finfo(np.float64).max
    rounding_hidden_addend = np.ldexp(1.0, 969)

    with pytest.raises(ValueError, match="sample_weight must have a finite total"):
        chart(np.array([maximum, rounding_hidden_addend, rounding_hidden_addend]))


@pytest.mark.parametrize(
    "chart",
    [
        lambda y, pred, **kwargs: lift_chart(y, pred, **kwargs).bins,
        lambda y, pred, **kwargs: double_lift_chart(y, pred, pred, **kwargs).bins,
        lambda y, pred, **kwargs: lorenz_curve(y, pred, **kwargs).curve,
        lambda y, pred, **kwargs: loss_ratio_chart(y, pred, **kwargs).bins,
    ],
)
def test_zero_effective_weight_rows_are_ignored_instead_of_forming_empty_bins(chart):
    y = np.array([1000.0, 2.0, 4.0])
    pred = np.array([-1000.0, 2.5, 3.5])
    weighted = chart(y, pred, sample_weight=np.array([0.0, 1.0, 1.0]))
    filtered = chart(y[1:], pred[1:])

    np.testing.assert_allclose(
        weighted.select_dtypes(include=[np.number]),
        filtered.select_dtypes(include=[np.number]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "chart",
    [
        lambda y, pred, **kwargs: lift_chart(y, pred, **kwargs).bins,
        lambda y, pred, **kwargs: double_lift_chart(y, pred, pred, **kwargs).bins,
        lambda y, pred, **kwargs: lorenz_curve(y, pred, **kwargs).curve,
        lambda y, pred, **kwargs: loss_ratio_chart(y, pred, **kwargs).bins,
    ],
)
def test_zero_exposure_rows_are_removed_by_combined_effective_weight(chart):
    y = np.array([1000.0, 2.0, 4.0])
    pred = np.array([-1000.0, 2.5, 3.5])
    weighted = chart(
        y,
        pred,
        sample_weight=np.ones(3),
        exposure=np.array([0.0, 1.0, 1.0]),
    )
    filtered = chart(y[1:], pred[1:])

    np.testing.assert_allclose(
        weighted.select_dtypes(include=[np.number]),
        filtered.select_dtypes(include=[np.number]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "chart",
    [
        lambda y, pred, **kwargs: lift_chart(y, pred, **kwargs).bins,
        lambda y, pred, **kwargs: loss_ratio_chart(y, pred, **kwargs).bins,
    ],
)
def test_weighted_means_avoid_intermediate_overflow(chart):
    maximum = np.finfo(np.float64).max
    result = chart(
        np.array([maximum, 1.0]),
        np.array([maximum, 1.0]),
        sample_weight=np.array([2.0, 1.0]),
        n_bins=1,
    )

    assert np.all(np.isfinite(result[["observed", "predicted"]]))


@pytest.mark.skipif(
    not EXTENDED_RANGE_AVAILABLE,
    reason="platform longdouble does not extend the float64 exponent range",
)
def test_weighted_mean_preserves_a_representable_subnormal_contribution():
    maximum = np.finfo(np.float64).max
    smallest = np.nextafter(0.0, 1.0)
    result = lift_chart(
        np.array([0.0, maximum]),
        np.array([0.0, maximum]),
        sample_weight=np.array([maximum, smallest]),
        n_bins=1,
    )

    assert result.bins.loc[0, "observed"] == smallest
    assert result.bins.loc[0, "predicted"] == smallest


def test_weighted_mean_clamps_rounding_to_the_input_convex_hull():
    import matplotlib.pyplot as plt

    maximum = np.finfo(np.float64).max
    figure, ax = plt.subplots()
    ax.set_autoscale_on(False)
    result = lift_chart(
        np.array([maximum, maximum]),
        np.array([maximum, maximum]),
        np.array([1e-15, 1e-6]),
        n_bins=1,
        ax=ax,
    )
    plt.close(figure)

    assert result.bins.loc[0, "observed"] == maximum
    assert result.bins.loc[0, "predicted"] == maximum


def test_extreme_aggregation_is_explicit_on_float64_only_longdouble_platforms(
    monkeypatch,
):
    smallest = np.nextafter(0.0, 1.0)
    almost_two = np.nextafter(2.0, 0.0)
    monkeypatch.setattr(validation_module, "_LONGDOUBLE_EXTENDS_FLOAT64", False)

    with pytest.raises(ValueError, match="requires an extended floating-point range"):
        lift_chart(
            np.array([almost_two, smallest]),
            np.array([almost_two, smallest]),
            sample_weight=np.array([smallest, 1.0]),
            n_bins=1,
        )


def test_float64_only_range_gate_allows_safe_cross_unit_single_row(
    monkeypatch,
):
    scale = np.ldexp(1.0, 500)
    smallest = np.nextafter(0.0, 1.0)
    monkeypatch.setattr(validation_module, "_LONGDOUBLE_EXTENDS_FLOAT64", False)

    lift = lift_chart(
        [scale],
        [scale],
        sample_weight=[smallest],
        n_bins=1,
    )
    lorenz = lorenz_curve(
        [scale],
        [scale],
        sample_weight=[smallest],
    )

    assert lift.bins.loc[0, "observed"] == scale
    assert lift.bins.loc[0, "predicted"] == scale
    assert np.all(np.isfinite(lorenz.curve))


def test_float64_only_range_gate_rejects_unsafe_normalized_products(
    monkeypatch,
):
    smallest = np.nextafter(0.0, 1.0)
    scale = np.ldexp(1.0, 600)
    monkeypatch.setattr(validation_module, "_LONGDOUBLE_EXTENDS_FLOAT64", False)

    with pytest.raises(ValueError, match="requires an extended floating-point range"):
        lift_chart(
            [scale, -scale, np.ldexp(1.0, 100), np.ldexp(1.0, 100)],
            [scale, -scale, np.ldexp(1.0, 100), np.ldexp(1.0, 100)],
            sample_weight=[
                smallest,
                smallest,
                np.ldexp(1.0, -600),
                np.ldexp(1.0, -600),
            ],
            n_bins=1,
        )

    with pytest.raises(ValueError, match="requires an extended floating-point range"):
        lorenz_curve(
            [scale, -scale, 1.0, 0.0],
            [scale, -scale, 1.0, 0.0],
            sample_weight=[
                np.ldexp(1.0, -600),
                np.ldexp(1.0, -600),
                np.ldexp(1.0, -600),
                1.0,
            ],
        )


def test_float64_only_lorenz_gate_rejects_unsafe_finite_products_sum(
    monkeypatch,
):
    maximum = np.finfo(np.float64).max
    monkeypatch.setattr(validation_module, "_LONGDOUBLE_EXTENDS_FLOAT64", False)

    with pytest.raises(ValueError, match="requires an extended floating-point range"):
        lorenz_curve([maximum, maximum], [1.0, 2.0])


def test_float64_only_aggregation_uses_compensated_summation(
    monkeypatch,
):
    values = np.array([1.0, 1e-16, -1.0])
    monkeypatch.setattr(validation_module, "_LONGDOUBLE_EXTENDS_FLOAT64", False)

    lift = lift_chart(values, values, n_bins=1)
    lorenz = lorenz_curve(values, values)

    assert lift.bins.loc[0, "observed"] == pytest.approx(1e-16 / 3.0)
    assert np.all(np.isfinite(lorenz.curve))
    assert lorenz.curve.iloc[-1]["cum_loss_share_model"] == pytest.approx(1.0)
    assert lorenz.gini_ratio == pytest.approx(1.0)


def test_double_lift_rejects_nonfinite_derived_sort_score():
    maximum = np.finfo(np.float64).max
    with pytest.raises(
        ValueError,
        match="y_pred_model / y_pred_current must contain only finite values",
    ):
        double_lift_chart(
            [1.0, 2.0],
            [maximum, 1.0],
            [1e-300, 1.0],
        )


@pytest.mark.skipif(
    not EXTENDED_RANGE_AVAILABLE,
    reason="platform longdouble does not extend the float64 exponent range",
)
def test_lorenz_scaling_avoids_weighted_loss_overflow():
    maximum = np.finfo(np.float64).max
    result = lorenz_curve(
        np.array([maximum, 1.0]),
        np.array([maximum, 1.0]),
        sample_weight=np.array([2.0, 1.0]),
    )

    assert np.all(np.isfinite(result.curve))
    assert np.isfinite(result.gini_model)
    assert np.isfinite(result.gini_perfect)
    assert np.isfinite(result.gini_ratio)


@pytest.mark.skipif(
    not EXTENDED_RANGE_AVAILABLE,
    reason="platform longdouble does not extend the float64 exponent range",
)
def test_lorenz_gini_preserves_extreme_positive_weight_ratios():
    maximum = np.finfo(np.float64).max
    smallest = np.nextafter(0.0, 1.0)
    result = lorenz_curve(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        sample_weight=np.array([maximum, smallest]),
    )

    assert result.gini_model == pytest.approx(1.0)
    assert result.gini_perfect == pytest.approx(1.0)
    assert result.gini_ratio == pytest.approx(1.0)


@pytest.mark.skipif(
    not EXTENDED_RANGE_AVAILABLE,
    reason="platform longdouble does not extend the float64 exponent range",
)
def test_lorenz_curve_preserves_balanced_extreme_weight_loss_products():
    maximum = np.finfo(np.float64).max
    smallest = np.nextafter(0.0, 1.0)
    result = lorenz_curve(
        np.array([smallest, maximum]),
        np.array([0.0, 1.0]),
        sample_weight=np.array([maximum, smallest]),
    )

    assert result.curve.loc[1, "cum_loss_share_model"] == pytest.approx(0.5)
    assert result.curve.loc[1, "cum_exposure_share"] == pytest.approx(1.0)


def test_two_row_reverse_ranking_gini_is_exact_and_scale_invariant():
    first = lorenz_curve(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        sample_weight=np.array([1.0, 2e-19]),
    )
    scaled = lorenz_curve(
        np.array([1e-5, 0.0]),
        np.array([0.0, 1.0]),
        sample_weight=np.array([1.0, 2e-19]),
    )

    assert first.gini_model == -first.gini_perfect
    assert first.gini_ratio == -1.0
    assert scaled.gini_ratio == -1.0


def test_lorenz_rejects_shares_or_gini_outside_float64_output_range():
    scale = np.ldexp(1.0, 500)
    y = np.array([scale, -scale, np.ldexp(1.0, -1000), 0.0])
    weights = np.array(
        [
            np.ldexp(1.0, -500),
            np.ldexp(1.0, -500),
            np.ldexp(1.0, -1000),
            1.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="Lorenz cumulative shares must be finite|Gini coefficients must be finite",
    ):
        lorenz_curve(y, y, sample_weight=weights)
