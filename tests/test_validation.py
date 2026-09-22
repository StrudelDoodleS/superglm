"""Tests for superglm.validation — actuarial validation toolkit (T6-T12)."""

from __future__ import annotations

import importlib.util

import matplotlib
import numpy as np
import pandas as pd
import pytest

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
    # Weight * value overflows binary64 before the power-of-two column scaling.
    maximum = np.finfo(np.float64).max
    result = chart(
        np.array([maximum, 2.0**900]),
        np.array([maximum, 2.0**900]),
        sample_weight=np.array([2.0, 1.0]),
        n_bins=1,
    )

    assert np.all(np.isfinite(result[["observed", "predicted"]]))


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


def test_float64_aggregation_allows_safe_cross_unit_single_row():
    scale = np.ldexp(1.0, 500)
    smallest = np.nextafter(0.0, 1.0)

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


def test_float64_lorenz_shares_do_not_require_a_representable_total_loss():
    maximum = np.finfo(np.float64).max
    result = lorenz_curve([maximum, maximum], [1.0, 2.0])
    np.testing.assert_array_equal(result.curve["cum_loss_share_model"], [0.0, 0.5, 1.0])


def test_float64_aggregation_uses_compensated_summation():
    values = np.array([1.0, 1e-16, -1.0])

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


def test_lorenz_scaling_avoids_weighted_loss_overflow():
    maximum = np.finfo(np.float64).max
    result = lorenz_curve(
        np.array([maximum, 2.0**900]),
        np.array([maximum, 2.0**900]),
        sample_weight=np.array([2.0, 1.0]),
    )

    assert np.all(np.isfinite(result.curve))
    assert np.isfinite(result.gini_model)
    assert np.isfinite(result.gini_perfect)
    assert np.isfinite(result.gini_ratio)


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
        match="Lorenz cumulative shares must be finite|Gini coefficients must be finite|validation inputs must span at most",
    ):
        lorenz_curve(y, y, sample_weight=weights)


@pytest.mark.parametrize("bits", [30, 40])
def test_weighted_aggregation_keeps_exact_product_residuals(bits):
    from fractions import Fraction

    from superglm.validation import _weighted_mean, _weighted_total

    d = 2.0**-bits
    values, weights = np.array([-1 - d, 1.0]), np.array([1 - d, 1.0])
    exact = sum(Fraction.from_float(y) * Fraction.from_float(w) for y, w in zip(values, weights))
    mean = exact / sum(map(Fraction.from_float, weights))
    assert _weighted_total(values, weights, "test") == float(exact)
    assert _weighted_mean(values, weights, "test") == float(mean)
    lift = lift_chart(values, values, sample_weight=weights, n_bins=1)
    assert lift.bins.loc[0, "observed"] == float(mean)


def test_lorenz_product_cancellation_preserves_positive_loss_and_perfect_ranking():
    from fractions import Fraction

    d = 2.0**-30
    values, weights = np.array([-1 - d, 1.0]), np.array([1 - d, 1.0])
    result = lorenz_curve(values, values, sample_weight=weights)
    exact_first = Fraction.from_float(values[0]) * Fraction.from_float(weights[0])
    exact_total = exact_first + 1
    assert len(result.curve) == 3  # Not the zero-loss fallback.
    assert result.gini_ratio == 1.0
    assert result.curve.loc[1, "cum_loss_share_model"] == float(exact_first / exact_total)
    assert result.curve.loc[2, "cum_loss_share_model"] == 1.0


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_weighted_mean_clips_in_scaled_units_at_adjacent_weight_boundary(sign):
    maximum = sign * np.finfo(float).max
    _, ax = plt.subplots()
    # This tests the returned aggregation, not Matplotlib's max-float margins.
    ax.set_ylim(-1.0, 1.0)
    result = lift_chart(
        [maximum, maximum],
        [maximum, maximum],
        sample_weight=[np.nextafter(1.0, 2.0), 1.0],
        n_bins=1,
        ax=ax,
    )
    assert result.bins.loc[0, "observed"] == maximum
    assert result.bins.loc[0, "predicted"] == maximum


def test_weighted_mean_keeps_exact_zero_before_hull_scaling():
    tiny = np.nextafter(0.0, 1.0)
    # Equal positive weights and opposite targets have exactly zero mean,
    # including when each physical weighted product is below float64 range.
    result = lift_chart(
        [-1e-100, 1e-100],
        [-1e-100, 1e-100],
        sample_weight=[tiny, tiny],
        n_bins=1,
    )
    assert result.bins.loc[0, "observed"] == 0.0
    assert result.bins.loc[0, "predicted"] == 0.0


@pytest.mark.parametrize("split_tie", [False, True])
def test_gini_retains_prefix_and_block_parts_through_pair_contractions(split_tie):
    from fractions import Fraction

    from superglm.validation import _normalized_gini

    small = 2.0**-55
    y = np.array([1.0, 1.0, 0.0, 1.0] if split_tie else [1.0, 0.0, 1.0])
    scores = np.array([0.0, 0.0, 1.0, 2.0] if split_tie else [0.0, 1.0, 2.0])
    weights = np.array([1.0, small, small, 1.0] if split_tie else [1.0, small, 1.0])
    targets, exact_weights = (
        list(map(Fraction.from_float, y)),
        list(map(Fraction.from_float, weights)),
    )

    def pair_sum(ordering):
        return sum(
            exact_weights[i] * exact_weights[j] * (targets[j] - targets[i])
            for i in range(len(y))
            for j in range(len(y))
            if ordering[i] < ordering[j]
        )

    model, perfect = pair_sum(scores), pair_sum(y)
    expected = float(model / perfect)
    result = lorenz_curve(y, scores, sample_weight=weights)
    scored = _normalized_gini(y, scores, weights)
    if not split_tie:
        assert model == 0 and perfect == Fraction(1, 2**54)
        assert result.gini_ratio == scored == 0.0
    else:
        assert model == -Fraction(1, 2**110)
        allowance = 8 * len(y) * np.finfo(float).eps
        assert result.gini_ratio == pytest.approx(expected, rel=allowance, abs=0)
        assert scored == pytest.approx(expected, rel=allowance, abs=0)


def test_lorenz_keeps_exposure_product_residuals_until_loss_reduction():
    from fractions import Fraction

    d = 2.0**-30
    y, weights, exposure = np.array([-1.0, 1.0]), np.array([1 - d, 1.0]), np.array([1 + d, 1.0])
    losses = [
        Fraction.from_float(a) * Fraction.from_float(b) * Fraction.from_float(c)
        for a, b, c in zip(y, weights, exposure, strict=True)
    ]
    total = sum(losses)
    assert total == Fraction(1, 2**60)
    result = lorenz_curve(y, y, sample_weight=weights, exposure=exposure)
    expected = [0.0, float(losses[0] / total), 1.0]
    np.testing.assert_array_equal(result.curve["cum_loss_share_model"], expected)
    np.testing.assert_array_equal(result.curve["cum_loss_share_perfect"], expected)


@pytest.mark.parametrize("consumer", ["lift", "lorenz", "gini"])
def test_ordinary_validation_does_not_initialize_native_kernels(monkeypatch, consumer):
    from numba.core.registry import CPUDispatcher

    import superglm.validation as validation

    def forbidden(*args, **kwargs):
        pytest.fail("ordinary validation must not pay for native initialization")

    monkeypatch.setattr(CPUDispatcher, "__call__", forbidden)
    observed = 1.0 + np.arange(128) / 16.0
    weights = 0.5 + np.arange(128) / 128.0
    if consumer == "lift":
        validation.lift_chart(observed, observed[::-1], sample_weight=weights)
    elif consumer == "lorenz":
        validation.lorenz_curve(observed, observed[::-1], sample_weight=weights)
    else:
        validation._normalized_gini(observed, observed[::-1], weights)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("with_exposure", [False, True])
def test_ordinary_validation_prefixes_match_exact_products(strided, with_exposure):
    import math
    from fractions import Fraction

    import superglm.validation as validation

    d = 2.0**-30
    values = np.array([-1 - d, 1.0, 2.0**-32, -(2.0**-32), 3.0, -3.0])
    weights = np.array([1 - d, 1.0, 1.0, 1.0, 1 + d, 1 + d])
    exposure = np.array([1 + d, 1.0, 1 + d, 1.0, 1 - d, 1 - d])
    if strided:
        values, weights, exposure = (
            np.repeat(array, 2)[::2] for array in (values, weights, exposure)
        )
    for array in (values, weights, exposure):
        array.flags.writeable = False
    mantissas, powers = validation._scaled_product_sums(
        weights,
        values,
        np.arange(1, len(values) + 1),
        exposure=exposure if with_exposure else None,
    )
    exact = Fraction(0)
    for i, (mantissa, power) in enumerate(zip(mantissas, powers, strict=True)):
        term = Fraction.from_float(weights[i]) * Fraction.from_float(values[i])
        if with_exposure:
            term *= Fraction.from_float(exposure[i])
        exact += term
        assert math.ldexp(float(mantissa), int(power)) == float(exact)


@pytest.mark.parametrize(
    "terms",
    [
        [1.0, 2.0**-53, 2.0**-110],
        [1.0, -(2.0**-54), -(2.0**-110)],
        [1.0, 2.0**-53, -(2.0**-110)],
        [1.0, -1.0, 2.0**-160],
    ],
)
def test_compensated_prefix_keeps_halfway_tail_and_cancellation(terms):
    from fractions import Fraction

    from superglm.validation import _ordinary_prefix_parts

    # high + low is Sum2's double-length prefix: within gamma_(k-1)**2 of
    # the exact sum of k terms (Ogita, Rump and Oishi 2005, Proposition 4.5).
    unit = Fraction(1, 2**53)
    high, low = _ordinary_prefix_parts((np.array(terms),))
    exact = absolute = Fraction(0)
    for count, (term, hi, lo) in enumerate(zip(terms, high, low, strict=True), start=1):
        exact += Fraction.from_float(term)
        absolute += abs(Fraction.from_float(term))
        gamma = (count - 1) * unit / (1 - (count - 1) * unit)
        approximation = Fraction.from_float(hi) + Fraction.from_float(lo)
        assert abs(exact - approximation) <= gamma**2 * absolute


@pytest.mark.parametrize("unit", [-500, 0, 500])
@pytest.mark.parametrize("excess", [0, 1])
def test_validation_reduction_range_depends_on_spread_not_units(unit, excess):
    import math

    import superglm.validation as validation

    values = np.array([2.0**unit, 2.0 ** (unit - validation._span_limit(2) - excess)])
    if excess:
        with pytest.raises(ValueError, match="validation inputs must span at most"):
            validation._scaled_product_total(np.ones(2), values)
        return
    assert math.ldexp(*validation._scaled_product_total(np.ones(2), values)) == math.fsum(values)


def test_ordinary_validation_gini_preserves_exact_weight_exposure_pairs_and_ties():
    from fractions import Fraction

    d = 2.0**-30
    observed, scores = np.array([1.0, 0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0, 1.0])
    weights = np.array([1 + d, 1.0, 1 - d, 2.0**-20])
    exposure = np.array([1 - d, 1.0, 1 + d, 1.5])
    mass = [Fraction.from_float(w) * Fraction.from_float(e) for w, e in zip(weights, exposure)]

    def pair_sum(ordering):
        return sum(
            mass[i] * mass[j] * Fraction.from_float(observed[j] - observed[i])
            for i in range(len(observed))
            for j in range(len(observed))
            if ordering[i] < ordering[j]
        )

    expected = float(pair_sum(scores) / pair_sum(observed))
    result = lorenz_curve(observed, scores, sample_weight=weights, exposure=exposure)
    assert result.gini_ratio == pytest.approx(
        expected, rel=8 * len(observed) * np.finfo(float).eps, abs=0
    )


def test_pair_contractions_skip_zero_targets_after_full_weight_prefix(monkeypatch):
    import superglm.validation as validation

    size = 128
    target = np.zeros(size)
    target[[1, 20, 100]] = [0.5, 1.0, 2.0]
    weights = 0.5 + np.arange(size) / size
    prefix_rows, product_rows = [], []
    prefix, product = validation._ordinary_prefix_parts, validation._two_product

    def counted_prefix(terms):
        prefix_rows.append(len(terms[0]))
        return prefix(terms)

    def counted_product(left, right):
        product_rows.append(np.size(left))
        return product(left, right)

    monkeypatch.setattr(validation, "_ordinary_prefix_parts", counted_prefix)
    monkeypatch.setattr(validation, "_two_product", counted_product)
    validation._weighted_pair_concordance(np.arange(size), weights, target)
    assert prefix_rows == [size]
    assert product_rows and set(product_rows) == {np.count_nonzero(target)}


@pytest.mark.parametrize("with_exposure", [False, True])
def test_total_only_reductions_skip_exact_zero_operands(monkeypatch, with_exposure):
    import superglm.validation as validation

    values = np.zeros(128)
    values[[1, 20, 100]] = [0.5, 1.0, 2.0]
    weights = np.ones_like(values)
    exposure = np.ones_like(values) if with_exposure else None
    if with_exposure:
        exposure[20] = 0.0
    visits = []
    original = validation._ordinary_product_terms

    def counted(left, right, exposure=None):
        visits.append(len(left))
        return original(left, right, exposure)

    monkeypatch.setattr(validation, "_ordinary_product_terms", counted)
    validation._scaled_product_total(weights, values, exposure=exposure)
    assert visits == [np.count_nonzero(values) - int(with_exposure)]
    validation._scaled_product_sums(
        weights, values, np.arange(1, len(values) + 1), exposure=exposure
    )
    assert visits[-1] == len(values)


def test_zero_target_rows_keep_their_exact_pair_weight_with_exposure():
    from fractions import Fraction

    from superglm.validation import _normalized_gini

    y = np.array([0.0, 2.0, 0.0, 1.0, 0.0, 0.0])
    scores = np.array([3.0, 0.0, 3.0, 1.0, 0.0, 4.0])
    weights = np.array([1.0, 0.75, 2.0, 1.5, 0.5, 1.25])
    exposure = np.array([0.5, 1.0, 0.75, 1.5, 2.0, 1.25])

    def exact_ratio(mass):
        def pairs(ordering):
            return sum(
                mass[i] * mass[j] * (Fraction.from_float(y[j]) - Fraction.from_float(y[i]))
                for i in range(len(y))
                for j in range(len(y))
                if ordering[i] < ordering[j]
            )

        return float(pairs(scores) / pairs(y))

    simple = list(map(Fraction.from_float, weights))
    assert _normalized_gini(y, scores, weights) == exact_ratio(simple)
    mass = [w * Fraction.from_float(e) for w, e in zip(simple, exposure, strict=True)]
    result = lorenz_curve(y, scores, sample_weight=weights, exposure=exposure)
    assert result.gini_ratio == exact_ratio(mass)


def test_lorenz_reuses_its_totals_and_score_orders_for_gini(monkeypatch):
    import superglm.validation as validation

    size = 128
    y = (np.arange(size) % 7).astype(float)
    scores = np.arange(size, dtype=float)[::-1]
    exposure = 0.5 + np.arange(size) / size
    totals, sorts = [], []
    total, sort = validation._scaled_product_total, np.argsort

    def counted_total(*args, **kwargs):
        totals.append(1)
        return total(*args, **kwargs)

    def counted_sort(values, *args, **kwargs):
        if np.array_equal(values, y) or np.array_equal(values, scores):
            sorts.append(1)
        return sort(values, *args, **kwargs)

    monkeypatch.setattr(validation, "_scaled_product_total", counted_total)
    monkeypatch.setattr(np, "argsort", counted_sort)
    validation.lorenz_curve(y, scores, exposure=exposure)
    assert len(totals) == 2
    assert len(sorts) == 2


@pytest.mark.parametrize("nonzero_residual", [False, True])
def test_ordinary_products_skip_only_an_identically_zero_low_channel(monkeypatch, nonzero_residual):
    import superglm.validation as validation

    d = 2.0**-30
    weights = np.ones(3)
    exposure = np.array([1 - d, 1.0, 0.5])
    if nonzero_residual:
        weights[0] = 1 + d
    calls = []
    product = validation._two_product

    def counted(*args):
        calls.append(1)
        return product(*args)

    monkeypatch.setattr(validation, "_two_product", counted)
    validation._ordinary_product_terms(weights, np.array([1.0, -1.0, 2.0]), exposure)
    assert len(calls) == 1 + 2 * int(nonzero_residual)


@pytest.mark.parametrize("zero_exposure", [False, True])
def test_lorenz_reused_work_matches_standalone_gini_with_ties(zero_exposure):
    import superglm.validation as validation

    y = np.array([1.0, 0.0, 3.0, 2.0, 1.0])
    scores = np.array([2.0, 1.0, 1.0, 0.0, 2.0])
    weights = np.array([0.5, 1.5, 1.0, 0.75, 2.0])
    exposure = np.array([1.25, 0.5, 1.0, 1.5, 0.75])
    if zero_exposure:
        exposure[2] = 0.0
    active = exposure > 0
    expected = validation._gini_coefficients(
        y[active], scores[active], weights[active], exposure=exposure[active]
    )
    result = validation.lorenz_curve(y, scores, sample_weight=weights, exposure=exposure)
    assert (result.gini_model, result.gini_perfect, result.gini_ratio) == expected


@pytest.mark.parametrize("unit_factor", [0, 1, 2])
def test_ordinary_product_terms_do_not_multiply_unit_factors(monkeypatch, unit_factor):
    import superglm.validation as validation

    factors = [np.array([0.75, 1.25, 1.5]), np.array([2.0, -0.5, 0.25]), np.array([0.5, 1.5, 2.0])]
    factors[unit_factor] = np.ones(3)
    calls = []
    original = validation._two_product

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(validation, "_two_product", counted)
    validation._ordinary_product_terms(*factors)
    assert len(calls) == 1


def test_unit_weight_prefixes_do_not_carry_zero_product_channels(monkeypatch):
    import superglm.validation as validation

    values = 0.5 + np.arange(16) / 16
    channels = []
    prefix = validation._ordinary_prefix_parts

    def counted(terms):
        channels.append(len(terms))
        return prefix(terms)

    monkeypatch.setattr(validation, "_ordinary_prefix_parts", counted)
    validation._scaled_product_sums(np.ones(16), values, np.arange(1, 17))
    validation._weighted_pair_concordance(
        np.arange(16), np.ones(16), np.arange(16.0), exposure=values
    )
    assert channels == [1, 1]


@pytest.mark.parametrize("unit_factor", [0, 1, 2])
def test_unit_factor_bypass_preserves_exact_products_and_readonly_strides(unit_factor):
    from fractions import Fraction

    from superglm.validation import _ordinary_product_terms

    d = 2.0**-30
    factors = [
        np.array([1 + d, -1.0, 0.5]),
        np.array([1 - d, 1.0, 2.0]),
        np.array([1 + d, 2.0, -0.5]),
    ]
    factors[unit_factor] = np.ones(3)
    factors = [np.repeat(factor, 2)[::2] for factor in factors]
    before = [factor.copy() for factor in factors]
    for factor in factors:
        factor.flags.writeable = False
    parts = _ordinary_product_terms(*factors)
    for row in range(3):
        expected = Fraction(1)
        for factor in factors:
            expected *= Fraction.from_float(factor[row])
        assert sum(Fraction.from_float(part[row]) for part in parts) == expected
    for factor, original in zip(factors, before, strict=True):
        np.testing.assert_array_equal(factor, original)


@pytest.mark.parametrize(
    "values", [[1.0, 2.0**-53, 2.0**-100], [1.0 + 2.0**-52, 2.0**-53, -(2.0**-100)]]
)
def test_totals_resolve_a_halfway_tie_from_a_distant_tail(values):
    from fractions import Fraction

    from superglm.validation import _weighted_total

    # math.fsum rounds the exact sum once, so the tail breaks the tie.
    values = np.array(values)
    exact = float(sum(map(Fraction.from_float, values)))
    assert _weighted_total(values, np.ones(3), "test") == exact
    result = double_lift_chart(values, np.ones(3), np.ones(3), n_bins=1)
    assert result.bins.loc[0, "target_sum"] == exact


def test_weighted_mean_hull_ignores_zero_weight_rows():
    from fractions import Fraction

    from superglm.validation import _weighted_mean

    values, weights = np.array([0.1, 0.1, 100.0]), np.array([0.1, 0.1, 0.0])
    exact = sum(
        Fraction.from_float(v) * Fraction.from_float(w)
        for v, w in zip(values, weights, strict=True)
    ) / sum(map(Fraction.from_float, weights))
    assert _weighted_mean(values, weights, "test") == float(exact) == 0.1


def test_weighted_mean_is_within_two_ulps():
    from fractions import Fraction

    from superglm.validation import _weighted_mean

    def exact_mean(values, weights):
        numerator = sum(
            Fraction.from_float(v) * Fraction.from_float(w)
            for v, w in zip(values, weights, strict=True)
        )
        return numerator / sum(map(Fraction.from_float, weights))

    # Correctly rounded numerator and weight total, then one division: three
    # roundings of at most u = eps/2 each, plus one subnormal quantum.
    eps, quantum = Fraction(np.finfo(float).eps), Fraction(np.nextafter(0.0, 1.0))
    rng = np.random.default_rng(20260922)
    cases = [(np.array([8.63, 5.41, 3.0, 4.23]), np.array([0.13, 0.21, 0.7, 0.68]))]
    for _ in range(300):
        size = int(rng.integers(2, 7))
        scale = 2.0 ** int(rng.choice([-1074, -1060, -600, 0, 600, 960]))
        values = rng.uniform(-1.0, 1.0, size) * scale
        weights = rng.uniform(0.0, 1.0, size) * 2.0 ** rng.integers(-40, 40, size)
        cases.append((values, weights))
    for values, weights in cases:
        exact = exact_mean(values, weights)
        error = abs(Fraction.from_float(_weighted_mean(values, weights, "test")) - exact)
        assert error <= Fraction(3, 2) * eps * abs(exact) + quantum


@pytest.mark.parametrize(
    ("y", "scores", "weights"),
    [
        (
            [-1.0, 1 + 2.0**-52, 1 + 5 * 2.0**-52, 1 + 11 * 2.0**-52, 1 + 15 * 2.0**-52],
            [0.0, 1.0, 2.0, 4.0, 3.0],
            [2.0**-33, 2.0**31, 2.0**31, 2.0**31, 2.0**31],
        ),
        (
            [-np.finfo(float).max, 0.0, np.finfo(float).max / 2, np.finfo(float).max],
            [1.0, 0.0, 3.0, 2.0],
            [1.0, 0.5, 2.0, 0.25],
        ),
    ],
)
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_gini_contracts_exact_target_differences(y, scores, weights):
    from fractions import Fraction

    from superglm.validation import _normalized_gini

    y, scores, weights = np.array(y), np.array(scores), np.array(weights)
    targets, mass = list(map(Fraction.from_float, y)), list(map(Fraction.from_float, weights))

    def pair_sum(ordering):
        return sum(
            mass[i] * mass[j] * (targets[j] - targets[i])
            for i in range(len(y))
            for j in range(len(y))
            if ordering[i] < ordering[j]
        )

    expected = float(pair_sum(scores) / pair_sum(y))
    # Model and perfect sums are each rounded once to 53 bits, then divided once.
    allowance = 2 * np.finfo(float).eps
    assert _normalized_gini(y, scores, weights) == pytest.approx(expected, rel=allowance, abs=0)
    result = lorenz_curve(y, scores, sample_weight=weights)
    assert result.gini_ratio == pytest.approx(expected, rel=allowance, abs=0)


def test_one_extreme_loss_or_exposure_is_unit_invariant():
    import superglm.validation as validation

    # One large loss or tiny exposure must not change the arithmetic, and a
    # power-of-two change of units must give bitwise-identical Ginis.
    rows = np.arange(64)
    exposure = 0.25 + (rows % 7) / 8
    exposure[1] = 1e-11
    losses = np.where(rows % 3 == 0, 100.0 * (1 + rows % 5), 0.0)
    losses[0] = 2.0**33
    scores = (rows * 37 % 16).astype(float)
    ginis = []
    for unit in (1.0, 2.0**-20):
        result = validation.lorenz_curve(losses * unit, scores, exposure=exposure)
        standalone = validation._gini_coefficients(losses * unit, scores, exposure=exposure)
        ginis.append((result.gini_model, result.gini_perfect, result.gini_ratio, *standalone))
    assert ginis[0] == ginis[1]


@pytest.mark.parametrize(
    "reduce",
    [
        # Weights spanning the full binary64 range on rows that carry mass.
        lambda: lift_chart(
            [np.nextafter(2.0, 0.0), np.nextafter(0.0, 1.0)],
            [np.nextafter(2.0, 0.0), np.nextafter(0.0, 1.0)],
            sample_weight=[np.nextafter(0.0, 1.0), 1.0],
            n_bins=1,
        ),
        lambda: lorenz_curve(
            [0.0, 1.0], [0.0, 1.0], sample_weight=[np.finfo(float).max, np.nextafter(0.0, 1.0)]
        ),
        lambda: lorenz_curve(
            [np.nextafter(0.0, 1.0), np.finfo(float).max],
            [0.0, 1.0],
            sample_weight=[np.finfo(float).max, np.nextafter(0.0, 1.0)],
        ),
        # Cancellation across 1200 binades within one column.
        lambda: lift_chart(
            [2.0**600, -(2.0**600), 2.0**100, 2.0**100],
            [2.0**600, -(2.0**600), 2.0**100, 2.0**100],
            sample_weight=[np.nextafter(0.0, 1.0)] * 2 + [2.0**-600] * 2,
            n_bins=1,
        ),
        lambda: __import__("superglm.validation", fromlist=["_"])._weighted_total(
            np.array([1.0, 2.0**-53, 2.0**-600]), np.ones(3), "test"
        ),
    ],
)
def test_validation_refuses_columns_beyond_the_reduction_range(reduce):
    # Flushing a column's tiny entries could drop a term that another column's
    # large factor makes significant, so such inputs are refused, not rounded.
    with pytest.raises(ValueError, match="validation inputs must span at most"):
        reduce()


def test_rows_without_mass_do_not_widen_the_range_check():
    from superglm.validation import _normalized_gini

    # The 1e100 target has zero weight, so it contributes nothing and must not
    # trigger the column-span refusal.
    with_outlier = _normalized_gini(
        np.array([1.0, 2.0, 1e100]), np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 0.0])
    )
    without = _normalized_gini(np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert with_outlier == without == 1.0


def test_unresolvable_gini_contraction_is_refused_not_zero():
    from superglm.validation import _normalized_gini

    # A perfect ordering whose pair mass lies below the compensated
    # contraction's error bound must not be reported as a Gini of zero.
    weights = np.array([2.0**-133, 2.0**-125, 2.0**-61, 2.0**-1])
    y = np.array([0.0, 0.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="Gini pair sums cancel below binary64 resolution"):
        _normalized_gini(y, np.array([0.0, 0.0, 1.0, 2.0]), weights)


def test_model_gini_is_refused_when_its_pair_sum_is_unresolved():
    from superglm.validation import _normalized_gini

    # The exact ratio is 1 - 2**-22 / (1 + 2**-23); the 2**-125 weight falls
    # below the model prefix's resolution, which reported a clipped 1.0.
    weights = np.array([2.0**-102, 2.0**-125, 2.0**-61, 2.0**-1])
    y = np.array([0.0, 0.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="Gini pair sums cancel below binary64 resolution"):
        _normalized_gini(y, np.array([0.0, 3.0, 2.0, 1.0]), weights)


def test_lorenz_refuses_signed_losses_whose_total_cancels_below_resolution():
    # The exact total is 2**-96 against terms of 4.6e6; the last prefix
    # computed zero, so the model curve ended at a share of 0 instead of 1.
    y = np.array([4613734.4, -1e-13, np.nextafter(1e-13, np.inf), 7e-4, -7e-4, -4613734.4])
    with pytest.raises(ValueError, match="Lorenz loss prefixes cancel below binary64 resolution"):
        lorenz_curve(y, np.arange(6.0))


@pytest.mark.parametrize(
    "chart",
    [
        lambda y, pred: lift_chart(y, pred, n_bins=1).bins["predicted"].iloc[0],
        lambda y, pred: loss_ratio_chart(y, pred, n_bins=1).bins["predicted"].iloc[0],
        lambda y, pred: double_lift_chart(y, pred, pred, n_bins=1).bins["model_avg"].iloc[0],
    ],
)
def test_charts_accept_predictions_at_the_mu_clip_floor(chart):
    from fractions import Fraction

    # 1e-50 is the positive-mean floor fitted models clip to; a two-factor
    # weighted mean may span 459 binades, well past its 166.
    predictions = np.array([0.3, 0.1, 1e-50])
    exact = float(sum(map(Fraction.from_float, predictions)) / 3)
    assert chart(np.array([1.0, 0.0, 2.0]), predictions) == pytest.approx(exact, rel=4e-16)


def test_lorenz_gini_without_exposure_uses_the_unweighted_limit():
    from superglm.validation import _normalized_gini

    # With no exposure the pair contraction multiplies three factors, not
    # five, so a target spanning 200 binades is inside its 288-binade limit.
    y, pred = np.array([1e-60, 1.0]), np.array([0.0, 1.0])
    assert lorenz_curve(y, pred).gini_ratio == _normalized_gini(y, pred) == 1.0


def test_constant_target_gini_is_zero_without_a_contraction():
    from superglm.validation import _normalized_gini

    # Weights 300 binades apart fit the totals' two-factor limit but not a
    # pair contraction's, which a constant target never needs.
    weights = np.array([1.0, 2.0**-300])
    assert _normalized_gini(np.ones(2), np.array([0.0, 1.0]), weights) == 0.0
