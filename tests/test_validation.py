"""Tests for superglm.validation — actuarial validation toolkit (T6-T12)."""

from __future__ import annotations

import importlib.util

import matplotlib
import numpy as np
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
