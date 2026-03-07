"""Tests for spline discretization impact analysis."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    DiscretizationResult,
    Numeric,
    Polynomial,
    Spline,
    SuperGLM,
    discretization_impact,
)
from superglm.distributions import Poisson
from superglm.penalties.group_lasso import GroupLasso


@pytest.fixture
def fitted_model():
    """Fit a model with spline, categorical, numeric, and polynomial features."""
    rng = np.random.default_rng(42)
    n = 3000
    x_spline = rng.uniform(18, 80, n)
    x_cat = rng.choice(["A", "B", "C"], n)
    x_num = rng.normal(0, 1, n)
    x_poly = rng.uniform(0, 10, n)
    eta = 0.02 * (x_spline - 40) ** 2 / 400 - 0.3 * (x_cat == "B") + 0.1 * x_num
    y = rng.poisson(np.exp(eta)).astype(float)
    exposure = rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame(
        {
            "age": x_spline,
            "region": x_cat,
            "score": x_num,
            "density": x_poly,
        }
    )

    m = SuperGLM(
        family=Poisson(),
        penalty=GroupLasso(lambda1=0.01),
        features={
            "age": Spline(n_knots=10, penalty="ssp"),
            "region": Categorical(),
            "score": Numeric(),
            "density": Polynomial(degree=3),
        },
    )
    m.fit(df, y, exposure=exposure)
    return m, df, y, exposure


class TestRatingTableSchema:
    def test_correct_columns(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10)
        expected_cols = {"bin_from", "bin_to", "relativity", "log_relativity", "n_obs", "exposure"}
        for name, table in result.tables.items():
            assert set(table.columns) == expected_cols, f"Wrong columns for {name}"

    def test_n_bins_rows(self, fitted_model):
        m, df, y, w = fitted_model
        for n_bins in [5, 10, 20]:
            result = m.discretization_impact(df, y, exposure=w, n_bins=n_bins)
            for name, table in result.tables.items():
                assert len(table) <= n_bins, (
                    f"Feature {name}: got {len(table)} rows for n_bins={n_bins}"
                )
                assert len(table) >= 1


class TestFeatureFiltering:
    def test_non_spline_features_excluded(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w)
        # Only spline and polynomial features should appear
        assert "region" not in result.tables
        assert "score" not in result.tables
        # Spline and polynomial should be present
        assert "age" in result.tables
        assert "density" in result.tables

    def test_features_param_restricts(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, features=["age"])
        assert list(result.tables.keys()) == ["age"]

    def test_invalid_feature_raises(self, fitted_model):
        m, df, y, w = fitted_model
        with pytest.raises(ValueError, match="Unknown feature"):
            m.discretization_impact(df, y, exposure=w, features=["nonexistent"])

    def test_non_continuous_feature_raises(self, fitted_model):
        m, df, y, w = fitted_model
        with pytest.raises(ValueError, match="not a spline or polynomial"):
            m.discretization_impact(df, y, exposure=w, features=["region"])


class TestBinCoverage:
    def test_bins_span_data_range(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10)
        for name, table in result.tables.items():
            x_vals = df[name].values
            assert table["bin_from"].iloc[0] <= x_vals.min() + 1e-10
            assert table["bin_to"].iloc[-1] >= x_vals.max() - 1e-10

    def test_no_gaps_between_bins(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10)
        for name, table in result.tables.items():
            if len(table) > 1:
                # Each bin_from should equal the previous bin_to
                for i in range(1, len(table)):
                    assert table["bin_to"].iloc[i - 1] == table["bin_from"].iloc[i], (
                        f"Gap between bins {i - 1} and {i} for {name}"
                    )

    def test_all_observations_assigned(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10)
        n = len(df)
        for name, table in result.tables.items():
            assert table["n_obs"].sum() == n, (
                f"Total n_obs={table['n_obs'].sum()} != {n} for {name}"
            )


class TestPredictions:
    def test_predictions_shape(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w)
        assert result.predictions.shape == (len(df),)
        assert result.original_predictions.shape == (len(df),)

    def test_predictions_positive(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w)
        assert np.all(result.predictions > 0)
        assert np.all(result.original_predictions > 0)

    def test_original_matches_model_predict(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w)
        expected = m.predict(df)
        np.testing.assert_allclose(result.original_predictions, expected, rtol=1e-10)


class TestMetrics:
    def test_all_metric_keys_present(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w)
        expected_keys = {
            "deviance_original",
            "deviance_discretized",
            "deviance_change",
            "deviance_change_pct",
            "max_abs_prediction_change_pct",
            "mean_abs_prediction_change_pct",
            "prediction_correlation",
        }
        assert set(result.metrics.keys()) == expected_keys

    def test_deviance_change_consistent(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w)
        expected_change = (
            result.metrics["deviance_discretized"] - result.metrics["deviance_original"]
        )
        assert abs(result.metrics["deviance_change"] - expected_change) < 1e-10

    def test_high_correlation(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=20)
        assert result.metrics["prediction_correlation"] > 0.95


class TestSmallImpactAtHighBinCount:
    def test_tiny_deviance_change_at_50_bins(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=50)
        assert abs(result.metrics["deviance_change_pct"]) < 1.0, (
            f"Deviance change {result.metrics['deviance_change_pct']:.4f}% too large at 50 bins"
        )

    def test_high_correlation_at_50_bins(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=50)
        assert result.metrics["prediction_correlation"] > 0.99


class TestConvenienceMethod:
    def test_model_method_matches_function(self, fitted_model):
        m, df, y, w = fitted_model
        r1 = m.discretization_impact(df, y, exposure=w, n_bins=10)
        r2 = discretization_impact(m, df, y, exposure=w, n_bins=10)
        np.testing.assert_array_equal(r1.predictions, r2.predictions)
        assert r1.metrics == r2.metrics

    def test_returns_discretization_result(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w)
        assert isinstance(result, DiscretizationResult)


class TestBinStrategy:
    def test_uniform_equal_width(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10, bin_strategy="uniform")
        for name, table in result.tables.items():
            widths = (table["bin_to"] - table["bin_from"]).values
            # All bins should have the same width
            np.testing.assert_allclose(
                widths, widths[0], rtol=1e-10, err_msg=f"Unequal bin widths for {name}"
            )

    def test_uniform_produces_result(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10, bin_strategy="uniform")
        assert isinstance(result, DiscretizationResult)
        assert result.predictions.shape == (len(df),)

    def test_winsorized_structure(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10, bin_strategy="winsorized")
        for name, table in result.tables.items():
            x_vals = df[name].values
            # First bin starts at data min, last ends at data max
            assert table["bin_from"].iloc[0] <= x_vals.min() + 1e-10
            assert table["bin_to"].iloc[-1] >= x_vals.max() - 1e-10
            # Should produce at least 3 bins (left tail, interior, right tail)
            assert len(table) >= 3, f"Expected >= 3 bins for {name}"
            # Bins should be contiguous
            if len(table) > 1:
                for i in range(1, len(table)):
                    assert table["bin_to"].iloc[i - 1] == table["bin_from"].iloc[i]

    def test_winsorized_produces_result(self, fitted_model):
        m, df, y, w = fitted_model
        result = m.discretization_impact(df, y, exposure=w, n_bins=10, bin_strategy="winsorized")
        assert isinstance(result, DiscretizationResult)

    def test_invalid_strategy_raises(self, fitted_model):
        m, df, y, w = fitted_model
        with pytest.raises(ValueError, match="Unknown bin_strategy"):
            m.discretization_impact(df, y, exposure=w, bin_strategy="bogus")

    def test_default_is_exposure_quantile(self, fitted_model):
        m, df, y, w = fitted_model
        r_default = m.discretization_impact(df, y, exposure=w, n_bins=10)
        r_explicit = m.discretization_impact(
            df, y, exposure=w, n_bins=10, bin_strategy="exposure_quantile"
        )
        np.testing.assert_array_equal(r_default.predictions, r_explicit.predictions)

    def test_all_strategies_cover_all_obs(self, fitted_model):
        m, df, y, w = fitted_model
        n = len(df)
        for strategy in ["exposure_quantile", "uniform", "winsorized"]:
            result = m.discretization_impact(df, y, exposure=w, n_bins=10, bin_strategy=strategy)
            for name, table in result.tables.items():
                assert table["n_obs"].sum() == n, f"strategy={strategy}, {name}: n_obs sum != {n}"


class TestDefaultExposure:
    def test_works_without_exposure(self, fitted_model):
        m, df, y, _ = fitted_model
        result = m.discretization_impact(df, y, n_bins=10)
        assert isinstance(result, DiscretizationResult)
        assert result.predictions.shape == (len(df),)
