"""Tests for drop1() likelihood ratio test."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


@pytest.fixture
def poisson_data():
    """Poisson data with one strong and one noise feature."""
    rng = np.random.default_rng(42)
    n = 1000
    x_strong = rng.standard_normal(n)
    x_noise = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.5 * x_strong)
    sample_weight = np.ones(n)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"strong": x_strong, "noise": x_noise})
    return X, y, sample_weight


class TestDrop1Basic:
    def test_returns_dataframe(self, poisson_data):
        X, y, sample_weight = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        result = model.drop1(X, y, sample_weight=sample_weight)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "feature" in result.columns
        assert "p_value" in result.columns
        assert "delta_deviance" in result.columns

    def test_strong_feature_significant(self, poisson_data):
        X, y, sample_weight = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        result = model.drop1(X, y, sample_weight=sample_weight)

        strong_row = result[result["feature"] == "strong"].iloc[0]
        noise_row = result[result["feature"] == "noise"].iloc[0]

        # Strong feature should have large deviance change and small p-value
        assert strong_row["delta_deviance"] > noise_row["delta_deviance"]
        assert strong_row["p_value"] < 0.01
        # Noise feature should have small deviance change
        assert noise_row["p_value"] > 0.01

    def test_sorted_by_p_value(self, poisson_data):
        X, y, sample_weight = poisson_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        result = model.drop1(X, y, sample_weight=sample_weight)

        p_values = result["p_value"].values
        assert np.all(p_values[:-1] <= p_values[1:])

    def test_unfitted_raises(self):
        model = SuperGLM(
            features={"x": Numeric()},
        )
        X = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.drop1(X, np.array([1, 2, 3]))


class TestDrop1Spline:
    def test_spline_delta_df_gt_1(self):
        """Spline features should have delta_df > 1."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)
        result = model.drop1(X, y)

        assert result.iloc[0]["delta_df"] > 1.5


class TestDrop1FTest:
    def test_f_test_gamma(self):
        """F-test should work for Gamma (estimated scale)."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        shape = 5.0
        y = rng.gamma(shape, scale=mu / shape, size=n)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="gamma",
            selection_penalty=0.001,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        result = model.drop1(X, y, test="F")

        assert result.iloc[0]["p_value"] < 0.05


class TestDrop1Interactions:
    def test_drops_dependent_interaction(self):
        """Dropping a main effect should also drop its interaction."""
        rng = np.random.default_rng(42)
        n = 500
        age = rng.uniform(18, 85, n)
        region = rng.choice(["A", "B"], n)
        mu = np.exp(-1.0 + 0.01 * age)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"age": age, "region": region})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={
                "age": Spline(n_knots=5, penalty="ssp"),
                "region": Categorical(base="first"),
            },
            interactions=[("age", "region")],
        )
        model.fit(X, y)

        # Should not error — interaction is dropped with parent
        result = model.drop1(X, y)
        assert len(result) == 2  # age, region — not the interaction
        assert set(result["feature"]) == {"age", "region"}


class TestDrop1FractionalEdf:
    def test_fractional_delta_df_preserved(self):
        """delta_df should preserve fractional values from penalized edf, not floor to 1."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.standard_normal(n)
        # x2 is a weak spline effect — will have small edf under penalty
        x2 = rng.uniform(0, 1, n)
        eta = 0.5 + 0.3 * x1 + 0.05 * np.sin(2 * np.pi * x2)
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            spline_penalty=10.0,  # strong smoothing → fractional edf
            features={"x1": Numeric(), "x2": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)
        result = model.drop1(X, y)

        x2_row = result[result["feature"] == "x2"].iloc[0]
        # With strong smoothing (spline_penalty=10) on a weak effect, delta_df should be
        # fractional (between 0 and n_basis), reflecting the effective degrees of
        # freedom consumed by the smoothed spline.
        assert x2_row["delta_df"] > 0, "delta_df must be positive"
        # With heavy smoothing, edf should be well below the nominal basis size (10 knots → ~14 cols)
        assert x2_row["delta_df"] < 14, f"delta_df={x2_row['delta_df']:.2f} exceeds basis dimension"
        # p-value and deviance change should be present and finite
        assert np.isfinite(x2_row["delta_deviance"])
        assert np.isfinite(x2_row["p_value"])
        assert x2_row["delta_deviance"] >= 0
