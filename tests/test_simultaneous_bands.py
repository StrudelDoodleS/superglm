"""Tests for simultaneous confidence bands."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


@pytest.fixture
def spline_model():
    """Fitted Poisson model with a spline feature."""
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.uniform(0, 10, n)
    mu = np.exp(0.5 + 0.3 * np.sin(x))
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="poisson",
        lambda1=0.001,
        features={"x": Spline(n_knots=10, penalty="ssp")},
    )
    model.fit(X, y)
    return model


class TestSimultaneousBands:
    def test_returns_dataframe(self, spline_model):
        result = spline_model.simultaneous_bands("x")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 200
        expected_cols = {
            "x",
            "log_relativity",
            "relativity",
            "se",
            "ci_lower_pointwise",
            "ci_upper_pointwise",
            "ci_lower_simultaneous",
            "ci_upper_simultaneous",
        }
        assert set(result.columns) == expected_cols

    def test_simultaneous_wider_than_pointwise(self, spline_model):
        result = spline_model.simultaneous_bands("x")
        # Simultaneous bands should always be wider
        sim_width = result["ci_upper_simultaneous"] - result["ci_lower_simultaneous"]
        pw_width = result["ci_upper_pointwise"] - result["ci_lower_pointwise"]
        assert np.all(sim_width >= pw_width - 1e-10)

    def test_smaller_alpha_gives_wider_bands(self, spline_model):
        r05 = spline_model.simultaneous_bands("x", alpha=0.05)
        r01 = spline_model.simultaneous_bands("x", alpha=0.01)

        w05 = (r05["ci_upper_simultaneous"] - r05["ci_lower_simultaneous"]).mean()
        w01 = (r01["ci_upper_simultaneous"] - r01["ci_lower_simultaneous"]).mean()
        assert w01 > w05

    def test_reproducible_with_seed(self, spline_model):
        r1 = spline_model.simultaneous_bands("x", seed=42)
        r2 = spline_model.simultaneous_bands("x", seed=42)
        np.testing.assert_array_equal(
            r1["ci_lower_simultaneous"].values,
            r2["ci_lower_simultaneous"].values,
        )

    def test_non_spline_raises(self):
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({"x": rng.standard_normal(n)})
        y = rng.poisson(1.0, n).astype(float)

        model = SuperGLM(
            family="poisson",
            lambda1=0.001,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        with pytest.raises(TypeError, match="spline"):
            model.simultaneous_bands("x")

    def test_unfitted_raises(self):
        model = SuperGLM(features={"x": Spline(n_knots=5)})
        with pytest.raises(RuntimeError, match="fitted"):
            model.simultaneous_bands("x")
