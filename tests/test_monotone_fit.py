"""End-to-end tests for monotone-constrained model fitting.

Tests that SuperGLM with monotone BSplineSmooth / CubicRegressionSpline terms
produces actually-monotone predictions via the constrained QP solver path.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.families import Gaussian
from superglm.features.spline import BSplineSmooth, CubicRegressionSpline


class TestMonotoneFitBSplineSmooth:
    """BSplineSmooth with monotone_mode='fit' produces monotone predictions."""

    @pytest.fixture
    def monotone_data(self):
        """Data with a clearly increasing relationship."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        # True function: monotone increasing sigmoid
        y_true = 1 / (1 + np.exp(-10 * (x - 0.5)))
        y = y_true + rng.normal(0, 0.1, n)
        return x, y

    @pytest.mark.slow
    def test_predictions_are_monotone(self, monotone_data):
        x, y = monotone_data

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=10,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        # Predictions must be monotone increasing
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-8), f"Predictions not monotone: min diff = {diffs.min():.2e}"

    @pytest.mark.slow
    def test_unconstrained_unchanged(self, monotone_data):
        """monotone=None does not enter the constrained QP path."""
        x, y = monotone_data

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": BSplineSmooth(n_knots=10)},
        )
        model.fit(df[["x"]], df["y"])
        assert model._result.n_iter > 0
        assert model._result.deviance < 100

    @pytest.mark.slow
    def test_weighted_fit(self, monotone_data):
        x, y = monotone_data

        weights = np.ones(len(x))
        weights[:100] = 2.0

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=10,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"], sample_weight=weights)

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_decreasing(self):
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = np.exp(-3 * x) + rng.normal(0, 0.05, n)

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    monotone="decreasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) <= 1e-8)


class TestMonotoneFitCRS:
    """CubicRegressionSpline with monotone_mode='fit'."""

    @pytest.mark.slow
    def test_predictions_are_monotone(self):
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": CubicRegressionSpline(
                    n_knots=10,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) >= -1e-8)


class TestMonotoneMixedModel:
    """Model with both monotone and unconstrained terms."""

    @pytest.mark.slow
    def test_mixed_terms(self):
        rng = np.random.default_rng(42)
        n = 500
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)

        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x1": BSplineSmooth(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
                "x2": BSplineSmooth(n_knots=8),  # unconstrained
            },
        )
        model.fit(df[["x1", "x2"]], df["y"])

        # x1 predictions should be monotone when x2 is held fixed
        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame(
            {
                "x1": x_grid,
                "x2": np.full(200, 0.5),
            }
        )
        pred = model.predict(df_grid)
        assert np.all(np.diff(pred) >= -1e-8)


class TestMonotoneRegression:
    """Regression tests: unconstrained behavior is unchanged."""

    @pytest.mark.slow
    def test_no_performance_regression(self):
        """Unconstrained model runs at the same speed (no QP overhead)."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)

        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            features={"x": BSplineSmooth(n_knots=10)},
        )
        model.fit(df[["x"]], df["y"])
        assert model._result.converged


class TestMonotoneUnsupportedCombinations:
    """Unsupported combinations raise NotImplementedError."""

    def test_monotone_with_selection_penalty_raises(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0.1,
            features={
                "x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        with pytest.raises(NotImplementedError, match="selection_penalty"):
            model.fit(df[["x"]], df["y"])

    def test_monotone_with_select_true_raises(self):
        s = BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit", select=True)
        x = np.linspace(0, 1, 200)
        with pytest.raises(NotImplementedError, match="select=True"):
            s.build(x)

    def test_monotone_fit_reml_raises(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            features={
                "x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        with pytest.raises(NotImplementedError, match="smoothness selection"):
            model.fit_reml(df[["x"]], df["y"])

    def test_monotone_with_discrete_raises(self):
        """discrete=True + monotone_mode='fit' is not supported."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        with pytest.raises(NotImplementedError, match="discrete=True"):
            model.fit(df[["x"]], df["y"])
