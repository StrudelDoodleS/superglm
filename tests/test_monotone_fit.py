"""End-to-end tests for monotone-constrained model fitting.

Tests that SuperGLM with monotone BSplineSmooth / CubicRegressionSpline terms
produces actually-monotone predictions via the constrained QP solver path.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.families import Gaussian
from superglm.features.spline import BSplineSmooth, CubicRegressionSpline, PSpline
from superglm.types import LambdaPolicy


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


class TestSCOPPenaltyInEDF:
    """SCOP penalty contributes to EDF and information criteria."""

    @pytest.mark.slow
    def test_group_edf_changes_with_lambda(self):
        """Per-group EDF should also respond to penalty changes."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model_low = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=0.01,
            features={"x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")},
        )
        model_low.fit(df[["x"]], df["y"])

        model_high = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=100.0,
            features={"x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")},
        )
        model_high.fit(df[["x"]], df["y"])

        # Group EDF should also respond (not pinned)
        edf_low = sum(model_low._group_edf.values())
        edf_high = sum(model_high._group_edf.values())
        assert edf_high < edf_low, (
            f"Group EDF should decrease with higher penalty: low={edf_low:.2f}, high={edf_high:.2f}"
        )

    @pytest.mark.slow
    def test_edf_changes_with_lambda(self):
        """Increasing lambda should reduce effective degrees of freedom."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model_low = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=0.01,
            features={"x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")},
        )
        model_low.fit(df[["x"]], df["y"])

        model_high = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            spline_penalty=100.0,
            features={"x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")},
        )
        model_high.fit(df[["x"]], df["y"])

        # Higher penalty should give lower EDF
        assert model_high._result.effective_df < model_low._result.effective_df


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

    def test_monotone_fit_reml_qp_passthrough(self):
        """QP monotone in fit_reml works via passthrough heuristic."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged

    def test_scop_monotone_with_discrete_works(self):
        """discrete=True + SCOP monotone_mode='fit' is now supported."""
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
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 50)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)


# ── PSpline SCOP engine tests ─────────────────────────────────────────────────


class TestMonotoneFitPSpline:
    """PSpline with monotone_mode='fit' uses SCOP engine."""

    @pytest.mark.slow
    def test_predictions_are_monotone(self):
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"

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
            features={"x": PSpline(n_knots=8, monotone="decreasing", monotone_mode="fit")},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) <= 1e-8)

    @pytest.mark.slow
    def test_mixed_scop_and_unconstrained(self):
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
                "x1": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8),
            },
        )
        model.fit(df[["x1", "x2"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x1": x_grid, "x2": np.full(200, 0.5)}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_unconstrained_pspline_unchanged(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={"x": PSpline(n_knots=8)},
        )
        model.fit(df[["x"]], df["y"])
        assert model._result.converged

    def test_scop_plus_qp_raises(self):
        """SCOP + QP monotone in same model raises NotImplementedError."""
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = x1 + x2 + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x1": PSpline(n_knots=5, monotone="increasing", monotone_mode="fit"),
                "x2": BSplineSmooth(n_knots=5, monotone="increasing", monotone_mode="fit"),
            },
        )
        with pytest.raises(NotImplementedError, match="SCOP.*QP"):
            model.fit(df[["x1", "x2"]], df["y"])


# ── fit_reml with fixed lambda tests ────────────────────────────────────────────


class TestMonotoneFixedLambdaREML:
    """fit_reml() works with monotone terms when all lambdas are fixed."""

    @pytest.mark.slow
    def test_fit_reml_with_fixed_lambda_policy_qp(self):
        """BSplineSmooth monotone with fixed lambda_policy works in fit_reml."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        # Predictions must be monotone increasing
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"

        # Model should have converged
        assert model._result.converged

        # REML lambdas should be set
        assert model._reml_lambdas is not None

    @pytest.mark.slow
    def test_fit_reml_with_fixed_lambda_policy_scop(self):
        """PSpline monotone with fixed lambda_policy works in fit_reml."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 1 / (1 + np.exp(-10 * (x - 0.5))) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": PSpline(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        # Predictions must be monotone increasing
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"

        assert model._result.converged
        assert model._reml_lambdas is not None

    def test_fit_reml_without_fixed_lambdas_works_qp(self):
        """fit_reml works for QP monotone via passthrough heuristic."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged

    @pytest.mark.slow
    def test_fit_reml_without_fixed_lambdas_works_scop(self):
        """fit_reml works for SCOP monotone with auto lambda (Phase 5a)."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit"),
            },
        )
        model.fit_reml(df[["x"]], y)
        assert model._result.converged
        assert model._reml_lambdas is not None

    @pytest.mark.slow
    def test_fit_reml_unchanged_without_monotone(self):
        """Normal REML still works without monotone terms."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            features={"x": BSplineSmooth(n_knots=10)},
        )
        model.fit_reml(df[["x"]], df["y"])
        assert model._result.converged


# ── discrete + monotone + fit_reml intersection tests ─────────────────────────


class TestDiscreteMonotoneREML:
    """The full intersection: discrete=True + monotone + fit_reml(fixed lambda).

    Tests at 250k rows to exercise the discretization performance path
    at meaningful scale.
    """

    @pytest.mark.slow
    def test_discrete_qp_monotone_fit_reml(self):
        """BSplineSmooth: discrete + monotone + fit_reml with fixed lambda."""
        rng = np.random.default_rng(42)
        n = 250_000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": BSplineSmooth(
                    n_knots=10,
                    monotone="increasing",
                    monotone_mode="fit",
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"
        assert model._result.converged
        assert model._reml_lambdas is not None

        # Verify the term is actually discretized
        from superglm.group_matrix import DiscretizedSSPGroupMatrix

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSSPGroupMatrix), (
            f"Expected DiscretizedSSPGroupMatrix, got {type(gm).__name__}"
        )

    @pytest.mark.slow
    def test_discrete_scop_monotone_fit_reml(self):
        """PSpline: discrete + monotone + fit_reml with fixed lambda."""
        rng = np.random.default_rng(42)
        n = 250_000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": PSpline(
                    n_knots=10,
                    monotone="increasing",
                    monotone_mode="fit",
                    lambda_policy=LambdaPolicy(mode="fixed", value=1.0),
                ),
            },
        )
        model.fit_reml(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8), f"min diff = {np.diff(pred).min():.2e}"
        assert model._result.converged
        assert model._reml_lambdas is not None

        # Verify the term used the discretized SCOP path
        from superglm.group_matrix import DiscretizedSCOPGroupMatrix

        gm = model._dm.group_matrices[0]
        assert isinstance(gm, DiscretizedSCOPGroupMatrix), (
            f"Expected DiscretizedSCOPGroupMatrix, got {type(gm).__name__}"
        )


# ── summary() monotone engine display tests ──────────────────────────────────────


class TestSummaryMonotoneEngine:
    """summary() shows monotone engine type (qp/scop) alongside direction."""

    @pytest.mark.slow
    def test_summary_shows_engine_for_qp(self):
        """BSplineSmooth monotone summary shows 'qp' engine."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": BSplineSmooth(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])
        summary_str = str(model.summary())

        assert "qp" in summary_str.lower()
        assert "mono=increasing (qp)" in summary_str

    @pytest.mark.slow
    def test_summary_shows_engine_for_scop(self):
        """PSpline monotone summary shows 'scop' engine."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})

        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            features={
                "x": PSpline(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])
        summary_str = str(model.summary())

        assert "scop" in summary_str.lower()
        assert "mono=increasing (scop)" in summary_str


class TestDiscreteQPMonotone:
    """discrete=True with QP monotone (BSplineSmooth/CRS)."""

    @pytest.mark.slow
    def test_discrete_bsplinesmooth_monotone(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit")},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_discrete_crs_monotone(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x": CubicRegressionSpline(n_knots=8, monotone="increasing", monotone_mode="fit")
            },
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_discrete_qp_mixed_model(self):
        """Monotone discrete + ordinary discrete in same model."""
        rng = np.random.default_rng(42)
        n = 1000
        x1 = np.sort(rng.uniform(0, 1, n))
        x2 = rng.uniform(0, 1, n)
        y = 2 * x1 + np.sin(2 * np.pi * x2) + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={
                "x1": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit"),
                "x2": PSpline(n_knots=8),
            },
        )
        model.fit(df[["x1", "x2"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x1": x_grid, "x2": np.full(200, 0.5)}))
        assert np.all(np.diff(pred) >= -1e-8)

        # Verify both terms are discretized
        from superglm.group_matrix import DiscretizedSSPGroupMatrix

        gms = model._dm.group_matrices
        groups = model._groups
        for gm, g in zip(gms, groups):
            assert isinstance(gm, DiscretizedSSPGroupMatrix), (
                f"Term {g.feature_name} should be discretized, got {type(gm).__name__}"
            )

    @pytest.mark.slow
    def test_discrete_vs_nondiscrete_parity(self):
        """Discrete and non-discrete monotone fits should give similar results."""
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})

        model_dense = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=False,
            features={"x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit")},
        )
        model_dense.fit(df[["x"]], df["y"])
        pred_dense = model_dense.predict(df_grid)

        model_disc = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": BSplineSmooth(n_knots=8, monotone="increasing", monotone_mode="fit")},
        )
        model_disc.fit(df[["x"]], df["y"])
        pred_disc = model_disc.predict(df_grid)

        np.testing.assert_allclose(pred_dense, pred_disc, atol=0.05)


class TestDiscreteSCOPMonotone:
    @pytest.mark.slow
    def test_discrete_pspline_monotone(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        model = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")},
        )
        model.fit(df[["x"]], df["y"])
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_discrete_scop_vs_nondiscrete_parity(self):
        rng = np.random.default_rng(42)
        n = 500
        x = np.sort(rng.uniform(0, 1, n))
        y = 2 * x + rng.normal(0, 0.2, n)
        df = pd.DataFrame({"x": x, "y": y})
        x_grid = np.linspace(0, 1, 200)
        df_grid = pd.DataFrame({"x": x_grid})

        model_dense = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=False,
            features={"x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")},
        )
        model_dense.fit(df[["x"]], df["y"])
        pred_dense = model_dense.predict(df_grid)

        model_disc = SuperGLM(
            family=Gaussian(),
            selection_penalty=0,
            discrete=True,
            features={"x": PSpline(n_knots=8, monotone="increasing", monotone_mode="fit")},
        )
        model_disc.fit(df[["x"]], df["y"])
        pred_disc = model_disc.predict(df_grid)

        np.testing.assert_allclose(pred_dense, pred_disc, atol=0.05)
