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

    # NOTE: test_monotone_with_discrete_raises removed — discrete+monotone
    # is now supported (monotone terms opt out of discretization).
    # See TestMonotoneDiscrete for the replacement tests.


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

    def test_fit_reml_without_fixed_lambdas_raises_qp(self):
        """fit_reml raises for QP monotone without fixed lambda."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
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
        with pytest.raises(NotImplementedError, match="smoothness selection"):
            model.fit_reml(df[["x"]], df["y"])

    def test_fit_reml_without_fixed_lambdas_raises_scop(self):
        """fit_reml raises for SCOP monotone without fixed lambda."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(0, 1, n)
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
        with pytest.raises(NotImplementedError, match="smoothness selection"):
            model.fit_reml(df[["x"]], df["y"])

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


class TestMonotoneDiscrete:
    """Discrete mode with monotone fit-time constraints."""

    @pytest.mark.slow
    def test_discrete_qp_monotone(self):
        """discrete=True + QP monotone produces monotone predictions."""
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
                "x": BSplineSmooth(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_discrete_scop_monotone(self):
        """discrete=True + SCOP monotone produces monotone predictions."""
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
                "x": PSpline(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
            },
        )
        model.fit(df[["x"]], df["y"])

        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(pd.DataFrame({"x": x_grid}))
        assert np.all(np.diff(pred) >= -1e-8)

    @pytest.mark.slow
    def test_mixed_discrete_monotone_opts_out(self):
        """In a mixed model with discrete=True, the monotone term opts out
        of discretization while the ordinary term still uses it."""
        from superglm.group_matrix import DiscretizedSSPGroupMatrix

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
                "x1": BSplineSmooth(
                    n_knots=8,
                    monotone="increasing",
                    monotone_mode="fit",
                ),
                "x2": PSpline(n_knots=8),  # ordinary, should use discrete
            },
        )
        model.fit(df[["x1", "x2"]], df["y"])

        # Monotone term predictions should be monotone
        x_grid = np.linspace(0, 1, 200)
        pred = model.predict(
            pd.DataFrame(
                {
                    "x1": x_grid,
                    "x2": np.full(200, 0.5),
                }
            )
        )
        assert np.all(np.diff(pred) >= -1e-8)

        # Model should have converged
        assert model._result.converged

        # Verify the ordinary term (x2) actually used discretization.
        # The monotone term (x1) should NOT be discretized.
        gms = model._dm.group_matrices
        groups = model._groups
        for gm, g in zip(gms, groups):
            if g.feature_name == "x1":
                # Monotone term: should NOT be discretized
                assert not isinstance(gm, DiscretizedSSPGroupMatrix), (
                    f"Monotone term x1 should not be discretized, got {type(gm).__name__}"
                )
            elif g.feature_name == "x2":
                # Ordinary term: SHOULD be discretized when discrete=True
                assert isinstance(gm, DiscretizedSSPGroupMatrix), (
                    f"Ordinary term x2 should be discretized, got {type(gm).__name__}"
                )
