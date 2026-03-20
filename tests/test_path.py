"""Tests for SuperGLM.fit_path() regularization path."""

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    GroupElasticNet,
    GroupLasso,
    Numeric,
    PathResult,
    Poisson,
    Spline,
    SuperGLM,
)


@pytest.fixture
def poisson_data():
    """Synthetic Poisson dataset with 3 features."""
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.uniform(0, 10, n)
    x2 = rng.choice(["A", "B", "C"], n)
    x3 = rng.normal(0, 1, n)
    eta = 0.1 * np.sin(x1) - 0.3 * (x2 == "B") + 0.1 * x3
    y = rng.poisson(np.exp(eta))
    sample_weight = np.ones(n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return df, y.astype(float), sample_weight


def _make_model(lambda1=None):
    return SuperGLM(
        family=Poisson(),
        penalty=GroupLasso(lambda1=lambda1),
        features={
            "x1": Spline(n_knots=8, penalty="ssp"),
            "x2": Categorical(),
            "x3": Numeric(),
        },
    )


class TestPathResult:
    def test_path_returns_correct_shape(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model()
        n_lambda = 20
        result = m.fit_path(df, y, sample_weight=w, n_lambda=n_lambda, lambda_ratio=1e-2)

        assert isinstance(result, PathResult)
        assert result.lambda_seq.shape == (n_lambda,)
        assert result.coef_path.shape[0] == n_lambda
        assert result.intercept_path.shape == (n_lambda,)
        assert result.deviance_path.shape == (n_lambda,)
        assert result.n_iter_path.shape == (n_lambda,)
        assert result.converged_path.shape == (n_lambda,)
        # Lambda sequence should be decreasing
        assert np.all(np.diff(result.lambda_seq) < 0)

    def test_increasing_activity_along_path(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model()
        result = m.fit_path(df, y, sample_weight=w, n_lambda=20, lambda_ratio=1e-3)

        # Coefficient norms should generally increase as lambda decreases
        norms = np.array([np.linalg.norm(c) for c in result.coef_path])
        assert norms[-1] >= norms[0], (
            f"Norm at lambda_min ({norms[-1]:.4f}) should be >= norm at lambda_max ({norms[0]:.4f})"
        )

    def test_lambda_min_nonzero(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model()
        result = m.fit_path(df, y, sample_weight=w, n_lambda=20, lambda_ratio=1e-3)

        # Last lambda is small — some coefficients should be nonzero
        assert np.any(np.abs(result.coef_path[-1]) > 1e-6)

    def test_warm_start_fewer_iters(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model()
        result = m.fit_path(df, y, sample_weight=w, n_lambda=20, lambda_ratio=1e-3)

        # Warm-started iterations (all but first) should average fewer iters
        # than the first cold-start iteration
        avg_warm = result.n_iter_path[1:].mean()
        first_cold = result.n_iter_path[0]
        assert avg_warm <= first_cold, (
            f"Warm-started avg {avg_warm:.1f} should be <= cold-start {first_cold}"
        )

    def test_deviance_monotone_decreasing(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model()
        result = m.fit_path(df, y, sample_weight=w, n_lambda=30, lambda_ratio=1e-3)

        # Deviance should weakly decrease as lambda decreases (model gets more flexible)
        diffs = np.diff(result.deviance_path)
        # Allow small numerical violations
        assert np.all(diffs < 1.0), (
            f"Deviance should decrease along path, max increase: {diffs.max():.4f}"
        )

    def test_custom_lambda_seq(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model()
        custom_lambdas = np.array([1.0, 0.5, 0.1, 0.01])
        result = m.fit_path(df, y, sample_weight=w, lambda_seq=custom_lambdas)

        assert len(result.lambda_seq) == 4
        np.testing.assert_array_equal(result.lambda_seq, custom_lambdas)
        assert result.coef_path.shape[0] == 4

    def test_fit_sets_last_result(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model()
        result = m.fit_path(df, y, sample_weight=w, n_lambda=10, lambda_ratio=1e-2)

        # After fit_path, predict() should use the last (least-regularized) fit
        preds = m.predict(df)
        assert preds.shape == (len(y),)
        # Coefficients should match the last path entry
        np.testing.assert_array_equal(m.result.beta, result.coef_path[-1])
        assert m.result.intercept == result.intercept_path[-1]

    def test_fit_still_works(self, poisson_data):
        df, y, w = poisson_data
        m = _make_model(lambda1=0.01)
        m.fit(df, y, sample_weight=w)

        assert m.result.converged
        assert m.result.deviance > 0
        preds = m.predict(df)
        assert preds.shape == (len(y),)
        assert np.all(preds > 0)

    def test_group_elastic_net_path(self, poisson_data):
        """GroupElasticNet works with warm-started reg path."""
        df, y, w = poisson_data
        m = SuperGLM(
            family=Poisson(),
            penalty=GroupElasticNet(alpha=0.7),
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Categorical(),
                "x3": Numeric(),
            },
        )

        result = m.fit_path(df, y, sample_weight=w, n_lambda=15, lambda_ratio=1e-2)

        assert isinstance(result, PathResult)
        assert result.lambda_seq.shape == (15,)
        # Norms should increase as lambda decreases
        norms = np.array([np.linalg.norm(c) for c in result.coef_path])
        assert norms[-1] >= norms[0]
        # Warm-started iterations should average fewer than cold-start
        avg_warm = result.n_iter_path[1:].mean()
        assert avg_warm <= result.n_iter_path[0]
