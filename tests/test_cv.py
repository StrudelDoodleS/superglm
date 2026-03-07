"""Tests for cross-validation lambda selection."""

import numpy as np
import pandas as pd
import pytest

from superglm import CVResult, GroupLasso, Spline, SuperGLM
from superglm.cv import _select_lambda


@pytest.fixture
def cv_data():
    """Synthetic Poisson data for CV tests."""
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.uniform(0, 100, n)
    x2 = rng.choice(["A", "B", "C"], n)
    mu = np.exp(-1.0 + 0.01 * x1 + 0.3 * (x2 == "B") + 0.5 * (x2 == "C"))
    y = rng.poisson(mu).astype(float)
    exposure = rng.uniform(0.5, 2.0, n)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    return df, y, exposure


class TestFitCV:
    def test_smoke(self, cv_data):
        """fit_cv runs and returns CVResult with correct fields."""
        df, y, exposure = cv_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(lambda1=0.01),
            features={"x1": Spline(n_knots=5, penalty="ssp")},
        )
        result = model.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            n_lambda=10,
            random_state=42,
        )
        assert isinstance(result, CVResult)
        assert len(result.lambda_seq) == 10
        assert len(result.mean_cv_deviance) == 10
        assert len(result.se_cv_deviance) == 10
        assert result.fold_deviance.shape == (3, 10)
        assert result.best_lambda > 0
        assert result.best_lambda_1se > 0
        assert 0 <= result.best_index < 10
        assert 0 <= result.best_index_1se < 10

    def test_1se_more_regularised(self, cv_data):
        """best_lambda_1se >= best_lambda (more regularised)."""
        df, y, exposure = cv_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(),
            splines=["x1"],
        )
        result = model.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            n_lambda=20,
            rule="min",
            random_state=42,
        )
        # 1se lambda is always >= min lambda (more regularised)
        assert result.best_lambda_1se >= result.best_lambda - 1e-12

    def test_refit_true(self, cv_data):
        """After fit_cv(refit=True), predict works."""
        df, y, exposure = cv_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(),
            splines=["x1"],
        )
        model.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            n_lambda=10,
            refit=True,
            random_state=42,
        )
        preds = model.predict(df[["x1"]])
        assert preds.shape == (len(y),)
        assert np.all(preds > 0)

    def test_refit_false(self, cv_data):
        """After fit_cv(refit=False), model is not fitted."""
        df, y, exposure = cv_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(),
            splines=["x1"],
        )
        model.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            n_lambda=10,
            refit=False,
            random_state=42,
        )
        with pytest.raises(RuntimeError, match="Not fitted"):
            model.predict(df[["x1"]])

    def test_rule_min_vs_1se(self, cv_data):
        """rule='min' and rule='1se' may select different lambdas."""
        df, y, exposure = cv_data
        model_min = SuperGLM(
            family="poisson",
            penalty=GroupLasso(),
            splines=["x1"],
        )
        result_min = model_min.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            n_lambda=20,
            rule="min",
            random_state=42,
        )
        model_1se = SuperGLM(
            family="poisson",
            penalty=GroupLasso(),
            splines=["x1"],
        )
        result_1se = model_1se.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            n_lambda=20,
            rule="1se",
            random_state=42,
        )
        # 1se selects a lambda >= min (more regularised)
        assert result_1se.best_lambda >= result_min.best_lambda - 1e-12

    def test_custom_lambda_seq(self, cv_data):
        """Explicit lambda_seq is respected."""
        df, y, exposure = cv_data
        lam_seq = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(),
            splines=["x1"],
        )
        result = model.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            lambda_seq=lam_seq,
            random_state=42,
        )
        np.testing.assert_array_equal(result.lambda_seq, lam_seq)
        assert result.fold_deviance.shape[1] == 5

    def test_reproducibility(self, cv_data):
        """Same random_state gives same result."""
        df, y, exposure = cv_data
        results = []
        for _ in range(2):
            model = SuperGLM(
                family="poisson",
                penalty=GroupLasso(),
                splines=["x1"],
            )
            r = model.fit_cv(
                df[["x1"]],
                y,
                exposure=exposure,
                n_folds=3,
                n_lambda=10,
                refit=False,
                random_state=42,
            )
            results.append(r)
        np.testing.assert_array_equal(results[0].fold_deviance, results[1].fold_deviance)
        assert results[0].best_lambda == results[1].best_lambda

    def test_deviance_finite(self, cv_data):
        """All CV deviance values are finite and positive."""
        df, y, exposure = cv_data
        model = SuperGLM(
            family="poisson",
            penalty=GroupLasso(),
            splines=["x1"],
        )
        result = model.fit_cv(
            df[["x1"]],
            y,
            exposure=exposure,
            n_folds=3,
            n_lambda=10,
            refit=False,
            random_state=42,
        )
        assert np.all(np.isfinite(result.fold_deviance))
        assert np.all(np.isfinite(result.mean_cv_deviance))
        assert np.all(result.se_cv_deviance >= 0)


class TestSelectLambda:
    def test_1se_rule(self):
        """1-SE rule selects the most regularised lambda within 1 SE."""
        lam = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
        mean_cv = np.array([2.0, 1.5, 1.0, 1.05, 1.1])
        se_cv = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

        best, best_1se, idx, idx_1se = _select_lambda(lam, mean_cv, se_cv, "min")
        assert best == 0.1  # index 2 has min
        # threshold = 1.0 + 0.1 = 1.1, so index 1 (mean=1.5) is above,
        # but index 2 (mean=1.0) is at or below
        assert idx_1se <= idx  # 1se index is at or before min index

    def test_min_rule(self):
        """min rule selects lambda at minimum CV deviance."""
        lam = np.array([1.0, 0.5, 0.1])
        mean_cv = np.array([2.0, 1.0, 1.5])
        se_cv = np.array([0.1, 0.1, 0.1])

        best, _, idx, _ = _select_lambda(lam, mean_cv, se_cv, "min")
        assert best == 0.5
        assert idx == 1

    def test_invalid_rule(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            _select_lambda(np.array([1.0]), np.array([1.0]), np.array([0.1]), "bad")
