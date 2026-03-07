"""Tests for refit_unpenalised()."""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline


@pytest.fixture
def selection_data():
    """Data where high lambda1 should zero the noise feature."""
    rng = np.random.default_rng(42)
    n = 1000
    x_strong = rng.standard_normal(n)
    x_noise = rng.standard_normal(n)
    mu = np.exp(0.5 + 0.5 * x_strong)
    exposure = np.ones(n)
    y = rng.poisson(mu * exposure).astype(float)
    X = pd.DataFrame({"strong": x_strong, "noise": x_noise})
    return X, y, exposure


class TestRefitBasic:
    def test_returns_new_model(self, selection_data):
        X, y, exposure = selection_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.5,  # high penalty to zero noise
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, exposure=exposure)

        refitted = model.refit_unpenalised(X, y, exposure=exposure)
        assert refitted is not model
        assert isinstance(refitted, SuperGLM)
        assert refitted._result is not None

    def test_drops_inactive_features(self):
        """With high enough lambda1, inactive features are excluded from refit."""
        rng = np.random.default_rng(123)
        n = 1000
        x_strong = rng.standard_normal(n)
        # Use categorical noise — group lasso zeros the entire group more easily
        noise_cat = rng.choice(["A", "B", "C", "D"], n)
        mu = np.exp(0.5 + 0.5 * x_strong)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"strong": x_strong, "noise_cat": noise_cat})

        model = SuperGLM(
            family="poisson",
            lambda1=50.0,
            features={"strong": Numeric(), "noise_cat": Categorical(base="first")},
        )
        model.fit(X, y)

        # Verify noise_cat was zeroed (group lasso zeros entire group)
        noise_groups = model._feature_groups("noise_cat")
        beta = model.result.beta
        assert all(np.linalg.norm(beta[g.sl]) < 1e-12 for g in noise_groups)

        refitted = model.refit_unpenalised(X, y)
        assert "strong" in refitted._specs
        assert "noise_cat" not in refitted._specs

    def test_lambda1_is_zero(self, selection_data):
        X, y, exposure = selection_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.5,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, exposure=exposure)
        refitted = model.refit_unpenalised(X, y, exposure=exposure)

        assert refitted.penalty.lambda1 == 0.0

    def test_refitted_coefficients_differ(self, selection_data):
        X, y, exposure = selection_data
        model = SuperGLM(
            family="poisson",
            lambda1=0.1,  # moderate penalty — keeps strong but shrinks
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, exposure=exposure)
        refitted = model.refit_unpenalised(X, y, exposure=exposure)

        # Refitted coefficients should generally differ (less shrinkage)
        if "strong" in refitted._specs:
            orig_groups = model._feature_groups("strong")
            refit_groups = refitted._feature_groups("strong")
            orig_beta = model.result.beta[orig_groups[0].sl]
            refit_beta = refitted.result.beta[refit_groups[0].sl]
            # Not identical — shrinkage removed
            assert not np.allclose(orig_beta, refit_beta, atol=1e-6)

    def test_unfitted_raises(self):
        model = SuperGLM(features={"x": Numeric()})
        X = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.refit_unpenalised(X, np.array([1, 2, 3]))

    def test_keep_smoothing_false(self):
        """keep_smoothing=False should set lambda2=0."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            lambda2=0.5,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)
        refitted = model.refit_unpenalised(X, y, keep_smoothing=False)

        assert refitted.lambda2 == 0.0
