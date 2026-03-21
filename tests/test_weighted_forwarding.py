"""Regression tests for sample_weight forwarding through profilers and CV.

These catch the bug where the exposure→sample_weight rename silently
dropped weight arguments from internal forwarding calls.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import NegativeBinomial, Tweedie
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.penalties.group_lasso import GroupLasso


@pytest.fixture()
def weighted_poisson_data():
    """Small weighted Poisson dataset."""
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.uniform(0, 1, n)
    w = rng.uniform(0.5, 2.0, n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu * w).astype(float) / w
    df = pd.DataFrame({"x": x})
    return df, y, w


@pytest.fixture()
def weighted_nb_data():
    """Small weighted NB2 dataset."""
    rng = np.random.default_rng(123)
    n = 1500
    x = rng.uniform(0, 1, n)
    w = rng.uniform(0.3, 1.5, n)
    mu = np.exp(1.0 + 0.5 * x)
    y = rng.negative_binomial(2, 2 / (mu + 2), size=n).astype(float)
    df = pd.DataFrame({"x": x})
    return df, y, w


class TestWeightedNBTheta:
    """Verify sample_weight flows through NB2 theta estimation."""

    def test_auto_theta_matches_explicit(self, weighted_nb_data):
        """theta='auto' with weights should match explicit estimate_theta with weights."""
        df, y, w = weighted_nb_data

        # Explicit weighted theta estimation
        m1 = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        m1.fit(df, y, sample_weight=w)
        result = m1.estimate_theta(df, y, sample_weight=w)
        theta_explicit = result.theta_hat

        # Auto weighted theta
        m2 = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        m2.fit(df, y, sample_weight=w)
        theta_auto = m2.family.theta

        assert theta_auto == pytest.approx(theta_explicit, rel=0.01)

    def test_estimate_theta_preserves_weights(self, weighted_nb_data):
        """After estimate_theta with weights, model._fit_weights should reflect those weights."""
        df, y, w = weighted_nb_data
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(df, y, sample_weight=w)
        model.estimate_theta(df, y, sample_weight=w)
        np.testing.assert_allclose(model._fit_weights, w)


class TestWeightedTweedieP:
    """Verify sample_weight flows through Tweedie p estimation."""

    def test_estimate_p_preserves_weights(self):
        """After estimate_p with weights, model._fit_weights should reflect those weights."""
        rng = np.random.default_rng(99)
        n = 1000
        x = rng.uniform(0, 1, n)
        w = rng.uniform(0.5, 2.0, n)
        from superglm.tweedie_profile import generate_tweedie_cpg

        mu = np.exp(0.5 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu=mu, phi=1.5, p=1.5, rng=rng)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(df, y, sample_weight=w)
        model.estimate_p(df, y, sample_weight=w, phi_method="mle")
        np.testing.assert_allclose(model._fit_weights, w)


class TestWeightedCV:
    """Verify sample_weight flows through fit_cv refit."""

    def test_cv_refit_uses_weights(self, weighted_poisson_data):
        """fit_cv(refit=True) should produce the same result as manual fit with weights."""
        df, y, w = weighted_poisson_data
        model_cv = SuperGLM(
            penalty=GroupLasso(lambda1=0.1),
            features={"x": Numeric()},
        )
        cv_result = model_cv.fit_cv(df, y, sample_weight=w, n_lambda=5, refit=True)

        # Manual refit at the same lambda with weights
        model_manual = SuperGLM(
            penalty=GroupLasso(lambda1=cv_result.best_lambda),
            features={"x": Numeric()},
        )
        model_manual.fit(df, y, sample_weight=w)

        np.testing.assert_allclose(model_cv._result.beta, model_manual._result.beta, atol=1e-8)


class TestCICache:
    """Verify profile CI caching avoids recomputation."""

    def test_tweedie_ci_cached(self):
        """Repeated .ci() calls at same alpha return cached result."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        from superglm.tweedie_profile import generate_tweedie_cpg

        y = generate_tweedie_cpg(n, mu=np.exp(0.5 * x), phi=1.5, p=1.5, rng=rng)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(df, y)
        result = model.estimate_p(df, y, phi_method="mle")

        ci1 = result.ci(alpha=0.05)
        ci2 = result.ci(alpha=0.05)
        assert ci1 is ci2  # exact same object = cached

    def test_nb_ci_cached(self, weighted_nb_data):
        """Repeated NB .ci() calls at same alpha return cached result."""
        df, y, w = weighted_nb_data
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(df, y, sample_weight=w)
        result = model.estimate_theta(df, y, sample_weight=w)

        ci1 = result.ci(alpha=0.05)
        ci2 = result.ci(alpha=0.05)
        assert ci1 is ci2  # exact same object = cached


class TestStaleProfileClear:
    """Verify fit() clears stale profile results from previous estimate_p/estimate_theta."""

    def test_fit_clears_tweedie_profile(self):
        """fit() at fixed p should not show profile results in summary."""
        rng = np.random.default_rng(42)
        n = 500
        from superglm.tweedie_profile import generate_tweedie_cpg

        x = rng.uniform(0, 1, n)
        y = generate_tweedie_cpg(n, mu=np.exp(0.5 * x), phi=1.5, p=1.5, rng=rng)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(df, y)
        model.estimate_p(df, y, phi_method="mle")
        assert model._tweedie_profile_result is not None

        # Refit at fixed p — should clear the stale profile result
        model.fit(df, y)
        assert model._tweedie_profile_result is None

    def test_fit_clears_nb_profile(self):
        """fit() at fixed theta should not show profile results in summary."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 1, n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.negative_binomial(5, 5 / (mu + 5), size=n).astype(float)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(df, y)
        model.estimate_theta(df, y)
        assert model._nb_profile_result is not None

        # Refit at fixed theta — should clear the stale profile result
        model.fit(df, y)
        assert model._nb_profile_result is None

    def test_fit_reml_clears_tweedie_profile(self):
        """fit_reml() should also clear stale profile results."""
        rng = np.random.default_rng(42)
        n = 500
        from superglm.tweedie_profile import generate_tweedie_cpg

        x = rng.uniform(0, 1, n)
        y = generate_tweedie_cpg(n, mu=np.exp(0.5 * x), phi=1.5, p=1.5, rng=rng)
        df = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Spline(n_knots=5)},
        )
        model.fit(df, y)
        model.estimate_p(df, y, phi_method="mle")
        assert model._tweedie_profile_result is not None

        model.fit_reml(df, y)
        assert model._tweedie_profile_result is None
