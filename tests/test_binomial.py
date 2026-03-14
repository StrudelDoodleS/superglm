"""Tests for binomial (Bernoulli) GLM support."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from superglm import SuperGLM
from superglm.distributions import Binomial, validate_response
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline

# ── Distribution math ──────────────────────────────────────────────


class TestBinomialDistribution:
    def test_variance_at_half(self):
        d = Binomial()
        mu = np.array([0.5])
        assert_allclose(d.variance(mu), [0.25])

    def test_variance_formula(self):
        d = Binomial()
        mu = np.array([0.1, 0.3, 0.7, 0.9])
        assert_allclose(d.variance(mu), mu * (1 - mu))

    def test_variance_derivative(self):
        d = Binomial()
        mu = np.array([0.2, 0.5, 0.8])
        assert_allclose(d.variance_derivative(mu), 1 - 2 * mu)

    def test_deviance_small_at_correct_prediction(self):
        d = Binomial()
        # d(0, ~0) ≈ 0, d(1, ~1) ≈ 0
        assert d.deviance_unit(np.array([0.0]), np.array([1e-15]))[0] < 1e-10
        assert d.deviance_unit(np.array([1.0]), np.array([1 - 1e-15]))[0] < 1e-10

    def test_deviance_nonneg(self):
        d = Binomial()
        rng = np.random.default_rng(42)
        y = rng.choice([0.0, 1.0], size=100)
        mu = rng.uniform(0.01, 0.99, size=100)
        assert np.all(d.deviance_unit(y, mu) >= -1e-10)

    def test_deviance_y_zero(self):
        d = Binomial()
        y = np.array([0.0])
        mu = np.array([0.3])
        expected = -2 * np.log(1 - 0.3)
        assert_allclose(d.deviance_unit(y, mu), expected, rtol=1e-10)

    def test_deviance_y_one(self):
        d = Binomial()
        y = np.array([1.0])
        mu = np.array([0.7])
        expected = -2 * np.log(0.7)
        assert_allclose(d.deviance_unit(y, mu), expected, rtol=1e-10)

    def test_log_likelihood(self):
        d = Binomial()
        y = np.array([1.0, 0.0, 1.0])
        mu = np.array([0.8, 0.2, 0.6])
        w = np.ones(3)
        expected = np.log(0.8) + np.log(0.8) + np.log(0.6)
        assert_allclose(d.log_likelihood(y, mu, w), expected, rtol=1e-10)

    def test_scale_known(self):
        assert Binomial().scale_known is True

    def test_default_link(self):
        assert Binomial().default_link == "logit"


# ── Resolver ───────────────────────────────────────────────────────


class TestBinomialResolver:
    def test_string_resolves(self):
        from superglm.distributions import resolve_distribution

        d = resolve_distribution("binomial")
        assert isinstance(d, Binomial)

    def test_error_message_includes_binomial(self):
        from superglm.distributions import resolve_distribution

        with pytest.raises(ValueError, match="binomial"):
            resolve_distribution("bogus")


# ── Response validation ────────────────────────────────────────────


class TestValidateResponse:
    def test_valid_binary(self):
        validate_response(np.array([0.0, 1.0, 0.0, 1.0]), Binomial())

    def test_invalid_values_raises(self):
        with pytest.raises(ValueError, match="Binomial family requires y"):
            validate_response(np.array([0.0, 0.5, 1.0]), Binomial())

    def test_count_data_raises(self):
        with pytest.raises(ValueError, match="Binomial family requires y"):
            validate_response(np.array([0.0, 2.0, 3.0]), Binomial())

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="Binomial family requires y"):
            validate_response(np.array([-1.0, 0.0, 1.0]), Binomial())


# ── Fit + predict (end-to-end) ─────────────────────────────────────


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.standard_normal(n)
    x2 = rng.choice(["A", "B", "C"], n)
    eta = -0.5 + 0.8 * x1 + (x2 == "B") * 0.5 - (x2 == "C") * 0.3
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


class TestBinomialFit:
    def test_fit_predict_probabilities(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        proba = m.predict(X)
        assert np.all(proba > 0) and np.all(proba < 1)

    def test_coefficients_reasonable(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        # x1 coef should be positive (true coef = 0.8)
        x1_groups = [g for g in m._groups if g.feature_name == "x1"]
        x1_beta = np.concatenate([m.result.beta[g.sl] for g in x1_groups])
        assert x1_beta[0] > 0

    def test_phi_is_one(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        assert m.result.phi == 1.0

    def test_converged(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        assert m.result.converged

    def test_auto_detect_fit(self, binary_data):
        X, y = binary_data
        m = SuperGLM(family="binomial", lambda1=0, splines=[])
        m.fit(X, y)
        proba = m.predict(X)
        assert np.all(proba > 0) and np.all(proba < 1)

    def test_spline_fit(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Spline(n_knots=5), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        proba = m.predict(X)
        assert np.all(proba > 0) and np.all(proba < 1)

    def test_group_lasso_fit(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0.05,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        proba = m.predict(X)
        assert np.all(proba > 0) and np.all(proba < 1)

    def test_invalid_y_raises(self, binary_data):
        X, _ = binary_data
        y_bad = np.random.default_rng(0).uniform(0, 5, len(X))
        m = SuperGLM(family="binomial", lambda1=0, features={"x1": Numeric()})
        with pytest.raises(ValueError, match="Binomial family requires y"):
            m.fit(X, y_bad)

    def test_predict_clipped_with_unbounded_link(self, binary_data):
        """Predictions should be clipped to (0, 1) even with an unbounded link."""
        from superglm.links import IdentityLink

        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            link=IdentityLink(),
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        proba = m.predict(X)
        assert np.all(proba > 0), f"min prediction {proba.min()} <= 0"
        assert np.all(proba < 1), f"max prediction {proba.max()} >= 1"


# ── Statsmodels parity ─────────────────────────────────────────────


class TestStatsmodelsParity:
    def test_unpenalised_matches_statsmodels(self, binary_data):
        """Unpenalised linear-only binomial should match statsmodels closely."""
        pytest.importorskip("statsmodels")
        from statsmodels.genmod.families import Binomial as SmBinomial
        from statsmodels.genmod.generalized_linear_model import GLM

        X, y = binary_data
        # Numeric-only for clean comparison
        x_arr = np.column_stack([np.ones(len(y)), X["x1"].values])

        sm_model = GLM(y, x_arr, family=SmBinomial())
        sm_result = sm_model.fit()

        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(standardize=False)},
        )
        m.fit(X[["x1"]], y)

        # Compare intercepts
        assert_allclose(m.result.intercept, sm_result.params[0], atol=0.05)

        # Compare slopes
        x1_groups = [g for g in m._groups if g.feature_name == "x1"]
        x1_beta = np.concatenate([m.result.beta[g.sl] for g in x1_groups])
        assert_allclose(x1_beta[0], sm_result.params[1], atol=0.05)

        # Compare predicted probabilities
        sm_proba = sm_result.predict(x_arr)
        sg_proba = m.predict(X[["x1"]])
        assert_allclose(sg_proba, sm_proba, atol=0.01)


# ── Metrics ────────────────────────────────────────────────────────


class TestBinomialMetrics:
    def test_deviance_positive(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        assert m.result.deviance > 0

    def test_null_deviance(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        metrics = m.metrics(X, y)
        assert metrics.null_deviance > m.result.deviance

    def test_quantile_residuals(self, binary_data):
        X, y = binary_data
        m = SuperGLM(
            family="binomial",
            lambda1=0,
            features={"x1": Numeric(), "x2": Categorical(base="first")},
        )
        m.fit(X, y)
        metrics = m.metrics(X, y)
        r = metrics.residuals("quantile")
        assert r.shape == y.shape
        # Should be roughly standard normal
        assert abs(np.mean(r)) < 0.3
        assert 0.5 < np.std(r) < 1.5
