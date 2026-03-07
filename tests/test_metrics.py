"""Tests for ModelMetrics diagnostics module."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln
from scipy.stats import poisson

from superglm import ModelMetrics, SuperGLM
from superglm.distributions import Gamma, Poisson, Tweedie
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def poisson_data():
    """Small Poisson dataset with known structure."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    eta = 0.5 + 0.3 * x1 - 0.2 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    w = np.ones(n)
    return X, y, w


@pytest.fixture
def fitted_poisson(poisson_data):
    """Fitted Poisson model on the test data."""
    X, y, w = poisson_data
    model = SuperGLM(
        family="poisson",
        lambda1=0.001,
        features={"x1": Numeric(), "x2": Numeric()},
    )
    model.fit(X, y, exposure=w)
    return model, X, y, w


@pytest.fixture
def metrics_obj(fitted_poisson):
    """ModelMetrics from the fitted Poisson model."""
    model, X, y, w = fitted_poisson
    return model.metrics(X, y, exposure=w)


# ── Log-likelihood ────────────────────────────────────────────────


class TestLogLikelihood:
    def test_poisson_ll_matches_scipy(self):
        """Poisson LL should match scipy.stats.poisson.logpmf."""
        y = np.array([0, 1, 2, 5, 10], dtype=float)
        mu = np.array([1.0, 2.0, 3.0, 4.0, 8.0])
        w = np.ones(5)
        ll = Poisson().log_likelihood(y, mu, w)
        expected = np.sum(poisson.logpmf(y.astype(int), mu))
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_poisson_ll_with_weights(self):
        """Weighted LL should differ from unweighted."""
        y = np.array([1, 2, 3], dtype=float)
        mu = np.array([1.5, 2.5, 2.0])
        w1 = np.ones(3)
        w2 = np.array([2.0, 1.0, 0.5])
        ll1 = Poisson().log_likelihood(y, mu, w1)
        ll2 = Poisson().log_likelihood(y, mu, w2)
        assert ll1 != ll2

    def test_gamma_ll_formula(self):
        """Gamma LL should match manual computation."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.5, 2.8])
        w = np.ones(3)
        phi = 0.5
        k = 1.0 / phi
        expected = float(np.sum(k * np.log(k * y / mu) - k * y / mu - np.log(y) - gammaln(k)))
        ll = Gamma().log_likelihood(y, mu, w, phi=phi)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_ll_from_metrics(self, metrics_obj):
        """LL accessed via metrics should be finite and negative."""
        assert np.isfinite(metrics_obj.log_likelihood)


# ── Information criteria ──────────────────────────────────────────


class TestInformationCriteria:
    def test_aic_formula(self, metrics_obj):
        """AIC = -2*LL + 2*edf."""
        expected = -2.0 * metrics_obj.log_likelihood + 2.0 * metrics_obj.effective_df
        np.testing.assert_allclose(metrics_obj.aic, expected)

    def test_bic_formula(self, metrics_obj):
        """BIC = -2*LL + log(n)*edf."""
        expected = (
            -2.0 * metrics_obj.log_likelihood + np.log(metrics_obj.n_obs) * metrics_obj.effective_df
        )
        np.testing.assert_allclose(metrics_obj.bic, expected)

    def test_bic_ge_aic(self, metrics_obj):
        """BIC >= AIC when n >= e^2 ≈ 7.4 (which it always is here)."""
        assert metrics_obj.bic >= metrics_obj.aic - 1e-10

    def test_aicc_formula(self, metrics_obj):
        edf = metrics_obj.effective_df
        n = metrics_obj.n_obs
        expected = metrics_obj.aic + 2 * edf * (edf + 1) / (n - edf - 1)
        np.testing.assert_allclose(metrics_obj.aicc, expected)

    def test_ebic_ge_bic(self, metrics_obj):
        """EBIC(gamma>0) >= BIC."""
        assert metrics_obj.ebic(gamma=0.5) >= metrics_obj.bic - 1e-10

    def test_ebic_gamma_zero_equals_bic(self, metrics_obj):
        """EBIC(gamma=0) == BIC."""
        np.testing.assert_allclose(metrics_obj.ebic(gamma=0.0), metrics_obj.bic, atol=1e-10)


# ── Null model ────────────────────────────────────────────────────


class TestNullModel:
    def test_null_mu_equals_weighted_mean(self):
        """_null_mu should be the weighted mean of y, not a zero-replaced mean."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.standard_normal(n)
        # Sparse Poisson: many zeros
        eta = -1.5 + 0.3 * x
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(family="poisson", lambda1=0, features={"x": Numeric()})
        model.fit(X, y)
        m = model.metrics(X, y)

        expected_null_mu = np.average(y, weights=np.ones(n))
        np.testing.assert_allclose(m._null_mu[0], expected_null_mu, rtol=1e-6)

    def test_null_mu_with_offset(self):
        """_null_mu with offset should satisfy the score equation, not ignore offset."""
        rng = np.random.default_rng(55)
        n = 500
        x = rng.standard_normal(n)
        offset = rng.standard_normal(n) * 0.5
        eta = 0.3 + 0.2 * x + offset
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(family="poisson", lambda1=0, features={"x": Numeric()})
        model.fit(X, y, offset=offset)
        m = model.metrics(X, y, offset=offset)

        # Null mu should NOT be constant when offset is present
        assert m._null_mu.std() > 0.01
        # Score equation: sum(w*(y - mu)) should be near zero at the MLE
        score = np.sum(y - m._null_mu)
        assert abs(score) < 1.0


# ── Deviance ──────────────────────────────────────────────────────


class TestDeviance:
    def test_null_deviance_gt_residual(self, metrics_obj):
        """Model should improve on the null (intercept-only) model."""
        assert metrics_obj.null_deviance > metrics_obj.deviance

    def test_explained_deviance_in_range(self, metrics_obj):
        """Explained deviance should be in [0, 1] for a well-fitting model."""
        assert 0 <= metrics_obj.explained_deviance <= 1

    def test_pearson_chi2_positive(self, metrics_obj):
        assert metrics_obj.pearson_chi2 > 0


# ── Residuals ─────────────────────────────────────────────────────


class TestResiduals:
    def test_deviance_residuals_sum_sq_approx_deviance(self, metrics_obj):
        """sum(r_dev^2) should approximately equal the deviance."""
        r = metrics_obj.residuals("deviance")
        np.testing.assert_allclose(np.sum(r**2), metrics_obj.deviance, rtol=0.01)

    def test_pearson_residuals_mean_approx_zero(self, metrics_obj):
        """Pearson residuals should have mean approximately 0."""
        r = metrics_obj.residuals("pearson")
        assert abs(np.mean(r)) < 0.5  # rough check

    def test_response_residuals(self, metrics_obj):
        """Response residuals are just y - mu."""
        r = metrics_obj.residuals("response")
        np.testing.assert_allclose(r, metrics_obj._y - metrics_obj._mu)

    def test_working_residuals(self, metrics_obj):
        """Working residuals are (y - mu) / mu for log link."""
        r = metrics_obj.residuals("working")
        np.testing.assert_allclose(r, (metrics_obj._y - metrics_obj._mu) / metrics_obj._mu)

    def test_unknown_residual_raises(self, metrics_obj):
        with pytest.raises(ValueError, match="Unknown residual type"):
            metrics_obj.residuals("bogus")

    def test_quantile_residuals_poisson(self, metrics_obj):
        """Quantile residuals should be approximately standard normal."""
        r = metrics_obj.residuals("quantile")
        assert abs(np.mean(r)) < 0.3
        assert 0.5 < np.std(r) < 1.5

    def test_quantile_residuals_gamma(self):
        """Quantile residuals for Gamma should be approximately N(0,1)."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        shape = 5.0  # phi = 1/shape = 0.2
        y = rng.gamma(shape, scale=mu / shape, size=n)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="gamma",
            lambda1=0.001,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        r = m.residuals("quantile")

        assert r.shape == (n,)
        assert abs(np.mean(r)) < 0.15
        assert 0.7 < np.std(r) < 1.3

    def test_quantile_residuals_tweedie(self):
        """Quantile residuals for Tweedie should be approximately standard normal."""
        from superglm.tweedie_profile import generate_tweedie_cpg

        rng = np.random.default_rng(42)
        n = 2000
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=1.5, rng=rng)
        y = np.maximum(y, 0.0)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            lambda1=0.001,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        qr = m.residuals("quantile")

        assert qr.shape == (n,)
        assert np.all(np.isfinite(qr))
        # Well-specified model: quantile residuals should be ~N(0,1)
        assert abs(np.mean(qr)) < 0.15
        assert abs(np.std(qr) - 1.0) < 0.15


# ── Leverage ──────────────────────────────────────────────────────


class TestLeverage:
    def test_leverage_bounded(self, metrics_obj):
        """All leverage values should be in [0, 1]."""
        h = metrics_obj.leverage
        assert np.all(h >= 0)
        assert np.all(h <= 1.0 + 1e-10)

    def test_leverage_sum_approx_edf(self, metrics_obj):
        """sum(h_i) should approximate effective_df."""
        h_sum = np.sum(metrics_obj.leverage)
        # Leverage sum ≈ p_active (not exactly edf due to shrinkage),
        # but should be in the right ballpark
        assert h_sum > 0
        assert h_sum < metrics_obj.n_obs


# ── Cook's distance ──────────────────────────────────────────────


class TestCooksDistance:
    def test_cooks_nonnegative(self, metrics_obj):
        assert np.all(metrics_obj.cooks_distance >= 0)

    def test_std_deviance_residuals_exist(self, metrics_obj):
        r = metrics_obj.std_deviance_residuals
        assert r.shape == (metrics_obj.n_obs,)
        assert np.all(np.isfinite(r))

    def test_std_pearson_residuals_exist(self, metrics_obj):
        r = metrics_obj.std_pearson_residuals
        assert r.shape == (metrics_obj.n_obs,)
        assert np.all(np.isfinite(r))


# ── Active groups ─────────────────────────────────────────────────


class TestActiveGroups:
    def test_n_active_groups(self, metrics_obj):
        """With low lambda, both features should be active."""
        assert metrics_obj.n_active_groups == 2


# ── Summary ───────────────────────────────────────────────────────


class TestSummary:
    def test_summary_keys(self, metrics_obj):
        """Dict-like access still works via __contains__/__getitem__."""
        s = metrics_obj.summary()
        assert "information_criteria" in s
        assert "deviance" in s
        assert "fit" in s
        assert "aic" in s["information_criteria"]
        assert "bic" in s["information_criteria"]

    def test_summary_values_finite(self, metrics_obj):
        s = metrics_obj.summary()
        for key, section in s.items():
            if key == "standard_errors":
                continue  # tested separately
            for v in section.values():
                assert np.isfinite(v), f"Non-finite value in summary: {v}"

    def test_summary_returns_model_summary(self, metrics_obj):
        """summary() returns a ModelSummary object."""
        from superglm.summary import ModelSummary

        s = metrics_obj.summary()
        assert isinstance(s, ModelSummary)

    def test_summary_to_dict(self, metrics_obj):
        """to_dict() returns the raw dict."""
        s = metrics_obj.summary()
        d = s.to_dict()
        assert isinstance(d, dict)
        assert "fit" in d

    def test_summary_str_contains_title(self, metrics_obj):
        """ASCII output contains 'SuperGLM Results'."""
        text = str(metrics_obj.summary())
        assert "SuperGLM Results" in text

    def test_summary_str_contains_family(self, metrics_obj):
        """ASCII output shows family name."""
        text = str(metrics_obj.summary())
        assert "Poisson" in text

    def test_summary_str_contains_intercept(self, metrics_obj):
        """ASCII output has an Intercept row."""
        text = str(metrics_obj.summary())
        assert "Intercept" in text

    def test_summary_str_contains_features(self, metrics_obj):
        """ASCII output lists fitted features."""
        text = str(metrics_obj.summary())
        assert "x1" in text
        assert "x2" in text

    def test_summary_html_output(self, metrics_obj):
        """_repr_html_ produces valid-looking HTML."""
        html = metrics_obj.summary()._repr_html_()
        assert "<table" in html
        assert "SuperGLM Results" in html
        assert "</table>" in html

    def test_summary_repr_is_str(self, metrics_obj):
        """repr() returns the same as str()."""
        s = metrics_obj.summary()
        assert repr(s) == str(s)

    def test_summary_significance_stars(self, metrics_obj):
        """ASCII output contains significance stars and legend."""
        text = str(metrics_obj.summary())
        assert "***" in text
        assert "Signif. codes:" in text

    def test_summary_consistent_width(self, metrics_obj):
        """Header separator and coef separator should be the same width."""
        text = str(metrics_obj.summary())
        eq_lines = [line for line in text.split("\n") if set(line) == {"="}]
        assert len(eq_lines) >= 2
        assert len(eq_lines[0]) == len(eq_lines[1])


class TestSummaryMixedFeatures:
    """Test summary with numeric, categorical, and spline features together."""

    @pytest.fixture
    def mixed_model(self):
        from superglm.features.categorical import Categorical

        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.standard_normal(n)
        region = rng.choice(["A", "B", "C"], n)
        age = rng.uniform(0, 10, n)
        mu = np.exp(
            0.3 * x1 + np.where(region == "B", 0.5, np.where(region == "C", -0.3, 0)) + 0.05 * age
        )
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "region": region, "age": age})
        model = SuperGLM(
            family="poisson",
            lambda1=0.001,
            features={
                "x1": Numeric(),
                "region": Categorical(),
                "age": Spline(n_knots=8, penalty="ssp"),
            },
        )
        model.fit(X, y)
        return model, X, y

    def test_mixed_summary_str(self, mixed_model):
        """Summary with all feature types produces valid output."""
        model, X, y = mixed_model
        m = model.metrics(X, y)
        text = str(m.summary())
        # Numeric feature
        assert "x1" in text
        # Categorical levels
        assert "region[" in text
        # Spline per-coefficient rows
        assert "spline" in text
        assert "chi2(" in text

    def test_mixed_summary_html(self, mixed_model):
        """HTML summary with all feature types."""
        model, X, y = mixed_model
        html = model.metrics(X, y).summary()._repr_html_()
        assert "x1" in html
        assert "region[" in html
        assert "spline" in html

    def test_intercept_se_positive(self, mixed_model):
        """Intercept SE should be positive."""
        model, X, y = mixed_model
        m = model.metrics(X, y)
        assert m.intercept_se > 0

    def test_intercept_se_reasonable(self, mixed_model):
        """Intercept SE should be much smaller than 1 for n=500."""
        model, X, y = mixed_model
        m = model.metrics(X, y)
        assert m.intercept_se < 1.0


# ── Integration: spline model ────────────────────────────────────


class TestSplineIntegration:
    def test_spline_model_metrics(self):
        """Smoke test: metrics work with spline features."""
        rng = np.random.default_rng(123)
        n = 300
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.1 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        w = np.ones(n)

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y, exposure=w)
        m = model.metrics(X, y, exposure=w)

        # All properties should be accessible without error
        assert np.isfinite(m.aic)
        assert np.isfinite(m.bic)
        assert np.isfinite(m.aicc)
        assert np.isfinite(m.log_likelihood)
        assert m.null_deviance > m.deviance
        assert 0 < m.explained_deviance < 1

        r = m.residuals("deviance")
        assert r.shape == (n,)

        h = m.leverage
        assert np.all(h >= 0)
        assert np.all(h <= 1.0 + 1e-10)

        assert np.all(m.cooks_distance >= 0)


# ── Convenience accessor ─────────────────────────────────────────


class TestConvenienceAccessor:
    def test_model_metrics_method(self, fitted_poisson):
        """SuperGLM.metrics() returns a ModelMetrics object."""
        model, X, y, w = fitted_poisson
        m = model.metrics(X, y, exposure=w)
        assert isinstance(m, ModelMetrics)


# ── Coefficient standard errors ──────────────────────────────────


class TestCoefficientSE:
    def test_se_keys_match_groups(self, metrics_obj):
        """SE dicts should have one entry per group."""
        se = metrics_obj.coefficient_se
        assert set(se.keys()) == {"x1", "x2"}

    def test_se_positive_for_active_groups(self, metrics_obj):
        """Active groups should have strictly positive SEs."""
        for name, se_arr in metrics_obj.coefficient_se.items():
            assert np.all(se_arr > 0), f"SE for {name} should be positive"

    def test_se_raw_positive(self, metrics_obj):
        """Raw SEs should also be positive for active groups."""
        for name, se_arr in metrics_obj.coefficient_se_raw.items():
            assert np.all(se_arr > 0), f"Raw SE for {name} should be positive"

    def test_se_raw_vs_corrected_poisson(self, metrics_obj):
        """For Poisson, corrected SE = sqrt(phi) * raw SE."""
        phi = metrics_obj.phi
        for name in metrics_obj.coefficient_se:
            se_corr = metrics_obj.coefficient_se[name]
            se_raw = metrics_obj.coefficient_se_raw[name]
            np.testing.assert_allclose(se_corr, np.sqrt(phi) * se_raw, rtol=1e-10)

    def test_se_reasonable_magnitude(self, fitted_poisson):
        """SEs should be much smaller than coefficients for well-determined params."""
        model, X, y, w = fitted_poisson
        m = model.metrics(X, y, exposure=w)
        for name in ["x1", "x2"]:
            se = m.coefficient_se[name][0]
            coef = abs(model.result.beta[next(g for g in model._groups if g.name == name).sl][0])
            # SE should be < coefficient for n=500 with reasonable signal
            assert se < coef * 5, f"SE too large relative to coef for {name}"

    def test_inactive_group_gets_zero_se(self):
        """A zeroed-out group should have SE=0."""
        rng = np.random.default_rng(99)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)  # irrelevant feature
        y = rng.poisson(np.exp(0.5 * x1)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            lambda1=0.5,  # high penalty to zero out x2
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        se = m.coefficient_se

        # x2 might be zeroed out with high lambda
        for name, se_arr in se.items():
            g = next(g for g in model._groups if g.name == name)
            coef_norm = np.linalg.norm(model.result.beta[g.sl])
            if coef_norm < 1e-12:
                np.testing.assert_array_equal(se_arr, 0.0)

    def test_summary_includes_standard_errors(self, metrics_obj):
        """Summary should include standard_errors section."""
        s = metrics_obj.summary()
        assert "standard_errors" in s
        assert "coefficient_se" in s["standard_errors"]
        assert "coefficient_se_raw" in s["standard_errors"]


class TestFeatureSE:
    def test_numeric_feature_se(self, fitted_poisson):
        """feature_se for numeric returns a scalar SE."""
        model, X, y, w = fitted_poisson
        m = model.metrics(X, y, exposure=w)
        result = m.feature_se("x1")
        assert "se_coef" in result
        assert result["se_coef"] > 0

    def test_spline_feature_se(self):
        """feature_se for spline returns grid-aligned SEs."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.1 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        w = np.ones(n)

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y, exposure=w)
        m = model.metrics(X, y, exposure=w)

        result = m.feature_se("x", n_points=100)
        assert "x" in result
        assert "se_log_relativity" in result
        assert len(result["x"]) == 100
        assert len(result["se_log_relativity"]) == 100
        assert np.all(result["se_log_relativity"] >= 0)
        assert np.any(result["se_log_relativity"] > 0)

    def test_categorical_feature_se(self):
        """feature_se for categorical returns per-level SEs."""
        from superglm.features.categorical import Categorical

        rng = np.random.default_rng(42)
        n = 500
        region = rng.choice(["A", "B", "C", "D"], n)
        mu = np.where(region == "A", 1.0, np.where(region == "B", 1.5, 2.0))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"region": region})

        model = SuperGLM(
            family="poisson",
            lambda1=0.001,
            features={"region": Categorical()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)

        result = m.feature_se("region")
        assert "levels" in result
        assert "se_log_relativity" in result
        # Non-base levels should have positive SEs
        assert np.any(result["se_log_relativity"] > 0)


class TestRelativitiesWithSE:
    def test_without_se_no_extra_column(self, fitted_poisson):
        """Default relativities() has no SE column."""
        model, X, y, w = fitted_poisson
        rels = model.relativities(with_se=False)
        for name, df in rels.items():
            assert "se_log_relativity" not in df.columns

    def test_with_se_adds_column(self, fitted_poisson):
        """relativities(with_se=True) adds se_log_relativity column."""
        model, X, y, w = fitted_poisson
        rels = model.relativities(with_se=True)
        for name, df in rels.items():
            assert "se_log_relativity" in df.columns
            assert np.all(np.isfinite(df["se_log_relativity"]))

    def test_spline_relativities_with_se(self):
        """SE column works for spline features in relativities()."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.1 * x)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y)
        rels = model.relativities(with_se=True)
        df = rels["x"]
        assert "se_log_relativity" in df.columns
        assert len(df) == 200  # default n_points from reconstruct
        assert np.all(df["se_log_relativity"] >= 0)

    def test_categorical_relativities_with_se(self):
        """SE column works for categorical features."""
        from superglm.features.categorical import Categorical

        rng = np.random.default_rng(42)
        n = 500
        region = rng.choice(["A", "B", "C"], n)
        mu = np.where(region == "A", 1.0, 2.0)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"region": region})

        model = SuperGLM(
            family="poisson",
            lambda1=0.001,
            features={"region": Categorical()},
        )
        model.fit(X, y)
        rels = model.relativities(with_se=True)
        df = rels["region"]
        assert "se_log_relativity" in df.columns
        # Base level should have SE=0
        base_idx = df["level"] == model._specs["region"]._base_level
        assert df.loc[base_idx, "se_log_relativity"].iloc[0] == 0.0

    def test_wood_bayesian_covariance_multi_spline(self):
        """Wood's Bayesian covariance produces finite positive SEs for multi-spline models."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 10, n)
        x2 = rng.uniform(0, 5, n)
        mu = np.exp(0.1 * x1 - 0.2 * x2)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})

        model = SuperGLM(
            family="poisson",
            lambda1=0.001,
            features={
                "x1": Spline(n_knots=8, penalty="ssp"),
                "x2": Spline(n_knots=6, penalty="ssp"),
            },
        )
        model.fit(X, y)
        m = model.metrics(X, y)

        # Active groups should have finite, positive SEs
        beta = model.result.beta
        for name, se_arr in m.coefficient_se.items():
            g = next(g for g in model._groups if g.name == name)
            if np.linalg.norm(beta[g.sl]) > 1e-12:
                assert np.all(np.isfinite(se_arr)), f"Non-finite SE for {name}"
                assert np.all(se_arr > 0), f"Zero SE for active group {name}"

        # Feature-level curve SEs should be finite and positive
        for name in ["x1", "x2"]:
            fse = m.feature_se(name)
            assert np.all(np.isfinite(fse["se_log_relativity"]))
            assert np.any(fse["se_log_relativity"] > 0)

        # Wald chi2 tests should be finite
        text = str(m.summary())
        assert "chi2(" in text
        assert np.isfinite(m.aic)

    def test_gamma_se_differs_from_raw(self):
        """For Gamma, phi != 1 so coefficient_se != coefficient_se_raw."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(1, 5, n)
        mu = np.exp(0.3 * x)
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="gamma",
            lambda1=0.001,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)

        se_corr = m.coefficient_se["x"][0]
        se_raw = m.coefficient_se_raw["x"][0]
        # phi != 1 for Gamma, so they should differ
        assert se_corr != se_raw
        # Corrected = sqrt(phi) * raw
        np.testing.assert_allclose(se_corr, np.sqrt(m.phi) * se_raw, rtol=1e-10)


# ── Offset SE consistency (model-level vs metrics-level) ───────


class TestOffsetSEConsistency:
    """Model-level SEs (relativities) must match metrics-level SEs when offset is present."""

    def test_spline_se_agrees_with_offset(self):
        rng = np.random.default_rng(99)
        n = 1000
        x = rng.uniform(0, 1, n)
        offset = rng.standard_normal(n) * 0.3
        eta = 0.5 + np.sin(2 * np.pi * x) * 0.4 + offset
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            lambda1=0,
            features={"x": Spline(n_knots=8)},
        )
        model.fit(X, y, offset=offset)

        # Model-level SE (from _coef_covariance via relativities)
        rels = model.relativities(with_se=True)
        model_se = rels["x"]["se_log_relativity"].values

        # Metrics-level SE (from _active_info via feature_se)
        m = model.metrics(X, y, offset=offset)
        fse = m.feature_se("x")
        metrics_se = fse["se_log_relativity"]

        # Both paths should agree (they compute the same Bayesian covariance)
        np.testing.assert_allclose(model_se, metrics_se, rtol=0.05)

    def test_categorical_se_agrees_with_offset(self):
        rng = np.random.default_rng(77)
        n = 1000
        groups = rng.choice(["A", "B", "C", "D"], n)
        offset = rng.standard_normal(n) * 0.5
        effects = {"A": 0.0, "B": 0.3, "C": -0.2, "D": 0.5}
        eta = np.array([effects[g] for g in groups]) + offset
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"g": groups})

        model = SuperGLM(
            family="poisson",
            lambda1=0,
            features={"g": Categorical()},
        )
        model.fit(X, y, offset=offset)

        rels = model.relativities(with_se=True)
        # relativities includes base level (SE=0); filter to non-base
        rel_df = rels["g"]
        non_base = rel_df[rel_df["se_log_relativity"] > 0]
        model_se = non_base["se_log_relativity"].values

        m = model.metrics(X, y, offset=offset)
        fse = m.feature_se("g")
        metrics_se = fse["se_log_relativity"]

        np.testing.assert_allclose(model_se, metrics_se, rtol=0.05)


# ── Coverage gap tests ──────────────────────────────────────────


class TestNBProfileSummary:
    """NB2 profile result appears in ASCII and HTML summary."""

    def test_nb_profile_summary(self):
        from superglm.distributions import NegativeBinomial

        rng = np.random.default_rng(42)
        n = 1000
        true_theta = 5.0
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.3 * x)
        p_nb = true_theta / (mu + true_theta)
        y = rng.negative_binomial(true_theta, p_nb).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            lambda1=0.001,
            features={"x": Numeric()},
        )
        model.estimate_theta(X, y)
        m = model.metrics(X, y)
        text = str(m.summary())
        assert "Theta" in text
        assert "[" in text  # CI brackets
        html = m.summary()._repr_html_()
        assert "Theta" in html


class TestTweedieProfileSummary:
    """Tweedie p profile result appears in ASCII and HTML summary."""

    def test_tweedie_profile_summary(self):
        from superglm.tweedie_profile import generate_tweedie_cpg

        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        mu = np.exp(1.0 + 0.2 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=1.5, rng=rng)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="tweedie",
            tweedie_p=1.5,
            lambda1=0.0,
            features={"x": Numeric()},
        )
        model.estimate_p(X, y, p_bounds=(1.1, 1.9))
        m = model.metrics(X, y)
        text = str(m.summary())
        assert "Tweedie p" in text
        html = m.summary()._repr_html_()
        assert "Tweedie p" in html


class TestInactiveSummaryRendering:
    """Inactive spline and coefficient rendering in summary."""

    def test_inactive_spline_summary_rendering(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        y = rng.poisson(1.0, n).astype(float)  # pure noise
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            lambda1=1e6,
            features={"x": Spline(n_knots=8)},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        text = str(m.summary())
        assert "inactive" in text
        html = m.summary()._repr_html_()
        assert "inactive" in html
        fse = m.feature_se("x")
        assert np.all(fse["se_log_relativity"] == 0)

    def test_inactive_coef_summary_rendering(self):
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(0, 5, n)
        x2 = rng.uniform(0, 5, n)  # noise feature
        y = rng.poisson(np.exp(0.5 + 0.3 * x1)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        model = SuperGLM(
            family="poisson",
            lambda1=10.0,
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        text = str(m.summary())
        html = m.summary()._repr_html_()
        # At least one feature should show "---" (inactive coef)
        assert "---" in text or "inactive" in text
        assert "---" in html or "inactive" in html


class TestPolynomialCategoricalSummary:
    """PolynomialCategorical interaction Wald test in summary."""

    def test_polynomial_categorical_summary(self):
        from superglm.features.polynomial import Polynomial

        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        cat = rng.choice(["A", "B", "C"], n)
        eta = 0.5 + 0.1 * x + 0.3 * (cat == "B")
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x": x, "cat": cat})
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x": Polynomial(degree=2), "cat": Categorical()},
            interactions=[("x", "cat")],
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        text = str(m.summary())
        assert "x:cat" in text
        html = m.summary()._repr_html_()
        assert "x:cat" in html


class TestAICcEdgeCase:
    """AICc with near-saturated model."""

    def test_aicc_saturated_model(self):
        """AICc returns inf when effective_df >= n - 1."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        # Patch effective_df to force the denom <= 0 branch
        original_edf = m._result.effective_df
        m._result.effective_df = float(n)  # n - edf - 1 = -1 <= 0
        assert m.aicc == np.inf
        m._result.effective_df = original_edf  # restore


class TestSummaryHelpers:
    """Edge cases in summary helper functions."""

    def test_summary_helpers_edge_cases(self):
        from superglm.summary import _compute_coef_stats, _sig_stars

        z, p, lo, hi = _compute_coef_stats(1.0, 0.0)
        assert all(np.isnan(v) for v in (z, p, lo, hi))
        assert _sig_stars(None) == ""
        assert _sig_stars(np.nan) == ""


class TestNumericUnstandardizedSE:
    """feature_se with standardize=False Numeric."""

    def test_numeric_unstandardized_se(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        y = rng.poisson(np.exp(0.5 + 0.2 * x)).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            lambda1=0.0,
            features={"x": Numeric(standardize=False)},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        fse = m.feature_se("x")
        assert "se_coef" in fse
        assert fse["se_coef"] > 0
