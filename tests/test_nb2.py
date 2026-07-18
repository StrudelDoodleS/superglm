"""Tests for Negative Binomial (NB2) distribution."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom

from superglm import NegativeBinomial, Spline, SuperGLM, SuperGLMRegressor
from superglm.distributions import resolve_distribution
from superglm.features.numeric import Numeric
from superglm.penalties.group_lasso import GroupLasso
from superglm.profiling.nb import NBProfileResult, estimate_nb_theta

# =====================================================================
# Helpers
# =====================================================================


def _generate_nb2(n, mu, theta, rng=None):
    """Simulate NB2(mu, theta) using scipy's nbinom."""
    if rng is None:
        rng = np.random.default_rng()
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)).copy()
    p = theta / (mu + theta)
    return rng.negative_binomial(theta, p).astype(np.float64)


# =====================================================================
# TestNB2Distribution
# =====================================================================


class TestNB2VarianceFunction:
    def test_basic(self):
        nb = NegativeBinomial(theta=5.0)
        mu = np.array([1.0, 2.0, 5.0, 10.0])
        expected = mu + mu**2 / 5.0
        np.testing.assert_allclose(nb.variance(mu), expected)

    def test_large_theta_approaches_poisson(self):
        """V(mu) = mu + mu^2/theta -> mu as theta -> inf."""
        nb = NegativeBinomial(theta=1e8)
        mu = np.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(nb.variance(mu), mu, rtol=1e-6)

    def test_small_theta_large_variance(self):
        nb = NegativeBinomial(theta=0.5)
        mu = np.array([5.0])
        expected = 5.0 + 25.0 / 0.5  # 55
        np.testing.assert_allclose(nb.variance(mu), expected)


class TestNB2DevianceUnit:
    def test_positive_y(self):
        nb = NegativeBinomial(theta=5.0)
        y = np.array([3.0, 7.0, 1.0])
        mu = np.array([2.0, 5.0, 3.0])
        d = nb.deviance_unit(y, mu)
        # All unit deviances should be non-negative
        assert np.all(d >= 0)

    def test_y_equals_mu(self):
        """Unit deviance at y=mu should be zero."""
        nb = NegativeBinomial(theta=5.0)
        mu = np.array([2.0, 5.0, 10.0])
        d = nb.deviance_unit(mu, mu)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_y_zero(self):
        """y=0 case uses special formula."""
        nb = NegativeBinomial(theta=5.0)
        y = np.array([0.0, 0.0])
        mu = np.array([2.0, 5.0])
        d = nb.deviance_unit(y, mu)
        expected = 2 * 5.0 * np.log((mu + 5.0) / 5.0)
        np.testing.assert_allclose(d, expected)
        assert np.all(d >= 0.0)

    def test_total_deviance_positive(self):
        nb = NegativeBinomial(theta=3.0)
        rng = np.random.default_rng(42)
        y = _generate_nb2(1000, mu=5.0, theta=3.0, rng=rng)
        mu = np.full_like(y, 5.0)
        d = nb.deviance_unit(y, mu)
        assert np.sum(d) > 0


class TestNB2LogLikelihood:
    def test_matches_scipy(self):
        """Log-likelihood should match scipy.stats.nbinom.logpmf."""
        nb = NegativeBinomial(theta=5.0)
        y = np.array([0, 1, 2, 5, 10], dtype=float)
        mu = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        weights = np.ones_like(y)

        ll_ours = nb.log_likelihood(y, mu, weights)

        # scipy nbinom: n=theta, p=theta/(mu+theta)
        p_nb = 5.0 / (3.0 + 5.0)
        ll_scipy = np.sum(nbinom.logpmf(y.astype(int), n=5.0, p=p_nb))

        np.testing.assert_allclose(ll_ours, ll_scipy, rtol=1e-10)

    def test_weighted(self):
        nb = NegativeBinomial(theta=5.0)
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([2.0, 2.0, 2.0])
        w1 = np.ones(3)
        w2 = np.array([2.0, 2.0, 2.0])
        ll1 = nb.log_likelihood(y, mu, w1)
        ll2 = nb.log_likelihood(y, mu, w2)
        np.testing.assert_allclose(ll2, 2.0 * ll1)


class TestNB2PoissonLimit:
    def test_large_theta_coefficients(self):
        """With very large theta, NB2 fit should give similar results to Poisson."""
        rng = np.random.default_rng(42)
        n = 5000
        x = rng.normal(0, 1, n)
        log_mu = 1.0 + 0.5 * x
        mu = np.exp(log_mu)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        # Poisson fit
        m_pois = SuperGLM(
            family="poisson", penalty=GroupLasso(lambda1=0.0), features={"x": Numeric()}
        )
        m_pois.fit(X, y)

        # NB2 with large theta
        m_nb = SuperGLM(
            family=NegativeBinomial(theta=1e6),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        m_nb.fit(X, y)

        np.testing.assert_allclose(m_nb.result.intercept, m_pois.result.intercept, atol=0.05)
        np.testing.assert_allclose(m_nb.result.beta, m_pois.result.beta, atol=0.05)


# =====================================================================
# TestNB2Fitting
# =====================================================================


class TestNB2FixedThetaFit:
    def test_convergence(self):
        """NB2 model with fixed theta converges on synthetic data."""
        rng = np.random.default_rng(42)
        n = 3000
        theta = 5.0
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=theta, rng=rng)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model.fit(X, y)

        assert model.result.converged
        # Check intercept near 1.0 and coef near 0.3
        np.testing.assert_allclose(model.result.intercept, 1.0, atol=0.15)

    def test_prediction_reasonable(self):
        rng = np.random.default_rng(42)
        n = 2000
        theta = 3.0
        mu_true = 5.0
        y = _generate_nb2(n, mu=mu_true, theta=theta, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        pred = model.predict(X)
        np.testing.assert_allclose(pred.mean(), mu_true, rtol=0.1)


# =====================================================================
# TestNB2Profile
# =====================================================================


class TestNB2ProfileTheta:
    def test_recovers_theta(self):
        """Profile estimation recovers theta from synthetic data."""
        rng = np.random.default_rng(42)
        n = 3000
        theta_true = 5.0
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=theta_true, rng=rng)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),  # initial guess
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )

        result = estimate_nb_theta(
            model,
            X,
            y,
            theta_bounds=(0.5, 20.0),
        )
        assert isinstance(result, NBProfileResult)
        np.testing.assert_allclose(result.theta_hat, theta_true, atol=2.0)

    def test_result_has_cache(self):
        rng = np.random.default_rng(42)
        n = 2000
        y = _generate_nb2(n, mu=5.0, theta=3.0, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )

        result = estimate_nb_theta(model, X, y, theta_bounds=(0.5, 15.0))
        assert len(result.cache) >= 1  # alternating alg converges in few iters
        assert result.n_evaluations >= 1

    def test_family_must_be_nb(self):
        model = SuperGLM(
            family="poisson", penalty=GroupLasso(lambda1=0.0), features={"x": Numeric()}
        )
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="NegativeBinomial"):
            estimate_nb_theta(model, X, y)

    def test_design_matrix_error_restores_temporary_family(self, monkeypatch):
        model = SuperGLM(
            family=NegativeBinomial(theta=2.5),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = np.array([1.0, 2.0, 3.0])

        def fail_build(*args, **kwargs):
            raise RuntimeError("build failed")

        monkeypatch.setattr(model, "_build_design_matrix", fail_build)

        with pytest.raises(RuntimeError, match="build failed"):
            estimate_nb_theta(model, X, y)

        assert model.family.theta == pytest.approx(2.5)


class TestNB2AutoTheta:
    @pytest.mark.parametrize("retain_fit_state", [True, False])
    def test_estimate_theta_reml_supports_intercept_only_zero_column_frame(self, retain_fit_state):
        rng = np.random.default_rng(20260823)
        n = 30
        X = pd.DataFrame(index=pd.RangeIndex(n))
        y = _generate_nb2(n, mu=3.0, theta=2.0, rng=rng)
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={},
            retain_fit_state=retain_fit_state,
        )

        result = model.estimate_theta(X, y, fit_mode="reml", maxiter=1)

        assert np.isfinite(result.theta_hat)
        assert (model._dm is not None) is retain_fit_state
        if retain_fit_state:
            assert model._dm.shape == (len(X), 0)
        assert model.result.beta.shape == (0,)
        assert model._last_fit_meta["method"] == "fit_reml"
        assert np.all(np.isfinite(model.predict(X)))

    @pytest.mark.parametrize("callback_stage", ["trace", "best_found", "final_refit"])
    @pytest.mark.parametrize("fit_mode", ["fit", "reml"])
    @pytest.mark.parametrize("retain_fit_state", [True, False])
    def test_callbacks_cannot_poison_profile_refit_inputs_or_configuration(
        self,
        monkeypatch,
        callback_stage,
        fit_mode,
        retain_fit_state,
    ):
        from superglm.model import fit_ops as fit_ops_module

        n = 24
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, n)})
        y = np.resize(np.array([0.0, 1.0, 2.0, 4.0]), n)
        sample_weight = np.linspace(0.5, 1.5, n)
        offset = np.linspace(-0.2, 0.2, n)
        baseline_X = X.copy(deep=True)
        baseline_y = y.copy()
        baseline_weight = sample_weight.copy()
        baseline_offset = offset.copy()
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
            retain_fit_state=retain_fit_state,
        )
        initial_config_revision = model._config_revision
        initial_fit_revision = model._fit_revision
        result = NBProfileResult(
            theta_hat=2.5,
            nll=1.2,
            n_evaluations=1,
            converged=True,
            cache={2.5: 1.2},
        )
        poisoned = False

        def poison_caller_state():
            nonlocal poisoned
            if poisoned:
                return
            poisoned = True
            X.iloc[:, 0] += 50.0
            y[:] += 7.0
            sample_weight[:] *= 3.0
            offset[:] += 1.0
            model._config.penalty.lambda1 = 0.5
            model._penalty_config.lambda1 = 0.75
            model._retain_fit_state = not retain_fit_state
            model._config_revision += 50
            model._fit_revision += 50

        def fake_profile(_candidate, _X, _y, **kwargs):
            if callback_stage == "trace":
                kwargs["trace_callback"]({"step": 0, "theta": 2.5, "nll": 1.2})
            return result

        monkeypatch.setattr("superglm.profiling.nb.estimate_nb_theta", fake_profile)
        fit_name = "_fit_reml_in_workspace" if fit_mode == "reml" else "_fit_in_workspace"
        real_fit = getattr(fit_ops_module, fit_name)
        captured = {}

        def capture_final_fit(candidate, X_arg, y_arg, weight_arg, offset_arg, **kwargs):
            captured["X"] = X_arg.copy(deep=True)
            captured["y"] = y_arg.copy()
            captured["weight"] = weight_arg.copy()
            captured["offset"] = offset_arg.copy()
            return real_fit(candidate, X_arg, y_arg, weight_arg, offset_arg, **kwargs)

        monkeypatch.setattr(fit_ops_module, fit_name, capture_final_fit)

        def progress_callback(stage, _payload):
            if stage == callback_stage:
                poison_caller_state()

        trace_callback = (
            (lambda _payload: poison_caller_state()) if callback_stage == "trace" else None
        )
        returned = model.estimate_theta(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode=fit_mode,
            trace_callback=trace_callback,
            progress_callback=progress_callback,
        )

        assert poisoned
        pd.testing.assert_frame_equal(captured["X"], baseline_X, check_column_type=False)
        np.testing.assert_array_equal(captured["y"], baseline_y)
        np.testing.assert_array_equal(captured["weight"], baseline_weight)
        np.testing.assert_array_equal(captured["offset"], baseline_offset)
        np.testing.assert_array_equal(model._nb_profile_result._y, baseline_y)
        np.testing.assert_array_equal(model._nb_profile_result._weights, baseline_weight)
        assert returned.theta_hat == pytest.approx(2.5)
        assert model.family.theta == pytest.approx(2.5)
        assert model._config.penalty.lambda1 == pytest.approx(0.0)
        assert model._penalty_config.lambda1 == pytest.approx(0.0)
        assert model._fit_state.resolved_penalty.lambda1 == pytest.approx(0.0)
        assert model.clone_unfitted().selection_penalty == pytest.approx(0.0)
        assert model._retain_fit_state is retain_fit_state
        assert model._config_revision == initial_config_revision + 1
        assert model._fit_revision == initial_fit_revision + 1
        if retain_fit_state:
            assert model._fit_X_ref is not X
            assert model._fit_y_ref is not y
            assert model._fit_sample_weight_ref is not sample_weight
            assert model._fit_offset_ref is not offset
            assert model._fit_data_guard.matches(
                baseline_X,
                baseline_y,
                baseline_weight,
                baseline_offset,
                fit_weights=model._fit_weights,
                fit_offset=model._fit_offset,
            )
        else:
            assert model._fit_X_ref is None
            assert model._fit_y_ref is None
            assert model._fit_sample_weight_ref is None
            assert model._fit_offset_ref is None

    @pytest.mark.parametrize("callback_stage", ["best_found", "final_refit"])
    def test_progress_callback_cannot_poison_nb_profile_publication(
        self, monkeypatch, callback_stage
    ):
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
        y = np.resize(np.array([0.0, 1.0, 2.0, 3.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        result = NBProfileResult(
            theta_hat=2.5,
            nll=1.2,
            n_evaluations=1,
            converged=True,
            cache={2.5: 1.2},
        )
        monkeypatch.setattr(
            "superglm.profiling.nb.estimate_nb_theta",
            lambda *_args, **_kwargs: result,
        )
        progress_events = []

        def poison_result(stage, payload):
            progress_events.append((stage, dict(payload["profile_estimate"])))
            if stage != callback_stage:
                return
            result.theta_hat = 9.0
            result.nll = -100.0
            result.cache[9.0] = -100.0

        returned = model.estimate_theta(X, y, progress_callback=poison_result)
        installed = model._nb_profile_result

        assert result.theta_hat == pytest.approx(9.0)
        assert returned.theta_hat == pytest.approx(2.5)
        assert installed.theta_hat == pytest.approx(2.5)
        assert model.family.theta == pytest.approx(2.5)
        assert 9.0 not in returned.cache
        assert 9.0 not in installed.cache
        assert returned is not installed
        assert returned._ci_cache is not installed._ci_cache
        assert [stage for stage, _ in progress_events] == ["best_found", "final_refit"]
        assert all(payload["value"] == pytest.approx(2.5) for _, payload in progress_events)
        assert all(payload["objective"] == pytest.approx(1.2) for _, payload in progress_events)

    def test_nb_profile_rejects_frame_metadata_without_copy_hooks(self, monkeypatch):
        class DeepcopyBomb:
            calls = 0

            def __deepcopy__(self, memo):
                type(self).calls += 1
                raise AssertionError("DataFrame metadata copy hook executed")

        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8)})
        X.attrs["unsafe"] = DeepcopyBomb()
        y = np.resize(np.array([0.0, 1.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        monkeypatch.setattr(
            "superglm.profiling.nb.estimate_nb_theta",
            lambda *_args, **_kwargs: pytest.fail("profile must not start"),
        )

        with pytest.raises(TypeError, match="Could not safely snapshot values in X"):
            model.estimate_theta(X, y)

        assert DeepcopyBomb.calls == 0
        assert model._result is None

    def test_nb_profile_rejects_mutable_object_cells_before_callbacks(self, monkeypatch):
        class MutableNumeric:
            def __init__(self, value):
                self.value = value

            def __float__(self):
                return float(self.value)

        cells = [MutableNumeric(value) for value in np.linspace(-1.0, 1.0, 8)]
        X = pd.DataFrame({"x": np.array(cells, dtype=object)})
        y = np.resize(np.array([0.0, 1.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        trace_calls = []
        monkeypatch.setattr(
            "superglm.profiling.nb.estimate_nb_theta",
            lambda *_args, **_kwargs: pytest.fail("profile must not start"),
        )

        with pytest.raises(TypeError, match="Could not safely snapshot values in X"):
            model.estimate_theta(X, y, trace_callback=trace_calls.append)

        assert trace_calls == []
        assert model._result is None

    def test_nb_profile_does_not_traverse_unused_mutable_columns(self, monkeypatch):
        class DeepcopyBomb:
            calls = 0

            def __deepcopy__(self, memo):
                type(self).calls += 1
                raise AssertionError("unused cell copy hook executed")

        n = 12
        X = pd.DataFrame(
            {
                "x": np.linspace(-1.0, 1.0, n),
                "unused": [DeepcopyBomb() for _ in range(n)],
            }
        )
        y = np.resize(np.array([0.0, 1.0, 2.0]), n)
        model = SuperGLM(
            family=NegativeBinomial(theta=1.0),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        monkeypatch.setattr(
            "superglm.profiling.nb.estimate_nb_theta",
            lambda *_args, **_kwargs: NBProfileResult(
                theta_hat=2.5,
                nll=1.2,
                n_evaluations=1,
                converged=True,
            ),
        )

        model.estimate_theta(X, y)

        assert DeepcopyBomb.calls == 0
        assert list(model._fit_X_ref.columns) == ["x"]
        assert model._fit_data_guard.x_columns == ("x",)

    def test_estimate_theta_inherit_preserves_reml_final_refit(self, monkeypatch):
        X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 80)})
        y = np.resize(np.array([1.0, 2.0, 3.0, 4.0]), len(X))
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"x": Numeric()},
        )
        model._last_fit_meta = {"method": "fit_reml"}
        result = NBProfileResult(
            theta_hat=2.5,
            nll=1.2,
            n_evaluations=1,
            converged=True,
        )
        configured_family = model._family_config
        configured_penalty = model._penalty_config
        configured_model = model._config
        configured_revision = model._config_revision

        def fake_profile(model_arg, X_arg, y_arg, sample_weight=None, offset=None, **kwargs):
            assert model_arg is not model
            assert model_arg.family.theta == "auto"
            assert X_arg is not X
            pd.testing.assert_frame_equal(X_arg, X, check_column_type=False)
            assert y_arg is not y
            np.testing.assert_array_equal(y_arg, y)
            return result

        monkeypatch.setattr("superglm.profiling.nb.estimate_nb_theta", fake_profile)

        returned = model.estimate_theta(X, y, fit_mode="inherit")

        assert returned is not result
        assert model._last_fit_meta["method"] == "fit_reml"
        assert model._config is not configured_model
        assert model._family_config is not configured_family
        assert model._penalty_config is not configured_penalty
        assert model._penalty_config.lambda1 == pytest.approx(configured_penalty.lambda1)
        assert model._config_revision == configured_revision + 1
        assert model._config.family.theta == pytest.approx(2.5)
        assert model.family.theta == pytest.approx(2.5)
        assert model.theta_ == pytest.approx(2.5)
        refit = model.clone_unfitted()
        assert refit.family.theta == pytest.approx(2.5)
        refit.fit(X, y)
        assert refit.family.theta == pytest.approx(2.5)
        assert refit.theta_ == pytest.approx(2.5)
        assert model._nb_profile_result is not result
        assert returned is not model._nb_profile_result
        assert model._fit_state.projections["_nb_profile_result"] is model._nb_profile_result

    def test_auto_theta_reml_uses_zero_selection_regime_when_unconfigured(self):
        rng = np.random.default_rng(20260718)
        x = np.linspace(-1.0, 1.0, 100)
        mu = np.exp(0.2 + 0.25 * x)
        y = rng.poisson(rng.gamma(shape=3.0, scale=mu / 3.0)).astype(np.float64)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            selection_penalty=None,
            features={"x": Spline(n_knots=5)},
        )

        model.fit_reml(X, y, max_reml_iter=1, max_pirls_iter=20)

        assert model.selection_penalty is None
        assert model.selection_penalty_ == pytest.approx(0.0)
        assert model.theta_ > 0.0

    def test_nb_profile_bcd_forwards_configured_smoothing(self, monkeypatch):
        rng = np.random.default_rng(20260719)
        x = np.linspace(-1.0, 1.0, 90)
        y = rng.poisson(np.exp(0.1 + 0.2 * x)).astype(np.float64)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=NegativeBinomial(theta=2.0),
            penalty=GroupLasso(lambda1=0.02),
            spline_penalty=0.75,
            features={"x": Spline(n_knots=5)},
        )
        from superglm.profiling import nb as nb_module

        real_fit_pirls = nb_module.fit_pirls
        seen_lambda2 = []

        def recording_fit_pirls(*args, **kwargs):
            seen_lambda2.append(kwargs.get("lambda2"))
            return real_fit_pirls(*args, **kwargs)

        monkeypatch.setattr(nb_module, "fit_pirls", recording_fit_pirls)

        estimate_nb_theta(model, X, y, maxiter=1)

        assert seen_lambda2 == [pytest.approx(0.75)]

    def test_estimate_theta_publishes_owned_detached_profile_result(self):
        rng = np.random.default_rng(20260720)
        x = np.linspace(-1.0, 1.0, 100)
        y = rng.poisson(np.exp(0.1 + 0.2 * x)).astype(np.float64)
        weights = np.linspace(0.5, 1.5, len(x))
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )

        returned = model.estimate_theta(X, y, sample_weight=weights, maxiter=1)
        installed = model._nb_profile_result
        installed_y = installed._y.copy()
        installed_weights = installed._weights.copy()

        assert returned is not installed
        assert returned.theta_hat == pytest.approx(installed.theta_hat)
        assert not np.shares_memory(installed._y, y)
        assert not np.shares_memory(installed._weights, weights)
        np.testing.assert_allclose(installed._mu, model._fit_mu, rtol=0.0, atol=0.0)
        for values in (installed._y, installed._mu, installed._weights):
            assert not values.flags.writeable
            with pytest.raises(ValueError):
                values.setflags(write=True)
        with pytest.raises(AttributeError, match="published"):
            returned.theta_hat = 99.0
        with pytest.raises(TypeError):
            returned.cache[99.0] = 0.0

        y[0] += 20.0
        weights[0] *= 20.0
        np.testing.assert_array_equal(installed._y, installed_y)
        np.testing.assert_array_equal(installed._weights, installed_weights)

    def test_auto_theta_flow(self):
        """nb_theta='auto' triggers profile estimation in fit()."""
        rng = np.random.default_rng(42)
        n = 2000
        theta_true = 5.0
        y = _generate_nb2(n, mu=5.0, theta=theta_true, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        # Configuration intent stays automatic; the learned value is fitted state.
        assert model.family.theta == "auto"
        assert model.theta_ > 0
        assert model.result.converged


# =====================================================================
# TestNB2QuantileResiduals
# =====================================================================


class TestNB2QuantileResiduals:
    def test_approx_normal(self):
        """Quantile residuals should be ~N(0,1) for well-specified NB2."""
        rng = np.random.default_rng(42)
        n = 5000
        theta = 5.0
        mu_true = 5.0
        y = _generate_nb2(n, mu=mu_true, theta=theta, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=theta),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        qr = metrics.residuals("quantile")

        # Should be approximately N(0,1)
        assert abs(qr.mean()) < 0.15
        assert abs(qr.std() - 1.0) < 0.15


# =====================================================================
# TestNB2MetricsSummary
# =====================================================================


class TestNB2MetricsSummary:
    def test_summary_works(self):
        rng = np.random.default_rng(42)
        n = 1000
        y = _generate_nb2(n, mu=5.0, theta=3.0, rng=rng)
        X = pd.DataFrame({"dummy": np.ones(n)})

        model = SuperGLM(
            family=NegativeBinomial(theta=3.0),
            penalty=GroupLasso(lambda1=0.0),
            features={"dummy": Numeric()},
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        summary = metrics.summary()
        text = str(summary)
        assert "NegativeBinomial" in text or "Neg. Binomial" in text


# =====================================================================
# TestNB2Sklearn
# =====================================================================


class TestNB2Sklearn:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(0, 1, n)
        mu = np.exp(1.0 + 0.3 * x)
        y = _generate_nb2(n, mu=mu, theta=5.0, rng=rng)
        X = pd.DataFrame({"x": x})

        reg = SuperGLMRegressor(
            family=NegativeBinomial(theta=5.0),
            selection_penalty=0.0,
        )
        reg.fit(X, y)
        pred = reg.predict(X)

        assert pred.shape == (n,)
        assert np.all(pred > 0)


# =====================================================================
# TestNB2Validation
# =====================================================================


class TestNB2InvalidTheta:
    def test_zero(self):
        with pytest.raises(ValueError, match="must be > 0"):
            NegativeBinomial(theta=0.0)

    def test_negative(self):
        with pytest.raises(ValueError, match="must be > 0"):
            NegativeBinomial(theta=-1.0)


class TestNB2ResolveDistribution:
    def test_resolve_object(self):
        dist = resolve_distribution(NegativeBinomial(theta=5.0))
        assert isinstance(dist, NegativeBinomial)
        assert dist.theta == 5.0

    def test_resolve_missing_theta(self):
        with pytest.raises(ValueError, match="requires parameters"):
            resolve_distribution("negative_binomial")

    def test_resolve_passthrough(self):
        nb = NegativeBinomial(theta=3.0)
        assert resolve_distribution(nb) is nb
