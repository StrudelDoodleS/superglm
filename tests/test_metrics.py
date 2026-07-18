"""Tests for ModelMetrics diagnostics module."""

from dataclasses import replace
from types import SimpleNamespace

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
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix, DiscretizedTensorGroupMatrix
from superglm.model.fit_state import FittedStateRevision


def _publish_result_revision(model, **changes) -> None:
    revision = FittedStateRevision.start(model)
    for result_name in ("_result", "_solver_result"):
        result = getattr(revision.model, result_name)
        for name, value in changes.items():
            setattr(result, name, value)
    revision.commit()


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
        selection_penalty=0.001,
        features={"x1": Numeric(), "x2": Numeric()},
    )
    model.fit(X, y, sample_weight=w)
    return model, X, y, w


@pytest.fixture
def metrics_obj(fitted_poisson):
    """ModelMetrics from the fitted Poisson model."""
    model, X, y, w = fitted_poisson
    return model.metrics(X, y, sample_weight=w)


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


class TestMetricsCaching:
    def test_metrics_reuses_fit_mu_on_training_data(self, fitted_poisson):
        """metrics() on fit data should reuse the cached fitted mean vector."""
        model, X, y, w = fitted_poisson
        metrics = model.metrics(X, y, sample_weight=w)

        assert metrics._mu is model._fit_mu

    def test_metrics_returns_cached_object_for_same_fit_refs(self, fitted_poisson):
        """Repeated metrics() on the exact fit refs should return the cached object."""
        model, X, y, w = fitted_poisson
        metrics1 = model.metrics(X, y, sample_weight=w)
        metrics2 = model.metrics(X, y, sample_weight=w)

        assert metrics1 is metrics2

    def test_mutated_fit_response_invalidates_identity_cache(self, fitted_poisson):
        """Object identity must not make mutated caller data look unchanged."""
        model, X, y, w = fitted_poisson
        cached = model.metrics(X, y, sample_weight=w)
        y[0] += 25.0

        refreshed = model.metrics(X, y, sample_weight=w)
        independent = model.metrics(X.copy(), y.copy(), sample_weight=w.copy())

        assert refreshed is not cached
        assert refreshed.log_likelihood == pytest.approx(independent.log_likelihood)

    def test_mutated_fit_features_invalidate_identity_cache(self, fitted_poisson):
        model, X, y, w = fitted_poisson
        cached = model.metrics(X, y, sample_weight=w)
        X.loc[X.index[0], "x1"] += 3.0

        refreshed = model.metrics(X, y, sample_weight=w)
        independent = model.metrics(X.copy(), y.copy(), sample_weight=w.copy())

        assert refreshed is not cached
        assert refreshed.log_likelihood == pytest.approx(independent.log_likelihood)

    def test_mutated_fit_weights_invalidate_identity_cache(self, fitted_poisson):
        model, X, y, w = fitted_poisson
        cached = model.metrics(X, y, sample_weight=w)
        w[0] *= 7.0

        refreshed = model.metrics(X, y, sample_weight=w)
        independent = model.metrics(X.copy(), y.copy(), sample_weight=w.copy())

        assert refreshed is not cached
        assert refreshed.log_likelihood == pytest.approx(independent.log_likelihood)

    def test_mutated_fit_offset_invalidates_identity_cache(self):
        x = np.linspace(-1.0, 1.0, 120)
        X = pd.DataFrame({"x": x})
        y = 0.4 + 0.7 * x
        offset = np.zeros_like(x)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y, offset=offset)
        cached = model.metrics(X, y, offset=offset)
        offset[0] = 2.0

        refreshed = model.metrics(X, y, offset=offset)
        independent = model.metrics(X.copy(), y.copy(), offset=offset.copy())

        assert refreshed is not cached
        assert refreshed.log_likelihood == pytest.approx(independent.log_likelihood)

    def test_changed_offset_recomputes_inverse_from_evaluation_working_weights(self):
        """Fit-time rank inverses are invalid when a new offset changes Fisher weights."""
        rng = np.random.default_rng(20260710)
        n = 240
        x = rng.normal(size=n)
        X = pd.DataFrame({"x": x})
        y = rng.poisson(np.exp(0.2 + 0.35 * x)).astype(float)
        weights = np.ones(n)
        fit_offset = np.zeros(n)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X, y, sample_weight=weights, offset=fit_offset)

        changed_offset = np.linspace(-0.8, 0.8, n)
        metrics = model.metrics(X, y, sample_weight=weights, offset=changed_offset)
        X_active, working_weights, inverse, _, _ = metrics._active_info
        X_active_dense = (
            X_active.toarray() if hasattr(X_active, "toarray") else np.asarray(X_active)
        )

        expected = np.linalg.inv(X_active_dense.T @ (working_weights[:, None] * X_active_dense))
        np.testing.assert_allclose(inverse, expected, rtol=1e-10, atol=1e-12)

        fit_inverse = model.result.rank_info.coefficient.pseudo_inverse()
        assert not np.allclose(inverse, fit_inverse, rtol=1e-5, atol=1e-8)

    def test_fit_rank_reuse_uses_solver_eta_before_binomial_mu_clipping(self):
        """Clipped public means must not change the retained fit working weights."""
        x = np.linspace(-1.0, 1.0, 120)
        X = pd.DataFrame({"x": x})
        y = (x > 0.0).astype(float)
        model = SuperGLM(
            family="binomial",
            selection_penalty=0.0,
            features={"x": Numeric()},
            max_iter=100,
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        _, working_weights, inverse, _, _ = metrics._active_info

        np.testing.assert_array_equal(working_weights, metrics._fit_working_weights)
        assert metrics._working_weights_match_fit(working_weights)
        np.testing.assert_allclose(
            inverse,
            model.result.rank_info.coefficient.pseudo_inverse(),
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "cauchit"])
    def test_unchanged_binomial_fit_reuses_rank_without_roundtrip_weight_comparison(self, link):
        """Fit identity, not inverse-link roundoff, controls rank-state reuse."""
        rng = np.random.default_rng(20260714)
        x = rng.normal(size=300)
        X = pd.DataFrame({"x": x})
        probability = 1.0 / (1.0 + np.exp(-(-0.25 + 0.7 * x)))
        y = rng.binomial(1, probability).astype(float)
        model = SuperGLM(
            family="binomial",
            link=link,
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X, y)

        metrics = model.metrics(X, y)
        _, working_weights, inverse, _, _ = metrics._active_info

        assert metrics._working_weights_match_fit(working_weights)
        np.testing.assert_allclose(
            inverse,
            model.result.rank_info.coefficient.pseudo_inverse(),
            rtol=0.0,
            atol=0.0,
        )

    def test_changed_offset_discrete_metrics_use_one_public_coordinate_system(self):
        """Changed-offset means, Fisher weights, and design use the exact public basis."""
        from superglm.distributions import _VARIANCE_FLOOR, clip_mu
        from superglm.model import base

        rng = np.random.default_rng(20260715)
        x = np.linspace(-3.0, 3.0, 360)
        X = pd.DataFrame({"x": x})
        y = rng.poisson(np.exp(0.2 + 0.35 * np.sin(x))).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins=16,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y, offset=np.zeros(len(X)))

        changed_offset = np.linspace(-0.45, 0.35, len(X))
        metrics = model.metrics(X, y, offset=changed_offset)
        design, working_weights, _, _, _ = metrics._active_info
        eta, working_mu = metrics._working_eta_mu
        expected_eta = base.predict_eta_exact(model, X, offset=changed_offset)
        expected_mu = clip_mu(model._link.inverse(expected_eta), model._distribution)
        expected_weights = model._link.deriv_inverse(expected_eta) ** 2 / np.maximum(
            model._distribution.variance(expected_mu), _VARIANCE_FLOOR
        )

        np.testing.assert_allclose(eta, expected_eta, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(working_mu, metrics._mu, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(working_weights, expected_weights, rtol=1e-13, atol=1e-15)
        assert type(design).__name__ == "EvaluationDesign"

    def test_different_gaussian_X_uses_evaluation_design_despite_equal_working_weights(self):
        """Evaluation diagnostics must not reuse fit geometry merely because W is unchanged."""
        x_fit = np.linspace(-1.0, 1.0, 80)
        X_fit = pd.DataFrame({"x": x_fit})
        y_fit = 1.25 + 0.8 * x_fit

        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X_fit, y_fit)

        x_eval = np.linspace(2.0, 5.0, len(X_fit))
        X_eval = pd.DataFrame({"x": x_eval})
        metrics = model.metrics(X_eval, 1.25 + 0.8 * x_eval)
        X_active, working_weights, inverse, _, _ = metrics._active_info
        X_active_dense = (
            X_active.toarray() if hasattr(X_active, "toarray") else np.asarray(X_active)
        )

        expected_design = x_eval[:, None]
        expected_inverse = np.linalg.inv(
            expected_design.T @ (working_weights[:, None] * expected_design)
        )
        expected_leverage = working_weights * np.sum(
            (expected_design @ expected_inverse) * expected_design,
            axis=1,
        )

        assert not metrics._working_weights_match_fit(working_weights)
        np.testing.assert_allclose(X_active_dense, expected_design, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(inverse, expected_inverse, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(metrics.leverage, expected_leverage, rtol=1e-12, atol=1e-12)

    def test_evaluation_large_translation_keeps_profiled_inference_stable(self):
        """Evaluation covariance must profile the intercept before rank work."""
        n = 1000
        x_fit = np.linspace(-1.0, 1.0, n)
        X_fit = pd.DataFrame({"x": x_fit})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X_fit, 1.0 + 0.75 * x_fit)

        x_eval = 1e12 + x_fit
        X_eval = pd.DataFrame({"x": x_eval})
        metrics = model.metrics(X_eval, 1.0 + 0.75 * x_fit)
        _, working_weights, _, inverse_augmented, _ = metrics._active_info

        anchored = x_eval - x_eval[0]
        centered = anchored - np.average(anchored, weights=working_weights)
        expected_gram = float(np.dot(working_weights, centered**2))

        assert metrics._active_centered_data_gram[0, 0] == pytest.approx(
            expected_gram,
            rel=1e-12,
        )
        assert inverse_augmented[1, 1] == pytest.approx(1.0 / expected_gram, rel=1e-12)

    def test_changed_offset_large_translation_uses_stable_grouped_centering(self):
        """Changed-W fit diagnostics must retain the fitted centered-system stability."""
        rng = np.random.default_rng(20260713)
        n = 500
        local_x = np.linspace(-1.0, 1.0, n)
        x = 1e10 + local_x
        X = pd.DataFrame({"x": x})
        y = rng.poisson(np.exp(0.1 + 0.2 * local_x)).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X, y, offset=np.zeros(n))

        metrics = model.metrics(X, y, offset=np.linspace(-0.3, 0.3, n))
        _, working_weights, _, inverse_augmented, _ = metrics._active_info
        anchored = x - x[0]
        centered = anchored - np.average(anchored, weights=working_weights)
        expected_gram = float(np.dot(working_weights, centered**2))

        assert metrics._active_centered_data_gram[0, 0] == pytest.approx(
            expected_gram,
            rel=1e-12,
        )
        assert inverse_augmented[1, 1] == pytest.approx(1.0 / expected_gram, rel=1e-12)

    def test_fit_discrete_tensor_inference_does_not_materialize_observation_rows(self, monkeypatch):
        """Fit-data SE, influence, and summary paths must stay on grouped tensor algebra."""
        support1 = np.linspace(18.0, 70.0, 12)
        support2 = np.linspace(0.0, 18.0, 9)
        pairs = np.array(np.meshgrid(support1, support2)).reshape(2, -1).T
        X = pd.DataFrame(
            {
                "age": np.tile(pairs[:, 0], 3),
                "vehicle_age": np.tile(pairs[:, 1], 3),
            }
        )
        eta = -0.7 + 0.012 * (X["age"].to_numpy() - 40.0) - 0.025 * X["vehicle_age"].to_numpy()
        rng = np.random.default_rng(20260711)
        y = rng.poisson(np.exp(eta)).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=True,
            n_bins=64,
            features={
                "age": Spline(n_knots=6, penalty="ssp"),
                "vehicle_age": Spline(n_knots=5, penalty="ssp"),
            },
            interactions=[("age", "vehicle_age")],
        )
        model.fit(X, y)
        assert any(
            isinstance(group_matrix, DiscretizedTensorGroupMatrix)
            for group_matrix in model._dm.group_matrices
        )
        metrics = model.metrics(X, y)

        monkeypatch.setattr(
            DiscretizedTensorGroupMatrix,
            "toarray",
            lambda _self: pytest.fail("metrics inference materialized full tensor rows"),
        )

        coefficient_se = metrics.coefficient_se
        edf, edf1 = metrics._influence_edf
        summary = metrics.summary()

        assert coefficient_se
        assert np.all(np.isfinite(edf))
        assert np.all(np.isfinite(edf1))
        assert summary._coef_rows

        _publish_result_revision(model, rank_info=None)
        legacy_metrics = ModelMetrics(model, X, y, _mu=model._fit_mu)
        assert legacy_metrics.coefficient_se
        assert legacy_metrics.summary()._coef_rows

        from superglm.model import state_ops

        legacy_design, _, _, _, _ = state_ops.fit_active_info(model)
        covariance, _ = state_ops.coef_covariance(model)
        assert isinstance(legacy_design, DesignMatrix)
        assert covariance.shape[0] == legacy_design.shape[1]

    def test_recomputed_influence_edf1_uses_centered_diag_f_squared(self):
        """Changed-W inference uses 2*diag(F)-diag(F@F) after profiling the intercept."""
        rng = np.random.default_rng(20260712)
        x = np.linspace(-2.0, 2.0, 240)
        X = pd.DataFrame({"x": x})
        y = rng.poisson(np.exp(0.1 + 0.4 * np.sin(1.5 * x))).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=2.5,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y, offset=np.zeros(len(X)))

        changed_offset = np.linspace(-0.9, 0.7, len(X))
        metrics = model.metrics(X, y, offset=changed_offset)
        X_active, W, _, inverse_augmented, _ = metrics._active_info
        X_dense = X_active.toarray() if hasattr(X_active, "toarray") else np.asarray(X_active)

        sum_w = float(np.sum(W))
        mean_x = (X_dense.T @ W) / sum_w
        X_centered = X_dense - mean_x
        data_gram = X_centered.T @ (W[:, None] * X_centered)
        influence = inverse_augmented[1:, 1:] @ data_gram
        expected_edf = np.diag(influence)
        expected_edf1 = 2.0 * expected_edf - np.diag(influence @ influence)

        actual_edf, actual_edf1 = metrics._influence_edf

        np.testing.assert_allclose(actual_edf, expected_edf, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(actual_edf1, expected_edf1, rtol=1e-11, atol=1e-12)

    def test_coef_table_fallback_uses_diag_f_squared(self, monkeypatch):
        """Standalone coefficient rows use diag(F@F), not row squared norms."""
        from superglm.inference.coef_tables import build_coef_rows
        from superglm.stats import wood_pvalue

        x = np.linspace(-3.0, 3.0, 180)
        X = pd.DataFrame({"x": x})
        y = 0.3 + np.sin(x) + 0.05 * np.cos(4.0 * x)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            spline_penalty=4.0,
            features={"x": Spline(n_knots=9, penalty="ssp")},
        )
        model.fit(X, y)
        p = len(model.result.beta)
        synthetic_rng = np.random.default_rng(411)
        factor = synthetic_rng.normal(size=(p, p))
        data_gram = factor.T @ factor + np.eye(p)
        penalty = np.diag(np.geomspace(0.05, 30.0, p))
        inverse = np.linalg.inv(data_gram + penalty)
        X_dense = np.linalg.cholesky(data_gram).T
        W = np.ones(p)
        inverse_augmented = np.zeros((p + 1, p + 1))
        inverse_augmented[0, 0] = 1.0 / p
        inverse_augmented[1:, 1:] = inverse
        active_groups = model._groups
        mean_x = np.mean(X_dense, axis=0)
        X_centered = X_dense - mean_x
        centered_data_gram = X_centered.T @ X_centered
        influence = inverse_augmented[1:, 1:] @ centered_data_gram
        expected_edf1 = 2.0 * np.diag(influence) - np.diag(influence @ influence)
        old_edf1 = 2.0 * np.diag(influence) - np.sum(influence * influence, axis=1)
        assert not np.allclose(expected_edf1, old_edf1, rtol=1e-8, atol=1e-10)
        grouped_design = DesignMatrix(
            [DenseGroupMatrix(X_dense)],
            n=p,
            p=p,
        )

        def expose_edf1(_beta, _X, _covariance, edf1, _residual_df):
            return 0.0, 1.0, float(edf1)

        monkeypatch.setattr(wood_pvalue, "wood_test_smooth", expose_edf1)
        rows = build_coef_rows(
            groups=model._groups,
            specs=model._specs,
            interaction_specs=model._interaction_specs,
            result=model.result,
            X_a=grouped_design,
            W=W,
            XtWX_inv=inverse,
            XtWX_inv_aug=inverse_augmented,
            active_groups=active_groups,
            known_scale=model._distribution.scale_known,
            group_edf_map=None,
            reml_lambdas=getattr(model, "_reml_lambdas", None),
            lambda2=model.lambda2,
            n_obs=p,
        )
        spline_row = next(row for row in rows if row.is_spline)

        assert spline_row.ref_df == pytest.approx(float(np.sum(expected_edf1)))

    def test_different_X_without_rank_info_uses_selected_evaluation_groups(self):
        """Legacy fitted results must not silently pair evaluation W with the fit design."""
        x_fit = np.linspace(-1.0, 1.0, 60)
        X_fit = pd.DataFrame({"x": x_fit})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X_fit, 0.5 + 1.7 * x_fit)
        _publish_result_revision(model, rank_info=None)

        x_eval = np.linspace(3.0, 7.0, len(X_fit))
        X_eval = pd.DataFrame({"x": x_eval})
        metrics = ModelMetrics(model, X_eval, 0.5 + 1.7 * x_eval)
        X_active, W, inverse, _, _ = metrics._active_info
        X_dense = X_active.toarray() if hasattr(X_active, "toarray") else np.asarray(X_active)
        expected = x_eval[:, None]

        np.testing.assert_allclose(X_dense, expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            inverse,
            np.linalg.inv(expected.T @ (W[:, None] * expected)),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_legacy_unpenalized_zero_coefficient_remains_selected(self):
        """Legacy fits must not confuse a valid zero estimate with deselection."""
        x = np.linspace(-1.0, 1.0, 100)
        X = pd.DataFrame({"x": x})
        y = 2.5 + 0.1 * np.cos(np.pi * x)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        assert model.result.beta[0] == pytest.approx(0.0, abs=1e-14)
        _publish_result_revision(model, rank_info=None)

        metrics = ModelMetrics(model, X, y, _mu=model._fit_mu)
        design, _, _, _, active_groups = metrics._active_info

        assert design.shape[1] == 1
        assert [group.name for group in active_groups] == ["x"]
        assert metrics.leverage.sum() > 0.9
        summary_row = next(row for row in model.summary()._coef_rows if row.name == "x")
        assert summary_row.se > 0.0
        assert summary_row.edf > 0.9

    def test_direct_metrics_without_X_uses_retained_fit_frame(self, fitted_poisson):
        """The documented fit-frame fallback must pass the resolved frame to predict."""
        model, X, y, w = fitted_poisson

        metrics = ModelMetrics(model, y=y, sample_weight=w)

        assert metrics._X is X
        np.testing.assert_allclose(metrics._mu, model.predict(X), rtol=0.0, atol=0.0)

    def test_evaluation_aliases_override_fit_time_estimability(self):
        """Rank loss on evaluation rows must mark individual aliases non-estimable."""
        rng = np.random.default_rng(20260716)
        x1 = rng.normal(size=240)
        x2 = rng.normal(size=240)
        X_fit = pd.DataFrame({"x1": x1, "x2": x2})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(X_fit, 0.5 + 0.8 * x1 - 0.3 * x2)

        X_eval = pd.DataFrame({"x1": x1, "x2": x1})
        metrics = model.metrics(X_eval, 0.5 + 0.5 * x1)
        rows = {row.name: row for row in metrics._build_coef_rows()}

        for name in ("x1", "x2"):
            assert not rows[name].estimable
            assert np.isnan(rows[name].se)
        assert metrics.leverage.sum() == pytest.approx(1.0, rel=1e-10, abs=1e-10)

    @pytest.mark.parametrize("alias_scale", [1.0, 1e4, 1e6, 1e8])
    def test_evaluation_alias_rank_is_stable_across_column_scales(self, alias_scale):
        """Moment-roundoff near the Gram cutoff must not revive an exact alias."""
        x = np.linspace(-2.0, 2.0, 120)
        X_fit = pd.DataFrame({"x1": x, "x2": np.cos(x)})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(X_fit, 1.0 + 2.0 * x)

        X_eval = pd.DataFrame({"x1": x, "x2": alias_scale * x})
        metrics = model.metrics(X_eval, 1.0 + 2.0 * x)
        rows = {row.name: row for row in metrics._build_coef_rows()}

        for name in ("x1", "x2"):
            assert not rows[name].estimable
            assert np.isnan(rows[name].se)
        assert metrics.leverage.sum() == pytest.approx(1.0, rel=1e-10, abs=1e-10)

    def test_evaluation_missing_category_marks_level_nonestimable(self):
        """Categorical rows propagate evaluation-rank loss into the estimable flag."""
        categories = ["A", "B", "C"]
        fit_levels = np.tile(categories, 60)
        X_fit = pd.DataFrame({"category": pd.Categorical(fit_levels, categories=categories)})
        y_fit = np.select(
            [fit_levels == "B", fit_levels == "C"],
            [1.5, 2.0],
            default=0.5,
        )
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"category": Categorical(base="first")},
        )
        model.fit(X_fit, y_fit)

        eval_levels = np.tile(["A", "B"], 60)
        X_eval = pd.DataFrame({"category": pd.Categorical(eval_levels, categories=categories)})
        metrics = model.metrics(X_eval, np.where(eval_levels == "B", 1.5, 0.5))
        row = next(row for row in metrics._build_coef_rows() if row.name == "category[C]")

        assert np.isnan(row.se)
        assert not row.estimable

    def test_numeric_evaluation_summary_does_not_build_smooth_R_factor(self, monkeypatch):
        """Parametric summaries must not eagerly pay for a Wood-test factorization."""
        import superglm.inference.coef_tables as coef_tables
        import superglm.inference.metrics as metrics_module

        x = np.linspace(-1.0, 1.0, 120)
        X_fit = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X_fit, 0.5 + 1.2 * x)
        metrics = model.metrics(pd.DataFrame({"x": x + 2.0}), 0.5 + 1.2 * (x + 2.0))

        def unexpected_factor(_gram):
            pytest.fail("numeric summary built an unused smooth-test factor")

        monkeypatch.setattr(metrics_module, "factor_from_gram", unexpected_factor)
        monkeypatch.setattr(coef_tables, "factor_from_gram", unexpected_factor)

        assert metrics._build_coef_rows()

    def test_changed_weight_inference_skips_discarded_augmented_decomposition(self, monkeypatch):
        """Reweighted inference decomposes only the raw and profiled systems it consumes."""
        import superglm.inference.covariance as covariance_module
        import superglm.inference.metrics as metrics_module
        from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan

        rng = np.random.default_rng(20260718)
        X = pd.DataFrame({"x": rng.normal(size=180), "z": rng.normal(size=180)})
        y = 0.4 + 0.7 * X["x"].to_numpy() - 0.2 * X["z"].to_numpy()
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric(), "z": Numeric()},
        )
        model.fit(X, y)
        metrics = model.metrics(X, y, sample_weight=np.linspace(0.7, 1.3, len(X)))

        covariance_calls = 0
        metrics_calls = 0
        original = covariance_module.decompose_gram

        def counted_covariance_decomposition(matrix, *args, **kwargs):
            nonlocal covariance_calls
            covariance_calls += 1
            return original(matrix, *args, **kwargs)

        def counted_metrics_decomposition(matrix, *args, **kwargs):
            nonlocal metrics_calls
            metrics_calls += 1
            return original(matrix, *args, **kwargs)

        monkeypatch.setattr(
            covariance_module,
            "decompose_gram",
            counted_covariance_decomposition,
        )
        monkeypatch.setattr(
            MatrixExecutionPlan,
            "moments",
            lambda *_args, **_kwargs: pytest.fail("rebuilt the grouped raw Gram"),
        )
        monkeypatch.setattr(metrics_module, "decompose_gram", counted_metrics_decomposition)

        _ = metrics._active_info

        assert covariance_calls == 0
        assert metrics_calls == 2  # raw inverse + reusable profiled data rank

    def test_numeric_evaluation_does_not_allocate_full_fitted_penalty(self, monkeypatch):
        """Evaluation penalty assembly stays in compact active coordinates."""
        import superglm.reml.penalty_algebra as penalty_algebra

        rng = np.random.default_rng(20260719)
        X_fit = pd.DataFrame({f"x{i}": rng.normal(size=100) for i in range(12)})
        features = {name: Numeric() for name in X_fit}
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features=features,
        )
        model.fit(X_fit, 0.5 + 0.2 * X_fit["x0"].to_numpy())
        X_eval = X_fit.copy()

        monkeypatch.setattr(
            penalty_algebra,
            "build_penalty_matrix",
            lambda *_args, **_kwargs: pytest.fail("allocated a full fitted-width penalty"),
        )

        metrics = model.metrics(X_eval, 0.5 + 0.2 * X_eval["x0"].to_numpy())
        assert metrics.coefficient_se

    def test_evaluation_skips_transforms_for_unselected_terms(self, monkeypatch):
        """Sparse evaluation should transform only terms represented in active coordinates."""
        rng = np.random.default_rng(20260717)
        X_fit = pd.DataFrame({"x1": rng.normal(size=180), "x2": rng.normal(size=180)})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=100.0,
            penalty_features=["x2"],
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(X_fit, 1.0 + 0.7 * X_fit["x1"].to_numpy())
        assert model.result.rank_info.selected_group_names == ("x1",)
        monkeypatch.setattr(
            model._specs["x2"],
            "transform",
            lambda _values: pytest.fail("unselected x2 transform was evaluated"),
        )

        X_eval = X_fit.copy()
        metrics = model.metrics(X_eval, 1.0 + 0.7 * X_eval["x1"].to_numpy())

        assert metrics.coefficient_se

    def test_dense_weighted_moments_are_stable_under_large_translation(self):
        """Legacy dense designs use anchor-centered accumulation before profiling."""
        from superglm.inference._metrics_design import weighted_moments

        x = 1e12 + np.linspace(-1.0, 1.0, 80)
        weights = np.linspace(0.5, 1.5, len(x))
        _, _, centered = weighted_moments(x[:, None], weights)
        shifted = x - x[0]
        expected = np.dot(weights, (shifted - np.average(shifted, weights=weights)) ** 2)

        assert centered[0, 0] == pytest.approx(expected, rel=1e-12)

    def test_evaluation_chunk_budget_accounts_for_live_row_buffers(self):
        """Advertised row-buffer budget covers transform, projection, and algebra temporaries."""
        from superglm.inference._metrics_design import (
            _MAX_DESIGN_CHUNK_BYTES,
            EvaluationDesign,
        )

        width = 4096
        fake_model = SimpleNamespace(result=SimpleNamespace(beta=np.zeros(width)))
        design = EvaluationDesign(fake_model, pd.DataFrame({"x": [0.0]}), np.arange(width))
        one_buffer_bytes = design.chunk_rows * width * np.dtype(np.float64).itemsize

        assert 5 * one_buffer_bytes <= _MAX_DESIGN_CHUNK_BYTES

    def test_sparse_evaluation_chunk_size_uses_selected_term_width(self):
        """A wide inactive model must not force tiny batches for one selected term."""
        from superglm.inference._metrics_design import EvaluationDesign

        width = 4096
        prediction_plan = {
            "features": [
                {"beta_idx": np.array([0])},
                {"beta_idx": np.arange(1, width)},
            ],
            "interactions": [],
        }
        fake_model = SimpleNamespace(
            result=SimpleNamespace(beta=np.zeros(width)),
            _prediction_plan=prediction_plan,
        )

        design = EvaluationDesign(fake_model, pd.DataFrame({"x": [0.0]}), np.array([0]))

        assert design.chunk_rows == 8192

    def test_evaluation_summary_omits_fit_only_categorical_level_counts(self):
        """Evaluation summaries must not label training category counts as evaluation counts."""
        category_fit = np.array(["A"] * 40 + ["B"] * 40)
        X_fit = pd.DataFrame({"category": category_fit})
        y_fit = np.where(category_fit == "B", 2.0, 0.5)
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"category": Categorical(base="first")},
        )
        model.fit(X_fit, y_fit)

        category_eval = np.array(["A"] * 10 + ["B"] * 70)
        X_eval = pd.DataFrame({"category": category_eval})
        metrics = model.metrics(X_eval, np.where(category_eval == "B", 2.0, 0.5))
        category_row = next(row for row in metrics._build_coef_rows() if "category[" in row.name)

        assert category_row.level_n_obs is None
        assert category_row.level_exposure_share is None

    def test_evaluation_summary_reuses_one_bounded_design_moment_pass(self, monkeypatch):
        """Inverse, R, and EDF construction share one evaluation-design traversal."""
        x_fit = np.linspace(-1.0, 1.0, 120)
        X_fit = pd.DataFrame({"x": x_fit})
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X_fit, 0.25 + 0.6 * x_fit)

        x_eval = np.linspace(2.0, 4.0, len(X_fit))
        X_eval = pd.DataFrame({"x": x_eval})
        metrics = model.metrics(X_eval, 0.25 + 0.6 * x_eval)
        spec = model._specs["x"]
        original_transform = spec.transform
        transformed_rows: list[int] = []

        def counted_transform(values):
            transformed_rows.append(len(values))
            return original_transform(values)

        monkeypatch.setattr(spec, "transform", counted_transform)

        metrics.summary()

        assert transformed_rows == [len(X_eval)]

    def test_direct_metrics_without_X_uses_fitted_design(self, fitted_poisson):
        """Simulation diagnostics may supply fitted means without repeating fit X."""
        model, _, y, weights = fitted_poisson
        metrics = ModelMetrics(model, y=y, sample_weight=weights, _mu=model._fit_mu)

        X_active, _, _, _, _ = metrics._active_info

        assert X_active.shape[0] == len(y)


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


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        pytest.param("poisson", True, id="poisson"),
        pytest.param("binomial", True, id="binomial"),
        pytest.param("negative_binomial", True, id="negative-binomial"),
        pytest.param("gaussian", False, id="gaussian"),
        pytest.param("gamma", False, id="gamma"),
        pytest.param("tweedie", False, id="tweedie"),
    ],
)
def test_metrics_known_scale_dispatch_matches_distribution_contract(family, expected):
    """Inference reference laws must follow the family's dispersion contract."""
    from superglm.distributions import Binomial, Gaussian, NegativeBinomial

    families = {
        "poisson": Poisson(),
        "binomial": Binomial(),
        "negative_binomial": NegativeBinomial(theta=2.5),
        "gaussian": Gaussian(),
        "gamma": Gamma(),
        "tweedie": Tweedie(p=1.5),
    }
    metrics = object.__new__(ModelMetrics)
    metrics._family = families[family]

    assert metrics._known_scale is expected


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

        model = SuperGLM(family="poisson", selection_penalty=0, features={"x": Numeric()})
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

        model = SuperGLM(family="poisson", selection_penalty=0, features={"x": Numeric()})
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
            selection_penalty=0.001,
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
        from superglm.profiling.tweedie import generate_tweedie_cpg

        rng = np.random.default_rng(42)
        n = 2000
        x = rng.standard_normal(n)
        mu = np.exp(1.0 + 0.3 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=1.5, rng=rng)
        y = np.maximum(y, 0.0)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.001,
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
        from superglm.inference.summary import ModelSummary

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

    def test_summary_str_contains_extended_ic(self, metrics_obj):
        """ASCII output shows AICc/BIC/EBIC in the header."""
        text = str(metrics_obj.summary())
        assert "AICc" in text
        assert "BIC" in text
        assert "EBIC" in text

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
        """All box-framed lines should be the same width."""
        text = str(metrics_obj.summary())
        # Lines starting with box-drawing border chars
        box_lines = [
            line
            for line in text.split("\n")
            if line and line[0] in "\u2554\u2551\u2560\u255f\u255a"
        ]
        assert len(box_lines) >= 4
        widths = {len(line) for line in box_lines}
        assert len(widths) == 1, f"Inconsistent widths: {widths}"


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
            selection_penalty=0.001,
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
            selection_penalty=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=w)
        m = model.metrics(X, y, sample_weight=w)

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
        m = model.metrics(X, y, sample_weight=w)
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
        m = model.metrics(X, y, sample_weight=w)
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
            selection_penalty=0.5,  # high penalty to zero out x2
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
        m = model.metrics(X, y, sample_weight=w)
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
            selection_penalty=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y, sample_weight=w)
        m = model.metrics(X, y, sample_weight=w)

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
            selection_penalty=0.001,
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
            selection_penalty=0.01,
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
            selection_penalty=0.001,
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
            selection_penalty=0.001,
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
            selection_penalty=0.001,
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
            selection_penalty=0,
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
            selection_penalty=0,
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
            selection_penalty=0.001,
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

    @staticmethod
    def _profile_result(
        *,
        phi_method="mle",
        method="brent",
        density_exact=True,
        ci_cache=None,
    ):
        def unexpected_ci(*args, **kwargs):
            raise AssertionError("summary reporting must not evaluate a Tweedie profile CI")

        return SimpleNamespace(
            p_hat=1.55,
            phi_hat=0.8,
            nll=11.0,
            method=method,
            phi_method=phi_method,
            density_exact=density_exact,
            _ci_cache={} if ci_cache is None else dict(ci_cache),
            ci=unexpected_ci,
            ci_details=unexpected_ci,
        )

    def test_tweedie_profile_summary(self):
        from superglm.profiling.tweedie import generate_tweedie_cpg

        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 5, n)
        mu = np.exp(1.0 + 0.2 * x)
        y = generate_tweedie_cpg(n, mu, phi=1.0, p=1.5, rng=rng)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.estimate_p(X, y, p_bounds=(1.1, 1.9), phi_method="mle")
        m = model.metrics(X, y)
        text = str(m.summary())
        assert "Tweedie p" in text
        html = m.summary()._repr_html_()
        assert "Tweedie p" in html

    def test_model_summary_reports_uncached_mle_ci_without_computing_it(self, fitted_poisson):
        model, _, _, _ = fitted_poisson
        profile = self._profile_result()
        model._tweedie_profile_result = profile
        model._summary_cache = None

        summary = model.summary()

        assert summary._info["tweedie_p_ci"] is None
        assert summary._info["tweedie_p_ci_status"] == "not computed"
        assert summary._info["tweedie_p_method"] == "Profile MLE (Brent)"
        assert "CI not computed" in str(summary)
        assert "CI not computed" in summary._repr_html_()

    def test_metrics_summary_reports_uncached_mle_ci_without_computing_it(self, fitted_poisson):
        model, X, y, weights = fitted_poisson
        profile = self._profile_result()
        model._tweedie_profile_result = profile

        summary = model.metrics(X, y, sample_weight=weights).summary()

        assert summary._info["tweedie_p_ci"] is None
        assert summary._info["tweedie_p_ci_status"] == "not computed"
        assert summary._info["tweedie_p_method"] == "Profile MLE (Brent)"

    def test_pearson_summary_ignores_stale_cached_lr_interval(self, fitted_poisson):
        model, X, y, weights = fitted_poisson
        profile = self._profile_result(
            phi_method="pearson",
            ci_cache={0.05: (1.4, 1.7)},
        )
        model._tweedie_profile_result = profile
        model._summary_cache = None

        model_summary = model.summary()
        metrics_summary = model.metrics(X, y, sample_weight=weights).summary()

        for summary in (model_summary, metrics_summary):
            assert summary._info["tweedie_p_ci"] is None
            assert summary._info["tweedie_p_ci_status"] == "unavailable for Pearson plug-in"
            assert summary._info["tweedie_p_method"] == (
                "Approximate profile (Brent; Pearson plug-in)"
            )
            assert "1.400" not in str(summary)
            assert "CI unavailable for Pearson plug-in" in str(summary)
            assert "CI unavailable for Pearson plug-in" in summary._repr_html_()

    def test_summary_requires_exact_mle_method_value_for_cached_ci(self, fitted_poisson):
        model, _, _, _ = fitted_poisson
        profile = self._profile_result(
            phi_method="MLE",
            ci_cache={0.05: (1.4, 1.7)},
        )
        model._tweedie_profile_result = profile
        model._summary_cache = None

        summary = model.summary()

        assert summary._info["tweedie_p_ci"] is None
        assert summary._info["tweedie_p_ci_status"] == "not computed"
        assert summary._info["tweedie_p_method"] == "Profile (Brent)"

    def test_model_summary_refreshes_when_cached_ci_or_search_identity_changes(
        self, fitted_poisson
    ):
        model, _, _, _ = fitted_poisson
        profile = self._profile_result()
        model._tweedie_profile_result = profile
        model._summary_cache = None

        uncached = model.summary()
        profile._ci_cache[0.05] = (1.4, 1.7)
        cached = model.summary()

        assert cached is not uncached
        assert cached._info["tweedie_p_ci"] == (1.4, 1.7)
        assert cached._info["tweedie_p_ci"] is profile._ci_cache[0.05]
        assert cached._info["tweedie_p_ci_status"] == "available"
        assert "1.550 [1.400, 1.700]" in str(cached)
        assert model.summary() is cached

        profile.method = "grid_refine"
        changed_method = model.summary()
        assert changed_method is not cached
        assert changed_method._info["tweedie_p_method"] == "Profile MLE (Grid Refine)"

        replacement = self._profile_result(
            method="grid_refine",
            ci_cache={0.05: (1.4, 1.7)},
        )
        model._tweedie_profile_result = replacement
        changed_search = model.summary()
        assert changed_search is not changed_method

    def test_model_summary_refreshes_for_equal_valued_ci_tuple_replacement(self, fitted_poisson):
        model, _, _, _ = fitted_poisson
        first_interval = tuple([1.4, 1.7])
        profile = self._profile_result(ci_cache={0.05: first_interval})
        model._tweedie_profile_result = profile
        model._summary_cache = None

        first_summary = model.summary()
        replacement_interval = tuple([1.4, 1.7])
        assert replacement_interval is not first_interval
        profile._ci_cache[0.05] = replacement_interval
        replacement_summary = model.summary()

        assert replacement_summary is not first_summary
        assert replacement_summary._info["tweedie_p_ci"] is replacement_interval
        assert replacement_summary._info["tweedie_p_ci_status"] == "available"

    def test_model_summary_refreshes_after_ci_cache_clear_and_recompute(self, fitted_poisson):
        model, _, _, _ = fitted_poisson
        first_interval = tuple([1.4, 1.7])
        profile = self._profile_result(ci_cache={0.05: first_interval})
        model._tweedie_profile_result = profile
        model._summary_cache = None

        cached_summary = model.summary()
        assert cached_summary._info["tweedie_p_ci"] is first_interval

        profile._ci_cache.clear()
        cleared_summary = model.summary()
        assert cleared_summary is not cached_summary
        assert cleared_summary._info["tweedie_p_ci"] is None

        recomputed_interval = tuple([1.4, 1.7])
        profile._ci_cache[0.05] = recomputed_interval
        recomputed_summary = model.summary()

        assert recomputed_summary is not cleared_summary
        assert recomputed_summary._info["tweedie_p_ci"] is recomputed_interval
        assert recomputed_summary._info["tweedie_p_ci_status"] == "available"

    @pytest.mark.parametrize(
        ("attribute", "new_value", "info_key", "expected"),
        [
            ("p_hat", 1.62, "tweedie_p", 1.62),
            ("phi_hat", 0.91, "tweedie_phi", 0.91),
            ("nll", 9.75, "tweedie_profile_nll", 9.75),
            ("method", "grid_refine", "tweedie_p_method", "Profile MLE (Grid Refine)"),
            (
                "density_exact",
                False,
                "tweedie_p_method",
                "Profile MLE (Brent; density approximation)",
            ),
        ],
    )
    def test_model_summary_refreshes_when_rendered_profile_state_changes(
        self,
        fitted_poisson,
        attribute,
        new_value,
        info_key,
        expected,
    ):
        model, _, _, _ = fitted_poisson
        interval = tuple([1.4, 1.7])
        profile = self._profile_result(ci_cache={0.05: interval})
        model._tweedie_profile_result = profile
        model._summary_cache = None

        before = model.summary()
        setattr(profile, attribute, new_value)
        after = model.summary()

        assert after is not before
        assert after._info[info_key] == expected
        assert after._info["tweedie_p_ci"] is interval

    def test_model_summary_refreshes_when_phi_method_becomes_exact_mle(self, fitted_poisson):
        model, _, _, _ = fitted_poisson
        interval = tuple([1.4, 1.7])
        profile = self._profile_result(phi_method="unknown", ci_cache={0.05: interval})
        model._tweedie_profile_result = profile
        model._summary_cache = None

        before = model.summary()
        profile.phi_method = "mle"
        after = model.summary()

        assert after is not before
        assert after._info["tweedie_p_method"] == "Profile MLE (Brent)"
        assert after._info["tweedie_p_ci"] is interval

    def test_summary_rejects_str_subclass_spoofing_exact_mle(self, fitted_poisson):
        class SpoofedMethod(str):
            def __eq__(self, other):
                return other == "mle"

        model, _, _, _ = fitted_poisson
        profile = self._profile_result(
            phi_method=SpoofedMethod("not-mle"),
            ci_cache={0.05: (1.4, 1.7)},
        )
        model._tweedie_profile_result = profile
        model._summary_cache = None

        summary = model.summary()

        assert summary._info["tweedie_p_ci"] is None
        assert summary._info["tweedie_p_ci_status"] == "not computed"
        assert summary._info["tweedie_p_method"] == "Profile (Brent)"

    def test_summary_rejects_dict_subclass_without_calling_overridden_get(self, fitted_poisson):
        calls = []

        class HostileCache(dict):
            def get(self, key, default=None):
                calls.append((key, default))
                return super().get(key, default)

        model, _, _, _ = fitted_poisson
        profile = self._profile_result()
        profile._ci_cache = HostileCache({0.05: (1.4, 1.7)})
        model._tweedie_profile_result = profile
        model._summary_cache = None

        summary = model.summary()

        assert calls == []
        assert summary._info["tweedie_p_ci"] is None
        assert summary._info["tweedie_p_ci_status"] == "not computed"

    def test_summary_rejects_tuple_subclass_without_calling_overridden_access(self, fitted_poisson):
        calls = []

        class HostileInterval(tuple):
            def __len__(self):
                calls.append("len")
                return super().__len__()

            def __iter__(self):
                calls.append("iter")
                return super().__iter__()

            def __getitem__(self, index):
                calls.append(("getitem", index))
                return super().__getitem__(index)

        model, _, _, _ = fitted_poisson
        profile = self._profile_result()
        profile._ci_cache[0.05] = HostileInterval((1.4, 1.7))
        model._tweedie_profile_result = profile
        model._summary_cache = None

        summary = model.summary()

        assert calls == []
        assert summary._info["tweedie_p_ci"] is None
        assert summary._info["tweedie_p_ci_status"] == "not computed"

    def test_profile_report_identity_stays_hashable_for_unhashable_legacy_metadata(self):
        from superglm.profiling._reporting import tweedie_profile_report_identity

        result = SimpleNamespace(
            p_hat=[],
            phi_hat={},
            nll=set(),
            method=[],
            phi_method={},
            density_exact=[],
            _ci_cache={},
        )

        identity = tweedie_profile_report_identity(result, 0.05)

        assert isinstance(hash(identity), int)

    def test_summary_qualifies_approximation_based_density(self, fitted_poisson):
        model, _, _, _ = fitted_poisson
        model._tweedie_profile_result = self._profile_result(
            density_exact=False,
            ci_cache={0.05: (1.4, 1.7)},
        )
        model._summary_cache = None

        summary = model.summary()

        assert summary._info["tweedie_p_method"] == ("Profile MLE (Brent; density approximation)")

    def test_summary_tolerates_legacy_profile_without_reporting_attributes(self, fitted_poisson):
        model, X, y, weights = fitted_poisson
        model._tweedie_profile_result = SimpleNamespace(
            p_hat=1.55,
            phi_hat=0.8,
            nll=11.0,
        )
        model._summary_cache = None

        model_summary = model.summary()
        metrics_summary = model.metrics(X, y, sample_weight=weights).summary()

        for summary in (model_summary, metrics_summary):
            assert summary._info["tweedie_p_ci"] is None
            assert summary._info["tweedie_p_ci_status"] == "not computed"


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
            selection_penalty=1e6,
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
            selection_penalty=10.0,
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
            selection_penalty=0.0,
            features={"x": Polynomial(degree=2), "cat": Categorical()},
            interactions=[("x", "cat")],
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        text = str(m.summary())
        assert "x:cat" in text
        html = m.summary()._repr_html_()
        assert "x:cat" in html


class TestPolynomialSummary:
    """Plain polynomial terms should be rendered as degree components."""

    def test_polynomial_summary_split_by_degree(self):
        from superglm.features.polynomial import Polynomial

        rng = np.random.default_rng(123)
        n = 400
        age = rng.uniform(18, 90, n)
        sample_weight = rng.uniform(0.3, 1.0, n)
        age_s = (age - 50.0) / 20.0
        mu = np.exp(-1.8 + 0.35 * age_s - 0.25 * age_s**2 + 0.08 * age_s**3)
        y = rng.poisson(mu * sample_weight).astype(float)
        X = pd.DataFrame({"age": age})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.001,
            features={"age": Polynomial(degree=3)},
        )
        model.fit(X, y, sample_weight=sample_weight)

        text = str(model.summary())
        html = model.summary()._repr_html_()

        assert "age P(3)" in text
        assert "age[P1]" in text
        assert "age[P2]" in text
        assert "age[P3]" in text

        assert "age P(3)" in html
        assert "age[P1]" in html
        assert "age[P2]" in html
        assert "age[P3]" in html


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
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        # Patch effective_df to force the denom <= 0 branch
        original_result = m._result
        m._result = replace(m._result, effective_df=float(n))
        assert m.aicc == np.inf
        m._result = original_result


class TestSummaryHelpers:
    """Edge cases in summary helper functions."""

    def test_summary_helpers_edge_cases(self):
        from superglm.inference.summary import _compute_coef_stats, _sig_stars

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
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        m = model.metrics(X, y)
        fse = m.feature_se("x")
        assert "se_coef" in fse
        assert fse["se_coef"] > 0


# ── model.summary() / model.diagnostics() API ──────────────────


class TestModelSummaryAPI:
    """model.summary() returns ModelSummary; model.diagnostics() returns dict."""

    @pytest.fixture
    def fitted(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.1 * x)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y)
        return model, X, y

    def test_summary_returns_model_summary(self, fitted):
        from superglm.inference.summary import ModelSummary

        model, _, _ = fitted
        s = model.summary()
        assert isinstance(s, ModelSummary)

    def test_summary_str(self, fitted):
        model, _, _ = fitted
        text = str(model.summary())
        assert "SuperGLM Results" in text
        assert "Poisson" in text
        assert "Intercept" in text

    def test_summary_repr_html(self, fitted):
        model, _, _ = fitted
        html = model.summary()._repr_html_()
        assert "<table" in html
        assert "SuperGLM Results" in html

    def test_print_summary(self, fitted, capsys):
        model, _, _ = fitted
        print(model.summary())
        out = capsys.readouterr().out
        assert "SuperGLM Results" in out

    def test_diagnostics_returns_dict(self, fitted):
        model, _, _ = fitted
        d = model.diagnostics()
        assert isinstance(d, dict)
        assert "_model" in d
        assert "x" in d

    def test_diagnostics_shape(self, fitted):
        model, _, _ = fitted
        d = model.diagnostics()
        assert "active" in d["x"]
        assert "group_norm" in d["x"]
        assert "n_params" in d["x"]

    def test_diagnostics_spline_metadata(self, fitted):
        model, _, _ = fitted
        d = model.diagnostics()
        assert "edf" in d["x"]
        assert "smoothing_lambda" in d["x"]
        assert "spline_kind" in d["x"]
        assert d["x"]["spline_kind"] == "PSpline"

    def test_summary_spline_edf_in_text(self, fitted):
        model, _, _ = fitted
        text = str(model.summary())
        assert "edf=" in text

    def test_summary_matches_metrics_summary(self, fitted):
        """model.summary() and model.metrics(X,y).summary() agree."""
        model, X, y = fitted
        s1 = model.summary()
        s2 = model.metrics(X, y).summary()
        # Both should have the same coefficient rows
        assert len(s1._coef_rows) == len(s2._coef_rows)
        for r1, r2 in zip(s1._coef_rows, s2._coef_rows):
            assert r1.name == r2.name
            if r1.coef is not None:
                assert abs(r1.coef - r2.coef) < 1e-10

    def test_summary_before_fit_raises(self):
        model = SuperGLM(family="poisson", features={"x": Numeric()})
        with pytest.raises(RuntimeError, match="No fit stats"):
            model.summary()

    def test_summary_immune_to_caller_mutation(self, fitted):
        """Mutating caller's y/sample_weight after fit must not change summary."""
        model, X, y = fitted
        ll_before = model.summary()["information_criteria"]["log_likelihood"]

        # Mutate the original arrays the caller passed to fit()
        y[:] = 999.0

        ll_after = model.summary()["information_criteria"]["log_likelihood"]
        assert ll_before == ll_after

    def test_no_train_y_or_train_mu_after_fit(self, fitted):
        """After refactor, model should not have _train_y or _train_mu."""
        model, _, _ = fitted
        assert not hasattr(model, "_train_y")
        assert not hasattr(model, "_train_mu")

    def test_summary_after_fit_reml(self):
        """model.summary() works after fit_reml()."""
        from superglm.inference.summary import ModelSummary

        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.1 * x)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit_reml(X, y)
        s = model.summary()
        assert isinstance(s, ModelSummary)
        assert "SuperGLM Results" in str(s)

    def test_summary_has_fit_stats(self, fitted):
        """model._fit_stats is populated after fit()."""
        from superglm.types import FitStats

        model, _, _ = fitted
        assert model._fit_stats is not None
        assert isinstance(model._fit_stats, FitStats)
        assert model._fit_stats.n_obs > 0
        assert np.isfinite(model._fit_stats.log_likelihood)
        assert np.isfinite(model._fit_stats.null_deviance)
        assert 0 <= model._fit_stats.explained_deviance <= 1

    def test_summary_standard_errors_key(self, fitted):
        """model.summary() includes standard_errors for backward compat."""
        model, _, _ = fitted
        s = model.summary()
        assert "standard_errors" in s
        assert "coefficient_se" in s["standard_errors"]
        assert "coefficient_se_raw" in s["standard_errors"]


class TestSummaryInferenceCache:
    """Tests for the shared fit-inference cache used by summary()."""

    def test_second_summary_does_not_recompute_gram(self, fitted_poisson):
        """Second summary() call should reuse cached inference info."""
        model, X, y, _ = fitted_poisson
        # First call populates caches
        s1 = model.summary()

        # Monkeypatch the gram path to raise if called again
        import unittest.mock

        with unittest.mock.patch(
            "superglm.model.state_ops.fit_inference_info",
            side_effect=AssertionError("inference recomputed"),
        ):
            s2 = model.summary()

        # Both summaries should have identical coef rows
        for r1, r2 in zip(s1._coef_rows, s2._coef_rows):
            assert r1.name == r2.name
            if r1.se is not None:
                assert r1.se == pytest.approx(r2.se)

    def test_group_edf_matches_inference_cache(self, fitted_poisson):
        """_group_edf should return the same values as _fit_inference_info."""
        model, _, _, _ = fitted_poisson
        gedf = model._group_edf
        inf = model._fit_inference_info
        assert gedf == inf["group_edf_map"]

    def test_cache_invalidated_after_refit(self, fitted_poisson):
        """Refitting should clear cached inference info."""
        model, X, y, _ = fitted_poisson
        _ = model._fit_inference_info  # populate cache
        assert "_fit_inference_info" in model.__dict__

        # Refit with different penalty
        model.fit(X, y, sample_weight=np.ones(len(y)))
        assert "_fit_inference_info" not in model.__dict__

    def test_summary_rank_deficient_active_system(self):
        """summary() must not crash when active columns are aliased."""
        rng = np.random.default_rng(77)
        n = 300
        x1 = rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1, "x2": x1})  # exact duplicate
        y = rng.poisson(np.exp(0.5 + 0.3 * x1)).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(X, y)
        # Must not raise LinAlgError
        s = model.summary()
        assert len(s._coef_rows) > 0

        # EDF should sum to ~1 (one effective parameter, aliased)
        coef_edfs = [r.edf for r in s._coef_rows if r.edf is not None and r.name != "Intercept"]
        total_edf = sum(coef_edfs)
        assert total_edf < 1.5, f"Aliased columns should share ~1 EDF, got {total_edf}"

    def test_summary_rank_deficient_smooth_term(self):
        """Pseudo-R fallback must produce valid Wood test results for smooth terms.

        Regression test for the path: singular XtWX → eigendecomposition pseudo-R
        (state_ops.py) → wood_test_smooth (metrics.py). Two identical splines on
        the same data force XtWX to be rank-deficient in the active system.
        """
        rng = np.random.default_rng(99)
        n = 400
        x = rng.uniform(0, 10, n)
        X = pd.DataFrame({"x": x, "x_dup": x})
        y = rng.poisson(np.exp(0.5 + 0.3 * np.sin(x))).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=8), "x_dup": Spline(n_knots=8)},
        )
        model.fit(X, y)

        # Must not raise LinAlgError
        s = model.summary()
        smooth_rows = [r for r in s._coef_rows if r.is_spline and r.active]
        assert len(smooth_rows) >= 1

        # Wood test p-values should be finite (not NaN/Inf)
        for r in smooth_rows:
            assert r.wald_p is not None, f"{r.name}: p-value is None"
            assert np.isfinite(r.wald_p), f"{r.name}: p-value is {r.wald_p}"
            assert 0.0 <= r.wald_p <= 1.0, f"{r.name}: p-value out of range: {r.wald_p}"

    def test_summary_near_aliased_edf_matches_metrics(self):
        """Near-aliased EDF from model.summary() should match model.metrics()."""
        rng = np.random.default_rng(88)
        n = 500
        x1 = rng.standard_normal(n)
        x2 = x1 + 1e-6 * rng.standard_normal(n)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        y = rng.poisson(np.exp(0.5 + 0.3 * x1)).astype(float)

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x1": Numeric(), "x2": Numeric()},
        )
        model.fit(X, y)

        s = model.summary()
        met = model.metrics(X, y)
        met_rows = met._build_coef_rows()

        for sr, mr in zip(s._coef_rows, met_rows):
            if sr.edf is not None and mr.edf is not None:
                np.testing.assert_allclose(
                    sr.edf,
                    mr.edf,
                    atol=0.05,
                    err_msg=f"EDF mismatch for {sr.name}",
                )

    def test_summary_sparse_categorical_se_matches_metrics(self):
        """Rare weakly-identified factor levels must keep large summary SEs."""
        rng = np.random.default_rng(0)
        levels = list("ABCDEFGHIJ")
        n = 8000
        x = rng.choice(
            levels,
            size=n,
            p=[0.22, 0.18, 0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.01],
        )
        w = np.ones(n)
        rare_mask = x == "J"
        w[rare_mask] = 1e-4

        coef = {
            lev: c for lev, c in zip(levels, [0.0, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.2, 0.5, -2.5])
        }
        eta = 1.5 + np.array([coef[v] for v in x])
        mu = np.exp(eta)
        counts = rng.poisson(mu * w).astype(float)
        counts[rare_mask] = 0.0
        y = counts / np.maximum(w, 1e-300)
        X = pd.DataFrame({"cat": pd.Categorical(x, categories=levels)})

        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"cat": Categorical(base="first")},
        )
        model.fit(X, y, sample_weight=w)

        s_model = model.summary()
        s_metrics = model.metrics(X, y, sample_weight=w).summary()

        row_model = next(r for r in s_model._coef_rows if r.name == "cat[J]")
        row_metrics = next(r for r in s_metrics._coef_rows if r.name == "cat[J]")

        assert row_metrics.se > 1.0
        assert np.isfinite(row_model.se)
        assert row_model.se == pytest.approx(row_metrics.se, rel=1e-6, abs=1e-8)
        assert row_model.p == pytest.approx(row_metrics.p, rel=1e-6, abs=1e-8)

    def test_metrics_summary_uses_own_edf_not_fit_cache(self):
        """ModelMetrics.summary() must compute EDF from its own weights, not fit cache."""
        rng = np.random.default_rng(123)
        n = 400
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(0.5 + 0.3 * np.sin(x))).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        w_fit = np.ones(n)
        model.fit_reml(X, y, sample_weight=w_fit)

        # Different weights — EDF should differ from fit-time values
        w_alt = rng.uniform(0.5, 2.0, n)
        met = model.metrics(X, y, sample_weight=w_alt)

        # Get the spline EDF from metrics' own influence diagonal
        edf_met, _ = met._influence_edf
        _, _, _, _, active_groups_met = met._active_info
        ag = active_groups_met[0]
        met_edf = float(np.sum(edf_met[ag.sl]))

        # Get the spline EDF from the summary coef row
        coef_rows = met._build_coef_rows()
        spline_row = next(r for r in coef_rows if r.is_spline)
        summary_edf = spline_row.edf

        # Summary EDF must match metrics' own EDF, not fit-time _group_edf
        np.testing.assert_allclose(
            summary_edf, met_edf, rtol=1e-10, err_msg="ModelMetrics.summary() used fit-time EDF"
        )


# ── Basis Detail ─────────────────────────────────────────────────


class TestBasisDetail:
    """Tests for the detail='basis' spline coefficient disclosure."""

    @pytest.fixture
    def spline_model(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.1 * x)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x": Spline(n_knots=8, penalty="ssp")},
        )
        model.fit(X, y)
        return model, X, y

    @pytest.fixture
    def numeric_model(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.standard_normal(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            features={"x": Numeric()},
        )
        model.fit(X, y)
        return model, X, y

    def test_default_summary_no_coef_detail_in_ascii(self, spline_model):
        model, _, _ = spline_model
        s = model.summary()
        text = str(s)
        # ASCII compact: no inline coef rows
        assert "Coef 1" not in text
        # But basis_detail is computed (for HTML closed disclosures)
        assert len(s._basis_detail) > 0

    def test_basis_detail_ascii(self, spline_model):
        model, _, _ = spline_model
        text = str(model.summary(detail="full"))
        assert "Coef 1" in text
        assert "Coef 2" in text

    def test_basis_detail_row_count(self, spline_model):
        model, _, _ = spline_model
        s = model.summary(detail="full")
        # Find the spline group
        spline_groups = [g for g in model._groups if g.feature_name == "x"]
        for g in spline_groups:
            if g.name in s._basis_detail:
                assert len(s._basis_detail[g.name]) == g.size

    def test_basis_detail_coef_stats_finite(self, spline_model):
        model, _, _ = spline_model
        s = model.summary(detail="full")
        for rows in s._basis_detail.values():
            for br in rows:
                assert np.isfinite(br.coef)
                assert br.se > 0
                assert np.isfinite(br.z)
                assert np.isfinite(br.p)
                assert np.isfinite(br.ci_low)
                assert np.isfinite(br.ci_high)

    def test_html_disclosure_closed_compact(self, spline_model):
        model, _, _ = spline_model
        html = model.summary(detail="compact")._repr_html_()
        # Compact: closed disclosure present
        assert "<details>" in html
        assert "<details open" not in html

    def test_html_disclosure_open_for_full(self, spline_model):
        model, _, _ = spline_model
        html = model.summary(detail="full")._repr_html_()
        assert "<details open>" in html

    def test_non_spline_no_disclosure(self, numeric_model):
        model, _, _ = numeric_model
        html = model.summary(detail="compact")._repr_html_()
        assert "<details>" not in html

    def test_inactive_spline_no_basis_detail(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.standard_normal(n)
        y = rng.poisson(1.0, n).astype(float)  # x has no effect
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            selection_penalty=1000.0,  # very high penalty to zero out the spline
            features={"x": Spline(n_knots=5, penalty="ssp")},
        )
        model.fit(X, y)
        s = model.summary(detail="full")
        # inactive splines should not have basis detail
        for g in model._groups:
            if g.feature_name == "x":
                assert g.name not in s._basis_detail

    def test_invalid_detail_raises(self, spline_model):
        model, _, _ = spline_model
        with pytest.raises(ValueError, match="detail="):
            model.summary(detail="bogus")

    def test_select_true_both_subgroups(self):
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.1 * x)
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})
        model = SuperGLM(
            family="poisson",
            features={"x": Spline(n_knots=8, penalty="ssp", select=True)},
        )
        model.fit_reml(X, y)
        s = model.summary(detail="full")
        # Single group with multi-penalty components (null + wiggle)
        subgroup_names = [g.name for g in model._groups if g.feature_name == "x"]
        assert len(subgroup_names) == 1  # single "x" group
        for name in subgroup_names:
            # Active subgroups should have basis detail
            g = next(g for g in model._groups if g.name == name)
            if np.linalg.norm(model.result.beta[g.sl]) > 1e-12:
                assert name in s._basis_detail
                assert len(s._basis_detail[name]) == g.size

    def test_model_and_metrics_summary_agree(self, spline_model):
        model, X, y = spline_model
        s1 = model.summary(detail="full")
        s2 = model.metrics(X, y).summary(detail="full")
        assert set(s1._basis_detail.keys()) == set(s2._basis_detail.keys())
        for key in s1._basis_detail:
            rows1 = s1._basis_detail[key]
            rows2 = s2._basis_detail[key]
            assert len(rows1) == len(rows2)
            for r1, r2 in zip(rows1, rows2):
                np.testing.assert_allclose(r1.coef, r2.coef, rtol=1e-10)
                np.testing.assert_allclose(r1.se, r2.se, rtol=1e-10)

    def test_basis_se_matches_main_summary_se(self, spline_model):
        """Basis SEs use the same known_scale-aware path as the main summary."""
        model, _, _ = spline_model
        s = model.summary(detail="full")
        # Get per-group SEs from the backward-compat dict
        se_dict = s["standard_errors"]["coefficient_se"]
        for g_name, basis_rows in s._basis_detail.items():
            main_se = se_dict[g_name]
            for br in basis_rows:
                np.testing.assert_allclose(
                    br.se,
                    main_se[br.basis_index],
                    rtol=1e-10,
                    err_msg=f"Basis SE mismatch for {g_name}[{br.basis_index}]",
                )
