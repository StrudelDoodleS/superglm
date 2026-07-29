"""Tests for refit_unpenalised()."""

import numpy as np
import pandas as pd
import pytest

from superglm import LambdaPolicy, RandomEffect, SuperGLM
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
    sample_weight = np.ones(n)
    y = rng.poisson(mu * sample_weight).astype(float)
    X = pd.DataFrame({"strong": x_strong, "noise": x_noise})
    return X, y, sample_weight


class TestRefitBasic:
    def test_returns_new_model(self, selection_data):
        X, y, sample_weight = selection_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.5,  # high penalty to zero noise
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)

        refitted = model.refit_unpenalised(X, y, sample_weight=sample_weight)
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
            selection_penalty=50.0,
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
        X, y, sample_weight = selection_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.5,
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        refitted = model.refit_unpenalised(X, y, sample_weight=sample_weight)

        assert refitted.penalty.lambda1 == 0.0

    def test_refitted_coefficients_differ(self, selection_data):
        X, y, sample_weight = selection_data
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.1,  # moderate penalty — keeps strong but shrinks
            features={"strong": Numeric(), "noise": Numeric()},
        )
        model.fit(X, y, sample_weight=sample_weight)
        refitted = model.refit_unpenalised(X, y, sample_weight=sample_weight)

        # Refitted coefficients should generally differ (less shrinkage)
        if "strong" in refitted._specs:
            orig_groups = model._feature_groups("strong")
            refit_groups = refitted._feature_groups("strong")
            orig_beta = model.result.beta[orig_groups[0].sl]
            refit_beta = refitted.result.beta[refit_groups[0].sl]
            # Not identical — shrinkage removed
            assert not np.allclose(orig_beta, refit_beta, atol=1e-6)

    def test_preserves_nonuniform_sample_weight(self):
        x = np.linspace(-2.0, 2.0, 160)
        y = 0.6 + 0.9 * x
        y[x > 0.8] += 2.5
        sample_weight = np.where(x > 0.8, 0.05, 3.0)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="gaussian",
            features={"x": Numeric()},
            selection_penalty=0.0,
        ).fit(X, y, sample_weight=sample_weight)

        weighted = model.refit_unpenalised(X, y, sample_weight=sample_weight)
        unweighted = model.refit_unpenalised(X, y)

        np.testing.assert_allclose(
            weighted.result.beta,
            model.result.beta,
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            weighted.result.intercept,
            model.result.intercept,
            rtol=1e-9,
            atol=1e-9,
        )
        assert not np.allclose(
            weighted.result.beta,
            unweighted.result.beta,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_unfitted_raises(self):
        model = SuperGLM(features={"x": Numeric()})
        X = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.refit_unpenalised(X, np.array([1, 2, 3]))

    def test_keep_smoothing_false(self):
        """keep_smoothing=False should set spline_penalty=0."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        mu = np.exp(0.5 + 0.3 * np.sin(x))
        y = rng.poisson(mu).astype(float)
        X = pd.DataFrame({"x": x})

        model = SuperGLM(
            family="poisson",
            selection_penalty=0.01,
            spline_penalty=0.5,
            features={"x": Spline(n_knots=10, penalty="ssp")},
        )
        model.fit(X, y)
        refitted = model.refit_unpenalised(X, y, keep_smoothing=False)

        assert refitted.lambda2 == 0.0


def test_refit_unpenalised_rejects_variance_component_terms_at_entry(monkeypatch):
    rng = np.random.default_rng(20260727)
    codes = np.repeat(np.arange(6), 12)
    X = pd.DataFrame({"group": np.array([f"group-{code}" for code in codes], dtype=object)})
    y = rng.normal(size=len(codes))
    model = SuperGLM(
        family="gaussian",
        features={
            "group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0)),
        },
        selection_penalty=0.0,
        direct_solve="structured",
    ).fit_reml(X, y, runtime_validation="skip")

    def fail_clone(*args, **kwargs):
        del args, kwargs
        raise AssertionError("model clone requested")

    monkeypatch.setattr(model, "_clone_without_features", fail_clone)
    with pytest.raises(
        NotImplementedError,
        match=r"refit_unpenalised\(\).*variance-component.*group",
    ):
        model.refit_unpenalised(X, y)


class TestKeepSmoothingWithTensorRemlLambdas:
    """RFC-8 bug-half regression: REML-fitted component-named lambdas fed back
    through the public ``lambda2`` setter (directly or via
    ``refit_unpenalised(keep_smoothing=True)``) must actually penalize the
    tensor block in the subsequent ``fit()``.
    """

    @pytest.fixture()
    def tensor_reml_fit(self):
        rng = np.random.default_rng(1234)
        n = 400
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        # Additive truth: REML pushes the tensor lambdas high, so silently
        # dropping the tensor penalty visibly changes predictions.
        eta = 0.4 + np.sin(2 * np.pi * x1) + 0.5 * x2
        y = rng.poisson(np.exp(eta)).astype(float)
        X = pd.DataFrame({"x1": x1, "x2": x2})
        model = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=5),
                "x2": Spline(kind="cr", n_knots=5),
            },
            interactions=[("x1", "x2")],
        )
        model.fit_reml(X, y, max_reml_iter=30)
        return model, X, y

    def test_component_named_setter_fit_reproduces_reml_fit(self, tensor_reml_fit):
        from superglm.model.fit_state import fitted_lambda2

        model, X, y = tensor_reml_fit
        lambdas = dict(fitted_lambda2(model))

        refit = SuperGLM(
            family="poisson",
            features={
                "x1": Spline(kind="cr", n_knots=5),
                "x2": Spline(kind="cr", n_knots=5),
            },
            interactions=[("x1", "x2")],
        )
        refit.lambda2 = lambdas
        refit.fit(X, y)

        np.testing.assert_allclose(refit.predict(X), model.predict(X), rtol=1e-2)

    def test_refit_unpenalised_keep_smoothing_keeps_tensor_penalty(self, tensor_reml_fit):
        model, X, y = tensor_reml_fit
        kept = model.refit_unpenalised(X, y, keep_smoothing=True)
        np.testing.assert_allclose(kept.predict(X), model.predict(X), rtol=1e-2)
