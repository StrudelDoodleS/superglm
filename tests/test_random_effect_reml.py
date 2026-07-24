"""Dense-oracle and structured REML tests for random effects."""

import numpy as np
import pandas as pd

from superglm import LambdaPolicy, RandomEffect, SuperGLM


def test_gaussian_fixed_lambda_matches_analytic_penalized_least_squares():
    levels = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    y = np.array([1.0, 1.5, 2.0, 2.5, -0.5, 0.0])
    weights = np.array([1.0, 2.0, 1.5, 0.5, 1.0, 3.0])
    lam = 2.0
    X = pd.DataFrame({"broker": levels})
    model = SuperGLM(
        family="gaussian",
        features={"broker": RandomEffect(lambda_policy=LambdaPolicy.fixed(lam))},
        selection_penalty=0,
        direct_solve="gram",
    )

    model.fit_reml(X, y, sample_weight=weights, max_reml_iter=2)

    codes = pd.Categorical(levels, categories=["a", "b", "c"]).codes
    Z = np.eye(3)[codes]
    design = np.column_stack([np.ones(len(y)), Z])
    penalty = np.diag([0.0, lam, lam, lam])
    expected = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty, design.T @ (weights * y)
    )

    assert model._reml_lambdas == {"broker": lam}
    np.testing.assert_allclose(model.result.intercept, expected[0], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(model.result.beta, expected[1:], rtol=1e-8, atol=1e-8)


def test_gaussian_estimates_finite_random_effect_lambda():
    rng = np.random.default_rng(20260724)
    n_levels = 12
    repeats = 20
    codes = np.repeat(np.arange(n_levels), repeats)
    level_effects = rng.normal(scale=0.7, size=n_levels)
    y = 1.5 + level_effects[codes] + rng.normal(scale=0.4, size=len(codes))
    X = pd.DataFrame({"broker": np.array([f"b{i}" for i in codes], dtype=object)})
    model = SuperGLM(
        family="gaussian",
        features={"broker": RandomEffect()},
        selection_penalty=0,
        direct_solve="gram",
    )

    model.fit_reml(X, y, max_reml_iter=15)

    lam = model._reml_lambdas["broker"]
    assert np.isfinite(lam)
    assert 1e-6 <= lam <= 1e10
    assert np.isfinite(model._reml_result.objective)
    assert np.isfinite(model.result.effective_df)
    assert isinstance(model._reml_result.converged, bool)


def test_poisson_fixed_lambda_with_offset_and_weights_matches_newton_reference():
    levels = np.array(["a", "a", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([1.0, 3.0, 0.0, 2.0, 4.0, 1.0, 3.0])
    exposure = np.array([1.0, 2.0, 0.8, 1.5, 2.5, 1.0, 1.8])
    weights = np.array([1.0, 1.5, 2.0, 0.75, 1.0, 2.0, 0.5])
    offset = np.log(exposure)
    lam = 1.75
    X = pd.DataFrame({"broker": levels})
    model = SuperGLM(
        family="poisson",
        features={"broker": RandomEffect(lambda_policy=LambdaPolicy.fixed(lam))},
        selection_penalty=0,
        direct_solve="gram",
    )

    model.fit_reml(X, y, sample_weight=weights, offset=offset, max_reml_iter=2)

    codes = pd.Categorical(levels, categories=["a", "b", "c"]).codes
    Z = np.eye(3)[codes]
    design = np.column_stack([np.ones(len(y)), Z])
    theta = np.zeros(4)
    theta[0] = np.log(np.sum(weights * y) / np.sum(weights * exposure))
    penalty = np.diag([0.0, lam, lam, lam])
    for _ in range(30):
        eta = design @ theta + offset
        mu = np.exp(eta)
        gradient = design.T @ (weights * (mu - y)) + penalty @ theta
        hessian = design.T @ ((weights * mu)[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if np.max(np.abs(step)) < 1e-12:
            break

    np.testing.assert_allclose(model.result.intercept, theta[0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(model.result.beta, theta[1:], rtol=1e-7, atol=1e-7)


def test_poisson_estimates_finite_random_effect_lambda_with_exposure_offset():
    rng = np.random.default_rng(8172)
    n_levels = 10
    repeats = 30
    codes = np.repeat(np.arange(n_levels), repeats)
    effects = rng.normal(scale=0.45, size=n_levels)
    exposure = rng.uniform(0.4, 2.0, size=len(codes))
    mean = exposure * np.exp(-0.3 + effects[codes])
    y = rng.poisson(mean).astype(float)
    X = pd.DataFrame({"broker": np.array([f"b{i}" for i in codes], dtype=object)})
    model = SuperGLM(
        family="poisson",
        features={"broker": RandomEffect()},
        selection_penalty=0,
        direct_solve="gram",
    )

    model.fit_reml(X, y, offset=np.log(exposure), max_reml_iter=15)

    lam = model._reml_lambdas["broker"]
    assert np.isfinite(lam)
    assert 1e-6 <= lam <= 1e10
    assert np.isfinite(model._reml_result.objective)
    assert np.all(np.isfinite(model.predict(X, offset=np.log(exposure))))


def test_poisson_structured_reml_matches_dense_lambda_trajectory():
    rng = np.random.default_rng(7814)
    n_levels = 18
    repeats = 18
    codes = np.repeat(np.arange(n_levels), repeats)
    effects = rng.normal(scale=0.4, size=n_levels)
    exposure = rng.uniform(0.5, 2.2, size=len(codes))
    y = rng.poisson(exposure * np.exp(-0.2 + effects[codes])).astype(float)
    X = pd.DataFrame({"broker": np.array([f"b{i}" for i in codes], dtype=object)})
    common = {
        "family": "poisson",
        "features": {"broker": RandomEffect()},
        "selection_penalty": 0,
    }

    dense = SuperGLM(**common, direct_solve="gram")
    structured = SuperGLM(**common, direct_solve="structured")
    dense.fit_reml(X, y, offset=np.log(exposure), max_reml_iter=8)
    structured.fit_reml(X, y, offset=np.log(exposure), max_reml_iter=8)

    np.testing.assert_allclose(structured.result.beta, dense.result.beta, atol=2e-8)
    np.testing.assert_allclose(structured.result.intercept, dense.result.intercept, atol=2e-8)
    np.testing.assert_allclose(
        structured._reml_lambdas["broker"],
        dense._reml_lambdas["broker"],
        rtol=2e-7,
    )
    assert len(structured._reml_result.lambda_history) == len(dense._reml_result.lambda_history)
    for structured_lambdas, dense_lambdas in zip(
        structured._reml_result.lambda_history,
        dense._reml_result.lambda_history,
        strict=True,
    ):
        np.testing.assert_allclose(
            structured_lambdas["broker"],
            dense_lambdas["broker"],
            rtol=2e-7,
        )


def test_gamma_structured_observed_reml_matches_dense():
    rng = np.random.default_rng(993)
    n_levels = 14
    repeats = 20
    codes = np.repeat(np.arange(n_levels), repeats)
    effects = rng.normal(scale=0.28, size=n_levels)
    mean = np.exp(0.35 + effects[codes])
    y = rng.gamma(shape=4.0, scale=mean / 4.0)
    X = pd.DataFrame({"broker": np.array([f"b{i}" for i in codes], dtype=object)})
    common = {
        "family": "gamma",
        "features": {"broker": RandomEffect()},
        "selection_penalty": 0,
    }

    dense = SuperGLM(**common, direct_solve="gram")
    structured = SuperGLM(**common, direct_solve="structured")
    dense.fit_reml(X, y, max_reml_iter=6)
    structured.fit_reml(X, y, max_reml_iter=6)

    np.testing.assert_allclose(structured.result.beta, dense.result.beta, atol=3e-8)
    np.testing.assert_allclose(structured.result.intercept, dense.result.intercept, atol=3e-8)
    np.testing.assert_allclose(
        structured._reml_lambdas["broker"],
        dense._reml_lambdas["broker"],
        rtol=3e-7,
    )
    np.testing.assert_allclose(
        structured._reml_result.objective,
        dense._reml_result.objective,
        atol=2e-8,
    )
