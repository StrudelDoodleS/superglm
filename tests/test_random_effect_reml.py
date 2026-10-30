"""Dense-oracle and structured REML tests for random effects."""

import numpy as np
import pandas as pd
import pytest

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


def test_a_nested_chain_starts_within_a_newton_step_of_its_optimum(monkeypatch):
    """A random effect's bootstrap fit is penalized at its levels' mean
    information, so the Fellner-Schall step from it places every level of a
    nested chain within one Newton step bound (5 in log lambda) of its REML
    optimum.  Here the intercept-only fit's working weight is the mean
    response on every row, so a level's information is that mean times its
    rows, less its share of the intercept.

    Almost unpenalized, a parent's indicators lie in the span of its
    children's, so along each parent-minus-children direction the curvature
    is the penalties alone and the posterior variance is of order 1/lambda.
    The step's trace term then outweighs the squared coefficients and returns
    about lambda times a constant for parent and child alike: from a 1e-4
    seed the first iterate sits 8.7 to 10 log units below the optimum here
    (on the 1M-row make/model/year chain, which then held the parent
    alternating by the step cap for twenty iterations, 11.6 units).
    Mutation: seeding random effects at 1e-4 again fails both asserts.
    """
    import superglm.reml.direct as direct

    rng = np.random.default_rng(20261030)
    n_parent, children, rows = 8, 6, 40
    child = np.repeat(np.arange(n_parent * children), rows)
    parent = child // children
    eta = (
        1.0 + rng.normal(0.0, 0.5, n_parent)[parent] + rng.normal(0.0, 0.3, child.max() + 1)[child]
    )
    y = rng.poisson(np.exp(eta)).astype(float)
    X = pd.DataFrame(
        {
            "make": np.array([f"m{p}" for p in parent], dtype=object),
            "make_model": np.array([f"m{p}|{c}" for p, c in zip(parent, child)], dtype=object),
        }
    )
    fits: list[tuple[str | None, dict[str, float]]] = []
    real = direct.fit_irls_direct

    def spy(*args, **kwargs):
        fits.append(((kwargs.get("debug_context") or {}).get("phase"), dict(kwargs["lambda2"])))
        return real(*args, **kwargs)

    monkeypatch.setattr(direct, "fit_irls_direct", spy)
    model = SuperGLM(
        family="poisson",
        features={"make": RandomEffect(), "make_model": RandomEffect(nested_in="make")},
        selection_penalty=0,
    )
    model.fit_reml(X, y)

    assert model._reml_result.converged, model._reml_result.termination_reason
    bootstrap = next(lambdas for phase, lambdas in fits if phase == "bootstrap")
    first = next(lambdas for phase, lambdas in fits if phase == "candidate")
    optimum = model._reml_lambdas
    information = {
        name: y.mean() * np.sum(counts * (1.0 - counts / len(y))) / len(counts)
        for name, counts in (("make", np.bincount(parent)), ("make_model", np.bincount(child)))
    }
    assert bootstrap == pytest.approx(information, rel=len(y) * np.finfo(float).eps)
    assert all(abs(np.log(first[name] / optimum[name])) < 5.0 for name in optimum), (first, optimum)


def test_a_random_effect_seed_moves_with_constant_weights(monkeypatch):
    """Gaussian identity with prior weights ``c`` multiplies the REML
    criterion's information and its optimum ``phi / sigma_b**2`` by ``c``, and
    the random effect's bootstrap seed, its levels' mean information, moves
    with them: the seed, every Newton iterate and the estimate are ``c``
    times the unit-weight fit's, in as many iterations.  ``c`` is a power of
    two, so the weights scale exactly and the fits differ by rounding alone,
    at most ``n u`` relative in each of the iterations taken.  Mutation: a
    fixed seed (1e-4 or 1.0) bootstraps every ``c`` alike and fails the first
    assert; with it removed, the iteration counts differ.
    """
    import superglm.reml.direct as direct

    rng = np.random.default_rng(20260724)
    codes = np.repeat(np.arange(12), 20)
    y = 1.5 + rng.normal(scale=0.7, size=12)[codes] + rng.normal(scale=0.4, size=len(codes))
    X = pd.DataFrame({"broker": np.array([f"b{i}" for i in codes], dtype=object)})
    bootstraps: list[float] = []
    real = direct.fit_irls_direct

    def spy(*args, **kwargs):
        if (kwargs.get("debug_context") or {}).get("phase") == "bootstrap":
            bootstraps.append(kwargs["lambda2"]["broker"])
        return real(*args, **kwargs)

    monkeypatch.setattr(direct, "fit_irls_direct", spy)
    fits = {}
    for c in (1.0, 2.0**-8, 2.0**8):
        model = SuperGLM(
            family="gaussian",
            features={"broker": RandomEffect()},
            selection_penalty=0,
            direct_solve="gram",
        )
        model.fit_reml(X, y, sample_weight=np.full(len(y), c))
        assert model._reml_result.converged, model._reml_result.termination_reason
        fits[c] = (bootstraps[-1], model._reml_result.n_reml_iter, model._reml_lambdas["broker"])

    seed, iterations, estimate = fits[1.0]
    rounding = iterations * len(y) * np.finfo(float).eps
    for c, (c_seed, c_iterations, c_estimate) in fits.items():
        assert c_seed == pytest.approx(c * seed, rel=len(y) * np.finfo(float).eps), c
        assert c_iterations == iterations, (c, fits)
        assert c_estimate == pytest.approx(c * estimate, rel=rounding), c
