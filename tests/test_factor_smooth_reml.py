"""Small dense REML oracles for fully penalized factor smooths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, LambdaPolicy, SuperGLM
from superglm.reml.penalty_algebra import build_penalty_matrix


def _gaussian_data() -> tuple[pd.DataFrame, np.ndarray]:
    x_level = np.linspace(-1.0, 1.0, 24)
    levels = np.array(["a", "b", "c"], dtype=object)
    x = np.tile(x_level, len(levels))
    group = np.repeat(levels, len(x_level))
    effects = {
        "a": 0.8 * np.sin(2.2 * x_level) + 0.2 * x_level,
        "b": -0.4 * np.cos(1.7 * x_level) - 0.3 * x_level,
        "c": 0.25 + 0.55 * x_level**2,
    }
    y = 1.1 + np.concatenate([effects[level] for level in levels])
    return pd.DataFrame({"x": x, "group": group}), y


def test_fixed_lambda_gaussian_fit_matches_explicit_penalized_least_squares() -> None:
    X, y = _gaussian_data()
    policies = {
        "wiggle": LambdaPolicy.fixed(1.4),
        "null_0": LambdaPolicy.fixed(0.7),
        "null_1": LambdaPolicy.fixed(2.1),
    }
    model = SuperGLM(
        family="gaussian",
        interactions=[FactorSmooth("x", group="group", k=6, lambda_policy=policies)],
        selection_penalty=0.0,
        direct_solve="gram",
    )

    model.fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")

    design = model._dm.toarray()
    augmented = np.column_stack([np.ones(len(X)), design])
    penalty = build_penalty_matrix(
        list(model._dm.group_matrices),
        model._groups,
        model._reml_result.lambdas,
        model._dm.p,
        reml_penalties=model._reml_penalties,
    )
    augmented_penalty = np.zeros((model._dm.p + 1, model._dm.p + 1))
    augmented_penalty[1:, 1:] = penalty
    expected = np.linalg.solve(
        augmented.T @ augmented + augmented_penalty,
        augmented.T @ y,
    )

    assert model.result.intercept == pytest.approx(expected[0], abs=2e-9)
    np.testing.assert_allclose(model.result.beta, expected[1:], rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_estimated_lambda_dense_factor_smooth_fit_is_finite(family: str) -> None:
    X, gaussian_y = _gaussian_data()
    if family == "gaussian":
        y = gaussian_y
    else:
        rng = np.random.default_rng(571)
        mean = np.exp(-0.3 + 0.35 * gaussian_y)
        y = rng.poisson(mean).astype(np.float64)
    model = SuperGLM(
        family=family,
        interactions=[FactorSmooth("x", group="group", k=6)],
        selection_penalty=0.0,
        direct_solve="gram",
    )

    model.fit_reml(
        X,
        y,
        max_reml_iter=8,
        reml_tol=1e-5,
        runtime_validation="skip",
    )

    assert np.all(np.isfinite(model.result.beta))
    assert np.isfinite(model.result.intercept)
    assert model._reml_result is not None
    assert set(model._reml_result.lambdas) == {
        "x:group:fs:wiggle",
        "x:group:fs:null_0",
        "x:group:fs:null_1",
    }
    assert all(np.isfinite(value) and value > 0.0 for value in model._reml_result.lambdas.values())
