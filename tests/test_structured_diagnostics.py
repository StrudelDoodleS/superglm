"""Compact generic diagnostics for structured credibility terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import (
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
)


def _fit_structured_diagnostic_case(
    basis: str,
) -> tuple[SuperGLM, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260727)
    n_levels = 5
    repeats = 24
    codes = np.repeat(np.arange(n_levels), repeats)
    rng.shuffle(codes)
    x = rng.uniform(-1.0, 1.0, size=len(codes))
    z = rng.normal(size=len(codes))
    labels = np.array([f"level-{code}" for code in codes], dtype=object)

    if basis == "re":
        effects = np.array([-0.45, -0.1, 0.2, 0.35, 0.0])
        y = 0.4 + 0.2 * z + effects[codes] + rng.normal(scale=0.12, size=len(codes))
        X = pd.DataFrame({"z": z, "group": labels})
        model = SuperGLM(
            family="gaussian",
            features={
                "z": Numeric(),
                "group": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.3)),
            },
            selection_penalty=0.0,
            direct_solve="structured",
        )
    else:
        amplitudes = np.array([0.65, -0.4, 0.25, -0.55, 0.05])
        if basis == "sz":
            amplitudes -= amplitudes.mean()
        y = (
            0.4
            + 0.2 * z
            + amplitudes[codes] * (x + 0.25 * x**2)
            + rng.normal(scale=0.12, size=len(codes))
        )
        X = pd.DataFrame({"x": x, "z": z, "group": labels})
        policies = {"wiggle": LambdaPolicy.fixed(1.3)}
        if basis == "fs":
            policies |= {
                "null_0": LambdaPolicy.fixed(0.8),
                "null_1": LambdaPolicy.fixed(1.1),
            }
        features = {"z": Numeric()}
        if basis == "sz":
            features["x"] = Spline(
                n_knots=5,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        model = SuperGLM(
            family="gaussian",
            features=features,
            interactions=[
                FactorSmooth(
                    "x",
                    group="group",
                    basis=basis,
                    k=6,
                    lambda_policy=policies,
                )
            ],
            selection_penalty=0.0,
            direct_solve="structured",
        )

    model.fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")
    return model, X, y


@pytest.fixture(scope="module", params=["re", "fs", "sz"])
def structured_diagnostic_case(request):
    return _fit_structured_diagnostic_case(request.param)


def _structured_spec(model):
    if "group" in model._specs:
        return model._specs["group"]
    return next(
        spec for spec in model._interaction_specs.values() if isinstance(spec, FactorSmooth)
    )


def test_term_importance_uses_compact_structured_score(
    structured_diagnostic_case,
    monkeypatch,
):
    model, X, _ = structured_diagnostic_case
    expected = model.term_importance(X)
    spec = _structured_spec(model)

    def fail_transform(*args, **kwargs):
        del args, kwargs
        raise AssertionError("structured transform must not be called")

    monkeypatch.setattr(spec, "transform", fail_transform)
    actual = model.term_importance(X)

    pd.testing.assert_frame_equal(actual, expected, rtol=2e-11, atol=2e-11)
