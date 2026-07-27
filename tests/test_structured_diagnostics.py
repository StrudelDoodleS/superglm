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
from superglm._frame import as_eager_frame
from superglm.distributions import clip_mu
from superglm.links import stabilize_eta


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


def _dense_holdout_drop_reference(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    offset: np.ndarray,
) -> pd.DataFrame:
    frame = as_eager_frame(X)
    blocks = [
        model._specs[name].transform(frame.column_array(name)) for name in model._feature_order
    ]
    blocks.extend(
        model._interaction_specs[name].transform(
            frame.column_array(model._interaction_specs[name].parent_names[0]),
            frame.column_array(model._interaction_specs[name].parent_names[1]),
        )
        for name in model._interaction_order
    )
    design = np.hstack(blocks)
    beta = model.result.beta
    eta_full = stabilize_eta(
        design @ beta + model.result.intercept + offset,
        model._link,
    )
    mu_full = clip_mu(model._link.inverse(eta_full), model._distribution)
    dev_full = float(np.sum(weights * model._distribution.deviance_unit(y, mu_full)))

    rows = []
    seen_features = set()
    for group in model._groups:
        if group.feature_name in seen_features:
            continue
        seen_features.add(group.feature_name)
        beta_drop = beta.copy()
        for feature_group in model._groups:
            if feature_group.feature_name == group.feature_name:
                beta_drop[feature_group.sl] = 0.0
        eta_drop = stabilize_eta(
            design @ beta_drop + model.result.intercept + offset,
            model._link,
        )
        mu_drop = clip_mu(model._link.inverse(eta_drop), model._distribution)
        dev_drop = float(np.sum(weights * model._distribution.deviance_unit(y, mu_drop)))
        rows.append(
            {
                "feature": group.feature_name,
                "delta_deviance": dev_drop - dev_full,
            }
        )
    return pd.DataFrame(rows)


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


def test_holdout_drop_uses_compact_structured_score(
    structured_diagnostic_case,
    monkeypatch,
):
    model, X, y = structured_diagnostic_case
    weights = np.linspace(0.7, 1.3, len(y))
    offset = np.linspace(-0.15, 0.2, len(y))
    expected = _dense_holdout_drop_reference(model, X, y, weights, offset)
    spec = _structured_spec(model)
    original_score = spec.score
    scored_betas = []

    def fail_transform(*args, **kwargs):
        del args, kwargs
        raise AssertionError("structured transform must not be called")

    def record_score(*args, **kwargs):
        beta = np.asarray(args[-1], dtype=np.float64)
        scored_betas.append(beta.copy())
        return original_score(*args, **kwargs)

    monkeypatch.setattr(spec, "transform", fail_transform)
    monkeypatch.setattr(spec, "score", record_score)
    actual = model.term_drop_diagnostics(
        X,
        y,
        mode="holdout",
        X_val=X.copy(),
        y_val=y.copy(),
        sample_weight_val=weights,
        offset_val=offset,
    )

    pd.testing.assert_frame_equal(actual, expected, rtol=2e-11, atol=2e-11)
    from superglm.model import base

    plan = base._prediction_plan(model)
    term = next(term for term in (*plan["features"], *plan["interactions"]) if term["spec"] is spec)
    assert len(scored_betas) == 1
    np.testing.assert_array_equal(
        scored_betas[0],
        model.result.beta[term["beta_idx"]],
    )
