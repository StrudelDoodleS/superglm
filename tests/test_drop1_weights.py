"""Regression tests for sample-weight propagation through drop1()."""

import numpy as np
import pandas as pd

from superglm import Numeric, SuperGLM
from superglm.inference._term_model_ops import drop1 as internal_drop1
from superglm.model import explain_ops


def _weighted_drop1_data():
    rng = np.random.default_rng(8417)
    n = 180
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    sample_weight = np.where(x > 0.0, 8.0, 0.2)
    eta = -0.4 + 0.65 * x - 0.3 * z
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    X = pd.DataFrame({"x": x, "z": z})
    return X, y, sample_weight


def test_public_drop1_forwards_nonconstant_sample_weight(monkeypatch):
    X, y, sample_weight = _weighted_drop1_data()
    model = SuperGLM(features={"x": Numeric(), "z": Numeric()})
    captured = {}

    def fake_drop1(model_arg, X_arg, y_arg, sample_weight=None, offset=None, *, test="Chisq"):
        captured.update(
            model=model_arg,
            X=X_arg,
            y=y_arg,
            sample_weight=sample_weight,
            offset=offset,
            test=test,
        )
        return pd.DataFrame()

    monkeypatch.setattr(explain_ops, "_drop1", fake_drop1)

    result = model.drop1(X, y, sample_weight=sample_weight)

    assert result.empty
    assert captured["model"] is model
    assert captured["X"] is X
    assert captured["y"] is y
    assert captured["sample_weight"] is sample_weight


def test_drop1_reduced_refits_match_manual_weighted_refits():
    X, y, sample_weight = _weighted_drop1_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    )
    model.fit(X, y, sample_weight=sample_weight)

    result = internal_drop1(model, X, y, sample_weight=sample_weight)

    for feature in model._feature_order:
        reduced = model._clone_without_features({feature})
        reduced.fit(X, y, sample_weight=sample_weight)
        reported = result.loc[result["feature"] == feature, "deviance_reduced"].item()
        assert np.isclose(reported, reduced.result.deviance, rtol=1e-10, atol=1e-10)
