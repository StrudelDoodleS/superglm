"""OrderedCategorical as an interaction parent: build-side enabler."""

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.ordered_categorical import (
    OrderedCategorical,
    resolve_interaction_parent,
)
from superglm.features.spline import Spline, _SplineBase

BANDS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age_band": rng.choice(BANDS, n),
            "region": rng.choice(list("ABCD"), n),
            "power": rng.uniform(20.0, 200.0, n),
        }
    )
    band_effect = df["age_band"].map({b: v for b, v in zip(BANDS, [0.4, 0.1, 0.0, 0.1, 0.3])})
    eta = -1.5 + band_effect + 0.002 * df["power"]
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    return df, y


def _oc():
    return OrderedCategorical(order=BANDS, basis=Spline(kind="ps", n_knots=4))


def test_resolver_is_identity_for_non_oc_and_none():
    x = np.array([1.0, 2.0])
    spec = Spline(kind="ps", n_knots=4)
    assert resolve_interaction_parent(spec, x) == (spec, x)
    assert resolve_interaction_parent(None, x) == (None, x)


def test_resolver_maps_oc_to_inner_spline_and_scores():
    spec = _oc()
    labels = np.array(["18-25", "56+", "36-45"], dtype=object)
    eff_spec, eff_x = resolve_interaction_parent(spec, labels)
    assert isinstance(eff_spec, _SplineBase)
    assert eff_spec is spec._spline
    expected = [spec._level_to_value[v] for v in labels]
    np.testing.assert_allclose(eff_x, expected)


def test_resolver_rejects_step_mode():
    with pytest.warns(FutureWarning):
        spec = OrderedCategorical(order=BANDS, basis="step")
    with pytest.raises(NotImplementedError, match="step"):
        resolve_interaction_parent(spec, np.array(BANDS, dtype=object))


def test_resolver_rejects_unseen_levels():
    spec = _oc()
    with pytest.raises(ValueError, match="unseen|levels"):
        resolve_interaction_parent(spec, np.array(["99+"], dtype=object))


def test_oc_categorical_interaction_fits():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "region": Categorical()},
        interactions=[("age_band", "region")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_oc_spline_tensor_interaction_fits():
    df, y = _frame()
    model = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model.fit_reml(df, y)
    assert np.isfinite(model._result.effective_df)


def test_oc_tensor_fit_matches_manual_score_mapping():
    # The OC×spline fit must equal the same model fitted on the scores directly.
    df, y = _frame()
    oc = _oc()
    model_oc = SuperGLM(
        family="poisson",
        features={"age_band": _oc(), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_oc.fit_reml(df, y)

    df_num = df.copy()
    df_num["age_band"] = df_num["age_band"].map(oc._level_to_value)
    model_num = SuperGLM(
        family="poisson",
        features={"age_band": Spline(kind="ps", n_knots=4), "power": Spline(kind="ps", n_knots=5)},
        interactions=[("age_band", "power")],
    )
    model_num.fit_reml(df_num, y)
    np.testing.assert_allclose(model_oc._result.deviance, model_num._result.deviance, rtol=1e-6)
