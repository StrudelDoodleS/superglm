import pickle

import numpy as np
import pandas as pd

from superglm import Categorical, Constraint, Numeric, Spline, SuperGLM
from superglm.features.spline import PSpline


def _sample_data(n: int = 500):
    rng = np.random.default_rng(123)
    age = rng.uniform(18.0, 85.0, n)
    density = rng.normal(size=n)
    region = rng.choice(["A", "B", "C", "D"], size=n, p=[0.2, 0.3, 0.3, 0.2])
    sample_weight = rng.uniform(0.4, 1.2, n)
    eta = -2.0 + 0.015 * (age - 45.0) + 0.1 * density + 0.25 * (region == "A")
    y = rng.poisson(np.exp(eta) * sample_weight).astype(float)
    X = pd.DataFrame({"age": age, "density": density, "region": region})
    return X, y, sample_weight


def _model(*, retain_fit_state: bool):
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        retain_fit_state=retain_fit_state,
        features={
            "age": Spline(n_knots=8, penalty="ssp"),
            "density": Numeric(),
            "region": Categorical(base="first"),
        },
    )


def test_fit_can_release_training_design_state_after_eager_inference():
    X, y, sample_weight = _sample_data()

    retained = _model(retain_fit_state=True).fit(X, y, sample_weight=sample_weight)
    released = _model(retain_fit_state=False).fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(released.predict(X), retained.predict(X), rtol=1e-10, atol=1e-10)

    assert released._dm is None
    assert released._fit_X_ref is None
    assert released._fit_y_ref is None
    assert released._fit_sample_weight_ref is None
    assert released._fit_weights is None

    assert "_fit_inference_info" in released.__dict__
    assert "_coef_covariance" in released.__dict__
    assert "_group_edf" in released.__dict__
    assert released.__dict__["_fit_inference_info"]["W"].size == 0

    summary = released.summary()
    assert summary["fit"]["n_obs"] == len(X)

    ti = released.term_inference("age", with_se=True)
    assert ti.se_log_relativity is not None
    assert np.all(np.asarray(ti.se_log_relativity) >= 0.0)


def test_released_fit_state_reduces_serialized_model_size():
    X, y, sample_weight = _sample_data(n=1200)

    retained = _model(retain_fit_state=True).fit(X, y, sample_weight=sample_weight)
    released = _model(retain_fit_state=False).fit(X, y, sample_weight=sample_weight)

    retained_size = len(pickle.dumps(retained, protocol=pickle.HIGHEST_PROTOCOL))
    released_size = len(pickle.dumps(released, protocol=pickle.HIGHEST_PROTOCOL))

    assert released_size < retained_size * 0.5


def test_fit_reml_can_release_fit_state_after_eager_inference():
    X, y, sample_weight = _sample_data()
    model = _model(retain_fit_state=False)

    model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=3)

    assert model._dm is None
    assert model._fit_weights is None
    assert "_fit_inference_info" in model.__dict__
    assert "_coef_covariance" in model.__dict__

    preds = model.predict(X.head(10))
    assert preds.shape == (10,)
    assert np.all(preds > 0.0)

    assert model.summary()["fit"]["n_obs"] == len(X)
    assert model.term_inference("age", with_se=True).ci_lower is not None


def test_scop_metrics_use_compact_inference_after_fit_state_release():
    rng = np.random.default_rng(20260718)
    x = np.linspace(0.0, 1.0, 120)
    X = pd.DataFrame({"x": x})
    y = 0.2 + 1.7 * x + rng.normal(0.0, 0.12, size=x.size)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=1.7,
        retain_fit_state=False,
        features={"x": PSpline(n_knots=6, constraint=Constraint.fit.increasing)},
    ).fit(X, y)

    assert model._solver_result.scop_inference is not None
    assert model._dm is None
    assert "_fit_active_info" not in model.__dict__
    compact = model.__dict__["_fit_inference_info"]

    metrics = model.metrics(X, y)
    _, _, inverse, augmented, _ = metrics._active_info

    np.testing.assert_allclose(inverse, compact["XtWX_inv"])
    np.testing.assert_allclose(augmented, compact["XtWX_inv_aug"])
    assert np.all(np.isfinite(metrics.leverage))
    assert "_fit_active_info" not in model.__dict__
