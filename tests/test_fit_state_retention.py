import pickle

import numpy as np
import pandas as pd
import pytest

import superglm.inference.metrics as metrics_module
from superglm import Categorical, Constraint, Numeric, RandomEffect, Spline, SuperGLM
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


def test_ordinary_metrics_use_compact_inference_after_fit_state_release():
    X, y, sample_weight = _sample_data(n=240)
    retained = _model(retain_fit_state=True).fit(X, y, sample_weight=sample_weight)
    released = _model(retain_fit_state=False).fit(X, y, sample_weight=sample_weight)

    retained_metrics = retained.metrics(X, y, sample_weight=sample_weight)
    released_metrics = released.metrics(X, y, sample_weight=sample_weight)
    compact = released.__dict__["_fit_inference_info"]

    _, _, inverse, augmented, _ = released_metrics._active_info

    assert released._dm is None
    np.testing.assert_allclose(inverse, compact["XtWX_inv"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(augmented, compact["XtWX_inv_aug"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        released_metrics.leverage,
        retained_metrics.leverage,
        rtol=1e-10,
        atol=1e-12,
    )
    for name, expected in retained_metrics.coefficient_se.items():
        np.testing.assert_allclose(
            released_metrics.coefficient_se[name],
            expected,
            rtol=1e-10,
            atol=1e-12,
        )
    np.testing.assert_allclose(released_metrics._active_R_factor, compact["R_a"])
    np.testing.assert_allclose(released_metrics._influence_edf[0], compact["edf"])
    np.testing.assert_allclose(released_metrics._influence_edf[1], compact["edf1"])


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


def test_released_metrics_reuse_compact_inference_for_equal_fit_geometry():
    X, y, sample_weight = _sample_data(n=240)
    model = _model(retain_fit_state=False).fit(X, y, sample_weight=sample_weight)

    metrics = model.metrics(X.copy(), y.copy(), sample_weight=sample_weight.copy())

    assert metrics._uses_compact_fit_inference
    np.testing.assert_allclose(
        metrics._active_info[2],
        model.__dict__["_fit_inference_info"]["XtWX_inv"],
        rtol=0.0,
        atol=0.0,
    )


def test_released_metrics_ignore_pandas_index_labels_when_geometry_matches():
    X, y, sample_weight = _sample_data(n=240)
    X.index = np.arange(1000, 1000 + 3 * len(X), 3)
    model = _model(retain_fit_state=False).fit(X, y, sample_weight=sample_weight)

    metrics = model.metrics(
        X.reset_index(drop=True),
        y.copy(),
        sample_weight=sample_weight.copy(),
    )

    assert metrics._uses_compact_fit_inference
    np.testing.assert_allclose(
        metrics._active_info[2],
        model.__dict__["_fit_inference_info"]["XtWX_inv"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("changed_geometry", ["rows", "weights", "offset"])
def test_released_metrics_reject_changed_inference_geometry(changed_geometry: str):
    X, y, sample_weight = _sample_data(n=240)
    model = _model(retain_fit_state=False).fit(X, y, sample_weight=sample_weight)
    evaluation_X = X.copy()
    evaluation_weights = sample_weight.copy()
    evaluation_offset = None
    if changed_geometry == "rows":
        evaluation_X["density"] = evaluation_X["density"].to_numpy()[::-1]
    elif changed_geometry == "weights":
        evaluation_weights *= np.linspace(0.5, 1.5, len(evaluation_weights))
    else:
        evaluation_offset = np.linspace(-0.2, 0.2, len(evaluation_X))

    metrics = model.metrics(
        evaluation_X,
        y,
        sample_weight=evaluation_weights,
        offset=evaluation_offset,
    )

    assert not metrics._uses_compact_fit_inference
    with pytest.raises(RuntimeError, match="fit geometry"):
        _ = metrics.coefficient_se


def _nested_data(sizes=(4, 12, 48), n=1200):
    """A strict make > model > variant chain with an exposure offset and prior weights.

    Two variants carry zero weight, so the fit centres its spline columns on the
    positive-weight rows while the public design centres them on all rows.
    """
    rng = np.random.default_rng(20260927)
    parents = [
        np.concatenate([np.arange(coarse), rng.integers(0, coarse, fine - coarse)])
        for coarse, fine in zip(sizes[:-1], sizes[1:], strict=True)
    ]
    codes = [rng.integers(0, sizes[-1], n)]
    for parent in reversed(parents):
        codes.insert(0, parent[codes[0]])
    x = rng.uniform(size=n)
    offset = np.log(rng.uniform(0.5, 2.0, n))
    eta = 0.3 * np.sin(6.0 * x) + sum(
        rng.normal(0.0, 0.25, size)[code] for size, code in zip(sizes, codes, strict=True)
    )
    y = rng.poisson(np.exp(eta + offset)).astype(float)
    labels = [codes[0].astype(str)]
    for code in codes[1:]:
        labels.append(np.char.add(np.char.add(labels[-1], ":"), code.astype(str)))
    X = pd.DataFrame({"x": x, "make": labels[0], "model": labels[1], "variant": labels[2]})
    weights = rng.uniform(0.5, 1.5, n)
    weights[np.isin(codes[-1], [3, 7])] = 0.0
    return X, y, weights, offset


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize(
    "chain", [("variant",), ("make", "model", "variant")], ids=["single", "nested"]
)
def test_pickled_structured_metrics_keep_the_compact_covariance(monkeypatch, chain, discrete):
    """A round trip leaves the caller's frame distinct from the retained copy.

    The training rows are then recognised by content, so the clone serves its
    retained compact covariance instead of re-forming the dense weighted Gram.
    """
    X, y, sample_weight, offset = _nested_data()
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        direct_solve="structured",
        discrete=discrete,
        features={"x": Spline(n_knots=6), **{name: RandomEffect() for name in chain}},
    ).fit_reml(X, y, sample_weight=sample_weight, offset=offset)
    assert model._reml_profile["structured_chain"] == chain
    clone = pickle.loads(pickle.dumps(model))

    def dense_gram(*_args):
        pytest.fail("metrics re-formed the dense weighted Gram")

    monkeypatch.setattr(metrics_module, "weighted_moments", dense_gram)
    restored = clone.metrics(X, y, sample_weight=sample_weight, offset=offset)
    assert restored._active_info[3] is clone._fit_inference_info["XtWX_inv_aug"]

    # Rows recognised by content are the training rows: leverage and edf read
    # the fit design, as the live model's own metrics do.  Any frame but the
    # retained one is predicted exactly, where a discrete fit's own means are
    # binned, so an equal copy is the reference for the deviance and for the
    # Pearson-scaled standard errors.
    live = model.metrics(X, y, sample_weight=sample_weight, offset=offset)
    copy = model.metrics(X.copy(), y, sample_weight=sample_weight, offset=offset)
    # Both sides read equal retained state; only the order of an at most
    # n-term reduction can separate them.
    rtol = len(y) * np.finfo(np.float64).eps
    pairs = [
        (restored.deviance, copy.deviance),
        (restored.leverage, live.leverage),
        (restored._influence_edf[0], live._influence_edf[0]),
        *((restored.coefficient_se[name], se) for name, se in copy.coefficient_se.items()),
    ]
    for actual, expected in pairs:
        atol = rtol * np.max(np.abs(expected))
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    changed_rows = X.assign(x=X["x"].to_numpy()[::-1])
    changed = clone.metrics(changed_rows, y, sample_weight=sample_weight, offset=offset)
    assert not changed._uses_compact_fit_inference
