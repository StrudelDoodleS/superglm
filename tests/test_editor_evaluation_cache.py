from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from superglm import Numeric, SuperGLM
from superglm.editor.evaluation import EvaluationDataset, coerce_dataset
from superglm.editor.evaluation_cache import (
    EvaluationCache,
    EvaluationKey,
    model_metric_signature,
)
from superglm.editor.metrics import compute_dataset_metrics


def _key(role: str, revision: int, *, split: str = "validation") -> EvaluationKey:
    return EvaluationKey(
        role=role,
        model_revision=revision,
        dataset_epoch=0,
        split=split,
        metric_signature=("gaussian", "identity", 1.0),
    )


def test_evaluation_key_is_frozen():
    key = _key("original", 0)

    with pytest.raises(FrozenInstanceError):
        key.split = "test"  # type: ignore[misc]


def test_polars_evaluation_dataset_keeps_native_frame_and_uses_its_row_count():
    X = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    dataset = coerce_dataset("validation", (X, np.array([1.0, 2.0, 3.0])))

    assert dataset is not None
    assert dataset.X is X
    assert dataset.n_obs == X.height

    class ArrayConversionForbidden:
        def __array__(self):
            raise AssertionError("n_obs must come from the native frame")

    assert (
        EvaluationDataset(
            "validation",
            "Validation",
            X,
            ArrayConversionForbidden(),
        ).n_obs
        == X.height
    )


def test_evaluation_cache_preserves_original_and_bounds_current_revisions():
    cache = EvaluationCache()
    original = _key("original", 0)
    current_1 = _key("current", 1)
    current_2 = _key("current", 2)

    assert cache.put(original, {"deviance": 1.0, "aic": 2.0}) is True
    cache.advance_current_revision(1)
    assert cache.put(current_1, {"deviance": 1.0, "aic": 2.0}) is True
    cache.advance_current_revision(2)
    assert cache.put(current_2, {"deviance": 3.0, "aic": 4.0}) is True

    assert cache.get(original) == {"deviance": 1.0, "aic": 2.0}
    assert cache.get(current_1) is None
    assert cache.get(current_2) == {"deviance": 3.0, "aic": 4.0}


def test_evaluation_cache_scalarizes_values_and_returns_isolated_copies():
    cache = EvaluationCache()
    key = _key("original", 0, split="train")
    source = {"deviance": np.float32(1.25), "effective_df": np.int64(3)}

    assert cache.put(key, source) is True
    source["deviance"] = np.float32(9.0)
    first = cache.get(key)
    assert first == {"deviance": 1.25, "effective_df": 3.0}
    assert first is not None
    assert all(type(value) is float for value in first.values())

    first["deviance"] = -1.0
    assert cache.get(key) == {"deviance": 1.25, "effective_df": 3.0}
    assert cache.persistent_values_are_scalar() is True


def test_evaluation_cache_rejects_stale_current_revision_writes():
    cache = EvaluationCache()
    cache.advance_current_revision(4)

    assert cache.put(_key("current", 3), {"deviance": 1.0}) is False
    assert cache.get(_key("current", 3)) is None
    assert cache.put(_key("current", 4), {"deviance": 2.0}) is True
    assert cache.get(_key("current", 4)) == {"deviance": 2.0}


def test_model_metric_signature_tracks_metric_affecting_model_state():
    class Family:
        p = 1.45
        theta = 2.75

    class Link:
        pass

    model = SimpleNamespace(
        _distribution=Family(),
        _link=Link(),
        result=SimpleNamespace(phi=np.float64(0.8), effective_df=np.float64(4.25)),
    )

    signature = model_metric_signature(model)

    assert signature == (
        Family.__module__,
        Family.__qualname__,
        1.45,
        2.75,
        Link.__module__,
        Link.__qualname__,
        0.8,
        4.25,
    )


@pytest.fixture
def weighted_offset_fit():
    rng = np.random.default_rng(20260711)
    n = 120
    x = rng.normal(size=n)
    X = pd.DataFrame({"x": x})
    sample_weight = rng.uniform(0.6, 1.8, size=n)
    offset = np.linspace(-0.7, 0.9, n)
    y = 0.9 + 0.35 * x + offset + rng.normal(0.0, 0.08, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)
    return model, X, y, sample_weight, offset


@pytest.mark.parametrize("use_retained_arrays", [False, True])
def test_matching_fit_dataset_uses_fit_artifacts_without_model_scoring(
    weighted_offset_fit,
    monkeypatch,
    use_retained_arrays,
):
    model, X, y, sample_weight, offset = weighted_offset_fit
    dataset = EvaluationDataset(
        "train",
        "Train",
        X,
        y,
        sample_weight=model._fit_weights if use_retained_arrays else sample_weight,
        offset=model._fit_offset if use_retained_arrays else offset,
        source="retained_fit_data" if use_retained_arrays else "supplied",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fit-identity metrics must not score the model")

    monkeypatch.setattr(model, "predict", forbidden)
    monkeypatch.setattr(model, "metrics", forbidden)

    metrics = compute_dataset_metrics(model, dataset)

    fit_stats = model._fit_stats
    assert fit_stats is not None
    edf = float(model.result.effective_df)
    log_likelihood = float(fit_stats.log_likelihood)
    aic = -2.0 * log_likelihood + 2.0 * edf
    denom = dataset.n_obs - edf - 1.0
    assert set(metrics) == {
        "deviance",
        "aic",
        "aicc",
        "bic",
        "log_likelihood",
        "explained_deviance",
        "pearson_chi2",
        "effective_df",
    }
    assert metrics["deviance"] == pytest.approx(model.result.deviance)
    assert metrics["aic"] == pytest.approx(aic)
    assert metrics["aicc"] == pytest.approx(aic + 2.0 * edf * (edf + 1.0) / denom)
    assert metrics["bic"] == pytest.approx(-2.0 * log_likelihood + np.log(dataset.n_obs) * edf)
    assert metrics["log_likelihood"] == pytest.approx(fit_stats.log_likelihood)
    assert metrics["explained_deviance"] == pytest.approx(fit_stats.explained_deviance)
    assert metrics["pearson_chi2"] == pytest.approx(fit_stats.pearson_chi2)
    assert metrics["effective_df"] == pytest.approx(edf)


def test_matching_unweighted_fit_dataset_uses_fit_artifacts(monkeypatch):
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 40)})
    y = 0.4 + 0.2 * X["x"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    model.fit(X, y)
    dataset = EvaluationDataset("train", "Train", X, y)

    monkeypatch.setattr(
        model,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("predict called")),
    )
    monkeypatch.setattr(
        model,
        "metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("metrics called")),
    )

    metrics = compute_dataset_metrics(model, dataset)

    assert metrics["deviance"] == pytest.approx(model.result.deviance)
    assert metrics["log_likelihood"] == pytest.approx(model._fit_stats.log_likelihood)


@pytest.mark.parametrize(
    ("split", "row_slice"),
    [("validation", slice(0, 36)), ("test", slice(72, 108))],
)
def test_validation_and_test_datasets_use_exact_prediction_fallback(
    weighted_offset_fit,
    monkeypatch,
    split,
    row_slice,
):
    from superglm.model.fit_ops import _compute_null_mu

    model, X, y, sample_weight, offset = weighted_offset_fit
    X_eval = X.iloc[row_slice].copy()
    y_eval = y[row_slice].copy()
    weight_eval = sample_weight[row_slice].copy()
    offset_eval = offset[row_slice].copy()
    dataset = EvaluationDataset(
        split,
        split.title(),
        X_eval,
        y_eval,
        sample_weight=weight_eval,
        offset=offset_eval,
    )
    predict_calls = []
    original_predict = model.predict

    def counted_predict(X_arg, *, offset=None):
        predict_calls.append(X_arg)
        return original_predict(X_arg, offset=offset)

    monkeypatch.setattr(model, "predict", counted_predict)
    monkeypatch.setattr(
        model,
        "metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("metrics called")),
    )

    metrics = compute_dataset_metrics(model, dataset)

    assert predict_calls == [X_eval]
    mu = np.asarray(original_predict(X_eval, offset=offset_eval), dtype=np.float64).ravel()
    deviance = float(np.sum(weight_eval * model._distribution.deviance_unit(y_eval, mu)))
    log_likelihood = float(
        model._distribution.log_likelihood(
            y_eval,
            mu,
            weight_eval,
            float(model.result.phi),
        )
    )
    null_mu = _compute_null_mu(
        y_eval,
        weight_eval,
        offset_eval,
        model._distribution,
        model._link,
    )
    null_deviance = float(np.sum(weight_eval * model._distribution.deviance_unit(y_eval, null_mu)))
    pearson = float(np.sum(weight_eval * (y_eval - mu) ** 2 / model._distribution.variance(mu)))
    assert metrics["deviance"] == pytest.approx(deviance)
    assert metrics["log_likelihood"] == pytest.approx(log_likelihood)
    assert metrics["explained_deviance"] == pytest.approx(1.0 - deviance / null_deviance)
    assert metrics["pearson_chi2"] == pytest.approx(pearson)


def test_equal_but_nonidentical_training_data_uses_prediction_fallback(
    weighted_offset_fit,
    monkeypatch,
):
    model, X, y, sample_weight, offset = weighted_offset_fit
    dataset = EvaluationDataset(
        "train",
        "Train",
        X.copy(),
        y.copy(),
        sample_weight=sample_weight.copy(),
        offset=offset.copy(),
    )
    calls = 0
    original_predict = model.predict

    def counted_predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", counted_predict)

    metrics = compute_dataset_metrics(model, dataset)

    assert calls == 1
    assert metrics["deviance"] == pytest.approx(model.result.deviance)


def test_dataset_metric_fallback_flattens_column_vector_inputs(
    weighted_offset_fit,
):
    model, X, y, sample_weight, offset = weighted_offset_fit
    rows = slice(36, 72)
    X_eval = X.iloc[rows].copy()
    y_eval = y[rows].copy()
    weight_eval = sample_weight[rows].copy()
    offset_eval = offset[rows].copy()
    flat = EvaluationDataset(
        "validation",
        "Validation",
        X_eval,
        y_eval,
        sample_weight=weight_eval,
        offset=offset_eval,
    )
    column = EvaluationDataset(
        "validation",
        "Validation",
        X_eval,
        y_eval.reshape(-1, 1),
        sample_weight=weight_eval.reshape(-1, 1),
        offset=offset_eval.reshape(-1, 1),
    )

    flat_metrics = compute_dataset_metrics(model, flat)
    column_metrics = compute_dataset_metrics(model, column)

    assert column_metrics == pytest.approx(flat_metrics)


def test_fit_identity_falls_back_when_scalar_fit_artifacts_are_unavailable(
    weighted_offset_fit,
    monkeypatch,
):
    model, X, y, sample_weight, offset = weighted_offset_fit
    dataset = EvaluationDataset(
        "train",
        "Train",
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )
    calls = 0
    original_predict = model.predict

    def counted_predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", counted_predict)
    monkeypatch.setattr(model, "_fit_stats", None)

    compute_dataset_metrics(model, dataset)

    assert calls == 1
