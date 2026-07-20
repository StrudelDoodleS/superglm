"""Backend-neutral dataframe compilation contracts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import polars as pl

from superglm import Categorical, Numeric, Spline, SuperGLM
from superglm._frame import EagerFrame
from superglm.model.base import auto_detect, model_build_design_matrix
from superglm.model.input_validation import validate_fit_input


def _compile_without_solving(model: SuperGLM, X, y: np.ndarray) -> SuperGLM:
    """Compile one native frame only as far as the DesignMatrix boundary."""
    validated = validate_fit_input(
        X,
        y,
        None,
        None,
        family=model._config.family,
        required_columns=tuple(model._config.splines or model._feature_order),
        check_all_columns=model._config.splines is not None,
    )
    if model._splines is not None:
        auto_detect(model, validated.X, validated.sample_weight)
    model_build_design_matrix(
        model,
        validated.X,
        validated.y,
        validated.sample_weight,
        validated.offset,
    )
    return model


def _assert_compiled_models_equal(left: SuperGLM, right: SuperGLM) -> None:
    assert left._feature_order == right._feature_order
    assert left._interaction_order == right._interaction_order
    assert left._groups == right._groups
    for name in left._feature_order:
        assert type(left._specs[name]) is type(right._specs[name])
    assert [type(group) for group in left._dm.group_matrices] == [
        type(group) for group in right._dm.group_matrices
    ]
    np.testing.assert_allclose(left._dm.toarray(), right._dm.toarray(), rtol=0.0, atol=0.0)
    for left_group, right_group in zip(left._dm.group_matrices, right._dm.group_matrices):
        left_penalty = getattr(left_group, "omega", None)
        right_penalty = getattr(right_group, "omega", None)
        if left_penalty is not None or right_penalty is not None:
            np.testing.assert_allclose(left_penalty, right_penalty, rtol=0.0, atol=0.0)


def _stored_values(value) -> list[object]:
    stored = list(getattr(value, "__dict__", {}).values())
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        stored.extend(getattr(value, name) for name in slots if hasattr(value, name))
    return stored


def test_dataframe_boundary_compiles_mixed_auto_detected_terms_identically() -> None:
    n_rows = 72
    row = np.arange(n_rows)
    data = {
        "numeric": np.linspace(-1.0, 1.0, n_rows),
        "flag": row % 2 == 0,
        "string_cat": np.array(["bronze", "silver", "gold"])[row % 3],
        "enum_cat": np.array(["low", "mid", "high", "mid"])[row % 4],
        "smooth": np.linspace(0.0, 4.0, n_rows),
    }
    pandas_X = pd.DataFrame(data)
    pandas_X["enum_cat"] = pd.Categorical(
        pandas_X["enum_cat"],
        categories=["low", "mid", "high"],
    )
    polars_X = pl.DataFrame(data).with_columns(
        pl.col("enum_cat").cast(pl.Enum(["low", "mid", "high"]))
    )
    y = 1.0 + 0.3 * data["numeric"] + 0.1 * data["flag"]

    def make_model() -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            splines=["smooth"],
            n_knots=6,
            categorical_base="first",
            interactions=[("numeric", "string_cat"), ("smooth", "enum_cat")],
        )

    pandas_model = _compile_without_solving(make_model(), pandas_X, y)
    polars_model = _compile_without_solving(make_model(), polars_X, y)

    _assert_compiled_models_equal(pandas_model, polars_model)


def test_dataframe_boundary_compiles_discrete_tensor_identically() -> None:
    n_rows = 96
    phase = np.linspace(0.0, 2.0 * np.pi, n_rows)
    data = {
        "left": np.linspace(-2.0, 2.0, n_rows),
        "right": np.sin(phase) + 0.1 * np.cos(3.0 * phase),
    }
    pandas_X = pd.DataFrame(data)
    polars_X = pl.DataFrame(data)
    y = 2.0 + 0.2 * data["left"] - 0.1 * data["right"]

    def make_model() -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            discrete=True,
            n_bins={"left": 18, "right": 16},
            features={
                "left": Spline(n_knots=6, penalty="ssp"),
                "right": Spline(n_knots=5, penalty="ssp"),
            },
            interactions=[("left", "right")],
        )

    pandas_model = _compile_without_solving(make_model(), pandas_X, y)
    polars_model = _compile_without_solving(make_model(), polars_X, y)

    _assert_compiled_models_equal(pandas_model, polars_model)


def test_dataframe_boundary_extracts_each_polars_column_once_per_compile(monkeypatch) -> None:
    X = pl.DataFrame(
        {
            "left": np.linspace(-1.0, 1.0, 40),
            "right": np.linspace(2.0, 4.0, 40),
        }
    )
    y = 0.5 + 0.2 * X["left"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"left": Numeric(), "right": Numeric()},
        interactions=[("left", "right")],
    )
    calls: Counter[object] = Counter()
    original = EagerFrame._extract_column

    def counted_extract(frame: EagerFrame, name: object):
        calls[name] += 1
        return original(frame, name)

    monkeypatch.setattr(EagerFrame, "_extract_column", counted_extract)

    _compile_without_solving(model, X, y)

    assert calls == Counter({"left": 1, "right": 1})


def test_dataframe_boundary_does_not_leak_adapter_into_matrix_execution_state() -> None:
    X = pl.DataFrame(
        {
            "left": np.linspace(-1.0, 1.0, 40),
            "right": np.linspace(2.0, 4.0, 40),
        }
    )
    y = 0.5 + 0.2 * X["left"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"left": Numeric(), "right": Numeric()},
    ).fit(X, y)

    boundary_types = (EagerFrame, pl.DataFrame)
    assert not any(isinstance(value, boundary_types) for value in _stored_values(model._dm))
    for group_matrix in model._dm.group_matrices:
        assert not any(isinstance(value, boundary_types) for value in _stored_values(group_matrix))

    # Retained-fit behavior deliberately keeps the caller's native frame, but
    # the private adapter itself must never become published model state.
    assert model._fit_state.projections["_fit_X_ref"] is X
    assert not any(isinstance(value, EagerFrame) for value in model._fit_state.projections.values())


def _mixed_prediction_frames() -> tuple[pd.DataFrame, pl.DataFrame, np.ndarray]:
    n_rows = 120
    row = np.arange(n_rows)
    numeric = np.linspace(-1.5, 1.5, n_rows)
    smooth = np.linspace(0.0, 3.0, n_rows)
    category = np.array(["a", "b", "c"])[row % 3]
    data = {"numeric": numeric, "smooth": smooth, "category": category}
    y = 1.5 + 0.25 * numeric + 0.2 * np.sin(smooth) + 0.15 * (category == "b")
    return pd.DataFrame(data), pl.DataFrame(data), y


def _mixed_prediction_model() -> SuperGLM:
    return SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "numeric": Numeric(),
            "smooth": Spline(n_knots=6, penalty="ssp"),
            "category": Categorical(base="first"),
        },
        interactions=[("numeric", "category"), ("smooth", "category")],
    )


def test_dataframe_boundary_predicts_mixed_terms_identically() -> None:
    pandas_X, polars_X, y = _mixed_prediction_frames()
    model = _mixed_prediction_model().fit(pandas_X, y)

    pandas_eta = model._predict_eta_exact(pandas_X)
    polars_eta = model._predict_eta_exact(polars_X)
    pandas_mu = model.predict(pandas_X)
    polars_mu = model.predict(polars_X)

    np.testing.assert_allclose(polars_eta, pandas_eta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(polars_mu, pandas_mu, rtol=0.0, atol=0.0)


def test_dataframe_boundary_fast_discrete_tensor_prediction_is_backend_neutral() -> None:
    n_rows = 100
    phase = np.linspace(0.0, 2.0 * np.pi, n_rows)
    data = {
        "left": np.linspace(-2.0, 2.0, n_rows),
        "right": np.sin(phase) + 0.1 * np.cos(3.0 * phase),
    }
    pandas_X = pd.DataFrame(data)
    polars_X = pl.DataFrame(data)
    y = 2.0 + 0.2 * data["left"] - 0.1 * data["right"]
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=True,
        n_bins={"left": 18, "right": 16},
        features={
            "left": Spline(n_knots=6, penalty="ssp"),
            "right": Spline(n_knots=5, penalty="ssp"),
        },
        interactions=[("left", "right")],
    ).fit(pandas_X, y)

    pandas_eta = model._predict_eta_fast_discrete(pandas_X)
    polars_eta = model._predict_eta_fast_discrete(polars_X)
    pandas_mu = model._predict_fast_discrete(pandas_X)
    polars_mu = model._predict_fast_discrete(polars_X)

    np.testing.assert_allclose(polars_eta, pandas_eta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(polars_mu, pandas_mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        model._predict_eta_exact(polars_X),
        model._predict_eta_exact(pandas_X),
        rtol=0.0,
        atol=0.0,
    )


def test_dataframe_boundary_prediction_extracts_shared_parents_once(monkeypatch) -> None:
    pandas_X, polars_X, y = _mixed_prediction_frames()
    model = _mixed_prediction_model().fit(pandas_X, y)
    calls: Counter[object] = Counter()
    original = EagerFrame._extract_column

    def counted_extract(frame: EagerFrame, name: object):
        calls[name] += 1
        return original(frame, name)

    monkeypatch.setattr(EagerFrame, "_extract_column", counted_extract)

    model.predict(polars_X)

    assert calls == Counter({"numeric": 1, "smooth": 1, "category": 1})


def test_dataframe_boundary_prediction_rejects_missing_and_unseen_categorical_data() -> None:
    pandas_X, polars_X, y = _mixed_prediction_frames()
    model = _mixed_prediction_model().fit(pandas_X, y)

    with np.testing.assert_raises_regex(ValueError, "missing required columns.*smooth"):
        model.predict(polars_X.drop("smooth"))

    with np.testing.assert_raises_regex(ValueError, "unseen categorical levels.*unseen"):
        model.predict(polars_X.with_columns(pl.lit("unseen").alias("category")))
