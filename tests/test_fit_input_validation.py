from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, SuperGLM
from superglm.distributions import (
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    validate_response,
)
from superglm.model.input_validation import validate_fit_input

ENTRYPOINTS = ("fit", "fit_path", "fit_reml")


def _model(*, family="gaussian") -> SuperGLM:
    return SuperGLM(
        family=family,
        selection_penalty=0.0,
        features={"x": Numeric()},
    )


def _call_entrypoint(
    model: SuperGLM,
    entrypoint: str,
    X,
    y,
    *,
    sample_weight=None,
    offset=None,
):
    kwargs = {"sample_weight": sample_weight, "offset": offset}
    if entrypoint == "fit_path":
        return model.fit_path(X, y, n_lambda=2, **kwargs)
    if entrypoint == "fit_reml":
        return model.fit_reml(X, y, max_reml_iter=1, **kwargs)
    return model.fit(X, y, **kwargs)


def _fail_if_feature_builds(*args, **kwargs):
    pytest.fail("feature built before fit input validation")


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("y", np.array([]), "y must be non-empty"),
        ("y", np.array([[0.0], [1.0]]), "y must be one-dimensional"),
        ("y", np.array([0.0, np.nan]), "y must contain only finite"),
        ("y", np.array([0.0, np.inf]), "y must contain only finite"),
        ("y", np.array(["not", "numeric"]), "y must contain only real numeric"),
        (
            "y",
            np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[D]"),
            "y must contain only real numeric",
        ),
        ("y", np.array([10**1000, 1], dtype=object), "y must contain only real numeric"),
        ("y", np.array([0.0]), "y must have length 2"),
        ("y", np.array([0.0 + 1.0j, 1.0]), "y must be real-valued"),
        ("sample_weight", np.array([1.0, -1.0]), "sample_weight must be nonnegative"),
        ("sample_weight", np.array([0.0, 0.0]), "sample_weight must not be all zero"),
        (
            "sample_weight",
            np.array([1.0, np.nan]),
            "sample_weight must contain only finite",
        ),
        (
            "sample_weight",
            np.array([1.0, np.inf]),
            "sample_weight must contain only finite",
        ),
        (
            "sample_weight",
            np.array(["not", "numeric"]),
            "sample_weight must contain only real numeric",
        ),
        (
            "sample_weight",
            np.array([1, 2], dtype="timedelta64[D]"),
            "sample_weight must contain only real numeric",
        ),
        (
            "sample_weight",
            np.array([[1.0], [1.0]]),
            "sample_weight must be one-dimensional",
        ),
        ("sample_weight", np.array([1.0]), "sample_weight must have length 2"),
        (
            "sample_weight",
            np.array([1.0 + 1.0j, 1.0]),
            "sample_weight must be real-valued",
        ),
        ("offset", np.array([0.0, np.inf]), "offset must contain only finite"),
        ("offset", np.array([0.0, np.nan]), "offset must contain only finite"),
        ("offset", np.array(["not", "numeric"]), "offset must contain only real numeric"),
        (
            "offset",
            np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[D]"),
            "offset must contain only real numeric",
        ),
        ("offset", np.array([[0.0], [0.0]]), "offset must be one-dimensional"),
        ("offset", np.array([0.0]), "offset must have length 2"),
        ("offset", np.array([0.0 + 1.0j, 0.0]), "offset must be real-valued"),
    ],
)
def test_fit_entrypoints_validate_vectors_before_feature_build(
    entrypoint: str,
    field: str,
    value,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame({"x": [0.0, 1.0]})
    arguments = {
        "y": np.array([0.0, 1.0]),
        "sample_weight": None,
        "offset": None,
    }
    arguments[field] = value
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match=message):
        _call_entrypoint(
            _model(),
            entrypoint,
            X,
            arguments["y"],
            sample_weight=arguments["sample_weight"],
            offset=arguments["offset"],
        )


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    ("X", "message"),
    [
        (np.array([[0.0], [1.0]]), "X must be a pandas DataFrame"),
        (pd.DataFrame({"x": []}), "X must be non-empty"),
        (pd.DataFrame(index=[0, 1]), "X is missing required columns.*x"),
        (
            pd.DataFrame(np.ones((2, 2)), columns=["x", "x"]),
            "X columns must be unique",
        ),
        (pd.DataFrame({"z": [0.0, 1.0]}), "X is missing required columns.*x"),
        (
            pd.DataFrame({"x": np.array([0.0 + 1.0j, 1.0])}),
            "X column 'x' must be real-valued",
        ),
        (
            pd.DataFrame({"x": np.array([0.0 + 1.0j, 1.0], dtype=object)}),
            "X column 'x' must be real-valued",
        ),
    ],
)
def test_fit_entrypoints_validate_dataframe_before_feature_build(
    entrypoint: str,
    X,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match=message):
        _call_entrypoint(_model(), entrypoint, X, np.array([0.0, 1.0]))


def test_intercept_only_fit_allows_zero_column_dataframe() -> None:
    X = pd.DataFrame(index=pd.RangeIndex(6))
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={},
    )

    model.fit(X, y)

    assert model._dm.shape == (len(X), 0)
    assert model.result.beta.shape == (0,)
    np.testing.assert_allclose(model.predict(X), np.mean(y))


def test_intercept_only_fit_ignores_unused_complex_column() -> None:
    X = pd.DataFrame({"unused": np.arange(6, dtype=float) + 1.0j})
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={},
    )

    model.fit(X, y)

    assert model._dm.shape == (len(X), 0)
    assert model.result.beta.shape == (0,)
    np.testing.assert_allclose(model.predict(X), np.mean(y))


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_auto_detect_validates_complex_additional_column_before_build(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        splines=["s"],
    )
    X = pd.DataFrame(
        {
            "s": [0.0, 1.0],
            "z": np.array([0.0 + 1.0j, 1.0]),
        }
    )
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="X column 'z' must be real-valued"):
        _call_entrypoint(model, entrypoint, X, np.array([0.0, 1.0]))


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    ("family", "y", "message"),
    [
        (Binomial(), np.array([0.0, 0.5]), "Binomial family requires y in"),
        (Poisson(), np.array([0.0, -1.0]), "Poisson family requires nonnegative y"),
        (
            NegativeBinomial(theta=2.0),
            np.array([0.0, -1.0]),
            "NegativeBinomial family requires nonnegative y",
        ),
        (Gamma(), np.array([1.0, 0.0]), "Gamma family requires strictly positive y"),
        (Tweedie(p=1.5), np.array([0.0, -1.0]), "Tweedie family requires nonnegative y"),
    ],
)
def test_fit_entrypoints_validate_response_domain_before_feature_build(
    entrypoint: str,
    family,
    y: np.ndarray,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match=message):
        _call_entrypoint(
            _model(family=family),
            entrypoint,
            pd.DataFrame({"x": [0.0, 1.0]}),
            y,
        )


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize(
    "weights",
    [
        np.array([1.0, 0.0]),
        np.array([1.0, -1.0]),
        np.array([0.0, 0.0]),
        np.array([10**1000, 1], dtype=object),
        np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[D]"),
        np.array([1, 2], dtype="timedelta64[D]"),
    ],
)
def test_tweedie_entrypoints_require_strictly_positive_weights_before_build(
    entrypoint: str,
    weights: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="weights must be finite and strictly positive"):
        _call_entrypoint(
            _model(family=Tweedie(p=1.5)),
            entrypoint,
            pd.DataFrame({"x": [0.0, 1.0]}),
            np.array([0.0, 1.0]),
            sample_weight=weights,
        )


class HookedGaussian(Gaussian):
    def validate_response(self, y: np.ndarray) -> None:
        if np.any(y == 42.0):
            raise ValueError("custom response rule rejected 42")


class AliasHookGaussian(Gaussian):
    validate_response = staticmethod(validate_response)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_custom_distribution_response_hook_runs_before_feature_build(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="custom response rule rejected 42"):
        _call_entrypoint(
            _model(family=HookedGaussian()),
            entrypoint,
            pd.DataFrame({"x": [0.0, 1.0]}),
            np.array([1.0, 42.0]),
        )


def test_module_validate_response_alias_is_not_called_recursively() -> None:
    validate_response(np.array([0.0, 1.0]), AliasHookGaussian())


@pytest.mark.parametrize(
    ("family", "valid_y"),
    [
        (Binomial(), np.array([0.0, 1.0])),
        (Poisson(), np.array([0.0, 1.5])),
        (NegativeBinomial(theta=2.0), np.array([0.0, 1.5])),
        (Gamma(), np.array([np.finfo(float).tiny, 1.0])),
        (Tweedie(p=1.5), np.array([0.0, 1.5])),
        (Gaussian(), np.array([-1.0, 1.0])),
    ],
)
def test_validate_response_accepts_documented_domain_boundaries(family, valid_y) -> None:
    validate_response(valid_y, family)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_prevalidation_failure_preserves_previous_fit(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = 0.5 + 0.3 * X["x"].to_numpy()
    model = _model()
    model.fit(X, y)
    result_before = model.result
    prediction_before = model.predict(X)
    state_before = model.__dict__.copy()
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="y must contain only finite"):
        _call_entrypoint(model, entrypoint, X, np.full(len(X), np.nan))

    assert model.result is result_before
    assert model.__dict__.keys() == state_before.keys()
    assert all(model.__dict__[name] is value for name, value in state_before.items())
    np.testing.assert_array_equal(model.predict(X), prediction_before)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_failed_first_fit_does_not_trigger_auto_detection(entrypoint: str) -> None:
    model = SuperGLM(family="gaussian", selection_penalty=0.0, splines=[])
    state_before = model.__dict__.copy()

    with pytest.raises(ValueError, match="y must contain only finite"):
        _call_entrypoint(
            model,
            entrypoint,
            pd.DataFrame({"x": [0.0, 1.0]}),
            np.array([0.0, np.nan]),
        )

    assert model.__dict__.keys() == state_before.keys()
    assert all(model.__dict__[name] is value for name, value in state_before.items())


def test_fit_path_auto_detects_valid_shorthand_configuration() -> None:
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = 0.5 + 0.3 * X["x"].to_numpy()
    model = SuperGLM(family="gaussian", selection_penalty=0.1, splines=[])

    result = model.fit_path(X, y, n_lambda=2)

    assert "x" in model._specs
    assert result.coef_path.shape[0] == 2


def test_validate_fit_input_returns_float64_vectors_without_mutating_callers() -> None:
    X = pd.DataFrame({"x": [0, 1]})
    y = np.array([0, 1], dtype=np.int32)
    weights = np.array([1, 2], dtype=np.int32)
    offset = np.array([0, 1], dtype=np.int32)

    validated = validate_fit_input(
        X,
        y,
        weights,
        offset,
        family=Gaussian(),
        required_columns=("x",),
    )

    assert validated.X is X
    assert validated.y.dtype == np.float64
    assert validated.sample_weight.dtype == np.float64
    assert validated.offset is not None and validated.offset.dtype == np.float64
    np.testing.assert_array_equal(y, [0, 1])
    np.testing.assert_array_equal(weights, [1, 2])
    np.testing.assert_array_equal(offset, [0, 1])
