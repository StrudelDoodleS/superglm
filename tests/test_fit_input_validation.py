from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from superglm import (
    Categorical,
    Constraint,
    LambdaPolicy,
    Numeric,
    PSpline,
    RandomEffect,
    Spline,
    SuperGLM,
)
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
from superglm.model.reml_setup import scop_group_spec
from superglm.types import GroupSlice

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
        (np.array([[0.0], [1.0]]), "X must be a pandas or eager Polars DataFrame"),
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


def test_explicit_intercept_only_intent_survives_clone() -> None:
    X = pd.DataFrame({"unused": np.arange(6, dtype=float)})
    y = np.arange(1.0, 7.0)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={},
    ).clone_unfitted()

    model.fit(X, y)

    assert model._config.features_explicit
    assert model._dm.shape == (len(X), 0)
    np.testing.assert_allclose(model.predict(X), np.mean(y))


def test_clone_preserves_omitted_versus_explicit_empty_feature_configuration() -> None:
    omitted = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
    ).clone_unfitted()
    explicit_empty = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={},
    ).clone_unfitted()

    assert not omitted._config.features_explicit
    assert omitted._config.constructor_kwargs()["features"] is None
    assert explicit_empty._config.features_explicit
    assert explicit_empty._config.constructor_kwargs()["features"] == {}

    X = pd.DataFrame({"unused": [0.0, 1.0]})
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="no features were configured"):
        omitted.fit(X, y)
    explicit_empty.fit(X, y)
    np.testing.assert_allclose(explicit_empty.predict(X), np.mean(y))


@pytest.mark.parametrize("pickle_roundtrip", [False, True], ids=["live-state", "pickle"])
def test_legacy_model_config_without_features_explicit_clones_and_refits(
    pickle_roundtrip: bool,
) -> None:
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 20)})
    y = 1.0 + 0.4 * X["x"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric()},
    ).fit(X, y)

    object.__delattr__(model._config, "features_explicit")
    del model._features_explicit
    if pickle_roundtrip:
        model = pickle.loads(pickle.dumps(model))

    clone = model.clone_unfitted().fit(X, y)
    model.fit(X, y)

    np.testing.assert_allclose(clone.predict(X), y, atol=1e-12)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-12)
    assert clone._config.features_explicit
    assert model._config.features_explicit


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_omitted_feature_configuration_rejects_nonempty_column_frame(
    entrypoint: str,
) -> None:
    X = pd.DataFrame({"ignored": np.linspace(-1.0, 1.0, 20)})
    y = 1.0 + X["ignored"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.1 if entrypoint == "fit_path" else 0.0,
    )

    with pytest.raises(ValueError, match="no features were configured.*features=\\{\\}"):
        _call_entrypoint(model, entrypoint, X, y)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_fit_entrypoints_reject_nonfinite_numeric_x_before_feature_build(
    entrypoint: str,
    bad_value: float,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {"x": [0.0, bad_value]}
    X = pd.DataFrame(data) if backend == "pandas" else pl.DataFrame(data)
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="X column 'x' must contain only finite values"):
        _call_entrypoint(_model(), entrypoint, X, np.array([0.0, 1.0]))


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("storage", ["object", "category"])
def test_fit_entrypoints_reject_nonfinite_numeric_like_pandas_columns(
    entrypoint: str,
    bad_value: float,
    storage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = pd.Series([0.0, bad_value], dtype=storage)
    X = pd.DataFrame({"x": values})
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="X column 'x' must contain only finite values"):
        _call_entrypoint(_model(), entrypoint, X, np.array([0.0, 1.0]))


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_fit_entrypoints_reject_array_valued_cells_before_feature_build(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pd.DataFrame(
        {
            "x": [
                np.array([0.0, 1.0]),
                np.array([2.0, 3.0]),
            ]
        }
    )
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="X column 'x' must contain only scalar values"):
        _call_entrypoint(_model(), entrypoint, X, np.array([0.0, 1.0]))


def test_hashable_tuple_categorical_levels_fit_and_predict() -> None:
    levels = pd.Series(
        [("low", 1), ("high", 2)] * 20,
        dtype=object,
    )
    X = pd.DataFrame({"group": levels})
    y = np.tile(np.array([1.0, 3.0]), 20)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"group": Categorical(base="first")},
    ).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-12)


@pytest.mark.parametrize("spec", [Numeric(), Spline(n_knots=4)])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_predict_rejects_nonfinite_numeric_x_consistently(
    spec,
    bad_value: float,
    backend: str,
) -> None:
    x = np.linspace(-1.0, 1.0, 40)
    X = pd.DataFrame({"x": x})
    y = 1.0 + 0.4 * x
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": spec},
    ).fit(X, y)
    bad_x = x.copy()
    bad_x[5] = bad_value
    X_bad = pd.DataFrame({"x": bad_x}) if backend == "pandas" else pl.DataFrame({"x": bad_x})

    with pytest.raises(ValueError, match="X column 'x' must contain only finite values"):
        model.predict(X_bad)


@pytest.mark.parametrize("spec", [Numeric(), Spline(n_knots=4)])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("storage", ["object", "category"])
def test_predict_rejects_nonfinite_numeric_like_pandas_columns(
    spec,
    bad_value: float,
    storage: str,
) -> None:
    x = np.linspace(-1.0, 1.0, 40)
    X = pd.DataFrame({"x": x})
    y = 1.0 + 0.4 * x
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": spec},
    ).fit(X, y)
    bad_x = x.astype(object)
    bad_x[5] = bad_value
    X_bad = pd.DataFrame({"x": pd.Series(bad_x, dtype=storage)})

    with pytest.raises(ValueError, match="X column 'x' must contain only finite values"):
        model.predict(X_bad)


def test_predict_rejects_numpy_matrix_at_dataframe_boundary() -> None:
    x = np.linspace(-1.0, 1.0, 20)
    X = pd.DataFrame({"x": x})
    model = _model().fit(X, 1.0 + 0.2 * x)

    with pytest.raises(ValueError, match="X must be a pandas or eager Polars DataFrame"):
        model.predict(x.reshape(-1, 1))


@pytest.mark.parametrize(
    ("offset", "message"),
    [
        (np.array([0.0, np.nan]), "offset must contain only finite values"),
        (np.array([0.0, np.inf]), "offset must contain only finite values"),
        (np.array([0.0, -np.inf]), "offset must contain only finite values"),
        (np.array([0.0]), "offset must have length 2, got 1"),
        (np.array(0.0), "offset must be one-dimensional"),
        (np.zeros((2, 1)), "offset must be one-dimensional"),
        (np.array([0.0 + 1.0j, 0.0]), "offset must be real-valued"),
    ],
)
def test_predict_validates_offset_before_broadcasting(offset, message) -> None:
    X = pd.DataFrame({"x": [0.0, 1.0]})
    model = _model().fit(X, np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match=message):
        model.predict(X, offset=offset)


@pytest.mark.parametrize(
    "label",
    [0, 0.0, False, "", None, ("numeric", "column")],
    ids=["int-zero", "float-zero", "false", "empty-string", "none", "tuple"],
)
def test_hashable_numeric_column_label_fits_clones_and_predicts(label) -> None:
    x = np.linspace(-1.0, 1.0, 40)
    X = pd.DataFrame({label: x})
    y = 2.0 + 0.5 * x
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={label: Numeric()},
    ).fit(X, y)

    assert model._groups[0].feature_name == label
    np.testing.assert_allclose(model.predict(X), y, atol=1e-12)
    assert "Intercept" in str(model.summary())
    payload = model.plot_data(label, ci=None, show_density=False)
    assert payload["terms"][0]["name"] == label
    assert model.plot(label, ci=None, show_density=False) is not None
    clone = model.clone_unfitted().fit(X, y)
    assert clone._groups[0].feature_name == label
    np.testing.assert_allclose(clone.predict(X), y, atol=1e-12)


@pytest.mark.parametrize(
    "label",
    [0, 0.0, False, "", None, ("numeric", "column")],
    ids=["int-zero", "float-zero", "false", "empty-string", "none", "tuple"],
)
def test_hashable_numeric_column_label_auto_detects_and_predicts(label) -> None:
    x = np.linspace(-1.0, 1.0, 40)
    X = pd.DataFrame({label: x})
    y = 2.0 + 0.5 * x
    with pytest.warns(FutureWarning, match="auto-detection is deprecated"):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            splines=[],
        )

    model.fit(X, y)

    assert model._feature_order == [label]
    assert model._groups[0].feature_name == label
    np.testing.assert_allclose(model.predict(X), y, atol=1e-12)


def test_omitted_plot_terms_and_explicit_none_label_are_distinct() -> None:
    columns = pd.Index([None, "z"], dtype=object)
    X = pd.DataFrame(
        np.column_stack(
            [
                np.linspace(-1.0, 1.0, 40),
                np.linspace(1.0, -1.0, 40),
            ]
        ),
        columns=columns,
    )
    y = 1.0 + 0.2 * X.iloc[:, 0].to_numpy() - 0.3 * X.iloc[:, 1].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={None: Numeric(), "z": Numeric()},
    ).fit(X, y)

    all_terms = model.plot_data(ci=None, show_density=False)
    none_term = model.plot_data(None, ci=None, show_density=False)

    assert [term["name"] for term in all_terms["terms"]] == [None, "z"]
    assert [term["name"] for term in none_term["terms"]] == [None]
    assert model.plot(ci=None, show_density=False) is not None
    assert model.plot(None, ci=None, show_density=False) is not None


@pytest.mark.parametrize("label", [0, False, None])
def test_scop_group_spec_uses_falsey_feature_name_exactly(label) -> None:
    spec = object()
    group = GroupSlice(
        name="display-name",
        start=0,
        end=1,
        feature_name=label,
        monotone_engine="scop",
    )

    assert scop_group_spec({label: spec}, group) is spec


@pytest.mark.parametrize("label", [0, None], ids=["zero", "none"])
def test_fixed_scop_lambda_uses_falsey_feature_label(label) -> None:
    x = np.linspace(0.0, 1.0, 80)
    X = pd.DataFrame({label: x})
    y = x**2
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            label: PSpline(
                n_knots=6,
                constraint=Constraint.fit.increasing,
                lambda_policy=LambdaPolicy.fixed(1.7),
            )
        },
    ).fit_reml(X, y)
    group = next(group for group in model._groups if group.feature_name == label)

    assert model._reml_lambdas[group.name] == pytest.approx(1.7)


@pytest.mark.parametrize("label", [0, None], ids=["zero", "none"])
def test_random_effect_reporting_uses_falsey_feature_label(label) -> None:
    levels = np.repeat(np.array(["a", "b", "c"], dtype=object), 20)
    X = pd.DataFrame({label: levels})
    y = np.repeat(np.array([-0.5, 0.0, 0.5]), 20)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            label: RandomEffect(lambda_policy=LambdaPolicy.fixed(1.2)),
        },
    ).fit_reml(X, y)

    report = model.random_effects(label)

    assert len(report.table) == 3
    assert set(report.table["level"]) == {"a", "b", "c"}


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
def test_auto_detect_refit_validates_new_complex_column_from_constructor_intent(
    entrypoint: str,
) -> None:
    x = np.linspace(-1.0, 1.0, 40)
    X = pd.DataFrame({"x": x})
    y = 0.5 + 0.3 * x
    model = SuperGLM(family="gaussian", selection_penalty=0.0, splines=[]).fit(X, y)
    result_before = model.result
    revision_before = model._fit_revision
    config_before = model._config
    specs_before = model._specs
    prediction_before = model.predict(X)

    X_bad = X.copy()
    X_bad["new_complex"] = np.linspace(0.0, 1.0, len(X)) + 1.0j

    with pytest.raises(ValueError, match="X column 'new_complex' must be real-valued"):
        _call_entrypoint(model, entrypoint, X_bad, y)

    assert model.result is result_before
    assert model._fit_revision == revision_before
    assert model._config is config_before
    assert model._specs is specs_before
    np.testing.assert_array_equal(model.predict(X), prediction_before)


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

    assert validated.X.native is X
    assert validated.X.backend == "pandas"
    assert validated.y.dtype == np.float64
    assert validated.sample_weight.dtype == np.float64
    assert validated.offset is not None and validated.offset.dtype == np.float64
    np.testing.assert_array_equal(y, [0, 1])
    np.testing.assert_array_equal(weights, [1, 2])
    np.testing.assert_array_equal(offset, [0, 1])


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_fit_entrypoints_accept_eager_polars_before_feature_build(entrypoint: str) -> None:
    X = pl.DataFrame({"x": np.linspace(-1.0, 1.0, 30)})
    y = 0.5 + 0.3 * X["x"].to_numpy()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.1 if entrypoint == "fit_path" else 0.0,
        features={"x": Numeric()},
    )

    result = _call_entrypoint(model, entrypoint, X, y)

    assert result is not None
    assert model.result.converged


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_fit_entrypoints_reject_lazy_polars_before_feature_build(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pl.DataFrame({"x": [0.0, 1.0]}).lazy()
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="eager.*collect"):
        _call_entrypoint(_model(), entrypoint, X, np.array([0.0, 1.0]))


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_fit_entrypoints_reject_missing_polars_columns_before_feature_build(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pl.DataFrame({"z": [0.0, 1.0]})
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="X is missing required columns.*x"):
        _call_entrypoint(_model(), entrypoint, X, np.array([0.0, 1.0]))


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_fit_entrypoints_reject_polars_row_count_mismatch_before_feature_build(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = pl.DataFrame({"x": [0.0, 1.0]})
    monkeypatch.setattr(Numeric, "build", _fail_if_feature_builds)

    with pytest.raises(ValueError, match="y must have length 2"):
        _call_entrypoint(_model(), entrypoint, X, np.array([0.0]))


@pytest.mark.parametrize(
    "X",
    [
        pd.DataFrame({"when": pd.date_range("2026-01-01", periods=3)}),
        pl.DataFrame({"when": pl.date_range(pl.date(2026, 1, 1), pl.date(2026, 1, 3), eager=True)}),
    ],
)
def test_auto_detection_rejects_unsupported_logical_dtype(X) -> None:
    model = SuperGLM(family="gaussian", selection_penalty=0.0, splines=[])

    with pytest.raises(ValueError, match="column 'when'.*unsupported.*[Dd]ate"):
        model.fit(X, np.array([1.0, 2.0, 3.0]))


def test_validate_fit_input_retains_the_native_frame_behind_the_adapter() -> None:
    X = pl.DataFrame({"x": [0.0, 1.0]})

    validated = validate_fit_input(
        X,
        np.array([0.0, 1.0]),
        None,
        None,
        family=Gaussian(),
        required_columns=("x",),
    )

    assert validated.X.native is X
    assert validated.X.backend == "polars"


def test_intercept_only_fit_accepts_polars_with_unused_columns() -> None:
    X = pl.DataFrame({"unused": np.arange(6)})
    y = np.arange(1.0, 7.0)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={},
    )

    model.fit(X, y)

    assert model._dm.shape == (len(X), 0)
    assert model.result.beta.shape == (0,)
