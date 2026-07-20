"""Backend-neutral dataframe compilation contracts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import polars as pl

from superglm import (
    Categorical,
    GroupLasso,
    NegativeBinomial,
    Numeric,
    Poisson,
    Spline,
    SuperGLM,
    Tweedie,
)
from superglm._frame import EagerFrame
from superglm.model.base import auto_detect, model_build_design_matrix
from superglm.model.input_validation import validate_fit_input
from superglm.profiling.tweedie import generate_tweedie_cpg


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


def _assert_fit_results_equal(
    pandas_model: SuperGLM,
    polars_model: SuperGLM,
    pandas_X: pd.DataFrame,
    polars_X: pl.DataFrame,
) -> None:
    np.testing.assert_allclose(
        polars_model.result.beta, pandas_model.result.beta, rtol=0.0, atol=0.0
    )
    assert polars_model.result.intercept == pandas_model.result.intercept
    assert polars_model.result.deviance == pandas_model.result.deviance
    assert polars_model.result.effective_df == pandas_model.result.effective_df
    assert polars_model.result.phi == pandas_model.result.phi
    assert polars_model.result.n_iter == pandas_model.result.n_iter
    assert polars_model.result.converged is pandas_model.result.converged
    np.testing.assert_allclose(
        polars_model.predict(polars_X),
        pandas_model.predict(pandas_X),
        rtol=0.0,
        atol=0.0,
    )
    pandas_log = pandas_model.result.iteration_log or ()
    polars_log = polars_model.result.iteration_log or ()
    assert [
        (entry.step_rejected, entry.trials_attempted, entry.accepted_alpha) for entry in polars_log
    ] == [
        (entry.step_rejected, entry.trials_attempted, entry.accepted_alpha) for entry in pandas_log
    ]


def _contains_boundary_adapter(value) -> bool:
    if isinstance(value, EagerFrame):
        return True
    if isinstance(value, dict):
        return any(
            _contains_boundary_adapter(key) or _contains_boundary_adapter(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_boundary_adapter(item) for item in value)
    return False


def test_dataframe_boundary_fit_is_backend_neutral_for_gaussian_and_poisson() -> None:
    rng = np.random.default_rng(20260720)
    n_rows = 180
    x = np.linspace(-1.5, 1.5, n_rows)
    z = rng.normal(size=n_rows)
    pandas_X = pd.DataFrame({"x": x, "z": z})
    polars_X = pl.DataFrame({"x": x, "z": z})

    for family, y in (
        ("gaussian", 1.2 + 0.3 * x - 0.15 * z),
        (Poisson(), rng.poisson(np.exp(0.2 + 0.15 * x - 0.1 * z)).astype(float)),
    ):

        def fitted(X) -> SuperGLM:
            return SuperGLM(
                family=family,
                selection_penalty=0.0,
                features={"x": Numeric(), "z": Numeric()},
            ).fit(X, y, record_diagnostics=True)

        pandas_model = fitted(pandas_X)
        polars_model = fitted(polars_X)
        _assert_fit_results_equal(pandas_model, polars_model, pandas_X, polars_X)


def test_dataframe_boundary_mixed_categorical_fit_is_backend_neutral() -> None:
    rng = np.random.default_rng(20260721)
    n_rows = 480
    numeric = rng.normal(size=n_rows)
    category = np.array([f"level_{index:02d}" for index in rng.integers(0, 24, n_rows)])
    y = 1.0 + 0.25 * numeric + 0.1 * (category == "level_03") + rng.normal(0.0, 0.05, n_rows)
    pandas_X = pd.DataFrame({"numeric": numeric, "category": category})
    polars_X = pl.DataFrame({"numeric": numeric, "category": category})

    def fitted(X) -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={
                "numeric": Numeric(),
                "category": Categorical(base="first"),
            },
        ).fit(X, y, record_diagnostics=True)

    pandas_model = fitted(pandas_X)
    polars_model = fitted(polars_X)
    _assert_fit_results_equal(pandas_model, polars_model, pandas_X, polars_X)


def test_dataframe_boundary_four_spline_discrete_fit_is_backend_neutral() -> None:
    rng = np.random.default_rng(20260722)
    n_rows = 240
    data = {f"x{index}": rng.uniform(-1.0, 1.0, n_rows) for index in range(4)}
    y = 1.0 + sum(
        (0.08 * (index + 1)) * np.sin(values) for index, values in enumerate(data.values())
    )
    pandas_X = pd.DataFrame(data)
    polars_X = pl.DataFrame(data)

    def fitted(X) -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            discrete=True,
            n_bins=32,
            features={name: Spline(n_knots=6, penalty="ssp") for name in data},
        ).fit(X, y, record_diagnostics=True)

    pandas_model = fitted(pandas_X)
    polars_model = fitted(polars_X)
    _assert_fit_results_equal(pandas_model, polars_model, pandas_X, polars_X)


def test_dataframe_boundary_reml_and_path_are_backend_neutral() -> None:
    rng = np.random.default_rng(20260723)
    n_rows = 180
    x = np.linspace(0.0, 1.0, n_rows)
    z = rng.normal(size=n_rows)
    y = 0.3 + np.sin(4.0 * x) + 0.1 * z
    pandas_X = pd.DataFrame({"x": x, "z": z})
    polars_X = pl.DataFrame({"x": x, "z": z})

    def reml_fitted(X) -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Spline(n_knots=7, penalty="ssp"), "z": Numeric()},
        ).fit_reml(X, y, max_reml_iter=4)

    pandas_reml = reml_fitted(pandas_X)
    polars_reml = reml_fitted(polars_X)
    _assert_fit_results_equal(pandas_reml, polars_reml, pandas_X, polars_X)
    assert polars_reml._reml_result.lambdas == pandas_reml._reml_result.lambdas
    assert polars_reml._reml_result.objective == pandas_reml._reml_result.objective
    assert polars_reml._reml_result.pirls_result.reml_hessian_rank == (
        pandas_reml._reml_result.pirls_result.reml_hessian_rank
    )

    def path_fitted(X):
        model = SuperGLM(
            family="gaussian",
            penalty=GroupLasso(lambda1=0.1),
            features={"x": Numeric(), "z": Numeric()},
        )
        path = model.fit_path(X, y, lambda_seq=np.array([0.2, 0.08, 0.02]))
        return model, path

    pandas_path_model, pandas_path = path_fitted(pandas_X)
    polars_path_model, polars_path = path_fitted(polars_X)
    for field in (
        "lambda_seq",
        "coef_path",
        "intercept_path",
        "deviance_path",
        "n_iter_path",
        "converged_path",
        "edf_path",
    ):
        np.testing.assert_allclose(
            getattr(polars_path, field),
            getattr(pandas_path, field),
            rtol=0.0,
            atol=0.0,
        )
    _assert_fit_results_equal(
        pandas_path_model,
        polars_path_model,
        pandas_X,
        polars_X,
    )


def test_dataframe_boundary_tweedie_and_nb_profiles_are_backend_neutral() -> None:
    rng = np.random.default_rng(20260724)
    n_rows = 180
    x = rng.normal(size=n_rows)
    mu = np.exp(0.4 + 0.2 * x)
    pandas_X = pd.DataFrame({"x": x})
    polars_X = pl.DataFrame({"x": x})

    tweedie_y = generate_tweedie_cpg(n_rows, mu=mu, phi=0.9, p=1.5, rng=rng)

    def tweedie_profiled(X):
        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        events = []
        trace_rows = []
        result = model.estimate_p(
            X,
            tweedie_y,
            p_bounds=(1.3, 1.7),
            xatol=1e-4,
            progress_callback=lambda phase, payload: events.append((phase, payload)),
            trace_callback=trace_rows.append,
        )
        return model, result, events, trace_rows

    pandas_tweedie, pandas_p, pandas_events, pandas_trace = tweedie_profiled(pandas_X)
    polars_tweedie, polars_p, polars_events, polars_trace = tweedie_profiled(polars_X)
    for field in (
        "p_hat",
        "phi_hat",
        "nll",
        "n_evaluations",
        "converged",
        "phi_used_fallback",
        "phi_fallback_reason",
        "density_method",
        "density_exact",
    ):
        assert getattr(polars_p, field) == getattr(pandas_p, field)
    pd.testing.assert_frame_equal(polars_p.search_trace, pandas_p.search_trace, check_exact=True)
    assert [phase for phase, _ in polars_events] == [phase for phase, _ in pandas_events]
    assert [type(row) for row in polars_trace] == [type(row) for row in pandas_trace]
    assert not _contains_boundary_adapter(polars_events)
    assert not _contains_boundary_adapter(polars_trace)
    _assert_fit_results_equal(pandas_tweedie, polars_tweedie, pandas_X, polars_X)

    theta = 2.5
    nb_y = rng.poisson(rng.gamma(shape=theta, scale=mu / theta)).astype(float)

    def nb_profiled(X):
        model = SuperGLM(
            family=NegativeBinomial(theta="auto"),
            selection_penalty=0.0,
            features={"x": Numeric()},
        )
        result = model.estimate_theta(X, nb_y, maxiter=4)
        return model, result

    pandas_nb, pandas_theta = nb_profiled(pandas_X)
    polars_nb, polars_theta = nb_profiled(polars_X)
    assert polars_theta.theta_hat == pandas_theta.theta_hat
    assert polars_theta.nll == pandas_theta.nll
    assert polars_theta.n_evaluations == pandas_theta.n_evaluations
    assert polars_theta.converged is pandas_theta.converged
    _assert_fit_results_equal(pandas_nb, polars_nb, pandas_X, polars_X)


def test_dataframe_boundary_failed_cross_backend_refit_rolls_back_atomically() -> None:
    x = np.linspace(-1.0, 1.0, 60)
    y = 0.5 + 0.3 * x
    pandas_X = pd.DataFrame({"x": x})
    polars_X = pl.DataFrame({"x": x})

    for initial_X, failing_X in ((pandas_X, polars_X), (polars_X, pandas_X)):
        model = SuperGLM(
            family="gaussian",
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(initial_X, y)
        result_before = model.result
        state_before = model._fit_state
        revision_before = model._fit_revision
        predictions_before = model.predict(initial_X)
        profile_sentinel = object()
        model._tweedie_profile_result = profile_sentinel

        with np.testing.assert_raises_regex(ValueError, "y must contain only finite"):
            model.fit(failing_X, np.full_like(y, np.nan))

        assert model.result is result_before
        assert model._fit_state is state_before
        assert model._fit_revision == revision_before
        assert model._tweedie_profile_result is profile_sentinel
        np.testing.assert_array_equal(model.predict(initial_X), predictions_before)
