from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, SuperGLM
from superglm.model import PathResult


def _dummy_model():
    result = SimpleNamespace(
        beta=np.array([0.1, -0.2, 0.3]),
        intercept=-1.25,
        n_iter=4,
        deviance=123.4,
        converged=True,
        phi=1.0,
        effective_df=2.75,
        iteration_log=None,
    )
    fit_stats = SimpleNamespace(
        log_likelihood=-61.2,
        null_log_likelihood=-75.0,
        null_deviance=150.0,
        explained_deviance=0.177,
        pearson_chi2=98.5,
        n_obs=50,
    )
    groups = [
        SimpleNamespace(
            name="DrivAge",
            feature_name="DrivAge",
            subgroup_type="spline",
            start=0,
            end=2,
            size=2,
            sl=slice(0, 2),
        ),
        SimpleNamespace(
            name="DrivAge:Area",
            feature_name="DrivAge:Area",
            subgroup_type="spline_categorical",
            start=2,
            end=3,
            size=1,
            sl=slice(2, 3),
        ),
    ]
    specs = {
        "DrivAge": SimpleNamespace(
            constraint_kind="increasing",
            constraint_mode="fit",
            select=False,
            n_knots=10,
        ),
        "Area": SimpleNamespace(base="A"),
    }
    return SimpleNamespace(
        family="poisson",
        link="log",
        penalty=SimpleNamespace(lambda1=0.0),
        lambda2=0.1,
        _distribution=SimpleNamespace(),
        _link=SimpleNamespace(),
        _direct_solve="auto",
        _discrete=True,
        _n_bins={"DrivAge": 64},
        _tol=1e-6,
        _max_iter=100,
        _convergence="deviance",
        _retain_fit_state=False,
        _feature_order=["DrivAge", "Area"],
        _interaction_order=["DrivAge:Area"],
        _pending_interactions=[],
        _specs=specs,
        _interaction_specs={"DrivAge:Area": SimpleNamespace(feat1="DrivAge", feat2="Area")},
        _groups=groups,
        _group_edf={"DrivAge": 1.5, "DrivAge:Area": 1.25},
        _reml_lambdas={"DrivAge": 2.0, "DrivAge:Area": 3.5},
        _reml_result=SimpleNamespace(
            lambdas={"DrivAge": 2.0, "DrivAge:Area": 3.5},
            n_reml_iter=5,
            converged=True,
            objective=42.0,
            lambda_history=[
                {"DrivAge": 1.0, "DrivAge:Area": 1.0},
                {"DrivAge": 2.0, "DrivAge:Area": 3.5},
            ],
            objective_history=[45.0, 42.0],
            inner_iter_history=[3, 2],
        ),
        _reml_profile={
            "irls_calls": 7,
            "irls_iters": 12,
            "interaction_mode": "fast_candidate",
            "effective_max_reml_iter": 3,
        },
        _last_fit_meta={"method": "fit_reml", "discrete": True},
        _fit_stats=fit_stats,
        _result=result,
        result=result,
    )


def test_superglm_does_not_expose_mlflow_submodule():
    assert importlib.util.find_spec("superglm.mlflow") is None


def test_training_telemetry_is_plain_json_ready_payload():
    from superglm.model import telemetry_ops

    assert callable(getattr(SuperGLM, "training_telemetry"))
    assert callable(getattr(SuperGLM, "reml_diagnostics"))

    telemetry = telemetry_ops.training_telemetry(_dummy_model())

    assert telemetry["fit"]["method"] == "fit_reml"
    assert telemetry["fit"]["deviance"] == pytest.approx(123.4)
    assert telemetry["fit"]["effective_df"] == pytest.approx(2.75)
    assert telemetry["fit"]["n_iter"] == 4
    assert telemetry["fit"]["converged"] is True
    assert telemetry["fit_statistics"]["null_deviance"] == pytest.approx(150.0)
    assert telemetry["features"]["feature_order"] == ["DrivAge", "Area"]
    assert telemetry["features"]["constraints"]["DrivAge"] == {
        "kind": "increasing",
        "mode": "fit",
    }
    assert telemetry["edf"]["by_feature"]["DrivAge:Area"] == pytest.approx(1.25)
    assert telemetry["reml"]["lambdas"] == {"DrivAge": 2.0, "DrivAge:Area": 3.5}
    assert telemetry["reml"]["lambda_history"][1]["DrivAge:Area"] == pytest.approx(3.5)
    assert telemetry["reml"]["profile"]["interaction_mode"] == "fast_candidate"

    json.dumps(telemetry)


def test_reml_diagnostics_returns_reml_only_payload():
    from superglm.model import telemetry_ops

    diagnostics = telemetry_ops.reml_diagnostics(_dummy_model())

    assert diagnostics["enabled"] is True
    assert diagnostics["n_reml_iter"] == 5
    assert diagnostics["converged"] is True
    assert diagnostics["objective"] == pytest.approx(42.0)
    assert diagnostics["lambdas"]["DrivAge"] == pytest.approx(2.0)
    assert diagnostics["objective_history"] == [45.0, 42.0]
    assert diagnostics["inner_iter_history"] == [3, 2]


def test_fitted_model_training_telemetry_method_smoke():
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 20)})
    y = np.array([0, 1] * 10, dtype=float)
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric()},
        selection_penalty=0.0,
        spline_penalty=0.0,
    )

    model.fit(X, y)
    telemetry = model.training_telemetry()

    assert telemetry["fit"]["method"] == "fit"
    assert telemetry["fit"]["deviance"] == pytest.approx(model.result.deviance)
    assert telemetry["features"]["feature_order"] == ["x"]
    json.dumps(telemetry)


def test_path_result_to_frame_and_telemetry():
    result = PathResult(
        lambda_seq=np.array([1.0, 0.1]),
        coef_path=np.array([[0.0, 0.0], [1.0, 2.0]]),
        intercept_path=np.array([-2.0, -1.0]),
        deviance_path=np.array([12.0, 9.5]),
        n_iter_path=np.array([3, 4]),
        converged_path=np.array([True, False]),
        edf_path=np.array([1.25, 1.75]),
    )

    frame = result.to_frame()
    assert list(frame.columns) == ["step", "lambda", "deviance", "n_iter", "converged", "edf"]
    assert frame.loc[1, "lambda"] == pytest.approx(0.1)
    assert frame.loc[1, "converged"] is False

    telemetry = result.to_telemetry()
    assert telemetry["lambda_seq"] == [1.0, 0.1]
    assert telemetry["deviance_path"] == [12.0, 9.5]
    assert telemetry["edf_path"] == [1.25, 1.75]
    json.dumps(telemetry)
