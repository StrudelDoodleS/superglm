from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class _FakeRun:
    def __init__(self, run_id: str):
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMLflow:
    def __init__(self):
        self.experiment_name = None
        self.start_runs = []
        self.params = {}
        self.metrics = {}
        self.tags = {}
        self.artifacts = {}

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, *, run_name=None, nested=False):
        self.start_runs.append({"run_name": run_name, "nested": nested})
        return _FakeRun(f"run-{len(self.start_runs)}")

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_dict(self, payload, artifact_file):
        self.artifacts[artifact_file] = payload


def _dummy_model():
    result = SimpleNamespace(
        beta=np.array([0.1, -0.2, 0.3]),
        intercept=-1.25,
        n_iter=4,
        deviance=123.4,
        converged=True,
        phi=1.0,
        effective_df=2.75,
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
        ),
        _reml_profile={"irls_calls": 7, "irls_iters": 12, "total_s": 1.25},
        _last_fit_meta={"method": "fit_reml", "discrete": True},
        _fit_stats=fit_stats,
        _result=result,
        result=result,
    )


def test_log_model_run_logs_governance_payload(monkeypatch):
    import superglm.mlflow as sg_mlflow

    fake = _FakeMLflow()
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)

    run_id = sg_mlflow.log_model_run(
        _dummy_model(),
        experiment_name="Pricing Governance",
        run_name="candidate-age-area",
        nested=True,
        run_type="candidate_interaction",
        git_sha="abc123",
        tags={"approval.stage": "development"},
        validation_metrics={
            "gini_model": 0.31,
            "lift": {"top_decile": 1.42},
            "calibration": {"mean_actual_to_expected": 1.01},
        },
    )

    assert run_id == "run-1"
    assert fake.experiment_name == "Pricing Governance"
    assert fake.start_runs == [{"run_name": "candidate-age-area", "nested": True}]

    assert fake.tags["superglm.run_type"] == "candidate_interaction"
    assert fake.tags["superglm.git_sha"] == "abc123"
    assert fake.tags["approval.stage"] == "development"

    assert fake.params["fit.method"] == "fit_reml"
    assert fake.params["discrete"] == "True"
    assert fake.params["features"] == '["DrivAge", "Area"]'
    assert fake.params["interactions"] == '["DrivAge:Area"]'
    assert fake.params["constraint.count"] == "1"

    assert fake.metrics["deviance"] == pytest.approx(123.4)
    assert fake.metrics["effective_df"] == pytest.approx(2.75)
    assert fake.metrics["converged"] == pytest.approx(1.0)
    assert fake.metrics["irls_iterations"] == pytest.approx(4.0)
    assert fake.metrics["reml_iterations"] == pytest.approx(5.0)
    assert fake.metrics["lambda.DrivAge"] == pytest.approx(2.0)
    assert fake.metrics["edf.DrivAge_Area"] == pytest.approx(1.25)
    assert fake.metrics["validation.gini_model"] == pytest.approx(0.31)
    assert fake.metrics["validation.lift.top_decile"] == pytest.approx(1.42)
    assert fake.metrics["validation.calibration.mean_actual_to_expected"] == pytest.approx(1.01)

    assert set(fake.artifacts) == {
        "model_config.json",
        "feature_schema.json",
        "lambdas.json",
        "edf_by_term.json",
        "validation_metrics.json",
    }
    assert fake.artifacts["model_config.json"]["fit"]["method"] == "fit_reml"
    assert fake.artifacts["feature_schema.json"]["features"]["DrivAge"]["constraint"] == {
        "kind": "increasing",
        "mode": "fit",
    }
    assert fake.artifacts["lambdas.json"] == {"DrivAge": 2.0, "DrivAge:Area": 3.5}
    assert fake.artifacts["edf_by_term.json"]["by_feature"]["DrivAge:Area"] == 1.25


def test_log_model_run_has_clear_error_when_mlflow_missing(monkeypatch):
    import superglm.mlflow as sg_mlflow

    def fail_import(name):
        if name == "mlflow":
            raise ImportError("not installed")
        raise AssertionError(name)

    monkeypatch.setattr(sg_mlflow.importlib, "import_module", fail_import)

    with pytest.raises(ImportError, match="MLflow logging requires the optional dependency"):
        sg_mlflow.log_model_run(_dummy_model())
