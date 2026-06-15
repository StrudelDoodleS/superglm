"""MLflow logging helpers for SuperGLM governance runs.

This module is intentionally optional: importing :mod:`superglm.mlflow` does
not import MLflow. The dependency is loaded only when ``log_model_run()`` is
called.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from superglm import __version__

_ARTIFACT_FILES = (
    "model_config.json",
    "feature_schema.json",
    "lambdas.json",
    "edf_by_term.json",
    "validation_metrics.json",
)


def log_model_run(
    model,
    *,
    experiment_name: str | None = None,
    run_name: str | None = None,
    nested: bool = False,
    run_type: str = "model_fit",
    tags: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    validation_metrics: Mapping[str, Any] | None = None,
    git_sha: str | None = None,
) -> str | None:
    """Log an already-fitted SuperGLM model run to MLflow.

    The helper records governance metadata, scalar fit metrics, and JSON
    artifacts. It does not fit, refit, or score the model. Pass validation,
    calibration, lift, or Gini metrics through ``validation_metrics`` when those
    have been computed by the caller.

    Parameters
    ----------
    model : SuperGLM-like object
        Fitted SuperGLM instance.
    experiment_name : str, optional
        MLflow experiment name to set before starting the run.
    run_name : str, optional
        MLflow run name.
    nested : bool
        Passed through to ``mlflow.start_run(nested=...)`` for candidate
        interaction searches or other parent/child workflows.
    run_type : str
        Governance run type tag, e.g. ``"model_fit"`` or
        ``"candidate_interaction"``.
    tags, params, metrics : mapping, optional
        Additional MLflow tags, params, and metrics to log.
    validation_metrics : mapping, optional
        Caller-computed validation metrics. Numeric leaves are logged as
        metrics under ``validation.*`` and the full mapping is logged as
        ``validation_metrics.json``.
    git_sha : str, optional
        Explicit source revision. If omitted, ``SUPERGLM_GIT_SHA`` or local
        ``git rev-parse HEAD`` is used when available.

    Returns
    -------
    str or None
        MLflow run id when available from the active run object.
    """
    result = _fitted_result(model)
    mlflow = _load_mlflow()
    git_sha = _resolve_git_sha(git_sha)
    validation_payload = _jsonable(validation_metrics or {})

    model_config = build_model_config(model, git_sha=git_sha)
    feature_schema = build_feature_schema(model)
    lambdas = _model_lambdas(model)
    edf_by_term = build_edf_by_term(model)

    run_tags = _run_tags(model, run_type=run_type, git_sha=git_sha)
    if tags:
        run_tags.update({str(k): _stringify_tag(v) for k, v in tags.items()})

    run_params = _run_params(model, model_config, feature_schema)
    if params:
        run_params.update({str(k): _stringify_param(v) for k, v in params.items()})

    run_metrics = _run_metrics(model, result, lambdas, edf_by_term)
    run_metrics.update(_flatten_numeric(validation_payload, prefix="validation"))
    if metrics:
        run_metrics.update(_flatten_numeric(_jsonable(metrics), prefix="custom"))

    artifacts = {
        "model_config.json": model_config,
        "feature_schema.json": feature_schema,
        "lambdas.json": lambdas,
        "edf_by_term.json": edf_by_term,
        "validation_metrics.json": validation_payload,
    }

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags(run_tags)
        mlflow.log_params(run_params)
        mlflow.log_metrics(run_metrics)
        for artifact_file in _ARTIFACT_FILES:
            _log_json_artifact(mlflow, artifacts[artifact_file], artifact_file)
        return getattr(getattr(run, "info", None), "run_id", None)


def build_model_config(model, *, git_sha: str | None = None) -> dict[str, Any]:
    """Return JSON-serializable model configuration for audit artifacts."""
    result = _fitted_result(model)
    fit_meta = dict(getattr(model, "_last_fit_meta", None) or {})
    reml = getattr(model, "_reml_result", None)
    profile = dict(getattr(model, "_reml_profile", None) or {})
    fit_stats = getattr(model, "_fit_stats", None)

    return {
        "package": {
            "name": "superglm",
            "version": __version__,
            "git_sha": git_sha,
        },
        "model": {
            "family": _class_or_value(
                getattr(model, "_distribution", None), getattr(model, "family", None)
            ),
            "link": _class_or_value(getattr(model, "_link", None), getattr(model, "link", None)),
            "penalty": type(getattr(model, "penalty", None)).__name__,
            "selection_penalty": _jsonable(
                getattr(getattr(model, "penalty", None), "lambda1", None)
            ),
            "spline_penalty": _jsonable(getattr(model, "lambda2", None)),
            "direct_solve": getattr(model, "_direct_solve", None),
            "discrete": bool(getattr(model, "_discrete", False)),
            "n_bins": _jsonable(getattr(model, "_n_bins", None)),
            "retain_fit_state": bool(getattr(model, "_retain_fit_state", True)),
        },
        "fit": {
            "method": fit_meta.get("method"),
            "fit_meta": _jsonable(fit_meta),
            "tol": _jsonable(getattr(model, "_tol", None)),
            "max_iter": _jsonable(getattr(model, "_max_iter", None)),
            "convergence": getattr(model, "_convergence", None),
            "deviance": float(result.deviance),
            "effective_df": float(result.effective_df),
            "phi": float(result.phi),
            "irls_iterations": int(result.n_iter),
            "converged": bool(result.converged),
        },
        "reml": {
            "enabled": reml is not None,
            "converged": None if reml is None else bool(reml.converged),
            "iterations": None if reml is None else int(reml.n_reml_iter),
            "objective": _jsonable(getattr(reml, "objective", None)),
            "profile": _jsonable(profile),
        },
        "fit_statistics": _jsonable(fit_stats),
    }


def build_feature_schema(model) -> dict[str, Any]:
    """Return JSON-serializable feature, interaction, and constraint metadata."""
    specs = getattr(model, "_specs", {}) or {}
    interaction_specs = getattr(model, "_interaction_specs", {}) or {}
    feature_order = list(getattr(model, "_feature_order", []) or specs.keys())
    interaction_order = list(getattr(model, "_interaction_order", []) or interaction_specs.keys())

    features = {}
    constraints = {}
    for name in feature_order:
        spec = specs.get(name)
        feature_row = _feature_spec_payload(spec)
        features[name] = feature_row
        constraint = feature_row.get("constraint")
        if constraint is not None:
            constraints[name] = constraint

    interactions = {
        name: _interaction_spec_payload(interaction_specs.get(name)) for name in interaction_order
    }

    return {
        "feature_order": feature_order,
        "interaction_order": interaction_order,
        "features": features,
        "interactions": interactions,
        "constraints": constraints,
        "groups": [
            {
                "name": str(getattr(group, "name", "")),
                "feature_name": str(getattr(group, "feature_name", "")),
                "subgroup_type": getattr(group, "subgroup_type", None),
                "size": int(_group_size(group)),
            }
            for group in getattr(model, "_groups", []) or []
        ],
    }


def build_edf_by_term(model) -> dict[str, Any]:
    """Return EDF by solver group and by feature term."""
    group_edf = _group_edf(model)
    by_feature: dict[str, float] = {}
    for group in getattr(model, "_groups", []) or []:
        group_name = str(getattr(group, "name", ""))
        feature_name = str(getattr(group, "feature_name", group_name))
        by_feature[feature_name] = by_feature.get(feature_name, 0.0) + float(
            group_edf.get(group_name, 0.0)
        )
    return {
        "by_group": {str(k): float(v) for k, v in group_edf.items()},
        "by_feature": by_feature,
        "total": float(sum(float(v) for v in group_edf.values())),
    }


def _load_mlflow():
    try:
        return importlib.import_module("mlflow")
    except ImportError as exc:
        raise ImportError(
            "MLflow logging requires the optional dependency 'mlflow'. "
            "Install it with `pip install mlflow` or add it to your environment."
        ) from exc


def _fitted_result(model):
    try:
        result = getattr(model, "result")
    except RuntimeError as exc:
        raise RuntimeError("log_model_run() requires a fitted SuperGLM model.") from exc
    if result is None:
        result = getattr(model, "_result", None)
    if result is None:
        raise RuntimeError("log_model_run() requires a fitted SuperGLM model.")
    return result


def _model_lambdas(model) -> dict[str, float]:
    lambdas = getattr(model, "_reml_lambdas", None)
    if lambdas is None:
        reml = getattr(model, "_reml_result", None)
        lambdas = getattr(reml, "lambdas", None)
    if lambdas is None:
        lam2 = getattr(model, "lambda2", None)
        return {} if lam2 is None else {"lambda2": float(lam2)}
    return {str(k): float(v) for k, v in lambdas.items()}


def _group_edf(model) -> dict[str, float]:
    try:
        group_edf = getattr(model, "_group_edf", None)
    except Exception:
        group_edf = None
    if not group_edf:
        return {}
    return {str(k): float(v) for k, v in group_edf.items()}


def _run_tags(model, *, run_type: str, git_sha: str | None) -> dict[str, str]:
    tags = {
        "superglm.run_type": str(run_type),
        "superglm.version": __version__,
        "superglm.fit_method": str((getattr(model, "_last_fit_meta", None) or {}).get("method")),
        "superglm.discrete": str(bool(getattr(model, "_discrete", False))),
    }
    if git_sha:
        tags["superglm.git_sha"] = git_sha
    return tags


def _run_params(
    model, model_config: dict[str, Any], feature_schema: dict[str, Any]
) -> dict[str, str]:
    features = feature_schema["feature_order"]
    interactions = feature_schema["interaction_order"]
    constraints = feature_schema["constraints"]
    fit = model_config["fit"]
    model_section = model_config["model"]
    params = {
        "superglm.version": __version__,
        "family": _stringify_param(model_section["family"]),
        "link": _stringify_param(model_section["link"]),
        "penalty": _stringify_param(model_section["penalty"]),
        "selection_penalty": _stringify_param(model_section["selection_penalty"]),
        "spline_penalty": _stringify_param(model_section["spline_penalty"]),
        "fit.method": _stringify_param(fit["method"]),
        "tol": _stringify_param(fit["tol"]),
        "max_iter": _stringify_param(fit["max_iter"]),
        "convergence": _stringify_param(fit["convergence"]),
        "direct_solve": _stringify_param(model_section["direct_solve"]),
        "discrete": _stringify_param(model_section["discrete"]),
        "n_bins": _stringify_param(model_section["n_bins"]),
        "feature.count": str(len(features)),
        "interaction.count": str(len(interactions)),
        "constraint.count": str(len(constraints)),
        "features": json.dumps(features),
        "interactions": json.dumps(interactions),
        "constraints": json.dumps(constraints, sort_keys=True),
    }
    profile = model_config["reml"]["profile"] or {}
    for key in ("pirls_tol", "reml_tol", "max_reml_iter", "max_pirls_iter", "interaction_mode"):
        if key in profile:
            params[key] = _stringify_param(profile[key])
    return params


def _run_metrics(
    model,
    result,
    lambdas: dict[str, float],
    edf_by_term: dict[str, Any],
) -> dict[str, float]:
    reml = getattr(model, "_reml_result", None)
    fit_stats = getattr(model, "_fit_stats", None)
    profile = getattr(model, "_reml_profile", None) or {}
    metrics = {
        "deviance": float(result.deviance),
        "effective_df": float(result.effective_df),
        "phi": float(result.phi),
        "converged": 1.0 if bool(result.converged) else 0.0,
        "irls_iterations": float(result.n_iter),
    }
    if reml is not None:
        metrics["reml_iterations"] = float(reml.n_reml_iter)
        metrics["reml_converged"] = 1.0 if bool(reml.converged) else 0.0
        if getattr(reml, "objective", None) is not None:
            metrics["reml_objective"] = float(reml.objective)
    if fit_stats is not None:
        for attr in (
            "log_likelihood",
            "null_log_likelihood",
            "null_deviance",
            "explained_deviance",
            "pearson_chi2",
            "n_obs",
        ):
            value = getattr(fit_stats, attr, None)
            if _is_number(value):
                metrics[attr] = float(value)
    for key, value in profile.items():
        if _is_number(value):
            metrics[_metric_key("profile", key)] = float(value)
    for key, value in lambdas.items():
        metrics[_metric_key("lambda", key)] = float(value)
    for key, value in edf_by_term.get("by_group", {}).items():
        metrics[_metric_key("edf", key)] = float(value)
    return metrics


def _feature_spec_payload(spec) -> dict[str, Any]:
    constraint = _constraint_payload(spec)
    payload = {
        "class": type(spec).__name__ if spec is not None else None,
        "constraint": constraint,
        "config": {},
    }
    if spec is None:
        return payload
    for attr in (
        "kind",
        "base",
        "select",
        "n_knots",
        "degree",
        "_n_knots",
        "_degree",
        "constraint_kind",
        "constraint_mode",
    ):
        if hasattr(spec, attr):
            payload["config"][attr] = _jsonable(getattr(spec, attr))
    return payload


def _interaction_spec_payload(spec) -> dict[str, Any]:
    payload = {
        "class": type(spec).__name__ if spec is not None else None,
        "parents": [],
        "config": {},
    }
    if spec is None:
        return payload
    parents = []
    for attr in ("feat1", "feat2", "name1", "name2", "spline_name", "cat_name"):
        value = getattr(spec, attr, None)
        if isinstance(value, str) and value not in parents:
            parents.append(value)
    payload["parents"] = parents
    for attr in ("constraint", "by", "tensor_kind", "marginal1", "marginal2"):
        if hasattr(spec, attr):
            payload["config"][attr] = _jsonable(getattr(spec, attr))
    return payload


def _constraint_payload(spec) -> dict[str, str] | None:
    if spec is None:
        return None
    kind = getattr(spec, "constraint_kind", None) or getattr(spec, "monotone", None)
    mode = getattr(spec, "constraint_mode", None) or getattr(spec, "monotone_mode", None)
    if kind is None:
        return None
    return {"kind": str(kind), "mode": str(mode)}


def _resolve_git_sha(git_sha: str | None) -> str | None:
    if git_sha:
        return str(git_sha)
    env_sha = os.environ.get("SUPERGLM_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def _log_json_artifact(mlflow, payload: dict[str, Any], artifact_file: str) -> None:
    if hasattr(mlflow, "log_dict"):
        mlflow.log_dict(payload, artifact_file)
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / artifact_file
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        mlflow.log_artifact(str(path))


def _flatten_numeric(value: Any, *, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}

    def visit(obj: Any, parts: list[str]) -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                visit(child, [*parts, str(key)])
            return
        if _is_number(obj):
            out[_metric_key(*parts)] = float(obj)

    visit(value, [prefix])
    return out


def _metric_key(*parts: str) -> str:
    raw = ".".join(str(part) for part in parts if str(part))
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return key or "metric"


def _class_or_value(obj: Any, fallback: Any = None) -> str | None:
    if obj is not None:
        return type(obj).__name__
    if fallback is None:
        return None
    return str(fallback)


def _group_size(group) -> int:
    size = getattr(group, "size", None)
    if size is not None:
        return int(size)
    return int(getattr(group, "end", 0) - getattr(group, "start", 0))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list | set):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict(orient="list"))
        except TypeError:
            return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            str(k): _jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_") and k != "figure"
        }
    return repr(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating | bool) and not isinstance(
        value, complex
    )


def _stringify_param(value: Any) -> str:
    value = _jsonable(value)
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _stringify_tag(value: Any) -> str:
    return _stringify_param(value)


__all__ = [
    "build_edf_by_term",
    "build_feature_schema",
    "build_model_config",
    "log_model_run",
]
