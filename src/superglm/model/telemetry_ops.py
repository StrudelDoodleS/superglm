"""Dependency-free training telemetry extraction helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from superglm import __version__


def training_telemetry(model) -> dict[str, Any]:
    """Return JSON-serializable telemetry for an already-fitted model.

    The payload is intentionally tracking-system agnostic. It contains plain
    Python objects that callers can send to MLflow, logs, files, model cards, or
    other governance systems.
    """
    result = _fitted_result(model)
    return _json_ready(
        {
            "package": {
                "name": "superglm",
                "version": __version__,
            },
            "model": _model_config(model),
            "rank_policy": _rank_policy_payload(result),
            "fit": _fit_payload(model, result),
            "fit_statistics": _fit_statistics_payload(model),
            "features": feature_schema(model),
            "edf": edf_by_term(model),
            "reml": reml_diagnostics(model),
        }
    )


def _rank_policy_payload(result) -> dict[str, Any]:
    """The rank policy this fit's zeros were decided under.

    ``RankPolicy.version`` is stamped onto every ``RankDecomposition`` and every
    ``RankInfo``, and before this was read by nothing -- so a governed fit could
    not be traced back to the rule that chose which coefficients are reported as
    exact zeros.  Recording it here is what the field is for.

    It is read off the fitted result rather than off the live shared policy, so
    a result carried across a version boundary reports the version that actually
    decided it rather than today's.
    """
    rank_info = getattr(result, "rank_info", None)
    version = getattr(rank_info, "policy_version", None)
    return {
        "version": int(version) if version is not None else None,
        "coordinate_space": getattr(rank_info, "coordinate_space", None),
    }


def reml_diagnostics(model) -> dict[str, Any]:
    """Return REML telemetry for an already-fitted model.

    If the model was not fit with REML, the returned payload has
    ``enabled=False`` and empty lambda/history fields.
    """
    reml = getattr(model, "_reml_result", None)
    lambdas = _model_lambdas(model)
    profile = getattr(model, "_reml_profile", None) or {}
    if reml is None:
        return _json_ready(
            {
                "enabled": False,
                "lambdas": {},
                "lambda_history": [],
                "profile": profile,
            }
        )
    return _json_ready(
        {
            "enabled": True,
            "lambdas": getattr(reml, "lambdas", None) or lambdas,
            "lambda_history": getattr(reml, "lambda_history", None) or [],
            "n_reml_iter": getattr(reml, "n_reml_iter", None),
            "converged": getattr(reml, "converged", None),
            "termination_reason": getattr(reml, "termination_reason", None),
            "objective": getattr(reml, "objective", None),
            "objective_history": getattr(reml, "objective_history", None),
            "inner_iter_history": getattr(reml, "inner_iter_history", None),
            "profile": profile,
        }
    )


def feature_schema(model) -> dict[str, Any]:
    """Return feature, interaction, group, and constraint telemetry."""
    specs = getattr(model, "_specs", {}) or {}
    interaction_specs = getattr(model, "_interaction_specs", {}) or {}
    feature_order = list(getattr(model, "_feature_order", []) or specs.keys())
    interaction_order = list(getattr(model, "_interaction_order", []) or interaction_specs.keys())

    features = {}
    constraints = {}
    for name in feature_order:
        payload = _feature_spec_payload(specs.get(name))
        features[name] = payload
        if payload["constraint"] is not None:
            constraints[name] = payload["constraint"]

    interactions = {
        name: _interaction_spec_payload(interaction_specs.get(name)) for name in interaction_order
    }

    groups = []
    for group in getattr(model, "_groups", []) or []:
        groups.append(
            {
                "name": str(getattr(group, "name", "")),
                "feature_name": str(getattr(group, "feature_name", "")),
                "subgroup_type": getattr(group, "subgroup_type", None),
                "start": getattr(group, "start", None),
                "end": getattr(group, "end", None),
                "size": _group_size(group),
            }
        )

    return _json_ready(
        {
            "feature_order": feature_order,
            "interaction_order": interaction_order,
            "features": features,
            "interactions": interactions,
            "constraints": constraints,
            "groups": groups,
        }
    )


def edf_by_term(model) -> dict[str, Any]:
    """Return effective degrees of freedom by solver group and feature term."""
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


def metrics_for_logging(model, *, prefix: str = "train") -> dict[str, float]:
    """Flatten common fitted-model telemetry into scalar metrics.

    This helper is independent of any tracking package. A caller can pass the
    returned dictionary directly to a logger such as ``mlflow.log_metrics``.
    """
    result = _fitted_result(model)
    metrics: dict[str, float] = {
        f"{prefix}.deviance": float(result.deviance),
        f"{prefix}.effective_df": float(result.effective_df),
        f"{prefix}.phi": float(result.phi),
        f"{prefix}.n_iter": float(result.n_iter),
        f"{prefix}.converged": 1.0 if bool(result.converged) else 0.0,
    }
    fit_stats = getattr(model, "_fit_stats", None)
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
                metrics[f"{prefix}.{attr}"] = float(value)
    for key, value in _model_lambdas(model).items():
        metrics[f"{prefix}.lambda.{_metric_part(key)}"] = float(value)
    for key, value in _group_edf(model).items():
        metrics[f"{prefix}.edf.{_metric_part(key)}"] = float(value)
    reml = getattr(model, "_reml_result", None)
    if reml is not None:
        metrics[f"{prefix}.reml.n_iter"] = float(reml.n_reml_iter)
        metrics[f"{prefix}.reml.converged"] = 1.0 if bool(reml.converged) else 0.0
        if getattr(reml, "objective", None) is not None:
            metrics[f"{prefix}.reml.objective"] = float(reml.objective)
    return metrics


def _model_config(model) -> dict[str, Any]:
    penalty = getattr(model, "penalty", None)
    return {
        "family": _class_or_value(
            getattr(model, "_distribution", None), getattr(model, "family", None)
        ),
        "link": _class_or_value(getattr(model, "_link", None), getattr(model, "link", None)),
        "penalty": type(penalty).__name__ if penalty is not None else None,
        "selection_penalty": getattr(penalty, "lambda1", None),
        "spline_penalty": getattr(model, "lambda2", None),
        "direct_solve": getattr(model, "_direct_solve", None),
        "discrete": bool(getattr(model, "_discrete", False)),
        "n_bins": getattr(model, "_n_bins", None),
        "tol": getattr(model, "_tol", None),
        "max_iter": getattr(model, "_max_iter", None),
        "convergence": getattr(model, "_convergence", None),
        "retain_fit_state": bool(getattr(model, "_retain_fit_state", True)),
    }


def _fit_payload(model, result) -> dict[str, Any]:
    fit_meta = dict(getattr(model, "_last_fit_meta", None) or {})
    return {
        "method": fit_meta.get("method"),
        "fit_meta": fit_meta,
        "deviance": float(result.deviance),
        "effective_df": float(result.effective_df),
        "phi": float(result.phi),
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "intercept": float(result.intercept),
    }


def _fit_statistics_payload(model) -> dict[str, Any]:
    fit_stats = getattr(model, "_fit_stats", None)
    if fit_stats is None:
        return {}
    return _json_ready(fit_stats)


def _fitted_result(model):
    try:
        result = getattr(model, "result")
    except RuntimeError as exc:
        raise RuntimeError("Training telemetry requires a fitted SuperGLM model.") from exc
    if result is None:
        result = getattr(model, "_result", None)
    if result is None:
        raise RuntimeError("Training telemetry requires a fitted SuperGLM model.")
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


def _feature_spec_payload(spec) -> dict[str, Any]:
    payload = {
        "class": type(spec).__name__ if spec is not None else None,
        "constraint": _constraint_payload(spec),
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
            payload["config"][attr] = getattr(spec, attr)
    return payload


def _interaction_spec_payload(spec) -> dict[str, Any]:
    payload = {
        "class": type(spec).__name__ if spec is not None else None,
        "parents": [],
        "config": {},
    }
    if spec is None:
        return payload
    parent_names = getattr(spec, "parent_names", None)
    if parent_names is not None:
        payload["parents"] = [str(parent) for parent in parent_names]
    else:
        parents = []
        for attr in ("feat1", "feat2", "name1", "name2", "spline_name", "cat_name"):
            value = getattr(spec, attr, None)
            if isinstance(value, str) and value not in parents:
                parents.append(value)
        payload["parents"] = parents
    for attr in ("constraint", "by", "tensor_kind"):
        if hasattr(spec, attr):
            payload["config"][attr] = getattr(spec, attr)
    return payload


def _constraint_payload(spec) -> dict[str, str] | None:
    if spec is None:
        return None
    kind = getattr(spec, "constraint_kind", None) or getattr(spec, "monotone", None)
    mode = getattr(spec, "constraint_mode", None) or getattr(spec, "monotone_mode", None)
    if kind is None:
        return None
    return {"kind": str(kind), "mode": str(mode)}


def _group_size(group) -> int:
    size = getattr(group, "size", None)
    if size is not None:
        return int(size)
    return int(getattr(group, "end", 0) - getattr(group, "start", 0))


def _class_or_value(obj: Any, fallback: Any = None) -> str | None:
    if obj is not None:
        return type(obj).__name__
    if fallback is None:
        return None
    return str(fallback)


def _json_ready(value: Any) -> Any:
    value = _jsonable(value)
    # Fail early if a new telemetry field is not JSON serializable.
    json.dumps(value)
    return value


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


def _metric_part(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return key or "metric"
