"""Model metric payloads for the editor."""

from __future__ import annotations

from typing import Any

import numpy as np

from superglm._utils import _explained_deviance
from superglm.distributions import weighted_log_likelihood
from superglm.editor.evaluation import (
    EvaluationDataset,
    _validate_evaluation_weights,
    named_metrics_dataset,
)
from superglm.solvers.dispersion import dispersion_likelihood_size, model_weight_semantics

METRIC_LABELS = {
    "deviance": "Deviance",
    "aic": "AIC",
    "aicc": "AICc",
    "bic": "BIC",
    "log_likelihood": "Log Likelihood",
    "explained_deviance": "Explained Deviance",
    "pearson_chi2": "Pearson Chi2",
    "effective_df": "Total EDF",
}


def metric_comparison_payload(
    metric: str,
    dataset: EvaluationDataset,
    original_metrics: dict[str, float],
    edited_metrics: dict[str, float],
    *,
    model_revision: int,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """Assemble one comparison payload from already-computed scalar dictionaries."""
    selected_metric = metric if metric in METRIC_LABELS else "deviance"
    original = original_metrics[selected_metric]
    edited = edited_metrics[selected_metric]
    return {
        "available": True,
        "model_revision": int(model_revision),
        "request_sequence": request_sequence,
        "metric": selected_metric,
        "label": METRIC_LABELS[selected_metric],
        "dataset": dataset.name,
        "dataset_label": dataset.label,
        "n_obs": dataset.n_obs,
        "original": original,
        "edited": edited,
        "delta": edited - original,
        "metrics": {"original": original_metrics, "edited": edited_metrics},
    }


def metrics_payload(
    session,
    metric: str,
    *,
    source: str = "in_force",
    dataset: str | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    # Metrics compare the immutable original fit with the in-force editor model
    # on the retained training frame. Manual coefficient-edit metrics are
    # prediction diagnostics; structural refits are fitted-model metrics.
    metric = metric if metric in METRIC_LABELS else "deviance"
    revision = session.model_revision if model_revision is None else int(model_revision)
    reference_model = getattr(session, "reference_model", session.model)
    if reference_model is None:
        return {
            "available": False,
            "model_revision": revision,
            "request_sequence": request_sequence,
            "metric": metric,
            "error": "No source model is attached.",
        }
    eval_dataset = named_metrics_dataset(session, dataset)
    if eval_dataset is None:
        return {
            "available": False,
            "model_revision": revision,
            "request_sequence": request_sequence,
            "metric": metric,
            "error": "No evaluation data is available.",
        }

    selected_model = reference_model if source == "original" else session.to_model()
    original_metrics = compute_dataset_metrics(reference_model, eval_dataset)
    edited_metrics = compute_dataset_metrics(selected_model, eval_dataset)
    return metric_comparison_payload(
        metric,
        eval_dataset,
        original_metrics,
        edited_metrics,
        model_revision=revision,
        request_sequence=request_sequence,
    )


def compute_dataset_metrics(model, dataset: EvaluationDataset) -> dict[str, float]:
    validated_weights = _validate_evaluation_weights(
        dataset.sample_weight,
        dataset.n_obs,
        family=model._distribution,
        weight_semantics=model_weight_semantics(model),
    )
    fit_artifacts = _fit_artifact_metrics(model, dataset)
    if fit_artifacts is not None:
        return fit_artifacts
    weights = validated_weights
    if weights is None:
        weights = np.ones(dataset.n_obs, dtype=np.float64)
    return _compute_metrics(model, dataset.X, dataset.y, weights, dataset.offset)


def _same_fit_dataset(model, dataset: EvaluationDataset) -> bool:
    fit_weight_ref = getattr(model, "_fit_sample_weight_ref", None)
    fit_weights = getattr(model, "_fit_weights", None)
    fit_offset_ref = getattr(model, "_fit_offset_ref", None)
    fit_offset = getattr(model, "_fit_offset", None)
    weights_match = dataset.sample_weight is fit_weight_ref or dataset.sample_weight is fit_weights
    offset_matches = dataset.offset is fit_offset_ref or dataset.offset is fit_offset
    return (
        dataset.X is getattr(model, "_fit_X_ref", None)
        and dataset.y is getattr(model, "_fit_y_ref", None)
        and weights_match
        and offset_matches
    )


def _fit_artifact_metrics(model, dataset: EvaluationDataset) -> dict[str, float] | None:
    fit_stats = getattr(model, "_fit_stats", None)
    if fit_stats is None or not _same_fit_dataset(model, dataset):
        return None

    edf = float(model.result.effective_df)
    n = dataset.n_obs
    weights = dataset.sample_weight
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    likelihood_size = dispersion_likelihood_size(
        weights,
        weight_semantics=model_weight_semantics(model),
    )
    log_likelihood = float(fit_stats.log_likelihood)
    aic = float(-2.0 * log_likelihood + 2.0 * edf)
    bic = float(-2.0 * log_likelihood + np.log(likelihood_size) * edf)
    denom = likelihood_size - edf - 1.0
    return {
        "deviance": float(model.result.deviance),
        "aic": aic,
        "aicc": float(aic + 2.0 * edf * (edf + 1.0) / denom) if denom > 0 else float("inf"),
        "bic": bic,
        "log_likelihood": log_likelihood,
        "explained_deviance": float(fit_stats.explained_deviance),
        "pearson_chi2": float(fit_stats.pearson_chi2),
        "effective_df": edf,
    }


def _compute_metrics(model, X, y, weights, offset) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    offset_arg = None if offset is None else np.asarray(offset, dtype=np.float64).ravel()
    mu = np.asarray(model.predict(X, offset=offset_arg), dtype=np.float64).ravel()
    if w.size != y_arr.size:
        raise ValueError(f"sample_weight has length {w.size}, expected {y_arr.size}.")
    if offset_arg is not None and offset_arg.size != y_arr.size:
        raise ValueError(f"offset has length {offset_arg.size}, expected {y_arr.size}.")
    if mu.size != y_arr.size:
        raise ValueError(f"Predictions have length {mu.size}, expected {y_arr.size}.")
    family = model._distribution
    phi = float(model.result.phi)
    edf = float(model.result.effective_df)
    likelihood_size = dispersion_likelihood_size(
        w,
        weight_semantics=model_weight_semantics(model),
    )
    deviance = float(np.sum(w * family.deviance_unit(y_arr, mu)))
    log_likelihood = float(
        weighted_log_likelihood(
            family, y_arr, mu, w, phi, weight_semantics=model_weight_semantics(model)
        )
    )
    aic = float(-2.0 * log_likelihood + 2.0 * edf)
    bic = float(-2.0 * log_likelihood + np.log(likelihood_size) * edf)
    denom = likelihood_size - edf - 1.0
    aicc = float(aic + 2.0 * edf * (edf + 1.0) / denom) if denom > 0 else float("inf")
    null_deviance, null_mu = _null_deviance_and_mu(model, y_arr, w, offset_arg)
    explained = _explained_deviance(deviance, null_deviance, y_arr, null_mu, w)
    variance = np.maximum(family.variance(mu), 1e-300)
    pearson = float(np.sum(w * (y_arr - mu) ** 2 / variance))
    return {
        "deviance": deviance,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "log_likelihood": log_likelihood,
        "explained_deviance": explained,
        "pearson_chi2": pearson,
        "effective_df": edf,
    }


def _null_deviance(model, y: np.ndarray, weights: np.ndarray, offset: np.ndarray | None) -> float:
    return _null_deviance_and_mu(model, y, weights, offset)[0]


def _null_deviance_and_mu(
    model,
    y: np.ndarray,
    weights: np.ndarray,
    offset: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    from superglm.model.fit_ops import _compute_null_mu

    mu = _compute_null_mu(
        y,
        weights,
        offset,
        model._distribution,
        model._link,
        weight_semantics=model_weight_semantics(model),
    )
    deviance = float(np.sum(weights * model._distribution.deviance_unit(y, mu)))
    return deviance, mu
