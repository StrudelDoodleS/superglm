"""Model metric payloads for the editor."""

from __future__ import annotations

from typing import Any

import numpy as np

from superglm.editor.evaluation import EvaluationDataset, named_metrics_dataset

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


def metrics_payload(
    session,
    metric: str,
    *,
    source: str = "in_force",
    dataset: str | None = None,
) -> dict[str, Any]:
    # Metrics compare the immutable original fit with the in-force editor model
    # on the retained training frame. Manual coefficient-edit metrics are
    # prediction diagnostics; structural refits are fitted-model metrics.
    metric = metric if metric in METRIC_LABELS else "deviance"
    reference_model = getattr(session, "reference_model", session.model)
    if reference_model is None:
        return {"available": False, "metric": metric, "error": "No source model is attached."}
    eval_dataset = named_metrics_dataset(session, dataset)
    if eval_dataset is None:
        return {
            "available": False,
            "metric": metric,
            "error": "No evaluation data is available.",
        }

    selected_model = reference_model if source == "original" else session.to_model()
    original_metrics = compute_dataset_metrics(reference_model, eval_dataset)
    edited_metrics = compute_dataset_metrics(selected_model, eval_dataset)
    original = original_metrics[metric]
    edited = edited_metrics[metric]
    return {
        "available": True,
        "metric": metric,
        "label": METRIC_LABELS[metric],
        "dataset": eval_dataset.name,
        "dataset_label": eval_dataset.label,
        "n_obs": eval_dataset.n_obs,
        "original": original,
        "edited": edited,
        "delta": edited - original,
        "metrics": {"original": original_metrics, "edited": edited_metrics},
    }


def compute_dataset_metrics(model, dataset: EvaluationDataset) -> dict[str, float]:
    fit_artifacts = _fit_artifact_metrics(model, dataset)
    if fit_artifacts is not None:
        return fit_artifacts
    weights = dataset.sample_weight
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
    log_likelihood = float(fit_stats.log_likelihood)
    aic = float(-2.0 * log_likelihood + 2.0 * edf)
    bic = float(-2.0 * log_likelihood + np.log(max(n, 1)) * edf)
    denom = n - edf - 1.0
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
    n = y_arr.size
    deviance = float(np.sum(w * family.deviance_unit(y_arr, mu)))
    log_likelihood = float(family.log_likelihood(y_arr, mu, w, phi))
    aic = float(-2.0 * log_likelihood + 2.0 * edf)
    bic = float(-2.0 * log_likelihood + np.log(max(n, 1)) * edf)
    denom = n - edf - 1.0
    aicc = float(aic + 2.0 * edf * (edf + 1.0) / denom) if denom > 0 else float("inf")
    null_deviance = _null_deviance(model, y_arr, w, offset_arg)
    explained = float(1.0 - deviance / null_deviance) if null_deviance > 0 else float("nan")
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
    from superglm.model.fit_ops import _compute_null_mu

    mu = _compute_null_mu(y, weights, offset, model._distribution, model._link)
    return float(np.sum(weights * model._distribution.deviance_unit(y, mu)))
