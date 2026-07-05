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
    weights = dataset.sample_weight
    if weights is None:
        weights = np.ones(dataset.n_obs, dtype=np.float64)
    return _compute_metrics(model, dataset.X, dataset.y, weights, dataset.offset)


def _compute_metrics(model, X, y, weights, offset) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    offset_arg = None if offset is None else np.asarray(offset, dtype=np.float64)
    mu = np.asarray(model.predict(X, offset=offset_arg), dtype=np.float64)
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
    null_deviance = _null_deviance(model, y_arr, w)
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


def _null_deviance(model, y: np.ndarray, weights: np.ndarray) -> float:
    y_bar = float(np.average(y, weights=weights))
    mu = np.full(y.size, y_bar, dtype=np.float64)
    return float(np.sum(weights * model._distribution.deviance_unit(y, mu)))
