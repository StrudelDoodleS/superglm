"""Model metric payloads for the editor."""

from __future__ import annotations

from typing import Any

import numpy as np

from superglm._utils import _explained_deviance
from superglm._weight_units import unit_log_likelihood
from superglm.editor.apply import model_for_split_scoring
from superglm.editor.errors import no_quote_message
from superglm.editor.evaluation import (
    EvaluationDataset,
    _validate_evaluation_weights,
    edited_statistics_dataset,
    named_metrics_dataset,
)
from superglm.editor.lss.availability import unavailable_evidence
from superglm.editor.lss.validation import lss_dataset_metrics
from superglm.editor.lss.view import is_lss
from superglm.solvers.dispersion import dispersion_likelihood_size, model_weight_semantics
from superglm.solvers.working_rows import pearson_chi2

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


ORIGINAL_UNSCORED = (
    "The original model has no quote for some of these rows, so the change against it is not shown."
)


def metric_comparison_payload(
    metric: str,
    dataset: EvaluationDataset,
    original_metrics: dict[str, Any],
    edited_metrics: dict[str, Any],
    *,
    model_revision: int,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """Assemble one comparison payload from already-computed scalar dictionaries.

    A dictionary holding an ``"error"`` (a split with rows that have no quote) is
    not compared. Where it is the edited model's, the payload is unavailable and
    says why; where only the original's, the edited scores are shown, the
    original's read None and the ``note`` says why there is no change.
    """
    selected_metric = metric if metric in METRIC_LABELS else "deviance"
    reason = edited_metrics.get("error")
    if reason is not None:
        return {
            "available": False,
            "model_revision": int(model_revision),
            "request_sequence": request_sequence,
            "metric": selected_metric,
            "dataset": dataset.name,
            "dataset_label": dataset.label,
            "n_obs": dataset.n_obs,
            "error": reason,
        }
    # An edit can restore the quotes the original model lacks: its scores are shown, and
    # only the comparison with the original, which has none, is withheld.
    unscored = "error" in original_metrics
    if unscored:
        original_metrics = dict.fromkeys(edited_metrics)
    original = original_metrics[selected_metric]
    edited = edited_metrics[selected_metric]
    payload = {
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
        "delta": None if unscored else edited - original,
        "metrics": {"original": original_metrics, "edited": edited_metrics},
    }
    if unscored:
        payload["note"] = ORIGINAL_UNSCORED
    return payload


def metrics_payload(
    session,
    metric: str,
    *,
    source: str = "in_force",
    dataset: str | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    # Compare both models on the same requested split.
    metric = metric if metric in METRIC_LABELS else "deviance"
    revision = session.model_revision if model_revision is None else int(model_revision)
    calm = unavailable_evidence(
        session.model,
        "metrics",
        model_revision=revision,
        request_sequence=request_sequence,
        metric=metric,
        source=source,
    )
    if calm is not None:
        return calm
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

    selected_model = (
        reference_model
        if source == "original"
        else model_for_split_scoring(
            session.model, session.terms, edited_statistics_dataset(session)
        )
    )
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


def compute_dataset_metrics(model, dataset: EvaluationDataset) -> dict[str, Any]:
    """Score ``dataset`` with the model's predictions, as ``model.metrics`` does.

    A dataset holding the fit's own objects is scored the same way: the fit's
    own statistics belong to its fitting design, which on a discrete fit is
    the binned one that ``predict`` does not reproduce.  When those objects
    are unchanged since the fit (its data guard), the reports the fit already
    gave are not repeated (weight-contract check, custom-family prior-weight
    report, ``predict``'s warnings), as in ``model.metrics``.  The predicted
    values are the same either way.

    Where a SuperGLM gives a weighted row no quote (``predict`` returns NaN for
    it), the dataset has no scores: the result is ``{"error": reason}``.
    """
    if is_lss(model):
        return lss_dataset_metrics(model, dataset)
    validated_weights = _validate_evaluation_weights(
        dataset.sample_weight,
        dataset.n_obs,
        family=model._distribution,
        weight_semantics=model_weight_semantics(model),
    )
    weights = validated_weights
    if weights is None:
        weights = np.ones(dataset.n_obs, dtype=np.float64)
    from superglm.model.base import NoPredictedMeanError

    try:
        return _compute_metrics(
            model,
            dataset.X,
            dataset.y,
            weights,
            dataset.offset,
            contract_checked=_same_fit_dataset(model, dataset),
        )
    except NoPredictedMeanError as exc:
        return {"error": no_quote_message(exc)}


def _same_fit_dataset(model, dataset: EvaluationDataset) -> bool:
    """Whether ``dataset`` holds the fit's own rows, unchanged since the fit checked them.

    Identity alone is not enough: an array mutated in place keeps its
    identity.  The fit's data guard confirms the values, as
    ``explain_ops.metrics`` does.
    """
    fit_weight_ref = getattr(model, "_fit_sample_weight_ref", None)
    fit_weights = getattr(model, "_fit_weights", None)
    fit_offset_ref = getattr(model, "_fit_offset_ref", None)
    fit_offset = getattr(model, "_fit_offset", None)
    weights_match = dataset.sample_weight is fit_weight_ref or dataset.sample_weight is fit_weights
    offset_matches = dataset.offset is fit_offset_ref or dataset.offset is fit_offset
    fit_data_guard = getattr(model, "_fit_data_guard", None)
    return bool(
        dataset.X is getattr(model, "_fit_X_ref", None)
        and dataset.y is getattr(model, "_fit_y_ref", None)
        and weights_match
        and offset_matches
        and fit_data_guard is not None
        and fit_data_guard.matches(
            dataset.X,
            dataset.y,
            dataset.sample_weight,
            dataset.offset,
            fit_weights=fit_weights,
            fit_offset=fit_offset,
            own_frame=True,
        )
    )


def _compute_metrics(
    model, X, y, weights, offset, *, contract_checked: bool = False
) -> dict[str, float]:
    from superglm.model import base
    from superglm.solvers.clip_rows import unheld_weights, weighted_deviance_sum

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    offset_arg = None if offset is None else np.asarray(offset, dtype=np.float64).ravel()
    if w.size != y_arr.size:
        raise ValueError(f"sample_weight has length {w.size}, expected {y_arr.size}.")
    if offset_arg is not None and offset_arg.size != y_arr.size:
        raise ValueError(f"offset has length {offset_arg.size}, expected {y_arr.size}.")
    # ``predict``'s values, refused where a weighted row has none (the split's reason).
    if contract_checked:
        # Without its warnings, as ``model.metrics`` scores the fit's own objects.
        predicted = base.predict_exact(model, X, offset=offset_arg, warn=False)
        base.require_priced_rows(model, predicted, w, "the scores")
    else:
        predicted = base.predict_priced(model, X, offset_arg, w, "the scores")
    # A binomial row past the solver's clip is read from the likelihood (``held``).
    predicted, held = base.score_rows(model, X, offset_arg, y_arr, w, predicted)
    mu = np.asarray(predicted, dtype=np.float64).ravel()
    family = model._distribution
    edf = float(model.result.effective_df)
    likelihood_size = dispersion_likelihood_size(
        w,
        weight_semantics=model_weight_semantics(model),
    )
    # A zero-weight row adds an exact 0, even where it has no quote (NaN).
    deviance = weighted_deviance_sum(family, y_arr, w, mu, held)
    # `compute_dataset_metrics` bypasses `ModelMetrics` entirely, so the
    # evaluation-boundary check is repeated here for editor validation and
    # test splits.  The fit's own objects were checked by the fit.
    if not contract_checked:
        from superglm.model.input_validation import check_weight_contract

        check_weight_contract(y_arr, w, family, model_weight_semantics(model))
    log_likelihood = float(
        unit_log_likelihood(
            family,
            y_arr,
            mu,
            unheld_weights(held, w),
            model.result,
            weight_semantics=model_weight_semantics(model),
            report_contract=not contract_checked,
        )
    )
    if held is not None:
        log_likelihood += held.log_likelihood_sum(w)
    aic = float(-2.0 * log_likelihood + 2.0 * edf)
    bic = float(-2.0 * log_likelihood + np.log(likelihood_size) * edf)
    denom = likelihood_size - edf - 1.0
    aicc = float(aic + 2.0 * edf * (edf + 1.0) / denom) if denom > 0 else float("inf")
    null_deviance, null_mu = _null_deviance_and_mu(
        model, y_arr, w, offset_arg, likelihood_rows=held is not None
    )
    explained = _explained_deviance(deviance, null_deviance, y_arr, null_mu, w)
    pearson = float(
        pearson_chi2(
            distribution=family,
            y=y_arr,
            mu=mu,
            sample_weight=unheld_weights(held, w),
            variance_floor=0.0,
        )
    )
    if held is not None:
        pearson += held.pearson_sum(w)
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


def _null_deviance_and_mu(
    model,
    y: np.ndarray,
    weights: np.ndarray,
    offset: np.ndarray | None,
    *,
    likelihood_rows: bool,
) -> tuple[float, np.ndarray]:
    """The intercept-only model's deviance and means on these rows (``fit_ops._null_state``).

    ``likelihood_rows`` says the model's own score read a held row from the
    likelihood; a fit that ended on the likelihood's rows always does.
    """
    from superglm.model.fit_ops import _null_state
    from superglm.solvers.clip_rows import on_likelihood_rows, weighted_deviance_sum

    mu, held = _null_state(
        y,
        weights,
        offset,
        model._distribution,
        model._link,
        weight_semantics=model_weight_semantics(model),
        likelihood_rows=likelihood_rows or on_likelihood_rows(model.result),
    )
    return weighted_deviance_sum(model._distribution, y, weights, mu, held), mu
