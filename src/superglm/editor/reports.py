"""Display-only validation and final-fit report payloads."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from superglm.editor.cv import FINAL_NOT_RUN, cv_report_payload, diagnose_report_payload
from superglm.editor.diagnose import fit_trace
from superglm.editor.display import fold_chart_display
from superglm.editor.evaluation import evaluation_datasets
from superglm.editor.lss.validation import LSS_METRIC_LABELS, lss_split_reason
from superglm.editor.lss.view import is_lss
from superglm.editor.metrics import METRIC_LABELS, compute_dataset_metrics
from superglm.editor.summaries import summary_payload

_LOGGER = logging.getLogger(__name__)

_REPORT_METRICS = (
    "deviance",
    "aic",
    "bic",
    "log_likelihood",
    "explained_deviance",
    "pearson_chi2",
    "effective_df",
)


def report_metrics(model) -> dict[str, str]:
    """The metrics a report shows for ``model``, with their labels.

    A SuperLSS's CRPS is in the units of the response, so its label carries the
    session's amount unit where it has one.
    """
    if is_lss(model):
        labels = dict(LSS_METRIC_LABELS)
        unit = getattr(model, "amount_unit", None)
        if unit:
            labels["crps"] = f"{labels['crps']} ({unit})"
        return labels
    return {metric: METRIC_LABELS[metric] for metric in _REPORT_METRICS}


def validation_report_payload(
    session,
    *,
    splits: list[dict[str, Any]] | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
    compares_edit: bool | None = None,
) -> dict[str, Any]:
    """Return train/validation/test metrics plus optional supplied CV evidence.

    ``compares_edit`` is decided by the caller with the model the report describes; without it
    the session is read now.
    """
    if splits is None:
        splits = _split_metrics(session)
    revision = session.model_revision if model_revision is None else int(model_revision)
    return {
        "available": bool(splits),
        "model_revision": revision,
        "request_sequence": request_sequence,
        "report": "validation",
        "title": "Validation Report",
        "note": _validation_note(session, splits),
        "metric_labels": report_metrics(session.model),
        "compares_edit": _compares_edit(session) if compares_edit is None else compares_edit,
        "splits": splits,
        "cv_report": getattr(session, "cv_report", None),
        "can_run_cv": False,
    }


def final_fit_report_payload(
    widget,
    *,
    splits: list[dict[str, Any]] | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
    model_override=None,
    final_fit=None,
    compares_edit: bool | None = None,
    final_fit_stale: bool | None = None,
) -> dict[str, Any]:
    """Return the current in-force model summary, split metrics and any Final fit.

    ``compares_edit`` and ``final_fit_stale`` are decided by the caller with the model the report
    describes; without them the session is read now.
    """
    if splits is None:
        splits = _split_metrics(widget.session)
    revision = widget.session.model_revision if model_revision is None else int(model_revision)
    summary = (
        {}
        if is_lss(widget.session.model)
        else {"summary": summary_payload(widget, "in_force", model_override=model_override)}
    )
    return {
        "available": bool(splits),
        "model_revision": revision,
        "request_sequence": request_sequence,
        "report": "final",
        "title": "Final Fit Report",
        "note": (
            "The model as it stands now. Where you moved a curve by hand, standard errors "
            "and fit statistics still describe the fit before the move."
        ),
        "metric_labels": report_metrics(widget.session.model),
        "compares_edit": (
            _compares_edit(widget.session) if compares_edit is None else compares_edit
        ),
        "splits": splits,
        **summary,
        "can_run_cv": False,
        "final_fit": _final_fit_section(widget, final_fit, final_fit_stale),
    }


def _compares_edit(session) -> bool:
    """Whether the model in force differs from the one the editor opened: only then is a change to show."""
    reference_model = getattr(session, "reference_model", session.model)
    return reference_model is not session.model or bool(session.edited_terms())


def _final_fit_status(final_model) -> dict[str, Any]:
    """How the refit's smoothing search ended, in the Diagnose tab's words, read from what it recorded.

    Nothing is recomputed. A refit whose record cannot be read says nothing about it.
    """
    try:
        trace = fit_trace(final_model, fit_id="final", label="Final fit")
    except Exception:
        _LOGGER.warning("The Final fit has no convergence status.", exc_info=True)
        return {"fit_status": None, "fit_verdict": None}
    return {"fit_status": trace["status"], "fit_verdict": trace["verdict"]}


def _final_fit_stale(widget, final_fit) -> bool:
    """Whether the model in force is no longer the one the Final fit was run on."""
    with widget._lock:
        return not final_fit.source.describes(widget.session)


def _final_fit_section(widget, final_fit, stale: bool | None = None) -> dict[str, Any]:
    """The Final fit on train and validation rows, once one has run."""
    if final_fit is None:
        return {"available": False, "note": FINAL_NOT_RUN}
    if is_lss(final_fit.model):
        compact = None
    else:
        compact = summary_payload(widget, "in_force", model_override=final_fit.model).get("compact")
    return {
        "available": True,
        "stale": _final_fit_stale(widget, final_fit) if stale is None else stale,
        "n_rows": final_fit.n_rows,
        "splits": list(final_fit.splits),
        "carried": [fold_chart_display(name, final_fit.model).label for name in final_fit.carried],
        # Waiting changes are never in a Final fit, and staging or undoing one
        # leaves the model revision alone, so the count is the one waiting now.
        "pending": len(widget.session.pending),
        "summary": compact,
        "test_scores": final_fit.test_scores,
        "has_test_split": widget.session._evaluation_data.get("test") is not None,
        **_final_fit_status(final_fit.model),
    }


def report_payload(
    widget,
    report: str,
    *,
    splits: list[dict[str, Any]] | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
    model_override=None,
    final_fit=None,
    compares_edit: bool | None = None,
    final_fit_stale: bool | None = None,
) -> dict[str, Any]:
    """Dispatch a named report for the local editor app."""
    if report == "cv":
        return cv_report_payload(widget, request_sequence=request_sequence)
    if report == "diagnose":
        return diagnose_report_payload(widget, request_sequence=request_sequence)
    if report == "final":
        return final_fit_report_payload(
            widget,
            splits=splits,
            model_revision=model_revision,
            request_sequence=request_sequence,
            model_override=model_override,
            final_fit=final_fit,
            compares_edit=compares_edit,
            final_fit_stale=final_fit_stale,
        )
    return validation_report_payload(
        widget.session,
        splits=splits,
        model_revision=model_revision,
        request_sequence=request_sequence,
        compares_edit=compares_edit,
    )


def split_metrics_payload(
    datasets,
    metric_pairs: dict[str, tuple[dict[str, float], dict[str, float]]],
    metrics: Iterable[str] = _REPORT_METRICS,
    model=None,
) -> list[dict[str, Any]]:
    """Assemble split rows from cached original/current scalar pairs.

    A split a SuperLSS ``model`` cannot score is a row with ``available``
    False and the reason as ``error``.
    """
    metrics = tuple(metrics)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        original, edited = metric_pairs[dataset.name]
        identity = {
            "name": dataset.name,
            "label": dataset.label,
            "n_obs": dataset.n_obs,
            "source": dataset.source,
        }
        reason = lss_split_reason(model, dataset) if is_lss(model) else None
        if reason is not None:
            rows.append({**identity, "available": False, "error": reason})
            continue
        rows.append(
            {
                **identity,
                "metrics": {
                    "original": {metric: original[metric] for metric in metrics},
                    "edited": {metric: edited[metric] for metric in metrics},
                    "delta": {metric: edited[metric] - original[metric] for metric in metrics},
                },
            }
        )
    return rows


def _split_metrics(session) -> list[dict[str, Any]]:
    reference_model = getattr(session, "reference_model", session.model)
    if reference_model is None:
        return []
    edited_model = session.to_model()
    metric_pairs: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for dataset in evaluation_datasets(session):
        original = compute_dataset_metrics(reference_model, dataset)
        edited = compute_dataset_metrics(edited_model, dataset)
        metric_pairs[dataset.name] = (original, edited)
    return split_metrics_payload(
        evaluation_datasets(session), metric_pairs, report_metrics(reference_model), reference_model
    )


def _validation_note(session, splits: list[dict[str, Any]]) -> str:
    if not splits:
        return "No train, validation, test, or retained fit data is available."
    explicit = getattr(session, "_evaluation_data", {})
    if not explicit:
        return "No explicit split data supplied; metrics are in-sample on retained fit data."
    if "validation" not in explicit:
        return "No validation split supplied; live editor metrics fall back to train data."
    return "Each split scored by the model as edited. Δ is the change from the last fit."


__all__ = [
    "final_fit_report_payload",
    "report_payload",
    "split_metrics_payload",
    "validation_report_payload",
]
