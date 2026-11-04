"""Display-only validation and final-fit report payloads."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from superglm.editor.cv import FINAL_NOT_RUN, cv_report_payload
from superglm.editor.evaluation import evaluation_datasets
from superglm.editor.lss.display import fold_chart_display
from superglm.editor.lss.validation import LSS_METRIC_LABELS, lss_split_reason
from superglm.editor.lss.view import is_lss
from superglm.editor.metrics import METRIC_LABELS, compute_dataset_metrics
from superglm.editor.summaries import summary_payload

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
    """The metrics a report shows for ``model``, with their labels."""
    if is_lss(model):
        return dict(LSS_METRIC_LABELS)
    return {metric: METRIC_LABELS[metric] for metric in _REPORT_METRICS}


def validation_report_payload(
    session,
    *,
    splits: list[dict[str, Any]] | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """Return train/validation/test metrics plus optional supplied CV evidence."""
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
) -> dict[str, Any]:
    """Return the current in-force model summary, split metrics and any Final fit."""
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
            "Current in-force model report. Structural edits are reflected in the "
            "feature definition; manual coefficient edits keep inference stale."
        ),
        "metric_labels": report_metrics(widget.session.model),
        "splits": splits,
        **summary,
        "can_run_cv": False,
        "final_fit": _final_fit_section(widget, final_fit, model_revision=revision),
    }


def _final_fit_section(widget, final_fit, *, model_revision: int) -> dict[str, Any]:
    """The Final fit on train and validation rows, once one has run."""
    if final_fit is None:
        return {"available": False, "note": FINAL_NOT_RUN}
    if is_lss(final_fit.model):
        compact = None
    else:
        compact = summary_payload(widget, "in_force", model_override=final_fit.model).get("compact")
    return {
        "available": True,
        "stale": final_fit.model_revision != model_revision,
        "n_rows": final_fit.n_rows,
        "splits": list(final_fit.splits),
        "carried": [
            fold_chart_display(name).label if is_lss(final_fit.model) else name
            for name in final_fit.carried
        ],
        # Waiting changes are never in a Final fit, and staging or undoing one
        # leaves the model revision alone, so the count is the one waiting now.
        "pending": len(widget.session.pending),
        "summary": compact,
        "test_scores": final_fit.test_scores,
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
) -> dict[str, Any]:
    """Dispatch a named report for the local editor app."""
    if report == "cv":
        return cv_report_payload(widget, request_sequence=request_sequence)
    if report == "final":
        return final_fit_report_payload(
            widget,
            splits=splits,
            model_revision=model_revision,
            request_sequence=request_sequence,
            model_override=model_override,
            final_fit=final_fit,
        )
    return validation_report_payload(
        widget.session,
        splits=splits,
        model_revision=model_revision,
        request_sequence=request_sequence,
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
