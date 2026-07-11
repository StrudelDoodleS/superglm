"""Display-only validation and final-fit report payloads."""

from __future__ import annotations

from typing import Any

from superglm.editor.evaluation import evaluation_datasets
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
        "metric_labels": {metric: METRIC_LABELS[metric] for metric in _REPORT_METRICS},
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
) -> dict[str, Any]:
    """Return the current in-force model summary and split metrics."""
    if splits is None:
        splits = _split_metrics(widget.session)
    revision = widget.session.model_revision if model_revision is None else int(model_revision)
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
        "metric_labels": {metric: METRIC_LABELS[metric] for metric in _REPORT_METRICS},
        "splits": splits,
        "summary": summary_payload(widget, "in_force"),
        "can_run_cv": False,
    }


def report_payload(
    widget,
    report: str,
    *,
    splits: list[dict[str, Any]] | None = None,
    model_revision: int | None = None,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """Dispatch a named report for the local editor app."""
    if report == "final":
        return final_fit_report_payload(
            widget,
            splits=splits,
            model_revision=model_revision,
            request_sequence=request_sequence,
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
) -> list[dict[str, Any]]:
    """Assemble split rows from cached original/current scalar pairs."""
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        original, edited = metric_pairs[dataset.name]
        rows.append(
            {
                "name": dataset.name,
                "label": dataset.label,
                "n_obs": dataset.n_obs,
                "source": dataset.source,
                "metrics": {
                    "original": {metric: original[metric] for metric in _REPORT_METRICS},
                    "edited": {metric: edited[metric] for metric in _REPORT_METRICS},
                    "delta": {
                        metric: edited[metric] - original[metric] for metric in _REPORT_METRICS
                    },
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
    return split_metrics_payload(evaluation_datasets(session), metric_pairs)


def _validation_note(session, splits: list[dict[str, Any]]) -> str:
    if not splits:
        return "No train, validation, test, or retained fit data is available."
    explicit = getattr(session, "_evaluation_data", {})
    if not explicit:
        return "No explicit split data supplied; metrics are in-sample on retained fit data."
    if "validation" not in explicit:
        return "No validation split supplied; live editor metrics fall back to train data."
    return "Split metrics are display-only evidence; CV fitting is supplied outside the editor."


__all__ = [
    "final_fit_report_payload",
    "report_payload",
    "split_metrics_payload",
    "validation_report_payload",
]
