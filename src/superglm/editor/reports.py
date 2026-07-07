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


def validation_report_payload(session) -> dict[str, Any]:
    """Return train/validation/test metrics plus optional supplied CV evidence."""
    splits = _split_metrics(session)
    return {
        "available": bool(splits),
        "report": "validation",
        "title": "Validation Report",
        "note": _validation_note(session, splits),
        "metric_labels": {metric: METRIC_LABELS[metric] for metric in _REPORT_METRICS},
        "splits": splits,
        "cv_report": getattr(session, "cv_report", None),
        "can_run_cv": False,
    }


def final_fit_report_payload(widget) -> dict[str, Any]:
    """Return the current in-force model summary and split metrics."""
    splits = _split_metrics(widget.session)
    return {
        "available": bool(splits),
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


def report_payload(widget, report: str) -> dict[str, Any]:
    """Dispatch a named report for the local editor app."""
    if report == "final":
        return final_fit_report_payload(widget)
    return validation_report_payload(widget.session)


def _split_metrics(session) -> list[dict[str, Any]]:
    reference_model = getattr(session, "reference_model", session.model)
    if reference_model is None:
        return []
    edited_model = session.to_model()
    rows: list[dict[str, Any]] = []
    for dataset in evaluation_datasets(session):
        original = compute_dataset_metrics(reference_model, dataset)
        edited = compute_dataset_metrics(edited_model, dataset)
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
    "validation_report_payload",
]
