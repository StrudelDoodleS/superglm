"""Notebook/editor frontend for editor sessions.

The default renderer is a tiny local HTML app served by the Python kernel. This
avoids custom Jupyter widget modules, which are often unavailable in VS Code.
"""

from __future__ import annotations

import atexit
import html
import io
import logging
import os
import secrets
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from superglm.editor import metrics as metrics_module
from superglm.editor import persistence, rating_preview
from superglm.editor.apply import materialize_edit_request
from superglm.editor.cv import (
    FINAL_NOT_RUN,
    FINAL_STALE,
    SUPERSEDED,
    CVRun,
    FinalFit,
    capture_cv_run,
    capture_final_fit,
    run_cv,
    run_final_fit,
)
from superglm.editor.errors import EditorClientError, EditorKeyError, EditorValueError
from superglm.editor.evaluation import (
    default_metrics_dataset,
    evaluation_datasets,
    named_metrics_dataset,
    training_export_dataset,
)
from superglm.editor.evaluation_cache import (
    DatasetMetricRequest,
    EvaluationCache,
    EvaluationKey,
    model_metric_signature,
)
from superglm.editor.evidence import EvidenceCoordinator, EvidenceKey
from superglm.editor.io import jsonable
from superglm.editor.jobs import JobContext, JobRunner
from superglm.editor.metrics import metric_comparison_payload, metrics_payload
from superglm.editor.native_dialogs import open_directory_path
from superglm.editor.notebook import NotebookTransport, notebook_view_class
from superglm.editor.payloads import (
    pending_payload,
    session_payload,
    timeline_payload,
    undo_redo_payload,
)
from superglm.editor.rating_preview import PREVIEW_IMPACT_BINS, RatingPreview
from superglm.editor.reports import report_payload, split_metrics_payload
from superglm.editor.server import EditorAppServer, create_editor_app
from superglm.editor.summaries import offset_label_payload, summary_payload
from superglm.inference.summary_levels import validate_level_display
from superglm.profiling._reporting import (
    cached_tweedie_profile_ci,
    profile_cautioned,
    reported_interval,
)

_LIVE_WIDGETS: set[EditorWidget] = set()
_LOGGER = logging.getLogger(__name__)

_EXPORT_MEDIA_TYPES = {
    "joblib": "application/octet-stream",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "final": "application/octet-stream",
    "structure": "application/json",
}
_EXPORT_DEFAULT_FILENAMES = {
    "joblib": "superglm_edited_model.joblib",
    "xlsx": "superglm_rating_tables.xlsx",
    "final": "superglm_final_model.joblib",
    "structure": "superglm_structure.json",
}
_EXPORT_SUFFIXES = {"joblib": ".joblib", "xlsx": ".xlsx", "final": ".joblib", "structure": ".json"}
_EXCEL_NEEDS_TRAINING_DATA = (
    "Excel export requires train_data or retained fit data; "
    "validation/test data are not substituted."
)


@dataclass(frozen=True)
class ExportResult:
    """Complete revision-pinned export ready for download or disk."""

    data: bytes
    filename: str
    media_type: str
    model_revision: int
    validation_scope: str | None = None


def _normalise_export_format(format: str) -> str:
    """Normalize user-facing export aliases to canonical formats."""
    normalized = str(format).strip().lower().lstrip(".")
    if normalized in {"joblib", "model"}:
        return "joblib"
    if normalized in {"xlsx", "excel"}:
        return "xlsx"
    if normalized in {"final", "final_fit"}:
        return "final"
    if normalized in {"structure", "json"}:
        return "structure"
    raise EditorValueError(f"Unsupported export format: {format!r}")


def _safe_export_filename(format: str, filename: str | None) -> str:
    """Return a basename with the canonical suffix for ``format``."""
    name = filename or _EXPORT_DEFAULT_FILENAMES[format]
    if not name or Path(name).name != name or "/" in name or "\\" in name:
        raise EditorValueError("filename must not contain directory separators")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise EditorValueError("filename must not contain control characters")
    suffix = Path(name).suffix.lower()
    expected = _EXPORT_SUFFIXES[format]
    if suffix and suffix != expected:
        raise EditorValueError(f"filename extension must be {expected} for {format} exports")
    if not suffix:
        name = f"{name}{expected}"
    return name


class EditorWidget:
    """Lightweight iframe app for an :class:`EditorSession`.

    In ``"server"`` mode the app uses a local HTTP server rather than a custom
    Jupyter widget model. It renders in VS Code as plain HTML and updates the
    live Python session via JSON requests to the kernel process. That needs a
    browser on the kernel's machine; ``"notebook"`` mode runs the same app
    inside the notebook cell instead and carries its requests over widget
    messages, for hosted notebooks such as Databricks (see
    :mod:`superglm.editor.notebook`).
    """

    def __init__(self, session, *, mode: str | None = None, **kwargs: Any):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected EditorWidget argument(s): {unknown}")
        self.mode = _display_mode(mode)
        if self.mode == "notebook":
            notebook_view_class()  # anywidget is optional: say so before starting anything
        self.session = session
        self.control_counts: dict[str, int] = {}
        self._offset_refit_model = None
        self._offset_refit_terms: list[str] = []
        self._offset_refit_labels: list[dict[str, Any]] = []
        self._offset_refit_revision: int | None = None
        self._profile_jobs: dict[str, dict[str, Any]] = {}
        self._profile_job_counter = 0
        self._profile_condition = threading.Condition(threading.RLock())
        # The Cross-validation tab's results, each kept with the revision it
        # ran on: a newer revision marks them stale rather than dropping them.
        self._cv_run: CVRun | None = None
        self._final_fit: FinalFit | None = None
        # Run CV and Final fit: one background job of each kind at a time.
        # A starter captures a job's inputs under the widget lock and returns
        # its work (run off the lock) and its publish step (which re-takes it).
        self._jobs = JobRunner(name=f"editor-{id(self):x}")
        self._job_starters: dict[
            str,
            Callable[[], tuple[Callable[[JobContext], Any], Callable[[Any], dict[str, Any]]]],
        ] = {"cv": self._cv_job, "final_fit": self._final_fit_job}
        self._rating_preview: RatingPreview | None = None
        # Free-level comparisons by term, for the fit in force. A hand edit
        # changes neither side of a comparison, so only a new fit, its token,
        # puts the cache aside.
        self._fit_model: Any = None
        self._fit_token = 0
        self._free_levels: dict[str, dict[str, Any]] = {}
        self._token = secrets.token_urlsafe(24)
        self.terms = session_payload(session, self.control_counts)
        self.selected_term = next(iter(self.terms), "")
        self._state_generation = 0
        self._chart_generation = 0
        self._lock = threading.RLock()
        self._closed = False
        self._evaluation_cache = EvaluationCache()
        self._evidence = EvidenceCoordinator(f"editor-{id(self):x}")
        self._server: EditorAppServer | None = None
        self._notebook: NotebookTransport | None = None
        self.host: str | None = None
        self.port: int | None = None
        self.url: str | None = None
        self.app_url: str | None = None
        if self.mode == "notebook":
            self._notebook = NotebookTransport(create_editor_app(self), self._token)
        else:
            # A local iframe avoids Jupyter widget-extension dependencies while
            # still letting Python own the authoritative edit state.
            self._server = EditorAppServer(self)
            self.host, self.port = self._server.host, self._server.port
            self.url = f"http://127.0.0.1:{self.port}"
            self.app_url = f"{self.url}?token={self._token}"
            self._server.start()
        _LIVE_WIDGETS.add(self)

    def _repr_mimebundle_(self, include=None, exclude=None) -> dict[str, Any] | None:
        if self._notebook is None:
            return None
        return self._notebook.mimebundle()

    def _repr_html_(self) -> str | None:
        if self.app_url is None:
            return None
        src = html.escape(self.app_url, quote=True)
        display_url = html.escape(self.app_url, quote=True)
        return (
            "<div style='width:100%;max-width:1180px'>"
            f"<iframe src='{src}' width='100%' height='720' "
            "style='border:1px solid #d0d7de;border-radius:6px;background:white'></iframe>"
            "<div style='font:12px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "color:#57606a;margin-top:4px'>"
            f"SuperGLM editor running at <a href='{display_url}' target='_blank'>{display_url}</a>"
            "</div></div>"
        )

    def close(self) -> None:
        """Stop the local editor server, or the in-notebook view."""
        if self._closed:
            return
        self._closed = True
        _LIVE_WIDGETS.discard(self)
        self._jobs.close()
        self._evidence.close()
        if self._server is not None:
            self._server.close()
        if self._notebook is not None:
            self._notebook.close()

    def _state(self) -> dict[str, Any]:
        with self._lock:
            self.terms = session_payload(self.session, self.control_counts)
            state = {
                "model_revision": self.session.model_revision,
                "fit_token": self._current_fit_token(),
                "selected_term": self.selected_term,
                "terms": self.terms,
                "selection": {
                    name: self.session.selection(name).astype(int).tolist()
                    for name in self.session.terms
                },
                "undo_redo": undo_redo_payload(self.session),
                "timeline": timeline_payload(self.session),
                "pending": pending_payload(self.session),
                # With the live edits, this says whether Revert has anything to
                # change: a structural step or a re-profile each make it False.
                "in_force_is_original": self.session.model is self.session.reference_model,
                # Export offers the Final fit model while it is current (D6).
                "final_fit": {
                    "available": self._final_fit is not None,
                    "stale": self._final_fit is not None
                    and self._final_fit.model_revision != self.session.model_revision,
                },
            }
            self._state_generation += 1
            state["state_generation"] = self._state_generation
            state["chart_generation"] = self._chart_generation
            return state

    def _select_term(self, term: str) -> None:
        if term not in self.session.terms:
            raise EditorKeyError(f"Unknown editable term: {term!r}")
        self.selected_term = term

    def _set_term(self, term: str) -> dict[str, Any]:
        with self._lock:
            self._select_term(term)
            return self._state()

    def _select(self, term: str, indices: list[int]) -> dict[str, Any]:
        with self._lock:
            self._select_term(term)
            self.session.select_indices(term, indices)
            return self._state()

    def _operate(self, operation: str, term: str | None = None) -> dict[str, Any]:
        with self._lock:
            if term is not None:
                self._select_term(term)
            target = self.selected_term
            if operation == "shift_up":
                self.session.shift(target, float(np.log(1.05)))
            elif operation == "shift_down":
                self.session.shift(target, float(np.log(0.95)))
            elif operation in {"isotonic_increasing", "increasing"}:
                self.session.isotonic(target, "increasing")
            elif operation in {"isotonic_decreasing", "decreasing"}:
                self.session.isotonic(target, "decreasing")
            elif operation == "smooth":
                self.session.smooth(target, 1.0)
            elif operation == "average":
                self._average_relativity(target)
            elif operation in {"linearise", "linearize"}:
                self.session.linear_interpolate(target, strength=0.5)
            elif operation == "level_left":
                self.session.level_left(target)
            elif operation == "level_right":
                self.session.level_right(target)
            elif operation == "snap_highest":
                self.session.snap_highest(target)
            elif operation == "snap_lowest":
                self.session.snap_lowest(target)
            elif operation == "reset_order":
                self.session.reset_level_order(target)
            elif operation == "reset":
                self.session.reset(target)
            elif operation == "select_all":
                editable = self.session.terms[target]
                self.session.select_indices(target, np.arange(editable.size, dtype=np.intp))
            elif operation == "undo":
                self.session.undo()
            elif operation == "redo":
                self.session.redo()
            else:
                raise EditorValueError(f"Unknown editor operation: {operation!r}")
            # A fixed-offset refit is conditional on the current edited factors,
            # so any value-changing edit invalidates the stored refit result.
            if operation not in {"select_all", "reset_order"}:
                self._invalidate_refit()
            if operation != "select_all":
                self._chart_generation += 1
            return self._state()

    def _drag(
        self,
        term: str,
        indices: list[int],
        delta: float = 0.0,
        values: list[float] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._select_term(term)
            self.session.select_indices(term, indices)
            if values is None:
                self.session.shift(term, float(delta))
            else:
                rel = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
                self.session.set_values(term, indices, np.log(rel))
            self._invalidate_refit()
            self._chart_generation += 1
            return self._state()

    def _control(
        self,
        term: str,
        handle_index: int,
        value: float,
        handle_count: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._select_term(term)
            if handle_count is not None:
                controls = self.session.control_points(term, n_handles=int(handle_count))
                self.control_counts[term] = int(controls["x"].size)
            self.session.move_control_point(
                term,
                int(handle_index),
                float(np.log(max(value, 1e-12))),
                n_handles=self.control_counts.get(term),
            )
            self._invalidate_refit()
            self._chart_generation += 1
            return self._state()

    def _set_control_count(self, term: str, count: int) -> dict[str, Any]:
        with self._lock:
            self._select_term(term)
            controls = self.session.control_points(term, n_handles=int(count))
            self.control_counts[term] = int(controls["x"].size)
            self._chart_generation += 1
            return self._state()

    def _average_relativity(self, term: str) -> None:
        # Average on the displayed relativity scale. This matches analyst
        # expectations for leveling factors better than averaging log effects.
        idx = self.session.selection(term)
        if idx.size == 0:
            raise EditorValueError(f"No points selected for term {term!r}.")
        idx = self.session._expand_collapsed_level_indices(term, idx)
        editable = self.session.terms[term]
        weights = np.ones(idx.size, dtype=np.float64)
        if editable.weights is not None:
            weights = np.asarray(editable.weights[idx], dtype=np.float64)
        if float(np.sum(weights)) <= 0.0:
            weights = None
        value = float(np.average(np.exp(editable.edited_log_effect[idx]), weights=weights))
        self.session.set_values(term, idx, np.full(idx.size, np.log(max(value, 1e-12))))

    def _current_model_for_evidence(self):
        """Resolve one current model without materializing under the widget lock."""
        with self._lock:
            revision = self.session.model_revision
            if not self.session.edited_terms():
                return self.session.model, revision
            epoch = self.session.edit_epoch
            base_model = self.session.model
            cached = self.session.cached_materialized_model(
                epoch,
                model_revision=revision,
                base_model=base_model,
            )
            if cached is not None:
                return cached, revision
            request = self.session.capture_materialization_request()
            assert request is not None

        key = EvidenceKey(request.model_revision, "materialize", "current")
        outcome = self._evidence.submit(
            key,
            lambda request=request: {"model": materialize_edit_request(request)},
        ).result()
        if outcome.status == "superseded" or outcome.payload is None:
            return None, request.model_revision
        model = outcome.payload["model"]

        with self._lock:
            if not self.session.publish_materialized_model(request, model):
                return None, request.model_revision
            return model, request.model_revision

    def _model_materialized_for_dataset(self, dataset):
        """Materialize a revision against one purpose-specific dataset without caching it."""
        with self._lock:
            request = self.session.capture_materialization_request(
                dataset=dataset,
                include_unchanged=True,
            )
            assert request is not None

        model = materialize_edit_request(request)

        with self._lock:
            if not self.session.materialization_request_is_current(request):
                return None, request.model_revision
            return model, request.model_revision

    def _metrics(
        self,
        metric: str,
        source: str | None = None,
        *,
        dataset: str | None = None,
        model_revision: int | None = None,
        request_sequence: int | None = None,
    ) -> dict[str, Any]:
        selected_source = "in_force" if source is None or source == "selected" else source
        with self._lock:
            current_revision = self.session.model_revision
            if model_revision is not None and int(model_revision) != current_revision:
                return _superseded_payload(int(model_revision), request_sequence)
            reference_model = getattr(self.session, "reference_model", self.session.model)

        if selected_source == "original":
            selected_model = reference_model
            revision = current_revision
        else:
            selected_model, revision = self._current_model_for_evidence()
        if selected_model is None:
            return _superseded_payload(revision, request_sequence)

        with self._lock:
            if revision != self.session.model_revision:
                return _superseded_payload(revision, request_sequence)
            reference_model = getattr(self.session, "reference_model", self.session.model)
            eval_dataset = named_metrics_dataset(self.session, dataset)
            if reference_model is None or eval_dataset is None:
                return metrics_payload(
                    self.session,
                    metric,
                    source=selected_source,
                    dataset=dataset,
                    model_revision=revision,
                    request_sequence=request_sequence,
                )

        original_metrics = self._dataset_metrics_for_evidence(
            "original", reference_model, eval_dataset, revision
        )
        if original_metrics is None:
            return _superseded_payload(revision, request_sequence)
        edited_metrics: dict[str, float]
        if selected_source == "original":
            edited_metrics = original_metrics
        else:
            current_metrics = self._dataset_metrics_for_evidence(
                "current", selected_model, eval_dataset, revision
            )
            if current_metrics is None:
                return _superseded_payload(revision, request_sequence)
            edited_metrics = current_metrics
        payload = metric_comparison_payload(
            metric,
            eval_dataset,
            original_metrics,
            edited_metrics,
            model_revision=revision,
            request_sequence=request_sequence,
        )
        with self._lock:
            if revision != self.session.model_revision:
                return _superseded_payload(revision, request_sequence)
        return payload

    def _dataset_metrics_for_evidence(
        self,
        role: Literal["original", "current"],
        model,
        dataset,
        revision: int,
    ) -> dict[str, float] | None:
        with self._lock:
            if revision != self.session.model_revision:
                return None
            self._evaluation_cache.advance_current_revision(revision)
            key = EvaluationKey(
                role=role,
                model_revision=0 if role == "original" else revision,
                dataset_epoch=0,
                split=dataset.name,
                metric_signature=model_metric_signature(model),
            )
            cached = self._evaluation_cache.get(key)
            if cached is not None:
                return cached
            request = DatasetMetricRequest(key=key, model=model, dataset=dataset)

        evidence_key = EvidenceKey(
            revision,
            "score",
            f"{role}:{dataset.name}:{request.key.metric_signature!r}",
        )

        def compute_metrics(request: DatasetMetricRequest = request) -> dict[str, Any]:
            return {
                "metrics": metrics_module.compute_dataset_metrics(
                    request.model,
                    request.dataset,
                )
            }

        outcome = self._evidence.submit(evidence_key, compute_metrics).result()
        if outcome.status == "superseded" or outcome.payload is None:
            return None
        values = outcome.payload["metrics"]
        with self._lock:
            if revision != self.session.model_revision:
                return None
            self._evaluation_cache.advance_current_revision(revision)
            self._evaluation_cache.put(request.key, values)
            return dict(values)

    def _summary(
        self,
        source: str,
        *,
        level_display: str = "expanded",
        model_revision: int | None = None,
        request_sequence: int | None = None,
    ) -> dict[str, Any]:
        if source == "selected":
            source = "in_force"
        if source in {"edited", "collapse"}:
            source = "in_force"
        source = source if source in {"original", "in_force", "refit"} else "in_force"

        with self._lock:
            current_revision = self.session.model_revision
            if model_revision is not None and int(model_revision) != current_revision:
                return _superseded_payload(int(model_revision), request_sequence)
            # A notebook-side edit advances the revision without passing through
            # this widget, so a stored fixed-offset refit can outlive its edits.
            if self._offset_refit_revision != current_revision:
                self._invalidate_refit()

        offset_terms_override: list[str] | None = None
        offset_labels_override: list[dict[str, Any]] | None = None
        if source == "in_force":
            model, revision = self._current_model_for_evidence()
            if model is None:
                return _superseded_payload(revision, request_sequence)
            with self._lock:
                if revision != self.session.model_revision:
                    return _superseded_payload(revision, request_sequence)
        else:
            with self._lock:
                revision = self.session.model_revision
                if source == "original":
                    model = self.session.reference_model
                else:
                    model = self._offset_refit_model
                    offset_terms_override = list(self._offset_refit_terms)
                    offset_labels_override = list(self._offset_refit_labels)

        payload = summary_payload(
            self,
            source,
            model_override=model,
            offset_terms_override=offset_terms_override,
            offset_labels_override=offset_labels_override,
            level_display=level_display,
        )
        with self._lock:
            if revision != self.session.model_revision:
                return _superseded_payload(revision, request_sequence)
        payload["model_revision"] = revision
        payload["request_sequence"] = request_sequence
        return payload

    def _report(
        self,
        report: str = "validation",
        *,
        model_revision: int | None = None,
        request_sequence: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            current_revision = self.session.model_revision
            if model_revision is not None and int(model_revision) != current_revision:
                return _superseded_payload(int(model_revision), request_sequence)
        if report == "cv":
            payload = report_payload(self, report, request_sequence=request_sequence)
            with self._lock:
                if payload["model_revision"] != self.session.model_revision:
                    return _superseded_payload(payload["model_revision"], request_sequence)
            return payload
        edited_model, revision = self._current_model_for_evidence()
        if edited_model is None:
            return _superseded_payload(revision, request_sequence)

        with self._lock:
            if revision != self.session.model_revision:
                return _superseded_payload(revision, request_sequence)
            datasets = tuple(evaluation_datasets(self.session))
            final_fit = self._final_fit
            reference_model = getattr(self.session, "reference_model", self.session.model)
            if reference_model is None:
                return report_payload(
                    self,
                    report,
                    model_revision=revision,
                    request_sequence=request_sequence,
                )

        summary_model = edited_model
        if report == "final" and self.session.edited_terms():
            fit_dataset = training_export_dataset(self.session)
            metrics_dataset = default_metrics_dataset(self.session)
            if fit_dataset is not None and (
                metrics_dataset is None or fit_dataset.name != metrics_dataset.name
            ):
                summary_model, summary_revision = self._model_materialized_for_dataset(fit_dataset)
                if summary_model is None or summary_revision != revision:
                    return _superseded_payload(summary_revision, request_sequence)

        metric_pairs: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
        for eval_dataset in datasets:
            original = self._dataset_metrics_for_evidence(
                "original", reference_model, eval_dataset, revision
            )
            if original is None:
                return _superseded_payload(revision, request_sequence)
            edited = self._dataset_metrics_for_evidence(
                "current", edited_model, eval_dataset, revision
            )
            if edited is None:
                return _superseded_payload(revision, request_sequence)
            metric_pairs[eval_dataset.name] = (original, edited)
        splits = split_metrics_payload(datasets, metric_pairs)
        payload = report_payload(
            self,
            report,
            splits=splits,
            model_revision=revision,
            request_sequence=request_sequence,
            model_override=summary_model,
            final_fit=final_fit,
        )
        with self._lock:
            if revision != self.session.model_revision:
                return _superseded_payload(revision, request_sequence)
        return payload

    def _save_model(
        self,
        *,
        directory: str | None = None,
        filename: str | None = None,
        path: str | None = None,
    ) -> dict[str, Any]:
        payload = self._export_file(
            "joblib",
            directory="." if directory is None else directory,
            filename=filename,
            path=path,
        )
        payload["message"] = f"Saved edited model to {payload['path']}"
        return payload

    def _export_bytes(self, format: str, filename: str | None = None) -> ExportResult:
        """Build one validated export from one captured model revision."""
        canonical_format = _normalise_export_format(format)
        safe_name = _safe_export_filename(canonical_format, filename)

        validation_scope: str | None = None
        if canonical_format == "structure":
            # No fit and no rows to score: built under the lock, on one revision.
            with self._lock:
                revision = self.session.model_revision
                data = self.session.export_structure().encode("utf-8")
        elif canonical_format in {"joblib", "final"}:
            model, revision = (
                self._final_fit_for_export()
                if canonical_format == "final"
                else self._current_model_for_evidence()
            )
            if model is None:
                raise RuntimeError("Export request was superseded.")
            with self._lock:
                history = self.session.editor_history_records()
            data, validation = persistence.serialize_validated_model(
                persistence.with_editor_history(model, history),
                dataset=default_metrics_dataset(self.session),
            )
            validation_scope = validation.scope
        else:
            payload, revision = self._rating_table_payload()
            if payload is None:
                raise RuntimeError("Export request was superseded.")
            from superglm.export.excel import write_rating_table_workbook

            buffer = io.BytesIO()
            write_rating_table_workbook(
                payload,
                buffer,
                sheet_name="Rating Tables",
                summary_sheet_name="Model Summary",
                impact_sheet_name="Discretization Impact",
            )
            data = buffer.getvalue()

        with self._lock:
            if revision != self.session.model_revision:
                raise RuntimeError("Export request was superseded by a newer model revision.")
        return ExportResult(
            data=data,
            filename=safe_name,
            media_type=_EXPORT_MEDIA_TYPES[canonical_format],
            model_revision=revision,
            validation_scope=validation_scope,
        )

    def _rating_table_payload(self, *, impact_bins: tuple[int, ...] | None = None):
        """The Excel export's rating-table payload for the current revision.

        One path for the workbook and its preview: the training split, the
        model materialised on it, and the builder's defaults, except the
        ``impact_bins`` the preview passes. ``(None, revision)`` when the
        revision moved while the model was materialised.
        """
        dataset = training_export_dataset(self.session)
        if dataset is None:
            raise EditorValueError(_EXCEL_NEEDS_TRAINING_DATA)
        model, revision = self._model_materialized_for_dataset(dataset)
        if model is None:
            return None, revision
        from superglm.export.rating_tables import build_rating_table_payload

        options: dict[str, Any] = {} if impact_bins is None else {"impact_bins": impact_bins}
        payload = build_rating_table_payload(
            model,
            dataset.X,
            dataset.y,
            sample_weight=dataset.sample_weight,
            offset=dataset.offset,
            **options,
        )
        return payload, revision

    def _current_fit_token(self) -> int:
        """A token that changes only when the fitted model in force is replaced."""
        if self.session.model is not self._fit_model:
            self._fit_model = self.session.model
            self._fit_token += 1
            self._free_levels = {}
        return self._fit_token

    def _free_level_comparison(self, term: str) -> dict[str, Any]:
        """``term``'s curve beside its levels fitted free, refitted once per fit in force.

        The refit is a full fit, like Refit, and holds the session while it runs.
        """
        from superglm.editor.free_levels import free_level_comparison

        with self._lock:
            token = self._current_fit_token()
            if term not in self._free_levels:
                self._free_levels[term] = free_level_comparison(self.session, term)
            return {**self._free_levels[term], "fit_token": token}

    def _rating_table(self, term: str) -> dict[str, Any]:
        """``term``'s block of the Excel rating table, for the Table view.

        Built once per model revision, outside the lock like the export, and
        reused while the revision stands.
        """
        with self._lock:
            if term not in self.session.terms:
                raise EditorKeyError(f"Unknown editable term: {term!r}")
            revision = self.session.model_revision
            preview = self._rating_preview
        if preview is None or preview.model_revision != revision:
            preview = self._build_rating_preview(revision)
            with self._lock:
                if preview.model_revision == self.session.model_revision:
                    self._rating_preview = preview
        return rating_preview.term_rating_table(preview, term)

    def _build_rating_preview(self, revision: int) -> RatingPreview:
        try:
            payload, revision = self._rating_table_payload(impact_bins=PREVIEW_IMPACT_BINS)
        except EditorClientError as exc:
            return RatingPreview(revision, None, exc.public_message)
        except (NotImplementedError, OverflowError, ValueError):
            # The builder's refusals carry backend text; the browser gets a
            # fixed sentence and the log keeps the cause.
            _LOGGER.info("The rating-table preview was refused.", exc_info=True)
            return RatingPreview(revision, None, rating_preview.refusal_reason(self.session.model))
        if payload is None:
            return RatingPreview(revision, None, rating_preview.SUPERSEDED)
        return RatingPreview(revision, payload, None)

    def _final_fit_for_export(self):
        """The Final fit model and its revision; refused before a run or once stale."""
        with self._lock:
            final = self._final_fit
            if final is None:
                raise EditorValueError(FINAL_NOT_RUN)
            if final.model_revision != self.session.model_revision:
                raise EditorValueError(FINAL_STALE)
            return final.model, final.model_revision

    def _export_file(
        self,
        format: str,
        *,
        directory: str = ".",
        filename: str | None = None,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Atomically write a completed export while its revision is current."""
        canonical_format = _normalise_export_format(format)
        if path:
            requested_path = Path(path).expanduser()
            safe_name = _safe_export_filename(canonical_format, requested_path.name)
            target = requested_path.with_name(safe_name)
        else:
            safe_name = _safe_export_filename(canonical_format, filename)
            target = Path(directory).expanduser() / safe_name

        result = self._export_bytes(canonical_format, target.name)
        with self._lock:
            if result.model_revision != self.session.model_revision:
                raise RuntimeError("Export request was superseded by a newer model revision.")
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(result.data)

        saved_path = target.resolve()
        return {
            "path": str(saved_path),
            "directory": str(saved_path.parent),
            "filename": saved_path.name,
            "message": f"Exported {saved_path.name} to {saved_path.parent}",
            "model_revision": result.model_revision,
            "validation_scope": result.validation_scope,
        }

    def _download_model(self, filename: str | None = None) -> tuple[bytes, str]:
        result = self._export_bytes("joblib", filename)
        return result.data, result.filename

    def _open_directory(self, path: str | None = None) -> dict[str, str]:
        opened = open_directory_path(path)
        return {"path": str(opened)}

    def _save_directory(self, path: str | None = None) -> dict[str, Any]:
        target = Path(path).expanduser() if path else Path.cwd()
        resolved = target.resolve()
        if not resolved.exists():
            raise EditorValueError("The selected directory does not exist.")
        if not resolved.is_dir():
            resolved = resolved.parent
        entries: list[dict[str, str]] = []
        for child in resolved.iterdir():
            try:
                if child.is_dir():
                    entries.append(
                        {
                            "kind": "directory",
                            "name": child.name,
                            "path": str(child.resolve()),
                        }
                    )
                elif child.is_file():
                    entries.append(
                        {
                            "kind": "file",
                            "name": child.name,
                            "path": str(child.resolve()),
                        }
                    )
            except OSError:
                continue
        entries.sort(key=lambda item: (item["kind"] != "directory", item["name"].casefold()))
        parent = resolved.parent
        return {
            "cwd": str(Path.cwd().resolve()),
            "path": str(resolved),
            "parent": None if parent == resolved else str(parent),
            "entries": entries,
        }

    def _refit_offset(
        self,
        method: str = "auto",
        *,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        level_display = validate_level_display(level_display)
        with self._lock:
            terms = self.session.edited_terms()
            if not terms:
                return {
                    "available": False,
                    "source": "refit",
                    "level_display": level_display,
                    "error": "No edited terms are available for a fixed-offset refit.",
                }
            refit_model = self.session.refit_with_edited_offset(method=method)
            self._offset_refit_model = refit_model
            self._offset_refit_terms = list(terms)
            self._offset_refit_labels = offset_label_payload(self.session, terms)
            self._offset_refit_revision = self.session.model_revision
            return summary_payload(self, "refit", level_display=level_display)

    def _profile_distribution(
        self,
        parameter: str,
        *,
        level_display: str = "expanded",
        progress_callback=None,
        **options: Any,
    ) -> dict[str, Any]:
        level_display = validate_level_display(level_display)
        with self._lock:
            profile_options = dict(options)
            if progress_callback is not None:
                profile_options["progress_callback"] = progress_callback
            result = self.session.reprofile_distribution(parameter, **profile_options)
            estimate = _profile_estimate_payload(result, parameter)
            if progress_callback is not None:
                progress_callback("finalizing", {"profile_estimate": estimate})
            if self.selected_term not in self.session.terms:
                self.selected_term = next(iter(self.session.terms), "")
            payload = summary_payload(self, "in_force", level_display=level_display)
            payload["profile_trace"] = _profile_trace_rows(result)
            payload["profile_estimate"] = estimate
            return payload

    def _start_profile_distribution_job(
        self,
        parameter: str,
        *,
        level_display: str = "expanded",
        **options: Any,
    ) -> dict[str, Any]:
        """Start a distribution-parameter profile job and return its status payload."""
        level_display = validate_level_display(level_display)
        with self._profile_condition:
            self._profile_job_counter += 1
            job_id = str(self._profile_job_counter)
            job = {
                "job_id": job_id,
                "parameter": parameter,
                "level_display": level_display,
                "options": jsonable(dict(options)),
                "status": "running",
                "phase": "profiling",
                "trace": [],
                "result": None,
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
            }
            self._profile_jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_profile_distribution_job,
            args=(job_id, parameter, level_display, dict(options)),
            name=f"superglm-profile-{job_id}",
            daemon=True,
        )
        thread.start()
        return self._profile_distribution_status(job_id)

    def _profile_distribution_status(self, job_id: str, *, wait: bool = False) -> dict[str, Any]:
        """Return the current profile job status, optionally waiting for completion."""
        with self._profile_condition:
            job = self._profile_jobs.get(str(job_id))
            if job is None:
                raise EditorKeyError(f"Unknown profile job: {job_id!r}")
            if wait:
                deadline = time.monotonic() + 30.0
                while job["status"] == "running":
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._profile_condition.wait(timeout=min(remaining, 0.25))
            return jsonable(dict(job))

    def _run_profile_distribution_job(
        self,
        job_id: str,
        parameter: str,
        level_display: str,
        options: dict[str, Any],
    ) -> None:
        def progress_callback(phase: str, payload: dict[str, Any] | None = None) -> None:
            payload = payload or {}
            with self._profile_condition:
                job = self._profile_jobs[job_id]
                job["phase"] = phase
                if "profile_estimate" in payload:
                    job["profile_estimate"] = _normalise_profile_estimate(
                        payload["profile_estimate"]
                    )
                # Live rows count the feasible candidates in search order, as
                # the completed trace does.
                trace = job["trace"]
                for row in payload.get("profile_trace", ()):
                    trace.append(jsonable({"step": len(trace), **row}))
                self._profile_condition.notify_all()

        try:
            payload = self._profile_distribution(
                parameter,
                level_display=level_display,
                progress_callback=progress_callback,
                **options,
            )
        except BaseException as exc:
            if isinstance(exc, EditorClientError):
                message = exc.public_message
            else:
                _LOGGER.exception("Unhandled SuperGLM editor profile error.")
                message = "internal editor error"
            with self._profile_condition:
                job = self._profile_jobs[job_id]
                job["status"] = "error"
                job["phase"] = "error"
                job["error"] = message
                job["finished_at"] = time.time()
                self._profile_condition.notify_all()
            return

        with self._profile_condition:
            job = self._profile_jobs[job_id]
            job["trace"] = payload["profile_trace"]
            job["status"] = "complete"
            job["phase"] = "complete"
            job["result"] = payload
            if "profile_estimate" in payload:
                job["profile_estimate"] = _normalise_profile_estimate(payload["profile_estimate"])
            job["finished_at"] = time.time()
            self._profile_condition.notify_all()
        # The profile replaced the in-force model, its selection and history.
        self._notify_changed()

    def _notify_changed(self) -> None:
        """Tell every in-notebook view that a job changed what they show."""
        if self._notebook is not None:
            self._notebook.notify_changed()

    def _job_start(self, kind: str) -> dict[str, Any]:
        """Capture a job's inputs under the lock, then start it off the lock."""
        starter = self._job_starters.get(kind)
        if starter is None:
            raise EditorValueError("Unknown job kind.")
        with self._lock:
            work, publish = starter()
        return self._jobs.status(self._jobs.start(kind, work, publish))

    def _cv_job(self):
        """Run CV on the in-force structure (D5, D7); the caller holds the lock."""
        plan = capture_cv_run(self.session)

        def publish(run: CVRun) -> dict[str, Any]:
            with self._lock:
                if not plan.is_current(self.session):
                    raise EditorValueError(SUPERSEDED)
                self._cv_run = run
            self._notify_changed()
            return {"model_revision": plan.model_revision, "n_folds": len(plan.folds)}

        return (lambda context: run_cv(plan, context)), publish

    def _final_fit_job(self):
        """Final fit on train and validation rows (D6); the caller holds the lock."""
        plan = capture_final_fit(self.session)

        def publish(final: FinalFit) -> dict[str, Any]:
            with self._lock:
                if not plan.is_current(self.session):
                    raise EditorValueError(SUPERSEDED)
                self._final_fit = final
            self._notify_changed()
            return {"model_revision": plan.model_revision, "n_rows": final.n_rows}

        return (lambda context: run_final_fit(plan, context)), publish

    def _job_status(self, job_id: str, *, wait: bool = False) -> dict[str, Any]:
        return self._jobs.status(job_id, wait=wait)

    def _job_cancel(self, job_id: str) -> dict[str, Any]:
        return self._jobs.cancel(job_id)

    def _structural_transition(
        self,
        operation: str,
        *,
        operation_start: float,
        fit_start: float,
        fit_end: float,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        with self._lock:
            summary_start = time.perf_counter()
            summary = jsonable(summary_payload(self, "in_force", level_display=level_display))
            summary_end = time.perf_counter()
            state_start = time.perf_counter()
            state = jsonable(self._state())
            state_end = time.perf_counter()
            return {
                "state": state,
                "summary": summary,
                "timing": {
                    "operation": operation,
                    "fit_ms": _elapsed_ms(fit_start, fit_end),
                    "summary_ms": _elapsed_ms(summary_start, summary_end),
                    "state_ms": _elapsed_ms(state_start, state_end),
                    "server_total_ms": _elapsed_ms(operation_start, state_end),
                },
            }

    def _structural_step(
        self,
        operation: str,
        apply: Callable[[str], Any],
        *,
        term: str | None = None,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        """Run one structural session change and return its atomic transition envelope."""
        level_display = validate_level_display(level_display)
        with self._lock:
            operation_start = time.perf_counter()
            if term is not None:
                self._select_term(term)
            target = self.selected_term
            selected_indices = self.session.selection(target).astype(int).tolist()
            selected_levels = self._selected_level_labels(target)
            fit_start = time.perf_counter()
            apply(target)
            fit_end = time.perf_counter()
            self._invalidate_refit()
            self._restore_selection(target, selected_levels, selected_indices)
            self._chart_generation += 1
            return self._structural_transition(
                operation,
                operation_start=operation_start,
                fit_start=fit_start,
                fit_end=fit_end,
                level_display=level_display,
            )

    def _collapse_levels(
        self,
        term: str | None = None,
        method: str = "auto",
        *,
        level_display: str = "expanded",
        keep_reference: bool = True,
    ) -> dict[str, Any]:
        return self._structural_step(
            "collapse_levels",
            lambda target: self.session.replace_with_collapsed_levels(
                target, method=method, keep_reference=keep_reference
            ),
            term=term,
            level_display=level_display,
        )

    def _ungroup_levels(
        self,
        term: str | None = None,
        method: str = "auto",
        *,
        level_display: str = "expanded",
        keep_reference: bool = True,
    ) -> dict[str, Any]:
        return self._structural_step(
            "ungroup_levels",
            lambda target: self.session.replace_with_ungrouped_levels(
                target, method=method, keep_reference=keep_reference
            ),
            term=term,
            level_display=level_display,
        )

    def _set_reference(
        self,
        term: str,
        level: str,
        method: str = "auto",
        *,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        return self._structural_step(
            "set_reference",
            lambda target: self.session.replace_with_reference_level(target, level, method=method),
            term=term,
            level_display=level_display,
        )

    def _special_levels(
        self,
        term: str,
        levels: list[str],
        *,
        special: bool = True,
        method: str = "auto",
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        """Take levels off an ordered term's curve, or put them back, and refit at once."""
        return self._structural_step(
            "special_levels" if special else "on_curve_levels",
            lambda target: self.session.replace_with_special_levels(
                target, levels, special=special, method=method
            ),
            term=term,
            level_display=level_display,
        )

    def _knots(
        self,
        term: str,
        params: dict[str, Any],
        *,
        method: str = "auto",
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        """Change a spline term's knots and refit at once."""
        return self._structural_step(
            "set_knots",
            lambda target: self.session.replace_with_knots(target, params, method=method),
            term=term,
            level_display=level_display,
        )

    def _shape_range(
        self,
        term: str,
        *,
        lo: str | float,
        hi: str | float,
        degree: int,
        join: str = "tangent",
        method: str = "auto",
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        return self._structural_step(
            "shape_range",
            lambda target: self.session.replace_with_shaped_range(
                target, lo=lo, hi=hi, degree=degree, join=join, method=method
            ),
            term=term,
            level_display=level_display,
        )

    def _stage(
        self,
        operation: str,
        term: str,
        params: dict[str, Any],
        *,
        keep_reference: bool = True,
        level_display: str = "expanded",
    ) -> dict[str, Any]:
        """Stage one structural change and return its transition envelope.

        Nothing is fitted, so the model revision, its evidence and any
        fixed-offset refit all stand; only the chart redraws what waits.
        """
        level_display = validate_level_display(level_display)
        with self._lock:
            operation_start = time.perf_counter()
            self._select_term(term)
            stage_start = time.perf_counter()
            self.session.stage_structural(operation, term, params, keep_reference=keep_reference)
            stage_end = time.perf_counter()
            self._chart_generation += 1
            return self._structural_transition(
                "stage",
                operation_start=operation_start,
                fit_start=stage_start,
                fit_end=stage_end,
                level_display=level_display,
            )

    def _refit_pending(self, *, level_display: str = "expanded") -> dict[str, Any]:
        """Refit every waiting change in one fit and return the transition envelope."""
        return self._structural_step(
            "refit_pending",
            lambda _target: self.session.refit_pending(),
            level_display=level_display,
        )

    def _set_unseen(self, term: str, unseen: str) -> dict[str, Any]:
        """Choose where the term's new levels go; nothing is refit (spec addendum S6)."""
        with self._lock:
            self._select_term(term)
            self.session.set_unseen(term, unseen)
            # A fixed-offset refit was cloned with the policy it replaces.
            self._invalidate_refit()
            return self._state()

    def _set_note(self, step_id: str, note: str) -> dict[str, Any]:
        """Write a note on one timeline entry; a note changes no model, so nothing is refit."""
        with self._lock:
            self.session.set_step_note(step_id, note)
            return {"ok": True, "state": self._state()}

    def _reorder_levels(self, term: str | None = None, target_index: int = 0) -> dict[str, Any]:
        with self._lock:
            if term is not None:
                self._select_term(term)
            self.session.reorder_levels(self.selected_term, target_index=int(target_index))
            self._chart_generation += 1
            return self._state()

    def _revert_to_original(self, *, level_display: str = "expanded") -> dict[str, Any]:
        return self._structural_step(
            "revert_to_original",
            lambda _target: self.session.revert_to_reference_model(),
            level_display=level_display,
        )

    def _selected_level_labels(self, term: str) -> list[str]:
        editable = self.session.terms.get(term)
        if editable is None or editable.levels is None:
            return []
        labels = editable.levels
        return [
            labels[int(index)]
            for index in self.session.selection(term)
            if 0 <= int(index) < len(labels)
        ]

    def _restore_selection(
        self,
        term: str,
        levels: list[str],
        fallback_indices: list[int],
    ) -> None:
        editable = self.session.terms.get(term)
        if editable is None:
            return
        if editable.levels is not None and levels:
            available = set(editable.levels)
            restored_levels = [level for level in levels if level in available]
            if restored_levels:
                self.session.select_levels(term, restored_levels)
                return
        valid_indices = [
            int(index) for index in fallback_indices if 0 <= int(index) < editable.size
        ]
        if valid_indices:
            self.session.select_indices(term, valid_indices)

    def _invalidate_refit(self) -> None:
        self._offset_refit_model = None
        self._offset_refit_terms = []
        self._offset_refit_labels = []
        self._offset_refit_revision = None


def _superseded_payload(
    model_revision: int,
    request_sequence: int | None,
) -> dict[str, Any]:
    return {
        "status": "superseded",
        "model_revision": int(model_revision),
        "request_sequence": request_sequence,
    }


_NOTEBOOK_FALLBACK = (
    "On Databricks the editor runs inside the notebook cell, which needs anywidget, and "
    "anywidget is not installed. The editor uses its local server instead, which a "
    "Databricks browser cannot reach. Install it with: %pip install 'superglm[notebook]'"
)


def _display_mode(mode: str | None) -> str:
    """The editor's display mode: ``mode``, else notebook on Databricks.

    A Databricks notebook's browser never reaches the cluster's local
    addresses, so the local server's page cannot load there. Without
    anywidget the automatic choice stays the local server, as before notebook
    mode existed, and warns how to install it.
    """
    if mode is None:
        if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
            return "server"
        try:
            notebook_view_class()
        except ImportError:
            warnings.warn(_NOTEBOOK_FALLBACK, UserWarning, stacklevel=4)
            return "server"
        return "notebook"
    if mode not in {"server", "notebook"}:
        raise EditorValueError("mode must be 'server', 'notebook' or None.")
    return mode


def _close_live_widgets() -> None:
    for widget in list(_LIVE_WIDGETS):
        widget.close()


def _profile_trace_rows(result: Any) -> list[dict[str, Any]]:
    evaluations = result.evaluations
    # An infeasible power has no objective to plot.
    feasible = evaluations[np.isfinite(evaluations["nll"])]
    return [jsonable({"step": step, **row}) for step, row in enumerate(feasible.to_dict("records"))]


def _profile_estimate_payload(result: Any, parameter: str) -> dict[str, Any]:
    key = parameter.lower().replace("-", "_")
    is_tweedie = key in {"tweedie", "tweedie_p", "p"}
    if is_tweedie:
        value = getattr(result, "p_hat", None)
        label = "p_hat"
        name = "p"
    elif key in {"nb2", "nb2_theta", "negative_binomial", "theta"}:
        value = getattr(result, "theta_hat", None)
        label = "theta_hat"
        name = "theta"
    else:
        value = None
        label = parameter
        name = parameter

    ci_low = None
    ci_high = None
    ci_status = None
    if is_tweedie:
        cached_ci, ci_status = cached_tweedie_profile_ci(result, 0.05)
        if cached_ci is not None:
            ci_low, ci_high = cached_ci
    elif name == "theta":
        # The recorded interval keeps its censoring and caution, which ci()
        # drops, and reading it raises no warning.
        try:
            (ci_low, ci_high), ci_status = reported_interval(
                result._interval(0.05), caution=profile_cautioned(result, 0.05)
            )
        except Exception:
            ci_status = "not computed"

    estimate = {
        "parameter": name,
        "label": label,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "objective": getattr(result, "nll", None),
        "objective_label": "loss",
        "lower_is_better": True,
    }
    if ci_status is not None:
        estimate["ci_status"] = ci_status
    return _normalise_profile_estimate(estimate)


def _normalise_profile_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(estimate)
    payload.setdefault("parameter", "")
    payload.setdefault("label", str(payload.get("parameter") or "estimate"))
    payload.setdefault("value", None)
    payload.setdefault("ci_low", None)
    payload.setdefault("ci_high", None)
    if payload.get("parameter") == "p":
        payload.setdefault(
            "ci_status",
            "available"
            if payload.get("ci_low") is not None and payload.get("ci_high") is not None
            else "not computed",
        )
    payload.setdefault("objective", None)
    payload.setdefault("objective_label", "loss")
    payload.setdefault("lower_is_better", True)
    return jsonable(payload)


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


atexit.register(_close_live_widgets)


__all__ = ["EditorWidget"]
