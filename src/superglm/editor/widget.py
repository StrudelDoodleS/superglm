"""Notebook/editor frontend for editor sessions.

The default renderer is a tiny local HTML app served by the Python kernel. This
avoids custom Jupyter widget modules, which are often unavailable in VS Code.
"""

from __future__ import annotations

import atexit
import html
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from superglm.editor import metrics as metrics_module
from superglm.editor import persistence
from superglm.editor.apply import materialize_edit_request
from superglm.editor.evaluation import evaluation_datasets, named_metrics_dataset
from superglm.editor.evaluation_cache import (
    DatasetMetricRequest,
    EvaluationCache,
    EvaluationKey,
    model_metric_signature,
)
from superglm.editor.evidence import EvidenceCoordinator, EvidenceKey
from superglm.editor.io import jsonable
from superglm.editor.metrics import metric_comparison_payload, metrics_payload
from superglm.editor.native_dialogs import open_directory_path
from superglm.editor.payloads import history_payload, session_payload
from superglm.editor.persistence import edited_model_for_export
from superglm.editor.reports import report_payload, split_metrics_payload
from superglm.editor.server import EditorAppServer
from superglm.editor.summaries import offset_label_payload, summary_payload

_LIVE_WIDGETS: set[EditorWidget] = set()


class EditorWidget:
    """Lightweight iframe app for an :class:`EditorSession`.

    The app uses a local HTTP server rather than a custom Jupyter widget model.
    It renders in VS Code as plain HTML and updates the live Python session via
    JSON requests to the kernel process.
    """

    def __init__(self, session, **kwargs: Any):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected EditorWidget argument(s): {unknown}")
        self.session = session
        self.control_counts: dict[str, int] = {}
        self._offset_refit_model = None
        self._offset_refit_terms: list[str] = []
        self._offset_refit_labels: list[dict[str, Any]] = []
        self._collapsed_refit_model = None
        self._collapsed_refit_info: dict[str, Any] | None = None
        self._collapse_info_history: list[dict[str, Any] | None] = []
        self._in_force_info: dict[str, Any] | None = None
        self._profile_jobs: dict[str, dict[str, Any]] = {}
        self._profile_job_counter = 0
        self._profile_condition = threading.Condition(threading.RLock())
        self._token = secrets.token_urlsafe(24)
        self.terms = session_payload(session, self.control_counts)
        self.selected_term = next(iter(self.terms), "")
        self._lock = threading.RLock()
        self._closed = False
        self._evaluation_cache = EvaluationCache()
        self._evidence = EvidenceCoordinator(f"editor-{id(self):x}")
        # A local iframe avoids Jupyter widget-extension dependencies while
        # still letting Python own the authoritative edit state.
        self._server = EditorAppServer(self)
        self.host, self.port = self._server.host, self._server.port
        self.url = f"http://127.0.0.1:{self.port}"
        self.app_url = f"{self.url}?token={self._token}"
        self._server.start()
        _LIVE_WIDGETS.add(self)

    def _repr_html_(self) -> str:
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
        """Stop the local editor server."""
        if self._closed:
            return
        self._closed = True
        _LIVE_WIDGETS.discard(self)
        self._evidence.close()
        self._server.close()

    def _state(self) -> dict[str, Any]:
        with self._lock:
            self.terms = session_payload(self.session, self.control_counts)
            return {
                "model_revision": self.session.model_revision,
                "selected_term": self.selected_term,
                "terms": self.terms,
                "selection": {
                    name: self.session.selection(name).astype(int).tolist()
                    for name in self.session.terms
                },
                "can_uncollapse_levels": self.session.can_uncollapse_levels(),
                "last_collapse": (
                    None
                    if not self.session.can_uncollapse_levels()
                    or self._collapsed_refit_info is None
                    else dict(self._collapsed_refit_info)
                ),
                "history": history_payload(self.session),
            }

    def _select_term(self, term: str) -> None:
        if term not in self.session.terms:
            raise KeyError(f"Unknown editable term: {term!r}")
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
                self.session.undo(target)
            elif operation == "redo":
                self.session.redo(target)
            else:
                raise ValueError(f"Unknown editor operation: {operation!r}")
            # A fixed-offset refit is conditional on the current edited factors,
            # so any value-changing edit invalidates the stored refit result.
            if operation not in {"select_all", "reset_order"}:
                self._invalidate_refit()
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
            return self._state()

    def _set_control_count(self, term: str, count: int) -> dict[str, Any]:
        with self._lock:
            self._select_term(term)
            controls = self.session.control_points(term, n_handles=int(count))
            self.control_counts[term] = int(controls["x"].size)
            return self._state()

    def _average_relativity(self, term: str) -> None:
        # Average on the displayed relativity scale. This matches analyst
        # expectations for leveling factors better than averaging log effects.
        idx = self.session.selection(term)
        if idx.size == 0:
            raise ValueError(f"No points selected for term {term!r}.")
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

    def _metrics(
        self,
        metric: str,
        source: str | None = None,
        *,
        dataset: str | None = None,
        model_revision: int | None = None,
        request_sequence: int | None = None,
    ) -> dict[str, Any]:
        selected_source = "in_force" if source in (None, "selected") else source
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
        edited_metrics = self._dataset_metrics_for_evidence(
            "current", selected_model, eval_dataset, revision
        )
        if edited_metrics is None:
            return _superseded_payload(revision, request_sequence)
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
        role: str,
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
        outcome = self._evidence.submit(
            evidence_key,
            lambda request=request: {
                "metrics": metrics_module.compute_dataset_metrics(
                    request.model,
                    request.dataset,
                )
            },
        ).result()
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

        if source == "in_force":
            model, revision = self._current_model_for_evidence()
            if model is None:
                return _superseded_payload(revision, request_sequence)
            with self._lock:
                if revision != self.session.model_revision:
                    return _superseded_payload(revision, request_sequence)
                collapse_info = None if self._in_force_info is None else dict(self._in_force_info)
            kwargs = {"collapse_info_override": collapse_info}
        else:
            with self._lock:
                revision = self.session.model_revision
                if source == "original":
                    model = self.session.reference_model
                    kwargs = {}
                else:
                    model = self._offset_refit_model
                    kwargs = {
                        "offset_terms_override": list(self._offset_refit_terms),
                        "offset_labels_override": list(self._offset_refit_labels),
                    }

        payload = summary_payload(
            self,
            source,
            model_override=model,
            **kwargs,
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
        edited_model, revision = self._current_model_for_evidence()
        if edited_model is None:
            return _superseded_payload(revision, request_sequence)

        with self._lock:
            if revision != self.session.model_revision:
                return _superseded_payload(revision, request_sequence)
            datasets = tuple(evaluation_datasets(self.session))
            reference_model = getattr(self.session, "reference_model", self.session.model)
            collapse_info = None if self._in_force_info is None else dict(self._in_force_info)
            if reference_model is None:
                return report_payload(
                    self,
                    report,
                    model_revision=revision,
                    request_sequence=request_sequence,
                )

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
            model_override=edited_model,
            collapse_info_override=collapse_info,
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
        model, revision = self._current_model_for_evidence()
        if model is None:
            return _superseded_payload(revision, None)
        with self._lock:
            if revision != self.session.model_revision:
                return _superseded_payload(revision, None)
            if path:
                target = Path(path).expanduser()
            else:
                name = filename or "superglm_edited_model.joblib"
                if Path(name).name != name:
                    raise ValueError("filename must not contain directory separators")
                target = Path(directory or ".").expanduser() / name
        saved = persistence.save_model(self.session, target, model_override=model)
        saved_path = saved.resolve()
        return {
            "path": str(saved_path),
            "directory": str(saved_path.parent),
            "filename": saved_path.name,
            "message": f"Saved edited model to {saved_path}",
        }

    def _download_model(self, filename: str | None = None) -> tuple[bytes, str]:
        import io

        import joblib

        model, revision = self._current_model_for_evidence()
        if model is None:
            raise RuntimeError("Edited model request was superseded.")
        with self._lock:
            if revision != self.session.model_revision:
                raise RuntimeError("Edited model request was superseded.")
            name = filename or "superglm_edited_model.joblib"
            if Path(name).name != name:
                raise ValueError("filename must not contain directory separators")
            if not Path(name).suffix:
                name = f"{name}.joblib"
        buffer = io.BytesIO()
        joblib.dump(edited_model_for_export(self.session, model_override=model), buffer)
        return buffer.getvalue(), name

    def _open_directory(self, path: str | None = None) -> dict[str, str]:
        opened = open_directory_path(path)
        return {"path": str(opened)}

    def _save_directory(self, path: str | None = None) -> dict[str, Any]:
        target = Path(path).expanduser() if path else Path.cwd()
        resolved = target.resolve()
        if not resolved.exists():
            raise ValueError(f"Directory does not exist: {resolved}")
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

    def _refit_offset(self, method: str = "auto") -> dict[str, Any]:
        with self._lock:
            terms = self.session.edited_terms()
            if not terms:
                return {
                    "available": False,
                    "source": "refit",
                    "error": "No edited terms are available for a fixed-offset refit.",
                }
            refit_model = self.session.refit_with_edited_offset(method=method)
            self._offset_refit_model = refit_model
            self._offset_refit_terms = list(terms)
            self._offset_refit_labels = offset_label_payload(self.session, terms)
            return summary_payload(self, "refit")

    def _profile_distribution(
        self,
        parameter: str,
        *,
        progress_callback=None,
        **options: Any,
    ) -> dict[str, Any]:
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
            payload = summary_payload(self, "in_force")
            payload["profile_trace"] = _profile_trace_rows(result)
            payload["profile_estimate"] = estimate
            return payload

    def _start_profile_distribution_job(self, parameter: str, **options: Any) -> dict[str, Any]:
        """Start a distribution-parameter profile job and return its status payload."""
        with self._profile_condition:
            self._profile_job_counter += 1
            job_id = str(self._profile_job_counter)
            job = {
                "job_id": job_id,
                "parameter": parameter,
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
            args=(job_id, parameter, dict(options)),
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
                raise KeyError(f"Unknown profile job: {job_id!r}")
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
        options: dict[str, Any],
    ) -> None:
        def trace_callback(row: dict[str, Any]) -> None:
            with self._profile_condition:
                job = self._profile_jobs[job_id]
                _merge_profile_trace_rows(job, [row])
                self._profile_condition.notify_all()

        def progress_callback(phase: str, payload: dict[str, Any] | None = None) -> None:
            with self._profile_condition:
                job = self._profile_jobs[job_id]
                job["phase"] = phase
                if payload:
                    if "profile_estimate" in payload:
                        job["profile_estimate"] = _normalise_profile_estimate(
                            payload["profile_estimate"]
                        )
                self._profile_condition.notify_all()

        try:
            payload = self._profile_distribution(
                parameter,
                progress_callback=progress_callback,
                trace_callback=trace_callback,
                **options,
            )
        except BaseException as exc:
            with self._profile_condition:
                job = self._profile_jobs[job_id]
                job["status"] = "error"
                job["phase"] = "error"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
                self._profile_condition.notify_all()
            return

        with self._profile_condition:
            job = self._profile_jobs[job_id]
            _merge_profile_trace_rows(job, payload.get("profile_trace", []))
            job["status"] = "complete"
            job["phase"] = "complete"
            job["result"] = payload
            if "profile_estimate" in payload:
                job["profile_estimate"] = _normalise_profile_estimate(payload["profile_estimate"])
            job["finished_at"] = time.time()
            self._profile_condition.notify_all()

    def _structural_transition(
        self,
        operation: str,
        *,
        operation_start: float,
        fit_start: float,
        fit_end: float,
    ) -> dict[str, Any]:
        with self._lock:
            summary_start = time.perf_counter()
            summary = jsonable(summary_payload(self, "in_force"))
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

    def _collapse_levels(self, term: str | None = None, method: str = "auto") -> dict[str, Any]:
        with self._lock:
            operation_start = time.perf_counter()
            if term is not None:
                self._select_term(term)
            target = self.selected_term
            selected_indices = self.session.selection(target).astype(int).tolist()
            selected_levels = self._selected_level_labels(target)
            previous_info = None if self._in_force_info is None else dict(self._in_force_info)
            fit_start = time.perf_counter()
            refit_model = self.session.replace_with_collapsed_levels(target, method=method)
            fit_end = time.perf_counter()
            self._collapse_info_history.append(previous_info)
            self._collapsed_refit_model = refit_model
            self._collapsed_refit_info = dict(getattr(refit_model, "_editor_level_collapse", {}))
            self._in_force_info = dict(self._collapsed_refit_info)
            self._invalidate_refit()
            self._restore_selection(target, selected_levels, selected_indices)
            return self._structural_transition(
                "collapse_levels",
                operation_start=operation_start,
                fit_start=fit_start,
                fit_end=fit_end,
            )

    def _ungroup_levels(self, term: str | None = None, method: str = "auto") -> dict[str, Any]:
        with self._lock:
            operation_start = time.perf_counter()
            if term is not None:
                self._select_term(term)
            target = self.selected_term
            selected_indices = self.session.selection(target).astype(int).tolist()
            selected_levels = self._selected_level_labels(target)
            previous_info = None if self._in_force_info is None else dict(self._in_force_info)
            fit_start = time.perf_counter()
            refit_model = self.session.replace_with_ungrouped_levels(target, method=method)
            fit_end = time.perf_counter()
            self._collapse_info_history.append(previous_info)
            self._collapsed_refit_model = refit_model
            self._collapsed_refit_info = dict(getattr(refit_model, "_editor_level_collapse", {}))
            self._in_force_info = dict(self._collapsed_refit_info)
            self._invalidate_refit()
            self._restore_selection(target, selected_levels, selected_indices)
            return self._structural_transition(
                "ungroup_levels",
                operation_start=operation_start,
                fit_start=fit_start,
                fit_end=fit_end,
            )

    def _reorder_levels(self, term: str | None = None, target_index: int = 0) -> dict[str, Any]:
        with self._lock:
            if term is not None:
                self._select_term(term)
            self.session.reorder_levels(self.selected_term, target_index=int(target_index))
            return self._state()

    def _uncollapse_levels(self) -> dict[str, Any]:
        with self._lock:
            operation_start = time.perf_counter()
            fit_start = time.perf_counter()
            restored_model = self.session.uncollapse_levels()
            fit_end = time.perf_counter()
            restored_info = (
                self._collapse_info_history.pop() if self._collapse_info_history else None
            )
            self._collapsed_refit_model = (
                restored_model if getattr(restored_model, "_editor_level_collapse", None) else None
            )
            self._collapsed_refit_info = None if restored_info is None else dict(restored_info)
            self._in_force_info = None if restored_info is None else dict(restored_info)
            self._invalidate_refit()
            if self.selected_term not in self.session.terms:
                self.selected_term = next(iter(self.session.terms), "")
            return self._structural_transition(
                "uncollapse_levels",
                operation_start=operation_start,
                fit_start=fit_start,
                fit_end=fit_end,
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


def _superseded_payload(
    model_revision: int,
    request_sequence: int | None,
) -> dict[str, Any]:
    return {
        "status": "superseded",
        "model_revision": int(model_revision),
        "request_sequence": request_sequence,
    }


def _close_live_widgets() -> None:
    for widget in list(_LIVE_WIDGETS):
        widget.close()


def _profile_trace_rows(result: Any) -> list[dict[str, Any]]:
    trace = getattr(result, "search_trace", None)
    if trace is None:
        cache = getattr(result, "cache", None)
        if isinstance(cache, dict):
            return [
                {"step": i, "theta": theta, "nll": nll, "source": "profile"}
                for i, (theta, nll) in enumerate(cache.items())
            ]
        return []
    if hasattr(trace, "to_dict"):
        rows = trace.to_dict("records")
    else:
        rows = list(trace)
    return [jsonable(row) for row in rows]


def _profile_estimate_payload(result: Any, parameter: str) -> dict[str, Any]:
    key = parameter.lower().replace("-", "_")
    if key in {"tweedie", "tweedie_p", "p"}:
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
    ci = getattr(result, "ci", None)
    if callable(ci):
        try:
            ci_low, ci_high = ci(alpha=0.05)
        except Exception:
            ci_low, ci_high = None, None

    return _normalise_profile_estimate(
        {
            "parameter": name,
            "label": label,
            "value": value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "objective": getattr(result, "nll", None),
            "objective_label": "loss",
            "lower_is_better": True,
        }
    )


def _normalise_profile_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(estimate)
    payload.setdefault("parameter", "")
    payload.setdefault("label", str(payload.get("parameter") or "estimate"))
    payload.setdefault("value", None)
    payload.setdefault("ci_low", None)
    payload.setdefault("ci_high", None)
    payload.setdefault("objective", None)
    payload.setdefault("objective_label", "loss")
    payload.setdefault("lower_is_better", True)
    return jsonable(payload)


def _elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000.0)


def _merge_profile_trace_rows(job: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    trace = job.setdefault("trace", [])
    seen = {_profile_trace_key(row) for row in trace}
    for row in rows:
        payload = jsonable(dict(row))
        key = _profile_trace_key(payload)
        if key in seen:
            continue
        trace.append(payload)
        seen.add(key)


def _profile_trace_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("step"),
        row.get("p"),
        row.get("theta"),
        row.get("phi"),
        row.get("nll"),
        row.get("source"),
    )


atexit.register(_close_live_widgets)


__all__ = ["EditorWidget"]
