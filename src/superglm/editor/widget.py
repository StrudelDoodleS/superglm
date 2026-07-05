"""Notebook/editor frontend for editor sessions.

The default renderer is a tiny local HTML app served by the Python kernel. This
avoids custom Jupyter widget modules, which are often unavailable in VS Code.
"""

from __future__ import annotations

import atexit
import html
import threading
from typing import Any

import numpy as np

from superglm.editor.metrics import metrics_payload
from superglm.editor.payloads import session_payload
from superglm.editor.reports import report_payload
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
        self.terms = session_payload(session, self.control_counts)
        self.selected_term = next(iter(self.terms), "")
        self._lock = threading.RLock()
        self._closed = False
        # A local iframe avoids Jupyter widget-extension dependencies while
        # still letting Python own the authoritative edit state.
        self._server = EditorAppServer(self)
        self.host, self.port = self._server.host, self._server.port
        self.url = f"http://127.0.0.1:{self.port}"
        self._server.start()
        _LIVE_WIDGETS.add(self)

    def _repr_html_(self) -> str:
        src = html.escape(self.url, quote=True)
        return (
            "<div style='width:100%;max-width:1180px'>"
            f"<iframe src='{src}' width='100%' height='720' "
            "style='border:1px solid #d0d7de;border-radius:6px;background:white'></iframe>"
            "<div style='font:12px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
            "color:#57606a;margin-top:4px'>"
            f"SuperGLM editor running at <a href='{src}' target='_blank'>{src}</a>"
            "</div></div>"
        )

    def close(self) -> None:
        """Stop the local editor server."""
        if self._closed:
            return
        self._closed = True
        _LIVE_WIDGETS.discard(self)
        self._server.close()

    def _state(self) -> dict[str, Any]:
        with self._lock:
            self.terms = session_payload(self.session, self.control_counts)
            return {
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
            }

    def _set_term(self, term: str) -> dict[str, Any]:
        with self._lock:
            if term not in self.session.terms:
                raise KeyError(f"Unknown editable term: {term!r}")
            self.selected_term = term
            return self._state()

    def _select(self, term: str, indices: list[int]) -> dict[str, Any]:
        with self._lock:
            self._set_term(term)
            self.session.select_indices(term, indices)
            return self._state()

    def _operate(self, operation: str, term: str | None = None) -> dict[str, Any]:
        with self._lock:
            if term is not None:
                self._set_term(term)
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
            self._set_term(term)
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
            self._set_term(term)
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
            self._set_term(term)
            controls = self.session.control_points(term, n_handles=int(count))
            self.control_counts[term] = int(controls["x"].size)
            return self._state()

    def _average_relativity(self, term: str) -> None:
        # Average on the displayed relativity scale. This matches analyst
        # expectations for leveling factors better than averaging log effects.
        idx = self.session.selection(term)
        if idx.size == 0:
            raise ValueError(f"No points selected for term {term!r}.")
        editable = self.session.terms[term]
        weights = np.ones(idx.size, dtype=np.float64)
        if editable.weights is not None:
            weights = np.asarray(editable.weights[idx], dtype=np.float64)
        value = float(np.average(np.exp(editable.edited_log_effect[idx]), weights=weights))
        self.session.set_values(term, idx, np.full(idx.size, np.log(max(value, 1e-12))))

    def _metrics(self, metric: str, source: str | None = None) -> dict[str, Any]:
        with self._lock:
            selected_source = "in_force" if source in (None, "selected") else source
            return metrics_payload(self.session, metric, source=selected_source)

    def _summary(self, source: str) -> dict[str, Any]:
        with self._lock:
            if source == "selected":
                source = "in_force"
            return summary_payload(self, source)

    def _report(self, report: str = "validation") -> dict[str, Any]:
        with self._lock:
            return report_payload(self, report)

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

    def _collapse_levels(self, term: str | None = None, method: str = "auto") -> dict[str, Any]:
        with self._lock:
            if term is not None:
                self._set_term(term)
            target = self.selected_term
            selected_indices = self.session.selection(target).astype(int).tolist()
            previous_info = None if self._in_force_info is None else dict(self._in_force_info)
            refit_model = self.session.replace_with_collapsed_levels(target, method=method)
            self._collapse_info_history.append(previous_info)
            self._collapsed_refit_model = refit_model
            self._collapsed_refit_info = dict(getattr(refit_model, "_editor_level_collapse", {}))
            self._in_force_info = dict(self._collapsed_refit_info)
            if selected_indices:
                self.session.select_indices(target, selected_indices)
            return summary_payload(self, "in_force")

    def _ungroup_levels(self, term: str | None = None, method: str = "auto") -> dict[str, Any]:
        with self._lock:
            if term is not None:
                self._set_term(term)
            target = self.selected_term
            selected_indices = self.session.selection(target).astype(int).tolist()
            previous_info = None if self._in_force_info is None else dict(self._in_force_info)
            refit_model = self.session.replace_with_ungrouped_levels(target, method=method)
            self._collapse_info_history.append(previous_info)
            self._collapsed_refit_model = refit_model
            self._collapsed_refit_info = dict(getattr(refit_model, "_editor_level_collapse", {}))
            self._in_force_info = dict(self._collapsed_refit_info)
            if selected_indices:
                self.session.select_indices(target, selected_indices)
            return summary_payload(self, "in_force")

    def _reorder_levels(self, term: str | None = None, target_index: int = 0) -> dict[str, Any]:
        with self._lock:
            if term is not None:
                self._set_term(term)
            self.session.reorder_levels(self.selected_term, target_index=int(target_index))
            return self._state()

    def _uncollapse_levels(self) -> dict[str, Any]:
        with self._lock:
            restored_model = self.session.uncollapse_levels()
            restored_info = (
                self._collapse_info_history.pop() if self._collapse_info_history else None
            )
            self._collapsed_refit_model = (
                restored_model if getattr(restored_model, "_editor_level_collapse", None) else None
            )
            self._collapsed_refit_info = None if restored_info is None else dict(restored_info)
            self._in_force_info = None if restored_info is None else dict(restored_info)
            if self.selected_term not in self.session.terms:
                self.selected_term = next(iter(self.session.terms), "")
            return summary_payload(self, "in_force")

    def _invalidate_refit(self) -> None:
        self._offset_refit_model = None
        self._offset_refit_terms = []
        self._offset_refit_labels = []


def _close_live_widgets() -> None:
    for widget in list(_LIVE_WIDGETS):
        widget.close()


atexit.register(_close_live_widgets)


__all__ = ["EditorWidget"]
