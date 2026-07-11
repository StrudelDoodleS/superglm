"""Python state core for editing fitted 1D SuperGLM effects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.editor import apply, persistence
from superglm.editor._types import EditableTerm, EditRecord
from superglm.editor.collapse import (
    clone_with_replaced_feature,
    collapsed_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.controls import control_curve_after_move
from superglm.editor.controls import control_points as _control_points
from superglm.editor.evaluation import coerce_evaluation_data, default_metrics_dataset
from superglm.editor.evaluation_cache import EditMaterializationRequest
from superglm.editor.level_order import (
    level_order_for_direction,
    level_order_for_labels,
    level_order_for_target,
)
from superglm.editor.operations import (
    anchored_isotonic_values,
    anchored_smooth_values,
    monotone_clamp_values,
)
from superglm.editor.refit import fit_refit_model
from superglm.editor.terms import (
    term_from_inference,
    term_offset_values,
    term_type_from_spec,
    term_weights_from_data,
    term_weights_from_fit,
)


class EditorSession:
    """Stateful editor for fitted 1D main effects.

    The session never mutates the source model. Edits are kept in memory until
    :meth:`to_model` is called.
    """

    def __init__(
        self,
        model,
        terms: dict[str, EditableTerm],
        *,
        n_points: int = 200,
        centering: str = "native",
        reference_model=None,
        evaluation_data: dict[str, Any] | None = None,
        cv_report: Any = None,
    ):
        self.model = model
        self.reference_model = model if reference_model is None else reference_model
        self.terms = terms
        self.n_points = int(n_points)
        self.centering = centering
        self._evaluation_data = dict(evaluation_data or {})
        self.cv_report = cv_report
        self._term_names = list(terms)
        self._selection: dict[str, NDArray[np.intp]] = {
            name: np.array([], dtype=np.intp) for name in terms
        }
        self._level_orders: dict[str, list[str]] = {}
        self.history: list[EditRecord] = []
        self.redo_stack: list[EditRecord] = []
        self.collapse_history: list[Any] = []
        self._model_revision = 0
        self._edit_epoch = 0
        self._materialized_edit_model = None
        self._materialized_edit_epoch: int | None = None

    @classmethod
    def from_model(
        cls,
        model,
        terms: list[str] | tuple[str, ...] | None = None,
        *,
        n_points: int = 200,
        centering: str = "native",
        with_se: bool = True,
        train_data=None,
        validation_data=None,
        test_data=None,
        cv_report: Any = None,
    ) -> EditorSession:
        """Build an editor session from fitted 1D main-effect inference."""
        if getattr(model, "_result", None) is None:
            raise RuntimeError("Model must be fitted before creating an editor session.")

        evaluation_data = coerce_evaluation_data(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
        )
        names = list(model._feature_order if terms is None else terms)
        editable = cls._editable_terms_from_model(
            model,
            names,
            n_points=n_points,
            centering=centering,
            with_se=with_se,
            train_data=evaluation_data.get("train"),
        )
        return cls(
            model,
            editable,
            n_points=n_points,
            centering=centering,
            reference_model=model,
            evaluation_data=evaluation_data,
            cv_report=cv_report,
        )

    @staticmethod
    def _editable_terms_from_model(
        model,
        names: list[str],
        *,
        n_points: int,
        centering: str,
        with_se: bool,
        train_data=None,
    ) -> dict[str, EditableTerm]:
        editable: dict[str, EditableTerm] = {}
        for name in names:
            if name in model._interaction_specs:
                raise ValueError(f"Interactions are not editable in v1: {name!r}")
            if name not in model._specs:
                raise KeyError(f"Term not found: {name!r}")
            ti = model.term_inference(
                name,
                with_se=with_se,
                n_points=n_points,
                centering=centering,
            )
            term = term_from_inference(ti)
            if centering != "native":
                native_ti = model.term_inference(
                    name,
                    with_se=False,
                    n_points=n_points,
                    centering="native",
                )
                EditorSession._attach_native_log_effect(term, native_ti)
            term.metadata["term_type"] = term_type_from_spec(model._specs[name])
            term.weights = (
                term_weights_from_data(train_data.X, train_data.sample_weight, name, term)
                if train_data is not None
                else term_weights_from_fit(model, name, term)
            )
            editable[name] = term
        return editable

    @staticmethod
    def _attach_native_log_effect(term: EditableTerm, native_ti) -> None:
        native = np.asarray(native_ti.log_relativity, dtype=np.float64).ravel()
        if term.levels is not None and native_ti.levels is not None:
            native_levels = [str(level) for level in native_ti.levels]
            native_by_level = dict(zip(native_levels, native, strict=True))
            term.metadata["native_original_log_effect"] = [
                float(native_by_level[level]) for level in term.levels
            ]
            return

        native_x = (
            None if native_ti.x is None else np.asarray(native_ti.x, dtype=np.float64).ravel()
        )
        if term.x is not None and native_x is not None:
            if native.shape == term.x.shape and np.allclose(native_x, term.x):
                values = native
            else:
                order = np.argsort(native_x)
                values = np.interp(term.x, native_x[order], native[order])
            term.metadata["native_original_log_effect"] = [float(value) for value in values]
            return

        if native.shape == term.original_log_effect.shape:
            term.metadata["native_original_log_effect"] = [float(value) for value in native]

    # Selection is term-local and always stored as display-grid indices. The UI
    # may select by x range or category label, but downstream edit operations
    # only deal with integer positions in the editable term arrays.
    @property
    def model_revision(self) -> int:
        """Semantic revision for prediction- or fit-evidence-changing state."""
        return self._model_revision

    @property
    def edit_epoch(self) -> int:
        """Monotonic invalidation token for the current edited-model materialization."""
        return self._edit_epoch

    def _advance_model_revision(self) -> None:
        self._model_revision += 1
        self._edit_epoch += 1
        self._materialized_edit_model = None
        self._materialized_edit_epoch = None

    def selection(self, term: str) -> NDArray[np.intp]:
        """Return selected point indices for a term."""
        self._require_term(term)
        return self._selection[term].copy()

    def select_indices(self, term: str, indices) -> EditorSession:
        """Select explicit point or level indices."""
        editable = self._require_term(term)
        idx = np.asarray(indices, dtype=np.intp).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= editable.size):
            raise IndexError(f"Selection indices out of range for term {term!r}.")
        self._selection[term] = np.unique(idx)
        return self

    def select_x(self, term: str, start: float, stop: float) -> EditorSession:
        """Select grid points whose x values fall inside [start, stop]."""
        editable = self._require_term(term)
        if editable.x is None:
            raise TypeError(f"Term {term!r} does not have a numeric x grid.")
        lo, hi = sorted((float(start), float(stop)))
        idx = np.flatnonzero((editable.x >= lo) & (editable.x <= hi)).astype(np.intp)
        self._selection[term] = idx
        return self

    def select_levels(self, term: str, levels: list[str] | tuple[str, ...]) -> EditorSession:
        """Select categorical or ordered-categorical levels by label."""
        editable = self._require_term(term)
        if editable.levels is None:
            raise TypeError(f"Term {term!r} does not have levels.")
        level_to_idx = {level: i for i, level in enumerate(editable.levels)}
        missing = [level for level in levels if level not in level_to_idx]
        if missing:
            raise KeyError(f"Unknown level(s) for term {term!r}: {missing}")
        self._selection[term] = np.array([level_to_idx[level] for level in levels], dtype=np.intp)
        return self

    def clear_selection(self, term: str) -> EditorSession:
        """Clear the active selection for a term."""
        self._require_term(term)
        self._selection[term] = np.array([], dtype=np.intp)
        return self

    # Manual edits operate on the link/log-effect scale. The browser displays
    # relativities, but keeping session state additive makes model-copy and
    # offset scoring unambiguous.
    def reset(self, term: str) -> EditorSession:
        """Reset selected values to the original fit, or the whole term if nothing is selected."""
        editable = self._require_term(term)
        idx = self._selection[term]
        if idx.size == 0:
            idx = np.arange(editable.size, dtype=np.intp)
        else:
            idx = self._expand_collapsed_level_indices(term, idx)
        idx = np.unique(np.asarray(idx, dtype=np.intp))
        before = editable.edited_log_effect[idx].copy()
        restored = editable.original_log_effect[idx].copy()
        editable.edited_log_effect[idx] = restored
        if idx.size == editable.size:
            self._clear_term_history(term)
        else:
            self._trim_term_history(term, idx)
        if not np.array_equal(before, restored):
            self._advance_model_revision()
        return self

    def shift(self, term: str, delta: float) -> EditorSession:
        """Add a link-scale delta to the selected values."""
        idx = self._require_edit_selection(term)
        editable = self._require_term(term)
        before = editable.edited_log_effect[idx].copy()
        after = before + float(delta)
        self._commit(term, "shift", idx, before, after, {"delta": float(delta)})
        return self

    def set_value(self, term: str, value: float) -> EditorSession:
        """Set selected values to one link-scale value."""
        idx = self._require_edit_selection(term)
        editable = self._require_term(term)
        before = editable.edited_log_effect[idx].copy()
        after = np.full(idx.size, float(value), dtype=np.float64)
        self._commit(term, "set_value", idx, before, after, {"value": float(value)})
        return self

    def set_values(self, term: str, indices, values) -> EditorSession:
        """Set explicit values on the internal link/log-effect scale."""
        editable = self._require_term(term)
        idx = np.asarray(indices, dtype=np.intp).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= editable.size):
            raise IndexError(f"Edit indices out of range for term {term!r}.")
        after = np.asarray(values, dtype=np.float64).ravel()
        if after.size != idx.size:
            raise ValueError(f"Expected {idx.size} values for term {term!r}, got {after.size}.")
        assignment_positions: dict[int, int] = {}
        unique_indices: list[int] = []
        unique_values: list[float] = []
        for raw_index, raw_value in zip(idx, after, strict=True):
            index = int(raw_index)
            value = float(raw_value)
            position = assignment_positions.get(index)
            if position is None:
                assignment_positions[index] = len(unique_indices)
                unique_indices.append(index)
                unique_values.append(value)
            elif unique_values[position] != value:
                raise ValueError(
                    f"Conflicting values were supplied for edit index {index} in term {term!r}."
                )
        idx = np.asarray(unique_indices, dtype=np.intp)
        after = np.asarray(unique_values, dtype=np.float64)
        idx, after = self._expand_collapsed_level_assignments(term, idx, after)
        before = editable.edited_log_effect[idx].copy()
        self._commit(term, "set_values", idx, before, after, {})
        return self

    def weighted_average(self, term: str) -> EditorSession:
        """Flatten the selected region to its weighted mean."""
        idx = self._require_edit_selection(term)
        editable = self._require_term(term)
        before = editable.edited_log_effect[idx].copy()
        weights = np.ones(idx.size, dtype=np.float64)
        if editable.weights is not None:
            weights = np.asarray(editable.weights[idx], dtype=np.float64)
        if float(np.sum(weights)) <= 0.0:
            weights = None
        value = float(np.average(before, weights=weights))
        after = np.full(idx.size, value, dtype=np.float64)
        self._commit(term, "weighted_average", idx, before, after, {"value": value})
        return self

    def linear_interpolate(self, term: str, strength: float = 1.0) -> EditorSession:
        """Move selected values toward a line between selected endpoints."""
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be between 0 and 1, got {strength!r}")
        idx = self._require_edit_selection(term)
        if idx.size < 2:
            return self
        editable = self._require_term(term)
        before = editable.edited_log_effect[idx].copy()
        x_vals = (
            np.arange(editable.size, dtype=np.float64)
            if editable.x is None
            else np.asarray(editable.x, dtype=np.float64)
        )
        x_sel = x_vals[idx]
        if abs(float(x_sel[-1] - x_sel[0])) < 1e-15:
            target = np.full(idx.size, float(before[0]), dtype=np.float64)
        else:
            target = np.interp(x_sel, [x_sel[0], x_sel[-1]], [before[0], before[-1]])
        after = (1.0 - float(strength)) * before + float(strength) * target
        self._commit(term, "linear_interpolate", idx, before, after, {"strength": float(strength)})
        return self

    def level_left(self, term: str) -> EditorSession:
        """Set selected values to the left-most selected value."""
        return self._level_selected(term, "level_left", "left")

    def level_right(self, term: str) -> EditorSession:
        """Set selected values to the right-most selected value."""
        return self._level_selected(term, "level_right", "right")

    def snap_highest(self, term: str) -> EditorSession:
        """Set selected values to the highest selected value."""
        return self._level_selected(term, "snap_highest", "highest")

    def snap_lowest(self, term: str) -> EditorSession:
        """Set selected values to the lowest selected value."""
        return self._level_selected(term, "snap_lowest", "lowest")

    def reorder_levels(
        self,
        term: str,
        direction: str | None = None,
        *,
        target_index: int | None = None,
    ) -> EditorSession:
        """Move selected categorical levels in display order."""
        editable = self._require_term(term)
        if editable.levels is None:
            raise TypeError(f"Term {term!r} does not have levels.")
        if self._is_ordered_level_term(term):
            raise TypeError(f"Ordered categorical term {term!r} cannot be display-reordered.")
        idx = self._selection[term]
        if idx.size == 0:
            return self
        if target_index is not None:
            order = level_order_for_target(editable.size, idx, int(target_index))
        else:
            order = level_order_for_direction(editable.size, idx, direction)
        if order == list(range(editable.size)):
            return self
        self._apply_level_order(term, np.asarray(order, dtype=np.intp))
        return self

    def reset_level_order(self, term: str) -> EditorSession:
        """Reset a categorical term's display order to the fitted model order."""
        editable = self._require_term(term)
        if editable.levels is None:
            raise TypeError(f"Term {term!r} does not have levels.")
        if self._is_ordered_level_term(term):
            self._level_orders.pop(term, None)
            return self
        order = level_order_for_labels(editable.levels, self._model_level_order(term))
        self._level_orders.pop(term, None)
        if order != list(range(editable.size)):
            self._apply_level_order(term, np.asarray(order, dtype=np.intp), persist=False)
        return self

    def level_order_changed(self, term: str) -> bool:
        """Return whether a categorical term has a custom display order."""
        editable = self._require_term(term)
        return (
            editable.levels is not None
            and not self._is_ordered_level_term(term)
            and term in self._level_orders
        )

    # Shape-changing operations anchor selected runs to adjacent unselected
    # values so local edits do not create artificial jumps at selection edges.
    def isotonic(self, term: str, direction: str = "increasing") -> EditorSession:
        """Apply weighted isotonic regression to selected values.

        Selected contiguous runs are anchored to adjacent unselected neighbors
        when available. This avoids visible jumps at edit boundaries. If no
        points are selected, the operation applies to the whole term.
        """
        if direction not in ("increasing", "decreasing"):
            raise ValueError(f"direction must be 'increasing' or 'decreasing', got {direction!r}")
        editable = self._require_term(term)
        idx = self._edit_selection_or_all(term)
        before = editable.edited_log_effect[idx].copy()
        if editable.levels is not None:
            after = monotone_clamp_values(editable.edited_log_effect, idx, direction)
        else:
            after = anchored_isotonic_values(
                editable.edited_log_effect,
                idx,
                None if editable.weights is None else editable.weights,
                direction,
            )
        self._commit(term, "isotonic", idx, before, after, {"direction": direction})
        return self

    def smooth(self, term: str, strength: float = 0.5) -> EditorSession:
        """Apply light local smoothing to selected values.

        Selected contiguous runs are anchored to adjacent unselected neighbors
        when available. This avoids visible jumps at edit boundaries. If no
        points are selected, the operation applies to the whole term.
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"strength must be between 0 and 1, got {strength!r}")
        editable = self._require_term(term)
        idx = self._edit_selection_or_all(term)
        before = editable.edited_log_effect[idx].copy()
        if strength == 0.0:
            after = before.copy()
        else:
            after = anchored_smooth_values(editable.edited_log_effect, idx, strength)
        self._commit(term, "smooth", idx, before, after, {"strength": float(strength)})
        return self

    # Control handles are fixed-x vertical handles. They either map to the
    # fitted spline basis directly or fall back to a local monotone cubic curve.
    def control_points(self, term: str, n_handles: int | None = None) -> dict[str, Any]:
        """Return fixed-x spline control handles for advanced curve editing."""
        editable = self._require_control_term(term)
        return _control_points(self.model, editable, n_handles=n_handles)

    def move_control_point(
        self,
        term: str,
        handle_index: int,
        log_effect: float,
        *,
        n_handles: int | None = None,
    ) -> EditorSession:
        """Move one spline control handle vertically and refit the displayed curve."""
        editable = self._require_control_term(term)
        handle_index = int(handle_index)
        before = editable.edited_log_effect.copy()
        after, metadata = control_curve_after_move(
            self.model,
            editable,
            handle_index,
            float(log_effect),
            n_handles=n_handles,
        )
        idx = np.arange(editable.size, dtype=np.intp)
        self._commit(
            term,
            "control_point",
            idx,
            before,
            after,
            {
                "handle_index": handle_index,
                "log_effect": float(log_effect),
                **metadata,
            },
        )
        return self

    def undo(self, term: str | None = None) -> EditorSession:
        """Undo the most recent edit."""
        if not self.history:
            return self
        record = self._pop_record(self.history, term)
        if record is None:
            return self
        current = self.terms[record.term].edited_log_effect[record.indices].copy()
        self.terms[record.term].edited_log_effect[record.indices] = record.before
        self.redo_stack.append(record)
        if not np.array_equal(current, record.before):
            self._advance_model_revision()
        return self

    def redo(self, term: str | None = None) -> EditorSession:
        """Redo the most recently undone edit."""
        if not self.redo_stack:
            return self
        record = self._pop_record(self.redo_stack, term)
        if record is None:
            return self
        current = self.terms[record.term].edited_log_effect[record.indices].copy()
        self.terms[record.term].edited_log_effect[record.indices] = record.after
        self.history.append(record)
        if not np.array_equal(current, record.after):
            self._advance_model_revision()
        return self

    def to_model(self, *, X=None, y=None, sample_weight=None, offset=None):
        """Return an edited copy of the source model.

        If explicit evaluation data is supplied, scalar fit statistics on the
        copy are refreshed against that data. Otherwise the fitted model's
        retained fit data is used when available.
        """
        if (
            X is None
            and y is None
            and sample_weight is None
            and offset is None
            and self.edited_terms()
        ):
            train = self._evaluation_data.get("train")
            if train is not None:
                X = train.X
                y = train.y
                sample_weight = train.sample_weight
                offset = train.offset
        return apply.apply_edits_to_model_copy_with_data(
            self.model,
            self.terms,
            X=X,
            y=y,
            sample_weight=sample_weight,
            offset=offset,
        )

    def capture_materialization_request(self) -> EditMaterializationRequest | None:
        """Capture only plot-scale edits and immutable evaluation references."""
        if not self.edited_terms():
            return None
        return EditMaterializationRequest(
            model_revision=self.model_revision,
            edit_epoch=self.edit_epoch,
            base_model=self.model,
            terms={name: term.copy() for name, term in self.terms.items()},
            dataset=default_metrics_dataset(self),
        )

    def cached_materialized_model(
        self,
        edit_epoch: int,
        *,
        model_revision: int | None = None,
        base_model=None,
    ):
        """Return the cached model only for matching live session identity."""
        if int(edit_epoch) != self.edit_epoch:
            return None
        if model_revision is not None and int(model_revision) != self.model_revision:
            return None
        if base_model is not None and base_model is not self.model:
            return None
        if self._materialized_edit_epoch != int(edit_epoch):
            return None
        return self._materialized_edit_model

    def publish_materialized_model(self, request: EditMaterializationRequest, model) -> bool:
        """Publish a private model only if its entire snapshot is still current."""
        if (
            request.model_revision != self.model_revision
            or request.edit_epoch != self.edit_epoch
            or request.base_model is not self.model
        ):
            return False
        self._materialized_edit_model = model
        self._materialized_edit_epoch = request.edit_epoch
        return True

    def save_model(self, path: str | Path) -> Path:
        return persistence.save_model(self, path)

    # Offset refits are conditional diagnostics: edited terms become fixed
    # link-scale factors and are removed from the refitted feature set.
    def edited_terms(self) -> list[str]:
        """Names of terms whose edited values differ from the source fit."""
        return [
            name
            for name, term in self.terms.items()
            if not np.allclose(
                term.edited_log_effect,
                term.original_log_effect,
                rtol=0.0,
                atol=1e-14,
            )
        ]

    def edited_offset(
        self,
        terms: str | list[str] | tuple[str, ...] | None = None,
        *,
        X=None,
    ) -> NDArray:
        """Return the link-scale offset implied by edited term curves.

        This is the additive offset to pass to ``fit``/``fit_reml`` when the
        edited terms should be treated as fixed rating factors. If ``terms`` is
        omitted, all changed terms in the session are used.
        """
        names = self._resolve_offset_terms(terms)
        X_ref = self._resolve_offset_frame(X)
        n = len(X_ref)
        offset = np.zeros(n, dtype=np.float64)
        for name in names:
            if name not in X_ref:
                raise KeyError(f"Offset data is missing column {name!r}.")
            offset += term_offset_values(self.terms[name], X_ref[name])
        return offset

    def edited_offset_factor(
        self,
        terms: str | list[str] | tuple[str, ...] | None = None,
        *,
        X=None,
    ) -> NDArray:
        """Return the multiplicative factor represented by ``edited_offset``."""
        return np.exp(self.edited_offset(terms, X=X))

    def refit_with_edited_offset(
        self,
        terms: str | list[str] | tuple[str, ...] | None = None,
        *,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        method: str = "auto",
        lambda1=...,
        lambda2=...,
        **fit_kwargs: Any,
    ):
        """Refit a copy with edited terms fixed as an offset.

        The returned model excludes the offset terms from its feature set and
        fits the remaining terms conditional on the edited offset. Inference on
        the returned model is therefore for the refitted remaining terms, not
        for the manually edited offset curves.
        """
        if self.model is None:
            raise RuntimeError("Cannot refit without a source model.")
        names = self._resolve_offset_terms(terms)
        self._require_not_interaction_parent(names, operation="fixed-offset refit")
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X,
            y,
            sample_weight,
            offset,
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")

        editor_offset = self.edited_offset(names, X=X_ref)
        combined_offset = editor_offset
        if base_offset is not None:
            combined_offset = np.asarray(base_offset, dtype=np.float64).ravel() + editor_offset

        refit_model = self.model._clone_without_features(
            set(names),
            lambda1=lambda1,
            lambda2=lambda2,
        )
        fit_refit_model(
            self.model,
            refit_model,
            method=method,
            X=X_ref,
            y=y_ref,
            sample_weight=sample_weight_ref,
            offset=combined_offset,
            fit_kwargs=fit_kwargs,
        )

        refit_model._editor_offset = {
            "format": "superglm.editor.offset.v1",
            "terms": list(names),
            "link_scale": True,
            "message": (
                "Manual editor terms are fixed as an offset; inference is "
                "conditional on those fixed offset factors."
            ),
        }
        return refit_model

    def reprofile_distribution(
        self,
        parameter: str,
        *,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        fit_mode: str = "inherit",
        progress_callback=None,
        **profile_kwargs: Any,
    ):
        """Explicitly re-estimate a distribution parameter for the in-force model."""
        if self.model is None:
            raise RuntimeError("Cannot reprofile without a source model.")
        if self.edited_terms():
            raise RuntimeError(
                "Cannot re-profile distribution parameters while manual coefficient edits "
                "are pending. Reset edits, apply them as fixed offsets, or refit first."
            )
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X,
            y,
            sample_weight,
            offset,
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")

        profile_model = self.model._clone_without_features(set())
        if getattr(self.model, "_last_fit_meta", None) is not None:
            profile_model._last_fit_meta = dict(self.model._last_fit_meta)

        key = parameter.lower().replace("-", "_")
        if key in {"tweedie", "tweedie_p", "p"}:
            profile_kwargs.setdefault("fit_mode", fit_mode)
            if progress_callback is not None:
                profile_kwargs.setdefault("progress_callback", progress_callback)
            result = profile_model.estimate_p(
                X_ref,
                y_ref,
                sample_weight=sample_weight_ref,
                offset=base_offset,
                **profile_kwargs,
            )
        elif key in {"nb2", "nb2_theta", "negative_binomial", "theta"}:
            profile_kwargs.setdefault("fit_mode", fit_mode)
            if progress_callback is not None:
                profile_kwargs.setdefault("progress_callback", progress_callback)
            result = profile_model.estimate_theta(
                X_ref,
                y_ref,
                sample_weight=sample_weight_ref,
                offset=base_offset,
                **profile_kwargs,
            )
        else:
            raise ValueError("parameter must be 'tweedie_p' or 'nb2_theta'.")

        self.replace_in_force_model(profile_model)
        self.collapse_history.clear()
        return result

    def refit_with_collapsed_levels(
        self,
        term: str,
        *,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        method: str = "auto",
        group_label: str | None = None,
        lambda1=...,
        lambda2=...,
        **fit_kwargs: Any,
    ):
        """Collapse selected categorical levels and refit a full model copy."""
        if self.model is None:
            raise RuntimeError("Cannot refit without a source model.")
        editable = self._require_term(term)
        idx = self._require_selection(term)
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X,
            y,
            sample_weight,
            offset,
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")

        replacement, metadata = collapsed_feature_spec(
            self.model,
            editable,
            idx,
            X=X_ref,
            group_label=group_label,
        )
        refit_model = clone_with_replaced_feature(
            self.model,
            term,
            replacement,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        resolved_method = fit_refit_model(
            self.model,
            refit_model,
            method=method,
            X=X_ref,
            y=y_ref,
            sample_weight=sample_weight_ref,
            offset=base_offset,
            fit_kwargs=fit_kwargs,
        )

        metadata["method"] = resolved_method
        refit_model._editor_level_collapse = metadata
        return refit_model

    def replace_with_collapsed_levels(self, term: str, **kwargs: Any):
        """Collapse selected levels, refit, and make the refit the in-force edit model."""
        previous_model = self.model
        refit_model = self.refit_with_collapsed_levels(term, **kwargs)
        self.collapse_history.append(previous_model)
        try:
            self.replace_in_force_model(refit_model)
        except Exception:
            self.collapse_history.pop()
            raise
        return refit_model

    def refit_with_ungrouped_levels(
        self,
        term: str,
        *,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        method: str = "auto",
        lambda1=...,
        lambda2=...,
        **fit_kwargs: Any,
    ):
        """Remove selected levels from collapsed groups and refit a model copy."""
        if self.model is None:
            raise RuntimeError("Cannot refit without a source model.")
        editable = self._require_term(term)
        idx = self._require_selection(term)
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X,
            y,
            sample_weight,
            offset,
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")

        replacement, metadata = ungrouped_feature_spec(
            self.model,
            editable,
            idx,
            X=X_ref,
        )
        refit_model = clone_with_replaced_feature(
            self.model,
            term,
            replacement,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        resolved_method = fit_refit_model(
            self.model,
            refit_model,
            method=method,
            X=X_ref,
            y=y_ref,
            sample_weight=sample_weight_ref,
            offset=base_offset,
            fit_kwargs=fit_kwargs,
        )

        metadata["method"] = resolved_method
        refit_model._editor_level_collapse = metadata
        return refit_model

    def replace_with_ungrouped_levels(self, term: str, **kwargs: Any):
        """Ungroup selected levels, refit, and make the refit the in-force model."""
        previous_model = self.model
        restore_previous = self._ungroup_restores_reference_model(term, **kwargs)
        restored_history_model = None
        clear_history_after_refit = False
        if (
            restore_previous
            and self.collapse_history
            and not self._model_has_collapsed_level_groups(self.collapse_history[-1])
        ):
            refit_model = self.collapse_history.pop()
            restored_history_model = refit_model
        else:
            refit_model = self.refit_with_ungrouped_levels(term, **kwargs)
            if restore_previous:
                clear_history_after_refit = True
            else:
                self.collapse_history.append(previous_model)
        try:
            self.replace_in_force_model(refit_model)
        except Exception:
            if restored_history_model is not None:
                self.collapse_history.append(restored_history_model)
            elif not clear_history_after_refit:
                self.collapse_history.pop()
            raise
        if clear_history_after_refit:
            self.collapse_history.clear()
        return refit_model

    def _ungroup_restores_reference_model(self, term: str, **kwargs: Any) -> bool:
        """Return whether ungrouping removes the last structural level collapse."""
        X_ref, _, _, _ = self._resolve_refit_data(
            kwargs.get("X"),
            kwargs.get("y"),
            kwargs.get("sample_weight"),
            kwargs.get("offset"),
        )
        replacement, _ = ungrouped_feature_spec(
            self.model,
            self._require_term(term),
            self._require_selection(term),
            X=X_ref,
        )
        return not self._has_collapsed_level_groups_after_replacement(term, replacement)

    def can_uncollapse_levels(self) -> bool:
        """Return whether a previous in-force model can be restored."""
        return bool(self.collapse_history)

    def uncollapse_levels(self):
        """Restore the previous in-force model from collapse history."""
        if not self.collapse_history:
            raise RuntimeError("No collapsed-level model is available to restore.")
        previous_model = self.collapse_history.pop()
        self.replace_in_force_model(previous_model)
        return previous_model

    def replace_in_force_model(self, model, *, with_se: bool = True) -> EditorSession:
        """Replace the editable in-force model while retaining the original reference model."""
        train_data = self._evaluation_data.get("train")
        new_terms = self._editable_terms_from_model(
            model,
            list(self._term_names),
            n_points=self.n_points,
            centering=self.centering,
            with_se=with_se,
            train_data=train_data,
        )
        new_selection = {name: np.array([], dtype=np.intp) for name in new_terms}

        old_model = self.model
        old_terms = self.terms
        old_selection = self._selection
        old_history = self.history
        old_redo_stack = self.redo_stack
        old_level_orders = self._level_orders

        try:
            self.model = model
            self.terms = new_terms
            self._selection = new_selection
            self.history = []
            self.redo_stack = []
            self._level_orders = {name: list(labels) for name, labels in old_level_orders.items()}
            self._reapply_level_orders()
        except Exception:
            self.model = old_model
            self.terms = old_terms
            self._selection = old_selection
            self.history = old_history
            self.redo_stack = old_redo_stack
            self._level_orders = old_level_orders
            raise

        old_history.clear()
        old_redo_stack.clear()
        self.history = old_history
        self.redo_stack = old_redo_stack
        self._advance_model_revision()
        return self

    # Save/load stores the edit artifact, not the fitted model. The model passed
    # to load supplies the feature definitions and grids used for validation.
    def save(self, path: str | Path) -> None:
        """Write an auditable JSON edit artifact."""
        persistence.save_session(self, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model,
    ) -> EditorSession:
        """Load an edit artifact against a fitted model."""
        return persistence.load_session(cls, path, model=model)

    def widget(self):
        """Return an optional notebook widget for this session."""
        from superglm.editor.widget import EditorWidget

        return EditorWidget(self)

    # Private guards keep public operations concise and ensure every history
    # record mutates exactly one term through the same commit path.
    def _resolve_offset_terms(
        self,
        terms: str | list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if terms is None:
            names = self.edited_terms()
            if not names:
                raise ValueError("No edited terms are available to convert into an offset.")
            return names
        names = [terms] if isinstance(terms, str) else list(terms)
        for name in names:
            self._require_term(name)
        return names

    def _require_not_interaction_parent(self, terms: list[str], *, operation: str) -> None:
        blocked: dict[str, list[str]] = {}
        for term in terms:
            interactions = self._interaction_names_for_parent(term)
            if interactions:
                blocked[term] = interactions
        if not blocked:
            return
        detail = "; ".join(
            f"{term!r}: {', '.join(interactions)}" for term, interactions in blocked.items()
        )
        raise ValueError(
            f"Cannot run {operation} for term(s) used by interaction(s): {detail}. "
            "Refit a model without those interactions first."
        )

    def _interaction_names_for_parent(self, term: str) -> list[str]:
        interactions: list[str] = []
        for name, spec in getattr(self.model, "_interaction_specs", {}).items():
            if term in getattr(spec, "parent_names", ()):
                interactions.append(str(name))
        return interactions

    def _resolve_offset_frame(self, X):
        if X is not None:
            return X
        train = self._evaluation_data.get("train")
        if train is not None:
            return train.X
        X_ref = getattr(self.model, "_fit_X_ref", None)
        if X_ref is None:
            raise RuntimeError("Fit feature data was not retained on the source model.")
        return X_ref

    def _resolve_refit_data(self, X, y, sample_weight, offset):
        if (X is None) != (y is None):
            raise ValueError("Explicit refit data requires both X and y.")
        train = self._evaluation_data.get("train")
        explicit_refit_data = X is not None and y is not None
        X_ref = (
            X
            if X is not None
            else (train.X if train is not None else self._resolve_offset_frame(None))
        )
        y_ref = (
            y
            if y is not None
            else (train.y if train is not None else getattr(self.model, "_fit_y_ref", None))
        )
        if sample_weight is not None:
            sample_weight_ref = sample_weight
        elif train is not None and not explicit_refit_data:
            sample_weight_ref = train.sample_weight
        elif explicit_refit_data:
            sample_weight_ref = None
        else:
            sample_weight_ref = getattr(self.model, "_fit_sample_weight_ref", None)
        if offset is not None:
            base_offset = offset
        elif train is not None and not explicit_refit_data:
            base_offset = train.offset
        elif explicit_refit_data:
            base_offset = None
        else:
            base_offset = getattr(self.model, "_fit_offset", None)
        return X_ref, y_ref, sample_weight_ref, base_offset

    def _require_term(self, term: str) -> EditableTerm:
        try:
            return self.terms[term]
        except KeyError as exc:
            raise KeyError(f"Unknown editable term: {term!r}") from exc

    def _require_selection(self, term: str) -> NDArray[np.intp]:
        self._require_term(term)
        idx = self._selection[term]
        if idx.size == 0:
            raise ValueError(f"No points selected for term {term!r}.")
        return idx

    def _require_edit_selection(self, term: str) -> NDArray[np.intp]:
        return self._expand_collapsed_level_indices(term, self._require_selection(term))

    def _selection_or_all(self, term: str) -> NDArray[np.intp]:
        editable = self._require_term(term)
        idx = self._selection[term]
        if idx.size == 0:
            return np.arange(editable.size, dtype=np.intp)
        return idx

    def _edit_selection_or_all(self, term: str) -> NDArray[np.intp]:
        idx = self._selection_or_all(term)
        return self._expand_collapsed_level_indices(term, idx)

    def _expand_collapsed_level_indices(self, term: str, idx: NDArray[np.intp]) -> NDArray[np.intp]:
        if idx.size == 0:
            return np.unique(idx)
        grouped = self._collapsed_level_index_groups(term)
        if not grouped:
            return np.unique(idx)
        selected: set[int] = set()
        for index in idx:
            selected.update(grouped.get(int(index), (int(index),)))
        return np.array(sorted(selected), dtype=np.intp)

    def _expand_collapsed_level_assignments(
        self,
        term: str,
        idx: NDArray[np.intp],
        values: NDArray[np.float64],
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        grouped = self._collapsed_level_index_groups(term)
        if idx.size == 0 or not grouped:
            return idx, values
        assignments: dict[int, float] = {}
        for raw_index, raw_value in zip(idx, values, strict=True):
            value = float(raw_value)
            for expanded_index in grouped.get(int(raw_index), (int(raw_index),)):
                existing = assignments.get(expanded_index)
                if existing is not None and not np.isclose(
                    existing,
                    value,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Conflicting values were supplied for collapsed level group "
                        f"in term {term!r}."
                    )
                assignments[expanded_index] = value
        expanded_idx = np.array(sorted(assignments), dtype=np.intp)
        expanded_values = np.array([assignments[int(index)] for index in expanded_idx])
        return expanded_idx, expanded_values

    def _collapsed_level_index_groups(self, term: str) -> dict[int, tuple[int, ...]]:
        editable = self._require_term(term)
        if editable.levels is None or self.model is None:
            return {}
        spec = getattr(self.model, "_specs", {}).get(term)
        grouping = getattr(spec, "_grouping", None)
        if grouping is None:
            return {}
        level_to_index = {str(level): i for i, level in enumerate(editable.levels)}
        index_groups: dict[int, tuple[int, ...]] = {}
        for group_label in grouping.grouped_levels:
            members = [
                level_to_index[str(level)]
                for level in grouping.group_to_originals.get(group_label, [])
                if str(level) in level_to_index
            ]
            if len(members) < 2:
                continue
            member_group = tuple(sorted(set(int(index) for index in members)))
            for index in member_group:
                index_groups[index] = member_group
        return index_groups

    def _require_control_term(self, term: str) -> EditableTerm:
        editable = self._require_term(term)
        if editable.x is None or editable.levels is not None:
            raise TypeError(f"Term {term!r} does not expose spline control handles.")
        if str(editable.metadata.get("term_type", editable.kind)) != "spline":
            raise TypeError(f"Term {term!r} does not expose spline control handles.")
        return editable

    def _apply_level_order(
        self,
        term: str,
        order: NDArray[np.intp],
        *,
        persist: bool = True,
    ) -> None:
        editable = self._require_term(term)
        inverse = {int(old): new for new, old in enumerate(order)}
        editable.original_log_effect = editable.original_log_effect[order]
        editable.edited_log_effect = editable.edited_log_effect[order]
        editable.x = np.arange(editable.size, dtype=np.float64)
        editable.levels = [editable.levels[int(i)] for i in order] if editable.levels else None
        native_original = editable.metadata.get("native_original_log_effect")
        if native_original is not None:
            native = np.asarray(native_original, dtype=np.float64).ravel()
            if native.shape == (editable.size,):
                editable.metadata["native_original_log_effect"] = native[order].tolist()
        if editable.weights is not None:
            editable.weights = editable.weights[order]
        if editable.ci_lower_log_effect is not None:
            editable.ci_lower_log_effect = editable.ci_lower_log_effect[order]
        if editable.ci_upper_log_effect is not None:
            editable.ci_upper_log_effect = editable.ci_upper_log_effect[order]
        self._selection[term] = np.array(
            sorted(inverse[int(i)] for i in self._selection[term]),
            dtype=np.intp,
        )
        for stack in (self.history, self.redo_stack):
            for record in stack:
                if record.term == term:
                    record.indices = np.array(
                        [inverse[int(i)] for i in record.indices], dtype=np.intp
                    )
        if persist and editable.levels is not None:
            native_order = self._model_level_order(term)
            if list(editable.levels) == native_order:
                self._level_orders.pop(term, None)
            else:
                self._level_orders[term] = [str(level) for level in editable.levels]

    def _reapply_level_orders(self) -> None:
        for term, labels in list(self._level_orders.items()):
            if (
                term not in self.terms
                or self.terms[term].levels is None
                or self._is_ordered_level_term(term)
            ):
                self._level_orders.pop(term, None)
                continue
            order = level_order_for_labels(self.terms[term].levels, labels)
            if order != list(range(self.terms[term].size)):
                self._apply_level_order(term, np.asarray(order, dtype=np.intp), persist=False)
            if self.terms[term].levels == self._model_level_order(term):
                self._level_orders.pop(term, None)
            else:
                self._level_orders[term] = [str(level) for level in self.terms[term].levels]

    def _model_level_order(self, term: str) -> list[str]:
        source_model = self.reference_model if self.reference_model is not None else self.model
        if source_model is None:
            return []
        ti = source_model.term_inference(
            term,
            with_se=False,
            n_points=self.n_points,
            centering=self.centering,
        )
        fresh = term_from_inference(ti)
        return [] if fresh.levels is None else [str(level) for level in fresh.levels]

    def _is_ordered_level_term(self, term: str) -> bool:
        from superglm.features.ordered_categorical import OrderedCategorical

        editable = self._require_term(term)
        if str(editable.metadata.get("term_type", editable.kind)) == "ordered categorical":
            return True
        source_model = self.reference_model if self.reference_model is not None else self.model
        spec = None if source_model is None else getattr(source_model, "_specs", {}).get(term)
        return isinstance(spec, OrderedCategorical)

    def _has_collapsed_level_groups_after_replacement(self, term: str, replacement) -> bool:
        for name, spec in self.model._specs.items():
            candidate = replacement if name == term else spec
            grouping = getattr(candidate, "_grouping", None)
            if grouping is None:
                continue
            if any(
                len([str(member) for member in grouping.group_to_originals.get(label, [])]) > 1
                for label in grouping.grouped_levels
            ):
                return True
        return False

    def _model_has_collapsed_level_groups(self, model) -> bool:
        for spec in getattr(model, "_specs", {}).values():
            grouping = getattr(spec, "_grouping", None)
            if grouping is None:
                continue
            if any(
                len([str(member) for member in grouping.group_to_originals.get(label, [])]) > 1
                for label in grouping.grouped_levels
            ):
                return True
        return False

    def _commit(
        self,
        term: str,
        operation: str,
        indices: NDArray[np.intp],
        before: NDArray,
        after: NDArray,
        params: dict[str, Any],
    ) -> None:
        indices = np.asarray(indices, dtype=np.intp).copy()
        before = np.asarray(before, dtype=np.float64).copy()
        after = np.asarray(after, dtype=np.float64).copy()
        changed = not np.array_equal(before, after)
        self.terms[term].edited_log_effect[indices] = after
        self.history.append(
            EditRecord(
                term=term,
                operation=operation,
                indices=indices,
                before=before,
                after=after,
                params=dict(params),
            )
        )
        self.redo_stack.clear()
        if changed:
            self._advance_model_revision()

    def _pop_record(self, records: list[EditRecord], term: str | None) -> EditRecord | None:
        if term is None:
            return records.pop() if records else None
        self._require_term(term)
        for i in range(len(records) - 1, -1, -1):
            if records[i].term == term:
                return records.pop(i)
        return None

    def _clear_term_history(self, term: str) -> None:
        self._require_term(term)
        self.history = [record for record in self.history if record.term != term]
        self.redo_stack = [record for record in self.redo_stack if record.term != term]

    def _trim_term_history(self, term: str, reset_indices: NDArray[np.intp]) -> None:
        self._require_term(term)
        reset = set(np.asarray(reset_indices, dtype=np.intp).tolist())
        self.history = self._records_without_indices(self.history, term, reset)
        self.redo_stack = self._records_without_indices(self.redo_stack, term, reset)

    @staticmethod
    def _records_without_indices(
        records: list[EditRecord],
        term: str,
        reset: set[int],
    ) -> list[EditRecord]:
        out: list[EditRecord] = []
        for record in records:
            if record.term != term:
                out.append(record)
                continue
            keep = np.array([int(index) not in reset for index in record.indices], dtype=bool)
            if not bool(np.any(keep)):
                continue
            out.append(
                EditRecord(
                    term=record.term,
                    operation=record.operation,
                    indices=record.indices[keep].copy(),
                    before=record.before[keep].copy(),
                    after=record.after[keep].copy(),
                    params=dict(record.params),
                )
            )
        return out

    def _level_selected(self, term: str, operation: str, mode: str) -> EditorSession:
        idx = self._require_edit_selection(term)
        editable = self._require_term(term)
        before = editable.edited_log_effect[idx].copy()
        if mode == "left":
            value = float(before[0])
        elif mode == "right":
            value = float(before[-1])
        elif mode == "highest":
            value = float(np.max(before))
        elif mode == "lowest":
            value = float(np.min(before))
        else:  # pragma: no cover - internal guard
            raise ValueError(f"Unknown level mode: {mode!r}")
        after = np.full(idx.size, value, dtype=np.float64)
        self._commit(term, operation, idx, before, after, {"value": value})
        return self


def edit(
    model,
    terms: list[str] | tuple[str, ...] | None = None,
    *,
    n_points: int = 200,
    centering: str = "native",
    with_se: bool = True,
) -> EditorSession:
    """Create an editor session for a fitted model."""
    return EditorSession.from_model(
        model,
        terms=terms,
        n_points=n_points,
        centering=centering,
        with_se=with_se,
    )
