"""Python state core for editing fitted 1D SuperGLM effects."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.distributional.response import require_scalar_response
from superglm.editor import apply, persistence, staging, unseen
from superglm.editor._types import (
    EditableTerm,
    EditRecord,
    PendingStep,
    SessionState,
    StructuralStep,
    new_step_id,
)
from superglm.editor.collapse import (
    collapsed_feature_spec,
    ungrouped_feature_spec,
)
from superglm.editor.controls import (
    CONTROL_HANDLE_TERM_TYPES,
    OrderedSplineGeometry,
    control_curve_after_move,
    least_change_coefficients,
    ordered_control_after_move,
    ordered_control_points,
    ordered_spline_geometry,
)
from superglm.editor.controls import control_points as _control_points
from superglm.editor.cv import _fit_in_force, check_cv_data
from superglm.editor.errors import (
    EditorClientError,
    EditorIndexError,
    EditorKeyError,
    EditorTypeError,
    EditorValueError,
)
from superglm.editor.evaluation import (
    EvaluationDataset,
    coerce_dataset,
    coerce_evaluation_data,
    default_metrics_dataset,
    training_export_dataset,
)
from superglm.editor.evaluation_cache import EditMaterializationRequest
from superglm.editor.level_order import (
    level_order_for_direction,
    level_order_for_labels,
    level_order_for_target,
)
from superglm.editor.lss.apply import (
    absorbed_message,
    kept_inside_rules,
    leaves_curve,
    moves_only_empty_values,
    prepare_ruled_edit,
)
from superglm.editor.lss.availability import (
    follow_training_rows,
    require_available,
)
from superglm.editor.lss.open import open_lss, refuse_unopenable_fit
from superglm.editor.lss.refit import lss_refit_with_drafts
from superglm.editor.lss.rules import holds_rules
from superglm.editor.lss.terms import lss_editable_terms
from superglm.editor.lss.view import LSSModelView, is_lss, predictor_of
from superglm.editor.operations import (
    anchored_smooth_values,
    monotone_clamp_values,
    running_extreme_values,
)
from superglm.editor.refit import EXPLICIT_PENALTY_ATTRIBUTE, fit_refit_model
from superglm.editor.shapes import shape_support
from superglm.editor.staging import _COLLAPSE_SENTENCES, _range_refusal
from superglm.editor.terms import (
    resolve_refit_method,
    term_from_inference,
    term_offset_values,
    term_type_from_spec,
    term_weights_from_data,
    term_weights_from_fit,
)
from superglm.editor.unseen import UnseenChoice
from superglm.features.rebuild import base_names_level, clone_with_replaced_features
from superglm.model_selection import CrossValidationResult
from superglm.solvers.dispersion import model_weight_semantics

# Re-profiling replaces the model and cannot be undone, so it would drop
# waiting changes with no way back.
_PROFILE_WHILE_WAITING = "Refit or undo the waiting changes before re-profiling."


def _edit_span(term: EditableTerm, indices: NDArray[np.intp]) -> dict[str, Any]:
    """The first and last edited point by axis position, as ``{"lo", "hi"}``.

    Named as a shaped range names its edges: the level on a categorical axis,
    in the order shown when the edit was made, and the x on a numeric one.
    Levels with others between them are no stretch of the axis, so they are
    also listed, in axis order, as ``"levels"``, which History names instead.
    Empty when the edit touched no point or the term has no axis values.
    """
    if indices.size == 0:
        return {}
    if term.levels is not None:
        positions = np.unique(indices)
        span: dict[str, Any] = {
            "lo": str(term.levels[int(positions[0])]),
            "hi": str(term.levels[int(positions[-1])]),
        }
        if int(positions[-1] - positions[0]) + 1 != positions.size:
            span["levels"] = [str(term.levels[int(i)]) for i in positions]
        return span
    if term.x is None:
        return {}
    x = np.asarray(term.x, dtype=np.float64)[indices]
    return {"lo": float(x.min()), "hi": float(x.max())}


def _mean_abs_eta(model, parameter: str | None) -> float:
    """The weighted mean of ``|eta|`` over the rows ``model``'s fit kept, 0 when it kept none.

    ``eta`` is the linear predictor of ``parameter`` (a SuperLSS view), or of the mean (a
    SuperGLM, ``parameter`` None), read from what the fit stored, so nothing is predicted.
    """
    if parameter is None:
        mu = getattr(model, "_fit_mu", None)
        if mu is None:
            return 0.0
        eta = model._link.link(np.asarray(mu, dtype=np.float64))
        weights = getattr(model, "_fit_weights", None)
    else:
        fitted = model.lss._require_fitted()
        rows = fitted.fit_state.retained_rows
        if rows is None:
            return 0.0
        names = [state.name for state in fitted.layout.predictors]
        eta = rows.fitted_eta[:, names.index(parameter)]
        weights = rows.likelihood_weights.values
    size = np.abs(np.asarray(eta, dtype=np.float64))
    if weights is None or not float(np.sum(weights)) > 0.0:
        return float(np.mean(size)) if size.size else 0.0
    return float(np.average(size, weights=np.asarray(weights, dtype=np.float64)))


def _fits_again(replacement, fitted) -> bool:
    """Whether fitting the ungrouped ``replacement`` on ``fitted``'s rows gives ``fitted`` again.

    Both ungrouped, they must agree on the kind of term, the declared universe,
    where new levels go and the reference: the level ``fitted`` resolved, or,
    when both name a policy, the policy it resolved it from, unless a level
    binding pinned it there. A policy is never that level, even one named
    "first".
    """
    base = str(replacement.base)
    policies = not base_names_level(replacement) and not base_names_level(fitted)
    same_reference = (base_names_level(replacement) and base == str(fitted._base_level)) or (
        policies and base == str(fitted.base) and getattr(fitted, "_pinned_base", None) is None
    )
    return (
        type(replacement) is type(fitted)
        and getattr(fitted, "_grouping", None) is None
        and getattr(replacement, "_declared_levels", None)
        == getattr(fitted, "_declared_levels", None)
        and getattr(replacement, "unseen", None) == getattr(fitted, "unseen", None)
        and same_reference
    )


def _explicit_penalties(model, lambda1, lambda2) -> dict[str, Any]:
    """The penalties a Refit of ``model`` records as given (``EXPLICIT_PENALTY_ATTRIBUTE``).

    ``model``'s own record, with this Refit's ``lambda1`` and ``lambda2``
    over it as its clone applies them: ``lambda2=None`` is no smoothing
    penalty, 0.0 (``clone_without_features``). ``...`` gives none.
    """
    if lambda2 is None:
        lambda2 = 0.0
    explicit = dict(getattr(model, EXPLICIT_PENALTY_ATTRIBUTE, {}))
    explicit.update(
        (name, value)
        for name, value in (("lambda1", lambda1), ("lambda2", lambda2))
        if value is not ...
    )
    return explicit


def _warn_refit_with(name: str, replacement: str) -> None:
    warnings.warn(
        f"EditorSession.{name} is deprecated and will be removed in a future release; "
        f"use EditorSession.{replacement}, which refits and records the change as one step.",
        FutureWarning,
        stacklevel=3,
    )


def _moved_curve(stale: EditableTerm, fresh: EditableTerm) -> tuple[EditableTerm, NDArray] | None:
    """``stale``'s edits moved onto ``fresh``'s curve, and the change of curve on its grid.

    ``fresh`` is the same term read from the model's new fit, in the model's level order;
    the result keeps ``stale``'s display order. ``None`` when the grid itself changed.
    """
    if stale.kind != fresh.kind or stale.size != fresh.size:
        return None
    if stale.levels is None:
        same_grid = fresh.levels is None and (
            (stale.x is None and fresh.x is None)
            or (stale.x is not None and fresh.x is not None and np.array_equal(stale.x, fresh.x))
        )
        if not same_grid:
            return None
        order = np.arange(stale.size, dtype=np.intp)
    else:
        if fresh.levels is None or sorted(map(str, stale.levels)) != sorted(map(str, fresh.levels)):
            return None
        position = {str(level): i for i, level in enumerate(fresh.levels)}
        order = np.array([position[str(level)] for level in stale.levels], dtype=np.intp)
    original = np.asarray(fresh.original_log_effect, dtype=np.float64)[order]
    change = original - stale.original_log_effect
    moved = stale.copy()
    moved.original_log_effect = original
    moved.edited_log_effect = original + (stale.edited_log_effect - stale.original_log_effect)
    # the exposure, as the bands, is the new fit's rows'
    for band in ("weights", "ci_lower_log_effect", "ci_upper_log_effect"):
        values = getattr(fresh, band)
        setattr(moved, band, None if values is None else np.asarray(values)[order])
    moved.metadata = dict(fresh.metadata)
    native = moved.metadata.get("native_original_log_effect")
    if native is not None:
        moved.metadata["native_original_log_effect"] = np.asarray(native)[order].tolist()
    return moved, change


def _moved_record(record: EditRecord, moved: dict[str, tuple[EditableTerm, NDArray]]) -> EditRecord:
    """An undo record of a term whose curve moved, with the same change on its new curve."""
    change = moved[record.term][1][record.indices] if record.term in moved else 0.0
    return replace(record, before=record.before + change, after=record.after + change)


class FitFollow(Enum):
    """What a session does when it next acts after its model's fit was replaced in place.

    A refit or ``apply_shape_postfit`` on the model object the session holds replaces that
    object's fit, which the session learns of only when it next acts: at the entry and exit of
    each public method, when ``session_payload`` reads it, and when a widget takes its lock.
    ``EditorSession._follow_fit_in_place`` is the one move, and ``FIT_FOLLOW_CHANGES`` says
    what each outcome changes.
    """

    IN_STEP = "in_step"  # the terms were read from the fit in force: nothing to do
    CARRIED = "carried"  # every grid kept: edits and their undo records move onto the new curves
    RESTARTED = "restarted"  # a grid moved: the session starts again on the model, and says so


# The parts of a session each outcome changes. A restart also leaves a notice, which the
# warning and the page's next state carry, when it dropped edits or a selection. Either move
# also takes the refit's own rows as the training split, and says so, when the model was
# refitted on other rows than it (``_follow_training_rows``).
FIT_FOLLOW_CHANGES: dict[FitFollow, frozenset[str]] = {
    FitFollow.IN_STEP: frozenset(),
    FitFollow.CARRIED: frozenset({"terms", "history", "redo_stack", "model_revision"}),
    FitFollow.RESTARTED: frozenset(
        {"terms", "history", "redo_stack", "pending", "selection", "model_revision"}
    ),
}

_EDITS_NOT_CARRIED = (
    "The model was refitted in place and {why}, so the edits and "
    "selection made on the earlier fit could not be moved onto the new curves. The editor has "
    "started again from the refitted model. Undo cannot bring them back: the model no longer "
    "holds the fit they were made on."
)


@dataclass(frozen=True)
class _CopyOf:
    """A New levels copy, by its fit: the model it was made from, the fit identity it stands
    for (its source's when it was made), and the choice that made it."""

    fit: Any
    source: Any
    stands_for: Any
    term: str
    policy: str


class EditorFitReplacedWarning(UserWarning):
    """Edits made on a fit that a refit in place replaced could not be carried over."""


def _following_its_model(method):
    """``method``, run once the session has followed its model's fit (``FitFollow``), and
    followed again after it, since Undo or Redo can put back a model refitted meanwhile."""

    @functools.wraps(method)
    def following(self, *args, **kwargs):
        if self._acting:
            return method(self, *args, **kwargs)
        self._follow_fit_in_place()
        self._acting = True
        try:
            result = method(self, *args, **kwargs)
        finally:
            self._acting = False
        self._follow_fit_in_place()
        return result

    return following


def _each_public_method_follows_its_model(cls):
    """Every public method of ``cls`` first follows the model's fit; getters do not."""
    for name, member in list(vars(cls).items()):
        if not name.startswith("_") and inspect.isfunction(member):
            setattr(cls, name, _following_its_model(member))
    return cls


@_each_public_method_follows_its_model
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
        cv: CrossValidationResult | None = None,
        cv_data: EvaluationDataset | None = None,
        with_se: bool = True,
    ):
        # Inside a public method, or following the fit: no second follow starts (``FitFollow``).
        self._acting = False
        # Each New levels copy, by its fit: the fit identity it stands for (``_fit_identity``).
        self._fit_sources: list[_CopyOf] = []
        self._notice: str | None = None
        self.model = model
        self.reference_model = model if reference_model is None else reference_model
        # The fit the session was opened on, which a supplied ``cv=`` result describes; a refit
        # or repair of the opened model in place replaces it (``cv._on_the_opened_fit``).
        self._opened_fit = _fit_in_force(self.reference_model)
        self.terms = terms
        self.n_points = int(n_points)
        self.centering = centering
        self.with_se = bool(with_se)
        self._evaluation_data = dict(evaluation_data or {})
        # Every SuperLSS view reads its training split here, the one store the follow writes.
        for view in (self.model, self.reference_model):
            if isinstance(view, LSSModelView):
                view.share_splits(self._evaluation_data)
        self.cv_report = cv_report
        if cv is not None and not isinstance(cv, CrossValidationResult):
            raise TypeError(
                "cv= takes the CrossValidationResult that superglm.cross_validate returns, "
                "not a splitter."
            )
        if cv_data is not None and cv is None:
            raise ValueError("cv_data= holds the rows a cv= result's folds index; pass cv= too.")
        self.cv = cv
        # The rows Run CV replays a supplied result's folds on: cv_data, else the train split,
        # checked again when a follow onto other data replaces that split.
        self._cv_data = cv_data
        self.cv_check = check_cv_data(cv, cv_data, training_export_dataset(self))
        self._term_names = list(terms)
        self._selection: dict[str, NDArray[np.intp]] = {
            name: np.array([], dtype=np.intp) for name in terms
        }
        self._level_orders: dict[str, list[str]] = {}
        self.history: list[EditRecord] = []
        self.redo_stack: list[EditRecord] = []
        # Each step keeps the fitted model in force before it, for the life of
        # the session, so Undo never refits: memory grows by one fit per step.
        self.structure_history: list[StructuralStep] = []
        self.structure_redo: list[StructuralStep] = []
        # Structural changes waiting for one Refit, oldest first, and
        # the ones Undo took back, latest last. Notes are kept by step id, apart
        # from the undo states, so they survive undo and redo.
        self.pending: list[PendingStep] = []
        self.pending_redo: list[PendingStep] = []
        self.step_notes: dict[str, str] = {}
        self._model_revision = 0
        self._edit_epoch = 0
        self._materialized_edit_model = None
        self._materialized_edit_epoch: int | None = None
        # Each predictor's mean |eta| (``additive_step``), with the fit identity it was read on.
        self._eta_sizes: dict[str | None, tuple[Any, float]] = {}

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
        cv: CrossValidationResult | None = None,
        cv_data=None,
        amount_unit: str | None = None,
    ) -> EditorSession:
        """Build an editor session from fitted 1D main-effect inference.

        ``cv`` is a :func:`superglm.cross_validate` result for the
        Cross-validation tab, and ``cv_data`` the ``(X, y[, sample_weight[,
        offset]])`` rows its folds index. Without ``cv_data`` the train data
        is used when its row count matches the folds. ``cv_report`` is the
        older free-form report the Validation tab shows.

        A :class:`~superglm.SuperLSS` model needs ``train_data``, with its
        offsets a dict keyed by parameter name. Its terms are keyed
        ``"<parameter>:<term>"``; a bare column name opens that column's term
        in every parameter. ``amount_unit`` names the currency or unit of the
        response.
        """
        if is_lss(model):
            require_scalar_response(model.family, "editor")
            return open_lss(
                cls,
                model,
                terms,
                n_points=n_points,
                centering=centering,
                with_se=with_se,
                train_data=train_data,
                validation_data=validation_data,
                test_data=test_data,
                cv_report=cv_report,
                cv=cv,
                cv_data=cv_data,
                amount_unit=amount_unit,
            )
        if amount_unit is not None:
            raise EditorValueError("amount_unit applies to SuperLSS models.")
        if getattr(model, "_result", None) is None:
            raise RuntimeError("Model must be fitted before creating an editor session.")

        evaluation_data = coerce_evaluation_data(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
            family=model._distribution,
            weight_semantics=model_weight_semantics(model),
        )
        cv_rows = coerce_dataset(
            "cv",
            cv_data,
            family=model._distribution,
            weight_semantics=model_weight_semantics(model),
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
            cv=cv,
            cv_data=cv_rows,
            with_se=with_se,
        )

    @property
    def terms(self) -> dict[str, EditableTerm]:
        """The editable terms, as of the session's last action (``FitFollow``)."""
        return self._terms

    @terms.setter
    def terms(self, terms: dict[str, EditableTerm]) -> None:
        self._terms = terms
        self._terms_fit = self._fit_identity(self.model)

    def _fit_identity(self, model) -> Any:
        """The fit ``model``'s curves were read from: a New levels copy's is its source's."""
        entry = self._copy_entry(model)
        return _fit_in_force(model) if entry is None else entry.stands_for

    def _copy_entry(self, model) -> _CopyOf | None:
        fit = _fit_in_force(model)
        return next((entry for entry in self._fit_sources if entry.fit is fit), None)

    def _note_copy_of(self, copy, source, term: str, policy: str) -> None:
        """Record that ``copy`` keeps ``source``'s fit, sending ``term``'s new levels to
        ``policy``, as a New levels choice's copy does."""
        entry = _CopyOf(_fit_in_force(copy), source, self._fit_identity(source), term, policy)
        self._fit_sources.append(entry)

    def _on_its_source(self, model, rebuilt: dict[int, Any]) -> Any:
        """``model``, or, for a New levels copy whose source's fit was replaced in place since
        the copy was made, the same choice made again on that source (``rebuilt`` memoises)."""
        entry = self._copy_entry(model)
        if entry is None:
            return model
        if id(model) not in rebuilt:
            source = self._on_its_source(entry.source, rebuilt)
            fresh = model
            if source is not entry.source or self._fit_identity(source) is not entry.stands_for:
                fresh = unseen.model_with_unseen(source, entry.term, entry.policy)
                self._note_copy_of(fresh, source, entry.term, entry.policy)
            rebuilt[id(model)] = fresh
        return rebuilt[id(model)]

    def _renew_copies(self, rebuilt: dict[int, Any]) -> None:
        """Put each stale New levels copy's renewal (``_on_its_source``) in force and in the
        Undo and Redo records, so that neither puts the replaced fit back."""

        def renewed(record: EditRecord) -> EditRecord:
            if not isinstance(record, UnseenChoice):
                return record
            before = self._on_its_source(record.model_before, rebuilt)
            return replace(
                record,
                model_before=before,
                model_after=self._on_its_source(record.model_after, rebuilt),
            )

        self.model = self._on_its_source(self.model, rebuilt)
        self.history = [renewed(record) for record in self.history]
        self.redo_stack = [renewed(record) for record in self.redo_stack]

    def _take_notice(self) -> str | None:
        """The notice a restart left for the page (``FitFollow.RESTARTED``), once."""
        notice, self._notice = self._notice, None
        return notice

    def _follow_fit_in_place(self) -> FitFollow:
        """Follow a refit or ``apply_shape_postfit`` that replaced the model's fit in place.

        The one move of ``FitFollow``. Each term's edits carry over as the same changes to its
        new curve, with their undo records, and the model revision moves on. When a term's
        grid changed, nothing carries over: the session starts again on the model, as
        ``replace_in_force_model`` does, and warns when that dropped edits or a selection.
        """
        if self._acting:
            return FitFollow.IN_STEP
        # A New levels copy stands for its source's fit as it was; a replaced source makes the
        # copy stale even though its own fit is unchanged.
        rebuilt: dict[int, Any] = {}
        renewed = self._on_its_source(self.model, rebuilt)
        if renewed is self.model and self._fit_identity(self.model) is self._terms_fit:
            return FitFollow.IN_STEP
        self._acting = True
        try:
            # A follow adopts only a fit that opening would: the refit may read censored rows.
            if isinstance(self.model, LSSModelView):
                refuse_unopenable_fit(self.model.lss)
            self._renew_copies(rebuilt)
            followed = self._follow_training_rows()
            outcome = self._carry_onto_the_fit_in_force()
            if followed is not None:
                self._notice = " ".join(filter(None, (self._notice, followed)))
            return outcome
        finally:
            self._acting = False

    def _follow_training_rows(self) -> str | None:
        """Take the rows the fit in force kept as the training split (``follow_training_rows``),
        and say so when they are not the data the session held; ``None`` otherwise."""
        train = self._evaluation_data.get("train")
        if not isinstance(self.model, LSSModelView) or train is None:
            return None
        self._evaluation_data["train"], notice = follow_training_rows(self.model.lss, train)
        if notice is not None:
            # Other data: a supplied result's folds are checked against the rows now in force.
            self.cv_check = check_cv_data(self.cv, self._cv_data, training_export_dataset(self))
            warnings.warn(notice, EditorFitReplacedWarning, stacklevel=5)
        return notice

    def _carry_onto_the_fit_in_force(self) -> FitFollow:
        stale = self._terms
        fresh = self._editable_terms_from_model(
            self.model,
            list(stale),
            n_points=self.n_points,
            centering=self.centering,
            with_se=self.with_se,
            train_data=self._evaluation_data.get("train"),
        )
        moves = {name: _moved_curve(stale[name], fresh[name]) for name in stale}
        moved = {name: move for name, move in moves.items() if move is not None}
        history = [_moved_record(record, moved) for record in self.history]
        redo_stack = [_moved_record(record, moved) for record in self.redo_stack]
        taken_back = [
            name
            for name, (term, _) in moved.items()
            if holds_rules(self.model, name)
            and not self._carry_inside_rules(term, history, redo_stack)
        ]
        if len(moved) < len(moves) or taken_back:
            dropped = bool(
                self.history
                or self.redo_stack
                or self.pending
                or self.pending_redo
                or any(len(chosen) for chosen in self._selection.values())
            )
            self.replace_in_force_model(self.model, with_se=self.with_se)
            if dropped:
                regridded = ", ".join(name for name, move in moves.items() if move is None)
                reasons = [f"the grid of {regridded} changed"] if regridded else []
                if taken_back:
                    reasons.append(f"the rules of {', '.join(taken_back)} take back an edit")
                why = " and ".join(reasons)
                self._notice = _EDITS_NOT_CARRIED.format(why=why)
                warnings.warn(self._notice, EditorFitReplacedWarning, stacklevel=4)
            return FitFollow.RESTARTED
        self._terms_fit = self._fit_identity(self.model)
        self._terms = {name: term for name, (term, _) in moved.items()}
        self.history = history
        self.redo_stack = redo_stack
        self._advance_model_revision()
        return FitFollow.CARRIED

    def _carry_inside_rules(
        self, term: EditableTerm, history: list[EditRecord], redo_stack: list[EditRecord]
    ) -> bool:
        """Keep the ruled ``term``'s carried edits inside its rules, as each was kept when made.

        Its records, history then redo, are replayed in order from its new fitted curve, each
        through ``kept_inside_rules`` and recorded whole, in place in ``history`` and
        ``redo_stack``; the curve the history ends on is the one shown. False when the rules
        take a carried edit back entirely: then nothing can be carried.
        """
        current = term.original_log_effect.copy()
        shown = current
        order = [(history, i) for i in range(len(history))]
        order += [(redo_stack, i) for i in reversed(range(len(redo_stack)))]
        for records, i in order:
            record = records[i]
            if not isinstance(record, EditRecord) or record.term != term.name:
                continue
            asked = current.copy()
            asked[record.indices] = record.after
            probe = term.copy()
            probe.edited_log_effect = current
            kept = kept_inside_rules(self.model, probe, asked)
            if kept is None:
                kept = asked
            elif not np.array_equal(asked, current) and leaves_curve(kept, current, asked):
                return False
            whole = np.arange(term.size, dtype=np.intp)
            records[i] = replace(record, indices=whole, before=current.copy(), after=kept)
            current = kept
            if records is history:
                shown = current
        term.edited_log_effect = shown.copy()
        return True

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
        if is_lss(model):
            return lss_editable_terms(
                model, names, n_points=n_points, with_se=with_se, train=train_data
            )
        editable: dict[str, EditableTerm] = {}
        # The rows a shape refit reads (_resolve_refit_data).
        refit_data = (
            (train_data.X, train_data.sample_weight)
            if train_data is not None
            else (
                getattr(model, "_fit_X_ref", None),
                getattr(model, "_fit_sample_weight_ref", None),
            )
        )
        for name in names:
            if name in model._interaction_specs:
                raise EditorValueError(f"Interactions are not editable in v1: {name!r}")
            if name not in model._specs:
                raise EditorKeyError(f"Term not found: {name!r}")
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
            if term.metadata["term_type"] == "piecewise":
                # The offset-scoring helper needs to know whether the term
                # holds flat or continues its end slopes past the knot vector.
                term.metadata["extrapolation"] = model._specs[name].extrapolation
            term.weights = (
                term_weights_from_data(train_data.X, train_data.sample_weight, name, term)
                if train_data is not None
                else term_weights_from_fit(model, name, term)
            )
            # Owned by this term, so it lives exactly as long as the in-force model.
            term.metadata["shape_support"] = shape_support(model, name, term.x, *refit_data)
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
            raise EditorIndexError(f"Selection indices out of range for term {term!r}.")
        self._selection[term] = np.unique(idx)
        return self

    def select_x(self, term: str, start: float, stop: float) -> EditorSession:
        """Select grid points whose x values fall inside [start, stop]."""
        editable = self._require_term(term)
        if editable.x is None:
            raise EditorTypeError(f"Term {term!r} does not have a numeric x grid.")
        lo, hi = sorted((float(start), float(stop)))
        idx = np.flatnonzero((editable.x >= lo) & (editable.x <= hi)).astype(np.intp)
        self._selection[term] = idx
        return self

    def select_levels(self, term: str, levels: list[str] | tuple[str, ...]) -> EditorSession:
        """Select categorical or ordered-categorical levels by label."""
        editable = self._require_term(term)
        if editable.levels is None:
            raise EditorTypeError(f"Term {term!r} does not have levels.")
        level_to_idx = {level: i for i, level in enumerate(editable.levels)}
        missing = [level for level in levels if level not in level_to_idx]
        if missing:
            raise EditorKeyError(f"Unknown level(s) for term {term!r}: {missing}")
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
        if idx.size < editable.size and holds_rules(self.model, term):
            # Part of a ruled curve put back can break the rule beside it, so the
            # reset is an edit, kept inside the rule like any other.
            self._commit(term, "reset", idx, before, restored, {})
            return self
        editable.edited_log_effect[idx] = restored
        if idx.size == editable.size:
            self._clear_term_history(term)
        else:
            self._trim_term_history(term, idx)
        if not np.array_equal(before, restored):
            # A change is a new action: the undone steps' future is gone.
            self.structure_redo.clear()
            self.pending_redo.clear()
            self._advance_model_revision()
        return self

    def additive_step(self, term: str) -> float:
        """What Increase adds to ``term``'s effect on a curve that shows the effect itself.

        5% of the weighted mean of ``|eta|``, the linear predictor ``term`` adds to, over
        the fit's training rows, so on an identity link it moves an average row by about
        what x1.05 moves it on a log link. Where that mean is 0, or the fit kept no rows,
        5% of the range of the term's fitted curve, and 0.05 when that is 0 too. The mean
        is read once per fit (``_fit_identity``).
        """
        editable = self._require_term(term)
        parameter = predictor_of(self.model, term)
        fit = self._fit_identity(self.model)
        read = self._eta_sizes.get(parameter)
        if read is None or read[0] is not fit:
            read = (fit, _mean_abs_eta(self.model, parameter))
            self._eta_sizes[parameter] = read
        size = read[1]
        if not size > 0.0:
            size = float(np.ptp(editable.original_log_effect)) if editable.size else 0.0
        return 0.05 * size if size > 0.0 else 0.05

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
            raise EditorIndexError(f"Edit indices out of range for term {term!r}.")
        after = np.asarray(values, dtype=np.float64).ravel()
        if after.size != idx.size:
            raise EditorValueError(
                f"Expected {idx.size} values for term {term!r}, got {after.size}."
            )
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
                raise EditorValueError(
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
            raise EditorValueError(f"strength must be between 0 and 1, got {strength!r}")
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
            raise EditorTypeError(f"Term {term!r} does not have levels.")
        if self._is_ordered_level_term(term):
            raise EditorTypeError(f"Ordered categorical term {term!r} cannot be display-reordered.")
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
            raise EditorTypeError(f"Term {term!r} does not have levels.")
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

    def set_unseen(self, term: str, policy: str) -> EditorSession:
        """Choose where ``term``'s levels unseen at fit go when the model predicts.

        ``policy`` is ``"error"`` (refuse them), ``"base"`` (rate them at the
        reference) or the label of one of the term's groups (give them that
        group's effect). Only a plain categorical has the choice. Nothing is
        refit: the in-force model becomes a copy with the new policy, and the
        choice is one entry that Undo takes back. The body lives in
        ``superglm.editor.unseen``.
        """
        unseen.set_unseen(self, term, policy)
        return self

    # Smoothing anchors selected runs to adjacent unselected values so it does
    # not create jumps at selection edges; a monotone edit does not.
    def isotonic(self, term: str, direction: str = "increasing") -> EditorSession:
        """Make selected values monotone by holding the highest value so far, or the lowest.

        Each contiguous run holds its running extreme until the curve passes it
        (``running_extreme_values``), not tied to its unselected neighbours, so
        the edit can step at a selection edge. A categorical term also clamps to
        the level before the run. If no points are selected, the operation
        applies to the whole term.
        """
        if direction not in ("increasing", "decreasing"):
            raise EditorValueError(
                f"direction must be 'increasing' or 'decreasing', got {direction!r}"
            )
        editable = self._require_term(term)
        idx = self._edit_selection_or_all(term)
        before = editable.edited_log_effect[idx].copy()
        if editable.levels is not None:
            after = monotone_clamp_values(editable.edited_log_effect, idx, direction)
        else:
            after = running_extreme_values(editable.edited_log_effect, idx, direction)
        self._commit(term, "isotonic", idx, before, after, {"direction": direction})
        return self

    def smooth(self, term: str, strength: float = 0.5) -> EditorSession:
        """Apply light local smoothing to selected values.

        Selected contiguous runs are anchored to adjacent unselected neighbors
        when available. This avoids visible jumps at edit boundaries. If no
        points are selected, the operation applies to the whole term.
        """
        if not 0.0 <= strength <= 1.0:
            raise EditorValueError(f"strength must be between 0 and 1, got {strength!r}")
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
        editable, geometry = self._require_control_term(term)
        if geometry is not None:
            coefficients = self.ordered_spline_coefficients(term, geometry)
            return ordered_control_points(geometry, coefficients, n_handles=n_handles)
        return _control_points(self.model, editable, n_handles=n_handles)

    def ordered_spline(self, term: str) -> OrderedSplineGeometry | str | None:
        """The fitted spline of an ordered term on its level axis.

        ``None`` unless the term is an ordered categorical with a spline basis;
        a fixed sentence when its handles are off.
        """
        return ordered_spline_geometry(self.model, self._require_term(term))

    def ordered_spline_coefficients(self, term: str, geometry: OrderedSplineGeometry) -> NDArray:
        """The spline coefficients behind an ordered term's current level effects.

        Starts from the coefficients the latest handle move on the term wrote,
        else the fit's, and keeps them wherever the edits since allow
        (``least_change_coefficients``).  Undo and Redo move the history, so
        the handles follow them.
        """
        prior = next(
            (
                record.params["coefficients"]
                for record in reversed(self.history)
                if record.term == term and "coefficients" in record.params
            ),
            None,
        )
        return least_change_coefficients(
            geometry, self._require_term(term).edited_log_effect, prior
        )

    def move_control_point(
        self,
        term: str,
        handle_index: int,
        log_effect: float,
        *,
        n_handles: int | None = None,
    ) -> EditorSession:
        """Move one spline control handle vertically and refit the displayed curve."""
        editable, geometry = self._require_control_term(term)
        handle_index = int(handle_index)
        if geometry is not None:
            self._move_ordered_spline_handle(
                term, geometry, handle_index, float(log_effect), n_handles=n_handles
            )
            return self
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
            span=False,
        )
        return self

    def _move_ordered_spline_handle(
        self,
        term: str,
        geometry: OrderedSplineGeometry,
        handle_index: int,
        log_effect: float,
        *,
        n_handles: int | None,
    ) -> None:
        """Set one coefficient, and every smooth level to the spline it draws.

        The levels become ``B(level positions) @ c``. Special levels have no
        place on the spline, so the edit leaves them alone. The record keeps
        the coefficients, so the next handle starts from them.
        """
        coefficients = self.ordered_spline_coefficients(term, geometry)
        moved, column = ordered_control_after_move(
            geometry, coefficients, handle_index, log_effect, n_handles=n_handles
        )
        smooth = geometry.level_index
        self._commit(
            term,
            "control_point",
            smooth,
            self.terms[term].edited_log_effect[smooth].copy(),
            geometry.level_basis @ moved,
            {
                "handle_index": handle_index,
                "log_effect": log_effect,
                "basis": "ordered_spline",
                "basis_index": column,
                "x": float(geometry.handle_x[column]),
                "coefficients": [float(value) for value in moved],
            },
            span=False,
        )

    def undo(self, term: str | None = None) -> EditorSession:
        """Undo the latest action, in time: an edit, a waiting change or an applied step.

        Undoing a waiting change puts back the term's previous draft; the model
        and the curves are unchanged, so the model revision stays. Undoing an
        applied step puts back the whole editor state from before it, edits and
        waiting changes included, without refitting. ``term`` limits the undo
        to that term's latest edit since the last structural step.
        """
        target = self.undo_target() if term is None else None
        if isinstance(target, PendingStep):
            self.pending_redo.append(self.pending.pop())
            return self
        if isinstance(target, StructuralStep):
            self._step_across(self.structure_history, self.structure_redo)
            return self
        if not self.history:
            return self
        index = self._latest_index(self.history, term)
        if index is None:
            return self
        record = self.history[index]
        self._rewrite_history(
            [None if i == index else kept for i, kept in enumerate(self.history)], undone=True
        )
        current = self.terms[record.term].edited_log_effect[record.indices].copy()
        self.terms[record.term].edited_log_effect[record.indices] = record.before
        self.redo_stack.append(record)
        if isinstance(record, UnseenChoice):
            unseen.step_across(self, record, undo=True)
        elif not np.array_equal(current, record.before):
            self._advance_model_revision()
        return self

    def redo(self, term: str | None = None) -> EditorSession:
        """Redo the latest undone action, in the order Undo took them back."""
        target = self.redo_target() if term is None else None
        if isinstance(target, PendingStep):
            self.pending.append(self.pending_redo.pop())
            return self
        if isinstance(target, StructuralStep):
            self._step_across(self.structure_redo, self.structure_history)
            return self
        if not self.redo_stack:
            return self
        record = self._pop_record(self.redo_stack, term)
        if record is None:
            return self
        current = self.terms[record.term].edited_log_effect[record.indices].copy()
        self.terms[record.term].edited_log_effect[record.indices] = record.after
        self.history.append(record)
        if isinstance(record, UnseenChoice):
            unseen.step_across(self, record, undo=False)
        elif not np.array_equal(current, record.after):
            self._advance_model_revision()
        return self

    def undo_target(self) -> EditRecord | PendingStep | StructuralStep | None:
        """What a plain :meth:`undo` takes next: the latest edit or waiting change, else a step."""
        return staging.undo_target(self)

    def redo_target(self) -> EditRecord | PendingStep | StructuralStep | None:
        """What a plain :meth:`redo` puts back next, in the reverse of the order Undo took."""
        return staging.redo_target(self)

    # Structural changes wait for one Refit; the bodies live in
    # ``superglm.editor.staging``.
    def draft_spec(self, term: str):
        """``term``'s spec as waiting changes leave it.

        See ``superglm.editor.staging.draft_spec``.
        """
        return staging.draft_spec(self, term)

    def stage_structural(
        self,
        operation: str,
        term: str,
        params: dict[str, Any],
        *,
        keep_reference: bool = True,
        X=None,
        sample_weight=None,
    ) -> PendingStep:
        """Stage one structural change to wait for a Refit.

        See ``superglm.editor.staging.stage_structural``.
        """
        return staging.stage_structural(
            self,
            operation,
            term,
            params,
            keep_reference=keep_reference,
            X=X,
            sample_weight=sample_weight,
        )

    def refit_pending(self, *, method: str = "auto", **refit_kwargs: Any) -> StructuralStep:
        """Apply every waiting change in one fit, as one step.

        See ``superglm.editor.staging.refit_pending``.
        """
        return staging.refit_pending(self, method=method, **refit_kwargs)

    def timeline_items(self) -> tuple[list[tuple[Any, str]], list[tuple[Any, str]]]:
        """Every action, split at the current position.

        See ``superglm.editor.staging.timeline_items``.
        """
        return staging.timeline_items(self)

    def set_step_note(self, step_id: str, note: str | None) -> None:
        """Write ``note`` on the timeline entry ``step_id``; an empty note removes it."""
        staging.set_step_note(self, step_id, note)

    def editor_history_records(self) -> list[dict[str, Any]]:
        """The timeline up to now as an exported model's ``_editor_history``."""
        return staging.editor_history_records(self)

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

    def capture_materialization_request(
        self,
        *,
        dataset=None,
        include_unchanged: bool = False,
    ) -> EditMaterializationRequest | None:
        """Capture plot-scale state and immutable evaluation references."""
        if not include_unchanged and not self.edited_terms():
            return None
        return EditMaterializationRequest(
            model_revision=self.model_revision,
            edit_epoch=self.edit_epoch,
            base_model=self.model,
            terms={name: term.copy() for name, term in self.terms.items()},
            dataset=default_metrics_dataset(self) if dataset is None else dataset,
        )

    def materialization_request_is_current(self, request: EditMaterializationRequest) -> bool:
        """Return whether a captured model snapshot still identifies this session state."""
        return (
            request.model_revision == self.model_revision
            and request.edit_epoch == self.edit_epoch
            and request.base_model is self.model
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
        if not self.materialization_request_is_current(request):
            return False
        self._materialized_edit_model = model
        self._materialized_edit_epoch = request.edit_epoch
        return True

    def save_model(self, path: str | Path) -> Path:
        return persistence.save_model(self, path)

    def export_structure(self, path: str | Path | None = None) -> str:
        """The in-force model's structural decisions as JSON, written to ``path`` too when given.

        The groupings, references, polynomial ranges and where new levels go,
        per feature, as :func:`superglm.read_structure` reads them; no
        coefficients and no hand edits. Changes waiting for a Refit are not
        included: the file is the last Refit.
        """
        return persistence.export_structure(self, path)

    # Offset refits are conditional diagnostics: edited terms become fixed
    # link-scale factors and are removed from the refitted feature set.
    def edited_terms(self) -> list[str]:
        """Names of terms whose edited values differ from the source fit."""
        from superglm.editor.display import display_for, is_edited

        return [
            name
            for name, term in self.terms.items()
            if is_edited(
                display_for(self.model, name), term.original_log_effect, term.edited_log_effect
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
        require_available(self.model, "refit_offset")
        names = self._resolve_offset_terms(terms)
        X_ref = self._resolve_offset_frame(X)
        frame = as_eager_frame(X_ref)
        for name in names:
            if name not in frame.columns:
                raise EditorKeyError(f"Offset data is missing column {name!r}.")
        n = len(frame)
        offset = np.zeros(n, dtype=np.float64)
        for name in names:
            offset += term_offset_values(self.terms[name], frame.column_array(name))
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
        require_available(self.model, "refit_offset")
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
        require_available(self.model, "profile_distribution")
        if self.model is None:
            raise RuntimeError("Cannot reprofile without a source model.")
        if self.pending:
            raise EditorValueError(_PROFILE_WHILE_WAITING)
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
            raise EditorValueError("parameter must be 'tweedie_p' or 'nb2_theta'.")

        self.replace_in_force_model(profile_model)
        self.structure_history.clear()
        self.structure_redo.clear()
        return result

    def refit_with_collapsed_levels(
        self,
        term: str,
        *,
        group_label: str | None = None,
        keep_reference: bool = True,
        **refit_kwargs: Any,
    ):
        """Collapse selected categorical levels and refit a full model copy.

        ``keep_reference`` holds the in-force reference level, or the group
        that takes it in; ``False`` lets the declared base policy choose again.
        ``refit_kwargs`` are ``X``, ``y``, ``sample_weight``, ``offset``,
        ``method``, ``lambda1``, ``lambda2`` and fit keywords.

        Deprecated: use :meth:`replace_with_collapsed_levels`, which records the
        change as one step.
        """
        _warn_refit_with("refit_with_collapsed_levels", "replace_with_collapsed_levels")
        editable = self._require_term(term)
        idx = self._require_selection(term)
        try:
            return self._refit_replacing(
                term,
                lambda X_ref: collapsed_feature_spec(
                    self.model,
                    editable,
                    idx,
                    X=X_ref,
                    group_label=group_label,
                    keep_reference=keep_reference,
                ),
                **refit_kwargs,
            )
        except EditorClientError:
            raise
        except ValueError as exc:
            # Collapsing bands can take away positions a shaped range, or the
            # free curve beside it, needs; the library refuses that at the refit.
            sentence = _range_refusal(exc, _COLLAPSE_SENTENCES)
            if sentence is None:
                raise
            raise EditorValueError(sentence) from exc

    def replace_with_collapsed_levels(
        self,
        term: str,
        *,
        group_label: str | None = None,
        keep_reference: bool = True,
        **refit_kwargs: Any,
    ):
        """Collapse the selected levels and refit at once, as one structural step.

        This is :meth:`stage_structural` followed by the refit of every waiting
        change; Undo takes the whole call back. ``refit_kwargs`` are ``X``,
        ``y``, ``sample_weight``, ``offset``, ``method``, ``lambda1``,
        ``lambda2`` and fit keywords.
        """
        params = {"levels": staging.selected_labels(self, term), "group_label": group_label}
        return staging.stage_and_refit(
            self, "collapse", term, params, keep_reference=keep_reference, **refit_kwargs
        )

    def refit_with_ungrouped_levels(
        self, term: str, *, keep_reference: bool = True, **refit_kwargs: Any
    ):
        """Remove selected levels from collapsed groups and refit a model copy.

        ``keep_reference`` holds the in-force reference; a reference group that
        loses members follows the members that stay.

        Deprecated: use :meth:`replace_with_ungrouped_levels`, which records the
        change as one step.
        """
        _warn_refit_with("refit_with_ungrouped_levels", "replace_with_ungrouped_levels")
        editable = self._require_term(term)
        idx = self._require_selection(term)
        return self._refit_replacing(
            term,
            lambda X_ref: ungrouped_feature_spec(
                self.model, editable, idx, X=X_ref, keep_reference=keep_reference
            ),
            **refit_kwargs,
        )

    def replace_with_ungrouped_levels(
        self, term: str, *, keep_reference: bool = True, **refit_kwargs: Any
    ):
        """Ungroup the selected levels and refit at once, as one structural step.

        An ungroup that reproduces the fit from before the latest step puts
        that fit back instead of refitting, as ``staging.stage_and_refit``
        states.
        """
        params = {"levels": staging.selected_labels(self, term)}
        return staging.stage_and_refit(
            self, "ungroup", term, params, keep_reference=keep_reference, **refit_kwargs
        )

    def _pre_collapse_model(
        self, term: str, idx: NDArray[np.intp], *, keep_reference: bool, **kwargs: Any
    ):
        """The model before the latest step, when this ungroup reproduces it exactly.

        Only a step that did nothing but collapse ``term``'s levels qualifies,
        and only if nothing since (a New levels choice, other penalties, rows,
        fit options or method) makes the earlier fit differ from the ungroup's.
        """
        step = self.structure_history[-1] if self.structure_history else None
        if step is None or not step.changes:
            return None
        if any(isinstance(record, UnseenChoice) for record in self.history):
            return None
        if any((change.term, change.operation) != (term, "collapse") for change in step.changes):
            return None
        replacement = self._ungrouped_replacement(
            term, idx, keep_reference=keep_reference, **kwargs
        )
        if self._has_collapsed_level_groups_after_replacement(term, replacement):
            return None
        previous = step.state.model
        if self._model_has_collapsed_level_groups(previous):
            return None
        given = _explicit_penalties(
            self.model, kwargs.get("lambda1", ...), kwargs.get("lambda2", ...)
        )
        if given != getattr(previous, EXPLICIT_PENALTY_ATTRIBUTE, {}):
            return None
        options = {"method", "lambda1", "lambda2"}
        if any(value is not None for name, value in kwargs.items() if name not in options):
            return None
        method = resolve_refit_method(self.model, kwargs.get("method", "auto"))
        if method != resolve_refit_method(previous, "auto"):
            return None
        return previous if _fits_again(replacement, previous._specs[term]) else None

    def _ungrouped_replacement(
        self, term: str, idx: NDArray[np.intp], *, keep_reference: bool, **kwargs: Any
    ):
        """The spec ungrouping the levels at display indices ``idx`` refits ``term`` with."""
        X_ref, _, _, _ = self._resolve_refit_data(
            kwargs.get("X"),
            kwargs.get("y"),
            kwargs.get("sample_weight"),
            kwargs.get("offset"),
        )
        replacement, _ = ungrouped_feature_spec(
            self.model,
            self._require_term(term),
            idx,
            X=X_ref,
            keep_reference=keep_reference,
        )
        return replacement

    def replace_with_reference_level(self, term: str, level: str, **refit_kwargs: Any):
        """Pin ``level`` as ``term``'s reference and refit at once, as one structural step."""
        return staging.stage_and_refit(
            self, "set_reference", term, {"level": level}, **refit_kwargs
        )

    def replace_with_special_levels(
        self, term: str, levels: list[str], *, special: bool = True, **refit_kwargs: Any
    ):
        """Take ``levels`` off ``term``'s curve (or, ``special=False``, put them back) and refit.

        One structural step, which one Undo takes back. ``levels`` are display
        labels of an ordered term; a level taken off is fitted with a free
        estimate of its own, as ``specials=`` declares.
        """
        operation = "special" if special else "on_curve"
        return staging.stage_and_refit(
            self, operation, term, {"levels": list(levels)}, **refit_kwargs
        )

    def replace_with_knots(self, term: str, params: dict[str, Any], **refit_kwargs: Any):
        """Change ``term``'s knots and refit at once, as one structural step.

        ``params`` is ``{"count", "strategy"}`` with an optional ``"alpha"``,
        ``{"positions": [...]}`` in chart coordinates, or ``{"reset": True}``
        for the opened model's knots (``superglm.editor.knots``).
        """
        return staging.stage_and_refit(self, "knots", term, dict(params), **refit_kwargs)

    def replace_with_basis(self, term: str, params: dict[str, Any], **refit_kwargs: Any):
        """Change ``term``'s basis and refit at once, as one structural step.

        ``params`` is ``{"kind": "ps" | "bs" | "cr" | "ns"}`` or
        ``{"select": True | False}``, the double penalty that lets the fit
        shrink the term towards a straight line and to nothing
        (``superglm.editor.basis``).
        """
        return staging.stage_and_refit(self, "basis", term, dict(params), **refit_kwargs)

    def replace_with_shaped_range(
        self, term: str, *, lo, hi, degree: int, join: str = "tangent", **refit_kwargs: Any
    ):
        """Pin ``term`` to a ``degree`` polynomial on ``[lo, hi]`` and refit at once.

        ``lo`` and ``hi`` are values on a numeric term (snapped outward to
        three significant figures of the fitted span) and band labels on an
        ordered one. ``join`` is ``"tangent"`` or ``"kink"`` (Corner).
        """
        params = {"lo": lo, "hi": hi, "degree": degree, "join": join}
        return staging.stage_and_refit(self, "shape", term, params, **refit_kwargs)

    def revert_to_reference_model(self):
        """Put the opened model back in force, in its own level order, as one structural step."""
        return self._push_structure(
            self.reference_model,
            operation="revert_to_original",
            term=None,
            label="revert to original model",
            level_orders={},
        )

    def _refit_replacing(
        self,
        term: str,
        build: Callable[[Any], tuple[Any, dict[str, Any]]],
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
        """Refit a copy of the model with ``term``'s spec replaced by ``build(X_ref)``.

        ``build`` returns the fresh replacement spec and the step's metadata;
        the metadata, with the resolved fit method, is stamped on the refit.
        """
        if self.model is None:
            raise RuntimeError("Cannot refit without a source model.")
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X, y, sample_weight, offset
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")
        replacement, metadata = build(X_ref)
        refit_model, metadata["method"] = self._refit_with_drafts(
            {term: replacement},
            X=X_ref,
            y=y_ref,
            sample_weight=sample_weight_ref,
            offset=base_offset,
            method=method,
            lambda1=lambda1,
            lambda2=lambda2,
            **fit_kwargs,
        )
        refit_model._editor_step = metadata
        return refit_model

    def _refit_with_drafts(
        self,
        drafts: dict[str, Any],
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
        """Fit one copy of the model with each term's spec replaced by its draft.

        Returns the fitted copy and the fit method used; the session itself is
        not changed. The waiting changes' Refit (:func:`staging.refit_pending`)
        fits through here.
        """
        X_ref, y_ref, sample_weight_ref, base_offset = self._resolve_refit_data(
            X, y, sample_weight, offset
        )
        if y_ref is None:
            raise RuntimeError("Fit response data was not retained on the source model.")
        if is_lss(self.model):
            refit_model = lss_refit_with_drafts(
                self.model,
                drafts,
                X=X_ref,
                y=y_ref,
                sample_weight=sample_weight_ref,
                offsets=base_offset,
                method=method,
                lambda1=lambda1,
                lambda2=lambda2,
                **fit_kwargs,
            )
            return refit_model, refit_model._fit_settings["method"]
        refit_model = clone_with_replaced_features(
            self.model, drafts, lambda1=lambda1, lambda2=lambda2
        )
        method_used = fit_refit_model(
            self.model,
            refit_model,
            method=method,
            X=X_ref,
            y=y_ref,
            sample_weight=sample_weight_ref,
            offset=base_offset,
            fit_kwargs=fit_kwargs,
        )
        explicit = _explicit_penalties(self.model, lambda1, lambda2)
        if explicit:
            setattr(refit_model, EXPLICIT_PENALTY_ATTRIBUTE, explicit)
        return refit_model, method_used

    def _push_structure(
        self,
        model,
        *,
        operation: str,
        term: str | None,
        label: str,
        level_orders: dict[str, list[str]] | None = None,
        state: SessionState | None = None,
        changes: tuple[PendingStep, ...] = (),
        step_id: str | None = None,
    ):
        """Put ``model`` in force as one structural step on the undo timeline.

        The step keeps the state before it: ``state`` when the caller captured
        it earlier (a change refitted at once is staged after that capture),
        else the live one. Like any new action, it ends the future of whatever
        was undone, so the kept state holds no redo. ``changes`` are the waiting
        changes a refit applied; ``step_id`` names the step after the one change
        it stands for.
        """
        kept = replace(
            self._capture_state() if state is None else state, redo_stack=[], pending_redo=()
        )
        step = StructuralStep(
            kept,
            operation,
            term,
            label,
            changes=tuple(changes),
            step_id=new_step_id() if step_id is None else step_id,
        )
        self.replace_in_force_model(model, level_orders=level_orders)
        self.structure_history.append(step)
        self.structure_redo.clear()
        return model

    def _capture_state(self) -> SessionState:
        """The live state, its lists copied: a step clears the live ones in place."""
        return SessionState(
            model=self.model,
            terms=self.terms,
            selection=dict(self._selection),
            level_orders={name: list(labels) for name, labels in self._level_orders.items()},
            history=list(self.history),
            redo_stack=list(self.redo_stack),
            pending=tuple(self.pending),
            pending_redo=tuple(self.pending_redo),
            fit=self._terms_fit,
            train=self._evaluation_data.get("train"),
        )

    def _step_across(self, source: list[StructuralStep], target: list[StructuralStep]) -> None:
        """Move the latest step from ``source`` to ``target``, swapping in the state it holds."""
        step = source.pop()
        target.append(replace(step, state=self._capture_state()))
        self.model = step.state.model
        self.terms = step.state.terms
        self._terms_fit = step.state.fit
        self._selection = step.state.selection
        self._level_orders = step.state.level_orders
        self.history = step.state.history
        self.redo_stack = step.state.redo_stack
        self.pending = list(step.state.pending)
        self.pending_redo = list(step.state.pending_redo)
        train = step.state.train
        if train is not None and train is not self._evaluation_data.get("train"):
            # The rows this side of the step was on; Run CV's rows are checked against them.
            self._evaluation_data["train"] = train
            self.cv_check = check_cv_data(self.cv, self._cv_data, training_export_dataset(self))
        self._advance_model_revision()

    def replace_in_force_model(
        self,
        model,
        *,
        with_se: bool = True,
        level_orders: dict[str, list[str]] | None = None,
    ) -> EditorSession:
        """Replace the editable in-force model while retaining the original reference model.

        Display level reorders carry over unless ``level_orders`` replaces them.
        Edits and waiting changes do not carry over. A SuperLSS is wrapped in the
        view of the model in force, over the same rows.
        """
        if is_lss(self.model) and not isinstance(model, LSSModelView):
            model = self.model.with_model(model)
        if isinstance(model, LSSModelView):
            model.share_splits(self._evaluation_data)
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
        old_terms, old_terms_fit = self._terms, self._terms_fit
        old_selection = self._selection
        old_history = self.history
        old_redo_stack = self.redo_stack
        old_pending, old_pending_redo = self.pending, self.pending_redo
        old_level_orders = self._level_orders
        carried_orders = old_level_orders if level_orders is None else level_orders

        try:
            self.model = model
            self.terms = new_terms
            self._selection = new_selection
            self.history = []
            self.redo_stack = []
            self.pending, self.pending_redo = [], []
            self._level_orders = {name: list(labels) for name, labels in carried_orders.items()}
            self._reapply_level_orders()
        except Exception:
            self.model = old_model
            self._terms, self._terms_fit = old_terms, old_terms_fit
            self._selection = old_selection
            self.history = old_history
            self.redo_stack = old_redo_stack
            self.pending, self.pending_redo = old_pending, old_pending_redo
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
        """Load an edit artifact against a fitted model.

        The artifact holds curve edits only. When the session it came from had
        structural steps or New levels choices in force, one warning names
        them: they were not restored. Those a revert to the original model
        undid are not named: loaded against that model, nothing is missing.
        """
        return persistence.load_session(cls, path, model=model)

    def widget(self, mode: str | None = None):
        """Return the editor app for this session, to display in a notebook.

        ``mode="server"`` serves the app from a local server, which the
        browser must be able to reach, as it can when it runs on the
        kernel's machine. ``mode="notebook"`` runs it inside the notebook
        cell over widget messages, for hosted notebooks such as Databricks,
        and needs ``pip install 'superglm[notebook]'``. ``None`` picks
        notebook on Databricks and server elsewhere; on Databricks without
        anywidget it keeps server and warns with the install line.
        """
        from superglm.editor.widget import EditorWidget

        return EditorWidget(self, mode=mode)

    # Private guards keep public operations concise and ensure every history
    # record mutates exactly one term through the same commit path.
    def _resolve_offset_terms(
        self,
        terms: str | list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        if terms is None:
            names = self.edited_terms()
            if not names:
                raise EditorValueError("No edited terms are available to convert into an offset.")
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
        raise EditorValueError(
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
            raise EditorValueError("Explicit refit data requires both X and y.")
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
            raise EditorKeyError(f"Unknown editable term: {term!r}") from exc

    def _require_selection(self, term: str) -> NDArray[np.intp]:
        self._require_term(term)
        idx = self._selection[term]
        if idx.size == 0:
            raise EditorValueError(f"No points selected for term {term!r}.")
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
                    raise EditorValueError(
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

    def _require_control_term(self, term: str) -> tuple[EditableTerm, OrderedSplineGeometry | None]:
        """The term and, for an ordered spline, its geometry; refuse a term without handles."""
        editable = self._require_term(term)
        geometry = ordered_spline_geometry(self.model, editable)
        if isinstance(geometry, OrderedSplineGeometry):
            return editable, geometry
        if isinstance(geometry, str):
            raise EditorTypeError(geometry)
        if editable.x is None or editable.levels is not None:
            raise EditorTypeError(f"Term {term!r} does not expose spline control handles.")
        if str(editable.metadata.get("term_type", editable.kind)) not in CONTROL_HANDLE_TERM_TYPES:
            raise EditorTypeError(f"Term {term!r} does not expose spline control handles.")
        return editable, None

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

    def _prepare_edit(self, term: str) -> None:
        """Build what the first edit of ``term`` reads, for a term a page has just shown."""
        prepare_ruled_edit(self.model, term)

    def _commit(
        self,
        term: str,
        operation: str,
        indices: NDArray[np.intp],
        before: NDArray,
        after: NDArray,
        params: dict[str, Any],
        *,
        span: bool = True,
    ) -> None:
        """Apply and record one edit.

        An edit on a selection also records, as ``lo`` and ``hi``, the
        stretch of axis it changed, which History names; ``span=False`` for
        a handle move, which changes the whole curve.

        An edit of a SuperLSS term with shape rules is kept inside them: the
        whole curve becomes the nearest one the rules allow
        (``kept_inside_rules``), and the record holds all of it. An edit the
        rules take back entirely is not recorded and is refused with the reason.
        """
        indices = np.asarray(indices, dtype=np.intp).copy()
        before = np.asarray(before, dtype=np.float64).copy()
        after = np.asarray(after, dtype=np.float64).copy()
        editable = self.terms[term]
        recorded = {**params, **_edit_span(editable, indices)} if span else dict(params)
        curve = editable.edited_log_effect.copy()
        curve[indices] = after
        kept = kept_inside_rules(self.model, editable, curve)
        if kept is not None:
            # The projection floors a value without rows at a tiny weight, so an edit of only
            # such values can move the curve by a few billionths: that is still none of it.
            if leaves_curve(kept, editable.edited_log_effect, curve) or moves_only_empty_values(
                editable, curve
            ):
                # The rules took the whole edit back: there is nothing to record or undo,
                # and an edit that asked for a change a chart shows says why it made none.
                # A rounding-scale ask, "Make increasing" on a curve its rule already holds
                # flat in places, is no change at all.
                if leaves_curve(curve, editable.edited_log_effect, curve):
                    return
                raise EditorValueError(absorbed_message(self.model, editable, curve))
            indices = np.arange(editable.size, dtype=np.intp)
            before, after = editable.edited_log_effect.copy(), kept
        changed = not np.array_equal(before, after)
        editable.edited_log_effect[indices] = after
        self.history.append(
            EditRecord(
                term=term,
                operation=operation,
                indices=indices,
                before=before,
                after=after,
                params=recorded,
            )
        )
        self._end_redo()
        if changed:
            self._advance_model_revision()

    def _end_redo(self) -> None:
        """A new action ends the future of whatever was undone."""
        self.redo_stack.clear()
        self.pending_redo.clear()
        self.structure_redo.clear()

    def _pop_record(self, records: list[EditRecord], term: str | None) -> EditRecord | None:
        if term is None:
            return records.pop() if records else None
        self._require_term(term)
        for i in range(len(records) - 1, -1, -1):
            if records[i].term == term:
                return records.pop(i)
        return None

    def _latest_index(self, records: list[EditRecord], term: str | None) -> int | None:
        """Index of the latest record, or of ``term``'s latest; None when there is none."""
        if term is None:
            return len(records) - 1 if records else None
        self._require_term(term)
        return next((i for i in range(len(records) - 1, -1, -1) if records[i].term == term), None)

    def _rewrite_history(self, records: list[EditRecord | None], *, undone: bool = False) -> None:
        """Keep ``records``' survivors as the history; waiting changes keep their place.

        ``records`` is the history with each dropped record replaced by None. A
        change staged after k records now follows the survivors among them.
        A record Undo takes back (``undone``) is not lost: it is redone before
        any change Undo took back earlier, so those changes keep their places.
        """
        kept = np.cumsum([0, *(record is not None for record in records)])
        dropped = np.arange(kept.size) - kept

        def moved(step: PendingStep) -> PendingStep:
            shift = int(dropped[min(step.history_position, len(records))])
            return (
                step
                if shift == 0
                else replace(step, history_position=step.history_position - shift)
            )

        self.pending = [moved(step) for step in self.pending]
        if not undone:
            self.pending_redo = [moved(step) for step in self.pending_redo]
        self.history = [record for record in records if record is not None]

    def _clear_term_history(self, term: str) -> None:
        self._require_term(term)
        # A New levels choice moves no value, so a reset of the values keeps it.
        self._rewrite_history(
            [
                None if record.term == term and not isinstance(record, UnseenChoice) else record
                for record in self.history
            ]
        )
        self.redo_stack = [record for record in self.redo_stack if record.term != term]

    def _trim_term_history(self, term: str, reset_indices: NDArray[np.intp]) -> None:
        self._require_term(term)
        reset = set(np.asarray(reset_indices, dtype=np.intp).tolist())
        self._rewrite_history(
            [self._record_without_indices(record, term, reset) for record in self.history]
        )
        trimmed = (self._record_without_indices(record, term, reset) for record in self.redo_stack)
        self.redo_stack = [record for record in trimmed if record is not None]

    @staticmethod
    def _record_without_indices(
        record: EditRecord, term: str, reset: set[int]
    ) -> EditRecord | None:
        """``record`` less ``term``'s reset points, keeping its id; None when nothing is left."""
        if record.term != term or isinstance(record, UnseenChoice):
            return record
        keep = np.array([int(index) not in reset for index in record.indices], dtype=bool)
        if not bool(np.any(keep)):
            return None
        return replace(
            record,
            indices=record.indices[keep].copy(),
            before=record.before[keep].copy(),
            after=record.after[keep].copy(),
            params=dict(record.params),
        )

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
    train_data=None,
    validation_data=None,
    test_data=None,
    cv: CrossValidationResult | None = None,
    cv_data=None,
    amount_unit: str | None = None,
) -> EditorSession:
    """Create an editor session for a fitted model.

    The split data are ``(X, y[, sample_weight[, offset]])`` tuples.
    ``cv`` is a :func:`superglm.cross_validate` result for the
    Cross-validation tab and ``cv_data`` the rows its folds index; see
    :meth:`EditorSession.from_model`, which also describes opening a
    :class:`~superglm.SuperLSS` model (``amount_unit``).
    """
    return EditorSession.from_model(
        model,
        terms=terms,
        n_points=n_points,
        centering=centering,
        with_se=with_se,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        cv=cv,
        cv_data=cv_data,
        amount_unit=amount_unit,
    )
