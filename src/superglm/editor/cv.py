"""The Cross-validation tab: a supplied result, Run CV and Final fit.

``edit(model, cv=result)`` hands the editor a :class:`CrossValidationResult`.
The tab shows its fold scores and, when it kept its fold models, each
term's relativities by fold, least stable first. Run CV replays the
result's own folds on the in-force structure with the hand edits put back
on every fold (D5): :class:`StoredFolds` feeds ``cross_validate`` the
recorded indices, and its per-fold hook is where progress is reported and a
cancel is honoured. Final fit refits the in-force structure on train and
validation rows together, puts the hand edits back, and is offered by
Export (D6).

Every refusal is a fixed sentence (``editor/errors.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.distributions import NegativeBinomial
from superglm.editor.carry import model_with_edited_curves, weighted_mean
from superglm.editor.errors import EditorClientError, EditorValueError
from superglm.editor.evaluation import EvaluationDataset, training_export_dataset
from superglm.editor.jobs import JobCancelledError
from superglm.editor.refit import EXPLICIT_PENALTY_ATTRIBUTE, fit_refit_model
from superglm.editor.terms import resolve_refit_method
from superglm.features.categorical import Categorical
from superglm.features.grouping import LevelGrouping, native_by_text
from superglm.features.rebuild import clone_with_replaced_features, rebuilt_categorical
from superglm.model.fit_state import configured_family, configured_lambda2, configured_penalty
from superglm.model_selection import (
    _BUILTIN_SCORERS,
    _POOLED_PARTS,
    CrossValidationResult,
    _data_fingerprint,
    _fold_row_count,
    cross_validate,
)
from superglm.plotting.comparison import _feature_beta, _score_levels
from superglm.plotting.curve_similarity import _summarize_against_fold_mean

if TYPE_CHECKING:
    from superglm.editor._types import EditableTerm

_LOGGER = logging.getLogger(__name__)

JOB_KINDS = ("cv", "final_fit")

NO_CV = (
    "No cross-validation result was supplied. Pass cv=cross_validate(..., "
    "return_estimators=True) to edit()."
)
NO_FOLDS = "This result has no fold indices, so its folds cannot be run again."
NO_ROWS = (
    "Run CV needs the rows the folds were drawn on. Pass cv_data=(X, y, sample_weight) to edit()."
)
ROWS_MISMATCH = "The CV data has {rows:,} rows, but the folds were drawn on {expected:,}."
TRAIN_ROWS_MISMATCH = (
    "The train data has {rows:,} rows, but the folds were drawn on {expected:,}. "
    "Pass cv_data=(X, y, sample_weight) to edit()."
)
FINGERPRINT_MISMATCH = (
    "The {data}'s {rows:,} rows are not the ones the folds were drawn on: their columns, "
    "dtypes, row order or values differ (a pandas frame and a polars one differ too). Pass "
    "the X, y, sample_weight and offset given to cross_validate as cv_data."
)
NO_FINGERPRINT = (
    "This result was made before cross_validate recorded a data fingerprint, so only "
    "the row count was checked."
)
NO_ESTIMATORS = (
    "Fold curves need the fold models: pass return_estimators=True to cross_validate, "
    "or run CV on the current model."
)
NO_FINAL_ROWS = "Final fit needs train_data, or a model that kept its fit data."
TRAIN_ONLY = "No validation data was supplied, so Final fit uses the train rows only."
SUPERSEDED = "The model changed while the job ran, so its result was not kept. Run it again."
MIXED_FRAMES = "Train and validation data must both be pandas or both be Polars data frames."
FOLD_FAILED = "Run CV stopped at fold {fold} and kept no result. {reason}"
FOLD_NOT_FITTED = "That fold could not be fitted or scored."
FINAL_SPLIT_MISSING = (
    "Final fit fits the train and validation rows together, so it needs {column} on both "
    "or on neither: the {have} data has them and the {lack} data does not. Pass them with "
    "the {lack} data to edit()."
)
UNCOVERED_LEVELS = (
    "{job}'s rows hold levels of {term!r} that its groups do not cover, {levels}; send "
    "New levels of {term!r} to a group to fit them there, or leave those rows out."
)
OUTSIDE_DECLARED_LEVELS = (
    "{job}'s rows hold levels of {term!r} that the model's levels= leaves out, {levels}; "
    "declare them in its levels=, or leave those rows out."
)
FINAL_NOT_RUN = "Run Final fit on all rows, on the Cross-validation tab, first."
FINAL_STALE = "The model changed after the final fit. Run Final fit on all rows again."

_METRICS = (
    ("deviance", "Mean deviance", True),
    ("gini", "Gini", False),
    ("nll", "Negative log-likelihood", True),
)
_DEFAULT_SCORING = tuple(name for name, _label, _lower in _METRICS)
_FOLD_COLUMNS = ("fold", "n_train", "n_test", "fit_time_s", "converged", "n_iter", "effective_df")


def waiting_sentence(count: int) -> str:
    """'1 change is waiting' or 'N changes are waiting'."""
    return "1 change is waiting" if count == 1 else f"{count} changes are waiting"


def _not_included(count: int) -> str:
    """The Final fit note on waiting changes, labelled: the tab shows it beside Run CV's reason."""
    if count == 1:
        return "Final fit: 1 waiting change is not included."
    return f"Final fit: {count} waiting changes are not included."


@dataclass(frozen=True)
class CVDataCheck:
    """The rows Run CV replays a supplied result's folds on, or why it cannot."""

    rows: EvaluationDataset | None
    reason: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class StoredFolds:
    """A splitter that yields recorded ``(train_idx, test_idx)`` pairs.

    ``cross_validate`` asks for the next fold only after scoring the last, so
    ``before_fold(index)`` runs between folds, before fold ``index`` is
    fitted. Raising there stops the run without fitting that fold.
    """

    folds: tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]
    before_fold: Callable[[int], None] | None = None

    def split(self, X=None, y=None, groups=None):
        del X, y, groups
        for index, (train, test) in enumerate(self.folds):
            if self.before_fold is not None:
                self.before_fold(index)
            yield train, test


@dataclass(frozen=True)
class CVRun:
    """A finished Run CV, kept on the widget with the revision it ran on."""

    result: CrossValidationResult
    terms: list[dict[str, Any]]
    model_revision: int
    carried: tuple[str, ...]


@dataclass(frozen=True)
class FinalFit:
    """A finished Final fit, kept on the widget with the revision it ran on."""

    model: Any
    model_revision: int
    n_rows: int
    splits: tuple[str, ...]
    carried: tuple[str, ...]
    pending: int


# ── The supplied result and its rows ─────────────────────────────


def check_cv_data(
    cv: CrossValidationResult | None,
    cv_rows: EvaluationDataset | None,
    fallback: EvaluationDataset | None,
) -> CVDataCheck:
    """Decide which rows Run CV replays ``cv``'s folds on.

    ``cv_rows`` is ``cv_data=`` when it was supplied; otherwise ``fallback``,
    the train data, is used when its row count matches the folds. A result
    that records a data fingerprint must match it. An older one gets the
    row-count check and a note saying so.
    """
    if cv is None:
        return CVDataCheck(None, NO_CV)
    folds = cv.fold_indices
    if not folds:
        return CVDataCheck(None, NO_FOLDS)
    rows = fallback if cv_rows is None else cv_rows
    if rows is None:
        return CVDataCheck(None, NO_ROWS)
    expected = _fold_row_count(cv.n_rows, folds)
    if rows.n_obs != expected:
        sentence = TRAIN_ROWS_MISMATCH if cv_rows is None else ROWS_MISMATCH
        return CVDataCheck(None, sentence.format(rows=rows.n_obs, expected=expected))
    if cv.data_fingerprint is None:
        return CVDataCheck(rows, note=NO_FINGERPRINT)
    try:
        held = _data_fingerprint(
            rows.X, rows.y, rows.sample_weight, rows.offset, cv.fingerprint_columns
        )
    except (KeyError, TypeError, ValueError):
        held = None
    if cv.data_fingerprint != held:
        data = "train data" if cv_rows is None else "CV data"
        return CVDataCheck(None, FINGERPRINT_MISMATCH.format(data=data, rows=rows.n_obs))
    return CVDataCheck(rows)


def run_cv_reason(session) -> str | None:
    """Why Run CV is disabled now, or None (D7: it waits for Refit)."""
    if session.cv_check.reason is not None:
        return session.cv_check.reason
    if session.pending:
        return f"Refit first: {waiting_sentence(len(session.pending))}."
    return None


def final_fit_datasets(session) -> tuple[EvaluationDataset, ...]:
    """Train and validation rows (D6); the test split stays held out."""
    train = training_export_dataset(session)
    if train is None:
        return ()
    validation = session._evaluation_data.get("validation")
    return (train,) if validation is None else (train, validation)


# Each column Final fit stacks: its attribute, its name in a refusal, and the
# value a split without it stands for.
_STACKED_COLUMNS = (("sample_weight", "sample weights", 1.0), ("offset", "offsets", 0.0))


def final_fit_reason(session) -> str | None:
    """Why Final fit cannot run now, as one fixed sentence; None when it can."""
    datasets = final_fit_datasets(session)
    if not datasets:
        return NO_FINAL_ROWS
    return _unstackable_reason(datasets)


def _unstackable_reason(datasets: Sequence[EvaluationDataset]) -> str | None:
    """Refuse a column one split lacks while another holds anything but its neutral value.

    A split without weights or an offset is stacked as weight 1 and offset 0.
    That is exact beside a split holding those values (a model that kept
    unweighted fit data keeps weights of 1), but beside log exposure it would
    fit the rows that lack it on exposure 1, silently.
    """
    for name, column, neutral in _STACKED_COLUMNS:
        lack = [dataset.name for dataset in datasets if getattr(dataset, name) is None]
        have = [
            dataset.name
            for dataset in datasets
            if getattr(dataset, name) is not None
            and np.any(np.asarray(getattr(dataset, name), dtype=np.float64) != neutral)
        ]
        if lack and have:
            return FINAL_SPLIT_MISSING.format(column=column, have=have[0], lack=lack[0])
    return None


# ── Relativities by fold ─────────────────────────────────────────


@dataclass(frozen=True)
class _TermGrid:
    """Where a term's fold curves are read: the editor's grid, levels in model order."""

    kind: str
    points: NDArray
    order: NDArray[np.intp]
    labels: list[str] | None


def _term_grid(term: EditableTerm) -> _TermGrid | None:
    if term.levels is not None:
        position = {label: index for index, label in enumerate(term.levels)}
        native = list(term.metadata.get("native_levels", term.levels))
        in_model_order = [level for level in native if str(level) in position]
        labels = [str(level) for level in in_model_order]
        return _TermGrid(
            kind="levels",
            points=np.asarray(in_model_order, dtype=object),
            order=np.asarray([position[label] for label in labels], dtype=np.intp),
            labels=labels,
        )
    if term.x is None or term.size < 2:
        # A linear term is one slope; its fold spread is a coefficient's.
        return None
    return _TermGrid(
        kind="continuous",
        points=np.asarray(term.x, dtype=np.float64),
        order=np.arange(term.size, dtype=np.intp),
        labels=None,
    )


def fold_log_curves(model, terms: Mapping[str, EditableTerm]) -> dict[str, NDArray[np.float64]]:
    """One fold model's log curve for each term, read on the editor's grid or levels.

    The editor's grid can reach past a fold's training range; a spline holds
    its end value there, as it does at predict time. A level the fold never
    saw is NaN, a gap in that fold's curve, whatever its unseen policy: one
    outside its level universe, or one ``cross_validate``'s shared universe
    gave it with no training rows, which it holds pinned and would predict at
    its pin. A fold that can score none of a term's points, a term of another
    kind, is left out of that term.
    """
    curves: dict[str, NDArray[np.float64]] = {}
    for name, term in terms.items():
        grid = _term_grid(term)
        if grid is None or name not in model._specs:
            continue
        spec = model._specs[name]
        try:
            beta = _feature_beta(model, name)
            if grid.kind == "levels":
                points = _levels_as_taken(spec, grid)
                values = _score_levels(spec, points, beta)
            else:
                values = np.asarray(spec.score(grid.points, beta), dtype=np.float64)
        except (KeyError, ValueError):
            continue
        if np.isnan(values).all():
            continue
        curves[name] = values
    return curves


def _levels_as_taken(spec, grid: _TermGrid) -> NDArray:
    """The grid's levels as ``spec`` takes them: its own value for each label it holds.

    A collapse leaves the in-force term with text labels, while a supplied
    fold model fitted on integer codes takes the integers and would read "1"
    as a level it never saw, a gap. A label the fold's levels lack, a grouped
    fold's member, is passed as it is.
    """
    own = native_by_text(getattr(spec, "_levels", ()))
    return np.asarray(
        [
            own.get(label, point)
            for label, point in zip(grid.labels or (), grid.points, strict=True)
        ],
        dtype=object,
    )


def fold_term_items(
    terms: Mapping[str, EditableTerm],
    fold_curves: Mapping[int, Mapping[str, NDArray[np.float64]]],
    *,
    held: Collection[str] = (),
) -> list[dict[str, Any]]:
    """One chart entry per term, least stable first, then the terms ``held``.

    Every curve is re-centred on its exposure-weighted mean log and shown as
    a relativity. A term's ``spread`` is the mean over folds of
    ``rmse_to_mean`` on that scale, and ``min_correlation`` the lowest
    ``correlation_to_mean`` (``plotting.curve_similarity``). A level a fold
    never saw is a gap (NaN) in that fold's curve, and its distances skip it.
    Each fold's curve keeps its fold number, so a fold missing from a term
    keeps its colour and its place in the tab's charts.

    A term ``held`` is a hand edit Run CV put back on every fold: the same
    curve on each, so its spread and min r measure nothing. It is marked
    ``held``, carries neither, and follows the measured terms.
    """
    items = []
    for name, term in terms.items():
        grid = _term_grid(term)
        curves = {
            index: by_term[name]
            for index, by_term in sorted(fold_curves.items())
            if name in by_term
        }
        if grid is not None and curves:
            items.append(_term_item(name, term, grid, curves, held=name in held))
    items.sort(key=_least_stable_first)
    return items


def _least_stable_first(item: dict[str, Any]) -> tuple[bool, float, str]:
    """Measured terms by falling spread, then the held terms; by name within each."""
    if item["held"]:
        return (True, 0.0, item["name"])
    return (False, -item["spread"], item["name"])


def _term_item(
    name: str, term: EditableTerm, grid: _TermGrid, curves, *, held: bool
) -> dict[str, Any]:
    weights = (
        np.ones(grid.order.size, dtype=np.float64)
        if term.weights is None
        else np.asarray(term.weights, dtype=np.float64)[grid.order]
    )
    if not float(np.sum(weights)) > 0.0:
        weights = np.ones(grid.order.size, dtype=np.float64)
    # Every curve is centred on the same points, the ones every fold has a
    # value at, so a fold's gap does not shift it against the others.
    shared = np.logical_and.reduce([~np.isnan(values) for values in curves.values()])

    def centred(log_values) -> NDArray[np.float64]:
        values = np.asarray(log_values, dtype=np.float64)
        points = shared if shared.any() else ~np.isnan(values)
        return np.exp(values - weighted_mean(values[points], weights[points]))

    folds = {f"Fold {index + 1}": centred(values) for index, values in curves.items()}
    vs_mean = None if held else _summarize_against_fold_mean(folds, weights)
    fit = term.original_log_effect[grid.order]
    edited = term.edited_log_effect[grid.order]
    changed = not np.allclose(edited, fit, rtol=0.0, atol=1e-14)
    return {
        "name": name,
        "kind": grid.kind,
        "x": None if grid.kind == "levels" else grid.points,
        "levels": grid.labels,
        "weights": weights,
        "folds": [
            {"fold": index, "label": label, "values": values}
            for index, (label, values) in zip(curves, folds.items(), strict=True)
        ],
        "fit": centred(fit),
        "edited": centred(edited) if changed else None,
        "held": held,
        "spread": None if vs_mean is None else float(vs_mean["rmse_to_mean"].mean()),
        "min_correlation": None if vs_mean is None else float(vs_mean["correlation_to_mean"].min()),
    }


# ── The tab's payload ────────────────────────────────────────────


@dataclass(frozen=True)
class CVTabView:
    """What the tab shows, captured under the widget lock."""

    supplied: CrossValidationResult | None
    check: CVDataCheck
    terms: dict[str, EditableTerm]
    run: CVRun | None
    final_fit: FinalFit | None
    model_revision: int
    model_changed: bool
    pending: int
    run_reason: str | None
    final_reason: str | None
    has_validation: bool


def capture_cv_view(session, *, run: CVRun | None, final_fit: FinalFit | None) -> CVTabView:
    """Copy what the tab reads; the caller holds the widget lock."""
    return CVTabView(
        supplied=session.cv,
        check=session.cv_check,
        terms={name: term.copy() for name, term in session.terms.items()},
        run=run,
        final_fit=final_fit,
        model_revision=session.model_revision,
        model_changed=session.model is not session.reference_model or bool(session.edited_terms()),
        pending=len(session.pending),
        run_reason=run_cv_reason(session),
        final_reason=final_fit_reason(session),
        has_validation="validation" in session._evaluation_data,
    )


def cv_report_payload(widget, *, request_sequence: int | None = None) -> dict[str, Any]:
    """The ``cv`` report: captured under the widget lock, built outside it."""
    with widget._lock:
        view = capture_cv_view(widget.session, run=widget._cv_run, final_fit=widget._final_fit)
    jobs = {kind: widget._jobs.latest(kind) for kind in JOB_KINDS}
    return cv_tab_payload(view, jobs=jobs, request_sequence=request_sequence)


def cv_tab_payload(
    view: CVTabView,
    *,
    jobs: Mapping[str, dict[str, Any] | None],
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """The tab: header, fold performance, the fold table's rows and relativities by fold."""
    results = []
    if view.supplied is not None:
        results.append(
            _result_payload(view.supplied, "As supplied", "supplied", None, view.model_changed)
        )
    run = view.run
    if run is not None:
        results.append(
            _result_payload(
                run.result,
                "Current model",
                "run",
                run.model_revision,
                run.model_revision != view.model_revision,
            )
        )
    final = view.final_fit
    final_notes = [
        _not_included(view.pending) if view.pending else None,
        None if view.has_validation else TRAIN_ONLY,
    ]
    return {
        "available": True,
        "report": "cv",
        "title": "Cross-validation",
        "note": "" if results else NO_CV,
        "model_revision": view.model_revision,
        "request_sequence": request_sequence,
        "header": _header(view),
        "pending": view.pending,
        "run_cv": {
            "available": view.run_reason is None,
            "reason": view.run_reason,
            "note": view.check.note,
        },
        "final_fit": {
            "available": view.final_reason is None,
            "reason": view.final_reason,
            "note": " ".join(note for note in final_notes if note) or None,
            "done": final is not None,
            "stale": final is not None and final.model_revision != view.model_revision,
            "n_rows": None if final is None else final.n_rows,
        },
        "metrics": [
            {"name": name, "label": label, "lower_is_better": lower}
            for name, label, lower in _METRICS
            if any(name in result["mean"] for result in results)
        ],
        "results": results,
        "relativities": _relativities(view),
        "jobs": dict(jobs),
    }


def _header(view: CVTabView) -> dict[str, Any]:
    supplied = view.supplied
    if supplied is None:
        return {"supplied": False, "n_folds": 0, "splitter": None, "n_rows": None}
    rows = None if view.check.rows is None else view.check.rows.n_obs
    return {
        "supplied": True,
        "n_folds": len(supplied.fold_indices or supplied.fold_scores),
        "splitter": supplied.splitter,
        "n_rows": rows if supplied.n_rows is None else supplied.n_rows,
    }


def _result_payload(
    result: CrossValidationResult,
    label: str,
    origin: str,
    model_revision: int | None,
    stale: bool,
) -> dict[str, Any]:
    names = [name for name, _label, _lower in _METRICS if name in result.fold_scores.columns]
    folds = [
        {
            **{column: record.get(column) for column in _FOLD_COLUMNS},
            "scores": {name: record.get(name) for name in names},
        }
        for record in result.fold_scores.to_dict("records")
    ]
    return {
        "label": label,
        "origin": origin,
        "model_revision": model_revision,
        "stale": bool(stale),
        "folds": folds,
        "mean": {name: result.mean_scores.get(name) for name in names},
        "std": {name: result.std_scores.get(name) for name in names},
        "pooled": {
            name: result.pooled_scores[name] for name in names if name in result.pooled_scores
        },
    }


def _relativities(view: CVTabView) -> dict[str, Any]:
    run = view.run
    if run is not None and run.terms:
        return {
            "available": True,
            "origin": "run",
            "stale": run.model_revision != view.model_revision,
            "note": None,
            "terms": run.terms,
        }
    estimators = [] if view.supplied is None else list(view.supplied.estimators or ())
    curves = {
        index: fold_log_curves(model, view.terms)
        for index, model in enumerate(estimators)
        if model is not None
    }
    if not curves:
        return {
            "available": False,
            "origin": None,
            "stale": False,
            "note": NO_ESTIMATORS,
            "terms": [],
        }
    return {
        "available": True,
        "origin": "supplied",
        "stale": view.model_changed,
        "note": None,
        "terms": fold_term_items(view.terms, curves),
    }


# ── Run CV ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Plan:
    """A job's inputs, captured under the widget lock with the model they were taken from.

    ``template`` is the in-force structure, unfitted, that the job fits.
    """

    model: Any
    template: Any
    model_revision: int

    def is_current(self, session) -> bool:
        return session.model_revision == self.model_revision and session.model is self.model


@dataclass(frozen=True)
class CVRunPlan(_Plan):
    """Everything Run CV reads."""

    rows: EvaluationDataset
    folds: tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]
    terms: dict[str, EditableTerm]
    edited: dict[str, EditableTerm]
    fit_mode: str
    scoring: tuple[str, ...]
    splitter: str | None
    n_points: int


def capture_cv_run(session) -> CVRunPlan:
    """Capture Run CV's inputs; refuse with the tab's reason while it is disabled."""
    reason = run_cv_reason(session)
    if reason is not None:
        raise EditorValueError(reason)
    cv = session.cv
    terms = {name: term.copy() for name, term in session.terms.items()}
    supplied = tuple(name for name in _DEFAULT_SCORING if name in cv.fold_scores.columns)
    return CVRunPlan(
        model=session.model,
        template=_declared_template(session),
        model_revision=session.model_revision,
        rows=session.cv_check.rows,
        folds=tuple(
            (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
            for train, test in cv.fold_indices
        ),
        terms=terms,
        edited={name: terms[name] for name in session.edited_terms()},
        fit_mode=resolve_refit_method(session.model, "auto"),
        scoring=supplied or _DEFAULT_SCORING,
        splitter=cv.splitter,
        n_points=session.n_points,
    )


def _declared_template(session):
    """The in-force structure, unfitted, to choose again what the opened model estimates.

    A Refit's model declares the selection penalty, smoothing and NB2 theta
    its own fit chose on all the training rows. Each fold and the Final fit
    choose them again on their own rows, as the opened model declares: the
    penalties as :meth:`superglm.structure.Structure.apply` does, and a
    ``theta="auto"`` too. Validation rows never set a fold's penalty or
    theta. A theta the opened model fixes stays as the in-force model holds
    it, re-profiled or not. A ``lambda1`` or ``lambda2`` passed to a Refit
    is the analyst's choice and stays.
    """
    opened = session.reference_model
    template = session.model.clone_unfitted()
    explicit = getattr(session.model, EXPLICIT_PENALTY_ATTRIBUTE, {})
    template.selection_penalty = explicit.get("lambda1", configured_penalty(opened).lambda1)
    template.lambda2 = explicit.get("lambda2", configured_lambda2(opened))
    family = configured_family(opened)
    if isinstance(family, NegativeBinomial) and family.theta == "auto":
        template.family = family
    return template


def _covering_template(template, X, job: str):
    """``template`` with each grouped categorical covering the levels ``X`` holds.

    A grouping built on the train rows does not cover a level only the CV
    data or the validation rows hold, and the fit refuses a level its
    grouping does not cover. Such a level goes where the term sends new
    levels: into the group its ``unseen`` names, as the in-force model
    predicts it and as ``Structure.apply(model, X=...)`` places it. A term
    with no group for new levels (``"error"`` or ``"base"``), or one whose
    ``levels=`` leaves the level out, refuses ``job`` in one sentence.
    """
    frame = as_eager_frame(X)
    replacements = {}
    for name, spec in template._specs.items():
        grouping = getattr(spec, "_grouping", None)
        if not isinstance(spec, Categorical) or grouping is None or name not in frame.columns:
            continue
        new = _uncovered_labels(frame.column_array(name), grouping)
        if not new:
            continue
        if spec._declared_levels is not None:
            raise EditorValueError(OUTSIDE_DECLARED_LEVELS.format(job=job, term=name, levels=new))
        if spec.unseen in ("error", "base"):
            raise EditorValueError(UNCOVERED_LEVELS.format(job=job, term=name, levels=new))
        widened = LevelGrouping(
            original_to_group={**grouping.original_to_group, **dict.fromkeys(new, spec.unseen)},
            group_to_originals={
                label: [*members, *(new if label == spec.unseen else ())]
                for label, members in grouping.group_to_originals.items()
            },
            all_original_levels=[*grouping.all_original_levels, *new],
            grouped_levels=list(grouping.grouped_levels),
        )
        replacements[name] = rebuilt_categorical(
            spec, spec, base=spec.base, grouping=widened, data=np.asarray(new, dtype=object)
        )
    return clone_with_replaced_features(template, replacements) if replacements else template


def _uncovered_labels(values, grouping) -> list[str]:
    """The labels in ``values`` that ``grouping`` does not map, as the text it matches by.

    Missing values are left to the fit, which refuses them.
    """
    values = np.asarray(values).ravel()
    present = values[~np.asarray(pd.isna(values), dtype=bool)]
    labels = pd.Series(present).astype(str).unique()
    return sorted(set(labels.tolist()) - set(grouping.original_to_group), key=str)


def run_cv(plan: CVRunPlan, context) -> CVRun:
    """Replay the stored folds on the in-force structure with the hand edits put back."""
    template = _covering_template(plan.template, plan.rows.X, "Run CV")
    recorder = _FoldRecorder(plan, context)
    try:
        result = cross_validate(
            template,
            plan.rows.X,
            plan.rows.y,
            cv=StoredFolds(plan.folds, before_fold=recorder.before_fold),
            sample_weight=plan.rows.sample_weight,
            offset=plan.rows.offset,
            fit_mode=plan.fit_mode,
            scoring=recorder.score,
            error_score="raise",
        )
    except JobCancelledError:
        raise
    except Exception as exc:
        # A fold that fails stops the run: the folds that did score are not
        # averaged as if they were all of them.
        if isinstance(exc, EditorClientError):
            reason = exc.public_message
        else:
            _LOGGER.warning("Run CV fold %d failed.", recorder.fold_number, exc_info=True)
            reason = FOLD_NOT_FITTED
        raise EditorValueError(
            FOLD_FAILED.format(fold=recorder.fold_number, reason=reason)
        ) from exc
    context.check()
    context.progress("curves")
    return CVRun(
        result=replace(result, pooled_scores=recorder.pooled_scores(), splitter=plan.splitter),
        terms=fold_term_items(plan.terms, recorder.curves, held=plan.edited),
        model_revision=plan.model_revision,
        carried=tuple(sorted(plan.edited)),
    )


class _FoldRecorder:
    """Run CV's per-fold hooks: progress and cancel between folds, edits before scoring.

    ``cross_validate`` fits each fold and hands the fitted model to
    :meth:`score`, which puts the hand edits back (D5) and computes the
    built-in scores on that edited model. A callable scorer's dict is not
    pooled by ``cross_validate``, so the pooled deviance and NLL are summed
    here from the same numerator and denominator parts it pools.
    """

    def __init__(self, plan: CVRunPlan, context) -> None:
        self._plan = plan
        self._context = context
        self._fold = -1
        self._frame = as_eager_frame(plan.rows.X)
        self._y = np.asarray(plan.rows.y, dtype=np.float64)
        self._totals = {name: [0.0, 0.0] for name in plan.scoring if name in _POOLED_PARTS}
        self.curves: dict[int, dict[str, NDArray[np.float64]]] = {}

    @property
    def fold_number(self) -> int:
        """The fold being fitted or scored, counted from 1."""
        return self._fold + 1

    def before_fold(self, index: int) -> None:
        self._context.progress("fold", fold=index + 1, n_folds=len(self._plan.folds))
        self._context.check()
        self._fold = index

    def score(self, model, X, y, *, sample_weight=None, offset=None) -> dict[str, float]:
        if self._plan.edited:
            train = self._plan.folds[self._fold][0]
            model = model_with_edited_curves(
                model,
                self._plan.edited,
                self._frame.take_rows(train),
                self._y[train],
                _take(self._plan.rows.sample_weight, train),
                _take(self._plan.rows.offset, train),
                n_points=self._plan.n_points,
            )
        scores: dict[str, float] = {}
        parts: dict[str, tuple[float, float]] = {}
        for name in self._plan.scoring:
            pooled = _POOLED_PARTS.get(name)
            if pooled is None:
                scorer = _BUILTIN_SCORERS[name]
                scores[name] = float(
                    scorer(model, X, y, sample_weight=sample_weight, offset=offset)
                )
                continue
            numerator, denominator = pooled(model, X, y, sample_weight=sample_weight, offset=offset)
            parts[name] = (numerator, denominator)
            scores[name] = numerator / denominator
        # Only a fold that scored completely joins the pooled totals.
        for name, (numerator, denominator) in parts.items():
            self._totals[name][0] += numerator
            self._totals[name][1] += denominator
        self.curves[self._fold] = fold_log_curves(model, self._plan.terms)
        return scores

    def pooled_scores(self) -> dict[str, float]:
        return {
            name: numerator / denominator
            for name, (numerator, denominator) in self._totals.items()
            if denominator > 0.0
        }


def _take(values, rows: NDArray[np.intp]):
    return None if values is None else np.asarray(values, dtype=np.float64)[rows]


# ── Final fit ────────────────────────────────────────────────────


@dataclass(frozen=True)
class FinalFitPlan(_Plan):
    """Everything Final fit reads."""

    datasets: tuple[EvaluationDataset, ...]
    edited: dict[str, EditableTerm]
    pending: int
    n_points: int


def capture_final_fit(session) -> FinalFitPlan:
    """Capture Final fit's inputs; refuse without training rows, or rows that cannot stack."""
    datasets = final_fit_datasets(session)
    reason = _unstackable_reason(datasets) if datasets else NO_FINAL_ROWS
    if reason is not None:
        raise EditorValueError(reason)
    return FinalFitPlan(
        model=session.model,
        template=_declared_template(session),
        model_revision=session.model_revision,
        datasets=datasets,
        edited={name: session.terms[name].copy() for name in session.edited_terms()},
        pending=len(session.pending),
        n_points=session.n_points,
    )


def run_final_fit(plan: FinalFitPlan, context) -> FinalFit:
    """Refit the in-force structure on train and validation rows, then put the edits back."""
    X, y, sample_weight, offset = _union_rows(plan.datasets)
    context.progress("fitting", n_rows=int(y.size))
    context.check()
    model = _covering_template(plan.template, X, "Final fit").clone_unfitted()
    fit_refit_model(
        plan.model,
        model,
        method="auto",
        X=X,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
    )
    context.check()
    if plan.edited:
        context.progress("carrying", terms=sorted(plan.edited))
        model = model_with_edited_curves(
            model, plan.edited, X, y, sample_weight, offset, n_points=plan.n_points
        )
        context.check()
    return FinalFit(
        model=model,
        model_revision=plan.model_revision,
        n_rows=int(y.size),
        splits=tuple(dataset.name for dataset in plan.datasets),
        carried=tuple(sorted(plan.edited)),
        pending=plan.pending,
    )


def _union_rows(datasets: Sequence[EvaluationDataset]):
    """The splits' rows stacked in order: one frame, response, weight and offset."""
    frames = [as_eager_frame(dataset.X) for dataset in datasets]
    backends = {frame.backend for frame in frames}
    if len(backends) > 1:
        raise EditorValueError(MIXED_FRAMES)
    if len(frames) == 1:
        X = frames[0].native
    elif backends == {"pandas"}:
        X = pd.concat([cast(pd.DataFrame, frame.native) for frame in frames], ignore_index=True)
    else:
        import polars as pl

        X = pl.concat(
            [cast(pl.DataFrame, frame.native) for frame in frames], how="vertical_relaxed"
        )
    y = np.concatenate([np.asarray(dataset.y, dtype=np.float64) for dataset in datasets])
    return X, y, _stacked(datasets, "sample_weight", 1.0), _stacked(datasets, "offset", 0.0)


def _stacked(datasets: Sequence[EvaluationDataset], name: str, fill: float):
    """One column across the splits, ``fill`` where a split lacks it; None if all do.

    :func:`capture_final_fit` has refused a fill beside values that differ from it.
    """
    columns = [getattr(dataset, name) for dataset in datasets]
    if all(column is None for column in columns):
        return None
    return np.concatenate(
        [
            np.full(dataset.n_obs, fill) if column is None else np.asarray(column, dtype=float)
            for dataset, column in zip(datasets, columns, strict=True)
        ]
    )
