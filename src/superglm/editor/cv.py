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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from superglm.editor.evaluation import EvaluationDataset, training_export_dataset
from superglm.model_selection import CrossValidationResult, _data_fingerprint
from superglm.plotting.comparison import _feature_beta, _score_levels
from superglm.plotting.curve_similarity import _summarize_against_fold_mean

if TYPE_CHECKING:
    from superglm.editor._types import EditableTerm

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
    "The CV data's response or weights differ from the data the folds were drawn on. "
    "Pass the same rows, in the same order, as cv_data."
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

_METRICS = (
    ("deviance", "Mean deviance", True),
    ("gini", "Gini", False),
    ("nll", "Negative log-likelihood", True),
)
_FOLD_COLUMNS = ("fold", "n_train", "n_test", "fit_time_s", "converged", "n_iter", "effective_df")


def waiting_sentence(count: int) -> str:
    """'1 change is waiting' or 'N changes are waiting'."""
    return "1 change is waiting" if count == 1 else f"{count} changes are waiting"


def _not_included(count: int) -> str:
    if count == 1:
        return "1 waiting change is not included."
    return f"{count} waiting changes are not included."


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

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        del X, y, groups
        return len(self.folds)


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
    if not cv.fold_indices:
        return CVDataCheck(None, NO_FOLDS)
    rows = fallback if cv_rows is None else cv_rows
    if rows is None:
        return CVDataCheck(None, NO_ROWS)
    expected = _expected_rows(cv)
    if rows.n_obs != expected:
        sentence = TRAIN_ROWS_MISMATCH if cv_rows is None else ROWS_MISMATCH
        return CVDataCheck(None, sentence.format(rows=rows.n_obs, expected=expected))
    if cv.data_fingerprint is None:
        return CVDataCheck(rows, note=NO_FINGERPRINT)
    if cv.data_fingerprint != _data_fingerprint(rows.y, rows.sample_weight):
        return CVDataCheck(None, FINGERPRINT_MISMATCH)
    return CVDataCheck(rows)


def _expected_rows(cv: CrossValidationResult) -> int:
    """The recorded row count, or one past the largest index an older result holds."""
    if cv.n_rows is not None:
        return int(cv.n_rows)
    return 1 + max(int(np.max(np.concatenate(fold))) for fold in cv.fold_indices)


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
    saw is NaN, a gap in that fold's curve: one outside its level universe,
    or one ``cross_validate``'s shared universe gave it with no training rows,
    which it holds pinned and would predict at its pin. A fold that can score
    none of a term's points, a term of another kind, is left out of that term.
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
                values = _score_levels(spec, grid.points, beta)
                values[_pinned_points(spec, grid.points)] = np.nan
            else:
                values = np.asarray(spec.score(grid.points, beta), dtype=np.float64)
        except (KeyError, ValueError):
            continue
        if np.isnan(values).all():
            continue
        curves[name] = values
    return curves


def _pinned_points(spec, points: NDArray) -> NDArray[np.bool_]:
    """Which level points ``spec`` holds pinned: levels, or specials, with no training rows.

    Labels compare as text, the editor's level namespace. A grouped
    categorical pins a group, which every member of it reads.
    """
    pinned = {
        str(level)
        for level in (*getattr(spec, "_pinned_levels", ()), *getattr(spec, "_pinned_specials", ()))
    }
    if not pinned:
        return np.zeros(len(points), dtype=bool)
    grouping = getattr(spec, "_grouping", None)
    group_of = (
        {}
        if grouping is None
        else {str(level): str(group) for level, group in grouping.original_to_group.items()}
    )
    labels = [str(point) for point in points]
    return np.array(
        [label in pinned or group_of.get(label) in pinned for label in labels], dtype=bool
    )


def fold_term_items(
    terms: Mapping[str, EditableTerm],
    fold_curves: Mapping[int, Mapping[str, NDArray[np.float64]]],
) -> list[dict[str, Any]]:
    """One chart entry per term, least stable first.

    Every curve is re-centred on its exposure-weighted mean log and shown as
    a relativity. A term's ``spread`` is the mean over folds of
    ``rmse_to_mean`` on that scale, and ``min_correlation`` the lowest
    ``correlation_to_mean`` (``plotting.curve_similarity``). A level a fold
    never saw is a gap (NaN) in that fold's curve, and its distances skip it.
    """
    items = []
    for name, term in terms.items():
        grid = _term_grid(term)
        curves = {
            f"Fold {index + 1}": by_term[name]
            for index, by_term in sorted(fold_curves.items())
            if name in by_term
        }
        if grid is not None and curves:
            items.append(_term_item(name, term, grid, curves))
    items.sort(key=lambda item: (-item["spread"], item["name"]))
    return items


def _term_item(name: str, term: EditableTerm, grid: _TermGrid, curves) -> dict[str, Any]:
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
        point_weights = weights[points]
        if not float(np.sum(point_weights)) > 0.0:
            point_weights = np.ones(point_weights.size, dtype=np.float64)
        return np.exp(values - np.average(values[points], weights=point_weights))

    folds = {label: centred(values) for label, values in curves.items()}
    vs_mean = _summarize_against_fold_mean(folds, weights)
    fit = term.original_log_effect[grid.order]
    edited = term.edited_log_effect[grid.order]
    changed = not np.allclose(edited, fit, rtol=0.0, atol=1e-14)
    return {
        "name": name,
        "kind": grid.kind,
        "x": None if grid.kind == "levels" else grid.points,
        "levels": grid.labels,
        "weights": weights,
        "folds": [{"label": label, "values": values} for label, values in folds.items()],
        "fit": centred(fit),
        "edited": centred(edited) if changed else None,
        "spread": float(vs_mean["rmse_to_mean"].mean()),
        "min_correlation": float(vs_mean["correlation_to_mean"].min()),
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
        final_reason=None if final_fit_datasets(session) else NO_FINAL_ROWS,
        has_validation="validation" in session._evaluation_data,
    )


def cv_report_payload(widget, *, request_sequence: int | None = None) -> dict[str, Any]:
    """The ``cv`` report: captured under the widget lock, built outside it."""
    with widget._lock:
        view = capture_cv_view(widget.session, run=widget._cv_run, final_fit=widget._final_fit)
    jobs = {kind: None for kind in JOB_KINDS}
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
