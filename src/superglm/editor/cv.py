"""The Cross-validation tab: a supplied result, Run CV and Final fit.

``edit(model, cv=result)`` hands the editor a :class:`CrossValidationResult`.
The tab shows its fold scores and, when it kept its fold models, each
term's relativities by fold, least stable first. Run CV replays the
result's own folds on the in-force structure with the hand edits put back
on every fold: :class:`StoredFolds` feeds ``cross_validate`` the
recorded indices, and its per-fold hook is where progress is reported and a
cancel is honoured. Final fit refits the in-force structure on train and
validation rows together, puts the hand edits back, and is offered by
Export.

Every refusal is a fixed sentence (``editor/errors.py``).
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import threading
import time
import weakref
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.diagnostics.separation import SeparationError
from superglm.distributional.api import SuperLSS
from superglm.distributional.cross_validation import parts_ratio
from superglm.distributional.rebuild import bind_replay_offsets
from superglm.distributions import NegativeBinomial
from superglm.editor.carry import model_with_edited_curves, weighted_mean
from superglm.editor.diagnose import (
    all_rows_worst_rows,
    diagnose_tab_payload,
    fit_trace,
    supplied_fold_traces,
    worst_rows,
)
from superglm.editor.errors import EditorClientError, EditorValueError
from superglm.editor.evaluation import EvaluationDataset, training_export_dataset
from superglm.editor.jobs import JobCancelledError
from superglm.editor.lss.availability import session_refit_reason
from superglm.editor.lss.display import FoldTermDisplay, fold_chart_display
from superglm.editor.lss.validation import (
    LSS_METRIC_LABELS,
    lss_dataset_metrics,
    lss_split_reason,
    offset_departs,
    stack_offsets,
)
from superglm.editor.lss.view import LSSModelView, column_of, is_lss
from superglm.editor.refit import EXPLICIT_PENALTY_ATTRIBUTE, fit_refit_model
from superglm.editor.terms import resolve_refit_method
from superglm.features.categorical import Categorical, _codes_against, _grouping_labels
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.grouping import LevelGrouping, native_by_text
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.random_effect import RandomEffect
from superglm.features.rebuild import (
    base_names_level,
    clone_with_replaced_features,
    rebuilt_categorical,
)
from superglm.model.fit_state import configured_family, configured_lambda2, configured_penalty
from superglm.model_selection import (
    FINGERPRINT_VERSION,
    CrossValidationResult,
    _data_fingerprint,
    _fingerprint_columns,
    _fold_row_count,
    _take_offset,
    cross_validate,
    fold_adapter,
)
from superglm.plotting.comparison import _feature_beta, _score_levels
from superglm.plotting.curve_similarity import _summarize_against_fold_mean
from superglm.reml.observed_geometry import (
    ObservedGeometryInfeasibleError,
    ObservedModeNotCertifiedError,
)

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
FINGERPRINT_OTHER_VERSION = (
    "This result's data fingerprint predates this version of superglm or comes from another "
    "one, so its rows cannot be checked; run cross_validate again with this version."
)
NO_FINGERPRINT = (
    "This result was made before cross_validate recorded a data fingerprint, so only the row "
    "count was checked."
)
UNFINGERPRINTED = (
    "Run CV cannot check that these rows are the ones this result's folds were drawn on: the "
    "columns its model reads could not be fingerprinted (objects whose text is not their value, "
    "or values that cannot be hashed). Convert those values to text or numbers and run "
    "cross_validate again."
)
NO_FIT_MODE = (
    "This result was made before cross_validate recorded its fit method, so Run CV fits "
    "each fold with {method}, as the current model was fitted."
)
OTHER_SCORERS = (
    "This result's {names} scores come from scorers of its own, not the built-in ones of those "
    "names, so Run CV's scores would not compare with them; run cross_validate with the "
    'built-in scorers named as text, such as scoring=("deviance", "gini").'
)
NO_SCORERS = (
    "This result was made before cross_validate recorded which scorers were built in, so Run "
    "CV takes its {names} columns for the built-in scorers of those names."
)
NO_ESTIMATORS = (
    "Fold curves need the fold models: pass return_estimators=True to cross_validate, "
    "or run CV on the current model."
)
NO_FINAL_ROWS = "Final fit needs train_data, or a model that kept its fit data."
TRAIN_ONLY = "No validation data was supplied, so Final fit uses the train rows only."
SUPERSEDED = "The model changed while the job ran, so its result was not kept. Run it again."
MIXED_FRAMES = "Train and validation data must both be pandas or both be Polars data frames."
FINAL_COLUMN_TYPES = (
    "Final fit stacks the train and validation rows, so a column the model reads must have one "
    "dtype in both; {column!r} does not, so pass it with the same dtype in both splits."
)
FINAL_COLUMN_MISSING = (
    "Final fit stacks the train and validation rows, so a column the model reads must be in "
    "both; {column!r} is missing from one, so pass it in both splits."
)
FOLD_FAILED = "Run CV stopped at fold {fold} and kept no result. {reason}"
FOLD_NOT_FITTED = "That fold could not be fitted or scored."
FOLD_SEPARATED = (
    "That fold's training rows hold a level or crossed cell with exposure but only boundary "
    "responses, such as no claims, so its effect is infinite and the model refuses it "
    "(separation='error'); group that level with a neighbour."
)
FOLD_SOLVER_FAILED = (
    "The solver failed numerically on that fold's training rows; the editor's log records "
    "where it failed."
)
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
    "{job}'s rows hold levels of {term!r} outside its list of levels, {levels}; open a model "
    "whose list includes them (levels=, bind_levels or a categorical dtype sets it), or leave "
    "those rows out."
)
OUTSIDE_ORDERED_LEVELS = (
    "{job}'s rows hold levels of {term!r} outside its ordered levels, {levels}; open a model "
    "whose order=, values= or specials= includes them, or leave those rows out."
)
OUTSIDE_ORDERED_GROUPS = (
    "{job}'s rows hold levels of {term!r} outside its ordered levels or its groups, {levels}; "
    "open a model whose order=, values= or specials= includes them and whose groups cover "
    "them, or leave those rows out."
)
MISSING_LEVELS = (
    "{job}'s rows hold missing values of {term!r}; code them as a level of their own, or "
    "leave those rows out."
)
FINAL_NOT_FITTED = (
    "Final fit could not fit the {rows} rows. Check them for values the model cannot fit, "
    "such as missing or non-finite values, a level a term does not declare, or a response "
    "its family does not allow."
)
FINAL_SEPARATED = (
    "Final fit could not fit the {rows} rows: a level or crossed cell there has exposure but "
    "only boundary responses, such as no claims, so its effect is infinite and the model "
    "refuses it (separation='error'); group that level with a neighbour, or leave those rows out."
)
FINAL_SOLVER_FAILED = (
    "The solver failed numerically while Final fit fitted the {rows} rows, so no model was "
    "kept; the editor's log records where it failed."
)
FINAL_NOT_RUN = "Run Final fit on all rows, on the Cross-validation tab, first."
FINAL_STALE = "The model changed after the final fit. Run Final fit on all rows again."

# name, label, column heading, and whether lower is better (None: closer to 1 is).
_METRICS = (
    ("deviance", "Mean deviance", "Deviance", True),
    ("gini", "Gini", "Gini", False),
    ("nll", "Negative log-likelihood", "NLL", True),
)
# The labels the Validation tab gives the same scores.
_LSS_METRICS = (
    ("nll", LSS_METRIC_LABELS["log_score"], LSS_METRIC_LABELS["log_score"], True),
    ("crps", LSS_METRIC_LABELS["crps"], LSS_METRIC_LABELS["crps"], True),
    ("gini", "Gini", "Gini", False),
    ("actual_expected", LSS_METRIC_LABELS["actual_expected"], "A/E", None),
)
_FOLD_COLUMNS = (
    "fold",
    "n_train",
    "n_test",
    "fit_time_s",
    "converged",
    "n_iter",
    "effective_df",
    "n_reml_iter",
)


def _metrics_of(model) -> tuple[tuple[str, str, str, bool | None], ...]:
    """The scores the tab shows and Run CV computes for ``model``."""
    return _LSS_METRICS if is_lss(model) else _METRICS


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


def _edited_values(term) -> tuple[tuple[Any, float], ...]:
    """A term's edited values by level or point, in a fixed order: putting the levels in another
    order on screen changes no value, so it changes nothing a result describes."""
    values = np.asarray(term.edited_log_effect, dtype=np.float64).tolist()
    keys = (
        term.levels
        if term.levels is not None
        else range(len(values))
        if term.x is None
        else np.asarray(term.x, dtype=np.float64).tolist()
    )
    return tuple(sorted(zip(keys, values, strict=True), key=lambda pair: str(pair[0])))


@dataclass(frozen=True)
class InForce:
    """What a result was computed for: the fitted model in force and the hand edits on it.

    Undo brings back the same model object and the same edited values, so a result made before an
    edit describes the model again once the edit is undone; the revision number only ever rises.
    """

    model: Any
    edits: tuple[tuple[str, tuple[tuple[Any, float], ...]], ...]

    @classmethod
    def of(cls, session) -> InForce:
        edits = tuple(
            (name, _edited_values(session.terms[name])) for name in sorted(session.edited_terms())
        )
        return cls(session.model, edits)

    def describes(self, session) -> bool:
        """Whether ``session`` holds the model this was computed for; call it under the widget lock."""
        return _same_fit(self.model, session.model) and self.edits == InForce.of(session).edits

    @cached_property
    def token(self) -> str:
        """This as text the page compares: equal exactly when ``describes`` would hold.

        It names the fit by a serial number given once per fitted model, never reused, and the
        edits by a digest of their values.
        """
        digest = hashlib.blake2b(repr(self.edits).encode(), digest_size=12).hexdigest()
        return f"{_fit_serial(self.model)}-{digest}"


_FIT_SERIALS: dict[int, tuple[weakref.ref, int]] = {}
# Re-entrant: ``forget`` runs wherever its fit dies, which can be a collection inside the
# locked block below, on the thread that holds the lock.
_FIT_SERIAL_LOCK = threading.RLock()
_NEXT_FIT_SERIAL = itertools.count(1)


def _fit_serial(model) -> int:
    """A number for the fit ``model`` holds, the same while that fit lives (see ``_same_fit``)."""
    fit = model.lss if is_lss(model) else model
    key = id(fit)
    with _FIT_SERIAL_LOCK:
        found = _FIT_SERIALS.get(key)
        if found is not None and found[0]() is fit:
            return found[1]
        serial = next(_NEXT_FIT_SERIAL)

        def forget(ref, key=key) -> None:
            with _FIT_SERIAL_LOCK:
                if _FIT_SERIALS.get(key, (None,))[0] is ref:
                    del _FIT_SERIALS[key]

        _FIT_SERIALS[key] = (weakref.ref(fit, forget), serial)
        return serial


@dataclass(frozen=True)
class CVRun:
    """A finished Run CV, kept on the widget with the model it ran on."""

    result: CrossValidationResult
    terms: list[dict[str, Any]]
    model_revision: int
    carried: tuple[str, ...]
    source: InForce
    fold_fits: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FinalFit:
    """A finished Final fit, kept on the widget with the model it ran on."""

    model: Any
    model_revision: int
    source: InForce
    n_rows: int
    splits: tuple[str, ...]
    carried: tuple[str, ...]
    pending: int
    # A SuperLSS has no coefficient summary: its scores on the held-out test split, if it has one.
    test_scores: dict[str, float] | None = None


# ── The supplied result and its rows ─────────────────────────────


def check_cv_data(
    cv: CrossValidationResult | None,
    cv_rows: EvaluationDataset | None,
    fallback: EvaluationDataset | None,
) -> CVDataCheck:
    """Decide which rows Run CV replays ``cv``'s folds on.

    ``cv_rows`` is ``cv_data=`` when it was supplied; otherwise ``fallback``,
    the train data, is used when its row count matches the folds. A result
    that records a data fingerprint must match it. An older result, made
    before there were fingerprints, gets the row-count check and a note
    saying so; one made under the recipe whose read columns could not be
    fingerprinted is refused, since nothing can check its rows.
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
        if cv.fingerprint_version is not None:
            return CVDataCheck(None, UNFINGERPRINTED)
        return CVDataCheck(rows, note=NO_FINGERPRINT)
    if cv.fingerprint_version != FINGERPRINT_VERSION:
        return CVDataCheck(None, FINGERPRINT_OTHER_VERSION)
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
    """Why Run CV is disabled now, or None (it waits for Refit)."""
    if session.cv_check.reason is not None:
        return session.cv_check.reason
    builtin = session.cv.builtin_scores
    if builtin is not None:
        others = [
            name for name in _replayed_scores(session.cv, session.model) if name not in builtin
        ]
        if others:
            return OTHER_SCORERS.format(names=others)
    if session.pending:
        return f"Refit first: {waiting_sentence(len(session.pending))}."
    return None


def final_fit_datasets(session) -> tuple[EvaluationDataset, ...]:
    """Train and validation rows; the test split stays held out."""
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
    return (
        session_refit_reason(session)
        or _unstackable_reason(datasets)
        or _split_reason(session, datasets)
    )


def _split_reason(session, datasets: Sequence[EvaluationDataset]) -> str | None:
    """The first split a SuperLSS cannot fit with its weights or responses; None for a SuperGLM.

    A split the model cannot score with its weights or responses cannot be fitted
    with them either. The answer is kept on the view with the splits it was read for.
    """
    view = session.model
    if not isinstance(view, LSSModelView):
        return None
    cached = view._final_split_reason
    if (
        cached is None
        or len(cached[0]) != len(datasets)
        or any(held is not dataset for held, dataset in zip(cached[0], datasets))
    ):
        reasons = (lss_split_reason(view, dataset) for dataset in datasets)
        view._final_split_reason = (tuple(datasets), next(filter(None, reasons), None))
    return view._final_split_reason[1]


def _unstackable_reason(datasets: Sequence[EvaluationDataset]) -> str | None:
    """Refuse a column one split lacks while another holds anything but its neutral value.

    A split without weights or an offset is stacked as weight 1 and offset 0.
    That is exact beside a split holding those values (a model that kept
    unweighted fit data keeps weights of 1), but beside log exposure it would
    fit the rows that lack it on exposure 1, silently.
    """
    for column, values, neutral in _stacked_parts(datasets):
        lack = [dataset.name for dataset, value in zip(datasets, values) if value is None]
        if not lack:
            continue
        have = [
            dataset.name
            for dataset, value in zip(datasets, values)
            if value is not None and offset_departs(value, neutral)
        ]
        if lack and have:
            return FINAL_SPLIT_MISSING.format(column=column, have=have[0], lack=lack[0])
    return None


def _stacked_parts(datasets: Sequence[EvaluationDataset]):
    """Each stacked column as (its name, one value per split, its neutral value).

    A SuperLSS's offsets are one column per parameter: a split whose mapping
    lacks a parameter lacks that parameter's offsets.
    """
    for name, column, neutral in _STACKED_COLUMNS:
        values = [getattr(dataset, name) for dataset in datasets]
        keys = dict.fromkeys(key for value in values if isinstance(value, Mapping) for key in value)
        if not keys:
            yield column, values, neutral
        for key in keys:
            parts = []
            for dataset, value in zip(datasets, values, strict=True):
                part = value.get(key) if isinstance(value, Mapping) else None
                if part is not None and name == "offset" and dataset.sample_weight is not None:
                    # Missing-source compatibility concerns the rows a fit
                    # keeps; dropped raw offsets need no numeric conversion.
                    part = np.asarray(part)[np.asarray(dataset.sample_weight) > 0.0]
                parts.append(part)
            yield f"{key} {column}", parts, neutral


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
        # A linear term is one slope; its fold variation is a coefficient's.
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
    # A SuperLSS fold's terms are read through the view; no sweep needs its rows here.
    reader = LSSModelView(model, None) if isinstance(model, SuperLSS) else model
    for name, term in terms.items():
        grid = _term_grid(term)
        if grid is None or name not in reader._specs:
            continue
        spec = reader._specs[name]
        try:
            beta = reader.term_coefficients(name) if is_lss(reader) else _feature_beta(reader, name)
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


def _plain_display(name: str) -> FoldTermDisplay:
    """A SuperGLM term is shown by its name, a relativity the right way up, in no group."""
    return FoldTermDisplay(label=name, short=name)


def fold_term_items(
    terms: Mapping[str, EditableTerm],
    fold_curves: Mapping[int, Mapping[str, NDArray[np.float64]]],
    *,
    held: Collection[str] = (),
    display: Callable[[str], FoldTermDisplay] = _plain_display,
) -> list[dict[str, Any]]:
    """One chart entry per term: the mean terms before the CV and overdispersion terms, and
    within those the least stable first, then the terms ``held``.

    The in-force curve ("fit", and "edited" when edited) is drawn exactly as the
    Editor tab draws the term, a relativity (``display``'s transform), and each
    fold's curve is moved by a constant factor so its exposure-weighted mean log
    matches the in-force curve's. A term's ``fold_gap`` is the mean over folds of
    ``rmse_to_mean`` with every curve centred on its own mean log (the shape
    alone, whatever level it is drawn at), and ``min_correlation`` the lowest
    ``correlation_to_mean`` (``plotting.curve_similarity``), or NaN when a
    fold has none (a fold that dropped the term has a flat curve). A level a
    fold never saw is a gap (NaN) that distances skip. Each curve keeps its
    fold number.

    A ``held`` term is a hand edit Run CV put back on every fold: the same
    curve on each, so it carries neither measure and follows the measured terms.

    ``display`` says how each term is shown (:class:`FoldTermDisplay`).
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
            items.append(_term_item(name, term, grid, curves, held=name in held, display=display))
    items.sort(key=_least_stable_first)
    return items


def _least_stable_first(item: dict[str, Any]) -> tuple[bool, int, float, str]:
    """CV and overdispersion terms after the rest; then measured terms by falling fold gap, those
    without one, and the held terms.

    By name within each. A fold gap is unavailable (NaN) where no fold shares
    a point with another, and NaN does not order against a number.
    """
    after_average = item["role"] == "spread"
    if item["held"]:
        return (after_average, 2, 0.0, item["name"])
    fold_gap = item["fold_gap"]
    if fold_gap is None or not np.isfinite(fold_gap):
        return (after_average, 1, 0.0, item["name"])
    return (after_average, 0, -fold_gap, item["name"])


def _term_item(
    name: str,
    term: EditableTerm,
    grid: _TermGrid,
    curves,
    *,
    held: bool,
    display: Callable[[str], FoldTermDisplay],
) -> dict[str, Any]:
    shown = display(name)
    weights = (
        np.ones(grid.order.size, dtype=np.float64)
        if term.weights is None
        else np.asarray(term.weights, dtype=np.float64)[grid.order]
    )
    if not float(np.sum(weights)) > 0.0:
        weights = np.ones(grid.order.size, dtype=np.float64)
    fit = term.original_log_effect[grid.order]
    edited = term.edited_log_effect[grid.order]
    changed = not np.allclose(edited, fit, rtol=0.0, atol=1e-14)
    # The in-force curve is drawn as the Editor tab draws the term. A held term's
    # folds carry its edit, so they follow the edited curve; every other fold
    # follows the fit.
    anchor = edited if held else fit
    # Every curve is read on the same points, the ones every fold has a
    # value at, so a fold's gap does not shift it against the others.
    shared = np.logical_and.reduce([~np.isnan(values) for values in curves.values()])

    def log_mean(log_values: NDArray[np.float64]) -> float:
        points = shared if shared.any() else ~np.isnan(log_values)
        return weighted_mean(log_values[points], weights[points])

    def centred(log_values) -> NDArray[np.float64]:
        values = np.asarray(log_values, dtype=np.float64)
        return shown.transform.to_display(values - log_mean(values))

    def aligned(log_values) -> NDArray[np.float64]:
        # A fold's curve moved by a constant factor so its exposure-weighted mean
        # log is the in-force curve's over the same points: folds compare on shape,
        # and sit at the level the Editor tab draws.
        values = np.asarray(log_values, dtype=np.float64)
        points = shared if shared.any() else ~np.isnan(values)
        level = (
            weighted_mean(anchor[points], weights[points])
            if np.isfinite(anchor[points]).all()
            else 0.0
        )
        return shown.transform.to_display(values - log_mean(values) + level)

    # The measures read the curves on the mean-one scale, so a term's gap does
    # not depend on the level the curves are drawn at; correlation is unaffected
    # by a common factor, a root-mean-square distance would scale with it.
    vs_mean = (
        None
        if held
        else _summarize_against_fold_mean(
            {f"Fold {index + 1}": centred(values) for index, values in curves.items()}, weights
        )
    )
    folds = {f"Fold {index + 1}": aligned(values) for index, values in curves.items()}
    return {
        "name": name,
        "label": shown.label,
        "short": shown.short,
        "group": shown.group,
        "role": shown.role,
        "value_label": shown.value,
        "kind": grid.kind,
        "x": None if grid.kind == "levels" else grid.points,
        "levels": grid.labels,
        "weights": weights,
        "folds": [
            {"fold": index, "label": label, "values": values}
            for index, (label, values) in zip(curves, folds.items(), strict=True)
        ],
        "fit": shown.transform.to_display(fit),
        "edited": shown.transform.to_display(edited) if changed else None,
        "held": held,
        "fold_gap": None if vs_mean is None else float(vs_mean["rmse_to_mean"].mean()),
        "min_correlation": (
            None if vs_mean is None else float(vs_mean["correlation_to_mean"].min(skipna=False))
        ),
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
    run_stale: bool
    final_stale: bool
    pending: int
    run_reason: str | None
    final_reason: str | None
    has_validation: bool
    has_test: bool
    replay_note: str | None = None
    metrics: tuple[tuple[str, str, str, bool | None], ...] = _METRICS
    display: Callable[[str], FoldTermDisplay] = _plain_display


def _same_fit(model, reference) -> bool:
    """Whether two models are one fit: a SuperLSS session holds its own view of each."""
    if is_lss(model):
        return model.lss is reference.lss
    return model is reference


def capture_cv_view(session, *, run: CVRun | None, final_fit: FinalFit | None) -> CVTabView:
    """Copy what the tab reads; the caller holds the widget lock."""
    return CVTabView(
        supplied=session.cv,
        check=session.cv_check,
        terms={name: term.copy() for name, term in session.terms.items()},
        run=run,
        final_fit=final_fit,
        model_revision=session.model_revision,
        model_changed=not _same_fit(session.model, session.reference_model)
        or bool(session.edited_terms()),
        run_stale=run is not None and not run.source.describes(session),
        final_stale=final_fit is not None and not final_fit.source.describes(session),
        pending=len(session.pending),
        run_reason=run_cv_reason(session),
        replay_note=_replay_note(session),
        final_reason=final_fit_reason(session),
        has_validation="validation" in session._evaluation_data,
        has_test="test" in session._evaluation_data,
        metrics=_metrics_of(session.model),
        display=_term_display(session.model),
    )


def _replay_note(session) -> str | None:
    """The tab's notes when a supplied result does not record its fit method or its scorers."""
    if session.cv is None:
        return None
    notes = []
    if session.cv.fit_mode is None:
        notes.append(NO_FIT_MODE.format(method=_replay_method(session)))
    replayed = _replayed_scores(session.cv, session.model)
    if session.cv.builtin_scores is None and replayed:
        notes.append(NO_SCORERS.format(names=list(replayed)))
    return " ".join(notes) or None


def _replayed_scores(cv: CrossValidationResult, model) -> tuple[str, ...]:
    """The score columns of ``cv`` that Run CV computes again: those named like a built-in."""
    return tuple(name for name, *_ in _metrics_of(model) if name in cv.fold_scores.columns)


def _term_display(model) -> Callable[[str], FoldTermDisplay]:
    """How the tab shows a term of ``model``."""
    return partial(fold_chart_display, family=model.family) if is_lss(model) else _plain_display


def cv_report_payload(widget, *, request_sequence: int | None = None) -> dict[str, Any]:
    """The ``cv`` report: captured under the widget lock, built outside it."""
    with widget._lock:
        view = capture_cv_view(widget.session, run=widget._cv_run, final_fit=widget._final_fit)
        source = InForce.of(widget.session)
    jobs = {kind: widget._jobs.latest(kind) for kind in JOB_KINDS}
    payload = cv_tab_payload(view, jobs=jobs, request_sequence=request_sequence)
    payload["in_force"] = source.token
    return payload


def _untraced_note(expected: Sequence[int], folds: Sequence[dict[str, Any]]) -> str:
    """A sentence naming the folds that kept no trace; empty when every one did."""
    traced = {fit["id"] for fit in folds}
    missing = [f"Fold {number}" for number in expected if f"fold-{number}" not in traced]
    return f" No trace for {', '.join(missing)}: see the log." if missing else ""


def diagnose_report_payload(widget, *, request_sequence: int | None = None) -> dict[str, Any]:
    """The ``diagnose`` report: captured under the widget lock, built outside it.

    A finished Run CV kept its folds' traces; a supplied result's are read once
    and kept, since its fold models and rows do not change.
    """
    with widget._lock:
        session = widget.session
        model = session.model
        revision = session.model_revision
        run = widget._cv_run
        supplied = session.cv
        rows = session.cv_check.rows
        train = training_export_dataset(session)
        kept = widget._diagnose_folds
        kept_rows = widget._diagnose_all_rows
        same_fit = _same_fit(model, session.reference_model)
        run_current = run is not None and run.source.describes(session)
        in_force = InForce.of(session).token
    if run is not None:
        folds = list(run.fold_fits)
        source = (
            "Folds from Run CV on the current model."
            if run_current
            else "Folds from Run CV on an earlier version of this model. "
            "Press Run CV to see the current one."
        ) + _untraced_note(range(1, len(run.result.fold_scores) + 1), folds)
    elif supplied is None:
        folds, source = (
            [],
            "The fit on all rows only. Open the editor with cv=<a cross_validate() result> to see "
            "each fold.",
        )
    elif supplied.estimators is None:
        folds = []
        source = (
            "The fit on all rows only: the supplied folds kept no models. Pass return_estimators=True to "
            "cross_validate(), or press Run CV, to see them."
        )
    else:
        if kept is None or kept[0] is not supplied:
            kept = (supplied, supplied_fold_traces(supplied, rows))
            with widget._lock:
                widget._diagnose_folds = kept
        folds = kept[1]
        source = (
            "Folds from the cross-validation result you supplied."
            if same_fit
            else "Folds from the cross-validation result you supplied, fitted on the model as first opened. "
            "Press Run CV to see the current one."
        ) + _untraced_note(
            [i + 1 for i, fold in enumerate(supplied.estimators) if fold is not None], folds
        )
    # The fit on all rows ranks its training rows once per fitted model, as the folds' are kept.
    if kept_rows is None or kept_rows[0] is not model:
        kept_rows = (model, all_rows_worst_rows(model, train))
        with widget._lock:
            widget._diagnose_all_rows = kept_rows
    return {
        **diagnose_tab_payload(
            model,
            folds=folds,
            model_rows=kept_rows[1],
            source=source,
            display=_term_display(model),
            model_revision=revision,
            request_sequence=request_sequence,
        ),
        "in_force": in_force,
    }


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
            _result_payload(
                view.supplied, "As supplied", "supplied", None, view.model_changed, view.metrics
            )
        )
    run = view.run
    if run is not None:
        results.append(
            _result_payload(
                run.result,
                "Current model",
                "run",
                run.model_revision,
                view.run_stale,
                view.metrics,
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
            "note": " ".join(note for note in (view.check.note, view.replay_note) if note) or None,
        },
        "final_fit": {
            "available": view.final_reason is None,
            "reason": view.final_reason,
            "note": " ".join(note for note in final_notes if note) or None,
            "done": final is not None,
            "stale": view.final_stale,
            "n_rows": None if final is None else final.n_rows,
            "has_test_split": view.has_test,
        },
        "metrics": [
            {"name": name, "label": label, "short": short, "lower_is_better": lower}
            for name, label, short, lower in view.metrics
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
    metrics: tuple[tuple[str, str, str, bool | None], ...],
) -> dict[str, Any]:
    names = [name for name, *_ in metrics if name in result.fold_scores.columns]
    held_out = _folds_partition_rows(result)
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
        "held_out": held_out,
        "folds": folds,
        "mean": {name: result.mean_scores.get(name) for name in names},
        "std": {name: result.std_scores.get(name) for name in names},
        "pooled": {
            name: result.pooled_scores[name] for name in names if name in result.pooled_scores
        },
    }


def _folds_partition_rows(result: CrossValidationResult) -> bool:
    """Whether every row is held out by exactly one fold, as the tab's picture of the folds says."""
    folds = result.fold_indices
    if not folds:
        return False
    rows = _fold_row_count(result.n_rows, folds)
    held = np.concatenate([np.asarray(test, dtype=np.intp) for _train, test in folds])
    return bool(np.all(np.bincount(held, minlength=rows) == 1))


def headline_score(model, result: CrossValidationResult) -> dict[str, Any] | None:
    """The first score the tab lists for ``model``, with its label, to say how a run ended."""
    for name, label, *_ in _metrics_of(model):
        if name in result.mean_scores:
            return {"label": label, "value": result.mean_scores[name]}
    return None


def _relativities(view: CVTabView) -> dict[str, Any]:
    run = view.run
    if run is not None and run.terms:
        return {
            "available": True,
            "origin": "run",
            "stale": view.run_stale,
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
        "terms": fold_term_items(view.terms, curves, display=view.display),
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
    source: InForce


@dataclass(frozen=True)
class CVRunPlan(_Plan):
    """Everything Run CV reads."""

    rows: EvaluationDataset
    folds: tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]
    terms: dict[str, EditableTerm]
    edited: dict[str, EditableTerm]
    fit_mode: str
    warm_start: bool
    fit_kwargs: dict[str, Any]
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
    supplied = _replayed_scores(cv, session.model)
    return CVRunPlan(
        model=session.model,
        template=_declared_template(session),
        model_revision=session.model_revision,
        source=InForce.of(session),
        rows=session.cv_check.rows,
        folds=tuple(
            (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
            for train, test in cv.fold_indices
        ),
        terms=terms,
        edited={name: terms[name] for name in session.edited_terms()},
        fit_mode=_replay_method(session),
        # As the supplied folds were fitted; a result that records neither
        # predates both, so its folds were fitted cold with no extra arguments.
        warm_start=bool(cv.warm_start),
        fit_kwargs=dict(cv.fit_kwargs or {}),
        scoring=supplied or tuple(name for name, *_ in _metrics_of(session.model)),
        splitter=cv.splitter,
        n_points=session.n_points,
    )


def _replay_method(session) -> str:
    """The method Run CV fits each fold with: the supplied result's, so the two
    columns differ only by structure and edits; the current model's when the
    result predates the record (the tab's note says so). Its ``warm_start``
    and ``fit_kwargs`` are replayed with it (``capture_cv_run``)."""
    return session.cv.fit_mode or resolve_refit_method(session.model, "auto")


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
    if is_lss(session.model):
        return session.model.lss._clone_with_replaced_terms({})
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

    A grouping built on the train rows does not cover a level only the CV or
    validation rows hold. Such a level goes into the group the term's
    ``unseen`` names, as the in-force model predicts it and as
    ``Structure.apply(model, X=...)`` places it. A term with no group for new
    levels (``"error"`` or ``"base"``), or whose universe (``levels=``,
    ``bind_levels``, or an ordered term's ``order=``, ``values=`` and
    ``specials=``) leaves the level out, refuses ``job`` in one sentence. Every
    such term, and a factor smooth's group, also refuses a missing value.
    """
    replacements = _covering_replacements(template, X, job)
    return clone_with_replaced_features(template, replacements) if replacements else template


def _covering_replacements(template, X, job: str) -> dict[str, Any]:
    """The grouped categoricals of ``template`` (a SuperGLM or a SuperLSS view), widened to cover ``X``.

    Keyed as ``template._specs`` is; each term reads its column ``column_of(template, name)``.
    See :func:`_covering_template` for the rules.
    """
    frame = as_eager_frame(X)
    _require_valued_labels(template, frame, job)
    bindings = dict(getattr(template, "_level_bindings", None) or ())
    replacements = {}
    for name, spec in template._specs.items():
        if not isinstance(spec, Categorical | OrderedCategorical | RandomEffect):
            continue
        column = column_of(template, name)
        if column not in frame.columns:
            continue
        values = np.asarray(frame.column_array(column)).ravel()
        if isinstance(spec, OrderedCategorical):
            outside = _outside_order(values, spec)
            if outside:
                sentence = (
                    OUTSIDE_ORDERED_LEVELS if spec._grouping is None else OUTSIDE_ORDERED_GROUPS
                )
                raise EditorValueError(sentence.format(job=job, term=name, levels=outside))
            continue
        grouping = getattr(spec, "_grouping", None)
        if grouping is None:
            outside = _outside_universe(values, frame, column, spec, bindings.get(name))
            if outside:
                raise EditorValueError(
                    OUTSIDE_DECLARED_LEVELS.format(job=job, term=name, levels=outside)
                )
            continue
        new = _uncovered_labels(values, grouping)
        if not new:
            continue
        if spec._declared_levels is not None:
            raise EditorValueError(OUTSIDE_DECLARED_LEVELS.format(job=job, term=name, levels=new))
        # A binding (or a categorical dtype) names the universe as levels= does.
        outside = _outside_universe(values, frame, column, spec, bindings.get(name))
        if outside:
            raise EditorValueError(
                OUTSIDE_DECLARED_LEVELS.format(job=job, term=name, levels=outside)
            )
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
            spec,
            spec,
            base=spec.base,
            grouping=widened,
            data=np.asarray(new, dtype=object),
            level=base_names_level(spec),
        )
    return replacements


def _require_valued_labels(template, X, job: str) -> None:
    """Refuse a missing value in a column read as labels, which its term refuses to fit and to predict.

    That is a level term's column or a factor smooth's group: each refuses a
    missing value when it scores a row as well as when it fits one.
    """
    frame = as_eager_frame(X)
    for name in sorted(_label_columns(template), key=str):
        if name in frame.columns and _holds_missing(frame, name):
            raise EditorValueError(MISSING_LEVELS.format(job=job, term=name))


def _outside_universe(
    values, frame, name: str, spec: Categorical | RandomEffect, binding
) -> list[str]:
    """The labels in ``values``, column ``name``, outside ``spec``'s universe, as text.

    The universe is the one the fit binds: ``levels=``, else a categorical
    dtype on the column (which holds every value it has), else ``binding``.
    Values match it as the fit codes them. ``values`` hold no missing value.
    """
    universe = spec._declared_levels
    if universe is None and frame.column_declared_categories(name) is None:
        universe = None if binding is None else binding.levels
    if universe is None:
        return []
    outside = values[_codes_against(values, list(universe)) < 0]
    return sorted({str(label) for label in pd.unique(outside).tolist()}, key=str)


def _outside_order(values, spec: OrderedCategorical) -> list[str]:
    """The labels in ``values`` that ordered ``spec``'s fit does not take, as text.

    Values match the declaration as the fit matches them (``_canonical``),
    and a grouped term's through its grouping: a label the declaration
    leaves out, or one its groups do not cover. ``values`` hold no missing
    value.
    """
    labels = spec._canonical(values)
    if spec._grouping is not None:
        labels = _grouping_labels(labels)
    unknown = set(pd.unique(labels).tolist()) - spec._known_levels
    return sorted({str(label) for label in unknown}, key=str)


def _uncovered_labels(values, grouping) -> list[str]:
    """The labels in ``values`` that ``grouping`` does not map, as the text it matches by.

    ``values`` hold no missing value.
    """
    labels = pd.Series(values).astype(str).unique()
    return sorted(set(labels.tolist()) - set(grouping.original_to_group), key=str)


def run_cv(plan: CVRunPlan, context) -> CVRun:
    """Replay the stored folds on the in-force structure with the hand edits put back."""
    # A level outside a term's universe is checked on the rows some fold trains
    # on (a test-only row is scored as new levels); a missing label is refused
    # when scoring too, so those are checked on every row a fold reads.
    frame = as_eager_frame(plan.rows.X)
    lss = is_lss(plan.model)
    read = np.unique(np.concatenate([np.concatenate(fold) for fold in plan.folds]))
    fitted = np.unique(np.concatenate([train for train, _test in plan.folds]))
    if lss and plan.rows.sample_weight is not None:
        # A SuperLSS drops a zero-weight row before it reads the row's labels.
        weights = np.asarray(plan.rows.sample_weight)
        read, fitted = read[weights[read] > 0.0], fitted[weights[fitted] > 0.0]
    # A SuperLSS template is an unfitted copy; its view names the columns the terms read.
    _require_valued_labels(plan.model if lss else plan.template, frame.take_rows(read), "Run CV")
    if lss:
        covering = _covering_replacements(plan.model, frame.take_rows(fitted), "Run CV")
        template = (
            plan.model.lss._clone_with_replaced_terms(covering) if covering else plan.template
        )
        if plan.rows.offset is not None:
            template = template._clone_with_replaced_terms({})
            bind_replay_offsets(template, plan.rows.X, plan.rows.offset)
    else:
        template = _covering_template(plan.template, frame.take_rows(fitted), "Run CV")
    recorder = _FoldRecorder(plan, context)
    try:
        result = cross_validate(
            template,
            plan.rows.X,
            plan.rows.y,
            cv=StoredFolds(plan.folds, before_fold=recorder.before_fold),
            sample_weight=plan.rows.sample_weight,
            **({} if lss else {"offset": plan.rows.offset}),
            fit_mode=plan.fit_mode,
            scoring=recorder.score,
            error_score="raise",
            fit_kwargs=plan.fit_kwargs,
            warm_start=plan.warm_start,
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
            reason = _fit_failure(exc, FOLD_SEPARATED, FOLD_SOLVER_FAILED, FOLD_NOT_FITTED)
        raise EditorValueError(
            FOLD_FAILED.format(fold=recorder.fold_number, reason=reason)
        ) from exc
    context.check()
    context.progress("curves")
    return CVRun(
        result=replace(
            result,
            pooled_scores=recorder.pooled_scores(),
            splitter=plan.splitter,
            builtin_scores=plan.scoring,
        ),
        terms=fold_term_items(
            plan.terms, recorder.curves, held=plan.edited, display=_term_display(plan.model)
        ),
        model_revision=plan.model_revision,
        carried=tuple(sorted(plan.edited)),
        source=plan.source,
        fold_fits=tuple(recorder.fold_fits),
    )


class _FoldRecorder:
    """Run CV's per-fold hooks: progress and cancel between folds, edits before scoring.

    ``cross_validate`` fits each fold and hands the fitted model to
    :meth:`score`, which puts the hand edits back and computes the
    built-in scores on that edited model. A callable scorer's dict is not
    pooled by ``cross_validate``, so the pooled deviance and NLL are summed
    here from the same numerator and denominator parts it pools.
    """

    def __init__(self, plan: CVRunPlan, context) -> None:
        self._plan = plan
        self._context = context
        self._fold = -1
        self._fold_started = 0.0
        self._frame = as_eager_frame(plan.rows.X)
        self._y = np.asarray(plan.rows.y, dtype=np.float64)
        self._scorers = fold_adapter(plan.template)
        self._totals = {
            name: [0.0, 0.0] for name in plan.scoring if name in self._scorers.pooled_parts
        }
        self._columns = _fingerprint_columns(plan.template, self._frame)
        self.curves: dict[int, dict[str, NDArray[np.float64]]] = {}
        self.fold_fits: list[dict[str, Any]] = []

    @property
    def fold_number(self) -> int:
        """The fold being fitted or scored, counted from 1."""
        return self._fold + 1

    def before_fold(self, index: int) -> None:
        self._context.progress("fold", fold=index + 1, n_folds=len(self._plan.folds))
        self._context.check()
        self._fold = index
        self._fold_started = time.perf_counter()

    def score(self, model, X, y, *, sample_weight=None, offset=None) -> dict[str, float]:
        fit_time_s = time.perf_counter() - self._fold_started
        fit = self._scorers.fit_summary(model, self._plan.fit_mode)
        fold_fit = model
        train, test = self._plan.folds[self._fold]
        if self._plan.edited:
            model = model_with_edited_curves(
                model,
                self._plan.edited,
                self._frame.take_rows(train),
                self._y[train],
                _take(self._plan.rows.sample_weight, train),
                _take_offset(self._plan.rows.offset, train),
                n_points=self._plan.n_points,
            )
        scores: dict[str, float] = {}
        parts: dict[str, tuple[float, float]] = {}
        for name in self._plan.scoring:
            pooled = self._scorers.pooled_parts.get(name)
            if pooled is None:
                scorer = self._scorers.scorers[name]
                scores[name] = float(
                    scorer(model, X, y, sample_weight=sample_weight, offset=offset)
                )
                continue
            numerator, denominator = pooled(model, X, y, sample_weight=sample_weight, offset=offset)
            parts[name] = (numerator, denominator)
            scores[name] = parts_ratio(numerator, denominator)
        # Only a fold that scored completely joins the pooled totals.
        for name, (numerator, denominator) in parts.items():
            self._totals[name][0] += numerator
            self._totals[name][1] += denominator
        self.curves[self._fold] = fold_log_curves(model, self._plan.terms)
        self._keep_trace(fold_fit, fit_time_s, model, X, y, sample_weight, offset)
        self._context.progress(
            "fold_done",
            fold=self.fold_number,
            n_folds=len(self._plan.folds),
            n_train=len(train),
            n_test=len(test),
            fit_time_s=fit_time_s,
            scores=scores,
            **fit,
        )
        return scores

    def _keep_trace(self, fold_fit, fit_time_s, scored, X, y, sample_weight, offset) -> None:
        """Keep what the Diagnose tab shows of this fold. It decides nothing about the scores."""
        train, test = self._plan.folds[self._fold]
        try:
            trace = fit_trace(
                fold_fit,
                fit_id=f"fold-{self.fold_number}",
                label=f"Fold {self.fold_number}",
                n_rows=len(train),
                seconds=fit_time_s,
            )
            trace["worst_rows"] = worst_rows(
                self._scorers, scored, X, y, sample_weight, offset, test, self._columns
            )
        except Exception:
            _LOGGER.warning(
                "Fold %d keeps its scores but has no Diagnose trace.",
                self.fold_number,
                exc_info=True,
            )
            return
        self.fold_fits.append(trace)

    def pooled_scores(self) -> dict[str, float]:
        return {
            name: numerator / denominator
            for name, (numerator, denominator) in self._totals.items()
            if denominator != 0.0
        }


def _take(values, rows: NDArray[np.intp]):
    return None if values is None else np.asarray(values, dtype=np.float64)[rows]


# ── Final fit ────────────────────────────────────────────────────


@dataclass(frozen=True)
class FinalFitPlan(_Plan):
    """Everything Final fit reads."""

    datasets: tuple[EvaluationDataset, ...]
    test: EvaluationDataset | None
    edited: dict[str, EditableTerm]
    pending: int
    n_points: int


def capture_final_fit(session) -> FinalFitPlan:
    """Capture Final fit's inputs; refuse without training rows, or rows that cannot stack."""
    datasets = final_fit_datasets(session)
    reason = (
        (
            session_refit_reason(session)
            or _unstackable_reason(datasets)
            or _split_reason(session, datasets)
        )
        if datasets
        else NO_FINAL_ROWS
    )
    if reason is not None:
        raise EditorValueError(reason)
    return FinalFitPlan(
        model=session.model,
        template=_declared_template(session),
        model_revision=session.model_revision,
        source=InForce.of(session),
        datasets=datasets,
        test=session._evaluation_data.get("test"),
        edited={name: session.terms[name].copy() for name in session.edited_terms()},
        pending=len(session.pending),
        n_points=session.n_points,
    )


def run_final_fit(plan: FinalFitPlan, context) -> FinalFit:
    """Refit the in-force structure on train and validation rows, then put the edits back."""
    lss = is_lss(plan.model)
    # A SuperLSS template is an unfitted copy; its view names the columns the terms read.
    X, y, sample_weight, offset = _union_rows(plan.datasets, plan.model if lss else plan.template)
    context.progress("fitting", n_rows=int(y.size))
    context.check()
    if lss:
        # Only the rows the fit keeps read a level: a zero-weight row is dropped first.
        kept = (
            X
            if sample_weight is None
            else as_eager_frame(X).take_rows(np.flatnonzero(np.asarray(sample_weight) > 0.0))
        )
        covering = _covering_replacements(plan.model, kept, "Final fit")
        model = plan.model.lss._clone_with_replaced_terms(covering) if covering else plan.template
    else:
        model = _covering_template(plan.template, X, "Final fit").clone_unfitted()
    try:
        fit_refit_model(
            plan.model,
            model,
            method="auto",
            X=X,
            y=y,
            sample_weight=sample_weight,
            offset=offset,
        )
    except (ValueError, *_SOLVER_FAILURES) as exc:
        # Separation and a solver failure say what they are. Any other
        # ValueError is a row the checks above do not name (a missing or
        # non-finite number, a level outside an interaction's universe, a
        # response the family disallows).
        if isinstance(exc, EditorClientError):
            raise
        _LOGGER.warning("Final fit failed.", exc_info=True)
        rows = "train and validation" if len(plan.datasets) > 1 else "train"
        sentence = _fit_failure(exc, FINAL_SEPARATED, FINAL_SOLVER_FAILED, FINAL_NOT_FITTED)
        raise EditorValueError(sentence.format(rows=rows)) from exc
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
        source=plan.source,
        n_rows=int(y.size),
        splits=tuple(dataset.name for dataset in plan.datasets),
        carried=tuple(sorted(plan.edited)),
        pending=plan.pending,
        test_scores=lss_dataset_metrics(model, plan.test) if lss and plan.test else None,
    )


# A numerical failure of the solver, as opposed to rows the fit refuses.
_SOLVER_FAILURES = (
    np.linalg.LinAlgError,
    FloatingPointError,
    ObservedGeometryInfeasibleError,
    ObservedModeNotCertifiedError,
)


def _fit_failure(exc: BaseException, separated: str, solver: str, other: str) -> str:
    """The sentence for a refit that raised ``exc``: ``separated``, ``solver`` or ``other``."""
    if isinstance(exc, SeparationError):
        return separated
    if isinstance(exc, _SOLVER_FAILURES):
        return solver
    return other


def _union_rows(datasets: Sequence[EvaluationDataset], template):
    """The splits' rows stacked in order: one frame, response, weight and offset.

    The frame holds the columns ``template`` reads (:func:`_stacked_columns`).
    """
    frames = [as_eager_frame(dataset.X) for dataset in datasets]
    backends = {frame.backend for frame in frames}
    if len(backends) > 1:
        raise EditorValueError(MIXED_FRAMES)
    if len(frames) == 1:
        X = frames[0].native
    elif backends == {"pandas"}:
        names, casts = _stacked_columns(frames, template)
        # Splits declaring one set of categories for a label column stack as one
        # categorical (pandas would stack differing ordered flags as object and
        # drop a category no row holds). A numeric column takes its cast.
        first = cast(pd.DataFrame, frames[0].native)
        labelled = _label_columns(template)
        categorical = {
            name: first[name].dtype
            for name in names
            if name in labelled and isinstance(first[name].dtype, pd.CategoricalDtype)
        }
        X = pd.concat(
            [
                cast(pd.DataFrame, frame.native)[names].astype({**casts, **categorical})
                for frame in frames
            ],
            ignore_index=True,
        )
    else:
        import polars as pl

        names, casts = _stacked_columns(frames, template)
        polars_types = {"float64": pl.Float64, "int64": pl.Int64}
        X = pl.concat(
            [
                cast(pl.DataFrame, frame.native)
                .select(names)
                .cast({name: polars_types[kind] for name, kind in casts.items()})
                for frame in frames
            ],
            how="vertical_relaxed",
        )
    y = np.concatenate([np.asarray(dataset.y, dtype=np.float64) for dataset in datasets])
    return X, y, _stacked(datasets, "sample_weight", 1.0), _stacked(datasets, "offset", 0.0)


def _stacked_columns(frames, template) -> tuple[list[str], dict[str, str]]:
    """The columns ``template`` reads, and the dtype to stack some as; refused if a split differs.

    The fit reads a column as its NumPy array and the categories its dtype
    declares, so splits that agree on those stack without changing a value.
    So do numbers of other kinds or widths, and bools, in a column a term reads
    as a number (read as float64, True as 1). A column read as labels is
    stricter: a level term reads a number's text, so integers of any width
    stack as int64 but 1 and 1.0, or True and 1, are two levels; a factor
    smooth's group follows the same rule, since a cast would relabel the
    levels the kept model matches. uint64 is left alone (pandas would stack it
    beside a signed integer as float64).
    """
    # A SuperLSS view names its columns; a SuperGLM reads its declared features.
    names = list(
        template.data_columns()
        if isinstance(template, LSSModelView)
        else _fingerprint_columns(template, frames[0])
    )
    levels = _label_columns(template)
    casts: dict[str, str] = {}
    for name in names:
        readings = {_reading(frame, name, labels=name in levels) for frame in frames}
        if len(readings) == 1:
            continue
        if ("absent", None) in readings:
            raise EditorValueError(FINAL_COLUMN_MISSING.format(column=name))
        # A missing value turns an integer level column into floats or objects,
        # so name it, as Run CV does, rather than the dtype it changed.
        if name in levels and any(_holds_missing(frame, name) for frame in frames):
            raise EditorValueError(MISSING_LEVELS.format(job="Final fit", term=name))
        kinds = [kind for kind, _ in readings]
        if not all(isinstance(kind, np.dtype) for kind in kinds):
            raise EditorValueError(FINAL_COLUMN_TYPES.format(column=name))
        if name not in levels and all(kind.kind in "iufb" for kind in kinds):
            casts[name] = "float64"
        elif name in levels and all(kind.kind in "iu" and kind != np.uint64 for kind in kinds):
            casts[name] = "int64"
        else:
            raise EditorValueError(FINAL_COLUMN_TYPES.format(column=name))
    return names, casts


def _label_columns(template) -> set[str]:
    """The columns ``template`` reads as labels: its level terms', and its factor smooths' groups."""
    labelled = {
        column_of(template, name)
        for name, spec in template._specs.items()
        if isinstance(spec, Categorical | OrderedCategorical | RandomEffect)
    }
    for spec in getattr(template, "_interaction_specs", {}).values():
        if isinstance(spec, FactorSmooth):
            labelled.add(spec.group)
    return labelled


def _holds_missing(frame, name) -> bool:
    return bool(np.asarray(pd.isna(frame.column_array(name)), dtype=bool).any())


def _reading(frame, name, *, labels: bool = True) -> tuple[Any, tuple | None]:
    """How the fit reads ``frame``'s column ``name``: its NumPy dtype, then its declared categories.

    Any value not a number or a bool reads as ``"values"`` (text, whatever
    its dtype is called), and a column the frame lacks as ``"absent"``.
    Categories compare by type and text, not as values: the fit reads a
    category's text, so -0.0 and 0.0 are equal numbers but two levels, and
    stacking 1 under the categories of "1" would lose the row. A column not
    read as ``labels`` is read as its numbers, and its categories are left out.
    """
    if name not in frame.columns:
        return "absent", None
    dtype = frame.column_array(name).dtype
    categories = frame.column_declared_categories(name) if labels else None
    kind = dtype if dtype.kind in "iufb" and categories is None else "values"
    if categories is None:
        return kind, None
    return kind, tuple(f"{type(category).__name__}:{category}" for category in categories)


def _stacked(datasets: Sequence[EvaluationDataset], name: str, fill: float):
    """One column across the splits, ``fill`` where a split lacks it; None if all do.

    :func:`capture_final_fit` has refused a fill beside values that differ from it.
    """
    columns = [getattr(dataset, name) for dataset in datasets]
    if all(column is None for column in columns):
        return None
    if any(isinstance(column, Mapping) for column in columns):
        return stack_offsets(datasets)
    return np.concatenate(
        [
            np.full(dataset.n_obs, fill) if column is None else np.asarray(column, dtype=float)
            for dataset, column in zip(datasets, columns, strict=True)
        ]
    )
