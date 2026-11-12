"""Open a fitted SuperLSS in the editor.

Every scalar-response family opens, in any parametrisation and on any link, with
terms in any parameter; each curve is drawn by the link it was fitted on
(``superglm.editor.display``). The refusals are checked in a fixed order, so a
model that breaks several rules is told about the first: a model never fitted,
truncated or censored rows, a column named by something other than a string,
missing training rows, a centering other than native, the offsets, then training
rows that are not the fit's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from superglm.distributional.prediction_design import _required_columns
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.evaluation import coerce_dataset, coerce_evaluation_data
from superglm.editor.lss.terms import lss_editable_terms
from superglm.editor.lss.view import NATIVE_ONLY, LSSModelView

_NEEDS_ROWS = (
    "Opening a SuperLSS model needs its training rows: pass "
    "train_data=(X, y, sample_weight, offsets), with offsets a dict keyed by parameter name, "
    "or None."
)


def _data_columns(lss) -> tuple[Any, ...]:
    return _required_columns(lss._require_fitted().compiled_predictors)


def _refuse_other_rows(lss, n_rows: int) -> None:
    """Refuse training rows of another length than the fit's: an edit recomputes the fit's rows.

    The rows may be the frame the fit was given or the rows it kept, as a hand
    edit reads them.
    """
    provenance = lss._model.fit_state.weight_provenance
    if n_rows not in (provenance.original_count, provenance.retained_count):
        raise EditorValueError(
            f"The editor needs the rows the model was fitted on: train_data has {n_rows} rows, "
            f"but the fit was given {provenance.original_count} and kept "
            f"{provenance.retained_count}. Pass the frame the model was fitted on."
        )


def _refuse_unopenable(lss, train_data, centering: str) -> None:
    if lss._model is None:
        raise RuntimeError("Model must be fitted before creating an editor session.")
    state = lss._model.fit_state
    if state.truncated_row_count or state.censored_row_count:
        raise EditorValueError(
            "The editor opens models fitted without truncation or censoring; this model has "
            f"{state.truncated_row_count} truncated and {state.censored_row_count} censored rows."
        )
    unnamed = [name for name in _data_columns(lss) if not isinstance(name, str)]
    if unnamed:
        raise EditorValueError(
            "The editor names each SuperLSS term '<parameter>:<column>', so it opens models "
            f"whose columns are named by strings; this model reads column {unnamed[0]!r}. Rename "
            "the column to a string and refit to open it in the editor."
        )
    if train_data is None:
        raise EditorValueError(_NEEDS_ROWS)
    if centering != "native":
        raise EditorValueError(NATIVE_ONLY)


def refuse_unopenable_fit(lss) -> None:
    """Refuse a fit the editor would not open, with opening's own sentence: a session that
    follows a refit in place adopts it only where opening a fresh session on it would.

    Only the checks a fit can change in place apply: rows and centering are the session's.
    """
    _refuse_unopenable(lss, (), "native")


def _keys_named(view: LSSModelView, name: str) -> list[str]:
    """The keys ``name`` opens: a composite key, or a column's term in every parameter."""
    if name in view._specs:
        return [name]
    if name in view._interaction_specs:
        raise EditorValueError(f"Interactions are not editable in v1: {name!r}")
    keys = [key for key in view._feature_order if view.column_of(key) == name]
    if keys:
        return keys
    if any(view.column_of(key) == name for key in view._interaction_specs):
        raise EditorValueError(f"Interactions are not editable in v1: {name!r}")
    raise EditorKeyError(f"Term not found: {name!r}")


def _term_keys(view: LSSModelView, terms: Sequence[str] | None) -> list[str]:
    if terms is None:
        return list(view._feature_order)
    return list(dict.fromkeys(key for name in terms for key in _keys_named(view, name)))


def open_lss(
    cls: type,
    model: Any,
    terms: Sequence[str] | None,
    *,
    n_points: int,
    centering: str,
    with_se: bool,
    train_data,
    validation_data,
    test_data,
    cv_report,
    cv,
    cv_data,
    amount_unit: str | None,
):
    """An ``EditorSession`` (``cls``) over a fitted SuperLSS, its terms keyed ``parameter:term``."""
    lss = model.lss if isinstance(model, LSSModelView) else model
    _refuse_unopenable(lss, train_data, centering)
    names = lss.parameter_names_
    evaluation_data = coerce_evaluation_data(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        family=lss.family,
        weight_semantics=lss.weight_semantics,
        offset_parameters=names,
    )
    cv_rows = coerce_dataset(
        "cv",
        cv_data,
        family=lss.family,
        weight_semantics=lss.weight_semantics,
        offset_parameters=names,
    )
    train = evaluation_data["train"]
    _refuse_other_rows(lss, train.n_obs)
    view = LSSModelView(lss, None, amount_unit=amount_unit, splits=evaluation_data)
    editable = lss_editable_terms(
        view, _term_keys(view, terms), n_points=n_points, with_se=with_se, train=train
    )
    return cls(
        view,
        editable,
        n_points=n_points,
        centering=centering,
        reference_model=view,
        evaluation_data=evaluation_data,
        cv_report=cv_report,
        cv=cv,
        cv_data=cv_rows,
        with_se=with_se,
    )
