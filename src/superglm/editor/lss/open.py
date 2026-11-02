"""Open a fitted SuperLSS in the editor.

The refusals are checked in a fixed order, so a model that breaks several rules
is told about the first: the family, the log-normal's form, a model never
fitted, truncated or censored rows, missing training rows, a centering other
than native, then the offsets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from superglm.distributional.families import LogNormalLS
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.evaluation import coerce_dataset, coerce_evaluation_data
from superglm.editor.lss.display import SPREAD_KINDS
from superglm.editor.lss.terms import lss_editable_terms
from superglm.editor.lss.view import NATIVE_ONLY, LSSModelView, Response

_OPENS = "The editor opens LogNormalLS (mean form), GammaLS and NegativeBinomialLS models"
_NEEDS_ROWS = (
    "Opening a SuperLSS model needs its training rows: pass "
    "train_data=(X, y, sample_weight, offsets), with offsets a dict keyed by parameter name, "
    "or None."
)


def _refuse_unopenable(lss, train_data, centering: str) -> None:
    family = lss.family
    if not isinstance(family, tuple(SPREAD_KINDS)):
        raise EditorValueError(f"{_OPENS}; this model is {type(family).__name__}.")
    if isinstance(family, LogNormalLS) and family.parametrisation != "mean":
        raise EditorValueError(
            "The editor opens LogNormalLS models in mean form; this one uses "
            f"parametrisation={family.parametrisation!r}."
        )
    if lss._model is None:
        raise RuntimeError("Model must be fitted before creating an editor session.")
    state = lss._model.fit_state
    if state.truncated_row_count or state.censored_row_count:
        raise EditorValueError(
            "The editor opens models fitted without truncation or censoring; this model has "
            f"{state.truncated_row_count} truncated and {state.censored_row_count} censored rows."
        )
    if train_data is None:
        raise EditorValueError(_NEEDS_ROWS)
    if centering != "native":
        raise EditorValueError(NATIVE_ONLY)


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
    response: Response,
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
    view = LSSModelView(lss, train.X, amount_unit=amount_unit, response=response)
    editable = lss_editable_terms(
        view, _term_keys(view, terms), n_points=n_points, with_se=with_se, train=train
    )
    return cls(
        view,
        editable,
        n_points=n_points,
        centering=centering,
        reference_model=view.with_model(lss),
        evaluation_data=evaluation_data,
        cv_report=cv_report,
        cv=cv,
        cv_data=cv_rows,
    )
