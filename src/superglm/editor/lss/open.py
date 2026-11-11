"""Open a fitted SuperLSS in the editor.

The refusals are checked in a fixed order, so a model that breaks several rules
is told about the first: the family, the log-normal's form, terms in a Tweedie
power, a model never fitted, truncated or censored rows, a parameter on a link other than its
default, a column named by something other than a string, missing training
rows, a centering other than native, the offsets, then training rows that are
not the fit's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from superglm.distributional.families import LogNormalLS, TweedieLSS
from superglm.distributional.prediction_design import _required_columns
from superglm.distributional.predictor import resolve_predictor_link
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.evaluation import coerce_dataset, coerce_evaluation_data
from superglm.editor.lss.display import SPREAD_FAMILIES
from superglm.editor.lss.terms import lss_editable_terms
from superglm.editor.lss.view import NATIVE_ONLY, LSSModelView

_OPENS = (
    "The editor opens LogNormalLS (mean form), GammaLS, NegativeBinomialLS and TweedieLSS models"
)
_NEEDS_ROWS = (
    "Opening a SuperLSS model needs its training rows: pass "
    "train_data=(X, y, sample_weight, offsets), with offsets a dict keyed by parameter name, "
    "or None."
)


#: Linear-predictor values at which a fitted link is compared with its parameter's default.
_LINK_PROBE = np.array([-2.0, -0.5, 0.0, 0.7, 2.0])


def _overridden_link(lss) -> str | None:
    """The first parameter whose fitted link is not its default; None when there is none.

    The curves show each parameter through its default link (a relativity, a
    coefficient of variation, ``1 / theta``), so a parameter fitted on another
    link would be shown, and edited, in the wrong units.
    """
    declared = {predictor.name: predictor for predictor in lss._predictors}
    states = lss._model.layout.predictors
    for parameter, state in zip(lss.family.parameters, states, strict=True):
        predictor = declared[state.name]
        if predictor.link is None:
            continue
        default = resolve_predictor_link(parameter, dataclasses.replace(predictor, link=None))
        fitted = np.asarray(state.link.inverse(_LINK_PROBE), dtype=np.float64)
        if not np.allclose(fitted, default.inverse(_LINK_PROBE), rtol=1e-12, atol=0.0):
            return state.name
    return None


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
    family = lss.family
    if not isinstance(family, SPREAD_FAMILIES):
        raise EditorValueError(f"{_OPENS}; this model is {type(family).__name__}.")
    if isinstance(family, LogNormalLS) and family.parametrisation != "mean":
        raise EditorValueError(
            "The editor opens LogNormalLS models in mean form; this one uses "
            f"parametrisation={family.parametrisation!r}."
        )
    if isinstance(family, TweedieLSS):
        power = next(p for p in lss._predictors if p.name == "power")
        terms = [str(name) for name in (*power.features, *power.interaction_order)]
        if terms:
            raise EditorValueError(
                "The editor shows a Tweedie model whose power is a constant; this model's power "
                f"has terms ({', '.join(terms)}). Refit it with the power as an intercept only, "
                "family.p(), to open it in the editor."
            )
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
    overridden = _overridden_link(lss)
    if overridden is not None:
        raise EditorValueError(
            f"The editor draws every parameter through its family's default link; this model "
            f"fits {overridden!r} on another link. Refit it without link= on {overridden!r} "
            "to open it in the editor."
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
    view = LSSModelView(lss, None, amount_unit=amount_unit, train=train)
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
