"""What the one-risk view draws: a few named risks, each priced by the fitted and the edited model.

A risk is a value for each feature column of the model. The browser names the
columns it changes and the rest take the reference row's value
(``LSSModelView.reference_row``). The payload carries each risk's link-scale
predictors, for the fitted model and for the model in force, in
``parameter_names_`` order with every offset at zero, so the browser rebuilds
the risk's distribution from them, as it does for the distribution panel.

It also carries what explains each risk: the predictors of the reference risk,
and how far each feature moves them when it alone takes the risk's value
(Baseline Shapley with the reference row as baseline: Sundararajan and Najmi,
*The many Shapley values for model explanation*, ICML 2020). In a model of
main effects those moves add up to the risk's own; what an interaction adds is
the difference, which the browser shows as its own row, as a break-down plot
does (Gosiewska and Biecek, *Do not trust additive explanations*, 2019).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from superglm._frame import as_eager_frame
from superglm.distributional.prediction_design import _required_columns
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.lss.apply import _native_levels
from superglm.editor.lss.panel import (
    NOT_SUPERLSS,
    _amount_axis,
    _eta_of_frame,
    _family_name,
    _parameters,
)
from superglm.editor.lss.view import LSSModelView, lss_key

MAX_RISKS = 8
# A refusal names this many of a column's levels.
SHOWN_LEVELS = 5


def _lss_view(session: Any) -> LSSModelView:
    if not isinstance(session.model, LSSModelView):
        raise EditorValueError(NOT_SUPERLSS)
    return session.model


def _scan(view: LSSModelView, column: str) -> tuple:
    """What the training rows hold in ``column``: its range, or its values by label and in order.

    Read once per view and kept on it: a view's training rows do not change.
    """
    if column not in view._column_scans:
        frame = as_eager_frame(view.X_train)
        values = frame.column_array(column)
        if frame.column_kind(column) == "numeric":
            numbers = np.asarray(values, dtype=np.float64)
            view._column_scans[column] = (float(np.nanmin(numbers)), float(np.nanmax(numbers)))
        else:
            seen = {str(v): v for v in pd.unique(values) if pd.notna(v)}
            ordered = [str(level) for level in pd.factorize(values, sort=True)[1]]
            view._column_scans[column] = (seen, ordered)
    return view._column_scans[column]


def _describe(session: Any, view: LSSModelView, column: str) -> tuple[dict, dict[str, Any]]:
    """The column's entry of ``features`` and, for levels, each label's training value."""
    reference = view.reference_row()[column].iloc[0]
    terms = {
        parameter: lss_key(parameter, column)
        for parameter in view.lss.parameter_names_
        if lss_key(parameter, column) in session.terms
    }
    head = {"column": column, "terms": terms}
    first, second = _scan(view, column)
    if isinstance(first, float):
        return {
            **head,
            "kind": "numeric",
            "min": first,
            "max": second,
            "reference": float(reference),
            "levels": None,
        }, {}
    # A label is the text of a training value, or of a level a term declares that no row holds.
    native = {
        **{k: v for key in terms.values() for k, v in _native_levels(view._specs[key]).items()},
        **first,
    }
    opened = next(
        (session.terms[key].levels for key in terms.values() if session.terms[key].levels), None
    )
    return {
        **head,
        "kind": "levels",
        "min": None,
        "max": None,
        "reference": str(reference),
        "levels": [str(level) for level in opened] if opened else second,
    }, native


def _complete(features: dict[str, dict], native: dict[str, dict], risk: Mapping) -> dict[str, Any]:
    """``risk`` with every feature column given a value; a refusal names what to change."""
    if not isinstance(risk, Mapping):
        raise EditorValueError("Each risk is a mapping from a data column to its value.")
    unknown = next((column for column in risk if column not in features), None)
    if unknown is not None:
        raise EditorKeyError(
            f"No feature of this model reads {unknown!r}; choose one of: {', '.join(features)}."
        )
    values = {column: feature["reference"] for column, feature in features.items()}
    for column, value in risk.items():
        if features[column]["kind"] == "numeric":
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise EditorValueError(
                    f"{column!r} takes a finite number; got {value!r}. Enter a number."
                )
            values[column] = float(value)
        else:
            if str(value) not in native[column]:
                known = features[column]["levels"]
                more = " and others" if len(known) > SHOWN_LEVELS else ""
                raise EditorValueError(
                    f"{value!r} is not a level of {column!r}, which holds "
                    f"{', '.join(known[:SHOWN_LEVELS])}{more}. Choose one of its levels."
                )
            values[column] = str(value)
    return values


def risk_payload(session: Any, risks: list[Mapping], *, edited: Any) -> dict[str, Any]:
    """The one-risk view's features, completed risks and link-scale predictors.

    ``edited`` is the model in force, as in ``distribution_panel_payload``. The
    fitted and the edited model are each asked once for all risks, in one
    frame, and the fitted answer is reused when nothing is edited.
    """
    view = _lss_view(session)
    if not isinstance(risks, list) or not 1 <= len(risks) <= MAX_RISKS:
        raise EditorValueError(f"Give between 1 and {MAX_RISKS} risks to compare.")
    columns = _required_columns(view._fitted.compiled_predictors)
    described = {column: _describe(session, view, column) for column in columns}
    features = {column: feature for column, (feature, _) in described.items()}
    native = {column: levels for column, (_, levels) in described.items()}
    completed = [_complete(features, native, risk) for risk in risks]

    # One frame: the reference risk, each risk, then each risk's features moved one at a time.
    reference = {column: feature["reference"] for column, feature in features.items()}
    alone = [{**reference, column: risk[column]} for risk in completed for column in columns]
    rows = [reference, *completed, *alone]
    frame = view.reference_row().loc[np.zeros(len(rows), dtype=np.intp)].reset_index(drop=True)
    for column in columns:
        frame[column] = [native[column].get(row[column], row[column]) for row in rows]
    fitted = _eta_of_frame(view.lss, frame)
    revised = edited.lss if isinstance(edited, LSSModelView) else edited
    edited_eta = fitted if revised is view.lss else _eta_of_frame(revised, frame)
    n = len(completed)
    own = slice(1, 1 + n)

    def effects(eta: np.ndarray) -> np.ndarray:
        return (eta[1 + n :] - eta[0]).reshape(n, len(columns), -1)

    counts = _family_name(view.family) == "negative_binomial"
    return {
        "family": _family_name(view.family),
        "parameters": _parameters(view),
        "amount_unit": view.amount_unit,
        "kind": "counts" if counts else "risk",
        "features": list(features.values()),
        "reference": {"fitted": fitted[0].tolist(), "edited": edited_eta[0].tolist()},
        "risks": [
            {
                "values": values,
                "eta": {"fitted": fitted_row.tolist(), "edited": edited_row.tolist()},
                "effects": {"fitted": fitted_alone.tolist(), "edited": edited_alone.tolist()},
            }
            for values, fitted_row, edited_row, fitted_alone, edited_alone in zip(
                completed,
                fitted[own],
                edited_eta[own],
                effects(fitted),
                effects(edited_eta),
                strict=True,
            )
        ],
        "amount_axis": None
        if counts
        else _amount_axis(view, np.vstack([fitted[own], edited_eta[own]])),
    }


def risk_row(session: Any, index: int | None = None) -> dict[str, Any]:
    """One training row's feature values; ``index=None`` draws a row at random."""
    view = _lss_view(session)
    frame = as_eager_frame(view.X_train)
    columns = _required_columns(view._fitted.compiled_predictors)
    arrays = {column: frame.column_array(column) for column in columns}
    complete = np.logical_and.reduce([pd.notna(array) for array in arrays.values()])
    if index is None:
        index = int(np.random.default_rng().choice(np.flatnonzero(complete)))
    elif isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(complete):
        raise EditorValueError(
            f"Row {index!r} is not a training row; choose 0 to {len(complete) - 1}."
        )
    elif not complete[index]:
        raise EditorValueError(f"Row {index} has a missing value; choose another row.")
    values = {
        column: float(array[index]) if frame.column_kind(column) == "numeric" else str(array[index])
        for column, array in arrays.items()
    }
    return {"values": values, "index": index}
