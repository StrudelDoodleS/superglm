"""What the distribution panel draws one feature from.

The panel shows, for each row of one feature, the predicted
distribution with every other column at the reference row (a numeric column at
its median, any other at its most frequent level; ``LSSModelView.reference_row``)
and every offset at zero, which is one year of exposure for a count model. A
numeric feature gets ``N_ROWS`` evenly spaced values over its training range for
the ridges and ``N_COLUMNS`` for the density fan. A categorical feature gets one
row per level, the ``MAX_LEVELS`` with the most exposure when there are more,
and its fan columns are those rows.

The payload carries each parameter's link-scale predictor, for the fitted model
and for the edited one, in ``parameter_names_`` order, together with the link
that maps it to the family's natural parameter. The browser rebuilds every law
from those, so a dragged curve redraws without a request.

The amount axis covers the 0.001 to 0.999 quantiles of the fitted and the edited
rows, capped at ``MAX_AXIS_DECADES`` below its upper end; ``clipped_below`` says
when the cap raised the lower end.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.distributional.api import SuperLSS
from superglm.distributional.families import (
    GammaLS,
    LogNormalLS,
    LowerBoundedLogLink,
    NegativeBinomialLS,
)
from superglm.editor._types import EditableTerm
from superglm.editor.errors import EditorKeyError, EditorValueError
from superglm.editor.lss.apply import _native_levels
from superglm.editor.lss.display import parameter_display
from superglm.editor.lss.view import LSSModelView, lss_key
from superglm.links import Link, LogLink

N_ROWS = 8
N_COLUMNS = 96
MAX_LEVELS = 12
# The amount axis spans these quantiles of the panel's rows, widened to whole decades.
AXIS_QUANTILES = (0.001, 0.999)
# A spread edit can put the lower quantile many decades below the rest, so the axis stops here.
MAX_AXIS_DECADES = 10

NOT_SUPERLSS = (
    "The distribution panel is for SuperLSS models; this editor holds a SuperGLM model. "
    "Open a fitted SuperLSS model in the editor to use it."
)
COUNTS_NOT_DRAWN = "Count distributions are not drawn in this panel."
NO_SWEEP = (
    "This term does not follow one feature through its levels or its range, so the panel does "
    "not draw its distribution."
)

_FAMILIES = {
    LogNormalLS: "lognormal_mean",
    GammaLS: "gamma",
    NegativeBinomialLS: "negative_binomial",
}


def link_spec(parameter: str, link: Link) -> dict[str, Any]:
    """``link`` as the browser reads it: ``log``, or ``floored_log`` with its floor."""
    if type(link) is LogLink:
        return {"kind": "log"}
    if type(link) is LowerBoundedLogLink:
        return {"kind": "floored_log", "floor": float(link.floor)}
    raise EditorValueError(
        f"The distribution panel draws parameters with a log link or a shifted log link; the "
        f"{parameter!r} parameter uses {type(link).__name__}. Refit it with a log link to see "
        "its distribution here."
    )


def _family_name(family: object) -> str:
    """The panel's name for ``family``; the editor opens only the families it names."""
    return next(name for cls, name in _FAMILIES.items() if isinstance(family, cls))


def _parameters(view: LSSModelView) -> list[dict[str, Any]]:
    parameters = []
    for state in view._fitted.layout.predictors:
        shown = parameter_display(state.name)
        parameters.append(
            {
                "name": state.name,
                "label": shown.label,
                "role": shown.role,
                "link": link_spec(state.name, state.link),
            }
        )
    return parameters


def _feature_terms(session: Any, view: LSSModelView, feature: str) -> tuple[str, dict[str, str]]:
    """The feature's column and, per parameter, the opened term that reads it."""
    column = view.column_of(feature) if feature in view._specs else feature
    keys = {
        parameter: lss_key(parameter, column)
        for parameter in view.lss.parameter_names_
        if lss_key(parameter, column) in session.terms
    }
    if not keys:
        features = ", ".join(sorted({view.column_of(key) for key in session.terms}))
        raise EditorKeyError(
            f"No term of this editor reads {feature!r}; choose one of: {features}."
        )
    return column, keys


def _kind(terms: list[EditableTerm]) -> str | None:
    """``levels`` or ``numeric`` when every term sweeps the feature the same way, else ``None``."""
    if all(term.levels is not None for term in terms):
        return "levels"
    if all(term.levels is None and term.x is not None for term in terms):
        return "numeric"
    return None


def _numeric_sweep(view: LSSModelView, column: str) -> dict[str, Any]:
    values = np.asarray(as_eager_frame(view.X_train).column_array(column), dtype=np.float64)
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    rows = np.linspace(low, high, N_ROWS).tolist()
    columns = np.linspace(low, high, N_COLUMNS).tolist()
    return {
        "rows": [{"label": f"{x:.4g}", "x": x, "level": None} for x in rows],
        "columns": [{"x": x, "level": None} for x in columns],
        "values": rows + columns,
        "cap_note": None,
    }


def _level_sweep(view: LSSModelView, column: str, term: EditableTerm) -> dict[str, Any]:
    """The levels with the most exposure, in the term's own order; a level is a fan column too."""
    assert term.levels is not None and term.weights is not None
    kept = sorted(np.argsort(-term.weights, kind="stable")[:MAX_LEVELS])
    levels = [term.levels[i] for i in kept]
    # A label is the text of a training value, or of a declared level absent from the rows.
    present = {
        **_native_levels(view._specs[term.name]),
        **{str(v): v for v in pd.unique(as_eager_frame(view.X_train).column_array(column))},
    }
    capped = len(term.levels) > MAX_LEVELS
    return {
        "rows": [{"label": level, "x": None, "level": level} for level in levels],
        "columns": [{"x": None, "level": level} for level in levels],
        "values": [present.get(level, level) for level in levels],
        "cap_note": (
            f"Showing the {MAX_LEVELS} levels with the most exposure of {len(term.levels)}."
            if capped
            else None
        ),
    }


def _eta(model: SuperLSS, view: LSSModelView, column: str, values: list[Any]) -> NDArray:
    """``model``'s link-scale predictors at the reference row with ``column`` set to ``values``."""
    frame = view.reference_row().loc[np.zeros(len(values), dtype=np.intp)].reset_index(drop=True)
    frame[column] = values
    fitted = model._require_fitted()
    # The panel's reference neutralizes every offset, including columns also
    # used as features. Keep the frame intact and supply complete zero sums.
    offsets = {name: np.zeros(len(frame)) for name in fitted.parameter_names}
    return fitted._predict_eta_resolved(fitted.prediction_coefficients, frame, offsets)


def _amount_axis(view: LSSModelView, rows: NDArray) -> dict[str, Any]:
    """Whole decades around the rows' quantiles, at most ``MAX_AXIS_DECADES`` wide."""
    theta = view._fitted.parameters_from_eta(rows)
    low, high = (
        np.asarray(view.family.quantile(np.full(len(theta), p), theta), dtype=np.float64)
        for p in AXIS_QUANTILES
    )
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise EditorValueError(
            "The distribution panel cannot draw this feature: a row's amounts leave the "
            "range of float64. Check the edited curves of its spread."
        )
    top = math.ceil(math.log10(float(high.max())))
    lowest = float(low.min())
    # A lower quantile can underflow to zero, which no decade reaches.
    wanted = math.floor(math.log10(lowest)) if lowest > 0.0 else -math.inf
    floor = top - MAX_AXIS_DECADES
    return {"min_log10": max(wanted, floor), "max_log10": top, "clipped_below": wanted < floor}


def _reference(view: LSSModelView, column: str) -> list[dict[str, Any]]:
    row = view.reference_row().iloc[0]
    return [
        {"column": name, "value": value.item() if isinstance(value, np.generic) else value}
        for name, value in row.items()
        if name != column
    ]


def _empty_panel() -> dict[str, Any]:
    return {
        "kind": "empty",
        "rows": [],
        "columns": [],
        "eta": {
            "fitted": {"rows": [], "columns": []},
            "edited": {"rows": [], "columns": []},
        },
        "amount_axis": None,
        "reference": [],
        "cap_note": None,
    }


def distribution_panel_payload(
    session: Any, feature: str, *, edited: SuperLSS | LSSModelView
) -> dict[str, Any]:
    """The panel's rows, fan columns, link-scale predictors and amount axis for ``feature``.

    ``feature`` is a data column, or a composite key such as ``"scale:age"``.
    ``edited`` is the model in force: the editor's view of the fitted model when
    nothing is edited, else the model with the curves written into it. The
    fitted and the edited model are each asked for their predictors once, the
    rows and the fan columns stacked in one frame, and the fitted model's
    answer is reused when nothing is edited.
    """
    view = session.model
    if not isinstance(view, LSSModelView):
        raise EditorValueError(NOT_SUPERLSS)
    family = _family_name(view.family)
    column, keys = _feature_terms(session, view, feature)
    terms = [session.terms[key] for key in keys.values()]
    head = {
        "feature": column,
        "family": family,
        "parameters": _parameters(view),
        "term_keys": keys,
        "amount_unit": view.amount_unit,
        "empty_reason": None,
    }
    kind = _kind(terms)
    if kind is None:
        return {**head, **_empty_panel(), "empty_reason": NO_SWEEP}

    sweep = (
        _numeric_sweep(view, column) if kind == "numeric" else _level_sweep(view, column, terms[0])
    )
    n_rows = len(sweep["rows"])
    fitted = _eta(view.lss, view, column, sweep["values"])
    revised = edited.lss if isinstance(edited, LSSModelView) else edited
    edited_eta = fitted if revised is view.lss else _eta(revised, view, column, sweep["values"])
    etas = {
        name: {
            "rows": eta[:n_rows].tolist(),
            "columns": (eta[n_rows:] if kind == "numeric" else eta).tolist(),
        }
        for name, eta in (("fitted", fitted), ("edited", edited_eta))
    }
    counts = family == "negative_binomial"
    shown_rows = np.vstack([fitted[:n_rows], edited_eta[:n_rows]])
    return {
        **head,
        "kind": "empty" if counts else kind,
        "empty_reason": COUNTS_NOT_DRAWN if counts else None,
        "rows": sweep["rows"],
        "columns": sweep["columns"],
        "eta": etas,
        "amount_axis": None if counts else _amount_axis(view, shown_rows),
        "reference": _reference(view, column),
        "cap_note": sweep["cap_note"],
    }
