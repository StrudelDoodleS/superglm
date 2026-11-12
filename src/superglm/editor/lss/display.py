"""The fields a SuperLSS term and its history entries carry, named by the term's parameter.

How a term is shown, and the names it is shown by, are decided in
:mod:`superglm.editor.display`, for both model kinds; the names are re-exported
here.
"""

from __future__ import annotations

from superglm.editor.display import (
    INVERSE_THETA,
    RELATIVITY,
    DisplayTransform,
    FoldTermDisplay,
    ParameterDisplay,
    directed_operation,
    display_for,
    fold_chart_display,
    parameter_display,
    parameter_display_for,
    term_y_label,
    y_label_for,
)
from superglm.editor.lss.view import LSSModelView, split_key

__all__ = [
    "INVERSE_THETA",
    "RELATIVITY",
    "DisplayTransform",
    "FoldTermDisplay",
    "ParameterDisplay",
    "directed_operation",
    "display_for",
    "fold_chart_display",
    "lss_history_fields",
    "lss_term_fields",
    "parameter_display",
    "y_label_for",
]


def lss_history_fields(view: LSSModelView, key: str | None) -> dict[str, str | bool]:
    """The parameter, its chip, its role and the term a history entry names; empty for no term.

    ``predictor_role`` is the chip's colour group, so the browser need not read the chip text.

    ``display_decreasing`` says the shown curve falls as the recorded effect rises,
    so History names a recorded shift, monotone fit or snap by the shown direction.
    """
    if key is None:
        return {}
    parameter, label = split_key(key)
    shown = parameter_display_for(view, parameter)
    return {
        "predictor": parameter,
        "predictor_label": shown.chip,
        "predictor_role": shown.role,
        "term_label": label,
        "display_decreasing": not display_for(view, key).increasing,
    }


def lss_term_fields(view: LSSModelView, key: str) -> dict[str, object]:
    """The payload fields a SuperLSS term carries beyond a SuperGLM term's."""
    parameter, label = split_key(key)
    shown = parameter_display_for(view, parameter)
    return {
        "parameter": parameter,
        "parameter_label": shown.label,
        "parameter_role": shown.role,
        "term_label": label,
        "display": display_for(view, key).to_payload(),
        "y_label": term_y_label(view, key),
        "x_label": label,
        "title": label,
    }
