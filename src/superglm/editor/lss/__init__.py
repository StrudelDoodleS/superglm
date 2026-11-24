"""Editing a fitted SuperLSS: its terms keyed ``parameter:term``, shown in readable units."""

import typing as _typing

from superglm._lazy import attach

if _typing.TYPE_CHECKING:
    from superglm.editor.lss.display import (
        INVERSE_THETA,
        RELATIVITY,
        DisplayTransform,
        ParameterDisplay,
        display_for,
        lss_term_fields,
        parameter_display,
    )
    from superglm.editor.lss.terms import LSSTermInference, lss_editable_terms
    from superglm.editor.lss.view import (
        LSSModelView,
        column_of,
        interaction_users,
        is_lss,
        lss_key,
        predictor_of,
        split_key,
    )

# Each name is read from its module on first access (``superglm._lazy``): the package's body
# imports none of its modules, so a thread that imports one of them by path while another thread
# runs this body never waits on a sibling that thread holds.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm.editor.lss.display": (
        "INVERSE_THETA",
        "RELATIVITY",
        "DisplayTransform",
        "ParameterDisplay",
        "display_for",
        "lss_term_fields",
        "parameter_display",
    ),
    "superglm.editor.lss.terms": ("LSSTermInference", "lss_editable_terms"),
    "superglm.editor.lss.view": (
        "LSSModelView",
        "column_of",
        "interaction_users",
        "is_lss",
        "lss_key",
        "predictor_of",
        "split_key",
    ),
}

__getattr__, __dir__ = attach(__name__, __path__, _LAZY_EXPORTS)

__all__ = [
    "INVERSE_THETA",
    "RELATIVITY",
    "DisplayTransform",
    "LSSModelView",
    "LSSTermInference",
    "ParameterDisplay",
    "column_of",
    "display_for",
    "interaction_users",
    "is_lss",
    "lss_editable_terms",
    "lss_key",
    "lss_term_fields",
    "parameter_display",
    "predictor_of",
    "split_key",
]
