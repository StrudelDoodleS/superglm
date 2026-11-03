"""The editor tools a SuperLSS model does not offer, and the sentence that says why.

Each tool refuses with its own sentence (``require_available``) and its payload
lists the same sentence beside it, so a hover text and the error a direct call
raises never differ. The structural tools (collapse, ungroup, reference, knots
and basis) work on a SuperLSS term as they do on a SuperGLM one and are absent
here.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from superglm.editor.errors import EditorValueError
from superglm.editor.lss.view import is_lss

LSS_UNAVAILABLE: Mapping[str, str] = MappingProxyType(
    {
        "shape": "SuperLSS fits ignore shape constraints, so a shaped range would not hold "
        "after Refit.",
        "special": "Special levels are not available for SuperLSS terms.",
        "free_levels": "The free-level comparison is not available for SuperLSS terms.",
        "unsmoothed": "The unsmoothed line is not available for SuperLSS terms.",
        "unseen": "New-level choices are not available for SuperLSS terms.",
        "refit_offset": "Refitting with an edited offset is not available for SuperLSS models.",
        "profile_distribution": "SuperLSS models estimate their distribution parameters in the "
        "fit itself.",
        "rating_table": "Rating tables are not available for SuperLSS models.",
        "summary": "The coefficient summary is not available here for SuperLSS models; call "
        "model.summary() in Python.",
        "metrics": "Compare SuperLSS fits in the Validation tab.",
        "export_structure": "This export is for SuperGLM models; export a SuperLSS model as "
        "joblib.",
        "export_xlsx": "This export is for SuperGLM models; export a SuperLSS model as joblib.",
        "save_session": "Session files hold SuperGLM models. Export the edited SuperLSS model as "
        "joblib; it keeps its history.",
        "cv": "Cross-validation is not available for SuperLSS models; Final fit refits on all "
        "rows.",
    }
)

# What the frontend reads to hide a tab: no cross-validation, but Final fit.
LSS_CAPABILITIES: Mapping[str, bool] = MappingProxyType({"cv": False, "final_fit": True})

# The tools tied to one term, listed on that term's payload.
TERM_TOOLS = ("shape", "special", "free_levels", "unsmoothed", "unseen")
# The tools of the whole session, listed on the editor state.
SESSION_TOOLS = (
    "refit_offset",
    "profile_distribution",
    "rating_table",
    "summary",
    "metrics",
    "cv",
    "export_structure",
    "export_xlsx",
    "save_session",
)


def require_available(model: object, tool: str) -> None:
    """Refuse ``tool`` with its sentence when ``model`` is a SuperLSS or the editor's view of one."""
    if is_lss(model):
        raise EditorValueError(LSS_UNAVAILABLE[tool])


def lss_availability(term_payload: dict[str, Any]) -> dict[str, Any]:
    """``term_payload`` with the shape tool's reason, and each term tool's reason listed."""
    return {
        **term_payload,
        "shape": {
            **term_payload["shape"],
            "available": False,
            "reason": LSS_UNAVAILABLE["shape"],
        },
        "unavailable": {tool: LSS_UNAVAILABLE[tool] for tool in TERM_TOOLS},
    }


def session_availability() -> dict[str, str]:
    """The reason for each tool of the whole session, for the editor state of a SuperLSS."""
    return {tool: LSS_UNAVAILABLE[tool] for tool in SESSION_TOOLS}
