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

import numpy as np

from superglm.distributional.weights import resolve_likelihood_weights
from superglm.editor.errors import EditorValueError
from superglm.editor.lss.view import LSSModelView, is_lss

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


def unavailable_evidence(model: object, tool: str, **fields: Any) -> dict[str, Any] | None:
    """The calm payload an evidence route answers for a tool a SuperLSS does not offer, else None.

    ``fields`` (the revision, request sequence and source) are echoed so the
    page files the answer like any other; ``error`` is the tool's sentence.
    """
    if not is_lss(model):
        return None
    return {"available": False, **fields, "error": LSS_UNAVAILABLE[tool]}


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


NO_FIT_SETTINGS = (
    "This model has no recorded fit settings, so the editor cannot refit it. Fit it again with "
    "this version of superglm, then open it in the editor."
)
OFFSETS_NOT_GIVEN = (
    "This model was fitted with offsets for {names}, and the editor has none for {them}, so "
    "Refit and Final fit would leave {them} out. Open the editor with "
    "train_data=(X, y, sample_weight, offsets), with offsets a dict such as {{'{first}': ...}}."
)
WEIGHTS_NOT_GIVEN = (
    "This model was fitted with sample weights, and the editor has none, so Refit and Final fit "
    "would give every row weight 1. Open the editor with "
    "train_data=(X, y, sample_weight, offsets)."
)


WEIGHTS_DIFFER = (
    "This model was fitted with other sample weights than the editor's training rows carry, so "
    "Refit and Final fit would fit a different model. Open the editor with the rows and "
    "weights the model was fitted on: train_data=(X, y, sample_weight, offsets)."
)


def refit_reason(model: Any, sample_weight: Any, offsets: Any, y: Any = None) -> str | None:
    """Why a SuperLSS (or the editor's view of one) cannot be refitted on these rows; None if it can.

    A replay needs the settings the fit recorded, which a model saved by an
    earlier superglm lacks, and the offsets and weights the fit was given: rows
    without them would refit a different model, silently. With the response
    ``y``, the weights must also resolve to the fit's own.
    """
    lss = model.lss if isinstance(model, LSSModelView) else model
    if lss._fit_settings is None:
        return NO_FIT_SETTINGS
    state = lss._model.fit_state
    missing = [
        name
        for name, semantics in state.null_model.offset_semantics.items()
        if semantics != "zero_offset" and (offsets is None or name not in offsets)
    ]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        them = "it" if len(missing) == 1 else "them"
        return OFFSETS_NOT_GIVEN.format(names=names, them=them, first=missing[0])
    provenance = state.weight_provenance
    if sample_weight is None and not (provenance.all_unit and provenance.dropped_count == 0):
        return WEIGHTS_NOT_GIVEN
    if y is not None and _weights_differ(state, sample_weight, np.asarray(y).size):
        return WEIGHTS_DIFFER
    return None


def session_refit_reason(session: Any) -> str | None:
    """Why the editor cannot refit its SuperLSS on its training rows; None for a SuperGLM."""
    if not is_lss(session.model):
        return None
    train = session._evaluation_data.get("train")
    if train is None:
        return refit_reason(session.model, None, None)
    return refit_reason(session.model, train.sample_weight, train.offset, train.y)


def _weights_differ(state: Any, sample_weight: Any, n_rows: int) -> bool:
    """Whether ``sample_weight`` on ``n_rows`` rows resolves to another root than the fit's.

    The fit records the digest of its resolved weights, so one pass over the
    rows' weights shows whether a replay would fit the same likelihood.
    """
    resolved = resolve_likelihood_weights(
        sample_weight, n_observations=n_rows, contract=state.weight_contract
    )
    return resolved.provenance.root_digest != state.weight_provenance.root_digest


def session_availability(session: Any) -> dict[str, str]:
    """The reason for each tool of the whole session that is off, for a SuperLSS's editor state."""
    off = {tool: LSS_UNAVAILABLE[tool] for tool in SESSION_TOOLS}
    reason = session_refit_reason(session)
    if reason is not None:
        off["refit"] = off["final_fit"] = reason
    return off
