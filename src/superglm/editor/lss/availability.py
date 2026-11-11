"""The editor tools a SuperLSS model does not offer, and the sentence that says why.

Each tool refuses with its own sentence (``require_available``) and its payload
lists the same sentence beside it, so a hover text and the error a direct call
raises never differ. The structural tools (collapse, ungroup, reference, knots
basis and shaped ranges) work on a SuperLSS term as they do on a SuperGLM one
and are absent here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.offsets import sources_of, sum_offset_sources
from superglm.distributional.weights import LikelihoodWeightError, resolve_likelihood_weights
from superglm.editor.errors import EditorValueError
from superglm.editor.evaluation import EvaluationDataset, coerce_dataset
from superglm.editor.lss.view import LSSModelView, fit_keeps_its_rows, is_lss, rows_describe_fit

LSS_UNAVAILABLE: Mapping[str, str] = MappingProxyType(
    {
        "special": "Special levels are not available for SuperLSS terms.",
        "free_levels": "The free-level comparison is not available for SuperLSS terms.",
        "unsmoothed": "The unsmoothed line is not available for SuperLSS terms.",
        "unseen": "New-level choices are not available for SuperLSS terms.",
        "refit_offset": "Refitting with an edited offset is not available for SuperLSS models.",
        "profile_distribution": "SuperLSS models estimate their distribution parameters in the "
        "fit itself.",
        "summary": "The coefficient summary is not available here for SuperLSS models; call "
        "model.summary() in Python.",
        "metrics": "Compare SuperLSS fits in the Validation tab.",
        "export_structure": "This export is for SuperGLM models; export a SuperLSS model as "
        "joblib.",
        "save_session": "Session files hold SuperGLM models. Export the edited SuperLSS model as "
        "joblib; it keeps its history.",
    }
)

# The tools tied to one term, listed on that term's payload.
TERM_TOOLS = ("special", "free_levels", "unsmoothed", "unseen")
# The tools of the whole session, listed on the editor state.
SESSION_TOOLS = (
    "refit_offset",
    "profile_distribution",
    "summary",
    "metrics",
    "export_structure",
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
    """``term_payload`` with each term tool's reason listed."""
    return {
        **term_payload,
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


_UNIT_ROUNDOFF = 2.0**-53

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
    column_only = {
        p.name
        for p in state.predictor_templates
        if sources_of(p) and all(isinstance(s, str) for s in sources_of(p))
    }
    missing = [
        name
        for name, semantics in state.null_model.offset_semantics.items()
        if semantics != "zero_offset"
        and name not in column_only
        and (offsets is None or name not in offsets)
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


TRAINING_ROWS_FOLLOWED = (
    "The model was refitted in place on other data than the editor's training data, which "
    "describe an earlier fit. The editor now takes the {n} rows the refit kept as its "
    "training data: its curves, training scores, Refit and Final fit read them."
)
TRAINING_ROWS_LOST = (
    "The model was refitted in place on other data than the editor's training data, and the "
    "refit kept none of its rows (retain_rows=False), so the editor has no training rows for "
    "it. Open the editor again with train_data=(X, y, sample_weight, offsets), the refit's rows."
)


def follow_training_rows(
    lss: Any, train: EvaluationDataset
) -> tuple[EvaluationDataset, str | None]:
    """The training split for the fit in force after a refit or repair in place, and the
    notice to give, ``None`` when ``train`` already described it (``rows_describe_fit``).

    Where the fit keeps its rows, the split is those rows, whatever changed: its frame,
    responses, weights and offsets. Otherwise ``train`` stays while it describes the fit.
    Raises ``EditorValueError`` when it does not, since nothing else gives the rows back.
    """
    described = rows_describe_fit(lss, train)
    notice = None if described else TRAINING_ROWS_FOLLOWED
    if not fit_keeps_its_rows(lss):
        if described:
            return train, None
        raise EditorValueError(TRAINING_ROWS_LOST)
    state = lss._require_fitted().fit_state
    rows = state.retained_rows
    assert rows is not None
    offsets = _array_offsets(state.predictor_templates, rows.likelihood_weights.input_positions)
    weights = rows.likelihood_weights
    followed = coerce_dataset(
        "train",
        (
            lss._term_training_frame(None),
            np.asarray(rows.response),
            # No weights only where the fit had none: a fit that dropped zero-weight rows was
            # given weights, as a session on its rows is (``refit_reason``).
            None
            if weights.provenance.all_unit and weights.provenance.dropped_count == 0
            else np.asarray(weights.values),
            offsets or None,
        ),
        family=lss.family,
        weight_semantics=lss.weight_semantics,
        offset_parameters=lss.parameter_names_,
    )
    assert followed is not None
    return followed, None if notice is None else notice.format(n=followed.n_obs)


def _array_offsets(predictors: Any, positions: NDArray[np.intp]) -> dict[str, NDArray]:
    """Each predictor's declared array offsets, summed at the kept ``positions``: the part of
    its offset a split supplies (``shape_prediction_offsets``), since its columns are read
    from the frame. Never the fit's total, whose columns would then count twice."""
    found: dict[str, NDArray] = {}
    for predictor in predictors:
        arrays = [source for source in sources_of(predictor) if not isinstance(source, str)]
        if arrays:
            values = (getattr(source, "values", source) for source in arrays)
            found[predictor.name] = sum_offset_sources(
                {predictor.name: tuple(np.asarray(value) for value in values)}, positions
            )[predictor.name]
    return found


def session_refit_reason(session: Any) -> str | None:
    """Why the editor cannot refit its SuperLSS on its training rows; None for a SuperGLM."""
    if not is_lss(session.model):
        return None
    train = session._evaluation_data.get("train")
    if train is None:
        return refit_reason(session.model, None, None)
    view = session.model
    if not isinstance(view, LSSModelView):
        return refit_reason(view, train.sample_weight, train.offset, train.y)
    if view._refit_reason is None or view._refit_reason[0] is not train:
        view._refit_reason = (train, refit_reason(view, train.sample_weight, train.offset, train.y))
    return view._refit_reason[1]


def _weights_differ(state: Any, sample_weight: Any, n_rows: int) -> bool:
    """Whether ``sample_weight`` on ``n_rows`` rows resolves to other kept weights than the fit's.

    On the frame the fit was given, the weights are compared by the digest the
    fit records of every row's weight, in order. On only the rows it kept, a fit
    that kept its rows is compared row by row and exactly, since the same inputs
    resolve to the same kept weights. Otherwise the kept weights are compared
    through the summary the fit records: their counts, extremes and unit flag
    exactly, and their sum and log sum to the rounding of one summation, since a
    sum evaluated afresh (perhaps on another machine) may differ in its last
    bits. The summary cannot see weights moved to other rows.
    """
    try:
        resolved = resolve_likelihood_weights(
            sample_weight, n_observations=n_rows, contract=state.weight_contract
        )
    except LikelihoodWeightError:
        # Weights the fit's contract refuses are not the weights it was fitted with.
        return True
    fit = state.weight_provenance
    if n_rows == fit.original_count:
        return resolved.provenance.root_digest != fit.root_digest
    if state.retained_rows is not None:
        return not np.array_equal(resolved.values, state.retained_rows.likelihood_weights.values)
    got = resolved.provenance
    exact = ("retained_count", "likelihood_count", "min_weight", "max_weight", "all_unit")
    if any(getattr(got, name) != getattr(fit, name) for name in exact):
        return True
    n = fit.retained_count
    largest_log = max(abs(math.log(fit.min_weight)), abs(math.log(fit.max_weight)))
    sum_tolerance = 2.0 * n * _UNIT_ROUNDOFF * fit.weight_sum
    log_tolerance = 2.0 * n * _UNIT_ROUNDOFF * n * largest_log
    return (
        abs(got.weight_sum - fit.weight_sum) > sum_tolerance
        or abs(got.log_weight_sum - fit.log_weight_sum) > log_tolerance
    )


def session_availability(session: Any) -> dict[str, str]:
    """The reason for each tool of the whole session that is off, for a SuperLSS's editor state."""
    off = {tool: LSS_UNAVAILABLE[tool] for tool in SESSION_TOOLS}
    reason = session_refit_reason(session)
    if reason is not None:
        off["refit"] = off["final_fit"] = reason
    return off
