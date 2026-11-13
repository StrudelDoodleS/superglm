"""What the Diagnose tab shows: how each fit's smoothing search went.

Everything here reads what a fit already recorded, so opening the tab refits
nothing. Only a fold's held-out rows need one prediction each.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm._lambda_box import LambdaBox, fitted_lambda_box
from superglm.diagnostics.convergence import (
    coefficient_nonconvergence_message,
    lss_nonconvergence_message,
    reml_nonconvergence_message,
)
from superglm.diagnostics.fit_report import phase_label, status_sentence
from superglm.editor.lss.view import is_lss
from superglm.model.fit_state import fitted_lambda2
from superglm.model.reml_setup import collect_reml_groups
from superglm.model_selection import _fingerprint_columns, _take_offset, fold_adapter
from superglm.reml.penalty_algebra import build_penalty_components

_LOGGER = logging.getLogger(__name__)

# EDF this close to the unpenalized part of a term reads as that shape.
SHAPE_MARGIN = 0.05
# A round within this much REML of the end has settled.
SETTLED_REML = 0.01
WORST_ROWS = 10
_LIMIT_OUTCOMES = frozenset({"upper_bound", "unresolved_cap", "exact_face"})
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_FIXED_STOPS = frozenset({"fixed_only", "fixed_lambdas"})
# A finding about these affects the numbers a user reads; the rest are about how the search went.
_NUMBER_IMPACTS = frozenset({"fit_validity", "fit_reliability", "inference_reliability"})
# The tab's status of a fit, from a SuperLSS report's status code.
_LSS_STATUS = {
    "converged_certified": "converged",
    "converged_exact_face": "converged",
    "converged_uncertified": "loose",
    "not_converged": "not_converged",
    "fixed_fit": "fixed",
}
_SEARCH_WORDS = {
    "efs": "Quick smoothing updates only",
    "efs+finish": "Quick smoothing updates, then Newton steps to an optimum",
    "efs+newton": "A few quick smoothing updates, then Newton steps",
}
_GLM_CONVERGED = "Converged. The smoothing search met its stopping test."
_GLM_EDITED = "The coefficients were edited by hand after fitting, so they are not the fit's own."
# A quoted term of a finding: ``'mean:power'``, or one of its penalties, ``'mean:power#wiggle'``.
_QUOTED_TERM = re.compile(r"'([^':#]+:[^'#]+)(?:#[^']*)?'")
_NO_SEARCH = "no smoothing search ran"
_STOP_TEXT = {
    **dict.fromkeys(_FIXED_STOPS, _NO_SEARCH),
    "lambda_change": "the smoothing parameters stopped changing",
    "lambda_tolerance": "the smoothing parameters changed by less than the set tolerance",
    "objective_plateau": "the REML score stopped improving by more than the stopping rule's threshold",
    "practical_plateau": "the REML score stopped improving by more than the quick stopping rule's "
    "threshold",
    "lambda_cap_unresolved": "a term reached the largest smoothing allowed and the search could "
    "not tell whether it should go higher",
    "endpoint_revalidation_failed": "a term at the largest smoothing allowed did not pass the "
    "check that it belongs there",
    "objective_rejected": "no smoothing step improved the REML score, so the search stopped "
    "before it passed its stopping test",
    "line_search_failed": "no smoothing step improved the REML score",
    "line_search_stalled": "no smoothing step improved the REML score",
    "coefficient_not_converged": "the coefficients did not settle",
    "stationary": "the search reached an optimum: the REML score no longer improves in any "
    "direction",
    "active_set_stationary": "the search reached an optimum: the REML score no longer improves "
    "in any direction",
    "gradient_unresolved": "the search could not tell the slope of the REML score from zero",
    "derivative_unavailable": "a slope the search needs could not be computed",
    "score_objective_tolerance": "the REML score is flat to within the set tolerance",
    "newton_step_tolerance": "the next step would change the smoothing parameters by less than "
    "the set tolerance",
    "start_objective_tolerance": "the starting smoothing parameters were already at the optimum",
    "converged_at_precision": "the REML score cannot be improved at the computer's numerical "
    "precision",
    "bootstrap_uncertified": "no coefficient fit at the starting smoothing parameters settled",
    "coefficients_revised": "the coefficients were edited after fitting",
}


def _stop_text(reason: str | None, rounds: int | None = None) -> str:
    """The stop in words after "Stopped because"; ``rounds`` names the budget a cap ran out of."""
    if reason is None:
        return _NO_SEARCH
    if reason in {"max_iterations", "max_reml_iter"}:
        count = (
            "its smoothing rounds"
            if rounds is None
            else f"{rounds} smoothing round{'' if rounds == 1 else 's'}"
        )
        return f"it reached its limit of {count} (max_reml_iter) before settling"
    return _STOP_TEXT.get(reason, reason.replace("_", " "))


def _search_text(outer: str) -> str | None:
    """Which smoothing search ran, in words; None if unknown."""
    return _SEARCH_WORDS.get(outer)


def _term_cell(
    edf: float, lam: float | None, null_space: int, *, at_limit: bool = False
) -> dict[str, Any]:
    """One term of one fit: its EDF, smoothing and the shape they leave it.

    ``null_space`` is what the term's penalty leaves unsmoothed: one dimension is
    a straight line, more is a simplest shape of that size. Zero claims no shape
    short of the term being smoothed away.
    """
    if lam is None:
        shape = "unsmoothed"
    elif edf <= SHAPE_MARGIN:
        shape = "flat"
    elif null_space > 0 and edf - null_space <= SHAPE_MARGIN:
        shape = "straight" if null_space == 1 else "simplest"
    else:
        shape = "curve"
    return {
        "edf": float(edf),
        "lambda": None if lam is None else float(lam),
        "shape": shape,
        "at_limit": at_limit,
    }


def _lss_cell(edf: float, components: Sequence[Any]) -> dict[str, Any]:
    """A SuperLSS term's cell from its smoothing components, reading the fit's own outcomes.

    What several penalties leave together is not recorded per term, so such a
    term claims no simplest shape.
    """
    if not components:
        return _term_cell(edf, None, 0)
    return _term_cell(
        edf,
        max(component.final_lambda for component in components),
        (components[0].null_space_dimension or 0) if len(components) == 1 else 0,
        at_limit=any(component.outcome in _LIMIT_OUTCOMES for component in components),
    )


def _lss_trace(lss) -> dict[str, Any]:
    fitted = lss._require_fitted()
    report = lss.diagnose()
    profile = report.profile
    smoothing = fitted.smoothing
    components: dict[str, list[Any]] = {}
    for component in profile.smoothing_components:
        components.setdefault(f"{component.predictor}:{component.term}", []).append(component)
    terms = {
        key: _lss_cell(edf, components.get(key, ()))
        for key, edf in fitted.inference.term_edf.items()
    }
    findings = sorted(
        report.findings,
        key=lambda found: _SEVERITY_ORDER.get(found.severity, len(_SEVERITY_ORDER)),
    )
    status = _LSS_STATUS[report.fit_status]
    outer = None if smoothing is None else getattr(smoothing.config, "outer", None)
    return {
        "status": status,
        "verdict": status_sentence(report.fit_status),
        "explanation": lss_nonconvergence_message(
            fitted.fitted_result,
            None if smoothing is None else smoothing.convergence_reason,
            getattr(fitted.result, "convergence_reason", None),
        )
        if status == "not_converged"
        else None,
        "search": None if outer is None else _search_text(outer),
        "rows": profile.n_observations,
        "rounds": fitted.fitted_result.n_smoothing_iter,
        "stop": None if smoothing is None else smoothing.convergence_reason,
        "converged": bool(fitted.fitted_result.converged),
        "seconds": profile.fit_seconds,
        "path": []
        if smoothing is None
        else [smoothing.initial_objective, *(item.objective_after for item in smoothing.history)],
        "terms": terms,
        "phases": [
            {
                "name": phase_label(phase.name),
                "seconds": phase.seconds,
                "share": phase.fit_share,
            }
            for phase in sorted(profile.phases, key=lambda phase: -phase.seconds)
            if phase.seconds > 0.0
        ],
        "findings": [
            {
                "severity": found.severity,
                "headline": found.headline,
                "observed": found.observed,
                "question": found.actions[0].question if found.actions else None,
                "affects_numbers": bool(_NUMBER_IMPACTS.intersection(found.impacts)),
            }
            for found in findings
        ],
    }


def _glm_penalties(model) -> list[tuple[Any, float]] | None:
    """Each smoothing penalty of a fit with its value, searched or held fixed.

    None where a fit with no search released the design that says which terms
    it penalised. A group-wide key sets every penalty of its group that is not
    listed by its full name, as it does in the solver.
    """
    reml = getattr(model, "_reml_result", None)
    if reml is not None:
        penalties = getattr(model, "_reml_penalties", None) or reml.reml_penalties or ()
        return [(item, float(reml.lambdas[item.name])) for item in penalties]
    if model._dm is None:
        return None
    matrices = model._dm.group_matrices
    fixed = fitted_lambda2(model)
    return [
        (
            item,
            float(
                fixed.get(item.name, fixed.get(item.group_name, 0.0))
                if isinstance(fixed, Mapping)
                else fixed
            ),
        )
        for item in build_penalty_components(matrices, collect_reml_groups(model._groups, matrices))
    ]


def _glm_cell(
    edf: float,
    width: int,
    penalties: Sequence[tuple[Any, float]],
    *,
    searched: bool,
    box: LambdaBox | None = None,
) -> dict[str, Any]:
    """A SuperGLM term's cell. Several penalties claim no simplest shape, as on a SuperLSS."""
    if not penalties:
        return _term_cell(edf, None, 0)
    lam = max(value for _, value in penalties)
    return _term_cell(
        edf,
        lam,
        width - round(penalties[0][0].rank) if len(penalties) == 1 else 0,
        at_limit=searched and (box or LambdaBox()).side(lam) == "ceiling",
    )


def _glm_terms(model, *, searched: bool) -> dict[Hashable, dict[str, Any]]:
    """Every term's cell, main effects and interactions, by which term owns each group."""
    group_edf = model._group_edf
    owner = {group.name: group.feature_name for group in model._groups}
    width: dict[Hashable, int] = {}
    edf: dict[Hashable, float] = {}
    for group in model._groups:
        name = group.feature_name
        width[name] = width.get(name, 0) + group.end - group.start
        edf[name] = edf.get(name, 0.0) + group_edf.get(group.name, 0.0)
    recorded = _glm_penalties(model)
    if recorded is None:
        return {
            name: {
                "edf": float(edf[name]),
                "lambda": None,
                "shape": "unrecorded",
                "at_limit": False,
            }
            for name in width
        }
    penalties: dict[Hashable, list[tuple[Any, float]]] = {}
    for penalty, value in recorded:
        penalties.setdefault(owner[penalty.group_name], []).append((penalty, value))
    box = fitted_lambda_box(model)
    return {
        name: _glm_cell(edf[name], width[name], penalties.get(name, ()), searched=searched, box=box)
        for name in width
    }


def _glm_verdict(model, reml) -> dict[str, Any]:
    """A SuperGLM fit's status, its sentence and, when it did not converge, the full statement."""
    if reml is not None and reml.termination_reason == "coefficients_revised":
        return {"status": "edited", "verdict": _GLM_EDITED, "explanation": None}
    message = (
        coefficient_nonconvergence_message(model._result)
        if reml is None
        else reml_nonconvergence_message(reml)
    )
    searched = reml is not None and reml.termination_reason not in _FIXED_STOPS
    if message is not None or not (
        bool(model._result.converged) and (reml is None or bool(reml.converged))
    ):
        return {
            "status": "not_converged",
            "verdict": status_sentence("not_converged"),
            "explanation": message,
        }
    if searched:
        return {"status": "converged", "verdict": _GLM_CONVERGED, "explanation": None}
    return {"status": "fixed", "verdict": status_sentence("fixed_fit"), "explanation": None}


def _glm_trace(model) -> dict[str, Any]:
    reml = getattr(model, "_reml_result", None)
    terms = _glm_terms(
        model, searched=reml is not None and reml.termination_reason not in _FIXED_STOPS
    )
    return {
        **_glm_verdict(model, reml),
        "search": None,
        "rows": getattr(getattr(model, "_fit_stats", None), "n_obs", None),
        "rounds": 0 if reml is None else int(reml.n_reml_iter),
        "stop": None if reml is None else reml.termination_reason,
        "converged": bool(model._result.converged) and (reml is None or bool(reml.converged)),
        "seconds": None,
        "path": [] if reml is None else [float(value) for value in reml.objective_history or ()],
        "terms": terms,
        "phases": [],
        "findings": [],
    }


def fit_trace(
    model,
    *,
    fit_id: str,
    label: str,
    n_rows: int | None = None,
    seconds: float | None = None,
) -> dict[str, Any]:
    """How the search behind one fit went, as the tab draws it."""
    trace = _lss_trace(getattr(model, "lss", model)) if is_lss(model) else _glm_trace(model)
    path = trace["path"]
    final = path[-1] if path else None
    trace.update(
        id=fit_id,
        label=label,
        stop_text=_stop_text(trace["stop"], trace["rounds"]),
        rows=trace["rows"] if n_rows is None else n_rows,
        seconds=trace["seconds"] if seconds is None else seconds,
        late_gain=path[len(path) // 2] - final if len(path) >= 2 else None,
        settled_round=next(
            (index for index, value in enumerate(path) if abs(value - final) <= SETTLED_REML),
            None,
        ),
    )
    return trace


def _plain(value: Any) -> Any:
    """A column's value as the page shows it."""
    item = value.item() if isinstance(value, np.generic) else value
    return item if isinstance(item, int | float | str | bool) or item is None else str(item)


def worst_rows(
    adapter,
    model,
    X,
    y: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None,
    offset,
    row_numbers: NDArray[np.intp],
    columns: Sequence[str],
) -> dict[str, Any]:
    """The held-out rows that cost ``model`` the most, worst first, as the page lists them.

    A row of zero weight was dropped by the fit and its scores, so it is not ranked.
    """
    frame = as_eager_frame(X)
    y = np.asarray(y, dtype=np.float64)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    kept = np.arange(len(y)) if weights is None else np.flatnonzero(weights > 0.0)
    costs = adapter.row_costs(
        model,
        frame.take_rows(kept),
        y[kept],
        None if weights is None else weights[kept],
        _take_offset(offset, kept),
    )
    ranked = np.argsort(-costs)[:WORST_ROWS]
    order = kept[ranked]
    predicted = adapter.predict(model, frame.take_rows(order), _take_offset(offset, order))
    values = {name: frame.column_array(name)[order] for name in columns}
    return {
        "columns": list(columns),
        "rows": [
            {
                "row": int(row_numbers[position]),
                "cost": float(costs[rank]) if np.isfinite(costs[rank]) else None,
                "response": float(y[position]),
                "predicted": float(predicted[index]),
                "weight": None if weights is None else float(weights[position]),
                "values": [_plain(values[name][index]) for name in columns],
            }
            for index, (rank, position) in enumerate(zip(ranked, order, strict=True))
        ],
    }


def supplied_fold_traces(cv, rows) -> list[dict[str, Any]]:
    """A trace per fold of a result that kept its fold models; held-out rows when ``rows`` match.

    A fold that failed kept no model, so it has no trace. A result that kept no
    fold indices says neither a fold's size nor which rows it held out. A fold
    that cannot be read is logged and left out, so the others still show.
    """
    traces = []
    times = cv.fold_scores["fit_time_s"] if "fit_time_s" in cv.fold_scores else None
    for index, model in enumerate(cv.estimators):
        if model is None:
            continue
        train, test = (None, None) if cv.fold_indices is None else cv.fold_indices[index]
        try:
            trace = fit_trace(
                model,
                fit_id=f"fold-{index + 1}",
                label=f"Fold {index + 1}",
                n_rows=None if train is None else len(train),
                seconds=None if times is None else float(times.iloc[index]),
            )
            if rows is not None and test is not None:
                trace["worst_rows"] = held_out_worst_rows(model, rows, np.asarray(test))
        except Exception:
            _LOGGER.warning("Fold %d has no Diagnose trace.", index + 1, exc_info=True)
            continue
        traces.append(trace)
    return traces


def held_out_worst_rows(model, rows, test: NDArray[np.intp]) -> dict[str, Any]:
    """:func:`worst_rows` over the ``test`` positions of ``rows``."""
    frame = as_eager_frame(rows.X).take_rows(test)
    weight = None if rows.sample_weight is None else np.asarray(rows.sample_weight)[test]
    return worst_rows(
        fold_adapter(model),
        model,
        frame,
        np.asarray(rows.y, dtype=np.float64)[test],
        weight,
        _take_offset(rows.offset, test),
        test,
        _fingerprint_columns(model, as_eager_frame(rows.X)),
    )


def _name_terms(text: str | None, label_of: Mapping[str, str]) -> str | None:
    """``text`` with each quoted term key said as the editor labels it."""
    if text is None:
        return None
    return _QUOTED_TERM.sub(
        lambda match: f"'{label_of[match.group(1)]}'" if match.group(1) in label_of else match[0],
        text,
    )


def all_rows_worst_rows(model, rows) -> dict[str, Any] | None:
    """The training rows that cost the fit on every row the most; None without the rows."""
    if rows is None:
        return None
    try:
        return held_out_worst_rows(getattr(model, "lss", model), rows, np.arange(rows.n_obs))
    except Exception:
        _LOGGER.warning("The fit on all rows has no costly rows to list.", exc_info=True)
        return None


def diagnose_tab_payload(
    model,
    *,
    folds: Sequence[dict[str, Any]],
    model_rows: dict[str, Any] | None = None,
    source: str,
    display: Callable[[str], Any],
    model_revision: int,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """The tab: this model's fit, then its folds, and every term they share."""
    whole = fit_trace(model, fit_id="model", label="All rows")
    if model_rows is not None:
        whole["worst_rows"] = model_rows
    fits = [whole, *folds]
    keys = dict.fromkeys(key for fit in fits for key in fit["terms"])
    shown = {key: display(key) for key in keys}
    label_of = {key: item.label for key, item in shown.items()}
    for fit in fits:
        for found in fit["findings"]:
            for field in ("headline", "observed", "question"):
                found[field] = _name_terms(found[field], label_of)
    terms = [
        {
            "name": key,
            "label": shown[key].label,
            "short": shown[key].short,
            "group": shown[key].group,
            "role": shown[key].role,
        }
        for key in sorted(keys, key=lambda key: shown[key].role == "spread")
    ]
    payload = {
        "report": "diagnose",
        "title": "Diagnose",
        "available": True,
        "note": source,
        "model_revision": model_revision,
        "fits": fits,
        "terms": terms,
        "cost_label": fold_adapter(getattr(model, "lss", model)).row_cost_label,
    }
    if request_sequence is not None:
        payload["request_sequence"] = request_sequence
    return payload
