"""What the Diagnose tab shows: how each fit's smoothing search went.

Everything here reads what a fit already recorded, so opening the tab refits
nothing. Only a fold's held-out rows need one prediction each.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import as_eager_frame
from superglm.editor.lss.view import is_lss
from superglm.model_selection import _fingerprint_columns, _take_offset, fold_adapter

# A smoothing parameter this large leaves a term at its unpenalized shape.
LIMIT_LAMBDA = 1.0e6
# EDF this close to the unpenalized part of a term reads as that shape.
SHAPE_MARGIN = 0.05
# A round within this much REML of the end has settled.
SETTLED_REML = 0.01
WORST_ROWS = 10
_LIMIT_OUTCOMES = frozenset({"upper_bound", "unresolved_cap", "exact_face"})
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_NO_SMOOTHING = "there was no smoothing to choose"
_STOP_TEXT = {
    "fixed_only": _NO_SMOOTHING,
    "fixed_lambdas": _NO_SMOOTHING,
    "lambda_change": "the smoothing stopped moving",
    "objective_plateau": "REML stopped improving",
    "practical_plateau": "REML stopped improving",
    "lambda_cap_unresolved": "a term reached the smoothing limit unresolved",
    "endpoint_revalidation_failed": "a term at the limit failed its recheck",
    "max_iterations": "it ran out of rounds",
    "max_reml_iter": "it ran out of rounds",
    "objective_rejected": "REML would not improve further",
    "coefficient_not_converged": "the coefficients did not converge",
    "stationary": "the REML gradient reached zero",
    "active_set_stationary": "the REML gradient reached zero",
    "gradient_unresolved": "the REML gradient could not be resolved",
    "derivative_unavailable": "a derivative was unavailable",
    "score_objective_tolerance": "the REML gradient reached its tolerance",
    "start_objective_tolerance": "the start was already optimal",
    "converged_at_precision": "REML reached working precision",
}


def _stop_text(reason: str | None) -> str:
    if reason is None:
        return _NO_SMOOTHING
    return _STOP_TEXT.get(reason, reason.replace("_", " "))


def _term_cell(
    edf: float, lam: float | None, null_space: int, *, at_cap: bool = False
) -> dict[str, Any]:
    """One term of one fit: its EDF, smoothing and the shape they leave it."""
    if lam is None:
        shape = "unsmoothed"
    elif edf <= SHAPE_MARGIN:
        shape = "flat"
    elif edf - null_space <= SHAPE_MARGIN:
        shape = "straight"
    else:
        shape = "curve"
    return {
        "edf": float(edf),
        "lambda": None if lam is None else float(lam),
        "shape": shape,
        "at_limit": at_cap or (lam is not None and lam >= LIMIT_LAMBDA),
    }


def _lss_trace(lss) -> dict[str, Any]:
    fitted = lss._require_fitted()
    report = lss.diagnose()
    profile = report.profile
    smoothing = fitted.smoothing
    components: dict[str, Any] = {}
    for component in profile.smoothing_components:
        key = f"{component.predictor}:{component.term}"
        held = components.get(key)
        if held is None or component.final_lambda > held.final_lambda:
            components[key] = component
    terms = {}
    for key, edf in fitted.inference.term_edf.items():
        component = components.get(key)
        terms[key] = (
            _term_cell(edf, None, 0)
            if component is None
            else _term_cell(
                edf,
                component.final_lambda,
                1 if component.null_space_dimension is None else component.null_space_dimension,
                at_cap=component.outcome in _LIMIT_OUTCOMES,
            )
        )
    findings = sorted(report.findings, key=lambda found: _SEVERITY_ORDER[found.severity])
    return {
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
                "name": phase.name.replace("_", " ").capitalize(),
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
            }
            for found in findings
        ],
    }


def _glm_lambda(lambdas: Mapping[str, float], name: str) -> float | None:
    """The term's largest smoothing parameter; a term with several names each ``name:part``."""
    own = [value for key, value in lambdas.items() if key == name or key.startswith(f"{name}:")]
    return max(own) if own else None


def _glm_trace(model) -> dict[str, Any]:
    reml = getattr(model, "_reml_result", None)
    lambdas = {} if reml is None else reml.lambdas
    terms = {}
    for name in model._specs:
        edf = model.term_inference(name, with_se=False).edf
        terms[name] = _term_cell(0.0 if edf is None else edf, _glm_lambda(lambdas, name), 1)
    return {
        "rows": None,
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
        stop_text=_stop_text(trace["stop"]),
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
) -> list[dict[str, Any]]:
    """The held-out rows that cost ``model`` the most, worst first, as the page lists them."""
    frame = as_eager_frame(X)
    costs = adapter.row_costs(model, frame, y, sample_weight, offset)
    order = np.argsort(-costs)[:WORST_ROWS]
    predicted = adapter.predict(model, frame.take_rows(order), _take_offset(offset, order))
    values = {name: frame.column_array(name)[order] for name in columns}
    return [
        {
            "row": int(row_numbers[position]),
            "cost": float(costs[position]) if np.isfinite(costs[position]) else None,
            "response": float(y[position]),
            "predicted": float(predicted[index]),
            "weight": None if sample_weight is None else float(sample_weight[position]),
            "values": [_plain(values[name][index]) for name in columns],
        }
        for index, position in enumerate(order)
    ]


def supplied_fold_traces(cv, rows) -> list[dict[str, Any]]:
    """A trace per fold of a result that kept its fold models; held-out rows when ``rows`` match.

    A fold that failed kept no model, so it has no trace.
    """
    traces = []
    times = cv.fold_scores["fit_time_s"] if "fit_time_s" in cv.fold_scores else None
    for index, ((train, test), model) in enumerate(
        zip(cv.fold_indices, cv.estimators, strict=True)
    ):
        if model is None:
            continue
        trace = fit_trace(
            model,
            fit_id=f"fold-{index + 1}",
            label=f"Fold {index + 1}",
            n_rows=len(train),
            seconds=None if times is None else float(times.iloc[index]),
        )
        if rows is not None:
            trace["worst_rows"] = held_out_worst_rows(model, rows, np.asarray(test))
        traces.append(trace)
    return traces


def held_out_worst_rows(model, rows, test: NDArray[np.intp]) -> list[dict[str, Any]]:
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


def diagnose_tab_payload(
    model,
    *,
    folds: Sequence[dict[str, Any]],
    source: str,
    display: Callable[[str], Any],
    columns: Sequence[str],
    model_revision: int,
    request_sequence: int | None = None,
) -> dict[str, Any]:
    """The tab: this model's fit, then its folds, and every term they share."""
    fits = [fit_trace(model, fit_id="model", label="This model"), *folds]
    keys = dict.fromkeys(key for fit in fits for key in fit["terms"])
    shown = {key: display(key) for key in keys}
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
        "columns": list(columns),
        "cost_label": fold_adapter(getattr(model, "lss", model)).row_cost_label,
    }
    if request_sequence is not None:
        payload["request_sequence"] = request_sequence
    return payload
