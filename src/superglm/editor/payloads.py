"""JSON state payloads for the editor frontend."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from superglm.editor.controls import CONTROL_HANDLE_TERM_TYPES
from superglm.editor.group_display import build_group_display
from superglm.editor.terms import term_from_inference

_MAX_INTERACTIVE_HANDLES = 420


def session_payload(
    session,
    control_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    # This is the sole shape consumed by the browser. It intentionally converts
    # link-scale session state to relativity-scale display arrays and includes
    # only JSON-safe primitives.
    payload: dict[str, dict[str, Any]] = {}
    for name, term in session.terms.items():
        x_values = list(range(term.size)) if term.x is None else [float(v) for v in term.x]
        weights = _term_weights(term)
        edit_delta = np.asarray(term.edited_log_effect - term.original_log_effect, dtype=np.float64)
        reference_log_effect = _reference_log_effect(session, name, term)
        ci_lower, ci_upper = _ci_payload(term, edit_delta)
        term_payload = {
            "kind": term.kind,
            "term_type": str(term.metadata.get("term_type", term.kind)),
            "x": x_values,
            "x_domain": _x_domain(x_values, pad_discrete=term.levels is not None),
            "y": [float(v) for v in np.exp(term.edited_log_effect)],
            "original_y": [float(v) for v in np.exp(reference_log_effect)],
            "previous_y": _previous_y(session, name, term),
            "controls": _controls_payload(
                session,
                name,
                term,
                None if control_counts is None else control_counts.get(name),
            ),
            "ci_lower_y": ci_lower,
            "ci_upper_y": ci_upper,
            "weights": weights,
            "exposure": _exposure_payload(term, x_values, weights),
            "handle_indices": _handle_indices(term).astype(int).tolist(),
            "impact": _impact_payload(term, weights, session.selection(name), reference_log_effect),
            "n_points": int(term.size),
            "levels": term.levels,
            "level_groups": _level_groups(session, name, term),
            "level_order_changed": _level_order_changed(session, name),
            "effective_df": _finite_float(term.metadata.get("edf")),
            "x_label": name,
            "y_label": "relativity",
            "title": name,
        }
        term_payload["group_display"] = build_group_display(term_payload)
        payload[name] = term_payload
    return payload


def history_payload(session) -> dict[str, Any]:
    """Return display metadata for the session's linear edit history."""
    active = _history_records_payload(getattr(session, "history", []))
    redo = _history_records_payload(getattr(session, "redo_stack", []))
    if active:
        active[0]["is_head"] = True
    return {"active": active, "redo": redo}


def _history_records_payload(records) -> list[dict[str, Any]]:
    parent_hash: str | None = None
    chronological = []
    for record in records:
        record_hash = _record_hash(record, parent_hash)
        chronological.append(
            {
                "hash": record_hash,
                "parent": parent_hash,
                "term": str(record.term),
                "operation": str(record.operation),
                "n_points": int(np.asarray(record.indices, dtype=np.intp).size),
                "params": _json_safe(record.params),
                "is_head": False,
            }
        )
        parent_hash = record_hash
    return list(reversed(chronological))


def _record_hash(record, parent_hash: str | None) -> str:
    data = {
        "parent": parent_hash,
        "term": str(record.term),
        "operation": str(record.operation),
        "indices": np.asarray(record.indices, dtype=np.intp).tolist(),
        "before": np.round(np.asarray(record.before, dtype=np.float64), 12).tolist(),
        "after": np.round(np.asarray(record.after, dtype=np.float64), 12).tolist(),
        "params": _json_safe(record.params),
    }
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:7]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def _previous_y(session, name: str, term) -> list[float] | None:
    for record in reversed(getattr(session, "history", [])):
        if record.term != name:
            continue
        previous = np.asarray(term.edited_log_effect, dtype=np.float64).copy()
        previous[np.asarray(record.indices, dtype=np.intp)] = np.asarray(
            record.before, dtype=np.float64
        )
        return [float(v) for v in np.exp(previous)]
    return None


def _level_order_changed(session, name: str) -> bool:
    checker = getattr(session, "level_order_changed", None)
    return bool(checker(name)) if checker is not None else False


def _level_groups(session, name: str, term) -> list[dict[str, Any]]:
    if term.levels is None:
        return []
    spec = getattr(session.model, "_specs", {}).get(name)
    grouping = getattr(spec, "_grouping", None)
    if grouping is None:
        return []
    level_to_index = {str(level): i for i, level in enumerate(term.levels)}
    groups = []
    for label in grouping.grouped_levels:
        levels = [str(level) for level in grouping.group_to_originals.get(label, [])]
        indices = [level_to_index[level] for level in levels if level in level_to_index]
        if len(indices) < 2:
            continue
        groups.append(
            {
                "label": str(label),
                "indices": indices,
                "levels": [levels[i] for i, level in enumerate(levels) if level in level_to_index],
            }
        )
    return groups


def _reference_log_effect(session, name: str, term) -> np.ndarray:
    reference_model = getattr(session, "reference_model", session.model)
    if reference_model is session.model:
        return np.asarray(term.original_log_effect, dtype=np.float64)
    try:
        ti = reference_model.term_inference(
            name,
            with_se=False,
            n_points=session.n_points,
            centering=session.centering,
        )
    except Exception:
        return np.asarray(term.original_log_effect, dtype=np.float64)
    reference = term_from_inference(ti)
    values = np.asarray(reference.original_log_effect, dtype=np.float64)
    if term.levels is not None and reference.levels is not None:
        by_level = {level: float(values[i]) for i, level in enumerate(reference.levels)}
        return np.array(
            [
                by_level.get(level, term.original_log_effect[i])
                for i, level in enumerate(term.levels)
            ]
        )
    if term.x is not None and reference.x is not None and values.size != term.size:
        return np.interp(
            np.asarray(term.x, dtype=np.float64),
            np.asarray(reference.x, dtype=np.float64),
            values,
        )
    if values.size == term.size:
        return values
    return np.asarray(term.original_log_effect, dtype=np.float64)


def _term_weights(term) -> list[float]:
    if term.weights is None:
        return [1.0] * term.size
    return [float(v) for v in np.asarray(term.weights, dtype=np.float64)]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _ci_payload(term, edit_delta: np.ndarray) -> tuple[list[float] | None, list[float] | None]:
    # CI bands come from the reference fit. After a manual edit, shift the band
    # by the edit delta for visual reference only; summaries suppress edited
    # inference separately.
    if term.ci_lower_log_effect is None or term.ci_upper_log_effect is None:
        return None, None
    ci_lower = np.asarray(term.ci_lower_log_effect, dtype=np.float64) + edit_delta
    ci_upper = np.asarray(term.ci_upper_log_effect, dtype=np.float64) + edit_delta
    return [float(v) for v in np.exp(ci_lower)], [float(v) for v in np.exp(ci_upper)]


def _x_domain(x_values: list[float], *, pad_discrete: bool = False) -> list[float]:
    if not x_values:
        return [0.0, 1.0]
    x_min = min(x_values)
    x_max = max(x_values)
    if abs(x_max - x_min) > 1e-15 and not pad_discrete:
        return [float(x_min), float(x_max)]
    if len(x_values) > 1:
        sorted_x = sorted(x_values)
        positive_spacings = [
            abs(sorted_x[i + 1] - sorted_x[i])
            for i in range(len(sorted_x) - 1)
            if abs(sorted_x[i + 1] - sorted_x[i]) > 1e-15
        ]
        if positive_spacings:
            pad = min(positive_spacings) * 0.5
        else:
            pad = max(abs(float(x_min)) * 0.1, 0.5)
    else:
        pad = max(abs(float(x_min)) * 0.1, 0.5)
    return [float(x_min - pad), float(x_max + pad)]


def _controls_payload(
    session,
    name: str,
    term,
    n_handles: int | None = None,
) -> dict[str, Any] | None:
    if str(term.metadata.get("term_type", term.kind)) not in CONTROL_HANDLE_TERM_TYPES:
        return None
    if term.x is None or term.levels is not None:
        return None
    try:
        controls = session.control_points(name, n_handles=n_handles)
    except TypeError:
        return None
    payload = {
        "x": [float(v) for v in controls["x"]],
        "y": [float(v) for v in np.exp(controls["log_effect"])],
        "log_effect": [float(v) for v in controls["log_effect"]],
        "basis_index": [int(v) for v in controls["basis_index"]],
        "count": int(len(controls["x"])),
        "min_count": int(controls["min_handles"]),
        "max_count": int(controls["max_handles"]),
    }
    if "basis" in controls:
        basis = np.asarray(controls["basis"], dtype=np.float64)
        payload["basis"] = [[float(v) for v in row] for row in basis]
    if "build_basis" in controls:
        build_basis = np.asarray(controls["build_basis"], dtype=np.float64)
        payload["build_basis"] = [[float(v) for v in row] for row in build_basis]
        payload["build_log_effect"] = [
            float(v) for v in np.asarray(controls["build_log_effect"], dtype=np.float64)
        ]
    return payload


def _handle_indices(term) -> np.ndarray:
    # Large spline grids stay interactive by drawing/editing a representative
    # subset of point markers while the full curve is still rendered.
    if term.levels is not None or term.size <= _MAX_INTERACTIVE_HANDLES:
        return np.arange(term.size, dtype=np.intp)
    return np.unique(np.linspace(0, term.size - 1, _MAX_INTERACTIVE_HANDLES, dtype=np.intp)).astype(
        np.intp
    )


def _exposure_payload(term, x_values: list[float], weights: list[float]) -> dict[str, Any]:
    # Discrete terms use bars; continuous terms use a smoothed density so the
    # exposure layer reads as a distribution instead of thousands of columns.
    w = _positive_weights(term, weights)
    if term.levels is not None or term.size <= 2:
        return {
            "kind": "bars",
            "x": [float(v) for v in x_values],
            "y": [float(v) for v in w],
            "label": "exposure",
        }
    return {
        "kind": "density",
        "x": [float(v) for v in x_values],
        "y": [float(v) for v in _smoothed_exposure_density(w)],
        "label": "exposure",
    }


def _positive_weights(term, weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.size != term.size or not np.any(w > 0):
        return np.ones(term.size, dtype=np.float64)
    return w


def _smoothed_exposure_density(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size < 3:
        return weights.copy()
    sigma = max(weights.size / 55.0, 1.25)
    radius = int(max(3, np.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(weights, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _impact_payload(
    term,
    weights: list[float],
    selection: np.ndarray,
    reference_log_effect: np.ndarray,
) -> dict[str, float]:
    w = _positive_weights(term, weights)
    reference = np.asarray(reference_log_effect, dtype=np.float64)
    if reference.shape != term.edited_log_effect.shape:
        reference = np.asarray(term.original_log_effect, dtype=np.float64)
    delta = np.asarray(term.edited_log_effect - reference, dtype=np.float64)
    total = float(np.sum(w))
    selected = np.zeros(term.size, dtype=bool)
    if selection.size:
        selected[np.asarray(selection, dtype=np.intp)] = True
    return {
        "total_weight": total,
        "weighted_mean_link_delta": float(np.average(delta, weights=w)),
        "weighted_mean_relativity": float(np.average(np.exp(delta), weights=w)),
        "max_abs_link_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "selected_weight_share": float(np.sum(w[selected]) / total) if total > 0 else 0.0,
    }
