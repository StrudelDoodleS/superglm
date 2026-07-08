"""Display-only grouped-level projections for the editor payload."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_group_display(term_payload: dict[str, Any]) -> dict[str, Any]:
    """Build collapsed display metadata for grouped categorical terms."""
    levels = term_payload.get("levels")
    groups = term_payload.get("level_groups") or []
    if not levels or not groups:
        return {"available": False, "default_mode": "expanded", "collapsed": None}

    source_groups = _source_groups(len(levels), groups)
    weights = np.asarray(term_payload.get("weights") or [1.0] * len(levels), dtype=np.float64)
    y = np.asarray(term_payload["y"], dtype=np.float64)
    original_y = np.asarray(term_payload["original_y"], dtype=np.float64)
    previous_y = _optional_array(term_payload.get("previous_y"))

    collapsed: dict[str, Any] = {
        "levels": [],
        "x": [float(i) for i in range(len(source_groups))],
        "y": [],
        "original_y": [],
        "previous_y": None,
        "weights": [],
        "source_indices": [],
        "source_levels": [],
        "is_group": [],
        "group_labels": [],
    }
    collapsed["x_domain"] = _x_domain(collapsed["x"])

    ci_lower = _optional_array(term_payload.get("ci_lower_y"))
    ci_upper = _optional_array(term_payload.get("ci_upper_y"))
    if ci_lower is not None and ci_upper is not None:
        collapsed["ci_lower_y"] = []
        collapsed["ci_upper_y"] = []
    if previous_y is not None:
        collapsed["previous_y"] = []

    for source_group in source_groups:
        indices = source_group["indices"]
        source_levels = [str(levels[i]) for i in indices]
        group_weight = weights[indices]
        label = str(source_group.get("label") or "+".join(source_levels))
        if len(indices) == 1:
            label = source_levels[0]

        collapsed["levels"].append(label)
        collapsed["source_indices"].append([int(i) for i in indices])
        collapsed["source_levels"].append(source_levels)
        collapsed["is_group"].append(len(indices) > 1)
        collapsed["group_labels"].append(label)
        collapsed["weights"].append(float(np.sum(group_weight)))
        collapsed["y"].append(_weighted_average(y[indices], group_weight))
        collapsed["original_y"].append(_weighted_average(original_y[indices], group_weight))
        if previous_y is not None:
            collapsed["previous_y"].append(_weighted_average(previous_y[indices], group_weight))

        if ci_lower is not None and ci_upper is not None:
            collapsed["ci_lower_y"].append(_weighted_average(ci_lower[indices], group_weight))
            collapsed["ci_upper_y"].append(_weighted_average(ci_upper[indices], group_weight))

    collapsed["exposure"] = {
        "kind": "bars",
        "x": collapsed["x"],
        "y": collapsed["weights"],
        "label": "exposure",
    }

    return {
        "available": True,
        "default_mode": _default_mode(term_payload),
        "collapsed": collapsed,
    }


def _source_groups(n_levels: int, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_by_first: dict[int, dict[str, Any]] = {}
    for group in groups:
        indices = sorted(
            {int(index) for index in group.get("indices", []) if 0 <= int(index) < n_levels}
        )
        if len(indices) >= 2:
            grouped_by_first[min(indices)] = {"label": group.get("label"), "indices": indices}

    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for index in range(n_levels):
        if index in used:
            continue
        group = grouped_by_first.get(index)
        if group is None:
            group = {"label": None, "indices": [index]}
        out.append(group)
        used.update(group["indices"])
    return out


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total > 0:
        return float(np.average(values, weights=weights))
    return float(np.mean(values))


def _optional_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _default_mode(term_payload: dict[str, Any]) -> str:
    term_type = str(term_payload.get("term_type", term_payload.get("kind", "")))
    return "collapsed" if term_type == "ordered categorical" else "expanded"


def _x_domain(x_values: list[float]) -> list[float]:
    if not x_values:
        return [0.0, 1.0]
    if len(x_values) == 1:
        return [x_values[0] - 0.5, x_values[0] + 0.5]
    return [float(x_values[0] - 0.5), float(x_values[-1] + 0.5)]
