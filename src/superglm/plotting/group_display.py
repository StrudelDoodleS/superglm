"""Display-only grouped-level projections for main-effect plots."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.features.ordered_categorical import OrderedCategorical
from superglm.inference.term import SmoothCurve, TermInference


@dataclass(frozen=True)
class GroupedTermDisplay:
    """Term projection plus mapping back to original categorical levels."""

    term: TermInference
    source_levels: list[list[str]]
    source_indices: list[list[int]]
    collapsed: bool


def project_grouped_term_for_display(
    model: Any,
    ti: TermInference,
    mode: str = "auto",
) -> GroupedTermDisplay:
    """Return a display projection for a grouped categorical term.

    The returned ``term`` may have grouped labels such as ``"A+B"`` for plotting,
    but the fitted model and inference object remain expanded over original levels.
    """
    resolved = resolve_grouped_level_display(mode, model, ti)
    levels = [str(level) for level in (ti.levels or [])]
    expanded = GroupedTermDisplay(
        term=ti,
        source_levels=[[level] for level in levels],
        source_indices=[[i] for i in range(len(levels))],
        collapsed=False,
    )
    grouping = _grouping_from_model(model, ti)
    if grouping is None or not levels or resolved == "expanded":
        return expanded

    groups = _source_groups(levels, grouping)
    if all(len(indices) == 1 for indices in groups):
        return expanded

    log_rel = _collapse_array(ti.log_relativity, groups)
    if log_rel is None:
        return expanded
    display_levels = ["+".join(levels[i] for i in indices) for indices in groups]
    display_term = replace(
        ti,
        levels=display_levels,
        log_relativity=log_rel,
        relativity=np.exp(log_rel),
        se_log_relativity=_collapse_array(ti.se_log_relativity, groups),
        ci_lower=_collapse_array(ti.ci_lower, groups),
        ci_upper=_collapse_array(ti.ci_upper, groups),
        smooth_curve=_collapsed_smooth_curve(ti, log_rel, len(display_levels)),
    )
    return GroupedTermDisplay(
        term=display_term,
        source_levels=[[levels[i] for i in indices] for indices in groups],
        source_indices=groups,
        collapsed=True,
    )


def resolve_grouped_level_display(mode: str, model: Any, ti: TermInference) -> str:
    """Resolve grouped level display mode for one term."""
    valid = {"auto", "expanded", "collapsed"}
    if mode not in valid:
        raise ValueError(
            f"grouped_level_display={mode!r} is not valid, expected one of {sorted(valid)}."
        )
    if mode != "auto":
        return mode
    grouping = _grouping_from_model(model, ti)
    if grouping is None:
        return "expanded"
    return "collapsed" if _is_ordered_categorical(model, ti) else "expanded"


def grouped_level_exposure(
    display: GroupedTermDisplay | None,
    X: pd.DataFrame | None,
    sample_weight: NDArray | None,
) -> NDArray | None:
    """Aggregate exposure for displayed categorical levels."""
    if display is None or X is None or sample_weight is None or display.term.name not in X.columns:
        return None
    raw = (
        pd.DataFrame(
            {
                "level": X[display.term.name].astype(str),
                "sample_weight": sample_weight,
            }
        )
        .groupby("level", sort=False)["sample_weight"]
        .sum()
    )
    return np.asarray(
        [
            float(sum(raw.get(source_level, 0.0) for source_level in source_levels))
            for source_levels in display.source_levels
        ],
        dtype=np.float64,
    )


def _grouping_from_model(model: Any, ti: TermInference) -> Any | None:
    spec = getattr(model, "_specs", {}).get(ti.name) if model is not None else None
    return getattr(spec, "_grouping", None)


def _is_ordered_categorical(model: Any, ti: TermInference) -> bool:
    spec = getattr(model, "_specs", {}).get(ti.name) if model is not None else None
    return isinstance(spec, OrderedCategorical)


def _source_groups(levels: list[str], grouping: Any) -> list[list[int]]:
    index_by_level = {str(level): i for i, level in enumerate(levels)}
    used: set[int] = set()
    groups: list[list[int]] = []
    for level in levels:
        index = index_by_level[level]
        if index in used:
            continue
        group_label = str(grouping.original_to_group.get(level, level))
        originals = [
            str(member) for member in grouping.group_to_originals.get(group_label, [level])
        ]
        indices = [index_by_level[original] for original in originals if original in index_by_level]
        if not indices:
            indices = [index]
        groups.append(indices)
        used.update(indices)
    return groups


def _collapse_array(values: NDArray | None, groups: list[list[int]]) -> NDArray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray([float(np.mean(arr[indices])) for indices in groups], dtype=np.float64)


def _collapsed_smooth_curve(
    ti: TermInference,
    log_rel: NDArray | None,
    n_levels: int,
) -> SmoothCurve | None:
    if ti.smooth_curve is None or log_rel is None or n_levels < 2:
        return None

    from scipy.interpolate import PchipInterpolator

    level_x = np.arange(n_levels, dtype=np.float64)
    curve_x = np.linspace(float(level_x[0]), float(level_x[-1]), 200)
    pchip = PchipInterpolator(level_x, log_rel)
    curve_log = pchip(curve_x)
    return SmoothCurve(
        x=curve_x,
        log_relativity=curve_log,
        relativity=np.exp(curve_log),
        level_x=level_x,
        se_log_relativity=None,
        ci_lower=None,
        ci_upper=None,
    )
