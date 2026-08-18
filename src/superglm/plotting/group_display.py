"""Display-only grouped-level projections for main-effect plots."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm.features.ordered_categorical import OrderedCategorical, group_axis_position
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
    group_indices = [group["indices"] for group in groups]
    if all(len(indices) == 1 for indices in group_indices):
        return expanded

    log_rel = _collapse_array(ti.log_relativity, group_indices)
    if log_rel is None:
        return expanded
    display_levels = [
        str(group["label"]) if len(group["indices"]) > 1 else levels[group["indices"][0]]
        for group in groups
    ]
    display_term = replace(
        ti,
        levels=display_levels,
        log_relativity=log_rel,
        relativity=np.exp(log_rel),
        se_log_relativity=_collapse_array(ti.se_log_relativity, group_indices),
        ci_lower=_collapse_array(ti.ci_lower, group_indices),
        ci_upper=_collapse_array(ti.ci_upper, group_indices),
        spline=None,
        level_is_special=_collapse_special_mask(ti.level_is_special, group_indices),
        smooth_curve=_collapsed_smooth_curve(ti, group_indices),
    )
    return GroupedTermDisplay(
        term=display_term,
        source_levels=[[levels[i] for i in indices] for indices in group_indices],
        source_indices=group_indices,
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
    X: EagerFrame | None,
    sample_weight: NDArray | None,
) -> NDArray | None:
    """Aggregate exposure for displayed categorical levels."""
    if display is None or X is None or sample_weight is None or display.term.name not in X.columns:
        return None
    raw = (
        pd.DataFrame(
            {
                "level": pd.Series(
                    X.column_array(display.term.name),
                    name=display.term.name,
                ).astype(str),
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


def _source_groups(levels: list[str], grouping: Any) -> list[dict[str, Any]]:
    index_by_level = {str(level): i for i, level in enumerate(levels)}
    used: set[int] = set()
    groups: list[dict[str, Any]] = []
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
        groups.append({"label": group_label, "indices": indices})
        used.update(indices)
    return groups


def _collapse_array(values: NDArray | None, groups: list[list[int]]) -> NDArray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray([float(np.mean(arr[indices])) for indices in groups], dtype=np.float64)


def _collapse_special_mask(mask: NDArray | None, groups: list[list[int]]) -> NDArray | None:
    """Collapse the free-level mask onto display groups.

    A group is free only when every member is; a grouping may not mix a special
    with ordered levels, so this is an all-or-nothing test.
    """
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    return np.asarray([bool(arr[indices].all()) for indices in groups], dtype=bool)


def _collapsed_smooth_curve(
    ti: TermInference,
    groups: list[list[int]],
) -> SmoothCurve | None:
    """Keep the fitted curve and move each marker to its group's own position.

    The curve itself is never rebuilt: collapsing levels is a display
    operation, and re-interpolating through the collapsed markers would
    draw a shape the model never fitted.

    Where a group sits is ``group_axis_position`` -- the ONE definition of that
    convention, shared with the spec that placed the group there in the first
    place (``features/ordered_categorical.py``).  This half used to spell the
    mean out itself, which agreed with the spec on every grouping but one and
    left issue #326's marker half a level width from its own fitted value.
    Sharing the function is what makes the collapse the exact inverse of the
    expansion rather than a second implementation that happens to match.

    ``level_x`` covers the smoothed levels only, so an all-special group has no
    position on the curve's axis and is dropped here rather than indexed into
    ``level_x``.  The renderers place those markers from ``level_is_special``,
    which ``_collapse_special_mask`` keeps parallel to the display levels.
    """
    curve = ti.smooth_curve
    if curve is None:
        return None
    if curve.level_x is None:
        # Without level positions there is nothing to collapse, and handing the
        # uncollapsed curve to a display term with fewer levels would put the
        # markers and the curve on incompatible axes.
        return None
    level_x = np.asarray(curve.level_x, dtype=np.float64)
    n_levels = len(ti.levels or [])
    special = (
        np.asarray(ti.level_is_special, dtype=bool)
        if ti.level_is_special is not None
        else np.zeros(n_levels, dtype=bool)
    )
    # Position of each smooth display level within level_x.
    smooth_pos = np.cumsum(~special) - 1
    collapsed: list[float] = []
    for indices in groups:
        idx = np.asarray(indices, dtype=np.intp)
        smooth_idx = idx[~special[idx]]
        if smooth_idx.size == 0:
            continue  # an all-free group has no place on the fitted curve
        collapsed.append(group_axis_position(level_x[smooth_pos[smooth_idx]]))
    return replace(curve, level_x=np.asarray(collapsed, dtype=np.float64))
