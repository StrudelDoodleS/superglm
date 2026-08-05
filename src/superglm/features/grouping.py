"""Level grouping for categorical features.

Provides ``LevelGrouping`` and ``collapse_levels()`` to let users merge
sparse categorical levels for fitting, with automatic expansion back to
original levels at inference/plotting time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class LevelGrouping:
    """Mapping between original categorical levels and grouped levels.

    Parameters
    ----------
    original_to_group : dict[str, str]
        Maps each original level to its group label.  Identity-mapped
        levels map to themselves.
    group_to_originals : dict[str, list[str]]
        Inverse mapping: group label -> list of original levels in that group.
    all_original_levels : list[str]
        All original level names, in their original order.
    grouped_levels : list[str]
        The unique group labels, in order.
    """

    original_to_group: dict[str, str]
    group_to_originals: dict[str, list[str]]
    all_original_levels: list[str]
    grouped_levels: list[str]


def _data_spelling(unique_levels: list[str]):
    """Return a function re-spelling a caller's level label the way the data does.

    Levels are compared as strings, but a caller and a column can render the same
    level differently: ``order=[1, 2, 9]`` against a float column spells them
    ``"1"`` and ``"1.0"``, and those never match. The declared labels then form a
    namespace of their own, the grouping is built in it, and every row of the
    column looks like an unseen level (issue #237).

    Falls back to the label unchanged when it is already present verbatim, when it
    is not numeric, or when no numeric level in the data equals it -- so a level
    that genuinely is not in the data still fails the caller's own check rather
    than being quietly renamed into one that is.
    """
    present = set(unique_levels)
    by_value: dict[float, str] = {}
    for level in unique_levels:
        try:
            value = float(level)
        except (TypeError, ValueError):
            continue
        by_value.setdefault(value, level)

    def respell(label) -> str:
        text = str(label)
        if text in present:
            return text
        try:
            value = float(text)
        except (TypeError, ValueError):
            return text
        return by_value.get(value, text)

    return respell


def collapse_levels(
    data: pd.Series | NDArray | list[str],
    *,
    from_level: str | None = None,
    below: str | None = None,
    groups: dict[str, list[str]] | None = None,
    order: list[str] | None = None,
) -> LevelGrouping:
    """Build a ``LevelGrouping`` from data and grouping rules.

    Three modes (mutually exclusive with ``groups``):

    - ``from_level="25"`` — levels at position >= ``"25"`` (in sorted order)
      collapse to group label ``"25+"``.
    - ``below="3"`` — levels at position < ``"3"`` collapse to ``"<3"``.
    - ``groups={"South": ["TX", "FL"]}`` — explicit mapping; unlisted levels
      are identity-mapped.

    ``from_level`` and ``below`` can be combined, but neither can be mixed
    with ``groups``.

    Parameters
    ----------
    data : Series, array, or list of str
        The categorical column.  Used to discover all unique levels.
    from_level : str or None
        Collapse levels at or after this position (inclusive) into a
        single ``"<from_level>+"`` group.
    below : str or None
        Collapse levels before this position (exclusive) into a single
        ``"<<below>"`` group.
    groups : dict[str, list[str]] or None
        Explicit mapping of group labels to lists of original levels.
    order : list[str] or None
        Optional explicit level ordering.  If not given, levels are
        sorted lexicographically.

    Returns
    -------
    LevelGrouping

    Raises
    ------
    ValueError
        If a level appears in multiple groups, if mentioned levels don't
        exist in data, or if ``from_level``/``below`` are used with ``groups``.
    """
    # Discover unique levels
    if isinstance(data, pd.Series):
        unique_levels = data.astype(str).unique().tolist()
    elif isinstance(data, np.ndarray):
        unique_levels = [str(v) for v in np.unique(data)]
    else:
        unique_levels = list(set(str(v) for v in data))

    # Re-spell the caller's labels the way the data does before anything compares
    # them; otherwise `order`/`groups` build a namespace of their own and every
    # row of the column reads as an unseen level (issue #237).
    respell = _data_spelling(unique_levels)
    if order is not None:
        all_levels = [respell(level) for level in order]
    else:
        all_levels = sorted(unique_levels)
    if groups is not None:
        groups = {label: [respell(m) for m in members] for label, members in groups.items()}
    if from_level is not None:
        from_level = respell(from_level)
    if below is not None:
        below = respell(below)

    level_set = set(all_levels)

    # Validate mode exclusivity
    positional = from_level is not None or below is not None
    explicit = groups is not None
    if positional and explicit:
        raise ValueError(
            "Cannot mix from_level/below with groups. "
            "Use either positional collapsing or explicit groups, not both."
        )

    if not positional and not explicit:
        # Identity grouping — nothing to collapse
        mapping: dict[str, str] = {lev: lev for lev in all_levels}
        inv: dict[str, list[str]] = {lev: [lev] for lev in all_levels}
        return LevelGrouping(
            original_to_group=mapping,
            group_to_originals=inv,
            all_original_levels=all_levels,
            grouped_levels=all_levels,
        )

    mapping = {}

    if explicit:
        # Explicit groups mode
        assert groups is not None
        seen: set[str] = set()
        for group_label, members in groups.items():
            for m in members:
                if str(m) not in level_set:
                    raise ValueError(
                        f"Level {m!r} in group {group_label!r} not found in data. "
                        f"Known levels: {sorted(level_set)}"
                    )
                if str(m) in seen:
                    raise ValueError(f"Level {m!r} appears in multiple groups.")
                seen.add(str(m))
                mapping[str(m)] = group_label

        # Identity-map unlisted levels
        for lev in all_levels:
            if lev not in mapping:
                mapping[lev] = lev
    else:
        # Positional mode (from_level and/or below)
        if from_level is not None and str(from_level) not in level_set:
            raise ValueError(f"from_level={from_level!r} not found in levels: {sorted(level_set)}")
        if below is not None and str(below) not in level_set:
            raise ValueError(f"below={below!r} not found in levels: {sorted(level_set)}")

        from_idx = None
        below_idx = None
        if from_level is not None:
            from_idx = all_levels.index(str(from_level))
        if below is not None:
            below_idx = all_levels.index(str(below))

        from_label = f"{from_level}+" if from_level is not None else None
        below_label = f"<{below}" if below is not None else None

        for i, lev in enumerate(all_levels):
            if below_idx is not None and i < below_idx:
                mapping[lev] = below_label
            elif from_idx is not None and i >= from_idx:
                mapping[lev] = from_label
            else:
                mapping[lev] = lev

    # Build inverse mapping and grouped_levels (preserving order)
    inv = {}
    grouped_levels_ordered = []
    for lev in all_levels:
        g = mapping[lev]
        if g not in inv:
            inv[g] = []
            grouped_levels_ordered.append(g)
        inv[g].append(lev)

    return LevelGrouping(
        original_to_group=mapping,
        group_to_originals=inv,
        all_original_levels=all_levels,
        grouped_levels=grouped_levels_ordered,
    )
