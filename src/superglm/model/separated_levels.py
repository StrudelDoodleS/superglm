"""A fit's separated categorical levels and crossed cells: its disclosure and its record.

A level, or crossed cell, whose rows' responses all sit on the response boundary has no
finite maximum-likelihood effect (``diagnostics.separation``); the build-time scan records
it on the term (``Categorical._separated_levels_at_fit``,
``CategoricalInteraction._separated_cells_at_fit``). The fit keeps every such level and its
coefficient is where the fit stopped. This module reads that one record: it prices the
disclosure a fit call makes when it returns, and names the coefficients and terms the
record covers for the readers that must not report them twice.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._warn import warn_caller
from superglm.diagnostics.separation import (
    SeparatedLevels,
    SeparatedLine,
    SeparationWarning,
    format_separated_levels,
    held_lines,
    lines_example,
    take_held_levels,
)


def separated_labels(spec: Any) -> tuple:
    """The levels, or crossed cells, the build-time scan recorded on ``spec``."""
    return tuple(getattr(spec, "_separated_levels_at_fit", ()) or ()) or tuple(
        getattr(spec, "_separated_cells_at_fit", ()) or ()
    )


def separated_terms(model) -> set[str]:
    """The terms with a recorded separated level or cell."""
    terms = [*model._specs.items(), *model._interaction_specs.items()]
    return {name for name, spec in terms if separated_labels(spec)}


def separated_columns(model, y: NDArray, sample_weight: NDArray) -> set[int]:
    """The coefficients a separated level or cell moves, which its own disclosure covers.

    Read from the scan's record (``separated_level_mask``, ``separated_cell_mask``).
    Under ``separation="ignore"`` the scan does not run and there is no record, so the
    scan's rule is applied to the design's own level codes, which name every non-base
    level and cell; their notices are silenced there.
    """
    from superglm.diagnostics.separation import (
        _separated_flags,
        response_boundaries,
        separated_cell_mask,
        separated_level_mask,
    )
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import CategoricalInteraction

    ignore = getattr(model, "_separation", "warn") == "ignore"
    boundaries = response_boundaries(model._distribution, model._link) if ignore else ()
    weights = np.asarray(sample_weight, dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    specs = {**model._specs, **model._interaction_specs}
    columns: set[int] = set()
    for group, matrix in zip(model._groups, model._dm.group_matrices, strict=True):
        spec = specs.get(group.feature_name)
        mask = None
        if isinstance(spec, Categorical) and group.size == len(spec._non_base):
            mask = separated_level_mask(spec, spec._non_base)
        elif isinstance(spec, CategoricalInteraction) and group.size == len(spec._pairs):
            mask = separated_cell_mask(spec, spec._pairs)
        else:
            continue
        codes = getattr(matrix, "codes", None)
        if mask is None and boundaries and codes is not None and len(codes) == len(response):
            # The base level's rows sit in the sink bin past the last column.
            mask = np.zeros(group.size + 1, dtype=bool)
            for boundary in boundaries:
                mask |= _separated_flags(codes, group.size + 1, response, weights, boundary)[0]
            mask = mask[: group.size]
        if mask is not None:
            columns.update(group.start + int(j) for j in np.flatnonzero(mask))
    return columns


def predicted_cells(model, term: str, frame) -> set[tuple]:
    """The cells of crossed term ``term`` the rows of ``frame`` are scored at.

    Each parent's raw values resolve as its main effect's do (grouping, then
    the unseen policy); ``predict`` has already reported unseen labels.
    """
    import warnings

    first, second = model._interaction_specs[term].parent_names
    codes = []
    for parent in (first, second):
        spec = model._specs[parent]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = spec._resolve_predict_labels(frame.column_array(parent))
            codes.append((spec._predict_codes(resolved), spec._levels))
    (codes1, levels1), (codes2, levels2) = codes
    both = (codes1 >= 0) & (codes2 >= 0)
    met = np.unique(np.stack([codes1[both], codes2[both]], axis=1), axis=0)
    return {(levels1[i], levels2[j]) for i, j in met}


def _raw_base(spec) -> Any | None:
    """A raw value ``spec`` scores at its base level: the base, or a member of the base group."""
    grouping = getattr(spec, "_grouping", None)
    if grouping is None:
        return spec._base_level
    members = grouping.original_to_group.items()
    return next((raw for raw, group in members if group == spec._base_level), None)


def _at_base(line: SeparatedLine, specs: dict[str, Any]) -> dict[str, Any] | None:
    """The raw values that score ``line``'s rows at its base, by column; ``None`` if unpriced.

    A level is moved to its term's base level, a crossed cell to the base cell (both
    parents at their base levels).  There is no base rate to state when the base is
    itself separated or, for a cell, holds no rows.
    """
    spec = specs[line.term]
    if line.is_base or line.base in separated_labels(spec):
        return None
    if line.kind == "levels":
        parents = {line.term: specs[line.term]}
    else:
        occupied = getattr(spec, "_occupied_cells_at_fit", None)
        if occupied is not None and tuple(line.base) not in set(occupied):
            return None
        parents = {name: specs[name] for name in spec.parent_names}
    values = {name: _raw_base(parent) for name, parent in parents.items()}
    return None if any(value is None for value in values.values()) else values


def _priced(
    model, entries: list[tuple[SeparatedLine, NDArray]], found: SeparatedLevels
) -> list[SeparatedLine]:
    """``entries``' lines with the published fit's response totals over their rows.

    Every line's rows are scored through the fit's own prediction path, offset
    included and every other term kept (an interaction the level enters, a
    spline): as they are, for the fitted total, and with the term's label set to
    its base (``_at_base``), for the total expected at the base's rate.  Both are
    finite for every link.  Each row counts its weight under frequency weights
    and once under prior weights (``SeparatedLevels.counts``).
    """
    import pandas as pd

    from superglm.model.base import _eta_to_mu, _predict_eta, _prediction_plan

    specs = {**model._specs, **model._interaction_specs}
    rows = np.concatenate([part for _, part in entries])
    plan = _prediction_plan(model)
    names = dict.fromkeys(
        [term["name"] for term in plan["features"]]
        + [parent for term in plan["interactions"] for parent in term["parent_names"]]
    )
    columns = {name: found.frame.column_array(name)[rows] for name in names}
    offset = None if found.offset is None else found.offset[rows]

    def mean(frame: dict[str, NDArray]) -> NDArray:
        eta = _predict_eta(
            model,
            pd.DataFrame(frame),
            offset,
            random_effects="conditional",
            fitted=True,
            warn=False,
        )
        return np.asarray(_eta_to_mu(model, eta), dtype=np.float64)

    spans, end = [], 0
    moved = dict(columns)
    for line, part in entries:
        start, end = end, end + len(part)
        values = _at_base(line, specs)
        spans.append((start, end, values is not None))
        for name, value in (values or {}).items():
            if moved[name] is columns[name]:
                moved[name] = columns[name].astype(object)
            moved[name][start:end] = value
    counts = found.counts(rows)
    fitted = counts * mean(columns)
    expected = counts * mean(moved) if any(priced for _, _, priced in spans) else None
    return [
        replace(
            line,
            fitted=float(np.sum(fitted[start:end])),
            expected=None if expected is None or not priced else float(np.sum(expected[start:end])),
        )
        for (line, _), (start, end, priced) in zip(entries, spans, strict=True)
    ]


def separated_levels_message(held: list[SeparatedLevels], model) -> str | None:
    """The one disclosure of a fit call whose builds found ``held``, priced by ``model``.

    ``model`` is the fit the call publishes.  The latest build's rows stand for each
    label, and only the labels ``model``'s own record holds are named: ``None`` when it
    holds none of them.
    """
    specs = {**model._specs, **model._interaction_specs}
    by_build: dict[int, tuple[SeparatedLevels, list[tuple[SeparatedLine, NDArray]]]] = {}
    for line, found, rows in held_lines(held):
        spec = specs.get(line.term)
        if spec is not None and line.label in separated_labels(spec):
            by_build.setdefault(id(found), (found, []))[1].append((line, rows))
    priced = [
        line for found, entries in by_build.values() for line in _priced(model, entries, found)
    ]
    if not priced:
        return None
    return format_separated_levels(priced, fitted=True, example=lines_example(priced, specs))


def disclose_held_levels(model) -> None:
    """Warn, priced by ``model``, of the separated levels and cells this fit call's builds found.

    ``model`` is the candidate a fit call is about to publish: the warning is formed
    and emitted before it is installed, so a warning raised as an error leaves the
    model's previous fit in place (``diagnostics.separation.disclosure_scope``).
    """
    held = take_held_levels()
    if not held:
        return
    message = separated_levels_message(held, model)
    if message is not None:
        warn_caller(message, SeparationWarning)


__all__ = [
    "disclose_held_levels",
    "predicted_cells",
    "separated_columns",
    "separated_labels",
    "separated_levels_message",
    "separated_terms",
]
