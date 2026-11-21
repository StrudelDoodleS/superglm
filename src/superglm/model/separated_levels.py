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

import functools
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.diagnostics.separation import (
    SeparatedLevels,
    SeparatedLine,
    format_separated_levels,
    level_lines,
    lines_example,
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


def _log_relativities(model, term: str) -> dict:
    """A fitted term's log relativity by level, or by cell: empty for a term the model lacks."""
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import CategoricalInteraction

    spec = {**model._specs, **model._interaction_specs}.get(term)
    blocks = [model.result.beta[g.sl] for g in model._groups if g.feature_name == term]
    if not blocks or not isinstance(spec, Categorical | CategoricalInteraction):
        return {}
    raw = spec.reconstruct(np.concatenate(blocks))
    if isinstance(spec, Categorical):
        return dict(raw["log_relativities"])
    return dict(zip(map(tuple, raw["pairs"]), raw["pair_log_relativities"], strict=True))


def _log_relativity(line: SeparatedLine, effects) -> float:
    """The fitted log relativity of ``line``'s level, or cell, against the term's base.

    ``effects(term)`` is ``_log_relativities``. A cell's is the whole categorical effect
    at it against the base cell: both parents' main effects and the interaction,
    wherever the fit put its walk.
    """
    if line.kind == "levels":
        return float(effects(line.term).get(line.label, 0.0))
    first, second = line.parents
    return float(
        effects(first).get(line.label[0], 0.0)
        + effects(second).get(line.label[1], 0.0)
        + effects(line.term).get(tuple(line.label), 0.0)
    )


def _priced(model, lines: list[tuple[SeparatedLine, NDArray]], found: SeparatedLevels) -> list:
    """``lines`` with the published fit's relativities and the responses their rows expect at base.

    One prediction over every line's rows: the fitted predictor, less the line's own
    effect against the base, is its rows' predictor at the base's rate.
    """
    from superglm.links import stabilize_eta
    from superglm.model.base import _eta_to_mu, _predict_eta

    effects = functools.cache(lambda term: _log_relativities(model, term))
    rows = np.concatenate([part for _, part in lines])
    eta = _predict_eta(
        model,
        found.frame.take_rows(rows),
        None if found.offset is None else found.offset[rows],
        random_effects="conditional",
        stabilize=False,
        fitted=True,
        warn=False,
    )
    priced = []
    end = 0
    for line, part in lines:
        start, end = end, end + len(part)
        if line.is_base:
            priced.append(line)
            continue
        delta = _log_relativity(line, effects)
        mean = _eta_to_mu(model, stabilize_eta(eta[start:end] - delta, model._link))
        expected = float(found.sample_weight[part] @ np.asarray(mean, dtype=np.float64))
        priced.append(replace(line, expected=expected, relativity=float(np.exp(delta))))
    return priced


def separated_levels_message(held: list[SeparatedLevels], model) -> str:
    """The one disclosure of a fit call whose builds found ``held``.

    ``model`` is the fit the call published, or ``None`` when it published none;
    each level is then stated as its build counted it. The latest build's rows
    stand for each level, and a published fit names only the levels its own
    record holds.
    """
    latest: dict[tuple, tuple[SeparatedLine, int, NDArray]] = {}
    for index, found in enumerate(held):
        rows = [part for finding in found.findings for part in finding.rows]
        for line, part in zip(level_lines(found), rows, strict=True):
            latest[(line.term, line.kind, line.label)] = (line, index, part)
    specs = {**held[-1].specs}
    priced: list[SeparatedLine] = []
    if model is not None:
        specs = {**model._specs, **model._interaction_specs}
        by_build: dict[int, list[tuple[SeparatedLine, NDArray]]] = {}
        for line, index, part in latest.values():
            spec = specs.get(line.term)
            if spec is not None and line.label in separated_labels(spec):
                by_build.setdefault(index, []).append((line, part))
        for index, lines in by_build.items():
            priced.extend(_priced(model, lines, held[index]))
    if priced:
        return format_separated_levels(priced, fitted=True, example=lines_example(priced, specs))
    found_lines = [line for line, _, _ in latest.values()]
    return format_separated_levels(found_lines, example=lines_example(found_lines, specs))


__all__ = [
    "predicted_cells",
    "separated_columns",
    "separated_labels",
    "separated_levels_message",
    "separated_terms",
]
