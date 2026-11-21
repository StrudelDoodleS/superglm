"""A fit's separated categorical levels and crossed cells: its disclosure and its record.

A level, or crossed cell, whose rows' responses all sit on the response boundary has no
finite maximum-likelihood effect (``diagnostics.separation``); the build-time scan records
it on the term (``Categorical._separated_levels_at_fit``,
``CategoricalInteraction._separated_cells_at_fit``). The fit keeps every such level and its
coefficient is where the fit stopped. This module reads that one record: it prices the
disclosure a fit call makes before it publishes, decides which reported contrasts every
reader leaves blank (one rule, below), and names the coefficients and terms the record
covers for the notices that must not report them twice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
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
    """The terms a recorded separated level or cell belongs to: its own, and a cell's parents."""
    terms = {name for name, spec in model._specs.items() if separated_labels(spec)}
    for name, spec in model._interaction_specs.items():
        if separated_labels(spec):
            terms.update({name, *spec.parent_names})
    return terms


# ── which reported contrasts the healthy rows leave undetermined ─────────────
#
# A separated level's rows sit on the response boundary, and the likelihood rises
# without bound along any direction that moves those rows' predictor toward it and
# leaves every other row's alone.  A reported contrast ``c' beta`` is determined by
# the fit only when no such direction moves it: when ``c`` lies in the row space of
# the design restricted to the healthy rows (positive weight, in no recorded level or
# cell).  That is Lenth's estimability test (``model.estimability_ops``, R's
# ``predict.lm``) on those rows: ``||c' N|| > tol ||c||`` flags ``c``, with ``N`` an
# orthonormal basis of the restricted design's null space and ``tol`` the same
# ``PREDICTION_TOLERANCE``.  Healthy contrasts keep their numbers.
#
# Each recorded term gives one block, the coordinates a separating direction can
# reach: the term, its categorical parents (a crossed term's), and the intercept.  A
# block's design rows are indicator rows, one per healthy occupied level or cell, so
# its null space is read from at most ``L1 x L2`` rows (``_rank_and_null_space``).
# ``Categorical`` and ``CategoricalInteraction`` columns carry no quadratic penalty,
# and a term a selection penalty targets is exempt from the scan, so a block's penalty
# adds no rows.  Coordinates run over ``[beta, intercept]``: the intercept is index
# ``p``.


@dataclass(frozen=True)
class _Record:
    """The separated labels a fit's readers test against, by term.

    ``occupied`` holds a crossed term's occupied cells, base row and column included;
    ``None`` reads every cell of its grid as occupied (a model saved before the build
    recorded them).
    """

    levels: dict[str, frozenset]
    cells: dict[str, frozenset]
    occupied: dict[str, frozenset | None]


def _spec_record(specs: dict, interaction_specs: dict) -> _Record:
    """What the build-time scan recorded on the specs."""
    levels = {
        name: frozenset(getattr(spec, "_separated_levels_at_fit", ()) or ())
        for name, spec in specs.items()
    }
    cells = {
        name: frozenset(map(tuple, getattr(spec, "_separated_cells_at_fit", ()) or ()))
        for name, spec in interaction_specs.items()
    }
    occupied = {}
    for name, spec in interaction_specs.items():
        held = getattr(spec, "_occupied_cells_at_fit", None)
        occupied[name] = None if held is None else frozenset(map(tuple, held))
    return _Record(
        levels={name: held for name, held in levels.items() if held},
        cells={name: held for name, held in cells.items() if held},
        occupied=occupied,
    )


def _term_group(groups, name: str, width: int):
    """``name``'s one group when it is ``width`` columns wide, else ``None``."""
    found = [group for group in groups if group.feature_name == name]
    return found[0] if len(found) == 1 and found[0].size == width else None


def _null_block(columns: list[int], rows: list[list[int]]) -> tuple[NDArray, NDArray] | None:
    """``(columns, N)``: an orthonormal basis of the null space of the block's indicator rows.

    ``rows`` lists, per healthy level or cell, the positions within ``columns`` its
    design row has a one at.  ``None`` when the rows determine every coordinate.
    """
    from superglm.diagnostics.separation import _rank_and_null_space

    design = np.zeros((len(rows), len(columns)))
    for i, hit in enumerate(rows):
        design[i, hit] = 1.0
    null = _rank_and_null_space(design)[1]
    return (np.asarray(columns, dtype=np.intp), null) if null.shape[1] else None


def _blocks(groups, specs: dict, interaction_specs: dict, record: _Record) -> list:
    """Each recorded term's block (module comment above), as ``(columns, N)``."""
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import CategoricalInteraction

    intercept = max((group.end for group in groups), default=0)
    blocks = []

    def position(spec, offset: int) -> dict:
        """Each non-base level's position within a block, its column's first at ``offset``."""
        return {level: offset + j for j, level in enumerate(spec._non_base)}

    for name, held in record.levels.items():
        spec = specs.get(name)
        group = None if spec is None else _term_group(groups, name, len(spec._non_base))
        if not isinstance(spec, Categorical) or group is None:
            continue
        at = position(spec, 1)
        pinned = set(spec._pinned_levels)
        healthy = [level for level in spec._levels if level not in held and level not in pinned]
        rows = [[0, *([at[level]] if level in at else [])] for level in healthy]
        block = _null_block([intercept, *range(group.start, group.end)], rows)
        if block is not None:
            blocks.append(block)
    for name, held in record.cells.items():
        spec = interaction_specs.get(name)
        if not isinstance(spec, CategoricalInteraction):
            continue
        first, second = (specs.get(parent) for parent in spec.parent_names)
        if not (isinstance(first, Categorical) and isinstance(second, Categorical)):
            continue
        parts = [
            _term_group(groups, parent, len(parent_spec._non_base))
            for parent, parent_spec in zip(spec.parent_names, (first, second), strict=True)
        ]
        cross = _term_group(groups, name, len(spec._pairs))
        if any(group is None for group in parts) or cross is None:
            continue
        group1, group2 = parts
        at1 = position(first, 1)
        at2 = position(second, 1 + group1.size)
        grid = {cell: g for g, cell in enumerate(spec._all_pairs or spec._pairs)}
        to_column = spec._grid_to_col
        offset = 1 + group1.size + group2.size
        occupied = record.occupied.get(name)
        # a row of a separated parent level is not a healthy row of the cross either
        out1, out2 = (
            {*record.levels.get(spec.parent_names[0], ()), *first._pinned_levels},
            {*record.levels.get(spec.parent_names[1], ()), *second._pinned_levels},
        )
        rows = []
        for a, b in product(first._levels, second._levels):
            if a in out1 or b in out2 or (a, b) in held:
                continue
            if occupied is not None and (a, b) not in occupied:
                continue
            hit = [0, *([at1[a]] if a in at1 else []), *([at2[b]] if b in at2 else [])]
            g = grid.get((a, b))
            column = None if g is None else (g if to_column is None else int(to_column[g]))
            if column is not None and column >= 0:
                hit.append(offset + column)
            rows.append(hit)
        columns = [
            intercept,
            *range(group1.start, group1.end),
            *range(group2.start, group2.end),
            *range(cross.start, cross.end),
        ]
        block = _null_block(columns, rows)
        if block is not None:
            blocks.append(block)
    return blocks


def _moves(blocks: list, contrasts: NDArray) -> NDArray:
    """Which rows of ``contrasts`` (over ``[beta, intercept]``) a block's null space moves."""
    from superglm.model.estimability_ops import PREDICTION_TOLERANCE

    contrasts = np.atleast_2d(np.asarray(contrasts, dtype=np.float64))
    size = np.linalg.norm(contrasts, axis=1)
    moved = np.zeros(len(contrasts), dtype=bool)
    for columns, null in blocks:
        share = np.linalg.norm(contrasts[:, columns] @ null, axis=1)
        moved |= share > PREDICTION_TOLERANCE * size
    return moved


def _coefficient_flags(groups, specs, interaction_specs, record: _Record) -> NDArray | None:
    """Over ``[beta, intercept]``: each coefficient a separating direction moves."""
    if not (record.levels or record.cells):
        return None
    from superglm.model.estimability_ops import PREDICTION_TOLERANCE

    flags = np.zeros(max((group.end for group in groups), default=0) + 1, dtype=bool)
    for columns, null in _blocks(groups, specs, interaction_specs, record):
        flags[columns] |= np.linalg.norm(null, axis=1) > PREDICTION_TOLERANCE
    return flags


def separated_coefficients(groups, specs: dict, interaction_specs: dict) -> NDArray | None:
    """Which coefficients' Wald statistics a recorded separated level or cell voids.

    Over ``[beta, intercept]`` (the intercept last); ``None`` when nothing is recorded.
    A coefficient is its own contrast ``e_j``: a separated level's own, every
    contrast against a separated base, a parent's main effect beside a separated cell
    in its base row or column, the intercept when it moves.
    """
    return _coefficient_flags(
        groups, specs, interaction_specs, _spec_record(specs, interaction_specs)
    )


def level_flags(
    groups, specs: dict, interaction_specs: dict, name: str, levels, *, centering: str = "native"
) -> tuple[NDArray | None, NDArray | None]:
    """``(flags, centre)`` for a ``Categorical``'s reported levels, each ``None`` when moot.

    ``flags`` marks the levels reported blank: a recorded level, and a level whose
    reported contrast a separating direction moves (against the base, or against the
    mean under ``centering="mean"``).  ``centre`` marks the levels a mean-centred
    report centres on: those whose contrast against the base is determined.  A
    separated base moves every contrast against it, so the centre is then the levels
    not recorded, and every centred contrast is reported blank as well: their
    covariance is a difference of entries the walk has grown to about ``1e14``.
    """
    from superglm.features.categorical import Categorical

    spec = specs.get(name)
    record = _spec_record(specs, interaction_specs)
    if not isinstance(spec, Categorical) or not (record.levels or record.cells):
        return None, None
    levels = list(levels)
    recorded = np.array([level in record.levels.get(name, ()) for level in levels], dtype=bool)
    group = _term_group(groups, name, len(spec._non_base))
    blocks = _blocks(groups, specs, interaction_specs, record) if group is not None else []
    native = np.zeros((len(levels), max((g.end for g in groups), default=0) + 1))
    if group is not None:
        at = {level: group.start + j for j, level in enumerate(spec._non_base)}
        for i, level in enumerate(levels):
            if level in at:
                native[i, at[level]] = 1.0
    against_base = recorded | _moves(blocks, native)
    centre = ~against_base
    if not centre.any():
        centre = ~recorded
    centre_on = centre if centre.any() and not centre.all() else None
    if centering != "mean":
        flags = against_base
    elif spec._base_level in record.levels.get(name, ()):
        flags = np.ones(len(levels), dtype=bool)
    else:
        mean = (native[centre] if centre_on is not None else native).mean(axis=0)
        flags = recorded | _moves(blocks, native - mean)
    return (flags if flags.any() else None), centre_on


def pair_flags(groups, specs: dict, interaction_specs: dict, name: str, pairs) -> NDArray | None:
    """Which of a crossed term's reported cells (``pairs``, the full non-base grid) are blank.

    A recorded cell, and a cell whose interaction coefficient a separating direction
    moves; a pruned cell's coefficient is zero by construction.  ``None`` when none is.
    """
    from superglm.features.interaction import CategoricalInteraction

    spec = interaction_specs.get(name)
    record = _spec_record(specs, interaction_specs)
    if not isinstance(spec, CategoricalInteraction) or not (record.levels or record.cells):
        return None
    pairs = [tuple(pair) for pair in pairs]
    recorded = np.array([pair in record.cells.get(name, ()) for pair in pairs], dtype=bool)
    flags = _coefficient_flags(groups, specs, interaction_specs, record)
    group = _term_group(groups, name, len(spec._pairs))
    if flags is not None and group is not None:
        column = {pair: group.start + j for j, pair in enumerate(spec._pairs)}
        recorded |= np.array([pair in column and flags[column[pair]] for pair in pairs])
    return recorded if recorded.any() else None


def _design_record(model, y: NDArray, sample_weight: NDArray) -> _Record:
    """The scan's rule applied to the design's own level codes (``separation="ignore"``).

    The scan does not run under ``"ignore"``, so there is no record; a categorical
    group's codes name each row's level, the base's rows in the sink bin past the last
    column, and two parents' codes name a crossed term's cell, base row and column
    included.
    """
    from superglm.diagnostics.separation import _separated_flags, response_boundaries
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import CategoricalInteraction

    boundaries = response_boundaries(model._distribution, model._link)
    weights = np.asarray(sample_weight, dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    coded: dict[str, tuple[NDArray, list]] = {}
    for group, matrix in zip(model._groups, model._dm.group_matrices, strict=True):
        spec = model._specs.get(group.feature_name)
        codes = getattr(matrix, "codes", None)
        if (
            isinstance(spec, Categorical)
            and group.size == len(spec._non_base)
            and codes is not None
            and len(codes) == len(response)
        ):
            coded[group.feature_name] = (np.asarray(codes), [*spec._non_base, spec._base_level])

    def separated(codes: NDArray, n_cells: int) -> NDArray:
        flags = np.zeros(n_cells, dtype=bool)
        for boundary in boundaries:
            flags |= _separated_flags(codes, n_cells, response, weights, boundary)[0]
        return flags

    levels = {}
    for name, (codes, labels) in coded.items():
        held = frozenset(labels[i] for i in np.flatnonzero(separated(codes, len(labels))))
        if held:
            levels[name] = held
    cells, occupied = {}, {}
    for name, spec in model._interaction_specs.items():
        if not isinstance(spec, CategoricalInteraction):
            continue
        parents = [coded.get(parent) for parent in spec.parent_names]
        if any(parent is None for parent in parents):
            continue
        (codes1, labels1), (codes2, labels2) = parents
        n2 = len(labels2)
        codes = codes1 * n2 + codes2
        held = separated(codes, len(labels1) * n2)
        live = np.bincount(codes[weights > 0], minlength=len(labels1) * n2) > 0
        if held.any():
            cells[name] = frozenset(
                (labels1[i // n2], labels2[i % n2]) for i in np.flatnonzero(held)
            )
        occupied[name] = frozenset(
            (labels1[i // n2], labels2[i % n2]) for i in np.flatnonzero(live)
        )
    return _Record(levels=levels, cells=cells, occupied=occupied)


def separated_columns(model, y: NDArray, sample_weight: NDArray) -> set[int]:
    """The coefficients a separated level or cell moves, which its own disclosure covers.

    Read from the scan's record by the readers' rule (``separated_coefficients``).
    Under ``separation="ignore"`` there is no record, so the scan's rule is applied to
    the design's level codes (``_design_record``), base levels and a crossed term's
    base row and column included; their notices are silenced there.
    """
    ignore = getattr(model, "_separation", "warn") == "ignore"
    record = (
        _design_record(model, y, sample_weight)
        if ignore
        else _spec_record(model._specs, model._interaction_specs)
    )
    flags = _coefficient_flags(model._groups, model._specs, model._interaction_specs, record)
    return set() if flags is None else {int(j) for j in np.flatnonzero(flags[:-1])}


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
    "level_flags",
    "pair_flags",
    "predicted_cells",
    "separated_coefficients",
    "separated_columns",
    "separated_labels",
    "separated_levels_message",
    "separated_terms",
]
