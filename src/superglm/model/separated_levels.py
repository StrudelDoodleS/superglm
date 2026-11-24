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


def record_separated_levels(spec: Any, findings=()) -> None:
    """Replace the categorical fit record with the typed labels from this build."""
    spec._separated_levels_at_fit = tuple(label for finding in findings for label in finding.labels)


def predicted_separated_levels(name, spec, values: NDArray) -> str | None:
    """Name the recorded levels that the fitted categorizer actually scores."""
    met = spec._predicted_levels(values)
    named = [label for label in separated_labels(spec) if label in met]
    return f"{name!r} levels {', '.join(map(str, named))}" if named else None


def warn_separated_predictions(hits: list[str]) -> None:
    """Send one advisory for all of a prediction call's affected terms."""
    if hits:
        warn_caller(
            "These rows reach categorical levels or crossed cells whose training rows all had "
            "a response of 0, or all of 1, so the data give their effect no finite estimate: "
            + "; ".join(hits)
            + ". Their predictions use the fitted value, which is where the fit stopped, not "
            "an estimate. Group those levels with similar levels, or fit the term as a "
            "RandomEffect, and refit to price them.",
            SeparationWarning,
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
# reach: the term, its categorical parents (a crossed term's), and the intercept.
# ``Categorical`` and ``CategoricalInteraction`` columns carry no quadratic penalty,
# and a term a selection penalty targets is exempt from the scan, so a block's penalty
# adds no rows.  Coordinates run over ``[beta, intercept]``: the intercept is index
# ``p``.  A treatment-coded level's block is read from its labels: beside a healthy
# base, each separated level's own column is a direction on its own; a separated base
# adds the one direction that moves the intercept against every healthy level.  A
# crossed term's block is read from its healthy occupied cells' treatment-coded
# structure, a bipartite graph of its levels (``_cross_null``).  A reader builds the blocks once
# (``separation_flags``) and tests only the terms a block holds; no other term's
# levels are read.


@dataclass(frozen=True)
class _Record:
    """The separated labels a fit's readers test against, by term.

    ``occupied`` holds a crossed term's occupied cells, base row and column included.  A
    model saved before the build recorded them holds ``None`` there, and its separated
    cells give no flags: the readers keep the figures they had until it is refitted.
    """

    levels: dict[str, frozenset]
    cells: dict[str, frozenset]
    occupied: dict[str, frozenset | None]
    bounded: dict[str, tuple[bool, bool, bool]]


def _spec_record(specs: dict, interaction_specs: dict) -> _Record:
    """What the build-time scan recorded on the specs."""
    levels = {
        name: frozenset(getattr(spec, "_separated_levels_at_fit", ()) or ())
        for name, spec in specs.items()
    }
    occupied = {}
    for name, spec in interaction_specs.items():
        held = getattr(spec, "_occupied_cells_at_fit", None)
        occupied[name] = None if held is None else frozenset(map(tuple, held))
    cells = {
        name: frozenset(map(tuple, getattr(spec, "_separated_cells_at_fit", ()) or ()))
        for name, spec in interaction_specs.items()
        if occupied[name] is not None
    }
    bounded = {
        name: tuple(getattr(spec, "_bounded_groups_at_fit", (False, False, False)))
        for name, spec in interaction_specs.items()
    }
    return _Record(
        levels={name: held for name, held in levels.items() if held},
        cells={name: held for name, held in cells.items() if held},
        occupied=occupied,
        bounded=bounded,
    )


@dataclass(frozen=True)
class _Block:
    """One recorded term's separating directions over ``[beta, intercept]``.

    Each coordinate in ``free`` is one on its own; ``null`` is an orthonormal basis of
    the rest over the coordinates ``columns``.  ``terms`` names the terms it holds.
    """

    terms: frozenset
    free: NDArray
    columns: NDArray
    null: NDArray


@dataclass(frozen=True)
class SeparationFlags:
    """A fit's blocks (module comment above), built once for one reader call."""

    record: _Record
    blocks: tuple[_Block, ...]
    width: int  # the coordinates ``[beta, intercept]``

    def of(self, name: str) -> list[_Block]:
        """The blocks that hold term ``name``'s columns."""
        return [block for block in self.blocks if name in block.terms]

    def coefficients(self) -> NDArray:
        """Over ``[beta, intercept]``: each coefficient a separating direction moves."""
        from superglm.model.estimability_ops import PREDICTION_TOLERANCE

        flags = np.zeros(self.width, dtype=bool)
        for block in self.blocks:
            flags[block.free] = True
            flags[block.columns] |= np.linalg.norm(block.null, axis=1) > PREDICTION_TOLERANCE
        return flags


def _term_group(groups, name: str, width: int):
    """``name``'s one group when it is ``width`` columns wide, else ``None``."""
    found = [group for group in groups if group.feature_name == name]
    return found[0] if len(found) == 1 and found[0].size == width else None


_NONE = np.zeros(0, dtype=np.intp)


def _level_block(name: str, spec, group, held: frozenset, intercept: int) -> _Block | None:
    """A treatment-coded term's block, from its labels (module comment above)."""
    at = {level: group.start + j for j, level in enumerate(spec._non_base)}
    free = np.array([at[level] for level in held if level in at], dtype=np.intp)
    columns, null = _NONE, np.zeros((0, 1))
    if spec._base_level in held:
        # no base row: the intercept less every healthy level's column moves no healthy row
        healthy = [at[level] for level in spec._non_base if level not in held]
        columns = np.array([intercept, *healthy], dtype=np.intp)
        null = np.full((len(columns), 1), -1.0 / np.sqrt(len(columns)))
        null[0, 0] = -null[0, 0]
    return _Block(frozenset({name}), free, columns, null) if free.size or columns.size else None


def _cross_null(
    node_row: NDArray,
    node_col: NDArray,
    cross: NDArray,
    n1: int,
    n2: int,
    width: int,
    bounded: tuple[bool, bool, bool] = (False, False, False),
    sep_row: NDArray | None = None,
    sep_col: NDArray | None = None,
):
    """``(free, kept, N)``: the null space of a crossed block's healthy indicator rows.

    Coordinates: ``0`` the intercept, ``1..n1`` and the next ``n2`` the parents', then the
    cross's.  A healthy cell has ones at ``0``, ``node_row`` (``0`` on the base row),
    ``node_col`` (``1 + n1 + n2`` on the base column) and its column ``cross`` (``-1``:
    none).  Pinning constraints ``bounded = (pin_a, pin_b, pin_c)`` merge pinned parent
    nodes to their base and treat all cross columns as absent under ``pin_c``.  When
    ``sep_row`` and ``sep_col`` are supplied, only components touched by at least one
    separated cell whose endpoints lie in distinct components are kept, excluding pure
    collinearity directions in ``ker X_all`` (e.g. under ``alias_prune=False``).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from superglm.diagnostics.separation import _pinning_edges

    offset = 1 + n1 + n2  # the first cross column, and the base column's node
    pin_c = bounded[2]
    edge = np.ones(len(cross), dtype=bool) if pin_c else (cross < 0)
    pin_r, pin_col = _pinning_edges(n1, n2, bounded)
    rows = np.concatenate([pin_r, node_row[edge]])
    cols = np.concatenate([pin_col, node_col[edge]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(offset + 1, offset + 1))
    _, label = connected_components(graph, directed=False)
    if pin_c:
        free = _NONE
    else:
        held = np.zeros(width, dtype=bool)
        held[cross[~edge]] = True
        free = np.flatnonzero(~held[offset:]) + offset
    if sep_row is not None and sep_col is not None:
        sr, sc = label[sep_row], label[sep_col]
        diff = sr != sc
        components = np.setdiff1d(np.union1d(sr[diff], sc[diff]), label[offset])
    else:
        components = np.setdiff1d(np.unique(label), label[offset])
    if not components.size:
        return free, _NONE, np.zeros((0, 0))
    on = label[:, None] == components[None, :]  # node x component
    x, y = on[: n1 + 1].astype(float), -on[n1 + 1 : offset].astype(float)
    directions = np.zeros((width, components.size))
    directions[0] = x[0]
    directions[1 : n1 + 1] = x[1:] - x[0]
    directions[n1 + 1 : offset] = y
    cells = ~edge
    if cells.any():
        y_at = np.vstack([y, np.zeros((1, components.size))])  # the base column's y is 0
        directions[cross[cells]] = -(x[node_row[cells]] + y_at[node_col[cells] - n1 - 1])
    kept = np.flatnonzero(np.any(directions != 0.0, axis=1))
    return free, kept, np.linalg.qr(directions[kept])[0]


def _cross_block(name: str, spec, specs: dict, groups, record: _Record, intercept: int):
    """A crossed term's block, from its healthy occupied cells (``_cross_null``)."""
    from superglm.features.categorical import Categorical

    first, second = (specs.get(parent) for parent in spec.parent_names)
    if not (isinstance(first, Categorical) and isinstance(second, Categorical)):
        return None
    parts = [
        _term_group(groups, parent, len(parent_spec._non_base))
        for parent, parent_spec in zip(spec.parent_names, (first, second), strict=True)
    ]
    cross = _term_group(groups, name, len(spec._pairs))
    if any(group is None for group in parts) or cross is None:
        return None
    group1, group2 = parts
    offset = 1 + group1.size + group2.size
    at1 = {level: 1 + j for j, level in enumerate(first._non_base)}
    at2 = {level: 1 + group1.size + j for j, level in enumerate(second._non_base)}
    index1 = {level: i for i, level in enumerate(first._levels)}
    index2 = {level: j for j, level in enumerate(second._levels)}
    occupied = record.occupied[name]
    if occupied is None:  # saved before the build recorded them: no flags (``_Record``)
        return None
    held = record.cells[name]
    # a row of a separated parent level is not a healthy row of the cross either
    out1, out2 = (
        {*record.levels.get(spec.parent_names[0], ()), *first._pinned_levels},
        {*record.levels.get(spec.parent_names[1], ()), *second._pinned_levels},
    )

    def mask(cells) -> NDArray:
        found = np.zeros((len(index1), len(index2)), dtype=bool)
        for a, b in cells:
            if a in index1 and b in index2:
                found[index1[a], index2[b]] = True
        return found

    occ = mask(occupied)
    healthy = occ & ~mask(held)
    healthy[[index1[a] for a in out1 if a in index1]] = False
    healthy[:, [index2[b] for b in out2 if b in index2]] = False
    column = np.full(healthy.shape, -1, dtype=np.intp)
    grid = spec._all_pairs or spec._pairs
    emitted = np.arange(len(grid)) if spec._grid_to_col is None else spec._grid_to_col
    for (a, b), g in zip(grid, emitted, strict=True):
        if g >= 0:
            column[index1[a], index2[b]] = offset + g
    row_nodes = np.array([at1.get(a, 0) for a in first._levels], dtype=np.intp)
    col_nodes = np.array([at2.get(b, offset) for b in second._levels], dtype=np.intp)
    i, j = np.nonzero(healthy)
    si, sj = np.nonzero(occ & ~healthy)
    columns = np.array(
        [
            intercept,
            *range(group1.start, group1.end),
            *range(group2.start, group2.end),
            *range(cross.start, cross.end),
        ],
        dtype=np.intp,
    )
    bounded = record.bounded.get(name, (False, False, False))
    free, kept, null = _cross_null(
        row_nodes[i],
        col_nodes[j],
        column[i, j],
        group1.size,
        group2.size,
        len(columns),
        bounded,
        row_nodes[si],
        col_nodes[sj],
    )
    if not (free.size or kept.size):
        return None
    return _Block(frozenset({name, *spec.parent_names}), columns[free], columns[kept], null)


def separation_flags(
    groups, specs: dict, interaction_specs: dict, record: _Record | None = None
) -> SeparationFlags | None:
    """The blocks a fit's record gives its readers; ``None`` when nothing is recorded.

    A reader that tests several terms, or one term twice, builds this once and passes it.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import CategoricalInteraction

    record = _spec_record(specs, interaction_specs) if record is None else record
    if not (record.levels or record.cells):
        return None
    intercept = max((group.end for group in groups), default=0)
    blocks = []
    for name, held in record.levels.items():
        spec = specs.get(name)
        group = (
            _term_group(groups, name, len(spec._non_base))
            if isinstance(spec, Categorical)
            else None
        )
        block = None if group is None else _level_block(name, spec, group, held, intercept)
        if block is not None:
            blocks.append(block)
    for name in record.cells:
        spec = interaction_specs.get(name)
        block = (
            _cross_block(name, spec, specs, groups, record, intercept)
            if isinstance(spec, CategoricalInteraction)
            else None
        )
        if block is not None:
            blocks.append(block)
    return SeparationFlags(record, tuple(blocks), intercept + 1)


def _moved(blocks: list[_Block], column: NDArray, weight: NDArray | None = None) -> NDArray:
    """Which of a term's reported contrasts a block's separating directions move.

    Row ``i``'s contrast is its coefficient ``e_column[i]`` (zero for ``-1``, no column),
    less ``sum_k weight[k] e_column[k]`` when ``weight`` is given (a centred report's
    mean).  Lenth's test per block, read from the block's rows at the term's columns, so
    no contrast matrix is formed.
    """
    from superglm.model.estimability_ops import PREDICTION_TOLERANCE

    has = column >= 0
    weight = np.zeros(len(column)) if weight is None else np.where(has, weight, 0.0)
    square = float(np.sum(weight**2))
    size = np.sqrt(np.maximum(has * (1.0 - 2.0 * weight) + square, 0.0))
    moved = np.zeros(len(column), dtype=bool)
    for block in blocks:
        free = has & np.isin(column, block.free)
        share = free * (1.0 - 2.0 * weight) + float(np.sum(weight[free] ** 2))
        if block.columns.size:
            position = np.full(max(block.columns.max(), column.max(initial=-1)) + 1, -1)
            position[block.columns] = np.arange(block.columns.size)
            at = np.where(has, position[np.maximum(column, 0)], -1)
            reach = np.where((at >= 0)[:, None], block.null[np.maximum(at, 0)], 0.0)
            share = share + np.sum((reach - weight @ reach) ** 2, axis=1)
        moved |= np.sqrt(np.maximum(share, 0.0)) > PREDICTION_TOLERANCE * size
    return moved


def separated_coefficients(
    groups, specs: dict, interaction_specs: dict, *, flags: SeparationFlags | None = None
) -> NDArray | None:
    """Which coefficients' Wald statistics a recorded separated level or cell voids.

    Over ``[beta, intercept]`` (the intercept last); ``None`` when nothing is recorded.
    A coefficient is its own contrast ``e_j``: a separated level's own, every
    contrast against a separated base, a parent's main effect beside a separated cell
    in its base row or column, the intercept when it moves.
    """
    flags = separation_flags(groups, specs, interaction_specs) if flags is None else flags
    return None if flags is None else flags.coefficients()


def level_flags(
    groups,
    specs: dict,
    interaction_specs: dict,
    name: str,
    levels,
    *,
    centering: str = "native",
    flags: SeparationFlags | None = None,
) -> tuple[NDArray | None, NDArray | None]:
    """``(flags, centre)`` for a ``Categorical``'s reported levels, each ``None`` when moot.

    ``flags`` marks the levels reported blank: a recorded level, and a level whose
    reported contrast a separating direction moves (against the base, or against the
    mean under ``centering="mean"``).  ``centre`` marks the levels a mean-centred
    report centres on: those whose contrast against the base is determined.  A
    separated base moves every contrast against it, so the centre is then the levels
    not recorded, and every centred contrast is reported blank as well: their
    covariance is a difference of entries the walk has grown to about ``1e14``.
    ``flags=`` takes the reader's ``separation_flags``.
    """
    from superglm.features.categorical import Categorical

    spec = specs.get(name)
    if not isinstance(spec, Categorical):
        return None, None
    flags = separation_flags(groups, specs, interaction_specs) if flags is None else flags
    held = frozenset() if flags is None else flags.record.levels.get(name, frozenset())
    blocks = [] if flags is None else flags.of(name)
    if not (held or blocks):
        return None, None
    levels = list(levels)
    recorded = np.fromiter((level in held for level in levels), dtype=bool, count=len(levels))
    group = _term_group(groups, name, len(spec._non_base))
    at = {} if group is None else {level: group.start + j for j, level in enumerate(spec._non_base)}
    column = np.fromiter((at.get(level, -1) for level in levels), dtype=np.intp, count=len(levels))
    against_base = recorded | _moved(blocks, column)
    centre = ~against_base
    if not centre.any():
        centre = ~recorded
    centre_on = centre if centre.any() and not centre.all() else None
    if centering != "mean":
        out = against_base
    elif spec._base_level in held:
        out = np.ones(len(levels), dtype=bool)
    else:
        mean_over = centre if centre_on is not None else np.ones(len(levels), dtype=bool)
        out = recorded | _moved(blocks, column, mean_over / np.count_nonzero(mean_over))
    return (out if out.any() else None), centre_on


def pair_flags(
    groups,
    specs: dict,
    interaction_specs: dict,
    name: str,
    pairs,
    *,
    flags: SeparationFlags | None = None,
) -> NDArray | None:
    """Which of a crossed term's reported cells (``pairs``, the full non-base grid) are blank.

    A recorded cell, and a cell whose interaction coefficient a separating direction
    moves; a pruned cell's coefficient is zero by construction.  ``None`` when none is.
    """
    from superglm.features.interaction import CategoricalInteraction

    spec = interaction_specs.get(name)
    if not isinstance(spec, CategoricalInteraction):
        return None
    flags = separation_flags(groups, specs, interaction_specs) if flags is None else flags
    held = frozenset() if flags is None else flags.record.cells.get(name, frozenset())
    blocks = [] if flags is None else flags.of(name)
    if not (held or blocks):
        return None
    pairs = [tuple(pair) for pair in pairs]
    recorded = np.array([pair in held for pair in pairs], dtype=bool)
    group = _term_group(groups, name, len(spec._pairs))
    if blocks and group is not None:
        position = {pair: group.start + j for j, pair in enumerate(spec._pairs)}
        column = np.fromiter((position.get(pair, -1) for pair in pairs), np.intp, len(pairs))
        recorded |= _moved(blocks, column)
    return recorded if recorded.any() else None


def marked_cells(model, name: str) -> list[str]:
    """Crossed term ``name``'s cells, as ``"a x b"``, whose own coefficient ``pair_flags`` flags.

    A cell with no coefficient (base row or column, pruned) reads 1 and is never marked.
    """
    pairs = [tuple(pair) for pair in model._interaction_specs[name]._pairs]
    flags = pair_flags(model._groups, model._specs, model._interaction_specs, name, pairs)
    if flags is None:
        return []
    return [f"{a} x {b}" for (a, b), flag in zip(pairs, flags, strict=True) if flag]


def separated_sources(model, name: str) -> list[str]:
    """The recorded separated levels and cells whose directions reach term ``name``'s factors.

    A main effect's own levels and the cells of every crossed term it is a parent of; a
    crossed term's own cells and, beside them, its parents' levels (``_cross_block``).
    """
    record = _spec_record(model._specs, model._interaction_specs)

    def levels(term: str, of: str = "") -> list[str]:
        held = record.levels.get(term, ())
        return [
            f"level {level}{of}" for level in separated_labels(model._specs[term]) if level in held
        ]

    def cells(term: str, of: str = "") -> list[str]:
        held = record.cells.get(term, ())
        labels = separated_labels(model._interaction_specs[term])
        return [f"cell {a} x {b}{of}" for a, b in map(tuple, labels) if (a, b) in held]

    if name in model._interaction_specs:
        if name not in record.cells:
            return []
        parents = model._interaction_specs[name].parent_names
        return cells(name) + [
            label for parent in parents for label in levels(parent, f" of {parent!r}")
        ]
    crossed = [
        label
        for term, spec in model._interaction_specs.items()
        if name in getattr(spec, "parent_names", ()) and term in record.cells
        for label in cells(term, f" of {term!r}")
    ]
    return levels(name) + crossed


def _design_record(model, y: NDArray, sample_weight: NDArray) -> _Record:
    """The scan's rule applied to the design's own level codes (``separation="ignore"``).

    The scan does not run under ``"ignore"``, so there is no record; a categorical
    group's codes name each row's level, the base's rows in the sink bin past the last
    column, and two parents' codes name a crossed term's cell, base row and column
    included.
    """
    from superglm.diagnostics.separation import (
        _crossed_boundary_flags,
        _separated_flags,
        response_boundaries,
    )
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import CategoricalInteraction
    from superglm.model.fit_state import fitted_penalty
    from superglm.penalties.base import selection_shrunk_group_names

    boundaries = response_boundaries(model._distribution, model._link)
    weights = np.asarray(sample_weight, dtype=np.float64)
    response = np.asarray(y, dtype=np.float64)
    caps_enforced = (getattr(model, "_last_fit_meta", None) or {}).get("method") != "fit_path"
    shrunk = selection_shrunk_group_names(fitted_penalty(model), model._groups)

    def penalised(term: str) -> bool:
        return any(g.name in shrunk for g in model._groups if g.feature_name == term)

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

    levels = {}
    for name, (codes, labels) in coded.items():
        if (caps_enforced and model._specs[name]._constraint_rules) or penalised(name):
            continue
        flags = np.zeros(len(labels), dtype=bool)
        for boundary in boundaries:
            flags |= _separated_flags(codes, len(labels), response, weights, boundary)[0]
        if flags.any():
            levels[name] = frozenset(labels[i] for i in np.flatnonzero(flags))
    cells, occupied, bounded = {}, {}, {}
    for name, spec in model._interaction_specs.items():
        if not isinstance(spec, CategoricalInteraction):
            continue
        parents = [coded.get(parent) for parent in spec.parent_names]
        if any(parent is None for parent in parents):
            continue
        p1, p2 = spec.parent_names
        spec1, spec2 = model._specs[p1], model._specs[p2]
        b = (
            bool((caps_enforced and spec1._constraint_rules) or penalised(p1)),
            bool((caps_enforced and spec2._constraint_rules) or penalised(p2)),
            penalised(name),
        )
        bounded[name] = b
        (codes1, labels1), (codes2, labels2) = parents
        n2 = len(labels2)
        live, by_boundary = _crossed_boundary_flags(
            codes1 * n2 + codes2,
            labels1,
            labels2,
            spec1,
            spec2,
            response,
            weights,
            boundaries,
            spec,
            b,
        )
        held = (
            np.logical_or.reduce(list(by_boundary.values()))
            if by_boundary
            else np.zeros(len(live), dtype=bool)
        )
        if held.any():
            cells[name] = frozenset(
                (labels1[i // n2], labels2[i % n2]) for i in np.flatnonzero(held)
            )
        occupied[name] = frozenset(
            (labels1[i // n2], labels2[i % n2]) for i in np.flatnonzero(live)
        )
    return _Record(levels=levels, cells=cells, occupied=occupied, bounded=bounded)


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
    flags = separation_flags(model._groups, model._specs, model._interaction_specs, record)
    if flags is None:
        return set()
    return {int(j) for j in np.flatnonzero(flags.coefficients()[:-1])}


def _cell_codes(model, term: str, columns: dict[str, NDArray]) -> NDArray:
    """Each row's cell of crossed term ``term`` as one code, ``-1`` off its grid.

    Each parent's raw values resolve as its main effect prices them (grouping, then
    the unseen policy, a pinned or ``unseen="base"`` label at the base level:
    ``Categorical._scored_codes``).
    """
    first, second = model._interaction_specs[term].parent_names
    codes1 = model._specs[first]._scored_codes(columns[first])
    codes2 = model._specs[second]._scored_codes(columns[second])
    on_grid = (codes1 >= 0) & (codes2 >= 0)
    return np.where(on_grid, codes1 * len(model._specs[second]._levels) + codes2, -1)


def predicted_cells(model, term: str, frame) -> set[tuple]:
    """The cells of crossed term ``term`` the rows of ``frame`` are scored at (``_cell_codes``).

    ``predict`` has already reported unseen labels.
    """
    parents = model._interaction_specs[term].parent_names
    codes = _cell_codes(model, term, {parent: frame.column_array(parent) for parent in parents})
    levels1, levels2 = (model._specs[parent]._levels for parent in parents)
    n2 = len(levels2)
    return {(levels1[code // n2], levels2[code % n2]) for code in np.unique(codes[codes >= 0])}


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
    itself separated.
    """
    spec = specs[line.term]
    if line.is_base or line.base in separated_labels(spec):
        return None
    names = [line.term] if line.kind == "levels" else list(spec.parent_names)
    values = {name: _raw_base(specs[name]) for name in names}
    return None if any(value is None for value in values.values()) else values


def _reaches_unpriced_cell(
    model, found: SeparatedLevels, moved: dict, span: slice, names: set, cache: dict
) -> bool:
    """Whether ``moved``'s rows in ``span`` meet an empty or separated crossed cell.

    Moving the columns ``names`` to their base can take a row into an unoccupied or
    separated cell, whose rate the data do not determine.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.interaction import CategoricalInteraction

    for term, spec in model._interaction_specs.items():
        if not (
            isinstance(spec, CategoricalInteraction)
            and names & set(spec.parent_names)
            and all(isinstance(model._specs.get(name), Categorical) for name in spec.parent_names)
        ):
            continue
        if term not in cache:
            live = found.sample_weight > 0
            held = {parent: found.frame.column_array(parent)[live] for parent in spec.parent_names}
            priced = np.unique(_cell_codes(model, term, held))
            sep_cells = separated_labels(spec)
            if sep_cells:
                p1, p2 = spec.parent_names
                idx1 = {level: i for i, level in enumerate(model._specs[p1]._levels)}
                idx2 = {level: j for j, level in enumerate(model._specs[p2]._levels)}
                n2 = len(idx2)
                unpriced = [
                    idx1[a] * n2 + idx2[b]
                    for a, b in map(tuple, sep_cells)
                    if a in idx1 and b in idx2
                ]
                if unpriced:
                    priced = np.setdiff1d(priced, unpriced)
            cache[term] = priced
        reached = _cell_codes(model, term, {name: moved[name][span] for name in spec.parent_names})
        if not np.isin(reached, cache[term]).all():
            return True
    return False


def _factor(model, line: SeparatedLine) -> float | None:
    """A separated level's fitted factor against its base, ``exp`` of its contrast, guarded.

    Stated on a log link for a level no crossed term involves: a cell, or a level an
    interaction also prices, has no one factor.
    """
    from superglm.inference._term_types import _safe_exp
    from superglm.links import LogLink

    if line.kind != "levels" or not isinstance(model._link, LogLink):
        return None
    crossed = (getattr(spec, "parent_names", ()) for spec in model._interaction_specs.values())
    if any(line.term in parents for parents in crossed):
        return None
    spec = model._specs[line.term]
    group = _term_group(model._groups, line.term, len(spec._non_base))
    if group is None or line.label not in spec._non_base:
        return None
    return float(_safe_exp(model.result.beta[group.start + list(spec._non_base).index(line.label)]))


def _priced(
    model, entries: list[tuple[SeparatedLine, NDArray]], found: SeparatedLevels
) -> list[SeparatedLine]:
    """``entries``' lines with the published fit's response totals over their rows.

    Every line's rows are scored through the fit's own prediction path, offset
    included and every other term kept (an interaction the level enters, a
    spline): as they are, for the fitted total, and with the term's label set to
    its base (``_at_base``), for the total expected at the base's rate.  Both are
    finite for every link and weight each row by its ``sample_weight``.  The
    expected total is omitted when the move takes a row into an empty or separated
    crossed cell (``_reaches_unpriced_cell``).  A total float64 cannot hold is
    stated so, and never stops the fit.
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
        for name, value in (values or {}).items():
            if moved[name] is columns[name]:
                moved[name] = columns[name].astype(object)
            base = np.empty(end - start, dtype=object)
            base.fill(value)  # one value per row, a tuple-valued level among them
            moved[name][start:end] = base
        spans.append((slice(start, end), values))
    cache: dict[str, NDArray] = {}
    priced = [
        values is not None
        and not _reaches_unpriced_cell(model, found, moved, span, set(values), cache)
        for span, values in spans
    ]
    weight = found.sample_weight[rows]
    with np.errstate(over="ignore", under="ignore"):
        fitted = weight * mean(columns)
        expected = weight * mean(moved) if any(priced) else None
        return [
            replace(
                line,
                fitted=float(np.sum(fitted[span])),
                expected=float(np.sum(expected[span])) if expected is not None and at else None,
                factor=_factor(model, line),
            )
            for (line, _), (span, _), at in zip(entries, spans, priced, strict=True)
        ]


def _reml_scale_note(model) -> str | None:
    """Estimated-dispersion REML clause when an estimated RandomEffect sits beside separated levels."""
    from superglm.features.random_effect import RandomEffect

    if getattr(model, "_reml_result", None) is None:
        return None
    if getattr(model._distribution, "scale_known", True):
        return None
    re_names = [
        name
        for name, spec in model._specs.items()
        if isinstance(spec, RandomEffect)
        and (spec._lambda_policy is None or spec._lambda_policy.mode == "estimate")
    ]
    if not re_names:
        return None
    joined = ", ".join(repr(name) for name in re_names)
    return (
        "With an estimated dispersion, keeping separated unpenalised levels shifts "
        f"the profiled REML scale and the REML-estimated variance component(s) of {joined} "
        "(larger when separated levels are many relative to positive-response rows)."
    )


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
    return format_separated_levels(
        priced,
        fitted=True,
        example=lines_example(priced, specs),
        reml_scale_note=_reml_scale_note(model),
    )


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
    "marked_cells",
    "pair_flags",
    "predicted_cells",
    "separated_coefficients",
    "separated_columns",
    "separated_labels",
    "separated_levels_message",
    "separated_sources",
    "separated_terms",
    "separation_flags",
]
