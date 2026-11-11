"""Immutable fitted prediction designs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import CompiledPredictor
from superglm.features.categorical import Categorical, unseen_routing_quiet
from superglm.features.factor_smooth import FactorSmooth, group_term_level_values
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    SplineCategorical,
    TensorInteraction,
)
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.random_effect import RandomEffect
from superglm.features.spline import (
    BSplineSmooth,
    CardinalCRSpline,
    CubicRegressionSpline,
    NaturalSpline,
    PSpline,
)
from superglm.types import GroupSlice


def _readonly_matrix(values: NDArray, *, name: str) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class JointPredictionDesign:
    """One owned local coefficient design per ordered family parameter."""

    parameter_names: tuple[str, ...]
    local: Mapping[str, NDArray[np.float64]]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_names, tuple) or not self.parameter_names:
            raise ValueError("parameter_names must be a non-empty tuple")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names must be unique")
        if not isinstance(self.local, Mapping):
            raise TypeError("local prediction designs must be a mapping")
        if tuple(self.local) != self.parameter_names:
            raise ValueError("prediction designs must follow parameter order")
        owned = {
            name: _readonly_matrix(self.local[name], name=name) for name in self.parameter_names
        }
        row_counts = {matrix.shape[0] for matrix in owned.values()}
        if len(row_counts) != 1:
            raise ValueError("prediction designs must share a row count")
        object.__setattr__(self, "local", MappingProxyType(owned))


def _term_indices(groups: tuple[GroupSlice, ...], feature_name: str) -> NDArray[np.intp]:
    pieces = [
        np.arange(group.start, group.end, dtype=np.intp)
        for group in groups
        if group.feature_name == feature_name
    ]
    if not pieces:
        raise RuntimeError(f"compiled predictor has no fitted group for {feature_name!r}")
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


def _as_contribution(values: Any, *, n_observations: int, term_name: str) -> NDArray:
    contribution = np.asarray(values, dtype=np.float64).ravel()
    if contribution.shape != (n_observations,) or not np.all(np.isfinite(contribution)):
        raise ValueError(f"prediction contribution for {term_name!r} is invalid")
    return contribution


def _score_feature(spec: Any, values: NDArray, coefficients: NDArray) -> NDArray:
    if hasattr(spec, "score"):
        return np.asarray(spec.score(values, coefficients), dtype=np.float64)
    transformed = spec.transform(values)
    return np.asarray(transformed @ coefficients, dtype=np.float64)


def warn_population_levels(named: Sequence[str]) -> None:
    """SuperGLM's prediction warning (``model.base``) for SuperLSS, once per call, on the caller's line."""
    if not named:
        return
    from superglm._warn import warn_caller
    from superglm.features.factor_smooth import SZ_POPULATION_PREDICTION

    warn_caller(SZ_POPULATION_PREDICTION + "; ".join(named) + ".")


def _folds_group_term(predictor: CompiledPredictor, interaction: Any) -> bool:
    """Whether an sz term reads its group's main term in this predictor into its levels."""
    return bool(getattr(interaction, "_folds_group_term", False)) and (
        interaction.group in predictor.compiled.feature_order
    )


def group_level_values(
    predictor: CompiledPredictor, interaction: Any, coefficients: NDArray
) -> NDArray | None:
    """A term on an sz term's group in the same predictor: its value at each of the term's levels.

    SuperLSS accepts such a term as redundant; where it leaves its per-level values unpenalized they alias the sz term's level
    constants, so the sz rule reads them as part of each level's line
    (``FactorSmooth._score_identified``).  ``coefficients`` are that term's;
    ``None`` without such a term, and for every sz term on the group but the
    one that folds it (``share_group_levels``): the group term's values are
    folded once.  A level the group term does not know takes its unseen
    value (``group_term_level_values``).
    """
    if not _folds_group_term(predictor, interaction):
        return None
    spec = predictor.compiled.specs[interaction.group]
    return group_term_level_values(
        lambda labels: _score_feature(spec, labels, coefficients), interaction._levels
    )


def _score_interaction(
    spec: Any,
    left: NDArray,
    right: NDArray,
    coefficients: NDArray,
    *,
    named: list | None = None,
    level_constants: NDArray | None = None,
) -> NDArray:
    """One interaction term's contribution; ``named`` collects the levels it predicts at the population.

    ``level_constants``: a redundant group term's value at each of an sz
    term's levels (``group_level_values``).
    """
    if getattr(spec, "_has_population_offset", False):
        # An sz term with a level left out of its population (#432, #457):
        # scored as SuperGLM predicts it.
        values, levels = spec._score_identified(
            left, right, coefficients, population=False, level_constants=level_constants
        )
        if named is not None:
            named.extend(levels)
        return np.asarray(values, dtype=np.float64)
    if hasattr(spec, "score"):
        return np.asarray(spec.score(left, right, coefficients), dtype=np.float64)
    transformed = spec.transform(left, right)
    return np.asarray(transformed @ coefficients, dtype=np.float64)


def _feature_design(
    spec: Any,
    values: NDArray,
    *,
    width: int,
    n_observations: int,
    term_name: str,
) -> NDArray[np.float64]:
    identity = np.eye(width, dtype=np.float64)
    return np.column_stack(
        tuple(
            _as_contribution(
                _score_feature(spec, values, identity[:, index]),
                n_observations=n_observations,
                term_name=term_name,
            )
            for index in range(width)
        )
    )


def _interaction_design(
    spec: Any,
    left: NDArray,
    right: NDArray,
    *,
    width: int,
    n_observations: int,
    term_name: str,
    named: list | None = None,
) -> NDArray[np.float64]:
    identity = np.eye(width, dtype=np.float64)
    return np.column_stack(
        tuple(
            _as_contribution(
                _score_interaction(spec, left, right, identity[:, index], named=named),
                n_observations=n_observations,
                term_name=term_name,
            )
            for index in range(width)
        )
    )


def _group_term_constants(
    predictors: Sequence[CompiledPredictor],
) -> dict[tuple[str, str], list[NDArray[np.float64]]]:
    """Each folding sz term's ``group_level_values`` per coefficient of its group's term.

    They depend on the terms and their levels, not on the rows, and ``at_each_level`` reads them
    inside ``warnings.catch_warnings``, which changes the process's filter list: they are read once
    per call on the caller's thread, never on a pool's helper. Keyed by ``(predictor, term)``.
    """
    found: dict[tuple[str, str], list[NDArray[np.float64]]] = {}
    for predictor in predictors:
        for name in predictor.compiled.interaction_order:
            interaction = predictor.compiled.interaction_specs[name]
            if _folds_group_term(predictor, interaction):
                indices = _term_indices(predictor.compiled.groups, interaction.group)
                identity = np.eye(len(indices), dtype=np.float64)
                found[(predictor.name, name)] = [
                    group_level_values(predictor, interaction, identity[:, column])
                    for column in range(len(indices))
                ]
    return found


def _add_group_term_columns(
    matrix: NDArray[np.float64],
    predictor: CompiledPredictor,
    interaction: Any,
    constants: Sequence[NDArray[np.float64]],
    left: NDArray,
    right: NDArray,
    *,
    width: int,
    intercept_width: int,
) -> None:
    """Add an sz term's dependence on a redundant group term's coefficients to that term's columns.

    The rule reads the group term's per-level values into each level's line
    (``constants``, ``_group_term_constants``), so the sz contribution is linear in the group
    term's coefficients too: one column of it per coefficient, at zero sz
    coefficients, added onto the group term's own design.
    """
    group_indices = _term_indices(predictor.compiled.groups, interaction.group)
    zeros = np.zeros(width, dtype=np.float64)
    for index, values in zip(group_indices, constants, strict=True):
        matrix[:, index + intercept_width] += _score_interaction(
            interaction, left, right, zeros, level_constants=values
        )


def _assign_term(
    matrix: NDArray[np.float64],
    assigned: NDArray[np.bool_],
    indices: NDArray[np.intp],
    values: NDArray[np.float64],
    *,
    term_name: str,
) -> None:
    if values.shape != (len(matrix), len(indices)):
        raise ValueError(f"prediction design for {term_name!r} has an invalid shape")
    if np.any(assigned[indices]):
        duplicates = indices[assigned[indices]].tolist()
        raise RuntimeError(
            f"prediction plan assigned local coefficient columns more than once: {duplicates}"
        )
    if np.all(np.diff(indices) == 1):
        matrix[:, indices[0] : indices[-1] + 1] = values
    else:
        matrix[:, indices] = values
    assigned[indices] = True


def _required_columns(predictors: Sequence[CompiledPredictor]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            name
            for predictor in predictors
            for name in (
                *predictor.compiled.feature_order,
                *(
                    parent
                    for interaction_name in predictor.compiled.interaction_order
                    for parent in predictor.compiled.interaction_specs[
                        interaction_name
                    ].parent_names
                ),
            )
        )
    )


#: Rows of design built at a time when a frame is streamed: 8192 rows of 204 columns are 13 MB.
DESIGN_BLOCK_ROWS = 8192

#: Rows from which a block that is not pooled holds BLAS at one thread. From there its ``x V``
#: product is large enough for BLAS to start its threads and too small to gain from them; below it
#: BLAS runs the product on one thread anyway, and a quote does not pay for the scope.
BLAS_ONE_THREAD_ROWS = 64

#: Terms whose ``score`` is ``transform(...) @ beta`` row for row, so one ``transform`` gives every
#: design column at once. Any other term (a custom spec, an ``sz`` term with levels at the population
#: curve or folding its group's main term) is read column by column through ``score``, which is exact
#: for any term linear in its coefficients.
_TRANSFORM_TERMS = (
    PSpline,
    BSplineSmooth,
    NaturalSpline,
    CubicRegressionSpline,
    CardinalCRSpline,
    Categorical,
    RandomEffect,
    Numeric,
    Polynomial,
    Piecewise,
    OrderedCategorical,
)
_TRANSFORM_PAIRS = (
    TensorInteraction,
    SplineCategorical,
    PolynomialCategorical,
    NumericCategorical,
    CategoricalInteraction,
    NumericInteraction,
    PolynomialInteraction,
)


#: Terms whose rows depend on their own covariate values alone: a row range of such a term is scored
#: apart from the rest, on any thread, and raises or warns of nothing a whole-frame call would not.
#: Label terms (a ``Categorical``, a random effect, an ordered or factor-smooth term) are not
#: among them: they check levels against the fitted universe and warn once for the frame.
ROW_INDEPENDENT_TERMS = (
    PSpline,
    BSplineSmooth,
    NaturalSpline,
    CubicRegressionSpline,
    CardinalCRSpline,
    Numeric,
    Polynomial,
    Piecewise,
    TensorInteraction,
    NumericInteraction,
    PolynomialInteraction,
)


def _label_column(spec: Any, column: NDArray) -> NDArray[np.intp]:
    """The design column of each row's level (``-1`` for none), after the spec's own checks."""
    if type(spec) is Categorical:
        resolved = spec._resolve_predict_labels(column)
        codes = spec._predict_codes(resolved)
        positions = {level: position for position, level in enumerate(spec._non_base)}
        lookup = np.fromiter(
            (positions.get(level, -1) for level in (*spec._levels, None)),
            dtype=np.intp,
            count=len(spec._levels) + 1,
        )
        return lookup[np.where(codes >= 0, codes, len(spec._levels))]
    return spec._prediction_codes(column)


def label_columns(
    frame: EagerFrame, predictors: Sequence[CompiledPredictor], *, quiet: bool = False
) -> dict[tuple[str, str], NDArray[np.intp]]:
    """Each ``Categorical`` and ``RandomEffect`` term's level column per row, coded once for the frame.

    Comparing and looking up object labels holds the GIL, and a block that did it for every
    term, ``c1`` once for each predictor that carries it, kept the pool's threads in single
    file. The column is read through the spec's own checks, so it raises and warns as a
    whole-frame ``score`` does (``quiet`` drops the unseen-level warning, for a caller whose
    whole-frame prediction gives it); the rows of a block are then a slice of it. Keyed by
    ``(predictor, term)``; a term that groups levels, or routes unseen ones to a group, is not
    here and is read from its labels. The frame keeps what it coded: a second call over it, the
    standard errors after the point predictions, reads the columns back, and a quiet entry
    never stands in for a call that is to warn. The key holds each level's type beside its value:
    ``False == 0``, yet pandas matches a bool only against bool levels, so equal universes of
    different types code a column differently.
    """
    found: dict[tuple[str, str], NDArray[np.intp]] = {}
    memo = frame._label_columns
    for predictor in predictors:
        for name in predictor.compiled.feature_order:
            spec = predictor.compiled.specs[name]
            if (
                type(spec) is Categorical
                and spec._grouping is None
                and spec.unseen in ("error", "base")
            ):
                key: tuple = (
                    name,
                    "label",
                    spec.unseen,
                    tuple(spec._levels),
                    tuple(map(type, spec._levels)),
                    tuple(spec._non_base),
                )
            elif type(spec) is RandomEffect:
                key = (
                    name,
                    "random effect",
                    spec.unseen,
                    tuple(spec._levels),
                    tuple(map(type, spec._levels)),
                )
            else:
                continue
            entry = memo.get(key)
            if entry is None or (entry[1] and not quiet):
                with unseen_routing_quiet() if quiet else nullcontext():
                    memo[key] = (_label_column(spec, frame.column_array(name)), quiet)
            found[(predictor.name, name)] = memo[key][0]
    return found


def _term_rows(values: Any, *, n_observations: int, width: int, term_name: str) -> NDArray:
    block = values.toarray() if sp.issparse(values) else np.asarray(values, dtype=np.float64)
    if block.shape != (n_observations, width):
        raise ValueError(f"prediction design for {term_name!r} has an invalid shape")
    if not np.all(np.isfinite(block)):
        raise ValueError(f"prediction contribution for {term_name!r} is invalid")
    return block


def _feature_rows(spec: Any, values: NDArray, *, width: int, term_name: str) -> NDArray:
    if type(spec) in _TRANSFORM_TERMS:
        return _term_rows(
            spec.transform(values), n_observations=len(values), width=width, term_name=term_name
        )
    return _feature_design(
        spec, values, width=width, n_observations=len(values), term_name=term_name
    )


def _interaction_rows(
    predictor: CompiledPredictor,
    interaction: Any,
    left: NDArray,
    right: NDArray,
    *,
    width: int,
    term_name: str,
    named: list | None,
) -> NDArray:
    rows_from_transform = type(interaction) in _TRANSFORM_PAIRS or (
        type(interaction) is FactorSmooth
        and not interaction._has_population_offset
        and not _folds_group_term(predictor, interaction)
    )
    if rows_from_transform:
        return _term_rows(
            interaction.transform(left, right),
            n_observations=len(left),
            width=width,
            term_name=term_name,
        )
    return _interaction_design(
        interaction,
        left,
        right,
        width=width,
        n_observations=len(left),
        term_name=term_name,
        named=named,
    )


def _predictor_rows(
    predictor: CompiledPredictor,
    state: Any,
    columns: Mapping[Any, NDArray],
    labels: Mapping[tuple[str, str], NDArray],
    folded: Mapping[tuple[str, str], Sequence[NDArray[np.float64]]],
    rows: slice,
    named: list[str] | None,
) -> NDArray[np.float64]:
    """One predictor's local design for the frame rows ``rows``.

    ``columns`` holds each covariate's column by its name, whatever the name is, and ``labels``
    the level columns of ``label_columns`` by ``(predictor, term)``: two mappings, because a
    tuple-valued column name could equal a ``(predictor, term)`` key. ``folded`` is
    ``_group_term_constants``."""
    n_rows = rows.stop - rows.start
    width = state.coefficient_slice.stop - state.coefficient_slice.start
    intercept_width = int(state.intercept_index is not None)
    if predictor.intercept != bool(intercept_width):
        raise ValueError("compiled predictor and layout intercept state disagree")
    matrix = np.zeros((n_rows, width), dtype=np.float64)
    assigned = np.zeros(width, dtype=np.bool_)
    if intercept_width:
        matrix[:, 0] = 1.0
        assigned[0] = True

    for name in predictor.compiled.feature_order:
        slope_indices = _term_indices(predictor.compiled.groups, name)
        coded = labels.get((predictor.name, name))
        if coded is not None:
            hot = coded[rows]
            values = np.zeros((n_rows, len(slope_indices)), dtype=np.float64)
            present = np.flatnonzero(hot >= 0)
            values[present, hot[present]] = 1.0
        else:
            values = _feature_rows(
                predictor.compiled.specs[name],
                columns[name][rows],
                width=len(slope_indices),
                term_name=name,
            )
        _assign_term(matrix, assigned, slope_indices + intercept_width, values, term_name=name)

    for name in predictor.compiled.interaction_order:
        interaction = predictor.compiled.interaction_specs[name]
        slope_indices = _term_indices(predictor.compiled.groups, name)
        left_name, right_name = interaction.parent_names
        left, right = columns[left_name][rows], columns[right_name][rows]
        levels: list = []
        values = _interaction_rows(
            predictor,
            interaction,
            left,
            right,
            width=len(slope_indices),
            term_name=name,
            named=levels,
        )
        if levels and named is not None:
            listed = ", ".join(dict.fromkeys(map(str, levels)))
            named.append(f"{predictor.name} term {name!r} levels {listed}")
        _assign_term(matrix, assigned, slope_indices + intercept_width, values, term_name=name)
        constants = folded.get((predictor.name, name))
        if constants is not None:
            _add_group_term_columns(
                matrix,
                predictor,
                interaction,
                constants,
                left,
                right,
                width=len(slope_indices),
                intercept_width=intercept_width,
            )

    if not np.all(assigned):
        missing = np.flatnonzero(~assigned).tolist()
        raise RuntimeError(f"prediction plan did not assign local coefficient columns {missing}")
    return matrix


def _validated_predictors(
    compiled_predictors: Sequence[CompiledPredictor], layout: StackedLayout
) -> tuple[CompiledPredictor, ...]:
    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    predictors = tuple(compiled_predictors)
    if not predictors or not all(
        isinstance(predictor, CompiledPredictor) for predictor in predictors
    ):
        raise TypeError("compiled_predictors must contain CompiledPredictor values")
    predictor_names = tuple(predictor.name for predictor in predictors)
    layout_names = tuple(state.name for state in layout.predictors)
    if predictor_names != layout_names:
        raise ValueError("compiled predictors must follow layout parameter order")
    for parameter_index, predictor in enumerate(predictors):
        if predictor.parameter_index != parameter_index:
            raise ValueError("compiled predictors must follow family parameter order")
    return predictors


def map_design_blocks(
    X: FrameLike | EagerFrame,
    compiled_predictors: Sequence[CompiledPredictor],
    layout: StackedLayout,
    fn: Any,
    *,
    start: int = 0,
    stop: int | None = None,
    block_rows: int | None = None,
    blas_one_thread: bool = False,
) -> list:
    """``[fn(rows, {predictor: local design}), ...]`` over blocks of rows, in row order.

    The blocks do not depend on the worker count, so the list is the same at any ``n_jobs``.
    More than one block runs on ``superglm``'s own pool (``_parallel.pooled_for``, BLAS held at
    one thread, the calling thread working beside its helpers); one block, which is every
    quote-sized frame, runs inline and starts nothing. A frame holds one block's design, and
    what ``fn`` makes of it, per worker, the workers capped by ``max_memory``, never a matrix
    with a row per observation. ``fn`` reads its block and returns what it keeps. A block does
    not give the unseen-level warning, because it counts only its own rows (the caller's
    whole-frame prediction says it once); that is a context variable set here, so concurrent
    calls share no warning state. ``blas_one_thread`` holds BLAS at one thread for a single
    block of ``BLAS_ONE_THREAD_ROWS`` rows or more: for a caller whose ``fn`` is the ``x V``
    product of the standard errors, where BLAS starts threads it cannot use.
    """
    predictors = _validated_predictors(compiled_predictors, layout)
    frame = as_eager_frame(X)
    names = _required_columns(predictors)
    frame.require_columns(names)
    columns: dict[Any, NDArray] = {name: frame.column_array(name) for name in names}
    first, last = int(start), len(frame) if stop is None else int(stop)
    if not 0 <= first <= last <= len(frame):
        raise ValueError("start and stop must select rows of X")
    step = DESIGN_BLOCK_ROWS if block_rows is None else int(block_rows)
    if step < 1:
        raise ValueError("block_rows must be a positive number of rows")
    spans = [slice(lo, min(lo + step, last)) for lo in range(first, last, step)]
    labels = label_columns(frame, predictors, quiet=True)
    folded = _group_term_constants(predictors)

    def run(rows: slice) -> Any:
        local = {
            state.name: _predictor_rows(predictor, state, columns, labels, folded, rows, None)
            for predictor, state in zip(predictors, layout.predictors, strict=True)
        }
        return fn(rows, local)

    with unseen_routing_quiet():
        if len(spans) <= 1:
            from superglm._blas_threads import pooled_blas_threads

            hold = blas_one_thread and last - first >= BLAS_ONE_THREAD_ROWS
            with pooled_blas_threads() if hold else nullcontext():
                return [run(rows) for rows in spans]
        from superglm._parallel import pooled_for

        widest = max(
            state.coefficient_slice.stop - state.coefficient_slice.start
            for state in layout.predictors
        )
        # a block's design, its product with a covariance block and a copy, 8 bytes each
        return pooled_for(run, spans, step * widest * 8 * 3)


def build_predictor_design(
    X: FrameLike | EagerFrame, predictor: CompiledPredictor, state: Any
) -> NDArray[np.float64]:
    """One predictor's local design, reading each column only as its own terms read it.

    A column one parameter reads as a number and another as levels is read here only as
    ``predictor`` reads it, so sweeping it through one term never hands the other a value.
    """
    frame = as_eager_frame(X)
    predictors = (predictor,)
    frame.require_columns(_required_columns(predictors))
    if not len(frame):
        return np.zeros((0, state.coefficient_slice.stop - state.coefficient_slice.start))
    columns: dict[Any, NDArray] = {
        name: frame.column_array(name) for name in _required_columns(predictors)
    }
    return _predictor_rows(
        predictor,
        state,
        columns,
        label_columns(frame, predictors),
        _group_term_constants(predictors),
        slice(0, len(frame)),
        None,
    )


def build_joint_prediction_design(
    X: FrameLike | EagerFrame,
    compiled_predictors: Sequence[CompiledPredictor],
    layout: StackedLayout,
    named: list[str] | None = None,
) -> JointPredictionDesign:
    """Reconstruct local fitted coefficient designs without mutating term state.

    The whole frame as one matrix per predictor; ``map_design_blocks`` streams large frames.
    ``named``, when given, collects the ``sz`` levels the design predicts at the population
    curve, as ``"<parameter> term '<term>' levels ..."``.
    """
    predictors = _validated_predictors(compiled_predictors, layout)
    frame = as_eager_frame(X)
    frame.require_columns(_required_columns(predictors))
    columns: dict[Any, NDArray] = {
        name: frame.column_array(name) for name in _required_columns(predictors)
    }
    labels = label_columns(frame, predictors)
    folded = _group_term_constants(predictors)
    rows = slice(0, len(frame))
    local = {
        state.name: _predictor_rows(predictor, state, columns, labels, folded, rows, named)
        if len(frame)
        else np.zeros((0, state.coefficient_slice.stop - state.coefficient_slice.start))
        for predictor, state in zip(predictors, layout.predictors, strict=True)
    }
    return JointPredictionDesign(
        parameter_names=tuple(state.name for state in layout.predictors), local=local
    )
