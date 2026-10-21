"""Immutable fitted prediction designs and covariance propagation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import CompiledPredictor
from superglm.types import GroupSlice


def _readonly_matrix(values: NDArray, *, name: str) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    result.setflags(write=False)
    return result


def _readonly_vector(values: NDArray) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError("prediction standard error must be a finite vector")
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
    from superglm.features.factor_smooth import SZ_POPULATION_PREDICTION
    from superglm.profiling._scalar import warn_caller

    warn_caller(SZ_POPULATION_PREDICTION + "; ".join(named) + ".")


def group_level_values(
    predictor: CompiledPredictor, interaction: Any, coefficients: NDArray
) -> NDArray | None:
    """A term on an sz term's group in the same predictor: its value at each of the term's levels.

    SuperLSS accepts such a term as redundant (0.40 required one); where it
    leaves its per-level values unpenalized they alias the sz term's level
    constants, so the sz rule reads them as part of each level's line
    (``FactorSmooth._score_identified``).  ``coefficients`` are that term's;
    ``None`` without such a term or a level the rule moves.
    """
    group = getattr(interaction, "group", None)
    if not getattr(interaction, "_has_population_offset", False):
        return None
    if group not in predictor.compiled.feature_order or not interaction._folds_group_term:
        return None
    levels = np.asarray(list(interaction._levels), dtype=object)
    return np.asarray(
        _score_feature(predictor.compiled.specs[group], levels, coefficients), dtype=np.float64
    ).ravel()


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


def _add_group_term_columns(
    matrix: NDArray[np.float64],
    predictor: CompiledPredictor,
    interaction: Any,
    left: NDArray,
    right: NDArray,
    *,
    width: int,
    intercept_width: int,
) -> None:
    """Add an sz term's dependence on a redundant group term's coefficients to that term's columns.

    The rule reads the group term's per-level values into each level's line
    (``group_level_values``), so the sz contribution is linear in the group
    term's coefficients too: one column of it per coefficient, at zero sz
    coefficients, added onto the group term's own design.
    """
    group = getattr(interaction, "group", None)
    if not getattr(interaction, "_has_population_offset", False):
        return
    if group not in predictor.compiled.feature_order or not interaction._folds_group_term:
        return
    group_indices = _term_indices(predictor.compiled.groups, group)
    zeros = np.zeros(width, dtype=np.float64)
    identity = np.eye(len(group_indices), dtype=np.float64)
    for column, index in enumerate(group_indices):
        constants = group_level_values(predictor, interaction, identity[:, column])
        matrix[:, index + intercept_width] += _score_interaction(
            interaction, left, right, zeros, level_constants=constants
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


def build_joint_prediction_design(
    X: FrameLike | EagerFrame,
    compiled_predictors: Sequence[CompiledPredictor],
    layout: StackedLayout,
    named: list[str] | None = None,
) -> JointPredictionDesign:
    """Reconstruct local fitted coefficient designs without mutating term state.

    ``named``, when given, collects the ``sz`` levels the design predicts at
    the population curve, as ``"<parameter> term '<term>' levels ..."``.
    """

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

    frame = as_eager_frame(X)
    frame.require_columns(_required_columns(predictors))
    local: dict[str, NDArray[np.float64]] = {}
    for parameter_index, (predictor, state) in enumerate(
        zip(predictors, layout.predictors, strict=True)
    ):
        if predictor.parameter_index != parameter_index:
            raise ValueError("compiled predictors must follow family parameter order")
        width = state.coefficient_slice.stop - state.coefficient_slice.start
        intercept_width = int(state.intercept_index is not None)
        if predictor.intercept != bool(intercept_width):
            raise ValueError("compiled predictor and layout intercept state disagree")
        matrix = np.zeros((len(frame), width), dtype=np.float64)
        assigned = np.zeros(width, dtype=np.bool_)
        if intercept_width:
            matrix[:, 0] = 1.0
            assigned[0] = True

        for name in predictor.compiled.feature_order:
            slope_indices = _term_indices(predictor.compiled.groups, name)
            indices = slope_indices + intercept_width
            values = _feature_design(
                predictor.compiled.specs[name],
                frame.column_array(name),
                width=len(slope_indices),
                n_observations=len(frame),
                term_name=name,
            )
            _assign_term(matrix, assigned, indices, values, term_name=name)

        for name in predictor.compiled.interaction_order:
            interaction = predictor.compiled.interaction_specs[name]
            slope_indices = _term_indices(predictor.compiled.groups, name)
            indices = slope_indices + intercept_width
            left_name, right_name = interaction.parent_names
            levels: list = []
            values = _interaction_design(
                interaction,
                frame.column_array(left_name),
                frame.column_array(right_name),
                width=len(slope_indices),
                n_observations=len(frame),
                term_name=name,
                named=levels,
            )
            if levels and named is not None:
                listed = ", ".join(dict.fromkeys(map(str, levels)))
                named.append(f"{predictor.name} term {name!r} levels {listed}")
            _assign_term(matrix, assigned, indices, values, term_name=name)
            _add_group_term_columns(
                matrix,
                predictor,
                interaction,
                frame.column_array(left_name),
                frame.column_array(right_name),
                width=len(slope_indices),
                intercept_width=intercept_width,
            )

        if not np.all(assigned):
            missing = np.flatnonzero(~assigned).tolist()
            raise RuntimeError(
                f"prediction plan did not assign local coefficient columns {missing}"
            )
        local[state.name] = matrix

    return JointPredictionDesign(parameter_names=layout_names, local=local)


def link_standard_errors(
    design: JointPredictionDesign,
    covariance: NDArray,
    layout: StackedLayout,
) -> Mapping[str, NDArray[np.float64]]:
    """Propagate fixed-smoothing coefficient covariance onto each link predictor."""

    if not isinstance(design, JointPredictionDesign):
        raise TypeError("design must be a JointPredictionDesign")
    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    try:
        covariance_values = np.asarray(covariance, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("covariance must be a finite square coefficient matrix") from error
    expected_shape = (layout.n_coefficients, layout.n_coefficients)
    if covariance_values.shape != expected_shape or not np.all(np.isfinite(covariance_values)):
        raise ValueError("covariance must be a finite square coefficient matrix")
    layout_names = tuple(state.name for state in layout.predictors)
    if design.parameter_names != layout_names:
        raise ValueError("prediction design must follow layout parameter order")

    result: dict[str, NDArray[np.float64]] = {}
    for state in layout.predictors:
        matrix = design.local[state.name]
        width = state.coefficient_slice.stop - state.coefficient_slice.start
        if matrix.shape[1] != width:
            raise ValueError(f"prediction design width for {state.name!r} does not match layout")
        block = covariance_values[state.coefficient_slice, state.coefficient_slice]
        variance = np.einsum("ij,jk,ik->i", matrix, block, matrix, optimize=True)
        tolerance = (
            64.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.max(np.abs(variance), initial=0.0)),
            )
        )
        if np.any(variance < -tolerance):
            raise ValueError("prediction variance is materially negative")
        result[state.name] = _readonly_vector(np.sqrt(np.maximum(variance, 0.0)))
    return MappingProxyType(result)
