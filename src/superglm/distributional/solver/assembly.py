"""Dense and grouped assembly for joint distributional geometry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._block_queue import BlockTask, run_block_tasks
from superglm._group_matrix._group_matrix_algebra import (
    PassFactors,
    PassTables,
    _BlockWeightCache,
    _pass_table,
)
from superglm._group_matrix._group_matrix_core import OrdinaryPanelGroupMatrix
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.solver._formation import CurvatureFormation, dense_formation
from superglm.distributional.solver._ordinary_packing import PredictorPanel
from superglm.distributional.solver.packing import packed_pairs
from superglm.solvers.rank import _symmetric_part


def _readonly(values: NDArray) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DenseJointGeometry:
    """Score and data/penalized curvature in the global coefficient layout.

    ``formation`` holds the data curvature's formation statistics when the
    assembly recorded them, for the terminal rank certificate. A dense assembly
    defers them as a callable that :meth:`formed` runs on first use, so only
    the geometry the certificate reads pays for them.
    """

    score_data: NDArray[np.float64]
    score_penalized: NDArray[np.float64]
    data_curvature: NDArray[np.float64]
    penalty: NDArray[np.float64]
    penalized_curvature: NDArray[np.float64]
    formation: CurvatureFormation | Callable[[], CurvatureFormation] | None = field(
        default=None, compare=False, repr=False
    )

    def formed(self) -> CurvatureFormation | None:
        """The formation statistics, formed on first use if the assembly deferred them."""
        if callable(self.formation):
            object.__setattr__(self, "formation", self.formation())
        return self.formation

    def __post_init__(self) -> None:
        score_data = _readonly(self.score_data)
        score_penalized = _readonly(self.score_penalized)
        data_curvature = _readonly(self.data_curvature)
        penalty = _readonly(self.penalty)
        penalized = _readonly(self.penalized_curvature)
        width = len(score_data)
        if score_data.shape != (width,) or score_penalized.shape != (width,):
            raise ValueError("joint score arrays must be vectors")
        expected = (width, width)
        if any(matrix.shape != expected for matrix in (data_curvature, penalty, penalized)):
            raise ValueError("joint curvature arrays must share the score dimension")
        if not np.array_equal(data_curvature, data_curvature.T):
            raise ValueError("data curvature must be symmetric")
        if not np.array_equal(penalty, penalty.T):
            raise ValueError("penalty must be symmetric")
        if not np.array_equal(penalized, data_curvature + penalty):
            raise ValueError("penalized curvature must equal data curvature plus penalty")
        object.__setattr__(self, "score_data", score_data)
        object.__setattr__(self, "score_penalized", score_penalized)
        object.__setattr__(self, "data_curvature", data_curvature)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "penalized_curvature", penalized)


def with_formation(geometry, formation: CurvatureFormation | None):
    """Return a just-published ``geometry`` carrying ``formation``.

    The arrays are already validated and read-only, so they are not copied.
    Anything other than a ``DenseJointGeometry`` passes through unchanged.
    """
    if formation is not None and isinstance(geometry, DenseJointGeometry):
        object.__setattr__(geometry, "formation", formation)
    return geometry


def dense_predictor_matrices(layout: StackedLayout) -> tuple[NDArray[np.float64], ...]:
    """Return the dense design matrix of every predictor in a stacked layout."""

    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    if not layout.predictors:
        raise ValueError("layout must contain at least one predictor")
    n_observations = layout.predictors[0].design.n
    matrices: list[NDArray[np.float64]] = []
    for state in layout.predictors:
        if state.design.n != n_observations or state.offset.shape != (n_observations,):
            raise ValueError("predictor row counts and offsets must agree")
        if not np.all(np.isfinite(state.offset)):
            raise ValueError(f"offset for predictor {state.name!r} must be finite")
        slope_width = state.design.p
        block_width = state.coefficient_slice.stop - state.coefficient_slice.start
        intercept_width = int(state.intercept_index is not None)
        if block_width != intercept_width + slope_width:
            raise ValueError(f"coefficient slice does not match predictor {state.name!r} design")
        if state.intercept_index is not None:
            if state.intercept_index != state.coefficient_slice.start:
                raise ValueError("predictor intercept must be first in its coefficient slice")
        slopes = (
            np.zeros((n_observations, 0), dtype=np.float64)
            if slope_width == 0
            else np.asarray(state.design.toarray(), dtype=np.float64)
        )
        if slopes.shape != (n_observations, slope_width) or not np.all(np.isfinite(slopes)):
            raise ValueError(f"dense design for predictor {state.name!r} has invalid state")
        if state.intercept_index is not None:
            # column_stack owns fresh float64 storage, so freezing it needs no
            # further full-design copy. Keep the defensive copy for slopes alone.
            matrix = np.column_stack((np.ones(n_observations), slopes))
            if type(slopes) is np.ndarray and type(matrix) is np.ndarray and matrix.flags.owndata:
                matrix.setflags(write=False)
            else:
                matrix = _readonly(matrix)
        else:
            matrix = _readonly(slopes)
        matrices.append(matrix)
    return tuple(matrices)


def _validated_coefficients(
    coefficients: NDArray,
    layout: StackedLayout,
) -> NDArray[np.float64]:
    try:
        values = np.asarray(coefficients, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "coefficients must be a finite vector with the global layout shape"
        ) from exc
    if values.shape != (layout.n_coefficients,) or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be a finite vector with the global layout shape")
    return values


def _evaluate_predictors_from_matrices(
    layout: StackedLayout,
    coefficients: NDArray,
    matrices: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """Evaluate predictors from matrices already validated for this layout."""
    values = _validated_coefficients(coefficients, layout)
    if len(matrices) != len(layout.predictors):
        raise ValueError("one dense matrix is required per layout predictor")
    n_observations = matrices[0].shape[0]
    eta = np.empty((n_observations, len(matrices)), dtype=np.float64)
    for parameter_index, (state, matrix) in enumerate(
        zip(layout.predictors, matrices, strict=True)
    ):
        expected_shape = (
            n_observations,
            state.coefficient_slice.stop - state.coefficient_slice.start,
        )
        if matrix.shape != expected_shape:
            raise ValueError(
                f"dense matrix for predictor {state.name!r} has shape {matrix.shape}; "
                f"expected {expected_shape}"
            )
        eta[:, parameter_index] = matrix @ values[state.coefficient_slice] + state.offset
    return _readonly(eta)


def _validated_channels(
    score_eta: NDArray,
    curvature_packed: NDArray,
    *,
    n_observations: int,
    k_parameters: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        score = np.asarray(score_eta, dtype=np.float64)
        curvature = np.asarray(curvature_packed, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("score and curvature channels must be finite numeric arrays") from exc
    expected_score = (n_observations, k_parameters)
    expected_curvature = (n_observations, k_parameters * (k_parameters + 1) // 2)
    if score.shape != expected_score:
        raise ValueError(f"score_eta must have shape {expected_score}; got {score.shape}")
    if curvature.shape != expected_curvature:
        raise ValueError(
            f"curvature_packed must have shape {expected_curvature}; got {curvature.shape}"
        )
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(curvature)):
        raise ValueError("score and curvature channels must contain only finite values")
    return score, curvature


def validated_dense_penalty(penalty: NDArray, width: int) -> NDArray[np.float64]:
    """Return ``penalty`` as a finite symmetric matrix of the global layout width."""

    try:
        values = np.asarray(penalty, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("penalty must be a finite symmetric matrix") from exc
    if values.shape != (width, width):
        raise ValueError(
            f"penalty shape {values.shape} does not match global layout {(width, width)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("penalty must contain only finite values")
    tolerance = 1.0e-12 * max(1.0, float(np.linalg.norm(values, ord=np.inf)))
    if not np.allclose(values, values.T, rtol=0.0, atol=tolerance):
        raise ValueError("penalty must be symmetric")
    return _symmetric_part(values)


def _assemble_dense_geometry_from_matrices(
    layout: StackedLayout,
    matrices: tuple[NDArray[np.float64], ...],
    score_eta: NDArray,
    curvature_packed: NDArray,
    *,
    penalty: NDArray,
    coefficients: NDArray,
) -> DenseJointGeometry:
    """Assemble the dense reference from predictor matrices retained by a caller."""
    coefficient_values = _validated_coefficients(coefficients, layout)
    n_observations = matrices[0].shape[0]
    k_parameters = len(matrices)
    score, curvature = _validated_channels(
        score_eta,
        curvature_packed,
        n_observations=n_observations,
        k_parameters=k_parameters,
    )
    penalty_matrix = validated_dense_penalty(penalty, layout.n_coefficients)

    score_data = np.zeros(layout.n_coefficients, dtype=np.float64)
    data_curvature = np.zeros(
        (layout.n_coefficients, layout.n_coefficients),
        dtype=np.float64,
    )
    for state, matrix in zip(layout.predictors, matrices, strict=True):
        score_data[state.coefficient_slice] = matrix.T @ score[:, state.parameter_index]

    for channel_index, (left_index, right_index) in enumerate(packed_pairs(k_parameters)):
        left_state = layout.predictors[left_index]
        right_state = layout.predictors[right_index]
        weights = curvature[:, channel_index]
        block = matrices[left_index].T @ (weights[:, None] * matrices[right_index])
        data_curvature[left_state.coefficient_slice, right_state.coefficient_slice] = block
        if left_index != right_index:
            data_curvature[right_state.coefficient_slice, left_state.coefficient_slice] = block.T

    data_curvature = _symmetric_part(data_curvature)
    score_penalized = score_data - penalty_matrix @ coefficient_values
    penalized_curvature = data_curvature + penalty_matrix
    return DenseJointGeometry(
        score_data=score_data,
        score_penalized=score_penalized,
        data_curvature=data_curvature,
        penalty=penalty_matrix,
        penalized_curvature=penalized_curvature,
        formation=partial(dense_formation, layout, matrices, curvature),
    )


def _grouped_predictor_plans(
    layout: StackedLayout,
) -> tuple[PredictorExecutionPlan, ...]:
    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    if not layout.predictors:
        raise ValueError("layout must contain at least one predictor")
    n_observations = layout.predictors[0].design.n
    plans: list[PredictorExecutionPlan] = []
    for state in layout.predictors:
        if state.design.n != n_observations or state.offset.shape != (n_observations,):
            raise ValueError("predictor row counts and offsets must agree")
        if not np.all(np.isfinite(state.offset)):
            raise ValueError(f"offset for predictor {state.name!r} must be finite")
        intercept = state.intercept_index is not None
        block_width = state.coefficient_slice.stop - state.coefficient_slice.start
        if block_width != int(intercept) + state.design.p:
            raise ValueError(f"coefficient slice does not match predictor {state.name!r} design")
        if intercept and state.intercept_index != state.coefficient_slice.start:
            raise ValueError("predictor intercept must be first in its coefficient slice")
        plans.append(PredictorExecutionPlan(state.design, intercept))
    return tuple(plans)


def _accumulation_groups(plan: PredictorExecutionPlan, panel: PredictorPanel | None) -> list:
    """A predictor's groups in accumulation order: its panel (a placeholder) first, if any."""
    if panel is None:
        return list(plan.design.group_matrices)
    placeholder = OrdinaryPanelGroupMatrix(
        np.broadcast_to(np.float64(0.0), (plan.design.n, panel.width))
    )
    return [placeholder, *(plan.design.group_matrices[i] for i in panel.rest)]


def _channel_group_pairs(pairs, groups):
    """``((channel, i, j), X_i, X_j)`` for every group pair of every curvature channel."""
    for channel, (left, right) in enumerate(pairs):
        for i, gm_i in enumerate(groups[left]):
            for j, gm_j in enumerate(groups[right]):
                if left != right or i < j:
                    yield (channel, i, j), gm_i, gm_j


def pass_table_bytes(layout: StackedLayout, panels: tuple[PredictorPanel | None, ...]) -> int:
    """The bytes a chunked assembly's pass tables keep when every pair's table is admitted."""
    plans = _grouped_predictor_plans(layout)
    groups = [_accumulation_groups(plan, panel) for plan, panel in zip(plans, panels, strict=True)]
    return sum(
        table[0]
        for _key, gm_i, gm_j in _channel_group_pairs(packed_pairs(len(plans)), groups)
        if (table := _pass_table(gm_i, gm_j)) is not None
    )


class GroupedGeometryAccumulator:
    """Accumulate row-block score and curvature into coefficient space.

    Every moments call of the assembly shares one :class:`PassFactors`, so a
    support is projected once per assembly rather than once per row block and
    channel; it is released when the geometry is published. ``tables`` keeps
    the by-bin aggregates of the blocks it admits across row blocks and
    contracts each once (:meth:`release_tables`); its pairs are admitted here,
    in the serial order, so the blocks of :meth:`add_block` may run on a pool.
    """

    def __init__(
        self,
        layout: StackedLayout,
        *,
        penalty: NDArray,
        coefficients: NDArray,
        tables: PassTables | None = None,
        panels: tuple[PredictorPanel | None, ...] | None = None,
    ) -> None:
        reference_plans = _grouped_predictor_plans(layout)
        self._layout = layout
        self._coefficient_values = _validated_coefficients(coefficients, layout)
        self._penalty = validated_dense_penalty(penalty, layout.n_coefficients)
        self._pairs = packed_pairs(len(reference_plans))
        self._expected_widths = tuple(plan.width for plan in reference_plans)
        self._score = np.zeros(layout.n_coefficients, dtype=np.float64)
        self._curvature = np.zeros(
            (layout.n_coefficients, layout.n_coefficients),
            dtype=np.float64,
        )
        self._channel_chunks = np.zeros(len(self._pairs), dtype=np.intp)
        self._score_chunks = 0
        self._finished = False
        self._factors: PassFactors | None = PassFactors()
        self._tables = tables
        # A predictor with a panel (``predictor_panels``) is accumulated in its
        # packed column order, the panel's columns first, and ``finish``
        # permutes the result back: ``order[c]`` is packed column c's coefficient.
        groups, columns, order = [], [], []
        for state, plan, panel in zip(
            layout.predictors,
            reference_plans,
            (None,) * len(reference_plans) if panels is None else panels,
            strict=True,
        ):
            start = state.coefficient_slice.start
            design_groups = _accumulation_groups(plan, panel)
            if panel is None:
                cursor = start + int(plan.intercept)
                order.extend(range(start, state.coefficient_slice.stop))
            else:
                cursor = start
                order.extend(start + panel.order)
            spans = []
            for group in design_groups:
                spans.append(slice(cursor, cursor + group.shape[1]))
                cursor += group.shape[1]
            groups.append(design_groups)
            columns.append(tuple(spans))
        # Each predictor's group columns in accumulation order, for pass tables.
        self._group_columns = tuple(columns)
        self._order = (
            None if panels is None or all(p is None for p in panels) else np.asarray(order)
        )
        if tables is not None:
            for key, gm_i, gm_j in _channel_group_pairs(self._pairs, groups):
                tables.reserve(key, gm_i, gm_j)

    def _validated_plans(
        self,
        plans: tuple[PredictorExecutionPlan, ...],
    ) -> tuple[PredictorExecutionPlan, ...]:
        if self._finished:
            raise RuntimeError("geometry accumulator is already finished")
        plan_tuple = tuple(plans)
        if len(plan_tuple) != len(self._layout.predictors) or any(
            not isinstance(plan, PredictorExecutionPlan) for plan in plan_tuple
        ):
            raise ValueError("one predictor execution plan is required per layout predictor")
        row_counts = {plan.design.n for plan in plan_tuple}
        if len(row_counts) != 1 or next(iter(row_counts)) < 1:
            raise ValueError("chunk predictor plans must share a positive row count")
        if tuple(plan.width for plan in plan_tuple) != self._expected_widths:
            raise ValueError("chunk predictor plan widths must match the global layout")
        return plan_tuple

    def add_score(
        self,
        plans: tuple[PredictorExecutionPlan, ...],
        score_eta: NDArray,
    ) -> None:
        """Accumulate all predictor score channels for one row block."""
        plan_tuple = self._validated_plans(plans)
        self._place_score(
            self._score_products(plan_tuple, self._validated_score(plan_tuple, score_eta))
        )

    def _validated_score(self, plan_tuple, score_eta: NDArray) -> NDArray:
        try:
            score = np.asarray(score_eta, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("chunk score must be a finite row-by-parameter array") from exc
        expected = (plan_tuple[0].design.n, len(plan_tuple))
        if score.shape != expected or not np.all(np.isfinite(score)):
            raise ValueError(f"chunk score must be finite with shape {expected}")
        return score

    def _score_products(self, plan_tuple, score: NDArray) -> list[NDArray]:
        return [
            plan.score(score[:, state.parameter_index])
            for state, plan in zip(self._layout.predictors, plan_tuple, strict=True)
        ]

    def _place_score(self, products: list[NDArray]) -> None:
        for state, product in zip(self._layout.predictors, products, strict=True):
            self._score[state.coefficient_slice] += product
        self._score_chunks += 1

    def add_curvature_channel(
        self,
        plans: tuple[PredictorExecutionPlan, ...],
        channel_index: int,
        weights: NDArray,
    ) -> None:
        """Accumulate one canonical packed curvature channel for one row block."""
        plan_tuple = self._validated_plans(plans)
        if (
            isinstance(channel_index, bool)
            or not isinstance(channel_index, int)
            or not 0 <= channel_index < len(self._pairs)
        ):
            raise ValueError("channel_index lies outside canonical packed order")
        try:
            weight_values = np.asarray(weights, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("chunk curvature channel must be a finite row vector") from exc
        expected = (plan_tuple[0].design.n,)
        if weight_values.shape != expected or not np.all(np.isfinite(weight_values)):
            raise ValueError(f"chunk curvature channel must be finite with shape {expected}")

        if self._tables is not None:
            raise RuntimeError("an accumulator with pass tables adds whole blocks (add_block)")
        left_index, right_index = self._pairs[channel_index]
        if left_index == right_index:
            block = plan_tuple[left_index].diagonal_moment(weight_values, _factors=self._factors)
        else:
            block = plan_tuple[left_index].cross_moment(
                plan_tuple[right_index], weight_values, _factors=self._factors
            )
        self._place_channel(channel_index, block)

    def add_block(
        self,
        plans: tuple[PredictorExecutionPlan, ...],
        score_eta: NDArray,
        curvature: NDArray,
        *,
        beside: BlockTask | None = None,
    ) -> None:
        """Accumulate one row block's score and every curvature channel.

        ``curvature`` holds one row vector per canonical packed channel. The
        group-pair blocks of every channel, the block's score products and
        ``beside`` (another product of the caller's over the block's rows) run
        as one queue (``run_block_tasks``), on the pool when their work
        warrants it; each channel's block is the one
        :meth:`add_curvature_channel` adds.
        """
        plan_tuple = self._validated_plans(plans)
        score = self._validated_score(plan_tuple, score_eta)
        channels = np.asarray(curvature, dtype=np.float64)
        expected = (len(self._pairs), plan_tuple[0].design.n)
        if channels.shape != expected or not np.all(np.isfinite(channels)):
            raise ValueError(f"chunk curvature channels must be finite with shape {expected}")
        tasks: list = []
        finishers = []
        for channel_index, (left_index, right_index) in enumerate(self._pairs):
            weights = channels[channel_index]
            if left_index == right_index:
                channel_tasks, finish = plan_tuple[left_index].diagonal_tasks(
                    weights, channel=channel_index
                )
            else:
                channel_tasks, finish = plan_tuple[left_index].cross_tasks(
                    plan_tuple[right_index], weights, channel=channel_index
                )
            for task in channel_tasks:
                task.index = len(tasks)
                tasks.append(task)
            finishers.append(finish)
        n_rows = plan_tuple[0].design.n
        side = [] if beside is None else [beside]
        side.append(
            BlockTask(
                index=0,
                cost=float(n_rows * sum(self._expected_widths)),
                nbytes=8 * n_rows,
                run=lambda _cache, _profile: self._score_products(plan_tuple, score),
                place=self._place_score,
            )
        )
        for task in side:
            task.index = len(tasks)
            tasks.append(task)
        add_table = None if self._tables is None else self._tables.add
        run_block_tasks(
            tasks, _BlockWeightCache(None, self._factors, add_table), None, dispatch_bound=True
        )
        for channel_index, finish in enumerate(finishers):
            self._place_channel(channel_index, finish())

    def _place_channel(self, channel_index: int, block: NDArray) -> None:
        left_index, right_index = self._pairs[channel_index]
        left_state = self._layout.predictors[left_index]
        right_state = self._layout.predictors[right_index]
        self._curvature[left_state.coefficient_slice, right_state.coefficient_slice] += block
        if left_index != right_index:
            self._curvature[right_state.coefficient_slice, left_state.coefficient_slice] += block.T
        self._channel_chunks[channel_index] += 1

    def release_tables(self) -> None:
        """Contract the pass tables into the curvature; later blocks contract their own."""
        tables, self._tables = self._tables, None
        if tables is None:
            return
        for (channel_index, left, right), block in tables.contracted(self._factors):
            left_index, right_index = self._pairs[channel_index]
            rows = self._group_columns[left_index][left]
            columns = self._group_columns[right_index][right]
            self._curvature[rows, columns] += block
            self._curvature[columns, rows] += block.T

    def finish(self) -> DenseJointGeometry:
        """Validate complete channel coverage and publish immutable geometry."""
        if self._finished:
            raise RuntimeError("geometry accumulator is already finished")
        if self._score_chunks < 1 or np.any(self._channel_chunks != self._score_chunks):
            raise RuntimeError("every score chunk must contribute every curvature channel")
        self.release_tables()
        self._finished = True
        self._factors = None
        if self._order is not None:
            score = np.empty_like(self._score)
            score[self._order] = self._score
            curvature = np.empty_like(self._curvature)
            curvature[np.ix_(self._order, self._order)] = self._curvature
            self._score, self._curvature = score, curvature
        if not np.array_equal(self._curvature, self._curvature.T):
            raise RuntimeError("grouped data curvature lost exact symmetry")

        score_penalized = self._score - self._penalty @ self._coefficient_values
        penalized_curvature = self._curvature + self._penalty
        return DenseJointGeometry(
            score_data=self._score,
            score_penalized=score_penalized,
            data_curvature=self._curvature,
            penalty=self._penalty,
            penalized_curvature=penalized_curvature,
        )


def assemble_grouped_geometry(
    layout: StackedLayout,
    score_eta: NDArray,
    curvature_packed: NDArray,
    *,
    penalty: NDArray,
    coefficients: NDArray,
) -> DenseJointGeometry:
    """Assemble joint geometry through symmetric and rectangular group plans."""
    plans = _grouped_predictor_plans(layout)
    n_observations = plans[0].design.n
    k_parameters = len(plans)
    score, curvature = _validated_channels(
        score_eta,
        curvature_packed,
        n_observations=n_observations,
        k_parameters=k_parameters,
    )
    accumulator = GroupedGeometryAccumulator(
        layout,
        penalty=penalty,
        coefficients=coefficients,
    )
    accumulator.add_score(plans, score)
    for channel_index in range(curvature.shape[1]):
        accumulator.add_curvature_channel(
            plans,
            channel_index,
            curvature[:, channel_index],
        )
    return accumulator.finish()
