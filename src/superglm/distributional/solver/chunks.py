"""Bounded-row evaluation for distributional predictors and likelihoods."""

from __future__ import annotations

import itertools
import math
import operator
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._block_queue import BlockTask
from superglm._group_matrix._group_matrix_algebra import PassTables
from superglm._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm._group_matrix._group_matrix_range import group_range_matvec, group_row_range
from superglm._parallel import parallel_config, pooled_for, pooled_row_blocks
from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    LikelihoodPlanValidatingFamily,
)
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.solver._formation import FormationAccumulator
from superglm.distributional.solver._joint_tables import (
    MORSEL_ROWS,
    STACK_BLOCK_ROWS,
    JointTablePlan,
    joint_table_plan,
    stack_in_range,
)
from superglm.distributional.solver._likelihood_cache import _BUILTIN_FAMILIES
from superglm.distributional.solver._ordinary_packing import predictor_panels
from superglm.distributional.solver.assembly import (
    DenseJointGeometry,
    GroupedGeometryAccumulator,
    _grouped_predictor_plans,
    pass_table_bytes,
    with_formation,
)
from superglm.distributional.solver.derivatives import (
    nearest_semidefinite_curvature,
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.group_matrix import DesignMatrix

ChunkSize = int | Literal["auto"]
# ``nearest_semidefinite`` is the terminal policy's last fallback: observed
# curvature with each row's block projected onto the semidefinite cone.
CurvatureSource = Literal["observed", "fisher", "nearest_semidefinite"]

# Counts eta, theta, natural/transformed derivatives, masks, family temporaries
# and the rows the stream's predictor products hold (:func:`chunk_row_floats`).
# It is a deterministic bound selector, not a claim about exact allocator RSS.
AUTO_CHUNK_SELECTOR = "distributional-auto-v3"
AUTO_CHUNK_MEMORY_BYTES = 8 * 1024 * 1024
AUTO_CONTRACTION_MEMORY_BYTES = 64 * 1024 * 1024
# The most rows the likelihood rule gives an automatic chunk.
AUTO_CHUNK_MAX_ROWS = 2**15
# The longest automatic chunk of a layout that streams in ranges
# (``layout_chunk_size``).
AUTO_STREAM_MAX_ROWS = 2**16
# A row chunk shorter than this stays on the calling thread: at about a microsecond
# a row in array kernels, waking a worker costs a measurable share of its work.
_MIN_POOLED_CHUNK_ROWS = 2**12
# A chunk's transient arrays while its family kernels run, per row.
_CHUNK_SCRATCH_ROW_BYTES = 1024
# Array kernels hold the GIL between NumPy calls, so past a handful of workers they
# wait on one another and run slower.
_ARRAY_CHUNK_WORKERS = 8

# Retained snapshots plus support vectors per pass; refresh uses bounded scratch.
# No observation-sized predictions are retained here.
_SUPPORT_PREDICTION_BYTES = 1024 * 1024


class _SupportPredictions:
    """Reuse small support products while checking mutable algebra each chunk."""

    def __init__(self):
        self.entries = {}
        self.nbytes = 0
        self._lock = threading.Lock()

    def __call__(self, group, beta):
        key = id(group)
        source = (group.B_unique, group.R_inv, beta)

        def held():
            entry = self.entries.get(key)
            if entry is not None and all(
                np.array_equal(a, b) for a, b in zip(source, entry[0], strict=True)
            ):
                return entry[1]
            return None

        values = held()
        if values is not None:
            return values
        # Row chunks may run on the pool: a miss is formed once, and a hit reads an
        # entry that is never written after it is published.
        with self._lock:
            values = held()
            if values is not None:
                return values
            entry = self.entries.pop(key, None)
            if entry is not None:
                self.nbytes -= sum(a.nbytes for a in entry[0]) + entry[1].nbytes
                del entry
            values = group.B_unique @ (group.R_inv @ beta)
            size = sum(a.nbytes for a in source) + values.nbytes
            if size <= _SUPPORT_PREDICTION_BYTES - self.nbytes:
                self.entries[key] = (tuple(a.copy() for a in source), values)
                self.nbytes += size
            return values


def _support_range(group, rows, beta, support_predictions):
    if support_predictions is None or type(group) not in (
        DiscretizedSSPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
    ):
        return None
    return group_range_matvec(group, rows.start, rows.stop, beta, support_predictions)


def _immutable_response(response: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.ascontiguousarray(response, dtype=np.float64)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _validate_bound_likelihood(
    family: DistributionalFamily,
    plan: FamilyLikelihoodPlan,
    response: NDArray[np.float64],
) -> tuple[FamilyLikelihoodPlan, NDArray[np.float64]]:
    """Validate one family likelihood at fixed-fit preparation."""

    if not isinstance(plan, FamilyLikelihoodPlan):
        raise UnsupportedLikelihoodContractError(
            "family.bind_likelihood() must return a FamilyLikelihoodPlan"
        )
    if not isinstance(plan.weights, ResolvedLikelihoodWeights):
        raise UnsupportedLikelihoodContractError(
            "family likelihood plans must own resolved likelihood weights"
        )
    if not isinstance(plan.plan_identifier, str) or not plan.plan_identifier.strip():
        raise UnsupportedLikelihoodContractError(
            "family likelihood plans require a non-empty identifier"
        )
    n_rows = len(plan.weights.values)
    if plan.weights.digest != plan.weights.root_digest or not np.array_equal(
        plan.weights.root_take_map,
        np.arange(n_rows, dtype=np.intp),
    ):
        raise UnsupportedLikelihoodContractError(
            "a fixed fit requires the complete prepared likelihood, not a row subset"
        )
    try:
        supplied = np.asarray(response, dtype=np.float64)
        if supplied.shape != (n_rows,) or not np.all(np.isfinite(supplied)):
            raise UnsupportedLikelihoodContractError(
                "the fitted response and family likelihood must contain the same finite rows"
            )
        canonical = (
            family.validate_likelihood_plan(supplied, plan)
            if isinstance(family, LikelihoodPlanValidatingFamily)
            else _immutable_response(supplied)
        )
    except UnsupportedLikelihoodContractError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedLikelihoodContractError(
            "family likelihood validation returned an invalid response"
        ) from exc
    if (
        not isinstance(canonical, np.ndarray)
        or canonical.shape != (n_rows,)
        or canonical.dtype != np.float64
        or canonical.flags.writeable
        or not np.all(np.isfinite(canonical))
    ):
        raise UnsupportedLikelihoodContractError(
            "family likelihood validation returned an invalid response"
        )
    return plan, canonical


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def resolve_chunk_size(
    n_observations: int,
    k_parameters: int,
    chunk_size: ChunkSize,
    *,
    row_floats: int = 0,
) -> int:
    """Resolve an explicit or memory-budgeted row bound.

    ``row_floats`` counts the floats a chunk's designs own per row, beyond the
    likelihood's own channels. An automatic bound holds at most
    ``AUTO_CHUNK_MAX_ROWS`` rows.
    """
    rows = _positive_integer(n_observations, name="n_observations")
    parameters = _positive_integer(k_parameters, name="k_parameters")
    if isinstance(row_floats, bool):
        raise TypeError("row_floats must be a non-negative integer")
    try:
        width = operator.index(row_floats)
    except TypeError as exc:
        raise TypeError("row_floats must be a non-negative integer") from exc
    if width < 0:
        raise ValueError("row_floats must be a non-negative integer")
    if chunk_size == "auto":
        budget_rows = AUTO_CHUNK_MEMORY_BYTES // (8 * _likelihood_row_floats(parameters, width))
        return min(rows, AUTO_CHUNK_MAX_ROWS, max(1, int(budget_rows)))
    if isinstance(chunk_size, str):
        raise ValueError("chunk_size string must be 'auto'")
    return min(rows, _positive_integer(chunk_size, name="chunk_size"))


def _likelihood_row_floats(k_parameters: int, row_floats: int) -> int:
    """Floats an automatic chunk holds per row: ``row_floats`` and the likelihood's channels."""
    n_channels = k_parameters * (k_parameters + 1) // 2
    return row_floats + 6 * k_parameters + 4 * n_channels + 4


def layout_chunk_size(layout: StackedLayout, chunk_size: ChunkSize) -> int:
    """:func:`resolve_chunk_size` for a layout's rows, predictors and chunk designs.

    A block contracts whole chunks, so an automatic chunk holds no more rows
    than the contraction allowance (``contraction_row_bytes``). A layout that
    streams in ranges contracts each chunk as it arrives, so its chunk may
    instead take the allowance its pass tables leave (``pass_table_bytes``),
    up to ``AUTO_STREAM_MAX_ROWS`` rows: fewer chunks spread each pass's fixed
    work over more rows. An allowance that holds every row is one block, which
    keeps no tables. Its likelihood arrays then hold at most that many rows of
    the floats :func:`_likelihood_row_floats` counts.
    """
    n = layout.predictors[0].design.n
    k = len(layout.predictors)
    if chunk_size != "auto":
        return resolve_chunk_size(n, k, chunk_size)
    floats = chunk_row_floats(layout)
    rows = resolve_chunk_size(n, k, "auto", row_floats=floats)
    panels = predictor_panels(layout)
    row_bytes = contraction_row_bytes(layout, panels)
    if row_bytes is None:
        return rows
    fits = max(1, AUTO_CONTRACTION_MEMORY_BYTES // row_bytes)
    if not _streams_in_ranges(layout):
        return min(rows, fits)
    if n <= AUTO_STREAM_MAX_ROWS and n * row_bytes <= AUTO_CONTRACTION_MEMORY_BYTES:
        return n
    stream = (AUTO_CONTRACTION_MEMORY_BYTES - pass_table_bytes(layout, panels)) // row_bytes
    return min(fits, max(rows, min(n, AUTO_STREAM_MAX_ROWS, stream)))


def _resolve_fitting_chunk_size(family, layout, likelihood_plan, chunk_size):
    """One row bound for every phase of a fit, or None for unchunked evaluation."""
    if chunk_size is None:
        return None
    return layout_chunk_size(layout, chunk_size)


@dataclass(frozen=True)
class RowChunk:
    """One contiguous, non-empty row range and its subset indices."""

    start: int
    stop: int
    indices: NDArray[np.intp]


def iter_row_chunks(
    n_observations: int,
    chunk_size: ChunkSize,
    *,
    k_parameters: int = 1,
    row_floats: int = 0,
):
    """Yield contiguous row chunks that cover every observation exactly once."""
    rows = _positive_integer(n_observations, name="n_observations")
    size = resolve_chunk_size(rows, k_parameters, chunk_size, row_floats=row_floats)
    for start in range(0, rows, size):
        stop = min(rows, start + size)
        indices = np.arange(start, stop, dtype=np.intp)
        indices.setflags(write=False)
        yield RowChunk(start=start, stop=stop, indices=indices)


@dataclass(frozen=True)
class LikelihoodChunk:
    """Current-row predictor state and transformed derivative channels."""

    rows: RowChunk
    plans: tuple[PredictorExecutionPlan, ...]
    eta: NDArray[np.float64]
    theta: NDArray[np.float64]
    optimizing_log_likelihood: NDArray[np.float64]
    parameter_independent_carrier: NDArray[np.float64]
    score_eta: NDArray[np.float64]
    curvature_packed: NDArray[np.float64]


@dataclass(frozen=True)
class ChunkedLikelihoodSums:
    """Scalar optimizing and fixed-carrier sums from one chunked pass."""

    optimizing_log_likelihood: float
    parameter_independent_carrier: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.optimizing_log_likelihood) or not np.isfinite(
            self.parameter_independent_carrier
        ):
            raise ValueError("chunked likelihood sums must be finite")

    @property
    def log_likelihood(self) -> float:
        return float(self.optimizing_log_likelihood + self.parameter_independent_carrier)


class _TrialDerivativeError(ValueError):
    """A numerical derivative failure permits a fresh value-only trial screen."""

    def __init__(self, reason, *, rows=None, plans=None):
        super().__init__(reason)
        self.rows = rows
        self.plans = plans


def _validated_coefficients(
    layout: StackedLayout,
    coefficients: NDArray,
) -> NDArray[np.float64]:
    try:
        values = np.asarray(coefficients, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("coefficients must be a finite global-layout vector") from exc
    if values.shape != (layout.n_coefficients,) or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be a finite global-layout vector")
    return values


def _predictor_values(
    layout: StackedLayout,
    coefficients: NDArray[np.float64],
    rows: RowChunk,
    *,
    include_offsets: bool,
    support_predictions=None,
) -> NDArray[np.float64]:
    """Evaluate a chunk without preparing coefficient-space geometry.

    Ordinary groups consume their bounded row range directly. Refused groups
    retain the generic subset path, consumed immediately. Value-only line
    searches and convergence checks need neither a chunk DesignMatrix nor
    retained PredictorExecutionPlans. Sum the ordered slope contributions
    before adding the intercept and offset, as geometry and prediction do.
    """
    eta = np.empty((len(rows.indices), len(layout.predictors)), dtype=np.float64)
    for state in layout.predictors:
        intercept = state.intercept_index is not None
        local = coefficients[state.coefficient_slice]
        values = np.zeros(len(rows.indices), dtype=np.float64)
        column = int(intercept)
        for group in state.design.group_matrices:
            width = group.shape[1]
            group_coefficients = local[column : column + width]
            contribution = _support_range(
                group,
                rows,
                group_coefficients,
                support_predictions if type(state.design) is DesignMatrix else None,
            )
            if contribution is None:
                contribution = group_range_matvec(group, rows.start, rows.stop, group_coefficients)
            if contribution is None:
                contribution = group.row_subset(rows.indices).matvec(group_coefficients)
            values += contribution
            column += width
        if intercept:
            if state.design.p:
                values += local[0]
            else:
                values[:] = local[0]
        if include_offsets:
            values += state.offset[rows.start : rows.stop]
        eta[:, state.parameter_index] = values
    if not np.all(np.isfinite(eta)):
        raise ValueError("chunk predictor evaluation produced non-finite values")
    return eta


def _row_design(state, rows: RowChunk, *, copy: bool = True):
    """One predictor's design restricted to a contiguous row range."""
    if type(state.design) is DesignMatrix:
        groups = []
        for group in state.design.group_matrices:
            child = group_row_range(group, rows.start, rows.stop, copy=copy)
            groups.append(group.row_subset(rows.indices) if child is None else child)
        return DesignMatrix(groups, len(rows.indices), state.design.p)
    return state.design.row_subset(rows.indices)


def _predictor_chunk(
    layout: StackedLayout,
    coefficients: NDArray[np.float64],
    rows: RowChunk,
    *,
    include_offsets: bool,
    support_predictions=None,
) -> tuple[NDArray[np.float64], tuple[PredictorExecutionPlan, ...]]:
    k_parameters = len(layout.predictors)
    eta = np.empty((len(rows.indices), k_parameters), dtype=np.float64)
    plans: list[PredictorExecutionPlan] = []
    for state in layout.predictors:
        design = _row_design(state, rows)
        intercept = state.intercept_index is not None
        plan = PredictorExecutionPlan(design, intercept)
        local = coefficients[state.coefficient_slice]
        slope_start = int(intercept)
        values = np.full(
            len(rows.indices),
            local[0] if intercept else 0.0,
            dtype=np.float64,
        )
        if design.p:
            if (
                support_predictions is not None
                and type(state.design) is DesignMatrix
                and not (
                    design._tabmat_vector_candidate and design._tabmat_holder.split is not None
                )
                and any(
                    type(group)
                    in (DiscretizedSSPGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)
                    for group in state.design.group_matrices
                )
            ):
                slopes = np.zeros(len(rows.indices), dtype=np.float64)
                column = slope_start
                for group, child in zip(
                    state.design.group_matrices, design.group_matrices, strict=True
                ):
                    width = group.shape[1]
                    beta = local[column : column + width]
                    contribution = _support_range(group, rows, beta, support_predictions)
                    slopes += child.matvec(beta) if contribution is None else contribution
                    column += width
                values += slopes
            else:
                values += design.matvec(local[slope_start:])
        if include_offsets:
            values += state.offset[rows.indices]
        eta[:, state.parameter_index] = values
        plans.append(plan)
    if not np.all(np.isfinite(eta)):
        raise ValueError("chunk predictor evaluation produced non-finite values")
    return eta, tuple(plans)


def _theta_chunk(
    layout: StackedLayout,
    eta: NDArray[np.float64],
    *,
    _recover_derivative_failure: bool = False,
) -> NDArray[np.float64]:
    theta = np.empty_like(eta)
    for state in layout.predictors:
        values = np.asarray(state.link.inverse(eta[:, state.parameter_index]), dtype=np.float64)
        if values.shape != (len(eta),):
            raise ValueError(f"inverse link for predictor {state.name!r} produced an invalid chunk")
        if not np.all(np.isfinite(values)):
            error = _TrialDerivativeError if _recover_derivative_failure else ValueError
            raise error(f"inverse link for predictor {state.name!r} produced an invalid chunk")
        theta[:, state.parameter_index] = values
    return theta


def _validate_chunk_inputs(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    coefficients: NDArray,
) -> tuple[NDArray[np.float64], FamilyLikelihoodPlan, NDArray[np.float64]]:
    if not isinstance(family, DistributionalFamily):
        raise TypeError("family must implement DistributionalFamily")
    if not isinstance(layout, StackedLayout) or not layout.predictors:
        raise TypeError("layout must be a non-empty StackedLayout")
    n_observations = layout.predictors[0].design.n
    try:
        response = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("response must be a finite row vector") from exc
    if response.shape != (n_observations,) or not np.all(np.isfinite(response)):
        raise UnsupportedLikelihoodContractError(
            "response and likelihood plan must match the layout rows"
        )
    if not isinstance(likelihood_plan, FamilyLikelihoodPlan) or not isinstance(
        likelihood_plan.weights, ResolvedLikelihoodWeights
    ):
        raise UnsupportedLikelihoodContractError(
            "family likelihood slicing requires a prepared likelihood plan"
        )
    if len(likelihood_plan.weights.values) != n_observations:
        raise UnsupportedLikelihoodContractError(
            "response and likelihood plan must match the layout rows"
        )
    return response, likelihood_plan, _validated_coefficients(layout, coefficients)


def _take_likelihood_rows(plan, rows, likelihood_cache, lock):
    if likelihood_cache is None:
        return plan.take(rows.indices)
    with lock:
        return likelihood_cache.take(plan, rows.indices, start=rows.start, stop=rows.stop)


def _evaluated_chunks(family, evaluate, n_observations, chunk_rows, *, held_floats):
    """``[evaluate(rows), ...]`` over the row chunks, yielded in row order.

    Each row's likelihood is its own term, so a package family whose kernels are
    array code runs its chunks on the pool. The chunks depend only on the row count
    and are reduced in row order, so the bits are the same at every worker count (Li
    and Wood 2020, sec. 4). A family whose kernels are compiled row loops pools their
    rows instead; there is one level of parallelism either way. A chunk too short to
    repay waking a worker stays on the calling thread, and so does a family from
    outside the package, whose ``evaluate_natural`` never promised to be thread-safe.
    ``held_floats`` is what a result holds per row: results wait for the consumer a
    window at a time, within the contraction allowance.
    """
    chunks = iter_row_chunks(n_observations, chunk_rows)
    if (
        type(family) not in _BUILTIN_FAMILIES
        or getattr(family, "pools_rows", False)
        or chunk_rows < _MIN_POOLED_CHUNK_ROWS
    ):
        yield from map(evaluate, chunks)
        return
    window = max(1, AUTO_CONTRACTION_MEMORY_BYTES // (8 * held_floats * chunk_rows))

    def evaluate_at_one_level(rows):
        with parallel_config(n_jobs=1):
            return evaluate(rows)

    for group in itertools.batched(chunks, window):
        results = pooled_for(
            evaluate_at_one_level,
            list(group),
            _CHUNK_SCRATCH_ROW_BYTES * chunk_rows,
            max_workers=_ARRAY_CHUNK_WORKERS,
        )
        for index, result in enumerate(results):
            results[index] = None
            yield result


def iter_likelihood_chunks(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    coefficients: NDArray,
    *,
    chunk_size: ChunkSize,
    curvature_source: CurvatureSource,
    _range_geometry: bool = False,
    _recover_derivative_failure: bool = False,
    likelihood_cache=None,
):
    """Yield bounded predictor derivatives and curvature for each row chunk.

    ``_range_geometry`` evaluates each chunk from the layout's own designs in
    row ranges, for an assembly that reads the rows itself. The default
    stream supplies owned child plans.
    """
    if curvature_source == "fisher" and not isinstance(family, ExpectedInformationFamily):
        raise ValueError("Fisher chunking requires expected_information_natural")
    response, plan, coefficient_values = _validate_chunk_inputs(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
    )
    links = tuple(state.link for state in layout.predictors)
    support_predictions = _SupportPredictions()
    range_plans = None
    if _range_geometry and all(
        type(state.design) is DesignMatrix
        and not (
            state.design._tabmat_vector_candidate and state.design._tabmat_holder.split is not None
        )
        for state in layout.predictors
    ):
        range_plans = tuple(
            PredictorExecutionPlan(state.design, state.intercept_index is not None)
            for state in layout.predictors
        )
    take_lock = threading.Lock()

    def evaluate(rows: RowChunk) -> LikelihoodChunk:
        if range_plans is None:
            eta, plans = _predictor_chunk(
                layout,
                coefficient_values,
                rows,
                include_offsets=True,
                support_predictions=support_predictions,
            )
        else:
            eta = _predictor_values(
                layout,
                coefficient_values,
                rows,
                include_offsets=True,
                support_predictions=support_predictions,
            )
            plans = range_plans
        try:
            theta = _theta_chunk(
                layout, eta, _recover_derivative_failure=_recover_derivative_failure
            )
        except (_TrialDerivativeError, FloatingPointError, OverflowError) as exc:
            if _recover_derivative_failure:
                raise _TrialDerivativeError(str(exc), rows=rows, plans=plans) from exc
            raise
        child_plan = _take_likelihood_rows(plan, rows, likelihood_cache, take_lock)
        if len(child_plan.weights.values) != len(rows.indices):
            raise UnsupportedLikelihoodContractError(
                "family likelihood slicing returned the wrong number of rows"
            )
        try:
            with (
                np.errstate(over="raise", invalid="raise", divide="raise")
                if _recover_derivative_failure
                else nullcontext()
            ):
                natural = family.evaluate_natural(
                    response[rows.indices],
                    theta,
                    child_plan,
                    derivative_order=2,
                )
        except (
            _NumericalEvaluationError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as exc:
            if _recover_derivative_failure:
                raise _TrialDerivativeError(str(exc), rows=rows, plans=plans) from exc
            raise
        except ValueError as exc:
            # Only proven parameter-domain failure is numerical here. Unknown
            # ValueErrors, including malformed source/derivative shapes, stay hard.
            if _recover_derivative_failure and any(
                not np.all(spec.support.contains(theta[:, index]))
                for index, spec in enumerate(family.parameters)
            ):
                raise _TrialDerivativeError(str(exc), rows=rows, plans=plans) from exc
            raise
        if natural.derivative_order != 2:
            raise UnsupportedLikelihoodContractError(
                "family must return exact derivative order 2 for chunk geometry"
            )
        if natural.valid is not None and not np.all(natural.valid):
            if _recover_derivative_failure:
                raise _TrialDerivativeError(
                    "chunk contains an invalid likelihood state", rows=rows, plans=plans
                )
            raise ValueError("chunk contains an invalid likelihood state")
        try:
            with (
                np.errstate(over="raise", invalid="raise", divide="raise")
                if _recover_derivative_failure
                else nullcontext()
            ):
                transformed = transform_natural_derivatives(natural, eta, links)
                if curvature_source == "observed":
                    curvature = transformed.curvature_packed
                elif curvature_source == "nearest_semidefinite":
                    curvature = nearest_semidefinite_curvature(transformed.curvature_packed)
                else:
                    assert isinstance(family, ExpectedInformationFamily)
                    information = family.expected_information_natural(theta, child_plan)
                    curvature = transform_natural_information(information, eta, links)
        except (
            _NumericalEvaluationError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as exc:
            if _recover_derivative_failure:
                raise _TrialDerivativeError(str(exc), rows=rows, plans=plans) from exc
            raise
        return LikelihoodChunk(
            rows=rows,
            plans=plans,
            eta=eta,
            theta=theta,
            optimizing_log_likelihood=transformed.optimizing_log_likelihood,
            parameter_independent_carrier=transformed.parameter_independent_carrier,
            score_eta=transformed.score_eta,
            curvature_packed=curvature,
        )

    k_parameters = len(layout.predictors)
    # eta, theta, score, packed curvature and the two likelihood rows, and the
    # predictor designs a chunk copied for itself.
    held_floats = 3 * k_parameters + k_parameters * (k_parameters + 1) // 2 + 2
    if range_plans is None:
        held_floats += _chunk_design_floats(layout)
    yield from _evaluated_chunks(
        family,
        evaluate,
        len(response),
        layout_chunk_size(layout, chunk_size),
        held_floats=held_floats,
    )


def evaluate_chunked_log_likelihood(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    coefficients: NDArray,
    *,
    chunk_size: ChunkSize,
    likelihood_cache=None,
    pass_factors=None,
) -> ChunkedLikelihoodSums:
    """Evaluate only the scalar weighted likelihood in bounded row chunks."""
    response, plan, coefficient_values = _validate_chunk_inputs(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
    )
    if pass_factors is not None:
        joint = _joint_plan(layout, pass_factors)
        compiled = _compiled_rows(family, layout, y, likelihood_plan, joint, pass_factors)
        if compiled is not None:
            tables, constants = joint.predictor_tables(coefficient_values)
            row_block = min(_COMPILED_ROW_BLOCK, layout_chunk_size(layout, chunk_size))
            total = _compiled_likelihood(
                compiled, joint, tables, constants, len(response), row_block
            )
            if total is not None:
                return ChunkedLikelihoodSums(
                    optimizing_log_likelihood=total, parameter_independent_carrier=0.0
                )
    support_predictions = _SupportPredictions()
    take_lock = threading.Lock()

    def evaluate(rows: RowChunk) -> tuple[float, float]:
        eta = _predictor_values(
            layout,
            coefficient_values,
            rows,
            include_offsets=True,
            support_predictions=support_predictions,
        )
        theta = _theta_chunk(layout, eta)
        child_plan = _take_likelihood_rows(plan, rows, likelihood_cache, take_lock)
        if len(child_plan.weights.values) != len(rows.indices):
            raise UnsupportedLikelihoodContractError(
                "family likelihood slicing returned the wrong number of rows"
            )
        natural = family.evaluate_natural(
            response[rows.indices],
            theta,
            child_plan,
            derivative_order=0,
        )
        if natural.derivative_order != 0:
            raise UnsupportedLikelihoodContractError(
                "family must return exact derivative order 0 for chunk values"
            )
        if natural.valid is not None and not np.all(natural.valid):
            raise ValueError("chunk contains an invalid likelihood state")
        return (
            float(np.sum(natural.optimizing_log_likelihood, dtype=np.float64)),
            float(np.sum(natural.parameter_independent_carrier, dtype=np.float64)),
        )

    optimizing_total = 0.0
    carrier_total = 0.0
    for optimizing, carrier in _evaluated_chunks(
        family,
        evaluate,
        len(response),
        layout_chunk_size(layout, chunk_size),
        held_floats=1,
    ):
        optimizing_total += optimizing
        carrier_total += carrier
    return ChunkedLikelihoodSums(
        optimizing_log_likelihood=optimizing_total,
        parameter_independent_carrier=carrier_total,
    )


def _evaluate_chunked_geometry(
    family,
    layout,
    y,
    likelihood_plan,
    coefficients,
    *,
    penalty,
    chunk_size,
    curvature_source,
    likelihood_cache=None,
    pass_factors=None,
) -> tuple[DenseJointGeometry, ChunkedLikelihoodSums]:
    """Return owned geometry and scalar likelihood sums from the same stream."""
    sums = [0.0, 0.0]
    geometry = assemble_chunked_geometry(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
        penalty=penalty,
        chunk_size=chunk_size,
        curvature_source=curvature_source,
        likelihood_cache=likelihood_cache,
        pass_factors=pass_factors,
        _likelihood_sums=sums,
    )
    if not all(np.isfinite(value) for value in sums):
        raise _TrialDerivativeError("chunked likelihood sums must be finite")
    return geometry, ChunkedLikelihoodSums(*sums)


# Owned row arrays a contraction block's design adds per row, by group type.
# Exact supports are row views; categories rebuild their codes; tensors and
# SCOP take the generic subset.
_BLOCK_ROW_ARRAYS = {
    DiscretizedSSPGroupMatrix: 0,
    SupportCompressedSSPGroupMatrix: 0,
    CategoricalGroupMatrix: 1,
    RandomEffectGroupMatrix: 1,
    DiscretizedTensorGroupMatrix: 3,
    DiscretizedSCOPGroupMatrix: 1,
}
# A spline-by-category term's level groups own their rows and level bins; the
# levels' rows are disjoint, so the term owns two arrays per block row.
_SPLINE_CATEGORY_GROUPS = (
    DiscretizedSplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
)
_SPLINE_CATEGORY_ROW_ARRAYS = 2
# Transient per-row gathers while one group pair is contracted, and the
# block's row indices.
_BLOCK_GATHER_FLOATS = 5
# Group types whose block rows' ownership is counted. Others (factor smooths,
# sparse groups, subclasses) may own rows in raw storage, so their layouts
# contract one likelihood chunk at a time.
_EXACT_SPLINES = (SparseSSPGroupMatrix, SplineCategoricalGroupMatrix)
_COUNTED_GROUPS = frozenset(
    (DenseGroupMatrix, *_EXACT_SPLINES, *_BLOCK_ROW_ARRAYS, *_SPLINE_CATEGORY_GROUPS)
)


def _max_row_values(matrix) -> int:
    """The most stored values one row of a CSR matrix holds."""
    return int(np.max(np.diff(matrix.indptr), initial=0))


def _group_row_floats(group, *, chunk: bool = False) -> int:
    """Owned floats per block row of one counted group outside spline-by-category.

    ``chunk`` counts a likelihood chunk's copy instead: no gram forms dense
    rows, and its rows may be the basis's fullest.
    """
    if type(group) not in _EXACT_SPLINES:
        return _BLOCK_ROW_ARRAYS.get(type(group), group.shape[1])
    # An exact spline's block slice owns a CSR copy of the block rows: values,
    # column indices and row pointers. A saturated basis's gram also holds one
    # dense copy of its rows (8 * width) when the stored values are not a dense
    # view, as one with a knot row's zeros dropped is not; it weights them in
    # sub-blocks, so no second copy.
    basis = group.B
    rows, width = basis.shape
    entry = basis.data.itemsize + basis.indices.itemsize
    dense = 0 if chunk else 8 * width
    stored = _max_row_values(basis) * rows if chunk else basis.nnz
    per_row = entry * stored + (basis.indptr.itemsize + dense) * rows
    if type(group) is SplineCategoricalGroupMatrix:
        # A level group also owns its rows' CSR, a float copy of its values, its
        # row indices and, for a saturated basis, a cached dense copy of its rows,
        # beside which its gram forms the weighted copy the 8 * width above pays
        # for; charging these on every row bounds each level.
        per_row += (entry + 8) * stored + (basis.indptr.itemsize + 8 + dense) * rows
    return -(-per_row // (8 * rows))


def _spline_category_row_floats(design) -> int:
    """Owned floats per block row of a design's spline-by-category terms.

    A term's level groups own disjoint rows, so one term owns two arrays per
    row; terms on the same categorical share its name, and a row belongs to
    one level group of each, so the terms on a categorical number at most the
    groups that share any one of its levels.
    """
    levels: dict[tuple, int] = {}
    for group in design.group_matrices:
        if type(group) in _SPLINE_CATEGORY_GROUPS:
            feature = group.spline_cat_feature
            key = (id(group), None) if feature is None else (feature, group.spline_cat_level)
            levels[key] = levels.get(key, 0) + 1
    terms: dict = {}
    for (feature, _level), count in levels.items():
        terms[feature] = max(terms.get(feature, 0), count)
    return _SPLINE_CATEGORY_ROW_ARRAYS * sum(terms.values())


def contraction_row_bytes(layout: StackedLayout, panels=None) -> int | None:
    """Bytes one row adds to a contraction block, or None for an uncounted design.

    The buffer holds ``K + K(K+1)/2`` channels. Each group adds its owned row
    arrays, spline-by-category terms as :func:`_spline_category_row_floats`
    counts them and exact splines their CSR slice; a dense group may form a
    weighted copy of its rows, so it counts its full width. A predictor's
    panel (``panels``) counts its rows and one weighted copy in place of the
    groups it packs.
    """
    k_parameters = len(layout.predictors)
    floats = k_parameters + k_parameters * (k_parameters + 1) // 2 + _BLOCK_GATHER_FLOATS
    for position, state in enumerate(layout.predictors):
        if type(state.design) is not DesignMatrix or any(
            type(group) not in _COUNTED_GROUPS for group in state.design.group_matrices
        ):
            return None
        panel = None if panels is None else panels[position]
        floats += _spline_category_row_floats(state.design)
        floats += sum(
            _group_row_floats(group)
            for index, group in enumerate(state.design.group_matrices)
            if type(group) not in _SPLINE_CATEGORY_GROUPS and (panel is None or index in panel.rest)
        )
        if panel is not None:
            floats += 2 * panel.width
    return np.dtype(np.float64).itemsize * floats


def _exact_spline_category_chunk_floats(design) -> int:
    """Owned floats per chunk row of a design's exact spline-by-category level groups.

    Every level group copies the chunk's basis rows. A row's own level group
    also copies its level rows, their values and row indices, and a row lies
    in one level of each term, the terms on one categorical sharing its name.
    A row holds at most its basis's most stored values.
    """
    slices = 0.0
    own: dict = {}
    levels: dict[tuple, int] = {}
    for group in design.group_matrices:
        if type(group) is not SplineCategoricalGroupMatrix:
            continue
        basis = group.B
        entry = basis.data.itemsize + basis.indices.itemsize
        slices += entry * _max_row_values(basis) + basis.indptr.itemsize
        feature = group.spline_cat_feature
        key = (id(group), None) if feature is None else (feature, group.spline_cat_level)
        levels[key] = levels.get(key, 0) + 1
        level = (entry + 8) * _max_row_values(group.B_level) + basis.indptr.itemsize + 8
        own[key[0]] = max(own.get(key[0], 0.0), level)
    terms: dict = {}
    for (feature, _level), count in levels.items():
        terms[feature] = max(terms.get(feature, 0), count)
    return math.ceil((slices + sum(count * own[f] for f, count in terms.items())) / 8)


def _streams_in_ranges(layout: StackedLayout) -> bool:
    """Whether the likelihood stream evaluates every predictor in row ranges, building no designs."""
    return all(
        type(state.design) is DesignMatrix
        and not (
            state.design._tabmat_vector_candidate and state.design._tabmat_holder.split is not None
        )
        for state in layout.predictors
    )


# Groups whose range product (``group_range_matvec``) holds one row vector.
_RANGE_PRODUCTS = (
    DenseGroupMatrix,
    CategoricalGroupMatrix,
    RandomEffectGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)


def _range_product_floats(group) -> int:
    """Floats per chunk row one group's range product holds while it is formed.

    A level group also holds its level rows and bins, an exact one its rows'
    stored basis values; a group the range products decline is copied
    (``row_subset``) and owns its chunk rows, as :func:`_group_row_floats`
    counts them.
    """
    kind = type(group)
    if kind in _RANGE_PRODUCTS:
        return 1
    if kind in _SPLINE_CATEGORY_GROUPS:
        return 3
    if kind is SplineCategoricalGroupMatrix:
        basis = group.B
        entry = basis.data.itemsize + basis.indices.itemsize
        return 3 + math.ceil((entry * _max_row_values(basis) + basis.indptr.itemsize) / 8)
    return 1 + (_group_row_floats(group, chunk=True) if kind in _COUNTED_GROUPS else group.shape[1])


def chunk_row_floats(layout: StackedLayout) -> int:
    """Floats per row the likelihood stream holds beside its channels for one chunk.

    A layout of plain designs streams in row ranges (``_predictor_values``):
    each predictor holds its running sum and one group's range product at a
    time. Otherwise each chunk copies its predictor designs
    (:func:`_chunk_design_floats`).
    """
    if not _streams_in_ranges(layout):
        return _chunk_design_floats(layout)
    return sum(
        1 + max(map(_range_product_floats, state.design.group_matrices), default=0)
        for state in layout.predictors
    )


def _chunk_design_floats(layout: StackedLayout) -> int:
    """Floats per row that one likelihood chunk's copied predictor designs own.

    ``_predictor_chunk`` copies each group's rows (:func:`_row_design`): what
    a contraction block's design owns (:func:`contraction_row_bytes`), with a
    support's bin codes copied too, exact spline-by-category levels counted
    as disjoint, and no gram's dense rows, which belong to the contraction's
    budget. A group whose rows are not counted, or a design that is not a
    plain ``DesignMatrix``, owns its width.
    """
    floats = 0
    for state in layout.predictors:
        design = state.design
        if type(design) is not DesignMatrix:
            floats += design.p
            continue
        floats += _spline_category_row_floats(design)
        floats += _exact_spline_category_chunk_floats(design)
        for group in design.group_matrices:
            kind = type(group)
            if kind in _SPLINE_CATEGORY_GROUPS or kind is SplineCategoricalGroupMatrix:
                continue
            if kind not in _COUNTED_GROUPS:
                floats += group.shape[1]
                continue
            floats += _group_row_floats(group, chunk=True)
            floats += kind in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix)
    return floats


# A block's level and bin sums run over up to all its rows, and the design then
# multiplies them by products of entries up to about 2**23. Where a sum of that many channel
# values could leave float64's range, the block contracts chunk by chunk, so its
# sums are the chunk stream's. The pass tables sum over every block they have
# seen, so they are contracted before a block that would take them past it.
_BLOCK_SUM_LIMIT = 2.0**1000


def _channel_peak(*channels: NDArray[np.float64]) -> float:
    return max(max(float(np.max(values)), -float(np.min(values))) for values in channels)


def contraction_block_rows(n_observations: int, row_bytes: int | None, chunk_rows: int) -> int:
    """Rows contracted at once: whole likelihood chunks within the contraction allowance.

    Binned crossproducts amortise over a block's rows (Li and Wood 2020,
    Algorithm 1), so a block takes all the chunks the allowance holds. A block
    holds at least one chunk, so a single chunk larger than the allowance forms
    its own block, as does every chunk of an uncounted layout.
    """
    if row_bytes is None:
        return min(n_observations, chunk_rows)
    chunks_per_block = max(1, AUTO_CONTRACTION_MEMORY_BYTES // (row_bytes * chunk_rows))
    return min(n_observations, chunks_per_block * chunk_rows)


def _joint_plan(layout: StackedLayout, factors) -> JointTablePlan | None:
    """The layout's joint-table plan, formed once per set of designs a ``PassFactors`` serves."""
    if factors is None:
        return joint_table_plan(layout, AUTO_CONTRACTION_MEMORY_BYTES)
    # What the plan reads of a layout beside its designs: the offsets and intercept positions.
    held = tuple((state.design, state.offset) for state in layout.predictors)
    key = tuple((id(design), id(offset)) for design, offset in held)
    key += tuple(state.intercept_index for state in layout.predictors)
    entry = factors.joint_plans.get(key)
    if entry is None:
        entry = (held, joint_table_plan(layout, AUTO_CONTRACTION_MEMORY_BYTES))
        factors.joint_plans[key] = entry
    return entry[1]


# Rows one pool task evaluates. Fixed, so the partition and the summed likelihood's
# rounding do not depend on the workers.
_COMPILED_ROW_BLOCK = 2**15


def _compiled_rows(family, layout, y, likelihood_plan, joint, factors):
    """The family's compiled predictor rows for this fit, once per session; None without them.

    Owner: the session's ``PassFactors``; lifetime: the session's fit. The rows read the
    response, the weights and the links, each fixed for an entry by its key; no weight,
    penalty or coefficient of a pass reaches them, so nothing invalidates an entry.
    """
    hook = getattr(family, "predictor_rows", None)
    # A subclass may override the likelihood the hook's rows evaluate.
    if hook is None or type(family) not in _BUILTIN_FAMILIES or joint is None or factors is None:
        return None
    links = tuple(state.link for state in layout.predictors)
    # ``joint`` answers whether one power serves every row. Every context of a session
    # holds the session's one response (``canonical_response``), so one entry serves them.
    key = ("rows", id(family), id(y), id(likelihood_plan), id(joint), *map(id, links))
    entry = factors.joint_plans.get(key)
    if entry is None:
        rows = hook(
            np.asarray(y, dtype=np.float64), likelihood_plan, links, joint.shares_last_predictor
        )
        entry = factors.joint_plans[key] = ((family, y, likelihood_plan, joint, links), rows)
    return entry[1]


def compiled_rows_serve(family, layout, y, likelihood_plan, factors) -> bool:
    """Whether this fit's passes take the family's compiled rows, row declines aside."""
    if getattr(family, "predictor_rows", None) is None:
        return False
    joint = _joint_plan(layout, factors)
    return _compiled_rows(family, layout, y, likelihood_plan, joint, factors) is not None


def _compiled_likelihood(rows, joint, tables, constants, count, block_rows) -> float | None:
    """Every row's likelihood through the compiled route, each task's total added in row order.

    None where a row needs the family's natural route: its refusal or its rescaled arithmetic.
    """

    def task(first: int, last: int):
        scratch = np.empty((len(joint.sides), last - first))
        joint.fill_predictors(scratch, 0, last - first, first, tables, constants)
        return rows.evaluate(first, 0, last - first, scratch, 0)

    total = 0.0
    for completed, value in pooled_row_blocks(task, count, block_rows):
        if not completed:
            return None
        total += value
    return total


def _compiled_geometry(rows, joint, joint_tables, tables, constants) -> float | None:
    """Every row through the compiled route into the tables; the rows' likelihood, or None.

    None where a row needs the natural route, and shards not yet started are skipped. One
    task per shard, its stack held to a block of rows, so memory and the bits depend on
    the shards alone.
    """
    if joint.one_table_set:
        return _compiled_one_table_set(rows, joint, joint_tables, tables, constants)
    n_rows = joint.index.shape[1]
    size = joint.shard_rows
    declined = [False]

    def task(shard: int) -> float:
        begin, end = shard * size, min((shard + 1) * size, n_rows)
        stack = np.empty((joint.stack_rows, min(STACK_BLOCK_ROWS, end - begin)))
        total = 0.0
        for start in range(begin, end, STACK_BLOCK_ROWS):
            if declined[0]:
                break
            count = min(STACK_BLOCK_ROWS, end - start)
            joint.fill_predictors(stack, 0, count, start, tables, constants)
            completed, value = rows.evaluate(start, 0, count, stack, 2)
            if not completed or not joint_tables.add_block(
                shard, stack, 0, count, start, joint.n_inputs
            ):
                declined[0] = True
                break
            total += value
        return total

    totals = pooled_for(task, list(range(joint.n_shards)), 8 * joint.stack_rows * STACK_BLOCK_ROWS)
    return None if declined[0] else sum(totals)


def _compiled_one_table_set(rows, joint, joint_tables, tables, constants) -> float | None:
    """:func:`_compiled_geometry` where the tables leave one set: morsels, then a task per table."""
    n_rows = joint.index.shape[1]
    stack = np.empty((joint.stack_rows, min(joint.block_rows, n_rows)))
    total = 0.0
    for start in range(0, n_rows, joint.block_rows):
        count = min(joint.block_rows, n_rows - start)

        def morsel(first: int, last: int, start: int = start) -> float | None:
            joint.fill_predictors(stack, first, last, start, tables, constants)
            completed, value = rows.evaluate(start, first, last, stack, 2)
            ready = completed and joint_tables.prepare_rows(stack, first, last, joint.n_inputs)
            return value if ready else None

        values = pooled_row_blocks(morsel, count, MORSEL_ROWS)
        if None in values:
            return None
        total += sum(values)
        joint_tables.add_by_table(stack, 0, count, start)
    return total


def assemble_chunked_geometry(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    coefficients: NDArray,
    *,
    penalty: NDArray,
    chunk_size: ChunkSize,
    curvature_source: CurvatureSource,
    likelihood_cache=None,
    pass_factors=None,
    _likelihood_sums: list[float] | None = None,
    _compiled: bool = True,
) -> DenseJointGeometry:
    """Stream likelihood chunks into one coefficient-space geometry.

    Binned crossproducts amortise only when each bin-pair weight table is
    contracted once over many rows (Li and Wood 2020, Algorithm 1). The
    chunks stream score and packed curvature channels into a buffer, and the
    group plans contract each block of chunks once; a block covers every row
    up to the allowance. A pass of several blocks keeps the by-bin and joint
    tables its pass tables admit, each block adding its own, and contracts
    them once, within the allowance the block leaves.
    """
    n_observations = layout.predictors[0].design.n
    k_parameters = len(layout.predictors)
    chunk_rows = layout_chunk_size(layout, chunk_size)
    # Joint tables keep no design rows and contract once per assembly, so a
    # block's rows hold only its channels and a block may run to what the shards'
    # tables leave of the allowance.
    joint = _joint_plan(layout, pass_factors)
    row_bytes = contraction_row_bytes(layout) if joint is None else 8 * joint.stack_rows
    block_rows = contraction_block_rows(n_observations, row_bytes, chunk_rows)
    if joint is not None:
        left = (AUTO_CONTRACTION_MEMORY_BYTES - joint.shard_bytes) // (row_bytes * chunk_rows)
        block_rows = min(block_rows, max(1, left) * chunk_rows)
    # A block of one chunk out of several contracts that chunk's rows as it
    # arrives, so nothing is buffered.
    per_chunk = joint is None and block_rows <= chunk_rows < n_observations
    # A block of many rows forms each group pair's own product, as the unchunked
    # assembly does; panels, like tables, serve a pass of many short blocks.
    panels = (None,) * k_parameters
    if per_chunk:
        panels = predictor_panels(layout)
        row_bytes = contraction_row_bytes(layout, panels)
    packed = any(panel is not None for panel in panels)
    table_budget = AUTO_CONTRACTION_MEMORY_BYTES - (row_bytes or 0) * block_rows
    tables = (
        PassTables(table_budget)
        if joint is None and block_rows < n_observations and table_budget > 0
        else None
    )
    accumulator = GroupedGeometryAccumulator(
        layout,
        penalty=penalty,
        coefficients=coefficients,
        tables=tables,
        panels=panels,
        factors=pass_factors,
        joint=joint,
    )
    # The stream evaluates rows in ranges and builds no chunk designs; each
    # block reads its rows from the layout's own designs.
    formation = FormationAccumulator(layout, range_geometry=True)
    # Rows and channel peak the pass tables have summed; a table entry sums at
    # most that many rows of that size.
    summed = [0, 0.0]
    # Rows the joint tables took, for the certificate's statistics.
    joint_rows = [0]

    def published() -> DenseJointGeometry:
        if joint is not None:
            accumulator.joint_tables.merge()
        if joint_rows[0]:
            formation.add_joint_rows(joint_rows[0], joint.n_shards, accumulator.joint_tables)
        return with_formation(accumulator.finish(), formation.finish())

    compiled = (
        _compiled_rows(family, layout, y, likelihood_plan, joint, pass_factors)
        if _compiled and curvature_source == "observed"
        else None
    )
    if compiled is not None:
        tables_, constants = joint.predictor_tables(np.asarray(coefficients, dtype=np.float64))
        total = _compiled_geometry(compiled, joint, accumulator.joint_tables, tables_, constants)
        if total is None:
            accumulator.joint_tables.release()
            return assemble_chunked_geometry(
                family, layout, y, likelihood_plan, coefficients, penalty=penalty,
                chunk_size=chunk_size, curvature_source=curvature_source,
                likelihood_cache=likelihood_cache, pass_factors=pass_factors,
                _likelihood_sums=_likelihood_sums, _compiled=False,
            )  # fmt: skip
        accumulator.note_joint_rows()
        joint_rows[0] = n_observations
        if _likelihood_sums is not None:
            _likelihood_sums[0] += total
        return published()

    buffered = 0 if per_chunk else block_rows
    n_channels = k_parameters * (k_parameters + 1) // 2
    # Channel-major, so each contracted channel is one contiguous row vector. A joint
    # block keeps its score channels after them, for one array to read.
    stack = np.empty((n_channels if joint is None else joint.stack_rows, buffered))
    curvature = stack[:n_channels]
    score = (
        np.empty((buffered, k_parameters), dtype=np.float64)
        if joint is None
        else stack[n_channels : joint.n_inputs]
    )

    def keep_tables_in_range(rows: int, peak: float) -> None:
        peak = max(summed[1], peak)
        if (summed[0] + rows) * peak > _BLOCK_SUM_LIMIT:
            accumulator.release_tables()
        summed[0] += rows
        summed[1] = peak

    def row_plans(start: int, stop: int):
        if start == 0 and stop == n_observations and not packed:
            return _grouped_predictor_plans(layout)
        indices = np.arange(start, stop, dtype=np.intp)
        indices.setflags(write=False)
        rows = RowChunk(start=start, stop=stop, indices=indices)
        plans = []
        for state, panel in zip(layout.predictors, panels, strict=True):
            if panel is None:
                design = _row_design(state, rows, copy=False)
                plans.append(PredictorExecutionPlan(design, state.intercept_index is not None))
                continue
            groups = state.design.group_matrices
            rest = []
            for index in panel.rest:
                child = group_row_range(groups[index], start, stop, copy=False)
                rest.append(groups[index].row_subset(indices) if child is None else child)
            design = panel.packed_design(panel.panel(state.design, start, stop), rest, stop - start)
            plans.append(PredictorExecutionPlan(design, False))
        return tuple(plans)

    formation_cost = k_parameters * (k_parameters + 1) // 2 + sum(
        len(state.design.group_matrices) for state in layout.predictors
    )

    def formation_task(start: int, stop: int, channels: NDArray) -> BlockTask:
        # The certificate's row majorants, formed beside the block's Gram blocks.
        return BlockTask(
            index=0,
            cost=float((stop - start) * formation_cost),
            nbytes=8 * (stop - start) * (k_parameters + 2),
            run=lambda _cache, _profile: formation.add(start, stop, channels),
            place=lambda _value: None,
        )

    def contract(start: int, stop: int) -> None:
        filled = stop - start
        step = filled
        if joint is not None and stack_in_range(stack[: joint.n_inputs], filled):
            accumulator.add_joint_rows(start, stop, stack)
            joint_rows[0] += filled
            return
        # Channels at the edge of float64's range keep the group routes, in their score layout.
        block_score = score[:filled] if joint is None else score[:, :filled].T
        if tables is not None or filled > chunk_rows:
            peak = _channel_peak(block_score, curvature[:, :filled])
            if tables is not None:
                keep_tables_in_range(filled, peak)
            if filled > chunk_rows and filled * peak > _BLOCK_SUM_LIMIT:
                step = chunk_rows
        for piece in range(start, stop, step):
            end = min(piece + step, stop)
            plans = row_plans(piece, end)
            rows = slice(piece - start, end - start)
            # One add per contracted piece, over its contiguous channel rows.
            accumulator.add_block(
                plans,
                block_score[rows],
                curvature[:, rows],
                beside=formation_task(piece, end, curvature[:, rows].T),
            )

    iterator = iter_likelihood_chunks(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
        chunk_size=chunk_size,
        curvature_source=curvature_source,
        _range_geometry=True,
        **({"_recover_derivative_failure": True} if _likelihood_sums is not None else {}),
        **({"likelihood_cache": likelihood_cache} if likelihood_cache is not None else {}),
    )
    # Chunks arrive in row order and cover every row once.
    block_start = position = 0
    try:
        for chunk in iterator:
            length = len(chunk.score_eta)
            if per_chunk:
                if tables is not None:
                    keep_tables_in_range(
                        length, _channel_peak(chunk.score_eta, chunk.curvature_packed)
                    )
                accumulator.add_block(
                    row_plans(position, position + length),
                    chunk.score_eta,
                    chunk.curvature_packed.T,
                    beside=formation_task(position, position + length, chunk.curvature_packed),
                )
                position = block_start = position + length
            else:
                if position + length - block_start > block_rows:
                    contract(block_start, position)
                    block_start = position
                rows = slice(position - block_start, position + length - block_start)
                position += length
                if joint is None:
                    score[rows] = chunk.score_eta
                else:
                    score[:, rows] = chunk.score_eta.T
                curvature[:, rows] = chunk.curvature_packed.T
            if _likelihood_sums is not None:
                _likelihood_sums[0] += float(
                    np.sum(chunk.optimizing_log_likelihood, dtype=np.float64)
                )
                _likelihood_sums[1] += float(
                    np.sum(chunk.parameter_independent_carrier, dtype=np.float64)
                )
            chunk = None
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if position != n_observations:
        raise RuntimeError("likelihood chunks must cover every row exactly once")
    if block_start < position:
        contract(block_start, position)
    return published()


def _joint_predictor_rows(joint, values, row_block, read, *, offsets: bool) -> list:
    """``[read(first, last, predictors), ...]`` per pool block: ``predictors`` is ``(K, rows)``."""
    tables, constants = joint.predictor_tables(values)
    predictors = len(joint.sides)

    def task(first: int, last: int):
        scratch = np.empty((predictors, last - first))
        joint.fill_predictors(scratch, 0, last - first, first, tables, constants, offsets=offsets)
        if not joint.predictors_finite(scratch, last - first):
            raise ValueError("chunk predictor evaluation produced non-finite values")
        return read(first, last, scratch)

    return pooled_row_blocks(task, joint.index.shape[1], row_block)


def maximum_chunked_predictor_change(
    layout: StackedLayout,
    coefficient_step: NDArray,
    *,
    chunk_size: ChunkSize,
    pass_factors=None,
) -> float:
    """Return the largest absolute linear-predictor change without full rows."""
    step = _validated_coefficients(layout, coefficient_step)
    n_observations = layout.predictors[0].design.n
    chunk_rows = layout_chunk_size(layout, chunk_size)
    joint = None if pass_factors is None else _joint_plan(layout, pass_factors)
    if joint is not None:

        def largest(_first, _last, change) -> float:
            return float(np.max(np.abs(change), initial=0.0))

        maximum = float(
            np.max(
                _joint_predictor_rows(
                    joint, step, min(_COMPILED_ROW_BLOCK, chunk_rows), largest, offsets=False
                )
            )
        )
        if not math.isfinite(maximum):
            raise ValueError("chunk predictor evaluation produced non-finite values")
        return maximum
    maximum = 0.0
    support_predictions = _SupportPredictions()
    for rows in iter_row_chunks(n_observations, chunk_rows):
        change = _predictor_values(
            layout,
            step,
            rows,
            include_offsets=False,
            support_predictions=support_predictions,
        )
        maximum = max(maximum, float(np.max(np.abs(change), initial=0.0)))
    return maximum


def materialize_terminal_predictions(
    layout: StackedLayout,
    coefficients: NDArray,
    *,
    chunk_size: ChunkSize,
    pass_factors=None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build required terminal ``n x K`` predictions in one final bounded pass."""
    coefficient_values = _validated_coefficients(layout, coefficients)
    n_observations = layout.predictors[0].design.n
    k_parameters = len(layout.predictors)
    eta = np.empty((n_observations, k_parameters), dtype=np.float64)
    theta = np.empty_like(eta)
    chunk_rows = layout_chunk_size(layout, chunk_size)
    joint = None if pass_factors is None else _joint_plan(layout, pass_factors)
    if joint is not None:

        def place(first: int, last: int, predictors: NDArray) -> None:
            eta[first:last] = predictors.T
            theta[first:last] = _theta_chunk(layout, eta[first:last])

        _joint_predictor_rows(
            joint, coefficient_values, min(_COMPILED_ROW_BLOCK, chunk_rows), place, offsets=True
        )
    else:
        support_predictions = _SupportPredictions()
        for rows in iter_row_chunks(n_observations, chunk_rows):
            eta_chunk = _predictor_values(
                layout,
                coefficient_values,
                rows,
                include_offsets=True,
                support_predictions=support_predictions,
            )
            theta_chunk = _theta_chunk(layout, eta_chunk)
            eta[rows.start : rows.stop] = eta_chunk
            theta[rows.start : rows.stop] = theta_chunk
    eta.setflags(write=False)
    theta.setflags(write=False)
    return eta, theta


def predictions_are(
    layout: StackedLayout,
    coefficients: NDArray,
    eta: NDArray,
    theta: NDArray,
    *,
    chunk_size: ChunkSize,
    pass_factors=None,
) -> bool:
    """Whether the coefficients' predictors and parameters are, bit for bit, ``eta`` and ``theta``."""
    n_observations = layout.predictors[0].design.n
    chunk_rows = layout_chunk_size(layout, chunk_size)
    joint = None if pass_factors is None else _joint_plan(layout, pass_factors)
    if joint is not None:

        def same(first: int, last: int, predictors: NDArray) -> bool:
            rows = predictors.T
            return np.array_equal(rows, eta[first:last]) and np.array_equal(
                _theta_chunk(layout, rows), theta[first:last]
            )

        return all(
            _joint_predictor_rows(
                joint, coefficients, min(_COMPILED_ROW_BLOCK, chunk_rows), same, offsets=True
            )
        )
    support_predictions = _SupportPredictions()
    for rows in iter_row_chunks(n_observations, chunk_rows):
        values = _predictor_values(
            layout,
            coefficients,
            rows,
            include_offsets=True,
            support_predictions=support_predictions,
        )
        if not np.array_equal(values, eta[rows.start : rows.stop]) or not np.array_equal(
            _theta_chunk(layout, values), theta[rows.start : rows.stop]
        ):
            return False
    return True


__all__ = [
    "AUTO_CHUNK_MEMORY_BYTES",
    "ChunkSize",
    "ChunkedLikelihoodSums",
    "LikelihoodChunk",
    "RowChunk",
    "assemble_chunked_geometry",
    "evaluate_chunked_log_likelihood",
    "iter_likelihood_chunks",
    "iter_row_chunks",
    "materialize_terminal_predictions",
    "maximum_chunked_predictor_change",
    "layout_chunk_size",
    "resolve_chunk_size",
]
