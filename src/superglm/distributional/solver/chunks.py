"""Bounded-row evaluation for distributional predictors and likelihoods."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    LikelihoodPlanValidatingFamily,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.solver.assembly import DenseJointGeometry, GroupedGeometryAccumulator
from superglm.distributional.solver.derivatives import (
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)

ChunkSize = int | Literal["auto"]
CurvatureSource = Literal["observed", "fisher"]

# Includes eta, theta, natural/transformed derivatives, masks, and family temporaries.
# It is a deterministic bound selector, not a claim about exact allocator RSS.
AUTO_CHUNK_SELECTOR = "distributional-auto-v1"
AUTO_CHUNK_MEMORY_BYTES = 8 * 1024 * 1024


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
    p_coefficients: int = 0,
) -> int:
    """Resolve an explicit or memory-budgeted row bound."""
    rows = _positive_integer(n_observations, name="n_observations")
    parameters = _positive_integer(k_parameters, name="k_parameters")
    if isinstance(p_coefficients, bool):
        raise TypeError("p_coefficients must be a non-negative integer")
    try:
        width = operator.index(p_coefficients)
    except TypeError as exc:
        raise TypeError("p_coefficients must be a non-negative integer") from exc
    if width < 0:
        raise ValueError("p_coefficients must be a non-negative integer")
    if chunk_size == "auto":
        n_channels = parameters * (parameters + 1) // 2
        estimated_float_columns = width + 6 * parameters + 4 * n_channels + 4
        budget_rows = AUTO_CHUNK_MEMORY_BYTES // (
            np.dtype(np.float64).itemsize * estimated_float_columns
        )
        return min(rows, max(1, int(budget_rows)))
    if isinstance(chunk_size, str):
        raise ValueError("chunk_size string must be 'auto'")
    return min(rows, _positive_integer(chunk_size, name="chunk_size"))


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
    p_coefficients: int = 0,
):
    """Yield contiguous row chunks that cover every observation exactly once."""
    rows = _positive_integer(n_observations, name="n_observations")
    size = resolve_chunk_size(
        rows,
        k_parameters,
        chunk_size,
        p_coefficients=p_coefficients,
    )
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

    @property
    def reported_log_likelihood(self) -> NDArray[np.float64]:
        result = np.array(
            self.optimizing_log_likelihood + self.parameter_independent_carrier,
            dtype=np.float64,
            copy=True,
        )
        result.setflags(write=False)
        return result


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


def _predictor_chunk(
    layout: StackedLayout,
    coefficients: NDArray[np.float64],
    rows: RowChunk,
    *,
    include_offsets: bool,
) -> tuple[NDArray[np.float64], tuple[PredictorExecutionPlan, ...]]:
    k_parameters = len(layout.predictors)
    eta = np.empty((len(rows.indices), k_parameters), dtype=np.float64)
    plans: list[PredictorExecutionPlan] = []
    for state in layout.predictors:
        design = state.design.row_subset(rows.indices)
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
) -> NDArray[np.float64]:
    theta = np.empty_like(eta)
    for state in layout.predictors:
        values = np.asarray(state.link.inverse(eta[:, state.parameter_index]), dtype=np.float64)
        if values.shape != (len(eta),) or not np.all(np.isfinite(values)):
            raise ValueError(f"inverse link for predictor {state.name!r} produced an invalid chunk")
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


def iter_likelihood_chunks(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    coefficients: NDArray,
    *,
    chunk_size: ChunkSize,
    curvature_source: CurvatureSource,
):
    """Yield bounded predictor derivatives and curvature for each row chunk."""
    if curvature_source not in ("observed", "fisher"):
        raise ValueError("curvature_source must be 'observed' or 'fisher'")
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
    k_parameters = len(links)
    for rows in iter_row_chunks(
        len(response),
        chunk_size,
        k_parameters=k_parameters,
        p_coefficients=layout.n_coefficients,
    ):
        eta, plans = _predictor_chunk(
            layout,
            coefficient_values,
            rows,
            include_offsets=True,
        )
        theta = _theta_chunk(layout, eta)
        child_plan = plan.take(rows.indices)
        if len(child_plan.weights.values) != len(rows.indices):
            raise UnsupportedLikelihoodContractError(
                "family likelihood slicing returned the wrong number of rows"
            )
        natural = family.evaluate_natural(
            response[rows.indices],
            theta,
            child_plan,
            derivative_order=2,
        )
        if natural.derivative_order != 2:
            raise ValueError("family must return exact derivative order 2 for chunk geometry")
        if natural.valid is not None and not np.all(natural.valid):
            raise ValueError("chunk contains an invalid likelihood state")
        transformed = transform_natural_derivatives(natural, eta, links)
        if curvature_source == "observed":
            curvature = transformed.curvature_packed
        else:
            assert isinstance(family, ExpectedInformationFamily)
            information = family.expected_information_natural(theta, child_plan)
            curvature = transform_natural_information(information, eta, links)
        yield LikelihoodChunk(
            rows=rows,
            plans=plans,
            eta=eta,
            theta=theta,
            optimizing_log_likelihood=transformed.optimizing_log_likelihood,
            parameter_independent_carrier=transformed.parameter_independent_carrier,
            score_eta=transformed.score_eta,
            curvature_packed=curvature,
        )


def evaluate_chunked_log_likelihood(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    coefficients: NDArray,
    *,
    chunk_size: ChunkSize,
) -> ChunkedLikelihoodSums:
    """Evaluate only the scalar weighted likelihood in bounded row chunks."""
    response, plan, coefficient_values = _validate_chunk_inputs(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
    )
    optimizing_total = 0.0
    carrier_total = 0.0
    for rows in iter_row_chunks(
        len(response),
        chunk_size,
        k_parameters=len(layout.predictors),
        p_coefficients=layout.n_coefficients,
    ):
        eta, _plans = _predictor_chunk(
            layout,
            coefficient_values,
            rows,
            include_offsets=True,
        )
        theta = _theta_chunk(layout, eta)
        child_plan = plan.take(rows.indices)
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
            raise ValueError("family must return exact derivative order 0 for chunk values")
        if natural.valid is not None and not np.all(natural.valid):
            raise ValueError("chunk contains an invalid likelihood state")
        optimizing_total += float(np.sum(natural.optimizing_log_likelihood, dtype=np.float64))
        carrier_total += float(np.sum(natural.parameter_independent_carrier, dtype=np.float64))
    return ChunkedLikelihoodSums(
        optimizing_log_likelihood=optimizing_total,
        parameter_independent_carrier=carrier_total,
    )


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
) -> DenseJointGeometry:
    """Stream likelihood chunks into one coefficient-space geometry."""
    accumulator = GroupedGeometryAccumulator(
        layout,
        penalty=penalty,
        coefficients=coefficients,
    )
    for chunk in iter_likelihood_chunks(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
        chunk_size=chunk_size,
        curvature_source=curvature_source,
    ):
        accumulator.add_score(chunk.plans, chunk.score_eta)
        for channel_index in range(chunk.curvature_packed.shape[1]):
            accumulator.add_curvature_channel(
                chunk.plans,
                channel_index,
                chunk.curvature_packed[:, channel_index],
            )
    return accumulator.finish()


def maximum_chunked_predictor_change(
    layout: StackedLayout,
    coefficient_step: NDArray,
    *,
    chunk_size: ChunkSize,
) -> float:
    """Return the largest absolute linear-predictor change without full rows."""
    step = _validated_coefficients(layout, coefficient_step)
    maximum = 0.0
    n_observations = layout.predictors[0].design.n
    for rows in iter_row_chunks(
        n_observations,
        chunk_size,
        k_parameters=len(layout.predictors),
        p_coefficients=layout.n_coefficients,
    ):
        change, _plans = _predictor_chunk(
            layout,
            step,
            rows,
            include_offsets=False,
        )
        maximum = max(maximum, float(np.max(np.abs(change), initial=0.0)))
    return maximum


def materialize_terminal_predictions(
    layout: StackedLayout,
    coefficients: NDArray,
    *,
    chunk_size: ChunkSize,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build required terminal ``n x K`` predictions in one final bounded pass."""
    coefficient_values = _validated_coefficients(layout, coefficients)
    n_observations = layout.predictors[0].design.n
    k_parameters = len(layout.predictors)
    eta = np.empty((n_observations, k_parameters), dtype=np.float64)
    theta = np.empty_like(eta)
    for rows in iter_row_chunks(
        n_observations,
        chunk_size,
        k_parameters=k_parameters,
        p_coefficients=layout.n_coefficients,
    ):
        eta_chunk, _plans = _predictor_chunk(
            layout,
            coefficient_values,
            rows,
            include_offsets=True,
        )
        theta_chunk = _theta_chunk(layout, eta_chunk)
        eta[rows.start : rows.stop] = eta_chunk
        theta[rows.start : rows.stop] = theta_chunk
    eta.setflags(write=False)
    theta.setflags(write=False)
    return eta, theta


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
    "resolve_chunk_size",
]
