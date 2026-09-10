"""Bounded-row evaluation for distributional predictors and likelihoods."""

from __future__ import annotations

import operator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
)
from superglm._group_matrix._group_matrix_range import group_range_matvec, group_row_range
from superglm.distributional._global_moment_policy import automatic_global_moment_budget
from superglm.distributional._panel_policy import automatic_small_group_panel_budget
from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    LikelihoodPlanValidatingFamily,
)
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.solver._global_moments import (
    GlobalMomentRefusalError,
    build_global_moment_plan,
    global_moment_chunk_size,
)
from superglm.distributional.solver._small_group_panels import build_small_group_panels
from superglm.distributional.solver.assembly import DenseJointGeometry, GroupedGeometryAccumulator
from superglm.distributional.solver.derivatives import (
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.group_matrix import DesignMatrix

ChunkSize = int | Literal["auto"]
CurvatureSource = Literal["observed", "fisher"]

# Includes eta, theta, natural/transformed derivatives, masks, and family temporaries.
# It is a deterministic bound selector, not a claim about exact allocator RSS.
AUTO_CHUNK_SELECTOR = "distributional-auto-v1"
AUTO_CHUNK_MEMORY_BYTES = 8 * 1024 * 1024

# Retained snapshots plus support vectors per pass; refresh uses bounded scratch.
# No observation-sized predictions are retained here.
_SUPPORT_PREDICTION_BYTES = 1024 * 1024


class _SupportPredictions:
    """Reuse small support products while checking mutable algebra each chunk."""

    def __init__(self):
        self.entries = {}
        self.nbytes = 0

    def __call__(self, group, beta):
        key = id(group)
        source = (group.B_unique, group.R_inv, beta)
        entry = self.entries.get(key)
        if entry is not None:
            snapshots, values = entry
            if all(np.array_equal(a, b) for a, b in zip(source, snapshots, strict=True)):
                return values
            self.nbytes -= sum(a.nbytes for a in snapshots) + values.nbytes
            del self.entries[key]
            del entry, snapshots, values
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


def _resolve_fitting_chunk_size(family, layout, likelihood_plan, chunk_size):
    """Resolve one row bound for all phases of an admitted global fit.

    Larger batches amortize native scheduling and likelihood preparation. The
    global assembler's shared estimate limits its additional workspace; the
    caller's likelihood arrays remain separately bounded by the returned rows.
    Explicit requests and unsupported models retain the ordinary selector.
    """
    if chunk_size is None:
        return None
    resolved = resolve_chunk_size(
        layout.predictors[0].design.n,
        len(layout.predictors),
        chunk_size,
        p_coefficients=layout.n_coefficients,
    )
    if type(chunk_size) is str and chunk_size == "auto":
        budget = automatic_global_moment_budget(family, likelihood_plan, layout)
        if budget is not None:
            return global_moment_chunk_size(layout, byte_budget=budget, minimum_chunk_size=resolved)
    return resolved


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
        if type(state.design) is DesignMatrix:
            groups = []
            for group in state.design.group_matrices:
                child = group_row_range(group, rows.start, rows.stop)
                groups.append(group.row_subset(rows.indices) if child is None else child)
            design = DesignMatrix(groups, len(rows.indices), state.design.p)
        else:
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


def _take_likelihood_rows(plan, rows, likelihood_cache):
    if likelihood_cache is None:
        return plan.take(rows.indices)
    return likelihood_cache.take(plan, rows.indices, start=rows.start, stop=rows.stop)


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

    Private global assembly may retain original plans and consume their row
    ranges directly. The default stream continues to supply owned child plans.
    """
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
    for rows in iter_row_chunks(
        len(response),
        chunk_size,
        k_parameters=k_parameters,
        p_coefficients=layout.n_coefficients,
    ):
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
        child_plan = _take_likelihood_rows(plan, rows, likelihood_cache)
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
    likelihood_cache=None,
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
    support_predictions = _SupportPredictions()
    for rows in iter_row_chunks(
        len(response),
        chunk_size,
        k_parameters=len(layout.predictors),
        p_coefficients=layout.n_coefficients,
    ):
        eta = _predictor_values(
            layout,
            coefficient_values,
            rows,
            include_offsets=True,
            support_predictions=support_predictions,
        )
        theta = _theta_chunk(layout, eta)
        child_plan = _take_likelihood_rows(plan, rows, likelihood_cache)
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
    small_group_panel_byte_budget: int | Literal["auto"] | None = "auto",
    likelihood_cache=None,
) -> DenseJointGeometry:
    """Stream one geometry, retaining the existing geometry-only entry point."""
    return _assemble_chunked_geometry(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
        penalty=penalty,
        chunk_size=chunk_size,
        curvature_source=curvature_source,
        small_group_panel_byte_budget=small_group_panel_byte_budget,
        likelihood_cache=likelihood_cache,
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
    small_group_panel_byte_budget="auto",
    likelihood_cache=None,
) -> tuple[DenseJointGeometry, ChunkedLikelihoodSums]:
    """Return owned geometry and scalar likelihood sums from the same stream."""
    sums = [0.0, 0.0]
    geometry = _assemble_chunked_geometry(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
        penalty=penalty,
        chunk_size=chunk_size,
        curvature_source=curvature_source,
        small_group_panel_byte_budget=small_group_panel_byte_budget,
        likelihood_cache=likelihood_cache,
        _likelihood_sums=sums,
    )
    if not all(np.isfinite(value) for value in sums):
        raise _TrialDerivativeError("chunked likelihood sums must be finite")
    return geometry, ChunkedLikelihoodSums(*sums)


def _assemble_chunked_geometry(
    family,
    layout,
    y,
    likelihood_plan,
    coefficients,
    *,
    penalty,
    chunk_size,
    curvature_source,
    small_group_panel_byte_budget="auto",
    likelihood_cache=None,
    _likelihood_sums=None,
) -> DenseJointGeometry:
    """Stream likelihood chunks into one coefficient-space geometry.

    Automatic execution may accumulate support moments across chunks in the
    admitted large mixed-layout scope, otherwise using bounded panels. Both
    have an additional 64 MiB allowance; the row chunk policy is unchanged.
    Explicit ``None`` requests grouped contraction; an integer requests the
    existing panel builder. Recoverable global numerical refusal releases the
    partial state before replaying the complete ordinary chunk stream.
    """
    automatic = (
        type(small_group_panel_byte_budget) is str and small_group_panel_byte_budget == "auto"
    )
    panel_byte_budget = (
        automatic_small_group_panel_budget(layout) if automatic else small_group_panel_byte_budget
    )
    if automatic:
        global_budget = automatic_global_moment_budget(family, likelihood_plan, layout)
        if global_budget is not None:
            built = build_global_moment_plan(
                layout,
                byte_budget=global_budget,
                chunk_size=resolve_chunk_size(
                    layout.predictors[0].design.n,
                    len(layout.predictors),
                    chunk_size,
                    p_coefficients=layout.n_coefficients,
                ),
            )
            if built.plan is not None:
                result = _try_global_geometry(
                    built.plan,
                    family,
                    layout,
                    y,
                    likelihood_plan,
                    coefficients,
                    penalty=penalty,
                    chunk_size=chunk_size,
                    curvature_source=curvature_source,
                    likelihood_cache=likelihood_cache,
                    _likelihood_sums=_likelihood_sums,
                )
                if result is not None:
                    return result
                if _likelihood_sums is not None:
                    _likelihood_sums[:] = [0.0, 0.0]
    return _assemble_grouped_chunk_geometry(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
        penalty=penalty,
        chunk_size=chunk_size,
        curvature_source=curvature_source,
        panel_byte_budget=panel_byte_budget,
        likelihood_cache=likelihood_cache,
        _likelihood_sums=_likelihood_sums,
    )


def _try_global_geometry(
    plan,
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
    _likelihood_sums=None,
) -> DenseJointGeometry | None:
    """Own one attempt; no exception/stream frame survives into fallback."""
    iterator = None
    chunk = None
    try:
        plan.reset(coefficients=coefficients, penalty=penalty)
        iterator = iter_likelihood_chunks(
            family,
            layout,
            y,
            likelihood_plan,
            coefficients,
            chunk_size=chunk_size,
            curvature_source=curvature_source,
            _range_geometry=hasattr(plan, "add_row_range"),
            **({"_recover_derivative_failure": True} if _likelihood_sums is not None else {}),
            **({"likelihood_cache": likelihood_cache} if likelihood_cache is not None else {}),
        )
        expected_start = 0
        n = layout.predictors[0].design.n
        for chunk in iterator:
            # The built-in iterator supplies contiguous positional ranges. Do
            # not accept a repeated, skipped or malformed stream as a numerical
            # refusal that could hide a source/iterator contract violation.
            rows = chunk.rows
            if (
                type(rows) is not RowChunk
                or rows.start != expected_start
                or not rows.start < rows.stop <= n
                or rows.stop - rows.start != len(chunk.score_eta)
            ):
                raise ValueError("global moment stream has inconsistent row ranges")
            if hasattr(plan, "add_row_range") and all(
                predictor.design is state.design
                for predictor, state in zip(chunk.plans, layout.predictors, strict=True)
            ):
                plan.add_row_range(
                    chunk.plans, rows.start, rows.stop, chunk.score_eta, chunk.curvature_packed
                )
            else:
                # Preserve streams supplied by callers which own child plans.
                plan.add_chunk(chunk.plans, chunk.score_eta, chunk.curvature_packed)
            if _likelihood_sums is not None:
                _likelihood_sums[0] += float(
                    np.sum(chunk.optimizing_log_likelihood, dtype=np.float64)
                )
                _likelihood_sums[1] += float(
                    np.sum(chunk.parameter_independent_carrier, dtype=np.float64)
                )
            expected_start = rows.stop
            chunk = None
        if expected_start != n:
            raise ValueError("global moment stream does not cover all observations")
        return plan.finish()
    except _TrialDerivativeError as exc:
        if exc.rows is not None and exc.plans is not None and hasattr(plan, "_prepare_chunk"):
            # A derivative can fail before add_row_range has checked live B/R,
            # indices and metadata. Preserve hard-source priority on this rare
            # path using the same preparation checks, without updating moments.
            rows = exc.rows
            k = len(layout.predictors)
            zero = np.zeros(())
            score = np.broadcast_to(zero, (rows.stop - rows.start, k))
            curvature = np.broadcast_to(zero, (len(score), k * (k + 1) // 2))
            row_range = (
                (rows.start, rows.stop)
                if all(
                    p.design is s.design for p, s in zip(exc.plans, layout.predictors, strict=True)
                )
                else None
            )
            try:
                plan._prepare_chunk(exc.plans, score, curvature, row_range=row_range)
            except GlobalMomentRefusalError as refusal:
                if not refusal.recoverable:
                    raise
        raise
    except GlobalMomentRefusalError as exc:
        if not exc.recoverable:
            raise
        return None
    finally:
        try:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        finally:
            chunk = None
            plan.close()


def _assemble_grouped_chunk_geometry(
    family,
    layout,
    y,
    likelihood_plan,
    coefficients,
    *,
    penalty,
    chunk_size,
    curvature_source,
    panel_byte_budget,
    likelihood_cache=None,
    _likelihood_sums=None,
) -> DenseJointGeometry:
    accumulator = GroupedGeometryAccumulator(
        layout,
        penalty=penalty,
        coefficients=coefficients,
    )
    iterator = iter_likelihood_chunks(
        family,
        layout,
        y,
        likelihood_plan,
        coefficients,
        chunk_size=chunk_size,
        curvature_source=curvature_source,
        **({"_recover_derivative_failure": True} if _likelihood_sums is not None else {}),
        **({"likelihood_cache": likelihood_cache} if likelihood_cache is not None else {}),
    )
    chunk = None
    try:
        for chunk in iterator:
            workspace = None
            if panel_byte_budget is not None:
                workspace = build_small_group_panels(
                    chunk.plans,
                    slice(0, chunk.plans[0].design.n),
                    byte_budget=panel_byte_budget,
                ).workspace
            try:
                accumulator.add_score(chunk.plans, chunk.score_eta)
                for channel_index in range(chunk.curvature_packed.shape[1]):
                    accumulator.add_curvature_channel(
                        chunk.plans,
                        channel_index,
                        chunk.curvature_packed[:, channel_index],
                        panel_workspace=workspace,
                    )
                if _likelihood_sums is not None:
                    _likelihood_sums[0] += float(
                        np.sum(chunk.optimizing_log_likelihood, dtype=np.float64)
                    )
                    _likelihood_sums[1] += float(
                        np.sum(chunk.parameter_independent_carrier, dtype=np.float64)
                    )
            finally:
                # Release before advancing or unwinding the likelihood stream.
                if workspace is not None:
                    workspace.close()
            chunk = None
    finally:
        chunk = None
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
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
    support_predictions = _SupportPredictions()
    n_observations = layout.predictors[0].design.n
    for rows in iter_row_chunks(
        n_observations,
        chunk_size,
        k_parameters=len(layout.predictors),
        p_coefficients=layout.n_coefficients,
    ):
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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build required terminal ``n x K`` predictions in one final bounded pass."""
    coefficient_values = _validated_coefficients(layout, coefficients)
    n_observations = layout.predictors[0].design.n
    k_parameters = len(layout.predictors)
    eta = np.empty((n_observations, k_parameters), dtype=np.float64)
    theta = np.empty_like(eta)
    support_predictions = _SupportPredictions()
    for rows in iter_row_chunks(
        n_observations,
        chunk_size,
        k_parameters=k_parameters,
        p_coefficients=layout.n_coefficients,
    ):
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
