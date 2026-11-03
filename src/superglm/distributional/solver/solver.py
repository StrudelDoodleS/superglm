"""Safeguarded dense fixed-lambda solver for distributional likelihoods."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.solver.certificate as rank_certificate
import superglm.distributional.solver.chunks as chunking
from superglm._roundoff import gamma
from superglm.distributional.family import (
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    WallDisclosingFamily,
    _likelihood_reuse_contract,
    expected_information_provider,
)
from superglm.distributional.kernels._common import _NumericalEvaluationError
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.results.endpoint_evidence import (
    CHUNKED_EXECUTION_BACKEND_IDENTIFIER,
    DENSE_EXECUTION_BACKEND_IDENTIFIER,
    CoefficientCurvature,
    ConvergenceReason,
    ExecutionBackendIdentifier,
)
from superglm.distributional.results.solver import (
    DenseSolverConfig,
    DenseSolverResult,
    RetainedCoordinates,
    SolverIteration,
    _newton_decrement_is_certified,
    _resolution_limited_decrement_is_within_objective_ulp,
    _score_within_rounding_floor,
    _validate_resolution_limited_stationarity,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver._formation import CurvatureFormation
from superglm.distributional.solver._likelihood_cache import (
    _is_builtin_gaussian_gamma,
    _LikelihoodCache,
    build_likelihood_cache,
)
from superglm.distributional.solver.assembly import (
    DenseJointGeometry,
    _assemble_dense_geometry_from_matrices,
    _evaluate_predictors_from_matrices,
    dense_predictor_matrices,
    validated_dense_penalty,
)
from superglm.distributional.solver.curvature import (
    CurvatureDecision,
    CurvaturePolicyState,
    resolve_curvature,
)
from superglm.distributional.solver.derivatives import (
    PredictorLikelihoodEvaluation,
    nearest_semidefinite_curvature,
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.distributional.walls import PRESSED_STEP, outward_step, wall_rows
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import IdentityLink, Link, LogLink
from superglm.reml._compensated import _dot2_quadratic_form, _native_operand
from superglm.reml._compensated import _nonzero_entries as _penalty_entries
from superglm.solvers.rank import (
    RankDecomposition,
    decompose_gram,
    try_decompose_verified_spd_gram,
)


class DenseSolverError(RuntimeError):
    """Raised when no finite safeguarded coefficient direction exists."""


_StopPolicy = Literal["ordinary", "score_only"]


def _readonly(values: NDArray) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _SolverContext:
    family: DistributionalFamily
    fisher_family: ExpectedInformationFamily | None
    layout: StackedLayout
    response: NDArray[np.float64]
    likelihood_plan: FamilyLikelihoodPlan
    penalty: NDArray[np.float64]
    links: tuple[Link, ...]
    coefficient_curvature: CoefficientCurvature
    chunk_size: int | None
    execution_backend_identifier: ExecutionBackendIdentifier
    dense_matrices: tuple[NDArray[np.float64], ...] | None
    coefficient_face: PenaltyFace | None
    likelihood_cache: _LikelihoodCache | None = None
    # Set only for the reduced fit after a rank-deficient terminal decision:
    # Newton steps then move the representative's retained coefficients alone.
    retained: RetainedCoordinates | None = None
    # The penalty's nonzero entries in row-major order. Derived, never passed:
    # __post_init__ rebuilds them with every context, including one that
    # dataclasses.replace derives with a new penalty.
    penalty_entries: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "penalty_entries", _penalty_entries(self.penalty))


@dataclass(frozen=True)
class _DenseObservedReuseOwner:
    family: DistributionalFamily
    layout: StackedLayout
    response: object
    likelihood_plan: FamilyLikelihoodPlan
    config: DenseSolverConfig
    chunk_size: chunking.ChunkSize | None
    coefficient_face: PenaltyFace | None
    stop_policy: _StopPolicy

    def matches(self, other: _DenseObservedReuseOwner) -> bool:
        return bool(
            self.family is other.family
            and self.layout is other.layout
            and self.response is other.response
            and self.likelihood_plan is other.likelihood_plan
            and self.config == other.config
            and self.chunk_size == other.chunk_size
            and self.coefficient_face is other.coefficient_face
            and self.stop_policy == other.stop_policy
        )


class _DenseObservedReuseSession:
    """Recognize certified observed results produced inside one fit session."""

    def __init__(self) -> None:
        self._results: dict[
            int,
            tuple[
                weakref.ReferenceType[DenseSolverResult],
                _DenseObservedReuseOwner,
                CurvatureFormation | None,
            ],
        ] = {}
        self._chunk_results: dict[int, _ChunkObservedReuseRecord] = {}
        self._dense: dict[int, tuple[StackedLayout, tuple[NDArray[np.float64], ...]]] = {}
        self._likelihood: (
            tuple[DistributionalFamily, FamilyLikelihoodPlan, _LikelihoodCache | None] | None
        ) = None

    def likelihood_cache(
        self, family: DistributionalFamily, plan: FamilyLikelihoodPlan
    ) -> _LikelihoodCache | None:
        """Share bounded static row preparation across coefficient fits."""
        entry = self._likelihood
        if entry is not None:
            if entry[0] is family and entry[1] is plan:
                return entry[2]
            if entry[2] is not None:
                entry[2].clear()
        cache = build_likelihood_cache(family, plan)
        self._likelihood = (family, plan, cache)
        return cache

    def dense_matrices(
        self,
        layout: StackedLayout,
        *,
        phase_recorder: FitPhaseRecorder | None = None,
    ) -> tuple[NDArray[np.float64], ...]:
        """Dense predictor matrices for ``layout``, built once per layout identity.

        Only an actual build is a ``dense_predictor_matrices`` phase
        observation; a memo hit is free and unrecorded.
        """
        entry = self._dense.get(id(layout))
        if entry is not None and entry[0] is layout:
            return entry[1]
        with measure_phase(phase_recorder, "dense_predictor_matrices"):
            matrices = dense_predictor_matrices(layout)
        self._dense[id(layout)] = (layout, matrices)
        return matrices

    def remembers(
        self,
        result: DenseSolverResult,
        owner: _DenseObservedReuseOwner,
    ) -> bool:
        entry = self._results.get(id(result))
        return bool(entry is not None and entry[0]() is result and entry[1].matches(owner))

    def remember(
        self,
        result: DenseSolverResult,
        owner: _DenseObservedReuseOwner,
        *,
        context: _SolverContext | None = None,
        score_data: NDArray[np.float64] | None = None,
        formation: CurvatureFormation | None = None,
    ) -> None:
        curvature = result.terminal_curvature
        if (
            owner.chunk_size is None
            and owner.coefficient_face is None
            and owner.stop_policy == "ordinary"
            and owner.config.coefficient_curvature == "observed"
            and result.config == owner.config
            and result.resolved_chunk_size is None
            and result.execution_backend_identifier == DENSE_EXECUTION_BACKEND_IDENTIFIER
            and result.coefficient_face is None
            and result.converged
            and curvature.requested_source == "observed"
            and curvature.actual_source == "observed"
            and curvature.fallback_count == 0
        ):
            key = id(result)
            self._results[key] = (
                self._source_reference(result, chunked=False),
                owner,
                formation,
            )
        elif (
            context is not None
            and score_data is not None
            and context.chunk_size is not None
            and owner.chunk_size is not None
            and owner.coefficient_face is None
            and owner.stop_policy == "ordinary"
            and owner.config.coefficient_curvature == "observed"
            and result.config == owner.config
            and result.resolved_chunk_size == context.chunk_size
            and result.execution_backend_identifier == CHUNKED_EXECUTION_BACKEND_IDENTIFIER
            and result.coefficient_face is None
            and result.converged
            and curvature.requested_source == "observed"
            and curvature.actual_source == "observed"
            and curvature.fallback_count == 0
        ):
            self._chunk_results[id(result)] = _ChunkObservedReuseRecord(
                source=self._source_reference(result, chunked=True),
                owner=owner,
                coefficients=_readonly(result.coefficients),
                score_data=_readonly(score_data),
                data_curvature=_readonly(result.terminal_data_curvature),
                formation=formation,
            )

    def _source_reference(
        self, result: DenseSolverResult, *, chunked: bool
    ) -> weakref.ReferenceType[DenseSolverResult]:
        """Evict a collected source without making its callback own this session."""
        session_reference = weakref.ref(self)
        key = id(result)

        def discard(_reference: weakref.ReferenceType[DenseSolverResult]) -> None:
            session = session_reference()
            if session is not None:
                entries = session._chunk_results if chunked else session._results
                entries.pop(key, None)

        return weakref.ref(result, discard)


@dataclass(frozen=True)
class _ChunkObservedReuseRecord:
    """Fit-owned aggregates; the source's row predictions are weakly referenced."""

    source: weakref.ReferenceType[DenseSolverResult]
    owner: _DenseObservedReuseOwner
    coefficients: NDArray[np.float64]
    score_data: NDArray[np.float64]
    data_curvature: NDArray[np.float64]
    # The source assembly's formation statistics: the data curvature is reused.
    formation: CurvatureFormation | None = None


@dataclass(frozen=True)
class _AcceptedState:
    coefficients: NDArray[np.float64]
    eta: NDArray[np.float64] | None
    theta: NDArray[np.float64] | None
    derivatives: PredictorLikelihoodEvaluation | None
    fisher_curvature_packed: NDArray[np.float64] | None
    optimizing_log_likelihood: float
    parameter_independent_carrier: float
    log_likelihood: float
    penalized_optimizing_log_likelihood: float
    penalized_log_likelihood: float


@dataclass(frozen=True)
class _Direction:
    step: NDArray[np.float64]
    decomposition: RankDecomposition
    levenberg_shift: float
    residual: float


@dataclass(frozen=True)
class _OptimizationRun:
    state: _AcceptedState
    geometry: DenseJointGeometry
    history: tuple[SolverIteration, ...]
    converged: bool
    reason: ConvergenceReason
    score_relative: float
    objective_relative_change: float
    step_relative: float
    # Coefficient-only evidence for this exact accepted state and fit context;
    # it never replaces the Fisher geometry used for coefficient directions.
    observed_geometry: DenseJointGeometry | None = None


def _validated_context(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    penalty: NDArray,
    *,
    coefficient_curvature: CoefficientCurvature,
    chunk_size: chunking.ChunkSize | None,
    coefficient_face: PenaltyFace | None,
    dense_matrices: tuple[NDArray[np.float64], ...] | None = None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> _SolverContext:
    if not isinstance(family, DistributionalFamily):
        raise TypeError("family must implement DistributionalFamily")
    fisher_family = expected_information_provider(family, likelihood_plan)
    if coefficient_curvature == "fisher" and fisher_family is None:
        if isinstance(family, ExpectedInformationFamily):
            raise ValueError(
                "coefficient_curvature='fisher' uses complete-row expected information, which "
                "truncated or censored rows do not have; use coefficient_curvature='observed'"
            )
        raise ValueError("Fisher coefficient curvature requires expected information capability")
    if not isinstance(layout, StackedLayout) or layout.n_coefficients < 1:
        raise ValueError("layout must contain at least one global coefficient")
    if coefficient_face is not None:
        if not isinstance(coefficient_face, PenaltyFace):
            raise TypeError("coefficient_face must be a PenaltyFace")
        coefficient_face.validate_layout(layout)
    expected_names = tuple(parameter.name for parameter in family.parameters)
    actual_names = tuple(state.name for state in layout.predictors)
    if actual_names != expected_names:
        raise ValueError(
            f"layout predictor order {actual_names} does not match family order {expected_names}"
        )
    try:
        response = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("response must be a finite vector") from exc
    n_observations = layout.predictors[0].design.n
    if response.shape != (n_observations,) or not np.all(np.isfinite(response)):
        raise UnsupportedLikelihoodContractError(
            "response and likelihood plan must match the layout rows"
        )
    root_likelihood_plan, response = chunking._validate_bound_likelihood(
        family,
        likelihood_plan,
        response,
    )
    penalty_matrix = validated_dense_penalty(penalty, layout.n_coefficients)
    try:
        decompose_gram(penalty_matrix)
    except ValueError as exc:
        raise ValueError(
            "penalty must be positive semidefinite under the shared rank policy"
        ) from exc
    links = tuple(state.link for state in layout.predictors)
    resolved_chunk_size = chunking._resolve_fitting_chunk_size(
        family, layout, root_likelihood_plan, chunk_size
    )
    likelihood_cache = None
    if resolved_chunk_size is not None:
        dense_matrices = None
        likelihood_cache = (
            build_likelihood_cache(family, root_likelihood_plan)
            if _reuse_session is None
            else _reuse_session.likelihood_cache(family, root_likelihood_plan)
        )
    elif dense_matrices is None:
        dense_matrices = dense_predictor_matrices(layout)
    elif len(dense_matrices) != len(layout.predictors) or any(
        matrix.shape[0] != n_observations for matrix in dense_matrices
    ):
        raise ValueError("memoised dense matrices do not match the layout")
    penalty_matrix = _readonly(penalty_matrix)
    return _SolverContext(
        family=family,
        fisher_family=fisher_family,
        layout=layout,
        response=response,
        likelihood_plan=root_likelihood_plan,
        penalty=penalty_matrix,
        links=links,
        coefficient_curvature=coefficient_curvature,
        chunk_size=resolved_chunk_size,
        execution_backend_identifier=(
            DENSE_EXECUTION_BACKEND_IDENTIFIER
            if resolved_chunk_size is None
            else CHUNKED_EXECUTION_BACKEND_IDENTIFIER
        ),
        dense_matrices=dense_matrices,
        coefficient_face=coefficient_face,
        likelihood_cache=likelihood_cache,
    )


def _theta_from_eta(context: _SolverContext, eta: NDArray) -> NDArray[np.float64]:
    theta = np.empty_like(eta, dtype=np.float64)
    for parameter_index, link in enumerate(context.links):
        values = np.asarray(link.inverse(eta[:, parameter_index]), dtype=np.float64)
        if values.shape != (len(eta),) or not np.all(np.isfinite(values)):
            raise ValueError(f"inverse link {parameter_index} produced an invalid parameter")
        theta[:, parameter_index] = values
    return _readonly(theta)


def _evaluate_state_unmeasured(
    context: _SolverContext,
    coefficients: NDArray,
    *,
    derivative_order: Literal[0, 2] = 2,
    prepared: _AcceptedState | None = None,
) -> _AcceptedState | None:
    """Build a new trial without mutating any previously accepted state."""
    try:
        coefficient_values = np.asarray(coefficients, dtype=np.float64)
        if coefficient_values.shape != (context.layout.n_coefficients,) or not np.all(
            np.isfinite(coefficient_values)
        ):
            return None
        coefficient_values = _readonly(coefficient_values)
        if context.chunk_size is not None:
            likelihood = chunking.evaluate_chunked_log_likelihood(
                context.family,
                context.layout,
                context.response,
                context.likelihood_plan,
                coefficient_values,
                chunk_size=context.chunk_size,
                likelihood_cache=context.likelihood_cache,
            )
            penalty_value = _half_penalty_quadratic(
                context.penalty, coefficient_values, context.penalty_entries
            )
            penalized_optimizing = likelihood.optimizing_log_likelihood - penalty_value
            penalized_reported = likelihood.log_likelihood - penalty_value
            if not np.isfinite(penalized_optimizing) or not np.isfinite(penalized_reported):
                return None
            return _AcceptedState(
                coefficients=coefficient_values,
                eta=None,
                theta=None,
                derivatives=None,
                fisher_curvature_packed=None,
                optimizing_log_likelihood=likelihood.optimizing_log_likelihood,
                parameter_independent_carrier=likelihood.parameter_independent_carrier,
                log_likelihood=likelihood.log_likelihood,
                penalized_optimizing_log_likelihood=penalized_optimizing,
                penalized_log_likelihood=penalized_reported,
            )

        if context.dense_matrices is None:
            raise RuntimeError("dense solver context is missing predictor matrices")
        if prepared is None:
            eta = _evaluate_predictors_from_matrices(
                context.layout,
                coefficient_values,
                context.dense_matrices,
            )
            theta = _theta_from_eta(context, eta)
        else:
            # A passing value screen already evaluated these exact coefficients.
            eta, theta = prepared.eta, prepared.theta
            if eta is None or theta is None:
                raise RuntimeError("dense trial is missing predictor values")
        natural = context.family.evaluate_natural(
            context.response,
            theta,
            context.likelihood_plan,
            derivative_order=derivative_order,
        )
        if natural.derivative_order != derivative_order:
            raise UnsupportedLikelihoodContractError(
                f"family must return exact derivative order {derivative_order}"
            )
        if natural.valid is not None and not np.all(natural.valid):
            return None
        derivatives = (
            transform_natural_derivatives(natural, eta, context.links)
            if derivative_order == 2
            else None
        )
        fisher = None
        if derivative_order == 2 and context.coefficient_curvature == "fisher":
            if context.fisher_family is None:
                raise RuntimeError("validated Fisher context is missing expected information")
            information_natural = context.fisher_family.expected_information_natural(
                theta,
                context.likelihood_plan,
            )
            fisher = transform_natural_information(information_natural, eta, context.links)
        optimizing_log_likelihood = float(
            np.sum(natural.optimizing_log_likelihood, dtype=np.float64)
        )
        carrier = float(np.sum(natural.parameter_independent_carrier, dtype=np.float64))
        log_likelihood = float(optimizing_log_likelihood + carrier)
        penalty_value = _half_penalty_quadratic(
            context.penalty, coefficient_values, context.penalty_entries
        )
        penalized_optimizing = optimizing_log_likelihood - penalty_value
        penalized_reported = log_likelihood - penalty_value
        if not np.isfinite(penalized_optimizing) or not np.isfinite(penalized_reported):
            return None
        return _AcceptedState(
            coefficients=coefficient_values,
            eta=eta,
            theta=theta,
            derivatives=derivatives,
            fisher_curvature_packed=fisher,
            optimizing_log_likelihood=optimizing_log_likelihood,
            parameter_independent_carrier=carrier,
            log_likelihood=log_likelihood,
            penalized_optimizing_log_likelihood=penalized_optimizing,
            penalized_log_likelihood=penalized_reported,
        )
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return None


def _evaluate_state(
    context: _SolverContext,
    coefficients: NDArray,
    *,
    phase_recorder: FitPhaseRecorder | None = None,
    derivative_order: Literal[0, 2] = 2,
    prepared: _AcceptedState | None = None,
) -> _AcceptedState | None:
    with measure_phase(phase_recorder, "likelihood_evaluation"):
        return _evaluate_state_unmeasured(
            context, coefficients, derivative_order=derivative_order, prepared=prepared
        )


def _fused_first_trial_eligible(context: _SolverContext) -> bool:
    contract = _likelihood_reuse_contract(context.family)
    return bool(
        context.chunk_size is not None
        and _is_builtin_gaussian_gamma(context.family)
        and contract is not None
        and contract.deterministic_chunk_replay
        and type(context.likelihood_plan) is contract.plan_type
        and type(context.likelihood_plan.weights) is ResolvedLikelihoodWeights
        and len(context.links) == len(context.layout.predictors)
        and all(state.link is link for state, link in zip(context.layout.predictors, context.links))
        and all(
            type(link) in (IdentityLink, LogLink, *contract.link_types) for link in context.links
        )
    )


def _evaluate_fused_trial(
    context: _SolverContext,
    coefficients: NDArray,
    source: CoefficientCurvature,
    phase_recorder: FitPhaseRecorder | None,
) -> tuple[_AcceptedState, DenseJointGeometry] | None:
    """Own one immediate trial; numerical derivative refusal retries its value."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (context.layout.n_coefficients,) or not np.all(np.isfinite(values)):
        return None
    values = _readonly(values)
    assert context.chunk_size is not None
    try:
        with measure_phase(phase_recorder, "curvature_gradient_assembly"):
            geometry, likelihood = chunking._evaluate_chunked_geometry(
                context.family,
                context.layout,
                context.response,
                context.likelihood_plan,
                values,
                penalty=context.penalty,
                chunk_size=context.chunk_size,
                curvature_source=source,
                likelihood_cache=context.likelihood_cache,
            )
    except chunking._TrialDerivativeError:
        return None
    penalty_value = _half_penalty_quadratic(context.penalty, values, context.penalty_entries)
    optimizing = likelihood.optimizing_log_likelihood - penalty_value
    reported = likelihood.log_likelihood - penalty_value
    if not np.isfinite(optimizing) or not np.isfinite(reported):
        return None
    state = _AcceptedState(
        coefficients=values,
        eta=None,
        theta=None,
        derivatives=None,
        fisher_curvature_packed=None,
        optimizing_log_likelihood=likelihood.optimizing_log_likelihood,
        parameter_independent_carrier=likelihood.parameter_independent_carrier,
        log_likelihood=likelihood.log_likelihood,
        penalized_optimizing_log_likelihood=optimizing,
        penalized_log_likelihood=reported,
    )
    return state, geometry


def _initial_coefficients(context: _SolverContext) -> NDArray[np.float64]:
    """Starting coefficients: the family's starting parameters, projected with the penalty.

    Each predictor's link-scale start, less its offset, is fitted by penalized
    weighted least squares with the starting penalty, as the first working
    regression of penalized IRLS and GAMLSS's RS algorithm (Stasinopoulos and
    Rigby 2007, §1.5). The penalty stops a sparse-tailed spline from absorbing
    a varying offset with huge coefficients.
    """
    initialized = context.family.initialize(context.response, context.likelihood_plan)
    initialized.validate_shape(
        n_observations=len(context.response),
        k_parameters=len(context.layout.predictors),
    )
    coefficients = np.zeros(context.layout.n_coefficients, dtype=np.float64)
    weights = context.likelihood_plan.weights.values
    if context.chunk_size is not None:
        for parameter_index, (state, link) in enumerate(
            zip(context.layout.predictors, context.links, strict=True)
        ):
            plan = PredictorExecutionPlan(
                state.design,
                state.intercept_index is not None,
            )
            if plan.width == 0:
                continue
            target = np.asarray(
                link.link(initialized.theta[:, parameter_index]),
                dtype=np.float64,
            )
            target = target - state.offset
            block = context.penalty[state.coefficient_slice, state.coefficient_slice]
            gram = plan.diagonal_moment(weights) + block
            rhs = plan.score(weights * target)
            coefficients[state.coefficient_slice] = decompose_gram(gram).solve(rhs)
        return _readonly(coefficients)

    if context.dense_matrices is None:
        raise RuntimeError("dense solver context is missing predictor matrices")
    matrices = context.dense_matrices
    square_root_weight = np.sqrt(weights)
    for parameter_index, (state, matrix, link) in enumerate(
        zip(context.layout.predictors, matrices, context.links, strict=True)
    ):
        if matrix.shape[1] == 0:
            continue
        target = np.asarray(link.link(initialized.theta[:, parameter_index]), dtype=np.float64)
        target = target - state.offset
        weighted_matrix = matrix * square_root_weight[:, None]
        block = context.penalty[state.coefficient_slice, state.coefficient_slice]
        gram = weighted_matrix.T @ weighted_matrix + block
        rhs = weighted_matrix.T @ (target * square_root_weight)
        coefficients[state.coefficient_slice] = decompose_gram(gram).solve(rhs)
    return _readonly(coefficients)


def _geometry(
    context: _SolverContext,
    state: _AcceptedState,
    source: chunking.CurvatureSource,
) -> DenseJointGeometry:
    if context.chunk_size is not None:
        return chunking.assemble_chunked_geometry(
            context.family,
            context.layout,
            context.response,
            context.likelihood_plan,
            state.coefficients,
            penalty=context.penalty,
            chunk_size=context.chunk_size,
            curvature_source=source,
            likelihood_cache=context.likelihood_cache,
        )
    if state.derivatives is None:
        raise RuntimeError("dense accepted state is missing derivative geometry")
    curvature = _dense_curvature_channels(context, state, source)
    if context.dense_matrices is None:
        raise RuntimeError("dense solver context is missing predictor matrices")
    return _assemble_dense_geometry_from_matrices(
        context.layout,
        context.dense_matrices,
        state.derivatives.score_eta,
        curvature,
        penalty=context.penalty,
        coefficients=state.coefficients,
    )


def _dense_curvature_channels(
    context: _SolverContext,
    state: _AcceptedState,
    source: chunking.CurvatureSource,
) -> NDArray[np.float64]:
    """The dense state's packed curvature channels of ``source``."""
    if state.derivatives is None:
        raise RuntimeError("dense accepted state is missing derivative geometry")
    if source == "nearest_semidefinite":
        return nearest_semidefinite_curvature(state.derivatives.curvature_packed)
    if source == "fisher":
        curvature = state.fisher_curvature_packed
        if curvature is None:
            if context.fisher_family is None or state.theta is None or state.eta is None:
                raise RuntimeError("dense Fisher geometry requires expected information")
            information_natural = context.fisher_family.expected_information_natural(
                state.theta,
                context.likelihood_plan,
            )
            curvature = transform_natural_information(
                information_natural,
                state.eta,
                context.links,
            )
        return curvature
    return state.derivatives.curvature_packed


def _measured_geometry(
    context: _SolverContext,
    state: _AcceptedState,
    source: chunking.CurvatureSource,
    phase_recorder: FitPhaseRecorder | None,
) -> DenseJointGeometry:
    with measure_phase(phase_recorder, "curvature_gradient_assembly"):
        return _geometry(context, state, source)


def _half_penalty_quadratic(
    penalty: NDArray,
    coefficients: NDArray,
    entries: tuple[NDArray, NDArray, NDArray] | None = None,
) -> float:
    """Return ``0.5 * b' P b`` without the naive form's cancellation noise.

    ``entries`` are ``P``'s nonzero entries from ``_penalty_entries``. The
    naive form errs by ``gamma_k |b|' |P| |b|`` (``k`` the most nonzeros in a
    row plus the nonzero row count), which at a capped smoothing parameter can
    exceed a Newton step's objective change and make line-search acceptance
    platform-dependent. Dot2 rows and outer product (Ogita, Rump and Oishi
    2005) bound it by ``u |b' P b| + u |b|' |P b|`` plus a ``gamma_n**2`` term;
    operands outside the normal range fall back to the naive form.

    The score keeps plain ``P @ b``, whose row error ``gamma_k_i (|P| |b|)_i``
    matches one ulp of ``b``; so at a cap stationarity is certified through
    the Newton decrement (``_score_within_rounding_floor``).
    """
    values = _native_operand(coefficients)
    matrix = np.asarray(penalty, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        # The kernel indexes the coefficients without bounds checks, so a pair
        # that does not conform takes the plain form, which raises numpy's own
        # ValueError, exactly as the penalty value did before the kernel.
        return 0.5 * float(values @ matrix @ values)
    if entries is None:
        entries = _penalty_entries(matrix)
    quadratic, valid = _dot2_quadratic_form(*entries, values)
    if not valid:
        quadratic = values @ matrix @ values
    return 0.5 * float(quadratic)


def _relative_score(score: NDArray, objective: float) -> float:
    return float(np.linalg.norm(score, ord=np.inf) / (1.0 + abs(objective)))


def _subspace(context: _SolverContext) -> PenaltyFace | RetainedCoordinates | None:
    """The coordinates Newton steps move: a penalty face, or a representative's retained ones."""
    return context.coefficient_face if context.retained is None else context.retained


def _optimization_score(context: _SolverContext, score: NDArray) -> NDArray[np.float64]:
    subspace = _subspace(context)
    if subspace is None:
        return np.asarray(score, dtype=np.float64)
    return subspace.reduce_vector(score)


def _optimization_score_relative(
    context: _SolverContext,
    score: NDArray,
    objective: float,
) -> float:
    reduced = _optimization_score(context, score)
    if reduced.size == 0:
        return 0.0
    return _relative_score(reduced, objective)


def _objective_and_step_is_certified(
    context: _SolverContext,
    config: DenseSolverConfig,
    geometry: DenseJointGeometry,
    state: _AcceptedState,
    direction: _Direction,
    score_relative: float,
    phase_recorder: FitPhaseRecorder | None,
) -> tuple[bool, DenseJointGeometry | None]:
    """Require retained score or a safe, scale-invariant stationarity proof."""
    if score_relative <= config.tolerance:
        return True, None
    optimization_score = _optimization_score(context, geometry.score_penalized)
    decomposition = direction.decomposition
    if (
        config.coefficient_curvature not in ("observed", "fisher")
        or direction.levenberg_shift != 0.0
        or direction.residual > config.residual_tolerance
        or decomposition.width != len(optimization_score)
        or decomposition.rank != len(optimization_score)
        or decomposition.rank_truncated
        or decomposition.used_svd_fallback
        or decomposition.resolution_limited
    ):
        return False, None
    certificate_geometry = geometry
    observed_geometry = None
    if config.coefficient_curvature == "fisher":
        certificate_geometry = _measured_geometry(
            context,
            state,
            "observed",
            phase_recorder,
        )
        observed_geometry = certificate_geometry
        if (
            _optimization_score_relative(
                context,
                certificate_geometry.score_penalized,
                state.penalized_optimizing_log_likelihood,
            )
            <= config.tolerance
        ):
            return True, observed_geometry
    return (
        _newton_decrement_is_certified(
            config=config,
            score=certificate_geometry.score_penalized,
            penalized_curvature=certificate_geometry.penalized_curvature,
            penalized_objective=state.penalized_optimizing_log_likelihood,
            face=_subspace(context),
            tolerance=config.tolerance,
        ),
        observed_geometry,
    )


def _policy_curvature(
    context: _SolverContext,
    matrix: NDArray,
) -> NDArray[np.float64]:
    if context.coefficient_face is None:
        return np.asarray(matrix, dtype=np.float64)
    return context.coefficient_face.reduce_matrix(matrix)


def _solve_residual(
    system: NDArray,
    step: NDArray,
    score: NDArray,
    decomposition: RankDecomposition,
    tolerance: float,
) -> float:
    """The solve's relative residual ``||H s - g|| / max(1, ||g||)``, or its backward error.

    A backward-stable solve leaves a residual of about ``u ||H|| ||s||``,
    which exceeds any fixed fraction of ``||g||`` once ``||H|| ||s|| >> ||g||``:
    a column far from zero beside an intercept puts the step along a
    direction the curvature barely resists. An untruncated solve that misses
    ``tolerance`` on the relative test is read by its normwise backward error
    ``||r|| / (||H|| ||s|| + ||g||)`` instead (Rigal and Gaches, *J. ACM* 14,
    1967; Higham 2002, Thm 7.1): the relative perturbation of ``H`` and ``g``
    it solves exactly, with ``||H||_2`` taken at its largest column norm, a
    lower bound. A truncated solve keeps the relative test, since its
    residual also holds the score outside the retained range.
    """
    residual = float(np.linalg.norm(system @ step - score))
    score_norm = float(np.linalg.norm(score))
    relative = residual / max(1.0, score_norm)
    if relative <= tolerance or decomposition.rank_truncated or not np.isfinite(relative):
        return relative
    curvature_norm = float(np.max(np.linalg.norm(system, axis=0), initial=0.0))
    scale = curvature_norm * float(np.linalg.norm(step)) + score_norm
    return residual / scale if scale > 0.0 else relative


def _solve_direction(
    matrix: NDArray,
    score: NDArray,
    config: DenseSolverConfig,
) -> _Direction:
    matrix = np.asarray(matrix, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(score)):
        raise DenseSolverError("coefficient system contains non-finite values")
    row_scale = np.max(np.abs(matrix), axis=1, initial=0.0)
    diagonal_scale = np.maximum(np.abs(np.diag(matrix)), np.finfo(np.float64).eps * row_scale)
    positive_scale = diagonal_scale[diagonal_scale > 0.0]
    if positive_scale.size == 0:
        raise DenseSolverError("coefficient curvature has no positive numerical scale")
    diagonal_scale = np.maximum(diagonal_scale, np.min(positive_scale) * np.finfo(float).eps)

    if config.coefficient_curvature == "fisher":
        decomposition = try_decompose_verified_spd_gram(
            matrix,
            residual_tol=config.residual_tolerance,
        )
        if decomposition is not None:
            step = decomposition.solve(score)
            residual = _solve_residual(
                matrix, step, score, decomposition, config.residual_tolerance
            )
            directional_derivative = float(score @ step)
            ascent_floor = (
                np.finfo(float).eps * float(np.linalg.norm(score)) * float(np.linalg.norm(step))
            )
            if (
                np.all(np.isfinite(step))
                and residual <= config.residual_tolerance
                and directional_derivative > ascent_floor
            ):
                return _Direction(
                    step=_readonly(step),
                    decomposition=decomposition,
                    levenberg_shift=0.0,
                    residual=residual,
                )

    shifts = [0.0]
    shifts.extend(
        config.initial_levenberg_shift * config.levenberg_growth**attempt
        for attempt in range(config.max_levenberg_attempts)
    )
    for shift in shifts:
        system = matrix if shift == 0.0 else matrix + shift * np.diag(diagonal_scale)
        try:
            decomposition = decompose_gram(system, residual_tol=config.residual_tolerance)
            step = decomposition.solve(score)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(step)):
            continue
        residual = _solve_residual(system, step, score, decomposition, config.residual_tolerance)
        directional_derivative = float(score @ step)
        ascent_floor = (
            np.finfo(float).eps * float(np.linalg.norm(score)) * float(np.linalg.norm(step))
        )
        if residual <= config.residual_tolerance and directional_derivative > ascent_floor:
            return _Direction(
                step=_readonly(step),
                decomposition=decomposition,
                levenberg_shift=float(shift),
                residual=residual,
            )
    raise DenseSolverError("unable to construct a certified ascent direction")


def _solve_coefficient_direction(
    context: _SolverContext,
    matrix: NDArray,
    score: NDArray,
    config: DenseSolverConfig,
) -> _Direction:
    face = _subspace(context)
    if face is None:
        return _solve_direction(matrix, score, config)
    reduced = _solve_direction(
        face.reduce_matrix(matrix),
        face.reduce_vector(score),
        config,
    )
    return replace(reduced, step=face.lift_vector(reduced.step))


def _cap_predictor_step(
    context: _SolverContext,
    state: _AcceptedState,
    step: NDArray,
    maximum: float,
) -> tuple[NDArray[np.float64], float]:
    if context.chunk_size is not None:
        maximum_change = chunking.maximum_chunked_predictor_change(
            context.layout,
            step,
            chunk_size=context.chunk_size,
        )
    else:
        if state.eta is None:
            raise RuntimeError("dense accepted state is missing predictor values")
        proposed = state.coefficients + step
        if context.dense_matrices is None:
            raise RuntimeError("dense solver context is missing predictor matrices")
        proposed_eta = _evaluate_predictors_from_matrices(
            context.layout,
            proposed,
            context.dense_matrices,
        )
        maximum_change = float(np.max(np.abs(proposed_eta - state.eta), initial=0.0))
    scale = 1.0 if maximum_change <= maximum else maximum / maximum_change
    return _readonly(scale * step), float(scale)


def _wall_step_unsettled(
    context: _SolverContext,
    state: _AcceptedState,
    geometry: DenseJointGeometry,
    config: DenseSolverConfig,
) -> bool:
    """Whether the next Newton step still carries a row near a wall towards it.

    A pressed fit keeps such a step however far it runs, while an interior fit
    stopped at a loose tolerance can show one too and loses it within a few more
    iterations; the wall disclosure needs the two apart.
    """
    if not isinstance(context.family, WallDisclosingFamily):
        return False
    rows = wall_rows(context.family, context.layout, state.coefficients, state.theta)
    if rows is None:
        return False
    try:
        direction = _solve_coefficient_direction(
            context, geometry.penalized_curvature, geometry.score_penalized, config
        )
    except DenseSolverError:
        return False
    return bool(np.any(rows.near & (outward_step(rows, direction.step) >= PRESSED_STEP)))


def _run_iterations(
    context: _SolverContext,
    initial_state: _AcceptedState,
    config: DenseSolverConfig,
    *,
    initial_geometry: DenseJointGeometry | None = None,
    stop_policy: _StopPolicy,
    start_iteration: int = 0,
    phase_recorder: FitPhaseRecorder | None = None,
) -> _OptimizationRun:
    state = initial_state
    observed_geometry: DenseJointGeometry | None = None
    history: list[SolverIteration] = []
    objective_relative_change = 0.0
    step_relative = 0.0
    geometry = (
        _measured_geometry(
            context,
            state,
            config.coefficient_curvature,
            phase_recorder,
        )
        if initial_geometry is None
        else initial_geometry
    )

    def finish(
        converged: bool,
        reason: ConvergenceReason,
        score_relative: float,
        *,
        no_step: bool = False,
    ) -> _OptimizationRun:
        # Reads the loop's current state; ``no_step`` records an exit that made no
        # movement, instead of the last accepted step's relative changes.
        return _OptimizationRun(
            state=state,
            geometry=geometry,
            observed_geometry=observed_geometry,
            history=tuple(history),
            converged=converged,
            reason=reason,
            score_relative=score_relative,
            objective_relative_change=0.0 if no_step else objective_relative_change,
            step_relative=0.0 if no_step else step_relative,
        )

    for local_iteration in range(config.max_iterations):
        score_relative = _optimization_score_relative(
            context,
            geometry.score_penalized,
            state.penalized_optimizing_log_likelihood,
        )
        # A score within its own rounding floor, whose Newton correction is
        # certified not to change the objective, cannot be resolved further:
        # another step only follows round-off.
        if score_relative <= config.tolerance or (
            config.coefficient_curvature == "observed"
            and _score_within_rounding_floor(
                config=config,
                penalty=context.penalty,
                coefficients=state.coefficients,
                score=geometry.score_penalized,
                penalized_curvature=geometry.penalized_curvature,
                face=_subspace(context),
                objective=state.penalized_optimizing_log_likelihood,
                tolerance=config.tolerance,
            )
        ):
            return finish(True, "score", score_relative)

        try:
            with measure_phase(phase_recorder, "coefficient_decomposition_solve"):
                direction = _solve_coefficient_direction(
                    context,
                    geometry.penalized_curvature,
                    geometry.score_penalized,
                    config,
                )
        except DenseSolverError:
            # No Levenberg damping of the curvature gives a certified ascent
            # direction. The accepted state is the best one reached: stop there,
            # unconverged, and let the fit disclose why instead of raising.
            return finish(False, "no_ascent_direction", score_relative)
        optimization_score = _optimization_score(context, geometry.score_penalized)
        with measure_phase(phase_recorder, "likelihood_evaluation"):
            capped_step, step_scale = _cap_predictor_step(
                context,
                state,
                direction.step,
                config.max_predictor_step,
            )
        accepted: _AcceptedState | None = None
        accepted_geometry = None
        alpha = 1.0
        backtracks = 0
        distinct_finite_trial_evaluated = False
        distinct_trial_improved = False
        reached_identical_candidate = False
        for attempt in range(config.max_backtracks + 1):
            candidate_coefficients = state.coefficients + alpha * capped_step
            if context.coefficient_face is not None:
                candidate_coefficients = context.coefficient_face.project(candidate_coefficients)
            applied_step = candidate_coefficients - state.coefficients
            if np.array_equal(candidate_coefficients, state.coefficients):
                reached_identical_candidate = True
                break
            # Only the immediate first ordinary chunked trial owns speculative
            # geometry. Rejected backtracks and certification keep their screens.
            screen = stop_policy == "ordinary" and context.chunk_size is None and attempt > 0
            fused = (
                _evaluate_fused_trial(
                    context, candidate_coefficients, config.coefficient_curvature, phase_recorder
                )
                if stop_policy == "ordinary"
                and attempt == 0
                and _fused_first_trial_eligible(context)
                else None
            )
            candidate = (
                fused[0]
                if fused is not None
                else _evaluate_state(
                    context,
                    candidate_coefficients,
                    phase_recorder=phase_recorder,
                    derivative_order=0 if screen else 2,
                )
            )
            directional_derivative = float(geometry.score_penalized @ applied_step)
            required = (
                state.penalized_optimizing_log_likelihood
                + config.armijo_constant * directional_derivative
            )
            if screen:
                candidate = (
                    _evaluate_state(
                        context,
                        candidate_coefficients,
                        phase_recorder=phase_recorder,
                        prepared=candidate,
                    )
                    if candidate is not None
                    and candidate.penalized_optimizing_log_likelihood >= required
                    else None
                )
            if candidate is not None:
                distinct_finite_trial_evaluated = True
                if stop_policy == "score_only":
                    distinct_trial_improved = bool(
                        distinct_trial_improved
                        or candidate.penalized_optimizing_log_likelihood
                        > state.penalized_optimizing_log_likelihood
                    )
            if (
                candidate is not None
                and candidate.penalized_optimizing_log_likelihood >= required
                and (
                    stop_policy == "ordinary"
                    or candidate.penalized_optimizing_log_likelihood
                    > state.penalized_optimizing_log_likelihood
                )
            ):
                accepted_geometry = None if fused is None else fused[1]
                if accepted_geometry is None and context.chunk_size is not None:
                    # A chunked trial whose geometry the family refuses is
                    # backtracked from, as a dense trial is, not accepted and raised.
                    try:
                        accepted_geometry = _measured_geometry(
                            context,
                            candidate,
                            config.coefficient_curvature,
                            phase_recorder,
                        )
                    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                        accepted_geometry = None
                if accepted_geometry is not None or context.chunk_size is None:
                    accepted = candidate
                    fused = None
                    break
            fused = None
            if attempt == config.max_backtracks:
                break
            alpha *= config.backtrack_factor
            backtracks += 1
        if accepted is None:
            # Exhausting finite trials can be rounding-limited even when a
            # small coefficient never becomes bit-identical. The same retained
            # stationarity proof is required; line-search failure is no proof.
            if stop_policy == "ordinary" and (
                reached_identical_candidate or distinct_finite_trial_evaluated
            ):
                certified, measured_observed = _objective_and_step_is_certified(
                    context,
                    config,
                    geometry,
                    state,
                    direction,
                    score_relative,
                    phase_recorder,
                )
                if measured_observed is not None:
                    observed_geometry = measured_observed
                del measured_observed
                if certified:
                    return finish(True, "objective_and_step", score_relative, no_step=True)
            retained_correction: NDArray[np.float64] | None = None
            if direction.decomposition.rank == len(optimization_score):
                try:
                    retained_correction = np.asarray(
                        direction.decomposition.solve(optimization_score),
                        dtype=np.float64,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    retained_correction = None
            resolution_limited = bool(
                stop_policy == "score_only"
                and context.coefficient_face is not None
                and config.coefficient_curvature == "observed"
                and direction.levenberg_shift == 0.0
                and direction.decomposition.rank == len(optimization_score)
                and direction.residual <= config.residual_tolerance
                and step_scale == 1.0
                and retained_correction is not None
                and _resolution_limited_decrement_is_within_objective_ulp(
                    optimization_score,
                    retained_correction,
                    state.penalized_optimizing_log_likelihood,
                )
                and distinct_finite_trial_evaluated
                and not distinct_trial_improved
                and reached_identical_candidate
            )
            return finish(
                resolution_limited,
                ("resolution_limited_stationarity" if resolution_limited else "line_search_failed"),
                score_relative,
            )

        if accepted_geometry is None:
            accepted_geometry = _measured_geometry(
                context,
                accepted,
                config.coefficient_curvature,
                phase_recorder,
            )
        objective_relative_change = abs(
            accepted.penalized_optimizing_log_likelihood - state.penalized_optimizing_log_likelihood
        ) / (1.0 + abs(state.penalized_optimizing_log_likelihood))
        accepted_step = accepted.coefficients - state.coefficients
        step_relative = float(
            np.linalg.norm(accepted_step, ord=np.inf)
            / (1.0 + np.linalg.norm(accepted.coefficients, ord=np.inf))
        )
        accepted_score_relative = _optimization_score_relative(
            context,
            accepted_geometry.score_penalized,
            accepted.penalized_optimizing_log_likelihood,
        )
        condition = direction.decomposition.pre_truncation_condition
        history.append(
            SolverIteration(
                iteration=start_iteration + local_iteration + 1,
                objective_before=state.penalized_optimizing_log_likelihood,
                objective_after=accepted.penalized_optimizing_log_likelihood,
                objective_relative_change=objective_relative_change,
                step_relative=step_relative,
                score_relative=accepted_score_relative,
                step_scale=step_scale * alpha,
                backtracks=backtracks,
                levenberg_shift=direction.levenberg_shift,
                rank=direction.decomposition.rank,
                condition_estimate=(float(condition) if np.isfinite(condition) else None),
                solve_residual=direction.residual,
            )
        )
        state = accepted
        # A refused certificate may survive a terminal exit, but never a
        # coefficient transition. Context, penalty and precision are fixed
        # for this run; no evidence is shared across independent fits.
        observed_geometry = None
        geometry = accepted_geometry
        if stop_policy == "score_only":
            if accepted_score_relative <= config.tolerance:
                return finish(True, "score", accepted_score_relative)
            continue
        if objective_relative_change <= config.tolerance and step_relative <= config.tolerance:
            certified, measured_observed = _objective_and_step_is_certified(
                context,
                config,
                geometry,
                state,
                direction,
                accepted_score_relative,
                phase_recorder,
            )
            if measured_observed is not None:
                observed_geometry = measured_observed
            del measured_observed
            if certified:
                return finish(True, "objective_and_step", accepted_score_relative)
            # A distinct tiny step is still progress: keep the iteration budget;
            # only an identical candidate returns ``line_search_failed`` below.
        if (
            objective_relative_change <= config.tolerance
            and accepted_score_relative <= config.tolerance
        ):
            return finish(True, "objective_and_score", accepted_score_relative)

    return finish(
        False,
        "max_iterations",
        _optimization_score_relative(
            context,
            geometry.score_penalized,
            state.penalized_optimizing_log_likelihood,
        ),
    )


def _reuse_observed_initial_result(
    context: _SolverContext,
    coefficients: NDArray[np.float64],
    session: _DenseObservedReuseSession,
    source: DenseSolverResult,
    owner: _DenseObservedReuseOwner,
) -> tuple[_AcceptedState, DenseJointGeometry] | None:
    """Re-penalize a certified same-session endpoint without likelihood refresh."""
    if source.eta is None or source.theta is None:
        raise ValueError("observed endpoint reuse requires retained predictor and parameter rows")
    optimizing = source.optimizing_log_likelihood
    if (
        context.coefficient_face is not None
        or context.coefficient_curvature != "observed"
        or source.family_likelihood_plan_identifier != context.likelihood_plan.plan_identifier
        or optimizing is None
        or not np.array_equal(coefficients, source.coefficients)
    ):
        return None

    if context.chunk_size is None:
        if not session.remembers(source, owner) or context.dense_matrices is None:
            return None
        eta = _evaluate_predictors_from_matrices(
            context.layout,
            coefficients,
            context.dense_matrices,
        )
        if not np.array_equal(eta, source.eta):
            return None
        theta = _theta_from_eta(context, eta)
        if not np.array_equal(theta, source.theta):
            return None

        score_data = source.terminal_score + source.penalty @ coefficients
        data_curvature = np.asarray(source.terminal_data_curvature, dtype=np.float64)
        formation = session._results[id(source)][2]
    else:
        record = session._chunk_results.get(id(source))
        if (
            record is None
            or record.source() is not source
            or not record.owner.matches(owner)
            or source.resolved_chunk_size != context.chunk_size
            or source.execution_backend_identifier != CHUNKED_EXECUTION_BACKEND_IDENTIFIER
            or not np.array_equal(coefficients, record.coefficients)
        ):
            return None
        support_predictions = chunking._SupportPredictions()
        for rows in chunking.iter_row_chunks(len(context.response), context.chunk_size):
            eta = chunking._predictor_values(
                context.layout,
                coefficients,
                rows,
                include_offsets=True,
                support_predictions=support_predictions,
            )
            theta = chunking._theta_chunk(context.layout, eta)
            if not np.array_equal(eta, source.eta[rows.start : rows.stop]) or not np.array_equal(
                theta, source.theta[rows.start : rows.stop]
            ):
                return None
        score_data = record.score_data
        data_curvature = record.data_curvature
        formation = record.formation

    penalty_value = _half_penalty_quadratic(context.penalty, coefficients, context.penalty_entries)
    penalized_optimizing = float(optimizing - penalty_value)
    penalized_reported = float(source.log_likelihood - penalty_value)
    score_penalized = score_data - context.penalty @ coefficients
    penalized_curvature = data_curvature + context.penalty
    if not (
        np.isfinite(penalized_optimizing)
        and np.isfinite(penalized_reported)
        and np.all(np.isfinite(score_penalized))
        and np.all(np.isfinite(penalized_curvature))
    ):
        return None

    state = _AcceptedState(
        coefficients=source.coefficients,
        eta=source.eta if context.chunk_size is None else None,
        theta=source.theta if context.chunk_size is None else None,
        derivatives=None,
        fisher_curvature_packed=None,
        optimizing_log_likelihood=float(optimizing),
        parameter_independent_carrier=source.parameter_independent_carrier,
        log_likelihood=source.log_likelihood,
        penalized_optimizing_log_likelihood=penalized_optimizing,
        penalized_log_likelihood=penalized_reported,
    )
    geometry = DenseJointGeometry(
        score_data=_readonly(score_data),
        score_penalized=_readonly(score_penalized),
        data_curvature=_readonly(data_curvature),
        penalty=context.penalty,
        penalized_curvature=_readonly(penalized_curvature),
        formation=formation,
    )
    return state, geometry


def _geometry_from(
    context: _SolverContext,
    state: _AcceptedState,
    source: chunking.CurvatureSource,
    config: DenseSolverConfig,
    coefficient_geometry: DenseJointGeometry,
    held: DenseJointGeometry | None,
    phase_recorder: FitPhaseRecorder | None,
) -> DenseJointGeometry:
    """The geometry for ``source``: the run's own, else one already held, else measured."""

    if config.coefficient_curvature == source:
        return coefficient_geometry
    if held is not None:
        return held
    return _measured_geometry(context, state, source, phase_recorder)


def _certified_terminal_curvature(
    context: _SolverContext,
    state: _AcceptedState,
    curvature: CurvatureDecision,
    geometry: DenseJointGeometry,
    phase_recorder: FitPhaseRecorder | None,
) -> CurvatureDecision:
    """The accepted terminal decision, its rank certified against forming the matrix.

    ``certificate.certify_terminal_rank`` reads the rows of the accepted
    curvature source once at most, and only when a direction lies within the
    formation bound of the cut. Newton steps keep the formed decision. A row
    pass that cannot evaluate the state it accepted, or runs out of memory,
    keeps the formed decision, marked uncertified: the fit is disclosed, never
    refused.
    """
    formation = geometry.formation
    if formation is None:
        return curvature
    source = curvature.telemetry.actual_source
    if source not in ("observed", "fisher", "nearest_semidefinite"):
        return curvature
    if context.chunk_size is None:
        if context.dense_matrices is None:
            return curvature

        def channels() -> NDArray[np.float64]:
            evaluated = state
            if evaluated.derivatives is None:
                evaluated = _evaluate_state(
                    context, state.coefficients, phase_recorder=phase_recorder
                )
            if evaluated is None or evaluated.derivatives is None:
                raise FloatingPointError("terminal state has no curvature channels")
            return _dense_curvature_channels(context, evaluated, source)

        rows = rank_certificate.dense_rows(context.layout, context.dense_matrices, channels)
    else:
        rows = rank_certificate.chunked_rows(context, state.coefficients, source, formation)
    try:
        return rank_certificate.certify_terminal_rank(
            curvature,
            formation,
            layout=context.layout,
            penalized=geometry.penalized_curvature,
            penalty=geometry.penalty,
            face=context.coefficient_face,
            rows=rows,
            product_roundings=rank_certificate.product_roundings(context.layout, formation),
        )
    except (
        _NumericalEvaluationError,
        FloatingPointError,
        OverflowError,
        MemoryError,
        np.linalg.LinAlgError,
    ):
        return replace(curvature, telemetry=replace(curvature.telemetry, certified=False))


def _drops_design_aliases(layout: StackedLayout, rank: RankDecomposition) -> bool:
    """Whether every direction ``rank`` drops is a near-dependency of the design itself.

    ``X N ~ 0`` is an alias; a direction only the weights flatten is a likelihood boundary
    (Lesaffre and Marx 1993). ``rho = ||X N|| / ||C N||`` on unit columns (Belsley, Kuh and
    Welsch 1980) centred beside an intercept (Marquardt and Snee 1975), so no unit or origin
    makes a resolved column an alias. The span is one when ``rho <= cutoff ** (1/4)`` on all
    of it, the log-midpoint between a dropped direction's ``sqrt(cutoff)`` and 1, or when
    ``X N`` is rounding, ``2 k gamma_k`` of the uncentred norms (a constant column, 0 / 0
    centred). A centred square within its cancellation, ``3 gamma_{n+2} sum x^2`` (Chan,
    Golub and LeVeque 1983), counts as zero, which can only refuse an alias. Columns are
    scaled to unit maximum (the test is basis-free); a non-finite span is not an alias.
    """
    null = rank.null_basis()
    null = null / np.max(np.abs(null), axis=0)
    width = null.shape[1]
    design_gram = np.zeros((width, width))
    scale_gram = np.zeros((width, width))
    rounding_gram = np.zeros((width, width))
    for state in layout.predictors:
        local = null[state.coefficient_slice]
        if not np.any(local):
            continue
        design = state.design
        squared = np.concatenate(
            [np.diag(group.gram(np.ones(design.n))) for group in design.group_matrices] or [[]]
        )
        intercept = state.intercept_index is not None
        if intercept:
            squared = np.concatenate(([float(design.n)], squared))
        slopes = local[int(intercept) :]
        for column in range(width):
            projected = np.full(design.n, local[0, column] if intercept else 0.0)
            if design.p:
                projected += design.matvec(slopes[:, column])
            back = design.rmatvec(projected) if design.p else np.empty(0)
            if intercept:
                back = np.concatenate(([projected.sum()], back))
            design_gram[:, column] += local.T @ back
        rounding_gram += local.T @ (squared[:, None] * local)
        if intercept:
            means = design.rmatvec(np.ones(design.n)) / design.n if design.p else squared[1:]
            centred = squared[1:] - design.n * means**2
            centred[centred <= 3.0 * gamma(design.n + 2) * squared[1:]] = 0.0
            squared[1:] = centred
            local = np.vstack((local[:1] + means @ slopes, slopes))
        scale_gram += local.T @ (squared[:, None] * local)
    resolution = (2.0 * null.shape[0] * gamma(null.shape[0])) ** 2
    difference = (
        0.5 * (design_gram + design_gram.T)
        - np.sqrt(rank.cutoff) * scale_gram
        - resolution * rounding_gram
    )
    return bool(np.all(np.isfinite(difference)) and np.linalg.eigvalsh(difference)[-1] <= 0.0)


def _reduced_fit_is_stationary(
    run: _OptimizationRun, retained: RetainedCoordinates, config: DenseSolverConfig
) -> bool:
    """Whether a fit on the retained coefficients is also stationary for the full problem.

    The reduced fit is the optimum under ``beta_J = 0``, and the full problem's when that
    constraint's multiplier, the score ``g_J``, vanishes: the Kuhn-Tucker test an active-set
    method makes before it keeps a variable fixed (Lawson and Hanson 1974, ch. 23). On ``K``
    that is the reduced run's own stop; on ``J`` it is the score stop's test, which the
    rounding floor cannot widen: the floor needs a certified Newton decrement of the full,
    here rank-deficient, curvature. An alias's ``g_J`` vanishes with the retained score;
    ``x`` and ``x^2`` far from zero pass the design test, but the data resolve them, and
    ``g_J`` stays, in the dropped columns' units.
    """
    if not run.converged:
        return False
    score = np.asarray(run.geometry.score_penalized, dtype=np.float64)[retained.dropped]
    objective = run.state.penalized_optimizing_log_likelihood
    return _relative_score(score, objective) <= config.tolerance


def _representative_state(
    context: _SolverContext,
    state: _AcceptedState,
    rank: RankDecomposition,
    retained: RetainedCoordinates,
    phase_recorder: FitPhaseRecorder | None,
) -> _AcceptedState | None:
    """The terminal state projected onto its certified representative: the reduced fit's start.

    Newton steps decide their rank uncertified and can leave ``beta``
    anywhere along the null space the terminal decision drops. The
    representative is ``beta - N N_J^-1 beta_J``, ``N`` the null basis of
    the penalized curvature: ``beta_J = 0``, and an exact alias's predictors
    and penalty move by rounding alone, so it is already the reduced
    optimum. ``None`` when ``N_J`` is singular or the likelihood there is
    not finite.
    """
    dropped = retained.dropped
    null = rank.null_basis()
    try:
        along = np.linalg.solve(null[dropped], state.coefficients[dropped])
    except np.linalg.LinAlgError:
        return None
    coefficients = state.coefficients - null @ along
    coefficients[dropped] = 0.0
    return _evaluate_state(context, coefficients, phase_recorder=phase_recorder)


def _fit_dense_fixed_lambda_core(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    penalty: NDArray,
    *,
    initial: NDArray | None = None,
    config: DenseSolverConfig | None = None,
    chunk_size: chunking.ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
    coefficient_face: PenaltyFace | None = None,
    _reuse_session: _DenseObservedReuseSession | None = None,
    _reuse_source: DenseSolverResult | None = None,
    stop_policy: _StopPolicy,
) -> DenseSolverResult:
    """Fit one fixed-penalty likelihood, optionally in bounded row chunks."""
    solver_config = DenseSolverConfig() if config is None else config
    if not isinstance(solver_config, DenseSolverConfig):
        raise TypeError("config must be a DenseSolverConfig")
    if _reuse_session is not None and not isinstance(
        _reuse_session,
        _DenseObservedReuseSession,
    ):
        raise TypeError("_reuse_session must be a _DenseObservedReuseSession")
    if _reuse_source is not None and not isinstance(_reuse_source, DenseSolverResult):
        raise TypeError("_reuse_source must be a DenseSolverResult")
    reuse_owner = _DenseObservedReuseOwner(
        family=family,
        layout=layout,
        response=y,
        likelihood_plan=likelihood_plan,
        config=solver_config,
        chunk_size=chunk_size,
        coefficient_face=coefficient_face,
        stop_policy=stop_policy,
    )
    memoised: tuple[NDArray[np.float64], ...] | None = None
    if chunk_size is None:
        if _reuse_session is not None:
            memoised = _reuse_session.dense_matrices(layout, phase_recorder=phase_recorder)
        else:
            with measure_phase(phase_recorder, "dense_predictor_matrices"):
                memoised = dense_predictor_matrices(layout)
    context = _validated_context(
        family,
        layout,
        y,
        likelihood_plan,
        penalty,
        coefficient_curvature=solver_config.coefficient_curvature,
        chunk_size=chunk_size,
        coefficient_face=coefficient_face,
        dense_matrices=memoised,
        _reuse_session=_reuse_session,
    )
    with measure_phase(phase_recorder, "initialization"):
        if initial is None:
            initial_coefficients = _initial_coefficients(context)
        else:
            try:
                initial_coefficients = np.asarray(initial, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "initial must be a finite vector with the global layout shape"
                ) from exc
            if initial_coefficients.shape != (layout.n_coefficients,) or not np.all(
                np.isfinite(initial_coefficients)
            ):
                raise ValueError("initial must be a finite vector with the global layout shape")
            initial_coefficients = _readonly(initial_coefficients)
        if context.coefficient_face is not None:
            initial_coefficients = context.coefficient_face.project(initial_coefficients)
        reused = (
            None
            if _reuse_session is None or _reuse_source is None
            else _reuse_observed_initial_result(
                context,
                initial_coefficients,
                _reuse_session,
                _reuse_source,
                reuse_owner,
            )
        )
        if reused is None:
            initial_state = _evaluate_state(
                context,
                initial_coefficients,
                phase_recorder=phase_recorder,
            )
            if initial_state is None:
                raise ValueError(
                    "initial coefficients do not define a finite valid likelihood state"
                )
            initial_geometry = None
        else:
            initial_state, initial_geometry = reused
    initial_optimizing_objective = initial_state.penalized_optimizing_log_likelihood
    initial_reported_objective = initial_state.penalized_log_likelihood

    run = _run_iterations(
        context,
        initial_state,
        solver_config,
        initial_geometry=initial_geometry,
        stop_policy=stop_policy,
        phase_recorder=phase_recorder,
    )
    state = run.state
    coefficient_geometry = run.geometry
    history = list(run.history)
    converged = run.converged
    reason = run.reason
    score_relative = run.score_relative
    objective_relative_change = run.objective_relative_change
    step_relative = run.step_relative
    observed_geometry = run.observed_geometry

    def adopt(run: _OptimizationRun) -> None:
        nonlocal state, coefficient_geometry, converged, reason, score_relative
        nonlocal objective_relative_change, step_relative
        state = run.state
        coefficient_geometry = run.geometry
        history.extend(run.history)
        converged = run.converged
        reason = run.reason
        score_relative = run.score_relative
        objective_relative_change = run.objective_relative_change
        step_relative = run.step_relative

    def settled(
        run_context: _SolverContext, stopped: _OptimizationRun, start_iteration: int
    ) -> _OptimizationRun | None:
        # Continue a fit stopped with a row still moving towards a wall, at a
        # hundredth of the tolerance: a pressed fit keeps moving, an interior one
        # settles. Only a continuation that converges replaces the stop.
        if not (
            stopped.converged
            and _wall_step_unsettled(run_context, stopped.state, stopped.geometry, solver_config)
        ):
            return None
        continuation_config = replace(
            solver_config,
            max_iterations=solver_config.terminal_retry_iterations,
            tolerance=max(0.01 * solver_config.tolerance, 10.0 * np.finfo(np.float64).eps),
        )
        try:
            # timed with the terminal retry, so a fit that runs both records the phase twice
            with measure_phase(phase_recorder, "terminal_observed_retry_fallback"):
                continued = _run_iterations(
                    run_context,
                    stopped.state,
                    continuation_config,
                    initial_geometry=(
                        stopped.geometry if stopped.state.derivatives is None else None
                    ),
                    stop_policy=stop_policy,
                    start_iteration=start_iteration,
                    phase_recorder=phase_recorder,
                )
        except DenseSolverError:
            return None
        return continued if continued.converged else None

    continued = settled(context, run, len(history))
    del run
    if continued is not None:
        adopt(continued)
        observed_geometry = continued.observed_geometry
    del continued

    with measure_phase(phase_recorder, "terminal_observed_retry_fallback"):
        # The stopped state's decision, and after a rank-deficient one the reduced fit's.
        reduced: RetainedCoordinates | None = None
        iteration_context = context
        while True:
            observed_geometry = _geometry_from(
                context,
                state,
                "observed",
                solver_config,
                coefficient_geometry,
                observed_geometry,
                phase_recorder,
            )
            fisher_available = context.fisher_family is not None
            requested_terminal_curvature = observed_geometry.penalized_curvature
            terminal_fisher_geometry: DenseJointGeometry | None = None
            # Filled only if the policy asks for its last fallback.
            terminal_semidefinite_geometry: list[DenseJointGeometry] = []
            if context.coefficient_face is not None and context.coefficient_face.reduced_width == 0:
                empty_matrix = _readonly(np.zeros((0, 0), dtype=np.float64))
                empty_rank = decompose_gram(empty_matrix)
                curvature = CurvatureDecision(
                    matrix=empty_matrix,
                    decomposition=empty_rank,
                    telemetry=CurvatureTelemetry(
                        requested_source="observed",
                        actual_source="observed",
                        reason=None,
                        minimum_eigenvalue=0.0,
                        rank=0,
                        condition_estimate=None,
                        fallback_count=0,
                        matrix_kind="penalized",
                    ),
                    retry_required=False,
                    state=CurvaturePolicyState(),
                )
            else:
                curvature = resolve_curvature(
                    "observed",
                    _policy_curvature(context, requested_terminal_curvature),
                    state=CurvaturePolicyState(),
                    matrix_kind="penalized",
                )
            if curvature.retry_required:
                retry_tolerance = max(
                    solver_config.tolerance * solver_config.terminal_retry_tolerance_factor,
                    10.0 * np.finfo(np.float64).eps,
                )
                retry_config = replace(
                    solver_config,
                    max_iterations=solver_config.terminal_retry_iterations,
                    tolerance=retry_tolerance,
                )
                try:
                    retry = _run_iterations(
                        iteration_context,
                        state,
                        retry_config,
                        initial_geometry=(
                            coefficient_geometry if state.derivatives is None else None
                        ),
                        stop_policy=stop_policy,
                        start_iteration=len(history),
                        phase_recorder=phase_recorder,
                    )
                except DenseSolverError:
                    # The mandatory tighter attempt has still occurred.  Preserve the
                    # accepted state and let the explicit curvature policy record the
                    # Fisher fallback rather than relabeling a failed solve as success.
                    retry = None
                if retry is not None and retry.reason == "no_ascent_direction":
                    # A retry that ran out of ascent directions is a failed solve
                    # too, handled as one: the main run's state and verdict stand.
                    retry = None
                if retry is not None:
                    adopt(retry)
                    observed_geometry = _geometry_from(
                        context,
                        state,
                        "observed",
                        retry_config,
                        coefficient_geometry,
                        retry.observed_geometry,
                        phase_recorder,
                    )
                    del retry
                fisher_matrix = None
                # A dense reused endpoint (or failed retry) carries no row derivatives,
                # so evaluate it once more at the same coefficients for row geometry.
                fallback_state = state
                if context.chunk_size is None and state.derivatives is None:
                    evaluated = _evaluate_state(
                        context,
                        state.coefficients,
                        phase_recorder=phase_recorder,
                    )
                    if evaluated is not None:
                        fallback_state = evaluated
                if fisher_available:
                    try:
                        fisher_geometry = _geometry_from(
                            context,
                            fallback_state,
                            "fisher",
                            retry_config,
                            coefficient_geometry,
                            None,
                            phase_recorder,
                        )
                    except (_NumericalEvaluationError, FloatingPointError, OverflowError):
                        # Expected information the family cannot represent at this
                        # state is no fallback; the semidefinite one below is.
                        fisher_geometry = None
                    if fisher_geometry is not None:
                        terminal_fisher_geometry = fisher_geometry
                        fisher_matrix = fisher_geometry.penalized_curvature

                def _nearest_semidefinite() -> NDArray[np.float64]:
                    geometry = _measured_geometry(
                        context,
                        fallback_state,
                        "nearest_semidefinite",
                        phase_recorder,
                    )
                    terminal_semidefinite_geometry.append(geometry)
                    return _policy_curvature(context, geometry.penalized_curvature)

                requested_terminal_curvature = observed_geometry.penalized_curvature
                curvature = resolve_curvature(
                    "observed",
                    _policy_curvature(context, requested_terminal_curvature),
                    fisher_matrix=(
                        None if fisher_matrix is None else _policy_curvature(context, fisher_matrix)
                    ),
                    semidefinite_fallback=_nearest_semidefinite,
                    state=curvature.state,
                    matrix_kind="penalized",
                )
            if (
                curvature.retry_required
                or curvature.matrix is None
                or curvature.decomposition is None
            ):
                raise RuntimeError("terminal curvature policy did not reach an accepted result")

            if curvature.telemetry.actual_source == "fisher":
                if terminal_fisher_geometry is None:
                    raise RuntimeError("Fisher fallback is missing its full curvature geometry")
                accepted_geometry = terminal_fisher_geometry
            elif curvature.telemetry.actual_source == "nearest_semidefinite":
                accepted_geometry = terminal_semidefinite_geometry[0]
            else:
                accepted_geometry = observed_geometry
            if not (
                context.coefficient_face is not None and context.coefficient_face.reduced_width == 0
            ):
                curvature = _certified_terminal_curvature(
                    context, state, curvature, accepted_geometry, phase_recorder
                )
            decomposition = curvature.decomposition
            if decomposition is None:  # the certificate keeps an accepted decomposition
                raise RuntimeError("terminal rank certificate lost the accepted decomposition")
            terminal_data_curvature = np.asarray(
                accepted_geometry.data_curvature,
                dtype=np.float64,
            )
            terminal_penalized_curvature = np.asarray(
                accepted_geometry.penalized_curvature,
                dtype=np.float64,
            )
            if reduced is not None or context.coefficient_face is not None:
                break
            # Coefficients the design aliases are dropped (Wood, Pya and Saefken 2016,
            # sec. 3.1.2). A direction only the weights flatten keeps its state.
            reduced = RetainedCoordinates.of(decomposition)
            if (
                reduced is None
                or not np.any(state.coefficients[reduced.dropped])
                or not _drops_design_aliases(context.layout, decomposition)
            ):
                reduced = None
                break
            # Repeat the Newton iteration to convergence on the retained coefficients,
            # then decide the curvature at its end.
            start = _representative_state(context, state, decomposition, reduced, phase_recorder)
            if start is None:
                converged, reason = False, "rank_reduced_at_stop"
                break
            iteration_context = replace(context, retained=reduced)
            reduced_runs = [
                _run_iterations(
                    iteration_context,
                    start,
                    solver_config,
                    stop_policy=stop_policy,
                    start_iteration=len(history),
                    phase_recorder=phase_recorder,
                )
            ]
            continued = settled(
                iteration_context, reduced_runs[0], len(history) + len(reduced_runs[0].history)
            )
            if continued is not None:
                reduced_runs.append(continued)
            if not _reduced_fit_is_stationary(reduced_runs[-1], reduced, solver_config):
                # The data resolve a dropped direction: the stopped state and its decision stand.
                reduced = None
                break
            for reduced_run in reduced_runs:
                adopt(reduced_run)
            observed_geometry = reduced_runs[-1].observed_geometry
            del reduced_runs, continued
        # The reduced fit's stop holds on the coefficients it moved, and the rank
        # certified at its end must keep exactly those.
        aliases = decomposition.structural_aliases
        if reduced is not None and (
            aliases is None or not np.array_equal(aliases, reduced.dropped)
        ):
            converged, reason = False, "rank_reduced_at_stop"
        if context.coefficient_face is None:
            terminal_rank = decomposition
            terminal_reduced_rank = None
        else:
            face = context.coefficient_face
            terminal_reduced_rank = decomposition
            terminal_rank = face.lift_rank_decomposition(terminal_reduced_rank)
        terminal_score_geometry = observed_geometry
        # A certified rank that dropped a retained direction; an uncertified band that
        # kept every direction is left to the validator below.
        if (
            reason == "resolution_limited_stationarity"
            and context.coefficient_face is not None
            and terminal_reduced_rank is not None
            and (
                terminal_reduced_rank.rank < context.coefficient_face.reduced_width
                or terminal_reduced_rank.rank_truncated
            )
        ):
            converged = False
            reason = "rank_reduced_at_stop"
        if reason == "resolution_limited_stationarity":
            try:
                _validate_resolution_limited_stationarity(
                    config=solver_config,
                    converged=converged,
                    terminal_curvature=curvature.telemetry,
                    score=terminal_score_geometry.score_penalized,
                    penalized_curvature=terminal_penalized_curvature,
                    penalized_objective=state.penalized_optimizing_log_likelihood,
                    face=context.coefficient_face,
                    retained_rank=terminal_reduced_rank,
                )
            except ValueError:
                converged = False
                reason = "line_search_failed"

        if context.chunk_size is None:
            if state.eta is None or state.theta is None:
                raise RuntimeError("dense terminal state is missing predictor values")
            terminal_eta, terminal_theta = state.eta, state.theta
        else:
            with measure_phase(phase_recorder, "likelihood_evaluation"):
                terminal_eta, terminal_theta = chunking.materialize_terminal_predictions(
                    context.layout,
                    state.coefficients,
                    chunk_size=context.chunk_size,
                )
        result = DenseSolverResult(
            config=solver_config,
            family_likelihood_plan_identifier=context.likelihood_plan.plan_identifier,
            resolved_chunk_size=context.chunk_size,
            execution_backend_identifier=context.execution_backend_identifier,
            coefficients=state.coefficients,
            eta=terminal_eta,
            theta=terminal_theta,
            penalty=context.penalty,
            initial_penalized_optimizing_log_likelihood=initial_optimizing_objective,
            initial_penalized_log_likelihood=initial_reported_objective,
            optimizing_log_likelihood=state.optimizing_log_likelihood,
            parameter_independent_carrier=state.parameter_independent_carrier,
            log_likelihood=state.log_likelihood,
            penalty_value=_half_penalty_quadratic(
                context.penalty, state.coefficients, context.penalty_entries
            ),
            penalized_optimizing_log_likelihood=state.penalized_optimizing_log_likelihood,
            penalized_log_likelihood=state.penalized_log_likelihood,
            terminal_score=terminal_score_geometry.score_penalized,
            score_relative=score_relative,
            objective_relative_change=objective_relative_change,
            step_relative=step_relative,
            converged=converged,
            convergence_reason=reason,
            iterations=len(history),
            history=tuple(history),
            backtracking_steps=sum(item.backtracks for item in history),
            terminal_data_curvature=terminal_data_curvature,
            terminal_penalized_curvature=terminal_penalized_curvature,
            terminal_rank=terminal_rank,
            terminal_curvature=curvature.telemetry,
            coefficient_face=context.coefficient_face,
            terminal_reduced_rank=terminal_reduced_rank,
        )
        if _reuse_session is not None:
            _reuse_session.remember(
                result,
                reuse_owner,
                context=context,
                score_data=terminal_score_geometry.score_data,
                formation=terminal_score_geometry.formation,
            )
        return result


def fit_dense_fixed_lambda(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    penalty: NDArray,
    *,
    initial: NDArray | None = None,
    config: DenseSolverConfig | None = None,
    chunk_size: chunking.ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
    coefficient_face: PenaltyFace | None = None,
    _reuse_session: _DenseObservedReuseSession | None = None,
    _reuse_source: DenseSolverResult | None = None,
) -> DenseSolverResult:
    """Fit one fixed-penalty likelihood, optionally in bounded row chunks."""
    return _fit_dense_fixed_lambda_core(
        family,
        layout,
        y,
        likelihood_plan,
        penalty,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        coefficient_face=coefficient_face,
        _reuse_session=_reuse_session,
        _reuse_source=_reuse_source,
        stop_policy="ordinary",
    )


def _fit_dense_fixed_lambda_score_only(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    penalty: NDArray,
    *,
    initial: NDArray | None = None,
    config: DenseSolverConfig | None = None,
    chunk_size: chunking.ChunkSize | None = None,
    phase_recorder: FitPhaseRecorder | None = None,
    coefficient_face: PenaltyFace | None = None,
    _reuse_session: _DenseObservedReuseSession | None = None,
) -> DenseSolverResult:
    """Fit until the retained score meets the solver tolerance, or its rounding floor.

    A score within its floor also needs a certified Newton decrement
    (``_score_within_rounding_floor``).

    ``_reuse_session`` only shares the session's memoised dense predictor
    matrices: a score-only endpoint is never remembered for result reuse.
    """
    return _fit_dense_fixed_lambda_core(
        family,
        layout,
        y,
        likelihood_plan,
        penalty,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        coefficient_face=coefficient_face,
        _reuse_session=_reuse_session,
        stop_policy="score_only",
    )


__all__ = [
    "DenseSolverConfig",
    "DenseSolverError",
    "DenseSolverResult",
    "fit_dense_fixed_lambda",
]
