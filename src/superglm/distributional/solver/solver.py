"""Safeguarded dense fixed-lambda solver for distributional likelihoods."""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

import superglm.distributional.solver.chunks as chunking
from superglm.distributional.family import (
    ConfigurableDistributionalFamily,
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    _likelihood_reuse_contract,
)
from superglm.distributional.layout import StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.result import (
    CHUNKED_EXECUTION_BACKEND_IDENTIFIER,
    DENSE_EXECUTION_BACKEND_IDENTIFIER,
    CoefficientCurvature,
    ConvergenceReason,
    DenseSolverConfig,
    DenseSolverResult,
    ExecutionBackendIdentifier,
    SolverIteration,
    _resolution_limited_decrement_is_within_objective_ulp,
    _validate_resolution_limited_stationarity,
)
from superglm.distributional.results.solver import _newton_decrement_is_certified
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver._likelihood_cache import (
    _is_builtin_gaussian_gamma,
    _LikelihoodCache,
    build_likelihood_cache,
)
from superglm.distributional.solver._reuse_digest import _ReuseDigest
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
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import IdentityLink, Link, LogLink
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
            int, tuple[weakref.ReferenceType[DenseSolverResult], _DenseObservedReuseOwner]
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
            certificate = _chunk_reuse_data_certificate(context)
            if certificate is not None:
                key = id(result)
                self._chunk_results[key] = _ChunkObservedReuseRecord(
                    source=self._source_reference(result, chunked=True),
                    owner=owner,
                    coefficients=_readonly(result.coefficients),
                    score_data=_readonly(score_data),
                    data_curvature=_readonly(result.terminal_data_curvature),
                    certificate=certificate,
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
    certificate: str


def _chunk_reuse_data_certificate(context: _SolverContext) -> str | None:
    """Certify supported fixed designs without materializing observation rows.

    Public design arrays can be mutable. Identity and equal predictors alone do
    not certify curvature: a design edit can leave the current predictor fixed.
    Unknown representations conservatively retain fresh likelihood evaluation.
    """
    with _ReuseDigest() as digest:
        return _build_chunk_reuse_data_certificate(context, digest)


def _build_chunk_reuse_data_certificate(
    context: _SolverContext, digest: _ReuseDigest
) -> str | None:
    from scipy.sparse import csr_matrix

    from superglm.distributional.weights import ResolvedLikelihoodWeights
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        FactorSmoothGroupMatrix,
        RandomEffectGroupMatrix,
        SparseGroupMatrix,
        SplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
        SupportCompressedSSPGroupMatrix,
    )
    from superglm.links import IdentityLink, LogLink

    contract = _likelihood_reuse_contract(context.family)
    weights = context.likelihood_plan.weights
    if (
        contract is None
        or type(context.likelihood_plan) is not contract.plan_type
        or type(weights) is not ResolvedLikelihoodWeights
    ):
        return None
    field = digest.field
    array = digest.array

    def builtin_array(values: object) -> bool:
        # An ndarray subclass can change arithmetic at identical bytes. Limit
        # this schema to ordinary real buffers, with at most 64 KiB per digest
        # block under array()'s 8192-element iterator.
        return bool(
            type(values) is np.ndarray
            and values.dtype.kind in "biuf"
            and values.dtype.itemsize <= 8
        )

    def splinecat_csr(matrix: object) -> bool:
        if type(matrix) is not csr_matrix:
            return False
        buffers = (matrix.data, matrix.indices, matrix.indptr)
        if not all(builtin_array(values) for values in buffers):
            return False
        # Read raw flags: the public properties can populate caches. Their
        # values affect SciPy dispatch even when the CSR bytes are unchanged.
        flags = (
            getattr(matrix, "_has_sorted_indices", None),
            getattr(matrix, "_has_canonical_format", None),
        )
        if any(value is not None and type(value) not in (bool, np.bool_) for value in flags):
            return False
        field((matrix.shape, matrix.dtype.str, flags))
        for values in buffers:
            array(values)
        return True

    field(
        json.dumps(
            cast(ConfigurableDistributionalFamily, context.family).to_config(), sort_keys=True
        )
    )
    field(context.likelihood_plan.plan_identifier)
    # Some built-in identifiers contain stored weight digests. The evaluator
    # consumes the live contract, so certify it independently of those digests.
    field(weights.provenance.contract.semantics)
    field((context.chunk_size, context.layout.n_coefficients))
    array(context.response)
    for name in contract.prepared_array_fields:
        array(getattr(context.likelihood_plan, name))
    for name in ("values", "geometry_values", "root_take_map", "input_positions"):
        array(getattr(weights, name))
    for state in context.layout.predictors:
        if type(state.link) not in (IdentityLink, LogLink, *contract.link_types):
            return None
        field((type(state.link).__name__, vars(state.link)))
        field(
            (
                state.name,
                state.parameter_index,
                state.coefficient_slice,
                state.intercept_index,
                state.design.shape,
            )
        )
        array(state.offset)
        for group in state.design.group_matrices:
            kind = type(group)
            field((kind.__name__, group.shape))
            if kind is DenseGroupMatrix:
                names = ("M",)
            elif kind in (CategoricalGroupMatrix, RandomEffectGroupMatrix):
                group = cast(CategoricalGroupMatrix | RandomEffectGroupMatrix, group)
                names = ("codes",)
                field(group.n_levels)
            elif kind is SparseGroupMatrix:
                group = cast(SparseGroupMatrix, group)
                matrix = group.M
                if type(matrix) is not csr_matrix:
                    return None
                field((matrix.shape, matrix.dtype.str))
                field(
                    (
                        getattr(matrix, "_has_sorted_indices", None),
                        getattr(matrix, "_has_canonical_format", None),
                    )
                )
                for values in (matrix.data, matrix.indices, matrix.indptr):
                    if type(values) is not np.ndarray:
                        return None
                    array(values)
                continue
            elif kind is FactorSmoothGroupMatrix:
                group = cast(FactorSmoothGroupMatrix, group)
                field(
                    (
                        group.n_levels,
                        group.coefficient_levels,
                        group.block_size,
                        group.raw_width,
                        group.factor_basis,
                        group.is_discrete,
                    )
                )
                # Row subsetting reconstructs the group from B, while exact
                # matvec and moments read the separately stored CSR buffers.
                # Certify both: public attributes can be replaced independently.
                if group.is_discrete:
                    if any(
                        value is not None
                        for value in (group.B, group._data, group._indices, group._indptr)
                    ):
                        return None
                    names = ("codes", "natural_map", "B_unique", "bin_idx")
                else:
                    matrix = group.B
                    if (
                        type(matrix) is not csr_matrix
                        or group.B_unique is not None
                        or group.bin_idx is not None
                    ):
                        return None
                    field((matrix.shape, matrix.dtype.str))
                    field(
                        (
                            getattr(matrix, "_has_sorted_indices", None),
                            getattr(matrix, "_has_canonical_format", None),
                        )
                    )
                    for values in (matrix.data, matrix.indices, matrix.indptr):
                        if type(values) is not np.ndarray:
                            return None
                        array(values)
                    names = ("codes", "natural_map", "_data", "_indices", "_indptr")
                if any(type(getattr(group, name)) is not np.ndarray for name in names):
                    return None
            elif kind in (
                SplineCategoricalGroupMatrix,
                DiscretizedSplineCategoricalGroupMatrix,
                SupportCompressedSplineCategoricalGroupMatrix,
            ):
                group = cast(
                    SplineCategoricalGroupMatrix
                    | DiscretizedSplineCategoricalGroupMatrix
                    | SupportCompressedSplineCategoricalGroupMatrix,
                    group,
                )
                if (
                    type(group.n_rows) is not int
                    or type(group._p_b) is not int
                    or (
                        group.spline_cat_feature is not None
                        and type(group.spline_cat_feature) is not str
                    )
                ):
                    return None
                # The feature selects a same-parent cross-Gram route.
                field((group.n_rows, group._p_b, group.spline_cat_feature))
                names = ("R_inv", "row_idx")
                if kind is SplineCategoricalGroupMatrix:
                    group = cast(SplineCategoricalGroupMatrix, group)
                    # Subsetting reads full B; matvec reads B_level; sparse
                    # moments read independent buffers. A saturated Gram can
                    # instead read its lazy dense copy. All remain live.
                    if not splinecat_csr(group.B) or not splinecat_csr(group.B_level):
                        return None
                    names += ("_data", "_indices", "_indptr")
                    dense = group._dense_level
                    if dense is None or dense is False:
                        field(("_dense_level", dense))
                    elif builtin_array(dense):
                        names += ("_dense_level",)
                    else:
                        return None
                    cache_names = ("_sorted_rows",)
                else:
                    group = cast(
                        DiscretizedSplineCategoricalGroupMatrix
                        | SupportCompressedSplineCategoricalGroupMatrix,
                        group,
                    )
                    if type(group.n_bins) is not int:
                        return None
                    field(group.n_bins)
                    names += ("B_unique", "bin_idx_level")
                    cache_names = ("_row_order", "_sorted_rows")
                # Do not sort, materialize, or populate a lookup to certify it.
                # Cache creation, replacement and mutation each change the
                # certificate, including independently replaced alignment maps.
                for name in cache_names:
                    value = getattr(group, name)
                    field((name, value is None))
                    if value is not None:
                        names += (name,)
                if any(not builtin_array(getattr(group, name)) for name in names):
                    return None
            elif kind in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix):
                group = cast(DiscretizedSSPGroupMatrix | SupportCompressedSSPGroupMatrix, group)
                names = ("B_unique", "R_inv", "bin_idx")
                field(group.n_bins)
            elif kind is DiscretizedSCOPGroupMatrix:
                group = cast(DiscretizedSCOPGroupMatrix, group)
                names = ("B_scop_unique", "bin_idx")
                field(group.n_bins)
            elif kind is DiscretizedTensorGroupMatrix:
                group = cast(DiscretizedTensorGroupMatrix, group)
                names = (
                    "B_unique",
                    "R_inv",
                    "bin_idx",
                    "B1_unique_t",
                    "B2_unique_t",
                    "idx1",
                    "idx2",
                )
                field((group.tensor_id, group.n_bins, group.n_bins1, group.n_bins2))
            else:
                return None
            for name in names:
                array(getattr(group, name))
    return digest.hexdigest()


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
    fisher_family = family if isinstance(family, ExpectedInformationFamily) else None
    if coefficient_curvature == "fisher" and fisher_family is None:
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
    return _SolverContext(
        family=family,
        fisher_family=fisher_family,
        layout=layout,
        response=response,
        likelihood_plan=root_likelihood_plan,
        penalty=_readonly(penalty_matrix),
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
            penalty_value = 0.5 * float(coefficient_values @ context.penalty @ coefficient_values)
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
        penalty_value = 0.5 * float(coefficient_values @ context.penalty @ coefficient_values)
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
    penalty_value = 0.5 * float(values @ context.penalty @ values)
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
            gram = plan.diagonal_moment(weights)
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
        weighted_target = target * square_root_weight
        local = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)[0]
        coefficients[state.coefficient_slice] = local
    return _readonly(coefficients)


def _geometry(
    context: _SolverContext,
    state: _AcceptedState,
    source: str,
) -> DenseJointGeometry:
    if context.chunk_size is not None:
        if source not in ("observed", "fisher"):
            raise ValueError("curvature source must be 'observed' or 'fisher'")
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
    if source == "observed":
        curvature = state.derivatives.curvature_packed
    elif source == "fisher":
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
    else:
        raise ValueError("curvature source must be 'observed' or 'fisher'")
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


def _measured_geometry(
    context: _SolverContext,
    state: _AcceptedState,
    source: str,
    phase_recorder: FitPhaseRecorder | None,
) -> DenseJointGeometry:
    with measure_phase(phase_recorder, "curvature_gradient_assembly"):
        return _geometry(context, state, source)


def _relative_score(score: NDArray, objective: float) -> float:
    return float(np.linalg.norm(score, ord=np.inf) / (1.0 + abs(objective)))


def _optimization_score(context: _SolverContext, score: NDArray) -> NDArray[np.float64]:
    if context.coefficient_face is None:
        return np.asarray(score, dtype=np.float64)
    return context.coefficient_face.reduce_vector(score)


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
            face=context.coefficient_face,
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
            residual = float(
                np.linalg.norm(matrix @ step - score) / max(1.0, float(np.linalg.norm(score)))
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
        residual = float(
            np.linalg.norm(system @ step - score) / max(1.0, float(np.linalg.norm(score)))
        )
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
    face = context.coefficient_face
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

    for local_iteration in range(config.max_iterations):
        score_relative = _optimization_score_relative(
            context,
            geometry.score_penalized,
            state.penalized_optimizing_log_likelihood,
        )
        if score_relative <= config.tolerance:
            return _OptimizationRun(
                state=state,
                geometry=geometry,
                observed_geometry=observed_geometry,
                history=tuple(history),
                converged=True,
                reason="score",
                score_relative=score_relative,
                objective_relative_change=objective_relative_change,
                step_relative=step_relative,
            )

        with measure_phase(phase_recorder, "coefficient_decomposition_solve"):
            direction = _solve_coefficient_direction(
                context,
                geometry.penalized_curvature,
                geometry.score_penalized,
                config,
            )
        optimization_score = _optimization_score(context, geometry.score_penalized)
        if (
            stop_policy == "ordinary"
            and config.newton_decrement_tolerance is not None
            and config.coefficient_curvature == "observed"
            and direction.levenberg_shift == 0.0
            and direction.decomposition.rank == len(optimization_score)
        ):
            if _newton_decrement_is_certified(
                config=config,
                score=geometry.score_penalized,
                penalized_curvature=geometry.penalized_curvature,
                penalized_objective=state.penalized_optimizing_log_likelihood,
                face=context.coefficient_face,
                tolerance=config.newton_decrement_tolerance,
            ):
                return _OptimizationRun(
                    state=state,
                    geometry=geometry,
                    observed_geometry=observed_geometry,
                    history=tuple(history),
                    converged=True,
                    reason="newton_decrement",
                    score_relative=score_relative,
                    objective_relative_change=objective_relative_change,
                    step_relative=float(
                        np.linalg.norm(direction.step, ord=np.inf)
                        / (1.0 + np.linalg.norm(state.coefficients, ord=np.inf))
                    ),
                )
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
                accepted = candidate
                accepted_geometry = None if fused is None else fused[1]
                fused = None
                break
            fused = None
            if attempt == config.max_backtracks:
                break
            alpha *= config.backtrack_factor
            backtracks += 1
        if accepted is None:
            if stop_policy == "ordinary" and reached_identical_candidate:
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
                    return _OptimizationRun(
                        state=state,
                        geometry=geometry,
                        observed_geometry=observed_geometry,
                        history=tuple(history),
                        converged=True,
                        reason="objective_and_step",
                        score_relative=score_relative,
                        objective_relative_change=0.0,
                        step_relative=0.0,
                    )
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
            return _OptimizationRun(
                state=state,
                geometry=geometry,
                observed_geometry=observed_geometry,
                history=tuple(history),
                converged=resolution_limited,
                reason=(
                    "resolution_limited_stationarity"
                    if resolution_limited
                    else "line_search_failed"
                ),
                score_relative=score_relative,
                objective_relative_change=objective_relative_change,
                step_relative=step_relative,
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
                return _OptimizationRun(
                    state=state,
                    geometry=geometry,
                    observed_geometry=observed_geometry,
                    history=tuple(history),
                    converged=True,
                    reason="score",
                    score_relative=accepted_score_relative,
                    objective_relative_change=objective_relative_change,
                    step_relative=step_relative,
                )
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
                return _OptimizationRun(
                    state=state,
                    geometry=geometry,
                    observed_geometry=observed_geometry,
                    history=tuple(history),
                    converged=True,
                    reason="objective_and_step",
                    score_relative=accepted_score_relative,
                    objective_relative_change=objective_relative_change,
                    step_relative=step_relative,
                )
            # A distinct tiny step is still a valid state transition.  Keep
            # the remaining iteration budget so a fresh direction can recover
            # stationarity; only an identical candidate has no progress to
            # retry and returns ``line_search_failed`` below.
        if (
            objective_relative_change <= config.tolerance
            and accepted_score_relative <= config.tolerance
        ):
            return _OptimizationRun(
                state=state,
                geometry=geometry,
                observed_geometry=observed_geometry,
                history=tuple(history),
                converged=True,
                reason="objective_and_score",
                score_relative=accepted_score_relative,
                objective_relative_change=objective_relative_change,
                step_relative=step_relative,
            )

    return _OptimizationRun(
        state=state,
        geometry=geometry,
        observed_geometry=observed_geometry,
        history=tuple(history),
        converged=False,
        reason="max_iterations",
        score_relative=_optimization_score_relative(
            context,
            geometry.score_penalized,
            state.penalized_optimizing_log_likelihood,
        ),
        objective_relative_change=objective_relative_change,
        step_relative=step_relative,
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
    else:
        record = session._chunk_results.get(id(source))
        if (
            record is None
            or record.source() is not source
            or not record.owner.matches(owner)
            or source.resolved_chunk_size != context.chunk_size
            or source.execution_backend_identifier != CHUNKED_EXECUTION_BACKEND_IDENTIFIER
            or not np.array_equal(coefficients, record.coefficients)
            or _chunk_reuse_data_certificate(context) != record.certificate
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

    penalty_value = 0.5 * float(coefficients @ context.penalty @ coefficients)
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
    )
    return state, geometry


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
    del run

    with measure_phase(phase_recorder, "terminal_observed_retry_fallback"):
        observed_geometry = (
            coefficient_geometry
            if solver_config.coefficient_curvature == "observed"
            else observed_geometry
            if observed_geometry is not None
            else _measured_geometry(
                context,
                state,
                "observed",
                phase_recorder,
            )
        )
        fisher_available = context.fisher_family is not None
        requested_terminal_curvature = observed_geometry.penalized_curvature
        terminal_fisher_geometry: DenseJointGeometry | None = None
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
                newton_decrement_tolerance=None,
            )
            try:
                retry = _run_iterations(
                    context,
                    state,
                    retry_config,
                    initial_geometry=(coefficient_geometry if state.derivatives is None else None),
                    stop_policy=stop_policy,
                    start_iteration=len(history),
                    phase_recorder=phase_recorder,
                )
            except DenseSolverError:
                # The mandatory tighter attempt has still occurred.  Preserve the
                # accepted state and let the explicit curvature policy record the
                # Fisher fallback rather than relabeling a failed solve as success.
                retry = None
            if retry is not None:
                state = retry.state
                coefficient_geometry = retry.geometry
                history.extend(retry.history)
                converged = retry.converged
                reason = retry.reason
                score_relative = retry.score_relative
                objective_relative_change = retry.objective_relative_change
                step_relative = retry.step_relative
                observed_geometry = (
                    coefficient_geometry
                    if retry_config.coefficient_curvature == "observed"
                    else retry.observed_geometry
                    if retry.observed_geometry is not None
                    else _measured_geometry(
                        context,
                        state,
                        "observed",
                        phase_recorder,
                    )
                )
                del retry
            fisher_matrix = None
            # A dense reused endpoint carries no row derivatives (see
            # _reuse_observed_initial_result), and a failed retry keeps that
            # state.  Dense Fisher geometry cannot be measured from it, so the
            # policy sees no Fisher fallback -- the same guard the retry above
            # applies when it reuses the accepted geometry instead.  Chunked
            # geometry is assembled from rows and never needs the derivatives.
            fisher_measurable = context.chunk_size is not None or state.derivatives is not None
            if fisher_available and fisher_measurable:
                fisher_geometry = (
                    coefficient_geometry
                    if retry_config.coefficient_curvature == "fisher"
                    else _measured_geometry(
                        context,
                        state,
                        "fisher",
                        phase_recorder,
                    )
                )
                terminal_fisher_geometry = fisher_geometry
                fisher_matrix = fisher_geometry.penalized_curvature
            requested_terminal_curvature = observed_geometry.penalized_curvature
            curvature = resolve_curvature(
                "observed",
                _policy_curvature(context, requested_terminal_curvature),
                fisher_matrix=(
                    None if fisher_matrix is None else _policy_curvature(context, fisher_matrix)
                ),
                state=curvature.state,
                matrix_kind="penalized",
            )
        if curvature.retry_required or curvature.matrix is None or curvature.decomposition is None:
            raise RuntimeError("terminal curvature policy did not reach an accepted result")

        if curvature.telemetry.actual_source == "fisher":
            if terminal_fisher_geometry is None:
                raise RuntimeError("Fisher fallback is missing its full curvature geometry")
            accepted_geometry = terminal_fisher_geometry
        else:
            accepted_geometry = observed_geometry
        terminal_data_curvature = np.asarray(
            accepted_geometry.data_curvature,
            dtype=np.float64,
        )
        terminal_penalized_curvature = np.asarray(
            accepted_geometry.penalized_curvature,
            dtype=np.float64,
        )
        if context.coefficient_face is None:
            terminal_rank = curvature.decomposition
            terminal_reduced_rank = None
        else:
            face = context.coefficient_face
            terminal_reduced_rank = curvature.decomposition
            terminal_rank = face.lift_rank_decomposition(terminal_reduced_rank)
        terminal_score_geometry = observed_geometry
        penalty_value = 0.5 * float(state.coefficients @ context.penalty @ state.coefficients)
        if context.chunk_size is None:
            if state.eta is None or state.theta is None:
                raise RuntimeError("dense terminal state is missing predictor values")
            terminal_eta = state.eta
            terminal_theta = state.theta
        else:
            with measure_phase(phase_recorder, "likelihood_evaluation"):
                terminal_eta, terminal_theta = chunking.materialize_terminal_predictions(
                    context.layout,
                    state.coefficients,
                    chunk_size=context.chunk_size,
                )
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
            penalty_value=penalty_value,
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
                result, reuse_owner, context=context, score_data=terminal_score_geometry.score_data
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
    """Fit until the retained score alone satisfies the solver tolerance.

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
