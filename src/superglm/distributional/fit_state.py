"""Transactional, immutable fitted-state publication for distributional models."""

from __future__ import annotations

import copy
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.inference import JointInference, compute_joint_inference
from superglm.distributional.layout import StackedLayout
from superglm.distributional.null_model import JointNullModel, fit_joint_null_model
from superglm.distributional.predictor import CompiledPredictor, Predictor
from superglm.distributional.result import (
    ConvergenceReason,
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSResult,
    DistributionalFitResult,
)
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    WeightContract,
    WeightProvenance,
)


def _readonly(values: NDArray, *, name: str) -> NDArray[np.float64]:
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numeric values")
    result.setflags(write=False)
    return result


def _frozen_float_mapping(
    values: Mapping[str, float],
    *,
    name: str,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key or isinstance(value, bool):
            raise ValueError(f"{name} must map non-empty names to finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must map non-empty names to finite numbers")
        result[key] = number
    return MappingProxyType(result)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _frozen_n_bins(value: int | Mapping[str, int]) -> int | Mapping[str, int]:
    if isinstance(value, Mapping):
        result: dict[str, int] = {}
        for name, count in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("requested_n_bins keys must be non-empty feature names")
            result[name] = _positive_integer(
                count,
                name=f"requested_n_bins[{name!r}]",
            )
        return MappingProxyType(result)
    return _positive_integer(value, name="requested_n_bins")


def _requested_chunk_size(value: ChunkSize | None) -> ChunkSize | None:
    if value is None or value == "auto":
        return value
    if isinstance(value, str):
        raise ValueError("requested_chunk_size string must be 'auto'")
    return _positive_integer(value, name="requested_chunk_size")


@dataclass(frozen=True)
class DistributionalRowState:
    """Optional retained training-row arrays, isolated from compact inference."""

    response: NDArray[np.float64]
    likelihood_weights: ResolvedLikelihoodWeights
    offsets: Mapping[str, NDArray[np.float64]]
    fitted_eta: NDArray[np.float64]
    fitted_parameters: NDArray[np.float64]
    null_eta: NDArray[np.float64]
    null_parameters: NDArray[np.float64]

    def __post_init__(self) -> None:
        response = _readonly(self.response, name="response")
        if not isinstance(self.likelihood_weights, ResolvedLikelihoodWeights):
            raise TypeError("likelihood_weights must be ResolvedLikelihoodWeights")
        weights = self.likelihood_weights.values
        fitted_eta = _readonly(self.fitted_eta, name="fitted_eta")
        fitted_parameters = _readonly(self.fitted_parameters, name="fitted_parameters")
        null_eta = _readonly(self.null_eta, name="null_eta")
        null_parameters = _readonly(self.null_parameters, name="null_parameters")
        n_observations = len(response)
        if response.ndim != 1 or weights.shape != (n_observations,) or np.any(weights <= 0.0):
            raise ValueError(
                "response and likelihood weights must be matching positive row vectors"
            )
        provenance = self.likelihood_weights.provenance
        if (
            self.likelihood_weights.digest != provenance.root_digest
            or provenance.retained_count != n_observations
            or not np.array_equal(
                self.likelihood_weights.root_take_map,
                np.arange(n_observations, dtype=np.intp),
            )
        ):
            raise ValueError("retained likelihood weights must be the fitted root carrier")
        if (
            fitted_eta.ndim != 2
            or fitted_parameters.shape != fitted_eta.shape
            or null_eta.shape != fitted_eta.shape
            or null_parameters.shape != fitted_eta.shape
            or fitted_eta.shape[0] != n_observations
        ):
            raise ValueError("retained predictor and parameter arrays must share row dimensions")
        if not isinstance(self.offsets, Mapping):
            raise TypeError("offsets must be a parameter mapping")
        offsets: dict[str, NDArray[np.float64]] = {}
        for name, values in self.offsets.items():
            if not isinstance(name, str) or not name:
                raise ValueError("offset names must be non-empty strings")
            offset = _readonly(values, name=f"offset for {name!r}")
            if offset.shape != (n_observations,):
                raise ValueError("retained offsets must match the response row count")
            offsets[name] = offset
        if len(offsets) != fitted_eta.shape[1]:
            raise ValueError("one retained offset is required per fitted parameter")
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "offsets", MappingProxyType(offsets))
        object.__setattr__(self, "fitted_eta", fitted_eta)
        object.__setattr__(self, "fitted_parameters", fitted_parameters)
        object.__setattr__(self, "null_eta", null_eta)
        object.__setattr__(self, "null_parameters", null_parameters)


@dataclass(frozen=True)
class CompactNullModel:
    """Row-free null-model definition and terminal diagnostics for metrics."""

    family_config: Mapping[str, Any]
    parameter_names: tuple[str, ...]
    link_types: Mapping[str, str]
    offset_semantics: Mapping[str, str]
    weight_contract: WeightContract
    weight_provenance: WeightProvenance
    family_likelihood_plan_identifier: str
    n_observations: int
    weight_sum: float
    coefficients: NDArray[np.float64]
    objective: float
    log_likelihood: float
    converged: bool
    convergence_reason: ConvergenceReason
    rank: int
    curvature_telemetry: CurvatureTelemetry

    def __post_init__(self) -> None:
        if not isinstance(self.family_config, Mapping) or not self.family_config:
            raise ValueError("family_config must be a non-empty mapping")
        family_config = MappingProxyType(copy.deepcopy(dict(self.family_config)))
        parameter_names = tuple(self.parameter_names)
        if not parameter_names or len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must be unique and non-empty")
        for name, values in (
            ("link_types", self.link_types),
            ("offset_semantics", self.offset_semantics),
        ):
            if not isinstance(values, Mapping) or tuple(values) != parameter_names:
                raise ValueError(f"{name} must follow parameter order")
            if any(not isinstance(value, str) or not value for value in values.values()):
                raise ValueError(f"{name} values must be non-empty strings")
        if not isinstance(self.weight_contract, WeightContract):
            raise TypeError("weight_contract must be a WeightContract")
        if not isinstance(self.weight_provenance, WeightProvenance):
            raise TypeError("weight_provenance must be WeightProvenance")
        if self.weight_provenance.contract != self.weight_contract:
            raise ValueError("null weight provenance must agree with its contract")
        if (
            not isinstance(self.family_likelihood_plan_identifier, str)
            or not self.family_likelihood_plan_identifier
        ):
            raise ValueError("null family likelihood plan identifier must be non-empty")
        if (
            isinstance(self.n_observations, bool)
            or not isinstance(self.n_observations, int)
            or self.n_observations < 1
        ):
            raise ValueError("n_observations must be a positive integer")
        if self.n_observations != self.weight_provenance.retained_count:
            raise ValueError("null row count must agree with weight provenance")
        weight_sum = float(self.weight_sum)
        coefficients = _readonly(self.coefficients, name="null coefficients")
        if (
            not math.isfinite(weight_sum)
            or weight_sum <= 0.0
            or coefficients.shape != (len(parameter_names),)
        ):
            raise ValueError("null weights and coefficients have invalid dimensions")
        if weight_sum != self.weight_provenance.weight_sum:
            raise ValueError("null weight sum must agree with weight provenance")
        for name in ("objective", "log_likelihood"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if not isinstance(self.curvature_telemetry, CurvatureTelemetry):
            raise TypeError("curvature_telemetry must be CurvatureTelemetry")
        object.__setattr__(self, "family_config", family_config)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "link_types", MappingProxyType(dict(self.link_types)))
        object.__setattr__(
            self,
            "offset_semantics",
            MappingProxyType(dict(self.offset_semantics)),
        )
        object.__setattr__(self, "weight_sum", weight_sum)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def weight_semantics(self) -> str:
        """Likelihood-weight semantic derived from the canonical contract."""
        return self.weight_contract.semantics


@dataclass(frozen=True)
class DistributionalFitState:
    """One fully validated revision ready for a single-reference publication."""

    revision: int
    weight_contract: WeightContract
    weight_provenance: WeightProvenance
    family_likelihood_plan_identifier: str
    requested_discrete: bool
    requested_n_bins: int | Mapping[str, int]
    requested_chunk_size: ChunkSize | None
    predictor_templates: tuple[Predictor, ...]
    compiled_predictors: tuple[CompiledPredictor, ...]
    layout: StackedLayout
    lambdas: Mapping[str, float]
    solver_result: DenseSolverResult
    smoothing: DistributionalEFSResult | None
    inference: JointInference
    null_model: CompactNullModel
    result: DistributionalFitResult
    retained_rows: DistributionalRowState | None
    exact_face_components: tuple[str, ...] = ()

    @property
    def requested_solver_config(self) -> DenseSolverConfig:
        """Return the ordinary coefficient policy requested for this fit."""

        if self.smoothing is None:
            return self.solver_result.config
        return self.smoothing.coefficient_fits[0].config

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise ValueError("revision must be a positive integer")
        if not isinstance(self.weight_contract, WeightContract):
            raise TypeError("weight_contract must be a WeightContract")
        if not isinstance(self.weight_provenance, WeightProvenance):
            raise TypeError("weight_provenance must be WeightProvenance")
        if self.weight_provenance.contract != self.weight_contract:
            raise ValueError("weight provenance must agree with the canonical contract")
        if (
            not isinstance(self.family_likelihood_plan_identifier, str)
            or not self.family_likelihood_plan_identifier
        ):
            raise ValueError("family likelihood plan identifier must be non-empty")
        if not isinstance(self.requested_discrete, bool):
            raise TypeError("requested_discrete must be bool")
        requested_n_bins = _frozen_n_bins(self.requested_n_bins)
        requested_chunk_size = _requested_chunk_size(self.requested_chunk_size)
        templates = tuple(self.predictor_templates)
        compiled = tuple(self.compiled_predictors)
        if not templates or not all(isinstance(value, Predictor) for value in templates):
            raise TypeError("predictor_templates must contain Predictor values")
        if not compiled or not all(isinstance(value, CompiledPredictor) for value in compiled):
            raise TypeError("compiled_predictors must contain CompiledPredictor values")
        if not isinstance(self.layout, StackedLayout):
            raise TypeError("layout must be a StackedLayout")
        if len(templates) != len(compiled) or len(compiled) != len(self.layout.predictors):
            raise ValueError("predictor templates, compilation, and layout must agree")
        template_names = tuple(value.name for value in templates)
        compiled_names = tuple(value.name for value in compiled)
        layout_names = tuple(value.name for value in self.layout.predictors)
        if template_names != compiled_names or compiled_names != layout_names:
            raise ValueError("predictor ordering must remain stable through fitted state")
        lambdas = _frozen_float_mapping(self.lambdas, name="lambdas")
        if tuple(lambdas) != self.layout.penalty_names:
            raise ValueError("lambda order must match the stacked layout")
        if not isinstance(self.solver_result, DenseSolverResult):
            raise TypeError("solver_result must be DenseSolverResult")
        if (
            self.solver_result.family_likelihood_plan_identifier
            != self.family_likelihood_plan_identifier
        ):
            raise ValueError("solver result must identify the fitted root likelihood plan")
        if self.solver_result.coefficients.shape != (self.layout.n_coefficients,):
            raise ValueError("solver result must match the stacked layout")
        try:
            exact_face_components = tuple(self.exact_face_components)
        except TypeError as exc:
            raise TypeError("exact_face_components must be an iterable of names") from exc
        if any(not isinstance(name, str) for name in exact_face_components):
            raise ValueError("exact_face_components must contain only names")
        exact_face_set = set(exact_face_components)
        if (
            any(name not in self.layout.penalty_names for name in exact_face_components)
            or len(exact_face_set) != len(exact_face_components)
            or tuple(name for name in self.layout.penalty_names if name in exact_face_set)
            != exact_face_components
        ):
            raise ValueError(
                "exact_face_components must be unique and follow the fitted penalty order"
            )
        solver_face = self.solver_result.coefficient_face
        accepted_face_components = () if solver_face is None else solver_face.component_names
        if exact_face_components != accepted_face_components:
            raise ValueError("exact face components must match the accepted terminal solver face")
        if solver_face is not None:
            solver_face.validate_layout(self.layout)
        fitted_penalty_lambdas = dict(lambdas)
        for name in exact_face_components:
            fitted_penalty_lambdas[name] = 0.0
        expected_terminal_penalty = self.layout.penalty_matrix(fitted_penalty_lambdas)
        if not np.array_equal(self.solver_result.penalty, expected_terminal_penalty):
            raise ValueError(
                "terminal penalty must agree with the fitted layout and fitted lambdas"
            )
        if self.smoothing is None and exact_face_components:
            raise ValueError("an exact coefficient face requires a smoothing result")
        if self.smoothing is not None:
            if not isinstance(self.smoothing, DistributionalEFSResult):
                raise TypeError("smoothing must be DistributionalEFSResult")
            if self.smoothing.terminal_fit is not self.solver_result:
                raise ValueError("solver result must be the accepted terminal EFS fit")
            if dict(self.smoothing.lambdas) != dict(lambdas):
                raise ValueError("smoothing and fitted lambdas must agree")
            if any(
                fit.family_likelihood_plan_identifier != self.family_likelihood_plan_identifier
                for fit in self.smoothing.coefficient_fits
            ):
                raise ValueError("every EFS coefficient fit must identify the fitted root plan")
        if not isinstance(self.inference, JointInference):
            raise TypeError("inference must be JointInference")
        if not isinstance(self.null_model, CompactNullModel):
            raise TypeError("null_model must be CompactNullModel")
        if (
            self.null_model.weight_contract != self.weight_contract
            or self.null_model.weight_provenance != self.weight_provenance
            or self.null_model.family_likelihood_plan_identifier
            != self.family_likelihood_plan_identifier
        ):
            raise ValueError("null metadata must agree with canonical likelihood state")
        if not isinstance(self.result, DistributionalFitResult):
            raise TypeError("result must be DistributionalFitResult")
        if self.result.exact_face_components != exact_face_components:
            raise ValueError("compact and solver exact-face components must agree")
        if not np.array_equal(self.result.coefficients, self.solver_result.coefficients):
            raise ValueError("compact and solver coefficients must agree")
        if not np.array_equal(self.result.covariance, self.inference.covariance):
            raise ValueError("compact and inference covariance must agree")
        if self.result.rank != self.inference.rank:
            raise ValueError("compact and inference rank must agree")
        if self.result.null_objective != self.null_model.objective:
            raise ValueError("compact fit and null objective must agree")
        if self.retained_rows is not None and not isinstance(
            self.retained_rows, DistributionalRowState
        ):
            raise TypeError("retained_rows must be DistributionalRowState or None")
        if self.retained_rows is not None:
            retained_weights = self.retained_rows.likelihood_weights
            if (
                retained_weights.provenance != self.weight_provenance
                or retained_weights.digest != self.weight_provenance.root_digest
            ):
                raise ValueError("retained rows must own the canonical root weight carrier")
        if self.weight_provenance.retained_count != self.layout.predictors[0].design.n:
            raise ValueError("weight provenance must match the retained layout rows")
        object.__setattr__(self, "predictor_templates", templates)
        object.__setattr__(self, "compiled_predictors", compiled)
        object.__setattr__(self, "lambdas", lambdas)
        object.__setattr__(self, "requested_n_bins", requested_n_bins)
        object.__setattr__(self, "requested_chunk_size", requested_chunk_size)
        object.__setattr__(self, "exact_face_components", exact_face_components)


def _compact_null(null: JointNullModel) -> CompactNullModel:
    link_types = {
        name: f"{type(link).__module__}.{type(link).__qualname__}"
        for name, link in null.parameter_links.items()
    }
    offset_semantics = {
        name: ("zero_offset" if not np.any(values != 0.0) else "nonzero_training_offset")
        for name, values in null.offsets.items()
    }
    return CompactNullModel(
        family_config=null.family_config,
        parameter_names=null.parameter_names,
        link_types=link_types,
        offset_semantics=offset_semantics,
        weight_contract=null.weight_contract,
        weight_provenance=null.weight_provenance,
        family_likelihood_plan_identifier=null.result.family_likelihood_plan_identifier,
        n_observations=null.n_observations,
        weight_sum=null.weight_sum,
        coefficients=null.result.coefficients,
        objective=null.objective,
        log_likelihood=null.result.log_likelihood,
        converged=null.converged,
        convergence_reason=null.convergence_reason,
        rank=null.result.terminal_rank.rank,
        curvature_telemetry=null.curvature_telemetry,
    )


def prepare_distributional_fit_state(
    family: DistributionalFamily,
    predictor_templates: Sequence[Predictor],
    compiled_predictors: Sequence[CompiledPredictor],
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    solver_result: DenseSolverResult,
    smoothing: DistributionalEFSResult | None,
    response: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    revision: int,
    retain_rows: bool,
    requested_discrete: bool,
    requested_n_bins: int | Mapping[str, int],
    requested_chunk_size: ChunkSize | None,
) -> DistributionalFitState:
    """Finish every terminal calculation before returning a publishable candidate."""
    if not isinstance(retain_rows, bool):
        raise TypeError("retain_rows must be bool")
    if solver_result.family_likelihood_plan_identifier != likelihood_plan.plan_identifier:
        raise ValueError("solver result must identify the supplied root likelihood plan")
    inference = compute_joint_inference(layout, solver_result)
    requested_solver_config = (
        solver_result.config if smoothing is None else smoothing.coefficient_fits[0].config
    )
    null = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=likelihood_plan,
        config=requested_solver_config,
    )
    compact_null = _compact_null(null)
    parameter_names = tuple(state.name for state in layout.predictors)
    predictor_coefficients = {
        state.name: solver_result.coefficients[state.coefficient_slice]
        for state in layout.predictors
    }
    smoothing_converged = None if smoothing is None else smoothing.converged
    n_inner_iter = (
        solver_result.iterations
        if smoothing is None
        else sum(item.iterations for item in smoothing.coefficient_fits)
    )
    face = solver_result.coefficient_face
    exact_face_components = () if face is None else face.component_names
    result = DistributionalFitResult(
        coefficients=solver_result.coefficients,
        coefficient_names=layout.coefficient_names,
        parameter_names=parameter_names,
        predictor_coefficients=predictor_coefficients,
        smoothing_parameters=lambdas,
        covariance=inference.covariance,
        total_effective_df=inference.total_edf,
        predictor_edf=inference.predictor_edf,
        intercept_edf=inference.intercept_edf,
        term_edf=inference.term_edf,
        log_likelihood=solver_result.log_likelihood,
        penalized_log_likelihood=solver_result.penalized_log_likelihood,
        null_objective=compact_null.objective,
        converged=solver_result.converged and smoothing_converged is not False,
        coefficient_converged=solver_result.converged,
        smoothing_converged=smoothing_converged,
        n_inner_iter=n_inner_iter,
        n_smoothing_iter=0 if smoothing is None else smoothing.iterations,
        rank=inference.rank,
        curvature_telemetry=solver_result.terminal_curvature,
        exact_face_components=exact_face_components,
    )
    rows = None
    if retain_rows:
        rows = DistributionalRowState(
            response=response,
            likelihood_weights=likelihood_plan.weights,
            offsets={state.name: state.offset for state in layout.predictors},
            fitted_eta=solver_result.eta,
            fitted_parameters=solver_result.theta,
            null_eta=null.result.eta,
            null_parameters=null.result.theta,
        )
    return DistributionalFitState(
        revision=revision,
        weight_contract=likelihood_plan.weights.provenance.contract,
        weight_provenance=likelihood_plan.weights.provenance,
        family_likelihood_plan_identifier=likelihood_plan.plan_identifier,
        requested_discrete=requested_discrete,
        requested_n_bins=requested_n_bins,
        requested_chunk_size=requested_chunk_size,
        predictor_templates=tuple(predictor_templates),
        compiled_predictors=tuple(compiled_predictors),
        layout=layout,
        lambdas=lambdas,
        solver_result=solver_result,
        smoothing=smoothing,
        inference=inference,
        null_model=compact_null,
        result=result,
        retained_rows=rows,
        exact_face_components=exact_face_components,
    )


__all__ = [
    "CompactNullModel",
    "DistributionalFitState",
    "DistributionalRowState",
    "prepare_distributional_fit_state",
]
