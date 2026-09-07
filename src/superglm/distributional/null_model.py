"""Joint intercept-only null fits using the distributional coefficient engine."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import PredictorState, StackedLayout
from superglm.distributional.result import (
    ConvergenceReason,
    DenseSolverConfig,
    DenseSolverResult,
)
from superglm.distributional.solver.solver import fit_dense_fixed_lambda
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    WeightProvenance,
)
from superglm.group_matrix import DesignMatrix
from superglm.links import Link


def _readonly(values: NDArray, *, name: str) -> NDArray[np.float64]:
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numeric values")
    result.setflags(write=False)
    return result


def _family_config(family: DistributionalFamily) -> Mapping[str, Any]:
    serializer = getattr(family, "to_config", None)
    if not callable(serializer):
        raise TypeError("distributional family must expose complete to_config() metadata")
    raw = serializer()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("family to_config() must return a non-empty mapping")
    config = copy.deepcopy(dict(raw))
    if any(not isinstance(key, str) or not key for key in config):
        raise ValueError("family configuration keys must be non-empty strings")
    return MappingProxyType(config)


def _null_layout(source: StackedLayout, n_observations: int) -> StackedLayout:
    predictors: list[PredictorState] = []
    coefficient_names: list[str] = []
    for index, state in enumerate(source.predictors):
        offset = _readonly(state.offset, name=f"offset for {state.name!r}")
        if offset.shape != (n_observations,):
            raise ValueError("source predictor offsets must match the response row count")
        predictors.append(
            PredictorState(
                name=state.name,
                parameter_index=index,
                link=state.link,
                design=DesignMatrix([], n_observations, 0),
                groups=(),
                coefficient_slice=slice(index, index + 1),
                intercept_index=index,
                offset=offset,
                penalties=(),
            )
        )
        coefficient_names.append(f"{state.name}:(intercept)")
    return StackedLayout(
        predictors=tuple(predictors),
        n_coefficients=len(predictors),
        coefficient_names=tuple(coefficient_names),
        term_slices=MappingProxyType({}),
        penalties=(),
    )


class NullModelFitError(RuntimeError):
    """Raised when a joint null coefficient fit reaches no converged state."""

    def __init__(self, result: DenseSolverResult):
        self.result = result
        super().__init__(
            f"joint distributional null model did not converge ({result.convergence_reason})"
        )


@dataclass(frozen=True)
class JointNullModel:
    """Immutable null-fit result plus the exact family/link/row semantics used."""

    family: DistributionalFamily
    family_config: Mapping[str, Any]
    layout: StackedLayout
    parameter_names: tuple[str, ...]
    parameter_links: Mapping[str, Link]
    offsets: Mapping[str, NDArray[np.float64]]
    likelihood_plan: FamilyLikelihoodPlan
    n_observations: int
    weight_sum: float
    objective: float
    converged: bool
    convergence_reason: ConvergenceReason
    curvature_telemetry: CurvatureTelemetry
    result: DenseSolverResult

    def __post_init__(self) -> None:
        if not isinstance(self.family, DistributionalFamily):
            raise TypeError("family must implement DistributionalFamily")
        if not isinstance(self.layout, StackedLayout):
            raise TypeError("layout must be a StackedLayout")
        if not isinstance(self.result, DenseSolverResult):
            raise TypeError("result must be a DenseSolverResult")
        if not isinstance(self.likelihood_plan, FamilyLikelihoodPlan):
            raise UnsupportedLikelihoodContractError(
                "likelihood_plan must implement FamilyLikelihoodPlan"
            )
        if self.result.family_likelihood_plan_identifier != self.likelihood_plan.plan_identifier:
            raise ValueError("null result must identify its root likelihood plan")
        parameter_names = tuple(self.parameter_names)
        family_names = tuple(parameter.name for parameter in self.family.parameters)
        layout_names = tuple(state.name for state in self.layout.predictors)
        if parameter_names != family_names or parameter_names != layout_names:
            raise ValueError("null family, parameter, and layout ordering must agree")
        if self.layout.n_coefficients != len(parameter_names):
            raise ValueError("null layout must contain exactly one coefficient per parameter")
        for index, state in enumerate(self.layout.predictors):
            if (
                state.coefficient_slice != slice(index, index + 1)
                or state.intercept_index != index
                or state.design.p != 0
                or state.groups
                or state.penalties
            ):
                raise ValueError("null layout must be intercept-only with no smooth penalties")
        if self.layout.term_slices or self.layout.penalties:
            raise ValueError("null layout cannot contain terms or penalties")

        if not isinstance(self.family_config, Mapping) or not self.family_config:
            raise ValueError("family_config must be a non-empty mapping")
        family_config = MappingProxyType(copy.deepcopy(dict(self.family_config)))
        if (
            not isinstance(self.parameter_links, Mapping)
            or tuple(self.parameter_links) != parameter_names
        ):
            raise ValueError("parameter_links must follow family parameter order")
        links = dict(self.parameter_links)
        if any(not isinstance(link, Link) for link in links.values()):
            raise TypeError("parameter_links values must implement Link")
        if any(links[state.name] is not state.link for state in self.layout.predictors):
            raise ValueError("parameter_links must identify the null layout links")

        if (
            isinstance(self.n_observations, bool)
            or not isinstance(self.n_observations, int)
            or self.n_observations < 1
        ):
            raise ValueError("n_observations must be a positive integer")
        weights = self.likelihood_plan.weights.values
        if weights.shape != (self.n_observations,) or np.any(weights <= 0.0):
            raise ValueError("sample_weight must be a strictly positive row vector")
        weight_sum = float(self.weight_sum)
        if not math.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("weight_sum must be finite and positive")
        if weight_sum != float(np.sum(weights, dtype=np.float64)):
            raise ValueError("weight_sum must equal the retained sample weights")
        if not isinstance(self.offsets, Mapping) or tuple(self.offsets) != parameter_names:
            raise ValueError("offsets must follow family parameter order")
        offsets: dict[str, NDArray[np.float64]] = {}
        for state in self.layout.predictors:
            offset = _readonly(self.offsets[state.name], name=f"offset for {state.name!r}")
            if offset.shape != (self.n_observations,) or not np.array_equal(offset, state.offset):
                raise ValueError("retained offsets must match the null layout")
            offsets[state.name] = offset

        objective = float(self.objective)
        if not math.isfinite(objective) or objective != self.result.objective:
            raise ValueError("null objective must equal the safeguarded solver objective")
        if not isinstance(self.converged, bool) or self.converged != self.result.converged:
            raise ValueError("null convergence flag must match the solver result")
        if self.convergence_reason != self.result.convergence_reason:
            raise ValueError("null convergence reason must match the solver result")
        if self.curvature_telemetry is not self.result.terminal_curvature:
            raise ValueError("null curvature telemetry must be the terminal solver telemetry")
        if self.result.penalty.shape != (len(parameter_names), len(parameter_names)) or np.any(
            self.result.penalty != 0.0
        ):
            raise ValueError("null result must have an exactly zero penalty")

        object.__setattr__(self, "family_config", family_config)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "parameter_links", MappingProxyType(links))
        object.__setattr__(self, "offsets", MappingProxyType(offsets))
        object.__setattr__(self, "weight_sum", weight_sum)
        object.__setattr__(self, "objective", objective)

    @property
    def sample_weight(self) -> NDArray[np.float64]:
        """Resolved weights derived from the executable likelihood plan."""
        return self.likelihood_plan.weights.values

    @property
    def weight_contract(self) -> WeightContract:
        return self.likelihood_plan.weights.provenance.contract

    @property
    def weight_provenance(self) -> WeightProvenance:
        return self.likelihood_plan.weights.provenance

    @property
    def weight_semantics(self) -> str:
        return self.weight_contract.semantics


def fit_joint_null_model(
    family: DistributionalFamily,
    source_layout: StackedLayout,
    y: NDArray,
    *,
    likelihood_plan: FamilyLikelihoodPlan,
    config: DenseSolverConfig | None = None,
) -> JointNullModel:
    """Fit one intercept per family parameter under the already-bound likelihood."""
    if not isinstance(family, DistributionalFamily):
        raise TypeError("family must implement DistributionalFamily")
    if not isinstance(source_layout, StackedLayout):
        raise TypeError("source_layout must be a StackedLayout")
    family_names = tuple(parameter.name for parameter in family.parameters)
    source_names = tuple(state.name for state in source_layout.predictors)
    if source_names != family_names:
        raise ValueError("source layout predictor order must match family parameters")
    try:
        response = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("y must be a non-empty finite vector") from exc
    if response.ndim != 1 or len(response) == 0 or not np.all(np.isfinite(response)):
        raise ValueError("y must be a non-empty finite vector")
    if not isinstance(likelihood_plan, FamilyLikelihoodPlan):
        raise UnsupportedLikelihoodContractError(
            "likelihood_plan must implement FamilyLikelihoodPlan"
        )
    weights = likelihood_plan.weights.values
    if weights.shape != response.shape:
        raise UnsupportedLikelihoodContractError("likelihood plan must match the response rows")
    solver_config = DenseSolverConfig() if config is None else config
    if not isinstance(solver_config, DenseSolverConfig):
        raise TypeError("config must be a DenseSolverConfig")

    layout = _null_layout(source_layout, len(response))
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients), dtype=np.float64)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        likelihood_plan,
        penalty,
        config=solver_config,
    )
    if not result.converged:
        raise NullModelFitError(result)
    links = {state.name: state.link for state in layout.predictors}
    offsets = {state.name: state.offset for state in layout.predictors}
    return JointNullModel(
        family=family,
        family_config=_family_config(family),
        layout=layout,
        parameter_names=family_names,
        parameter_links=links,
        offsets=offsets,
        likelihood_plan=likelihood_plan,
        n_observations=len(response),
        weight_sum=float(np.sum(weights, dtype=np.float64)),
        objective=result.objective,
        converged=result.converged,
        convergence_reason=result.convergence_reason,
        curvature_telemetry=result.terminal_curvature,
        result=result,
    )


__all__ = ["JointNullModel", "NullModelFitError", "fit_joint_null_model"]
