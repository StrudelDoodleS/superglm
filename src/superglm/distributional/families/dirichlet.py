"""Joint Dirichlet regression with explicitly named composition components."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.families._base import array_digest, immutable
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    FamilyCapabilities,
    FamilyLikelihoodPlan,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
    _validated_parameter_matrix,
)
from superglm.distributional.kernels.dirichlet import (
    NORMALIZATION_POLICY,
    dirichlet_curvature_direction,
    evaluate_dirichlet_rows,
    normalized_mean,
    prepare_log_shares,
)
from superglm.distributional.response import (
    ResponseSpec,
    unvalidated_response,
    validated_response_array,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import Link, LogLink, _own_deriv4_inverse
from superglm.terms import TermInput

if TYPE_CHECKING:
    from superglm.distributional.binding import BoundPredictor


@dataclass(frozen=True)
class DirichletLikelihoodPlan:
    """Immutable response-only preparation, owned for this plan's lifetime.

    Identity is supplied canonical-order float64 response bytes, labels,
    normalization policy, weights and observation contract. Prepared log bytes
    are derived, not identity authority. Theta, links, basis and penalty changes
    need no invalidation; a new response/configuration/row take needs a new plan.
    """

    weights: ResolvedLikelihoodWeights
    components: tuple[str, ...]
    normalization_policy: str
    exact_response: np.ndarray
    log_shares: np.ndarray
    response_digest: str
    observation: ObservationContract = COMPLETE_OBSERVATION

    @classmethod
    def prepare(cls, weights, components, response, log_shares):
        response = immutable(response)
        return cls(
            weights,
            components,
            NORMALIZATION_POLICY,
            response,
            immutable(log_shares),
            array_digest(b"DirichletLSS/response/v1\0", response),
        )

    @property
    def plan_identifier(self) -> str:
        payload = json.dumps(
            (
                "DirichletLSS/v1",
                self.components,
                self.normalization_policy,
                self.exact_response.shape,
                self.response_digest,
                self.weights.digest,
                self.observation.kind,
                self.observation.schema_version,
            ),
            ensure_ascii=True,
        ).encode()
        return "DirichletLSS/v1:" + hashlib.sha256(payload).hexdigest()

    def take(self, indices: NDArray[np.integer]) -> DirichletLikelihoodPlan:
        weights = self.weights.take(indices)
        return self.prepare(
            weights, self.components, self.exact_response[indices], self.log_shares[indices]
        )


@dataclass(frozen=True)
class DirichletLSS:
    """Positive interior compositions with one positive alpha per named component.

    Rows within 8 gamma_(K+1) of unit mass denote the mathematical
    normalization of their supplied shares. Materially nonunit rows are
    rejected. Frequency weights repeat the entire joint density once per row;
    only unit prior weights are defined. Log-alpha effects change both mean
    composition and concentration.
    """

    components: tuple[str, ...]
    stacked_curvature_directions: ClassVar[bool] = True
    diagnoses_any_curvature_fallback: ClassVar[bool] = True

    def __post_init__(self):
        if not isinstance(self.components, tuple) or len(self.components) < 2:
            raise ValueError("components must be an ordered tuple of at least two unique labels")
        ResponseSpec((len(self.components),), self.components)

    @property
    def component_parameters(self) -> Mapping[str, str]:
        return MappingProxyType(
            {component: f"alpha_{j}" for j, component in enumerate(self.components)}
        )

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return tuple(
            ParameterSpec(f"alpha_{j}", LogLink(), component, ParameterSupport(lower=0.0), "fisher")
            for j, component in enumerate(self.components)
        )

    @property
    def response_spec(self) -> ResponseSpec:
        return ResponseSpec((len(self.components),), self.components)

    @property
    def default_prediction_name(self) -> str:
        return "mean_composition"

    @property
    def default_prediction_columns(self) -> tuple[str, ...]:
        return self.components

    @property
    def capabilities(self) -> FamilyCapabilities:
        return FamilyCapabilities(
            max_derivative_order=2,
            expected_information=True,
            cdf=False,
            quantile=False,
            random=False,
            response_mean=True,
        )

    def alpha(
        self,
        component: str,
        *terms: TermInput,
        intercept: bool = True,
        offset: Any = None,
        link: Literal[
            "log",
            "identity",
            "logit",
            "probit",
            "cloglog",
            "cauchit",
            "inverse",
            "inverse_squared",
            "sqrt",
        ]
        | Link
        | None = None,
    ) -> BoundPredictor:
        """Declare a component's concentration, including fixed link-scale offsets.

        ``offset`` accepts columns, row arrays, or lists of these sources.
        New-row mappings use ``component_parameters[component]`` and replace
        only the array-source sum; column contributions come from the new frame.
        """
        from superglm.distributional.binding import bind_predictor

        if not isinstance(component, str) or component not in self.component_parameters:
            raise ValueError(
                f"unknown Dirichlet component {component!r}; expected one of {self.components}"
            )
        return bind_predictor(
            self,
            self.component_parameters[component],
            *terms,
            intercept=intercept,
            link=link,
            offset=offset,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "DirichletLSS",
            "components": self.components,
            "normalization_policy": NORMALIZATION_POLICY,
        }

    def _response(self, y, n):
        raw = unvalidated_response(y, family=self, n_rows=n)
        return validated_response_array(raw, family=self, n_rows=n)

    def bind_likelihood(
        self, y: NDArray, weights: ResolvedLikelihoodWeights, observation: ObservationContract
    ) -> DirichletLikelihoodPlan:
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "DirichletLSS requires resolved likelihood weights"
            )
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "DirichletLSS requires complete vector observations"
            )
        if weights.provenance.contract.semantics == "prior" and np.any(weights.values != 1):
            raise UnsupportedLikelihoodContractError(
                "DirichletLSS has no nonunit prior-weight law; use frequency weights"
            )
        raw = unvalidated_response(y, family=self, n_rows=len(weights.values))
        try:
            response = self._response(raw, len(raw))
            logs = prepare_log_shares(response)
        except (ValueError, TypeError, OverflowError) as exc:
            for i in range(len(raw)):
                try:
                    prepare_log_shares(self._response(raw[i : i + 1], 1))
                except (ValueError, TypeError, OverflowError) as row_exc:
                    raise ValueError(
                        f"{row_exc} (first at input row {weights.input_positions[i]})"
                    ) from exc
            raise
        return DirichletLikelihoodPlan.prepare(weights, self.components, response, logs)

    def _plan(self, plan, n):
        if not isinstance(plan, DirichletLikelihoodPlan):
            raise UnsupportedLikelihoodContractError(
                "DirichletLSS requires its own bound likelihood plan"
            )
        if (
            plan.components != self.components
            or plan.normalization_policy != NORMALIZATION_POLICY
            or plan.observation != COMPLETE_OBSERVATION
        ):
            raise UnsupportedLikelihoodContractError(
                "Dirichlet component/configuration identity does not match the plan"
            )
        if plan.exact_response.shape != (n, len(self.components)) or plan.weights.values.shape != (
            n,
        ):
            raise UnsupportedLikelihoodContractError(
                "Dirichlet response shape does not match the plan"
            )
        return plan

    def validate_likelihood_plan(
        self, y: NDArray, plan: FamilyLikelihoodPlan
    ) -> NDArray[np.float64]:
        plan = self._plan(plan, len(y))
        response = self._response(y, len(plan.weights.values))
        if (
            not np.array_equal(response, plan.exact_response)
            or array_digest(b"DirichletLSS/response/v1\0", response) != plan.response_digest
        ):
            raise UnsupportedLikelihoodContractError(
                "Dirichlet response identity differs from the bound plan"
            )
        return plan.exact_response

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = self.validate_likelihood_plan(y, plan)
        # Validation above checks the concrete plan and its response identity.
        plan = cast(DirichletLikelihoodPlan, plan)
        composition = np.exp(plan.log_shares)
        weight = plan.weights.values / math.fsum(plan.weights.values)
        mean = np.array(
            [math.fsum(weight * composition[:, j]) for j in range(len(self.components))]
        )
        variance = np.array(
            [
                math.fsum(weight * (composition[:, j] - mean[j]) ** 2)
                for j in range(len(self.components))
            ]
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            candidates = mean * (1 - mean) / variance - 1
        candidates = candidates[(candidates > 0) & np.isfinite(candidates)]
        # A finite moment fallback is a start only, never a concentration wall.
        concentration = (
            float(np.median(candidates)) if len(candidates) else float(len(self.components))
        )
        if np.all(composition == composition[0]):
            concentration = float(len(self.components))
        theta = np.broadcast_to(mean * concentration, response.shape).copy()
        core = self.evaluate_natural(response, theta, plan)
        if not np.all(core.valid):
            raise ValueError(
                "Dirichlet moment start has nonrepresentable natural channels; supply a valid initial state"
            )
        return InitialParameterState(theta)

    def evaluate_natural(
        self, y: NDArray, theta: NDArray, plan: FamilyLikelihoodPlan, *, derivative_order: int = 2
    ) -> NaturalLikelihoodEvaluation:
        response = self.validate_likelihood_plan(y, plan)
        plan = cast(DirichletLikelihoodPlan, plan)
        parameters = np.asarray(theta, dtype=np.float64)
        result = evaluate_dirichlet_rows(
            response,
            parameters,
            plan.weights.values,
            derivative_order=derivative_order,
            log_shares=plan.log_shares,
        )
        return NaturalLikelihoodEvaluation(
            result.optimizing_log_likelihood,
            result.parameter_independent_carrier,
            result.score,
            result.hessian_packed,
            result.valid,
        )

    def expected_information_natural(
        self, theta: NDArray, plan: FamilyLikelihoodPlan
    ) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="DirichletLSS"
        )
        plan = self._plan(plan, len(parameters))
        # Fisher information is response-independent. A remote supplied row
        # can have an unrepresentable density at an otherwise admissible theta.
        mean = normalized_mean(parameters)
        result = evaluate_dirichlet_rows(mean, parameters, plan.weights.values, derivative_order=2)
        if not result.valid.all():
            raise UnsupportedLikelihoodContractError(
                "Dirichlet Fisher information has nonrepresentable natural channels"
            )
        return immutable(-result.hessian_packed)

    def _direction(self, y, eta, first, second, links, plan):
        response = self.validate_likelihood_plan(y, plan)
        predictors = np.asarray(eta, dtype=float)
        if (
            predictors.shape != response.shape
            or not np.isfinite(predictors).all()
            or len(links) != len(self.components)
        ):
            raise ValueError("Dirichlet predictors, links and response must have matching shape")
        first = np.asarray(first, dtype=float)
        single = first.ndim == 2
        first = first[None] if single else first
        if first.ndim != 3 or first.shape[1:] != response.shape or not np.isfinite(first).all():
            raise ValueError("Dirichlet directions must have shape (n, K) or (m, n, K)")
        if second is not None:
            second = np.asarray(second, dtype=float)
            second = second[None] if single else second
            if second.shape != first.shape or not np.isfinite(second).all():
                raise ValueError("Dirichlet second directions must match the first directions")
        derivatives = [np.empty_like(predictors) for _ in range(4)]
        theta = np.empty_like(predictors)
        for j, link in enumerate(links):
            methods = (
                getattr(link, "deriv_inverse", None),
                getattr(link, "deriv2_inverse", None),
                getattr(link, "deriv3_inverse", None),
                _own_deriv4_inverse(link),
            )
            if any(not callable(method) for method in methods[: 3 if second is None else 4]):
                raise UnsupportedLikelihoodContractError(
                    "Dirichlet optional curvature directions need inverse-link derivatives through order four"
                )
            theta[:, j] = link.inverse(predictors[:, j])
            for d, method in enumerate(methods):
                derivatives[d][:, j] = 0 if method is None else method(predictors[:, j])
        if not np.all(self.evaluate_natural(response, theta, plan).valid):
            raise UnsupportedLikelihoodContractError(
                "Dirichlet optional direction has an invalid core state"
            )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                result = np.stack(
                    [
                        dirichlet_curvature_direction(
                            response,
                            theta,
                            plan.weights.values,
                            derivatives,
                            a,
                            None if second is None else second[i],
                            log_shares=plan.log_shares,
                        )
                        for i, a in enumerate(first)
                    ]
                )
        except (ValueError, OverflowError, FloatingPointError) as exc:
            raise UnsupportedLikelihoodContractError(
                f"Dirichlet optional curvature direction unavailable: {exc}"
            ) from exc
        return immutable(result[0] if single else result)

    def predictor_curvature_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        eta_direction: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        return self._direction(y, eta, eta_direction, None, links, plan)

    def predictor_curvature_second_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        direction_a: NDArray,
        direction_b: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        return self._direction(y, eta, direction_a, direction_b, links, plan)

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="DirichletLSS"
        )
        return normalized_mean(parameters)

    def response_covariance(self, theta: NDArray) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="DirichletLSS"
        )
        mean = normalized_mean(parameters)
        covariance = np.empty((len(mean), len(self.components), len(self.components)))
        for i, (a, mu) in enumerate(zip(parameters, mean, strict=True)):
            total = math.fsum(a)
            inverse = (1.0 / total) / (1 + 1.0 / total) if total > 1 else 1.0 / (total + 1)
            covariance[i] = -(mu[:, None] * mu[None, :]) * inverse
            for j in range(len(mu)):
                covariance[i, j, j] = mu[j] * (math.fsum(np.delete(a, j)) / total) * inverse
        return immutable(covariance)

    def diagnose_repeated_curvature_failure(self, y, weights):
        retained = self._response(np.asarray(y)[weights.input_positions], len(weights.values))
        logs = prepare_log_shares(retained)
        if np.all(logs == logs[0]):
            return ValueError(
                "Constant Dirichlet compositions have an unbounded concentration likelihood; the accepted finite state is not a finite-concentration MLE"
            )
        return None
