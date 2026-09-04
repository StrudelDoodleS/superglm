"""Normalized Gamma family parameterized by mean and coefficient of variation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm.distributional.families._base import (
    array_digest,
    immutable,
    readonly,
    typed_plan,
    validated_float_response,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    FamilyCapabilities,
    FamilyLikelihoodPlan,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
    _validated_derivative_order,
    _validated_parameter_matrix,
)
from superglm.distributional.kernels.gamma import (
    GammaInitializationError as GammaInitializationError,
)
from superglm.distributional.kernels.gamma import (
    evaluate_gamma_rows,
    gamma_expected_information,
    gamma_expected_shortfall,
    gamma_predictor_curvature_directional,
    initialize_gamma,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import Link, LogLink


def _prior_weight_vector(weights: NDArray, n_observations: int) -> NDArray[np.float64]:
    """Per-row prior weights broadcast to the parameter rows of one call."""
    values = np.broadcast_to(np.asarray(weights, dtype=np.float64), (n_observations,))
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("prior weights must be finite and strictly positive")
    return values


_CAPABILITIES = FamilyCapabilities(
    max_derivative_order=2,
    expected_information=True,
    cdf=True,
    quantile=True,
    random=False,
    response_mean=True,
)

GammaRowLaw = Literal[
    "gamma-mean-cv-prior-dispersion-over-w/v1",
    "gamma-mean-cv-literal-replication/v1",
]
GammaInvariant = Literal["conditional-mean", "literal-row-replication"]


def _response_digest(values: NDArray[np.float64]) -> str:
    return array_digest(b"GammaLS/response/v1\0", values)


def _carrier_digest(values: NDArray[np.float64]) -> str:
    return array_digest(b"GammaLS/carrier/v1\0", values)


def _validated_response(y: NDArray) -> NDArray[np.float64]:
    return validated_float_response(
        y,
        message="y must be a non-empty finite strictly positive vector",
        lower=0.0,
        lower_inclusive=False,
    )


@dataclass(frozen=True)
class GammaLikelihoodPlan:
    """One exact-response Gamma likelihood law and immutable row carriers."""

    weights: ResolvedLikelihoodWeights
    row_law: GammaRowLaw
    invariant: GammaInvariant
    family_config: tuple[str, str]
    observation: ObservationContract
    exact_response: NDArray[np.float64]
    response_digest: str
    parameter_independent_carrier: NDArray[np.float64]
    carrier_digest: str

    @classmethod
    def _from_prepared(
        cls,
        *,
        weights: ResolvedLikelihoodWeights,
        row_law: GammaRowLaw,
        invariant: GammaInvariant,
        family_config: tuple[str, str],
        observation: ObservationContract,
        exact_response: NDArray,
        carrier: NDArray,
    ) -> GammaLikelihoodPlan:
        response = immutable(exact_response)
        immutable_carrier = immutable(carrier)
        return cls(
            weights=weights,
            row_law=row_law,
            invariant=invariant,
            family_config=family_config,
            observation=observation,
            exact_response=response,
            response_digest=_response_digest(response),
            parameter_independent_carrier=immutable_carrier,
            carrier_digest=_carrier_digest(immutable_carrier),
        )

    @property
    def plan_identifier(self) -> str:
        payload = "\0".join(
            (
                "GammaLS/v1",
                self.row_law,
                self.invariant,
                *self.family_config,
                self.observation.kind,
                str(self.observation.schema_version),
                self.weights.digest,
                self.response_digest,
                self.carrier_digest,
            )
        ).encode("utf-8")
        return f"GammaLS/v1:{hashlib.sha256(payload).hexdigest()}"

    def take(self, indices: NDArray[np.integer]) -> GammaLikelihoodPlan:
        return GammaLikelihoodPlan._from_prepared(
            weights=self.weights.take(indices),
            row_law=self.row_law,
            invariant=self.invariant,
            family_config=self.family_config,
            observation=self.observation,
            exact_response=self.exact_response[indices],
            carrier=self.parameter_independent_carrier[indices],
        )


def _validated_plan(
    plan: FamilyLikelihoodPlan,
    *,
    n_observations: int,
) -> GammaLikelihoodPlan:
    return typed_plan(plan, GammaLikelihoodPlan, n_observations, family_name="GammaLS")


@dataclass(frozen=True)
class GammaLS:
    """Gamma family with natural parameters mean and coefficient of variation."""

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("mean", LogLink(), "mean", ParameterSupport(lower=0.0), "fisher"),
            ParameterSpec("scale", LogLink(), "scale", ParameterSupport(lower=0.0), "fisher"),
        )

    @property
    def default_prediction_name(self) -> str:
        return "conditional_mean"

    @property
    def capabilities(self) -> FamilyCapabilities:
        return _CAPABILITIES

    def to_config(self) -> dict[str, Any]:
        return {"type": "GammaLS", "parameterization": "mean_cv"}

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> GammaLikelihoodPlan:
        response = _validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "GammaLS supports only the complete-observation contract"
            )
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError("GammaLS requires resolved likelihood weights")
        if len(weights.values) != len(response):
            raise UnsupportedLikelihoodContractError(
                "Gamma response rows do not match resolved likelihood-weight rows"
            )
        semantics = weights.provenance.contract.semantics
        with np.errstate(over="ignore", invalid="ignore"):
            carrier = -np.log(response)
            if semantics == "frequency":
                carrier = weights.values * carrier
        if not np.all(np.isfinite(carrier)):
            raise UnsupportedLikelihoodContractError(
                "Gamma response carrier is not finite and representable"
            )
        if semantics == "prior":
            row_law: GammaRowLaw = "gamma-mean-cv-prior-dispersion-over-w/v1"
            invariant: GammaInvariant = "conditional-mean"
        elif semantics == "frequency":
            row_law = "gamma-mean-cv-literal-replication/v1"
            invariant = "literal-row-replication"
        else:  # pragma: no cover - resolved certification owns this branch
            raise UnsupportedLikelihoodContractError(
                "GammaLS does not support the resolved likelihood-weight contract"
            )
        return GammaLikelihoodPlan._from_prepared(
            weights=weights,
            row_law=row_law,
            invariant=invariant,
            family_config=("GammaLS/v1", "mean-cv/v1"),
            observation=observation,
            exact_response=response,
            carrier=carrier,
        )

    def validate_likelihood_plan(
        self,
        y: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        response = _validated_response(y)
        candidate = _validated_plan(plan, n_observations=len(response))
        semantics = candidate.weights.provenance.contract.semantics
        expected_law = (
            ("gamma-mean-cv-prior-dispersion-over-w/v1", "conditional-mean")
            if semantics == "prior"
            else ("gamma-mean-cv-literal-replication/v1", "literal-row-replication")
        )
        if (candidate.row_law, candidate.invariant) != expected_law:
            raise UnsupportedLikelihoodContractError(
                "GammaLS likelihood law does not match its weight semantics"
            )
        if candidate.family_config != ("GammaLS/v1", "mean-cv/v1"):
            raise UnsupportedLikelihoodContractError(
                "GammaLS likelihood has an invalid family configuration"
            )
        if candidate.observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "GammaLS likelihood has an unsupported observation contract"
            )
        bound_response = candidate.exact_response
        if (
            not isinstance(bound_response, np.ndarray)
            or bound_response.dtype != np.float64
            or bound_response.shape != response.shape
            or bound_response.flags.writeable
            or not np.all(np.isfinite(bound_response))
            or np.any(bound_response <= 0.0)
        ):
            raise UnsupportedLikelihoodContractError(
                "GammaLS likelihood owns an invalid bound response"
            )
        if response.tobytes(order="C") != bound_response.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "GammaLS likelihood response does not match the fitted response"
            )
        return bound_response

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = _validated_response(y)
        gamma_plan = _validated_plan(plan, n_observations=len(response))
        theta = initialize_gamma(
            response,
            gamma_plan.weights.values,
            gamma_plan.weights.provenance.contract.semantics,
        )
        return InitialParameterState(theta=theta)

    def evaluate_natural(
        self,
        y: NDArray,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        order = _validated_derivative_order(derivative_order)
        response = _validated_response(y)
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=len(response),
            parameters=self.parameters,
            family_name="GammaLS",
        )
        gamma_plan = _validated_plan(plan, n_observations=len(response))
        evaluated = evaluate_gamma_rows(
            response,
            parameters[:, 0],
            parameters[:, 1],
            gamma_plan.weights.values,
            gamma_plan.weights.provenance.contract.semantics,
            derivative_order=order,
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluated.optimizing_log_likelihood,
            parameter_independent_carrier=gamma_plan.parameter_independent_carrier,
            score=evaluated.score,
            hessian_packed=evaluated.hessian_packed,
            valid=evaluated.valid,
        )

    def expected_information_natural(
        self,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="GammaLS",
        )
        gamma_plan = _validated_plan(plan, n_observations=len(parameters))
        return gamma_expected_information(
            parameters[:, 0],
            parameters[:, 1],
            gamma_plan.weights.values,
            gamma_plan.weights.provenance.contract.semantics,
        )

    def predictor_curvature_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        eta_direction: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        """Differentiate observed predictor curvature along one predictor path."""

        response = _validated_response(y)
        link_tuple = tuple(links)
        if len(link_tuple) != 2 or any(type(link) is not LogLink for link in link_tuple):
            raise UnsupportedLikelihoodContractError(
                "Gamma endpoint curvature derivatives require the built-in log links"
            )
        gamma_plan = _validated_plan(plan, n_observations=len(response))
        return gamma_predictor_curvature_directional(
            gamma_plan.exact_response,
            eta,
            eta_direction,
            gamma_plan.weights.values,
            gamma_plan.weights.provenance.contract.semantics,
        )

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="GammaLS",
        )
        return readonly(parameters[:, 0])

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """``P(Y <= y)`` per row: shape ``1/cv^2`` and scale ``mean cv^2``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        if np.any(~np.isfinite(response)):
            raise ValueError("GammaLS CDF thresholds must be finite")
        out = np.zeros_like(response)
        interior = response > 0.0
        if np.any(interior):
            inside_values = values[interior]
            cv2 = inside_values[:, 1] * inside_values[:, 1]
            out[interior] = special.gammainc(
                1.0 / cv2,
                response[interior] / (inside_values[:, 0] * cv2),
            )
        return readonly(out)

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """The ``p``-quantile per row, ``p`` inside ``(0, 1)``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
            raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
        cv2 = values[:, 1] * values[:, 1]
        return readonly(values[:, 0] * cv2 * special.gammaincinv(1.0 / cv2, probabilities))

    def expected_shortfall(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """``E[Y | Y > q_p]`` for shape ``1 / cv^2`` at unit prior weight."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        cv2 = values[:, 1] * values[:, 1]
        return gamma_expected_shortfall(probabilities, values[:, 0], 1.0 / cv2)

    def variance(self, theta: NDArray) -> NDArray[np.float64]:
        """``Var(Y) = (mean cv)^2`` per row at unit prior weight."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        spread = values[:, 0] * values[:, 1]
        return readonly(spread * spread)

    def variance_prior_weighted(self, theta: NDArray, weights: NDArray) -> NDArray[np.float64]:
        """``Var(Y) = (mean cv)^2 / w`` per row: shape ``w / cv^2``, scale ``mean cv^2 / w``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        prior = _prior_weight_vector(weights, len(values))
        spread = values[:, 0] * values[:, 1]
        return readonly(spread * spread / prior)

    def cdf_prior_weighted(
        self, y: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """``P(Y <= y)`` per row: shape ``w / cv^2`` and scale ``mean cv^2 / w``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        prior = _prior_weight_vector(weights, len(values))
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        if np.any(~np.isfinite(response)):
            raise ValueError("GammaLS CDF thresholds must be finite")
        out = np.zeros_like(response)
        interior = response > 0.0
        if np.any(interior):
            inside_values = values[interior]
            inside_prior = prior[interior]
            cv2 = inside_values[:, 1] * inside_values[:, 1]
            out[interior] = special.gammainc(
                inside_prior / cv2,
                inside_prior * response[interior] / (inside_values[:, 0] * cv2),
            )
        return readonly(out)

    def quantile_prior_weighted(
        self, p: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """The prior-weighted ``p``-quantile per row, ``p`` inside ``(0, 1)``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        prior = _prior_weight_vector(weights, len(values))
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
            raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
        cv2 = values[:, 1] * values[:, 1]
        return readonly(
            values[:, 0] * cv2 / prior * special.gammaincinv(prior / cv2, probabilities)
        )

    def expected_shortfall_prior_weighted(
        self, p: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """``E[Y | Y > q_p]`` for weighted shape ``w / cv^2``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        prior = _prior_weight_vector(weights, len(values))
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        cv2 = values[:, 1] * values[:, 1]
        return gamma_expected_shortfall(probabilities, values[:, 0], prior / cv2)
