"""Gaussian location-scale family with a configured standard-deviation floor."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
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
from superglm.distributional.kernels.gaussian import (
    evaluate_gaussian_rows,
    gaussian_expected_information,
    gaussian_predictor_curvature_directional,
    initialize_gaussian,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import IdentityLink, Link


def _validate_scale_floor(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("scale_floor must be finite and non-negative")
    try:
        floor = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scale_floor must be finite and non-negative") from exc
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("scale_floor must be finite and non-negative")
    return floor


def _carrier_digest(values: NDArray[np.float64]) -> str:
    return array_digest(b"GaussianLS/carrier/v1\0", values)


@dataclass(frozen=True)
class LowerBoundedLogLink:
    """Shifted log link: ``eta = log(value - floor)``."""

    floor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "floor", _validate_scale_floor(self.floor))

    def link(self, mu: NDArray) -> NDArray:
        values = np.asarray(mu, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= self.floor):
            raise ValueError("link input must be finite and strictly above the configured floor")
        return np.log(values - self.floor)

    def inverse(self, eta: NDArray) -> NDArray:
        with np.errstate(over="ignore", invalid="ignore"):
            return self.floor + np.exp(np.asarray(eta, dtype=np.float64))

    def deriv(self, mu: NDArray) -> NDArray:
        values = np.asarray(mu, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= self.floor):
            raise ValueError(
                "derivative input must be finite and strictly above the configured floor"
            )
        return 1.0 / (values - self.floor)

    def deriv_inverse(self, eta: NDArray) -> NDArray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(np.asarray(eta, dtype=np.float64))

    def deriv2_inverse(self, eta: NDArray) -> NDArray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(np.asarray(eta, dtype=np.float64))

    def deriv3_inverse(self, eta: NDArray) -> NDArray:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(np.asarray(eta, dtype=np.float64))


def _prior_weight_vector(weights: NDArray, n_observations: int) -> NDArray[np.float64]:
    """Per-row prior weights broadcast to the parameter rows of one call."""
    values = np.broadcast_to(np.asarray(weights, dtype=np.float64), (n_observations,))
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("prior weights must be finite and strictly positive")
    return values


def _normal_expected_shortfall(
    probabilities: NDArray[np.float64],
    location: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise ValueError("expected-shortfall probabilities must lie strictly inside (0, 1)")
    z = special.ndtri(probabilities)
    log_survival = np.log1p(-probabilities)
    standardised = np.exp(-0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - log_survival)
    return readonly(location + scale * standardised)


_CAPABILITIES = FamilyCapabilities(
    max_derivative_order=2,
    expected_information=True,
    cdf=True,
    quantile=True,
    random=False,
    response_mean=True,
)


GaussianRowLaw = Literal[
    "normal-variance-sigma2-over-w/v1",
    "normal-literal-replication/v1",
]
GaussianInvariant = Literal["conditional-location", "literal-row-replication"]


@dataclass(frozen=True)
class GaussianLikelihoodPlan:
    """One bound Gaussian likelihood law and its positional weight carrier."""

    weights: ResolvedLikelihoodWeights
    row_law: GaussianRowLaw
    invariant: GaussianInvariant
    family_config: tuple[str, float]
    observation: ObservationContract
    parameter_independent_carrier: NDArray[np.float64] = field(init=False, repr=False)
    carrier_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_static_contract()
        carrier = (
            0.5 * np.log(self.weights.values)
            if self.weights.provenance.contract.semantics == "prior"
            else np.zeros(len(self.weights.values), dtype=np.float64)
        )
        self._set_prepared_carrier(carrier)

    def _validate_static_contract(self) -> None:
        if not isinstance(self.weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "Gaussian likelihood plan requires resolved likelihood weights"
            )
        semantics = self.weights.provenance.contract.semantics
        expected = (
            ("normal-variance-sigma2-over-w/v1", "conditional-location")
            if semantics == "prior"
            else ("normal-literal-replication/v1", "literal-row-replication")
        )
        if (self.row_law, self.invariant) != expected:
            raise UnsupportedLikelihoodContractError(
                "Gaussian likelihood plan disagrees with its resolved weight contract"
            )
        if (
            not isinstance(self.family_config, tuple)
            or len(self.family_config) != 2
            or self.family_config[0] != "GaussianLS/v1"
            or isinstance(self.family_config[1], bool)
            or not isinstance(self.family_config[1], int | float)
            or not math.isfinite(self.family_config[1])
            or self.family_config[1] < 0.0
        ):
            raise UnsupportedLikelihoodContractError(
                "Gaussian likelihood plan has an invalid family configuration"
            )
        if self.observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "Gaussian likelihood plan has an unsupported observation contract"
            )

    def _set_prepared_carrier(self, values: NDArray) -> None:
        carrier = immutable(values)
        semantics = self.weights.provenance.contract.semantics
        if carrier.shape != self.weights.values.shape or not np.all(np.isfinite(carrier)):
            raise UnsupportedLikelihoodContractError(
                "Gaussian likelihood carrier must be finite and match resolved weights"
            )
        if semantics == "frequency" and np.any(carrier != 0.0):
            raise UnsupportedLikelihoodContractError(
                "frequency Gaussian likelihood plans require a zero carrier"
            )
        object.__setattr__(self, "parameter_independent_carrier", carrier)
        object.__setattr__(self, "carrier_digest", _carrier_digest(carrier))

    @property
    def plan_identifier(self) -> str:
        payload = "\0".join(
            (
                "GaussianLS/v1",
                self.row_law,
                self.invariant,
                self.family_config[0],
                repr(self.family_config[1]),
                self.observation.kind,
                str(self.observation.schema_version),
                self.weights.digest,
                self.carrier_digest,
            )
        ).encode("utf-8")
        return f"GaussianLS/v1:{hashlib.sha256(payload).hexdigest()}"

    def take(self, indices: NDArray[np.integer]) -> GaussianLikelihoodPlan:
        return GaussianLikelihoodPlan(
            weights=self.weights.take(indices),
            row_law=self.row_law,
            invariant=self.invariant,
            family_config=self.family_config,
            observation=self.observation,
        )


def _validated_plan(
    plan: FamilyLikelihoodPlan, n_observations: int, scale_floor: float
) -> GaussianLikelihoodPlan:
    gaussian_plan = typed_plan(
        plan, GaussianLikelihoodPlan, n_observations, family_name="GaussianLS"
    )
    if gaussian_plan.family_config != ("GaussianLS/v1", scale_floor):
        raise UnsupportedLikelihoodContractError("Gaussian family configuration mismatch")
    return gaussian_plan


def _validated_response(y: NDArray) -> NDArray[np.float64]:
    return validated_float_response(y, message="y must be a non-empty finite vector")


@dataclass(frozen=True)
class GaussianLS:
    """Gaussian family parameterized by location and standard deviation."""

    scale_floor: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_floor", _validate_scale_floor(self.scale_floor))

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="location",
                default_link=IdentityLink(),
                role="location",
                support=ParameterSupport(),
                curvature="fisher",
            ),
            ParameterSpec(
                name="scale",
                default_link=LowerBoundedLogLink(self.scale_floor),
                role="scale",
                support=ParameterSupport(lower=self.scale_floor),
                curvature="fisher",
            ),
        )

    @property
    def default_prediction_name(self) -> str:
        return "conditional_mean"

    @property
    def capabilities(self) -> FamilyCapabilities:
        return _CAPABILITIES

    def to_config(self) -> dict[str, Any]:
        """Return JSON-safe complete family configuration."""
        return {"type": type(self).__name__, "scale_floor": self.scale_floor}

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> GaussianLikelihoodPlan:
        response = _validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "GaussianLS supports only the complete-observation contract"
            )
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "GaussianLS requires resolved likelihood weights"
            )
        if len(weights.values) != len(response):
            raise UnsupportedLikelihoodContractError(
                "Gaussian response rows do not match resolved likelihood-weight rows"
            )
        if weights.provenance.contract.semantics == "prior":
            return GaussianLikelihoodPlan(
                weights=weights,
                row_law="normal-variance-sigma2-over-w/v1",
                invariant="conditional-location",
                family_config=("GaussianLS/v1", self.scale_floor),
                observation=observation,
            )
        if weights.provenance.contract.semantics == "frequency":
            return GaussianLikelihoodPlan(
                weights=weights,
                row_law="normal-literal-replication/v1",
                invariant="literal-row-replication",
                family_config=("GaussianLS/v1", self.scale_floor),
                observation=observation,
            )
        raise UnsupportedLikelihoodContractError(
            "GaussianLS does not support the resolved likelihood-weight contract"
        )

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = _validated_response(y)
        gaussian_plan = _validated_plan(plan, len(response), self.scale_floor)
        theta = initialize_gaussian(
            response,
            gaussian_plan.weights.values,
            gaussian_plan.weights.provenance.contract.semantics,
            self.scale_floor,
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
            family_name="GaussianLS",
        )
        gaussian_plan = _validated_plan(plan, len(response), self.scale_floor)
        evaluated = evaluate_gaussian_rows(
            response,
            parameters[:, 0],
            parameters[:, 1],
            gaussian_plan.weights.values,
            gaussian_plan.weights.provenance.contract.semantics,
            derivative_order=order,
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluated.optimizing_log_likelihood,
            parameter_independent_carrier=gaussian_plan.parameter_independent_carrier,
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
            family_name="GaussianLS",
        )
        gaussian_plan = _validated_plan(plan, len(parameters), self.scale_floor)
        return gaussian_expected_information(
            parameters[:, 1],
            gaussian_plan.weights.values,
            gaussian_plan.weights.provenance.contract.semantics,
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
        if (
            len(link_tuple) != 2
            or type(link_tuple[0]) is not IdentityLink
            or type(link_tuple[1]) is not LowerBoundedLogLink
            or link_tuple[1].floor != self.scale_floor
        ):
            raise UnsupportedLikelihoodContractError(
                "Gaussian endpoint curvature derivatives require the built-in identity and "
                "configured lower-bounded log links"
            )
        gaussian_plan = _validated_plan(plan, len(response), self.scale_floor)
        return gaussian_predictor_curvature_directional(
            response,
            eta,
            eta_direction,
            gaussian_plan.weights.values,
            gaussian_plan.weights.provenance.contract.semantics,
            scale_floor=self.scale_floor,
        )

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="GaussianLS",
        )
        return readonly(parameters[:, 0])

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """``P(Y <= y)`` per row from ``(location, scale)``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        return readonly(special.ndtr((response - values[:, 0]) / values[:, 1]))

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """The ``p``-quantile per row from ``(location, scale)``, ``p`` inside ``(0, 1)``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
            raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
        return readonly(values[:, 0] + values[:, 1] * special.ndtri(probabilities))

    def expected_shortfall(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """``E[Y | Y > q_p]`` for the unit-weight Gaussian row law."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        return _normal_expected_shortfall(probabilities, values[:, 0], values[:, 1])

    def variance(self, theta: NDArray) -> NDArray[np.float64]:
        """``Var(Y) = sigma^2`` per row at unit prior weight."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        return readonly(values[:, 1] * values[:, 1])

    def variance_prior_weighted(self, theta: NDArray, weights: NDArray) -> NDArray[np.float64]:
        """``Var(Y) = sigma^2 / w`` per row: the prior weight scales the variance."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        law = _prior_weight_vector(weights, len(values))
        return readonly(values[:, 1] * values[:, 1] / law)

    def cdf_prior_weighted(
        self, y: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """``P(Y <= y)`` per row when a prior weight scales the variance to ``sigma^2 / w``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        scale = values[:, 1] / np.sqrt(_prior_weight_vector(weights, len(values)))
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        return readonly(special.ndtr((response - values[:, 0]) / scale))

    def quantile_prior_weighted(
        self, p: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """The prior-weighted ``p``-quantile per row, ``p`` inside ``(0, 1)``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        scale = values[:, 1] / np.sqrt(_prior_weight_vector(weights, len(values)))
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
            raise ValueError("quantile probabilities must lie strictly inside (0, 1)")
        return readonly(values[:, 0] + scale * special.ndtri(probabilities))

    def expected_shortfall_prior_weighted(
        self, p: NDArray, theta: NDArray, weights: NDArray
    ) -> NDArray[np.float64]:
        """``E[Y | Y > q_p]`` when the prior weight scales variance by ``1 / w``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GaussianLS"
        )
        scale = values[:, 1] / np.sqrt(_prior_weight_vector(weights, len(values)))
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        return _normal_expected_shortfall(probabilities, values[:, 0], scale)
