"""Two-parameter log-normal family, mean-parametrised by default."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.families._base import (
    array_digest,
    immutable,
    readonly,
    typed_plan,
    validated_float_response,
)
from superglm.distributional.families.gaussian import LowerBoundedLogLink
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
from superglm.distributional.kernels.log_normal import (
    LogNormalDomainError,
    initialize_log_normal,
    location_expected_information,
    location_of_mean,
    location_rows,
    log_normal_cdf,
    log_normal_expected_shortfall,
    log_normal_expected_shortfall_from_mean,
    log_normal_quantile,
    mean_expected_information,
    mean_of_location,
    mean_rows,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import IdentityLink, LogLink

Parametrisation = Literal["mean", "location"]
LogNormalRowLaw = Literal[
    "unit-prior-explicitly-equals-frequency/v1",
    "log-normal-literal-replication/v1",
]
_FAMILY_NAME = "LogNormalLS"
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_CAPABILITIES = FamilyCapabilities(
    max_derivative_order=2,
    expected_information=True,
    cdf=True,
    quantile=True,
    random=False,
    response_mean=True,
)


def _validated_response(y: NDArray) -> NDArray[np.float64]:
    return validated_float_response(
        y,
        message="y must be a non-empty finite strictly positive vector",
        lower=0.0,
        lower_inclusive=False,
    )


def _validated_scale_floor(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError("scale_floor must be a finite non-negative number")
    return float(value)


@dataclass(frozen=True)
class LogNormalLikelihoodPlan:
    """One exact-response log-normal law and immutable carriers."""

    weights: ResolvedLikelihoodWeights
    row_law: LogNormalRowLaw
    family_config: tuple[str, str, str]
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
        row_law: LogNormalRowLaw,
        family_config: tuple[str, str, str],
        observation: ObservationContract,
        exact_response: NDArray,
        carrier: NDArray,
    ) -> LogNormalLikelihoodPlan:
        response = immutable(exact_response)
        immutable_carrier = immutable(carrier)
        return cls(
            weights=weights,
            row_law=row_law,
            family_config=family_config,
            observation=observation,
            exact_response=response,
            response_digest=array_digest(b"LogNormalLS/response/v1\0", response),
            parameter_independent_carrier=immutable_carrier,
            carrier_digest=array_digest(b"LogNormalLS/carrier/v1\0", immutable_carrier),
        )

    @property
    def plan_identifier(self) -> str:
        payload = "\0".join(
            (
                "LogNormalLS/v1",
                self.row_law,
                *self.family_config,
                self.observation.kind,
                str(self.observation.schema_version),
                self.weights.digest,
                self.response_digest,
                self.carrier_digest,
            )
        ).encode("utf-8")
        return f"LogNormalLS/v1:{hashlib.sha256(payload).hexdigest()}"

    def take(self, indices: NDArray[np.integer]) -> LogNormalLikelihoodPlan:
        return LogNormalLikelihoodPlan._from_prepared(
            weights=self.weights.take(indices),
            row_law=self.row_law,
            family_config=self.family_config,
            observation=self.observation,
            exact_response=self.exact_response[indices],
            carrier=self.parameter_independent_carrier[indices],
        )

    @property
    def multiplier(self) -> NDArray[np.float64]:
        """The replication mass per row: the weights under frequency, ones under unit prior."""
        if self.row_law == "log-normal-literal-replication/v1":
            return self.weights.values
        return np.ones_like(self.weights.values)


@dataclass(frozen=True)
class LogNormalLS:
    """Log-normal with natural parameters ``(mean | location, scale)``.

    ``log Y ~ N(mu, sigma^2)`` on ``y > 0``.  The default mean form puts
    ``E[Y]`` first under a log link, so its relativities multiply the mean and
    the scale predictor only redistributes mass within a cell; the location
    form puts ``mu`` first under an identity link, where relativities multiply
    every quantile.  The mean always exists, so neither form has an invalid
    region.
    """

    parametrisation: Parametrisation = "mean"
    scale_floor: float = 0.01

    def __post_init__(self) -> None:
        if self.parametrisation not in ("mean", "location"):
            raise ValueError("parametrisation must be 'mean' or 'location'")
        object.__setattr__(self, "scale_floor", _validated_scale_floor(self.scale_floor))

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        first = (
            ParameterSpec("mean", LogLink(), "mean", ParameterSupport(lower=0.0), "observed")
            if self.parametrisation == "mean"
            else ParameterSpec(
                "location", IdentityLink(), "location", ParameterSupport(), "observed"
            )
        )
        return (
            first,
            ParameterSpec(
                "scale",
                LowerBoundedLogLink(self.scale_floor),
                "scale",
                ParameterSupport(lower=self.scale_floor),
                "observed",
            ),
        )

    @property
    def default_prediction_name(self) -> str:
        return "conditional_mean"

    @property
    def capabilities(self) -> FamilyCapabilities:
        return _CAPABILITIES

    def to_config(self) -> dict[str, Any]:
        return {
            "type": _FAMILY_NAME,
            "parametrisation": self.parametrisation,
            "scale_floor": self.scale_floor,
        }

    @property
    def _family_config(self) -> tuple[str, str, str]:
        return (
            "LogNormalLS/v1",
            f"log-normal-{self.parametrisation}/v1",
            f"scale_floor={self.scale_floor!r}",
        )

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> LogNormalLikelihoodPlan:
        response = _validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "LogNormalLS supports only the complete-observation contract"
            )
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "LogNormalLS requires resolved likelihood weights"
            )
        if len(weights.values) != len(response):
            raise UnsupportedLikelihoodContractError(
                "log-normal response rows do not match resolved likelihood-weight rows"
            )
        semantics = weights.provenance.contract.semantics
        if semantics == "prior":
            if not weights.provenance.all_unit:
                raise UnsupportedLikelihoodContractError(
                    "LogNormalLS is not a reproductive family, so non-unit prior weights "
                    "have no likelihood law; fit claim-level rows, carry exposure through "
                    "offsets, or use weight_semantics='frequency' for integer replication"
                )
            row_law: LogNormalRowLaw = "unit-prior-explicitly-equals-frequency/v1"
            multiplier = np.ones_like(weights.values)
        elif semantics == "frequency":
            row_law = "log-normal-literal-replication/v1"
            multiplier = weights.values
        else:  # pragma: no cover - resolved certification owns this branch
            raise UnsupportedLikelihoodContractError(
                "LogNormalLS does not support the resolved likelihood-weight contract"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            carrier = -multiplier * (np.log(response) + _HALF_LOG_TWO_PI)
        if not np.all(np.isfinite(carrier)):
            raise UnsupportedLikelihoodContractError(
                "log-normal response carrier is not finite and representable"
            )
        return LogNormalLikelihoodPlan._from_prepared(
            weights=weights,
            row_law=row_law,
            family_config=self._family_config,
            observation=observation,
            exact_response=response,
            carrier=carrier,
        )

    def _plan(self, plan: FamilyLikelihoodPlan, n_observations: int) -> LogNormalLikelihoodPlan:
        candidate = typed_plan(
            plan, LogNormalLikelihoodPlan, n_observations, family_name=_FAMILY_NAME
        )
        if candidate.family_config != self._family_config:
            raise UnsupportedLikelihoodContractError(
                "LogNormalLS likelihood was prepared for another configuration"
            )
        return candidate

    def validate_likelihood_plan(
        self, y: NDArray, plan: FamilyLikelihoodPlan
    ) -> NDArray[np.float64]:
        response = _validated_response(y)
        candidate = self._plan(plan, len(response))
        if candidate.observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "LogNormalLS likelihood has an unsupported observation contract"
            )
        bound = candidate.exact_response
        if (
            bound.dtype != np.float64
            or bound.shape != response.shape
            or bound.flags.writeable
            or not np.all(np.isfinite(bound))
            or np.any(bound <= 0.0)
        ):
            raise UnsupportedLikelihoodContractError(
                "LogNormalLS likelihood owns an invalid bound response"
            )
        if response.tobytes(order="C") != bound.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "LogNormalLS likelihood response does not match the fitted response"
            )
        return bound

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = _validated_response(y)
        bound = self._plan(plan, len(response))
        theta = initialize_log_normal(
            response,
            bound.multiplier,
            parametrisation=self.parametrisation,
            scale_floor=self.scale_floor,
        )
        return InitialParameterState(theta=theta)

    def _theta(self, theta: NDArray, n_observations: int | None) -> NDArray[np.float64]:
        return _validated_parameter_matrix(
            theta,
            n_observations=n_observations,
            parameters=self.parameters,
            family_name=_FAMILY_NAME,
        )

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
        values = self._theta(theta, len(response))
        bound = self._plan(plan, len(response))
        rows = mean_rows if self.parametrisation == "mean" else location_rows
        evaluated = rows(
            response, values[:, 0], values[:, 1], bound.multiplier, derivative_order=order
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluated.optimizing_log_likelihood,
            parameter_independent_carrier=bound.parameter_independent_carrier,
            score=evaluated.score,
            hessian_packed=evaluated.hessian_packed,
            valid=evaluated.valid,
        )

    def expected_information_natural(
        self, theta: NDArray, plan: FamilyLikelihoodPlan
    ) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        bound = self._plan(plan, len(values))
        if self.parametrisation == "mean":
            return mean_expected_information(values[:, 0], values[:, 1], bound.multiplier)
        return location_expected_information(values[:, 1], bound.multiplier)

    def _location_coordinates(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.parametrisation == "mean":
            return location_of_mean(values[:, 0], values[:, 1])
        return values[:, 0]

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        if self.parametrisation == "mean":
            return readonly(values[:, 0])
        return mean_of_location(values[:, 0], values[:, 1])

    def variance(self, theta: NDArray) -> NDArray[np.float64]:
        """``Var(Y) = E[Y]^2 (exp(sigma^2) - 1)`` per row, in either parametrisation.

        There is no prior-weighted companion: the log-normal is not a
        reproductive family, so this family refuses non-unit prior weights at
        the fit and has no weighted law to report a second moment from.
        """
        values = self._theta(theta, None)
        mean = (
            values[:, 0]
            if self.parametrisation == "mean"
            else mean_of_location(values[:, 0], values[:, 1])
        )
        scale = values[:, 1]
        return readonly(np.asarray(mean, dtype=np.float64) ** 2 * np.expm1(scale * scale))

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        if not np.all(np.isfinite(response)):
            raise LogNormalDomainError("log-normal y must be a finite non-empty vector")
        out = np.zeros_like(response)
        interior = response > 0.0
        if np.any(interior):
            inside_values = values[interior]
            out[interior] = log_normal_cdf(
                response[interior],
                self._location_coordinates(inside_values),
                inside_values[:, 1],
            )
        return readonly(out)

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        return log_normal_quantile(probabilities, self._location_coordinates(values), values[:, 1])

    def expected_shortfall(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """``E[Y | Y > q_p]`` per row in either natural parametrisation."""
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        if self.parametrisation == "mean":
            return log_normal_expected_shortfall_from_mean(
                probabilities, values[:, 0], values[:, 1]
            )
        return log_normal_expected_shortfall(
            probabilities, self._location_coordinates(values), values[:, 1]
        )
