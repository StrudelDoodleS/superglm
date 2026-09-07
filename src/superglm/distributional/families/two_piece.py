"""Epsilon-skew two-piece families: log-normal on ``y > 0`` and normal on the line."""

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
from superglm.distributional.families._links import BoundedLogitLink
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
from superglm.distributional.kernels.two_piece import (
    TwoPieceDomainError as TwoPieceDomainError,
)
from superglm.distributional.kernels.two_piece import (
    initialize_two_piece,
    location_expected_information,
    location_of_mean,
    location_rows,
    mean_expected_information,
    mean_of_location,
    mean_rows,
    real_line_mean,
    two_piece_cdf,
    two_piece_quantile,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import IdentityLink, LogLink

Parametrisation = Literal["mean", "location"]
TwoPieceRowLaw = Literal[
    "unit-prior-explicitly-equals-frequency/v1",
    "two-piece-epsilon-skew-literal-replication/v1",
]
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_CAPABILITIES = FamilyCapabilities(
    max_derivative_order=2,
    expected_information=True,
    cdf=True,
    quantile=True,
    random=False,
    response_mean=True,
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


def _validated_skew_bound(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError("skew_bound must be a finite number strictly inside (0, 1)")
    return float(value)


@dataclass(frozen=True)
class TwoPieceLikelihoodPlan:
    """One exact-response two-piece law and immutable carriers, for either family."""

    weights: ResolvedLikelihoodWeights
    row_law: TwoPieceRowLaw
    family_name: str
    family_config: tuple[str, ...]
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
        row_law: TwoPieceRowLaw,
        family_name: str,
        family_config: tuple[str, ...],
        observation: ObservationContract,
        exact_response: NDArray,
        carrier: NDArray,
    ) -> TwoPieceLikelihoodPlan:
        response = immutable(exact_response)
        immutable_carrier = immutable(carrier)
        domain = f"{family_name}/response/v1\0".encode()
        carrier_domain = f"{family_name}/carrier/v1\0".encode()
        return cls(
            weights=weights,
            row_law=row_law,
            family_name=family_name,
            family_config=family_config,
            observation=observation,
            exact_response=response,
            response_digest=array_digest(domain, response),
            parameter_independent_carrier=immutable_carrier,
            carrier_digest=array_digest(carrier_domain, immutable_carrier),
        )

    @property
    def plan_identifier(self) -> str:
        payload = "\0".join(
            (
                f"{self.family_name}/v1",
                self.row_law,
                *self.family_config,
                self.observation.kind,
                str(self.observation.schema_version),
                self.weights.digest,
                self.response_digest,
                self.carrier_digest,
            )
        ).encode("utf-8")
        return f"{self.family_name}/v1:{hashlib.sha256(payload).hexdigest()}"

    def take(self, indices: NDArray[np.integer]) -> TwoPieceLikelihoodPlan:
        return TwoPieceLikelihoodPlan._from_prepared(
            weights=self.weights.take(indices),
            row_law=self.row_law,
            family_name=self.family_name,
            family_config=self.family_config,
            observation=self.observation,
            exact_response=self.exact_response[indices],
            carrier=self.parameter_independent_carrier[indices],
        )

    @property
    def multiplier(self) -> NDArray[np.float64]:
        """The replication mass per row: the weights under frequency, ones under unit prior."""
        if self.row_law == "two-piece-epsilon-skew-literal-replication/v1":
            return self.weights.values
        return np.ones_like(self.weights.values)


def _resolved_multiplier(
    family_name: str,
    weights: ResolvedLikelihoodWeights,
    n_rows: int,
) -> tuple[TwoPieceRowLaw, NDArray[np.float64]]:
    if not isinstance(weights, ResolvedLikelihoodWeights):
        raise UnsupportedLikelihoodContractError(
            f"{family_name} requires resolved likelihood weights"
        )
    if len(weights.values) != n_rows:
        raise UnsupportedLikelihoodContractError(
            f"{family_name} response rows do not match resolved likelihood-weight rows"
        )
    semantics = weights.provenance.contract.semantics
    if semantics == "prior":
        if not weights.provenance.all_unit:
            raise UnsupportedLikelihoodContractError(
                f"{family_name} is not a reproductive family, so non-unit prior weights have "
                "no likelihood law; fit claim-level rows, carry exposure through offsets, or "
                "use weight_semantics='frequency' for integer replication"
            )
        return "unit-prior-explicitly-equals-frequency/v1", np.ones_like(weights.values)
    if semantics == "frequency":
        return "two-piece-epsilon-skew-literal-replication/v1", weights.values
    raise UnsupportedLikelihoodContractError(  # pragma: no cover - resolution owns this branch
        f"{family_name} does not support the resolved likelihood-weight contract"
    )


def _scale_and_skew_specs(scale_floor: float, skew_bound: float) -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(
            "scale",
            LowerBoundedLogLink(scale_floor),
            "scale",
            ParameterSupport(lower=scale_floor),
            "observed",
        ),
        ParameterSpec(
            "skew",
            BoundedLogitLink(-skew_bound, skew_bound),
            "shape",
            ParameterSupport(lower=-skew_bound, upper=skew_bound),
            "observed",
        ),
    )


@dataclass(frozen=True)
class TwoPieceLogNormalLSS:
    """Two-piece log-normal with natural parameters ``(mean | location, scale, skew)``.

    ``log Y = mu + sigma W`` with ``W`` epsilon-skew two-piece standard normal;
    the right piece is the wide one, so a positive ``skew`` predictor means a
    heavier right tail on the log scale.  The default mean form puts ``E[Y]``
    first under a log link, so its relativities multiply the mean.
    """

    parametrisation: Parametrisation = "mean"
    scale_floor: float = 0.01
    skew_bound: float = 0.9

    def __post_init__(self) -> None:
        if self.parametrisation not in ("mean", "location"):
            raise ValueError("parametrisation must be 'mean' or 'location'")
        object.__setattr__(self, "scale_floor", _validated_scale_floor(self.scale_floor))
        object.__setattr__(self, "skew_bound", _validated_skew_bound(self.skew_bound))

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        first = (
            ParameterSpec("mean", LogLink(), "mean", ParameterSupport(lower=0.0), "observed")
            if self.parametrisation == "mean"
            else ParameterSpec(
                "location", IdentityLink(), "location", ParameterSupport(), "observed"
            )
        )
        return (first, *_scale_and_skew_specs(self.scale_floor, self.skew_bound))

    @property
    def default_prediction_name(self) -> str:
        return "conditional_mean"

    @property
    def capabilities(self) -> FamilyCapabilities:
        return _CAPABILITIES

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "TwoPieceLogNormalLSS",
            "parametrisation": self.parametrisation,
            "scale_floor": self.scale_floor,
            "skew_bound": self.skew_bound,
        }

    @property
    def _family_config(self) -> tuple[str, ...]:
        return (
            "TwoPieceLogNormalLSS/v1",
            f"epsilon-skew-{self.parametrisation}/v1",
            f"scale_floor={self.scale_floor!r}",
            f"skew_bound={self.skew_bound!r}",
        )

    def _validated_response(self, y: NDArray) -> NDArray[np.float64]:
        return validated_float_response(
            y,
            message="y must be a non-empty finite strictly positive vector",
            lower=0.0,
            lower_inclusive=False,
        )

    def bind_likelihood(
        self, y: NDArray, weights: ResolvedLikelihoodWeights, observation: ObservationContract
    ) -> TwoPieceLikelihoodPlan:
        response = self._validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "TwoPieceLogNormalLSS supports only the complete-observation contract"
            )
        row_law, multiplier = _resolved_multiplier("TwoPieceLogNormalLSS", weights, len(response))
        with np.errstate(over="ignore", invalid="ignore"):
            carrier = -multiplier * (np.log(response) + _HALF_LOG_TWO_PI)
        if not np.all(np.isfinite(carrier)):
            raise UnsupportedLikelihoodContractError(
                "two-piece log-normal response carrier is not finite and representable"
            )
        return TwoPieceLikelihoodPlan._from_prepared(
            weights=weights,
            row_law=row_law,
            family_name="TwoPieceLogNormalLSS",
            family_config=self._family_config,
            observation=observation,
            exact_response=response,
            carrier=carrier,
        )

    def _plan(self, plan: FamilyLikelihoodPlan, n_observations: int) -> TwoPieceLikelihoodPlan:
        candidate = typed_plan(
            plan, TwoPieceLikelihoodPlan, n_observations, family_name="TwoPieceLogNormalLSS"
        )
        if candidate.family_config != self._family_config:
            raise UnsupportedLikelihoodContractError(
                "TwoPieceLogNormalLSS likelihood was prepared for another configuration"
            )
        return candidate

    def validate_likelihood_plan(
        self, y: NDArray, plan: FamilyLikelihoodPlan
    ) -> NDArray[np.float64]:
        response = self._validated_response(y)
        candidate = self._plan(plan, len(response))
        if candidate.observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "TwoPieceLogNormalLSS likelihood has an unsupported observation contract"
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
                "TwoPieceLogNormalLSS likelihood owns an invalid bound response"
            )
        if response.tobytes(order="C") != bound.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "TwoPieceLogNormalLSS likelihood response does not match the fitted response"
            )
        return bound

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = self._validated_response(y)
        bound = self._plan(plan, len(response))
        theta = initialize_two_piece(
            np.log(response),  # the location parameter lives on the log scale
            bound.multiplier,
            parametrisation=self.parametrisation,
            scale_floor=self.scale_floor,
            skew_bound=self.skew_bound,
        )
        return InitialParameterState(theta=theta)

    def _theta(self, theta: NDArray, n_observations: int | None) -> NDArray[np.float64]:
        return _validated_parameter_matrix(
            theta,
            n_observations=n_observations,
            parameters=self.parameters,
            family_name="TwoPieceLogNormalLSS",
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
        response = self._validated_response(y)
        values = self._theta(theta, len(response))
        bound = self._plan(plan, len(response))
        if self.parametrisation == "mean":
            evaluated = mean_rows(
                response,
                values[:, 0],
                values[:, 1],
                values[:, 2],
                bound.multiplier,
                derivative_order=order,
            )
        else:
            evaluated = location_rows(
                np.log(response),
                values[:, 0],
                values[:, 1],
                values[:, 2],
                bound.multiplier,
                derivative_order=order,
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
            return mean_expected_information(
                values[:, 0], values[:, 1], values[:, 2], bound.multiplier
            )
        return location_expected_information(values[:, 1], values[:, 2], bound.multiplier)

    def _location_coordinates(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.parametrisation == "mean":
            return location_of_mean(values[:, 0], values[:, 1], values[:, 2])
        return values[:, 0]

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        if self.parametrisation == "mean":
            return readonly(values[:, 0])
        return mean_of_location(values[:, 0], values[:, 1], values[:, 2])

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        if not np.all(np.isfinite(response)):
            raise TwoPieceDomainError("two-piece variate must be a non-empty finite vector")
        out = np.zeros_like(response)
        interior = response > 0.0
        if np.any(interior):
            inside_values = values[interior]
            out[interior] = two_piece_cdf(
                np.log(response[interior]),
                self._location_coordinates(inside_values),
                inside_values[:, 1],
                inside_values[:, 2],
            )
        return readonly(out)

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        variate = two_piece_quantile(
            probabilities, self._location_coordinates(values), values[:, 1], values[:, 2]
        )
        with np.errstate(over="ignore"):
            return readonly(np.exp(variate))


@dataclass(frozen=True)
class TwoPieceNormalLSS:
    """Epsilon-skew two-piece normal on the real line, ``(location, scale, skew)``.

    The same kernel as ``TwoPieceLogNormalLSS`` with the identity variate and
    no mean loading: ``E[Y] = location + 2 skew * scale * sqrt(2/pi)`` is a
    functional rather than a natural parameter, so this family has no mean
    form.
    """

    scale_floor: float = 0.01
    skew_bound: float = 0.9

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_floor", _validated_scale_floor(self.scale_floor))
        object.__setattr__(self, "skew_bound", _validated_skew_bound(self.skew_bound))

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("location", IdentityLink(), "location", ParameterSupport(), "observed"),
            *_scale_and_skew_specs(self.scale_floor, self.skew_bound),
        )

    @property
    def default_prediction_name(self) -> str:
        return "conditional_mean"

    @property
    def capabilities(self) -> FamilyCapabilities:
        return _CAPABILITIES

    def to_config(self) -> dict[str, Any]:
        return {
            "type": "TwoPieceNormalLSS",
            "scale_floor": self.scale_floor,
            "skew_bound": self.skew_bound,
        }

    @property
    def _family_config(self) -> tuple[str, ...]:
        return (
            "TwoPieceNormalLSS/v1",
            "epsilon-skew-location/v1",
            f"scale_floor={self.scale_floor!r}",
            f"skew_bound={self.skew_bound!r}",
        )

    def _validated_response(self, y: NDArray) -> NDArray[np.float64]:
        return validated_float_response(y, message="y must be a non-empty finite vector")

    def bind_likelihood(
        self, y: NDArray, weights: ResolvedLikelihoodWeights, observation: ObservationContract
    ) -> TwoPieceLikelihoodPlan:
        response = self._validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "TwoPieceNormalLSS supports only the complete-observation contract"
            )
        row_law, multiplier = _resolved_multiplier("TwoPieceNormalLSS", weights, len(response))
        carrier = -multiplier * _HALF_LOG_TWO_PI
        return TwoPieceLikelihoodPlan._from_prepared(
            weights=weights,
            row_law=row_law,
            family_name="TwoPieceNormalLSS",
            family_config=self._family_config,
            observation=observation,
            exact_response=response,
            carrier=carrier,
        )

    def _plan(self, plan: FamilyLikelihoodPlan, n_observations: int) -> TwoPieceLikelihoodPlan:
        candidate = typed_plan(
            plan, TwoPieceLikelihoodPlan, n_observations, family_name="TwoPieceNormalLSS"
        )
        if candidate.family_config != self._family_config:
            raise UnsupportedLikelihoodContractError(
                "TwoPieceNormalLSS likelihood was prepared for another configuration"
            )
        return candidate

    def validate_likelihood_plan(
        self, y: NDArray, plan: FamilyLikelihoodPlan
    ) -> NDArray[np.float64]:
        response = self._validated_response(y)
        candidate = self._plan(plan, len(response))
        if candidate.observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "TwoPieceNormalLSS likelihood has an unsupported observation contract"
            )
        bound = candidate.exact_response
        if (
            bound.dtype != np.float64
            or bound.shape != response.shape
            or bound.flags.writeable
            or not np.all(np.isfinite(bound))
        ):
            raise UnsupportedLikelihoodContractError(
                "TwoPieceNormalLSS likelihood owns an invalid bound response"
            )
        if response.tobytes(order="C") != bound.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "TwoPieceNormalLSS likelihood response does not match the fitted response"
            )
        return bound

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = self._validated_response(y)
        bound = self._plan(plan, len(response))
        theta = initialize_two_piece(
            response,
            bound.multiplier,
            parametrisation="location",
            scale_floor=self.scale_floor,
            skew_bound=self.skew_bound,
        )
        return InitialParameterState(theta=theta)

    def _theta(self, theta: NDArray, n_observations: int | None) -> NDArray[np.float64]:
        return _validated_parameter_matrix(
            theta,
            n_observations=n_observations,
            parameters=self.parameters,
            family_name="TwoPieceNormalLSS",
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
        response = self._validated_response(y)
        values = self._theta(theta, len(response))
        bound = self._plan(plan, len(response))
        evaluated = location_rows(
            response,
            values[:, 0],
            values[:, 1],
            values[:, 2],
            bound.multiplier,
            derivative_order=order,
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
        return location_expected_information(values[:, 1], values[:, 2], bound.multiplier)

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        return real_line_mean(values[:, 0], values[:, 1], values[:, 2])

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        return two_piece_cdf(response, values[:, 0], values[:, 1], values[:, 2])

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        return two_piece_quantile(probabilities, values[:, 0], values[:, 1], values[:, 2])


__all__ = [
    "TwoPieceDomainError",
    "TwoPieceLikelihoodPlan",
    "TwoPieceLogNormalLSS",
    "TwoPieceNormalLSS",
]
