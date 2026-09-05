"""Generalized Pareto family for threshold excesses, scale-parametrised.

The response is the excess ``y >= 0`` over a threshold the caller chooses; the
threshold is not a family argument.  The documented recipe is a body model below
``u``, ``P(Y > u)`` from a binary fit, and this family on ``y - u`` above.
"""

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
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    FamilyCapabilities,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
    _validated_derivative_order,
    _validated_parameter_matrix,
)
from superglm.distributional.kernels.generalized_pareto import (
    GeneralizedParetoDomainError as GeneralizedParetoDomainError,
)
from superglm.distributional.kernels.generalized_pareto import (
    expected_information,
    generalized_pareto_cdf,
    generalized_pareto_expected_shortfall,
    generalized_pareto_mean,
    generalized_pareto_quantile,
    initialize_generalized_pareto,
    scale_rows,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import LogLink

GeneralizedParetoRowLaw = Literal[
    "unit-prior-explicitly-equals-frequency/v1",
    "gpd-excess-literal-replication/v1",
]
_CAPABILITIES = FamilyCapabilities(
    max_derivative_order=2,
    expected_information=True,
    cdf=True,
    quantile=True,
    random=False,
    response_mean=True,
)


def _validated_wall(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite real number")
    wall = float(value)
    if not math.isfinite(wall):
        raise ValueError(f"{name} must be a finite real number")
    return wall


def _validated_response(y: NDArray) -> NDArray[np.float64]:
    return validated_float_response(
        y,
        message="y must be a non-empty finite non-negative vector of threshold excesses",
        lower=0.0,
        lower_inclusive=True,
    )


@dataclass(frozen=True)
class GeneralizedParetoLikelihoodPlan:
    """One exact-response generalized Pareto law and immutable carriers."""

    weights: ResolvedLikelihoodWeights
    row_law: GeneralizedParetoRowLaw
    family_config: tuple[str, str, str]
    observation: ObservationContract
    exact_response: NDArray[np.float64]
    response_digest: str
    parameter_independent_carrier: NDArray[np.float64]
    carrier_digest: str

    @classmethod
    def _from_prepared(
        cls, *, weights, row_law, family_config, observation, exact_response, carrier
    ):
        response = immutable(exact_response)
        immutable_carrier = immutable(carrier)
        return cls(
            weights=weights,
            row_law=row_law,
            family_config=family_config,
            observation=observation,
            exact_response=response,
            response_digest=array_digest(b"GeneralizedParetoLSS/response/v1\0", response),
            parameter_independent_carrier=immutable_carrier,
            carrier_digest=array_digest(b"GeneralizedParetoLSS/carrier/v1\0", immutable_carrier),
        )

    @property
    def plan_identifier(self) -> str:
        payload = "\0".join(
            (
                "GeneralizedParetoLSS/v1",
                self.row_law,
                *self.family_config,
                self.observation.kind,
                str(self.observation.schema_version),
                self.weights.digest,
                self.response_digest,
                self.carrier_digest,
            )
        ).encode("utf-8")
        return f"GeneralizedParetoLSS/v1:{hashlib.sha256(payload).hexdigest()}"

    def take(self, indices: NDArray[np.integer]) -> GeneralizedParetoLikelihoodPlan:
        return GeneralizedParetoLikelihoodPlan._from_prepared(
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
        if self.row_law == "gpd-excess-literal-replication/v1":
            return self.weights.values
        return np.ones_like(self.weights.values)


@dataclass(frozen=True)
class GeneralizedParetoLSS:
    """Generalized Pareto on excesses with natural parameters ``(scale, shape)``.

    The shape carries a two-wall logit.  This release enforces
    ``0 <= shape_lower < shape_upper <= 1``: a non-negative shape keeps the
    support ``[0, inf)`` for every row, and a shape below one keeps the mean
    finite.  A negative lower wall needs the response-dependent support slot,
    which this public family does not implement.
    """

    shape_lower: float = 0.0
    shape_upper: float = 1.0

    def __post_init__(self) -> None:
        lower = _validated_wall(self.shape_lower, name="shape_lower")
        upper = _validated_wall(self.shape_upper, name="shape_upper")
        if lower < 0.0:
            raise ValueError(
                "shape_lower must be non-negative in this release: a negative generalized Pareto "
                "shape makes the support depend on the fitted parameters, which needs the "
                "response-dependent support validation this family does not implement"
            )
        if not lower < upper:
            raise ValueError("shape walls must satisfy shape_lower < shape_upper")
        if upper > 1.0:
            raise ValueError(
                "shape_upper must not exceed one: the generalized Pareto mean, which is the "
                "default prediction, is infinite at shape one and above"
            )
        object.__setattr__(self, "shape_lower", lower)
        object.__setattr__(self, "shape_upper", upper)

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("scale", LogLink(), "scale", ParameterSupport(lower=0.0), "observed"),
            ParameterSpec(
                "shape",
                BoundedLogitLink(self.shape_lower, self.shape_upper),
                "shape",
                ParameterSupport(lower=self.shape_lower, upper=self.shape_upper),
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
            "type": "GeneralizedParetoLSS",
            "shape_lower": self.shape_lower,
            "shape_upper": self.shape_upper,
        }

    @property
    def _family_config(self) -> tuple[str, str, str]:
        return (
            "GeneralizedParetoLSS/v1",
            f"shape_lower={self.shape_lower!r}",
            f"shape_upper={self.shape_upper!r}",
        )

    def bind_likelihood(self, y, weights, observation) -> GeneralizedParetoLikelihoodPlan:
        response = _validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "GeneralizedParetoLSS supports only the complete-observation contract"
            )
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "GeneralizedParetoLSS requires resolved likelihood weights"
            )
        if len(weights.values) != len(response):
            raise UnsupportedLikelihoodContractError(
                "generalized Pareto response rows do not match resolved likelihood-weight rows"
            )
        semantics = weights.provenance.contract.semantics
        if semantics == "prior":
            if not weights.provenance.all_unit:
                raise UnsupportedLikelihoodContractError(
                    "GeneralizedParetoLSS is not a reproductive family, so non-unit prior weights "
                    "have no likelihood law; fit claim-level excess rows, carry exposure through "
                    "offsets, or use weight_semantics='frequency' for integer replication"
                )
            row_law: GeneralizedParetoRowLaw = "unit-prior-explicitly-equals-frequency/v1"
        elif semantics == "frequency":
            row_law = "gpd-excess-literal-replication/v1"
        else:  # pragma: no cover - resolved certification owns this branch
            raise UnsupportedLikelihoodContractError(
                "GeneralizedParetoLSS does not support the resolved likelihood-weight contract"
            )
        # the optimising log likelihood is the whole log density: no parameter-free carrier term
        carrier = np.zeros(len(response), dtype=np.float64)
        return GeneralizedParetoLikelihoodPlan._from_prepared(
            weights=weights,
            row_law=row_law,
            family_config=self._family_config,
            observation=observation,
            exact_response=response,
            carrier=carrier,
        )

    def _plan(self, plan, n_observations: int) -> GeneralizedParetoLikelihoodPlan:
        candidate = typed_plan(
            plan,
            GeneralizedParetoLikelihoodPlan,
            n_observations,
            family_name="GeneralizedParetoLSS",
        )
        if candidate.family_config != self._family_config:
            raise UnsupportedLikelihoodContractError(
                "GeneralizedParetoLSS likelihood was prepared for another configuration"
            )
        return candidate

    def validate_likelihood_plan(self, y, plan) -> NDArray[np.float64]:
        response = _validated_response(y)
        candidate = self._plan(plan, len(response))
        if candidate.observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError(
                "GeneralizedParetoLSS likelihood has an unsupported observation contract"
            )
        bound = candidate.exact_response
        if (
            bound.dtype != np.float64
            or bound.shape != response.shape
            or bound.flags.writeable
            or not np.all(np.isfinite(bound))
            or np.any(bound < 0.0)
        ):
            raise UnsupportedLikelihoodContractError(
                "GeneralizedParetoLSS likelihood owns an invalid bound response"
            )
        if response.tobytes(order="C") != bound.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "GeneralizedParetoLSS likelihood response does not match the fitted response"
            )
        return bound

    def initialize(self, y, plan) -> InitialParameterState:
        response = _validated_response(y)
        bound = self._plan(plan, len(response))
        theta = initialize_generalized_pareto(
            response,
            bound.multiplier,
            shape_lower=self.shape_lower,
            shape_upper=self.shape_upper,
        )
        return InitialParameterState(theta=theta)

    def _theta(self, theta, n_observations):
        return _validated_parameter_matrix(
            theta,
            n_observations=n_observations,
            parameters=self.parameters,
            family_name="GeneralizedParetoLSS",
        )

    def evaluate_natural(
        self, y, theta, plan, *, derivative_order: int = 2
    ) -> NaturalLikelihoodEvaluation:
        order = _validated_derivative_order(derivative_order)
        response = _validated_response(y)
        values = self._theta(theta, len(response))
        bound = self._plan(plan, len(response))
        evaluated = scale_rows(
            response, values[:, 0], values[:, 1], bound.multiplier, derivative_order=order
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluated.optimizing_log_likelihood,
            parameter_independent_carrier=bound.parameter_independent_carrier,
            score=evaluated.score,
            hessian_packed=evaluated.hessian_packed,
            valid=evaluated.valid,
        )

    def expected_information_natural(self, theta, plan) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        bound = self._plan(plan, len(values))
        return expected_information(values[:, 0], values[:, 1], bound.multiplier)

    def default_prediction(self, theta) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        return generalized_pareto_mean(values[:, 0], values[:, 1])

    def cdf(self, y, theta) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        return readonly(generalized_pareto_cdf(response, values[:, 0], values[:, 1]))

    def quantile(self, p, theta) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        return readonly(generalized_pareto_quantile(probabilities, values[:, 0], values[:, 1]))

    def expected_shortfall(self, p, theta) -> NDArray[np.float64]:
        """``E[Y | Y > q_p]`` for the generalized Pareto excess law."""
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        return readonly(
            generalized_pareto_expected_shortfall(probabilities, values[:, 0], values[:, 1])
        )


__all__ = ["GeneralizedParetoLikelihoodPlan", "GeneralizedParetoLSS"]
