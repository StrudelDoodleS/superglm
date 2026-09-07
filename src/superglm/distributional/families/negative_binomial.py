"""Exact normalized NB2 family in natural mean-size coordinates."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm._count_lattice import _not_a_whole_number
from superglm.distributional.families._base import (
    array_digest,
    immutable,
    response_row_count,
    typed_plan,
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
from superglm.distributional.kernels.negative_binomial import (
    NegativeBinomialDerivativeRepresentationError,
    NegativeBinomialPoissonBoundaryError,
    evaluate_negative_binomial_rows,
    has_resolved_poisson_boundary,
    initialize_negative_binomial,
)
from superglm.distributional.kernels.negative_binomial import (
    NegativeBinomialInitializationError as NegativeBinomialInitializationError,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import Link, LogLink

_FLOAT = np.float64
_MAX_EXACT_COUNT = 2**53
_FAMILY_CONFIGURATION = ("NegativeBinomialLS/v1", "nb2-mean-theta/v1")


def _plan_identifier(weights, response, count, carrier) -> str:
    law = (
        ("nb2-mean-theta-prior-scaled-count/v1", "conditional-mean")
        if weights.provenance.contract.semantics == "prior"
        else ("nb2-mean-theta-literal-replication/v1", "literal-row-replication")
    )
    identity = ("NegativeBinomialLS/v1", *law, *_FAMILY_CONFIGURATION, "complete", "1")
    digests = (
        array_digest(f"NegativeBinomialLS/{name}/v1\0".encode(), values)
        for name, values in zip(("response", "count", "carrier"), (response, count, carrier))
    )
    payload = "\0".join((*identity, weights.digest, *digests)).encode()
    return f"NegativeBinomialLS/v1:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class NegativeBinomialLikelihoodPlan:
    """Prepared NB2 rows sharing one resolved likelihood-weight carrier."""

    weights: ResolvedLikelihoodWeights
    exact_response: NDArray[np.float64]
    exact_count: NDArray[np.float64]
    parameter_independent_carrier: NDArray[np.float64]
    _plan_identifier: str

    @classmethod
    def _from_prepared(cls, weights, response, count, carrier) -> NegativeBinomialLikelihoodPlan:
        exact_response, exact_count, exact_carrier = map(immutable, (response, count, carrier))
        identifier = _plan_identifier(weights, exact_response, exact_count, exact_carrier)
        return cls(weights, exact_response, exact_count, exact_carrier, identifier)

    @property
    def plan_identifier(self) -> str:
        return self._plan_identifier

    def take(self, indices: NDArray[np.integer]) -> NegativeBinomialLikelihoodPlan:
        return type(self)._from_prepared(
            self.weights.take(indices),
            self.exact_response[indices],
            self.exact_count[indices],
            self.parameter_independent_carrier[indices],
        )


def _validated_response(y: NDArray) -> NDArray[np.float64]:
    message = "y must be a non-empty finite non-negative vector"
    try:
        source = np.asarray(y)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if source.ndim != 1 or len(source) == 0 or source.dtype.kind not in {"f", "i", "u", "O"}:
        raise ValueError(message)
    if source.dtype.kind == "O" and any(
        isinstance(value, bool | np.bool_ | complex | np.complexfloating)
        or not isinstance(value, int | float | np.number)
        for value in source
    ):
        raise ValueError(message)
    try:
        with np.errstate(over="raise", invalid="raise"):
            response = source.astype(_FLOAT, copy=False)
    except (FloatingPointError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError(message)
    if any(
        int(source_value) != float(converted_value)
        if isinstance(source_value, int | np.integer)
        else source_value != converted_value
        for source_value, converted_value in zip(source, response, strict=True)
    ):
        raise ValueError("y must be losslessly representable as exact float64 values")
    return response


def _typed_plan(plan: object, n_observations: int) -> NegativeBinomialLikelihoodPlan:
    return typed_plan(
        plan, NegativeBinomialLikelihoodPlan, n_observations, family_name="NegativeBinomialLS"
    )


def _certified_count_mapping(response, weights, semantics) -> NDArray[np.float64]:
    if semantics == "frequency":
        if np.any(_not_a_whole_number(response)):
            raise UnsupportedLikelihoodContractError("frequency NB2 responses must be counts")
        count = np.rint(response)
    elif semantics == "prior":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = weights * response
        if not np.all(np.isfinite(scaled)):
            raise UnsupportedLikelihoodContractError("prior NB2 counts must be representable")
        count = np.rint(scaled)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            inverse = count / weights
            lower_inverse = (count - 1.0) / weights
            upper_inverse = (count + 1.0) / weights
        exact = (
            (inverse == response)
            & ((count == 0.0) | (lower_inverse != response))
            & ((count == _MAX_EXACT_COUNT - 1.0) | (upper_inverse != response))
        )
        if np.any(~exact):
            raise UnsupportedLikelihoodContractError(
                "prior NB2 response must have one unique exact binary64 integer count"
            )
    else:
        raise UnsupportedLikelihoodContractError("NB2 plan has unsupported weight semantics")
    if np.any(count < 0.0) or np.any(count >= _MAX_EXACT_COUNT):
        raise UnsupportedLikelihoodContractError("NB2 counts must be in [0, 2**53)")
    return np.asarray(count, dtype=_FLOAT)


def _factorial_carrier(count, weights, semantics) -> NDArray[np.float64]:
    multiplier = np.ones_like(weights) if semantics == "prior" else weights
    with np.errstate(over="ignore", invalid="ignore"):
        carrier = -multiplier * special.gammaln(count + 1.0)
    if not np.all(np.isfinite(carrier)):
        raise UnsupportedLikelihoodContractError("NB2 factorial carrier is not representable")
    return np.asarray(carrier, dtype=_FLOAT)


_CAPABILITIES = FamilyCapabilities(2, False, False, False, False, True)
_PARAMETERS = (
    ParameterSpec("mean", LogLink(), "mean", ParameterSupport(lower=0.0), "observed"),
    ParameterSpec("theta", LogLink(), "size", ParameterSupport(lower=0.0), "observed"),
)


@dataclass(frozen=True)
class NegativeBinomialLS:
    """NB2 family with natural parameters conditional mean and size theta."""

    parameters = _PARAMETERS
    default_prediction_name = "conditional_mean"
    capabilities = _CAPABILITIES

    def to_config(self) -> dict[str, object]:
        return {"type": "NegativeBinomialLS", "parameterization": "nb2_mean_theta"}

    def response_boundaries(self, links: Sequence[Link]) -> tuple[tuple[str, ...], ...]:
        """All-zero rows drive the mean to 0 or the size to 0 under log links.

        ``P(Y = 0) = (theta / (theta + mu))^theta`` tends to one as
        ``log mu -> -inf`` and as ``log theta -> -inf``.
        """
        mean_link, theta_link = tuple(links)
        return (
            ("zero",) if isinstance(mean_link, LogLink) else (),
            ("zero",) if isinstance(theta_link, LogLink) else (),
        )

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> NegativeBinomialLikelihoodPlan:
        response = _validated_response(y)
        if observation != COMPLETE_OBSERVATION:
            raise UnsupportedLikelihoodContractError("NegativeBinomialLS requires complete rows")
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "NegativeBinomialLS requires resolved likelihood weights"
            )
        if len(weights.values) != len(response):
            raise UnsupportedLikelihoodContractError("NB2 response and weights differ in length")
        semantics = weights.provenance.contract.semantics
        count = _certified_count_mapping(response, weights.values, semantics)
        carrier = _factorial_carrier(count, weights.values, semantics)
        return NegativeBinomialLikelihoodPlan._from_prepared(weights, response, count, carrier)

    def validate_likelihood_plan(
        self,
        y: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        response = _validated_response(y)
        candidate = _typed_plan(plan, len(response))
        if response.tobytes(order="C") != candidate.exact_response.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "NegativeBinomialLS likelihood response does not match the fitted response"
            )
        return candidate.exact_response

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        candidate = _typed_plan(plan, response_row_count(y))
        theta = initialize_negative_binomial(
            candidate.exact_response,
            candidate.exact_count,
            candidate.weights.values,
            candidate.weights.provenance.contract.semantics,
        )
        return InitialParameterState(theta=theta)

    def diagnose_repeated_curvature_failure(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
    ) -> Exception | None:
        if type(self) is not NegativeBinomialLS:
            return None
        supplied = np.asarray(y)
        if supplied.ndim != 1:
            return None
        retained = np.array(supplied[weights.input_positions], copy=True)
        response = _validated_response(retained)
        if response.shape != weights.values.shape:
            return None
        semantics = weights.provenance.contract.semantics
        exact_count = _certified_count_mapping(response, weights.values, semantics)
        if not has_resolved_poisson_boundary(
            response,
            exact_count,
            weights.values,
            semantics,
        ):
            return None
        return NegativeBinomialPoissonBoundaryError(
            "The negative-binomial fit could not establish a stable finite theta and is "
            "Poisson-like at the diagnostic boundary"
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
        n_observations = response_row_count(y)
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=n_observations,
            parameters=self.parameters,
            family_name="NegativeBinomialLS",
        )
        candidate = _typed_plan(plan, n_observations)
        evaluated = evaluate_negative_binomial_rows(
            candidate.exact_count,
            parameters[:, 0],
            parameters[:, 1],
            candidate.weights.values,
            candidate.weights.provenance.contract.semantics,
            derivative_order=order,
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluated.optimizing_log_likelihood,
            parameter_independent_carrier=candidate.parameter_independent_carrier,
            score=evaluated.score,
            hessian_packed=evaluated.hessian_packed,
            valid=evaluated.valid,
        )

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="NegativeBinomialLS",
        )
        return immutable(parameters[:, 0])


__all__ = [
    "NegativeBinomialDerivativeRepresentationError",
    "NegativeBinomialInitializationError",
    "NegativeBinomialLikelihoodPlan",
    "NegativeBinomialLS",
    "NegativeBinomialPoissonBoundaryError",
]
