"""Normalized Gamma family parameterized by mean and coefficient of variation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm.distributional.families._base import (
    array_digest,
    contract_second_direction,
    immutable,
    readonly,
    second_direction_arrays,
    typed_plan,
    validated_float_response,
)
from superglm.distributional.families._predictors import (
    MeanPredictor,
    ScalePredictor,
)
from superglm.distributional.families._response_bounds import (
    ResponseBounds,
    TailChannels,
    bind_observation,
    bounded_channels,
    bounded_higher_derivatives,
    censored_carrier,
    log_window_values,
    refuse_bounded,
)
from superglm.distributional.families._variance import _variance_product
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    FamilyCapabilities,
    FamilyLikelihoodPlan,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
    _register_likelihood_reuse_contract,
    _validated_derivative_order,
    _validated_parameter_matrix,
)
from superglm.distributional.kernels._declared_bounds import gamma_window_jets
from superglm.distributional.kernels.gamma import (
    GammaInitializationError as GammaInitializationError,
)
from superglm.distributional.kernels.gamma import (
    evaluate_gamma_rows,
    gamma_expected_information,
    gamma_expected_shortfall,
    gamma_predictor_curvature_directional,
    gamma_predictor_fourth_derivatives,
    initialize_gamma,
)
from superglm.distributional.likelihood_identity import (
    LOG_ERROR_UNITS,
    CarrierRounding,
    carrier_rounding_scale,
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
    censored_response=True,
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

    legacy_identifier_tag: ClassVar[str] = "GammaLS/v1"

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
                "GammaLS/v2",
                self.row_law,
                self.invariant,
                *self.family_config,
                self.observation.kind,
                str(self.observation.schema_version),
                self.weights.digest,
                self.response_digest,
                *(() if self.observation.bounds is None else (self.observation.bounds.digest,)),
            )
        ).encode("utf-8")
        return f"GammaLS/v2:{hashlib.sha256(payload).hexdigest()}"

    def carrier_rounding(self) -> CarrierRounding:
        """The carrier ``-log y`` (times ``w`` for frequency weights) varies by platform."""
        carrier = self.parameter_independent_carrier
        return CarrierRounding(carrier, carrier_rounding_scale(carrier), LOG_ERROR_UNITS)

    def take(self, indices: NDArray[np.integer]) -> GammaLikelihoodPlan:
        return GammaLikelihoodPlan._from_prepared(
            weights=self.weights.take(indices),
            row_law=self.row_law,
            invariant=self.invariant,
            family_config=self.family_config,
            observation=self.observation.take(indices),
            exact_response=self.exact_response[indices],
            carrier=self.parameter_independent_carrier[indices],
        )


def _validated_plan(
    plan: FamilyLikelihoodPlan,
    *,
    n_observations: int,
) -> GammaLikelihoodPlan:
    return typed_plan(plan, GammaLikelihoodPlan, n_observations, family_name="GammaLS")


def _row_law(plan: GammaLikelihoodPlan) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per row, the prior weight in the shape ``w/cv^2`` and the replication mass."""
    weights = plan.weights.values
    if plan.weights.provenance.contract.semantics == "frequency":
        return np.ones_like(weights), weights
    return weights, np.ones_like(weights)


def _bounded_predictor_share(
    plan: GammaLikelihoodPlan, eta: NDArray, order: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Bounded rows' packed predictor derivatives of ``order`` (3 or 4).

    Returns the derivatives a bounded row adds to its complete-row ones, the
    rows whose complete-row derivatives stay (all but the censored), and the
    rows whose window is representable.  Each window's jets are formed once.
    """
    bounds = plan.observation.bounds
    assert bounds is not None
    prior, multiplier = _row_law(plan)
    predictors = np.asarray(eta, dtype=np.float64)
    zeros = np.zeros((len(prior), order + 1))

    def window_jets(lower: NDArray, upper: NDArray, rows: NDArray, jet_order: int):
        return gamma_window_jets(
            lower, upper, predictors[rows], prior[rows], order=jet_order, coordinates="log"
        )

    third, fourth, valid = bounded_higher_derivatives(
        zeros if order == 3 else None,
        zeros if order == 4 else None,
        np.ones(len(prior), dtype=bool),
        response=plan.exact_response,
        bounds=bounds,
        multiplier=multiplier,
        window_jets=window_jets,
    )
    share = third if order == 3 else fourth
    assert share is not None
    return share, (bounds.side == 0) & valid, valid


@dataclass(frozen=True)
class GammaLS(MeanPredictor, ScalePredictor):
    """Positive responses with mean and coefficient-of-variation predictors.

    Declare ``family.mean(...)`` and ``family.scale(...)``. Both use log
    links. Here ``scale`` is the coefficient of variation, so the unit-law
    variance is ``mean**2 * scale**2``. It is the square root of GLM Gamma
    dispersion, and differs from the scale argument of ``scipy.stats.gamma``.

    Results use the column names ``mean`` and ``scale`` in that order.
    Responses must be strictly positive. An empty helper call estimates a
    constant parameter through an intercept-only predictor.

    See Also
    --------
    SuperLSS : Construct and fit a model with these predictors.
    """

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

    def censored_recession(self, bounds: ResponseBounds) -> NDArray[np.bool_]:
        """Every censored row escapes along the mean; none is flagged on the scale.

        At a fixed coefficient of variation the mean is a scale parameter, and
        ``d^2 log f/(dy dm) = a/m^2 > 0`` is a monotone likelihood ratio in
        ``m``, so within any truncation window ``P(Y > c)`` rises and
        ``P(Y <= c)`` falls strictly with the mean.  Along the coefficient of
        variation, at a fixed mean, ``P(Y <= c)`` nears one both as ``cv``
        grows and, for ``c`` above the mean, as it shrinks, and ``P(Y > c)``
        nears its supremum as ``cv`` shrinks only for ``c`` below the mean and
        peaks at an interior ``cv`` above it, so the direction depends on the
        fitted mean.
        """
        return np.column_stack((bounds.side != 0, np.zeros(bounds.n_rows, dtype=bool)))

    def to_config(self) -> dict[str, Any]:
        return {"type": "GammaLS", "parameterization": "mean_cv"}

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> GammaLikelihoodPlan:
        response = _validated_response(y)
        observation = bind_observation(
            observation, response, weights, support_lower=0.0, family_name="GammaLS"
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
        carrier = censored_carrier(carrier, observation.bounds)
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
        observation = candidate.observation
        if observation != COMPLETE_OBSERVATION and (
            observation.bounds is None or observation.bounds.n_rows != len(response)
        ):
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
        optimizing, score, hessian, valid = (
            evaluated.optimizing_log_likelihood,
            evaluated.score,
            evaluated.hessian_packed,
            evaluated.valid,
        )
        bounds = gamma_plan.observation.bounds
        if bounds is not None:
            prior, multiplier = _row_law(gamma_plan)

            def window(lower: NDArray, upper: NDArray, rows: NDArray) -> TailChannels:
                jets = gamma_window_jets(lower, upper, parameters[rows], prior[rows], order=order)
                return jets[0], *(jets[k] if order >= k else None for k in (1, 2))

            optimizing, score, hessian, valid = bounded_channels(
                optimizing,
                score,
                hessian,
                valid,
                response=response,
                bounds=bounds,
                multiplier=multiplier,
                window=window,
                derivative_order=order,
            )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=optimizing,
            parameter_independent_carrier=gamma_plan.parameter_independent_carrier,
            score=score,
            hessian_packed=hessian,
            valid=valid,
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
        refuse_bounded(gamma_plan.observation, "GammaLS expected information")
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
        complete = gamma_predictor_curvature_directional(
            gamma_plan.exact_response,
            eta,
            eta_direction,
            gamma_plan.weights.values,
            gamma_plan.weights.provenance.contract.semantics,
        )
        if gamma_plan.observation.bounds is None:
            return complete
        third, kept, valid = _bounded_predictor_share(gamma_plan, eta, 3)
        v = np.asarray(eta_direction, dtype=np.float64)
        result = np.where(kept[:, None], complete, 0.0) - np.column_stack(
            [third[:, j] * v[:, 0] + third[:, j + 1] * v[:, 1] for j in range(3)]
        )
        result[~valid] = 0.0
        return readonly(result)

    def predictor_curvature_second_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        direction_a: NDArray,
        direction_b: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        """Differentiate observed predictor curvature twice, along ``a`` then ``b``."""

        response = _validated_response(y)
        link_tuple = tuple(links)
        if len(link_tuple) != 2 or any(type(link) is not LogLink for link in link_tuple):
            raise UnsupportedLikelihoodContractError(
                "Gamma endpoint curvature derivatives require the built-in log links"
            )
        gamma_plan = _validated_plan(plan, n_observations=len(response))
        single = np.ndim(direction_a) == 2
        predictors, first, second = second_direction_arrays(
            response, eta, direction_a, direction_b, 2
        )
        fourth = gamma_predictor_fourth_derivatives(
            gamma_plan.exact_response,
            predictors,
            gamma_plan.weights.values,
            gamma_plan.weights.provenance.contract.semantics,
        )
        valid = None
        if gamma_plan.observation.bounds is not None:
            share, kept, valid = _bounded_predictor_share(gamma_plan, predictors, 4)
            fourth = np.where(kept[:, None], fourth, 0.0) + share
        result = contract_second_direction(fourth, first, second)
        if valid is not None:
            result[..., ~valid, :] = 0.0
        return readonly(result[0] if single else result)

    def default_prediction(self, theta: NDArray) -> NDArray[np.float64]:
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="GammaLS",
        )
        return readonly(parameters[:, 0])

    def log_window_probability(
        self,
        lower: NDArray,
        upper: NDArray,
        theta: NDArray,
        weights: NDArray | None = None,
    ) -> NDArray[np.float64]:
        """``log P(lower < Y <= upper)`` per row, shape ``w/cv^2`` with prior weights ``w``."""
        values = _validated_parameter_matrix(
            theta, n_observations=None, parameters=self.parameters, family_name="GammaLS"
        )
        prior = (
            np.ones(len(values)) if weights is None else _prior_weight_vector(weights, len(values))
        )

        def window(lo: NDArray, hi: NDArray, rows: NDArray) -> TailChannels:
            return gamma_window_jets(lo, hi, values[rows], prior[rows], order=0)[0], None, None

        return readonly(log_window_values(lower, upper, len(values), window, support_lower=0.0))

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
        if (
            np.any(~np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or np.any(probabilities >= 1.0)
        ):
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
        mean, cv = values[:, 0], values[:, 1]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            spread = mean * cv
            square = spread * spread
            result = square / prior
        unsafe = ~np.isfinite(square) | (square < np.finfo(float).tiny)
        for index in np.flatnonzero(unsafe):
            result[index] = _variance_product(
                (mean[index], cv[index], mean[index], cv[index]), (prior[index],)
            )
        return readonly(result)

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
        if (
            np.any(~np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or np.any(probabilities >= 1.0)
        ):
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


_register_likelihood_reuse_contract(
    GammaLS,
    GammaLikelihoodPlan,
    prepared_array_fields=("exact_response", "parameter_independent_carrier"),
    deterministic_chunk_replay=True,
)
