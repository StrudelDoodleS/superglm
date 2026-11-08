"""Exact normalized NB2 family in natural mean-size coordinates."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm._count_lattice import _not_a_whole_number
from superglm.distributional.families._base import (
    array_digest,
    immutable,
    scalar_response_row_count,
    typed_plan,
    validated_observation_response,
)
from superglm.distributional.families._predictors import (
    MeanPredictor,
    ThetaPredictor,
)
from superglm.distributional.families._response_bounds import (
    bind_observation,
    bounded_channels,
    refuse_bounded,
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
    _register_likelihood_reuse_contract,
    _validated_derivative_order,
    _validated_parameter_matrix,
)
from superglm.distributional.kernels._negative_binomial_windows import (
    CountWindowRepresentationError,
    count_window_endpoints,
    negative_binomial_normalized_jets,
    negative_binomial_window_jets,
)
from superglm.distributional.kernels.negative_binomial import (
    NegativeBinomialDerivativeRepresentationError,
    NegativeBinomialPoissonBoundaryError,
    evaluate_negative_binomial_rows,
    has_resolved_poisson_boundary,
    initialize_negative_binomial,
    negative_binomial_certified_predictor_curvature_directional,
    negative_binomial_certified_predictor_curvature_second_directional,
    negative_binomial_expected_information,
    poisson_limit_row_count,
)
from superglm.distributional.kernels.negative_binomial import (
    NegativeBinomialInitializationError as NegativeBinomialInitializationError,
)
from superglm.distributional.likelihood_identity import (
    LGAMMA_ERROR_UNITS,
    CarrierRounding,
    carrier_rounding_scale,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)
from superglm.links import Link, LogLink

_FLOAT = np.float64
_MAX_EXACT_COUNT = 2**53
_FAMILY_CONFIGURATION = ("NegativeBinomialLS/v1", "nb2-mean-theta/v1")


def _plan_identifier(weights, response, count, observation=COMPLETE_OBSERVATION) -> str:
    law = (
        ("nb2-mean-theta-prior-scaled-count/v1", "conditional-mean")
        if weights.provenance.contract.semantics == "prior"
        else ("nb2-mean-theta-literal-replication/v1", "literal-row-replication")
    )
    identity = ("NegativeBinomialLS/v2", *law, *_FAMILY_CONFIGURATION, "complete", "1")
    digests = (
        array_digest(f"NegativeBinomialLS/{name}/v1\0".encode(), values)
        for name, values in zip(("response", "count"), (response, count))
    )
    payload = "\0".join((*identity, weights.digest, *digests)).encode()
    if observation.bounds is not None:
        # Preserve the complete plan's historical identity byte-for-byte.
        # Original thresholds remain authoritative even for equal count sets.
        payload += b"\0finite-nb-bounded/v1\0" + observation.bounds.digest.encode()
        return f"NegativeBinomialLS/v3:{hashlib.sha256(payload).hexdigest()}"
    return f"NegativeBinomialLS/v2:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class NegativeBinomialLikelihoodPlan:
    """Prepared NB2 rows sharing one resolved likelihood-weight carrier."""

    legacy_identifier_tag: ClassVar[str] = "NegativeBinomialLS/v1"

    weights: ResolvedLikelihoodWeights
    exact_response: NDArray[np.float64]
    exact_count: NDArray[np.float64]
    parameter_independent_carrier: NDArray[np.float64]
    _plan_identifier: str
    observation: ObservationContract = COMPLETE_OBSERVATION

    @classmethod
    def _from_prepared(
        cls, weights, response, count, carrier, observation=COMPLETE_OBSERVATION
    ) -> NegativeBinomialLikelihoodPlan:
        exact_response, exact_count, exact_carrier = map(immutable, (response, count, carrier))
        identifier = _plan_identifier(weights, exact_response, exact_count, observation)
        return cls(weights, exact_response, exact_count, exact_carrier, identifier, observation)

    @property
    def exact_row_mask(self) -> NDArray[np.bool_]:
        bounds = self.observation.bounds
        mask = np.ones(len(self.exact_response), dtype=bool) if bounds is None else bounds.side == 0
        return np.frombuffer(mask.tobytes(), dtype=np.bool_)

    @property
    def plan_identifier(self) -> str:
        return self._plan_identifier

    def carrier_rounding(self) -> CarrierRounding:
        """The carrier ``-m log(count!)`` varies by platform."""
        carrier = self.parameter_independent_carrier
        multiplier = (
            self.weights.values
            if self.weights.provenance.contract.semantics == "frequency"
            else 1.0
        )
        scale = carrier_rounding_scale(carrier, multiplier, 1.0)
        return CarrierRounding(carrier, scale, LGAMMA_ERROR_UNITS)

    def take(self, indices: NDArray[np.integer]) -> NegativeBinomialLikelihoodPlan:
        return type(self)._from_prepared(
            self.weights.take(indices),
            self.exact_response[indices],
            self.exact_count[indices],
            self.parameter_independent_carrier[indices],
            self.observation.take(indices),
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
    if not _lossless(source, response):
        raise ValueError("y must be losslessly representable as exact float64 values")
    return response


def _lossless(source: NDArray, response: NDArray[np.float64]) -> bool:
    """Whether every converted value equals its source exactly.

    Vectorised by dtype: a float source compares exactly against its float64
    image; an integer source of magnitude at most ``2**53`` converts exactly,
    so only larger integers (and object arrays) are compared one by one, as
    exact integer-float comparisons.
    """
    if source.dtype.kind == "f":
        return not bool(np.any(source != response))
    if source.dtype.kind in "iu":
        large = np.flatnonzero((source > _MAX_EXACT_COUNT) | (source < -_MAX_EXACT_COUNT))
        return all(int(source[index]) == float(response[index]) for index in large)
    return not any(
        int(source_value) != float(converted_value)
        if isinstance(source_value, int | np.integer)
        else source_value != converted_value
        for source_value, converted_value in zip(source, response, strict=True)
    )


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


def _count_sets(response, count, observation, prior):
    bounds = observation.bounds
    assert bounds is not None
    L, U, empty = count_window_endpoints(bounds.lower, bounds.upper, prior)
    lower = np.where(bounds.side == 1, response, bounds.lower)
    upper = np.where(bounds.side == -1, response, bounds.upper)
    OL, OU, observed_empty = count_window_endpoints(lower, upper, prior)
    exact = bounds.side == 0
    OL[exact], OU[exact] = count[exact], count[exact]
    observed_empty[exact] = (count[exact] < L[exact]) | (count[exact] > U[exact])
    return L, U, OL, OU, empty | observed_empty


_CAPABILITIES = FamilyCapabilities(2, True, False, False, False, True, censored_response=True)


@dataclass(frozen=True)
class NegativeBinomialLS(MeanPredictor, ThetaPredictor):
    """Negative-binomial responses with mean and size predictors.

    Declare ``family.mean(...)`` and ``family.theta(...)``. Both use log
    links. The unit count law has variance ``mean + mean**2 / theta``, so
    larger theta means less overdispersion. ``theta`` is the size parameter
    in the NB2 parametrization.

    Results use the column names ``mean`` and ``theta`` in that order.
    ``family.theta()`` estimates a constant size; a declaration with terms
    lets size vary between rows. The observation law for counts or weighted
    rates follows the model's weight contract.

    See Also
    --------
    SuperLSS : Construct and fit a model with these predictors.
    """

    # Its curvature directions take stacked (m, n, K) directions on one state evaluation.
    stacked_curvature_directions: ClassVar[bool] = True

    default_prediction_name = "conditional_mean"
    capabilities = _CAPABILITIES
    # The expected information is only the terminal curvature fallback; starting
    # penalties stay at the 0.1 start, which converges where a scaled start may not.
    information_scaled_start: ClassVar[bool] = False
    # Any curvature fallback gets this family's Poisson-boundary diagnosis,
    # which checks the data itself.
    diagnoses_any_curvature_fallback: ClassVar[bool] = True

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec("mean", LogLink(), "mean", ParameterSupport(lower=0.0), "observed"),
            ParameterSpec("theta", LogLink(), "size", ParameterSupport(lower=0.0), "observed"),
        )

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
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "NegativeBinomialLS requires resolved likelihood weights"
            )
        n = scalar_response_row_count(y)
        if len(weights.values) != n:
            raise UnsupportedLikelihoodContractError("NB2 response and weights differ in length")
        bounds = observation.bounds if isinstance(observation, ObservationContract) else None
        response = validated_observation_response(
            y,
            censored=None if bounds is None else bounds.side != 0,
            exact_validator=_validated_response,
            input_positions=weights.input_positions,
        )
        # Validate the original windows before support/count conversion. Keep
        # their original digest in the plan even when support clipping is inert.
        bind_observation(
            observation,
            response,
            weights,
            support_lower=0.0,
            family_name="NegativeBinomialLS",
            atom_at_lower=True,
        )
        semantics = weights.provenance.contract.semantics
        exact = np.ones(n, dtype=bool) if bounds is None else bounds.side == 0
        count = np.zeros(n)
        count[exact] = _certified_count_mapping(response[exact], weights.values[exact], semantics)
        if bounds is None:
            carrier = _factorial_carrier(count, weights.values, semantics)
        else:
            prior = weights.values if semantics == "prior" else np.ones(n)
            try:
                *_, empty = _count_sets(response, count, observation, prior)
            except CountWindowRepresentationError as exc:
                # Failure-only per-row replay attaches input provenance.
                for row in range(n):
                    try:
                        take = np.array([row])
                        _count_sets(
                            response[take], count[take], observation.take(take), prior[take]
                        )
                    except CountWindowRepresentationError as row_exc:
                        raise UnsupportedLikelihoodContractError(
                            f"{row_exc} (input row {weights.input_positions[row]})"
                        ) from exc
                raise
            if np.any(empty):
                row = int(weights.input_positions[np.flatnonzero(empty)[0]])
                raise ValueError(
                    f"NB2 observation has zero probability: empty count set (input row {row})"
                )
            carrier = np.zeros(n)
        return NegativeBinomialLikelihoodPlan._from_prepared(
            weights, response, count, carrier, observation
        )

    def validate_likelihood_plan(
        self,
        y: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        candidate = _typed_plan(plan, scalar_response_row_count(y))
        response = validated_observation_response(
            y,
            censored=~candidate.exact_row_mask,
            exact_validator=_validated_response,
            input_positions=candidate.weights.input_positions,
        )
        if response.tobytes(order="C") != candidate.exact_response.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "NegativeBinomialLS likelihood response does not match the fitted response"
            )
        return candidate.exact_response

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        candidate = _typed_plan(plan, scalar_response_row_count(y))
        if not candidate.observation.is_complete:
            prior = (
                candidate.weights.values
                if candidate.weights.provenance.contract.semantics == "prior"
                else np.ones(len(candidate.exact_count))
            )
            _, _, OL, OU, _ = _count_sets(
                candidate.exact_response, candidate.exact_count, candidate.observation, prior
            )
            representative = np.where(
                np.isfinite(OU), np.floor(0.5 * OL + 0.5 * OU), np.maximum(OL, 1.0)
            )
            mean = max(
                float(np.average(representative / prior, weights=candidate.weights.values)), 0.1
            )
            return InitialParameterState(theta=np.tile([mean, max(mean, 1.0)], (len(prior), 1)))
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

    def limit_row_count(self, theta: NDArray, plan: FamilyLikelihoodPlan) -> tuple[int, int]:
        """The size's parameter index and how many rows the kernel treats as Poisson.

        Rows with ``max(mean, count - 1)`` below ``2**-52`` times the size, in
        effective units. The count informs the disclosure only; the terminal
        rank decision is the test. A derived family reports no rows.
        """
        if type(self) is not NegativeBinomialLS:
            return 1, 0
        candidate = _typed_plan(plan, len(theta))
        if not candidate.observation.is_complete:
            return 1, 0
        semantics = candidate.weights.provenance.contract.semantics
        n = poisson_limit_row_count(
            theta[:, 0], theta[:, 1], candidate.weights.values, semantics, candidate.exact_count
        )
        return 1, n

    def limit_row_disclosure(
        self,
        n_limit: int,
        n_rows: int,
        constant_offered: bool,
        offset_varies: bool = False,
    ) -> str:
        """Say that the size is not identified, with ``n_limit`` rows at the Poisson limit.

        ``constant_offered`` (see ``LimitRowDisclosingFamily``) is true when a
        constant size can be offered and the size has terms to drop; otherwise
        only the Poisson family is suggested. A derived family gets no text.
        """
        if type(self) is not NegativeBinomialLS:
            return ""
        head = (
            f"SuperLSS NegativeBinomialLS fit: {n_limit} of {n_rows} rows have a size at the "
            "Poisson limit (no overdispersion left there; rows close to it are not counted), "
            "and the fit's final curvature is singular in a size direction, so the size "
            "coefficients are not identified and the fit may stop short of convergence."
        )
        if not constant_offered:
            return f"{head} No row identifies even a constant size; use the Poisson family instead."
        dropped = "no terms and no size offset" if offset_varies else "no terms"
        remedy = (
            f"Fit a constant size (family.theta() with {dropped}) or the Poisson family instead."
        )
        return f"{head} {remedy}"

    def expected_information_natural(
        self,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        """Packed natural-scale Fisher information of each row's mean and size."""
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="NegativeBinomialLS",
        )
        candidate = _typed_plan(plan, len(parameters))
        refuse_bounded(candidate.observation, "NegativeBinomialLS expected information")
        return negative_binomial_expected_information(
            parameters[:, 0],
            parameters[:, 1],
            candidate.weights.values,
            candidate.weights.provenance.contract.semantics,
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
        n_observations = scalar_response_row_count(y)
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=n_observations,
            parameters=self.parameters,
            family_name="NegativeBinomialLS",
        )
        candidate = _typed_plan(plan, n_observations)
        if candidate.observation.bounds is not None:
            prior = (
                candidate.weights.values
                if candidate.weights.provenance.contract.semantics == "prior"
                else np.ones(n_observations)
            )
            mass = (
                np.ones(n_observations)
                if candidate.weights.provenance.contract.semantics == "prior"
                else candidate.weights.values
            )
            L, U, OL, OU, _ = _count_sets(
                candidate.exact_response, candidate.exact_count, candidate.observation, prior
            )

            def normalized(rows, requested_order):
                jet = negative_binomial_normalized_jets(
                    L[rows],
                    U[rows],
                    OL[rows],
                    OU[rows],
                    parameters[rows],
                    prior[rows],
                    order=requested_order,
                )
                return (
                    jet[0],
                    jet[1] if requested_order >= 1 else None,
                    jet[2] if requested_order == 2 else None,
                )

            def window(lo, hi, rows):
                jet = negative_binomial_window_jets(
                    lo, hi, parameters[rows], prior[rows], order=order
                )
                return jet[0], jet[1] if order >= 1 else None, jet[2] if order == 2 else None

            value, score, hessian, valid = bounded_channels(
                np.zeros(n_observations),
                np.zeros((n_observations, 2)) if order >= 1 else None,
                np.zeros((n_observations, 3)) if order == 2 else None,
                np.ones(n_observations, dtype=bool),
                response=candidate.exact_response,
                bounds=candidate.observation.bounds,
                multiplier=mass,
                window=window,
                derivative_order=order,
                normalized_rows=np.arange(n_observations),
                normalized=normalized,
                carrier=candidate.parameter_independent_carrier,
            )
            return NaturalLikelihoodEvaluation(
                optimizing_log_likelihood=value,
                parameter_independent_carrier=candidate.parameter_independent_carrier,
                score=score,
                hessian_packed=hessian,
                valid=valid,
            )
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

    def predictor_curvature_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        eta_direction: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        """Differentiate observed predictor curvature along one predictor path."""

        values, _certificate = self.certified_predictor_curvature_directional_derivative(
            y, eta, eta_direction, links, plan
        )
        return values

    def certified_predictor_curvature_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        eta_direction: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The curvature direction and the certified error of its ``l_bbb`` channel."""

        link_tuple = tuple(links)
        if len(link_tuple) != 2 or any(type(link) is not LogLink for link in link_tuple):
            raise UnsupportedLikelihoodContractError(
                "NB2 endpoint curvature derivatives require the built-in log links"
            )
        candidate = _typed_plan(plan, scalar_response_row_count(y))
        refuse_bounded(candidate.observation, "NegativeBinomialLS curvature direction")
        return negative_binomial_certified_predictor_curvature_directional(
            candidate.exact_count,
            eta,
            eta_direction,
            candidate.weights.values,
            candidate.weights.provenance.contract.semantics,
        )

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

        values, _certificate = self.certified_predictor_curvature_second_directional_derivative(
            y, eta, direction_a, direction_b, links, plan
        )
        return values

    def certified_predictor_curvature_second_directional_derivative(
        self,
        y: NDArray,
        eta: NDArray,
        direction_a: NDArray,
        direction_b: NDArray,
        links: Sequence[Link],
        plan: FamilyLikelihoodPlan,
        *,
        step: float = 1.0e-3,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The second curvature direction and the bounded error of its ``l_bbbb`` channel.

        The series bound does not depend on the derivative ``step``.
        """

        del step
        link_tuple = tuple(links)
        if len(link_tuple) != 2 or any(type(link) is not LogLink for link in link_tuple):
            raise UnsupportedLikelihoodContractError(
                "NB2 endpoint curvature derivatives require the built-in log links"
            )
        candidate = _typed_plan(plan, scalar_response_row_count(y))
        refuse_bounded(candidate.observation, "NegativeBinomialLS second curvature direction")
        return negative_binomial_certified_predictor_curvature_second_directional(
            candidate.exact_count,
            eta,
            direction_a,
            direction_b,
            candidate.weights.values,
            candidate.weights.provenance.contract.semantics,
        )

    def log_window_probability(self, lower, upper, theta, weights=None) -> NDArray[np.float64]:
        """``log P(lower < N/w <= upper)``; omitted weights mean unit exposure."""
        parameters = _validated_parameter_matrix(
            theta,
            n_observations=None,
            parameters=self.parameters,
            family_name="NegativeBinomialLS",
        )
        prior = np.ones(len(parameters)) if weights is None else weights
        values = negative_binomial_window_jets(lower, upper, parameters, prior, order=0)[0]
        unresolved = np.isnan(values) | np.isposinf(values)
        if np.any(unresolved):
            raise UnsupportedLikelihoodContractError(
                "NegativeBinomialLS window probability is numerically unresolved at rows "
                f"{np.flatnonzero(unresolved).tolist()} (parameter range or work/convergence limit)"
            )
        return immutable(values)

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


_register_likelihood_reuse_contract(
    NegativeBinomialLS,
    NegativeBinomialLikelihoodPlan,
    prepared_array_fields=("exact_response", "exact_count", "parameter_independent_carrier"),
)
