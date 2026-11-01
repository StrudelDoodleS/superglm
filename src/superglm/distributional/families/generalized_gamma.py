"""Prentice generalized gamma family, mean-parametrised by default."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.families._base import (
    array_digest,
    immutable,
    natural_curvature_direction,
    natural_curvature_second_direction,
    readonly,
    typed_plan,
    validated_float_response,
)
from superglm.distributional.families._predictors import (
    LocationPredictor,
    MeanPredictor,
    ScalePredictor,
    ShapePredictor,
)
from superglm.distributional.families._response_bounds import (
    ResponseBounds,
    TailChannels,
    bind_observation,
    bounded_channels,
    bounded_higher_derivatives,
    censored_carrier,
    contract_third,
    log_window_values,
    refuse_bounded,
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
    _register_likelihood_reuse_contract,
    _validated_derivative_order,
    _validated_parameter_matrix,
)
from superglm.distributional.kernels._declared_bounds import gg_window_jets
from superglm.distributional.kernels.generalized_gamma import (
    GeneralizedGammaDomainError as GeneralizedGammaDomainError,
)
from superglm.distributional.kernels.generalized_gamma import (
    generalized_gamma_cdf,
    generalized_gamma_expected_shortfall,
    generalized_gamma_expected_shortfall_from_mean,
    generalized_gamma_quantile,
    generalized_gamma_variance,
    initialize_generalized_gamma,
    location_expected_information,
    location_hessian_direction,
    location_of_mean,
    location_rows,
    location_third_fourth,
    mean_exists,
    mean_expected_information,
    mean_hessian_direction,
    mean_of_location,
    mean_rows,
    mean_third_fourth,
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
from superglm.links import IdentityLink, Link, LogLink

Parametrisation = Literal["mean", "location"]
GeneralizedGammaRowLaw = Literal[
    "unit-prior-explicitly-equals-frequency/v1",
    "gg-prentice-literal-replication/v1",
]
_FAMILY_NAME = "GeneralizedGammaLSS"
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_CAPABILITIES = FamilyCapabilities(
    max_derivative_order=2,
    expected_information=True,
    cdf=True,
    quantile=True,
    random=False,
    response_mean=True,
    censored_response=True,
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
class GeneralizedGammaLikelihoodPlan:
    """One exact-response generalized gamma law and immutable carriers."""

    legacy_identifier_tag: ClassVar[str] = "GeneralizedGammaLSS/v1"

    weights: ResolvedLikelihoodWeights
    row_law: GeneralizedGammaRowLaw
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
        row_law: GeneralizedGammaRowLaw,
        family_config: tuple[str, str, str],
        observation: ObservationContract,
        exact_response: NDArray,
        carrier: NDArray,
    ) -> GeneralizedGammaLikelihoodPlan:
        response = immutable(exact_response)
        immutable_carrier = immutable(carrier)
        return cls(
            weights=weights,
            row_law=row_law,
            family_config=family_config,
            observation=observation,
            exact_response=response,
            response_digest=array_digest(b"GeneralizedGammaLSS/response/v1\0", response),
            parameter_independent_carrier=immutable_carrier,
            carrier_digest=array_digest(b"GeneralizedGammaLSS/carrier/v1\0", immutable_carrier),
        )

    @property
    def plan_identifier(self) -> str:
        payload = "\0".join(
            (
                "GeneralizedGammaLSS/v2",
                self.row_law,
                *self.family_config,
                self.observation.kind,
                str(self.observation.schema_version),
                self.weights.digest,
                self.response_digest,
                *(() if self.observation.bounds is None else (self.observation.bounds.digest,)),
            )
        ).encode("utf-8")
        return f"GeneralizedGammaLSS/v2:{hashlib.sha256(payload).hexdigest()}"

    def carrier_rounding(self) -> CarrierRounding:
        """The carrier ``-m (log y + 0.5 log 2 pi)`` varies by platform."""
        carrier = self.parameter_independent_carrier
        multiplier = (
            self.weights.values
            if self.weights.provenance.contract.semantics == "frequency"
            else 1.0
        )
        scale = carrier_rounding_scale(carrier, multiplier, 2.0 * abs(_HALF_LOG_TWO_PI))
        return CarrierRounding(carrier, scale, LOG_ERROR_UNITS)

    def take(self, indices: NDArray[np.integer]) -> GeneralizedGammaLikelihoodPlan:
        return GeneralizedGammaLikelihoodPlan._from_prepared(
            weights=self.weights.take(indices),
            row_law=self.row_law,
            family_config=self.family_config,
            observation=self.observation.take(indices),
            exact_response=self.exact_response[indices],
            carrier=self.parameter_independent_carrier[indices],
        )

    @property
    def multiplier(self) -> NDArray[np.float64]:
        """The replication mass per row: the weights under frequency, ones under unit prior."""
        if self.row_law == "gg-prentice-literal-replication/v1":
            return self.weights.values
        return np.ones_like(self.weights.values)


@dataclass(frozen=True)
class GeneralizedGammaLSS(MeanPredictor, LocationPredictor, ScalePredictor, ShapePredictor):
    """Generalized gamma with natural parameters ``(mean | location, scale, shape)``.

    Prentice's ``(mu, sigma, Q)`` law on ``y > 0``: ``shape = 0`` is the
    log-normal, ``shape = 1`` the Weibull, ``shape = scale`` the gamma.  The
    default mean form puts ``E[Y]`` first under a log link, so its relativities
    multiply the mean; the location form puts the log-scale location first
    under an identity link and admits an infinite mean.

    Parameters
    ----------
    parametrisation : {"mean", "location"}, default="mean"
        Choose ``family.mean(...)`` for the conditional response mean or
        ``family.location(...)`` for Prentice's log-scale location. Declare
        ``family.scale(...)`` and ``family.shape(...)`` in either form.
        Scale is Prentice's sigma and shape is Q. Scale need not equal the
        standard deviation of the response or its logarithm.
    scale_floor : float, default=0.01
        Nonnegative lower bound on scale. Its default link is
        ``log(scale - scale_floor)``. Shape uses an identity link.

    Notes
    -----
    Changing the first parameter changes what its additive predictor
    describes. The mean form requires a finite response mean. Results
    contain ``mean, scale, shape`` or ``location, scale, shape``.
    ``predict`` returns the response mean in both forms, including infinity
    where the mean does not exist in the location form.
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
            ParameterSpec("shape", IdentityLink(), "shape", ParameterSupport(), "observed"),
        )

    @property
    def default_prediction_name(self) -> str:
        return "conditional_mean"

    @property
    def capabilities(self) -> FamilyCapabilities:
        return _CAPABILITIES

    def censored_recession(self, bounds: ResponseBounds) -> NDArray[np.bool_]:
        """Every censored row escapes along the location or mean; none along scale or shape.

        At a fixed scale and shape ``log Y = mu + sigma W`` is a location
        family in ``mu``, and the mean form's ``mu = log m - log C(sigma, Q)``
        is a strictly increasing shift.  ``W``'s density is log-concave (``d^2
        log f/dw^2 = -e^(Q w)``), so the family has a monotone likelihood
        ratio in ``mu``, and within any truncation window ``P(Y > c)`` rises
        and ``P(Y <= c)`` falls strictly with ``mu``.  Along the scale or the
        shape a bound probability's limit depends on the fitted location and
        the other parameter, so those predictors are not flagged.
        """
        return np.column_stack((bounds.side != 0, np.zeros((bounds.n_rows, 2), dtype=bool)))

    def to_config(self) -> dict[str, Any]:
        return {
            "type": _FAMILY_NAME,
            "parametrisation": self.parametrisation,
            "scale_floor": self.scale_floor,
        }

    @property
    def _family_config(self) -> tuple[str, str, str]:
        return (
            "GeneralizedGammaLSS/v1",
            f"prentice-{self.parametrisation}/v1",
            f"scale_floor={self.scale_floor!r}",
        )

    def bind_likelihood(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> GeneralizedGammaLikelihoodPlan:
        response = _validated_response(y)
        observation = bind_observation(
            observation, response, weights, support_lower=0.0, family_name=_FAMILY_NAME
        )
        if not isinstance(weights, ResolvedLikelihoodWeights):
            raise UnsupportedLikelihoodContractError(
                "GeneralizedGammaLSS requires resolved likelihood weights"
            )
        if len(weights.values) != len(response):
            raise UnsupportedLikelihoodContractError(
                "generalized gamma response rows do not match resolved likelihood-weight rows"
            )
        semantics = weights.provenance.contract.semantics
        if semantics == "prior":
            if not weights.provenance.all_unit:
                raise UnsupportedLikelihoodContractError(
                    "GeneralizedGammaLSS is not a reproductive family, so non-unit prior "
                    "weights have no likelihood law; fit claim-level rows, carry exposure "
                    "through offsets, or use weight_semantics='frequency' for integer "
                    "replication"
                )
            row_law: GeneralizedGammaRowLaw = "unit-prior-explicitly-equals-frequency/v1"
            multiplier = np.ones_like(weights.values)
        elif semantics == "frequency":
            row_law = "gg-prentice-literal-replication/v1"
            multiplier = weights.values
        else:  # pragma: no cover - resolved certification owns this branch
            raise UnsupportedLikelihoodContractError(
                "GeneralizedGammaLSS does not support the resolved likelihood-weight contract"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            carrier = -multiplier * (np.log(response) + _HALF_LOG_TWO_PI)
        carrier = censored_carrier(carrier, observation.bounds)
        if not np.all(np.isfinite(carrier)):
            raise UnsupportedLikelihoodContractError(
                "generalized gamma response carrier is not finite and representable"
            )
        return GeneralizedGammaLikelihoodPlan._from_prepared(
            weights=weights,
            row_law=row_law,
            family_config=self._family_config,
            observation=observation,
            exact_response=response,
            carrier=carrier,
        )

    def _plan(
        self, plan: FamilyLikelihoodPlan, n_observations: int
    ) -> GeneralizedGammaLikelihoodPlan:
        candidate = typed_plan(
            plan, GeneralizedGammaLikelihoodPlan, n_observations, family_name=_FAMILY_NAME
        )
        if candidate.family_config != self._family_config:
            raise UnsupportedLikelihoodContractError(
                "GeneralizedGammaLSS likelihood was prepared for another configuration"
            )
        return candidate

    def validate_likelihood_plan(
        self,
        y: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        response = _validated_response(y)
        candidate = self._plan(plan, len(response))
        observation = candidate.observation
        if observation != COMPLETE_OBSERVATION and (
            observation.bounds is None or observation.bounds.n_rows != len(response)
        ):
            raise UnsupportedLikelihoodContractError(
                "GeneralizedGammaLSS likelihood has an unsupported observation contract"
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
                "GeneralizedGammaLSS likelihood owns an invalid bound response"
            )
        if response.tobytes(order="C") != bound.tobytes(order="C"):
            raise UnsupportedLikelihoodContractError(
                "GeneralizedGammaLSS likelihood response does not match the fitted response"
            )
        return bound

    def initialize(self, y: NDArray, plan: FamilyLikelihoodPlan) -> InitialParameterState:
        response = _validated_response(y)
        bound = self._plan(plan, len(response))
        theta = initialize_generalized_gamma(
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
            response,
            values[:, 0],
            values[:, 1],
            values[:, 2],
            bound.multiplier,
            derivative_order=order,
        )
        optimizing, score, hessian, valid = (
            evaluated.optimizing_log_likelihood,
            evaluated.score,
            evaluated.hessian_packed,
            evaluated.valid,
        )
        if bound.observation.bounds is not None:

            def window(lower: NDArray, upper: NDArray, at: NDArray) -> TailChannels:
                jets = self._window_jets(lower, upper, values[at], order)
                return jets[0], *(jets[k] if order >= k else None for k in (1, 2))

            optimizing, score, hessian, valid = bounded_channels(
                optimizing,
                score,
                hessian,
                valid,
                response=response,
                bounds=bound.observation.bounds,
                multiplier=bound.multiplier,
                window=window,
                derivative_order=order,
            )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=optimizing,
            parameter_independent_carrier=bound.parameter_independent_carrier,
            score=score,
            hessian_packed=hessian,
            valid=valid,
        )

    def _window_jets(self, lower: NDArray, upper: NDArray, theta: NDArray, order: int):
        return gg_window_jets(
            lower, upper, theta, order=order, mean_form=self.parametrisation == "mean"
        )

    def _bounded_higher(
        self,
        plan: GeneralizedGammaLikelihoodPlan,
        response: NDArray,
        theta: NDArray,
        third: NDArray | None,
        fourth: NDArray | None,
    ):
        """:func:`bounded_higher_derivatives` with each bounded row's window formed once."""
        bounds = plan.observation.bounds
        assert bounds is not None

        def window_jets(lower: NDArray, upper: NDArray, at: NDArray, order: int):
            return self._window_jets(lower, upper, theta[at], order)

        return bounded_higher_derivatives(
            third,
            fourth,
            np.ones(len(response), dtype=bool),
            response=response,
            bounds=bounds,
            multiplier=plan.multiplier,
            window_jets=window_jets,
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
        bound = self._plan(plan, len(response))
        rows = (
            mean_hessian_direction if self.parametrisation == "mean" else location_hessian_direction
        )

        windows_valid = np.ones(len(response), dtype=bool)

        def hessian_direction(theta: NDArray, natural: NDArray) -> NDArray[np.float64]:
            complete = rows(
                response, theta[:, 0], theta[:, 1], theta[:, 2], natural, bound.multiplier
            )
            if bound.observation.bounds is None:
                return complete
            share, _, valid = self._bounded_higher(
                bound, response, theta, np.zeros((len(response), 10)), None
            )
            windows_valid[:] = valid
            kept = (bound.observation.bounds.side == 0)[:, None]
            return np.where(kept, complete, 0.0) + contract_third(share, natural)

        result = natural_curvature_direction(
            self, response, eta, eta_direction, links, plan, hessian_direction
        )
        if np.all(windows_valid):
            return result
        result = result.copy()
        result[~windows_valid] = 0.0
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
        bound = self._plan(plan, len(response))
        rows = mean_third_fourth if self.parametrisation == "mean" else location_third_fourth

        windows_valid = np.ones(len(response), dtype=bool)

        def natural_derivatives(theta: NDArray, natural_a: NDArray, natural_b: NDArray):
            third, fourth = rows(response, theta[:, 0], theta[:, 1], theta[:, 2], bound.multiplier)
            if bound.observation.bounds is None:
                return third, fourth
            third, fourth, valid = self._bounded_higher(bound, response, theta, third, fourth)
            windows_valid[:] = valid
            return third, fourth

        result = natural_curvature_second_direction(
            self, response, eta, direction_a, direction_b, links, plan, natural_derivatives
        )
        if np.all(windows_valid):
            return result
        result = result.copy()
        result[..., ~windows_valid, :] = 0.0
        return readonly(result)

    def expected_information_natural(
        self,
        theta: NDArray,
        plan: FamilyLikelihoodPlan,
    ) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        bound = self._plan(plan, len(values))
        refuse_bounded(bound.observation, "GeneralizedGammaLSS expected information")
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

    def variance(self, theta: NDArray) -> NDArray[np.float64]:
        """Population variance, with infinity for a divergent second moment."""
        values = self._theta(theta, None)
        return generalized_gamma_variance(
            values[:, 0], values[:, 1], values[:, 2], parametrisation=self.parametrisation
        )

    def log_window_probability(
        self,
        lower: NDArray,
        upper: NDArray,
        theta: NDArray,
        weights: NDArray | None = None,
    ) -> NDArray[np.float64]:
        """``log P(lower < Y <= upper)`` per row; NaN where the mean form's mean does not exist."""
        if weights is not None and not np.all(np.asarray(weights, dtype=np.float64) == 1.0):
            raise UnsupportedLikelihoodContractError(
                "GeneralizedGammaLSS has no prior-weighted row law, so its window probability "
                "takes no non-unit prior weights"
            )
        values = self._theta(theta, None)

        def window(lo: NDArray, hi: NDArray, at: NDArray) -> TailChannels:
            return self._window_jets(lo, hi, values[at], 0)[0], None, None

        return readonly(log_window_values(lower, upper, len(values), window, support_lower=0.0))

    def cdf(self, y: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        if self.parametrisation == "mean" and not np.all(mean_exists(values[:, 1], values[:, 2])):
            raise GeneralizedGammaDomainError("generalized gamma mean does not exist for every row")
        response = np.broadcast_to(np.asarray(y, dtype=np.float64), (len(values),))
        if not np.all(np.isfinite(response)):
            raise GeneralizedGammaDomainError("y must be finite")
        out = np.zeros_like(response)
        interior = response > 0.0
        if np.any(interior):
            inside_values = values[interior]
            out[interior] = generalized_gamma_cdf(
                response[interior],
                self._location_coordinates(inside_values),
                inside_values[:, 1],
                inside_values[:, 2],
            )
        return readonly(out)

    def quantile(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        return generalized_gamma_quantile(
            probabilities, self._location_coordinates(values), values[:, 1], values[:, 2]
        )

    def expected_shortfall(self, p: NDArray, theta: NDArray) -> NDArray[np.float64]:
        """``E[Y | Y > q_p]`` per row in either natural parametrisation."""
        values = self._theta(theta, None)
        probabilities = np.broadcast_to(np.asarray(p, dtype=np.float64), (len(values),))
        if self.parametrisation == "mean":
            return generalized_gamma_expected_shortfall_from_mean(
                probabilities,
                values[:, 0],
                values[:, 1],
                values[:, 2],
            )
        return generalized_gamma_expected_shortfall(
            probabilities,
            self._location_coordinates(values),
            values[:, 1],
            values[:, 2],
        )

    def diagnose_repeated_curvature_failure(
        self,
        y: NDArray,
        weights: ResolvedLikelihoodWeights,
    ) -> Exception | None:
        del y, weights
        if self.parametrisation != "mean":
            return None
        return GeneralizedGammaDomainError(
            "GeneralizedGammaLSS curvature failed repeatedly in the mean form; a likely cause "
            "is a cell pressing against the infinite-mean boundary (shape < 0 with "
            "scale * |shape| -> 1). Inspect predict_parameters() for rows with scale * |shape| "
            "near 1, or fit GeneralizedGammaLSS(parametrisation='location') for tail work "
            "where an infinite mean is legitimate"
        )


_register_likelihood_reuse_contract(
    GeneralizedGammaLSS,
    GeneralizedGammaLikelihoodPlan,
    prepared_array_fields=("exact_response", "parameter_independent_carrier"),
    link_types=(LowerBoundedLogLink,),
)
