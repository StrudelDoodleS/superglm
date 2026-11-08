"""Internal immutable scalar-law specification and reusable family mechanics.

Named classes select static specifications and explicit optional mixins. This
is not a runtime family registry or a public arbitrary-law factory.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from typing import ClassVar

import numpy as np

from superglm.distributional.families._base import (
    array_digest,
    immutable,
    natural_curvature_direction,
    natural_curvature_second_direction,
    readonly,
    typed_plan,
    validated_float_response,
    validated_observation_response,
)
from superglm.distributional.families._response_bounds import (
    bind_observation,
    bounded_channels,
    bounded_higher_derivatives,
    censored_carrier,
    contract_third,
    log_window_values,
    refuse_bounded,
)
from superglm.distributional.family import (
    FamilyCapabilities,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    validated_derivative_order,
    validated_parameter_matrix,
)
from superglm.distributional.likelihood_identity import CarrierRounding, carrier_rounding_scale
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    _readonly_array,
)


@dataclass(frozen=True)
class DeclaredScalarSpec:
    law_key: str
    schema_version: int
    kernel_module: str
    parameters: tuple[ParameterSpec, ...]
    response_lower: float
    response_lower_inclusive: bool
    capabilities: FamilyCapabilities
    default_prediction_name: str | None = None
    supported_operations: tuple[str, ...] = ()
    carrier_rounding_parameters: tuple[int, float] | None = None

    def __post_init__(self):
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("unsupported declared scalar schema version")
        if not self.law_key or not self.kernel_module or not self.parameters:
            raise ValueError("declared scalar specifications require law, kernel and parameters")
        names = tuple(p.name for p in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("declared parameter names must be unique")
        if self.capabilities.max_derivative_order != 2:
            raise ValueError("declared scalar core evaluations support orders zero through two")
        if self.carrier_rounding_parameters is not None:
            units, beta = self.carrier_rounding_parameters
            if isinstance(units, bool) or not isinstance(units, int) or units < 0:
                raise ValueError("carrier error units must be a nonnegative integer")
            if not np.isfinite(beta) or beta < 0:
                raise ValueError("carrier rounding beta must be finite and nonnegative")


@dataclass(frozen=True)
class DeclaredScalarPlan:
    weights: ResolvedLikelihoodWeights
    row_law: str
    law_key: str
    schema_version: int
    family_config: str
    observation: ObservationContract
    exact_response: np.ndarray
    response_digest: str
    parameter_independent_carrier: np.ndarray
    carrier_digest: str
    carrier_rounding_parameters: tuple[int, float] | None

    @classmethod
    def _from_prepared(
        cls,
        *,
        weights,
        row_law,
        law_key,
        schema_version,
        family_config,
        observation,
        exact_response,
        carrier,
        carrier_rounding_parameters=None,
    ):
        if schema_version != 1 or isinstance(schema_version, bool):
            raise UnsupportedLikelihoodContractError(
                "unsupported declared scalar plan schema version"
            )
        response, carrier = immutable(exact_response), immutable(carrier)
        return cls(
            weights,
            row_law,
            law_key,
            schema_version,
            family_config,
            observation,
            response,
            array_digest(b"DeclaredScalar/response/v1\0", response),
            carrier,
            array_digest(b"DeclaredScalar/carrier/v1\0", carrier),
            carrier_rounding_parameters,
        )

    @property
    def plan_identifier(self):
        payload = "\0".join(
            (
                "DeclaredScalar/v1",
                self.law_key,
                str(self.schema_version),
                self.family_config,
                self.row_law,
                self.weights.digest,
                self.response_digest,
                # Transcendental carriers can differ across CPUs/libm. Their
                # stored sums use the existing rounding certificate below;
                # uncertified carriers retain exact identity instead.
                self.carrier_digest if self.carrier_rounding_parameters is None else "",
                repr(self.carrier_rounding_parameters),
                self.observation.kind,
                str(self.observation.schema_version),
                "" if self.observation.bounds is None else self.observation.bounds.digest,
            )
        )
        return "DeclaredScalar/v1:" + hashlib.sha256(payload.encode()).hexdigest()

    @property
    def multiplier(self):
        # Both admitted contracts have exactly this multiplier: prior has
        # already been required to be all ones. No row-law/name switch.
        return self.weights.values

    @property
    def carrier_rounding(self):
        # The existing replay boundary explicitly accepts None for plans
        # without this optional certificate. Never infer one for an arbitrary
        # carrier merely because its values are finite.
        return None if self.carrier_rounding_parameters is None else self._carrier_rounding

    def _carrier_rounding(self):
        units, beta = self.carrier_rounding_parameters
        carrier = self.parameter_independent_carrier
        scale = carrier_rounding_scale(carrier, self.multiplier, beta)
        return CarrierRounding(carrier, scale, units)

    def take(self, indices):
        selected = self.weights.take(indices)  # validates the positional selection
        return type(self)._from_prepared(
            weights=selected,
            row_law=self.row_law,
            law_key=self.law_key,
            schema_version=self.schema_version,
            family_config=self.family_config,
            observation=self.observation.take(indices),
            exact_response=self.exact_response[indices],
            carrier=self.parameter_independent_carrier[indices],
            carrier_rounding_parameters=self.carrier_rounding_parameters,
        )

    def __reduce__(self):
        return (
            _restore_plan,
            (
                dict(
                    weights=self.weights,
                    row_law=self.row_law,
                    law_key=self.law_key,
                    schema_version=self.schema_version,
                    family_config=self.family_config,
                    observation=self.observation,
                    exact_response=self.exact_response,
                    carrier=self.parameter_independent_carrier,
                    carrier_rounding_parameters=self.carrier_rounding_parameters,
                ),
            ),
        )


def _restore_plan(state):
    # NumPy's ordinary pickle round trip re-arms arrays. Preserve weight
    # provenance/digests, but restore their immutable buffers as well as the
    # response, carrier and observation bounds. No process registry is needed.
    state = dict(state)
    w = state["weights"]
    fields = (
        "values",
        "geometry_values",
        "root_take_map",
        "input_positions",
        "dropped_input_positions",
    )
    state["weights"] = replace(w, **{key: _readonly_array(getattr(w, key)) for key in fields})
    obs = state["observation"]
    if obs.bounds is not None:
        state["observation"] = obs.take(np.arange(obs.bounds.n_rows))
    return DeclaredScalarPlan._from_prepared(**state)


class DeclaredScalarFamily:
    spec: ClassVar[DeclaredScalarSpec]

    def __post_init__(self):
        # Binding snapshots require independently owned default links; the
        # specification remains a static constant shared by named classes.
        object.__setattr__(self, "_parameters", deepcopy(self.spec.parameters))
        kernel = self._kernel
        if kernel.COORDINATES != "natural" or kernel.PARAMETERS != tuple(
            p.name for p in self.parameters
        ):
            raise ValueError(
                "declared natural coordinates/parameter order disagree with the kernel"
            )
        for operation in self.spec.supported_operations:
            if not callable(getattr(self, operation, None)):
                raise TypeError(f"declared operation {operation!r} has no implementation")
        for flag, operation in (
            ("cdf", "cdf"),
            ("quantile", "quantile"),
            ("response_mean", "default_prediction"),
            ("expected_information", "expected_information_natural"),
            ("censored_response", "log_window_probability"),
        ):
            if getattr(self.capabilities, flag) != callable(getattr(self, operation, None)):
                raise TypeError(
                    f"declared capability {flag!r} disagrees with the concrete protocol"
                )

    @property
    def parameters(self):
        return self._parameters

    @property
    def capabilities(self):
        return self.spec.capabilities

    @property
    def _kernel(self):
        return import_module(self.spec.kernel_module)

    def to_config(self):
        return {
            "type": type(self).__name__,
            "law_key": self.spec.law_key,
            "schema_version": self.spec.schema_version,
            "parametrisation": tuple(p.name for p in self.parameters),
        }

    @property
    def _family_config(self):
        # Include static numerical/support/capability choices as well as the
        # concrete class's public configuration. Artifact digests are build
        # evidence, not semantic identity.
        static = {
            "kernel": self.spec.kernel_module,
            "support": (self.spec.response_lower, self.spec.response_lower_inclusive),
            "capabilities": asdict(self.capabilities),
            "operations": self.spec.supported_operations,
            "prediction": self.spec.default_prediction_name,
            "carrier_rounding": self.spec.carrier_rounding_parameters,
            "parameters": [
                (
                    p.name,
                    p.role,
                    asdict(p.support),
                    p.curvature,
                    type(p.default_link).__module__,
                    type(p.default_link).__qualname__,
                    vars(p.default_link),
                )
                for p in self.parameters
            ],
        }
        return json.dumps((self.to_config(), static), sort_keys=True)

    def _response(self, y, observation=None, weights=None):
        def exact_response(values):
            response = validated_float_response(
                values,
                message=f"{type(self).__name__} requires a non-empty finite response vector in its support",
                lower=self.spec.response_lower,
                lower_inclusive=True,
            )
            if not self.spec.response_lower_inclusive and np.any(
                response == self.spec.response_lower
            ):
                raise ValueError(
                    f"{type(self).__name__} exact response is outside its open support"
                )
            return response

        bounds = getattr(observation, "bounds", None)
        if bounds is None:
            return exact_response(y)
        return validated_observation_response(
            y,
            censored=bounds.side != 0,
            exact_validator=exact_response,
            input_positions=getattr(weights, "input_positions", None),
        )

    def _density_response(self, response, observation):
        """Protect density kernels from censored limits outside their support."""
        bounds = observation.bounds
        if bounds is None:
            return response
        replace = (bounds.side != 0) & (response <= self.spec.response_lower)
        return (
            np.where(replace, self.spec.response_lower + 1.0, response)
            if np.any(replace)
            else response
        )

    def _theta(self, theta, n_observations=None):
        return validated_parameter_matrix(
            theta,
            n_observations=n_observations,
            parameters=self.parameters,
            family_name=type(self).__name__,
        )

    def bind_likelihood(self, y, weights, observation):
        response = self._response(y, observation, weights)
        if (
            not isinstance(weights, ResolvedLikelihoodWeights)
            or weights.values.shape != response.shape
        ):
            raise UnsupportedLikelihoodContractError(
                "declared response rows require matching resolved likelihood weights"
            )
        semantics = weights.provenance.contract.semantics
        if semantics == "prior" and not weights.provenance.all_unit:
            raise UnsupportedLikelihoodContractError(
                f"{type(self).__name__} has no non-unit prior weight law; use integer frequency replication"
            )
        if semantics not in ("frequency", "prior"):
            raise UnsupportedLikelihoodContractError(
                "unsupported declared likelihood weight semantics"
            )
        if not self.capabilities.censored_response and not observation.is_complete:
            raise UnsupportedLikelihoodContractError(
                "this declared law supports complete observations only"
            )
        observation = bind_observation(
            observation,
            response,
            weights,
            support_lower=self.spec.response_lower,
            family_name=type(self).__name__,
        )
        carrier = censored_carrier(
            self._kernel.carrier(self._density_response(response, observation), weights.values),
            observation.bounds,
        )
        return DeclaredScalarPlan._from_prepared(
            weights=weights,
            row_law=f"{semantics}/literal-replication/v1",
            law_key=self.spec.law_key,
            schema_version=self.spec.schema_version,
            family_config=self._family_config,
            observation=observation,
            exact_response=response,
            carrier=carrier,
            carrier_rounding_parameters=self.spec.carrier_rounding_parameters,
        )

    def _plan(self, plan, n_observations):
        candidate = typed_plan(
            plan, DeclaredScalarPlan, n_observations, family_name=type(self).__name__
        )
        if (
            candidate.law_key != self.spec.law_key
            or candidate.schema_version != self.spec.schema_version
            or candidate.family_config != self._family_config
            or candidate.carrier_rounding_parameters != self.spec.carrier_rounding_parameters
        ):
            raise UnsupportedLikelihoodContractError(
                "declared likelihood was prepared for another law, schema or configuration"
            )
        return candidate

    def validate_likelihood_plan(self, y, plan):
        response = self._response(
            y, getattr(plan, "observation", None), getattr(plan, "weights", None)
        )
        bound = self._plan(plan, len(response))
        owned = bound.exact_response
        if owned.flags.writeable or owned.shape != response.shape or owned.dtype != np.float64:
            raise UnsupportedLikelihoodContractError(
                "declared likelihood owns an invalid bound response"
            )
        if not np.array_equal(response, owned):
            raise UnsupportedLikelihoodContractError(
                "declared likelihood response does not match the fitted response"
            )
        return owned

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        order = validated_derivative_order(derivative_order)
        response = self.validate_likelihood_plan(y, plan)
        theta = self._theta(theta, len(response))
        bound = self._plan(plan, len(response))
        result = self._kernel.density_jets(
            self._density_response(response, bound.observation),
            theta,
            bound.multiplier,
            order=order,
        )
        channels = [*result.jets, None, None][:3]
        valid = result.valid
        if bound.observation.bounds is not None:

            def window(lo, hi, rows):
                jets, accepted = self._kernel.window_jets(lo, hi, theta[rows], order=order)
                # Core probability validity is separate from optional higher
                # directional admission. Invalid windows must reach the
                # existing bounded-channel finite-placeholder convention.
                jets = tuple(
                    np.where(accepted if j == 0 else accepted[:, None], x, np.nan)
                    for j, x in enumerate(jets)
                )
                return (*jets, None, None)[:3]

            *channels, valid = bounded_channels(
                *channels,
                valid,
                response=response,
                bounds=bound.observation.bounds,
                multiplier=bound.multiplier,
                window=window,
                derivative_order=order,
            )
        # Finite placeholders preserve the shared invalid-row convention.
        for j, channel in enumerate(channels):
            if channel is not None:
                channels[j] = np.where(valid if j == 0 else valid[:, None], channel, 0.0)
        return NaturalLikelihoodEvaluation(
            *channels[:1], bound.parameter_independent_carrier, *channels[1:], valid=valid
        )


class DeclaredWindowMixin:
    def log_window_probability(self, lower, upper, theta, weights=None):
        if weights is not None and not np.all(np.asarray(weights) == 1.0):
            raise UnsupportedLikelihoodContractError("no non-unit prior weighted window law")
        values = self._theta(theta)

        def window(lo, hi, rows):
            jets, valid = self._kernel.window_jets(lo, hi, values[rows], order=0)
            if not np.all(valid):
                raise ValueError("window probability is unresolved in the numerical range")
            return jets[0], None, None

        return readonly(
            log_window_values(
                lower, upper, len(values), window, support_lower=self.spec.response_lower
            )
        )


class DeclaredDistributionMixin:
    def cdf(self, y, theta):
        return self._kernel.cdf(y, self._theta(theta))

    def quantile(self, p, theta):
        return self._kernel.quantile(p, self._theta(theta))


class DeclaredPredictionMixin:
    @property
    def default_prediction_name(self):
        return self.spec.default_prediction_name

    def default_prediction(self, theta):
        return self._kernel.mean(self._theta(theta))

    def variance(self, theta):
        return self._kernel.variance(self._theta(theta))


class DeclaredInformationMixin:
    def expected_information_natural(self, theta, plan):
        values = self._theta(theta)
        bound = self._plan(plan, len(values))
        refuse_bounded(bound.observation, "declared complete-row expected information")
        return self._kernel.expected_information(values, bound.multiplier)


class DeclaredShortfallMixin:
    def expected_shortfall(self, p, theta):
        return self._kernel.expected_shortfall(p, self._theta(theta))


class DeclaredDirectionsMixin:
    stacked_curvature_directions: ClassVar[bool] = True

    def _higher(self, y, theta, plan, order):
        response = self.validate_likelihood_plan(y, plan)
        result = self._kernel.density_jets(
            self._density_response(response, plan.observation), theta, plan.multiplier, order=order
        )
        third, fourth = result.jets[3], result.jets[4] if order == 4 else None
        valid = result.valid
        if plan.observation.bounds is not None:
            valid = valid.copy()
            valid[plan.observation.bounds.censored_rows()] = True

            def windows(lo, hi, rows, order):
                if not np.all(self._kernel.analytic_window_admitted(lo, hi, theta[rows])):
                    raise UnsupportedLikelihoodContractError(
                        "window has no admitted analytic higher direction (quadrature or cancellation); difference the order-two likelihood"
                    )
                jets, admitted = self._kernel.window_jets(lo, hi, theta[rows], order=order)
                if not np.all(admitted):
                    raise UnsupportedLikelihoodContractError(
                        "window higher derivatives are outside the numerical envelope"
                    )
                return jets

            third, fourth, valid = bounded_higher_derivatives(
                third,
                fourth,
                valid,
                response=response,
                bounds=plan.observation.bounds,
                multiplier=plan.multiplier,
                window_jets=windows,
            )
        if not np.all(valid):
            raise UnsupportedLikelihoodContractError(
                "density higher derivatives are outside the numerical envelope"
            )
        return third, fourth

    def predictor_curvature_directional_derivative(self, y, eta, eta_direction, links, plan):
        def direction(theta, natural):
            third, _ = self._higher(y, theta, plan, 3)
            return contract_third(third, natural)

        return natural_curvature_direction(self, y, eta, eta_direction, links, plan, direction)

    def predictor_curvature_second_directional_derivative(
        self, y, eta, direction_a, direction_b, links, plan
    ):
        def directions(theta, natural_a, natural_b):
            return self._higher(y, theta, plan, 4)

        return natural_curvature_second_direction(
            self, y, eta, direction_a, direction_b, links, plan, directions
        )


def positive_endpoint_start(y, plan, *, scale_factor):
    """Frequency-aware median start, never an imputed likelihood response."""
    values = np.asarray(y)
    mass = plan.multiplier
    positive = values > 0
    if not positive.any() and plan.observation.bounds is not None:
        bounds = plan.observation.bounds
        values = np.concatenate((bounds.lower, bounds.upper))
        mass = np.tile(mass, 2)
        positive = np.isfinite(values) & (values > 0)
    if not positive.any():
        warnings.warn(
            "No positive response information: unit-scale start; a finite optimum is not identified",
            UserWarning,
            stacklevel=3,
        )
        scale = 1.0
    else:
        values, mass = values[positive], mass[positive]
        positions = np.argsort(values, kind="stable")
        cumulative = np.cumsum(mass[positions])
        scale = values[positions[np.searchsorted(cumulative, 0.5 * cumulative[-1])]] * scale_factor
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("finite initialization scale is outside the numerical range")
    return InitialParameterState(np.tile([scale, 2.0], (len(y), 1)))
