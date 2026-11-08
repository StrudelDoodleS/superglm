"""Lomax severity with positive scale and unrestricted positive tail exponent."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from superglm.distributional.families._declared import (
    DeclaredDirectionsMixin,
    DeclaredDistributionMixin,
    DeclaredInformationMixin,
    DeclaredPredictionMixin,
    DeclaredScalarFamily,
    DeclaredScalarPlan,
    DeclaredScalarSpec,
    DeclaredShortfallMixin,
    DeclaredWindowMixin,
    positive_endpoint_start,
)
from superglm.distributional.families._predictors import ScalePredictor, ShapePredictor
from superglm.distributional.family import (
    FamilyCapabilities,
    ParameterSpec,
    ParameterSupport,
    _register_likelihood_reuse_contract,
)
from superglm.links import LogLink

_SPEC = DeclaredScalarSpec(
    law_key="lomax/scale-tail/v1",
    schema_version=1,
    kernel_module="superglm.distributional.kernels.lomax",
    parameters=tuple(
        ParameterSpec(name, LogLink(), name, ParameterSupport(lower=0.0), "observed")
        for name in ("scale", "shape")
    ),
    response_lower=0.0,
    response_lower_inclusive=True,
    capabilities=FamilyCapabilities(2, True, True, True, False, True, censored_response=True),
    default_prediction_name="conditional_mean",
    carrier_rounding_parameters=(0, 0.0),  # identically zero, no transcendental error
    supported_operations=(
        "log_window_probability",
        "predictor_curvature_directional_derivative",
        "predictor_curvature_second_directional_derivative",
        "cdf",
        "quantile",
        "default_prediction",
        "variance",
        "expected_information_natural",
        "expected_shortfall",
    ),
)


def initialize_parameters(y, plan):
    return positive_endpoint_start(y, plan, scale_factor=1 / (np.sqrt(2) - 1))


@dataclass(frozen=True)
class LomaxLSS(
    DeclaredDistributionMixin,
    DeclaredPredictionMixin,
    DeclaredInformationMixin,
    DeclaredShortfallMixin,
    DeclaredWindowMixin,
    DeclaredDirectionsMixin,
    DeclaredScalarFamily,
    ScalePredictor,
    ShapePredictor,
):
    """Lomax(scale, shape) on y >= 0; both predictors use positive log links."""

    spec: ClassVar[DeclaredScalarSpec] = _SPEC

    def initialize(self, y, plan):
        response = self.validate_likelihood_plan(y, plan)
        return initialize_parameters(response, plan)


_register_likelihood_reuse_contract(
    LomaxLSS,
    DeclaredScalarPlan,
    prepared_array_fields=("exact_response", "parameter_independent_carrier"),
    deterministic_chunk_replay=True,
)
