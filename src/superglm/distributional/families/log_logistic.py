"""Log-logistic severity with scale/median and positive tail exponent."""

from dataclasses import dataclass
from typing import ClassVar

from superglm.distributional.families._declared import (
    DeclaredDirectionsMixin,
    DeclaredDistributionMixin,
    DeclaredPredictionMixin,
    DeclaredScalarFamily,
    DeclaredScalarPlan,
    DeclaredScalarSpec,
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
from superglm.distributional.likelihood_identity import LOG_ERROR_UNITS
from superglm.links import LogLink

_SPEC = DeclaredScalarSpec(
    law_key="log-logistic/scale-tail/v1",
    schema_version=1,
    kernel_module="superglm.distributional.kernels.log_logistic",
    parameters=tuple(
        ParameterSpec(name, LogLink(), name, ParameterSupport(lower=0.0), "observed")
        for name in ("scale", "shape")
    ),
    response_lower=0.0,
    response_lower_inclusive=False,
    capabilities=FamilyCapabilities(2, False, True, True, False, True, censored_response=True),
    default_prediction_name="conditional_mean",
    carrier_rounding_parameters=(LOG_ERROR_UNITS, 0.0),  # -mass*log(y), no constant term
    supported_operations=(
        "cdf",
        "quantile",
        "default_prediction",
        "variance",
        "log_window_probability",
        "predictor_curvature_directional_derivative",
        "predictor_curvature_second_directional_derivative",
    ),
)


def initialize_parameters(y, plan):
    return positive_endpoint_start(y, plan, scale_factor=1.0)


@dataclass(frozen=True)
class LogLogisticLSS(
    DeclaredDistributionMixin,
    DeclaredPredictionMixin,
    DeclaredWindowMixin,
    DeclaredDirectionsMixin,
    DeclaredScalarFamily,
    ScalePredictor,
    ShapePredictor,
):
    """Fisk(scale, shape) on y>0 with log links; scale is the median.

    No expected-information or expected-shortfall capability is supplied.
    Mean and variance are infinite at shape<=1 and shape<=2 respectively.
    """

    spec: ClassVar[DeclaredScalarSpec] = _SPEC

    def initialize(self, y, plan):
        return initialize_parameters(self.validate_likelihood_plan(y, plan), plan)


_register_likelihood_reuse_contract(
    LogLogisticLSS,
    DeclaredScalarPlan,
    prepared_array_fields=("exact_response", "parameter_independent_carrier"),
    deterministic_chunk_replay=True,
)
