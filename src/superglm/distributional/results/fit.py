"""The published distributional fit result."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.results.solver import _frozen_float_mapping, _readonly_finite
from superglm.distributional.telemetry import CurvatureTelemetry


def _frozen_array_mapping(
    values: Mapping[str, NDArray],
    *,
    name: str,
) -> Mapping[str, NDArray[np.float64]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, NDArray[np.float64]] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        array = _readonly_finite(value, name=f"{name}[{key!r}]")
        if array.ndim != 1:
            raise ValueError(f"{name} values must be one-dimensional")
        result[key] = array
    return MappingProxyType(result)


@dataclass(frozen=True)
class DistributionalFitResult:
    """Compact immutable public-scale state for one complete fitted revision."""

    coefficients: NDArray[np.float64]
    coefficient_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    predictor_coefficients: Mapping[str, NDArray[np.float64]]
    smoothing_parameters: Mapping[str, float]
    covariance: NDArray[np.float64]
    total_effective_df: float
    predictor_edf: Mapping[str, float]
    intercept_edf: Mapping[str, float]
    term_edf: Mapping[str, float]
    log_likelihood: float
    penalized_log_likelihood: float
    null_objective: float
    converged: bool
    coefficient_converged: bool
    smoothing_converged: bool | None
    n_inner_iter: int
    n_smoothing_iter: int
    rank: int
    curvature_telemetry: CurvatureTelemetry
    exact_face_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        coefficients = _readonly_finite(self.coefficients, name="coefficients")
        covariance = _readonly_finite(self.covariance, name="covariance")
        if coefficients.ndim != 1 or covariance.shape != (len(coefficients), len(coefficients)):
            raise ValueError("coefficients and covariance must share one global coordinate width")
        covariance_tolerance = (
            64.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.linalg.norm(covariance, ord=np.inf)),
            )
        )
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=covariance_tolerance):
            raise ValueError("covariance must be symmetric")

        coefficient_names = tuple(self.coefficient_names)
        parameter_names = tuple(self.parameter_names)
        if (
            len(coefficient_names) != len(coefficients)
            or len(set(coefficient_names)) != len(coefficient_names)
            or any(not isinstance(name, str) or not name for name in coefficient_names)
        ):
            raise ValueError("coefficient_names must uniquely name every coefficient")
        if (
            not parameter_names
            or len(set(parameter_names)) != len(parameter_names)
            or any(not isinstance(name, str) or not name for name in parameter_names)
        ):
            raise ValueError("parameter_names must be unique non-empty strings")
        predictor_coefficients = _frozen_array_mapping(
            self.predictor_coefficients,
            name="predictor_coefficients",
        )
        if tuple(predictor_coefficients) != parameter_names:
            raise ValueError("predictor coefficient views must follow parameter order")
        concatenated = np.concatenate(tuple(predictor_coefficients.values()))
        if not np.array_equal(concatenated, coefficients):
            raise ValueError("predictor coefficient views must partition coefficients")

        smoothing = _frozen_float_mapping(
            self.smoothing_parameters,
            name="smoothing_parameters",
            nonnegative=True,
        )
        exact_face_components = tuple(self.exact_face_components)
        exact_face_set = set(exact_face_components)
        if (
            any(
                not isinstance(name, str) or name not in smoothing for name in exact_face_components
            )
            or len(exact_face_set) != len(exact_face_components)
            or tuple(name for name in smoothing if name in exact_face_set) != exact_face_components
        ):
            raise ValueError("exact_face_components must be unique smoothing names in fitted order")
        predictor_edf = _frozen_float_mapping(self.predictor_edf, name="predictor_edf")
        if tuple(predictor_edf) != parameter_names:
            raise ValueError("predictor EDF must follow parameter order")
        intercept_edf = _frozen_float_mapping(self.intercept_edf, name="intercept_edf")
        term_edf = _frozen_float_mapping(self.term_edf, name="term_edf")

        for name in (
            "total_effective_df",
            "log_likelihood",
            "penalized_log_likelihood",
            "null_objective",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("converged", "coefficient_converged"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.smoothing_converged is not None and not isinstance(self.smoothing_converged, bool):
            raise TypeError("smoothing_converged must be bool or None")
        expected_converged = self.coefficient_converged and self.smoothing_converged is not False
        if self.converged != expected_converged:
            raise ValueError("overall convergence must combine coefficient and smoothing state")
        for name in ("n_inner_iter", "n_smoothing_iter", "rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.rank > len(coefficients):
            raise ValueError("rank cannot exceed coefficient width")
        if not isinstance(self.curvature_telemetry, CurvatureTelemetry):
            raise TypeError("curvature_telemetry must be CurvatureTelemetry")

        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "coefficient_names", coefficient_names)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "predictor_coefficients", predictor_coefficients)
        object.__setattr__(self, "smoothing_parameters", smoothing)
        object.__setattr__(self, "exact_face_components", exact_face_components)
        object.__setattr__(self, "predictor_edf", predictor_edf)
        object.__setattr__(self, "intercept_edf", intercept_edf)
        object.__setattr__(self, "term_edf", term_edf)
