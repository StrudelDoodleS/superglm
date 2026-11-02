"""The published distributional fit result."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from superglm.distributional._common import frozen_float_mapping
from superglm.distributional.results.solver import _readonly_finite
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
    # Disclosures a fit makes instead of refusing, each also warned at fit time.
    warnings: tuple[str, ...] = ()
    fit_id: str = field(default_factory=lambda: str(uuid4()), repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "total_effective_df",
            "log_likelihood",
            "penalized_log_likelihood",
            "null_objective",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self, "coefficients", _readonly_finite(self.coefficients, name="coefficients")
        )
        object.__setattr__(self, "covariance", _readonly_finite(self.covariance, name="covariance"))
        object.__setattr__(self, "coefficient_names", tuple(self.coefficient_names))
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(
            self,
            "predictor_coefficients",
            _frozen_array_mapping(self.predictor_coefficients, name="predictor_coefficients"),
        )
        object.__setattr__(
            self,
            "smoothing_parameters",
            frozen_float_mapping(
                self.smoothing_parameters, name="smoothing_parameters", nonnegative=True
            ),
        )
        object.__setattr__(self, "exact_face_components", tuple(self.exact_face_components))
        for name in ("predictor_edf", "intercept_edf", "term_edf"):
            object.__setattr__(self, name, frozen_float_mapping(getattr(self, name), name=name))
