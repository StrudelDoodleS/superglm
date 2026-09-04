"""Primitive Gaussian location-scale numerical kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._common import (
    WeightSemantics,
    float64_vector,
    positive_weights,
    readonly,
    readonly_bool,
    validated_derivative_order,
    validated_semantics,
)


def _scale_floor(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("scale_floor must be finite and non-negative")
    try:
        floor = float(value)  # ty: ignore[invalid-argument-type] -- validated conversion boundary
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scale_floor must be finite and non-negative") from exc
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("scale_floor must be finite and non-negative")
    return floor


def _aligned_vectors(
    *values_and_names: tuple[NDArray, str],
) -> tuple[NDArray[np.float64], ...]:
    arrays = tuple(float64_vector(values, name) for values, name in values_and_names)
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise ValueError("Gaussian row arrays must have the same shape")
    return arrays


@dataclass(frozen=True)
class GaussianKernelEvaluation:
    optimizing_log_likelihood: NDArray[np.float64]
    score: NDArray[np.float64] | None
    hessian_packed: NDArray[np.float64] | None
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        optimizing = np.asarray(self.optimizing_log_likelihood)
        if optimizing.ndim != 1 or not np.all(np.isfinite(optimizing)):
            raise ValueError("optimizing_log_likelihood must be a finite one-dimensional array")
        n_rows = len(optimizing)
        object.__setattr__(self, "optimizing_log_likelihood", readonly(optimizing))

        valid = np.asarray(self.valid)
        if valid.dtype != np.dtype(np.bool_) or valid.shape != (n_rows,):
            raise ValueError("valid must be a one-dimensional boolean array matching rows")
        object.__setattr__(self, "valid", readonly_bool(valid))

        for name, width in (("score", 2), ("hessian_packed", 3)):
            values = getattr(self, name)
            if values is None:
                continue
            array = np.asarray(values)
            if array.shape != (n_rows, width) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite ({n_rows}, {width}) array matching rows")
            object.__setattr__(self, name, readonly(array))


def initialize_gaussian(
    response: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    scale_floor: float,
) -> NDArray[np.float64]:
    response_values, weight_values = _aligned_vectors(
        (response, "response"),
        (weights, "weights"),
    )
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(weight_values, weight_semantics)
    floor = _scale_floor(scale_floor)
    weight_sum = float(np.sum(weight_values))
    location = float(np.dot(weight_values, response_values) / weight_sum)
    residual = response_values - location
    normalizer_mass = (
        len(weight_values)
        if weight_semantics == "prior"
        else sum(int(weight) for weight in weight_values)
    )
    raw_scale = math.sqrt(float(np.dot(weight_values, residual * residual) / normalizer_mass))
    margin = max(1.0e-8, np.sqrt(np.finfo(np.float64).eps) * max(1.0, abs(location)))
    scale = max(raw_scale, floor + margin)
    theta = np.column_stack(
        (
            np.full(len(response_values), location),
            np.full(len(response_values), scale),
        )
    )
    return readonly(theta)


def _gaussian_prior_row_channels(
    base_normalizer: NDArray[np.float64],
    residual: NDArray[np.float64],
    residual_2: NDArray[np.float64],
    inverse_scale: NDArray[np.float64],
    inverse_scale_2: NDArray[np.float64],
    weights: NDArray[np.float64],
    derivative_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None]:
    score = None
    hessian = None
    optimizing = base_normalizer - 0.5 * weights * residual_2 * inverse_scale_2
    if derivative_order >= 1:
        score = np.column_stack(
            (
                weights * residual * inverse_scale_2,
                -inverse_scale + weights * residual_2 * inverse_scale**3,
            )
        )
    if derivative_order == 2:
        hessian = np.column_stack(
            (
                -weights * inverse_scale_2,
                -2.0 * weights * residual * inverse_scale**3,
                inverse_scale_2 - 3.0 * weights * residual_2 * inverse_scale**4,
            )
        )
    return optimizing, score, hessian


def _gaussian_frequency_row_channels(
    base_normalizer: NDArray[np.float64],
    residual: NDArray[np.float64],
    residual_2: NDArray[np.float64],
    inverse_scale: NDArray[np.float64],
    inverse_scale_2: NDArray[np.float64],
    weights: NDArray[np.float64],
    derivative_order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None]:
    score = None
    hessian = None
    optimizing = weights * (base_normalizer - 0.5 * residual_2 * inverse_scale_2)
    if derivative_order >= 1:
        score = weights[:, None] * np.column_stack(
            (
                residual * inverse_scale_2,
                -inverse_scale + residual_2 * inverse_scale**3,
            )
        )
    if derivative_order == 2:
        hessian = weights[:, None] * np.column_stack(
            (
                -inverse_scale_2,
                -2.0 * residual * inverse_scale**3,
                inverse_scale_2 - 3.0 * residual_2 * inverse_scale**4,
            )
        )
    return optimizing, score, hessian


def evaluate_gaussian_rows(
    response: NDArray,
    location: NDArray,
    scale: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    *,
    derivative_order: int,
) -> GaussianKernelEvaluation:
    response_values, location_values, scale_values, weight_values = _aligned_vectors(
        (response, "response"),
        (location, "location"),
        (scale, "scale"),
        (weights, "weights"),
    )
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(weight_values, weight_semantics)
    order = validated_derivative_order(derivative_order)
    if np.any(scale_values <= 0.0):
        raise ValueError("scale must be finite and strictly positive")
    residual = response_values - location_values
    inverse_scale = 1.0 / scale_values
    inverse_scale_2 = inverse_scale * inverse_scale
    residual_2 = residual * residual
    base_normalizer = -np.log(scale_values) - 0.5 * np.log(2.0 * np.pi)
    if weight_semantics == "prior":
        optimizing, score, hessian = _gaussian_prior_row_channels(
            base_normalizer=base_normalizer,
            residual=residual,
            residual_2=residual_2,
            inverse_scale=inverse_scale,
            inverse_scale_2=inverse_scale_2,
            weights=weight_values,
            derivative_order=order,
        )
    else:
        optimizing, score, hessian = _gaussian_frequency_row_channels(
            base_normalizer=base_normalizer,
            residual=residual,
            residual_2=residual_2,
            inverse_scale=inverse_scale,
            inverse_scale_2=inverse_scale_2,
            weights=weight_values,
            derivative_order=order,
        )
    return GaussianKernelEvaluation(
        optimizing_log_likelihood=optimizing,
        score=score,
        hessian_packed=hessian,
        valid=np.ones(len(response_values), dtype=bool),
    )


def gaussian_expected_information(
    scale: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    scale_values, weight_values = _aligned_vectors(
        (scale, "scale"),
        (weights, "weights"),
    )
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(weight_values, weight_semantics)
    if np.any(scale_values <= 0.0):
        raise ValueError("scale must be finite and strictly positive")
    inverse_scale_2 = 1.0 / scale_values**2
    if weight_semantics == "prior":
        information = np.column_stack(
            (
                weight_values * inverse_scale_2,
                np.zeros(len(scale_values)),
                2.0 * inverse_scale_2,
            )
        )
    else:
        information = weight_values[:, None] * np.column_stack(
            (
                inverse_scale_2,
                np.zeros(len(scale_values)),
                2.0 * inverse_scale_2,
            )
        )
    return readonly(information)


def _gaussian_predictor_curvature_channels(
    response: NDArray[np.float64],
    predictors: NDArray[np.float64],
    direction: NDArray[np.float64],
    scale: NDArray[np.float64],
    scale_increment: NDArray[np.float64],
    weights: NDArray[np.float64],
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    if semantics == "prior":
        normalizer_mass = np.ones(len(response), dtype=np.float64)
    else:
        normalizer_mass = weights
    residual_mass = weights
    residual = response - predictors[:, 0]
    location_direction = direction[:, 0]
    scale_direction = direction[:, 1]
    result = np.empty((len(response), 3), dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse_scale = 1.0 / scale
        inverse_scale_2 = inverse_scale * inverse_scale
        scale_ratio = scale_increment * inverse_scale
        scaled_residual = residual * inverse_scale
        result[:, 0] = (
            -2.0
            * residual_mass
            * scale_increment
            * inverse_scale
            * inverse_scale_2
            * scale_direction
        )
        result[:, 1] = (
            2.0
            * residual_mass
            * scale_increment
            * inverse_scale
            * inverse_scale_2
            * (-location_direction + residual * (1.0 - 3.0 * scale_ratio) * scale_direction)
        )
        result[:, 2] = (
            -2.0
            * residual_mass
            * residual
            * inverse_scale_2
            * scale_ratio
            * (3.0 * scale_ratio - 1.0)
            * location_direction
            + (
                normalizer_mass * scale_ratio * (1.0 - scale_ratio) * (1.0 - 2.0 * scale_ratio)
                + residual_mass
                * scaled_residual**2
                * scale_ratio
                * (-12.0 * scale_ratio**2 + 9.0 * scale_ratio - 1.0)
            )
            * scale_direction
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Gaussian predictor-curvature derivative is not representable")
    return result


def gaussian_predictor_curvature_directional(
    response: NDArray,
    eta: NDArray,
    eta_direction: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    *,
    scale_floor: float,
) -> NDArray[np.float64]:
    response_values, weight_values = _aligned_vectors(
        (response, "response"),
        (weights, "weights"),
    )
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(weight_values, weight_semantics)
    floor = _scale_floor(scale_floor)
    try:
        predictors = np.asarray(eta, dtype=np.float64)
        direction = np.asarray(eta_direction, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Gaussian predictor state and direction must be finite n-by-2 arrays"
        ) from exc
    expected = (len(response_values), 2)
    if (
        predictors.shape != expected
        or direction.shape != expected
        or not np.all(np.isfinite(predictors))
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("Gaussian predictor state and direction must be finite n-by-2 arrays")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale_increment = np.exp(predictors[:, 1])
        scale = floor + scale_increment
    if not np.all(np.isfinite(scale)) or np.any(scale <= floor):
        raise ValueError("scale parameter must be finite and strictly above scale_floor")
    result = _gaussian_predictor_curvature_channels(
        response_values,
        predictors,
        direction,
        scale,
        scale_increment,
        weight_values,
        weight_semantics,
    )
    return readonly(result)
