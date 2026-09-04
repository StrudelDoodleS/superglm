"""Primitive normalized Gamma mean-CV numerical kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm.distributional.kernels._common import (
    WeightSemantics,
    positive_weights,
    readonly,
    readonly_bool,
    validated_derivative_order,
    validated_semantics,
)

_FLOAT = np.float64
_EPS = np.finfo(_FLOAT).eps
_MIN_FLOAT = np.nextafter(_FLOAT(0.0), _FLOAT(1.0))
_MAX_FLOAT = np.finfo(_FLOAT).max
_LOG_MIN_FLOAT = math.log(_MIN_FLOAT)
_LOG_MAX_FLOAT = math.log(_MAX_FLOAT)
_LOG_TWO = math.log(2.0)
_EULER = float(np.euler_gamma)
_INITIAL_RHO_RTOL = math.sqrt(_EPS)
_TAIL_INVERSE_RTOL = 64.0 * math.sqrt(_EPS)


class GammaInitializationError(ValueError):
    """Raised when Gamma initialization cannot produce an executable state."""


def _as_positive_vector(values: object, *, name: str) -> NDArray[np.float64]:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype(np.float64)
        or values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError(
            f"{name} must be a non-empty finite strictly positive one-dimensional "
            "float64 NumPy array"
        )
    return values


@dataclass(frozen=True)
class GammaKernelEvaluation:
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


def _small_a_residual(a: float) -> float:
    """Evaluate a(psi(a)-log(a)) from its convergent origin expansion."""

    total = -1.0 - a * (_EULER + math.log(a))
    power = a * a
    sign = 1.0
    for n in range(1, 80):
        term = sign * float(special.zeta(n + 1, 1.0)) * power
        total += term
        power *= a
        sign = -sign
        # zeta decreases to one, so this bounds the remaining geometric tail.
        if abs(power) / (1.0 - a) <= _EPS * max(1.0, abs(total)) / 8.0:
            return total
    raise ValueError("small-shape digamma series did not reach float64 accuracy")


def _small_j_residual(a: float) -> float:
    """Evaluate a(a trigamma(a)-1) from its convergent origin expansion."""

    total = 1.0 - a
    power = a * a
    sign = 1.0
    for n in range(0, 80):
        term = sign * (n + 1.0) * float(special.zeta(n + 2, 1.0)) * power
        total += term
        power *= a
        sign = -sign
        tail = (n + 2.0) * abs(power) / (1.0 - a) ** 2
        if tail <= _EPS * max(1.0, abs(total)) / 8.0:
            return total
    raise ValueError("small-shape trigamma series did not reach float64 accuracy")


def _small_j_log_derivative(a: float) -> float:
    """Evaluate a d/da[a(a trigamma(a)-1)] near the origin."""

    total = -a
    power = a * a
    sign = 1.0
    for n in range(0, 80):
        total += sign * (n + 1.0) * (n + 2.0) * float(special.zeta(n + 2, 1.0)) * power
        power *= a
        sign = -sign
        tail = (n + 2.0) * (n + 3.0) * abs(power) / (1.0 - a) ** 3
        if tail <= _EPS * max(1.0, abs(total)) / 8.0:
            return total
    raise ValueError("small-shape tetragamma combination did not reach float64 accuracy")


def _small_log_normalizer(a: float) -> float:
    """Evaluate a log(a)-a-log Gamma(a) from the log-Gamma origin series."""

    total = (a + 1.0) * math.log(a) + (_EULER - 1.0) * a
    power = a * a
    sign = -1.0
    for n in range(2, 80):
        term = sign * float(special.zeta(n, 1.0)) * power / n
        total += term
        power *= a
        sign = -sign
        tail = abs(power) / ((n + 1.0) * (1.0 - a))
        if tail <= _EPS * max(1.0, abs(total)) / 8.0:
            return total
    raise ValueError("small-shape log-normalizer series did not reach float64 accuracy")


def _large_a_residual(a: NDArray) -> NDArray[np.float64]:
    inverse = 1.0 / a
    inverse2 = inverse * inverse
    return -0.5 + inverse * (
        -1.0 / 12.0
        + inverse2
        * (
            1.0 / 120.0
            + inverse2 * (-1.0 / 252.0 + inverse2 * (1.0 / 240.0 - inverse2 * 1.0 / 132.0))
        )
    )


def _large_j_residual(a: NDArray) -> NDArray[np.float64]:
    inverse = 1.0 / a
    inverse2 = inverse * inverse
    return 0.5 + inverse * (
        1.0 / 6.0
        + inverse2
        * (
            -1.0 / 30.0
            + inverse2
            * (
                1.0 / 42.0
                + inverse2 * (-1.0 / 30.0 + inverse2 * (5.0 / 66.0 - inverse2 * 691.0 / 2730.0))
            )
        )
    )


def _large_j_log_derivative(a: NDArray) -> NDArray[np.float64]:
    inverse = 1.0 / a
    inverse2 = inverse * inverse
    return inverse * (
        -1.0 / 6.0
        + inverse2
        * (
            1.0 / 10.0
            + inverse2
            * (
                -5.0 / 42.0
                + inverse2 * (7.0 / 30.0 + inverse2 * (-15.0 / 22.0 + inverse2 * 7601.0 / 2730.0))
            )
        )
    )


def _large_log_normalizer(a: NDArray) -> NDArray[np.float64]:
    inverse = 1.0 / a
    inverse2 = inverse * inverse
    correction = inverse * (
        -1.0 / 12.0
        + inverse2 * (1.0 / 360.0 + inverse2 * (-1.0 / 1260.0 + inverse2 * (1.0 / 1680.0)))
    )
    return 0.5 * np.log(a / (2.0 * math.pi)) + correction


def _scaled_digamma_residual(shape: NDArray) -> NDArray[np.float64]:
    """Return ``a * (digamma(a) - log(a))`` without endpoint cancellation."""

    values = _as_positive_vector(shape, name="shape")
    result = np.empty_like(values)
    small = values < 0.25
    for index in np.flatnonzero(small):
        result[index] = _small_a_residual(float(values[index]))
    large = values >= 16.0
    inverse = np.zeros_like(values)
    inverse[large] = 1.0 / values[large]
    asymptotic = large & ((691.0 / 32760.0) * inverse**11 <= _EPS / 8.0)
    result[asymptotic] = _large_a_residual(values[asymptotic])
    direct = ~(small | asymptotic)
    direct_values = values[direct]
    result[direct] = direct_values * (special.digamma(direct_values) - np.log(direct_values))
    return result


def _scaled_trigamma_residual(shape: NDArray) -> NDArray[np.float64]:
    """Return ``a * (a * trigamma(a) - 1)`` as a bounded combination."""

    values = _as_positive_vector(shape, name="shape")
    result = np.empty_like(values)
    small = values < 0.25
    for index in np.flatnonzero(small):
        result[index] = _small_j_residual(float(values[index]))
    large = values >= 16.0
    inverse = np.zeros_like(values)
    inverse[large] = 1.0 / values[large]
    asymptotic = large & ((7.0 / 6.0) * inverse**13 <= _EPS / 8.0)
    result[asymptotic] = _large_j_residual(values[asymptotic])
    direct = ~(small | asymptotic)
    direct_values = values[direct]
    result[direct] = direct_values * (direct_values * special.polygamma(1, direct_values) - 1.0)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("Gamma shape information is not representable")
    return result


def _scaled_trigamma_log_derivative(shape: NDArray) -> NDArray[np.float64]:
    """Return the log-shape derivative of the bounded trigamma residual."""

    values = _as_positive_vector(shape, name="shape")
    result = np.empty_like(values)
    small = values < 0.25
    for index in np.flatnonzero(small):
        result[index] = _small_j_log_derivative(float(values[index]))
    large = values >= 32.0
    result[large] = _large_j_log_derivative(values[large])
    direct = ~(small | large)
    direct_values = values[direct]
    j_residual = direct_values * (direct_values * special.polygamma(1, direct_values) - 1.0)
    result[direct] = (
        2.0 * j_residual + direct_values + direct_values**3 * special.polygamma(2, direct_values)
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Gamma shape-curvature derivative is not representable")
    return result


def _gamma_log_normalizer(shape: NDArray) -> NDArray[np.float64]:
    """Return ``a log(a) - a - log Gamma(a)`` stably."""

    values = _as_positive_vector(shape, name="shape")
    result = np.empty_like(values)
    small = values < 0.25
    for index in np.flatnonzero(small):
        result[index] = _small_log_normalizer(float(values[index]))
    large = values >= 16.0
    inverse = np.zeros_like(values)
    inverse[large] = 1.0 / values[large]
    asymptotic = large & (
        (1.0 / 1188.0) * inverse**9 <= _EPS * np.maximum(1.0, np.log(values)) / 8.0
    )
    result[asymptotic] = _large_log_normalizer(values[asymptotic])
    direct = ~(small | asymptotic)
    direct_values = values[direct]
    result[direct] = (
        direct_values * np.log(direct_values) - direct_values - special.gammaln(direct_values)
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Gamma log normalizer is not representable")
    return result


def _binary_product_divide(
    numerators: tuple[float, ...], denominators: tuple[float, ...] = ()
) -> float:
    """Compose float64 products/divisions with one final binary rounding."""

    if any(value == 0.0 for value in numerators):
        return math.copysign(0.0, math.prod(math.copysign(1.0, value) for value in numerators))
    mantissa = 1.0
    exponent = 0
    sign = 1.0
    for value in numerators:
        if not math.isfinite(value):
            raise ValueError("non-finite product primitive")
        if value < 0.0:
            sign = -sign
        part, shift = math.frexp(abs(value))
        mantissa *= part
        exponent += shift
        mantissa, shift = math.frexp(mantissa)
        exponent += shift
    for value in denominators:
        if not math.isfinite(value) or value == 0.0:
            raise ValueError("invalid division primitive")
        if value < 0.0:
            sign = -sign
        part, shift = math.frexp(abs(value))
        mantissa /= part
        exponent -= shift
        mantissa, shift = math.frexp(mantissa)
        exponent += shift
    try:
        return math.copysign(math.ldexp(mantissa, exponent), sign)
    except OverflowError as exc:
        raise ValueError("final natural channel is not representable") from exc


def _log_ratio(y: float, mean: float) -> float:
    y_mantissa, y_exponent = math.frexp(y)
    mean_mantissa, mean_exponent = math.frexp(mean)
    relative_mantissa = (y_mantissa - mean_mantissa) / mean_mantissa
    mantissa_log = (
        math.log1p(relative_mantissa)
        if abs(relative_mantissa) <= 0.125
        else math.log(y_mantissa / mean_mantissa)
    )
    return mantissa_log + (y_exponent - mean_exponent) * _LOG_TWO


def _deviance_from_t(t: float) -> float:
    """Return t-log1p(t) with a certified convergent-series tail."""

    power = t * t
    total = 0.0
    for n in range(2, 160):
        term = power / n if n % 2 == 0 else -power / n
        total += term
        power *= t
        if t >= 0.0:
            tail = abs(power) / (n + 1.0)
        else:
            tail = abs(power) / ((n + 1.0) * (1.0 - abs(t)))
        if tail <= _EPS * max(abs(total), _MIN_FLOAT) / 8.0:
            return total
    raise ValueError("near-equality Gamma ratio series did not converge")


def _stable_row_sum(*terms: NDArray) -> NDArray[np.float64]:
    arrays = tuple(np.asarray(term, dtype=_FLOAT) for term in terms)
    total = np.array(arrays[0], copy=True)
    magnitude = np.abs(arrays[0])
    for term in arrays[1:]:
        total += term
        magnitude += np.abs(term)
    ambiguous = np.abs(total) <= 8.0 * _EPS * magnitude
    for index in np.flatnonzero(ambiguous):
        total[index] = math.fsum(float(term[index]) for term in arrays)
    return total


def _vector_deviance_from_t(t: NDArray) -> NDArray[np.float64]:
    total = np.zeros_like(t, dtype=_FLOAT)
    power = np.square(t, dtype=_FLOAT)
    for n in range(2, 26):
        total += power / n if n % 2 == 0 else -power / n
        power *= t
    return total


def gamma_shape_plus_one_log_increment(
    shape: NDArray,
    threshold: NDArray,
) -> NDArray[np.float64]:
    """Return ``log(x**a exp(-x) / Gamma(a+1))`` without forming ``a+1``."""
    shape_values = _as_positive_vector(shape, name="shape")
    threshold_values = np.asarray(threshold, dtype=_FLOAT)
    if (
        threshold_values.ndim != 1
        or threshold_values.shape != shape_values.shape
        or np.any(~np.isfinite(threshold_values))
        or np.any(threshold_values < 0.0)
    ):
        raise ValueError("Gamma recurrence thresholds must be aligned finite non-negative values")

    result = np.full(shape_values.shape, -np.inf, dtype=_FLOAT)
    positive_rows = np.flatnonzero(threshold_values > 0.0)
    if positive_rows.size:
        positive_shape = shape_values[positive_rows]
        positive_threshold = threshold_values[positive_rows]
        relative_offset = (positive_threshold - positive_shape) / positive_shape
        deviance = np.empty_like(relative_offset)
        close = np.abs(relative_offset) <= 0.125
        for local_index in np.flatnonzero(close):
            deviance[local_index] = _deviance_from_t(float(relative_offset[local_index]))
        far = ~close
        if np.any(far):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                ratio = positive_threshold[far] / positive_shape[far]
                log_ratio = np.log(positive_threshold[far]) - np.log(positive_shape[far])
                deviance[far] = ratio - 1.0 - log_ratio
        with np.errstate(over="ignore", invalid="ignore"):
            result[positive_rows] = (
                _gamma_log_normalizer(positive_shape)
                - np.log(positive_shape)
                - positive_shape * deviance
            )
    return readonly(result)


def gamma_expected_shortfall(
    p: NDArray,
    mean: NDArray,
    shape: NDArray,
) -> NDArray[np.float64]:
    """Upper conditional mean for a Gamma law, with inverse-tail certification.

    The shifted-shape tail is divided by the survival probability before the
    result is scaled by ``mean``.  This order preserves subnormal means.  The
    regularized-gamma recurrence is used for every row, so float64 rounding of
    ``shape + 1`` can neither erase the increment nor turn it into two.
    """
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise ValueError("expected-shortfall probabilities must lie strictly inside (0, 1)")
    mean_values = _as_positive_vector(mean, name="mean")
    shape_values = _as_positive_vector(shape, name="shape")
    if probabilities.shape != mean_values.shape or probabilities.shape != shape_values.shape:
        raise ValueError("Gamma expected-shortfall row arrays must have the same shape")

    survival = 1.0 - probabilities
    threshold = special.gammainccinv(shape_values, survival)
    achieved_survival = special.gammaincc(shape_values, threshold)
    inverse_resolved = ((threshold == 0.0) & (survival == 1.0)) | (
        np.isfinite(threshold)
        & (threshold > 0.0)
        & np.isfinite(achieved_survival)
        & (np.abs(achieved_survival - survival) <= _TAIL_INVERSE_RTOL * survival)
    )
    if not np.all(inverse_resolved):
        rows = np.flatnonzero(~inverse_resolved).tolist()
        raise ValueError(f"Gamma expected shortfall cannot be certified in float64 for rows {rows}")

    log_increment = gamma_shape_plus_one_log_increment(shape_values, threshold)
    ratio = 1.0 + np.exp(log_increment - np.log(survival))

    with np.errstate(over="ignore", invalid="ignore"):
        result = mean_values * ratio
        quantile = mean_values * (threshold / shape_values)
    certified = (
        np.isfinite(ratio)
        & (ratio > 0.0)
        & (np.isposinf(result) | (np.isfinite(result) & (result >= quantile)))
    )
    if not np.all(certified):
        rows = np.flatnonzero(~certified).tolist()
        raise ValueError(f"Gamma expected shortfall cannot be certified in float64 for rows {rows}")
    return readonly(result)


def _validated_scaled_ratio_inputs(
    y: NDArray,
    mean: NDArray,
    shape: NDArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    response = _as_positive_vector(y, name="y")
    location = _as_positive_vector(mean, name="mean")
    a_values = _as_positive_vector(shape, name="shape")
    if response.shape != location.shape or response.shape != a_values.shape:
        raise ValueError("scaled ratio inputs must share one shape")
    return response, location, a_values


def _vector_scaled_ratio_candidates(
    response: NDArray[np.float64],
    location: NDArray[np.float64],
    a_values: NDArray[np.float64],
    *,
    derivative_order: int,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    az = np.empty_like(response) if derivative_order == 2 else None
    at = np.empty_like(response) if derivative_order >= 1 else None
    ad = np.empty_like(response)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio = response / location
        log_ratio = np.log(ratio)
        deviance = _stable_row_sum(ratio, -np.ones_like(ratio), -log_ratio)
        candidate_az = a_values * ratio if az is not None else None
        candidate_at = a_values * (ratio - 1.0) if at is not None else None
        candidate_ad = a_values * deviance
    _, response_exponent = np.frexp(response)
    _, location_exponent = np.frexp(location)
    close = np.abs(response_exponent - location_exponent) <= 1
    t = np.full_like(response, np.inf)
    t[close] = (response[close] - location[close]) / location[close]
    near = close & (np.abs(t) <= 0.125)
    candidate_ad[near] = a_values[near] * _vector_deviance_from_t(t[near])
    safe = np.isfinite(ratio) & (ratio > 0.0) & np.isfinite(candidate_ad)
    if az is not None:
        assert candidate_az is not None
        safe &= np.isfinite(candidate_az)
        az[safe] = candidate_az[safe]
    if at is not None:
        assert candidate_at is not None
        candidate_at[near] = a_values[near] * t[near]
        safe &= np.isfinite(candidate_at)
        at[safe] = candidate_at[safe]
    ad[safe] = candidate_ad[safe]
    return az, at, ad, safe


def _fill_scaled_ratio_fallback_rows(
    response: NDArray[np.float64],
    location: NDArray[np.float64],
    a_values: NDArray[np.float64],
    az: NDArray[np.float64] | None,
    at: NDArray[np.float64] | None,
    ad: NDArray[np.float64],
    safe: NDArray[np.bool_],
) -> None:
    for index in np.flatnonzero(~safe):
        y_scalar = float(response[index])
        mean_scalar = float(location[index])
        a_scalar = float(a_values[index])
        y_exponent = math.frexp(y_scalar)[1]
        mean_exponent = math.frexp(mean_scalar)[1]
        near = abs(y_exponent - mean_exponent) <= 1
        t = (y_scalar - mean_scalar) / mean_scalar if near else math.inf
        near = near and abs(t) <= 0.125
        if near:
            if az is not None:
                az[index] = _binary_product_divide((a_scalar, y_scalar), (mean_scalar,))
            if at is not None:
                at[index] = _binary_product_divide((a_scalar, t)) if t != 0.0 else 0.0
            d = _deviance_from_t(t)
            ad[index] = _binary_product_divide((a_scalar, d)) if d != 0.0 else 0.0
            continue
        log_z = _log_ratio(y_scalar, mean_scalar)
        az_value = _binary_product_divide((a_scalar, y_scalar), (mean_scalar,))
        a_log_z = _binary_product_divide((a_scalar, log_z)) if log_z != 0.0 else 0.0
        if az is not None:
            az[index] = az_value
        if at is not None:
            at[index] = math.fsum((az_value, -a_scalar))
        ad[index] = math.fsum((az_value, -a_scalar, -a_log_z))


def _scaled_ratio_terms(
    y: NDArray,
    mean: NDArray,
    shape: NDArray,
    *,
    derivative_order: int = 2,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64]]:
    if (
        isinstance(derivative_order, bool | np.bool_)
        or not isinstance(derivative_order, int | np.integer)
        or int(derivative_order) not in (0, 1, 2)
    ):
        raise ValueError("derivative_order must be an integer from zero through two")
    derivative_order = int(derivative_order)
    response, location, a_values = _validated_scaled_ratio_inputs(y, mean, shape)
    az, at, ad, safe = _vector_scaled_ratio_candidates(
        response,
        location,
        a_values,
        derivative_order=derivative_order,
    )
    _fill_scaled_ratio_fallback_rows(response, location, a_values, az, at, ad, safe)
    if any(values is not None and not np.all(np.isfinite(values)) for values in (az, at, ad)):
        raise ValueError("scaled Gamma ratio terms are not representable")
    return az, at, ad


def _shape_from_scale(
    scale: NDArray[np.float64],
    weights: NDArray[np.float64],
    semantics: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    multiplier = np.ones_like(scale) if semantics == "prior" else weights
    numerator = weights if semantics == "prior" else np.ones_like(scale)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        shape, exponent = np.frexp(numerator)
        scale_mantissa, scale_exponent = np.frexp(scale)
        for _ in range(2):
            shape /= scale_mantissa
            exponent -= scale_exponent
            shape, shift = np.frexp(shape)
            exponent += shift
        shape = np.ldexp(shape, exponent)
    fallback = ~np.isfinite(shape) | (shape <= 0.0)
    for index in np.flatnonzero(fallback):
        value = _binary_product_divide(
            (float(numerator[index]),),
            (float(scale[index]), float(scale[index])),
        )
        if value == 0.0 or not math.isfinite(value):
            raise ValueError("derived Gamma shape is not representable")
        shape[index] = value
    return shape, multiplier


def _channel(
    name: str,
    numerators: tuple[float, ...],
    denominators: tuple[float, ...] = (),
    *,
    positive: bool = False,
) -> float:
    mathematical_zero = any(value == 0.0 for value in numerators)
    value = _binary_product_divide(numerators, denominators)
    if not math.isfinite(value):
        raise ValueError(f"Gamma {name} is not representable")
    if value == 0.0 and not mathematical_zero:
        raise ValueError(f"Gamma {name} is not representable after natural-scale underflow")
    if positive and value <= 0.0:
        raise ValueError(f"Gamma {name} must remain strictly positive and representable")
    return value


def _vector_channel(
    name: str,
    numerators: tuple[NDArray | float, ...],
    denominators: tuple[NDArray | float, ...] = (),
    *,
    positive: bool = False,
) -> NDArray[np.float64]:
    shape = next(
        np.asarray(value).shape for value in (*numerators, *denominators) if np.ndim(value)
    )
    numerator_arrays = tuple(np.broadcast_to(value, shape) for value in numerators)
    denominator_arrays = tuple(np.broadcast_to(value, shape) for value in denominators)
    result = np.ones(shape, dtype=_FLOAT)
    mathematical_zero = np.zeros(shape, dtype=bool)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        for value in numerator_arrays:
            mathematical_zero |= value == 0.0
            result *= value
        for value in denominator_arrays:
            result /= value
    fallback = ~np.isfinite(result) | ((result == 0.0) & ~mathematical_zero)
    if positive:
        fallback |= result <= 0.0
    for index in np.flatnonzero(fallback):
        result[index] = _channel(
            name,
            tuple(float(value[index]) for value in numerator_arrays),
            tuple(float(value[index]) for value in denominator_arrays),
            positive=positive,
        )
    return result


def _gamma_score_channels(
    response: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    shape: NDArray[np.float64],
    multiplier: NDArray[np.float64],
    at: NDArray[np.float64],
    ad: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    a_residual = _scaled_digamma_residual(shape)
    b = _stable_row_sum(a_residual, ad)
    score = np.empty((len(response), 2), dtype=_FLOAT)
    _, response_exponent = np.frexp(response)
    _, mean_exponent = np.frexp(mean)
    close = np.abs(response_exponent - mean_exponent) <= 1
    far = ~close
    t = np.empty_like(response)
    t[close] = (response[close] - mean[close]) / mean[close]
    score[close, 0] = _vector_channel(
        "mean score",
        (multiplier[close], shape[close], t[close]),
        (mean[close],),
    )
    score[far, 0] = _vector_channel(
        "mean score",
        (multiplier[far], at[far]),
        (mean[far],),
    )
    score[:, 1] = _vector_channel("scale score", (2.0, multiplier, b), (scale,))
    return score, b, close, far, t


def _gamma_hessian_channels(
    response: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    shape: NDArray[np.float64],
    multiplier: NDArray[np.float64],
    az: NDArray[np.float64],
    at: NDArray[np.float64],
    b: NDArray[np.float64],
    close: NDArray[np.bool_],
    far: NDArray[np.bool_],
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    j_residual = _scaled_trigamma_residual(shape)
    hessian = np.empty((len(response), 3), dtype=_FLOAT)
    hessian[close, 0] = _vector_channel(
        "mean Hessian",
        (-1.0, multiplier[close], shape[close], 1.0 + 2.0 * t[close]),
        (mean[close], mean[close]),
    )
    hessian[far, 0] = _vector_channel(
        "mean Hessian",
        (multiplier[far], _stable_row_sum(shape[far], -2.0 * az[far])),
        (mean[far], mean[far]),
    )
    hessian[close, 1] = _vector_channel(
        "mean-scale Hessian",
        (-2.0, multiplier[close], shape[close], t[close]),
        (mean[close], scale[close]),
    )
    hessian[far, 1] = _vector_channel(
        "mean-scale Hessian",
        (-2.0, multiplier[far], at[far]),
        (mean[far], scale[far]),
    )
    scale_numerator = _stable_row_sum(2.0 * j_residual, 3.0 * b)
    hessian[:, 2] = _vector_channel(
        "scale Hessian",
        (-2.0, multiplier, scale_numerator),
        (scale, scale),
    )
    return hessian


def evaluate_gamma_rows(
    response: NDArray,
    mean: NDArray,
    scale: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
    *,
    derivative_order: int,
) -> GammaKernelEvaluation:
    response_values = _as_positive_vector(response, name="response")
    mean_values = _as_positive_vector(mean, name="mean")
    scale_values = _as_positive_vector(scale, name="scale")
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(_as_positive_vector(weights, name="weights"), weight_semantics)
    if any(
        values.shape != response_values.shape
        for values in (mean_values, scale_values, weight_values)
    ):
        raise ValueError("Gamma row arrays must have the same shape")
    order = validated_derivative_order(derivative_order)
    shape, multiplier = _shape_from_scale(scale_values, weight_values, weight_semantics)
    normalizer = _gamma_log_normalizer(shape)
    az, at, ad = _scaled_ratio_terms(
        response_values,
        mean_values,
        shape,
        derivative_order=order,
    )
    value_primitive = _stable_row_sum(normalizer, -ad)
    optimizing = _vector_channel("optimizing likelihood", (multiplier, value_primitive))
    score = None
    hessian = None
    if order >= 1:
        assert at is not None
        score, b, close, far, t = _gamma_score_channels(
            response_values,
            mean_values,
            scale_values,
            shape,
            multiplier,
            at,
            ad,
        )
        if order == 2:
            assert az is not None
            hessian = _gamma_hessian_channels(
                response_values,
                mean_values,
                scale_values,
                shape,
                multiplier,
                az,
                at,
                b,
                close,
                far,
                t,
            )
    return GammaKernelEvaluation(optimizing, score, hessian, np.ones(len(response_values), bool))


def gamma_expected_information(
    mean: NDArray,
    scale: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    mean_values = _as_positive_vector(mean, name="mean")
    scale_values = _as_positive_vector(scale, name="scale")
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(_as_positive_vector(weights, name="weights"), weight_semantics)
    if scale_values.shape != mean_values.shape or weight_values.shape != mean_values.shape:
        raise ValueError("Gamma row arrays must have the same shape")
    shape, multiplier = _shape_from_scale(scale_values, weight_values, weight_semantics)
    j_residual = _scaled_trigamma_residual(shape)
    information = np.zeros((len(mean_values), 3), dtype=_FLOAT)
    information[:, 0] = _vector_channel(
        "mean information",
        (multiplier, shape),
        (mean_values, mean_values),
        positive=True,
    )
    information[:, 2] = _vector_channel(
        "scale information",
        (4.0, multiplier, j_residual),
        (scale_values, scale_values),
        positive=True,
    )
    return readonly(information)


def _gamma_initial_rho_bounds(
    weights: NDArray[np.float64],
    semantics: str,
) -> tuple[float, float]:
    log_weights = np.log(weights)
    lower = max(
        -2.0 * _LOG_MAX_FLOAT,
        float(np.max(_LOG_MIN_FLOAT - log_weights)) if semantics == "prior" else _LOG_MIN_FLOAT,
    )
    upper = min(
        -2.0 * _LOG_MIN_FLOAT,
        float(np.min(_LOG_MAX_FLOAT - log_weights)) if semantics == "prior" else _LOG_MAX_FLOAT,
    )
    margin = 8.0 * _EPS * max(1.0, abs(lower), abs(upper))
    return lower + margin, upper - margin


def _gamma_initial_candidate_rhos(
    lower: float,
    upper: float,
    *preferred: float | None,
) -> tuple[float, ...]:
    """Return one bounded bidirectional executable-state search sequence."""

    raw_candidates = [*preferred, lower, upper]
    center = min(max(-math.log(_EPS), lower), upper)
    steps = math.ceil(max(center - lower, upper - center) / 32.0)
    raw_candidates.append(center)
    for multiplier in range(1, steps + 1):
        offset = 32.0 * multiplier
        raw_candidates.extend((min(center + offset, upper), max(center - offset, lower)))
    candidates: list[float] = []
    for rho in raw_candidates:
        if rho is None or not math.isfinite(rho):
            continue
        candidate = min(max(rho, lower), upper)
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _gamma_initial_target(
    response: NDArray[np.float64],
    mean: float,
    weights: NDArray[np.float64],
    semantics: str,
    rho: float,
) -> float | None:
    try:
        k = math.exp(rho)
        with np.errstate(over="raise", invalid="raise"):
            shape = weights * k if semantics == "prior" else np.full(len(response), k)
        if not np.all(np.isfinite(shape)) or np.any(shape <= 0.0):
            return None
        a_residual = _scaled_digamma_residual(shape)
        location = np.full(len(response), mean)
        _, _, a_deviance = _scaled_ratio_terms(response, location, shape, derivative_order=0)
        multiplier = np.ones(len(response), dtype=_FLOAT) if semantics == "prior" else weights
        with np.errstate(over="raise", invalid="raise"):
            terms = multiplier * (a_residual + a_deviance)
        target = math.fsum(float(term) for term in terms)
    except (FloatingPointError, OverflowError, ValueError):
        return None
    return target if math.isfinite(target) else None


def _refine_gamma_initial_rho(
    response: NDArray[np.float64],
    mean: float,
    weights: NDArray[np.float64],
    semantics: str,
    seed_rho: float,
) -> float | None:
    """Best-effort initializer refinement; final-fit certification lives elsewhere."""

    lower, upper = _gamma_initial_rho_bounds(weights, semantics)
    center = min(max(seed_rho, lower), upper)
    left = right = center
    left_target = right_target = _gamma_initial_target(response, mean, weights, semantics, center)
    step = 1.0
    while left > lower and (left_target is None or left_target >= 0.0):
        left = max(lower, center - step)
        left_target = _gamma_initial_target(response, mean, weights, semantics, left)
        step *= 2.0
    step = 1.0
    while right < upper and (right_target is None or right_target <= 0.0):
        right = min(upper, center + step)
        right_target = _gamma_initial_target(response, mean, weights, semantics, right)
        step *= 2.0
    if left_target is None or right_target is None or left_target >= 0.0 or right_target <= 0.0:
        return None

    for _ in range(256):
        tolerance = _INITIAL_RHO_RTOL * max(1.0, abs(left), abs(right))
        if right - left <= tolerance:
            return left + 0.5 * (right - left)
        rho = left + 0.5 * (right - left)
        if rho in (left, right):
            return rho
        target = _gamma_initial_target(response, mean, weights, semantics, rho)
        if target is None:
            return None
        if target < 0.0:
            left = rho
        elif target > 0.0:
            right = rho
        else:
            return rho
    return None


def _executable_gamma_initial_state(
    response: NDArray[np.float64],
    weights: NDArray[np.float64],
    semantics: WeightSemantics,
    mean: float,
    rho: float,
) -> NDArray[np.float64]:
    scale = math.exp(-0.5 * rho)
    if not math.isfinite(scale) or scale <= 0.0:
        raise GammaInitializationError("Gamma initialization scale is not representable")
    theta = np.column_stack(
        (
            np.full(len(response), mean),
            np.full(len(response), scale),
        )
    )
    evaluate_gamma_rows(
        response,
        theta[:, 0],
        theta[:, 1],
        weights,
        semantics,
        derivative_order=2,
    )
    gamma_expected_information(theta[:, 0], theta[:, 1], weights, semantics)
    return readonly(theta)


def initialize_gamma(
    response: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    response_values = _as_positive_vector(response, name="response")
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(_as_positive_vector(weights, name="weights"), weight_semantics)
    if response_values.shape != weight_values.shape:
        raise ValueError("Gamma row arrays must have the same shape")
    lower, upper = _gamma_initial_rho_bounds(weight_values, weight_semantics)
    if np.all(response_values == response_values[0]):
        mean = float(response_values[0])
        last_error: ValueError | None = None
        for rho in _gamma_initial_candidate_rhos(lower, upper, -math.log(_EPS)):
            try:
                return _executable_gamma_initial_state(
                    response_values, weight_values, weight_semantics, mean, rho
                )
            except ValueError as exc:
                last_error = exc
        raise GammaInitializationError(
            "constant-response Gamma initialization exhausted its bounded "
            "representable-state search"
        ) from last_error

    normalized_weights = weight_values / float(np.sum(weight_values, dtype=_FLOAT))
    mean = float(np.dot(normalized_weights, response_values))
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("Gamma initialization mean is not representable")
    residual = response_values - mean
    residual_scale = float(np.max(np.abs(residual)))
    denominator = (
        len(response_values)
        if weight_semantics == "prior"
        else float(np.sum(weight_values, dtype=_FLOAT))
    )
    scaled_ss = float(np.dot(weight_values, (residual / residual_scale) ** 2))
    log_seed = (
        math.log(residual_scale)
        + 0.5 * (math.log(scaled_ss) - math.log(denominator))
        - math.log(mean)
    )
    seed_rho = min(max(-2.0 * log_seed, lower), upper)
    refined_rho = _refine_gamma_initial_rho(
        response_values, mean, weight_values, weight_semantics, seed_rho
    )
    last_error = None
    for rho in _gamma_initial_candidate_rhos(lower, upper, refined_rho, seed_rho):
        try:
            return _executable_gamma_initial_state(
                response_values, weight_values, weight_semantics, mean, rho
            )
        except ValueError as exc:
            last_error = exc
    raise GammaInitializationError(
        "Gamma initialization exhausted its bounded executable-state search"
    ) from last_error


def _gamma_predictor_curvature_channels(
    response: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
    direction: NDArray[np.float64],
    weights: NDArray[np.float64],
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    shape, multiplier = _shape_from_scale(scale, weights, semantics)
    az, at, ad = _scaled_ratio_terms(response, mean, shape, derivative_order=2)
    assert az is not None and at is not None
    a_residual = _scaled_digamma_residual(shape)
    j_residual = _scaled_trigamma_residual(shape)
    j_log_derivative = _scaled_trigamma_log_derivative(shape)
    b = _stable_row_sum(a_residual, ad)
    mean_direction = direction[:, 0]
    scale_direction = direction[:, 1]
    result = np.empty((len(response), 3), dtype=_FLOAT)
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, 0] = -multiplier * az * (mean_direction + 2.0 * scale_direction)
        result[:, 1] = 2.0 * multiplier * (-az * mean_direction - 2.0 * at * scale_direction)
        result[:, 2] = (
            4.0
            * multiplier
            * (-at * mean_direction - 2.0 * (b + j_residual + j_log_derivative) * scale_direction)
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Gamma predictor-curvature derivative is not representable")
    return result


def gamma_predictor_curvature_directional(
    response: NDArray,
    eta: NDArray,
    eta_direction: NDArray,
    weights: NDArray,
    semantics: WeightSemantics,
) -> NDArray[np.float64]:
    response_values = _as_positive_vector(response, name="response")
    weight_semantics = validated_semantics(semantics)
    weight_values = positive_weights(_as_positive_vector(weights, name="weights"), weight_semantics)
    if response_values.shape != weight_values.shape:
        raise ValueError("Gamma row arrays must have the same shape")
    try:
        predictors = np.asarray(eta, dtype=_FLOAT)
        direction = np.asarray(eta_direction, dtype=_FLOAT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Gamma predictor state and direction must be finite n-by-2 arrays"
        ) from exc
    expected = (len(response_values), 2)
    if (
        predictors.shape != expected
        or direction.shape != expected
        or not np.all(np.isfinite(predictors))
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("Gamma predictor state and direction must be finite n-by-2 arrays")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        mean = np.exp(predictors[:, 0])
        scale = np.exp(predictors[:, 1])
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("mean and scale parameters must be finite and strictly positive")
    if np.any(mean <= 0.0) or np.any(scale <= 0.0):
        raise ValueError("mean and scale parameters must be finite and strictly positive")
    result = _gamma_predictor_curvature_channels(
        response_values,
        mean,
        scale,
        direction,
        weight_values,
        weight_semantics,
    )
    return readonly(result)
