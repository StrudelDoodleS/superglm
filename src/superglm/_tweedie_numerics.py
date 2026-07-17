from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import SupportsFloat

import numpy as np
from numpy.typing import NDArray

PHI_LOWER_BOUND = 1e-12
_DEVIANCE_DTYPE = np.dtype(np.longdouble)
_LOWER_POWER_BOUNDARY = 1e-3
_SERIES_LOG_RATIO_BOUNDARY = 1.0


class TweedieNumericalError(RuntimeError):
    """Exact Tweedie arithmetic could not be represented or certified."""


@dataclass(frozen=True)
class CompoundPoissonGammaParameters:
    rate: NDArray[np.float64]
    shape: float
    scale: NDArray[np.float64]


def normalize_real_scalar(name: str, value: object) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one finite real scalar")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_tweedie_power(value: object) -> float:
    p = normalize_real_scalar("p", value)
    if not 1.0 < p < 2.0:
        raise ValueError(f"Tweedie p must be in (1, 2), got {p}")
    return p


def normalize_positive_scalar(name: str, value: object) -> float:
    result = normalize_real_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return result


def _as_real_float64_array(name: str, value: object) -> NDArray[np.float64]:
    """Convert a real numeric array without silently coercing non-real values."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real numeric array") from exc
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        result: NDArray[np.float64] = np.asarray(raw, dtype=np.float64)
    return result


def _scaled_positive_ratio(
    shape: tuple[int, ...],
    numerators: tuple[object, ...],
    denominators: tuple[object, ...],
    *,
    binary_exponent: object = 0,
) -> NDArray[np.float64]:
    """Multiply and divide positive factors without overflowing intermediates."""
    mantissa = np.ones(shape, dtype=np.float64)
    exponent = np.broadcast_to(np.asarray(binary_exponent, dtype=np.int64), shape).copy()
    for value in numerators:
        factor_mantissa, factor_exponent = np.frexp(np.asarray(value, dtype=np.float64))
        mantissa *= factor_mantissa
        exponent += factor_exponent
    for value in denominators:
        factor_mantissa, factor_exponent = np.frexp(np.asarray(value, dtype=np.float64))
        mantissa /= factor_mantissa
        exponent -= factor_exponent
    mantissa, adjustment = np.frexp(mantissa)
    exponent += adjustment
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result: NDArray[np.float64] = np.ldexp(mantissa, exponent)
    return result


def _scaled_power_deviance(
    base: object,
    power: SupportsFloat,
    numerators: tuple[object, ...],
    denominators: tuple[object, ...] = (),
) -> NDArray[np.float64]:
    """Compose a positive deviance power product in authoritative float64.

    Keeping the power of ``base`` as a compensated binary exponent avoids both
    an unrepresentable intermediate power and the precision loss of a final
    large logarithm/exponential round trip.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    float_power = float(power)

    base_mantissa, base_exponent = np.frexp(base_arr)
    normalized_mantissa = 2.0 * base_mantissa
    normalized_exponent = base_exponent.astype(np.int64) - 1

    exponent_factor = normalized_exponent.astype(np.float64)
    exponent_product = float_power * exponent_factor
    splitter = 134_217_729.0  # 2**27 + 1, for an error-free float64 product split.
    split_power = splitter * float_power
    power_high = split_power - (split_power - float_power)
    power_low = float_power - power_high
    exponent_error = (power_high * exponent_factor - exponent_product) + (
        power_low * exponent_factor
    )

    binary_exponent = np.floor(exponent_product).astype(np.int64)
    fractional_exponent = exponent_product - binary_exponent + exponent_error
    below_zero = fractional_exponent < 0.0
    if np.any(below_zero):
        fractional_exponent[below_zero] += 1.0
        binary_exponent[below_zero] -= 1
    at_least_one = fractional_exponent >= 1.0
    if np.any(at_least_one):
        fractional_exponent[at_least_one] -= 1.0
        binary_exponent[at_least_one] += 1

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        fractional_power = np.exp2(fractional_exponent + float_power * np.log2(normalized_mantissa))
    result = _scaled_positive_ratio(
        base_arr.shape,
        (*numerators, fractional_power),
        denominators,
        binary_exponent=binary_exponent,
    )
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise TweedieNumericalError("unit deviance could not be represented as finite")
    return result


def _scaled_response_factored_deviance(
    y: object,
    mu: object,
    power_minus_one: SupportsFloat,
    log_correction: object,
) -> NDArray[np.float64]:
    """Compose ``2 * y * mu**(1 - p) * correction`` in float64."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        correction = np.exp(np.asarray(log_correction, dtype=np.float64))
    return _scaled_power_deviance(
        mu,
        -float(power_minus_one),
        (2.0, y, correction),
    )


def compound_poisson_gamma_parameters(
    mu: object,
    phi: object,
    p: object,
    *,
    weights: object | None = None,
) -> CompoundPoissonGammaParameters:
    power = normalize_tweedie_power(p)
    mu_arr = _as_real_float64_array("mu", mu)
    weight_arr = (
        np.ones_like(mu_arr) if weights is None else _as_real_float64_array("weights", weights)
    )
    phi_raw = np.asarray(phi)
    if phi_raw.ndim == 0:
        dispersion: float | NDArray[np.float64] = normalize_positive_scalar("phi", phi)
    elif phi_raw.ndim == 1 and phi_raw.shape == mu_arr.shape:
        dispersion = _as_real_float64_array("phi", phi)
    else:
        raise ValueError("phi must be one positive real scalar or match the one-dimensional mu")
    if mu_arr.ndim != 1 or weight_arr.shape != mu_arr.shape:
        raise ValueError("mu and weights must be matching one-dimensional arrays")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")
    if np.any(~np.isfinite(dispersion)) or np.any(dispersion <= 0.0):
        raise ValueError("phi must be finite and strictly positive")
    if np.any(~np.isfinite(weight_arr)) or np.any(weight_arr <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    q = 2.0 - power
    r = power - 1.0
    mu_to_q = np.power(mu_arr, q)
    mu_to_r = np.power(mu_arr, r)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        effective_phi = dispersion / weight_arr
        rate = mu_to_q / (effective_phi * q)
        scale = effective_phi * r * mu_to_r

    invalid_rate = ~np.isfinite(rate) | (rate <= 0.0)
    if np.any(invalid_rate):
        stable_rate = _scaled_positive_ratio(
            mu_arr.shape,
            (mu_to_q, weight_arr),
            (dispersion, q),
        )
        rate[invalid_rate] = stable_rate[invalid_rate]
    invalid_scale = ~np.isfinite(scale) | (scale <= 0.0)
    if np.any(invalid_scale):
        stable_scale = _scaled_positive_ratio(
            mu_arr.shape,
            (dispersion, r, mu_to_r),
            (weight_arr,),
        )
        scale[invalid_scale] = stable_scale[invalid_scale]

    if np.any(~np.isfinite(rate)) or np.any(rate <= 0.0):
        raise TweedieNumericalError("Poisson rate could not be represented as finite and positive")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise TweedieNumericalError("Gamma scale could not be represented as finite and positive")
    shape = q / r
    return CompoundPoissonGammaParameters(rate=rate, shape=shape, scale=scale)


def _clamp_ulp_negative(
    values: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Clamp a negative no larger than one local ULP; reject anything material."""
    value_arr, reference_arr = np.broadcast_arrays(
        np.asarray(values),
        np.asarray(reference),
    )
    absolute_reference = np.abs(reference_arr)
    if np.any(~np.isfinite(absolute_reference)):
        raise TweedieNumericalError("unit deviance ratio reference was not finite")
    with np.errstate(over="ignore", invalid="ignore"):
        next_larger = np.nextafter(
            absolute_reference,
            np.full_like(absolute_reference, np.inf),
        )
        upward_ulp = next_larger - absolute_reference
    downward_ulp = absolute_reference - np.nextafter(
        absolute_reference,
        np.zeros_like(absolute_reference),
    )
    tolerance: NDArray[np.float64] = np.where(
        np.isfinite(upward_ulp),
        upward_ulp,
        downward_ulp,
    )
    if np.any(value_arr < -tolerance):
        raise TweedieNumericalError("unit deviance ratio became materially negative")
    result: NDArray[np.float64] = np.maximum(value_arr, 0.0)
    return result


def _certified_log_positive(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Take a logarithm only after certifying a dimensionless positive factor."""
    certified = _clamp_ulp_negative(values, np.ones_like(values))
    if np.any(certified <= 0.0) or np.any(~np.isfinite(certified)):
        raise TweedieNumericalError("unit deviance ratio was not certified positive")
    result: NDArray[np.float64] = np.log(certified)
    return result


def _series_deviance_ratio(log_ratio: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Return half-deviance divided by ``mu**q`` as a ratio series.

    Dividing each coefficient by ``(1 - p) * q`` analytically turns it into
    ``1 + q + ... + q**(k - 2)``. That geometric sum stays well-conditioned
    at both power boundaries.
    """
    term: NDArray[np.float64] = log_ratio * log_ratio / 2.0
    series: NDArray[np.float64] = term.copy()
    geometric_sum = 1.0
    q_power = 1.0
    for order in range(3, 32):
        term *= log_ratio / order
        q_power *= q
        geometric_sum += q_power
        series += geometric_sum * term
    return series


def _close_deviance_ratio(log_ratio: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Return the ratio series used by the explicit close-ratio branch."""
    return _series_deviance_ratio(log_ratio, q)


def _lower_boundary_composition(
    log_ratio: NDArray[np.float64], r: float, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compose after factoring ``r = p - 1`` around the lower boundary.

    The second result contains the correction after a leading
    ``exp(log_ratio)``. The caller combines that response power with
    ``mu**q`` before rounding whenever it is present.
    """
    result = np.empty_like(log_ratio)
    above_correction = np.full_like(log_ratio, np.nan)
    series = np.abs(log_ratio) <= _SERIES_LOG_RATIO_BOUNDARY
    if np.any(series):
        ratio_deviance = _series_deviance_ratio(log_ratio[series], q)
        result[series] = _certified_log_positive(ratio_deviance)

    above = ~series & (log_ratio > 0.0)
    if np.any(above):
        z = log_ratio[above]
        factor = -np.expm1(-r * z) / r + np.expm1(-z)
        correction = _certified_log_positive(factor) - np.log(q)
        above_correction[above] = correction
        result[above] = z + correction

    below = ~series & ~above
    if np.any(below):
        z = log_ratio[below]
        scaled_term = np.empty_like(z)
        exponent = -r * z
        direct = exponent <= 0.5
        if np.any(direct):
            scaled_term[direct] = np.exp(z[direct]) * np.expm1(exponent[direct]) / r
        if np.any(~direct):
            scaled_term[~direct] = (np.exp(q * z[~direct]) - np.exp(z[~direct])) / r
        factor = -np.expm1(z) - scaled_term
        result[below] = _certified_log_positive(factor) - np.log(q)
    return result, above_correction


def _lower_power_log_deviance_ratio(
    log_ratio: NDArray[np.float64], r: float, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the lower-power log ratio and response-power correction."""
    return _lower_boundary_composition(log_ratio, r, q)


def _ordinary_log_deviance_ratio(
    log_ratio: NDArray[np.float64], r: float, q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ordinary log ratio and response-power correction."""
    if r <= q:
        return _lower_boundary_composition(log_ratio, r, q)

    result = np.empty_like(log_ratio)
    above_correction = np.full_like(log_ratio, np.nan)
    series = np.abs(log_ratio) <= _SERIES_LOG_RATIO_BOUNDARY
    if np.any(series):
        ratio_deviance = _series_deviance_ratio(log_ratio[series], q)
        result[series] = _certified_log_positive(ratio_deviance)

    above = ~series & (log_ratio > 0.0)
    if np.any(above):
        z = log_ratio[above]
        factor = -np.expm1(-z) - np.exp(-r * z) * (-np.expm1(-q * z) / q)
        correction = _certified_log_positive(factor) - np.log(r)
        above_correction[above] = correction
        result[above] = z + correction

    below = ~series & ~above
    if np.any(below):
        z = log_ratio[below]
        factor = -np.expm1(q * z) / q + np.expm1(z)
        result[below] = _certified_log_positive(factor) - np.log(r)
    return result, above_correction


def _representable_unit_deviance(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentiate a final log deviance only when float64 can represent it."""
    value_dtype = np.asarray(log_values).dtype
    log_float_max = np.log(np.asarray(np.finfo(np.float64).max, dtype=value_dtype))
    if np.any(~np.isfinite(log_values)) or np.any(log_values > log_float_max):
        raise TweedieNumericalError("unit deviance could not be represented as finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        extended_result = np.exp(log_values)
        result: NDArray[np.float64] = np.asarray(extended_result, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise TweedieNumericalError("unit deviance could not be represented as finite")
    return result


def tweedie_unit_deviance(y: object, mu: object, p: object) -> NDArray[np.float64]:
    power = normalize_tweedie_power(p)
    y_arr, mu_arr = np.broadcast_arrays(
        _as_real_float64_array("y", y),
        _as_real_float64_array("mu", mu),
    )
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and nonnegative")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")

    work_type = _DEVIANCE_DTYPE.type
    work_power = work_type(power)
    two = work_type(2.0)
    q = two - work_power
    result = np.empty(y_arr.shape, dtype=np.float64)
    equal = y_arr == mu_arr
    zero = (y_arr == 0.0) & ~equal
    result[equal] = 0.0
    if np.any(zero):
        result[zero] = _scaled_power_deviance(
            mu_arr[zero],
            q,
            (2.0,),
            (q,),
        )

    positive = ~(equal | zero)
    if np.any(positive):
        positive_y = np.asarray(y_arr[positive], dtype=_DEVIANCE_DTYPE)
        positive_mu = np.asarray(mu_arr[positive], dtype=_DEVIANCE_DTYPE)
        log_y = np.log(positive_y)
        log_mu = np.log(positive_mu)
        log_ratio = log_y - log_mu
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            ratio = positive_y / positive_mu
        work_tiny = work_type(np.finfo(_DEVIANCE_DTYPE).tiny)
        usable_ratio = np.isfinite(ratio) & (ratio >= work_tiny)
        if np.any(usable_ratio):
            log_ratio[usable_ratio] = np.log(ratio[usable_ratio])

        close = np.abs(log_ratio) <= 1e-3
        if np.any(close):
            # Subtracting nearby logarithms loses the precision needed by the remainder series.
            relative_difference = (positive_y[close] - positive_mu[close]) / positive_mu[close]
            log_ratio[close] = np.log1p(relative_difference)

        r = work_power - work_type(1.0)
        lower_power = ~close & (r <= _LOWER_POWER_BOUNDARY)
        ordinary = ~(close | lower_power)
        log_ratio_deviance = np.empty_like(log_ratio)
        above_correction = np.full_like(log_ratio, np.nan)

        if np.any(close):
            ratio_deviance = _close_deviance_ratio(log_ratio[close], q)
            log_ratio_deviance[close] = _certified_log_positive(ratio_deviance)
        if np.any(lower_power):
            branch_log_ratio, branch_correction = _lower_power_log_deviance_ratio(
                log_ratio[lower_power], r, q
            )
            log_ratio_deviance[lower_power] = branch_log_ratio
            above_correction[lower_power] = branch_correction
        if np.any(ordinary):
            branch_log_ratio, branch_correction = _ordinary_log_deviance_ratio(
                log_ratio[ordinary], r, q
            )
            log_ratio_deviance[ordinary] = branch_log_ratio
            above_correction[ordinary] = branch_correction

        log_half_deviance = q * log_mu + log_ratio_deviance
        response_factored = np.isfinite(above_correction)
        if np.any(response_factored):
            log_half_deviance[response_factored] = (
                log_y[response_factored]
                - r * log_mu[response_factored]
                + above_correction[response_factored]
            )
        log_deviance = np.log(two) + log_half_deviance

        values = np.empty_like(log_ratio, dtype=np.float64)
        direct_usable = np.zeros(log_ratio.shape, dtype=np.bool_)
        if np.any(response_factored):
            values[response_factored] = _scaled_response_factored_deviance(
                positive_y[response_factored],
                positive_mu[response_factored],
                r,
                above_correction[response_factored],
            )
            direct_usable[response_factored] = True
        series_region = np.abs(log_ratio) <= _SERIES_LOG_RATIO_BOUNDARY
        mean_factored = ~(series_region | response_factored)
        if np.any(mean_factored):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                mean_correction = np.exp(
                    np.asarray(log_ratio_deviance[mean_factored], dtype=np.float64)
                )
            values[mean_factored] = _scaled_power_deviance(
                positive_mu[mean_factored],
                q,
                (2.0, mean_correction),
            )
            direct_usable[mean_factored] = True
        if np.any(series_region):
            ratio_deviance = _series_deviance_ratio(log_ratio[series_region], q)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                scale = np.power(positive_mu[series_region], q)
                direct_extended = scale * (two * ratio_deviance)
                direct_float = np.asarray(direct_extended, dtype=np.float64)
            usable = (
                (scale >= work_tiny)
                & np.isfinite(direct_extended)
                & np.isfinite(direct_float)
                & (direct_float > 0.0)
            )
            series_indices = np.flatnonzero(series_region)
            direct_indices = series_indices[usable]
            direct_usable[direct_indices] = True
            values[direct_indices] = direct_float[usable]

        unresolved = ~direct_usable
        if np.any(unresolved):
            values[unresolved] = _representable_unit_deviance(log_deviance[unresolved])
        result[positive] = values
    return result
