from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction
from numbers import Integral
from typing import NoReturn

import numpy as np
from numpy.typing import NDArray

from ._tweedie_numerics import (
    TweedieNumericalError,
    _as_real_float64_array,
    normalize_positive_scalar,
    normalize_real_scalar,
    normalize_tweedie_power,
    tweedie_unit_deviance,
)

_DEFAULT_RTOL = 1e-12
_DEFAULT_MAX_TERMS = 1_000_000
_MIN_RTOL = float(16.0 * np.finfo(np.float64).eps)
_MAX_SAFE_INDEX = 2**52
_LOG_TWO_PI = math.log(2.0 * math.pi)
_FLOAT_EPS = np.finfo(np.float64).eps
_LONGDOUBLE_EPS = np.finfo(np.longdouble).eps
_FLOAT_MAX = np.finfo(np.float64).max
_DECIMAL_PARAMETER_PRECISION = 96
_DIRECT_REANCHOR_INTERVAL = 4
_BESSEL_ASYMPTOTIC_MIN_ARGUMENT = Decimal("1000")
_BESSEL_ASYMPTOTIC_TERMS = 12
_DECIMAL_LOG_TWO_PI = Decimal(
    "1.8378770664093454835606594728112352797227949472755668256343030809655313918545"
)


@dataclass(frozen=True)
class TweedieDensityDiagnostics:
    n_positive: int
    n_exact: int
    n_approximate: int
    max_terms: int
    exact: bool
    certified: bool
    requested_rtol: float
    max_relative_tail_error: float
    method: str = "compound_poisson_series"


@dataclass(frozen=True)
class TweedieDensityEvaluation:
    logpdf: NDArray[np.float64]
    log_phi_score: NDArray[np.float64]
    diagnostics: TweedieDensityDiagnostics


class TweedieDensityError(RuntimeError):
    """An exact Tweedie density evaluation could not certify its arithmetic."""

    def __init__(
        self,
        *,
        observation_index: int,
        power: float,
        dispersion: float,
        term_count: int,
        requested_rtol: float,
        reason: str,
    ) -> None:
        self.observation_index = observation_index
        self.power = power
        self.dispersion = dispersion
        self.term_count = term_count
        self.requested_rtol = requested_rtol
        self.reason = reason
        super().__init__(
            "Tweedie density certification failed at observation "
            f"{observation_index} after {term_count} terms: {reason}"
        )


@dataclass(frozen=True)
class _CompoundParameters:
    alpha: np.longdouble
    lam: np.longdouble
    scaled_y: np.longdouble
    rate_numerator: np.longdouble
    scaled_y_numerator: np.longdouble
    one_minus_power_offset: np.longdouble
    two_minus_power: np.longdouble
    log_alpha: np.longdouble
    log_lam: np.longdouble
    log_scaled_y: np.longdouble
    log_y: np.longdouble
    alpha_decimal: Decimal
    lam_decimal: Decimal
    scaled_y_decimal: Decimal
    log_y_decimal: Decimal
    y_input: float
    mu_input: float
    phi_input: float
    power_input: float
    weight_input: float


@dataclass(frozen=True)
class _SeriesResult:
    logpdf: float
    log_phi_score: float
    term_count: int
    relative_error: float
    method: str = "compound_poisson_series"


_LogTerm = tuple[np.longdouble, float]


def _readonly(values: object) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_density_arrays(
    y: object,
    mu: object,
    weights: object | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    y_arr = _as_real_float64_array("y", y)
    mu_arr = _as_real_float64_array("mu", mu)
    weight_arr = (
        np.ones(y_arr.shape, dtype=np.float64)
        if weights is None
        else _as_real_float64_array("weights", weights)
    )

    if y_arr.ndim != 1 or mu_arr.ndim != 1 or weight_arr.ndim != 1:
        raise ValueError("y, mu, and weights must be one-dimensional arrays")
    if mu_arr.shape != y_arr.shape or weight_arr.shape != y_arr.shape:
        raise ValueError("y, mu, and weights must have the same length")
    if np.any(~np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and nonnegative")
    if np.any(~np.isfinite(mu_arr)) or np.any(mu_arr <= 0.0):
        raise ValueError("mu must be finite and strictly positive")
    if np.any(~np.isfinite(weight_arr)) or np.any(weight_arr <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    return y_arr, mu_arr, weight_arr


def _normalize_rtol(value: object) -> float:
    result = normalize_real_scalar("rtol", value)
    if not 0.0 < result < 1.0:
        raise ValueError("rtol must be strictly between zero and one")
    if result < _MIN_RTOL:
        raise ValueError(f"rtol must be at least {_MIN_RTOL!r} for float64 output")
    return result


def _normalize_max_terms(value: object) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Integral):
        raise TypeError("max_terms must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError("max_terms must be strictly positive")
    if result > _DEFAULT_MAX_TERMS:
        raise ValueError(f"max_terms must not exceed {_DEFAULT_MAX_TERMS}")
    return result


def _raise_density_error(
    *,
    observation_index: int,
    power: float,
    dispersion: float,
    term_count: int,
    requested_rtol: float,
    reason: str,
) -> NoReturn:
    raise TweedieDensityError(
        observation_index=observation_index,
        power=power,
        dispersion=dispersion,
        term_count=term_count,
        requested_rtol=requested_rtol,
        reason=reason,
    )


def _compound_parameters(
    y: float,
    mu: float,
    phi: float,
    power: float,
    weight: float,
    *,
    observation_index: int,
    requested_rtol: float,
) -> _CompoundParameters:
    try:
        with localcontext() as context:
            context.prec = _DECIMAL_PARAMETER_PRECISION
            y_decimal = Decimal.from_float(y)
            mu_decimal = Decimal.from_float(mu)
            phi_decimal = Decimal.from_float(phi)
            power_decimal = Decimal.from_float(power)
            weight_decimal = Decimal.from_float(weight)
            one = Decimal(1)
            two = Decimal(2)
            one_minus_power_offset_decimal = power_decimal - one
            two_minus_power_decimal = two - power_decimal
            effective_phi = phi_decimal / weight_decimal
            log_mu = mu_decimal.ln()
            mu_to_two_minus_p = (two_minus_power_decimal * log_mu).exp()
            mu_to_p_minus_one = (one_minus_power_offset_decimal * log_mu).exp()
            rate_numerator_decimal = mu_to_two_minus_p / effective_phi
            scaled_y_numerator_decimal = y_decimal / (effective_phi * mu_to_p_minus_one)
            alpha_decimal = two_minus_power_decimal / one_minus_power_offset_decimal
            lam_decimal = rate_numerator_decimal / two_minus_power_decimal
            scaled_y_decimal = scaled_y_numerator_decimal / one_minus_power_offset_decimal
            decimal_values = (
                alpha_decimal,
                lam_decimal,
                scaled_y_decimal,
                rate_numerator_decimal,
                scaled_y_numerator_decimal,
                one_minus_power_offset_decimal,
                two_minus_power_decimal,
                alpha_decimal.ln(),
                lam_decimal.ln(),
                scaled_y_decimal.ln(),
                y_decimal.ln(),
            )
    except (DecimalException, OverflowError, ValueError):
        _raise_density_error(
            observation_index=observation_index,
            power=power,
            dispersion=phi,
            term_count=0,
            requested_rtol=requested_rtol,
            reason="compound parameters were not representable",
        )

    with np.errstate(all="ignore"):
        (
            alpha,
            lam,
            scaled_y,
            rate_numerator,
            scaled_y_numerator,
            one_minus_power_offset,
            two_minus_power,
            log_alpha,
            log_lam,
            log_scaled_y,
            log_y,
        ) = (np.longdouble(str(value)) for value in decimal_values)

    parameters = (alpha, lam, scaled_y)
    if any(
        not bool(np.isfinite(value)) or value <= 0.0 or value > np.longdouble(_FLOAT_MAX)
        for value in parameters
    ):
        _raise_density_error(
            observation_index=observation_index,
            power=power,
            dispersion=phi,
            term_count=0,
            requested_rtol=requested_rtol,
            reason="compound parameters were not representable",
        )

    if not all(bool(np.isfinite(value)) for value in (log_alpha, log_lam, log_scaled_y, log_y)):
        _raise_density_error(
            observation_index=observation_index,
            power=power,
            dispersion=phi,
            term_count=0,
            requested_rtol=requested_rtol,
            reason="compound parameters were not representable",
        )
    return _CompoundParameters(
        alpha=alpha,
        lam=lam,
        scaled_y=scaled_y,
        rate_numerator=rate_numerator,
        scaled_y_numerator=scaled_y_numerator,
        one_minus_power_offset=one_minus_power_offset,
        two_minus_power=two_minus_power,
        log_alpha=log_alpha,
        log_lam=log_lam,
        log_scaled_y=log_scaled_y,
        log_y=log_y,
        alpha_decimal=alpha_decimal,
        lam_decimal=lam_decimal,
        scaled_y_decimal=scaled_y_decimal,
        log_y_decimal=decimal_values[-1],
        y_input=y,
        mu_input=mu,
        phi_input=phi,
        power_input=power,
        weight_input=weight,
    )


def _stirling_error(value: float) -> float:
    """Return the positive correction to the leading log-Gamma expansion."""
    shifted = np.longdouble(value)
    shift = max(0, int(math.ceil(16.0 - value)))
    shifted += np.longdouble(shift)
    inverse = np.longdouble(1.0) / shifted
    inverse_squared = inverse * inverse
    polynomial = np.longdouble(1.0) / np.longdouble(12.0) + inverse_squared * (
        -np.longdouble(1.0) / np.longdouble(360.0)
        + inverse_squared
        * (
            np.longdouble(1.0) / np.longdouble(1260.0)
            + inverse_squared
            * (
                -np.longdouble(1.0) / np.longdouble(1680.0)
                + inverse_squared
                * (
                    np.longdouble(1.0) / np.longdouble(1188.0)
                    - inverse_squared * np.longdouble(691.0) / np.longdouble(360360.0)
                )
            )
        )
    )
    result = inverse * polynomial
    original = np.longdouble(value)
    for offset in range(shift):
        point = original + np.longdouble(offset)
        result += (point + np.longdouble(0.5)) * np.log1p(
            np.longdouble(1.0) / point
        ) - np.longdouble(1.0)
    return float(result)


def _stable_positive_log_ratio(
    numerator: np.longdouble,
    denominator: np.longdouble,
    fallback_log_ratio: np.longdouble,
) -> np.longdouble:
    relative_difference = (numerator - denominator) / denominator
    if abs(relative_difference) <= np.longdouble(0.5):
        with np.errstate(all="ignore"):
            return np.longdouble(np.log1p(relative_difference))
    return fallback_log_ratio


def _stable_log_term(
    index: int,
    parameters: _CompoundParameters,
) -> _LogTerm:
    return _high_precision_log_term(index, parameters)


def _high_precision_log_term(
    index: int,
    parameters: _CompoundParameters,
) -> _LogTerm:
    with localcontext() as context:
        context.prec = _DECIMAL_PARAMETER_PRECISION
        count = Decimal(index)
        gamma_shape_decimal = count * parameters.alpha_decimal
        poisson_center = (
            -parameters.lam_decimal + count + count * (parameters.lam_decimal.ln() - count.ln())
        )
        gamma_center = (
            -parameters.scaled_y_decimal
            + gamma_shape_decimal
            + gamma_shape_decimal * (parameters.scaled_y_decimal.ln() - gamma_shape_decimal.ln())
        )
        core_decimal = (
            poisson_center
            - (_DECIMAL_LOG_TWO_PI + count.ln()) / Decimal(2)
            + gamma_center
            + gamma_shape_decimal.ln() / Decimal(2)
            - parameters.log_y_decimal
            - _DECIMAL_LOG_TWO_PI / Decimal(2)
        )

    with np.errstate(all="ignore"):
        core = np.longdouble(str(core_decimal))
        gamma_shape = np.longdouble(str(gamma_shape_decimal))
    if (
        not bool(np.isfinite(core))
        or not bool(np.isfinite(gamma_shape))
        or gamma_shape <= 0.0
        or gamma_shape > np.longdouble(_FLOAT_MAX)
    ):
        raise ArithmeticError("non-finite series arithmetic")

    poisson_stirling_error = np.longdouble(_stirling_error(float(index)))
    gamma_stirling_error = np.longdouble(_stirling_error(float(gamma_shape)))
    value = core - poisson_stirling_error - gamma_stirling_error
    error = (
        np.longdouble(4.0) * abs(np.spacing(core))
        + np.longdouble(64.0)
        * np.longdouble(_LONGDOUBLE_EPS)
        * (np.longdouble(1.0) + abs(core) + abs(poisson_stirling_error) + abs(gamma_stirling_error))
        + np.longdouble(16.0)
        * np.longdouble(_FLOAT_EPS)
        * (abs(poisson_stirling_error) + abs(gamma_stirling_error) + np.longdouble(_FLOAT_EPS))
    )
    return np.longdouble(value), float(error)


def _one_minus_scaled_log1p(index: int) -> np.longdouble:
    """Return ``1 - (index + 1) * log1p(1 / index)`` stably."""
    inverse = np.longdouble(1.0) / np.longdouble(index)
    if inverse > np.longdouble(0.01):
        return np.longdouble(np.longdouble(1.0) - np.longdouble(index + 1) * np.log1p(inverse))

    term = inverse
    result = np.longdouble(0.0)
    for order in range(1, 20):
        coefficient = np.longdouble((-1) ** order) / np.longdouble(order * (order + 1))
        result += coefficient * term
        term *= inverse
    return result


def _stable_log_gamma_increment(
    shape: np.longdouble,
    increment: np.longdouble,
) -> tuple[np.longdouble, np.longdouble]:
    shift = max(0, int(math.ceil(16.0 - float(shape))))
    shifted_shape = shape + np.longdouble(shift)
    shifted_next = shifted_shape + increment
    log_increment = np.log1p(increment / shifted_shape)
    first_error = np.longdouble(_stirling_error(float(shifted_shape)))
    next_error = np.longdouble(_stirling_error(float(shifted_next)))
    value = (
        increment * np.log(shifted_shape)
        + (shifted_next - np.longdouble(0.5)) * log_increment
        - increment
        + next_error
        - first_error
    )
    recurrence_magnitude = np.longdouble(0.0)
    for offset in range(shift):
        recurrence_term = np.log1p(increment / (shape + np.longdouble(offset)))
        value -= recurrence_term
        recurrence_magnitude += abs(recurrence_term)
    error = np.longdouble(128.0) * np.longdouble(_LONGDOUBLE_EPS) * (
        np.longdouble(1.0)
        + abs(value)
        + abs(increment * np.log(shifted_shape))
        + recurrence_magnitude
    ) + np.longdouble(8.0) * np.longdouble(_FLOAT_EPS) * (
        abs(first_error) + abs(next_error) + np.longdouble(_FLOAT_EPS)
    )
    return np.longdouble(value), np.longdouble(error)


def _stable_log_forward_ratio(
    index: int,
    parameters: _CompoundParameters,
) -> tuple[np.longdouble, np.longdouble]:
    index_ld = np.longdouble(index)
    next_index_ld = np.longdouble(index + 1)
    gamma_shape = index_ld * parameters.alpha
    next_gamma_shape = next_index_ld * parameters.alpha
    if not bool(np.isfinite(next_gamma_shape)) or next_gamma_shape > np.longdouble(_FLOAT_MAX):
        raise ArithmeticError("non-finite series arithmetic")

    poisson_ratio = _stable_positive_log_ratio(
        parameters.lam,
        next_index_ld,
        parameters.log_lam - np.log(next_index_ld),
    )
    gamma_scale_ratio = _stable_positive_log_ratio(
        parameters.scaled_y,
        gamma_shape,
        parameters.log_scaled_y - np.log(gamma_shape),
    )
    lam_radius = np.longdouble(4.0) * abs(np.spacing(parameters.lam))
    scaled_y_radius = np.longdouble(4.0) * abs(np.spacing(parameters.scaled_y))
    alpha_radius = abs(np.spacing(parameters.alpha))
    parameter_error = lam_radius / parameters.lam

    if parameters.alpha == np.longdouble(1.0):
        log_ratio = poisson_ratio + gamma_scale_ratio
        parameter_error += scaled_y_radius / parameters.scaled_y
        error = (
            np.longdouble(64.0)
            * np.longdouble(_LONGDOUBLE_EPS)
            * (np.longdouble(1.0) + abs(poisson_ratio) + abs(gamma_scale_ratio))
            + parameter_error
        )
        return np.longdouble(log_ratio), np.longdouble(error)

    if gamma_shape < np.longdouble(16.0):
        gamma_difference, gamma_error = _stable_log_gamma_increment(
            gamma_shape,
            parameters.alpha,
        )
        log_ratio = poisson_ratio + parameters.alpha * parameters.log_scaled_y - gamma_difference
        digamma_bound = (
            abs(parameters.log_scaled_y)
            + index_ld
            * (abs(np.log(gamma_shape)) + np.longdouble(1.0) / gamma_shape + np.longdouble(1.0))
            + next_index_ld
            * (
                abs(np.log(next_gamma_shape))
                + np.longdouble(1.0) / next_gamma_shape
                + np.longdouble(1.0)
            )
        )
        parameter_error += (
            parameters.alpha * scaled_y_radius / parameters.scaled_y + alpha_radius * digamma_bound
        )
        error = (
            gamma_error
            + np.longdouble(64.0)
            * np.longdouble(_LONGDOUBLE_EPS)
            * (
                np.longdouble(1.0)
                + abs(poisson_ratio)
                + abs(parameters.alpha * parameters.log_scaled_y)
                + abs(gamma_difference)
            )
            + parameter_error
        )
        return np.longdouble(log_ratio), np.longdouble(error)

    log_increment = np.log1p(np.longdouble(1.0) / index_ld)
    stirling_difference = np.longdouble(
        _stirling_error(float(next_gamma_shape)) - _stirling_error(float(gamma_shape))
    )
    correction = (
        parameters.alpha * _one_minus_scaled_log1p(index)
        + np.longdouble(0.5) * log_increment
        - stirling_difference
    )
    log_ratio = poisson_ratio + parameters.alpha * gamma_scale_ratio + correction
    gamma_shape_radius = index_ld * alpha_radius + abs(np.spacing(gamma_shape))
    parameter_error += parameters.alpha * (
        scaled_y_radius / parameters.scaled_y + gamma_shape_radius / gamma_shape
    ) + alpha_radius * (
        abs(gamma_scale_ratio)
        + abs(_one_minus_scaled_log1p(index))
        + np.longdouble(1.0) / gamma_shape
        + np.longdouble(1.0) / next_gamma_shape
    )
    error = (
        np.longdouble(128.0)
        * np.longdouble(_LONGDOUBLE_EPS)
        * (
            np.longdouble(1.0)
            + abs(poisson_ratio)
            + abs(parameters.alpha * gamma_scale_ratio)
            + abs(correction)
        )
        + np.longdouble(16.0)
        * np.longdouble(_FLOAT_EPS)
        * (
            abs(np.longdouble(_stirling_error(float(next_gamma_shape))))
            + abs(np.longdouble(_stirling_error(float(gamma_shape))))
            + np.longdouble(_FLOAT_EPS)
        )
        + parameter_error
    )
    return np.longdouble(log_ratio), np.longdouble(error)


def _forward_ratio_interval(
    index: int,
    parameters: _CompoundParameters,
) -> tuple[np.longdouble, np.longdouble, np.longdouble]:
    ratio, error = _stable_log_forward_ratio(index, parameters)
    return ratio, ratio - error, ratio + error


def _upper_series_term(
    index: int,
    term: _LogTerm,
    parameters: _CompoundParameters,
) -> _LogTerm:
    ratio, ratio_error = _stable_log_forward_ratio(index, parameters)
    return term[0] + ratio, float(np.longdouble(term[1]) + ratio_error)


def _lower_series_term(
    index: int,
    term: _LogTerm,
    parameters: _CompoundParameters,
) -> _LogTerm:
    ratio, ratio_error = _stable_log_forward_ratio(index - 1, parameters)
    return term[0] - ratio, float(np.longdouble(term[1]) + ratio_error)


def _find_mode_index(parameters: _CompoundParameters) -> int:
    try:
        ratio, _, _ = _forward_ratio_interval(1, parameters)
    except (ArithmeticError, OverflowError, ValueError) as exc:
        raise ArithmeticError("series mode could not be bracketed safely") from exc
    if ratio <= 0.0:
        return 1

    low = 1
    high = 2
    while True:
        if high >= _MAX_SAFE_INDEX:
            raise ArithmeticError("series mode could not be bracketed safely")
        try:
            ratio, _, _ = _forward_ratio_interval(high, parameters)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise ArithmeticError("series mode could not be bracketed safely") from exc
        if ratio <= 0.0:
            break
        low = high + 1
        high *= 2

    while low < high:
        middle = (low + high) // 2
        try:
            ratio, _, _ = _forward_ratio_interval(middle, parameters)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise ArithmeticError("series mode could not be bracketed safely") from exc
        if ratio > 0.0:
            low = middle + 1
        else:
            high = middle
    return low


def _geometric_tail_bounds(
    *,
    mode: int,
    low: int,
    high: int,
    log_mode_term: np.longdouble,
    low_term: _LogTerm,
    high_term: _LogTerm,
    lower_candidate: _LogTerm | None,
    upper_candidate: _LogTerm,
) -> tuple[np.longdouble, np.longdouble]:
    mass = np.longdouble(0.0)
    centered_moment = np.longdouble(0.0)

    if lower_candidate is not None:
        lower_log, lower_error = lower_candidate
        low_log, low_error = low_term
        log_ratio_upper = lower_log - low_log + lower_error + low_error
        if log_ratio_upper >= 0.0:
            return np.longdouble(np.inf), np.longdouble(np.inf)
        with np.errstate(all="ignore"):
            ratio = np.exp(log_ratio_upper)
            denominator = np.longdouble(1.0) - ratio
            boundary_weight = np.exp(low_log - log_mode_term + np.longdouble(low_error))
            lower_mass = boundary_weight * ratio / denominator
            distance = np.longdouble(mode - low)
            lower_center = (
                boundary_weight
                * ratio
                * (distance / denominator + np.longdouble(1.0) / denominator**2)
            )
        mass += lower_mass
        centered_moment += lower_center

    upper_log, upper_error = upper_candidate
    high_log, high_error = high_term
    log_ratio_upper = upper_log - high_log + upper_error + high_error
    if log_ratio_upper >= 0.0:
        return np.longdouble(np.inf), np.longdouble(np.inf)
    with np.errstate(all="ignore"):
        ratio = np.exp(log_ratio_upper)
        denominator = np.longdouble(1.0) - ratio
        boundary_weight = np.exp(high_log - log_mode_term + np.longdouble(high_error))
        upper_mass = boundary_weight * ratio / denominator
        distance = np.longdouble(high - mode)
        upper_center = (
            boundary_weight * ratio * (distance / denominator + np.longdouble(1.0) / denominator**2)
        )
    mass += upper_mass
    centered_moment += upper_center
    return mass, centered_moment


def _compensated_add(
    total: np.longdouble,
    compensation: np.longdouble,
    value: np.longdouble,
) -> tuple[np.longdouble, np.longdouble]:
    adjusted = value - compensation
    updated = total + adjusted
    return updated, np.longdouble((updated - total) - adjusted)


def _decimal_score_base(
    parameters: _CompoundParameters,
    mode: int,
    *,
    precision: int,
) -> Decimal:
    with localcontext() as context:
        context.prec = precision
        y = Decimal.from_float(parameters.y_input)
        mu = Decimal.from_float(parameters.mu_input)
        phi = Decimal.from_float(parameters.phi_input)
        power = Decimal.from_float(parameters.power_input)
        weight = Decimal.from_float(parameters.weight_input)
        one = Decimal(1)
        two = Decimal(2)
        one_minus_power_offset = power - one
        two_minus_power = two - power
        effective_phi = phi / weight
        log_mu = mu.ln()
        rate_numerator = (two_minus_power * log_mu).exp() / effective_phi
        scaled_y_numerator = y / (effective_phi * (one_minus_power_offset * log_mu).exp())
        numerator = (
            rate_numerator * one_minus_power_offset
            + (scaled_y_numerator - Decimal(mode)) * two_minus_power
        )
        return numerator / (two_minus_power * one_minus_power_offset)


def _score_base_and_radius(
    parameters: _CompoundParameters,
    mode: int,
) -> tuple[np.longdouble, np.longdouble]:
    previous: Decimal | None = None
    for precision in (80, 128, 192, 288, 432, 648, 972):
        current = _decimal_score_base(parameters, mode, precision=precision)
        with np.errstate(all="ignore"):
            value = np.longdouble(str(current))
        if not bool(np.isfinite(value)) or abs(value) > np.longdouble(_FLOAT_MAX):
            raise ArithmeticError("non-finite series arithmetic")
        if previous is not None:
            with localcontext() as context:
                context.prec = precision
                decimal_difference = abs(current - previous)
            with np.errstate(all="ignore"):
                difference = abs(np.longdouble(str(decimal_difference)))
                spacing = abs(np.spacing(value))
            radius = difference + np.longdouble(4.0) * spacing
            if bool(np.isfinite(radius)) and difference <= spacing:
                return value, radius
        previous = current
    raise ArithmeticError("score base could not be certified")


def _alpha_one_term_budget_is_provably_insufficient(
    parameters: _CompoundParameters,
    mode: int,
    max_terms: int,
    requested_rtol: float,
) -> bool:
    """Prove that an alpha-one series cannot certify within ``max_terms``.

    At ``p=1.5`` the positive-series terms have forward ratio
    ``x / (j * (j + 1))``, where ``x = 4 * y * weight**2 / phi**2``.
    Exact rational arithmetic independently locates the mode, including cases
    where the general floating-point search selects the equally high adjacent
    term at an exact tie.  A monotone lower bound on the next ``max_terms``
    upper-side terms then proves that every truncation of that size omits more
    relative mass than ``requested_rtol``.  Returning ``False`` means only that
    this narrow proof is inconclusive; it never certifies a series by itself.
    """
    if (
        parameters.power_input != 1.5
        or parameters.y_input <= 0.0
        or mode <= 1
        or mode <= max_terms + 1
    ):
        return False

    y = Fraction.from_float(parameters.y_input)
    weight = Fraction.from_float(parameters.weight_input)
    phi = Fraction.from_float(parameters.phi_input)
    series_argument = 4 * y * weight * weight / (phi * phi)
    numerator = series_argument.numerator
    denominator = series_argument.denominator

    exact_mode = max(1, math.isqrt(numerator // denominator))
    while exact_mode * (exact_mode + 1) * denominator < numerator:
        exact_mode += 1
    while exact_mode > 1 and (exact_mode - 1) * exact_mode * denominator >= numerator:
        exact_mode -= 1
    if exact_mode <= 1 or exact_mode <= max_terms + 1:
        return False
    mode = exact_mode

    q = Fraction(
        mode * (mode - 1),
        (mode + max_terms - 1) * (mode + max_terms),
    )
    delta = 1 - q
    if max_terms * delta > 1:
        return False

    binomial_lower_bound = (
        1
        - max_terms * delta
        + Fraction(max_terms * (max_terms - 1), 2) * delta**2
        - Fraction(max_terms * (max_terms - 1) * (max_terms - 2), 6) * delta**3
    )
    return binomial_lower_bound > max_terms * Fraction.from_float(requested_rtol)


def _bessel_asymptotic_interval(
    order: int,
    argument: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    """Bound ``exp(-x) * sqrt(2*pi*x) * I_order(x)`` for large ``x``.

    The finite sum is Hankel's large-argument expansion.  The remainder bound
    follows by applying the DLMF 10.40.10--12 bound to ``K_order(-x)`` and the
    exact integer-order continuation identity relating that value to
    ``I_order(x)``.  Only orders zero and one are needed by the alpha-one
    Tweedie score.  The deliberately loose ``terms + 1`` bound for
    ``chi(terms)`` keeps the implementation small while remaining far below
    the requested tolerance in this branch.
    """
    coefficient = Decimal(1)
    inverse_power = Decimal(1)
    dominant_sum = Decimal(1)
    dominant_correction = Decimal(0)
    recessive_sum = Decimal(1)
    next_term = Decimal(0)

    for index in range(1, _BESSEL_ASYMPTOTIC_TERMS + 1):
        odd = 2 * index - 1
        coefficient *= Decimal(4 * order * order - odd * odd) / Decimal(8 * index)
        inverse_power /= argument
        term = coefficient * inverse_power
        if index < _BESSEL_ASYMPTOTIC_TERMS:
            signed_term = term if index % 2 == 0 else -term
            dominant_sum += signed_term
            dominant_correction += signed_term
            recessive_sum += term
        else:
            next_term = abs(term)

    # On the negative real ray the K remainder contributes at most
    # 4*chi(l)*|a_l|/x**l*exp(3*pi/(4*x)).  Here chi(l) < l + 1 and
    # 3*pi/4 < 4.  The positive-ray K term is exponentially recessive;
    # x >= 1000 makes exp(-2000) a conservative bound for exp(-2*x).
    continuation_radius = (
        Decimal(4 * (_BESSEL_ASYMPTOTIC_TERMS + 1)) * next_term * (Decimal(4) / argument).exp()
    )
    recessive_radius = Decimal(-2000).exp() * (abs(recessive_sum) + next_term)
    decimal_rounding_radius = Decimal(64).scaleb(-_DECIMAL_PARAMETER_PRECISION + 4)
    radius = continuation_radius + recessive_radius + decimal_rounding_radius
    return dominant_sum, dominant_correction, radius


def _certified_alpha_one_bessel(
    parameters: _CompoundParameters,
    *,
    requested_rtol: float,
    max_terms: int,
) -> _SeriesResult | None:
    """Return the exact alpha-one Bessel resummation when it certifies cheaply."""
    if parameters.power_input != 1.5 or max_terms < _BESSEL_ASYMPTOTIC_TERMS:
        return None

    try:
        with localcontext() as context:
            context.prec = _DECIMAL_PARAMETER_PRECISION
            y = Decimal.from_float(parameters.y_input)
            mu = Decimal.from_float(parameters.mu_input)
            effective_phi = Decimal.from_float(parameters.phi_input) / Decimal.from_float(
                parameters.weight_input
            )
            root_y = y.sqrt()
            root_mu = mu.sqrt()
            product = Decimal(4) * y / (effective_phi * effective_phi)
            argument = Decimal(4) * root_y / effective_phi
            if argument < _BESSEL_ASYMPTOTIC_MIN_ARGUMENT:
                return None

            scaled_i0, correction_i0, radius_i0 = _bessel_asymptotic_interval(0, argument)
            scaled_i1, correction_i1, radius_i1 = _bessel_asymptotic_interval(1, argument)
            lower_i0 = scaled_i0 - radius_i0
            lower_i1 = scaled_i1 - radius_i1
            upper_i1 = scaled_i1 + radius_i1
            if lower_i0 <= 0 or lower_i1 <= 0:
                return None

            root_difference = (y - mu) / (root_y + root_mu)
            squared_difference = root_difference * root_difference
            squared_difference *= Decimal(2) / (effective_phi * root_mu)
            correction_difference = correction_i1 - correction_i0
            score = squared_difference + argument * correction_difference / scaled_i1

            total_radius = radius_i0 + radius_i1
            # Preserve the shared I1 perturbation: q = 1 - I0 / I1 decreases
            # with I0 and increases with I1, so these paired endpoints bound q.
            score_lower = (
                squared_difference + argument * (correction_difference - total_radius) / lower_i1
            )
            score_upper = (
                squared_difference + argument * (correction_difference + total_radius) / upper_i1
            )
            score_radius = max(abs(score - score_lower), abs(score_upper - score))
            score_scale = max(Decimal(1), abs(score))
            score_relative_error = score_radius / score_scale
            density_relative_error = radius_i1 / lower_i1
            binary64_rounding = Decimal.from_float(8.0 * _FLOAT_EPS)
            relative_error = max(density_relative_error, score_relative_error) + binary64_rounding
            if relative_error > Decimal.from_float(requested_rtol):
                return None

            log_four_pi = _DECIMAL_LOG_TWO_PI + Decimal(2).ln()
            logpdf = (
                -squared_difference
                + Decimal("0.25") * product.ln()
                - Decimal("0.5") * log_four_pi
                + scaled_i1.ln()
                - parameters.log_y_decimal
            )
            logpdf_float = float(logpdf)
            score_float = float(score)
    except (DecimalException, OverflowError, ValueError):
        return None

    if not math.isfinite(logpdf_float) or not math.isfinite(score_float):
        return None
    return _SeriesResult(
        logpdf=logpdf_float,
        log_phi_score=score_float,
        term_count=_BESSEL_ASYMPTOTIC_TERMS,
        relative_error=float(relative_error),
        method="compound_poisson_bessel",
    )


def _certified_series(
    parameters: _CompoundParameters,
    *,
    observation_index: int,
    power: float,
    dispersion: float,
    requested_rtol: float,
    max_terms: int,
) -> _SeriesResult:
    try:
        mode = _find_mode_index(parameters)
        if _alpha_one_term_budget_is_provably_insufficient(
            parameters,
            mode,
            max_terms,
            requested_rtol,
        ):
            _raise_density_error(
                observation_index=observation_index,
                power=power,
                dispersion=dispersion,
                term_count=max_terms,
                requested_rtol=requested_rtol,
                reason="term limit reached before both tails were certified",
            )
        central_term = _stable_log_term(mode, parameters)
        mode_term: _LogTerm = (central_term[0], 0.0)
        lower_candidate = _lower_series_term(mode, mode_term, parameters) if mode > 1 else None
        upper_candidate = _upper_series_term(mode, mode_term, parameters)
        base_score, base_radius = _score_base_and_radius(parameters, mode)
    except (ArithmeticError, OverflowError, ValueError) as exc:
        reason = str(exc) or "non-finite series arithmetic"
        if reason not in {
            "series mode could not be bracketed safely",
            "non-finite series arithmetic",
        }:
            reason = "non-finite series arithmetic"
        _raise_density_error(
            observation_index=observation_index,
            power=power,
            dispersion=dispersion,
            term_count=0,
            requested_rtol=requested_rtol,
            reason=reason,
        )

    log_mode_term = mode_term[0]
    low = high = mode
    low_term = high_term = mode_term
    scaled_sum = np.longdouble(1.0)
    scaled_sum_compensation = np.longdouble(0.0)
    scaled_sum_uncertainty = np.longdouble(0.0)
    positive_centered_sum = np.longdouble(0.0)
    positive_centered_compensation = np.longdouble(0.0)
    negative_centered_sum = np.longdouble(0.0)
    negative_centered_compensation = np.longdouble(0.0)
    centered_sum_uncertainty = np.longdouble(0.0)
    term_count = 1
    relative_error = np.longdouble(np.inf)
    arithmetic_only_steps = 0
    score_multiplier = np.longdouble(1.0) / parameters.one_minus_power_offset

    while True:
        tail_mass, tail_center = _geometric_tail_bounds(
            mode=mode,
            low=low,
            high=high,
            log_mode_term=log_mode_term,
            low_term=low_term,
            high_term=high_term,
            lower_candidate=lower_candidate,
            upper_candidate=upper_candidate,
        )
        centered_sum = positive_centered_sum - negative_centered_sum
        included_moment = np.longdouble(mode) * scaled_sum + centered_sum
        with np.errstate(all="ignore"):
            summation_uncertainty = (
                np.longdouble(4.0) * np.longdouble(_LONGDOUBLE_EPS)
                + np.longdouble(2.0)
                * np.longdouble(term_count)
                * np.longdouble(_LONGDOUBLE_EPS) ** 2
            ) * scaled_sum
            total_sum_uncertainty = scaled_sum_uncertainty + summation_uncertainty
            scaled_sum_lower = scaled_sum - total_sum_uncertainty
            mass_tail_relative = tail_mass / scaled_sum
            mass_relative = (tail_mass + total_sum_uncertainty) / scaled_sum_lower
            moment_tail = np.longdouble(mode) * tail_mass + tail_center
            moment_uncertainty = (
                np.longdouble(mode) * total_sum_uncertainty + centered_sum_uncertainty
            )
            moment_tail_relative = moment_tail / included_moment
            moment_relative = (moment_tail + moment_uncertainty) / (
                included_moment - moment_uncertainty
            )
            centered_mean = centered_sum / scaled_sum
            score = base_score - score_multiplier * centered_mean
            score_tail_radius = (
                score_multiplier * (tail_center + abs(centered_mean) * tail_mass) / scaled_sum_lower
            )
            centered_mean_radius = (
                centered_sum_uncertainty + abs(centered_mean) * total_sum_uncertainty
            ) / scaled_sum_lower
            score_scale = max(np.longdouble(1.0), abs(score))
            score_relative = score_tail_radius / score_scale
            accumulator_radius = (
                np.longdouble(8.0)
                * np.longdouble(_LONGDOUBLE_EPS)
                * score_multiplier
                * (positive_centered_sum + negative_centered_sum + abs(centered_mean) * scaled_sum)
                / scaled_sum
            )
            output_radius = np.longdouble(0.5 * math.ulp(float(score)))
            rounding_relative = (
                accumulator_radius
                + score_multiplier * centered_mean_radius
                + base_radius
                + output_radius
            ) / score_scale
            relative_error = max(
                mass_relative,
                moment_relative,
                score_relative + rounding_relative,
            )

        truncation_certified = (
            mass_tail_relative <= requested_rtol
            and moment_tail_relative <= requested_rtol
            and score_relative <= requested_rtol
        )
        if truncation_certified and relative_error > requested_rtol:
            arithmetic_only_steps += 1
        else:
            arithmetic_only_steps = 0
        if arithmetic_only_steps >= 4096:
            _raise_density_error(
                observation_index=observation_index,
                power=power,
                dispersion=dispersion,
                term_count=term_count,
                requested_rtol=requested_rtol,
                reason="arithmetic precision was insufficient for certification",
            )

        if bool(np.isfinite(relative_error)) and relative_error <= requested_rtol:
            with np.errstate(all="ignore"):
                logpdf = log_mode_term + np.log(scaled_sum)
            if not bool(np.isfinite(logpdf)) or not bool(np.isfinite(score)):
                _raise_density_error(
                    observation_index=observation_index,
                    power=power,
                    dispersion=dispersion,
                    term_count=term_count,
                    requested_rtol=requested_rtol,
                    reason="non-finite series arithmetic",
                )
            return _SeriesResult(
                logpdf=float(logpdf),
                log_phi_score=float(score),
                term_count=term_count,
                relative_error=float(relative_error),
            )

        if term_count >= max_terms:
            _raise_density_error(
                observation_index=observation_index,
                power=power,
                dispersion=dispersion,
                term_count=term_count,
                requested_rtol=requested_rtol,
                reason="term limit reached before both tails were certified",
            )

        take_lower = lower_candidate is not None and (lower_candidate[0] >= upper_candidate[0])
        candidate = lower_candidate if take_lower else upper_candidate
        if candidate is None:
            candidate = upper_candidate
            take_lower = False
        with np.errstate(all="ignore"):
            scaled_weight = np.exp(candidate[0] - log_mode_term)
            scaled_weight_uncertainty = scaled_weight * np.expm1(np.longdouble(candidate[1]))
        if not bool(np.isfinite(scaled_weight)) or not bool(np.isfinite(scaled_weight_uncertainty)):
            _raise_density_error(
                observation_index=observation_index,
                power=power,
                dispersion=dispersion,
                term_count=term_count,
                requested_rtol=requested_rtol,
                reason="non-finite series arithmetic",
            )

        if take_lower:
            low -= 1
            low_term = candidate
            scaled_sum_uncertainty += scaled_weight_uncertainty
            centered_sum_uncertainty += np.longdouble(mode - low) * scaled_weight_uncertainty
            scaled_sum, scaled_sum_compensation = _compensated_add(
                scaled_sum,
                scaled_sum_compensation,
                scaled_weight,
            )
            negative_centered_sum, negative_centered_compensation = _compensated_add(
                negative_centered_sum,
                negative_centered_compensation,
                np.longdouble(mode - low) * scaled_weight,
            )
            try:
                if low <= 1:
                    lower_candidate = None
                elif (
                    parameters.alpha < np.longdouble(0.5)
                    and (mode - (low - 1)) % _DIRECT_REANCHOR_INTERVAL == 0
                ):
                    direct_term = _stable_log_term(low - 1, parameters)
                    lower_candidate = (
                        direct_term[0],
                        direct_term[1] + central_term[1],
                    )
                else:
                    lower_candidate = _lower_series_term(low, low_term, parameters)
            except (ArithmeticError, OverflowError, ValueError):
                _raise_density_error(
                    observation_index=observation_index,
                    power=power,
                    dispersion=dispersion,
                    term_count=term_count,
                    requested_rtol=requested_rtol,
                    reason="non-finite series arithmetic",
                )
        else:
            high += 1
            high_term = candidate
            scaled_sum_uncertainty += scaled_weight_uncertainty
            centered_sum_uncertainty += np.longdouble(high - mode) * scaled_weight_uncertainty
            scaled_sum, scaled_sum_compensation = _compensated_add(
                scaled_sum,
                scaled_sum_compensation,
                scaled_weight,
            )
            positive_centered_sum, positive_centered_compensation = _compensated_add(
                positive_centered_sum,
                positive_centered_compensation,
                np.longdouble(high - mode) * scaled_weight,
            )
            try:
                if (
                    parameters.alpha < np.longdouble(0.5)
                    and (high + 1 - mode) % _DIRECT_REANCHOR_INTERVAL == 0
                ):
                    direct_term = _stable_log_term(high + 1, parameters)
                    upper_candidate = (
                        direct_term[0],
                        direct_term[1] + central_term[1],
                    )
                else:
                    upper_candidate = _upper_series_term(high, high_term, parameters)
            except (ArithmeticError, OverflowError, ValueError):
                _raise_density_error(
                    observation_index=observation_index,
                    power=power,
                    dispersion=dispersion,
                    term_count=term_count,
                    requested_rtol=requested_rtol,
                    reason="non-finite series arithmetic",
                )
        term_count += 1


def evaluate_tweedie_density(
    y: object,
    mu: object,
    phi: object,
    p: object,
    *,
    weights: object | None = None,
    rtol: object = _DEFAULT_RTOL,
    max_terms: object = _DEFAULT_MAX_TERMS,
) -> TweedieDensityEvaluation:
    """Evaluate the exact compound-Poisson/Gamma density for ``1 < p < 2``.

    ``rtol`` bounds omitted series mass, the first moment used by the dispersion
    score, and score arithmetic.  At ``p == 1.5``, sufficiently large series
    modes use the exact modified-Bessel resummation with explicit asymptotic
    remainder intervals for both density and score.  The returned log-density
    is rounded on its natural log scale to float64; ``rtol`` is not a
    relative-error promise for exponentiating that rounded value.
    """
    y_arr, mu_arr, weight_arr = _validate_density_arrays(y, mu, weights)
    dispersion = normalize_positive_scalar("phi", phi)
    power = normalize_tweedie_power(p)
    tolerance = _normalize_rtol(rtol)
    term_limit = _normalize_max_terms(max_terms)

    logpdf = np.empty(y_arr.shape, dtype=np.float64)
    score = np.empty(y_arr.shape, dtype=np.float64)
    max_terms_used = 0
    max_relative_error = 0.0
    n_positive = 0
    exact_methods: set[str] = set()

    for index, (y_value, mu_value, weight_value) in enumerate(
        zip(y_arr, mu_arr, weight_arr, strict=True)
    ):
        if y_value == 0.0:
            with np.errstate(all="ignore"):
                effective_phi = np.longdouble(dispersion) / np.longdouble(weight_value)
                rate = np.power(
                    np.longdouble(mu_value),
                    np.longdouble(2.0 - power),
                ) / (effective_phi * np.longdouble(2.0 - power))
            if not bool(np.isfinite(rate)) or rate < 0.0 or rate > np.longdouble(_FLOAT_MAX):
                _raise_density_error(
                    observation_index=index,
                    power=power,
                    dispersion=dispersion,
                    term_count=0,
                    requested_rtol=tolerance,
                    reason="compound parameters were not representable",
                )
            logpdf[index] = -float(rate)
            score[index] = float(rate)
            continue

        n_positive += 1
        parameters = _compound_parameters(
            float(y_value),
            float(mu_value),
            dispersion,
            power,
            float(weight_value),
            observation_index=index,
            requested_rtol=tolerance,
        )
        result = _certified_alpha_one_bessel(
            parameters,
            requested_rtol=tolerance,
            max_terms=term_limit,
        )
        if result is None:
            result = _certified_series(
                parameters,
                observation_index=index,
                power=power,
                dispersion=dispersion,
                requested_rtol=tolerance,
                max_terms=term_limit,
            )
        logpdf[index] = result.logpdf
        score[index] = result.log_phi_score
        max_terms_used = max(max_terms_used, result.term_count)
        max_relative_error = max(max_relative_error, result.relative_error)
        exact_methods.add(result.method)

    if exact_methods == {"compound_poisson_bessel"}:
        method = "compound_poisson_bessel"
    elif len(exact_methods) > 1:
        method = "hybrid_compound_poisson_exact"
    else:
        method = "compound_poisson_series"

    return TweedieDensityEvaluation(
        logpdf=_readonly(logpdf),
        log_phi_score=_readonly(score),
        diagnostics=TweedieDensityDiagnostics(
            n_positive=n_positive,
            n_exact=len(y_arr),
            n_approximate=0,
            max_terms=max_terms_used,
            exact=True,
            certified=True,
            requested_rtol=tolerance,
            max_relative_tail_error=max_relative_error,
            method=method,
        ),
    )


def approximate_tweedie_logpdf(
    y: object,
    mu: object,
    phi: object,
    p: object,
    *,
    weights: object | None = None,
) -> TweedieDensityEvaluation:
    """Return a permanently labelled saddlepoint approximation."""
    y_arr, mu_arr, weight_arr = _validate_density_arrays(y, mu, weights)
    dispersion = normalize_positive_scalar("phi", phi)
    power = normalize_tweedie_power(p)
    positive = y_arr > 0.0
    logpdf = np.empty(y_arr.shape, dtype=np.float64)
    score = np.full(y_arr.shape, np.nan, dtype=np.float64)

    deviance = np.empty(0, dtype=np.float64)
    if np.any(positive):
        try:
            deviance = tweedie_unit_deviance(
                y_arr[positive],
                mu_arr[positive],
                power,
            )
        except (TweedieNumericalError, ArithmeticError, OverflowError, ValueError) as exc:
            raise TweedieDensityError(
                observation_index=int(np.flatnonzero(positive)[0]),
                power=power,
                dispersion=dispersion,
                term_count=0,
                requested_rtol=0.0,
                reason="saddlepoint arithmetic was not representable",
            ) from exc

    positive_offset = 0
    for index, (y_value, mu_value, weight_value) in enumerate(
        zip(y_arr, mu_arr, weight_arr, strict=True)
    ):
        with np.errstate(all="ignore"):
            log_effective_phi = np.log(np.longdouble(dispersion)) - np.log(
                np.longdouble(weight_value)
            )
            log_rate = (
                np.longdouble(2.0 - power) * np.log(np.longdouble(mu_value))
                - log_effective_phi
                - np.log(np.longdouble(2.0 - power))
            )
        if not bool(np.isfinite(log_effective_phi)) or not bool(np.isfinite(log_rate)):
            _raise_density_error(
                observation_index=index,
                power=power,
                dispersion=dispersion,
                term_count=0,
                requested_rtol=0.0,
                reason="saddlepoint arithmetic was not representable",
            )

        if not positive[index]:
            with np.errstate(all="ignore"):
                rate = np.exp(log_rate)
            if not bool(np.isfinite(rate)) or rate > np.longdouble(_FLOAT_MAX):
                _raise_density_error(
                    observation_index=index,
                    power=power,
                    dispersion=dispersion,
                    term_count=0,
                    requested_rtol=0.0,
                    reason="saddlepoint arithmetic was not representable",
                )
            logpdf[index] = -float(rate)
            score[index] = float(rate)
            continue

        with np.errstate(all="ignore"):
            inverse_effective_phi = np.exp(-log_effective_phi)
            value = -np.longdouble(0.5) * (
                np.longdouble(_LOG_TWO_PI)
                + log_effective_phi
                + np.longdouble(power) * np.log(np.longdouble(y_value))
                + np.longdouble(deviance[positive_offset]) * inverse_effective_phi
            )
        positive_offset += 1
        if not bool(np.isfinite(value)) or abs(value) > np.longdouble(_FLOAT_MAX):
            _raise_density_error(
                observation_index=index,
                power=power,
                dispersion=dispersion,
                term_count=0,
                requested_rtol=0.0,
                reason="saddlepoint arithmetic was not representable",
            )
        logpdf[index] = float(value)

    n_positive = int(np.count_nonzero(positive))
    n_zero = len(y_arr) - n_positive
    return TweedieDensityEvaluation(
        logpdf=_readonly(logpdf),
        log_phi_score=_readonly(score),
        diagnostics=TweedieDensityDiagnostics(
            n_positive=n_positive,
            n_exact=n_zero,
            n_approximate=n_positive,
            max_terms=0,
            exact=n_positive == 0,
            certified=n_positive == 0,
            requested_rtol=0.0,
            max_relative_tail_error=0.0 if n_positive == 0 else math.inf,
            method="saddlepoint",
        ),
    )
