"""Serial strict-IEEE Numba core for the internal Tweedie LSS point kernel."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange  # type: ignore[import-untyped]
from numpy.typing import NDArray

import superglm.distributional.kernels._fourth_generated as _generated

KERNEL_OK = 0
KERNEL_MODE_RATIO = 1
KERNEL_MODE_RANGE = 2
KERNEL_MODE_BRACKET = 3
KERNEL_MAX_TERMS = 4
KERNEL_WINDOW_RANGE = 5
KERNEL_UPPER_RATIO = 6
KERNEL_LOWER_RATIO = 7
KERNEL_PEAK = 8
KERNEL_MASS = 9
KERNEL_SCORE_MOMENTS = 10
KERNEL_HESSIAN_MOMENTS = 11
KERNEL_MEAN_SCORE_SCALE = 12
KERNEL_MEAN_SCORE = 13
KERNEL_MEAN_HESSIAN_SCALE = 14
KERNEL_MEAN_HESSIAN = 15
KERNEL_ZERO_RATE = 16
KERNEL_CANONICAL_SCALE = 17
KERNEL_SERIES_BASE = 18
KERNEL_ROW_VALUE = 19
KERNEL_ROW_SCORE = 20
KERNEL_ROW_DERIVATIVES = 21
KERNEL_COMPLETE_VALUE = 22
KERNEL_COMPLETE_SCORE = 23
KERNEL_COMPLETE_HESSIAN = 24
KERNEL_REQUIRED_WORK = 25

_MAX_SAFE_MODE = 2**52
_BERNOULLI = (
    1.0,
    -0.5,
    1.0 / 6.0,
    0.0,
    -1.0 / 30.0,
    0.0,
    1.0 / 42.0,
    0.0,
    -1.0 / 30.0,
    0.0,
    5.0 / 66.0,
    0.0,
)


# Keep these two small special-function dependencies in this module.  Numba's
# on-disk cache does not invalidate a cached caller when an imported compiled
# callee changes in another file.
@njit(cache=True)
def _digamma_positive(value: float) -> float:
    """Return digamma(value) for a finite positive scalar."""
    if not math.isfinite(value) or value <= 0.0:
        return math.nan
    result = 0.0
    x = value
    while x < 12.0:
        result -= 1.0 / x
        x += 1.0
    inverse = 1.0 / x
    inverse_squared = inverse * inverse
    correction = inverse_squared * (
        1.0 / 12.0
        - inverse_squared
        * (
            1.0 / 120.0
            - inverse_squared
            * (
                1.0 / 252.0
                - inverse_squared
                * (
                    1.0 / 240.0
                    - inverse_squared * (5.0 / 660.0 - inverse_squared * (691.0 / 32760.0))
                )
            )
        )
    )
    return result + math.log(x) - 0.5 * inverse - correction


@njit(cache=True)
def _digamma_trigamma_positive(value: float) -> tuple[float, float]:
    """Return digamma(value) and trigamma(value) with one recurrence."""
    if not math.isfinite(value) or value <= 0.0:
        return math.nan, math.nan
    digamma_result = 0.0
    trigamma_result = 0.0
    x = value
    while x < 12.0:
        inverse = 1.0 / x
        digamma_result -= inverse
        trigamma_result += 1.0 / (x * x)
        x += 1.0

    inverse = 1.0 / x
    inverse_squared = inverse * inverse
    digamma_correction = inverse_squared * (
        1.0 / 12.0
        - inverse_squared
        * (
            1.0 / 120.0
            - inverse_squared
            * (
                1.0 / 252.0
                - inverse_squared
                * (
                    1.0 / 240.0
                    - inverse_squared * (5.0 / 660.0 - inverse_squared * (691.0 / 32760.0))
                )
            )
        )
    )
    digamma = digamma_result + math.log(x) - 0.5 * inverse - digamma_correction

    trigamma_tail = inverse + 0.5 * inverse_squared
    trigamma_tail += (
        inverse
        * inverse_squared
        * (
            1.0 / 6.0
            - inverse_squared
            * (
                1.0 / 30.0
                - inverse_squared
                * (
                    1.0 / 42.0
                    - inverse_squared
                    * (
                        1.0 / 30.0
                        - inverse_squared * (5.0 / 66.0 - inverse_squared * (691.0 / 2730.0))
                    )
                )
            )
        )
    )
    return digamma, trigamma_result + trigamma_tail


@njit(cache=True)
def _compensated_add(total: float, correction: float, value: float) -> tuple[float, float]:
    updated = total + value
    if abs(total) >= abs(value):
        correction += (total - updated) + value
    else:
        correction += (value - updated) + total
    return updated, correction


@njit(cache=True)
def _sum2(first: float, second: float) -> float:
    total, correction = _compensated_add(0.0, 0.0, first)
    total, correction = _compensated_add(total, correction, second)
    return total + correction


@njit(cache=True)
def _sum3(first: float, second: float, third: float) -> float:
    total, correction = _compensated_add(0.0, 0.0, first)
    total, correction = _compensated_add(total, correction, second)
    total, correction = _compensated_add(total, correction, third)
    return total + correction


@njit(cache=True)
def _bernoulli_polynomial(order: int, value: float) -> float:
    total = 0.0
    correction = 0.0
    binomial = 1.0
    for index in range(order + 1):
        term = binomial * _BERNOULLI[index] * value ** (order - index)
        total, correction = _compensated_add(total, correction, term)
        if index < order:
            binomial *= float(order - index) / float(index + 1)
    return total + correction


@njit(cache=True)
def _fill_log_gamma_increment_coefficients(alpha: float, coefficients: NDArray[np.float64]) -> None:
    for order in range(1, 11):
        sign = 1.0 if order % 2 == 1 else -1.0
        coefficients[order - 1] = (
            sign
            * (_bernoulli_polynomial(order + 1, alpha) - _BERNOULLI[order + 1])
            / float(order * (order + 1))
        )


@njit(cache=True)
def _log_gamma_increment(
    x: float,
    alpha: float,
    coefficients: NDArray[np.float64],
) -> float:
    integer_alpha = int(alpha)
    if alpha == float(integer_alpha) and 1 <= integer_alpha <= 32:
        total = 0.0
        correction = 0.0
        for offset in range(integer_alpha):
            total, correction = _compensated_add(
                total,
                correction,
                math.log(x + float(offset)),
            )
        return total + correction
    if x < 4096.0:
        return math.lgamma(x + alpha) - math.lgamma(x)
    if math.isnan(coefficients[0]):
        _fill_log_gamma_increment_coefficients(alpha, coefficients)
    inverse = 1.0 / x
    power_value = inverse
    correction = 0.0
    for index in range(10):
        correction += coefficients[index] * power_value
        power_value *= inverse
    return alpha * math.log(x) + correction


@njit(cache=True)
def _log_adjacent_ratio(
    j: int,
    zeta: float,
    alpha: float,
    coefficients: NDArray[np.float64],
) -> float:
    increment = _log_gamma_increment(alpha * float(j), alpha, coefficients)
    return zeta - math.log(float(j + 1)) - increment


@njit(cache=True)
def _locate_series_mode(
    zeta: float,
    alpha: float,
    coefficients: NDArray[np.float64],
) -> tuple[int, int, float, float]:
    first = _log_adjacent_ratio(1, zeta, alpha, coefficients)
    if not math.isfinite(first):
        return KERNEL_MODE_RATIO, 0, math.nan, math.nan
    if first <= 0.0:
        return KERNEL_OK, 1, math.nan, first

    lower = 1
    upper = 2
    while True:
        ratio = _log_adjacent_ratio(upper, zeta, alpha, coefficients)
        if not math.isfinite(ratio):
            return KERNEL_MODE_RATIO, 0, math.nan, math.nan
        if ratio <= 0.0:
            break
        if upper >= _MAX_SAFE_MODE:
            return KERNEL_MODE_RANGE, 0, math.nan, math.nan
        lower = upper
        upper = min(2 * upper, _MAX_SAFE_MODE)

    while upper - lower > 1:
        midpoint = lower + (upper - lower) // 2
        ratio = _log_adjacent_ratio(midpoint, zeta, alpha, coefficients)
        if not math.isfinite(ratio):
            return KERNEL_MODE_RATIO, 0, math.nan, math.nan
        if ratio > 0.0:
            lower = midpoint
        else:
            upper = midpoint

    before = _log_adjacent_ratio(upper - 1, zeta, alpha, coefficients)
    after = _log_adjacent_ratio(upper, zeta, alpha, coefficients)
    if before <= 0.0 or after > 0.0:
        return KERNEL_MODE_BRACKET, 0, math.nan, math.nan
    return KERNEL_OK, upper, before, after


@njit(cache=True)
def _term_derivative_channels(
    j: int,
    zeta_p: float,
    zeta_pp: float,
    inverse_r: float,
    derivative_order: int,
) -> tuple[float, float, float, float]:
    j_float = float(j)
    q_rho = -j_float * inverse_r
    aj = (inverse_r - 1.0) * j_float
    inverse_r2 = inverse_r * inverse_r
    if derivative_order == 1:
        digamma = _digamma_positive(aj)
        q_p = j_float * (zeta_p + digamma * inverse_r2)
        return q_rho, q_p, math.nan, math.nan
    digamma, trigamma = _digamma_trigamma_positive(aj)
    q_p = j_float * (zeta_p + digamma * inverse_r2)
    inverse_r3 = inverse_r2 * inverse_r
    q_rho_p = j_float * inverse_r2
    q_pp = (
        j_float * zeta_pp
        - j_float * j_float * trigamma * inverse_r2 * inverse_r2
        - 2.0 * j_float * digamma * inverse_r3
    )
    return q_rho, q_p, q_rho_p, q_pp


# Series tuples are (status, log_sum, mean_q_rho, mean_q_p, mean_q_rho_p,
# mean_q_pp, variance_q_rho, covariance_q_rho_p, variance_q_p, terms).
@njit(cache=True)
def _series_failure(status: int, terms: int):
    return (
        status,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        terms,
    )


@njit(cache=True)
def _series_summary(
    zeta: float,
    zeta_p: float,
    zeta_pp: float,
    inverse_r: float,
    alpha: float,
    derivative_order: int,
    max_terms: int,
    log_cutoff: float,
    coefficients: NDArray[np.float64],
):
    coefficients[0] = math.nan  # Lazy scratch, reset for this row's alpha.
    status, mode, mode_lower_ratio, mode_upper_ratio = _locate_series_mode(
        zeta,
        alpha,
        coefficients,
    )
    if status != KERNEL_OK:
        return _series_failure(status, 0)
    terms = 1

    mode_float = float(mode)
    peak = mode_float * zeta - math.lgamma(mode_float + 1.0) - math.lgamma(alpha * mode_float)
    if not math.isfinite(peak):
        return _series_failure(KERNEL_PEAK, terms)

    mass = 1.0
    anchor_rho = math.nan
    anchor_p = math.nan
    anchor_rho_p = math.nan
    anchor_pp = math.nan
    mean_delta_rho = math.nan
    mean_delta_p = math.nan
    mean_delta_rho_p = math.nan
    mean_delta_pp = math.nan
    variance_rho_mass = math.nan
    covariance_mass = math.nan
    variance_p_mass = math.nan
    if derivative_order >= 1:
        anchor_rho, anchor_p, anchor_rho_p, anchor_pp = _term_derivative_channels(
            mode,
            zeta_p,
            zeta_pp,
            inverse_r,
            derivative_order,
        )
        mean_delta_rho = 0.0
        mean_delta_p = 0.0
        mean_delta_rho_p = 0.0
        mean_delta_pp = 0.0
        variance_rho_mass = 0.0
        covariance_mass = 0.0
        variance_p_mass = 0.0

    relative_log = 0.0
    current = mode
    while current > 1:
        if terms >= max_terms:
            return _series_failure(KERNEL_MAX_TERMS, 0)
        if current == mode:
            ratio_value = mode_lower_ratio
        else:
            ratio_value = _log_adjacent_ratio(current - 1, zeta, alpha, coefficients)
        if not math.isfinite(ratio_value) or ratio_value <= 0.0:
            return _series_failure(KERNEL_LOWER_RATIO, 0)
        relative_log -= ratio_value
        current -= 1
        terms += 1
        relative = math.exp(relative_log)
        new_mass = mass + relative
        if derivative_order >= 1:
            q_rho, q_p, q_rho_p, q_pp = _term_derivative_channels(
                current,
                zeta_p,
                zeta_pp,
                inverse_r,
                derivative_order,
            )
            ratio = relative / new_mass
            centered_rho = q_rho - anchor_rho
            centered_p = q_p - anchor_p
            delta_rho = centered_rho - mean_delta_rho
            delta_p = centered_p - mean_delta_p
            mean_delta_rho += ratio * delta_rho
            mean_delta_p += ratio * delta_p
            if derivative_order == 2:
                variance_rho_mass += relative * delta_rho * (centered_rho - mean_delta_rho)
                covariance_mass += relative * delta_rho * (centered_p - mean_delta_p)
                variance_p_mass += relative * delta_p * (centered_p - mean_delta_p)
                mean_delta_rho_p += ratio * ((q_rho_p - anchor_rho_p) - mean_delta_rho_p)
                mean_delta_pp += ratio * ((q_pp - anchor_pp) - mean_delta_pp)
        mass = new_mass
        if relative_log <= -log_cutoff:
            break

    relative_log = 0.0
    current = mode
    while True:
        if terms >= max_terms:
            return _series_failure(KERNEL_MAX_TERMS, 0)
        if current >= _MAX_SAFE_MODE:
            return _series_failure(KERNEL_WINDOW_RANGE, 0)
        if current == mode:
            ratio_value = mode_upper_ratio
        else:
            ratio_value = _log_adjacent_ratio(current, zeta, alpha, coefficients)
        if not math.isfinite(ratio_value) or ratio_value > 0.0:
            return _series_failure(KERNEL_UPPER_RATIO, 0)
        relative_log += ratio_value
        current += 1
        terms += 1
        relative = math.exp(relative_log)
        new_mass = mass + relative
        if derivative_order >= 1:
            q_rho, q_p, q_rho_p, q_pp = _term_derivative_channels(
                current,
                zeta_p,
                zeta_pp,
                inverse_r,
                derivative_order,
            )
            ratio = relative / new_mass
            centered_rho = q_rho - anchor_rho
            centered_p = q_p - anchor_p
            delta_rho = centered_rho - mean_delta_rho
            delta_p = centered_p - mean_delta_p
            mean_delta_rho += ratio * delta_rho
            mean_delta_p += ratio * delta_p
            if derivative_order == 2:
                variance_rho_mass += relative * delta_rho * (centered_rho - mean_delta_rho)
                covariance_mass += relative * delta_rho * (centered_p - mean_delta_p)
                variance_p_mass += relative * delta_p * (centered_p - mean_delta_p)
                mean_delta_rho_p += ratio * ((q_rho_p - anchor_rho_p) - mean_delta_rho_p)
                mean_delta_pp += ratio * ((q_pp - anchor_pp) - mean_delta_pp)
        mass = new_mass
        if relative_log <= -log_cutoff:
            break

    if not math.isfinite(mass) or mass <= 0.0:
        return _series_failure(KERNEL_MASS, terms)
    log_sum = peak + math.log(mass)
    if derivative_order == 0:
        return (
            KERNEL_OK,
            log_sum,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            terms,
        )

    mean_rho = _sum2(anchor_rho, mean_delta_rho)
    mean_p = _sum2(anchor_p, mean_delta_p)
    if not math.isfinite(mean_rho) or not math.isfinite(mean_p):
        return _series_failure(KERNEL_SCORE_MOMENTS, terms)
    if derivative_order == 1:
        return (
            KERNEL_OK,
            log_sum,
            mean_rho,
            mean_p,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            terms,
        )

    variance_rho = variance_rho_mass / mass
    covariance = covariance_mass / mass
    variance_p = variance_p_mass / mass
    mean_rho_p = _sum2(anchor_rho_p, mean_delta_rho_p)
    mean_pp = _sum2(anchor_pp, mean_delta_pp)
    if not (
        math.isfinite(mean_rho_p)
        and math.isfinite(mean_pp)
        and math.isfinite(variance_rho)
        and math.isfinite(covariance)
        and math.isfinite(variance_p)
    ):
        return _series_failure(KERNEL_HESSIAN_MOMENTS, terms)
    return (
        KERNEL_OK,
        log_sum,
        mean_rho,
        mean_p,
        mean_rho_p,
        mean_pp,
        variance_rho,
        covariance,
        variance_p,
        terms,
    )


@njit(cache=True)
def _mean_score_channel(y: float, mean: float, rho: float, power: float) -> tuple[int, float]:
    exponent = -rho - power * math.log(mean)
    scale = math.exp(exponent)
    if not math.isfinite(scale):
        if math.isfinite(exponent):
            return KERNEL_REQUIRED_WORK, math.nan
        return KERNEL_MEAN_SCORE_SCALE, math.nan
    if scale == 0.0:
        return KERNEL_MEAN_SCORE_SCALE, math.nan
    score = (y - mean) * scale
    if not math.isfinite(score):
        return KERNEL_MEAN_SCORE, math.nan
    return KERNEL_OK, score


@njit(cache=True)
def _mean_hessian_channel(
    y: float,
    mean: float,
    rho: float,
    power: float,
) -> tuple[int, float]:
    exponent = -rho - (power + 1.0) * math.log(mean)
    scale = math.exp(exponent)
    if not math.isfinite(scale):
        if math.isfinite(exponent):
            return KERNEL_REQUIRED_WORK, math.nan
        return KERNEL_MEAN_HESSIAN_SCALE, math.nan
    if scale == 0.0:
        return KERNEL_MEAN_HESSIAN_SCALE, math.nan
    numerator = _sum2((power - 1.0) * mean, -power * y)
    hessian = numerator * scale
    if not math.isfinite(hessian):
        return KERNEL_MEAN_HESSIAN, math.nan
    return KERNEL_OK, hessian


# Row tuples are status, value, score (mu, phi, p), packed Hessian
# (mu-mu, mu-phi, mu-p, phi-phi, phi-p, p-p), then terms.
@njit(cache=True)
def _row_failure(status: int):
    return (
        status,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        0,
    )


@njit(cache=True)
def _natural_scale_hessian(hessian_log_scale: float, score_log_scale: float, scale: float):
    """Apply (H_log-scale - s_log-scale)/scale² without losing the square."""
    difference = hessian_log_scale - score_log_scale
    scale_squared = scale * scale
    if (
        math.isfinite(scale_squared)
        and scale_squared >= np.finfo(np.float64).tiny
        and math.isfinite(difference)
    ):
        return difference / scale_squared
    if math.isfinite(difference):
        return (difference / scale) / scale
    return (hessian_log_scale / scale) / scale - (score_log_scale / scale) / scale


@njit(cache=True)
def _zero_row(
    mean: float,
    modeled_phi: float,
    power: float,
    rho: float,
    derivative_order: int,
):
    log_mean = math.log(mean)
    s = 2.0 - power
    h = s * log_mean - rho - math.log(s)
    lam = math.exp(h)
    if not math.isfinite(lam):
        if math.isfinite(h):
            return _row_failure(KERNEL_REQUIRED_WORK)
        return _row_failure(KERNEL_ZERO_RATE)
    if lam == 0.0:
        return _row_failure(KERNEL_ZERO_RATE)
    value = -lam
    if derivative_order == 0:
        return (
            KERNEL_OK,
            value,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0,
        )

    status, mean_score = _mean_score_channel(0.0, mean, rho, power)
    if status != KERNEL_OK:
        return _row_failure(status)
    score_rho = lam
    score_p = lam * (log_mean - 1.0 / s)
    score_phi = score_rho / modeled_phi
    if derivative_order == 1:
        return (
            KERNEL_OK,
            value,
            mean_score,
            score_phi,
            score_p,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            0,
        )

    status, mean_hessian = _mean_hessian_channel(0.0, mean, rho, power)
    if status != KERNEL_OK:
        return _row_failure(status)
    hessian_rho_rho = -lam
    hessian_rho_p = lam * (1.0 / s - log_mean)
    hessian_pp = -lam * (1.0 / (s * s) + (-log_mean + 1.0 / s) ** 2)
    hessian_mu_phi = -mean_score / modeled_phi
    hessian_mu_p = -log_mean * mean_score
    hessian_phi_phi = _natural_scale_hessian(hessian_rho_rho, score_rho, modeled_phi)
    if modeled_phi * modeled_phi == 0.0 and not math.isfinite(hessian_phi_phi):
        return _row_failure(KERNEL_REQUIRED_WORK)
    hessian_phi_p = hessian_rho_p / modeled_phi
    return (
        KERNEL_OK,
        value,
        mean_score,
        score_phi,
        score_p,
        mean_hessian,
        hessian_mu_phi,
        hessian_mu_p,
        hessian_phi_phi,
        hessian_phi_p,
        hessian_pp,
        0,
    )


@njit(cache=True)
def _positive_row(
    y: float,
    mean: float,
    modeled_phi: float,
    power: float,
    rho: float,
    derivative_order: int,
    max_terms: int,
    log_cutoff: float,
    coefficients: NDArray[np.float64],
):
    log_y = math.log(y)
    log_mean = math.log(mean)
    r = power - 1.0
    s = 2.0 - power
    inverse_r = 1.0 / r
    inverse_s = 1.0 / s
    alpha = s * inverse_r

    a_exponent = -r * log_mean
    c_exponent = s * log_mean
    f_exponent = -rho
    a_scale = math.exp(a_exponent)
    a_term = y * a_scale
    c_term = math.exp(c_exponent)
    f_term = math.exp(f_exponent)
    if not (
        math.isfinite(a_term)
        and a_term > 0.0
        and math.isfinite(c_term)
        and c_term > 0.0
        and math.isfinite(f_term)
        and f_term > 0.0
    ):
        if (
            (math.isfinite(a_exponent) and math.isinf(a_scale))
            or (math.isfinite(c_exponent) and math.isinf(c_term))
            or (math.isfinite(f_exponent) and math.isinf(f_term))
        ):
            return _row_failure(KERNEL_REQUIRED_WORK)
        return _row_failure(KERNEL_CANONICAL_SCALE)
    canonical = _sum2(-a_term * inverse_r, -c_term * inverse_s)
    b_value = f_term * canonical

    zeta = alpha * (log_y - math.log(r)) - math.log(s) - rho * inverse_r
    zeta_p = math.nan
    zeta_pp = math.nan
    if derivative_order >= 1:
        d_value = math.log(r) - log_y + rho
        zeta_p = (d_value - s) * inverse_r * inverse_r + inverse_s
        if derivative_order == 2:
            zeta_pp = (
                -2.0 * d_value + 2.0 + s
            ) * inverse_r * inverse_r * inverse_r + inverse_s * inverse_s
    if not math.isfinite(b_value) or not math.isfinite(zeta):
        return _row_failure(KERNEL_SERIES_BASE)
    if derivative_order >= 1 and not math.isfinite(zeta_p):
        return _row_failure(KERNEL_SERIES_BASE)
    if derivative_order == 2 and not math.isfinite(zeta_pp):
        return _row_failure(KERNEL_SERIES_BASE)

    summary = _series_summary(
        zeta,
        zeta_p,
        zeta_pp,
        inverse_r,
        alpha,
        derivative_order,
        max_terms,
        log_cutoff,
        coefficients,
    )
    status = summary[0]
    if status != KERNEL_OK:
        return _row_failure(status)
    log_sum = summary[1]
    mean_q_rho = summary[2]
    mean_q_p = summary[3]
    mean_q_rho_p = summary[4]
    mean_q_pp = summary[5]
    variance_q_rho = summary[6]
    covariance_q_rho_p = summary[7]
    variance_q_p = summary[8]
    terms = summary[9]

    value = _sum3(b_value, -log_y, log_sum)
    if not math.isfinite(value):
        return _row_failure(KERNEL_ROW_VALUE)
    if derivative_order == 0:
        return (
            KERNEL_OK,
            value,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            terms,
        )

    inverse_r2 = inverse_r * inverse_r
    canonical_p = _sum2(
        a_term * (log_mean * inverse_r + inverse_r2),
        c_term * (log_mean * inverse_s - inverse_s * inverse_s),
    )
    b_p = f_term * canonical_p
    score_rho = _sum2(-b_value, mean_q_rho)
    score_p = _sum2(b_p, mean_q_p)
    status, mean_score = _mean_score_channel(y, mean, rho, power)
    if status != KERNEL_OK:
        return _row_failure(status)
    score_phi = score_rho / modeled_phi
    if derivative_order == 1:
        if not (math.isfinite(mean_score) and math.isfinite(score_phi) and math.isfinite(score_p)):
            return _row_failure(KERNEL_ROW_SCORE)
        return (
            KERNEL_OK,
            value,
            mean_score,
            score_phi,
            score_p,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            terms,
        )

    inverse_r3 = inverse_r2 * inverse_r
    canonical_pp = _sum2(
        a_term
        * (-log_mean * log_mean * inverse_r - 2.0 * log_mean * inverse_r2 - 2.0 * inverse_r3),
        c_term
        * (
            -log_mean * log_mean * inverse_s
            + 2.0 * log_mean * inverse_s * inverse_s
            - 2.0 * inverse_s * inverse_s * inverse_s
        ),
    )
    b_pp = f_term * canonical_pp
    status, mean_hessian = _mean_hessian_channel(y, mean, rho, power)
    if status != KERNEL_OK:
        return _row_failure(status)
    hessian_rho_rho = _sum2(b_value, variance_q_rho)
    hessian_rho_p = _sum3(-b_p, mean_q_rho_p, covariance_q_rho_p)
    hessian_pp = _sum3(b_pp, mean_q_pp, variance_q_p)
    hessian_mu_phi = -mean_score / modeled_phi
    hessian_mu_p = -log_mean * mean_score
    hessian_phi_phi = _natural_scale_hessian(hessian_rho_rho, score_rho, modeled_phi)
    if modeled_phi * modeled_phi == 0.0 and not math.isfinite(hessian_phi_phi):
        return _row_failure(KERNEL_REQUIRED_WORK)
    hessian_phi_p = hessian_rho_p / modeled_phi
    if not (
        math.isfinite(mean_score)
        and math.isfinite(score_phi)
        and math.isfinite(score_p)
        and math.isfinite(mean_hessian)
        and math.isfinite(hessian_mu_phi)
        and math.isfinite(hessian_mu_p)
        and math.isfinite(hessian_phi_phi)
        and math.isfinite(hessian_phi_p)
        and math.isfinite(hessian_pp)
    ):
        return _row_failure(KERNEL_ROW_DERIVATIVES)
    return (
        KERNEL_OK,
        value,
        mean_score,
        score_phi,
        score_p,
        mean_hessian,
        hessian_mu_phi,
        hessian_mu_p,
        hessian_phi_phi,
        hessian_phi_p,
        hessian_pp,
        terms,
    )


@njit(cache=True)
def _allocate_batch_outputs(n_rows: int, derivative_order: int):
    values = np.empty(n_rows, dtype=np.float64)
    if derivative_order >= 1:
        scores = np.empty((n_rows, 3), dtype=np.float64)
    else:
        scores = np.empty((0, 3), dtype=np.float64)
    if derivative_order == 2:
        hessians = np.empty((n_rows, 6), dtype=np.float64)
    else:
        hessians = np.empty((0, 6), dtype=np.float64)
    terms = np.empty(n_rows, dtype=np.int64)
    valid = np.ones(n_rows, dtype=np.bool_)
    return values, scores, hessians, terms, valid


@njit(cache=True)
def _evaluate_tweedie_batch_row(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    weight_mode: int,
    derivative_order: int,
    max_terms: int,
    log_cutoff: float,
    row: int,
    values: NDArray[np.float64],
    scores: NDArray[np.float64],
    hessians: NDArray[np.float64],
    terms: NDArray[np.int64],
) -> int:
    response = y[row]
    modeled_mean = mean[row]
    modeled_phi = dispersion[row]
    modeled_power = power[row]
    row_weight = weight[row]
    rho = math.log(modeled_phi)
    multiplier = 1.0
    if weight_mode == 0:
        rho -= math.log(row_weight)
    else:
        multiplier = row_weight

    if response == 0.0:
        evaluated = _zero_row(
            modeled_mean,
            modeled_phi,
            modeled_power,
            rho,
            derivative_order,
        )
    else:
        coefficients = np.empty(10, dtype=np.float64)
        evaluated = _positive_row(
            response,
            modeled_mean,
            modeled_phi,
            modeled_power,
            rho,
            derivative_order,
            max_terms,
            log_cutoff,
            coefficients,
        )
    status = evaluated[0]
    if status != KERNEL_OK:
        return status

    values[row] = multiplier * evaluated[1]
    terms[row] = evaluated[11]
    if derivative_order >= 1:
        scores[row, 0] = multiplier * evaluated[2]
        scores[row, 1] = multiplier * evaluated[3]
        scores[row, 2] = multiplier * evaluated[4]
    if derivative_order == 2:
        hessians[row, 0] = multiplier * evaluated[5]
        hessians[row, 1] = multiplier * evaluated[6]
        hessians[row, 2] = multiplier * evaluated[7]
        hessians[row, 3] = multiplier * evaluated[8]
        hessians[row, 4] = multiplier * evaluated[9]
        hessians[row, 5] = multiplier * evaluated[10]
    return KERNEL_OK


@njit(cache=True)
def _complete_batch_status(
    values: NDArray[np.float64],
    scores: NDArray[np.float64],
    hessians: NDArray[np.float64],
    derivative_order: int,
) -> int:
    if not np.all(np.isfinite(values)):
        return KERNEL_COMPLETE_VALUE
    if derivative_order >= 1 and not np.all(np.isfinite(scores)):
        return KERNEL_COMPLETE_SCORE
    if derivative_order == 2 and not np.all(np.isfinite(hessians)):
        return KERNEL_COMPLETE_HESSIAN
    return KERNEL_OK


# Batch tuples append (status, failing_row): row failures carry their index,
# while complete-output failures use -1 for Python-boundary normalization.
@njit(cache=True)
def _evaluate_tweedie_batch_core(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    weight_mode: int,
    derivative_order: int,
    max_terms: int,
    log_cutoff: float,
):
    """Evaluate one validated contiguous batch serially."""
    outputs = _allocate_batch_outputs(y.size, derivative_order)
    values, scores, hessians, terms, valid = outputs
    for row in range(y.size):
        status = _evaluate_tweedie_batch_row(
            y,
            mean,
            dispersion,
            power,
            weight,
            weight_mode,
            derivative_order,
            max_terms,
            log_cutoff,
            row,
            values,
            scores,
            hessians,
            terms,
        )
        if status != KERNEL_OK:
            return values, scores, hessians, terms, valid, status, row
    status = _complete_batch_status(values, scores, hessians, derivative_order)
    return values, scores, hessians, terms, valid, status, -1


@njit(cache=True, parallel=True)
def _evaluate_tweedie_batch_parallel_core(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    weight_mode: int,
    derivative_order: int,
    max_terms: int,
    log_cutoff: float,
):
    """Evaluate independent rows in parallel, preserving row arithmetic."""
    outputs = _allocate_batch_outputs(y.size, derivative_order)
    values, scores, hessians, terms, valid = outputs
    statuses = np.empty(y.size, dtype=np.int64)
    for row in prange(y.size):  # ty: ignore[not-iterable] -- Numba loop primitive
        statuses[row] = _evaluate_tweedie_batch_row(
            y,
            mean,
            dispersion,
            power,
            weight,
            weight_mode,
            derivative_order,
            max_terms,
            log_cutoff,
            row,
            values,
            scores,
            hessians,
            terms,
        )
    for row in range(y.size):
        status = statuses[row]
        if status != KERNEL_OK:
            return values, scores, hessians, terms, valid, status, row
    status = _complete_batch_status(values, scores, hessians, derivative_order)
    return values, scores, hessians, terms, valid, status, -1


# ---------------------------------------------------------------- third derivatives
#
# The analytic directional derivative of the natural Hessian.  The series part
# log W = log sum_j w_j has the joint cumulants of the term derivatives under
# pi_j ~ w_j as its derivatives; with X = q_rho = -j/r, Y = q_p and
# q_rho_p = j/r^2, q_rho_pp = -2j/r^3 linear in j, the third-order ones are
#   W_rrr = -mu3(j)/r^3,
#   W_rrp = -2 Var(j)/r^3 + E[dj^2 dY]/r^2,
#   W_rpp = -2 E[j]/r^3 + 2 Cov(j, Y)/r^2 - Cov(q_pp, j)/r - E[dj dY^2]/r,
#   W_ppp = E[q_ppp] + 3 Cov(q_pp, Y) + E[dY^3],
# two passes over the series window the Hessian uses: weighted means first,
# then central moments about them.


@njit(cache=True)
def _polygamma_012(value: float) -> tuple[float, float, float]:
    """digamma, trigamma and tetragamma of a finite positive scalar."""
    digamma_result = 0.0
    trigamma_result = 0.0
    tetragamma_result = 0.0
    x = value
    while x < 12.0:
        inverse = 1.0 / x
        digamma_result -= inverse
        trigamma_result += inverse * inverse
        tetragamma_result -= 2.0 * inverse * inverse * inverse
        x += 1.0
    inverse = 1.0 / x
    i2 = inverse * inverse
    digamma = (
        digamma_result
        + math.log(x)
        - 0.5 * inverse
        - i2
        * (
            1.0 / 12.0
            - i2
            * (
                1.0 / 120.0
                - i2
                * (1.0 / 252.0 - i2 * (1.0 / 240.0 - i2 * (5.0 / 660.0 - i2 * (691.0 / 32760.0))))
            )
        )
    )
    trigamma = (
        trigamma_result
        + inverse
        + 0.5 * i2
        + inverse
        * i2
        * (
            1.0 / 6.0
            - i2
            * (
                1.0 / 30.0
                - i2 * (1.0 / 42.0 - i2 * (1.0 / 30.0 - i2 * (5.0 / 66.0 - i2 * (691.0 / 2730.0))))
            )
        )
    )
    tetragamma = tetragamma_result - i2 * (
        1.0
        + inverse
        + 0.5 * i2
        - i2
        * i2
        * (
            1.0 / 6.0
            - i2
            * (
                1.0 / 6.0
                - i2 * (3.0 / 10.0 - i2 * (5.0 / 6.0 - i2 * (691.0 / 210.0 - i2 * (35.0 / 2.0))))
            )
        )
    )
    return digamma, trigamma, tetragamma


@njit(cache=True)
def _third_term_channels(
    j: int, zeta_p: float, zeta_pp: float, zeta_ppp: float, inverse_r: float
) -> tuple[float, float, float]:
    """``(q_p, q_pp, q_ppp)`` of series term ``j``."""
    j_float = float(j)
    digamma, trigamma, tetragamma = _polygamma_012((inverse_r - 1.0) * j_float)
    r2 = inverse_r * inverse_r
    r3 = r2 * inverse_r
    r4 = r2 * r2
    q_p = j_float * (zeta_p + digamma * r2)
    q_pp = j_float * zeta_pp - j_float * j_float * trigamma * r4 - 2.0 * j_float * digamma * r3
    q_ppp = (
        j_float * zeta_ppp
        + j_float * j_float * j_float * tetragamma * r4 * r2
        + 6.0 * j_float * j_float * trigamma * r4 * inverse_r
        + 6.0 * j_float * digamma * r4
    )
    return q_p, q_pp, q_ppp


@njit(cache=True)
def _series_third_cumulants(
    zeta: float,
    zeta_p: float,
    zeta_pp: float,
    zeta_ppp: float,
    inverse_r: float,
    alpha: float,
    max_terms: int,
    log_cutoff: float,
    coefficients: NDArray[np.float64],
):
    """Status, then ``(W_r, W_p, W_rr, W_rp, W_pp, W_rrr, W_rrp, W_rpp, W_ppp)``."""
    coefficients[0] = math.nan
    status, mode, mode_lower_ratio, mode_upper_ratio = _locate_series_mode(
        zeta, alpha, coefficients
    )
    nan9 = (
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )
    if status != KERNEL_OK:
        return status, nan9
    anchor_p, anchor_pp, anchor_ppp = _third_term_channels(
        mode, zeta_p, zeta_pp, zeta_ppp, inverse_r
    )
    mode_float = float(mode)
    # Pass 1: mass and anchored first moments; the window is the Hessian's.
    mass = 1.0
    sum_j = 0.0
    sum_p = 0.0
    sum_pp = 0.0
    sum_ppp = 0.0
    lowest = mode
    highest = mode
    terms = 1
    for direction in (-1, 1):
        relative_log = 0.0
        current = mode
        while True:
            if direction < 0:
                if current <= 1:
                    break
                ratio_value = (
                    mode_lower_ratio
                    if current == mode
                    else _log_adjacent_ratio(current - 1, zeta, alpha, coefficients)
                )
                if not math.isfinite(ratio_value) or ratio_value <= 0.0:
                    return KERNEL_LOWER_RATIO, nan9
                relative_log -= ratio_value
                current -= 1
            else:
                if current >= _MAX_SAFE_MODE:
                    return KERNEL_WINDOW_RANGE, nan9
                ratio_value = (
                    mode_upper_ratio
                    if current == mode
                    else _log_adjacent_ratio(current, zeta, alpha, coefficients)
                )
                if not math.isfinite(ratio_value) or ratio_value > 0.0:
                    return KERNEL_UPPER_RATIO, nan9
                relative_log += ratio_value
                current += 1
            terms += 1
            if terms > max_terms:
                return KERNEL_MAX_TERMS, nan9
            relative = math.exp(relative_log)
            q_p, q_pp, q_ppp = _third_term_channels(current, zeta_p, zeta_pp, zeta_ppp, inverse_r)
            mass += relative
            sum_j += relative * (float(current) - mode_float)
            sum_p += relative * (q_p - anchor_p)
            sum_pp += relative * (q_pp - anchor_pp)
            sum_ppp += relative * (q_ppp - anchor_ppp)
            if direction < 0:
                lowest = current
            else:
                highest = current
            if relative_log <= -log_cutoff:
                break
    offset_j = sum_j / mass
    offset_p = sum_p / mass
    mean_j = mode_float + offset_j
    mean_p = _sum2(anchor_p, offset_p)
    mean_pp = _sum2(anchor_pp, sum_pp / mass)
    mean_ppp = _sum2(anchor_ppp, sum_ppp / mass)
    # Pass 2: central moments about the pass-1 means, over the same window.
    v_jj = 0.0
    c_jy = 0.0
    v_yy = 0.0
    m_jjj = 0.0
    m_jjy = 0.0
    m_jyy = 0.0
    m_yyy = 0.0
    c_2j = 0.0
    c_2y = 0.0
    for direction in (-1, 1):
        relative_log = 0.0
        current = mode
        while True:
            if direction < 0:
                if current != mode:
                    ratio_value = (
                        mode_lower_ratio
                        if current + 1 == mode
                        else _log_adjacent_ratio(current, zeta, alpha, coefficients)
                    )
                    relative_log -= ratio_value
            else:
                if current >= highest:
                    break
                ratio_value = (
                    mode_upper_ratio
                    if current == mode
                    else _log_adjacent_ratio(current, zeta, alpha, coefficients)
                )
                relative_log += ratio_value
                current += 1
            relative = math.exp(relative_log)
            q_p, q_pp, _ = _third_term_channels(current, zeta_p, zeta_pp, zeta_ppp, inverse_r)
            dj = (float(current) - mode_float) - offset_j
            dy = (q_p - anchor_p) - offset_p
            d2 = q_pp - mean_pp
            v_jj += relative * dj * dj
            c_jy += relative * dj * dy
            v_yy += relative * dy * dy
            m_jjj += relative * dj * dj * dj
            m_jjy += relative * dj * dj * dy
            m_jyy += relative * dj * dy * dy
            m_yyy += relative * dy * dy * dy
            c_2j += relative * d2 * dj
            c_2y += relative * d2 * dy
            if direction < 0:
                if current <= lowest:
                    break
                current -= 1
    inverse_mass = 1.0 / mass
    v_jj *= inverse_mass
    c_jy *= inverse_mass
    v_yy *= inverse_mass
    m_jjj *= inverse_mass
    m_jjy *= inverse_mass
    m_jyy *= inverse_mass
    m_yyy *= inverse_mass
    c_2j *= inverse_mass
    c_2y *= inverse_mass
    r2 = inverse_r * inverse_r
    r3 = r2 * inverse_r
    w_r = -mean_j * inverse_r
    w_rr = v_jj * r2
    w_rp = _sum2(mean_j * r2, -c_jy * inverse_r)
    w_pp = _sum2(mean_pp, v_yy)
    w_rrr = -m_jjj * r3
    w_rrp = _sum2(-2.0 * v_jj * r3, m_jjy * r2)
    w_rpp = _sum3(-2.0 * mean_j * r3 + 2.0 * c_jy * r2, -c_2j * inverse_r, -m_jyy * inverse_r)
    w_ppp = _sum3(mean_ppp, 3.0 * c_2y, m_yyy)
    return KERNEL_OK, (w_r, mean_p, w_rr, w_rp, w_pp, w_rrr, w_rrp, w_rpp, w_ppp)


@njit(cache=True)
def _tweedie_third_row(
    y: float,
    mean: float,
    phi: float,
    power: float,
    rho: float,
    max_terms: int,
    log_cutoff: float,
    coefficients: NDArray[np.float64],
):
    """Status, then the ten natural third derivatives in ``(mu, phi, p)``.

    Order: ``mmm, mmf, mmp, mff, mfp, mpp, fff, ffp, fpp, ppp``.  The
    ``(rho, p)`` block converts to ``phi`` by ``d/dphi = (1/phi) d/drho``.
    """
    log_mean = math.log(mean)
    r = power - 1.0
    s = 2.0 - power
    inverse_r = 1.0 / r
    inverse_s = 1.0 / s
    nan10 = (
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )
    scale_mu = math.exp(-rho - power * log_mean)
    mean_score = (y - mean) * scale_mu
    scale_mu2 = math.exp(-rho - (power + 1.0) * log_mean)
    mean_hessian = scale_mu2 * _sum2((power - 1.0) * mean, -power * y)
    scale_mu3 = math.exp(-rho - (power + 2.0) * log_mean)
    l_mmm = power * scale_mu3 * _sum2((power + 1.0) * y, -(power - 1.0) * mean)
    l_mmp = scale_mu2 * _sum3(mean, -y, -log_mean * _sum2((power - 1.0) * mean, -power * y))
    l_mmr = -mean_hessian
    l_mrr = mean_score
    l_mrp = log_mean * mean_score
    l_mpp = log_mean * log_mean * mean_score
    l_mr = -mean_score
    if y == 0.0:
        lam = math.exp(s * log_mean - rho - math.log(s))
        h_p = -log_mean + inverse_s
        h_pp = inverse_s * inverse_s
        h_ppp = 2.0 * h_pp * inverse_s
        l_r = lam
        l_rr = -lam
        l_rrr = lam
        l_rp = lam * h_p
        l_rrp = -lam * h_p
        l_rpp = lam * (h_p * h_p + h_pp)
        l_ppp = -lam * (h_p * h_p * h_p + 3.0 * h_p * h_pp + h_ppp)
    else:
        log_y = math.log(y)
        alpha = s * inverse_r
        a_term = y * math.exp(-r * log_mean)
        c_term = math.exp(s * log_mean)
        f_term = math.exp(-rho)
        canonical = _sum2(-a_term * inverse_r, -c_term * inverse_s)
        b_value = f_term * canonical
        r2 = inverse_r * inverse_r
        r3 = r2 * inverse_r
        r4 = r2 * r2
        s2 = inverse_s * inverse_s
        s3 = s2 * inverse_s
        s4 = s2 * s2
        lm2 = log_mean * log_mean
        lm3 = lm2 * log_mean
        b_p = f_term * _sum2(
            a_term * (log_mean * inverse_r + r2), c_term * (log_mean * inverse_s - s2)
        )
        b_pp = f_term * _sum2(
            a_term * (-lm2 * inverse_r - 2.0 * log_mean * r2 - 2.0 * r3),
            c_term * (-lm2 * inverse_s + 2.0 * log_mean * s2 - 2.0 * s3),
        )
        b_ppp = f_term * _sum2(
            a_term * (lm3 * inverse_r + 3.0 * lm2 * r2 + 6.0 * log_mean * r3 + 6.0 * r4),
            c_term * (lm3 * inverse_s - 3.0 * lm2 * s2 + 6.0 * log_mean * s3 - 6.0 * s4),
        )
        zeta = alpha * (log_y - math.log(r)) - math.log(s) - rho * inverse_r
        d_value = math.log(r) - log_y + rho
        zeta_p = (d_value - s) * r2 + inverse_s
        zeta_pp = (-2.0 * d_value + 2.0 + s) * r3 + s2
        zeta_ppp = (6.0 * d_value - 9.0 - 2.0 * s) * r4 + 2.0 * s3
        if not (math.isfinite(b_value) and math.isfinite(zeta) and math.isfinite(zeta_ppp)):
            return KERNEL_SERIES_BASE, nan10
        status, w = _series_third_cumulants(
            zeta, zeta_p, zeta_pp, zeta_ppp, inverse_r, alpha, max_terms, log_cutoff, coefficients
        )
        if status != KERNEL_OK:
            return status, nan10
        w_r, _, w_rr, w_rp, _, w_rrr, w_rrp, w_rpp, w_ppp = w
        l_r = _sum2(-b_value, w_r)
        l_rr = _sum2(b_value, w_rr)
        l_rrr = _sum2(-b_value, w_rrr)
        l_rp = _sum2(-b_p, w_rp)
        l_rrp = _sum2(b_p, w_rrp)
        l_rpp = _sum2(-b_pp, w_rpp)
        l_ppp = _sum2(b_ppp, w_ppp)
    inverse_phi = 1.0 / phi
    inverse_phi2 = inverse_phi * inverse_phi
    t_mmf = l_mmr * inverse_phi
    t_mff = _sum2(l_mrr, -l_mr) * inverse_phi2
    t_mfp = l_mrp * inverse_phi
    t_fff = _sum3(l_rrr, -3.0 * l_rr, 2.0 * l_r) * inverse_phi2 * inverse_phi
    t_ffp = _sum2(l_rrp, -l_rp) * inverse_phi2
    t_fpp = l_rpp * inverse_phi
    values = (l_mmm, t_mmf, l_mmp, t_mff, t_mfp, l_mpp, t_fff, t_ffp, t_fpp, l_ppp)
    for value in values:
        if not math.isfinite(value):
            return KERNEL_ROW_DERIVATIVES, nan10
    return KERNEL_OK, values


@njit(cache=True, parallel=True)
def _tweedie_hessian_direction_core(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    weight_mode: int,
    direction: NDArray[np.float64],
    max_terms: int,
    log_cutoff: float,
):
    """Packed ``dH(v)`` in ``(mu, phi, p)`` per row; status and first failing row last."""
    n_rows = y.size
    result = np.empty((n_rows, 6), dtype=np.float64)
    statuses = np.empty(n_rows, dtype=np.int64)
    for row in prange(n_rows):  # ty: ignore[not-iterable] -- Numba loop primitive
        coefficients = np.empty(10, dtype=np.float64)
        rho = math.log(dispersion[row])
        multiplier = 1.0
        if weight_mode == 0:
            rho -= math.log(weight[row])
        else:
            multiplier = weight[row]
        status, t = _tweedie_third_row(
            y[row], mean[row], dispersion[row], power[row], rho, max_terms, log_cutoff, coefficients
        )
        statuses[row] = status
        if status != KERNEL_OK:
            continue
        v_m = direction[row, 0]
        v_f = direction[row, 1]
        v_p = direction[row, 2]
        t_mmm, t_mmf, t_mmp, t_mff, t_mfp, t_mpp, t_fff, t_ffp, t_fpp, t_ppp = t
        result[row, 0] = multiplier * (t_mmm * v_m + t_mmf * v_f + t_mmp * v_p)
        result[row, 1] = multiplier * (t_mmf * v_m + t_mff * v_f + t_mfp * v_p)
        result[row, 2] = multiplier * (t_mmp * v_m + t_mfp * v_f + t_mpp * v_p)
        result[row, 3] = multiplier * (t_mff * v_m + t_fff * v_f + t_ffp * v_p)
        result[row, 4] = multiplier * (t_mfp * v_m + t_ffp * v_f + t_fpp * v_p)
        result[row, 5] = multiplier * (t_mpp * v_m + t_fpp * v_f + t_ppp * v_p)
    for row in range(n_rows):
        if statuses[row] != KERNEL_OK:
            return result, statuses[row], row
    return result, KERNEL_OK, -1


# ---------------------------------------------------------------- fourth order
#
# The fourth-order kernel calls the generated closed forms of
# ``_fourth_generated`` compiled here.  Numba's on-disk cache keys a caller on
# its own file only, so the generator rewrites this digest of its output
# whenever it writes the module, which invalidates these cached callers too.
_FOURTH_GENERATED_DIGEST = "94dfd057670ec60498fdf6535ea203284de57f7d483f0c697493ae031f821c6a"

_positive_closed = njit(cache=True)(_generated.tweedie_positive_closed)
_zero_closed = njit(cache=True)(_generated.tweedie_zero_closed)
_zeta_derivatives = njit(cache=True)(_generated.tweedie_zeta_derivatives)
_term_derivatives = njit(cache=True)(_generated.tweedie_term_derivatives)
_series_cumulants = njit(cache=True)(_generated.tweedie_series_cumulants)


@njit(cache=True)
def _polygamma_0123(value: float) -> tuple[float, float, float, float]:
    """Digamma, trigamma, tetragamma and pentagamma of a finite positive scalar.

    The recurrence lifts the argument to at least 16, where the asymptotic
    expansions through ``B_14`` (DLMF 5.11.2 and 5.15.8, differentiated) leave
    a relative remainder below ``1e-16`` in every channel.
    """
    shifted_digamma = 0.0
    shifted_trigamma = 0.0
    shifted_tetragamma = 0.0
    shifted_pentagamma = 0.0
    x = value
    while x < 16.0:
        inverse = 1.0 / x
        inverse2 = inverse * inverse
        shifted_digamma -= inverse
        shifted_trigamma += inverse2
        shifted_tetragamma -= 2.0 * inverse2 * inverse
        shifted_pentagamma += 6.0 * inverse2 * inverse2
        x += 1.0
    inverse = 1.0 / x
    i2 = inverse * inverse
    digamma = 0.0
    trigamma = 0.0
    tetragamma = 0.0
    pentagamma = 0.0
    power = i2
    for k in range(_POLYGAMMA_TERMS):
        b = _POLYGAMMA_BERNOULLI[k]
        order = 2.0 * (k + 1)
        digamma -= b / order * power
        trigamma += b * power * inverse
        tetragamma -= (order + 1.0) * b * power * i2
        pentagamma += (order + 1.0) * (order + 2.0) * b * power * i2 * inverse
        power *= i2
    digamma += shifted_digamma + math.log(x) - 0.5 * inverse
    trigamma += shifted_trigamma + inverse + 0.5 * i2
    tetragamma += shifted_tetragamma - i2 - i2 * inverse
    pentagamma += shifted_pentagamma + 2.0 * i2 * inverse + 3.0 * i2 * i2
    return digamma, trigamma, tetragamma, pentagamma


_POLYGAMMA_TERMS = 7
_POLYGAMMA_BERNOULLI = np.array(
    [1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0, -691.0 / 2730.0, 7.0 / 6.0]
)


@njit(cache=True)
def _fourth_term_channels(
    j: int, zeta1: float, zeta2: float, zeta3: float, zeta4: float, r: float
) -> tuple[float, float, float, float]:
    """``(q_p, q_pp, q_ppp, q_pppp)`` of series term ``j`` (generated)."""
    j_float = float(j)
    p0, p1, p2, p3 = _polygamma_0123((1.0 / r - 1.0) * j_float)
    return _term_derivatives(j_float, r, zeta1, zeta2, zeta3, zeta4, p0, p1, p2, p3)


@njit(cache=True)
def _series_fourth_cumulants(
    zeta: float,
    zeta1: float,
    zeta2: float,
    zeta3: float,
    zeta4: float,
    r: float,
    alpha: float,
    max_terms: int,
    log_cutoff: float,
    coefficients: NDArray[np.float64],
):
    """Status, then ``W``'s 14 derivatives of orders one to four in ``(rho, p)``.

    The window is the Hessian's (walked out from the mode until a term falls
    below ``exp(-log_cutoff)`` of it); pass one forms the anchored means of
    ``j`` and the term derivatives ``Y, Z, V, U``, pass two the central
    moments about them, and the generated ``tweedie_series_cumulants`` turns
    those into the derivatives of the window's log-sum.
    """
    coefficients[0] = math.nan
    nan14 = (
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )
    status, mode, mode_lower_ratio, mode_upper_ratio = _locate_series_mode(
        zeta, alpha, coefficients
    )
    if status != KERNEL_OK:
        return status, nan14
    anchor_y, anchor_z, anchor_v, anchor_u = _fourth_term_channels(
        mode, zeta1, zeta2, zeta3, zeta4, r
    )
    mode_float = float(mode)
    mass = 1.0
    sum_j = 0.0
    sum_y = 0.0
    sum_z = 0.0
    sum_v = 0.0
    sum_u = 0.0
    lowest = mode
    highest = mode
    terms = 1
    for direction in (-1, 1):
        relative_log = 0.0
        current = mode
        while True:
            if direction < 0:
                if current <= 1:
                    break
                ratio_value = (
                    mode_lower_ratio
                    if current == mode
                    else _log_adjacent_ratio(current - 1, zeta, alpha, coefficients)
                )
                if not math.isfinite(ratio_value) or ratio_value <= 0.0:
                    return KERNEL_LOWER_RATIO, nan14
                relative_log -= ratio_value
                current -= 1
            else:
                if current >= _MAX_SAFE_MODE:
                    return KERNEL_WINDOW_RANGE, nan14
                ratio_value = (
                    mode_upper_ratio
                    if current == mode
                    else _log_adjacent_ratio(current, zeta, alpha, coefficients)
                )
                if not math.isfinite(ratio_value) or ratio_value > 0.0:
                    return KERNEL_UPPER_RATIO, nan14
                relative_log += ratio_value
                current += 1
            terms += 1
            if terms > max_terms:
                return KERNEL_MAX_TERMS, nan14
            relative = math.exp(relative_log)
            q_y, q_z, q_v, q_u = _fourth_term_channels(current, zeta1, zeta2, zeta3, zeta4, r)
            mass += relative
            sum_j += relative * (float(current) - mode_float)
            sum_y += relative * (q_y - anchor_y)
            sum_z += relative * (q_z - anchor_z)
            sum_v += relative * (q_v - anchor_v)
            sum_u += relative * (q_u - anchor_u)
            if direction < 0:
                lowest = current
            else:
                highest = current
            if relative_log <= -log_cutoff:
                break
    offset_j = sum_j / mass
    offset_y = sum_y / mass
    offset_z = sum_z / mass
    mean_j = mode_float + offset_j
    mean_y = _sum2(anchor_y, offset_y)
    mean_z = _sum2(anchor_z, offset_z)
    mean_v = _sum2(anchor_v, sum_v / mass)
    mean_u = _sum2(anchor_u, sum_u / mass)
    c_jj = 0.0
    c_jy = 0.0
    c_jz = 0.0
    c_jv = 0.0
    c_yy = 0.0
    c_yz = 0.0
    c_yv = 0.0
    c_zz = 0.0
    c_jjj = 0.0
    c_jjy = 0.0
    c_jjz = 0.0
    c_jyy = 0.0
    c_jyz = 0.0
    c_yyy = 0.0
    c_yyz = 0.0
    c_jjjj = 0.0
    c_jjjy = 0.0
    c_jjyy = 0.0
    c_jyyy = 0.0
    c_yyyy = 0.0
    for direction in (-1, 1):
        relative_log = 0.0
        current = mode
        while True:
            if direction < 0:
                if current != mode:
                    ratio_value = (
                        mode_lower_ratio
                        if current + 1 == mode
                        else _log_adjacent_ratio(current, zeta, alpha, coefficients)
                    )
                    relative_log -= ratio_value
            else:
                if current >= highest:
                    break
                ratio_value = (
                    mode_upper_ratio
                    if current == mode
                    else _log_adjacent_ratio(current, zeta, alpha, coefficients)
                )
                relative_log += ratio_value
                current += 1
            relative = math.exp(relative_log)
            q_y, q_z, q_v, _ = _fourth_term_channels(current, zeta1, zeta2, zeta3, zeta4, r)
            dj = (float(current) - mode_float) - offset_j
            dy = (q_y - anchor_y) - offset_y
            dz = (q_z - anchor_z) - offset_z
            dv = q_v - mean_v
            wj = relative * dj
            wy = relative * dy
            c_jj += wj * dj
            c_jy += wj * dy
            c_jz += wj * dz
            c_jv += wj * dv
            c_yy += wy * dy
            c_yz += wy * dz
            c_yv += wy * dv
            c_zz += relative * dz * dz
            c_jjj += wj * dj * dj
            c_jjy += wj * dj * dy
            c_jjz += wj * dj * dz
            c_jyy += wj * dy * dy
            c_jyz += wj * dy * dz
            c_yyy += wy * dy * dy
            c_yyz += wy * dy * dz
            c_jjjj += wj * dj * dj * dj
            c_jjjy += wj * dj * dj * dy
            c_jjyy += wj * dj * dy * dy
            c_jyyy += wj * dy * dy * dy
            c_yyyy += wy * dy * dy * dy
            if direction < 0:
                if current <= lowest:
                    break
                current -= 1
    inverse_mass = 1.0 / mass
    values = _series_cumulants(
        r,
        mean_j,
        mean_y,
        mean_z,
        mean_v,
        mean_u,
        c_jj * inverse_mass,
        c_jy * inverse_mass,
        c_jz * inverse_mass,
        c_jv * inverse_mass,
        c_yy * inverse_mass,
        c_yz * inverse_mass,
        c_yv * inverse_mass,
        c_zz * inverse_mass,
        c_jjj * inverse_mass,
        c_jjy * inverse_mass,
        c_jjz * inverse_mass,
        c_jyy * inverse_mass,
        c_jyz * inverse_mass,
        c_yyy * inverse_mass,
        c_yyz * inverse_mass,
        c_jjjj * inverse_mass,
        c_jjjy * inverse_mass,
        c_jjyy * inverse_mass,
        c_jyyy * inverse_mass,
        c_yyyy * inverse_mass,
    )
    return KERNEL_OK, values


# Packed positions of the (rho, p) block inside the (mu, rho, p) orders one to
# four: W's derivatives land on the entries that never differentiate in mu.
_SERIES_POSITIONS = np.array([1, 2, 6, 7, 8, 15, 16, 17, 18, 29, 30, 31, 32, 33])


@njit(cache=True)
def _tweedie_fourth_row(
    y: float,
    mean: float,
    power: float,
    rho: float,
    max_terms: int,
    log_cutoff: float,
    coefficients: NDArray[np.float64],
    out: NDArray[np.float64],
) -> int:
    """All 34 derivatives of orders one to four in ``(mu, rho, p)``, ``rho = log phi``.

    The closed part (generated) carries every derivative through ``mu``; the
    positive row's series ``W(rho, p)`` adds its cumulants to the rest.
    Returns the status; ``out`` holds the values.
    """
    log_mean = math.log(mean)
    r = power - 1.0
    s = 2.0 - power
    f_term = math.exp(-rho)
    c_term = math.exp(s * log_mean)
    m_term = math.exp(-log_mean)
    if y == 0.0:
        closed = _zero_closed(f_term, 0.0, c_term, m_term, log_mean, r, s)
        for index in range(34):
            out[index] = closed[index]
    else:
        log_y = math.log(y)
        a_term = y * math.exp(-r * log_mean)
        closed = _positive_closed(f_term, a_term, c_term, m_term, log_mean, r, s)
        alpha = s / r
        d_value = math.log(r) - log_y + rho
        zeta = alpha * (log_y - math.log(r)) - math.log(s) - rho / r
        zeta1, zeta2, zeta3, zeta4 = _zeta_derivatives(r, s, d_value)
        if not (math.isfinite(zeta) and math.isfinite(zeta4)):
            return KERNEL_SERIES_BASE
        status, series = _series_fourth_cumulants(
            zeta, zeta1, zeta2, zeta3, zeta4, r, alpha, max_terms, log_cutoff, coefficients
        )
        if status != KERNEL_OK:
            return status
        for index in range(34):
            out[index] = closed[index]
        for index in range(14):
            out[_SERIES_POSITIONS[index]] += series[index]
    for index in range(34):
        if not math.isfinite(out[index]):
            return KERNEL_ROW_DERIVATIVES
    return KERNEL_OK


@njit(cache=True, parallel=True)
def _tweedie_fourth_core(
    y: NDArray[np.float64],
    mean: NDArray[np.float64],
    dispersion: NDArray[np.float64],
    power: NDArray[np.float64],
    weight: NDArray[np.float64],
    weight_mode: int,
    max_terms: int,
    log_cutoff: float,
):
    """``(n, 34)`` derivatives in ``(mu, rho, p)``, weighted; status and first failing row last."""
    n_rows = y.size
    result = np.empty((n_rows, 34), dtype=np.float64)
    statuses = np.empty(n_rows, dtype=np.int64)
    for row in prange(n_rows):  # ty: ignore[not-iterable] -- Numba loop primitive
        coefficients = np.empty(10, dtype=np.float64)
        rho = math.log(dispersion[row])
        multiplier = 1.0
        if weight_mode == 0:
            rho -= math.log(weight[row])
        else:
            multiplier = weight[row]
        status = _tweedie_fourth_row(
            y[row], mean[row], power[row], rho, max_terms, log_cutoff, coefficients, result[row]
        )
        statuses[row] = status
        if status == KERNEL_OK and multiplier != 1.0:
            for index in range(34):
                result[row, index] *= multiplier
    for row in range(n_rows):
        if statuses[row] != KERNEL_OK:
            return result, statuses[row], row
    return result, KERNEL_OK, -1


__all__ = [
    "_tweedie_fourth_core",
    "_evaluate_tweedie_batch_core",
    "_evaluate_tweedie_batch_parallel_core",
    "_tweedie_hessian_direction_core",
]
