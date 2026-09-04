"""Serial strict-IEEE Numba core for the internal Tweedie LSS point kernel."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange  # type: ignore[import-untyped]
from numpy.typing import NDArray

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
) -> tuple[int, int]:
    first = _log_adjacent_ratio(1, zeta, alpha, coefficients)
    if not math.isfinite(first):
        return KERNEL_MODE_RATIO, 0
    if first <= 0.0:
        return KERNEL_OK, 1

    lower = 1
    upper = 2
    while True:
        ratio = _log_adjacent_ratio(upper, zeta, alpha, coefficients)
        if not math.isfinite(ratio):
            return KERNEL_MODE_RATIO, 0
        if ratio <= 0.0:
            break
        if upper >= _MAX_SAFE_MODE:
            return KERNEL_MODE_RANGE, 0
        lower = upper
        upper = min(2 * upper, _MAX_SAFE_MODE)

    while upper - lower > 1:
        midpoint = lower + (upper - lower) // 2
        ratio = _log_adjacent_ratio(midpoint, zeta, alpha, coefficients)
        if not math.isfinite(ratio):
            return KERNEL_MODE_RATIO, 0
        if ratio > 0.0:
            lower = midpoint
        else:
            upper = midpoint

    before = _log_adjacent_ratio(upper - 1, zeta, alpha, coefficients)
    after = _log_adjacent_ratio(upper, zeta, alpha, coefficients)
    if before <= 0.0 or after > 0.0:
        return KERNEL_MODE_BRACKET, 0
    return KERNEL_OK, upper


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
    _fill_log_gamma_increment_coefficients(alpha, coefficients)
    status, mode = _locate_series_mode(zeta, alpha, coefficients)
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
    modeled_phi_squared = modeled_phi * modeled_phi
    if modeled_phi_squared == 0.0:
        return _row_failure(KERNEL_REQUIRED_WORK)
    hessian_phi_phi = (hessian_rho_rho - score_rho) / modeled_phi_squared
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
    modeled_phi_squared = modeled_phi * modeled_phi
    if modeled_phi_squared == 0.0:
        return _row_failure(KERNEL_REQUIRED_WORK)
    hessian_phi_phi = (hessian_rho_rho - score_rho) / modeled_phi_squared
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


__all__ = ["_evaluate_tweedie_batch_core", "_evaluate_tweedie_batch_parallel_core"]
