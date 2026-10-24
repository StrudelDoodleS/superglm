"""Per-row Python reference for the NB2 near-Poisson theta series (superglm v0.40.0).

The production kernel evaluates this series in compiled code
(``superglm.distributional.kernels._negative_binomial_numba``).  This is the
v0.40.0 loop, verbatim, so the equivalence test can hold the compiled rows to
its bits.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import gamma
from superglm.distributional.kernels.negative_binomial import (
    _FLOAT,
    _POWER_SUM_BERNOULLI,
    _SERIES_ORDER,
    _SERIES_RATIO,
    _TINY,
)


def _gamma(operations):
    """The kernel's bounds count in eps = 2u: ``n eps / (1 - n eps)`` is gamma_2n."""
    return gamma(2 * operations)


def _scaled_power_sums(count: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized float64 power sums and absolute roundoff bounds."""

    values = np.zeros(_SERIES_ORDER, dtype=_FLOAT)
    errors = np.zeros(_SERIES_ORDER, dtype=_FLOAT)
    if count < 2:
        values[0] = float(count)
        return values, errors

    inverse_count = _FLOAT(1.0) / _FLOAT(count)
    coefficients = np.zeros(_SERIES_ORDER, dtype=_FLOAT)
    for power in range(_SERIES_ORDER):
        choose = _FLOAT(1.0)
        for term in range(power + 1):
            coefficients[term] = choose * _POWER_SUM_BERNOULLI[term]
            if term < power:
                choose *= _FLOAT(power + 1 - term) / _FLOAT(term + 1)
        polynomial = coefficients[power]
        absolute = abs(polynomial)
        for term in range(power - 1, -1, -1):
            polynomial = coefficients[term] + inverse_count * polynomial
            absolute = abs(coefficients[term]) + inverse_count * absolute
        denominator = _FLOAT(power + 1)
        values[power] = polynomial / denominator
        errors[power] = _gamma(4 * power + 8) * absolute / denominator + _TINY
    return values, errors


def _series_hessian_step(
    order: int,
    coefficient: float,
    coefficient_abs: float,
    coefficient_error: float,
    x: float,
    state: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Advance only the Hessian Horner state for one series coefficient."""

    weighted, transformed, weighted_absolute, weighted_error = state
    return (
        order * coefficient + x * weighted,
        (1.0 - order) * coefficient + x * transformed,
        order * coefficient_abs + x * weighted_absolute,
        order * coefficient_error + x * weighted_error,
    )


def _theta_series(
    counts: NDArray[np.int64],
    mean: NDArray[np.float64],
    theta: NDArray[np.float64],
    remainder_error: NDArray[np.float64],
    remainder_scale: NDArray[np.float64],
    derivative_order: Literal[1, 2],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the fixed order-12 series only when its error beats recurrence."""

    n_rows = len(counts)
    use = np.zeros((n_rows, derivative_order), dtype=np.bool_)
    natural = np.zeros((n_rows, derivative_order), dtype=_FLOAT)
    direct = np.zeros((n_rows, derivative_order), dtype=_FLOAT)
    candidate_scale = np.maximum(mean, np.maximum(counts.astype(_FLOAT) - 1.0, 0.0))
    candidates = np.flatnonzero(candidate_scale / theta <= _SERIES_RATIO)

    for row in candidates:
        count, row_mean, row_theta = int(counts[row]), float(mean[row]), float(theta[row])
        scale = float(candidate_scale[row])
        x = scale / row_theta
        polynomial = absolute = coefficient_error_absolute = 0.0
        if derivative_order == 2:
            weighted = transformed = weighted_absolute = weighted_error = 0.0
        sums, sum_errors = _scaled_power_sums(count)
        count_scale = _FLOAT(count) / _FLOAT(scale) if count > 1 else _FLOAT(0.0)
        for order in range(_SERIES_ORDER, 1, -1):
            if count > 1:
                offset_factor = float(count_scale**order)
                offset = float(sums[order - 1]) * offset_factor
                offset_error = (
                    abs(offset_factor) * float(sum_errors[order - 1])
                    + _gamma(order + 2) * abs(offset)
                    + _TINY
                )
            else:
                offset = offset_error = 0.0
            mean_term = (row_mean / scale) ** (order - 1) * (
                ((order - 1.0) / order * row_mean - count) / scale
            )
            sign = -1.0 if order % 2 == 0 else 1.0
            coefficient = sign * (offset + mean_term)
            coefficient_abs = abs(offset) + abs(mean_term)
            polynomial = coefficient + x * polynomial
            absolute = coefficient_abs + x * absolute
            coefficient_error_absolute = offset_error + x * coefficient_error_absolute
            if derivative_order == 2:
                weighted, transformed, weighted_absolute, weighted_error = _series_hessian_step(
                    order,
                    coefficient,
                    coefficient_abs,
                    offset_error,
                    x,
                    (weighted, transformed, weighted_absolute, weighted_error),
                )

        natural[row, 0] = x * x * polynomial
        direct[row, 0] = scale * x * polynomial
        mean_ratio = row_mean / row_theta
        offset_ratio = max(count - 1, 0) / row_theta
        delta = abs(row_mean - count)
        score_tail = (
            count / row_theta * offset_ratio**_SERIES_ORDER / (1.0 - offset_ratio)
            + mean_ratio ** (_SERIES_ORDER + 1) / (_SERIES_ORDER + 1.0)
            + delta / row_theta * mean_ratio**_SERIES_ORDER / (1.0 - mean_ratio)
        )
        series_score_error = (
            x * x * (_gamma(80) * absolute + (1.0 + _gamma(32)) * coefficient_error_absolute)
            + score_tail
            + _TINY
        )
        total = row_mean + row_theta
        recurrence_score_scale = count / total * (
            (row_mean + max(count - 1, 0)) / row_theta
        ) + float(remainder_scale[row])
        recurrence_score_error = _gamma(count + 64) * recurrence_score_scale + float(
            remainder_error[row]
        )
        use[row, 0] = series_score_error < recurrence_score_error

        if derivative_order == 1:
            continue
        natural[row, 1] = -(x * x / row_theta) * weighted
        direct[row, 1] = scale * x * transformed
        hessian_tail = (
            count
            / row_theta**2
            * offset_ratio**_SERIES_ORDER
            * (_SERIES_ORDER + 1.0 - _SERIES_ORDER * offset_ratio)
            / (1.0 - offset_ratio) ** 2
            + mean_ratio ** (_SERIES_ORDER + 1) / row_theta / (1.0 - mean_ratio)
            + delta
            / row_theta**2
            * mean_ratio**_SERIES_ORDER
            * (_SERIES_ORDER + 1.0 - _SERIES_ORDER * mean_ratio)
            / (1.0 - mean_ratio) ** 2
        )
        series_hessian_error = (
            x
            * x
            / row_theta
            * (_gamma(96) * weighted_absolute + (1.0 + _gamma(32)) * weighted_error)
            + hessian_tail
            + _TINY
        )
        recurrence_hessian_scale = (
            recurrence_score_scale * (1.0 / row_theta + 1.0 / total)
            + (row_mean / total) * mean_ratio / total
        )
        use[row, 1] = series_hessian_error < (_gamma(count + 80) * recurrence_hessian_scale + _TINY)

    return use, natural, direct
