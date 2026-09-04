"""Independent Gamma mean-CV row laws for family-kernel tests.

This module deliberately uses SciPy's normalized Gamma density and numerical
perturbations. It does not import any production distributional-family code.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal

import numpy as np
import scipy.stats
from scipy import special

WeightLaw = Literal["prior", "frequency"]
GammaAsymptoticCombination = Literal["A", "J", "C"]

_EPS = np.finfo(np.float64).eps
# One normalized SciPy row contributes shape/scale transforms, logarithms,
# gammaln, and their final arithmetic before the stencil combines the sample.
_ROW_REFERENCE_OPERATIONS = 16
_FIRST_STENCIL_OPERATIONS = _ROW_REFERENCE_OPERATIONS + 8
_DIAGONAL_STENCIL_OPERATIONS = _ROW_REFERENCE_OPERATIONS + 10
_MIXED_STENCIL_OPERATIONS = _ROW_REFERENCE_OPERATIONS + 32

_BERNOULLI_EVEN = (
    Fraction(1, 6),
    Fraction(-1, 30),
    Fraction(1, 42),
    Fraction(-1, 30),
    Fraction(5, 66),
    Fraction(-691, 2730),
    Fraction(7, 6),
    Fraction(-3617, 510),
    Fraction(43867, 798),
    Fraction(-174611, 330),
    Fraction(854513, 138),
)
_DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
    "208998628034825342117067982148086513282306647093844609550582231725359408128"
)


@dataclass(frozen=True)
class GammaAsymptoticOracle:
    """Exact signed remainder intervals and a rounded retained expansion."""

    coarse_interval: tuple[Fraction, Fraction]
    tight_interval: tuple[Fraction, Fraction]
    retained_float: float


def _gamma_asymptotic_term(
    combination: GammaAsymptoticCombination,
    n: int,
    shape: Fraction,
) -> Fraction:
    bernoulli = _BERNOULLI_EVEN[n - 1]
    inverse_power = Fraction(1, 1) / shape ** (2 * n - 1)
    if combination == "A":
        return -bernoulli * inverse_power / (2 * n)
    if combination == "J":
        return bernoulli * inverse_power
    return -bernoulli * inverse_power / (2 * n * (2 * n - 1))


def _signed_remainder_interval(partial: Fraction, omitted: Fraction) -> tuple[Fraction, Fraction]:
    endpoint = partial + omitted
    return (min(partial, endpoint), max(partial, endpoint))


def gamma_asymptotic_oracle(
    combination: GammaAsymptoticCombination,
    shape: float,
) -> GammaAsymptoticOracle:
    """Return independent Bernoulli remainder evidence for one large shape."""

    if not np.isfinite(shape) or shape <= 0.0:
        raise ValueError("shape must be finite and strictly positive")
    retained_terms = {"A": 5, "J": 6, "C": 4}[combination]
    rational_shape = Fraction.from_float(float(shape))
    terms = tuple(
        _gamma_asymptotic_term(combination, n, rational_shape)
        for n in range(1, len(_BERNOULLI_EVEN) + 1)
    )
    coarse_partial = sum(terms[:retained_terms], Fraction())
    tight_partial = sum(terms[:10], Fraction())
    coarse_interval = _signed_remainder_interval(
        coarse_partial,
        terms[retained_terms],
    )
    tight_interval = _signed_remainder_interval(tight_partial, terms[10])

    if combination == "A":
        retained_float = float(Fraction(-1, 2) + coarse_partial)
    elif combination == "J":
        retained_float = float(Fraction(1, 2) + coarse_partial)
    else:
        with localcontext() as context:
            context.prec = 120
            decimal_shape = Decimal(rational_shape.numerator) / Decimal(rational_shape.denominator)
            correction = Decimal(coarse_partial.numerator) / Decimal(coarse_partial.denominator)
            retained_float = float((decimal_shape / (2 * _DECIMAL_PI)).ln() / 2 + correction)
    return GammaAsymptoticOracle(
        coarse_interval=coarse_interval,
        tight_interval=tight_interval,
        retained_float=retained_float,
    )


def gamma_row_reference(
    y: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    semantics: Literal["prior", "frequency"],
) -> np.ndarray:
    """Return literal normalized Gamma rows from SciPy."""

    if semantics == "prior":
        shape = weights / scale**2
        gamma_scale = mean * scale**2 / weights
        multiplier = np.ones_like(weights)
    else:
        shape = 1.0 / scale**2
        gamma_scale = mean * scale**2
        multiplier = weights
    return multiplier * scipy.stats.gamma.logpdf(
        y,
        a=shape,
        scale=gamma_scale,
    )


def _row_value(
    y: float,
    mean: float,
    scale: float,
    weight: float,
    semantics: WeightLaw,
) -> float:
    return float(
        gamma_row_reference(
            np.array([y]),
            np.array([mean]),
            np.array([scale]),
            np.array([weight]),
            semantics,
        )[0]
    )


def _supported_step(parameter: float, exponent: float, step_scale: float) -> float:
    step = step_scale * _EPS**exponent * max(1.0, abs(parameter))
    # Both five-point diagonal stencils reach two steps toward zero.
    return min(step, np.nextafter(0.5 * parameter, 0.0))


def _roundoff_bound(weighted_sample_magnitude: float, denominator: float, operations: int) -> float:
    accumulation = operations * _EPS / (1.0 - operations * _EPS)
    return accumulation * weighted_sample_magnitude / denominator


@dataclass(frozen=True)
class _FiniteDifferenceState:
    score: np.ndarray
    hessian: np.ndarray
    score_roundoff: np.ndarray
    hessian_roundoff: np.ndarray


def _finite_difference_state(
    y: float,
    mean: float,
    scale: float,
    weight: float,
    semantics: WeightLaw,
    *,
    step_scale: float = 1.0,
) -> _FiniteDifferenceState:

    h_first = np.array(
        [
            _supported_step(mean, 1.0 / 5.0, step_scale),
            _supported_step(scale, 1.0 / 5.0, step_scale),
        ]
    )
    h_second = np.array(
        [
            _supported_step(mean, 1.0 / 6.0, step_scale),
            _supported_step(scale, 1.0 / 6.0, step_scale),
        ]
    )

    def value(parameters: np.ndarray) -> float:
        return _row_value(y, parameters[0], parameters[1], weight, semantics)

    center = np.array([mean, scale], dtype=np.float64)
    score = np.empty(2, dtype=np.float64)
    score_roundoff = np.empty(2, dtype=np.float64)
    diagonal = np.empty(2, dtype=np.float64)
    diagonal_roundoff = np.empty(2, dtype=np.float64)
    for channel in range(2):
        h1 = h_first[channel]
        offsets = []
        for multiple in (-2.0, -1.0, 1.0, 2.0):
            point = center.copy()
            point[channel] += multiple * h1
            offsets.append(value(point))
        score[channel] = (offsets[0] - 8.0 * offsets[1] + 8.0 * offsets[2] - offsets[3]) / (
            12.0 * h1
        )
        score_magnitude = float(np.dot(np.array([1.0, 8.0, 8.0, 1.0]), np.abs(offsets)))
        score_roundoff[channel] = _roundoff_bound(
            score_magnitude,
            12.0 * h1,
            _FIRST_STENCIL_OPERATIONS,
        )

        h2 = h_second[channel]
        second_offsets = []
        for multiple in (-2.0, -1.0, 0.0, 1.0, 2.0):
            point = center.copy()
            point[channel] += multiple * h2
            second_offsets.append(value(point))
        diagonal[channel] = (
            -second_offsets[0]
            + 16.0 * second_offsets[1]
            - 30.0 * second_offsets[2]
            + 16.0 * second_offsets[3]
            - second_offsets[4]
        ) / (12.0 * h2 * h2)
        diagonal_magnitude = float(
            np.dot(
                np.array([1.0, 16.0, 30.0, 16.0, 1.0]),
                np.abs(second_offsets),
            )
        )
        diagonal_roundoff[channel] = _roundoff_bound(
            diagonal_magnitude,
            12.0 * h2 * h2,
            _DIAGONAL_STENCIL_OPERATIONS,
        )

    hm, hs = h_second
    mixed = 0.0
    mixed_magnitude = 0.0
    coefficients = {-2: 1.0, -1: -8.0, 1: 8.0, 2: -1.0}
    for i, ci in coefficients.items():
        for j, cj in coefficients.items():
            point = center + np.array([i * hm, j * hs])
            sample = value(point)
            mixed += ci * cj * sample
            mixed_magnitude += abs(ci * cj) * abs(sample)
    mixed /= 144.0 * hm * hs
    mixed_roundoff = _roundoff_bound(
        mixed_magnitude,
        144.0 * hm * hs,
        _MIXED_STENCIL_OPERATIONS,
    )
    return _FiniteDifferenceState(
        score=score,
        hessian=np.array([diagonal[0], mixed, diagonal[1]]),
        score_roundoff=score_roundoff,
        hessian_roundoff=np.array([diagonal_roundoff[0], mixed_roundoff, diagonal_roundoff[1]]),
    )


def finite_difference_row_derivatives(
    y: float,
    mean: float,
    scale: float,
    weight: float,
    semantics: WeightLaw,
    *,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return five-point score and packed raw Hessian for one scalar row."""

    state = _finite_difference_state(y, mean, scale, weight, semantics, step_scale=step_scale)
    return state.score, state.hessian


def finite_difference_error_bounds(
    y: float,
    mean: float,
    scale: float,
    weight: float,
    semantics: WeightLaw,
) -> tuple[np.ndarray, np.ndarray]:
    """Return oracle-only roundoff plus fourth-order truncation envelopes."""

    coarse = _finite_difference_state(y, mean, scale, weight, semantics)
    fine = _finite_difference_state(y, mean, scale, weight, semantics, step_scale=0.5)
    score_bound = (
        np.abs(coarse.score - fine.score) / 15.0
        + (16.0 * fine.score_roundoff + coarse.score_roundoff) / 15.0
    )
    hessian_bound = (
        np.abs(coarse.hessian - fine.hessian) / 15.0
        + (16.0 * fine.hessian_roundoff + coarse.hessian_roundoff) / 15.0
    )
    return score_bound, hessian_bound


def gamma_scaled_target_interval(
    y: np.ndarray,
    mean: float,
    weights: np.ndarray,
    semantics: WeightLaw,
    rho: float,
) -> tuple[float, float]:
    """Return an independent float64 interval for the scaled initialization target."""

    response = np.asarray(y, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    k = float(np.exp(rho))
    shape = mass * k if semantics == "prior" else np.full(len(response), k)
    multiplier = np.ones(len(response)) if semantics == "prior" else mass
    z = response / mean
    digamma_term = shape * special.digamma(shape)
    log_shape_term = shape * np.log(shape)
    az = shape * z
    a_log_z = shape * np.log(z)
    terms = multiplier * (digamma_term - log_shape_term + az - shape - a_log_z)
    value = float(np.sum(terms, dtype=np.float64))
    primitive_magnitude = multiplier * (
        np.abs(digamma_term) + np.abs(log_shape_term) + np.abs(az) + shape + np.abs(a_log_z)
    )
    error = (
        32.0
        * _EPS
        * float(
            np.sum(primitive_magnitude, dtype=np.float64)
            + np.sum(np.abs(terms), dtype=np.float64)
            + np.sum(multiplier, dtype=np.float64)
        )
    )
    return value, error


def gamma_scaled_ratio_oracle(
    y: np.ndarray,
    mean: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray:
    """Return independent high-precision ``(a*z, a*t, a*d)`` rows."""

    rows: list[tuple[float, float, float]] = []
    with localcontext() as context:
        context.prec = 100
        for y_value, mean_value, shape_value in zip(y, mean, shape, strict=True):
            response = Decimal.from_float(float(y_value))
            location = Decimal.from_float(float(mean_value))
            a = Decimal.from_float(float(shape_value))
            ratio = response / location
            az = a * ratio
            at = a * (ratio - 1)
            ad = a * (ratio - 1 - ratio.ln())
            rows.append((float(az), float(at), float(ad)))
    return np.asarray(rows, dtype=np.float64)
