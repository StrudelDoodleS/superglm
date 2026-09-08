"""Execution witnesses and independent error checks for Gamma origin series."""

from __future__ import annotations

import math
import sys
from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np
import pytest
from scipy import special

from superglm.distributional.kernels import gamma as gamma_kernel

_EPS = np.finfo(np.float64).eps
_MIN_FLOAT = np.nextafter(0.0, 1.0)
_ORIGIN_HELPERS = (
    ("_scaled_digamma_residual", "_small_a_residual"),
    ("_scaled_trigamma_residual", "_small_j_residual"),
    ("_scaled_trigamma_log_derivative", "_small_j_log_derivative"),
    ("_gamma_log_normalizer", "_small_log_normalizer"),
)
_BERNOULLI = (
    (1, 6),
    (-1, 30),
    (1, 42),
    (-1, 30),
    (5, 66),
    (-691, 2730),
    (7, 6),
    (-3617, 510),
    (43867, 798),
    (-174611, 330),
    (854513, 138),
)


@lru_cache
def _decimal_zeta(exponent: int) -> Decimal:
    """Euler--Maclaurin oracle, independent of the production SciPy table.

    For x**(-s), the remainder is bounded by the first omitted Bernoulli
    term. Ten corrections at N=64 put it below 1e-35 for the exponents
    used here; Decimal rounding at precision 96 is much smaller again.
    """
    with localcontext() as context:
        context.prec = 96
        endpoint = Decimal(64)
        total = sum(Decimal(k) ** -exponent for k in range(1, 64))
        total += endpoint ** (1 - exponent) / (exponent - 1)
        total += endpoint**-exponent / 2
        for order, (numerator, denominator) in enumerate(_BERNOULLI, 1):
            rising = math.prod(range(exponent, exponent + 2 * order - 1))
            term = (
                Decimal(numerator)
                / denominator
                / math.factorial(2 * order)
                * rising
                * endpoint ** -(exponent + 2 * order - 1)
            )
            if order == len(_BERNOULLI):
                assert abs(term) < Decimal("1e-35")
            else:
                total += term
        return total


@lru_cache
def _decimal_euler() -> Decimal:
    with localcontext() as context:
        context.prec = 96
        endpoint = Decimal(64)
        total = sum(Decimal(1) / k for k in range(1, 65))
        total -= endpoint.ln() + 1 / (2 * endpoint)
        for order, (numerator, denominator) in enumerate(_BERNOULLI, 1):
            term = Decimal(numerator) / (denominator * 2 * order) / endpoint ** (2 * order)
            if order == len(_BERNOULLI):
                assert abs(term) < Decimal("1e-35")
            else:
                total += term
        return total


def _origin_oracle(shape: Decimal) -> tuple[list[Decimal], list[Decimal]]:
    """Retain 64 terms with a common truncation enclosure below 1e-32."""
    euler = _decimal_euler()
    log_shape = shape.ln()
    values = [
        -1 - shape * (euler + log_shape),
        1 - shape,
        -shape,
        (shape + 1) * log_shape + (euler - 1) * shape,
    ]
    magnitudes = [
        1 + abs(shape * (euler + log_shape)),
        1 + shape,
        shape,
        abs((shape + 1) * log_shape) + abs((euler - 1) * shape),
    ]
    for order in range(2, 65):
        term = (-1) ** order * _decimal_zeta(order) * shape**order
        terms = [term, (order - 1) * term, order * (order - 1) * term, -term / order]
        for channel in range(4):
            values[channel] += terms[channel]
            magnitudes[channel] += abs(terms[channel])
    # The third combination has the largest absolute geometric tail.
    tail = _decimal_zeta(2) * 64 * 65 * shape**65 / (1 - shape) ** 3
    assert tail < Decimal("1e-32")
    return values, magnitudes


def _coefficient_error_bound(shape: Decimal, channel: int) -> Decimal:
    """Propagate independently enclosed library coefficient errors linearly."""
    total = Decimal(0)
    for order in range(2, 65):
        coefficient = Decimal.from_float(float(gamma_kernel._ORIGIN_ZETA[order - 2]))
        factor = (1, order - 1, order * (order - 1), Decimal(1) / order)[channel]
        total += abs(coefficient - _decimal_zeta(order)) * shape**order * factor
    return total


@pytest.mark.parametrize(("helper_name", "scalar_name"), _ORIGIN_HELPERS)
def test_small_shape_batches_do_not_execute_python_scalar_series(
    monkeypatch: pytest.MonkeyPatch, helper_name: str, scalar_name: str
) -> None:
    """The unfixed implementation executes one Python series per small row."""
    scalar = getattr(gamma_kernel, scalar_name)
    calls = 0

    def counted(shape: float) -> float:
        nonlocal calls
        calls += 1
        return scalar(shape)

    monkeypatch.setattr(gamma_kernel, scalar_name, counted)
    shapes = np.linspace(0.001, np.nextafter(0.25, 0.0), 257)
    getattr(gamma_kernel, helper_name)(shapes)
    assert calls == 0


@pytest.mark.parametrize("channel", range(4))
def test_origin_series_obey_scale_aware_decimal_error_bounds(channel: int) -> None:
    shapes = np.array(
        [
            _MIN_FLOAT,
            np.finfo(np.float64).tiny / 4,
            np.finfo(np.float64).tiny,
            1e-150,
            1e-20,
            1e-8,
            1e-5,
            0.003,
            0.01,
            0.125,
            np.nextafter(0.25, 0.0),
            0.25,
            np.nextafter(0.25, np.inf),
        ]
    )
    actual = getattr(gamma_kernel, _ORIGIN_HELPERS[channel][0])(shapes)
    with localcontext() as context:
        context.prec = 96
        # At most 80 retained terms: power formation, coefficient products,
        # and accumulation have at most 2*80+12 rounding steps on a path.
        # Library coefficient errors are enclosed independently below.
        unit_roundoff = Decimal.from_float(_EPS / 2)
        steps = 2 * 80 + 12
        gamma_n = steps * unit_roundoff / (1 - steps * unit_roundoff)
        for shape, observed in zip(shapes, actual, strict=True):
            decimal_shape = Decimal.from_float(float(shape))
            expected, magnitudes = _origin_oracle(decimal_shape)
            tail_budget = Decimal.from_float(_EPS / 8) * max(1, abs(expected[channel]))
            coefficient_error = _coefficient_error_bound(decimal_shape, channel)
            bound = (
                gamma_n * magnitudes[channel] + coefficient_error + tail_budget + Decimal("1e-32")
            )
            assert abs(Decimal.from_float(float(observed)) - expected[channel]) <= bound
            if shape < 0.25:
                scalar = getattr(gamma_kernel, _ORIGIN_HELPERS[channel][1])(float(shape))
                assert abs(Decimal.from_float(scalar) - expected[channel]) <= bound


def test_subnormal_shape_preserves_the_nonzero_log_curvature_derivative() -> None:
    shapes = np.array([_MIN_FLOAT, np.finfo(np.float64).tiny / 4, 1e-150])
    actual = gamma_kernel._scaled_trigamma_log_derivative(shapes)
    # All higher terms are below half an ulp of -a on this fixture.
    np.testing.assert_array_equal(actual, -shapes)


def test_origin_table_preserves_scalar_special_function_coefficients() -> None:
    expected = np.array([float(special.zeta(exponent, 1.0)) for exponent in range(2, 82)])
    np.testing.assert_array_equal(gamma_kernel._ORIGIN_ZETA, expected)


def _old_small_j_log_derivative(shape: float) -> float:
    """Literal legacy series retaining its underbounded geometric tail."""
    total = -shape
    power = shape * shape
    sign = 1.0
    for n in range(80):
        total += sign * (n + 1.0) * (n + 2.0) * float(special.zeta(n + 2, 1.0)) * power
        power *= shape
        sign = -sign
        tail = (n + 2.0) * (n + 3.0) * abs(power) / (1.0 - shape) ** 3
        if tail <= _EPS * max(1.0, abs(total)) / 8.0:
            return total
    raise AssertionError("legacy series did not converge")


def test_origin_tail_bound_covers_zeta_coefficients() -> None:
    # Without a zeta upper bound, the first omitted term of dJ/dlog(a)
    # exceeds the EPS/8 budget even though the geometric stopping test passes.
    shape = (19 * _EPS / (20 * 48)) ** (1 / 3)
    actual = gamma_kernel._scaled_trigamma_log_derivative(np.array([shape]))[0]
    scalar = gamma_kernel._small_j_log_derivative(shape)
    old_scalar = _old_small_j_log_derivative(shape)
    with localcontext() as context:
        context.prec = 96
        expected, magnitudes = _origin_oracle(Decimal.from_float(shape))
        bound = Decimal.from_float(_EPS / 8) + Decimal.from_float(8 * _EPS) * magnitudes[2]
        assert abs(Decimal.from_float(old_scalar) - expected[2]) > bound
        assert abs(Decimal.from_float(float(actual)) - expected[2]) <= bound
        assert abs(Decimal.from_float(scalar) - expected[2]) <= bound


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_small_shape_signed_natural_channels_match_decimal_likelihood(semantics: str) -> None:
    mean = np.array([0.5, 1.0, 2.0, 8.0, 64.0])
    response = mean * np.array([0.25, 0.5, 1.0, 4.0, 128.0])
    scale = np.full(5, 4.0)
    weights = (
        np.array([0.125, 0.5, 1.0, 2.0, 3.0])
        if semantics == "prior"
        else np.array([1.0, 2.0, 3.0, 5.0, 11.0])
    )
    result = gamma_kernel.evaluate_gamma_rows(
        response, mean, scale, weights, semantics, derivative_order=2
    )
    information = gamma_kernel.gamma_expected_information(mean, scale, weights, semantics)
    assert result.score is not None
    assert result.hessian_packed is not None
    with localcontext() as context:
        context.prec = 96
        for row in range(len(mean)):
            y, mu, sigma, weight = map(
                Decimal.from_float, (response[row], mean[row], scale[row], weights[row])
            )
            multiplier = weight if semantics == "frequency" else Decimal(1)
            a = (weight if semantics == "prior" else Decimal(1)) / sigma**2
            (a_residual, j_residual, _, normalizer), _ = _origin_oracle(a)
            z = y / mu
            ad = a * (z - 1 - z.ln())
            b = a_residual + ad
            expected = [
                multiplier * (normalizer - ad),
                multiplier * a * (z - 1) / mu,
                2 * multiplier * b / sigma,
                -multiplier * a * (2 * z - 1) / mu**2,
                -2 * multiplier * a * (z - 1) / (mu * sigma),
                -2 * multiplier * (3 * b + 2 * j_residual) / sigma**2,
                multiplier * a / mu**2,
                4 * multiplier * j_residual / sigma**2,
            ]
            actual = [
                result.optimizing_log_likelihood[row],
                *result.score[row],
                *result.hessian_packed[row],
                information[row, 0],
                information[row, 2],
            ]
            # This fixture stays away from cancellation in nonzero channels.
            # 200 rounding steps cover the series and natural-scale products.
            gamma_n = Decimal.from_float(200 * _EPS / (1 - 200 * _EPS))
            for observed, reference in zip(actual, expected, strict=True):
                assert np.sign(observed) == (1 if reference > 0 else -1 if reference < 0 else 0)
                assert abs(Decimal.from_float(float(observed)) - reference) <= gamma_n * abs(
                    reference
                )


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_smallest_shape_retains_finite_natural_channels(semantics: str) -> None:
    mean = np.ones(1)
    scale = np.array([math.ldexp(1.0, 537)])
    weights = np.ones(1)
    result = gamma_kernel.evaluate_gamma_rows(
        mean, mean, scale, weights, semantics, derivative_order=2
    )
    information = gamma_kernel.gamma_expected_information(mean, scale, weights, semantics)
    assert result.score is not None
    assert result.hessian_packed is not None
    np.testing.assert_array_equal(result.score, [[0.0, -math.ldexp(1.0, -536)]])
    np.testing.assert_array_equal(result.hessian_packed, [[-_MIN_FLOAT, 0.0, 2 * _MIN_FLOAT]])
    np.testing.assert_array_equal(information, [[_MIN_FLOAT, 0.0, 4 * _MIN_FLOAT]])


def _old_initial_target(
    response: np.ndarray, mean: float, weights: np.ndarray, semantics: str, rho: float
) -> float | None:
    """Literal pre-optimization target, including its summand and fsum order."""
    try:
        k = math.exp(rho)
        with np.errstate(over="raise", invalid="raise"):
            shape = weights * k if semantics == "prior" else np.full(len(response), k)
        if not np.all(np.isfinite(shape)) or np.any(shape <= 0.0):
            return None
        a_residual = gamma_kernel._scaled_digamma_residual(shape)
        location = np.full(len(response), mean)
        _, _, a_deviance = gamma_kernel._scaled_ratio_terms(
            response, location, shape, derivative_order=0
        )
        multiplier = np.ones(len(response), dtype=np.float64) if semantics == "prior" else weights
        with np.errstate(over="raise", invalid="raise"):
            terms = multiplier * (a_residual + a_deviance)
        target = math.fsum(float(term) for term in terms)
    except (FloatingPointError, OverflowError, ValueError):
        return None
    return target if math.isfinite(target) else None


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_initial_target_evaluates_identical_shapes_once(
    monkeypatch: pytest.MonkeyPatch, semantics: str
) -> None:
    original = gamma_kernel._scaled_digamma_residual
    batch_sizes = []

    def counted(shape: np.ndarray) -> np.ndarray:
        batch_sizes.append(len(shape))
        return original(shape)

    monkeypatch.setattr(gamma_kernel, "_scaled_digamma_residual", counted)
    response = np.linspace(0.125, 8.0, 257)
    weights = np.full(len(response), 2.0) if semantics == "prior" else np.arange(1.0, 258.0)
    target = gamma_kernel._gamma_initial_target(response, 2.0, weights, semantics, -3.0)
    assert target is not None
    assert batch_sizes == [1]


def test_initial_target_does_not_approximate_nearby_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    original = gamma_kernel._scaled_digamma_residual
    batch_sizes = []

    def counted(shape: np.ndarray) -> np.ndarray:
        batch_sizes.append(len(shape))
        return original(shape)

    monkeypatch.setattr(gamma_kernel, "_scaled_digamma_residual", counted)
    weights = np.array([1.0, np.nextafter(1.0, np.inf)])
    target = gamma_kernel._gamma_initial_target(np.array([1.0, 2.0]), 1.5, weights, "prior", 0.0)
    assert target is not None
    assert batch_sizes == [2]


def test_initial_target_reduction_has_no_python_generator_frames() -> None:
    target_code = gamma_kernel._gamma_initial_target.__code__
    calls = 0

    def profile(frame, event, arg):
        nonlocal calls
        if (
            event == "call"
            and frame.f_code.co_name == "<genexpr>"
            and frame.f_back is not None
            and frame.f_back.f_code is target_code
        ):
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        response = np.linspace(0.125, 8.0, 257)
        target = gamma_kernel._gamma_initial_target(
            response, 2.0, np.ones(len(response)), "prior", -3.0
        )
    finally:
        sys.setprofile(previous)
    assert target is not None
    assert calls == 0


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_initial_target_preserves_literal_summands_and_refusals(semantics: str) -> None:
    maximum = np.finfo(np.float64).max
    response = np.array([0.125, 0.25, 1.0, 2.0, 8.0])
    weights = np.array([1.0, 2.0, 3.0, 5.0, 11.0])
    cases = [
        (-3.0, response, 1.0, np.ones(5)),
        (-3.0, response, 1.0, np.full(5, 2.0)),
        (-3.0, response, 1.0, weights),
        (-740.0, np.ones(2), 1.0, np.ones(2)),
        (0.0, np.array([_MIN_FLOAT, maximum]), maximum, np.ones(2)),
        (1000.0, response, 1.0, weights),
        (-1000.0, response, 1.0, weights),
        (0.0, response, 0.0, weights),
        (0.0, response, 1.0, np.zeros(5)),
        (0.0, np.ones(2), 1.0, np.full(2, maximum)),
        (0.0, np.empty(0), 1.0, np.empty(0)),
    ]
    for rho, y, mean, weight in cases:
        expected = _old_initial_target(y, mean, weight, semantics, rho)
        actual = gamma_kernel._gamma_initial_target(y, mean, weight, semantics, rho)
        assert actual == expected


def test_initial_target_keeps_fsum_at_a_cancellation_boundary() -> None:
    response = np.array([1.0, 1.001, 4.0])
    a_residual = gamma_kernel._scaled_digamma_residual(np.ones(1))[0]
    positive_term = a_residual + 3.0 - math.log(4.0)
    weights = np.array([2.0**52, 1.0, round(-(2.0**52) * a_residual / positive_term)])
    expected = _old_initial_target(response, 1.0, weights, "frequency", 0.0)
    actual = gamma_kernel._gamma_initial_target(response, 1.0, weights, "frequency", 0.0)
    assert expected is not None
    assert abs(expected) < 8 * _EPS * 2**52 * abs(a_residual)
    assert actual == expected
    _, _, deviance = gamma_kernel._scaled_ratio_terms(
        response, np.ones(3), np.ones(3), derivative_order=0
    )
    assert expected != float(np.sum(weights * (a_residual + deviance)))
