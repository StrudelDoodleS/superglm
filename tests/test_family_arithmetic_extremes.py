"""Analytic regressions for finite built-in family arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributional.families import (
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    TweedieLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.distributions import (
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    initial_mean,
    weighted_log_likelihood,
)
from tests._generalized_gamma_lss_oracles import mp_derivatives, mp_log_density

_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2.0
_MIN_SUBNORMAL = np.nextafter(0.0, 1.0)
_FIRST_ORDERS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_SECOND_ORDERS = ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))


def _assert_forward(actual, expected, operations, *, scale=None):
    """A gamma_n rounding allowance plus gradual-underflow rounding."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    magnitude = np.abs(expected) if scale is None else np.asarray(scale, dtype=np.float64)
    gamma = operations * _UNIT_ROUNDOFF / (1.0 - operations * _UNIT_ROUNDOFF)
    allowance = gamma * magnitude + operations * _MIN_SUBNORMAL
    assert np.all(np.isfinite(actual))
    assert np.all(np.abs(actual - expected) <= allowance), (actual, expected, allowance)


def _evaluate(family, y, theta, *, weight=1.0, semantics="frequency", order=2):
    response = np.array([y], dtype=np.float64)
    weights = resolve_likelihood_weights(
        np.array([weight], dtype=np.float64),
        n_observations=1,
        contract=WeightContract(semantics),
    )
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    return family.evaluate_natural(
        response, np.array([theta], dtype=np.float64), plan, derivative_order=order
    )


@pytest.mark.parametrize("q", [-1e-8, -5e-9, -1e-9, 1e-9, 5e-9, 1e-8])
def test_generalized_gamma_small_nonzero_shape_preserves_density_and_its_derivatives(q):
    mp = pytest.importorskip("mpmath")
    point = (-20.0, 0.02, q)
    evaluated = _evaluate(GeneralizedGammaLSS(parametrisation="location"), 1.0, point)
    assert evaluated.valid[0]
    with mp.workdps(100):
        expected_value = float(mp_log_density(1.0, *point) + mp.log(2 * mp.pi) / 2)
    expected_score = mp_derivatives(
        lambda mu, sigma, shape: mp_log_density(1.0, mu, sigma, shape),
        point,
        _FIRST_ORDERS,
        dps=100,
    )
    expected_hessian = mp_derivatives(
        lambda mu, sigma, shape: mp_log_density(1.0, mu, sigma, shape),
        point,
        _SECOND_ORDERS,
        dps=100,
    )
    # A 25-term Horner evaluation costs at most 48 multiply/add roundings.
    # For |Q*w| <= 1e-5 its absolute coefficient sum differs from the
    # result by <1.0001; 32 more operations cover forming/scaling each
    # channel and perturbing its well-conditioned w,sigma,Q arguments.
    operations = 2 * 25 + 32
    _assert_forward(evaluated.optimizing_log_likelihood[0], expected_value, operations)
    _assert_forward(evaluated.score[0], expected_score, operations)
    _assert_forward(evaluated.hessian_packed[0], expected_hessian, operations)


@pytest.mark.parametrize("q", [-1e-80, 1e-80])
def test_generalized_gamma_shape_size_does_not_replace_the_represented_product(q):
    mp = pytest.importorskip("mpmath")
    point = (-1e70, 1.0, q)
    evaluated = _evaluate(GeneralizedGammaLSS(parametrisation="location"), 1.0, point)
    assert evaluated.valid[0]
    # Q*w is about 1e-10; all second derivatives remain representable.
    # 160 decimal digits also cover the cancellation in the independent
    # uncentered Prentice density, whose Gamma shape is about 1e160.
    with mp.workdps(160):
        expected_value = float(mp_log_density(1.0, *point) + mp.log(2 * mp.pi) / 2)
    with mp.workdps(160):
        scales = tuple(mp.mpf(abs(value)) for value in point)
        unit_point = tuple(mp.mpf(value) / scale for value, scale in zip(point, scales))

        def density_at_relative_coordinates(*coordinates):
            parameters = tuple(value * scale for value, scale in zip(coordinates, scales))
            return mp_log_density(1.0, *parameters)

        def derivatives(orders):
            return [
                float(
                    mp.diff(density_at_relative_coordinates, unit_point, order)
                    / mp.fprod(scale**degree for scale, degree in zip(scales, order))
                )
                for order in orders
            ]

        # Relative oracle coordinates prevent an absolute step in mu from
        # subtracting O(1e162) values to recover an O(1) second derivative.
        # Divide the resulting exact coordinate factors in high precision.
        expected_score = derivatives(_FIRST_ORDERS)
        expected_hessian = derivatives(_SECOND_ORDERS)
    for actual, expected in (
        (evaluated.optimizing_log_likelihood[0], expected_value),
        (evaluated.score[0], expected_score),
        (evaluated.hessian_packed[0], expected_hessian),
    ):
        _assert_forward(actual, expected, 2 * 25 + 32)


@pytest.mark.parametrize("q", [-1e-100, -1e-9, 0.0, 1e-9, 1e-100])
def test_generalized_gamma_retains_the_finite_shape_normalizer(q):
    mp = pytest.importorskip("mpmath")
    evaluated = _evaluate(GeneralizedGammaLSS(parametrisation="location"), 1.0, [0.0, 1.0, q])
    with mp.workdps(100):
        shape = mp.mpf(q)
        expected = float(-(shape**2 / 12 - shape**6 / 360))
    # The next Stirling term is |Q|^10/1260, below 1e-93 here;
    # q*q/12 needs two arithmetic roundings. Retain a six-operation
    # allowance for the surrounding density additions and coefficient.
    _assert_forward(evaluated.optimizing_log_likelihood[0], expected, 6)
    if q:
        assert evaluated.optimizing_log_likelihood[0] < 0.0
    else:
        assert evaluated.optimizing_log_likelihood[0] == 0.0
        _assert_forward(evaluated.score[0], [0.0, -1.0, 0.0], 6)
        _assert_forward(evaluated.hessian_packed[0], [-1.0, 0.0, 0.0, 1.0, 0.0, -1.0 / 6], 6)


@pytest.mark.parametrize("count", [0.0, 1.0, 3.0, 12.0, 31.0])
@pytest.mark.parametrize("theta", [1e16, 1e100, 1e300])
@pytest.mark.parametrize("semantics", ["frequency", "prior"])
def test_scalar_negative_binomial_finite_theta_density_preserves_the_poisson_limit(
    count, theta, semantics
):
    mp = pytest.importorskip("mpmath")
    mean, weight = 1.25, 2.0
    family = NegativeBinomial(theta=theta)
    actual = weighted_log_likelihood(
        family,
        np.array([count]),
        np.array([mean]),
        np.array([weight]),
        weight_semantics=semantics,
    )
    with mp.workdps(100):
        y, mu, size, w = map(mp.mpf, (count, mean, theta, weight))
        if semantics == "prior":
            y, mu, size = w * y, w * mu, w * size
            multiplier = mp.mpf(1)
        else:
            multiplier = w
        # The exact finite-theta rising factorial keeps this oracle
        # independent of a rounded-away Gamma argument increment.
        rising = mp.fsum(mp.log1p(mp.mpf(j) / size) for j in range(int(y)))
        expected = float(
            multiplier
            * (rising - mp.loggamma(y + 1) + y * mp.log(mu) - (size + y) * mp.log1p(mu / size))
        )
    # Each finite rising-factor term needs division, log1p and addition;
    # the density's remaining logarithms/products need fewer than 32 ops.
    _assert_forward(actual, expected, 3 * int(max(count, 2 * count)) + 32)


@pytest.mark.parametrize("family", [Poisson(), Gamma()])
def test_scalar_deviance_preserves_the_positive_near_mean_remainder(family):
    mp = pytest.importorskip("mpmath")
    y, mu = 100000001.0, 100000000.0
    with mp.workdps(80):
        response, mean = mp.mpf(y), mp.mpf(mu)
        ratio = response / mean
        expected = float(
            2 * (response * mp.log(ratio) - response + mean)
            if isinstance(family, Poisson)
            else 2 * (ratio - 1 - mp.log(ratio))
        )
    actual = family.deviance_unit(np.array([y]), np.array([mu]))[0]
    assert actual > 0.0
    # At |delta|=1e-8, the absolute coefficient condition number is
    # below 1.0001. Twenty-four series terms and their scaling need
    # fewer than 4*25 arithmetic operations.
    _assert_forward(actual, expected, 4 * 25)


@pytest.mark.parametrize("semantics", ["frequency", "prior"])
@pytest.mark.parametrize(
    ("family", "y", "mu", "phi"),
    [
        (Poisson(), 1e15, 1e15, 1.0),
        (Gamma(), 1e30, 1e30, 1e-290),
        (Gaussian(), 0.0, 0.0, 1e308),
        (Gaussian(), 1e155, 0.0, 1e308),
    ],
)
def test_scalar_likelihood_finite_normalizers_and_residuals(family, y, mu, phi, semantics):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(340):
        response, mean, dispersion = map(mp.mpf, (y, mu, phi))
        if isinstance(family, Poisson):
            expected = response * mp.log(mean) - mean - mp.loggamma(response + 1)
        elif isinstance(family, Gamma):
            shape = 1 / dispersion
            expected = (
                shape * mp.log(shape * response / mean)
                - shape * response / mean
                - mp.log(response)
                - mp.loggamma(shape)
            )
        else:
            expected = (
                -(mp.log(2 * mp.pi) + mp.log(dispersion) + (response - mean) ** 2 / dispersion) / 2
            )
    actual = weighted_log_likelihood(
        family,
        np.array([y]),
        np.array([mu]),
        np.ones(1),
        phi,
        weight_semantics=semantics,
    )
    # Stable scalar expressions have at most 32 arithmetic/transcendental
    # evaluations at these non-cancelling finite-density fixtures.
    _assert_forward(actual, float(expected), 32)


def test_initial_mean_does_not_materialize_overflowing_weighted_responses():
    y = np.array([1e9, 1e9 + 1.0])
    weights = np.full(2, 1e300)
    expected = 1000000000.5
    # The exact normalized dot has two products, one sum and one division.
    _assert_forward(initial_mean(y, weights, Gaussian()), expected, 8)
    _assert_forward(initial_mean(y, np.ones(2), Gaussian()), expected, 8)


@pytest.mark.parametrize("semantics", ["frequency", "prior"])
def test_gaussian_natural_channels_avoid_intermediate_residual_and_scale_powers(semantics):
    mp = pytest.importorskip("mpmath")
    value = 1e155
    evaluated = _evaluate(GaussianLS(), value, [0.0, value], semantics=semantics)
    with mp.workdps(80):
        sigma = mp.mpf(value)
        expected_value = float(-mp.log(sigma) - mp.log(2 * mp.pi) / 2 - mp.mpf("0.5"))
        score = [float(1 / sigma), 0.0]
        hessian = [float(-1 / sigma**2), float(-2 / sigma**2), float(-2 / sigma**2)]
    assert evaluated.valid[0]
    _assert_forward(evaluated.optimizing_log_likelihood[0], expected_value, 24)
    _assert_forward(evaluated.score[0], score, 24)
    _assert_forward(evaluated.hessian_packed[0], hessian, 24)
    assert np.all(evaluated.hessian_packed[0] < 0.0)


def test_two_piece_normal_keeps_representable_subnormal_natural_hessian():
    mp = pytest.importorskip("mpmath")
    value = 1e155
    evaluated = _evaluate(TwoPieceNormalLSS(), value, [0.0, value, 0.0])
    with mp.workdps(80):
        sigma = mp.mpf(value)
        expected = [
            float(-1 / sigma**2),
            float(-2 / sigma**2),
            float(-2 / sigma),
            float(-2 / sigma**2),
            float(-2 / sigma),
            -3.0,
        ]
    assert evaluated.valid[0]
    _assert_forward(evaluated.hessian_packed[0], expected, 32)
    assert np.all(evaluated.hessian_packed[0] < 0.0)


@pytest.mark.parametrize(("y", "scale"), [(1e300, 1e-30), (1e200, 1e-20)])
def test_generalized_pareto_positive_tail_has_finite_requested_channels(y, scale):
    mp = pytest.importorskip("mpmath")
    evaluated = _evaluate(GeneralizedParetoLSS(), y, [scale, 0.5])
    with mp.workdps(100):
        response, psi, xi = mp.mpf(y), mp.mpf(scale), mp.mpf("0.5")
        t = response / psi
        z = xi * t
        log_support = mp.log1p(z)
        v = t / (1 + z)
        expected_value = float(-mp.log(psi) - (1 + 1 / xi) * log_support)
        expected_score = [
            float((-1 + (1 + xi) * v) / psi),
            float(log_support / xi**2 - (1 + 1 / xi) * v),
        ]
        expected_hessian = [
            float((1 - (1 + xi) * v * (2 + z) / (1 + z)) / psi**2),
            float((v - (1 + xi) * v**2) / psi),
            float(-2 * log_support / xi**3 + 2 * v / xi**2 + (1 + 1 / xi) * v**2),
        ]
    assert evaluated.valid[0]
    _assert_forward(evaluated.optimizing_log_likelihood[0], expected_value, 48)
    _assert_forward(evaluated.score[0], expected_score, 48)
    _assert_forward(evaluated.hessian_packed[0], expected_hessian, 48)


def test_tweedie_atom_retains_finite_dispersion_hessian_after_prior_scaling():
    mp = pytest.importorskip("mpmath")
    phi, weight = 1e155, 1e200
    evaluated = _evaluate(TweedieLSS(), 0.0, [1.0, phi, 1.5], weight=weight, semantics="prior")
    with mp.workdps(80):
        dispersion, mass = mp.mpf(phi), mp.mpf(weight)
        rate = 2 * mass / dispersion
        expected = float(-2 * rate / dispersion**2)
    assert evaluated.valid[0]
    assert evaluated.hessian_packed[0, 3] < 0.0
    # The zero-atom path uses log/exp before the two natural-scale divisions;
    # their relative argument error is bounded by gamma_16 times |log rate|.
    _assert_forward(evaluated.hessian_packed[0, 3], expected, 16 * 128)


def test_tweedie_positive_row_retains_the_same_finite_dispersion_hessian():
    mp = pytest.importorskip("mpmath")
    phi = 1e155
    evaluated = _evaluate(TweedieLSS(), 1.0, [1.0, phi, 1.5], weight=phi, semantics="prior")
    with mp.workdps(100):
        # At p=3/2 each compound-Poisson jump is exponential. Summing
        # lambda^n/(n! Gamma(n)) gives the Bessel-I1 density below.
        # Relative dispersion coordinates keep oracle differentiation
        # independent of the large modeled dispersion.
        def log_density(relative_phi):
            rate, jump_scale = 2 / relative_phi, relative_phi / 2
            return (
                -rate
                - 1 / jump_scale
                + mp.log(rate / jump_scale) / 2
                + mp.log(mp.besseli(1, 2 * mp.sqrt(rate / jump_scale)))
            )

        expected = float(mp.diff(log_density, mp.mpf(1), 2) / mp.mpf(phi) ** 2)
    assert evaluated.valid[0]
    assert evaluated.hessian_packed[0, 3] != 0.0
    _assert_forward(evaluated.hessian_packed[0, 3], expected, 16 * 128)


@pytest.mark.parametrize("family", [Poisson(), Gamma(), Gaussian(), NegativeBinomial(5.0)])
@pytest.mark.parametrize("semantics", ["frequency", "prior"])
def test_scalar_repaired_densities_retain_ordinary_weight_laws(family, semantics):
    mp = pytest.importorskip("mpmath")
    y, mu, weight, phi = 3.0, 2.0, 2.0, 0.7
    with mp.workdps(80):
        response, mean, mass, dispersion = map(mp.mpf, (y, mu, weight, phi))
        multiplier = mass if semantics == "frequency" else mp.mpf(1)
        if isinstance(family, Gaussian):
            variance = dispersion if semantics == "frequency" else dispersion / mass
            density = -(mp.log(2 * mp.pi * variance) + (response - mean) ** 2 / variance) / 2
        elif isinstance(family, Gamma):
            shape = (1 if semantics == "frequency" else mass) / dispersion
            density = (
                shape * mp.log(shape / mean)
                + (shape - 1) * mp.log(response)
                - shape * response / mean
                - mp.loggamma(shape)
            )
        else:
            if semantics == "prior":
                response, mean = mass * response, mass * mean
            if isinstance(family, Poisson):
                density = response * mp.log(mean) - mean - mp.loggamma(response + 1)
            else:
                size = mp.mpf(5) * (mass if semantics == "prior" else 1)
                density = (
                    mp.loggamma(size + response)
                    - mp.loggamma(size)
                    - mp.loggamma(response + 1)
                    + size * mp.log(size / (size + mean))
                    + response * mp.log(mean / (size + mean))
                )
        expected = float(multiplier * density)
    actual = weighted_log_likelihood(
        family,
        np.array([y]),
        np.array([mu]),
        np.array([weight]),
        phi,
        weight_semantics=semantics,
    )
    _assert_forward(actual, expected, 64)


@pytest.mark.parametrize("family", [Poisson(), Gamma(), Gaussian(), NegativeBinomial(5.0)])
@pytest.mark.parametrize("semantics", ["frequency", "prior"])
def test_zero_mass_rows_still_contribute_zero_to_repaired_scalar_densities(family, semantics):
    kwargs = {"weight_semantics": semantics}
    expected = weighted_log_likelihood(
        family, np.array([1.0]), np.array([1.0]), np.ones(1), **kwargs
    )
    actual = weighted_log_likelihood(
        family, np.array([1.0, 1e300]), np.ones(2), np.array([1.0, 0.0]), **kwargs
    )
    _assert_forward(actual, expected, 4)


@pytest.mark.parametrize(("theta", "mu"), [(0.5, 2.0), (5.0, 2.0), (5.0, 10.0)])
@pytest.mark.parametrize("semantics", ["frequency", "prior"])
def test_negative_binomial_keeps_ordinary_fractional_count_interpolation(theta, mu, semantics):
    mp = pytest.importorskip("mpmath")
    counts = np.array([0.25, 2.5, 64.0, 65.0])
    weights = np.array([1.0, 2.0, 1.0, 3.0])
    with mp.workdps(80):
        expected_rows = []
        for count, weight in zip(counts, weights):
            response, mean, size, mass = map(mp.mpf, (count, mu, theta, weight))
            multiplier = mass
            if semantics == "prior":
                response, mean, size = mass * response, mass * mean, mass * size
                multiplier = 1
            expected_rows.append(
                multiplier
                * (
                    mp.loggamma(size + response)
                    - mp.loggamma(size)
                    - mp.loggamma(response + 1)
                    + size * mp.log(size / (size + mean))
                    + response * mp.log(mean / (size + mean))
                )
            )
        expected = float(mp.fsum(expected_rows))
    actual = weighted_log_likelihood(
        NegativeBinomial(theta), counts, np.full(4, mu), weights, weight_semantics=semantics
    )
    _assert_forward(actual, expected, 4 * 32)


@pytest.mark.parametrize(
    ("response", "weights", "expected"),
    [([0.25, 0.75], [1e308, 1e308], 0.5), ([2.0**-600], [2.0**-600], 2.0**-600)],
)
def test_initial_mean_detects_finite_results_from_nonrepresentable_intermediates(
    response, weights, expected
):
    _assert_forward(initial_mean(np.array(response), np.array(weights), Gaussian()), expected, 8)


def test_initial_mean_exceptional_sum_preserves_weight_response_exponent_pairing():
    response = np.array([1e308, 0.0, 1e-300])
    weights = np.array([1e-320, 1e308, 1e-320])
    # The leading numerator term divided by the leading denominator is
    # exactly weights[0]. The remaining rational correction is <2^-3000,
    # far below half a subnormal ulp (2^-1075), so rounding is unambiguous.
    expected = weights[0]
    assert initial_mean(response, weights, Gaussian()) == expected


@pytest.mark.parametrize(
    ("y", "weight", "phi", "semantics"),
    [(1e200, 1e-200, 1.0, "frequency"), (1e160, 1e-320, 1e-300, "prior")],
)
def test_gaussian_weighted_density_does_not_materialize_an_unweighted_quadratic(
    y, weight, phi, semantics
):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(340):
        response, mass, dispersion = map(mp.mpf, (y, weight, phi))
        quadratic = mass * response**2 / dispersion
        normalizer = mp.log(2 * mp.pi) + mp.log(dispersion)
        if semantics == "frequency":
            normalizer *= mass
        else:
            normalizer -= mp.log(mp.mpf(max(weight, 1e-300)))
        expected = float(-(normalizer + quadratic) / 2)
    actual = weighted_log_likelihood(
        Gaussian(),
        np.array([y]),
        np.zeros(1),
        np.array([weight]),
        phi,
        weight_semantics=semantics,
    )
    # Binary exponent composition needs at most six mantissa operations,
    # plus the mean subtraction, normalizer and final density operations.
    _assert_forward(actual, expected, 32)


@pytest.mark.parametrize("semantics", ["frequency", "prior"])
def test_gaussian_true_unrepresentable_quadratic_keeps_negative_infinite_density(semantics):
    # The exact residual square is 2^1400, beyond binary64's exponent range.
    actual = weighted_log_likelihood(
        Gaussian(),
        np.array([2.0**700]),
        np.zeros(1),
        np.ones(1),
        weight_semantics=semantics,
    )
    assert actual == -np.inf


@pytest.mark.parametrize("family", [GaussianLS(), TwoPieceNormalLSS()])
def test_frequency_mass_rescues_location_scale_hessian_before_unit_underflow(family):
    scale, weight = 2.0**550, 2.0**50
    theta = [0.0, scale] if isinstance(family, GaussianLS) else [0.0, scale, 0.0]
    evaluated = _evaluate(family, scale, theta, weight=weight)
    expected = [-(2.0**-1050), -(2.0**-1049), -(2.0**-1049)]
    if isinstance(family, TwoPieceNormalLSS):
        expected = [
            expected[0],
            expected[1],
            -(2.0**-499),
            expected[2],
            -(2.0**-499),
            -3.0 * weight,
        ]
    assert evaluated.valid[0]
    _assert_forward(evaluated.hessian_packed[0], expected, 32)


def test_generalized_pareto_frequency_mass_rescues_tail_scale_hessian():
    from fractions import Fraction

    scale, weight = 2.0**550, 2.0**50
    evaluated = _evaluate(GeneralizedParetoLSS(), scale, [scale, 0.5], weight=weight)
    expected = float(Fraction(-2, 3) / (1 << 1050))
    assert evaluated.valid[0]
    _assert_forward(evaluated.hessian_packed[0, 0], expected, 48)


def test_tweedie_natural_hessian_does_not_round_a_subnormal_scale_square():
    from fractions import Fraction

    phi, weight = 3.0 * 2.0**-538, _MIN_SUBNORMAL
    evaluated = _evaluate(TweedieLSS(), 0.0, [1.0, phi, 1.5], weight=weight, semantics="prior")
    expected = float(-4 * Fraction.from_float(weight) / Fraction.from_float(phi) ** 3)
    assert evaluated.valid[0]
    # Forming log(phi)-log(weight) and exponentiating propagates the
    # absolute log-argument error; |log input|<1024 on this fixture.
    _assert_forward(evaluated.hessian_packed[0, 3], expected, 4 * 1024)


def test_tweedie_natural_hessian_scales_terms_before_an_overflowing_difference():
    mp = pytest.importorskip("mpmath")
    power, phi = 1.1, 1.5
    mean = float(np.exp(1.0 / (2.0 - power)))
    weight = 0.385 * np.finfo(np.float64).max
    with mp.workdps(100):
        mu, dispersion, p, mass = map(mp.mpf, (mean, phi, power, weight))
        rate = mass * mu ** (2 - p) / (dispersion * (2 - p))
        expected = float(-2 * rate / dispersion**2)
    evaluated = _evaluate(TweedieLSS(), 0.0, [mean, phi, power], weight=weight, semantics="prior")
    assert evaluated.valid[0]
    _assert_forward(evaluated.hessian_packed[0, 3], expected, 4 * 1024)


def test_weighted_channel_retains_ordinary_multiplication_and_zero_factors(monkeypatch):
    from superglm.distributional.kernels import _weighted

    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary and exact-zero channels must not use scalar recovery")

    monkeypatch.setattr(_weighted, "_binary_product_divide", forbidden)
    unit = np.array([0.25, -1.1, 0.0])
    mass = np.array([3.0, 5.0, 7.0])
    result = _weighted.weighted_natural_channel(unit, mass, (unit,))
    np.testing.assert_array_equal(result, mass * unit)
    # A zero numerator divided by an underflowed denominator square can
    # produce a NaN candidate even though the exact channel is zero.
    zero = _weighted.weighted_natural_channel(
        np.array([np.nan]), np.array([3.0]), (0.0,), (1e-200, 1e-200)
    )
    np.testing.assert_array_equal(zero, np.zeros(1))


def test_weighted_channel_recovers_a_rounded_subnormal_unit_value():
    from fractions import Fraction

    from superglm.distributional.kernels._weighted import weighted_natural_channel

    factor, mass = 3.0 * 2.0**-538, 2.0**52
    unit = factor * factor  # rounds 2.25 minimum subnormals to two
    exact = Fraction.from_float(mass) * Fraction.from_float(factor) ** 2
    expected = float(exact)
    assert mass * unit != expected
    actual = weighted_natural_channel(np.array([unit]), np.array([mass]), (factor, factor))[0]
    _assert_forward(actual, expected, 8)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_weighted_channel_keeps_true_overflow_and_rounds_final_underflow(sign):
    from superglm.distributional.kernels._weighted import weighted_natural_channel

    overflow = weighted_natural_channel(
        np.array([sign * np.inf]), np.ones(1), (sign, 2.0**700, 2.0**700)
    )
    assert overflow[0] == sign * np.inf
    underflow = weighted_natural_channel(
        np.array([sign * 0.0]), np.ones(1), (sign, 2.0**-700, 2.0**-700)
    )
    assert underflow[0] == 0.0
    assert np.signbit(underflow[0]) == (sign < 0.0)


@pytest.mark.parametrize("operand", ["multiplier", "numerator", "denominator"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_weighted_channel_keeps_nan_factors_independent_of_their_sign_bit(operand, sign):
    from superglm.distributional.kernels._weighted import weighted_natural_channel

    values = {name: np.array([1.0]) for name in ("multiplier", "numerator", "denominator")}
    values[operand][0] = np.copysign(np.nan, sign)
    actual = weighted_natural_channel(
        np.array([np.nan]), values["multiplier"], (values["numerator"],), (values["denominator"],)
    )
    assert np.isnan(actual[0])
