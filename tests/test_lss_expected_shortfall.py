"""Certified family-owned upper expected-shortfall functionals."""

from __future__ import annotations

import warnings
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import special

from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.generalized_gamma import (
    GeneralizedGammaDomainError,
    GeneralizedGammaLSS,
)
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.families.two_piece import (
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.posterior import resolve_quantity

_EPS = np.finfo(np.float64).eps
_DECIMAL_PI = Decimal("3.1415926535897932384626433832795028841971693993751")


def _gamma_shape_plus_one_tail_ratio_oracle(shape: float, p: float) -> float:
    """Q(shape+1, x)/(1-p) via an independent high-precision recurrence."""
    survival = 1.0 - p
    threshold = special.gammainccinv(shape, survival)
    with localcontext() as context:
        context.prec = 80
        a = Decimal.from_float(shape)
        x = Decimal.from_float(float(threshold))
        s = Decimal.from_float(survival)
        t = x / a - 1
        deviance = t - (1 + t).ln()
        inverse = 1 / a
        stirling_remainder = inverse / 12 - inverse**3 / 360 + inverse**5 / 1260 - inverse**7 / 1680
        log_increment = -a * deviance - (2 * _DECIMAL_PI * a).ln() / 2 - stirling_remainder
        return float(1 + log_increment.exp() / s)


def _generalized_gamma_tail_fraction(p: float, sigma: float, shape: float) -> float:
    """Upper first-moment share divided by upper probability, from the gamma law."""
    survival = 1.0 - p
    if shape == 0.0:
        return float(special.ndtr(sigma - special.ndtri(p)) / survival)
    kappa = 1.0 / (shape * shape)
    shifted_shape = kappa + sigma / shape
    if shifted_shape <= 0.0:
        return np.inf
    if shape > 0.0:
        threshold = special.gammainccinv(kappa, survival)
        moment_share = special.gammaincc(shifted_shape, threshold)
    else:
        threshold = special.gammaincinv(kappa, survival)
        moment_share = special.gammainc(shifted_shape, threshold)
    return float(moment_share / survival)


def _generalized_gamma_location_mean(mu: float, sigma: float, shape: float) -> float:
    if shape == 0.0:
        return float(np.exp(mu + 0.5 * sigma * sigma))
    kappa = 1.0 / (shape * shape)
    shifted_shape = kappa + sigma / shape
    if shifted_shape <= 0.0:
        return np.inf
    log_loading = (
        (sigma / shape) * np.log(shape * shape)
        + special.gammaln(shifted_shape)
        - special.gammaln(kappa)
    )
    return float(np.exp(mu + log_loading))


def test_gaussian_expected_shortfall_is_exact_at_the_last_probability_below_one():
    family = GaussianLS()
    p = np.nextafter(1.0, 0.0)
    theta = np.array([[1.25, 0.75]])

    z = special.ndtri(p)
    want = theta[0, 0] + theta[0, 1] * np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * (1.0 - p))
    got = family.expected_shortfall(np.array([p]), theta)

    assert got[0] == pytest.approx(want, rel=16.0 * _EPS)
    assert not got.flags.writeable


def test_gaussian_prior_weighted_expected_shortfall_uses_the_scaled_row_law():
    family = GaussianLS()
    p = np.array([0.9, 0.995])
    theta = np.array([[1.0, 2.0], [-0.5, 0.75]])
    weights = np.array([0.25, 4.0])
    scale = theta[:, 1] / np.sqrt(weights)
    z = special.ndtri(p)
    want = theta[:, 0] + scale * np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * (1.0 - p))

    np.testing.assert_allclose(
        family.expected_shortfall_prior_weighted(p, theta, weights),
        want,
        rtol=16.0 * _EPS,
        atol=0.0,
    )


def test_gamma_expected_shortfall_uses_the_exact_upper_gamma_recurrence():
    family = GammaLS()
    p = np.array([0.9, np.nextafter(1.0, 0.0)])
    theta = np.array([[2.0, 0.5], [7.0, 1.2]])
    shape = 1.0 / theta[:, 1] ** 2
    survival = 1.0 - p
    x = special.gammainccinv(shape, survival)
    log_increment = shape * np.log(x) - x - special.gammaln(shape + 1.0)
    want = theta[:, 0] * (1.0 + np.exp(log_increment) / survival)

    np.testing.assert_allclose(family.expected_shortfall(p, theta), want, rtol=64.0 * _EPS)


def test_gamma_expected_shortfall_preserves_a_subnormal_exponential_tail():
    family = GammaLS()
    p = np.nextafter(1.0, 0.0)
    mean = 1.0e-310
    theta = np.array([[mean, 1.0]])
    log_survival = np.log1p(-p)
    # Shape one is exponential, so memorylessness gives ES = q_p + mean.
    want = mean * (1.0 - log_survival)

    got = family.expected_shortfall(np.array([p]), theta)[0]
    quantile = family.quantile(np.array([p]), theta)[0]

    assert got >= quantile
    assert got == pytest.approx(want, rel=64.0 * _EPS, abs=0.0)


def test_gamma_expected_shortfall_refuses_an_unresolved_zero_inverse():
    family = GammaLS()
    p = 0.9
    theta = np.array([[1.0, 100.0]])
    shape = 1.0 / theta[0, 1] ** 2
    threshold = special.gammainccinv(shape, 1.0 - p)

    assert threshold == 0.0
    assert special.gammaincc(shape, threshold) == 1.0
    assert 1.0 - p != 1.0

    with pytest.raises(ValueError, match="expected shortfall cannot be certified"):
        family.expected_shortfall(np.array([p]), theta)


def test_gamma_prior_weighted_expected_shortfall_refuses_an_unresolved_zero_inverse():
    family = GammaLS()
    p = 0.9
    theta = np.array([[1.0, 1.0]])
    weights = np.array([1.0e-4])
    shape = weights[0] / theta[0, 1] ** 2
    threshold = special.gammainccinv(shape, 1.0 - p)

    assert threshold == 0.0
    assert special.gammaincc(shape, threshold) == 1.0
    assert 1.0 - p != 1.0

    with pytest.raises(ValueError, match="expected shortfall cannot be certified"):
        family.expected_shortfall_prior_weighted(np.array([p]), theta, weights)


def test_gamma_prior_weighted_expected_shortfall_uses_the_weighted_shape():
    family = GammaLS()
    p = np.array([0.9, 0.995])
    theta = np.array([[2.0, 0.5], [7.0, 1.2]])
    weights = np.array([0.25, 3.0])
    shape = weights / theta[:, 1] ** 2
    survival = 1.0 - p
    x = special.gammainccinv(shape, survival)
    want = theta[:, 0] * special.gammaincc(shape + 1.0, x) / survival

    np.testing.assert_allclose(
        family.expected_shortfall_prior_weighted(p, theta, weights),
        want,
        rtol=16.0 * _EPS,
    )


def test_gamma_expected_shortfall_refuses_an_uncertifiable_weighted_shape():
    family = GammaLS()
    theta = np.array([[2.0, 0.5]])
    p = np.array([0.99])
    weights = np.array([1.0e32])
    assert family.quantile_prior_weighted(p, theta, weights)[0] >= theta[0, 0]

    with pytest.raises(ValueError, match="expected shortfall cannot be certified"):
        family.expected_shortfall_prior_weighted(p, theta, weights)


@pytest.mark.parametrize(
    "weight",
    [
        2_499_999_999_999_999.0,
        2_499_999_999_999_999.5,
        2_500_000_000_000_000.0,
        2_500_000_000_000_000.5,
    ],
)
def test_gamma_expected_shortfall_uses_the_exact_recurrence_across_float_parity(weight):
    family = GammaLS()
    p = 0.9
    theta = np.array([[1.0, 0.5]])
    weights = np.array([weight])
    shape = float(weight / theta[0, 1] ** 2)
    want = _gamma_shape_plus_one_tail_ratio_oracle(shape, p)

    got = family.expected_shortfall_prior_weighted(np.array([p]), theta, weights)[0]
    quantile = family.quantile_prior_weighted(np.array([p]), theta, weights)[0]

    assert got >= quantile
    assert got == pytest.approx(want, rel=16.0 * _EPS)


@pytest.mark.parametrize(
    ("parametrisation", "theta", "mean"),
    [
        ("mean", np.array([[2.5, 0.8]]), 2.5),
        ("location", np.array([[0.2, 0.8]]), float(np.exp(0.2 + 0.5 * 0.8**2))),
    ],
)
def test_log_normal_expected_shortfall_uses_its_stable_normal_tail(parametrisation, theta, mean):
    family = LogNormalLS(parametrisation=parametrisation)
    p = 0.999
    z = special.ndtri(p)
    want = mean * special.ndtr(theta[0, 1] - z) / (1.0 - p)

    got = family.expected_shortfall(np.array([p]), theta)[0]

    assert got == pytest.approx(want, rel=16.0 * _EPS)


@pytest.mark.parametrize(
    ("family", "theta"),
    [
        (LogNormalLS(parametrisation="mean"), np.array([[2.0, 1.0e8]])),
        (GeneralizedGammaLSS(parametrisation="mean"), np.array([[2.0, 1.0e8, 0.0]])),
    ],
)
def test_mean_parametrisations_use_the_supplied_mean_without_location_cancellation(family, theta):
    p = 0.9
    # At this scale Phi(sigma-z_p) rounds to one, so the exact row-law answer
    # is the supplied mean divided by the represented upper probability.
    want = theta[0, 0] / (1.0 - p)

    got = family.expected_shortfall(np.array([p]), theta)[0]

    assert got == pytest.approx(want, rel=16.0 * _EPS)


@pytest.mark.parametrize(
    ("family", "theta"),
    [
        (LogNormalLS(parametrisation="mean", scale_floor=0.0), np.array([[1.0, 1.0e-16]])),
        (LogNormalLS(parametrisation="location", scale_floor=0.0), np.array([[0.0, 1.0e-16]])),
        (
            GeneralizedGammaLSS(parametrisation="mean", scale_floor=0.0),
            np.array([[1.0, 1.0e-16, 0.0]]),
        ),
        (
            GeneralizedGammaLSS(parametrisation="location", scale_floor=0.0),
            np.array([[0.0, 1.0e-16, 0.0]]),
        ),
    ],
)
def test_log_normal_limits_preserve_tail_dominance_when_the_gap_rounds_to_zero(family, theta):
    p = np.nextafter(1.0, 0.0)
    sigma = theta[0, 1]
    z = special.ndtri(p)
    # The inverse-Mills bound makes the positive mathematical log(ES/q) gap
    # smaller than one float64 ulp for this fixture.
    assert sigma / z < _EPS

    got = family.expected_shortfall(np.array([p]), theta)[0]
    quantile = family.quantile(np.array([p]), theta)[0]

    assert got >= quantile
    assert got == quantile


def test_generalized_pareto_expected_shortfall_stays_certified_near_shape_one():
    family = GeneralizedParetoLSS()
    p = 0.99
    scale = 2.0
    shape = 1.0 - 1.0e-8
    theta = np.array([[scale, shape]])
    quantile = scale * np.expm1(-shape * np.log1p(-p)) / shape
    want = (quantile + scale) / (1.0 - shape)

    got = family.expected_shortfall(np.array([p]), theta)[0]

    assert got == pytest.approx(want, rel=128.0 * _EPS)


def test_generalized_gamma_expected_shortfall_covers_both_gamma_tails_and_zero_shape():
    family = GeneralizedGammaLSS(parametrisation="location")
    p = 0.99
    theta = np.array(
        [
            [0.2, 0.8, 0.7],
            [-0.3, 0.6, 0.0],
            [0.1, 0.8, -0.5],
        ]
    )
    want = np.array(
        [
            _generalized_gamma_location_mean(mu, sigma, shape)
            * _generalized_gamma_tail_fraction(p, sigma, shape)
            for mu, sigma, shape in theta
        ]
    )

    np.testing.assert_allclose(
        family.expected_shortfall(np.full(len(theta), p), theta),
        want,
        rtol=1024.0 * _EPS,
    )


def test_generalized_gamma_mean_parametrisation_preserves_the_given_mean():
    family = GeneralizedGammaLSS(parametrisation="mean")
    p = 0.99
    theta = np.array(
        [
            [2.5, 0.8, 0.7],
            [2.5, 0.6, 0.0],
            [2.5, 0.8, -0.5],
        ]
    )
    want = theta[:, 0] * np.array(
        [_generalized_gamma_tail_fraction(p, sigma, shape) for _, sigma, shape in theta]
    )

    np.testing.assert_allclose(
        family.expected_shortfall(np.full(len(theta), p), theta),
        want,
        rtol=1024.0 * _EPS,
    )


def test_generalized_gamma_returns_positive_infinity_when_the_upper_first_moment_diverges():
    family = GeneralizedGammaLSS(parametrisation="location")
    theta = np.array([[0.1, 1.5, -0.8]])

    got = family.expected_shortfall(np.array([0.9]), theta)

    assert np.isposinf(got[0])
    assert np.isposinf(resolve_quantity(family, ("expected_shortfall", 0.9))(theta)[0])


def test_generalized_gamma_checks_moment_existence_before_the_zero_shape_limit():
    family = GeneralizedGammaLSS(parametrisation="location")
    theta = np.array([[-4.5e16, 3.0e8, -5.0e-9]])
    assert theta[0, 1] * abs(theta[0, 2]) > 1.0

    got = family.expected_shortfall(np.array([0.9]), theta)

    assert np.isposinf(got[0])


def test_generalized_gamma_refuses_an_uncertified_extreme_negative_near_zero_tail():
    family = GeneralizedGammaLSS(parametrisation="location")
    p = np.nextafter(1.0, 0.0)
    theta = np.array([[0.2, 0.8, -1.0e-6]])
    quantile = family.quantile(np.array([p]), theta)[0]
    assert quantile == pytest.approx(763.0580930009103, rel=64.0 * _EPS)

    with pytest.raises(GeneralizedGammaDomainError, match="expected shortfall cannot be certified"):
        family.expected_shortfall(np.array([p]), theta)


@pytest.mark.parametrize("shape", [1.0e-8, np.nextafter(1.0e-8, np.inf)])
def test_generalized_gamma_unit_shift_uses_the_exact_recurrence_across_float_parity(shape):
    family = GeneralizedGammaLSS(parametrisation="mean", scale_floor=0.0)
    p = 0.9
    theta = np.array([[1.0, shape, shape]])
    kappa = float(1.0 / (shape * shape))
    want = _gamma_shape_plus_one_tail_ratio_oracle(kappa, p)

    got = family.expected_shortfall(np.array([p]), theta)[0]
    quantile = family.quantile(np.array([p]), theta)[0]

    assert got >= quantile
    assert got == pytest.approx(want, rel=16.0 * _EPS)


def test_generalized_gamma_refuses_an_unrepresented_generic_shape_increment():
    family = GeneralizedGammaLSS(parametrisation="mean", scale_floor=0.0)
    shape = 1.0e-8
    sigma = 1.5e-8
    kappa = 1.0 / (shape * shape)
    shift = sigma / shape
    assert (kappa + shift) - kappa != shift

    with pytest.raises(
        GeneralizedGammaDomainError,
        match="shifted gamma shape increment is unresolved",
    ):
        family.expected_shortfall(
            np.array([0.9]),
            np.array([[1.0, sigma, shape]]),
        )


def test_generalized_gamma_far_negative_tail_uses_a_resolved_log_quantile_ratio():
    family = GeneralizedGammaLSS(parametrisation="location")
    p = np.nextafter(1.0, 0.0)
    theta = np.array([[0.0, 0.5, -1.2]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = family.expected_shortfall(np.array([p]), theta)[0]
    quantile = family.quantile(np.array([p]), theta)[0]

    assert quantile == pytest.approx(3_404_344_115.451713, rel=64.0 * _EPS)
    assert got == pytest.approx(8_510_860_288.629291, rel=64.0 * _EPS)
    assert got >= quantile


@pytest.mark.parametrize(
    ("parametrisation", "theta"),
    [
        ("mean", np.array([[1.0, 0.8, -0.5]])),
        ("mean", np.array([[1.0, 0.8, 0.5]])),
        ("mean", np.array([[1.0, 0.5, 2.0]])),
        ("location", np.array([[0.0, 0.8, -0.5]])),
        ("location", np.array([[0.0, 0.8, 0.5]])),
        ("location", np.array([[0.0, 0.5, 2.0]])),
    ],
)
def test_generalized_gamma_survival_one_boundary_rounds_expected_shortfall_to_mean(
    parametrisation, theta
):
    family = GeneralizedGammaLSS(parametrisation=parametrisation)
    p = np.nextafter(0.0, 1.0)
    expected_mean = family.default_prediction(theta)[0]
    if parametrisation == "location":
        assert expected_mean == pytest.approx(
            _generalized_gamma_location_mean(*theta[0]), rel=16.0 * _EPS
        )

    got = family.expected_shortfall(np.array([p]), theta)[0]
    quantile = family.quantile(np.array([p]), theta)[0]

    if theta[0, 2] == 2.0:
        assert quantile == 0.0
    else:
        assert quantile > 0.0
    assert got == expected_mean
    assert got >= quantile


@pytest.mark.parametrize("shape", [-2.0e-8, -5.0e-9, 0.0, 5.0e-9, 2.0e-8])
def test_generalized_gamma_tail_dominates_its_quantile_across_the_zero_shape_threshold(shape):
    family = GeneralizedGammaLSS(parametrisation="location")
    p = 0.9
    mu = 0.2
    sigma = 0.8
    theta = np.array([[mu, sigma, shape]])
    lognormal_limit = (
        np.exp(mu + 0.5 * sigma * sigma) * special.ndtr(sigma - special.ndtri(p)) / (1.0 - p)
    )

    got = family.expected_shortfall(np.array([p]), theta)[0]
    quantile = family.quantile(np.array([p]), theta)[0]

    assert got >= quantile
    assert got == pytest.approx(
        lognormal_limit,
        rel=64.0 * max(abs(shape), _EPS),
    )


@pytest.mark.parametrize(
    "family",
    [TweedieLSS(), TwoPieceNormalLSS(), TwoPieceLogNormalLSS(), NegativeBinomialLS()],
)
def test_uncertified_families_refuse_expected_shortfall(family):
    with pytest.raises(NotImplementedError, match="expected shortfall"):
        resolve_quantity(family, ("expected_shortfall", 0.9))


def test_unit_prior_weights_take_the_bit_identical_unit_expected_shortfall_path():
    cases = (
        (GaussianLS(), np.array([[1.0, 2.0], [-0.5, 0.75]])),
        (GammaLS(), np.array([[2.0, 0.5], [7.0, 1.2]])),
    )
    for family, theta in cases:
        unit = resolve_quantity(family, ("expected_shortfall", 0.995))(theta)
        weighted = resolve_quantity(
            family, ("expected_shortfall", 0.995), weights=np.ones(len(theta))
        )(theta)
        assert np.array_equal(weighted, unit), type(family).__name__


def test_non_unit_weights_refuse_an_unweighted_only_expected_shortfall_family():
    family = LogNormalLS()
    with pytest.raises(NotImplementedError, match="prior-weighted expected shortfall"):
        resolve_quantity(
            family,
            ("expected_shortfall", 0.9),
            weights=np.array([0.5]),
        )


@pytest.mark.parametrize(
    ("family", "theta"),
    [
        (GaussianLS(), np.array([[0.0, 1.0]])),
        (GammaLS(), np.array([[1.0, 0.5]])),
        (LogNormalLS(), np.array([[1.0, 0.5]])),
        (GeneralizedParetoLSS(), np.array([[1.0, 0.5]])),
        (GeneralizedGammaLSS(), np.array([[1.0, 0.5, 0.2]])),
    ],
)
def test_family_expected_shortfall_requires_an_interior_probability(family, theta):
    with pytest.raises(ValueError, match="strictly inside"):
        family.expected_shortfall(np.array([1.0]), theta)
