"""Positive-support CDFs distinguish query support from likelihood support."""

from __future__ import annotations

import math

import numpy as np
import pytest

from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels.generalized_gamma import (
    GeneralizedGammaDomainError,
    generalized_gamma_cdf,
)
from superglm.distributional.kernels.generalized_pareto import (
    GeneralizedParetoDomainError,
    generalized_pareto_cdf,
)
from superglm.distributional.kernels.log_normal import LogNormalDomainError, log_normal_cdf
from superglm.distributional.kernels.two_piece import TwoPieceDomainError
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

_MIXED_THRESHOLD = np.array([-1.0, 0.0, 1.0])
_UNIT_MEDIAN_MEAN = math.exp(0.5)
_FAMILY_CASES = (
    pytest.param(
        GammaLS(),
        np.tile([1.0, 1.0], (3, 1)),
        -math.expm1(-1.0),
        id="gamma",
    ),
    pytest.param(
        LogNormalLS(),
        np.tile([_UNIT_MEDIAN_MEAN, 1.0], (3, 1)),
        0.5,
        id="log-normal-mean",
    ),
    pytest.param(
        LogNormalLS(parametrisation="location"),
        np.tile([0.0, 1.0], (3, 1)),
        0.5,
        id="log-normal-location",
    ),
    pytest.param(
        GeneralizedGammaLSS(),
        np.tile([_UNIT_MEDIAN_MEAN, 1.0, 0.0], (3, 1)),
        0.5,
        id="generalized-gamma-mean",
    ),
    pytest.param(
        GeneralizedGammaLSS(parametrisation="location"),
        np.tile([0.0, 1.0, 0.0], (3, 1)),
        0.5,
        id="generalized-gamma-location",
    ),
    pytest.param(
        GeneralizedParetoLSS(),
        np.tile([1.0, 0.5], (3, 1)),
        5.0 / 9.0,
        id="generalized-pareto",
    ),
    pytest.param(
        TwoPieceLogNormalLSS(),
        np.tile([_UNIT_MEDIAN_MEAN, 1.0, 0.0], (3, 1)),
        0.5,
        id="two-piece-log-normal-mean",
    ),
    pytest.param(
        TwoPieceLogNormalLSS(parametrisation="location"),
        np.tile([0.0, 1.0, 0.0], (3, 1)),
        0.5,
        id="two-piece-log-normal-location",
    ),
)
_EXTREME_OUTSIDE_FAMILY_CASES = (
    pytest.param(
        GammaLS(),
        np.array([[1.0, 1.0e200], [1.0, 1.0]]),
        -math.expm1(-1.0),
        id="gamma",
    ),
    pytest.param(
        LogNormalLS(),
        np.array([[1.0, 1.0e200], [_UNIT_MEDIAN_MEAN, 1.0]]),
        0.5,
        id="log-normal-mean",
    ),
    pytest.param(
        LogNormalLS(parametrisation="location"),
        np.array([[1.0e308, 1.0e200], [0.0, 1.0]]),
        0.5,
        id="log-normal-location",
    ),
    pytest.param(
        GeneralizedGammaLSS(),
        np.array([[1.0, 1.0e200, 1.0e200], [_UNIT_MEDIAN_MEAN, 1.0, 0.0]]),
        0.5,
        id="generalized-gamma-mean",
    ),
    pytest.param(
        GeneralizedGammaLSS(parametrisation="location"),
        np.array([[1.0e308, 1.0e200, 1.0e200], [0.0, 1.0, 0.0]]),
        0.5,
        id="generalized-gamma-location",
    ),
    pytest.param(
        GeneralizedParetoLSS(),
        np.array([[1.0e-200, 0.5], [1.0, 0.5]]),
        5.0 / 9.0,
        id="generalized-pareto",
    ),
    pytest.param(
        TwoPieceLogNormalLSS(),
        np.array([[1.0, 1.0e200, 0.0], [_UNIT_MEDIAN_MEAN, 1.0, 0.0]]),
        0.5,
        id="two-piece-log-normal-mean",
    ),
    pytest.param(
        TwoPieceLogNormalLSS(parametrisation="location"),
        np.array([[1.0e308, 1.0e200, 0.8], [0.0, 1.0, 0.0]]),
        0.5,
        id="two-piece-log-normal-location",
    ),
)
_INVALID_FAMILY_CASES = (
    pytest.param(GammaLS(), np.array([[0.0, 1.0]]), id="gamma"),
    pytest.param(LogNormalLS(), np.array([[0.0, 1.0]]), id="log-normal-mean"),
    pytest.param(
        LogNormalLS(parametrisation="location"),
        np.array([[0.0, 0.0]]),
        id="log-normal-location",
    ),
    pytest.param(
        GeneralizedGammaLSS(),
        np.array([[0.0, 1.0, 0.0]]),
        id="generalized-gamma-mean",
    ),
    pytest.param(
        GeneralizedGammaLSS(parametrisation="location"),
        np.array([[0.0, 0.0, 0.0]]),
        id="generalized-gamma-location",
    ),
    pytest.param(GeneralizedParetoLSS(), np.array([[0.0, 0.5]]), id="generalized-pareto"),
    pytest.param(
        TwoPieceLogNormalLSS(),
        np.array([[0.0, 1.0, 0.0]]),
        id="two-piece-log-normal-mean",
    ),
    pytest.param(
        TwoPieceLogNormalLSS(parametrisation="location"),
        np.array([[0.0, 0.0, 0.0]]),
        id="two-piece-log-normal-location",
    ),
)


def test_log_normal_kernel_masks_mixed_rows_at_the_lower_support_endpoint() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = log_normal_cdf(_MIXED_THRESHOLD, np.zeros(3), np.ones(3))

    np.testing.assert_array_equal(result, [0.0, 0.0, 0.5])
    assert not result.flags.writeable


def test_generalized_gamma_kernel_masks_mixed_rows_at_the_lower_support_endpoint() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = generalized_gamma_cdf(_MIXED_THRESHOLD, np.zeros(3), np.ones(3), np.zeros(3))

    np.testing.assert_array_equal(result, [0.0, 0.0, 0.5])
    assert not result.flags.writeable


def test_generalized_pareto_kernel_preserves_both_support_endpoints() -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = generalized_pareto_cdf(
            np.array([-1.0, 0.0, 0.5, 2.0, 9.0]),
            np.ones(5),
            np.full(5, -0.5),
        )

    np.testing.assert_array_equal(result[[0, 1, 3, 4]], [0.0, 0.0, 1.0, 1.0])
    assert result[2] == pytest.approx(0.4375, rel=0.0, abs=2.0e-15)
    assert not result.flags.writeable


@pytest.mark.parametrize(
    ("cdf", "parameters", "interior_cdf"),
    (
        pytest.param(
            log_normal_cdf,
            (np.array([1.0e308, 0.0]), np.array([1.0e200, 1.0])),
            0.5,
            id="log-normal",
        ),
        pytest.param(
            generalized_gamma_cdf,
            (
                np.array([1.0e308, 0.0]),
                np.array([1.0e200, 1.0]),
                np.array([1.0e200, 0.0]),
            ),
            0.5,
            id="generalized-gamma",
        ),
        pytest.param(
            generalized_pareto_cdf,
            (np.array([1.0e-200, 1.0]), np.array([1.0e200, -0.5])),
            0.75,
            id="generalized-pareto",
        ),
    ),
)
def test_kernels_do_not_evaluate_extreme_parameters_on_outside_support_rows(
    cdf, parameters: tuple[np.ndarray, ...], interior_cdf: float
) -> None:
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = cdf(np.array([-1.0, 1.0]), *parameters)

    np.testing.assert_array_equal(result[:1], [0.0])
    assert result[1] == pytest.approx(interior_cdf, rel=0.0, abs=2.0e-15)
    assert not result.flags.writeable


@pytest.mark.parametrize(("family", "theta", "interior_cdf"), _FAMILY_CASES)
def test_family_adapters_mask_mixed_rows_at_the_lower_support_endpoint(
    family, theta: np.ndarray, interior_cdf: float
) -> None:
    with np.errstate(divide="raise", invalid="raise"):
        result = family.cdf(_MIXED_THRESHOLD, theta)

    np.testing.assert_array_equal(result[:2], [0.0, 0.0])
    assert result[2] == pytest.approx(interior_cdf, rel=0.0, abs=2.0e-15)
    assert not result.flags.writeable


@pytest.mark.parametrize(("family", "theta", "interior_cdf"), _FAMILY_CASES)
def test_family_adapters_broadcast_a_scalar_lower_endpoint(
    family, theta: np.ndarray, interior_cdf: float
) -> None:
    del interior_cdf
    with np.errstate(divide="raise", invalid="raise"):
        result = family.cdf(0.0, theta)

    np.testing.assert_array_equal(result, np.zeros(3))
    assert result.shape == (3,)
    assert not result.flags.writeable


@pytest.mark.parametrize(("family", "theta", "interior_cdf"), _EXTREME_OUTSIDE_FAMILY_CASES)
def test_family_cdf_does_not_derive_extreme_outside_row_parameters(
    family, theta: np.ndarray, interior_cdf: float
) -> None:
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = family.cdf(np.array([-1.0, 1.0]), theta)

    np.testing.assert_array_equal(result[:1], [0.0])
    assert result[1] == pytest.approx(interior_cdf, rel=0.0, abs=2.0e-15)
    assert not result.flags.writeable


def test_gamma_prior_weighted_cdf_masks_only_outside_support_rows() -> None:
    family = GammaLS()
    theta = np.tile([1.0, 1.0], (3, 1))
    weights = np.array([1.0, 2.0, 3.0])

    with np.errstate(divide="raise", invalid="raise"):
        result = family.cdf_prior_weighted(_MIXED_THRESHOLD, theta, weights)

    np.testing.assert_array_equal(result[:2], [0.0, 0.0])
    expected = 1.0 - math.exp(-3.0) * (1.0 + 3.0 + 9.0 / 2.0)
    assert result[2] == pytest.approx(expected, rel=0.0, abs=2.0e-15)
    assert not result.flags.writeable


def test_gamma_prior_cdf_does_not_square_an_extreme_outside_row_cv() -> None:
    family = GammaLS()
    theta = np.array([[1.0, 1.0e200], [1.0, 1.0]])

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = family.cdf_prior_weighted(np.array([-1.0, 1.0]), theta, np.array([2.0, 3.0]))

    np.testing.assert_array_equal(result[:1], [0.0])
    expected = 1.0 - math.exp(-3.0) * (1.0 + 3.0 + 9.0 / 2.0)
    assert result[1] == pytest.approx(expected, rel=0.0, abs=2.0e-15)
    assert not result.flags.writeable


def test_gamma_prior_weights_are_validated_when_every_threshold_is_outside_support() -> None:
    theta = np.tile([1.0, 1.0], (3, 1))

    with pytest.raises(ValueError, match="prior weights must be finite and strictly positive"):
        GammaLS().cdf_prior_weighted(-1.0, theta, np.array([1.0, 0.0, 2.0]))


@pytest.mark.parametrize(("family", "theta"), _INVALID_FAMILY_CASES)
def test_family_cdfs_validate_theta_when_every_threshold_is_outside_support(
    family, theta: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        family.cdf(-1.0, theta)


def test_gamma_prior_cdf_validates_theta_when_every_threshold_is_outside_support() -> None:
    with pytest.raises(ValueError):
        GammaLS().cdf_prior_weighted(-1.0, np.array([[0.0, 1.0]]), np.ones(1))


@pytest.mark.parametrize(
    ("response", "theta"),
    (
        pytest.param(
            -1.0,
            np.array([[2.0, 2.5, -0.5]]),
            id="all-outside-support",
        ),
        pytest.param(
            np.array([-1.0, 1.0]),
            np.array([[2.0, 2.5, -0.5], [_UNIT_MEDIAN_MEAN, 1.0, 0.0]]),
            id="invalid-outside-row-with-valid-interior-row",
        ),
    ),
)
def test_generalized_gamma_mean_cdf_validates_conditional_domain_before_support_mask(
    response, theta: np.ndarray
) -> None:
    with pytest.raises(
        GeneralizedGammaDomainError,
        match="generalized gamma mean does not exist for every row",
    ):
        GeneralizedGammaLSS().cdf(response, theta)


@pytest.mark.parametrize(
    ("cdf", "parameters"),
    (
        pytest.param(
            log_normal_cdf,
            (np.zeros(1), np.zeros(1)),
            id="log-normal",
        ),
        pytest.param(
            generalized_gamma_cdf,
            (np.zeros(1), np.zeros(1), np.zeros(1)),
            id="generalized-gamma",
        ),
        pytest.param(
            generalized_pareto_cdf,
            (np.zeros(1), np.zeros(1)),
            id="generalized-pareto",
        ),
    ),
)
def test_kernel_cdfs_validate_parameters_when_every_query_is_outside_support(
    cdf, parameters: tuple[np.ndarray, ...]
) -> None:
    with pytest.raises(ValueError):
        cdf(np.array([-1.0]), *parameters)


@pytest.mark.parametrize(
    ("family", "response"),
    (
        pytest.param(GammaLS(), np.array([1.0, 0.0]), id="gamma"),
        pytest.param(LogNormalLS(), np.array([1.0, 0.0]), id="log-normal"),
        pytest.param(GeneralizedGammaLSS(), np.array([1.0, 0.0]), id="generalized-gamma"),
        pytest.param(GeneralizedParetoLSS(), np.array([0.0, -1.0]), id="generalized-pareto"),
        pytest.param(TwoPieceLogNormalLSS(), np.array([1.0, 0.0]), id="two-piece-log-normal"),
    ),
)
def test_cdf_support_does_not_relax_likelihood_response_validation(family, response) -> None:
    weights = resolve_likelihood_weights(
        np.ones(2), n_observations=2, contract=WeightContract(semantics="frequency")
    )

    with pytest.raises(ValueError):
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)


def test_positive_support_kernels_keep_rejecting_nonfinite_thresholds() -> None:
    gamma = GammaLS()
    gamma_theta = np.array([[1.0, 0.5]])
    for threshold in (np.nan, -np.inf, np.inf):
        with pytest.raises(ValueError, match="finite"):
            gamma.cdf(threshold, gamma_theta)
        with pytest.raises(ValueError, match="finite"):
            gamma.cdf_prior_weighted(threshold, gamma_theta, np.ones(1))

    with pytest.raises(LogNormalDomainError):
        log_normal_cdf(np.array([np.nan]), np.zeros(1), np.ones(1))
    with pytest.raises(GeneralizedGammaDomainError):
        generalized_gamma_cdf(np.array([np.inf]), np.zeros(1), np.ones(1), np.zeros(1))
    with pytest.raises(GeneralizedParetoDomainError):
        generalized_pareto_cdf(np.array([-np.inf]), np.ones(1), np.zeros(1))
    with pytest.raises(TwoPieceDomainError):
        TwoPieceLogNormalLSS(parametrisation="location").cdf(np.nan, np.array([[0.0, 1.0, 0.0]]))


def test_tweedie_cdf_preserves_its_atom_at_zero() -> None:
    result = TweedieLSS().cdf(np.array([0.0]), np.array([[1.0, 1.0, 1.5]]))

    assert result[0] == pytest.approx(math.exp(-2.0), rel=1.0e-14)
    assert result[0] > 0.0
