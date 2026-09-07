"""Structural distribution-function and expected-shortfall family protocols."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import superglm.distributional as distributional
from superglm import SuperLSS
from superglm.distributional import DistributionFunctionFamily, GammaLS, GaussianLS, Predictor
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS, TwoPieceNormalLSS
from superglm.features import Numeric


def _frame(n=120, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    return pd.DataFrame({"x": x}), rng


class _DistributionFunctionsOnly:
    def cdf(self, y, theta):
        return np.zeros(len(theta))

    def quantile(self, p, theta):
        return np.zeros(len(theta))


class _ExpectedShortfallOnly:
    def expected_shortfall(self, p, theta):
        return np.zeros(len(theta))


class _PriorWeightedExpectedShortfallOnly:
    def expected_shortfall_prior_weighted(self, p, theta, weights):
        return np.zeros(len(theta))


def test_protocol_membership_is_structural():
    distribution_families = (
        GaussianLS(),
        GammaLS(),
        TweedieLSS(),
        GeneralizedGammaLSS(),
    )
    assert all(isinstance(family, DistributionFunctionFamily) for family in distribution_families)
    assert all(
        family.capabilities.cdf and family.capabilities.quantile for family in distribution_families
    )
    assert not isinstance(NegativeBinomialLS(), DistributionFunctionFamily)
    # Both two-piece families keep their declared capabilities aligned with
    # their structural protocol implementation.
    for two_piece in (TwoPieceLogNormalLSS(), TwoPieceNormalLSS()):
        assert isinstance(two_piece, DistributionFunctionFamily)
        assert two_piece.capabilities.cdf and two_piece.capabilities.quantile
    # LogNormalLS declares and implements both in each parametrisation.
    for log_normal in (LogNormalLS(), LogNormalLS(parametrisation="location")):
        assert isinstance(log_normal, DistributionFunctionFamily)
        assert log_normal.capabilities.cdf and log_normal.capabilities.quantile


def test_expected_shortfall_protocols_are_independent_and_structural():
    expected_shortfall = getattr(distributional, "ExpectedShortfallFamily", None)
    prior_weighted = getattr(distributional, "PriorWeightedExpectedShortfallFamily", None)
    assert expected_shortfall is not None
    assert prior_weighted is not None

    supported = (
        GaussianLS(),
        GammaLS(),
        LogNormalLS(),
        LogNormalLS(parametrisation="location"),
        GeneralizedParetoLSS(),
        GeneralizedGammaLSS(),
        GeneralizedGammaLSS(parametrisation="location"),
    )
    unsupported = (
        TweedieLSS(),
        TwoPieceNormalLSS(),
        TwoPieceLogNormalLSS(),
        NegativeBinomialLS(),
        _DistributionFunctionsOnly(),
    )
    assert all(isinstance(family, expected_shortfall) for family in supported)
    assert not any(isinstance(family, expected_shortfall) for family in unsupported)
    assert isinstance(_ExpectedShortfallOnly(), expected_shortfall)
    assert not isinstance(_ExpectedShortfallOnly(), DistributionFunctionFamily)

    assert isinstance(GaussianLS(), prior_weighted)
    assert isinstance(GammaLS(), prior_weighted)
    assert not isinstance(LogNormalLS(), prior_weighted)
    assert isinstance(_PriorWeightedExpectedShortfallOnly(), prior_weighted)
    assert not isinstance(_PriorWeightedExpectedShortfallOnly(), expected_shortfall)


def test_gaussian_and_gamma_functionals_match_scipy():
    theta_gaussian = np.array([[1.0, 0.5], [-2.0, 2.0]])
    y = np.array([1.2, -1.0])
    normal = stats.norm(theta_gaussian[:, 0], theta_gaussian[:, 1])
    assert np.allclose(GaussianLS().cdf(y, theta_gaussian), normal.cdf(y), rtol=0, atol=1e-15)
    p = np.array([0.3, 0.95])
    assert np.allclose(GaussianLS().quantile(p, theta_gaussian), normal.ppf(p), rtol=1e-14, atol=0)
    theta_gamma = np.array([[2.0, 0.5], [7.0, 1.2]])  # (mean, cv)
    shape = 1.0 / theta_gamma[:, 1] ** 2
    scale = theta_gamma[:, 0] * theta_gamma[:, 1] ** 2
    gamma = stats.gamma(shape, scale=scale)
    yg = np.array([1.5, 3.0])
    assert np.allclose(GammaLS().cdf(yg, theta_gamma), gamma.cdf(yg), rtol=0, atol=1e-14)
    assert np.allclose(GammaLS().quantile(p, theta_gamma), gamma.ppf(p), rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="strictly inside"):
        GaussianLS().quantile(np.array([0.0, 0.5]), theta_gaussian)
    with pytest.raises(ValueError, match="strictly inside"):
        GammaLS().quantile(np.array([0.5, 1.0]), theta_gamma)


def test_facade_round_trips_quantile_and_cdf_and_refuses_without_the_protocol():
    frame, rng = _frame()
    y = rng.gamma(4.0, np.exp(0.4 * frame["x"].to_numpy()) / 4.0)
    model = SuperLSS(
        family=GammaLS(),
        predictors=(Predictor("mean", {"x": Numeric()}), Predictor("scale", {})),
    ).fit(frame, y)
    quantiles = model.predict_quantile(frame, 0.9)
    assert quantiles.shape == (len(frame),) and not quantiles.flags.writeable
    assert np.allclose(model.predict_cdf(frame, quantiles), 0.9, rtol=0, atol=1e-12)
    per_row = model.predict_quantile(frame, np.linspace(0.05, 0.95, len(frame)))
    assert per_row.shape == (len(frame),)
    counts = rng.poisson(2.0, len(frame)).astype(float)
    nb = SuperLSS(
        family=NegativeBinomialLS(),
        predictors=(Predictor("mean", {"x": Numeric()}), Predictor("theta", {})),
    ).fit(frame, counts)
    with pytest.raises(NotImplementedError, match="distribution function"):
        nb.predict_cdf(frame, 1.0)
    with pytest.raises(NotImplementedError, match="distribution function"):
        nb.predict_quantile(frame, 0.5)
