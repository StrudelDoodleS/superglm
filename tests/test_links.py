"""Tests for Link protocol and LogLink/LogitLink implementations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from superglm.distributions import Binomial, Gamma, Poisson, Tweedie
from superglm.links import Link, LogitLink, LogLink, resolve_link


class TestLogLink:
    def test_inverse_of_link(self):
        link = LogLink()
        mu = np.array([0.5, 1.0, 2.0, 10.0])
        assert_allclose(link.inverse(link.link(mu)), mu)

    def test_link_of_inverse(self):
        link = LogLink()
        eta = np.array([-1.0, 0.0, 1.0, 3.0])
        assert_allclose(link.link(link.inverse(eta)), eta)

    def test_deriv(self):
        link = LogLink()
        mu = np.array([0.5, 1.0, 2.0, 10.0])
        assert_allclose(link.deriv(mu), 1.0 / mu)

    def test_deriv_inverse(self):
        link = LogLink()
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        assert_allclose(link.deriv_inverse(eta), np.exp(eta))

    def test_deriv_and_deriv_inverse_are_reciprocal(self):
        link = LogLink()
        eta = np.array([-1.0, 0.0, 1.0, 2.0])
        mu = link.inverse(eta)
        assert_allclose(link.deriv(mu) * link.deriv_inverse(eta), 1.0)

    def test_satisfies_protocol(self):
        assert isinstance(LogLink(), Link)


class TestResolveLink:
    def test_from_string(self):
        link = resolve_link("log", Poisson())
        assert isinstance(link, LogLink)

    def test_none_uses_default(self):
        link = resolve_link(None, Poisson())
        assert isinstance(link, LogLink)

    def test_none_uses_gamma_default(self):
        link = resolve_link(None, Gamma())
        assert isinstance(link, LogLink)

    def test_none_uses_tweedie_default(self):
        link = resolve_link(None, Tweedie(p=1.5))
        assert isinstance(link, LogLink)

    def test_passthrough(self):
        original = LogLink()
        link = resolve_link(original, Poisson())
        assert link is original

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown link"):
            resolve_link("not_a_link", Poisson())

    def test_none_uses_binomial_default(self):
        link = resolve_link(None, Binomial())
        assert isinstance(link, LogitLink)

    def test_logit_from_string(self):
        link = resolve_link("logit", Binomial())
        assert isinstance(link, LogitLink)


class TestLogitLink:
    def test_inverse_of_link(self):
        link = LogitLink()
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert_allclose(link.inverse(link.link(mu)), mu, atol=1e-12)

    def test_link_of_inverse(self):
        link = LogitLink()
        eta = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert_allclose(link.link(link.inverse(eta)), eta, atol=1e-12)

    def test_deriv(self):
        link = LogitLink()
        mu = np.array([0.2, 0.5, 0.8])
        expected = 1.0 / (mu * (1 - mu))
        assert_allclose(link.deriv(mu), expected)

    def test_deriv_inverse(self):
        link = LogitLink()
        eta = np.array([-2.0, 0.0, 2.0])
        from scipy.special import expit

        p = expit(eta)
        expected = p * (1 - p)
        assert_allclose(link.deriv_inverse(eta), expected)

    def test_deriv_and_deriv_inverse_are_reciprocal(self):
        link = LogitLink()
        eta = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        mu = link.inverse(eta)
        assert_allclose(link.deriv(mu) * link.deriv_inverse(eta), 1.0, atol=1e-12)

    def test_deriv2_inverse(self):
        link = LogitLink()
        eta = np.array([-2.0, 0.0, 2.0])
        from scipy.special import expit

        p = expit(eta)
        expected = p * (1 - p) * (1 - 2 * p)
        assert_allclose(link.deriv2_inverse(eta), expected)

    def test_satisfies_protocol(self):
        assert isinstance(LogitLink(), Link)

    def test_extreme_eta_stable(self):
        """LogitLink should not overflow on extreme eta values."""
        link = LogitLink()
        eta = np.array([-500.0, -100.0, 100.0, 500.0])
        mu = link.inverse(eta)
        assert np.all(np.isfinite(mu))
        assert np.all(mu >= 0) and np.all(mu <= 1)
