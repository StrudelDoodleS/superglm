"""Tests for Link protocol and all link implementations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from superglm.distributions import Binomial, Gamma, Gaussian, Poisson, Tweedie
from superglm.links import (
    CauchitLink,
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    Link,
    LogitLink,
    LogLink,
    NegativeBinomialLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
    resolve_link,
)


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

    def test_none_uses_gaussian_default(self):
        link = resolve_link(None, Gaussian())
        assert isinstance(link, IdentityLink)

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


# ── Generic link property tests ──────────────────────────────────────


def _numerical_deriv(f, x, h=1e-7):
    """Central finite-difference derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)


# Links with their valid mu/eta ranges for testing
_UNIT_LINKS = [
    (IdentityLink(), np.array([0.5, 1.0, 2.0, 5.0]), np.array([-1.0, 0.0, 1.0, 3.0])),
    (LogLink(), np.array([0.5, 1.0, 2.0, 10.0]), np.array([-1.0, 0.0, 1.0, 3.0])),
    (LogitLink(), np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-2.0, -0.5, 0.0, 0.5, 2.0])),
    (ProbitLink(), np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-1.5, -0.5, 0.0, 0.5, 1.5])),
    (CloglogLink(), np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-2.0, -1.0, -0.3, 0.0, 0.8])),
    (CauchitLink(), np.array([0.1, 0.3, 0.5, 0.7, 0.9]), np.array([-3.0, -0.7, 0.0, 0.7, 3.0])),
    (InverseLink(), np.array([0.5, 1.0, 2.0, 5.0]), np.array([0.2, 0.5, 1.0, 2.0])),
    (InverseSquaredLink(), np.array([0.5, 1.0, 2.0, 5.0]), np.array([0.04, 0.25, 1.0, 4.0])),
    (SqrtLink(), np.array([0.25, 1.0, 4.0, 9.0]), np.array([0.5, 1.0, 2.0, 3.0])),
    (PowerLink(power=2.0), np.array([0.5, 1.0, 2.0, 3.0]), np.array([0.25, 1.0, 4.0, 9.0])),
    (PowerLink(power=-1.0), np.array([0.5, 1.0, 2.0, 5.0]), np.array([0.2, 0.5, 1.0, 2.0])),
    (
        NegativeBinomialLink(theta=2.0),
        np.array([0.5, 1.0, 3.0, 10.0]),
        np.array([-1.4, -0.7, -0.3, -0.1]),
    ),
]


class TestAllLinksRoundtrip:
    @pytest.mark.parametrize(
        "link,mu,_eta", _UNIT_LINKS, ids=lambda x: type(x).__name__ if hasattr(x, "link") else ""
    )
    def test_inverse_of_link(self, link, mu, _eta):
        assert_allclose(link.inverse(link.link(mu)), mu, atol=1e-10)

    @pytest.mark.parametrize(
        "link,_mu,eta", _UNIT_LINKS, ids=lambda x: type(x).__name__ if hasattr(x, "link") else ""
    )
    def test_link_of_inverse(self, link, _mu, eta):
        assert_allclose(link.link(link.inverse(eta)), eta, atol=1e-10)


class TestAllLinksDerivatives:
    @pytest.mark.parametrize(
        "link,mu,_eta", _UNIT_LINKS, ids=lambda x: type(x).__name__ if hasattr(x, "link") else ""
    )
    def test_deriv_matches_numerical(self, link, mu, _eta):
        analytical = link.deriv(mu)
        numerical = _numerical_deriv(link.link, mu)
        assert_allclose(analytical, numerical, rtol=1e-5)

    @pytest.mark.parametrize(
        "link,_mu,eta", _UNIT_LINKS, ids=lambda x: type(x).__name__ if hasattr(x, "link") else ""
    )
    def test_deriv_inverse_matches_numerical(self, link, _mu, eta):
        analytical = link.deriv_inverse(eta)
        numerical = _numerical_deriv(link.inverse, eta)
        assert_allclose(analytical, numerical, rtol=1e-5)

    @pytest.mark.parametrize(
        "link,_mu,eta", _UNIT_LINKS, ids=lambda x: type(x).__name__ if hasattr(x, "link") else ""
    )
    def test_deriv_and_deriv_inverse_reciprocal(self, link, _mu, eta):
        mu = link.inverse(eta)
        product = link.deriv(mu) * link.deriv_inverse(eta)
        assert_allclose(product, 1.0, atol=1e-10)

    @pytest.mark.parametrize(
        "link,_mu,eta", _UNIT_LINKS, ids=lambda x: type(x).__name__ if hasattr(x, "link") else ""
    )
    def test_deriv2_inverse_matches_numerical(self, link, _mu, eta):
        analytical = link.deriv2_inverse(eta)
        numerical = _numerical_deriv(link.deriv_inverse, eta)
        assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)


class TestAllLinksProtocol:
    @pytest.mark.parametrize(
        "link,_mu,_eta", _UNIT_LINKS, ids=lambda x: type(x).__name__ if hasattr(x, "link") else ""
    )
    def test_satisfies_protocol(self, link, _mu, _eta):
        assert isinstance(link, Link)


# ── Individual link edge cases ───────────────────────────────────────


class TestIdentityLink:
    def test_link_is_copy(self):
        link = IdentityLink()
        mu = np.array([1.0, 2.0])
        eta = link.link(mu)
        assert_allclose(eta, mu)
        mu[0] = 999.0
        assert eta[0] != 999.0  # should be a copy

    def test_deriv2_inverse_is_zero(self):
        link = IdentityLink()
        eta = np.array([-1.0, 0.0, 1.0])
        assert_allclose(link.deriv2_inverse(eta), 0.0)


class TestProbitLink:
    def test_symmetry(self):
        link = ProbitLink()
        mu = np.array([0.3, 0.7])
        eta = link.link(mu)
        assert_allclose(eta[0], -eta[1], atol=1e-12)

    def test_midpoint(self):
        link = ProbitLink()
        assert_allclose(link.link(np.array([0.5])), [0.0], atol=1e-12)
        assert_allclose(link.inverse(np.array([0.0])), [0.5], atol=1e-12)


class TestCloglogLink:
    def test_asymmetric(self):
        """cloglog is not symmetric around 0.5 like logit."""
        link = CloglogLink()
        assert link.link(np.array([0.5]))[0] != 0.0

    def test_extreme_mu_stable(self):
        link = CloglogLink()
        mu = np.array([1e-10, 1 - 1e-10])
        eta = link.link(mu)
        assert np.all(np.isfinite(eta))


class TestInverseLink:
    def test_positive_eta(self):
        link = InverseLink()
        mu = np.array([0.5, 1.0, 2.0])
        eta = link.link(mu)
        assert_allclose(eta, [2.0, 1.0, 0.5])


class TestSqrtLink:
    def test_values(self):
        link = SqrtLink()
        mu = np.array([4.0, 9.0, 16.0])
        assert_allclose(link.link(mu), [2.0, 3.0, 4.0])
        assert_allclose(link.inverse(np.array([2.0, 3.0, 4.0])), mu)


class TestPowerLink:
    def test_power_zero_raises(self):
        with pytest.raises(ValueError, match="log link"):
            PowerLink(power=0)

    def test_power_one_is_identity(self):
        link = PowerLink(power=1.0)
        mu = np.array([0.5, 1.0, 3.0])
        assert_allclose(link.link(mu), mu)
        assert_allclose(link.inverse(mu), mu)


class TestNegativeBinomialLink:
    def test_theta_positive_required(self):
        with pytest.raises(ValueError, match="theta must be > 0"):
            NegativeBinomialLink(theta=0)
        with pytest.raises(ValueError, match="theta must be > 0"):
            NegativeBinomialLink(theta=-1)

    def test_eta_always_negative(self):
        link = NegativeBinomialLink(theta=1.0)
        mu = np.array([0.1, 1.0, 10.0, 100.0])
        eta = link.link(mu)
        assert np.all(eta < 0)


class TestResolveLinkShortcuts:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("identity", IdentityLink),
            ("probit", ProbitLink),
            ("cloglog", CloglogLink),
            ("cauchit", CauchitLink),
            ("inverse", InverseLink),
            ("inverse_squared", InverseSquaredLink),
            ("sqrt", SqrtLink),
        ],
    )
    def test_shortcut_resolves(self, name, cls):
        link = resolve_link(name, Poisson())
        assert isinstance(link, cls)
