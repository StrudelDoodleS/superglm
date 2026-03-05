"""Tests for Link protocol and LogLink implementation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from superglm.distributions import Gamma, Poisson, Tweedie
from superglm.links import Link, LogLink, resolve_link


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
            resolve_link("identity", Poisson())
