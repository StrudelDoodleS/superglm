"""Tests for Link protocol and all link implementations."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from superglm import Numeric, SuperGLM
from superglm.distributions import Binomial, Gamma, Gaussian, Poisson, Tweedie, clip_mu
from superglm.links import (
    _LINK_SHORTCUTS,
    _STABILIZE_ETA_BY_LINK_TYPE,
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
    stabilize_eta,
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


def _adjacent_float64_values(start, toward, count=4096):
    """Return a consecutive float64 neighborhood, including both endpoints."""
    values = np.empty(count, dtype=np.float64)
    values[0] = start
    for index in range(1, count):
        values[index] = np.nextafter(values[index - 1], toward)
    return values


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
    (
        PowerLink(power=1.0),
        np.array([-3.0, -1.0, 0.0, 2.0]),
        np.array([-3.0, -1.0, 0.0, 2.0]),
    ),
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
        mu = np.array([-3.0, -0.5, 0.0, 3.0])
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


def test_stabilize_eta_dispatch_covers_every_registered_shortcut():
    assert set(_LINK_SHORTCUTS.values()) <= set(_STABILIZE_ETA_BY_LINK_TYPE)


@pytest.mark.parametrize(
    "link",
    [
        IdentityLink(),
        LogLink(),
        LogitLink(),
        ProbitLink(),
        CloglogLink(),
        CauchitLink(),
        InverseLink(),
        InverseSquaredLink(),
        SqrtLink(),
        PowerLink(1.0),
        PowerLink(2.0),
        NegativeBinomialLink(theta=2.0),
    ],
    ids=lambda link: (
        type(link).__name__ + (f"-{link.power:g}" if isinstance(link, PowerLink) else "")
    ),
)
def test_every_named_stabilization_path_dispatches_to_its_registered_handler(link):
    eta = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    handler = _STABILIZE_ETA_BY_LINK_TYPE[type(link)]

    assert_allclose(stabilize_eta(eta, link), handler(eta, link))


def test_sqrt_and_cauchit_use_link_specific_eta_ranges():
    sqrt_eta = np.array([-50.0, -1.0, 20.0, 50.0])
    cauchit_eta = np.array([-100.0, 0.0, 100.0])

    assert_allclose(stabilize_eta(sqrt_eta, SqrtLink()), sqrt_eta)
    assert_allclose(stabilize_eta(cauchit_eta, CauchitLink()), cauchit_eta)


def test_cauchit_stabilization_maps_infinite_eta_to_achievable_probability_resolution():
    link = CauchitLink()
    stabilized = stabilize_eta(np.array([-np.inf, np.inf]), link)
    probability = link.inverse(stabilized)

    assert np.all(np.isfinite(stabilized))
    assert np.all(np.abs(stabilized) > 20.0)
    assert 0.0 < probability[0] <= 1e-15
    assert 1.0 - 1e-15 <= probability[1] < 1.0
    assert_allclose(probability[0], 1.0 - probability[1], atol=np.finfo(float).eps)


def test_sqrt_stabilization_preserves_negative_branch_and_only_bounds_overflow():
    link = SqrtLink()
    eta = np.array([-np.inf, -25.0, -2.0, 0.0, 2.0, 25.0, np.inf])
    stabilized = stabilize_eta(eta, link)
    mu = link.inverse(stabilized)

    assert stabilized[1] == -25.0
    assert stabilized[2] == -2.0
    assert stabilized[-2] == 25.0
    assert np.all(np.isfinite(stabilized))
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(link.deriv_inverse(stabilized) ** 2))
    assert_allclose(mu, mu[::-1])


def test_sqrt_stabilization_preserves_signed_zero_and_predictors_past_the_binary_band():
    """Regression guard for a fix that already shipped, in commit 67b90f8.

    67b90f8 gave SqrtLink its own ``stabilize_eta`` entry (``_sqrt_eta``, band
    +/-6.7e153).  Before it, SqrtLink matched no branch and inherited the
    catch-all ``np.clip(eta, -20.0, 20.0)``.  So this cannot fail against the
    branch parent 37a1c18 -- 67b90f8 is its ancestor.  To break it, restore the
    old dispatch: map ``SqrtLink`` to ``_binary_eta`` in
    ``_STABILIZE_ETA_BY_LINK_TYPE``; the +/-50 and +/-1e150 entries then come
    back as +/-20.
    """
    link = SqrtLink()
    # The subnormal and signed-zero entries guard the +/-6.7e153 branch; the
    # +/-50 and +/-1e150 entries are what make this discriminate, since the
    # binary-link band this link used to inherit clipped all four to +/-20.
    eta = np.array([-1.0e150, -50.0, -1.0e-150, -0.0, 0.0, 1.0e-150, 50.0, 1.0e150])
    stabilized = stabilize_eta(eta, link)

    np.testing.assert_array_equal(stabilized, eta)
    assert np.signbit(stabilized[3])
    assert not np.signbit(stabilized[4])
    assert np.all(np.isfinite(link.deriv_inverse(stabilized)))
    assert np.all(np.isfinite(link.inverse(stabilized)))


def test_sqrt_stabilization_round_trips_means_from_subnormal_to_past_the_old_cap():
    """Regression guard for a fix that already shipped, in commit 67b90f8.

    Same shipped fix as the test above, seen from the mean side: an eta cap of
    20 put a hard ceiling of mu = 400 on any sqrt-link fit.  It cannot fail
    against the branch parent 37a1c18, since 67b90f8 is its ancestor.  To break
    it, map ``SqrtLink`` to ``_binary_eta`` in ``_STABILIZE_ETA_BY_LINK_TYPE``;
    mu = 1e6 then round-trips back as 400.
    """
    link = SqrtLink()
    # 5e2 and 1e6 lie above the mu = 400 ceiling that an eta cap of 20 imposed.
    means = np.array([np.nextafter(0.0, 1.0), 1.0e-300, 1.0e-30, 1.0e-16, 5.0e2, 1.0e6])
    eta = stabilize_eta(link.link(means), link)

    recovered = link.inverse(eta)
    np.testing.assert_array_equal(recovered[:3], means[:3])
    np.testing.assert_array_equal(recovered[-1], means[-1])
    np.testing.assert_allclose(recovered, means, rtol=2.0e-16, atol=0.0)


def test_sqrt_derivative_is_reciprocal_through_subnormal_positive_means():
    link = SqrtLink()
    means = np.array(
        [
            np.nextafter(0.0, 1.0),
            np.float64("1e-320"),
            np.float64("1e-300"),
            np.float64("1e-30"),
        ]
    )
    eta = link.link(means)

    assert np.all(np.isfinite(link.deriv(means)))
    assert_allclose(
        link.deriv(means) * link.deriv_inverse(eta),
        np.ones_like(means),
        rtol=2e-15,
        atol=0.0,
    )


def test_sqrt_derivative_reports_zero_and_out_of_domain_limits():
    derivative = SqrtLink().deriv(np.array([0.0, -0.0, -1.0]))

    assert np.all(np.isposinf(derivative[:2]))
    assert np.isnan(derivative[2])


def test_cauchit_stabilization_round_trips_at_probability_resolution_boundary():
    link = CauchitLink()
    stabilized = stabilize_eta(np.array([-np.inf, np.inf]), link)
    mu = link.inverse(stabilized)

    assert_allclose(link.link(mu), stabilized, rtol=2e-15, atol=0.0)
    assert_allclose(
        link.deriv(mu) * link.deriv_inverse(stabilized),
        np.ones(2),
        rtol=2e-15,
        atol=0.0,
    )


def test_cauchit_tail_formulas_cover_consecutive_endpoint_floats():
    link = CauchitLink()
    probabilities = np.concatenate(
        [
            _adjacent_float64_values(1e-15, np.inf),
            _adjacent_float64_values(1.0 - 1e-15, -np.inf),
        ]
    )
    eta = link.link(probabilities)
    recovered = link.inverse(eta)

    np.testing.assert_array_max_ulp(recovered, probabilities, maxulp=1)
    assert_allclose(link.link(recovered), eta, rtol=2e-15, atol=0.0)
    assert_allclose(
        link.deriv(probabilities) * link.deriv_inverse(eta),
        np.ones_like(probabilities),
        rtol=2e-15,
        atol=0.0,
    )


def test_cauchit_tail_branches_preserve_symmetry():
    lower = np.ldexp(np.ones(48), -np.arange(2, 50))
    upper = 1.0 - lower
    link = CauchitLink()
    lower_eta = link.link(lower)
    upper_eta = link.link(upper)

    np.testing.assert_array_equal(lower_eta, -upper_eta)
    np.testing.assert_array_max_ulp(
        link.inverse(lower_eta),
        1.0 - link.inverse(upper_eta),
        maxulp=1,
    )
    np.testing.assert_array_equal(link.deriv(lower), link.deriv(upper))
    np.testing.assert_array_equal(
        link.deriv_inverse(lower_eta),
        link.deriv_inverse(upper_eta),
    )


def test_cloglog_stabilization_avoids_inverse_and_derivative_saturation():
    link = CloglogLink()
    stabilized = stabilize_eta(np.array([-np.inf, np.inf]), link)
    probability = link.inverse(stabilized)
    derivative = link.deriv_inverse(stabilized)

    assert np.all(np.isfinite(stabilized))
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert np.all(np.isfinite(derivative) & (derivative > 0.0))


def test_cloglog_stable_forms_cover_consecutive_endpoint_floats():
    link = CloglogLink()
    probabilities = np.concatenate(
        [
            _adjacent_float64_values(1e-15, np.inf),
            _adjacent_float64_values(2.06115362456e-9, np.inf),
            _adjacent_float64_values(1.0 - 1e-6, -np.inf),
            _adjacent_float64_values(1.0 - 1e-15, -np.inf),
        ]
    )
    eta = link.link(probabilities)
    recovered = link.inverse(eta)

    assert_allclose(recovered, probabilities, rtol=8e-15, atol=0.0)
    assert_allclose(link.link(recovered), eta, rtol=2e-15, atol=0.0)
    assert_allclose(
        link.deriv(probabilities) * link.deriv_inverse(eta),
        np.ones_like(probabilities),
        rtol=2e-14,
        atol=0.0,
    )


def test_cloglog_stabilization_cap_stays_strictly_inside_binomial_clip_mu():
    link = CloglogLink()
    cap = stabilize_eta(np.array([np.inf]), link)
    mu = link.inverse(cap)
    tail = 1.0 - mu[0]

    # ``clip_mu`` clamps binomial means to [1e-7, 1 - 1e-7].  A cap that lands
    # on (or past) that boundary lets clip_mu rewrite mu behind stabilize_eta's
    # back: eta and mu stop agreeing, and both edges of the eta band end up the
    # same distance from the probability boundary.
    np.testing.assert_array_equal(clip_mu(mu, Binomial()), mu)
    assert tail > 1e-7
    # It must nonetheless resolve the tail masses an exposure-weighted binomial
    # MLE lands on; a cap at 1 - 1e-6 truncated every fit finer than that.
    assert tail < 1e-6
    assert link.deriv_inverse(cap)[0] > 0.0


def test_power_one_has_exact_identity_derivatives_at_all_float_extremes():
    link = PowerLink(power=1.0)
    values = np.array([-np.inf, -3.0, -0.0, 0.0, 3.0, np.inf, np.nan])

    inverse = link.inverse(values)

    assert inverse is not values
    np.testing.assert_array_equal(inverse, values)
    np.testing.assert_array_equal(link.deriv(values), np.ones_like(values))
    np.testing.assert_array_equal(link.deriv_inverse(values), np.ones_like(values))
    np.testing.assert_array_equal(link.deriv2_inverse(values), np.zeros_like(values))
    np.testing.assert_array_equal(link.deriv3_inverse(values), np.zeros_like(values))


def test_power_one_normalizes_scalar_and_list_inputs_to_copied_arrays():
    link = PowerLink(power=1.0)

    scalar = link.inverse(-2.5)
    vector = link.inverse([-2.5, 0.0, 3.0])

    assert isinstance(scalar, np.ndarray)
    assert scalar.shape == ()
    assert scalar.item() == -2.5
    assert isinstance(vector, np.ndarray)
    np.testing.assert_array_equal(vector, np.array([-2.5, 0.0, 3.0]))


def test_poisson_sqrt_fit_recovers_means_above_old_400_clip():
    rng = np.random.default_rng(224)
    x = np.linspace(-1.0, 1.0, 600)
    expected_eta = 32.0 + 5.0 * x
    y = rng.poisson(expected_eta**2).astype(float)
    X = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="poisson",
        link="sqrt",
        selection_penalty=0.0,
        features={"x": Numeric()},
        tol=1e-9,
    ).fit(X, y)
    fitted = model.predict(X)

    assert model.result.converged
    assert fitted.max() > 1_000.0
    assert_allclose(fitted, expected_eta**2, rtol=0.08)


def test_poisson_sqrt_fit_recovers_from_exact_zero_initial_eta():
    y = np.ones(30)
    X = pd.DataFrame(index=np.arange(len(y)))
    offset = -np.ones(len(y))
    model = SuperGLM(
        family="poisson",
        link="sqrt",
        selection_penalty=0.0,
        features={},
        max_iter=200,
    ).fit(X, y, offset=offset)

    assert model.result.converged
    assert_allclose(model.predict(X, offset=offset), y, rtol=1e-10, atol=1e-10)


def test_poisson_sqrt_fit_preserves_offset_driven_negative_eta_branch():
    y = np.full(30, 100.0)
    X = pd.DataFrame(index=np.arange(len(y)))
    offset = np.full(len(y), -20.0)
    model = SuperGLM(
        family="poisson",
        link="sqrt",
        selection_penalty=0.0,
        features={},
    ).fit(X, y, offset=offset)

    assert model.result.converged
    assert_allclose(model.predict(X, offset=offset), y, rtol=1e-10, atol=1e-10)


def test_binomial_cauchit_fit_reaches_probabilities_outside_old_logit_band():
    rng = np.random.default_rng(225)
    x = np.linspace(-4.0, 4.0, 1_200)
    expected_eta = 8.0 * x
    probability = CauchitLink().inverse(expected_eta)
    y = rng.binomial(1, probability).astype(float)
    X = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="binomial",
        link="cauchit",
        selection_penalty=0.0,
        features={"x": Numeric()},
        max_iter=300,
        tol=1e-9,
    ).fit(X, y)
    low, high = model.predict(pd.DataFrame({"x": [-4.0, 4.0]}))

    assert model.result.converged
    assert low < 0.015
    assert high > 0.985


def test_binomial_cloglog_fit_recovers_from_saturated_positive_initial_eta():
    y = np.tile(np.array([0.0, 1.0]), 30)
    X = pd.DataFrame(index=np.arange(len(y)))
    offset = np.full(len(y), 20.0)
    model = SuperGLM(
        family="binomial",
        link="cloglog",
        selection_penalty=0.0,
        features={},
        max_iter=200,
    ).fit(X, y, offset=offset)

    assert model.result.converged
    assert_allclose(model.predict(X, offset=offset), 0.5, rtol=1e-7, atol=1e-7)


def test_binomial_cloglog_fit_reaches_weighted_mean_mle_in_the_upper_tail():
    # For an intercept-only binomial fit the MLE mean is the weighted mean of y
    # whatever the link is, so cloglog has to land where logit lands.  Here that
    # mean carries 3.33e-07 of tail mass, comfortably inside clip_mu's 1e-7
    # binomial bound, so nothing about it is separation.
    y = np.array([1.0, 0.0])
    X = pd.DataFrame(index=np.arange(2))
    sample_weight = np.array([3.0e6, 1.0])
    expected_tail = 1.0 / 3_000_001.0

    fits = {}
    for link in ("cloglog", "logit"):
        model = SuperGLM(
            family="binomial",
            link=link,
            selection_penalty=0.0,
            features={},
            tol=1e-14,
            max_iter=500,
        ).fit(X, y, sample_weight=sample_weight)
        assert model.result.converged
        fits[link] = (float(model.predict(X)[0]), model.result.deviance)

    cloglog_mu, cloglog_deviance = fits["cloglog"]
    logit_mu, logit_deviance = fits["logit"]

    assert_allclose(1.0 - cloglog_mu, expected_tail, rtol=1e-6, atol=0.0)
    assert_allclose(1.0 - cloglog_mu, 1.0 - logit_mu, rtol=1e-6, atol=0.0)
    assert_allclose(cloglog_deviance, logit_deviance, rtol=1e-9, atol=0.0)


def test_power_one_fit_matches_identity_with_negative_fitted_values():
    x = np.linspace(-2.0, 2.0, 80)
    y = -3.0 + 0.75 * x
    X = pd.DataFrame({"x": x})
    predictions = []

    for link in (IdentityLink(), PowerLink(1.0)):
        model = SuperGLM(
            family="gaussian",
            link=link,
            selection_penalty=0.0,
            features={"x": Numeric()},
        ).fit(X, y)
        assert model.result.converged
        predictions.append(model.predict(X))

    assert np.min(predictions[1]) < 0.0
    assert_allclose(predictions[1], predictions[0], atol=1e-12)
