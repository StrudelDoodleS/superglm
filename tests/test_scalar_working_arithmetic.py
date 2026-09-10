"""Working geometry must not discard a representable weighted result.

The references use independent family identities on represented inputs. The
unfixed implementation fails 48 of the original 318 parameterized cases.
"""

from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from superglm.distributions import (
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    clip_mu,
)
from superglm.links import IdentityLink, LogitLink, LogLink, SqrtLink, stabilize_eta
from superglm.model.state_ops import _solver_space_working_weights
from superglm.solvers.working_rows import coefficient_working_rows, pearson_chi2

SPECS = (
    ("gaussian", Gaussian, IdentityLink, None),
    ("gamma", Gamma, LogLink, None),
    ("poisson", Poisson, LogLink, None),
    ("binomial", Binomial, LogitLink, None),
    ("nb2", lambda: NegativeBinomial(2.0), LogLink, 2.0),
    ("nb2_small_theta", lambda: NegativeBinomial(1e-300), LogLink, 1e-300),
    ("nb2_large_theta", lambda: NegativeBinomial(1e200), LogLink, 1e200),
    ("tweedie", lambda: Tweedie(1.5), LogLink, 1.5),
    ("tweedie_near_one", lambda: Tweedie(1.01), LogLink, 1.01),
    ("tweedie_near_two", lambda: Tweedie(1.99), LogLink, 1.99),
)


def _d(value):
    return Decimal.from_float(float(value))


def _variance(name, mu, parameter):
    if name == "gaussian":
        return Decimal(1)
    if name == "gamma":
        return mu * mu
    if name == "poisson":
        return mu
    if name == "binomial":
        return mu * (1 - mu)
    if name.startswith("nb2"):
        return mu + mu * mu / _d(parameter)
    return mu ** _d(parameter)


def _assert_bounded(actual, expected):
    assert np.isfinite(float(expected)), "the reference must be representable"
    u = _d(np.finfo(float).eps / 2)
    # More roundings than the range-safe product/quotient, including power
    # evaluation on these bounded exponents, plus gradual-underflow allowance.
    bound = 64 * u / (1 - 64 * u) * abs(expected)
    bound += 16 * _d(np.nextafter(0.0, 1.0))
    assert np.isfinite(actual)
    assert abs(_d(actual) - expected) <= bound, (actual, expected, bound)


def _rebuild(family, link, eta, weights):
    solver = SimpleNamespace(beta=np.empty(0), intercept=float(eta[0]))
    model = SimpleNamespace(
        _solver_pirls_result=lambda: solver,
        _dm=SimpleNamespace(matvec=lambda beta: np.zeros(1)),
        _fit_offset=None,
        _link=link,
        _distribution=family,
        _fit_weights=weights,
    )
    return _solver_space_working_weights(model)


@pytest.mark.parametrize("spec", SPECS, ids=[s[0] for s in SPECS])
@pytest.mark.parametrize("location", ("ordinary", "large", "small"))
@pytest.mark.parametrize("weight", (0.0, 2.0, 1e250, 1e-270, 1e-300))
@pytest.mark.parametrize("consumer", ("coefficient_rows", "retained_state"))
def test_fisher_weight_preserves_representable_curvature(spec, location, weight, consumer):
    name, factory, link_factory, parameter = spec
    family, link = factory(), link_factory()
    if name == "binomial":
        eta = np.array([{"ordinary": 0.0, "large": 16.0, "small": -16.0}[location]])
    elif name == "gaussian":
        eta = np.array([{"ordinary": 4.0, "large": 1e30, "small": -1e30}[location]])
    else:
        eta = np.log(np.array([{"ordinary": 4.0, "large": 1e30, "small": 1e-30}[location]]))
    np.testing.assert_array_equal(stabilize_eta(eta, link), eta)
    mu = link.inverse(eta)
    np.testing.assert_array_equal(clip_mu(mu, family), mu)
    weights = np.array([weight])
    with np.errstate(all="ignore"):
        if consumer == "coefficient_rows":
            actual = coefficient_working_rows(
                distribution=family,
                link=link,
                y=np.ones(1),
                mu=mu,
                eta=eta,
                sample_weight=weights,
                prefer_observed=False,
            ).weights[0]
        else:
            actual = _rebuild(family, link, eta, weights)[0]
    with localcontext() as context:
        context.prec = 180
        m, w = _d(mu[0]), _d(weight)
        if name in ("gaussian", "gamma"):
            expected = w
        elif name == "poisson":
            expected = w * m
        elif name == "binomial":
            expected = w * m * (1 - m)
        elif name.startswith("nb2"):
            expected = w * m / (1 + m / _d(parameter))
        else:
            expected = w * m ** (2 - _d(parameter))
        _assert_bounded(actual, expected)


BASE_SPECS = tuple(
    s for s in SPECS if s[0] in ("gaussian", "gamma", "poisson", "binomial", "nb2", "tweedie")
)


@pytest.mark.parametrize("spec", BASE_SPECS, ids=[s[0] for s in BASE_SPECS])
@pytest.mark.parametrize("case", ("ordinary", "large_residual", "small_intermediate"))
def test_pearson_preserves_representable_contribution(spec, case):
    name, factory, _, parameter = spec
    if case == "ordinary":
        mean, y, weight = (0.5, 1.0, 2.0) if name == "binomial" else (2.0, 3.0, 2.0)
    elif name == "binomial":
        mean, y, weight = (0.5, 1.0, 1e250) if case == "large_residual" else (2e-7, 0.0, 1e-300)
    elif name == "gaussian":
        mean, y, weight = (0.0, 1e200, 1e-100) if case == "large_residual" else (0.0, 1e-200, 1e200)
    else:
        mean, y, weight = (
            (1e30, 1e200, 1e-100)
            if case == "large_residual"
            else (1e-30, 2e-30 if name == "gamma" else 0.0, 1e-270)
        )
    family = factory()
    np.testing.assert_array_equal(clip_mu(np.array([mean]), family), np.array([mean]))
    with np.errstate(all="ignore"):
        actual = pearson_chi2(
            distribution=family,
            y=np.array([y]),
            mu=np.array([mean]),
            sample_weight=np.array([weight]),
        )
    with localcontext() as context:
        context.prec = 180
        variance = max(_variance(name, _d(mean), parameter), _d(1e-100))
        _assert_bounded(actual, _d(weight) * (_d(y) - _d(mean)) ** 2 / variance)


@pytest.mark.parametrize("eta_value", (-20.0, 20.0))
def test_binomial_weight_preserves_the_clipped_mean_geometry(eta_value):
    family, link = Binomial(), LogitLink()
    eta = np.array([eta_value])
    mu = clip_mu(link.inverse(eta), family)
    weights = np.array([1e-300])
    actual = _rebuild(family, link, eta, weights)[0]
    with localcontext() as context:
        context.prec = 180
        raw_mean = _d(link.inverse(eta)[0])
        derivative = raw_mean * (1 - raw_mean)
        expected = _d(weights[0]) * derivative**2 / (_d(mu[0]) * (1 - _d(mu[0])))
        _assert_bounded(actual, expected)


def test_pearson_can_scale_a_residual_that_overflows_before_weighting():
    with np.errstate(all="ignore"):
        actual = pearson_chi2(
            distribution=Gaussian(),
            y=np.array([1e308, 1e308]),
            mu=np.array([-1e308, -1e308]),
            sample_weight=np.array([1e-310, 0.0]),
        )
    with localcontext() as context:
        context.prec = 180
        _assert_bounded(actual, _d(1e-310) * (_d(1e308) - _d(-1e308)) ** 2)


def test_nb2_pearson_can_recover_when_unweighted_variance_overflows():
    with np.errstate(all="ignore"):
        actual = pearson_chi2(
            distribution=NegativeBinomial(1e-300),
            y=np.array([1e31]),
            mu=np.array([1e30]),
            sample_weight=np.array([1e300]),
        )
    with localcontext() as context:
        context.prec = 180
        expected = _d(1e300) * (_d(1e31) - _d(1e30)) ** 2
        expected /= _variance("nb2", _d(1e30), 1e-300)
        _assert_bounded(actual, expected)


def test_reconstruction_preserves_the_poisson_sqrt_limit_at_zero():
    actual = _rebuild(Poisson(), SqrtLink(), np.zeros(1), np.array([2.0]))
    np.testing.assert_array_equal(actual, np.array([8.0]))


def test_family_subclasses_retain_their_own_variance():
    class DoubleVarianceGamma(Gamma):
        def variance(self, mu):
            return 2 * mu**2

    actual = _rebuild(DoubleVarianceGamma(), LogLink(), np.zeros(1), np.array([2.0]))
    np.testing.assert_array_equal(actual, np.array([1.0]))


def test_alternate_log_link_uses_the_derivative_before_mean_clipping():
    """Binomial/log must not substitute its clipped mean for exp(eta)."""
    family, link = Binomial(), LogLink()
    eta = np.array([np.log(2.0)])
    mu = clip_mu(link.inverse(eta), family)
    rows = coefficient_working_rows(
        distribution=family,
        link=link,
        y=np.zeros(1),
        mu=mu,
        eta=eta,
        sample_weight=np.ones(1),
        prefer_observed=False,
    )
    with localcontext() as context:
        context.prec = 180
        mean, derivative = _d(mu[0]), _d(link.inverse(eta)[0])
        _assert_bounded(rows.weights[0], derivative**2 / (mean * (1 - mean)))
        _assert_bounded(rows.response[0], _d(eta[0]) - mean / derivative)
    np.testing.assert_array_equal(rows.weights, _rebuild(family, link, eta, np.ones(1)))


@pytest.mark.parametrize("family", (Gamma(), Tweedie(1.5)))
@pytest.mark.parametrize("mean,weight", ((1e-250, 1e200), (1e250, 1e-200)))
def test_unfloored_reporting_keeps_a_representable_pearson_sum(family, mean, weight):
    """Reporting's unfloored variance may not itself fit in binary64."""
    with np.errstate(all="ignore"):
        actual = pearson_chi2(
            distribution=family,
            y=np.array([2 * mean]),
            mu=np.array([mean]),
            sample_weight=np.array([weight]),
            variance_floor=0.0,
        )
    with localcontext() as context:
        context.prec = 180
        power = Decimal(2) if type(family) is Gamma else _d(family.p)
        expected = _d(weight) * (_d(2 * mean) - _d(mean)) ** 2 / _d(mean) ** power
        _assert_bounded(actual, expected)
