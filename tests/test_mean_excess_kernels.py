"""Row-wise mean excess ``e(v) = E[Y - v | Y > v]`` of nine distributional families.

Every value is held to ``|err| <= 64 u kappa |e|``, ``u = 2**-53``, against a
reference that shares no arithmetic with the kernel:

- mpmath at 60 digits or more, evaluating the textbook form at the exact
  float64 inputs (a ratio of incomplete gamma, normal or beta integrals minus
  ``v``, whose cancellation the extra digits absorb);
- exact rational values where the law has them (gamma shapes 1, 2 and 3,
  the generalised Pareto, Lomax);
- SciPy quadrature of ``exp(log S(v + s) - log S(v))`` from ``scipy.stats``
  at ordinary amounts, which checks each family's parameter map.

``kappa`` is the conditioning of the form, stated beside each test.  It is 1
where the kernel is a sum of positive terms and otherwise the bounded loss the
kernel's docstring gives, times the size of any logarithm the form must round
(``|log v|``, ``|mu|``, ``sigma^2/2``, ``|Q w|``).  64 covers the ten to
twenty roundings of a form at libm's one ulp each; it is not a measured value.

A law no fit produces (a CV at or below ``1e-6``, a gamma or log-logistic
shape at or above ``1e8``, a log-scale above 20, a subnormal amount) is held
to three things only: no crash, no NaN and no negative value.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import integrate, stats

from superglm import (
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogLogisticLSS,
    LogNormalLS,
    LomaxLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import MeanExcessFamily, ParameterSupport
from superglm.distributional.kernels import _mean_excess as kernel
from superglm.distributional.kernels.generalized_gamma import GeneralizedGammaDomainError

mp = pytest.importorskip("mpmath")

U = 2.0**-53
C = 64.0
REAL_LINE = ParameterSupport()
POSITIVE = ParameterSupport(lower=0.0)
NON_NEGATIVE = ParameterSupport(lower=0.0, lower_inclusive=True)


def _column(*values: float) -> list[np.ndarray]:
    return [np.array([float(value)]) for value in values]


def _relative_error(got: float, reference) -> float:
    return float(abs(mp.mpf(float(got)) - reference) / abs(reference))


def _assert_close(got: float, reference, kappa: float = 1.0) -> None:
    assert math.isfinite(got) and got > 0.0, f"got {got!r}"
    error = _relative_error(got, reference)
    bound = C * U * kappa
    assert error <= bound, f"relative error {error:.3e} exceeds 64 u kappa = {bound:.3e}"


# ------------------------------------------------------------------ references


def _upper(z):
    return mp.erfc(z / mp.sqrt(2)) / 2


def _phi(z):
    return mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi)


def _normal_reference(z: float):
    with mp.workdps(80):
        z = mp.mpf(z)
        return _phi(z) / _upper(z) - z


def _log_normal_reference(v: float, mu: float, sigma: float):
    """``E[Y] Q(z - sigma)/Q(z) - v``, the textbook form."""
    with mp.workdps(400):
        v, mu, sigma = mp.mpf(v), mp.mpf(mu), mp.mpf(sigma)
        z = (mp.log(v) - mu) / sigma
        return mp.exp(mu + sigma**2 / 2) * _upper(z - sigma) / _upper(z) - v


def _two_piece_reference(v: float, mu: float, sigma: float, eps: float, *, log: bool):
    """``E[Y; Y > v]/S(v) - v`` of the epsilon-skew law, each side from its own closed form."""
    with mp.workdps(900):
        v, mu, eps = mp.mpf(v), mp.mpf(mu), mp.mpf(eps)
        left, right = mp.mpf(sigma) * (1 - eps), mp.mpf(sigma) * (1 + eps)
        t = (mp.log(v) if log else v) - mu
        if t < 0:
            z = t / left
            survival = (1 + eps) / 2 + (1 - eps) * (_upper(z) - mp.mpf(1) / 2)
            if log:
                first = (1 - eps) * mp.exp(left**2 / 2) * (_upper(z - left) - _upper(-left))
                first += (1 + eps) * mp.exp(right**2 / 2) * _upper(-right)
            else:
                first = (1 - eps) * left * (_phi(z) - _phi(0)) + (1 + eps) * right * _phi(0)
        else:
            z = t / right
            survival = (1 + eps) * _upper(z)
            if log:
                first = (1 + eps) * mp.exp(right**2 / 2) * _upper(z - right)
            else:
                first = (1 + eps) * right * _phi(z)
        if log:
            return mp.exp(mu) * first / survival - v
        return first / survival - t


def _gamma_reference(shape: float, x: float):
    """Unit-scale gamma: ``Gamma(a + 1, x)/Gamma(a, x) - x``."""
    with mp.workdps(60):
        a, x = mp.mpf(shape), mp.mpf(x)
        return mp.gammainc(a + 1, x, mp.inf) / mp.gammainc(a, x, mp.inf) - x


def _gamma_integral_reference(shape: float, x: float):
    """The same value at a large shape, as a ratio of positive integrals over the excess."""
    with mp.workdps(40):
        a, x = mp.mpf(shape), mp.mpf(x)
        h = mp.sqrt(a)
        knots = [0, 1, 4, 15, mp.inf]

        def density(t):
            return mp.exp((a - 1) * mp.log1p(h * t / x) - h * t)

        return h * mp.quad(lambda t: t * density(t), knots) / mp.quad(density, knots)


def _generalized_gamma_reference(v: float, mu: float, sigma: float, q: float):
    """``v [Gamma(k + r, x)/(x^r Gamma(k, x)) - 1]``, upper gammas for ``Q > 0``, lower below."""
    with mp.workdps(80):
        v, mu, sigma, q = mp.mpf(v), mp.mpf(mu), mp.mpf(sigma), mp.mpf(q)
        k, r = 1 / q**2, sigma / q
        x = k * mp.exp(q * (mp.log(v) - mu) / sigma)
        if q > 0 and k > 3e4:
            # mpmath's incomplete gamma is slow at this shape: integrate over the gamma excess.
            h = x / (x - k + 1) if x > k + mp.sqrt(k) else mp.sqrt(k)
            knots = [0, 1, 5, 20, 80, mp.inf]

            def density(t):
                return mp.exp((k - 1) * mp.log1p(h * t / x) - h * t)

            numerator = mp.quad(lambda t: density(t) * mp.expm1(r * mp.log1p(h * t / x)), knots)
            return v * numerator / mp.quad(density, knots)
        if q > 0:
            return v * (mp.gammainc(k + r, x, mp.inf) / (x**r * mp.gammainc(k, x, mp.inf)) - 1)
        return v * (mp.gammainc(k + r, 0, x) / (x**r * mp.gammainc(k, 0, x)) - 1)


def _generalized_gamma_log_survival(v: float, mu: float, sigma: float, q: float) -> float:
    with mp.workdps(80):
        v, mu, sigma, q = mp.mpf(v), mp.mpf(mu), mp.mpf(sigma), mp.mpf(q)
        k = 1 / q**2
        x = k * mp.exp(q * (mp.log(v) - mu) / sigma)
        bounds = (x, mp.inf) if q > 0 else (0, x)
        return float(mp.log(mp.gammainc(k, *bounds, regularized=True)))


def _log_logistic_reference(v: float, scale: float, shape: float):
    """``(s/a) B_S(1 - 1/a, 1/a)/S`` with ``S = 1/(1 + (v/s)^a)``."""
    with mp.workdps(450):
        v, s, a = mp.mpf(v), mp.mpf(scale), mp.mpf(shape)
        survival = 1 / (1 + (v / s) ** a)
        return s * mp.betainc(1 - 1 / a, 1 / a, 0, survival) / (a * survival)


def _quadrature_reference(log_survival, v: float, scale: float) -> float:
    """``int_0^inf S(v + s)/S(v) ds``; ``scale`` places the nodes."""
    base = log_survival(v)

    def integrand(t: float) -> float:
        if t >= 1.0:
            return 0.0
        s = scale * t / (1.0 - t)
        return math.exp(log_survival(v + s) - base) * scale / (1.0 - t) ** 2

    return integrate.quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=400)[0]


# ------------------------------------------------------------------ the normal


def _normal_conditioning(z: float) -> float:
    """``phi/Q - z`` is a sum for ``z <= 0`` and a fraction above 3; between, it loses ``r/(r - z)``."""
    if z <= 0.0 or z > 3.0:
        return 1.0
    reference = _normal_reference(z)
    return float((reference + z) / reference)


@pytest.mark.parametrize(
    "z",
    [-1e300, -1e5, -40.0, -8.0, -3.0, -1.0, -1e-9, 0.0, 1e-9, 0.5, 1.0, 2.0, 2.999, 3.0]
    + [3.0001, 3.5, 5.0, 8.0, 20.0, 38.0, 200.0],
)
def test_the_normal_excess_matches_mpmath_on_both_sides_of_its_switches(z):
    got = kernel._normal_excess(np.array([z]))[0]
    _assert_close(got, _normal_reference(z), _normal_conditioning(z))


def test_the_normal_far_tail_matches_its_asymptotic_series():
    """At ``z = 400`` the series ``1/z - 2/z^3 + 10/z^5 - 74/z^7`` is exact to ``u``."""
    z = Fraction(400)
    series = 1 / z - 2 / z**3 + 10 / z**5 - 74 / z**7
    got = GaussianLS().mean_excess(np.array([403.0]), np.array([[3.0, 1.0]]))[0]
    assert abs(got / float(series) - 1.0) <= C * U


def test_the_gaussian_family_scales_and_shifts_the_normal_excess():
    theta = np.array([[0.0, 1.0], [2.5, 0.3], [-7.0, 40.0], [1e6, 0.02]])
    amount = np.array([0.7, 3.4, -100.0, 1e6 + 0.05])
    got = GaussianLS().mean_excess(amount, theta)
    for value, v, (location, scale) in zip(got, amount, theta, strict=True):
        with mp.workdps(80):
            z = (mp.mpf(v) - mp.mpf(location)) / mp.mpf(scale)
            reference = mp.mpf(scale) * (_phi(z) / _upper(z) - z)
        _assert_close(value, reference, _normal_conditioning(float(z)))


# ------------------------------------------------------------------ two-piece normal


@pytest.mark.parametrize("eps", [-0.99999999, -0.6, 0.0, 0.5, 0.9])
@pytest.mark.parametrize("z", [-1e6, -40.0, -5.0, -1.0, -1e-3, -1e-8, 0.0, 1e-8, 0.7, 12.0, 42.0])
def test_the_two_piece_normal_matches_each_side_of_its_join(eps, z):
    """Left of the join the window integral loses under a factor 3; right, as the normal."""
    scale = 1.5
    v = z * scale * (1.0 - eps if z < 0.0 else 1.0 + eps)
    got = kernel.two_piece_normal_mean_excess(*_column(v, 0.0, scale, eps))[0]
    kappa = 3.0 if z < 0.0 else _normal_conditioning(z)
    _assert_close(got, _two_piece_reference(v, 0.0, scale, eps, log=False), kappa)


# ------------------------------------------------------------------ log-normal


def _log_normal_conditioning(v: float, mu: float, sigma: float, reference) -> float:
    """``v expm1(I)``: what the form rounds, and what ``expm1`` does with it.

    Above the median ``I`` carries an absolute error of ``u (1 + I)``; below it the tail mean
    is assembled in logs from ``mu + sigma^2/2``.  ``expm1`` divides an absolute error by
    ``1 - e^-I``, at least ``1 - e^(-1/4)`` where the closed form is read, hence the 4.5.
    The last term is the cancellation in ``log v - mu``, which sets ``z``.
    """
    log_v = math.log(v)
    drop = float(mp.log1p(reference / mp.mpf(v)))
    rounded = drop if log_v >= mu else abs(mu) + 0.5 * sigma * sigma
    return 4.5 * (2.0 + rounded) + (abs(log_v) + abs(mu)) / abs(log_v - mu)


@pytest.mark.parametrize("sigma", [0.01, 0.3, 1.0, 2.5, 7.0, 8.5, 12.0, 20.0])
@pytest.mark.parametrize("z", [-60.0, -6.0, -1.0, 0.2, 2.0, 8.0, 37.0, 38.5])
def test_the_log_normal_matches_mpmath_from_the_body_to_survivals_below_1e_300(sigma, z):
    v = math.exp(min(max(sigma * z, -700.0), 700.0))
    got = kernel.log_normal_mean_excess(*_column(v, 0.0, sigma))[0]
    reference = _log_normal_reference(v, 0.0, sigma)
    if reference > mp.mpf(2) ** 1023:
        assert got == math.inf
        return
    _assert_close(got, reference, _log_normal_conditioning(v, 0.0, sigma, reference))


@pytest.mark.parametrize(
    ("v", "mu", "sigma"),
    [(3.0, 1.0, 0.01), (2.0, 0.5, 3.0), (math.exp(-158.0), -500.0, 9.0), (1e-200, -400.0, 16.0)],
)
def test_the_log_normal_carries_a_location(v, mu, sigma):
    got = kernel.log_normal_mean_excess(*_column(v, mu, sigma))[0]
    reference = _log_normal_reference(v, mu, sigma)
    _assert_close(got, reference, _log_normal_conditioning(v, mu, sigma, reference))


def test_the_log_normal_family_agrees_in_both_parametrisations():
    sigma = np.array([0.2, 1.3, 0.05])
    location = np.array([0.4, -1.0, 2.0])
    amount = np.array([2.0, 5.0, 8.1])
    by_location = LogNormalLS(parametrisation="location").mean_excess(
        amount, np.column_stack((location, sigma))
    )
    mean = np.exp(location + 0.5 * sigma * sigma)
    by_mean = LogNormalLS().mean_excess(amount, np.column_stack((mean, sigma)))
    for index in range(3):
        reference = _log_normal_reference(amount[index], location[index], sigma[index])
        kappa = _log_normal_conditioning(amount[index], location[index], sigma[index], reference)
        _assert_close(by_location[index], reference, kappa)
        # The mean form rounds exp and log once more on the way back to mu.
        _assert_close(by_mean[index], reference, 2.0 * kappa)


# ------------------------------------------------------------------ two-piece log-normal


@pytest.mark.parametrize("eps", [-0.88, -0.55, 0.0, 0.6])
@pytest.mark.parametrize("sigma", [0.1, 0.4, 1.2, 5.0])
@pytest.mark.parametrize("z", [-9.5, -2.2, -0.3, -1e-3, -1e-8, 0.0, 1e-8, 0.5, 12.0, 38.0])
def test_the_two_piece_log_normal_matches_below_at_and_above_its_join(eps, sigma, z):
    """Left: the window share loses under 2.5 and the right piece adds its median excess."""
    left, right = sigma * (1.0 - eps), sigma * (1.0 + eps)
    v = math.exp(z * (left if z < 0.0 else right))
    got = kernel.two_piece_log_normal_mean_excess(*_column(v, 0.0, sigma, eps))[0]
    reference = _two_piece_reference(v, 0.0, sigma, eps, log=True)
    if z > 0.0:
        kappa = _log_normal_conditioning(v, 0.0, right, reference)
    else:
        # The right piece's excess at its median, where log(E/m) is under sigma_r^2/2 + log 2.
        kappa = 3.0 + 4.5 * (3.0 + 0.5 * right * right)
    _assert_close(got, reference, kappa)


@pytest.mark.parametrize("sigma", [0.1, 1.0])
@pytest.mark.parametrize("z", [-1.0, -1e-8, 0.0, 1e-8])
def test_a_two_piece_log_normal_with_almost_no_right_piece_stays_positive_and_exact(sigma, z):
    """``eps = -0.99999999``: the right piece holds ``5e-9`` of the mass."""
    eps = -0.99999999
    v = math.exp(z * sigma * (1.0 - eps if z < 0.0 else 1.0 + eps))
    got = kernel.two_piece_log_normal_mean_excess(*_column(v, 0.0, sigma, eps))[0]
    _assert_close(got, _two_piece_reference(v, 0.0, sigma, eps, log=True), 3.0 + 4.5 * 3.0)


def test_the_two_piece_log_normal_family_agrees_in_both_parametrisations():
    theta = np.array([[0.3, 0.4, -0.5], [0.3, 0.4, -0.5], [0.3, 1.2, 0.6]])
    amount = np.array([0.8, 2.0, 1.5])
    location = TwoPieceLogNormalLSS(parametrisation="location")
    got = location.mean_excess(amount, theta)
    mean_theta = np.column_stack((location.default_prediction(theta), theta[:, 1:]))
    by_mean = TwoPieceLogNormalLSS().mean_excess(amount, mean_theta)
    for index in range(3):
        reference = _two_piece_reference(amount[index], *theta[index], log=True)
        right = theta[index, 1] * (1.0 + theta[index, 2])
        kappa = 3.0 + _log_normal_conditioning(amount[index], theta[index, 0], right, reference)
        _assert_close(got[index], reference, kappa)
        _assert_close(by_mean[index], reference, 4.0 * kappa)


# ------------------------------------------------------------------ gamma


def _refuse(rows, x):
    raise AssertionError("the continued fraction must not ask for the direct form")


@pytest.mark.parametrize("shape", [1, 2, 3])
@pytest.mark.parametrize("x", [8, 30, 700])
def test_the_gamma_family_matches_its_rational_values(shape, x):
    """Shapes 1, 2, 3: ``sum_j (a - j) x^j/j! / sum_j x^j/j!`` over ``j < a``, exactly."""
    partial = [Fraction(x) ** j / math.factorial(j) for j in range(shape)]
    exact = sum((shape - j) * term for j, term in enumerate(partial)) / sum(partial)
    theta = np.array([[float(shape), 1.0 / math.sqrt(shape)]])
    got = GammaLS().mean_excess(np.array([float(x)]), theta)[0]
    assert abs(got / float(exact) - 1.0) <= C * U


@pytest.mark.parametrize("shape", [1e-8, 1e-4, 0.01, 0.3, 2.5, 25.0, 137.5, 1e4])
@pytest.mark.parametrize("depth", [1e-12, 0.5, 4.0, 8.0, 40.0])
def test_the_gamma_fraction_matches_mpmath_from_its_switch_to_the_deep_tail(shape, depth):
    """``x = a + (1.5 + depth)(sqrt(a) + 1)``; a sum of positive terms, so ``kappa = 1``."""
    x = shape + (1.5 + depth) * (math.sqrt(shape) + 1.0)
    got = kernel.gamma_mean_excess(*_column(x, shape, shape), _refuse)[0]
    _assert_close(got, _gamma_reference(shape, x))


def test_the_gamma_fraction_goes_deeper_where_32_levels_do_not_settle():
    """At ``a = 1e-8, x = 1.51`` the value at 32 levels is off by ``8e-11``; at 128 it settles."""
    shape, x = np.array([1e-8, 2.5]), np.array([1.51, 755.0])
    got, levels = kernel._gamma_excess(shape, x - shape)
    _assert_close(got[0], _gamma_reference(1e-8, 1.51))
    _assert_close(got[1], _gamma_reference(2.5, 755.0))
    assert levels[0] > levels[1] == 64


def test_the_gamma_fraction_keeps_the_distance_from_a_large_shape():
    """At ``a = 4e7`` forming ``x + 2n + 1 - a`` instead of ``(x - a) + (2n + 1)`` costs ``1e-12``."""
    shape = 4e7 + 0.3
    gap = 1.5 * math.sqrt(shape) + 1.5 + 2.0**-20
    got = kernel._gamma_excess(np.array([shape]), np.array([gap]))[0][0]
    with mp.workdps(40):
        x = mp.mpf(shape) + mp.mpf(gap)
    _assert_close(got, _gamma_integral_reference(shape, x))


@pytest.mark.parametrize("cv", [1.8, 0.63, 0.2, 0.01])
@pytest.mark.parametrize("share", [1e-6, 0.3, 0.9, 1.0, "half", "switch"])
def test_the_gamma_family_matches_mpmath_at_and_below_its_switch(cv, share):
    """The direct form: positive terms up to the mean, a loss under 5 above.

    The log density reads ``(x - a)/a`` from a rounded ``x``, which costs
    ``u sqrt(a)`` in standard deviations, so ``kappa = 5 (1 + sqrt(a))``.
    """
    mean = 3.0
    with mp.workdps(60):
        shape = 1 / mp.mpf(cv) ** 2
    root = math.sqrt(float(shape))
    if share == "half":
        x = float(shape) + 0.75 * root + 0.75
    elif share == "switch":
        x = float(shape) + 1.5 * root + 1.5 - 1e-9 * (root + 1.0)
    else:
        x = float(shape) * share
    v = float(x / shape * mean)
    got = GammaLS().mean_excess(np.array([v]), np.array([[mean, cv]]))[0]
    with mp.workdps(60):
        scale = mp.mpf(mean) * mp.mpf(cv) ** 2
        reference = scale * _gamma_reference(shape, mp.mpf(v) / scale)
    _assert_close(got, reference, 5.0 * (1.0 + root))


# ------------------------------------------------------------------ generalised gamma

_LOCATION_FORM = GeneralizedGammaLSS(parametrisation="location")


def _generalized_gamma(v: float, sigma: float, q: float, mu: float = 0.0, family=_LOCATION_FORM):
    return family.mean_excess(np.array([v]), np.array([[mu, sigma, q]]))[0]


def _body_reference(v: float, sigma: float, q: float):
    """The value, and ``kappa`` for ``v expm1(D)``: ``D`` is rounded to ``u (2 + |log S|)`` and
    ``expm1`` divides by ``1 - e^-D = e/(v + e)``."""
    reference = _generalized_gamma_reference(v, 0.0, sigma, q)
    log_survival = _generalized_gamma_log_survival(v, 0.0, sigma, q)
    return reference, (2.0 + abs(log_survival)) * (1.0 + v / float(reference))


def _tail_conditioning(v: float, sigma: float, q: float) -> float:
    """The fractions' difference costs ``1/sigma``; ``x = k e^(Q w)`` rounds ``Q w``."""
    return 1.0 + 1.0 / sigma + abs(q * math.log(v) / sigma)


@pytest.mark.parametrize("v", [1.2, 3.0, 5.0, 10.0, 12.0])
def test_a_weibull_far_beyond_its_scale_is_positive_and_exact(v):
    """``mu = 0, sigma = 0.1, Q = 1``: at ``v = 5`` the log survival is ``-9.8e6``.

    A difference of the two log tails returns ``-9.3e-9`` there; the value is ``5.1e-8``.
    """
    with mp.workdps(60):
        x = mp.mpf(v) ** 10
        reference = mp.mpf(v) * mp.quad(
            lambda t: mp.expm1(mp.log1p(t / x) / 10) * mp.exp(-t), [0, 1, 5, 30, mp.inf]
        )
    _assert_close(_generalized_gamma(v, 0.1, 1.0), reference, _tail_conditioning(v, 0.1, 1.0))


@pytest.mark.parametrize("v", [1.021506891788, 1.040575003943, 1.041234596709])
def test_a_weighted_law_below_the_fit_floor_is_read_not_refused(v):
    """``sigma = 0.01005, Q = 2`` under the default floor of 0.01: the weighted scale is 0.00995."""
    sigma, q = 0.01005, 2.0
    assert sigma / math.sqrt(1.0 + sigma * q) < _LOCATION_FORM.scale_floor
    reference = _generalized_gamma_reference(v, 0.0, sigma, q)
    _assert_close(_generalized_gamma(v, sigma, q), reference, _tail_conditioning(v, sigma, q))


@pytest.mark.parametrize("v", [0.99, 1.0, 1.01])
def test_a_weighted_law_below_the_fit_floor_is_read_in_the_body_too(v):
    """The same law at ordinary amounts, where both log survivals come from the window kernel."""
    sigma, q = 0.01005, 2.0
    _assert_close(_generalized_gamma(v, sigma, q), *_body_reference(v, sigma, q))


@pytest.mark.parametrize(
    ("sigma", "q", "v"),
    [
        (0.8, 0.5, 24.00028740885),
        (0.8, 0.5, 3957.514734412),
        (0.8, 0.5, 4815.543806646),
        (0.014, 8.0, 1.011956891992),
        (0.014, 8.0, 1.019094187715),
        (0.05, 15.0, 1.026882783476),
        (0.05, 15.0, 1.041031412011),
        (0.1, 0.05, 1.767856661411),
        (0.1, 0.05, 19.65607294437),
        (0.3, 0.001, 6.04566873752),
        (0.3, 0.001, 67112.72114126),
        (0.3, -0.5, 31.88263164243),
        (0.3, -0.5, 1.426278411077e45),
        (0.3, -0.5, 1.426278411077e51),
        (0.3, -0.05, 6.5),
        (0.3, -0.05, 1e6),
        (0.2, -2.0, 50.0),
        (0.2, -2.0, 1e150),
    ],
)
def test_the_generalised_gamma_tail_matches_mpmath_at_either_sign_of_its_shape(sigma, q, v):
    """Survivals from ``1e-9`` to ``1e-340``, read by the pair of continued fractions."""
    reference = _generalized_gamma_reference(v, 0.0, sigma, q)
    _assert_close(_generalized_gamma(v, sigma, q), reference, _tail_conditioning(v, sigma, q))


@pytest.mark.parametrize("sigma", [0.05, 0.3, 0.8, 1.5])
@pytest.mark.parametrize("q", [0.05, 0.5, 2.0, -0.05, -0.5])
@pytest.mark.parametrize("w", [-2.0, 0.0, 1.0])
def test_the_generalised_gamma_body_matches_mpmath(sigma, q, w):
    """The log-ratio form at ordinary amounts, both signs of the shape."""
    v = math.exp(sigma * w)
    _assert_close(_generalized_gamma(v, sigma, q), *_body_reference(v, sigma, q))


def test_a_generalised_gamma_at_shape_zero_is_the_log_normal():
    amount = np.array([0.4, 3.0, 2e5])
    theta = np.column_stack((np.full(3, 0.2), np.full(3, 0.7), np.zeros(3)))
    expected = LogNormalLS(parametrisation="location").mean_excess(amount, theta[:, :2])
    np.testing.assert_array_equal(_LOCATION_FORM.mean_excess(amount, theta), expected)


def test_a_generalised_gamma_without_a_mean_reads_infinity_in_location_form():
    theta = np.array([[0.0, 2.0, -0.6], [0.0, 2.0, -0.5], [0.0, 2.0, -0.4]])
    got = _LOCATION_FORM.mean_excess(np.array([2.0, 1e-5, 2.0]), theta)
    assert got[0] == math.inf and got[1] == math.inf
    _assert_close(got[2], _generalized_gamma_reference(2.0, 0.0, 2.0, -0.4), 64.0)
    with pytest.raises(GeneralizedGammaDomainError, match="mean does not exist"):
        GeneralizedGammaLSS().mean_excess(np.array([2.0]), np.array([[1.0, 2.0, -0.6]]))


def test_the_generalised_gamma_mean_form_agrees_with_its_location_form():
    theta = np.array([[0.2, 0.4, 0.7], [0.2, 0.4, -0.7], [-1.0, 0.1, 1.0]])
    amount = np.array([3.0, 3.0, 0.9])
    mean_theta = np.column_stack((_LOCATION_FORM.default_prediction(theta), theta[:, 1:]))
    by_location = _LOCATION_FORM.mean_excess(amount, theta)
    by_mean = GeneralizedGammaLSS().mean_excess(amount, mean_theta)
    # The mean form finds mu = log m - log C again, one rounding of each log away.
    np.testing.assert_allclose(by_mean, by_location, rtol=C * U * 64.0)


# ------------------------------------------------------------------ Pareto tails


def test_the_generalised_pareto_and_lomax_are_their_closed_forms():
    v, scale, shape = Fraction(7, 2), Fraction(5, 4), Fraction(3, 8)
    got = GeneralizedParetoLSS().mean_excess(
        np.array([float(v)]), np.array([[float(scale), float(shape)]])
    )[0]
    assert abs(got / float((scale + shape * v) / (1 - shape)) - 1.0) <= 4.0 * U
    tail = Fraction(5, 2)
    got = LomaxLSS().mean_excess(np.array([float(v)]), np.array([[float(scale), float(tail)]]))[0]
    assert abs(got / float((scale + v) / (tail - 1)) - 1.0) <= 4.0 * U


@pytest.mark.parametrize("family", [LomaxLSS(), LogLogisticLSS()], ids=["lomax", "log-logistic"])
@pytest.mark.parametrize("shape", [0.3, 1.0])
def test_a_shape_at_or_below_one_has_no_mean_and_reads_infinity(family, shape):
    got = family.mean_excess(np.array([-1.0, 0.5, 40.0]), np.tile([2.0, shape], (3, 1)))
    assert np.all(got == math.inf)


# ------------------------------------------------------------------ log-logistic


@pytest.mark.parametrize("logit", [-800.0, -690.0, -40.0, -2.0, 0.0, 3.0, 4.2, 40.0, 690.0])
def test_the_log_logistic_at_shape_two_is_the_arctangent_law(logit):
    """``e(v) = s atan(s/v) (1 + (v/s)^2)``, from ``F = 1e-300`` to ``S = 1e-300``."""
    scale = 1.75
    v = scale * math.exp(logit / 2.0)
    got = kernel.log_logistic_mean_excess(*_column(v, scale, 2.0))[0]
    with mp.workdps(60):
        ratio = mp.mpf(v) / mp.mpf(scale)
        reference = mp.mpf(scale) * mp.atan(1 / ratio) * (1 + ratio**2)
    _assert_close(got, reference)


@pytest.mark.parametrize("shape", [1.0001, 1.02, 3.7, 30.0, 300.0])
@pytest.mark.parametrize("logit", [-800.0, -40.0, -1.0, 1e-6, 2.0, 4.1, 4.2, 40.0, 820.0])
def test_the_log_logistic_matches_the_incomplete_beta(shape, logit):
    """Where ``F`` leaves float64 the form is ``s (C - 1) + (s - v)``, with ``C - 1`` rounded to ``u``."""
    v = math.exp(min(logit / shape, 700.0))
    got = kernel.log_logistic_mean_excess(*_column(v, 1.0, shape))[0]
    kappa = 1.0 + shape / 700.0 if logit < -700.0 else 1.0
    _assert_close(got, _log_logistic_reference(v, 1.0, shape), kappa)


@pytest.mark.parametrize("offset", [3e-7, -3e-7, 7e-5])
def test_a_concentrated_log_logistic_keeps_its_digits_beside_the_median(offset):
    """Shape 3000 just off the median: ``log(v/s)`` from a rounded quotient is off by ``3000 u``."""
    scale, shape = 1.7, 3000.0
    v = scale * (1.0 + offset)
    got = kernel.log_logistic_mean_excess(*_column(v, scale, shape))[0]
    _assert_close(got, _log_logistic_reference(v, scale, shape))


def test_the_log_logistic_takes_an_amount_over_scale_ratio_beyond_float64():
    """``v/s = 1e350``: the log ratio is 805.9, and the excess is ``v`` to ``1e-700``."""
    got = LogLogisticLSS().mean_excess(np.array([1e250]), np.array([[1e-100, 2.0]]))[0]
    _assert_close(got, _log_logistic_reference(1e250, 1e-100, 2.0))


# ------------------------------------------------------------------ every family

_TWO_PIECE = (0.4, 0.5, -0.3)
FAMILIES = [
    pytest.param(GaussianLS(), (1.0, 2.0), REAL_LINE, id="gaussian"),
    pytest.param(GammaLS(), (3.0, 0.5), POSITIVE, id="gamma"),
    pytest.param(LogNormalLS(), (3.0, 0.5), POSITIVE, id="log-normal-mean"),
    pytest.param(LogNormalLS(parametrisation="location"), (1.0, 0.5), POSITIVE, id="log-normal"),
    pytest.param(GeneralizedGammaLSS(), (3.0, 0.5, 0.4), POSITIVE, id="gen-gamma-mean"),
    pytest.param(_LOCATION_FORM, (1.0, 0.5, -0.4), POSITIVE, id="gen-gamma"),
    pytest.param(GeneralizedParetoLSS(), (2.0, 0.3), NON_NEGATIVE, id="gen-pareto"),
    pytest.param(LomaxLSS(), (2.0, 3.0), NON_NEGATIVE, id="lomax"),
    pytest.param(LogLogisticLSS(), (2.0, 3.0), POSITIVE, id="log-logistic"),
    pytest.param(TwoPieceNormalLSS(), _TWO_PIECE, REAL_LINE, id="two-piece-normal"),
    pytest.param(TwoPieceLogNormalLSS(), (3.0, 0.5, 0.3), POSITIVE, id="two-piece-log-mean"),
    pytest.param(
        TwoPieceLogNormalLSS(parametrisation="location"), _TWO_PIECE, POSITIVE, id="two-piece-log"
    ),
]


def _two_piece_log_survival(t: float, mu: float, sigma: float, eps: float) -> float:
    w = (t - mu) / sigma
    if w < 0.0:
        return math.log1p(-(1.0 - eps) * stats.norm.cdf(w / (1.0 - eps)))
    return math.log1p(eps) + stats.norm.logsf(w / (1.0 + eps))


def _scipy_log_survival(family, theta):
    """The law's log survival from ``scipy.stats`` at the documented parameter map."""
    name, form = type(family).__name__, getattr(family, "parametrisation", "location")
    first, second = theta[0], theta[1]
    if name == "GaussianLS":
        return stats.norm(first, second).logsf
    if name == "GammaLS":
        return stats.gamma(a=1.0 / second**2, scale=first * second**2).logsf
    if name == "LogNormalLS":
        median = first * math.exp(-0.5 * second**2) if form == "mean" else math.exp(first)
        return stats.lognorm(s=second, scale=median).logsf
    if name == "GeneralizedParetoLSS":
        return stats.genpareto(c=second, scale=first).logsf
    if name == "LomaxLSS":
        return stats.lomax(c=second, scale=first).logsf
    if name == "LogLogisticLSS":
        return stats.fisk(c=second, scale=first).logsf
    if name == "TwoPieceNormalLSS":
        return lambda t: _two_piece_log_survival(t, first, second, theta[2])
    location = float(family._location_coordinates(np.array([theta]))[0])
    if name == "GeneralizedGammaLSS":
        q = theta[2]
        law = stats.gengamma(
            a=1.0 / q**2, c=q / second, scale=math.exp(location) * (q * q) ** (second / q)
        )
        return law.logsf
    return lambda t: _two_piece_log_survival(math.log(t), location, second, theta[2])


@pytest.mark.parametrize(("family", "theta", "support"), FAMILIES)
def test_each_family_declares_its_support_and_the_protocol(family, theta, support):
    assert isinstance(family, MeanExcessFamily)
    assert family.response_support == support


@pytest.mark.parametrize(("family", "theta", "support"), FAMILIES)
def test_each_family_matches_quadrature_of_its_scipy_survival(family, theta, support):
    """``int_0^inf S(v + s)/S(v) ds`` from ``scipy.stats``: an independent parameter map."""
    theta = np.asarray(theta)
    mean = float(family.default_prediction(theta[None, :])[0])
    for factor in (0.3, 1.0, 2.5, 6.0):
        v = factor * abs(mean)
        got = family.mean_excess(np.array([v]), theta[None, :])[0]
        reference = _quadrature_reference(_scipy_log_survival(family, theta), v, got)
        # Quadrature, not the kernel, sets this tolerance.
        assert got == pytest.approx(reference, rel=1e-10)


@pytest.mark.parametrize(
    ("family", "theta", "support"), [case for case in FAMILIES if case.values[2] != REAL_LINE]
)
def test_each_family_reads_the_mean_less_the_amount_below_its_support(family, theta, support):
    rows = np.tile(theta, (3, 1))
    amount = np.array([-2.5, 0.0, -1e-300])
    got = family.mean_excess(amount, rows)
    np.testing.assert_allclose(got, family.default_prediction(rows) - amount, rtol=16.0 * U)


@pytest.mark.parametrize(("family", "theta", "support"), FAMILIES)
def test_each_family_evaluates_rows_independently_and_broadcasts_one_amount(family, theta, support):
    base = np.asarray(theta)
    rows = np.array([base, base * np.where(np.arange(len(base)) == 0, 1.5, 1.0), base, base])
    amount = np.array([0.25, 4.0, 4.0, -1.0]) * abs(float(base[0])) + 0.5
    together = family.mean_excess(amount, rows)
    alone = [family.mean_excess(amount[i : i + 1], rows[i : i + 1])[0] for i in range(4)]
    np.testing.assert_array_equal(together, alone)
    np.testing.assert_array_equal(
        family.mean_excess(amount[1], rows), family.mean_excess(np.full(4, amount[1]), rows)
    )
    assert together[1] != together[2]
    assert not together.flags.writeable


@pytest.mark.parametrize(("family", "theta", "support"), FAMILIES)
@pytest.mark.parametrize("amount", [math.nan, math.inf, -math.inf])
def test_each_family_refuses_an_amount_that_is_not_finite(family, theta, support, amount):
    with pytest.raises(ValueError, match="mean excess amounts must be finite"):
        family.mean_excess(np.array([1.0, amount]), np.tile(theta, (2, 1)))


# ------------------------------------------------------------------ laws no fit produces

_AMOUNTS = [5e-324, 1e-315, 1e-300, 1e-100, 1e-8, 0.5, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 1.1, 2.0]
_AMOUNTS += [5.0, 1e3, 1e8, 1e100, 1e250, 1e308]
_SCALES = [1e-300, 1e-12, 1e-9, 1e-6, 25.0, 60.0, 1e3]
_SKEWS = [-0.99999999, -0.88, 0.0, 0.9, 0.99999999]
_SHAPES = [1e-12, 1e-9, 1.0000001e-8, 1e-6, 1e-4, 50.0, 1e3]
_WIDE = {"scale_floor": 0.0, "skew_bound": 0.999999999}
EXTREMES = [
    pytest.param(
        GaussianLS(scale_floor=0.0),
        [(m, s) for m in (0.0, 1.0, -1e6, 1e300) for s in [*_SCALES, 1e300]],
        id="gaussian",
    ),
    pytest.param(
        TwoPieceNormalLSS(**_WIDE),
        [(m, s, e) for m in (0.0, 1.0) for s in _SCALES for e in _SKEWS],
        id="two-piece-normal",
    ),
    pytest.param(
        LogNormalLS(parametrisation="location", scale_floor=0.0),
        [(m, s) for m in (0.0, 1.0, -700.0, 700.0, 1e4) for s in _SCALES],
        id="log-normal",
    ),
    pytest.param(
        TwoPieceLogNormalLSS(parametrisation="location", **_WIDE),
        [(m, s, e) for m in (0.0, 1.0, -700.0, 700.0) for s in _SCALES for e in _SKEWS],
        id="two-piece-log-normal",
    ),
    pytest.param(
        GammaLS(),
        [(m, cv) for m in (1e-300, 1.0, 3.0, 1e300) for cv in (1e-150, 1e-100, 1e-8, 1e-6, 1e-4)],
        id="gamma",
    ),
    pytest.param(
        GeneralizedGammaLSS(parametrisation="location", scale_floor=0.0),
        [
            (m, s, sign * q)
            for m in (0.0, 1.0, -700.0, 700.0)
            for s in [1e-12, 1e-9, 1e-6, 0.3, 25.0, 60.0]
            for q in [*_SHAPES, 0.3, 1e8]
            for sign in (1.0, -1.0)
        ],
        id="generalised-gamma",
    ),
    pytest.param(
        LogLogisticLSS(),
        [(s, a) for s in (1e-300, 1e-100, 1.0, 1e300) for a in (1e8, 1e12, 1e16, 1e300)],
        id="log-logistic",
    ),
]


@pytest.mark.parametrize(("family", "laws"), EXTREMES)
def test_a_law_no_fit_produces_gives_no_crash_no_nan_and_no_negative_value(family, laws):
    """CV at or below ``1e-6``, shape at or above ``1e8``, log-scale above 20, subnormal amounts."""
    amounts = np.array(_AMOUNTS)
    if family.response_support.lower is None:
        amounts = np.concatenate((amounts, -amounts))
    for law in laws:
        with np.errstate(all="raise", under="ignore"):
            got = family.mean_excess(amounts, np.tile(law, (len(amounts), 1)))
        assert not np.any(np.isnan(got)), (law, amounts[np.isnan(got)])
        assert np.all(got >= 0.0), (law, amounts[got < 0.0])


@pytest.mark.parametrize(
    "got",
    [
        # Log-normal sigma 1e-9 at v = 1.1: 1.154e-17, lost when the panel ends coincide.
        lambda: kernel.log_normal_mean_excess(*_column(1.1, 0.0, 1e-9))[0],
        # Log-normal sigma 1 at a subnormal amount: E[Y] = 1.6487, where expm1(I) overflows.
        lambda: kernel.log_normal_mean_excess(*_column(1e-315, 0.0, 1.0))[0],
        # Two-piece log-normal, left scale 47: exp(sigma_l^2/2) overflows; the excess is 15.4.
        lambda: kernel.two_piece_log_normal_mean_excess(
            *_column(math.exp(-47.0), 0.0, 25.0, -0.88)
        )[0],
        # Gamma shape 1e16, 1.5 standard deviations up: 4.39e7 in units of the scale.
        lambda: kernel._gamma_excess(np.array([1e16]), np.array([1.5e8 + 4.0]))[0][0],
        # Log-logistic shape 1e16 at twice the scale: 2e-16.
        lambda: kernel.log_logistic_mean_excess(*_column(2.0, 1.0, 1e16))[0],
    ],
    ids=["narrow-log-normal", "subnormal-amount", "wide-left-piece", "gamma-1e16", "fisk-1e16"],
)
def test_the_reproduced_extreme_cases_are_positive_and_finite(got):
    """Each returned zero, ``inf``, an overflow error or a wrong value in the first design."""
    value = got()
    assert math.isfinite(value) and value > 0.0
