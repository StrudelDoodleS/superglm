"""SymPy derivative oracle for every family's published derivative channels.

Each family's per-row log-density is written symbolically, differentiated by
``sympy.diff`` and evaluated with mpmath at 50 significant digits at the same
float64 points the kernels see.  The kernels' first and second derivatives
(``evaluate_natural``) are checked in the natural parameters, the analytic
curvature directions (the third derivatives the LAML endgame uses) in the
predictors through each family's default links, and the single-predictor
families' IRLS weight and its two eta-derivatives through their links.

The Tweedie density is a series in the compound-Poisson count.  Its oracle
differentiates one generic term symbolically, sums the term derivatives over
every term above 1e-60 of the largest, and takes log-sum derivatives by the
quotient rule, so it shares nothing with the kernel's anchored two-pass
cumulants.

Where a kernel states its error, the channel is held to that statement,
derived in the unit roundoff ``u`` from the channel's absolute-term scale
formed in mpmath from the kernel's own float64 inputs (the "derived bounds"
section): the two-piece location direction (``gamma_8`` per entry), the
log-normal mean direction (``gamma_16``), the NB2 direction (``gamma_11``
closed forms and the certified ``l_bbb`` bound) and the shared chain rule
(``gamma_{2K+12}``).  These hold on any IEEE binary64 host.

The family-level channels (scores, Hessians and predictor directions through
``evaluate_natural`` and the links) have no stated bound: no score or Hessian
kernel states one, and every predictor direction inherits their error through
the chain rule, as do the Tweedie series channels and the generalized gamma
mean loading.  They are held to their worst error over these grids, in ``u``
of the row's channel scale (the largest oracle value of that derivative order
in that row), times four (``_MARGIN``), with a floor of ``4u``.  Those values
were measured on one host, x86-64 Linux on an AMD EPYC 9645 (Zen 5, AVX-512)
with the locked NumPy and SciPy wheels; the native Windows, macOS ARM64 and
Linux ARM64 CI jobs pass them at 7305fe25, but a special function rounding
differently elsewhere can move a measurement, and a failure there is a
re-measurement, not by itself a kernel defect.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
mp = pytest.importorskip("mpmath")

from superglm.distributional.families import (  # noqa: E402
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogNormalLS,
    NegativeBinomialLS,
    TweedieLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import COMPLETE_OBSERVATION  # noqa: E402
from superglm.distributional.kernels.two_piece import (  # noqa: E402
    location_of_mean as two_piece_location_of_mean,
)
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights  # noqa: E402

_U = 2.0**-53
_MARGIN = 4
_DIGITS = 50


def _mpf(value: float):
    return mp.mpf(float(value))


def _link_expression(link, eta):
    name = type(link).__name__
    if name == "LogLink":
        return sp.exp(eta)
    if name == "IdentityLink":
        return eta
    if name == "LowerBoundedLogLink":
        return sp.Float(link.floor, _DIGITS) + sp.exp(eta)
    if name in ("BoundedLogitLink", "BoundedPowerLink"):
        lower, upper = sp.Float(link.lower, _DIGITS), sp.Float(link.upper, _DIGITS)
        return lower + (upper - lower) / (1 + sp.exp(-eta))
    raise AssertionError(name)


class _SymbolicRow:
    """First, second and third partial derivatives of one symbolic row, lambdified."""

    def __init__(self, expression, symbols, extra=()):
        self.symbols = tuple(symbols)
        arguments = (*self.symbols, *extra)
        k = len(symbols)
        self.first = [sp.lambdify(arguments, sp.diff(expression, s), "mpmath") for s in symbols]
        self.second = {
            (a, b): sp.lambdify(arguments, sp.diff(expression, symbols[a], symbols[b]), "mpmath")
            for a in range(k)
            for b in range(a, k)
        }
        self.third = {
            triple: sp.lambdify(
                arguments, sp.diff(expression, *(symbols[i] for i in triple)), "mpmath"
            )
            for triple in itertools.combinations_with_replacement(range(k), 3)
        }

    def evaluate(self, point, extra=()):
        values = [_mpf(v) for v in (*point, *extra)]
        with mp.workdps(_DIGITS):
            return (
                [f(*values) for f in self.first],
                {key: f(*values) for key, f in self.second.items()},
                {key: f(*values) for key, f in self.third.items()},
            )


def _third(third, left, right, axis):
    return third[tuple(sorted((left, right, axis)))]


def _error_in_u(computed, oracle) -> float:
    scale = max(abs(value) for value in oracle)
    if scale == 0:
        return max(abs(float(value)) for value in computed) / np.finfo(float).tiny
    return float(max(abs(_mpf(c) - o) for c, o in zip(computed, oracle, strict=True)) / scale) / _U


# ---------------------------------------------------------------- the nine LSS families

_Y = sp.Symbol("y", real=True)


def _densities():
    """Symbolic optimizing log-density in natural parameters, per family."""
    a, b, c = sp.symbols("a b c", real=True)
    y = _Y
    gaussian = -sp.log(b) - (y - a) ** 2 / (2 * b**2)
    shape = 1 / b**2
    gamma = (
        shape * sp.log(shape)
        - shape * sp.log(a)
        + (shape - 1) * sp.log(y)
        - shape * y / a
        - sp.loggamma(shape)
    )
    lognormal_location = -sp.log(b) - (sp.log(y) - a) ** 2 / (2 * b**2)
    lognormal_mean = lognormal_location.subs(a, sp.log(a) - b**2 / 2)
    nb = (
        sp.loggamma(y + b)
        - sp.loggamma(b)
        + b * sp.log(b)
        + y * sp.log(a)
        - (y + b) * sp.log(a + b)
    )
    gp = -sp.log(a) - (1 + 1 / b) * sp.log(1 + b * y / a)

    def two_piece(variate, location, side):
        s = 1 - c if side < 0 else 1 + c
        return -sp.log(b) - ((variate - location) / (b * s)) ** 2 / 2

    def two_piece_mean(side):
        a1, a2 = b * (1 - c), b * (1 + c)
        phi = lambda x: sp.erfc(-x / sp.sqrt(2)) / 2  # noqa: E731
        log_k = sp.log(
            (1 - c) * sp.exp(a1**2 / 2) * phi(-a1) + (1 + c) * sp.exp(a2**2 / 2) * phi(a2)
        )
        return two_piece(sp.log(y), sp.log(a) - log_k, side)

    def gengamma(location):
        k = 1 / c**2
        w = (sp.log(y) - location) / b
        return (
            sp.log(c**2) / 2
            - sp.log(b)
            + k * sp.log(k)
            + k * (c * w - sp.exp(c * w))
            - sp.loggamma(k)
        )

    log_c = (2 * b / c) * sp.log(c**2) / 2 + sp.loggamma(1 / c**2 + b / c) - sp.loggamma(1 / c**2)
    return {
        "gaussian": (gaussian, (a, b)),
        "gamma": (gamma, (a, b)),
        "lognormal_mean": (lognormal_mean, (a, b)),
        "lognormal_location": (lognormal_location, (a, b)),
        "nb": (nb, (a, b)),
        "gp": (gp, (a, b)),
        "twopiece_normal": (lambda side: two_piece(y, a, side), (a, b, c)),
        "twopiece_lognormal_location": (lambda side: two_piece(sp.log(y), a, side), (a, b, c)),
        "twopiece_lognormal_mean": (two_piece_mean, (a, b, c)),
        "gengamma_location": (gengamma(a), (a, b, c)),
        "gengamma_mean": (gengamma(sp.log(a) - log_c), (a, b, c)),
    }


# (family, density key, rows of (y, natural parameters)); each grid reaches the
# edges its hand formulas treat specially.
_LSS_CASES = {
    "gaussian": (GaussianLS(), [(0.3, (1.1, 0.02)), (-4.0, (2.0, 0.7)), (9.5, (-3.0, 40.0))]),
    "gamma": (GammaLS(), [(1.3, (2.0, 0.05)), (0.2, (5.0, 0.9)), (40.0, (3.0, 3.0))]),
    "lognormal_mean": (
        LogNormalLS(),
        [(3.0, (2.0, 0.05)), (0.4, (7.0, 1.2)), (900.0, (5.0, 30.0))],
    ),
    "lognormal_location": (
        LogNormalLS(parametrisation="location"),
        [(3.0, (0.8, 0.05)), (0.4, (1.9, 1.2)), (900.0, (-1.0, 30.0))],
    ),
    # Near Poisson (theta/mu to 1e12) the series route; small theta the recurrence.
    "nb": (
        NegativeBinomialLS(),
        [
            (0.0, (0.3, 1e12)),
            (3.0, (0.5, 1e9)),
            (2.0, (0.07, 1.5)),
            (7.0, (4.0, 0.3)),
            (1.0, (2.0, 40.0)),
        ],
    ),
    # Shape from 1e-3 (series in z) to 0.95 (tail form), excess at zero and far out.
    "gp": (
        GeneralizedParetoLSS(),
        [(0.0, (3.0, 0.4)), (2.0, (5.0, 1e-3)), (900.0, (40.0, 0.3)), (3e4, (2.0, 0.95))],
    ),
    # Both pieces, skew near either bound.
    "twopiece_normal": (
        TwoPieceNormalLSS(),
        [(0.5, (1.0, 0.7, 0.6)), (2.5, (1.0, 0.7, -0.85)), (-3.0, (0.2, 2.0, 0.88))],
    ),
    "twopiece_lognormal_location": (
        TwoPieceLogNormalLSS(parametrisation="location"),
        [(2.0, (1.0, 0.7, 0.6)), (9.0, (1.0, 0.7, -0.85))],
    ),
    "twopiece_lognormal_mean": (
        TwoPieceLogNormalLSS(),
        [(2.0, (4.0, 0.7, 0.6)), (9.0, (3.0, 1.5, -0.85)), (40.0, (30.0, 0.05, 0.3))],
    ),
    # Large k (Q = 0.02, k = 2500), the k = 30 switch region, small k (Q = 1.3)
    # and negative Q with a mean (sigma |Q| < 1).
    "gengamma_location": (
        GeneralizedGammaLSS(parametrisation="location"),
        [
            (3.0, (1.0, 0.6, 0.02)),
            (3.0, (0.7, 0.4, 0.17)),
            (0.5, (0.2, 1.1, 1.3)),
            (8.0, (1.5, 0.5, -0.3)),
        ],
    ),
    "gengamma_mean": (
        GeneralizedGammaLSS(),
        [
            (3.0, (2.5, 0.6, 0.02)),
            (3.0, (2.0, 0.4, 0.17)),
            (0.5, (1.5, 1.1, 1.3)),
            (8.0, (5.0, 0.5, -0.3)),
        ],
    ),
}

# Measured worst error in u of each row's channel scale over these grids, as
# (score, Hessian, third), on x86-64 Linux (AMD EPYC 9645, Zen 5); each is held
# to _MARGIN times its measurement.  No kernel states a bound for these
# family-level channels (see the module docstring); the derived-bound tests
# below hold the kernels that do.
_MEASURED = {
    "gamma": (1.9, 2.2, 4.4),
    "gaussian": (2.8, 3.2, 2.0),
    "gengamma_location": (7.8, 9.4, 25.1),
    "gengamma_mean": (7.1, 4.4, 156.0),
    "gp": (1.8, 4.8, 20.2),
    "lognormal_location": (6.4, 7.0, 5.5),
    "lognormal_mean": (4.2, 4.2, 85.4),
    "nb": (0.8, 1.9, 31.0),
    "twopiece_lognormal_location": (2.2, 2.4, 3.0),
    "twopiece_lognormal_mean": (1.3, 24.2, 260.8),
    "twopiece_normal": (2.5, 2.6, 33.9),
    # Tweedie's series (anchored cumulants) cancels its canonical part in the
    # p channels: interior rows (p <= 1.62) lose up to 2e-14 of the Hessian
    # scale and 1e-13 of the direction scale.  At p = 1.94 the published p-p
    # Hessian itself is good to 3.4e-12 of its row scale (2.2e-11 of its value)
    # and the direction inherits that; this is v0.40.0's Hessian, recorded
    # here, not loosened.
    "tweedie": (55.1, 202.9, 903.0),
    "tweedie_upper_wall": (427.9, 31014.9, 16500.5),
}


def _plan(family, y):
    resolved = resolve_likelihood_weights(
        None, n_observations=len(y), contract=WeightContract("prior")
    )
    return family.bind_likelihood(np.asarray(y, dtype=float), resolved, COMPLETE_OBSERVATION)


def _row_oracles(name, y_value, theta, symbols, eta_symbols, links):
    """(natural, predictor) evaluators for one row of a closed-form family."""
    density = _densities()[name][0]
    if callable(density):  # two-piece: the piece is fixed by the residual's sign
        location = theta[0]
        if "mean" in name:
            location = float(two_piece_location_of_mean(*(np.array([v]) for v in theta))[0])
        residual = (np.log(y_value) if "lognormal" in name else y_value) - location
        density = density(-1 if residual < 0 else 1)
    expression = density.subs(_Y, sp.Float(y_value, _DIGITS))
    natural = _SymbolicRow(expression, symbols)
    substitution = {
        s: _link_expression(link, e) for s, link, e in zip(symbols, links, eta_symbols, strict=True)
    }
    return natural, _SymbolicRow(expression.subs(substitution), eta_symbols)


def _measure(family, rows, oracles) -> list[float]:
    """Worst error in u of the score, Hessian and curvature directions over ``rows``."""
    links = tuple(spec.default_link for spec in family.parameters)
    worst = [0.0, 0.0, 0.0]
    for (y_value, theta), (natural, predictor) in zip(rows, oracles, strict=True):
        first, second, _ = natural.evaluate(theta)
        y = np.array([y_value])
        plan = _plan(family, y)
        evaluated = family.evaluate_natural(y, np.array([theta]), plan, derivative_order=2)
        pairs = list(second)
        worst[0] = max(worst[0], _error_in_u(evaluated.score[0], first))
        worst[1] = max(
            worst[1], _error_in_u(evaluated.hessian_packed[0], [second[p] for p in pairs])
        )
        eta = np.array(
            [[float(link.link(np.array([v]))[0]) for link, v in zip(links, theta, strict=True)]]
        )
        _, _, third = predictor.evaluate(eta[0])
        for axis in range(len(theta)):
            direction = np.zeros_like(eta)
            direction[0, axis] = 1.0
            computed = family.predictor_curvature_directional_derivative(
                y, eta, direction, links, plan
            )[0]
            oracle = [-_third(third, left, right, axis) for left, right in pairs]
            worst[2] = max(worst[2], _error_in_u(computed, oracle))
    return worst


def _closed_form_check(name: str):
    family, rows = _LSS_CASES[name]
    symbols = _densities()[name][1]
    eta_symbols = sp.symbols(f"e0:{len(symbols)}", real=True)
    links = tuple(spec.default_link for spec in family.parameters)
    oracles = [_row_oracles(name, y, theta, symbols, eta_symbols, links) for y, theta in rows]
    return _measure(family, rows, oracles)


def _assert_within(name: str, worst) -> None:
    for order, (error, measured) in enumerate(zip(worst, _MEASURED[name], strict=True)):
        assert error <= _MARGIN * max(measured, 1.0), (name, order, error)


@pytest.mark.parametrize("name", sorted(_LSS_CASES))
def test_lss_derivatives_match_the_symbolic_oracle(name: str) -> None:
    _assert_within(name, _closed_form_check(name))


# ---------------------------------------------------------------- Tweedie (series)


class _TweedieRow:
    """Tweedie row derivatives in ``coordinates`` (natural or predictor).

    ``l = b - log y + log S`` with ``b = (-y mu^(1-p)/(p-1) - mu^(2-p)/(2-p))/phi``
    and ``S = sum_j exp(L_j)``, ``L_j = j zeta - lgamma(j+1) - lgamma(alpha j)``;
    the zero row is ``-mu^(2-p)/(phi (2-p))``.  One generic ``L_j`` is
    differentiated symbolically and summed numerically; ``log S`` follows by the
    quotient rule.
    """

    def __init__(self, y_value, coordinates, mu, phi, p):
        self.y = y_value
        j = sp.Symbol("j", positive=True)
        r, s = p - 1, 2 - p
        y = sp.Float(y_value, _DIGITS)
        if y_value == 0:
            self.closed = _SymbolicRow(-(mu**s) / (phi * s), coordinates)
            return
        self.closed = _SymbolicRow((-y * mu ** (-r) / r - mu**s / s) / phi - sp.log(y), coordinates)
        alpha = s / r
        zeta = alpha * (sp.log(y) - sp.log(r)) - sp.log(s) - sp.log(phi) / r
        term = j * zeta - sp.loggamma(j + 1) - sp.loggamma(alpha * j)
        self.term = _SymbolicRow(term, coordinates, extra=(j,))
        self.value = sp.lambdify((*coordinates, j), term, "mpmath")

    def evaluate(self, point):
        first, second, third = self.closed.evaluate(point)
        if self.y == 0:
            return first, second, third
        k = len(point)
        values = [_mpf(v) for v in point]
        with mp.workdps(_DIGITS):
            logs, j, top = [], 1, None
            while True:  # every term above exp(-140) of the largest
                value = self.value(*values, j)
                logs.append(value)
                top = value if top is None or value > top else top
                if j > 3 and value < top - 140:
                    break
                j += 1
            s0, s1 = mp.mpf(0), [mp.mpf(0)] * k
            s2 = {key: mp.mpf(0) for key in second}
            s3 = {key: mp.mpf(0) for key in third}
            for index, value in enumerate(logs, start=1):
                weight = mp.exp(value - top)
                l1, l2, l3 = self.term.evaluate(point, extra=(index,))
                s0 += weight
                s1 = [acc + weight * d for acc, d in zip(s1, l1, strict=True)]
                for a, b in s2:
                    s2[a, b] += weight * (l2[a, b] + l1[a] * l1[b])
                for a, b, c in s3:
                    s3[a, b, c] += weight * (
                        l3[a, b, c]
                        + l2[a, b] * l1[c]
                        + l2[a, c] * l1[b]
                        + l2[b, c] * l1[a]
                        + l1[a] * l1[b] * l1[c]
                    )
            m1 = [v / s0 for v in s1]
            for index in range(k):
                first[index] += m1[index]
            for a, b in s2:
                second[a, b] += s2[a, b] / s0 - m1[a] * m1[b]
            for a, b, c in s3:
                third[a, b, c] += (
                    s3[a, b, c] / s0
                    - (s2[a, b] * m1[c] + s2[a, c] * m1[b] + s2[b, c] * m1[a]) / s0
                    + 2 * m1[a] * m1[b] * m1[c]
                )
        return first, second, third


# Zero and positive rows, small to large phi; the power walls of [1.05, 1.95]
# are their own group because the published p-p channel cancels there (below).
_TWEEDIE_ROWS = {
    "tweedie": [
        (0.0, (3.0, 2.0, 1.06)),
        (0.5, (1.5, 0.6, 1.3)),
        (30.0, (12.0, 4.0, 1.62)),
        (2000.0, (300.0, 900.0, 1.06)),
    ],
    "tweedie_upper_wall": [(0.0, (0.2, 40.0, 1.94)), (7.0, (2.0, 3.0, 1.94))],
}


def _tweedie_check(name: str):
    family = TweedieLSS()
    natural = sp.symbols("mu phi p", positive=True)
    eta = sp.symbols("e0:3", real=True)
    links = tuple(spec.default_link for spec in family.parameters)
    mapped = [_link_expression(link, e) for link, e in zip(links, eta, strict=True)]
    rows = _TWEEDIE_ROWS[name]
    oracles = [(_TweedieRow(y, natural, *natural), _TweedieRow(y, eta, *mapped)) for y, _ in rows]
    return _measure(family, rows, oracles)


@pytest.mark.parametrize("name", sorted(_TWEEDIE_ROWS))
def test_tweedie_derivatives_match_the_symbolic_series_oracle(name: str) -> None:
    _assert_within(name, _tweedie_check(name))


# ---------------------------------------------------------------- certified NB2 bound


def test_nb2_theta_third_derivative_is_inside_its_certified_bound() -> None:
    """``l_bbb`` cancels by theta^2 near Poisson; its error stays inside the bound it returns."""
    from superglm.distributional.kernels.negative_binomial import (
        negative_binomial_log_third_derivatives,
    )

    a, b = sp.symbols("a b", real=True)
    for y_value, mean, theta in [
        (0.0, 0.3, 1e12),
        (3.0, 0.5, 1e9),
        (5.0, 1.0, 2.0**40),
        (2.0, 0.07, 1.5),
        (7.0, 4.0, 0.3),
        (40.0, 35.0, 1e4),
    ]:
        mu, size = sp.exp(a), sp.exp(b)
        density = (
            sp.loggamma(y_value + size)
            - sp.loggamma(size)
            + size * sp.log(size)
            + y_value * sp.log(mu)
            - (y_value + size) * sp.log(mu + size)
        )
        third = _SymbolicRow(density, (a, b)).evaluate((np.log(mean), np.log(theta)))[2]
        computed, bound = negative_binomial_log_third_derivatives(
            np.array([y_value]),
            np.exp(np.array([np.log(mean)])),
            np.exp(np.array([np.log(theta)])),
            np.ones(1),
            "prior",
        )
        assert abs(_mpf(computed[0, 3]) - third[1, 1, 1]) <= _mpf(bound[0])


# ---------------------------------------------------------------- single-predictor families


def _single_cases():
    from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
    from superglm.links import IdentityLink, LogitLink, LogLink

    eta = sp.Symbol("eta", real=True)
    exp, logistic = sp.exp(eta), 1 / (1 + sp.exp(-eta))
    return eta, {
        "poisson/log": (Poisson(), LogLink(), exp, lambda m: m),
        "gamma/log": (Gamma(), LogLink(), exp, lambda m: m**2),
        "tweedie/log": (Tweedie(1.4), LogLink(), exp, lambda m: m ** sp.Float(1.4, _DIGITS)),
        "nb/log": (
            NegativeBinomial(2.5),
            LogLink(),
            exp,
            lambda m: m + m**2 / sp.Float(2.5, _DIGITS),
        ),
        "binomial/logit": (Binomial(), LogitLink(), logistic, lambda m: m * (1 - m)),
        "gaussian/identity": (Gaussian(), IdentityLink(), eta, lambda m: sp.Integer(1)),
    }


# Measured worst error in u of (W, dW/deta, d2W/deta2) over the grid below;
# Gamma/log and Gaussian/identity are exact (constant W).
_SINGLE_MEASURED = {
    "poisson/log": (0.3, 0.3, 1.2),
    "gamma/log": (0.0, 0.0, 0.0),
    "tweedie/log": (0.2, 3.6, 3.1),
    "nb/log": (0.6, 11.2, 79.4),
    "binomial/logit": (0.6, 3.2, 24.8),
    "gaussian/identity": (0.0, 0.0, 0.0),
}


@pytest.mark.parametrize(
    "name",
    ["poisson/log", "gamma/log", "tweedie/log", "nb/log", "binomial/logit", "gaussian/identity"],
)
def test_single_predictor_weight_channels_match_the_symbolic_oracle(name: str) -> None:
    """``W = g'^2/V(g)`` and its first two eta-derivatives, against ``sympy.diff``."""
    from superglm.reml.w_derivatives import compute_d2W_deta2, compute_dW_deta
    from superglm.solvers.working_rows import fisher_working_weights

    eta, cases = _single_cases()
    distribution, link, inverse, variance = cases[name]
    weight = sp.diff(inverse, eta) ** 2 / variance(inverse)
    oracle = [
        sp.lambdify(eta, sp.diff(weight, eta, order) if order else weight, "mpmath")
        for order in range(3)
    ]
    grid = np.array([-4.0, -1.3, -0.2, 0.0, 0.7, 2.1, 4.5])
    if name == "gaussian/identity":
        grid = grid * 10.0
    mu = link.inverse(grid)
    ones = np.ones_like(grid)
    computed = (
        fisher_working_weights(
            distribution=distribution, link=link, mu=mu, eta=grid, sample_weight=ones
        ),
        compute_dW_deta(link, distribution, mu, grid, ones),
        compute_d2W_deta2(link, distribution, mu, grid, ones),
    )
    worst = []
    for values, function in zip(computed, oracle, strict=True):
        with mp.workdps(_DIGITS):
            exact = [function(_mpf(value)) for value in grid]
        scale = max(abs(v) for v in exact)
        errors = [abs(_mpf(c) - e) for c, e in zip(values, exact, strict=True)]
        worst.append(float(max(errors) / scale) / _U if scale else float(max(errors)))
    measured = _SINGLE_MEASURED[name]
    for error, bound in zip(worst, measured, strict=True):
        assert error <= _MARGIN * max(bound, 1.0), (name, worst)


# ---------------------------------------------------------------- mutation


def test_a_wrong_coefficient_fails_the_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generalized Pareto E'(z) series with one coefficient off by (j+4) -> (j+5) is caught."""
    import superglm.distributional.kernels.generalized_pareto as kernel

    def wrong_series_f(z):
        return kernel._piecewise(
            z, lambda j: (-1.0) ** j * (j + 1) * (j + 2) * (j + 3) / (j + 5), lambda a: a
        )

    _assert_within("gp", _closed_form_check("gp"))
    monkeypatch.setattr(kernel, "series_f", wrong_series_f)
    with pytest.raises(AssertionError):
        _assert_within("gp", _closed_form_check("gp"))


# ---------------------------------------------------------------- derived bounds
#
# Where a kernel states its error, the channel is held to that statement: gamma_k
# of the channel's absolute-term scale, the scale formed here in mpmath from the
# same float64 inputs the kernel receives, so the bound is independent of the host.


def _gamma(k: int):
    return mp.mpf(k) * mp.mpf(_U) / (1 - mp.mpf(k) * mp.mpf(_U))


_TWO_PIECE_DIRECTION_ROWS = [
    # (t, (mu, sigma, eps)): both pieces, a residual 2^-40 of sigma on either
    # side of the join, and skew near either bound.
    (0.5, (1.0, 0.7, 0.6)),
    (2.5, (1.0, 0.7, -0.85)),
    (-3.0, (0.2, 2.0, 0.88)),
    (1.0 + 2.0**-40, (1.0, 0.9, 0.4)),
    (1.0 - 2.0**-40, (1.0, 0.9, 0.4)),
    (40.0, (-3.0, 0.05, -0.2)),
]


def test_two_piece_location_direction_is_within_its_stated_bound() -> None:
    """Each ``T`` entry is within ``gamma_8`` of its term scale; a channel sums three.

    The channel is ``m sum_k T_k v_k``: the entries' ``gamma_8``, one rounding
    per product with ``v``, two additions and the multiplier, so ``gamma_12`` of
    ``m sum_k S(T_k) |v_k|``.  ``S(T)`` is ``|T|`` for the monomials and
    ``(12 u^2 + 2)/sigma^3`` for ``T_sss = (12 u^2 - 2)/sigma^3``.  The oracle is
    the residual form at the float64 ``t`` the kernel receives.
    """
    from superglm.distributional.kernels.two_piece import location_hessian_direction

    a, b, c = sp.symbols("a b c", real=True)
    t_symbol = sp.Symbol("t", real=True)
    direction = np.array([[0.7, -1.3, 0.4]])
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for t_value, theta in _TWO_PIECE_DIRECTION_ROWS:
        side = -1 if t_value - theta[0] < 0 else 1
        s = 1 - c if side < 0 else 1 + c
        density = -sp.log(b) - ((t_symbol - a) / (b * s)) ** 2 / 2
        third = _SymbolicRow(density.subs(t_symbol, sp.Float(t_value, _DIGITS)), (a, b, c))
        third = third.evaluate(theta)[2]
        computed = location_hessian_direction(
            np.array([t_value]), *(np.array([v]) for v in theta), direction, np.ones(1)
        )[0]
        with mp.workdps(_DIGITS):
            mu, sigma, eps = (_mpf(v) for v in theta)
            u = (_mpf(t_value) - mu) / (sigma * (1 + side * eps))
            for column, (left, right) in enumerate(pairs):
                exact = mp.mpf(0)
                scale = mp.mpf(0)
                for axis in range(3):
                    entry = _third(third, left, right, axis)
                    entry_scale = (
                        (12 * u**2 + 2) / sigma**3
                        if (left, right, axis) == (1, 1, 1)
                        else abs(entry)
                    )
                    exact += entry * _mpf(direction[0, axis])
                    scale += entry_scale * abs(_mpf(direction[0, axis]))
                assert abs(_mpf(computed[column]) - exact) <= _gamma(12) * scale, (
                    t_value,
                    theta,
                    column,
                )


_LOG_NORMAL_DIRECTION_ROWS = [
    (3.0, (np.log(2.0), np.log(0.05 - 0.01))),
    (0.4, (np.log(7.0), np.log(1.2 - 0.01))),
    (900.0, (np.log(5.0), np.log(30.0 - 0.01))),
    # The residual crosses zero and the scale sits just above its floor.
    (np.exp(np.log(2.0) - 0.5 * 0.3**2), (np.log(2.0), np.log(0.3 - 0.01))),
    (1.5, (0.2, -30.0)),
]


def test_log_normal_mean_direction_is_within_its_stated_bound() -> None:
    """``gamma_16`` of each channel's absolute-term scale, as the kernel states.

    With ``kappa = e^b``, ``sigma = floor + kappa``, ``u = kappa/sigma``,
    ``f = floor/sigma`` and the residual's terms ``R = |log y| + |a| + sigma^2/2``,
    the factors' scales are ``2u/sigma^2`` (``l_aab``),
    ``u (floor + 2 kappa)/sigma (1 + 2R/sigma^2)`` (``l_abb``) and
    ``u f (floor + kappa)/sigma + 3 kappa^2 f + u (1 + 9u + 12u^2)((R/sigma)^2 + R)``
    (``l_bbb``); a channel's scale is ``m`` times the sum of its factors' scales
    times ``|d|``.  The oracle differentiates the mean-form density in the predictors.
    """
    from superglm.distributional.kernels.log_normal import mean_curvature_directional

    family = LogNormalLS()
    floor = family.scale_floor
    links = tuple(spec.default_link for spec in family.parameters)
    symbols = _densities()["lognormal_mean"][1]
    eta_symbols = sp.symbols("e0:2", real=True)
    direction = np.array([[0.9, -1.7]])
    for y_value, eta in _LOG_NORMAL_DIRECTION_ROWS:
        theta = tuple(float(link.inverse(np.array([e]))[0]) for link, e in zip(links, eta))
        _, predictor = _row_oracles("lognormal_mean", y_value, theta, symbols, eta_symbols, links)
        third = predictor.evaluate(eta)[2]
        computed = mean_curvature_directional(
            np.array([y_value]), np.array([eta]), direction, np.ones(1), scale_floor=floor
        )[0]
        with mp.workdps(_DIGITS):
            kappa = mp.exp(_mpf(eta[1]))
            sigma = _mpf(floor) + kappa
            u, f = kappa / sigma, _mpf(floor) / sigma
            residual_scale = abs(mp.log(_mpf(y_value))) + abs(_mpf(eta[0])) + sigma**2 / 2
            s_aab = 2 * u / sigma**2
            s_abb = u * (_mpf(floor) + 2 * kappa) / sigma * (1 + 2 * residual_scale / sigma**2)
            s_bbb = (
                u * f * (_mpf(floor) + kappa) / sigma
                + 3 * kappa**2 * f
                + u * (1 + 9 * u + 12 * u**2) * ((residual_scale / sigma) ** 2 + residual_scale)
            )
            d_a, d_b = (abs(_mpf(v)) for v in direction[0])
            scales = (s_aab * d_b, s_aab * d_a + s_abb * d_b, s_abb * d_a + s_bbb * d_b)
            for column, (left, right) in enumerate([(0, 0), (0, 1), (1, 1)]):
                exact = -sum(
                    _third(third, left, right, axis) * _mpf(direction[0, axis]) for axis in range(2)
                )
                assert abs(_mpf(computed[column]) - exact) <= _gamma(16) * scales[column], (
                    y_value,
                    eta,
                    column,
                )


def test_nb2_direction_is_within_its_closed_form_and_certified_bounds() -> None:
    """``l_aaa``, ``l_aab``, ``l_abb`` within ``gamma_11`` of their scales, ``l_bbb`` certified.

    In the log coordinates the direction is ``-sum_k l_ijk d_k``: the closed forms'
    ``gamma_11``, one rounding per product and one addition give ``gamma_13`` of
    ``sum_k S_ijk |d_k|`` beside the certificate ``|d_b| e`` on ``bb``; the scales
    are ``|l_aaa|``, ``s (|y r| + 2 mu p)`` and ``|l_abb|`` (``q = mu/t``,
    ``p = theta/t``, ``r = (theta - mu)/t``, ``s = qp``, ``t = mu + theta``).
    """
    from superglm.distributional.kernels.negative_binomial import (
        negative_binomial_certified_predictor_curvature_directional,
    )

    a, b = sp.symbols("a b", real=True)
    direction = np.array([[1.1, -0.6]])
    for y_value, (mean, theta) in _LSS_CASES["nb"][1]:
        mu, size = sp.exp(a), sp.exp(b)
        density = (
            sp.loggamma(y_value + size)
            - sp.loggamma(size)
            + size * sp.log(size)
            + y_value * sp.log(mu)
            - (y_value + size) * sp.log(mu + size)
        )
        eta = (float(np.log(mean)), float(np.log(theta)))
        third = _SymbolicRow(density, (a, b)).evaluate(eta)[2]
        computed, certificate = negative_binomial_certified_predictor_curvature_directional(
            np.array([y_value]), np.array([eta]), direction, np.ones(1), "prior"
        )
        with mp.workdps(_DIGITS):
            m, th = mp.exp(_mpf(eta[0])), mp.exp(_mpf(eta[1]))
            total = m + th
            q, p, r = m / total, th / total, (th - m) / total
            s = q * p
            entries = {
                (0, 0, 0): abs(third[0, 0, 0]),
                (0, 0, 1): s * (abs(y_value * r) + 2 * m * p),
                (0, 1, 1): abs(third[0, 1, 1]),
                (1, 1, 1): mp.mpf(0),
            }
            for column, (left, right) in enumerate([(0, 0), (0, 1), (1, 1)]):
                exact = -sum(
                    _third(third, left, right, axis) * _mpf(direction[0, axis]) for axis in range(2)
                )
                scale = sum(
                    entries[tuple(sorted((left, right, axis)))] * abs(_mpf(direction[0, axis]))
                    for axis in range(2)
                )
                bound = _gamma(13) * scale + _mpf(certificate[0, column])
                assert abs(_mpf(computed[0, column]) - exact) <= bound, (y_value, column)


@pytest.mark.parametrize("k_parameters", [2, 3])
def test_chain_rule_is_within_gamma_2k_plus_12(k_parameters: int) -> None:
    """The shared chain rule holds its stated ``gamma_{2K+12}`` of the absolute-term scale.

    Fed float64 inputs, it is compared with the same formula evaluated in
    mpmath on those inputs, so only its own arithmetic is measured; the scale is
    the sum of the absolute values of every product in the formula.
    """
    from superglm.distributional.kernels._curvature_direction import (
        predictor_curvature_direction,
    )

    rng = np.random.default_rng(31 + k_parameters)
    n_rows, k = 200, k_parameters
    channels = k * (k + 1) // 2
    score = rng.normal(size=(n_rows, k)) * np.exp(rng.uniform(-8, 8, (n_rows, k)))
    hessian = rng.normal(size=(n_rows, channels)) * np.exp(rng.uniform(-8, 8, (n_rows, channels)))
    hessian_direction = rng.normal(size=(n_rows, channels))
    first, second, third = (rng.normal(size=(n_rows, k)) for _ in range(3))
    direction = rng.normal(size=(n_rows, k))
    computed = predictor_curvature_direction(
        score, hessian, hessian_direction, first, second, third, direction
    )
    pairs = [(left, right) for left in range(k) for right in range(left, k)]
    index = {pair: column for column, pair in enumerate(pairs)}

    def entry(values, row, left, right):
        return _mpf(values[row, index[(min(left, right), max(left, right))]])

    with mp.workdps(_DIGITS):
        for row in range(n_rows):
            g1 = [_mpf(v) for v in first[row]]
            g2 = [_mpf(v) for v in second[row]]
            g3 = [_mpf(v) for v in third[row]]
            d = [_mpf(v) for v in direction[row]]
            v = [g1[i] * d[i] for i in range(k)]
            for column, (left, right) in enumerate(pairs):
                h = entry(hessian, row, left, right)
                dh = entry(hessian_direction, row, left, right)
                terms = [
                    g1[left] * g1[right] * dh,
                    h * g2[left] * d[left] * g1[right],
                    h * g1[left] * g2[right] * d[right],
                ]
                if left == right:
                    terms += [g2[left] * entry(hessian, row, left, r) * v[r] for r in range(k)]
                    terms.append(_mpf(score[row, left]) * g3[left] * d[left])
                exact = -sum(terms)
                scale = sum(abs(term) for term in terms)
                assert abs(_mpf(computed[row, column]) - exact) <= _gamma(2 * k + 12) * scale
