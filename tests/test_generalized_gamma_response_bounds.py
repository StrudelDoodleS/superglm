"""GeneralizedGammaLSS with truncated and censored rows (issue 544, stage 2).

Fits are checked against direct maximum likelihood in scipy on
``scipy.stats.gengamma`` reparametrised (``a = 1/Q^2``, ``c = Q/sigma``, scale
``e^mu a^(-sigma/Q)``), which evaluates its tails by ``scipy.special.gammainc``
and shares nothing with the fit; the complete-row fit of the same rows is the
unfixed behaviour.  At ``Q = 0`` the bounded law is the log-normal's, checked
against ``LogNormalLS``.

The bounded rows' channels to order four are checked against an mpmath
oracle of the bounded log-likelihood in natural coordinates.  Its log tails
are ``tests/test_gg_uniform_tail.reference``: Temme's ``Q``-expansion of the
density in exact rationals at ``Q = 0`` and small ``|Q| |w|^3``, and
``mp.gammainc`` on a ``Q``-stencil elsewhere, both at 70 digits or more.
Those ``(w, Q)`` Taylor coefficients are carried to ``(mu | m, sigma, Q)`` by
truncated series arithmetic in mpmath (``w = (log y - mu)/sigma``, and for the
mean form ``mu = log m - log C(sigma, Q)`` with ``log C`` differentiated by
``mp.diff`` from ``loggamma``), the window is ``log(T(near) - T(far))`` on those
series, and a density term is ``mp.diff`` of the closed-form log density.  None
of it shares a formula with ``gg_log_tail``, ``compose_derivatives``, the window
combiner or the quadrature.

Error model.  A bounded row's channels carry ``gg_log_tail``'s error in its
``(w, Q)`` partials (``tests/test_gg_log_tail.py``, ``tests/test_gg_uniform_tail.py``:
below ``1e-12`` of the row's same-order scale at orders 0-2 and ``2e-10`` at
order 4 on these rows' shapes), mapped by the chain rule through ``w``, which
rounds each product once, and divided by the window's tail probability, which
multiplies it by ``1 + kappa`` on two-ended windows.  Errors are in ``u`` of
the row's term scale, the largest oracle entry of that order of either term
(density or censored window, and truncation window), as in
``tests/test_gamma_ls_response_bounds.py``; each order is held to its worst
measured error on this grid times four (``_MARGIN``).  Measured on x86-64
Linux (AMD EPYC 9645, Zen 5).
"""

from __future__ import annotations

import functools
import itertools
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from superglm import (
    GeneralizedGammaLSS,
    LogNormalLS,
    SeparationError,
    SuperLSS,
    cat,
    distributional,
    s,
)
from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
from superglm.distributional.kernels import _declared_bounds
from superglm.distributional.kernels.generalized_gamma import (
    generalized_gamma_cdf,
    generalized_gamma_quantile,
    location_of_mean,
    mean_of_location,
)
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink

mp = pytest.importorskip("mpmath")

from tests.test_gg_uniform_tail import expansion_reference, reference  # noqa: E402

_U = 2.0**-53
_MARGIN = 4
_INF = np.inf
_FORMS = ("location", "mean")


def _intercepts(n: int) -> pd.DataFrame:
    return pd.DataFrame({"one": np.ones(n)})


def _model(form: str, **kwargs) -> SuperLSS:
    family = GeneralizedGammaLSS(parametrisation=form)
    first = family.mean() if form == "mean" else family.location()
    return SuperLSS(family, first, family.scale(), family.shape(), **kwargs)


def _fit(form: str, y, **kwargs) -> np.ndarray:
    """Fitted ``(mu, sigma, Q)`` of an intercept-only fit, in location coordinates."""
    model = _model(form, weight_semantics=kwargs.pop("weight_semantics", "prior"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(y)), y, **kwargs)
    first, sigma, q = model.predict_parameters(_intercepts(1)).iloc[0].to_numpy()
    if form == "mean":
        first = location_of_mean(np.array([first]), np.array([sigma]), np.array([q]))[0]
    return np.array([first, sigma, q])


def _law(mu, sigma, q):
    shape = 1.0 / q**2
    return stats.gengamma(shape, q / sigma, scale=np.exp(mu) * shape ** (-sigma / q))


def _draw(rng, n, mu=1.0, sigma=0.6, q=0.5):
    shape = 1.0 / q**2
    return np.exp(mu + sigma * np.log(rng.gamma(shape, 1.0, n) / shape) / q)


def _mle(negative_log_likelihood, start) -> np.ndarray:
    def objective(t):
        return negative_log_likelihood(t[0], np.exp(t[1]), t[2])

    result = optimize.minimize(
        objective,
        (start[0], np.log(start[1]), start[2]),
        method="Nelder-Mead",
        options={"xatol": 1e-11, "fatol": 1e-11, "maxiter": 40_000, "maxfev": 40_000},
    )
    return np.array([result.x[0], np.exp(result.x[1]), result.x[2]])


# -- fits against scipy -------------------------------------------------------


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("kind", ["truncated", "censored"])
def test_intercept_fits_match_direct_maximum_likelihood(form, kind):
    rng = np.random.default_rng(560)
    y = _draw(rng, 3_000)
    if kind == "truncated":
        floor = float(np.quantile(y, 0.3))
        paid, hit = y[y > floor], np.zeros(int(np.sum(y > floor)), dtype=bool)
        bounds = {"truncation": (floor, None)}
    else:
        floor, limit = 0.0, float(np.quantile(y, 0.85))
        paid, hit = np.minimum(y, limit), y >= limit
        bounds = {"censored": hit}

    def nll(mu, sigma, q):
        law = _law(mu, sigma, q)
        rows = law.logpdf(paid)
        if kind == "censored":
            rows = np.where(hit, law.logsf(limit), rows)
        else:
            rows = rows - law.logsf(floor)
        return -math.fsum(rows)

    expected = _mle(nll, (1.2, 0.5, 0.3))
    np.testing.assert_allclose(_fit(form, paid, **bounds), expected, rtol=1e-6)
    # Reading the same rows as complete is the unfixed behaviour, and it is biased.
    complete = _fit(form, paid)
    assert np.max(np.abs(complete - expected) / np.abs(expected)) > 0.02


def test_left_censored_rows_and_frequency_weights_match_scipy():
    rng = np.random.default_rng(561)
    y = _draw(rng, 3_000, mu=0.5, sigma=0.8, q=-0.4)
    weight = rng.integers(1, 4, len(y)).astype(float)
    floor = float(np.quantile(y, 0.2))
    hit = y <= floor
    paid = np.maximum(y, floor)

    def nll(mu, sigma, q):
        law = _law(mu, sigma, q)
        return -math.fsum(weight * np.where(hit, law.logcdf(floor), law.logpdf(paid)))

    fitted = _fit(
        "location",
        paid,
        censored=np.where(hit, "left", "none"),
        sample_weight=weight,
        weight_semantics="frequency",
    )
    np.testing.assert_allclose(fitted, _mle(nll, (0.4, 0.7, -0.3)), rtol=1e-6)


def test_a_generalized_gamma_body_reads_the_burr_layer_better_than_a_log_normal_one():
    """The layer method: the body on ``min(y, u)`` right-censored at ``u``, ``P(Y > u)`` from it.

    Burr XII with ``c = 2``, ``d = 1.25``, scale 1,000, ``u`` its 85% point, on
    20,000 draws.  A direct scipy maximum likelihood on the same rows reads
    ``P(Y > u)`` ``11.27%`` high from a log-normal body and ``2.62%`` low from a
    generalized gamma one.
    """
    truth = stats.burr12(c=2.0, d=1.25, scale=1_000.0)
    y = truth.rvs(20_000, random_state=np.random.default_rng(20261031))
    limit = float(truth.ppf(0.85))
    paid, hit = np.minimum(y, limit), y >= limit
    X = _intercepts(len(y))
    errors = {}
    log_normal, generalized = LogNormalLS(), GeneralizedGammaLSS(parametrisation="location")
    for name, family, model in (
        ("log-normal", log_normal, SuperLSS(log_normal, log_normal.mean(), log_normal.scale())),
        ("generalized gamma", generalized, _model("location")),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, paid, censored=hit)
        theta = model.predict_parameters(_intercepts(1)).to_numpy()
        errors[name] = (1.0 - family.cdf(np.array([limit]), theta)[0]) / 0.15 - 1.0
    assert errors["log-normal"] == pytest.approx(0.1127, abs=2e-4)
    assert errors["generalized gamma"] == pytest.approx(-0.0262, abs=2e-4)


def test_at_zero_shape_the_bounded_law_is_the_log_normal():
    lower = np.array([-_INF, 0.4, 0.4, 2.0, 1.0 - 1e-7])
    upper = np.array([3.0, _INF, 5.0, 2.0 * (1 + 1e-4), 1.0])
    mu, sigma = 0.3, 0.7
    gg = GeneralizedGammaLSS(parametrisation="location").log_window_probability(
        lower, upper, np.tile((mu, sigma, 0.0), (5, 1))
    )
    log_normal = LogNormalLS(parametrisation="location").log_window_probability(
        lower, upper, np.tile((mu, sigma), (5, 1))
    )
    np.testing.assert_allclose(gg, log_normal, rtol=1e-10, atol=1e-10)
    # Score and Hessian in (mu, sigma) on bounded rows, the shape's held at zero.
    y = np.array([2.0, 1.5, 3.0, 2.0, 1.0])
    bounds = ResponseBounds.from_arrays(lower, upper, np.array([0, 1, -1, 1, 0]))
    weights = resolve_likelihood_weights(None, n_observations=5, contract=WeightContract("prior"))
    pairs = {}
    for family, theta in (
        (GeneralizedGammaLSS(parametrisation="location"), np.tile((mu, sigma, 0.0), (5, 1))),
        (LogNormalLS(parametrisation="location"), np.tile((mu, sigma), (5, 1))),
    ):
        plan = family.bind_likelihood(y, weights, bounded_observation(bounds))
        pairs[type(family)] = family.evaluate_natural(y, theta, plan, derivative_order=2)
    gg_rows, ln_rows = pairs[GeneralizedGammaLSS], pairs[LogNormalLS]
    np.testing.assert_allclose(
        gg_rows.optimizing_log_likelihood + gg_rows.parameter_independent_carrier,
        ln_rows.optimizing_log_likelihood + ln_rows.parameter_independent_carrier,
        rtol=1e-10,
    )
    np.testing.assert_allclose(gg_rows.score[:, :2], ln_rows.score, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        gg_rows.hessian_packed[:, [0, 1, 3]], ln_rows.hessian_packed, rtol=1e-10, atol=1e-12
    )
    # A bounded fit on log-normal data finds a shape near zero and the log-normal's law.
    rng = np.random.default_rng(562)
    data = np.exp(rng.normal(1.0, 0.5, 4_000))
    limit = float(np.quantile(data, 0.85))
    paid, hit = np.minimum(data, limit), data >= limit
    fitted = _fit("location", paid, censored=hit)
    log_normal_family = LogNormalLS(parametrisation="location")
    reference_model = SuperLSS(
        log_normal_family, log_normal_family.location(), log_normal_family.scale()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_model.fit(_intercepts(len(paid)), paid, censored=hit)
    expected = reference_model.predict_parameters(_intercepts(1)).iloc[0].to_numpy()
    assert abs(fitted[2]) < 0.1
    np.testing.assert_allclose(fitted[:2], expected, rtol=0.02)


# -- bounded channels against the mpmath oracle -------------------------------

# (y, lower, upper, side, mu, sigma, Q): censored rows at Q = 0 on both sides;
# both sides of the uniform tail's edge |Q| = 0.2887; the body between the
# engines (0.2887 < |Q| < 0.5) near the diagonal, both signs; a far tail at
# small Q outside the uniform domain; Q = 1.5; complete rows truncated on two
# ends and on one; narrow windows on both routes (a truncation window and a
# censored row's observed window).
_ORACLE_ROWS = (
    (2.0, -_INF, _INF, 1, 0.3, 0.7, 0.0),
    (1.0, -_INF, 4.0, -1, 0.3, 0.7, 0.0),
    (1.6, -_INF, _INF, 1, 0.3, 0.7, 0.2887),
    (1.6, 0.5, _INF, 1, 0.3, 0.7, 0.2888),
    (math.exp(0.3 + 0.7 * 0.3), -_INF, _INF, 1, 0.3, 0.7, 0.4),
    (math.exp(0.3 - 0.7 * 0.5), 0.5, _INF, -1, 0.3, 0.7, -0.45),
    (math.exp(0.3 + 0.7 * 6.0), -_INF, _INF, 1, 0.3, 0.7, 0.15),
    (3.0, -_INF, _INF, 1, 0.3, 0.6, 1.5),
    (1.2, 0.3, 4.0, 0, 0.2, 0.9, 0.6),
    (2.5, 1.0, _INF, 0, 0.4, 0.5, -1.2),
    (1.0, 1.0 - 1e-9, 1.0 + 1e-3, 0, 0.1, 0.6, 0.3),
    (2.0, -_INF, 2.0 * (1.0 + 1e-4), 1, 0.5, 0.6, 0.1),
)
_FREQUENCY = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 2.0])
# Worst measured error per order (value, score, Hessian, third, fourth), in u.
_MEASURED = {
    "location": (21.7, 180.0, 462.0, 859.0, 1.79e3),
    "mean": (21.7, 148.0, 412.0, 2.04e3, 8.24e3),
}
_KEYS = tuple(
    (a, b, c)
    for t in range(5)
    for a in range(t, -1, -1)
    for b in range(t - a, -1, -1)
    for c in (t - a - b,)
)


def _times(x: dict, y: dict) -> dict:
    out: dict = {}
    for (e1, v1), (e2, v2) in itertools.product(x.items(), y.items()):
        key = tuple(i + j for i, j in zip(e1, e2, strict=True))
        if sum(key) <= 4:
            out[key] = out.get(key, 0) + v1 * v2
    return out


def _powers(x: dict) -> list[dict]:
    powers = [{(0, 0, 0): mp.mpf(1)}]
    for _ in range(4):
        powers.append(_times(powers[-1], x))
    return powers


def _plus(*terms) -> dict:
    out: dict = {}
    for scale, series in terms:
        for key, value in series.items():
            out[key] = out.get(key, 0) + scale * value
    return out


def _log_series(x: dict) -> dict:
    x0 = x[(0, 0, 0)]
    rest = {k: v / x0 for k, v in x.items() if k != (0, 0, 0)}
    powers = _powers(rest)
    out = _plus(*(((-1) ** (n + 1) / mp.mpf(n), powers[n]) for n in range(1, 5)))
    out[(0, 0, 0)] = mp.log(x0)
    return out


def _exp_series(x: dict) -> dict:
    rest = {k: v for k, v in x.items() if k != (0, 0, 0)}
    powers = _powers(rest)
    out = _plus(*((1 / mp.factorial(n), powers[n]) for n in range(5)))
    return {k: mp.exp(x[(0, 0, 0)]) * v for k, v in out.items()}


def _log_loading(sigma, q):
    """``log C(sigma, Q) = log E[Y] - mu``, with its ``Q = 0`` limit ``sigma^2/2``."""
    if q == 0:
        return sigma**2 / 2
    k = 1 / q**2
    return (sigma / q) * mp.log(q**2) + mp.loggamma(k + sigma / q) - mp.loggamma(k)


def _coordinates(form, row):
    """``dw`` and ``dQ`` as series in the natural increments, with the row's ``(w, Q)``."""
    y, _, _, _, mu, sigma, q = (mp.mpf(v) if isinstance(v, float) else v for v in row)
    unit = [{(0, 0, 0): 0, key: mp.mpf(1)} for key in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    shrink = _powers({(0, 1, 0): -1 / sigma})
    geometric = _plus(*((1, shrink[n]) for n in range(5)))
    if form == "location":
        d_mu = unit[0]
    else:
        mean = mp.exp(mu + _log_loading(sigma, q))
        ratio = _powers({(1, 0, 0): 1 / mean})
        d_log_mean = _plus(*(((-1) ** (n + 1) / mp.mpf(n), ratio[n]) for n in range(1, 5)))
        # At Q = 0 the default step is so small that log C's terms cancel past
        # the working precision; a fixed step of 1e-20 leaves an O(1e-40) error.
        step = {"h": mp.mpf("1e-20")} if q == 0 else {}
        with mp.workdps(250 if q == 0 else 80):
            loading = {
                (0, i, j): mp.diff(_log_loading, (sigma, q), (i, j), **step)
                / (mp.factorial(i) * mp.factorial(j))
                for i in range(5)
                for j in range(5 - i)
                if i + j >= 1
            }
        d_mu = _plus((1, d_log_mean), (-1, loading))
    w = (mp.log(y) - mu) / sigma
    dw = _plus((1, _times(_plus((1, {(0, 0, 0): w}), (-1 / sigma, d_mu)), geometric)))
    dw[(0, 0, 0)] = dw.get((0, 0, 0), 0) - w
    return dw, unit[2], w, q


def _log_tail_series(form, row, point, lower_tail):
    """``log F`` or ``log S`` at ``point`` as a series in the natural increments."""
    moved = (point, *row[1:])
    dw, dq, w, q = _coordinates(form, moved)
    lower, upper = reference(w, q)
    taylor = lower if lower_tail else upper
    p_w, p_q = _powers(dw), _powers(dq)
    return _plus(*((c, _times(p_w[i], p_q[j])) for (i, j), c in taylor.items()))


def _window_series(form, row, lo, hi):
    """``log P(lo < Y <= hi)`` as a series, read from the smaller tail at the near end."""
    if not lo > 0:
        return _log_tail_series(form, row, hi, True)
    if hi == _INF:
        return _log_tail_series(form, row, lo, False)
    lower_tail = math.log(0.5 * (lo + hi)) < row[4]
    near, far = (hi, lo) if lower_tail else (lo, hi)
    difference = _plus(
        (1, _exp_series(_log_tail_series(form, row, near, lower_tail))),
        (-1, _exp_series(_log_tail_series(form, row, far, lower_tail))),
    )
    return _log_series(difference)


def _density_series(form, row):
    y, _, _, _, mu0, sigma0, q0 = row

    def log_density(p0, sigma, q):
        mu = p0 if form == "location" else mp.log(p0) - _log_loading(sigma, q)
        k = 1 / q**2
        u = q * (mp.log(y) - mu) / sigma
        return (
            mp.log(abs(q))
            + k * mp.log(k)
            - mp.loggamma(k)
            + k * (u - mp.exp(u))
            - mp.log(sigma * y)
        )

    first = mp.mpf(mu0) if form == "location" else mp.exp(mu0 + _log_loading(sigma0, q0))
    point = (first, mp.mpf(sigma0), mp.mpf(q0))
    return {
        key: mp.diff(log_density, point, key)
        / (mp.factorial(key[0]) * mp.factorial(key[1]) * mp.factorial(key[2]))
        for key in _KEYS
    }


def _terms(form, row):
    y, lower, upper, side, *_ = row
    if side == 0:
        terms = [_density_series(form, row)]
    else:
        terms = [
            _window_series(form, row, y, upper) if side > 0 else _window_series(form, row, lower, y)
        ]
    if lower > -_INF or upper < _INF:
        terms.append({k: -v for k, v in _window_series(form, row, lower, upper).items()})
    return terms


def _plain(series: dict, order: int) -> list:
    """Packed plain partials of one order, in ``combinations_with_replacement`` order."""
    out = []
    for combo in itertools.combinations_with_replacement(range(3), order):
        key = tuple(combo.count(k) for k in range(3))
        factor = math.prod(math.factorial(e) for e in key)
        out.append(series.get(key, 0) * factor)
    return out


@functools.cache
def _oracle(form):
    """Per row, the bounded log-likelihood's partials and term scale per order."""
    rows = []
    with mp.workdps(50):
        for row in _ORACLE_ROWS:
            terms = _terms(form, row)
            total = _plus(*((1, t) for t in terms))
            partials = [_plain(total, k) for k in range(5)]
            scales = [max(abs(v) for t in terms for v in _plain(t, k)) for k in range(5)]
            rows.append((partials, scales))
    return tuple(rows)


def _channel_errors(form: str, semantics: str) -> list[float]:
    family = GeneralizedGammaLSS(parametrisation=form)
    y = np.array([row[0] for row in _ORACLE_ROWS])
    bounds = ResponseBounds.from_arrays(
        *(np.array([row[k] for row in _ORACLE_ROWS]) for k in (1, 2, 3))
    )
    weights = None if semantics == "prior" else _FREQUENCY
    resolved = resolve_likelihood_weights(
        weights, n_observations=len(y), contract=WeightContract(semantics)
    )
    plan = family.bind_likelihood(y, resolved, bounded_observation(bounds))
    location = np.array([row[4:] for row in _ORACLE_ROWS])
    theta = location.copy()
    if form == "mean":
        theta[:, 0] = mean_of_location(*location.T)
    evaluation = family.evaluate_natural(y, theta, plan, derivative_order=2)
    links = (IdentityLink(),) * 3
    pairs = tuple(itertools.combinations_with_replacement(range(3), 2))
    third = [
        family.predictor_curvature_directional_derivative(
            y, theta, np.tile(np.eye(3)[axis], (len(y), 1)), links, plan
        )
        for axis in range(3)
    ]
    first = np.zeros((len(pairs), len(y), 3))
    second = np.zeros_like(first)
    for index, (a, b) in enumerate(pairs):
        first[index, :, a] = 1.0
        second[index, :, b] = 1.0
    fourth = family.predictor_curvature_second_directional_derivative(
        y, theta, first, second, links, plan
    )
    assert np.all(evaluation.valid)
    column3 = {key: c for c, key in enumerate(itertools.combinations_with_replacement(range(3), 3))}
    column4 = {key: c for c, key in enumerate(itertools.combinations_with_replacement(range(3), 4))}
    worst = [0.0] * 5
    for i, (partials, scales) in enumerate(_oracle(form)):
        mass = 1.0 if weights is None else weights[i]
        value = (
            evaluation.optimizing_log_likelihood[i] + evaluation.parameter_independent_carrier[i]
        )
        computed = (
            [value],
            evaluation.score[i],
            evaluation.hessian_packed[i],
            [third[axis][i, c] for axis in range(3) for c in range(6)],
            [fourth[k, i, c] for k in range(6) for c in range(6)],
        )
        exact = (
            *partials[:3],
            [
                -partials[3][column3[tuple(sorted((*pair, axis)))]]
                for axis in range(3)
                for pair in pairs
            ],
            [-partials[4][column4[tuple(sorted((*p, *r)))]] for r in pairs for p in pairs],
        )
        for order, (got, want) in enumerate(zip(computed, exact, strict=True)):
            error = max(abs(mp.mpf(float(g)) - mass * e) for g, e in zip(got, want, strict=True))
            worst[order] = max(worst[order], float(error / (mass * scales[order])) / _U)
    return worst


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_bounded_channels_match_the_mpmath_oracle_to_order_four(form, semantics):
    worst = _channel_errors(form, semantics)
    for order, (error, measured) in enumerate(zip(worst, _MEASURED[form], strict=True)):
        assert error <= _MARGIN * max(measured, 1.0), (order, error)


# -- smoothing parameters -----------------------------------------------------


def test_fit_reml_with_censored_rows_takes_the_analytic_laml_derivatives(monkeypatch):
    rng = np.random.default_rng(563)
    x = rng.uniform(0.0, 1.0, 5_000)
    y = _draw(rng, len(x), mu=0.0, sigma=0.6, q=0.4) * np.exp(6.9 + 0.5 * np.sin(5.0 * x))
    limit = float(np.quantile(y, 0.9))
    X, paid, censored = pd.DataFrame({"x": x}), np.minimum(y, limit), y >= limit
    hook = GeneralizedGammaLSS.predictor_curvature_directional_derivative
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(args[-1].observation.is_complete)
        return hook(self, *args, **kwargs)

    def fitted():
        family = GeneralizedGammaLSS(parametrisation="location")
        model = SuperLSS(family, family.location(s("x")), family.scale(), family.shape())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(X, paid, censored=censored, outer="efs+newton")
        grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 7)})
        return model.predict_parameters(grid).to_numpy()

    monkeypatch.setattr(GeneralizedGammaLSS, "predictor_curvature_directional_derivative", counted)
    analytic = fitted()
    assert calls and not any(calls)

    def refuse(*args, **kwargs):
        raise UnsupportedLikelihoodContractError("differenced route")

    for name in (
        "predictor_curvature_directional_derivative",
        "predictor_curvature_second_directional_derivative",
    ):
        monkeypatch.setattr(GeneralizedGammaLSS, name, refuse)
    np.testing.assert_allclose(analytic, fitted(), rtol=1e-5)


# -- windows, residuals and predictions ---------------------------------------


@pytest.mark.parametrize("relative_width", [1e-12, 1e-6, 0.02, 0.06])
@pytest.mark.parametrize("form", _FORMS)
def test_narrow_windows_keep_their_mass_on_both_sides_of_the_switch(relative_width, form):
    mu, sigma, q = 0.2, 0.5, 0.6
    start = 1.7
    theta = np.array([[mu, sigma, q]])
    if form == "mean":
        theta[0, 0] = mean_of_location(*theta.T)[0]
    family = GeneralizedGammaLSS(parametrisation=form)
    value = family.log_window_probability(start, start * (1.0 + relative_width), theta)[0]
    with mp.workdps(40):
        shape = 1 / mp.mpf(q) ** 2

        def point(y):
            return shape * mp.exp(mp.mpf(q) * (mp.log(mp.mpf(y)) - mu) / sigma)

        exact = mp.log(
            mp.gammainc(
                shape, point(start), point(start * (1.0 + relative_width)), regularized=True
            )
        )
    assert value == pytest.approx(float(exact), rel=1e-12, abs=1e-12)


def test_truncated_and_censored_pit_is_uniform():
    rng = np.random.default_rng(564)
    y = _draw(rng, 6_000)
    y = y[y > 1.0]
    limit = float(np.quantile(y, 0.85))
    paid, hit = np.minimum(y, limit), y >= limit
    model = _model("location")
    X = _intercepts(len(paid))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, paid, truncation=(1.0, None), censored=hit)
    payload = model.residual_set(X, paid, truncation=(1.0, None), censored=hit)
    assert payload.randomised_rows == hit.sum()
    assert stats.kstest(payload.pit, "uniform").pvalue > 1e-3


def test_predictions_near_zero_shape_read_the_fitted_law():
    """Below ``|Q| = 1e-8`` the CDF is the uniform tail and the quantile its inverse."""
    w = np.array([-3.0, -1.0, 0.0, 0.7, 2.5])
    for q in (5e-9, -5e-9, 0.0):
        shape = np.full(len(w), q)
        cdf = generalized_gamma_cdf(np.exp(w), np.zeros(len(w)), np.ones(len(w)), shape)
        exact = np.array([float(mp.exp(expansion_reference(v, q)[0][(0, 0)])) for v in w])
        np.testing.assert_allclose(cdf, exact, rtol=1e-15)
        # The log-normal it replaces is within 1e-8 of it.
        assert np.max(np.abs(cdf - stats.norm.cdf(w))) < 1e-8
        p = np.array([1e-6, 0.1, 0.5, 0.9, 1 - 1e-6])
        back = generalized_gamma_cdf(
            generalized_gamma_quantile(p, np.zeros(5), np.ones(5), np.full(5, q)),
            np.zeros(5),
            np.ones(5),
            np.full(5, q),
        )
        np.testing.assert_allclose(back, p, rtol=1e-13)


# -- contract, slicing and separation -----------------------------------------


def test_bounded_plans_hash_their_bounds_slice_them_and_refuse_fisher():
    family = GeneralizedGammaLSS()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = resolve_likelihood_weights(None, n_observations=4, contract=WeightContract("prior"))
    sides = np.array([0, 0, 1, 0])

    def plan(top):
        bounds = ResponseBounds.from_arrays(
            np.full(4, -_INF), np.array([5.0, 5.0, 3.5, top]), sides
        )
        return family.bind_likelihood(y, weights, bounded_observation(bounds))

    bounded, other = plan(6.0), plan(7.0)
    complete = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    assert len({bounded.plan_identifier, other.plan_identifier, complete.plan_identifier}) == 3
    assert bounded.parameter_independent_carrier[2] == 0.0
    assert complete.parameter_independent_carrier[2] != 0.0
    child = bounded.take(np.array([2, 3]))
    assert child.observation.bounds == ResponseBounds.from_arrays(
        np.full(2, -_INF), np.array([3.5, 6.0]), np.array([1, 0])
    )
    theta = np.tile((2.0, 0.5, 0.3), (4, 1))
    np.testing.assert_array_equal(
        family.evaluate_natural(y, theta, bounded).score[[2, 3]],
        family.evaluate_natural(y[[2, 3]], theta[[2, 3]], child).score,
    )
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.expected_information_natural(theta, bounded)
    model = SuperLSS(
        family, family.mean(), family.scale(), family.shape(), coefficient_curvature="fisher"
    )
    with pytest.raises(ValueError, match="truncated or censored rows"):
        model.fit(_intercepts(4), y, truncation=(None, 6.0))
    with pytest.raises(UnsupportedLikelihoodContractError, match="prior weights"):
        family.log_window_probability(1.0, 2.0, theta[:1], np.array([2.0]))
    assert isinstance(family, distributional.ResponseWindowFamily)
    assert isinstance(family, distributional.CensoredSeparationFamily)


def test_a_location_level_with_only_right_censored_rows_is_disclosed_as_separated():
    rng = np.random.default_rng(565)
    group = rng.choice(["a", "b", "c"], 1_500)
    loss = _draw(rng, 1_500)
    limit = float(np.quantile(loss, 0.9))
    censored = (loss >= limit) | (group == "c")
    paid = np.where(censored, limit, loss)
    family = GeneralizedGammaLSS(parametrisation="location")
    strict = SuperLSS(
        family, family.location(cat("g")), family.scale(), family.shape(), separation="error"
    )
    with pytest.raises(SeparationError, match=r"(?s)Censoring separation.*'location:g'"):
        strict.fit(pd.DataFrame({"g": group}), paid, censored=censored)
    flags = family.censored_recession(
        ResponseBounds.from_arrays(np.full(3, -_INF), np.full(3, _INF), np.array([1, -1, 0]))
    )
    np.testing.assert_array_equal(
        flags, [[True, False, False], [True, False, False], [False, False, False]]
    )


def test_a_window_deep_in_the_lower_tail_is_read_from_the_lower_tail():
    """Read by reflection, ``kappa ~ 1/P`` and its fourth power overflows below ``P ~ 1e-77``."""
    theta = np.array([[0.0, 0.3, 0.5]])
    jets = _declared_bounds.gg_window_jets(
        np.array([1e-14]), np.array([2e-14]), theta, order=4, mean_form=False
    )
    assert all(np.all(np.isfinite(level)) for level in jets)
