"""GammaLS with truncated and censored rows (issue 544, stage 2).

Fits are checked against direct maximum likelihood on the truncated or
censored likelihood in scipy, an independent reference; the complete-row fit
of the same rows is the unfixed behaviour.  The bounded rows' derivative
channels are checked against an mpmath oracle of the bounded log-likelihood:
the regularised ``mp.gammainc`` window differentiated numerically by
``mp.diff`` at 40 digits, which shares no formula with the incomplete-gamma
engine's series and continued fraction, the window combiner or the
quadrature.

Error model.  A bounded row's channels carry the engine's error in its
``(a, x)`` partials (``tests/test_incomplete_gamma_engine.py``: at most
330, 332, 578, 5.4e3 and 4.4e4 ``u`` of the row's same-order scale at orders
0-4 for shapes in ``[1e-2, 1e4]``), mapped to the parameters by the generated
chain rule, which is exact in its own terms and rounds each product once, and
divided by the window's tail probability, which multiplies it by ``1 + kappa``
(``kappa = T(far)/(T(near) - T(far))``) on two-ended windows.  The rows below
keep ``kappa`` under about 10 except the narrow ones, which the quadrature
integrates with no tail difference.  Errors are in ``u`` of the row's largest
oracle entry of that order, as in ``tests/test_family_derivatives_symbolic.py``,
and each order is held to its worst measured error on this grid times four
(``_MARGIN``), with a floor of ``4u``; the measurements are on x86-64 Linux
(AMD EPYC 9645, Zen 5).
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats

from superglm import (
    ConvergenceWarning,
    GammaLS,
    SeparationError,
    SeparationWarning,
    SuperLSS,
    cat,
    distributional,
    s,
)
from superglm.distributional.families import gamma as gamma_family
from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
from superglm.distributional.kernels import _declared_bounds
from superglm.distributional.observation import ResponseBounds
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import LogLink

mp = pytest.importorskip("mpmath")

_U = 2.0**-53
_MARGIN = 4
_INF = np.inf


def _intercepts(n: int) -> pd.DataFrame:
    return pd.DataFrame({"one": np.ones(n)})


def _fit(y, *, weight_semantics="prior", **kwargs) -> tuple[float, float]:
    family = GammaLS()
    model = SuperLSS(family, family.mean(), family.scale(), weight_semantics=weight_semantics)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(y)), y, **kwargs)
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    return float(row["mean"]), float(row["scale"])


def _law(mean, cv, weight=1.0):
    shape = weight / cv**2
    return stats.gamma(shape, scale=mean / shape)


def _mle(negative_log_likelihood, start=(np.log(5000.0), 0.0)) -> np.ndarray:
    """Maximum-likelihood ``(log mean, log cv)``; the objective is scaled to order one."""
    scale = abs(negative_log_likelihood(np.asarray(start)))
    result = optimize.minimize(
        lambda t: negative_log_likelihood(t) / scale,
        np.asarray(start),
        method="Nelder-Mead",
        options={"xatol": 1e-11, "fatol": 1e-15, "maxiter": 20_000},
    )
    assert result.success
    return result.x


def _log_information(negative_log_likelihood, point, step=1e-4) -> np.ndarray:
    """Central-difference Hessian of the negative log-likelihood in ``(log mean, log cv)``."""
    hessian = np.empty((2, 2))
    unit = np.eye(2) * step
    for i in range(2):
        for j in range(2):
            corners = [
                negative_log_likelihood(point + a * unit[i] + b * unit[j]) * a * b
                for a in (1, -1)
                for b in (1, -1)
            ]
            hessian[i, j] = sum(corners) / (4 * step * step)
    return hessian


# -- issue 544's own case ------------------------------------------------------

_MEAN, _CV, _N = 5_000.0, 1.2, 200_000


@functools.cache
def _issue_claims() -> tuple[np.ndarray, float]:
    shape = 1.0 / _CV**2
    y = np.random.default_rng(544).gamma(shape, _MEAN / shape, _N)
    return y, float(_law(_MEAN, _CV).ppf(0.9))


def _sufficient_log_likelihood(t, n, total, log_total):
    """``sum log f(y_i)`` of ``n`` unit-weight gamma rows from ``sum y`` and ``sum log y``."""
    mean, cv = np.exp(t)
    shape = 1.0 / cv**2
    return (
        n * (shape * np.log(shape / mean) - special.gammaln(shape))
        + (shape - 1.0) * log_total
        - shape / mean * total
    )


def _assert_recovers(fitted, reference, information):
    np.testing.assert_allclose(fitted, np.exp(reference), rtol=2e-6)
    # Within four standard errors of the truth, from the observed information.
    errors = np.sqrt(np.diag(np.linalg.inv(information)))
    assert np.all(np.abs(np.log(fitted) - np.log([_MEAN, _CV])) < 4.0 * errors)
    return errors


def test_issue_544_truncated_large_losses_recover_the_parent_gamma():
    y, u = _issue_claims()
    large = y[y > u]
    stats_ = (len(large), large.sum(), np.log(large).sum())

    def nll(t):
        mean, cv = np.exp(t)
        shape = 1.0 / cv**2
        tail = np.log(special.gammaincc(shape, shape * u / mean))
        return -(_sufficient_log_likelihood(t, *stats_) - len(large) * tail)

    fitted = _fit(large, truncation=(u, None))
    reference = _mle(nll)
    errors = _assert_recovers(fitted, reference, _log_information(nll, reference))
    # Read as complete, the large losses describe a different law entirely.
    complete = _fit(large)
    assert abs(np.log(complete[0] / _MEAN)) > 5.0 * errors[0]
    assert complete[1] < 0.5 * _CV


def test_issue_544_censored_layer_recovers_the_parent_gamma():
    y, u = _issue_claims()
    paid, hit = np.minimum(y, u), y > u
    body = y[~hit]
    stats_ = (len(body), body.sum(), np.log(body).sum())

    def nll(t):
        mean, cv = np.exp(t)
        shape = 1.0 / cv**2
        tail = np.log(special.gammaincc(shape, shape * u / mean))
        return -(_sufficient_log_likelihood(t, *stats_) + hit.sum() * tail)

    fitted = _fit(paid, censored=hit)
    reference = _mle(nll)
    errors = _assert_recovers(fitted, reference, _log_information(nll, reference))
    complete = _fit(paid)
    assert abs(np.log(complete[0] / _MEAN)) > 20.0 * errors[0]


# -- intercept-only fits under both weight semantics --------------------------


def _weighted_problem(seed=546, n=4_000):
    rng = np.random.default_rng(seed)
    weight = rng.integers(1, 4, n).astype(float)
    shape = weight / 0.8**2
    y = rng.gamma(shape, 900.0 / shape)
    keep = y > 150.0
    y, weight = y[keep], weight[keep]
    limit = float(np.quantile(y, 0.85))
    return np.minimum(y, limit), y >= limit, weight, limit


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_truncated_and_censored_intercepts_match_scipy_under_weights(semantics):
    paid, hit, weight, limit = _weighted_problem()
    # Prior weights reweight the law (shape w/cv^2); frequency weights replicate rows.
    law_weight, mass = (weight, 1.0) if semantics == "prior" else (1.0, weight)

    def nll(t):
        law = _law(*np.exp(t), law_weight)
        rows = np.where(hit, law.logsf(limit), law.logpdf(paid)) - law.logsf(150.0)
        return -np.sum(mass * rows)

    fitted = _fit(
        paid,
        weight_semantics=semantics,
        sample_weight=weight,
        truncation=(150.0, None),
        censored=hit,
    )
    np.testing.assert_allclose(fitted, np.exp(_mle(nll, (np.log(900.0), 0.0))), rtol=1e-6)


def test_left_censored_rows_match_scipy():
    rng = np.random.default_rng(547)
    y = rng.gamma(2.0, 50.0, 5_000)
    floor = float(np.quantile(y, 0.2))
    hit = y <= floor
    paid = np.maximum(y, floor)

    def nll(t):
        law = _law(*np.exp(t))
        return -np.sum(np.where(hit, law.logcdf(floor), law.logpdf(paid)))

    fitted = _fit(paid, censored=np.where(hit, "left", "none"))
    np.testing.assert_allclose(fitted, np.exp(_mle(nll, (np.log(100.0), 0.0))), rtol=1e-6)


# -- bounded channels against the mpmath oracle -------------------------------

# (y, lower, upper, side, mean, cv, weight): the issue's rows, a far tail, a
# left-censored row, a two-ended window at a = 0.11, narrow windows on both
# routes (a truncation window and a censored row's observed window), and
# censored rows under truncation, one with a prior weight.
_ORACLE_ROWS = (
    (9_000.0, -_INF, _INF, 1, 5_000.0, 1.2, 1.0),
    (14_000.0, 11_000.0, _INF, 0, 5_000.0, 1.2, 1.0),
    (3_000.0, -_INF, 11_000.0, 0, 5_000.0, 1.2, 1.0),
    (2.2, -_INF, _INF, 1, 1.0, 0.3, 1.0),
    (0.5, -_INF, _INF, -1, 1.0, 0.3, 1.0),
    (1.3, 0.5, 2.0, 0, 1.0, 3.0, 1.0),
    (1.0, 1.0 - 1e-9, 1.0 + 1e-3, 0, 1.2, 0.5, 1.0),
    (2.0, -_INF, 2.0 * (1.0 + 1e-4), 1, 1.5, 0.8, 1.0),
    (4.0, 1.0, _INF, 1, 2.0, 0.7, 2.5),
    (0.8, -_INF, 3.0, -1, 2.0, 0.7, 0.4),
)
# Integer replication counts for the same rows under frequency semantics.
_FREQUENCY = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0])
# Worst measured error per order (value, score, Hessian, third, fourth), in u.
_MEASURED = (3.5, 84.2, 130.9, 194.4, 938.9)


def _oracle(row, natural: bool):
    """The row's bounded log-likelihood terms at unit mass, as mpmath functions of two parameters.

    The first term is the log density, or a censored row's observed window;
    the second, on a truncated row, is minus its truncation window.
    """
    y, lower, upper, side, _, _, weight = row
    w = mp.mpf(weight)

    def law(p0, p1):
        mean, cv = (p0, p1) if natural else (mp.exp(p0), mp.exp(p1))
        return mean, w / cv**2

    def window(lo, hi):
        def value(p0, p1):
            mean, shape = law(p0, p1)
            start = 0 if not lo > 0 else shape * mp.mpf(lo) / mean
            end = mp.inf if hi == _INF else shape * mp.mpf(hi) / mean
            return mp.log(mp.gammainc(shape, start, end, regularized=True))

        return value

    def density(p0, p1):
        mean, shape = law(p0, p1)
        point = mp.mpf(y)
        return (
            (shape - 1) * mp.log(point)
            - point * shape / mean
            - mp.loggamma(shape)
            - shape * mp.log(mean / shape)
        )

    terms = [density if side == 0 else window(y, upper) if side > 0 else window(lower, y)]
    if lower > -_INF or upper < _INF:
        truncation = window(lower, upper)
        terms.append(lambda p0, p1: -truncation(p0, p1))
    return terms


def _partials(function, point, order):
    """``[[d^k f / dp0^(k-j) dp1^j for j]]`` for ``k = 0..order``."""
    return [[function(*point)]] + [
        [mp.diff(function, point, (k - j, j)) for j in range(k + 1)] for k in range(1, order + 1)
    ]


def _summed(terms, point, order):
    """The row's partials and, per order, its term scale: the largest entry of either term."""
    parts = [_partials(term, point, order) for term in terms]
    total = [[sum(values) for values in zip(*levels)] for levels in zip(*parts)]
    scale = [max(abs(v) for part in parts for v in part[k]) for k in range(order + 1)]
    return total, scale


@functools.cache
def _oracle_partials(rows):
    """Per row, natural partials to order 2 and log-coordinate partials to order 4."""
    with mp.workdps(40):
        return tuple(
            (
                _summed(_oracle(row, True), (mp.mpf(row[4]), mp.mpf(row[5])), 2),
                _summed(_oracle(row, False), (mp.log(mp.mpf(row[4])), mp.log(mp.mpf(row[5]))), 4),
            )
            for row in rows
        )


def _bound_plan(rows, semantics, weights):
    y = np.array([row[0] for row in rows])
    bounds = ResponseBounds.from_arrays(*(np.array([row[k] for row in rows]) for k in (1, 2, 3)))
    resolved = resolve_likelihood_weights(
        weights, n_observations=len(y), contract=WeightContract(semantics)
    )
    return y, GammaLS().bind_likelihood(y, resolved, bounded_observation(bounds))


def _error_in_u(computed, oracle, scale) -> float:
    return float(max(abs(mp.mpf(float(c)) - o) for c, o in zip(computed, oracle)) / scale) / _U


_PAIRS = ((0, 0), (0, 1), (1, 1))


def _channel_errors(semantics: str) -> list[float]:
    """Worst error in u per order over the oracle rows under one weight semantics.

    Under prior semantics the weight enters the shape; under frequency
    semantics it multiplies the unit-weight row, so the oracle row is
    evaluated at unit weight and scaled.
    """
    family = GammaLS()
    weights = np.array([row[6] for row in _ORACLE_ROWS]) if semantics == "prior" else _FREQUENCY
    y, plan = _bound_plan(_ORACLE_ROWS, semantics, weights)
    theta = np.array([[row[4], row[5]] for row in _ORACLE_ROWS])
    eta = np.log(theta)
    rows = _ORACLE_ROWS
    if semantics == "frequency":
        rows = tuple(row[:6] + (1.0,) for row in _ORACLE_ROWS)
    oracle = _oracle_partials(rows)
    evaluation = family.evaluate_natural(y, theta, plan, derivative_order=2)
    links = (LogLink(), LogLink())
    third = [
        family.predictor_curvature_directional_derivative(
            y, eta, np.tile(np.eye(2)[axis], (len(y), 1)), links, plan
        )
        for axis in range(2)
    ]
    first = np.zeros((3, len(y), 2))
    second = np.zeros_like(first)
    for index, (q, r) in enumerate(_PAIRS):
        first[index, :, q] = 1.0
        second[index, :, r] = 1.0
    fourth = family.predictor_curvature_second_directional_derivative(
        y, eta, first, second, links, plan
    )
    assert np.all(evaluation.valid)
    worst = [0.0] * 5
    for i, ((natural, natural_scale), (log, log_scale)) in enumerate(oracle):
        mass = 1.0 if semantics == "prior" else weights[i]
        value = (
            evaluation.optimizing_log_likelihood[i] + evaluation.parameter_independent_carrier[i]
        )
        computed = (
            [value],
            evaluation.score[i],
            evaluation.hessian_packed[i],
            [third[axis][i, c] for axis in range(2) for c in range(3)],
            [fourth[k, i, c] for k in range(3) for c in range(3)],
        )
        exact = (
            *natural,
            # -d H_pq / d eta_axis = -l_{p q axis}; the packed index counts the ones.
            [-log[3][p + q + axis] for axis in range(2) for p, q in _PAIRS],
            [-log[4][p + q + a + b] for a, b in _PAIRS for p, q in _PAIRS],
        )
        scales = (*natural_scale, log_scale[3], log_scale[4])
        errors = [
            _error_in_u(got, [mass * e for e in want], mass * scale)
            for got, want, scale in zip(computed, exact, scales, strict=True)
        ]
        worst = [max(w, e) for w, e in zip(worst, errors)]
    return worst


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_bounded_channels_match_the_mpmath_oracle_to_order_four(semantics):
    worst = _channel_errors(semantics)
    for order, (error, measured) in enumerate(zip(worst, _MEASURED, strict=True)):
        assert error <= _MARGIN * max(measured, 1.0), (order, error)


# -- smoothing parameters -----------------------------------------------------


def _smooth_claims(n=6_000, seed=548):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    mean = 1_000.0 * np.exp(0.6 * np.sin(5.0 * x))
    y = rng.gamma(1.5, mean / 1.5)
    limit = float(np.quantile(y, 0.9))
    return pd.DataFrame({"x": x}), np.minimum(y, limit), y >= limit


def test_fit_reml_with_censored_rows_takes_the_analytic_laml_derivatives(monkeypatch):
    X, paid, censored = _smooth_claims()
    calls = []
    third_hook = GammaLS.predictor_curvature_directional_derivative

    def counted(self, *args, **kwargs):
        result = third_hook(self, *args, **kwargs)
        calls.append(args[-1].observation.is_complete)
        return result

    def fitted():
        family = GammaLS()
        model = SuperLSS(family, family.mean(s("x")), family.scale())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(X, paid, censored=censored, outer="efs+newton")
        grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 7)})
        return model, model.predict_parameters(grid).to_numpy()

    monkeypatch.setattr(GammaLS, "predictor_curvature_directional_derivative", counted)
    model, analytic = fitted()
    # The LAML gradient took the bounded plan's analytic third derivatives.
    assert calls and not any(calls)
    rows = model._require_fitted().fit_state.retained_rows
    family = GammaLS()
    plan = family.bind_likelihood(rows.response, rows.likelihood_weights, rows.observation)
    pairs = np.zeros((1, len(rows.response), 2))
    pairs[0, :, 0] = 1.0
    fourth = family.predictor_curvature_second_directional_derivative(
        rows.response, rows.fitted_eta, pairs, pairs, (LogLink(), LogLink()), plan
    )
    assert np.all(np.isfinite(fourth))

    def refuse(*args, **kwargs):
        raise UnsupportedLikelihoodContractError("differenced route")

    monkeypatch.setattr(GammaLS, "predictor_curvature_directional_derivative", refuse)
    monkeypatch.setattr(GammaLS, "predictor_curvature_second_directional_derivative", refuse)
    _, differenced = fitted()
    np.testing.assert_allclose(analytic, differenced, rtol=1e-5)


# -- windows, residuals and diagnostics ---------------------------------------


def test_window_probabilities_match_scipy_in_the_body_and_keep_the_tails():
    lower = np.array([-_INF, 2.0, 300.0, 0.0, 4_000.0])
    upper = np.array([900.0, _INF, 1_500.0, 40.0, _INF])
    mean, cv = 700.0, 0.6
    law = _law(mean, cv)
    expected = np.log(law.cdf(upper) - law.cdf(np.maximum(lower, 0.0)))
    expected[[1, 4]] = law.logsf(lower[[1, 4]])
    theta = np.tile((mean, cv), (5, 1))
    np.testing.assert_allclose(
        GammaLS().log_window_probability(lower, upper, theta), expected, rtol=1e-12
    )
    # Prior weights reweight the law: shape w/cv^2.
    weighted = GammaLS().log_window_probability(lower, upper, theta, np.full(5, 2.5))
    law = _law(mean, cv, 2.5)
    expected = np.log(law.cdf(upper) - law.cdf(np.maximum(lower, 0.0)))
    expected[[1, 4]] = law.logsf(lower[[1, 4]])
    np.testing.assert_allclose(weighted, expected, rtol=1e-12)
    # Far in the upper tail, where 1 - cdf is zero, the window is the log survival.
    far = GammaLS().log_window_probability(1e5, _INF, np.array([[mean, cv]]))
    assert far[0] == pytest.approx(_law(mean, cv).logsf(1e5), rel=1e-12)


@pytest.mark.parametrize("relative_width", [1e-12, 1e-6, 0.02, 0.06])
def test_narrow_windows_keep_their_mass_on_both_sides_of_the_switch(relative_width):
    theta = np.array([[2.0, 0.5]])
    start = 1.7
    value = GammaLS().log_window_probability(start, start * (1.0 + relative_width), theta)[0]
    with mp.workdps(40):
        shape = mp.mpf(4)
        exact = mp.log(
            mp.gammainc(
                shape,
                shape * mp.mpf(start) / 2,
                shape * mp.mpf(start * (1.0 + relative_width)) / 2,
                regularized=True,
            )
        )
    assert value == pytest.approx(float(exact), rel=1e-12, abs=1e-12)


def test_truncated_and_censored_pit_is_uniform():
    rng = np.random.default_rng(549)
    shape = 1.0 / 0.9**2
    y = rng.gamma(shape, 300.0 / shape, 6_000)
    y = y[y > 50.0]
    limit = float(np.quantile(y, 0.85))
    paid, hit = np.minimum(y, limit), y >= limit
    family = GammaLS()
    model = SuperLSS(family, family.mean(), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(paid)), paid, truncation=(50.0, None), censored=hit)
    X = _intercepts(len(paid))
    payload = model.residual_set(X, paid, truncation=(50.0, None), censored=hit)
    assert payload.randomised_rows == hit.sum()
    assert stats.kstest(payload.pit, "uniform").pvalue > 1e-3
    with pytest.warns(UserWarning, match="reads every row it is given as complete"):
        unread = model.residuals(X, paid, kind="pit")
    assert stats.kstest(unread, "uniform").pvalue < 1e-6


# -- contract, slicing and separation -----------------------------------------


def test_bounded_gamma_plans_hash_their_bounds_slice_them_and_refuse_fisher():
    family = GammaLS()
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
    assert complete.parameter_independent_carrier[2] == -np.log(3.0)
    child = bounded.take(np.array([2, 3]))
    assert child.observation.bounds == ResponseBounds.from_arrays(
        np.full(2, -_INF), np.array([3.5, 6.0]), np.array([1, 0])
    )
    theta = np.tile((2.0, 0.5), (4, 1))
    np.testing.assert_array_equal(
        family.evaluate_natural(y, theta, bounded).score[[2, 3]],
        family.evaluate_natural(y[[2, 3]], theta[[2, 3]], child).score,
    )
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.expected_information_natural(theta, bounded)
    model = SuperLSS(family, family.mean(), family.scale(), coefficient_curvature="fisher")
    with pytest.raises(ValueError, match="truncated or censored rows"):
        model.fit(_intercepts(4), y, truncation=(None, 6.0))
    assert isinstance(family, distributional.ResponseWindowFamily)
    assert isinstance(family, distributional.CensoredSeparationFamily)


def test_a_mean_level_with_only_right_censored_rows_is_disclosed_as_separated():
    rng = np.random.default_rng(550)
    group = rng.choice(["a", "b", "c"], 1_500)
    loss = rng.gamma(1.5, 3_000.0 / 1.5, 1_500)
    limit = 5_000.0
    censored = (loss >= limit) | (group == "c")
    paid = np.where(censored, limit, loss)
    X = pd.DataFrame({"g": group})
    family = GammaLS()
    # The scan runs before any coefficient is fitted.
    strict = SuperLSS(family, family.mean(cat("g")), family.scale(), separation="error")
    with pytest.raises(SeparationError, match=r"(?s)Censoring separation.*'mean:g'"):
        strict.fit(X, paid, censored=censored)
    # The scale's direction depends on the fitted mean, so it is never flagged.
    flags = family.censored_recession(
        ResponseBounds.from_arrays(np.full(3, -_INF), np.full(3, _INF), np.array([1, -1, 0]))
    )
    np.testing.assert_array_equal(flags, [[True, False], [True, False], [False, False]])


def test_a_scale_collapsing_beside_a_censored_level_reads_every_window_and_is_disclosed(
    monkeypatch,
):
    """Complete rows sharing one response send the cv to zero beside a level censored at its mean.

    The likelihood rises without bound as the shape ``1/cv^2`` grows, so the
    fit walks it to the rounding limit with level ``c``'s rows on the window's
    diagonal.  Above shape 1e4 those rows are read by the uniform expansion,
    whose cost does not grow with the shape, so every window stays finite and
    no trial is lost to an invalid row; the fit discloses that it stopped
    unconverged, and the identified levels keep their shared response.
    """
    rng = np.random.default_rng(550)
    group = rng.choice(["a", "b", "c"], 600)
    censored = group == "c"
    paid = np.where(censored, 5_000.0, 4_000.0)
    X = pd.DataFrame({"g": group})
    window = gamma_family.gamma_window_jets
    reads = []

    def counted(lower, upper, parameters, weight, *, order, coordinates="natural"):
        jets = window(lower, upper, parameters, weight, order=order, coordinates=coordinates)
        cv = np.exp(parameters[:, 1]) if coordinates == "log" else parameters[:, 1]
        finite = np.isfinite(jets[0]) & np.all([np.isfinite(j).all(axis=1) for j in jets[1:]], 0)
        reads.append((float(np.max(weight / cv**2)), len(finite) - int(finite.sum())))
        return jets

    monkeypatch.setattr(gamma_family, "gamma_window_jets", counted)
    family = GammaLS()
    model = SuperLSS(family, family.mean(cat("g")), family.scale())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, paid, censored=censored)
    categories = {type(w.message) for w in caught}
    assert {SeparationWarning, ConvergenceWarning} <= categories
    telemetry = model.training_telemetry()
    assert not telemetry.converged
    _, invalid = np.array(reads).T
    assert invalid.sum() == 0
    means = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))["mean"]
    np.testing.assert_allclose(means, 4_000.0, rtol=1e-9)


@pytest.mark.parametrize("coordinates", ["natural", "log"])
def test_windows_on_the_diagonal_stay_finite_at_every_unbounded_shape(coordinates):
    """Rows censored at their own mean read finite windows at every shape a collapsing fit reaches.

    Forced rather than reached by a fit: shapes ``1e4`` to ``1e30`` at unit
    weight, both tails, orders 0-4.  The incomplete-gamma engine alone returns
    NaN from about shape ``3e8``; above ``1e4`` the uniform expansion reads them.
    """
    shapes = np.logspace(4, 30, 27)
    n = len(shapes)
    mean, cv = np.full(n, 5_000.0), shapes**-0.5
    parameters = np.column_stack((mean, cv))
    if coordinates == "log":
        parameters = np.log(parameters)
    orders = (0, 1, 2) if coordinates == "natural" else (3, 4)
    for lower, upper in ((mean, np.full(n, np.inf)), (np.zeros(n), mean)):
        for order in orders:
            jets = _declared_bounds.gamma_window_jets(
                lower, upper, parameters, np.ones(n), order=order, coordinates=coordinates
            )
            assert all(np.all(np.isfinite(jet)) for jet in jets)


@pytest.mark.parametrize("coordinates", ["natural", "log"])
def test_a_censored_row_at_an_unbounded_shape_reads_the_log_normal_limit(coordinates):
    """At shape ``1e20`` a right-censored row has ``log S = log Phi(-w) (1 + O(q))``.

    ``q = cv = 1e-10`` and ``w = log(y/mean)/q`` (mean one, so ``log(y/mean)``
    is formed without loss).  The uniform expansion is analytic through
    ``q = 0`` (``kernels/_gg_uniform_tail.py``); the incomplete-gamma engine
    would need ``O(sqrt(a)) = 1e10`` terms here.  With ``h`` the normal hazard
    at ``w``, ``d/dlog mean = h/q`` and ``d^2/dlog mean^2 = -h (h - w)/q^2``.
    """
    q, w = 1e-10, 0.3
    y = np.array([np.exp(w * q)])
    log_theta = np.array([[0.0, np.log(q)]])
    theta = log_theta if coordinates == "log" else np.exp(log_theta)
    jets = _declared_bounds.gamma_window_jets(
        y, np.array([_INF]), theta, np.ones(1), order=2, coordinates=coordinates
    )
    w_exact = float(mp.log(mp.mpf(y[0])) / mp.exp(mp.mpf(log_theta[0, 1])))
    log_sf = stats.norm.logsf(w_exact)
    hazard = np.exp(stats.norm.logpdf(w_exact) - log_sf)
    d1, d2 = hazard / q, -hazard * (hazard - w_exact) / q**2
    # The natural coordinate is the mean itself, here one: d^2/dmean^2 = d2 - d1.
    expected = (log_sf, d1, d2 if coordinates == "log" else d2 - d1)
    got = (jets[0][0], jets[1][0, 0], jets[2][0, 0])
    np.testing.assert_allclose(got, expected, rtol=1e-8)


@pytest.mark.parametrize(("z", "upper"), [(0.7, True), (-0.7, False)])
def test_a_censored_row_above_the_engines_shapes_meets_the_oracle_to_order_four(z, upper):
    """At shape ``1e5``, ``z/sqrt(a)`` from the mean, a censored row's log-scale jet meets mpmath.

    Above shape 1e4 the row is read by the uniform expansion.  Mean one, so the
    point enters without rounding; errors in ``u`` of the order's largest entry,
    held to four times the measured (1.1, 2.6, 6.4, 57, 262).
    """
    a = 1e5
    y = np.array([np.exp(z / np.sqrt(a))])
    theta = np.array([[0.0, -0.5 * np.log(a)]])
    lower, upper_end = (y, np.array([_INF])) if upper else (np.zeros(1), y)
    jets = _declared_bounds.gamma_window_jets(
        lower, upper_end, theta, np.ones(1), order=4, coordinates="log"
    )

    def tail(p0, p1):
        shape = 1 / mp.exp(p1) ** 2
        x = shape * mp.mpf(y[0]) / mp.exp(p0)
        ends = (x, mp.inf) if upper else (0, x)
        return mp.log(mp.gammainc(shape, *ends, regularized=True))

    measured = (1.12, 2.56, 6.35, 56.9, 262.0)
    with mp.workdps(80):
        point = (mp.mpf(0), mp.mpf(theta[0, 1]))
        for k, worst in enumerate(measured):
            exact = [mp.diff(tail, point, (k - j, j)) for j in range(k + 1)]
            got = [jets[0][0]] if k == 0 else list(jets[k][0])
            scale = max(abs(e) for e in exact)
            error = max(abs(mp.mpf(float(g)) - e) for g, e in zip(got, exact, strict=True))
            assert error <= _MARGIN * max(worst, 1.0) * _U * scale, k


def test_a_window_deep_in_the_lower_tail_is_read_from_the_lower_tail():
    """Read by reflection, ``kappa ~ 1/P`` and its fourth power overflows below ``P ~ 1e-77``."""
    theta = np.array([[0.0, np.log(0.3)]])
    lower, upper = np.array([1e-14]), np.array([2e-14])
    jets = _declared_bounds.gamma_window_jets(
        lower, upper, theta, np.ones(1), order=4, coordinates="log"
    )
    assert all(np.all(np.isfinite(level)) for level in jets)

    def window(p0, p1):
        shape = 1 / mp.exp(p1) ** 2
        scale = shape / mp.exp(p0)
        return mp.log(
            mp.gammainc(shape, scale * mp.mpf(1e-14), scale * mp.mpf(2e-14), regularized=True)
        )

    with mp.workdps(40):
        point = (mp.mpf(0), mp.log(mp.mpf(0.3)))
        exact = [mp.diff(window, point, (4 - j, j)) for j in range(5)]
    scale = max(abs(e) for e in exact)
    assert max(abs(mp.mpf(float(g)) - e) for g, e in zip(jets[4][0], exact)) <= 1e-9 * scale


def test_a_censored_rows_curvature_does_not_depend_on_the_response_units():
    """The gamma depends on ``y`` and the mean only through ``y/mean``: rescaling both keeps every jet.

    A right-censored row at ``y = mean``, CV 0.5, in units 1, ``1e110`` and
    ``1e-110``.  The log-mean jet is unit-free, and the natural-mean Hessian
    scales by ``mean^-2``; a kernel forming ``y/mean^3`` underflows or
    overflows there while the Hessian stays representable.
    """
    reference = None
    for unit in (1.0, 1e110, 1e-110):
        y, theta = np.array([unit]), np.array([[unit, 0.5]])
        natural = _declared_bounds.gamma_window_jets(
            y, np.array([_INF]), theta, np.ones(1), order=2
        )
        log_mean_hessian = unit**2 * natural[2][0, 0] + unit * natural[1][0, 0]
        log = _declared_bounds.gamma_window_jets(
            y, np.array([_INF]), np.log(theta), np.ones(1), order=4, coordinates="log"
        )
        row = (log_mean_hessian, *(np.atleast_1d(level[0]) for level in log))
        assert all(np.all(np.isfinite(level)) for level in row), unit
        if reference is None:
            reference = row
        # log(1e110) is rounded, so log coordinates move x by about 250u.
        for got, expected in zip(row, reference, strict=True):
            scale = np.max(np.abs(expected))
            assert np.max(np.abs(np.subtract(got, expected))) <= 1e-12 * scale, unit


def test_a_tiny_left_censoring_threshold_keeps_a_representable_hessian():
    """At mean = CV = 1 the law is exponential: ``log S(c) = -c/mean``, every mean derivative exact.

    ``c = 1e-160`` puts the engine's point at ``x = 1e-160``, where
    ``x^-j`` overflows from ``j = 2``.  ``d^2 log S/dmean^2 = -2c`` and the
    log-mean derivatives alternate ``c, -c, c, -c``.
    """
    c = 1e-160
    lower, upper = np.array([c]), np.array([_INF])
    natural = _declared_bounds.gamma_window_jets(lower, upper, np.ones((1, 2)), np.ones(1), order=2)
    log = _declared_bounds.gamma_window_jets(
        lower, upper, np.zeros((1, 2)), np.ones(1), order=4, coordinates="log"
    )
    assert all(np.all(np.isfinite(level)) for level in (*natural, *log))
    np.testing.assert_allclose(
        [natural[0][0], natural[1][0, 0], natural[2][0, 0]], [-c, c, -2 * c], rtol=1e-13
    )
    np.testing.assert_allclose([log[k][0, 0] for k in range(1, 5)], [c, -c, c, -c], rtol=1e-13)


@pytest.mark.parametrize("coordinates", ["natural", "log"])
@pytest.mark.parametrize("steps", [0, 1, 7])
def test_a_censored_row_near_a_non_unit_mean_keeps_its_standardized_distance(coordinates, steps):
    """At CV ``1e-12`` the uniform tail reads ``w = log(y/mean)/q``: the log must be centred.

    Mean 49 and ``y`` that many float steps above it.  ``log((1/mean) y)``
    carries an absolute ``u``, which ``1/q`` turns into a displacement of
    order ``u/q = 1e-4``; at ``y = mean`` the survival is one half.  Shape
    ``1e24`` is the log-normal limit, ``log S = log Phi(-w) (1 + O(q))``.
    """
    q, mean = 1e-12, 49.0
    y = mean
    for _ in range(steps):
        y = np.nextafter(y, _INF)
    theta = np.array([[mean, q]])
    theta = np.log(theta) if coordinates == "log" else theta
    jets = _declared_bounds.gamma_window_jets(
        np.array([y]), np.array([_INF]), theta, np.ones(1), order=0, coordinates=coordinates
    )
    with mp.workdps(40):
        exact_mean = mp.exp(mp.mpf(theta[0, 0])) if coordinates == "log" else mp.mpf(mean)
        w = float(mp.log(mp.mpf(y) / exact_mean) / mp.mpf(q))
    # log coordinates round log 49 itself: exp of it moves the mean by about 4u.
    allowance = 1e-9 if coordinates == "natural" else 4 * _U / q
    assert abs(jets[0][0] - stats.norm.logsf(w)) <= allowance + 1e-9


def test_a_collapsing_cv_at_the_mean_survives_with_probability_one_half():
    """The review's row: mean = y = 49, CV ``1e-16``, right-censored."""
    got = GammaLS().log_window_probability(
        np.array([49.0]), np.array([_INF]), np.array([[49.0, 1e-16]])
    )
    np.testing.assert_allclose(got, [np.log(0.5)], rtol=1e-12)


@pytest.mark.parametrize(("y", "cv"), [(1e14, 1e7), (1e16, 1e8)])
def test_a_right_censored_row_at_a_vanishing_shape_reads_its_small_upper_tail(y, cv):
    """Shape ``cv^-2`` (``1e-14``, ``1e-16``) at ``x = 1``: ``Q ~ a E1(1)`` is not read as ``1 - P``."""
    got = GammaLS().log_window_probability(np.array([y]), np.array([_INF]), np.array([[1.0, cv]]))
    with mp.workdps(40):
        exact = mp.log(mp.gammainc(1 / mp.mpf(cv) ** 2, 1, mp.inf, regularized=True))
    np.testing.assert_allclose(got, [float(exact)], rtol=1e-13)
