"""Binomial fits whose rows pass the solver's working clip (issue #734).

The solver holds a binomial linear predictor inside a band (logit and probit
``+-20``, cloglog ``[-20, 2.736]``) and its mean inside ``[1e-7, 1 - 1e-7]``.
A row held *against its outcome* (an event below the lower edge, a non-event
above the upper one) has a likelihood score of order one that the held row no
longer carries, so the fit can stop away from the maximum-likelihood estimate
and report ``converged``.

Every reference is the binomial likelihood itself, written in closed form in
the linear predictor so that neither tail is clipped, and solved by SciPy.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq, minimize, minimize_scalar, root
from scipy.special import expit, log_ndtr
from scipy.stats import norm

from superglm import Categorical, Numeric, Spline, SuperGLM
from superglm.links import LogitLink

# ── the likelihood in closed form in eta ───────────────────────────


def _log_likelihood_rows(link: str, y, eta):
    """``y log p + (1 - y) log(1 - p)`` per row, accurate in both tails."""
    if link == "logit":
        log_p, log_q = -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta)
    elif link == "probit":
        log_p, log_q = log_ndtr(eta), log_ndtr(-eta)
    else:  # cloglog: 1 - p = exp(-exp(eta))
        t = np.exp(eta)
        log_q = -t
        with np.errstate(divide="ignore"):
            log_p = np.where(t > math.log(2.0), np.log1p(-np.exp(-t)), np.log(-np.expm1(-t)))
    return y * log_p + (1.0 - y) * log_q


def _score_information_rows(link: str, y, eta):
    """Per row, ``dl/deta`` and the Fisher information ``E[-d2l/deta2]``, accurate in both tails."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if link == "logit":
            p, q = expit(eta), expit(-eta)
            return y * q - (1.0 - y) * p, p * q
        if link == "probit":
            log_phi = -0.5 * eta * eta - 0.5 * math.log(2.0 * math.pi)
            mills_event = np.exp(log_phi - log_ndtr(eta))  # phi / Phi
            mills_non_event = np.exp(log_phi - log_ndtr(-eta))  # phi / (1 - Phi)
            return y * mills_event - (1.0 - y) * mills_non_event, mills_event * mills_non_event
        t = np.exp(eta)  # cloglog
        ratio = np.where(t > 0.0, t / np.expm1(t), 1.0)  # (dp/deta) / p
        return y * ratio - (1.0 - y) * t, t * ratio


class _Case:
    """One fixture, its design and the likelihood's own root."""

    def __init__(self, link, y, offset, x=None, weights=None):
        self.link = link
        self.y = np.asarray(y, dtype=float)
        self.offset = np.asarray(offset, dtype=float)
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.weights = np.ones(len(self.y)) if weights is None else np.asarray(weights, float)
        columns = [np.ones(len(self.y))] + ([] if self.x is None else [self.x])
        self.X = np.column_stack(columns)

    def score(self, beta):
        rows, _ = _score_information_rows(self.link, self.y, self.X @ beta + self.offset)
        return self.X.T @ (self.weights * rows)

    def information(self, beta):
        _, rows = _score_information_rows(self.link, self.y, self.X @ beta + self.offset)
        return self.X.T @ (self.X * (self.weights * rows)[:, None])

    def deviance(self, beta):
        rows = _log_likelihood_rows(self.link, self.y, self.X @ beta + self.offset)
        return -2.0 * float(np.sum(self.weights * rows))

    def root(self):
        """The maximum-likelihood coefficients: SciPy on the exact score."""
        if self.X.shape[1] == 1:
            # concave in the intercept: the score changes sign once
            beta = np.array(
                [brentq(lambda b: self.score(np.array([b]))[0], -30.0, 30.0, xtol=1e-14)]
            )
        else:
            beta = root(self.score, np.zeros(2), jac=lambda b: -self.information(b), tol=1e-13).x
        return beta + np.linalg.solve(self.information(beta), self.score(beta))

    def fit(self, link=None, **options):
        frame = pd.DataFrame(index=range(len(self.y)))
        features = {}
        if self.x is not None:
            frame, features = pd.DataFrame({"x": self.x}), {"x": Numeric()}
        model = SuperGLM(
            family="binomial",
            link=self.link if link is None else link,
            features=features,
            weight_semantics="frequency",
            **options,
        )
        model.fit(frame, self.y, offset=self.offset, sample_weight=self.weights)
        coefficients = np.r_[model.result.intercept, np.asarray(model.result.beta, dtype=float)]
        return model, coefficients


_FIVE = np.zeros(5)
_SLOPE_X = np.linspace(-1.0, 1.0, 40)
_SLOPE_Y = (np.arange(40) % 3 == 0).astype(float)

#: every case holds one row against its outcome at the likelihood's root
_HELD_AGAINST = {
    # below the logit band of -20
    "logit_event_at_minus_35": lambda: _Case("logit", np.r_[_FIVE, 1.0], np.r_[_FIVE, -35.0]),
    # inside the band, past the 1e-7 mean floor only (eta = -16.9 at the root)
    "logit_event_past_the_mean_floor": lambda: _Case(
        "logit", np.r_[_FIVE, 1.0], np.r_[_FIVE, -15.5]
    ),
    "logit_non_event_at_plus_35": lambda: _Case(
        "logit", np.r_[np.ones(5), 0.0], np.r_[_FIVE, 35.0]
    ),
    # a slope beside the intercept, and a frequency weight on the held row
    "logit_slope_and_weighted_event": lambda: _Case(
        "logit",
        np.r_[_SLOPE_Y, 1.0],
        np.r_[np.zeros(40), -30.0],
        x=np.r_[_SLOPE_X, 0.5],
        weights=np.r_[np.ones(40), 3.0],
    ),
    # probit: the band of +-20 is far, the mean floor acts from 5.2 standard deviations
    "probit_event_at_minus_9": lambda: _Case("probit", np.r_[_FIVE, 1.0], np.r_[_FIVE, -9.0]),
    "probit_non_event_at_plus_9": lambda: _Case(
        "probit", np.r_[np.ones(5), 0.0], np.r_[_FIVE, 9.0]
    ),
    "cloglog_event_at_minus_35": lambda: _Case("cloglog", np.r_[_FIVE, 1.0], np.r_[_FIVE, -35.0]),
    # above the cloglog cap of 2.736 at the root (eta = 2.87)
    "cloglog_non_event_past_the_cap": lambda: _Case(
        "cloglog", np.r_[(np.arange(50) % 2).astype(float), 0.0], np.r_[np.zeros(50), 4.5]
    ),
}


@pytest.fixture(params=sorted(_HELD_AGAINST))
def held_against(request) -> _Case:
    return _HELD_AGAINST[request.param]()


# ── what a user reads ──────────────────────────────────────────────


def test_a_row_held_against_its_outcome_leaves_the_fit_at_the_likelihoods_root(held_against):
    """Converged, and every coefficient within a thousandth of its standard error of the root.

    On the unfixed solver the logit and cloglog fits sit 0.97 to 6.8 standard
    errors away and the probit fits 1,040, all reported converged; the cloglog
    fit above its cap stops with every trial step rejected.
    """
    reference = held_against.root()
    standard_error = np.sqrt(np.diag(np.linalg.inv(held_against.information(reference))))
    model, coefficients = held_against.fit()
    assert model.result.converged
    assert np.all(np.abs(coefficients - reference) <= 1e-3 * standard_error)


def test_its_default_standard_errors_are_the_likelihoods_information(held_against):
    """``sqrt(diag(I^-1))`` with ``I = X' W X`` the Fisher information at the root."""
    reference = held_against.root()
    standard_error = np.sqrt(np.diag(np.linalg.inv(held_against.information(reference))))
    model, _ = held_against.fit()
    reported = np.array([row.se for row in model.summary()._coef_rows], dtype=float)
    np.testing.assert_allclose(reported, standard_error, rtol=1e-3)


def test_its_deviance_counts_the_held_row(held_against):
    """The fit's deviance is ``-2 log L``: an event at ``eta = -36`` adds 72, not ``-2 log 1e-7``."""
    model, _ = held_against.fit()
    exact = held_against.deviance(held_against.root())
    np.testing.assert_allclose(model.result.deviance, exact, rtol=1e-6)


def _fitted(case):
    """The fit, its coefficients, its frame and the likelihood's predictor at those coefficients."""
    model, coefficients = case.fit()
    frame = (
        pd.DataFrame(index=range(len(case.y))) if case.x is None else pd.DataFrame({"x": case.x})
    )
    return model, coefficients, frame, case.X @ coefficients + case.offset


def _statistics(case, eta):
    """Log-likelihood, null deviance and Pearson statistic of the likelihood at ``eta``."""
    log_likelihood = float(np.sum(case.weights * _log_likelihood_rows(case.link, case.y, eta)))
    null = _Case(case.link, case.y, case.offset, weights=case.weights)  # the intercept alone
    score, information = _score_information_rows(case.link, case.y, eta)
    # (y - p)^2 / (p (1 - p)) is the squared score over the information, under any link
    pearson = float(np.sum(case.weights * score**2 / information))
    return log_likelihood, null.deviance(null.root()), pearson


def test_its_summary_reads_the_likelihood(held_against):
    """``summary()``'s log-likelihood, AIC, BIC, null deviance and Pearson statistic are the likelihood's.

    The fit's deviance already is.  On the unfixed readers the logit fit with
    an event at -35 printed a log-likelihood of -17.23 beside a deviance of
    75.00 (the likelihood's is -37.50), an AIC of 36.47, a null deviance of
    32.28 for a model that is its own null, and a Pearson statistic of 1e7.
    """
    model, coefficients, _, eta = _fitted(held_against)
    log_likelihood, null_deviance, pearson = _statistics(held_against, eta)
    summary = model.summary()
    criteria, deviance = summary["information_criteria"], summary["deviance"]
    size, replicates = len(coefficients), float(np.sum(held_against.weights))
    np.testing.assert_allclose(criteria["log_likelihood"], log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(criteria["aic"], -2.0 * log_likelihood + 2.0 * size, rtol=1e-9)
    np.testing.assert_allclose(
        criteria["bic"], -2.0 * log_likelihood + math.log(replicates) * size, rtol=1e-9
    )
    np.testing.assert_allclose(deviance["null_deviance"], null_deviance, rtol=1e-9)
    np.testing.assert_allclose(
        deviance["explained_deviance"], 1.0 + 2.0 * log_likelihood / null_deviance, atol=1e-9
    )
    np.testing.assert_allclose(summary["fit"]["pearson_chi2"], pearson, rtol=1e-9)


def test_its_metrics_read_the_likelihood(held_against):
    """``metrics()`` on the fitted rows agrees with the fit: one deviance, one log-likelihood.

    The unfixed metrics read the floored mean, so ``metrics().deviance`` was
    34.47 for a fit whose own deviance was 75.00.
    """
    model, _, frame, eta = _fitted(held_against)
    log_likelihood, null_deviance, pearson = _statistics(held_against, eta)
    metrics = model.metrics(
        frame, held_against.y, sample_weight=held_against.weights, offset=held_against.offset
    )
    np.testing.assert_allclose(metrics.deviance, -2.0 * log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(metrics.log_likelihood, log_likelihood, rtol=1e-9)
    np.testing.assert_allclose(metrics.null_deviance, null_deviance, rtol=1e-9)
    np.testing.assert_allclose(metrics.pearson_chi2, pearson, rtol=1e-9)
    np.testing.assert_allclose(metrics.deviance, model.result.deviance, rtol=1e-9)


def test_its_residuals_are_the_likelihoods(held_against):
    """Deviance, Pearson, response, working and quantile residuals against 60-digit definitions.

    The held row's unfixed residuals were the floored mean's: a deviance
    residual of 5.68 where ``sqrt(-2 log p)`` is 8.53 for the logit event at
    -35, and a response residual one float short of 1.
    """
    model, _, frame, eta = _fitted(held_against)
    y, weights = held_against.y, held_against.weights
    metrics = model.metrics(frame, y, sample_weight=weights, offset=held_against.offset)
    log_p, log_q, event_score, non_event_score = (
        np.array(column) for column in zip(*(_exact_rows(held_against.link, v) for v in eta))
    )
    event = y > 0.0  # every fixture's response is 0 or 1
    expected = {
        "deviance": np.where(event, np.sqrt(-2.0 * log_p), -np.sqrt(-2.0 * log_q))
        * np.sqrt(weights),
        "pearson": np.where(event, np.exp(0.5 * (log_q - log_p)), -np.exp(0.5 * (log_p - log_q)))
        * np.sqrt(weights),
        "response": np.where(event, np.exp(log_q), -np.exp(log_p)),
        "working": np.where(event, 1.0 / non_event_score, -1.0 / event_score),
    }
    for kind, reference in expected.items():
        np.testing.assert_allclose(metrics.residuals(kind), reference, rtol=1e-9, err_msg=kind)
    # an outcome of probability below the residual's own 1e-10 floor sits on that floor
    rare = np.where(event, log_p, log_q) < math.log(1e-10)
    np.testing.assert_allclose(
        metrics.residuals("quantile")[rare],
        np.where(event, norm.ppf(1.0 - 1e-10), norm.ppf(1e-10))[rare],
        rtol=1e-12,
    )


def test_one_distant_event_does_not_move_the_other_rows_predictions():
    """Five non-events and one event share an intercept: each of the five is priced at 1/5.

    The sixth row's offset of -35 puts its own probability at ``2e-16``, so the
    score ``1 - 5 p = 0`` gives ``p = 1/5``.  The unfixed fit reports 0.0041.
    The fit is asked for the digits the comparison reads: at the default
    ``tol=1e-6`` its stop leaves the prediction 2e-6 from the root.
    """
    case = _HELD_AGAINST["logit_event_at_minus_35"]()
    model, _ = case.fit(tol=1e-10)
    predicted = model.predict(pd.DataFrame(index=range(1)), offset=np.zeros(1))
    np.testing.assert_allclose(predicted, 0.2, rtol=1e-6)


@pytest.mark.parametrize("discrete", [False, True])
def test_a_penalised_fit_balances_the_likelihoods_score(discrete):
    """With an unpenalised intercept the likelihood's row scores sum to zero at any smoothing.

    The reference needs no smoothing parameter: the penalty does not act on
    the intercept, so ``sum_i w_i dl_i/deta = 0`` at the penalised mode.  One
    event at an offset of -30 is held against its outcome; on the unfixed
    solver the scores sum to 0.98, the score the held row lost.
    """
    from superglm.solvers.mode_score import linear_predictor

    x = np.linspace(0.0, 1.0, 300)
    y = ((np.arange(300) * 7) % 10 < 3.0 + 4.0 * x).astype(float)
    frame = pd.DataFrame({"x": np.r_[x, 0.5]})
    response, offset = np.r_[y, 1.0], np.r_[np.zeros(300), -30.0]
    model = SuperGLM(
        family="binomial", link="logit", features={"x": Spline(n_knots=5)}, discrete=discrete
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, response, offset=offset)
    eta = linear_predictor(model._dm, model._solver_pirls_result(), model._fit_offset)
    score, _ = _score_information_rows("logit", response, np.asarray(eta, dtype=float))
    assert model.result.converged
    assert abs(score.sum()) <= 1e-6 * np.abs(score).sum()


@pytest.mark.parametrize("discrete", [False, True])
def test_a_penalised_fits_summary_reads_the_likelihood(discrete):
    """A REML fit's printed log-likelihood is ``-deviance / 2`` at its own fitted predictor.

    Same fixture as above.  The unfixed summary printed -215.08 beside a
    deviance of 457.47.
    """
    from superglm.solvers.mode_score import linear_predictor

    x = np.linspace(0.0, 1.0, 300)
    y = ((np.arange(300) * 7) % 10 < 3.0 + 4.0 * x).astype(float)
    frame = pd.DataFrame({"x": np.r_[x, 0.5]})
    response, offset = np.r_[y, 1.0], np.r_[np.zeros(300), -30.0]
    model = SuperGLM(
        family="binomial", link="logit", features={"x": Spline(n_knots=5)}, discrete=discrete
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, response, offset=offset)
    eta = np.asarray(
        linear_predictor(model._dm, model._solver_pirls_result(), model._fit_offset), dtype=float
    )
    log_likelihood = float(np.sum(_log_likelihood_rows("logit", response, eta)))
    summary = model.summary()
    np.testing.assert_allclose(
        summary["information_criteria"]["log_likelihood"], log_likelihood, rtol=1e-9
    )
    np.testing.assert_allclose(summary["deviance"]["deviance"], -2.0 * log_likelihood, rtol=1e-9)


def _observed_information_rows(link: str, y, eta):
    """``-d2l/deta2`` per row, by central differences of the closed-form score."""
    step = 1e-5
    ahead, _ = _score_information_rows(link, y, eta + step)
    behind, _ = _score_information_rows(link, y, eta - step)
    return (behind - ahead) / (2.0 * step)


def _laplace_criterion(link, X, S, y, weights, offset):
    """Wood's (2011) Laplace criterion of one smoothing parameter, as a function of its log.

    ``V = -l(b) + lam b'Sb / 2 + log|X'WX + lam S| / 2 - rank(S) log(lam) / 2``
    with ``W`` the observed information and ``b`` the penalised mode, found by
    damped Newton steps on the closed-form likelihood.
    """
    rank = np.linalg.matrix_rank(S)
    state = {"beta": np.zeros(X.shape[1])}

    def penalised(beta, lam):
        rows = _log_likelihood_rows(link, y, X @ beta + offset)
        return float(np.sum(weights * rows) - 0.5 * lam * beta @ S @ beta)

    def criterion(log_lam):
        lam, beta = math.exp(log_lam), state["beta"]
        for _ in range(200):
            eta = X @ beta + offset
            score, _ = _score_information_rows(link, y, eta)
            curvature = weights * _observed_information_rows(link, y, eta)
            hessian = X.T @ (X * curvature[:, None]) + lam * S
            step = np.linalg.solve(hessian, X.T @ (weights * score) - lam * S @ beta)
            length, base = 1.0, penalised(beta, lam)
            while penalised(beta + length * step, lam) < base - 1e-12 and length > 1e-10:
                length *= 0.5
            beta = beta + length * step
            if np.max(np.abs(length * step)) < 1e-11:
                break
        state["beta"] = beta
        return -penalised(beta, lam) + 0.5 * np.linalg.slogdet(hessian)[1] - 0.5 * rank * log_lam

    return criterion


@pytest.mark.parametrize(("link", "held_offset"), [("probit", -9.0), ("cloglog", -30.0)])
def test_a_penalised_fit_selects_the_likelihoods_smoothing_parameter(link, held_offset):
    """The REML smoothing parameter is the minimiser of the exact Laplace criterion (SciPy).

    Five weighted events are held against their outcome.  The criterion's
    Hessian reads each row's observed information: about 1 for a probit event
    in the lower tail, and about 0 as the clipped mean gives it, which put
    the unfixed probit fit's smoothing parameter 3.2% from the minimiser.  The
    reference shares the model's design and penalty matrix and nothing of its
    solver.
    """
    from superglm.reml.penalty_algebra import build_penalty_matrix

    n = 300
    x = (np.arange(n) + 0.5) / n
    y = (np.arange(n) * 0.6180339887498949) % 1.0 < 0.15 + 0.7 * np.sin(2.5 * np.pi * x) ** 2
    frame = pd.DataFrame({"x": np.r_[x, np.linspace(0.2, 0.8, 5)]})
    response = np.r_[y.astype(float), np.ones(5)]
    offset, weights = (
        np.r_[np.zeros(n), np.full(5, held_offset)],
        np.r_[np.ones(n), np.full(5, 4.0)],
    )
    model = SuperGLM(
        family="binomial",
        link=link,
        features={"x": Spline(n_knots=8)},
        weight_semantics="frequency",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, response, offset=offset, sample_weight=weights)
    (name, fitted), design = next(iter(model._reml_lambdas.items())), model._dm
    X = np.column_stack([np.ones(design.n)] + [design.matvec(unit) for unit in np.eye(design.p)])
    S = np.zeros((design.p + 1, design.p + 1))
    S[1:, 1:] = build_penalty_matrix(
        list(design.group_matrices), model._groups, {name: 1.0}, design.p, model._reml_penalties
    )
    criterion = _laplace_criterion(link, X, S, response, weights, offset)
    start = math.log(fitted)
    reference = minimize_scalar(criterion, bracket=(start - 1.0, start + 1.0), tol=1e-10)
    assert model.result.converged
    assert abs(start - reference.x) <= {"probit": 1e-3, "cloglog": 2e-4}[link]


@pytest.mark.parametrize(
    ("link", "etas"),
    [
        ("logit", [-40.0, -9.0, 0.3, 9.0, 40.0]),
        ("probit", [-20.5, -9.0, -5.3, 0.3, 5.3, 9.0]),
        ("cloglog", [-40.0, -20.5, -9.0, 0.3, 2.8, 4.5]),
    ],
)
def test_tail_curvature_rows_are_the_likelihoods_derivatives(link, etas):
    """Observed information and its first two ``eta`` derivatives against 400-digit differences.

    These are the rows the Laplace criterion and its smoothing-parameter
    derivatives read on a held row.  The logit's and cloglog's are sums of
    terms of one sign, held to ``1e-11`` of their value.  The probit's form
    ``l1 + eta`` by a difference that loses ``eta^2`` per derivative, so each
    is held to ``8 u eta^6`` of a curvature of order one: ``7e-8`` at -20.5.
    """
    import mpmath as mp

    from superglm.links import CloglogLink, LogitLink, ProbitLink
    from superglm.solvers.clip_rows import tail_observed_curvature

    def log_likelihood(response, x):
        if link == "logit":
            return -mp.log1p(mp.exp(-x)) if response else -mp.log1p(mp.exp(x))
        if link == "probit":
            return mp.log(mp.ncdf(x)) if response else mp.log(mp.ncdf(-x))
        return mp.log(-mp.expm1(-mp.exp(x))) if response else -mp.exp(x)

    link_object = {"logit": LogitLink, "probit": ProbitLink, "cloglog": CloglogLink}[link]()
    for response in (1, 0):
        rows = tail_observed_curvature(
            link_object, np.full(len(etas), float(response)), np.asarray(etas)
        )
        with mp.workdps(400):
            exact = np.array(
                [
                    [
                        float(
                            -mp.diff(
                                lambda x: log_likelihood(response, x),
                                eta,
                                order,
                                h=mp.mpf(10) ** -40,
                            )
                        )
                        for eta in etas
                    ]
                    for order in (2, 3, 4)
                ]
            )
        cancellation = 8.0 * 2.0**-53 * np.maximum(1.0, np.abs(etas)) ** 6
        allowance = cancellation if link == "probit" else 1e-300
        for row, reference in zip(rows, exact, strict=True):
            assert np.all(np.abs(row - reference) <= 1e-11 * np.abs(reference) + allowance)


class _UndeclaredLogit(LogitLink):
    """A logit the solver has no closed-form tail rows for: exact types declare those."""


def test_a_stop_the_clip_decided_is_not_reported_as_converged():
    """A link without tail rows of its own keeps the clip, and says the fit did not converge.

    The fit below stops 6.8 standard errors from the likelihood's root: its
    score there is 0.98, not zero.  A fit may stop at such a point only as
    not converged, with a warning that names the held rows.
    """
    case = _HELD_AGAINST["logit_event_at_minus_35"]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, coefficients = case.fit(link=_UndeclaredLogit())
    step = np.linalg.solve(case.information(coefficients), case.score(coefficients))
    newton_decrement = float(step @ case.score(coefficients))
    assert newton_decrement > 1.0  # the control: this state is far from the root
    assert not model.result.converged
    assert any("clip" in str(item.message) for item in caught)


def test_default_standard_error_when_every_row_sits_in_a_tail():
    """Offsets of +-19 and +-35 with events at the positive ones: no row is against its outcome.

    The fit is at the likelihood's root today (see the control below); its
    default intercept error is 11,844.5 where ``1 / sqrt(sum p (1 - p))`` at
    the fitted predictor is 2,987.3.
    """
    offset = np.tile([-19.0, 19.0, -35.0, 35.0], 10)
    case = _Case("logit", (offset > 0.0).astype(float), offset)
    model, coefficients = case.fit()
    expected = 1.0 / math.sqrt(case.information(coefficients)[0, 0])
    np.testing.assert_allclose(model.summary()._coef_rows[0].se, expected, rtol=1e-3)


@pytest.mark.pins_behaviour
def test_rows_held_on_the_side_of_their_outcome_leave_the_fit_at_the_root():
    """Pins: a fit whose held rows all agree with their outcome is already at the root.

    Every row of this fixture is past the 1e-7 mean floor and half are past
    the +-20 band, each on the side its response agrees with (non-events
    low, events high).  The held rows' scores are below ``1e-7`` each, the
    intercept is 0 by symmetry, and the fit reports converged.  The fix for
    #734 must leave this fit where it is.
    """
    offset = np.tile([-19.0, 19.0, -35.0, 35.0], 10)
    case = _Case("logit", (offset > 0.0).astype(float), offset)
    model, coefficients = case.fit()
    assert model.result.converged
    assert abs(case.score(coefficients)[0]) <= 1e-12
    assert abs(coefficients[0]) <= 1e-6


# ── the rows themselves, and the value that says which a fit reads ──


def _exact_rows(link: str, eta: float):
    """``log p``, ``log(1 - p)``, ``p'/p``, ``p'/(1 - p)`` at 60 digits: the definitions, not the closed forms."""
    import mpmath as mp

    with mp.workdps(60):
        x = mp.mpf(eta)
        if link == "logit":
            p, q = 1 / (1 + mp.exp(-x)), 1 / (1 + mp.exp(x))
            slope = p * q
        elif link == "probit":
            p, q, slope = mp.ncdf(x), mp.ncdf(-x), mp.npdf(x)
        else:
            q = mp.exp(-mp.exp(x))
            p, slope = -mp.expm1(-mp.exp(x)), mp.exp(x) * q
        # each log from the smaller of the pair, so neither rounds to log(1)
        log_p = mp.log(p) if p < q else mp.log1p(-q)
        log_q = mp.log(q) if q < p else mp.log1p(-p)
        return tuple(float(v) for v in (log_p, log_q, slope / p, slope / q))


@pytest.mark.parametrize(
    ("link", "etas"),
    [
        ("logit", [-700.0, -40.0, -9.0, 0.3, 9.0, 40.0, 700.0]),
        ("probit", [-40.0, -9.0, -5.3, 0.3, 5.3, 9.0, 30.0]),
        ("cloglog", [-40.0, -20.5, -9.0, 0.3, 2.8, 4.5]),
    ],
)
def test_tail_rows_are_the_likelihoods_rows_in_both_tails(link, etas):
    """Log-likelihood, score and Fisher information against the definitions at 60 digits.

    The predictors pass the working clip on both sides (``+-20``, the cloglog
    cap of 2.736, the mean floor); a row read at the clip is wrong there by
    orders of magnitude, so each number is held to ``1e-11`` of its value.
    """
    from superglm.links import CloglogLink, LogitLink, ProbitLink
    from superglm.solvers.clip_rows import tail_rows

    link_object = {"logit": LogitLink, "probit": ProbitLink, "cloglog": CloglogLink}[link]()
    eta = np.asarray(etas)
    log_p, log_q, event_score, non_event_score = (
        np.array(column) for column in zip(*(_exact_rows(link, value) for value in etas))
    )
    for response, log_likelihood, score in (
        (1.0, log_p, event_score),
        (0.0, log_q, -non_event_score),
    ):
        rows = tail_rows(link_object, np.full(len(etas), response), eta)
        np.testing.assert_allclose(rows[0], log_likelihood, rtol=1e-11, atol=1e-300)
        np.testing.assert_allclose(rows[1], score, rtol=1e-11, atol=1e-300)
        np.testing.assert_allclose(rows[2], event_score * non_event_score, rtol=1e-11, atol=1e-300)
        assert np.all(rows[3] >= 0.0)


def test_the_rows_a_fit_reads_move_one_way():
    """``ClipRows`` walks its table: ``working`` to ``likelihood`` on a held row, and never back."""
    from superglm.solvers.clip_rows import (
        CLIP_ROWS_MOVES,
        ClipRows,
        clip_rows_after,
        note_held_against,
    )

    assert CLIP_ROWS_MOVES == {
        ("working", "held_against"): "likelihood",
        ("likelihood", "held_against"): "likelihood",
    }
    rows = ClipRows()
    assert rows.state == "working"
    for (state, _event), destination in CLIP_ROWS_MOVES.items():
        rows.state = state
        assert note_held_against(rows) == (state != destination)
        assert rows.state == destination
    # a fit publishes the rows it ended on, and a solve that continues from it starts there
    case = _HELD_AGAINST["logit_event_at_minus_35"]()
    model, _ = case.fit()
    assert model._solver_pirls_result().clip_rows == "likelihood"
    assert clip_rows_after(model._solver_pirls_result()).state == "likelihood"


def _group_lasso_fit(link, lam, held_offset):
    """A group-lasso fit of a slope and a three-level factor, one weighted event held against its outcome."""
    n = 60
    x = np.linspace(-1.0, 1.0, n)
    y = ((np.arange(n) * 7) % 10 < 3.0 + 3.0 * x).astype(float)
    frame = pd.DataFrame(
        {"x": np.r_[x, 0.5], "g": [*np.array(["a", "b", "c"])[np.arange(n) % 3], "a"]}
    )
    response, weights = np.r_[y, 1.0], np.r_[np.ones(n), 3.0]
    offset = np.r_[np.zeros(n), held_offset]
    model = SuperGLM(
        family="binomial",
        link=link,
        features={"x": Numeric(), "g": Categorical()},
        selection_penalty=lam,
        weight_semantics="frequency",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(frame, response, offset=offset, sample_weight=weights)
    return model, response, weights, offset, caught


@pytest.mark.parametrize(
    ("link", "lam", "held_offset"),
    [("logit", 0.5, -30.0), ("logit", 3.0, -30.0), ("probit", 0.5, -9.0), ("cloglog", 3.0, -30.0)],
)
def test_a_group_lasso_fit_reaches_the_penalised_likelihoods_minimum(link, lam, held_offset):
    """The proximal engine's fit minimises ``-log L + lam sum_g w_g ||b_g||`` (SciPy on the exact likelihood).

    A selection penalty takes the fit through the proximal engine, which moves
    to the likelihood's rows as the direct solver does.  The reference shares
    the model's design and group weights, and minimises over the groups the
    fit kept; a group the fit dropped must satisfy the zero group's condition,
    ``||X_g' s|| <= lam w_g`` on the likelihood's score.  The unfixed engine
    stopped at the clip, not converged, 0.2 to 1.7 from these coefficients.
    """
    from superglm.model.fit_state import fitted_penalty

    model, response, weights, offset, caught = _group_lasso_fit(link, lam, held_offset)
    result, design, groups = model._solver_pirls_result(), model._dm, model._groups
    X = np.column_stack([design.matvec(unit) for unit in np.eye(design.p)])
    lambda1 = fitted_penalty(model).lambda1
    beta = np.asarray(result.beta, dtype=float)
    kept = np.flatnonzero(beta != 0.0)

    def objective(theta):
        slopes = np.zeros(design.p)
        slopes[kept] = theta[1:]
        rows = _log_likelihood_rows(link, response, theta[0] + X @ slopes + offset)
        penalty = sum(group.weight * np.linalg.norm(slopes[group.sl]) for group in groups)
        return -float(np.sum(weights * rows)) + lambda1 * penalty

    fitted = np.r_[result.intercept, beta[kept]]
    reference = minimize(objective, fitted, method="BFGS", options={"gtol": 1e-10})
    eta = result.intercept + X @ beta + offset
    score, _ = _score_information_rows(link, response, eta)
    gradient = X.T @ (weights * score)
    np.testing.assert_allclose(fitted, reference.x, atol=1e-4)
    assert result.converged and result.clip_rows == "likelihood"
    assert not any("clip" in str(item.message) for item in caught)
    exact_deviance = -2.0 * float(np.sum(weights * _log_likelihood_rows(link, response, eta)))
    np.testing.assert_allclose(model.result.deviance, exact_deviance, rtol=1e-9)
    for group in groups:
        if not np.any(beta[group.sl]):
            assert np.linalg.norm(gradient[group.sl]) <= lambda1 * group.weight * (1.0 + 1e-6)


def test_a_selection_penalty_fit_stopped_by_the_clip_is_not_reported_as_converged():
    """Under a link with no tail rows the proximal engine keeps the clip, and says so.

    With a penalty too small to move the answer it stops 0.97 standard errors
    from the likelihood's root: not converged, ``clip_held``, with the warning.
    """
    case = _HELD_AGAINST["logit_slope_and_weighted_event"]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, coefficients = case.fit(link=_UndeclaredLogit(), selection_penalty=1e-6)
    reference = case.root()
    standard_error = np.sqrt(np.diag(np.linalg.inv(case.information(reference))))
    assert np.max(np.abs(coefficients - reference) / standard_error) > 0.5  # the control
    assert not model.result.converged
    assert model._solver_pirls_result().termination_reason == "clip_held"
    assert any("clip" in str(item.message) for item in caught)
