"""``predict`` and the scores built on it, on binomial rows past the solver's clip (issue #743).

The solver holds a binomial predictor inside a band and its mean inside
``[1e-7, 1 - 1e-7]`` while it iterates.  ``predict`` returned that clipped
mean, so a row priced at ``1e-16`` read ``1e-7``, and ``metrics()`` on other
rows, the holdout term drop, the cross-validation scores and the editor's
metrics scored it there: an event at ``1e-16`` counted ``-2 log 1e-7 = 32.2``
of deviance where the likelihood gives 73.7.

Every reference is the binomial likelihood in closed form in the linear
predictor ``intercept + slope x + offset``, re-formed here from the fitted
coefficients, so neither tail is clipped.  That re-forming rounds the
predictor by a few ulp of 37, about ``1e-14``, which each figure carries
forward at most once: ``_RTOL`` is ``1e-10``.  The clipped figures miss by a
factor of two.
"""

from __future__ import annotations

import functools
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import expit, log_ndtr

from superglm import Numeric, SuperGLM
from superglm.distributions import Binomial, clip_mu
from superglm.editor.evaluation import EvaluationDataset
from superglm.editor.metrics import compute_dataset_metrics
from superglm.links import stabilize_eta
from superglm.model_selection import (
    _pooled_deviance_parts,
    _pooled_nll_parts,
    cross_validate,
    fold_adapter,
)
from superglm.stats.model_tests import _per_obs_log_likelihood

_RTOL = 1e-10
_LINKS = ("logit", "probit", "cloglog")
#: predictors at which the event's probability, then the non-event's, is about 1e-16
_TAIL_ETA = {"logit": (-36.8, 36.8), "probit": (-8.2, 8.2), "cloglog": (-36.8, 3.6)}

_N = 200
_TRAIN_X = np.linspace(-1.0, 1.0, _N)
#: events on 30% of the rows below zero and 60% of those above, in a fixed pattern
_TRAIN_Y = ((np.arange(_N) * 7) % 10 < 3 + 3 * (_TRAIN_X > 0.0)).astype(float)

#: two rows in each tail, one against its outcome and one agreeing, then four ordinary rows
_X = np.array([0.5, -0.5, 0.25, -0.25, -0.8, -0.3, 0.3, 0.8])
_Y = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
_WEIGHTS = np.array([2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0])
_TAIL = slice(0, 4)


def _log_p_log_q(link: str, eta):
    """``log p`` and ``log(1 - p)`` at the predictor ``eta``, accurate in both tails."""
    if link == "logit":
        return -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta)
    if link == "probit":
        return log_ndtr(eta), log_ndtr(-eta)
    t = np.exp(eta)  # cloglog: 1 - p = exp(-exp(eta))
    return np.where(t > math.log(2.0), np.log1p(-np.exp(-t)), np.log(-np.expm1(-t))), -t


def _log_information(link: str, eta):
    """Logarithm of the Fisher information ``p'^2 / (p (1 - p))`` at ``eta``."""
    log_p, log_q = _log_p_log_q(link, eta)
    if link == "logit":
        return log_p + log_q
    if link == "probit":
        return -eta * eta - math.log(2.0 * math.pi) - log_p - log_q
    return 2.0 * eta + log_q - log_p  # cloglog: p' = exp(eta) (1 - p)


def _model(link: str) -> SuperGLM:
    return SuperGLM(
        family="binomial", link=link, features={"x": Numeric()}, weight_semantics="frequency"
    )


def _predictor(model, x, offset):
    return model.result.intercept + float(np.asarray(model.result.beta)[0]) * x + offset


class _Rows:
    """A fit on ordinary rows, and eight other rows whose first four sit in the two tails."""

    def __init__(self, link: str, tail: tuple[float, float] | None = None):
        self.link = link
        self.model = _model(link)
        self.model.fit(pd.DataFrame({"x": _TRAIN_X}), _TRAIN_Y)
        low, high = _TAIL_ETA[link] if tail is None else tail
        target = np.r_[low, low, high, high, np.zeros(4)]
        self.offset = np.r_[(target - _predictor(self.model, _X, 0.0))[_TAIL], np.zeros(4)]
        self.frame = pd.DataFrame({"x": _X})

    def log_likelihood_rows(self, model=None, x=_X):
        """Per row, ``y log p + (1 - y) log(1 - p)`` at ``model``'s predictor on ``x``."""
        log_p, log_q = _log_p_log_q(self.link, self.eta(model, x))
        return np.where(_Y > 0.0, log_p, log_q)

    def eta(self, model=None, x=_X):
        return _predictor(self.model if model is None else model, x, self.offset)

    def deviance(self, model=None, x=_X) -> float:
        return -2.0 * float(np.sum(_WEIGHTS * self.log_likelihood_rows(model, x)))


@pytest.fixture(scope="module", params=_LINKS)
def rows(request) -> _Rows:
    return _Rows(request.param)


def test_predict_returns_the_links_inverse_past_the_clip(rows):
    """A row priced at ``1e-16`` predicts ``1e-16``, and one at ``1 - 1e-16`` its float64 value.

    The unfixed ``predict`` returned ``1e-7`` and ``1 - 1e-7`` for them.
    """
    log_p, _ = _log_p_log_q(rows.link, rows.eta())
    predicted = rows.model.predict(rows.frame, offset=rows.offset)
    np.testing.assert_allclose(predicted, np.exp(log_p), rtol=_RTOL)
    assert np.all(predicted[:2] < 1e-15)
    assert np.all(predicted[2:4] > 1.0 - 1e-15)


@pytest.mark.pins_behaviour
def test_predict_inside_the_clip_is_the_same_bit_for_bit(rows):
    """Pins: a row clear of the band and the mean floor predicts what it always did.

    The training rows' probabilities lie between 0.1 and 0.9, so the solver's
    band and floor leave them alone and ``predict`` is the clipped inverse of
    its own predictor, to the last bit, for each link.
    """
    frame = pd.DataFrame({"x": _TRAIN_X})
    link = rows.model._link
    eta = rows.model._predict_eta_raw_exact(frame, None)
    clipped = clip_mu(link.inverse(stabilize_eta(eta, link)), Binomial())
    assert np.array_equal(rows.model.predict(frame), clipped)
    assert 0.1 < clipped.min() and clipped.max() < 0.9


def test_metrics_on_other_rows_read_the_likelihood(rows):
    """Deviance, log-likelihood, Pearson statistic, residuals and leverage at each row's predictor.

    The unfixed metrics read the floored mean: 66.6 of deviance on the logit
    rows where the likelihood gives 150.1.  The leverage is ``W_i x_i' (X' W
    X)^-1 x_i`` with ``W`` the Fisher information at the row's own predictor.
    """
    eta, log_l = rows.eta(), rows.log_likelihood_rows()
    log_p, log_q = _log_p_log_q(rows.link, eta)
    metrics = rows.model.metrics(rows.frame, _Y, sample_weight=_WEIGHTS, offset=rows.offset)
    np.testing.assert_allclose(metrics.deviance, rows.deviance(), rtol=_RTOL)
    np.testing.assert_allclose(metrics.log_likelihood, -0.5 * rows.deviance(), rtol=_RTOL)
    # (y - p)^2 / (p (1 - p)) is the odds against the outcome that occurred
    odds_against = np.exp(np.where(_Y > 0.0, log_q - log_p, log_p - log_q))
    np.testing.assert_allclose(metrics.pearson_chi2, np.sum(_WEIGHTS * odds_against), rtol=_RTOL)
    np.testing.assert_allclose(
        metrics.residuals("deviance"),
        np.where(_Y > 0.0, 1.0, -1.0) * np.sqrt(-2.0 * _WEIGHTS * log_l),
        rtol=_RTOL,
    )
    information = _WEIGHTS * np.exp(_log_information(rows.link, eta))
    design = np.column_stack([np.ones(len(_X)), _X])
    covariance = np.linalg.inv(design.T @ (design * information[:, None]))
    # the four ordinary rows carry the information: their 2 x 2 system has condition 1e2
    np.testing.assert_allclose(
        metrics.leverage,
        information * np.einsum("ij,jk,ik->i", design, covariance, design),
        rtol=1e-8,
    )


def test_metrics_on_other_rows_equal_a_recomputation_from_predict(rows):
    """Where float64 holds a probability's two sides, ``predict`` gives the row's own deviance.

    The four ordinary rows and the two in the lower tail: ``-2 [y log p + (1 -
    y) log(1 - p)]`` from ``predict``'s ``p`` is the squared deviance residual
    over the weight.  In the upper tail ``1 - p`` rounds to 0 in ``p`` itself.
    The unfixed pair disagreed with the likelihood together: both read 1e-7.
    """
    held = np.r_[0, 1, 4:8]
    p = rows.model.predict(rows.frame, offset=rows.offset)[held]
    recomputed = -2.0 * np.where(_Y[held] > 0.0, np.log(p), np.log1p(-p))
    metrics = rows.model.metrics(rows.frame, _Y, sample_weight=_WEIGHTS, offset=rows.offset)
    residual = metrics.residuals("deviance")[held]
    np.testing.assert_allclose(residual**2 / _WEIGHTS[held], recomputed, rtol=_RTOL)
    np.testing.assert_allclose(recomputed[0], -2.0 * rows.log_likelihood_rows()[0], rtol=_RTOL)


def test_holdout_term_drop_reads_the_likelihood(rows):
    """Dropping ``x`` changes the deviance by the likelihood's own difference.

    Without ``x`` each row's predictor is the intercept plus its offset, the
    model's at ``x = 0``.  The unfixed drop read both deviances at the floored
    mean.
    """
    table = rows.model.term_drop_diagnostics(
        pd.DataFrame({"x": _TRAIN_X}),
        _TRAIN_Y,
        mode="holdout",
        X_val=rows.frame,
        y_val=_Y,
        sample_weight_val=_WEIGHTS,
        offset_val=rows.offset,
    )
    expected = rows.deviance(x=np.zeros(len(_X))) - rows.deviance()
    np.testing.assert_allclose(table["delta_deviance"].to_numpy(), [expected], rtol=1e-8)


class _OneFold:
    """Train on the ordinary rows, score the eight others."""

    def split(self, X, y=None, groups=None):
        yield np.arange(_N), np.arange(_N, _N + len(_X))


def test_cross_validation_scores_read_the_likelihood(rows):
    """The fold's deviance and negative log-likelihood, a unit of weight, at its own predictor.

    The unfixed scores read the floored mean: 8.3 a row on the logit rows
    where the likelihood gives 12.5.
    """
    frame = pd.DataFrame({"x": np.r_[_TRAIN_X, _X]})
    result = cross_validate(
        _model(rows.link),
        frame,
        np.r_[_TRAIN_Y, _Y],
        cv=_OneFold(),
        sample_weight=np.r_[np.ones(_N), _WEIGHTS],
        offset=np.r_[np.zeros(_N), rows.offset],
        scoring=("deviance", "nll"),
        return_estimators=True,
    )
    mean_deviance = rows.deviance(result.estimators[0]) / float(np.sum(_WEIGHTS))
    scores = result.fold_scores.iloc[0]
    np.testing.assert_allclose(scores["deviance"], mean_deviance, rtol=_RTOL)
    np.testing.assert_allclose(scores["nll"], 0.5 * mean_deviance, rtol=_RTOL)
    np.testing.assert_allclose(result.pooled_scores["deviance"], mean_deviance, rtol=_RTOL)
    np.testing.assert_allclose(result.pooled_scores["nll"], 0.5 * mean_deviance, rtol=_RTOL)


def test_row_costs_read_the_likelihood(rows):
    """Each row's cost, which ranks the editor's worst rows, is its weighted ``-2 log L``.

    The unfixed cost of an event priced at ``1e-16`` was the floor's 32.2 a
    unit of weight, below an ordinary row's share of a larger weight.
    """
    costs = fold_adapter(rows.model).row_costs(rows.model, rows.frame, _Y, _WEIGHTS, rows.offset)
    np.testing.assert_allclose(costs, -2.0 * _WEIGHTS * rows.log_likelihood_rows(), rtol=_RTOL)


def test_editor_metrics_read_the_likelihood(rows):
    """The editor's scores of a split are ``metrics()``'s, which are the likelihood's.

    The unfixed editor read every figure at the floored mean.
    """
    dataset = EvaluationDataset(
        name="validation",
        label="Validation",
        X=rows.frame,
        y=_Y,
        sample_weight=_WEIGHTS,
        offset=rows.offset,
    )
    scores = compute_dataset_metrics(rows.model, dataset)
    metrics = rows.model.metrics(rows.frame, _Y, sample_weight=_WEIGHTS, offset=rows.offset)
    np.testing.assert_allclose(scores["deviance"], rows.deviance(), rtol=_RTOL)
    np.testing.assert_allclose(scores["log_likelihood"], -0.5 * rows.deviance(), rtol=_RTOL)
    for name in ("deviance", "log_likelihood", "aic", "bic", "explained_deviance", "pearson_chi2"):
        np.testing.assert_allclose(scores[name], getattr(metrics, name), rtol=_RTOL, err_msg=name)


def test_per_row_log_likelihoods_read_the_likelihood(rows):
    """The rows the Vuong test compares are ``y log p + (1 - y) log(1 - p)`` at the predictor.

    The unfixed rows read the floored mean: ``log 1e-7 = -16.1`` for an event
    priced at ``1e-16``.
    """
    density = _per_obs_log_likelihood(rows.model, rows.frame, _Y, rows.offset)
    np.testing.assert_allclose(density, rows.log_likelihood_rows(), rtol=_RTOL)


@pytest.mark.pins_behaviour
def test_scores_of_rows_inside_the_clip_are_the_same_bit_for_bit(rows):
    """Pins: with no row past the clip, every score is the mean-based sum it always was.

    The four ordinary rows, scored alone: ``metrics()``, the editor and the
    holdout term drop read ``sum w d(y, mu)`` at ``predict``'s mean, with the
    same operations in the same order, so the figures are equal, not close.
    """
    keep = slice(4, 8)
    frame, y, weights = rows.frame.iloc[keep], _Y[keep], _WEIGHTS[keep]
    mu = rows.model.predict(frame)
    deviance = float(np.sum(weights * Binomial().deviance_unit(y, mu)))
    assert rows.model.metrics(frame, y, sample_weight=weights).deviance == deviance
    dataset = EvaluationDataset(name="v", label="V", X=frame, y=y, sample_weight=weights)
    assert compute_dataset_metrics(rows.model, dataset)["deviance"] == deviance
    intercept = rows.model.result.intercept
    without_x = rows.model._link.inverse(np.full(4, intercept))
    table = rows.model.term_drop_diagnostics(
        pd.DataFrame({"x": _TRAIN_X}),
        _TRAIN_Y,
        mode="holdout",
        X_val=frame,
        y_val=y,
        sample_weight_val=weights,
    )
    dropped = float(np.sum(weights * Binomial().deviance_unit(y, without_x)))
    np.testing.assert_allclose(table["delta_deviance"].to_numpy(), [dropped - deviance], rtol=1e-12)


@pytest.mark.pins_behaviour
def test_a_fit_that_never_moved_keeps_its_own_figures_on_its_own_rows():
    """Pins: ``metrics()`` on the rows of a fit that holds none against its outcome is the fit's.

    Every row sits past the mean floor on the side of its response, so the
    fit stays on the clipped rows (#734) and its deviance is the clipped
    one, ``2e-7`` a row.  ``metrics()`` reads those rows as the fit does, on
    the fitted frame and on a copy of it.
    """
    offset = np.tile([-19.0, 19.0, -35.0, 35.0], 10)
    y, frame = (offset > 0.0).astype(float), pd.DataFrame(index=range(40))
    model = SuperGLM(family="binomial", features={}, weight_semantics="frequency")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # rows in the tails
        model.fit(frame, y, offset=offset)
        for rows_ in (frame, frame.copy()):
            deviance = model.metrics(rows_, y, offset=offset).deviance
            np.testing.assert_allclose(deviance, model.result.deviance, rtol=1e-12)
    np.testing.assert_allclose(model.result.deviance, 40 * 2e-7, rtol=1e-6)


# ── rows whose loss, or whose weight, leaves the ordinary range ─────


@functools.cache
def _fit(link: str) -> SuperGLM:
    model = _model(link)
    model.fit(pd.DataFrame({"x": _TRAIN_X}), _TRAIN_Y)
    return model


def _rows_at(link: str, eta):
    """The fit under ``link``, rows at ``x = 0`` and the offsets that put them at ``eta``."""
    model = _fit(link)
    eta = np.asarray(eta, dtype=np.float64)
    return model, pd.DataFrame({"x": np.zeros(eta.size)}), eta - _predictor(model, 0.0, 0.0)


def _dataset(frame, y, weights, offset) -> EvaluationDataset:
    return EvaluationDataset(
        name="v", label="V", X=frame, y=y, sample_weight=weights, offset=offset
    )


#: cloglog predictors of an event priced at 2e-9, a non-event whose ``e^eta`` overflows, an ordinary row
_OVERFLOW_ETA = [-20.0, 800.0, 0.0]
_OVERFLOW_Y = np.array([1.0, 0.0, 0.0])


@pytest.mark.filterwarnings("ignore:.*non-integral sample_weight")
@pytest.mark.parametrize(
    ("weight", "loss"),
    [(math.exp(-200.0), 2.0 * math.exp(600.0)), (1.0, math.inf)],
    ids=["float64-holds-the-weighted-loss", "past-float64"],
)
def test_a_non_event_past_the_overflow_of_exp_keeps_its_weighted_loss(weight, loss):
    """A cloglog non-event at 800 costs ``2 w e^800``: ``7.5e260`` at ``w = e^-200``, else infinity.

    ``e^800`` overflows.  The unfixed scores then dropped the likelihood for
    every row past the clip, so that row cost the clip's ``-2 log 2e-7 = 30.8``
    a unit of weight and ranked below the event at -20 beside it, which cost
    32.2 where the likelihood gives 40.
    """
    model, frame, offset = _rows_at("cloglog", _OVERFLOW_ETA)
    weights = np.array([1.0, weight, 1.0])
    event = float(_log_p_log_q("cloglog", np.array([-20.0]))[0][0])
    expected = np.array([-2.0 * event, loss, 2.0])
    costs = fold_adapter(model).row_costs(model, frame, _OVERFLOW_Y, weights, offset)
    np.testing.assert_allclose(costs, expected, rtol=_RTOL)
    assert np.argmax(costs) == 1
    metrics = model.metrics(frame, _OVERFLOW_Y, sample_weight=weights, offset=offset)
    np.testing.assert_allclose(metrics.deviance, expected.sum(), rtol=_RTOL)
    np.testing.assert_allclose(metrics.log_likelihood, -0.5 * expected.sum(), rtol=_RTOL)
    scores = compute_dataset_metrics(model, _dataset(frame, _OVERFLOW_Y, weights, offset))
    np.testing.assert_allclose(scores["deviance"], expected.sum(), rtol=_RTOL)
    density = _per_obs_log_likelihood(model, frame, _OVERFLOW_Y, offset, sample_weight=weights)
    np.testing.assert_allclose(density, -0.5 * expected, rtol=_RTOL)


def test_an_edit_scored_on_a_loss_past_float64_has_no_statistics_and_says_why():
    """An editor's model re-scored on a cloglog non-event at 800 withholds its statistics.

    Its deviance on those rows is infinite, which no summary can print.  The
    unfixed re-scoring returned to the clipped rows there and printed 65.1.
    """
    from superglm.editor import edit

    model = _model("cloglog")
    model.fit(pd.DataFrame({"x": _TRAIN_X}), _TRAIN_Y)
    frame = pd.DataFrame({"x": np.zeros(3)})
    offset = np.array(_OVERFLOW_ETA) - _predictor(model, 0.0, 0.0)
    session = edit(model, train_data=(pd.DataFrame({"x": _TRAIN_X}), _TRAIN_Y, None, None))
    scored = session.to_model(X=frame, y=_OVERFLOW_Y, offset=offset)
    assert np.isnan(scored.result.deviance)
    with pytest.raises(RuntimeError, match="probability too small for float64"):
        scored.summary()


@pytest.mark.pins_behaviour
def test_a_zero_weight_row_past_the_overflow_of_exp_adds_an_exact_zero():
    """Pins: a cloglog non-event at 800 with weight 0 has a cost and a residual of 0.

    At the clipped mean it always had.  Read from the likelihood its unit
    deviance is infinite, and multiplying that by the weight gives NaN under
    a RuntimeWarning: the product is the row's own, an exact 0.
    """
    model, frame, offset = _rows_at("cloglog", _OVERFLOW_ETA)
    weights = np.array([1.0, 0.0, 1.0])
    metrics = model.metrics(frame, _OVERFLOW_Y, sample_weight=weights, offset=offset)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        costs = fold_adapter(model).row_costs(model, frame, _OVERFLOW_Y, weights, offset)
        residuals = metrics.residuals("deviance")
        standardized = metrics.std_deviance_residuals
    assert costs[1] == 0.0 and residuals[1] == 0.0 and standardized[1] == 0.0
    np.testing.assert_allclose(residuals**2, costs, rtol=_RTOL)
    assert np.all(np.isfinite(standardized))


@pytest.mark.filterwarnings("ignore:.*non-integral sample_weight")
def test_a_row_of_small_weight_is_not_rounded_out_of_the_sums():
    """An event at a logit of -100 with weight 1e-26, beside two agreeing rows: 2e-24 of deviance.

    The agreeing rows' clipped terms are 2e-7 each.  The unfixed sums added
    the likelihood's terms less the clipped ones to the clipped total, which
    rounds 2e-24 away: deviance, log-likelihood and null deviance all read 0.
    """
    eta = np.array([-100.0, 100.0, -100.0])
    model, frame, offset = _rows_at("logit", eta)
    y, weights = np.array([1.0, 1.0, 0.0]), np.array([1e-26, 1.0, 1.0])

    def deviance(shift: float) -> float:
        log_p, log_q = _log_p_log_q("logit", eta + shift)
        return -2.0 * float(np.sum(weights * np.where(y > 0.0, log_p, log_q)))

    metrics = model.metrics(frame, y, sample_weight=weights, offset=offset)
    scores = compute_dataset_metrics(model, _dataset(frame, y, weights, offset))
    assert 1.9e-24 < deviance(0.0) < 2.1e-24
    for value in (metrics.deviance, -2.0 * metrics.log_likelihood, scores["deviance"]):
        np.testing.assert_allclose(value, deviance(0.0), rtol=_RTOL)
    np.testing.assert_allclose(-2.0 * scores["log_likelihood"], deviance(0.0), rtol=_RTOL)
    for parts, share in ((_pooled_deviance_parts, 1.0), (_pooled_nll_parts, 0.5)):
        total, _ = parts(model, frame, y, sample_weight=weights, offset=offset)
        np.testing.assert_allclose(total, share * deviance(0.0), rtol=_RTOL)
    # the null model moves every predictor by the one shift at which its score is 0
    shift = brentq(
        lambda b: float(np.sum(weights * (y * expit(-eta - b) - (1.0 - y) * expit(eta + b)))),
        -150.0,
        150.0,
        xtol=1e-12,
    )
    np.testing.assert_allclose(metrics.null_deviance, deviance(shift), rtol=1e-8)


def test_the_fits_own_statistics_keep_a_row_of_small_weight():
    """The statistics ``summary()`` prints, which an editor edit re-forms, sum the same way.

    The three rows above as a fit on the likelihood's rows holds them, the
    event at weight 1e-60: the unfixed log-likelihood was 0 where the
    likelihood gives -7.4e-44, and its Pearson sum held 6 digits of 2.7e-17.
    """
    from superglm.links import LogitLink
    from superglm.model.fit_ops import _compute_fit_stats

    eta = np.array([-100.0, 100.0, -100.0])
    y, weights = np.array([1.0, 1.0, 0.0]), np.array([1e-60, 1.0, 1.0])
    family, link = Binomial(), LogitLink()
    mu = clip_mu(link.inverse(stabilize_eta(eta, link)), family)
    stats = _compute_fit_stats(
        y, mu, weights, None, family, link, 1.0, weight_semantics="frequency", likelihood_eta=eta
    )
    log_p, log_q = _log_p_log_q("logit", eta)
    log_likelihood = float(np.sum(weights * np.where(y > 0.0, log_p, log_q)))
    np.testing.assert_allclose(stats.log_likelihood, log_likelihood, rtol=_RTOL)
    odds_against = np.exp(np.where(y > 0.0, log_q - log_p, log_p - log_q))
    np.testing.assert_allclose(stats.pearson_chi2, np.sum(weights * odds_against), rtol=_RTOL)


@pytest.mark.filterwarnings("ignore:.*non-integral sample_weight")
def test_pearson_weights_a_row_before_it_squares_it():
    """An event at a logit of -800 with weight 1e-100 adds ``1e-100 e^800 = 2.7e247`` to Pearson's sum.

    The unfixed sum squared the residual ``e^400`` first, which overflows, and
    read infinity.  At -1500 the residual ``e^750`` overflows alone, and with
    weight 1e-200 the weighted one is ``e^(750 - 230.3)``: it read infinity too.
    """
    model, frame, offset = _rows_at("logit", [-800.0, 0.0, 0.0])
    y, weights = np.array([1.0, 0.0, 1.0]), np.array([1e-100, 1.0, 1.0])
    expected = math.exp(800.0 + math.log(1e-100)) + 2.0
    metrics = model.metrics(frame, y, sample_weight=weights, offset=offset)
    np.testing.assert_allclose(metrics.pearson_chi2, expected, rtol=_RTOL)
    scores = compute_dataset_metrics(model, _dataset(frame, y, weights, offset))
    np.testing.assert_allclose(scores["pearson_chi2"], expected, rtol=_RTOL)
    model, frame, offset = _rows_at("logit", [-1500.0, 0.0, 0.0])
    weights = np.array([1e-200, 1.0, 1.0])
    residual = model.metrics(frame, y, sample_weight=weights, offset=offset).residuals("pearson")
    np.testing.assert_allclose(residual[0], math.exp(750.0 + 0.5 * math.log(1e-200)), rtol=_RTOL)


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_cooks_distance_past_float64_is_formed_from_the_score(link):
    """Cook's distance where the Pearson residual overflows and the leverage underflows.

    ``w r^2 h`` is ``(w s)^2 g`` with ``s`` the likelihood's score and ``g = x'
    (X' W X)^-1 x``: an event at a probit predictor of -800 has the inverse
    Mills ratio for its score, about 800.  The unfixed distance multiplied an
    infinite ``r^2`` by a leverage of 0 and read NaN.
    """
    far = _Rows(link, (-800.0, 800.0))
    eta = far.eta()
    log_p, log_q = _log_p_log_q(link, eta)
    log_information = _log_information(link, eta)
    information = _WEIGHTS * np.exp(log_information)
    design = np.column_stack([np.ones(len(_X)), _X])
    covariance = np.linalg.inv(design.T @ (design * information[:, None]))
    form = np.einsum("ij,jk,ik->i", design, covariance, design)
    # s^2 is the odds against the outcome that occurred times the information, in logs
    square_score = np.exp(np.where(_Y > 0.0, log_q - log_p, log_p - log_q) + log_information)
    metrics = far.model.metrics(far.frame, _Y, sample_weight=_WEIGHTS, offset=far.offset)
    spread = (1.0 - information * form) ** 2 * metrics.effective_df
    expected = _WEIGHTS**2 * square_score * form / spread
    assert (
        np.all(np.isfinite(expected)) and expected[0] > 1.0
    )  # the control: the row is influential
    np.testing.assert_allclose(metrics.cooks_distance, expected, rtol=1e-8)


@pytest.mark.pins_behaviour
def test_drop1_of_a_fit_that_kept_no_state_reads_the_fits_own_deviance():
    """Pins: ``drop1`` under weights of ``2^-600`` re-forms the fit's deviance without its state.

    The rows past the clip all agree with their outcomes, so the fit reads
    them at the clip (#734).  A fit that kept no fitted mean has its deviance
    re-formed from a prediction, and ``predict`` now returns those rows' own
    probabilities, 2e-7 of deviance a row below the fit's: the re-formed
    deviance reads them as the fit does, and equals the one from its state.
    """
    n = 30
    offset = np.where(np.arange(n) % 2 == 0, -30.0, 30.0)
    y = (offset > 0.0).astype(float)
    y[:6], offset[:6] = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 0.0
    weights, frame = np.full(n, 2.0**-600), pd.DataFrame({"x": np.linspace(-1.0, 1.0, n)})
    tables = []
    for retain in (True, False):
        model = SuperGLM(family="binomial", features={"x": Numeric()}, retain_fit_state=retain)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # rows in the tails
            model.fit(frame, y, sample_weight=weights, offset=offset)
            table = model.drop1(frame, y, sample_weight=weights, offset=offset)
        assert model.result.weight_exponent  # the control: drop1 re-forms the deviance
        tables.append(table[["deviance_full", "deviance_reduced"]].to_numpy())
    np.testing.assert_allclose(tables[1], tables[0], rtol=1e-12)
