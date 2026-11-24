"""Binomial rows the working clip holds, read from the likelihood in the linear predictor.

The solver holds a binomial predictor inside a band (``links.stabilize_eta``:
logit and probit ``+-20``, cloglog ``[-20, 2.736]``) and its mean inside
``[1e-7, 1 - 1e-7]`` (``distributions.clip_mu``).  IRLS is the fixed-point
iteration ``beta+ = beta + (X'WX + S)^-1 (X's - S beta)``: its fixed points
are the roots of the working score ``s``, whatever positive ``W`` is, so a
guard may bound the metric ``W`` but not change ``s`` (Green 1984, JRSSB
46(2), section 1.2).  A held row's working score is not the likelihood's.  On
a row held *against its outcome* (an event below the lower edge, a non-event
above the upper one) the likelihood's score is of order one and the held
row's is not, so the fit's fixed point is not the maximum-likelihood estimate.

``tail_rows`` gives the likelihood's own rows in closed form in the unclipped
predictor, accurate in both tails, for the logit, probit and cloglog links
(exact types: a subclass may change the inverse link and must not inherit
them).  ``log(1 + e^x)`` and ``log(1 - e^-x)`` follow Maechler (2012,
*Accurately Computing log(1 - exp(-|a|))*, Rmpfr vignette); the probit rows
are the inverse Mills ratios.

``ClipRows`` is the one tagged value that says which rows a fit reads where
the clip holds one, with its table of moves and its only writer.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Binomial, Distribution, clip_mu
from superglm.links import (
    CloglogLink,
    IdentityLink,
    Link,
    LogitLink,
    LogLink,
    ProbitLink,
    stabilize_eta,
)

type ClipRowsState = Literal["working", "likelihood"]

#: The moves of one fit's ``ClipRows``, by event.  ``held_against`` is a
#: coefficient solve ending on a state that holds a row against its outcome.
#: One way: no event returns a fit to the working rows.
CLIP_ROWS_MOVES: dict[tuple[ClipRowsState, str], ClipRowsState] = {
    ("working", "held_against"): "likelihood",
    ("likelihood", "held_against"): "likelihood",
}

# A held row's working response stays within this many units of its
# predictor: its iteration weight is at least ``|score| / 2^26``.
_RESPONSE_REACH = 2.0**26


class ClipRows:
    """Which rows a fit reads where the working clip holds one: ``working`` or ``likelihood``.

    ``working`` is the solver's clipped row, on every fit whose coefficient
    solves end clear of a row held against its outcome.  ``likelihood`` is the
    row of ``tail_rows``.  The solves of one REML fit share one value, so a
    later solve starts on the rows an earlier one moved to.
    """

    __slots__ = ("state",)

    def __init__(self) -> None:
        self.state: ClipRowsState = "working"


def note_held_against(rows: ClipRows) -> bool:
    """Record a solve that ended on a row held against its outcome; whether the rows changed.

    The only writer of ``ClipRows.state`` (``CLIP_ROWS_MOVES``).
    """
    before = rows.state
    rows.state = CLIP_ROWS_MOVES[(before, "held_against")]
    return rows.state != before


def on_likelihood_rows(result: object) -> bool:
    """Whether the fit behind ``result`` ended on the likelihood's rows (``PIRLSResult.clip_rows``)."""
    return getattr(result, "clip_rows", "working") == "likelihood"


def clip_rows_after(result: object) -> ClipRows:
    """The rows a solve that continues from ``result`` starts on (``PIRLSResult.clip_rows``)."""
    rows = ClipRows()
    if on_likelihood_rows(result):
        note_held_against(rows)
    return rows


def clip_can_hold_against(family: object, link: object) -> bool:
    """Whether the working clip can hold a row of this family and link against its outcome.

    A binomial fit under any link but the log and identity ones, whose clipped
    rows are the boundary of their mean space and have their own verdict
    (``irls_state.mean_space_boundary_rows``).
    """
    return isinstance(family, Binomial) and not isinstance(link, LogLink | IdentityLink)


def declares_tail_rows(family: object, link: object) -> bool:
    """Whether ``tail_rows`` has this family and link's rows.  Exact types."""
    return type(family) is Binomial and type(link) in (LogitLink, ProbitLink, CloglogLink)


def held_rows(
    family: Distribution, link: Link, eta_unclipped: NDArray, eta: NDArray, mu: NDArray
) -> tuple[NDArray, NDArray]:
    """The rows the working clip holds at the lower end, and at the upper end.

    ``eta`` and ``mu`` are the state's stabilised predictor and clipped mean.
    A row is held when ``stabilize_eta`` moved its predictor or ``clip_mu``
    moved its mean; the inverse link is evaluated again only on the rows whose
    mean sits on one of ``clip_mu``'s two bounds.
    """
    low = eta_unclipped < eta
    high = eta_unclipped > eta
    floor, cap = clip_mu(np.array([0.0, 1.0]), family)
    edge = np.flatnonzero((mu == floor) | (mu == cap))
    if edge.size:
        with np.errstate(over="ignore", under="ignore"):
            raw = np.asarray(link.inverse(np.array(eta[edge], dtype=np.float64)))
        low[edge] |= raw < mu[edge]
        high[edge] |= raw > mu[edge]
    return low, high


def held_against_outcome(y: NDArray, weights: NDArray, low: NDArray, high: NDArray) -> NDArray:
    """The positive-weight held rows whose response contradicts the end they are held at."""
    return ((low & (y > 0.0)) | (high & (y < 1.0))) & (weights > 0.0)


def tail_rows(link: object, y: NDArray, eta: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Per row of unit weight: log-likelihood, score, Fisher information and observed information.

    All in the unclipped ``eta``, with ``l = y log p + (1 - y) log(1 - p)``:

    ======== ============================ ====================== ===========
    link     ``l``                        ``dl/deta``            Fisher
    ======== ============================ ====================== ===========
    logit    ``-y log(1 + e^-eta)         ``y q - (1 - y) p``    ``p q``
             - (1 - y) log(1 + e^eta)``
    probit   ``y log Phi(eta)             ``y l1 - (1 - y) l0``  ``l1 l0``
             + (1 - y) log Phi(-eta)``
    cloglog  ``y log(1 - e^-t)            ``y r - (1 - y) t``    ``t r``
             - (1 - y) t``
    ======== ============================ ====================== ===========

    with ``q = 1 - p``, the inverse Mills ratios ``l1 = phi / Phi`` and ``l0 =
    phi / (1 - Phi)``, ``t = e^eta`` and ``r = t / (e^t - 1)``.  The observed
    information ``-d2l/deta2`` is the logit's Fisher information, and the
    probit and cloglog rows of ``working_rows``; it is never negative (each
    log-likelihood is concave in ``eta``; Pratt 1981, JASA 76(373)), and is
    read only as an iteration's metric.  A predictor whose likelihood float64
    cannot hold gives a non-finite row.
    """
    from scipy.special import expit, log_ndtr

    from superglm.solvers.working_rows import _cloglog_observed_rows, _probit_observed_rows

    y = np.asarray(y, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    unit = np.ones_like(eta)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if type(link) is LogitLink:
            p, q = expit(eta), expit(-eta)
            log_likelihood = -(y * np.logaddexp(0.0, -eta) + (1.0 - y) * np.logaddexp(0.0, eta))
            information = p * q
            return log_likelihood, y * q - (1.0 - y) * p, information, information
        if type(link) is ProbitLink:
            observed, score = _probit_observed_rows(y=y, eta=eta, sample_weight=unit)
            log_phi = -0.5 * eta * eta - 0.5 * math.log(2.0 * math.pi)
            log_p, log_q = log_ndtr(eta), log_ndtr(-eta)
            information = np.exp(2.0 * log_phi - log_p - log_q)
            return y * log_p + (1.0 - y) * log_q, score, information, observed
        if type(link) is CloglogLink:
            observed, _ = _cloglog_observed_rows(y=y, eta=eta, sample_weight=unit)
            t = np.exp(eta)
            growth = np.expm1(t)
            # r = t / (e^t - 1): 1 where t underflows, 0 where e^t overflows
            ratio = np.where(np.isfinite(growth), np.where(t > 0.0, t / growth, 1.0), 0.0)
            log_p = np.where(t > math.log(2.0), np.log1p(-np.exp(-t)), np.log(-np.expm1(-t)))
            # each term only where the response carries it: 0 * inf is not a row
            event, non_event = y > 0.0, y < 1.0
            log_likelihood = np.where(event, y * log_p, 0.0) - np.where(
                non_event, (1.0 - y) * t, 0.0
            )
            score = np.where(event, y * ratio, 0.0) - np.where(non_event, (1.0 - y) * t, 0.0)
            information = np.where(np.isfinite(growth), t * ratio, 0.0)
            return log_likelihood, score, information, observed
    raise TypeError(f"no tail rows are declared for {type(link).__name__}")


def tail_working_rows(
    link: object, y: NDArray, eta: NDArray, weights: NDArray, *, observed: bool
) -> tuple[NDArray, NDArray]:
    """A held row's working weight and response about its own unclipped ``eta``.

    The weight is ``w`` times the row's information (the observed one for an
    iteration's metric, Fisher's for the exported rows), raised to ``|score| /
    2^26`` so the response ``eta + score / weight`` stays within ``2^26`` of the
    predictor.  ``weight * (response - eta)`` is the likelihood's score either
    way, so the fit's fixed point is the likelihood's.  A zero-weight row has
    weight 0 and response ``eta``.
    """
    _, score, fisher, observed_information = tail_rows(link, y, eta)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = weights * score
        # the observed rows are a metric only: one float64 cannot form adds nothing
        information = np.nan_to_num(observed_information, nan=0.0) if observed else fisher
        weight = np.maximum(weights * information, np.abs(score) / _RESPONSE_REACH)
        step = np.zeros_like(score)
        np.divide(score, weight, out=step, where=weight > 0.0)
    return weight, eta + step


def likelihood_working_rows(
    family: Distribution,
    link: Link,
    y: NDArray,
    weights: NDArray,
    eta_unclipped: NDArray,
    eta: NDArray,
    mu: NDArray,
    working_weights: NDArray,
    response: NDArray,
    *,
    observed: bool,
) -> tuple[NDArray, NDArray]:
    """Working weights and response with every held row's read from the likelihood.

    A held row takes ``tail_working_rows`` about its unclipped predictor,
    which is every other row's own, so ``weights * (response -
    eta_unclipped)`` is the likelihood's score on every row.  The same two
    arrays when the clip holds no positive-weight row.
    """
    low, high = held_rows(family, link, eta_unclipped, eta, mu)
    held = np.flatnonzero((low | high) & (weights > 0.0))
    if not held.size:
        return working_weights, response
    working_weights = np.array(working_weights, dtype=np.float64, copy=True)
    response = np.array(response, dtype=np.float64, copy=True)
    working_weights[held], response[held] = tail_working_rows(
        link, y[held], eta_unclipped[held], weights[held], observed=observed
    )
    return working_weights, response


def count_held_against(
    family: Distribution,
    link: Link,
    y: NDArray,
    weights: NDArray,
    eta_unclipped: NDArray,
    eta: NDArray,
    mu: NDArray,
) -> int:
    """Positive-weight rows a state holds against their outcome; 0 where the clip cannot."""
    if not clip_can_hold_against(family, link):
        return 0
    low, high = held_rows(family, link, eta_unclipped, eta, mu)
    return int(np.count_nonzero(held_against_outcome(y, weights, low, high)))


def tail_information(link: object, eta: NDArray, weights: NDArray) -> NDArray:
    """``w`` times the Fisher information of ``tail_rows`` at the unclipped ``eta``."""
    return weights * tail_rows(link, np.zeros_like(eta, dtype=np.float64), eta)[2]


def tail_deviance(link: object, y: NDArray, eta: NDArray, weights: NDArray) -> NDArray:
    """``-2 w l`` per row at the unclipped ``eta``: the binomial unit deviance, unclipped."""
    with np.errstate(invalid="ignore"):
        return -2.0 * weights * tail_rows(link, y, eta)[0]


def likelihood_deviance(
    family: Distribution,
    link: Link,
    y: NDArray,
    weights: NDArray,
    eta_unclipped: NDArray,
    eta: NDArray,
    mu: NDArray,
    deviance: float,
) -> float:
    """``deviance`` with every held row's term read from the likelihood.

    ``deviance`` is the clipped state's, ``sum w d(y, mu)``; the positive-weight
    rows the clip holds trade their clipped term for ``-2 w l`` at the
    unclipped predictor, and every other row keeps its own.
    """
    low, high = held_rows(family, link, eta_unclipped, eta, mu)
    held = np.flatnonzero((low | high) & (weights > 0.0))
    if not held.size:
        return deviance
    clipped = weights[held] * family.deviance_unit(y[held], mu[held])
    exact = tail_deviance(link, y[held], eta_unclipped[held], weights[held])
    with np.errstate(invalid="ignore"):
        return float(deviance + np.sum(exact - clipped))


def likelihood_information(
    family: Distribution,
    link: Link,
    eta_unclipped: NDArray,
    eta: NDArray,
    mu: NDArray,
    sample_weight: NDArray,
    information: NDArray,
) -> NDArray:
    """``information`` with every held row's entry read from the likelihood, for a declared pair.

    The Fisher weights a fitted model's covariance reads: a held row's clipped
    weight is the clip's (slope at the band's edge over the variance at the
    floored mean), not ``p'^2 / (p (1 - p))`` at its predictor.  Unchanged, and
    the same array, when no row is held or the pair declares no tail rows.
    """
    if not declares_tail_rows(family, link):
        return information
    eta_unclipped = np.asarray(eta_unclipped, dtype=np.float64)
    low, high = held_rows(family, link, eta_unclipped, eta, mu)
    held = np.flatnonzero(low | high)
    if not held.size:
        return information
    result = np.array(information, dtype=np.float64, copy=True)
    result[held] = tail_information(
        link, eta_unclipped[held], np.asarray(sample_weight, dtype=np.float64)[held]
    )
    return result


def _probit_event_curvature(eta: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """``a = -d2 log Phi / deta2 = l1 (l1 + eta)`` and its first two derivatives, ``l1 = phi / Phi``.

    ``l1' = -a`` gives ``a' = l1 - a (2 l1 + eta)`` and ``a'' = 2 a (a - 1) - a' (2 l1 + eta)``.
    """
    from scipy.special import log_ndtr

    mills = np.exp(-0.5 * eta * eta - 0.5 * math.log(2.0 * math.pi) - log_ndtr(eta))
    curvature = mills * (mills + eta)
    first = mills - curvature * (2.0 * mills + eta)
    return curvature, first, 2.0 * curvature * (curvature - 1.0) - first * (2.0 * mills + eta)


def _cloglog_event_curvature(t: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """``c = -d2 log(1 - e^-t) / deta2`` and its first two derivatives in ``eta``, ``t = e^eta``.

    With ``r = t / (e^t - 1)``, ``m = t / (1 - e^-t) - 1`` and ``k = 1 - r = t - m``,
    which have ``r' = -r m``, ``m' = (m + 1) k`` and ``k' = r m``:
    ``c = r m``, ``c' = r g`` and ``c'' = r (g' - m g)``, where
    ``g = (m + 1) k - m^2`` and ``g' = (m + 1) (k^2 + r m - 2 m k)``.
    ``m`` takes its series ``t/2 + t^2/12 - ...`` below ``t = 0.1``, where the
    difference cancels.
    """
    growth = np.expm1(t)
    ratio = np.where(np.isfinite(growth), np.where(t > 0.0, t / growth, 1.0), 0.0)
    small = t < 0.1
    gap = -np.expm1(-t)
    # t + expm1(-t) = t^2/2 - t^3/6 + ..., summed where the difference would cancel
    series, term, ts = np.zeros_like(t), np.ones_like(t), np.where(small, t, 0.0)
    for order in range(1, 13):
        term = term * (-ts) / order
        if order >= 2:
            series = series + term
    excess = np.where(t > 0.0, np.where(small, series, t + np.expm1(-t)) / gap, 0.0)
    rest = t - excess
    slope = (excess + 1.0) * rest - excess * excess
    slope_first = (excess + 1.0) * (rest * rest + ratio * excess - 2.0 * excess * rest)
    return ratio * excess, ratio * slope, ratio * (slope_first - excess * slope)


def tail_observed_curvature(
    link: object, y: NDArray, eta: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Per row of unit weight: the observed information ``-d2l/deta2`` and its first two derivatives in ``eta``.

    The rows of the Laplace criterion's Hessian and of its derivatives in the
    smoothing parameters (Wood 2011, JRSSB 73(1), section 3.4 and appendix C),
    in the unclipped predictor:

    ======== ====================== ========================================
    link     event                  non-event
    ======== ====================== ========================================
    logit    ``p q``, ``p q (q - p)``, ``p q (1 - 6 p q)`` for either outcome
    probit   ``a(eta)``, ``a'``,    ``a(-eta)``, ``-a'(-eta)``, ``a''(-eta)``
             ``a''``
    cloglog  ``c``, ``c'``, ``c''`` ``t``, ``t``, ``t``
    ======== ====================== ========================================

    (``_probit_event_curvature``, ``_cloglog_event_curvature``).  The logit's
    are its Fisher information's too.
    """
    y = np.asarray(y, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if type(link) is LogitLink:
            from scipy.special import expit

            p, q = expit(eta), expit(-eta)
            information = p * q
            return information, information * (q - p), information * (1.0 - 6.0 * information)
        if type(link) is ProbitLink:
            event = _probit_event_curvature(eta)
            mirrored = _probit_event_curvature(-eta)
            non_event = (mirrored[0], -mirrored[1], mirrored[2])
        elif type(link) is CloglogLink:
            t = np.exp(eta)
            event, non_event = _cloglog_event_curvature(t), (t, t, t)
        else:
            raise TypeError(f"no tail rows are declared for {type(link).__name__}")
        curvature, first, second = (
            _by_outcome(y, mine, other) for mine, other in zip(event, non_event, strict=True)
        )
        return curvature, first, second


def likelihood_observed_curvature(
    family: Distribution,
    link: Link,
    y: NDArray,
    weights: NDArray,
    eta_unclipped: NDArray,
    eta: NDArray,
    mu: NDArray,
    rows: tuple[NDArray | None, NDArray | None, NDArray | None],
) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
    """Observed weights and their first two ``eta`` derivatives, every held row's from the likelihood.

    ``rows`` are the clipped state's, ``None`` where one was not formed; a held
    row takes ``w`` times ``tail_observed_curvature`` at its unclipped
    predictor, the row a fit on the likelihood's rows converged on.  The same
    tuple when the clip holds no row.
    """
    low, high = held_rows(family, link, eta_unclipped, eta, mu)
    held = np.flatnonzero(low | high)
    if not held.size:
        return rows
    weight = np.asarray(weights, dtype=np.float64)[held]
    exact = tail_observed_curvature(link, np.asarray(y)[held], np.asarray(eta_unclipped)[held])
    patched: list[NDArray | None] = []
    for row, value in zip(rows, exact, strict=True):
        if row is not None:
            row = np.array(row, dtype=np.float64, copy=True)
            with np.errstate(invalid="ignore", over="ignore"):
                row[held] = np.where(weight > 0.0, weight * value, 0.0)
        patched.append(row)
    return patched[0], patched[1], patched[2]


def likelihood_fisher_curvature(
    family: Distribution,
    link: Link,
    weights: NDArray,
    eta_unclipped: NDArray,
    eta: NDArray,
    mu: NDArray,
    rows: tuple[NDArray | None, NDArray | None, NDArray | None],
) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
    """Fisher weights and their first two ``eta`` derivatives, every held logit row's from the likelihood.

    The logit's Fisher information is its observed one, so its held rows are
    ``likelihood_observed_curvature``'s.  Any other pair's ``rows`` come back
    as they are: its Laplace criterion reads observed rows.
    """
    if type(family) is not Binomial or type(link) is not LogitLink:
        return rows
    return likelihood_observed_curvature(
        family, link, np.zeros_like(eta_unclipped), weights, eta_unclipped, eta, mu, rows
    )


def _by_outcome(y: NDArray, event: NDArray, non_event: NDArray) -> NDArray:
    """``y * event + (1 - y) * non_event``, each term only where the response carries it."""
    with np.errstate(invalid="ignore", over="ignore"):
        return np.where(y > 0.0, y * event, 0.0) + np.where(y < 1.0, (1.0 - y) * non_event, 0.0)


class HeldRows:
    """The rows the working clip holds at one state, as the likelihood reads them at unit weight.

    A fit on the likelihood's rows (``on_likelihood_rows``) converged on these,
    so every figure a user reads from it takes them in place of the clipped
    mean's on the held rows ``index``.  With ``p`` the mean at the unclipped
    predictor, and ``e1 = p'/p`` and ``e0 = p'/(1 - p)`` the event's and the
    non-event's score of ``tail_rows``:

    ================= ======================================================
    unit deviance     ``-2 [y log p + (1 - y) log(1 - p)]``
    response residual ``y - p = y (1 - p) - (1 - y) p``
    Pearson residual  ``(y - p) / sqrt(p (1 - p)) = y sqrt(e1 / e0) - (1 - y) sqrt(e0 / e1)``
    working residual  ``(y - p) / p' = y / e0 - (1 - y) / e1``
    ================= ======================================================

    Each is a sum of one term per outcome, so neither tail cancels.  A figure
    float64 cannot hold (a Pearson residual past ``1.3e154``) is infinite.
    """

    __slots__ = ("index", "y", "log_p", "log_q", "event_score", "non_event_score", "_clipped")

    def __init__(
        self,
        family: Distribution,
        link: object,
        index: NDArray,
        y: NDArray,
        eta: NDArray,
        mu: NDArray,
    ) -> None:
        self.index = index
        self.y = y
        self.log_p, self.event_score = tail_rows(link, np.ones_like(eta), eta)[:2]
        self.log_q, non_event = tail_rows(link, np.zeros_like(eta), eta)[:2]
        self.non_event_score = -non_event
        # the clipped mean's unit deviance and squared Pearson residual, which the sums trade away
        self._clipped = (
            np.asarray(family.deviance_unit(y, mu), dtype=np.float64),
            (y - mu) ** 2 / family.variance(mu),
        )

    @property
    def deviance(self) -> NDArray:
        return -2.0 * _by_outcome(self.y, self.log_p, self.log_q)

    @property
    def complement(self) -> NDArray:
        """``1 - p``."""
        return np.exp(self.log_q)

    @property
    def residual(self) -> NDArray:
        return _by_outcome(self.y, self.complement, -np.exp(self.log_p))

    @property
    def pearson_residual(self) -> NDArray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            odds_against = np.sqrt(self.event_score / self.non_event_score)
            odds = np.sqrt(self.non_event_score / self.event_score)
        return _by_outcome(self.y, odds_against, -odds)

    @property
    def working_residual(self) -> NDArray:
        with np.errstate(divide="ignore", over="ignore"):
            return _by_outcome(self.y, 1.0 / self.non_event_score, -1.0 / self.event_score)

    def pearson(self, weights: NDArray) -> NDArray:
        """``sqrt(w)`` times the Pearson residual, an exact 0 on a row of weight 0."""
        weight = np.asarray(weights, dtype=np.float64)[self.index]
        with np.errstate(invalid="ignore"):
            return np.where(weight > 0.0, np.sqrt(weight) * self.pearson_residual, 0.0)

    def patched(self, rows: NDArray, held: NDArray) -> NDArray:
        """A copy of the per-row figure ``rows`` with the held rows' entries ``held``."""
        result = np.array(rows, dtype=np.float64, copy=True)
        result[self.index] = held
        return result

    def _gap(self, weights: NDArray, exact: NDArray, clipped: NDArray) -> float:
        weight = np.asarray(weights, dtype=np.float64)[self.index]
        carried = weight > 0.0
        with np.errstate(invalid="ignore", over="ignore"):
            return float(
                np.sum(weight[carried] * exact[carried] - weight[carried] * clipped[carried])
            )

    def deviance_gap(self, weights: NDArray) -> float:
        """``sum w d`` on the likelihood's rows less the clipped mean's, over the held rows.

        A log-likelihood changes by ``-1/2`` of it: the binomial unit deviance
        is ``-2 l`` under either weight contract.
        """
        return self._gap(weights, self.deviance, self._clipped[0])

    def pearson_gap(self, weights: NDArray) -> float:
        """Pearson's ``sum w (y - p)^2 / (p (1 - p))`` less the clipped mean's, over the held rows."""
        with np.errstate(over="ignore"):
            return self._gap(weights, self.pearson_residual**2, self._clipped[1])


def held_likelihood_rows(
    family: Distribution, link: Link, y: NDArray, eta_unclipped: NDArray
) -> HeldRows | None:
    """The likelihood's reading of the rows the clip holds at ``eta_unclipped``, or ``None``.

    ``None`` where the pair declares no tail rows or the clip holds no row.
    """
    if not declares_tail_rows(family, link):
        return None
    eta_unclipped = np.asarray(eta_unclipped, dtype=np.float64)
    eta = stabilize_eta(eta_unclipped, link)
    mu = clip_mu(link.inverse(eta), family)
    low, high = held_rows(family, link, eta_unclipped, eta, mu)
    index = np.flatnonzero(low | high)
    if not index.size:
        return None
    y = np.asarray(y, dtype=np.float64)
    return HeldRows(family, link, index, y[index], eta_unclipped[index], mu[index])


# the null intercept's bracket doubles this many times about the offsets' median
_NULL_BRACKET_DOUBLINGS = 64
_NULL_SCORE_BOUND = 2.0**1000


def likelihood_null_eta(
    family: Distribution, link: Link, y: NDArray, weights: NDArray, offset: NDArray | None
) -> NDArray | None:
    """The intercept-only model's unclipped predictor at the likelihood's maximum, or ``None``.

    The intercept ``b`` is the root of ``sum_i w_i dl_i/deta (b + offset_i)`` on
    the rows of ``tail_rows``.  Each row's log-likelihood is concave in its
    predictor (Pratt 1981, JASA 76(373)), so the sum falls in ``b`` and has one
    root, which Brent's method (1973, *Algorithms for Minimization without
    Derivatives*, ch. 4) finds inside a bracket doubled about minus the
    offsets' median.  ``None`` where the pair declares no tail rows, the
    positive-weight responses are all 0 or all 1 (no finite root), or no
    bracket is found.
    """
    if not declares_tail_rows(family, link):
        return None
    from scipy.optimize import brentq

    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    shift = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=np.float64)
    carried = weights > 0.0
    response, mass, carried_shift = y[carried], weights[carried], shift[carried]
    if not (np.any(response > 0.0) and np.any(response < 1.0)):
        return None

    def score(intercept: float) -> float:
        with np.errstate(invalid="ignore", over="ignore"):
            total = float(np.sum(mass * tail_rows(link, response, intercept + carried_shift)[1]))
        # an overflowed score keeps its sign and stays a number Brent's method can bracket
        return min(max(total, -_NULL_SCORE_BOUND), _NULL_SCORE_BOUND)

    centre = -float(np.median(carried_shift))
    reach = 1.0
    for _ in range(_NULL_BRACKET_DOUBLINGS):
        below, above = score(centre - reach), score(centre + reach)
        if math.isnan(below) or math.isnan(above):
            return None
        if below > 0.0 > above:
            return float(brentq(score, centre - reach, centre + reach)) + shift
        if below <= 0.0:
            centre -= reach
        if above >= 0.0:
            centre += reach
        reach *= 2.0
    return None
