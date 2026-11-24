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
from superglm.links import CloglogLink, IdentityLink, Link, LogitLink, LogLink, ProbitLink

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


def clip_rows_after(result: object) -> ClipRows:
    """The rows a solve that continues from ``result`` starts on (``PIRLSResult.clip_rows``)."""
    rows = ClipRows()
    if getattr(result, "clip_rows", "working") == "likelihood":
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
