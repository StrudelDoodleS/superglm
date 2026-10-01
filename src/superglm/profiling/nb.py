"""NB2 shape theta: the profile-score root, alternating with the mean fit.

Given a fitted mean, the NB2 log-likelihood in theta has a closed-form score
(Lawless 1987). theta is its bracketed root, alternating with a warm-started
refit of the mean until theta settles: the scheme of Venables & Ripley (2002,
ch. 7.4) and MASS ``glm.nb``. A fixed-start Newton step can ascend the
negative log-likelihood where the profile information turns negative, so the
root is bracketed. The interval inverts the likelihood-ratio test on the
fixed-mean profile.

References
----------
- Venables & Ripley (2002): Modern Applied Statistics with S, Ch 7.4.
- Lawless (1987): Negative binomial and mixed Poisson regression,
  Canadian Journal of Statistics 15(3), 209-225.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import digamma

from superglm.distributions import NegativeBinomial, clip_mu, weighted_log_likelihood
from superglm.links import stabilize_eta
from superglm.model.base import model_has_lambda1_targets, resolve_selection_penalty_for_fit
from superglm.model.fit_ops import (
    _reject_monotone_fit_conflicts,
    _solve_coefficients,
    _uses_direct_solver,
)
from superglm.model.fit_state import configured_family, configured_lambda2, configured_penalty
from superglm.model.reml_setup import collect_reml_groups
from superglm.profiling._scalar import (
    Interval,
    RecordedObjective,
    censoring_warnings,
    likelihood_ratio_interval,
    profile_plot,
    warn_caller,
)
from superglm.reml.penalty_algebra import build_penalty_context
from superglm.solvers.dispersion import (
    FREQUENCY_WEIGHTS,
    PRIOR_WEIGHTS,
    dispersion_likelihood_size,
    model_weight_semantics,
)
from superglm.solvers.mode_score import linear_predictor

#: Default search range for the NB2 shape parameter. Deliberately wide: these
#: are numerical guard rails for the bracketed solve, not a statistical prior.
#: The historical default of (0.1, 50.0) excluded routinely occurring true
#: values at both ends (heavy overdispersion sits below 0.1; near-Poisson data
#: pushes the profile optimum far above 50).
_THETA_DEFAULT_BOUNDS: tuple[float, float] = (1e-8, 1e8)
#: Geometric step used to bracket a sign change of the profile score.
_THETA_BRACKET_FACTOR = 10.0
#: The root is located to 1e-8 relative, far below the six significant digits
#: theta_hat is published at.
_THETA_ROOT_RTOL = 1e-8
#: The interval searches at least this range, widened to hold theta_hat.
_CI_RANGE = (0.01, 500.0)
#: theta_hat is published to six significant digits, a rounding of up to 5e-6
#: relative; roots to 1e-6 in log theta place each endpoint five times finer
#: than that at every scale the ten-decade range admits.
_CI_LOG_XTOL = 1e-6
#: Direct-route mean fits stop at this relative objective change, as master's
#: did. At the model's default 1e-6 the unrounded theta moved 4e-6 to 9e-6
#: relative, a unit in the sixth published digit; at 1e-8 theta_hat matches
#: master on every characterisation book for one or two more IRLS iterations
#: per fit. BCD converges linearly (40% more iterations at 1e-8) and master
#: left it at 1e-6, the model default.
_DIRECT_MEAN_FIT_TOL = 1e-8


class NBThetaBoundWarning(UserWarning):
    """The NB2 theta estimate sits on an active search bound.

    The reported ``theta_hat`` is the constrained boundary value, not an
    interior maximum-likelihood estimate, and the accompanying result carries
    ``converged=False``. An active upper bound usually means the data are no
    more dispersed than Poisson at the fitted mean (the profile likelihood
    increases toward the Poisson limit ``theta -> inf``); an active lower
    bound means overdispersion beyond the searchable range.
    """


#: Above this PSI ARGUMENT the profile score switches to its large-argument
#: expansion.  The threshold is compared against ``a = theta`` under the
#: frequency contract and ``a = w * theta`` under the prior one, because the
#: expansion is in the psi argument and it is that argument, not theta, which
#: has to be large for the series to converge.
#:
#: The naive form obtains an O(theta^-2) score by cancelling digamma,
#: logarithm, and ratio terms whose leading parts are O(theta^-1) computed
#: from O(log theta)-sized intermediates, so float64 loses the sign to
#: roundoff around theta ~ 1e7-1e8 -- exactly the near-Poisson regime the
#: widened bounds admit.
#:
#: The truncation error is governed by ``a`` alone: the expansion keeps the
#: Bernoulli terms through ``1/(12 a^2)`` and drops ``1/(120 a^4)``.  Measured
#: at the switch (``a = 1e5``, across w from 1 down to 1e-4 with theta raised
#: to match), the dropped tail is **6.7e-17 relative to the score** -- below
#: eps, and the same figure at every (w, theta) pair with the same product,
#: which is what confirms the switch belongs on ``a``.  The naive form is
#: still accurate there, so the two branches agree to ~1e-9 relative across
#: the switch.
_THETA_SCORE_ASYMPTOTIC_MIN = 1e5


def theta_score(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    theta: float,
    *,
    weight_semantics: str,
) -> float:
    """Closed-form NB2 profile score dl/dtheta at fixed mu (Lawless 1987).

    Under the prior contract ``w Y ~ NB2(w mu, w theta)``, and differentiating
    that in ``theta`` leaves every term below unchanged except the digamma
    pair, which becomes ``psi(w(y+theta)) - psi(w theta)``.  The rest cancels
    exactly: ``log(w theta) - log(w theta + w mu)`` is ``log theta -
    log(theta + mu)``, and ``(w y + w theta)/(w mu + w theta)`` is
    ``(y + theta)/(mu + theta)``.  At ``w == 1`` the two arms are the same
    expression.

    For large theta the direct expression cancels catastrophically: each of
    ``digamma(y+theta) - digamma(theta)``, ``log(theta) - log(theta+mu)``,
    and ``1 - (y+theta)/(mu+theta)`` is O(theta^-1) while their sum is
    O(theta^-2), so beyond ~1e7 the sign that drives the bracketing solve is
    roundoff rather than likelihood geometry. Above the switch point the
    score is evaluated by a controlled expansion built from the asymptotic
    psi series (Abramowitz & Stegun 6.3.18):

        psi(theta+y) - psi(theta)
            = log((theta+y)/theta) + (1/theta - 1/(theta+y))/2
              + (1/theta^2 - 1/(theta+y)^2)/12 - O(theta^-4)

    which combines with the remaining terms into

        score_i = [log1p(x) - x] + y/(2 theta (theta+y))
                  + (1/theta^2 - 1/(theta+y)^2)/12,
        x = (y - mu)/(theta + mu),

    every term individually O(theta^-2) with no cancellation: log1p(x) - x
    is -x^2/2 + O(x^3) evaluated with error ~eps*|x|, negligible against the
    other O(theta^-2) terms whenever it matters.
    """
    prior = weight_semantics == PRIOR_WEIGHTS
    if prior:
        # A zero prior weight is a row observed with infinite variance, so it
        # leaves the likelihood rather than contributing zero to it.  Both
        # digamma arguments would land on the psi pole and ``0 * (-inf + inf)``
        # is ``nan``, so the row has to go before the score is formed rather
        # than be multiplied out after.  ``nb_nll`` already satisfies this
        # row-deletion identity by construction; the score is the arm that did
        # not.  The frequency arm needs no subset -- a zero replication count
        # multiplies a finite row contribution -- and keeping its summation
        # verbatim is what stops shipped numbers drifting.
        carried = weights > 0.0
        if not np.all(carried):
            y, mu, weights = y[carried], mu[carried], weights[carried]
        if weights.size == 0:
            return 0.0
    # The prior contract scales every psi argument by the row weight. The
    # frequency arm's scale is 1.0, and x * 1.0 and x / 1.0 are exact, so its
    # score is unchanged bit for bit.
    psi_scale = weights if prior else 1.0
    # The expansion is in the psi argument; switching on the smallest one keeps
    # every row inside the regime the expansion was derived for. The minimum is
    # taken over the carried rows only, so one dropped row cannot pin every
    # theta to the direct branch.
    switch_argument = theta * float(np.min(weights)) if prior else theta
    if switch_argument >= _THETA_SCORE_ASYMPTOTIC_MIN:
        shifted = theta + y
        x = (y - mu) / (theta + mu)
        # psi(w a) - psi(w b) carries 1/w on the first correction and 1/w**2
        # on the second; the leading log term is scale-free.
        psi_tail = (
            0.5 * y / (theta * shifted) / psi_scale
            + (1.0 / theta**2 - 1.0 / shifted**2) / 12.0 / psi_scale**2
        )
        return float(np.sum(weights * (np.log1p(x) - x + psi_tail)))
    return float(
        np.sum(
            weights
            * (
                digamma(psi_scale * (y + theta))
                - digamma(psi_scale * theta)
                + np.log(theta)
                + 1.0
                - np.log(theta + mu)
                - (y + theta) / (mu + theta)
            )
        )
    )


def _theta_moment_start(
    y: NDArray, mu: NDArray, weights: NDArray, *, weight_semantics: str
) -> float:
    """Moment estimate of theta at a fitted mean; ``inf`` when there is no excess dispersion.

    Frequency counts replicate rows with Var(Y) = mu + mu^2 / theta, so
    sum(w ((y - mu)^2 - mu)) = sum(w mu^2) / theta. Under prior weights
    Var(Y) = (mu + mu^2 / theta) / w, so sum(w (y - mu)^2 - mu) = sum(mu^2) / theta
    over the rows that carry information. A start only: the bracketed root
    decides theta, and ``inf`` starts it at the upper bound, as does a ratio
    that is not a number.
    """
    if weight_semantics == PRIOR_WEIGHTS:
        carried = weights > 0.0
        y, mu, weights = y[carried], mu[carried], weights[carried]
        numerator, denominator = np.sum(mu * mu), np.sum(weights * (y - mu) ** 2 - mu)
    else:
        # The ratio is free of the counts' common scale, so the sums take w / max w
        # and stay finite where the counts' products overflow.
        weights = weights / np.max(weights)
        numerator, denominator = np.sum(weights * mu * mu), np.sum(weights * ((y - mu) ** 2 - mu))
    with np.errstate(invalid="ignore"):
        start = float(numerator / denominator) if denominator > 0.0 else math.inf
    return math.inf if math.isnan(start) else start


@dataclass(frozen=True)
class ThetaSolve:
    """The fixed-mean score root, or the bound the score never changed sign before."""

    theta: float
    at_lower: bool
    at_upper: bool

    @property
    def at_bound(self) -> bool:
        return self.at_lower or self.at_upper

    @property
    def side(self) -> str | None:
        """The bound the solve stopped on, or None at an interior root."""
        return "lower" if self.at_lower else "upper" if self.at_upper else None


def solve_theta(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    theta_start: float,
    *,
    weight_semantics: str,
    bounds: tuple[float, float],
) -> ThetaSolve:
    """Root of the fixed-mean NB2 score, bracketed by decades from the start (MASS ``theta.ml``).

    The walk moves the way the likelihood rises, so every bracket end lies on
    the ascending side and the root is the profile maximum. A score that keeps
    its sign up to a bound puts the maximum at or past it; that bound is
    returned and flagged.
    """
    lower, upper = bounds

    def score(theta: float) -> float:
        return theta_score(y, mu, weights, theta, weight_semantics=weight_semantics)

    start = min(max(theta_start, lower), upper)
    start_score = score(start)
    if start_score == 0.0:
        return ThetaSolve(start, start <= lower, start >= upper)
    direction = 1.0 if start_score > 0.0 else -1.0
    bound = upper if direction > 0.0 else lower
    near, far, far_score = start, start, start_score
    while far_score * direction > 0.0:
        if far == bound:
            return ThetaSolve(bound, direction < 0.0, direction > 0.0)
        near, far = far, min(max(far * _THETA_BRACKET_FACTOR**direction, lower), upper)
        far_score = score(far)
    root = brentq(
        score,
        min(near, far),
        max(near, far),
        xtol=np.finfo(np.float64).tiny,
        rtol=_THETA_ROOT_RTOL,
        maxiter=100,
    )
    return ThetaSolve(float(root), False, False)


def nb_nll(
    y: NDArray, mu: NDArray, weights: NDArray, theta: float, *, weight_semantics: str
) -> float:
    """Mean negative NB2 log-likelihood per unit of the contract's size, from the family density."""
    log_likelihood = weighted_log_likelihood(
        NegativeBinomial(theta), y, mu, weights, weight_semantics=weight_semantics
    )
    return -log_likelihood / dispersion_likelihood_size(weights, weight_semantics=weight_semantics)


class _MeanFit:
    """The NB2 mean at a given theta on one design, each fit warm-started from the last."""

    def __init__(self, model, X, y, sample_weight, offset):
        if model._splines is not None and not model._specs:
            model._auto_detect_features(X, sample_weight)
        # The design does not depend on theta, and the family may still read
        # "auto": build it under a numeric placeholder, then restore the family.
        family = configured_family(model)
        model.family = NegativeBinomial(theta=1.0)
        try:
            self.y, self.w, offset = model._build_design_matrix(X, y, sample_weight, offset)
        finally:
            model.family = family
        self.model = model
        self.penalty = configured_penalty(model)
        resolve_selection_penalty_for_fit(model, self.penalty, self.y, self.w)
        self.has_lambda1_targets = model_has_lambda1_targets(model)
        # Refused here, before the alternation, not by the publication refit after it.
        _reject_monotone_fit_conflicts(model, self.penalty, self.has_lambda1_targets)
        direct = _uses_direct_solver(model, self.penalty, self.has_lambda1_targets)
        self.tol = min(model._tol, _DIRECT_MEAN_FIT_TOL) if direct else model._tol
        # Once per design: RandomEffect and FactorSmooth penalties reach the
        # direct solver only as REML components.
        reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices) if direct else []
        self.reml_penalties = (
            build_penalty_context(model._dm.group_matrices, reml_groups)[0] if reml_groups else None
        )
        self.offset = np.zeros_like(self.y) if offset is None else offset
        self.warm_beta = self.warm_intercept = None

    def fit(self, theta: float) -> NDArray:
        model = self.model
        model._distribution = NegativeBinomial(theta)
        result = _solve_coefficients(
            model,
            self.y,
            self.w,
            self.offset,
            penalty=self.penalty,
            lambda2=configured_lambda2(model),
            has_lambda1_targets=self.has_lambda1_targets,
            max_iter=model._max_iter,
            tol=self.tol,
            record_diagnostics=False,
            convergence=model._convergence,
            beta_init=self.warm_beta,
            intercept_init=self.warm_intercept,
            reml_penalties=self.reml_penalties,
        )
        self.warm_beta, self.warm_intercept = result.beta, result.intercept
        eta = stabilize_eta(linear_predictor(model._dm, result, self.offset), model._link)
        return clip_mu(model._link.inverse(eta), model._distribution)


def estimate_nb_theta(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    theta_bounds: tuple[float, float] = _THETA_DEFAULT_BOUNDS,
    xatol: float = 1e-2,
    maxiter: int = 30,
    on_evaluation: Callable[[dict], None] | None = None,
) -> NBProfileResult:
    """Alternate the mean fit with the score root until theta moves by at most ``xatol`` relative.

    The caller has validated the inputs, the family and the bounds. The first
    root starts from a moment estimate at the first mean and later ones from
    the previous theta; the lower bound floors the relative step's scale.
    ``on_evaluation`` receives each step as ``{"theta", "nll"}`` while the
    alternation runs.
    """
    mean = _MeanFit(model, X, y, sample_weight, offset)
    semantics = model_weight_semantics(model)
    theta, rows, settled = 1.0, [], False
    for _ in range(maxiter):
        mu = mean.fit(theta)
        start = (
            theta if rows else _theta_moment_start(mean.y, mu, mean.w, weight_semantics=semantics)
        )
        solve = solve_theta(
            mean.y, mu, mean.w, start, weight_semantics=semantics, bounds=theta_bounds
        )
        nll = nb_nll(mean.y, mu, mean.w, solve.theta, weight_semantics=semantics)
        rows.append({"theta": solve.theta, "nll": nll})
        if on_evaluation is not None:
            on_evaluation(dict(rows[-1]))
        settled = abs(solve.theta - theta) <= xatol * max(solve.theta, theta_bounds[0])
        theta = solve.theta
        if settled:
            break
    # Published to six significant digits, the precision the family reports.
    theta_hat = float(f"{theta:.6g}")
    messages = _warn_unsettled(solve, settled, theta_bounds, theta_hat, maxiter)
    # An unsettled theta_hat is the last iterate, not the fixed-mean optimum:
    # the interval inverts the profile from that optimum instead, and says so.
    # A bound is censored instead.
    caution = (
        None
        if settled or solve.at_bound
        else (
            f"theta_hat={theta_hat:g} is the alternation's last iterate after {maxiter} mean "
            "fits, not the optimum at the published mean; the theta interval is inverted from "
            "that optimum."
        )
    )
    return NBProfileResult(
        theta_hat=theta_hat,
        nll=nb_nll(mean.y, mu, mean.w, theta_hat, weight_semantics=semantics),
        converged=settled and not solve.at_bound,
        evaluations=pd.DataFrame(rows, columns=["theta", "nll"]),
        warnings=messages,
        _y=mean.y,
        _mu=mu,
        _weights=mean.w,
        _weight_semantics=semantics,
        _bound_side=solve.side,
        _caution=caution,
    )


def _warn_unsettled(solve, settled, bounds, theta_hat, maxiter) -> list[str]:
    """Warn about, and return, an estimate on a bound and an alternation out of steps."""
    messages = []
    if solve.at_bound:
        messages.append(_bound_message(solve, bounds, theta_hat))
        warn_caller(messages[-1], NBThetaBoundWarning)
    if not settled:
        # MASS glm.nb warns when its alternation limit is reached; an unsettled
        # theta_hat is the last iterate, not the fixed point.
        messages.append(
            f"NB2 theta alternation did not settle in {maxiter} mean fits; "
            f"theta_hat={theta_hat:g} is the last iterate and the result reports converged=False."
        )
        warn_caller(messages[-1])
    return messages


def _bound_message(solve: ThetaSolve, bounds: tuple[float, float], theta_hat: float) -> str:
    side, bound, interpretation = (
        ("lower", bounds[0], "the data show overdispersion beyond the searchable range")
        if solve.at_lower
        else (
            "upper",
            bounds[1],
            "the profile likelihood increases toward the Poisson limit"
            " (the data are at most Poisson-dispersed at the fitted mean)",
        )
    )
    return (
        f"NB2 theta estimate hit the {side} search bound {bound:g}: "
        f"{interpretation}. theta_hat={theta_hat:g} is a constrained "
        "boundary value, not an interior optimum, and the result reports "
        "converged=False. Widen theta_bounds to search further, or "
        "reconsider the family."
    )


@dataclass
class NBProfileResult:
    """Profile-likelihood estimate of the NB2 shape theta.

    ``nll`` is the mean negative log-likelihood at ``theta_hat`` and the
    published fitted mean, which the interval and the plot measure against.
    ``evaluations`` lists each alternation step's theta and NLL in order.
    """

    theta_hat: float
    nll: float
    converged: bool
    evaluations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["theta", "nll"])
    )
    warnings: list[str] = field(default_factory=list)
    _y: NDArray | None = field(default=None, repr=False)
    _mu: NDArray | None = field(default=None, repr=False)
    _weights: NDArray | None = field(default=None, repr=False)
    _weight_semantics: str = field(default=FREQUENCY_WEIGHTS, repr=False)
    # The estimation bound theta_hat sits on ("lower"/"upper"), or None when interior.
    _bound_side: str | None = field(default=None, repr=False)
    _ci_cache: dict[float, Interval] = field(default_factory=dict, repr=False)
    # Why theta_hat is not the published mean's profile optimum, which the
    # interval is inverted from regardless.
    _caution: str | None = field(default=None, repr=False)
    # (theta, NLL, whether theta is the end of the interval's range) at that
    # optimum, solved once per published mean.
    _optimum_point: tuple[float, float, bool] | None = field(default=None, repr=False)
    # Cautions about one interval: theta_hat outside it, or a higher optimum it found.
    _ci_cautions: dict[float, list[str]] = field(default_factory=dict, repr=False)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a pickle; a result saved by superglm 0.35.0 is restated in this layout.

        One path for a result pickled on its own (what ``estimate_theta``
        returns) and for one inside a saved model (`_restate_v0_35`).
        """
        if "n_evaluations" in state:
            state = _restate_v0_35(state)
        self.__dict__.update(state)

    def _at_mean(self, y: NDArray, mu: NDArray, weights: NDArray) -> NBProfileResult:
        """The estimate restated at a fitted mean, whose NLL and interval it then describes."""
        y = np.array(y, dtype=np.float64)
        # The interval re-reads the response long after the caller's array may
        # have changed.
        y.setflags(write=False)
        return replace(
            self,
            nll=nb_nll(y, mu, weights, self.theta_hat, weight_semantics=self._weight_semantics),
            warnings=list(self.warnings),
            _y=y,
            _mu=mu,
            _weights=weights,
            _ci_cache={},
            _optimum_point=None,
            _ci_cautions={},
        )

    def _profile_nll(self, theta: float) -> float:
        return nb_nll(
            self._y, self._mu, self._weights, theta, weight_semantics=self._weight_semantics
        )

    @property
    def _size(self) -> float:
        return dispersion_likelihood_size(self._weights, weight_semantics=self._weight_semantics)

    def interval(self, alpha: float = 0.05) -> Interval:
        """Likelihood-ratio interval for theta on the fixed-mean profile, with censoring flags.

        A censored side, and a theta_hat that is not the fixed-mean optimum, is
        recorded in ``warnings`` when first computed and warned about on every
        call; the interval is then inverted from that optimum.
        """
        interval = self._interval(alpha)
        for message in [self._caution] * (self._caution is not None) + self._interval_warnings(
            alpha
        ):
            warn_caller(message)
        return interval

    def _interval_warnings(self, alpha: float) -> list[str]:
        """What one computed interval adds to the estimate's own caution."""
        alpha = float(alpha)
        censored = censoring_warnings(
            self._ci_cache[alpha], alpha, "theta", lambda _: "where its search stopped"
        )
        return self._ci_cautions[alpha] + censored

    def _optimum(self) -> tuple[float, float]:
        """theta and NLL of the fixed-mean profile's maximum the interval inverts from.

        That is the score root at the published mean, one bracketed O(n) solve.
        theta_hat sits there only to the alternation's tolerance, and not at all
        where the published mean was fitted differently (a REML publication, or
        an alternation or joint refinement that stopped short).
        """
        if self._optimum_point is None:
            low, high = self._log_search_range()
            root = solve_theta(
                self._y,
                self._mu,
                self._weights,
                self.theta_hat,
                weight_semantics=self._weight_semantics,
                bounds=(math.exp(low), math.exp(high)),
            )
            at_end = root.at_lower or root.at_upper
            self._optimum_point = (root.theta, self._profile_nll(root.theta), at_end)
        return self._optimum_point[:2]

    def _interval(self, alpha: float) -> Interval:
        """The interval, computed once and recorded; reports read it without a warning."""
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self._caution is not None and self._caution not in self.warnings:
            self.warnings.append(self._caution)
        if alpha not in self._ci_cache:
            optimum, optimum_nll = self._optimum()
            # A recorded value sums n row log-densities, each evaluated without
            # cancellation, so it errs by round-off only: recursive summation by
            # at most (n - 1) u sum|w l_i| (Higham 2002, section 4.2), where
            # sum|w l_i| = size * nll since no row's log-probability is positive.
            # eps = 2u leaves each row its own rounding.
            roundoff = self._y.size * np.finfo(np.float64).eps * abs(optimum_nll)
            # Rooted in log theta: the range spans up to eighteen decades, and
            # an endpoint's own magnitude is its only yardstick.
            found, log_centre, centre_nll = likelihood_ratio_interval(
                RecordedObjective(lambda log_theta: self._profile_nll(math.exp(log_theta))),
                math.log(optimum),
                optimum_nll,
                self._log_search_range(),
                alpha=alpha,
                scale=self._size,
                xtol=_CI_LOG_XTOL,
                error=lambda _: roundoff,
            )
            interval = Interval(
                math.exp(found.lower),
                math.exp(found.upper),
                found.lower_censored,
                found.upper_censored,
            )
            self._ci_cache[alpha] = interval
            level = f"{100.0 * (1.0 - alpha):g}%"
            cautions = []
            if log_centre != math.log(optimum):
                # The profile evaluated there, at exactly this float, is the new optimum.
                self._optimum_point = (math.exp(log_centre), centre_nll, False)
                cautions.append(
                    f"the {level} interval's search found theta={math.exp(log_centre):.6g} "
                    f"above the published mean's score root theta={optimum:.6g} on its profile "
                    "likelihood; the interval is inverted from that higher point."
                )
            if not interval.lower <= self.theta_hat <= interval.upper:
                centre, _, at_end = self._optimum_point
                where = (
                    f"at or beyond theta={centre:.6g}, where the interval's range ends"
                    if at_end
                    else f"theta={centre:.6g}"
                )
                cautions.append(
                    f"theta_hat={self.theta_hat:g} lies outside its {level} interval "
                    f"[{interval.lower:.4g}, {interval.upper:.4g}], which is inverted from the "
                    f"published mean's profile optimum, {where}; theta_hat was estimated at a "
                    "different mean, or before its alternation settled."
                )
            self._ci_cautions[alpha] = cautions
            # Near-Poisson data leave the upper side censored: the statistic
            # stays under its cutoff all the way to the searched range's end.
            self.warnings.extend(self._interval_warnings(alpha))
        return self._ci_cache[alpha]

    def _log_search_range(self) -> tuple[float, float]:
        """Where each side of the interval may look, in log theta.

        A side whose estimation bound held theta_hat stops there, censored: the
        estimate established no optimum beyond it.
        """
        log_hat = math.log(self.theta_hat)
        lower = math.log(min(_CI_RANGE[0], self.theta_hat / 100.0))
        upper = math.log(max(_CI_RANGE[1], self.theta_hat * 100.0))
        return (
            log_hat if self._bound_side == "lower" else lower,
            log_hat if self._bound_side == "upper" else upper,
        )

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """``(lower, upper)`` of :meth:`interval`; a censored side is where its search stopped."""
        interval = self.interval(alpha)
        return interval.lower, interval.upper

    def profile_plot(self, alpha: float = 0.05, ax=None):
        """Likelihood-ratio statistic on a 40-point log grid around the interval."""
        interval = self.interval(alpha)
        # The statistic is measured from the optimum the interval was inverted
        # from, evaluated at that same float, so it is exactly zero there.
        optimum, optimum_nll = self._optimum()
        # Each point is an O(n) likelihood; the grid reaches 30% of the
        # interval's log width past either end for context.
        low, high = math.log(interval.lower), math.log(interval.upper)
        margin = 0.3 * (high - low)
        grid = np.exp(np.linspace(low - margin, high + margin, 40))
        values = {float(theta): self._profile_nll(theta) for theta in (*grid, optimum)}
        ax = profile_plot(
            values,
            optimum,
            optimum_nll,
            scale=self._size,
            alpha=alpha,
            interval=interval,
            label="theta",
            ax=ax,
        )
        if self._caution is not None or self._ci_cautions[float(alpha)]:
            # theta_hat, where it is not the optimum the statistic is measured from.
            ax.axvline(self.theta_hat, c="k", lw=1, ls=":")
        ax.set_xscale("log")
        return ax


def _restate_v0_35(saved: dict[str, Any]) -> dict[str, Any]:
    """The fields of an ``NBProfileResult`` unpickled from superglm 0.35.0, in this layout.

    v0.35.0 kept each interval as a ``(lower, upper)`` tuple in ``_ci_cache``
    and no endpoint status: a side whose likelihood-ratio statistic never
    reached its cutoff was returned at the end of its search range,
    ``min(0.01, theta_hat / 100)`` below and ``max(500, theta_hat * 100)``
    above (``_CI_RANGE``, unchanged since), so a side at that end is censored.
    Its iterates (``cache``: theta to six significant digits, then NLL, the
    last at the published mean) become ``evaluations``.  The saved response,
    mean and weights are kept, so an interval at another level is inverted on
    the same fixed-mean profile; v0.35.0 recorded no estimation bound, so that
    interval's search range is not stopped at one.
    """
    theta_hat = float(saved["theta_hat"])
    ends = (min(_CI_RANGE[0], theta_hat / 100.0), max(_CI_RANGE[1], theta_hat * 100.0))
    intervals = {
        float(alpha): Interval(
            float(lower), float(upper), float(lower) == ends[0], float(upper) == ends[1]
        )
        for alpha, (lower, upper) in (saved.get("_ci_cache") or {}).items()
    }
    iterates = saved.get("cache") or {}
    restated = NBProfileResult(
        theta_hat=theta_hat,
        nll=float(saved["nll"]),
        converged=bool(saved["converged"]),
        evaluations=pd.DataFrame(
            {"theta": list(map(float, iterates)), "nll": list(map(float, iterates.values()))},
            columns=["theta", "nll"],
        ),
        _y=saved.get("_y"),
        _mu=saved.get("_mu"),
        _weights=saved.get("_weights"),
        _weight_semantics=saved.get("_weight_semantics", FREQUENCY_WEIGHTS),
        _ci_cache=intervals,
        _ci_cautions={alpha: [] for alpha in intervals},
    )
    for alpha in intervals:
        restated.warnings.extend(restated._interval_warnings(alpha))
    return vars(restated)
