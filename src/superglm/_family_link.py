"""What is true of a family and a link together.

A family declares what is true of it alone (``distributions``) and a link what
is true of it alone (``links``). ``pair_of`` resolves the two into one object
that answers every question about the pair, so generic code reads a fact and
never names a class. Each fit resolves it once.

Two kinds of fact:

- an *exact* fact is keyed by the exact shipped classes. A subclass may change
  a likelihood or an inverse link, so it never inherits a closed form, a
  curvature claim or a cache key, and a user-defined family or link takes the
  generic path;
- an *inherited* fact follows the members the classes declare and
  ``isinstance``, so a subclass keeps its parent's domain.

The pair holds the family and the link and reads their parameters (Tweedie
``p``, the negative binomial's ``theta``, a power link's exponent) when asked,
so it is never stale and never stored on a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import (
    SHIPPED_FAMILIES,
    Binomial,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    mean_domain,
)
from superglm.links import (
    _BINOMIAL_CLIP_MU_EPS,
    SHIPPED_LINKS,
    CloglogLink,
    IdentityLink,
    InverseLink,
    LogitLink,
    LogLink,
    NegativeBinomialLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
    outside_eta_domain,
)

_CANONICAL = frozenset(
    {(Gaussian, IdentityLink), (Poisson, LogLink), (Binomial, LogitLink), (Gamma, InverseLink)}
)
# The Fisher weight over the sample weight, where it is one number for every row.
_CONSTANT_FISHER_WEIGHT = {
    (Gaussian, IdentityLink): 1.0,
    (Gamma, LogLink): 1.0,
    (Poisson, SqrtLink): 4.0,
}
_OBSERVED_NEWTON = frozenset({(Gamma, LogLink), (Tweedie, LogLink)})
_CONCAVE_OBSERVED_ROWS = _OBSERVED_NEWTON | {
    (NegativeBinomial, LogLink),
    (Binomial, ProbitLink),
    (Binomial, CloglogLink),
}
_PRESOLVE = frozenset(
    {
        (Poisson, LogLink),
        (Binomial, LogitLink),
        (Binomial, ProbitLink),
        (Binomial, CloglogLink),
        (Gamma, LogLink),
        (Tweedie, LogLink),
        (NegativeBinomial, LogLink),
    }
)
_CONVEX_DEVIANCE_LINKS: dict[type, tuple[type, ...]] = {
    Poisson: (LogLink, IdentityLink),
    Gamma: (InverseLink, LogLink),
    NegativeBinomial: (LogLink,),
    Tweedie: (LogLink,),
    Binomial: (LogitLink, ProbitLink, CloglogLink, LogLink),
}
_TAIL_ROWS = frozenset({(Binomial, LogitLink), (Binomial, ProbitLink), (Binomial, CloglogLink)})
_LOG_VARIANCE_POWER = {Gaussian: 0.0, Poisson: 1.0, Gamma: 2.0}
_FIXED_CACHE_LINKS = (LogLink, IdentityLink, LogitLink)
# ``clip_mu`` caps a binomial mean at ``1 - _BINOMIAL_CLIP_MU_EPS``: under the
# log link that cap is reached at ``eta = log1p(-eps)``.
_BINOMIAL_LOG_CAP_ETA = math.log1p(-_BINOMIAL_CLIP_MU_EPS)


def _parameters_match(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 32.0 * np.finfo(np.float64).eps * scale


def _eta_reaches_one(eta_unclipped: NDArray, weights: NDArray) -> bool:
    return bool(np.any((eta_unclipped >= 0.0) & (weights > 0.0)))


@dataclass(frozen=True, slots=True)
class FamilyLinkPair:
    """A family and a link, and what is true of the two together."""

    distribution: Any
    link: Any

    @property
    def _exact(self) -> tuple[type, type]:
        return type(self.distribution), type(self.link)

    # ── Exact facts ──────────────────────────────────────────────

    @property
    def family_is_shipped(self) -> bool:
        return type(self.distribution) in SHIPPED_FAMILIES

    @property
    def link_is_shipped(self) -> bool:
        return type(self.link) in SHIPPED_LINKS

    @property
    def is_shipped(self) -> bool:
        return self.family_is_shipped and self.link_is_shipped

    @property
    def keys_fixed_caches(self) -> bool:
        """Whether the family and link may key a cache that outlives one weight vector."""
        return self.family_is_shipped and type(self.link) in _FIXED_CACHE_LINKS

    @property
    def fisher_equals_observed(self) -> bool:
        """Whether the residual term in Wood's Newton rows is identically zero: a canonical link.

        A power link is canonical at the family's own exponent, and the negative
        binomial's link only at the family's ``theta``.
        """
        family, link = self._exact
        if (family, link) in _CANONICAL:
            return True
        if (family, link) == (NegativeBinomial, NegativeBinomialLink):
            theta = self.distribution.theta
            return theta != "auto" and _parameters_match(float(theta), float(self.link.theta))
        if link is not PowerLink or family not in (Gaussian, Gamma, Tweedie):
            return False
        if family is Tweedie:
            return _parameters_match(float(self.link.power), 1.0 - float(self.distribution.p))
        return _parameters_match(float(self.link.power), 1.0 if family is Gaussian else -1.0)

    @property
    def constant_fisher_weight(self) -> float | None:
        """``(dmu/deta)**2 / V(mu)`` where it is one number for every row, else ``None``.

        A fit can reuse ``X'WX`` only then. Deliberately short, so that reuse
        never changes the fitted problem.
        """
        return _CONSTANT_FISHER_WEIGHT.get(self._exact)

    @property
    def linear_model(self) -> bool:
        """Whether the fit is linear least squares: one solve, with no weights to update."""
        return self._exact == (Gaussian, IdentityLink)

    @property
    def observed_newton(self) -> bool:
        """Whether the pair declares exact observed-Newton rows for PIRLS.

        The declaration is the whole curvature choice: a fit whose pair is
        declared takes observed rows on every iteration its route allows,
        whatever their values.
        """
        return self._exact in _OBSERVED_NEWTON

    @property
    def observed_rows_concave(self) -> bool:
        """Whether the pair's observed working rows exist and are never negative.

        Declared for Gamma/log and Tweedie/log (``observed_newton``), NB2/log
        and binomial probit and cloglog (log-concave CDFs; Pratt 1981, JASA
        76(373)), each with rows formed in ``eta`` that stay non-negative in
        floating point. Probit and cloglog use closed forms in ``eta``, since
        the generic row reads the clipped mean and goes negative past the clip.
        Binomial/log stays on Fisher rows (its maximum can sit on the mean-space
        boundary), as does Poisson/sqrt (not declared); Gaussian/log and
        binomial/cauchit rows go negative.
        """
        return self._exact in _CONCAVE_OBSERVED_ROWS

    @property
    def presolve_eligible(self) -> bool:
        """Whether a cold exact REML fit may start from the cached-W performance iteration.

        On these pairs the cached-W working model carries the exact loop's
        curvature (Fisher on a canonical link, observed rows where
        ``observed_rows_concave`` declares them) and the pre-solve pays.
        """
        return self._exact in _PRESOLVE

    @property
    def leverage_route_always(self) -> bool:
        """Whether exact REML takes the one-pass leverage route for W(rho) at any work.

        Binomial/logit: its derivative Grams cost several leverage passes.
        """
        return self._exact == (Binomial, LogitLink)

    @property
    def deviance_convex_in_eta(self) -> bool:
        """Whether the unit deviance is convex in the predictor, whatever the response.

        A root of the intercept's score is then the profile's minimum. True on a
        canonical link, where the deviance is twice the convex cumulant less a
        linear term, and where the second derivative in the predictor is
        positive by hand: the Poisson's under the identity link (2y/eta**2), the
        gamma's and the Tweedie's (1 < p < 2) under the log link, the negative
        binomial's under the log link (2(y+t)t e**eta/(t+e**eta)**2), and the
        binomial's under the probit, cloglog and log links, whose probabilities
        and complements are log-concave.
        """
        family, link = self._exact
        return link in _CONVEX_DEVIANCE_LINKS.get(family, ())

    @property
    def tail_rows(self) -> bool:
        """Whether ``clip_rows.tail_rows`` has the pair's likelihood rows in ``eta``."""
        return self._exact in _TAIL_ROWS

    @property
    def log_link_variance_power(self) -> float | None:
        """``p`` where the link is the log and the variance is ``mu**p``, else ``None``."""
        family, link = self._exact
        if link is not LogLink:
            return None
        if family is Tweedie:
            return float(self.distribution.p)
        return _LOG_VARIANCE_POWER.get(family)

    @property
    def reaches_zero_mean(self) -> bool:
        """Whether a finite predictor gives a mean of exactly zero that the family accepts.

        Poisson with the square-root link: unlike the log, ``mu = eta**2``
        represents zero and arbitrarily small means directly, so the fit takes
        no positive floor and its objective has the response's own scale.
        """
        return self._exact == (Poisson, SqrtLink)

    # ── Inherited facts ──────────────────────────────────────────

    @property
    def mean_space_violation(self) -> Callable[[NDArray, NDArray], bool] | None:
        """The test that a linear predictor leaves the family's mean space, or ``None``.

        ``None`` when every ``eta`` the link accepts maps into the family's
        means. The log link's inverse ``exp(eta)`` covers ``(0, inf)``, which is
        wider than the binomial probability space ``(0, 1)``: a positive-weight
        row with ``eta >= 0`` has no binomial likelihood (``log(1 - mu)`` is
        undefined), so it is not a state at all. ``clip_mu`` would hide it
        behind a mean of ``1 - 1e-7``, whose deviance is finite and flat in
        ``eta`` and whose Fisher weight ``exp(2 eta) / V(mu)`` pins the row
        where it landed: replacing an out-of-range fitted value instead of the
        coefficients is Wacholder's (1986) device, which can return estimates
        whose fitted probabilities are invalid. The feasible set ``{beta : X
        beta + offset < 0}`` is convex and the log-likelihood concave, so a line
        search from a feasible state finds a feasible step by halving, as R's
        ``glm`` and ``glm2`` do (Donoghoe & Marschner 2018, J. Stat. Softw.
        86(9), sections 2 and 3.2).
        """
        if isinstance(self.distribution, Binomial) and isinstance(self.link, LogLink):
            return _eta_reaches_one
        return None

    @property
    def mean_space_boundary(self) -> Callable[[NDArray], NDArray] | None:
        """The rows whose mean ``clip_mu`` caps at the mean-space boundary, or ``None``.

        A binomial/log row with ``exp(eta) >= 1 - 1e-7`` has its mean replaced
        by the cap, so its deviance is flat in ``eta`` there: the fitted
        objective is no longer the binomial likelihood, and a state holding such
        a row is at the boundary of the parameter space ``{beta : X beta +
        offset < 0}``, where the maximum is not a stationary point (Donoghoe &
        Marschner 2018, J. Stat. Softw. 86(9), sections 2 and 4.1). A
        binomial/identity row whose mean ``link.inverse(eta)`` ``clip_mu``
        replaces, at either end, is at the boundary of ``{beta : 0 < X beta +
        offset < 1}`` in the same way (Donoghoe & Marschner 2014, Comput. Stat.
        Data Anal. 80).
        """
        if self.mean_space_violation is not None:
            return lambda eta: eta >= _BINOMIAL_LOG_CAP_ETA
        if isinstance(self.distribution, Binomial) and isinstance(self.link, IdentityLink):
            link = self.link

            def at_either_end(eta: NDArray) -> NDArray:
                mean = link.inverse(np.array(eta, dtype=float))
                return (mean <= _BINOMIAL_CLIP_MU_EPS) | (mean >= 1.0 - _BINOMIAL_CLIP_MU_EPS)

            return at_either_end
        return None

    @property
    def clip_can_hold_against(self) -> bool:
        """Whether the working clip can hold a row of the pair against its outcome.

        A binomial fit under any link but the log and identity ones, whose
        clipped rows are the boundary of their mean space and have their own
        verdict (``mean_space_boundary``).
        """
        return isinstance(self.distribution, Binomial) and not isinstance(
            self.link, LogLink | IdentityLink
        )

    @property
    def link_domain_rows(self) -> Callable[[NDArray], NDArray] | None:
        """The rows whose linear predictor leaves the link inverse's domain, or ``None``.

        The domain the link declares (``links.outside_eta_domain``). Under the
        identity link, a family whose means are positive is held to a positive
        mean ``link.inverse(eta) > 0`` as well, and below the family's upper
        bound, through the link's own inverse, so a subclass that shifts it
        keeps its own domain. A mean ``mu <= 0`` is invalid (R's
        ``poisson()$validmu``), and ``glm.fit`` halves a step that produces one.
        The floor would take a crossed row as ``mu = 1e-50``, whose Poisson
        Fisher weight ``1 / mu`` is ``1e50``. The inverse is evaluated on a copy
        of ``eta``: a state's ``eta`` is read-only, and a subclass's inverse may
        write in place.
        """
        declared = outside_eta_domain(self.link)
        if declared is not None:
            return declared
        if self.distribution is None or not isinstance(self.link, IdentityLink):
            return None
        domain = mean_domain(self.distribution)
        if not domain.positive:
            return None
        link, upper = self.link, domain.upper

        def mean_outside(eta_unclipped: NDArray) -> NDArray:
            mean = link.inverse(np.array(eta_unclipped, dtype=float))
            return (mean <= 0.0) | (mean >= upper)

        return mean_outside

    @property
    def link_domain_violation(self) -> Callable[[NDArray, NDArray], bool] | None:
        """The test that a positive-weight row is one of ``link_domain_rows``, or ``None``."""
        rows = self.link_domain_rows
        if rows is None:
            return None
        return lambda eta_unclipped, weights: bool(np.any(rows(eta_unclipped) & (weights > 0.0)))


def pair_of(distribution: Any, link: Any) -> FamilyLinkPair:
    """The pair of ``distribution`` and ``link``. Never raises, whatever the two are."""
    return FamilyLinkPair(distribution, link)
