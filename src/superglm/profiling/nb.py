"""Negative Binomial theta estimation via alternating GLM + safeguarded solve.

Alternates between fitting the GLM at the current theta (PIRLS) and updating
theta on the closed-form NB2 profile score (digamma/trigamma). The alternating
scheme is the standard one described by Venables & Ripley (2002, ch. 7.4);
the profile score and information are classical NB2 maximum-likelihood theory
(Lawless 1987). Converges in 3-5 outer iterations instead of the ~14
black-box evaluations required by Brent profiling.

For NB2: V(mu) = mu + mu^2/theta. The key insight is that given fitted mu,
the profile likelihood for theta has a closed-form score and information,
so theta can be updated analytically without refitting the GLM.

The inner theta update is a bracketed scalar root find (Brent) on the profile
score, started from a method-of-moments estimate. An unsafeguarded Newton
iteration on this score can ascend the negative log-likelihood from a poor
start (the profile information is not globally positive), and a silent clip
into narrow bounds then publishes the wrong end of the parameter space with
``converged=True``; the bracketing solve removes both failure modes and any
remaining active bound is reported honestly.

References
----------
- Venables & Ripley (2002): Modern Applied Statistics with S, Ch 7.4
  (alternating GLM fit / profile-score update for the NB shape).
- Lawless (1987): Negative binomial and mixed Poisson regression,
  Canadian Journal of Statistics 15(3), 209-225 (NB2 profile score,
  information, and moment estimation).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import digamma, gammaln

from superglm._frame import as_eager_frame
from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.model.fit_state import (
    FrozenMapping,
    configured_family,
    configured_lambda2,
    configured_penalty,
)
from superglm.penalties.base import penalty_has_targets
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import fit_pirls

#: Default search range for the NB2 shape parameter. Deliberately wide: these
#: are numerical guard rails for the bracketed solve, not a statistical prior.
#: The historical default of (0.1, 50.0) excluded routinely occurring true
#: values at both ends (heavy overdispersion sits below 0.1; near-Poisson data
#: pushes the profile optimum far above 50) and, combined with an
#: unsafeguarded Newton step, published the wrong clamp end silently.
_THETA_DEFAULT_BOUNDS: tuple[float, float] = (1e-8, 1e8)

#: Geometric step used to bracket a sign change of the profile score.
_THETA_BRACKET_FACTOR = 10.0


def _theta_cache_key(value: float) -> float:
    """Cache key for a theta iterate: six SIGNIFICANT digits.

    Decimal rounding (``round(value, 6)``) collapses every theta below 5e-7
    onto the impossible key 0.0 - which ``profile_plot`` then feeds to the
    NB2 likelihood as a zero shape parameter - and merges distinct small-
    theta iterates onto one entry. Significant-digit rounding matches how
    ``theta_hat`` itself is published, so the publication lookup and the
    iteration entries share one convention at every scale.
    """
    return float(f"{float(value):.6g}")


class NBThetaBoundWarning(UserWarning):
    """The NB2 theta estimate sits on an active search bound.

    The reported ``theta_hat`` is the constrained boundary value, not an
    interior maximum-likelihood estimate, and the accompanying result carries
    ``converged=False``. An active upper bound usually means the data are no
    more dispersed than Poisson at the fitted mean (the profile likelihood
    increases toward the Poisson limit ``theta -> inf``); an active lower
    bound means overdispersion beyond the searchable range.
    """


@dataclass
class NBProfileResult:
    """Result of NB theta parameter estimation."""

    theta_hat: float
    nll: float
    n_evaluations: int
    converged: bool
    cache: Mapping[float, float] = field(default_factory=dict)

    # Set after estimation to enable .ci()
    _y: NDArray | None = field(default=None, repr=False)
    _mu: NDArray | None = field(default=None, repr=False)
    _weights: NDArray | None = field(default=None, repr=False)

    _ci_cache: dict[float, tuple[float, float]] = field(default_factory=dict, repr=False)
    _publication_locked: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: object) -> None:
        if self.__dict__.get("_publication_locked", False):
            raise AttributeError(f"published NBProfileResult is immutable; cannot rebind {name!r}")
        object.__setattr__(self, name, value)

    def _published_with_data(
        self,
        y: NDArray,
        mu: NDArray,
        weights: NDArray,
    ) -> NBProfileResult:
        """Return an owned result synchronized to one final fitted mean vector."""
        y_owned = _immutable_array_copy(np.asarray(y, dtype=np.float64))
        mu_owned = _immutable_array_copy(np.asarray(mu, dtype=np.float64))
        weights_owned = _immutable_array_copy(np.asarray(weights, dtype=np.float64))
        nll = _nb2_nll(y_owned, mu_owned, weights_owned, float(self.theta_hat))
        cache = dict(self.cache)
        cache[_theta_cache_key(self.theta_hat)] = nll
        published = type(self)(
            theta_hat=float(self.theta_hat),
            nll=nll,
            n_evaluations=int(self.n_evaluations),
            converged=bool(self.converged),
            cache=FrozenMapping(cache),
            _y=y_owned,
            _mu=mu_owned,
            _weights=weights_owned,
        )
        object.__setattr__(published, "_publication_locked", True)
        return published

    def _detached_public_copy(self) -> NBProfileResult:
        """Return a distinct immutable handle without duplicating immutable row buffers."""
        cache = self.cache if isinstance(self.cache, FrozenMapping) else FrozenMapping(self.cache)
        detached = type(self)(
            theta_hat=float(self.theta_hat),
            nll=float(self.nll),
            n_evaluations=int(self.n_evaluations),
            converged=bool(self.converged),
            cache=cache,
            _y=self._y,
            _mu=self._mu,
            _weights=self._weights,
            _ci_cache=dict(self._ci_cache),
        )
        object.__setattr__(detached, "_publication_locked", True)
        return detached

    def __deepcopy__(self, memo: dict[int, object]) -> NBProfileResult:
        existing = memo.get(id(self))
        if existing is not None:
            return existing  # type: ignore[return-value]
        if self.__dict__.get("_publication_locked", False):
            result = self._detached_public_copy()
        else:
            result = type(self)(
                theta_hat=float(self.theta_hat),
                nll=float(self.nll),
                n_evaluations=int(self.n_evaluations),
                converged=bool(self.converged),
                cache=copy.deepcopy(self.cache, memo),
                _y=copy.deepcopy(self._y, memo),
                _mu=copy.deepcopy(self._mu, memo),
                _weights=copy.deepcopy(self._weights, memo),
                _ci_cache=copy.deepcopy(self._ci_cache, memo),
            )
        memo[id(self)] = result
        return result

    def __getstate__(self) -> dict[str, object]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, object]) -> None:
        published = bool(state.get("_publication_locked", False))
        object.__setattr__(self, "_publication_locked", False)
        for name, value in state.items():
            if name != "_publication_locked":
                object.__setattr__(self, name, value)
        if published:
            for name in ("_y", "_mu", "_weights"):
                value = getattr(self, name)
                if value is not None:
                    object.__setattr__(self, name, _immutable_array_copy(value))
            object.__setattr__(self, "cache", FrozenMapping(self.cache))
            object.__setattr__(self, "_publication_locked", True)

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Profile likelihood confidence interval for theta.

        Requires that the result was produced by ``estimate_nb_theta``.
        Results are cached so repeated calls (e.g. from summary()) are free.
        """
        if alpha in self._ci_cache:
            return self._ci_cache[alpha]
        if self._y is None or self._mu is None or self._weights is None:
            raise RuntimeError(
                "Profile CI requires fitted mu. Use estimate_nb_theta() to produce this result."
            )
        result = profile_ci_theta(self._y, self._mu, self._weights, self.theta_hat, alpha=alpha)
        self._ci_cache[alpha] = result
        return result

    def profile_plot(
        self,
        *,
        alpha: float = 0.05,
        n_points: int = 100,
        ax=None,
    ):
        """Profile deviance plot for NB2 theta.

        Shows the profile deviance curve with the MLE, confidence interval
        bounds, and chi-squared cutoff. Cheap — each evaluation is O(n)
        with no refitting.

        Parameters
        ----------
        alpha : float
            Significance level for CI (default 0.05).
        n_points : int
            Number of grid points for the curve.
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._y is None or self._mu is None or self._weights is None:
            raise RuntimeError(
                "Profile plot requires fitted mu. Use estimate_nb_theta() to produce this result."
            )

        import matplotlib.pyplot as plt
        from scipy.stats import chi2

        ci_lo, ci_hi = self.ci(alpha=alpha)

        # Grid extends beyond CI for visual context. The positive floor must
        # scale with the estimate: a fixed 0.01 made every theta in the newly
        # admitted (1e-8, 0.01) band unplottable.
        margin = 0.3 * (ci_hi - ci_lo)
        grid_lo = max(min(0.01, 0.1 * self.theta_hat), ci_lo - margin)
        grid_hi = ci_hi + margin
        theta_grid = np.linspace(grid_lo, grid_hi, n_points)

        w_sum = float(np.sum(self._weights))
        nll_hat = _nb2_nll(self._y, self._mu, self._weights, self.theta_hat)
        deviance = np.array(
            [
                2.0 * w_sum * (_nb2_nll(self._y, self._mu, self._weights, t) - nll_hat)
                for t in theta_grid
            ]
        )

        cutoff = chi2.ppf(1.0 - alpha, 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()

        ax.plot(theta_grid, deviance, color="steelblue", linewidth=1.5)

        # Mark cached iteration points (re-evaluated on the fixed-mu profile)
        if self.cache:
            cache_thetas = np.array(sorted(self.cache.keys()))
            cache_dev = np.array(
                [
                    2.0 * w_sum * (_nb2_nll(self._y, self._mu, self._weights, t) - nll_hat)
                    for t in cache_thetas
                ]
            )
            ax.scatter(
                cache_thetas,
                cache_dev,
                color="darkorange",
                s=35,
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
                label=f"Iterations ({len(cache_thetas)})",
            )

        ax.axhline(
            cutoff,
            linestyle="--",
            color="grey",
            linewidth=0.8,
            label=f"{100 * (1 - alpha):.0f}% cutoff",
        )
        ax.axvline(
            self.theta_hat,
            linestyle=":",
            color="black",
            linewidth=0.8,
            label=f"MLE = {self.theta_hat:.3f}",
        )
        ax.fill_betweenx(
            [0, cutoff],
            ci_lo,
            ci_hi,
            alpha=0.10,
            color="firebrick",
            label=f"{100 * (1 - alpha):.0f}% CI: [{ci_lo:.3f}, {ci_hi:.3f}]",
        )

        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("Profile deviance")
        ax.set_title(r"NB2 $\theta$ profile likelihood")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        return fig


@dataclass(frozen=True)
class _ThetaSolve:
    """One safeguarded solve of the fixed-mu NB2 profile score."""

    theta: float
    converged: bool
    at_lower: bool
    at_upper: bool
    n_score_evaluations: int

    @property
    def at_bound(self) -> bool:
        return self.at_lower or self.at_upper


#: Above this theta the profile score switches to its large-theta expansion.
#: The naive form obtains an O(theta^-2) score by cancelling digamma,
#: logarithm, and ratio terms whose leading parts are O(theta^-1) computed
#: from O(log theta)-sized intermediates, so float64 loses the sign to
#: roundoff around theta ~ 1e7-1e8 -- exactly the near-Poisson regime the
#: widened bounds admit. At 1e5 the expansion's dropped psi tail is
#: O(theta^-5) while the score itself is O(theta^-2): eleven orders of
#: headroom, and the naive form is still accurate there, so the two branches
#: agree to ~1e-9 relative across the switch.
_THETA_SCORE_ASYMPTOTIC_MIN = 1e5


def _theta_profile_score(y: NDArray, mu: NDArray, weights: NDArray, theta: float) -> float:
    """Closed-form NB2 profile score dl/dtheta at fixed mu (Lawless 1987).

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
    if theta >= _THETA_SCORE_ASYMPTOTIC_MIN:
        shifted = theta + y
        x = (y - mu) / (theta + mu)
        core = np.log1p(x) - x
        psi_tail = 0.5 * y / (theta * shifted) + (1.0 / theta**2 - 1.0 / shifted**2) / 12.0
        return float(np.sum(weights * (core + psi_tail)))
    return float(
        np.sum(
            weights
            * (
                digamma(y + theta)
                - digamma(theta)
                + np.log(theta)
                + 1.0
                - np.log(theta + mu)
                - (y + theta) / (mu + theta)
            )
        )
    )


def _theta_moment_start(y: NDArray, mu: NDArray, weights: NDArray) -> float | None:
    """Method-of-moments start for theta from V(mu) = mu + mu^2/theta.

    Solves ``sum(w * ((y - mu)^2 - mu)) = sum(w * mu^2) / theta`` for theta.
    Returns None when the weighted excess dispersion is non-positive (the data
    are at most Poisson-dispersed at this mu), in which case the profile
    optimum lies at or beyond the upper search bound.
    """
    numerator = float(np.sum(weights * mu * mu))
    denominator = float(np.sum(weights * ((y - mu) ** 2 - mu)))
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or numerator <= 0.0
        or denominator <= 0.0
    ):
        return None
    return numerator / denominator


def _theta_ml(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    theta: float,
    *,
    bounds: tuple[float, float] = _THETA_DEFAULT_BOUNDS,
    max_iter: int = 100,
    eps: float = 1e-8,
) -> _ThetaSolve:
    """Safeguarded maximization of the NB2 profile log-likelihood over theta.

    Maximises the fixed-mu profile log-likelihood by bracketing a sign change
    of the closed-form profile score (geometric expansion from the start
    value) and solving it with Brent's method. Every accepted iterate is
    therefore on the correct side of the likelihood: the solve cannot ascend
    the negative log-likelihood the way an unguarded Newton step can when the
    profile information turns negative away from the optimum.

    If the score does not change sign inside ``bounds`` the profile optimum
    lies at or beyond the corresponding bound; the bound value is returned
    with ``converged=False`` and the matching ``at_lower``/``at_upper`` flag
    set, so callers can report the active constraint instead of publishing it
    as a converged interior estimate.

    Each score evaluation is O(n) with no matrix operations.
    """
    lower = float(bounds[0])
    upper = float(bounds[1])
    if not (0.0 < lower < upper) or not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"theta bounds must satisfy 0 < lower < upper < inf, got {bounds!r}")
    theta0 = float(min(max(float(theta), lower), upper))

    evaluations = 0

    def score(value: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return _theta_profile_score(y, mu, weights, value)

    s0 = score(theta0)
    if not np.isfinite(s0):
        raise FloatingPointError("NB2 profile score is not finite at the starting theta")
    if s0 == 0.0:
        return _ThetaSolve(theta0, True, theta0 <= lower, theta0 >= upper, evaluations)

    if s0 > 0.0:
        # Likelihood increasing: the optimum lies to the right of theta0.
        bracket_lo = theta0
        current = theta0
        while True:
            if current >= upper:
                return _ThetaSolve(upper, False, False, True, evaluations)
            current = min(current * _THETA_BRACKET_FACTOR, upper)
            s_current = score(current)
            if not np.isfinite(s_current):
                raise FloatingPointError("NB2 profile score overflowed while bracketing theta")
            if s_current <= 0.0:
                bracket_hi = current
                break
            bracket_lo = current
    else:
        # Likelihood decreasing: the optimum lies to the left of theta0.
        bracket_hi = theta0
        current = theta0
        while True:
            if current <= lower:
                return _ThetaSolve(lower, False, True, False, evaluations)
            current = max(current / _THETA_BRACKET_FACTOR, lower)
            s_current = score(current)
            if not np.isfinite(s_current):
                raise FloatingPointError("NB2 profile score overflowed while bracketing theta")
            if s_current >= 0.0:
                bracket_lo = current
                break
            bracket_hi = current

    root = float(
        brentq(
            score,
            bracket_lo,
            bracket_hi,
            xtol=np.finfo(np.float64).tiny,
            rtol=max(float(eps), 4.0 * np.finfo(np.float64).eps),
            maxiter=int(max_iter),
        )
    )
    return _ThetaSolve(root, True, False, False, evaluations)


def _immutable_array_copy(value: NDArray) -> NDArray:
    """Copy onto a bytes-backed buffer whose write flag cannot be restored."""
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _nb2_nll(y: NDArray, mu: NDArray, weights: NDArray, theta: float) -> float:
    """Weighted mean negative NB2 log-likelihood."""
    ll = (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1)
        + theta * np.log(theta / (mu + theta))
        + y * np.log(mu / (mu + theta))
    )
    return -float(np.sum(weights * ll)) / float(np.sum(weights))


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
    verbose: bool = False,
    trace_callback=None,
) -> NBProfileResult:
    """Estimate NB2 theta via alternating GLM fit + safeguarded profile solve.

    Algorithm (Venables & Ripley 2002, ch. 7.4):
      1. Build design matrix once, calibrate lambda.
      2. Alternate: fit GLM at current theta (PIRLS with warm starts)
         → update theta by a bracketed root find on the closed-form profile
         score (Lawless 1987), started from a method-of-moments estimate on
         the first pass and warm-started thereafter.
      3. Converge when |theta_new - theta_old| < xatol (~3-5 iterations).

    Parameters
    ----------
    model : SuperGLM
        A configured but *unfitted* model with features already added.
        Must have a NegativeBinomial family (e.g. ``families.nb2(theta=1.0)``).
    X : pandas or eager Polars DataFrame
        Feature matrix.
    y : array-like
        Response variable (counts).
    sample_weight : array-like, optional
        Frequency weights (sample_weight). Must be frequency weights, not
        variance weights — theta estimation assumes each observation's
        log-likelihood contribution is scaled by its sample_weight.
    offset : array-like, optional
        Offset added to the linear predictor.
    theta_bounds : tuple
        Search range for theta, default ``(1e-8, 1e8)``. These are numerical
        guard rails, not a statistical prior. If the estimate lands on a
        bound the result reports ``converged=False`` and an
        ``NBThetaBoundWarning`` is emitted — a bounded value is a constrained
        boundary report, never a silent interior estimate.
    xatol : float
        Convergence tolerance on theta for the outer alternation, applied
        RELATIVE to the current estimate (with the lower search bound as the
        scale floor): the alternation stops when
        ``|theta_new - theta| <= xatol * max(|theta_new|, lower_bound)``.
        The historical absolute reading was only sound over the old
        (0.1, 50.0) range; across (1e-8, 1e8) an absolute 1e-2 would accept
        order-of-magnitude jumps below 0.01 and demand needless precision
        near the top.
    maxiter : int
        Maximum outer iterations (GLM fits).
    verbose : bool
        Print progress.

    Returns
    -------
    NBProfileResult
    """
    from superglm.distributions import NegativeBinomial

    X = as_eager_frame(X)

    # Validate family
    family = configured_family(model)
    if not isinstance(family, NegativeBinomial):
        raise ValueError(
            f"estimate_nb_theta requires a NegativeBinomial family, got {family!r}. "
            "Use families.nb2(theta=...) to create one."
        )

    y = np.asarray(y, dtype=np.float64)

    # --- One-time setup: build design matrix and calibrate lambda ---
    if model._splines is not None and not model._specs:
        model._auto_detect_features(X, sample_weight)

    # Temporary theta for _build_design_matrix (DM doesn't depend on theta)
    from superglm.distributions import NegativeBinomial

    saved_family = configured_family(model)
    model.family = NegativeBinomial(theta=1.0)
    try:
        y_arr, w_arr, offset_arr = model._build_design_matrix(X, y, sample_weight, offset)
    finally:
        model.family = saved_family

    penalty = configured_penalty(model)
    from superglm.model.base import resolve_selection_penalty_for_fit

    resolve_selection_penalty_for_fit(model, penalty, y_arr, w_arr)

    if offset_arr is None:
        offset_arr = np.zeros(len(y_arr))

    dm = model._dm
    groups = model._groups
    link = model._link

    # Use direct solver when lambda1=0 (no L1 penalty → no BCD needed)
    _use_direct = penalty.lambda1 is not None and (
        penalty.lambda1 == 0 or not penalty_has_targets(penalty, groups)
    )
    reml_penalties = None
    if _use_direct:
        from superglm.model.reml_setup import collect_reml_groups
        from superglm.reml.penalty_algebra import build_penalty_context

        reml_groups = collect_reml_groups(groups, dm.group_matrices)
        if reml_groups:
            reml_penalties, _penalty_caches, _penalty_ranks = build_penalty_context(
                dm.group_matrices,
                reml_groups,
            )

    # --- Alternating estimation ---
    # theta = 1.0 only seeds the first working GLM fit (the fitted mean is
    # weakly theta-sensitive); the first profile solve restarts from a
    # method-of-moments estimate at that mean, so the search begins where the
    # data point instead of at an arbitrary fixed value.
    theta = 1.0
    warm_beta = None
    warm_intercept = None
    cache: dict[float, float] = {}
    converged = False
    theta_solve: _ThetaSolve | None = None

    for iteration in range(maxiter):
        # Step 1: Fit GLM at current theta (warm-started after first iter)
        dist = NegativeBinomial(theta)
        if _use_direct:
            pirls_result, _ = fit_irls_direct(
                X=dm,
                y=y_arr,
                weights=w_arr,
                family=dist,
                link=link,
                groups=groups,
                lambda2=configured_lambda2(model),
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                direct_solve=getattr(model, "_direct_solve", "auto"),
                reml_penalties=reml_penalties,
            )
        else:
            pirls_result = fit_pirls(
                X=dm,
                y=y_arr,
                weights=w_arr,
                family=dist,
                link=link,
                groups=groups,
                penalty=penalty,
                offset=offset_arr,
                lambda2=configured_lambda2(model),
                beta_init=warm_beta,
                intercept_init=warm_intercept,
            )

        eta = stabilize_eta(
            dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr, link
        )
        mu = clip_mu(link.inverse(eta), dist)
        warm_beta = pirls_result.beta
        warm_intercept = pirls_result.intercept

        # Step 2: safeguarded profile solve for theta given mu
        if iteration == 0:
            moment_start = _theta_moment_start(y_arr, mu, w_arr)
            # A non-positive moment denominator means the data are at most
            # Poisson-dispersed at this mean: the profile optimum sits at or
            # beyond the upper bound, so start the solve there and let the
            # score decide.
            theta_start = moment_start if moment_start is not None else theta_bounds[1]
        else:
            theta_start = theta
        theta_solve = _theta_ml(y_arr, mu, w_arr, theta_start, bounds=theta_bounds)
        theta_new = theta_solve.theta

        nll = _nb2_nll(y_arr, mu, w_arr, theta_new)
        cache[_theta_cache_key(theta_new)] = nll
        if trace_callback is not None:
            trace_callback(
                {
                    "step": iteration,
                    "theta": theta_new,
                    "nll": nll,
                    "n_iter": pirls_result.n_iter,
                    "fit_converged": pirls_result.converged,
                    "source": "newton",
                }
            )

        if verbose:
            print(
                f"  iter={iteration + 1}  theta={theta_new:.4f}  "
                f"nll={nll:.4f}  pirls_iters={pirls_result.n_iter}"
            )

        convergence_scale = max(abs(theta_new), float(theta_bounds[0]))
        if abs(theta_new - theta) <= xatol * convergence_scale:
            theta = theta_new
            converged = True
            break
        theta = theta_new

    # Round to six significant digits (not six decimals: a decimal round would
    # collapse a small-theta estimate toward zero and poison the likelihood).
    theta_hat = float(f"{theta:.6g}")
    at_bound = theta_solve is not None and theta_solve.at_bound
    if at_bound:
        assert theta_solve is not None
        side = "lower" if theta_solve.at_lower else "upper"
        bound_value = theta_bounds[0] if theta_solve.at_lower else theta_bounds[1]
        interpretation = (
            "the data show overdispersion beyond the searchable range"
            if theta_solve.at_lower
            else "the profile likelihood increases toward the Poisson limit"
            " (the data are at most Poisson-dispersed at the fitted mean)"
        )
        warnings.warn(
            f"NB2 theta estimate hit the {side} search bound {bound_value:g}: "
            f"{interpretation}. theta_hat={theta_hat:g} is a constrained "
            "boundary value, not an interior optimum, and the result reports "
            "converged=False. Widen theta_bounds to search further, or "
            "reconsider the family.",
            NBThetaBoundWarning,
            stacklevel=2,
        )
    nll_final = cache.get(theta_hat, _nb2_nll(y_arr, mu, w_arr, theta_hat))

    result = NBProfileResult(
        theta_hat=theta_hat,
        nll=nll_final,
        n_evaluations=iteration + 1,
        converged=converged and not at_bound,
        cache=cache,
        _y=y_arr,
        _mu=mu,
        _weights=w_arr,
    )
    return result._published_with_data(y_arr, mu, w_arr)


def profile_ci_theta(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    theta_hat: float,
    *,
    alpha: float = 0.05,
    theta_range: tuple[float, float] = (0.01, 500.0),
) -> tuple[float, float]:
    """Profile likelihood confidence interval for NB2 theta.

    Given fitted mu (held fixed), evaluates the NB2 profile log-likelihood
    at different theta values and inverts the LRT at the chi-squared cutoff.
    This is O(n) per evaluation with no matrix operations or refitting.

    Parameters
    ----------
    y : array
        Response (counts).
    mu : array
        Fitted means from the GLM.
    weights : array
        Frequency weights.
    theta_hat : float
        MLE of theta.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    theta_range : tuple
        Search range for the CI endpoints.

    Returns
    -------
    (ci_lower, ci_upper) : tuple of float
    """
    from scipy.stats import chi2

    w_sum = float(np.sum(weights))
    nll_hat = _nb2_nll(y, mu, weights, theta_hat)
    cutoff = chi2.ppf(1.0 - alpha, 1)

    def objective(theta: float) -> float:
        return 2.0 * w_sum * (_nb2_nll(y, mu, weights, theta) - nll_hat) - cutoff

    # The search range must contain theta_hat, which the widened default
    # theta estimation bounds no longer guarantee for the fixed default range.
    lo = min(theta_range[0], theta_hat / 100.0)
    hi = max(theta_range[1], theta_hat * 100.0)

    # Root tolerances must scale with the endpoint, not sit at a fixed
    # absolute 1e-4: below theta_hat ~ 1e-4 that constant exceeded the
    # entire lower bracket, so brentq returned an arbitrary in-bracket point
    # instead of the LRT crossing. A relative tolerance pins each endpoint
    # to six significant digits at every scale (an endpoint's own magnitude
    # is the only correct yardstick - even a theta_hat-proportional absolute
    # tolerance mis-scales when the lower endpoint sits far below
    # theta_hat); the near-zero xtol merely satisfies brentq's positivity
    # requirement.
    try:
        ci_lower = brentq(objective, lo, theta_hat, xtol=1e-300, rtol=1e-6)
    except ValueError:
        ci_lower = lo

    try:
        ci_upper = brentq(objective, theta_hat, hi, xtol=1e-300, rtol=1e-6)
    except ValueError:
        ci_upper = hi

    return (ci_lower, ci_upper)
