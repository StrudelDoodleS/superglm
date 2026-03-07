"""Negative Binomial theta estimation via alternating GLM + Newton.

Implements the MASS::glm.nb algorithm: alternate between fitting the GLM
at the current theta (PIRLS) and updating theta via Newton on the closed-form
profile score (digamma/trigamma). Converges in 3-5 outer iterations instead
of the ~14 black-box evaluations required by Brent profiling.

For NB2: V(mu) = mu + mu^2/theta. The key insight is that given fitted mu,
the profile likelihood for theta has a closed-form score and information,
so theta can be updated analytically without refitting the GLM.

References
----------
- Venables & Ripley (2002): Modern Applied Statistics with S, Ch 7.4
- MASS::glm.nb and MASS::theta.ml source code
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, gammaln, polygamma

from superglm.solvers.pirls import fit_pirls


@dataclass
class NBProfileResult:
    """Result of NB theta parameter estimation."""

    theta_hat: float
    nll: float
    n_evaluations: int
    converged: bool
    cache: dict[float, float] = field(default_factory=dict)

    # Set after estimation to enable .ci()
    _y: NDArray | None = field(default=None, repr=False)
    _mu: NDArray | None = field(default=None, repr=False)
    _weights: NDArray | None = field(default=None, repr=False)

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Profile likelihood confidence interval for theta.

        Requires that the result was produced by ``estimate_nb_theta``.
        """
        if self._y is None or self._mu is None or self._weights is None:
            raise RuntimeError(
                "Profile CI requires fitted mu. Use estimate_nb_theta() to produce this result."
            )
        return profile_ci_theta(self._y, self._mu, self._weights, self.theta_hat, alpha=alpha)

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

        # Grid extends beyond CI for visual context
        margin = 0.3 * (ci_hi - ci_lo)
        grid_lo = max(0.01, ci_lo - margin)
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


def _theta_ml(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    theta: float,
    *,
    bounds: tuple[float, float] = (0.1, 50.0),
    max_iter: int = 10,
    eps: float = 1e-6,
) -> float:
    """Newton iteration for NB2 theta given fitted mu.

    Equivalent to MASS::theta.ml. Maximises the NB2 profile log-likelihood
    over theta with mu held fixed. Typically converges in 2-4 iterations.

    The score and information have closed forms in terms of digamma/trigamma,
    so each Newton step is O(n) with no matrix operations.
    """
    for _ in range(max_iter):
        # Score: dℓ/dθ
        score = np.sum(
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
        # Information: -d²ℓ/dθ²
        info = np.sum(
            weights
            * (
                -polygamma(1, y + theta)
                + polygamma(1, theta)
                - 1.0 / theta
                + 2.0 / (mu + theta)
                - (y + theta) / (mu + theta) ** 2
            )
        )
        if abs(info) < 1e-20:
            break
        delta = score / info
        theta_new = np.clip(theta + delta, bounds[0], bounds[1])
        if abs(theta_new - theta) / (theta + 1e-10) < eps:
            theta = float(theta_new)
            break
        theta = float(theta_new)
    return theta


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
    exposure=None,
    offset=None,
    *,
    theta_bounds: tuple[float, float] = (0.1, 50.0),
    xatol: float = 1e-2,
    maxiter: int = 30,
    verbose: bool = False,
) -> NBProfileResult:
    """Estimate NB2 theta via alternating GLM fit + Newton (MASS::glm.nb).

    Algorithm:
      1. Build design matrix once, calibrate lambda.
      2. Alternate: fit GLM at current theta (PIRLS with warm starts)
         → update theta via Newton on the profile score (MASS::theta.ml).
      3. Converge when |theta_new - theta_old| < xatol (~3-5 iterations).

    Parameters
    ----------
    model : SuperGLM
        A configured but *unfitted* model with features already added.
        Must have family="negative_binomial" or a NegativeBinomial instance.
    X : DataFrame
        Feature matrix.
    y : array-like
        Response variable (counts).
    exposure : array-like, optional
        Frequency weights (exposure). Must be frequency weights, not
        variance weights — theta estimation assumes each observation's
        log-likelihood contribution is scaled by its exposure.
    offset : array-like, optional
        Offset added to the linear predictor.
    theta_bounds : tuple
        Bounds for theta, default (0.1, 50.0).
    xatol : float
        Convergence tolerance on theta (absolute).
    maxiter : int
        Maximum outer iterations (GLM fits).
    verbose : bool
        Print progress.

    Returns
    -------
    NBProfileResult
    """
    from superglm.distributions import NegativeBinomial

    # Validate family
    family = model.family
    is_nb = (isinstance(family, str) and family == "negative_binomial") or isinstance(
        family, NegativeBinomial
    )
    if not is_nb:
        raise ValueError(f"estimate_nb_theta requires family='negative_binomial', got {family!r}")

    y = np.asarray(y, dtype=np.float64)

    # --- One-time setup: build design matrix and calibrate lambda ---
    if model._splines is not None and not model._specs:
        model._auto_detect_features(X, exposure)

    # Temporary theta for _build_design_matrix (DM doesn't depend on theta)
    saved_theta = model.nb_theta
    model.nb_theta = 1.0
    y_arr, w_arr, offset_arr = model._build_design_matrix(X, y, exposure, offset)
    model.nb_theta = saved_theta

    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = model._compute_lambda_max(y_arr, w_arr) * 0.1

    if offset_arr is None:
        offset_arr = np.zeros(len(y_arr))

    dm = model._dm
    groups = model._groups
    link = model._link
    penalty = model.penalty

    # --- Alternating estimation ---
    theta = 1.0  # initial value (MASS also starts simple)
    warm_beta = None
    warm_intercept = None
    cache: dict[float, float] = {}
    converged = False

    for iteration in range(maxiter):
        # Step 1: Fit GLM at current theta (warm-started after first iter)
        dist = NegativeBinomial(theta)
        pirls_result = fit_pirls(
            X=dm,
            y=y_arr,
            weights=w_arr,
            family=dist,
            link=link,
            groups=groups,
            penalty=penalty,
            offset=offset_arr,
            beta_init=warm_beta,
            intercept_init=warm_intercept,
        )

        eta = np.clip(
            dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
            -20,
            20,
        )
        mu = np.maximum(link.inverse(eta), 1e-10)
        warm_beta = pirls_result.beta
        warm_intercept = pirls_result.intercept

        # Step 2: Newton update for theta given mu
        theta_new = _theta_ml(y_arr, mu, w_arr, theta, bounds=theta_bounds)

        nll = _nb2_nll(y_arr, mu, w_arr, theta_new)
        cache[round(theta_new, 6)] = nll

        if verbose:
            print(
                f"  iter={iteration + 1}  theta={theta_new:.4f}  "
                f"nll={nll:.4f}  pirls_iters={pirls_result.n_iter}"
            )

        if abs(theta_new - theta) < xatol:
            theta = theta_new
            converged = True
            break
        theta = theta_new

    theta_hat = round(theta, 6)
    nll_final = cache.get(theta_hat, _nb2_nll(y_arr, mu, w_arr, theta_hat))

    return NBProfileResult(
        theta_hat=theta_hat,
        nll=nll_final,
        n_evaluations=iteration + 1,
        converged=converged,
        cache=cache,
        _y=y_arr,
        _mu=mu,
        _weights=w_arr,
    )


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
    from scipy.optimize import brentq
    from scipy.stats import chi2

    w_sum = float(np.sum(weights))
    nll_hat = _nb2_nll(y, mu, weights, theta_hat)
    cutoff = chi2.ppf(1.0 - alpha, 1)

    def objective(theta: float) -> float:
        return 2.0 * w_sum * (_nb2_nll(y, mu, weights, theta) - nll_hat) - cutoff

    # Find lower bound
    lo = theta_range[0]
    try:
        ci_lower = brentq(objective, lo, theta_hat, xtol=1e-4)
    except ValueError:
        ci_lower = lo

    # Find upper bound
    hi = theta_range[1]
    try:
        ci_upper = brentq(objective, theta_hat, hi, xtol=1e-4)
    except ValueError:
        ci_upper = hi

    return (ci_lower, ci_upper)
