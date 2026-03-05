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
        score = np.sum(weights * (
            digamma(y + theta) - digamma(theta)
            + np.log(theta) + 1.0 - np.log(theta + mu)
            - (y + theta) / (mu + theta)
        ))
        # Information: -d²ℓ/dθ²
        info = np.sum(weights * (
            -polygamma(1, y + theta) + polygamma(1, theta)
            - 1.0 / theta + 2.0 / (mu + theta)
            - (y + theta) / (mu + theta) ** 2
        ))
        if abs(info) < 1e-20:
            break
        delta = score / info
        theta_new = np.clip(theta + delta, bounds[0], bounds[1])
        if abs(theta_new - theta) / (theta + 1e-10) < eps:
            theta = float(theta_new)
            break
        theta = float(theta_new)
    return theta


def _nb2_nll(
    y: NDArray, mu: NDArray, weights: NDArray, theta: float
) -> float:
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
        raise ValueError(
            f"estimate_nb_theta requires family='negative_binomial', got {family!r}"
        )

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
            -20, 20,
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
    )
