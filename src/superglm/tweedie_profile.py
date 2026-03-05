"""Tweedie profile likelihood — estimate p from data.

For p ∈ (1, 2), the Tweedie distribution is a compound Poisson-Gamma.
Given a configured SuperGLM, this module builds the design matrix once,
then loops Brent iterations calling fit_pirls directly with warm starts.

References
----------
- Dunn & Smyth (2005): Series evaluation of Tweedie EDMs
- Yang, Qian & Zou (2018): Insurance Premium Prediction via Tweedie CPMs
- Jørgensen (1997): Theory of dispersion models
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import wright_bessel

from superglm.solvers.pirls import fit_pirls

# ---------------------------------------------------------------------------
# Compound Poisson-Gamma simulation
# ---------------------------------------------------------------------------


def generate_tweedie_cpg(
    n: int,
    mu: float | NDArray,
    phi: float,
    p: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Simulate Tweedie(mu, phi, p) via compound Poisson-Gamma.

    Parameters
    ----------
    n : int
        Number of samples.
    mu : float or array of shape (n,)
        Mean parameter.
    phi : float
        Dispersion parameter (>0).
    p : float
        Power parameter, must be in (1, 2).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    y : ndarray of shape (n,)
        Simulated responses (non-negative, with exact zeros).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), (n,)).copy()

    # CPG parameters (Jørgensen 1997)
    lam = np.power(mu, 2 - p) / ((2 - p) * phi)  # Poisson rate
    alpha = (2 - p) / (p - 1)  # Gamma shape per claim
    beta = phi * (p - 1) * np.power(mu, p - 1)  # Gamma scale per claim

    # Vectorised: draw N ~ Poisson(lam), then Y|N ~ Gamma(alpha*N, beta)
    N = rng.poisson(lam)
    y = np.zeros(n, dtype=np.float64)
    pos = N > 0
    if np.any(pos):
        # Gamma additive property: sum of N iid Gamma(alpha, beta) = Gamma(N*alpha, beta)
        y[pos] = rng.gamma(alpha * N[pos], scale=beta[pos])

    return y


# ---------------------------------------------------------------------------
# Tweedie log-pdf
# ---------------------------------------------------------------------------


def tweedie_logpdf(
    y: NDArray,
    mu: NDArray,
    phi: float,
    p: float,
    *,
    weights: NDArray | None = None,
    t_arg_limit: float = 50.0,
) -> NDArray:
    """Exact Tweedie log-density with saddlepoint fallback.

    Parameters
    ----------
    y, mu : arrays of shape (n,)
        Observations and fitted means.
    phi : float
        Dispersion parameter.
    p : float
        Power parameter in (1, 2).
    weights : array of shape (n,), optional
        Observation weights (e.g. exposure). Effective phi = phi / w.
    t_arg_limit : float
        Switch to saddlepoint when wright_bessel argument t >= this.

    Returns
    -------
    logpdf : ndarray of shape (n,)
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    n = len(y)

    phi_eff = np.full(n, phi, dtype=np.float64)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        phi_eff = phi / weights

    logpdf = np.zeros(n, dtype=np.float64)

    # --- Case 1: y = 0 (point mass) ---
    zero = y == 0
    if np.any(zero):
        # P(Y=0) = exp(-lambda) where lambda = mu^(2-p) / ((2-p) * phi_eff)
        logpdf[zero] = -np.power(mu[zero], 2 - p) / ((2 - p) * phi_eff[zero])

    # --- Case 2: y > 0 (continuous density via wright_bessel) ---
    pos = y > 0
    if np.any(pos):
        y_p = y[pos]
        mu_p = mu[pos]
        phi_p = phi_eff[pos]

        # EDM cumulant terms
        theta = np.power(mu_p, 1 - p) / (1 - p)
        kappa = np.power(mu_p, 2 - p) / (2 - p)

        alpha = (2 - p) / (1 - p)  # < 0 for p in (1,2)

        # Wright-Bessel argument: t = ((p-1)*phi/y)^alpha / ((2-p)*phi)
        log_base = np.log((p - 1) * phi_p) - np.log(y_p)
        log_t = alpha * log_base - np.log((2 - p) * phi_p)
        t = np.exp(np.clip(log_t, -700, 700))

        # Partition into wright_bessel-safe vs saddlepoint
        use_wb = t < t_arg_limit
        results = np.zeros(len(y_p), dtype=np.float64)

        if np.any(use_wb):
            t_wb = t[use_wb]
            y_wb = y_p[use_wb]
            mu_wb = mu_p[use_wb]
            phi_wb = phi_p[use_wb]
            theta_wb = theta[use_wb]
            kappa_wb = kappa[use_wb]

            with np.errstate(all="ignore"):
                wb = wright_bessel(-alpha, 0.0, t_wb)

            valid = np.isfinite(wb) & (wb > 1e-300)
            if np.any(valid):
                log_a = np.log(wb[valid]) - np.log(y_wb[valid])
                results_wb = np.full(len(t_wb), -np.inf, dtype=np.float64)
                results_wb[valid] = (
                    log_a + (y_wb[valid] * theta_wb[valid] - kappa_wb[valid]) / phi_wb[valid]
                )
                # Fallback for invalid wb within the wb branch
                invalid = ~valid
                if np.any(invalid):
                    results_wb[invalid] = _saddlepoint(
                        y_wb[invalid], mu_wb[invalid], phi_wb[invalid], p
                    )
                results[use_wb] = results_wb

        use_sp = ~use_wb
        if np.any(use_sp):
            results[use_sp] = _saddlepoint(y_p[use_sp], mu_p[use_sp], phi_p[use_sp], p)

        logpdf[pos] = results

    return logpdf


def _saddlepoint(y: NDArray, mu: NDArray, phi: NDArray, p: float) -> NDArray:
    """Saddlepoint approximation to the Tweedie log-density."""
    y_safe = np.maximum(y, 1e-300)
    term1 = y * (np.power(y_safe, 1 - p) - np.power(mu, 1 - p)) / (1 - p)
    term2 = (np.power(y_safe, 2 - p) - np.power(mu, 2 - p)) / (2 - p)
    deviance = 2 * (term1 - term2)
    return -0.5 * np.log(2 * np.pi * phi * np.power(y_safe, p)) - deviance / (2 * phi)


# ---------------------------------------------------------------------------
# Dispersion estimation
# ---------------------------------------------------------------------------


def estimate_phi(
    y: NDArray,
    mu: NDArray,
    p: float,
    *,
    weights: NDArray | None = None,
    df_resid: int | None = None,
) -> float:
    """Weighted Pearson estimate of dispersion parameter phi.

    phi_hat = sum(w * (y - mu)^2 / mu^p) / denom

    where denom = df_resid if provided, else sum(w) (i.e. no df correction).
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu_safe = np.maximum(mu, 1e-10)
    variance_fn = np.power(mu_safe, p)
    pearson = (y - mu) ** 2 / variance_fn

    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        numer = float(np.sum(weights * pearson))
        denom = float(df_resid if df_resid is not None else np.sum(weights))
    else:
        numer = float(np.sum(pearson))
        denom = float(df_resid if df_resid is not None else len(y))
    return numer / denom


# ---------------------------------------------------------------------------
# Profile likelihood result
# ---------------------------------------------------------------------------


@dataclass
class TweedieProfileResult:
    """Result of Tweedie power parameter estimation."""

    p_hat: float
    phi_hat: float
    nll: float
    n_evaluations: int
    converged: bool
    cache: dict[float, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Profile likelihood optimiser
# ---------------------------------------------------------------------------


def estimate_tweedie_p(
    model,
    X,
    y,
    exposure=None,
    offset=None,
    *,
    p_bounds: tuple[float, float] = (1.05, 1.95),
    xatol: float = 1e-3,
    maxiter: int = 30,
    verbose: bool = False,
) -> TweedieProfileResult:
    """Estimate the Tweedie power parameter via profile likelihood.

    Builds the design matrix once, then for each candidate p calls
    fit_pirls directly with warm starts from the previous solution.
    Optimised via bounded Brent (scipy minimize_scalar).

    Parameters
    ----------
    model : SuperGLM
        A configured but *unfitted* model with features already added.
        Must have family="tweedie" or a Tweedie distribution instance.
    X : DataFrame
        Feature matrix.
    y : array-like
        Response variable.
    exposure : array-like, optional
        Frequency weights (exposure). Must be frequency weights, not
        variance weights — phi estimation and the profile likelihood
        for p assume frequency weight scaling.
    offset : array-like, optional
        Offset added to the linear predictor.
    p_bounds : tuple
        Bounds for p search, default (1.05, 1.95).
    xatol : float
        Tolerance for Brent's method.
    maxiter : int
        Maximum iterations for the optimiser.
    verbose : bool
        Print progress.

    Returns
    -------
    TweedieProfileResult
    """
    from superglm.distributions import Tweedie

    # Validate family
    family = model.family
    is_tweedie = (isinstance(family, str) and family == "tweedie") or isinstance(family, Tweedie)
    if not is_tweedie:
        raise ValueError(f"estimate_tweedie_p requires family='tweedie', got {family!r}")

    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if exposure is None else np.asarray(exposure, dtype=np.float64)

    # --- One-time setup: build design matrix and calibrate lambda ---
    if model._splines is not None and not model._specs:
        model._auto_detect_features(X, exposure)

    # Set a temporary p so _build_design_matrix can resolve the distribution.
    # The design matrix itself doesn't depend on p at all.
    saved_p = model.tweedie_p
    model.tweedie_p = 1.5  # midpoint, any valid value works
    y_arr, w_arr, offset_arr = model._build_design_matrix(X, y, exposure, offset)
    model.tweedie_p = saved_p

    # Calibrate lambda1 once if not already set
    if model.penalty.lambda1 is None:
        model.penalty.lambda1 = model._compute_lambda_max(y_arr, w_arr) * 0.1

    # Ensure offset is an array for eta computation
    if offset_arr is None:
        offset_arr = np.zeros(len(y_arr))

    dm = model._dm
    groups = model._groups
    link = model._link
    penalty = model.penalty

    # Warm-start state (updated across Brent evaluations)
    warm_beta = None
    warm_intercept = None
    # Track the last PIRLS result for final phi estimation
    last_p_eval = None
    last_mu = None

    cache: dict[float, float] = {}
    n_evals = 0

    def objective(p: float) -> float:
        nonlocal n_evals, warm_beta, warm_intercept, last_p_eval, last_mu
        key = round(p, 6)
        if key in cache:
            return cache[key]

        dist = Tweedie(p)
        result = fit_pirls(
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

        eta = np.clip(dm.matvec(result.beta) + result.intercept + offset_arr, -20, 20)
        mu = np.maximum(link.inverse(eta), 1e-10)

        phi = max(estimate_phi(y_arr, mu, p, weights=w_arr), 1e-10)

        ll = tweedie_logpdf(y_arr, mu, phi, p, weights=w_arr)
        nll = -np.sum(w_arr * ll) / np.sum(w_arr)

        # Update warm starts for next evaluation
        warm_beta = result.beta
        warm_intercept = result.intercept
        last_p_eval = p
        last_mu = mu

        cache[key] = nll
        n_evals += 1
        if verbose:
            print(f"  p={p:.4f}  phi={phi:.4f}  nll={nll:.4f}  pirls_iters={result.n_iter}")
        return nll

    result = minimize_scalar(
        objective,
        bounds=p_bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    p_hat = round(result.x, 6)
    nll = result.fun

    # Get phi at p_hat. If the last evaluation was at p_hat, reuse mu;
    # otherwise do one final (warm-started) fit.
    if last_p_eval is not None and round(last_p_eval, 6) == p_hat:
        mu_final = last_mu
    else:
        dist = Tweedie(p_hat)
        final_result = fit_pirls(
            X=dm, y=y_arr, weights=w_arr, family=dist, link=link,
            groups=groups, penalty=penalty, offset=offset_arr,
            beta_init=warm_beta, intercept_init=warm_intercept,
        )
        eta = np.clip(dm.matvec(final_result.beta) + final_result.intercept + offset_arr, -20, 20)
        mu_final = np.maximum(link.inverse(eta), 1e-10)

    phi_hat = max(estimate_phi(y_arr, mu_final, p_hat, weights=w_arr), 1e-10)

    return TweedieProfileResult(
        p_hat=p_hat,
        phi_hat=phi_hat,
        nll=nll,
        n_evaluations=n_evals,
        converged=result.success if hasattr(result, "success") else True,
        cache=cache,
    )
