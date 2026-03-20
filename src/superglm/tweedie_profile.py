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
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import wright_bessel

from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.penalties.base import penalty_has_targets
from superglm.solvers.irls_direct import fit_irls_direct
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
    t_arg_limit: float = 1e4,
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
        Observation weights (e.g. sample_weight). Effective phi = phi / w.
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
    df_resid: float | None = None,
) -> float:
    """Weighted Pearson estimate of dispersion parameter phi.

    phi_hat = sum(w * (y - mu)^2 / mu^p) / denom

    Under the prior-weight convention used here,
    ``Var(Y_i) = phi * mu_i^p / w_i``. Therefore
    ``E[w_i * (Y_i - mu_i)^2 / mu_i^p] = phi`` for each observation, and the
    natural denominator is the residual observation count rather than the sum
    of weights.

    where denom = df_resid if provided, else n_obs (i.e. no df correction).

    Note: for frequency-weighted data, callers should pass
    ``df_resid = sum(weights) - edf``, not ``n - edf``.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu_safe = np.maximum(mu, 1e-10)
    variance_fn = np.power(mu_safe, p)
    pearson = (y - mu) ** 2 / variance_fn

    denom = float(df_resid if df_resid is not None else len(y))
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        numer = float(np.sum(weights * pearson))
    else:
        numer = float(np.sum(pearson))
    return numer / denom


def _profile_phi(
    y: NDArray,
    mu: NDArray,
    p: float,
    *,
    weights: NDArray | None = None,
    df_resid: float | None = None,
    phi_method: str = "pearson",
) -> tuple[float, float]:
    """Profile out phi and return ``(phi_hat, mean_nll)`` for fixed ``(mu, p)``."""
    if phi_method == "pearson":
        phi_hat = max(estimate_phi(y, mu, p, weights=weights, df_resid=df_resid), 1e-10)
        ll = tweedie_logpdf(y, mu, phi_hat, p, weights=weights)
        return phi_hat, float(-np.mean(ll))

    if phi_method != "mle":
        raise ValueError(
            f"phi_method={phi_method!r} is not valid, expected one of ['mle', 'pearson']"
        )

    phi_init = max(estimate_phi(y, mu, p, weights=weights, df_resid=df_resid), 1e-10)
    log_phi_init = float(np.log(phi_init))
    log_lo = max(np.log(1e-12), log_phi_init - 8.0)
    log_hi = min(np.log(1e12), log_phi_init + 8.0)

    def objective(log_phi: float) -> float:
        ll = tweedie_logpdf(y, mu, float(np.exp(log_phi)), p, weights=weights)
        return float(-np.mean(ll))

    opt = minimize_scalar(
        objective,
        bounds=(log_lo, log_hi),
        method="bounded",
        options={"xatol": 1e-3, "maxiter": 50},
    )
    phi_hat = float(np.exp(opt.x))
    return phi_hat, float(opt.fun)


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

    # Stored to enable .ci()
    _objective: Any = field(default=None, repr=False)
    _ll_scale: float = field(default=0.0, repr=False)

    _ci_cache: dict[float, tuple[float, float]] = field(default_factory=dict, repr=False)

    def ci(self, alpha: float = 0.05) -> tuple[float, float]:
        """Profile likelihood confidence interval for Tweedie p.

        Requires that the result was produced by ``estimate_tweedie_p``.
        Results are cached so repeated calls (e.g. from summary()) are free.
        """
        if alpha in self._ci_cache:
            return self._ci_cache[alpha]
        if self._objective is None:
            raise RuntimeError(
                "Profile CI requires the objective function. Use "
                "estimate_tweedie_p() to produce this result."
            )
        result = profile_ci_p(self._objective, self.p_hat, self.nll, self._ll_scale, alpha=alpha)
        self._ci_cache[alpha] = result
        return result

    def profile_plot(
        self,
        *,
        alpha: float = 0.05,
        n_points: int = 50,
        ax=None,
    ):
        """Profile deviance plot for Tweedie power parameter p.

        Evaluates the profile objective on a grid. Each grid point that
        is not already cached requires a PIRLS refit.

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
        if self._objective is None:
            raise RuntimeError(
                "Profile plot requires the objective function. Use "
                "estimate_tweedie_p() to produce this result."
            )

        import matplotlib.pyplot as plt
        from scipy.stats import chi2

        # Snapshot original Brent evaluation points before CI/grid pollute cache
        orig_cache = dict(self.cache)

        ci_lo, ci_hi = self.ci(alpha=alpha)

        # Grid must cover CI and all cached Brent points
        margin = 0.2 * (ci_hi - ci_lo)
        grid_lo = max(1.01, ci_lo - margin)
        grid_hi = min(1.99, ci_hi + margin)
        if orig_cache:
            grid_lo = min(grid_lo, min(orig_cache.keys()) - 0.005)
            grid_hi = max(grid_hi, max(orig_cache.keys()) + 0.005)
            grid_lo = max(1.01, grid_lo)
            grid_hi = min(1.99, grid_hi)
        p_grid = np.linspace(grid_lo, grid_hi, n_points)

        nll_values = np.array([self._objective(p) for p in p_grid])
        deviance = 2.0 * self._ll_scale * (nll_values - self.nll)

        cutoff = chi2.ppf(1.0 - alpha, 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.get_figure()

        ax.plot(p_grid, deviance, color="steelblue", linewidth=1.5)

        # Mark original Brent evaluation points (not grid/CI evals)
        if orig_cache:
            cache_ps = np.array(sorted(orig_cache.keys()))
            cache_dev = (
                2.0 * self._ll_scale * (np.array([orig_cache[p] for p in cache_ps]) - self.nll)
            )
            ax.scatter(
                cache_ps,
                cache_dev,
                color="darkorange",
                s=35,
                zorder=5,
                edgecolors="white",
                linewidths=0.5,
                label=f"Brent evals ({len(cache_ps)})",
            )

        ax.axhline(
            cutoff,
            linestyle="--",
            color="grey",
            linewidth=0.8,
            label=f"{100 * (1 - alpha):.0f}% cutoff",
        )
        ax.axvline(
            self.p_hat,
            linestyle=":",
            color="black",
            linewidth=0.8,
            label=f"MLE = {self.p_hat:.3f}",
        )
        ax.fill_betweenx(
            [0, cutoff],
            ci_lo,
            ci_hi,
            alpha=0.10,
            color="firebrick",
            label=f"{100 * (1 - alpha):.0f}% CI: [{ci_lo:.3f}, {ci_hi:.3f}]",
        )

        ax.set_xlabel("p")
        ax.set_ylabel("Profile deviance")
        ax.set_title("Tweedie p profile likelihood")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        return fig


# ---------------------------------------------------------------------------
# Profile likelihood optimiser
# ---------------------------------------------------------------------------


def estimate_tweedie_p(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    p_bounds: tuple[float, float] = (1.05, 1.95),
    xatol: float = 1e-3,
    maxiter: int = 30,
    verbose: bool = False,
    fit_mode: str = "fit",
    phi_method: str = "pearson",
) -> TweedieProfileResult:
    """Estimate the Tweedie power parameter via profile likelihood.

    Dispatches to one of two internal paths based on *fit_mode*:

    - ``"fit"`` (default): builds the design matrix once and calls
      ``fit_pirls`` directly with warm starts for each candidate p.
    - ``"fit_reml"``: calls ``model.fit_reml()`` for each candidate p,
      re-estimating smoothing parameters at every evaluation.

    Both paths use bounded Brent (scipy minimize_scalar) over p.

    Parameters
    ----------
    model : SuperGLM
        A configured but *unfitted* model with features already added.
        Must have a Tweedie family (e.g. ``families.tweedie(p=1.5)``).
    X : DataFrame
        Feature matrix.
    y : array-like
        Response variable.
    sample_weight : array-like, optional
        Frequency weights (sample_weight). Must be frequency weights, not
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
    fit_mode : {"fit", "fit_reml"}
        Fitting regime for each candidate p evaluation.
    phi_method : {"pearson", "mle"}
        How to profile out ``phi`` at each candidate ``p``.

    Returns
    -------
    TweedieProfileResult
    """
    from superglm.distributions import Tweedie

    # Validate family
    family = model.family
    if not isinstance(family, Tweedie):
        raise ValueError(
            f"estimate_tweedie_p requires a Tweedie family, got {family!r}. "
            "Use families.tweedie(p=...) to create one."
        )

    _VALID_FIT_MODES = {"fit", "fit_reml"}
    if fit_mode not in _VALID_FIT_MODES:
        raise ValueError(
            f"fit_mode={fit_mode!r} is not valid, expected one of {sorted(_VALID_FIT_MODES)}"
        )
    _VALID_PHI_METHODS = {"pearson", "mle"}
    if phi_method not in _VALID_PHI_METHODS:
        raise ValueError(
            f"phi_method={phi_method!r} is not valid, expected one of {sorted(_VALID_PHI_METHODS)}"
        )

    if fit_mode == "fit_reml":
        return _estimate_tweedie_p_reml(
            model,
            X,
            y,
            sample_weight,
            offset,
            p_bounds=p_bounds,
            xatol=xatol,
            maxiter=maxiter,
            verbose=verbose,
            phi_method=phi_method,
        )

    return _estimate_tweedie_p_fit(
        model,
        X,
        y,
        sample_weight,
        offset,
        p_bounds=p_bounds,
        xatol=xatol,
        maxiter=maxiter,
        verbose=verbose,
        phi_method=phi_method,
    )


def _estimate_tweedie_p_fit(
    model,
    X,
    y,
    sample_weight,
    offset,
    *,
    p_bounds,
    xatol,
    maxiter,
    verbose,
    phi_method,
) -> TweedieProfileResult:
    """Profile p using fit_pirls (original path)."""
    from superglm.distributions import Tweedie

    y = np.asarray(y, dtype=np.float64)

    # --- One-time setup: build design matrix and calibrate lambda ---
    if model._splines is not None and not model._specs:
        model._auto_detect_features(X, sample_weight)

    # Set a temporary p so _build_design_matrix can resolve the distribution.
    # The design matrix itself doesn't depend on p at all.
    saved_family = model.family
    model.family = Tweedie(p=1.5)  # midpoint, any valid value works
    y_arr, w_arr, offset_arr = model._build_design_matrix(X, y, sample_weight, offset)
    model.family = saved_family

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

    # Use direct solver when lambda1=0 (no L1 penalty → no BCD needed),
    # matching the dispatcher in fit_ops.fit().
    _use_direct = penalty.lambda1 is not None and (
        penalty.lambda1 == 0 or not penalty_has_targets(penalty, groups)
    )

    # Warm-start state (updated across Brent evaluations)
    warm_beta = None
    warm_intercept = None
    # Track the last PIRLS result for final phi estimation
    last_p_eval = None
    last_mu = None
    last_edf = None

    cache: dict[float, float] = {}
    n_evals = 0

    def objective(p: float) -> float:
        nonlocal n_evals, warm_beta, warm_intercept, last_p_eval, last_mu, last_edf
        key = round(p, 6)
        if key in cache:
            return cache[key]

        dist = Tweedie(p)
        if _use_direct:
            result, _ = fit_irls_direct(
                X=dm,
                y=y_arr,
                weights=w_arr,
                family=dist,
                link=link,
                groups=groups,
                lambda2=model.lambda2,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
            )
        else:
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

        eta = stabilize_eta(dm.matvec(result.beta) + result.intercept + offset_arr, link)
        mu = clip_mu(link.inverse(eta), dist)
        # sum(weights) - edf for frequency-weighted data (matches statsmodels)
        df_resid = max(float(np.sum(w_arr)) - float(result.effective_df), 1.0)

        phi, nll = _profile_phi(
            y_arr,
            mu,
            p,
            weights=w_arr,
            df_resid=df_resid,
            phi_method=phi_method,
        )

        # Update warm starts for next evaluation
        warm_beta = result.beta
        warm_intercept = result.intercept
        last_p_eval = p
        last_mu = mu
        last_edf = float(result.effective_df)

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
        edf_final = last_edf
    else:
        dist = Tweedie(p_hat)
        if _use_direct:
            final_result, _ = fit_irls_direct(
                X=dm,
                y=y_arr,
                weights=w_arr,
                family=dist,
                link=link,
                groups=groups,
                lambda2=model.lambda2,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
            )
        else:
            final_result = fit_pirls(
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
        eta = stabilize_eta(
            dm.matvec(final_result.beta) + final_result.intercept + offset_arr, link
        )
        mu_final = clip_mu(link.inverse(eta), dist)
        edf_final = float(final_result.effective_df)

    df_resid_final = max(float(np.sum(w_arr)) - float(edf_final), 1.0)
    phi_hat, _ = _profile_phi(
        y_arr,
        mu_final,
        p_hat,
        weights=w_arr,
        df_resid=df_resid_final,
        phi_method=phi_method,
    )

    return TweedieProfileResult(
        p_hat=p_hat,
        phi_hat=phi_hat,
        nll=nll,
        n_evaluations=n_evals,
        converged=result.success if hasattr(result, "success") else True,
        cache=cache,
        _objective=objective,
        _ll_scale=float(len(y_arr)),
    )


def _estimate_tweedie_p_reml(
    model,
    X,
    y,
    sample_weight,
    offset,
    *,
    p_bounds,
    xatol,
    maxiter,
    verbose,
    phi_method,
) -> TweedieProfileResult:
    """Profile p using fit_reml for each candidate."""
    from superglm.distributions import Tweedie

    y_np = np.asarray(y, dtype=np.float64)
    w_arr = (
        np.asarray(sample_weight, dtype=np.float64)
        if sample_weight is not None
        else np.ones(len(y_np))
    )

    cache: dict[float, float] = {}
    n_evals = 0
    last_p_eval = None
    last_mu = None
    last_edf = None

    def objective(p: float) -> float:
        nonlocal n_evals, last_p_eval, last_mu, last_edf
        key = round(p, 6)
        if key in cache:
            return cache[key]

        # Set p and refit via REML
        model.family = Tweedie(p=p)
        model.fit_reml(X, y, offset=offset)

        mu = np.maximum(model.predict(X), 1e-10)
        df_resid = max(float(np.sum(w_arr)) - float(model.result.effective_df), 1.0)
        phi, nll = _profile_phi(
            y_np,
            mu,
            p,
            weights=w_arr,
            df_resid=df_resid,
            phi_method=phi_method,
        )

        last_p_eval = p
        last_mu = mu
        last_edf = float(model.result.effective_df)

        cache[key] = nll
        n_evals += 1
        if verbose:
            reml_iters = model._reml_result.n_reml_iter if hasattr(model, "_reml_result") else "?"
            print(f"  p={p:.4f}  phi={phi:.4f}  nll={nll:.4f}  reml_iters={reml_iters}")
        return nll

    result = minimize_scalar(
        objective,
        bounds=p_bounds,
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    p_hat = round(result.x, 6)
    nll = result.fun

    # Ensure we have mu at p_hat for phi
    if last_p_eval is None or round(last_p_eval, 6) != p_hat:
        model.family = Tweedie(p=p_hat)
        model.fit_reml(X, y, offset=offset)
        last_mu = np.maximum(model.predict(X), 1e-10)
        last_edf = float(model.result.effective_df)

    df_resid_final = max(float(np.sum(w_arr)) - float(last_edf), 1.0)
    phi_hat, _ = _profile_phi(
        y_np,
        last_mu,
        p_hat,
        weights=w_arr,
        df_resid=df_resid_final,
        phi_method=phi_method,
    )

    return TweedieProfileResult(
        p_hat=p_hat,
        phi_hat=phi_hat,
        nll=nll,
        n_evaluations=n_evals,
        converged=result.success if hasattr(result, "success") else True,
        cache=cache,
        _objective=objective,
        _ll_scale=float(len(y_np)),
    )


def profile_ci_p(
    objective,
    p_hat: float,
    nll_hat: float,
    ll_scale: float,
    *,
    alpha: float = 0.05,
    p_range: tuple[float, float] = (1.02, 1.98),
) -> tuple[float, float]:
    """Profile likelihood confidence interval for Tweedie power p.

    Each evaluation calls ``objective(p)`` which refits the GLM via PIRLS.
    The objective returns mean NLL per observation, so the LRT
    statistic is ``2 * ll_scale * (mean_nll(p) - mean_nll(p_hat))``.

    Parameters
    ----------
    objective : callable
        Profile objective ``p -> mean_nll``.
    p_hat : float
        MLE of p.
    nll_hat : float
        Mean NLL at p_hat.
    ll_scale : float
        Number of effective observations used to convert mean NLL to total.
    alpha : float
        Significance level.
    p_range : tuple
        Search range for CI endpoints.

    Returns
    -------
    (ci_lower, ci_upper) : tuple of float
    """
    from scipy.optimize import brentq
    from scipy.stats import chi2

    cutoff = chi2.ppf(1.0 - alpha, 1)

    def g(p: float) -> float:
        nll = objective(p)
        return 2.0 * ll_scale * (nll - nll_hat) - cutoff

    # Lower bound
    lo = p_range[0]
    try:
        ci_lower = brentq(g, lo, p_hat, xtol=1e-3)
    except ValueError:
        ci_lower = lo

    # Upper bound
    hi = p_range[1]
    try:
        ci_upper = brentq(g, p_hat, hi, xtol=1e-3)
    except ValueError:
        ci_upper = hi

    return (ci_lower, ci_upper)
