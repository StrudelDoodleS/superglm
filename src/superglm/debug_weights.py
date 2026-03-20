"""Diagnostic utilities for debugging IRLS working weight issues.

Usage (at work, where you have the data):

    from superglm.debug_weights import compare_irls_weights

    # After fitting your superglm model:
    report = compare_irls_weights(model, X, y, exposure=exposure)
    print(report)

    # Or just inspect superglm's iteration log:
    model.fit(X, y, exposure=exposure, record_diagnostics=True)
    print(model.iteration_diagnostics())
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compare_irls_weights(
    model,
    X: pd.DataFrame,
    y: NDArray,
    exposure: NDArray | None = None,
    offset: NDArray | None = None,
    max_iter: int = 5,
) -> pd.DataFrame:
    """Compare IRLS working weights between superglm and statsmodels.

    Fits the same data with statsmodels GLM and logs per-iteration W stats
    side-by-side.  Useful for diagnosing why superglm diverges on data
    that statsmodels handles.

    Parameters
    ----------
    model : SuperGLM
        A fitted superglm model (must already be fitted).
    X : DataFrame
        Feature data used for fitting.
    y : array-like
        Response variable.
    exposure : array-like, optional
        Frequency weights / exposure.
    offset : array-like, optional
        Offset term.
    max_iter : int
        Number of statsmodels IRLS iterations to log (default 5).

    Returns
    -------
    DataFrame
        Side-by-side comparison of W stats per iteration.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError(
            "statsmodels is required for compare_irls_weights(). "
            "Install with: pip install statsmodels"
        )

    from superglm.distributions import (
        Binomial,
        Gamma,
        Gaussian,
        NegativeBinomial,
        Poisson,
        Tweedie,
    )
    from superglm.links import IdentityLink, LogitLink, LogLink

    # Map superglm family to statsmodels
    family = model._distribution
    link = model._link

    if isinstance(family, Poisson):
        sm_family = sm.families.Poisson()
    elif isinstance(family, Gaussian):
        sm_family = sm.families.Gaussian()
    elif isinstance(family, Gamma):
        sm_family = sm.families.Gamma()
    elif isinstance(family, Binomial):
        sm_family = sm.families.Binomial()
    elif isinstance(family, NegativeBinomial):
        sm_family = sm.families.NegativeBinomial(alpha=1.0 / family.theta)
    elif isinstance(family, Tweedie):
        sm_family = sm.families.Tweedie(var_power=family.p)
    else:
        raise ValueError(f"Unsupported family for comparison: {type(family)}")

    # Map link
    if isinstance(link, LogLink):
        sm_family.link = sm.families.links.Log()
    elif isinstance(link, LogitLink):
        sm_family.link = sm.families.links.Logit()
    elif isinstance(link, IdentityLink):
        sm_family.link = sm.families.links.Identity()

    # Build design matrix — intercept-only for comparison
    # Use the raw X columns as numeric features
    X_sm = sm.add_constant(X.select_dtypes(include=[np.number]).values)

    freq_weights = exposure if exposure is not None else np.ones(len(y))
    sm_offset = offset if offset is not None else None

    rows = []
    try:
        sm_model = sm.GLM(
            y,
            X_sm,
            family=sm_family,
            freq_weights=freq_weights,
            offset=sm_offset,
        )
        # Per-iteration stats: fit with maxiter=1,2,...,max_iter
        # and record W at each step. statsmodels doesn't expose
        # per-iteration internals, so we re-fit with increasing
        # maxiter and warm-start from the intercept-only model.
        for it in range(1, max_iter + 1):
            sm_result = sm_model.fit(maxiter=it, disp=0)
            mu_it = sm_result.mu
            w_family = sm_family.weights(mu_it)
            W_it = freq_weights * w_family
            n_sm = len(W_it)
            k = min(5, n_sm)
            top_idx = np.argpartition(W_it, -k)[-k:]
            bot_idx = np.argpartition(W_it, k)[:k]
            rows.append(
                {
                    "iter": it,
                    "source": "statsmodels",
                    "W_min": float(W_it.min()),
                    "W_max": float(W_it.max()),
                    "W_ratio": float(W_it.max() / max(W_it.min(), 1e-300)),
                    "mu_min": float(mu_it.min()),
                    "mu_max": float(mu_it.max()),
                    "deviance": float(sm_result.deviance),
                    "converged": sm_result.converged,
                    "top_W_obs": list(top_idx[np.argsort(W_it[top_idx])[::-1]]),
                    "bottom_W_obs": list(bot_idx[np.argsort(W_it[bot_idx])]),
                }
            )
    except Exception as e:
        rows.append(
            {
                "iter": "error",
                "source": "statsmodels",
                "W_min": np.nan,
                "W_max": np.nan,
                "W_ratio": np.nan,
                "mu_min": np.nan,
                "mu_max": np.nan,
                "deviance": np.nan,
                "converged": False,
                "top_W_obs": [],
                "bottom_W_obs": [],
                "error": str(e),
            }
        )

    # Add superglm iteration log if available
    if model.result.iteration_log is not None:
        for d in model.result.iteration_log:
            rows.append(
                {
                    "iter": d.iteration,
                    "source": "superglm",
                    "W_min": d.w_min,
                    "W_max": d.w_max,
                    "W_ratio": d.w_ratio,
                    "mu_min": d.mu_min,
                    "mu_max": d.mu_max,
                    "deviance": d.deviance,
                    "converged": None,
                    "top_W_obs": list(d.top_w_indices),
                    "bottom_W_obs": list(d.bottom_w_indices),
                    "cond_estimate": d.cond_estimate,
                    "used_svd_fallback": d.used_svd_fallback,
                }
            )

    # Sort by source then iteration for easy comparison
    df = pd.DataFrame(rows)
    return df.sort_values(["iter", "source"]).reset_index(drop=True)


def inspect_worst_observations(
    model,
    X: pd.DataFrame,
    y: NDArray,
    exposure: NDArray | None = None,
    iteration: int = 1,
) -> pd.DataFrame:
    """Show the observations with extreme working weights at a given iteration.

    Parameters
    ----------
    model : SuperGLM
        A fitted model with ``record_diagnostics=True``.
    X : DataFrame
        Original feature data.
    y : array-like
        Response variable.
    exposure : array-like, optional
        Frequency weights.
    iteration : int
        Which IRLS iteration to inspect (1-based).

    Returns
    -------
    DataFrame
        Rows for the top-5 and bottom-5 W observations, showing their
        feature values, y, exposure, and which end of W they're on.
    """
    log = model.result.iteration_log
    if log is None:
        raise RuntimeError("No iteration diagnostics. Refit with fit(record_diagnostics=True).")

    # Find the requested iteration
    entry = None
    for d in log:
        if d.iteration == iteration:
            entry = d
            break
    if entry is None:
        available = [d.iteration for d in log]
        raise ValueError(f"Iteration {iteration} not found. Available: {available}")

    top_idx = entry.top_w_indices
    bot_idx = entry.bottom_w_indices
    all_idx = np.concatenate([top_idx, bot_idx])

    rows = []
    exp = exposure if exposure is not None else np.ones(len(y))
    for i in all_idx:
        row = {"obs_index": int(i), "y": float(y[i]), "exposure": float(exp[i])}
        # Add feature values
        for col in X.columns:
            row[col] = X.iloc[i][col]
        row["W_group"] = "top_5" if i in top_idx else "bottom_5"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
