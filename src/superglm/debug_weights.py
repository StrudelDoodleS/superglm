"""Diagnostic utilities for debugging IRLS working weight issues.

Usage (at work, where you have the data):

    from superglm.debug_weights import compare_irls_weights

    # After fitting your superglm model:
    report = compare_irls_weights(model, X, y)
    print(report)

    # Or just inspect superglm's iteration log:
    model.fit(X, y, record_diagnostics=True)
    print(model.iteration_diagnostics())
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import FrameLike, as_eager_frame
from superglm.solvers.pirls import _extreme_weight_indices, _positive_working_weight_stats

logger = logging.getLogger(__name__)


def compare_irls_weights(
    model,
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
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
    X : pandas or eager Polars DataFrame
        Feature data used for fitting.
    y : array-like
        Response variable.
    sample_weight : array-like, optional
        Fitting weights, read under the model's declared
        ``weight_semantics``.
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

    frame = as_eager_frame(X)

    # Build design matrix — intercept-only for comparison
    # Use the raw X columns as numeric features
    numeric_columns = tuple(name for name in frame.columns if frame.column_kind(name) == "numeric")
    numeric_values = (
        np.column_stack([frame.column_array(name) for name in numeric_columns])
        if numeric_columns
        else np.empty((len(frame), 0), dtype=np.float64)
    )
    X_sm = sm.add_constant(numeric_values)

    comparison_weights = sample_weight if sample_weight is not None else np.ones(len(y))
    # statsmodels exposes the two supported contracts as separate arguments.
    # They enter its IRLS working weights identically, but choosing the matching
    # argument also preserves the family's likelihood and dispersion semantics.
    sm_weight_kwargs = (
        {"var_weights": comparison_weights}
        if isinstance(family, Tweedie)
        else {"freq_weights": comparison_weights}
    )
    sm_offset = offset if offset is not None else None

    rows = []
    try:
        sm_model = sm.GLM(
            y,
            X_sm,
            family=sm_family,
            offset=sm_offset,
            **sm_weight_kwargs,
        )
        # Per-iteration stats: fit with maxiter=1,2,...,max_iter
        # and record W at each step. statsmodels doesn't expose
        # per-iteration internals, so we re-fit with increasing
        # maxiter and warm-start from the intercept-only model.
        for it in range(1, max_iter + 1):
            sm_result = sm_model.fit(maxiter=it, disp=0)
            mu_it = sm_result.mu
            w_family = sm_family.weights(mu_it)
            W_it = comparison_weights * w_family
            _, _, w_ratio = _positive_working_weight_stats(W_it)
            top_idx, bot_idx = _extreme_weight_indices(W_it)
            rows.append(
                {
                    "iter": it,
                    "source": "statsmodels",
                    "W_min": float(W_it.min()),
                    "W_max": float(W_it.max()),
                    "W_ratio": w_ratio,
                    "mu_min": float(mu_it.min()),
                    "mu_max": float(mu_it.max()),
                    "deviance": float(sm_result.deviance),
                    "converged": sm_result.converged,
                    "top_W_obs": list(top_idx),
                    "bottom_W_obs": list(bot_idx),
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
                    "raw_W_min": d.raw_w_min,
                    "raw_W_max": d.raw_w_max,
                    "raw_W_ratio": d.raw_w_ratio,
                    "mu_min": d.mu_min,
                    "mu_max": d.mu_max,
                    "eta_min": d.eta_min,
                    "eta_max": d.eta_max,
                    "eta_min_unclipped": d.eta_min_unclipped,
                    "eta_max_unclipped": d.eta_max_unclipped,
                    "eta_clipped": d.eta_clipped,
                    "working_mu_min": d.working_mu_min,
                    "working_mu_max": d.working_mu_max,
                    "working_eta_min": d.working_eta_min,
                    "working_eta_max": d.working_eta_max,
                    "working_eta_min_unclipped": d.working_eta_min_unclipped,
                    "working_eta_max_unclipped": d.working_eta_max_unclipped,
                    "working_eta_clipped": d.working_eta_clipped,
                    "deviance": d.deviance,
                    "converged": None,
                    "step_halvings": d.step_halvings,
                    "step_rejected": d.step_rejected,
                    "rank_truncated": d.rank_truncated,
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
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    iteration: int = 1,
) -> pd.DataFrame:
    """Show the observations with extreme working weights at a given iteration.

    Parameters
    ----------
    model : SuperGLM
        A fitted model with ``record_diagnostics=True``.
    X : pandas or eager Polars DataFrame
        Original feature data.
    y : array-like
        Response variable.
    sample_weight : array-like, optional
        Fitting weights, read under the model's declared
        ``weight_semantics``.
    iteration : int
        Which IRLS iteration to inspect (1-based).

    Returns
    -------
    DataFrame
        Rows for the top-5 and bottom-5 W observations, showing their
        feature values, y, sample_weight, and which end of W they're on.
    """
    frame = as_eager_frame(X)
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
    weights = sample_weight if sample_weight is not None else np.ones(len(y))
    for i in all_idx:
        row: dict[object, Any] = {
            "obs_index": int(i),
            "y": float(y[i]),
            "sample_weight": float(weights[i]),
        }
        # Add feature values
        for col in frame.columns:
            row[col] = frame.column_array(col)[i]
        row["W_group"] = "top_5" if i in top_idx else "bottom_5"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
