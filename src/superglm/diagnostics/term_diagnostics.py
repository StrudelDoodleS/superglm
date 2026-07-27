"""Term importance and drop-term diagnostics.

# Internal submodules: import siblings directly, not through this __init__.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.inference._term_helpers import _resolve_group_lambda

if TYPE_CHECKING:
    pass


# ── Term importance (Phase 7) ─────────────────────────────────────


def term_importance(
    model,
    X: FrameLike,
    sample_weight: NDArray | None = None,
) -> pd.DataFrame:
    """Weighted variance of each term's contribution to eta.

    For each group, computes the centered variance of X_g @ beta_g
    (the partial linear predictor). Aggregates subgroups at the
    feature level for select=True.

    Parameters
    ----------
    model : SuperGLM
        A fitted model.
    X : pandas or eager Polars DataFrame
        Data to evaluate on (typically training data).
    sample_weight, sample_weight : array-like, optional
        Frequency weights for weighted variance.

    Returns
    -------
    DataFrame
        Columns: term, feature, subgroup_type, variance_eta, sd_eta,
        edf, lambda, group_norm.
    """
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling term_importance().")

    frame = as_eager_frame(X)
    beta = model.result.beta
    weights = sample_weight if sample_weight is not None else np.ones(len(frame))
    w_sum = np.sum(weights)
    group_edf = model._group_edf or {}
    reml_lam = getattr(model, "_reml_lambdas", None) or {}
    lambda2 = getattr(model, "lambda2", None)

    from superglm.model import base

    plan = base._prediction_plan(model)
    terms_by_name = {term["name"]: term for term in (*plan["features"], *plan["interactions"])}

    def _diag_lambda(g_name):
        return _resolve_group_lambda(g_name, reml_lam, lambda2)

    rows = []
    for g in model._groups:
        b_g = beta[g.sl]
        norm_g = float(np.linalg.norm(b_g))

        if norm_g < 1e-12:
            rows.append(
                {
                    "term": g.name,
                    "feature": g.feature_name,
                    "subgroup_type": g.subgroup_type,
                    "variance_eta": 0.0,
                    "sd_eta": 0.0,
                    "edf": group_edf.get(g.name),
                    "lambda": _diag_lambda(g.name),
                    "group_norm": norm_g,
                }
            )
            continue

        term = terms_by_name.get(g.feature_name)
        if term is None:
            raise RuntimeError(f"prediction plan does not define fitted term {g.feature_name!r}")
        term_indices = np.asarray(term["beta_idx"], dtype=np.intp)
        group_positions = (term_indices >= g.start) & (term_indices < g.end)
        if np.count_nonzero(group_positions) != g.size:
            raise RuntimeError(
                f"prediction plan coefficient layout disagrees with group {g.name!r}"
            )
        term_beta = np.zeros(len(term_indices), dtype=np.float64)
        term_beta[group_positions] = beta[term_indices[group_positions]]
        eta_g = base._score_prediction_term_local_exact(term, frame, term_beta)

        # Centered weighted variance
        wmean = np.sum(weights * eta_g) / w_sum
        eta_centered = eta_g - wmean
        var_eta = float(np.sum(weights * eta_centered**2) / w_sum)

        rows.append(
            {
                "term": g.name,
                "feature": g.feature_name,
                "subgroup_type": g.subgroup_type,
                "variance_eta": var_eta,
                "sd_eta": float(np.sqrt(var_eta)),
                "edf": group_edf.get(g.name),
                "lambda": _diag_lambda(g.name),
                "group_norm": norm_g,
            }
        )

    return pd.DataFrame(rows)


# ── Drop-term diagnostics (Phase 8) ──────────────────────────────


def term_drop_diagnostics(
    model,
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    mode: str = "refit",
    X_val: FrameLike | None = None,
    y_val: NDArray | None = None,
) -> pd.DataFrame:
    """Drop-term diagnostics wrapper.

    Parameters
    ----------
    mode : {"refit", "holdout"}
        ``"refit"``: calls ``drop1()`` and adds delta_aic, delta_bic columns.
        ``"holdout"``: zeros each term's contribution on validation set,
        computes loss delta without refitting.
    """

    if mode == "refit":
        return _drop_term_refit(model, X, y, sample_weight, offset)
    elif mode == "holdout":
        if X_val is None or y_val is None:
            raise ValueError("mode='holdout' requires X_val and y_val.")
        return _drop_term_holdout(model, as_eager_frame(X_val), y_val, sample_weight)
    else:
        raise ValueError(f"mode must be 'refit' or 'holdout', got {mode!r}")


def _drop_term_refit(model, X, y, sample_weight, offset) -> pd.DataFrame:
    """Refit-based drop-term diagnostics using drop1()."""

    drop1_df = model.drop1(X, y, offset=offset)

    # Compute IC deltas
    full_ll = model._fit_stats.log_likelihood if model._fit_stats else 0.0
    full_edf = model.result.effective_df
    n = len(y)

    full_aic = -2.0 * full_ll + 2.0 * full_edf
    full_bic = -2.0 * full_ll + np.log(n) * full_edf

    # Add delta columns
    result = drop1_df.copy()
    if "delta_aic" not in result.columns:
        result["delta_aic"] = result["aic"] - full_aic if "aic" in result.columns else np.nan
        result["delta_bic"] = result["bic"] - full_bic if "bic" in result.columns else np.nan

    return result


def _drop_term_holdout(
    model,
    X_val: EagerFrame,
    y_val,
    sample_weight,
) -> pd.DataFrame:
    """Holdout-based drop-term diagnostics (zero each term, compute loss delta)."""

    if model._result is None:
        raise RuntimeError("Model must be fitted.")

    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model import base

    beta = model.result.beta
    dist = model._distribution
    w = sample_weight if sample_weight is not None else np.ones(len(y_val))
    y_arr = np.asarray(y_val, dtype=np.float64)

    plan = base._prediction_plan(model)
    terms = [*plan["features"], *plan["interactions"]]
    eta_raw = np.full(len(X_val), model.result.intercept, dtype=np.float64)
    contributions: dict[str, NDArray[np.floating]] = {}
    for term in terms:
        contribution = base._score_prediction_term_exact(term, X_val, beta)
        contributions[term["name"]] = contribution
        eta_raw += contribution

    eta_full = stabilize_eta(eta_raw, model._link)
    mu_full = clip_mu(model._link.inverse(eta_full), dist)
    dev_full = float(np.sum(w * dist.deviance_unit(y_arr, mu_full)))

    rows = []
    for term in terms:
        eta_drop = stabilize_eta(
            eta_raw - contributions[term["name"]],
            model._link,
        )
        mu_drop = clip_mu(model._link.inverse(eta_drop), dist)
        dev_drop = float(np.sum(w * dist.deviance_unit(y_arr, mu_drop)))

        rows.append(
            {
                "feature": term["name"],
                "delta_deviance": dev_drop - dev_full,
            }
        )

    return pd.DataFrame(rows)
