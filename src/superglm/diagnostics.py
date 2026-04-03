"""Term importance, drop-term diagnostics, and spline redundancy.

These are diagnostic-only functions — they do not modify the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


# ── Term importance (Phase 7) ─────────────────────────────────────


def term_importance(
    model,
    X: pd.DataFrame,
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
    X : DataFrame
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

    beta = model.result.beta
    weights = sample_weight if sample_weight is not None else np.ones(len(X))
    w_sum = np.sum(weights)
    group_edf = model._group_edf or {}
    reml_lam = getattr(model, "_reml_lambdas", None) or {}

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
                    "lambda": reml_lam.get(g.name) if isinstance(reml_lam, dict) else None,
                    "group_norm": norm_g,
                }
            )
            continue

        # Compute partial eta for this group
        spec = model._specs.get(g.feature_name)
        ispec = model._interaction_specs.get(g.feature_name) if spec is None else None

        if spec is not None:
            B_g = spec.transform(np.asarray(X[g.feature_name]))
            eta_g = B_g @ b_g
        elif ispec is not None:
            p1, p2 = ispec.parent_names
            B_g = ispec.transform(np.asarray(X[p1]), np.asarray(X[p2]))
            eta_g = B_g @ b_g
        else:
            eta_g = np.zeros(len(X))

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
                "lambda": reml_lam.get(g.name) if isinstance(reml_lam, dict) else None,
                "group_norm": norm_g,
            }
        )

    return pd.DataFrame(rows)


# ── Drop-term diagnostics (Phase 8) ──────────────────────────────


def term_drop_diagnostics(
    model,
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    mode: str = "refit",
    X_val: pd.DataFrame | None = None,
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
        return _drop_term_holdout(model, X_val, y_val, sample_weight)
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


def _drop_term_holdout(model, X_val, y_val, sample_weight) -> pd.DataFrame:
    """Holdout-based drop-term diagnostics (zero each term, compute loss delta)."""

    if model._result is None:
        raise RuntimeError("Model must be fitted.")

    beta = model.result.beta.copy()
    mu_full = model.predict(X_val)
    dist = model._distribution
    w = sample_weight if sample_weight is not None else np.ones(len(y_val))

    # Full model deviance
    y_arr = np.asarray(y_val, dtype=np.float64)
    dev_full = float(np.sum(w * dist.deviance_unit(y_arr, mu_full)))

    rows = []
    seen_features = set()
    for g in model._groups:
        if g.feature_name in seen_features:
            continue
        seen_features.add(g.feature_name)

        # Zero this feature's coefficients
        feature_groups = [fg for fg in model._groups if fg.feature_name == g.feature_name]
        beta_zeroed = beta.copy()
        for fg in feature_groups:
            beta_zeroed[fg.sl] = 0.0

        # Predict with zeroed feature
        blocks = []
        for name in model._feature_order:
            spec = model._specs[name]
            blocks.append(spec.transform(np.asarray(X_val[name])))
        for iname in model._interaction_order:
            ispec = model._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            blocks.append(ispec.transform(np.asarray(X_val[p1]), np.asarray(X_val[p2])))

        from superglm.distributions import clip_mu
        from superglm.links import stabilize_eta

        eta = np.hstack(blocks) @ beta_zeroed + model.result.intercept
        eta = stabilize_eta(eta, model._link)
        mu_drop = clip_mu(model._link.inverse(eta), model._distribution)
        dev_drop = float(np.sum(w * dist.deviance_unit(y_arr, mu_drop)))

        rows.append(
            {
                "feature": g.feature_name,
                "delta_deviance": dev_drop - dev_full,
            }
        )

    return pd.DataFrame(rows)


# ── Spline redundancy diagnostics (Phase 9) ──────────────────────


@dataclass
class SplineRedundancyReport:
    """Redundancy diagnostics for one spline feature."""

    feature_name: str
    n_knots: int
    knot_locations: NDArray
    knot_spacing: NDArray
    support_mass: NDArray  # fraction of data near each knot
    adjacent_basis_corr: NDArray
    coef_energy_penalized: NDArray
    effective_rank: float
    small_singular_values: NDArray = field(default_factory=lambda: np.array([]))


def spline_redundancy(
    model,
    X: pd.DataFrame,
    sample_weight: NDArray | None = None,
) -> dict[str, SplineRedundancyReport]:
    """Spline redundancy diagnostics for all spline features.

    Diagnostic-only. No auto-pruning. Interpretation: "try lower k and refit".
    """
    from superglm.features.spline import _SplineBase

    if model._result is None:
        raise RuntimeError("Model must be fitted.")

    results = {}

    for name, spec in model._specs.items():
        if not isinstance(spec, _SplineBase):
            continue

        x_col = np.asarray(X[name], dtype=np.float64)

        # Knot info
        interior_knots = spec.fitted_knots
        if interior_knots is None or len(interior_knots) == 0:
            continue

        knot_spacing = np.diff(interior_knots)

        # Support mass: fraction of data near each knot
        n_knots = len(interior_knots)
        support_mass = np.zeros(n_knots)
        for i, kn in enumerate(interior_knots):
            # Count data within half a knot spacing on each side
            if i == 0:
                lo = spec._lo
            else:
                lo = 0.5 * (interior_knots[i - 1] + kn)
            if i == n_knots - 1:
                hi = spec._hi
            else:
                hi = 0.5 * (kn + interior_knots[i + 1])
            support_mass[i] = np.sum((x_col >= lo) & (x_col <= hi)) / len(x_col)

        # Adjacent basis correlation
        B = spec.transform(x_col)
        n_cols = B.shape[1]
        adj_corr = np.zeros(max(n_cols - 1, 0))
        for j in range(n_cols - 1):
            c1, c2 = B[:, j], B[:, j + 1]
            s1, s2 = np.std(c1), np.std(c2)
            if s1 > 1e-12 and s2 > 1e-12:
                adj_corr[j] = float(np.corrcoef(c1, c2)[0, 1])

        # Coefficient energy in penalized directions
        beta = model.result.beta
        groups = [g for g in model._groups if g.feature_name == name]
        beta_combined = np.concatenate([beta[g.sl] for g in groups])
        coef_energy = beta_combined**2

        # Effective rank via singular values of transformed basis
        sv = np.linalg.svd(B, compute_uv=False)
        sv_norm = sv / sv[0] if sv[0] > 1e-12 else sv
        effective_rank = float(np.sum(sv_norm > 1e-4))
        small_sv = sv_norm[sv_norm < 0.01]

        results[name] = SplineRedundancyReport(
            feature_name=name,
            n_knots=n_knots,
            knot_locations=interior_knots,
            knot_spacing=knot_spacing,
            support_mass=support_mass,
            adjacent_basis_corr=adj_corr,
            coef_energy_penalized=coef_energy,
            effective_rank=effective_rank,
            small_singular_values=small_sv,
        )

    return results
