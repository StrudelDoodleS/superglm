"""Spline discretization impact analysis.

Answers the question: "If I bin this spline into N buckets, how do my
predictions and model metrics change?"

This is a read-only analysis tool — no refitting. It takes a fitted model,
discretizes spline contributions analytically, and reports the impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superglm.model import SuperGLM


@dataclass
class DiscretizationResult:
    """Result of discretizing smooth spline curves into rating tables.

    Attributes
    ----------
    tables : dict[str, DataFrame]
        Per-feature rating tables with columns: bin_from, bin_to,
        relativity, log_relativity, n_obs, exposure.
    predictions : NDArray
        Predictions using discretized (binned) curves.
    original_predictions : NDArray
        Original smooth predictions.
    metrics : dict[str, float]
        Comparison metrics between original and discretized predictions.
    """

    tables: dict[str, pd.DataFrame]
    predictions: NDArray
    original_predictions: NDArray
    metrics: dict[str, float]


def _exposure_weighted_quantile_edges(
    x: NDArray, exposure: NDArray, n_bins: int
) -> NDArray:
    """Compute bin edges so each bin has roughly equal total exposure."""
    order = np.argsort(x)
    x_sorted = x[order]
    exp_sorted = exposure[order]
    cum_exp = np.cumsum(exp_sorted)
    total = cum_exp[-1]

    edges = [x_sorted[0]]
    for i in range(1, n_bins):
        target = total * i / n_bins
        idx = np.searchsorted(cum_exp, target, side="right")
        idx = min(idx, len(x_sorted) - 1)
        edges.append(x_sorted[idx])
    edges.append(x_sorted[-1])

    # Deduplicate: if repeated values collapse bins, keep unique edges
    edges = np.unique(edges)
    return edges


def _uniform_edges(x: NDArray, n_bins: int) -> NDArray:
    """Compute equal-width bin edges across the data range."""
    return np.linspace(x.min(), x.max(), n_bins + 1)


def _winsorized_edges(
    x: NDArray, exposure: NDArray, n_bins: int
) -> NDArray:
    """Exposure-quantile binning on [p5, p95] interior, with tail bins."""
    if n_bins < 3:
        # Not enough bins for tail+interior+tail, fall back to exposure quantile
        return _exposure_weighted_quantile_edges(x, exposure, n_bins)

    p5, p95 = np.percentile(x, [5, 95])
    x_min, x_max = x.min(), x.max()

    # If percentiles collapse (very little spread), fall back
    if p5 >= p95:
        return _exposure_weighted_quantile_edges(x, exposure, n_bins)

    # Interior: exposure-quantile on observations within [p5, p95]
    interior_mask = (x >= p5) & (x <= p95)
    n_interior = n_bins - 2
    interior_edges = _exposure_weighted_quantile_edges(
        x[interior_mask], exposure[interior_mask], n_interior
    )

    # Assemble: [x_min, p5, ...interior..., p95, x_max]
    edges = np.concatenate([[x_min], interior_edges, [x_max]])
    edges = np.unique(edges)
    return edges


def _compute_edges(
    x: NDArray, exposure: NDArray, n_bins: int, strategy: str
) -> NDArray:
    """Dispatch to the appropriate binning strategy."""
    if strategy == "exposure_quantile":
        return _exposure_weighted_quantile_edges(x, exposure, n_bins)
    elif strategy == "uniform":
        return _uniform_edges(x, n_bins)
    elif strategy == "winsorized":
        return _winsorized_edges(x, exposure, n_bins)
    else:
        raise ValueError(
            f"Unknown bin_strategy: {strategy!r}. "
            "Use 'exposure_quantile', 'uniform', or 'winsorized'."
        )


def _is_continuous_feature(model: SuperGLM, name: str) -> bool:
    """Check if a feature is a spline or polynomial (has 'x' in reconstruct)."""
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import Spline

    return isinstance(model._specs[name], (Spline, Polynomial))


def discretization_impact(
    model: SuperGLM,
    X: pd.DataFrame,
    y: NDArray,
    exposure: NDArray | None = None,
    *,
    n_bins: int = 100,
    bin_strategy: str = "exposure_quantile",
    features: list[str] | None = None,
) -> DiscretizationResult:
    """Analyse the impact of discretizing smooth spline/polynomial curves.

    For each spline/polynomial feature, the smooth per-observation
    log-relativity is replaced with an exposure-weighted bin average.
    Predictions are recomputed and compared to the originals.

    Parameters
    ----------
    model : SuperGLM
        A fitted SuperGLM model.
    X : DataFrame
        Data used for analysis (typically training data).
    y : NDArray
        Response variable.
    exposure : NDArray, optional
        Exposure weights. Defaults to ones.
    n_bins : int
        Number of bins per feature (default 100).
    bin_strategy : str
        Binning strategy: ``"exposure_quantile"`` (default) places bin
        edges so each bin has equal total exposure; ``"uniform"`` uses
        equal-width bins; ``"winsorized"`` uses exposure-quantile on the
        interior [p5, p95] with dedicated tail bins.
    features : list[str], optional
        Subset of spline/polynomial feature names to discretize.
        None means all spline/polynomial features.

    Returns
    -------
    DiscretizationResult
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    exposure = np.ones(n) if exposure is None else np.asarray(exposure, dtype=np.float64)

    result = model.result  # raises if not fitted
    beta = result.beta
    intercept = result.intercept

    # Determine which features to discretize
    if features is not None:
        for name in features:
            if name not in model._specs:
                raise ValueError(f"Unknown feature: {name}")
            if not _is_continuous_feature(model, name):
                raise ValueError(
                    f"Feature '{name}' is not a spline or polynomial — "
                    "only continuous features can be discretized."
                )
        target_features = features
    else:
        target_features = [
            name for name in model._feature_order
            if _is_continuous_feature(model, name)
        ]

    # Build per-feature design matrices and log-relativities
    # Also reconstruct the full eta for the original predictions
    blocks = []
    for name in model._feature_order:
        spec = model._specs[name]
        x_col = np.asarray(X[name])
        blocks.append(spec.transform(x_col))

    eta_orig = np.hstack(blocks) @ beta + intercept
    original_predictions = np.exp(eta_orig)

    # For each target feature, compute the delta (binned - smooth)
    tables: dict[str, pd.DataFrame] = {}
    total_delta = np.zeros(n)

    for name in target_features:
        spec = model._specs[name]
        g = next(g for g in model._groups if g.name == name)
        x_raw = np.asarray(X[name], dtype=np.float64)

        # Per-observation smooth log-relativity for this feature
        design_block = spec.transform(x_raw)
        log_rel_smooth = design_block @ beta[g.sl]

        # Compute bin edges using the selected strategy
        edges = _compute_edges(x_raw, exposure, n_bins, bin_strategy)
        actual_n_bins = len(edges) - 1

        # Assign observations to bins
        bin_idx = np.digitize(x_raw, edges, right=False)
        # digitize returns 1-based; clip to valid range
        bin_idx = np.clip(bin_idx, 1, actual_n_bins) - 1

        # Compute exposure-weighted mean log-relativity per bin
        bin_log_rel = np.zeros(actual_n_bins)
        bin_exposure = np.zeros(actual_n_bins)
        bin_n_obs = np.zeros(actual_n_bins, dtype=int)

        for b in range(actual_n_bins):
            mask = bin_idx == b
            if np.any(mask):
                w = exposure[mask]
                bin_exposure[b] = w.sum()
                bin_n_obs[b] = mask.sum()
                bin_log_rel[b] = np.average(log_rel_smooth[mask], weights=w)

        # Build rating table
        table_rows = []
        for b in range(actual_n_bins):
            table_rows.append({
                "bin_from": edges[b],
                "bin_to": edges[b + 1],
                "relativity": np.exp(bin_log_rel[b]),
                "log_relativity": bin_log_rel[b],
                "n_obs": bin_n_obs[b],
                "exposure": bin_exposure[b],
            })
        tables[name] = pd.DataFrame(table_rows)

        # Per-observation delta: replace smooth with bin mean
        binned_log_rel = bin_log_rel[bin_idx]
        total_delta += binned_log_rel - log_rel_smooth

    # Discretized predictions
    eta_disc = eta_orig + total_delta
    predictions = np.exp(eta_disc)

    # Compute metrics
    dist = model._distribution
    dev_orig_unit = dist.deviance_unit(y, original_predictions)
    dev_disc_unit = dist.deviance_unit(y, predictions)
    deviance_original = float(np.sum(exposure * dev_orig_unit))
    deviance_discretized = float(np.sum(exposure * dev_disc_unit))
    deviance_change = deviance_discretized - deviance_original
    deviance_change_pct = (
        100.0 * deviance_change / deviance_original if deviance_original > 0 else 0.0
    )

    # Prediction comparison
    safe_orig = np.where(original_predictions > 0, original_predictions, 1e-300)
    abs_pct_change = np.abs(predictions - original_predictions) / safe_orig * 100.0

    metrics = {
        "deviance_original": deviance_original,
        "deviance_discretized": deviance_discretized,
        "deviance_change": deviance_change,
        "deviance_change_pct": deviance_change_pct,
        "max_abs_prediction_change_pct": float(np.max(abs_pct_change)),
        "mean_abs_prediction_change_pct": float(np.mean(abs_pct_change)),
        "prediction_correlation": float(np.corrcoef(original_predictions, predictions)[0, 1]),
    }

    return DiscretizationResult(
        tables=tables,
        predictions=predictions,
        original_predictions=original_predictions,
        metrics=metrics,
    )
