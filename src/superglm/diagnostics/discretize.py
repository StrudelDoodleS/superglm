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

from superglm._frame import FrameLike, as_eager_frame

if TYPE_CHECKING:
    from superglm.model import SuperGLM


@dataclass
class DiscretizationResult:
    """Result of discretizing smooth spline curves into rating tables.

    Attributes
    ----------
    tables : dict[str, DataFrame]
        Per-feature rating tables with columns: bin_from, bin_to,
        relativity, log_relativity, n_obs, sample_weight. ``n_obs`` is always
        the physical row count. ``sample_weight`` is the supplied weight total
        in the bin (frequency mass for non-Tweedie; EDM prior-weight mass for
        Tweedie) and is reported for display rather than reinterpreted as a
        Tweedie replication count.
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


def _validated_discretization_weights(
    model,
    sample_weight,
    n_rows: int,
) -> tuple[NDArray, NDArray]:
    """Return likelihood/display weights and family-appropriate geometry mass."""
    from superglm.distributions import Tweedie

    if sample_weight is None:
        weights = np.ones(n_rows, dtype=np.float64)
    elif isinstance(model._distribution, Tweedie):
        from superglm._utils import _validate_strict_prior_weights

        weights = _validate_strict_prior_weights(sample_weight, n_rows)
    else:
        from superglm.model.input_validation import _finite_vector

        weights = _finite_vector("sample_weight", sample_weight, n_rows)
        if np.any(weights < 0.0):
            raise ValueError("sample_weight must be nonnegative")
        if not np.any(weights > 0.0):
            raise ValueError("sample_weight must not be all zero")

    weight_total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(weight_total):
        raise ValueError("sample_weight must have a finite sum")

    # Non-Tweedie weights are frequency mass and therefore shape the same
    # support as literal replicated rows. Tweedie weights are prior precision:
    # fit-time spline/discrete geometry remains a function of physical rows.
    geometry_weight = (
        np.ones(n_rows, dtype=np.float64) if isinstance(model._distribution, Tweedie) else weights
    )
    return weights, geometry_weight


def _weighted_quantile_edges(x: NDArray, sample_weight: NDArray, n_bins: int) -> NDArray:
    """Compute edges with roughly equal geometry mass in each bin."""
    positive = sample_weight > 0.0
    x = np.asarray(x[positive], dtype=np.float64)
    sample_weight = np.asarray(sample_weight[positive], dtype=np.float64)
    order = np.argsort(x)
    x_sorted = x[order]
    weight_sorted = sample_weight[order]
    cumulative_weight = np.cumsum(weight_sorted)
    total = cumulative_weight[-1]

    if x_sorted[0] == x_sorted[-1]:
        return np.array([x_sorted[0], x_sorted[0]], dtype=np.float64)

    edges = [x_sorted[0]]
    for i in range(1, n_bins):
        target = total * i / n_bins
        idx = np.searchsorted(cumulative_weight, target, side="right")
        idx = min(idx, len(x_sorted) - 1)
        edges.append(x_sorted[idx])
    edges.append(x_sorted[-1])

    # Deduplicate: if repeated values collapse bins, keep unique edges
    edges = np.unique(edges)
    if len(edges) == 1:
        return np.repeat(edges, 2)
    return edges


def _uniform_edges(x: NDArray, n_bins: int) -> NDArray:
    """Compute equal-width bin edges across the data range."""
    return np.linspace(x.min(), x.max(), n_bins + 1)


def _weighted_percentiles(
    x: NDArray,
    sample_weight: NDArray,
    quantiles: NDArray,
) -> NDArray:
    """Percentiles matching NumPy on literal integer row replication."""
    positive = sample_weight > 0.0
    x_active = np.asarray(x[positive], dtype=np.float64)
    weight_active = np.asarray(sample_weight[positive], dtype=np.float64)
    order = np.argsort(x_active)
    x_sorted = x_active[order]
    cumulative_weight = np.cumsum(weight_active[order])
    total = float(cumulative_weight[-1])
    if total <= 1.0:
        indices = np.searchsorted(
            cumulative_weight,
            total * np.asarray(quantiles, dtype=np.float64),
            side="right",
        )
        return x_sorted[np.clip(indices, 0, len(x_sorted) - 1)]

    positions = (total - 1.0) * np.asarray(quantiles, dtype=np.float64)
    lower_positions = np.floor(positions)
    upper_positions = np.ceil(positions)
    lower_indices = np.searchsorted(cumulative_weight, lower_positions, side="right")
    upper_indices = np.searchsorted(cumulative_weight, upper_positions, side="right")
    lower = x_sorted[np.clip(lower_indices, 0, len(x_sorted) - 1)]
    upper = x_sorted[np.clip(upper_indices, 0, len(x_sorted) - 1)]
    return lower + (positions - lower_positions) * (upper - lower)


def _winsorized_edges(x: NDArray, sample_weight: NDArray, n_bins: int) -> NDArray:
    """Geometry-quantile binning on the [p5, p95] interior, with tail bins."""
    if n_bins < 3:
        # Not enough bins for tail+interior+tail, fall back to weight quantiles.
        return _weighted_quantile_edges(x, sample_weight, n_bins)

    positive = sample_weight > 0.0
    x_geometry = x[positive]
    p5, p95 = _weighted_percentiles(
        x,
        sample_weight,
        np.array([0.05, 0.95], dtype=np.float64),
    )
    x_min, x_max = x_geometry.min(), x_geometry.max()

    # If percentiles collapse (very little spread), fall back
    if p5 >= p95:
        return _weighted_quantile_edges(x, sample_weight, n_bins)

    # Interior: geometry-weight quantiles on observations within [p5, p95].
    interior_mask = (x >= p5) & (x <= p95)
    if not np.any(sample_weight[interior_mask] > 0.0):
        return _weighted_quantile_edges(x, sample_weight, n_bins)
    n_interior = n_bins - 2
    interior_edges = _weighted_quantile_edges(
        x[interior_mask], sample_weight[interior_mask], n_interior
    )

    # Assemble: [x_min, p5, ...interior..., p95, x_max]
    edges = np.concatenate([[x_min], interior_edges, [x_max]])
    edges = np.unique(edges)
    return edges


def _compute_edges(x: NDArray, sample_weight: NDArray, n_bins: int, strategy: str) -> NDArray:
    """Dispatch to the appropriate binning strategy."""
    if strategy == "exposure_quantile":
        return _weighted_quantile_edges(x, sample_weight, n_bins)
    elif strategy == "uniform":
        return _uniform_edges(x[sample_weight > 0.0], n_bins)
    elif strategy == "winsorized":
        return _winsorized_edges(x, sample_weight, n_bins)
    else:
        raise ValueError(
            f"Unknown bin_strategy: {strategy!r}. "
            "Use 'exposure_quantile', 'uniform', or 'winsorized'."
        )


def _is_continuous_feature(model: SuperGLM, name: str) -> bool:
    """Check if a feature is a spline or polynomial (has 'x' in reconstruct)."""
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    return isinstance(model._specs[name], _SplineBase | Polynomial)


def _weighted_correlation(x: NDArray, y: NDArray, weights: NDArray) -> float:
    """Correlation under frequency mass or unit physical-row mass."""
    positive = weights > 0.0
    x_active = np.asarray(x[positive], dtype=np.float64)
    y_active = np.asarray(y[positive], dtype=np.float64)
    mass = np.asarray(weights[positive], dtype=np.float64)
    mass /= np.sum(mass, dtype=np.float64)
    x_centered = x_active - float(np.sum(mass * x_active))
    y_centered = y_active - float(np.sum(mass * y_active))
    variance_x = float(np.sum(mass * x_centered**2))
    variance_y = float(np.sum(mass * y_centered**2))
    if variance_x <= 0.0 or variance_y <= 0.0:
        return float("nan")
    correlation = float(np.sum(mass * x_centered * y_centered) / np.sqrt(variance_x * variance_y))
    return float(np.clip(correlation, -1.0, 1.0))


def discretization_impact(
    model: SuperGLM,
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    offset: NDArray | None = None,
    n_bins: int = 100,
    bin_strategy: str = "exposure_quantile",
    features: list[str] | None = None,
) -> DiscretizationResult:
    """Analyse the impact of discretizing smooth spline/polynomial curves.

    For each spline/polynomial feature, the smooth per-observation
    log-relativity is replaced with a family-appropriate bin average.
    Predictions are recomputed and compared to the originals.

    For non-Tweedie families, ``sample_weight`` is case/frequency mass:
    bin geometry, bin averages, mean prediction change, and prediction
    correlation match literal integer row replication. Zero-frequency rows
    retain predictions and physical ``n_obs`` entries but cannot change bin
    geometry or summary metrics. For Tweedie, weights are finite, strictly
    positive EDM prior weights. They weight deviance and the displayed
    ``sample_weight`` totals, while bin geometry, bin averages, and pure
    prediction-comparison summaries use physical rows.

    Parameters
    ----------
    model : SuperGLM
        A fitted SuperGLM model.
    X : pandas or eager Polars DataFrame
        Data used for analysis (typically training data).
    y : NDArray
        Response variable.
    sample_weight : NDArray, optional
        Nonnegative case/frequency weights for non-Tweedie models. For
        Tweedie models, finite, strictly positive EDM prior weights. Defaults
        to ones.
    offset : NDArray, optional
        Link-scale offset aligned to ``X``. Used when comparing original and
        discretized predictions for offset-fitted models.
    n_bins : int
        Number of bins per feature (default 100).
    bin_strategy : str
        Binning strategy: ``"exposure_quantile"`` (the retained public name)
        places edges at equal geometry-weight mass; ``"uniform"`` uses
        equal-width bins; ``"winsorized"`` uses geometry-weight quantiles on
        the interior [p5, p95] with dedicated tail bins. Geometry weight means
        frequency mass for non-Tweedie models and unit physical-row mass for
        Tweedie.
    features : list[str], optional
        Subset of spline/polynomial feature names to discretize.
        None means all spline/polynomial features.

    Returns
    -------
    DiscretizationResult
    """
    frame = as_eager_frame(X)
    result = model.result  # raises if not fitted
    n = len(frame)
    if n == 0:
        raise ValueError("X and y must be non-empty")

    from superglm.distributions import validate_response
    from superglm.model.input_validation import _finite_vector

    y = _finite_vector("y", y, n, require_nonempty=True)
    validate_response(y, model._distribution)
    evaluation_weight, geometry_weight = _validated_discretization_weights(
        model,
        sample_weight,
        n,
    )
    if offset is not None:
        offset = _finite_vector("offset", offset, n)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int | np.integer) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    n_bins = int(n_bins)

    beta = result.beta
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model import base

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
            name for name in model._feature_order if _is_continuous_feature(model, name)
        ]

    from superglm.model.input_validation import validate_x_columns

    validate_x_columns(frame, target_features)
    eta_orig = base.predict_eta_exact(model, frame, offset=offset)
    original_predictions = clip_mu(model._link.inverse(eta_orig), model._distribution)
    plan = base._prediction_plan(model)
    terms_by_name = {term["name"]: term for term in plan["features"]}

    # For each target feature, compute the delta (binned - smooth)
    tables: dict[str, pd.DataFrame] = {}
    total_delta = np.zeros(n)

    for name in target_features:
        x_raw = frame.column_array(name, dtype=np.float64)

        # Per-observation smooth log-relativity for this feature
        term = terms_by_name.get(name)
        if term is None:
            raise RuntimeError(f"prediction plan does not define fitted term {name!r}")
        beta_feature = beta[np.asarray(term["beta_idx"], dtype=np.intp)]
        log_rel_smooth = np.asarray(
            base._score_prediction_term_local_exact(term, frame, beta_feature),
            dtype=np.float64,
        ).ravel()

        # Compute bin edges using the selected strategy
        edges = _compute_edges(x_raw, geometry_weight, n_bins, bin_strategy)
        actual_n_bins = len(edges) - 1

        # Assign observations to bins
        bin_idx = np.digitize(x_raw, edges, right=False)
        # digitize returns 1-based; clip to valid range
        bin_idx = np.clip(bin_idx, 1, actual_n_bins) - 1

        # Frequency-weighted for non-Tweedie; physical-row mean for Tweedie.
        bin_log_rel = np.zeros(actual_n_bins)
        bin_weight = np.zeros(actual_n_bins)
        bin_n_obs = np.zeros(actual_n_bins, dtype=int)

        for b in range(actual_n_bins):
            mask = bin_idx == b
            if np.any(mask):
                bin_n_obs[b] = mask.sum()
                bin_weight[b] = float(np.sum(evaluation_weight[mask], dtype=np.float64))
                geometry_mass = geometry_weight[mask]
                if np.any(geometry_mass > 0.0):
                    bin_log_rel[b] = np.average(
                        log_rel_smooth[mask],
                        weights=geometry_mass,
                    )

        # Build rating table
        table_rows = []
        for b in range(actual_n_bins):
            table_rows.append(
                {
                    "bin_from": edges[b],
                    "bin_to": edges[b + 1],
                    "relativity": np.exp(bin_log_rel[b]),
                    "log_relativity": bin_log_rel[b],
                    "n_obs": bin_n_obs[b],
                    "sample_weight": bin_weight[b],
                }
            )
        tables[name] = pd.DataFrame(table_rows)

        # Per-observation delta: replace smooth with bin mean
        binned_log_rel = bin_log_rel[bin_idx]
        total_delta += binned_log_rel - log_rel_smooth

    # Discretized predictions
    eta_disc = stabilize_eta(eta_orig + total_delta, model._link)
    predictions = clip_mu(model._link.inverse(eta_disc), model._distribution)

    # Compute metrics
    dist = model._distribution
    dev_orig_unit = dist.deviance_unit(y, original_predictions)
    dev_disc_unit = dist.deviance_unit(y, predictions)
    deviance_original = float(np.sum(evaluation_weight * dev_orig_unit))
    deviance_discretized = float(np.sum(evaluation_weight * dev_disc_unit))
    deviance_change = deviance_discretized - deviance_original
    deviance_change_pct = (
        100.0 * deviance_change / deviance_original if deviance_original > 0 else 0.0
    )

    # Prediction-only summaries follow geometry semantics: literal frequency
    # rows for non-Tweedie, physical rows for Tweedie prior weights.
    safe_orig = np.maximum(np.abs(original_predictions), 1e-300)
    abs_pct_change = np.abs(predictions - original_predictions) / safe_orig * 100.0
    summary_active = geometry_weight > 0.0
    summary_weight = geometry_weight[summary_active]
    mean_abs_prediction_change_pct = float(
        np.average(abs_pct_change[summary_active], weights=summary_weight)
    )

    metrics = {
        "deviance_original": deviance_original,
        "deviance_discretized": deviance_discretized,
        "deviance_change": deviance_change,
        "deviance_change_pct": deviance_change_pct,
        "max_abs_prediction_change_pct": float(np.max(abs_pct_change[summary_active])),
        "mean_abs_prediction_change_pct": mean_abs_prediction_change_pct,
        "prediction_correlation": _weighted_correlation(
            original_predictions,
            predictions,
            geometry_weight,
        ),
    }

    return DiscretizationResult(
        tables=tables,
        predictions=predictions,
        original_predictions=original_predictions,
        metrics=metrics,
    )
