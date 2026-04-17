"""Fold-curve similarity helpers for cross-validation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.plotting.comparison import _build_term_comparison_data


def _weighted_rmse(
    left: NDArray[np.float64], right: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    diff2 = (left - right) ** 2
    return float(np.sqrt(np.average(diff2, weights=weights)))


def _weighted_max_abs_diff(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    del weights
    return float(np.max(np.abs(left - right)))


def _curve_correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if np.allclose(left, left[0]) and np.allclose(right, right[0]):
        return 1.0
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _pairwise_curve_similarity(
    curves: Mapping[str, NDArray[np.float64]],
    weights: NDArray[np.float64],
    *,
    labels: list[str],
) -> dict[str, pd.DataFrame]:
    """Compute pairwise fold similarity matrices for one set of evaluated curves."""
    weights = np.asarray(weights, dtype=np.float64)
    rmse = np.zeros((len(labels), len(labels)), dtype=np.float64)
    max_abs = np.zeros((len(labels), len(labels)), dtype=np.float64)
    corr = np.eye(len(labels), dtype=np.float64)

    for i, left_label in enumerate(labels):
        left = np.asarray(curves[left_label], dtype=np.float64)
        for j, right_label in enumerate(labels):
            right = np.asarray(curves[right_label], dtype=np.float64)
            rmse[i, j] = _weighted_rmse(left, right, weights)
            max_abs[i, j] = _weighted_max_abs_diff(left, right, weights)
            corr[i, j] = _curve_correlation(left, right)

    return {
        "rmse": pd.DataFrame(rmse, index=labels, columns=labels),
        "max_abs_diff": pd.DataFrame(max_abs, index=labels, columns=labels),
        "correlation": pd.DataFrame(corr, index=labels, columns=labels),
    }


def _summarize_against_fold_mean(
    curves: Mapping[str, NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> pd.DataFrame:
    """Summarize each fold curve against the fold-mean curve."""
    labels = list(curves)
    stacked = np.vstack([np.asarray(curves[label], dtype=np.float64) for label in labels])
    mean_curve = np.mean(stacked, axis=0)
    rows = []
    for label in labels:
        curve = np.asarray(curves[label], dtype=np.float64)
        rows.append(
            {
                "fold": label,
                "rmse_to_mean": _weighted_rmse(
                    curve, mean_curve, np.asarray(weights, dtype=np.float64)
                ),
                "max_abs_diff_to_mean": _weighted_max_abs_diff(
                    curve, mean_curve, np.asarray(weights, dtype=np.float64)
                ),
                "correlation_to_mean": _curve_correlation(curve, mean_curve),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def build_cv_curve_similarity(
    *,
    models: list[Any],
    X: pd.DataFrame,
    sample_weight: NDArray | None = None,
    n_points: int = 200,
) -> dict[str, Any]:
    """Build fold-curve similarity diagnostics for all comparable main effects."""
    labeled_models = {f"fold_{i}": model for i, model in enumerate(models) if model is not None}
    if not labeled_models:
        return {}

    comparison_data = _build_term_comparison_data(
        models=labeled_models,
        terms=None,
        X=X,
        sample_weight=sample_weight,
        n_points=n_points,
    )
    result: dict[str, Any] = {}

    for term in comparison_data["terms"]:
        name = term["name"]
        family = term["family"]
        domain = term["domain"]
        series = term["series"]
        support = term["support"]

        if family == "continuous":
            weights = np.asarray(support["density"], dtype=np.float64)
        else:
            weights = np.asarray(support["density"], dtype=np.float64)

        response_curves = {
            label: np.asarray(entry["response"], dtype=np.float64)
            for label, entry in series.items()
        }
        link_curves = {
            label: np.asarray(entry["link"], dtype=np.float64) for label, entry in series.items()
        }
        labels = list(series.keys())

        result[name] = {
            "family": family,
            "domain": domain,
            "support": support,
            "curves": {
                "response": response_curves,
                "link": link_curves,
            },
            "pairwise": {
                "response": _pairwise_curve_similarity(response_curves, weights, labels=labels),
                "link": _pairwise_curve_similarity(link_curves, weights, labels=labels),
            },
            "vs_mean": {
                "response": _summarize_against_fold_mean(response_curves, weights),
                "link": _summarize_against_fold_mean(link_curves, weights),
            },
        }

    return result
