"""Local helpers for the SCOP lambda-sensitivity benchmark."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_LAMBDA_GRID_FACTORS = np.array(
    [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    dtype=np.float64,
)


def build_lambda_grid(baseline_lambda: float) -> NDArray[np.float64]:
    """Build a log-symmetric lambda sweep around a fitted baseline."""
    baseline = float(baseline_lambda)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("baseline_lambda must be positive and finite")
    return baseline * _LAMBDA_GRID_FACTORS


def _as_1d_float_array(name: str, values: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integration_weights(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.size == 1:
        return np.ones(1, dtype=np.float64)

    spacing = np.diff(x)
    if np.any(spacing <= 0.0):
        raise ValueError("x must be strictly increasing")

    weights = np.empty_like(x)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights / np.sum(weights)


def curve_similarity_metrics(
    x: NDArray[np.float64] | list[float],
    reference_curve: NDArray[np.float64] | list[float],
    other_curve: NDArray[np.float64] | list[float],
) -> dict[str, float]:
    """Compare two evaluated curves on a shared x-grid."""
    x_values = _as_1d_float_array("x", x)
    reference = _as_1d_float_array("reference_curve", reference_curve)
    other = _as_1d_float_array("other_curve", other_curve)

    if x_values.shape != reference.shape or reference.shape != other.shape:
        raise ValueError("x, reference_curve, and other_curve must have the same shape")

    weights = _integration_weights(x_values)
    diff = other - reference
    mse = float(np.average(diff**2, weights=weights))
    max_abs_diff = float(np.max(np.abs(diff)))

    if mse == 0.0:
        r2 = 1.0
    else:
        centered_reference = reference - np.average(reference, weights=weights)
        baseline_mse = float(np.average(centered_reference**2, weights=weights))
        r2 = 0.0 if baseline_mse <= np.finfo(np.float64).eps else 1.0 - mse / baseline_mse

    return {
        "rmse": float(np.sqrt(mse)),
        "max_abs_diff": max_abs_diff,
        "r2": float(r2),
    }
