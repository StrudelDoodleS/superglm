"""Violation measures of a curve, used by the shape-repair tests as oracles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def monotonicity_violation(values: NDArray, direction: str) -> float:
    """Maximum monotonicity violation: max backwards step size."""
    diffs = np.diff(values)
    if direction == "increasing":
        violations = np.maximum(0.0, -diffs)
    else:
        violations = np.maximum(0.0, diffs)
    return float(np.max(violations)) if len(violations) > 0 else 0.0


def curvature_violation(values: NDArray, kind: str) -> float:
    """Return the largest violation of a convexity or concavity requirement."""

    diffs2 = np.diff(values, n=2)
    if kind == "convex":
        bad = np.maximum(0.0, -diffs2)
    elif kind == "concave":
        bad = np.maximum(0.0, diffs2)
    else:
        raise ValueError(f"Unsupported curvature kind: {kind!r}")
    return float(np.max(bad)) if len(bad) else 0.0
