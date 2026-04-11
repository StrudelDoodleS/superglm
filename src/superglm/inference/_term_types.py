"""Internal term-inference types and small shared utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

# Overflow guard for exp(log_rel ± z * se).  exp(500) ≈ 1.4e217, safely
# within float64 range; quasi-separated levels get large but finite CIs
# instead of inf/nan.
_MAX_LOG_REL = 500.0


def _safe_exp(x: np.ndarray | float) -> np.ndarray | float:
    """Exponentiate with overflow protection for CI bounds."""
    return cast(np.ndarray | float, np.exp(np.clip(x, -_MAX_LOG_REL, _MAX_LOG_REL)))


@dataclass(frozen=True)
class SplineMetadata:
    """Knot and basis metadata for a spline term."""

    kind: str  # e.g. "PSpline", "NaturalSpline", "CubicRegressionSpline"
    knot_strategy: str  # "uniform", "quantile", "quantile_tempered", "explicit"
    interior_knots: NDArray
    boundary: tuple[float, float]
    n_basis: int
    degree: int
    extrapolation: str  # "clip", "extend", "error"
    knot_alpha: float | None = None  # only for "quantile_tempered"


@dataclass(frozen=True)
class SmoothCurve:
    """Continuous fitted curve for plotting (not for rating tables).

    Attached to ``TermInference.smooth_curve`` for features like
    ``OrderedCategorical(basis="spline")`` where the underlying variable is
    categorical but a smooth curve is fit through the level midpoints.
    """

    x: NDArray
    log_relativity: NDArray
    relativity: NDArray
    level_x: NDArray | None = None  # numeric x positions of the K levels
    se_log_relativity: NDArray | None = None
    ci_lower: NDArray | None = None
    ci_upper: NDArray | None = None


@dataclass(frozen=True)
class TermInference:
    """Per-term inference result.

    Holds the fitted curve (or levels/slope), uncertainty measures, and
    metadata for a single model term.  Returned by
    ``SuperGLM.term_inference()``.
    """

    # Identity
    name: str
    kind: str  # "spline", "categorical", "numeric", "polynomial"
    active: bool

    # Curve / levels / slope
    x: NDArray | None = None  # grid for spline/polynomial, None otherwise
    levels: list[str] | None = None  # for categorical
    log_relativity: NDArray | None = None
    relativity: NDArray | None = None

    # Uncertainty (pointwise)
    se_log_relativity: NDArray | None = None
    ci_lower: NDArray | None = None  # pointwise lower
    ci_upper: NDArray | None = None  # pointwise upper

    # Uncertainty (simultaneous) — only when simultaneous=True
    ci_lower_simultaneous: NDArray | None = None
    ci_upper_simultaneous: NDArray | None = None
    critical_value_simultaneous: float | None = None

    # Centering
    absorbs_intercept: bool = True
    centering_mode: str = "training_mean_zero_unweighted"

    # Smoothness / penalty
    edf: float | None = None
    smoothing_lambda: float | dict[str, float] | None = None

    # Spline-specific metadata
    spline: SplineMetadata | None = None

    # Smooth curve for plotting (OrderedCategorical spline mode)
    smooth_curve: SmoothCurve | None = None

    # Monotonicity
    monotone: str | None = None  # "increasing", "decreasing", or None
    monotone_repaired: bool = False

    # CI alpha used
    alpha: float = 0.05

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a tidy DataFrame for plotting or export."""
        if self.kind in ("spline", "polynomial"):
            d: dict[str, Any] = {
                "x": self.x,
                "log_relativity": self.log_relativity,
                "relativity": self.relativity,
            }
            if self.se_log_relativity is not None:
                d["se_log_relativity"] = self.se_log_relativity
            if self.ci_lower is not None:
                d["ci_lower"] = self.ci_lower
                d["ci_upper"] = self.ci_upper
            if self.ci_lower_simultaneous is not None:
                d["ci_lower_simultaneous"] = self.ci_lower_simultaneous
                d["ci_upper_simultaneous"] = self.ci_upper_simultaneous
            return pd.DataFrame(d)

        elif self.kind == "categorical":
            d = {
                "level": self.levels,
                "log_relativity": self.log_relativity,
                "relativity": self.relativity,
            }
            if self.se_log_relativity is not None:
                d["se_log_relativity"] = self.se_log_relativity
            if self.ci_lower is not None:
                d["ci_lower"] = self.ci_lower
                d["ci_upper"] = self.ci_upper
            return pd.DataFrame(d)

        else:
            # numeric
            d = {
                "label": ["per_unit"],
                "log_relativity": self.log_relativity,
                "relativity": self.relativity,
            }
            if self.se_log_relativity is not None:
                d["se_log_relativity"] = self.se_log_relativity
            if self.ci_lower is not None:
                d["ci_lower"] = self.ci_lower
                d["ci_upper"] = self.ci_upper
            return pd.DataFrame(d)


@dataclass(frozen=True)
class InteractionInference:
    """Per-interaction inference result (lighter than TermInference)."""

    name: str
    kind: str  # "spline_categorical", "categorical", "numeric_categorical", etc.
    active: bool

    # For spline×categorical: per-level curves
    x: NDArray | None = None
    levels: list[str] | None = None
    per_level: dict[str, dict[str, NDArray]] | None = None

    # For categorical×categorical: per-pair
    pairs: list[tuple[str, str]] | None = None
    log_relativity: NDArray | dict[str, float] | None = None
    relativity: NDArray | dict[str, float] | None = None

    # For numeric×categorical: per-level slopes
    relativities_per_unit: dict[str, float] | None = None
    log_relativities_per_unit: dict[str, float] | None = None

    # For numeric×numeric: single product coefficient
    relativity_per_unit_unit: float | None = None
    coef: float | None = None


__all__ = [
    "_MAX_LOG_REL",
    "_safe_exp",
    "InteractionInference",
    "SmoothCurve",
    "SplineMetadata",
    "TermInference",
]
