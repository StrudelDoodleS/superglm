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
    ``OrderedCategorical(basis=Spline(...))`` where the underlying variable is
    categorical but a smooth curve is fit through the level midpoints.
    """

    x: NDArray
    log_relativity: NDArray
    relativity: NDArray
    level_x: NDArray | None = None  # numeric x positions of the K *smooth* levels
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
    kind: str  # "spline", "categorical", "numeric", "polynomial", "piecewise"
    active: bool

    # Curve / levels / slope
    x: NDArray | None = None  # grid for spline/polynomial, knots for piecewise
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

    # Piecewise only: covariance of the per-knot log relativities (J+2 square,
    # base row/col identically zero; its diagonal is se_log_relativity**2).
    # Carried so plotting can evaluate the exact off-knot band
    # var f(x) = h1^2 V11 + 2 h1 h2 V12 + h2^2 V22 on a display grid --
    # pointwise SEs at the knots are not enough, because the variance between
    # knots is a quadratic form of BOTH adjacent hats and their covariance.
    knot_covariance: NDArray | None = None

    # Smooth curve for plotting (OrderedCategorical spline mode)
    smooth_curve: SmoothCurve | None = None

    # Free (unpenalised) levels held out of the smooth: parallel to ``levels``.
    # None when the term has no specials, so existing terms are unchanged.
    level_is_special: NDArray[np.bool_] | None = None

    # Monotonicity
    monotone: str | None = None  # "increasing", "decreasing", or None
    monotone_repaired: bool = False

    # CI alpha used
    alpha: float = 0.05

    # --- Fields below are APPENDED.  Add new ones here, never in the middle.
    #
    # ``TermInference`` is public: ``SuperGLM.term_inference()`` returns it and
    # callers construct it.  A dataclass field inserted between existing ones
    # renumbers every positional argument after it, and because all of these
    # are optional with defaults, that is a SILENT reinterpretation rather
    # than a TypeError -- the call still succeeds and every argument from the
    # insertion point on lands one field to the left of where it was written.
    # ``centering_shift`` was first written between ``centering_mode`` and
    # ``edf``, where a caller's positional ``edf=3.0`` became a centering
    # shift of 3.0 and ``edf`` silently became ``None``.  That is the worst
    # possible field to absorb a stray value: it is added to the exported base
    # relativity, so a stale positional caller would have rated every risk in
    # the workbook at ``exp(3.0)`` -- 20x -- of the model's premium.
    # ``test_term_inference_field_order_is_append_only`` pins the order.

    # The constant a reporting centering removed from this term, on the log
    # scale: ``log_relativity`` as reported is the fitted contribution MINUS
    # ``centering_shift``.  Zero for every term the centering left alone --
    # ``centering="native"``, a single-valued ``Numeric``, an
    # ``OrderedCategorical`` (which ``_recenter_term`` never reaches, because
    # it is already anchored on its base level).
    #
    # Scope: this describes ``SuperGLM.term_inference``.  The sibling public
    # surface ``SuperGLM.relativities(centering="mean")`` runs a SEPARATE
    # centering (``_term_model_ops._center_df``) that records no shift and,
    # unlike this one, does shift an ``OrderedCategorical`` spline term.  The
    # two disagree today; see ``test_the_two_mean_centerings_disagree_on_an_
    # ordered_categorical``, which pins the divergence so that whoever wires
    # ``centering_shift`` into the plot-data path finds it deliberately
    # measured rather than inheriting a silently different constant.
    #
    # It is recorded rather than re-derived because the two are not the same
    # number.  A consumer who reconstructs it as ``mean(log_relativity)`` over
    # the values it can see gets zero for a mean-centered term (correct but
    # useless), the whole level mean for an ``OrderedCategorical`` that was
    # never shifted at all, and the EXPANDED level mean for a grouped
    # categorical whose shift was computed on its grouped levels.  Anything
    # that has to add the constant back -- the rating-table export folds it
    # into ``base_relativity`` so the workbook still multiplies out to
    # ``model.predict`` -- must use the constant that was actually subtracted.
    centering_shift: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a tidy DataFrame for plotting or export."""
        # "piecewise" belongs with the x-bearing kinds, not with the numeric
        # fallback: its values are indexed by knot position, so dropping x would
        # leave J+2 log-relativities with nothing saying where they sit.
        if self.kind in ("spline", "polynomial", "piecewise"):
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
