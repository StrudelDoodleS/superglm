"""Actuarial validation toolkit for model comparison and calibration assessment.

Provides lift charts, double lift charts, Lorenz curves with Gini coefficients,
and loss ratio charts following CAS RPM 2016 methodology.

All functions accept raw numpy arrays and are usable with any model framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from superglm._utils import _default_weights, _ensure_array

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass(frozen=True)
class LiftChartResult:
    """Result from :func:`lift_chart`.

    Attributes
    ----------
    bins : pd.DataFrame
        One row per quantile bin with columns: ``bin``, ``exposure_share``,
        ``observed``, ``predicted``, ``obs_pred_ratio``.
    figure : matplotlib.figure.Figure or None
        The generated figure, or ``None`` if an external ``ax`` was provided.
    """

    bins: pd.DataFrame
    figure: Figure | None


@dataclass(frozen=True)
class DoubleLiftChartResult:
    """Result from :func:`double_lift_chart`.

    Attributes
    ----------
    bins : pd.DataFrame
        One row per quantile bin with columns: ``bin``, ``n_rows``,
        ``exposure_sum``, ``exposure_share``, ``target_sum``,
        ``actual_avg``, ``model_avg``, ``current_avg``,
        ``actual_index``, ``model_index``, ``current_index``,
        ``sort_score_min``, ``sort_score_median``, ``sort_score_max``.
    figure : matplotlib.figure.Figure or None
        The generated figure, or ``None`` if an external ``ax`` was provided.
    """

    bins: pd.DataFrame
    figure: Figure | None


@dataclass(frozen=True)
class LorenzCurveResult:
    """Result from :func:`lorenz_curve`.

    Attributes
    ----------
    curve : pd.DataFrame
        Lorenz curve data with columns: ``cum_exposure_share``,
        ``cum_loss_share_ordered``, ``cum_loss_share_model``,
        ``cum_loss_share_perfect``.
    gini_model : float
        Gini coefficient for the model ordering.
    gini_perfect : float
        Gini coefficient for perfect-foresight ordering.
    gini_ratio : float
        Normalised Gini: ``gini_model / gini_perfect``.
    figure : object or None
        The generated matplotlib or Plotly figure, or ``None`` if an external
        matplotlib ``ax`` was provided.
    """

    curve: pd.DataFrame
    gini_model: float
    gini_perfect: float
    gini_ratio: float
    figure: Any | None


@dataclass(frozen=True)
class LossRatioChartResult:
    """Result from :func:`loss_ratio_chart`.

    Attributes
    ----------
    bins : pd.DataFrame
        One row per quantile bin with columns: ``bin``, ``exposure_share``,
        ``observed``, ``predicted``.
    figure : matplotlib.figure.Figure or None
        The generated figure, or ``None`` if an external ``ax`` was provided.
    """

    bins: pd.DataFrame
    figure: Figure | None


# ── Private helpers ──────────────────────────────────────────────


_LONGDOUBLE_EXTENDS_FLOAT64 = (
    np.finfo(np.longdouble).max > np.finfo(np.float64).max
    and np.finfo(np.longdouble).tiny < np.finfo(np.float64).tiny
)


def _require_extended_range(name: str, *values: NDArray) -> None:
    """Reject arithmetic that float64-only ``longdouble`` cannot represent."""
    if _LONGDOUBLE_EXTENDS_FLOAT64:
        return
    for value in values:
        magnitudes = np.abs(np.asarray(value, dtype=np.float64))
        exponents = np.frexp(magnitudes[magnitudes != 0.0])[1]
        # Dividing by an array's largest scale can retain at most the float64
        # subnormal exponent range. Check each operand independently: different
        # physical units across values and weights are not themselves unsafe.
        if exponents.size and int(np.max(exponents) - np.min(exponents)) >= 1074:
            raise ValueError(f"{name} requires an extended floating-point range on this platform")


def _validated_vector(name: str, value, n_rows: int | None = None) -> NDArray:
    """Return one finite numeric public-chart vector."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if n_rows is None and raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if getattr(raw.dtype, "kind", None) in {"M", "m"}:
        raise ValueError(f"{name} must contain only real numeric values")
    if n_rows is not None and len(raw) != n_rows:
        raise ValueError(f"{name} must have length {n_rows}, got {len(raw)}")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only real numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validated_n_bins(n_bins) -> int:
    """Validate the shared quantile-bin boundary."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int | np.integer) or n_bins <= 0:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    return int(n_bins)


def _validated_chart_inputs(
    y_obs,
    *,
    sample_weight=None,
    optional_vectors: tuple[str, ...] = (),
    **vectors,
) -> tuple[NDArray, dict[str, NDArray], NDArray]:
    """Validate and remove zero-effective-weight rows for every public chart."""
    observed = _validated_vector("y_obs", y_obs)
    n_rows = len(observed)
    normalized = {
        name: _validated_vector(name, value, n_rows)
        for name, value in vectors.items()
        if value is not None or name not in optional_vectors
    }
    weights = (
        np.ones(n_rows, dtype=np.float64)
        if sample_weight is None
        else _validated_vector("sample_weight", sample_weight, n_rows)
    )
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be nonnegative")
    if not np.any(weights > 0.0):
        raise ValueError("sample_weight must not be all zero")

    exposure = normalized.get("exposure")
    if exposure is not None and np.any(exposure < 0.0):
        raise ValueError("exposure must be nonnegative")
    with np.errstate(over="ignore", invalid="ignore"):
        effective_weight = weights if exposure is None else weights * exposure
    if not np.all(np.isfinite(effective_weight)):
        raise ValueError("sample_weight * exposure must contain only finite values")
    positive = effective_weight > 0.0
    if not np.any(positive):
        raise ValueError("sample_weight * exposure must not be all zero")
    try:
        total_effective_weight = math.fsum(effective_weight[positive].tolist())
    except OverflowError:
        total_effective_weight = float("inf")
    if not np.isfinite(total_effective_weight):
        source = "sample_weight * exposure" if exposure is not None else "sample_weight"
        raise ValueError(f"{source} must have a finite total")
    if not np.all(positive):
        observed = observed[positive]
        normalized = {name: values[positive] for name, values in normalized.items()}
        weights = weights[positive]

    return observed, normalized, weights


def _weighted_mean(values: NDArray, weights: NDArray, name: str) -> float:
    """Return a finite weighted mean without overflowing intermediate products."""
    _require_extended_range(f"{name} weighted mean", values, weights)
    if not _LONGDOUBLE_EXTENDS_FLOAT64:
        float_values = np.asarray(values, dtype=np.float64)
        scale = np.max(np.abs(float_values))
        if scale == 0.0:
            return 0.0
        normalized_values = float_values / scale
        float_weights = np.asarray(weights, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            products = float_weights * normalized_values
        if np.any(~np.isfinite(products)) or np.any(
            (products == 0.0) & (float_weights != 0.0) & (normalized_values != 0.0)
        ):
            raise ValueError(
                f"{name} weighted mean requires an extended floating-point range on this platform"
            )
        total_weight = math.fsum(float_weights.tolist())
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            raise ValueError(f"{name} weights must have a finite positive total")
        mean_scaled = math.fsum(products.tolist()) / total_weight
        mean_scaled = float(
            np.clip(mean_scaled, np.min(normalized_values), np.max(normalized_values))
        )
        result = float(scale * mean_scaled)
        if not np.isfinite(result):
            raise ValueError(f"{name} weighted mean must be finite")
        return result
    extended_weights = np.asarray(weights, dtype=np.longdouble)
    total_weight = np.sum(extended_weights, dtype=np.longdouble)
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        raise ValueError(f"{name} weights must have a finite positive total")
    extended_values = np.asarray(values, dtype=np.longdouble)
    scale = np.max(np.abs(extended_values))
    if scale == 0.0:
        return 0.0
    normalized_values = extended_values / scale
    mean_scaled = np.sum(
        extended_weights * normalized_values,
        dtype=np.longdouble,
    )
    mean_scaled /= total_weight
    # A weighted mean must lie in the convex hull.  Rounding the normalized
    # shares can otherwise produce 1 + 1 ulp and overflow when rescaled by the
    # largest finite float.
    mean_scaled = np.clip(
        mean_scaled,
        np.min(normalized_values),
        np.max(normalized_values),
    )
    result = float(scale * mean_scaled)
    if not np.isfinite(result):
        raise ValueError(f"{name} weighted mean must be finite")
    return float(result)


def _weighted_total(values: NDArray, weights: NDArray, name: str) -> float:
    """Return a finite weighted total, rejecting a mathematically overflowing result."""
    _require_extended_range(f"{name} weighted total", values, weights)
    if not _LONGDOUBLE_EXTENDS_FLOAT64:
        float_values = np.asarray(values, dtype=np.float64)
        float_weights = np.asarray(weights, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            products = float_values * float_weights
        if np.any(~np.isfinite(products)) or np.any(
            (products == 0.0) & (float_values != 0.0) & (float_weights != 0.0)
        ):
            raise ValueError(
                f"{name} weighted total requires an extended floating-point range on this platform"
            )
        try:
            float_result = math.fsum(products.tolist())
        except OverflowError:
            float_result = float("inf")
        if not np.isfinite(float_result):
            raise ValueError(f"{name} weighted total must be finite")
        return float_result
    extended_result = np.sum(
        np.asarray(values, dtype=np.longdouble) * np.asarray(weights, dtype=np.longdouble),
        dtype=np.longdouble,
    )
    if not np.isfinite(extended_result) or abs(extended_result) > np.finfo(np.float64).max:
        raise ValueError(f"{name} weighted total must be finite")
    return float(extended_result)


def _finite_ratio(numerator: float, denominator: float, name: str) -> float:
    """Divide two finite scalars, retaining the established zero-denominator NaN."""
    if denominator == 0.0:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = float(np.float64(numerator) / np.float64(denominator))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quantile_bins(sort_values: NDArray, weights: NDArray, n_bins: int) -> NDArray:
    """Assign observations to equal-weight quantile bins.

    Returns an integer array of bin indices (0-based).
    """
    n = len(sort_values)
    order = np.argsort(sort_values, kind="stable")
    cum_w = np.cumsum(weights[order])
    total_w = cum_w[-1]
    if total_w <= 0:
        return np.zeros(n, dtype=int)
    bin_edges = np.linspace(0, total_w, n_bins + 1)
    # Assign each observation to the appropriate bin
    bins = np.searchsorted(bin_edges[1:], cum_w, side="left")
    bins = np.clip(bins, 0, n_bins - 1)
    # Map back to original order
    result = np.empty(n, dtype=int)
    result[order] = bins
    return result


def _compensated_cumsum(values: NDArray) -> NDArray:
    """Return Neumaier-compensated float64 cumulative sums."""
    result = np.empty(len(values), dtype=np.float64)
    total = 0.0
    correction = 0.0
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        updated = total + float(value)
        if abs(total) >= abs(value):
            correction += (total - updated) + float(value)
        else:
            correction += (float(value) - updated) + total
        total = updated
        result[index] = total + correction
    return result


def _float64_block_sums(values: NDArray, block_starts: NDArray) -> NDArray:
    """Sum consecutive tie blocks with compensated scalar summation."""
    array = np.asarray(values, dtype=np.float64)
    ends = np.concatenate([block_starts[1:], [len(array)]])
    return np.asarray(
        [
            math.fsum(array[int(start) : int(end)].tolist())
            for start, end in zip(block_starts, ends, strict=True)
        ],
        dtype=np.float64,
    )


def _lorenz_cumulative_by_score(
    scores: NDArray,
    exposures: NDArray,
    losses: NDArray,
    *,
    total_exp: float,
    total_loss: float,
) -> tuple[NDArray, NDArray]:
    """Lorenz cumulative shares after collapsing tied scores into one block."""
    order = np.argsort(scores, kind="stable")
    scores_sorted = scores[order]
    exp_sorted = exposures[order]
    loss_sorted = losses[order]

    _, block_starts = np.unique(scores_sorted, return_index=True)
    if _LONGDOUBLE_EXTENDS_FLOAT64:
        exp_blocks = np.add.reduceat(exp_sorted, block_starts)
        loss_blocks = np.add.reduceat(loss_sorted, block_starts)
        cum_exp = np.cumsum(exp_blocks, dtype=np.longdouble) / np.longdouble(total_exp)
        cum_loss = np.cumsum(loss_blocks, dtype=np.longdouble) / np.longdouble(total_loss)
    else:
        exp_blocks = _float64_block_sums(exp_sorted, block_starts)
        loss_blocks = _float64_block_sums(loss_sorted, block_starts)
        cum_exp = _compensated_cumsum(exp_blocks) / float(total_exp)
        cum_loss = _compensated_cumsum(loss_blocks) / float(total_loss)
    float64_max = np.longdouble(np.finfo(np.float64).max)
    if (
        np.any(~np.isfinite(cum_exp))
        or np.any(~np.isfinite(cum_loss))
        or np.any(np.abs(cum_exp) > float64_max)
        or np.any(np.abs(cum_loss) > float64_max)
    ):
        raise ValueError("Lorenz cumulative shares must be finite")
    return np.asarray(cum_exp, dtype=np.float64), np.asarray(cum_loss, dtype=np.float64)


def _weighted_pair_concordance(scores, weights, centered_target) -> np.longdouble:
    """Sum weighted target differences between ordered, tie-collapsed blocks."""
    order = np.argsort(scores, kind="stable")
    scores_sorted = scores[order]
    arithmetic_dtype = np.longdouble if _LONGDOUBLE_EXTENDS_FLOAT64 else np.float64
    weights = np.asarray(weights, dtype=arithmetic_dtype)
    centered_target = np.asarray(centered_target, dtype=arithmetic_dtype)
    weights_sorted = weights[order]
    target_totals_sorted = (weights * centered_target)[order]

    _, block_starts = np.unique(scores_sorted, return_index=True)
    if _LONGDOUBLE_EXTENDS_FLOAT64:
        weight_blocks = np.add.reduceat(weights_sorted, block_starts)
        target_blocks = np.add.reduceat(target_totals_sorted, block_starts)
    else:
        weight_blocks = _float64_block_sums(weights_sorted, block_starts)
        target_blocks = _float64_block_sums(target_totals_sorted, block_starts)
    # Build exclusive prefixes directly. Subtracting the current block from an
    # inclusive cumulative total loses a small prior block when the next block
    # is much larger, corrupting even a two-row reverse ranking.
    if _LONGDOUBLE_EXTENDS_FLOAT64:
        prior_weights = np.concatenate(
            [
                np.zeros(1, dtype=np.longdouble),
                np.cumsum(weight_blocks[:-1], dtype=np.longdouble),
            ]
        )
        prior_targets = np.concatenate(
            [
                np.zeros(1, dtype=np.longdouble),
                np.cumsum(target_blocks[:-1], dtype=np.longdouble),
            ]
        )
    else:
        prior_weights = np.concatenate([[0.0], _compensated_cumsum(weight_blocks[:-1])])
        prior_targets = np.concatenate([[0.0], _compensated_cumsum(target_blocks[:-1])])
    terms = prior_weights * target_blocks - prior_targets * weight_blocks
    if _LONGDOUBLE_EXTENDS_FLOAT64:
        return np.sum(terms, dtype=np.longdouble)
    return np.longdouble(math.fsum(np.asarray(terms, dtype=np.float64).tolist()))


def _gini_coefficients(y_obs, y_pred, sample_weight=None) -> tuple[float, float, float]:
    """Return stable model, perfect, and normalized tie-collapsed Gini values."""
    y_obs = _ensure_array(y_obs)
    y_pred = _ensure_array(y_pred)
    weights = _default_weights(sample_weight, len(y_obs))
    if y_obs.size == 0:
        return 0.0, 0.0, 0.0
    _require_extended_range("Gini aggregation", y_obs, weights)

    extended_weights = np.asarray(weights, dtype=np.longdouble)
    extended_target = np.asarray(y_obs, dtype=np.longdouble)
    target_scale = np.max(np.abs(extended_target))
    if not np.any(extended_weights > 0.0) or target_scale == 0.0:
        return 0.0, 0.0, 0.0
    total_weight = np.sum(extended_weights, dtype=np.longdouble)
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return 0.0, 0.0, 0.0
    normalized_weights = extended_weights / total_weight
    scaled_target = extended_target / target_scale
    loss_products = normalized_weights * scaled_target
    if not _LONGDOUBLE_EXTENDS_FLOAT64:
        float_weights = np.asarray(normalized_weights, dtype=np.float64)
        float_target = np.asarray(scaled_target, dtype=np.float64)
        with np.errstate(under="ignore", invalid="ignore"):
            float_loss_products = float_weights * float_target
        if np.any((float_loss_products == 0.0) & (float_weights != 0.0) & (float_target != 0.0)):
            raise ValueError(
                "Gini aggregation requires an extended floating-point range on this platform"
            )
        total_loss = np.longdouble(math.fsum(float_loss_products.tolist()))
    else:
        total_loss = np.sum(loss_products, dtype=np.longdouble)
    if total_loss <= 0.0 or not np.isfinite(total_loss):
        return 0.0, 0.0, 0.0

    # The usual 1 - 2*AUC calculation loses all precision when the target is
    # nearly constant. Pair concordance is algebraically equivalent, while
    # centering removes the common target level before any subtraction.
    centered_target = scaled_target - np.min(scaled_target)
    if not _LONGDOUBLE_EXTENDS_FLOAT64:
        float_centered = np.asarray(centered_target, dtype=np.float64)
        with np.errstate(under="ignore", invalid="ignore"):
            float_centered_products = float_weights * float_centered
        if np.any(
            (float_centered_products == 0.0) & (float_weights != 0.0) & (float_centered != 0.0)
        ):
            raise ValueError(
                "Gini aggregation requires an extended floating-point range on this platform"
            )
    perfect_pair_sum = _weighted_pair_concordance(
        scaled_target,
        normalized_weights,
        centered_target,
    )
    if perfect_pair_sum <= 0.0:
        return 0.0, 0.0, 0.0
    model_pair_sum = _weighted_pair_concordance(y_pred, normalized_weights, centered_target)
    gini_model_extended = model_pair_sum / total_loss
    gini_perfect_extended = perfect_pair_sum / total_loss
    float64_max = np.longdouble(np.finfo(np.float64).max)
    if (
        not np.isfinite(gini_model_extended)
        or not np.isfinite(gini_perfect_extended)
        or abs(gini_model_extended) > float64_max
        or abs(gini_perfect_extended) > float64_max
    ):
        raise ValueError("Gini coefficients must be finite")
    gini_model = float(gini_model_extended)
    gini_perfect = float(gini_perfect_extended)
    gini_ratio = np.clip(model_pair_sum / perfect_pair_sum, -1.0, 1.0)
    return gini_model, gini_perfect, float(gini_ratio)


def _normalized_gini(y_obs, y_pred, sample_weight=None) -> float:
    """Return a stable, tie-collapsed Gini ratio without creating a plot."""
    return _gini_coefficients(y_obs, y_pred, sample_weight)[2]


def _make_ax(ax: Axes | None):
    """Return (ax, fig_or_None). If ax is None, create a new figure."""
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax, None
    fig, ax_new = plt.subplots()
    return ax_new, fig


# ── Public functions ─────────────────────────────────────────────


def lift_chart(
    y_obs,
    y_pred,
    sample_weight=None,
    exposure=None,
    *,
    n_bins: int = 10,
    ax: Axes | None = None,
) -> LiftChartResult:
    """Lift chart: observed vs predicted across equal-exposure quantile bins.

    Parameters
    ----------
    y_obs : array-like
        Observed response values.
    y_pred : array-like
        Predicted response values.
    sample_weight : array-like or None
        Observation weights for aggregation.
    exposure : array-like or None
        Exposure measure for rate models. When provided, bins are
        equal-exposure quantiles and averages are exposure-weighted.
    n_bins : int
        Number of quantile bins.
    ax : matplotlib Axes or None
        If provided, plot onto this axes (``figure`` in result will be None).

    Returns
    -------
    LiftChartResult
        Contains a ``bins`` DataFrame and an optional ``figure``.
    """
    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred=y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        optional_vectors=("exposure",),
    )
    y_pred = vectors["y_pred"]
    n_bins = _validated_n_bins(n_bins)
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    # Bin by predicted value, using exposure as bin weights
    bin_weights = w * exp
    bins_idx = _quantile_bins(y_pred, bin_weights, n_bins)

    rows = []
    total_exp = (w * exp).sum()
    for b in range(n_bins):
        mask = bins_idx == b
        if not mask.any():
            continue
        wb = w[mask]
        eb = exp[mask]
        we = wb * eb
        we_sum = we.sum()
        obs_mean = _weighted_mean(y_obs[mask], we, "y_obs")
        pred_mean = _weighted_mean(y_pred[mask], we, "y_pred")
        exp_share = we_sum / total_exp if total_exp > 0 else 0.0
        ratio = _finite_ratio(obs_mean, pred_mean, "observed / predicted ratio")
        rows.append(
            {
                "bin": b + 1,
                "exposure_share": exp_share,
                "observed": obs_mean,
                "predicted": pred_mean,
                "obs_pred_ratio": ratio,
            }
        )

    df = pd.DataFrame(rows)

    ax_plot, fig = _make_ax(ax)
    x = np.arange(len(df))
    width = 0.35
    ax_plot.bar(x - width / 2, df["observed"], width, label="Observed", color="C0")
    ax_plot.bar(x + width / 2, df["predicted"], width, label="Predicted", color="C1")
    ax2 = ax_plot.twinx()
    ax2.plot(x, df["obs_pred_ratio"], "ko-", markersize=4, label="A/E ratio")
    ax2.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("A/E ratio")
    ax_plot.set_xticks(x)
    ax_plot.set_xticklabels(df["bin"].astype(int))
    ax_plot.set_xlabel("Bin")
    ax_plot.set_ylabel("Mean value")
    ax_plot.set_title("Lift Chart")
    ax_plot.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    return LiftChartResult(bins=df, figure=fig)


def double_lift_chart(
    y_obs,
    y_pred_model,
    y_pred_current,
    sample_weight=None,
    exposure=None,
    *,
    n_bins: int = 10,
    labels: tuple[str, str, str] = ("Actual", "Model", "Current"),
    ax: Axes | None = None,
) -> DoubleLiftChartResult:
    """CAS-style double lift chart (CAS RPM 2016 methodology).

    Sorts by the ratio ``y_pred_model / y_pred_current``, bins into
    equal-exposure quantiles, and plots three indexed series: Actual,
    Model, and Current — each indexed to its own overall average.

    This is the standard actuarial double lift chart for comparing a
    new model against a current/baseline model on holdout data.

    Parameters
    ----------
    y_obs : array-like
        Observed response values (frequency, severity, or loss ratio).
    y_pred_model : array-like
        New model predictions (holdout).
    y_pred_current : array-like
        Current/baseline/manual predictions (holdout).
    sample_weight : array-like or None
        Observation weights (case/frequency weights).
    exposure : array-like or None
        Exposure measure for rate models.
    n_bins : int
        Number of equal-exposure quantile bins.
    labels : tuple of (str, str, str)
        Display labels as ``(Actual, Model, Current)``. Each element
        names the corresponding series in the plot legend and axis labels.
    ax : matplotlib Axes or None
        If provided, plot onto this axes (``figure`` in result will be None).

    Returns
    -------
    DoubleLiftChartResult
        Contains a ``bins`` DataFrame and an optional ``figure``.

    References
    ----------
    CAS RPM 2016, "Predictive Modeling — Lift and Double Lift Charts",
    https://www.casact.org/sites/default/files/presentation/rpm_2016_presentations_pm-lm-4.pdf
    """
    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred_model=y_pred_model,
        y_pred_current=y_pred_current,
        sample_weight=sample_weight,
        exposure=exposure,
        optional_vectors=("exposure",),
    )
    y_pred_model = vectors["y_pred_model"]
    y_pred_current = vectors["y_pred_current"]
    n_bins = _validated_n_bins(n_bins)
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    # Sort score: model / current (with epsilon guard)
    eps = 1e-10
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sort_score = y_pred_model / np.maximum(y_pred_current, eps)
    if not np.all(np.isfinite(sort_score)):
        raise ValueError("y_pred_model / y_pred_current must contain only finite values")

    # Equal-exposure bins based on sort score
    bin_weights = w * exp
    bins_idx = _quantile_bins(sort_score, bin_weights, n_bins)

    # Overall exposure-weighted averages (for indexing)
    total_we = bin_weights.sum()
    overall_actual = _weighted_mean(y_obs, bin_weights, "y_obs")
    overall_model = _weighted_mean(y_pred_model, bin_weights, "y_pred_model")
    overall_current = _weighted_mean(y_pred_current, bin_weights, "y_pred_current")

    rows = []
    for b in range(n_bins):
        mask = bins_idx == b
        if not mask.any():
            continue
        we = bin_weights[mask]
        we_sum = we.sum()
        if we_sum <= 0:
            continue

        actual_avg = _weighted_mean(y_obs[mask], we, "y_obs")
        model_avg = _weighted_mean(y_pred_model[mask], we, "y_pred_model")
        current_avg = _weighted_mean(y_pred_current[mask], we, "y_pred_current")

        rows.append(
            {
                "bin": b + 1,
                "n_rows": int(mask.sum()),
                "exposure_sum": float(we_sum),
                "exposure_share": we_sum / total_we,
                "target_sum": _weighted_total(y_obs[mask], we, "y_obs"),
                "actual_avg": actual_avg,
                "model_avg": model_avg,
                "current_avg": current_avg,
                "actual_index": _finite_ratio(
                    actual_avg,
                    overall_actual,
                    "actual index",
                ),
                "model_index": _finite_ratio(
                    model_avg,
                    overall_model,
                    "model index",
                ),
                "current_index": _finite_ratio(
                    current_avg,
                    overall_current,
                    "current index",
                ),
                "sort_score_min": float(sort_score[mask].min()),
                "sort_score_median": float(np.median(sort_score[mask])),
                "sort_score_max": float(sort_score[mask].max()),
            }
        )

    df = pd.DataFrame(rows)

    # ── Plot ──────────────────────────────────────────────────────
    ax_plot, fig = _make_ax(ax)

    x = np.arange(len(df))
    lbl_actual, lbl_model, lbl_current = labels

    # Exposure-share bars on secondary axis (behind lines)
    ax_exp = ax_plot.twinx()
    ax_exp.bar(
        x,
        df["exposure_share"],
        width=0.8,
        alpha=0.08,
        color="grey",
        label="Exposure share",
        zorder=1,
    )
    ax_exp.set_ylabel("Exposure share", fontsize=8, color="grey")
    ax_exp.tick_params(axis="y", colors="grey")

    # Three indexed series
    ax_plot.plot(
        x,
        df["actual_index"],
        "o-",
        label=lbl_actual,
        color="C0",
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax_plot.plot(
        x,
        df["model_index"],
        "s-",
        label=lbl_model,
        color="C1",
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax_plot.plot(
        x,
        df["current_index"],
        "^-",
        label=lbl_current,
        color="C2",
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )
    ax_plot.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")

    ax_plot.set_xticks(x)
    ax_plot.set_xticklabels(df["bin"].astype(int))
    ax_plot.set_xlabel(f"Bin (sorted by {lbl_model} / {lbl_current} predicted rate)")
    ax_plot.set_ylabel("Indexed rate (bin avg / overall avg)")
    ax_plot.set_title("Double Lift Chart")
    ax_plot.legend(loc="upper left", fontsize=8)
    ax_exp.legend(loc="upper right", fontsize=7)

    return DoubleLiftChartResult(bins=df, figure=fig)


def lorenz_curve(
    y_obs,
    y_pred,
    sample_weight=None,
    exposure=None,
    *,
    engine: str = "matplotlib",
    ax: Axes | None = None,
) -> LorenzCurveResult:
    """Lorenz curve with Gini coefficient computation.

    Parameters
    ----------
    y_obs : array-like
        Observed response values.
    y_pred : array-like
        Predicted response values.
    sample_weight : array-like or None
        Observation weights.
    exposure : array-like or None
        Exposure measure. When provided, the Lorenz curve uses
        cumulative exposure share on the x-axis.
    engine : {"matplotlib", "plotly"}
        Plotting backend. ``"plotly"`` requires the optional plotly dependency.
    ax : matplotlib Axes or None
        If provided, plot onto this axes. Only valid with
        ``engine="matplotlib"``.

    Returns
    -------
    LorenzCurveResult
        Contains ``curve`` DataFrame, ``gini_model``, ``gini_perfect``,
        ``gini_ratio``, and an optional ``figure``.
    """
    if engine not in {"matplotlib", "plotly"}:
        raise ValueError(f"engine={engine!r} is not valid, expected 'matplotlib' or 'plotly'.")
    if engine == "plotly" and ax is not None:
        raise ValueError("ax= is only supported with engine='matplotlib'.")

    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred=y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        optional_vectors=("exposure",),
    )
    y_pred = vectors["y_pred"]
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    _require_extended_range("Lorenz aggregation", w, exp, y_obs)
    if not _LONGDOUBLE_EXTENDS_FLOAT64:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            float_exposures = np.asarray(w, dtype=np.float64) * np.asarray(exp, dtype=np.float64)
            float_losses = float_exposures * np.asarray(y_obs, dtype=np.float64)
        try:
            absolute_loss_total = math.fsum(np.abs(float_losses).tolist())
            exposure_total = math.fsum(float_exposures.tolist())
        except OverflowError:
            absolute_loss_total = float("inf")
            exposure_total = float("inf")
        if (
            np.any(~np.isfinite(float_exposures))
            or np.any(~np.isfinite(float_losses))
            or not np.isfinite(exposure_total)
            or not np.isfinite(absolute_loss_total)
            or np.any((float_losses == 0.0) & (float_exposures != 0.0) & (np.asarray(y_obs) != 0.0))
        ):
            raise ValueError(
                "Lorenz aggregation requires an extended floating-point range on this platform"
            )
    if _LONGDOUBLE_EXTENDS_FLOAT64:
        exposures = np.asarray(w, dtype=np.longdouble) * np.asarray(exp, dtype=np.longdouble)
        losses = exposures * np.asarray(y_obs, dtype=np.longdouble)
        total_loss = np.sum(losses, dtype=np.longdouble)
        total_exp = np.sum(exposures, dtype=np.longdouble)
    else:
        exposures = float_exposures
        losses = float_losses
        total_loss = math.fsum(losses.tolist())
        total_exp = math.fsum(exposures.tolist())

    if total_loss <= 0 or total_exp <= 0:
        # Degenerate: all zeros or no exposure
        curve_df = pd.DataFrame(
            {
                "cum_exposure_share": [0.0, 1.0],
                "cum_loss_share_ordered": [0.0, 1.0],
                "cum_loss_share_model": [0.0, 1.0],
                "cum_loss_share_perfect": [0.0, 1.0],
            }
        )
        if engine == "plotly":
            try:
                import plotly.graph_objects as go
            except ImportError:
                raise ImportError(
                    "plotly is required for engine='plotly'. Install it with: pip install plotly"
                ) from None

            from superglm.plotting.common import _PLOTLY_TEXT, _apply_plotly_theme

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[0.0, 1.0],
                    y=[0.0, 1.0],
                    mode="lines",
                    name="Random",
                    line=dict(color=_PLOTLY_TEXT, dash="dash", width=1.2),
                )
            )
            _apply_plotly_theme(
                fig, hovermode="x unified", height=460, margin=dict(t=72, r=28, b=72, l=72)
            )
            fig.update_layout(title="Lorenz Curve (degenerate)")
            fig.update_xaxes(title_text="Cumulative exposure share", range=[0.0, 1.0])
            fig.update_yaxes(title_text="Cumulative loss share", range=[0.0, 1.0])
        else:
            ax_plot, fig = _make_ax(ax)
            ax_plot.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
            ax_plot.set_title("Lorenz Curve (degenerate)")
            ax_plot.legend(fontsize=7)
        return LorenzCurveResult(
            curve=curve_df, gini_model=0.0, gini_perfect=0.0, gini_ratio=0.0, figure=fig
        )

    # Order by model predictions (ascending = lowest risk first), collapsing
    # tied predictions into a single block so within-tie row order carries no
    # fake ranking information.
    cum_exp_model, cum_loss_model = _lorenz_cumulative_by_score(
        y_pred,
        exposures,
        losses,
        total_exp=total_exp,
        total_loss=total_loss,
    )

    # Order by actual loss ratio (ascending = lowest actual risk first)
    # For perfect foresight ordering
    loss_ratio = np.where(exp > 0, y_obs, 0.0)
    cum_exp_perfect, cum_loss_perfect = _lorenz_cumulative_by_score(
        loss_ratio,
        exposures,
        losses,
        total_exp=total_exp,
        total_loss=total_loss,
    )

    # Random ordering = diagonal
    # Prepend (0, 0)
    cum_exp_m = np.concatenate([[0.0], cum_exp_model])
    cum_loss_m = np.concatenate([[0.0], cum_loss_model])
    cum_exp_p = np.concatenate([[0.0], cum_exp_perfect])
    cum_loss_p = np.concatenate([[0.0], cum_loss_perfect])

    # Gini coefficients
    gini_model, gini_perfect, gini_ratio = _gini_coefficients(
        y_obs,
        y_pred,
        exposures,
    )

    # Build curve DataFrame — use model ordering x-axis for all curves
    # Random ordering diagonal: cum_loss_share == cum_exposure_share
    curve_df = pd.DataFrame(
        {
            "cum_exposure_share": cum_exp_m,
            "cum_loss_share_ordered": cum_exp_m,
            "cum_loss_share_model": cum_loss_m,
            "cum_loss_share_perfect": np.interp(cum_exp_m, cum_exp_p, cum_loss_p),
        }
    )

    # Plot
    if engine == "plotly":
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for engine='plotly'. Install it with: pip install plotly"
            ) from None

        from superglm.plotting.common import (
            _PLOTLY_LINE_COLOR,
            _PLOTLY_SIM_FILL,
            _PLOTLY_TEXT,
            _apply_plotly_theme,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="Random",
                line=dict(color=_PLOTLY_TEXT, dash="dash", width=1.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cum_exp_m,
                y=cum_loss_m,
                mode="lines",
                name="Model",
                line=dict(color=_PLOTLY_LINE_COLOR, width=2.4),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=cum_exp_p,
                y=cum_loss_p,
                mode="lines",
                name="Perfect",
                line=dict(color=_PLOTLY_SIM_FILL, width=2.0),
                opacity=0.85,
            )
        )
        _apply_plotly_theme(
            fig,
            hovermode="x unified",
            height=460,
            margin=dict(t=72, r=28, b=72, l=72),
        )
        fig.update_layout(title=f"Lorenz Curve (Gini ratio = {gini_ratio:.3f})")
        fig.update_xaxes(title_text="Cumulative exposure share", range=[0.0, 1.0])
        fig.update_yaxes(title_text="Cumulative loss share", range=[0.0, 1.0])
    else:
        ax_plot, fig = _make_ax(ax)
        ax_plot.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
        ax_plot.plot(cum_exp_m, cum_loss_m, "-", color="C0", linewidth=1.2, label="Model")
        ax_plot.plot(
            cum_exp_p, cum_loss_p, "-", color="C2", linewidth=1.0, alpha=0.7, label="Perfect"
        )
        ax_plot.set_xlabel("Cumulative exposure share")
        ax_plot.set_ylabel("Cumulative loss share")
        ax_plot.set_title(f"Lorenz Curve (Gini ratio = {gini_ratio:.3f})")
        ax_plot.legend(fontsize=8)

    return LorenzCurveResult(
        curve=curve_df,
        gini_model=gini_model,
        gini_perfect=gini_perfect,
        gini_ratio=gini_ratio,
        figure=fig,
    )


def loss_ratio_chart(
    y_obs,
    y_pred,
    sample_weight=None,
    exposure=None,
    *,
    n_bins: int = 10,
    feature_values=None,
    feature_name: str | None = None,
    ax: Axes | None = None,
) -> LossRatioChartResult:
    """Loss ratio chart: observed vs predicted loss ratios per bin.

    Parameters
    ----------
    y_obs : array-like
        Observed response values.
    y_pred : array-like
        Predicted response values.
    sample_weight : array-like or None
        Observation weights.
    exposure : array-like or None
        Exposure measure for rate models.
    n_bins : int
        Number of quantile bins.
    feature_values : array-like or None
        If provided, bin by this feature's values instead of predicted values.
    feature_name : str or None
        Label for the feature axis.
    ax : matplotlib Axes or None
        If provided, plot onto this axes.

    Returns
    -------
    LossRatioChartResult
        Contains a ``bins`` DataFrame and an optional ``figure``.
    """
    y_obs, vectors, w = _validated_chart_inputs(
        y_obs,
        y_pred=y_pred,
        sample_weight=sample_weight,
        exposure=exposure,
        feature_values=feature_values,
        optional_vectors=("exposure", "feature_values"),
    )
    y_pred = vectors["y_pred"]
    n_bins = _validated_n_bins(n_bins)
    n = len(y_obs)
    exp = vectors.get("exposure", np.ones(n, dtype=float))

    # Determine what to bin by
    if feature_values is not None:
        sort_vals = vectors["feature_values"]
        x_label = feature_name or "Feature"
    else:
        sort_vals = y_pred
        x_label = "Predicted value"

    bin_weights = w * exp
    bins_idx = _quantile_bins(sort_vals, bin_weights, n_bins)

    rows = []
    total_exp = bin_weights.sum()
    for b in range(n_bins):
        mask = bins_idx == b
        if not mask.any():
            continue
        we = w[mask] * exp[mask]
        we_sum = we.sum()
        obs_lr = _weighted_mean(y_obs[mask], we, "y_obs")
        pred_lr = _weighted_mean(y_pred[mask], we, "y_pred")
        rows.append(
            {
                "bin": b + 1,
                "exposure_share": we_sum / total_exp if total_exp > 0 else 0.0,
                "observed": obs_lr,
                "predicted": pred_lr,
            }
        )

    df = pd.DataFrame(rows)

    ax_plot, fig = _make_ax(ax)
    x = np.arange(len(df))
    width = 0.35
    ax_plot.bar(x - width / 2, df["observed"], width, label="Observed", color="C0")
    ax_plot.bar(x + width / 2, df["predicted"], width, label="Predicted", color="C1")

    # Volume overlay
    ax2 = ax_plot.twinx()
    ax2.bar(x, df["exposure_share"], width=0.8, alpha=0.15, color="grey", label="Exposure share")
    ax2.set_ylabel("Exposure share")

    ax_plot.set_xticks(x)
    ax_plot.set_xticklabels(df["bin"].astype(int))
    ax_plot.set_xlabel(x_label)
    ax_plot.set_ylabel("Loss ratio")
    ax_plot.set_title("Loss Ratio Chart")
    ax_plot.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    return LossRatioChartResult(bins=df, figure=fig)
