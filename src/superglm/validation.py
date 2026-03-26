"""Actuarial validation toolkit for model comparison and calibration assessment.

Provides lift charts, double lift charts, Lorenz curves with Gini coefficients,
and loss ratio charts following CAS RPM 2016 methodology.

All functions accept raw numpy arrays and are usable with any model framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass(frozen=True)
class LiftChartResult:
    """Result from :func:`lift_chart`."""

    bins: pd.DataFrame
    figure: Figure | None


@dataclass(frozen=True)
class DoubleLiftChartResult:
    """Result from :func:`double_lift_chart`."""

    bins: pd.DataFrame
    figure: Figure | None


@dataclass(frozen=True)
class LorenzCurveResult:
    """Result from :func:`lorenz_curve`."""

    curve: pd.DataFrame
    gini_model: float
    gini_perfect: float
    gini_ratio: float
    figure: Figure | None


@dataclass(frozen=True)
class LossRatioChartResult:
    """Result from :func:`loss_ratio_chart`."""

    bins: pd.DataFrame
    figure: Figure | None


# ── Private helpers ──────────────────────────────────────────────


def _ensure_array(a) -> NDArray:
    return np.asarray(a, dtype=float)


def _default_weights(w, n: int) -> NDArray:
    if w is None:
        return np.ones(n, dtype=float)
    return _ensure_array(w)


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


def _gini(cum_x: NDArray, cum_y: NDArray) -> float:
    """Gini coefficient from a Lorenz curve via trapezoidal rule.

    Gini = 1 - 2 * AUC(lorenz_curve).
    """
    auc = float(np.trapezoid(cum_y, cum_x))
    return 1.0 - 2.0 * auc


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
    """
    y_obs = _ensure_array(y_obs)
    y_pred = _ensure_array(y_pred)
    n = len(y_obs)
    w = _default_weights(sample_weight, n)
    exp = _ensure_array(exposure) if exposure is not None else np.ones(n, dtype=float)

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
        obs_mean = np.sum(we * y_obs[mask]) / we_sum if we_sum > 0 else 0.0
        pred_mean = np.sum(we * y_pred[mask]) / we_sum if we_sum > 0 else 0.0
        exp_share = we_sum / total_exp if total_exp > 0 else 0.0
        ratio = obs_mean / pred_mean if pred_mean != 0 else np.nan
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
    y_pred_a,
    y_pred_b,
    sample_weight=None,
    exposure=None,
    *,
    n_bins: int = 10,
    labels: tuple[str, str] = ("Model A", "Model B"),
    ax: Axes | None = None,
) -> DoubleLiftChartResult:
    """Double lift chart: compare two models' A/E ratios across bins.

    Bins are defined by model A's predictions.

    Parameters
    ----------
    y_obs : array-like
        Observed response values.
    y_pred_a, y_pred_b : array-like
        Predictions from model A and model B.
    sample_weight : array-like or None
        Observation weights.
    exposure : array-like or None
        Exposure measure for rate models.
    n_bins : int
        Number of quantile bins.
    labels : tuple of str
        Labels for the two models.
    ax : matplotlib Axes or None
        If provided, plot onto this axes.

    Returns
    -------
    DoubleLiftChartResult
    """
    y_obs = _ensure_array(y_obs)
    y_pred_a = _ensure_array(y_pred_a)
    y_pred_b = _ensure_array(y_pred_b)
    n = len(y_obs)
    w = _default_weights(sample_weight, n)
    exp = _ensure_array(exposure) if exposure is not None else np.ones(n, dtype=float)

    bin_weights = w * exp
    bins_idx = _quantile_bins(y_pred_a, bin_weights, n_bins)

    rows = []
    total_exp = bin_weights.sum()
    for b in range(n_bins):
        mask = bins_idx == b
        if not mask.any():
            continue
        we = w[mask] * exp[mask]
        we_sum = we.sum()
        obs_mean = np.sum(we * y_obs[mask]) / we_sum if we_sum > 0 else 0.0
        pred_a_mean = np.sum(we * y_pred_a[mask]) / we_sum if we_sum > 0 else 0.0
        pred_b_mean = np.sum(we * y_pred_b[mask]) / we_sum if we_sum > 0 else 0.0
        ae_a = obs_mean / pred_a_mean if pred_a_mean != 0 else np.nan
        ae_b = obs_mean / pred_b_mean if pred_b_mean != 0 else np.nan
        rows.append(
            {
                "bin": b + 1,
                "exposure_share": we_sum / total_exp if total_exp > 0 else 0.0,
                "observed": obs_mean,
                f"predicted_{labels[0]}": pred_a_mean,
                f"predicted_{labels[1]}": pred_b_mean,
                f"ae_ratio_{labels[0]}": ae_a,
                f"ae_ratio_{labels[1]}": ae_b,
            }
        )

    df = pd.DataFrame(rows)

    ax_plot, fig = _make_ax(ax)
    x = np.arange(len(df))
    ax_plot.plot(x, df[f"ae_ratio_{labels[0]}"], "o-", label=labels[0], markersize=4)
    ax_plot.plot(x, df[f"ae_ratio_{labels[1]}"], "s-", label=labels[1], markersize=4)
    ax_plot.axhline(1.0, color="grey", linewidth=0.7, linestyle="--")
    ax_plot.set_xticks(x)
    ax_plot.set_xticklabels(df["bin"].astype(int))
    ax_plot.set_xlabel("Bin (sorted by " + labels[0] + " prediction)")
    ax_plot.set_ylabel("A/E ratio")
    ax_plot.set_title("Double Lift Chart")
    ax_plot.legend(fontsize=8)

    return DoubleLiftChartResult(bins=df, figure=fig)


def lorenz_curve(
    y_obs,
    y_pred,
    sample_weight=None,
    exposure=None,
    *,
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
    ax : matplotlib Axes or None
        If provided, plot onto this axes.

    Returns
    -------
    LorenzCurveResult
    """
    y_obs = _ensure_array(y_obs)
    y_pred = _ensure_array(y_pred)
    n = len(y_obs)
    w = _default_weights(sample_weight, n)
    exp = _ensure_array(exposure) if exposure is not None else np.ones(n, dtype=float)

    losses = w * y_obs * exp  # total loss per observation
    exposures = w * exp

    total_loss = losses.sum()
    total_exp = exposures.sum()

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
        ax_plot, fig = _make_ax(ax)
        ax_plot.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
        ax_plot.set_title("Lorenz Curve (degenerate)")
        ax_plot.legend(fontsize=7)
        return LorenzCurveResult(
            curve=curve_df, gini_model=0.0, gini_perfect=0.0, gini_ratio=0.0, figure=fig
        )

    # Order by model predictions (ascending = lowest risk first)
    order_model = np.argsort(y_pred)
    cum_exp_model = np.cumsum(exposures[order_model]) / total_exp
    cum_loss_model = np.cumsum(losses[order_model]) / total_loss

    # Order by actual loss ratio (ascending = lowest actual risk first)
    # For perfect foresight ordering
    loss_ratio = np.where(exp > 0, y_obs, 0.0)
    order_perfect = np.argsort(loss_ratio)
    cum_loss_perfect = np.cumsum(losses[order_perfect]) / total_loss
    cum_exp_perfect = np.cumsum(exposures[order_perfect]) / total_exp

    # Random ordering = diagonal
    # Prepend (0, 0)
    cum_exp_m = np.concatenate([[0.0], cum_exp_model])
    cum_loss_m = np.concatenate([[0.0], cum_loss_model])
    cum_exp_p = np.concatenate([[0.0], cum_exp_perfect])
    cum_loss_p = np.concatenate([[0.0], cum_loss_perfect])

    # Gini coefficients
    gini_model = _gini(cum_exp_m, cum_loss_m)
    gini_perfect = _gini(cum_exp_p, cum_loss_p)
    gini_ratio = gini_model / gini_perfect if gini_perfect > 0 else 0.0

    # Build curve DataFrame — use model ordering x-axis for all curves
    # Interpolate perfect curve onto model's x-axis
    cum_loss_ordered = np.linspace(0, 1, n + 1)  # random = diagonal
    curve_df = pd.DataFrame(
        {
            "cum_exposure_share": cum_exp_m,
            "cum_loss_share_ordered": cum_loss_ordered,
            "cum_loss_share_model": cum_loss_m,
            "cum_loss_share_perfect": np.interp(cum_exp_m, cum_exp_p, cum_loss_p),
        }
    )

    # Plot
    ax_plot, fig = _make_ax(ax)
    ax_plot.plot(cum_exp_m, cum_loss_ordered, "k--", linewidth=0.7, label="Random")
    ax_plot.plot(cum_exp_m, cum_loss_m, "-", color="C0", linewidth=1.2, label="Model")
    ax_plot.plot(cum_exp_p, cum_loss_p, "-", color="C2", linewidth=1.0, alpha=0.7, label="Perfect")
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
    """
    y_obs = _ensure_array(y_obs)
    y_pred = _ensure_array(y_pred)
    n = len(y_obs)
    w = _default_weights(sample_weight, n)
    exp = _ensure_array(exposure) if exposure is not None else np.ones(n, dtype=float)

    # Determine what to bin by
    if feature_values is not None:
        sort_vals = _ensure_array(feature_values)
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
        obs_lr = np.sum(we * y_obs[mask]) / we_sum if we_sum > 0 else 0.0
        pred_lr = np.sum(we * y_pred[mask]) / we_sum if we_sum > 0 else 0.0
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
