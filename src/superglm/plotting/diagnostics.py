"""R-style 4-panel residual diagnostic plots for SuperGLM models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from superglm.model import SuperGLM


@numba.njit(cache=True)
def _lowess_core(x_s, y_s, k, n_steps):
    """Numba-accelerated LOWESS with tricube kernel and sliding window."""
    n = len(x_s)
    y_hat = np.empty(n)
    robustness_w = np.ones(n)

    for _iteration in range(1 + n_steps):
        left = 0
        for i in range(n):
            # Slide window right: k nearest neighbours are contiguous in sorted x
            while left + k < n:
                if x_s[i] - x_s[left] > x_s[left + k] - x_s[i]:
                    left += 1
                else:
                    break

            right = min(left + k, n)
            h = max(x_s[i] - x_s[left], x_s[right - 1] - x_s[i]) + 1e-10

            # Single-pass weighted regression: accumulate sufficient statistics
            sw = 0.0
            swx = 0.0
            swy = 0.0
            swxx = 0.0
            swxy = 0.0
            for j in range(left, right):
                u = abs(x_s[j] - x_s[i]) / h
                if u < 1.0:
                    u3 = u * u * u
                    t = 1.0 - u3
                    w = t * t * t * robustness_w[j]
                else:
                    w = 0.0
                sw += w
                wx = w * x_s[j]
                swx += wx
                swy += w * y_s[j]
                swxx += wx * x_s[j]
                swxy += w * x_s[j] * y_s[j]

            if sw < 1e-12:
                y_hat[i] = y_s[i]
                continue

            mx = swx / sw
            my = swy / sw
            den = swxx - swx * swx / sw
            if abs(den) < 1e-12:
                y_hat[i] = my
            else:
                num = swxy - swx * swy / sw
                y_hat[i] = my + (num / den) * (x_s[i] - mx)

        # Update robustness weights after initial fit and each robustness step
        if _iteration < n_steps:
            # Median absolute residual
            abs_resid = np.empty(n)
            for i in range(n):
                abs_resid[i] = abs(y_s[i] - y_hat[i])
            med_resid = np.median(abs_resid)
            if med_resid < 1e-12:
                break
            inv6m = 1.0 / (6.0 * med_resid)
            for i in range(n):
                u = (y_s[i] - y_hat[i]) * inv6m
                au = abs(u)
                if au < 1.0:
                    t = 1.0 - u * u
                    robustness_w[i] = t * t
                else:
                    robustness_w[i] = 0.0

    return y_hat


def _lowess(x: NDArray, y: NDArray, *, frac: float = 0.6, n_steps: int = 2) -> NDArray:
    """LOWESS smoother (numba-accelerated, tricube kernel, robustness iterations)."""
    n = len(x)
    if n < 3:
        return y.copy()

    order = np.argsort(x)
    x_s = np.ascontiguousarray(x[order], dtype=np.float64)
    y_s = np.ascontiguousarray(y[order], dtype=np.float64)
    k = max(int(np.ceil(frac * n)), 3)

    y_hat = _lowess_core(x_s, y_s, k, n_steps)

    result = np.empty(n)
    result[order] = y_hat
    return result


def plot_diagnostics(
    model: SuperGLM,
    X,
    y: NDArray,
    sample_weight: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    residual_type: str = "deviance",
    figsize: tuple[float, float] | None = None,
    max_points: int = 25_000,
    seed: int = 42,
) -> Figure:
    """Create an R-style 2x2 residual diagnostic figure.

    Parameters
    ----------
    model : SuperGLM
        A fitted SuperGLM model.
    X : pd.DataFrame
        Design matrix.
    y : NDArray
        Response vector.
    sample_weight : NDArray or None
        Optional observation weights.
    offset : NDArray or None
        Optional offset.
    residual_type : str
        Residual type for Panel 1. One of ``"deviance"``, ``"pearson"``,
        ``"response"``, ``"working"``, ``"quantile"``.
    figsize : tuple or None
        Figure size ``(width, height)`` in inches. Defaults to ``(10, 8)``.
    max_points : int
        Maximum number of points to plot and smooth. When ``n > max_points``,
        a random subsample is drawn for scatter plots and LOWESS smoothers.
        Set to 0 to disable subsampling.
    seed : int
        Random seed for quantile residuals and subsampling.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with 4 diagnostic subplots.
    """
    valid_types = {"deviance", "pearson", "response", "working", "quantile"}
    if residual_type not in valid_types:
        raise ValueError(
            f"Invalid residual_type={residual_type!r}. Must be one of {sorted(valid_types)}."
        )

    if figsize is None:
        figsize = (10, 8)

    # Build metrics object
    m = model.metrics(X, y, sample_weight, offset)
    mu = m._mu
    n = m.n_obs

    # Residuals for Panel 1
    resid_p1 = m.residuals(residual_type, seed=seed)

    # Standardized deviance residuals for Panels 2-4
    std_dev_resid = m.std_deviance_residuals

    # Leverage for Panel 4
    leverage = m.leverage

    # Subsample for plotting/smoothing if n is large
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_points, replace=False)
        mu = mu[idx]
        resid_p1 = resid_p1[idx]
        std_dev_resid = std_dev_resid[idx]
        leverage = leverage[idx]
        n = max_points

    # Build the family/link label for suptitle
    family_name = type(model._distribution).__name__
    link_name = type(model._link).__name__

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Diagnostics: {family_name} family, {link_name}", fontsize=12, y=0.98)

    # ── Panel 1: Residuals vs Fitted ──────────────────────────────
    ax1 = axes[0, 0]
    ax1.scatter(mu, resid_p1, s=6, alpha=0.4, edgecolors="none", color="C0")
    ax1.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    # Lowess smoother
    if n > 3:
        smoother = _lowess(mu, resid_p1)
        order = np.argsort(mu)
        ax1.plot(mu[order], smoother[order], color="red", linewidth=1.2)
    ax1.set_xlabel("Fitted values")
    ax1.set_ylabel(f"{residual_type.capitalize()} residuals")
    ax1.set_title("Residuals vs Fitted")

    # ── Panel 2: Normal Q-Q ───────────────────────────────────────
    ax2 = axes[0, 1]
    sorted_resid = np.sort(std_dev_resid)
    n_resid = len(sorted_resid)
    theoretical_q = stats.norm.ppf((np.arange(1, n_resid + 1) - 0.375) / (n_resid + 0.25))
    ax2.scatter(theoretical_q, sorted_resid, s=6, alpha=0.4, edgecolors="none", color="C0")
    # 45-degree reference line
    lims = [
        min(theoretical_q.min(), sorted_resid.min()),
        max(theoretical_q.max(), sorted_resid.max()),
    ]
    ax2.plot(lims, lims, color="red", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Theoretical quantiles")
    ax2.set_ylabel("Std. deviance residuals")
    ax2.set_title("Normal Q-Q")

    # ── Panel 3: Scale-Location ───────────────────────────────────
    ax3 = axes[1, 0]
    sqrt_abs_std = np.sqrt(np.abs(std_dev_resid))
    ax3.scatter(mu, sqrt_abs_std, s=6, alpha=0.4, edgecolors="none", color="C0")
    if n > 3:
        smoother3 = _lowess(mu, sqrt_abs_std)
        order = np.argsort(mu)
        ax3.plot(mu[order], smoother3[order], color="red", linewidth=1.2)
    ax3.set_xlabel("Fitted values")
    ax3.set_ylabel(r"$\sqrt{|\mathrm{Std.\ residuals}|}$")
    ax3.set_title("Scale-Location")

    # ── Panel 4: Leverage vs Std. Residuals ───────────────────────
    ax4 = axes[1, 1]
    ax4.scatter(leverage, std_dev_resid, s=6, alpha=0.4, edgecolors="none", color="C0")
    ax4.set_xlabel("Leverage")
    ax4.set_ylabel("Std. deviance residuals")
    ax4.set_title("Residuals vs Leverage")

    # Cook's distance contours
    p = m.effective_df
    phi = m.phi
    h_range = np.linspace(1e-4, min(leverage.max() * 1.2, 0.999), 200)
    for threshold, ls in [(0.5, "--"), (1.0, "-")]:
        # Cook's D: C = r^2 * h / ((1-h)^2 * p * phi)
        # Solve for r: r = +/- sqrt(C * (1-h)^2 * p * phi / h)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_sq = threshold * (1 - h_range) ** 2 * p * phi / h_range
        r_sq = np.where(r_sq > 0, r_sq, np.nan)
        r_vals = np.sqrt(r_sq)
        valid = np.isfinite(r_vals)
        if valid.any():
            ax4.plot(
                h_range[valid],
                r_vals[valid],
                color="grey",
                linewidth=0.7,
                linestyle=ls,
                label=f"Cook's D={threshold}",
            )
            ax4.plot(
                h_range[valid],
                -r_vals[valid],
                color="grey",
                linewidth=0.7,
                linestyle=ls,
            )
    ax4.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
