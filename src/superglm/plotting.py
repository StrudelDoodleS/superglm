"""Relativity plotting for SuperGLM models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray


def _exposure_kde(x_vals, exposure, grid, bw_factor=0.03):
    """Weighted KDE for exposure distribution, returned on *grid*."""
    bw = bw_factor * (grid[-1] - grid[0])
    diff = grid[:, None] - x_vals[None, :]
    kernel = np.exp(-0.5 * (diff / bw) ** 2)
    density = kernel @ exposure
    return density / density.max()  # normalise to [0, 1]


def plot_relativities(
    relativities: dict[str, pd.DataFrame],
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create a grid of relativity plots from ``SuperGLM.relativities()`` output.

    Parameters
    ----------
    relativities : dict[str, DataFrame]
        Output of :meth:`SuperGLM.relativities`.
    X : DataFrame, optional
        Training data.  When provided together with *exposure*, an exposure
        distribution is shown on each subplot (filled area for continuous
        features, bars for categoricals).
    exposure : array-like, optional
        Exposure weights corresponding to rows of *X*.
    ncols : int
        Number of subplot columns (default 2).
    figsize : tuple, optional
        Figure size ``(width, height)``.  Auto-sized if *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    names = list(relativities.keys())
    n = len(names)
    if n == 0:
        fig, _ = plt.subplots()
        return fig

    show_exposure = X is not None and exposure is not None
    if show_exposure:
        exposure = np.asarray(exposure, dtype=np.float64)

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, name in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        df = relativities[name]

        if "x" in df.columns:
            # Continuous (spline / polynomial)
            if show_exposure and name in X.columns:
                x_vals = np.asarray(X[name], dtype=np.float64)
                grid = np.asarray(df["x"], dtype=np.float64)
                density = _exposure_kde(x_vals, exposure, grid)
                ax2 = ax.twinx()
                ax2.fill_between(
                    grid, 0, density,
                    color="lightgrey", alpha=0.4, label="Exposure",
                )
                ax2.set_ylim(0, 1.3)  # leave headroom
                ax2.set_yticks([])
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
            line = ax.plot(
                df["x"], df["relativity"],
                color="steelblue", linewidth=1.5, label="Relativity",
            )
            ax.axhline(1.0, linestyle="--", color="grey", linewidth=0.8)

        elif "level" in df.columns:
            # Categorical — horizontal bars
            if show_exposure and name in X.columns:
                level_exp = (
                    pd.DataFrame({"level": X[name], "exposure": exposure})
                    .groupby("level", sort=False)["exposure"]
                    .sum()
                )
                exp_vals = [level_exp.get(lv, 0.0) for lv in df["level"]]
                ax2 = ax.twinx()
                ax2.barh(
                    df["level"], exp_vals,
                    color="lightgrey", alpha=0.4, label="Exposure",
                )
                ax2.set_yticks([])
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
            ax.barh(
                df["level"], df["relativity"],
                color="steelblue", label="Relativity",
            )
            ax.axvline(1.0, linestyle="--", color="grey", linewidth=0.8)

        elif "label" in df.columns:
            # Numeric — single horizontal bar
            ax.barh(df["label"], df["relativity"], color="steelblue")
            ax.axvline(1.0, linestyle="--", color="grey", linewidth=0.8)

        ax.set_title(name)
        if idx % ncols == 0:
            ax.set_ylabel("Relativity")

    # Single legend for the whole figure
    if show_exposure:
        handles, labels = [], []
        # Grab from first axis that has both
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break
        # Add the exposure patch from twin axis
        for ax_row in axes:
            for ax in ax_row:
                for child in ax.figure.get_axes():
                    h2, l2 = child.get_legend_handles_labels()
                    for h, l in zip(h2, l2):
                        if l == "Exposure" and l not in labels:
                            handles.append(h)
                            labels.append(l)
                            break
                if "Exposure" in labels:
                    break
            if "Exposure" in labels:
                break
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.8)

    # Hide unused subplots
    for idx in range(len(names), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig
