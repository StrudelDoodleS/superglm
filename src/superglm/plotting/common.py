"""Shared styling constants and KDE helpers for plotting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ── Visual language constants ──────────────────────────────────────
_LINE_COLOR = "#006FDD"
_LINE_WIDTH = 1.35
_PW_FILL = "#F58518"
_PW_ALPHA = 0.24
_PW_EDGE_ALPHA = 0.9
_PW_EDGE_LW = 1.0
_SIM_FILL = "#4C78A8"
_SIM_ALPHA = 0.16
_SIM_EDGE_ALPHA = 0.85
_SIM_EDGE_LW = 0.95
_EXP_FILL = "#F4D35E"
_EXP_EDGE = "#D8A10F"
_EXP_EDGE_LW = 1.1
_REF_COLOR = "0.45"
_REF_LW = 0.8
_KNOT_COLOR = "#006FDD"
_CAT_BAR_COLOR = "#006FDD"


def _exposure_kde(x_vals, sample_weight, grid, bw_factor=0.03):
    """Weighted KDE for sample_weight distribution, returned on *grid*."""
    bw = bw_factor * (grid[-1] - grid[0])
    diff = grid[:, None] - x_vals[None, :]
    kernel = np.exp(-0.5 * (diff / bw) ** 2)
    density = kernel @ sample_weight
    return density / density.max()  # normalise to [0, 1]


def _kde_2d(
    x1_vals: NDArray,
    x2_vals: NDArray,
    weights: NDArray,
    x1_grid: NDArray,
    x2_grid: NDArray,
    bw_factor: float = 0.05,
) -> NDArray:
    """Exposure-weighted 2D KDE evaluated on a meshgrid.

    Returns a (len(x2_grid), len(x1_grid)) array normalised to [0, 1].
    """
    bw1 = bw_factor * (x1_grid[-1] - x1_grid[0])
    bw2 = bw_factor * (x2_grid[-1] - x2_grid[0])

    # Vectorised product of two 1D Gaussian kernels
    d1 = np.exp(-0.5 * ((x1_grid[:, None] - x1_vals[None, :]) / bw1) ** 2)  # (g1, n)
    d2 = np.exp(-0.5 * ((x2_grid[:, None] - x2_vals[None, :]) / bw2) ** 2)  # (g2, n)
    # Weighted outer-product sum: D[j, i] = Σ_k w_k * d1[i,k] * d2[j,k]
    D = (d2 * weights[None, :]) @ d1.T  # (g2, g1)
    peak = D.max()
    if peak > 0:
        D /= peak
    return D


def _make_continuous_figure(
    needs_strip: bool,
    figsize: tuple[float, float] | None,
) -> tuple:
    """Create a figure for a continuous term with optional density strip.

    Returns (fig, ax_main, ax_density_or_None).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if needs_strip:
        if figsize is None:
            figsize = (7, 5.5)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[4.2, 1.0], hspace=0.16)
        ax = fig.add_subplot(gs[0])
        ax_den = fig.add_subplot(gs[1])
        ax_den.tick_params(axis="x", labelbottom=False)
        ax.set_zorder(ax_den.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.tick_params(axis="x", labelbottom=True, pad=-2)
    else:
        if figsize is None:
            figsize = (7, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
        ax_den = None
    return fig, ax, ax_den
