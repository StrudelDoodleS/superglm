"""Shared styling constants and KDE helpers for plotting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ── Matplotlib visual-language constants ───────────────────────────
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
_SPECIAL_COLOR = "#7A3EA1"

# ── Plotly-specific color overrides ────────────────────────────────
_PLOTLY_LINE_COLOR = "#E10600"
_PLOTLY_PW_FILL = "#FFD323"
_PLOTLY_SIM_FILL = "#2F61D5"
_PLOTLY_EXP_FILL = "#FFD323"
_PLOTLY_EXP_EDGE = "#A85C00"
_PLOTLY_KNOT_COLOR = "#FFD323"
_PLOTLY_CAT_BAR_COLOR = "#E10600"
_PLOTLY_SPECIAL_COLOR = "#1E63D7"
_PLOTLY_PAPER = "#f5efe3"
_PLOTLY_PANEL = "#fffdf8"
_PLOTLY_GRID = "rgba(23, 20, 17, 0.10)"
_PLOTLY_AXIS = "rgba(23, 20, 17, 0.34)"
_PLOTLY_TEXT = "#171411"
_PLOTLY_FONT = "Avenir Next, Segoe UI, Helvetica Neue, sans-serif"
_PLOTLY_COLORWAY = [
    "#E10600",
    "#FFD323",
    "#171411",
    "#A80C13",
    "#1E63D7",
    "#F97316",
    "#0F766E",
    "#6D28D9",
]
_PLOTLY_SURFACE_SCALE = [
    [0.0, "#163C8C"],
    [0.22, "#1E67D8"],
    [0.48, "#F8F3E7"],
    [0.72, "#FFD323"],
    [1.0, "#E10600"],
]
_PLOTLY_DENSITY_SCALE = [
    [0.0, "rgba(255, 255, 255, 0.0)"],
    [0.14, "rgba(255, 212, 71, 0.0)"],
    [0.34, "rgba(255, 212, 71, 0.16)"],
    [0.58, "rgba(255, 170, 0, 0.42)"],
    [0.82, "rgba(220, 61, 38, 0.68)"],
    [1.0, "rgba(120, 0, 0, 0.9)"],
]


def _ordered_level_spacing(x: NDArray) -> float:
    """Smallest positive gap between ordered-category level positions.

    Used by both backends to size exposure bars and axis padding when
    ``SmoothCurve.level_x`` places levels at their fitted x-positions
    rather than at consecutive integers.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 1.0
    diffs = np.diff(np.sort(x))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(diffs.min())


def _ordered_level_step(x: NDArray) -> float:
    """Median positive spacing of ordered level positions (1.0 when unknown)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        return 1.0
    diffs = np.diff(np.sort(arr))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _special_level_positions(
    level_x: NDArray, n_specials: int, *, gap_steps: float = 2.0
) -> NDArray:
    """X positions for free (special) levels, set off after the ordered levels.

    Each special takes one level-step, starting ``gap_steps`` steps past the
    last ordered level so the detached points read as a separate block.  With
    the default ``level_x = 0..K-1`` spacing this puts them on the integer
    positions ``K+1, K+2, ...``, one empty slot clear of the curve.
    """
    if n_specials <= 0:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(level_x, dtype=np.float64)
    step = _ordered_level_step(arr)
    start = (float(arr.max()) if arr.size else 0.0) + gap_steps * step
    return start + step * np.arange(n_specials, dtype=np.float64)


def _level_positions_with_specials(
    level_x: NDArray, level_is_special: NDArray | None, n_levels: int
) -> NDArray:
    """Positions for every displayed level, ordered levels first.

    ``level_x`` is ``SmoothCurve.level_x`` — the ordered levels only — and
    ``level_is_special`` is the mask parallel to ``TermInference.levels``.
    The result is ``n_levels`` long so callers can zip it against ``levels``
    without plotly truncating to ``min(len(x), len(y))`` and dropping the
    specials from markers, bars and tick labels.
    """
    arr = np.asarray(level_x, dtype=np.float64)
    mask = (
        np.zeros(n_levels, dtype=bool)
        if level_is_special is None
        else np.asarray(level_is_special, dtype=bool)
    )
    if mask.size != n_levels:
        raise ValueError(f"level_is_special has {mask.size} entries for {n_levels} levels.")
    n_ordered = int((~mask).sum())
    if arr.size != n_ordered:
        raise ValueError(
            f"level_x carries {arr.size} positions for {n_ordered} ordered levels; "
            "it must cover the ordered levels only."
        )
    positions = np.empty(n_levels, dtype=np.float64)
    positions[~mask] = arr
    positions[mask] = _special_level_positions(arr, int(mask.sum()))
    return positions


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


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to an rgba string."""
    color = color.lstrip("#")
    if len(color) != 6:
        return color
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _apply_plotly_theme(
    fig,
    *,
    height: int | None = None,
    hovermode: str = "closest",
    showlegend: bool = True,
    legend_orientation: str | None = "h",
    legend_y: float = -0.08,
    legend_x: float = 0.5,
    margin: dict | None = None,
) -> None:
    """Apply a stronger default Plotly visual language."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=_PLOTLY_PAPER,
        plot_bgcolor=_PLOTLY_PANEL,
        colorway=_PLOTLY_COLORWAY,
        font=dict(family=_PLOTLY_FONT, size=13, color=_PLOTLY_TEXT),
        hoverlabel=dict(
            bgcolor="rgba(255, 253, 248, 0.96)",
            bordercolor="rgba(24, 33, 43, 0.14)",
            font=dict(family=_PLOTLY_FONT, size=12, color=_PLOTLY_TEXT),
        ),
        hovermode=hovermode,
        showlegend=showlegend,
    )
    if height is not None:
        fig.update_layout(height=height)
    if legend_orientation is not None:
        fig.update_layout(
            legend=dict(
                orientation=legend_orientation,
                yanchor="top",
                y=legend_y,
                xanchor="center",
                x=legend_x,
                bgcolor="rgba(255, 253, 248, 0.88)",
                bordercolor="rgba(24, 33, 43, 0.10)",
                borderwidth=1,
            )
        )
    if margin is not None:
        fig.update_layout(margin=margin)

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor=_PLOTLY_AXIS,
        gridcolor=_PLOTLY_GRID,
        zeroline=False,
        ticks="outside",
        tickcolor=_PLOTLY_AXIS,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor=_PLOTLY_AXIS,
        gridcolor=_PLOTLY_GRID,
        zeroline=False,
        ticks="outside",
        tickcolor=_PLOTLY_AXIS,
    )


def _apply_plotly_scene_style(
    fig,
    *,
    x_title: str,
    y_title: str,
    z_title: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> None:
    """Apply consistent 3D styling for Plotly scene plots."""
    scene = dict(
        bgcolor=_PLOTLY_PAPER,
        aspectmode="manual",
        aspectratio=dict(x=1.18, y=1.0, z=0.72),
        camera=dict(eye=dict(x=1.55, y=1.35, z=0.88)),
        xaxis=dict(
            title=x_title,
            showbackground=True,
            backgroundcolor=_PLOTLY_PANEL,
            gridcolor=_PLOTLY_GRID,
            linecolor=_PLOTLY_AXIS,
            zerolinecolor=_PLOTLY_GRID,
            ticks="outside",
            showspikes=False,
        ),
        yaxis=dict(
            title=y_title,
            showbackground=True,
            backgroundcolor=_PLOTLY_PANEL,
            gridcolor=_PLOTLY_GRID,
            linecolor=_PLOTLY_AXIS,
            zerolinecolor=_PLOTLY_GRID,
            ticks="outside",
            showspikes=False,
        ),
        zaxis=dict(
            title=z_title,
            showbackground=True,
            backgroundcolor="#F1ECE1",
            gridcolor=_PLOTLY_GRID,
            linecolor=_PLOTLY_AXIS,
            zerolinecolor=_PLOTLY_GRID,
            ticks="outside",
            showspikes=False,
        ),
    )
    if x_range is not None:
        scene["xaxis"]["range"] = list(x_range)
    if y_range is not None:
        scene["yaxis"]["range"] = list(y_range)
    if z_range is not None:
        scene["zaxis"]["range"] = list(z_range)
    fig.update_layout(scene=scene)


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
