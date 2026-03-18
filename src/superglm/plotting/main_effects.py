"""Main-effect relativity plotting (spline, numeric, categorical panels)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from numpy.typing import NDArray

from superglm.plotting.common import (
    _EXP_EDGE,
    _EXP_EDGE_LW,
    _EXP_FILL,
    _KNOT_COLOR,
    _LINE_COLOR,
    _LINE_WIDTH,
    _PW_ALPHA,
    _PW_EDGE_ALPHA,
    _PW_EDGE_LW,
    _PW_FILL,
    _REF_COLOR,
    _REF_LW,
    _SIM_ALPHA,
    _SIM_EDGE_ALPHA,
    _SIM_EDGE_LW,
    _SIM_FILL,
    _exposure_kde,
    _make_continuous_figure,
)

if TYPE_CHECKING:
    from superglm.inference import TermInference


def plot_relativities(
    terms: list[TermInference],
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    sample_weight: NDArray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    with_ci: bool = True,
    interval: str | None = "pointwise",
    show_exposure: bool = True,
    show_knots: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
) -> Figure:
    """Create a grid of relativity plots from ``TermInference`` objects.

    Parameters
    ----------
    terms : list[TermInference]
        Per-term inference objects from :meth:`SuperGLM.term_inference`.
    X : DataFrame, optional
        Training data for exposure density overlays.
    exposure : array-like, optional
        Exposure / frequency weights.
    sample_weight : array-like, optional
        Alias for *exposure*.
    ncols : int
        Number of subplot columns (default 2).
    figsize : tuple, optional
        Figure size.  Auto-sized if *None*.
    with_ci : bool
        When *False*, forces ``interval=None`` (no bands).
    interval : {"pointwise", "simultaneous", "both", None}
        ``"pointwise"``: orange CI band only.
        ``"simultaneous"``: blue simultaneous band only.
        ``"both"``: nested (simultaneous outside, pointwise inside).
        ``None``: no uncertainty bands.
        For categorical/numeric terms, ``"simultaneous"`` and ``"both"``
        silently fall back to pointwise CI.
    show_exposure : bool
        Show exposure density strip below continuous panels (default *True*).
    show_knots : bool
        Show interior knot positions as minor x-axis ticks (default *False*).
    title, subtitle : str, optional
        Figure-level title and subtitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if exposure is not None and sample_weight is not None:
        raise TypeError(
            "plot_relativities() received both 'exposure' and 'sample_weight'. "
            "Use only 'sample_weight'; 'exposure' is a backward-compatible alias."
        )
    if sample_weight is not None:
        exposure = sample_weight

    if not with_ci:
        interval = None

    return _plot_relativities_new(
        terms,
        X=X,
        exposure=exposure,
        ncols=ncols,
        figsize=figsize,
        interval=interval,
        show_exposure=show_exposure,
        show_knots=show_knots,
        title=title,
        subtitle=subtitle,
    )


# ── New TermInference-based plotting ─────────────────────────────


def _plot_spline_panel(ax, ti: TermInference, interval: str | None, show_knots: bool):
    """Render a spline/polynomial relativity panel."""
    x = ti.x
    rel = ti.relativity

    ax.axhline(1.0, linestyle="--", linewidth=_REF_LW, color=_REF_COLOR, zorder=0)

    # Simultaneous band (outer)
    if interval in ("simultaneous", "both") and ti.ci_lower_simultaneous is not None:
        sim_lo = ti.ci_lower_simultaneous
        sim_hi = ti.ci_upper_simultaneous
        ax.fill_between(
            x,
            sim_lo,
            sim_hi,
            color=_SIM_FILL,
            alpha=_SIM_ALPHA,
            label="95% simultaneous band",
            zorder=1,
        )
        ax.plot(
            x,
            sim_lo,
            color=_SIM_FILL,
            linestyle="--",
            linewidth=_SIM_EDGE_LW,
            alpha=_SIM_EDGE_ALPHA,
            zorder=2,
        )
        ax.plot(
            x,
            sim_hi,
            color=_SIM_FILL,
            linestyle="--",
            linewidth=_SIM_EDGE_LW,
            alpha=_SIM_EDGE_ALPHA,
            zorder=2,
        )

    # Pointwise band (inner)
    if interval in ("pointwise", "both") and ti.ci_lower is not None:
        pw_lo = ti.ci_lower
        pw_hi = ti.ci_upper
        ax.fill_between(
            x,
            pw_lo,
            pw_hi,
            color=_PW_FILL,
            alpha=_PW_ALPHA,
            label="95% pointwise CI",
            zorder=3,
        )
        ax.plot(
            x,
            pw_lo,
            color=_PW_FILL,
            linestyle="--",
            linewidth=_PW_EDGE_LW,
            alpha=_PW_EDGE_ALPHA,
            zorder=4,
        )
        ax.plot(
            x,
            pw_hi,
            color=_PW_FILL,
            linestyle="--",
            linewidth=_PW_EDGE_LW,
            alpha=_PW_EDGE_ALPHA,
            zorder=4,
        )

    ax.plot(x, rel, color=_LINE_COLOR, linewidth=_LINE_WIDTH, label="Relativity", zorder=5)

    if show_knots and ti.spline is not None and ti.spline.interior_knots.size > 0:
        knots = ti.spline.interior_knots
        ax.xaxis.set_minor_locator(FixedLocator(knots))
        ax.tick_params(
            axis="x",
            which="minor",
            length=4,
            width=1.0,
            color=_KNOT_COLOR,
            direction="in",
        )

    ax.set_title(ti.name, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_density_strip(
    ax_d,
    feature_name: str,
    X: pd.DataFrame,
    exposure: NDArray,
    x_grid: NDArray,
    show_knots: bool,
    knots: NDArray | None,
):
    """Render the exposure density strip beneath a spline panel."""
    x_vals = X[feature_name].to_numpy(dtype=np.float64)
    density = _exposure_kde(x_vals, exposure, x_grid)

    ax_d.fill_between(x_grid, 0.0, density, color=_EXP_FILL, alpha=0.95, linewidth=0)
    ax_d.plot(x_grid, density, color=_EXP_EDGE, linewidth=_EXP_EDGE_LW)
    ax_d.set_ylim(0.0, 1.05)
    ax_d.set_yticks([])
    ax_d.set_xlabel(feature_name)

    if show_knots and knots is not None and len(knots) > 0:
        ax_d.xaxis.set_minor_locator(FixedLocator(knots))
        ax_d.tick_params(
            axis="x",
            which="minor",
            length=4,
            width=1.0,
            color=_KNOT_COLOR,
            direction="in",
        )

    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.spines["left"].set_visible(False)
    ax_d.grid(False)


def _plot_numeric_panel_continuous(
    ax,
    ti: TermInference,
    interval: str | None,
    x_grid: NDArray,
):
    """Render a numeric term as a flat line + flat CI band over the feature range.

    Uses the same continuous visual language as spline panels: horizontal
    relativity line across ``x_grid`` with optional constant CI band(s).
    """
    rel = float(np.asarray(ti.relativity).ravel()[0])
    x = x_grid

    # Reference line
    ax.axhline(1.0, linestyle="--", color=_REF_COLOR, linewidth=_REF_LW, zorder=0)

    # CI band (flat, constant across x range)
    if interval is not None and ti.ci_lower is not None:
        ci_lo = float(np.asarray(ti.ci_lower).ravel()[0])
        ci_hi = float(np.asarray(ti.ci_upper).ravel()[0])
        ax.fill_between(
            x,
            ci_lo,
            ci_hi,
            color=_PW_FILL,
            alpha=_PW_ALPHA,
            linewidth=0,
            label="Pointwise 95% CI",
        )
        ax.plot(
            x,
            np.full_like(x, ci_lo),
            color=_PW_FILL,
            alpha=_PW_EDGE_ALPHA,
            linewidth=_PW_EDGE_LW,
            linestyle="--",
        )
        ax.plot(
            x,
            np.full_like(x, ci_hi),
            color=_PW_FILL,
            alpha=_PW_EDGE_ALPHA,
            linewidth=_PW_EDGE_LW,
            linestyle="--",
        )

    # Flat relativity line
    ax.plot(x, np.full_like(x, rel), color=_LINE_COLOR, linewidth=_LINE_WIDTH, label="Relativity")

    ax.set_ylabel("Relativity")
    ax.set_title(ti.name, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_term(
    ti: TermInference,
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    interval: str | None = "pointwise",
    show_exposure: bool = True,
    show_knots: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> Figure:
    """Plot a single term's relativity.

    This is the core single-term plotting function.  All term types
    (spline, polynomial, numeric, categorical) are handled.

    Parameters
    ----------
    ti : TermInference
        Inference result from :meth:`SuperGLM.term_inference`.
    X : DataFrame, optional
        Training data for exposure overlays.
    exposure : array-like, optional
        Exposure / frequency weights.
    interval : {"pointwise", "simultaneous", "both", None}
        Band style.  For categoricals, simultaneous/both fall back to pointwise.
    show_exposure : bool
        Show exposure distribution (density strip for continuous, vertical
        bars for categorical).
    show_knots : bool
        Show interior knot ticks (spline only).
    figsize : tuple, optional
        Figure size.
    title, subtitle : str, optional
        Title and subtitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    weighted = exposure is not None
    if exposure is not None:
        exposure = np.asarray(exposure, dtype=np.float64)
    elif X is not None and show_exposure:
        # Fall back to uniform weights (observation counts) when no
        # sample_weight is provided.
        exposure = np.ones(len(X), dtype=np.float64)

    density_label = "Weight\ndensity" if weighted else "Obs.\ndensity"
    weight_label = "Weight" if weighted else "Count"
    has_density = show_exposure and X is not None and exposure is not None

    if ti.kind in ("spline", "polynomial"):
        needs_strip = has_density and ti.name in X.columns
        fig, ax, ax_den = _make_continuous_figure(needs_strip, figsize)

        _plot_spline_panel(ax, ti, interval, show_knots)
        ax.set_ylabel("Relativity")

        if ax_den is not None:
            knots = ti.spline.interior_knots if ti.spline is not None else None
            _plot_density_strip(ax_den, ti.name, X, exposure, ti.x, show_knots, knots)
            ax_den.set_ylabel(density_label, fontsize=8)

    elif ti.kind == "numeric":
        needs_strip = has_density and ti.name in X.columns
        if X is not None and ti.name in X.columns:
            x_vals = X[ti.name].to_numpy(dtype=np.float64)
            x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
        else:
            x_grid = np.linspace(0.0, 1.0, 200)

        fig, ax, ax_den = _make_continuous_figure(needs_strip, figsize)
        _plot_numeric_panel_continuous(ax, ti, interval, x_grid)

        if ax_den is not None:
            _plot_density_strip(ax_den, ti.name, X, exposure, x_grid, False, None)
            ax_den.set_ylabel(density_label, fontsize=8)

    elif ti.kind == "categorical" and ti.smooth_curve is not None:
        if figsize is None:
            figsize = (max(6, len(ti.levels) * 0.9 + 1.5), 4.5)
        fig, ax = plt.subplots(figsize=figsize)
        _plot_ordered_spline_panel(
            ax,
            ti,
            interval,
            X=X,
            exposure=exposure if has_density else None,
            weight_label=weight_label,
        )

    elif ti.kind == "categorical":
        if figsize is None:
            figsize = (max(5, len(ti.levels) * 0.9 + 1.5), 4.5)
        fig, ax = plt.subplots(figsize=figsize)
        _plot_categorical_panel_vertical(
            ax,
            ti,
            interval,
            X=X,
            exposure=exposure if has_density else None,
            weight_label=weight_label,
        )

    else:
        if figsize is None:
            figsize = (7, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Unknown term kind: {ti.kind!r}", transform=ax.transAxes, ha="center")

    # ── Legend ──
    all_axes = fig.get_axes()
    legend_handles = []
    legend_labels = []
    for a in all_axes:
        for h, lab in zip(*a.get_legend_handles_labels()):
            if lab not in legend_labels:
                legend_handles.append(h)
                legend_labels.append(lab)

    if (
        show_knots
        and ti.kind in ("spline", "polynomial")
        and ti.spline is not None
        and ti.spline.interior_knots.size > 0
    ):
        knot_handle = Line2D(
            [0],
            [0],
            color=_KNOT_COLOR,
            marker="|",
            linestyle="None",
            markersize=9,
            markeredgewidth=1.1,
            label="Interior knots",
        )
        legend_handles.append(knot_handle)
        legend_labels.append("Interior knots")

    # tight_layout is incompatible with explicit GridSpec — only call for plain subplots
    has_gs = any(
        hasattr(ax, "get_gridspec") and ax.get_gridspec() is not None for ax in fig.get_axes()
    )

    has_title = title is not None
    has_subtitle = subtitle is not None
    has_legend = bool(legend_handles)

    layout_top = 0.96
    title_y = None
    subtitle_y = None
    legend_y = None

    if has_title and has_subtitle and has_legend:
        layout_top = 0.72
        title_y = 0.988
        subtitle_y = 0.910
        legend_y = 0.860
    elif has_title and has_legend:
        layout_top = 0.82
        title_y = 0.982
        legend_y = 0.915
    elif has_title and has_subtitle:
        layout_top = 0.78
        title_y = 0.988
        subtitle_y = 0.916
    elif has_legend and has_subtitle:
        layout_top = 0.83
        subtitle_y = 0.958
        legend_y = 0.915
    elif has_title:
        layout_top = 0.88
        title_y = 0.982
    elif has_subtitle:
        layout_top = 0.89
        subtitle_y = 0.960
    elif has_legend:
        layout_top = 0.90
        legend_y = 0.965

    if has_gs:
        fig.subplots_adjust(top=layout_top)
    else:
        fig.tight_layout(rect=[0, 0, 1, layout_top])

    if has_legend:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=min(len(legend_handles), 4),
            frameon=False,
            fontsize=9,
        )

    # ── Title / subtitle ──
    if has_title and title_y is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=title_y)
    if has_subtitle and subtitle_y is not None:
        fig.text(0.5, subtitle_y, subtitle, ha="center", fontsize=10.5, color="#444444")

    return fig


def _plot_ordered_spline_panel(
    ax,
    ti: TermInference,
    interval: str | None,
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    weight_label: str = "Weight",
):
    """Render an OrderedCategorical(spline) panel.

    Levels at evenly-spaced integer positions with a smooth PCHIP
    interpolation through the fitted relativities, plus per-level
    error bars and exposure bars.
    """
    levels = list(ti.levels)
    level_rel = np.asarray(ti.relativity)
    n_levels = len(levels)
    x_pos = np.arange(n_levels, dtype=float)

    # Exposure bars in background
    if exposure is not None and X is not None and ti.name in X.columns:
        level_exp = (
            pd.DataFrame({"level": X[ti.name], "exposure": exposure})
            .groupby("level", sort=False)["exposure"]
            .sum()
        )
        exp_vals = np.array([level_exp.get(lv, 0.0) for lv in levels])
        ax2 = ax.twinx()
        ax2.bar(
            x_pos,
            exp_vals,
            width=0.6,
            color=_EXP_FILL,
            edgecolor=_EXP_EDGE,
            linewidth=_EXP_EDGE_LW,
            alpha=1.0,
            zorder=0,
            label=weight_label,
        )
        ymax = float(exp_vals.max()) if exp_vals.size else 0.0
        ax2.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
        ax2.set_ylabel(weight_label, color=_EXP_EDGE)
        ax2.tick_params(axis="y", colors=_EXP_EDGE, labelsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(_EXP_EDGE)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.axhline(1.0, linestyle="--", linewidth=_REF_LW, color=_REF_COLOR, zorder=0)

    # Smooth interpolation through level relativities (PCHIP)
    if n_levels >= 2:
        from scipy.interpolate import PchipInterpolator

        pchip = PchipInterpolator(x_pos, level_rel)
        x_fine = np.linspace(x_pos[0], x_pos[-1], 200)
        ax.plot(
            x_fine,
            pchip(x_fine),
            color=_LINE_COLOR,
            linewidth=_LINE_WIDTH,
            alpha=0.6,
            zorder=4,
        )

    # Per-level dots with error bars
    if interval is not None and ti.ci_lower is not None:
        ci_lo = np.asarray(ti.ci_lower)
        ci_hi = np.asarray(ti.ci_upper)
        ax.errorbar(
            x_pos,
            level_rel,
            yerr=[level_rel - ci_lo, ci_hi - level_rel],
            fmt="o",
            color=_LINE_COLOR,
            markersize=7,
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            label="Relativity",
            zorder=5,
        )
    else:
        ax.scatter(
            x_pos,
            level_rel,
            color=_LINE_COLOR,
            s=50,
            zorder=5,
            label="Relativity",
        )

    ax.set_xticks(x_pos)
    rot = 45 if n_levels > 8 else 0
    ha = "right" if rot else "center"
    ax.set_xticklabels(levels, rotation=rot, ha=ha, fontsize=8)
    ax.set_xlim(-0.5, n_levels - 0.5)
    ax.set_ylabel("Relativity")
    ax.set_title(ti.name, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_categorical_panel_vertical(
    ax,
    ti: TermInference,
    interval: str | None,
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    weight_label: str = "Weight",
):
    """Render a categorical panel with vertical orientation.

    Levels on x-axis, relativity on y-axis.  Optional exposure bars
    in the background.
    """
    levels = list(ti.levels)
    rel = np.asarray(ti.relativity)
    x_pos = np.arange(len(levels))

    # Exposure bars in background
    if exposure is not None and X is not None and ti.name in X.columns:
        level_exp = (
            pd.DataFrame({"level": X[ti.name], "exposure": exposure})
            .groupby("level", sort=False)["exposure"]
            .sum()
        )
        exp_vals = np.array([level_exp.get(lv, 0.0) for lv in levels])
        ax2 = ax.twinx()
        ax2.bar(
            x_pos,
            exp_vals,
            width=0.6,
            color=_EXP_FILL,
            edgecolor=_EXP_EDGE,
            linewidth=_EXP_EDGE_LW,
            alpha=1.0,
            zorder=0,
            label=weight_label,
        )
        ymax = float(exp_vals.max()) if exp_vals.size else 0.0
        ax2.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
        ax2.set_ylabel(weight_label, color=_EXP_EDGE)
        ax2.tick_params(axis="y", colors=_EXP_EDGE, labelsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(_EXP_EDGE)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    # Relativity line + markers + error bars
    ax.plot(x_pos, rel, color=_LINE_COLOR, linewidth=_LINE_WIDTH, alpha=0.6, zorder=4)
    if interval is not None and ti.ci_lower is not None:
        ci_lo = np.asarray(ti.ci_lower)
        ci_hi = np.asarray(ti.ci_upper)
        ax.errorbar(
            x_pos,
            rel,
            yerr=[rel - ci_lo, ci_hi - rel],
            fmt="o",
            color=_LINE_COLOR,
            markersize=7,
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            label="Relativity",
            zorder=5,
        )
    else:
        ax.scatter(x_pos, rel, color=_LINE_COLOR, s=50, zorder=5, label="Relativity")

    ax.axhline(1.0, linestyle="--", color=_REF_COLOR, linewidth=_REF_LW, zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Relativity")
    ax.set_title(ti.name, fontweight="bold")
    ax.grid(alpha=0.22, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_relativities_new(
    terms: list[TermInference],
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    interval: str | None = "pointwise",
    show_exposure: bool = True,
    show_knots: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
) -> Figure:
    """TermInference-based relativity grid with the new visual language."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(terms)
    if n == 0:
        fig, _ = plt.subplots()
        return fig

    weighted = exposure is not None
    if exposure is not None:
        exposure = np.asarray(exposure, dtype=np.float64)
    elif X is not None and show_exposure:
        exposure = np.ones(len(X), dtype=np.float64)

    density_label = "Weight\ndensity" if weighted else "Obs.\ndensity"
    weight_label = "Weight" if weighted else "Count"
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    has_density = show_exposure and X is not None and exposure is not None
    _CONTINUOUS_KINDS = ("spline", "polynomial", "numeric")
    any_density = has_density and any(
        ti.kind in _CONTINUOUS_KINDS and ti.name in X.columns for ti in terms
    )

    if any_density:
        # 2-row layout: main panel + density strip per row
        if figsize is None:
            figsize = (5 * ncols, 5.2 * nrows + 0.5)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            nrows * 2,
            ncols,
            figure=fig,
            height_ratios=[4.2, 1.0] * nrows,
            hspace=0.16,
        )
        fig.subplots_adjust(top=0.88 if title else 0.95, wspace=0.26)

        main_axes = []
        density_axes = []
        for idx in range(n):
            r, c = divmod(idx, ncols)
            ti = terms[idx]
            uses_strip = ti.kind in _CONTINUOUS_KINDS and ti.name in X.columns
            if uses_strip:
                ax_main = fig.add_subplot(gs[r * 2, c])
                ax_den = fig.add_subplot(gs[r * 2 + 1, c])
                # Keep x labels on the main panel; the strip just shows shape/support.
                ax_den.tick_params(axis="x", labelbottom=False)
                ax_main.set_zorder(ax_den.get_zorder() + 1)
                ax_main.patch.set_visible(False)
                ax_main.tick_params(axis="x", labelbottom=True, pad=-2)
            else:
                # No density strip — span both rows to reclaim the space
                ax_main = fig.add_subplot(gs[r * 2 : r * 2 + 2, c])
                ax_den = None
            main_axes.append(ax_main)
            density_axes.append(ax_den)

        # Hide unused grid cells
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            fig.add_subplot(gs[r * 2 : r * 2 + 2, c]).set_visible(False)
    else:
        # Simple single-row layout
        if figsize is None:
            figsize = (5 * ncols, 3.5 * nrows)
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.subplots_adjust(top=0.88 if title else 0.95)
        main_axes = [axes_arr[idx // ncols][idx % ncols] for idx in range(n)]
        density_axes = [None] * n

        for idx in range(n, nrows * ncols):
            axes_arr[idx // ncols][idx % ncols].set_visible(False)

    # ── Render each panel ──
    for idx, ti in enumerate(terms):
        ax = main_axes[idx]
        ax_den = density_axes[idx]

        if ti.kind in ("spline", "polynomial"):
            _plot_spline_panel(ax, ti, interval, show_knots)
            if idx % ncols == 0:
                ax.set_ylabel("Relativity")

            if ax_den is not None:
                knots = ti.spline.interior_knots if ti.spline is not None else None
                _plot_density_strip(ax_den, ti.name, X, exposure, ti.x, show_knots, knots)
                if idx % ncols == 0:
                    ax_den.set_ylabel(density_label, fontsize=8)
                ax_den.set_xlabel("")

        elif ti.kind == "categorical" and ti.smooth_curve is not None:
            _plot_ordered_spline_panel(
                ax,
                ti,
                interval,
                X=X,
                exposure=exposure if has_density else None,
                weight_label=weight_label,
            )

        elif ti.kind == "categorical":
            _plot_categorical_panel_vertical(
                ax,
                ti,
                interval,
                X=X,
                exposure=exposure if has_density else None,
                weight_label=weight_label,
            )

        elif ti.kind == "numeric":
            if X is not None and ti.name in X.columns:
                x_vals = X[ti.name].to_numpy(dtype=np.float64)
                x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
            else:
                x_grid = np.linspace(0.0, 1.0, 200)
            _plot_numeric_panel_continuous(ax, ti, interval, x_grid)
            if idx % ncols == 0:
                ax.set_ylabel("Relativity")

            if ax_den is not None:
                _plot_density_strip(ax_den, ti.name, X, exposure, x_grid, False, None)
                if idx % ncols == 0:
                    ax_den.set_ylabel(density_label, fontsize=8)
                ax_den.set_xlabel("")

        else:
            ax.set_visible(False)

    # ── Figure-level legend ──
    legend_handles = []
    legend_labels = []
    for ax in main_axes:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if li not in legend_labels:
                legend_handles.append(hi)
                legend_labels.append(li)

    if show_knots and any(
        ti.spline is not None and ti.spline.interior_knots.size > 0
        for ti in terms
        if ti.kind in ("spline", "polynomial")
    ):
        knot_handle = Line2D(
            [0],
            [0],
            color=_KNOT_COLOR,
            marker="|",
            linestyle="None",
            markersize=9,
            markeredgewidth=1.1,
            label="Interior knots",
        )
        legend_handles.append(knot_handle)
        legend_labels.append("Interior knots")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93 if title else 0.99),
            ncol=min(len(legend_handles), 4),
            frameon=False,
            fontsize=9,
        )

    # ── Title / subtitle ──
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(
            0.5,
            0.935 if title else 0.97,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#444444",
        )

    if not any_density:
        fig.tight_layout(rect=[0, 0, 1, 0.93 if title else 0.95])
    return fig
