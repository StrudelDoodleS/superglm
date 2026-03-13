"""Relativity plotting for SuperGLM models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superglm.inference import TermInference

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


def _exposure_kde(x_vals, exposure, grid, bw_factor=0.03):
    """Weighted KDE for exposure distribution, returned on *grid*."""
    bw = bw_factor * (grid[-1] - grid[0])
    diff = grid[:, None] - x_vals[None, :]
    kernel = np.exp(-0.5 * (diff / bw) ** 2)
    density = kernel @ exposure
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


def plot_relativities(
    terms,
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
    """Create a grid of relativity plots.

    Accepts either a ``list[TermInference]`` (new path, used by
    ``model.plot_relativities()``) or a ``dict[str, DataFrame]`` (legacy
    path, for backward compatibility with ``model.relativities()`` output).

    Parameters
    ----------
    terms : list[TermInference] or dict[str, DataFrame]
        Per-term inference objects **or** legacy relativities dict.
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
        Show exposure density strip below spline panels (default *True*).
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

    if isinstance(terms, dict):
        return _plot_relativities_legacy(
            terms,
            X=X,
            exposure=exposure,
            ncols=ncols,
            figsize=figsize,
            with_ci=with_ci,
        )

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


def _plot_categorical_panel(ax, ti: TermInference, interval: str | None):
    """Render a categorical relativity panel (horizontal bars)."""
    levels = ti.levels
    rel = ti.relativity

    ax.barh(levels, rel, color=_CAT_BAR_COLOR)

    # Simultaneous/both silently falls back to pointwise for categoricals
    if interval is not None and ti.ci_lower is not None:
        ci_lo = ti.ci_lower
        ci_hi = ti.ci_upper
        ax.errorbar(
            rel,
            levels,
            xerr=[rel - ci_lo, ci_hi - rel],
            fmt="none",
            ecolor="black",
            capsize=3,
            label="95% CI",
        )

    ax.axvline(1.0, linestyle="--", color=_REF_COLOR, linewidth=_REF_LW)
    ax.set_title(ti.name, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_numeric_panel(ax, ti: TermInference, interval: str | None):
    """Render a numeric (single per-unit bar) relativity panel."""
    label = "per_unit"
    rel = ti.relativity

    ax.barh([label], rel, color=_CAT_BAR_COLOR)

    # Simultaneous/both silently falls back to pointwise for numerics
    if interval is not None and ti.ci_lower is not None:
        ci_lo = ti.ci_lower
        ci_hi = ti.ci_upper
        ax.errorbar(
            rel,
            [label],
            xerr=[rel - ci_lo, ci_hi - rel],
            fmt="none",
            ecolor="black",
            capsize=3,
            label="95% CI",
        )

    ax.axvline(1.0, linestyle="--", color=_REF_COLOR, linewidth=_REF_LW)
    ax.set_title(ti.name, fontweight="bold")
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

    if exposure is not None:
        exposure = np.asarray(exposure, dtype=np.float64)

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    has_density = show_exposure and X is not None and exposure is not None
    any_density = has_density and any(
        ti.kind in ("spline", "polynomial") and ti.name in X.columns for ti in terms
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
            hspace=0.06,
        )
        fig.subplots_adjust(top=0.88 if title else 0.95, wspace=0.26)

        main_axes = []
        density_axes = []
        for idx in range(n):
            r, c = divmod(idx, ncols)
            ti = terms[idx]
            uses_strip = ti.kind in ("spline", "polynomial") and ti.name in X.columns
            ax_main = fig.add_subplot(gs[r * 2, c])
            if uses_strip:
                ax_den = fig.add_subplot(gs[r * 2 + 1, c], sharex=ax_main)
                # Hide x labels on main panel — density strip shows them
                plt.setp(ax_main.get_xticklabels(), visible=False)
            else:
                # No density strip — merge the two rows visually
                ax_den = fig.add_subplot(gs[r * 2 + 1, c])
                ax_den.set_visible(False)
            main_axes.append(ax_main)
            density_axes.append(ax_den)

        # Hide unused grid cells
        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            fig.add_subplot(gs[r * 2, c]).set_visible(False)
            fig.add_subplot(gs[r * 2 + 1, c]).set_visible(False)
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

            # Density strip (only when ax_den is visible — set up in layout phase)
            if ax_den is not None and ax_den.get_visible():
                knots = ti.spline.interior_knots if ti.spline is not None else None
                _plot_density_strip(ax_den, ti.name, X, exposure, ti.x, show_knots, knots)
                if idx % ncols == 0:
                    ax_den.set_ylabel("Exposure\ndensity", fontsize=8)

        elif ti.kind == "categorical":
            _plot_categorical_panel(ax, ti, interval)

        elif ti.kind == "numeric":
            _plot_numeric_panel(ax, ti, interval)

        else:
            ax.set_visible(False)
            if ax_den is not None:
                ax_den.set_visible(False)

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


# ── Legacy dict-based plotting (backward compatibility) ──────────


def _plot_relativities_legacy(
    relativities: dict[str, pd.DataFrame],
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    with_ci: bool = True,
) -> Figure:
    """Original dict-based relativity grid (steelblue, twin-axis exposure)."""
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
                    grid,
                    0,
                    density,
                    color="lightgrey",
                    alpha=0.4,
                    label="Exposure",
                )
                ax2.set_ylim(0, 1.3)
                ax2.set_yticks([])
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
            if with_ci and "se_log_relativity" in df.columns:
                se = df["se_log_relativity"].to_numpy()
                log_rel = df["log_relativity"].to_numpy()
                ci_lo = np.exp(log_rel - 1.96 * se)
                ci_hi = np.exp(log_rel + 1.96 * se)
                ax.fill_between(
                    df["x"],
                    ci_lo,
                    ci_hi,
                    alpha=0.25,
                    color="steelblue",
                    label="95% CI",
                )
            ax.plot(
                df["x"],
                df["relativity"],
                color="steelblue",
                linewidth=1.5,
                label="Relativity",
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
                ax2.barh(df["level"], exp_vals, color="lightgrey", alpha=0.4, label="Exposure")
                ax2.set_yticks([])
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
            ax.barh(df["level"], df["relativity"], color="steelblue", label="Relativity")
            if with_ci and "se_log_relativity" in df.columns:
                se = df["se_log_relativity"].to_numpy()
                log_rel = df["log_relativity"].to_numpy()
                rel = df["relativity"].to_numpy()
                ci_lo = np.exp(log_rel - 1.96 * se)
                ci_hi = np.exp(log_rel + 1.96 * se)
                ax.errorbar(
                    rel,
                    df["level"],
                    xerr=[rel - ci_lo, ci_hi - rel],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    label="95% CI",
                )
            ax.axvline(1.0, linestyle="--", color="grey", linewidth=0.8)

        elif "label" in df.columns:
            # Numeric — single horizontal bar
            ax.barh(df["label"], df["relativity"], color="steelblue")
            if with_ci and "se_log_relativity" in df.columns:
                se = df["se_log_relativity"].to_numpy()
                log_rel = df["log_relativity"].to_numpy()
                rel = df["relativity"].to_numpy()
                ci_lo = np.exp(log_rel - 1.96 * se)
                ci_hi = np.exp(log_rel + 1.96 * se)
                ax.errorbar(
                    rel,
                    df["label"],
                    xerr=[rel - ci_lo, ci_hi - rel],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    label="95% CI",
                )
            ax.axvline(1.0, linestyle="--", color="grey", linewidth=0.8)

        ax.set_title(name)
        if idx % ncols == 0:
            ax.set_ylabel("Relativity")

    # Single legend for the whole figure
    if show_exposure or with_ci:
        handles, labels = [], []
        for ax_row in axes:
            for ax in ax_row:
                h, lab = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, lab
                    break
            if handles:
                break
        for ax_row in axes:
            for ax in ax_row:
                for child in ax.figure.get_axes():
                    h2, l2 = child.get_legend_handles_labels()
                    for h, lab in zip(h2, l2):
                        if lab == "Exposure" and lab not in labels:
                            handles.append(h)
                            labels.append(lab)
                            break
                if "Exposure" in labels:
                    break
            if "Exposure" in labels:
                break
        if handles:
            fig.legend(handles, labels, loc="lower right", fontsize=9, framealpha=0.8)

    for idx in range(len(names), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig


# ── Interaction plotting ──────────────────────────────────────────


def plot_interaction(
    model,
    name: str,
    *,
    engine: str = "matplotlib",
    with_ci: bool = True,
    figsize: tuple[float, float] | None = None,
    colormap: str | None = None,
    show_contours: bool = True,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
):
    """Plot an interaction surface/effect.

    Parameters
    ----------
    model : SuperGLM
        A fitted model with the named interaction.
    name : str
        Interaction name, e.g. ``"DrivAge:Area"``.
    engine : {"matplotlib", "plotly"}
        Plotting backend.  ``"plotly"`` requires plotly to be installed.
    with_ci : bool
        Show confidence bands where applicable (varying-coefficient only).
    figsize : tuple, optional
        Figure size for matplotlib.
    colormap : str, optional
        Colormap / colorscale name.
    show_contours : bool
        For surface plots: show iso-relativity contour lines on the surface
        (default True).
    X : DataFrame, optional
        Training data. When provided with *exposure*, overlays an
        exposure-weighted density on surface plots (projected on the floor
        for 3D plotly, contour overlay for matplotlib).
    exposure : array-like, optional
        Exposure weights corresponding to rows of *X*.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if name not in model._interaction_specs:
        raise KeyError(
            f"Interaction not found: {name!r}. Available: {list(model._interaction_specs.keys())}"
        )

    raw = model.reconstruct_feature(name)
    ispec = model._interaction_specs[name]
    parent_names = ispec.parent_names

    density_data = None
    if X is not None and exposure is not None:
        exposure = np.asarray(exposure, dtype=np.float64)
        if parent_names[0] in X.columns and parent_names[1] in X.columns:
            density_data = (
                np.asarray(X[parent_names[0]], dtype=np.float64),
                np.asarray(X[parent_names[1]], dtype=np.float64),
                exposure,
            )

    if engine == "matplotlib":
        return _dispatch_mpl(
            raw, name, parent_names, with_ci, figsize, colormap, show_contours, density_data
        )
    elif engine == "plotly":
        return _dispatch_plotly(
            raw, name, parent_names, with_ci, colormap, show_contours, density_data
        )
    else:
        raise ValueError(f"Unknown engine {engine!r}. Use 'matplotlib' or 'plotly'.")


def _dispatch_mpl(raw, name, parent_names, with_ci, figsize, colormap, show_contours, density_data):
    if "per_level" in raw and "x" in raw:
        return _plot_varying_coefficient_mpl(raw, name, parent_names, with_ci, figsize, colormap)
    elif "pairs" in raw:
        return _plot_categorical_heatmap_mpl(raw, name, parent_names, figsize, colormap)
    elif "relativities_per_unit" in raw:
        return _plot_numeric_categorical_bars_mpl(raw, name, parent_names, figsize)
    elif "relativity_per_unit_unit" in raw:
        return _plot_numeric_interaction_bar_mpl(raw, name, parent_names, figsize)
    elif "x1" in raw and "x2" in raw:
        return _plot_surface_mpl(
            raw, name, parent_names, figsize, colormap, show_contours, density_data
        )
    else:
        raise ValueError(f"Cannot determine plot type for interaction {name!r}.")


def _dispatch_plotly(raw, name, parent_names, with_ci, colormap, show_contours, density_data):
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        raise ImportError(
            "plotly is required for engine='plotly'. Install it with: pip install plotly"
        ) from None

    if "per_level" in raw and "x" in raw:
        return _plot_varying_coefficient_plotly(raw, name, parent_names, with_ci, colormap)
    elif "pairs" in raw:
        return _plot_categorical_heatmap_plotly(raw, name, parent_names, colormap)
    elif "relativities_per_unit" in raw:
        return _plot_numeric_categorical_bars_plotly(raw, name, parent_names)
    elif "relativity_per_unit_unit" in raw:
        return _plot_numeric_interaction_bar_plotly(raw, name, parent_names)
    elif "x1" in raw and "x2" in raw:
        return _plot_surface_plotly(raw, name, parent_names, colormap, show_contours, density_data)
    else:
        raise ValueError(f"Cannot determine plot type for interaction {name!r}.")


# ── matplotlib helpers ────────────────────────────────────────────


def _plot_varying_coefficient_mpl(raw, name, parent_names, with_ci, figsize, colormap):
    """Multi-line plot: one curve per categorical level (incl. base)."""
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (8, 5)
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap(colormap or "tab10")
    x = raw["x"]
    levels = raw["levels"]
    base = raw.get("base_level", "")

    # Base level: flat at 1.0 (no interaction effect)
    if base:
        ax.plot(
            x,
            np.ones_like(x),
            color=cmap(0),
            linewidth=1.5,
            linestyle="--",
            label=f"{base} (base)",
        )

    for i, level in enumerate(levels):
        level_data = raw["per_level"][level]
        color = cmap((i + 1) % cmap.N)
        ax.plot(x, level_data["relativity"], color=color, linewidth=1.5, label=level)
        if with_ci and "se_log_relativity" in level_data:
            se = level_data["se_log_relativity"]
            log_rel = level_data["log_relativity"]
            ci_lo = np.exp(log_rel - 1.96 * se)
            ci_hi = np.exp(log_rel + 1.96 * se)
            ax.fill_between(x, ci_lo, ci_hi, alpha=0.15, color=color)

    ax.axhline(1.0, linestyle=":", color="grey", linewidth=0.6, alpha=0.5)
    ax.set_xlabel(parent_names[0])
    ax.set_ylabel("Relativity")
    ax.set_title(name)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_categorical_heatmap_mpl(raw, name, parent_names, figsize, colormap):
    """Annotated heatmap for categorical x categorical interaction (incl. base)."""
    import matplotlib.pyplot as plt

    pairs = raw["pairs"]
    log_rels = raw["log_relativities"]
    rels = raw["relativities"]

    # Full level lists including base (base pairs have relativity=1.0)
    levels1 = raw.get("levels1") or list(dict.fromkeys(p[0] for p in pairs))
    levels2 = raw.get("levels2") or list(dict.fromkeys(p[1] for p in pairs))

    grid = np.zeros((len(levels1), len(levels2)))
    text_grid = np.full_like(grid, "1.00", dtype=object)
    for (l1, l2), key in zip(pairs, log_rels):
        i = levels1.index(l1)
        j = levels2.index(l2)
        grid[i, j] = log_rels[key]
        text_grid[i, j] = f"{rels[key]:.2f}"

    if figsize is None:
        figsize = (max(5, len(levels2) * 1.2), max(4, len(levels1) * 0.8))
    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(grid.min()), abs(grid.max())) or 1.0
    im = ax.imshow(
        grid,
        cmap=colormap or "RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(range(len(levels2)))
    ax.set_xticklabels(levels2)
    ax.set_yticks(range(len(levels1)))
    ax.set_yticklabels(levels1)
    ax.set_xlabel(parent_names[1])
    ax.set_ylabel(parent_names[0])
    ax.set_title(name)

    for i in range(len(levels1)):
        for j in range(len(levels2)):
            ax.text(j, i, text_grid[i, j], ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Log-Relativity")
    fig.tight_layout()
    return fig


def _plot_numeric_categorical_bars_mpl(raw, name, parent_names, figsize):
    """Vertical bar chart: one bar per categorical level (incl. base)."""
    import matplotlib.pyplot as plt

    base = raw.get("base_level", "")
    non_base = raw["levels"]
    levels = [base] + list(non_base) if base else list(non_base)
    rels = (
        [1.0] + [raw["relativities_per_unit"][lv] for lv in non_base]
        if base
        else [raw["relativities_per_unit"][lv] for lv in non_base]
    )
    colors = (
        ["lightgrey"] + ["steelblue"] * len(non_base) if base else ["steelblue"] * len(non_base)
    )

    if figsize is None:
        figsize = (max(4, len(levels) * 0.8), 4)
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(levels, rels, color=colors)
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=0.8)
    ax.set_ylabel(f"Relativity per unit {parent_names[0]}")
    ax.set_title(name)
    fig.tight_layout()
    return fig


def _plot_numeric_interaction_bar_mpl(raw, name, parent_names, figsize):
    """Single vertical bar for numeric x numeric interaction."""
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (3, 4)
    fig, ax = plt.subplots(figsize=figsize)

    rel = raw["relativity_per_unit_unit"]
    label = f"{parent_names[0]} x {parent_names[1]}"
    ax.bar([label], [rel], color="steelblue")
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=0.8)
    ax.set_ylabel("Relativity per unit x unit")
    ax.set_title(name)
    fig.tight_layout()
    return fig


def _plot_surface_mpl(raw, name, parent_names, figsize, colormap, show_contours, density_data):
    """Filled contour plot for polynomial x polynomial interaction."""
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (7, 5)
    fig, ax = plt.subplots(figsize=figsize)

    x1 = raw["x1"]
    x2 = raw["x2"]
    X1, X2 = np.meshgrid(x1, x2)
    Z = raw["relativity"]

    cf = ax.contourf(X1, X2, Z, levels=20, cmap=colormap or "viridis")
    if show_contours:
        ax.contour(X1, X2, Z, levels=20, colors="grey", linewidths=0.3, alpha=0.5)
    fig.colorbar(cf, ax=ax, label="Relativity")

    if density_data is not None:
        d1, d2, w = density_data
        D = _kde_2d(d1, d2, w, x1, x2)
        ax.contour(
            X1,
            X2,
            D,
            levels=5,
            colors="white",
            linewidths=1.0,
            linestyles="solid",
            alpha=0.7,
        )

    ax.set_xlabel(parent_names[0])
    ax.set_ylabel(parent_names[1])
    ax.set_title(name)
    fig.tight_layout()
    return fig


# ── plotly helpers ────────────────────────────────────────────────


def _plot_varying_coefficient_plotly(raw, name, parent_names, with_ci, colormap):
    """Interactive multi-line plot with hover (incl. base)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    x = raw["x"]
    levels = raw["levels"]
    base = raw.get("base_level", "")

    # +1 for base level color
    n_total = len(levels) + (1 if base else 0)
    colors = _plotly_colors(n_total, colormap)

    # Base level: flat at 1.0
    if base:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.ones_like(x),
                mode="lines",
                name=f"{base} (base)",
                line=dict(color=colors[0], dash="dash"),
            )
        )

    for i, level in enumerate(levels):
        ci = i + 1 if base else i
        level_data = raw["per_level"][level]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=level_data["relativity"],
                mode="lines",
                name=level,
                line=dict(color=colors[ci]),
            )
        )
        if with_ci and "se_log_relativity" in level_data:
            se = level_data["se_log_relativity"]
            log_rel = level_data["log_relativity"]
            ci_hi = np.exp(log_rel + 1.96 * se)
            ci_lo = np.exp(log_rel - 1.96 * se)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([ci_hi, ci_lo[::-1]]),
                    fill="toself",
                    fillcolor=colors[ci],
                    opacity=0.15,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.add_hline(y=1.0, line_dash="dot", line_color="grey", opacity=0.4)
    fig.update_layout(
        title=name,
        xaxis_title=parent_names[0],
        yaxis_title="Relativity",
    )
    return fig


def _plot_categorical_heatmap_plotly(raw, name, parent_names, colormap):
    """Interactive heatmap with hover annotations (incl. base)."""
    import plotly.graph_objects as go

    pairs = raw["pairs"]
    log_rels = raw["log_relativities"]
    rels = raw["relativities"]

    # Full level lists including base (base pairs have relativity=1.0)
    levels1 = raw.get("levels1") or list(dict.fromkeys(p[0] for p in pairs))
    levels2 = raw.get("levels2") or list(dict.fromkeys(p[1] for p in pairs))

    z = np.zeros((len(levels1), len(levels2)))
    text = np.full_like(z, "1.000", dtype=object)
    for (l1, l2), key in zip(pairs, log_rels):
        i = levels1.index(l1)
        j = levels2.index(l2)
        z[i, j] = log_rels[key]
        text[i, j] = f"{rels[key]:.3f}"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=levels2,
            y=levels1,
            colorscale=colormap or "RdBu_r",
            zmid=0,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Log-Relativity"),
        )
    )
    fig.update_layout(
        title=name,
        xaxis_title=parent_names[1],
        yaxis_title=parent_names[0],
    )
    return fig


def _plot_numeric_categorical_bars_plotly(raw, name, parent_names):
    """Interactive vertical bar chart (incl. base)."""
    import plotly.graph_objects as go

    base = raw.get("base_level", "")
    non_base = raw["levels"]
    levels = [base] + list(non_base) if base else list(non_base)
    rels = (
        [1.0] + [raw["relativities_per_unit"][lv] for lv in non_base]
        if base
        else [raw["relativities_per_unit"][lv] for lv in non_base]
    )
    colors = ["lightgrey"] + ["#636EFA"] * len(non_base) if base else ["#636EFA"] * len(non_base)

    fig = go.Figure(go.Bar(x=levels, y=rels, marker_color=colors))
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=name,
        yaxis_title=f"Relativity per unit {parent_names[0]}",
    )
    return fig


def _plot_numeric_interaction_bar_plotly(raw, name, parent_names):
    """Interactive single vertical bar for numeric x numeric."""
    import plotly.graph_objects as go

    rel = raw["relativity_per_unit_unit"]
    label = f"{parent_names[0]} x {parent_names[1]}"

    fig = go.Figure(go.Bar(x=[label], y=[rel]))
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=name,
        yaxis_title="Relativity per unit x unit",
    )
    return fig


def _plot_surface_plotly(raw, name, parent_names, colormap, show_contours, density_data):
    """3D interactive surface plot."""
    import plotly.graph_objects as go

    x1 = raw["x1"]
    x2 = raw["x2"]
    Z = raw["relativity"]

    contours_z = {}
    if show_contours:
        contours_z = dict(
            show=True,
            usecolormap=True,
            highlightcolor="white",
            highlightwidth=1,
        )

    fig = go.Figure(
        go.Surface(
            x=x1,
            y=x2,
            z=Z,
            colorscale=colormap or "Viridis",
            colorbar=dict(title="Relativity", x=1.0),
            contours_z=contours_z,
            name="Relativity",
        )
    )

    if density_data is not None:
        d1, d2, w = density_data
        D = _kde_2d(d1, d2, w, x1, x2)
        z_floor = float(Z.min()) - 0.05 * (Z.max() - Z.min() or 1.0)
        fig.add_trace(
            go.Surface(
                x=x1,
                y=x2,
                z=np.full_like(D, z_floor),
                surfacecolor=D,
                colorscale="Hot_r",
                opacity=0.7,
                showscale=True,
                colorbar=dict(title="Exposure<br>Density", x=1.12, len=0.5, y=0.25),
                name="Exposure density",
                hovertemplate=(
                    f"{parent_names[0]}: %{{x:.1f}}<br>"
                    f"{parent_names[1]}: %{{y:.1f}}<br>"
                    "Density: %{surfacecolor:.3f}<extra>Exposure</extra>"
                ),
            )
        )

    fig.update_layout(
        title=name,
        scene=dict(
            xaxis_title=parent_names[0],
            yaxis_title=parent_names[1],
            zaxis_title="Relativity",
        ),
    )
    return fig


def _plotly_colors(n: int, colormap: str | None = None) -> list[str]:
    """Generate *n* distinct hex colors for plotly traces."""
    if colormap is not None:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(colormap)
        return [
            f"rgba({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)},{c[3]})"
            for c in (cmap(i / max(n - 1, 1)) for i in range(n))
        ]

    # Default qualitative palette (Plotly's standard)
    defaults = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    return [defaults[i % len(defaults)] for i in range(n)]
