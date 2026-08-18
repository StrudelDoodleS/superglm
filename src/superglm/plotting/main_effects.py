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

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
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
    _SPECIAL_COLOR,
    _exposure_kde,
    _level_positions_with_specials,
    _make_continuous_figure,
    _ordered_level_spacing,
    piecewise_display_term,
)
from superglm.plotting.group_display import (
    GroupedTermDisplay,
    grouped_level_exposure,
    project_grouped_term_for_display,
)

if TYPE_CHECKING:
    from superglm.inference.term import TermInference


def plot_relativities(
    terms: list[TermInference],
    *,
    model=None,
    X: FrameLike | None = None,
    sample_weight: NDArray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    with_ci: bool = True,
    interval: str | None = "pointwise",
    show_exposure: bool = True,
    show_knots: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    grouped_level_display: str = "auto",
) -> Figure:
    """Create a grid of relativity plots from ``TermInference`` objects.

    Parameters
    ----------
    terms : list[TermInference]
        Per-term inference objects from :meth:`SuperGLM.term_inference`.
    X : pandas or eager Polars DataFrame, optional
        Training data for sample_weight density overlays.
    sample_weight : array-like, optional
        Weights for the display-density overlay: non-Tweedie case/frequency
        weights or Tweedie EDM prior weights.
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
        Show sample_weight density strip below continuous panels (default *True*).
    show_knots : bool
        Show interior knot positions as minor x-axis ticks (default *False*).
    title, subtitle : str, optional
        Figure-level title and subtitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not with_ci:
        interval = None
    frame = None if X is None else as_eager_frame(X)

    return _plot_relativities_new(
        terms,
        model=model,
        X=frame,
        sample_weight=sample_weight,
        ncols=ncols,
        figsize=figsize,
        interval=interval,
        show_exposure=show_exposure,
        show_knots=show_knots,
        title=title,
        subtitle=subtitle,
        grouped_level_display=grouped_level_display,
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
    X: EagerFrame,
    sample_weight: NDArray,
    x_grid: NDArray,
    show_knots: bool,
    knots: NDArray | None,
):
    """Render the sample_weight density strip beneath a spline panel."""
    x_vals = X.column_array(feature_name, dtype=np.float64)
    density = _exposure_kde(x_vals, sample_weight, x_grid)

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
    model=None,
    X: FrameLike | None = None,
    sample_weight: NDArray | None = None,
    interval: str | None = "pointwise",
    show_exposure: bool = True,
    show_knots: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    grouped_level_display: str = "auto",
) -> Figure:
    """Plot a single term's relativity.

    This is the core single-term plotting function.  All term types
    (spline, polynomial, numeric, categorical) are handled.

    Parameters
    ----------
    ti : TermInference
        Inference result from :meth:`SuperGLM.term_inference`.
    X : pandas or eager Polars DataFrame, optional
        Training data for sample_weight overlays.
    sample_weight : array-like, optional
        Weights for the display-density overlay: non-Tweedie case/frequency
        weights or Tweedie EDM prior weights.
    interval : {"pointwise", "simultaneous", "both", None}
        Band style.  For categoricals, simultaneous/both fall back to pointwise.
    show_exposure : bool
        Show the weighted observation distribution (density strip for
        continuous, vertical bars for categorical).
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

    frame = None if X is None else as_eager_frame(X)
    weighted = sample_weight is not None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    elif frame is not None and show_exposure:
        # Fall back to uniform weights (observation counts) when no
        # sample_weight is provided.
        sample_weight = np.ones(len(frame), dtype=np.float64)

    density_label = "Weight\ndensity" if weighted else "Obs.\ndensity"
    weight_label = "Weight" if weighted else "Count"
    has_density = show_exposure and frame is not None and sample_weight is not None

    display: GroupedTermDisplay | None = None
    if ti.kind == "categorical":
        display = project_grouped_term_for_display(model, ti, grouped_level_display)
        ti = display.term
    elif ti.kind == "piecewise":
        # Bands and density both need a display grid: the knot grid draws a
        # CI band whose interior linearly interpolates the knot limits, which
        # the covariance says is wrong between knots.
        ti = piecewise_display_term(ti)

    if ti.kind in ("spline", "polynomial", "piecewise"):
        needs_strip = has_density and ti.name in frame.columns
        fig, ax, ax_den = _make_continuous_figure(needs_strip, figsize)

        _plot_spline_panel(ax, ti, interval, show_knots)
        ax.set_ylabel("Relativity")

        if ax_den is not None:
            knots = ti.spline.interior_knots if ti.spline is not None else None
            _plot_density_strip(ax_den, ti.name, frame, sample_weight, ti.x, show_knots, knots)
            ax_den.set_ylabel(density_label, fontsize=8)

    elif ti.kind == "numeric":
        needs_strip = has_density and ti.name in frame.columns
        if frame is not None and ti.name in frame.columns:
            x_vals = frame.column_array(ti.name, dtype=np.float64)
            x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
        else:
            x_grid = np.linspace(0.0, 1.0, 200)

        fig, ax, ax_den = _make_continuous_figure(needs_strip, figsize)
        _plot_numeric_panel_continuous(ax, ti, interval, x_grid)

        if ax_den is not None:
            _plot_density_strip(ax_den, ti.name, frame, sample_weight, x_grid, False, None)
            ax_den.set_ylabel(density_label, fontsize=8)

    elif ti.kind == "categorical" and ti.smooth_curve is not None:
        if figsize is None:
            figsize = (max(6, len(ti.levels) * 0.9 + 1.5), 4.5)
        fig, ax = plt.subplots(figsize=figsize)
        _plot_ordered_spline_panel(
            ax,
            ti,
            interval,
            X=frame,
            sample_weight=sample_weight if has_density else None,
            weight_label=weight_label,
            display=display,
        )

    elif ti.kind == "categorical":
        if figsize is None:
            figsize = (max(5, len(ti.levels) * 0.9 + 1.5), 4.5)
        fig, ax = plt.subplots(figsize=figsize)
        _plot_categorical_panel_vertical(
            ax,
            ti,
            interval,
            X=frame,
            sample_weight=sample_weight if has_density else None,
            weight_label=weight_label,
            display=display,
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
    X: EagerFrame | None = None,
    sample_weight: NDArray | None = None,
    weight_label: str = "Weight",
    display: GroupedTermDisplay | None = None,
):
    """Render an OrderedCategorical(spline) panel.

    Ordered levels sit at their spline x-positions under the fitted curve.
    Free (special) levels are detached points past the end of the curve,
    separated by a visible gap, with their own ticks and exposure bars.
    """
    levels = list(ti.levels)
    level_rel = np.asarray(ti.relativity)
    n_levels = len(levels)
    curve = ti.smooth_curve
    is_special = (
        np.asarray(ti.level_is_special, dtype=bool)
        if ti.level_is_special is not None
        else np.zeros(n_levels, dtype=bool)
    )
    level_x = (
        np.asarray(curve.level_x, dtype=np.float64)
        if curve is not None and curve.level_x is not None
        # Fallback grid covers the ordered levels only, as level_x itself does.
        else np.arange(int((~is_special).sum()), dtype=np.float64)
    )
    x_pos = _level_positions_with_specials(level_x, ti.level_is_special, n_levels)
    # Smallest positive gap over the *displayed* positions, so bars sized here
    # never overlap either the ordered levels or the detached free block.
    spacing = _ordered_level_spacing(x_pos)

    # Exposure bars in background
    if sample_weight is not None and X is not None and ti.name in X.columns:
        exp_vals = grouped_level_exposure(display, X, sample_weight)
        if exp_vals is None:
            level_exp = (
                pd.DataFrame(
                    {
                        "level": X.column_array(ti.name),
                        "sample_weight": sample_weight,
                    }
                )
                .groupby("level", sort=False)["sample_weight"]
                .sum()
            )
            exp_vals = np.array([level_exp.get(lv, 0.0) for lv in levels])
        ax2 = ax.twinx()
        ax2.bar(
            x_pos,
            exp_vals,
            width=spacing * 0.6,
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

    # Fitted smooth curve (never an interpolation through the markers)
    if curve is not None:
        ax.plot(
            np.asarray(curve.x, dtype=np.float64),
            np.asarray(curve.relativity, dtype=np.float64),
            color=_LINE_COLOR,
            linewidth=_LINE_WIDTH,
            alpha=0.6,
            zorder=4,
        )

    # Per-level dots with error bars — ordered and free levels drawn separately
    marker_specs = (
        ("Relativity", ~is_special, "o", _LINE_COLOR),
        ("Free levels", is_special, "D", _SPECIAL_COLOR),
    )
    if interval is not None and ti.ci_lower is not None:
        ci_lo = np.asarray(ti.ci_lower)
        ci_hi = np.asarray(ti.ci_upper)
        yerr = np.vstack([level_rel - ci_lo, ci_hi - level_rel])
        for label, mask, marker, color in marker_specs:
            if not mask.any():
                continue
            ax.errorbar(
                x_pos[mask],
                level_rel[mask],
                yerr=yerr[:, mask],
                fmt=marker,
                color=color,
                markersize=7,
                ecolor="#333333",
                elinewidth=1.2,
                capsize=4,
                label=label,
                zorder=5,
            )
    else:
        for label, mask, marker, color in marker_specs:
            if not mask.any():
                continue
            ax.scatter(
                x_pos[mask],
                level_rel[mask],
                color=color,
                s=50,
                marker=marker,
                zorder=5,
                label=label,
            )

    if is_special.any() and (~is_special).any():
        divider = 0.5 * (float(x_pos[~is_special].max()) + float(x_pos[is_special].min()))
        ax.axvline(divider, linestyle=":", linewidth=_REF_LW, color=_REF_COLOR, zorder=1)

    ax.set_xticks(x_pos)
    rot = 45 if n_levels > 8 else 0
    ha = "right" if rot else "center"
    ax.set_xticklabels(levels, rotation=rot, ha=ha, fontsize=8)
    # The markers and the fitted curve need not span the same range, so the
    # limits are the UNION of the marker padding and the curve's own extent and
    # neither can clip the other.  Which way they disagree depends on the panel:
    # in ``expanded`` mode the markers sit at the declared level values while
    # the curve spans the fitted axis (group means), so a leading or trailing
    # merge leaves the curve short of the outermost markers -- ggplot2's
    # ``geom_smooth(fullrange=FALSE)`` default, where the smoothing line is not
    # expanded to the range of the plot.  A detached ``specials=`` block reaches
    # past the curve on the other side.
    lo = float(x_pos.min()) - spacing / 2.0
    hi = float(x_pos.max()) + spacing / 2.0
    if curve is not None:
        curve_x = np.asarray(curve.x, dtype=np.float64)
        if curve_x.size:
            lo = min(lo, float(curve_x.min()))
            hi = max(hi, float(curve_x.max()))
    ax.set_xlim(lo, hi)
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
    X: EagerFrame | None = None,
    sample_weight: NDArray | None = None,
    weight_label: str = "Weight",
    display: GroupedTermDisplay | None = None,
):
    """Render a categorical panel with vertical orientation.

    Levels on x-axis, relativity on y-axis.  Optional sample_weight bars
    in the background.
    """
    levels = list(ti.levels)
    rel = np.asarray(ti.relativity)
    x_pos = np.arange(len(levels))

    # Exposure bars in background
    if sample_weight is not None and X is not None and ti.name in X.columns:
        exp_vals = grouped_level_exposure(display, X, sample_weight)
        if exp_vals is None:
            level_exp = (
                pd.DataFrame(
                    {
                        "level": X.column_array(ti.name),
                        "sample_weight": sample_weight,
                    }
                )
                .groupby("level", sort=False)["sample_weight"]
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
    model=None,
    X: EagerFrame | None = None,
    sample_weight: NDArray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    interval: str | None = "pointwise",
    show_exposure: bool = True,
    show_knots: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    grouped_level_display: str = "auto",
) -> Figure:
    """TermInference-based relativity grid with the new visual language."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(terms)
    if n == 0:
        fig, _ = plt.subplots()
        return fig

    weighted = sample_weight is not None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    elif X is not None and show_exposure:
        sample_weight = np.ones(len(X), dtype=np.float64)

    density_label = "Weight\ndensity" if weighted else "Obs.\ndensity"
    weight_label = "Weight" if weighted else "Count"
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    has_density = show_exposure and X is not None and sample_weight is not None
    _CONTINUOUS_KINDS = ("spline", "polynomial", "piecewise", "numeric")
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
        display: GroupedTermDisplay | None = None
        display_ti = ti
        if ti.kind == "categorical":
            display = project_grouped_term_for_display(model, ti, grouped_level_display)
            display_ti = display.term
        elif ti.kind == "piecewise":
            display_ti = piecewise_display_term(ti)

        if display_ti.kind in ("spline", "polynomial", "piecewise"):
            _plot_spline_panel(ax, display_ti, interval, show_knots)
            if idx % ncols == 0:
                ax.set_ylabel("Relativity")

            if ax_den is not None:
                knots = display_ti.spline.interior_knots if display_ti.spline is not None else None
                _plot_density_strip(
                    ax_den, display_ti.name, X, sample_weight, display_ti.x, show_knots, knots
                )
                if idx % ncols == 0:
                    ax_den.set_ylabel(density_label, fontsize=8)
                ax_den.set_xlabel("")

        elif display_ti.kind == "categorical" and display_ti.smooth_curve is not None:
            _plot_ordered_spline_panel(
                ax,
                display_ti,
                interval,
                X=X,
                sample_weight=sample_weight if has_density else None,
                weight_label=weight_label,
                display=display,
            )

        elif display_ti.kind == "categorical":
            _plot_categorical_panel_vertical(
                ax,
                display_ti,
                interval,
                X=X,
                sample_weight=sample_weight if has_density else None,
                weight_label=weight_label,
                display=display,
            )

        elif display_ti.kind == "numeric":
            if X is not None and display_ti.name in X.columns:
                x_vals = X.column_array(display_ti.name, dtype=np.float64)
                x_grid = np.linspace(x_vals.min(), x_vals.max(), 200)
            else:
                x_grid = np.linspace(0.0, 1.0, 200)
            _plot_numeric_panel_continuous(ax, display_ti, interval, x_grid)
            if idx % ncols == 0:
                ax.set_ylabel("Relativity")

            if ax_den is not None:
                _plot_density_strip(ax_den, display_ti.name, X, sample_weight, x_grid, False, None)
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
