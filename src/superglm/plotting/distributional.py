"""Matplotlib renderers for the distributional inference suite.

Every function here takes a payload built in :mod:`superglm.distributional` --
never a model -- and returns a :class:`~matplotlib.figure.Figure` drawn in the
notebook editor's chart grammar: the fitted thing in ``.edited`` blue, the
reference in ``.original`` grey dashes, bands in the ``.ci`` fill, simultaneous
outlines and whiskers in ``.ci-whisker``, exposure and counts in the
``.exposure`` yellow, observations as ``.point`` and flagged ones as
``.point.selected``.  The constants come from
:mod:`superglm.plotting.editor_style` and from nowhere else, so a figure from
this module and a chart in the editor read as one product.

A figure is therefore reproducible from its payload alone: nothing below calls
the model, resolves weights or decides a statistic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke
from numpy.typing import NDArray
from scipy import stats

from superglm.distributional.checks.binned import BinnedCheck, BinnedCheck2D
from superglm.distributional.checks.calibration import ActualExpected, CalibrationPayload
from superglm.distributional.checks.compare import Comparison
from superglm.distributional.checks.pit import PITPayload
from superglm.distributional.checks.qq import QQPayload
from superglm.distributional.checks.worm import WormPanel, WormPayload
from superglm.distributional.residuals import ResidualSet, replication_sample
from superglm.distributional.surfaces import DensityFan, Histogram, Portfolio, RiskCurves, Spread
from superglm.distributional.terms import ParameterTermEffect
from superglm.plotting.editor_style import (
    CHART,
    LABEL_PT,
    PANEL,
    TOKENS,
    apply_panel_frame,
    diverging_cmap,
    matplotlib_context,
    sequential_cmap,
)

__all__ = [
    "plot_actual_expected",
    "plot_binned",
    "plot_binned_2d",
    "plot_calibration",
    "plot_comparison",
    "plot_density_fan",
    "plot_diagnostics_figure",
    "plot_pit",
    "plot_portfolio",
    "plot_qq",
    "plot_risk_curves",
    "plot_spread",
    "plot_term_effect",
    "plot_term_grid",
    "plot_worm",
]

#: Above this many order statistics the Q-Q cloud is drawn as a curve.
_DENSE_CLOUD = 2000
#: Fraction of a panel's height the exposure strip occupies.
_STRIP_HEIGHT = 0.12
#: Half-width of a 95 per cent normal interval, for whiskers on a standard error.
_TWO_SIDED_95 = 1.959963984540054
#: Above this many level labels the tick text is laid at an angle to stay legible.
_CROWDED_LEVELS = 12
#: Angle a crowded level axis writes its labels at.
_CROWDED_ROTATION = 45.0


# --------------------------------------------------------------------------- #
# The chart grammar as matplotlib keyword arguments
# --------------------------------------------------------------------------- #


def _colour(entry: dict[str, Any], key: str = "color") -> tuple[float, float, float, float]:
    """One chart class's colour as RGBA, carrying the class's own opacity."""
    value = entry[key]
    alpha = float(entry.get("alpha", 1.0))
    if isinstance(value, str):
        return to_rgba(value, alpha)
    red, green, blue = value
    return (red / 255.0, green / 255.0, blue / 255.0, alpha)


def _line(entry: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Line keywords for a chart class: its colour, width and dash pattern."""
    kwargs: dict[str, Any] = {"color": _colour(entry), "linewidth": entry.get("width", 1.0)}
    dash = entry.get("dash")
    if dash is not None:
        kwargs["linestyle"] = (0.0, tuple(float(value) for value in dash))
    kwargs.update(overrides)
    return kwargs


def _fill(entry: dict[str, Any]) -> dict[str, Any]:
    """Fill keywords for a band: the class's colour, no stroke."""
    return {"color": _colour(entry), "linewidth": 0.0}


def _points(entry: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Marker keywords for ``.point`` or ``.point.selected``."""
    kwargs: dict[str, Any] = {
        "facecolor": _colour(entry, "face"),
        "edgecolor": _colour(entry, "edge"),
        "linewidths": entry.get("width", CHART["point"]["width"]),
        "s": 26.0,
        "zorder": 3,
    }
    kwargs.update(overrides)
    return kwargs


def _bars(entry: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Bar keywords for the ``.exposure`` yellow: fill, edge and edge width."""
    kwargs: dict[str, Any] = {
        "color": _colour(entry, "fill"),
        "edgecolor": entry["edge"],
        "linewidth": entry["width"],
    }
    kwargs.update(overrides)
    return kwargs


def _whiskers(**overrides: Any) -> dict[str, Any]:
    """Error-bar keywords in ``.ci-whisker``."""
    entry = CHART["ci_whisker"]
    kwargs: dict[str, Any] = {
        "fmt": "none",
        "ecolor": _colour(entry),
        "elinewidth": entry["width"],
        "capsize": 3.0,
        "zorder": 2,
    }
    kwargs.update(overrides)
    return kwargs


def _label(ax: Axes, x: float, y: float, text: str, *, color: str | None = None, **kwargs: Any):
    """Write an in-plot label: 700 weight, 11 pt, white halo."""
    artist = ax.text(
        x,
        y,
        text,
        fontsize=LABEL_PT,
        fontweight=700,
        color=TOKENS["text"] if color is None else color,
        **kwargs,
    )
    artist.set_path_effects([withStroke(linewidth=3, foreground="white")])
    return artist


def _note(ax: Axes, text: str) -> None:
    """Say why a panel is empty, in the middle of it."""
    _label(
        ax, 0.5, 0.5, text, color=TOKENS["muted"], ha="center", va="center", transform=ax.transAxes
    )


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


def _single_panel(ax: Axes | None) -> tuple[Figure, Axes]:
    """The caller's axes, or a new figure of one editor-sized panel."""
    if ax is not None:
        # ``root=True``: an axes drawn into a subfigure still returns the figure
        # the caller owns, which is the one a renderer lays out and returns.
        figure = ax.get_figure(root=True)
        if figure is None:  # pragma: no cover - an axes always has its figure
            raise ValueError("ax must belong to a figure")
        return figure, ax
    return plt.subplots(figsize=(PANEL["width_in"], PANEL["height_in"]))


def _panel_grid(fig: Figure | None, n_panels: int, ncols: int) -> tuple[Figure, list[Axes]]:
    """A grid of ``n_panels`` panels, each keeping the editor's panel aspect."""
    columns = max(1, min(int(ncols), n_panels))
    rows = math.ceil(n_panels / columns)
    if fig is None:
        fig = plt.figure(
            figsize=(PANEL["width_in"] * columns, PANEL["height_in"] * rows),
        )
    axes = list(np.ravel(fig.subplots(rows, columns, squeeze=False)))
    for spare in axes[n_panels:]:
        fig.delaxes(spare)
    return fig, axes[:n_panels]


def _finish(fig: Figure, axes: Iterable[Axes], *, suptitle: bool = False) -> Figure:
    """Frame every panel and lay the figure out."""
    for ax in axes:
        apply_panel_frame(ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95) if suptitle else None)
    return fig


def _twin(ax: Axes) -> Axes:
    """A right-hand axes in the editor's frame, without a second grid.

    The twin frames itself, because a second pass of :func:`apply_panel_frame`
    -- what ``_finish`` does to the panels it is handed -- would put back the
    grid this turns off.
    """
    twin = ax.twinx()
    apply_panel_frame(twin)
    twin.grid(False)
    return twin


def _step(positions: NDArray[np.float64]) -> float:
    """Bar width for a coordinate: most of the median gap between positions."""
    values = np.asarray(positions, dtype=np.float64)
    if values.size < 2:
        return 0.8
    gaps = np.diff(np.sort(values))
    positive = gaps[gaps > 0.0]
    return 0.8 * float(np.median(positive)) if positive.size else 0.8


def _level_ticks(ax: Axes, positions: NDArray[np.float64], levels: Sequence[str] | None) -> None:
    """Label a level coordinate with its levels, when the payload names them.

    Past a dozen levels horizontal labels collide -- two dozen mileage bands
    render as one grey smear -- so a crowded axis writes them at an angle
    instead of dropping every second one, which would leave a reader unable to
    say which bar is which.
    """
    if levels is None:
        return
    ax.set_xticks(np.asarray(positions, dtype=np.float64))
    crowded = len(levels) > _CROWDED_LEVELS
    ax.set_xticklabels(
        [str(level) for level in levels],
        rotation=_CROWDED_ROTATION if crowded else 0.0,
        ha="right" if crowded else "center",
    )


def _exposure_strip(ax: Axes, positions: NDArray[np.float64], exposure: NDArray) -> None:
    """Draw exposure as a strip along the bottom of ``ax``, keeping its limits."""
    values = np.asarray(exposure, dtype=np.float64)
    coordinates = np.asarray(positions, dtype=np.float64)
    if values.shape != coordinates.shape:
        raise ValueError("exposure must carry one value per grid point or level")
    peak = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return
    low, high = ax.get_ylim()
    heights = _STRIP_HEIGHT * (high - low) * values / peak
    ax.bar(
        coordinates,
        heights,
        bottom=low,
        width=_step(coordinates),
        zorder=1,
        **_bars(CHART["exposure"]),
    )
    ax.set_ylim(low, high)


# --------------------------------------------------------------------------- #
# Distribution checks
# --------------------------------------------------------------------------- #


def _draw_qq(ax: Axes, payload: QQPayload) -> None:
    grid = payload.theoretical
    ax.fill_between(grid, payload.envelope_lower, payload.envelope_upper, **_fill(CHART["ci"]))
    ax.plot(grid, grid, **_line(CHART["original"]))
    if len(payload.observed) > _DENSE_CLOUD:
        ax.plot(grid, payload.observed, **_line(CHART["edited"]))
    else:
        ax.scatter(grid, payload.observed, **_points(CHART["point"]))
    subsampled = ", subsampled" if payload.subsampled else ""
    ax.set_title(
        f"Q-Q of the quantile residuals: {payload.n_rows} rows, "
        f"{payload.n_sim} simulations{subsampled}"
    )
    ax.set_xlabel("theoretical quantile")
    ax.set_ylabel("observed quantile residual")


def plot_qq(payload: QQPayload, *, ax: Axes | None = None) -> Figure:
    """Draw the Q-Q payload: its simulated envelope, cloud and theoretical line."""
    with matplotlib_context():
        fig, ax = _single_panel(ax)
        _draw_qq(ax, payload)
        return _finish(fig, [ax])


def _q_statistic_rows(payload: WormPayload) -> dict[str, pd.Series]:
    table = payload.q_statistics
    if table is None:
        return {}
    return {str(row["group"]): row for _, row in table.iterrows()}


def _q_statistic_text(row: pd.Series) -> str:
    return "  ".join(
        f"{name} {float(row[column]):+.2f}"
        for name, column in (
            ("mean", "mean_z"),
            ("var", "variance_z"),
            ("skew", "skewness_z"),
            ("kurt", "kurtosis_z"),
        )
    )


def _draw_worm_panel(ax: Axes, panel: WormPanel, row: pd.Series | None) -> None:
    ax.fill_between(panel.band_z, -panel.band, panel.band, **_fill(CHART["ci"]))
    ax.plot(panel.band_z, panel.band, **_line(CHART["ci_whisker"]))
    ax.plot(panel.band_z, -panel.band, **_line(CHART["ci_whisker"]))
    ax.axhline(0.0, **_line(CHART["zero"]))
    ax.scatter(panel.z, panel.deviation, **_points(CHART["point"]))
    interval = ""
    if panel.interval is not None:
        interval = f" [{panel.interval[0]:.3g}, {panel.interval[1]:.3g}]"
    ax.set_title(f"{panel.label}{interval}: n = {panel.n}")
    ax.set_xlabel("theoretical quantile")
    ax.set_ylabel("deviation")
    if row is not None:
        # A worm curls up at its ends, so the bottom-left corner is exactly
        # where the lowest points sit; the panel's top-left is empty by
        # construction and a filled box keeps the line off the envelope.
        _label(
            ax,
            0.02,
            0.96,
            _q_statistic_text(row),
            color=TOKENS["red"] if bool(row["flagged"]) else TOKENS["muted"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="#ffffff", edgecolor="#d0d7de"),
        )


def plot_worm(
    payload: WormPayload, *, fig: Figure | None = None, ncols: int | None = None
) -> Figure:
    """Draw one worm per panel of the payload, with its Q-statistics beside it."""
    panels = payload.panels
    with matplotlib_context():
        columns = int(ncols) if ncols else min(2, len(panels))
        fig, axes = _panel_grid(fig, len(panels), columns)
        rows = _q_statistic_rows(payload)
        overall = rows.pop("all", None) if len(panels) > 1 else None
        for ax, panel in zip(axes, panels, strict=True):
            _draw_worm_panel(ax, panel, rows.get(panel.label))
        if overall is not None:
            fig.suptitle(f"all rows: {_q_statistic_text(overall)}", fontweight=700)
        return _finish(fig, axes, suptitle=overall is not None)


def _draw_pit(ax: Axes, payload: PITPayload) -> None:
    edges = payload.edges
    ax.bar(
        edges[:-1],
        payload.counts,
        width=np.diff(edges),
        align="edge",
        **_bars(CHART["exposure"]),
    )
    ax.fill_between(
        [float(edges[0]), float(edges[-1])],
        [payload.band_lower] * 2,
        [payload.band_upper] * 2,
        **_fill(CHART["ci"]),
    )
    ax.axhline(payload.expected, **_line(CHART["zero"]))
    ax.set_title(f"PIT histogram: {payload.n_rows} rows in {payload.n_bins} bins")
    ax.set_xlabel("probability integral transform")
    ax.set_ylabel("count")


def plot_pit(payload: PITPayload, *, ax: Axes | None = None) -> Figure:
    """Draw the PIT histogram against its expected count and uniform band."""
    with matplotlib_context():
        fig, ax = _single_panel(ax)
        _draw_pit(ax, payload)
        return _finish(fig, [ax])


# --------------------------------------------------------------------------- #
# Binned checks
# --------------------------------------------------------------------------- #


_BINNED_MOMENTS = (
    ("mean", 0.0, "mean residual"),
    ("sd", 1.0, "residual sd"),
    ("skew", 0.0, "residual skew"),
)


def plot_binned(payload: BinnedCheck, *, fig: Figure | None = None) -> Figure:
    """Draw the binned mean, standard deviation and skewness against their bands."""
    centers = np.asarray(payload.centers, dtype=np.float64)
    with matplotlib_context():
        fig, axes = _panel_grid(fig, len(_BINNED_MOMENTS), 1)
        for ax, (moment, reference, label) in zip(axes, _BINNED_MOMENTS, strict=True):
            values = np.asarray(getattr(payload, moment), dtype=np.float64)
            lower = np.asarray(getattr(payload, f"{moment}_lower"), dtype=np.float64)
            upper = np.asarray(getattr(payload, f"{moment}_upper"), dtype=np.float64)
            ax.fill_between(centers, lower, upper, **_fill(CHART["ci"]))
            ax.axhline(reference, **_line(CHART["zero"]))
            flagged = (lower > reference) | (upper < reference)
            ax.scatter(centers[~flagged], values[~flagged], **_points(CHART["point"]))
            ax.scatter(centers[flagged], values[flagged], **_points(CHART["point_selected"]))
            ax.set_ylabel(label)
            _level_ticks(ax, centers, payload.levels)
        axes[0].set_title(f"binned residual moments by {payload.covariate}")
        axes[-1].set_xlabel(payload.covariate)
        return _finish(fig, axes)


def plot_binned_2d(payload: BinnedCheck2D, *, ax: Axes | None = None) -> Figure:
    """Draw the binned mean residual over two covariates, centred at zero."""
    mean = np.asarray(payload.mean, dtype=np.float64)
    counts = np.asarray(payload.count)
    finite = np.abs(mean[np.isfinite(mean)])
    limit = float(finite.max()) if finite.size and finite.max() > 0.0 else 1.0
    x_edges = np.asarray(payload.x_edges, dtype=np.float64)
    y_edges = np.asarray(payload.y_edges, dtype=np.float64)
    with matplotlib_context():
        fig, ax = _single_panel(ax)
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(mean.T),
            cmap=diverging_cmap(),
            vmin=-limit,
            vmax=limit,
        )
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        for i, x_value in enumerate(x_centers):
            for j, y_value in enumerate(y_centers):
                if counts[i, j] > 0:
                    _label(ax, x_value, y_value, str(int(counts[i, j])), ha="center", va="center")
        fig.colorbar(mesh, ax=ax, label="mean quantile residual")
        ax.set_title(f"mean residual by {payload.covariates[0]} and {payload.covariates[1]}")
        ax.set_xlabel(payload.covariates[0])
        ax.set_ylabel(payload.covariates[1])
        return _finish(fig, [ax])


# --------------------------------------------------------------------------- #
# Actual versus expected and calibration
# --------------------------------------------------------------------------- #


def plot_actual_expected(payload: ActualExpected, *, ax: Axes | None = None) -> Figure:
    """Draw the actual-over-expected ratio per bin above its exposure."""
    centers = np.asarray(payload.centers, dtype=np.float64)
    ratio = np.asarray(payload.ratio, dtype=np.float64)
    error = _TWO_SIDED_95 * np.asarray(payload.ratio_se, dtype=np.float64)
    with matplotlib_context():
        fig, ax = _single_panel(ax)
        exposure_axis = _twin(ax)
        exposure_axis.bar(
            centers,
            np.asarray(payload.weight, dtype=np.float64),
            width=_step(centers),
            zorder=1,
            **_bars(CHART["exposure"]),
        )
        exposure_axis.set_ylabel("exposure")
        peak = float(np.max(payload.weight))
        if peak > 0.0:
            # Exposure is context, not the subject: keep it to a strip along the
            # bottom rather than letting it fill the panel behind the ratio.
            exposure_axis.set_ylim(0.0, peak / 0.45)
        ax.set_zorder(exposure_axis.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.axhline(1.0, **_line(CHART["zero"]))
        ax.plot(centers, ratio, **_line(CHART["edited"]), zorder=2)
        ax.errorbar(centers, ratio, yerr=error, **_whiskers())
        flagged = np.abs(ratio - 1.0) > error
        ax.scatter(centers[~flagged], ratio[~flagged], **_points(CHART["point"]))
        ax.scatter(centers[flagged], ratio[flagged], **_points(CHART["point_selected"]))
        _level_ticks(ax, centers, payload.levels)
        # A ratio of non-negative totals cannot go below zero, and the whiskers
        # of a thin bin otherwise drag the axis under it; a signed target that
        # really does reach below zero keeps its own floor.
        ax.set_ylim(bottom=float(np.nanmin(np.append(ratio - error, 0.0))))
        overall = float(np.sum(payload.actual)) / float(np.sum(payload.expected))
        ax.set_title(f"{payload.covariate}: actual over expected, overall {overall:.3f}")
        ax.set_xlabel(payload.covariate)
        ax.set_ylabel("actual / expected")
        return _finish(fig, [ax])


def _draw_coverage(ax: Axes, payload: CalibrationPayload) -> None:
    frame = payload.coverage
    for group, block in frame.groupby("group", sort=False):
        level = np.asarray(block["level"], dtype=np.float64)
        realised = np.asarray(block["realised"], dtype=np.float64)
        if str(group) == "all":
            ax.plot(level, realised, label="all", **_line(CHART["edited"]))
            error = _TWO_SIDED_95 * np.asarray(block["se"], dtype=np.float64)
            outside = np.abs(realised - level) > error
            ax.scatter(level[~outside], realised[~outside], **_points(CHART["point"]))
            ax.scatter(level[outside], realised[outside], **_points(CHART["point_selected"]))
        else:
            ax.plot(level, realised, **_line(CHART["basis_contribution"]))
    levels = np.asarray(payload.levels, dtype=np.float64)
    ax.plot(levels, levels, **_line(CHART["original"]))
    ax.set_title("interval coverage")
    ax.set_xlabel("nominal level")
    ax.set_ylabel("realised coverage")


def _tail_tick(threshold: float, group: str) -> str:
    """A tail-table tick: the threshold and its group, short enough to read.

    The decile groups arrive spelled ``exceedance:decile 7``, which at eleven
    ticks per threshold leaves a column of text taller than the panel; the
    reader needs the threshold and which decile, and nothing else.
    """
    label = str(group)
    _, _, decile = label.rpartition("decile ")
    short = "all" if label == "all" else f"d{decile.strip()}"
    return f"{float(threshold):.3g} \u00b7 {short}"


def _draw_tails(ax: Axes, payload: CalibrationPayload) -> None:
    frame = payload.tails
    if frame.empty:
        _note(ax, "no thresholds requested")
        ax.set_title("tail exceedance")
        return
    positions = np.arange(len(frame), dtype=np.float64)
    ax.bar(
        positions - 0.2,
        np.asarray(frame["expected"], dtype=np.float64),
        width=0.4,
        color=_colour(CHART["ci"]),
        edgecolor=_colour(CHART["ci_whisker"]),
        linewidth=CHART["ci_whisker"]["width"],
        label="expected",
    )
    ax.bar(
        positions + 0.2,
        np.asarray(frame["realised"], dtype=np.float64),
        width=0.4,
        label="realised",
        **_bars(CHART["exposure"]),
    )
    ax.errorbar(
        positions - 0.2,
        np.asarray(frame["expected"], dtype=np.float64),
        yerr=np.asarray(frame["se"], dtype=np.float64),
        **_whiskers(),
    )
    one_threshold = len(payload.thresholds) == 1
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            _tail_tick(threshold, group)
            for threshold, group in zip(frame["threshold"], frame["group"], strict=True)
        ],
        rotation=90,
    )
    ax.legend(fontsize=LABEL_PT)
    named = f" over {payload.thresholds[0]:.3g}" if one_threshold else ""
    ax.set_title(f"tail exceedance{named}: expected against realised")
    ax.set_ylabel("weighted count")


def _draw_quantile_calibration(ax: Axes, payload: CalibrationPayload) -> None:
    frame = payload.quantiles.sort_values("expected")
    expected = np.asarray(frame["expected"], dtype=np.float64)
    realised = np.asarray(frame["realised_exceedance"], dtype=np.float64)
    error = _TWO_SIDED_95 * np.asarray(frame["se"], dtype=np.float64)
    ax.fill_between(expected, realised - error, realised + error, **_fill(CHART["ci"]))
    ax.plot(expected, realised, **_line(CHART["edited"]))
    ax.scatter(expected, realised, **_points(CHART["point"]))
    ax.plot(expected, expected, **_line(CHART["original"]))
    ax.set_title("quantile calibration")
    ax.set_xlabel("nominal exceedance 1 - p")
    ax.set_ylabel("realised exceedance")


def _draw_reliability(ax: Axes, payload: CalibrationPayload) -> None:
    if not payload.reliability:
        _note(ax, "no thresholds requested")
        ax.set_title("reliability")
        return
    # One colour per threshold along the single-hue scale, as the risk curves
    # do: with three exceedance levels drawn in one blue the panel says which
    # curve belongs to which threshold only in the legend.
    scale = sequential_cmap()
    count = len(payload.reliability)
    for index, (threshold, curve) in enumerate(payload.reliability.items()):
        x = np.asarray(curve.x, dtype=np.float64)
        position = 1.0 if count == 1 else index / (count - 1)
        ax.fill_between(x, curve.lower, curve.upper, **_fill(CHART["ci"]))
        ax.plot(
            x,
            curve.calibrated,
            label=f"> {float(threshold):g}",
            **_line(CHART["edited"], color=scale(position)),
        )
        ax.plot(x, x, **_line(CHART["original"]))
    ax.legend(fontsize=LABEL_PT)
    ax.set_title("reliability of the exceedance forecast")
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("calibrated probability")


def plot_calibration(payload: CalibrationPayload, *, fig: Figure | None = None) -> Figure:
    """Draw coverage, tail exceedance, quantile calibration and reliability."""
    with matplotlib_context():
        fig, axes = _panel_grid(fig, 4, 2)
        _draw_coverage(axes[0], payload)
        _draw_tails(axes[1], payload)
        _draw_quantile_calibration(axes[2], payload)
        _draw_reliability(axes[3], payload)
        return _finish(fig, axes)


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #


def _draw_segment_scores(ax: Axes, payload: Comparison) -> None:
    table = payload.by_segment
    if table is None:
        labels = ["all"]
        mean = np.array([float(payload.overall["mean_diff"])])
        error = np.array([float(payload.overall["se"])])
    else:
        labels = [str(label) for label in table.index]
        mean = np.asarray(table["mean_diff"], dtype=np.float64)
        error = np.asarray(table["se"], dtype=np.float64)
    positions = np.arange(len(labels), dtype=np.float64)
    ax.axhline(0.0, **_line(CHART["zero"]))
    ax.errorbar(positions, mean, yerr=_TWO_SIDED_95 * error, **_whiskers())
    ax.scatter(positions, mean, **_points(CHART["point"]))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(f"{payload.score} score difference (a - b), negative favours a")
    ax.set_ylabel("mean difference")


def plot_comparison(payload: Comparison, *, fig: Figure | None = None) -> Figure:
    """Draw the segment score differences and, when present, the Murphy diagram."""
    murphy = payload.murphy
    with matplotlib_context():
        fig, axes = _panel_grid(fig, 1 if murphy is None else 3, 1)
        _draw_segment_scores(axes[0], payload)
        if murphy is not None:
            grid = np.asarray(murphy.thresholds, dtype=np.float64)
            axes[1].plot(grid, murphy.a, label="a", **_line(CHART["edited"]))
            axes[1].plot(grid, murphy.b, label="b", **_line(CHART["original"]))
            axes[1].legend(fontsize=LABEL_PT)
            axes[1].set_title(f"Murphy diagram at the {murphy.level:g} quantile")
            axes[1].set_ylabel("mean elementary score")
            difference = np.asarray(murphy.difference, dtype=np.float64)
            error = np.asarray(murphy.difference_se, dtype=np.float64)
            axes[2].fill_between(grid, difference - error, difference + error, **_fill(CHART["ci"]))
            axes[2].plot(grid, difference, **_line(CHART["edited"]))
            axes[2].axhline(0.0, **_line(CHART["zero"]))
            axes[2].set_title("elementary score difference, one standard error")
            axes[2].set_xlabel("threshold")
            axes[2].set_ylabel("a - b")
        return _finish(fig, axes)


# --------------------------------------------------------------------------- #
# Term effects
# --------------------------------------------------------------------------- #


def _multiplier_axis(ax: Axes, effect: NDArray[np.float64], multiplier: NDArray) -> Axes | None:
    """A right-hand axis reading the link-scale effect as its multiplier."""
    values = np.asarray(effect, dtype=np.float64)
    factors = np.asarray(multiplier, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(factors)
    if np.count_nonzero(finite) < 2:
        return None
    order = np.argsort(values[finite])
    ordered_effect = values[finite][order]
    ordered_factor = factors[finite][order]
    if ordered_effect[-1] <= ordered_effect[0]:
        return None
    twin = _twin(ax)
    twin.set_ylim(ax.get_ylim())
    ticks = np.linspace(float(ordered_effect[0]), float(ordered_effect[-1]), 5)
    twin.set_yticks(ticks)
    twin.set_yticklabels(
        [f"{value:.3g}" for value in np.interp(ticks, ordered_effect, ordered_factor)]
    )
    twin.set_ylabel("multiplier")
    return twin


def _draw_term_effect(ax: Axes, effect: ParameterTermEffect, exposure: NDArray | None) -> None:
    values = np.asarray(effect.effect, dtype=np.float64)
    if effect.x is not None:
        positions = np.asarray(effect.x, dtype=np.float64)
        ax.fill_between(positions, effect.lower, effect.upper, **_fill(CHART["ci"]))
        if effect.lower_simultaneous is not None and effect.upper_simultaneous is not None:
            ax.plot(positions, effect.lower_simultaneous, **_line(CHART["ci_whisker"]))
            ax.plot(positions, effect.upper_simultaneous, **_line(CHART["ci_whisker"]))
        ax.axhline(0.0, **_line(CHART["zero"]))
        ax.plot(positions, values, **_line(CHART["edited"]))
    elif effect.levels is not None:
        positions = np.arange(len(effect.levels), dtype=np.float64)
        ax.axhline(0.0, **_line(CHART["zero"]))
        ax.errorbar(
            positions,
            values,
            yerr=np.vstack([values - effect.lower, effect.upper - values]),
            **_whiskers(capsize=4.0),
        )
        # A free special level is a different kind of number from a point on
        # the smooth -- it is fitted on its own rows and shares none of the
        # smooth's shape -- so it is drawn as a flagged marker and named.
        special = (
            np.zeros(len(positions), dtype=bool)
            if effect.special is None
            else np.asarray(effect.special, dtype=bool)
        )
        ax.scatter(positions[~special], values[~special], **_points(CHART["point"]))
        if special.any():
            ax.scatter(
                positions[special],
                values[special],
                label="special",
                **_points(CHART["point_selected"]),
            )
            ax.legend(fontsize=LABEL_PT)
        _level_ticks(ax, positions, effect.levels)
    else:  # pragma: no cover - a term reports a grid or its levels
        raise ValueError("a term effect carries either a grid or a set of levels")
    ax.set_xlabel(effect.term)
    ax.set_ylabel(f"effect on the {effect.link} scale")
    ax.set_title(f"{effect.parameter}: {effect.term} (edf {effect.edf:.2f})")
    if exposure is not None:
        _exposure_strip(ax, positions, exposure)
    if effect.multiplier is not None:
        _multiplier_axis(ax, values, effect.multiplier)


def plot_term_effect(
    effect: ParameterTermEffect, *, ax: Axes | None = None, exposure: NDArray | None = None
) -> Figure:
    """Draw one term of one predictor on its own link scale, with its bands."""
    with matplotlib_context():
        fig, ax = _single_panel(ax)
        _draw_term_effect(ax, effect, exposure)
        return _finish(fig, [ax])


def plot_term_grid(
    effects: Sequence[ParameterTermEffect], *, parameter: str | None = None, ncols: int = 3
) -> Figure:
    """Draw a grid of term panels, optionally only those of one parameter."""
    selected = [
        effect for effect in effects if parameter is None or str(effect.parameter) == str(parameter)
    ]
    if not selected:
        raise ValueError(f"no term of parameter {parameter!r} to plot")
    with matplotlib_context():
        fig, axes = _panel_grid(None, len(selected), ncols)
        for ax, effect in zip(axes, selected, strict=True):
            _draw_term_effect(ax, effect, None)
        return _finish(fig, axes)


# --------------------------------------------------------------------------- #
# Surfaces
# --------------------------------------------------------------------------- #


def plot_risk_curves(payload: RiskCurves, *, ax: Axes | None = None) -> Figure:
    """Draw one predicted-quantile curve per level, thicker the further out it is."""
    x = np.asarray(payload.x, dtype=np.float64)
    quantiles = np.asarray(payload.quantiles, dtype=np.float64)
    ranks = np.argsort(np.argsort(quantiles)).astype(np.float64)
    base = CHART["edited"]["width"]
    spread = ranks / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(ranks)
    with matplotlib_context():
        fig, ax = _single_panel(ax)
        for index, quantile in enumerate(payload.quantiles):
            ax.fill_between(x, payload.lower[index], payload.upper[index], **_fill(CHART["ci"]))
            ax.plot(
                x,
                payload.values[index],
                label=f"q{quantile:g}",
                **_line(CHART["edited"], linewidth=base * (0.6 + 0.8 * float(spread[index]))),
            )
        _level_ticks(ax, x, payload.levels)
        ax.legend(fontsize=LABEL_PT)
        ax.set_title(
            f"{payload.covariate}: predicted quantiles with their "
            f"{payload.level:.0%} posterior band"
        )
        ax.set_xlabel(payload.covariate)
        ax.set_ylabel("response")
        return _finish(fig, [ax])


def plot_density_fan(payload: DensityFan, *, ax: Axes | None = None, contours: int = 6) -> Figure:
    """Draw the conditional density over the covariate sweep as a heatmap.

    ``contours`` iso-density lines (reference style) sit on the heatmap, and
    the payload's quantile curves, when it carries them, are drawn as fitted
    lines of increasing width so the fan reads as a set of centiles.
    """
    x = np.asarray(payload.x, dtype=np.float64)
    y_grid = np.asarray(payload.y_grid, dtype=np.float64)
    density = np.asarray(payload.density, dtype=np.float64)
    with matplotlib_context():
        fig, ax = _single_panel(ax)
        mesh = ax.pcolormesh(x, y_grid, density.T, cmap=sequential_cmap(), shading="auto")
        fig.colorbar(mesh, ax=ax, label="density")
        positive = density[density > 0.0]
        if contours > 0 and positive.size and len(x) > 1:
            levels = np.unique(np.quantile(positive, np.linspace(0.5, 0.98, contours)))
            if levels.size >= 2:
                ax.contour(
                    x,
                    y_grid,
                    density.T,
                    levels=levels,
                    colors=[CHART["original"]["color"]],
                    linewidths=CHART["original"]["width"] * 0.6,
                    linestyles="solid",
                )
        if payload.quantiles is not None and payload.quantile_levels:
            widths = np.linspace(1.2, CHART["edited"]["width"], len(payload.quantile_levels))
            for level, curve, width in zip(
                payload.quantile_levels, np.asarray(payload.quantiles), widths, strict=True
            ):
                ax.plot(
                    x, curve, color=CHART["edited"]["color"], linewidth=width, label=f"q{level:g}"
                )
            ax.legend(loc="upper left", frameon=False)
        _level_ticks(ax, x, payload.levels)
        ax.set_title(f"conditional density along {payload.covariate}")
        ax.set_xlabel(payload.covariate)
        ax.set_ylabel("response")
        return _finish(fig, [ax])


def _draw_histogram(ax: Axes, histogram: Histogram, title: str, xlabel: str) -> None:
    edges = np.asarray(histogram.edges, dtype=np.float64)
    ax.bar(
        edges[:-1],
        np.asarray(histogram.counts, dtype=np.float64),
        width=np.diff(edges),
        align="edge",
        **_bars(CHART["exposure"]),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("rows")


def plot_spread(payload: Spread, *, fig: Figure | None = None) -> Figure:
    """Draw the sharpness histograms and the spread among identically priced rows."""
    names = list(payload.parameters)
    table = payload.identically_priced
    with matplotlib_context():
        fig, axes = _panel_grid(fig, len(names) + 2, 2)
        for ax, name in zip(axes, names, strict=False):
            _draw_histogram(ax, payload.parameters[name], f"fitted {name}", name)
        _draw_histogram(
            axes[len(names)],
            payload.tail_quantile,
            f"predicted q{payload.tail_p:g}",
            f"q{payload.tail_p:g} of the response",
        )
        ax = axes[-1]
        mean = np.asarray(table["mean"], dtype=np.float64)
        low = np.asarray(table["p_lo"], dtype=np.float64)
        high = np.asarray(table["p_hi"], dtype=np.float64)
        ax.vlines(
            mean,
            low,
            high,
            colors=[_colour(CHART["ci_whisker"])],
            linewidths=CHART["ci_whisker"]["width"],
        )
        for center, top, ratio in zip(mean, high, table["ratio"], strict=True):
            if np.isfinite(ratio):
                # Upright labels: a bin whose cheapest row is near-impossible
                # carries a ratio of many orders of magnitude, and lying them
                # flat runs them into one another.
                _label(
                    ax,
                    center,
                    top,
                    f"{float(ratio):.3g}x",
                    ha="center",
                    va="bottom",
                    rotation=90,
                )
        ax.set_title(f"identically priced rows: spread of P(Y > {payload.threshold:.3g})")
        ax.set_xlabel(f"predicted {payload.by} in {payload.n_bins} equal-count bins")
        ax.set_ylabel("exceedance probability")
        return _finish(fig, axes)


def plot_portfolio(payload: Portfolio, *, ax: Axes | None = None) -> Figure:
    """Draw the simulated total loss per segment, and the book's own histogram.

    The histogram of ``total_draws`` is a second panel below the quantile rows,
    so a caller who supplies ``ax`` gets the rows on it and no histogram.
    """
    table = payload.by_segment
    labels: list[str] = [] if table is None else [str(value) for value in table["segment"]]
    columns = [f"q{value:g}" for value in payload.quantiles]
    rows: list[NDArray[np.float64]] = []
    if table is not None:
        rows = [np.asarray(table[column], dtype=np.float64) for column in columns]
    quantile_rows = (
        np.column_stack(rows) if rows else np.empty((0, len(payload.quantiles)), dtype=np.float64)
    )
    book = np.array([payload.total_quantiles[value] for value in payload.quantiles])
    values = np.vstack([quantile_rows, book])
    labels.append("all")
    positions = np.arange(len(labels), dtype=np.float64)
    means = np.append(
        np.asarray(table["mean_total"], dtype=np.float64) if table is not None else [],
        payload.total_mean,
    )
    draws = payload.total_draws
    with matplotlib_context():
        if ax is None and draws is not None:
            fig, axes = _panel_grid(None, 2, 1)
            ax = axes[0]
        else:
            fig, ax = _single_panel(ax)
            axes = [ax]
        ax.hlines(
            positions,
            values.min(axis=1),
            values.max(axis=1),
            colors=[_colour(CHART["ci_whisker"])],
            linewidths=CHART["ci_whisker"]["width"],
        )
        ax.scatter(
            values.ravel(),
            np.repeat(positions, values.shape[1]),
            **_points(CHART["point"]),
        )
        ax.scatter(means, positions, marker="D", color=_colour(CHART["edited"]), s=30.0, zorder=4)
        for quantile, value in zip(payload.quantiles, values[-1], strict=True):
            _label(
                ax,
                float(value),
                float(positions[-1]) + 0.12,
                f"q{quantile:g}",
                ha="center",
                va="bottom",
                rotation=90,
            )
        ax.set_ylim(-0.6, float(positions[-1]) + 0.9)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_title(
            f"simulated total loss over {payload.n_draws} draws"
            f"{' with parameter uncertainty' if payload.parameter_uncertainty else ''}"
        )
        ax.set_xlabel("total")
        if len(axes) > 1 and draws is not None:
            axes[1].hist(np.asarray(draws, dtype=np.float64), bins=30, **_bars(CHART["exposure"]))
            axes[1].set_title("book total per draw")
            axes[1].set_xlabel("total")
            axes[1].set_ylabel("draws")
        return _finish(fig, axes)


# --------------------------------------------------------------------------- #
# The six-panel diagnostics figure
# --------------------------------------------------------------------------- #


def _draw_residual_density(ax: Axes, values: NDArray[np.float64]) -> None:
    ax.hist(values, bins=40, density=True, **_bars(CHART["exposure"]))
    grid = np.linspace(-4.0, 4.0, 200)
    ax.plot(grid, stats.norm.pdf(grid), **_line(CHART["original"]))
    ax.set_title("residual density against N(0, 1)")
    ax.set_xlabel("quantile residual")
    ax.set_ylabel("density")


def _draw_residuals_against(ax: Axes, x: NDArray, y: NDArray, max_points: int) -> None:
    if len(x) <= max_points:
        ax.scatter(x, y, **_points(CHART["point"], s=10.0, linewidths=0.7, zorder=2))
    else:
        ax.hexbin(x, y, gridsize=min(80, max(30, len(x) // 5000)), cmap=sequential_cmap(), mincnt=1)
    ax.axhline(0.0, **_line(CHART["zero"]))


def _draw_residual_sd(ax: Axes, x: NDArray, y: NDArray, n_bins: int = 20) -> int:
    """Draw the residual spread in equal-count bins of ``x``; return the bin count."""
    order = np.argsort(x)
    # Two rows to a bin at least: a standard deviation of one row is not one.
    groups = np.array_split(order, max(1, min(n_bins, len(order) // 2)))
    centers = np.array([float(np.mean(x[group])) for group in groups])
    spread = np.array([float(np.std(y[group], ddof=1)) for group in groups])
    ax.axhline(1.0, **_line(CHART["zero"]))
    ax.plot(centers, spread, **_line(CHART["edited"]))
    ax.scatter(centers, spread, **_points(CHART["point"]))
    return len(groups)


def plot_diagnostics_figure(
    qq: QQPayload,
    worm: WormPayload,
    pit: PITPayload,
    residuals: ResidualSet,
    *,
    max_points: int = 50_000,
) -> Figure:
    """Draw the six-panel distributional diagnostic."""
    index = replication_sample(residuals)
    values = np.asarray(residuals.quantile, dtype=np.float64)[index]
    eta = np.asarray(residuals.eta, dtype=np.float64)[index]
    scale_eta = eta[:, 1] if eta.shape[1] >= 2 else eta[:, 0]
    scale_name = "scale" if eta.shape[1] >= 2 else "location"
    with matplotlib_context():
        fig, axes = _panel_grid(None, 6, 2)
        _draw_qq(axes[0], qq)
        rows = _q_statistic_rows(worm)
        panel = worm.panels[0]
        _draw_worm_panel(axes[1], panel, rows.get(panel.label))
        _draw_pit(axes[2], pit)
        _draw_residual_density(axes[3], values)
        _draw_residuals_against(axes[4], eta[:, 0], values, max_points)
        axes[4].set_title("quantile residuals against the location eta")
        axes[4].set_xlabel("location eta")
        axes[4].set_ylabel("quantile residual")
        bins = _draw_residual_sd(axes[5], scale_eta, values)
        axes[5].set_title(f"residual sd in {bins} bins of the {scale_name} eta")
        axes[5].set_xlabel(f"{scale_name} eta")
        axes[5].set_ylabel("residual sd")
        return _finish(fig, axes)
