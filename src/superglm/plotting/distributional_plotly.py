"""Plotly renderers for the distributional checking and inference payloads.

Every function here takes one payload built in ``superglm.distributional`` and
returns a :class:`plotly.graph_objects.Figure` drawn in the notebook editor's
grammar: the ``superglm_editor`` template registered on import of this module,
the blue of the model under study, the grey dashed reference, the soft-blue
band, the yellow exposure bar and the red flag. Every colour is read from
:mod:`superglm.plotting.editor_style` rather than written here, so the editor's
palette and the suite's cannot drift apart.

The matplotlib renderers in ``superglm.plotting.distributional`` draw the same
payloads with the same semantics; these add what an interactive figure can say
that a static one cannot -- the numbers on hover, and the green of a selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from superglm.plotting.editor_style import (
    BODY_PT,
    CHART,
    DIVERGING,
    LABEL_PT,
    PANEL,
    SEQUENTIAL,
    TOKENS,
    register_plotly_template,
)

try:  # pragma: no cover - exercised by the absence of the dependency
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    raise ImportError(
        "plotly is required for superglm.plotting.distributional_plotly; "
        "install it with: pip install plotly"
    ) from None

__all__ = [
    "TEMPLATE",
    "plotly_actual_expected",
    "plotly_binned",
    "plotly_binned_2d",
    "plotly_calibration",
    "plotly_comparison",
    "plotly_density_fan",
    "plotly_diagnostics_figure",
    "plotly_pit",
    "plotly_portfolio",
    "plotly_qq",
    "plotly_risk_curves",
    "plotly_spread",
    "plotly_term_effect",
    "plotly_term_grid",
    "plotly_worm",
]


# --------------------------------------------------------------------------- #
# The grammar: the editor's classes as plotly properties
# --------------------------------------------------------------------------- #


def _rgba(color: str | tuple[int, int, int], alpha: float) -> str:
    """``rgba(r, g, b, a)`` from a chart colour: a hex string or an rgb triple."""
    if isinstance(color, str):
        text = color.lstrip("#")
        red, green, blue = (int(text[index : index + 2], 16) for index in (0, 2, 4))
    else:
        red, green, blue = (int(channel) for channel in color)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _dash(pattern: tuple[int, int]) -> str:
    """A CSS dash array as plotly's dash-length string: ``(7, 5)`` -> ``7px,5px``."""
    return ",".join(f"{int(length)}px" for length in pattern)


#: ``.ci``: every pointwise band, envelope and consistency band.
BAND_FILL = _rgba(CHART["ci"]["color"], CHART["ci"]["alpha"])
#: ``.ci-whisker``: simultaneous outlines and every standard-error whisker.
WHISKER_COLOR = _rgba(CHART["ci_whisker"]["color"], CHART["ci_whisker"]["alpha"])
#: The editor's active-basis green, reserved for an interactive selection.
SELECTED_COLOR = "rgba(22, 163, 74, 0.62)"
#: ``.exposure``: histogram bars, bin counts and exposure strips.
EXPOSURE_FILL = _rgba(CHART["exposure"]["fill"], CHART["exposure"]["alpha"])
_FITTED = dict(color=CHART["edited"]["color"], width=CHART["edited"]["width"])
_REFERENCE = dict(
    color=CHART["original"]["color"],
    width=CHART["original"]["width"],
    dash=_dash(CHART["original"]["dash"]),
)
_ZERO = dict(
    color=CHART["zero"]["color"],
    width=CHART["zero"]["width"],
    dash=_dash(CHART["zero"]["dash"]),
)
_OUTLINE = dict(color=WHISKER_COLOR, width=CHART["ci_whisker"]["width"])
_EXPOSURE_MARKER = dict(
    color=EXPOSURE_FILL,
    line=dict(color=CHART["exposure"]["edge"], width=CHART["exposure"]["width"]),
)
_SEQUENTIAL_SCALE = [[0.0, SEQUENTIAL[0]], [1.0, SEQUENTIAL[1]]]
_DIVERGING_SCALE = [[0.0, DIVERGING[0]], [0.5, DIVERGING[1]], [1.0, DIVERGING[2]]]

#: One panel is 940 x 520; a grid keeps that aspect per panel within this width.
_PANEL_WIDTH = int(PANEL["width_in"] * 100)
_PANEL_HEIGHT = int(PANEL["height_in"] * 100)
_MAX_WIDTH = 1400
_ASPECT = _PANEL_WIDTH / _PANEL_HEIGHT
#: Bins of the residual histogram and of an equal-count trend line.
_DENSITY_BINS = 60
_TREND_BINS = 20
#: Above this many level labels the tick text is laid at an angle to stay legible.
_CROWDED_LEVELS = 12
#: Angle a crowded level axis writes its labels at.
_CROWDED_ROTATION = 45


def _line(style: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A fresh copy of one of the grammar's line styles."""
    return {**style, **overrides}


def _point_marker(flagged: Any = None, *, size: float = 6.0) -> dict[str, Any]:
    """``.point``, turning ``.point.selected`` red wherever ``flagged`` is true."""
    face, edge = CHART["point"]["face"], CHART["point"]["edge"]
    width = CHART["point"]["width"]
    if flagged is None:
        return dict(size=size, color=face, line=dict(color=edge, width=width))
    flags = np.asarray(flagged, dtype=bool)
    return dict(
        size=size,
        color=np.where(flags, CHART["point_selected"]["face"], face).tolist(),
        line=dict(
            color=np.where(flags, CHART["point_selected"]["edge"], edge).tolist(),
            width=width,
        ),
    )


def _selection() -> dict[str, Any]:
    """The interactive selected state: the editor's active green."""
    return dict(marker=dict(color=SELECTED_COLOR))


def _whisker(array: Any, arrayminus: Any = None) -> dict[str, Any]:
    """``.ci-whisker`` error bars, symmetric unless ``arrayminus`` is given."""
    bars = dict(
        type="data",
        array=np.asarray(array, dtype=np.float64),
        color=WHISKER_COLOR,
        thickness=CHART["ci_whisker"]["width"],
        width=0,
    )
    if arrayminus is not None:
        bars["symmetric"] = False
        bars["arrayminus"] = np.asarray(arrayminus, dtype=np.float64)
    return bars


def _sequential_color(position: float) -> str:
    """A hex colour ``position`` of the way along the single-hue scale."""
    ends = [
        np.array([int(value.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)
        for value in SEQUENTIAL
    ]
    blended = ends[0] + (ends[1] - ends[0]) * float(np.clip(position, 0.0, 1.0))
    return "#{:02x}{:02x}{:02x}".format(*(int(round(channel)) for channel in blended))


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


# --------------------------------------------------------------------------- #
# Figure plumbing
# --------------------------------------------------------------------------- #


def _grid(count: int, *, cols: int = 2) -> tuple[int, int]:
    """Rows and columns for ``count`` panels, one column when there is one panel."""
    columns = 1 if count <= 1 else int(cols)
    return int(np.ceil(count / columns)), columns


def _geometry(rows: int, cols: int) -> tuple[int, int]:
    """The figure size that keeps the editor's panel aspect in every cell."""
    width = min(_PANEL_WIDTH * cols, _MAX_WIDTH) if cols > 1 else _PANEL_WIDTH
    return int(width), int(round(width / cols / _ASPECT * rows))


def _finalise(
    fig: go.Figure,
    *,
    rows: int = 1,
    cols: int = 1,
    title: str | None = None,
    showlegend: bool = True,
) -> go.Figure:
    """Put the editor template, the panel geometry and the legend on ``fig``."""
    width, height = _geometry(rows, cols)
    fig.update_layout(
        template=TEMPLATE,
        width=width,
        height=height,
        showlegend=showlegend,
        hovermode="closest",
        margin=dict(l=70, r=40, t=90 if title else 56, b=76),
        legend=dict(orientation="h", yanchor="top", y=-0.09, xanchor="center", x=0.5),
    )
    if title is not None:
        fig.update_layout(
            title=dict(text=title, x=0.0, xref="paper", xanchor="left", yref="container", y=0.98)
        )
    return fig


def _style_titles(fig: go.Figure, colors: Sequence[str] | None = None) -> None:
    """Give the subplot titles the editor's body type, red where flagged."""
    for index, note in enumerate(fig.layout.annotations):
        color = TOKENS["text"] if colors is None else colors[index]
        note.font.update(size=BODY_PT, color=color)


def _axis_titles(fig: go.Figure, x: str | None = None, y: str | None = None, **cell: Any) -> None:
    if x is not None:
        fig.update_xaxes(title_text=x, title_font=dict(size=LABEL_PT), **cell)
    if y is not None:
        fig.update_yaxes(title_text=y, title_font=dict(size=LABEL_PT), **cell)


def _categorical_ticks(fig: go.Figure, positions: Any, levels: Sequence[str], **cell: Any) -> None:
    """Label a numeric sweep axis with the level names it stands for.

    Past a dozen levels horizontal labels collide, so a crowded axis writes
    them at an angle rather than thinning them out and leaving a reader unable
    to say which mark is which.
    """
    fig.update_xaxes(
        tickmode="array",
        tickvals=np.asarray(positions, dtype=np.float64),
        ticktext=[str(level) for level in levels],
        tickangle=_CROWDED_ROTATION if len(levels) > _CROWDED_LEVELS else 0,
        **cell,
    )


def _cell(row: int | None, col: int | None) -> dict[str, int]:
    return {} if row is None else {"row": row, "col": int(col or 1)}


def _band_trace(
    x: Any,
    lower: Any,
    upper: Any,
    *,
    name: str,
    showlegend: bool = True,
) -> go.Scatter:
    """``.ci``: a filled band with no stroke, closed around ``x``."""
    grid = np.asarray(x, dtype=np.float64)
    return go.Scatter(
        x=np.concatenate([grid, grid[::-1]]),
        y=np.concatenate(
            [np.asarray(upper, dtype=np.float64), np.asarray(lower, dtype=np.float64)[::-1]]
        ),
        fill="toself",
        fillcolor=BAND_FILL,
        line=dict(width=0),
        mode="lines",
        hoverinfo="skip",
        name=name,
        showlegend=showlegend,
    )


def _flat_line(x: Any, value: float, *, name: str, style: dict[str, Any]) -> go.Scatter:
    """A constant reference line spanning the domain of ``x``."""
    grid = np.asarray(x, dtype=np.float64)
    span = [float(np.nanmin(grid)), float(np.nanmax(grid))]
    return go.Scatter(
        x=span,
        y=[float(value), float(value)],
        mode="lines",
        line=_line(style),
        hoverinfo="skip",
        name=name,
        showlegend=False,
    )


def _centers(edges: Any) -> np.ndarray:
    values = np.asarray(edges, dtype=np.float64)
    return 0.5 * (values[:-1] + values[1:])


def _scatter_or_binned(
    fig: go.Figure,
    x: Any,
    y: Any,
    *,
    name: str,
    max_points: int,
    hovertemplate: str,
    row: int | None = None,
    col: int | None = None,
    flagged: Any = None,
) -> None:
    """Markers below ``max_points`` rows, a sequential 2-D histogram above."""
    horizontal = np.asarray(x, dtype=np.float64)
    vertical = np.asarray(y, dtype=np.float64)
    cell = _cell(row, col)
    if len(horizontal) > int(max_points):
        fig.add_trace(
            go.Histogram2d(
                x=horizontal,
                y=vertical,
                colorscale=_SEQUENTIAL_SCALE,
                nbinsx=60,
                nbinsy=60,
                name=name,
                showscale=False,
                hovertemplate="%{x:.3g}, %{y:.3g}<br>rows %{z:,.0f}<extra></extra>",
            ),
            **cell,
        )
        return
    fig.add_trace(
        go.Scatter(
            x=horizontal,
            y=vertical,
            mode="markers",
            name=name,
            marker=_point_marker(flagged),
            selected=_selection(),
            unselected=dict(marker=dict(opacity=0.3)),
            hovertemplate=hovertemplate,
            showlegend=False,
        ),
        **cell,
    )


def _equal_count_sd(x: Any, y: Any, n_bins: int = _TREND_BINS) -> tuple[np.ndarray, np.ndarray]:
    """Bin ``x`` into equal-count bins and take the standard deviation of ``y``."""
    horizontal = np.asarray(x, dtype=np.float64)
    vertical = np.asarray(y, dtype=np.float64)
    order = np.argsort(horizontal)
    centers, spreads = [], []
    for chunk in np.array_split(order, min(int(n_bins), max(len(order), 1))):
        if len(chunk) < 2:
            continue
        centers.append(float(np.median(horizontal[chunk])))
        spreads.append(float(np.std(vertical[chunk], ddof=1)))
    return np.asarray(centers, dtype=np.float64), np.asarray(spreads, dtype=np.float64)


TEMPLATE = register_plotly_template()


# --------------------------------------------------------------------------- #
# Q-Q, worm and PIT
# --------------------------------------------------------------------------- #


def _qq_panel(
    fig: go.Figure,
    payload: Any,
    *,
    max_points: int,
    row: int | None = None,
    col: int | None = None,
) -> None:
    theoretical = np.asarray(payload.theoretical, dtype=np.float64)
    observed = np.asarray(payload.observed, dtype=np.float64)
    lower = np.asarray(payload.envelope_lower, dtype=np.float64)
    upper = np.asarray(payload.envelope_upper, dtype=np.float64)
    cell = _cell(row, col)

    fig.add_trace(
        _band_trace(
            theoretical,
            lower,
            upper,
            name=f"{payload.n_sim}-replicate envelope",
            showlegend=row is None,
        ),
        **cell,
    )
    span = [float(theoretical.min()), float(theoretical.max())]
    fig.add_trace(
        go.Scatter(
            x=span,
            y=span,
            mode="lines",
            line=_line(_REFERENCE),
            name="theoretical",
            hoverinfo="skip",
            showlegend=row is None,
        ),
        **cell,
    )
    outside = (observed < lower) | (observed > upper)
    _scatter_or_binned(
        fig,
        theoretical,
        observed,
        name="quantile residuals",
        max_points=max_points,
        hovertemplate=(
            "theoretical %{x:.3f}<br>observed %{y:.3f}"
            "<br>envelope %{customdata[0]:.3f} to %{customdata[1]:.3f}<extra></extra>"
        ),
        flagged=outside,
        **cell,
    )
    if isinstance(fig.data[-1], go.Scatter):
        fig.data[-1].update(customdata=np.column_stack([lower, upper]))


def plotly_qq(payload: Any) -> go.Figure:
    """The Q-Q plot of the quantile residuals against its simulation envelope."""
    fig = go.Figure()
    _qq_panel(fig, payload, max_points=len(payload.observed) + 1)
    subsample = " on a subsample" if payload.subsampled else ""
    _axis_titles(fig, x="theoretical quantile", y="observed quantile")
    return _finalise(
        fig,
        title=(
            f"Q-Q of quantile residuals — {payload.n_rows:,} rows, "
            f"{payload.n_sim} replicates{subsample}"
        ),
    )


def _worm_panel(
    fig: go.Figure,
    panel: Any,
    *,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
) -> None:
    band_z = np.asarray(panel.band_z, dtype=np.float64)
    half = np.asarray(panel.band, dtype=np.float64)
    cell = _cell(row, col)
    fig.add_trace(
        _band_trace(band_z, -half, half, name="pointwise envelope", showlegend=showlegend), **cell
    )
    fig.add_trace(_flat_line(band_z, 0.0, name="theoretical", style=_REFERENCE), **cell)
    fig.add_trace(
        go.Scatter(
            x=np.asarray(panel.z, dtype=np.float64),
            y=np.asarray(panel.deviation, dtype=np.float64),
            mode="lines",
            line=_line(_FITTED),
            name=str(panel.label),
            showlegend=showlegend,
            hovertemplate="z %{x:.3f}<br>deviation %{y:.4f}<extra></extra>",
        ),
        **cell,
    )


def _worm_titles(payload: Any) -> tuple[list[str], list[str]]:
    """Panel titles and their colour: red where a Q-statistic exceeds two."""
    flagged: set[str] = set()
    table = payload.q_statistics
    if table is not None and len(table):
        flagged = {str(name) for name in table.loc[table["flagged"], "group"]}
    titles = [f"{panel.label} · n = {panel.n:,}" for panel in payload.panels]
    colors = [
        CHART["point_selected"]["face"] if str(panel.label) in flagged else TOKENS["text"]
        for panel in payload.panels
    ]
    return titles, colors


def plotly_worm(payload: Any) -> go.Figure:
    """One worm per covariate interval, each against its pointwise envelope."""
    titles, colors = _worm_titles(payload)
    rows, cols = _grid(len(payload.panels))
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.13, horizontal_spacing=0.08
    )
    for index, panel in enumerate(payload.panels):
        row, col = divmod(index, cols)
        _worm_panel(fig, panel, row=row + 1, col=col + 1, showlegend=index == 0)
    _style_titles(fig, colors)
    _axis_titles(fig, x="normal quantile", y="deviation")
    covariate = "" if payload.covariate is None else f" by {payload.covariate}"
    return _finalise(fig, rows=rows, cols=cols, title=f"Worm plot{covariate}")


def _pit_panel(
    fig: go.Figure,
    payload: Any,
    *,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
) -> None:
    edges = np.asarray(payload.edges, dtype=np.float64)
    counts = np.asarray(payload.counts, dtype=np.float64)
    centers = _centers(edges)
    span = np.array([float(edges[0]), float(edges[-1])])
    cell = _cell(row, col)

    fig.add_trace(
        _band_trace(
            span,
            np.full(2, payload.band_lower),
            np.full(2, payload.band_upper),
            name="consistency band",
            showlegend=showlegend,
        ),
        **cell,
    )
    fig.add_trace(_flat_line(span, payload.expected, name="uniform", style=_ZERO), **cell)
    outside = (counts < payload.band_lower) | (counts > payload.band_upper)
    marker = dict(_EXPOSURE_MARKER)
    marker["color"] = np.where(outside, CHART["point_selected"]["face"], EXPOSURE_FILL).tolist()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=float(np.min(np.diff(edges))) * 0.96,
            marker=marker,
            name="PIT counts",
            showlegend=showlegend,
            hovertemplate=(
                "PIT %{x:.3f}<br>count %{y:,.0f}"
                f"<br>expected {payload.expected:,.1f}"
                f" ({payload.band_lower:,.1f} to {payload.band_upper:,.1f})<extra></extra>"
            ),
        ),
        **cell,
    )


def plotly_pit(payload: Any) -> go.Figure:
    """The PIT histogram against the uniform count and its consistency band."""
    fig = go.Figure()
    _pit_panel(fig, payload)
    _axis_titles(fig, x="probability integral transform", y="rows")
    return _finalise(fig, title=f"PIT histogram — {payload.n_bins} bins, {payload.n_rows:,} rows")


# --------------------------------------------------------------------------- #
# Binned residual checks
# --------------------------------------------------------------------------- #


def _binned_statistic(
    fig: go.Figure,
    payload: Any,
    statistic: str,
    reference: float,
    *,
    row: int,
    col: int,
    showlegend: bool,
) -> None:
    centers = np.asarray(payload.centers, dtype=np.float64)
    value = np.asarray(getattr(payload, statistic), dtype=np.float64)
    lower = np.asarray(getattr(payload, f"{statistic}_lower"), dtype=np.float64)
    upper = np.asarray(getattr(payload, f"{statistic}_upper"), dtype=np.float64)
    counts = np.asarray(payload.n, dtype=np.float64)

    fig.add_trace(
        _band_trace(centers, lower, upper, name="bootstrap band", showlegend=showlegend),
        row=row,
        col=col,
    )
    fig.add_trace(
        _flat_line(centers, reference, name=f"{statistic} under the fit", style=_ZERO),
        row=row,
        col=col,
    )
    excluded = (lower > reference) | (upper < reference)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=value,
            mode="lines+markers",
            line=_line(_FITTED),
            marker=_point_marker(excluded),
            selected=_selection(),
            name=statistic,
            showlegend=showlegend,
            customdata=np.column_stack([lower, upper, counts]),
            hovertemplate=(
                f"{payload.covariate} %{{x:.4g}}<br>{statistic} %{{y:.4f}}"
                "<br>band %{customdata[0]:.4f} to %{customdata[1]:.4f}"
                "<br>n = %{customdata[2]:,.0f}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


def plotly_binned(payload: Any) -> go.Figure:
    """Binned mean, standard deviation and skewness with the bin counts."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("mean", "standard deviation", "skewness", "rows per bin"),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    for index, (statistic, reference) in enumerate((("mean", 0.0), ("sd", 1.0), ("skew", 0.0))):
        row, col = divmod(index, 2)
        _binned_statistic(
            fig, payload, statistic, reference, row=row + 1, col=col + 1, showlegend=index == 0
        )

    centers = np.asarray(payload.centers, dtype=np.float64)
    fig.add_trace(
        go.Bar(
            x=centers,
            y=np.asarray(payload.n, dtype=np.float64),
            marker=dict(_EXPOSURE_MARKER),
            name="rows",
            showlegend=False,
            hovertemplate=f"{payload.covariate} %{{x:.4g}}<br>n = %{{y:,.0f}}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    if payload.levels is not None:
        for row in (1, 2):
            for col in (1, 2):
                _categorical_ticks(fig, centers, payload.levels, row=row, col=col)
    _style_titles(fig)
    _axis_titles(fig, x=payload.covariate)
    return _finalise(fig, rows=2, cols=2, title=f"Binned residual check on {payload.covariate}")


def plotly_binned_2d(payload: Any) -> go.Figure:
    """The signed mean residual over a grid of two covariates, centred at zero."""
    mean = np.asarray(payload.mean, dtype=np.float64)
    count = np.asarray(payload.count, dtype=np.float64)
    fig = go.Figure(
        go.Heatmap(
            x=_centers(payload.x_edges),
            y=_centers(payload.y_edges),
            z=mean.T,
            customdata=count.T,
            zmid=0.0,
            colorscale=_DIVERGING_SCALE,
            colorbar=dict(title=dict(text="mean", font=dict(size=LABEL_PT)), thickness=12),
            hovertemplate=(
                f"{payload.covariates[0]} %{{x:.4g}}<br>{payload.covariates[1]} %{{y:.4g}}"
                "<br>mean %{z:.4f}<br>n = %{customdata:,.0f}<extra></extra>"
            ),
        )
    )
    _axis_titles(fig, x=payload.covariates[0], y=payload.covariates[1])
    return _finalise(
        fig,
        title=f"Mean residual over {payload.covariates[0]} and {payload.covariates[1]}",
        showlegend=False,
    )


# --------------------------------------------------------------------------- #
# Actual versus expected and calibration
# --------------------------------------------------------------------------- #


def plotly_actual_expected(payload: Any) -> go.Figure:
    """The model's expectation against the realised total, and their ratio."""
    centers = np.asarray(payload.centers, dtype=np.float64)
    actual = np.asarray(payload.actual, dtype=np.float64)
    expected = np.asarray(payload.expected, dtype=np.float64)
    ratio = np.asarray(payload.ratio, dtype=np.float64)
    error = np.asarray(payload.ratio_se, dtype=np.float64)
    counts = np.asarray(payload.n, dtype=np.float64)
    weight = np.asarray(payload.weight, dtype=np.float64)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.46, 0.3, 0.24],
        vertical_spacing=0.07,
        subplot_titles=("actual and expected", "actual / expected", "exposure"),
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=expected,
            mode="lines",
            line=_line(_FITTED),
            name="expected",
            hovertemplate="%{x:.4g}<br>expected %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=actual,
            mode="markers",
            marker=_point_marker(),
            selected=_selection(),
            name="actual",
            customdata=np.column_stack([counts, weight]),
            hovertemplate=(
                "%{x:.4g}<br>actual %{y:.4f}"
                "<br>n = %{customdata[0]:,.0f}<br>weight %{customdata[1]:,.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(_flat_line(centers, 1.0, name="parity", style=_ZERO), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=ratio,
            mode="markers",
            marker=_point_marker(np.abs(ratio - 1.0) > 2.0 * error),
            selected=_selection(),
            error_y=_whisker(error),
            name="A / E",
            customdata=np.column_stack([error, counts]),
            hovertemplate=(
                "%{x:.4g}<br>A/E %{y:.4f} ± %{customdata[0]:.4f}"
                "<br>n = %{customdata[1]:,.0f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=centers,
            y=weight,
            marker=dict(_EXPOSURE_MARKER),
            name="weight",
            showlegend=False,
            hovertemplate="%{x:.4g}<br>weight %{y:,.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    if payload.levels is not None:
        for row in (1, 2, 3):
            _categorical_ticks(fig, centers, payload.levels, row=row, col=1)
    # A ratio of non-negative totals cannot go below zero, and the whiskers of a
    # thin bin otherwise drag the axis under it; a signed target that really
    # does price a bin below zero keeps its own floor.
    fig.update_yaxes(minallowed=float(np.nanmin(np.append(ratio - error, 0.0))), row=2, col=1)
    _style_titles(fig)
    _axis_titles(fig, x=payload.covariate, row=3, col=1)
    return _finalise(
        fig,
        rows=3,
        title=(
            f"Actual versus expected on {payload.covariate} — "
            f"{payload.variance_law} variance, {payload.weight_semantics} weights"
        ),
    )


def _overall(table: Any) -> Any:
    """The rows of a calibration table that describe the whole book."""
    return table[table["group"] == "all"]


def plotly_calibration(payload: Any) -> go.Figure:
    """Coverage by level, the tail table, quantile calibration and reliability."""
    coverage = _overall(payload.coverage)
    tails = _overall(payload.tails)
    quantiles = payload.quantiles
    has_tails = bool(len(tails))
    has_reliability = bool(len(payload.reliability))

    titles = ["coverage by level", "quantile calibration"]
    if has_tails:
        titles.append("tail exceedance")
    if has_reliability:
        titles.append("reliability")
    rows, cols = _grid(len(titles))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.09,
    )

    levels = coverage["level"].to_numpy(dtype=np.float64)
    realised = coverage["realised"].to_numpy(dtype=np.float64)
    error = coverage["se"].to_numpy(dtype=np.float64)
    fig.add_trace(
        go.Scatter(
            x=levels,
            y=levels,
            mode="lines",
            line=_line(_ZERO),
            name="nominal",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=levels,
            y=realised,
            mode="markers",
            marker=_point_marker(np.abs(realised - levels) > 2.0 * error),
            selected=_selection(),
            error_y=_whisker(error),
            name="realised coverage",
            customdata=np.column_stack([error, coverage["n"].to_numpy(dtype=np.float64)]),
            hovertemplate=(
                "level %{x:.3f}<br>realised %{y:.4f} ± %{customdata[0]:.4f}"
                "<br>n = %{customdata[1]:,.0f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    probability = quantiles["p"].to_numpy(dtype=np.float64)
    exceedance = quantiles["realised_exceedance"].to_numpy(dtype=np.float64)
    quantile_error = quantiles["se"].to_numpy(dtype=np.float64)
    second = (1, 2) if cols == 2 else (2, 1)
    fig.add_trace(
        go.Scatter(
            x=probability,
            y=quantiles["expected"].to_numpy(dtype=np.float64),
            mode="lines",
            line=_line(_ZERO),
            name="1 − p",
            hoverinfo="skip",
        ),
        row=second[0],
        col=second[1],
    )
    fig.add_trace(
        go.Scatter(
            x=probability,
            y=exceedance,
            mode="markers",
            marker=_point_marker(
                np.abs(exceedance - quantiles["expected"].to_numpy(dtype=np.float64))
                > 2.0 * quantile_error
            ),
            selected=_selection(),
            error_y=_whisker(quantile_error),
            name="realised exceedance",
            hovertemplate="p %{x:.3f}<br>exceedance %{y:.4f}<extra></extra>",
        ),
        row=second[0],
        col=second[1],
    )

    if has_tails:
        thresholds = tails["threshold"].to_numpy(dtype=np.float64)
        fig.add_trace(
            go.Bar(
                x=thresholds,
                y=tails["expected"].to_numpy(dtype=np.float64),
                marker=dict(color=CHART["edited"]["color"]),
                name="expected exceedances",
                hovertemplate="threshold %{x:.4g}<br>expected %{y:,.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        tail_error = tails["se"].to_numpy(dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=tails["realised"].to_numpy(dtype=np.float64),
                mode="markers",
                marker=_point_marker(
                    np.abs(
                        tails["realised"].to_numpy(dtype=np.float64)
                        - tails["expected"].to_numpy(dtype=np.float64)
                    )
                    > 2.0 * tail_error
                ),
                selected=_selection(),
                error_y=_whisker(tail_error),
                name="realised exceedances",
                customdata=np.column_stack([tails["log_score"].to_numpy(dtype=np.float64)]),
                hovertemplate=(
                    "threshold %{x:.4g}<br>realised %{y:,.2f}"
                    "<br>log score %{customdata[0]:.4f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=thresholds,
            ticktext=[
                _tail_tick(threshold, group)
                for threshold, group in zip(thresholds, tails["group"], strict=True)
            ],
            row=2,
            col=1,
        )

    if has_reliability:
        cell = dict(row=2, col=2)
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line=_line(_REFERENCE),
                name="perfect calibration",
                hoverinfo="skip",
            ),
            **cell,
        )
        # One colour per threshold along the single-hue scale, as the risk
        # curves do: three exceedance levels drawn in one blue say which curve
        # is which only in the legend.
        count = len(payload.reliability)
        for index, (threshold, curve) in enumerate(payload.reliability.items()):
            position = 1.0 if count == 1 else index / (count - 1)
            grid = np.asarray(curve.x, dtype=np.float64)
            fig.add_trace(
                _band_trace(
                    grid,
                    np.asarray(curve.lower, dtype=np.float64),
                    np.asarray(curve.upper, dtype=np.float64),
                    name=f"{threshold:.4g} band",
                    showlegend=False,
                ),
                **cell,
            )
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=np.asarray(curve.calibrated, dtype=np.float64),
                    mode="lines",
                    line=_line(_FITTED, color=_sequential_color(position)),
                    name=f"reliability at {threshold:.4g}",
                    customdata=np.column_stack([np.asarray(curve.count, dtype=np.float64)]),
                    hovertemplate=(
                        "predicted %{x:.4f}<br>observed %{y:.4f}"
                        "<br>n = %{customdata[0]:,.0f}<extra></extra>"
                    ),
                ),
                **cell,
            )

    _style_titles(fig)
    return _finalise(
        fig,
        rows=rows,
        cols=cols,
        title=(f"Calibration — {payload.n_rows:,} rows, {payload.weight_semantics} weights"),
    )


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #


def plotly_comparison(payload: Any) -> go.Figure:
    """The score difference by segment, and the Murphy diagram behind it."""
    segments = payload.by_segment
    if segments is None:
        labels = ["all"]
        difference = np.array([float(payload.overall["mean_diff"])])
        error = np.array([float(payload.overall["se"])])
        counts = np.array([float(payload.overall["n"])])
        statistic = np.array([float(payload.overall["t"])])
    else:
        labels = [str(name) for name in segments.index]
        difference = segments["mean_diff"].to_numpy(dtype=np.float64)
        error = segments["se"].to_numpy(dtype=np.float64)
        counts = segments["n"].to_numpy(dtype=np.float64)
        statistic = segments["t"].to_numpy(dtype=np.float64)

    murphy = payload.murphy
    titles = [f"{payload.score} score difference by segment"]
    if murphy is not None:
        titles += [f"Murphy diagram at quantile {murphy.level:g}", "elementary score difference"]
    fig = make_subplots(rows=len(titles), cols=1, subplot_titles=titles, vertical_spacing=0.12)

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=np.zeros(len(labels)),
            mode="lines",
            line=_line(_ZERO),
            name="no difference",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=difference,
            mode="markers",
            marker=_point_marker(np.abs(statistic) > 2.0),
            selected=_selection(),
            error_y=_whisker(error),
            name="mean difference",
            customdata=np.column_stack([error, counts, statistic]),
            hovertemplate=(
                "%{x}<br>difference %{y:.5f} ± %{customdata[0]:.5f}"
                "<br>n = %{customdata[1]:,.0f}<br>t %{customdata[2]:.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    if murphy is not None:
        thresholds = np.asarray(murphy.thresholds, dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=np.asarray(murphy.a, dtype=np.float64),
                mode="lines",
                line=_line(_FITTED),
                name="model a",
                hovertemplate="threshold %{x:.4g}<br>score %{y:.5f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=np.asarray(murphy.b, dtype=np.float64),
                mode="lines",
                line=_line(_REFERENCE),
                name="model b",
                hovertemplate="threshold %{x:.4g}<br>score %{y:.5f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        gap = np.asarray(murphy.difference, dtype=np.float64)
        margin = 1.96 * np.asarray(murphy.difference_se, dtype=np.float64)
        fig.add_trace(
            _band_trace(thresholds, gap - margin, gap + margin, name="95% band"), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=thresholds,
                y=gap,
                mode="lines",
                line=_line(_FITTED),
                name="a − b",
                customdata=np.column_stack([margin]),
                hovertemplate=(
                    "threshold %{x:.4g}<br>a − b %{y:.5f} ± %{customdata[0]:.5f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(_flat_line(thresholds, 0.0, name="no difference", style=_ZERO), row=3, col=1)

    _style_titles(fig)
    _axis_titles(fig, y="difference", row=1, col=1)
    overall = payload.overall
    return _finalise(
        fig,
        rows=len(titles),
        title=(
            f"{payload.score} score: mean difference {float(overall['mean_diff']):.5f} "
            f"± {float(overall['se']):.5f} (t = {float(overall['t']):.2f}, "
            f"n = {int(overall['n']):,})"
        ),
    )


# --------------------------------------------------------------------------- #
# Per-parameter term effects
# --------------------------------------------------------------------------- #


def _effect_domain(effect: Any) -> tuple[Any, bool]:
    """The plotting domain of a term: its levels, or its swept x."""
    if effect.levels is not None:
        return [str(level) for level in effect.levels], True
    return np.asarray(effect.x, dtype=np.float64), False


def _effect_hover(effect: Any) -> tuple[np.ndarray, str]:
    """Hover columns and template: the estimate, its error and its band."""
    columns = [
        np.asarray(effect.se, dtype=np.float64),
        np.asarray(effect.lower, dtype=np.float64),
        np.asarray(effect.upper, dtype=np.float64),
    ]
    template = (
        f"{effect.term} %{{x}}<br>{effect.parameter} effect %{{y:.4f}}"
        "<br>se %{customdata[0]:.4f}"
        "<br>band %{customdata[1]:.4f} to %{customdata[2]:.4f}"
    )
    if effect.multiplier is not None:
        columns.append(np.asarray(effect.multiplier, dtype=np.float64))
        template += "<br>multiplier %{customdata[3]:.4f}"
    return np.column_stack(columns), template + "<extra></extra>"


def _effect_traces(
    fig: go.Figure,
    effect: Any,
    *,
    row: int | None = None,
    col: int | None = None,
    showlegend: bool = True,
) -> None:
    domain, categorical = _effect_domain(effect)
    value = np.asarray(effect.effect, dtype=np.float64)
    lower = np.asarray(effect.lower, dtype=np.float64)
    upper = np.asarray(effect.upper, dtype=np.float64)
    simultaneous = effect.lower_simultaneous is not None
    customdata, hovertemplate = _effect_hover(effect)
    name = f"{effect.parameter}:{effect.term}"
    cell = _cell(row, col)

    if categorical:
        fig.add_trace(
            go.Scatter(
                x=domain,
                y=np.zeros(len(domain)),
                mode="lines",
                line=_line(_ZERO),
                name="zero",
                hoverinfo="skip",
                showlegend=False,
            ),
            **cell,
        )
        # A free special level is a different kind of number from a point on
        # the smooth -- fitted on its own rows, sharing none of the smooth's
        # shape -- so it leaves the term's trace and is named in the legend.
        special = (
            np.zeros(len(domain), dtype=bool)
            if effect.special is None
            else np.asarray(effect.special, dtype=bool)
        )
        ordinary = ~special
        fig.add_trace(
            go.Scatter(
                x=[level for level, keep in zip(domain, ordinary, strict=True) if keep],
                y=value[ordinary],
                mode="markers",
                marker=_point_marker((lower[ordinary] > 0.0) | (upper[ordinary] < 0.0), size=9),
                selected=_selection(),
                error_y=_whisker((upper - value)[ordinary], (value - lower)[ordinary]),
                name=name,
                showlegend=showlegend,
                customdata=customdata[ordinary],
                hovertemplate=hovertemplate,
            ),
            **cell,
        )
        if special.any():
            fig.add_trace(
                go.Scatter(
                    x=[level for level, keep in zip(domain, special, strict=True) if keep],
                    y=value[special],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=CHART["point_selected"]["face"],
                        line=dict(
                            color=CHART["point_selected"]["edge"],
                            width=CHART["point"]["width"],
                        ),
                    ),
                    selected=_selection(),
                    error_y=_whisker((upper - value)[special], (value - lower)[special]),
                    name="special",
                    showlegend=showlegend,
                    customdata=customdata[special],
                    hovertemplate=hovertemplate,
                ),
                **cell,
            )
        if simultaneous:
            low = np.asarray(effect.lower_simultaneous, dtype=np.float64)
            high = np.asarray(effect.upper_simultaneous, dtype=np.float64)
            xs: list[Any] = []
            ys: list[Any] = []
            for index, level in enumerate(domain):
                xs += [level, level, None]
                ys += [float(low[index]), float(high[index]), None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=_line(_OUTLINE),
                    name="simultaneous",
                    hoverinfo="skip",
                    showlegend=showlegend,
                ),
                **cell,
            )
        return

    fig.add_trace(
        _band_trace(
            domain,
            lower,
            upper,
            name=f"{int(round((1 - effect.alpha) * 100))}% pointwise",
            showlegend=showlegend,
        ),
        **cell,
    )
    if simultaneous:
        for bound, label in (
            (effect.lower_simultaneous, "simultaneous"),
            (effect.upper_simultaneous, None),
        ):
            fig.add_trace(
                go.Scatter(
                    x=domain,
                    y=np.asarray(bound, dtype=np.float64),
                    mode="lines",
                    line=_line(_OUTLINE),
                    name=label or "simultaneous",
                    showlegend=showlegend and label is not None,
                    hoverinfo="skip",
                ),
                **cell,
            )
    fig.add_trace(_flat_line(domain, 0.0, name="zero", style=_ZERO), **cell)
    fig.add_trace(
        go.Scatter(
            x=domain,
            y=value,
            mode="lines",
            line=_line(_FITTED),
            name=name,
            showlegend=showlegend,
            customdata=customdata,
            hovertemplate=hovertemplate,
        ),
        **cell,
    )


def _effect_title(effect: Any) -> str:
    return (
        f"{effect.parameter}:{effect.term} — {effect.link} link, "
        f"edf {effect.edf:.2f}, {effect.covariance_kind} covariance"
    )


def plotly_term_effect(effect: Any, *, exposure: Any = None) -> go.Figure:
    """One term of one predictor, with its pointwise and simultaneous bands."""
    domain, _ = _effect_domain(effect)
    if exposure is None:
        fig = go.Figure()
        _effect_traces(fig, effect)
        _axis_titles(fig, x=effect.term, y=f"{effect.parameter} effect")
        return _finalise(fig, title=_effect_title(effect))

    strip = np.asarray(exposure, dtype=np.float64)
    if strip.shape != (len(domain),):
        raise ValueError(
            f"exposure must give one exposure per plotted point; got {strip.shape} "
            f"for {len(domain)} points"
        )
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06
    )
    _effect_traces(fig, effect, row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=domain,
            y=strip,
            marker=dict(_EXPOSURE_MARKER),
            name="exposure",
            showlegend=False,
            hovertemplate="%{x}<br>exposure %{y:,.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _axis_titles(fig, x=effect.term, row=2, col=1)
    _axis_titles(fig, y=f"{effect.parameter} effect", row=1, col=1)
    return _finalise(fig, rows=2, title=_effect_title(effect))


def plotly_term_grid(effects: Sequence[Any], *, parameter: str | None = None) -> go.Figure:
    """A grid of term panels, optionally only those of one parameter."""
    selected = [effect for effect in effects if parameter is None or effect.parameter == parameter]
    if not selected:
        raise ValueError(
            f"no term to draw: none of the {len(list(effects))} effects belongs to "
            f"parameter {parameter!r}"
        )
    rows, cols = _grid(len(selected))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{item.parameter}:{item.term}" for item in selected],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    for index, effect in enumerate(selected):
        row, col = divmod(index, cols)
        _effect_traces(fig, effect, row=row + 1, col=col + 1, showlegend=index == 0)
        _axis_titles(fig, x=effect.term, row=row + 1, col=col + 1)
    _style_titles(fig)
    scope = "every parameter" if parameter is None else parameter
    return _finalise(fig, rows=rows, cols=cols, title=f"Term effects — {scope}")


# --------------------------------------------------------------------------- #
# Risk curves, density fan, spread and portfolio
# --------------------------------------------------------------------------- #


def plotly_risk_curves(payload: Any) -> go.Figure:
    """Predicted quantiles along one covariate, each with its posterior band."""
    x = np.asarray(payload.x, dtype=np.float64)
    fig = go.Figure()
    count = len(payload.quantiles)
    for index, quantile in enumerate(payload.quantiles):
        position = 1.0 if count == 1 else index / (count - 1)
        color = _sequential_color(position)
        fig.add_trace(
            _band_trace(
                x,
                np.asarray(payload.lower[index], dtype=np.float64),
                np.asarray(payload.upper[index], dtype=np.float64),
                name=f"q{quantile:g} {payload.level:g} band",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(payload.values[index], dtype=np.float64),
                mode="lines",
                line=dict(color=color, width=CHART["edited"]["width"]),
                name=f"quantile {quantile:g}",
                customdata=np.column_stack(
                    [
                        np.asarray(payload.lower[index], dtype=np.float64),
                        np.asarray(payload.upper[index], dtype=np.float64),
                    ]
                ),
                hovertemplate=(
                    f"{payload.covariate} %{{x:.4g}}<br>quantile {quantile:g} %{{y:.4f}}"
                    "<br>band %{customdata[0]:.4f} to %{customdata[1]:.4f}<extra></extra>"
                ),
            )
        )
    if payload.levels is not None:
        _categorical_ticks(fig, x, payload.levels)
    _axis_titles(fig, x=payload.covariate, y="response")
    return _finalise(
        fig,
        title=(
            f"Risk curves along {payload.covariate} — {payload.n_draws:,} draws, "
            f"{payload.level:g} posterior band"
        ),
    )


def plotly_density_fan(payload: Any) -> go.Figure:
    """The conditional density of the response along one covariate sweep."""
    density = np.asarray(payload.density, dtype=np.float64)
    fig = go.Figure(
        go.Heatmap(
            x=np.asarray(payload.x, dtype=np.float64),
            y=np.asarray(payload.y_grid, dtype=np.float64),
            z=density.T,
            colorscale=_SEQUENTIAL_SCALE,
            colorbar=dict(title=dict(text="density", font=dict(size=LABEL_PT)), thickness=12),
            hovertemplate=(
                f"{payload.covariate} %{{x:.4g}}<br>response %{{y:.4g}}"
                "<br>density %{z:.4g}<extra></extra>"
            ),
        )
    )
    positive = density[density > 0.0]
    if positive.size and len(payload.x) > 1:
        levels = np.unique(np.quantile(positive, np.linspace(0.5, 0.98, 6)))
        if levels.size >= 2:
            fig.add_trace(
                go.Contour(
                    x=np.asarray(payload.x, dtype=np.float64),
                    y=np.asarray(payload.y_grid, dtype=np.float64),
                    z=density.T,
                    contours=dict(
                        coloring="none",
                        start=float(levels[0]),
                        end=float(levels[-1]),
                        size=float((levels[-1] - levels[0]) / max(len(levels) - 1, 1)),
                    ),
                    line=dict(color=CHART["original"]["color"], width=1.0),
                    showscale=False,
                    hoverinfo="skip",
                    name="iso-density",
                )
            )
    quantiles = getattr(payload, "quantiles", None)
    levels_q = getattr(payload, "quantile_levels", None)
    if quantiles is not None and levels_q:
        widths = np.linspace(1.2, CHART["edited"]["width"], len(levels_q))
        for level, curve, width in zip(levels_q, np.asarray(quantiles), widths, strict=True):
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(payload.x, dtype=np.float64),
                    y=np.asarray(curve, dtype=np.float64),
                    mode="lines",
                    line=dict(color=CHART["edited"]["color"], width=float(width)),
                    name=f"q{level:g}",
                    hovertemplate=f"q{level:g} %{{y:.4g}}<extra></extra>",
                )
            )
    if payload.levels is not None:
        _categorical_ticks(fig, payload.x, payload.levels)
    _axis_titles(fig, x=payload.covariate, y="response")
    return _finalise(
        fig,
        title=f"Conditional density along {payload.covariate}",
        showlegend=quantiles is not None,
    )


def _histogram_bar(histogram: Any, *, name: str, showlegend: bool) -> go.Bar:
    edges = np.asarray(histogram.edges, dtype=np.float64)
    return go.Bar(
        x=_centers(edges),
        y=np.asarray(histogram.counts, dtype=np.float64),
        width=float(np.min(np.diff(edges))) * 0.96,
        marker=dict(_EXPOSURE_MARKER),
        name=name,
        showlegend=showlegend,
        hovertemplate=f"{name} %{{x:.4g}}<br>rows %{{y:,.0f}}<extra></extra>",
    )


def plotly_spread(payload: Any) -> go.Figure:
    """How sharp the fitted parameters are, and how wide the identical prices."""
    names = list(payload.parameters)
    titles = [f"{name} across the book" for name in names]
    titles.append(f"{payload.tail_p:g} quantile across the book")
    titles.append(f"spread among rows priced alike (by {payload.by})")
    rows, cols = _grid(len(titles))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    for index, name in enumerate(names):
        row, col = divmod(index, cols)
        fig.add_trace(
            _histogram_bar(payload.parameters[name], name=name, showlegend=index == 0),
            row=row + 1,
            col=col + 1,
        )
    row, col = divmod(len(names), cols)
    fig.add_trace(
        _histogram_bar(payload.tail_quantile, name="tail quantile", showlegend=False),
        row=row + 1,
        col=col + 1,
    )

    table = payload.identically_priced
    mean = table["mean"].to_numpy(dtype=np.float64)
    low = table["p_lo"].to_numpy(dtype=np.float64)
    high = table["p_hi"].to_numpy(dtype=np.float64)
    ratio = table["ratio"].to_numpy(dtype=np.float64)
    counts = table["n"].to_numpy(dtype=np.float64)
    row, col = divmod(len(names) + 1, cols)
    cell = dict(row=row + 1, col=col + 1)
    fig.add_trace(_band_trace(mean, low, high, name="within-bin spread", showlegend=False), **cell)
    for bound, label in ((high, "upper percentile"), (low, "lower percentile")):
        fig.add_trace(
            go.Scatter(
                x=mean,
                y=bound,
                mode="lines",
                line=_line(_OUTLINE),
                name=label,
                showlegend=False,
                customdata=np.column_stack([ratio, counts]),
                hovertemplate=(
                    f"{payload.by} %{{x:.4g}}<br>exceedance %{{y:.5f}}"
                    "<br>ratio %{customdata[0]:.3f}<br>n = %{customdata[1]:,.0f}<extra></extra>"
                ),
            ),
            **cell,
        )
    _style_titles(fig)
    return _finalise(
        fig,
        rows=rows,
        cols=cols,
        title=(
            f"Parameter spread — exceedance of {payload.threshold:,.4g} "
            f"over {payload.n_bins} price bins"
        ),
    )


def plotly_portfolio(payload: Any) -> go.Figure:
    """The simulated total for the book, and the totals of its segments."""
    segments = payload.by_segment
    titles = ["simulated total"]
    if segments is not None:
        titles.append(f"total by {payload.by}")
    fig = make_subplots(rows=len(titles), cols=1, subplot_titles=titles, vertical_spacing=0.14)

    if payload.total_draws is not None:
        fig.add_trace(
            go.Histogram(
                x=np.asarray(payload.total_draws, dtype=np.float64),
                nbinsx=_TREND_BINS * 2,
                marker=dict(_EXPOSURE_MARKER),
                name="draws",
                hovertemplate="total %{x:,.4g}<br>draws %{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=payload.total_mean,
            line_color=CHART["edited"]["color"],
            line_width=CHART["edited"]["width"],
            row=1,
            col=1,
        )
        for quantile in payload.quantiles:
            fig.add_vline(
                x=payload.total_quantiles[quantile],
                line_color=CHART["zero"]["color"],
                line_width=CHART["zero"]["width"],
                line_dash=_ZERO["dash"],
                row=1,
                col=1,
            )
    else:
        asked = list(payload.quantiles)
        fig.add_trace(
            go.Scatter(
                x=asked,
                y=[payload.total_quantiles[value] for value in asked],
                mode="lines+markers",
                line=_line(_FITTED),
                marker=_point_marker(),
                selected=_selection(),
                name="total quantiles",
                hovertemplate="quantile %{x:g}<br>total %{y:,.4g}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if segments is not None:
        labels = [str(name) for name in segments["segment"]]
        mean = segments["mean_total"].to_numpy(dtype=np.float64)
        columns = [f"q{value:g}" for value in payload.quantiles]
        low = segments[columns[0]].to_numpy(dtype=np.float64)
        high = segments[columns[-1]].to_numpy(dtype=np.float64)
        counts = segments["n"].to_numpy(dtype=np.float64)
        fig.add_trace(
            go.Bar(
                x=labels,
                y=mean,
                marker=dict(color=CHART["edited"]["color"]),
                name="mean total",
                customdata=np.column_stack([counts]),
                hovertemplate=(
                    "%{x}<br>mean total %{y:,.4g}<br>n = %{customdata[0]:,.0f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=mean,
                mode="markers",
                marker=_point_marker(),
                selected=_selection(),
                error_y=_whisker(high - mean, mean - low),
                name=f"{columns[0]} to {columns[-1]}",
                customdata=np.column_stack([low, high]),
                hovertemplate=(
                    "%{x}<br>%{customdata[0]:,.4g} to %{customdata[1]:,.4g}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    _style_titles(fig)
    uncertainty = "with" if payload.parameter_uncertainty else "without"
    return _finalise(
        fig,
        rows=len(titles),
        title=(
            f"Portfolio total — {payload.n_draws:,} draws {uncertainty} parameter "
            f"uncertainty, mean {payload.total_mean:,.4g}, sd {payload.total_sd:,.4g}"
        ),
    )


# --------------------------------------------------------------------------- #
# The six-panel diagnostics figure
# --------------------------------------------------------------------------- #


def _q_statistic_note(fig: go.Figure, payload: Any, *, axis: str) -> None:
    """Write a worm panel's Q-statistics into its top-left corner, in a white box.

    A worm curls up at its ends, so the bottom-left corner is exactly where the
    lowest points sit; the top-left is empty by construction and a filled box
    keeps the text off the envelope.  The four numbers are the standardised
    moments of Royston and Wright (2000), the same ones the matplotlib panel
    writes, red where the panel is flagged.
    """
    table = payload.q_statistics
    if table is None:
        return
    label = str(payload.panels[0].label)
    matching = table[table["group"].astype(str) == label]
    if not len(matching):
        return
    entry = matching.iloc[0]
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref=f"x{axis} domain",
        yref=f"y{axis} domain",
        xanchor="left",
        yanchor="top",
        text="  ".join(
            f"{short} {float(entry[column]):+.2f}"
            for short, column in (
                ("mean", "mean_z"),
                ("var", "variance_z"),
                ("skew", "skewness_z"),
                ("kurt", "kurtosis_z"),
            )
        ),
        showarrow=False,
        align="left",
        font=dict(
            size=LABEL_PT,
            color=(CHART["point_selected"]["face"] if bool(entry["flagged"]) else TOKENS["muted"]),
        ),
        bgcolor="#ffffff",
        bordercolor="#d0d7de",
        borderwidth=1,
        borderpad=3,
    )


def plotly_diagnostics_figure(
    qq: Any,
    worm: Any,
    pit: Any,
    residuals: Any,
    *,
    max_points: int = 50_000,
) -> go.Figure:
    """Draw the six distributional diagnostic panels on one figure.

    Q-Q with its envelope, the worm, the PIT histogram, the residual density
    against the standard normal, the residuals against the first parameter's
    linear predictor, and the residual standard deviation in bins of the
    second's.  The two scatter panels bin their rows into a sequential
    ``Histogram2d`` above ``max_points`` rows rather than drawing a marker each.
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Q-Q of quantile residuals",
            "worm",
            "PIT histogram",
            "residual density",
            "residuals against η₁",
            "residual sd against η₂",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.07,
    )
    _qq_panel(fig, qq, max_points=max_points, row=1, col=1)
    _worm_panel(fig, worm.panels[0], row=1, col=2, showlegend=False)
    _pit_panel(fig, pit, row=1, col=3, showlegend=False)

    values = np.asarray(residuals.quantile, dtype=np.float64)
    finite = values[np.isfinite(values)]
    low, high = (float(bound) for bound in np.percentile(finite, [0.5, 99.5]))
    clipped = finite[(finite >= low) & (finite <= high)]
    fig.add_trace(
        go.Histogram(
            x=clipped,
            nbinsx=_DENSITY_BINS,
            histnorm="probability density",
            marker=dict(_EXPOSURE_MARKER),
            name="residual density",
            showlegend=False,
            hovertemplate="residual %{x:.3f}<br>density %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    grid = np.linspace(low, high, 200)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi),
            mode="lines",
            line=_line(_REFERENCE),
            name="N(0, 1)",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    eta = np.asarray(residuals.eta, dtype=np.float64)
    location = eta[:, 0]
    scale = eta[:, 1] if eta.shape[1] > 1 else eta[:, 0]
    _scatter_or_binned(
        fig,
        location,
        values,
        name="residuals",
        max_points=max_points,
        hovertemplate="η₁ %{x:.4f}<br>residual %{y:.4f}<extra></extra>",
        row=2,
        col=2,
    )
    fig.add_trace(_flat_line(location, 0.0, name="zero", style=_ZERO), row=2, col=2)

    centers, spreads = _equal_count_sd(scale, values)
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=spreads,
            mode="lines+markers",
            line=_line(_FITTED),
            marker=_point_marker(),
            selected=_selection(),
            name="residual sd",
            showlegend=False,
            hovertemplate="η₂ %{x:.4f}<br>sd %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=3,
    )
    fig.add_trace(_flat_line(centers, 1.0, name="unit sd", style=_ZERO), row=2, col=3)

    _style_titles(fig)
    _q_statistic_note(fig, worm, axis="2")
    return _finalise(
        fig,
        rows=2,
        cols=3,
        title=(
            f"Distributional diagnostics — {residuals.n_rows:,} rows, "
            f"{residuals.weight_semantics} weights, {residuals.clipped_rows:,} clipped"
        ),
        showlegend=False,
    )
