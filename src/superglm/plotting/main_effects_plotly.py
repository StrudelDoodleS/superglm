"""Interactive Plotly explorer for main effects with response/link scale toggle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.inference import TermInference
from superglm.plotting.common import (
    _EXP_EDGE_LW,
    _LINE_WIDTH,
    _PLOTLY_EXP_EDGE,
    _PLOTLY_EXP_FILL,
    _PLOTLY_KNOT_COLOR,
    _PLOTLY_LINE_COLOR,
    _PLOTLY_PANEL,
    _PLOTLY_PW_FILL,
    _PLOTLY_SIM_FILL,
    _PLOTLY_TEXT,
    _PW_ALPHA,
    _PW_EDGE_ALPHA,
    _PW_EDGE_LW,
    _SIM_ALPHA,
    _SIM_EDGE_ALPHA,
    _SIM_EDGE_LW,
    _apply_plotly_theme,
    _exposure_kde,
    _hex_to_rgba,
)

_LINE_COLOR = _PLOTLY_LINE_COLOR
_PW_FILL = _PLOTLY_PW_FILL
_SIM_FILL = _PLOTLY_SIM_FILL
_EXP_FILL = _PLOTLY_EXP_FILL
_EXP_EDGE = _PLOTLY_EXP_EDGE
_KNOT_COLOR = _PLOTLY_KNOT_COLOR
_CI_LINE_POINTWISE = "rgba(24, 33, 43, 0.58)"
_CI_LINE_SIMULTANEOUS = "rgba(24, 33, 43, 0.78)"
_ERROR_BAR_COLOR = "rgba(24, 33, 43, 0.72)"
_SPLIT_TOP_DOMAIN = [0.31, 1.0]
_SPLIT_BOTTOM_DOMAIN = [0.0, 0.23]
_COLLAPSED_TOP_DOMAIN = [0.08, 1.0]
_COLLAPSED_BOTTOM_DOMAIN = [0.0, 0.01]


def _resolve_plotly_style(style: dict[str, Any] | None) -> dict[str, Any]:
    """Merge plotly explorer style overrides with defaults."""
    defaults = {
        "line_color": _LINE_COLOR,
        "bar_color": _LINE_COLOR,
        "density_fill_color": _EXP_FILL,
        "density_edge_color": _EXP_EDGE,
        "error_bar_color": _ERROR_BAR_COLOR,
        "text_color": _PLOTLY_TEXT,
        "text_outline_color": "rgba(255, 255, 255, 0.96)",
        "line_width": _LINE_WIDTH * 1.7,
        "curve_line_width": _LINE_WIDTH * 1.5,
        "bar_opacity": 0.7,
        "density_opacity": 0.92,
    }
    aliases = {
        "spline_color": "line_color",
        "curve_color": "line_color",
        "font_color": "text_color",
        "font_border_color": "text_outline_color",
        "weight_density_color": "density_fill_color",
        "exposure_density_color": "density_fill_color",
        "weight_density_edge_color": "density_edge_color",
        "exposure_density_edge_color": "density_edge_color",
    }

    merged = defaults.copy()
    if not style:
        return merged

    for key, value in style.items():
        canonical = aliases.get(key, key)
        if canonical not in defaults:
            raise ValueError(
                f"Unknown plotly_style key {key!r}. Expected one of {sorted(defaults)} "
                f"or aliases {sorted(aliases)}."
            )
        merged[canonical] = value
    return merged


@dataclass(frozen=True)
class _TraceEntry:
    """Trace visibility metadata for a single term."""

    term_idx: int
    default_visibility: bool | str
    is_basis: bool = False


@dataclass(frozen=True)
class _LinkVariant:
    """Link-scale data for one trace (used by the scale toggle).

    ``None`` values mean "same as response — no change on toggle".
    """

    y: Any = None
    base: Any = None
    text: Any = None
    hovertemplate: str | None = None
    error_y_array: Any = None
    error_y_arrayminus: Any = None


@dataclass(frozen=True)
class _XAxisConfig:
    """Per-term x-axis configuration.  Y-axis is controlled by the scale toggle."""

    top_x_title: str = ""
    top_x_type: str = "linear"
    top_tickvals: list[float] | None = None
    top_ticktext: list[str] | None = None
    bottom_x_title: str = ""
    bottom_y_title: str = "Diagnostics"
    bottom_x_type: str = "linear"
    bottom_tickvals: list[float] | None = None
    bottom_ticktext: list[str] | None = None
    overlay_density_top: bool = False
    top_secondary_y_title: str = "Density"


def plot_main_effects_plotly(
    model,
    terms: list[TermInference],
    *,
    X: pd.DataFrame | None = None,
    sample_weight: NDArray | None = None,
    interval: str | None = "pointwise",
    ci_style: str = "band",
    show_exposure: bool = True,
    show_knots: bool = False,
    show_bases: bool = False,
    scale: str = "response",
    categorical_display: str = "auto",
    title: str | None = None,
    subtitle: str | None = None,
    style: dict[str, Any] | None = None,
):
    """Build a Plotly main-effect explorer with a term dropdown and scale toggle.

    The figure includes two sets of interactive controls:

    * **Scale toggle** (top-left) — switches all visible traces between
      response scale (relativities) and link scale (η).
    * **Term dropdown** (below the scale toggle) — selects which term to
      display.  Switching terms resets to response scale.

    Parameters
    ----------
    model : SuperGLM
        Fitted model instance.
    terms : list of TermInference
        Per-term inference objects from ``model.term_inference()``.
    X : DataFrame, optional
        Training data for exposure/weight density overlays.
    sample_weight : array-like, optional
        Observation weights for density calculation.
    interval : {"pointwise", "simultaneous", "both", None}
        Confidence band style.
    ci_style : {"band", "lines"}
        ``"band"`` (default) draws filled bands with dashed edges.
        ``"lines"`` draws neutral charcoal CI lines only (no fill).
    show_exposure : bool
        Show exposure density in a lower panel.
    show_knots : bool
        Show interior knot markers on spline curves.
    show_bases : bool
        Show coefficient-weighted B-spline basis contributions (link scale).
    scale : {"response", "link"}
        Initial y-axis scale.  The toggle allows switching at any time.
    title, subtitle : str, optional
        Figure title and subtitle text.
    style : dict, optional
        Plotly trace-style overrides such as ``line_color``, ``bar_color``,
        ``density_fill_color``, ``density_edge_color``, ``error_bar_color``,
        ``text_color``, and ``text_outline_color``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive main-effect explorer figure.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for engine='plotly'. Install it with: pip install plotly"
        ) from None
    from superglm.features.ordered_categorical import OrderedCategorical

    if not terms:
        return go.Figure()

    unsupported_ordered_step = [
        ti.name
        for ti in terms
        if isinstance(model._specs.get(ti.name), OrderedCategorical)
        and model._specs[ti.name].basis == "step"
    ]
    if unsupported_ordered_step:
        joined = ", ".join(unsupported_ordered_step)
        raise NotImplementedError(
            "Plotly main-effects explorer does not yet support "
            f"OrderedCategorical(basis='step') for: {joined}. "
            "Use engine='matplotlib' for now."
        )
    valid_categorical_display = {"auto", "bars", "markers", "bars+markers"}
    if categorical_display not in valid_categorical_display:
        raise ValueError(
            f"categorical_display={categorical_display!r} is not valid, expected one of "
            f"{sorted(valid_categorical_display)}."
        )
    style_cfg = _resolve_plotly_style(style)

    weighted = sample_weight is not None
    if X is not None and sample_weight is None:
        sample_weight = np.ones(len(X), dtype=np.float64)
    elif sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)

    density_available = X is not None and sample_weight is not None
    density_visible = density_available and show_exposure
    needs_lower_panel = density_available

    fig = make_subplots(
        rows=2 if needs_lower_panel else 1,
        cols=1,
        row_heights=[0.75, 0.25] if needs_lower_panel else None,
        vertical_spacing=0.08 if needs_lower_panel else None,
        shared_xaxes=True if needs_lower_panel else False,
    )

    entries: list[_TraceEntry] = []
    link_variants: list[_LinkVariant] = []
    x_configs: list[_XAxisConfig] = []
    density_y_title = "Weight density" if weighted else "Obs. density"

    for term_idx, ti in enumerate(terms):
        x_configs.append(
            _add_term_traces(
                fig,
                model,
                ti,
                term_idx,
                entries,
                link_variants,
                X=X,
                sample_weight=sample_weight,
                interval=interval,
                ci_style=ci_style,
                density_visible=density_visible,
                density_y_title=density_y_title,
                show_knots=show_knots,
                show_bases=show_bases,
                needs_lower_panel=needs_lower_panel,
                style_cfg=style_cfg,
                categorical_display=categorical_display,
            )
        )

    # Reference line — 1.0 (response) or 0.0 (link), moved by toggle
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="rgba(100,100,100,0.6)",
        line_width=1.0,
        row=1,
        col=1,
    )

    # ── Initial visibility ────────────────────────────────────
    initial_term = 0
    for i, entry in enumerate(entries):
        if entry.is_basis and scale == "response":
            # Basis traces are link-scale diagnostics — fully hide on response scale
            # so they don't distort autorange or appear visually.
            fig.data[i].visible = False
        elif entry.term_idx == initial_term:
            fig.data[i].visible = entry.default_visibility
        else:
            fig.data[i].visible = False

    # ── Build scale toggle restyle / relayout ─────────────────
    n_traces = len(fig.data)
    response_ys: list[Any] = []
    link_ys: list[Any] = []
    response_hovers: list[Any] = []
    link_hovers: list[Any] = []
    response_bases: list[Any] = []
    link_bases: list[Any] = []
    response_texts: list[Any] = []
    link_texts: list[Any] = []
    resp_err_arr: list[Any] = []
    resp_err_minus: list[Any] = []
    link_err_arr: list[Any] = []
    link_err_minus: list[Any] = []
    has_errors = False

    for i, lv in enumerate(link_variants):
        trace = fig.data[i]
        r_y = _ensure_list(trace.y)

        if entries[i].is_basis:
            # Basis traces are link-scale diagnostics — null out response y-data
            # so they don't contribute to autorange on response scale.
            response_ys.append([None] * len(r_y))
            link_ys.append(r_y)  # real basis data shown on link scale
        else:
            response_ys.append(r_y)
            link_ys.append(_ensure_list(lv.y) if lv.y is not None else r_y)

        response_hovers.append(trace.hovertemplate)
        link_hovers.append(
            lv.hovertemplate if lv.hovertemplate is not None else trace.hovertemplate
        )
        response_bases.append(_ensure_list(getattr(trace, "base", None)))
        link_bases.append(
            _ensure_list(lv.base)
            if lv.base is not None
            else _ensure_list(getattr(trace, "base", None))
        )
        response_texts.append(_ensure_list(getattr(trace, "text", None)))
        link_texts.append(
            _ensure_list(lv.text)
            if lv.text is not None
            else _ensure_list(getattr(trace, "text", None))
        )

        r_ea = _extract_error_array(trace)
        r_em = _extract_error_arrayminus(trace)
        l_ea = _ensure_list(lv.error_y_array) if lv.error_y_array is not None else r_ea
        l_em = _ensure_list(lv.error_y_arrayminus) if lv.error_y_arrayminus is not None else r_em
        resp_err_arr.append(r_ea)
        resp_err_minus.append(r_em)
        link_err_arr.append(l_ea)
        link_err_minus.append(l_em)
        if r_ea is not None or l_ea is not None:
            has_errors = True

    # Build visible lists for the scale toggle.  Non-basis traces use their
    # current visibility (the dropdown controls that independently).  Basis
    # traces are hidden on response scale and restored on link scale.
    any_basis = any(e.is_basis for e in entries)
    if any_basis:
        response_vis: list[bool | str | None] = []
        link_vis: list[bool | str | None] = []
        for entry in entries:
            if entry.is_basis:
                response_vis.append(False)
                link_vis.append(entry.default_visibility)
            else:
                # None means "leave this trace's visibility unchanged" in
                # Plotly restyle — the dropdown controls non-basis visibility.
                response_vis.append(None)
                link_vis.append(None)

    response_restyle: dict[str, Any] = {
        "y": response_ys,
        "hovertemplate": response_hovers,
    }
    link_restyle: dict[str, Any] = {
        "y": link_ys,
        "hovertemplate": link_hovers,
    }
    if any(base is not None for base in response_bases + link_bases):
        response_restyle["base"] = response_bases
        link_restyle["base"] = link_bases
    if any(text is not None for text in response_texts + link_texts):
        response_restyle["text"] = response_texts
        link_restyle["text"] = link_texts
    if any_basis:
        response_restyle["visible"] = response_vis
        link_restyle["visible"] = link_vis
    if has_errors:
        response_restyle["error_y.array"] = resp_err_arr
        response_restyle["error_y.arrayminus"] = resp_err_minus
        link_restyle["error_y.array"] = link_err_arr
        link_restyle["error_y.arrayminus"] = link_err_minus

    response_relayout: dict[str, Any] = {
        "yaxis.title.text": "Relativity",
        "yaxis.autorange": True,
    }
    link_relayout: dict[str, Any] = {
        "yaxis.title.text": "η (link scale)",
        "yaxis.autorange": True,
    }
    # Move reference hline between scales
    if fig.layout.shapes:
        response_relayout["shapes[0].y0"] = 1.0
        response_relayout["shapes[0].y1"] = 1.0
        link_relayout["shapes[0].y0"] = 0.0
        link_relayout["shapes[0].y1"] = 0.0

    # ── Layout ────────────────────────────────────────────────
    base_title = title or "Model effects"
    if subtitle is not None:
        helper_subtitle = subtitle
    elif len(terms) > 1:
        helper_subtitle = "Use the dropdown to switch terms and the buttons to toggle scale."
    else:
        helper_subtitle = "Use the buttons above to toggle between response and link scale."

    fig.update_layout(
        title=dict(
            text=_compose_title(base_title, helper_subtitle),
            x=0.0,
            xref="paper",
            xanchor="left",
            yref="container",
            y=0.98,
        ),
    )
    top_margin = 180 if len(terms) > 1 else 150
    _apply_plotly_theme(
        fig,
        height=700 if needs_lower_panel else 520,
        hovermode="x unified",
        margin=dict(l=70, r=40, t=top_margin, b=100 if needs_lower_panel else 80),
        legend_y=-0.18 if needs_lower_panel else -0.08,
    )
    if needs_lower_panel:
        fig.update_layout(
            yaxis3=dict(
                overlaying="y",
                anchor="x",
                side="right",
                showgrid=False,
                zeroline=False,
                range=[0.0, 1.05],
                title_text=density_y_title,
                visible=False,
            )
        )

    _apply_axis_config(fig, x_configs[initial_term], needs_lower_panel)
    fig.update_yaxes(title_text="Relativity", row=1, col=1)
    if needs_lower_panel:
        fig.update_yaxes(title_text=x_configs[initial_term].bottom_y_title, row=2, col=1)
        fig.update_yaxes(range=[0.0, 1.05], row=2, col=1)
        fig.update_xaxes(showline=True, linewidth=1, linecolor="#ccc", mirror=True, row=2, col=1)
        fig.update_yaxes(showline=True, linewidth=1, linecolor="#ccc", mirror=True, row=2, col=1)

    # ── Build updatemenus ─────────────────────────────────────
    updatemenus: list[dict[str, Any]] = []

    # Scale toggle (top-left): Response | Link
    updatemenus.append(
        dict(
            type="buttons",
            direction="left",
            x=0.0,
            y=1.25,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Response",
                    method="update",
                    args=[response_restyle, response_relayout],
                ),
                dict(
                    label="Link",
                    method="update",
                    args=[link_restyle, link_relayout],
                ),
            ],
            active=0 if scale == "response" else 1,
            showactive=True,
            bgcolor=_PLOTLY_PANEL,
            bordercolor="rgba(24, 33, 43, 0.10)",
            borderwidth=1,
            font=dict(size=12),
            pad=dict(r=10),
        )
    )

    # Term dropdown (stacked below the scale toggle)
    # Each button resets to response scale to avoid stale link-scale state.
    if len(terms) > 1:
        buttons = []
        for term_idx, ti in enumerate(terms):
            visibility = [
                (
                    False
                    if entry.is_basis
                    else (entry.default_visibility if entry.term_idx == term_idx else False)
                )
                for entry in entries
            ]
            restyle = {
                "visible": visibility,
                "y": response_ys,
                "hovertemplate": response_hovers,
            }
            if "base" in response_restyle:
                restyle["base"] = response_bases
            if has_errors:
                restyle["error_y.array"] = resp_err_arr
                restyle["error_y.arrayminus"] = resp_err_minus
            relayout = _layout_update(
                x_configs[term_idx],
                needs_lower_panel=needs_lower_panel,
                title=_compose_title(base_title, helper_subtitle, term_name=ti.name),
            )
            relayout["yaxis.title.text"] = "Relativity"
            relayout["updatemenus[0].active"] = 0
            if fig.layout.shapes:
                relayout["shapes[0].y0"] = 1.0
                relayout["shapes[0].y1"] = 1.0
            buttons.append(
                dict(
                    label=ti.name,
                    method="update",
                    args=[restyle, relayout],
                )
            )
        updatemenus.append(
            dict(
                type="dropdown",
                x=0.0,
                y=1.09,
                xanchor="left",
                yanchor="top",
                direction="down",
                buttons=buttons,
                showactive=True,
                bgcolor=_PLOTLY_PANEL,
                bordercolor="rgba(24, 33, 43, 0.10)",
                borderwidth=1,
                font=dict(size=13),
            )
        )

    fig.update_layout(updatemenus=updatemenus)
    annotations = [
        dict(
            text="Scale",
            x=0.0,
            y=1.28,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11, color=_PLOTLY_TEXT),
        )
    ]
    if len(terms) > 1:
        annotations.append(
            dict(
                text="Term",
                x=0.0,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=11, color=_PLOTLY_TEXT),
            )
        )
    fig.update_layout(annotations=annotations)

    # ── Apply initial scale if link ───────────────────────────
    if scale == "link":
        for i in range(n_traces):
            lv = link_variants[i]
            if lv.y is not None:
                fig.data[i].y = _ensure_list(lv.y)
            if lv.base is not None:
                fig.data[i].base = _ensure_list(lv.base)
            if lv.text is not None:
                fig.data[i].text = _ensure_list(lv.text)
            if lv.hovertemplate is not None:
                fig.data[i].hovertemplate = lv.hovertemplate
            if lv.error_y_array is not None:
                fig.data[i].error_y.array = _ensure_list(lv.error_y_array)
                fig.data[i].error_y.arrayminus = _ensure_list(lv.error_y_arrayminus)
        if fig.layout.shapes:
            fig.layout.shapes[0].y0 = 0.0
            fig.layout.shapes[0].y1 = 0.0
        fig.update_yaxes(title_text="η (link scale)", row=1, col=1)

    return fig


# ── Per-term dispatcher ───────────────────────────────────────


def _add_term_traces(
    fig,
    model,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    X: pd.DataFrame | None,
    sample_weight: NDArray | None,
    interval: str | None,
    ci_style: str,
    density_visible: bool,
    density_y_title: str,
    show_knots: bool,
    show_bases: bool,
    needs_lower_panel: bool,
    style_cfg: dict[str, Any],
    categorical_display: str,
) -> _XAxisConfig:
    """Append all traces for one term (response Y) and collect link variants."""

    if ti.kind in ("spline", "polynomial"):
        _add_continuous_term_traces(
            fig,
            ti,
            term_idx,
            entries,
            link_variants,
            interval=interval,
            ci_style=ci_style,
            style_cfg=style_cfg,
        )
        if show_knots or show_bases:
            _add_spline_diagnostic_traces(
                fig,
                model,
                ti,
                term_idx,
                entries,
                link_variants,
                show_knots=show_knots,
                show_bases=show_bases,
            )
        if needs_lower_panel:
            _add_continuous_density_trace(
                fig,
                ti,
                term_idx,
                entries,
                link_variants,
                X=X,
                sample_weight=sample_weight,
                density_visible=density_visible,
                style_cfg=style_cfg,
            )
        return _XAxisConfig(
            top_x_title=ti.name,
            top_x_type="linear",
            bottom_x_title=ti.name,
            bottom_y_title=density_y_title,
            bottom_x_type="linear",
        )

    if ti.kind == "numeric":
        _add_numeric_term_trace(fig, ti, term_idx, entries, link_variants, style_cfg=style_cfg)
        if needs_lower_panel:
            _add_numeric_density_trace(
                fig,
                ti,
                term_idx,
                entries,
                link_variants,
                X=X,
                sample_weight=sample_weight,
                density_visible=density_visible,
                style_cfg=style_cfg,
            )
        return _XAxisConfig(
            top_x_title="Effect",
            top_x_type="category",
            bottom_x_title=ti.name,
            bottom_y_title=density_y_title,
            bottom_x_type="linear",
        )

    # categorical — density bars added BEFORE term traces for z-order (behind)
    _add_categorical_density_trace(
        fig,
        ti,
        term_idx,
        entries,
        link_variants,
        X=X,
        sample_weight=sample_weight,
        density_visible=density_visible,
        style_cfg=style_cfg,
        overlay_top=True,
    )
    _add_categorical_term_trace(
        fig,
        ti,
        term_idx,
        entries,
        link_variants,
        style_cfg=style_cfg,
        categorical_display=categorical_display,
    )
    # show_knots for OrderedCategorical spline terms
    if show_knots and ti.spline is not None:
        _add_spline_diagnostic_traces(
            fig,
            model,
            ti,
            term_idx,
            entries,
            link_variants,
            show_knots=show_knots,
            show_bases=False,  # basis contributions not shown for categoricals
        )
    if ti.smooth_curve is not None and ti.smooth_curve.level_x is not None:
        tickvals = np.asarray(ti.smooth_curve.level_x, dtype=np.float64).tolist()
        ticktext = [str(level) for level in ti.levels]
        return _XAxisConfig(
            top_x_title=ti.name,
            top_x_type="linear",
            top_tickvals=tickvals,
            top_ticktext=ticktext,
            bottom_x_title=ti.name,
            bottom_y_title=density_y_title,
            bottom_x_type="linear",
            bottom_tickvals=tickvals,
            bottom_ticktext=ticktext,
            overlay_density_top=True,
            top_secondary_y_title="Exposure",
        )
    return _XAxisConfig(
        top_x_title=ti.name,
        top_x_type="category",
        bottom_x_title=ti.name,
        bottom_y_title=density_y_title,
        bottom_x_type="category",
        overlay_density_top=True,
        top_secondary_y_title="Exposure",
    )


# ── Continuous (spline / polynomial) traces ───────────────────


def _add_continuous_term_traces(
    fig,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    interval,
    ci_style: str = "band",
    style_cfg: dict[str, Any],
):
    """Add fitted effect + CI bands.  Traces use response Y; link variants stored."""
    import plotly.graph_objects as go

    x = ti.x
    y_resp = np.asarray(ti.relativity)
    y_link = np.asarray(ti.log_relativity)

    # ── Precompute CI on both scales ──────────────────────────
    ci_lo_resp_pw = np.asarray(ti.ci_lower) if ti.ci_lower is not None else None
    ci_hi_resp_pw = np.asarray(ti.ci_upper) if ti.ci_upper is not None else None
    ci_lo_resp_sim = (
        np.asarray(ti.ci_lower_simultaneous) if ti.ci_lower_simultaneous is not None else None
    )
    ci_hi_resp_sim = (
        np.asarray(ti.ci_upper_simultaneous) if ti.ci_upper_simultaneous is not None else None
    )

    if ti.se_log_relativity is not None:
        from scipy.stats import norm

        z = norm.ppf(1.0 - ti.alpha / 2.0)
        se = np.asarray(ti.se_log_relativity)
        ci_lo_link_pw = y_link - z * se
        ci_hi_link_pw = y_link + z * se
        c_sim = ti.critical_value_simultaneous if ti.critical_value_simultaneous is not None else z
        ci_lo_link_sim = y_link - c_sim * se
        ci_hi_link_sim = y_link + c_sim * se
    else:
        ci_lo_link_pw = ci_hi_link_pw = None
        ci_lo_link_sim = ci_hi_link_sim = None

    # ── Simultaneous CI bands ─────────────────────────────────
    if interval in ("simultaneous", "both") and ci_lo_resp_sim is not None:
        _l_sim_lo = ci_lo_link_sim if ci_lo_link_sim is not None else ci_lo_resp_sim
        _l_sim_hi = ci_hi_link_sim if ci_hi_link_sim is not None else ci_hi_resp_sim
        ci_name_sim = f"{int((1 - ti.alpha) * 100)}% simultaneous band"

        _sim_w = 1.35 if ci_style == "lines" else _SIM_EDGE_LW
        sim_line = (
            dict(color=_CI_LINE_SIMULTANEOUS, width=_sim_w, dash="longdash")
            if ci_style == "lines"
            else dict(color=_hex_to_rgba(_SIM_FILL, _SIM_EDGE_ALPHA), width=_sim_w, dash="dash")
        )
        if ci_style == "band":
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([ci_hi_resp_sim, ci_lo_resp_sim[::-1]]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(_SIM_FILL, _SIM_ALPHA),
                    line=dict(width=0),
                    hoverinfo="skip",
                    name=ci_name_sim,
                    legendgroup=f"{ti.name}:sim",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
            link_variants.append(
                _LinkVariant(y=np.concatenate([_l_sim_hi, _l_sim_lo[::-1]]).tolist())
            )

        # Edge lines
        first_edge = True
        for r_bound, l_bound in [
            (ci_hi_resp_sim, _l_sim_hi),
            (ci_lo_resp_sim, _l_sim_lo),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=r_bound,
                    mode="lines",
                    line=sim_line,
                    hoverinfo="skip",
                    name=ci_name_sim if (ci_style == "lines" and first_edge) else None,
                    legendgroup=f"{ti.name}:sim",
                    showlegend=(ci_style == "lines" and first_edge),
                ),
                row=1,
                col=1,
            )
            entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
            link_variants.append(_LinkVariant(y=l_bound.tolist()))
            first_edge = False

    # ── Pointwise CI bands ────────────────────────────────────
    if interval in ("pointwise", "both") and ci_lo_resp_pw is not None:
        _l_pw_lo = ci_lo_link_pw if ci_lo_link_pw is not None else ci_lo_resp_pw
        _l_pw_hi = ci_hi_link_pw if ci_hi_link_pw is not None else ci_hi_resp_pw
        ci_name_pw = f"{int((1 - ti.alpha) * 100)}% pointwise CI"

        _pw_w = 1.2 if ci_style == "lines" else _PW_EDGE_LW
        pw_line = (
            dict(color=_CI_LINE_POINTWISE, width=_pw_w, dash="dash")
            if ci_style == "lines"
            else dict(color=_hex_to_rgba(_PW_FILL, _PW_EDGE_ALPHA), width=_pw_w, dash="dash")
        )
        if ci_style == "band":
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([ci_hi_resp_pw, ci_lo_resp_pw[::-1]]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(_PW_FILL, _PW_ALPHA),
                    line=dict(width=0),
                    hoverinfo="skip",
                    name=ci_name_pw,
                    legendgroup=f"{ti.name}:pw",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
            link_variants.append(
                _LinkVariant(y=np.concatenate([_l_pw_hi, _l_pw_lo[::-1]]).tolist())
            )

        # Edge lines
        first_edge = True
        for r_bound, l_bound in [
            (ci_hi_resp_pw, _l_pw_hi),
            (ci_lo_resp_pw, _l_pw_lo),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=r_bound,
                    mode="lines",
                    line=pw_line,
                    hoverinfo="skip",
                    name=ci_name_pw if (ci_style == "lines" and first_edge) else None,
                    legendgroup=f"{ti.name}:pw",
                    showlegend=(ci_style == "lines" and first_edge),
                ),
                row=1,
                col=1,
            )
            entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
            link_variants.append(_LinkVariant(y=l_bound.tolist()))
            first_edge = False

    # ── Main fitted line ──────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_resp,
            mode="lines",
            name="Relativity",
            line=dict(color=style_cfg["line_color"], width=style_cfg["line_width"]),
            legendgroup=f"{ti.name}:line",
            hovertemplate=f"{ti.name}: %{{x:.3f}}<br>Relativity: %{{y:.4f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
    link_variants.append(
        _LinkVariant(
            y=y_link.tolist(),
            hovertemplate=f"{ti.name}: %{{x:.3f}}<br>η: %{{y:.4f}}<extra></extra>",
        )
    )


# ── Numeric (single-slope) trace ─────────────────────────────


def _add_numeric_term_trace(
    fig,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    style_cfg: dict[str, Any],
):
    """Add a single-bar numeric slope summary with both-scale error bars."""
    import plotly.graph_objects as go

    resp_val = float(np.asarray(ti.relativity)[0])
    link_val = float(np.asarray(ti.log_relativity)[0])

    resp_error_y = None
    link_err_up = None
    link_err_down = None
    ci_lo_resp = None
    ci_hi_resp = None

    if ti.ci_lower is not None and ti.ci_upper is not None:
        ci_lo_resp = float(np.asarray(ti.ci_lower)[0])
        ci_hi_resp = float(np.asarray(ti.ci_upper)[0])
        resp_error_y = dict(
            type="data",
            symmetric=False,
            array=[ci_hi_resp - resp_val],
            arrayminus=[resp_val - ci_lo_resp],
            color=style_cfg["error_bar_color"],
        )
        if ti.se_log_relativity is not None:
            from scipy.stats import norm

            z = norm.ppf(1.0 - ti.alpha / 2.0)
            se = float(np.asarray(ti.se_log_relativity)[0])
            link_err_up = [z * se]
            link_err_down = [z * se]
        else:
            link_err_up = [np.log(ci_hi_resp) - link_val]
            link_err_down = [link_val - np.log(ci_lo_resp)]

    fig.add_trace(
        go.Bar(
            x=["per unit"],
            y=[resp_val],
            name="Relativity",
            marker_color=style_cfg["bar_color"],
            error_y=resp_error_y,
            legendgroup=f"{ti.name}:line",
            hovertemplate=_numeric_hovertemplate(
                ti.name,
                "Relativity",
                ci_low=ci_lo_resp if ti.ci_lower is not None and ti.ci_upper is not None else None,
                ci_high=ci_hi_resp if ti.ci_lower is not None and ti.ci_upper is not None else None,
                alpha=ti.alpha,
            ),
        ),
        row=1,
        col=1,
    )
    entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
    link_variants.append(
        _LinkVariant(
            y=[link_val],
            hovertemplate=_numeric_hovertemplate(
                ti.name,
                "η",
                ci_low=(link_val - link_err_down[0]) if link_err_down is not None else None,
                ci_high=(link_val + link_err_up[0]) if link_err_up is not None else None,
                alpha=ti.alpha,
            ),
            error_y_array=link_err_up,
            error_y_arrayminus=link_err_down,
        )
    )


# ── Categorical trace ────────────────────────────────────────


def _add_categorical_term_trace(
    fig,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    style_cfg: dict[str, Any],
    categorical_display: str,
):
    """Add categorical or ordered-categorical effect traces.

    Always renders line+markers (no bars for relativities).
    Ordered categoricals get markers + smooth curve overlay.
    Unordered categoricals get lines+markers.
    """
    import plotly.graph_objects as go

    resp_y = np.asarray(ti.relativity)
    link_y = np.asarray(ti.log_relativity)
    curve = ti.smooth_curve
    is_ordered = curve is not None

    # _resolve_categorical_display kept callable for backward compat
    _resolve_categorical_display(categorical_display, len(ti.levels))

    # Error bars — compute for both scales
    resp_error_y = None
    link_err_up = None
    link_err_down = None
    ci_lo_resp = None
    ci_hi_resp = None
    ci_lo_link = None
    ci_hi_link = None

    if ti.ci_lower is not None and ti.ci_upper is not None:
        ci_lo_resp = np.asarray(ti.ci_lower)
        ci_hi_resp = np.asarray(ti.ci_upper)
        resp_error_y = dict(
            type="data",
            symmetric=False,
            array=ci_hi_resp - resp_y,
            arrayminus=resp_y - ci_lo_resp,
            color=style_cfg["error_bar_color"],
        )
        if ti.se_log_relativity is not None:
            from scipy.stats import norm

            z = norm.ppf(1.0 - ti.alpha / 2.0)
            se = np.asarray(ti.se_log_relativity)
            link_err_up = (z * se).tolist()
            link_err_down = (z * se).tolist()
            ci_lo_link = link_y - z * se
            ci_hi_link = link_y + z * se
        else:
            ci_lo_link = np.log(ci_lo_resp)
            ci_hi_link = np.log(ci_hi_resp)
            link_err_up = (ci_hi_link - link_y).tolist()
            link_err_down = (link_y - ci_lo_link).tolist()

    if is_ordered:
        # OrderedCategorical: markers with error bars + smooth curve overlay
        level_x = (
            np.asarray(curve.level_x, dtype=np.float64)
            if curve is not None and curve.level_x is not None
            else np.arange(len(ti.levels), dtype=np.float64)
        )
        level_labels = [str(level) for level in ti.levels]
        if ci_lo_resp is not None and ci_hi_resp is not None:
            customdata = np.column_stack([resp_y, ci_lo_resp, ci_hi_resp, ci_lo_link, ci_hi_link])
            resp_hover_marker = (
                f"{ti.name}: %{{hovertext}}<br>Relativity: %{{y:.4f}}"
                f"<br>{int((1 - ti.alpha) * 100)}% CI: [%{{customdata[1]:.4f}}, %{{customdata[2]:.4f}}]"
                "<extra></extra>"
            )
            link_hover = (
                f"{ti.name}: %{{hovertext}}<br>η: %{{y:.4f}}"
                f"<br>{int((1 - ti.alpha) * 100)}% CI: [%{{customdata[3]:.4f}}, %{{customdata[4]:.4f}}]"
                "<extra></extra>"
            )
        else:
            customdata = np.column_stack([resp_y])
            resp_hover_marker = (
                f"{ti.name}: %{{hovertext}}<br>Relativity: %{{y:.4f}}<extra></extra>"
            )
            link_hover = f"{ti.name}: %{{hovertext}}<br>η: %{{y:.4f}}<extra></extra>"

        # Markers trace with error bars
        fig.add_trace(
            go.Scatter(
                x=level_x.tolist(),
                y=resp_y,
                mode="markers",
                name="Relativity",
                marker=dict(
                    size=9,
                    color=style_cfg["line_color"],
                    line=dict(color=style_cfg["text_outline_color"], width=0.8),
                ),
                error_y=resp_error_y,
                customdata=customdata,
                hovertext=level_labels,
                legendgroup=f"{ti.name}:markers",
                hovertemplate=resp_hover_marker,
            ),
            row=1,
            col=1,
        )
        entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
        link_variants.append(
            _LinkVariant(
                y=link_y.tolist(),
                hovertemplate=link_hover,
                error_y_array=link_err_up,
                error_y_arrayminus=link_err_down,
            )
        )

        # Smooth curve overlay
        resp_curve_y = np.asarray(curve.relativity)
        link_curve_y = np.log(resp_curve_y)
        fig.add_trace(
            go.Scatter(
                x=curve.x,
                y=resp_curve_y,
                mode="lines",
                name="Smooth curve",
                line=dict(color=style_cfg["line_color"], width=style_cfg["curve_line_width"]),
                legendgroup=f"{ti.name}:curve",
                hovertemplate=f"{ti.name}: %{{x:.3f}}<br>Relativity: %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
        link_variants.append(_LinkVariant(y=link_curve_y.tolist()))
    else:
        # Unordered Categorical: lines+markers with error bars
        if ci_lo_resp is not None and ci_hi_resp is not None:
            customdata = np.column_stack([ci_lo_resp, ci_hi_resp, ci_lo_link, ci_hi_link])
            resp_hover = (
                f"{ti.name}: %{{x}}<br>Relativity: %{{y:.4f}}"
                f"<br>{int((1 - ti.alpha) * 100)}% CI: [%{{customdata[0]:.4f}}, %{{customdata[1]:.4f}}]"
                "<extra></extra>"
            )
        else:
            customdata = None
            resp_hover = f"{ti.name}: %{{x}}<br>Relativity: %{{y:.4f}}<extra></extra>"
        link_hover = (
            f"{ti.name}: %{{x}}<br>η: %{{y:.4f}}"
            + (
                f"<br>{int((1 - ti.alpha) * 100)}% CI: [%{{customdata[2]:.4f}}, %{{customdata[3]:.4f}}]"
                if ci_lo_link is not None and ci_hi_link is not None
                else ""
            )
            + "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=list(ti.levels),
                y=resp_y,
                mode="lines+markers",
                name="Relativity",
                marker=dict(
                    size=9,
                    color=style_cfg["line_color"],
                    line=dict(color=style_cfg["text_outline_color"], width=0.8),
                ),
                line=dict(color=style_cfg["line_color"], width=style_cfg["curve_line_width"]),
                error_y=resp_error_y,
                legendgroup=f"{ti.name}:markers",
                customdata=customdata,
                hovertemplate=resp_hover,
            ),
            row=1,
            col=1,
        )
        entries.append(_TraceEntry(term_idx=term_idx, default_visibility=True))
        link_variants.append(
            _LinkVariant(
                y=link_y.tolist(),
                hovertemplate=link_hover,
                error_y_array=link_err_up,
                error_y_arrayminus=link_err_down,
            )
        )


# ── Density traces (scale-invariant) ─────────────────────────


def _add_continuous_density_trace(
    fig,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    X: pd.DataFrame | None,
    sample_weight: NDArray | None,
    density_visible: bool,
    style_cfg: dict[str, Any],
):
    """Add density strip for continuous terms."""
    import plotly.graph_objects as go

    if X is None or sample_weight is None or ti.name not in X.columns:
        return

    x_vals = X[ti.name].to_numpy(dtype=np.float64)
    density = _exposure_kde(x_vals, sample_weight, np.asarray(ti.x))
    fig.add_trace(
        go.Scatter(
            x=np.asarray(ti.x),
            y=density,
            mode="lines",
            fill="tozeroy",
            name="Exposure density",
            legendgroup=f"{ti.name}:density",
            line=dict(color=style_cfg["density_edge_color"], width=_EXP_EDGE_LW),
            fillcolor=_hex_to_rgba(style_cfg["density_fill_color"], 0.72),
            hovertemplate=f"{ti.name}: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    entries.append(
        _TraceEntry(term_idx=term_idx, default_visibility=True if density_visible else "legendonly")
    )
    link_variants.append(_LinkVariant())  # unchanged across scales


def _add_numeric_density_trace(
    fig,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    X: pd.DataFrame | None,
    sample_weight: NDArray | None,
    density_visible: bool,
    style_cfg: dict[str, Any],
):
    """Add density strip for numeric terms."""
    import plotly.graph_objects as go

    if X is None or sample_weight is None or ti.name not in X.columns:
        return

    x_vals = X[ti.name].to_numpy(dtype=np.float64)
    x_grid = np.linspace(float(x_vals.min()), float(x_vals.max()), 200)
    density = _exposure_kde(x_vals, sample_weight, x_grid)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=density,
            mode="lines",
            fill="tozeroy",
            name="Exposure density",
            legendgroup=f"{ti.name}:density",
            line=dict(color=style_cfg["density_edge_color"], width=_EXP_EDGE_LW),
            fillcolor=_hex_to_rgba(style_cfg["density_fill_color"], 0.72),
            hovertemplate=f"{ti.name}: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    entries.append(
        _TraceEntry(term_idx=term_idx, default_visibility=True if density_visible else "legendonly")
    )
    link_variants.append(_LinkVariant())


def _add_categorical_density_trace(
    fig,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    X: pd.DataFrame | None,
    sample_weight: NDArray | None,
    density_visible: bool,
    style_cfg: dict[str, Any],
    overlay_top: bool = False,
):
    """Add absolute exposure bars for categorical terms.

    Shows raw sample_weight sums per level (no normalization).
    When overlay_top=True, bars render on the right y-axis (y3)
    in the top panel with low opacity.
    """
    import plotly.graph_objects as go

    if X is None or sample_weight is None or ti.name not in X.columns:
        return

    levels = list(ti.levels)
    exp = (
        pd.DataFrame({"level": X[ti.name].astype(str), "sample_weight": sample_weight})
        .groupby("level", sort=False)["sample_weight"]
        .sum()
    )
    weights = np.array([float(exp.get(level, 0.0)) for level in levels], dtype=np.float64)

    bar_opacity = 0.3

    if ti.smooth_curve is not None and ti.smooth_curve.level_x is not None:
        level_x = np.asarray(ti.smooth_curve.level_x, dtype=np.float64)
        fig.add_trace(
            go.Bar(
                x=level_x.tolist(),
                y=weights,
                width=_ordered_bar_width(level_x),
                customdata=np.array(levels, dtype=object),
                name="Exposure",
                legendgroup=f"{ti.name}:density",
                marker_color=style_cfg["density_fill_color"],
                marker_line_color=style_cfg["density_edge_color"],
                marker_line_width=_EXP_EDGE_LW,
                opacity=bar_opacity,
                hovertemplate=(
                    f"{ti.name}: %{{customdata}}<br>Exposure: %{{y:,.0f}}<extra></extra>"
                ),
            ),
            row=1 if overlay_top else 2,
            col=1,
        )
        if overlay_top:
            fig.data[-1].yaxis = "y3"
        entries.append(
            _TraceEntry(
                term_idx=term_idx, default_visibility=True if density_visible else "legendonly"
            )
        )
        link_variants.append(_LinkVariant())
        return

    fig.add_trace(
        go.Bar(
            x=levels,
            y=weights,
            name="Exposure",
            legendgroup=f"{ti.name}:density",
            marker_color=style_cfg["density_fill_color"],
            marker_line_color=style_cfg["density_edge_color"],
            marker_line_width=_EXP_EDGE_LW,
            opacity=bar_opacity,
            hovertemplate=f"{ti.name}: %{{x}}<br>Exposure: %{{y:,.0f}}<extra></extra>",
        ),
        row=1 if overlay_top else 2,
        col=1,
    )
    if overlay_top:
        fig.data[-1].yaxis = "y3"
    entries.append(
        _TraceEntry(term_idx=term_idx, default_visibility=True if density_visible else "legendonly")
    )
    link_variants.append(_LinkVariant())


# ── Spline diagnostic overlays (knots + basis) ───────────────


def _add_spline_diagnostic_traces(
    fig,
    model,
    ti: TermInference,
    term_idx: int,
    entries: list[_TraceEntry],
    link_variants: list[_LinkVariant],
    *,
    show_knots: bool,
    show_bases: bool,
):
    """Add knot markers and coefficient-weighted basis contributions.

    Knots sit on whichever curve the scale toggle is showing.
    Basis contributions show β_j B_j(x) on the link scale and are
    visible on both response and link views.

    For OrderedCategorical (ti.x is None), uses smooth_curve data for the
    interpolation grid and skips basis contributions.
    """
    import plotly.graph_objects as go

    spec = model._specs.get(ti.name)
    is_categorical_spline = ti.kind == "categorical" and ti.spline is not None
    if not is_categorical_spline and (spec is None or not hasattr(spec, "_basis_matrix")):
        return

    # Use smooth_curve data for categoricals (ti.x is None)
    if ti.x is None and ti.smooth_curve is not None:
        x_grid = np.asarray(ti.smooth_curve.x)
        resp_y_curve = np.asarray(ti.smooth_curve.relativity)
        link_y_curve = np.log(resp_y_curve)
    elif ti.x is not None:
        x_grid = np.asarray(ti.x)
        resp_y_curve = np.asarray(ti.relativity)
        link_y_curve = np.asarray(ti.log_relativity)
    else:
        return

    # ── Knots ─────────────────────────────────────────────────
    if show_knots and ti.spline is not None and ti.spline.interior_knots.size > 0:
        knots = ti.spline.interior_knots
        knot_resp_y = np.interp(knots, x_grid, resp_y_curve)
        knot_link_y = np.interp(knots, x_grid, link_y_curve)
        fig.add_trace(
            go.Scatter(
                x=knots,
                y=knot_resp_y,
                mode="markers",
                name="Interior knots",
                legendgroup=f"{ti.name}:knots",
                marker=dict(
                    symbol="diamond",
                    size=7,
                    color="white",
                    line=dict(color=_KNOT_COLOR, width=1.5),
                ),
                hovertemplate=(
                    f"{ti.name}: %{{x:.2f}}<br>Relativity: %{{y:.4f}}<extra>knot</extra>"
                ),
            ),
            row=1,
            col=1,
        )
        entries.append(
            _TraceEntry(term_idx=term_idx, default_visibility=True if show_knots else "legendonly")
        )
        link_variants.append(
            _LinkVariant(
                y=knot_link_y.tolist(),
                hovertemplate=f"{ti.name}: %{{x:.2f}}<br>η: %{{y:.4f}}<extra>knot</extra>",
            )
        )

    # ── Basis contributions (link-scale only) ─────────────────
    if not show_bases or ti.kind != "spline":
        return

    basis = spec._basis_matrix(x_grid).toarray()
    if basis.shape[1] == 0:
        return

    from superglm.model.state_ops import feature_groups

    groups = feature_groups(model, ti.name)
    beta_combined = np.concatenate([model.result.beta[g.sl] for g in groups])
    R_inv = getattr(spec, "_R_inv", None)
    beta_orig = R_inv @ beta_combined if R_inv is not None else beta_combined

    if len(beta_orig) != basis.shape[1]:
        return

    colors = _basis_colors(basis.shape[1])

    for j in range(basis.shape[1]):
        contribution = basis[:, j] * beta_orig[j]
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=contribution,
                mode="lines",
                name="Basis contributions" if j == 0 else f"Basis {j + 1}",
                legendgroup=f"{ti.name}:basis",
                showlegend=(j == 0),
                line=dict(color=colors[j], width=1.0, dash="dot"),
                opacity=0.55,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        entries.append(
            _TraceEntry(
                term_idx=term_idx,
                default_visibility=True if show_bases else "legendonly",
                is_basis=True,
            )
        )
        link_variants.append(_LinkVariant())


def _supports_spline_diagnostics(model, ti: TermInference) -> bool:
    """Whether a term can show knots/basis diagnostics."""
    # Categoricals with spline metadata (OrderedCategorical spline mode) support knots
    if ti.spline is not None:
        return True
    spec = model._specs.get(ti.name)
    return ti.kind == "spline" and spec is not None and hasattr(spec, "_basis_matrix")


# ── Axis / layout helpers ─────────────────────────────────────


def _apply_axis_config(fig, cfg: _XAxisConfig, needs_lower_panel: bool) -> None:
    """Apply x-axis titles/types for the active term (y-axis set by toggle)."""
    fig.update_xaxes(
        title_text=cfg.top_x_title,
        type=cfg.top_x_type,
        tickmode="array" if cfg.top_tickvals is not None else "auto",
        tickvals=cfg.top_tickvals,
        ticktext=cfg.top_ticktext,
        row=1,
        col=1,
    )
    if needs_lower_panel:
        fig.update_xaxes(
            title_text=cfg.bottom_x_title,
            type=cfg.bottom_x_type,
            tickmode="array" if cfg.bottom_tickvals is not None else "auto",
            tickvals=cfg.bottom_tickvals,
            ticktext=cfg.bottom_ticktext,
            row=2,
            col=1,
        )
        if cfg.overlay_density_top:
            fig.update_layout(
                xaxis=dict(showticklabels=True),
                xaxis2=dict(showticklabels=False, title_text="", matches=None),
                yaxis=dict(domain=_COLLAPSED_TOP_DOMAIN),
                yaxis2=dict(domain=_COLLAPSED_BOTTOM_DOMAIN, showticklabels=False, title_text=""),
                yaxis3=dict(
                    visible=True,
                    title_text=cfg.top_secondary_y_title,
                    autorange=True,
                ),
            )
            fig.layout.xaxis.matches = None
        else:
            fig.update_layout(
                xaxis=dict(showticklabels=False),
                xaxis2=dict(showticklabels=True, matches=None),
                yaxis=dict(domain=_SPLIT_TOP_DOMAIN),
                yaxis2=dict(
                    domain=_SPLIT_BOTTOM_DOMAIN,
                    showticklabels=True,
                    title_text=cfg.bottom_y_title,
                    range=[0.0, 1.05],
                ),
                yaxis3=dict(visible=False),
            )
            fig.layout.xaxis.matches = "x2"


def _layout_update(cfg: _XAxisConfig, *, needs_lower_panel: bool, title: str) -> dict[str, Any]:
    """Plotly updatemenu relayout payload for one active term.

    Does NOT set yaxis.title.text — that is controlled by the scale toggle.
    """
    layout: dict[str, Any] = {
        "title.text": title,
        "xaxis.title.text": cfg.top_x_title,
        "xaxis.type": cfg.top_x_type,
        "xaxis.tickmode": "array" if cfg.top_tickvals is not None else "auto",
        "xaxis.tickvals": cfg.top_tickvals,
        "xaxis.ticktext": cfg.top_ticktext,
        "xaxis.autorange": True,
        "yaxis.autorange": True,
    }
    if needs_lower_panel:
        if cfg.overlay_density_top:
            layout.update(
                {
                    "xaxis.showticklabels": True,
                    "xaxis.matches": None,
                    "xaxis2.showticklabels": False,
                    "xaxis2.matches": None,
                    "xaxis2.title.text": "",
                    "yaxis.domain": _COLLAPSED_TOP_DOMAIN,
                    "yaxis2.domain": _COLLAPSED_BOTTOM_DOMAIN,
                    "yaxis2.showticklabels": False,
                    "yaxis2.title.text": "",
                    "yaxis3.visible": True,
                    "yaxis3.title.text": cfg.top_secondary_y_title,
                    "yaxis3.autorange": True,
                }
            )
        else:
            layout.update(
                {
                    "xaxis.showticklabels": False,
                    "xaxis.matches": "x2",
                    "xaxis2.title.text": cfg.bottom_x_title,
                    "xaxis2.type": cfg.bottom_x_type,
                    "xaxis2.tickmode": "array" if cfg.bottom_tickvals is not None else "auto",
                    "xaxis2.tickvals": cfg.bottom_tickvals,
                    "xaxis2.ticktext": cfg.bottom_ticktext,
                    "xaxis2.showticklabels": True,
                    "xaxis2.matches": None,
                    "yaxis.domain": _SPLIT_TOP_DOMAIN,
                    "yaxis2.domain": _SPLIT_BOTTOM_DOMAIN,
                    "yaxis2.title.text": cfg.bottom_y_title,
                    "yaxis2.showticklabels": True,
                    "xaxis2.autorange": True,
                    "yaxis2.autorange": True,
                    "yaxis2.range": [0.0, 1.05],
                    "yaxis3.visible": False,
                }
            )
    return layout


def _compose_title(base_title: str, subtitle: str, *, term_name: str | None = None) -> str:
    """Compose a compact Plotly title with optional active-term note."""
    if term_name is not None:
        base_title = f"{base_title} - {term_name}"
    return f"{base_title}<br><sup>{subtitle}</sup>"


# ── Utility helpers ───────────────────────────────────────────


def _basis_colors(n: int) -> list[str]:
    """Palette for basis overlays."""
    base = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#EECA3B",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]
    return [base[i % len(base)] for i in range(n)]


def _ordered_bar_width(x: NDArray) -> float:
    """Reasonable bar width for ordered-category numeric positions."""
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.6
    diffs = np.diff(np.sort(x))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.6
    return float(np.min(diffs) * 0.72)


def _resolve_categorical_display(mode: str, n_levels: int) -> str:
    """Resolve categorical display mode, applying the auto threshold."""
    if mode == "auto":
        return "markers" if n_levels > 30 else "bars+markers"
    return mode


def _numeric_hovertemplate(
    name: str,
    y_label: str,
    *,
    ci_low: float | None,
    ci_high: float | None,
    alpha: float,
) -> str:
    """Hovertemplate for one-point numeric summaries with optional CI text."""
    template = f"{name}: per unit<br>{y_label}: %{{y:.4f}}"
    if ci_low is not None and ci_high is not None:
        template += f"<br>{int((1 - alpha) * 100)}% CI: [{ci_low:.4f}, {ci_high:.4f}]"
    return template + "<extra></extra>"


def _ensure_list(val: Any) -> Any:
    """Convert arrays/tuples to plain lists for JSON in restyle args."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, tuple):
        return list(val)
    return val


def _extract_error_array(trace) -> list | None:
    """Get error_y.array from a Plotly trace, or None."""
    try:
        arr = trace.error_y.array
        if arr is not None:
            return _ensure_list(arr)
    except AttributeError:
        pass
    return None


def _extract_error_arrayminus(trace) -> list | None:
    """Get error_y.arrayminus from a Plotly trace, or None."""
    try:
        arr = trace.error_y.arrayminus
        if arr is not None:
            return _ensure_list(arr)
    except AttributeError:
        pass
    return None
