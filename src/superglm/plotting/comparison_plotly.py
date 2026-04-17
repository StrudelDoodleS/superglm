"""Plotly renderer for labeled fitted-model term comparisons."""

from __future__ import annotations

from typing import Any

from superglm.plotting.common import _EXP_EDGE_LW, _PLOTLY_PANEL, _apply_plotly_theme, _hex_to_rgba
from superglm.plotting.main_effects_plotly import _resolve_plotly_style


def plot_term_comparison_plotly(
    payload: dict[str, Any],
    *,
    title: str | None = None,
    subtitle: str | None = None,
    style: dict[str, Any] | None = None,
):
    """Render term-comparison payloads as a Plotly explorer."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for term comparison plots. Install it with: pip install plotly"
        ) from None

    terms = payload["terms"]
    style_cfg = _resolve_plotly_style(style)
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    entries: list[dict[str, Any]] = []
    response_ys: list[Any] = []
    link_ys: list[Any] = []

    colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#be185d",
        "#4f46e5",
    ]

    for term_idx, term in enumerate(terms):
        family = term["family"]
        support = term["support"]

        if family == "continuous":
            x = term["domain"]["x"]
            for series_idx, (label, series) in enumerate(term["series"].items()):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=series["response"],
                        mode="lines",
                        name=label,
                        line=dict(color=colors[series_idx % len(colors)], width=2.0),
                        hovertemplate=f"{label}<br>{term['name']}: %{{x:.3f}}<br>Relativity: %{{y:.4f}}<extra></extra>",
                        visible=(term_idx == 0),
                    ),
                    row=1,
                    col=1,
                )
                entries.append({"term_idx": term_idx, "trace_type": "series"})
                response_ys.append(series["response"])
                link_ys.append(series["link"])

            if support.get("mode") == "by_label":
                for series_idx, (label, support_series) in enumerate(support["series"].items()):
                    color = colors[series_idx % len(colors)]
                    fig.add_trace(
                        go.Scatter(
                            x=support_series["x"],
                            y=support_series["density"],
                            mode="lines",
                            name="Exposure density",
                            legendgroup=f"{term['name']}:{label}:density",
                            showlegend=False,
                            line=dict(color=color, width=_EXP_EDGE_LW),
                            hovertemplate=(
                                f"{label}<br>{term['name']}: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>"
                            ),
                            visible=(term_idx == 0),
                        ),
                        row=2,
                        col=1,
                    )
                    entries.append({"term_idx": term_idx, "trace_type": "support"})
                    response_ys.append(support_series["density"])
                    link_ys.append(support_series["density"])
            else:
                fig.add_trace(
                    go.Scatter(
                        x=support["x"],
                        y=support["density"],
                        mode="lines",
                        fill="tozeroy",
                        name="Exposure density",
                        legendgroup=f"{term['name']}:density",
                        line=dict(color=style_cfg["density_edge_color"], width=_EXP_EDGE_LW),
                        fillcolor=_hex_to_rgba(style_cfg["density_fill_color"], 0.72),
                        hovertemplate=f"{term['name']}: %{{x:.3f}}<br>Density: %{{y:.3f}}<extra></extra>",
                        visible=(term_idx == 0),
                    ),
                    row=2,
                    col=1,
                )
                entries.append({"term_idx": term_idx, "trace_type": "support"})
                response_ys.append(support["density"])
                link_ys.append(support["density"])
        else:
            levels = term["domain"]["levels"]
            for series_idx, (label, series) in enumerate(term["series"].items()):
                fig.add_trace(
                    go.Scatter(
                        x=levels,
                        y=series["response"],
                        mode="markers+lines",
                        name=label,
                        marker=dict(color=colors[series_idx % len(colors)], size=8),
                        line=dict(color=colors[series_idx % len(colors)], width=1.5),
                        hovertemplate=f"{label}<br>{term['name']}: %{{x}}<br>Relativity: %{{y:.4f}}<extra></extra>",
                        visible=(term_idx == 0),
                    ),
                    row=1,
                    col=1,
                )
                entries.append({"term_idx": term_idx, "trace_type": "series"})
                response_ys.append(series["response"])
                link_ys.append(series["link"])

            if support.get("mode") == "by_label":
                for series_idx, (label, support_series) in enumerate(support["series"].items()):
                    color = colors[series_idx % len(colors)]
                    fig.add_trace(
                        go.Bar(
                            x=support_series["levels"],
                            y=support_series["density"],
                            name=label,
                            legendgroup=f"{term['name']}:{label}:density",
                            showlegend=False,
                            marker_color=color,
                            marker_line_color=color,
                            marker_line_width=_EXP_EDGE_LW,
                            opacity=0.35,
                            hovertemplate=(
                                f"{label}<br>{term['name']}: %{{x}}<br>Exposure: %{{y:,.0f}}<extra></extra>"
                            ),
                            visible=(term_idx == 0),
                        ),
                        row=2,
                        col=1,
                    )
                    entries.append({"term_idx": term_idx, "trace_type": "support"})
                    response_ys.append(support_series["density"])
                    link_ys.append(support_series["density"])
            else:
                fig.add_trace(
                    go.Bar(
                        x=support["levels"],
                        y=support["density"],
                        name="Exposure",
                        legendgroup=f"{term['name']}:density",
                        marker_color=style_cfg["density_fill_color"],
                        marker_line_color=style_cfg["density_edge_color"],
                        marker_line_width=_EXP_EDGE_LW,
                        opacity=0.3,
                        hovertemplate=f"{term['name']}: %{{x}}<br>Exposure: %{{y:,.0f}}<extra></extra>",
                        visible=(term_idx == 0),
                    ),
                    row=2,
                    col=1,
                )
                entries.append({"term_idx": term_idx, "trace_type": "support"})
                response_ys.append(support["density"])
                link_ys.append(support["density"])

    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="rgba(100,100,100,0.6)",
        line_width=1.0,
        row=1,
        col=1,
    )

    _apply_plotly_theme(
        fig,
        height=700,
        hovermode="x unified",
        margin=dict(l=70, r=40, t=150, b=100),
        legend_y=-0.15,
    )
    fig.update_yaxes(title_text="Relativity", row=1, col=1)
    fig.update_yaxes(title_text="Support", row=2, col=1)

    base_title = title or "Term comparison"
    helper_subtitle = (
        subtitle or "Use the dropdown to switch terms and the buttons to toggle scale."
    )
    fig.update_layout(
        title=dict(
            text=f"{base_title}<br><sup>{helper_subtitle}</sup>",
            x=0.0,
            xref="paper",
            xanchor="left",
            yref="container",
            y=0.98,
        )
    )
    fig.update_layout(barmode="group")

    response_restyle = {"y": response_ys}
    link_restyle = {"y": link_ys}
    response_relayout = {
        "yaxis.title.text": "Relativity",
        "shapes[0].y0": 1.0,
        "shapes[0].y1": 1.0,
    }
    link_relayout = {
        "yaxis.title.text": "η (link scale)",
        "yaxis.autorange": True,
        "shapes[0].y0": 0.0,
        "shapes[0].y1": 0.0,
    }

    dropdown_buttons = []
    for term_idx, term in enumerate(terms):
        visible = [entry["term_idx"] == term_idx for entry in entries]
        dropdown_buttons.append(
            dict(
                label=term["name"],
                method="update",
                args=[
                    {"visible": visible, "y": response_ys},
                    {
                        "yaxis.title.text": "Relativity",
                        "shapes[0].y0": 1.0,
                        "shapes[0].y1": 1.0,
                        "updatemenus[0].active": 0,
                    },
                ],
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.23,
                xanchor="left",
                yanchor="top",
                bgcolor=_PLOTLY_PANEL,
                bordercolor="rgba(24, 33, 43, 0.10)",
                borderwidth=1,
                buttons=[
                    dict(
                        label="Response",
                        method="update",
                        args=[response_restyle, response_relayout],
                    ),
                    dict(label="Link", method="update", args=[link_restyle, link_relayout]),
                ],
                active=0,
                showactive=True,
            ),
            dict(
                type="dropdown",
                x=0.0,
                y=1.08,
                xanchor="left",
                yanchor="top",
                bgcolor=_PLOTLY_PANEL,
                bordercolor="rgba(24, 33, 43, 0.10)",
                borderwidth=1,
                buttons=dropdown_buttons,
                showactive=True,
            ),
        ],
        annotations=[
            dict(
                text="Scale",
                x=0.0,
                y=1.26,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            ),
            dict(
                text="Term",
                x=0.0,
                y=1.11,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
            ),
        ],
    )
    return fig
