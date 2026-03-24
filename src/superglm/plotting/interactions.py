"""Interaction surface / effect plotting."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.plotting.common import (
    _PLOTLY_CAT_BAR_COLOR,
    _PLOTLY_DENSITY_SCALE,
    _PLOTLY_LINE_COLOR,
    _PLOTLY_SURFACE_SCALE,
    _apply_plotly_scene_style,
    _apply_plotly_theme,
    _hex_to_rgba,
    _kde_2d,
)

_CAT_BAR_COLOR = _PLOTLY_CAT_BAR_COLOR
_LINE_COLOR = _PLOTLY_LINE_COLOR


def plot_interaction(
    model,
    name: str,
    *,
    engine: str = "matplotlib",
    with_ci: bool = True,
    figsize: tuple[float, float] | None = None,
    colormap: str | None = None,
    show_contours: bool = True,
    n_points: int = 200,
    interaction_view: str = "surface",
    surface_opacity: float = 0.96,
    show_main_effect_walls: bool = False,
    X: pd.DataFrame | None = None,
    sample_weight: NDArray | None = None,
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
    n_points : int
        Grid resolution for interaction curves/surfaces. For surface plots this
        produces an ``n_points x n_points`` evaluation grid.
    interaction_view : {"surface", "contour", "contour_pair"}
        Plotly view for continuous x continuous interactions. ``"surface"``
        (default) returns the 3D surface. ``"contour"`` returns a 2D contour
        map. ``"contour_pair"`` returns a two-panel figure with the
        interaction contour and an exposure HDR-mass contour view. Ignored by
        matplotlib and non-surface interaction types.
    surface_opacity : float
        Opacity for Plotly 3D surface interactions (default 0.96). Ignored by
        other interaction plot types and the matplotlib backend.
    show_main_effect_walls : bool
        For Plotly 3D surface plots: project parent main-effect curves on the
        surface walls (default False).
    X : DataFrame, optional
        Training data. When provided with *sample_weight*, overlays an
        sample_weight-weighted density on surface plots (projected on the floor
        for 3D plotly, contour overlay for matplotlib).
    sample_weight : array-like, optional
        Exposure weights corresponding to rows of *X*.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if name not in model._interaction_specs:
        raise KeyError(
            f"Interaction not found: {name!r}. Available: {list(model._interaction_specs.keys())}"
        )

    ispec = model._interaction_specs[name]
    parent_names = ispec.parent_names
    raw = _reconstruct_interaction(model, name, n_points=n_points)

    density_data = None
    if X is not None and sample_weight is None:
        sample_weight = np.ones(len(X), dtype=np.float64)
    if X is not None and sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        p0, p1 = parent_names[0], parent_names[1]
        if (
            p0 in X.columns
            and p1 in X.columns
            and pd.api.types.is_numeric_dtype(X[p0])
            and pd.api.types.is_numeric_dtype(X[p1])
        ):
            density_data = (
                np.asarray(X[p0], dtype=np.float64),
                np.asarray(X[p1], dtype=np.float64),
                sample_weight,
            )

    if engine == "matplotlib":
        return _dispatch_mpl(
            raw, name, parent_names, with_ci, figsize, colormap, show_contours, density_data
        )
    elif engine == "plotly":
        return _dispatch_plotly(
            model,
            raw,
            name,
            parent_names,
            with_ci,
            colormap,
            show_contours,
            density_data,
            n_points,
            interaction_view,
            surface_opacity,
            show_main_effect_walls,
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


def _dispatch_plotly(
    model,
    raw,
    name,
    parent_names,
    with_ci,
    colormap,
    show_contours,
    density_data,
    n_points,
    interaction_view,
    surface_opacity,
    show_main_effect_walls,
):
    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError:
        raise ImportError(
            "plotly is required for engine='plotly'. Install it with: pip install plotly"
        ) from None

    valid_views = {"surface", "contour", "contour_pair"}
    if interaction_view not in valid_views:
        raise ValueError(
            f"interaction_view={interaction_view!r} is not valid, expected one of "
            f"{sorted(valid_views)}."
        )

    if "per_level" in raw and "x" in raw:
        if interaction_view != "surface":
            raise ValueError(
                f"interaction_view={interaction_view!r} is only supported for "
                "continuous x continuous interaction surfaces."
            )
        return _plot_varying_coefficient_plotly(raw, name, parent_names, with_ci, colormap)
    elif "pairs" in raw:
        if interaction_view != "surface":
            raise ValueError(
                f"interaction_view={interaction_view!r} is only supported for "
                "continuous x continuous interaction surfaces."
            )
        return _plot_categorical_heatmap_plotly(raw, name, parent_names, colormap)
    elif "relativities_per_unit" in raw:
        if interaction_view != "surface":
            raise ValueError(
                f"interaction_view={interaction_view!r} is only supported for "
                "continuous x continuous interaction surfaces."
            )
        return _plot_numeric_categorical_bars_plotly(raw, name, parent_names)
    elif "relativity_per_unit_unit" in raw:
        if interaction_view != "surface":
            raise ValueError(
                f"interaction_view={interaction_view!r} is only supported for "
                "continuous x continuous interaction surfaces."
            )
        return _plot_numeric_interaction_bar_plotly(raw, name, parent_names)
    elif "x1" in raw and "x2" in raw:
        if interaction_view == "contour":
            return _plot_surface_contour_plotly(raw, name, parent_names, colormap)
        if interaction_view == "contour_pair":
            return _plot_surface_contour_pair_plotly(
                raw,
                name,
                parent_names,
                colormap,
                density_data,
            )
        return _plot_surface_plotly(
            model,
            raw,
            name,
            parent_names,
            colormap,
            show_contours,
            density_data,
            n_points,
            surface_opacity,
            show_main_effect_walls,
        )
    else:
        raise ValueError(f"Cannot determine plot type for interaction {name!r}.")


def _reconstruct_interaction(model, name: str, *, n_points: int) -> dict:
    """Reconstruct an interaction, threading grid size when supported."""
    ispec = model._interaction_specs[name]
    groups = [g for g in model._groups if g.feature_name == name]
    beta = np.concatenate([model.result.beta[g.sl] for g in groups])

    params = inspect.signature(ispec.reconstruct).parameters
    if "n_points" in params:
        return ispec.reconstruct(beta, n_points=n_points)
    return ispec.reconstruct(beta)


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
                line=dict(color=colors[0], dash="dash", width=2.2),
                hovertemplate=(
                    f"{parent_names[1]}: {base}<br>{parent_names[0]}: %{{x:.3f}}"
                    "<br>Relativity: 1.0000<extra></extra>"
                ),
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
                line=dict(color=colors[ci], width=2.6),
                hovertemplate=(
                    f"{parent_names[1]}: {level}<br>{parent_names[0]}: %{{x:.3f}}"
                    "<br>Relativity: %{y:.4f}<extra></extra>"
                ),
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
                    fillcolor=_hex_to_rgba(colors[ci], 0.16),
                    opacity=0.15,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.add_hline(y=1.0, line_dash="dot", line_color="grey", opacity=0.4)
    fig.update_layout(
        title=dict(text=name, x=0.0, xanchor="left"),
        xaxis_title=parent_names[0],
        yaxis_title="Relativity",
    )
    _apply_plotly_theme(fig, height=520, hovermode="x unified")
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
            xgap=2,
            ygap=2,
            hovertemplate=(
                f"{parent_names[0]}: %{{y}}<br>"
                f"{parent_names[1]}: %{{x}}<br>"
                "Log-relativity: %{z:.4f}<br>"
                "Relativity: %{text}<extra></extra>"
            ),
            colorbar=dict(title="Log-Relativity"),
        )
    )
    fig.update_layout(
        title=dict(text=name, x=0.0, xanchor="left"),
        xaxis_title=parent_names[1],
        yaxis_title=parent_names[0],
    )
    _apply_plotly_theme(fig, height=max(420, 120 + 58 * len(levels1)), hovermode="closest")
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
    colors = (
        ["#D9D4C7"] + [_CAT_BAR_COLOR] * len(non_base) if base else [_CAT_BAR_COLOR] * len(non_base)
    )

    fig = go.Figure(
        go.Bar(
            x=levels,
            y=rels,
            marker=dict(color=colors, line=dict(color="rgba(24, 33, 43, 0.22)", width=1)),
            text=[f"{r:.3f}" for r in rels],
            textposition="outside",
            hovertemplate="%{x}<br>Relativity: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=dict(text=name, x=0.0, xanchor="left"),
        yaxis_title=f"Relativity per unit {parent_names[0]}",
    )
    _apply_plotly_theme(fig, height=460, hovermode="closest")
    return fig


def _plot_numeric_interaction_bar_plotly(raw, name, parent_names):
    """Interactive single vertical bar for numeric x numeric."""
    import plotly.graph_objects as go

    rel = raw["relativity_per_unit_unit"]
    label = f"{parent_names[0]} x {parent_names[1]}"

    fig = go.Figure(
        go.Bar(
            x=[label],
            y=[rel],
            marker=dict(color=_LINE_COLOR, line=dict(color="rgba(24, 33, 43, 0.22)", width=1)),
            text=[f"{rel:.3f}"],
            textposition="outside",
            hovertemplate=f"{label}<br>Relativity: %{{y:.4f}}<extra></extra>",
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=dict(text=name, x=0.0, xanchor="left"),
        yaxis_title="Relativity per unit x unit",
    )
    _apply_plotly_theme(fig, height=430, hovermode="closest")
    return fig


def _plot_surface_plotly(
    model,
    raw,
    name,
    parent_names,
    colormap,
    show_contours,
    density_data,
    n_points,
    surface_opacity,
    show_main_effect_walls,
):
    """3D interactive surface plot."""
    import plotly.graph_objects as go

    surface_opacity = float(np.clip(surface_opacity, 0.05, 1.0))
    x1 = np.asarray(raw["x1"], dtype=np.float64)
    x2 = np.asarray(raw["x2"], dtype=np.float64)
    Z = np.asarray(raw["relativity"], dtype=np.float64)
    x_pad = 0.1 * max(float(x1.max() - x1.min()), 1e-6)
    y_pad = 0.1 * max(float(x2.max() - x2.min()), 1e-6)
    z_span = max(float(Z.max() - Z.min()), 1e-6)
    z_floor = float(Z.min()) - 0.18 * z_span
    z_density = z_floor + 0.02 * z_span

    contours_z = {}
    if show_contours:
        contours_z = dict(
            show=True,
            usecolormap=False,
            color="rgba(22, 20, 17, 0.58)",
            highlightcolor="rgba(255,255,255,0.92)",
            highlightwidth=1,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x1,
            y=x2,
            z=Z,
            colorscale=colormap or _PLOTLY_SURFACE_SCALE,
            colorbar=dict(title="Relativity", x=1.0, len=0.76, y=0.58),
            contours_z=contours_z,
            showscale=True,
            opacity=surface_opacity,
            lighting=dict(ambient=0.68, diffuse=0.72, roughness=0.34, specular=0.18, fresnel=0.06),
            lightposition=dict(x=120, y=90, z=140),
            name="Relativity",
            hovertemplate=(
                f"{parent_names[0]}: %{{x:.3f}}<br>"
                f"{parent_names[1]}: %{{y:.3f}}<br>"
                "Relativity: %{z:.4f}<extra></extra>"
            ),
        )
    )

    if density_data is not None:
        d1, d2, w = density_data
        D = _kde_2d(d1, d2, w, x1, x2)
        # Suppress the diffuse low-density wash so the floor reads as a
        # concentration map rather than a tinted rectangle.
        D_focus = np.clip((D - 0.06) / 0.94, 0.0, 1.0)
        fig.add_trace(
            go.Surface(
                x=x1,
                y=x2,
                z=np.full_like(D_focus, z_density),
                surfacecolor=D_focus,
                customdata=D,
                colorscale=_PLOTLY_DENSITY_SCALE,
                cmin=0.0,
                cmax=1.0,
                opacity=0.96,
                showscale=True,
                colorbar=dict(title="Exposure<br>Density", x=1.12, len=0.5, y=0.25),
                lighting=dict(ambient=1.0, diffuse=0.35, roughness=1.0, specular=0.0),
                name="Exposure density",
                hovertemplate=(
                    f"{parent_names[0]}: %{{x:.3f}}<br>"
                    f"{parent_names[1]}: %{{y:.3f}}<br>"
                    "Density: %{customdata:.3f}<extra>Exposure</extra>"
                ),
            )
        )

    if show_main_effect_walls:
        wall_traces = _surface_main_effect_wall_traces(
            model, parent_names, x1, x2, z_density, n_points
        )
        for trace in wall_traces:
            fig.add_trace(trace)

    subtitle = "Surface plus parent main-effect wall traces" if show_main_effect_walls else ""
    title_text = f"{name}<br><sup>{subtitle}</sup>" if subtitle else name
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.0,
            xanchor="left",
        ),
        margin=dict(l=40, r=110, t=88, b=20),
    )
    _apply_plotly_theme(fig, height=760, hovermode="closest", legend_y=1.02, legend_x=0.0)
    _apply_plotly_scene_style(
        fig,
        x_title=parent_names[0],
        y_title=parent_names[1],
        z_title="Relativity",
        x_range=(float(x1.min()), float(x1.max() + x_pad)),
        y_range=(float(x2.min() - y_pad), float(x2.max())),
        z_range=(float(z_floor), float(Z.max() + 0.08 * z_span)),
    )
    return fig


def _plot_surface_contour_plotly(raw, name, parent_names, colormap):
    """2D contour plot for continuous x continuous interactions."""
    import plotly.graph_objects as go

    x1 = np.asarray(raw["x1"], dtype=np.float64)
    x2 = np.asarray(raw["x2"], dtype=np.float64)
    z = np.asarray(raw["relativity"], dtype=np.float64)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x1,
            y=x2,
            z=z,
            colorscale=colormap or _PLOTLY_SURFACE_SCALE,
            ncontours=18,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(color="rgba(22, 20, 17, 0.56)", width=1),
            colorbar=dict(title="Relativity"),
            hovertemplate=(
                f"{parent_names[0]}: %{{x:.3f}}<br>"
                f"{parent_names[1]}: %{{y:.3f}}<br>"
                "Relativity: %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(text=name, x=0.0, xanchor="left"),
        xaxis_title=parent_names[0],
        yaxis_title=parent_names[1],
    )
    fig.update_xaxes(range=[float(x1.min()), float(x1.max())], automargin=True)
    fig.update_yaxes(range=[float(x2.min()), float(x2.max())], automargin=True)
    _apply_plotly_theme(fig, height=520, hovermode="closest", showlegend=False)
    return fig


def _plot_surface_contour_pair_plotly(raw, name, parent_names, colormap, density_data):
    """Two-panel contour + exposure HDR-mass view for continuous interactions."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if density_data is None:
        raise ValueError(
            "interaction_view='contour_pair' requires X and sample_weight so the "
            "exposure HDR contours can be computed."
        )

    x1 = np.asarray(raw["x1"], dtype=np.float64)
    x2 = np.asarray(raw["x2"], dtype=np.float64)
    z = np.asarray(raw["relativity"], dtype=np.float64)
    d1, d2, w = density_data
    density = _kde_2d(d1, d2, w, x1, x2)
    mass_field = _highest_density_mass_field(density)
    hdr_scale = [
        [0.0, "#7A0403"],
        [0.12, "#B1121B"],
        [0.24, "#D54B39"],
        [0.38, "#EE8F55"],
        [0.52, "#F6C977"],
        [0.68, "#FBE5AD"],
        [0.84, "#FEF4D8"],
        [1.0, "#FFFDF7"],
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=("Interaction contour", "Exposure HDR mass"),
    )
    fig.add_trace(
        go.Contour(
            x=x1,
            y=x2,
            z=z,
            colorscale=colormap or _PLOTLY_SURFACE_SCALE,
            ncontours=18,
            contours=dict(coloring="heatmap", showlines=True),
            line=dict(color="rgba(22, 20, 17, 0.56)", width=1),
            colorbar=dict(title="Relativity", x=0.44, len=0.72, y=0.5),
            hovertemplate=(
                f"{parent_names[0]}: %{{x:.3f}}<br>"
                f"{parent_names[1]}: %{{y:.3f}}<br>"
                "Relativity: %{z:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            x=x1,
            y=x2,
            z=mass_field,
            customdata=density,
            colorscale=hdr_scale,
            zmin=0.0,
            zmax=1.0,
            autocontour=False,
            contours=dict(
                coloring="fill",
                showlines=False,
                start=0.1,
                end=0.9,
                size=0.1,
            ),
            colorbar=dict(title="HDR<br>Mass", x=1.03, len=0.72, y=0.5),
            hovertemplate=(
                f"{parent_names[0]}: %{{x:.3f}}<br>"
                f"{parent_names[1]}: %{{y:.3f}}<br>"
                "HDR mass: %{z:.0%}<br>"
                "Exposure density: %{customdata:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    for level, color in _hdr_border_levels(hdr_scale):
        fig.add_trace(
            go.Contour(
                x=x1,
                y=x2,
                z=mass_field,
                autocontour=False,
                contours=dict(
                    coloring="none",
                    showlines=True,
                    showlabels=True,
                    start=level,
                    end=level,
                    size=1e-6,
                    labelformat=".0%",
                    labelfont=dict(size=11, color=color),
                ),
                line=dict(color=color, width=1.35),
                showscale=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text=f"{name}<br><sup>Interaction contour plus exposure HDR mass contours</sup>",
            x=0.0,
            xanchor="left",
        ),
        margin=dict(l=52, r=120, t=90, b=50),
    )
    fig.update_xaxes(
        title_text=parent_names[0], range=[float(x1.min()), float(x1.max())], row=1, col=1
    )
    fig.update_yaxes(
        title_text=parent_names[1], range=[float(x2.min()), float(x2.max())], row=1, col=1
    )
    fig.update_xaxes(
        title_text=parent_names[0], range=[float(x1.min()), float(x1.max())], row=1, col=2
    )
    fig.update_yaxes(
        title_text=parent_names[1], range=[float(x2.min()), float(x2.max())], row=1, col=2
    )
    _apply_plotly_theme(fig, height=520, hovermode="closest", showlegend=False)
    return fig


def _highest_density_mass_field(density: np.ndarray) -> np.ndarray:
    """Map each cell to the mass of its highest-density superlevel set."""
    flat = np.asarray(density, dtype=np.float64).ravel()
    total = float(np.sum(flat))
    if total <= 0:
        return np.zeros_like(density, dtype=np.float64)

    _, inverse = np.unique(flat, return_inverse=True)
    mass_by_value = np.bincount(inverse, weights=flat)
    cumulative_desc = np.cumsum(mass_by_value[::-1]) / total
    mass_field = cumulative_desc[::-1][inverse]
    return mass_field.reshape(density.shape)


def _hdr_border_levels(colorscale: list[list[float | str]]) -> list[tuple[float, str]]:
    """Return decile HDR levels with darker sampled colors for contrast."""
    from plotly.colors import sample_colorscale

    levels = [round(i / 10.0, 1) for i in range(1, 10)]
    base_colors = sample_colorscale(colorscale, [1.0 - level for level in levels], colortype="rgb")
    out: list[tuple[float, str]] = []
    for level, color in zip(levels, base_colors):
        mid_weight = max(0.0, 1.0 - abs(level - 0.5) / 0.45)
        darken = 0.18 + 0.28 * (mid_weight**1.2)
        out.append((level, _darken_rgb(color, darken)))
    return out


def _darken_rgb(color: str, amount: float) -> str:
    """Darken an ``rgb(r, g, b)`` color string by ``amount``."""
    if not color.startswith("rgb(") or not color.endswith(")"):
        return color
    parts = [int(p.strip()) for p in color[4:-1].split(",")]
    factor = max(0.0, min(1.0, 1.0 - amount))
    darkened = [int(round(channel * factor)) for channel in parts]
    return f"rgb({darkened[0]}, {darkened[1]}, {darkened[2]})"


def _surface_main_effect_wall_traces(model, parent_names, x1, x2, z_floor, n_points):
    """Build projected parent main-effect traces for numeric surface plots."""
    import plotly.graph_objects as go

    traces = []
    x_pad = 0.06 * max(float(x1.max() - x1.min()), 1e-6)
    y_pad = 0.06 * max(float(x2.max() - x2.min()), 1e-6)

    for axis, parent in enumerate(parent_names):
        if parent not in model._specs:
            continue
        ti = model.term_inference(parent, with_se=False, n_points=n_points)
        if ti.kind not in ("spline", "polynomial", "numeric"):
            continue

        if ti.x is None or ti.relativity is None:
            continue

        if axis == 0:
            traces.append(
                go.Scatter3d(
                    x=np.asarray(ti.x, dtype=np.float64),
                    y=np.full(len(ti.x), float(x2.min() - y_pad)),
                    z=np.asarray(ti.relativity, dtype=np.float64),
                    mode="lines",
                    name=f"Main effect: {parent}",
                    showlegend=False,
                    line=dict(color="#1E67D8", width=6),
                    hovertemplate=(
                        f"{parent}: %{{x:.3f}}<br>Main-effect relativity: %{{z:.4f}}<extra></extra>"
                    ),
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=np.full(len(ti.x), float(x1.max() + x_pad)),
                    y=np.asarray(ti.x, dtype=np.float64),
                    z=np.asarray(ti.relativity, dtype=np.float64),
                    mode="lines",
                    name=f"Main effect: {parent}",
                    showlegend=False,
                    line=dict(color="#D61F2C", width=6),
                    hovertemplate=(
                        f"{parent}: %{{y:.3f}}<br>Main-effect relativity: %{{z:.4f}}<extra></extra>"
                    ),
                )
            )

    if traces:
        traces.append(
            go.Scatter3d(
                x=[float(x1.min()), float(x1.max())],
                y=[float(x2.min()), float(x2.min())],
                z=[float(z_floor), float(z_floor)],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(color="rgba(24, 33, 43, 0.18)", width=2, dash="dash"),
            )
        )
    return traces


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
