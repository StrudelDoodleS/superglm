"""Interaction surface / effect plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.plotting.common import _kde_2d


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
    if X is not None and exposure is None:
        exposure = np.ones(len(X), dtype=np.float64)
    if X is not None and exposure is not None:
        exposure = np.asarray(exposure, dtype=np.float64)
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
