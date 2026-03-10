"""Relativity plotting for SuperGLM models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray


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
    relativities: dict[str, pd.DataFrame],
    *,
    X: pd.DataFrame | None = None,
    exposure: NDArray | None = None,
    sample_weight: NDArray | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    with_ci: bool = True,
) -> Figure:
    """Create a grid of relativity plots from ``SuperGLM.relativities()`` output.

    Parameters
    ----------
    relativities : dict[str, DataFrame]
        Output of :meth:`SuperGLM.relativities`.
    X : DataFrame, optional
        Training data.  When provided together with *exposure*, an exposure
        distribution is shown on each subplot (filled area for continuous
        features, bars for categoricals).
    exposure : array-like, optional
        Backward-compatible alias for ``sample_weight``.
    sample_weight : array-like, optional
        Exposure/frequency weights corresponding to rows of *X*.
    ncols : int
        Number of subplot columns (default 2).
    figsize : tuple, optional
        Figure size ``(width, height)``.  Auto-sized if *None*.
    with_ci : bool
        If *True* (default) and ``se_log_relativity`` is present in the
        DataFrames, draw 95 % confidence bands / error bars.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if exposure is not None and sample_weight is not None:
        raise TypeError(
            "plot_relativities() received both 'exposure' and 'sample_weight'. "
            "Use only 'sample_weight'; 'exposure' is a backward-compatible alias."
        )
    if sample_weight is not None:
        exposure = sample_weight

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
                ax2.set_ylim(0, 1.3)  # leave headroom
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
                ax2.barh(
                    df["level"],
                    exp_vals,
                    color="lightgrey",
                    alpha=0.4,
                    label="Exposure",
                )
                ax2.set_yticks([])
                ax.set_zorder(ax2.get_zorder() + 1)
                ax.patch.set_visible(False)
            ax.barh(
                df["level"],
                df["relativity"],
                color="steelblue",
                label="Relativity",
            )
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
        # Grab from first axis that has both
        for ax_row in axes:
            for ax in ax_row:
                h, lab = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, lab
                    break
            if handles:
                break
        # Add the exposure patch from twin axis
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

    # Hide unused subplots
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
