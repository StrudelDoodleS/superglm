"""Build Part 4 figure: simulated hexagonal spatial grid with ring neighbourhoods.

Run from the repository root:
    uv run python docs/scripts/ncv_spatial_grid.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import Spline, SuperGLM

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "docs" / "_static" / "superglm.mplstyle"
OUT = ROOT / "docs" / "images" / "ncv" / "spatial_ring_grid.png"

COLOR_INK = "#15171C"
COLOR_RED = "#D6402B"
COLOR_YELLOW = "#F4B942"


def simulate_hex_grid(seed: int = 104, side: int = 16):
    """Build a rectangular odd-r hex grid in pure NumPy with a smooth trend and 1-ring shock."""
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    row = rows.ravel()
    col = cols.ravel()
    n = len(row)

    # Convert odd-r offset coordinates to axial (q, r) for exact hex distance.
    q = col - (row // 2)
    r = row

    # Pointy-top hex centres in grid units and normalised to [-1, 1] for modelling.
    hx = np.sqrt(3.0) * (q + 0.5 * r)
    hy = 1.5 * r
    east = 2.0 * (hx - hx.min()) / (hx.max() - hx.min()) - 1.0
    north = 2.0 * (hy - hy.min()) / (hy.max() - hy.min()) - 1.0

    # Exact axial hex distance (0 = same cell, 1 = immediate 6-hex ring).
    dq = np.abs(q[:, None] - q[None, :])
    dr = np.abs(r[:, None] - r[None, :])
    ds = np.abs((q[:, None] + r[:, None]) - (q[None, :] + r[None, :]))
    hex_dist = (dq + dr + ds) // 2

    # Local spatial disturbance shared across each cell and its immediate 1-ring.
    raw = rng.normal(0.0, 1.0, n)
    w = (hex_dist <= 1).astype(float)
    w /= np.sqrt(np.sum(w**2, axis=1, keepdims=True))
    spatial_noise = 0.50 * (w @ raw) + rng.normal(0.0, 0.12, n)

    truth = (
        0.85 * np.exp(-1.6 * ((east - 0.25) ** 2 + (north + 0.15) ** 2))
        - 0.40 * east
        + 0.35 * north
    )
    y = truth + spatial_noise
    X = pd.DataFrame({"east": east, "north": north})
    # Because two cells at hex distance 2 share a common neighbour in the 1-ring shock,
    # removing a 2-ring neighbourhood (up to 19 cells) blocks the local correlation.
    neighbours = [np.flatnonzero(hex_dist[i] <= 2) for i in range(n)]
    return X, y, truth, hx, hy, row, col, hex_dist, neighbours


def draw_hex_map(
    ax: plt.Axes,
    hx: np.ndarray,
    hy: np.ndarray,
    values: np.ndarray,
    norm: mcolors.Normalize,
    cmap: mcolors.Colormap,
    title: str,
    highlight_focal: int | None = None,
    highlight_ring: np.ndarray | None = None,
) -> None:
    for x0, y0, val in zip(hx, hy, values, strict=True):
        patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.98,
            orientation=0.0,
            facecolor=cmap(norm(val)),
            edgecolor="white",
            linewidth=0.35,
            zorder=1,
        )
        ax.add_patch(patch)

    if highlight_ring is not None:
        for idx in highlight_ring:
            if idx == highlight_focal:
                continue
            patch = mpatches.RegularPolygon(
                (hx[idx], hy[idx]),
                numVertices=6,
                radius=0.96,
                orientation=0.0,
                facecolor="none",
                edgecolor=COLOR_YELLOW,
                linewidth=2.2,
                zorder=3,
            )
            ax.add_patch(patch)
    if highlight_focal is not None:
        patch = mpatches.RegularPolygon(
            (hx[highlight_focal], hy[highlight_focal]),
            numVertices=6,
            radius=0.96,
            orientation=0.0,
            facecolor="none",
            edgecolor=COLOR_RED,
            linewidth=2.6,
            zorder=4,
        )
        ax.add_patch(patch)

    ax.set_aspect("equal")
    ax.set_xlim(hx.min() - 1.2, hx.max() + 1.2)
    ax.set_ylim(hy.min() - 1.2, hy.max() + 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=COLOR_INK, pad=6)


def main() -> None:
    plt.style.use(str(STYLE))
    X, y, truth, hx, hy, row, col, hex_dist, neighbours = simulate_hex_grid()

    features = {"east": Spline(kind="cr", k=8), "north": Spline(kind="cr", k=8)}
    interactions = [("east", "north")]

    with warnings.catch_warnings(record=True) as w_reml:
        warnings.simplefilter("always")
        m_reml = SuperGLM(family="gaussian", features=features, interactions=interactions).fit_reml(
            X, y
        )

    with warnings.catch_warnings(record=True) as w_ncv:
        warnings.simplefilter("always")
        m_ncv = SuperGLM(family="gaussian", features=features, interactions=interactions).fit_ncv(
            X, y, neighbours=neighbours
        )

    pred_reml = m_reml.predict(X)
    pred_ncv = m_ncv.predict(X)
    rmse_reml = float(np.sqrt(np.mean((pred_reml - truth) ** 2)))
    rmse_ncv = float(np.sqrt(np.mean((pred_ncv - truth) ** 2)))
    edf_reml = float(m_reml.result.effective_df)
    edf_ncv = float(m_ncv.result.effective_df)
    ncv_diag = m_ncv.ncv_diagnostics()
    nb = ncv_diag["neighbourhoods"]

    # Pick an interior cell to outline its 2-ring neighbourhood on the Truth panel.
    focal = int(np.flatnonzero((row == 8) & (col == 7))[0])
    ring = np.flatnonzero(hex_dist[focal] <= 2)

    lo = float(np.min(np.concatenate([truth, pred_reml, pred_ncv])))
    hi = float(np.max(np.concatenate([truth, pred_reml, pred_ncv])))
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    cmap = plt.colormaps["YlGnBu"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    fig.subplots_adjust(left=0.02, right=0.90, top=0.86, bottom=0.06, wspace=0.08)

    draw_hex_map(
        axes[0],
        hx,
        hy,
        truth,
        norm,
        cmap,
        "True spatial risk\n(example 2-ring outlined)",
        highlight_focal=focal,
        highlight_ring=ring,
    )
    axes[0].annotate(
        "scored cell +\n18 neighbours",
        xy=(hx[focal] + 1.4, hy[focal] - 1.2),
        xytext=(hx[focal] + 3.0, hy[focal] - 9.8),
        fontsize=8.3,
        fontweight="bold",
        color=COLOR_INK,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": COLOR_RED,
            "lw": 1.1,
        },
        arrowprops={"arrowstyle": "->", "color": COLOR_RED, "lw": 1.2},
    )

    draw_hex_map(
        axes[1],
        hx,
        hy,
        pred_reml,
        norm,
        cmap,
        f"fit_reml\nEDF {edf_reml:.1f} · RMSE {rmse_reml:.3f}",
    )
    draw_hex_map(
        axes[2],
        hx,
        hy,
        pred_ncv,
        norm,
        cmap,
        f"fit_ncv (2-ring neighbours)\nEDF {edf_ncv:.1f} · RMSE {rmse_ncv:.3f}",
    )

    cax = fig.add_axes([0.92, 0.14, 0.018, 0.68])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("Spatial risk effect", fontsize=9.2, color=COLOR_INK)
    cbar.ax.tick_params(labelsize=8.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    print(
        f"HEX_GRID cells={len(X)} side=16 ring_radius=2 "
        f"nb_count={nb['count']} nb_min={nb['size_min']} nb_median={nb['size_median']:.0f} nb_max={nb['size_max']}"
    )
    print(f"REML edf={edf_reml:.2f} rmse_truth={rmse_reml:.4f} warnings={len(w_reml)}")
    print(
        f"NCV edf={edf_ncv:.2f} rmse_truth={rmse_ncv:.4f} "
        f"n_iter={ncv_diag['n_iter']} n_eval={ncv_diag['n_evaluations']} "
        f"termination={ncv_diag['termination_reason']} converged={ncv_diag['converged']} warnings={len(w_ncv)}"
    )
    print(f"COMPARISON rmse_reduction_pct={(1.0 - rmse_ncv / rmse_reml) * 100.0:.1f}")
    print(f"WROTE {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
