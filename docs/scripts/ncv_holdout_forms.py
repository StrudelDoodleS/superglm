"""Build the four-panel diagram of what fit_ncv holds out and scores.

Run from the repository root:
    uv run python docs/scripts/ncv_holdout_forms.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "docs" / "_static" / "superglm.mplstyle"
OUT = ROOT / "docs" / "images" / "ncv" / "holdout_forms.png"

# High-contrast palette that reads cleanly on white canvas in light and dark themes.
COLOR_FIT_FILL = "#E5E7EB"
COLOR_FIT_EDGE = "#6B7280"
COLOR_REMOVED_FILL = "#F4B942"
COLOR_REMOVED_EDGE = "#92400E"
COLOR_SCORED_FILL = "#D6402B"
COLOR_SCORED_EDGE = "#15171C"
COLOR_INK = "#15171C"
COLOR_MUTED = "#6B7280"


def _style_panel(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=COLOR_INK, pad=18)
    ax.text(
        0.0,
        1.03,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.8,
        color=COLOR_MUTED,
        va="bottom",
        ha="left",
    )
    ax.grid(False)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_dots(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    roles: np.ndarray,
    size: float = 75.0,
) -> None:
    """Plot dots by role: 0 = in fit, 1 = removed only, 2 = removed and scored."""
    fit_mask = roles == 0
    rem_mask = roles == 1
    scr_mask = roles == 2

    if np.any(fit_mask):
        ax.scatter(
            xs[fit_mask],
            ys[fit_mask],
            s=size,
            facecolor=COLOR_FIT_FILL,
            edgecolor=COLOR_FIT_EDGE,
            linewidth=0.9,
            zorder=2,
        )
    if np.any(rem_mask):
        ax.scatter(
            xs[rem_mask],
            ys[rem_mask],
            s=size * 1.15,
            facecolor=COLOR_REMOVED_FILL,
            edgecolor=COLOR_REMOVED_EDGE,
            linewidth=1.2,
            zorder=3,
        )
    if np.any(scr_mask):
        ax.scatter(
            xs[scr_mask],
            ys[scr_mask],
            s=size * 1.25,
            facecolor=COLOR_SCORED_FILL,
            edgecolor=COLOR_SCORED_EDGE,
            linewidth=1.2,
            zorder=4,
        )


def main() -> None:
    plt.style.use(str(STYLE))
    rng = np.random.default_rng(20261126)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.84, bottom=0.05, wspace=0.14, hspace=0.36)

    # ── Panel 1: Single rows ─────────────────────────────────────────────────
    ax = axes[0, 0]
    _style_panel(
        ax,
        "1. Single rows — fit_ncv(X, y)",
        "One row leaves the fit and is scored; every other row stays in.",
    )
    n1 = 24
    x1 = np.linspace(0.08, 0.92, n1)
    y1 = 0.5 + 0.25 * np.sin(2.0 * np.pi * x1) + rng.normal(0.0, 0.08, n1)
    roles1 = np.zeros(n1, dtype=int)
    focal1 = 10
    roles1[focal1] = 2
    _draw_dots(ax, x1, y1, roles1, size=82)
    ax.annotate(
        "scored row",
        xy=(x1[focal1], y1[focal1]),
        xytext=(x1[focal1] + 0.14, y1[focal1] + 0.20),
        fontsize=8.5,
        color=COLOR_SCORED_FILL,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLOR_SCORED_FILL, "lw": 1.1},
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.05, 1.02)
    ax.text(
        0.5, 0.03, "covariate", transform=ax.transAxes, ha="center", fontsize=8.5, color=COLOR_MUTED
    )

    # ── Panel 2: Whole group ─────────────────────────────────────────────────
    ax = axes[0, 1]
    _style_panel(
        ax,
        "2. Whole group — groups=df['policy_id']",
        "Every row of one policy or fleet leaves and is scored together.",
    )
    n_groups = 6
    rows_per = 5
    gx_list, gy_list, groles_list = [], [], []
    group_centers_x = np.linspace(0.11, 0.89, n_groups)
    held_group = 2
    for g in range(n_groups):
        cx = group_centers_x[g]
        ys = np.linspace(0.24, 0.78, rows_per)
        xs = cx + rng.uniform(-0.022, 0.022, rows_per)
        gx_list.append(xs)
        gy_list.append(ys)
        groles_list.append(np.full(rows_per, 2 if g == held_group else 0, dtype=int))
        rect = mpatches.FancyBboxPatch(
            (cx - 0.055, 0.17),
            0.11,
            0.68,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor="#FEF2F2" if g == held_group else "#F8FAFC",
            edgecolor=COLOR_SCORED_FILL if g == held_group else "#CBD5E1",
            linewidth=1.3 if g == held_group else 0.9,
            linestyle="-" if g == held_group else "--",
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            0.08,
            f"P{g + 1}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold" if g == held_group else "normal",
            color=COLOR_SCORED_FILL if g == held_group else COLOR_MUTED,
        )
    _draw_dots(
        ax,
        np.concatenate(gx_list),
        np.concatenate(gy_list),
        np.concatenate(groles_list),
        size=76,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.96)

    # ── Panel 3: Row with its neighbours ─────────────────────────────────────
    ax = axes[1, 0]
    _style_panel(
        ax,
        "3. Row with its neighbours — neighbours=near",
        "The focal cell is scored; its ring of neighbours also leaves the fit.",
    )
    pts_q, pts_r = [], []
    for q in range(-3, 4):
        for r in range(-2, 3):
            if abs(q + r) <= 3:
                pts_q.append(q)
                pts_r.append(r)
    pts_q = np.array(pts_q)
    pts_r = np.array(pts_r)
    hx = np.sqrt(3.0) * (pts_q + 0.5 * pts_r)
    hy = 1.5 * pts_r
    dist = (np.abs(pts_q) + np.abs(pts_r) + np.abs(pts_q + pts_r)) // 2
    roles3 = np.where(dist == 0, 2, np.where(dist == 1, 1, 0))

    # Draw subtle hex outlines around every cell so the ring structure pops
    for x0, y0, role in zip(hx, hy, roles3, strict=True):
        face = "#FEE2E2" if role == 2 else ("#FEF3C7" if role == 1 else "#F8FAFC")
        edge = COLOR_SCORED_FILL if role == 2 else (COLOR_REMOVED_EDGE if role == 1 else "#CBD5E1")
        lw = 1.4 if role > 0 else 0.7
        hex_patch = mpatches.RegularPolygon(
            (x0, y0),
            numVertices=6,
            radius=0.96,
            orientation=0.0,
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=1 if role == 0 else 2,
        )
        ax.add_patch(hex_patch)
    _draw_dots(ax, hx, hy, roles3, size=68)
    ax.set_aspect("equal")
    ax.set_xlim(hx.min() - 1.3, hx.max() + 1.3)
    ax.set_ylim(hy.min() - 1.3, hy.max() + 1.3)

    # ── Panel 4: Later period (TimeSeriesSplit) ──────────────────────────────
    ax = axes[1, 1]
    _style_panel(
        ax,
        "4. Later period — cv=TimeSeriesSplit(...)",
        "Earlier periods fit; the next period is scored; future rows stay out.",
    )
    n_periods = 5
    per_period = 5
    period_x = np.linspace(0.12, 0.88, n_periods)
    px_list, py_list, proles_list = [], [], []
    # Split that trains on periods 0, 1, 2; scores period 3; excludes period 4 (future)
    for p in range(n_periods):
        cx = period_x[p]
        ys = np.linspace(0.26, 0.78, per_period)
        xs = cx + rng.uniform(-0.02, 0.02, per_period)
        role = 0 if p <= 2 else (2 if p == 3 else 1)
        px_list.append(xs)
        py_list.append(ys)
        proles_list.append(np.full(per_period, role, dtype=int))
        face = "#F8FAFC" if role == 0 else ("#FEF2F2" if role == 2 else "#FEF3C7")
        edge = "#CBD5E1" if role == 0 else (COLOR_SCORED_FILL if role == 2 else COLOR_REMOVED_EDGE)
        rect = mpatches.FancyBboxPatch(
            (cx - 0.068, 0.19),
            0.136,
            0.66,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3 if role > 0 else 0.9,
            linestyle="-" if role > 0 else "--",
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            0.09,
            f"Year {p + 1}",
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold" if role == 2 else "normal",
            color=COLOR_SCORED_FILL
            if role == 2
            else (COLOR_REMOVED_EDGE if role == 1 else COLOR_MUTED),
        )
    _draw_dots(
        ax,
        np.concatenate(px_list),
        np.concatenate(py_list),
        np.concatenate(proles_list),
        size=76,
    )
    ax.annotate(
        "",
        xy=(0.94, 0.01),
        xytext=(0.06, 0.01),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": COLOR_MUTED, "lw": 1.0},
    )
    ax.text(
        0.5,
        -0.04,
        "time (one forward split shown)",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.3,
        color=COLOR_MUTED,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.96)

    # ── Shared legend at top ─────────────────────────────────────────────────
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=COLOR_FIT_FILL,
            markeredgecolor=COLOR_FIT_EDGE,
            markeredgewidth=1.0,
            label="In fit (training rows)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor=COLOR_SCORED_FILL,
            markeredgecolor=COLOR_SCORED_EDGE,
            markeredgewidth=1.2,
            label="Removed from fit and scored",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=COLOR_REMOVED_FILL,
            markeredgecolor=COLOR_REMOVED_EDGE,
            markeredgewidth=1.2,
            label="Removed from fit only (not scored in this split)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.6,
        handletextpad=0.5,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(
        f"Wrote {OUT.relative_to(ROOT)} (panels=4, hex_cells={len(pts_q)}, ring_removed={int(np.sum(roles3 >= 1))})"
    )


if __name__ == "__main__":
    main()
