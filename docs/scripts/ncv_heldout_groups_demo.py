"""Build Part 5 figure and table: actual vs held-out expected by group (Poisson fleets).

Run from the repository root:
    uv run python docs/scripts/ncv_heldout_groups_demo.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import Spline, SuperGLM

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "docs" / "_static" / "superglm.mplstyle"
OUT = ROOT / "docs" / "images" / "ncv" / "heldout_groups_ae.png"

COLOR_INK = "#15171C"
COLOR_RED = "#D6402B"
COLOR_YELLOW = "#F4B942"
COLOR_MUTED = "#6B7280"
COLOR_BAND = "#F3F4F6"


def simulate_fleet_book(seed: int = 20261128, n_fleets: int = 20, rows_per_fleet: int = 35):
    """Simulate a Poisson motor-fleet claim-frequency book with fleet-level shocks."""
    rng = np.random.default_rng(seed)
    n = n_fleets * rows_per_fleet
    fleet_names = np.array([f"Fleet_{i + 1:02d}" for i in range(n_fleets)])
    g_idx = np.repeat(np.arange(n_fleets), rows_per_fleet)
    fleets = fleet_names[g_idx]

    centers = np.linspace(1.0, 14.0, n_fleets) + rng.uniform(-0.6, 0.6, n_fleets)
    veh_age = np.clip(centers[g_idx] + rng.normal(0.0, 1.4, n), 0.0, 15.0)
    exposure = rng.uniform(8.0, 22.0, n)

    fleet_shock = rng.normal(0.0, 0.22, n_fleets)
    fleet_shock[3] = 0.52  # Fleet_04: elevated experience
    fleet_shock[15] = -0.48  # Fleet_16: favourable experience

    true_log_rate = -2.2 + 0.55 * np.exp(-veh_age / 3.0) + 0.008 * (veh_age - 6.0) ** 2
    claims = rng.poisson(exposure * np.exp(true_log_rate + fleet_shock[g_idx])).astype(float)
    X = pd.DataFrame({"veh_age": veh_age})
    return X, claims, exposure, fleets


def main() -> None:
    plt.style.use(str(STYLE))
    X, claims, exposure, fleets = simulate_fleet_book()
    offset = np.log(exposure)
    spec = {"veh_age": Spline(kind="cr", k=14)}

    with warnings.catch_warnings(record=True) as w_ncv:
        warnings.simplefilter("always")
        model = SuperGLM(family="poisson", features=spec).fit_ncv(
            X, claims, offset=offset, groups=fleets
        )

    diag = model.ncv_diagnostics()
    table = model.ncv_heldout_groups()
    sorted_tbl = table.sort_values("actual_over_expected", ascending=False)

    exp_vals = table["heldout_expected"].to_numpy()
    act_vals = table["actual"].to_numpy()
    ae_vals = table["actual_over_expected"].to_numpy()
    names = table.index.to_numpy()

    # Highlight fleets more than 25% away from held-out expected (A/E > 1.25 or < 0.75).
    flagged = (ae_vals > 1.25) | (ae_vals < 0.75)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    lo = min(float(exp_vals.min()), float(act_vals.min())) * 0.88
    hi = max(float(exp_vals.max()), float(act_vals.max())) * 1.08
    xs = np.linspace(lo, hi, 200)

    # +/- 25% reference band around A/E = 1.0 for visual orientation
    ax.fill_between(
        xs,
        0.75 * xs,
        1.25 * xs,
        color=COLOR_BAND,
        alpha=0.9,
        label="Within ±25% of held-out expected",
        zorder=1,
    )
    ax.plot(
        xs,
        xs,
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=1.4,
        label="Actual = held-out expected (A/E = 1.0)",
        zorder=2,
    )

    ax.scatter(
        exp_vals[~flagged],
        act_vals[~flagged],
        s=58,
        facecolor=COLOR_INK,
        edgecolor="white",
        linewidth=0.9,
        label="Fleets within ±25%",
        zorder=4,
    )
    ax.scatter(
        exp_vals[flagged],
        act_vals[flagged],
        s=76,
        facecolor=COLOR_RED,
        edgecolor=COLOR_INK,
        linewidth=1.1,
        label="Standout fleets (|A/E − 1| > 0.25)",
        zorder=5,
    )

    # Label every flagged fleet cleanly without overlapping points or the legend.
    label_offsets = {
        "Fleet_20": (9, -3),
        "Fleet_04": (9, -3),
        "Fleet_01": (-28, 10),
        "Fleet_16": (9, -5),
        "Fleet_10": (9, -4),
    }
    for name, ex, ac, ae in zip(
        names[flagged], exp_vals[flagged], act_vals[flagged], ae_vals[flagged], strict=True
    ):
        dx, dy = label_offsets.get(str(name), (8, 4))
        ax.annotate(
            f"{name} ({ae:.2f})",
            xy=(ex, ac),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.6,
            fontweight="bold",
            color=COLOR_RED,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Actual against held-out expected claims by fleet (from ncv_heldout_groups)")
    ax.set_xlabel("Held-out expected claims (predicted with that fleet removed)")
    ax.set_ylabel("Actual claims")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.94)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    print(
        f"FLEET_BOOK n_rows={len(X)} n_fleets={len(table)} total_claims={int(claims.sum())} "
        f"total_heldout_expected={table['heldout_expected'].sum():.1f} "
        f"overall_ae={claims.sum() / table['heldout_expected'].sum():.3f} "
        f"edf={model.term_inference('veh_age').edf:.2f} "
        f"criterion={diag['criterion']:.2f} criterion_mean={diag['criterion_mean']:.4f} "
        f"n_iter={diag['n_iter']} termination={diag['termination_reason']} "
        f"converged={diag['converged']} warnings={len(w_ncv)}"
    )
    print(
        f"FLAGGED_COUNT count={int(flagged.sum())} names={','.join(sorted_tbl.index[sorted_tbl['actual_over_expected'].gt(1.25) | sorted_tbl['actual_over_expected'].lt(0.75)].tolist())}"
    )
    print("TABLE_TOP3_AND_BOTTOM3")
    excerpt = pd.concat([sorted_tbl.head(3), sorted_tbl.iloc[[7, 8]], sorted_tbl.tail(3)])
    for grp, row_s in excerpt.iterrows():
        print(
            f"ROW group={grp} rows={int(row_s['rows'])} weight={row_s['weight']:.0f} "
            f"actual={row_s['actual']:.0f} heldout_expected={row_s['heldout_expected']:.1f} "
            f"actual_over_expected={row_s['actual_over_expected']:.2f} "
            f"heldout_deviance={row_s['heldout_deviance']:.1f}"
        )
    print(f"WROTE {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
