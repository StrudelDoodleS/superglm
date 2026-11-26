"""Build Parts 2 and 3 figures: grouped-noise curve fit and held-out error vs smoothing.

Run from the repository root:
    uv run python docs/scripts/ncv_grouped_curve.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import LambdaPolicy, Spline, SuperGLM

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "docs" / "_static" / "superglm.mplstyle"
OUT_CURVE = ROOT / "docs" / "images" / "ncv" / "grouped_curve_comparison.png"
OUT_SWEEP = ROOT / "docs" / "images" / "ncv" / "criterion_vs_smoothing.png"

COLOR_INK = "#15171C"
COLOR_RED = "#D6402B"
COLOR_YELLOW = "#F4B942"
COLOR_MUTED = "#6B7280"
COLOR_DOTS = "#CBD5E1"


def true_curve(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) + 0.5 * x


def simulate_grouped_data(
    seed: int = 20261126,
    n_groups: int = 30,
    rows_per_group: int = 20,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate rows grouped in local stretches of x with shared group shocks."""
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    centers = np.linspace(0.03, 0.97, n_groups) + rng.uniform(-0.015, 0.015, n_groups)
    centers = np.clip(centers, 0.02, 0.98)
    x = np.clip(centers[groups] + rng.normal(0.0, 0.025, n), 0.0, 1.0)
    group_shock = rng.normal(0.0, 0.45, n_groups)
    y = true_curve(x) + group_shock[groups] + rng.normal(0.0, 0.12, n)
    return pd.DataFrame({"x": x}), y, groups


def main() -> None:
    plt.style.use(str(STYLE))
    X, y, groups = simulate_grouped_data()
    x = X["x"].to_numpy()
    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 300)})
    xg = grid["x"].to_numpy()
    truth_g = true_curve(xg)

    spec = {"x": Spline(kind="ps", k=25)}

    with warnings.catch_warnings(record=True) as w_reml:
        warnings.simplefilter("always")
        m_reml = SuperGLM(family="gaussian", features=spec).fit_reml(X, y)

    with warnings.catch_warnings(record=True) as w_ncv:
        warnings.simplefilter("always")
        m_ncv = SuperGLM(family="gaussian", features=spec).fit_ncv(X, y, groups=groups)

    pred_reml = m_reml.predict(grid)
    pred_ncv = m_ncv.predict(grid)
    rmse_reml = float(np.sqrt(np.mean((pred_reml - truth_g) ** 2)))
    rmse_ncv = float(np.sqrt(np.mean((pred_ncv - truth_g) ** 2)))

    reml_diag = m_reml.reml_diagnostics()
    ncv_diag = m_ncv.ncv_diagnostics()
    lam_reml = float(reml_diag["lambdas"]["x"])
    lam_ncv = float(ncv_diag["lambdas"]["x"])
    edf_reml = float(m_reml.term_inference("x").edf)
    edf_ncv = float(m_ncv.term_inference("x").edf)

    # Evaluate the exact leave-group-out held-out error at REML's lambda and over a log grid.
    def eval_criterion_mean(lam: float) -> float:
        fixed_model = SuperGLM(
            family="gaussian",
            features={"x": Spline(kind="ps", k=25, lambda_policy=LambdaPolicy.fixed(lam))},
        ).fit_ncv(X, y, groups=groups)
        return float(fixed_model.ncv_diagnostics()["criterion_mean"])

    sweep_lambdas = np.logspace(-2, 5, 45)
    sweep_errors = np.array([eval_criterion_mean(float(lam)) for lam in sweep_lambdas])
    crit_at_reml = eval_criterion_mean(lam_reml)
    crit_at_ncv = float(ncv_diag["criterion_mean"])

    # ── Figure 1 (Part 2): Truth vs fit_reml vs fit_ncv ──────────────────────
    OUT_CURVE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))

    # Individual rows in pale neutral grey; group means as ringed dots so the
    # shared group shifts are visible at a glance.
    ax.scatter(
        x, y, s=11, color=COLOR_DOTS, alpha=0.85, linewidth=0, label="Rows (20 per group)", zorder=1
    )
    gx_mean = np.array([x[groups == g].mean() for g in range(30)])
    gy_mean = np.array([y[groups == g].mean() for g in range(30)])
    ax.scatter(
        gx_mean,
        gy_mean,
        s=34,
        facecolor="white",
        edgecolor=COLOR_MUTED,
        linewidth=1.1,
        label="Group means (30 groups)",
        zorder=2,
    )

    ax.plot(
        xg, truth_g, color=COLOR_MUTED, linestyle="--", linewidth=1.9, label="True curve", zorder=3
    )
    ax.plot(
        xg,
        pred_reml,
        color=COLOR_RED,
        linewidth=2.1,
        label=f"fit_reml (EDF {edf_reml:.1f}, RMSE {rmse_reml:.3f})",
        zorder=4,
    )
    ax.plot(
        xg,
        pred_ncv,
        color=COLOR_INK,
        linewidth=2.4,
        label=f"fit_ncv by group (EDF {edf_ncv:.1f}, RMSE {rmse_ncv:.3f})",
        zorder=5,
    )

    ax.set_title("When rows in a group share noise, fit_reml chases the group shocks")
    ax.set_xlabel("Covariate x")
    ax.set_ylabel("Response y")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(OUT_CURVE, dpi=160)
    plt.close(fig)

    # ── Figure 2 (Part 3): Held-out error against smoothing ────────────────────
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(
        sweep_lambdas,
        sweep_errors,
        color=COLOR_INK,
        linewidth=2.2,
        label="Leave-group-out mean squared error",
        zorder=2,
    )
    ax.axvline(lam_reml, color=COLOR_RED, linestyle="--", linewidth=1.4, zorder=3)
    ax.axvline(lam_ncv, color=COLOR_INK, linestyle=":", linewidth=1.6, zorder=3)
    ax.scatter(
        [lam_reml],
        [crit_at_reml],
        s=68,
        facecolor=COLOR_RED,
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
        label=f"fit_reml choice (λ = {lam_reml:.2f}, MSE = {crit_at_reml:.3f})",
    )
    ax.scatter(
        [lam_ncv],
        [crit_at_ncv],
        s=74,
        facecolor=COLOR_YELLOW,
        edgecolor=COLOR_INK,
        linewidth=1.4,
        zorder=6,
        label=f"fit_ncv choice (λ = {lam_ncv:.1f}, MSE = {crit_at_ncv:.3f})",
    )

    y_span = float(sweep_errors.max() - sweep_errors.min())
    ax.annotate(
        f"fit_reml\nλ = {lam_reml:.2f}\nEDF {edf_reml:.1f}",
        xy=(lam_reml, crit_at_reml),
        xytext=(lam_reml * 0.14, crit_at_reml - 0.30 * y_span),
        fontsize=9.0,
        color=COLOR_RED,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLOR_RED, "lw": 1.1},
    )
    ax.annotate(
        f"fit_ncv\nλ = {lam_ncv:.1f}\nEDF {edf_ncv:.1f}",
        xy=(lam_ncv, crit_at_ncv),
        xytext=(lam_ncv * 0.08, crit_at_ncv + 0.28 * y_span),
        fontsize=9.0,
        color=COLOR_INK,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLOR_INK, "lw": 1.1},
    )

    ax.set_xscale("log")
    ax.set_title("Leave-group-out held-out error against smoothing parameter")
    ax.set_xlabel("Smoothing parameter λ (log scale; larger is smoother)")
    ax.set_ylabel("Mean held-out squared error per row")
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_SWEEP, dpi=160)
    plt.close(fig)

    print(
        "GROUPED_DATA n_rows=600 n_groups=30 rows_per_group=20 group_shock_sd=0.45 row_noise_sd=0.12 basis_k=25"
    )
    print(
        f"REML lambda={lam_reml:.4f} edf={edf_reml:.2f} rmse_truth={rmse_reml:.4f} heldout_mse={crit_at_reml:.4f} warnings={len(w_reml)}"
    )
    print(
        f"NCV lambda={lam_ncv:.2f} edf={edf_ncv:.2f} rmse_truth={rmse_ncv:.4f} "
        f"heldout_mse={crit_at_ncv:.4f} criterion={ncv_diag['criterion']:.2f} "
        f"n_iter={ncv_diag['n_iter']} n_eval={ncv_diag['n_evaluations']} "
        f"termination={ncv_diag['termination_reason']} converged={ncv_diag['converged']} warnings={len(w_ncv)}"
    )
    print(
        f"COMPARISON rmse_ratio_ncv_over_reml={rmse_ncv / rmse_reml:.3f} "
        f"rmse_reduction_pct={(1.0 - rmse_ncv / rmse_reml) * 100.0:.1f} "
        f"heldout_mse_reduction_pct={(1.0 - crit_at_ncv / crit_at_reml) * 100.0:.1f}"
    )
    print(f"WROTE {OUT_CURVE.relative_to(ROOT)} and {OUT_SWEEP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
