"""Build Part 6 multi-seed comparison figure and Poisson one-step accuracy table.

Repeats the grouped curve simulation across 50 random seeds in three regimes:
  1. Correlated groups (30 groups of 20 rows), held out by group.
  2. Few groups (5 groups of 120 rows), held out by group (< 10 warning).
  3. Independent rows (600 rows, no group shock), single-row leave-out.

Also measures the accuracy of Poisson's one-step leave-out approximation against
exact brute-force leave-group-out refits as rows per coefficient decrease.

Run from the repository root:
    uv run python docs/scripts/ncv_when_to_use_study.py
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
OUT = ROOT / "docs" / "images" / "ncv" / "when_to_use_comparison.png"

COLOR_INK = "#15171C"
COLOR_RED = "#D6402B"
COLOR_YELLOW = "#F4B942"
COLOR_MUTED = "#6B7280"
COLOR_WIN_BG = "#F0FDF4"
COLOR_LOSE_BG = "#FEF2F2"


def true_curve(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x) + 0.5 * x


def sim_grouped(
    rng: np.random.Generator,
    n_groups: int,
    rows_per: int,
    *,
    center_sd: float,
    shock_sd: float = 0.45,
    row_sd: float = 0.12,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n = n_groups * rows_per
    groups = np.repeat(np.arange(n_groups), rows_per)
    centers = np.linspace(0.03, 0.97, n_groups) + rng.uniform(-0.015, 0.015, n_groups)
    centers = np.clip(centers, 0.02, 0.98)
    x = np.clip(centers[groups] + rng.normal(0.0, center_sd, n), 0.0, 1.0)
    y = true_curve(x) + rng.normal(0.0, shock_sd, n_groups)[groups] + rng.normal(0.0, row_sd, n)
    return pd.DataFrame({"x": x}), y, groups


def sim_independent(
    rng: np.random.Generator, n: int = 600, noise_sd: float = 0.466
) -> tuple[pd.DataFrame, np.ndarray, None]:
    x = rng.uniform(0.0, 1.0, n)
    y = true_curve(x) + rng.normal(0.0, noise_sd, n)
    return pd.DataFrame({"x": x}), y, None


def run_multiseed_study(n_seeds: int = 50) -> list[dict]:
    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 200)})
    truth_g = true_curve(grid["x"].to_numpy())
    spec = {"x": Spline(kind="ps", k=25)}

    regimes = [
        (
            "correlated_30g",
            "Correlated groups\n(30 groups × 20 rows)",
            lambda rng: sim_grouped(rng, 30, 20, center_sd=0.025),
            "groups",
        ),
        (
            "few_5g",
            "Few groups\n(5 groups × 120 rows)",
            lambda rng: sim_grouped(rng, 5, 120, center_sd=0.10),
            "groups",
        ),
        (
            "independent_rows",
            "Independent rows\n(600 rows, single-row NCV)",
            lambda rng: sim_independent(rng, 600),
            "rows",
        ),
    ]

    results: list[dict] = []
    for key, label, gen, mode in regimes:
        rmse_reml = np.empty(n_seeds)
        rmse_ncv = np.empty(n_seeds)
        reml_iters = np.empty(n_seeds, dtype=int)
        ncv_evals = np.empty(n_seeds, dtype=int)
        ncv_inner_solves = np.empty(n_seeds, dtype=int)
        ncv_factorisations = np.empty(n_seeds, dtype=int)
        few_warnings = 0

        for s in range(n_seeds):
            rng = np.random.default_rng(3000 + s)
            X, y, groups = gen(rng)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_reml = SuperGLM(family="gaussian", features=spec).fit_reml(X, y)

            with warnings.catch_warnings(record=True) as wn:
                warnings.simplefilter("always")
                if mode == "groups":
                    m_ncv = SuperGLM(family="gaussian", features=spec).fit_ncv(X, y, groups=groups)
                else:
                    m_ncv = SuperGLM(family="gaussian", features=spec).fit_ncv(X, y)

            if any("fewer than 10" in str(w.message) for w in wn):
                few_warnings += 1

            rmse_reml[s] = float(np.sqrt(np.mean((m_reml.predict(grid) - truth_g) ** 2)))
            rmse_ncv[s] = float(np.sqrt(np.mean((m_ncv.predict(grid) - truth_g) ** 2)))
            reml_iters[s] = int(m_reml.reml_diagnostics()["n_reml_iter"])
            dn = m_ncv.ncv_diagnostics()
            ncv_evals[s] = int(dn["n_evaluations"])
            ncv_inner_solves[s] = int(dn["work"]["inner_solves"])
            ncv_factorisations[s] = int(dn["work"]["factorisations"])

        ratios = rmse_ncv / rmse_reml
        row = {
            "key": key,
            "label": label,
            "n_seeds": n_seeds,
            "ncv_wins": int(np.sum(rmse_ncv < rmse_reml)),
            "med_reml": float(np.median(rmse_reml)),
            "med_ncv": float(np.median(rmse_ncv)),
            "mean_reml": float(np.mean(rmse_reml)),
            "mean_ncv": float(np.mean(rmse_ncv)),
            "med_ratio": float(np.median(ratios)),
            "p10_ratio": float(np.quantile(ratios, 0.10)),
            "p90_ratio": float(np.quantile(ratios, 0.90)),
            "max_ratio": float(np.max(ratios)),
            "med_reml_iter": float(np.median(reml_iters)),
            "med_ncv_eval": float(np.median(ncv_evals)),
            "med_ncv_inner": float(np.median(ncv_inner_solves)),
            "med_ncv_fact": float(np.median(ncv_factorisations)),
            "few_warnings": few_warnings,
            "ratios": ratios,
        }
        results.append(row)
        print(
            f"REGIME key={key} seeds={n_seeds} ncv_wins={row['ncv_wins']} "
            f"med_reml={row['med_reml']:.4f} med_ncv={row['med_ncv']:.4f} "
            f"mean_reml={row['mean_reml']:.4f} mean_ncv={row['mean_ncv']:.4f} "
            f"med_ratio={row['med_ratio']:.3f} p10_ratio={row['p10_ratio']:.3f} "
            f"p90_ratio={row['p90_ratio']:.3f} max_ratio={row['max_ratio']:.3f} "
            f"med_reml_iter={row['med_reml_iter']:.0f} med_ncv_eval={row['med_ncv_eval']:.0f} "
            f"med_ncv_inner={row['med_ncv_inner']:.0f} med_ncv_fact={row['med_ncv_fact']:.0f} "
            f"few_warnings={few_warnings}"
        )

    return results


def plot_multiseed_study(results: list[dict]) -> None:
    plt.style.use(str(STYLE))
    rng_plot = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    y_positions = np.arange(len(results))[::-1]

    max_x = max(float(r["max_ratio"]) for r in results) * 1.07
    min_x = min(float(np.min(r["ratios"])) for r in results) * 0.90

    ax.axvspan(min_x, 1.0, color=COLOR_WIN_BG, alpha=0.85, zorder=0)
    ax.axvspan(1.0, max_x, color=COLOR_LOSE_BG, alpha=0.65, zorder=0)
    ax.axvline(1.0, color=COLOR_MUTED, linestyle="--", linewidth=1.4, zorder=1)

    for y_pos, row in zip(y_positions, results, strict=True):
        ratios = row["ratios"]
        jitter = rng_plot.uniform(-0.14, 0.14, size=len(ratios))
        better = ratios < 1.0
        ax.scatter(
            ratios[better],
            y_pos + jitter[better],
            s=28,
            facecolor=COLOR_INK,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.72,
            zorder=3,
        )
        ax.scatter(
            ratios[~better],
            y_pos + jitter[~better],
            s=32,
            facecolor=COLOR_RED,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.82,
            zorder=3,
        )

        # Interquartile bar + median diamond
        q25, q75 = np.quantile(ratios, [0.25, 0.75])
        ax.plot([q25, q75], [y_pos, y_pos], color=COLOR_INK, linewidth=3.0, zorder=4)
        ax.scatter(
            [row["med_ratio"]],
            [y_pos],
            marker="D",
            s=66,
            facecolor=COLOR_YELLOW,
            edgecolor=COLOR_INK,
            linewidth=1.3,
            zorder=5,
        )
        ax.text(
            max_x - 0.02,
            y_pos + 0.27,
            f"median {row['med_ratio']:.2f}×  ({row['ncv_wins']}/{row['n_seeds']} closer)",
            va="center",
            ha="right",
            fontsize=8.6,
            fontweight="bold",
            color=COLOR_INK,
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#E5E7EB"},
            zorder=6,
        )

    ax.text(
        min_x + 0.03,
        len(results) - 0.22,
        "← fit_ncv closer to truth",
        fontsize=8.8,
        fontweight="bold",
        color="#166534",
        ha="left",
        va="center",
    )
    ax.text(
        1.04,
        len(results) - 0.22,
        "fit_reml closer to truth →",
        fontsize=8.8,
        fontweight="bold",
        color=COLOR_RED,
        ha="left",
        va="center",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in results], fontsize=9.5, color=COLOR_INK)
    ax.set_ylim(-0.48, len(results) - 0.02)
    ax.set_xlim(min_x, max_x)
    ax.grid(axis="x", color="#E5E3DC", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_title("RMSE ratio against true curve across 50 seeds (fit_ncv / fit_reml)")
    ax.set_xlabel("RMSE ratio (values below 1 mean fit_ncv is closer to the true curve)")
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(f"WROTE {OUT.relative_to(ROOT)}")


def run_poisson_onestep_check() -> None:
    """Compare Poisson one-step leave-out deviance against exact leave-group-out refits."""
    cases = [
        ("ample", 30, 40, 1, 10),
        ("moderate", 15, 40, 3, 10),
        ("thin", 4, 35, 3, 10),
    ]
    for key, n_groups, rows_per, n_terms, k in cases:
        rng = np.random.default_rng(777)
        n = n_groups * rows_per
        g = np.repeat(np.arange(n_groups), rows_per)
        data = {f"x{j}": rng.uniform(0.0, 1.0, n) for j in range(n_terms)}
        X = pd.DataFrame(data)
        eta = -0.2 + 0.5 * np.sin(2.0 * np.pi * data["x0"])
        if n_terms >= 2:
            eta = eta + 0.3 * (data["x1"] - 0.5)
        if n_terms >= 3:
            eta = eta - 0.25 * (data["x2"] - 0.5) ** 2
        eta = eta + rng.normal(0.0, 0.25, n_groups)[g]
        y = rng.poisson(np.exp(eta)).astype(float)

        interior = np.linspace(0.0, 1.0, k - 2)[1:-1].tolist()
        feats = {
            f"x{j}": Spline(kind="ps", knots=interior, boundary=(0.0, 1.0)) for j in range(n_terms)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SuperGLM(family="poisson", features=feats).fit_ncv(X, y, groups=g)
        diag = model.ncv_diagnostics()
        lambdas = diag["lambdas"]
        onestep_crit = float(diag["criterion"])

        exact_dev = 0.0
        for grp in range(n_groups):
            keep = g != grp
            drop = g == grp
            ref_feats = {
                f"x{j}": Spline(
                    kind="ps",
                    knots=interior,
                    boundary=(0.0, 1.0),
                    lambda_policy=LambdaPolicy.fixed(lambdas[f"x{j}"]),
                )
                for j in range(n_terms)
            }
            ref = SuperGLM(family="poisson", features=ref_feats).fit_reml(X.iloc[keep], y[keep])
            mu_hat = ref.predict(X.iloc[drop])
            yv = y[drop]
            pos = yv > 0
            log_term = np.zeros_like(yv)
            log_term[pos] = yv[pos] * np.log(yv[pos] / mu_hat[pos])
            dev = 2.0 * np.sum(log_term - (yv - mu_hat))
            exact_dev += float(dev)

        rel_err_pct = abs(onestep_crit - exact_dev) / exact_dev * 100.0
        p_total = 1 + n_terms * (k - 1)
        print(
            f"POISSON_ONESTEP case={key} n_rows={n} n_groups={n_groups} p={p_total} "
            f"rows_per_coef={n / p_total:.1f} onestep_dev={onestep_crit:.3f} "
            f"exact_dev={exact_dev:.3f} rel_err_pct={rel_err_pct:.3f} "
            f"termination={diag['termination_reason']} converged={diag['converged']}"
        )


def main() -> None:
    results = run_multiseed_study(n_seeds=50)
    plot_multiseed_study(results)
    run_poisson_onestep_check()


if __name__ == "__main__":
    main()
