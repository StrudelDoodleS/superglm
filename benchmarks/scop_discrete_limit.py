"""Sweep discrete=True monotone REML scaling and identifiability.

Tracked benchmark harness for SCOP monotone `PSpline` terms.

Outputs:
- benchmarks/results/scop_discrete_limit_results.csv
- benchmarks/results/scop_discrete_limit_scaling.png
- benchmarks/results/scop_discrete_limit_identifiability.png
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.spline import Spline

RESULTS_DIR = Path("benchmarks/results")
CSV_PATH = RESULTS_DIR / "scop_discrete_limit_results.csv"
SCALING_PNG = RESULTS_DIR / "scop_discrete_limit_scaling.png"
IDENT_PNG = RESULTS_DIR / "scop_discrete_limit_identifiability.png"


@dataclass
class RunRow:
    sweep: str
    model_type: str
    n: int
    n_scop: int
    k: int
    runtime_s: float
    converged: bool
    outer_iters: int
    inner_avg: float
    inner_max: int
    max_abs_native_mean_scop: float
    max_abs_weighted_mean_scop: float
    max_abs_native_mean_all_splines: float
    min_monotone_diff: float
    intercept: float


def make_dataset(n: int, n_scop: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    eta = np.full(n, -1.7)

    for j in range(n_scop):
        x = rng.uniform(0.0, 1.0, size=n)
        data[f"x{j + 1}"] = x
        eta += 0.55 * np.log1p(6.0 * x)

    z1 = rng.uniform(0.0, 1.0, size=n)
    z2 = rng.uniform(0.0, 1.0, size=n)
    data["z1"] = z1
    data["z2"] = z2
    eta += 0.35 * np.sin(2.0 * math.pi * z1)
    eta += -0.45 * (z2 - 0.5) ** 2

    exposure = rng.uniform(0.4, 1.6, size=n)
    mu = np.exp(eta)
    counts = rng.poisson(exposure * mu)
    y = counts / exposure
    return pd.DataFrame(data), y.astype(np.float64), exposure.astype(np.float64)


def make_features(n_scop: int, k: int, monotone: bool) -> dict[str, object]:
    features: dict[str, object] = {}
    for j in range(n_scop):
        if monotone:
            features[f"x{j + 1}"] = Spline(
                kind="ps",
                k=k,
                monotone="increasing",
                monotone_mode="fit",
            )
        else:
            features[f"x{j + 1}"] = Spline(kind="ps", k=k)
    features["z1"] = Spline(kind="cr", k=12)
    features["z2"] = Spline(kind="cr", k=10)
    return features


def _combined_beta(model: SuperGLM, feature_name: str) -> np.ndarray:
    groups = [g for g in model._groups if g.feature_name == feature_name]
    return np.concatenate([model.result.beta[g.sl] for g in groups])


def _term_eta(model: SuperGLM, X: pd.DataFrame, feature_name: str) -> np.ndarray:
    spec = model._specs[feature_name]
    beta = _combined_beta(model, feature_name)
    x = X[feature_name].to_numpy(dtype=np.float64)
    return np.asarray(spec.transform(x) @ beta, dtype=np.float64)


def _term_monotone_min_diff(model: SuperGLM, feature_name: str) -> float:
    spec = model._specs[feature_name]
    beta = _combined_beta(model, feature_name)
    grid = np.linspace(spec._lo, spec._hi, 300)
    eta_grid = np.asarray(spec.transform(grid) @ beta, dtype=np.float64)
    return float(np.min(np.diff(eta_grid)))


def run_case(
    *,
    sweep: str,
    n: int,
    n_scop: int,
    k: int,
    monotone: bool,
    seed: int,
    max_reml_iter: int,
) -> RunRow:
    X, y, exposure = make_dataset(n=n, n_scop=n_scop, seed=seed)
    model = SuperGLM(
        family="poisson",
        discrete=True,
        features=make_features(n_scop=n_scop, k=k, monotone=monotone),
    )

    t0 = time.perf_counter()
    model.fit_reml(X, y, sample_weight=exposure, max_reml_iter=max_reml_iter)
    runtime_s = time.perf_counter() - t0

    rr = model._reml_result
    inner_hist = list(rr.inner_iter_history or [])

    spline_names = [f"x{j + 1}" for j in range(n_scop)] + ["z1", "z2"]
    scop_names = [f"x{j + 1}" for j in range(n_scop)] if monotone else []

    native_means_all = []
    native_means_scop = []
    weighted_means_scop = []
    min_diffs = []

    for name in spline_names:
        eta = _term_eta(model, X, name)
        native_means_all.append(abs(float(np.mean(eta))))
        if name in scop_names:
            native_means_scop.append(abs(float(np.mean(eta))))
            weighted_means_scop.append(abs(float(np.average(eta, weights=exposure))))
            min_diffs.append(_term_monotone_min_diff(model, name))

    return RunRow(
        sweep=sweep,
        model_type="scop" if monotone else "baseline",
        n=n,
        n_scop=n_scop,
        k=k,
        runtime_s=runtime_s,
        converged=bool(rr.converged),
        outer_iters=int(rr.n_reml_iter),
        inner_avg=float(np.mean(inner_hist)) if inner_hist else float(model.result.n_iter),
        inner_max=max(inner_hist) if inner_hist else int(model.result.n_iter),
        max_abs_native_mean_scop=max(native_means_scop) if native_means_scop else 0.0,
        max_abs_weighted_mean_scop=max(weighted_means_scop) if weighted_means_scop else 0.0,
        max_abs_native_mean_all_splines=max(native_means_all) if native_means_all else 0.0,
        min_monotone_diff=min(min_diffs) if min_diffs else float("nan"),
        intercept=float(model.result.intercept),
    )


def run_suite(max_reml_iter: int) -> pd.DataFrame:
    rows: list[RunRow] = []

    for n_scop in (1, 2):
        for n in (50_000, 100_000, 250_000, 500_000):
            for monotone in (False, True):
                rows.append(
                    run_case(
                        sweep="n",
                        n=n,
                        n_scop=n_scop,
                        k=14,
                        monotone=monotone,
                        seed=1000 + 17 * n_scop + n // 1000,
                        max_reml_iter=max_reml_iter,
                    )
                )

    for n_scop in (1, 2, 3, 4, 5, 6):
        for monotone in (False, True):
            rows.append(
                run_case(
                    sweep="terms",
                    n=100_000,
                    n_scop=n_scop,
                    k=14,
                    monotone=monotone,
                    seed=2000 + 31 * n_scop,
                    max_reml_iter=max_reml_iter,
                )
            )

    for k in (10, 14, 20, 30, 40):
        rows.append(
            run_case(
                sweep="k",
                n=100_000,
                n_scop=2,
                k=k,
                monotone=True,
                seed=3000 + k,
                max_reml_iter=max_reml_iter,
            )
        )

    df = pd.DataFrame(asdict(r) for r in rows)
    df.to_csv(CSV_PATH, index=False)
    return df


def plot_scaling(df: pd.DataFrame) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    n_df = df[df["sweep"] == "n"].copy()
    for n_scop, ax in zip((1, 2), axes[:2], strict=True):
        sub = n_df[n_df["n_scop"] == n_scop]
        for model_type, color in (("baseline", "#f4a261"), ("scop", "#2a9d8f")):
            ss = sub[sub["model_type"] == model_type].sort_values("n")
            ax.plot(ss["n"], ss["runtime_s"], marker="o", label=model_type, color=color)
        ax.set_title(f"Runtime vs n ({n_scop} monotone term{'s' if n_scop > 1 else ''})")
        ax.set_xlabel("n")
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    term_df = df[df["sweep"] == "terms"].copy()
    ax = axes[2]
    for model_type, color in (("baseline", "#f4a261"), ("scop", "#2a9d8f")):
        ss = term_df[term_df["model_type"] == model_type].sort_values("n_scop")
        ax.plot(ss["n_scop"], ss["runtime_s"], marker="o", label=model_type, color=color)
    ax.set_title("Runtime vs number of monotone terms (n=100k)")
    ax.set_xlabel("n_scop")
    ax.set_ylabel("seconds")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle("discrete=True fit_reml scaling for monotone PSplines", y=1.02)
    fig.tight_layout()
    fig.savefig(SCALING_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_identifiability(df: pd.DataFrame) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    k_df = df[df["sweep"] == "k"].sort_values("k")
    ax = axes[0]
    ax.plot(
        k_df["k"],
        k_df["max_abs_native_mean_scop"],
        marker="o",
        color="#264653",
        label="max |native mean|",
    )
    ax.plot(
        k_df["k"],
        k_df["max_abs_weighted_mean_scop"],
        marker="o",
        color="#e76f51",
        label="max |weighted mean|",
    )
    ax.set_yscale("log")
    ax.set_title("SCOP centering vs basis dimension (n=100k, 2 terms)")
    ax.set_xlabel("k")
    ax.set_ylabel("absolute mean contribution")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    term_df = df[(df["sweep"] == "terms") & (df["model_type"] == "scop")].sort_values("n_scop")
    ax = axes[1]
    ax.plot(
        term_df["n_scop"],
        term_df["min_monotone_diff"],
        marker="o",
        color="#2a9d8f",
        label="min diff on grid",
    )
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("Monotonicity floor vs number of SCOP terms (n=100k)")
    ax.set_xlabel("n_scop")
    ax.set_ylabel("minimum first difference")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle("SCOP identifiability and monotonicity diagnostics", y=1.02)
    fig.tight_layout()
    fig.savefig(IDENT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    print("\nN sweep")
    print(
        df[df["sweep"] == "n"][
            ["model_type", "n_scop", "n", "runtime_s", "outer_iters", "inner_avg", "converged"]
        ].to_string(index=False)
    )

    print("\nTerm-count sweep")
    print(
        df[df["sweep"] == "terms"][
            ["model_type", "n_scop", "runtime_s", "outer_iters", "inner_avg", "converged"]
        ].to_string(index=False)
    )

    print("\nBasis-dimension sweep (SCOP)")
    print(
        df[df["sweep"] == "k"][
            [
                "k",
                "runtime_s",
                "outer_iters",
                "inner_avg",
                "max_abs_native_mean_scop",
                "max_abs_weighted_mean_scop",
                "min_monotone_diff",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reml-iter", type=int, default=20)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = run_suite(max_reml_iter=args.max_reml_iter)
    plot_scaling(df)
    plot_identifiability(df)
    print_summary(df)
    print(f"\nWrote {CSV_PATH}")
    print(f"Wrote {SCALING_PNG}")
    print(f"Wrote {IDENT_PNG}")


if __name__ == "__main__":
    main()
