"""REML benchmark harness — SuperGLM side.

Runs SuperGLM fit_reml() across configurations and datasets, exports
structured JSON results + shared CSV data for the R companion script.

Configurations:
  - fit_reml(discrete=False)
  - fit_reml(discrete=True)
  - Poisson and Gamma families
  - Small synthetic (800 rows) and large MTPL2 (678k rows)

Usage:
    uv run python benchmarks/reml_benchmark_harness.py [--no-mtpl2]

Outputs:
    benchmarks/results/superglm_results.json
    benchmarks/results/bench_synthetic_poisson.csv
    benchmarks/results/bench_synthetic_gamma.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from superglm.features.categorical import Categorical
from superglm.features.spline import CubicRegressionSpline
from superglm.model import SuperGLM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Synthetic data generators ────────────────────────────────────


def make_synthetic_poisson(n=800, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = 0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2}), y, np.ones(n)


def make_synthetic_gamma(n=800, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    eta = 1.0 + np.sin(2 * np.pi * x1) + 0.5 * x2
    mu = np.exp(eta)
    shape = 5.0
    y = rng.gamma(shape=shape, scale=mu / shape)
    y = np.maximum(y, 1e-4)
    return pd.DataFrame({"x1": x1, "x2": x2}), y, np.ones(n)


def load_mtpl2():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "freMTPL2freq.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    df["LogDensity"] = np.log1p(df["Density"])
    y_freq = (df["ClaimNb"] / df["Exposure"]).values
    exposure = df["Exposure"].values
    return df, y_freq, exposure


# ── Benchmark runner ─────────────────────────────────────────────


def run_one(
    name: str,
    df: pd.DataFrame,
    y: np.ndarray,
    exposure: np.ndarray,
    features: dict,
    family: str,
    discrete: bool,
    max_reml_iter: int = 30,
    n_reps: int = 1,
) -> dict:
    """Run a single benchmark configuration, return structured result."""
    times = []
    result_info = None

    for rep in range(n_reps):
        model = SuperGLM(
            family=family,
            lambda1=0,
            features=features,
            discrete=discrete,
        )
        t0 = time.perf_counter()
        model.fit_reml(df, y, exposure=exposure, max_reml_iter=max_reml_iter)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if rep == 0:
            r = model._reml_result
            result_info = {
                "deviance": float(model.result.deviance),
                "effective_df": float(model.result.effective_df),
                "phi": float(model.result.phi),
                "n_reml_iter": r.n_reml_iter,
                "converged": r.converged,
                "n_pirls_iter": model.result.n_iter,
                "lambdas": {k: float(v) for k, v in model._reml_lambdas.items()},
            }
            # Capture phase-level profile if available
            profile = getattr(model, "_reml_profile", None)
            if profile:
                result_info["profile"] = {
                    k: round(v, 4) if isinstance(v, float) else v for k, v in profile.items()
                }

    return {
        "name": name,
        "n": len(df),
        "family": family,
        "discrete": discrete,
        "wall_time_s": float(np.median(times)),
        "wall_times": [float(t) for t in times],
        **result_info,
    }


# ── Benchmark cases ──────────────────────────────────────────────


def run_synthetic_benchmarks():
    """Run small synthetic benchmarks (Poisson + Gamma, exact + discrete)."""
    results = []

    for family, make_data in [
        ("poisson", make_synthetic_poisson),
        ("gamma", make_synthetic_gamma),
    ]:
        df, y, w = make_data()

        # Export CSV for R companion
        export_df = df.copy()
        export_df["y"] = y
        export_df["w"] = w
        csv_path = os.path.join(RESULTS_DIR, f"bench_synthetic_{family}.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"  Exported {csv_path}")

        for discrete in [False, True]:
            label = f"synthetic_{family}_{'disc' if discrete else 'exact'}"
            print(f"  Running {label} (n={len(df)})...", end=" ", flush=True)
            r = run_one(
                name=label,
                df=df,
                y=y,
                exposure=w,
                features={
                    "x1": CubicRegressionSpline(n_knots=8),
                    "x2": CubicRegressionSpline(n_knots=8),
                },
                family=family,
                discrete=discrete,
                n_reps=3,
            )
            print(
                f"{r['wall_time_s']:.2f}s, dev={r['deviance']:.1f}, "
                f"edf={r['effective_df']:.2f}, reml_iter={r['n_reml_iter']}"
            )
            results.append(r)

    return results


def run_mtpl2_benchmarks():
    """Run MTPL2 benchmarks (Poisson, exact + discrete)."""
    data = load_mtpl2()
    if data is None:
        print("  MTPL2 data not found, skipping")
        return []

    df, y_freq, exposure = data
    n = len(df)
    print(f"  MTPL2 loaded: {n:,} rows")

    # Export prepared CSV for R companion (if not already present)
    r_csv = os.path.join(RESULTS_DIR, "bench_mtpl2.csv")
    if not os.path.exists(r_csv):
        export_df = df[["DrivAge", "VehAge", "BonusMalus", "LogDensity", "Area"]].copy()
        export_df["y_freq"] = y_freq
        export_df["Exposure"] = exposure
        export_df.to_csv(r_csv, index=False)
        print(f"  Exported {r_csv}")

    # Basis size matching with mgcv:
    # mgcv bs="cr" k=K gives K basis functions (K-1 after identifiability).
    # SuperGLM CRS n_knots=N gives N+4 B-splines, N+2 after natural constraints
    # (intercept handled separately). To match mgcv k=20: n_knots=18 → 20 cols.
    # k=15: n_knots=13 → 15 cols.
    results = []
    for discrete in [False, True]:
        label = f"mtpl2_poisson_{'disc' if discrete else 'exact'}"
        print(f"  Running {label} (n={n:,})...", end=" ", flush=True)
        r = run_one(
            name=label,
            df=df,
            y=y_freq,
            exposure=exposure,
            features={
                "DrivAge": CubicRegressionSpline(n_knots=18),  # matches mgcv k=20
                "VehAge": CubicRegressionSpline(n_knots=13),  # matches mgcv k=15
                "BonusMalus": CubicRegressionSpline(n_knots=13),  # matches mgcv k=15
                "Area": Categorical(base="most_exposed"),
            },
            family="poisson",
            discrete=discrete,
            n_reps=1,  # single run for large data
        )
        print(
            f"{r['wall_time_s']:.2f}s, dev={r['deviance']:.1f}, "
            f"edf={r['effective_df']:.2f}, reml_iter={r['n_reml_iter']}"
        )
        results.append(r)

    return results


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="REML benchmark harness (SuperGLM)")
    parser.add_argument("--no-mtpl2", action="store_true", help="Skip MTPL2 benchmarks")
    args = parser.parse_args()

    _ensure_results_dir()

    print("=" * 60)
    print("SuperGLM REML Benchmark Harness")
    print("=" * 60)

    all_results = []

    print("\n── Synthetic benchmarks ──")
    all_results.extend(run_synthetic_benchmarks())

    if not args.no_mtpl2:
        print("\n── MTPL2 benchmarks ──")
        all_results.extend(run_mtpl2_benchmarks())

    # Print phase-level profiles
    profiled = [r for r in all_results if "profile" in r]
    if profiled:
        print("\n── Phase-level profiles ──")
        for r in profiled:
            p = r["profile"]
            total = p.get("total_s", r["wall_time_s"])
            print(f"\n  {r['name']} (total={total:.2f}s)")
            print(f"    {'Phase':<30s} {'Time':>8s} {'%':>6s}")
            print(f"    {'-' * 48}")

            phases = [
                ("DM build", p.get("dm_build_s", 0)),
                ("IRLS total", p.get("irls_total_s", 0)),
                ("  working quantities", p.get("irls_working_s", 0)),
                ("  gram assembly (X'WX)", p.get("irls_gram_s", 0)),
                ("  solve (eigh)", p.get("irls_solve_s", 0)),
                ("  deviance check", p.get("irls_deviance_s", 0)),
                ("  finalize (inv+edf)", p.get("irls_finalize_s", 0)),
                ("REML objective", p.get("reml_objective_s", 0)),
                ("REML gradient", p.get("reml_gradient_s", 0)),
                ("REML W(rho) correction", p.get("reml_w_correction_s", 0)),
                ("REML Hessian + Newton", p.get("reml_hessian_newton_s", 0)),
                ("REML line search", p.get("reml_linesearch_s", 0)),
                ("REML FP update", p.get("reml_fp_update_s", 0)),
            ]
            for label, t in phases:
                pct = 100 * t / total if total > 0 else 0
                print(f"    {label:<30s} {t:>7.2f}s {pct:>5.1f}%")

            # Counters
            print(f"    {'─' * 48}")
            print(f"    REML outer iters:    {p.get('reml_n_outer_iter', '?')}")
            print(f"    IRLS calls:          {p.get('irls_calls', '?')}")
            print(f"    Total IRLS iters:    {p.get('irls_iters', '?')}")
            print(f"    Line search fits:    {p.get('reml_n_linesearch_fits', '?')}")

    # Write results
    out_path = os.path.join(RESULTS_DIR, "superglm_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "tool": "superglm",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python_version": sys.version.split()[0],
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
