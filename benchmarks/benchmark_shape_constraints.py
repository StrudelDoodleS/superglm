"""Benchmark dense and discrete fit_reml() for shape-constrained spline paths.

Compares:
- unconstrained PSpline
- monotone PSpline (SCOP)
- convex PSpline (SCOP)
- convex BSplineSmooth (QP)

Usage:
    uv run python benchmarks/benchmark_shape_constraints.py
    uv run python benchmarks/benchmark_shape_constraints.py --n 2000 --reps 1

Outputs:
    benchmarks/results/benchmark_shape_constraints.csv
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import BSplineSmooth, Constraint, PSpline, SuperGLM

RESULTS_DIR = Path("benchmarks/results")
CSV_PATH = RESULTS_DIR / "benchmark_shape_constraints.csv"

CONFIG_ENGINES = {
    "ps_unconstrained": "baseline",
    "ps_monotone": "scop",
    "ps_convex": "scop",
    "bs_convex": "qp",
}


@dataclass
class RunRow:
    config: str
    engine: str
    mode: str
    rep: int
    n: int
    runtime_s: float
    converged: bool
    n_reml_iter: int
    n_pirls_iter: int
    lambda_x: float
    min_first_derivative: float
    min_second_derivative: float


def make_dataset(n: int, seed: int, noise_sd: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a one-feature Gaussian dataset compatible with all benchmarked constraints."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    eta = 0.35 + 0.65 * x + 1.75 * x**2
    y = eta + rng.normal(0.0, noise_sd, size=n)
    return pd.DataFrame({"x": x}), y.astype(np.float64)


def make_feature(config: str):
    """Construct the feature spec for a named benchmark configuration."""
    if config == "ps_unconstrained":
        return PSpline(n_knots=10)
    if config == "ps_monotone":
        return PSpline(n_knots=10, constraint=Constraint.fit.increasing)
    if config == "ps_convex":
        return PSpline(n_knots=10, constraint=Constraint.fit.convex)
    if config == "bs_convex":
        return BSplineSmooth(n_knots=10, constraint=Constraint.fit.convex)
    raise ValueError(f"Unknown config: {config!r}")


def curvature_diagnostics(model: SuperGLM, feature_name: str) -> tuple[float, float]:
    """Approximate first- and second-derivative floors on a regular prediction grid."""
    spec = model._specs[feature_name]
    grid = np.linspace(spec._lo, spec._hi, 300)
    pred = np.asarray(model.predict(pd.DataFrame({feature_name: grid})), dtype=np.float64)
    first = np.gradient(pred, grid)
    second = np.gradient(first, grid)
    return float(first.min()), float(second.min())


def run_case(
    *,
    config: str,
    discrete: bool,
    rep: int,
    df: pd.DataFrame,
    y: np.ndarray,
    max_reml_iter: int,
) -> RunRow:
    """Fit a single model and capture runtime plus simple shape diagnostics."""
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={"x": make_feature(config)},
    )

    t0 = time.perf_counter()
    model.fit_reml(df, y, max_reml_iter=max_reml_iter)
    runtime_s = time.perf_counter() - t0

    min_first_derivative, min_second_derivative = curvature_diagnostics(model, "x")

    return RunRow(
        config=config,
        engine=CONFIG_ENGINES[config],
        mode="discrete" if discrete else "dense",
        rep=rep,
        n=len(df),
        runtime_s=runtime_s,
        converged=bool(model._result.converged),
        n_reml_iter=int(model._reml_result.n_reml_iter),
        n_pirls_iter=int(model._result.n_iter),
        lambda_x=float(model._reml_lambdas["x"]),
        min_first_derivative=min_first_derivative,
        min_second_derivative=min_second_derivative,
    )


def run_suite(n: int, reps: int, seed: int, noise_sd: float, max_reml_iter: int) -> pd.DataFrame:
    """Run the full dense/discrete comparison suite and persist raw rows."""
    rows: list[RunRow] = []

    for rep in range(reps):
        df, y = make_dataset(n=n, seed=seed + rep, noise_sd=noise_sd)
        for discrete in (False, True):
            for config in CONFIG_ENGINES:
                rows.append(
                    run_case(
                        config=config,
                        discrete=discrete,
                        rep=rep,
                        df=df,
                        y=y,
                        max_reml_iter=max_reml_iter,
                    )
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(asdict(row) for row in rows)
    out.to_csv(CSV_PATH, index=False)
    return out


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact median-by-configuration summary."""
    summary = (
        df.groupby(["mode", "config", "engine"], as_index=False)
        .agg(
            runtime_s=("runtime_s", "median"),
            n_reml_iter=("n_reml_iter", "median"),
            n_pirls_iter=("n_pirls_iter", "median"),
            lambda_x=("lambda_x", "median"),
            min_first_derivative=("min_first_derivative", "median"),
            min_second_derivative=("min_second_derivative", "median"),
            converged=("converged", "all"),
        )
        .sort_values(["mode", "config"])
    )
    print("Shape constraint benchmark: median runtime over reps")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=10_000, help="Number of rows in the synthetic data."
    )
    parser.add_argument("--reps", type=int, default=3, help="Benchmark repetitions per config.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--noise-sd", type=float, default=0.05, help="Gaussian noise SD.")
    parser.add_argument(
        "--max-reml-iter",
        type=int,
        default=20,
        help="Maximum REML outer iterations per fit.",
    )
    args = parser.parse_args()

    df = run_suite(
        n=args.n,
        reps=args.reps,
        seed=args.seed,
        noise_sd=args.noise_sd,
        max_reml_iter=args.max_reml_iter,
    )
    print_summary(df)
    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
