"""Synthetic multi-SCOP scaling benchmark."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import Constraint, PSpline, SuperGLM

RESULTS_DIR = Path("benchmarks/results")
CSV_PATH = RESULTS_DIR / "multi_scop_scaling.csv"

RUN_ROW_COLUMNS = (
    "mode",
    "n_constrained",
    "n",
    "k",
    "runtime_s",
    "peak_mem_mb",
    "converged",
    "n_reml_iter",
    "n_pirls_iter",
    "n_floor",
    "n_active",
)


@dataclass(frozen=True)
class MultiSCOPScenario:
    mode: str
    n_constrained: int
    n: int
    k: int


@dataclass(frozen=True)
class RunRow:
    mode: str
    n_constrained: int
    n: int
    k: int
    runtime_s: float
    peak_mem_mb: float
    converged: bool
    n_reml_iter: int
    n_pirls_iter: int
    n_floor: int
    n_active: int


def build_scenarios() -> list[MultiSCOPScenario]:
    scenarios = []
    for n_constrained in (1, 2, 4, 8, 16):
        scenarios.append(MultiSCOPScenario("discrete", n_constrained, 100_000, 12))
    for n_constrained in (1, 2, 4):
        scenarios.append(MultiSCOPScenario("exact", n_constrained, 100_000, 12))
    return scenarios


def summarize_lambda_activity(
    lambdas: dict[str, float], *, floor: float, active_threshold: float
) -> dict[str, int]:
    n_floor = sum(float(value) <= floor * 1.000001 for value in lambdas.values())
    n_active = sum(float(value) > active_threshold for value in lambdas.values())
    return {"n_floor": n_floor, "n_active": n_active}


def make_dataset(n: int, n_constrained: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    eta = np.full(n, -0.2, dtype=np.float64)

    for j in range(n_constrained):
        support = max(20, n // 50)
        if n > 1:
            support = min(support, n - 1)
        else:
            support = 1
        x = np.repeat(np.linspace(0.0, 1.0, support, dtype=np.float64), n // support + 1)[:n]
        rng.shuffle(x)
        data[f"x{j + 1}"] = x
        eta += 0.15 * x + 0.35 * x**2

    y = eta + rng.normal(0.0, 0.05, size=n)
    return pd.DataFrame(data), y.astype(np.float64)


def summarize_rows(rows: list[RunRow | dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(RUN_ROW_COLUMNS))

    normalized_rows = [asdict(row) if isinstance(row, RunRow) else dict(row) for row in rows]
    return pd.DataFrame(normalized_rows)[list(RUN_ROW_COLUMNS)]


def make_features(n_constrained: int, k: int) -> dict[str, object]:
    features: dict[str, object] = {}
    for j in range(n_constrained):
        constraint = Constraint.fit.convex if j % 2 == 0 else Constraint.fit.concave
        features[f"x{j + 1}"] = PSpline(n_knots=k, constraint=constraint)
    return features


def fit_scenario(scenario: MultiSCOPScenario, *, seed: int) -> RunRow:
    X, y = make_dataset(scenario.n, scenario.n_constrained, seed=seed)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=(scenario.mode == "discrete"),
        features=make_features(scenario.n_constrained, scenario.k),
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit_reml(X, y, max_reml_iter=20)
    runtime_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lambda_stats = summarize_lambda_activity(
        {k: float(v) for k, v in model._reml_lambdas.items()},
        floor=1e-4,
        active_threshold=1e-3,
    )

    return RunRow(
        mode=scenario.mode,
        n_constrained=scenario.n_constrained,
        n=scenario.n,
        k=scenario.k,
        runtime_s=float(runtime_s),
        peak_mem_mb=float(peak / (1024 * 1024)),
        converged=bool(model._reml_result.converged),
        n_reml_iter=int(model._reml_result.n_reml_iter),
        n_pirls_iter=int(model._result.n_iter),
        n_floor=lambda_stats["n_floor"],
        n_active=lambda_stats["n_active"],
    )


def plot_scaling(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    metrics = [
        ("runtime_s", "Runtime (s)", "multi_scop_runtime.png"),
        ("peak_mem_mb", "Peak memory (MB)", "multi_scop_memory.png"),
        ("n_reml_iter", "REML iterations", "multi_scop_reml_iters.png"),
        ("n_active", "Active lambdas", "multi_scop_active_lambdas.png"),
    ]

    for metric, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for mode, color in (("discrete", "#2a9d8f"), ("exact", "#e76f51")):
            sub = df[df["mode"] == mode]
            if sub.empty:
                continue
            ax.plot(sub["n_constrained"], sub[metric], marker="o", label=mode, color=color)
        ax.set_xlabel("n_constrained")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        out = output_dir / filename
        fig.tight_layout()
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-constrained", type=int, default=16)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = [
        s
        for s in build_scenarios()
        if s.n_constrained <= args.max_constrained
        and (s.mode == "discrete" or s.n_constrained <= 4)
    ]

    rows = []
    for idx, scenario in enumerate(scenarios, start=1):
        scenario = MultiSCOPScenario(
            mode=scenario.mode,
            n_constrained=scenario.n_constrained,
            n=args.n,
            k=args.k,
        )
        print(
            f"[{idx}/{len(scenarios)}] running mode={scenario.mode} "
            f"n_constrained={scenario.n_constrained} n={scenario.n} k={scenario.k}"
        )
        row = fit_scenario(scenario, seed=args.seed + idx)
        rows.append(row)
        print(
            f"  done in {row.runtime_s:.3f}s, peak_mem={row.peak_mem_mb:.2f}MB, "
            f"reml_iter={row.n_reml_iter}, active={row.n_active}, floor={row.n_floor}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    summary.to_csv(CSV_PATH, index=False)
    print(f"wrote {CSV_PATH}")
    for path in plot_scaling(summary, RESULTS_DIR):
        print(f"wrote {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
