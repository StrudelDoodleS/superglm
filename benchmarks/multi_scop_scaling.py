"""Core helpers for the synthetic multi-SCOP scaling benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
