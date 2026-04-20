from __future__ import annotations

import cProfile
import io
import json
import pstats
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfileScenario:
    name: str
    engine: str
    n: int
    k: int
    n_constrained: int
    repeated_support: bool
    discrete: bool
    use_fremtpl: bool


def make_synthetic_dataset(
    scenario: ProfileScenario, seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for j in range(scenario.n_constrained):
        if scenario.repeated_support:
            support = max(10, scenario.n // 20)
            x = np.repeat(np.linspace(0.0, 1.0, support), scenario.n // support + 1)[: scenario.n]
        else:
            x = np.linspace(0.0, 1.0, scenario.n)
        data[f"x{j + 1}"] = x
    X = pd.DataFrame(data)
    y = 0.4 + 0.7 * X.iloc[:, 0].to_numpy() + 1.2 * X.iloc[:, 0].to_numpy() ** 2
    w = np.ones(len(X), dtype=float)
    return X, y.astype(float), w


def summarize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        [
            "scenario",
            "engine",
            "mode",
            "n",
            "k",
            "n_constrained",
            "runtime_s",
            "n_reml_iter",
            "n_pirls_iter",
            "peak_mem_mb",
        ]
    ]


def write_profile_artifacts(
    base_dir: Path, stem: str, profile_stats: dict[str, Any], memory_stats: dict[str, Any]
) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    cpu_txt = base_dir / f"{stem}_cpu.txt"
    memory_json = base_dir / f"{stem}_memory.json"
    cpu_txt.write_text(str(profile_stats), encoding="utf-8")
    memory_json.write_text(json.dumps(memory_stats, indent=2), encoding="utf-8")
    return {"cpu_txt": cpu_txt, "memory_json": memory_json}


def profile_callstack_and_memory(fn: Callable[[], Any]) -> tuple[Any, str, dict[str, float]]:
    profiler = cProfile.Profile()
    tracemalloc.start()
    profiler.enable()
    result = fn()
    profiler.disable()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative").print_stats(30)
    return result, stats_stream.getvalue(), {"peak_mb": peak / (1024 * 1024)}
