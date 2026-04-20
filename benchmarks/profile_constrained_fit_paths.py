"""Profile constrained fit paths across representative SCOP and QP scenarios.

Usage:
    PYTHONPATH=src uv run python benchmarks/profile_constrained_fit_paths.py
    PYTHONPATH=src uv run python benchmarks/profile_constrained_fit_paths.py --max-n 10000 --reps 1

Outputs:
    benchmarks/results/constrained_fit_paths_summary.csv
    benchmarks/results/constrained_fit_paths/*.txt
    benchmarks/results/constrained_fit_paths/*.json
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from benchmarks._constrained_fit_profile import (
        ProfileScenario,
        build_scenarios,
        make_synthetic_dataset,
        profile_callstack_and_memory,
        summarize_rows,
        write_profile_artifacts,
    )
except ModuleNotFoundError:
    from _constrained_fit_profile import (
        ProfileScenario,
        build_scenarios,
        make_synthetic_dataset,
        profile_callstack_and_memory,
        summarize_rows,
        write_profile_artifacts,
    )
from superglm import BSplineSmooth, Constraint, PSpline, SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import CubicRegressionSpline

RESULTS_DIR = Path("benchmarks/results")
SUMMARY_CSV = RESULTS_DIR / "constrained_fit_paths_summary.csv"
ARTIFACT_DIR = RESULTS_DIR / "constrained_fit_paths"


def scenario_mode(scenario: ProfileScenario) -> str:
    return "discrete" if scenario.discrete else "exact"


def scenario_stem(scenario: ProfileScenario) -> str:
    return f"{scenario.name}_{scenario_mode(scenario)}_n{scenario.n}"


def _constraint_for_index(index: int):
    return Constraint.fit.convex if index % 2 == 0 else Constraint.fit.concave


def _constrained_feature(engine: str, n_knots: int, index: int):
    constraint = _constraint_for_index(index)
    if engine == "scop":
        return PSpline(n_knots=n_knots, constraint=constraint)
    if engine == "qp":
        return BSplineSmooth(n_knots=n_knots, constraint=constraint)
    raise ValueError(f"Unsupported engine: {engine!r}")


def build_features(scenario: ProfileScenario) -> dict[str, object]:
    if scenario.use_fremtpl:
        feature_names = ["BonusMalus", "LogDensity"][: scenario.n_constrained]
        features: dict[str, object] = {
            name: _constrained_feature(scenario.engine, scenario.k, index)
            for index, name in enumerate(feature_names)
        }
        features["DrivAge"] = CubicRegressionSpline(n_knots=12)
        features["VehAge"] = CubicRegressionSpline(n_knots=10)
        features["Area"] = Categorical(base="most_exposed")
        return features

    return {
        f"x{index + 1}": _constrained_feature(scenario.engine, scenario.k, index)
        for index in range(scenario.n_constrained)
    }


def _find_fremtpl_path() -> Path | None:
    target = Path("data/freMTPL2freq.parquet")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / target
        if candidate.exists():
            return candidate
    return None


def load_dataset(
    scenario: ProfileScenario,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not scenario.use_fremtpl:
        return make_synthetic_dataset(scenario, seed=seed)

    path = _find_fremtpl_path()
    if path is None:
        raise FileNotFoundError("freMTPL2freq.parquet not found in this checkout or its parents")

    df = pd.read_parquet(path)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    df["LogDensity"] = np.log1p(df["Density"])
    if scenario.n < len(df):
        df = df.head(scenario.n).copy()
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy(dtype=np.float64)
    exposure = df["Exposure"].to_numpy(dtype=np.float64)
    X = df[["BonusMalus", "LogDensity", "DrivAge", "VehAge", "Area"]].copy()
    return X, y, exposure


def build_model(scenario: ProfileScenario) -> SuperGLM:
    return SuperGLM(
        family="poisson" if scenario.use_fremtpl else "gaussian",
        selection_penalty=0.0,
        discrete=scenario.discrete,
        features=build_features(scenario),
    )


def fit_model(
    scenario: ProfileScenario,
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    max_reml_iter: int,
) -> SuperGLM:
    model = build_model(scenario)
    if scenario.use_fremtpl:
        return model.fit_reml(X, y, exposure=weight, max_reml_iter=max_reml_iter)
    return model.fit_reml(X, y, sample_weight=weight, max_reml_iter=max_reml_iter)


def time_fit(
    scenario: ProfileScenario,
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    max_reml_iter: int,
) -> tuple[SuperGLM, float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    model = fit_model(scenario, X, y, weight, max_reml_iter=max_reml_iter)
    runtime_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return model, runtime_s, peak / (1024 * 1024)


def summarize_scenario(
    scenario: ProfileScenario,
    run_rows: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    return {
        "scenario": scenario.name,
        "engine": scenario.engine,
        "mode": scenario_mode(scenario),
        "n": scenario.n,
        "k": scenario.k,
        "n_constrained": scenario.n_constrained,
        "runtime_s": float(np.median([float(row["runtime_s"]) for row in run_rows])),
        "n_reml_iter": int(np.median([int(row["n_reml_iter"]) for row in run_rows])),
        "n_pirls_iter": int(np.median([int(row["n_pirls_iter"]) for row in run_rows])),
        "peak_mem_mb": float(np.median([float(row["peak_mem_mb"]) for row in run_rows])),
    }


def select_representatives(
    scenarios: list[ProfileScenario],
    *,
    fremtpl_available: bool,
) -> set[str]:
    chosen: dict[tuple[str, bool, bool], ProfileScenario] = {}
    for scenario in sorted(
        (s for s in scenarios if fremtpl_available or not s.use_fremtpl),
        key=lambda s: (s.n, s.discrete, s.n_constrained),
        reverse=True,
    ):
        key = (scenario.engine, scenario.discrete, scenario.n_constrained > 1)
        chosen.setdefault(key, scenario)
    return {scenario.name for scenario in chosen.values()}


def capture_artifacts(
    scenario: ProfileScenario,
    X: pd.DataFrame,
    y: np.ndarray,
    weight: np.ndarray,
    max_reml_iter: int,
    artifact_dir: Path,
) -> dict[str, Path]:
    _, cpu_stats, memory_stats = profile_callstack_and_memory(
        lambda: fit_model(scenario, X, y, weight, max_reml_iter=max_reml_iter)
    )
    return write_profile_artifacts(
        base_dir=artifact_dir,
        stem=scenario_stem(scenario),
        profile_stats={
            "scenario": scenario.name,
            "mode": scenario_mode(scenario),
            "stats": cpu_stats,
        },
        memory_stats={"scenario": scenario.name, "mode": scenario_mode(scenario), **memory_stats},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=500_000, help="Largest scenario row count.")
    parser.add_argument("--reps", type=int, default=2, help="Timing repetitions per scenario.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for synthetic data.")
    parser.add_argument(
        "--max-reml-iter",
        type=int,
        default=20,
        help="Maximum REML outer iterations for each fit.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory used for summary CSV and profiling artifacts.",
    )
    args = parser.parse_args()

    summary_csv = args.results_dir / SUMMARY_CSV.name
    artifact_dir = args.results_dir / ARTIFACT_DIR.name
    args.results_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios(max_n=args.max_n)
    fremtpl_available = _find_fremtpl_path() is not None
    representatives = select_representatives(
        scenarios,
        fremtpl_available=fremtpl_available,
    )

    print(f"profiling {len(scenarios)} scenarios")
    if not fremtpl_available and any(s.use_fremtpl for s in scenarios):
        print("freMTPL2 data not found; freMTPL2-backed scenarios will be skipped")

    summary_rows_raw: list[dict[str, float | int | str]] = []

    for index, scenario in enumerate(scenarios):
        label = (
            f"{scenario.name} "
            f"(engine={scenario.engine}, mode={scenario_mode(scenario)}, "
            f"n={scenario.n:,}, k={scenario.k}, n_constrained={scenario.n_constrained})"
        )
        print(f"[{index + 1}/{len(scenarios)}] running {label}")
        try:
            X, y, weight = load_dataset(scenario, seed=args.seed + index)
        except FileNotFoundError as err:
            print(f"skipping {scenario.name}: {err}")
            continue

        run_rows: list[dict[str, float | int | str]] = []
        for _ in range(args.reps):
            model, runtime_s, peak_mem_mb = time_fit(
                scenario,
                X,
                y,
                weight,
                max_reml_iter=args.max_reml_iter,
            )
            run_rows.append(
                {
                    "runtime_s": runtime_s,
                    "n_reml_iter": int(model._reml_result.n_reml_iter),
                    "n_pirls_iter": int(model._result.n_iter),
                    "peak_mem_mb": peak_mem_mb,
                }
            )

        summary_row = summarize_scenario(scenario, run_rows)
        summary_rows_raw.append(summary_row)
        print(
            f"done in {summary_row['runtime_s']:.3f}s, peak_mem={summary_row['peak_mem_mb']:.2f}MB"
        )

        if scenario.name in representatives:
            artifact_paths = capture_artifacts(
                scenario,
                X,
                y,
                weight,
                max_reml_iter=args.max_reml_iter,
                artifact_dir=artifact_dir,
            )
            for path in artifact_paths.values():
                print(f"wrote {path}")

    if summary_rows_raw:
        summary_df = summarize_rows(summary_rows_raw)
    else:
        summary_df = pd.DataFrame(
            columns=[
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
        )
    summary_df.to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
