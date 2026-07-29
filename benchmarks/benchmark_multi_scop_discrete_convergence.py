"""Targeted benchmark for the multi-SCOP discrete cleanup toggle.

For each dataset repeat, the harness runs both discrete REML execution orders:
- baseline then optimized
- optimized then baseline

The harness covers a synthetic multi-SCOP Poisson-rate dataset plus the full
freMTPL2 frequency dataset and writes a side-by-side summary CSV.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median, median_low
from unittest.mock import patch

import numpy as np
import pandas as pd

import superglm.reml.scop_efs as scop_efs
from superglm import Categorical, Constraint, CubicRegressionSpline, PSpline, SuperGLM

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path("benchmarks/results")
CSV_PATH = RESULTS_DIR / "multi_scop_discrete_convergence.csv"
ABS_RESULTS_DIR = REPO_ROOT / RESULTS_DIR
ABS_CSV_PATH = REPO_ROOT / CSV_PATH

DATA_PATH = REPO_ROOT / "data" / "freMTPL2freq.parquet"
if not DATA_PATH.exists() and REPO_ROOT.parent.name == ".worktrees":
    DATA_PATH = REPO_ROOT.parent.parent / "data" / "freMTPL2freq.parquet"

MAX_REML_ITER = 20
SYNTHETIC_N = 25_000


@dataclass(frozen=True)
class VariantResult:
    runtime_s: float
    n_reml_iter: int
    n_pirls_iter: int
    predictions: np.ndarray
    lambdas: dict[str, float]
    converged: bool
    cleanup_gate_calls: int
    cleanup_gate_true_count: int
    frozen_count: int
    freeze_iter: int


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    n_rows: int
    repeats: int
    execution_order: str
    baseline_runtime_s: float
    optimized_runtime_s: float
    speedup_x: float
    baseline_n_reml_iter: int
    optimized_n_reml_iter: int
    baseline_n_pirls_iter: int
    optimized_n_pirls_iter: int
    baseline_converged: bool
    optimized_converged: bool
    baseline_cleanup_gate_calls: int
    optimized_cleanup_gate_calls: int
    baseline_cleanup_gate_true_count: int
    optimized_cleanup_gate_true_count: int
    baseline_frozen_count: int
    optimized_frozen_count: int
    baseline_freeze_iter: int
    optimized_freeze_iter: int
    pred_rmse: float
    pred_max_abs_diff: float
    lambda_max_abs_diff: float
    lambda_keys_match: bool
    baseline_lambdas_json: str
    optimized_lambdas_json: str


def _make_model() -> SuperGLM:
    return SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "DrivAge": PSpline(
                n_knots=12,
                penalty="ssp",
                constraint=Constraint.fit.concave,
            ),
            "VehAge": CubicRegressionSpline(n_knots=10),
            "BonusMalus": PSpline(
                n_knots=12,
                penalty="ssp",
                constraint=Constraint.fit.concave,
            ),
            "LogDensity": CubicRegressionSpline(n_knots=10),
            "Area": Categorical(base="most_exposed"),
        },
    )


def _cleanup_disabled(*, discrete: bool, scop_term_count: int) -> bool:
    del discrete, scop_term_count
    return False


def _aggregate_inner_pirls_iter(model: SuperGLM) -> int:
    """Aggregate inner PIRLS work across the REML loop, not just the final refit."""
    return int(sum(model._reml_result.inner_iter_history or []))


def _median_float(values: list[float]) -> float:
    return float(median(values))


def _median_int(values: list[int]) -> int:
    return int(median_low(values))


def _execution_orders_for_repeat(
    repeat_index: int,
) -> tuple[tuple[str, str], tuple[str, str]]:
    paired_orders = (("baseline", "optimized"), ("optimized", "baseline"))
    return paired_orders if repeat_index % 2 == 0 else tuple(reversed(paired_orders))


def _format_execution_order_summary(
    execution_orders_by_repeat: list[tuple[tuple[str, str], tuple[str, str]]],
) -> str:
    return "|".join(
        "&".join("->".join(execution_order) for execution_order in repeat_orders)
        for repeat_orders in execution_orders_by_repeat
    )


def _aggregate_lambda_metrics(
    *,
    lambda_max_abs_diffs: list[float],
    lambda_keys_matches: list[bool],
) -> tuple[float, bool]:
    lambda_keys_match = all(lambda_keys_matches)
    if not lambda_keys_match:
        return float("nan"), False
    return _median_float(lambda_max_abs_diffs), True


def _representative_pair_index(
    baseline_results: list[VariantResult],
    optimized_results: list[VariantResult],
) -> int:
    baseline_runtime_median = _median_float([result.runtime_s for result in baseline_results])
    optimized_runtime_median = _median_float([result.runtime_s for result in optimized_results])

    return min(
        range(len(baseline_results)),
        key=lambda idx: (
            abs(baseline_results[idx].runtime_s - baseline_runtime_median)
            + abs(optimized_results[idx].runtime_s - optimized_runtime_median)
        ),
    )


def _fit_variant(
    *,
    cleanup_enabled: bool,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    max_reml_iter: int = MAX_REML_ITER,
) -> VariantResult:
    model = _make_model()
    gate_calls = 0
    gate_true_count = 0
    freeze_calls = 0
    frozen_names_seen: set[str] = set()
    freeze_iter = 0
    original_gate = scop_efs._multi_scop_discrete_cleanup_enabled
    original_freeze = scop_efs._freeze_multi_scop_discrete_lambdas

    def disabled_gate(*, discrete: bool, scop_term_count: int) -> bool:
        nonlocal gate_calls
        gate_calls += 1
        return _cleanup_disabled(discrete=discrete, scop_term_count=scop_term_count)

    def recorded_gate(*, discrete: bool, scop_term_count: int) -> bool:
        nonlocal gate_calls, gate_true_count
        gate_calls += 1
        enabled = original_gate(discrete=discrete, scop_term_count=scop_term_count)
        gate_true_count += int(enabled)
        return enabled

    def recorded_freeze(
        *,
        active_names: set[str],
        frozen_names: set[str],
        lambdas_new: dict[str, float],
        stable_counts: dict[str, int],
    ) -> tuple[set[str], set[str]]:
        nonlocal freeze_calls, freeze_iter
        freeze_calls += 1
        active_out, frozen_out = original_freeze(
            active_names=active_names,
            frozen_names=frozen_names,
            lambdas_new=lambdas_new,
            stable_counts=stable_counts,
        )
        newly_frozen = set(frozen_out) - set(frozen_names)
        if newly_frozen:
            frozen_names_seen.update(newly_frozen)
            if freeze_iter == 0:
                freeze_iter = freeze_calls
        return active_out, frozen_out

    gate_impl = recorded_gate if cleanup_enabled else disabled_gate

    with (
        patch.object(
            scop_efs,
            "_multi_scop_discrete_cleanup_enabled",
            new=gate_impl,
        ),
        patch.object(
            scop_efs,
            "_freeze_multi_scop_discrete_lambdas",
            new=recorded_freeze,
        ),
    ):
        t0 = time.perf_counter()
        model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=max_reml_iter)
        runtime_s = time.perf_counter() - t0

    if cleanup_enabled and gate_calls == 0:
        raise RuntimeError("optimized variant never consulted the cleanup gate")
    if cleanup_enabled and gate_true_count == 0:
        raise RuntimeError(
            "optimized variant consulted the cleanup gate but it never returned True"
        )

    return VariantResult(
        runtime_s=float(runtime_s),
        n_reml_iter=int(model._reml_result.n_reml_iter),
        n_pirls_iter=_aggregate_inner_pirls_iter(model),
        predictions=np.asarray(model.predict(X), dtype=np.float64),
        lambdas={name: float(value) for name, value in sorted(model._reml_lambdas.items())},
        converged=bool(model._reml_result.converged),
        cleanup_gate_calls=gate_calls,
        cleanup_gate_true_count=gate_true_count,
        frozen_count=len(frozen_names_seen),
        freeze_iter=freeze_iter,
    )


def _prediction_metrics(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    diff = np.asarray(other, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def _lambda_max_abs_diff(reference: dict[str, float], other: dict[str, float]) -> float:
    if set(reference) != set(other):
        return float("nan")
    return max(abs(other[name] - reference[name]) for name in reference) if reference else 0.0


def _make_synthetic_data(
    n: int = SYNTHETIC_N,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    driv_age = rng.uniform(18.0, 85.0, size=n)
    veh_age = rng.uniform(0.0, 20.0, size=n)
    bonus_malus = rng.uniform(50.0, 150.0, size=n)
    density = rng.uniform(1.0, 5000.0, size=n)
    area = rng.choice(["A", "B", "C", "D"], size=n, p=[0.45, 0.25, 0.2, 0.1])

    eta = (
        -2.3
        - 0.018 * (driv_age - 45.0) ** 2 / 25.0
        - 0.0015 * (bonus_malus - 90.0) ** 2 / 12.0
        + 0.02 * np.sin(veh_age / 3.0)
        + 0.08 * np.log(np.clip(density, 1.0, None))
        + np.where(area == "B", 0.10, 0.0)
        + np.where(area == "C", -0.08, 0.0)
        + np.where(area == "D", 0.04, 0.0)
    )
    exposure = rng.uniform(0.2, 1.5, size=n)
    claim_nb = rng.poisson(exposure * np.exp(eta))

    X = pd.DataFrame(
        {
            "DrivAge": driv_age,
            "VehAge": veh_age,
            "BonusMalus": bonus_malus,
            "LogDensity": np.log(np.clip(density, 1.0, None)),
            "Area": area,
        }
    )
    y = claim_nb.astype(np.float64) / exposure
    return X, y.astype(np.float64), exposure.astype(np.float64)


def _load_fremtpl2_freq() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"freMTPL2freq.parquet not found at {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH).copy()
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    # Match the branch's SCOP sensitivity freMTPL2 prep for this feature set.
    df["LogDensity"] = np.log(df["Density"].clip(lower=1.0))

    X = df[["DrivAge", "VehAge", "BonusMalus", "LogDensity", "Area"]].copy()
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy(dtype=np.float64)
    sample_weight = df["Exposure"].to_numpy(dtype=np.float64)
    return X, y, sample_weight


def _summarize_dataset(
    dataset: str,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    repeats: int,
) -> SummaryRow:
    print(f"\nRunning {dataset} ({len(X):,} rows)")
    baseline_results: list[VariantResult] = []
    optimized_results: list[VariantResult] = []
    pred_rmses: list[float] = []
    pred_max_abs_diffs: list[float] = []
    lambda_max_abs_diffs: list[float] = []
    lambda_keys_matches: list[bool] = []
    execution_orders_by_repeat: list[tuple[tuple[str, str], tuple[str, str]]] = []

    for repeat_index in range(repeats):
        repeat_orders = _execution_orders_for_repeat(repeat_index)
        execution_orders_by_repeat.append(repeat_orders)
        print(
            f"  repeat {repeat_index + 1}/{repeats} "
            f"orders={' & '.join('->'.join(order) for order in repeat_orders)}"
        )

        for execution_order in repeat_orders:
            by_name: dict[str, VariantResult] = {}
            for variant_name in execution_order:
                cleanup_enabled = variant_name == "optimized"
                by_name[variant_name] = _fit_variant(
                    cleanup_enabled=cleanup_enabled,
                    X=X,
                    y=y,
                    sample_weight=sample_weight,
                )

            baseline = by_name["baseline"]
            optimized = by_name["optimized"]
            baseline_results.append(baseline)
            optimized_results.append(optimized)

            pred_metrics = _prediction_metrics(baseline.predictions, optimized.predictions)
            pred_rmses.append(pred_metrics["rmse"])
            pred_max_abs_diffs.append(pred_metrics["max_abs_diff"])
            lambda_keys_matches.append(set(baseline.lambdas) == set(optimized.lambdas))
            lambda_max_abs_diffs.append(_lambda_max_abs_diff(baseline.lambdas, optimized.lambdas))

            speedup_x = (
                float(baseline.runtime_s / optimized.runtime_s)
                if optimized.runtime_s > 0.0
                else float("nan")
            )
            print(
                "    "
                f"order={'->'.join(execution_order)} "
                f"baseline={baseline.runtime_s:.3f}s "
                f"optimized={optimized.runtime_s:.3f}s "
                f"speedup={speedup_x:.3f}x "
                f"baseline_gate_calls={baseline.cleanup_gate_calls} "
                f"optimized_gate_calls={optimized.cleanup_gate_calls} "
                f"optimized_gate_true={optimized.cleanup_gate_true_count} "
                f"optimized_frozen={optimized.frozen_count} "
                f"optimized_freeze_iter={optimized.freeze_iter} "
                f"pred_rmse={pred_metrics['rmse']:.3e} "
                f"pred_max_abs={pred_metrics['max_abs_diff']:.3e}"
            )

    representative_idx = _representative_pair_index(baseline_results, optimized_results)
    representative_baseline = baseline_results[representative_idx]
    representative_optimized = optimized_results[representative_idx]
    baseline_runtime_s = _median_float([result.runtime_s for result in baseline_results])
    optimized_runtime_s = _median_float([result.runtime_s for result in optimized_results])
    lambda_max_abs_diff, lambda_keys_match = _aggregate_lambda_metrics(
        lambda_max_abs_diffs=lambda_max_abs_diffs,
        lambda_keys_matches=lambda_keys_matches,
    )
    speedup_x = (
        float(baseline_runtime_s / optimized_runtime_s)
        if optimized_runtime_s > 0.0
        else float("nan")
    )
    print(
        "  "
        f"median_baseline={baseline_runtime_s:.3f}s "
        f"median_optimized={optimized_runtime_s:.3f}s "
        f"orders={_format_execution_order_summary(execution_orders_by_repeat)} "
        f"speedup={speedup_x:.3f}x "
        f"median_optimized_gate_calls={_median_int([r.cleanup_gate_calls for r in optimized_results])} "
        f"median_optimized_gate_true={_median_int([r.cleanup_gate_true_count for r in optimized_results])} "
        f"median_optimized_frozen={_median_int([r.frozen_count for r in optimized_results])} "
        f"median_optimized_freeze_iter={_median_int([r.freeze_iter for r in optimized_results])} "
        f"lambda_keys_match={lambda_keys_match} "
        f"median_pred_rmse={_median_float(pred_rmses):.3e} "
        f"median_pred_max_abs={_median_float(pred_max_abs_diffs):.3e}"
    )

    return SummaryRow(
        dataset=dataset,
        n_rows=int(len(X)),
        repeats=repeats,
        execution_order=_format_execution_order_summary(execution_orders_by_repeat),
        baseline_runtime_s=baseline_runtime_s,
        optimized_runtime_s=optimized_runtime_s,
        speedup_x=speedup_x,
        baseline_n_reml_iter=_median_int([result.n_reml_iter for result in baseline_results]),
        optimized_n_reml_iter=_median_int([result.n_reml_iter for result in optimized_results]),
        baseline_n_pirls_iter=_median_int([result.n_pirls_iter for result in baseline_results]),
        optimized_n_pirls_iter=_median_int([result.n_pirls_iter for result in optimized_results]),
        baseline_converged=all(result.converged for result in baseline_results),
        optimized_converged=all(result.converged for result in optimized_results),
        baseline_cleanup_gate_calls=_median_int(
            [result.cleanup_gate_calls for result in baseline_results]
        ),
        optimized_cleanup_gate_calls=_median_int(
            [result.cleanup_gate_calls for result in optimized_results]
        ),
        baseline_cleanup_gate_true_count=_median_int(
            [result.cleanup_gate_true_count for result in baseline_results]
        ),
        optimized_cleanup_gate_true_count=_median_int(
            [result.cleanup_gate_true_count for result in optimized_results]
        ),
        baseline_frozen_count=_median_int([result.frozen_count for result in baseline_results]),
        optimized_frozen_count=_median_int([result.frozen_count for result in optimized_results]),
        baseline_freeze_iter=_median_int([result.freeze_iter for result in baseline_results]),
        optimized_freeze_iter=_median_int([result.freeze_iter for result in optimized_results]),
        pred_rmse=_median_float(pred_rmses),
        pred_max_abs_diff=_median_float(pred_max_abs_diffs),
        lambda_max_abs_diff=lambda_max_abs_diff,
        lambda_keys_match=lambda_keys_match,
        baseline_lambdas_json=json.dumps(representative_baseline.lambdas, sort_keys=True),
        optimized_lambdas_json=json.dumps(representative_optimized.lambdas, sort_keys=True),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of paired repeats per dataset; each repeat runs both orders",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def main() -> None:
    args = _parse_args()
    ABS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_row = _summarize_dataset(
        "synthetic",
        *_make_synthetic_data(),
        repeats=args.repeats,
    )
    fremtpl_row = _summarize_dataset(
        "freMTPL2",
        *_load_fremtpl2_freq(),
        repeats=args.repeats,
    )
    summary = pd.DataFrame([asdict(synthetic_row), asdict(fremtpl_row)])
    summary.to_csv(ABS_CSV_PATH, index=False)

    visible_columns = [
        "dataset",
        "n_rows",
        "repeats",
        "execution_order",
        "baseline_runtime_s",
        "optimized_runtime_s",
        "speedup_x",
        "baseline_n_reml_iter",
        "optimized_n_reml_iter",
        "baseline_n_pirls_iter",
        "optimized_n_pirls_iter",
        "baseline_converged",
        "optimized_converged",
        "baseline_cleanup_gate_calls",
        "optimized_cleanup_gate_calls",
        "baseline_cleanup_gate_true_count",
        "optimized_cleanup_gate_true_count",
        "baseline_frozen_count",
        "optimized_frozen_count",
        "baseline_freeze_iter",
        "optimized_freeze_iter",
        "pred_rmse",
        "pred_max_abs_diff",
        "lambda_max_abs_diff",
    ]
    print("\nSummary rows")
    print(
        summary[visible_columns].to_string(index=False, float_format=lambda value: f"{value:.6g}")
    )
    print(f"\nWrote {ABS_CSV_PATH}")


if __name__ == "__main__":
    main()
