"""Targeted benchmark for the multi-SCOP discrete cleanup toggle.

Runs the discrete REML path twice per dataset:
- baseline: force-disable the multi-SCOP discrete cleanup
- optimized: use the branch's current cleanup behavior

The harness covers a synthetic multi-SCOP Poisson-rate dataset plus the full
freMTPL2 frequency dataset and writes a side-by-side summary CSV.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    n_rows: int
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
    original_gate = scop_efs._multi_scop_discrete_cleanup_enabled

    def disabled_gate(*, discrete: bool, scop_term_count: int) -> bool:
        nonlocal gate_calls
        gate_calls += 1
        return _cleanup_disabled(discrete=discrete, scop_term_count=scop_term_count)

    def recorded_gate(*, discrete: bool, scop_term_count: int) -> bool:
        nonlocal gate_calls
        gate_calls += 1
        return original_gate(discrete=discrete, scop_term_count=scop_term_count)

    gate_impl = recorded_gate if cleanup_enabled else disabled_gate

    with patch.object(
        scop_efs,
        "_multi_scop_discrete_cleanup_enabled",
        new=gate_impl,
    ):
        t0 = time.perf_counter()
        model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=max_reml_iter)
        runtime_s = time.perf_counter() - t0

    if cleanup_enabled and gate_calls == 0:
        raise RuntimeError("optimized variant never consulted the cleanup gate")

    return VariantResult(
        runtime_s=float(runtime_s),
        n_reml_iter=int(model._reml_result.n_reml_iter),
        n_pirls_iter=_aggregate_inner_pirls_iter(model),
        predictions=np.asarray(model.predict(X), dtype=np.float64),
        lambdas={name: float(value) for name, value in sorted(model._reml_lambdas.items())},
        converged=bool(model._reml_result.converged),
        cleanup_gate_calls=gate_calls,
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
    execution_order: tuple[str, str],
) -> SummaryRow:
    print(f"\nRunning {dataset} ({len(X):,} rows)")
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
    pred_metrics = _prediction_metrics(baseline.predictions, optimized.predictions)
    speedup_x = (
        float(baseline.runtime_s / optimized.runtime_s)
        if optimized.runtime_s > 0.0
        else float("nan")
    )
    print(
        "  "
        f"baseline={baseline.runtime_s:.3f}s "
        f"optimized={optimized.runtime_s:.3f}s "
        f"order={'->'.join(execution_order)} "
        f"speedup={speedup_x:.3f}x "
        f"optimized_gate_calls={optimized.cleanup_gate_calls} "
        f"pred_rmse={pred_metrics['rmse']:.3e} "
        f"pred_max_abs={pred_metrics['max_abs_diff']:.3e}"
    )

    return SummaryRow(
        dataset=dataset,
        n_rows=int(len(X)),
        execution_order="->".join(execution_order),
        baseline_runtime_s=baseline.runtime_s,
        optimized_runtime_s=optimized.runtime_s,
        speedup_x=speedup_x,
        baseline_n_reml_iter=baseline.n_reml_iter,
        optimized_n_reml_iter=optimized.n_reml_iter,
        baseline_n_pirls_iter=baseline.n_pirls_iter,
        optimized_n_pirls_iter=optimized.n_pirls_iter,
        baseline_converged=baseline.converged,
        optimized_converged=optimized.converged,
        baseline_cleanup_gate_calls=baseline.cleanup_gate_calls,
        optimized_cleanup_gate_calls=optimized.cleanup_gate_calls,
        pred_rmse=pred_metrics["rmse"],
        pred_max_abs_diff=pred_metrics["max_abs_diff"],
        lambda_max_abs_diff=_lambda_max_abs_diff(baseline.lambdas, optimized.lambdas),
        lambda_keys_match=set(baseline.lambdas) == set(optimized.lambdas),
        baseline_lambdas_json=json.dumps(baseline.lambdas, sort_keys=True),
        optimized_lambdas_json=json.dumps(optimized.lambdas, sort_keys=True),
    )


def main() -> None:
    ABS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_row = _summarize_dataset(
        "synthetic",
        *_make_synthetic_data(),
        execution_order=("baseline", "optimized"),
    )
    fremtpl_row = _summarize_dataset(
        "freMTPL2",
        *_load_fremtpl2_freq(),
        execution_order=("optimized", "baseline"),
    )
    summary = pd.DataFrame([asdict(synthetic_row), asdict(fremtpl_row)])
    summary.to_csv(ABS_CSV_PATH, index=False)

    visible_columns = [
        "dataset",
        "n_rows",
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
        "optimized_cleanup_gate_calls",
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
