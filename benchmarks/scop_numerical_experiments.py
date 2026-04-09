"""Benchmark exact and approximate SCOP numerical-method prototypes.

Compares the current direct joint SCOP solve against private prototype modes:
- exact MINRES solve
- inexact MINRES solve
- cross-block truncation
- combined inexact + truncation
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import PSpline, Spline
from superglm.solvers.scop_newton import configure_scop_prototype, reset_scop_prototype

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scop_discrete_limit import make_dataset, make_features

RESULTS_DIR = Path("benchmarks/results")
CSV_PATH = RESULTS_DIR / "scop_numerical_experiments.csv"

MTPL2_PATH = Path("/home/mhick/python_projects/superglm/scratch/r_experiments/mtpl2_prepared.csv")


@dataclass
class ExperimentRow:
    case_name: str
    mode: str
    runtime_s: float
    converged: bool
    outer_iters: int
    inner_avg: float
    pred_rmse_vs_direct: float
    pred_corr_vs_direct: float
    max_abs_log_lambda_diff: float
    min_monotone_diff: float


def _min_monotone_diff(model: SuperGLM, feature_name: str) -> float:
    spec = model._specs[feature_name]
    groups = [g for g in model._groups if g.feature_name == feature_name]
    beta = np.concatenate([model.result.beta[g.sl] for g in groups])
    grid = np.linspace(spec._lo, spec._hi, 400)
    eta = np.asarray(spec.transform(grid) @ beta, dtype=np.float64)
    return float(np.min(np.diff(eta)))


def _min_monotone_diff_all(model: SuperGLM, feature_names: list[str]) -> float:
    return min(_min_monotone_diff(model, feature_name) for feature_name in feature_names)


def _max_abs_log_lambda_diff(ref: dict[str, float], other: dict[str, float]) -> float:
    diffs = []
    for key, value in ref.items():
        other_value = other.get(key, value)
        if value > 0 and other_value > 0:
            diffs.append(abs(math.log(other_value) - math.log(value)))
    return max(diffs) if diffs else 0.0


def _fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    features: dict[str, object],
    *,
    max_reml_iter: int,
) -> SuperGLM:
    model = SuperGLM(family="poisson", discrete=True, features=features)
    model.fit_reml(X, y, sample_weight=w, max_reml_iter=max_reml_iter)
    return model


def _run_mode(
    *,
    mode_name: str,
    prototype_cfg: dict[str, object],
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    features: dict[str, object],
    compare_X: pd.DataFrame,
    monotone_features: list[str],
    direct_pred: np.ndarray,
    direct_lambdas: dict[str, float],
    max_reml_iter: int,
) -> ExperimentRow:
    reset_scop_prototype()
    configure_scop_prototype(**prototype_cfg)
    try:
        t0 = time.perf_counter()
        model = _fit_model(X, y, w, features, max_reml_iter=max_reml_iter)
        runtime_s = time.perf_counter() - t0
    finally:
        reset_scop_prototype()

    rr = model._reml_result
    pred = np.asarray(model.predict(compare_X), dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred - direct_pred) ** 2)))
    corr = float(np.corrcoef(pred, direct_pred)[0, 1])
    inner_hist = rr.inner_iter_history or []

    return ExperimentRow(
        case_name="",
        mode=mode_name,
        runtime_s=runtime_s,
        converged=bool(rr.converged),
        outer_iters=int(rr.n_reml_iter),
        inner_avg=float(np.mean(inner_hist)) if inner_hist else float(model.result.n_iter),
        pred_rmse_vs_direct=rmse,
        pred_corr_vs_direct=corr,
        max_abs_log_lambda_diff=_max_abs_log_lambda_diff(direct_lambdas, rr.lambdas),
        min_monotone_diff=_min_monotone_diff_all(model, monotone_features),
    )


def _synthetic_case(
    n: int, n_scop: int, k: int, seed: int
) -> tuple[str, pd.DataFrame, np.ndarray, np.ndarray, dict[str, object], list[str]]:
    X, y, exposure = make_dataset(n=n, n_scop=n_scop, seed=seed)
    features = make_features(n_scop=n_scop, k=k, monotone=True)
    monotone_features = [f"x{j + 1}" for j in range(n_scop)]
    return f"synthetic_n{n}_scop{n_scop}_k{k}", X, y, exposure, features, monotone_features


def _load_mtpl2() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(MTPL2_PATH)
    exposure = df["Exposure"].to_numpy(dtype=np.float64)
    y = df["y_freq"].to_numpy(dtype=np.float64)
    return df, y, exposure


def _mtpl2_case_one_scop() -> tuple[
    str, pd.DataFrame, np.ndarray, np.ndarray, dict[str, object], list[str]
]:
    df, y, exposure = _load_mtpl2()
    features = {
        "DrivAge": Spline(kind="cr", k=16),
        "VehAge": Spline(kind="cr", k=12),
        "BonusMalus": PSpline(n_knots=12, monotone="increasing", monotone_mode="fit"),
        "LogDensity": Spline(kind="cr", k=10),
        "Area": Categorical(base="first"),
    }
    return "mtpl2_freq_1scop_bonusmalus", df, y, exposure, features, ["BonusMalus"]


def _mtpl2_case_two_scop(
    second_scop: str = "LogDensity",
) -> tuple[str, pd.DataFrame, np.ndarray, np.ndarray, dict[str, object], list[str]]:
    df, y, exposure = _load_mtpl2()
    monotone_features = ["BonusMalus", second_scop]
    features: dict[str, object] = {
        "DrivAge": Spline(kind="cr", k=16),
        "VehAge": Spline(kind="cr", k=12),
        "BonusMalus": PSpline(n_knots=12, monotone="increasing", monotone_mode="fit"),
        "Area": Categorical(base="first"),
    }
    if second_scop == "LogDensity":
        features["LogDensity"] = PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")
    elif second_scop == "VehAge":
        features["VehAge"] = PSpline(n_knots=10, monotone="increasing", monotone_mode="fit")
    else:
        raise ValueError(
            f"Unsupported second_scop={second_scop!r}; expected 'LogDensity' or 'VehAge'"
        )
    return (
        f"mtpl2_freq_2scop_bonusmalus_{second_scop.lower()}",
        df,
        y,
        exposure,
        features,
        monotone_features,
    )


def run_suite(*, max_reml_iter: int, mtpl2_second_scop: str) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        _synthetic_case(n=100_000, n_scop=10, k=14, seed=202602),
    ]
    if MTPL2_PATH.exists():
        cases.append(_mtpl2_case_one_scop())
        cases.append(_mtpl2_case_two_scop(second_scop=mtpl2_second_scop))

    modes = [
        ("direct", {}),
        (
            "minres_exact",
            {
                "solve_mode": "minres",
                "iterative_q_total_min": 1,
                "iterative_rtol": 1e-10,
                "iterative_maxiter": 400,
            },
        ),
        (
            "minres_inexact",
            {
                "solve_mode": "minres_inexact",
                "iterative_q_total_min": 1,
                "iterative_rtol": 1e-4,
                "iterative_maxiter": 40,
            },
        ),
        ("dropcross_1e-3", {"cross_block_rel_tol": 1e-3}),
        (
            "minres_inexact_dropcross_1e-3",
            {
                "solve_mode": "minres_inexact",
                "iterative_q_total_min": 1,
                "iterative_rtol": 1e-4,
                "iterative_maxiter": 40,
                "cross_block_rel_tol": 1e-3,
            },
        ),
    ]

    rows: list[ExperimentRow] = []
    for case_name, X, y, w, features, monotone_features in cases:
        print(f"\n=== {case_name} ===")
        t0 = time.perf_counter()
        direct_model = _fit_model(X, y, w, features, max_reml_iter=max_reml_iter)
        direct_runtime_s = time.perf_counter() - t0
        direct_rr = direct_model._reml_result
        compare_idx = np.linspace(0, len(X) - 1, min(len(X), 4096), dtype=np.intp)
        compare_X = X.iloc[compare_idx].copy()
        direct_pred = np.asarray(direct_model.predict(compare_X), dtype=np.float64)
        direct_inner = direct_rr.inner_iter_history or []
        direct_row = ExperimentRow(
            case_name=case_name,
            mode="direct",
            runtime_s=float("nan"),
            converged=bool(direct_rr.converged),
            outer_iters=int(direct_rr.n_reml_iter),
            inner_avg=float(np.mean(direct_inner))
            if direct_inner
            else float(direct_model.result.n_iter),
            pred_rmse_vs_direct=0.0,
            pred_corr_vs_direct=1.0,
            max_abs_log_lambda_diff=0.0,
            min_monotone_diff=_min_monotone_diff_all(direct_model, monotone_features),
        )
        direct_row.runtime_s = direct_runtime_s
        rows.append(direct_row)
        print(
            f"{direct_row.mode:28s} {direct_row.runtime_s:7.3f}s "
            f"outer={direct_row.outer_iters:2d} inner={direct_row.inner_avg:4.2f}"
        )

        direct_lambdas = direct_model._reml_result.lambdas
        direct_pred = np.asarray(direct_model.predict(compare_X), dtype=np.float64)

        for mode_name, cfg in modes[1:]:
            row = _run_mode(
                mode_name=mode_name,
                prototype_cfg=cfg,
                X=X,
                y=y,
                w=w,
                features=features,
                compare_X=compare_X,
                monotone_features=monotone_features,
                direct_pred=direct_pred,
                direct_lambdas=direct_lambdas,
                max_reml_iter=max_reml_iter,
            )
            row.case_name = case_name
            rows.append(row)
            print(
                f"{row.mode:28s} {row.runtime_s:7.3f}s outer={row.outer_iters:2d} "
                f"inner={row.inner_avg:4.2f} rmse={row.pred_rmse_vs_direct:8.5f} "
                f"dloglam={row.max_abs_log_lambda_diff:7.4f}"
            )

    df = pd.DataFrame(asdict(r) for r in rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote {CSV_PATH}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reml-iter", type=int, default=20)
    parser.add_argument(
        "--mtpl2-second-scop",
        choices=["LogDensity", "VehAge"],
        default="LogDensity",
        help="Second monotone feature for the 2-SCOP MTPL2 stress case.",
    )
    args = parser.parse_args()
    run_suite(max_reml_iter=args.max_reml_iter, mtpl2_second_scop=args.mtpl2_second_scop)


if __name__ == "__main__":
    main()
