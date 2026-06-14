"""Discrete MTPL frequency benchmark for a tensor interaction stress case.

Uses the same freMTPL2freq data preparation and 80/20 split as the other MTPL
frequency benchmarks, but compares:

    s(DrivAge) + s(VehAge) + s(BonusMalus) + Area

against:

    s(DrivAge) + s(VehAge) + s(BonusMalus) + ti(DrivAge, BonusMalus) + Area

The immediate goal is to provide a tracked reproduction of the current
``discrete=True fit_reml`` tensor-interaction failure mode so later refactors
have a fixed target.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.validation import lorenz_curve

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "freMTPL2freq.parquet"
if not DATA_PATH.exists() and ROOT.parent.name == ".worktrees":
    DATA_PATH = ROOT.parent.parent / "data" / "freMTPL2freq.parquet"
OUT_DIR = ROOT / "benchmarks" / "results"
OUT_JSON = OUT_DIR / "tensor_ti_freq.json"
OUT_TRAIN_CSV = OUT_DIR / "tensor_ti_freq_train.csv"
OUT_TEST_CSV = OUT_DIR / "tensor_ti_freq_test.csv"
TIMEOUT_S = 120.0
CASE_TIMEOUT_S = 180.0
PROFILE_TIMING_KEYS = (
    "reml_rebuild_dm_s",
    "reml_map_beta_s",
    "reml_penalty_context_s",
    "reml_tensor_summary_s",
    "fit_prime_caches_s",
    "fit_runtime_canonicalize_s",
    "fit_release_state_s",
    "irls_working_s",
    "irls_gram_s",
    "irls_solve_s",
    "irls_eta_s",
    "irls_deviance_s",
    "irls_deviance_eval_s",
    "irls_total_s",
    "irls_calls",
    "irls_iters",
    "block_diag_tensor_s",
    "block_diag_discrete_ssp_s",
    "block_diag_other_s",
    "block_cross_tensor_own_margin_s",
    "block_cross_tensor_main_s",
    "block_cross_tensor_tensor_s",
    "block_cross_tensor_spline_cat_s",
    "block_cross_spline_cat_spline_cat_s",
    "block_cross_disc_disc_s",
    "block_cross_disc_other_s",
    "block_cross_cat_cat_s",
    "block_cross_fallback_s",
    "block_hist2d_s",
    "block_tabmat_s",
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    interactions: tuple[tuple[str, str], ...] = ()

    @property
    def with_ti(self) -> bool:
        return ("DrivAge", "BonusMalus") in self.interactions


def build_superglm_cases() -> tuple[BenchmarkCase, ...]:
    """Cases used to track one- and multi-interaction scaling.

    The first two cases intentionally preserve the original benchmark ordering:
    downstream PR summaries compare ``baseline_discrete`` against
    ``baseline_plus_ti_discrete``.
    """
    return (
        BenchmarkCase("baseline_discrete"),
        BenchmarkCase("baseline_plus_ti_discrete", (("DrivAge", "BonusMalus"),)),
        BenchmarkCase(
            "baseline_plus_2_tensors_discrete",
            (("DrivAge", "BonusMalus"), ("DrivAge", "VehAge")),
        ),
        BenchmarkCase(
            "baseline_plus_3_tensors_discrete",
            (("DrivAge", "BonusMalus"), ("DrivAge", "VehAge"), ("VehAge", "BonusMalus")),
        ),
        BenchmarkCase("baseline_plus_spline_cat_discrete", (("DrivAge", "Area"),)),
        BenchmarkCase(
            "baseline_plus_2_spline_cat_discrete",
            (("DrivAge", "Area"), ("BonusMalus", "Area")),
        ),
        BenchmarkCase(
            "baseline_plus_mixed_tensor_spline_cat_discrete",
            (("DrivAge", "BonusMalus"), ("VehAge", "Area")),
        ),
    )


def _safe_delta(case: dict, baseline: dict, key: str) -> float | int | None:
    if case.get(key) is None or baseline.get(key) is None:
        return None
    value = case.get(key, 0.0) - baseline.get(key, 0.0)
    if isinstance(case.get(key), int) and isinstance(baseline.get(key), int):
        return int(value)
    return float(value)


def build_case_deltas(rows: list[dict]) -> dict:
    baseline = rows[0]
    one_tensor = next(
        (row for row in rows if row.get("model") == "baseline_plus_ti_discrete"),
        rows[1] if len(rows) > 1 else baseline,
    )
    legacy_keys = (
        "fit_s",
        "predict_test_median_s",
        "gini_model",
        "gini_ratio",
        "effective_df",
        "reml_n_linesearch_fits",
        "reml_linesearch_s",
    )
    deltas = {key: _safe_delta(one_tensor, baseline, key) for key in legacy_keys}
    deltas.update({key: _safe_delta(one_tensor, baseline, key) for key in PROFILE_TIMING_KEYS})
    deltas["by_case"] = {
        str(row["model"]): {
            key: _safe_delta(row, baseline, key)
            for key in (
                "fit_s",
                "predict_test_median_s",
                "gini_model",
                "gini_ratio",
                "effective_df",
                "coef_count",
                "n_groups",
                "n_smoothing_params",
                "n_reml_iter",
                "irls_calls",
                "irls_iters",
                "irls_gram_s",
                "irls_solve_s",
                "irls_eta_s",
                "reml_pirls_s",
                "reml_hessian_newton_s",
            )
        }
        for row in rows[1:]
    }
    return deltas


def load_freq() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(DATA_PATH)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    y_freq = (df["ClaimNb"] / df["Exposure"]).to_numpy(dtype=float)
    exposure = df["Exposure"].to_numpy(dtype=float)
    X = df[["DrivAge", "VehAge", "BonusMalus", "Area"]].copy()
    return X, y_freq, exposure


def split_data(
    X: pd.DataFrame, y: np.ndarray, w: np.ndarray, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(idx))
    tr, te = idx[:n_train], idx[n_train:]
    return (
        X.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        y[tr],
        y[te],
        w[tr],
        w[te],
    )


def build_features(discrete: bool) -> dict:
    return {
        "DrivAge": Spline(kind="cr", k=20, penalty="ssp", discrete=discrete),
        "VehAge": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
        "BonusMalus": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
        "Area": Categorical(base="most_exposed"),
    }


def _fit_case_result(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    *,
    interactions: tuple[tuple[str, str], ...],
) -> dict:
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.0,
        discrete=True,
        n_bins=256,
        features=build_features(discrete=True),
        interactions=list(interactions) if interactions else None,
    )
    t0 = time.perf_counter()
    model.fit_reml(X_train, y_train, sample_weight=w_train, max_reml_iter=30)
    fit_s = time.perf_counter() - t0
    profile_timings = {key: float(model._reml_profile.get(key, 0.0)) for key in PROFILE_TIMING_KEYS}
    runtime_validate = bool(model._reml_profile.get("fit_runtime_canonicalize_validate", False))
    runtime_validate_reason = str(
        model._reml_profile.get("fit_runtime_canonicalize_validate_reason", "")
    )

    if fit_s > TIMEOUT_S:
        return {
            "model": name,
            "with_ti": ("DrivAge", "BonusMalus") in interactions,
            "interactions": [f"{a}:{b}" for a, b in interactions],
            "n_interactions": len(interactions),
            "timed_out": True,
            "timeout_s": TIMEOUT_S,
            "fit_s": fit_s,
            "predict_test_median_s": None,
            "gini_model": None,
            "gini_ratio": None,
            "effective_df": None,
            "n_reml_iter": int(model._reml_result.n_reml_iter),
            "converged": False,
            "reml_n_linesearch_fits": int(model._reml_profile.get("reml_n_linesearch_fits", 0)),
            "reml_linesearch_s": float(model._reml_profile.get("reml_linesearch_s", 0.0)),
            "reml_n_outer_iter": int(model._reml_profile.get("reml_n_outer_iter", 0)),
            "reml_pirls_s": float(model._reml_profile.get("reml_pirls_s", 0.0)),
            "fit_runtime_canonicalize_validate": runtime_validate,
            "fit_runtime_canonicalize_validate_reason": runtime_validate_reason,
            "coef_count": int(model._result.beta.size),
            "n_groups": len(model._groups),
            "n_smoothing_params": 0
            if model._reml_result is None
            else len(model._reml_result.lambdas),
            **profile_timings,
        }

    mu = model.predict(X_test)
    lorenz = lorenz_curve(y_test, mu, exposure=w_test)

    _ = model.predict(X_test)
    predict_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model.predict(X_test)
        predict_times.append(time.perf_counter() - t0)

    return {
        "model": name,
        "with_ti": ("DrivAge", "BonusMalus") in interactions,
        "interactions": [f"{a}:{b}" for a, b in interactions],
        "n_interactions": len(interactions),
        "fit_s": fit_s,
        "predict_test_median_s": float(np.median(predict_times)),
        "gini_model": float(lorenz.gini_model),
        "gini_ratio": float(lorenz.gini_ratio),
        "effective_df": float(model.result.effective_df),
        "coef_count": int(model._result.beta.size),
        "n_groups": len(model._groups),
        "n_smoothing_params": len(model._reml_result.lambdas),
        "n_reml_iter": int(model._reml_result.n_reml_iter),
        "converged": bool(model._reml_result.converged),
        "reml_n_linesearch_fits": int(model._reml_profile.get("reml_n_linesearch_fits", 0)),
        "reml_linesearch_s": float(model._reml_profile.get("reml_linesearch_s", 0.0)),
        "reml_n_outer_iter": int(model._reml_profile.get("reml_n_outer_iter", 0)),
        "reml_pirls_s": float(model._reml_profile.get("reml_pirls_s", 0.0)),
        "reml_objective_s": float(model._reml_profile.get("reml_objective_s", 0.0)),
        "reml_hessian_newton_s": float(model._reml_profile.get("reml_hessian_newton_s", 0.0)),
        "reml_n_analytical_iters": int(model._reml_profile.get("reml_n_analytical_iters", 0)),
        "fit_runtime_canonicalize_validate": runtime_validate,
        "fit_runtime_canonicalize_validate_reason": runtime_validate_reason,
        **profile_timings,
    }


def _fit_case_worker(
    queue: mp.Queue,
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    interactions: tuple[tuple[str, str], ...],
) -> None:
    try:
        queue.put(
            _fit_case_result(
                name,
                X_train,
                X_test,
                y_train,
                y_test,
                w_train,
                w_test,
                interactions=interactions,
            )
        )
    except BaseException as exc:  # pragma: no cover - benchmark failure path
        queue.put(
            {
                "model": name,
                "with_ti": ("DrivAge", "BonusMalus") in interactions,
                "interactions": [f"{a}:{b}" for a, b in interactions],
                "n_interactions": len(interactions),
                "error": repr(exc),
            }
        )


def fit_case(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    *,
    interactions: tuple[tuple[str, str], ...],
    timeout_s: float = TIMEOUT_S,
) -> dict:
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_fit_case_worker,
        args=(queue, name, X_train, X_test, y_train, y_test, w_train, w_test, interactions),
    )
    proc.start()
    proc.join(CASE_TIMEOUT_S)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "model": name,
            "with_ti": ("DrivAge", "BonusMalus") in interactions,
            "interactions": [f"{a}:{b}" for a, b in interactions],
            "n_interactions": len(interactions),
            "timed_out": True,
            "timeout_s": timeout_s,
            "fit_s": timeout_s,
            "predict_test_median_s": None,
            "gini_model": None,
            "gini_ratio": None,
            "effective_df": None,
            "n_reml_iter": None,
            "converged": False,
            "reml_n_linesearch_fits": None,
            "reml_linesearch_s": None,
            "reml_n_outer_iter": None,
            "reml_pirls_s": None,
            "reml_objective_s": None,
            "reml_hessian_newton_s": None,
            "reml_n_analytical_iters": None,
            "coef_count": None,
            "n_groups": None,
            "n_smoothing_params": None,
        }

    result = queue.get()
    result.setdefault("timed_out", False)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, w = load_freq()
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(X, y, w)

    export_train = X_train.copy()
    export_train["y_freq"] = y_train
    export_train["Exposure"] = w_train
    export_train.to_csv(OUT_TRAIN_CSV, index=False)

    export_test = X_test.copy()
    export_test["y_freq"] = y_test
    export_test["Exposure"] = w_test
    export_test.to_csv(OUT_TEST_CSV, index=False)

    rows = [
        fit_case(
            case.name,
            X_train,
            X_test,
            y_train,
            y_test,
            w_train,
            w_test,
            interactions=case.interactions,
        )
        for case in build_superglm_cases()
    ]

    out = {
        "dataset": "freMTPL2freq",
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": "claim_rate",
        "weight": "exposure",
        "feature_set": ["DrivAge", "VehAge", "BonusMalus", "Area"],
        "interaction": "DrivAge:BonusMalus",
        "case_matrix": [
            {"name": case.name, "interactions": [f"{a}:{b}" for a, b in case.interactions]}
            for case in build_superglm_cases()
        ],
        "split_seed": 42,
        "timeout_s": TIMEOUT_S,
        "results": rows,
        "deltas": build_case_deltas(rows),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print("Discrete MTPL tensor benchmark")
    print("=" * 88)
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")
    print()
    for row in rows:
        print(
            f"{row['model']:<28s} "
            f"fit={row['fit_s']:>7.2f}s  "
            f"predict={row['predict_test_median_s'] if row['predict_test_median_s'] is None else format(row['predict_test_median_s'], '.4f')}s  "
            f"gini={row['gini_model'] if row['gini_model'] is None else format(row['gini_model'], '.6f')}  "
            f"gini_ratio={row['gini_ratio'] if row['gini_ratio'] is None else format(row['gini_ratio'], '.6f')}  "
            f"edf={row['effective_df'] if row['effective_df'] is None else format(row['effective_df'], '8.2f')}  "
            f"p={row['coef_count']}  "
            f"groups={row['n_groups']}  "
            f"sp={row['n_smoothing_params']}  "
            f"ls_fits={row['reml_n_linesearch_fits']}  "
            f"converged={row['converged']}  "
            f"timed_out={row['timed_out']}"
        )
    print()
    print(
        "Delta vs baseline: "
        f"fit={out['deltas']['fit_s']:+.2f}s  "
        f"predict={out['deltas']['predict_test_median_s']}s  "
        f"gini={out['deltas']['gini_model']}  "
        f"gini_ratio={out['deltas']['gini_ratio']}  "
        f"edf={out['deltas']['effective_df']}  "
        f"ls_fits={out['deltas']['reml_n_linesearch_fits']}  "
        f"ls_time={out['deltas']['reml_linesearch_s']}"
    )
    print()
    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved train CSV: {OUT_TRAIN_CSV}")
    print(f"Saved test CSV:  {OUT_TEST_CSV}")


if __name__ == "__main__":
    main()
