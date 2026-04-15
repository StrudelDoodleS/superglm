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
TIMEOUT_S = 30.0
CASE_TIMEOUT_S = 60.0


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
    with_ti: bool,
) -> dict:
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        spline_penalty=0.0,
        discrete=True,
        n_bins=256,
        features=build_features(discrete=True),
        interactions=[("DrivAge", "BonusMalus")] if with_ti else None,
    )
    t0 = time.perf_counter()
    model.fit_reml(X_train, y_train, sample_weight=w_train, max_reml_iter=30)
    fit_s = time.perf_counter() - t0

    if fit_s > TIMEOUT_S:
        return {
            "model": name,
            "with_ti": with_ti,
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
        "with_ti": with_ti,
        "fit_s": fit_s,
        "predict_test_median_s": float(np.median(predict_times)),
        "gini_model": float(lorenz.gini_model),
        "gini_ratio": float(lorenz.gini_ratio),
        "effective_df": float(model.result.effective_df),
        "n_reml_iter": int(model._reml_result.n_reml_iter),
        "converged": bool(model._reml_result.converged),
        "reml_n_linesearch_fits": int(model._reml_profile.get("reml_n_linesearch_fits", 0)),
        "reml_linesearch_s": float(model._reml_profile.get("reml_linesearch_s", 0.0)),
        "reml_n_outer_iter": int(model._reml_profile.get("reml_n_outer_iter", 0)),
        "reml_pirls_s": float(model._reml_profile.get("reml_pirls_s", 0.0)),
        "reml_objective_s": float(model._reml_profile.get("reml_objective_s", 0.0)),
        "reml_hessian_newton_s": float(model._reml_profile.get("reml_hessian_newton_s", 0.0)),
        "reml_n_analytical_iters": int(model._reml_profile.get("reml_n_analytical_iters", 0)),
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
    with_ti: bool,
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
                with_ti=with_ti,
            )
        )
    except BaseException as exc:  # pragma: no cover - benchmark failure path
        queue.put({"model": name, "with_ti": with_ti, "error": repr(exc)})


def fit_case(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    *,
    with_ti: bool,
    timeout_s: float = TIMEOUT_S,
) -> dict:
    ctx = mp.get_context("fork")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_fit_case_worker,
        args=(queue, name, X_train, X_test, y_train, y_test, w_train, w_test, with_ti),
    )
    proc.start()
    proc.join(CASE_TIMEOUT_S)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "model": name,
            "with_ti": with_ti,
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

    baseline = fit_case(
        "baseline_discrete",
        X_train,
        X_test,
        y_train,
        y_test,
        w_train,
        w_test,
        with_ti=False,
    )
    with_ti = fit_case(
        "baseline_plus_ti_discrete",
        X_train,
        X_test,
        y_train,
        y_test,
        w_train,
        w_test,
        with_ti=True,
    )
    rows = [baseline, with_ti]

    out = {
        "dataset": "freMTPL2freq",
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": "claim_rate",
        "weight": "exposure",
        "feature_set": ["DrivAge", "VehAge", "BonusMalus", "Area"],
        "interaction": "DrivAge:BonusMalus",
        "split_seed": 42,
        "timeout_s": TIMEOUT_S,
        "results": rows,
        "deltas": {
            "fit_s": float(with_ti["fit_s"] - baseline["fit_s"]),
            "predict_test_median_s": (
                None
                if with_ti["predict_test_median_s"] is None
                else float(with_ti["predict_test_median_s"] - baseline["predict_test_median_s"])
            ),
            "gini_model": (
                None
                if with_ti["gini_model"] is None
                else float(with_ti["gini_model"] - baseline["gini_model"])
            ),
            "gini_ratio": (
                None
                if with_ti["gini_ratio"] is None
                else float(with_ti["gini_ratio"] - baseline["gini_ratio"])
            ),
            "effective_df": (
                None
                if with_ti["effective_df"] is None
                else float(with_ti["effective_df"] - baseline["effective_df"])
            ),
            "reml_n_linesearch_fits": (
                None
                if with_ti["reml_n_linesearch_fits"] is None
                else int(with_ti["reml_n_linesearch_fits"] - baseline["reml_n_linesearch_fits"])
            ),
            "reml_linesearch_s": (
                None
                if with_ti["reml_linesearch_s"] is None
                else float(with_ti["reml_linesearch_s"] - baseline["reml_linesearch_s"])
            ),
        },
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
