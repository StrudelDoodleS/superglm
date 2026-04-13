"""Frequency benchmark: holdout Gini for the basic freMTPL2 feature set.

Uses the same data prep and 80/20 split as the existing MTPL frequency
benchmarks:

    s(DrivAge) + s(VehAge) + s(BonusMalus) + Area

Reports raw and normalized holdout Gini for exact and discrete SuperGLM REML
fits, along with holdout prediction timing, and writes the results to JSON.
"""

from __future__ import annotations

import json
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
OUT_JSON = OUT_DIR / "freq_gini.json"


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


def build_superglm_features(discrete: bool) -> dict:
    return {
        "DrivAge": Spline(kind="cr", k=20, penalty="ssp", discrete=discrete),
        "VehAge": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
        "BonusMalus": Spline(kind="cr", k=15, penalty="ssp", discrete=discrete),
        "Area": Categorical(base="most_exposed"),
    }


def fit_superglm_gini(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    w_train: np.ndarray,
    w_test: np.ndarray,
    *,
    discrete: bool,
) -> dict:
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=256,
        features=build_superglm_features(discrete),
    )
    t0 = time.perf_counter()
    model.fit_reml(X_train, y_train, sample_weight=w_train, max_reml_iter=30)
    elapsed = time.perf_counter() - t0

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
        "time_s": elapsed,
        "predict_test_median_s": float(np.median(predict_times)),
        "gini_model": float(lorenz.gini_model),
        "gini_perfect": float(lorenz.gini_perfect),
        "gini_ratio": float(lorenz.gini_ratio),
        "test_pred_mean": float(mu.mean()),
        "effective_df": float(model.result.effective_df),
        "n_reml_iter": int(model._reml_result.n_reml_iter),
        "converged": bool(model._reml_result.converged),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, w = load_freq()
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(X, y, w)

    rows = [
        fit_superglm_gini(
            "superglm_exact",
            X_train,
            X_test,
            y_train,
            y_test,
            w_train,
            w_test,
            discrete=False,
        ),
        fit_superglm_gini(
            "superglm_discrete",
            X_train,
            X_test,
            y_train,
            y_test,
            w_train,
            w_test,
            discrete=True,
        ),
    ]

    out = {
        "dataset": "freMTPL2freq",
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": "claim_rate",
        "weight": "exposure",
        "feature_set": ["DrivAge", "VehAge", "BonusMalus", "Area"],
        "split_seed": 42,
        "results": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print("Frequency benchmark: holdout Gini")
    print("=" * 72)
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}")
    print()
    for row in rows:
        print(
            f"{row['model']:<18s} "
            f"time={row['time_s']:>7.2f}s  "
            f"predict={row['predict_test_median_s']:.4f}s  "
            f"gini={row['gini_model']:.6f}  "
            f"gini_ratio={row['gini_ratio']:.6f}  "
            f"edf={row['effective_df']:>7.2f}"
        )
    print()
    print(f"Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
