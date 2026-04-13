"""Personal-lines serving benchmark for SuperGLM quote-time prediction.

Measures insurer-shaped inference workloads on the basic MTPL frequency feature
set:

    s(DrivAge) + s(VehAge) + s(BonusMalus) + Area

Two benchmark modes are reported:

1. predict_only:
   Reuses an already-materialized pandas DataFrame slice and measures raw
   model.predict latency.

2. request_path:
   Simulates a quote handler more closely by building a fresh DataFrame from
   request-like records before calling model.predict.

Also reports a simple warm concurrency simulation for small quote batches.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "freMTPL2freq.parquet"
if not DATA_PATH.exists() and ROOT.parent.name == ".worktrees":
    DATA_PATH = ROOT.parent.parent / "data" / "freMTPL2freq.parquet"
RESULTS_DIR = ROOT / "benchmarks" / "results"
OUT_JSON = RESULTS_DIR / "personal_lines_serving.json"

BATCH_SIZES = [1, 5, 20, 100]
CONCURRENCY_LEVELS = [1, 4, 8, 16]
N_SERIAL_REPS = 200
N_CONCURRENCY_REQUESTS = 256


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


def fit_model(
    X_train: pd.DataFrame, y_train: np.ndarray, w_train: np.ndarray, *, discrete: bool
) -> SuperGLM:
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=256,
        features=build_features(discrete),
    )
    model.fit_reml(X_train, y_train, sample_weight=w_train, max_reml_iter=30)
    return model


def sample_batches(
    X: pd.DataFrame, batch_size: int, n_batches: int, *, seed: int
) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    batches: list[pd.DataFrame] = []
    for _ in range(n_batches):
        idx = rng.integers(0, len(X), size=batch_size)
        batches.append(X.iloc[idx].reset_index(drop=True))
    return batches


def summarize(times: list[float]) -> dict[str, float]:
    arr = np.array(times, dtype=float)
    return {
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "p99_s": float(np.percentile(arr, 99)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
    }


def run_serial_benchmark(
    model: SuperGLM, batches: list[pd.DataFrame], *, request_path: bool
) -> dict[str, float]:
    times: list[float] = []
    for batch in batches[:5]:
        if request_path:
            payload = batch.to_dict(orient="records")
            _ = model.predict(pd.DataFrame.from_records(payload))
        else:
            _ = model.predict(batch)

    for batch in batches:
        t0 = time.perf_counter()
        if request_path:
            payload = batch.to_dict(orient="records")
            _ = model.predict(pd.DataFrame.from_records(payload))
        else:
            _ = model.predict(batch)
        times.append(time.perf_counter() - t0)
    return summarize(times)


def run_concurrency_benchmark(
    model: SuperGLM,
    batches: list[pd.DataFrame],
    *,
    request_path: bool,
    max_workers: int,
) -> dict[str, float]:
    queue = batches[:N_CONCURRENCY_REQUESTS]

    def task(batch: pd.DataFrame) -> float:
        t0 = time.perf_counter()
        if request_path:
            payload = batch.to_dict(orient="records")
            _ = model.predict(pd.DataFrame.from_records(payload))
        else:
            _ = model.predict(batch)
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        latencies = list(pool.map(task, queue))
    elapsed = time.perf_counter() - t0

    out = summarize(latencies)
    out["throughput_req_per_s"] = float(len(queue) / elapsed)
    out["total_elapsed_s"] = float(elapsed)
    return out


def benchmark_model(
    name: str,
    model: SuperGLM,
    X_test: pd.DataFrame,
) -> dict[str, object]:
    serial: dict[str, dict[str, dict[str, float]]] = {}
    concurrency: dict[str, dict[str, dict[str, float]]] = {}

    for batch_size in BATCH_SIZES:
        batches = sample_batches(X_test, batch_size, N_SERIAL_REPS, seed=42 + batch_size)
        serial[str(batch_size)] = {
            "predict_only": run_serial_benchmark(model, batches, request_path=False),
            "request_path": run_serial_benchmark(model, batches, request_path=True),
        }

    for workers in CONCURRENCY_LEVELS:
        batches = sample_batches(X_test, 1, N_CONCURRENCY_REQUESTS, seed=100 + workers)
        concurrency[str(workers)] = {
            "predict_only": run_concurrency_benchmark(
                model, batches, request_path=False, max_workers=workers
            ),
            "request_path": run_concurrency_benchmark(
                model, batches, request_path=True, max_workers=workers
            ),
        }

    return {
        "model": name,
        "serial": serial,
        "concurrency_batch1": concurrency,
    }


def print_summary(result: dict[str, object]) -> None:
    print(f"\n{result['model']}")
    print("-" * len(str(result["model"])))
    for batch_size in BATCH_SIZES:
        serial = result["serial"][str(batch_size)]
        raw = serial["predict_only"]
        req = serial["request_path"]
        print(
            f"  batch={batch_size:>3d}  "
            f"raw p50={raw['median_s'] * 1000:.2f}ms p95={raw['p95_s'] * 1000:.2f}ms  "
            f"req p50={req['median_s'] * 1000:.2f}ms p95={req['p95_s'] * 1000:.2f}ms"
        )
    print("  concurrency(batch=1 request-path):")
    for workers in CONCURRENCY_LEVELS:
        item = result["concurrency_batch1"][str(workers)]["request_path"]
        print(
            f"    workers={workers:>2d}  "
            f"p50={item['median_s'] * 1000:.2f}ms  "
            f"p95={item['p95_s'] * 1000:.2f}ms  "
            f"throughput={item['throughput_req_per_s']:.1f} req/s"
        )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, w = load_freq()
    X_train, X_test, y_train, _, w_train, _ = split_data(X, y, w)

    print("Personal-lines serving benchmark")
    print("=" * 72)
    print(f"Rows: train={len(X_train):,} test={len(X_test):,}")
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        print(f"{var}={os.environ.get(var, '(not set)')}")

    exact = fit_model(X_train, y_train, w_train, discrete=False)
    discrete = fit_model(X_train, y_train, w_train, discrete=True)

    results = {
        "dataset": "freMTPL2freq",
        "feature_set": ["DrivAge", "VehAge", "BonusMalus", "Area"],
        "split_seed": 42,
        "batch_sizes": BATCH_SIZES,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "serial_reps": N_SERIAL_REPS,
        "concurrency_requests": N_CONCURRENCY_REQUESTS,
        "models": [
            benchmark_model("superglm_exact", exact, X_test),
            benchmark_model("superglm_discrete", discrete, X_test),
        ],
    }

    for model_result in results["models"]:
        print_summary(model_result)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
