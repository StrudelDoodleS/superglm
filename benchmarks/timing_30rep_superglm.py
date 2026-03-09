"""30-rep timing study for SuperGLM discrete REML on MTPL2.

Measures wall time for fit_reml(discrete=True) across 30 repetitions
with fixed thread count. Reports median, mean, std, min, max.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        uv run python benchmarks/timing_30rep_superglm.py
"""

import json
import os
import time

import numpy as np
import pandas as pd

from superglm.features.categorical import Categorical
from superglm.features.spline import CubicRegressionSpline
from superglm.model import SuperGLM

N_REPS = 30
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_mtpl2():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "freMTPL2freq.parquet")
    df = pd.read_parquet(path)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    df["LogDensity"] = np.log1p(df["Density"])
    y_freq = (df["ClaimNb"] / df["Exposure"]).values
    exposure = df["Exposure"].values
    return df, y_freq, exposure


def main():
    # Report thread settings
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        print(f"  {var}={os.environ.get(var, '(not set)')}")

    df, y_freq, exposure = load_mtpl2()
    n = len(df)
    print(f"  MTPL2: {n:,} rows")

    features = {
        "DrivAge": CubicRegressionSpline(n_knots=18),
        "VehAge": CubicRegressionSpline(n_knots=13),
        "BonusMalus": CubicRegressionSpline(n_knots=13),
        "Area": Categorical(base="most_exposed"),
    }

    times = []
    deviances = []
    edfs = []

    for i in range(N_REPS):
        model = SuperGLM(
            family="poisson",
            lambda1=0,
            features=features,
            discrete=True,
        )
        t0 = time.perf_counter()
        model.fit_reml(df, y_freq, exposure=exposure, max_reml_iter=30)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        deviances.append(float(model.result.deviance))
        edfs.append(float(model.result.effective_df))

        tag = " (warmup)" if i == 0 else ""
        print(
            f"  rep {i + 1:2d}/{N_REPS}: {elapsed:.3f}s  "
            f"dev={deviances[-1]:.1f}  edf={edfs[-1]:.2f}{tag}"
        )

    times_arr = np.array(times)
    # Separate warmup
    warmup_time = times_arr[0]
    steady_times = times_arr[1:]

    result = {
        "tool": "superglm",
        "method": "fit_reml_discrete_cached_w",
        "n": n,
        "n_reps": N_REPS,
        "threads": os.environ.get("OMP_NUM_THREADS", "default"),
        "warmup_s": float(warmup_time),
        "all_times_s": [float(t) for t in times],
        "steady_times_s": [float(t) for t in steady_times],
        "median_s": float(np.median(steady_times)),
        "mean_s": float(np.mean(steady_times)),
        "std_s": float(np.std(steady_times)),
        "min_s": float(np.min(steady_times)),
        "max_s": float(np.max(steady_times)),
        "p10_s": float(np.percentile(steady_times, 10)),
        "p90_s": float(np.percentile(steady_times, 90)),
        "deviance": float(np.median(deviances)),
        "effective_df": float(np.median(edfs)),
    }

    print(f"\n  === SuperGLM discrete REML — {N_REPS} reps (excluding warmup) ===")
    print(f"  warmup:  {warmup_time:.3f}s")
    print(f"  median:  {result['median_s']:.3f}s")
    print(f"  mean:    {result['mean_s']:.3f}s")
    print(f"  std:     {result['std_s']:.3f}s")
    print(f"  min:     {result['min_s']:.3f}s")
    print(f"  max:     {result['max_s']:.3f}s")
    print(f"  p10:     {result['p10_s']:.3f}s")
    print(f"  p90:     {result['p90_s']:.3f}s")
    print(f"  dev:     {result['deviance']:.1f}")
    print(f"  edf:     {result['effective_df']:.2f}")

    out_path = os.path.join(RESULTS_DIR, "superglm_30rep.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
