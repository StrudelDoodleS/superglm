"""Compare complete Tweedie REML fits with LIL and CSR row expansion.

All inputs are generated here. Timings exclude profiling and memory sampling.
Each memory comparison uses a fresh subprocess and a warmed fit.

Run with the development dependencies installed::

    uv run --no-sync python benchmarks/benchmark_ordered_row_expansion.py \
        --output /tmp/ordered-row-expansion.json
"""

from __future__ import annotations

import argparse
import cProfile
import ctypes
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import scipy
import scipy.sparse as sp
from threadpoolctl import threadpool_info

import superglm
import superglm.reml.direct as direct
from superglm import Categorical, OrderedCategorical, Spline, SuperGLM, families

_PRODUCTION_EXPANDER = OrderedCategorical._expand_rows
_LEVELS = [f"L{i:02}" for i in range(12)]


def _lil_row_expansion(info, ordered_mask):
    """The row-expansion implementation on master at ebd30f9a."""
    expanded = sp.lil_matrix((len(ordered_mask), info.columns.shape[1]), dtype=np.float64)
    expanded[np.flatnonzero(ordered_mask)] = info.columns
    return replace(info, columns=expanded.tocsr())


def _data(rows, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 13, (2, rows))
    category = rng.integers(0, 6, rows)
    labels = np.array(_LEVELS + ["Unknown"])
    X = pd.DataFrame({"band_a": labels[a], "band_b": labels[b], "category": category.astype(str)})
    mu = np.exp(0.3 + 0.4 * np.sin(a / 3) + 0.2 * np.cos(b / 3) + 0.08 * category)
    # Compound Poisson-gamma response with Tweedie p=1.5 and dispersion 1.
    counts = rng.poisson(2 * np.sqrt(mu))
    y = rng.gamma(counts.astype(float), np.sqrt(mu) / 2)
    return X, y


def _fit(data, discrete, variant, pools=None):
    X, y = data
    features = {
        name: OrderedCategorical(
            order=_LEVELS,
            specials=["Unknown"],
            basis=Spline(kind="cr", n_knots=5, knot_strategy="uniform"),
        )
        for name in ["band_a", "band_b"]
    }
    features["category"] = Categorical()
    model = SuperGLM(family=families.tweedie(p=1.5), discrete=discrete, features=features)
    expander = _lil_row_expansion if variant == "lil" else _PRODUCTION_EXPANDER
    if pools is not None:
        underlying = expander

        def expander(info, mask):
            if not pools:
                pools.extend(
                    {key: pool[key] for key in ["internal_api", "prefix", "num_threads"]}
                    for pool in threadpool_info()
                )
            return underlying(info, mask)

    with patch.object(OrderedCategorical, "_expand_rows", staticmethod(expander)):
        started = time.perf_counter()
        model.fit_reml(X, y)
        elapsed = time.perf_counter() - started
    diag = model.reml_diagnostics()
    assert model.result.converged and diag["converged"]
    metrics = {
        "variant": variant,
        "wall_s": elapsed,
        "dm_build_s": diag["profile"]["dm_build_s"],
        "backend": model.result.direct_backend,
        "iterations": diag["n_reml_iter"],
        "coefficients": len(model.result.beta) + 1,
        "storage": [type(group).__name__ for group in model._dm.group_matrices],
    }
    return model, metrics


def _profile(data, discrete, variant):
    profiler, pools = cProfile.Profile(), []
    with (
        patch.object(
            direct, "optimize_discrete_reml_cached_w", wraps=direct.optimize_discrete_reml_cached_w
        ) as cached,
        patch.object(direct, "fit_irls_direct", wraps=direct.fit_irls_direct) as exact,
    ):
        model, _ = profiler.runcall(_fit, data, discrete, variant, pools)
    profiler.create_stats()
    rows = []
    for key, stats in profiler.stats.items():
        if key[2] not in {"_expand_rows", "_lil_row_expansion"} and not (
            Path(key[0]).name == "_lil.py" and key[2].startswith("_set")
        ):
            continue
        rows.append(
            {
                "file": Path(key[0]).name,
                "function": key[2],
                "calls": stats[1],
                "self_s": stats[2],
                "cumulative_s": stats[3],
                "callers": sorted({caller[2] for caller in stats[4]}),
            }
        )
    del model
    gc.collect()
    return {
        "cached_optimizer_calls": cached.call_count,
        "exact_solver_calls": exact.call_count,
        "fit_threadpools": pools,
        "row_expansion": rows,
    }


def _memory(args):
    data = _data(args.rows, args.seed)
    model, _ = _fit(data, bool(args.discrete), args.variant)
    del model
    gc.collect()
    # Release free libc arenas after warmup when the platform supports it.
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(0)
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    done = threading.Event()

    def sample():
        while not done.wait(0.002):
            samples.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        model, metrics = _fit(data, bool(args.discrete), args.variant)
        retained = process.memory_info().rss
        samples.append(retained)
    finally:
        done.set()
        monitor.join()
    return {
        "baseline_rss_mib": baseline / 2**20,
        "peak_rss_mib": max(samples) / 2**20,
        "peak_increase_mib": (max(samples) - baseline) / 2**20,
        "retained_increase_mib": (retained - baseline) / 2**20,
        "sampling_interval_s": 0.002,
        "malloc_trim_used": trim is not None,
        "dispatch": {key: metrics[key] for key in ["backend", "storage", "iterations"]},
    }


def _comparison(args):
    data = _data(args.rows, args.seed)
    report = {
        "rows": args.rows,
        "seed": args.seed,
        "repeats": args.repeats,
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(),
            "superglm": superglm.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ["SUPERGLM_BLAS_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        },
        "cases": [],
    }
    for discrete in [True, False]:
        profiles = {variant: _profile(data, discrete, variant) for variant in ["lil", "csr"]}
        for key in ["cached_optimizer_calls", "exact_solver_calls", "fit_threadpools"]:
            assert profiles["lil"][key] == profiles["csr"][key]
        runs, reference, dispatch = [], None, None
        for repeat in range(args.repeats):
            for variant in ["lil", "csr"] if repeat % 2 == 0 else ["csr", "lil"]:
                model, metrics = _fit(data, discrete, variant)
                values = {
                    "beta": model.result.beta.copy(),
                    "intercept": model.result.intercept,
                    "predictions": model.predict(data[0]),
                    "lambdas": model.reml_diagnostics()["lambdas"],
                    "deviance": model.result.deviance,
                }
                current_dispatch = {
                    key: metrics[key]
                    for key in ["backend", "iterations", "storage", "coefficients"]
                }
                if reference is None:
                    reference, dispatch = values, current_dispatch
                metrics["identical"] = {
                    key: bool(np.array_equal(value, reference[key]))
                    for key, value in values.items()
                }
                assert all(metrics["identical"].values())
                assert current_dispatch == dispatch
                runs.append(metrics)
                del model, values
                gc.collect()
        report["cases"].append(
            {
                "discrete": discrete,
                "runs": runs,
                "profiles": profiles,
                "medians": {
                    variant: {
                        metric: statistics.median(
                            run[metric] for run in runs if run["variant"] == variant
                        )
                        for metric in ["wall_s", "dm_build_s"]
                    }
                    for variant in ["lil", "csr"]
                },
            }
        )
    # Keep child-process memory measurements separate from all timed fits.
    for case in report["cases"]:
        case["memory"] = {}
        for variant in ["lil", "csr"]:
            child = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--memory",
                    "--rows",
                    str(args.rows),
                    "--seed",
                    str(args.seed),
                    "--discrete",
                    str(int(case["discrete"])),
                    "--variant",
                    variant,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            case["memory"][variant] = json.loads(child.stdout)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=64123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--memory", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--discrete", type=int, choices=[0, 1], default=1, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=["lil", "csr"], default="csr", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    report = _memory(args) if args.memory else _comparison(args)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
