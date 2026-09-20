"""Acceptance diagnostics using the archived exact-fit and ten-pair fixtures.

Run in a fresh process with all numerical thread environment variables pinned.
Source selection happens before importing the package. Output must be new.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import resource
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument(
    "--case",
    choices=["gaussian10", "gaussian50", "poisson30", "support10", "tensor10", "gamma1"],
    required=True,
)
parser.add_argument("--rows", type=int, default=100_000)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
args.source = args.source.resolve()
args.output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(args.source / "src"))

import numba
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_info

import superglm
from superglm import Spline, SuperGLM

assert Path(superglm.__file__).resolve().is_relative_to(args.source / "src")


def source_hash():
    return hashlib.sha256(
        json.dumps(
            {
                str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((args.source / "src" / "superglm").rglob("*.py"))
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def fixture():
    if args.case == "gamma1":
        rng = np.random.default_rng(909)
        x2, x3 = rng.uniform(-1, 1, args.rows), rng.uniform(-1, 1, args.rows)
        mu = np.exp(0.5 + 0.8 * np.sin(2.4 * x2) + 0.3 * x3 + 0.6 * np.sin(2.0 * x2) * x3)
        return (
            pd.DataFrame({"x2": x2, "x3": x3}),
            rng.gamma(6.0, mu / 6.0),
            None,
            dict(
                family="gamma",
                discrete=True,
                n_bins=48,
                k=8,
                interactions=[("x2", "x3")],
            ),
        )
    if args.case == "support10":
        rng = np.random.default_rng(20260920)
        order = np.arange(args.rows) % 38000
        rng.shuffle(order)
        values = np.linspace(-3, 3, 38000)[order]
        frame = pd.DataFrame({"x": values})
        y = 0.3 * values + np.sin(values) + rng.normal(0, 0.5, args.rows)
        return (
            frame,
            y,
            None,
            dict(
                family="gaussian",
                discrete=False,
                k=10,
                selection_penalty=None,
            ),
        )
    if args.case == "tensor10":
        rng = np.random.default_rng(20260919)
        columns = [f"x{i}" for i in range(6)]
        frame = pd.DataFrame({name: rng.uniform(-1, 1, args.rows) for name in columns})
        exposure = rng.uniform(0.2, 1.0, args.rows)
        eta = (
            -1.6
            + 0.6 * np.sin(2.5 * frame.x0)
            + 0.4 * frame.x1**2
            + 0.3 * frame.x2
            - 0.3 * np.cos(2.0 * frame.x3)
            + 0.5 * frame.x0 * frame.x1
            + 0.4 * np.sin(2.0 * frame.x2) * frame.x3
            + 0.3 * frame.x4 * frame.x5
        )
        y = rng.poisson(np.exp(eta) * exposure)
        pairs = [(columns[i], columns[j]) for i in range(6) for j in range(i + 1, 6)][:10]
        config = dict(
            family="poisson",
            discrete=True,
            n_bins=256,
            k=10,
            interactions=pairs,
        )
        return frame, y, np.log(exposure), config
    rng = np.random.default_rng(1909)
    frame = pd.DataFrame({name: rng.uniform(-1, 1, args.rows) for name in ["x", "z"]})
    eta = 0.3 + 0.7 * np.sin(3 * frame.x) + 0.3 * frame.z**2
    gaussian = np.asarray(eta + rng.normal(0, 0.5, args.rows))
    poisson = rng.poisson(np.exp(eta)).astype(float)
    family = "poisson" if args.case == "poisson30" else "gaussian"
    k = {"gaussian10": 10, "gaussian50": 50, "poisson30": 30}[args.case]
    columns = ["x", "z"] if family == "poisson" else ["x"]
    config = dict(family=family, discrete=False, k=k)
    return frame[columns], poisson if family == "poisson" else gaussian, None, config


def fit(frame, y, offset, config, max_iter):
    options = config.copy()
    k = options.pop("k")
    model = SuperGLM(**options, features={name: Spline(kind="ps", k=k) for name in frame})
    model.fit_reml(frame, y, offset=offset, max_reml_iter=max_iter)
    return model


source_before = source_hash()
X, y, offset, config = fixture()
fixture_digest = hashlib.sha256()
for value in (X.to_numpy(), y, offset):
    if value is not None:
        value = np.ascontiguousarray(value)
        fixture_digest.update(str((value.shape, value.dtype.str)).encode())
        fixture_digest.update(value.tobytes())
warm_rows = min(2000, len(X))
fit(X.iloc[:warm_rows], y[:warm_rows], None if offset is None else offset[:warm_rows], config, 2)
profiler = cProfile.Profile() if args.profile else None
load_before = os.getloadavg()
peak_before_fit = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
if profiler:
    profiler.enable()
start, cpu_start = time.perf_counter(), time.process_time()
model = fit(X, y, offset, config, 30)
wall, cpu = time.perf_counter() - start, time.process_time() - cpu_start
peak_after_fit = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
if profiler:
    profiler.disable()
diagnostics = model.reml_diagnostics()
result = model._reml_result
np.savez_compressed(
    args.output / "outputs.npz", prediction=model.predict(X, offset=offset), beta=model.result.beta
)
record = {
    "case": args.case,
    "rows": args.rows,
    "source": str(args.source),
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source, text=True
    ).strip(),
    "source_before": source_before,
    "source_after": source_hash(),
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
    "fixture_sha256": fixture_digest.hexdigest(),
    "profiled": args.profile,
    "warmup": "same design, first 2000 rows, two REML iterations",
    "wall_seconds": wall,
    "cpu_seconds": cpu,
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "peak_before_fit_mib": peak_before_fit,
    "fit_peak_rss_mib": peak_after_fit,
    "converged": bool(result.converged),
    "termination_reason": result.termination_reason,
    "outer_iterations": int(result.n_reml_iter),
    "states": len(diagnostics.get("lambda_history", [])),
    "objective": float(result.objective),
    "deviance": float(model.result.deviance),
    "edf": float(model.result.effective_df),
    "lambdas": diagnostics["lambdas"],
    "backend": getattr(model.result, "direct_backend", None),
    "group_types": [type(group).__name__ for group in model._dm.group_matrices],
    "support_rows": [
        len(group.B_unique) if hasattr(group, "B_unique") else None
        for group in model._dm.group_matrices
    ],
    "numba_threads": numba.get_num_threads(),
    "fit_profile": diagnostics.get("profile"),
    "threadpools": threadpool_info(),
    "thread_environment": {
        name: os.environ.get(name)
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "SUPERGLM_BLAS_THREADS",
        )
    },
    "load_before": load_before,
    "load_after": os.getloadavg(),
}
if profiler:
    profiler.dump_stats(str(args.output / "fit.prof"))
    record["kernel_calls"] = [
        {
            "file": filename,
            "line": line,
            "name": name,
            "calls": counts[1],
            "cumulative_seconds": counts[3],
        }
        for (filename, line, name), counts in pstats.Stats(profiler).stats.items()
        if "_group_matrix" in filename or name in {"build_centered_system", "fit_irls_direct"}
    ]
(args.output / "receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
assert record["source_before"] == record["source_after"], "Source changed during fit"
print(
    json.dumps(
        {
            key: record[key]
            for key in [
                "case",
                "rows",
                "source",
                "profiled",
                "wall_seconds",
                "converged",
                "outer_iterations",
            ]
        }
    ),
    flush=True,
)
