"""Bounded synthetic complete-fit probe; see the research scaling report.

Each invocation owns one fresh, single-threaded worker. This is a descriptive
scaling experiment, not an accuracy benchmark or an asymptotic-law estimator.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

from benchmark_housing_tensor import retained_model_storage, run_isolated, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
FEATURES = tuple(f"x{i}" for i in range(8))
SEED = 202609132


def interaction_pairs():
    """Seven rounds of four disjoint pairs, covering the complete graph."""
    circle = list(FEATURES)
    pairs = []
    for _ in range(7):
        pairs.extend(tuple(sorted((circle[i], circle[-1 - i]))) for i in range(4))
        circle = [circle[0], circle[-1], *circle[1:-1]]
    assert len(set(pairs)) == 28
    return pairs


def fixture(n, *, held_out=False):
    """Fixed response law and nested training prefixes, independent of model M."""
    import numpy as np
    import pandas as pd

    offset = 100 if held_out else 0
    values = np.column_stack(
        [np.random.default_rng(SEED + offset + i).uniform(0, 1, n) for i in range(8)]
    )
    waves = np.sin(2 * np.pi * values)
    mean = waves.sum(axis=1) / np.sqrt(8)
    for j, (left, right) in enumerate(interaction_pairs()):
        mean += 0.3 * (-1) ** j * waves[:, FEATURES.index(left)] * waves[:, FEATURES.index(right)]
    response = mean + np.random.default_rng(SEED + offset + 8).normal(0, 0.3, n)
    digest = hashlib.sha256(values.astype("<f8").tobytes())
    digest.update(response.astype("<f8").tobytes())
    return pd.DataFrame(values, columns=FEATURES), response, digest.hexdigest()


def worker(args):
    import numpy as np
    from threadpoolctl import threadpool_info

    from superglm import Spline, SuperGLM

    package_hash = source_fingerprint()
    train, response, data_hash = fixture(args.rows)
    test, test_response, test_hash = fixture(1024, held_out=True)
    pairs = interaction_pairs()[: args.interactions]
    interaction_k = args.k if args.interaction_k is None else args.interaction_k
    model = SuperGLM(
        family="gaussian",
        features={
            name: Spline(kind="cr", k=args.k, knot_strategy="uniform", penalty="ssp")
            for name in FEATURES
        },
        interactions=pairs if args.interaction_k is None else [],
        selection_penalty=0.0,
        spline_penalty=0.1,
        discrete=True,
        n_bins=64,
    )
    if args.interaction_k is not None:
        for left, right in pairs:
            model._add_interaction(left, right, n_knots=(interaction_k - 2, interaction_k - 2))
    profiler = cProfile.Profile() if args.profile else None
    started_utc = datetime.now(UTC).isoformat()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        try:
            if args.mode == "reml":
                model.fit_reml(train, response)
            else:
                model.fit(train, response)
        finally:
            elapsed = time.perf_counter() - started
            rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(args.output / "fit.prof")
    finished_utc = datetime.now(UTC).isoformat()
    retained = retained_model_storage(model)
    telemetry = model.training_telemetry()
    width = 8 * (args.k - 1) + args.interactions * (interaction_k - 1) ** 2
    assert len(model.result.beta) == width
    train_prediction = model.predict(train)
    test_prediction = model.predict(test)
    assert np.isfinite(train_prediction).all() and np.isfinite(test_prediction).all()
    np.savez_compressed(
        args.output / "predictions.npz",
        train=train_prediction,
        test=test_prediction,
        beta=model.result.beta,
    )
    reml = telemetry["reml"]
    converged = bool(model.result.converged) and (args.mode != "reml" or bool(reml["converged"]))
    result = {
        "status": "converged" if converged else "not_converged",
        "rows": args.rows,
        "k": args.k,
        "interaction_k": interaction_k,
        "interactions": args.interactions,
        "pairs": pairs,
        "mode": args.mode,
        "profiled": args.profile,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "fit_seconds": elapsed,
        "fit_end_peak_process_rss_mib": rss_mib,
        "retained_model_storage": retained,
        "coefficient_count_without_intercept": width,
        "nominal_penalty_component_count": 8 + 2 * args.interactions,
        "fitted_smoothing_parameter_count": len(reml["lambdas"]) if args.mode == "reml" else 0,
        "resolved_direct_backend": model.result.direct_backend,
        "mse": {
            "train": float(np.mean((response - train_prediction) ** 2)),
            "test": float(np.mean((test_response - test_prediction) ** 2)),
        },
        "warnings": [str(item.message) for item in caught],
        "data_sha256": data_hash,
        "test_data_sha256": test_hash,
        "package_source_sha256": package_hash,
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_housing_tensor.py").read_bytes()
        ).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("superglm", "numpy", "scipy", "pandas", "numba", "threadpoolctl")
            },
            "threadpools": threadpool_info(),
        },
        "backend_groups": [type(group).__name__ for group in model._dm.group_matrices],
        "telemetry": telemetry,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "fit_seconds", "mse")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--interactions", type=int, choices=range(29), default=1)
    parser.add_argument("--k", type=int, choices=range(4, 11), default=6)
    parser.add_argument(
        "--interaction-k",
        type=int,
        choices=range(4, 11),
        help="Override tensor marginal width using its existing knot option; additive k stays fixed",
    )
    parser.add_argument("--mode", choices=("fixed", "reml"), default="reml")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 128 or args.rows > 8192 or not 0 < args.timeout <= 300:
        parser.error("Probe budget: 128 <= rows <= 8192 and 0 < timeout <= 300")
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--rows",
        str(args.rows),
        "--interactions",
        str(args.interactions),
        "--k",
        str(args.k),
        "--mode",
        args.mode,
        "--output",
        str(args.output),
    ]
    if args.profile:
        command.append("--profile")
    if args.interaction_k is not None:
        command.extend(["--interaction-k", str(args.interaction_k)])
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    receipt = run_isolated(
        command, log_path=args.output / "worker.log", timeout=args.timeout, env=env
    )
    receipt.update(command=command, timeout_seconds=args.timeout, profiled=args.profile)
    (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
