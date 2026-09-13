"""California housing geographic tensor complete-fit benchmark.

Run one case in a fresh process. Data loading is outside the fit clock but
inside the process timeout. See housing_tensor.md for the frozen specification.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import pstats
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

COLUMNS = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)
GEO = ("Latitude", "Longitude")
LOG_COLUMNS = ("AveRooms", "AveBedrms", "Population", "AveOccup")
CASES = {
    "base20": (20, "quantile"),
    "rows20": (20, "quantile_rows"),
    "support30": (30, "quantile"),
    "rows30": (30, "quantile_rows"),
}
SEED = 202609131
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).with_name("housing_tensor_reference.json")
REFERENCE_ARRAYS = Path(__file__).with_name("housing_tensor_predictions.npz")


def data_fingerprint(frame):
    """Hash ordered raw numeric values, independent of parquet metadata."""
    import numpy as np

    columns = [*COLUMNS, "MedHouseVal"]
    values = np.asarray(frame.loc[:, columns], dtype="<f8", order="C")
    if values.shape != (20640, 9) or not np.isfinite(values).all():
        raise ValueError("Expected the complete finite 20,640-row California housing dataset")
    digest = hashlib.sha256(json.dumps(columns, separators=(",", ":")).encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def prepare_data(frame, *, expected_fingerprint):
    """Validate identity before applying the frozen transforms and split."""
    import numpy as np

    if data_fingerprint(frame) != expected_fingerprint:
        raise ValueError("California housing data fingerprint differs from the reference")
    features = frame.loc[:, COLUMNS].copy()
    response = frame["MedHouseVal"].to_numpy(dtype=float)
    for column in LOG_COLUMNS:
        features[column] = np.log1p(features[column])
    order = np.random.default_rng(SEED).permutation(len(frame))
    splits = {}
    for name, indices in zip(
        ("train", "valid", "test"), np.split(order, [12384, 16512]), strict=True
    ):
        splits[name] = (features.iloc[indices].reset_index(drop=True), response[indices])
    return splits, hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()


def run_isolated(command, *, log_path, timeout, env):
    """Run and reap one owned subprocess, including after its deadline."""
    start = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            try:
                returncode = process.wait(timeout=timeout)
                status = "success" if returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
                status = "timeout"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return {
        "status": status,
        "pid": process.pid,
        "returncode": returncode,
        "process_seconds": time.perf_counter() - start,
    }


def build_model(case):
    """Keep six other k=20 specifications fixed and include only the geo pair."""
    from superglm import Spline, SuperGLM

    k, knots = CASES[case]
    features = {
        column: Spline(
            kind="cr",
            k=k if column in GEO else 20,
            knot_strategy=knots if column in GEO else "quantile",
            penalty="ssp",
        )
        for column in COLUMNS
    }
    return SuperGLM(
        family="gaussian",
        features=features,
        interactions=[GEO],
        selection_penalty=0.0,
        discrete=True,
        n_bins=256,
    )


def source_fingerprint(package_directory=None):
    if package_directory is None:
        import superglm

        package_directory = Path(superglm.__file__).resolve().parent
    if package_directory.resolve() != (ROOT / "src" / "superglm").resolve():
        raise RuntimeError("The imported SuperGLM package is outside this benchmark checkout")
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "superglm").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(args):
    import resource
    import warnings

    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_info

    import superglm

    package_directory = Path(superglm.__file__).resolve().parent
    package_source_hash = source_fingerprint(package_directory)
    reference_bytes = REFERENCE.read_bytes()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    reference = json.loads(reference_bytes)
    array_hash = hashlib.sha256(REFERENCE_ARRAYS.read_bytes()).hexdigest()
    if array_hash != reference["prediction_archive_sha256"]:
        raise ValueError("Prediction reference archive fingerprint differs from the receipt")
    if args.data is None:
        from sklearn.datasets import fetch_california_housing

        frame = fetch_california_housing(as_frame=True).frame
    else:
        frame = pd.read_parquet(args.data)
    splits, split_hash = prepare_data(frame, expected_fingerprint=reference["data_fingerprint"])
    if split_hash != reference["split_sha256"]:
        raise ValueError("Split fingerprint differs from the frozen reference")
    model = build_model(args.case)
    train, y = splits["train"]
    threadpools_before = threadpool_info()
    profiler = cProfile.Profile() if args.profile else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        try:
            model.fit_reml(train, y, sample_weight=np.ones(len(y)))
        finally:
            elapsed = time.perf_counter() - start
            if profiler is not None:
                profiler.disable()
                profiler.dump_stats(args.output / "fit.prof")
                with (args.output / "profile.txt").open("w") as stream:
                    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
                    stats.print_stats(50)
                    stats.print_callers(25)
                    stats.print_callees(25)
    telemetry = model.training_telemetry()
    if not model.result.converged or not telemetry["reml"]["converged"]:
        raise RuntimeError("The benchmark fit did not converge")
    k, _ = CASES[args.case]
    expected_width = 6 * 19 + 2 * (k - 1) + (k - 1) ** 2
    if len(model.result.beta) != expected_width:
        raise RuntimeError("The fitted basis width differs from the benchmark specification")
    arrays = {}
    losses = {}
    comparison = {}
    with np.load(REFERENCE_ARRAYS) as baseline:
        for split, (features, response) in splits.items():
            prediction = model.predict(features)
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Nonfinite {split} predictions")
            losses[split] = float(np.mean((response - prediction) ** 2))
            arrays[f"{split}_prediction"] = prediction
            arrays[f"{split}_y"] = response
            if split != "train":
                np.testing.assert_array_equal(response, baseline[f"{split}_y"])
                old = baseline[f"{args.case}_{split}"]
                difference = prediction - old
                comparison[split] = {
                    "max_abs_prediction_difference": float(np.max(np.abs(difference))),
                    "rms_prediction_difference": float(np.sqrt(np.mean(difference**2))),
                    "exact_prediction_match": bool(np.array_equal(prediction, old)),
                    "mse_difference": losses[split] - reference["cases"][args.case][f"{split}_mse"],
                }
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    result = {
        "case": args.case,
        "status": "success",
        "profiled": args.profile,
        "fit_seconds": elapsed,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor,
        "coefficient_count_without_intercept": len(model.result.beta),
        "mse": losses,
        "reference_comparison": comparison,
        "warnings": [str(item.message) for item in caught],
        "data_fingerprint": reference["data_fingerprint"],
        "split_sha256": split_hash,
        "imported_package_directory": str(package_directory),
        "package_source_sha256": package_source_hash,
        "reference_json_sha256": reference_hash,
        "reference_predictions_sha256": array_hash,
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "superglm",
                    "numpy",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "threadpoolctl",
                )
            },
            "threadpools_before_fit": threadpools_before,
            "threadpools_after_fit": threadpool_info(),
        },
        "backend_groups": [type(group).__name__ for group in model._dm.group_matrices],
        "telemetry": telemetry,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "case": args.case,
                "fit_seconds": elapsed,
                "mse": losses,
                "reference_comparison": comparison,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="rows30")
    parser.add_argument(
        "--data",
        type=Path,
        help="Original California housing parquet; otherwise use sklearn's cache/fetcher",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Instrument the fit; do not compare this timing with unprofiled timings",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Whole-worker deadline in seconds, including imports and data loading",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.worker:
        worker(args)
        return 0
    if args.output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        args.output = ROOT / ".benchmark-artifacts" / "housing-tensor" / f"{args.case}-{stamp}"
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        args.case,
        "--output",
        str(args.output),
    ]
    if args.data is not None:
        command.extend(["--data", str(args.data.resolve())])
    if args.profile:
        command.append("--profile")
    env = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[variable] = "1"
    result = run_isolated(
        command, log_path=args.output / "worker.log", timeout=args.timeout, env=env
    )
    result.update(
        {
            "case": args.case,
            "profiled": args.profile,
            "timeout_seconds": args.timeout,
            "command": command,
            "output": str(args.output),
        }
    )
    (args.output / "run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
