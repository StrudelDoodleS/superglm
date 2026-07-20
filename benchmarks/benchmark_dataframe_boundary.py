"""Benchmark dataframe-boundary overhead without contaminating timed samples.

Each authoritative timing runs in a fresh subprocess. Data construction, model
construction, warm prediction setup, Python allocation tracing, and kernel-call
instrumentation are outside the timed operation. Generated JSON belongs in a
temporary directory, not in the repository.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tabmat

from superglm import Categorical, Numeric, Spline, SuperGLM
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
from superglm._group_matrix._group_matrix_execution import MatrixExecutionPlan

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX only
    resource = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SCHEMA_VERSION = 1
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
COMPRESSED_TYPES = (
    DiscretizedSSPGroupMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedTensorGroupMatrix,
)


@dataclass(frozen=True)
class PreparedScenario:
    """A constructed scenario whose measured operation is ready to run."""

    model: SuperGLM
    operation: str
    X: object
    y: np.ndarray
    sample_weight: np.ndarray | None = None
    offset: np.ndarray | None = None
    kwargs: Mapping[str, object] | None = None


ScenarioFactory = Callable[[str, float], PreparedScenario]


def _rows(base: int, scale: float, minimum: int) -> int:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scenario scale must be finite and positive")
    return max(minimum, int(round(base * scale)))


def _frame(data: Mapping[str, object], backend: str):
    if backend == "pandas":
        return pd.DataFrame(data)
    if backend == "polars":
        import polars as pl

        return pl.DataFrame(data)
    raise ValueError(f"unsupported benchmark backend: {backend}")


def _ordinary_mixed_fit(backend: str, scale: float) -> PreparedScenario:
    n = _rows(6_000, scale, 720)
    rng = np.random.default_rng(2101)
    numeric = {f"x{index}": rng.normal(size=n) for index in range(4)}
    codes = np.resize(np.arange(180, dtype=np.int64), n)
    rng.shuffle(codes)
    levels = np.asarray([f"level_{code:03d}" for code in codes], dtype=object)
    eta = -0.25 + 0.14 * numeric["x0"] - 0.09 * numeric["x1"] + (codes % 13 - 6) * 0.018
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    X = _frame({**numeric, "category": levels}, backend)
    features = {name: Numeric() for name in numeric}
    features["category"] = Categorical(base="first")
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features=features,
        direct_solve="gram",
    )
    return PreparedScenario(model, "fit", X, y)


def _ordinary_scalar_fit(backend: str, scale: float) -> PreparedScenario:
    n = _rows(60_000, scale, 1_000)
    rng = np.random.default_rng(2102)
    data = {f"x{index}": rng.normal(size=n) for index in range(16)}
    beta = np.linspace(-0.12, 0.15, len(data))
    design = np.column_stack(tuple(data.values()))
    y = 0.4 + design @ beta + rng.normal(scale=0.35, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={name: Numeric() for name in data},
        direct_solve="gram",
    )
    return PreparedScenario(model, "fit", _frame(data, backend), y.astype(np.float64))


def _discrete_four_spline_fit(backend: str, scale: float) -> PreparedScenario:
    n = _rows(10_000, scale, 600)
    rng = np.random.default_rng(2103)
    data = {f"x{index}": rng.uniform(-1.0, 1.0, n) for index in range(4)}
    y = (
        0.2
        + np.sin(np.pi * data["x0"])
        + 0.45 * data["x1"] ** 2
        - 0.3 * data["x2"]
        + 0.15 * np.cos(2.0 * np.pi * data["x3"])
        + rng.normal(scale=0.2, size=n)
    )
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={name: Spline(n_knots=10, penalty="ssp") for name in data},
        discrete=True,
        n_bins=128,
    )
    return PreparedScenario(model, "fit", _frame(data, backend), y.astype(np.float64))


def _spline_reml(backend: str, scale: float) -> PreparedScenario:
    n = _rows(2_500, scale, 500)
    rng = np.random.default_rng(2104)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.normal(size=n)
    y = 0.25 + np.sin(np.pi * x) + 0.18 * z + rng.normal(scale=0.18, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=10, penalty="ssp"), "z": Numeric()},
    )
    return PreparedScenario(
        model,
        "fit_reml",
        _frame({"x": x, "z": z}, backend),
        y.astype(np.float64),
        kwargs={"max_reml_iter": 8, "max_pirls_iter": 40},
    )


def _predict_exact(backend: str, scale: float) -> PreparedScenario:
    n_train = _rows(4_000, scale, 600)
    n_predict = _rows(60_000, scale, 1_000)
    rng = np.random.default_rng(2105)
    train_x = rng.normal(size=n_train)
    train_codes = np.resize(np.arange(140, dtype=np.int64), n_train)
    rng.shuffle(train_codes)
    train_levels = np.asarray([f"level_{code:03d}" for code in train_codes], dtype=object)
    y = rng.poisson(np.exp(-0.2 + 0.12 * train_x + (train_codes % 9 - 4) * 0.02)).astype(np.float64)
    features = {"x": Numeric(), "category": Categorical(base="first")}
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features=features,
        direct_solve="gram",
    )
    model.fit(_frame({"x": train_x, "category": train_levels}, backend), y)
    predict_x = rng.normal(size=n_predict)
    predict_codes = np.resize(np.arange(140, dtype=np.int64), n_predict)
    predict_levels = np.asarray([f"level_{code:03d}" for code in predict_codes], dtype=object)
    X = _frame({"x": predict_x, "category": predict_levels}, backend)
    model.predict(X)
    return PreparedScenario(model, "predict", X, y)


def _predict_fast_discrete(backend: str, scale: float) -> PreparedScenario:
    n_train = _rows(4_000, scale, 600)
    n_predict = _rows(60_000, scale, 1_000)
    rng = np.random.default_rng(2106)
    x1 = rng.uniform(-1.0, 1.0, n_train)
    x2 = rng.uniform(-1.0, 1.0, n_train)
    y = np.sin(np.pi * x1) + 0.4 * x2**2 + rng.normal(scale=0.2, size=n_train)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x1": Spline(n_knots=10, penalty="ssp"),
            "x2": Spline(n_knots=10, penalty="ssp"),
        },
        discrete=True,
        n_bins=128,
    )
    model.fit(_frame({"x1": x1, "x2": x2}, backend), y.astype(np.float64))
    X = _frame(
        {
            "x1": rng.uniform(-1.0, 1.0, n_predict),
            "x2": rng.uniform(-1.0, 1.0, n_predict),
        },
        backend,
    )
    model._predict_fast_discrete(X)
    return PreparedScenario(model, "predict_fast_discrete", X, y.astype(np.float64))


SCENARIOS: dict[str, ScenarioFactory] = {
    "ordinary_mixed_fit": _ordinary_mixed_fit,
    "ordinary_scalar_fit": _ordinary_scalar_fit,
    "discrete_four_spline_fit": _discrete_four_spline_fit,
    "spline_reml": _spline_reml,
    "predict_exact": _predict_exact,
    "predict_fast_discrete": _predict_fast_discrete,
}


def _run_operation(prepared: PreparedScenario):
    kwargs = dict(prepared.kwargs or {})
    if prepared.operation == "fit":
        return prepared.model.fit(
            prepared.X,
            prepared.y,
            sample_weight=prepared.sample_weight,
            offset=prepared.offset,
            **kwargs,
        )
    if prepared.operation == "fit_reml":
        return prepared.model.fit_reml(
            prepared.X,
            prepared.y,
            sample_weight=prepared.sample_weight,
            offset=prepared.offset,
            **kwargs,
        )
    if prepared.operation == "predict":
        return prepared.model.predict(prepared.X, offset=prepared.offset)
    if prepared.operation == "predict_fast_discrete":
        return prepared.model._predict_fast_discrete(prepared.X, offset=prepared.offset)
    raise ValueError(f"unknown benchmark operation: {prepared.operation}")


@contextmanager
def _kernel_counts() -> Iterator[dict[str, int]]:
    calls = {
        "sandwich": 0,
        "matvec": 0,
        "transpose_matvec": 0,
        "stable_grouped_moments": 0,
        "compressed_block_kernels": 0,
    }
    patched: list[tuple[type, str, object]] = []

    def patch_method(owner: type, name: str, counter: str) -> None:
        original = getattr(owner, name)

        def counted(self, *args, **kwargs):
            calls[counter] += 1
            return original(self, *args, **kwargs)

        setattr(owner, name, counted)
        patched.append((owner, name, original))

    for name in ("sandwich", "matvec", "transpose_matvec"):
        patch_method(tabmat.SplitMatrix, name, name)

    original_moments = MatrixExecutionPlan._moments_impl

    def counted_moments(self, *args, **kwargs):
        if not self._ordinary_indices:
            calls["stable_grouped_moments"] += 1
        return original_moments(self, *args, **kwargs)

    MatrixExecutionPlan._moments_impl = counted_moments
    patched.append((MatrixExecutionPlan, "_moments_impl", original_moments))

    for matrix_type in COMPRESSED_TYPES:
        for name in ("gram", "gram_rmatvec", "matvec", "rmatvec"):
            if hasattr(matrix_type, name):
                patch_method(matrix_type, name, "compressed_block_kernels")
    try:
        yield calls
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)


def _rss_peak_bytes() -> int | None:
    if resource is None:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _matrix_metadata(model: SuperGLM) -> dict[str, object]:
    dm = getattr(model, "_dm", None)
    matrices = () if dm is None else dm.group_matrices
    return {
        "group_types": [type(matrix).__name__ for matrix in matrices],
        "group_shapes": [list(matrix.shape) for matrix in matrices],
        "compressed": [isinstance(matrix, COMPRESSED_TYPES) for matrix in matrices],
        "tabmat_built": bool(getattr(dm, "_tabmat_built", False)),
    }


def _numerical_record(prepared: PreparedScenario, operation_result) -> dict[str, object]:
    model = prepared.model
    result = model.result
    prediction = (
        np.asarray(operation_result, dtype=np.float64)
        if prepared.operation.startswith("predict")
        else np.asarray(model.predict(prepared.X, offset=prepared.offset), dtype=np.float64)
    )
    projection = np.sin(np.arange(1, len(prediction) + 1) * 0.6180339887498948)
    reml = getattr(model, "_reml_result", None)
    return {
        "coefficient_values": np.asarray(result.beta, dtype=np.float64).tolist(),
        "coefficient_checksum": float(np.sum(result.beta, dtype=np.float64)),
        "intercept": float(result.intercept),
        "prediction_checksum": float(np.sum(prediction, dtype=np.float64)),
        "prediction_projection": float(np.dot(prediction, projection)),
        "deviance": float(result.deviance),
        "effective_df": float(result.effective_df),
        "phi": float(result.phi),
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "reml_objective": None if reml is None else float(reml.objective),
        "reml_rank": None if reml is None else getattr(reml, "rank", None),
        "lambda_values": (
            {} if reml is None else {str(key): float(value) for key, value in reml.lambdas.items()}
        ),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor()


def _metadata(backend: str) -> dict[str, object]:
    return {
        "backend": backend,
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "pandas": pd.__version__,
        "polars": _package_version("polars"),
        "narwhals": _package_version("narwhals"),
        "tabmat": _package_version("tabmat"),
        "numba": _package_version("numba"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_NAMES},
    }


def _worker_record(name: str, backend: str, scale: float, repeat: int) -> dict[str, object]:
    prepared = SCENARIOS[name](backend, scale)
    rss_before = _rss_peak_bytes()
    started = time.perf_counter_ns()
    result = _run_operation(prepared)
    wall_time_s = (time.perf_counter_ns() - started) / 1_000_000_000.0
    rss_peak = _rss_peak_bytes()

    diagnostic = SCENARIOS[name](backend, scale)
    with _kernel_counts() as kernel_calls:
        tracemalloc.start()
        try:
            _run_operation(diagnostic)
            _, python_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    return {
        "scenario": name,
        "repeat": repeat,
        "wall_time_s": wall_time_s,
        "python_peak_bytes": int(python_peak),
        "rss_before_bytes": rss_before,
        "rss_peak_bytes": rss_peak,
        "rss_delta_bytes": (
            None if rss_before is None or rss_peak is None else max(0, int(rss_peak - rss_before))
        ),
        "kernel_calls": dict(kernel_calls),
        "matrix": _matrix_metadata(prepared.model),
        "numerical": _numerical_record(prepared, result),
    }


def _worker_command(name: str, backend: str, scale: float, repeat: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--scenario",
        name,
        "--backend",
        backend,
        "--scale",
        str(scale),
        "--repeat-index",
        str(repeat),
        "--output",
        str(output),
    ]


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    for name in THREAD_ENVIRONMENT_NAMES:
        environment[name] = "1"
    return environment


def _run_worker(name: str, backend: str, scale: float, repeat: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="superglm-frame-benchmark-") as directory:
        output = Path(directory) / "record.json"
        completed = subprocess.run(
            _worker_command(name, backend, scale, repeat, output),
            cwd=REPO_ROOT,
            env=_worker_environment(),
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"benchmark worker failed for {name}:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return json.loads(output.read_text(encoding="utf-8"))


def _mad(values: Sequence[float]) -> float:
    median = statistics.median(values)
    return float(statistics.median(abs(value - median) for value in values))


def _summaries(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["scenario"]), []).append(sample)
    summaries: dict[str, object] = {}
    for name, records in grouped.items():
        first = records[0]
        for record in records[1:]:
            if record["matrix"] != first["matrix"]:
                raise ValueError(f"matrix metadata changed across repeats for {name}")
            if record["kernel_calls"] != first["kernel_calls"]:
                raise ValueError(f"kernel call counts changed across repeats for {name}")
            if not _numerically_equal(record["numerical"], first["numerical"]):
                raise ValueError(f"numerical results changed across repeats for {name}")
        wall = [float(record["wall_time_s"]) for record in records]
        python_peaks = [int(record["python_peak_bytes"]) for record in records]
        rss_deltas = [
            int(record["rss_delta_bytes"])
            for record in records
            if record["rss_delta_bytes"] is not None
        ]
        summaries[name] = {
            "sample_count": len(records),
            "raw_wall_time_s": wall,
            "median_wall_time_s": float(statistics.median(wall)),
            "mad_wall_time_s": _mad(wall),
            "median_python_peak_bytes": int(statistics.median(python_peaks)),
            "median_rss_delta_bytes": (
                None if not rss_deltas else int(statistics.median(rss_deltas))
            ),
            "kernel_calls": records[-1]["kernel_calls"],
            "matrix": records[-1]["matrix"],
            "numerical": records[-1]["numerical"],
        }
    return summaries


def _selected_scenarios(value: str | None) -> tuple[str, ...]:
    if value is None or value == "all":
        return tuple(SCENARIOS)
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(names) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown benchmark scenarios: {unknown}")
    return names


def _run_suite(args: argparse.Namespace) -> dict[str, object]:
    names = _selected_scenarios(args.scenario)
    warmups = 0 if args.smoke else args.warmups
    repeats = 1 if args.smoke else args.repeats
    scale = min(args.scale, 0.08) if args.smoke else args.scale
    for warmup in range(warmups):
        ordered = names if warmup % 2 == 0 else tuple(reversed(names))
        for name in ordered:
            _run_worker(name, args.backend, scale, -(warmup + 1))

    samples: list[dict[str, object]] = []
    for repeat in range(repeats):
        ordered = names if repeat % 2 == 0 else tuple(reversed(names))
        for name in ordered:
            samples.append(_run_worker(name, args.backend, scale, repeat))
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": _metadata(args.backend),
        "config": {
            "backend": args.backend,
            "scenarios": list(names),
            "warmups": warmups,
            "repeats": repeats,
            "scale": scale,
            "smoke": bool(args.smoke),
        },
        "samples": samples,
        "scenarios": _summaries(samples),
    }


def _numerically_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if not _numerically_equal(a, b):
                return False
        elif isinstance(a, list) and isinstance(b, list):
            if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
                return False
        elif isinstance(a, float | int) and isinstance(b, float | int):
            if not np.isclose(a, b, rtol=1e-10, atol=1e-12):
                return False
        elif a != b:
            return False
    return True


def _compare(left_path: Path, right_path: Path, *, backends: bool) -> int:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    left_scenarios = left["scenarios"]
    right_scenarios = right["scenarios"]
    if left_scenarios.keys() != right_scenarios.keys():
        raise ValueError("benchmark scenario sets differ")
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    for name in left_scenarios:
        before = left_scenarios[name]
        after = right_scenarios[name]
        before_time = float(before["median_wall_time_s"])
        after_time = float(after["median_wall_time_s"])
        change = 100.0 * (after_time / before_time - 1.0)
        threshold = 3.0 if name.startswith("predict") else 5.0
        if not backends and change > threshold:
            failures.append(f"{name}: wall time regressed {change:.2f}%")
        if before["matrix"] != after["matrix"]:
            failures.append(f"{name}: matrix structure or dispatch state changed")
        if before["kernel_calls"] != after["kernel_calls"]:
            failures.append(f"{name}: actual kernel call counts changed")
        if not _numerically_equal(before["numerical"], after["numerical"]):
            failures.append(f"{name}: numerical record changed")
        before_memory = before["median_python_peak_bytes"]
        after_memory = after["median_python_peak_bytes"]
        memory_change = 100.0 * (after_memory / before_memory - 1.0)
        if not backends and after_memory - before_memory > max(1_048_576, 0.05 * before_memory):
            failures.append(f"{name}: traced peak grew {memory_change:.2f}%")
        rows.append(
            {
                "scenario": name,
                "before_s": before_time,
                "after_s": after_time,
                "change_pct": change,
                "python_peak_change_pct": memory_change,
            }
        )
    print(json.dumps({"comparisons": rows, "failures": failures}, indent=2))
    return 1 if failures else 0


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pandas", "polars"), default="pandas")
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--compare-backends", nargs=2, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.compare:
        return _compare(*args.compare, backends=False)
    if args.compare_backends:
        return _compare(*args.compare_backends, backends=True)
    if args.output is None:
        raise SystemExit("--output is required for benchmark runs")
    if args.worker:
        names = _selected_scenarios(args.scenario)
        if len(names) != 1:
            raise SystemExit("benchmark worker requires exactly one --scenario")
        payload = _worker_record(names[0], args.backend, args.scale, args.repeat_index)
    else:
        payload = _run_suite(args)
    _write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
