"""Reproducible fit-state and trace performance benchmark.

The wall-time suite intentionally keeps data generation and model construction
outside the timed fit call.  Each recorded sample runs in a fresh subprocess so
``ru_maxrss`` is a useful native-memory high-water mark.  Authoritative wall
time and RSS come from an uninstrumented fit.  A second fresh model in the same
worker supplies diagnostic ``tracemalloc`` and Tabmat-call counts, so neither
instrument perturbs the timing gate.  ``tracemalloc`` does not include all
NumPy, BLAS, or Tabmat native allocations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform as platform_module
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tabmat

from superglm import Categorical, Numeric, Spline, SuperGLM

try:
    import resource
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    resource = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SCHEMA_VERSION = 1
NUMERICAL_KEYS = (
    "deviance",
    "effective_df",
    "prediction_checksum",
    "prediction_projection",
    "prediction_l2",
    "intercept",
    "phi",
    "log_likelihood",
    "null_log_likelihood",
    "null_deviance",
    "explained_deviance",
    "pearson_chi2",
)
VECTOR_KEYS = ("prediction_values", "beta_values")
THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True)
class PreparedCase:
    """One fully constructed case whose fit call is ready to time."""

    model: SuperGLM
    X: pd.DataFrame
    y: np.ndarray
    sample_weight: np.ndarray | None
    offset: np.ndarray | None
    fit_method: str
    fit_kwargs: Mapping[str, object]


CaseFactory = Callable[[float], PreparedCase]


def _scaled_rows(base: int, scale: float) -> int:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("case scale must be finite and positive")
    return max(80, int(round(base * scale)))


def _smooth_gaussian_data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = rng.normal(size=n)
    y = 0.3 + np.sin(np.pi * x) + 0.25 * z + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"x": x, "z": z}), y.astype(np.float64)


def _dense_fit(scale: float) -> PreparedCase:
    n = _scaled_rows(4_000, scale)
    X, y = _smooth_gaussian_data(n, seed=1001)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Numeric(), "z": Numeric()},
    )
    return PreparedCase(model, X, y, None, None, "fit", {})


def _categorical_fit(scale: float) -> PreparedCase:
    # Keep every level represented even in reduced-size smoke runs so this
    # remains a high-cardinality Tabmat fixture at every supported scale.
    n = max(640, _scaled_rows(6_000, scale))
    rng = np.random.default_rng(1002)
    x = rng.normal(size=n)
    category_codes = np.resize(np.arange(160, dtype=np.int64), n)
    rng.shuffle(category_codes)
    category = np.asarray([f"level_{code:03d}" for code in category_codes], dtype=object)
    category_effect = (category_codes % 11 - 5) * 0.025
    eta = -0.3 + 0.18 * x + category_effect
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    X = pd.DataFrame({"x": x, "category": category})
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        features={"x": Numeric(), "category": Categorical(base="first")},
        direct_solve="gram",
    )
    return PreparedCase(model, X, y, None, None, "fit", {})


def _spline_fit(scale: float) -> PreparedCase:
    n = _scaled_rows(3_000, scale)
    X, y = _smooth_gaussian_data(n, seed=1003)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=8, penalty="ssp"), "z": Numeric()},
    )
    return PreparedCase(model, X, y, None, None, "fit", {})


def _reml_case(scale: float, *, discrete: bool, retain_fit_state: bool, seed: int) -> PreparedCase:
    n = _scaled_rows(1_200, scale)
    X, y = _smooth_gaussian_data(n, seed=seed)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x": Spline(n_knots=8, penalty="ssp"), "z": Numeric()},
        discrete=discrete,
        n_bins=128,
        retain_fit_state=retain_fit_state,
    )
    return PreparedCase(
        model,
        X,
        y,
        None,
        None,
        "fit_reml",
        {"max_reml_iter": 8, "max_pirls_iter": 40},
    )


def _exact_reml(scale: float) -> PreparedCase:
    return _reml_case(scale, discrete=False, retain_fit_state=True, seed=1004)


def _discrete_reml(scale: float) -> PreparedCase:
    return _reml_case(scale, discrete=True, retain_fit_state=True, seed=1005)


def _compact_reml(scale: float) -> PreparedCase:
    return _reml_case(scale, discrete=True, retain_fit_state=False, seed=1006)


CASES: dict[str, CaseFactory] = {
    "dense_fit": _dense_fit,
    "categorical_fit": _categorical_fit,
    "spline_fit": _spline_fit,
    "exact_reml": _exact_reml,
    "discrete_reml": _discrete_reml,
    "compact_reml": _compact_reml,
}


def _execution_order(case_names: Sequence[str], repeat_index: int) -> tuple[str, ...]:
    """Alternate case order to reduce systematic thermal/order bias."""
    names = tuple(case_names)
    return names if repeat_index % 2 == 0 else tuple(reversed(names))


def _matrix_backend_metadata(model: SuperGLM) -> dict[str, object]:
    """Describe the matrix backend already selected during the timed fit."""
    dm = getattr(model, "_dm", None)
    tabmat_built = bool(getattr(dm, "_tabmat_built", False))
    split = getattr(dm, "_tabmat_split", None) if tabmat_built else None
    components = getattr(split, "matrices", ()) if split is not None else ()
    return {
        "tabmat_built": tabmat_built,
        "tabmat_prepared": split is not None,
        "tabmat_split_type": None if split is None else type(split).__name__,
        "tabmat_component_types": [type(component).__name__ for component in components],
    }


@contextmanager
def _count_tabmat_kernel_calls() -> Iterator[dict[str, int]]:
    """Count actual SplitMatrix kernel dispatch in one isolated worker."""
    method_names = ("sandwich", "transpose_matvec")
    originals = {name: getattr(tabmat.SplitMatrix, name) for name in method_names}
    calls = dict.fromkeys(method_names, 0)

    def counted_method(name: str):
        original = originals[name]

        def counted(self, *args, **kwargs):
            calls[name] += 1
            return original(self, *args, **kwargs)

        return counted

    for name in method_names:
        setattr(tabmat.SplitMatrix, name, counted_method(name))
    try:
        yield calls
    finally:
        for name, original in originals.items():
            setattr(tabmat.SplitMatrix, name, original)


def compare_runs(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    numerical_rtol: float = 1e-10,
    numerical_atol: float = 1e-12,
) -> list[str]:
    """Return numerical-fidelity failures between two case records."""
    failures: list[str] = []
    for key in NUMERICAL_KEYS:
        if key not in before or key not in after:
            failures.append(f"{key}: missing from benchmark record")
            continue
        before_value = float(before[key])
        after_value = float(after[key])
        if not np.isclose(
            before_value,
            after_value,
            rtol=numerical_rtol,
            atol=numerical_atol,
        ):
            failures.append(f"{key}: {before_value!r} != {after_value!r}")

    for key in VECTOR_KEYS:
        if key not in before or key not in after:
            failures.append(f"{key}: missing from benchmark record")
            continue
        before_values = np.asarray(before[key], dtype=np.float64)
        after_values = np.asarray(after[key], dtype=np.float64)
        if before_values.shape != after_values.shape:
            failures.append(
                f"{key}: shapes differ: {before_values.shape!r} != {after_values.shape!r}"
            )
        elif not np.allclose(
            before_values,
            after_values,
            rtol=numerical_rtol,
            atol=numerical_atol,
        ):
            max_error = float(np.max(np.abs(before_values - after_values)))
            failures.append(f"{key}: pointwise values differ (max_abs_error={max_error!r})")

    for key in ("reml_objective",):
        if key not in before or key not in after:
            failures.append(f"{key}: missing from benchmark record")
            continue
        before_value = before[key]
        after_value = after[key]
        if before_value is None and after_value is None:
            continue
        if (
            before_value is None
            or after_value is None
            or not np.isclose(
                float(before_value),
                float(after_value),
                rtol=numerical_rtol,
                atol=numerical_atol,
            )
        ):
            failures.append(f"{key}: {before_value!r} != {after_value!r}")

    for key in ("converged", "reml_converged", "overall_converged"):
        if key not in before or key not in after:
            failures.append(f"{key}: missing from benchmark record")
        elif before[key] != after[key]:
            failures.append(f"{key}: {before[key]!r} != {after[key]!r}")

    if "n_obs" not in before or "n_obs" not in after:
        failures.append("n_obs: missing from benchmark record")
    elif int(before["n_obs"]) != int(after["n_obs"]):
        failures.append(f"n_obs: {before['n_obs']!r} != {after['n_obs']!r}")

    before_lambdas = before.get("lambda_values")
    after_lambdas = after.get("lambda_values")
    if not isinstance(before_lambdas, Mapping) or not isinstance(after_lambdas, Mapping):
        failures.append("lambda_values: missing or not a mapping")
    elif set(before_lambdas) != set(after_lambdas):
        failures.append(
            f"lambda_values: key sets differ: {sorted(before_lambdas)} != {sorted(after_lambdas)}"
        )
    else:
        for name in sorted(before_lambdas):
            before_value = float(before_lambdas[name])
            after_value = float(after_lambdas[name])
            if not np.isclose(
                before_value,
                after_value,
                rtol=numerical_rtol,
                atol=numerical_atol,
            ):
                failures.append(f"lambda_values.{name}: {before_value!r} != {after_value!r}")
    return failures


def _validate_worker_record(record: Mapping[str, object]) -> None:
    required = {
        "case",
        "repeat",
        "order",
        "wall_time_s",
        "python_peak_bytes",
        "rss_before_fit_bytes",
        "rss_peak_bytes",
        "rss_peak_delta_bytes",
        "deviance",
        "effective_df",
        "prediction_checksum",
        "prediction_projection",
        "prediction_l2",
        "prediction_values",
        "beta_values",
        "intercept",
        "phi",
        "log_likelihood",
        "null_log_likelihood",
        "null_deviance",
        "explained_deviance",
        "pearson_chi2",
        "n_obs",
        "reml_objective",
        "lambda_values",
        "n_iter",
        "converged",
        "reml_converged",
        "overall_converged",
        "reml_diagnostics",
        "profile",
        "matrix_backend",
        "tabmat_kernel_calls",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"worker record missing required field: {', '.join(missing)}")

    finite_fields = (
        "wall_time_s",
        "python_peak_bytes",
        "deviance",
        "effective_df",
        "prediction_checksum",
        "prediction_projection",
        "prediction_l2",
        "intercept",
        "phi",
        "log_likelihood",
        "null_log_likelihood",
        "null_deviance",
        "explained_deviance",
        "pearson_chi2",
        "n_iter",
    )
    for key in finite_fields:
        try:
            value = float(record[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"worker field {key} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"worker field {key} must be finite")

    for key in ("rss_before_fit_bytes", "rss_peak_bytes", "rss_peak_delta_bytes"):
        rss_value = record[key]
        if rss_value is not None and (
            not math.isfinite(float(rss_value)) or float(rss_value) < 0.0
        ):
            raise ValueError(f"worker field {key} must be finite and nonnegative")

    reml_objective = record["reml_objective"]
    if reml_objective is not None and not math.isfinite(float(reml_objective)):
        raise ValueError("worker field reml_objective must be finite when present")
    lambda_values = record["lambda_values"]
    if not isinstance(lambda_values, Mapping) or any(
        not math.isfinite(float(value)) for value in lambda_values.values()
    ):
        raise ValueError("worker field lambda_values must be a finite numeric mapping")
    reml_diagnostics = record["reml_diagnostics"]
    if record["reml_converged"] is None:
        if reml_diagnostics is not None or reml_objective is not None or lambda_values:
            raise ValueError("ordinary-fit record contains REML-only fidelity fields")
    elif not isinstance(reml_diagnostics, Mapping):
        raise ValueError("REML record must contain reml_diagnostics")
    expected_overall = bool(record["converged"]) and (
        record["reml_converged"] is None or bool(record["reml_converged"])
    )
    if bool(record["overall_converged"]) != expected_overall:
        raise ValueError("worker field overall_converged is inconsistent")
    for key in VECTOR_KEYS:
        values = np.asarray(record[key], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"worker field {key} must be a finite numeric vector")
    try:
        n_obs = int(record["n_obs"])
    except (TypeError, ValueError) as exc:
        raise ValueError("worker field n_obs must be a positive integer") from exc
    if n_obs < 1 or float(n_obs) != float(record["n_obs"]):
        raise ValueError("worker field n_obs must be a positive integer")


def _median_numeric(records: Sequence[Mapping[str, object]], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def _summarize_samples(samples: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["case"]), []).append(sample)

    summaries: dict[str, dict[str, object]] = {}
    for case, records in grouped.items():

        def median_optional(key: str) -> int | None:
            values = [float(record[key]) for record in records if record.get(key) is not None]
            return int(statistics.median(values)) if values else None

        def median_optional_float(key: str) -> float | None:
            values = [float(record[key]) for record in records if record.get(key) is not None]
            return float(statistics.median(values)) if values else None

        lambda_key_sets = {frozenset(record["lambda_values"]) for record in records}
        if len(lambda_key_sets) != 1:
            raise ValueError(f"inconsistent lambda keys for benchmark case {case}")
        lambda_keys = next(iter(lambda_key_sets))
        lambda_values = {
            name: float(
                statistics.median(float(record["lambda_values"][name]) for record in records)
            )
            for name in sorted(lambda_keys)
        }
        reml_convergence = [
            bool(record["reml_converged"])
            for record in records
            if record["reml_converged"] is not None
        ]

        def stable_vector(key: str) -> list[float]:
            first = np.asarray(records[0][key], dtype=np.float64)
            for record in records[1:]:
                candidate = np.asarray(record[key], dtype=np.float64)
                if first.shape != candidate.shape or not np.allclose(
                    first,
                    candidate,
                    rtol=1e-12,
                    atol=1e-14,
                ):
                    raise ValueError(f"inconsistent {key} across benchmark repeats for {case}")
            return first.tolist()

        summary: dict[str, object] = {
            "sample_count": len(records),
            "median_wall_time_s": _median_numeric(records, "wall_time_s"),
            "median_python_peak_bytes": int(_median_numeric(records, "python_peak_bytes")),
            "median_rss_before_fit_bytes": median_optional("rss_before_fit_bytes"),
            "median_rss_peak_bytes": median_optional("rss_peak_bytes"),
            "median_rss_peak_delta_bytes": median_optional("rss_peak_delta_bytes"),
            "deviance": _median_numeric(records, "deviance"),
            "effective_df": _median_numeric(records, "effective_df"),
            "prediction_checksum": _median_numeric(records, "prediction_checksum"),
            "prediction_projection": _median_numeric(records, "prediction_projection"),
            "prediction_l2": _median_numeric(records, "prediction_l2"),
            "prediction_values": stable_vector("prediction_values"),
            "beta_values": stable_vector("beta_values"),
            "intercept": _median_numeric(records, "intercept"),
            "phi": _median_numeric(records, "phi"),
            "log_likelihood": _median_numeric(records, "log_likelihood"),
            "null_log_likelihood": _median_numeric(records, "null_log_likelihood"),
            "null_deviance": _median_numeric(records, "null_deviance"),
            "explained_deviance": _median_numeric(records, "explained_deviance"),
            "pearson_chi2": _median_numeric(records, "pearson_chi2"),
            "n_obs": int(_median_numeric(records, "n_obs")),
            "reml_objective": median_optional_float("reml_objective"),
            "lambda_values": lambda_values,
            "median_n_iter": _median_numeric(records, "n_iter"),
            "converged": all(bool(record["converged"]) for record in records),
            "reml_converged": all(reml_convergence) if reml_convergence else None,
            "overall_converged": all(bool(record["overall_converged"]) for record in records),
            "all_converged": all(bool(record["overall_converged"]) for record in records),
            "all_reml_converged": all(reml_convergence) if reml_convergence else None,
        }
        summaries[case] = summary
    return summaries


def _validate_suite_quality(case_summaries: Mapping[str, Mapping[str, object]]) -> None:
    """Reject a baseline that would normalize failed solver or REML fits."""
    for case, summary in case_summaries.items():
        if not bool(summary.get("overall_converged")):
            raise ValueError(f"benchmark case {case} did not converge")
        reml_converged = summary.get("reml_converged")
        if reml_converged is not None:
            if not bool(reml_converged):
                raise ValueError(f"benchmark REML case {case} did not converge")
            objective = summary.get("reml_objective")
            if objective is None or not math.isfinite(float(objective)):
                raise ValueError(f"benchmark REML case {case} has no finite objective")


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _git_dirty() -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(completed.stdout.strip())


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform_module.processor()


def _version_metadata(*, environment: Mapping[str, str] | None = None) -> dict[str, object]:
    effective_environment: Mapping[str, str] = os.environ if environment is None else environment
    try:
        tabmat_version = importlib.metadata.version("tabmat")
    except importlib.metadata.PackageNotFoundError:
        tabmat_version = "not-installed"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tabmat": tabmat_version,
        "platform": sys.platform,
        "os_release": platform_module.platform(),
        "machine": platform_module.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "pythonhashseed": effective_environment.get("PYTHONHASHSEED"),
        "thread_environment": {
            name: effective_environment.get(name) for name in THREAD_ENVIRONMENT_NAMES
        },
    }


def _rss_peak_bytes() -> int | None:
    if resource is None:
        return None
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # Linux and the BSDs report KiB; macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def _run_worker_case(
    case: str,
    *,
    repeat: int,
    order: int,
    case_scale: float,
) -> dict[str, object]:
    if case not in CASES:
        raise ValueError(f"unknown benchmark fixture: {case}")

    prepared = CASES[case](case_scale)
    fit_callable = getattr(prepared.model, prepared.fit_method)

    rss_before_fit_bytes = _rss_peak_bytes()
    started = time.perf_counter()
    fit_callable(
        prepared.X,
        prepared.y,
        sample_weight=prepared.sample_weight,
        offset=prepared.offset,
        **prepared.fit_kwargs,
    )
    wall_time_s = time.perf_counter() - started
    rss_peak_bytes = _rss_peak_bytes()
    rss_peak_delta_bytes = (
        None
        if rss_before_fit_bytes is None or rss_peak_bytes is None
        else max(0, rss_peak_bytes - rss_before_fit_bytes)
    )

    matrix_backend = _matrix_backend_metadata(prepared.model)
    profile = _json_safe(getattr(prepared.model, "_reml_profile", None) or {})
    reml_result = getattr(prepared.model, "_reml_result", None)
    result = prepared.model.result
    fit_stats = prepared.model._fit_stats

    prediction = np.asarray(
        prepared.model.predict(prepared.X, offset=prepared.offset),
        dtype=np.float64,
    )
    projection_index = np.arange(1, prediction.size + 1, dtype=np.float64)
    projection_weights = np.sin(projection_index * 0.6180339887498948)
    if reml_result is None:
        reml_converged = None
        lambda_values: dict[str, float] = {}
        reml_diagnostics = None
    else:
        reml_converged = bool(reml_result.converged)
        lambda_values = {
            str(name): float(value) for name, value in sorted(reml_result.lambdas.items())
        }
        reml_diagnostics = {
            "n_reml_iter": int(reml_result.n_reml_iter),
            "lambda_history": _json_safe(reml_result.lambda_history),
            "objective_history": _json_safe(reml_result.objective_history),
            "inner_iter_history": _json_safe(reml_result.inner_iter_history),
            "scop_fisher_fallbacks": int(reml_result.scop_fisher_fallbacks),
        }

    # Python allocation tracing is deliberately performed on a second fresh
    # model so it cannot perturb the production wall-time measurement above.
    allocation_prepared = CASES[case](case_scale)
    allocation_fit = getattr(allocation_prepared.model, allocation_prepared.fit_method)
    with _count_tabmat_kernel_calls() as tabmat_kernel_calls:
        tracemalloc.start()
        try:
            allocation_fit(
                allocation_prepared.X,
                allocation_prepared.y,
                sample_weight=allocation_prepared.sample_weight,
                offset=allocation_prepared.offset,
                **allocation_prepared.fit_kwargs,
            )
            _, python_peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    record: dict[str, object] = {
        "case": case,
        "repeat": int(repeat),
        "order": int(order),
        "wall_time_s": float(wall_time_s),
        "python_peak_bytes": int(python_peak_bytes),
        "rss_before_fit_bytes": rss_before_fit_bytes,
        "rss_peak_bytes": rss_peak_bytes,
        "rss_peak_delta_bytes": rss_peak_delta_bytes,
        "deviance": float(result.deviance),
        "effective_df": float(result.effective_df),
        "prediction_checksum": float(np.sum(prediction, dtype=np.float64)),
        "prediction_projection": float(np.dot(prediction, projection_weights)),
        "prediction_l2": float(np.dot(prediction, prediction)),
        "prediction_values": prediction.tolist(),
        "beta_values": np.asarray(result.beta, dtype=np.float64).tolist(),
        "intercept": float(result.intercept),
        "phi": float(result.phi),
        "log_likelihood": float(fit_stats.log_likelihood),
        "null_log_likelihood": float(fit_stats.null_log_likelihood),
        "null_deviance": float(fit_stats.null_deviance),
        "explained_deviance": float(fit_stats.explained_deviance),
        "pearson_chi2": float(fit_stats.pearson_chi2),
        "n_obs": int(fit_stats.n_obs),
        "reml_objective": (None if reml_result is None else float(reml_result.objective)),
        "lambda_values": lambda_values,
        "n_iter": int(result.n_iter),
        "n_reml_iter": (None if reml_result is None else int(reml_result.n_reml_iter)),
        "converged": bool(result.converged),
        "reml_converged": reml_converged,
        "overall_converged": bool(result.converged) and (reml_converged is None or reml_converged),
        "reml_diagnostics": reml_diagnostics,
        "profile": profile,
        "matrix_backend": matrix_backend,
        "tabmat_kernel_calls": dict(tabmat_kernel_calls),
        "tabmat_kernel_calls_measurement": "python-memory-diagnostic-fit",
        "solver_calls": None,
        "evaluation_counts": {},
        "metadata": _version_metadata(),
    }
    _validate_worker_record(record)
    return record


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _worker_command(
    case: str,
    *,
    repeat: int,
    order: int,
    case_scale: float,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--fixture",
        case,
        "--repeat-index",
        str(repeat),
        "--order-index",
        str(order),
        "--case-scale",
        str(case_scale),
        "--output",
        str(output),
    ]


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    for name in THREAD_ENVIRONMENT_NAMES:
        environment[name] = "1"
    return environment


def _run_worker_subprocess(
    case: str,
    *,
    repeat: int,
    order: int,
    case_scale: float,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="superglm-benchmark-") as tmp:
        output = Path(tmp) / "worker.json"
        completed = subprocess.run(
            _worker_command(
                case,
                repeat=repeat,
                order=order,
                case_scale=case_scale,
                output=output,
            ),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_worker_environment(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark worker failed for {case}:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        record = json.loads(output.read_text(encoding="utf-8"))
    _validate_worker_record(record)
    return record


def _comparison_report(
    baseline: Mapping[str, object],
    candidate_cases: Mapping[str, object],
) -> dict[str, object]:
    baseline_cases = baseline.get("cases")
    if not isinstance(baseline_cases, Mapping):
        raise ValueError("comparison file must contain a 'cases' mapping")
    baseline_case_names = set(baseline_cases)
    candidate_case_names = set(candidate_cases)
    if baseline_case_names != candidate_case_names:
        raise ValueError(
            "comparison case sets differ: "
            f"{sorted(baseline_case_names)} != {sorted(candidate_case_names)}"
        )
    report: dict[str, object] = {}
    for case in sorted(baseline_case_names):
        before = baseline_cases[case]
        after = candidate_cases[case]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError(f"comparison case {case} must be a mapping")
        report[case] = {"numerical_failures": compare_runs(before, after)}
    return report


def _validate_comparison_context(
    baseline: Mapping[str, object],
    candidate_metadata: Mapping[str, object],
    candidate_config: Mapping[str, object],
) -> None:
    """Reject performance comparisons made across incompatible environments."""
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("comparison schema_version does not match this harness")
    if baseline.get("suite") != "wall-time":
        raise ValueError("comparison suite must be 'wall-time'")
    baseline_config = baseline.get("config")
    baseline_metadata = baseline.get("metadata")
    if not isinstance(baseline_config, Mapping) or not isinstance(baseline_metadata, Mapping):
        raise ValueError("comparison file must contain config and metadata mappings")

    for key in (
        "warmups",
        "repeats",
        "case_scale",
        "case_names",
        "measurement_contract",
        "optimization_targets",
    ):
        if baseline_config.get(key) != candidate_config.get(key):
            raise ValueError(
                f"comparison {key} differs: {baseline_config.get(key)!r} != "
                f"{candidate_config.get(key)!r}"
            )

    metadata_keys = (
        "python",
        "numpy",
        "pandas",
        "tabmat",
        "platform",
        "os_release",
        "machine",
        "cpu_model",
        "thread_environment",
        "pythonhashseed",
    )
    for key in metadata_keys:
        if baseline_metadata.get(key) != candidate_metadata.get(key):
            raise ValueError(
                f"comparison metadata {key} differs: {baseline_metadata.get(key)!r} != "
                f"{candidate_metadata.get(key)!r}"
            )


def _strip_fidelity_vectors(
    samples: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Remove repeat-duplicate vectors after they are frozen in case summaries."""
    return [
        {key: value for key, value in sample.items() if key not in VECTOR_KEYS}
        for sample in samples
    ]


def _run_wall_time_suite(args: argparse.Namespace) -> dict[str, object]:
    case_names = tuple(CASES)
    for warmup in range(args.warmups):
        for order, case in enumerate(_execution_order(case_names, warmup)):
            _run_worker_subprocess(
                case,
                repeat=-(warmup + 1),
                order=order,
                case_scale=args.case_scale,
            )

    samples: list[dict[str, object]] = []
    execution_orders: list[list[str]] = []
    for repeat in range(args.repeats):
        ordered_cases = _execution_order(case_names, repeat)
        execution_orders.append(list(ordered_cases))
        for order, case in enumerate(ordered_cases):
            samples.append(
                _run_worker_subprocess(
                    case,
                    repeat=repeat,
                    order=order,
                    case_scale=args.case_scale,
                )
            )

    case_summaries = _summarize_samples(samples)
    _validate_suite_quality(case_summaries)
    metadata = _version_metadata(environment=_worker_environment())
    config: dict[str, object] = {
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "case_scale": float(args.case_scale),
        "case_names": list(case_names),
        "execution_orders": execution_orders,
        "measurement_contract": {
            "authoritative": ["wall_time_s", "rss_peak_bytes", "rss_peak_delta_bytes"],
            "diagnostic": ["python_peak_bytes", "profile", "tabmat_kernel_calls"],
            "phase_timings_are_regression_gates": False,
        },
        "optimization_targets": {
            "categorical_fit": "dispatch prepared Tabmat kernels instead of dense centering",
        },
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "suite": "wall-time",
        "metadata": metadata,
        "config": config,
        "samples": _strip_fidelity_vectors(samples),
        "cases": case_summaries,
    }
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        _validate_comparison_context(baseline, metadata, config)
        payload["comparison"] = _comparison_report(baseline, case_summaries)
    return payload


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("wall-time", "trace-overhead"), default="wall-time")
    parser.add_argument("--warmups", type=_positive_int, default=2)
    parser.add_argument("--repeats", type=_positive_int, default=10)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fixture", choices=tuple(CASES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--case-scale", type=float, default=1.0)
    parser.add_argument("--repeat-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--order-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker:
        if args.fixture is None:
            raise SystemExit("--worker requires --fixture")
        payload = _run_worker_case(
            args.fixture,
            repeat=args.repeat_index,
            order=args.order_index,
            case_scale=args.case_scale,
        )
    else:
        if args.suite != "wall-time":
            raise SystemExit("trace-overhead suite is added by the authoritative trace plan")
        if args.repeats < 1:
            raise SystemExit("--repeats must be at least 1 for a parent run")
        payload = _run_wall_time_suite(args)
    _write_json(args.output, payload)
    comparison = payload.get("comparison")
    if isinstance(comparison, Mapping) and any(
        bool(details.get("numerical_failures"))
        for details in comparison.values()
        if isinstance(details, Mapping)
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
