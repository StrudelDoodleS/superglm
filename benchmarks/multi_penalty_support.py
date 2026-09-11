"""Compare complete shared-penalty fits in fresh baseline/candidate processes.

Run with ``uv run python -m benchmarks.multi_penalty_support --case gaussian
--out receipt.json --label candidate``. Select frozen package source through
PYTHONPATH. Wall time is omitted unless --measure-time is explicitly supplied
after arranging exclusive execution on a quiet machine.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import warnings
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from benchmarks import rank_deficient_complete_fit
from benchmarks.rank_deficient_complete_fit import (
    _DispatchSampler,
    _git_state,
    _native_pool_receipt,
)
from threadpoolctl import threadpool_limits

import superglm
from superglm import GammaLS, GaussianLS, Predictor, Spline, SuperGLM, SuperLSS
from superglm.features.interaction import TensorInteraction
from superglm.reml import multi_penalty, penalty_algebra


def _source_identity() -> dict:
    package = Path(superglm.__file__).resolve().parent
    files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    commit, dirty = _git_state()
    return {
        "package_path": str(package),
        "source_digest": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "source_file_count": len(files),
        "git_commit": commit,
        "git_dirty": dirty,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_files": {
            "benchmarks.rank_deficient_complete_fit": {
                "path": str(Path(rank_deficient_complete_fit.__file__).resolve()),
                "sha256": hashlib.sha256(
                    Path(rank_deficient_complete_fit.__file__).read_bytes()
                ).hexdigest(),
            }
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


@contextmanager
def _kernel_dispatch(sampler=None):
    """Count existing entry points and aliases without retaining result arrays."""
    counts = Counter()
    ranks = Counter()
    errors = Counter()
    support_shapes = Counter()
    compiled_results = Counter()
    compiled_signatures = []
    patches = []
    targets = [
        (multi_penalty, "similarity_transform_logdet"),
        (multi_penalty, "_evaluate_penalty_support"),
        (multi_penalty, "_evaluate_penalty_summary"),
        (multi_penalty, "_reference_correct_once"),
        (multi_penalty, "_triangular_solve"),
        (multi_penalty, "_compensated_dot"),
        (multi_penalty, "_dot2_value"),
        (multi_penalty, "_inverse_gram_enclosed"),
        (multi_penalty, "_positive_product"),
        (multi_penalty, "_positive_native_product"),
        (penalty_algebra, "evaluate_tensor_pair_logdet_summaries"),
    ]

    def wrapper(original, name):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            counts[name] += 1
            if sampler is not None:
                sampler.sample(f"{name}:call")
            if name in {"_evaluate_penalty_support", "_evaluate_penalty_summary"}:
                support = args[0]
                rows = sum(len(root) for root in support.component_roots)
                support_shapes[
                    f"width={support.Q_plus.shape[0]},rank={support.rank},rows={rows}"
                ] += 1
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                errors[f"{name}:{type(exc).__name__}"] += 1
                raise
            finally:
                if sampler is not None:
                    sampler.sample(f"{name}:return")
            if hasattr(result, "rank"):
                ranks[f"{name}:{result.rank}"] += 1
            if name == "_dot2_value":
                compiled_results["native" if result[1] else "fallback_requested"] += 1
            return result

        return wrapped

    for owner, name in targets:
        original = getattr(owner, name, None)
        if original is None:
            continue
        replacement = wrapper(original, name)
        for module_name, module in tuple(sys.modules.items()):
            if not module_name.startswith("superglm.") or module is None:
                continue
            for attribute, value in tuple(vars(module).items()):
                if value is original:
                    patches.append((module, attribute, original))
                    setattr(module, attribute, replacement)
    try:
        yield {
            "calls": counts,
            "returned_ranks": ranks,
            "errors": errors,
            "support_shapes": support_shapes,
            "compiled_dot2_results": compiled_results,
            "compiled_dot2_signatures": compiled_signatures,
        }
    finally:
        compiled = next((original for _, name, original in patches if name == "_dot2_value"), None)
        if compiled is not None:
            compiled_signatures.extend(map(str, compiled.nopython_signatures))
        for module, attribute, original in reversed(patches):
            setattr(module, attribute, original)


def _fixture(case: str, discrete: bool, *, tensor_rows: int = 2000):
    rng = np.random.default_rng(248)
    if case == "scalar_tensor":
        x, z = rng.uniform(-1.0, 1.0, (2, tensor_rows))
        frame = pd.DataFrame({"x": x, "z": z})
        y = rng.poisson(np.exp(0.2 + 0.6 * np.sin(np.pi * x) + 0.4 * z + 0.5 * x * z)).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=1.0,
            features={"x": Spline(n_knots=12), "z": Spline(n_knots=12)},
            interactions=[("x", "z")],
            discrete=discrete,
            n_bins=512,
        )
        return model, frame, y, None
    x = np.linspace(-1.0, 1.0, 384)
    frame = pd.DataFrame({"x": x})
    if case == "scalar":
        y = rng.poisson(np.exp(0.5 + np.sin(np.pi * x))).astype(float)
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            spline_penalty=1.0,
            features={"x": Spline(kind="cr", n_knots=8, m=(1, 2, 3))},
            discrete=discrete,
            n_bins=512,
        )
        return model, frame, y, None
    gamma = case == "gamma"
    first = "mean" if gamma else "location"
    if gamma:
        mean = np.exp(0.4 + 0.8 * x)
        cv = np.exp(-0.8 + 0.25 * x)
        y = mean * rng.gamma(shape=1 / cv**2, scale=cv**2)
    else:
        y = 0.4 + 0.8 * x + np.exp(-0.4 + 0.35 * x) * rng.normal(size=x.size)
    if case == "tensor":
        frame["z"] = rng.permutation(x)
        y += 0.3 * np.sin(np.pi * frame.z.to_numpy())
        predictors = [
            Predictor(
                name,
                {"x": Spline(n_knots=4), "z": Spline(n_knots=4)},
                interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
            )
            for name in (first, "scale")
        ]
        lambdas = {
            f"{name}:{component}": 1.0
            for name in (first, "scale")
            for component in ("x#wiggle", "z#wiggle", "x:z#margin_x", "x:z#margin_z")
        }
    else:
        predictors = [
            Predictor(first, {"x": Spline(kind="cr", k=6, m=(1, 2))}),
            Predictor("scale", {"x": Spline(kind="cr", k=5, m=(1, 2))}),
        ]
        lambdas = {
            f"{first}:x#d1": 3.0,
            f"{first}:x#d2": 2.0,
            "scale:x#d1": 1.0,
            "scale:x#d2": 2.0,
        }
    from superglm.distributional.binding import _bind_predictor_template

    bound_family = GammaLS() if gamma else GaussianLS(scale_floor=0.0)
    model = SuperLSS(
        bound_family,
        *(_bind_predictor_template(bound_family, predictor) for predictor in predictors),
        discrete=discrete,
        n_bins=512,
    )
    return model, frame, y, lambdas


def _fit_outputs(model, frame, scalar):
    if scalar:
        result = model.result
        rank = result.rank_info
        smoothing = getattr(model, "_reml_result", None)
        return {
            "coefficient_converged": result.converged,
            "iterations": result.n_iter,
            "deviance": result.deviance,
            "effective_df": result.effective_df,
            "prediction": model.predict(frame),
            "coefficients": result.beta,
            "data_rank": rank.data.rank,
            "coefficient_rank": rank.coefficient.rank,
            "data_method": rank.data.method,
            "coefficient_method": rank.coefficient.method,
            "design_class": type(model._dm).__name__,
            "group_matrix_classes": dict(
                Counter(type(group).__name__ for group in model._dm.group_matrices)
            ),
            "smoothing_converged": None if smoothing is None else smoothing.converged,
            "smoothing_iterations": None if smoothing is None else smoothing.n_reml_iter,
            "smoothing_reason": None if smoothing is None else smoothing.termination_reason,
            "objective": None if smoothing is None else smoothing.objective,
            "curvature_source": None if smoothing is None else smoothing.curvature_source,
            "lambda_history": None if smoothing is None else smoothing.lambda_history,
            "objective_history": None if smoothing is None else smoothing.objective_history,
            "inner_iter_history": None if smoothing is None else smoothing.inner_iter_history,
            "lambdas": getattr(model, "_reml_lambdas", None),
        }
    state = model._require_fitted().fit_state
    fit = state.solver_result
    smoothing = state.smoothing
    fits = (fit,) if smoothing is None else smoothing.coefficient_fits
    return {
        "coefficient_converged": fit.converged,
        "coefficient_reason": fit.convergence_reason,
        "coefficient_score_relative": fit.score_relative,
        "coefficient_tolerance": fit.config.tolerance,
        "coefficient_rank": fit.terminal_rank.rank,
        "coefficient_count": len(fit.coefficients),
        "rank_method": fit.terminal_rank.method,
        "execution_backend": fit.execution_backend_identifier,
        "resolved_chunk_size": fit.resolved_chunk_size,
        "curvature_source": fit.terminal_curvature,
        "n_coefficient_fits": len(fits),
        "n_inner_iterations": sum(item.iterations for item in fits),
        "n_fallback_fits": sum(item.terminal_curvature.fallback_count > 0 for item in fits),
        "theta": fit.theta,
        "coefficients": fit.coefficients,
        "log_likelihood": fit.log_likelihood,
        "penalized_optimizing_log_likelihood": fit.penalized_optimizing_log_likelihood,
        "lambdas": dict(state.lambdas),
        "initial_lambdas": None if smoothing is None else dict(smoothing.initial_lambdas),
        "smoothing_converged": None if smoothing is None else smoothing.converged,
        "smoothing_reason": None if smoothing is None else smoothing.convergence_reason,
        "objective": None if smoothing is None else smoothing.objective,
        "terminal_gradient": None if smoothing is None else smoothing.terminal_gradient,
        "penalty_component_ranks": {item.name: item.rank for item in state.layout.penalties},
    }


def _json_value(value):
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("gaussian", "gamma", "scalar", "tensor", "scalar_tensor"), required=True
    )
    parser.add_argument("--mode", choices=("fixed", "reml"), default="reml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--measure-time", action="store_true")
    parser.add_argument("--tensor-rows", type=int, default=2000)
    args = parser.parse_args()
    if args.tensor_rows < 1:
        parser.error("--tensor-rows must be positive")
    if args.tensor_rows != 2000 and args.case != "scalar_tensor":
        parser.error("--tensor-rows applies only to --case scalar_tensor")
    load_before = os.getloadavg()
    cores = len(os.sched_getaffinity(0))
    if args.measure_time and load_before[0] > 2 * cores:
        parser.error("machine load exceeds the repository wall-time threshold")
    model, frame, y, lambdas = _fixture(args.case, args.discrete, tensor_rows=args.tensor_rows)
    data_bytes = frame.to_numpy().tobytes() + y.tobytes()
    receipt = {
        "label": args.label,
        "case": args.case,
        "mode": args.mode,
        "discrete_requested": args.discrete,
        "rows": len(y),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "provenance": _source_identity(),
        "load_before": load_before,
        "available_cores": cores,
        "wall_time_status": "measured; see comparison execution conditions"
        if args.measure_time
        else "unmeasured",
        "peak_rss_measures_fits": 1,
    }
    sampler = None if args.measure_time else _DispatchSampler()
    scalar = args.case in {"scalar", "scalar_tensor"}
    dispatch_context = (
        nullcontext({"status": "not instrumented; collect a separate unmeasured receipt"})
        if args.measure_time
        else _kernel_dispatch(sampler)
    )
    with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=1):
        with sampler if sampler is not None else nullcontext(), dispatch_context as dispatch:
            started = time.perf_counter() if args.measure_time else None
            cpu_started = time.process_time() if args.measure_time else None
            if scalar:
                if args.mode == "reml":
                    model.fit_reml(frame, y, max_reml_iter=40, reml_tol=1e-6)
                else:
                    model.fit(frame, y)
            elif args.mode == "reml":
                model.fit_reml(
                    frame,
                    y,
                    initial_lambda=1.0,
                    max_reml_iter=60,
                    outer="efs+newton",
                    practical_reml=False,
                )
            else:
                model.fit(frame, y, lambdas=lambdas, max_inner_iter=150, inner_tol=1e-10)
            elapsed = None if started is None else time.perf_counter() - started
            cpu_elapsed = None if cpu_started is None else time.process_time() - cpu_started
        receipt["kernel_dispatch"] = dispatch
        receipt["warnings"] = [str(item.message) for item in recorded]
    receipt["fit_seconds"] = elapsed if args.measure_time else None
    receipt["fit_cpu_seconds"] = cpu_elapsed if args.measure_time else None
    receipt["load_after"] = os.getloadavg()
    rss_unit = 1024.0**2 if sys.platform == "darwin" else 1024.0
    receipt["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_unit
    receipt.update(_native_pool_receipt(sampler))
    receipt["outputs"] = _fit_outputs(model, frame, scalar)
    if not scalar:
        receipt["fit_diagnostics"] = model.diagnose().to_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_value(receipt), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(args.out), "source": receipt["provenance"]["source_digest"]}))


if __name__ == "__main__":
    main()
