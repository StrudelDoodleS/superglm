"""Measure complete-fit cost and retained row histories on the C1 fixture.

Run each source/mode in a fresh process. Profiling is a separate run.
The existing C1 generator supplies the model, data and deterministic split.
"""

from __future__ import annotations

import argparse
import cProfile
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
from collections.abc import Mapping
from pathlib import Path


def source_digest(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resident_bytes() -> int:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def history_census(smoothing) -> dict:
    """Count row arrays, including fit objects held by endpoint contexts."""
    import numpy as np

    from superglm.distributional.results.solver import DenseSolverResult

    stack, seen, fits, buffers = [smoothing], set(), [], {}
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, DenseSolverResult):
            fits.append(item)
            for value in (item.eta, item.theta):
                if value is None:
                    continue
                owner = value
                while isinstance(owner.base, np.ndarray):
                    owner = owner.base
                base = owner.base if owner.base is not None else owner
                buffers[id(base)] = max(buffers.get(id(base), 0), owner.nbytes)
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            stack.extend(getattr(item, field.name) for field in dataclasses.fields(item))
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
    return {
        "history_fits": len(smoothing.coefficient_fits),
        "history_fits_with_rows": sum(fit.eta is not None for fit in smoothing.coefficient_fits),
        "reachable_solver_results": len(fits),
        "reachable_solver_results_with_rows": sum(fit.eta is not None for fit in fits),
        "unique_reachable_solver_row_bytes": sum(buffers.values()),
        "scope": "Solver results reachable through smoothing dataclasses and containers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=1_048_576)
    parser.add_argument("--retain-rows", type=int, choices=(0, 1), default=1)
    parser.add_argument("--discrete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--history", choices=("default", "full"), default="default")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source / "src"))

    import numpy as np
    import pandas as pd
    from c3_c1_complete_fit import gaussian_fragmented_fixture, jsonable
    from discrete_performance import background_audit, process_snapshot
    from threadpoolctl import threadpool_info, threadpool_limits

    import superglm

    package = Path(superglm.__file__).resolve().parent
    if package != args.source / "src/superglm":
        raise RuntimeError(f"Wrong source imported: {package}")
    before_hash = source_digest(package)
    with threadpool_limits(limits=1, user_api="blas"):
        template, frame, y, holdout, provenance = gaussian_fragmented_fixture(args.n, 4)
        from superglm.distributional.binding import _bind_predictor_template

        bound_family = template.family
        model = superglm.SuperLSS(
            bound_family,
            *(
                _bind_predictor_template(bound_family, predictor)
                for predictor in template.predictors
            ),
            discrete=args.discrete,
            n_bins=256,
        )
        del template
        inputs = {
            "frame": hashlib.sha256(
                pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
            ).hexdigest(),
            "response": hashlib.sha256(y.tobytes()).hexdigest(),
            "holdout": hashlib.sha256(
                pd.util.hash_pandas_object(holdout, index=True).to_numpy().tobytes()
            ).hexdigest(),
        }
        warm_start = time.perf_counter()
        superglm.warmup()
        warm_seconds = time.perf_counter() - warm_start
        fit_kwargs = {"retain_rows": bool(args.retain_rows)}
        if args.history == "full":
            fit_kwargs["retain_history_rows"] = True
        profiler = cProfile.Profile() if args.profile else None
        rss_before = resident_bytes()
        load_before = os.getloadavg()
        activity_before = process_snapshot()
        if profiler is not None:
            profiler.enable()
        cpu_start, start = time.process_time(), time.perf_counter()
        model.fit_reml(
            frame,
            y,
            initial_lambda=0.1,
            outer="efs",
            practical_reml=True,
            max_reml_iter=60,
            max_inner_iter=100,
            reml_tol=1e-6,
            inner_tol=1e-7,
            **fit_kwargs,
        )
        elapsed, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
        if profiler is not None:
            profiler.disable()
        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        rss_after = resident_bytes()
        activity_after = process_snapshot()
        activity = background_audit(activity_before, activity_after, {os.getpid()})
        smoothing = model._require_fitted().smoothing
        result = model.result_
        terminal = smoothing.terminal_fit
        arrays = {
            "coefficients": result.coefficients,
            "covariance": model.covariance_,
            "train_parameters": model.predict_parameters(frame).to_numpy(),
            "holdout_parameters": model.predict_parameters(holdout).to_numpy(),
            "lambdas": np.asarray(list(result.smoothing_parameters.values())),
            "terminal_score": terminal.terminal_score,
            "terminal_curvature": terminal.terminal_penalized_curvature,
        }
        np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
        record = {
            "config": vars(args),
            "fixture": {"provenance": provenance, "rows": args.n, "inputs": inputs},
            "source_sha256": before_hash,
            "source_stable": before_hash == source_digest(package),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fixture_generator_sha256": hashlib.sha256(
                Path(__file__).with_name("c3_c1_complete_fit.py").read_bytes()
            ).hexdigest(),
            "fit_seconds": None if args.profile else elapsed,
            "profile_seconds": elapsed if args.profile else None,
            "fit_cpu_seconds": cpu_seconds,
            "warmup_seconds": warm_seconds,
            "fit_peak_process_rss_bytes": peak_bytes,
            "resident_before_fit_bytes": rss_before,
            "resident_after_fit_bytes": rss_after,
            "load_before": load_before,
            "load_after": os.getloadavg(),
            "background_activity": {
                key: activity[key]
                for key in (
                    "external_cpu_cores_lower_bound",
                    "passes_activity_screen",
                    "caveat",
                )
            },
            "native_pools": threadpool_info(),
            "history_census": history_census(smoothing),
            "result": {
                name: getattr(result, name)
                for name in (
                    "converged",
                    "coefficient_converged",
                    "smoothing_converged",
                    "n_inner_iter",
                    "n_smoothing_iter",
                    "log_likelihood",
                    "penalized_log_likelihood",
                    "total_effective_df",
                    "rank",
                    "smoothing_parameters",
                    "exact_face_components",
                )
            },
            "smoothing_reason": smoothing.convergence_reason,
            "history": smoothing.history,
            "coefficient_fit_backends": [
                fit.execution_backend_identifier for fit in smoothing.coefficient_fits
            ],
            "coefficient_fit_iterations": [fit.iterations for fit in smoothing.coefficient_fits],
            "phases": model._fit_phase_snapshot.as_dict(),
            "environment": {
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("superglm", "numpy", "scipy", "pandas", "numba")
                },
                "numba_workers": int(os.environ.get("NUMBA_NUM_THREADS", "0")),
            },
        }
        if not record["source_stable"]:
            raise RuntimeError("Source changed during measurement")
        args.out.write_text(json.dumps(jsonable(record), indent=2, allow_nan=False) + "\n")
        if profiler is not None:
            profiler.dump_stats(str(args.out.with_suffix(".prof")))
        print(
            json.dumps(
                jsonable(
                    {
                        key: record[key]
                        for key in (
                            "fit_seconds",
                            "profile_seconds",
                            "fit_peak_process_rss_bytes",
                            "resident_after_fit_bytes",
                            "history_census",
                            "result",
                        )
                    }
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
