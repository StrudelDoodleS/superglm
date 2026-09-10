"""One fresh-process complete-fit receipt for QP, SCOP or sum-to-zero repair.

Select package source with PYTHONPATH, then run ``python -m
benchmarks.solver_repair_complete_fit --case qp --label baseline --out receipt.json``.
The default is untimed and records actual solver dispatch. ``--measure-time``
disables solver instrumentation; arrange exclusive execution separately.
``qp`` is the inactive-constraint control; ``qp_binding`` requires observed
active-set iterations. ``scop_single`` means one SCOP group using the default
joint routing, not coverage of the separate ``scop_newton_step`` kernel.
"""

from __future__ import annotations

import argparse
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
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from benchmarks import (
    _constrained_fit_profile,
    profile_constrained_fit_paths,
    profile_structured_credibility,
    rank_deficient_complete_fit,
)
from benchmarks.rank_deficient_complete_fit import (
    _DispatchSampler,
    _git_state,
    _native_pool_receipt,
)
from threadpoolctl import threadpool_limits

import superglm
from superglm.solvers import constrained_qp, scop_newton, sum_to_zero

CASES = ("qp", "qp_binding", "scop_single", "scop_joint_discrete", "sum_to_zero")
CASE_DESCRIPTIONS = {
    "qp": "QP with the existing inactive-constraint response",
    "qp_binding": "QP with a negative-quadratic response opposing the convex constraint",
    "scop_single": "One SCOP group using default joint routing",
    "scop_joint_discrete": "Two SCOP groups using discrete joint routing",
    "sum_to_zero": "Structured sum-to-zero factor smooth with a global spline",
}


def _json_value(value):
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


def _digest(value):
    payload = json.dumps(_json_value(value), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_identity():
    package = Path(superglm.__file__).resolve().parent
    files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    helpers = (
        _constrained_fit_profile,
        profile_constrained_fit_paths,
        profile_structured_credibility,
        rank_deficient_complete_fit,
    )
    commit, dirty = _git_state()
    return {
        "package_path": str(package),
        "source_digest": _digest(files),
        "source_files": files,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_files": {
            module.__name__: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            }
            for module in helpers
        },
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def _fixture(case, seed):
    if case == "sum_to_zero":
        config = profile_structured_credibility.CaseConfig(
            n=600,
            levels=12,
            family="gaussian",
            discrete=False,
            random_effects=0,
            secondary_levels=None,
            small_width=2,
            weights="nonuniform",
            seed=7401 if seed is None else seed,
            structured_term="factor_smooth",
            block_size=5,
            global_spline=True,
            factor_basis="sz",
        )
        prepared = profile_structured_credibility.prepare_case(config)
        model = profile_structured_credibility._new_model(prepared, backend="structured")
        return (
            model,
            prepared.X,
            prepared.y,
            prepared.sample_weight,
            prepared.offset,
            asdict(config),
        )
    config = _constrained_fit_profile.ProfileScenario(
        name=case,
        engine="qp" if case in {"qp", "qp_binding"} else "scop",
        n=600,
        k=8,
        n_constrained=2 if case == "scop_joint_discrete" else 1,
        repeated_support=False,
        discrete=case == "scop_joint_discrete",
        use_fremtpl=False,
    )
    seed = 42 if seed is None else seed
    frame, response, weights = profile_constrained_fit_paths.load_dataset(config, seed=seed)
    if case == "qp_binding":
        response = -np.square(frame["x1"].to_numpy())
    model = profile_constrained_fit_paths.build_model(config)
    return model, frame, response, weights, None, {**asdict(config), "seed": seed}


def _input_identity(frame, response, weights, offset):
    # Object-valued frames must hash their values, never pointer-array bytes.
    parts = {
        "frame": {
            "columns": list(frame.columns),
            "dtypes": list(map(str, frame.dtypes)),
            "index": frame.index.to_numpy(),
            "values": frame.to_numpy().tolist(),
        },
        "response": response,
        "sample_weight": weights,
        "offset": offset,
    }
    hashes = {name: _digest(value) for name, value in parts.items()}
    return _digest(hashes), hashes


def _attributes(value, names):
    return {name: getattr(value, name, None) for name in names}


@contextmanager
def _solver_dispatch(sampler=None):
    """Observe Python call/return events, including aliases, without replacing solvers."""
    targets = {
        constrained_qp.solve_constrained_qp.__code__: "solve_constrained_qp",
        scop_newton.scop_newton_step.__code__: "scop_newton_step",
        scop_newton.scop_joint_newton_step.__code__: "scop_joint_newton_step",
        scop_newton._compute_cross_gram.__code__: "scop_cross_gram",
        scop_newton._disc_disc_2d_hist.__code__: "discrete_cross_histogram",
        sum_to_zero.SumToZeroBlockFactor.__init__.__code__: "SumToZeroBlockFactor.__init__",
        sum_to_zero.SumToZeroBlockFactor.solve.__code__: "SumToZeroBlockFactor.solve",
        sum_to_zero.ProfiledSumToZeroBlockFactor.solve.__code__: "ProfiledSumToZeroBlockFactor.solve",
    }
    receipt = {
        "status": "instrumented, unmeasured",
        "calls": Counter(),
        "returned_calls": Counter(),
        "callers": Counter(),
        "scop_group_counts": Counter(),
        "scop_support_layouts": Counter(),
        "scop_cross_routes": Counter(),
        "scop_cross_histogram_returns": 0,
        "qp_results": [],
        "scop_steps": [],
        "sum_to_zero_factors": [],
        "observer_errors": [],
    }

    def observe(frame, event, result):
        name = targets.get(frame.f_code)
        if name is None or event not in {"call", "return"}:
            return
        local = frame.f_locals
        try:
            if sampler is not None:
                sampler.sample(f"{name}:{event}")
            if event == "call":
                receipt["calls"][name] += 1
                caller = frame.f_back.f_code.co_name if frame.f_back is not None else ""
                receipt["callers"][f"{name} <- {caller}"] += 1
                if name == "scop_joint_newton_step":
                    states = local["scop_states"]
                    receipt["scop_group_counts"][str(len(states))] += 1
                    layout = [
                        {
                            "group": int(index),
                            "basis_shape": list(state["B_scop"].shape),
                            "bin_indices": state.get("bin_idx") is not None,
                        }
                        for index, state in sorted(states.items())
                    ]
                    receipt["scop_support_layouts"][json.dumps(layout, sort_keys=True)] += 1
                return
            if result is not None:
                receipt["returned_calls"][name] += 1
            if name == "solve_constrained_qp" and result is not None:
                item = _attributes(
                    result,
                    ("active_set", "n_iter", "converged", "rank", "width", "method", "condition"),
                )
                item["constraint_slack"] = local["A"] @ result.beta - local["b"]
                receipt["qp_results"].append(_json_value(item))
            elif name in {"scop_newton_step", "scop_joint_newton_step"} and result is not None:
                results = result.items() if isinstance(result, dict) else [(None, result)]
                for index, step in results:
                    item = _attributes(
                        step,
                        (
                            "objective_before",
                            "objective_after",
                            "step_norm",
                            "used_fisher_fallback",
                            "linear_solver",
                            "linear_iterations",
                            "discarded_directions",
                        ),
                    )
                    receipt["scop_steps"].append(_json_value({"group": index, **item}))
            elif name == "scop_cross_gram" and result is not None:
                route = tuple(local[key].get("bin_idx") is not None for key in ("st_i", "st_j"))
                receipt["scop_cross_routes"][str(route)] += 1
            elif name == "discrete_cross_histogram" and result is not None:
                if frame.f_back.f_code is scop_newton._compute_cross_gram.__code__:
                    receipt["scop_cross_histogram_returns"] += 1
            elif name == "SumToZeroBlockFactor.__init__":
                item = _attributes(
                    local["self"],
                    ("rank", "shape", "n_levels", "block_size", "used_dense_fallback", "_logdet"),
                )
                receipt["sum_to_zero_factors"].append(_json_value(item))
        except Exception as error:
            receipt["observer_errors"].append(f"{name}: {type(error).__name__}: {error}")

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        yield receipt
    finally:
        sys.setprofile(previous)


def _dispatch_requirements(case, dispatch):
    requirements = {"observer_errors_absent": not dispatch["observer_errors"]}
    if case in {"qp", "qp_binding"}:
        requirements["qp_result_observed"] = bool(dispatch["qp_results"])
        if case == "qp_binding":
            requirements["binding_active_set_solve_observed"] = any(
                item["active_set"] and item["n_iter"] > 0 and item["converged"]
                for item in dispatch["qp_results"]
            )
    elif case.startswith("scop_"):
        groups = "1" if case == "scop_single" else "2"
        requirements["joint_step_returned"] = (
            dispatch["returned_calls"]["scop_joint_newton_step"] > 0
        )
        requirements["requested_group_count_observed"] = dispatch["scop_group_counts"][groups] > 0
        if case == "scop_joint_discrete":
            requirements["cross_histogram_returned"] = dispatch["scop_cross_histogram_returns"] > 0
    else:
        requirements["factor_rank_published"] = any(
            item["rank"] is not None for item in dispatch["sum_to_zero_factors"]
        )
        for name in ("SumToZeroBlockFactor.solve", "ProfiledSumToZeroBlockFactor.solve"):
            requirements[f"{name}_returned"] = dispatch["returned_calls"][name] > 0
    return requirements


def _fit_outputs(model, frame, offset):
    result = model.result
    smoothing = getattr(model, "_reml_result", None)
    rank = getattr(result, "rank_info", None)
    design = getattr(model, "_dm", None)
    outputs = {
        "prediction": model.predict(frame, offset=offset),
        "coefficients": result.beta,
        "coefficient_converged": result.converged,
        "coefficient_reason": getattr(result, "termination_reason", None),
        "iterations": result.n_iter,
        "intercept": result.intercept,
        "deviance": result.deviance,
        "effective_df": result.effective_df,
        "phi": result.phi,
        "log_det_H": getattr(result, "log_det_H", None),
        "reml_hessian_rank": getattr(result, "reml_hessian_rank", None),
        "direct_backend": getattr(result, "direct_backend", None),
        "direct_fallback_reason": getattr(result, "direct_fallback_reason", None),
        "group_matrix_classes": None
        if design is None
        else dict(Counter(type(group).__name__ for group in design.group_matrices)),
        "data_rank": None if rank is None else rank.data.rank,
        "coefficient_rank": None if rank is None else rank.coefficient.rank,
        "rank_geometry": {
            name: _attributes(
                getattr(rank, name, None),
                ("rank", "width", "method", "rank_truncated", "resolution_limited", "log_pdet"),
            )
            for name in ("data", "augmented", "coefficient")
        },
        "lambdas": getattr(model, "_reml_lambdas", None),
        "smoothing_converged": getattr(smoothing, "converged", None),
        "smoothing_iterations": getattr(smoothing, "n_reml_iter", None),
        "smoothing_reason": getattr(smoothing, "termination_reason", None),
        "objective": getattr(smoothing, "objective", None),
        **_attributes(
            smoothing,
            (
                "curvature_source",
                "lambda_history",
                "objective_history",
                "inner_iter_history",
                "scop_step_norms",
                "scop_fisher_fallbacks",
                "managed_cleanup_active_history",
                "managed_cleanup_frozen_history",
            ),
        ),
    }
    outputs["iteration_log"] = [
        {name: value for name, value in asdict(item).items() if not name.endswith("_id")}
        for item in (getattr(result, "iteration_log", None) or ())
    ]
    outputs["scop_states"] = {
        str(name): {
            key: state[key]
            for key in ("beta_eff", "gamma_eff", "last_step_norm", "last_fisher_fallback")
            if key in state
        }
        for name, state in (getattr(smoothing, "scop_states", None) or {}).items()
    }
    outputs["constraint_slacks"] = {
        group.name: group.constraints.A @ result.beta[group.sl] - group.constraints.b
        for group in (getattr(model, "_groups", None) or ())
        if group.constraints is not None
    }
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, help="Override seed: constrained defaults 42, SZ defaults 7401."
    )
    parser.add_argument("--measure-time", action="store_true")
    args = parser.parse_args()
    model, frame, response, weights, offset, config = _fixture(args.case, args.seed)
    data_hash, input_hashes = _input_identity(frame, response, weights, offset)
    receipt = {
        "schema_version": 1,
        "label": args.label,
        "case": args.case,
        "case_description": CASE_DESCRIPTIONS[args.case],
        "mode": "reml",
        "rows": len(response),
        "configuration": config,
        "fit_controls": {"max_reml_iter": 40},
        "data_sha256": data_hash,
        "input_sha256": input_hashes,
        "provenance": _source_identity(),
        "load_before": os.getloadavg(),
        "wall_time_status": "measured" if args.measure_time else "unmeasured",
        "peak_rss_measures_fits": 1,
        "solver_instrumented": not args.measure_time,
    }
    sampler = None if args.measure_time else _DispatchSampler()
    context = (
        nullcontext({"status": "not instrumented; use a separate unmeasured receipt", "calls": {}})
        if args.measure_time
        else _solver_dispatch(sampler)
    )
    elapsed = None
    error = None
    with warnings.catch_warnings(record=True) as recorded, threadpool_limits(limits=1):
        with sampler if sampler is not None else nullcontext(), context as dispatch:
            started = time.perf_counter() if args.measure_time else None
            try:
                model.fit_reml(
                    frame,
                    response,
                    sample_weight=weights,
                    offset=offset,
                    max_reml_iter=40,
                )
            except Exception as failure:
                error = {"type": type(failure).__name__, "message": str(failure)}
            finally:
                if started is not None:
                    elapsed = time.perf_counter() - started
        receipt["kernel_dispatch"] = dispatch
        receipt["warnings"] = [str(item.message) for item in recorded]
    if not args.measure_time:
        requirements = _dispatch_requirements(args.case, dispatch)
        dispatch["requirements"] = requirements
        missing = [name for name, passed in requirements.items() if not passed]
        if error is None and missing:
            error = {
                "phase": "dispatch",
                "type": "MissingSolverDispatch",
                "message": f"Missing observed solver evidence: {', '.join(missing)}.",
            }
    receipt["fit_seconds"] = elapsed
    receipt["fit_error"] = error
    receipt["load_after"] = os.getloadavg()
    rss_unit = 1024.0**2 if sys.platform == "darwin" else 1024.0
    receipt["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_unit
    receipt["memory_scope"] = (
        "whole fresh process through one complete fit; before output prediction"
    )
    receipt.update(_native_pool_receipt(sampler))
    try:
        receipt["outputs"] = None if error is not None else _fit_outputs(model, frame, offset)
    except Exception as failure:
        error = {"phase": "outputs", "type": type(failure).__name__, "message": str(failure)}
        receipt["fit_error"] = error
        receipt["outputs"] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_value(receipt), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(args.out),
                "source": receipt["provenance"]["source_digest"],
                "error": error,
            }
        )
    )
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
