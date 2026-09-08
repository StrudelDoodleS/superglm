"""Numerical C3 stress replay; diagnostic work is deliberately not timed.

Select the implementation with PYTHONPATH, using the same interpreter for both
arms. The pure fixture module belongs to this script's checkout. For example::

    PYTHONPATH=/path/to/baseline/src .venv/bin/python \
        benchmarks/c3_c1_stress_diagnosis.py gpd --source /path/to/baseline \
        --outer efs+newton \
        --strict --start .01 --oracle --out /tmp/gpd.json

JSON and NPZ are the authoritative receipts, accompanied by SHA256 hashes.
Oracle probes are separate from the fit and never alter its stopping decision.
The optional --row-reference needs R/mgcv for Tweedie, or mpmath for GPD.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from c3_c1_complete_fit import digest, source_receipt
from scipy import stats
from threadpoolctl import threadpool_info, threadpool_limits


def plain(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def oracle_gradient(model, y, matrices):
    """Independent refitted-objective finite differences, with resolution evidence."""
    from superglm.distributional.family import COMPLETE_OBSERVATION
    from superglm.distributional.smoothing.derivatives import laml_derivatives
    from superglm.distributional.smoothing.objective import joint_laplace_objective
    from superglm.distributional.solver.solver import fit_dense_fixed_lambda

    fitted = model._require_fitted()
    smoothing = fitted.smoothing
    base = smoothing.terminal_fit
    plan = model.family.bind_likelihood(
        y, fitted.fit_state.retained_rows.likelihood_weights, COMPLETE_OBSERVATION
    )
    derivatives = laml_derivatives(
        model.family,
        fitted.layout,
        y,
        plan,
        lambdas=smoothing.lambdas,
        fit=base,
        dense_matrices=matrices,
        want_hessian=False,
    )
    config = dataclasses.replace(base.config, tolerance=1e-11, max_iterations=500)
    rows = []
    for index, name in enumerate(derivatives.names):
        estimates = []
        for step in (2e-2, 1e-2, 5e-3):
            objectives = []
            for sign in (1, -1):
                lambdas = dict(smoothing.lambdas)
                lambdas[name] *= np.exp(sign * step)
                result = fit_dense_fixed_lambda(
                    model.family,
                    fitted.layout,
                    y,
                    plan,
                    fitted.layout.penalty_matrix(lambdas),
                    initial=base.coefficients,
                    config=config,
                )
                provenance = (
                    result.terminal_rank.method == base.terminal_rank.method
                    and result.terminal_rank.rank == base.terminal_rank.rank
                    and result.terminal_curvature.actual_source
                    == base.terminal_curvature.actual_source
                )
                if not result.converged or not provenance:
                    rows.append({"name": name, "refused": result.convergence_reason})
                    break
                objectives.append(
                    joint_laplace_objective(result, layout=fitted.layout, lambdas=lambdas)
                )
            if len(objectives) != 2:
                break
            estimates.append((objectives[0] - objectives[1]) / (2 * step))
        if len(estimates) == 3:
            extrapolated = (4 * estimates[-1] - estimates[-2]) / 3
            rows.append(
                {
                    "name": name,
                    "analytic": float(derivatives.gradient[index]),
                    "analytic_certificate": float(derivatives.gradient_certificate[index]),
                    "central_estimates": estimates,
                    "richardson": extrapolated,
                    "empirical_resolution": abs(estimates[-1] - estimates[-2]),
                    "difference": float(derivatives.gradient[index] - extrapolated),
                }
            )
    return rows


def row_reference(case, y, weights, parameters):
    """Independent density/derivative references at representative fitted rows."""
    indices = np.unique(
        np.r_[np.linspace(0, len(y) - 1, 32, dtype=int), np.argmin(y), np.argmax(y)]
    )
    if case == "tweedie":
        from superglm.distributional.families.tweedie import evaluate_tweedie_rows

        selected = parameters[indices]
        with tempfile.TemporaryDirectory(prefix="c3-tweedie-reference-") as directory:
            source = Path(directory) / "rows.csv"
            destination = Path(directory) / "reference.csv"
            np.savetxt(
                source, np.column_stack([y[indices], selected, weights[indices]]), delimiter=","
            )
            expression = (
                "a<-commandArgs(TRUE); x<-as.matrix(read.csv(a[1],header=FALSE)); "
                "v<-mgcv::ldTweedie(x[,1],mu=x[,2],phi=x[,3]/x[,5],p=x[,4]); "
                "write.table(v[,1],a[2],row.names=FALSE,col.names=FALSE,sep=','); "
                "cat(as.character(packageVersion('mgcv')))"
            )
            version = subprocess.check_output(
                ["Rscript", "-e", expression, str(source), str(destination)], text=True
            ).strip()
            reference = np.loadtxt(destination, delimiter=",")
        evaluated = evaluate_tweedie_rows(
            y[indices],
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            weights[indices],
            "prior",
            derivative_order=0,
        )
        return {
            "source": f"installed mgcv {version} ldTweedie, public callable; phi/weight",
            "indices": indices,
            "reference": reference,
            "production": evaluated.log_likelihood,
        }
    import mpmath as mp

    from superglm.distributional.kernels.generalized_pareto import scale_rows

    selected = parameters[indices]
    evaluated = scale_rows(
        y[indices], selected[:, 0], selected[:, 1], weights[indices], derivative_order=2
    )
    score = []
    hessian = []
    with mp.workdps(60):
        for index in indices:
            observation = mp.mpf(float(y[index]))
            weight = mp.mpf(float(weights[index]))
            scale, shape = (mp.mpf(float(value)) for value in parameters[index])

            def log_density(sigma, xi):
                return weight * (-mp.log(sigma) - (1 + 1 / xi) * mp.log1p(xi * observation / sigma))

            score.append(
                [float(mp.diff(log_density, (scale, shape), order)) for order in ((1, 0), (0, 1))]
            )
            hessian.append(
                [
                    float(mp.diff(log_density, (scale, shape), order))
                    for order in ((2, 0), (1, 1), (0, 2))
                ]
            )
    return {
        "source": "mpmath 60 decimal digit derivatives of GPD log density",
        "indices": indices,
        "score_scaled_max_error": float(
            np.max(np.abs(evaluated.score - score) / (1 + np.abs(score)))
        ),
        "hessian_scaled_max_error": float(
            np.max(np.abs(evaluated.hessian_packed - hessian) / (1 + np.abs(hessian)))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("gpd", "tweedie"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--outer", choices=("efs", "efs+newton"), default="efs")
    parser.add_argument("--start", type=float, default=0.1)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--acceleration", choices=("none", "multisecant"), default="multisecant")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--row-reference", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source.resolve()
    source_before = source_receipt(source_root)
    evidence_files = {
        "harness_sha256": Path(__file__),
        "fixtures_sha256": Path(__file__).with_name("_c3_c1_fixtures.py"),
        "provenance_helper_sha256": Path(__file__).with_name("c3_c1_complete_fit.py"),
    }
    evidence_before = {name: digest(path) for name, path in evidence_files.items()}
    # Snapshot source before importing any package or fixture implementation.
    from _c3_c1_fixtures import fixture, lss_model, marked_book

    import superglm
    from superglm.distributional import GeneralizedParetoLSS
    from superglm.distributional.assembly import dense_predictor_matrices

    imported = Path(superglm.__file__).resolve()
    if not imported.is_relative_to(source_root / "src"):
        raise RuntimeError(f"Wrong source imported: {imported}; expected {source_root / 'src'}")
    from numba import get_num_threads, set_num_threads

    threadpools_before = threadpool_info()
    numba_threads_before = get_num_threads()
    set_num_threads(1)
    if args.case == "tweedie":
        model, frame, y, weights = fixture(10000, mix=0.75, family="tweedie")
    else:
        book = marked_book(10000, tail=True)
        keep = (book["policy"] < 10000) & (book["losses"] > 1000)
        frame = book["frame"].iloc[book["policy"][keep]].reset_index(drop=True)
        y = book["losses"][keep] - 1000
        weights = np.ones(len(y))
        model = lss_model(GeneralizedParetoLSS(), 3)
    with threadpool_limits(limits=1):
        model.fit_reml(
            frame,
            y,
            sample_weight=weights,
            max_reml_iter=args.iterations,
            outer=args.outer,
            initial_lambda=args.start,
            practical_reml=not args.strict,
            acceleration=args.acceleration,
        )
        fitted = model._require_fitted()
        smoothing = fitted.smoothing
        matrices = dense_predictor_matrices(fitted.layout)
        parameters = model.predict_parameters(frame).to_numpy()
        covariance = model.covariance_
        eta_variance = np.column_stack(
            [
                np.einsum(
                    "ij,jk,ik->i",
                    matrix,
                    covariance[predictor.coefficient_slice, predictor.coefficient_slice],
                    matrix,
                )
                for predictor, matrix in zip(fitted.layout.predictors, matrices, strict=True)
            ]
        )
        coefficient_fields = (
            "converged",
            "convergence_reason",
            "iterations",
            "backtracking_steps",
            "score_relative",
            "objective_relative_change",
            "step_relative",
            "penalized_log_likelihood",
            "history",
        )
        report = {
            "schema": 2,
            "case": args.case,
            "source": source_before,
            "imported_module": str(imported),
            **evidence_before,
            "python": sys.version,
            "executable": sys.executable,
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pandas", "numba", "tabmat", "threadpoolctl")
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
            "threadpools_before": threadpools_before,
            "threadpools_during_fit": threadpool_info(),
            "numba_threads_before": numba_threads_before,
            "numba_threads_during_fit": get_num_threads(),
            "input_sha256": {
                name: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                for name, array in (("frame", frame.to_numpy()), ("y", y), ("weights", weights))
            },
            "settings": {key: value for key, value in vars(args).items() if key != "out"},
            "timing": "UNMEASURED: numerical diagnostic, potentially concurrent",
            "n": len(y),
            "q": len(model.result_.coefficients),
            "converged": model.result_.converged,
            "certified": model.smoothing_certified_,
            "reason": smoothing.convergence_reason,
            "objective": smoothing.objective,
            "edf": model.result_.total_effective_df,
            "log_likelihood": model.result_.log_likelihood,
            "lambdas": smoothing.lambdas,
            "unresolved": smoothing.unresolved_upper_bound,
            "endpoint_directions": smoothing.terminal_endpoint_directions,
            "terminal_projected_gradient_norm": smoothing.terminal_projected_gradient_norm,
            "terminal_gradient": smoothing.terminal_gradient,
            "terminal_gradient_certificate": smoothing.terminal_gradient_certificate,
            "stationarity_bar": smoothing.stationarity_bar,
            "smoothing_config": smoothing.config,
            "terminal_fit_index": smoothing.terminal_fit_index,
            "terminal_evidence_fresh": smoothing.terminal_evidence_fresh,
            "outer_history": smoothing.history,
            "coefficient_fits": [
                {name: getattr(fit, name) for name in coefficient_fields}
                for fit in smoothing.coefficient_fits
            ],
        }
        if args.case == "gpd":
            reference = stats.genpareto.logpdf(y, c=parameters[:, 1], scale=parameters[:, 0])
            report["scipy_log_likelihood"] = float(np.sum(weights * reference))
            report["scipy_log_likelihood_difference"] = float(
                model.result_.log_likelihood - np.sum(weights * reference)
            )
        if args.oracle:
            report["laml_oracle"] = oracle_gradient(model, y, matrices)
        if args.row_reference:
            report["row_reference"] = row_reference(args.case, y, weights, parameters)
    source_after = source_receipt(source_root)
    report["threadpools_after"] = threadpool_info()
    report["numba_threads_after"] = get_num_threads()
    report["source_after"] = {
        name: source_after[name] for name in ("sha", "diff_sha256", "source_tree_sha256")
    }
    report["source_stable_during_fit"] = all(
        source_before[name] == source_after[name] for name in ("sha", "source_tree_sha256")
    )
    report["evidence_files_after"] = {name: digest(path) for name, path in evidence_files.items()}
    report["evidence_files_stable_during_fit"] = evidence_before == report["evidence_files_after"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plain(report), indent=2) + "\n")
    arrays_path = args.out.with_suffix(".npz")
    np.savez(
        arrays_path,
        parameters=parameters,
        eta_variance=eta_variance,
        covariance=covariance,
        coefficients=model.result_.coefficients,
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.out, arrays_path)
    }
    args.out.with_suffix(".sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if not report["source_stable_during_fit"] or not report["evidence_files_stable_during_fit"]:
        raise RuntimeError("Source or evidence helpers changed during fit; reject this receipt")
    print(json.dumps({key: report[key] for key in ("case", "n", "q", "reason", "objective")}))


if __name__ == "__main__":
    main()
