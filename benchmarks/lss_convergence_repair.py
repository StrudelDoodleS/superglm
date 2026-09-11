"""Replay the freMTPL Gamma convergence failures without search dependencies.

Run in a fresh process for each receipt. Use PYTHONPATH to select the baseline
source tree while keeping this script and environment unchanged.
Default runs are untimed with synchronous solver-event pool observations.
Pass --measure-time for a separate fit without observation hooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import superglm
from superglm import Categorical, GammaLS, Predictor, Spline, SuperLSS

if __package__:
    from . import rank_deficient_complete_fit as pool_helpers
else:
    import rank_deficient_complete_fit as pool_helpers

NUMS = ["VehAge", "DrivAge", "BonusMalus", "LogDensity", "VehPower"]
CATS = ["Area", "VehBrand", "VehGas", "Region"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["all", "bohb", "constant", "fixed"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--initial-lambda", type=float)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--full", action="store_true", help="Fit training and validation claims")
    parser.add_argument("--measure-time", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(args.data, dtype=dict.fromkeys(CATS, str))
    train = data.loc[data.split == "train"].sort_values("budget_order").reset_index(drop=True)
    full = data.loc[data.split != "test"].reset_index(drop=True)
    fit_data = full if args.scenario == "bohb" or args.full else train
    levels = {name: sorted(train[name].unique().tolist()) for name in CATS}

    def specifications(names):
        return {
            name: Spline("cr", k=8) if name in NUMS else Categorical(levels=levels[name])
            for name in names
        }

    scale_names = (
        ["DrivAge", "VehPower"]
        if args.scenario == "bohb"
        else []
        if args.scenario == "constant"
        else NUMS + CATS
    )
    from superglm.distributional.binding import _bind_predictor_template

    bound_family = GammaLS()
    model = SuperLSS(
        bound_family,
        *(
            _bind_predictor_template(bound_family, predictor)
            for predictor in [
                Predictor("mean", specifications(NUMS + CATS)),
                Predictor("scale", specifications(scale_names)),
            ]
        ),
        coefficient_curvature="observed",
        discrete=args.discrete,
    )
    source = Path(superglm.__file__).resolve()
    source_root = source.parents[2]
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, capture_output=True
    )
    commit = git_result.stdout.strip() if git_result.returncode == 0 else None
    source_hashes = {
        str(path.relative_to(source.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.parent.rglob("*.py"))
    }
    receipt = {
        "label": args.label,
        "scenario": args.scenario,
        "source": str(source),
        "commit": commit,
        "source_digest": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_files": {
            "benchmarks.rank_deficient_complete_fit": {
                "path": str(Path(pool_helpers.__file__).resolve()),
                "sha256": hashlib.sha256(Path(pool_helpers.__file__).read_bytes()).hexdigest(),
            }
        },
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "n_rows": len(fit_data),
        "mean_features": NUMS + CATS,
        "scale_features": scale_names,
        "discrete_requested": args.discrete,
        "initial_lambda_requested": args.initial_lambda,
        "strict_requested": args.strict,
        "maximum_response": float(fit_data.y.max()),
        "wall_time_status": "measured" if args.measure_time else "unmeasured",
    }
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sampler = None if args.measure_time else pool_helpers._DispatchSampler()
    if sampler is None:
        observer = nullcontext()
    else:
        from superglm.distributional.solver import solver as coefficient_solver
        from superglm.solvers import rank

        observer = pool_helpers._native_pool_dispatch(
            sampler,
            {
                "coefficient_core": coefficient_solver._fit_dense_fixed_lambda_core,
                "decompose_gram": rank.decompose_gram,
                "decompose_factor": rank.decompose_factor,
            },
        )
    with threadpool_limits(limits=1), sampler if sampler is not None else nullcontext(), observer:
        start = time.perf_counter() if args.measure_time else None
        if args.scenario == "fixed":
            model.fit(
                fit_data[NUMS + CATS],
                fit_data.y.to_numpy(),
                lambdas={
                    f"{parameter}:{name}#wiggle": 0.1
                    for parameter in ("mean", "scale")
                    for name in NUMS
                },
            )
        else:
            options = {} if args.initial_lambda is None else {"initial_lambda": args.initial_lambda}
            model.fit_reml(
                fit_data[NUMS + CATS],
                fit_data.y.to_numpy(),
                max_reml_iter=100,
                practical_reml=not args.strict,
                **options,
            )
        receipt["fit_seconds"] = None if start is None else time.perf_counter() - start
    receipt.update(pool_helpers._native_pool_receipt(sampler))
    receipt["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    receipt["prefit_peak_rss_mib"] = rss_before / 1024
    state = model._require_fitted().fit_state
    fit = state.solver_result
    smoothing = state.smoothing
    fits = (fit,) if smoothing is None else smoothing.coefficient_fits
    receipt.update(
        converged=model.result_.converged,
        coefficient_converged=fit.converged,
        coefficient_reason=fit.convergence_reason,
        coefficient_score_relative=fit.score_relative,
        coefficient_tolerance=fit.config.tolerance,
        coefficient_rank=fit.terminal_rank.rank,
        coefficient_count=len(fit.coefficients),
        coefficients=fit.coefficients.tolist(),
        execution_backend=fit.execution_backend_identifier,
        initial_lambdas=None if smoothing is None else dict(smoothing.initial_lambdas),
        final_lambdas=dict(state.lambdas),
        smoothing_reason=None if smoothing is None else smoothing.convergence_reason,
        smoothing_converged=None if smoothing is None else smoothing.converged,
        smoothing_objective=None if smoothing is None else smoothing.objective,
        n_coefficient_fits=len(fits),
        n_inner_iterations=sum(item.iterations for item in fits),
        n_fisher_fallback_fits=sum(item.terminal_curvature.fallback_count > 0 for item in fits),
        min_cv=float(fit.theta[:, 1].min()),
        max_cv=float(fit.theta[:, 1].max()),
        mean_range=[float(fit.theta[:, 0].min()), float(fit.theta[:, 0].max())],
        training_nll=-fit.log_likelihood / len(fit_data),
        terminal_curvature=asdict(fit.terminal_curvature),
        diagnostic=model.diagnose().to_dict(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / f"{args.label}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, default=_json_value))
    rows = fit_data[["row_id", "IDpol", "y"]].copy()
    rows["mu"] = fit.theta[:, 0]
    rows["cv"] = fit.theta[:, 1]
    rows.to_csv(args.out / f"{args.label}_predictions.csv", index=False)
    print(
        json.dumps(
            {
                key: value
                for key, value in receipt.items()
                if key not in {"diagnostic", "native_pools"}
            },
            indent=2,
            default=_json_value,
        )
    )
    print(f"Receipt: {receipt_path}")


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
