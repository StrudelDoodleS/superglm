"""Compare existing SuperLSS smoothing policies in separate processes.

This is an investigation of optimizer completion, not a change to the default
policy. Run each case and route in a fresh process. Exclude warm-up runs from
timing summaries, rotate route order, and keep profiles separate from timings.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import resource
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from c3_c1_complete_fit import data_fixture, environment_snapshot, jsonable, source_receipt
from threadpoolctl import threadpool_info, threadpool_limits

import superglm
from superglm import (
    GaussianLS,
    NegativeBinomialLS,
    Numeric,
    Predictor,
    RandomEffect,
    Spline,
    SuperLSS,
    TensorInteraction,
)
from superglm.distributional.prediction_design import (
    build_joint_prediction_design,
    link_standard_errors,
)
from superglm.distributional.smoothing import derivatives, loop, newton
from superglm.distributional.solver import solver
from superglm.distributional.timing import FitPhaseRecorder

CASES = (
    "gaussian-correlated",
    "gaussian-crossed",
    "gaussian-mixed",
    "gaussian-outward",
    "gamma-real",
    "nb2-interior",
    "nb2-real",
    "tweedie-stress",
    "gpd-tail",
)
ROUTES = {
    "efs": {"outer": "efs", "practical_reml": True},
    "newton": {"outer": "efs+newton", "practical_reml": True},
    "newton-strict": {"outer": "efs+newton", "practical_reml": False},
}
CASE_OPTIONS = {
    "gaussian-outward": {
        "lambdas": {"location:effect#wiggle": 1.0e6},
        "max_lambda": 1.0e6 * np.exp(1.8),
        "max_log_step": 0.6,
        "max_reml_iter": 20,
        "reml_tol": 1.0e-8,
        "inner_tol": 1.0e-10,
        "reml_plateau_tol": 1.0e-6,
    }
}


def gaussian_fixture(crossed):
    """The original API-test fixture and its independent crossed control."""
    if crossed:
        x, z = np.meshgrid(np.linspace(-1.0, 1.0, 20), np.linspace(-1.0, 1.0, 20), indexing="ij")
        x, z = x.ravel(), z.ravel()
        frame = pd.DataFrame({"x": x, "z": z})
        mean = (
            0.65
            + 0.7 * x
            + 0.4 * z
            + 0.5 * np.sin(np.pi * x)
            + 0.3 * np.cos(np.pi * z)
            + 0.4 * np.sin(np.pi * x) * np.cos(np.pi * z)
        )
        sigma = 0.02 + np.exp(-1.45 + 0.25 * z)
        y = mean + np.random.default_rng(20260912).normal(scale=sigma)
        weights = np.ones(len(frame))
        offsets = {name: np.zeros(len(frame)) for name in ("location", "scale")}
    else:
        n = 160
        x = np.linspace(-1.0, 1.0, n)
        z = np.cos(np.linspace(0.0, 2.0 * np.pi, n))
        sigma = 0.18 + np.exp(-1.45 + 0.25 * z)
        y = 0.65 + 0.7 * x + np.random.default_rng(20260723).normal(scale=sigma)
        frame = pd.DataFrame({"x": x, "z": z}, index=pd.Index(np.arange(n) + 100, name="row"))
        weights = np.linspace(0.7, 1.4, n)
        offsets = {
            "location": np.linspace(-0.08, 0.11, n),
            "scale": 0.03 * np.sin(np.linspace(0.0, 2.0 * np.pi, n)),
        }
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.02),
        predictors=(
            Predictor(
                "location",
                {"x": Spline(n_knots=4), "z": Spline(n_knots=4)},
                interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
            ),
            Predictor("scale", {"z": Numeric()}),
        ),
        discrete=True,
        n_bins=32,
    )
    return model, frame, y, weights, offsets, frame, offsets, {"evaluation": "training rows"}


def make_fixture(case, data):
    if case == "gaussian-outward":
        labels = np.repeat(np.array(["a", "b", "c", "d"]), 10)
        frame = pd.DataFrame({"effect": labels})
        y = np.random.default_rng(7).normal(size=len(frame))
        model = SuperLSS(
            family=GaussianLS(scale_floor=1.0e-4),
            predictors=(
                Predictor("location", {"effect": RandomEffect()}),
                Predictor("scale", {}),
            ),
        )
        return model, frame, y, None, None, frame, None, {"evaluation": "training rows"}
    if case == "nb2-interior":
        rng = np.random.default_rng(20260912)
        x = rng.uniform(-1.0, 1.0, 3000)
        mu = np.exp(0.6 + 0.7 * np.sin(np.pi * x))
        theta = np.exp(0.5 + 0.3 * x)
        frame = pd.DataFrame({"x": x})
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
        model = SuperLSS(
            family=NegativeBinomialLS(),
            predictors=(
                Predictor("mean", {"x": Spline(kind="cr", k=8)}),
                Predictor("theta", {"x": Numeric()}),
            ),
            discrete=True,
            n_bins=256,
        )
        evaluation = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 300)})
        return model, frame, y, None, None, evaluation, None, {"evaluation": "fixed grid"}
    if case.startswith("gaussian-") and case != "gaussian-mixed":
        return gaussian_fixture(case == "gaussian-crossed")
    fixture, n, knots = {
        "gaussian-mixed": ("gaussian-fragmented", 4096, 4),
        "gamma-real": ("severity-gamma", 0, 4),
        "nb2-real": ("nb2", 20000, 4),
        "tweedie-stress": ("tweedie-stress", 10000, 4),
        "gpd-tail": ("gpd-tail", 10000, 3),
    }[case]
    return data_fixture(
        SimpleNamespace(
            fixture=fixture,
            n=n,
            knots=knots,
            mix=None,
            levels=4,
            data=data,
            holdout_every=10,
            replicate=1,
            support_size=None,
            discrete=True,
            n_bins=256,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    report = {
        "case": args.case,
        "route": args.route,
        "options": {**CASE_OPTIONS.get(args.case, {}), **ROUTES[args.route]},
        "source": source_receipt(root),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "superglm",
                "numpy",
                "scipy",
                "numba",
                "tabmat",
            )
        },
        "status": "started",
        "profile": args.profile,
        "imported_sources": {
            module.__name__: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            }
            for module in (superglm, loop, newton, derivatives, solver)
        },
    }
    profiler = cProfile.Profile() if args.profile else None
    phases = FitPhaseRecorder() if args.profile else None
    try:
        if any(
            not Path(entry["path"]).is_relative_to(root / "src")
            for entry in report["imported_sources"].values()
        ):
            raise RuntimeError("the worker must import SuperGLM from the recorded checkout")
        with threadpool_limits(limits=1):
            model, frame, y, weights, offsets, evaluation, evaluation_offsets, provenance = (
                make_fixture(args.case, args.data)
            )
            report.update(
                rows=len(frame),
                evaluation_rows=len(evaluation),
                provenance=provenance,
                family=model.family.to_config(),
                threadpools=threadpool_info(),
                input_hashes={
                    "frame": hashlib.sha256(frame.to_csv(index=True).encode()).hexdigest(),
                    "response": hashlib.sha256(np.asarray(y).tobytes()).hexdigest(),
                    "evaluation": hashlib.sha256(
                        evaluation.to_csv(index=True).encode()
                    ).hexdigest(),
                },
            )
            if weights is not None:
                report["input_hashes"]["weights"] = hashlib.sha256(
                    np.asarray(weights).tobytes()
                ).hexdigest()
            for name, value in (offsets or {}).items():
                report["input_hashes"][f"offset:{name}"] = hashlib.sha256(
                    np.asarray(value).tobytes()
                ).hexdigest()
            for name, value in (evaluation_offsets or {}).items():
                report["input_hashes"][f"evaluation_offset:{name}"] = hashlib.sha256(
                    np.asarray(value).tobytes()
                ).hexdigest()
            report["environment_before"] = environment_snapshot()
            cpu_start, wall_start = time.process_time(), time.perf_counter()
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if profiler is not None:
                        profiler.enable()
                    model.fit_reml(
                        frame,
                        y,
                        sample_weight=weights,
                        offsets=offsets,
                        phase_recorder=phases,
                        **report["options"],
                    )
            finally:
                if profiler is not None:
                    profiler.disable()
                    profiler.dump_stats(str(args.out.with_suffix(".prof")))
                report["fit_wall_seconds"] = time.perf_counter() - wall_start
                report["fit_cpu_seconds"] = time.process_time() - cpu_start
                report["peak_process_rss_bytes"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
                report["environment_after"] = environment_snapshot()
                report["warnings"] = [str(w.message) for w in caught]
            fitted = model._require_fitted()
            smoothing = fitted.smoothing
            result = fitted.result
            design = build_joint_prediction_design(
                evaluation, fitted.compiled_predictors, fitted.layout
            )
            errors = link_standard_errors(design, fitted.covariance, fitted.layout)
            theta = fitted.predict_parameters(evaluation, offsets=evaluation_offsets)
            payload = {
                "theta": theta,
                "eta": fitted.predict_eta(evaluation, offsets=evaluation_offsets),
                "covariance": fitted.covariance,
            }
            payload.update({f"link_se:{name}": value for name, value in errors.items()})
            np.savez_compressed(args.out.with_suffix(".npz"), **payload)
            report.update(
                status="completed",
                coefficients=len(fitted.coefficients),
                converged=model.result_.converged,
                coefficient_converged=result.converged,
                coefficient_reason=result.convergence_reason,
                coefficient_score_relative=result.score_relative,
                smoothing_reason=smoothing.convergence_reason,
                smoothing_certified=smoothing.matched_certified,
                stationarity_bar=smoothing.stationarity_bar,
                projected_gradient=smoothing.terminal_projected_gradient_norm,
                gradient_certificate=smoothing.terminal_gradient_certificate,
                smoothing_objective=smoothing.objective,
                smoothing_iterations=smoothing.iterations,
                coefficient_fits=len(smoothing.coefficient_fits),
                coefficient_iterations=sum(f.iterations for f in smoothing.coefficient_fits),
                newton_iterations=smoothing.newton_iterations,
                bfgs_iterations=smoothing.bfgs_fallback_iterations,
                lambdas=dict(smoothing.lambdas),
                parameters=fitted.parameter_names,
                actual_backend=result.execution_backend_identifier,
                terminal_curvature=result.terminal_curvature,
                terminal_rank=result.terminal_rank.rank,
                diagnostic_codes=[finding.code for finding in model.diagnose().findings],
                phases=None if phases is None else phases.snapshot().as_dict(),
            )
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    args.out.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "case",
                    "route",
                    "status",
                    "smoothing_reason",
                    "smoothing_certified",
                    "fit_wall_seconds",
                )
            }
        ),
        flush=True,
    )
    if report["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
