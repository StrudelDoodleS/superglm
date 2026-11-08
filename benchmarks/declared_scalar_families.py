"""Sequential, fresh-process complete-fit receipts for declared scalar families.

Example (final run, allow minutes):
  uv run python benchmarks/declared_scalar_families.py --output /tmp/declared.json

Use --rows 160 --repeats 1 --case lomax-intercept for a harness smoke test.
Such a smoke test is not representative performance evidence. --baseline-source
must point to an independently prepared checkout's src directory; it reruns
Gamma/log-normal controls there in this same session. The direct pilot adapters
use identical kernels/laws/links/penalties and are abstraction-cost controls,
not independent numerical oracles. Profiling/instrumentation is untimed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

BASELINE = "6643ed9c94497eba0efb8869b945a17c5d380caf"
# law, rows, smooth, variable shape predictor, bounded, true shape, misspecified
CASES = {
    "lomax-intercept": ("lomax", 2000, False, False, False, 2.5, False),
    "log-logistic-intercept": ("log_logistic", 2000, False, False, False, 2.5, False),
    "lomax-smooth-20k": ("lomax", 20000, True, True, False, 2.5, False),
    "log-logistic-smooth-20k": ("log_logistic", 20000, True, True, False, 2.5, False),
    "lomax-smooth-100k": ("lomax", 100000, True, True, False, 2.5, False),
    "log-logistic-smooth-100k": ("log_logistic", 100000, True, True, False, 2.5, False),
    "lomax-constant-tail": ("lomax", 20000, True, False, False, 2.5, False),
    "log-logistic-constant-tail": ("log_logistic", 20000, True, False, False, 2.5, False),
    "lomax-bounded": ("lomax", 20000, True, True, True, 2.5, False),
    "log-logistic-bounded": ("log_logistic", 20000, True, True, True, 2.5, False),
    "lomax-infinite-mean": ("lomax", 2000, False, False, False, 0.75, False),
    "lomax-infinite-variance": ("lomax", 2000, False, False, False, 1.5, False),
    "log-logistic-infinite-mean": ("log_logistic", 2000, False, False, False, 0.75, False),
    "log-logistic-infinite-variance": ("log_logistic", 2000, False, False, False, 1.5, False),
    "lomax-misspecified": ("lomax", 2000, True, False, False, 0.75, True),
    "log-logistic-misspecified": ("log_logistic", 2000, True, False, False, 0.75, True),
    "gpd-overlap": ("gpd", 2000, False, False, False, 2.5, False),
    "gamma-control": ("gamma", 2000, True, False, False, 2.5, False),
    "log-normal-control": ("log_normal", 2000, True, False, False, 2.5, False),
}


def jsonable(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return {k: jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    return str(value)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def compare(candidate, control, *, comparison):
    """Keep numerical and work differences alongside complete-fit costs."""
    import numpy as np

    def difference(name):
        delta = np.asarray(candidate[name], dtype=float) - np.asarray(control[name], dtype=float)
        return float(np.max(np.abs(delta)))

    return dict(
        case=candidate["case"],
        comparison=comparison,
        candidate_sha=candidate["source_sha"],
        control_sha=control["source_sha"],
        warm_time_ratio=float(
            np.median(candidate["warm_fit_seconds"]) / np.median(control["warm_fit_seconds"])
        ),
        peak_rss_ratio=candidate["memory"]["peak_rss_bytes"] / control["memory"]["peak_rss_bytes"],
        retained_bytes_difference=candidate["memory"]["retained_array_bytes"]
        - control["memory"]["retained_array_bytes"],
        retained_cache_bytes_difference=candidate["memory"]["retained_cache_array_bytes"]
        - control["memory"]["retained_cache_array_bytes"],
        objective_difference=candidate["objective"] - control["objective"],
        log_likelihood_difference=candidate["log_likelihood"] - control["log_likelihood"],
        maximum_parameter_difference=difference("parameters"),
        maximum_coefficient_difference=difference("coefficients"),
        maximum_covariance_difference=difference("covariance"),
        maximum_standard_error_difference=difference("coefficient_se"),
        maximum_quantile_difference={
            p: float(
                np.max(
                    np.abs(
                        np.asarray(values, dtype=float)
                        - np.asarray(control["quantiles"][p], dtype=float)
                    )
                )
            )
            for p, values in candidate["quantiles"].items()
        },
        edf_difference=candidate["edf"] - control["edf"],
        rank_difference=candidate["rank"] - control["rank"],
        iterations_difference=candidate["iterations"] - control["iterations"],
        same_actual_backend=candidate["actual_backend"] == control["actual_backend"],
        same_generated_work=candidate["generated_dispatch"] == control["generated_dispatch"],
        equal_input=candidate["input_sha256"] == control["input_sha256"],
        candidate_fit_status=candidate["fit_status"],
        control_fit_status=control["fit_status"],
    )


def family_for(law, variant):
    import superglm

    names = {
        "lomax": "LomaxLSS",
        "log_logistic": "LogLogisticLSS",
        "gpd": "GeneralizedParetoLSS",
        "gamma": "GammaLS",
        "log_normal": "LogNormalLS",
    }
    cls = getattr(superglm, names[law])
    if variant == "shared":
        return cls()
    from superglm.distributional.families._declared import DeclaredScalarPlan
    from superglm.distributional.family import _register_likelihood_reuse_contract

    kernel = importlib.import_module("superglm.distributional.kernels." + law)

    # One benchmark-local class per isolated worker, never serialized or
    # registered as a production family. Core likelihood semantics agree.
    @dataclass(frozen=True)
    class Direct(cls):
        def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
            import numpy as np

            from superglm.distributional.families._response_bounds import bounded_channels
            from superglm.distributional.family import (
                NaturalLikelihoodEvaluation,
                validated_derivative_order,
            )

            order = validated_derivative_order(derivative_order)
            response = self.validate_likelihood_plan(y, plan)
            theta = self._theta(theta, len(response))
            result = kernel.density_jets(response, theta, plan.multiplier, order=order)
            channels = [*result.jets, None, None][:3]
            valid = result.valid
            if plan.observation.bounds is not None:

                def window(lo, hi, rows):
                    jets, valid = kernel.window_jets(lo, hi, theta[rows], order=order)
                    jets = tuple(
                        np.where(valid if j == 0 else valid[:, None], a, np.nan)
                        for j, a in enumerate(jets)
                    )
                    return (*jets, None, None)[:3]

                *channels, valid = bounded_channels(
                    *channels,
                    valid,
                    response=response,
                    bounds=plan.observation.bounds,
                    multiplier=plan.multiplier,
                    window=window,
                    derivative_order=order,
                )
            channels = [
                None if a is None else np.where(valid if j == 0 else valid[:, None], a, 0.0)
                for j, a in enumerate(channels)
            ]
            return NaturalLikelihoodEvaluation(
                channels[0], plan.parameter_independent_carrier, *channels[1:], valid=valid
            )

    _register_likelihood_reuse_contract(
        Direct,
        DeclaredScalarPlan,
        prepared_array_fields=("exact_response", "parameter_independent_carrier"),
        deterministic_chunk_replay=True,
    )
    return Direct()


def make_data(case, n, seed, *, observed):
    import numpy as np
    import pandas as pd

    law, _, smooth, _, bounded, base_shape, misspecified = CASES[case]
    rng = np.random.default_rng(seed)

    def batch(count):
        x = rng.uniform(-1.0, 1.0, count)
        scale = np.exp(0.4 + 0.3 * np.sin(2 * x)) if smooth else np.full(count, 2.0)
        # The constant-tail fit deliberately competes on the SAME variable-tail data.
        shape = base_shape * np.exp(0.15 * x) if smooth else np.full(count, base_shape)
        u = rng.uniform(np.finfo(float).eps, 1 - np.finfo(float).eps, count)
        if law in ("lomax", "gpd"):
            latent = scale * np.expm1(-np.log1p(-u) / shape)
        elif law == "log_logistic":
            latent = scale * np.exp((np.log(u) - np.log1p(-u)) / shape)
        elif law == "gamma":
            latent = rng.gamma(4.0, scale / 4)
        else:
            latent = rng.lognormal(np.log(scale) - 0.125, 0.5)
        if misspecified:
            latent = np.where(
                rng.random(count) < 0.8,
                rng.lognormal(np.log(scale) - 0.125, 0.5),
                scale * np.expm1(-np.log1p(-u) / 0.75),
            )
        return x, scale, shape, latent

    if observed and bounded:
        pieces = []
        kept = 0
        while kept < n:
            values = batch(max(2 * (n - kept), 100))
            mask = (values[-1] > 0.2) & (values[-1] <= 10.0)
            pieces.append(tuple(a[mask] for a in values))
            kept += int(mask.sum())
        x, scale, shape, latent = (np.concatenate([v[j] for v in pieces])[:n] for j in range(4))
    else:
        x, scale, shape, latent = batch(n)
    y = np.minimum(latent, 3.0) if observed and bounded else latent.copy()
    kwargs = {"truncation": (0.2, 10.0), "censored": latent > 3.0} if observed and bounded else {}
    return pd.DataFrame({"x": x}), y, scale, shape, kwargs


def model_and_data(args, family):
    from superglm import SuperLSS, bind_predictor, s

    law, rows, smooth, variable, bounded, _, _ = CASES[args.case]
    rows = args.rows or rows
    frame, y, _, _, bounds = make_data(args.case, rows, 6211, observed=True)
    names = [p.name for p in family.parameters]
    predictors = [
        bind_predictor(family, names[0], *((s("x", k=8),) if smooth else ())),
        bind_predictor(family, names[1], *((s("x", k=5),) if variable else ())),
    ]
    model = SuperLSS(family, *predictors, discrete=args.discrete)
    return model, frame, y, bounds, smooth


def complete_fit(args, family):
    model, frame, y, bounds, smooth = model_and_data(args, family)
    options = dict(bounds, retain_rows=not args.compact)
    if smooth:
        model.fit_reml(
            frame,
            y,
            outer=args.outer,
            initial_lambda=args.initial_lambda,
            max_reml_iter=args.max_reml_iter,
            **options,
        )
    else:
        model.fit(frame, y, **options)
    return model, frame, y


def retained_memory(root):
    import numpy as np

    seen, buffers = set(), set()
    total = cache = 0

    def visit(obj, path):
        nonlocal total, cache
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            base = obj
            while isinstance(getattr(base, "base", None), np.ndarray):
                base = base.base
            owner = base.base if isinstance(getattr(base, "base", None), bytes) else base
            if id(owner) not in buffers:
                buffers.add(id(owner))
                size = len(owner) if isinstance(owner, bytes) else base.nbytes
                total += size
                if any(word in path for word in ("cache", "table", "prepared")):
                    cache += size
        elif isinstance(obj, Mapping):
            for key, value in obj.items():
                visit(value, path + "." + str(key))
        elif isinstance(obj, (list, tuple)):
            for j, value in enumerate(obj):
                visit(value, path + "." + str(j))
        elif (
            not isinstance(obj, type)
            and hasattr(obj, "__dict__")
            and type(obj).__module__.startswith(("superglm", "scipy"))
        ):
            visit(vars(obj), path)

    visit(root, "model")
    return {
        "retained_array_bytes": total,
        "retained_cache_array_bytes": cache,
        "scope": "unique NumPy buffers reachable from the model; cache/table/prepared path subset",
        "excluded": "Python object headers and process-wide native/JIT caches; these contribute to peak RSS",
    }


def peak_rss():
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))


def validation(model, case):
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import expit, ndtr

    from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
    from superglm.distributional.observation import ResponseBounds
    from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

    law, _, smooth, _, bounded, _, misspecified = CASES[case]
    frame, latent, scale, shape, _ = make_data(case, 2000, 6212, observed=False)
    family = model.family
    predicted = model.predict_parameters(frame).to_numpy()
    threshold, attachment, width, probability = 2.0, 1.0, 2.0, 0.95  # fixed before draws
    lo, hi = np.full(len(frame), 0.2), np.full(len(frame), 10.0)
    eligible = (latent > 0.2) & (latent <= 10.0) if bounded else np.ones(len(frame), bool)
    paid = np.minimum(latent[eligible], 3.0) if bounded else latent[eligible]
    obs = (
        bounded_observation(
            ResponseBounds.from_arrays(
                lo[eligible], hi[eligible], (latent[eligible] > 3.0).astype(int)
            )
        )
        if bounded
        else COMPLETE_OBSERVATION
    )
    weights = resolve_likelihood_weights(
        None, n_observations=len(paid), contract=WeightContract("prior")
    )
    plan = family.bind_likelihood(paid, weights, obs)
    likelihood = family.evaluate_natural(paid, predicted[eligible], plan, derivative_order=0)
    sf = 1 - family.cdf(threshold, predicted)
    q = family.quantile(probability, predicted)

    def true_survival(y):
        if misspecified:
            return 0.8 * ndtr((np.log(scale) - 0.125 - np.log(y)) / 0.5) + 0.2 * np.exp(
                -0.75 * np.log1p(y / scale)
            )
        if law in ("lomax", "gpd"):
            return np.exp(-shape * np.log1p(y / scale))
        if law == "log_logistic":
            return expit(-shape * (np.log(y) - np.log(scale)))
        if law == "gamma":
            from scipy.special import gammaincc

            return gammaincc(4.0, 4 * y / scale)
        return ndtr((np.log(scale) - 0.125 - np.log(y)) / 0.5)

    # Independent scalar quadrature of the averaged known survival; bounded
    # targets exist across every tested moment domain and mixture.
    truth, error = quad(
        lambda z: float(np.mean(true_survival(z))),
        attachment,
        attachment + width,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    fitted_layer, _ = quad(
        lambda z: float(np.mean(1 - family.cdf(z, predicted))),
        attachment,
        attachment + width,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    observed_layer = np.minimum(np.maximum(latent - attachment, 0.0), width)
    tail_rows = latent > threshold
    # Complete synthetic latent losses are available; capped real observations
    # alone would not validate these latent quantile/layer metrics.
    tail_y = np.where(tail_rows, latent, threshold)
    tail_obs = bounded_observation(
        ResponseBounds.from_arrays(
            np.full(len(frame), -np.inf), np.full(len(frame), np.inf), np.where(tail_rows, 0, -1)
        )
    )
    tw = resolve_likelihood_weights(
        None, n_observations=len(frame), contract=WeightContract("prior")
    )
    tp = family.bind_likelihood(tail_y, tw, tail_obs)
    tail_score = family.evaluate_natural(
        tail_y, predicted, tp, derivative_order=0
    ).reported_log_likelihood
    return dict(
        heldout_seed=6212,
        heldout_rows=len(frame),
        observed_rows=int(eligible.sum()),
        threshold=threshold,
        layer_attachment=attachment,
        layer_width=width,
        latent_truth_available=True,
        observed_law_log_score=float(np.mean(likelihood.reported_log_likelihood)),
        censored_tail_log_score=float(np.mean(tail_score)),
        exceedance_count=int(tail_rows.sum()),
        predicted_exceedance_count=float(sf.sum()),
        exceedance_calibration=float(tail_rows.mean() - sf.mean()),
        quantile_probability=probability,
        quantile_loss=float(np.mean((probability - (latent < q)) * (latent - q))),
        true_layer_mean=truth,
        oracle_quadrature_error=error,
        predicted_layer_mean=fitted_layer,
        heldout_layer_mean=float(observed_layer.mean()),
        layer_model_error=fitted_layer - truth,
        mean_domain_count=int((shape > 1).sum()),
        variance_domain_count=int((shape > 2).sum()),
        fitted_mean_domain_count=int(np.isfinite(family.default_prediction(predicted)).sum()),
        fitted_variance_domain_count=int(np.isfinite(family.variance(predicted)).sum()),
    )


def worker(args):
    started = time.perf_counter()
    import numpy as np
    import pandas as pd
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits

    import superglm

    import_seconds = time.perf_counter() - started
    source = Path(superglm.__file__).resolve().parents[2]
    record = dict(
        case=args.case,
        variant=args.variant,
        baseline_sha=BASELINE,
        source=str(source),
        source_sha=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        import_seconds=import_seconds,
        train_seed=6211,
        requested_discrete=args.discrete,
        compact=args.compact,
        rows=args.rows or CASES[args.case][1],
        case_definition=CASES[args.case],
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        threads=1,
    )
    try:
        hashes = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "src/superglm").rglob("*.py"))
        }
        record["source_digest"] = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest()
        start = time.perf_counter()
        family = family_for(CASES[args.case][0], args.variant)
        record["family_initialization_seconds"] = time.perf_counter() - start
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            model, frame, y = complete_fit(args, family)
            record["cold_fit_seconds"] = time.perf_counter() - start
            record["cold_import_family_fit_seconds"] = (
                import_seconds
                + record["family_initialization_seconds"]
                + record["cold_fit_seconds"]
            )
            record["input_sha256"] = hashlib.sha256(
                frame.to_numpy().tobytes() + y.tobytes()
            ).hexdigest()
            times = []
            for _ in range(args.repeats):
                del model
                gc.collect()
                start = time.perf_counter()
                model, frame, y = complete_fit(args, family)
                times.append(time.perf_counter() - start)
            fitted = model._require_fitted()
            result = fitted.result
            evaluation = pd.DataFrame({"x": np.linspace(-1, 1, 11)})
            covariance = model.covariance_
            record.update(
                warm_fit_seconds=times,
                warm_fit_summary=dict(
                    minimum=min(times), median=float(np.median(times)), maximum=max(times)
                ),
                log_likelihood=result.log_likelihood,
                objective=result.penalized_log_likelihood,
                coefficients=result.coefficients,
                parameters=model.predict_parameters(evaluation).to_numpy(),
                quantiles={str(p): model.predict_quantile(evaluation, p) for p in (0.5, 0.9, 0.99)},
                coefficient_se=np.sqrt(np.diag(covariance)),
                covariance=covariance,
                edf=float(np.trace(covariance @ result.terminal_data_curvature)),
                rank=result.terminal_rank.rank,
                actual_backend=result.execution_backend_identifier,
                coefficient_converged=result.converged,
                coefficient_reason=result.convergence_reason,
                iterations=result.iterations,
                retries=result.backtracking_steps,
                diagnostic=model.diagnose().to_dict(),
                native_pools=threadpool_info(),
                memory={**retained_memory(model), "peak_rss_bytes": peak_rss()},
                validation=validation(model, args.case),
            )
            record["fit_status"] = "converged" if result.converged else "coefficient_not_converged"
            if fitted.smoothing is not None:
                smoothing = fitted.smoothing
                if result.converged:
                    record["fit_status"] = (
                        "smoothing_certified"
                        if smoothing.matched_certified
                        else "smoothing_uncertified"
                    )
                record["smoothing"] = {
                    name: getattr(smoothing, name, None)
                    for name in (
                        "convergence_reason",
                        "matched_certified",
                        "objective",
                        "iterations",
                        "lambdas",
                        "newton_iterations",
                        "bfgs_fallback_iterations",
                        "terminal_projected_gradient_norm",
                        "terminal_gradient_certificate",
                        "smoothing_hessian_certificate",
                    )
                }
                record["smoothing"]["coefficient_fits"] = len(smoothing.coefficient_fits)
                record["smoothing"]["total_coefficient_iterations"] = sum(
                    f.iterations for f in smoothing.coefficient_fits
                )
            # A separate untimed full fit establishes the actual generated
            # calls. Python counters never contaminate reported timing runs.
            dispatch = {}
            module_name = {
                "lomax": "lomax",
                "log_logistic": "log_logistic",
                "gpd": "generalized_pareto",
                "gamma": "gamma",
                "log_normal": "log_normal_mean",
            }[CASES[args.case][0]]
            generated = importlib.import_module(
                "superglm.distributional.kernels._declared." + module_name
            )
            originals = {name: getattr(generated, name) for name in generated.LAWS}
            for name, fn in originals.items():

                def counted(*a, _name=name, _fn=fn, **kw):
                    receipt = dispatch.setdefault(_name, dict(calls=0, rows=0))
                    receipt["calls"] += 1
                    receipt["rows"] += len(a[0])
                    return _fn(*a, **kw)

                setattr(generated, name, counted)
            try:
                complete_fit(args, family)
            finally:
                for name, fn in originals.items():
                    setattr(generated, name, fn)
            record.update(
                status="completed",
                generated_dispatch=dispatch,
                warnings=list(dict.fromkeys(str(w.message) for w in caught)),
            )
    except Exception:
        record.update(status="failed", error=traceback.format_exc())
    write(args.output, record)
    return 0 if record["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--rows", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--outer", choices=("efs", "efs+finish", "efs+newton"), default="efs+finish"
    )
    parser.add_argument("--initial-lambda", type=float, default=1.0)
    parser.add_argument("--max-reml-iter", type=int, default=100)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--variant", choices=("shared", "direct"), default="shared", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.repeats < 1 or (args.rows is not None and args.rows < 20):
        parser.error("repeats>=1 and rows>=20 are required")
    if args.worker:
        args.case = args.case[0]
        return worker(args)
    selected = list(dict.fromkeys(args.case or list(CASES)))
    controls = ("gamma-control", "log-normal-control")
    if args.baseline_source:
        baseline_root = args.baseline_source.resolve().parent
        actual = subprocess.check_output(
            ["git", "-C", str(baseline_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(baseline_root), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
        ).strip()
        if actual != BASELINE or dirty or not (args.baseline_source / "superglm").is_dir():
            parser.error(f"baseline source must be a clean src/ at {BASELINE}; got {actual}")
        # A narrowed pilot selection must still contain both sides of every
        # requested drift comparison, not only the old-source half.
        selected.extend(case for case in controls if case not in selected)
    report = dict(
        baseline_sha=BASELINE,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=sys.argv,
        case_sizes_overridden=args.rows is not None,
        cases=[],
        baseline_controls="same-session baseline"
        if args.baseline_source
        else "not run: supply --baseline-source; candidate Gamma/log-normal are drift controls only",
        native_coverage=platform.platform(),
        comparisons=[],
    )
    directory = args.output.with_suffix(".cases")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for case in selected:
        variants = (
            ("shared", "direct") if CASES[case][0] in ("lomax", "log_logistic") else ("shared",)
        )
        jobs.extend((case, variant, None) for variant in variants)
    if args.baseline_source:
        jobs.extend((case, "shared", args.baseline_source) for case in controls)
    for i, (case, variant, source) in enumerate(jobs):
        path = directory / f"{i:02d}-{case}-{variant}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--case",
            case,
            "--variant",
            variant,
            "--output",
            str(path.resolve()),
            "--repeats",
            str(args.repeats),
            "--outer",
            args.outer,
            "--initial-lambda",
            str(args.initial_lambda),
            "--max-reml-iter",
            str(args.max_reml_iter),
        ]
        if args.rows:
            command.extend(("--rows", str(args.rows)))
        command.extend(
            flag
            for flag, enabled in (("--discrete", args.discrete), ("--compact", args.compact))
            if enabled
        )
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMBA_NUM_THREADS="1",
        )
        if source:
            env["PYTHONPATH"] = str(source.resolve())
        start = time.perf_counter()
        with path.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt = (
            json.loads(path.read_text())
            if path.exists()
            else dict(
                status="failed",
                case=case,
                variant=variant,
                error="worker produced no receipt; see log",
            )
        )
        receipt.update(
            process_seconds=time.perf_counter() - start,
            returncode=completed.returncode,
            baseline_control=source is not None,
            receipt_path=str(path.resolve()),
        )
        report["cases"].append(receipt)
        write(args.output, report)  # a failed/interrupted run retains completed cells
        print(json.dumps(dict(case=case, variant=variant, status=receipt["status"])), flush=True)
    for case in selected:
        pair = [
            r
            for r in report["cases"]
            if r["case"] == case and not r["baseline_control"] and r["status"] == "completed"
        ]
        if len(pair) == 2:
            shared, direct = pair
            report["comparisons"].append(compare(shared, direct, comparison="shared/direct"))
    if args.baseline_source:
        for case in controls:
            pair = [r for r in report["cases"] if r["case"] == case and r["status"] == "completed"]
            if len(pair) == 2:
                candidate, baseline = pair
                report["comparisons"].append(
                    compare(candidate, baseline, comparison="candidate/baseline")
                )
    report["status"] = (
        "completed" if all(r["status"] == "completed" for r in report["cases"]) else "failed"
    )
    write(args.output, report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
