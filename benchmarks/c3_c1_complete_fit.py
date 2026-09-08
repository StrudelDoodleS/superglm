"""Serial, fresh-process public SuperLSS complete-fit comparisons.

See c3_c1_complete_fit.md. The worker writes raw receipts itself: transformed
tool stdout and tool elapsed times are not benchmark evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    import numpy as np

    if dataclasses.is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def environment_snapshot():
    processes = []
    # Command lines are examined only for fixed category labels, never recorded.
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            name = (entry / "comm").read_text().strip()
            command = (entry / "cmdline").read_bytes().lower()
            categories = [
                label
                for label in ("headroom", "kompress", "pylance", "pytest", "node", "python")
                if label.encode() in command
            ]
            stat = (entry / "stat").read_text().split(")", 1)[1].split()
            processes.append(
                {
                    "pid": int(entry.name),
                    "name": name,
                    "categories": categories,
                    "cpu_ticks": int(stat[11]) + int(stat[12]),
                }
            )
        except (OSError, ValueError, IndexError):
            continue
    return {
        "time_ns": time.time_ns(),
        "load_average": os.getloadavg(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "process_activity": processes,
    }


def source_receipt(source):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    diff = git("diff", "HEAD", "--binary")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    # Hash production source independently, including ignored/untracked modules.
    files = {str(p.relative_to(source)): digest(p) for p in sorted((source / "src").rglob("*.py"))}
    return {
        "root": str(source),
        "sha": git("rev-parse", "HEAD").decode().strip(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files_sha256": files,
        "source_tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "untracked": [p.decode() for p in untracked if p],
    }


def data_fixture(args):
    import numpy as np
    import pandas as pd
    from _c3_c1_fixtures import fixture, lss_model, marked_book

    from superglm import SuperLSS
    from superglm.distributional import GaussianLS, Predictor
    from superglm.distributional.families import GammaLS, GeneralizedParetoLSS, NegativeBinomialLS
    from superglm.features import Categorical, Spline

    n = (
        args.n
        if args.n is not None
        else {
            "tweedie-stress": 10000,
            "tweedie-friendly": 100000,
            "gpd-tail": 10000,
            "severity-gaussian": 0,
            "severity-gamma": 0,
            "nb2": 100000,
            "factor-smooth": 10000,
            "gaussian": 1000,
        }[args.fixture]
    )
    knots = args.knots if args.knots is not None else (3 if args.fixture == "gpd-tail" else 4)
    offsets, holdout_offsets, provenance = None, None, {}
    if args.fixture in ("tweedie-stress", "tweedie-friendly", "gaussian"):
        mix = (
            args.mix
            if args.mix is not None
            else (0.75 if args.fixture == "tweedie-stress" else 0.0)
        )
        model, frame, y, weight = fixture(
            n, knots, mix, family=("gaussian" if args.fixture == "gaussian" else "tweedie")
        )
        rng = np.random.default_rng(5916)
        x = rng.uniform(size=2000)
        holdout = pd.DataFrame({"x": x, "z": mix * x + (1 - mix) * rng.uniform(size=2000)})
        provenance.update(seed=5915, holdout_seed=5916, mix=mix)
    elif args.fixture == "gpd-tail":
        book = marked_book(n, tail=True)
        mask = (book["policy"] < n) & (book["losses"] > 1000)
        frame = book["frame"].iloc[book["policy"][mask]].reset_index(drop=True)
        y, weight = book["losses"][mask] - 1000, None
        holdout = book["frame"].iloc[n:].reset_index(drop=True)
        model = lss_model(GeneralizedParetoLSS(), knots)
        provenance.update(seed=9506, training_policies=n, threshold=1000)
    elif args.fixture == "factor-smooth":
        rng = np.random.default_rng(9507)
        x = rng.uniform(size=n + 2000)
        g = rng.integers(0, args.levels, len(x))
        all_frame = pd.DataFrame({"x": x, "g": [f"g{i}" for i in g]})
        y = (
            np.sin(5 * x) * (1 + g / args.levels) + np.exp(-0.5 + 0.2 * x) * rng.normal(size=len(x))
        )[:n]
        frame, holdout, weight = all_frame.iloc[:n].copy(), all_frame.iloc[n:].copy(), None
        model = SuperLSS(
            family=GaussianLS(),
            predictors=[
                Predictor(
                    p, {"x": Spline(n_knots=knots), "g": Categorical()}, interactions=[("x", "g")]
                )
                for p in (parameter.name for parameter in GaussianLS().parameters)
            ],
        )
        provenance.update(seed=9507, levels=args.levels)
    else:
        freq_path = args.data / "freMTPL2freq.parquet"
        frame = pd.read_parquet(freq_path).sort_values("IDpol").reset_index(drop=True)
        provenance["data_sha256"] = {freq_path.name: digest(freq_path)}
        if args.fixture.startswith("severity"):
            sev_path = args.data / "freMTPL2sev.parquet"
            sev = pd.read_parquet(sev_path).groupby("IDpol", as_index=False)["ClaimAmount"].sum()
            frame = frame.merge(sev, on="IDpol", how="inner", validate="one_to_one")
            provenance["data_sha256"][sev_path.name] = digest(sev_path)
            y = frame["ClaimAmount"].to_numpy(dtype=float)
            if args.fixture == "severity-gaussian":
                y = np.log(y)
            family = GaussianLS() if args.fixture == "severity-gaussian" else GammaLS()
        else:
            y = frame["ClaimNb"].to_numpy(dtype=float)
            family = NegativeBinomialLS()
        # The held-out book is deterministic and disjoint; n=0 uses all but every tenth policy.
        hold_mask = (
            np.arange(len(frame)) % args.holdout_every == args.holdout_every - 1
            if args.holdout_every
            else np.arange(len(frame)) < min(2000, len(frame))
        )
        holdout = frame.loc[hold_mask].reset_index(drop=True)
        if args.holdout_every:
            frame, y = frame.loc[~hold_mask].reset_index(drop=True), y[~hold_mask]
        if n:
            frame, y = frame.iloc[:n].copy(), y[:n]
        if args.fixture == "nb2":
            if (frame["Exposure"] <= 0).any() or (holdout["Exposure"] <= 0).any():
                raise ValueError("NB2 log exposure requires positive raw Exposure")
            offsets = {family.parameters[0].name: np.log(frame["Exposure"].to_numpy())}
            holdout_offsets = {family.parameters[0].name: np.log(holdout["Exposure"].to_numpy())}
        for part in (frame, holdout):
            part["DrivAge"] = part["DrivAge"].clip(18, 90)
            part["VehAge"] = part["VehAge"].clip(0, 20)
            part["BonusMalus"] = part["BonusMalus"].clip(50, 150)
        frame = frame[["DrivAge", "VehAge", "BonusMalus"]]
        holdout = holdout[frame.columns]
        weight = None
        model = SuperLSS(
            family=family,
            predictors=[
                Predictor(p.name, {name: Spline(n_knots=knots) for name in frame.columns})
                for p in family.parameters
            ],
        )
        provenance.update(
            covariate_clips={"DrivAge": [18, 90], "VehAge": [0, 20], "BonusMalus": [50, 150]},
            response="log aggregate claim amount"
            if args.fixture == "severity-gaussian"
            else args.fixture,
            holdout_rule=(
                f"sorted IDpol, every {args.holdout_every}th policy"
                if args.holdout_every
                else "in-sample first 2000; full training book"
            ),
        )
    if args.replicate != 1:
        if args.replicate < 1 or not args.fixture.startswith("severity"):
            raise ValueError("--replicate requires a positive factor and a severity fixture")
        provenance["row_replication"] = {
            "factor": args.replicate,
            "original_training_rows": len(frame),
            "meaning": "Synthetic copies of observed policies; no additional independent real observations",
        }
        frame = pd.concat([frame] * args.replicate, ignore_index=True)
        y = np.tile(y, args.replicate)
    if args.support_size:
        # Both representations get these identical finite-support covariates.
        for name in frame.select_dtypes(include="number").columns:
            lo, hi = frame[name].min(), frame[name].max()
            for part in (frame, holdout):
                normalized = np.clip((part[name] - lo) / (hi - lo), 0, 1)
                part[name] = lo + np.rint(normalized * (args.support_size - 1)) * (hi - lo) / (
                    args.support_size - 1
                )
        provenance["finite_support"] = args.support_size
    model = SuperLSS(
        family=model.family, predictors=model.predictors, discrete=args.discrete, n_bins=args.n_bins
    )
    return model, frame, np.asarray(y), weight, offsets, holdout, holdout_offsets, provenance


class CallRecorder:
    """Separate profile run: counts actual Python dispatch without replacing it."""

    def __init__(self):
        self.calls, self.shapes = Counter(), Counter()

    def __call__(self, frame, event, arg):
        if event != "call":
            return
        name, module = frame.f_code.co_name, frame.f_globals.get("__name__", "")
        selected = (
            module.startswith("superglm")
            and name
            in (
                "toarray",
                "row_subset",
                "assemble_chunked_geometry",
                "evaluate_chunked_log_likelihood",
                "materialize_terminal_predictions",
                "_predictor_chunk",
                "fit_dense",
                "_run_iterations",
                "_evaluate_state_unmeasured",
                "_geometry",
                "_evaluate_joint_laml",
            )
        ) or (module.startswith("numpy") and name == "argsort")
        if selected:
            owner = frame.f_locals.get("self")
            key = f"{module}.{type(owner).__name__ + '.' if owner is not None else ''}{name}"
            self.calls[key] += 1
            array = frame.f_locals.get("a") if name == "argsort" else owner
            shape = getattr(array, "shape", None)
            if shape is not None:
                self.shapes[f"{key}:{tuple(shape)}"] += 1


def worker(args):
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    import superglm

    source = args.worker_source.resolve()
    imported = Path(superglm.__file__).resolve()
    if not imported.is_relative_to(source / "src"):
        raise RuntimeError(f"Wrong source imported: {imported}; expected {source / 'src'}")
    report = {
        "schema": 1,
        "config": vars(args),
        "source": source_receipt(source),
        "imported_module": str(imported),
        "harness_sha256": digest(__file__),
        "fixtures_sha256": digest(Path(__file__).with_name("_c3_c1_fixtures.py")),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "numba", "tabmat", "threadpoolctl", "superglm")
        },
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "environment_before": environment_snapshot(),
        "timing_status": "unmeasured",
    }
    output = args.worker_output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(limits=args.threads):
            report["threadpools"] = threadpool_info()
            model, frame, y, weight, offsets, holdout, holdout_offsets, provenance = data_fixture(
                args
            )
            inputs = {
                "train_frame": frame.to_csv(index=False).encode(),
                "holdout_frame": holdout.to_csv(index=False).encode(),
                "y": y.tobytes(),
            }
            if weight is not None:
                inputs["weight"] = np.asarray(weight).tobytes()
            for key, value in (offsets or {}).items():
                inputs[f"offset:{key}"] = np.asarray(value).tobytes()
            for key, value in (holdout_offsets or {}).items():
                inputs[f"holdout_offset:{key}"] = np.asarray(value).tobytes()
            report["fixture"] = {
                "rows": len(y),
                "holdout_rows": len(holdout),
                "provenance": provenance,
                "fingerprints": {k: hashlib.sha256(v).hexdigest() for k, v in inputs.items()},
            }
            warm_start = time.perf_counter()
            if args.warmup:
                superglm.warmup()
            report["warmup_seconds"] = time.perf_counter() - warm_start
            report["environment_fit_start"] = environment_snapshot()
            quiet = os.getloadavg()[0] <= 2 * len(os.sched_getaffinity(0))
            if args.measure_time and (not args.quiet_profile or not quiet or args.instrument):
                raise RuntimeError(
                    "Timing needs --quiet-profile, load <= 2x available CPUs, and no --instrument"
                )
            recorder = CallRecorder()
            if args.instrument:
                sys.setprofile(recorder)
            start = time.perf_counter()
            try:
                model.fit_reml(
                    frame,
                    y,
                    sample_weight=weight,
                    offsets=offsets,
                    outer=args.outer,
                    initial_lambda=args.initial_lambda,
                    practical_reml=args.practical_reml,
                    max_reml_iter=args.max_reml_iter,
                    max_inner_iter=args.max_inner_iter,
                    reml_tol=args.reml_tol,
                    inner_tol=args.inner_tol,
                    max_lambda=args.max_lambda,
                )
            finally:
                elapsed = time.perf_counter() - start
                sys.setprofile(None)
                report["peak_fit_process_rss_bytes"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
                report["environment_fit_end"] = environment_snapshot()
                report["instrumentation"] = {
                    "enabled": args.instrument,
                    "calls": recorder.calls,
                    "shape_counts": recorder.shapes,
                }
            if args.measure_time and os.getloadavg()[0] <= 2 * len(os.sched_getaffinity(0)):
                report.update(
                    timing_status="operator-asserted quiet serial measurement", fit_seconds=elapsed
                )
            result, smoothing = model.result_, model._require_fitted().smoothing
            terminal = smoothing.terminal_fit
            report["result"] = {
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
                    "parameter_names",
                    "coefficient_names",
                    "smoothing_parameters",
                    "predictor_edf",
                    "term_edf",
                    "curvature_telemetry",
                    "exact_face_components",
                )
            }
            report["result"].update(
                q=len(result.coefficients),
                smoothing_reason=model.smoothing_convergence_reason_,
                smoothing_certified=model.smoothing_certified_,
            )
            report["smoothing"] = {
                name: getattr(smoothing, name, None)
                for name in (
                    "objective",
                    "initial_objective",
                    "convergence_reason",
                    "terminal_raw_max_log_step",
                    "terminal_gradient",
                    "terminal_gradient_certificate",
                    "terminal_projected_gradient_norm",
                    "terminal_endpoint_directions",
                    "unresolved_upper_bound",
                    "beyond_cap_components",
                    "newton_iterations",
                    "bfgs_fallback_iterations",
                    "stationarity_bar",
                )
            }
            report["smoothing"]["history"] = [
                {field.name: getattr(item, field.name) for field in dataclasses.fields(item)}
                for item in smoothing.history
            ]
            report["coefficient_fits"] = [
                {
                    name: getattr(fit, name, None)
                    for name in (
                        "converged",
                        "convergence_reason",
                        "execution_backend_identifier",
                        "resolved_chunk_size",
                        "iterations",
                        "log_likelihood",
                        "penalized_log_likelihood",
                        "terminal_curvature",
                        "score_relative",
                        "step_relative",
                        "objective_relative_change",
                        "backtracking_steps",
                    )
                }
                for fit in smoothing.coefficient_fits
            ]
            report["phase_snapshot"] = (
                model._fit_phase_snapshot.as_dict() if model._fit_phase_snapshot else None
            )
            arrays = {
                "coefficients": result.coefficients,
                "covariance": model.covariance_,
                "train_parameters": model.predict_parameters(frame, offsets=offsets).to_numpy(),
                "holdout_parameters": model.predict_parameters(
                    holdout, offsets=holdout_offsets
                ).to_numpy(),
                "smoothing_parameters": np.asarray(list(result.smoothing_parameters.values())),
                "edf": np.asarray(result.total_effective_df),
                "objective": np.asarray(smoothing.objective),
                "log_likelihood": np.asarray(result.log_likelihood),
                "terminal_score": terminal.terminal_score,
                "terminal_penalized_curvature": terminal.terminal_penalized_curvature,
            }
            for name in ("smoothing_hessian", "smoothing_hessian_certificate"):
                value = getattr(smoothing, name, None)
                if value is not None:
                    arrays[name] = value
            np.savez_compressed(output.with_suffix(".npz"), **arrays)
            report["npz_sha256"] = digest(output.with_suffix(".npz"))
            report["status"] = "ok"
    except Exception as exc:
        report.update(
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    finally:
        report["environment_after"] = environment_snapshot()
        final_source = source_receipt(source)
        report["source_after"] = {
            key: final_source[key] for key in ("sha", "diff_sha256", "source_tree_sha256")
        }
        report["source_stable"] = all(
            report["source"][key] == report["source_after"][key]
            for key in ("sha", "source_tree_sha256")
        )
        if not report["source_stable"]:
            report["timing_status"] = "unmeasured: source changed during run"
            report.pop("fit_seconds", None)
        report["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        write_json(output, report)
    return 0 if report["status"] == "ok" else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", default=[], metavar="LABEL=ROOT")
    p.add_argument("--out", type=Path, default=Path(".benchmark-artifacts/c3-c1"))
    p.add_argument(
        "--fixture",
        choices=(
            "tweedie-stress",
            "tweedie-friendly",
            "gpd-tail",
            "gaussian",
            "severity-gaussian",
            "severity-gamma",
            "nb2",
            "factor-smooth",
        ),
        default="gaussian",
    )
    p.add_argument("--n", type=int)
    p.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Severity only: synthesize repeated training rows; holdout is unchanged",
    )
    p.add_argument("--knots", type=int)
    p.add_argument("--mix", type=float)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--support-size", type=int)
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    p.add_argument(
        "--holdout-every",
        type=int,
        default=10,
        help="Real data: exclude every kth policy; 0 keeps full book and reports in-sample validation",
    )
    p.add_argument("--discrete", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--n-bins", type=int, default=256)
    p.add_argument("--outer", choices=("efs", "efs+newton"), default="efs")
    p.add_argument("--initial-lambda", type=float, default=0.1)
    p.add_argument("--max-lambda", type=float, default=1.0e10)
    p.add_argument("--practical-reml", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-reml-iter", type=int, default=60)
    p.add_argument("--max-inner-iter", type=int, default=100)
    p.add_argument("--reml-tol", type=float, default=1.0e-6)
    p.add_argument("--inner-tol", type=float, default=1.0e-7)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--instrument", action="store_true")
    p.add_argument("--measure-time", action="store_true")
    p.add_argument("--quiet-profile", help="Operator assertion identifying a quiet machine/session")
    p.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return p


def main():
    args = parser().parse_args()
    if args.worker_source:
        return worker(args)
    if not args.source:
        raise SystemExit("Supply --source LABEL=/absolute/worktree; repeated arms run serially")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"runs": [], "environment_before": environment_snapshot()}
    common = sys.argv[1:]
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = str(args.threads)
    for repeat in range(args.repeat):
        for specification in args.source:
            label, root = specification.split("=", 1)
            if not label or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in label
            ):
                raise SystemExit(
                    "Source labels must contain only letters, numbers, underscores, hyphens"
                )
            source = Path(root).resolve()
            output = (args.out / f"{label}-{repeat}.json").resolve()
            env["PYTHONPATH"] = str(source / "src")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *common,
                "--worker-source",
                str(source),
                "--worker-output",
                str(output),
                "--data",
                str(args.data.resolve()),
            ]
            with output.with_suffix(".log").open("w") as log:
                completed = subprocess.run(
                    cmd, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT
                )
            manifest["runs"].append(
                {
                    "label": label,
                    "repeat": repeat,
                    "returncode": completed.returncode,
                    "json": str(output),
                    "json_sha256": digest(output) if output.exists() else None,
                    "log_sha256": digest(output.with_suffix(".log")),
                }
            )
            write_json(args.out / "manifest.json", manifest)
    manifest["environment_after"] = environment_snapshot()
    write_json(args.out / "manifest.json", manifest)
    print(f"Raw receipts: {args.out.resolve() / 'manifest.json'}")
    return int(any(run["returncode"] for run in manifest["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
