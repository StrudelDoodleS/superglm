"""Bounded mixed-interaction ablations with matching parent-basis controls.

The old test splits are exploratory audits, having already been inspected.
This runner does not implement automatic interaction or basis discovery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import benchmark_real_interactions as base
from benchmark_housing_tensor import retained_model_storage, run_isolated, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
MAX_REML_ITER = 100
DATASETS = ("uci_bike_sharing", "ames_housing")


def build_targeted_model(state, family, parent_k, pairs):
    from superglm import Categorical, Numeric, Spline, SuperGLM

    features = {}
    for name, spec in state["features"].items():
        if spec["kind"] == "spline":
            features[name] = Spline(kind="cr", k=parent_k, knot_strategy="uniform", penalty="ssp")
        elif spec["kind"] == "categorical":
            features[name] = Categorical(levels=spec["levels"])
        else:
            features[name] = Numeric()
    return SuperGLM(
        family=family,
        features=features,
        interactions=[tuple(pair) for pair in pairs],
        selection_penalty=0.0,
        discrete=True,
        n_bins=64,
    )


def variant_menu(dataset):
    if dataset == "uci_bike_sharing":
        pairs = {
            "additive": [],
            "clock": [["hr", "workingday"]],
            "clock_weather": [["hr", "workingday"], ["temp", "hr"]],
        }
    elif dataset == "ames_housing":
        pairs = {"additive": [], "area_type": [["Gr Liv Area", "Bldg Type"]]}
    else:
        raise ValueError(f"No targeted menu for {dataset}")
    return [
        {"arm": f"k{k}_{name}", "parent_k": k, "pairs": value}
        for k in (4, 6)
        for name, value in pairs.items()
    ]


def select_variant(records):
    candidates = [
        (record["validation"]["primary_loss"], record["P"], name)
        for name, record in records.items()
        if record.get("status") == "converged"
        and math.isfinite(record.get("validation", {}).get("primary_loss", math.inf))
    ]
    return min(candidates)[2] if candidates else None


def code_identity():
    files = (
        "benchmark_targeted_interactions.py",
        "benchmark_real_interactions.py",
        "benchmark_housing_tensor.py",
        "interaction_datasets.py",
    )
    hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in files
    }
    return {
        "package_source_sha256": source_fingerprint(),
        "benchmark_script_sha256": base.digest_json(hashes),
        "benchmark_files_sha256": hashes,
    }


def fit_worker(args, record):
    from interaction_datasets import load_dataset, read_manifest
    from threadpoolctl import threadpool_info

    entry = next(item for item in read_manifest(args.manifest) if item["id"] == args.dataset)
    variant = next(item for item in variant_menu(args.dataset) if item["arm"] == args.arm)
    started = time.perf_counter()
    frame = load_dataset(args.dataset, root=args.data_root, manifest=args.manifest)
    record["loading_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    state, rows, split_hash = base.prepare_dataset(frame, entry)
    train = base.transform_features(frame.iloc[rows["train"]].loc[:, entry["features"]], state)
    valid = base.transform_features(frame.iloc[rows["valid"]].loc[:, entry["features"]], state)
    y = base.response_values(frame.iloc[rows["train"]], entry)
    valid_y = base.response_values(frame.iloc[rows["valid"]], entry)
    record.update(
        preprocessing_seconds=time.perf_counter() - started,
        variant=variant,
        max_reml_iter=MAX_REML_ITER,
        numerical_tolerances="unchanged production defaults",
        rows={name: len(indices) for name, indices in rows.items()},
        full_table_rows=len(frame),
        subsampling="none; every source row belongs to one split",
        raw_predictor_count=len(entry["features"]),
        family=base.family_for(entry),
        split=entry["split"],
        split_sha256=split_hash,
        preprocessing=state,
        preprocessing_sha256=base.digest_json(state),
        data_sha256=entry["source"]["sha256"],
        manifest_entry_sha256=base.digest_json(entry),
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("superglm", "numpy", "scipy", "pandas", "numba", "scikit-learn")
            },
            "threadpools": threadpool_info(),
        },
    )
    started = time.perf_counter()
    model = build_targeted_model(state, record["family"], variant["parent_k"], variant["pairs"])
    record["model_setup_seconds"] = time.perf_counter() - started
    record["status"] = "fitting"
    base.write_json(args.output / "result.json", record)
    started = time.perf_counter()
    try:
        model.fit_reml(train, y, max_reml_iter=MAX_REML_ITER)
    finally:
        record["fit_seconds"] = time.perf_counter() - started
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        record["fit_end_peak_process_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
        )
    record["retained_model_storage"] = retained_model_storage(model)
    telemetry = model.training_telemetry()
    record.update(
        telemetry=telemetry,
        P=len(model.result.beta),
        fitted_smoothing_parameter_count=len(telemetry["reml"]["lambdas"]),
        resolved_direct_backend=model.result.direct_backend,
        backend_groups=[type(group).__name__ for group in model._dm.group_matrices],
        interaction_classes={
            name: type(spec).__name__ for name, spec in model._interaction_specs.items()
        },
    )
    converged = bool(model.result.converged) and bool(telemetry["reml"]["converged"])
    record["status"] = "converged" if converged else "not_converged"
    if not converged:
        return
    started = time.perf_counter()
    record["validation"] = base.score_predictions(valid_y, model.predict(valid), record["family"])
    record["validation_prediction_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    model_path = args.output / "model.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with model_path.open("rb") as stream:
        record["model_pickle_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    record["serialization_seconds"] = time.perf_counter() - started
    record["model_pickle_bytes"] = model_path.stat().st_size


def worker(args):
    started = time.perf_counter()
    record = {"dataset": args.dataset, "arm": args.arm, "stage": args.stage, "status": "starting"}
    record["started_utc"] = datetime.now(UTC).isoformat()
    result_path = args.output / ("result.json" if args.stage == "fit" else "evaluation.json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            record.update(code_identity())
            protocol = json.loads((args.case_root.parent / "protocol.json").read_text())
            for key in ("package_source_sha256", "benchmark_script_sha256"):
                if protocol[key] != record[key]:
                    raise ValueError(f"Source changed after protocol at {key}")
            record["git_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            if args.stage == "fit":
                fit_worker(args, record)
            else:
                choice = json.loads((args.case_root / "choice.json").read_text())
                if args.arm not in choice["evaluation_arms"]:
                    raise ValueError("Arm is outside the persisted evaluation plan")
                base.evaluation_worker(args, record)
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
            record["traceback"] = traceback.format_exc()
        finally:
            record["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ]
            record["worker_seconds"] = time.perf_counter() - started
            record["finished_utc"] = datetime.now(UTC).isoformat()
            base.write_json(result_path, record)
    return 0 if record["status"] in {"converged", "evaluated"} else 1


def launch(args, dataset, arm, stage, timeout):
    case_root = args.output / dataset
    output = case_root / arm
    output.mkdir(parents=True, exist_ok=stage == "evaluate")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--dataset",
        dataset,
        "--arm",
        arm,
        "--stage",
        stage,
        "--output",
        str(output),
        "--case-root",
        str(case_root),
        "--manifest",
        str(args.manifest),
        "--data-root",
        str(args.data_root),
    ]
    env = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        env[variable] = "1"
    process = run_isolated(command, log_path=output / f"{stage}.log", timeout=timeout, env=env)
    process.update(command=command, timeout_seconds=timeout)
    base.write_json(output / f"{stage}_process.json", process)
    path = output / ("result.json" if stage == "fit" else "evaluation.json")
    record = base.load_worker_receipt(path, stage)
    if process["status"] == "timeout":
        record.update(status="timeout", warnings_complete=False)
    elif process["status"] != "success" and record.get("status") != "not_converged":
        record["status"] = "error"
    if "finished_utc" not in record:
        record.update(warnings_complete=False, parent_finished_utc=datetime.now(UTC).isoformat())
    base.write_json(path, record)
    record["process"] = process
    print(
        json.dumps(
            {
                "dataset": dataset,
                "arm": arm,
                "stage": stage,
                "status": record["status"],
                "process_seconds": process["process_seconds"],
            }
        ),
        flush=True,
    )
    return record


def persist_choice(case_root, records):
    chosen_by_k = {
        str(k): select_variant(
            {name: item for name, item in records.items() if name.startswith(f"k{k}_")}
        )
        for k in (4, 6)
    }
    evaluation = {
        name
        for k, winner in chosen_by_k.items()
        if winner is not None
        for name in (f"k{k}_additive", winner)
        if records[name]["status"] == "converged"
    }
    choice = {
        "chosen_arm": select_variant(records),
        "chosen_by_parent_k": chosen_by_k,
        "chosen_before_test_evaluation": True,
        "evaluation_arms": sorted(evaluation),
        "validation_scores": {name: record.get("validation") for name, record in records.items()},
        "fit_result_sha256": {
            name: hashlib.sha256((case_root / name / "result.json").read_bytes()).hexdigest()
            for name, record in records.items()
            if record["status"] == "converged"
        },
    }
    base.write_json(case_root / "choice.json", choice)
    return choice


def run_suite(args):
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        **code_identity(),
        "started_utc": datetime.now(UTC).isoformat(),
        "menus": {dataset: variant_menu(dataset) for dataset in args.datasets},
        "max_reml_iter": MAX_REML_ITER,
        "numerical_tolerances": "unchanged production defaults for every arm",
        "fit_worker_budget_seconds": args.total_fit_budget,
        "per_fit_worker_timeout_seconds": args.fit_timeout,
        "evaluation": "validation winner at each k plus its additive control; all choices saved before test",
        "test_status": "previously inspected test blocks; exploratory audit, not fresh confirmation",
        "candidate_source": "prior-informed fixed menu; no automatic discovery claim",
    }
    base.write_json(args.output / "protocol.json", protocol)
    suite = {"protocol": protocol, "datasets": {}}
    spent = 0.0
    for dataset in args.datasets:
        case_root = args.output / dataset
        case_root.mkdir()
        records = {}
        for variant in variant_menu(dataset):
            arm = variant["arm"]
            if spent >= args.total_fit_budget:
                records[arm] = {"status": "budget_exhausted"}
            else:
                records[arm] = launch(
                    args, dataset, arm, "fit", min(args.fit_timeout, args.total_fit_budget - spent)
                )
                spent += records[arm]["process"]["process_seconds"]
            suite["datasets"][dataset] = {"arms": records}
            suite["fit_worker_process_seconds_spent"] = spent
            base.write_json(args.output / "suite.json", suite)
        suite["datasets"][dataset]["choice"] = persist_choice(case_root, records)
        base.write_json(args.output / "suite.json", suite)
    for dataset, case in suite["datasets"].items():
        for arm in case["choice"]["evaluation_arms"]:
            case["arms"][arm]["evaluation"] = launch(
                args, dataset, arm, "evaluate", args.evaluation_timeout
            )
            base.write_json(args.output / "suite.json", suite)
    suite["finished_utc"] = datetime.now(UTC).isoformat()
    base.write_json(args.output / "suite.json", suite)
    return (
        0
        if all(
            record["status"] == "converged"
            and ("evaluation" not in record or record["evaluation"]["status"] == "evaluated")
            for case in suite["datasets"].values()
            for record in case["arms"].values()
        )
        else 1
    )


def main(argv=None):
    from interaction_datasets import DEFAULT_ROOT, MANIFEST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-timeout", type=float, default=120)
    parser.add_argument("--evaluation-timeout", type=float, default=30)
    parser.add_argument("--total-fit-budget", type=float, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("fit", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0
        for key in ("fit_timeout", "evaluation_timeout", "total_fit_budget")
    ):
        parser.error("Time budgets must be finite and positive")
    if args.fit_timeout > 180 or len(set(args.datasets)) != len(args.datasets):
        parser.error("At most 180 seconds per worker; unique datasets required")
    for key in ("output", "manifest", "data_root", "case_root"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    return worker(args) if args.worker else run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
