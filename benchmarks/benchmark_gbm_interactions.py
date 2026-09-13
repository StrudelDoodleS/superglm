"""Bounded GBM structural comparisons on the existing exploratory splits.

Run the fixed 24-fit menu in serial fresh processes. These already inspected
splits support exploratory diagnostics, not fresh confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
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

import numpy as np
import pandas as pd
from benchmark_housing_tensor import retained_model_storage, run_isolated, source_fingerprint
from benchmark_real_interactions import (
    DEFAULT_DATASETS,
    ROOT,
    SEED,
    digest_json,
    family_for,
    prepare_dataset,
    response_values,
    score_predictions,
    transform_features,
    write_json,
)

STRUCTURES = {"additive": "no_interactions", "pairwise": "pairwise", "unrestricted": None}
CONFIGS = {"leaves15": {"max_leaf_nodes": 15}, "leaves31": {"max_leaf_nodes": 31}}
COMMON = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "min_samples_leaf": 20,
    "max_bins": 255,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "early_stopping": False,
    "categorical_features": "from_dtype",
    "random_state": SEED,
}
ARMS = {
    f"{structure}_{config}": (structure, config) for structure in STRUCTURES for config in CONFIGS
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity():
    import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as implementation

    return {
        "package_source_sha256": source_fingerprint(ROOT / "src" / "superglm"),
        "benchmark_files_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "benchmark_real_interactions.py",
                "benchmark_housing_tensor.py",
                "interaction_datasets.py",
            )
        },
        "sklearn_gradient_boosting_py_sha256": file_hash(inspect.getfile(implementation)),
        "sklearn_version": importlib.metadata.version("scikit-learn"),
    }


def native_features(frame, state):
    """Keep the original adapter values; mark frozen unordered categories natively."""
    result = transform_features(frame, state)
    for name, spec in state["features"].items():
        if spec["kind"] == "categorical":
            result[name] = pd.Categorical(result[name], categories=spec["levels"], ordered=False)
            if result[name].isna().any():
                raise ValueError("Training category contract produced an unknown label")
    return result


def build_model(family, structure, config):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    loss = {"binomial": "log_loss", "gaussian": "squared_error", "poisson": "poisson"}[family]
    cls = HistGradientBoostingClassifier if family == "binomial" else HistGradientBoostingRegressor
    return cls(loss=loss, interaction_cst=STRUCTURES[structure], **COMMON, **CONFIGS[config])


def predict(model, frame, family):
    return model.predict_proba(frame)[:, 1] if family == "binomial" else model.predict(frame)


def metrics(response, prediction, family):
    result = score_predictions(response, prediction, family)
    if family == "binomial":
        result["class_counts"] = {str(value): int(np.sum(response == value)) for value in (0, 1)}
    elif family == "gaussian":
        result["rmse"] = math.sqrt(result["mse"])
    return result


def tree_diagnostics(model):
    """Inspect actual sklearn trees; branch feature counts are structural only."""
    trees = [tree for iteration in model._predictors for tree in iteration]
    largest_branch = 0
    for tree in trees:
        stack = [(0, frozenset())]
        while stack:
            index, features = stack.pop()
            node = tree.nodes[index]
            if node["is_leaf"]:
                largest_branch = max(largest_branch, len(features))
            else:
                features = features | {int(node["feature_idx"])}
                stack.extend((int(node[side]), features) for side in ("left", "right"))
    return {
        "estimator_class": f"{type(model).__module__}.{type(model).__name__}",
        "n_iter": int(model.n_iter_),
        "trees_per_iteration": int(model.n_trees_per_iteration_),
        "tree_count": len(trees),
        "node_count": sum(len(tree.nodes) for tree in trees),
        "leaf_count": sum(int(np.sum(tree.nodes["is_leaf"])) for tree in trees),
        "max_distinct_features_per_branch": largest_branch,
        "requested_interaction_cst": model.interaction_cst,
        "native_categorical_count": 0
        if model.is_categorical_ is None
        else int(np.sum(model.is_categorical_)),
        "scope": "Observed fitted HistGradientBoosting trees; path support does not prove statistical interaction strength",
    }


def choose_validation_models(records):
    """Select a capacity per class and a class overall; additive can win."""
    winners = {}
    for structure in STRUCTURES:
        candidates = []
        for config in CONFIGS:
            arm = f"{structure}_{config}"
            record = records.get(arm, {})
            loss = record.get("validation", {}).get("primary_loss", math.inf)
            if record.get("status") == "fitted" and math.isfinite(loss):
                candidates.append((loss, arm))
        winners[structure] = min(candidates)[1] if candidates else None
    ranked = [
        (records[arm]["validation"]["primary_loss"], index, arm)
        for index, arm in enumerate(winners.values())
        if arm is not None
    ]
    return {"selected_by_structure": winners, "chosen_arm": min(ranked)[2] if ranked else None}


def persist_choice(case_root, records):
    choice = {
        **choose_validation_models(records),
        "chosen_before_test_evaluation": True,
        "selected_utc": datetime.now(UTC).isoformat(),
        "validation_scores": {arm: record.get("validation") for arm, record in records.items()},
        "fit_result_sha256": {
            arm: file_hash(case_root / arm / "result.json")
            for arm, record in records.items()
            if record["status"] == "fitted"
        },
    }
    # Refuse to revise a selection once it has authorized test evaluation.
    with (case_root / "choice.json").open("x") as stream:
        json.dump(choice, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return choice


def runtime():
    from threadpoolctl import threadpool_info

    pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools):
        raise ValueError("Every observed numerical thread pool must use one thread")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl")
        },
        "threadpools": pools,
    }


def fit_worker(args, record):
    from interaction_datasets import load_dataset, read_manifest

    entry = next(entry for entry in read_manifest(args.manifest) if entry["id"] == args.dataset)
    if entry["schema"]["rows"] > 30000:
        raise ValueError("Declared full table exceeds the 30,000-row budget")
    started = time.perf_counter()
    frame = load_dataset(args.dataset, root=args.data_root, manifest=args.manifest)
    record["loading_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    state, rows, split_hash = prepare_dataset(frame, entry)
    train = native_features(frame.iloc[rows["train"]].loc[:, entry["features"]], state)
    valid = native_features(frame.iloc[rows["valid"]].loc[:, entry["features"]], state)
    y = response_values(frame.iloc[rows["train"]], entry)
    valid_y = response_values(frame.iloc[rows["valid"]], entry)
    record["preprocessing_seconds"] = time.perf_counter() - started
    record.update(
        rows={name: len(indices) for name, indices in rows.items()},
        full_table_rows=len(frame),
        subsampling="none; every source row belongs to one split",
        raw_predictor_count=len(entry["features"]),
        family=family_for(entry),
        split=entry["split"],
        split_sha256=split_hash,
        preprocessing=state,
        preprocessing_sha256=digest_json(state),
        data_sha256=entry["source"]["sha256"],
        manifest_entry_sha256=digest_json(entry),
    )
    started = time.perf_counter()
    model = build_model(record["family"], *ARMS[args.arm])
    record["model_setup_seconds"] = time.perf_counter() - started
    record["model_parameters"] = model.get_params()
    record["runtime"] = runtime()
    record["status"] = "fitting"
    write_json(args.output / "result.json", record)
    started = time.perf_counter()
    try:
        model.fit(train, y)
    finally:
        record["fit_seconds"] = time.perf_counter() - started
        record["fit_end_peak_process_rss_mib"] = peak_rss()
    record["retained_model_storage"] = retained_model_storage(model)
    record["tree_diagnostics"] = tree_diagnostics(model)
    structure, _ = ARMS[args.arm]
    bound = {"additive": 1, "pairwise": 2, "unrestricted": len(train.columns)}[structure]
    if record["tree_diagnostics"]["max_distinct_features_per_branch"] > bound:
        raise ValueError("Fitted tree violates its structural constraint")
    if model.n_iter_ != COMMON["max_iter"] or model.do_early_stopping_:
        raise ValueError("Fitted iteration budget differs from the preregistered menu")
    started = time.perf_counter()
    record["validation"] = metrics(
        valid_y, predict(model, valid, record["family"]), record["family"]
    )
    record["validation_prediction_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    model_path = args.output / "model.pkl"
    with model_path.open("xb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    record["model_pickle_sha256"] = file_hash(model_path)
    record["model_pickle_bytes"] = model_path.stat().st_size
    record["serialization_seconds"] = time.perf_counter() - started
    record["status"] = "fitted"  # Fixed boosting budget, not a convergence certificate.


def evaluation_worker(args, record):
    from interaction_datasets import load_dataset, read_manifest

    choice_path = args.case_root / "choice.json"
    if not choice_path.exists():
        raise ValueError("Test evaluation requires a persisted validation choice")
    choice = json.loads(choice_path.read_text())
    if choice.get("chosen_arm") is None or not choice.get("chosen_before_test_evaluation"):
        raise ValueError("Test evaluation requires a persisted validation choice")
    if args.arm not in choice["selected_by_structure"].values():
        raise ValueError("Test evaluation permits only validation-selected capacities")
    fits = {}
    for arm, expected in choice["fit_result_sha256"].items():
        path = args.case_root / arm / "result.json"
        if file_hash(path) != expected:
            raise ValueError("Fit result changed after validation choice")
        fits[arm] = json.loads(path.read_text())
    recomputed = choose_validation_models(fits)
    if any(recomputed[key] != choice[key] for key in recomputed):
        raise ValueError("Persisted choice differs from frozen validation results")
    fitted = fits[args.arm]
    if fitted["source"] != record["source"]:
        raise ValueError("Source changed after fit")
    if fitted["protocol_sha256"] != record["protocol_sha256"]:
        raise ValueError("Protocol changed after fit")
    entry = next(entry for entry in read_manifest(args.manifest) if entry["id"] == args.dataset)
    if digest_json(entry) != fitted["manifest_entry_sha256"]:
        raise ValueError("Manifest entry changed after fit")
    frame = load_dataset(args.dataset, root=args.data_root, manifest=args.manifest)
    state, rows, split_hash = prepare_dataset(frame, entry)
    if split_hash != fitted["split_sha256"] or digest_json(state) != fitted["preprocessing_sha256"]:
        raise ValueError("Split or preprocessing changed after fit")
    model_path = args.output / "model.pkl"
    if file_hash(model_path) != fitted["model_pickle_sha256"]:
        raise ValueError("Owned model artifact differs from fitted receipt")
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    started = time.perf_counter()
    test = native_features(frame.iloc[rows["test"]].loc[:, entry["features"]], state)
    test_y = response_values(frame.iloc[rows["test"]], entry)
    prediction = predict(model, test, fitted["family"])
    record["test"] = metrics(test_y, prediction, fitted["family"])
    record["test_prediction_seconds"] = time.perf_counter() - started
    record["choice_sha256"] = file_hash(choice_path)
    record["validation_chosen_arm"] = choice["chosen_arm"]
    np.savez_compressed(
        args.output / "test_predictions.npz",
        row_index=rows["test"],
        response=test_y,
        prediction=prediction,
    )
    record["test_predictions_sha256"] = file_hash(args.output / "test_predictions.npz")
    record["status"] = "evaluated"


def peak_rss():
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def worker(args):
    started = time.perf_counter()
    record = {
        "dataset": args.dataset,
        "arm": args.arm,
        "structure": ARMS[args.arm][0],
        "config": ARMS[args.arm][1],
        "stage": args.stage,
        "status": "starting",
        "started_utc": datetime.now(UTC).isoformat(),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            protocol_path = args.case_root.parent / "protocol.json"
            protocol = json.loads(protocol_path.read_text())
            record["source"] = source_identity()
            record["protocol_sha256"] = file_hash(protocol_path)
            if record["source"] != protocol["source"]:
                raise ValueError("Source changed after protocol freeze")
            record["git_head"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            if args.stage == "fit":
                fit_worker(args, record)
            else:
                evaluation_worker(args, record)
        except Exception as error:
            record.update(
                status="error",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        finally:
            record["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ]
            record["worker_seconds"] = time.perf_counter() - started
            record["worker_end_peak_process_rss_mib"] = peak_rss()
            record["finished_utc"] = datetime.now(UTC).isoformat()
            write_json(
                args.output / ("result.json" if args.stage == "fit" else "evaluation.json"), record
            )
    return 0 if record["status"] in {"fitted", "evaluated"} else 1


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
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = "1"
    receipt = run_isolated(command, log_path=output / f"{stage}.log", timeout=timeout, env=env)
    receipt.update(command=command, timeout_seconds=timeout)
    path = output / ("result.json" if stage == "fit" else "evaluation.json")
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "error", "error": "Worker produced no receipt"}
    )
    if receipt["status"] != "success":
        result["status"] = receipt["status"]
        result["warnings_complete"] = receipt["status"] != "timeout"
    result["process"] = receipt
    write_json(path, result)
    print(
        json.dumps(
            {
                "dataset": dataset,
                "arm": arm,
                "stage": stage,
                "status": result["status"],
                "process_seconds": receipt["process_seconds"],
            }
        ),
        flush=True,
    )
    return result


def run_suite(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol = {
        "source": source_identity(),
        "datasets": list(DEFAULT_DATASETS),
        "structures": STRUCTURES,
        "capacity_menu": CONFIGS,
        "common_parameters": COMMON,
        "registered_utc": datetime.now(UTC).isoformat(),
        "interpretation": "Exploratory same-split diagnostic; test outcomes were inspected in the earlier SuperGLM pilot",
        "selection": "Minimum validation primary loss within each class, then across classes; exact class ties prefer additive, then pairwise; capacity ties prefer fewer leaves",
        "evaluation": "Persist all dataset choices before any test evaluation; evaluate only each class's selected capacity without refitting",
        "memory_scope": "Fit-end RSS includes imports, raw data, preprocessing and fit; retained payload excludes Python object overhead and unknown extension allocations; pickle contains estimator only, adapter state is separate",
        "fit_worker_budget_seconds": args.total_fit_budget,
        "per_fit_worker_timeout_seconds": args.fit_timeout,
        "per_evaluation_worker_timeout_seconds": args.evaluation_timeout,
    }
    write_json(args.output / "protocol.json", protocol)
    suite = {"schema_version": 1, "protocol": protocol, "datasets": {}}
    spent = 0.0
    for dataset in DEFAULT_DATASETS:
        case_started = time.perf_counter()
        records = {}
        # Interleave classes within capacity so an exhausted cap remains explicit.
        for config in CONFIGS:
            for structure in STRUCTURES:
                arm = f"{structure}_{config}"
                if spent >= args.total_fit_budget:
                    records[arm] = {"status": "budget_exhausted"}
                else:
                    records[arm] = launch(
                        args,
                        dataset,
                        arm,
                        "fit",
                        min(args.fit_timeout, args.total_fit_budget - spent),
                    )
                    spent += records[arm]["process"]["process_seconds"]
        case_root = args.output / dataset
        case_root.mkdir(parents=True, exist_ok=True)
        choice = persist_choice(case_root, records)
        suite["datasets"][dataset] = {
            "arms": records,
            "choice": choice,
            "matched_menu_completed": all(
                record["status"] == "fitted" for record in records.values()
            ),
            "costs": {
                "all_tuning_and_selection_wall_seconds": time.perf_counter() - case_started,
                "all_fit_worker_process_seconds": sum(
                    record.get("process", {}).get("process_seconds", 0.0)
                    for record in records.values()
                ),
                "all_model_fit_seconds": sum(
                    record.get("fit_seconds", 0.0) for record in records.values()
                ),
                "class_tuning_process_seconds": {
                    structure: sum(
                        record.get("process", {}).get("process_seconds", 0.0)
                        for arm, record in records.items()
                        if ARMS[arm][0] == structure
                    )
                    for structure in STRUCTURES
                },
            },
        }
        suite["fit_worker_process_seconds_spent"] = spent
        write_json(args.output / "suite.json", suite)
    suite["all_choices_persisted_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "suite.json", suite)
    for dataset, case in suite["datasets"].items():
        for arm in case["choice"]["selected_by_structure"].values():
            if arm is not None:
                case["arms"][arm]["evaluation"] = launch(
                    args, dataset, arm, "evaluate", args.evaluation_timeout
                )
        case["costs"]["all_evaluation_worker_process_seconds"] = sum(
            record.get("evaluation", {}).get("process", {}).get("process_seconds", 0.0)
            for record in case["arms"].values()
        )
        case["costs"]["all_tuning_selection_and_evaluation_seconds"] = (
            case["costs"]["all_tuning_and_selection_wall_seconds"]
            + case["costs"]["all_evaluation_worker_process_seconds"]
        )
        write_json(args.output / "suite.json", suite)
    suite["suite_end_to_end_seconds"] = time.perf_counter() - started
    suite["finished_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output / "suite.json", suite)
    return (
        0
        if all(
            case["matched_menu_completed"]
            and all(
                case["arms"][arm].get("evaluation", {}).get("status") == "evaluated"
                for arm in case["choice"]["selected_by_structure"].values()
            )
            for case in suite["datasets"].values()
        )
        else 1
    )


def main(argv=None):
    from interaction_datasets import DEFAULT_ROOT, MANIFEST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fit-timeout", type=float, default=120)
    parser.add_argument("--total-fit-budget", type=float, default=600)
    parser.add_argument("--evaluation-timeout", type=float, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DEFAULT_DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("fit", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name, cap in (("fit_timeout", 120), ("total_fit_budget", 600), ("evaluation_timeout", 120)):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0 or value > cap:
            parser.error(f"--{name.replace('_', '-')} must be positive and at most {cap} seconds")
    if args.worker and any(
        getattr(args, name) is None for name in ("dataset", "arm", "stage", "case_root")
    ):
        parser.error("Worker requires dataset, arm, stage and case-root")
    for name in ("output", "manifest", "data_root"):
        setattr(args, name, getattr(args, name).resolve())
    return worker(args) if args.worker else run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
