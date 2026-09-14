"""One frozen procedure on twelve real sources and one protein-decoy control.

Training-only GBM proposals are heuristics. Validation chooses a small pair
set and basis resolution; fresh test outcomes never revise these choices.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import sys
import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

import benchmark_gbm_interactions as gbm
import benchmark_real_interactions as base
import benchmark_targeted_interactions as targeted
import numpy as np
from benchmark_housing_tensor import retained_model_storage, run_isolated

MAX_P = 512
MAX_Q = 96
MAX_REML_ITER = 100
MAX_ROWS = 300000
COUNTS = (0, 1, 2, 4)


def nominal_size(state, parent_k, pairs=()):
    specs = state["features"]
    widths = {
        name: parent_k - 1
        if spec["kind"] == "spline"
        else len(spec["levels"]) - 1
        if spec["kind"] == "categorical"
        else 1
        for name, spec in specs.items()
    }
    p = sum(widths.values())
    q = sum(spec["kind"] == "spline" for spec in specs.values())
    for left, right in pairs:
        a, b = specs[left]["kind"], specs[right]["kind"]
        if {a, b} == {"spline", "numeric"}:
            raise ValueError("Unsupported spline/numeric interaction")
        p += widths[left] * widths[right]
        if a == b == "spline":
            q += 2
        elif a == "spline":
            q += widths[right]
        elif b == "spline":
            q += widths[left]
    return {"P": p, "q": q}


def admit_pairs(state, pairs):
    accepted, skipped = [], []
    seen = set()
    for pair in pairs:
        left, right = pair
        key = frozenset(pair)
        if len(key) != 2 or key in seen or not key <= state["features"].keys():
            raise ValueError("Proposal pairs must be distinct known predictors without duplicates")
        seen.add(key)
        if len(accepted) == max(COUNTS):
            skipped.append({"pair": pair, "reason": "pair_count_cap"})
            continue
        kinds = {state["features"][name]["kind"] for name in pair}
        if kinds == {"spline", "numeric"}:
            skipped.append({"pair": pair, "reason": "unsupported_parent_types"})
            continue
        estimate = nominal_size(state, 6, [*accepted, pair])
        if estimate["P"] > MAX_P or estimate["q"] > MAX_Q:
            skipped.append({"pair": pair, "reason": "representation_budget", "nominal": estimate})
            continue
        accepted.append(list(pair))
    return {
        "pairs": accepted,
        "parent_resolutions": [4, 6]
        if any(spec["kind"] == "spline" for spec in state["features"].values())
        else [4],
        "skipped": skipped,
        "final_nominal": nominal_size(state, 6, accepted),
        "budget": {"P": MAX_P, "q": MAX_Q, "pairs": max(COUNTS), "admission_parent_k": 6},
    }


def build_model(state, family, parent_k, pairs):
    model = targeted.build_targeted_model(state, family, parent_k, [])
    for left, right in pairs:
        kinds = {state["features"][name]["kind"] for name in (left, right)}
        options = {"n_knots": (parent_k - 2, parent_k - 2)} if kinds == {"spline"} else {}
        model._add_interaction(left, right, **options)
    return model


def score(response, prediction, family, weights):
    from sklearn.metrics import mean_poisson_deviance, mean_squared_error

    if response.shape != prediction.shape or response.shape != weights.shape:
        raise ValueError("Response, prediction and weight shapes must match")
    if not all(np.isfinite(value).all() for value in (response, prediction, weights)):
        raise ValueError("Response, prediction and weights must be finite")
    if np.any(weights < 0) or not math.isfinite(float(weights.sum())) or weights.sum() <= 0:
        raise ValueError("Weights must be nonnegative with finite positive sum")
    mse = float(mean_squared_error(response, prediction, sample_weight=weights))
    loss = (
        float(mean_poisson_deviance(response, prediction, sample_weight=weights))
        if family == "poisson"
        else mse
    )
    if family not in ("poisson", "gaussian"):
        raise ValueError("This fixed batch supports Gaussian and Poisson responses")
    return {
        "primary_loss": loss,
        "mse": mse,
        "rows": len(response),
        "weight_sum": float(weights.sum()),
    }


def convergence_status(telemetry, nominal_q):
    reml = telemetry["reml"]
    required = nominal_q > 0
    if required and not reml["enabled"]:
        raise ValueError("Missing REML result for a model with smoothing parameters")
    inner = bool(telemetry["fit"]["converged"])
    outer = bool(reml.get("converged", False)) if required else None
    return {
        "coefficient_converged": inner,
        "reml_required": required,
        "reml_converged": outer,
        "combined_converged": inner and (outer if required else True),
    }


def choose_models(records):
    chosen = targeted.select_variant(records)
    additive = targeted.select_variant(
        {name: item for name, item in records.items() if name.endswith("_s0")}
    )
    matching = None if chosen is None else chosen.split("_")[0] + "_s0"
    evaluation = set()
    if additive is not None and chosen is not None:
        evaluation.update((additive, chosen))
        if records.get(matching, {}).get("status") == "converged":
            evaluation.add(matching)
    matching_available = records.get(matching, {}).get("status") == "converged"
    return {
        "chosen_arm": chosen,
        "additive_arm": additive,
        "matching_additive_arm": matching,
        "evaluation_arms": sorted(evaluation),
        "matching_comparison_status": "available" if matching_available else "unavailable",
        "interaction_comparison_eligible": bool(
            matching_available and additive is not None and chosen != matching
        ),
    }


def persist_choice(case_root, records):
    choice = {
        **choose_models(records),
        "selected_utc": datetime.now(UTC).isoformat(),
        "validation_scores": {name: item.get("validation") for name, item in records.items()},
        "fit_result_sha256": {
            name: gbm.file_hash(case_root / name / "result.json")
            for name, item in records.items()
            if item["status"] == "converged"
        },
    }
    with (case_root / "choice.json").open("x") as stream:
        json.dump(choice, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return choice


def source_identity():
    return {
        "existing": gbm.source_identity(),
        "new_files": {
            name: gbm.file_hash(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "broad_interaction_data.py",
                "interaction_pair_proposals.py",
                "benchmark_targeted_interactions.py",
            )
        },
    }


def prepared(args, record):
    import broad_interaction_data as data

    started = time.perf_counter()
    result = data.load_prepared(args.dataset, data_root=args.data_root)
    if len(result["frame"]) > MAX_ROWS:
        raise ValueError("Full source exceeds declared row budget")
    record["loading_and_preprocessing_seconds"] = time.perf_counter() - started
    record["data"] = result["metadata"]
    record["data_identity_sha256"] = base.digest_json(result["metadata"])
    record["rows"] = {name: len(rows) for name, rows in result["rows"].items()}
    record["raw_predictor_count"] = len(result["entry"]["features"])
    return result


def partition(result, name):
    import broad_interaction_data as data

    frame = result["frame"].iloc[result["rows"][name]]
    return (
        frame.loc[:, result["entry"]["features"]],
        data.response_values(frame, result["entry"]),
        np.asarray(result["sample_weight"])[result["rows"][name]],
    )


def proposal_worker(args, record):
    from interaction_pair_proposals import discover_pairs

    result = prepared(args, record)
    raw, y, weight = partition(result, "train")
    record["status"] = "proposing"
    base.write_json(args.output / "result.json", record)
    record["proposal"] = discover_pairs(
        raw, result["state"], y, record["data"]["family"], sample_weight=weight
    )
    record["admission"] = admit_pairs(result["state"], record["proposal"]["pairs"])
    record["status"] = "proposed"


def fit_worker(args, record):
    result = prepared(args, record)
    proposal_path = args.case_root / "proposals" / "result.json"
    proposal = json.loads(proposal_path.read_text())
    if (
        proposal["status"] != "proposed"
        or record["data_identity_sha256"] != proposal["data_identity_sha256"]
    ):
        raise ValueError("Proposal data identity or completion differs")
    if (
        proposal["source"] != record["source"]
        or proposal["protocol_sha256"] != record["protocol_sha256"]
    ):
        raise ValueError("Proposal source or protocol differs")
    k, count = (int(part[1:]) for part in args.arm.split("_"))
    available = proposal["admission"]["pairs"]
    if k not in proposal["admission"]["parent_resolutions"] or count not in {
        min(n, len(available)) for n in COUNTS
    }:
        raise ValueError("Arm is outside the declared menu")
    pairs = available[:count]
    estimate = nominal_size(result["state"], k, pairs)
    record.update(
        parent_k=k,
        pairs=pairs,
        nominal=estimate,
        proposal_result_sha256=gbm.file_hash(proposal_path),
    )
    if estimate["P"] > MAX_P or estimate["q"] > MAX_Q:
        record["status"] = "representation_refused"
        return
    raw, y, weight = partition(result, "train")
    valid_raw, valid_y, valid_weight = partition(result, "valid")
    started = time.perf_counter()
    train = base.transform_features(raw, result["state"])
    valid = base.transform_features(valid_raw, result["state"])
    model = build_model(result["state"], record["data"]["family"], k, pairs)
    record["transform_and_model_setup_seconds"] = time.perf_counter() - started
    record["max_reml_iter"] = MAX_REML_ITER
    record["numerical_tolerances"] = "unchanged production defaults"
    record["status"] = "fitting"
    base.write_json(args.output / "result.json", record)
    started = time.perf_counter()
    try:
        model.fit_reml(train, y, sample_weight=weight, max_reml_iter=MAX_REML_ITER)
    finally:
        record["fit_seconds"] = time.perf_counter() - started
        record["fit_end_peak_process_rss_mib"] = gbm.peak_rss()
    record["retained_model_storage"] = retained_model_storage(model)
    telemetry = model.training_telemetry()
    record.update(
        telemetry=telemetry,
        P=len(model.result.beta),
        q=len(telemetry["reml"]["lambdas"]),
        resolved_direct_backend=model.result.direct_backend,
        backend_group_counts=dict(
            collections.Counter(type(group).__name__ for group in model._dm.group_matrices)
        ),
        interaction_classes={
            name: type(spec).__name__ for name, spec in model._interaction_specs.items()
        },
    )
    if record["P"] > MAX_P or record["q"] > MAX_Q:
        raise ValueError("Actual representation exceeds the admitted budget")
    record["convergence"] = convergence_status(telemetry, estimate["q"])
    converged = record["convergence"]["combined_converged"]
    record["status"] = "converged" if converged else "not_converged"
    if not converged:
        return
    started = time.perf_counter()
    record["validation"] = score(
        valid_y, model.predict(valid), record["data"]["family"], valid_weight
    )
    record["validation_prediction_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    model_path = args.output / "model.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    record["model_pickle_sha256"] = gbm.file_hash(model_path)
    record["model_pickle_bytes"] = model_path.stat().st_size
    record["serialization_seconds"] = time.perf_counter() - started


def evaluation_worker(args, record):
    choice_path = args.case_root / "choice.json"
    choice = json.loads(choice_path.read_text())
    if args.arm not in choice["evaluation_arms"] or choice["additive_arm"] is None:
        raise ValueError("Evaluation requires a saved choice and converged additive comparator")
    fits = {}
    for arm, expected in choice["fit_result_sha256"].items():
        path = args.case_root / arm / "result.json"
        if gbm.file_hash(path) != expected:
            raise ValueError("Fit receipt changed after validation selection")
        fits[arm] = json.loads(path.read_text())
    recomputed = choose_models(fits)
    if any(choice[key] != value for key, value in recomputed.items()):
        raise ValueError("Frozen evaluation plan disagrees with validation losses")
    fitted = fits[args.arm]
    result = prepared(args, record)
    for key in ("source", "protocol_sha256", "data_identity_sha256"):
        if record[key] != fitted[key]:
            raise ValueError(f"Evaluation identity changed at {key}")
    model_path = args.output / "model.pkl"
    if gbm.file_hash(model_path) != fitted["model_pickle_sha256"]:
        raise ValueError("Owned model hash differs from the fit receipt")
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    raw, y, weight = partition(result, "test")
    started = time.perf_counter()
    prediction = model.predict(base.transform_features(raw, result["state"]))
    record["test"] = score(y, prediction, record["data"]["family"], weight)
    record["test_prediction_seconds"] = time.perf_counter() - started
    record["choice_sha256"] = gbm.file_hash(choice_path)
    prediction_path = args.output / "test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        row_index=result["rows"]["test"],
        response=y,
        prediction=prediction,
        sample_weight=weight,
    )
    record["test_predictions_sha256"] = gbm.file_hash(prediction_path)
    record["status"] = "evaluated"


def worker(args):
    started = time.perf_counter()
    record = {
        "dataset": args.dataset,
        "arm": args.arm,
        "stage": args.stage,
        "status": "starting",
        "started_utc": datetime.now(UTC).isoformat(),
    }
    result_path = args.output / ("evaluation.json" if args.stage == "evaluate" else "result.json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            record["source"] = source_identity()
            protocol_path = args.case_root.parent / "protocol.json"
            protocol = json.loads(protocol_path.read_text())
            if protocol["source"] != record["source"]:
                raise ValueError("Source changed after protocol freeze")
            record["protocol_sha256"] = gbm.file_hash(protocol_path)
            record["runtime"] = gbm.runtime()
            {"propose": proposal_worker, "fit": fit_worker, "evaluate": evaluation_worker}[
                args.stage
            ](args, record)
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
            record["worker_end_peak_process_rss_mib"] = gbm.peak_rss()
            record["finished_utc"] = datetime.now(UTC).isoformat()
            base.write_json(result_path, record)
    return 0 if record["status"] in ("proposed", "converged", "evaluated") else 1


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
        "--data-root",
        str(args.data_root),
    ]
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    receipt = run_isolated(command, log_path=output / f"{stage}.log", timeout=timeout, env=env)
    receipt.update(command=command, timeout_seconds=timeout)
    base.write_json(output / f"{stage}_process.json", receipt)
    path = output / ("evaluation.json" if stage == "evaluate" else "result.json")
    record = (
        json.loads(path.read_text())
        if path.exists()
        else {"status": "error", "error": "Worker produced no receipt"}
    )
    if receipt["status"] == "timeout":
        record.update(status="timeout", warnings_complete=False)
    elif receipt["status"] != "success" and record["status"] not in (
        "not_converged",
        "representation_refused",
    ):
        record["status"] = "error"
    record["process"] = receipt
    print(
        json.dumps(
            {
                "dataset": dataset,
                "arm": arm,
                "stage": stage,
                "status": record["status"],
                "seconds": round(receipt["process_seconds"], 3),
            }
        ),
        flush=True,
    )
    return record


def remaining_budget(args, spent, case_spent, stage):
    if stage == "evaluate":
        return max(0.0, min(args.total_budget - spent, args.case_budget - case_spent))
    return max(0.0, min(0.8 * args.total_budget - spent, 0.75 * args.case_budget - case_spent))


def run_suite(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source": source_identity(),
        "started_utc": datetime.now(UTC).isoformat(),
        "datasets": args.datasets,
        "parent_k": [4, 6],
        "zero_smoothing_policy": "One k label when there are no spline parents; coefficient convergence only because no smoothing optimization is needed",
        "prefix_counts": list(COUNTS),
        "short_list_policy": "Truncate counts to available admitted pairs and remove duplicates",
        "max_reml_iter": MAX_REML_ITER,
        "numerical_tolerances": "unchanged production defaults",
        "max_p": MAX_P,
        "max_q": MAX_Q,
        "max_source_rows": MAX_ROWS,
        "fit_timeout_seconds": args.fit_timeout,
        "proposal_timeout_seconds": 90,
        "per_dataset_all_worker_budget_seconds": args.case_budget,
        "total_all_worker_budget_seconds": args.total_budget,
        "search_budget_fraction_global": 0.8,
        "search_budget_fraction_per_dataset": 0.75,
        "selection": "Minimum validation loss among converged fits; ties smaller actual P then name",
        "evaluation": "All choices persisted before any test; winner, best validation-selected additive, and matching-k additive if different",
        "proposal": "Fixed pairwise HGBT leaves15 and 200 rounds on training only; path split-gain ranking is a heuristic",
        "claims": "One fixed procedure; twelve real sources and one protein-decoy control; no guaranteed discovery or optimal basis count",
    }
    base.write_json(args.output / "protocol.json", protocol)
    suite = {"schema_version": 1, "protocol": protocol, "datasets": {}}
    spent = 0.0
    for dataset in args.datasets:
        case_root = args.output / dataset
        case_root.mkdir()
        case_spent = 0.0
        if remaining_budget(args, spent, case_spent, "propose") <= 0:
            suite["datasets"][dataset] = {"status": "budget_exhausted", "arms": {}}
            continue
        proposal = launch(
            args,
            dataset,
            "proposals",
            "propose",
            min(90, remaining_budget(args, spent, case_spent, "propose")),
        )
        spent += proposal["process"]["process_seconds"]
        case_spent += proposal["process"]["process_seconds"]
        case = {"proposal": proposal, "arms": {}}
        suite["datasets"][dataset] = case
        if proposal["status"] != "proposed":
            case["status"] = "proposal_failed"
        else:
            count = len(proposal["admission"]["pairs"])
            resolutions = proposal["admission"]["parent_resolutions"]
            # Baselines first at both resolutions, preserving comparison under a case deadline.
            menu = [(k, 0) for k in resolutions] + [
                (k, n) for n in sorted({min(n, count) for n in COUNTS} - {0}) for k in resolutions
            ]
            for k, n in menu:
                arm = f"k{k}_s{n}"
                remaining = remaining_budget(args, spent, case_spent, "fit")
                if remaining <= 0:
                    case["arms"][arm] = {"status": "budget_exhausted"}
                else:
                    record = launch(args, dataset, arm, "fit", min(args.fit_timeout, remaining))
                    spent += record["process"]["process_seconds"]
                    case_spent += record["process"]["process_seconds"]
                    case["arms"][arm] = record
                suite["search_worker_process_seconds"] = spent
                base.write_json(args.output / "suite.json", suite)
            case["choice"] = persist_choice(case_root, case["arms"])
            case["status"] = (
                "search_complete"
                if all(r["status"] == "converged" for r in case["arms"].values())
                else "search_incomplete"
            )
        case["search_worker_process_seconds"] = case_spent
        suite["search_worker_process_seconds"] = spent
        base.write_json(args.output / "suite.json", suite)
    suite["all_choices_persisted_utc"] = datetime.now(UTC).isoformat()
    suite["all_worker_process_seconds"] = spent
    base.write_json(args.output / "suite.json", suite)
    for dataset, case in suite["datasets"].items():
        case_spent = case.get("search_worker_process_seconds", 0.0)
        case["evaluation_worker_process_seconds"] = 0.0
        case["all_worker_process_seconds"] = case_spent
        for arm in case.get("choice", {}).get("evaluation_arms", []):
            remaining = remaining_budget(args, spent, case_spent, "evaluate")
            if remaining <= 0:
                evaluation = {"status": "budget_exhausted"}
            else:
                evaluation = launch(args, dataset, arm, "evaluate", min(60, remaining))
                cost = evaluation["process"]["process_seconds"]
                spent += cost
                case_spent += cost
                case["evaluation_worker_process_seconds"] += cost
            case["arms"][arm]["evaluation"] = evaluation
            case["all_worker_process_seconds"] = case_spent
            suite["all_worker_process_seconds"] = spent
            base.write_json(args.output / "suite.json", suite)
    suite["suite_end_to_end_seconds"] = time.perf_counter() - started
    suite["finished_utc"] = datetime.now(UTC).isoformat()
    base.write_json(args.output / "suite.json", suite)
    complete = all(
        case.get("status") == "search_complete"
        and case["choice"]["evaluation_arms"]
        and all(
            case["arms"][arm]["evaluation"]["status"] == "evaluated"
            for arm in case["choice"]["evaluation_arms"]
        )
        for case in suite["datasets"].values()
    )
    return 0 if complete else 1


def main(argv=None):
    import broad_interaction_data as data
    from interaction_datasets import DEFAULT_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=data.DATASETS, default=list(data.DATASETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fit-timeout", type=float, default=120)
    parser.add_argument("--case-budget", type=float, default=240)
    parser.add_argument("--total-budget", type=float, default=1800)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=data.DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("propose", "fit", "evaluate"), help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name, maximum in (("fit_timeout", 120), ("case_budget", 240), ("total_budget", 1800)):
        if not math.isfinite(getattr(args, name)) or not 0 < getattr(args, name) <= maximum:
            parser.error(f"{name} must be finite, positive and <= {maximum}")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Datasets must be unique")
    for name in ("output", "data_root", "case_root"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    return worker(args) if args.worker else run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
