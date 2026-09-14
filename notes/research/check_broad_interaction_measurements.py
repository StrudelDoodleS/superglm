"""Audit the frozen broad run and emit a compact, reproducible measurement record.

This replays stored scores and data preparation, without fitting models or
loading model pickles. Run from the measured worktree with its Python environment.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

if not __debug__:
    raise RuntimeError("Research audit requires enabled assertions; remove -O/PYTHONOPTIMIZE")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCE_PARSER = argparse.ArgumentParser(add_help=False)
SOURCE_PARSER.add_argument("--source-root", type=Path, default=REPO)
SOURCE_REPO = SOURCE_PARSER.parse_known_args()[0].source_root.resolve()
sys.path[:0] = [str(SOURCE_REPO / "benchmarks"), str(SOURCE_REPO / "src")]

import benchmark_broad_interactions as broad  # noqa: E402
import benchmark_gbm_interactions as gbm  # noqa: E402
import benchmark_real_interactions as base  # noqa: E402
import broad_interaction_data as data  # noqa: E402


def read_json(path):
    return json.loads(path.read_text())


def data_identity(metadata):
    """Exclude only local locations; retain all content, code and split evidence."""
    return {
        key: value
        for key, value in metadata.items()
        if key not in ("source_path", "source_registry")
    }


def check_menu(protocol, admission, arms):
    counts = {0, *(min(count, len(admission["pairs"])) for count in protocol["prefix_counts"])}
    expected = {f"k{k}_s{count}" for k in admission["parent_resolutions"] for count in counts}
    if set(arms) != expected:
        raise ValueError(
            f"Recorded arm menu differs: missing={expected - set(arms)}, extra={set(arms) - expected}"
        )
    return len(expected)


def compare_selected_models(choice, summaries):
    if choice["chosen_arm"] is None or choice["additive_arm"] is None:
        raise ValueError(
            "Audit scope requires a selected model and a converged additive comparator"
        )
    winner, additive = (summaries[choice[key]] for key in ("chosen_arm", "additive_arm"))
    matching = (
        summaries[choice["matching_additive_arm"]]
        if choice["matching_comparison_status"] == "available"
        else None
    )
    winner_loss = winner["evaluation"]["test"]["primary_loss"]
    gains = {
        label: 100 * (1 - winner_loss / baseline["evaluation"]["test"]["primary_loss"])
        if baseline is not None
        else None
        for label, baseline in (
            ("vs_best_additive_percent", additive),
            ("vs_matching_k_percent", matching),
        )
    }
    outcome = (
        "additive_retained"
        if not winner["pairs"]
        else "matching_additive_unavailable"
        if matching is None
        else "selected_interactions_lower_test_loss"
        if min(gains.values()) > 0
        else "selected_interactions_not_lower_than_both_controls"
    )
    return {
        "outcome": outcome,
        "test_comparison": gains,
        "selected_fit_ratio_vs_best_additive": winner["fit_seconds"] / additive["fit_seconds"],
        "selected_fit_ratio_vs_matching_k": winner["fit_seconds"] / matching["fit_seconds"]
        if matching is not None
        else None,
    }


def check_sum(values, recorded):
    """Allow the rounding bound for a sequential sum of positive durations."""
    terms = list(values)
    n_eps = (len(terms) + 2) * np.finfo(float).eps
    gamma = n_eps / (1 - n_eps)
    assert math.isclose(math.fsum(terms), recorded, rel_tol=gamma, abs_tol=0)


def fit_summary(record):
    keys = (
        "status",
        "parent_k",
        "pairs",
        "nominal",
        "P",
        "q",
        "fit_seconds",
        "fit_end_peak_process_rss_mib",
        "retained_model_storage",
        "resolved_direct_backend",
        "backend_group_counts",
        "interaction_classes",
        "convergence",
        "validation",
        "validation_prediction_seconds",
        "model_pickle_sha256",
        "model_pickle_bytes",
        "loading_and_preprocessing_seconds",
        "transform_and_model_setup_seconds",
        "serialization_seconds",
        "worker_seconds",
        "worker_end_peak_process_rss_mib",
        "warnings",
        "warnings_complete",
        "error",
        "traceback",
        "started_utc",
        "finished_utc",
        "parent_finished_utc",
        "incomplete_receipt",
    )
    result = {key: record[key] for key in keys if key in record}
    telemetry = record.get("telemetry", {})
    result["coefficient_telemetry"] = telemetry.get("fit")
    reml = telemetry.get("reml", {})
    result["reml_telemetry"] = {
        key: reml[key]
        for key in ("enabled", "n_reml_iter", "converged", "termination_reason", "objective")
        if key in reml
    }
    result["process_seconds"] = record.get("process", {}).get("process_seconds", 0)
    result["process_status"] = record.get("process", {}).get("status")
    return result


def summarize(root, data_root=data.DEFAULT_ROOT):
    protocol = read_json(root / "protocol.json")
    suite = read_json(root / "suite.json")
    assert suite["protocol"] == protocol
    assert broad.source_identity() == protocol["source"], (
        "Source differs from the frozen protocol; use --source-root and its recorded environment"
    )
    protocol_hash = gbm.file_hash(root / "protocol.json")
    receipt_index = {}

    def index(path):
        digest = gbm.file_hash(path)
        receipt_index[str(path.relative_to(root))] = digest
        return digest

    index(root / "protocol.json")
    index(root / "suite.json")
    cases = {}
    all_process_costs, search_costs, evaluation_costs = [], [], []
    evaluation_starts, choice_times = [], []
    fit_statuses, evaluation_statuses = collections.Counter(), collections.Counter()
    missing_total_fields = []
    incomplete_identity_workers = []
    unverified_threadpool_workers = []
    declared_arm_count = 0
    for name, case in suite["datasets"].items():
        if case.get("status") == "budget_exhausted":
            raise ValueError(f"Audit scope excludes datasets skipped before proposal: {name}")
        case_root = root / name
        prepared = data.load_prepared(name, data_root=data_root)
        proposal = case["proposal"]
        assert proposal["status"] == "proposed"
        assert data_identity(proposal["data"]) == data_identity(prepared["metadata"])
        assert proposal["data_identity_sha256"] == base.digest_json(proposal["data"])
        if (
            broad.admit_pairs(prepared["state"], proposal["proposal"]["pairs"])
            != proposal["admission"]
        ):
            raise ValueError("Recorded admission differs from re-prepared state and proposed pairs")
        declared_arm_count += check_menu(protocol, proposal["admission"], case["arms"])
        proposal_hash = index(case_root / "proposals" / "result.json")
        choice = read_json(case_root / "choice.json")
        assert choice == case["choice"]
        decision = broad.choose_models(case["arms"])
        assert all(choice[key] == value for key, value in decision.items())
        choice_hash = index(case_root / "choice.json")
        choice_times.append(choice["selected_utc"])
        assert choice["selected_utc"] <= suite["all_choices_persisted_utc"]
        assert choice["validation_scores"] == {
            arm: record.get("validation") for arm, record in case["arms"].items()
        }
        assert set(choice["fit_result_sha256"]) == {
            arm for arm, record in case["arms"].items() if record["status"] == "converged"
        }
        records = {"proposals": proposal, **case["arms"]}
        case_search, case_evaluation = [], []
        summaries = {}
        for arm, record in records.items():
            is_proposal = arm == "proposals"
            stage = "propose" if is_proposal else "fit"
            folder = case_root / arm
            raw = read_json(folder / "result.json")
            assert raw == {k: v for k, v in record.items() if k not in ("process", "evaluation")}
            failed_fit = not is_proposal and raw["status"] in ("error", "timeout")
            identity = {
                "source": protocol["source"],
                "protocol_sha256": protocol_hash,
                "data_identity_sha256": proposal["data_identity_sha256"],
                "data": proposal["data"],
            }
            if not is_proposal:
                identity["proposal_result_sha256"] = proposal_hash
            missing_identity = identity.keys() - raw.keys()
            if missing_identity:
                assert failed_fit, "Only failed fits may lack worker identity evidence"
                incomplete_identity_workers.append(f"{name}/{arm}")
            for key, value in identity.items():
                if key in raw:
                    assert raw[key] == value, f"Worker identity differs at {name}/{arm}: {key}"
            assert raw.get("finished_utc", raw.get("parent_finished_utc")) <= choice["selected_utc"]
            threadpools = raw.get("runtime", {}).get("threadpools")
            if threadpools is None:
                assert failed_fit, "Only failed fits may lack threadpool evidence"
                unverified_threadpool_workers.append(f"{name}/{arm}")
            else:
                assert all(pool["num_threads"] == 1 for pool in threadpools)
            process_path = folder / f"{stage}_process.json"
            assert read_json(process_path) == record["process"]
            index(process_path)
            index(folder / f"{stage}.log")
            digest = index(folder / "result.json")
            if "incomplete_receipt" in raw:
                assert failed_fit
                incomplete = raw["incomplete_receipt"]
                filename = incomplete["path"]
                assert Path(filename).name == filename
                assert filename == f"{stage}_incomplete_{incomplete['sha256']}.bin"
                incomplete_path = folder / filename
                assert index(incomplete_path) == incomplete["sha256"]
                assert incomplete_path.stat().st_size == incomplete["bytes"]
            case_search.append(record["process"]["process_seconds"])
            if is_proposal:
                continue
            fit_statuses[raw["status"]] += 1
            summaries[arm] = fit_summary(record)
            if missing_identity:
                summaries[arm]["identity_evidence"] = "incomplete"
            if raw["status"] == "converged":
                assert choice["fit_result_sha256"][arm] == digest
                assert raw["convergence"]["combined_converged"]
                assert raw["P"] == raw["nominal"]["P"]
                assert raw["q"] == raw["nominal"]["q"]
                assert index(folder / "model.pkl") == raw["model_pickle_sha256"]
            else:
                assert "evaluation" not in record
                model_path = folder / "model.pkl"
                if failed_fit:
                    if "validation" in raw:
                        summaries[arm]["selection_eligible"] = False
                    if model_path.exists():
                        model_hash = index(model_path)
                        if "model_pickle_sha256" in raw:
                            assert model_hash == raw["model_pickle_sha256"]
                        summaries[arm]["unevaluated_model_artifact"] = {
                            "sha256": model_hash,
                            "bytes": model_path.stat().st_size,
                            "trusted_for_evaluation": False,
                        }
                else:
                    assert "validation" not in raw and not model_path.exists()
            if "evaluation" not in record:
                assert arm not in choice["evaluation_arms"]
                continue
            assert arm in choice["evaluation_arms"]
            evaluation = record["evaluation"]
            raw_evaluation = read_json(folder / "evaluation.json")
            assert raw_evaluation == {k: v for k, v in evaluation.items() if k != "process"}
            assert evaluation["source"] == protocol["source"]
            assert evaluation["protocol_sha256"] == protocol_hash
            assert evaluation["data"] == proposal["data"]
            assert evaluation["choice_sha256"] == choice_hash
            assert evaluation["data_identity_sha256"] == proposal["data_identity_sha256"]
            assert evaluation["started_utc"] >= suite["all_choices_persisted_utc"]
            assert all(pool["num_threads"] == 1 for pool in evaluation["runtime"]["threadpools"])
            evaluation_starts.append(evaluation["started_utc"])
            evaluation_statuses[evaluation["status"]] += 1
            assert evaluation["status"] == "evaluated"
            assert read_json(folder / "evaluate_process.json") == evaluation["process"]
            for filename in ("evaluation.json", "evaluate_process.json", "evaluate.log"):
                index(folder / filename)
            prediction_path = folder / "test_predictions.npz"
            assert index(prediction_path) == evaluation["test_predictions_sha256"]
            with np.load(prediction_path, allow_pickle=False) as arrays:
                rows = prepared["rows"]["test"]
                np.testing.assert_array_equal(arrays["row_index"], rows)
                expected_y = data.response_values(prepared["frame"].iloc[rows], prepared["entry"])
                np.testing.assert_array_equal(arrays["response"], expected_y)
                np.testing.assert_array_equal(
                    arrays["sample_weight"], prepared["sample_weight"][rows]
                )
                recomputed = broad.score(
                    arrays["response"],
                    arrays["prediction"],
                    prepared["metadata"]["family"],
                    arrays["sample_weight"],
                )
                assert recomputed == evaluation["test"]
            case_evaluation.append(evaluation["process"]["process_seconds"])
            summaries[arm]["evaluation"] = {
                key: evaluation[key]
                for key in (
                    "status",
                    "test",
                    "test_prediction_seconds",
                    "test_predictions_sha256",
                    "started_utc",
                    "finished_utc",
                    "worker_end_peak_process_rss_mib",
                    "warnings",
                )
            }
            summaries[arm]["evaluation"]["process_seconds"] = case_evaluation[-1]
        check_sum(case_search, case["search_worker_process_seconds"])
        check_sum(case_evaluation, case["evaluation_worker_process_seconds"])
        if "all_worker_process_seconds" in case:
            check_sum(case_search + case_evaluation, case["all_worker_process_seconds"])
        else:
            missing_total_fields.append(name)
        search_costs.extend(case_search)
        evaluation_costs.extend(case_evaluation)
        all_process_costs.extend(case_search + case_evaluation)
        comparison = compare_selected_models(choice, summaries)
        additive = summaries[choice["additive_arm"]]
        inner_proposal = proposal["proposal"]
        chosen_pair_keys = {frozenset(pair) for pair in proposal["admission"]["pairs"]}
        cases[name] = {
            "status": case["status"],
            **comparison,
            "data": proposal["data"],
            "rows": proposal["rows"],
            "raw_predictor_count": proposal["raw_predictor_count"],
            "choice": choice,
            "search_worker_process_seconds": math.fsum(case_search),
            "evaluation_worker_process_seconds": math.fsum(case_evaluation),
            "all_worker_process_seconds": math.fsum(case_search + case_evaluation),
            "sum_model_fit_seconds": math.fsum(
                arm.get("fit_seconds", 0) for arm in summaries.values()
            ),
            "search_model_fit_equivalents": math.fsum(
                arm.get("fit_seconds", 0) for arm in summaries.values()
            )
            / additive["fit_seconds"],
            "search_process_equivalents": math.fsum(case_search) / additive["process_seconds"],
            "all_process_equivalents": math.fsum(case_search + case_evaluation)
            / additive["process_seconds"],
            "proposal": {
                key: inner_proposal[key]
                for key in (
                    "policy",
                    "scope",
                    "model_parameters",
                    "timing",
                    "tree_diagnostics",
                    "retained_model_storage",
                    "fit_end_peak_process_rss_mib",
                )
            },
            "admitted_pairs": proposal["admission"]["pairs"],
            "admitted_pair_scores": [
                {"rank": rank, **item}
                for rank, item in enumerate(inner_proposal["scores"], 1)
                if frozenset(item["pair"]) in chosen_pair_keys
            ],
            "proposal_pair_count": len(inner_proposal["pairs"]),
            "skip_reasons": dict(
                collections.Counter(item["reason"] for item in proposal["admission"]["skipped"])
            ),
            "fits": summaries,
        }
        cases[name]["proposal"]["process_seconds"] = proposal["process"]["process_seconds"]
    check_sum(search_costs, suite["search_worker_process_seconds"])
    check_sum(all_process_costs, suite["all_worker_process_seconds"])
    assert max(choice_times) <= min(evaluation_starts)
    return {
        "schema_version": 1,
        "raw_root": os.path.relpath(root, REPO),
        "protocol": protocol,
        "runtime": next(iter(suite["datasets"].values()))["proposal"]["runtime"],
        "audit": {
            "source_identity_matches": True,
            "all_choices_precede_all_test_evaluations": True,
            "all_data_and_split_identities_match_repreparation": not incomplete_identity_workers,
            "all_declared_arms_accounted_for": declared_arm_count,
            "test_scores_exactly_replayed_from_saved_arrays": sum(evaluation_statuses.values()),
            "all_worker_threadpools_one_thread": not unverified_threadpool_workers,
            **(
                {"failed_workers_with_incomplete_identity": incomplete_identity_workers}
                if incomplete_identity_workers
                else {}
            ),
            **(
                {"workers_with_unverified_threadpools": unverified_threadpool_workers}
                if unverified_threadpool_workers
                else {}
            ),
            "total_derivation": "Sum every proposal, fit and evaluation process receipt; compare existing raw counters with a positive-sum rounding bound.",
            "cases_without_optional_raw_total_field": missing_total_fields,
            "scope": "Audit the completed frozen batch. This script expects successful proposals and all selected evaluations, while preserving failed fits.",
        },
        "totals": {
            "source_rows": sum(c["data"]["eligibility"]["source_rows"] for c in cases.values()),
            "retained_rows": sum(c["data"]["eligibility"]["retained_rows"] for c in cases.values()),
            "real_source_outcomes": dict(
                collections.Counter(
                    c["outcome"] for c in cases.values() if c["data"]["counts_as_real_source"]
                )
            ),
            "control_outcomes": dict(
                collections.Counter(
                    c["outcome"] for c in cases.values() if not c["data"]["counts_as_real_source"]
                )
            ),
            "fit_statuses": dict(fit_statuses),
            "evaluation_statuses": dict(evaluation_statuses),
            "search_worker_process_seconds": math.fsum(search_costs),
            "evaluation_worker_process_seconds": math.fsum(evaluation_costs),
            "all_worker_process_seconds": math.fsum(all_process_costs),
            "suite_end_to_end_seconds": suite["suite_end_to_end_seconds"],
            "all_choices_persisted_utc": suite["all_choices_persisted_utc"],
            "earliest_test_worker_utc": min(evaluation_starts),
            "finished_utc": suite["finished_utc"],
        },
        "datasets": cases,
        "raw_artifact_sha256": receipt_index,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, parents=[SOURCE_PARSER])
    parser.add_argument(
        "--run-root",
        type=Path,
        default=SOURCE_REPO / ".benchmark-artifacts/broad-interactions/frozen-20260914",
    )
    parser.add_argument("--data-root", type=Path, default=data.DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_root.resolve(), data_root=args.data_root.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "totals": result["totals"], "audit": result["audit"]},
            indent=2,
        )
    )
