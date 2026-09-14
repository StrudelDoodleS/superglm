"""Validate and collect the additive-relative interaction research runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_times(value):
    if isinstance(value, dict):
        return {
            key: without_times(item)
            for key, item in value.items()
            if not key.endswith("_s") and "seconds" not in key
        }
    if isinstance(value, list):
        return [without_times(item) for item in value]
    return value


def collect(base):
    records = []
    groups = {}
    for path in sorted(base.glob("*/run.json")):
        directory = path.parent
        record = {
            "case": directory.name,
            "run": json.loads(path.read_text()),
            "artifacts": {
                item.name: digest(item) for item in sorted(directory.iterdir()) if item.is_file()
            },
        }
        result_path = directory / "result.json"
        if result_path.exists():
            record["result"] = json.loads(result_path.read_text())
            groups.setdefault(directory.name.rsplit("-r", 1)[0], []).append(record)
        else:
            record["failure_log"] = (directory / "worker.log").read_text()
        records.append(record)
    if not records:
        raise ValueError(f"No worker receipts in {base}")
    summaries = {}
    for key, pair in sorted(groups.items()):
        if len(pair) != 2:
            raise ValueError(f"Expected two repetitions for {key}, found {len(pair)}")
        left, right = [item["result"] for item in pair]
        if without_times(left["telemetry"]) != without_times(right["telemetry"]):
            raise ValueError(f"Numerical telemetry changed between repetitions: {key}")
        if left["retained_model_storage"] != right["retained_model_storage"]:
            raise ValueError(f"Retained storage changed between repetitions: {key}")
        with np.load(base / pair[0]["case"] / "predictions.npz") as a:
            with np.load(base / pair[1]["case"] / "predictions.npz") as b:
                if a.files != b.files or not all(
                    np.array_equal(a[name], b[name]) for name in a.files
                ):
                    raise ValueError(f"Prediction/coefficient replay failed: {key}")
        summaries[key] = {
            "mode": left["mode"],
            "status": left["status"],
            "M": left["interactions"],
            "parent_k": left["k"],
            "interaction_k": left.get("interaction_k", left["k"]),
            "P": left["coefficient_count_without_intercept"],
            "q": left["fitted_smoothing_parameter_count"],
            "fit_seconds": sorted([left["fit_seconds"], right["fit_seconds"]]),
            "fit_seconds_median": median([left["fit_seconds"], right["fit_seconds"]]),
            "peak_rss_mib_median": median(
                [left["fit_end_peak_process_rss_mib"], right["fit_end_peak_process_rss_mib"]]
            ),
            "retained_payload_bytes": left["retained_model_storage"]["total_payload_bytes"],
            "test_mse": left["mse"]["test"],
            "outer_iterations": left["telemetry"]["reml"].get("n_reml_iter"),
            "backend": left["resolved_direct_backend"],
            "exact_repeated_numerical_outputs": True,
            "exact_repeated_retained_payload": True,
        }
    for item in summaries.values():
        baseline = summaries[f"m0-{item['mode']}"]
        if baseline["status"] != "converged":
            raise ValueError("Additive baseline did not converge")
        item["time_ratio_to_additive"] = item["fit_seconds_median"] / baseline["fit_seconds_median"]
        item["rss_increment_mib"] = item["peak_rss_mib_median"] - baseline["peak_rss_mib_median"]
    results = [record["result"] for record in records if "result" in record]
    for field in ("package_source_sha256", "data_sha256", "test_data_sha256"):
        if len({item[field] for item in results}) != 1:
            raise ValueError(f"Unmatched experiment field: {field}")
    return {
        "date": "2026-09-13",
        "scope": "Unchanged production baseline plus existing lower-knot interaction controls",
        "ratio_definition": "Ratio of two-run median complete fit times, within smoothing mode",
        "ratio_limit": "Illustrative budgets; no universal guarantee or empirical scaling exponent",
        "collector_sha256": digest(Path(__file__)),
        "current_runner_sha256": digest(Path(__file__).with_name("benchmark_many_interactions.py")),
        "validation": {
            "worker_receipts": len(records),
            "completed_fits": len(results),
            "failed_setups": len(records) - len(results),
            "paired_cases": len(summaries),
            "source_and_input_hashes_match": True,
        },
        "summaries": summaries,
        "runs": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / ".benchmark-artifacts/cheap-interactions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notes/research/2026-09-13-cheap-interaction-measurements.json",
    )
    args = parser.parse_args()
    receipt = collect(args.input)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt["summaries"], indent=2))
    print(json.dumps(receipt["validation"]))


if __name__ == "__main__":
    main()
