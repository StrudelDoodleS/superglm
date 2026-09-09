"""Verify raw timing evidence without approving quietness or a performance claim."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_timing_final import (
    ACTIVITY_LIMITATION, EXPECTED_SHA, ORDER, SOURCES, activity_audit, digest, write_json,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def observed_chunk_policy(n_rows, n_parameters, n_coefficients):
    """Read the pinned candidate's observed-assembly selector, without fitting."""
    source = SOURCES["candidate"].resolve()
    code = """
import json
import sys
from superglm.distributional.solver import chunks
n, k, p = map(int, sys.argv[1:])
print(json.dumps({
    "selector": chunks.AUTO_CHUNK_SELECTOR,
    "memory_budget_bytes": chunks.AUTO_CHUNK_MEMORY_BYTES,
    "resolved_rows": chunks.resolve_chunk_size(n, k, "auto", p_coefficients=p),
    "source_file": chunks.__file__,
}))
"""
    env = dict(os.environ, PYTHONPATH=str(source / "src"))
    policy = json.loads(subprocess.check_output(
        [sys.executable, "-c", code, str(n_rows), str(n_parameters), str(n_coefficients)],
        cwd=source, env=env, text=True,
    ))
    source_file = source / "src/superglm/distributional/solver/chunks.py"
    require(Path(policy["source_file"]).resolve() == source_file, "wrong chunk selector import")
    require(policy["selector"] == "distributional-auto-v1", "unreviewed chunk selector")
    budget = policy["memory_budget_bytes"]
    require(isinstance(budget, int) and 0 < budget <= 8 * 1024 * 1024, "unexpected assembly budget")
    channels = n_parameters * (n_parameters + 1) // 2
    columns = n_coefficients + 6 * n_parameters + 4 * channels + 4
    row_bytes = np.dtype(np.float64).itemsize * columns
    expected = min(n_rows, max(1, budget // row_bytes))
    require(policy["resolved_rows"] == expected, "observed chunk selector formula changed")
    require(0 < expected <= n_rows and expected * row_bytes <= budget, "unbounded assembly chunk")
    policy.update({
        "source_sha256": digest(source_file),
        "n_observations": n_rows, "n_parameters": n_parameters,
        "n_coefficients": n_coefficients, "estimated_float64_columns": columns,
        "estimated_bytes_per_row": row_bytes,
        "formula": "min(n, max(1, budget_bytes // (8 * (p + 6*k + 4*k*(k+1)//2 + 4))))",
        "scope": "Observed coefficient assembly row bound; estimated temporary budget, not exact allocator RSS or a derivative/posterior replay bound.",
    })
    return policy


def distribution(values):
    return {
        "values": values, "minimum": min(values),
        "median": statistics.median(values), "maximum": max(values),
    }


def difference(values, reference):
    require(values.shape == reference.shape, "numerical array shapes differ")
    require(np.all(np.isfinite(values)) and np.all(np.isfinite(reference)), "nonfinite numerical output")
    delta = values - reference
    return {
        "shape": list(values.shape),
        "max_absolute_difference": float(np.max(np.abs(delta), initial=0.0)),
        "relative_l2_difference": float(np.linalg.norm(delta.ravel()) / max(
            np.linalg.norm(reference.ravel()), np.finfo(float).tiny
        )),
        "max_scaled_difference": float(np.max(
            np.abs(delta) / (1.0 + np.maximum(np.abs(reference), np.abs(values))), initial=0.0
        )),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="new directory created by run_timing_final.py")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    controller_path = folder / "controller.json"
    preflight_path = folder / "quiet-window.json"
    controller = json.loads(controller_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    records = controller["runs"]
    require(controller["audit_status"] == "pending_current_window_review", "unexpected controller audit state")
    require(controller["expected_source_sha"] == EXPECTED_SHA, "source pins differ from this final window")
    require([record["label"] for record in records] == list(ORDER), "expected six BCCBBC workers")
    require(controller["harness_stable"] and controller["fixtures_stable"], "harness changed during the window")
    require(controller["controller_sha256"] == digest(Path(__file__).with_name("run_timing_final.py")), "controller script changed since execution")
    allowed_config_differences = {"source", "out", "discrete", "worker_source", "worker_output"}
    expected_config = {
        "fixture": "severity-gaussian", "n": 0, "replicate": 20, "knots": 12,
        "outer": "efs+newton", "practical_reml": False, "initial_lambda": 0.1,
        "max_reml_iter": 100, "max_inner_iter": 100, "inner_tol": 1.0e-7,
        "reml_tol": 1.0e-6, "max_lambda": 1.0e10, "holdout_every": 10,
        "n_bins": 256, "threads": 1, "repeat": 1, "warmup": False,
        "instrument": False, "measure_time": True,
    }
    array_names = (
        "coefficients", "smoothing_parameters", "train_parameters", "holdout_parameters",
        "covariance", "objective", "edf", "log_likelihood",
    )
    runs, reports, arrays = [], [], []
    chunk_policy = observed_chunk_policy(449000, 2, 92)
    for index, record in enumerate(records):
        label = record["label"]
        require(record["index"] == index and record["returncode"] == 0, "worker order or return code invalid")
        path = Path(record["json"])
        require(path.resolve().is_relative_to(folder), "worker receipt is outside this window")
        manifest_path = path.parent / "manifest.json"
        require(digest(manifest_path) == record["manifest_sha256"], "manifest hash mismatch")
        manifest_record = json.loads(manifest_path.read_text())["runs"][0]
        for field in ("index", "label", "worker_pid", "returncode", "json", "json_sha256", "log_sha256", "npz_sha256"):
            require(manifest_record[field] == record[field], f"manifest/controller disagree: {field}")
        require(digest(path) == record["json_sha256"], "JSON hash mismatch")
        require(digest(path.with_suffix(".log")) == record["log_sha256"], "log hash mismatch")
        require(digest(path.with_suffix(".npz")) == record["npz_sha256"], "NPZ hash mismatch")
        report = json.loads(path.read_text())
        require(report["npz_sha256"] == record["npz_sha256"], "worker NPZ hash mismatch")
        require(report["status"] == "ok" and report["source_stable"], "worker unsuccessful or source changed")
        require(report["source"]["sha"] == EXPECTED_SHA[label], "wrong executed source revision")
        require(Path(report["source"]["root"]).resolve() == SOURCES[label].resolve(), "wrong source worktree")
        require(Path(report["imported_module"]).resolve().is_relative_to(SOURCES[label] / "src"), "wrong package import")
        require(report["harness_sha256"] == controller["harness_sha256"], "wrong executed harness")
        require(report["fixtures_sha256"] == controller["fixtures_sha256"], "wrong executed fixture helper")
        require(report["fixture"]["provenance"]["data_sha256"] == controller["data_sha256"], "input files changed")
        for field, expected in expected_config.items():
            require(report["config"][field] == expected, f"unexpected config {field}")
        require(Path(report["config"]["data"]).resolve() == Path(controller["data"]), "wrong data directory")
        require(report["config"]["discrete"] == (label == "candidate"), "wrong representation arm")
        require(all(value == "1" for value in report["thread_environment"].values()), "thread environment not one")
        require(bool(report["threadpools"]) and all(pool["num_threads"] == 1 for pool in report["threadpools"]), "runtime threadpool not one")
        require(report["fixture"]["rows"] == 449000 and report["fixture"]["holdout_rows"] == 2494, "wrong severity replication fixture")
        result, smoothing = report["result"], report["smoothing"]
        require(result["q"] == 92 and result["rank"] == 92, "unexpected coefficient width or rank")
        require(result["smoothing_certified"] and result["coefficient_converged"], "uncertified terminal fit")
        backend = sorted({fit["execution_backend_identifier"] for fit in report["coefficient_fits"]})
        expected_backend = "distributional-chunked-v1" if label == "candidate" else "distributional-dense-v1"
        require(backend == [expected_backend], "actual coefficient backend does not match the arm")
        if label == "candidate":
            require(len(result["parameter_names"]) == chunk_policy["n_parameters"], "chunk policy parameter count mismatch")
            require(all(fit["resolved_chunk_size"] == chunk_policy["resolved_rows"] for fit in report["coefficient_fits"]), "candidate chunk size differs from observed-assembly auto policy")
        require(report["fit_seconds"] > 0, "missing worker fit clock")
        require(report["peak_process_rss_bytes"] >= report["peak_fit_process_rss_bytes"] > 0, "invalid RSS high-water marks")
        if reports:
            first = reports[0]
            for field in ("dependencies", "python", "executable", "harness_sha256", "fixtures_sha256", "threadpools"):
                require(report[field] == first[field], f"environment mismatch: {field}")
            require(report["fixture"]["fingerprints"] == first["fixture"]["fingerprints"], "actual fitting inputs differ")
            for field in ("coefficient_names", "parameter_names"):
                require(result[field] == first["result"][field], f"model names differ: {field}")
            require(list(result["smoothing_parameters"]) == list(first["result"]["smoothing_parameters"]), "penalty order differs")
            require(
                {key: value for key, value in report["config"].items() if key not in allowed_config_differences}
                == {key: value for key, value in first["config"].items() if key not in allowed_config_differences},
                "unapproved config difference between workers",
            )
        activity = activity_audit(report["environment_fit_start"], report["environment_fit_end"], record["worker_pid"])
        require(activity["worker"] is not None, "exact worker PID missing from fit activity endpoints")
        require(not activity["negative_tick_delta_pids"], "CPU tick identity ambiguity in activity endpoints")
        runs.append({
            "run": f"{index}-{label}", "arm": label, "worker_pid": record["worker_pid"],
            "json": str(path), "json_sha256": record["json_sha256"],
            "npz_sha256": record["npz_sha256"], "log_sha256": record["log_sha256"],
            "source_sha": report["source"]["sha"],
            "source_tree_sha256": report["source"]["source_tree_sha256"],
            "diff_sha256": report["source"]["diff_sha256"],
            "raw_worker_timing_status": report["timing_status"],
            "observed_fit_seconds": report["fit_seconds"],
            "peak_fit_process_rss_bytes": report["peak_fit_process_rss_bytes"],
            "peak_process_rss_bytes": report["peak_process_rss_bytes"],
            "q": result["q"], "rank": result["rank"], "certified": result["smoothing_certified"],
            "reason": result["smoothing_reason"], "n_smoothing_iter": result["n_smoothing_iter"],
            "n_inner_iter": result["n_inner_iter"], "coefficient_fit_count": len(report["coefficient_fits"]),
            "objective": smoothing["objective"], "log_likelihood": result["log_likelihood"],
            "edf": result["total_effective_df"],
            "gradient_norm": smoothing["terminal_projected_gradient_norm"],
            "stationarity_bar": smoothing["stationarity_bar"], "backend_identifiers": backend,
            "resolved_chunk_sizes": sorted({fit["resolved_chunk_size"] for fit in report["coefficient_fits"]}, key=lambda value: -1 if value is None else value),
            "phase_work_counts": report["phase_snapshot"]["counts"], "activity": activity,
        })
        reports.append(report)
        with np.load(path.with_suffix(".npz")) as values:
            arrays.append({name: values[name] for name in array_names})
    arms = {}
    for label in SOURCES:
        selected = [run for run in runs if run["arm"] == label]
        require(len(selected) == 3, "each arm needs three repetitions")
        require(len({run["source_tree_sha256"] for run in selected}) == 1, "production tree changed within arm")
        require(len({run["diff_sha256"] for run in selected}) == 1, "working-tree diff changed within arm")
        arms[label] = {
            "worker_fit_seconds": distribution([run["observed_fit_seconds"] for run in selected]),
            "peak_fit_process_rss_bytes": distribution([run["peak_fit_process_rss_bytes"] for run in selected]),
            "peak_process_rss_bytes": distribution([run["peak_process_rss_bytes"] for run in selected]),
            "reasons": [run["reason"] for run in selected],
            "work_counts_identical_within_arm": all(run["phase_work_counts"] == selected[0]["phase_work_counts"] for run in selected),
        }
    comparisons = []
    reference = arrays[0]
    for run, values in zip(runs, arrays, strict=True):
        item = {"run": run["run"], "reference": runs[0]["run"]}
        item["arrays"] = {name: difference(values[name], reference[name]) for name in array_names}
        require(np.all(np.diag(values["covariance"]) >= 0), "negative coefficient variance")
        item["coefficient_standard_errors"] = difference(
            np.sqrt(np.diag(values["covariance"])), np.sqrt(np.diag(reference["covariance"]))
        )
        comparisons.append(item)
    summary = {
        "schema": 1, "audit_status": "pending_current_window_review",
        "timing_benefit": "unmeasured_pending_review", "validated_speedup_ratio": None,
        "performance_claim": None, "historical_approval_inherited": False,
        "scope": "Local single-thread fresh-process comparison on 20-fold replicated severity Gaussian data: 449000 training rows from 22450 policies, unchanged 2494-policy holdout, 92 coefficients.",
        "clock_policy": "Only raw worker fit_seconds enter time distributions; no tool or controller elapsed times are fit evidence.",
        "rss_policy": "Both values are process lifetime RSS high-water marks. The fit-return mark includes imports/data preparation; the final mark also includes reporting/predictions. Neither measures isolated fit allocation.",
        "cache_policy": controller["cache_policy"], "activity_limitations": ACTIVITY_LIMITATION,
        "candidate_observed_assembly_chunk_policy": chunk_policy,
        "controller_sha256": digest(controller_path), "preflight_sha256": digest(preflight_path),
        "summary_script_sha256": digest(__file__), "preflight": preflight,
        "input_config": reports[0]["config"], "dependencies": reports[0]["dependencies"],
        "fixture": reports[0]["fixture"], "arms": arms, "runs": runs,
        "numerical_comparisons": comparisons,
        "numerical_review_status": "Finite outputs, dimensions, identities, certification and source/input integrity checked; difference magnitudes are observations awaiting current-window review, not a new numerical equivalence certificate.",
    }
    output = (args.out or folder / "summary.json").resolve()
    write_json(output, summary)
    print(json.dumps({"summary": str(output), "audit_status": summary["audit_status"], "arms": arms}), flush=True)


if __name__ == "__main__":
    main()
