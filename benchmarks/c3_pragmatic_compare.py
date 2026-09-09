"""Compare saved complete fits without rerunning them.

Run from this worktree with ``uv run python benchmarks/c3_pragmatic_compare.py``.
The sibling c3-c1-completion worktree supplies the three original-policy runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    record = json.loads(path.read_text())
    assert record["status"] == "ok"
    assert record["source_stable"]
    for key in ["sha", "source_tree_sha256"]:
        assert record["source"][key] == record["source_after"][key]
    assert digest(path.with_suffix(".npz")) == record["npz_sha256"]
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert float(arrays["objective"]) == record["smoothing"]["objective"]
    assert record["result"]["parameter_names"] == ["mean", "theta"]
    assert arrays["holdout_parameters"].shape == (record["fixture"]["holdout_rows"], 2)
    assert (arrays["holdout_parameters"] > 0).all()
    assert float(arrays["log_likelihood"]) == record["result"]["log_likelihood"]
    assert float(arrays["edf"]) == record["result"]["total_effective_df"]
    np.testing.assert_array_equal(
        arrays["smoothing_parameters"], list(record["result"]["smoothing_parameters"].values())
    )
    return record, arrays


def quantiles(values):
    levels = [0, 0.001, 0.01, 0.5, 0.9, 0.95, 0.99, 0.999, 1]
    return dict(zip(map(str, levels), np.quantile(values, levels).tolist(), strict=True))


def differences(candidate, baseline):
    delta = candidate - baseline
    assert np.isfinite(delta).all()
    result = {
        "mean_signed": float(np.mean(delta)),
        "mean_absolute": float(np.mean(np.abs(delta))),
        "absolute_quantiles": quantiles(np.abs(delta)),
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(baseline)),
    }
    if (baseline > 0).all():
        result["absolute_relative_quantiles"] = quantiles(np.abs(delta) / baseline)
    return result


def observables(arrays):
    mu, theta = arrays["holdout_parameters"].T
    result = {
        "mu": mu,
        "theta": theta,
        "variance": mu + mu**2 / theta,
        "probability_any_claim": -np.expm1(-theta * np.log1p(mu / theta)),
    }
    assert all(np.isfinite(value).all() for value in result.values())
    assert (result["variance"] >= mu).all()
    assert ((result["probability_any_claim"] > 0) & (result["probability_any_claim"] < 1)).all()
    return result


def run_summary(path, record, arrays):
    smoothing = record["smoothing"]
    history = smoothing["history"]
    fits = record["coefficient_fits"]
    terminal_index = next(h["accepted_fit_index"] for h in reversed(history) if h["accepted"])
    terminal_fit = fits[terminal_index]
    assert terminal_fit["log_likelihood"] == record["result"]["log_likelihood"]
    covariance = arrays["covariance"]
    curvature = arrays["terminal_penalized_curvature"]
    width = len(arrays["coefficients"])
    inverse_residual = covariance @ curvature - np.eye(width)
    inverse_backward_error = float(
        np.linalg.norm(inverse_residual, np.inf)
        / (np.linalg.norm(covariance, np.inf) * np.linalg.norm(curvature, np.inf) + 1)
    )
    assert inverse_backward_error <= 128 * width * np.finfo(float).eps
    assert len(history) == record["result"]["n_smoothing_iter"]
    assert smoothing["newton_iterations"] == sum(h["stage"] == "newton" for h in history)
    assert smoothing["bfgs_fallback_iterations"] == sum(h["step_source"] == "bfgs" for h in history)
    config_keys = [
        "fixture",
        "n",
        "knots",
        "data",
        "holdout_every",
        "discrete",
        "n_bins",
        "outer",
        "initial_lambda",
        "max_lambda",
        "practical_reml",
        "max_reml_iter",
        "max_inner_iter",
        "reml_tol",
        "inner_tol",
        "threads",
        "instrument",
        "measure_time",
    ]
    return {
        "raw_json": str(path),
        "raw_json_sha256": digest(path),
        "raw_npz_sha256": record["npz_sha256"],
        "source": {
            key: record["source"][key] for key in ["sha", "diff_sha256", "source_tree_sha256"]
        },
        "source_stable": record["source_stable"],
        "source_after": record["source_after"],
        "source_stability_note": "HEAD and source-file hash stayed fixed. A changed repository diff hash can reflect concurrent non-source edits; both hashes are retained.",
        "harness_sha256": record["harness_sha256"],
        "fixtures_sha256": record["fixtures_sha256"],
        "config": {key: record["config"][key] for key in config_keys},
        "converged": record["result"]["converged"],
        "coefficient_converged": record["result"]["coefficient_converged"],
        "smoothing_converged": record["result"]["smoothing_converged"],
        "smoothing_certified": record["result"]["smoothing_certified"],
        "smoothing": {key: value for key, value in smoothing.items() if key != "history"},
        "lambdas": record["result"]["smoothing_parameters"],
        "edf": record["result"]["total_effective_df"],
        "log_likelihood": record["result"]["log_likelihood"],
        "penalized_log_likelihood": record["result"]["penalized_log_likelihood"],
        "terminal_coefficient_fit": terminal_fit,
        "covariance_provenance": {
            "kind": "conditional covariance: inverse terminal observed penalized curvature",
            "api": "DistributionalGLM.covariance_ -> compute_joint_inference -> terminal_pseudo_inverse",
            "inverse_max_absolute_residual": float(np.max(np.abs(inverse_residual))),
            "inverse_normalized_backward_error": inverse_backward_error,
            "terminal_penalized_score_l2": float(np.linalg.norm(arrays["terminal_score"])),
            "local_newton_decrement_squared": float(
                arrays["terminal_score"] @ covariance @ arrays["terminal_score"]
            ),
        },
        "exact_face_components": record["result"]["exact_face_components"],
        "work": {
            "outer_iterations": len(history),
            "accepted_outer_iterations": sum(h["accepted"] for h in history),
            "stages": dict(Counter(h["stage"] for h in history)),
            "coefficient_fits": len(fits),
            "coefficient_fit_iterations_total": sum(f["iterations"] for f in fits),
            "coefficient_fit_backtracking_steps_total": sum(f["backtracking_steps"] for f in fits),
            "inner_iterations_reported": record["result"]["n_inner_iter"],
        },
        "dispatch": {
            "coefficient_fit_backends": dict(
                Counter(f["execution_backend_identifier"] for f in fits)
            ),
            "resolved_chunk_sizes": sorted({f["resolved_chunk_size"] for f in fits}),
            "terminal_curvature": record["result"]["curvature_telemetry"],
            "instrumentation_enabled": record["instrumentation"]["enabled"],
            "limitation": "Recorded coefficient-fit backend dispatch; uninstrumented derivative dispatch is not inferred.",
        },
        "memory": {
            "peak_fit_process_rss_bytes": record["peak_fit_process_rss_bytes"],
            "peak_process_rss_bytes": record["peak_process_rss_bytes"],
            "interpretation": "Observed process RSS, not an isolated memory-performance comparison.",
        },
        "timing_status": "UNMEASURED",
        "raw_timing_status": record["timing_status"],
        "holdout_observable_quantiles": {
            key: quantiles(value) for key, value in observables(arrays).items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--baseline-root", type=Path, default=root.parent / "c3-c1-completion")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = args.baseline_root / ".benchmark-artifacts"
    paths = {
        "recovery_newton_practical": root
        / ".benchmark-artifacts/c3-practical/nb2-recovery/recovery-0.json",
        "old_newton_practical": baseline / "c3-practical/nb2-practical-original/before-0.json",
        "old_efs_practical": baseline / "c3-practical/nb2-efs-practical-original/before-0.json",
        "old_newton_strict": baseline / "c3-c1/nb2-k12-candidate-discrete256/candidate-0.json",
    }
    loaded = {name: load(path) for name, path in paths.items()}
    candidate_record, candidate_arrays = loaded["recovery_newton_practical"]
    assert candidate_record["source"]["sha"] == "8b66828d032878057815fb1acbb28028872c4cc8"
    assert candidate_record["smoothing"]["terminal_gradient"] is None
    for record, arrays in loaded.values():
        assert record["fixture"] == candidate_record["fixture"]
        assert (
            record["result"]["coefficient_names"] == candidate_record["result"]["coefficient_names"]
        )
        assert set(candidate_arrays) <= set(arrays)
        assert all(arrays[key].shape == candidate_arrays[key].shape for key in candidate_arrays)
    practical_arrays = loaded["old_newton_practical"][1]
    strict_arrays = loaded["old_newton_strict"][1]
    assert all(
        np.array_equal(practical_arrays[key], strict_arrays[key]) for key in practical_arrays
    )
    comparisons = {}
    candidate_observables = observables(candidate_arrays)
    for name, (record, arrays) in loaded.items():
        if name in {"recovery_newton_practical", "old_newton_strict"}:
            continue
        baseline_observables = observables(arrays)
        covariance = candidate_arrays["covariance"] - arrays["covariance"]
        comparisons[name] = {
            "direction": "recovery minus baseline; pointwise relative errors divide by baseline",
            "objective_delta": float(candidate_arrays["objective"] - arrays["objective"]),
            "lambda_ratios": {
                key: value / record["result"]["smoothing_parameters"][key]
                for key, value in candidate_record["result"]["smoothing_parameters"].items()
            },
            "edf_delta": float(candidate_arrays["edf"] - arrays["edf"]),
            "log_likelihood_delta": float(
                candidate_arrays["log_likelihood"] - arrays["log_likelihood"]
            ),
            "holdout": {
                key: differences(value, baseline_observables[key])
                for key, value in candidate_observables.items()
            },
            "covariance": {
                "max_absolute": float(np.max(np.abs(covariance))),
                "relative_frobenius": float(
                    np.linalg.norm(covariance) / np.linalg.norm(arrays["covariance"])
                ),
                "diagonal": differences(
                    np.diag(candidate_arrays["covariance"]), np.diag(arrays["covariance"])
                ),
            },
            "coefficients": differences(candidate_arrays["coefficients"], arrays["coefficients"]),
        }
    receipt = {
        "schema": 1,
        "scope": "Frozen gradient-recovery stage only; excludes subsequent practical-boundary policy changes.",
        "interpretation": "These are different stopping solutions on identical data and representation, not a representation-parity experiment. Practical convergence is not certified smoothing stationarity.",
        "important_findings": [
            "Recovery versus old rejected Newton barely changes predictions but changes covariance by 12.40% in relative Frobenius norm. Covariance is consistent with each run's terminal observed penalized curvature; theta smoothing penalties change by +7.96%, +9.45%, and -22.54%.",
            "Recovery terminal coefficients stop by objective_and_step with 28 backtracks and relative score 8.839e-6 (inner tolerance 1e-7). Near-fixed coefficients with changed penalties explain differing inference geometry; this receipt does not establish a tightly stationary conditional mode.",
            "Versus plain practical EFS, recovery theta maximum is 283710 versus 239.904, despite similar medians and p99. Maximum relative variance change is 11.73%; maximum absolute any-claim probability change is 0.01904. Covariance relative Frobenius change is 36.98%.",
            "The old strict and old practical Newton runs have bitwise-identical saved arrays; their different iteration budgets and practical flags did not change the rejected stopping solution.",
        ],
        "timing": "UNMEASURED: concurrent fits and Kompress activity; no speed or isolated memory-performance claim.",
        "formulae": {
            "variance": "mu + mu**2 / theta",
            "probability_any_claim": "-expm1(-theta * log1p(mu / theta))",
        },
        "fixture": candidate_record["fixture"],
        "dependencies": candidate_record["dependencies"],
        "python": candidate_record["python"],
        "comparison_script_sha256": digest(Path(__file__)),
        "checks": {
            "artifact_hashes_valid": True,
            "source_stable_all_runs": True,
            "fixture_and_parameter_order_identical": True,
            "all_arrays_finite": True,
            "old_newton_strict_and_practical_all_arrays_bitwise_equal": True,
            "history_counts_match_metadata": True,
        },
        "runs": {name: run_summary(paths[name], *data) for name, data in loaded.items()},
        "comparisons": comparisons,
    }
    output = args.output or root / "benchmarks/c3_pragmatic_convergence_receipt.json"
    # Repository writes use apply_patch; stdout permits a caller to review/apply the artifact.
    if args.output:
        output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    else:
        print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
