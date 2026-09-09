"""Compare the completed finite-NB2 repair fit with historical stopping solutions.

Uses saved files only. Run with --output /tmp/c3-nb2-final-receipt.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from c3_pragmatic_compare import differences, digest, load, observables, run_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=root.parent / "c3-c1-completion")
    args = parser.parse_args()
    baseline = args.baseline_root / ".benchmark-artifacts/c3-practical"
    paths = {
        "final": root / ".benchmark-artifacts/c3-practical/nb2-final/final-0.json",
        "old_newton_practical": baseline / "nb2-practical-original/before-0.json",
        "old_efs_practical": baseline / "nb2-efs-practical-original/before-0.json",
        "recovery_only": root / ".benchmark-artifacts/c3-practical/nb2-recovery/recovery-0.json",
    }
    loaded = {name: load(path) for name, path in paths.items()}
    final, arrays = loaded["final"]
    assert final["source"]["sha"] == "5f994c8f6ac0501606594e2f36bfc0cd24050ec1"
    assert final["source"]["diff_sha256"] == final["source_after"]["diff_sha256"]
    smoothing = final["smoothing"]
    assert smoothing["convergence_reason"] == "stationary"
    assert final["result"]["converged"] and final["result"]["smoothing_certified"]
    assert not smoothing["unresolved_upper_bound"] and not smoothing["beyond_cap_components"]
    assert not final["result"]["exact_face_components"]
    names = list(final["result"]["smoothing_parameters"])
    assert names == list(smoothing["terminal_gradient"])
    assert names == list(smoothing["terminal_gradient_certificate"])
    gradient = np.asarray(list(smoothing["terminal_gradient"].values()))
    certificate = np.asarray(list(smoothing["terminal_gradient_certificate"].values()))
    assert np.isfinite(gradient).all() and np.isfinite(certificate).all()
    assert (certificate >= 0).all()
    assert np.max(np.abs(gradient)) == smoothing["terminal_projected_gradient_norm"]
    assert np.max(np.abs(gradient) + certificate) < smoothing["stationarity_bar"]
    accepted = [item for item in smoothing["history"] if item["accepted"]]
    assert accepted[-1]["lambdas_after"] == final["result"]["smoothing_parameters"]
    assert accepted[-1]["objective_after"] == smoothing["objective"]
    last_source_gradient = np.asarray(list(accepted[-1]["gradient"].values()))
    assert np.max(np.abs(last_source_gradient)) != np.max(np.abs(gradient))
    terminal_fit = final["coefficient_fits"][accepted[-1]["accepted_fit_index"]]
    assert terminal_fit["score_relative"] <= final["config"]["inner_tol"]
    assert terminal_fit["backtracking_steps"] == 0
    for record, values in loaded.values():
        assert final["fixture"] == record["fixture"]
        assert final["result"]["coefficient_names"] == record["result"]["coefficient_names"]
        assert final["result"]["parameter_names"] == record["result"]["parameter_names"]
        for key in ["coefficients", "covariance", "train_parameters", "holdout_parameters"]:
            assert arrays[key].shape == values[key].shape
    covariance = arrays["covariance"]
    curvature = arrays["terminal_penalized_curvature"]
    width = len(arrays["coefficients"])
    epsilon = np.finfo(float).eps
    symmetry_bound = 64 * width * epsilon * np.linalg.norm(covariance, 2)
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    assert symmetry_error <= symmetry_bound
    covariance_eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    curvature_eigenvalues = np.linalg.eigvalsh((curvature + curvature.T) / 2)
    assert covariance_eigenvalues[0] > symmetry_bound
    assert curvature_eigenvalues[0] > 64 * width * epsilon * np.linalg.norm(curvature, 2)
    inverse_residual = covariance @ curvature - np.eye(width)
    backward_error = float(
        np.linalg.norm(inverse_residual, np.inf)
        / (np.linalg.norm(covariance, np.inf) * np.linalg.norm(curvature, np.inf) + 1)
    )
    assert backward_error <= 128 * width * epsilon
    score = arrays["terminal_score"]
    correction = covariance @ score
    decrement_squared = float(score @ correction)
    assert decrement_squared >= 0
    final_observables = observables(arrays)
    comparisons = {}
    for name, (record, values) in loaded.items():
        if name == "final":
            continue
        previous_observables = observables(values)
        comparisons[name] = {
            "direction": "final minus historical baseline; relative errors divide by baseline",
            "objective_delta": smoothing["objective"] - record["smoothing"]["objective"],
            "edf_delta": final["result"]["total_effective_df"]
            - record["result"]["total_effective_df"],
            "log_likelihood_delta": final["result"]["log_likelihood"]
            - record["result"]["log_likelihood"],
            "lambda_ratios": {
                key: value / record["result"]["smoothing_parameters"][key]
                for key, value in final["result"]["smoothing_parameters"].items()
            },
            "holdout": {
                key: differences(value, previous_observables[key])
                for key, value in final_observables.items()
            },
            "covariance": differences(covariance, values["covariance"]),
            "conditional_coefficient_standard_errors": differences(
                np.sqrt(np.diag(covariance)), np.sqrt(np.diag(values["covariance"]))
            ),
            "coefficients": differences(arrays["coefficients"], values["coefficients"]),
        }
    receipt = {
        "schema": 1,
        "scope": "Completed original freMTPL2 NB2 workload after accepted-state recovery, practical-boundary policy, and finite-NB2 low-ratio arithmetic repair.",
        "interpretation": "The final fit meets the solver's strict stationary criterion on this workload. Historical runs use different stopping solutions; these comparisons are not representation parity, proof of a global optimum, or an inference-equivalence claim.",
        "important_findings": [
            "The final coefficient solve stops by objective_and_score with zero backtracks and relative score 1.43e-8, below the configured 1e-7. Local Newton decrement squared falls from about 0.1476 at the recovery-only endpoint to 1.68e-7.",
            "Conditional uncertainty changes materially: covariance relative Frobenius changes are 62.93% versus old rejected Newton, 47.75% versus recovery alone, and 29.66% versus plain practical EFS. All compared covariances remain those of their own stopping solutions.",
            "Final held-out theta maximum is about 3.06 million, versus 0.284 million after recovery alone. Final-versus-recovery relative L2 changes are 0.103% for mean, 1.160% for variance, and 0.266% for probability of any claim; maximum absolute claim-probability change is 0.00637.",
            "The reported EFS fixed-point residual is not the exact-profile LAML gradient used for the final Newton stationarity decision. The latter is evaluated at the accepted final state; its published Richardson indicators are numerical diagnostics, not a complete error bound.",
        ],
        "timing_status": "UNMEASURED",
        "timing_note": "Potentially concurrent numerical work and Kompress activity; no complete-fit speed or isolated memory-performance claim.",
        "fixture": final["fixture"],
        "dependencies": final["dependencies"],
        "python": final["python"],
        "formulae": {
            "variance": "mu + mu**2 / theta",
            "probability_any_claim": "-expm1(-theta * log1p(mu / theta))",
        },
        "smoothing_stationarity": {
            "maximum_absolute_gradient": float(np.max(np.abs(gradient))),
            "maximum_gradient_certificate": float(np.max(certificate)),
            "maximum_absolute_gradient_plus_certificate": float(
                np.max(np.abs(gradient) + certificate)
            ),
            "maximum_gradient_divided_by_one_plus_absolute_objective": float(
                np.max(np.abs(gradient)) / (1 + abs(smoothing["objective"]))
            ),
            "gradient_to_stationarity_bar_ratio": float(
                np.max(np.abs(gradient)) / smoothing["stationarity_bar"]
            ),
            "last_step_source_gradient_max_absolute": float(np.max(np.abs(last_source_gradient))),
            "stationarity_bar": smoothing["stationarity_bar"],
            "gradient_provenance_limitation": "Final lambda/objective/order and gradient/certificate metadata are checked directly against saved accepted-state records. This comparison does not recompute LAML derivatives; freshness additionally relies on the reviewed endgame evaluating its terminal derivative pass at the accepted final state.",
            "reviewed_source_provenance": "At frozen 5f994c8f, accepting step 9 clears the derivative workspace, updates current to accepted fit 9, and starts a fresh derivative pass using that fit and its lambdas. The terminal gradient differs from step 9's source gradient. No derivative-failure recovery occurs at this terminal.",
            "certificate_interpretation": "Published derivative indicators are Richardson diagnostics; the maximum-gradient-plus-indicator check is not asserted to enclose every source of numerical error.",
        },
        "terminal_coefficient_accuracy": {
            "convergence_reason": terminal_fit["convergence_reason"],
            "relative_score": terminal_fit["score_relative"],
            "configured_inner_tolerance": final["config"]["inner_tol"],
            "backtracking_steps": terminal_fit["backtracking_steps"],
            "penalized_score_l2": float(np.linalg.norm(score)),
            "local_newton_decrement_squared": decrement_squared,
            "local_newton_correction_posterior_metric": float(np.sqrt(decrement_squared)),
            "local_quadratic_remaining_gain": decrement_squared / 2,
            "coefficient_correction_max_absolute": float(np.max(np.abs(correction))),
            "limitation": "Local quadratic error indicators at the saved geometry; not a global error certificate.",
        },
        "conditional_covariance_validation": {
            "kind": "conditional on fitted smoothing parameters; inverse terminal observed penalized curvature",
            "minimum_covariance_eigenvalue": float(covariance_eigenvalues[0]),
            "minimum_curvature_eigenvalue": float(curvature_eigenvalues[0]),
            "curvature_condition_2": float(curvature_eigenvalues[-1] / curvature_eigenvalues[0]),
            "symmetry_max_absolute_error": symmetry_error,
            "symmetry_bound": float(symmetry_bound),
            "inverse_max_absolute_residual": float(np.max(np.abs(inverse_residual))),
            "inverse_normalized_backward_error": backward_error,
            "backward_error_bound": 128 * width * epsilon,
            "limitation": "Conditional covariance excludes smoothing-parameter uncertainty and this check does not certify inferential adequacy for a particular use.",
        },
        "runs": {name: run_summary(paths[name], *value) for name, value in loaded.items()},
        "comparisons": comparisons,
        "comparison_script_sha256": digest(Path(__file__)),
        "shared_comparison_helper_sha256": digest(root / "benchmarks/c3_pragmatic_compare.py"),
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
