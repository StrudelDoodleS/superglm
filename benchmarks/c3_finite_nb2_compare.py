"""Summarize saved NB2 mode diagnostics without evaluating or fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / ".benchmark-artifacts/c3-practical/nb2-mode-probe"
    paths = {name: directory / f"{name}.json" for name in ["probe", "after"]}
    old, new = [json.loads(paths[name].read_text()) for name in paths]
    input_path = root / old["input"]
    input_record = json.loads(input_path.read_text())
    assert old["input"] == new["input"]
    assert sha(input_path) == old["input_sha256"] == new["input_sha256"]
    assert sha(input_path.with_suffix(".npz")) == input_record["npz_sha256"]
    for record in [old, new]:
        for key, value in record["fingerprints"].items():
            assert value == input_record["fixture"]["fingerprints"][key]
        assert record["lambdas"] == input_record["result"]["smoothing_parameters"]
        assert record["fresh_saved_gradient_max_difference"] == 0
        assert record["fresh_gradient_l2"] == record["saved_gradient_l2"]
        assert record["predicted_directional_slope"] == record["saved_directional_slope"]
    equal_fields = [
        "fingerprints",
        "lambdas",
        "saved_gradient_l2",
        "fresh_gradient_l2",
        "predicted_directional_slope",
        "full_newton_relative_correction",
        "prediction_max_difference",
        "stable_objective_direction_checks",
    ]
    assert all(old[key] == new[key] for key in equal_fields)
    assert old["minimum_supported_mean_theta_ratio"] == 2.0**-26
    assert new["minimum_supported_mean_theta_ratio"] == 2.0**-52
    before = {item["step"]: item for item in old["curve"]}
    after = {item["step"]: item for item in new["curve"]}
    assert before.keys() == after.keys()
    assert before[1.0]["rejection"]["type"] == "NegativeBinomialPoissonBoundaryError"
    assert after[1.0]["rejection"] is None
    assert all(item["rejection"] is None for item in after.values())
    assert all(
        before[step]["stable_reported_penalized"] == after[step]["stable_reported_penalized"]
        for step in before
    )
    gain = after[1.0]["reported_penalized"] - after[0.0]["reported_penalized"]
    independent_gain = (
        after[1.0]["stable_reported_penalized"] - after[0.0]["stable_reported_penalized"]
    )
    gain_error = abs(gain - independent_gain)
    error_bound = (
        64
        * np.finfo(float).eps
        * max(abs(after[1.0]["reported_penalized"]), abs(after[0.0]["reported_penalized"]))
    )
    assert gain > 0 and independent_gain > 0 and gain_error <= error_bound
    assert after[1.0]["boundary_count"] == 0
    assert 2.0**-52 < after[1.0]["boundary_ratio"] < 2.0**-26
    test_log = directory / "focused-tests.log"
    assert "168 passed" in test_log.read_text()
    implementation_files = [
        "src/superglm/distributional/kernels/negative_binomial.py",
        "tests/test_negative_binomial_low_mean_ratio.py",
    ]
    commit = subprocess.check_output(["git", "rev-parse", "5f994c8f"], cwd=root, text=True).strip()
    committed_hashes = {}
    for name in implementation_files:
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
        committed_hashes[name] = hashlib.sha256(blob).hexdigest()
        assert sha(root / name) == committed_hashes[name]
    receipt = {
        "schema": 1,
        "scope": "Saved coefficient-mode diagnostic before/after finite-NB2 guard repair. This is not a completed full-book refit or a new smoothing-convergence claim.",
        "timing_status": "UNMEASURED",
        "implementation_commit": commit,
        "committed_file_sha256": committed_hashes,
        "current_reproduction_script": {
            "path": "benchmarks/c3_nb2_mode_probe.py",
            "sha256": sha(root / "benchmarks/c3_nb2_mode_probe.py"),
            "note": "This helper is not present in implementation commit 5f994c8f; its current hash does not identify the exact script bytes used for the historical runs.",
        },
        "execution_provenance_limitation": "The diagnostic JSONs do not embed their executed source commit or script hash. The hashes above identify the committed implementation/reproduction script, not an authenticated run-time snapshot. Raw input hashes, reconstructed fingerprints, geometry checks, and diagnostic outputs are verified directly.",
        "raw_diagnostics": {
            name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()
        },
        "saved_mode_input": {
            "path": str(input_path),
            "json_sha256": sha(input_path),
            "npz_sha256": input_record["npz_sha256"],
            "source_sha": input_record["source"]["sha"],
            "source_tree_sha256": input_record["source"]["source_tree_sha256"],
            "fingerprints": old["fingerprints"],
            "lambdas": old["lambdas"],
        },
        "guard_scope": {
            "old_lower_mean_theta_ratio": 2.0**-26,
            "new_lower_mean_theta_ratio": 2.0**-52,
            "unchanged_lower_theta_mean_ratio": 2.0**-26,
            "explanation": "Only the low mean/theta side is extended. The opposite tail guard and finite numerical-domain checks remain. This remains finite NB2; no exact Poisson face, infinite theta, or substitution of Poisson likelihood is implemented.",
            "weight_note": "The mode probe reconstructs the book under prior-weight semantics. Unit weights on this fixture do not test a new frequency-weight fit; the focused low-ratio tests cover both weight contracts.",
        },
        "same_mode_checks": {
            key: new[key] for key in equal_fields if key not in ["fingerprints", "lambdas"]
        },
        "full_newton_step": {
            "before": before[1.0],
            "after": after[1.0],
            "package_reported_penalized_gain": gain,
            "independent_penalized_gain": independent_gain,
            "gain_absolute_error": gain_error,
            "gain_comparison_roundoff_bound": error_bound,
            "optimizing_penalized_gain": after[1.0]["optimizing_penalized"]
            - after[0.0]["optimizing_penalized"],
            "interpretation": "Previously refused full Newton direction now has a finite package evaluation and improves both package objectives. This saved-mode diagnostic does not itself run the solver or establish the eventual refitted optimum.",
        },
        "curve_summary": [
            {
                "step": step,
                "old_refused": before[step]["rejection"] is not None,
                "new_refused": after[step]["rejection"] is not None,
                "independent_penalized_gain": after[step]["stable_reported_penalized"]
                - after[0.0]["stable_reported_penalized"],
                "package_vs_independent_log_likelihood": after[step][
                    "package_vs_stable_likelihood"
                ],
            }
            for step in before
        ],
        "independent_reference": "Integer-count log-likelihood recurrence accumulated in numpy.longdouble, with scipy gammaln for the parameter-independent count factorial; distinct from the package kernel. Central directional differences converge toward the freshly assembled slope.",
        "independent_regression_oracle": "The committed low-ratio test module separately uses a 220-decimal-digit Decimal finite-count recurrence for natural/log likelihood derivatives. It covers counts 0/1/7, means including the failing insurance row, ratios 2^-30/2^-40/2^-52, and prior/frequency weight contracts, plus boundary and higher-order directional checks.",
        "point_diagnostic_note": new.get("metadata_note"),
        "validation": {
            "focused_log": str(test_log),
            "focused_log_sha256": sha(test_log),
            "reported_pass_count": 168,
            "test_files": [
                "tests/test_negative_binomial_low_mean_ratio.py",
                "tests/test_negative_binomial_lss_kernel.py",
                "tests/test_negative_binomial_lss_family.py",
                "tests/test_negative_binomial_lss_efs.py",
            ],
            "limitation": "Pass count is verified from the saved log; this summarizer does not rerun tests.",
        },
        "comparison_script_sha256": sha(Path(__file__)),
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
