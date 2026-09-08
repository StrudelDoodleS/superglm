"""Reproduce the GPD negative-control receipt from saved artifacts; never refit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    record = json.loads(path.read_text())
    sidecar = path.with_suffix(".sha256.json")
    if sidecar.exists():
        for filename, expected in json.loads(sidecar.read_text()).items():
            assert sha(path.parent / filename) == expected
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert (arrays["eta_variance"] > 0).all()
    assert record["source_stable_during_fit"]
    assert record["evidence_files_stable_during_fit"]
    assert arrays["parameters"].shape == (record["n"], 2)
    assert arrays["covariance"].shape == (record["q"], record["q"])
    return record, arrays


def window(record, arrays):
    if "coefficient_fit_theta" not in arrays:
        return {
            "available": False,
            "reason": "Historical artifact did not save per-fit parameters.",
        }
    config = record["smoothing_config"]
    history = record["outer_history"]
    accepted = [item for item in history if item["accepted"]]
    tail = accepted[-config["plateau_iterations"] :]
    rows = []
    for item in tail:
        source = arrays["coefficient_fit_theta"][item["source_fit_index"]]
        candidate = arrays["coefficient_fit_theta"][item["accepted_fit_index"]]
        relative = np.abs(candidate - source) / (1 + np.maximum(np.abs(source), np.abs(candidate)))
        change = float(relative.max())
        rows.append(
            {
                "iteration": item["iteration"],
                "source_fit_index": item["source_fit_index"],
                "accepted_fit_index": item["accepted_fit_index"],
                "relative_objective_change": item["objective_relative_change"],
                "all_parameter_change": change,
                "parameter_change_by_column": relative.max(axis=0).tolist(),
                "max_accepted_log_step": item["max_accepted_log_step"],
                "objective_gate_pass": item["objective_relative_change"]
                <= config["plateau_tolerance"],
                "parameter_gate_pass": change <= config["practical_parameter_tolerance"],
            }
        )
    joint = [r["objective_gate_pass"] and r["parameter_gate_pass"] for r in rows]
    assert not all(joint), "Negative control unexpectedly satisfies its last accepted window"
    first_source = arrays["coefficient_fit_theta"][tail[0]["source_fit_index"]]
    last_candidate = arrays["coefficient_fit_theta"][tail[-1]["accepted_fit_index"]]
    cumulative_relative = np.abs(last_candidate - first_source) / (
        1 + np.maximum(np.abs(first_source), np.abs(last_candidate))
    )
    cumulative_objective = abs(tail[-1]["objective_after"] - tail[0]["objective_before"]) / (
        1 + abs(tail[0]["objective_before"])
    )
    cumulative_parameter = float(cumulative_relative.max())
    assert cumulative_parameter > config["practical_parameter_tolerance"]
    return {
        "available": True,
        "required_consecutive_accepted_steps": config["plateau_iterations"],
        "parameter_scale": "abs(candidate-source)/(1+max(abs(source),abs(candidate))); maximum over all rows and both natural parameters",
        "rows": rows,
        "objective_and_parameter_window_pass": all(joint),
        "cumulative_window": {
            "relative_objective_change": cumulative_objective,
            "all_parameter_change": cumulative_parameter,
            "parameter_change_by_column": cumulative_relative.max(axis=0).tolist(),
            "objective_gate_pass": cumulative_objective <= config["plateau_tolerance"],
            "parameter_gate_pass": cumulative_parameter <= config["practical_parameter_tolerance"],
        },
        "limitation": "This is a necessary-window check. Rejections, lower-bound pressure, exact-face state, and terminal boundary assessment impose additional gates.",
    }


def difference(candidate, reference):
    delta = candidate - reference
    return {
        "max_absolute": float(np.abs(delta).max()),
        "relative_l2_or_frobenius": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
        "stable_relative_max": float(
            (np.abs(delta) / (1 + np.maximum(np.abs(candidate), np.abs(reference)))).max()
        ),
    }


def summary(path, record, arrays):
    history = record["outer_history"]
    fits = record["coefficient_fits"]
    scalar_keys = [
        "converged",
        "certified",
        "reason",
        "objective",
        "edf",
        "log_likelihood",
        "lambdas",
        "unresolved",
        "terminal_projected_gradient_norm",
        "terminal_gradient",
        "terminal_gradient_certificate",
        "stationarity_bar",
        "terminal_fit_index",
        "terminal_evidence_fresh",
        "terminal_raw_log_steps",
        "endpoint_directions",
    ]
    return {
        "raw_json": str(path),
        "raw_json_sha256": sha(path),
        "raw_npz_sha256": sha(path.with_suffix(".npz")),
        "source": {
            key: record["source"][key] for key in ["sha", "diff_sha256", "source_tree_sha256"]
        },
        "source_after": record["source_after"],
        "source_stable_during_fit": record["source_stable_during_fit"],
        "harness_sha256": record["harness_sha256"],
        "fixtures_sha256": record["fixtures_sha256"],
        "provenance_helper_sha256": record["provenance_helper_sha256"],
        "evidence_files_stable_during_fit": record["evidence_files_stable_during_fit"],
        "settings": record["settings"],
        "smoothing_config": record["smoothing_config"],
        "result": {key: record.get(key) for key in scalar_keys},
        "work": {
            "outer_iterations": len(history),
            "accepted_outer_iterations": sum(h["accepted"] for h in history),
            "stages": dict(Counter(h["stage"] for h in history)),
            "coefficient_fits": len(fits),
            "coefficient_iterations_total": sum(f["iterations"] for f in fits),
            "coefficient_backtracks_total": sum(f["backtracking_steps"] for f in fits),
        },
        "last_accepted_window": window(record, arrays),
        "timing_status": "UNMEASURED",
        "raw_timing_status": record["timing"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=root.parent / "c3-c1-completion")
    args = parser.parse_args()
    artifacts = root / ".benchmark-artifacts/c3-practical/gpd-outward"
    paths = {
        name: artifacts / f"{name}.json"
        for name in ["before", "after", "before-widecap", "after-widecap"]
    }
    paths["historical_strict"] = (
        args.baseline_root / "benchmarks/results/c3-c1-stress-v2/current_gpd_newton_001.json"
    )
    loaded = {name: load(path) for name, path in paths.items()}
    reference, reference_arrays = loaded["historical_strict"]
    assert reference["reason"] == "stationary" and reference["certified"]
    for record, arrays in loaded.values():
        assert record["input_sha256"] == reference["input_sha256"]
        assert arrays["parameters"].shape == reference_arrays["parameters"].shape
    pairs = {}
    for old, new in [("before", "after"), ("before-widecap", "after-widecap")]:
        a, aa = loaded[old]
        b, ba = loaded[new]
        assert a["smoothing_config"] == b["smoothing_config"]
        assert a["harness_sha256"] == b["harness_sha256"]
        assert set(aa) == set(ba)
        assert all(np.array_equal(aa[key], ba[key]) for key in aa)
        result_fields = [
            "objective",
            "edf",
            "log_likelihood",
            "lambdas",
            "reason",
            "converged",
            "certified",
            "terminal_fit_index",
            "outer_history",
            "coefficient_fits",
        ]
        assert all(a[key] == b[key] for key in result_fields)
        pairs[new] = {
            "baseline": old,
            "all_saved_arrays_bitwise_equal": True,
            "result_history_and_work_equal": True,
            "same_config_and_harness": True,
        }
    comparisons = {}
    for name in ["after", "after-widecap"]:
        record, arrays = loaded[name]
        comparisons[name] = {
            "reference": "historical_strict",
            "objective_delta": record["objective"] - reference["objective"],
            "edf_delta": record["edf"] - reference["edf"],
            "parameters_all": difference(arrays["parameters"], reference_arrays["parameters"]),
            "parameters_by_column": [
                difference(arrays["parameters"][:, k], reference_arrays["parameters"][:, k])
                for k in range(2)
            ],
            "conditional_linear_predictor_se": difference(
                np.sqrt(arrays["eta_variance"]), np.sqrt(reference_arrays["eta_variance"])
            ),
            "conditional_linear_predictor_se_by_column": [
                difference(
                    np.sqrt(arrays["eta_variance"][:, k]),
                    np.sqrt(reference_arrays["eta_variance"][:, k]),
                )
                for k in range(2)
            ],
            "covariance": difference(arrays["covariance"], reference_arrays["covariance"]),
        }
    receipt = {
        "schema": 1,
        "scope": "Paired GPD negative controls for the practical-boundary policy; these controls retain meaningful parameter motion and should not claim practical convergence.",
        "interpretation": "Within each pair, unchanged outputs demonstrate the gate continues to refuse. The historical certified strict fit uses a different start and outer method: comparisons quantify different stopping solutions, not equivalence or representation parity.",
        "timing": "UNMEASURED: numerical diagnostics potentially concurrent; no speed claim.",
        "harness_note": "The helper changed between the original-cap and wide-cap pairs. Each run records its actual harness hash; equality is asserted only within each pair.",
        "inference_note": "Conditional linear-predictor SE is sqrt(eta_variance), computed from predictor design and the conditional coefficient covariance; it excludes smoothing-parameter uncertainty.",
        "parameter_columns": ["scale", "shape"],
        "linear_predictor_columns": ["log(scale)", "bounded-logit(shape)"],
        "gate_findings": [
            "Original cap 1e10 and plateau tolerance 2e-6: objective changes pass the last three accepted steps, but parameter movement is 0.0030234, 0.0019716, 0.00096098. Only one consecutive step passes the 0.001 parameter tolerance; the required three-step window fails. The terminal cap refusal is not an accepted small step.",
            "Wide cap 1e12 and default plateau tolerance 1e-7: last three accepted objective changes are 3.2425e-7, 3.7134e-7, 2.3531e-9, with parameter movements 0.0019716, 0.0039288, 0.000022061. Only the last step passes both tests; a later objective rejection cannot complete the window.",
            "The strict reference has a lower objective by about 0.045-0.048, but substantial uncertainty differences remain: coefficient covariance Frobenius differences are about 32-33%, and shape-link conditional SE relative L2 differences about 10.5%. No equivalence claim follows from a small objective gap.",
        ],
        "input_sha256": reference["input_sha256"],
        "n": reference["n"],
        "q": reference["q"],
        "comparison_script_sha256": sha(Path(__file__)),
        "pairs": pairs,
        "runs": {name: summary(paths[name], *value) for name, value in loaded.items()},
        "versus_historical_strict": comparisons,
    }
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
