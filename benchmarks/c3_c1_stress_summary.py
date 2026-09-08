"""Validate and compact source-bound C3 diagnostic receipts without solver imports."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(directory):
    sources = {}
    fits = []
    reference_checks = {}
    for path in sorted(directory.glob("*.json")):
        receipt = json.loads(path.read_text())
        if receipt.get("schema") != 2 or "coefficient_fits" not in receipt:
            continue
        hashes = json.loads(path.with_suffix(".sha256.json").read_text())
        for filename, expected in hashes.items():
            if digest(directory / filename) != expected:
                raise ValueError(f"Receipt checksum mismatch: {filename}")
        source = receipt["source"]
        tree_hash = hashlib.sha256(
            json.dumps(source["source_files_sha256"], sort_keys=True).encode()
        ).hexdigest()
        if tree_hash != source["source_tree_sha256"]:
            raise ValueError(f"Invalid source file tree hash: {path.name}")
        if (
            not receipt["source_stable_during_fit"]
            or not receipt["evidence_files_stable_during_fit"]
        ):
            raise ValueError(f"Source or evidence changed: {path.name}")
        for name in ("sha", "source_tree_sha256"):
            if source[name] != receipt["source_after"][name]:
                raise ValueError(f"Source stability mismatch: {path.name}")
        for name, value in receipt["evidence_files_after"].items():
            if receipt[name] != value:
                raise ValueError(f"Evidence stability mismatch: {path.name}")
        if not Path(receipt["imported_module"]).is_relative_to(Path(source["root"]) / "src"):
            raise ValueError(f"Imported source mismatch: {path.name}")
        source_key = source["sha"] + ":" + tree_hash
        sources[source_key] = {
            key: source[key] for key in ("root", "sha", "source_tree_sha256", "source_files_sha256")
        }
        terminal = receipt["coefficient_fits"][receipt["terminal_fit_index"]]
        fields = (
            "case",
            "settings",
            "n",
            "q",
            "converged",
            "certified",
            "reason",
            "objective",
            "edf",
            "lambdas",
            "unresolved",
            "terminal_projected_gradient_norm",
            "terminal_gradient",
            "terminal_gradient_certificate",
            "stationarity_bar",
            "terminal_evidence_fresh",
            "input_sha256",
            "harness_sha256",
            "fixtures_sha256",
            "provenance_helper_sha256",
            "source_stable_during_fit",
            "evidence_files_stable_during_fit",
            "python",
            "executable",
            "dependencies",
            "smoothing_config",
            "thread_environment",
            "threadpools_before",
            "threadpools_during_fit",
            "threadpools_after",
            "numba_threads_before",
            "numba_threads_during_fit",
            "numba_threads_after",
        )
        record = {key: receipt[key] for key in fields}
        record.update(
            raw_file=path.name,
            raw_json_sha256=digest(path),
            raw_npz_sha256=digest(path.with_suffix(".npz")),
            source_key=source_key,
            coefficient_fit_count=len(receipt["coefficient_fits"]),
            terminal_coefficient_converged=terminal["converged"],
            terminal_coefficient_reason=terminal["convergence_reason"],
        )
        if receipt["reason"] == "stationary":
            bar = receipt["stationarity_bar"]
            norm = receipt["terminal_projected_gradient_norm"]
            certificate = receipt["terminal_gradient_certificate"]
            finite = all(math.isfinite(value) for value in (bar, norm, *certificate.values()))
            expected_bar = receipt["smoothing_config"]["tolerance"] * (
                1 + abs(receipt["objective"])
            )
            passes = (
                finite
                and bar > 0
                and 0 <= norm <= bar
                and all(0 <= value <= bar for value in certificate.values())
                and receipt["converged"]
                and receipt["certified"]
                and receipt["terminal_evidence_fresh"]
                and terminal["converged"]
                and not receipt["unresolved"]
                and bar == expected_bar
            )
            if not passes:
                raise ValueError(f"Invalid terminal stationary authority: {path.name}")
            record["reported_terminal_contract_recomputed"] = True
            record["raw_terminal_gradient_max_abs"] = max(
                abs(value) for value in receipt["terminal_gradient"].values()
            )
        if receipt["numba_threads_during_fit"] != 1 or any(
            pool["num_threads"] != 1 for pool in receipt["threadpools_during_fit"]
        ):
            raise ValueError(f"Expected one runtime thread: {path.name}")
        fits.append(record)
        references = {
            key: receipt[key]
            for key in ("scipy_log_likelihood_difference", "row_reference", "laml_oracle")
            if key in receipt
        }
        if references:
            reference_checks[path.name] = references
    if not fits:
        raise ValueError("No schema-2 fit receipts found")
    comparisons = {}
    for source in ("baseline", "current"):
        for case in ("gpd", "tweedie"):
            pairs = []
            for left, right in itertools.combinations(("001", "01", "1"), 2):
                with (
                    np.load(directory / f"{source}_{case}_newton_{left}.npz") as a,
                    np.load(directory / f"{source}_{case}_newton_{right}.npz") as b,
                ):
                    prediction = (a["parameters"] - b["parameters"]) / a["parameters"]
                    se_a, se_b = np.sqrt(a["eta_variance"]), np.sqrt(b["eta_variance"])
                    standard_error = (se_a - se_b) / se_a
                    pairs.append(
                        {
                            "starts": [left, right],
                            "prediction_relative_rms": np.sqrt(
                                np.mean(prediction**2, axis=0)
                            ).tolist(),
                            "conditional_link_se_relative_rms": np.sqrt(
                                np.mean(standard_error**2, axis=0)
                            ).tolist(),
                        }
                    )
            comparisons[f"{source}_{case}"] = pairs
    return {
        "schema": 2,
        "timing": "UNMEASURED: numerical diagnostics, potentially concurrent",
        "authority_scope": (
            "Recomputes the existing API first-order stopping contract from published terminal "
            "values; Richardson-refinement indicators are not total derivative-error enclosures. "
            "This does not prove an exact-gradient bound, a local minimum, or global optimality."
        ),
        "summarizer_sha256": digest(Path(__file__)),
        "raw_directory": str(directory),
        "sources": sources,
        "fits": fits,
        "start_comparisons": comparisons,
        "independent_references": reference_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    files = sorted(args.directory.glob("*.json")) + sorted(args.directory.glob("*.npz"))
    manifest = {
        path.name: digest(path)
        for path in files
        if path.name not in ("manifest.sha256.json", "summary.json")
    }
    (args.directory / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Verified {len(result['fits'])} fit receipts across {len(result['sources'])} source trees"
    )


if __name__ == "__main__":
    main()
