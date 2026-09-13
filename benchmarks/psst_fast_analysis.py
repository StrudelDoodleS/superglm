"""Compare FAST variants with PSST using matched frozen-study records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from psst_detection_analysis import LABELS, paired_summary, proportion, threshold_summary

REFERENCES = ("fast_default", "fast_purify")
NAMES = {"fast_default": "FAST default", "fast_purify": "FAST Purify", "psst": "PSST"}


def index_rows(rows):
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Study contains duplicate dataset identities")
    return indexed


def completed_method(row, name):
    method = row.get("methods", {}).get(name, {})
    return (
        method
        if "dispatch" in row and all(k in method for k in ("target_rank", "target_z", "max_z"))
        else {}
    )


def join_rows(psst, fast):
    psst, fast = index_rows(psst), index_rows(fast)
    if psst.keys() != fast.keys():
        raise ValueError("Study dataset identities differ")
    result = []
    for identity, left in sorted(psst.items()):
        right = fast[identity]
        row = {"id": identity}
        for field in ("case", "design", "strength", "replicate", "phase"):
            if left[field] != right[field]:
                raise ValueError(f"Dataset {identity} differs in {field}")
            row[field] = left[field]
        row["methods"] = {
            "psst": completed_method(left, "corrected"),
            "fast_default": completed_method(right, "old"),
            "fast_purify": completed_method(right, "corrected"),
        }
        result.append(row)
    return result


def paired_values(rows, field, reference, limit=None):
    values = []
    for row in rows:
        pair = []
        for method in (reference, "psst"):
            metrics = row["methods"][method]
            value = (
                bool(metrics) and metrics[field] <= limit
                if limit is not None
                else metrics.get(field, -np.inf)
            )
            pair.append(value)
        values.append(pair)
    return np.array(values)


def audit_fast_gains(rows):
    """Native failures can be finite negative sentinels, not only NaN/Inf."""
    count, minimum = 0, np.inf
    for row in rows:
        for name in ("old", "corrected"):
            method = completed_method(row, name)
            if not method:
                continue
            gains = np.array(list(method["all_z"].values()))
            if len(gains) != 435 or not np.isfinite(gains).all() or np.any(gains < 0):
                raise ValueError(f"Invalid native FAST gain: {row['id']}/{name}")
            count += len(gains)
            minimum = min(minimum, float(gains.min()))
    return {"scores": count, "minimum": minimum, "negative_or_nonfinite": 0}


def verify_shared_fits(psst, fast):
    """Repeated baseline/common candidate fits must reproduce their metrics."""
    fast = index_rows(fast)
    count, max_absolute = 0, 0.0
    for row in psst:
        other = fast[row["id"]]
        for name in row.get("refits", {}).keys() & other.get("refits", {}).keys():
            left, right = row["refits"][name], other["refits"][name]
            if ("error" in left) != ("error" in right):
                raise ValueError(f"Shared fit failure differs: {row['id']}/{name}")
            for metric in ("validation_loss", "test_loss", "test_risk"):
                if metric not in left or metric not in right:
                    continue
                error = abs(left[metric] - right[metric])
                # This checks deterministic replay, not solver forward accuracy.
                bound = 32 * np.finfo(float).eps * max(1.0, abs(left[metric]), abs(right[metric]))
                if not np.isfinite(error) or error > bound:
                    raise ValueError(f"Shared fit metric differs: {row['id']}/{name}/{metric}")
                max_absolute = max(max_absolute, error)
                count += 1
    return {"metric_comparisons": count, "maximum_absolute_difference": max_absolute}


def clustered_recovery(rows, reference, repeats=5000):
    grouped = defaultdict(lambda: defaultdict(list))
    values = paired_values(rows, "target_rank", reference, limit=3)
    for row, pair in zip(rows, values, strict=True):
        grouped[row["design"]][row["replicate"]].append(pair)
    rng = np.random.default_rng(87199)
    bootstrap = np.zeros(repeats)
    for _, replicates in sorted(grouped.items()):
        if len({len(block) for block in replicates.values()}) != 1:
            raise ValueError("Unbalanced replicate blocks")
        block = np.array([np.mean(v, axis=0) for _, v in sorted(replicates.items())])
        weight = sum(len(v) for v in replicates.values()) / len(rows)
        draw = rng.integers(0, len(block), size=(repeats, len(block)))
        bootstrap += weight * (block[:, 1] - block[:, 0])[draw].mean(axis=1)
    return {
        "n": len(rows),
        "reference_count": int(values[:, 0].sum()),
        "psst_count": int(values[:, 1].sum()),
        "reference_rate": float(values[:, 0].mean()),
        "psst_rate": float(values[:, 1].mean()),
        "difference": float(np.diff(values.mean(axis=0))[0]),
        "difference_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def compare(rows, reference):
    calibration, audit, signal = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        if row["phase"] == "signal":
            signal[row["case"], row["strength"]].append(row)
        elif row["phase"] == "calibration":
            calibration[row["design"]].append(row)
        elif row["phase"] == "audit":
            audit[row["design"]].append(row)
        else:
            raise ValueError("Only frozen study rows may enter the comparison")
    for groups in (calibration, audit, signal):
        for group in groups.values():
            group.sort(key=lambda row: row["replicate"])
    cal = {
        design: paired_values(group, "max_z", reference) for design, group in calibration.items()
    }
    result = {
        "method_slots": {"old": NAMES[reference], "corrected": "PSST"},
        "difference_direction": "Positive favours PSST",
        "overall_top3": clustered_recovery([r for g in signal.values() for r in g], reference),
        "null_audit": {},
        "cells": [],
    }
    for design, group in sorted(audit.items()):
        result["null_audit"][design] = threshold_summary(
            cal[design], paired_values(group, "max_z", reference), seed=917
        )
    for (case, strength), group in sorted(signal.items()):
        cell = {"case": case, "strength": strength, "n": len(group)}
        for k in (1, 3, 10):
            values = paired_values(group, "target_rank", reference, limit=k)
            summary = paired_summary(values[:, 0], values[:, 1])
            summary.update(old=proportion(values[:, 0]), corrected=proportion(values[:, 1]))
            cell[f"top{k}"] = summary
        for name, field in (("planted_detection", "target_z"), ("any_detection", "max_z")):
            cell[name] = threshold_summary(
                cal[group[0]["design"]], paired_values(group, field, reference), seed=611
            )
        measured = [
            row
            for row in group
            if all(
                metric in row["methods"][method]
                for method in (reference, "psst")
                for metric in ("test_loss_gain", "test_risk_gain")
            )
        ]
        cell["prediction_measured"] = len(measured)
        for metric in ("test_loss_gain", "test_risk_gain"):
            if measured:
                cell[metric] = paired_summary(
                    [r["methods"][reference][metric] for r in measured],
                    [r["methods"]["psst"][metric] for r in measured],
                )
        cell["selected_model_changed"] = sum(
            r["methods"][reference]["selected"] != r["methods"]["psst"]["selected"]
            for r in measured
        )
        result["cells"].append(cell)
    return result


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"fast_default": "#b07228", "fast_purify": "#7868a0", "psst": "#007d80"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for metric, title, filename in (
        ("top3", "Planted-pair recovery with three candidates", "psst-fast-recovery"),
        (
            "planted_detection",
            "Planted-pair detection at separately calibrated sweep cutoffs",
            "psst-fast-detection",
        ),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        for ax, case in zip(axes.flat, LABELS, strict=True):
            for method in (*REFERENCES, "psst"):
                reference = "fast_default" if method == "psst" else method
                slot = "corrected" if method == "psst" else "old"
                cells = [c for c in result[reference]["cells"] if c["case"] == case]
                x = [c["strength"] for c in cells]
                y = [100 * c[metric][slot]["rate"] for c in cells]
                ax.plot(x, y, "o-", label=NAMES[method], color=colors[method], markersize=4)
            ax.set_title(LABELS[case], loc="left", fontsize=11)
            ax.set_ylim(0, 103)
            ax.set_xticks([0.02, 0.04, 0.06, 0.08, 0.12])
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        fig.suptitle(title, fontsize=15, x=0.06, ha="left")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.955), ncol=3, frameon=False
        )
        fig.supxlabel(
            "Interaction strength α; Gaussian response scale / Poisson log-mean scale",
            fontsize=10,
            y=0.055,
        )
        fig.supylabel("Datasets recovered (%)", fontsize=10)
        fig.text(
            0.06,
            0.015,
            "100 datasets per point; pointwise and paired uncertainty in the JSON receipt.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0.02, 0.08, 1, 0.91))
        fig.savefig(output / f"{filename}.png", dpi=160)
        fig.savefig(output / f"{filename}.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--psst", type=Path, default=Path(".benchmark-artifacts/psst-detection-study/final")
    )
    parser.add_argument(
        "--fast", type=Path, default=Path(".benchmark-artifacts/psst-detection-study/fast-final")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmark-artifacts/psst-detection-study/fast-final/analysis"),
    )
    args = parser.parse_args()
    raw = {"psst": args.psst / "study.jsonl", "fast": args.fast / "study.jsonl"}
    records = {
        name: [json.loads(line) for line in path.read_text().splitlines()]
        for name, path in raw.items()
    }
    manifests = {
        name: json.loads((path.parent / "study-manifest.json").read_text())
        for name, path in raw.items()
    }
    for name in raw:
        if len(records[name]) != manifests[name]["tasks"]:
            raise ValueError(f"Incomplete {name} study")
    for key in manifests["psst"]:
        if key == "arguments":
            for argument in manifests["psst"][key]:
                if (
                    argument != "output"
                    and manifests["psst"][key][argument] != manifests["fast"][key][argument]
                ):
                    raise ValueError(f"Manifest argument differs: {argument}")
        elif manifests["psst"][key] != manifests["fast"][key]:
            raise ValueError(f"Manifest differs: {key}")
    rows = join_rows(records["psst"], records["fast"])
    fast_gain_audit = audit_fast_gains(records["fast"])
    result = {reference: compare(rows, reference) for reference in REFERENCES}
    result.update(
        datasets=len(rows),
        manifests=manifests,
        fast_metadata=json.loads((args.fast / "fast-metadata.json").read_text()),
        raw_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in raw.items()
        },
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_fit_replay=verify_shared_fits(records["psst"], records["fast"]),
        fast_gain_audit=fast_gain_audit,
        failed_screens={
            method: [r["id"] for r in rows if not r["methods"][method]] for method in NAMES
        },
        execution={},
        limitations=[
            "Equal weighting describes this fixed synthetic grid, not all applications.",
            "Pointwise intervals are not simultaneous claims across all cells.",
            "Prediction uses 20 datasets per cell and is exploratory.",
            "FAST strength is not a z-score; each method gets its own calibration cutoff.",
            "Timing includes instrumentation and both score variants per run; it is not an isolated speed benchmark.",
            "The grid omits sharp steps, other sample sizes and sparse-count insurance data.",
            "This compares screeners for SuperGLM refits, not full EBM/GBM pipelines.",
        ],
    )
    for name, dataset in records.items():
        screened = [r for r in dataset if "dispatch" in r]
        result["execution"][name] = {
            "successful_screens": len(screened),
            "dispatch_totals": {
                field: sum(r["dispatch"][field] for r in screened)
                for field in (
                    ("dense_ladders", "structured_ladders")
                    if name == "psst"
                    else ("default_calls", "purify_calls")
                )
            },
            "fit_seconds_median": float(np.median([r["fit_seconds"] for r in screened])),
            "screen_seconds_median": float(np.median([r["screen_seconds"] for r in screened])),
            "unique_candidate_refits": sum(len(r["refits"]) - 1 for r in dataset if "refits" in r),
            "candidate_fit_failures": sum(
                "error" in v for r in dataset for v in r.get("refits", {}).values()
            ),
            "test_evaluation_failures": sum(
                "test_error" in v for r in dataset for v in r.get("refits", {}).values()
            ),
            "warning_datasets": sum(bool(r["warnings"]) for r in dataset),
            "max_worker_peak_rss_mib": max(r["worker_peak_rss_mib"] for r in dataset),
            "sum_worker_elapsed_seconds": sum(r["elapsed_seconds"] for r in dataset),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, args.output)
    print(json.dumps({name: result[name]["overall_top3"] for name in REFERENCES}, indent=2))


if __name__ == "__main__":
    main()
