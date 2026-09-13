"""Matched prediction comparisons against the existing SuperGLM results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from psst_booster_study import BACKENDS
from psst_detection_analysis import LABELS

MODELS = ("additive", "psst", "fast_default", "fast_purify", *BACKENDS)
NAMES = {
    "additive": "Additive SuperGLM",
    "psst": "PSST + SuperGLM",
    "fast_default": "FAST default + SuperGLM",
    "fast_purify": "FAST Purify + SuperGLM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
}


def read_references(psst, fast):
    reference = {}
    for path, methods in (
        (psst, {"corrected": "psst"}),
        (fast, {"old": "fast_default", "corrected": "fast_purify"}),
    ):
        identities = set()
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["phase"] != "signal" or row["replicate"] >= 20:
                    continue
                identity = row["id"]
                if identity in identities:
                    raise ValueError("Duplicate reference dataset")
                identities.add(identity)
                if path == psst:
                    reference[identity] = {
                        key: row[key] for key in ("id", "case", "strength", "replicate", "design")
                    }
                    reference[identity]["models"] = {
                        "additive": {
                            key: row["refits"]["baseline"][key]
                            for key in ("test_loss", "test_risk")
                        }
                    }
                target = reference[identity]
                for key in ("case", "strength", "replicate", "design"):
                    if target[key] != row[key]:
                        raise ValueError(f"Reference metadata differs for {identity}/{key}")
                for slot, method in methods.items():
                    selected = row["methods"][slot]["selected"]
                    target["models"][method] = {
                        key: row["refits"][selected][key] for key in ("test_loss", "test_risk")
                    }
        if len(identities) != 600 or set(reference) != identities:
            raise ValueError("Expected the same 600 complete reference datasets")
    return reference


def join_boosters(reference, records, backends=BACKENDS):
    grouped = defaultdict(dict)
    for row in records:
        backend, identity = row["backend"], row["id"]
        if backend not in backends:
            raise ValueError(f"Unknown backend {backend}")
        if identity in grouped[backend]:
            raise ValueError("Booster study has duplicate dataset identities")
        grouped[backend][identity] = row
    for backend in backends:
        if set(grouped[backend]) != set(reference):
            raise ValueError(f"Dataset identities differ for {backend}")
    output = {}
    for identity, row in reference.items():
        output[identity] = {**row, "models": dict(row["models"])}
        for backend in backends:
            record = grouped[backend][identity]
            for key in ("case", "strength", "replicate", "design"):
                if key in row and row[key] != record[key]:
                    raise ValueError(f"Dataset metadata differs for {backend}/{identity}/{key}")
            output[identity]["models"][backend] = record
    return output


def clustered_difference(rows, model, metric, repeats=5000):
    usable = [r for r in rows if all(metric in r["models"][m] for m in ("psst", model))]
    if not usable:
        return {"planned": len(rows), "measured": 0}
    values = np.array([[r["models"][m][metric] for m in ("psst", model)] for r in usable])
    if not np.isfinite(values).all():
        raise ValueError("Finite matched metrics required")
    groups = defaultdict(list)
    for row, value in zip(usable, values, strict=True):
        # Fitting seeds are shared even across data designs. Keep their cases
        # together in family-level bootstrap samples.
        groups[row["replicate"]].append(value[1] - value[0])
    rng = np.random.default_rng(91673)
    ordered = [block for _, block in sorted(groups.items())]
    sums = np.array([sum(block) for block in ordered])
    counts = np.array([len(block) for block in ordered])
    draw = rng.integers(0, len(ordered), size=(repeats, len(ordered)))
    bootstrap = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    difference = values[:, 1] - values[:, 0]
    return {
        "planned": len(rows),
        "measured": len(usable),
        "psst_mean": float(values[:, 0].mean()),
        "model_mean": float(values[:, 1].mean()),
        "difference": float(difference.mean()),
        "difference_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "model_better": int(np.sum(difference < 0)),
        "model_worse": int(np.sum(difference > 0)),
        "replicate_blocks": len(groups),
        "joint_designs": sorted({row["design"] for row in usable}),
    }


def summarize_group(rows):
    return {
        metric: {model: clustered_difference(rows, model, metric) for model in MODELS}
        for metric in ("test_loss", "test_risk")
    }


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "additive": "#8c939c",
        "psst": "#007d80",
        "fast_purify": "#7868a0",
        "xgboost": "#bb7026",
        "catboost": "#c14e65",
        "lightgbm": "#4275b3",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, case in zip(axes.flat, LABELS, strict=True):
        cells = sorted(
            [c for c in result["cells"] if c["case"] == case], key=lambda c: c["strength"]
        )
        for model, color in colors.items():
            ax.plot(
                [c["strength"] for c in cells],
                [c["test_risk"][model].get("model_mean", np.nan) for c in cells],
                "o-",
                color=color,
                label=NAMES[model],
                markersize=3,
            )
        ax.set_title(LABELS[case], loc="left", fontsize=11)
        ax.set_ylabel("Prediction MSE" if case.startswith("gaussian") else "Poisson KL")
        ax.set_xticks([0.02, 0.04, 0.06, 0.08, 0.12])
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Prediction risk against the known generating mean", x=0.06, ha="left", fontsize=15
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.955),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.supxlabel("Interaction strength α", y=0.05)
    fig.text(
        0.06,
        0.012,
        "Lower is better. 20 datasets per point; paired uncertainty is reported in the receipt.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.88))
    fig.savefig(output / "psst-boosters-prediction.png", dpi=160)
    fig.savefig(output / "psst-boosters-prediction.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path(".benchmark-artifacts/psst-detection-study/boosters-v1")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path(".benchmark-artifacts/psst-detection-study")
    )
    args = parser.parse_args()
    paths = {
        "boosters": args.input / "study.jsonl",
        "psst": args.reference / "final/study.jsonl",
        "fast": args.reference / "fast-final/study.jsonl",
    }
    records = [json.loads(line) for line in paths["boosters"].read_text().splitlines()]
    manifest = json.loads((args.input / "study-manifest.json").read_text())
    reference_manifest = json.loads((args.reference / "final/study-manifest.json").read_text())
    if len(records) != manifest["tasks"] or manifest["tasks"] != 1800:
        raise ValueError("Expected 1800 completed library/dataset jobs")
    if (
        manifest["source_hashes"]["psst_detection_study.py"] != reference_manifest["script_sha256"]
        or manifest["package_source_sha256"] != reference_manifest["package_source_sha256"]
        or manifest["numpy"] != reference_manifest["numpy"]
    ):
        raise ValueError("Generator, numerical source or NumPy differs from the reference study")
    rows = list(join_boosters(read_references(paths["psst"], paths["fast"]), records).values())
    result = {
        "manifest": manifest,
        "reference_manifest": reference_manifest,
        "raw_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        },
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "difference_direction": "Positive means higher error than PSST-selected SuperGLM",
        "families": {},
        "cases": {},
        "cells": [],
        "execution": {},
        "limitations": [
            "Exploratory comparison on the existing smooth synthetic generators, not a broad tabular benchmark.",
            "Native category handling and equal depth menus do not equate tree capacity or fitting compute.",
            "Depth and early stopping use validation; test is evaluated only after selection.",
            "Intervals are pointwise; a replicate block includes all its data designs because fitting seeds are shared.",
            "Fresh per-job process RSS is not comparable to the prior reusable-worker SuperGLM high-water mark.",
            "This compares standalone prediction; it does not report GBM interaction-recovery rates.",
        ],
    }
    for family in ("gaussian", "poisson"):
        result["families"][family] = summarize_group(
            [r for r in rows if r["case"].startswith(family)]
        )
    for case in LABELS:
        group = [r for r in rows if r["case"] == case]
        result["cases"][case] = summarize_group(group)
        for strength in sorted({r["strength"] for r in group}):
            result["cells"].append(
                {
                    "case": case,
                    "strength": strength,
                    **summarize_group([r for r in group if r["strength"] == strength]),
                }
            )
    for backend in BACKENDS:
        group = [r for r in records if r["backend"] == backend]
        fitted = [r for r in group if "candidates" in r]
        result["execution"][backend] = {
            "jobs": len(group),
            "evaluation_failures": [r["id"] for r in group if r["status"] != "ok"],
            "candidate_failures": [
                {"id": r["id"], "candidate": name, "error": v["error"]}
                for r in fitted
                for name, v in r["candidates"].items()
                if "error" in v
            ],
            "all_booster_candidates_failed": sum(r.get("candidate_failures") == 3 for r in group),
            "selected": dict(Counter(r["selected"] for r in fitted)),
            "median_tuning_seconds": float(
                np.median([r["tuning_seconds"] + r["preprocessing_seconds"] for r in fitted])
            ),
            "median_test_predict_seconds": float(
                np.median([r["test_predict_seconds"] for r in group if "test_predict_seconds" in r])
            ),
            "median_process_peak_rss_mib": float(
                np.median([r["process_peak_rss_mib"] for r in group])
            ),
            "max_process_peak_rss_mib": max(r["process_peak_rss_mib"] for r in group),
            "median_best_rounds": float(
                np.median([r["candidates"][r["selected"]]["best_rounds"] for r in fitted])
            ),
            "candidate_fits_at_1000_round_limit": sum(
                candidate.get("trained_rounds") == 1000
                for r in fitted
                for candidate in r["candidates"].values()
            ),
            "selected_best_at_1000_round_limit": sum(
                r["candidates"][r["selected"]]["best_rounds"] == 1000 for r in fitted
            ),
            "unique_pids": len({r["pid"] for r in group}),
            "warning_jobs": sum(bool(r["warnings"]) for r in group),
            "sum_job_elapsed_seconds": sum(r["elapsed_seconds"] for r in group),
        }
    output = args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, output)
    print(json.dumps({"families": result["families"], "execution": result["execution"]}, indent=2))


if __name__ == "__main__":
    main()
