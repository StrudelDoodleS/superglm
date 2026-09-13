"""Summarize the prespecified paired study without retuning cutoffs or cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

METHODS = ("old", "corrected")
LABELS = {
    "gaussian_bilinear": "Gaussian: bilinear",
    "gaussian_wave": "Gaussian: smooth wave",
    "gaussian_spline_cat": "Gaussian: smooth × category",
    "gaussian_cat_cat": "Gaussian: category × category",
    "poisson_wave": "Poisson: smooth wave",
    "gaussian_correlated_wave": "Gaussian: correlated wave",
}


def cutoff(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    k = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.partition(values, k - 1)[k - 1]) if k <= len(values) else math.inf


def proportion(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    count = int(values.sum())
    p = count / n
    z = float(norm.ppf(0.975))
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "count": count,
        "rate": p,
        "wilson_ci": [max(0.0, center - half), min(1.0, center + half)],
    }


def paired_summary(old, corrected, seed=0, repeats=2000):
    old, corrected = np.asarray(old, dtype=float), np.asarray(corrected, dtype=float)
    if old.shape != corrected.shape or old.ndim != 1 or len(old) == 0:
        raise ValueError("Nonempty matched one-dimensional observations required")
    difference = corrected - old
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(old), size=(repeats, len(old)))
    means = difference[draw].mean(axis=1)
    return {
        "n": len(old),
        "old_mean": float(old.mean()),
        "corrected_mean": float(corrected.mean()),
        "difference": float(difference.mean()),
        "difference_ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "improved": int(np.sum(difference > 0)),
        "worsened": int(np.sum(difference < 0)),
        "unchanged": int(np.sum(difference == 0)),
    }


def threshold_summary(calibration, scores, seed=0, repeats=2000):
    calibration, scores = np.asarray(calibration), np.asarray(scores)
    if (
        calibration.ndim != 2
        or scores.ndim != 2
        or calibration.shape[1] != 2
        or scores.shape[1] != 2
    ):
        raise ValueError("Calibration and evaluation must have two matched method columns")
    cuts = np.array([cutoff(calibration[:, i]) for i in range(2)])
    observed = scores > cuts
    rng = np.random.default_rng(seed)
    cal_draw = rng.integers(0, len(calibration), size=(repeats, len(calibration)))
    k = math.ceil((len(calibration) + 1) * 0.95) - 1
    boot_cuts = np.partition(calibration[cal_draw], k, axis=1)[:, k, :]
    eval_draw = rng.integers(0, len(scores), size=(repeats, len(scores)))
    rates = (scores[eval_draw] > boot_cuts[:, None, :]).mean(axis=1)
    result = {
        "n": len(scores),
        "calibration_n": len(calibration),
        "cutoffs": dict(zip(METHODS, cuts.tolist(), strict=True)),
    }
    for i, method in enumerate(METHODS):
        result[method] = proportion(observed[:, i])
        result[method]["bootstrap_ci"] = np.quantile(rates[:, i], [0.025, 0.975]).tolist()
    result["difference"] = float(observed[:, 1].mean() - observed[:, 0].mean())
    result["difference_ci"] = np.quantile(rates[:, 1] - rates[:, 0], [0.025, 0.975]).tolist()
    result["improved"] = int(np.sum(observed[:, 1] & ~observed[:, 0]))
    result["worsened"] = int(np.sum(observed[:, 0] & ~observed[:, 1]))
    return result


def screen_succeeded(row):
    """A later validation/test failure does not erase completed screen results."""
    return "dispatch" in row and all(
        key in row.get("methods", {}).get(method, {})
        for method in METHODS
        for key in ("target_rank", "target_z", "max_z")
    )


def score_matrix(rows, field):
    return np.array(
        [
            [
                row["methods"][method][field] if screen_succeeded(row) else -np.inf
                for method in METHODS
            ]
            for row in rows
        ]
    )


def recovery(rows, limit):
    values = np.array(
        [
            [
                screen_succeeded(row) and row["methods"][method]["target_rank"] <= limit
                for method in METHODS
            ]
            for row in rows
        ]
    )
    result = paired_summary(values[:, 0], values[:, 1])
    result.update({method: proportion(values[:, i]) for i, method in enumerate(METHODS)})
    return result


def clustered_overall(rows, repeats=5000):
    """Equal-weight study average, clustering all shared-seed cells by design."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = [
            float(screen_succeeded(row) and row["methods"][m]["target_rank"] <= 3) for m in METHODS
        ]
        grouped[row["design"]][row["replicate"]].append(value)
    rng = np.random.default_rng(87199)
    bootstrap = np.zeros(repeats)
    means = np.zeros(2)
    designs = {}
    for design, by_rep in sorted(grouped.items()):
        counts = {len(values) for values in by_rep.values()}
        if len(counts) != 1:
            raise ValueError("Unbalanced replicate blocks in overall comparison")
        block = np.array([np.mean(values, axis=0) for _, values in sorted(by_rep.items())])
        weight = sum(len(v) for v in by_rep.values()) / len(rows)
        differences = block[:, 1] - block[:, 0]
        draw = rng.integers(0, len(block), size=(repeats, len(block)))
        bootstrap += weight * differences[draw].mean(axis=1)
        means += weight * block.mean(axis=0)
        designs[design] = {"replicate_blocks": len(block), "weight": weight}
    return {
        "n": len(rows),
        "old_rate": float(means[0]),
        "corrected_rate": float(means[1]),
        "difference": float(means[1] - means[0]),
        "difference_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "design_blocks": designs,
        "scope": "Equal weighting of these fixed study cells, not a distribution over real applications.",
    }


def summarize(rows, manifest):
    if len(rows) != manifest["tasks"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Study is incomplete or contains duplicate datasets")
    calibration = defaultdict(list)
    audit = defaultdict(list)
    signal = defaultdict(list)
    for row in rows:
        if row["phase"] == "calibration":
            calibration[row["design"]].append(row)
        elif row["phase"] == "audit":
            audit[row["design"]].append(row)
        elif row["phase"] == "signal":
            signal[(row["case"], row["strength"])].append(row)
        else:
            raise ValueError("Pilot/engineering rows must not enter final study")
    for groups in (calibration, audit, signal):
        for group in groups.values():
            group.sort(key=lambda row: row["replicate"])
    cal_scores = {key: score_matrix(group, "max_z") for key, group in calibration.items()}
    result = {
        "manifest": manifest,
        "datasets": len(rows),
        "failures": [r for r in rows if r["status"] != "ok"],
        "null_audit": {},
        "cells": [],
        "limitations": [
            "All intervals are pointwise; individual scenario wins are not multiplicity-adjusted discoveries.",
            "Zero discordances and a zero-width bootstrap interval do not prove equivalence.",
            "Predictive comparisons have 20 datasets per cell and are exploratory.",
            "Paired cutoff intervals resample calibration as well as audit/signal datasets.",
            "Independent Gaussian shapes and strengths share seeds; overall recovery clusters those cells by replicate and design.",
            "The study isolates normalization; it does not compare all historical numerical/routing changes or FAST.",
        ],
    }
    for design, group in sorted(audit.items()):
        result["null_audit"][design] = threshold_summary(
            cal_scores[design], score_matrix(group, "max_z"), seed=917
        )
    for (case, strength), group in sorted(signal.items()):
        cell = {
            "case": case,
            "strength": strength,
            "n": len(group),
            "failed_screens": sum(not screen_succeeded(r) for r in group),
        }
        for k in (1, 3, 10):
            cell[f"top{k}"] = recovery(group, k)
        cell["planted_detection"] = threshold_summary(
            cal_scores[group[0]["design"]], score_matrix(group, "target_z"), seed=611
        )
        cell["any_detection"] = threshold_summary(
            cal_scores[group[0]["design"]], score_matrix(group, "max_z"), seed=612
        )
        ok = [r for r in group if screen_succeeded(r)]
        cell["shortlist_changed"] = sum(
            set(r["methods"]["old"]["top3"]) != set(r["methods"]["corrected"]["top3"]) for r in ok
        )
        selected = [r for r in group if r["replicate"] < manifest["arguments"]["refit_replicates"]]
        measured = [
            r
            for r in selected
            if all(
                metric in r.get("methods", {}).get(m, {})
                for m in METHODS
                for metric in ("test_loss_gain", "test_risk_gain")
            )
        ]
        cell["prediction_planned"] = len(selected)
        cell["prediction_measured"] = len(measured)
        cell["refit_failures"] = sum(
            "error" in metrics for r in selected for metrics in r.get("refits", {}).values()
        )
        if measured:
            cell["test_loss_gain"] = paired_summary(
                [r["methods"]["old"]["test_loss_gain"] for r in measured],
                [r["methods"]["corrected"]["test_loss_gain"] for r in measured],
            )
            cell["test_risk_gain"] = paired_summary(
                [r["methods"]["old"]["test_risk_gain"] for r in measured],
                [r["methods"]["corrected"]["test_risk_gain"] for r in measured],
            )
            cell["selected_model_changed"] = sum(
                r["methods"]["old"]["selected"] != r["methods"]["corrected"]["selected"]
                for r in measured
            )
        result["cells"].append(cell)
    result["overall_top3"] = clustered_overall([r for group in signal.values() for r in group])
    ok = [r for r in rows if screen_succeeded(r)]
    result["execution"] = {
        "successful_screens": len(ok),
        "candidate_scores_per_method": 435 * len(ok),
        "actual_dense_ladders": sum(r["dispatch"]["dense_ladders"] for r in ok),
        "actual_structured_ladders": sum(r["dispatch"]["structured_ladders"] for r in ok),
        "unique_candidate_refits": sum(len(r.get("refits", {})) - 1 for r in ok if "refits" in r),
        "warning_datasets": sum(bool(r["warnings"]) for r in rows),
        "fit_seconds_median": float(np.median([r["fit_seconds"] for r in ok])),
        "screen_seconds_median": float(np.median([r["screen_seconds"] for r in ok])),
        "max_worker_peak_rss_mib": max(r["worker_peak_rss_mib"] for r in rows),
        "sum_worker_elapsed_seconds": sum(r["elapsed_seconds"] for r in rows),
    }
    return result


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"old": "#7e8794", "corrected": "#007d80"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for metric, title, filename in (
        (
            "top3",
            "Recovery of the planted pair in a three-candidate shortlist",
            "psst-shortlist-recovery",
        ),
        (
            "planted_detection",
            "Planted pair exceeds a cutoff targeting 5% any-pair false alarms",
            "psst-calibrated-detection",
        ),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        for ax, case in zip(axes.flat, LABELS, strict=True):
            cells = sorted(
                [c for c in result["cells"] if c["case"] == case], key=lambda c: c["strength"]
            )
            x = [c["strength"] for c in cells]
            for method in METHODS:
                y = np.array([c[metric][method]["rate"] for c in cells]) * 100
                interval = "wilson_ci" if metric == "top3" else "bootstrap_ci"
                bands = np.array([c[metric][method][interval] for c in cells]) * 100
                ax.plot(
                    x,
                    y,
                    "o-",
                    color=colors[method],
                    label="Previous denominator" if method == "old" else "Corrected denominator",
                    markersize=4,
                )
                ax.fill_between(x, bands[:, 0], bands[:, 1], color=colors[method], alpha=0.12)
            ax.set_title(LABELS[case], loc="left", fontsize=11)
            ax.set_ylim(0, 103)
            ax.set_xticks([0.02, 0.04, 0.06, 0.08, 0.12])
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        fig.suptitle(title, fontsize=15, x=0.06, ha="left")
        fig.supxlabel(
            "Interaction strength α; Gaussian response scale / Poisson log-mean scale", fontsize=10
        )
        fig.supylabel("Datasets recovered (%)", fontsize=10)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.955), ncol=2, frameon=False
        )
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.91))
        fig.savefig(output / f"{filename}.png", dpi=160)
        fig.savefig(output / f"{filename}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ax, case in zip(axes.flat, LABELS, strict=True):
        cells = sorted(
            [c for c in result["cells"] if c["case"] == case and "test_risk_gain" in c],
            key=lambda c: c["strength"],
        )
        x = np.array([c["strength"] for c in cells])
        values = np.array([c["test_risk_gain"]["difference"] for c in cells])
        intervals = np.array([c["test_risk_gain"]["difference_ci"] for c in cells])
        ax.axhline(0, color="#7e8794", linewidth=1)
        if len(cells):
            # Draw endpoints directly: percentile intervals need not contain
            # their point estimate in every finite-sample bootstrap.
            ax.vlines(x, intervals[:, 0], intervals[:, 1], color=colors["corrected"], linewidth=2)
            ax.plot(x, values, "o-", color=colors["corrected"], markersize=4)
        ax.set_title(LABELS[case], loc="left", fontsize=11)
        ax.set_xticks([0.02, 0.04, 0.06, 0.08, 0.12])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        ax.set_ylabel("MSE improvement" if "gaussian" in case else "Poisson KL improvement")
    fig.suptitle(
        "Prediction benefit from the correction after three candidate refits",
        fontsize=15,
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.91,
        "Positive favours corrected scores; 20 datasets per point; paired 95% intervals; validation selects, test evaluates.",
        fontsize=9,
    )
    fig.supxlabel("Interaction strength α", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.89))
    fig.savefig(output / "psst-prediction-gain.png", dpi=160)
    fig.savefig(output / "psst-prediction-gain.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path(".benchmark-artifacts/psst-detection-study/final")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".benchmark-artifacts/psst-detection-study/final/analysis"),
    )
    args = parser.parse_args()
    raw = args.input / "study.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    manifest = json.loads((args.input / "study-manifest.json").read_text())
    result = summarize(rows, manifest)
    result["raw_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, args.output)
    print(
        json.dumps(
            {
                "overall": result["overall_top3"],
                "execution": result["execution"],
                "null_audit": result["null_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
