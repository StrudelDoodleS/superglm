"""Plot ten saved Gaussian interaction terms without fitting or selecting models."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.spatial import Delaunay

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
import benchmark_broad_interactions as broad  # noqa: E402
import benchmark_real_interactions as base  # noqa: E402
import broad_interaction_data as data  # noqa: E402

RUN = REPO / ".benchmark-artifacts/broad-interactions/frozen-20260914"
OUTPUT = Path(__file__).with_name("figures") / "2026-09-14-broad-interactions"
CASES = {
    "uci_airfoil": ("airfoil", "Airfoil", "Sound pressure contribution (dB)", 1),
    "uci_concrete": ("concrete", "Concrete", "Strength contribution (MPa)", 1),
    "kaggle_king_county_sales": (
        "king-county",
        "King County housing",
        "Price contribution ($1,000)",
        1000,
    ),
}
LABELS = {
    "frequency": "Frequency (Hz)",
    "suction-side-displacement-thickness": "Boundary-layer displacement thickness (mm)",
    "attack-angle": "Angle of attack (degrees)",
    "Cement": "Cement (kg/m³)",
    "Water": "Water (kg/m³)",
    "Age": "Curing age (days)",
    "Blast Furnace Slag": "Blast furnace slag (kg/m³)",
    "sqft_living": "Living area (sq ft)",
    "grade": "Building grade",
    "lat": "Latitude (degrees)",
    "long": "Longitude (degrees)",
}
LOG_AXES = {"frequency", "suction-side-displacement-thickness", "Age", "sqft_living"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_values(name, values):
    return values * 1000 if name == "suction-side-displacement-thickness" else values


def raw_axis(name, values):
    lo, hi = np.quantile(values, [0.01, 0.99])
    if name == "grade":
        return np.arange(np.ceil(lo), np.floor(hi) + 1)
    if name in LOG_AXES:
        assert lo > 0
        return np.geomspace(lo, hi, 121)
    return np.linspace(lo, hi, 121)


def configure_axis(ax, axis, name):
    getattr(ax, f"set_{axis}label")(LABELS[name], fontsize=10)
    if name in LOG_AXES:
        getattr(ax, f"set_{axis}scale")("log")
        ticks = {
            "frequency": [200, 500, 1000, 2000, 5000, 10000],
            "suction-side-displacement-thickness": [0.5, 1, 2, 5, 10, 20, 40],
            "Age": [3, 7, 14, 28, 90, 180, 365],
            "sqft_living": [800, 1000, 2000, 3000, 5000],
        }[name]
        getattr(ax, f"set_{axis}ticks")(ticks)
    else:
        getattr(ax, f"{axis}axis").set_major_locator(MaxNLocator(nbins=5, integer=name == "grade"))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    getattr(ax, f"{axis}axis").set_major_formatter(formatter)
    ax.tick_params(axis=axis, which="minor", labelsize=0)


def case_surfaces(dataset, measurement):
    case = measurement["datasets"][dataset]
    chosen = case["choice"]["chosen_arm"]
    fit = case["fits"][chosen]
    folder = RUN / dataset / chosen
    model_path = folder / "model.pkl"
    assert sha(model_path) == fit["model_pickle_sha256"]
    prepared = data.load_prepared(dataset)
    assert prepared["metadata"] == case["data"]
    assert prepared["metadata"]["family"] == "gaussian"
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    assert type(model._link).__name__ == "IdentityLink"
    raw = prepared["frame"].iloc[prepared["rows"]["train"]]
    raw = raw.loc[:, prepared["entry"]["features"]]
    assert np.all(prepared["sample_weight"][prepared["rows"]["train"]] == 1)
    test_raw, _, _ = broad.partition(prepared, "test")
    test = base.transform_features(test_raw, prepared["state"])
    with np.load(folder / "test_predictions.npz", allow_pickle=False) as saved:
        assert sha(folder / "test_predictions.npz") == fit["evaluation"]["test_predictions_sha256"]
        np.testing.assert_array_equal(model.predict(test), saved["prediction"])
    surfaces = []
    for left, right in fit["pairs"]:
        term_name = f"{left}:{right}"
        spec = model._interaction_specs[term_name]
        assert type(spec).__name__ == "TensorInteraction"
        groups = [group for group in model._groups if group.feature_name == term_name]
        beta = np.concatenate([model.result.beta[group.sl] for group in groups])
        observed = [raw[name].to_numpy(dtype=float) for name in (left, right)]
        assert all(np.isfinite(values).all() for values in observed)
        axes = [
            raw_axis(name, values) for name, values in zip((left, right), observed, strict=True)
        ]
        mesh = np.meshgrid(*axes)
        scaled = [
            (values.ravel() - prepared["state"]["features"][name]["center"])
            / prepared["state"]["features"][name]["scale"]
            for name, values in zip((left, right), mesh, strict=True)
        ]
        effect = spec.score(*scaled, beta).reshape(mesh[0].shape)
        assert np.isfinite(effect).all()
        # Use only observed raw pair geometry for this display mask. A convex
        # hull does not certify density, identifiability, or statistical precision.
        observed_pairs = np.column_stack(observed)
        center = observed_pairs.mean(axis=0)
        scale = observed_pairs.std(axis=0)
        assert np.all(scale > 0)
        hull = Delaunay(np.unique((observed_pairs - center) / scale, axis=0))
        query = (np.column_stack([values.ravel() for values in mesh]) - center) / scale
        supported = (hull.find_simplex(query) >= 0).reshape(effect.shape)
        surfaces.append(
            {
                "name": term_name,
                "parents": [left, right],
                "axes": axes,
                "observed": observed,
                "effect": effect,
                "supported": supported,
            }
        )
    return case, fit, surfaces


def plot_case(dataset, case, fit, surfaces, pdf):
    slug, title, unit, divisor = CASES[dataset]
    n_rows = (len(surfaces) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12.3, 4.6 * n_rows + 0.7), squeeze=False)
    fig.subplots_adjust(
        left=0.085,
        right=0.94,
        bottom=0.17 if n_rows == 1 else 0.105,
        top=0.82 if n_rows == 1 else 0.86,
        hspace=0.47,
        wspace=0.47,
    )
    fig.suptitle(
        f"{title}: fitted interaction contributions",
        x=0.085,
        ha="left",
        y=0.98,
        fontsize=19,
        fontweight="normal",
    )
    reduction = case["test_comparison"]["vs_best_additive_percent"]
    fig.text(
        0.085,
        0.915 if n_rows == 1 else 0.94,
        f"Selected {len(surfaces)}-pair model: {reduction:.1f}% lower test MSE than its additive control",
        fontsize=11,
        color="#444444",
    )
    receipt = {
        "dataset": dataset,
        "chosen_arm": case["choice"]["chosen_arm"],
        "model_sha256": fit["model_pickle_sha256"],
        "test_prediction_replay": "exact",
        "response_display_divisor": divisor,
        "surfaces": [],
    }
    array_payload = {}
    for i, (ax, surface) in enumerate(zip(axes.flat, surfaces, strict=True)):
        left, right = surface["parents"]
        x, y = [
            display_values(name, values)
            for name, values in zip(surface["parents"], surface["axes"], strict=True)
        ]
        effect = surface["effect"] / divisor
        effect_masked = np.ma.masked_where(~surface["supported"], effect)
        limit = float(np.max(np.abs(effect_masked)))
        assert limit > 0
        ax.set_facecolor("#e0e2e5")
        mesh = ax.pcolormesh(
            x,
            y,
            effect_masked,
            shading="nearest",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
            rasterized=True,
        )
        observed_x, observed_y = [
            display_values(name, values)
            for name, values in zip(surface["parents"], surface["observed"], strict=True)
        ]
        ax.scatter(
            observed_x, observed_y, s=3, alpha=0.17, c="#151515", linewidths=0, rasterized=True
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        configure_axis(ax, "x", left)
        configure_axis(ax, "y", right)
        ax.set_title(
            f"{chr(65 + i)}. {LABELS[left].split(' (')[0]} ×\n{LABELS[right].split(' (')[0]}",
            fontsize=11,
            loc="left",
            pad=9,
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.045)
        cbar.set_label(unit, fontsize=9)
        cbar.ax.tick_params(labelsize=9)
        cbar.locator = MaxNLocator(nbins=5)
        cbar.update_ticks()
        for spine in ax.spines.values():
            spine.set_color("#b0b0b0")
        ax.tick_params(labelsize=9)
        prefix = f"pair_{i}"
        array_payload.update(
            {
                f"{prefix}_x": surface["axes"][0],
                f"{prefix}_y": surface["axes"][1],
                f"{prefix}_contribution": surface["effect"],
                f"{prefix}_inside_hull": surface["supported"],
            }
        )
        receipt["surfaces"].append(
            {
                "name": surface["name"],
                "grid_shape": list(effect.shape),
                "raw_axis_bounds": [[float(v[0]), float(v[-1])] for v in surface["axes"]],
                "training_rows": len(observed_x),
                "visible_grid_cells": int(surface["supported"].sum()),
                "colour_limit_display_units": limit,
                "visible_effect_range_original_units": [
                    float(np.min(surface["effect"][surface["supported"]])),
                    float(np.max(surface["effect"][surface["supported"]])),
                ],
            }
        )
    fig.text(
        0.085,
        0.065 if n_rows == 1 else 0.046,
        "Red adds to the fitted prediction; blue subtracts. Colour scales differ by panel. Dots are training observations.",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.085,
        0.027 if n_rows == 1 else 0.021,
        "Axes show the central 98% of each predictor. Grey is outside the observed pair convex hull, not an uncertainty band.",
        fontsize=9,
        color="#444444",
    )
    png = OUTPUT / f"{slug}.png"
    svg = OUTPUT / f"{slug}.svg"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    npz = OUTPUT / f"{slug}-surfaces.npz"
    np.savez_compressed(npz, **array_payload)
    receipt["artifacts"] = {path.name: sha(path) for path in (png, svg, npz)}
    return receipt


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    measurement_path = Path(__file__).with_name("2026-09-14-broad-interaction-measurements.json")
    measurement = json.loads(measurement_path.read_text())
    assert broad.source_identity() == measurement["protocol"]["source"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    receipt = {
        "plot_script_sha256": sha(Path(__file__)),
        "measurement_sha256": sha(measurement_path),
        "scope": "Saved Gaussian identity-link tensor terms; no fits or selection; raw outcome units; no exponential relativity conversion or extra centering.",
        "view": "Training marginal 1st-99th percentiles, integer building grades, log display axes as declared in the script; observed-pair convex-hull mask is visual only.",
        "cases": [],
    }
    with PdfPages(OUTPUT / "interaction-surfaces.pdf") as pdf:
        for dataset in CASES:
            case, fit, surfaces = case_surfaces(dataset, measurement)
            receipt["cases"].append(plot_case(dataset, case, fit, surfaces, pdf))
    receipt["pdf_sha256"] = sha(OUTPUT / "interaction-surfaces.pdf")
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
