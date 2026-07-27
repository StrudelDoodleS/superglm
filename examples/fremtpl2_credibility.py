"""French MTPL demonstration of random effects and factor smooths.

The source is OpenML data set 41214, ``freMTPL2freq``:
https://www.openml.org/d/41214

The preprocessing caps follow scikit-learn's insurance-pricing example:
https://scikit-learn.org/stable/auto_examples/linear_model/
plot_tweedie_regression_insurance_claims.html

Run from the repository root:

    uv run python examples/fremtpl2_credibility.py --max-rows 30000

The script compares one common baseline with fixed versus random vehicle-brand
effects, a driver-age-by-region factor smooth, and both credibility terms
together. It writes held-out metrics, credibility tables, curves, and a compact
diagnostic plot.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from superglm import (
    Categorical,
    FactorSmooth,
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
)

OPENML_DATA_ID = 41214
DEFAULT_VARIANTS = ("baseline", "brand_fixed", "re", "fs", "re_fs")
MODEL_COLUMNS = (
    "DrivAge",
    "VehAge",
    "BonusMalus",
    "LogDensity",
    "VehPower",
    "VehGas",
    "Area",
    "Region",
    "VehBrand",
)


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load a local copy or fetch the public OpenML frame."""
    if path is None:
        return fetch_openml(
            data_id=OPENML_DATA_ID,
            as_frame=True,
            parser="auto",
        ).frame
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError("--data must be a CSV or Parquet file")


def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented caps and derive log population density."""
    required = {
        "ClaimNb",
        "Exposure",
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"freMTPL2freq columns are missing: {missing}")
    frame = raw.copy()
    frame["ClaimNb"] = frame["ClaimNb"].astype(np.float64).clip(upper=4.0)
    frame["Exposure"] = frame["Exposure"].astype(np.float64).clip(lower=1.0e-3, upper=1.0)
    frame["VehAge"] = frame["VehAge"].astype(np.float64).clip(upper=20.0)
    frame["DrivAge"] = frame["DrivAge"].astype(np.float64).clip(upper=90.0)
    frame["BonusMalus"] = frame["BonusMalus"].astype(np.float64).clip(upper=150.0)
    frame["Density"] = frame["Density"].astype(np.float64)
    frame["LogDensity"] = np.log1p(frame["Density"])
    return frame


def sample_and_split(
    frame: pd.DataFrame,
    *,
    max_rows: int | None,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take a reproducible claim-stratified sample and holdout."""
    sampled = frame
    if max_rows is not None and max_rows < len(frame):
        sampled, _unused = train_test_split(
            frame,
            train_size=max_rows,
            random_state=seed,
            stratify=frame["ClaimNb"].to_numpy() > 0.0,
        )
    train, test = train_test_split(
        sampled,
        test_size=test_size,
        random_state=seed + 1,
        stratify=sampled["ClaimNb"].to_numpy() > 0.0,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _baseline_features() -> dict[str, Any]:
    return {
        "DrivAge": Spline(kind="ps", k=8),
        "VehAge": Spline(kind="ps", k=6),
        "BonusMalus": Spline(kind="ps", k=7),
        "LogDensity": Numeric(),
        "VehPower": Categorical(),
        "VehGas": Categorical(),
        "Area": Categorical(),
    }


def make_model(
    variant: str,
    *,
    discrete: bool = True,
    n_bins: int = 256,
) -> SuperGLM:
    """Construct one member of the common-baseline comparison."""
    if variant not in DEFAULT_VARIANTS:
        raise ValueError(f"variant must be one of {DEFAULT_VARIANTS}")
    features = _baseline_features()
    interactions = []
    if variant == "brand_fixed":
        features["VehBrand"] = Categorical()
    elif variant in {"re", "re_fs"}:
        features["VehBrand"] = RandomEffect()
    if variant in {"fs", "re_fs"}:
        interactions.append(FactorSmooth("DrivAge", group="Region", k=6))
    return SuperGLM(
        family="poisson",
        features=features,
        interactions=interactions,
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=n_bins,
        direct_solve="auto" if variant in {"baseline", "brand_fixed"} else "structured",
        retain_fit_state=True,
        tol=1.0e-8,
        max_iter=100,
    )


def poisson_metrics(
    claims: np.ndarray,
    prediction: np.ndarray,
    exposure: np.ndarray,
) -> dict[str, float]:
    """Return exposure-normalized count deviance and aggregate calibration."""
    y = np.asarray(claims, dtype=np.float64)
    mu = np.asarray(prediction, dtype=np.float64)
    weight = np.asarray(exposure, dtype=np.float64)
    if y.shape != mu.shape or y.shape != weight.shape:
        raise ValueError("claims, prediction, and exposure must be aligned")
    if np.any(y < 0.0) or np.any(mu < 0.0) or np.any(weight <= 0.0):
        raise ValueError("Poisson metrics require non-negative outcomes/predictions and exposure")
    unit = np.array(mu, copy=True)
    positive = y > 0.0
    safe_mu = np.maximum(mu[positive], np.finfo(np.float64).tiny)
    unit[positive] = y[positive] * np.log(y[positive] / safe_mu) - (y[positive] - mu[positive])
    mean_deviance = float(2.0 * np.sum(unit) / np.sum(weight))
    actual_total = float(np.sum(y))
    predicted_total = float(np.sum(mu))
    calibration = predicted_total / actual_total if actual_total > 0.0 else np.nan
    return {
        "mean_poisson_deviance": mean_deviance,
        "claim_calibration": calibration,
        "actual_claims": actual_total,
        "predicted_claims": predicted_total,
    }


def fit_variants(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    variants: tuple[str, ...],
    discrete: bool,
    n_bins: int,
    max_reml_iter: int,
    reml_tol: float,
) -> tuple[dict[str, SuperGLM], pd.DataFrame]:
    """Fit all requested variants and evaluate one shared holdout."""
    models: dict[str, SuperGLM] = {}
    rows: list[dict[str, Any]] = []
    train_offset = np.log(train["Exposure"].to_numpy())
    test_offset = np.log(test["Exposure"].to_numpy())
    y_train = train["ClaimNb"].to_numpy()
    y_test = test["ClaimNb"].to_numpy()
    for variant in variants:
        model = make_model(variant, discrete=discrete, n_bins=n_bins)
        started = time.perf_counter()
        model.fit_reml(
            train.loc[:, MODEL_COLUMNS],
            y_train,
            offset=train_offset,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            pirls_tol=1.0e-8,
            max_pirls_iter=100,
            runtime_validation="skip",
        )
        fit_seconds = time.perf_counter() - started
        prediction = model.predict(
            test.loc[:, MODEL_COLUMNS],
            offset=test_offset,
        )
        metrics = poisson_metrics(
            y_test,
            prediction,
            test["Exposure"].to_numpy(),
        )
        fit_diagnostics = model.diagnostics()["_model"]
        rows.append(
            {
                "variant": variant,
                "fit_seconds": fit_seconds,
                "backend": model.result.direct_backend,
                "converged": bool(fit_diagnostics["converged"]),
                "reml_iterations": int(fit_diagnostics["n_iter"]),
                "effective_df": model.result.effective_df,
                **metrics,
            }
        )
        models[variant] = model
        print(
            f"{variant:>8}: {fit_seconds:7.2f}s, "
            f"deviance={metrics['mean_poisson_deviance']:.6f}, "
            f"A/E={metrics['claim_calibration']:.4f}, "
            f"backend={model.result.direct_backend}"
        )
    return models, pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    models: dict[str, SuperGLM],
    metrics: pd.DataFrame,
    train: pd.DataFrame,
    *,
    plot_levels: int,
    make_plot: bool,
) -> None:
    """Write tabular reports and a compact visual explanation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "heldout_metrics.csv", index=False)
    re_model = models.get("re_fs") or models.get("re")
    fs_model = models.get("re_fs") or models.get("fs")
    re_report = None
    fs_report = None
    if re_model is not None:
        re_report = re_model.random_effects(
            "VehBrand",
            exposure=train["Exposure"].to_numpy(),
        )
        re_report.table.to_csv(output_dir / "vehicle_brand_random_effect.csv", index=False)
    if fs_model is not None:
        full_report = fs_model.factor_smooth("DrivAge:Region:fs", grid=80)
        selected = full_report.table.nlargest(plot_levels, "fit_weight")["level"].tolist()
        fs_report = fs_model.factor_smooth(
            "DrivAge:Region:fs",
            grid=80,
            levels=selected,
        )
        fs_report.table.to_csv(output_dir / "driver_age_region_credibility.csv", index=False)
        fs_report.curves.to_csv(output_dir / "driver_age_region_curves.csv", index=False)
    if not make_plot:
        return

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ordered = metrics.set_index("variant").loc[
        [name for name in DEFAULT_VARIANTS if name in set(metrics["variant"])]
    ]
    baseline = float(ordered["mean_poisson_deviance"].iloc[0])
    improvement = 100.0 * (baseline - ordered["mean_poisson_deviance"]) / baseline
    axes[0].bar(ordered.index, improvement, color="#4255a4")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Held-out Poisson deviance improvement (%)")
    axes[0].set_title("Predictive value")

    if re_report is not None:
        table = re_report.table.sort_values("effect")
        points = axes[1].scatter(
            table["effect"],
            table["level"],
            c=table["credibility"],
            cmap="viridis",
            s=55,
        )
        figure.colorbar(points, ax=axes[1], label="Credibility", fraction=0.05, pad=0.04)
        axes[1].axvline(0.0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Brand log relativity")
        axes[1].set_title("Random-effect shrinkage")
    else:
        axes[1].set_axis_off()

    if fs_report is not None:
        for level, curve in fs_report.curves.groupby("level", sort=False):
            axes[2].plot(
                curve["DrivAge"],
                np.exp(curve["effect"]),
                label=str(level),
            )
        axes[2].axhline(1.0, color="black", linewidth=0.8)
        axes[2].set_xlabel("Driver age")
        axes[2].set_ylabel("Regional deviation relativity")
        axes[2].set_title("Factor-smooth deviations")
        axes[2].legend(ncol=2, fontsize=8)
    else:
        axes[2].set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / "credibility_demo.png", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("fremtpl2_credibility_results"))
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--max-reml-iter", type=int, default=12)
    parser.add_argument("--reml-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--discrete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plot-levels", type=int, default=6)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rows is not None and args.max_rows < 500:
        raise ValueError("--max-rows must be at least 500")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must lie between zero and one")
    raw = load_data(args.data)
    frame = prepare_frame(raw)
    train, test = sample_and_split(
        frame,
        max_rows=args.max_rows,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(
        f"Source rows={len(frame):,}; sampled train={len(train):,}, "
        f"test={len(test):,}; observed claims={test['ClaimNb'].sum():.0f}"
    )
    models, metrics = fit_variants(
        train,
        test,
        variants=tuple(args.variants),
        discrete=args.discrete,
        n_bins=args.n_bins,
        max_reml_iter=args.max_reml_iter,
        reml_tol=args.reml_tol,
    )
    write_outputs(
        args.output_dir,
        models,
        metrics,
        train,
        plot_levels=args.plot_levels,
        make_plot=args.plot,
    )
    print(f"Wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
