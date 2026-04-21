"""Benchmark-local SCOP lambda sensitivity experiment harness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from superglm import Constraint, LambdaPolicy, PSpline, SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import CubicRegressionSpline

RESULTS_DIR = Path("benchmarks/results/scop_lambda_sensitivity")
_LAMBDA_GRID_FACTORS = np.array(
    [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SensitivityScenario:
    name: str
    discrete: bool
    n_constrained: int


SCENARIOS = {
    "single_scop_exact": SensitivityScenario("single_scop_exact", discrete=False, n_constrained=1),
    "single_scop_discrete": SensitivityScenario(
        "single_scop_discrete", discrete=True, n_constrained=1
    ),
    "multi_scop_exact": SensitivityScenario("multi_scop_exact", discrete=False, n_constrained=2),
    "multi_scop_discrete": SensitivityScenario(
        "multi_scop_discrete", discrete=True, n_constrained=2
    ),
}


def build_lambda_grid(baseline_lambda: float) -> np.ndarray:
    """Build a log-symmetric lambda sweep around a fitted baseline."""
    baseline = float(baseline_lambda)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("baseline_lambda must be positive and finite")
    return baseline * _LAMBDA_GRID_FACTORS


def _as_1d_float_array(name: str, values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integration_weights(x: np.ndarray) -> np.ndarray:
    if x.size == 1:
        return np.ones(1, dtype=np.float64)

    spacing = np.diff(x)
    if np.any(spacing <= 0.0):
        raise ValueError("x must be strictly increasing")

    weights = np.empty_like(x)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights / np.sum(weights)


def curve_similarity_metrics(
    x: np.ndarray | list[float],
    reference_curve: np.ndarray | list[float],
    other_curve: np.ndarray | list[float],
) -> dict[str, float]:
    """Compare two evaluated curves on a shared x-grid."""
    x_values = _as_1d_float_array("x", x)
    reference = _as_1d_float_array("reference_curve", reference_curve)
    other = _as_1d_float_array("other_curve", other_curve)

    if x_values.shape != reference.shape or reference.shape != other.shape:
        raise ValueError("x, reference_curve, and other_curve must have the same shape")

    weights = _integration_weights(x_values)
    diff = other - reference
    mse = float(np.average(diff**2, weights=weights))
    max_abs_diff = float(np.max(np.abs(diff)))

    if mse == 0.0:
        r2 = 1.0
    else:
        centered_reference = reference - np.average(reference, weights=weights)
        baseline_mse = float(np.average(centered_reference**2, weights=weights))
        r2 = 0.0 if baseline_mse <= np.finfo(np.float64).eps else 1.0 - mse / baseline_mse

    return {
        "rmse": float(np.sqrt(mse)),
        "max_abs_diff": max_abs_diff,
        "r2": float(r2),
    }


def summarize_result_rows(rows: list[dict]) -> pd.DataFrame:
    columns = ["scenario", "comparison", "target", "r2", "max_abs_diff", "rmse"]
    if not rows:
        return pd.DataFrame(columns=columns)

    field_order = columns.copy()
    seen = set(field_order)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                field_order.append(key)
    return pd.DataFrame(rows)[field_order]


def find_fremtpl_freq_path() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    candidate = root / "data" / "freMTPL2freq.parquet"
    if candidate.exists():
        return candidate
    if root.parent.name == ".worktrees":
        candidate = root.parent.parent / "data" / "freMTPL2freq.parquet"
        if candidate.exists():
            return candidate
    return None


def load_fremtpl_freq_dataset(n: int | None = None, *, seed: int = 42):
    data_path = find_fremtpl_freq_path()
    if data_path is None:
        raise FileNotFoundError("freMTPL2freq.parquet not found")

    df = pd.read_parquet(data_path)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["DrivAge"] = df["DrivAge"].clip(18, 90)
    df["VehAge"] = df["VehAge"].clip(0, 20)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150)
    df["LogDensity"] = np.log(df["Density"].clip(lower=1.0))

    if n is not None and n < len(df):
        rng = np.random.default_rng(seed)
        take = rng.permutation(len(df))[:n]
        df = df.iloc[take].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    y = (df["ClaimNb"] / df["Exposure"]).to_numpy(dtype=float)
    w = df["Exposure"].to_numpy(dtype=float)
    X = df[["DrivAge", "VehAge", "BonusMalus", "LogDensity", "Area"]].copy()
    return X, y, w


def split_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(idx))
    tr, te = idx[:n_train], idx[n_train:]
    return (
        X.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        y[tr],
        y[te],
        w[tr],
        w[te],
    )


def _constrained_feature_name(index: int) -> str:
    return "DrivAge" if index == 0 else "BonusMalus"


def build_features(
    scenario: SensitivityScenario,
    *,
    constrained: bool,
    fixed_lambdas: dict[str, float] | None = None,
) -> dict[str, object]:
    features: dict[str, object] = {
        "DrivAge": CubicRegressionSpline(n_knots=12),
        "VehAge": CubicRegressionSpline(n_knots=10),
        "BonusMalus": CubicRegressionSpline(n_knots=12),
        "LogDensity": CubicRegressionSpline(n_knots=10),
        "Area": Categorical(base="most_exposed"),
    }

    for i in range(scenario.n_constrained):
        name = _constrained_feature_name(i)
        if constrained:
            policy = (
                LambdaPolicy.fixed(fixed_lambdas[name])
                if fixed_lambdas is not None and name in fixed_lambdas
                else None
            )
            features[name] = PSpline(
                n_knots=12,
                penalty="ssp",
                constraint=Constraint.fit.concave,
                lambda_policy=policy,
            )
        else:
            features[name] = PSpline(n_knots=12, penalty="ssp")
    return features


def fit_model(
    scenario: SensitivityScenario,
    *,
    constrained: bool,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    fixed_lambdas: dict[str, float] | None = None,
) -> SuperGLM:
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=scenario.discrete,
        features=build_features(scenario, constrained=constrained, fixed_lambdas=fixed_lambdas),
    )
    model.fit_reml(X_train, y_train, sample_weight=w_train, max_reml_iter=20)
    return model


def extract_constrained_curve(model: SuperGLM, feature_name: str) -> tuple[np.ndarray, np.ndarray]:
    curve = model.reconstruct_feature(feature_name)
    return np.asarray(curve["x"], dtype=float), np.asarray(curve["log_relativity"], dtype=float)


def prediction_similarity(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    idx = np.arange(len(reference), dtype=float)
    return curve_similarity_metrics(idx, reference, other)


def compare_models(
    scenario: SensitivityScenario,
    comparison: str,
    reference: SuperGLM,
    other: SuperGLM,
    *,
    X_eval: pd.DataFrame,
) -> list[dict]:
    rows = []

    for i in range(scenario.n_constrained):
        name = _constrained_feature_name(i)
        x_ref, curve_ref = extract_constrained_curve(reference, name)
        x_other, curve_other = extract_constrained_curve(other, name)
        if not np.allclose(x_ref, x_other):
            raise ValueError(f"Curve grids do not match for feature {name}")
        metrics = curve_similarity_metrics(x_ref, curve_ref, curve_other)
        rows.append(
            {
                "scenario": scenario.name,
                "comparison": comparison,
                "target": "curve",
                "feature": name,
                **metrics,
            }
        )

    pred_ref = np.asarray(reference.predict(X_eval), dtype=float)
    pred_other = np.asarray(other.predict(X_eval), dtype=float)
    rows.append(
        {
            "scenario": scenario.name,
            "comparison": comparison,
            "target": "prediction",
            **prediction_similarity(pred_ref, pred_other),
        }
    )
    return rows


def save_curve_plot(
    scenario: SensitivityScenario,
    feature_name: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for label, (x, y) in curves.items():
        ax.plot(x, y, label=label)
    ax.set_title(f"{scenario.name}: {feature_name} link-scale curve")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("log_relativity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prediction_scatter(
    scenario: SensitivityScenario,
    label: str,
    reference: np.ndarray,
    other: np.ndarray,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(reference, other, s=6, alpha=0.35)
    lo = min(reference.min(), other.min())
    hi = max(reference.max(), other.max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_title(f"{scenario.name}: {label}")
    ax.set_xlabel("reference prediction")
    ax.set_ylabel("comparison prediction")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def passthrough_lambdas_from_unconstrained(
    scenario: SensitivityScenario,
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
) -> dict[str, float]:
    unconstrained = fit_model(
        scenario,
        constrained=False,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
    )
    return {name: float(unconstrained._reml_lambdas[name]) for name in unconstrained._reml_lambdas}


def run_scenario(
    scenario: SensitivityScenario,
    *,
    grid_limit: int | None,
    sample_n: int,
    seed: int,
) -> tuple[pd.DataFrame, list[Path]]:
    print(
        f"running {scenario.name} "
        f"(mode={'discrete' if scenario.discrete else 'exact'}, n_constrained={scenario.n_constrained})"
    )
    X, y, w = load_fremtpl_freq_dataset(sample_n, seed=seed)
    X_train, X_eval, y_train, _y_eval, w_train, _w_eval = split_dataset(X, y, w, seed=seed)

    integrated = fit_model(
        scenario,
        constrained=True,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
    )
    baseline_lambdas = {
        _constrained_feature_name(i): float(integrated._reml_lambdas[_constrained_feature_name(i)])
        for i in range(scenario.n_constrained)
    }
    print(f"  integrated lambdas: {baseline_lambdas}")

    passthrough_lambdas = passthrough_lambdas_from_unconstrained(
        scenario,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
    )
    constrained_passthrough = fit_model(
        scenario,
        constrained=True,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        fixed_lambdas=passthrough_lambdas,
    )
    print(f"  passthrough lambdas: {passthrough_lambdas}")

    rows = compare_models(
        scenario, "passthrough", integrated, constrained_passthrough, X_eval=X_eval
    )
    artifacts: list[Path] = []

    for i in range(scenario.n_constrained):
        name = _constrained_feature_name(i)
        artifacts.append(
            save_curve_plot(
                scenario,
                name,
                {
                    "integrated": extract_constrained_curve(integrated, name),
                    "passthrough": extract_constrained_curve(constrained_passthrough, name),
                },
                RESULTS_DIR / f"{scenario.name}_{name}_passthrough_curve.png",
            )
        )

    artifacts.append(
        save_prediction_scatter(
            scenario,
            "integrated_vs_passthrough",
            np.asarray(integrated.predict(X_eval), dtype=float),
            np.asarray(constrained_passthrough.predict(X_eval), dtype=float),
            RESULTS_DIR / f"{scenario.name}_passthrough_prediction.png",
        )
    )

    factors = _LAMBDA_GRID_FACTORS if grid_limit is None else _LAMBDA_GRID_FACTORS[:grid_limit]
    for factor in factors:
        fixed = {name: baseline_lambdas[name] * float(factor) for name in baseline_lambdas}
        compare = fit_model(
            scenario,
            constrained=True,
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            fixed_lambdas=fixed,
        )
        rows.extend(
            compare_models(
                scenario,
                f"fixed_x{factor:g}",
                integrated,
                compare,
                X_eval=X_eval,
            )
        )

    return summarize_result_rows(rows), artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--grid-limit", type=int, default=None)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary, artifacts = run_scenario(
        scenario,
        grid_limit=args.grid_limit,
        sample_n=args.n,
        seed=args.seed,
    )
    output_csv = RESULTS_DIR / f"{scenario.name}_summary.csv"
    summary.to_csv(output_csv, index=False)
    print(f"wrote {output_csv}")
    for path in artifacts:
        print(f"wrote {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
