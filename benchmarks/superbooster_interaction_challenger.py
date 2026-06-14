"""Benchmark SuperGLM challengers using GBM-discovered interaction rankings.

Workflow:
1. fit a main-effects SuperGLM backbone
2. fit a SuperBooster-style XGBoost correction model on top
3. estimate parent-term pair interaction strengths from XGBoost
4. refit cumulative SuperGLM challengers using the top-ranked interactions
5. compare holdout performance against the backbone, hybrid, and pure XGBoost
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.validation import lorenz_curve

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "freMTPL2freq.parquet"
if not DATA_PATH.exists() and ROOT.parent.name == ".worktrees":
    DATA_PATH = ROOT.parent.parent / "data" / "freMTPL2freq.parquet"
OUT_DIR = ROOT / "benchmarks" / "results"
OUT_JSON = OUT_DIR / "superbooster_interaction_challenger.json"

INTERACTION_SAMPLE_ROWS = 30_000
TOP_K_MODELS = (1, 3)


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, correction: float = 1e-10) -> float:
    """Mean Poisson deviance on the count scale."""
    yt = y_true.astype(float) + correction
    yp = y_pred.astype(float) + correction
    return float(2.0 * np.mean(yp - yt - yt * np.log(yp / yt)))


def load_freq() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load and clip the local freMTPL2 frequency data."""
    df = pd.read_parquet(DATA_PATH)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4).astype(float)
    df["Exposure"] = df["Exposure"].clip(lower=0.01).astype(float)
    df["DrivAge"] = df["DrivAge"].clip(18, 90).astype(float)
    df["VehAge"] = df["VehAge"].clip(0, 20).astype(float)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150).astype(float)
    X = df[["DrivAge", "VehAge", "BonusMalus", "Area"]].copy()
    y_count = df["ClaimNb"].to_numpy(dtype=float)
    exposure = df["Exposure"].to_numpy(dtype=float)
    offset = np.log(exposure)
    return X, y_count, exposure, offset


def split_data(
    X: pd.DataFrame,
    y_count: np.ndarray,
    exposure: np.ndarray,
    offset: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a 60/20/20 split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = len(X)
    tr_end = int(0.6 * n)
    va_end = int(0.8 * n)
    tr, va, te = idx[:tr_end], idx[tr_end:va_end], idx[va_end:]

    def pack(rows: np.ndarray) -> dict[str, Any]:
        return {
            "X": X.iloc[rows].reset_index(drop=True),
            "y_count": y_count[rows],
            "exposure": exposure[rows],
            "offset": offset[rows],
        }

    return {"train": pack(tr), "valid": pack(va), "test": pack(te)}


def build_features() -> dict[str, object]:
    """Main-effect feature set."""
    return {
        "DrivAge": Spline(kind="cr", k=20, penalty="ssp", discrete=True),
        "VehAge": Spline(kind="cr", k=15, penalty="ssp", discrete=True),
        "BonusMalus": Spline(kind="cr", k=15, penalty="ssp", discrete=True),
        "Area": Categorical(base="most_exposed"),
    }


def build_booster_frame(X: pd.DataFrame, area_levels: list[str]) -> pd.DataFrame:
    """Raw-term booster matrix."""
    area_cols = [f"Area__{level}" for level in area_levels]
    area = pd.Categorical(X["Area"].astype(str), categories=area_levels)
    area_dummies = pd.get_dummies(area, prefix="Area", prefix_sep="__", dtype=float)
    area_dummies = area_dummies.reindex(columns=area_cols, fill_value=0.0)
    return pd.concat(
        [
            X[["DrivAge", "VehAge", "BonusMalus"]].reset_index(drop=True),
            area_dummies.reset_index(drop=True),
        ],
        axis=1,
    )


def fit_superglm(
    X: pd.DataFrame,
    y_count: np.ndarray,
    offset: np.ndarray,
    *,
    interactions: list[tuple[str, str]] | None = None,
    max_reml_iter: int = 20,
) -> SuperGLM:
    """Fit one Poisson SuperGLM on counts + log offset."""
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=256,
        features=build_features(),
        interactions=interactions,
        direct_solve="auto",
    )
    model.fit_reml(X, y_count, offset=offset, max_reml_iter=max_reml_iter)
    return model


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    base_margin_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    base_margin_valid: np.ndarray,
) -> xgb.Booster:
    """Fit one XGBoost Poisson model with early stopping."""
    return xgb.train(
        {
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "tree_method": "hist",
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 50,
            "base_score": 0.0,
            "nthread": 4,
            "seed": 42,
        },
        xgb.DMatrix(
            X_train.to_numpy(dtype=np.float32),
            label=y_train.astype(np.float32),
            base_margin=base_margin_train.astype(np.float32),
        ),
        num_boost_round=400,
        evals=[
            (
                xgb.DMatrix(
                    X_valid.to_numpy(dtype=np.float32),
                    label=y_valid.astype(np.float32),
                    base_margin=base_margin_valid.astype(np.float32),
                ),
                "valid",
            )
        ],
        early_stopping_rounds=20,
        verbose_eval=False,
    )


def evaluate_count_model(
    name: str,
    mu_count: np.ndarray,
    y_count: np.ndarray,
    exposure: np.ndarray,
) -> dict[str, float | str]:
    """Holdout metrics on count and rate views."""
    mu_rate = mu_count / exposure
    y_rate = y_count / exposure
    lorenz = lorenz_curve(y_rate, mu_rate, exposure=exposure)
    return {
        "model": name,
        "poisson_dev_count": poisson_deviance(y_count, mu_count),
        "mean_count_pred": float(np.mean(mu_count)),
        "gini_model": float(lorenz.gini_model),
        "gini_ratio": float(lorenz.gini_ratio),
    }


def extract_reml_debug(model: SuperGLM) -> dict[str, Any]:
    """Extract bootstrap + REML diagnostics from a fitted SuperGLM."""
    reml_result = model._reml_result
    profile = model._reml_profile or {}
    bootstrap_summary = profile.get("reml_bootstrap_summary")
    bootstrap_components = profile.get("reml_bootstrap_components") or []
    smallest_components = sorted(
        bootstrap_components,
        key=lambda row: row["lam_fp_clipped"],
    )[:5]
    return {
        "converged": bool(reml_result.converged),
        "n_reml_iter": int(reml_result.n_reml_iter),
        "objective": None if reml_result.objective is None else float(reml_result.objective),
        "effective_df": float(model.result.effective_df),
        "final_lambdas": {k: float(v) for k, v in reml_result.lambdas.items()},
        "bootstrap_summary": bootstrap_summary,
        "bootstrap_smallest_components": smallest_components,
        "lambda_history_head": [
            {k: float(v) for k, v in row.items()} for row in reml_result.lambda_history[:3]
        ],
        "lambda_history_tail": [
            {k: float(v) for k, v in row.items()} for row in reml_result.lambda_history[-3:]
        ],
    }


def rank_parent_interactions(
    booster: xgb.Booster,
    X_valid_boost: pd.DataFrame,
    *,
    sample_rows: int,
) -> list[dict[str, float | str]]:
    """Rank parent-term pair interactions from XGBoost interaction SHAP values."""
    rng = np.random.default_rng(7)
    n = len(X_valid_boost)
    take = min(sample_rows, n)
    idx = rng.choice(n, size=take, replace=False)
    X_sample = X_valid_boost.iloc[idx].reset_index(drop=True)

    interaction_tensor = booster.predict(
        xgb.DMatrix(
            X_sample.to_numpy(dtype=np.float32),
            base_margin=np.zeros(len(X_sample), dtype=np.float32),
        ),
        pred_interactions=True,
    )

    term_slices = {
        "DrivAge": [0],
        "VehAge": [1],
        "BonusMalus": [2],
        "Area": list(range(3, X_valid_boost.shape[1])),
    }
    parent_terms = list(term_slices)
    rows: list[dict[str, float | str]] = []

    for i, left in enumerate(parent_terms):
        for right in parent_terms[i + 1 :]:
            block = interaction_tensor[
                :,
                np.ix_(term_slices[left], term_slices[right])[0],
                np.ix_(term_slices[left], term_slices[right])[1],
            ]
            # Collapse the feature-block interaction to one value per row.
            block_sum = np.asarray(block.sum(axis=(1, 2)), dtype=float)
            rows.append(
                {
                    "pair": f"{left}:{right}",
                    "left": left,
                    "right": right,
                    "mean_abs_interaction": float(np.mean(np.abs(block_sum))),
                }
            )

    rows.sort(key=lambda r: r["mean_abs_interaction"], reverse=True)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y_count, exposure, offset = load_freq()
    split = split_data(X, y_count, exposure, offset)
    train = split["train"]
    valid = split["valid"]
    test = split["test"]

    main_model = fit_superglm(train["X"], train["y_count"], train["offset"])
    interaction_seed_model = fit_superglm(
        train["X"],
        train["y_count"],
        train["offset"],
        interactions=[("DrivAge", "VehAge")],
        max_reml_iter=15,
    )

    area_levels = sorted(pd.Series(train["X"]["Area"]).astype(str).unique().tolist())
    Xb_train = build_booster_frame(train["X"], area_levels)
    Xb_valid = build_booster_frame(valid["X"], area_levels)
    Xb_test = build_booster_frame(test["X"], area_levels)

    eta_train = main_model._predict_eta_exact(train["X"], train["offset"])
    eta_valid = main_model._predict_eta_exact(valid["X"], valid["offset"])
    eta_test = main_model._predict_eta_exact(test["X"], test["offset"])

    hybrid_booster = fit_xgb(
        Xb_train,
        train["y_count"],
        eta_train,
        Xb_valid,
        valid["y_count"],
        eta_valid,
    )
    pure_booster = fit_xgb(
        Xb_train,
        train["y_count"],
        train["offset"],
        Xb_valid,
        valid["y_count"],
        valid["offset"],
    )

    interaction_ranking = rank_parent_interactions(
        hybrid_booster,
        Xb_valid,
        sample_rows=INTERACTION_SAMPLE_ROWS,
    )
    ranked_pairs = [(row["left"], row["right"]) for row in interaction_ranking]

    results: list[dict[str, float | str]] = []
    diagnostics: dict[str, Any] = {
        "superglm_main": extract_reml_debug(main_model),
        "superglm_seed_interaction": extract_reml_debug(interaction_seed_model),
    }
    results.append(
        evaluate_count_model(
            "superglm_main",
            main_model.predict(test["X"], offset=test["offset"]),
            test["y_count"],
            test["exposure"],
        )
    )
    results.append(
        evaluate_count_model(
            "superglm_seed_interaction",
            interaction_seed_model.predict(test["X"], offset=test["offset"]),
            test["y_count"],
            test["exposure"],
        )
    )

    for k in TOP_K_MODELS:
        chosen = ranked_pairs[:k]
        challenger = fit_superglm(
            train["X"],
            train["y_count"],
            train["offset"],
            interactions=chosen,
            max_reml_iter=15,
        )
        diagnostics[f"superglm_top{k}_gbm_interactions"] = {
            "chosen_pairs": [f"{a}:{b}" for a, b in chosen],
            **extract_reml_debug(challenger),
        }
        results.append(
            evaluate_count_model(
                f"superglm_top{k}_gbm_interactions",
                challenger.predict(test["X"], offset=test["offset"]),
                test["y_count"],
                test["exposure"],
            )
        )

    eta_hybrid = hybrid_booster.predict(
        xgb.DMatrix(
            Xb_test.to_numpy(dtype=np.float32),
            base_margin=eta_test.astype(np.float32),
        ),
        output_margin=True,
    )
    results.append(
        evaluate_count_model(
            "hybrid",
            np.exp(eta_hybrid),
            test["y_count"],
            test["exposure"],
        )
    )

    eta_pure = pure_booster.predict(
        xgb.DMatrix(
            Xb_test.to_numpy(dtype=np.float32),
            base_margin=test["offset"].astype(np.float32),
        ),
        output_margin=True,
    )
    results.append(
        evaluate_count_model(
            "xgboost_pure",
            np.exp(eta_pure),
            test["y_count"],
            test["exposure"],
        )
    )

    results.sort(key=lambda row: row["poisson_dev_count"])

    payload = {
        "dataset": "freMTPL2freq",
        "split": {
            "n_train": int(len(train["X"])),
            "n_valid": int(len(valid["X"])),
            "n_test": int(len(test["X"])),
        },
        "backbone": {
            "effective_df": float(main_model.result.effective_df),
        },
        "seed_interaction_model": {
            "effective_df": float(interaction_seed_model.result.effective_df),
        },
        "interaction_ranking_sample_rows": INTERACTION_SAMPLE_ROWS,
        "interaction_ranking": interaction_ranking,
        "hybrid_boost_rounds": int(hybrid_booster.best_iteration + 1),
        "pure_boost_rounds": int(pure_booster.best_iteration + 1),
        "diagnostics": diagnostics,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print("SuperBooster interaction challenger benchmark")
    print("=" * 72)
    for row in interaction_ranking:
        print(f"{row['pair']:<24s} mean|interaction|={row['mean_abs_interaction']:.6f}")
    print("-" * 72)
    for row in results:
        print(
            f"{row['model']:<32s} "
            f"dev={row['poisson_dev_count']:.6f} "
            f"gini={row['gini_model']:.6f} "
            f"gini_ratio={row['gini_ratio']:.6f}"
        )
    print(f"Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
