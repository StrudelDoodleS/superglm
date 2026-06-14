"""Prototype benchmark for SuperBooster-style corrected-term extraction.

This is a full-dataset spike on the local freMTPL2 frequency parquet.

It fits:
1. a Poisson SuperGLM backbone on claim counts with log(Exposure) offset
2. an XGBoost booster with backbone eta passed through ``base_margin``

It then compares two explainability paths over the *full* dataset:

- full method:
  compute XGBoost ``pred_contribs`` for every row
- compressed method:
  group rows by exact XGBoost leaf signature, compute SHAP once per unique
  signature, then broadcast back

The compressed path is exact if equal leaf signatures imply equal TreeSHAP
values, which should hold for XGBoost tree ensembles.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.model import base as model_base

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "freMTPL2freq.parquet"
if not DATA_PATH.exists() and ROOT.parent.name == ".worktrees":
    DATA_PATH = ROOT.parent.parent / "data" / "freMTPL2freq.parquet"
OUT_DIR = ROOT / "benchmarks" / "results"
OUT_JSON = OUT_DIR / "superbooster_shap_compression_spike.json"


def timer(fn, *args, **kwargs):
    """Run one callable and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def load_freq() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the full local MTPL frequency dataset."""
    df = pd.read_parquet(DATA_PATH)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4).astype(float)
    df["Exposure"] = df["Exposure"].clip(lower=0.01).astype(float)
    df["DrivAge"] = df["DrivAge"].clip(18, 90).astype(float)
    df["VehAge"] = df["VehAge"].clip(0, 20).astype(float)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150).astype(float)
    X = df[["DrivAge", "VehAge", "BonusMalus", "Area"]].copy()
    y = df["ClaimNb"].to_numpy(dtype=float)
    offset = np.log(df["Exposure"].to_numpy(dtype=float))
    return X, y, offset


def build_features() -> dict[str, object]:
    """Feature set for the SuperGLM backbone."""
    return {
        "DrivAge": Spline(kind="cr", k=20, penalty="ssp", discrete=True),
        "VehAge": Spline(kind="cr", k=15, penalty="ssp", discrete=True),
        "BonusMalus": Spline(kind="cr", k=15, penalty="ssp", discrete=True),
        "Area": Categorical(base="most_exposed"),
    }


def build_booster_frame(X: pd.DataFrame, area_levels: list[str]) -> pd.DataFrame:
    """Raw-term booster feature matrix."""
    area_cols = [f"Area__{level}" for level in area_levels]
    area = pd.Categorical(X["Area"].astype(str), categories=area_levels)
    area_dummies = pd.get_dummies(area, prefix="Area", prefix_sep="__", dtype=float)
    area_dummies = area_dummies.reindex(columns=area_cols, fill_value=0.0)
    out = pd.concat(
        [
            X[["DrivAge", "VehAge", "BonusMalus"]].reset_index(drop=True).astype(float),
            area_dummies.reset_index(drop=True).astype(float),
        ],
        axis=1,
    )
    return out[["DrivAge", "VehAge", "BonusMalus", *area_cols]]


def backbone_terms(model: SuperGLM, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Exact backbone term contributions on the link scale."""
    plan = model_base._prediction_plan(model)
    beta_all = model.result.beta
    out: dict[str, np.ndarray] = {}
    for term in plan["features"]:
        values = np.asarray(X[term["name"]])
        beta = beta_all[term["beta_idx"]]
        out[term["name"]] = np.asarray(
            model_base._score_feature(term["spec"], values, beta),
            dtype=float,
        ).ravel()
    return out


def aggregate_shap(
    shap_matrix: np.ndarray,
    term_slices: dict[str, list[int]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Aggregate booster SHAP columns back to parent terms plus bias."""
    bias = shap_matrix[:, -1]
    out: dict[str, np.ndarray] = {}
    for term_name, cols in term_slices.items():
        out[term_name] = shap_matrix[:, cols].sum(axis=1)
    return out, bias


def reconstruct_eta(
    model: SuperGLM,
    offset: np.ndarray,
    backbone_term_map: dict[str, np.ndarray],
    shap_term_map: dict[str, np.ndarray],
    shap_bias: np.ndarray,
) -> np.ndarray:
    """Rebuild the hybrid eta from backbone terms plus booster corrections."""
    eta = np.full(len(offset), model.result.intercept, dtype=float)
    eta += shap_bias
    eta += offset
    for term_name in model._feature_order:
        eta += backbone_term_map[term_name]
        eta += shap_term_map[term_name]
    return eta


def unique_leaf_groups(leaf_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (representative_row_indices, inverse_group_index) for leaf signatures."""
    leaf_contig = np.ascontiguousarray(leaf_matrix)
    row_view = leaf_contig.view(
        np.dtype((np.void, leaf_contig.dtype.itemsize * leaf_contig.shape[1]))
    ).ravel()
    _, rep_idx, inverse = np.unique(row_view, return_index=True, return_inverse=True)
    return rep_idx.astype(np.intp, copy=False), inverse.astype(np.intp, copy=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, offset = load_freq()
    n_rows = len(X)
    print(f"dataset_rows={n_rows:,}")

    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=256,
        features=build_features(),
    )
    _, t_backbone_fit = timer(model.fit_reml, X, y, offset=offset, max_reml_iter=20)
    print(
        f"backbone_fit_s={t_backbone_fit:.3f} "
        f"edf={model.result.effective_df:.2f} "
        f"reml_iter={model._reml_result.n_reml_iter}"
    )

    eta_backbone, t_eta = timer(model._predict_eta_exact, X, offset)
    backbone_term_map, t_terms = timer(backbone_terms, model, X)

    area_levels = sorted(pd.Series(X["Area"]).astype(str).unique().tolist())
    Xb, t_bfeat = timer(build_booster_frame, X, area_levels)
    area_cols = [f"Area__{level}" for level in area_levels]
    term_slices = {
        "DrivAge": [0],
        "VehAge": [1],
        "BonusMalus": [2],
        "Area": list(range(3, 3 + len(area_cols))),
    }

    dtrain = xgb.DMatrix(
        Xb.to_numpy(dtype=np.float32),
        label=y.astype(np.float32),
        base_margin=eta_backbone.astype(np.float32),
    )
    params = {
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
    }
    booster, t_booster_fit = timer(
        xgb.train,
        params,
        dtrain,
        num_boost_round=128,
        verbose_eval=False,
    )
    print(f"booster_fit_s={t_booster_fit:.3f} booster_rounds={booster.num_boosted_rounds()}")

    pred_margin_dm = xgb.DMatrix(
        Xb.to_numpy(dtype=np.float32),
        base_margin=eta_backbone.astype(np.float32),
    )
    eta_total, t_boost_pred = timer(booster.predict, pred_margin_dm, output_margin=True)

    zero_margin = np.zeros(n_rows, dtype=np.float32)

    # --- full method ---
    full_dm = xgb.DMatrix(Xb.to_numpy(dtype=np.float32), base_margin=zero_margin)
    shap_full, t_full_shap = timer(booster.predict, full_dm, pred_contribs=True)
    (shap_terms_full, shap_bias_full), t_full_agg = timer(aggregate_shap, shap_full, term_slices)
    eta_recon_full, t_full_recon = timer(
        reconstruct_eta,
        model,
        offset,
        backbone_term_map,
        shap_terms_full,
        shap_bias_full,
    )
    full_recon_max_err = float(np.max(np.abs(eta_total - eta_recon_full)))
    full_recon_mean_err = float(np.mean(np.abs(eta_total - eta_recon_full)))
    total_full_explain = t_bfeat + t_eta + t_terms + t_full_shap + t_full_agg + t_full_recon

    # --- compressed method ---
    leaf_dm = xgb.DMatrix(Xb.to_numpy(dtype=np.float32), base_margin=zero_margin)
    leaf_matrix, t_pred_leaf = timer(booster.predict, leaf_dm, pred_leaf=True)
    leaf_matrix = np.asarray(leaf_matrix, dtype=np.int32)
    (rep_idx, inverse), t_unique = timer(unique_leaf_groups, leaf_matrix)

    rep_dm = xgb.DMatrix(
        Xb.iloc[rep_idx].to_numpy(dtype=np.float32),
        base_margin=np.zeros(len(rep_idx), dtype=np.float32),
    )
    shap_rep, t_rep_shap = timer(booster.predict, rep_dm, pred_contribs=True)

    def broadcast_rep_shap(shap_rows: np.ndarray, inv: np.ndarray) -> np.ndarray:
        return shap_rows[inv]

    shap_compressed, t_broadcast = timer(broadcast_rep_shap, shap_rep, inverse)
    (shap_terms_compressed, shap_bias_compressed), t_comp_agg = timer(
        aggregate_shap,
        shap_compressed,
        term_slices,
    )
    eta_recon_compressed, t_comp_recon = timer(
        reconstruct_eta,
        model,
        offset,
        backbone_term_map,
        shap_terms_compressed,
        shap_bias_compressed,
    )
    comp_recon_max_err = float(np.max(np.abs(eta_total - eta_recon_compressed)))
    comp_recon_mean_err = float(np.mean(np.abs(eta_total - eta_recon_compressed)))
    total_compressed_explain = (
        t_bfeat
        + t_eta
        + t_terms
        + t_pred_leaf
        + t_unique
        + t_rep_shap
        + t_broadcast
        + t_comp_agg
        + t_comp_recon
    )

    shap_bias_diff = float(np.max(np.abs(shap_bias_full - shap_bias_compressed)))
    term_diffs = {
        term_name: float(
            np.max(np.abs(shap_terms_full[term_name] - shap_terms_compressed[term_name]))
        )
        for term_name in model._feature_order
    }

    n_unique = int(len(rep_idx))
    compression_ratio = float(n_rows / n_unique)

    results = {
        "dataset": "freMTPL2freq",
        "n_rows": n_rows,
        "target": "ClaimNb",
        "offset": "log(Exposure)",
        "backbone": {
            "family": "poisson",
            "discrete": True,
            "n_bins": 256,
            "fit_reml_time_s": t_backbone_fit,
            "effective_df": float(model.result.effective_df),
            "n_reml_iter": int(model._reml_result.n_reml_iter),
        },
        "booster": {
            "backend": "xgboost",
            "objective": "count:poisson",
            "num_boost_round": int(booster.num_boosted_rounds()),
            "fit_time_s": t_booster_fit,
            "predict_margin_time_s": t_boost_pred,
        },
        "shared_explain_setup": {
            "build_booster_features_s": t_bfeat,
            "backbone_eta_s": t_eta,
            "backbone_terms_s": t_terms,
            "booster_feature_columns": int(Xb.shape[1]),
            "term_names": list(model._feature_order),
        },
        "full_method": {
            "shap_time_s": t_full_shap,
            "aggregate_time_s": t_full_agg,
            "reconstruct_time_s": t_full_recon,
            "total_explain_time_s": total_full_explain,
            "rows_per_second": n_rows / total_full_explain,
            "shap_matrix_mb": shap_full.nbytes / 1e6,
            "recon_max_abs_err": full_recon_max_err,
            "recon_mean_abs_err": full_recon_mean_err,
        },
        "compressed_method": {
            "pred_leaf_time_s": t_pred_leaf,
            "unique_grouping_time_s": t_unique,
            "representative_shap_time_s": t_rep_shap,
            "broadcast_time_s": t_broadcast,
            "aggregate_time_s": t_comp_agg,
            "reconstruct_time_s": t_comp_recon,
            "total_explain_time_s": total_compressed_explain,
            "rows_per_second": n_rows / total_compressed_explain,
            "n_unique_leaf_signatures": n_unique,
            "compression_ratio": compression_ratio,
            "representative_shap_matrix_mb": shap_rep.nbytes / 1e6,
            "broadcast_shap_matrix_mb": shap_compressed.nbytes / 1e6,
            "recon_max_abs_err": comp_recon_max_err,
            "recon_mean_abs_err": comp_recon_mean_err,
            "max_bias_diff_vs_full": shap_bias_diff,
            "max_term_diff_vs_full": term_diffs,
        },
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))

    print("SuperBooster SHAP compression spike")
    print("=" * 72)
    print(f"shared: build_booster={t_bfeat:.3f}s eta={t_eta:.3f}s backbone_terms={t_terms:.3f}s")
    print(
        f"full: shap={t_full_shap:.3f}s agg={t_full_agg:.3f}s recon={t_full_recon:.3f}s "
        f"total={total_full_explain:.3f}s rows_per_s={n_rows / total_full_explain:.1f}"
    )
    print(
        f"compressed: pred_leaf={t_pred_leaf:.3f}s unique={t_unique:.3f}s "
        f"rep_shap={t_rep_shap:.3f}s broadcast={t_broadcast:.3f}s agg={t_comp_agg:.3f}s "
        f"recon={t_comp_recon:.3f}s total={total_compressed_explain:.3f}s "
        f"rows_per_s={n_rows / total_compressed_explain:.1f}"
    )
    print(
        f"leaf_signatures={n_unique:,} compression_ratio={compression_ratio:.2f}x "
        f"full_recon_max_err={full_recon_max_err:.3e} "
        f"compressed_recon_max_err={comp_recon_max_err:.3e}"
    )
    print(f"Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
