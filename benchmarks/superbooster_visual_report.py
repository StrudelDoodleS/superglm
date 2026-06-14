"""Generate a local visual report for the SuperBooster prototype.

The report includes:
- backbone main-effect explorer (Plotly)
- a SuperGLM interaction contour view
- booster SHAP-style diagnostics

Outputs a standalone HTML report under ``benchmarks/results``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import xgboost as xgb
from plotly.subplots import make_subplots

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.model import base as model_base

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "freMTPL2freq.parquet"
if not DATA_PATH.exists() and ROOT.parent.name == ".worktrees":
    DATA_PATH = ROOT.parent.parent / "data" / "freMTPL2freq.parquet"
OUT_DIR = ROOT / "benchmarks" / "results"
OUT_HTML = OUT_DIR / "superbooster_visual_report.html"
OUT_JSON = OUT_DIR / "superbooster_visual_report.json"


def load_freq() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load the local MTPL frequency data."""
    df = pd.read_parquet(DATA_PATH)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4).astype(float)
    df["Exposure"] = df["Exposure"].clip(lower=0.01).astype(float)
    df["DrivAge"] = df["DrivAge"].clip(18, 90).astype(float)
    df["VehAge"] = df["VehAge"].clip(0, 20).astype(float)
    df["BonusMalus"] = df["BonusMalus"].clip(50, 150).astype(float)
    X = df[["DrivAge", "VehAge", "BonusMalus", "Area"]].copy()
    y = df["ClaimNb"].to_numpy(dtype=float)
    exposure = df["Exposure"].to_numpy(dtype=float)
    offset = np.log(exposure)
    return X, y, exposure, offset


def build_features() -> dict[str, object]:
    """Feature set reused across backbone and interaction models."""
    return {
        "DrivAge": Spline(kind="cr", k=20, penalty="ssp", discrete=True),
        "VehAge": Spline(kind="cr", k=15, penalty="ssp", discrete=True),
        "BonusMalus": Spline(kind="cr", k=15, penalty="ssp", discrete=True),
        "Area": Categorical(base="most_exposed"),
    }


def build_booster_frame(X: pd.DataFrame, area_levels: list[str]) -> pd.DataFrame:
    """Construct raw-term booster features."""
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


def aggregate_shap(
    shap_matrix: np.ndarray,
    term_slices: dict[str, list[int]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Aggregate raw XGBoost SHAP columns back to parent terms."""
    bias = shap_matrix[:, -1]
    out: dict[str, np.ndarray] = {}
    for term_name, cols in term_slices.items():
        out[term_name] = shap_matrix[:, cols].sum(axis=1)
    return out, bias


def unique_leaf_groups(leaf_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return representative row indices and inverse group map for leaf signatures."""
    leaf_contig = np.ascontiguousarray(leaf_matrix)
    row_view = leaf_contig.view(
        np.dtype((np.void, leaf_contig.dtype.itemsize * leaf_contig.shape[1]))
    ).ravel()
    _, rep_idx, inverse = np.unique(row_view, return_index=True, return_inverse=True)
    return rep_idx.astype(np.intp, copy=False), inverse.astype(np.intp, copy=False)


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


def weighted_bin_summary(
    x: np.ndarray,
    backbone: np.ndarray,
    correction: np.ndarray,
    weight: np.ndarray,
    *,
    n_bins: int = 40,
) -> pd.DataFrame:
    """Weighted binned summary for a continuous feature."""
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(float(np.min(x)), float(np.max(x)), min(n_bins, 10) + 1)
    bins = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
    df = pd.DataFrame(
        {
            "x": x,
            "backbone": backbone,
            "correction": correction,
            "corrected": backbone + correction,
            "weight": weight,
            "bin": bins,
        }
    )

    def wavg(g: pd.DataFrame, col: str) -> float:
        return float(np.average(g[col], weights=g["weight"]))

    summary = (
        df.groupby("bin", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "x_mean": wavg(g, "x"),
                    "backbone": wavg(g, "backbone"),
                    "correction": wavg(g, "correction"),
                    "corrected": wavg(g, "corrected"),
                    "weight": float(g["weight"].sum()),
                }
            )
        )
        .reset_index(drop=True)
    )
    summary["weight_share"] = summary["weight"] / summary["weight"].sum()
    return summary


def weighted_level_summary(
    level: pd.Series,
    backbone: np.ndarray,
    correction: np.ndarray,
    weight: np.ndarray,
) -> pd.DataFrame:
    """Weighted summary for a categorical feature."""
    df = pd.DataFrame(
        {
            "level": level.astype(str).to_numpy(),
            "backbone": backbone,
            "correction": correction,
            "corrected": backbone + correction,
            "weight": weight,
        }
    )

    def wavg(g: pd.DataFrame, col: str) -> float:
        return float(np.average(g[col], weights=g["weight"]))

    summary = (
        df.groupby("level", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "backbone": wavg(g, "backbone"),
                    "correction": wavg(g, "correction"),
                    "corrected": wavg(g, "corrected"),
                    "weight": float(g["weight"].sum()),
                }
            )
        )
        .reset_index()
        .sort_values("level")
    )
    summary["weight_share"] = summary["weight"] / summary["weight"].sum()
    return summary


def interaction_correction_surface(
    x1: np.ndarray,
    x2: np.ndarray,
    delta: np.ndarray,
    weight: np.ndarray,
    *,
    n_bins: int = 22,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exposure-weighted mean correction and exposure share on a 2D grid."""
    x1_edges = np.unique(np.quantile(x1, np.linspace(0.0, 1.0, n_bins + 1)))
    x2_edges = np.unique(np.quantile(x2, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(x1_edges) < 3:
        x1_edges = np.linspace(float(np.min(x1)), float(np.max(x1)), min(n_bins, 10) + 1)
    if len(x2_edges) < 3:
        x2_edges = np.linspace(float(np.min(x2)), float(np.max(x2)), min(n_bins, 10) + 1)

    x1_codes = np.clip(np.digitize(x1, x1_edges[1:-1], right=True), 0, len(x1_edges) - 2)
    x2_codes = np.clip(np.digitize(x2, x2_edges[1:-1], right=True), 0, len(x2_edges) - 2)
    n1 = len(x1_edges) - 1
    n2 = len(x2_edges) - 1

    weight_sum = np.zeros((n2, n1), dtype=float)
    delta_weight_sum = np.zeros((n2, n1), dtype=float)
    x1_weight_sum = np.zeros((n2, n1), dtype=float)
    x2_weight_sum = np.zeros((n2, n1), dtype=float)

    for i in range(len(delta)):
        r = x2_codes[i]
        c = x1_codes[i]
        w = weight[i]
        weight_sum[r, c] += w
        delta_weight_sum[r, c] += w * delta[i]
        x1_weight_sum[r, c] += w * x1[i]
        x2_weight_sum[r, c] += w * x2[i]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_delta = delta_weight_sum / weight_sum
        mean_x1 = x1_weight_sum / weight_sum
        mean_x2 = x2_weight_sum / weight_sum
    mean_delta[weight_sum == 0] = np.nan
    mean_x1[weight_sum == 0] = np.nan
    mean_x2[weight_sum == 0] = np.nan

    effect_df = pd.DataFrame(
        {
            "x1": mean_x1.ravel(),
            "x2": mean_x2.ravel(),
            "correction": mean_delta.ravel(),
        }
    ).dropna()
    density_df = pd.DataFrame(
        {
            "x1": mean_x1.ravel(),
            "x2": mean_x2.ravel(),
            "weight_share": (weight_sum / np.sum(weight_sum)).ravel(),
        }
    ).dropna()
    return effect_df, density_df


def sample_rows(
    X: pd.DataFrame,
    exposure: np.ndarray,
    offset: np.ndarray,
    n_rows: int,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Sample aligned rows from the full dataset."""
    if n_rows >= len(X):
        return X.copy(), exposure.copy(), offset.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_rows, replace=False)
    return (
        X.iloc[idx].reset_index(drop=True),
        exposure[idx].copy(),
        offset[idx].copy(),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, exposure, offset = load_freq()

    # Density overlays don't need every row to look realistic.
    X_plot, exposure_plot, _ = sample_rows(X, exposure, offset, 200_000, seed=42)

    backbone = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=256,
        features=build_features(),
    )
    backbone.fit_reml(X, y, offset=offset, max_reml_iter=20)

    main_effects_fig = backbone.plot(
        engine="plotly",
        X=X_plot,
        sample_weight=exposure_plot,
        title="Backbone Main Effects",
        subtitle="SuperGLM Poisson backbone on ClaimNb with log(Exposure) offset",
    )

    interaction_model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=256,
        features=build_features(),
        interactions=[("DrivAge", "VehAge")],
    )
    interaction_model.fit_reml(X, y, offset=offset, max_reml_iter=15)

    interaction_fig = interaction_model.plot(
        "DrivAge:VehAge",
        engine="plotly",
        interaction_view="contour_pair",
        X=X_plot,
        sample_weight=exposure_plot,
    )
    interaction_fig.update_layout(title_text="Backbone Interaction View: DrivAge × VehAge")

    eta_backbone = backbone._predict_eta_exact(X, offset)
    backbone_term_map = backbone_terms(backbone, X)
    area_levels = sorted(pd.Series(X["Area"]).astype(str).unique().tolist())
    area_cols = [f"Area__{level}" for level in area_levels]
    Xb = build_booster_frame(X, area_levels)

    dtrain = xgb.DMatrix(
        Xb.to_numpy(dtype=np.float32),
        label=y.astype(np.float32),
        base_margin=eta_backbone.astype(np.float32),
    )
    booster = xgb.train(
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
        dtrain,
        num_boost_round=128,
        verbose_eval=False,
    )

    term_slices = {
        "DrivAge": [0],
        "VehAge": [1],
        "BonusMalus": [2],
        "Area": list(range(3, 3 + len(area_cols))),
    }

    # Exact compressed full-dataset SHAP for portfolio-level importance.
    zero_margin = np.zeros(len(X), dtype=np.float32)
    leaf_dm = xgb.DMatrix(Xb.to_numpy(dtype=np.float32), base_margin=zero_margin)
    leaf_matrix = np.asarray(booster.predict(leaf_dm, pred_leaf=True), dtype=np.int32)
    rep_idx, inverse = unique_leaf_groups(leaf_matrix)
    rep_dm = xgb.DMatrix(
        Xb.iloc[rep_idx].to_numpy(dtype=np.float32),
        base_margin=np.zeros(len(rep_idx), dtype=np.float32),
    )
    shap_rep = booster.predict(rep_dm, pred_contribs=True)
    shap_terms_rep, shap_bias_rep = aggregate_shap(shap_rep, term_slices)
    shap_terms_full = {
        term_name: shap_terms_rep[term_name][inverse] for term_name in backbone._feature_order
    }
    shap_bias_full = shap_bias_rep[inverse]
    delta_boost = shap_bias_full.copy()
    for term_name in backbone._feature_order:
        delta_boost += shap_terms_full[term_name]
    term_importance = {
        term_name: float(np.mean(np.abs(shap_terms_full[term_name])))
        for term_name in backbone._feature_order
    }
    importance_df = (
        pd.DataFrame(
            {
                "term": list(term_importance.keys()),
                "mean_abs_shap": list(term_importance.values()),
            }
        )
        .sort_values("mean_abs_shap", ascending=True)
        .reset_index(drop=True)
    )
    shap_bar_fig = px.bar(
        importance_df,
        x="mean_abs_shap",
        y="term",
        orientation="h",
        title="Booster SHAP Importance by Parent Term",
        labels={"mean_abs_shap": "Mean |SHAP correction| on link scale", "term": ""},
    )

    corrected_terms_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "DrivAge corrected term",
            "VehAge corrected term",
            "BonusMalus corrected term",
            "Area corrected term",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    line_palette = {
        "backbone": "#1E63D7",
        "correction": "#D94841",
        "corrected": "#0F766E",
    }
    continuous_terms = ["DrivAge", "VehAge", "BonusMalus"]
    subplot_map = {"DrivAge": (1, 1), "VehAge": (1, 2), "BonusMalus": (2, 1)}
    for term_name in continuous_terms:
        summary = weighted_bin_summary(
            X[term_name].to_numpy(dtype=float),
            backbone_term_map[term_name],
            shap_terms_full[term_name],
            exposure,
        )
        row, col = subplot_map[term_name]
        for series_name in ["backbone", "correction", "corrected"]:
            corrected_terms_fig.add_trace(
                go.Scatter(
                    x=summary["x_mean"],
                    y=summary[series_name],
                    mode="lines+markers",
                    name=series_name.title(),
                    marker=dict(size=5),
                    line=dict(color=line_palette[series_name], width=2.5),
                    legendgroup=series_name,
                    showlegend=(term_name == "DrivAge"),
                ),
                row=row,
                col=col,
            )
        corrected_terms_fig.update_xaxes(title_text=term_name, row=row, col=col)
        corrected_terms_fig.update_yaxes(title_text="Link-scale term", row=row, col=col)

    area_summary = weighted_level_summary(
        X["Area"],
        backbone_term_map["Area"],
        shap_terms_full["Area"],
        exposure,
    )
    for series_name in ["backbone", "correction", "corrected"]:
        corrected_terms_fig.add_trace(
            go.Bar(
                x=area_summary["level"],
                y=area_summary[series_name],
                name=series_name.title(),
                marker_color=line_palette[series_name],
                legendgroup=series_name,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    corrected_terms_fig.update_xaxes(title_text="Area", row=2, col=2)
    corrected_terms_fig.update_yaxes(title_text="Link-scale term", row=2, col=2)
    corrected_terms_fig.update_layout(
        title_text="Backbone vs Booster Correction vs Corrected Term",
        height=900,
        barmode="group",
        template="plotly_white",
    )

    effect_df, density_df = interaction_correction_surface(
        X["DrivAge"].to_numpy(dtype=float),
        X["VehAge"].to_numpy(dtype=float),
        delta_boost,
        exposure,
    )
    correction_heatmap_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Booster correction heatmap", "Exposure-share heatmap"),
        horizontal_spacing=0.10,
    )
    correction_heatmap_fig.add_trace(
        go.Histogram2d(
            x=effect_df["x1"],
            y=effect_df["x2"],
            z=effect_df["correction"],
            histfunc="avg",
            colorscale="RdBu",
            zmid=0.0,
            colorbar=dict(title="Correction", x=0.46),
            nbinsx=22,
            nbinsy=22,
            showscale=True,
        ),
        row=1,
        col=1,
    )
    correction_heatmap_fig.add_trace(
        go.Histogram2d(
            x=density_df["x1"],
            y=density_df["x2"],
            z=density_df["weight_share"],
            histfunc="sum",
            colorscale="YlOrBr",
            colorbar=dict(title="Exposure share", x=1.02),
            nbinsx=22,
            nbinsy=22,
            showscale=True,
        ),
        row=1,
        col=2,
    )
    correction_heatmap_fig.update_xaxes(title_text="DrivAge", row=1, col=1)
    correction_heatmap_fig.update_yaxes(title_text="VehAge", row=1, col=1)
    correction_heatmap_fig.update_xaxes(title_text="DrivAge", row=1, col=2)
    correction_heatmap_fig.update_yaxes(title_text="VehAge", row=1, col=2)
    correction_heatmap_fig.update_layout(
        title_text="Booster Total Correction over DrivAge × VehAge",
        height=520,
        template="plotly_white",
    )

    # Sample rows for more detailed SHAP visuals.
    X_shap, _, _ = sample_rows(X, exposure, offset, 30_000, seed=7)
    Xb_shap = build_booster_frame(X_shap, area_levels)
    shap_sample = booster.predict(
        xgb.DMatrix(
            Xb_shap.to_numpy(dtype=np.float32),
            base_margin=np.zeros(len(X_shap), dtype=np.float32),
        ),
        pred_contribs=True,
    )
    shap_terms_sample, shap_bias_sample = aggregate_shap(shap_sample, term_slices)
    delta_sample = shap_sample.sum(axis=1)

    shap_diag = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "DrivAge dependence",
            "BonusMalus dependence",
            "Area correction by level",
            "Total booster correction",
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )
    shap_diag.add_trace(
        go.Scattergl(
            x=X_shap["DrivAge"],
            y=shap_terms_sample["DrivAge"],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.28,
                color=X_shap["VehAge"],
                colorscale="Turbo",
                colorbar=dict(title="VehAge", x=0.46, len=0.38),
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    shap_diag.add_trace(
        go.Scattergl(
            x=X_shap["BonusMalus"],
            y=shap_terms_sample["BonusMalus"],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.28,
                color=X_shap["DrivAge"],
                colorscale="Viridis",
                colorbar=dict(title="DrivAge", x=1.02, len=0.38),
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    for area_level in sorted(X_shap["Area"].astype(str).unique().tolist()):
        mask = X_shap["Area"].astype(str) == area_level
        shap_diag.add_trace(
            go.Box(
                x=np.repeat(area_level, mask.sum()),
                y=shap_terms_sample["Area"][mask.to_numpy()],
                name=area_level,
                boxpoints="outliers",
                marker_size=2,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    shap_diag.add_trace(
        go.Histogram(
            x=delta_sample,
            nbinsx=80,
            marker=dict(color="#D94841"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    shap_diag.update_xaxes(title_text="DrivAge", row=1, col=1)
    shap_diag.update_yaxes(title_text="SHAP correction", row=1, col=1)
    shap_diag.update_xaxes(title_text="BonusMalus", row=1, col=2)
    shap_diag.update_yaxes(title_text="SHAP correction", row=1, col=2)
    shap_diag.update_xaxes(title_text="Area", row=2, col=1)
    shap_diag.update_yaxes(title_text="Area SHAP correction", row=2, col=1)
    shap_diag.update_xaxes(title_text="Total booster correction", row=2, col=2)
    shap_diag.update_yaxes(title_text="Count", row=2, col=2)
    shap_diag.update_layout(
        title_text="Booster SHAP Diagnostics",
        height=900,
        template="plotly_white",
    )

    # Lightweight report metadata.
    metadata = {
        "dataset": "freMTPL2freq",
        "n_rows": int(len(X)),
        "plot_density_rows": int(len(X_plot)),
        "shap_detail_rows": int(len(X_shap)),
        "leaf_signature_count": int(len(rep_idx)),
        "compression_ratio": float(len(X) / len(rep_idx)),
        "backbone_effective_df": float(backbone.result.effective_df),
        "interaction_effective_df": float(interaction_model.result.effective_df),
        "booster_rounds": int(booster.num_boosted_rounds()),
        "term_importance": term_importance,
        "delta_boost_mean": float(np.mean(delta_boost)),
        "delta_boost_std": float(np.std(delta_boost)),
        "report_html": str(OUT_HTML),
    }
    OUT_JSON.write_text(json.dumps(metadata, indent=2))

    sections = [
        (
            "Backbone Main Effects",
            "Exact SuperGLM main-effect explorer with exposure overlays from a large sample.",
            main_effects_fig,
        ),
        (
            "Backbone Interaction",
            "Continuous interaction contour pair for DrivAge × VehAge from a SuperGLM with an explicit interaction term.",
            interaction_fig,
        ),
        (
            "Booster SHAP Importance",
            "Exact compressed SHAP importance on the full dataset, aggregated back to parent terms.",
            shap_bar_fig,
        ),
        (
            "Corrected Terms",
            "Exposure-weighted overlays showing the original backbone term, the booster correction, and the corrected final term for each parent feature.",
            corrected_terms_fig,
        ),
        (
            "Correction Heatmap",
            "Average total booster correction across DrivAge × VehAge, alongside exposure share over the same grid.",
            correction_heatmap_fig,
        ),
        (
            "Booster SHAP Diagnostics",
            "Dependence, categorical correction, and overall correction diagnostics from a 30k-row SHAP sample.",
            shap_diag,
        ),
    ]

    html_parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>SuperBooster Visual Report</title>",
        "<style>",
        "body { font-family: 'Avenir Next', 'Segoe UI', sans-serif; margin: 0; background: #f6f1e8; color: #171411; }",
        ".wrap { max-width: 1400px; margin: 0 auto; padding: 28px 24px 56px; }",
        "h1, h2 { margin: 0 0 12px; }",
        ".lede { max-width: 900px; font-size: 18px; line-height: 1.5; margin-bottom: 22px; }",
        ".meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 0 0 24px; }",
        ".card { background: #fffdf8; border: 1px solid rgba(23, 20, 17, 0.10); border-radius: 16px; padding: 14px 16px; box-shadow: 0 10px 24px rgba(24, 33, 43, 0.06); }",
        ".section { margin-top: 28px; background: #fffdf8; border: 1px solid rgba(23, 20, 17, 0.10); border-radius: 20px; padding: 18px 18px 8px; box-shadow: 0 10px 24px rgba(24, 33, 43, 0.06); }",
        ".section p { margin: 0 0 12px; color: #4b433a; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='wrap'>",
        "<h1>SuperBooster Visual Report</h1>",
        "<p class='lede'>A one-off visual readout for the current SuperGLM + XGBoost prototype on the full local freMTPL2 frequency dataset.</p>",
        "<div class='meta'>",
        f"<div class='card'><strong>Rows</strong><br>{len(X):,}</div>",
        f"<div class='card'><strong>Backbone EDF</strong><br>{backbone.result.effective_df:.2f}</div>",
        f"<div class='card'><strong>Interaction EDF</strong><br>{interaction_model.result.effective_df:.2f}</div>",
        f"<div class='card'><strong>Booster Rounds</strong><br>{booster.num_boosted_rounds()}</div>",
        f"<div class='card'><strong>Leaf Signatures</strong><br>{len(rep_idx):,}</div>",
        f"<div class='card'><strong>SHAP Compression</strong><br>{len(X) / len(rep_idx):.2f}x</div>",
        "</div>",
    ]
    first = True
    for heading, description, fig in sections:
        html_parts.append("<section class='section'>")
        html_parts.append(f"<h2>{heading}</h2>")
        html_parts.append(f"<p>{description}</p>")
        html_parts.append(
            pio.to_html(
                fig,
                include_plotlyjs="cdn" if first else False,
                full_html=False,
            )
        )
        html_parts.append("</section>")
        first = False
    html_parts.extend(["</div>", "</body>", "</html>"])
    OUT_HTML.write_text("\n".join(html_parts))

    print(f"Saved HTML: {OUT_HTML}")
    print(f"Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
