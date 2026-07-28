"""Stress battery for support compression beyond freMTPL2's easy cardinalities.

A: exactness -- the same tensor fit with detection on vs off; report beta,
   deviance, EDF and prediction deltas (the "is it really exact" evidence).
B: high-cardinality integer covariate -- Density (~1.6k distinct) as a spline
   plus ti(DrivAge, Density) whose joint support is ~35k of 100k rows.
C: worst case -- a jittered (continuous) covariate, where the gate must
   decline; measures the residual cost of the uncompressed tensor path.
D: two tensors sharing a marginal -- cross-gram cell pressure.

Recorded results for this box live in docs/audit/2026-07-28/architecture-audit.md
section J.5.
"""

import time

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm._group_matrix import _group_matrix_support as support_mod
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

RNG = np.random.default_rng(0)


def _read_frame() -> pd.DataFrame:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("freMTPL2freq.parquet", "freMTPL2freq.csv"):
        path = root / "data" / name
        if path.exists():
            return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    raise FileNotFoundError("freMTPL2freq data not found under data/")


def prep(n=100_000):
    df = _read_frame()
    df["ClaimNb"] = np.asarray(df["ClaimNb"], dtype=float).clip(0, 4)
    df["Exposure"] = np.asarray(df["Exposure"], dtype=float).clip(1e-3, 1.0)
    idx = np.random.default_rng(0).choice(len(df), size=n, replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    X = df[["DrivAge", "VehAge", "BonusMalus", "VehPower", "Density", "Area"]].copy()
    for col in X.columns[:-1]:
        X[col] = X[col].astype(float)
    X["Area"] = X["Area"].astype(str)
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy()
    w = df["Exposure"].to_numpy()
    return X, y, w


def cardinalities(X, pairs):
    print("distinct values:", {c: int(X[c].nunique()) for c in X.columns if c != "Area"})
    for a, b in pairs:
        joint = len(np.unique(X[a].to_numpy() * 1e9 + X[b].to_numpy()))
        print(f"joint support {a}:{b} = {joint}")


def build(features, interactions, discrete=False):
    model = SuperGLM(family="poisson", selection_penalty=None, discrete=discrete, features=features)
    for a, b in interactions:
        model._add_interaction(a, b)
    return model


def fit_timed(model, X, y, w):
    t0 = time.perf_counter()
    model.fit_reml(X, y, sample_weight=w)
    return time.perf_counter() - t0


def representations(model):
    summary = model.design_summary()
    cols = [c for c in ("term", "representation", "lossless_support") if c in summary.columns]
    return summary[cols].to_string(index=False)


def main():
    X, y, w = prep()
    base = {c: Spline(kind="ps", k=10) for c in ("DrivAge", "VehAge", "BonusMalus", "VehPower")}
    base["Area"] = Categorical(base="first")

    print("=== cardinalities ===", flush=True)
    cardinalities(
        X,
        [("DrivAge", "BonusMalus"), ("DrivAge", "Density"), ("VehAge", "BonusMalus")],
    )

    print("\n=== A: exactness of compression (detection on vs off) ===", flush=True)
    m_on = build(dict(base), [("DrivAge", "BonusMalus")])
    t_on = fit_timed(m_on, X, y, w)
    saved = support_mod.detect_row_support
    try:
        support_mod.detect_row_support = lambda *a, **k: None
        m_off = build(dict(base), [("DrivAge", "BonusMalus")])
        t_off = fit_timed(m_off, X, y, w)
    finally:
        support_mod.detect_row_support = saved
    beta_on, beta_off = m_on._result.beta, m_off._result.beta
    scale = max(float(np.max(np.abs(beta_off))), 1.0)
    print(f"fit times: compressed {t_on:.2f}s vs uncompressed {t_off:.2f}s", flush=True)
    print(
        f"max|dbeta| = {np.max(np.abs(beta_on - beta_off)):.3e} (rel {np.max(np.abs(beta_on - beta_off)) / scale:.3e})"
    )
    print(f"deviance: {m_on._result.deviance:.10f} vs {m_off._result.deviance:.10f}")
    print(f"effective_df: {m_on._result.effective_df:.8f} vs {m_off._result.effective_df:.8f}")
    mu_on = m_on.predict(X)
    mu_off = m_off.predict(X)
    print(f"max rel |dmu| = {np.max(np.abs(mu_on - mu_off) / np.maximum(mu_off, 1e-12)):.3e}")

    print("\n=== B: high-cardinality Density spline + ti(DrivAge, Density) ===", flush=True)
    feats_b = dict(base)
    feats_b["Density"] = Spline(kind="ps", k=10)
    m_b = build(feats_b, [("DrivAge", "Density")])
    t_b = fit_timed(m_b, X, y, w)
    print(f"exact fit {t_b:.2f}s", flush=True)
    print(representations(m_b))
    m_b_disc = build(feats_b, [("DrivAge", "Density")], discrete=True)
    t_bd = fit_timed(m_b_disc, X, y, w)
    print(f"discrete fit {t_bd:.2f}s")

    print("\n=== C: continuous covariate (jittered Density), nothing compresses ===", flush=True)
    X_c = X.copy()
    X_c["Density"] = X_c["Density"] + RNG.uniform(0.0, 1.0, len(X_c))
    print(f"jittered distinct = {X_c['Density'].nunique()}")
    m_c = build(feats_b, [("DrivAge", "Density")])
    t_c = fit_timed(m_c, X_c, y, w)
    print(f"exact fit, detection enabled {t_c:.2f}s", flush=True)
    try:
        support_mod.detect_row_support = lambda *a, **k: None
        m_c_off = build(feats_b, [("DrivAge", "Density")])
        t_c_off = fit_timed(m_c_off, X_c, y, w)
    finally:
        support_mod.detect_row_support = saved
    print(f"exact fit, detection disabled {t_c_off:.2f}s (overhead = {t_c - t_c_off:+.2f}s)")
    print(representations(m_c))

    print("\n=== D: two tensors sharing BonusMalus (cross-gram cell pressure) ===", flush=True)
    m_d = build(dict(base), [("DrivAge", "BonusMalus"), ("VehAge", "BonusMalus")])
    t_d = fit_timed(m_d, X, y, w)
    print(f"exact fit {t_d:.2f}s", flush=True)
    prof = m_d._reml_profile
    for key in sorted(prof):
        if "block_cross" in key or "fallback" in key:
            print(f"  {key} = {prof[key]}")
    print(representations(m_d))


if __name__ == "__main__":
    main()
