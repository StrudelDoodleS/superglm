"""PSST end-to-end on freMTPL2: screen all spline pairs, confirm the top hit.

The Emblem-replacement story in one script: fit mains, rank every candidate
ti() pair with screen_interactions, refit the top-ranked pairs as real ti()
terms, and report the deviance they buy.  Records wall time for each stage.

Results for the reference box are recorded in
docs/superpowers/plans/2026-07-28-interaction-screening.md (Task 5).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

SPLINES = ("DrivAge", "VehAge", "BonusMalus", "VehPower", "Density")


def load(n=100_000, seed=0):
    root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(root / "data" / "freMTPL2freq.parquet")
    df["ClaimNb"] = np.asarray(df["ClaimNb"], dtype=float).clip(0, 4)
    df["Exposure"] = np.asarray(df["Exposure"], dtype=float).clip(1e-3, 1.0)
    idx = np.random.default_rng(seed).choice(len(df), size=n, replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    X = df[[*SPLINES, "Area"]].copy()
    for col in SPLINES:
        X[col] = X[col].astype(float)
    X["Area"] = X["Area"].astype(str)
    y = (df["ClaimNb"] / df["Exposure"]).to_numpy()
    w = df["Exposure"].to_numpy()
    return X, y, w


def build(pairs=()):
    features = {c: Spline(kind="ps", k=10) for c in SPLINES}
    features["Area"] = Categorical(base="first")
    model = SuperGLM(family="poisson", selection_penalty=None, discrete=False, features=features)
    for a, b in pairs:
        model._add_interaction(a, b)
    return model


def main():
    X, y, w = load()

    t0 = time.perf_counter()
    mains = build().fit_reml(X, y, sample_weight=w)
    t_fit = time.perf_counter() - t0
    print(f"mains fit_reml: {t_fit:.2f}s  deviance={mains._result.deviance:.2f}")

    t0 = time.perf_counter()
    table = mains.screen_interactions(X, y, sample_weight=w)
    t_screen = time.perf_counter() - t0
    print(f"PSST sweep ({len(table)} pairs): {t_screen:.3f}s")
    print(table.to_string(index=False))

    top = [
        (row.feature_a, row.feature_b)
        for row in table.head(2).itertuples()
        if np.isfinite(row.statistic)
    ]
    t0 = time.perf_counter()
    confirmed = build(pairs=top).fit_reml(X, y, sample_weight=w)
    t_refit = time.perf_counter() - t0
    gain = mains._result.deviance - confirmed._result.deviance
    print(
        f"confirmatory refit (+{len(top)} ti): {t_refit:.2f}s  "
        f"deviance={confirmed._result.deviance:.2f}  gain={gain:.2f}"
    )


if __name__ == "__main__":
    main()
