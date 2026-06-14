"""Tests for REML bootstrap diagnostics on the discrete path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.spline import Spline


def test_discrete_reml_records_bootstrap_lambda_diagnostics() -> None:
    """fit_reml(discrete=True) should retain bootstrap lambda diagnostics."""
    rng = np.random.default_rng(42)
    n = 240
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    eta = 0.1 + np.sin(2 * np.pi * x1) + 0.5 * np.cos(2 * np.pi * x2) + 0.4 * x1 * x2
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})

    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=64,
        features={
            "x1": Spline(kind="cr", k=8, penalty="ssp", discrete=True),
            "x2": Spline(kind="cr", k=8, penalty="ssp", discrete=True),
        },
        interactions=[("x1", "x2")],
    )

    model.fit_reml(X, y, max_reml_iter=5)

    profile = model._reml_profile
    assert profile is not None
    assert "reml_bootstrap_summary" in profile
    assert "reml_bootstrap_components" in profile

    summary = profile["reml_bootstrap_summary"]
    components = profile["reml_bootstrap_components"]

    assert summary["n_components"] == len(components)
    assert summary["boot_phi"] > 0
    assert summary["boot_inv_phi"] > 0
    assert summary["lam_fp_min"] <= summary["lam_fp_max"]
    assert len(components) > 0

    required = {
        "name",
        "group_name",
        "rank",
        "quad",
        "trace_term",
        "denom",
        "lam_fp_raw",
        "lam_fp_clipped",
        "beta_norm",
        "omega_frob",
        "block_dim",
    }
    assert required.issubset(components[0])
