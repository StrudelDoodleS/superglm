"""Tests for REML bootstrap diagnostics on the discrete path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.spline import Spline
from superglm.reml.penalty_algebra import build_penalty_context


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


def test_projected_tensor_penalty_ranks_use_solver_space() -> None:
    """Side-constrained tensor penalties should report projected-space ranks."""
    rng = np.random.default_rng(123)
    n = 300
    X = pd.DataFrame(
        {
            "A": rng.uniform(18.0, 90.0, n),
            "B": rng.uniform(0.0, 20.0, n),
            "C": np.clip(50.0 + 100.0 * rng.beta(2.0, 4.0, n), 50.0, 150.0),
        }
    )
    y = rng.poisson(0.2, n).astype(float)
    sample_weight = np.ones(n)

    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        n_bins=64,
        features={
            "A": Spline(kind="cr", n_knots=10, penalty="ssp", discrete=True),
            "B": Spline(kind="cr", n_knots=8, penalty="ssp", discrete=True),
            "C": Spline(kind="cr", n_knots=8, penalty="ssp", discrete=True),
        },
        interactions=[("B", "C"), ("A", "C"), ("A", "B")],
    )
    model._build_design_matrix(X, y, sample_weight, None)

    reml_groups = [
        (idx, group)
        for idx, (group_matrix, group) in enumerate(zip(model._dm.group_matrices, model._groups))
        if getattr(group_matrix, "omega", None) is not None
        or getattr(group_matrix, "omega_components", None) is not None
    ]
    penalties, _, _ = build_penalty_context(model._dm.group_matrices, reml_groups)

    projected_components = [
        pc
        for pc in penalties
        if ":" in pc.group_name
        and getattr(model._dm.group_matrices[pc.group_index], "projection", None) is not None
    ]
    assert projected_components

    eps_thresh = np.finfo(float).eps ** (2 / 3)
    for component in projected_components:
        eigvals = np.linalg.eigvalsh(component.omega_ssp)
        thresh = eps_thresh * max(float(eigvals.max()), 1e-12)
        solver_rank = float(np.sum(eigvals > thresh))
        assert component.rank == solver_rank
        assert component.rank <= component.omega_ssp.shape[0]
