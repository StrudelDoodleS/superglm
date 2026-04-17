"""Tests for fold curve similarity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import Categorical, Spline, SuperGLM


def test_pairwise_similarity_matrices_have_expected_diagonals():
    from superglm.plotting.curve_similarity import _pairwise_curve_similarity

    labels = ["fold_0", "fold_1", "fold_2"]
    curves = {
        "fold_0": np.array([1.0, 2.0, 3.0]),
        "fold_1": np.array([1.0, 2.0, 3.0]),
        "fold_2": np.array([2.0, 3.0, 4.0]),
    }
    weights = np.array([1.0, 2.0, 1.0])

    result = _pairwise_curve_similarity(curves, weights, labels=labels)

    np.testing.assert_allclose(np.diag(result["rmse"]), 0.0)
    np.testing.assert_allclose(np.diag(result["max_abs_diff"]), 0.0)
    np.testing.assert_allclose(np.diag(result["correlation"]), 1.0)


def test_weighting_changes_rmse_in_expected_direction():
    from superglm.plotting.curve_similarity import _pairwise_curve_similarity

    curves = {
        "fold_0": np.array([0.0, 0.0, 10.0]),
        "fold_1": np.array([0.0, 0.0, 0.0]),
    }
    low_tail = np.array([10.0, 10.0, 1.0])
    high_tail = np.array([1.0, 1.0, 10.0])

    low_tail_rmse = _pairwise_curve_similarity(curves, low_tail, labels=["fold_0", "fold_1"])[
        "rmse"
    ]
    high_tail_rmse = _pairwise_curve_similarity(curves, high_tail, labels=["fold_0", "fold_1"])[
        "rmse"
    ]

    assert high_tail_rmse.loc["fold_0", "fold_1"] > low_tail_rmse.loc["fold_0", "fold_1"]


def test_fold_mean_distances_are_reported():
    from superglm.plotting.curve_similarity import _summarize_against_fold_mean

    curves = {
        "fold_0": np.array([1.0, 2.0, 3.0]),
        "fold_1": np.array([1.0, 2.0, 3.0]),
        "fold_2": np.array([2.0, 3.0, 4.0]),
    }
    weights = np.array([1.0, 1.0, 1.0])

    summary = _summarize_against_fold_mean(curves, weights)

    assert list(summary.columns) == ["rmse_to_mean", "max_abs_diff_to_mean", "correlation_to_mean"]
    assert summary.index.tolist() == ["fold_0", "fold_1", "fold_2"]


def test_build_cv_curve_similarity_returns_both_scales_for_all_comparable_terms():
    from superglm.plotting.curve_similarity import build_cv_curve_similarity

    rng = np.random.default_rng(7)
    n = 160
    x = rng.uniform(0, 10, n)
    band = rng.choice(["A", "B", "C", "D"], n)
    w = rng.uniform(0.5, 1.2, n)
    eta = -1.3 + 0.2 * np.sin(x) + 0.15 * (band == "C")
    y = rng.poisson(np.exp(eta) * w).astype(float)
    X = pd.DataFrame({"x": x, "band": band})

    models = []
    for seed in [1, 2, 3]:
        idx = np.random.default_rng(seed).choice(n, size=int(0.8 * n), replace=False)
        model = SuperGLM(
            features={
                "x": Spline(n_knots=6),
                "band": Categorical(base="first"),
            }
        )
        model.fit(X.iloc[idx], y[idx], sample_weight=w[idx])
        models.append(model)

    similarity = build_cv_curve_similarity(models=models, X=X, sample_weight=w, n_points=51)

    assert set(similarity) == {"x", "band"}
    assert "response" in similarity["x"]["pairwise"]
    assert "link" in similarity["x"]["pairwise"]
    assert len(similarity["x"]["domain"]["x"]) == 51
