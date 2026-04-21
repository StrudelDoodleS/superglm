import numpy as np
import pytest
from benchmarks.scop_lambda_sensitivity import (
    build_lambda_grid,
    curve_similarity_metrics,
    summarize_result_rows,
)


def test_build_lambda_grid_is_log_symmetric_around_baseline():
    values = build_lambda_grid(0.5)
    assert np.allclose(
        values,
        0.5 * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]),
    )


def test_build_lambda_grid_rejects_non_positive_baseline():
    with pytest.raises(ValueError, match="baseline_lambda must be positive"):
        build_lambda_grid(0.0)


def test_curve_similarity_metrics_match_identical_curves():
    x = np.linspace(0.0, 1.0, 50)
    y = x**2
    metrics = curve_similarity_metrics(x, y, y)
    assert metrics["r2"] == 1.0
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["rmse"] == 0.0


def test_curve_similarity_metrics_use_x_spacing_for_rmse_and_r2():
    x = np.array([0.0, 0.1, 1.0], dtype=np.float64)
    ref = np.zeros_like(x)
    other = np.array([0.0, 10.0, 10.0], dtype=np.float64)

    metrics = curve_similarity_metrics(x, ref, other)

    assert metrics["max_abs_diff"] == 10.0
    assert metrics["rmse"] == pytest.approx(np.sqrt(95.0))
    assert metrics["r2"] == 0.0


def test_curve_similarity_metrics_reject_mismatched_shapes():
    x = np.linspace(0.0, 1.0, 5)

    with pytest.raises(ValueError, match="same shape"):
        curve_similarity_metrics(x, np.zeros(5), np.zeros(4))


def test_summarize_result_rows_has_expected_columns():
    df = summarize_result_rows(
        [
            {
                "scenario": "demo",
                "comparison": "baseline",
                "target": "curve",
                "r2": 1.0,
                "max_abs_diff": 0.0,
                "rmse": 0.0,
            }
        ]
    )

    assert list(df.columns) == [
        "scenario",
        "comparison",
        "target",
        "r2",
        "max_abs_diff",
        "rmse",
    ]


def test_summarize_result_rows_preserves_column_order_for_empty_input():
    df = summarize_result_rows([])

    assert df.empty
    assert list(df.columns) == [
        "scenario",
        "comparison",
        "target",
        "r2",
        "max_abs_diff",
        "rmse",
    ]
