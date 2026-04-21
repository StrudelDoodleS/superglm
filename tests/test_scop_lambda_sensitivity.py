import numpy as np
from benchmarks.scop_lambda_sensitivity import (
    build_lambda_grid,
    curve_similarity_metrics,
)


def test_build_lambda_grid_is_log_symmetric_around_baseline():
    values = build_lambda_grid(0.5)
    assert np.allclose(
        values,
        0.5 * np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]),
    )


def test_curve_similarity_metrics_match_identical_curves():
    x = np.linspace(0.0, 1.0, 50)
    y = x**2
    metrics = curve_similarity_metrics(x, y, y)
    assert metrics["r2"] == 1.0
    assert metrics["max_abs_diff"] == 0.0
    assert metrics["rmse"] == 0.0
