import math
from dataclasses import asdict

import numpy as np
import pandas as pd
from benchmarks.benchmark_multi_scop_discrete_convergence import (
    SummaryRow,
    _aggregate_lambda_metrics,
    _execution_orders_for_repeat,
    _prediction_metrics,
)


def test_prediction_metrics_zero_for_identical_inputs():
    yhat = np.array([0.25, 1.5, 2.75], dtype=np.float64)

    metrics = _prediction_metrics(yhat, yhat.copy())

    assert metrics["rmse"] == 0.0
    assert metrics["max_abs_diff"] == 0.0


def test_execution_orders_for_repeat_include_both_orders_once():
    assert _execution_orders_for_repeat(0) == (
        ("baseline", "optimized"),
        ("optimized", "baseline"),
    )
    assert _execution_orders_for_repeat(1) == (
        ("optimized", "baseline"),
        ("baseline", "optimized"),
    )


def test_aggregate_lambda_metrics_returns_nan_when_any_keys_mismatch():
    lambda_max_abs_diff, lambda_keys_match = _aggregate_lambda_metrics(
        lambda_max_abs_diffs=[0.25, float("nan"), 0.5],
        lambda_keys_matches=[True, False, True],
    )

    assert lambda_keys_match is False
    assert math.isnan(lambda_max_abs_diff)


def test_aggregate_lambda_metrics_uses_median_when_all_keys_match():
    lambda_max_abs_diff, lambda_keys_match = _aggregate_lambda_metrics(
        lambda_max_abs_diffs=[0.25, 0.5, 0.75],
        lambda_keys_matches=[True, True, True],
    )

    assert lambda_keys_match is True
    assert lambda_max_abs_diff == 0.5


def test_summary_csv_columns_include_gate_and_order_fields():
    row = SummaryRow(
        dataset="synthetic",
        n_rows=10,
        repeats=3,
        execution_order=(
            "baseline->optimized&optimized->baseline|"
            "optimized->baseline&baseline->optimized|"
            "baseline->optimized&optimized->baseline"
        ),
        baseline_runtime_s=1.0,
        optimized_runtime_s=0.5,
        speedup_x=2.0,
        baseline_n_reml_iter=5,
        optimized_n_reml_iter=4,
        baseline_n_pirls_iter=15,
        optimized_n_pirls_iter=12,
        baseline_converged=True,
        optimized_converged=True,
        baseline_cleanup_gate_calls=3,
        optimized_cleanup_gate_calls=3,
        baseline_cleanup_gate_true_count=0,
        optimized_cleanup_gate_true_count=3,
        baseline_frozen_count=0,
        optimized_frozen_count=1,
        baseline_freeze_iter=0,
        optimized_freeze_iter=3,
        pred_rmse=0.0,
        pred_max_abs_diff=0.0,
        lambda_max_abs_diff=0.0,
        lambda_keys_match=True,
        baseline_lambdas_json="{}",
        optimized_lambdas_json="{}",
    )
    summary = pd.DataFrame([asdict(row)])

    expected_columns = {
        "repeats",
        "execution_order",
        "baseline_cleanup_gate_calls",
        "optimized_cleanup_gate_calls",
        "baseline_cleanup_gate_true_count",
        "optimized_cleanup_gate_true_count",
        "baseline_frozen_count",
        "optimized_frozen_count",
        "baseline_freeze_iter",
        "optimized_freeze_iter",
    }

    assert expected_columns.issubset(summary.columns)
