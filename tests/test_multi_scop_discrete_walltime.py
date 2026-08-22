import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from benchmarks.benchmark_multi_scop_discrete_convergence import (
    SummaryRow,
    _aggregate_lambda_metrics,
    _execution_orders_for_repeat,
    _prediction_metrics,
)

import superglm.reml.scop_efs as scop_efs
from superglm import Categorical, Constraint, CubicRegressionSpline, PSpline, SuperGLM
from superglm.solvers.pirls import PIRLSResult


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


def test_managed_cleanup_can_freeze_floor_pinned_lambda(monkeypatch):
    pirls_result = PIRLSResult(
        beta=np.array([0.0]),
        intercept=0.0,
        n_iter=1,
        deviance=0.0,
        converged=True,
        phi=1.0,
        effective_df=1.0,
    )
    lambda_updates = iter(
        [
            {"DrivAge": 0.25, "BonusMalus": 1.0e-4},
            {"DrivAge": 0.20, "BonusMalus": 1.0e-4},
            {"DrivAge": 0.18, "BonusMalus": 1.0e-4},
            {"DrivAge": 0.16, "BonusMalus": 1.0e-4},
            {"DrivAge": 0.14, "BonusMalus": 1.0e-4},
        ]
    )
    active_name_calls: list[set[str]] = []
    penalties = [
        SimpleNamespace(name="DrivAge"),
        SimpleNamespace(name="BonusMalus"),
    ]

    def make_mode(lambdas):
        return SimpleNamespace(
            lambdas=lambdas.copy(),
            result=pirls_result,
            scop_states={},
            penalty_components=penalties,
            hessian_inverse=np.eye(1),
            evaluation=SimpleNamespace(value=1.0),
            objective=1.0,
            curvature_source="fisher",
        )

    monkeypatch.setattr(
        scop_efs,
        "_fit_scop_reml_mode",
        lambda context, lambdas, **kwargs: make_mode(lambdas),
    )
    monkeypatch.setattr(
        scop_efs,
        "_backtrack_scop_efs_candidate",
        lambda context, current, proposed_lambdas, **kwargs: (
            make_mode(proposed_lambdas),
            True,
        ),
    )
    monkeypatch.setattr(
        scop_efs,
        "_finalize_scop_reml_mode",
        lambda context, mode: mode.result,
    )

    def fake_joint_efs_lambda_step(*args, **kwargs):
        del kwargs
        active_name_calls.append(set(args[5]))
        return next(lambda_updates), args[7], {}

    monkeypatch.setattr(scop_efs, "_joint_efs_lambda_step", fake_joint_efs_lambda_step)
    monkeypatch.setattr(
        scop_efs,
        "_multi_scop_discrete_cleanup_names",
        lambda **kwargs: {"DrivAge", "BonusMalus"},
    )

    result = scop_efs.optimize_scop_efs_reml(
        dm=SimpleNamespace(group_matrices=[], p=1),
        distribution=SimpleNamespace(scale_known=True),
        link=SimpleNamespace(),
        groups=[
            SimpleNamespace(monotone_engine="scop"),
            SimpleNamespace(monotone_engine="scop"),
        ],
        y=np.array([0.0]),
        sample_weight=np.ones(1),
        offset_arr=np.zeros(1),
        lambdas={"DrivAge": 1.0, "BonusMalus": 1.0},
        estimated_names={"DrivAge", "BonusMalus"},
        max_reml_iter=4,
        reml_penalties=penalties,
        weight_semantics="frequency",
    )

    assert active_name_calls == [
        {"DrivAge", "BonusMalus"},
        {"DrivAge", "BonusMalus"},
        {"DrivAge", "BonusMalus"},
        {"DrivAge", "BonusMalus"},
        {"DrivAge"},
    ]
    assert result.managed_cleanup_freeze_iter == 3
    assert result.managed_cleanup_active_history is not None
    assert result.managed_cleanup_frozen_history is not None
    assert result.managed_cleanup_active_history[2] == ["DrivAge"]
    assert result.managed_cleanup_frozen_history[2] == ["BonusMalus"]


def _make_multi_scop_data(n: int = 1500, seed: int = 42):
    rng = np.random.default_rng(seed)
    driv_age = rng.uniform(18.0, 85.0, size=n)
    veh_age = rng.uniform(0.0, 20.0, size=n)
    bonus_malus = rng.uniform(50.0, 150.0, size=n)
    density = rng.uniform(10.0, 5000.0, size=n)
    area = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    eta = (
        -2.3
        - 0.018 * (driv_age - 45.0) ** 2 / 25.0
        - 0.0015 * (bonus_malus - 90.0) ** 2 / 12.0
        + 0.02 * np.sin(veh_age / 3.0)
        + 0.08 * np.log(density)
        + np.where(area == "B", 0.1, 0.0)
        + np.where(area == "C", -0.08, 0.0)
    )
    exposure = rng.uniform(0.2, 1.5, size=n)
    y = rng.poisson(exposure * np.exp(eta)).astype(float) / exposure
    X = pd.DataFrame(
        {
            "DrivAge": driv_age,
            "VehAge": veh_age,
            "BonusMalus": bonus_malus,
            "LogDensity": np.log(density),
            "Area": area,
        }
    )
    return X, y, exposure.astype(float)


@pytest.mark.slow
def test_reml_result_exposes_multi_scop_cleanup_metrics():
    X, y, sample_weight = _make_multi_scop_data(seed=11)
    model = SuperGLM(
        family="poisson",
        selection_penalty=0.0,
        discrete=True,
        features={
            "DrivAge": PSpline(n_knots=10, penalty="ssp", constraint=Constraint.fit.concave),
            "VehAge": CubicRegressionSpline(n_knots=8),
            "BonusMalus": PSpline(n_knots=10, penalty="ssp", constraint=Constraint.fit.concave),
            "LogDensity": CubicRegressionSpline(n_knots=8),
            "Area": Categorical(base="most_exposed"),
        },
    )

    model.fit_reml(X, y, sample_weight=sample_weight, max_reml_iter=20)
    reml_result = model._reml_result
    managed_cleanup_names = {"BonusMalus", "DrivAge"}

    assert reml_result.converged
    assert reml_result.managed_cleanup_names == sorted(managed_cleanup_names)
    assert reml_result.managed_cleanup_active_history is not None
    assert reml_result.managed_cleanup_frozen_history is not None
    assert reml_result.managed_cleanup_frozen_names is not None
    assert len(reml_result.managed_cleanup_active_history) == reml_result.n_reml_iter
    assert len(reml_result.managed_cleanup_frozen_history) == reml_result.n_reml_iter
    assert reml_result.n_reml_iter >= 3

    active_history = [set(names) for names in reml_result.managed_cleanup_active_history]
    frozen_history = [set(names) for names in reml_result.managed_cleanup_frozen_history]
    observed_freeze_iter = next(
        (
            idx + 1
            for idx, (prev_names, curr_names) in enumerate(
                zip([set(), *frozen_history[:-1]], frozen_history, strict=False)
            )
            if curr_names != prev_names
        ),
        None,
    )

    for active_names, frozen_names in zip(active_history, frozen_history, strict=False):
        assert active_names <= managed_cleanup_names
        assert frozen_names <= managed_cleanup_names
        assert active_names.isdisjoint(frozen_names)
        assert active_names | frozen_names == managed_cleanup_names

    assert frozen_history[0] == set()
    for previous, current in zip(frozen_history, frozen_history[1:], strict=False):
        assert previous <= current
    assert reml_result.managed_cleanup_frozen_names == sorted(frozen_history[-1])
    assert reml_result.managed_cleanup_freeze_iter == observed_freeze_iter

    if observed_freeze_iter is None:
        assert all(names == managed_cleanup_names for names in active_history)
        assert all(not names for names in frozen_history)
    else:
        assert frozen_history[observed_freeze_iter - 1]
        assert active_history[observed_freeze_iter - 1] != managed_cleanup_names
