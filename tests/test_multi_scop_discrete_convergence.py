from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import superglm.reml.scop_efs as scop_efs
from superglm import Categorical, Constraint, CubicRegressionSpline, PSpline, SuperGLM
from superglm.solvers.pirls import PIRLSResult


def test_cleanup_enabled_only_for_multi_scop_discrete():
    assert scop_efs._multi_scop_discrete_cleanup_enabled(discrete=True, scop_term_count=2)
    assert not scop_efs._multi_scop_discrete_cleanup_enabled(discrete=True, scop_term_count=1)
    assert not scop_efs._multi_scop_discrete_cleanup_enabled(discrete=False, scop_term_count=2)


def test_cleanup_names_include_only_discrete_scop_estimated_names():
    cleanup_names = scop_efs._multi_scop_discrete_cleanup_names(
        estimated_names={"DrivAge", "BonusMalus", "VehAge"},
        scop_states={
            0: {"group_name": "DrivAge", "bin_idx": np.array([0, 1], dtype=np.intp)},
            1: {"group_name": "BonusMalus", "bin_idx": np.array([1, 0], dtype=np.intp)},
        },
        scop_term_count=2,
    )

    assert cleanup_names == {"DrivAge", "BonusMalus"}


def test_cleanup_names_disable_mixed_bootstrap_scop_states():
    cleanup_names = scop_efs._multi_scop_discrete_cleanup_names(
        estimated_names={"DrivAge", "BonusMalus", "VehAge"},
        scop_states={
            0: {"group_name": "DrivAge", "bin_idx": np.array([0, 1], dtype=np.intp)},
            1: {"group_name": "BonusMalus", "bin_idx": None},
        },
        scop_term_count=2,
    )

    assert cleanup_names == set()


def test_cleanup_names_require_multiple_eligible_estimated_discrete_scop_terms():
    cleanup_names = scop_efs._multi_scop_discrete_cleanup_names(
        estimated_names={"DrivAge", "VehAge"},
        scop_states={
            0: {"group_name": "DrivAge", "bin_idx": np.array([0, 1], dtype=np.intp)},
            1: {"group_name": "BonusMalus", "bin_idx": np.array([1, 0], dtype=np.intp)},
        },
        scop_term_count=2,
    )

    assert cleanup_names == set()


def test_first_near_floor_iteration_does_not_immediately_freeze():
    stable_counts = {"BonusMalus": 2}
    active_names = {"BonusMalus"}
    frozen_names = set()

    stable_counts = scop_efs._update_multi_scop_discrete_stability_counts(
        lambdas_old={"BonusMalus": 1.2e-4},
        lambdas_new={"BonusMalus": 1.0e-4},
        active_names=active_names,
        stable_counts=stable_counts,
    )
    assert stable_counts["BonusMalus"] == 1

    active_names, frozen_names = scop_efs._freeze_multi_scop_discrete_lambdas(
        active_names=active_names,
        frozen_names=frozen_names,
        lambdas_new={"BonusMalus": 1.0e-4},
        stable_counts=stable_counts,
    )

    assert active_names == {"BonusMalus"}
    assert frozen_names == set()


def test_floor_pinned_lambda_freezes_after_stability_window():
    stable_counts = {"DrivAge": 0, "BonusMalus": 2}
    active_names = {"DrivAge", "BonusMalus"}
    frozen_names = set()

    stable_counts = scop_efs._update_multi_scop_discrete_stability_counts(
        lambdas_old={"DrivAge": 0.12, "BonusMalus": 1.0e-4},
        lambdas_new={"DrivAge": 0.118, "BonusMalus": 1.0e-4},
        active_names=active_names,
        stable_counts=stable_counts,
    )
    assert stable_counts["BonusMalus"] == 3
    active_names, frozen_names = scop_efs._freeze_multi_scop_discrete_lambdas(
        active_names=active_names,
        frozen_names=frozen_names,
        lambdas_new={"DrivAge": 0.118, "BonusMalus": 1.0e-4},
        stable_counts=stable_counts,
    )

    assert active_names == {"DrivAge"}
    assert frozen_names == {"BonusMalus"}


def test_empty_cleanup_path_uses_legacy_plateau_convergence(monkeypatch):
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
            {"term": 1.0},
            {"term": float(np.exp(0.005))},
            {"term": float(np.exp(0.010))},
            {"term": float(np.exp(0.015))},
        ]
    )
    penalties = [SimpleNamespace(name="term")]

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
    monkeypatch.setattr(
        scop_efs,
        "_joint_efs_lambda_step",
        lambda *args, **kwargs: (next(lambda_updates), {}, {}),
    )
    monkeypatch.setattr(scop_efs, "_multi_scop_discrete_cleanup_names", lambda **kwargs: set())

    def fail_if_helper_used(**kwargs):
        raise AssertionError("unexpected helper plateau check")

    monkeypatch.setattr(
        scop_efs,
        "_multi_scop_discrete_plateau_converged",
        fail_if_helper_used,
    )

    result = scop_efs.optimize_scop_efs_reml(
        dm=SimpleNamespace(group_matrices=[], p=1),
        distribution=SimpleNamespace(scale_known=True),
        link=SimpleNamespace(),
        groups=[SimpleNamespace(monotone_engine="scop")],
        y=np.array([0.0]),
        sample_weight=np.ones(1),
        offset_arr=np.zeros(1),
        lambdas={"term": 2.0},
        estimated_names={"term"},
        max_reml_iter=5,
        reml_penalties=penalties,
    )

    assert result.converged
    assert result.n_reml_iter == 3


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


def _make_model() -> SuperGLM:
    return SuperGLM(
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


@pytest.mark.slow
def test_multi_scop_discrete_cleanup_preserves_predictions(monkeypatch):
    X, y, w = _make_multi_scop_data()
    consulted_calls: list[tuple[bool, int]] = []
    original = scop_efs._multi_scop_discrete_cleanup_enabled

    def record_cleanup_gate(*, discrete, scop_term_count):
        consulted_calls.append((discrete, scop_term_count))
        return original(discrete=discrete, scop_term_count=scop_term_count)

    monkeypatch.setattr(
        scop_efs,
        "_multi_scop_discrete_cleanup_enabled",
        record_cleanup_gate,
    )
    optimized = _make_model()
    optimized.fit_reml(X, y, sample_weight=w, max_reml_iter=20)
    assert optimized._reml_result.converged
    assert (True, 2) in consulted_calls

    monkeypatch.setattr(
        scop_efs,
        "_multi_scop_discrete_cleanup_enabled",
        lambda *, discrete, scop_term_count: False,
    )
    baseline = _make_model()
    baseline.fit_reml(X, y, sample_weight=w, max_reml_iter=20)
    assert baseline._reml_result.converged

    pred_opt = optimized.predict(X)
    pred_base = baseline.predict(X)
    np.testing.assert_allclose(pred_opt, pred_base, rtol=1e-4, atol=1e-6)
    assert set(optimized._reml_lambdas) == set(baseline._reml_lambdas)
