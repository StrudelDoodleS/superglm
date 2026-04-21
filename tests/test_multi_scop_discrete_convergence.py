import numpy as np
import pandas as pd
import pytest

import superglm.reml.scop_efs as scop_efs
from superglm import Categorical, Constraint, CubicRegressionSpline, PSpline, SuperGLM


def test_cleanup_enabled_only_for_multi_scop_discrete():
    assert scop_efs._multi_scop_discrete_cleanup_enabled(discrete=True, scop_term_count=2)
    assert not scop_efs._multi_scop_discrete_cleanup_enabled(discrete=True, scop_term_count=1)
    assert not scop_efs._multi_scop_discrete_cleanup_enabled(discrete=False, scop_term_count=2)


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
    active_names, frozen_names = scop_efs._freeze_multi_scop_discrete_lambdas(
        active_names=active_names,
        frozen_names=frozen_names,
        lambdas_new={"DrivAge": 0.118, "BonusMalus": 1.0e-4},
        stable_counts=stable_counts,
    )

    assert active_names == {"DrivAge"}
    assert frozen_names == {"BonusMalus"}
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
    assert (True, 2) in consulted_calls

    monkeypatch.setattr(
        scop_efs,
        "_multi_scop_discrete_cleanup_enabled",
        lambda *, discrete, scop_term_count: False,
    )
    baseline = _make_model()
    baseline.fit_reml(X, y, sample_weight=w, max_reml_iter=20)

    pred_opt = optimized.predict(X)
    pred_base = baseline.predict(X)
    np.testing.assert_allclose(pred_opt, pred_base, rtol=1e-4, atol=1e-6)
    assert set(optimized._reml_lambdas) == set(baseline._reml_lambdas)
