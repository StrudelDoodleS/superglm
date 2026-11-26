"""Measurements and named findings behind the scalar convergence report."""

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    ConvergenceWarning,
    FitPrecisionWarning,
    Numeric,
    Spline,
    SuperGLM,
)
from superglm.diagnostics.separation import SeparationWarning
from superglm.group_matrix import DenseGroupMatrix


@pytest.mark.parametrize("weight", [1.0, 1e5], ids=["relative_bar", "capped_bar"])
@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
def test_iteration_cap_reports_the_budget_and_last_test(discrete, weight):
    """The gradient bar is the one the stop compared: relative, capped at ``1e5 * tol``."""
    rng = np.random.default_rng(284)
    x = np.linspace(0, 1, 128)
    y = rng.poisson(np.exp(np.sin(6 * x)))
    model = SuperGLM(family="poisson", features={"x": Spline(k=6)}, discrete=discrete)
    with pytest.warns(ConvergenceWarning, match=r"max_reml_iter.*model\.diagnostics\("):
        model.fit_reml(
            pd.DataFrame({"x": x}),
            y,
            sample_weight=np.full(x.size, weight),
            max_reml_iter=1,
            reml_tol=1e-8,
        )
    report = model.diagnostics()["_model"]["convergence"]
    smoothing = report["smoothing"]
    assert smoothing["reason"] == "max_reml_iter"
    assert smoothing["iterations"] == smoothing["evidence"]["iteration_limit"] == 1
    evidence = smoothing["evidence"]
    assert evidence["test"] == "iteration_limit"
    relative = 1e-8 * (1 + abs(evidence["objective"]))
    assert (relative < 1e-3) == (weight == 1.0)
    assert evidence["gradient_bar"] == pytest.approx(min(relative, 1e-3))
    assert evidence["objective_change"] is None
    assert evidence["objective_roundoff"] is None
    assert report["coefficients"]["converged"]


def test_a_saved_result_without_stop_evidence_still_reports_the_reason():
    x = np.linspace(0, 1, 128)
    y = np.sin(6 * x) + x**2
    model = SuperGLM(family="gaussian", features={"x": Spline(k=6)})
    with pytest.warns(ConvergenceWarning, match="max_reml_iter"):
        model.fit_reml(pd.DataFrame({"x": x}), y, max_reml_iter=1)
    model._reml_result.__dict__.pop("stop_evidence", None)
    restored = pickle.loads(pickle.dumps(model))
    smoothing = restored.diagnostics()["_model"]["convergence"]["smoothing"]
    assert smoothing["reason"] == "max_reml_iter"
    assert smoothing["evidence"] is None


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize("active_set", [False, True], ids=["score", "frozen"])
def test_smoothing_report_names_the_actual_candidate_test(monkeypatch, discrete, active_set):
    from superglm.reml import direct
    from superglm.reml import discrete as binned

    engine = binned if discrete else direct
    freeze, evaluate = engine.freeze_flat_directions, engine.evaluate_reml_candidate

    def freeze_all(*args, **kwargs):
        decision = freeze(*args, **kwargs)
        return replace(
            decision,
            frozen=np.ones_like(decision.frozen),
            shoulder=np.zeros_like(decision.shoulder),
        )

    def force_test(**kwargs):
        candidate = evaluate(**kwargs)
        return replace(candidate, converged=False) if active_set else candidate

    monkeypatch.setattr(engine, "freeze_flat_directions", freeze_all)
    monkeypatch.setattr(engine, "evaluate_reml_candidate", force_test)
    x = np.linspace(0, 1, 80)
    model = SuperGLM(family="gaussian", features={"x": Spline(k=6)}, discrete=discrete)
    model.fit_reml(pd.DataFrame({"x": x}), np.sin(6 * x), reml_tol=0.1, max_reml_iter=2)
    smoothing = model.diagnostics()["_model"]["convergence"]["smoothing"]
    assert smoothing["iterations"] == 2
    evidence = smoothing["evidence"]
    assert evidence["objective_change"] <= evidence["objective_change_bar"]
    assert evidence["objective_roundoff"] > 0
    assert evidence["within_objective_roundoff"] == (
        evidence["objective_change"] <= evidence["objective_roundoff"]
    )
    if active_set:
        assert smoothing["reason"] == "active_set_stationary"
        assert evidence["test"] == "frozen_active_set"
        assert evidence["gradient_bar"] is None
    else:
        assert smoothing["reason"] == "score_objective_tolerance"
        assert evidence["test"] == "score_and_objective"
        assert evidence["gradient_norm"] < evidence["gradient_bar"]


@pytest.mark.parametrize("pair", [True, False], ids=["pair", "three_columns"])
def test_rank_dependencies_read_only_named_pair_columns(monkeypatch, pair):
    x = np.linspace(-1, 1, 64)
    z = x**2
    other = np.cos(9 * x)
    fitted = {"x": x, "z": z, "other": other, "dependent": (3 - 2 * x) if pair else x + z}
    frame = pd.DataFrame(fitted)
    model = SuperGLM(family="gaussian", features={name: Numeric() for name in fitted})
    model.fit(frame, x + np.sin(6 * x))
    # Diagnostics already reads EDF. Isolate the new column reads from that
    # existing computation and its cache.
    _ = model._group_edf

    reads = []
    original = DenseGroupMatrix.matvec

    def read(matrix, coefficients):
        reads.append(matrix)
        return original(matrix, coefficients)

    monkeypatch.setattr(DenseGroupMatrix, "matvec", read)
    monkeypatch.setattr(DenseGroupMatrix, "toarray", lambda *_: pytest.fail("dense group read"))
    convergence = model.diagnostics()["_model"]["convergence"]
    terms = ["x", "dependent"] if pair else ["x", "z", "dependent"]
    assert convergence["rank_dependencies"] == [terms]
    aliases = convergence["near_aliases"]
    if pair:
        assert aliases[0]["columns"] == ["x[0]", "dependent[0]"]
        assert aliases[0]["correlation"] == pytest.approx(-1.0, abs=8 * np.finfo(float).eps)
        assert reads == [model._dm.group_matrices[i] for i in (0, 3)]
    else:
        assert aliases == []
        assert reads == []


@pytest.mark.parametrize("order", ["given", "permuted"])
def test_a_light_first_row_does_not_set_the_alias_sign(order):
    """Sol on #461: centring on the first row rounded the heavy rows' variation away."""
    t = np.linspace(-1, 1, 65)
    frame = pd.DataFrame({"x": np.r_[1.0, 1e-18 * t], "z": np.r_[1.0, -1e-18 * t]})
    weights = np.r_[1e-80, np.ones(65)]
    y = np.r_[0.0, t + np.sin(6 * t)]
    if order == "permuted":
        rows = np.random.default_rng(461).permutation(66)
        frame, weights, y = frame.iloc[rows].reset_index(drop=True), weights[rows], y[rows]
    model = SuperGLM(family="gaussian", features={"x": Numeric(), "z": Numeric()})
    with pytest.warns(FitPrecisionWarning, match="sample-weight spread"):
        model.fit(frame, y, sample_weight=weights)
    (alias,) = model.diagnostics()["_model"]["convergence"]["near_aliases"]
    assert alias["columns"] == ["x[0]", "z[0]"]
    assert alias["correlation"] == pytest.approx(-1.0, abs=1e-12)


def test_alias_pairs_do_not_depend_on_the_null_basis():
    """Sol on #461: with nullity three, each basis vector mixed both duplicate pairs."""
    x = np.linspace(-1, 1, 64)
    z = x**2
    fitted = {"x": x, "z": z, "xcopy": x, "zcopy": z, "xboth": x + z}
    model = SuperGLM(family="gaussian", features={name: Numeric() for name in fitted})
    model.fit(pd.DataFrame(fitted), x + np.sin(6 * x))
    aliases = model.diagnostics()["_model"]["convergence"]["near_aliases"]
    assert sorted(alias["columns"] for alias in aliases) == [
        ["x[0]", "xcopy[0]"],
        ["z[0]", "zcopy[0]"],
    ]
    for alias in aliases:
        assert alias["correlation"] == pytest.approx(1.0, abs=1e-12)


def test_null_pairs_are_invariant_under_rotation_of_the_basis():
    from superglm.model.report_ops import _null_pairs
    from superglm.solvers.rank import SHARED_RANK_POLICY

    # null space of [x, z, xcopy, zcopy, xboth]
    basis = np.array(
        [[1.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0], [1.0, 1.0, 0.0, 0.0, -1.0]]
    ).T
    rotation = np.linalg.qr(np.random.default_rng(3).standard_normal((3, 3)))[0]
    rcond = SHARED_RANK_POLICY.factor_rcond
    assert sorted(_null_pairs(basis, rcond)) == [(0, 2), (1, 3)]
    assert sorted(_null_pairs(basis @ rotation, rcond)) == [(0, 2), (1, 3)]


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
def test_active_set_evidence_reads_the_terminal_hold_set(monkeypatch, discrete):
    """Sol on #461: a term held first at the stopping candidate kept its score in the report."""
    from superglm.reml import direct
    from superglm.reml import discrete as binned

    engine = binned if discrete else direct
    freeze, evaluate = engine.freeze_flat_directions, engine.evaluate_reml_candidate
    calls = []

    def freeze_late(*args, **kwargs):
        decision = freeze(*args, **kwargs)
        calls.append(None)
        held = np.ones_like if len(calls) > 1 else np.zeros_like
        return replace(
            decision, frozen=held(decision.frozen), shoulder=np.zeros_like(decision.shoulder)
        )

    monkeypatch.setattr(engine, "freeze_flat_directions", freeze_late)
    monkeypatch.setattr(
        engine, "evaluate_reml_candidate", lambda **k: replace(evaluate(**k), converged=False)
    )
    x = np.linspace(0, 1, 80)
    model = SuperGLM(family="gaussian", features={"x": Spline(k=6)}, discrete=discrete)
    model.fit_reml(pd.DataFrame({"x": x}), np.sin(6 * x), reml_tol=0.1, max_reml_iter=2)
    smoothing = model.diagnostics()["_model"]["convergence"]["smoothing"]
    assert smoothing["reason"] == "active_set_stationary"
    assert smoothing["evidence"]["gradient_norm"] == 0.0


def test_separation_report_names_the_recorded_level():
    frame = pd.DataFrame({"rating": ["mixed", "all zero", "other"] * 32})
    y = np.tile([0, 0, 1, 1, 0, 0], 16)
    model = SuperGLM(family="binomial", features={"rating": Categorical()})
    with pytest.warns(SeparationWarning, match="all zero"):
        model.fit(frame, y)
    separation = model.diagnostics()["_model"]["convergence"]["separation"]
    assert separation["levels"] == {"rating": ["all zero"]}
