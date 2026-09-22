"""Gamma REML reuses only fixed Fisher data, never a fitted weighted system."""

import weakref

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from superglm import Categorical, Spline, SuperGLM
from superglm._fit_trace import TraceRun
from superglm.distributions import Gamma, Gaussian, Poisson
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DesignMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import IdentityLink, LogLink
from superglm.reml import direct, observed_geometry
from superglm.solvers import centered_system, irls_direct
from superglm.solvers.irls_state import _IRLSStepDecision
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.types import GroupSlice, LinearConstraintSet


def test_mixed_gamma_optimizer_builds_one_coefficient_data_gram(monkeypatch):
    # Removing the optimizer handoff repeats a real weighted Gram for each
    # bootstrap/candidate/trial, even though exact Gamma/log W is invariant.
    x = np.linspace(0.03, 0.97, 48)
    cat = np.arange(len(x)) % 3 == 0
    frame = pd.DataFrame({"x": x, "cat": np.where(cat, "b", "a")})
    y = np.exp(0.2 + 0.3 * np.sin(2 * np.pi * x) + 0.12 * cat)
    y *= 1.0 + 0.1 * np.cos(8 * np.pi * x)
    phases, grams = [], []
    active = []
    owners = []
    original_fit = direct.fit_irls_direct
    original_gram = centered_system.centered_gram_rhs
    original_observed = observed_geometry.build_centered_system
    original_system = irls_direct.build_centered_system
    observed_builds, outside_builds = [], []

    def fit(**kwargs):
        assert {type(gm) for gm in kwargs["X"].group_matrices} == {
            SparseSSPGroupMatrix,
            CategoricalGroupMatrix,
        }
        phase = kwargs["debug_context"]["phase"]
        if kwargs.get("_fisher_data_reuse") is not None:
            owners.append(weakref.ref(kwargs["_fisher_data_reuse"]))
        phases.append(phase)
        active.append(phase)
        try:
            return original_fit(**kwargs)
        finally:
            active.pop()

    def gram(**kwargs):
        value = original_gram(**kwargs)
        if active:
            grams.append(active[-1])
        return value

    def observed(**kwargs):
        observed_builds.append(kwargs.get("_data"))
        return original_observed(**kwargs)

    def system(**kwargs):
        if not active:
            outside_builds.append(kwargs.get("_data"))
        return original_system(**kwargs)

    monkeypatch.setattr(direct, "fit_irls_direct", fit)
    monkeypatch.setattr(centered_system, "centered_gram_rhs", gram)
    monkeypatch.setattr(observed_geometry, "build_centered_system", observed)
    monkeypatch.setattr(irls_direct, "build_centered_system", system)
    model = SuperGLM(
        family="gamma",
        selection_penalty=0.0,
        discrete=False,
        direct_solve="gram",
        features={"x": Spline(n_knots=5, penalty="ssp"), "cat": Categorical()},
    )
    model.fit_reml(frame, y, max_reml_iter=3)
    assert {"bootstrap", "candidate", "line_search"} <= set(phases)
    assert len(grams) == 1, (phases, grams)
    assert owners and all(owner() is None for owner in owners)
    assert len(observed_builds) >= 2 and all(data is None for data in observed_builds)
    assert outside_builds and all(data is None for data in outside_builds)


def _problem():
    x = np.tile(np.array([-1.0, 0.0, 1.0]), 2)
    categorical = np.array([0, 0, 0, -1, -1, -1])
    dm = DesignMatrix(
        [
            SparseSSPGroupMatrix(sparse.csr_matrix(x[:, None]), np.ones((1, 1))),
            CategoricalGroupMatrix(categorical, n_levels=1),
        ],
        n=6,
        p=2,
    )
    groups = [GroupSlice("x", 0, 1), GroupSlice("cat", 1, 2)]
    dense = np.column_stack((x, categorical == 0))
    return dm, groups, Gamma(), LogLink(), dense


def _fit(problem, entry, *, W=None, y=None, penalty=1.0, **kwargs):
    dm, groups, family, link, _ = problem
    return irls_direct.fit_irls_direct(
        dm,
        np.ones(dm.n) if y is None else y,
        np.ones(dm.n) if W is None else W,
        family,
        link,
        groups,
        lambda2=penalty,
        S_override=kwargs.pop("S_override", penalty * np.eye(dm.p)),
        beta_init=kwargs.pop("beta_init", np.zeros(dm.p)),
        intercept_init=0.0,
        direct_solve=kwargs.pop("direct_solve", "gram"),
        weight_semantics=kwargs.pop("weight_semantics", "frequency"),
        _fisher_data_reuse=entry,
        **kwargs,
    )


def _assert_system(system, X, W, z, S):
    # A dense, independent oracle; the bound accounts for O(n) reductions and
    # two centering operations, scaled by operand norms rather than roundoff sign.
    xc = X - np.sum(W[:, None] * X, axis=0) / W.sum()
    zc = z - np.dot(W, z) / W.sum()
    gram = xc.T @ (W[:, None] * xc)
    rhs = xc.T @ (W * zc)
    scale = max(
        1.0,
        np.linalg.norm(xc) ** 2 * np.max(W),
        np.linalg.norm(xc) * np.linalg.norm(W * zc),
        np.linalg.norm(S),
    )
    bound = 64 * len(W) * np.finfo(float).eps * scale
    for actual, expected in (
        (system.data_gram, gram),
        (system.rhs, rhs),
        (system.hessian, gram + S),
    ):
        assert np.linalg.norm(actual - expected) <= bound


def test_hits_refresh_rhs_penalty_and_changed_weights_rebuild(monkeypatch):
    problem = _problem()
    entry = centered_system._FisherDataReuse()
    original = irls_direct.build_centered_system
    hits = []

    def checked(**kwargs):
        system = original(**kwargs)
        hits.append(kwargs.get("_data") is not None)
        _assert_system(system, problem[-1], kwargs["W"], kwargs["z_off"], kwargs["penalty"])
        return system

    monkeypatch.setattr(irls_direct, "build_centered_system", checked)
    W = np.ones(6)
    for i, penalty in enumerate((1.0, 3.0, 2.0)):
        if i == 2:
            W[:] = np.arange(1.0, 7.0)  # In-place change defeats an identity-only key.
        result, inverse = _fit(
            problem,
            entry,
            W=W,
            penalty=penalty,
            beta_init=np.array([0.2, -0.1]) if i == 1 else np.zeros(2),
        )
        assert result.converged
        xc = problem[-1] - np.sum(W[:, None] * problem[-1], axis=0) / W.sum()
        hessian = xc.T @ (W[:, None] * xc) + penalty * np.eye(2)
        bound = 32 * len(W) * np.finfo(float).eps * np.linalg.cond(hessian)
        assert np.linalg.norm(hessian @ inverse - np.eye(2)) <= bound
        if i != 1:
            # y=mu=1 gives an exactly zero working RHS at the known optimum.
            np.testing.assert_array_equal(result.beta, np.zeros(2))
    assert hits == [False, True, False]
    assert entry.weights.nbytes == W.nbytes
    assert not np.shares_memory(entry.weights, W)
    assert not entry.weights.flags.writeable
    assert len(entry.data) == 3 and entry.data[1].shape == (2,) and entry.data[2].shape == (2, 2)


@pytest.mark.parametrize("location", [0.0, 2.0**20])
def test_data_hit_uses_both_rhs_authorities_and_current_validation(location):
    problem = _problem()
    dm = problem[0]
    dm.group_matrices[0].B = sparse.csr_matrix(dm.group_matrices[0].B.toarray() + location)
    X = dm.toarray()
    W = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0])
    z = np.array([3.0, -1.0, 2.0, -2.0, 1.0, 0.0])
    first = centered_system.build_centered_system(dm=dm, W=W, z_off=z, penalty=np.eye(2))
    data = first.sum_w, first.mean_x, first.data_gram
    changed = centered_system.build_centered_system(
        dm=dm, W=W, z_off=-z, penalty=3 * np.eye(2), _data=data
    )
    _assert_system(changed, X, W, -z, 3 * np.eye(2))
    with pytest.raises(ValueError, match="row count"):
        centered_system.build_centered_system(
            dm=dm, W=W, z_off=z[:-1], penalty=np.eye(2), _data=data
        )
    with pytest.raises(ValueError, match="penalty must have shape"):
        centered_system.build_centered_system(
            dm=dm, W=W, z_off=z, penalty=np.ones((1, 1)), _data=data
        )


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "semantics",
        "custom_family",
        "custom_link",
        "custom_group",
        "trace",
        "debug",
        "constraint",
        "qr",
        "structured",
    ],
)
def test_changed_or_excluded_contract_cannot_consume_saved_data(monkeypatch, change):
    problem = _problem()
    entry = centered_system._FisherDataReuse()
    _fit(problem, entry)
    original = irls_direct.build_centered_system
    hits = []

    def checked(**kwargs):
        hits.append(kwargs.get("_data") is not None)
        return original(**kwargs)

    monkeypatch.setattr(irls_direct, "build_centered_system", checked)
    dm, groups, family, link, dense = problem
    kwargs = {}
    if change == "owner":
        dm = _problem()[0]
    elif change == "semantics":
        kwargs["weight_semantics"] = "prior"
    elif change == "custom_family":

        class CustomGamma(Gamma):
            pass

        family = CustomGamma()
    elif change == "custom_link":

        class CustomLog(LogLink):
            pass

        link = CustomLog()
    elif change == "custom_group":

        class CustomSSP(SparseSSPGroupMatrix):
            pass

        dm = DesignMatrix(
            [CustomSSP(dm.group_matrices[0].B, np.ones((1, 1))), dm.group_matrices[1]], n=6, p=2
        )
    elif change == "trace":
        kwargs["trace_run"] = TraceRun("fresh")
    elif change == "debug":
        kwargs["debug_recorder"] = object()
    elif change == "constraint":
        groups[0].constraints = LinearConstraintSet(np.ones((1, 1)), np.zeros(1))
    elif change == "structured":
        dm = DesignMatrix(
            [dm.group_matrices[0], RandomEffectGroupMatrix(np.zeros(6, dtype=int), 1)], n=6, p=2
        )
        kwargs["direct_solve"] = "structured"
    else:
        kwargs["direct_solve"] = change
    result, _ = _fit((dm, groups, family, link, dense), entry, **kwargs)
    assert result.converged
    assert not any(hits)
    if change not in {"owner", "semantics"}:
        assert entry.data is None and entry.weights is None


@pytest.mark.parametrize("bad", ["weights", "penalty", "iteration_budget", "unfinished"])
def test_exception_or_nonconvergence_releases_payload(bad):
    problem = _problem()
    entry = centered_system._FisherDataReuse()
    _fit(problem, entry)
    if bad == "unfinished":
        result, _ = _fit(problem, entry, y=np.arange(1.0, 7.0), max_iter=1)
        assert not result.converged
    elif bad == "iteration_budget":
        with pytest.raises(ValueError, match="max_iter"):
            _fit(problem, entry, max_iter=0)
    else:
        with pytest.raises(ValueError):
            _fit(
                problem,
                entry,
                **(
                    {"W": np.full(6, np.nan)}
                    if bad == "weights"
                    else {"S_override": np.ones((1, 1))}
                ),
            )
    assert entry.data is None and entry.weights is None and entry.owners == ()


def test_observed_rescue_and_fisher_return_require_fresh_systems(monkeypatch):
    problem = _problem()
    entry = centered_system._FisherDataReuse()
    _fit(problem, entry)
    rows, builds, remembered = [], [], []
    original_rows = irls_direct.coefficient_working_rows
    original_build = irls_direct.build_centered_system
    original_remember = centered_system._FisherDataReuse.remember
    original_select = irls_direct._select_irls_trial
    selections = 0

    def working(**kwargs):
        result = original_rows(**kwargs)
        rows.append(result.curvature_source)
        return result

    def build(**kwargs):
        builds.append((rows[-1], kwargs.get("_data") is not None))
        return original_build(**kwargs)

    def remember(self, *args):
        remembered.append(rows[-1])
        return original_remember(self, *args)

    def reject_twice(*args, **kwargs):
        nonlocal selections
        selections += 1
        if selections <= 2:
            return _IRLSStepDecision(0.0, 0, True, trials_attempted=21)
        return original_select(*args, **kwargs)

    monkeypatch.setattr(irls_direct, "coefficient_working_rows", working)
    monkeypatch.setattr(irls_direct, "build_centered_system", build)
    monkeypatch.setattr(centered_system._FisherDataReuse, "remember", remember)
    monkeypatch.setattr(irls_direct, "_select_irls_trial", reject_twice)
    result, _ = _fit(problem, entry, y=np.array([0.8, 1.2, 1.1, 1.3, 0.9, 1.5]))
    assert result.converged
    assert rows[:3] == ["fisher", "observed", "fisher"]
    assert builds[:3] == [("fisher", True), ("observed", False), ("fisher", False)]
    assert remembered == ["fisher"]


def test_scop_and_non_gamma_do_not_populate_the_payload():
    problem = _problem()
    entry = centered_system._FisherDataReuse()
    _fit(problem, entry)
    dm, groups, _, link, dense = problem
    _fit((dm, groups, Poisson(), link, dense), entry)
    assert entry.data is None and entry.weights is None

    _fit(problem, entry)
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    irls_direct.fit_irls_direct(
        np.array([[-1.0], [1.0]]),
        np.array([-1.0, 1.0]),
        np.ones(2),
        Gaussian(),
        IdentityLink(),
        [GroupSlice("shape", 0, 1, monotone_engine="scop", scop_reparameterization=reparam)],
        lambda2=0.0,
        beta_init=np.ones(1),
        intercept_init=0.0,
        max_iter=1,
        S_override=np.zeros((1, 1)),
        _scop_joint=False,
        scop_state_init={0: {"beta_eff": np.zeros(1), "S_scop": np.zeros((1, 1))}},
        _compute_scop_postfit_inference=False,
        weight_semantics="frequency",
        _fisher_data_reuse=entry,
    )
    assert entry.data is None and entry.weights is None


def test_gram_certificate_refusal_clears_payload(monkeypatch):
    # Exercise the real factor-based fallback after a Gram certificate refuses.
    # A validated solve must not republish that refused route's cached data.
    problem = _problem()
    entry = centered_system._FisherDataReuse()
    _fit(problem, entry)
    monkeypatch.setattr(irls_direct, "decompose_gram_if_authoritative", lambda *args: None)
    result, inverse = _fit(problem, entry)
    assert result.converged
    np.testing.assert_allclose(inverse, np.diag([1 / 5, 1 / 2.5]), rtol=32 * np.finfo(float).eps)
    assert entry.data is None and entry.weights is None
